#!/usr/bin/env python3
"""spec-short-prompt-check.py — the short-prompt path under speculative decode.

Upstream vLLM tracked a prefill that is misread as a uniform-decode batch:
when the number of scheduled tokens for a request happens to equal
`uniform_decode_query_len * num_reqs`, the prefill can be dispatched into
the CUDA graph captured for speculative decode. For hybrid models with
recurrent-state layers the state write is then skipped and the output is
deterministic garbage (upstream issue #53051). The trigger is a prompt of
exactly `uniform_decode_query_len` tokens — for this recipe's speed profile
(k = 5 draft tokens) that is 6 tokens, so a 6-token prompt is enough.

This script probes exactly that shape and asks whether the answer is sane:

  1. it resolves `uniform_decode_query_len` — from the running engine's own
     command line when a container is named, from the speculative config you
     pass, or from an explicit --query-len;
  2. it builds a prompt of EXACTLY that many tokens (the server's own
     /tokenize endpoint is the counter, so the count is the engine's);
  3. it sends it at temperature 0 and checks the completion: non-empty,
     printable, not degenerate, and — when the prompt carries a marker —
     echoing the marker back.

It is a READ-ONLY probe: one request to a serving engine. Run it in a boot
window against a lane you are accepting, not against one carrying traffic.

Usage:
    python3 launch/tp3-kit/spec-short-prompt-check.py \\
        --base-url http://HEAD_IP:8889 --model ds4flash

    # measure the trap length from the running engine itself:
    python3 launch/tp3-kit/spec-short-prompt-check.py \\
        --base-url http://HEAD_IP:8889 --model ds4flash \\
        --container ds4flash-tp3-head

Exit codes: 0 = the short-prompt path answered sanely, 1 = it did not,
2 = the probe could not be set up (that is a setup error, not a verdict).
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import urllib.error
import urllib.request

MARKER = "ALPHA"

# Candidate prompts, each carrying the marker the answer must echo back.
CANDIDATES = (
    "Please repeat exactly: {m}",
    "Say {m}",
    "Echo the word {m} back to me.",
    "Repeat this word: {m}",
    "Please output the word {m} and nothing else.",
)

# Words appended to reach the exact token count. Each is short on purpose.
FILLERS = (
    " now", " please", " exactly", " again", " once", " ok", " thanks",
    " right", " only", " truly", " just",
)

# Fallback: a marker-free prompt of an exact length (single-token words).
FALLBACK_WORDS = ("a", "the", "is", "it", "of", "to")


def post(url: str, payload: dict, timeout: int) -> dict:
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def get(url: str, timeout: int) -> dict:
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def token_count(base_url: str, prompt: str, model: str, timeout: int) -> int | None:
    """The engine's own token count for a prompt (the /tokenize endpoint)."""
    try:
        out = post(f"{base_url}/tokenize", {"model": model, "prompt": prompt}, timeout)
    except (urllib.error.URLError, OSError, ValueError):
        return None
    count = out.get("count")
    if isinstance(count, int):
        return count
    tokens = out.get("tokens")
    if isinstance(tokens, list):
        return len(tokens)
    return None


def build_exact_prompt(base_url: str, base: str, model: str, target: int,
                        timeout: int) -> str | None:
    """A prompt whose token count is exactly `target`."""
    n = token_count(base_url, base, model, timeout)
    if n is None:
        return None
    if n == target:
        return base
    if n > target:
        return None
    current = base
    for filler in FILLERS:
        candidate = current + filler
        n = token_count(base_url, candidate, model, timeout)
        if n is None:
            return None
        if n == target:
            return candidate
        if n > target:
            continue  # this filler overshoots — try the next one
        current = candidate
    return None


def build_fallback_prompt(base_url: str, model: str, target: int,
                          timeout: int) -> str | None:
    """A marker-free prompt of exactly `target` tokens (no echo expectation)."""
    for word in FALLBACK_WORDS:
        lo, hi = 1, max(target * 4, 8)
        best = None
        while lo <= hi:
            mid = (lo + hi) // 2
            candidate = " ".join([word] * mid)
            n = token_count(base_url, candidate, model, timeout)
            if n is None:
                return None
            if n == target:
                best = candidate
                break
            if n < target:
                lo = mid + 1
            else:
                hi = mid - 1
        if best is not None:
            return best
    return None


def engine_query_len_from_container(container: str) -> tuple[int | None, str, str]:
    """(uniform_decode_query_len, source, runner) from the running engine.

    Reads the engine process's own command line inside the container and
    applies the engine's own rule (a decode step submits one query for the
    sampled token plus one per draft token, so the uniform decode query
    length is 1 + num_speculative_tokens). No import of the engine package:
    the lane already owns the GPU.
    """
    snippet = (
        "import json,sys\n"
        "argv=open('/proc/1/cmdline','rb').read().decode().split('\\x00')\n"
        "spec=None\n"
        "for i,a in enumerate(argv):\n"
        "    if a=='--speculative-config' and i+1<len(argv): spec=argv[i+1]\n"
        "    elif a.startswith('--speculative-config='): spec=a.split('=',1)[1]\n"
        "env=open('/proc/1/environ','rb').read().decode(errors='replace')\n"
        "runner=''\n"
        "for part in env.split('\\x00'):\n"
        "    if part.startswith('VLLM_USE_V2_MODEL_RUNNER='):\n"
        "        runner=part.split('=',1)[1]\n"
        "print(json.dumps({'spec':spec,'runner':runner,'argv':argv[:4]}))\n"
    )
    try:
        res = subprocess.run(
            ["docker", "exec", "-i", container, "python3", "-"],
            input=snippet, capture_output=True, text=True, timeout=120,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return None, f"unavailable ({exc.__class__.__name__})", ""
    if res.returncode != 0:
        return None, f"unavailable (probe failed in {container})", ""
    try:
        data = json.loads(res.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        return None, "unavailable (unreadable probe output)", ""

    runner = data.get("runner") or ""
    runner_note = " (engine runner: V2)" if runner == "1" else (
        " (engine runner: V1)" if runner == "0" else ""
    )
    raw = data.get("spec")
    if not raw:
        return None, "the engine runs WITHOUT speculative decode" + runner_note, runner
    try:
        cfg = json.loads(raw)
    except ValueError:
        return None, "unreadable --speculative-config on the engine" + runner_note, runner
    k = cfg.get("num_speculative_tokens")
    if not isinstance(k, int):
        return None, "the engine's speculative config has no num_speculative_tokens" + runner_note, runner
    return 1 + k, "the running engine's own command line" + runner_note, runner


def looks_degenerate(text: str) -> str | None:
    """Return a reason string when the completion is not a sane answer."""
    stripped = text.strip()
    if not stripped:
        return "the completion is empty"
    if any(ord(c) < 32 and c not in "\n\t\r" for c in stripped):
        return "the completion contains control characters"
    if any("　" <= c <= "鿿" or "＀" <= c <= "￯" for c in stripped):
        return "the completion contains CJK characters"
    if "�" in stripped:
        return "the completion contains a replacement character"
    words = stripped.split()
    if len(words) >= 12:
        uniq = len(set(words))
        if uniq <= max(2, len(words) // 12):
            return "the completion repeats the same few words"
    if len(stripped) >= 24:
        for size in (4, 6, 8):
            block = stripped[:size]
            if block and stripped.count(block) >= max(3, len(stripped) // (size * 2)):
                return f"the completion repeats a {size}-character block"
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--base-url", required=True,
                    help="the engine's OpenAI-compatible base URL")
    ap.add_argument("--model", default="ds4flash")
    ap.add_argument("--query-len", type=int, default=None,
                    help="the trap length to probe (skips the measurement)")
    ap.add_argument("--container", default=None,
                    help="a container running the engine, on THIS host")
    ap.add_argument("--spec-config", default=None,
                    help="the engine's --speculative-config JSON, if known")
    ap.add_argument("--max-tokens", type=int, default=32)
    ap.add_argument("--timeout", type=int, default=120)
    args = ap.parse_args()

    base = args.base_url.rstrip("/")

    # ---- 0. the engine must be up
    try:
        models = get(f"{base}/v1/models", args.timeout)
    except (urllib.error.URLError, OSError, ValueError) as exc:
        print(f"cannot reach {base}/v1/models ({exc.__class__.__name__}) — "
              "is the engine serving?", file=sys.stderr)
        return 2
    names = [m.get("id") for m in models.get("data", [])]
    print(f"engine up: {', '.join(str(n) for n in names) or 'no model listed'}")

    # ---- 1. the trap length
    source = ""
    if args.query_len is not None:
        query_len = args.query_len
        source = "--query-len (given)"
    elif args.container:
        query_len, source, _runner = engine_query_len_from_container(args.container)
    elif args.spec_config:
        try:
            cfg = json.loads(args.spec_config)
        except ValueError:
            print("--spec-config is not valid JSON", file=sys.stderr)
            return 2
        k = cfg.get("num_speculative_tokens")
        if not isinstance(k, int):
            print("--spec-config has no num_speculative_tokens", file=sys.stderr)
            return 2
        query_len = 1 + k
        source = "the speculative config you passed"
    else:
        query_len = None
        source = ""

    if query_len is None:
        print("could not resolve the trap length:", file=sys.stderr)
        print("  pass --query-len N, or --container <container> (the engine's own "
              "command line), or --spec-config '<json>'", file=sys.stderr)
        return 2
    print(f"uniform_decode_query_len = {query_len}  (source: {source})")

    # ---- 2. a prompt of exactly that many tokens
    prompt = None
    echo_expected = False
    for template in CANDIDATES:
        candidate = build_exact_prompt(base, template.format(m=MARKER), args.model,
                                      query_len, args.timeout)
        if candidate is not None:
            prompt = candidate
            echo_expected = True
            break
    if prompt is None:
        prompt = build_fallback_prompt(base, args.model, query_len, args.timeout)
        echo_expected = False
    if prompt is None:
        print(f"could not build a prompt of exactly {query_len} tokens — "
              "the /tokenize endpoint did not answer", file=sys.stderr)
        return 2
    measured = token_count(base, prompt, args.model, args.timeout)
    print(f"probe prompt: {measured} token(s) (target {query_len})")
    if measured != query_len:
        print("the probe prompt is not the intended length — aborting",
              file=sys.stderr)
        return 2

    # ---- 3. ask, at temperature 0
    try:
        out = post(f"{base}/v1/completions", {
            "model": args.model,
            "prompt": prompt,
            "max_tokens": args.max_tokens,
            "temperature": 0,
        }, args.timeout)
    except urllib.error.HTTPError as exc:
        print(f"the engine rejected the probe (HTTP {exc.code})", file=sys.stderr)
        return 1
    except (urllib.error.URLError, OSError, ValueError) as exc:
        print(f"the probe request failed ({exc.__class__.__name__})",
              file=sys.stderr)
        return 2

    choices = out.get("choices") or []
    if not choices:
        print("the engine returned no choice", file=sys.stderr)
        return 1
    text = choices[0].get("text") or ""
    usage = out.get("usage") or {}
    reported = usage.get("prompt_tokens")

    print(f"engine-reported prompt tokens: {reported}")
    print(f"completion: {text.strip()[:200]!r}")

    problems = []
    if isinstance(reported, int) and reported != query_len:
        problems.append(
            f"the engine scheduled {reported} prompt tokens, not {query_len} — "
            "the probe did not hit the intended shape"
        )
    reason = looks_degenerate(text)
    if reason:
        problems.append(reason)
    if echo_expected and MARKER.lower() not in text.lower():
        problems.append(
            f"the completion does not echo the marker {MARKER} — a prefill that "
            "was misread as a decode batch answers with garbage"
        )
    if not echo_expected:
        print("note: no marker prompt of this length could be built — the "
              "completeness check is structural only")

    if problems:
        print("\nSPEC-SHORT-PROMPT FAIL")
        for p in problems:
            print(f"  - {p}")
        return 1
    print("\nSPEC-SHORT-PROMPT PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
