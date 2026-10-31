# VALIDATION — ds4flash-tp3

The acceptance gates, and the measurement method behind every number in
this repository's README and FIXLOG. The bring-up crash ladders live in
`FIXLOG.md`; this file is the method and the commands.

Run the gates after every boot, in the order below. A lane that passes a
single probe is not validated — the gate is a mixed battery, and the
last section says what that means.

The placeholders in the commands are the ones the profile env files use:
`HEAD_IP` (the head's management-LAN address), `API_PORT` (the served
port) and `SERVED_NAME` (the model name clients send). Set them from
your staged env file:

```sh
# on the head box, from the staged env:
set -a; . ~/ds4flash-tp3/.env.tp3-ds-karmic; set +a   # speed profile
#   (.env.tp3-ds for the capacity profile)
export HEAD_IP API_PORT SERVED_NAME
```

---

## 1. Health and the served model

```sh
curl -s -o /dev/null -w "%{http_code}\n" http://$HEAD_IP:$API_PORT/health
# expected: 200

curl -s http://$HEAD_IP:$API_PORT/v1/models | jq -r '.data[].id'
# expected: the served names (SERVED_NAME and its alias)
```

## 2. The per-profile boot-log gates

These are the lines the boot must print; they are the profile's own
proof that the TP=3 padding and the executor landed.

```sh
# speed profile: the two padding lines, the drafter, the NCCL banner
docker logs ds4flash-tp3-head 2>&1 | grep -E \
  "Padded DeepSeek V4.1 attention for TP3|Padded DeepSeek V4 MoE intermediate"
docker logs ds4flash-tp3-head 2>&1 | grep -iE "dspark.*(enabled|spec)" | head -3
docker logs ds4flash-tp3-head 2>&1 | grep -iE "NCCL version" | head -1
# expected: both padding lines present; the drafter enabled; one NCCL banner

# capacity profile: the virtual-TP pad lines and the NCCL banner
docker logs ds4flash-tp3-head 2>&1 | grep -iE "NCCL version" | head -1
# expected: the banner names the image's NCCL version, never the baked shim
```

### Boot hygiene

Workers first, head last. Drop the filesystem caches on every rank
before the boot (the driver OOMs during load otherwise), re-apply the
fabric MTU after a reboot, and never `docker restart` — a fresh
`docker run` every time.

```sh
# on every rank, before the boot
sync; echo 3 > /proc/sys/vm/drop_caches
ip link set enp1s0f0np0 mtu 9000
ip link set enp1s0f1np1 mtu 9000
ip -br link | grep -E 'enp1s0f[01]np[01]'
# expected: both CX-7 links UP at mtu 9000
```

Poll `/health` with a bound generous for the stack: a speculative build
spends 12-15+ minutes autotuning and priming thousands of kernels before
first light, and declaring a hang at minute 8 throws away a good boot.

## 3. The engine-identity gate — which engine answered

The gate runs inside the container before the engine starts, and it is
re-runnable at any time:

```sh
docker exec ds4flash-tp3-head python3 /opt/tp3-kit/boot-gate-identity.py --profile speed
# expected: ENGINE-IDENTITY PASS

# with the manifest filled in, --strict also fails on an unpinned claim
docker exec ds4flash-tp3-head python3 /opt/tp3-kit/boot-gate-identity.py --profile speed --strict
```

The shipped manifest is a template, so the default run reports
`NOT PINNED` per claim and still passes. Filling it in (see the
README's "Proving which engine answered") turns this into the adoption
gate: a re-pinned image fails it until it is re-audited.

## 4. The container command, without starting it

The printed command is the command the box would run — it is built in
one place and executed from the same array.

```sh
PRINT_CMD=1 ~/.local/libexec/ds4flash-tp3/ds4flash-tp3-karmic-spark start   # speed profile
PRINT_CMD=1 ~/.local/libexec/ds4flash-tp3/ds4flash-tp3-spark start          # capacity profile
# expected: the argv, one argument per line, then a copy-pasteable line.
# Nothing is started: no container, no GPU.
```

Run it on the box being started, with that box's staged env — the role
derivation (`hostname -s` against `HEAD_HOST`/`W1_HOST`/`W2_HOST`) is
what fills the rank and the node address.

## 5. Deterministic generation (temp 0) — short prompts

Send the same prompt three times; the outputs must be identical.

```sh
for i in 1 2 3; do
  curl -s http://$HEAD_IP:$API_PORT/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d "{\"model\":\"$SERVED_NAME\",\"messages\":[{\"role\":\"user\",\"content\":\"What is 17*23? Just the number.\"}],\"max_tokens\":30,\"temperature\":0,\"chat_template_kwargs\":{\"thinking\":false}}" \
    | jq -r '.choices[0].message.content'
done
# expected: identical output x3
```

**What this gate does not cover.** These prompts are short — they never
cross the chunked-prefill boundary. The failure mode that matters lives
in the *last chunk* of a chunked prefill when that chunk runs as a
piecewise CUDA graph. Gate 9 is the one that reaches it; run both.

## 6. Strict JSON compliance

```sh
curl -s http://$HEAD_IP:$API_PORT/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d "{\"model\":\"$SERVED_NAME\",\"messages\":[{\"role\":\"user\",\"content\":\"Return a JSON object with key answer and value 42. Nothing else.\"}],\"max_tokens\":50,\"temperature\":0,\"chat_template_kwargs\":{\"thinking\":false}}" \
  | jq -r '.choices[0].message.content'
# expected: exactly {"answer":42} (reasoning stripped by the parser)
```

## 7. Tool calling

```sh
curl -s http://$HEAD_IP:$API_PORT/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d "{\"model\":\"$SERVED_NAME\",\"messages\":[{\"role\":\"user\",\"content\":\"What is 3+5? One word.\"}],\"max_tokens\":30,\"temperature\":0,\"tools\":[{\"type\":\"function\",\"function\":{\"name\":\"calc\",\"description\":\"calculator\",\"parameters\":{\"type\":\"object\",\"properties\":{\"expr\":{\"type\":\"string\"}}}}}],\"tool_choice\":\"auto\",\"chat_template_kwargs\":{\"thinking\":false}}" \
  | jq -r '.choices[0].message.content'
# expected: a direct answer (no tool call needed for simple math); no error
```

## 8. Short prompts under speculative decode

A prefill whose scheduled token count equals
`uniform_decode_query_len * num_reqs` can be misread as a uniform-decode
batch (upstream vLLM #53051). The probe resolves that length from the
running engine, builds a prompt of exactly that many tokens, and checks
the answer at temperature 0.

```sh
python3 launch/tp3-kit/spec-short-prompt-check.py \
  --base-url http://$HEAD_IP:$API_PORT --model $SERVED_NAME \
  --container ds4flash-tp3-head
# expected: SPEC-SHORT-PROMPT PASS
```

Read-only: one request. Run it against a lane being accepted, not one
carrying traffic.

If the probe cannot build a prompt carrying an echo marker, it falls back
to a marker-free prompt of exactly the right token count and checks only
that the answer is sane — the marker assertion is then **not** made. Read
the probe's own output to see which path it took: a fallback run proves
the shape is accepted, not that the marker survived.

## 9. The corruption battery

Next to every speed run, on the same boot: exact repeats at temperature
0, a strict-JSON request, and a needle in a haystack that is longer
than anything exercised so far. A speed win on a lane that corrupts
output is not a win.

```sh
# 9a. exact repeat — any drift or non-ASCII is a failure
for i in 1 2; do
  curl -s http://$HEAD_IP:$API_PORT/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d "{\"model\":\"$SERVED_NAME\",\"messages\":[{\"role\":\"user\",\"content\":\"List three prime numbers.\"}],\"max_tokens\":40,\"temperature\":0,\"chat_template_kwargs\":{\"thinking\":false}}" \
    | jq -r '.choices[0].message.content' | tee "/tmp/repeat-$i.txt"
done
diff /tmp/repeat-1.txt /tmp/repeat-2.txt && echo "REPEAT IDENTICAL"

# 9b. strict JSON — the same gate as section 6, in the battery
# 9c. needle in a haystack — build a haystack with a fresh nonce, and
#     ask for the nonce back. The nonce is what makes this honest:
#     without it, prefix caching replays the "new" prompt and reports
#     cache hits as retrieval.
NONCE=$(python3 -c "import uuid; print(uuid.uuid4().hex[:12])")
python3 - "$NONCE" "$SERVED_NAME" > /tmp/haystack.json <<'PY'
import json, sys
nonce, model = sys.argv[1], sys.argv[2]
filler = "The quarterly logistics review covers inventory, freight, and demand planning. " * 900
print(json.dumps({
    "model": model,
    "messages": [{"role": "user", "content":
                  f"{filler}\nThe access code for the locked cabinet is {nonce}. "
                  f"Reply with the access code and nothing else.\n{filler}"}],
    "max_tokens": 24, "temperature": 0,
    "chat_template_kwargs": {"thinking": False},
}))
PY
curl -s http://$HEAD_IP:$API_PORT/v1/chat/completions \
  -H "Content-Type: application/json" -d @/tmp/haystack.json \
  | jq -r '.choices[0].message.content'
# expected: the nonce, verbatim
```

## 10. Anti-contamination

Every prefill-class prompt embeds a fresh random nonce (gate 9c is the
worked example). Without one, prefix caching replays the "new" prompt
and reports cache hits as prefill — measured in the 10–15× range on
this stack. The same rule applies to every long-context claim: a fresh
nonce per measurement, never a replayed payload.

## 11. The idle gate

Before any solo, tier or speed reading, require an idle engine:

```sh
curl -s http://$HEAD_IP:$API_PORT/metrics | grep -E "^vllm:num_requests_(running|waiting)"
# expected: both zero
```

A "slow" reading taken with traffic in flight — warmup replays,
scheduled jobs, co-tenant requests, other clients — is an unlabeled
mixed number, not a property of the boot. Do not act on it; re-measure
idle. Post-boot warmup traffic overlaps exactly the window a fresh boot
first gets probed: let it settle, or verify idleness.

## 12. The workload split for acceptance-bound metrics

With speculative decoding, speed depends on draft acceptance, which
depends on what the model is writing. Measure the classes separately —
high-draftability (count/code), free prose, N-way aggregate (4/8/16),
prefill at a realistic long-context size — and publish the split, not a
single number. The acceptance histogram comes from the engine's own
metrics:

```sh
curl -s http://$HEAD_IP:$API_PORT/metrics \
  | grep -E "^vllm:spec_decode_num_accepted_tokens_per_pos_total"
# expected: one counter per draft position
```

A single headline number hides all of this.

## 13. The KV pool, from the boot log

Quote the engine's own pool, never an estimate.

```sh
docker logs ds4flash-tp3-head 2>&1 | grep -E "GPU KV cache size|Maximum concurrency"
# expected: the pool and the concurrency figures the boot reports
```

## 14. Image identity

The digest must be the pinned one, on every box:

```sh
docker image inspect eugr/spark-vllm-b12x:nightly-20261004@sha256:66147fc1420dbd59a7bb1411697d81a8231c298105b61c5962179cd146b4a41b \
  --format '{{range .RepoDigests}}{{println .}}{{end}}' \
  | grep -F 'sha256:66147fc1420dbd59a7bb1411697d81a8231c298105b61c5962179cd146b4a41b'
# expected: the pinned digest
```

The digest proves which manifest was pulled; the engine-identity gate
(section 3) proves what executes inside the container.

## 15. The wheelhouse (capacity profile)

The capacity profile installs from a local wheelhouse and fetches
nothing at boot. The boot verifies it first — this is the same check,
run by hand:

```sh
docker exec ds4flash-tp3-head python3 /opt/tp3-kit/verify-wheelhouse.py \
  --wheelhouse /wheelhouse
# expected: WHEELHOUSE PASS
```

If the wheelhouse is missing or its sha256s do not match the pre-flight
record, the boot refuses to install from it — which is the point.

## 16. The offline CSF toolchain

The CSF converter runs on stock PyTorch, with no GPU and no engine:

```sh
python3 tools/csf/roundtrip_validate.py SOURCE_SHARD CONVERTED_SHARD
# expected: every compressed plane decodes bit-exact to the source, and
# every pass-through tensor is byte-identical

python3 tools/validate-tp3-plan.py PATH_TO_virtual_tp.py
# expected: the virtual-TP plan for the checkpoint's geometry validates
```

## 17. Recording discipline

Every cycle's full output is teed to a timestamped file; on failure,
all three ranks' logs are captured before anything else happens.

```sh
OUT="$HOME/measurements/$(date +%Y%m%d-%H%M%S)-cycle.log"
mkdir -p "$(dirname "$OUT")"
# ... the measurement command, teed:
your-measurement-command 2>&1 | tee -a "$OUT"
```

Keep a running log with, per cycle: the exact config delta, the boot
facts (pool, time-to-healthy), all the numbers per workload, the
decision, and the trigger — what caused what. Numbers that exist only
in a terminal scrollback do not exist. End the window with a summary
table and a "do not re-derive" trigger map.

## 18. The mixed battery — the acceptance gate

A boot is not accepted because single-class probes pass. The gate is a
**mixed battery**: interactive streams, chat-class requests, and the
co-tenant services, if any, all in flight at the same time, with host
memory sampled throughout.

```sh
# co-tenant view while the lane is under the mixed load
watch -n 2 "free -g | awk '/Mem/ {print \$3, \$7}'"
# expected: the head box keeps its documented headroom floor
```

A lane that only survives isolated benchmarks is not validated for the
two-surface operating mode this recipe targets (coding harness + chat
simultaneously). Build the mixed battery from the classes above and run
it beside every adoption decision.

---

## Honest reporting rules

- A repeated-boot jitter band (a few percent on the KV pool, a few
  percent on throughput samples) means ranges, not points — and A/B
  verdicts need non-overlapping samples, not vibes.
- Speed settings interact with capture sizes and pool sizes; when a
  faster setting costs memory, publish both variants and the workload
  each suits.
- Where the measurement disagrees with the model card's official
  recommendation, publish both and say which weights the measurement
  was taken on.
- "Not tested / not claimed" is a section in the README, not a
  footnote. Sustained-load windows, contexts beyond what was exercised,
  and untested quantizations stay unclaimed.

## Pass criteria

Every gate above must pass. A failed gate is diagnosed forward from the
FIXLOG, never rolled back automatically.
