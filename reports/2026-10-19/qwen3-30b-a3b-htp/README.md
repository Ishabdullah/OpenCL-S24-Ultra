# Qwen3-30B-A3B HTP expert validation

This report tests model-level routed expert execution on the S24 Ultra after
the synthetic HTP0 `MUL_MAT_ID` correctness sweep. The selected artifact is
the official [Qwen3-30B-A3B GGUF repository](https://huggingface.co/Qwen/Qwen3-30B-A3B-GGUF),
Q4_K_M file `Qwen3-30B-A3B-Q4_K_M.gguf`. The model card describes the
`qwen3moe` architecture with 30.5B total / 3.3B active parameters, 128
experts and 8 active experts per token. A mirror's listing reports an
18.56 GB Q4_K_M file; the actual official download's response size and
completed byte count are recorded below. Q4_K_M is selected because the
pinned HTP backend accepts Q4_K and Q6_K `MUL_MAT_ID`, unlike IQ1/IQ2.

The file is larger than physical RAM. All model loads must set `-lm mmap`.
The repository's previous 8.4 GiB HTP experiment measured an eager-load OOM
under `-lm auto`; explicit mmap fixed that case. Because this GGUF is over
twice as large, this campaign starts with one HTP-offloaded repeating
transformer layer plus the output layer (`-ngl 2`), a requested context of 64
(the runtime clamps this to 256) and one generated token. The other tensors remain
mmap-backed and CPU-resident. The smoke pins CPU work to the A520 cluster at
low scheduler priority. It sets `LLAMA_MMAP_PREFETCH=0`, an experiment-only
switch in the local llama.cpp checkout that retains mmap but skips Linux
`MAP_POPULATE`/whole-file `POSIX_MADV_WILLNEED` at startup, and passes
`--no-repack` to prevent a second CPU_REPACK copy of the full model. Do not run
unless the device is connected to AC, has at
least 5 GiB `MemAvailable`, battery is at least 40%, CPU is at most 55 C, HTP
is at most 50 C, and battery is at most 40 C. During a child process, stop it
if `MemAvailable` falls below the repository's 2 GiB retained-headroom floor,
battery falls below 45%, CPU exceeds 70 C for two consecutive one-second
samples, battery exceeds 42 C, HTP exceeds
60 C or the 15-minute watchdog fires. Only the test child is signaled. The
HTP stop was raised prospectively from 55 C after Attempt 26; earlier attempt
records retain their run-time limit.

## Gates

1. Complete and hash the download with a streaming tool such as `sha256sum`;
   never read a whole multi-gigabyte model into a Python `bytes` object.
   Record free storage and battery.
2. Read only the GGUF header and tensor directory. Require `qwen3moe`, at
   least one routed expert tensor, and all `ffn_*_exps` tensor types in the
   tested HTP `MUL_MAT_ID` support set: Q4_0, Q4_1, Q8_0, IQ4_NL, MXFP4,
   Q4_K, Q5_K, Q6_K. If any routed tensor is outside that set, stop without
   inference.
3. Run a short HTP0 smoke generation with the experimental routing trace
   enabled and op fusion disabled. This keeps routed matmuls as ordinary
   `MUL_MAT_ID` nodes so the current trace hook can read their route IDs.
   Require HTP0 session creation, finite completion, routed expert trace
   batches or backend `execute-op MUL_MAT_ID` records for gate/up/down with
   HTP0 buffers, and no memory/thermal guard stop. The cache trace is
   diagnostic; it synchronizes the queue and must not be used for speed
   measurements.
4. Only after the one-layer smoke passes, run the full-vocabulary CPU-versus-
   one-layer-HTP logit comparison with the same mmap model and a one-token
   prompt, one vector at a time. Require matching top-1 token, finite full
   vocabularies, and mean KL divergence at or below 0.001, matching the
   project's prior forced-prefix numerical gate. Comparison runs now require
   CPU temperature at or below 48 C at admission and insert a monitored
   180-second idle pause between CPU and HTP passes. Expand
   HTP layer count only in a later fresh attempt after reviewing memory and
   thermal telemetry. Do not run an all-layer throughput test in this initial
   campaign; the existing 8.4 GiB precedent does not establish safe HTP buffer
   headroom for this 18.56 GB model.

## Evidence

The first reported Termux crash likely came from my metadata command using
`Path.read_bytes()` on the complete 18.56 GB model to hash it; the llama.cpp
process had not started at that point. The hash was already collected safely
with streaming `sha256sum`. Attempts 1–3 were stopped at the 60 C CPU guard
while `load_tensors` ran. Attempts 1–3 retained at least 5.83 GiB available
with no swap use. Attempt 4 disabled mmap prefetch but still touched the
17.2 GiB `CPU_REPACK` buffer: RSS reached 5.7 GiB, swap reached 2.0 GiB, and
available memory bottomed at 2.95 GiB before the thermal guard stopped it.
Attempt 5 was refused before launch because CPU temperature exceeded the
55 C admission limit. Attempt 6 added `--no-repack`; it completed a one-token
generation with peak RSS 1.57 GiB, minimum available memory 5.70 GiB, negligible
swap, and CPU temperature below 53 C. Its trace gate failed because `-ngl 1`
placed only the output layer on HTP, with zero repeating layers. Attempts 7
and 8 used `-ngl 2`, placing one repeating layer plus the output layer on HTP;
both completed one-token generation with no guard event. Attempt 7 peaked at
1.56 GiB RSS and retained 5.89 GiB available; attempt 8 peaked at 1.50 GiB RSS
and retained 6.06 GiB available. Neither emitted a routed expert trace. Attempt
8 enabled `GGML_HEXAGON_VERBOSE=1`, but the CLI verbosity threshold remained at
4 and suppressed backend debug messages. Attempt 9 raised it to 5 and
completed with no guard event (peak RSS 1.61 GiB, minimum available memory
6.27 GiB, no swap). Its HTP0 backend records show layer 47's gate and up Q4_K
`MUL_MAT_ID` and down Q6_K `MUL_MAT_ID` with HTP0 source and destination
buffers; the one-token inference completed. The cache trace remains empty
despite these records, so there is a hook/collection gap. The raw log and
parsed evidence are preserved in `attempt-9/smoke.log` and
`attempt-9/htp-route-analysis.json`. The runner now accepts all three
backend-execution records as route evidence when the cache trace is absent.
The bounded load and model-level expert operations are demonstrated. Attempt
10 started the original eight-step comparison but the CPU pass's multi-token
prompt prefill crossed the 60 C guard; it was stopped with at least 6.79 GiB
available memory and no crash. The comparator is now reduced to the single
`Hi` prompt vector with batch/ubatch 1, which still compares every vocabulary
logit while limiting CPU work. Attempt 12 compared the full 151,936-token
vocabulary vector: top-1 matched, all values were finite, mean KL was 0.000572
(the project's established limit is 0.001), and RMSE was 0.03427. The vector
was not bit-identical; maximum absolute error was 0.16733. Peak RSS was 1.62
GiB, minimum available memory 6.13 GiB, swap use zero, and maximum CPU
temperature 47.1 C. This passes the bounded one-vector numerical gate.
Attempt 13 extended the comparison to four sequential forced-prefix vectors.
All four top-1 tokens matched and logits were finite, but mean KL was 0.001740
and RMSE was 0.11135, so it fails the 0.001 quality gate. It completed without
a guard stop (peak RSS 3.17 GiB, minimum available memory 6.62 GiB, peak swap
45.8 MiB; CPU max 57.2 C and HTP max 44.0 C). Do not expand offload or make
quality/performance claims until the multi-step logit drift is understood.
Attempt 14 repeated the four-vector run with per-step reporting but crossed the
60 C guard after the first HTP vector; step 1 repeated KL 0.000572/RMSE 0.03427.
Attempt 15 used two vectors but crossed the guard during the HTP pass before it
produced a comparison metric. Both preserved more than 5.8 GiB available
memory and exited through the owned-child thermal guard. The comparator now
cools for 180 seconds after releasing the CPU model before starting HTP, and
comparison admission requires CPU <=48 C. Attempt 16 completed two vectors
after the monitored inter-pass rest. Step 1 remained at KL 0.000572/RMSE
0.03427; step 2 rose to KL 0.001756/RMSE 0.13257. Both top-1 tokens still
matched, while the two-step mean KL was 0.001164, above the 0.001 gate. This
locates the first measured quality-gate failure at the second forced-prefix
position. The run stayed within guards (peak RSS 2.30 GiB, minimum available
memory 6.05 GiB, peak swap 47.5 MiB, CPU max 57.2 C). Attempt 17 completed
four vectors with the inter-pass rest and per-step reporting. KL by step was
0.000572, 0.001756, 0.002420, and 0.002213; RMSE was 0.03427, 0.13257,
0.10295, and 0.14228. All top-1 tokens matched and all logits were finite.
The aggregate remains outside the quality gate (mean KL 0.001740). Peak RSS
was 3.17 GiB, minimum available memory 6.37 GiB, peak swap 45.5 MiB, CPU max
56.1 C, and HTP max 42.4 C. These measurements compare CPU execution against
a final layer entirely on HTP (attention and experts), so they cannot yet
attribute the error to the routed expert kernels. The next attribution run
uses `GGML_HEXAGON_OPFILTER=^MUL_MAT_ID$` to leave the layer's other supported
ops on HTP while sending expert projections to CPU; it records whether any
expert `execute-op` was still submitted to HTP.

Attempt 18 completed that expert-offload control over two vectors. No expert
`MUL_MAT_ID` execute records appeared, confirming that the routed projections
were kept off HTP. Step KL was 0.000666 and 0.001865 (mean 0.001266), with both
top-1 tokens matching; the quality gate still failed. Thus the drift is not
removed by forcing expert matrix operations to CPU and is unlikely to come
from those kernels alone. The run stayed within memory guards (peak RSS 2.74
GiB, minimum available memory 6.38 GiB, peak swap 2.9 MiB). For the next
attribution step, the comparator and guard now support an inverse filter that
allows `MUL_MAT_ID` plus `NONE` metadata ops on HTP, while logging every HTP
`execute-op` record.
This will test routed expert execution with the rest of the layer, including
attention, on CPU. It is admitted only after CPU temperature is at most 48 C,
with AC power, at least 5 GiB available memory, and the existing live guards.
Attempt 19 validated the inverse regex and safely loaded the model, but graph
reservation aborted before the HTP decode because the KV cache had been
pre-placed on HTP while its required `NONE` metadata operation was filtered
out. This was a child-process scheduler abort (`returncode -6`), not a Termux
crash; minimum available memory remained 6.73 GiB and no comparison vectors
were produced. The next retry keeps KQV/KV allocation on CPU specifically for
the expert-only mode, while leaving the expert `MUL_MAT_ID` weights eligible
for HTP. Its output will include every HTP operation record to verify the
actual placement before interpreting logits.
Attempt 20 applied CPU KV/KQV placement, but graph reservation then found that
an HTP-resident expert weight also requires the backend to accept its `NONE`
metadata node. It again aborted before inference (minimum available memory
6.45 GiB, zero swap). The inverse filter now permits `NONE` as a metadata-only
operation alongside `MUL_MAT_ID`, while continuing to reject other compute
operations. Attempt 21 completed the two-vector CPU-vs-expert-only-HTP
comparison. The log contains six HTP executions, all `MUL_MAT_ID` (gate, up,
and down, repeated once per vector); no attention or other compute op was
executed on HTP. Both top-1 tokens matched and all logits were finite. Per-step
KL was 0.000239 and 0.000151 (mean 0.000195), passing the 0.001 quality gate;
RMSE was 0.05370 and maximum absolute error 0.36080. CPU-only expert execution
in Attempt 18 had mean KL 0.001266, while the original whole-layer HTP route
had mean KL 0.001164 over two vectors in Attempt 16. This result indicates the
routed expert kernels alone do not reproduce that measured multi-step drift;
the attention/other operations or their interaction with HTP placement remain
the stronger suspects. Attempt 21 stayed within guards: peak RSS 2.63 GiB,
minimum available memory 5.91 GiB, peak swap 52.9 MiB, maximum CPU 56.8 C, HTP
41.6 C, and battery 28.0 C. Attempt 22 repeated the expert-on-CPU control
with full backend op logging. It reproduced mean KL 0.001266 (step KL
0.000666/0.001865), with matching top-1 tokens. Across its two vectors, HTP
ran 102 operations: 16 `MUL_MAT` (attention Q/K/V/output, router, and final
logits), four each of `ROPE`, `SET_ROWS`, and `SOFT_MAX`, plus `MUL`, `ADD`,
normalization, cache/read/selection, and related graph ops. This confirms the
control was broader than attention alone and that MoE combine and router work
also remained on HTP. Attempt 22 stayed within guards (peak RSS 2.70 GiB,
minimum available memory 6.63 GiB, peak swap 40.8 MiB, CPU max 56.8 C, HTP
max 42.0 C, battery max 28.5 C). The next run uses an attention-core op
whitelist (`MUL_MAT`, `SOFT_MAX`, `ROPE`, `SET_ROWS`, `CONT`, plus `NONE`
metadata) while routed `MUL_MAT_ID` expert ops and other compute ops fall back
to CPU. Since the whitelist also admits router and output `MUL_MAT` ops, the
backend log will be checked to document exactly what ran on HTP.
Attempt 23 reached HTP graph reservation but aborted before decode because
the pre-allocated HTP KV value cache also needed a `RESHAPE` operation. No
vectors were produced; memory and thermal telemetry remained safe. The
attention-core whitelist now also permits layout-only `RESHAPE`, `VIEW`,
`PERMUTE`, and `TRANSPOSE` operations so HTP-resident cache tensors can be
shaped without broadening the compute whitelist.
Attempt 24 then completed with that layout allowance. HTP executed 30
operations across two vectors: 16 `MUL_MAT` plus four each of `ROPE`,
`SET_ROWS`, and `SOFT_MAX`, and two `CONT`. No expert `MUL_MAT_ID` ran on HTP.
The per-step KL exactly reproduced Attempt 22 at 0.000666 and 0.001865 (mean
0.001266), so the expert kernels are not needed to reproduce this drift. The
run stayed within guards: peak RSS 2.53 GiB, minimum available memory 6.47
GiB, peak swap 48.95 MiB, maximum CPU 54.1 C, HTP 40.9 C, and battery 28.3 C.
The next control will allow only dense `MUL_MAT` and metadata/layout ops on
HTP, keeping KV/KQV on CPU. This will help distinguish dense projection
differences from attention softmax, RoPE, and cache-update operations.
Attempt 25 completed that dense-matmul-only control. The backend log recorded
12 HTP executions across two vectors, all `MUL_MAT`: Q, K, V, and attention
output projections for block 47, the block-47 router, and the final output
projection, each repeated for the second vector. KV/KQV remained CPU-side and
no `MUL_MAT_ID` expert operation ran on HTP. Both top-1 tokens matched and all
logits were finite, but mean KL rose to 0.001471 (step 1: 0.000666; step 2:
0.002275), above the 0.001 quality gate; RMSE was 0.10123. Dense HTP matmuls
therefore suffice to reproduce a quality-gate failure in this setup, though
this whitelist still combines attention projections, router, and output
projection. The run stayed within guards (peak RSS 2.50 GiB, minimum available
memory 6.77 GiB, peak swap 57.3 MiB, max CPU 52.9 C, HTP 40.5 C, battery
28.4 C). The next discriminating test should place only block-47 attention
weights on HTP and force router/output weights to CPU, or add tensor-aware
backend filtering; the current op-type filter cannot separate these dense
matmuls by tensor name.

After the CPU-side thread pilot reached the updated sustained stop, Attempt 26
tested a bounded increase to `-ngl 4` (three repeating layers plus output)
with a one-token smoke. It passed and logged all three expert projection
families on HTP across blocks 45–47. Minimum `MemAvailable` was 5.09 GiB,
peak RSS 1.42 GiB, peak swap 45.5 MiB, and HTP max 45.5 C. CPU peaked at
72.0 C in a transient sample; the current policy stops only after two
consecutive one-second readings above 70 C, and no sustained stop occurred.
Evidence is under `attempt-26-ngl4/`. This establishes a viable bounded
placement for a short throughput pilot, not a quality or speed result.

The guarded model runner now uses the prospective CPU stop policy of two
consecutive one-second readings above 70 C. Earlier attempts retain their
recorded 60 C stop policy.

Attempt 27 increased the bounded placement to `-ngl 6` (five repeating layers
plus output) for a one-token smoke before measuring throughput. The smoke
completed with all three expert projection families on HTP across blocks
43–47 (15 routed expert operations). CPU peaked at 60.7 C and HTP at 46.3 C;
minimum `MemAvailable` was 3.79 GiB, peak RSS 2.08 GiB, and peak swap 40.4
MiB. No thermal or memory guard fired. The smoke reports an unrepresentative
single-token timing and does not establish a speed result. Evidence is under
`attempt-27-ngl6/`.
