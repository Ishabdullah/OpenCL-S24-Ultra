# HTP routed expert operation validation

This report validates the pinned Snapdragon 8 Gen 3 Hexagon backend's
`MUL_MAT_ID` (routed expert matmul) path with bounded synthetic matrices.
It separates backend-kernel support from full-model validation. A model
benchmark is valid only after its routed expert tensor types pass the HTP
support gate and runtime logs confirm that `MUL_MAT_ID` ran on HTP0.

## Why a synthetic gate is used

The local MoE candidates are Qwen3-Coder-Next REAP-40B-A3B `i1-IQ1_S`
(7.85 GiB) and Qwen3.6-35B-A3B `IQ1_M`. The first has IQ1_S routed experts;
the latter's routed experts use IQ1/IQ2 types. This pinned HTP backend rejects
IQ1/IQ2 `MUL_MAT_ID` weights, so those models cannot exercise HTP expert
matmuls. `-dev HTP0` can still offload supported dense/attention operations;
that is not evidence that experts ran on HTP. The earlier Qwen3-Coder-Next
benchmark's `--n-cpu-moe 0` result must therefore not be presented as proof
of HTP execution for its IQ1_S experts.

The synthetic tests use small, bounded matrices and do not load a model or
allocate a model-sized buffer. They exercise HTP0 v75 and compare its results
with the backend test harness's expected CPU result. Unsupported low-bit
types are explicitly checked at the support gate.

## Test plan and safety limits

1. Confirm HTP0 registration and the support matrix for the supported expert
   quantizations.
2. Run one CPU-comparison correctness case per supported quantization:
   Q4_0, Q4_1, Q8_0, IQ4_NL, MXFP4, Q4_K, Q5_K and Q6_K, with 4 matrices,
   2 selected experts, 64 output rows and 16 input columns. K-block types use
   the backend's valid 768-wide reduction shape.
3. Confirm IQ1_S, IQ1_M, IQ2_XXS, IQ2_XS and IQ2_S are rejected by HTP's
   `MUL_MAT_ID` support gate.
4. Retain command output, source/binary SHA256s, battery, thermal readings,
   available-memory samples, peak test-process RSS and exit statuses.

The runner refuses to start below 2 GiB `MemAvailable`, below 40% battery,
above 60 C CPU or above 42 C battery temperature. It kills only its own
child if available memory falls below 768 MiB or a check exceeds five minutes.
No full model is loaded. Use `-lm mmap` for any later HTP model run; the
repository research documents an eager-load/OOM failure when HTP0 is selected
without explicit mmap.

## Run and analyze

```sh
python reports/2026-10-19/htp-expert-ops/run-htp-expert-ops.py
python reports/2026-10-19/htp-expert-ops/run-htp-expert-ops.py --run \
  --out reports/2026-10-19/htp-expert-ops/attempt-N
```

`run.json` records preflight/postflight state and provenance. Each check has
its own `.log`. The initial and guarded reruns are preserved in `attempt-1/`
and `attempt-2/`; the final provenance-complete rerun is in `attempt-3/`. It
started at 80% battery, 32.9 C battery sensor, 47.5 C maximum CPU sensor
and 5.71 GiB `MemAvailable`. Across checks, minimum available memory was
5.65 GiB and peak test-process RSS was 23 MiB.
All three check processes exited 0. The correctness check passed 8/8 expert
matmul cases; no full model was loaded and no crash or watchdog stop occurred.

The experimental Hexagon backend library was rebuilt incrementally from the
modified pinned checkout before the tests. These are backend correctness
tests, not a throughput test. The trace-only
Hexagon cache hook was compiled into the local backend library, but this run
does not validate that hook or prove model routing IDs are captured. No
compatible full-model MoE GGUF is currently installed, so the next full model
test requires identifying a model whose routed weights use a supported type,
checking its size against device memory/storage limits, then repeating the
support gate before downloading or benchmarking it.

## Device result

The support probe reports all eight supported types above as supported and
the five IQ1/IQ2 types as unsupported. All eight supported synthetic expert
matmul correctness cases passed on HTP0. This proves those kernel paths work
for the tested shapes; it does not establish model-level numerical fidelity,
cache locality, throughput, or compatibility with every shape of those
quantizations.
