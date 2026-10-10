# Cache benchmark with starting-temperature admission, October 10, 2026

Follow-up to the [longer balanced trial](../cache-balanced/README.md), which
had unmatched starting temperatures. This trial uses the same cached CPU
expert implementation, model, prompt, sampling seed, 64 evaluated positions,
six threads and baseline/cache/cache/baseline order. Every subsequent
full-vocabulary logit vector must match the first baseline bit for bit.

## Admission policy

The C++ benchmark requests admission after each fresh context and batch are
created, before prefill and outside timed forward calls. A Python supervisor
samples readable battery, maximum CPU/CPU-subsystem and maximum HTP HMX/HVX
temperatures every two seconds. Only a fresh, pass-specific approval releases
the benchmark. Contexts remain allocated while waiting; each cache starts empty.

The first pass must have a continuous 30-second stable window below battery
34°C, CPU 55°C and HTP 45°C. The medians of that window become a frozen
reference for all four passes. Every sample in each admission window must
lie within **±0.4°C battery, ±4°C CPU and ±4°C HTP** of that reference.
The first window must satisfy those stability bands as well. Missing or
nonfinite required sensors, or sample gaps over 3.5 seconds, reset the hold.
This matches starting conditions within stated bands; it does not make
sensor readings exactly equal or control temperatures during inference.

The original 30-second minimum rest remains between passes. The runner waits for further cooling without changing system settings or
thermal throttling controls. Ambient conditions and any external cooling
are not measured. Each admission has a 900-second timeout;
the complete child has a 4800-second limit and 400 MiB available-memory floor.
A failed gate stops the child and retains partial artifacts; bands are not
widened. No energy or sustained-thermal conclusion follows from this test.

## Validation and reproduction

Three synthetic thermal-policy tests cover the continuous hold, frozen
reference, hotter/missing/nonfinite sensor rejection, initial ceilings and
sampling gaps. Four benchmark-analysis tests include raw admission-window
validation and rejection of a changed reference or out-of-band sample.
The updated C++ benchmark compiled with `-Wall -Wextra` without warnings.

```sh
clang++ -std=c++17 -O2 -Wall -Wextra scripts/moe-cache-benchmark.cpp \
  -I.work-npu/llama.cpp/include -I.work-npu/llama.cpp/ggml/include \
  -L.work-npu/build/bin -Wl,-rpath,"$PWD/.work-npu/build/bin" \
  -lllama -lggml -lggml-base -o .work-npu/build/bin/moe-cache-benchmark
python reports/2026-10-10/cache-thermal-matched/run-benchmark.py
python scripts/analyze-moe-cache-benchmark.py reports/2026-10-10/cache-thermal-matched
```

The supervisor intentionally refuses to reuse an existing `admission`
directory. To repeat, preserve the artifacts and copy the runner into a new
report directory at the same depth. The model path is the local
Qwen3.6-35B-A3B UD-IQ1_M GGUF; the existing opt-in CPU cache must be built.
Without `MOE_BENCH_ADMISSION_DIR`, the benchmark retains its previous fixed-rest
behavior. Timing still includes only synchronized forward calls, with cache
allocation and refill included; cooling, setup and verification are excluded.

Artifacts include thermal JSONL, admissions JSON with the exact stable-window
bounds and samples, reference tokens/logits verification counts, per-pass
latency/I/O, process memory, exit status, cache counters and source/binary
checksums. The analyzer validates each accepted window against the raw trace.

## Outcome: incomplete, no comparative result

The supervisor stopped only its own child with SIGTERM after cached pass 2
failed to meet admission within 900 seconds. This is the expected timeout
path, not a model or memory failure. Total wall time was 24.49
minutes. Minimum sampled available memory was 4.33 GiB;
peak sampled process RSS was 5.67 GiB.

The first baseline waited 242.6 seconds and was admitted at
**25.7°C battery, 36.6°C CPU, 34.6°C HTP**. Its stable-window reference
medians were **25.7°C / 36.8°C / 34.6°C**, frozen for every pass. The
16-sample, 30.08-second accepted window validates against the raw trace.
The baseline completed all 64 positions in **279.316 seconds**, reading
**237.18 GiB**. Its sampled tokens match the earlier trial exactly.

During the cached pass's admission wait, battery temperature fell from
32.9°C to 28.8°C, with a minimum of
28.8°C. It never entered the allowed **25.3–26.1°C** band.
The cache context was set up but never performed timed inference; no cache
statistics, paired speedup or replayed-logit correctness result is available
from this attempt. `outcome.json` records the incomplete status. The full
comparison analyzer deliberately rejects this stopped run.

The recorded trace peaks were **32.9°C battery,
76.7°C CPU and 70.1°C HTP**. They cannot be used to compare modes:
only the baseline ran. The earlier 2.35× observation remains an unmatched
starting-temperature result.

**Next:** establish a common starting reference that can be recovered after
inference, after letting the phone thermally equilibrate, and repeat in a
fresh report directory. This attempt does not identify why the initial cold
reference could not be recovered; ambient conditions and external cooling
were not recorded. Preserve the declared bands and verify every admitted
window instead of reinterpreting this incomplete attempt as a comparison.

## Later user direction: prioritize generation speed

After this stopped attempt, the user requested temperatures a few degrees
higher and no more testing of this part. Future admission policy is now
`relaxed-plus-3c-v1`: each tolerance and first-window ceiling increases by
3°C (battery ±3.4°C, CPU/HTP ±7°C; initial ceilings 37/58/48°C).
The policy and analyzer recognize the new version; historical artifacts
above retain their original strict bands. No additional thermal-policy tests
or trial were run for this adjustment. Generation throughput is the priority;
ordinary performance trials can omit the optional admission gate.
