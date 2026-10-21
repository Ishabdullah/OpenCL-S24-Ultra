# Qwen3-30B-A3B guarded thread sweep

This is the first throughput sweep for `Qwen3-30B-A3B-Q4_K_M.gguf`, using
the earlier Qwen3.6 80-slot sweep only as a starting hypothesis. It holds
the cache at 80 slots / 3200 MiB and compares threads in balanced order
4 → 3 → 2 → 2 → 3 → 4. The default bounded HTP placement is one repeating
layer plus output (`--ngl 2`); `--ngl 4` and `--ngl 6` select three and five
repeating layers plus output for smoke-qualified diagnostics. Pass 1 generates up to 64 positions; later passes
replay those exact tokens and must match every logit bit-for-bit. The timed
metric is synchronized decode forward time, excluding prefill and setup.

The target has already passed bounded HTP expert-operation validation. This
runner deliberately uses the demonstrated safe placement: mmap, no prefetch,
no weight repacking, batch/ubatch 1, context at least 256, and `n_gpu_layers=2`
(one repeating layer plus output). It does not request full-model HTP offload.
The 80-slot cache and 64-position pass length are experimental; a later cache
capacity sweep should follow only after this thread comparison completes.

Admission follows the earlier physical thread-sweep protocol: AC, battery ≥40%,
MemAvailable ≥5 GiB, CPU ≤55°C, battery
≤40°C, and HTP ≤50°C. The live run stops its child below 2 GiB MemAvailable,
after two consecutive one-second CPU readings >70°C, immediately at battery
>42°C or HTP >60°C, battery <45%, or after one hour. The HTP stop was raised
prospectively from 55°C; prior attempt records retain their run-time limit.
Between passes the benchmark is paused for at least 180 seconds, up to 300,
while the device cools. Model loading is mmap-backed and `LLAMA_MMAP_PREFETCH=0`
avoids whole-file prefetch. Only the benchmark process is stopped.

Run the dry inspection with:

```sh
python reports/2026-10-19/qwen3-30b-a3b-thread-sweep/run-benchmark.py
```

Physical execution requires an attended, plugged-in device and passing live
preflight:

```sh
python reports/2026-10-19/qwen3-30b-a3b-thread-sweep/run-benchmark.py --run
```

Use a fresh `--out-dir` for each subsequent attempt so prior evidence is
preserved. The CPU stop policy was raised prospectively to two consecutive
one-second readings >70°C; existing attempt records retain the 60°C policy
that was active when they ran.

If a full-cluster run hits the thermal hard stop before producing a pass, run
the one-thread, two-position CPU0–1 pilot with `--pilot` and a fresh
`--out-dir`. It is a load/thermal diagnostic, not a full-cluster throughput
comparison. The prompt is `Hi`, matching the validated one-token HTP
comparison; prefill is excluded from decode timing.

## Progress

The dedicated C++ runner compiled and its dry-run plan was inspected. The
existing six lifecycle/replay tests for the source sweep runner pass. Initial
preflight snapshots are in `preflight.jsonl`. Once gates passed, attempt 1
began mmap model loading but the copied cooldown monitor exited immediately:
it searched for `moe-thermal-monitor.py` beside itself instead of under
`scripts/`. The supervisor stopped its child cleanly before any pass began.
`run.json` records return code -15, zero passes, and 5.22 GiB minimum available
memory; `benchmark.jsonl` is empty. Termux remained running. The cooldown import
path was fixed for attempt 2. That attempt reached mmap placement (CPU-mapped
buffer 17,063 MiB; HTP buffer 680 MiB), then the live thermal guard saw a CPU
sensor peak of 84.9°C and stopped the child before any benchmark pass. Minimum
available memory was 4.48 GiB, peak RSS 502 MiB, and swap remained zero; Termux
stayed alive. Attempt 2 evidence is under `attempt-02/`. Attempt 3 limited only
model loading to CPUs 0–1 at nice 10 and restored full affinity before decode.
The first pass began, then the thermal guard caught an 80.6°C sensor peak and
stopped the child; memory remained at least 4.46 GiB and no swap was used.
Attempt 3 evidence is under `attempt-03/`. The next diagnostic uses two threads
and pins the entire two-position pilot to CPUs 0–1, matching the earlier safe
HTP load procedure. Attempt 4's two-thread CPU0–1 pilot reached graph
reservation and started pass 1, but one CPU reading hit 61.1°C; a direct sample
five seconds later was 44.8°C. It stopped before a pass result, with minimum
available memory 4.38 GiB, peak RSS 1.33 GiB, and no swap. This single sensor
excursion motivated the tested two-consecutive-sample CPU stop; the thermal
log retains every sample. Attempt 4 evidence is under `attempt-04-pilot/`.
Attempt 5 repeated the two-thread pilot and produced two consecutive CPU
readings above 60°C, peaking at 70.9°C; it was stopped before a pass result.
Memory remained at least 4.42 GiB, peak RSS 2.63 GiB, and swap stayed zero.
Attempt 5 evidence is under `attempt-05-pilot/`. Since the validated comparator
uses one CPU thread and a one-token prompt, the next pilot reduces to those
known-safe CPU settings before drawing conclusions about the 2–4 thread sweep.
Attempt 6 used that one-thread/`Hi` pilot but retained batch/ubatch 32. It also
reached pass 1, then sustained readings above 60°C peaked at 70.5°C. MemAvailable
bottomed at 3.40 GiB, RSS peaked at 3.52 GiB, VmSwap reached 558 MiB, and the
child was stopped with no completed pass. Termux remained responsive. Evidence
is under `attempt-06-pilot/`. Attempt 7 matched the validated batch/ubatch 1
setting. It still hit a sustained CPU peak of 65.4°C during pass 1; MemAvailable
bottomed at 3.23 GiB, RSS peaked at 4.26 GiB, VmSwap reached 276 MiB, and no
pass completed. Termux remained responsive. Evidence is under
`attempt-07-pilot/`. The 2–4 thread sweep has not produced a speed datapoint;
with this one-layer HTP placement, even the one-thread pilot crosses the tested
thermal stop. The next experiment must reduce CPU layer work through a fresh,
memory-bounded HTP-placement study before attempting throughput again.
The current stop policy is sustained CPU >70°C (two consecutive one-second
samples), battery >42°C, HTP >55°C, memory below 2 GiB, or battery below 45%.
Attempt 8 used the new 70°C policy for the one-thread, two-position pilot.
Its first pass completed and recorded one timed decode evaluation at 0.214408
tok/s, but a later replay was interrupted after two consecutive CPU samples
over 70°C (peak 77.1°C). MemAvailable stayed above the hard floor (minimum
3.18 GiB); peak RSS was 4.36 GiB and peak VmSwap was 2.35 GiB. The replay
therefore did not verify and this single evaluation is diagnostic only, not a
thread-sweep speed result. Termux remained responsive. Evidence is under
`attempt-08-pilot/`.
Attempt 9 used the updated runner with `n_gpu_layers=4` (three repeating
layers plus output), but retained the one-thread CPU0–1 pilot. It passed
preflight and the 180-second cool hold, entered the first decode evaluation,
then stopped at the sustained CPU guard (70.5°C peak). MemAvailable bottomed
at 2.35 GiB, RSS peaked at 3.73 GiB, and VmSwap at 1.66 GiB. No pass completed
and no speed result was produced. Evidence is under `attempt-09-ngl4-pilot/`.
The pilot now accepts 2–4 threads across the normal CPU affinity. Attempt 10
tested that four-thread candidate at `n_gpu_layers=4` after preflight and the
180-second cool hold. It entered the first decode evaluation but the sustained
CPU guard stopped it before a pass completed; the sampled peak reached 95.0°C.
MemAvailable stayed above the 2 GiB hard floor (minimum 3.48 GiB), RSS peaked
at 2.19 GiB, and swap remained zero. No speed result was produced. Evidence is
under `attempt-10-ngl4-4thread-pilot/`. Given this thermal response, no further
CPU thread sweep should run with the current placement; the next work is to
reduce CPU-side layer execution and reassess the HTP/memory budget first.
Attempt 11 repeated the same four-thread, `n_gpu_layers=4` pilot after the
phone's CPU and battery-saver settings were restored. CPU stayed below the
70°C stop (peak 64.2°C), but HTP reached 55.7°C and triggered its unchanged
hard stop. MemAvailable bottomed at 2.34 GiB, RSS peaked at 3.58 GiB, and
VmSwap at 1.44 GiB. It stopped before a pass completed, so it produced no
speed result. Post-stop readings cooled to CPU 54.9°C and HTP 49.4°C; AC
power remained connected. Evidence is under
`attempt-11-ngl4-4thread-restored-settings/`. This run shifts the observed
thermal bottleneck to HTP under the restored settings; neither the CPU nor
HTP budget currently supports a complete pass at this placement.
For Attempt 12 onward, the user-directed HTP hard stop is 60°C; the CPU stop
remains 70°C for two consecutive one-second samples, and all other limits are
unchanged.
Attempt 12 reran the four-thread, `n_gpu_layers=4` pilot under the 60°C HTP
limit. Pass 1 recorded one timed decode evaluation at 0.158788 tok/s, then
completed the 180-second cooldown. During pass 2, the two-sample CPU guard
stopped the child at a peak of 83.7°C; HTP peaked at 50.6°C. MemAvailable
bottomed at 2.50 GiB, RSS peaked at 3.77 GiB, and VmSwap at 1.06 GiB. Replay
did not complete, so the first-pass speed is diagnostic only and is not a
verified thread result. Termux remained responsive and post-stop readings
cooled to CPU 52.2°C / HTP 43.6°C. Evidence is under
`attempt-12-ngl4-4thread-htp60/`.
Attempt 13 tested two threads at `n_gpu_layers=4` after a passing preflight
and the normal 180-second cool hold. It entered the first decode evaluation,
then the sustained CPU guard stopped the run before a pass completed: CPU
peaked at 78.3°C, while HTP peaked at 47.5°C. MemAvailable bottomed at 3.76
GiB, RSS peaked at 2.03 GiB, and swap remained unused. No speed result was
produced. Termux stayed responsive and the run metadata records the thermal
stop. Evidence is under
`attempt-13-ngl4-2thread-pilot/`.
Attempt 14 was the first `n_gpu_layers=6`, one-thread pilot. It passed preflight
and the 180-second cool hold, but the cooldown supervisor watched the prior
`ngl4` executable path and exited before inference. The parent correctly
stopped the child; no pass or speed result was produced and model memory use
was negligible. The supervisor now follows the selected executable. Evidence
is under `attempt-14-ngl6-1thread-pilot/`.
Attempt 15 used the corrected supervisor and completed pass 1 at 80 slots /
3200 MiB, one thread, `n_gpu_layers=6`; its timed decode evaluation took
8.231 seconds (0.1215 tok/s, diagnostic only). During replay setup, `MemAvailable`
fell to 1.80 GiB, triggering the 2 GiB hard floor before pass 2. Peak RSS was
3.81 GiB and peak VmSwap 2.93 GiB; CPU/HTP remained below their stops. The
replay was incomplete, so this is not a verified speed result. Evidence is
under `attempt-15-ngl6-1thread-pilot/`. For the next pilot, reduce the cache
from 80 to 64 slots (3200 to 2560 MiB) to create more memory headroom.
Attempt 16 made that adjustment and completed both passes at 64 slots / 2560
MiB with one thread and `n_gpu_layers=6`. Pass 1 measured 0.138595 tok/s and
the exact replay measured 0.140194 tok/s; both evaluated the same two
positions, and all logits matched bit-for-bit. CPU peaked at 59.2°C, HTP at
47.9°C, minimum `MemAvailable` was 2.39 GiB, peak RSS 3.24 GiB, and peak
VmSwap 2.20 GiB. The run stayed above the 2 GiB memory floor but with only
about 0.39 GiB of headroom, so treat this as a verified pilot rather than a
sustained setting. Evidence is under
`attempt-16-ngl6-1thread-64slot-pilot/`.
Attempt 17 tested two threads with the same `n_gpu_layers=6` and 64-slot /
2560 MiB cache. It was stopped during model loading before pass 1 when CPU
reached 78.7°C; HTP peaked at 47.5°C and minimum `MemAvailable` was 3.01 GiB.
No speed result was produced. Higher counts under the same normal-affinity
configuration are not warranted under the current CPU guard. Evidence is under
`attempt-17-ngl6-2thread-64slot-pilot/`.
Attempt 18 pinned two threads to CPUs 0–1 with the same placement. Pass 1
completed at 0.08096 tok/s, slower than the verified one-thread result, but
the replay hit the sustained CPU stop at 76.7°C (HTP peaked at 53.3°C) before
verification. Minimum `MemAvailable` was 2.26 GiB, peak RSS 3.09 GiB, and
peak VmSwap 2.53 GiB. No two-thread speed result is verified. Evidence is
under `attempt-18-ngl6-2thread-a520-64slot-pilot/`.

## Unified CPU / OpenCL / HTP build

The unified experiment build is isolated in `.work-unified-source/` and
`.work-unified-build/`. It combines the already-patched NPU source (including
the CPU expert-cache implementation) with the Qualcomm OpenCL-SPHAL patch,
then enables both `GGML_OPENCL` and `GGML_HEXAGON`. The existing `.work/` and
`.work-npu/` builds are retained unchanged. The benchmark runner links to this
single build for every mode, so changing slot count or backend selection does
not require rebuilding llama.cpp.

The unchanged DSP kernel library and its generated FastRPC stubs are copied
from the already-validated NPU build rather than recompiling the SDK's x86
`qaic` generator under emulation. SHA-256 of the copied
`libggml-htp-v75.so` matches the source artifact exactly
(`76397df3c982c39ab7d0919d58139b00b4bd03f3c39e3a56857b091519b337eb`); the
unified host-side Hexagon backend is compiled against those artifacts.

Run `run-benchmark.py --help` for the pilot controls. `--backend cpu` selects
CPU-only, `gpu` selects OpenCL, `npu` selects HTP, and `gpu+npu` requests both
accelerators in one model load. CPU remains available as the normal fallback
for unsupported operations, including CPU expert-cache slots. `--ngl 2|4|6`
controls the tail layer window distributed over the requested accelerator
list. For mixed mode, llama.cpp's default split weights device layer assignment
by reported free memory; the model-load log must be checked to record the
actual split. Always use a fresh `--out-dir`, and retain the default
`cooldown-only` policy: temperatures are logged during inference and the
180–300 second between-pass cooldown remains active; the memory, battery, and
watchdog gates also remain active.

Attempts 19 and 20 were unified GPU+NPU smoke/replay pilots, one thread,
`n_gpu_layers=6`, under the cooldown-only policy. Both completed two passes
with bit-identical logits. Attempt 19 used 64 slots / 2560 MiB: the first
decode was 0.0972 tok/s and its one-token replay was 0.3491 tok/s; pooled
rate was 0.1521 tok/s, minimum `MemAvailable` 2.67 GiB, peak RSS 3.76 GiB,
and peak VmSwap 2.49 GiB. The large cold/replay difference means this is a
verified load smoke, not a reliable speed estimate. Attempt 20 used 80 slots
/ 3200 MiB: decode rates were 0.1348 and 0.1351 tok/s, pooled 0.1349 tok/s;
minimum `MemAvailable` 2.65 GiB, peak RSS 4.19 GiB, and peak VmSwap 2.83 GiB.
This 80-slot combination completed above the 2 GiB floor. Evidence is under
`attempt-19-unified-gpu-npu-ngl6-64slot-pilot/` and
`attempt-20-unified-gpu-npu-ngl6-80slot-pilot/`.

Use `--pilot-tokens 16` for the next speed comparisons; the default two-token
pilot is only a low-cost load/replay smoke. Compare CPU-only, GPU-only,
NPU-only, and GPU+NPU at both cache sizes, one thread first. Each completed
run must replay identical logits. The prior NPU-only 80-slot replay approached
the memory floor, so retain the 2 GiB `MemAvailable` floor and record any
incomplete replay as diagnostic only.

Attempt 21 extended the mixed 64-slot test to 16 generated tokens for a more
stable speed estimate. The tool/Termux session disappeared during pass 1,
step 1/16, after telemetry recorded a transient CPU sensor peak of 84.1°C and
HTP 59.6°C. `MemAvailable` stayed above 3.3 GiB and the memory guard did not
fire. No pass result was written, so this run is incomplete and has no speed
value. The exact termination cause is not established. Its partial telemetry
is retained under `attempt-21-unified-gpu-npu-ngl6-64slot-16token/`.

For long runs that must survive switching away from the Termux screen, launch
the wrapper through a detached session; it acquires a Termux wake lock for the
run and releases it on normal exit. For example:

```sh
mkdir -p reports/2026-10-19/qwen3-30b-a3b-thread-sweep/attempt-22-unified-gpu-npu-ngl6-64slot-16token
nohup setsid -f scripts/run-unified-qwen-pilot.sh gpu+npu 64 16 6 \
  reports/2026-10-19/qwen3-30b-a3b-thread-sweep/attempt-22-unified-gpu-npu-ngl6-64slot-16token \
  > reports/2026-10-19/qwen3-30b-a3b-thread-sweep/attempt-22-unified-gpu-npu-ngl6-64slot-16token/launcher.log 2>&1 < /dev/null
```

Attempt 22 repeated the mixed 64-slot configuration with 16 output tokens
after Attempt 21 was interrupted. It completed both passes with all 16 replay
positions bit-identical. Cold/replay decode rates were 0.2452 and 0.3361
tok/s (15 decode evaluations each; pooled 0.2835 tok/s). Minimum
`MemAvailable` was 3.02 GiB, peak RSS 3.95 GiB, and peak `VmSwap` 2.37 GiB.
The first pass reached a CPU sensor maximum of 58.0°C; temperatures are now
telemetry only during inference, while the initial and inter-pass cooldowns
remain. The detached launcher stayed alive through switching screens and the
run completed normally. Evidence is under
`attempt-22-unified-gpu-npu-ngl6-64slot-16token/`.

The runner has no inference thermal-stop option. Temperatures remain in
telemetry and control the initial and inter-pass cooldown gates; memory,
battery, and watchdog controls also remain active.

Attempt 23 measured the same unified GPU+NPU setup at 80 slots / 3200 MiB
for 16 generated tokens per pass. Both passes completed with all 16 replay
positions identical. Cold and replay decode rates were 0.2477 and 0.2459
tok/s (pooled 0.2468 tok/s). Minimum `MemAvailable` was 2.83 GiB, peak RSS
4.08 GiB, and peak `VmSwap` 2.84 GiB. Compared with Attempt 22's 64-slot
pooled 0.2835 tok/s, the 80-slot cache did not improve throughput in this
mixed configuration; the rate difference is small enough that more repeats
would be needed to treat it as a reliable regression. Evidence is under
`attempt-23-unified-gpu-npu-ngl6-80slot-16token/`.

Attempt 24 isolated OpenCL at 64 slots / 2560 MiB, `n_gpu_layers=6`, with 16
tokens per pass. Both passes completed and all 16 replay positions matched.
Decode rates were 0.2523 and 0.2485 tok/s (pooled 0.2504 tok/s). Minimum
`MemAvailable` was 4.84 GiB, peak RSS 4.82 GiB, and peak `VmSwap` 2.16 GiB.
The log confirms six of 49 layers offloaded to Adreno; the OpenCL model buffer
was 57.01 MiB and several tensors fell back to CPU. This validates a working
OpenCL path, but its speed is close to the mixed GPU+NPU result, so the added
HTP placement has not yet shown a clear gain in this short comparison. See
`attempt-24-unified-gpu-only-ngl6-64slot-16token/`.

Attempt 25 isolated HTP at 64 slots / 2560 MiB for 16 tokens per pass. Both
passes completed with all replay positions identical. Decode rates were
0.2198 and 0.2136 tok/s (pooled 0.2167 tok/s). Minimum `MemAvailable` was
2.52 GiB, peak RSS 3.37 GiB, and peak `VmSwap` 2.48 GiB. This was slower than
GPU-only and mixed mode and left less memory headroom, so HTP-only is not the
current speed or memory winner. Evidence is under
`attempt-25-unified-npu-only-ngl6-64slot-16token/`.

Attempt 26 isolated CPU at 64 slots / 2560 MiB for 16 tokens per pass.
Replay was bit-identical. Decode rates were 0.2572 and 0.2600 tok/s
(pooled 0.2586 tok/s), the fastest standalone backend result in this comparison.
Minimum `MemAvailable` was 4.89 GiB, peak RSS 4.79 GiB, and peak `VmSwap`
2.04 GiB. The highest recorded CPU sensor value was 55.3°C; no inference
temperature stop is configured. Evidence is under
`attempt-26-unified-cpu-only-64slot-16token/`.

Attempt 27 measured CPU-only at 80 slots / 3200 MiB with the same 16-token
plan. Replay was bit-identical; rates were 0.2544 and 0.2580 tok/s (pooled
0.2562 tok/s). Minimum `MemAvailable` was 4.32 GiB, peak RSS 4.99 GiB, and
peak `VmSwap` 2.65 GiB. This is effectively tied with CPU at 64 slots
(0.2586 tok/s pooled), so the 80-slot CPU cache adds no demonstrated speed
benefit. See `attempt-27-unified-cpu-only-80slot-16token/`.

Attempt 28 tested mixed GPU+NPU mode at 64 slots with `n_gpu_layers=2`.
Both passes completed with bit-identical replay. Decode rates were 0.2579
and 0.2574 tok/s (pooled 0.2576 tok/s), with minimum `MemAvailable` 4.51
GiB, peak RSS 4.49 GiB, and peak `VmSwap` 2.09 GiB. The load assigned two
layers to Adreno, with 259.66 MiB of HTP model buffer and 11.40 MiB of OpenCL
model buffer. This is stable but does not beat CPU-only at 64 slots. Evidence
is under `attempt-28-unified-gpu-npu-ngl2-64slot-16token/`.

Attempt 29 tested the same mixed mode at 64 slots with `n_gpu_layers=4`.
Replay was bit-identical, but rates differed sharply: 0.2694 tok/s cold and
0.6668 tok/s replay (pooled 0.3837 tok/s). The run reached CPU 63.9°C and
HTP 53.7°C; minimum `MemAvailable` was 3.74 GiB, peak RSS 4.31 GiB, and peak
`VmSwap` 2.06 GiB. The load assigned four layers to Adreno, with 680.18 MiB
HTP and 22.80 MiB OpenCL model buffers. The cold/replay spread makes this an
outlier pending a repeat; temperatures were recorded and the normal cooldown
ran, with no inference temperature stop. Evidence is under
`attempt-29-unified-gpu-npu-ngl4-64slot-16token/`.

Attempt 30 repeated NGL=4 at 64 slots. Replay was bit-identical and the rates
were 0.2681 and 0.2517 tok/s (pooled 0.2596 tok/s), confirming the previous
0.6668 tok/s replay was an outlier. Minimum `MemAvailable` was 3.77 GiB, peak
RSS 3.92 GiB, and peak `VmSwap` 2.16 GiB. The two NGL=4 passes together are
consistent with CPU-only and NGL=2 around 0.26 tok/s; no clear acceleration
benefit is established yet. See
`attempt-30-unified-gpu-npu-ngl4-64slot-16token-repeat/`.

## Qwen3 CPU thread-count sweep

The Qwen3 CPU-only sweep tests 1–8 threads in one loaded model process at a
fixed 64-slot / 2560 MiB cache. Each pass uses the same generated 16-token
sequence, then the next thread count is replay-verified bit-for-bit. There is
a 180–300 second cooldown between every pass, with memory, battery, telemetry,
and watchdog checks retained. The process uses all allowed CPUs (0–7) for
every thread count after model load, avoiding the earlier one-thread pinning
difference. The runner writes one result row per count, including synchronized
decode tokens/second, and rejects any missing or mismatched pass.

Launch the physical sweep detached so switching screens does not end its
Termux session:

```sh
mkdir -p reports/2026-10-19/qwen3-30b-a3b-thread-sweep/attempt-31-cpu-threads-1-to-8-64slot-16token
nohup setsid -f scripts/run-qwen-cpu-thread-sweep.sh 64 16 \
  reports/2026-10-19/qwen3-30b-a3b-thread-sweep/attempt-31-cpu-threads-1-to-8-64slot-16token \
  > reports/2026-10-19/qwen3-30b-a3b-thread-sweep/attempt-31-cpu-threads-1-to-8-64slot-16token/launcher.log 2>&1 < /dev/null
```

## Accelerator layer-count sweep

The isolated GPU and HTP tests previously covered only six offloaded layers;
the mixed-backend runs covered two and four. They did not establish the best
layer count. The new sweep tests NGL 1–49 separately on GPU and HTP at one CPU
thread, 64 cache slots, and 16 generated positions. Every layer count gets two
passes with bit-identical replay, and the runner keeps the 180-second
cooldown-only policy. The expanded sweep uses a 3 GiB retained-memory floor.
At or beyond NGL 25, three consecutive per-layer throughput declines stop that
backend's sweep; the controller then proceeds to the other backend. One CPU
thread is the default for Qwen3 runs until the user changes it.

Use `scripts/run-qwen-accelerator-layer-sweep.sh --run --backend both` for the
full sweep. Evidence is written to a timestamped directory under
`reports/2026-10-19/qwen3-30b-a3b-thread-sweep/accelerator-layer-sweeps/`.

### Explicit mixed accelerator placement

`run-benchmark.py` now accepts `--backend gpu+npu --ngl 3 --gpu-layers 2
--npu-layers 1`. The runner passes a 2:1 `tensor_split` to llama.cpp and
checks the per-layer device assignment log before marking placement verified.
The one-thread pilot completed at 64 slots / 16 tokens with a pooled decode
rate of 0.439 tok/s and bit-identical replay. The loader assigned 2 layers to
Adreno OpenCL, 1 to HTP, and the remaining 46 layers to CPU. This controls the
accelerator-layer ratio; it does not make the full model work evenly divided
across CPU, GPU, and HTP. See `accelerator-layer-sweeps/attempt-20261020T1022Z-triad-explicit-2gpu-1htp/`.

### Triad profiling by layer

The unified runner now accepts `--cpu-layers` as an explicit check on the
remaining CPU prefix and `--layer-profile` to collect per-layer decode timing.
For example, the current 46 CPU / 2 GPU / 1 HTP placement is described with
`--backend gpu+npu --ngl 3 --gpu-layers 2 --npu-layers 1 --cpu-layers 46`.
The CPU count must equal `49 - --ngl`; the loader assigns that prefix to CPU
and uses the tensor split to distribute the accelerator tail.

Layer profiling uses llama.cpp's graph callback at each `l_out-N` boundary.
This forces synchronization while collecting each interval, so the resulting
per-layer timings are diagnostic and must not be compared with ordinary decode
throughput. The runner also verifies the actual CPU/GPU/HTP assignment log and
requires timing samples for all 49 layers before writing
`layer-profile-summary.json`. Device timing sums are a first estimate of the
work share; they include boundary transfer and synchronization costs, and the
next placement should be selected from those measured results rather than
assuming equal layer counts mean equal work.

The first complete 2 GPU / 1 HTP profile put layers 0–45 on CPU, 46–47 on
OpenCL, and output layer 48 on HTP. Its layer-boundary time sum attributed
99.56% to CPU, 0.10% to OpenCL, and 0.34% to HTP; this is an instrumented
estimate, not a true device-utilization measurement. The 1 GPU / 2 HTP profile
kept 46 CPU layers, moved layer 47 from OpenCL to HTP, and measured diagnostic
decode rates of 0.748 and 0.698 tok/s across its two replay passes (pooled
0.722 tok/s). The 2 GPU / 1 HTP profile recorded 0.481 and 0.401 tok/s
(pooled 0.438 tok/s). Both replayed bit-identically. The 1/2 split's minimum
available memory was 4.15 GiB, with peak process swap of about 1.87 GiB; the
2/1 split had 4.44 GiB minimum available memory. See
`accelerator-layer-sweeps/attempt-20261020Ttriad-layer-profile-full-2gpu-1htp/`
and `accelerator-layer-sweeps/attempt-20261020Ttriad-layer-profile-1gpu-2htp/`.
The earlier incomplete 2/1 capture is retained separately and marked as
missing the output projection timing.

The 1 GPU / 3 HTP profile moved layer 45 to OpenCL and layers 46–48 to HTP.
Its two diagnostic pass rates were 0.337 and 0.699 tok/s (pooled 0.455), so
the extra HTP layer did not show a repeatable gain; minimum available memory
was 3.86 GiB and peak process swap was about 2.15 GiB. The subsequent unprofiled
16-token replay of 1 GPU / 2 HTP measured 0.536 and 0.845 tok/s, pooled
0.656 tok/s, with bit-identical verification. Minimum available memory was
3.93 GiB and peak process swap about 1.88 GiB. This is the best verified
ordinary-throughput triad result so far. See
`accelerator-layer-sweeps/attempt-20261020Ttriad-layer-profile-1gpu-3htp/`
and `accelerator-layer-sweeps/attempt-20261020Ttriad-speed-1gpu-2htp-16tok/`.
The unprofiled 16-token 1 GPU / 3 HTP replay measured 0.456 and 0.406 tok/s
(pooled 0.429 tok/s), confirming the slowdown versus 1 GPU / 2 HTP. Replay was
bit-identical; minimum available memory was 3.49 GiB and peak process swap was
about 2.11 GiB. Evidence is under
`accelerator-layer-sweeps/attempt-20261020Ttriad-speed-1gpu-3htp-16tok/`.
The requested 2-thread / 1 GPU / 2 HTP 16-token attempt was stopped by the
user during pass 1, before a complete timing record; its last logged progress
was step 8/16. Telemetry recorded a 95.0°C CPU sensor peak and 68.5°C HTP peak,
with 3.95 GiB minimum available memory and 2.53 GiB peak process swap. It has
no valid speed or replay result and is retained as an interrupted attempt at
`accelerator-layer-sweeps/attempt-20261020Ttriad-speed-2threads-1gpu-2htp-16tok/`.

### Output-layer placement and CPU-thread follow-up (October 20)

The bounded 16-HTP-layer runner supports independently assigning the output
layer. With 16 CPU / 16 GPU / 16 HTP repeating layers, two runs per placement
gave median first-pass rates of 0.399 tok/s for CPU output (17/16/16), 0.388
tok/s for HTP output (16/16/17), and 0.380 tok/s for GPU output (16/17/16).
CPU output led, but replicate ranges overlap; see
`output-layer-speed-comparison.json`.

The CPU-output placement was then tested at 2–5 CPU threads with the same
32-position prompt and bounded cache. First-pass rates were 0.316, 0.259,
0.212, and 0.226 tok/s, respectively. Each run passed two-pass exact replay.
Two threads led among these new counts, but all were slower than the prior
one-thread median, so one thread remains the measured best. The 2–5 thread
replay rates varied substantially; see `cpu-output-thread-comparison.json` and
the `attempt-20261020-cpu-output-threads-*` evidence directories.

No speculative-decoding test has been run on the S24 Ultra. External reports
for the ordinary Qwen3-0.6B draft paired with Qwen3-30B-A3B show conflicting,
hardware-specific results, including a slowdown; the stronger SpecForge
numbers use a specialized draft and SGLang, not this phone's runner. Details
and source links are in `docs/BOUNDED_EXPERT_CACHE_HANDOFF.md`.
