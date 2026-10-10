# Longer balanced-order cache experiment, October 10, 2026

This follows the [32-slot prototype correctness gate](../cache/README.md).
It measures the same cache implementation on a longer sampled continuation,
with baseline/cache/cache/baseline order rather than a baseline-first pair.
No cache, installer or backend implementation was changed for this experiment.

## Method

One Qwen3.6-35B-A3B UD-IQ1_M model is loaded on HTP0 with forced mmap.
Four fresh contexts run sequentially, using slot counts 0/32/32/0, with
30 seconds of rest after each completed context. Each cached context
starts with an empty cache, bounded to 32 slots per canonical expert
weight tensor and 1280 MiB total expert payload per CPU backend.

The raw completion prompt asks for a Python binary search tree with
insert/search/delete, tests and edge cases. The first baseline samples a
continuation with seed 20261010, top-k 40, temperature 0.8 and repeat
penalty 1.1 over 64 tokens. The prompt is accepted into sampler history.
The continuation has 64 sampled tokens and no early EOG. Its first 63
output tokens are forward-evaluated, giving 64 evaluated positions including
prefill. The later three passes replay those identical input tokens and
compare every vocabulary logit bit for bit against the first pass.
The sampled text/token IDs are stored in `benchmark.jsonl`; this is a
performance/correctness workload, not a code-quality evaluation.

All contexts use six threads, batch/microbatch 128, Flash Attention off,
no MTP, and the existing NPU computation with CPU fallback for the model's
low-bit expert operations. Model load, context setup, sampling, logit
verification, host bookkeeping and rest periods are excluded from timing.
Only synchronized forward calls are timed, separately for prefill and
63 decode evaluations. Cache allocation/refill work inside those calls
is included. Storage reads are process `/proc/self/io` deltas around the
same calls; they are not direct cache-miss-byte measurements.

This order places the two baselines at positions 1/4 and the two cache
passes at positions 2/3, balancing their mean chronological position.
It provides one baseline-first pair and one cache-first pair. It does
not reset the OS page cache, scheduling or thermal state, and it uses
only two repetitions per mode on one prompt, trajectory, session and device.

## Temperature measurements

Temperature readings are accessible through `/sys/class/thermal`.
The collector discovers sensor names from each zone's `type` file and
reads its `temp` file in millidegrees Celsius. It records battery, CPU,
CPU-subsystem, Hexagon HMX and Hexagon HVX sensors every five seconds.
Sensor names and individual readings are retained in `thermal.jsonl`;
CPU and HTP summaries use the maximum of the respective sampled sensors.
The temperature called `temperature=0.8` in benchmark configuration is
sampler temperature, separate from these physical temperatures.

The initial probe checked the protected `/sys/class/power_supply/battery`
endpoint and incorrectly concluded temperature readings were unavailable.
After the user's correction, all thermal-zone paths were rechecked.
`termux-battery-status` reported 33.9°C, agreeing with the battery thermal
zone's 33900 millidegrees. Logging was attached during the first baseline,
so its initial segment has no continuous sensor trace. Initial untagged
samples belong to that first baseline; later records carry pass/stage tags.
Future runs of the saved runner launch the collector from process start.

Readings are observations, not matched-temperature admission gates.
Five-second sampling and phase tags derived from progress messages may
miss brief peaks and a small part of prefill. Rest/setup readings are
excluded from per-pass summaries. No energy or sustained-thermal benefit
is established by this experiment.

## Reproduction

With the existing experimental cache patch applied and built:

```sh
python scripts/apply-moe-cache.py
clang++ -std=c++17 -O2 -Wall -Wextra scripts/moe-cache-benchmark.cpp \
  -I.work-npu/llama.cpp/include -I.work-npu/llama.cpp/ggml/include \
  -L.work-npu/build/bin -Wl,-rpath,"$PWD/.work-npu/build/bin" \
  -lllama -lggml -lggml-base -o .work-npu/build/bin/moe-cache-benchmark
python reports/2026-10-10/cache-balanced/run-benchmark.py
python scripts/analyze-moe-cache-benchmark.py reports/2026-10-10/cache-balanced
python tests/test_moe_cache_benchmark_analysis.py
```

The saved runner contains the tested local model path, 64-token limit,
400 MiB available-memory floor and 1500-second wall limit. It replaces run
artifacts when rerun. The current recorded run's analysis uses
`--partial-first-thermal` to annotate the mid-pass collector attachment;
a fresh run starts collection immediately and should omit that option.
Temperature monitoring is observational and does not stop the benchmark.

Artifacts: `benchmark.jsonl` (reference tokens/text and per-pass measurements),
`thermal.jsonl` (physical sensor readings), `cache-stats.json` (cache counters),
`analysis.json` (validated comparison), `run.json` (exit status and observed
memory), and `provenance.json`. Verbose runtime logs are generated and
ignored by git. Saved cache counters permit re-analysis without the log.

## Results

All four passes completed with exit code zero and no watchdog stop.
Each later pass matched every one of the 248,320 vocabulary logits at
all 64 evaluated positions. That is 192 full-vocabulary comparisons
against the initial reference, including the final cache-disabled control.
Three benchmark-analysis tests passed, covering paired arithmetic, pooled
throughput, incomplete/failed verification, and excluding rest temperatures.
The C++ benchmark compiled with `-Wall -Wextra` without warnings.

| Pass | Mode | Prefill s | Decode s (63 evaluations) | Total evaluation s | Decode tok/s | Storage GiB |
|---:|---|---:|---:|---:|---:|---:|
| 1 | Baseline | 7.47 | 227.40 | 234.87 | 0.277 | 202.21 |
| 2 | Cache 32 | 6.41 | 107.88 | 114.29 | 0.584 | 36.01 |
| 3 | Cache 32 | 6.15 | 125.02 | 131.17 | 0.504 | 37.20 |
| 4 | Baseline | 6.91 | 334.59 | 341.50 | 0.188 | 236.08 |

Across two repetitions per mode, mean evaluation time was
**288.19 s baseline versus 122.73 s cached**
(2.35× observed evaluation speedup). Pooled decode
throughput was **0.224 versus 0.541 tok/s**
(2.41×). Mean process storage reads per pass were
**219.15 versus 36.61 GiB**, a
**83.30% reduction**.

The baseline-first pair showed 2.06× evaluation speedup and
82.19% fewer reads. The cache-first pair showed
2.60× and 84.24% fewer reads. Caching was faster
in both orders, including when the subsequent baseline had already seen
the same weights/trajectory. This strengthens the earlier single-pair
evidence that the implemented cache helps this workload. It does not
remove thermal, page-cache, scheduling or memory-pressure confounding.

Both fresh cached contexts had identical counters: 7,680 requests,
34,554 unique-expert hits, 25,926 misses, 22,086
evictions, 7.39 GiB copied, and 120 oversized-batch
fallbacks. All 120 prefill expert operations fell back; the hit fraction
for accepted decode expert requests was
57.13%. Each allocated exactly
1,168,375,808 bytes (1.09 GiB). There were no budget/identity fallbacks.
Hit fraction is not storage-read reduction: slot copies and compact
execution change access patterns, and the individual sources of the
larger I/O reduction have not been isolated.

Wall time, including setup/rest, was 15.50 minutes. Minimum
sampled system available memory was 4.04 GiB; peak
sampled process RSS was 6.07 GiB, including model loading.

## Observed thermal conditions and limits

| Pass | Battery first → last sampled °C | Highest sampled CPU °C | Highest sampled HTP °C | Evaluation samples |
|---:|---:|---:|---:|---:|
| 1* | 35.5 → 37.8 | 68.9 | 59.6 | 32 |
| 2 | 37.9 → 39.2 | 84.9 | 64.6 | 22 |
| 3 | 39.0 → 39.5 | 73.6 | 66.6 | 25 |
| 4 | 39.3 → 42.0 | 72.4 | 65.4 | 66 |

*Pass 1 temperature coverage begins mid-pass. Values are sampled
observations, not exact admission temperatures. The entire trace, including
rest/setup, reached 42.2°C battery temperature. The 84.9°C CPU peak is
retained in the raw readings rather than discarded. The first baseline
took 234.87 s, while the final baseline took 341.50 s, a large spread.
Temperature and other environmental conditions changed; these measurements
cannot determine how much of that difference is thermal throttling.

**Decision:** keep the cache opt-in. Bit-exact behavior, bounded payload
and the reduction in I/O are supported by repeated runs. The observed
throughput benefit is specific to this trace/device/session; its precise
size is not thermally controlled. The next measurement should admit
passes at matched battery/CPU temperatures before tuning cache capacity
or adding prefetch/MTP. This experiment does not establish an energy or
sustained-thermal advantage, cross-prompt quality, or a universal speedup.
