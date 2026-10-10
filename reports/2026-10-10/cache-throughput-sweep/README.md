# Qwen3.6 expert-cache and CPU-thread throughput screen, October 10, 2026

The user's goal is faster token generation. This screen measures 32, 64,
96 and 128 cache slots with six CPU threads, then selected four/eight-thread
variants, bracketed by the established 32-slot/six-thread setting.
It follows the [0.584 tok/s best observed decode run](../cache-balanced/README.md).

## Method

The cache/backend implementation is unchanged. The benchmark now accepts
`MOE_BENCH_PLAN=slots:threads:MiB,...` for per-context settings. Without a
plan it retains the earlier four-pass baseline/cache/cache/baseline default.
One model load is shared by fresh sequential contexts; each expert cache
starts empty. All use the same Qwen3.6-35B-A3B UD-IQ1_M model, HTP0 with
forced mmap, batch/microbatch 128, Flash Attention off and no MTP.

The first 32-slot/six-thread context samples the existing binary-search-tree
prompt with seed 20261010, top-k 40, temperature 0.8 and repeat penalty 1.1.
All subsequent contexts replay the identical input tokens and require
bit-identical full-vocabulary logits at every evaluated position. There
are 64 evaluated positions: prefill plus 63 decode calls. This isolates
configuration effects on a shared trajectory, not code quality.

| Pass | Expert slots per weight tensor | CPU threads | Cache budget MiB |
|---:|---:|---:|---:|
| 1 | 32 | 6 | 1280 |
| 2 | 64 | 6 | 2560 |
| 3 | 96 | 6 | 3840 |
| 4 | 128 | 6 | 5120 |
| 5 | 96 | 4 | 3840 |
| 6 | 96 | 8 | 3840 |
| 7 | 64 | 4 | 2560 |
| 8 | 64 | 8 | 2560 |
| 9 | 32 | 6 | 1280 |

Nominal expert payload scales from 1.09 GiB (32) to 4.35 GiB (128);
cache counters report actual allocation and any budget/oversized fallback.
There are fixed 30-second rests, observed temperatures and no temperature
admission waits. The watchdog stops only its child below 768 MiB available
memory or after 2700 seconds. It does not change system settings.

Only synchronized forward calls are timed. Prefill and decode are measured
separately; cache allocations/refills within those calls count. Model load,
context setup, sampling, verification and rests are excluded. Process storage
reads are measured across the same calls. Most configurations get one trial;
this is screening, with chronological/environmental variation. Compare a
promising selected setting against the control in both orders before making
a reliable improvement claim.

## Reproduction

With the existing opt-in CPU cache built:

```sh
clang++ -std=c++17 -O2 -Wall -Wextra scripts/moe-cache-benchmark.cpp \
  -I.work-npu/llama.cpp/include -I.work-npu/llama.cpp/ggml/include \
  -L.work-npu/build/bin -Wl,-rpath,"$PWD/.work-npu/build/bin" \
  -lllama -lggml -lggml-base -o .work-npu/build/bin/moe-cache-benchmark
python reports/2026-10-10/cache-throughput-sweep/run-benchmark.py
python scripts/analyze-moe-cache-sweep.py reports/2026-10-10/cache-throughput-sweep
```

The runner clears inherited thermal admission and cache/backend overrides,
sets the exact plan and required runtime library paths, and records process
memory and exit status. Rerunning replaces artifacts: preserve results first.
The benchmark compiled with `-Wall -Wextra` without warnings. No further
thermal-policy tests were run, as directed by the user.

## Cooling change during this screen

The user requested longer cooling breaks during pass 6. The first six passes
used 30-second rests; before passes 7–9 an attached supervisor adds a cooling pause only
our benchmark at the untimed between-pass boundary. It waits at least 180
seconds and up to 300 seconds, preferring battery ≤40°C, CPU ≤55°C and HTP
≤50°C before resuming. Those readings are goals, with a bounded wait. The
supervisor never pauses a forward call; paused time is excluded from latency.
`cooldown.jsonl` records the pauses and before/after readings. This changed
cooling protocol is a limitation of the screening comparisons. Subsequent
comparisons must use longer breaks from the start.

## Recorded results

All nine passes completed, exit 0, with no watchdog stop. All 512 replayed
full-vocabulary logit vectors were bit-identical. Reference tokens match the
earlier balanced experiment. The controls finished at 0.519 and 0.534 tok/s
(pooled 0.526). The fastest screened setting was 64 slots/four threads at
**0.984 tok/s**, 1.87× the pooled controls in this session. Because cooling
changed mid-screen, the selected setting is followed by a cooled comparison.

| Pass | Slots | CPU threads | Decode tok/s | Prefill s | Expert payload GiB | Decode reads GiB |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 32 | 6 | 0.519 | 6.58 | 1.09 | 32.39 |
| 2 | 64 | 6 | 0.598 | 10.79 | 2.18 | 21.70 |
| 3 | 96 | 6 | 0.602 | 10.63 | 3.26 | 16.41 |
| 4 | 128 | 6 | 0.835 | 19.02 | 4.35 | 13.54 |
| 5 | 96 | 4 | 0.781 | 12.63 | 3.26 | 15.75 |
| 6 | 96 | 8 | 0.405 | 20.92 | 3.26 | 16.36 |
| 7 | 64 | 4 | 0.984 | 7.95 | 2.18 | 19.63 |
| 8 | 64 | 8 | 0.385 | 12.03 | 2.18 | 20.51 |
| 9 | 32 | 6 | 0.534 | 6.72 | 1.09 | 29.76 |

The 128-slot/six-thread setting reached 0.835 tok/s, using 4.35 GiB of
expert payload. Four threads improved the observed 96-slot rate from 0.602
to 0.781 tok/s; eight-thread variants were 0.385–0.405 tok/s. These are
screening observations, with single repetitions for most settings.

Minimum sampled available RAM was **1.93 GiB**, above the 768 MiB
guard; peak sampled RSS was **6.05 GiB**. Every context stayed within
its payload budget, with zero budget or identity fallback. Larger capacities
reduced oversized-prefill fallback from 120 weight operations (32 slots) to
27 (64), 6 (96), and zero (128). Contexts were fresh and cache counters
repeat exactly for equal capacities, including across thread settings.

Wall time was **32.90 minutes**. The three added cooling pauses lasted
250.5, 180.2, 182.2 seconds, in addition to the configured
rest/context setup. All met the soft cooling goals before resuming.

[Repeated follow-up with cooling throughout](../cache-throughput-confirmation/README.md).
