# Controlled CPU Thread Sweep at 80-Slot Cache Capacity

This experiment evaluates CPU thread count (**4 vs 3 vs 2 threads**) with cache capacity held constant at **80 slots** (3200 MiB payload budget) on Qwen3.6-35B-A3B UD-IQ1_M under Hexagon NPU offload (`-dev HTP0 -lm mmap`).

## Motivation

The October 13 capacity sweep proved that 80 slots delivers the highest sustained decode rate (1.184 tok/s pooled) under 4 threads. However, multi-threaded CPU execution generates significant thermal load and can experience thread scheduling contention on the 8-core Snapdragon 8 Gen 3 (which features a 1+5+2 core topology: 1 Cortex-X4 prime core, 5 Cortex-A720 performance cores, and 2 Cortex-A520 efficiency cores). Testing 4 vs 3 vs 2 threads isolates whether lower CPU concurrency reduces thermal throttling or cache-line bounce, potentially improving sustained throughput.

## Experimental Design

- **Execution order**: **4 → 3 → 2 → 2 → 3 → 4** threads (ABCCBA balanced order)
- **Cache setting**: Fixed at **80 slots** (3200 MiB budget) throughout all 6 passes
- **Token length**: Fixed at **64 output tokens** per pass (1 prefill + 63 decode)
- **Cooling routine**: Initial cooling + 5 inter-pass pauses targeting $\ge$180s with ceilings:
  - Battery $\le$ 40°C
  - CPU max $\le$ 55°C
  - HTP max $\le$ 50°C
- **Bit-exact replay gate**: Pass 1 establishes reference logits and teacher tokens; passes 2–6 must match every evaluated position bit-identically.
- **Telemetry**: Full memory sampling (`/proc/meminfo`, `VmRSS`, `VmSwap`, `smaps_rollup`), thermal zones, and storage reads.

## Execution

Dry-run inspection (verifies binaries, models, and outputs plan without running):
```sh
python reports/2026-10-19/thread-sweep-80-slots/run-benchmark.py
```

Run test suite:
```sh
python -m unittest tests/test_thread_sweep_80.py
```

Physical execution (requires attended device and plugged-in charger):
```sh
python reports/2026-10-19/thread-sweep-80-slots/run-benchmark.py --run
```

Post-run analysis:
```sh
python scripts/analyze-moe-cache-sweep.py reports/2026-10-19/thread-sweep-80-slots
```

## Result

The six-pass run completed with return code 0, five of five cooldowns met
their battery/CPU/HTP targets, and all 320 replayed positions matched the
reference. Pooled decode rates were 0.739762 tok/s at 4 threads, 0.720888
tok/s at 3 threads, and 0.673327 tok/s at 2 threads. Four threads narrowly
led the pooled comparison; three threads produced the fastest individual
pass (0.921541 tok/s). The two trials per setting varied substantially, so
the 4-thread selection is provisional and the absolute rates should not be
compared as stable across sessions.

The runner sampled a minimum 2,117,044 KiB MemAvailable and peak process RSS
of 5,671,436 KiB. Full per-pass memory, thermal, cooldown and provenance
records are retained alongside `analysis.json` and `run.json`.
