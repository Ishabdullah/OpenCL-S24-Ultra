# Fastest settings with five CPU threads — October 10, 2026

Changed only CPU thread count from four to five: same Qwen3.6-35B-A3B
UD-IQ1_M, 64 cache slots, 2560 MiB budget, HTP0, forced mmap, batch/ubatch
128, Flash Attention off, prompt, sampler/seed and 64 output tokens. Both
contexts started with empty caches, with cooling before launch and between
runs. The backend and benchmark binaries were unchanged.

| Trial | Prefill seconds | Decode seconds (63 calls) | Decode tokens/sec |
|---|---:|---:|---:|
| 1 | 14.080218 | 83.879110 | 0.751081 |
| 2 | 10.957473 | 95.496917 | 0.659707 |

**Pooled five-thread rate: 0.702435 tokens/sec.** Earlier cooled four-thread
runs measured 0.853519 and 1.333348, pooled 1.040793. Five threads were
32.51% slower by that historical pooled comparison. The
four-thread peak remains the best recorded result. These sessions were not
an interleaved four/five-thread control, so the difference does not isolate
thread count from scheduling, page cache or starting conditions.

All **64 replayed full-vocabulary logit vectors matched bit for bit**.
The reference's sampled token trajectory matched the historical four-thread trajectory.
Cache counters matched across both runs: 2,336,751,616 allocated bytes,
45,384 hits, 20,232 misses, 12,552 evictions, 27 oversized selected sets,
and zero budget/identity fallbacks. The payload is allocated, not locked
resident memory. Forward timing includes cache work, excludes sampling,
setup and cooling. This remains one prompt/device with two repetitions.

Initial cooling lasted 180.22 seconds;
between-run cooling lasted 180.24 seconds. Both
reached the soft goals. The configured 30-second rest/context setup is
additional. No strict temperature admission or thermal-policy test was used.
Child and both monitors exited successfully, with no guard stop. Benchmark
wall time was 435.76 seconds excluding initial cooling;
minimum available RAM 2.97 GiB,
peak RSS 6.65 GiB.

`benchmark.jsonl`, `cache-stats.json`, `thermal.jsonl`, cooling files,
`run.json`, `results.csv` and `provenance.json` preserve measurements and hashes.

```sh
# Reanalyze existing artifacts:
python scripts/analyze-moe-cache-sweep.py reports/2026-10-10/cache-64-threads-5
# Rerun with cooling; overwrites this report directory's measurements:
python reports/2026-10-10/cache-64-threads-5/run-benchmark.py
```

[Current best settings and handoff](../../../docs/QWEN36_PERFORMANCE_HANDOFF.md).
