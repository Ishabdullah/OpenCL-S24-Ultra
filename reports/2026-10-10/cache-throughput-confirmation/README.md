# Throughput follow-up with cooling between every run, October 10, 2026

The screen found 0.835 tok/s with 128 slots/six threads, then 0.984 tok/s
with 64 slots/four threads after a longer cooling break. This follow-up
compares the best observed smaller-cache setting against 128 slots with
four threads, and also checks 128 slots/six threads under the same cooling
routine. Model, prompt, token trajectory and runtime flags match the screen.

The plan is **64/4, 128/4, 128/6, 128/4, 64/4** (slots/CPU threads).
The smaller-cache controls and larger-cache four-thread runs each occur
twice and share a mean chronological position. That gives a smaller-first
and larger-first pair, with one intervening six-thread observation.
Each context starts with an empty cache; all later full-vocabulary logits
must match the first context bit for bit at all 64 evaluated positions.
The plan in `benchmark.jsonl` is authoritative for thread count and budget.

There is a 3–5 minute cooling period before launching the model, and a
3–5 minute cooling pause after every completed pass, alongside the configured rest/context setup. The supervisor pauses only at
untimed between-pass boundaries, preferring battery ≤40°C, CPU ≤55°C and
HTP ≤50°C, then resumes by the five-minute limit. These are practical
cooling goals; starting temperatures need not match exactly. Initial and
between-pass readings and durations are saved. No thermal-policy tests
are performed. Forward timing excludes cooling, model load/setup, sampling
and verification; decode timing includes cache refills and evictions.

Memory floor is 768 MiB available; child wall limit is 2100 seconds after
initial cooling. Only the benchmark child is terminated on a guard trigger.
The expert payload is approximately 2.18 GiB at 64 slots and 4.35 GiB at
128 slots, with MiB budgets 2560 and 5120 respectively. The experimental
cache remains opt-in and the installer defaults stay unchanged.

With the updated cache-enabled benchmark binary:

```sh
python reports/2026-10-10/cache-throughput-confirmation/run-benchmark.py
python scripts/analyze-moe-cache-sweep.py reports/2026-10-10/cache-throughput-confirmation
```

Reruns replace artifacts, so preserve the result directory first. These
measurements cover one prompt/trajectory/device/session. They establish
observed throughput with repeated settings, not universal or sustained
thermal performance. Raw measurements, counters, cooling events, memory
status and provenance are retained for review.

## Completed results

**Fastest observed Qwen3.6 decode: 1.333348 tok/s, 64 slots and four CPU
threads.** This is synchronized forward throughput; sampling is excluded.
The repeated 64/four setting pooled **1.040793 tok/s**, versus **0.886255**
for 128/four: a 17.44% throughput advantage using half the expert payload.
64/four won in both comparison orders. Its two rates span 0.854–1.333;
128/four spans 0.768–1.048. Peak throughput is not a sustained guarantee.

| Pass | Slots | Threads | Prefill seconds | Decode seconds (63 calls) | Decode tok/s |
|---|---:|---:|---:|---:|---:|
| 1 | 64 | 4 | 8.503163 | 73.812064 | 0.853519 |
| 2 | 128 | 4 | 10.340700 | 82.028743 | 0.768023 |
| 3 | 128 | 6 | 16.300547 | 100.407362 | 0.627444 |
| 4 | 128 | 4 | 11.465021 | 60.142466 | 1.047513 |
| 5 | 64 | 4 | 7.078802 | 47.249473 | **1.333348** |

All **256 replayed full-vocabulary logit vectors matched bit for bit**.
Cache counters matched across repeated capacities; budget/identity fallback
counts were zero. The 64-slot selected-set overflow count was 27 per pass;
128 slots had none. The larger cache read less storage during decode
(mean 13.60 GiB versus 19.00 GiB) but generated slower in the repeated test.
Allocated payload is not locked/resident memory; OS reclaim/swap and
scheduling were not isolated, and no per-pass PSS/VmSwap trace was captured.

Initial cooling lasted 180.23 seconds; subsequent pauses were 220.30,
180.25, 180.23 and 180.25 seconds, all reaching the soft goals. Configured
30-second rests and setup add to those pauses. Cooling readings precede
context setup and do not establish identical temperatures at timed starts.
Elapsed benchmark time was 1321.87 seconds (22.03 minutes), excluding initial
cooling. Minimum available RAM was 1.25 GiB, peak process RSS 6.57 GiB.
The child and both monitors exited successfully; no guard stopped the run.

The earlier 32-slot ABBA peak was 0.583982 tok/s: this new peak is 2.28×
that observation, across sessions/settings. The follow-up has no cache-off
control, so it establishes a preferred tested setting rather than a fresh
cache-on/off causal comparison.

`results.csv` retains the per-pass measurements; `best-settings.json`
records the selected configuration. `benchmark.jsonl`, `cache-stats.json`,
`thermal.jsonl`, cooling files, `run.json` and `provenance.json` retain the
underlying evidence. See the [handoff](../../../docs/QWEN36_PERFORMANCE_HANDOFF.md)
for build, launch and continuation instructions.
