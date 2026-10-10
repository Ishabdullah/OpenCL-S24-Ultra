# Qwen3.6 performance handoff — October 10, 2026

Current best observed Qwen3.6 decode is **1.333348 tokens/sec**, with
**64 expert-cache slots, four CPU threads and a 2560 MiB cache budget**.
Two runs with cooling throughout pooled **1.040793 tokens/sec**. The run
completed; no benchmark or monitor is left running. This handoff accompanies
the experimental cache, analyzers, numerical checks and saved measurements.

## Start here

Read [the cooled follow-up](../reports/2026-10-10/cache-throughput-confirmation/README.md),
its [best settings](../reports/2026-10-10/cache-throughput-confirmation/best-settings.json),
and [the saved state](../.codex_state.json). The chronological
[triad plan](QWEN3_TRIAD_PLAN.md) retains previous decisions and failures.
`execution_history.log` is local, append-only and gitignored.

The target is Samsung Galaxy S24 Ultra / Snapdragon 8 Gen 3, Termux,
using Qwen3.6-35B-A3B-UD-IQ1_M. Local model:

```text
/data/data/com.termux/files/home/models/qwen3.6-35b-a3b-mtp/Qwen3.6-35B-A3B-UD-IQ1_M.gguf
```

The llama.cpp pin is `e358d59178377be4c58ba567925e05faadbccb57`.
The generated checkout is `.work-npu/llama.cpp`; binaries are in
`.work-npu/build/bin`. Generated builds and model weights are not committed.
Older Qwen3-Coder-Next numbers belong to a different model and must remain
separate from this Qwen3.6 record.

## What changed and what was established

Low-bit IQ1/IQ2 expert `MUL_MAT_ID` operations in this model fall back to
CPU. HTP0 executes supported operations; a layer-offload count alone does
not prove NPU expert execution. The experimental CPU cache copies immutable
canonical expert weights into bounded slots, evicts by recency and pins a
complete selected set during execution. Private IDs and cloned operation
metadata preserve the original graph/tensors. Unsupported layouts,
oversized selected sets and budget/identity failures use original weights.

The cache is opt-in through `GGML_CPU_MOE_CACHE_SLOTS` and
`GGML_CPU_MOE_CACHE_MIB`, with a separate byte budget per CPU backend.
No pruning, MTP or prefetch change was added here. Installer pins/defaults
are unchanged. Allocated expert payload at 64 slots is 2,336,751,616 bytes
(2.18 GiB), versus 4,673,503,232 bytes (4.35 GiB) at 128 slots. These are
allocations, not locked/resident memory; metadata/workspaces add overhead.

| Cooled follow-up setting | Trials | Decode rates, tok/s | Pooled tok/s |
|---|---:|---|---:|
| 64 slots / 4 threads | 2 | 0.853519, **1.333348** | **1.040793** |
| 128 slots / 4 threads | 2 | 0.768023, 1.047513 | 0.886255 |
| 128 slots / 6 threads | 1 | 0.627444 | 0.627444 |

64/four won in both comparison orders, with 17.44% higher pooled throughput
than 128/four. The earlier 32-slot ABBA peak was 0.583982 tok/s; the new peak
is 2.28× that separate observation. Do not promise a sustained 1.333 rate.
All 256 replayed full-vocabulary vectors in the follow-up were bit-identical.

Timing covers synchronized forward calls: one 23-token prefill and 63
single-token decode evaluations per context. Sampling, loading, context
setup, checks and cooling are excluded; cache refills/evictions are included.
The reference samples 64 outputs and subsequent contexts replay the same
inputs. One prompt/trajectory/device/session, remaining OS scheduling/page
cache effects and the observed variation limit generalization. Larger
caches reduced reads but were slower pooled; PSS/VmSwap was not recorded,
so a specific memory-residency cause has not been established.

## Reproduce the implementation and measurements

The existing local CPU backend is already patched and built. On a newly
installed pinned NPU checkout, apply the packaged patch and rebuild:

```sh
python scripts/apply-moe-cache.py
cmake --build .work-npu/build --target ggml-cpu -j2
clang++ -std=c++17 -O2 -Wall -Wextra scripts/moe-cache-benchmark.cpp \
  -I.work-npu/llama.cpp/include -I.work-npu/llama.cpp/ggml/include \
  -L.work-npu/build/bin -Wl,-rpath,"$PWD/.work-npu/build/bin" \
  -lllama -lggml -lggml-base -o .work-npu/build/bin/moe-cache-benchmark
```

The applier checks the pin and manifest SHA256s, is idempotent, and refuses
to overwrite unexpected CPU source. The patch and manifest live in
`patches/experimental-cpu-moe-cache.*`; the reviewable header is
`scripts/moe-slot-cache.h`. Existing installer/backend patches are retained.

Practical generation command with the selected setting:

```sh
GGML_CPU_MOE_CACHE_SLOTS=64 GGML_CPU_MOE_CACHE_MIB=2560 \
  ./scripts/run-npu.sh \
  /data/data/com.termux/files/home/models/qwen3.6-35b-a3b-mtp/Qwen3.6-35B-A3B-UD-IQ1_M.gguf \
  -lm mmap -t 4 -no-cnv -n 64 --seed 20261010 --top-k 40 --temp 0.8 \
  --repeat-penalty 1.1 --repeat-last-n 64 \
  -p 'Write a Python binary search tree implementation with insert, search, delete, and unit tests. Explain the edge cases:'
```

This launcher command applies the measured settings; its end-to-end CLI
throughput has not been separately measured. **Keep `-lm mmap` whenever
HTP0 is selected.** AUTO otherwise disables mmap because Hexagon reports
no mmap support, causing an eager model read and previously observed OOM.
For direct binary invocation, use only the repository's build/bin as
`LD_LIBRARY_PATH`, and `.work-npu/build/ggml/src/ggml-hexagon` as
`ADSP_LIBRARY_PATH`. Mixing Termux's libc++ can cause a missing
`_ZNSt6__ndk113__hash_memoryEPKvm` symbol.

The completed five-pass plan is `64:4:2560,128:4:5120,128:6:5120,128:4:5120,64:4:2560`
(slots:threads:MiB). The runner sets library paths, clears stale cache/backend
and admission variables, records sensors, and supervises only its own child.

```sh
# Reanalyze saved data without running inference:
python scripts/analyze-moe-cache-sweep.py reports/2026-10-10/cache-throughput-confirmation

# Runs inference and OVERWRITES this directory's measurements:
python reports/2026-10-10/cache-throughput-confirmation/run-benchmark.py
```

For new work, copy/adapt the runner into a new report directory before
launching it. Preserve these historical artifacts. Benchmark stdout JSONL
and extracted cache counters are committed; verbose `.log` files are local
and ignored. The analyzer can use committed `cache-stats.json` without the
original stderr log. Cache-off ABBA reproduction is in the linked Phase J
report. Profiling must stay off for throughput comparisons.

## Cooling and user preferences

The user wants faster tokens/sec, practical temperature tolerances, and
**cooling between runs**. Use 180–300-second pauses before the first run and
between every pass. Soft goals are battery ≤40°C, CPU ≤55°C and HTP ≤50°C;
resume by five minutes rather than waiting for an exact temperature match.
The benchmark's 30-second rest/context setup adds to these pauses. Pause
only at untimed boundaries; never interrupt a timed forward call.

The follow-up's initial pause was 180.23 seconds and interpass pauses were
220.30, 180.25, 180.23 and 180.25 seconds. All reached the soft goals.
Cooling snapshots precede context setup; timed starts are not exactly
matched. Thermal-zone sensors and `termux-battery-status` are readable.

Do not repeat strict thermal-policy testing: the user explicitly asked to
skip it and raised future admission bands/ceilings by 3°C. The optional
admission script now uses battery ±3.4°C, CPU/HTP ±7°C and initial ceilings
37/58/48°C. It is not used by the throughput follow-up. Historical Phase K
used the original strict policy and timed out after 900 seconds before the
cached pass; preserve its incomplete outcome rather than claiming a result.

## Evidence and verification

| Phase / report | Result |
|---|---|
| [H: mapping](../reports/2026-10-10/README.md) | Eight 24-token trials; frozen first-six set covers 92.37% of held-out selections but requires 4.47 GiB expert weights. No pruning claim. |
| [I: cache](../reports/2026-10-10/cache/README.md) | Unit and graph checks; Q2_K/IQ2_XXS/IQ1_S/IQ1_M at 1/6 threads; 16 full-model positions bit-exact; patch apply/idempotence/reverse checks passed. |
| [J: ABBA](../reports/2026-10-10/cache-balanced/README.md) | 192 exact replayed vectors; pooled baseline 0.224202 vs cached 0.541009 tok/s; 83.30% mean storage-read reduction. Starting temperatures unmatched. |
| [K: admission](../reports/2026-10-10/cache-thermal-matched/README.md) | Baseline completed; cached pass never admitted. Three original-policy and four analyzer tests passed before later relaxation; relaxed policy tests were not rerun. |
| [L: screen](../reports/2026-10-10/cache-throughput-sweep/README.md) | Nine contexts, 512 exact vectors; best 0.984233. Longer cooling introduced partway through, a comparison limitation. |
| [M: cooled follow-up](../reports/2026-10-10/cache-throughput-confirmation/README.md) | Five contexts, 256 exact vectors, new peak/pooled records above; no budget/identity fallback. |

The follow-up child and monitors exited 0. Runtime was 22.03 minutes,
excluding initial cooling; minimum available RAM 1.25 GiB and peak RSS
6.57 GiB. No watchdog stop occurred. Source/binary hashes are in each report's
`provenance.json`; raw passes, memory summary, cooling and sensor records
are retained. The benchmark compiled with `-Wall -Wextra` without warnings.
No extra inference or thermal-policy tests are needed merely to resume.

Older known issues remain in `.codex_state.json`: CPU-only MTP thread races
and intermittent shared-binary exit-time Scudo aborts. They were deliberately
left outside this throughput work; none stopped this completed follow-up.

## Next work

1. Start from the 64-slot/four-thread setting and keep the cooling routine.
   Screen intermediate 80/96-slot capacities at four threads, then two/three
   threads. Confirm promising changes in both orders with repeated controls.
2. Save each experiment in a new report directory with its exact plan,
   source/binary hashes, raw timings, memory and cooling records. Maintain
   full-logit comparisons and full-model fallback behavior.
3. Capture per-pass PSS/VmSwap if diagnosing why larger caches read less yet
   generate slower. Avoid inferring residency from allocated payload alone.
4. Broaden prompts/output lengths only after confirming a setting. Preserve
   model quality and document sampling-inclusive CLI rates separately from
   synchronized forward rates.

The next agent should inspect git status and the saved state before editing.
The current implementation is opt-in; changing installer defaults or
claiming wider compatibility would require additional validation beyond
this completed work.
