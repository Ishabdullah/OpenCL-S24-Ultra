# Investigation evidence snapshot

Read the [full report](../../OPENCL_PERFORMANCE_ANALYSIS.md) first. This export freezes evidence through October 3 local time / October 4 UTC, when the owner asked to stop, report, commit and push. Tests are paused; there is no promise that the prepared experiment matrix is complete.

## Files

- `data/runs.json`: 467 standard result records with exact configurations, command records, logged allocations, raw repetition timings, numerical/latency metrics, source text/result hashes, selected library paths, temperature ranges, frequency ceilings and recomputed exclusions. Large per-thread traces and original stdout/stderr text stay on the phone.
- `data/llama-bench.csv`: 443 PP/TG rows, including invalid rows with explicit reasons; internal repetition times and standard deviations retained.
- `data/whole-response.csv`: latency and numerical-helper rows. `validation_run=true` means reference/verification work is included and timing is diagnostic.
- `data/allocations.csv`: sequential logged allocation events, not an additive physical-memory total. `record_index` preserves event order.
- `data/model-metadata.*`: model architecture, stored tensor parameters, quantization, file/weight sizes, blocks, attention/KV dimensions and trained context.
- `data/sustained-requests.csv` and `sustained-telemetry.csv`: completed request records and sampled thermal/memory/clock/cgroup data. Safety-censored and scheduling-invalid blocks remain explicit. The JSON summaries distinguish controlled thermal observations from fully valid performance blocks.
- Supporting JSON files: repeated NPU pairs, quality/HMX proof, profiling, source/artifact checkpoints, configuration/ELF evidence, preservation audits and explicit user pause.
- `data/without-standard-result.json`: directories with no standard result. Some are standalone diagnostics, some queued/interrupted; absence is not counted as a completed benchmark.
- `patches/`: actual experimental source diffs with bases and SHA-256 manifest. These are not wired into install.sh.
- `helpers/`: measured sustained helper v1, thermal probe, extended NPU numerical helper and snapshot validity classifier. Latest unvalidated sustained-helper v2 is intentionally not presented as a measured artifact.
- `export_snapshot.py`: offline export of existing evidence only; it does not run benchmarks. It requires the preserved investigation directory and its validation.py.

## Interpretation rules

`passes_current_run_checks` means the classifier detected no exclusion. It does not prove equal thermal state, the best CPU configuration, a matched pair, numerical generality or a positive confirmation. Compare exact settings and power epochs before calculating ratios. Older runs lack some later monitoring; that absence is not proof of stable scheduling.

CSV nested cells contain JSON. Public path strings use `~` and `$PREFIX` placeholders. Expand them deliberately if reconstructing a command; do not paste a JSON argument array into a shell and assume expansion.

The original local records are retained under `~/llama-opencl-build/performance-investigation/runs`. Original stdout/stderr and raw-record SHA-256 hashes are included for provenance. The public export omits proprietary SDK/runtime binaries, vendor libraries, GGUFs and large build artifacts. This is a research evidence snapshot, not a standalone NPU installer or complete accelerator reproduction package.

## Offline regeneration

From a copy of the local preserved investigation:

```sh
python reports/2026-10-03/export_snapshot.py /path/to/performance-investigation
```

This replaces generated snapshot data. Do not regenerate a published historical snapshot with future results; publish a new dated report instead. Original helpers/documentation/export code use the project MIT license; upstream patch context retains the notice in `docs/LLAMA_CPP_LICENSE.txt`.
