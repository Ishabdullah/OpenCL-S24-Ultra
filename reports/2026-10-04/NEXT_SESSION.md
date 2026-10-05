# Next-session handoff

Paused by explicit user request at October 4 local / October 5 UTC. No inference/build/controller/battery monitor remains running, no automatic resume is queued, and the temporary Termux wake lock is released. The user intends to resume tomorrow or later. Resume only after an explicit instruction; do not change the original `~/llama.cpp`.

## What is established

1. The generic OpenCL and native Hexagon baselines are preserved, and the separate public installers remain unchanged.
2. Improved CPU/NPU comparisons on 1.5B/4B have three independent process pairs. Warm-response comparisons also have three independent rounds per mode. Full NPU has a demonstrated benefit on these tested workloads.
3. Two600-second equal-offer blocks per backend, run in reversed orders, reproduce the 1.5B full-NPU response advantage. The large first CPU slowdown does not replicate; no general thermal/energy claim follows.
4. An opt-in checked registration window allows full Mistral7B NPU inference, with 33/33 layers, finite logits and bounded numerical validation. The exact proprietary registration limit is not known.
5. Partial and full 7B NPU-prefill/CPU-decode handoff preserves model/context identity, positions and byte-checked KV without replay. The one-way prototype is correct within its tested architecture scope; it is not an adaptive scheduler.
6. Eight actual canonical CPU memory probes across512/2048/8192/16384 contexts show the smaller Mistral7B overtaking Qwen7B RSS at8K/16K due to larger KV. These are allocation probes, not full-context ingestion tests.
7. The first long7B screen favors full NPU warm latency, but best-CPU variability and independent confirmation remain unresolved. Canonical layout alone and the first paging delay do not explain slow hybrid decode.

## First actions after explicit resume

- Read `README.md`, machine-readable source/configuration evidence, `data/plans/`, and local `~/llama-opencl-build/performance-investigation/EXPERIMENT_LOG.md`. Do not rerun successful quality gates merely to recover context.
- Refresh Termux battery/device/public-thermal/foreground information and owned-process inventory. Restart only the intentional battery monitor and required temporary wake lock. Record explicit resume in `conditions.json`; never clear a battery or phone-use hold without its condition resolving.
- Establish a new measurement epoch and five-minute settling period. Battery<=20% holds demanding work; recovery requires>=25%. Retain the 1536 MiB memory reserve and temperature/background/affinity guards. Charging versus unplugged results remain separate.
- Preserve/check working hashes before any new source experiment. All builds/libraries must be isolated; `LD_LIBRARY_PATH` contains only the selected build/bin, with its own DSP path. Do not globally append `$PREFIX/lib` or resolve libraries from the original installation.
- Use fresh run IDs. The interrupted nine-process plan in `data/plans/` is a record, not a resumable claim that eight missing results exist. One CPU confirmation finished; the next NPU had not started inference.

## Prioritized remaining work

| Priority | Work | Required evidence / decision |
|---|---|---|
| 1 | Explain CPU and hybrid7B decode variability | Per-operation/worker timing, actual CPU utilization/clocks, faults/file residency, retained HTP-session effects. Test one hypothesis at a time; no attribution from correlations alone. |
| 2 | Finish best-practical7B comparison | At least three independently launched, rotated CPU/full NPU/hybrid rounds with the same prompt/context/attention/batch/thread/power settings. Include switch+first decode and warm total; keep initialization/admission separate. |
| 3 | Optimize registration policy experimentally | Foreground profiling of map/unmap/flush costs, safe buffer granularity/window controls. Existing background profile is excluded;2048-vs3200 mapping-budget timing failed scheduling controls and establishes no speed effect. |
| 4 | Complete spatial partial offload | Qualified low/medium/high/all layer counts on representative4B/7B; distinguish CPU/NPU or CPU/OpenCL layer splits from phase handoff. Test high partial 7B gates before throughput. |
| 5 | Qualify Qwen2.5-Coder7B full NPU | Model-specific forced-prefix logits and actual free generation under safe allocation; Mistral support is not generic7B support. |
| 6 | Batch and occupied-context scaling | Safe128/512/2048/8192 prompt workloads with matched generation; actual occupied context, KV allocations and memory pressure. Higher-context allocation success alone is insufficient. |
| 7 | Equal-work GPU/hybrid sustained behavior | Fixed offered requests with backlog/deadline/censoring, order reversal and repeated blocks. Compare useful responses and clocks, not unequal saturated heating; no thermal immunity or higher safe operating temperature claims. |
| 8 | Confirm or reject practical genericGPU benefit | Repeat long-prompt candidates against tunedFA-on CPU. Preserve negative results; no confirmed GPU crossover yet. |
| 9 | Isolated specializedAdreno investigation | Explicit narrow workaround, profile/quality before speed; retain original driver fault uncertainty. Generic installer stays specializedOFF. |
| 10 | Capacity and adaptive scheduler decision | Compare memory-efficient CPU against accelerator physical/shared storage. No additional RAM or14B impossibility demonstrated. Bidirectional/multi-turn restoration and a measured adaptive switching advantage remain unimplemented. |

## Prepared local plans and code

Local investigation root is `~/llama-opencl-build/performance-investigation`; `npu-investigation` holds plans/checkpoints; `runs` retains immutable commands/text/telemetry. Do not copy generated outputs to the public repository.

- `active-mistral-charging-confirm-plan.txt`: interrupted rotated nine-process confirmation; recreate with fresh IDs after inspecting variability.
- `active-mistral-CPU2048-charging-plan.txt`: completed four-point screen. Best initial point t6/coremaskFC/ub256, but its confirmation had much slower decode; not a stable optimum.
- `active-mistral-highpartial-quality-plan.txt`: prepared, unrun high-partial numerical gates.
- `active-mistral-window-retry-quality-plan.txt`: prepared larger-window numerical gates; larger windows are not presumed safe/faster.
- `active-generic-cadence-plan.txt`: prepared, unrun genericOpenCL equal-offer block.
- `source-npu-registration-hybrid` / `build-npu-registration-hybrid`: actual combined one-way plain-Qwen/non-SWA-LLAMA prototype. Preserved NPU base53d539ccc327858498c055909d19468954b90531; combined patchSHA8a4d09ac572c10c9744af4917601ca3abf8f969643809c527cdd837ef1c13e21.

Published experimental patch applicability passes against fresh installer-patched source. Its native build reused unchanged objects/DSP, so a complete fresh build remains necessary before installer integration. Review `patches/README.md` before using the snapshots. No new factory-fresh7B installer claim should be made.

## Publication and preservation

This snapshot includes compact CSV/JSON, source helpers, actual patch diffs, hashes and failed/excluded evidence. Proprietary Qualcomm/vendor/SDK/compiler binaries, models, raw huge kernel traces and generated builds remain local. The main README links the latest report and leaves installation instructions intact.

Final preservation check matched five generic source hashes, eleven generic binaries and two NPU artifacts; preserved NPU source is clean. Original read-only HEAD/status match the prior audit, with optional Git locks disabled. This final check does not repeat the earlier109483-entry metadata scan. See `data/final-preservation-check-oct04.json`.
