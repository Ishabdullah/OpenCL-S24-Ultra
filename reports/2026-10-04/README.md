# October 4 performance checkpoint

**Paused at the user's request, October 4 local / October 5 UTC.** Investigation remains incomplete. The unplugged phone reached the 20% battery reserve, so demanding tests were held. After the user connected the charger, separate charging-epoch CPU tuning resumed after recovery/settling. The user subsequently requested a report, commit/push and a pause until another day. All experiment controllers and the battery monitor were stopped. Charging and unplugged results are not pooled. Results below are measurements on one S24 Ultra, not universal backend recommendations. No installer default changes accompany this snapshot. Start next session with [NEXT_SESSION.md](NEXT_SESSION.md).

## Findings

- Native NPU prompt processing repeatedly wins on the tested 1.5B/4B workloads. Full NPU also reduced resident-model response latency in both orders of the 1.5B equal-work experiment below.
- The isolated registration-window implementation enables actual full Mistral 7B NPU inference. It also supports a numerically checked one-way NPU-prefill/CPU-decode handoff without model/context reload or prompt replay.
- Full NPU is the fastest mode in the first 2K-prompt Mistral screen. CPU tuning, independent repetitions and investigation of CPU variability are still needed before recommending that configuration.
- Hybrid correctness is established, but hybrid is not the fastest measured 7B mode. Its CPU decode remains unexpectedly slow. The switch call alone understates the effective transition cost.
- Architecture changes actual RAM requirements enough that the smaller Mistral 7B overtakes Qwen 7B at long allocated contexts.
- No final generic OpenCL crossover, general thermal winner, energy saving or extra memory capacity has been established. No adaptive Codey-OS scheduler has been implemented or validated.

## Method and scope

Use the same GGUF, quantization, synthetic tokenized coding prompt, generation count, actual context, F16 KV, attention, CPU thread/core selection, batch/microbatch, priority and profiling state within each comparison. Tests use existing models only. Each demanding process is serialized; cold admission, battery/power epoch, public thermal state, per-thread placement and system-memory reserve are recorded. Android background/affinity transitions invalidate controlled timings. Failed, interrupted, admission-skipped and excluded records are preserved.

The afternoon epoch is `api-20261004T141923Z-UNPLUGGED`. Morning results have a separate epoch and are not pooled. Model load and post-load cold-admission waiting are outside warm response time and explicitly recorded. These are resident-model, synthetic-workload results, not cold application launch, conversational quality or energy measurements. The first output token comes from prefill; 128 outputs require 127 decode evaluations. Forced-reference numerical checks and traced/profiling runs are diagnostics, not performance samples.

An active benchmark keeps a 1536 MiB system MemAvailable reserve. Battery <=20% prevents new demanding tests; recovery requires >=25% and fresh five-minute settling. Active battery >=43 C or public SEVERE thermal state stops a run. Public thermal NONE does not mean unrestricted clocks. Sensors at different chip locations cannot establish which backend tolerates a higher safe temperature.

Machine-readable data retain settings, argument arrays, artifact hashes where captured, buffer allocations, exclusions and observed RSS/PSS/MemAvailable. Original text logs and full telemetry remain on the phone. `~/` paths in archived records are provenance placeholders, not portable shell commands. See [data](data/) and [source/provenance notes](patches/README.md).

## Repeated CPU/NPU results against improved CPU settings

Morning unplugged llama-bench confirmation used six CPU threads on cores 2-7, Flash Attention ON, batch 512/microbatch 128, F16 KV and profiling OFF. Three independently launched alternating pairs each ran three internal repetitions. Means and between-process sample SD:

| Model, Q4_K_M | CPU PP tok/s | NPU PP tok/s | CPU TG tok/s | NPU TG tok/s |
|---|---:|---:|---:|---:|
| Qwen2.5-Coder 1.5B | 146.36 +/- 7.90 | 1278.35 +/- 13.89 | 28.54 +/- 2.56 | 33.79 +/- 0.12 |
| Qwen3 4B | 46.39 +/- 0.91 | 559.04 +/- 1.69 | 12.95 +/- 0.63 | 14.47 +/- 0.02 |

Standalone llama-bench TG is not TG after the PP workload. Three independent warm-response comparisons at p512/g64/c1024 measured CPU/full NPU/hybrid mean seconds of **5.221/2.368/2.387** for 1.5B and **16.591/5.670/7.750** for 4B. Full NPU and hybrid beat the CPU comparison in all three processes for each model; full NPU is preferable to the tested 4B hybrid. The 1.5B hybrid/full-NPU difference is too small to recommend switching. The historical 4B CPU maximum around 15.56 tok/s occurred while charging and is not a matched unplugged counterexample or comparison.

## Equal offered work and sustained behavior

Qwen 1.5B Q4_K_M, p512/g128/c1024, six threads/core mask FC, FA ON, b512/ub128, profiling OFF. One resident model receives 30 requests every 20 seconds for 600 seconds. Repeat in reverse backend order. Both CPU and NPU completed 60/60 total requests with finite logits, the same output hash, and no missed deadlines.

| Order | CPU mean response s | NPU mean response s | CPU PP / TG tok/s | NPU PP / TG tok/s |
|---|---:|---:|---:|---:|
| CPU then NPU | 10.547 | 4.396 | 105.16 / 22.38 | 1329.48 / 31.68 |
| NPU then CPU | 8.044 | 4.378 | 137.09 / 29.50 | 1329.53 / 31.82 |

NPU response was 58.3% shorter in the first order and 45.6% shorter in the reverse order. There are **two independent blocks per backend**, not 60 independent thermal trials. NPU last/first-third decode ratios were 0.9968 and 0.9988. CPU ratios were 0.7541 and 0.9930: the pronounced first CPU slowdown did not replicate. One CPU request decoded faster than typical NPU. This supports this workload's repeated NPU response advantage, not a general conclusion that CPU inevitably throttles or NPU is immune. Global device current is not attributable inference energy. A matched sustained GPU/hybrid comparison remains pending.

## Full 7B NPU capacity and handoff

Initial full Mistral 7B failed in FastRPC registration despite sufficient physical-memory reserve. Controlled SCALE probes distinguished retained registrations from physical storage: retained 8 x 512 MiB registration failed on the eighth, while preserving allocation addresses/data but explicitly unregistering between operations allowed all eight original and remapped graphs. An opt-in 2048 MiB registration window flushes pending work and removes idle, unpinned weight registrations while preserving storage. It protects current-operation inputs/outputs and rejects unsupported composite/peer sessions. This does not establish a universal exact 4 GiB hardware limit, reduce physical weights, or produce zero-copy inference.

The window passed forced-prefix numerical gates on 1.5B, 4B and Mistral 7B. Mistral had **33/33 layers offloaded, 4510.39 MiB HTP weights, 128 MiB KV at c1024**, finite logits and 32/32 top predictions matching CPU (mean KL 0.00000961). Registration churn remains a performance concern. A background-restricted profile observed 11 new mappings/token and mapping-call wall time about 63.8% of decode, but is excluded from foreground performance attribution; flush/wait also includes asynchronous execution. It is a profiling lead, not a measured foreground pure-overhead percentage.

The combined isolated prototype reuses the working handoff and registration source. Its architecture guard additionally permits plain LLAMA with no sliding-window attention; it still checks for the plain KV-cache type. It does not support arbitrary recurrent/SWA architectures or concurrent contexts. Mistral is LLAMA/GQA, not traditional MHA.

| Numerical handoff gate, p128/g32/c1024 | Partial 16/33 layers | Full 33/33 layers |
|---|---:|---:|
| CPU top predictions / finite logits | 32/32 / yes | 32/32 / yes |
| Mean KL | 0.00001588 | 0.000047481 |
| Allocated KV copied to CPU | 60 MiB | 128 MiB |
| Byte-checked KV base tensors | 30 | 64 |
| Weight readback at switch | 0 | 0 |
| Mapped canonical CPU weight shadows | 1981.77 MiB | 4095.05 MiB |
| Converted NPU weights released | 2179.86 MiB | 4510.39 MiB |

Model/cache identities and positions were preserved, and the new scheduler was CPU-only. CPU-resident KV portions were reused. Mapped GGUF shadows avoid device weight readback, but are a second representation beside converted accelerator weights and are not guaranteed resident. The allocation copies are full allocated KV buffers, not just occupied rows. Quality-helper handoff 88/148 ms includes byte checks and is diagnostic. There is no bit-identical-logit claim.

## First longer 7B response screen

Mistral 7B Q4_K_M: p2048/g128/c2560, t6/core mask FC, FA ON, b512/ub128, F16 KV, nice 10, profiling OFF; registration window 2048 MiB, MBUF 512 MiB, DSP operation mapping budget 3200 MiB. Sequence CPU -> full NPU -> full hybrid -> CPU. Additional canonical CPU control disables CPU repacking with the same workload.

| Mode | PP tok/s | TG tok/s | Warm total s | Init / post-load admission s | Peak RSS MiB | Min MemAvailable MiB |
|---|---:|---:|---:|---:|---:|---:|
| CPU repacked, first bracket | 10.51 | 6.01 | 216.32 | 6.63 / 12.03 | 6711 | 2619 |
| Full NPU | 238.48 | 3.50 | 44.87 | 12.03 / 154.14 | 4184 | 2022 |
| NPU prefill -> CPU decode | 231.74 | 1.00 | 136.24 | 11.80 / 268.24 | 4434 | 1772 |
| CPU repacked, second bracket | 11.02 | 1.31 | 283.04 | 6.36 / 206.22 | 6624 | 2256 |
| CPU canonical, later control | 8.95 | 4.99 | 254.54 | 2.28 / 32.04 | 4535 | 5663 |

All completed with finite logits; the four main modes matched all 128 free-generated tokens. This synthetic prefix/output is repetitive; agreement is not broad task-quality validation. One NPU/hybrid process and two CPU brackets are a **screen**, not independent confirmation. CPU microbatch/thread tuning and replicated winner comparisons remain pending. Load plus cold-admission waiting dominate some start-to-finish runs; these times cannot be discarded when evaluating cold-start product use.

Full hybrid switched in **267.081 ms**, copied **320 MiB KV**, released 4510.39 MiB NPU weights and performed zero weight readback. Its first CPU decode took **2.271 s**, compared with about 0.19-0.21 s for CPU-only. Handoff plus first decode was therefore about **2.538 s**. But the first-token fault penalty cannot explain the entire slow 127-step decode. Canonical CPU later reached 4.99 tok/s versus hybrid's 1.00, so canonical layout alone cannot explain it either.

CPU brackets had the same prompt/output and stable sampled cgroups/placement yet markedly different decode throughput. Their median sampled clock ceilings and clocks were comparable. CPU0 used about 598% process CPU during decode versus CPU1 about 306% and hybrid about 238%. CPU1 had substantial file-residency/page-fault differences; exact causality remains unmeasured. Forced FA ON avoids the AUTO-attention device-selection branch, and the inspected attention builder has no direct NPU placement branch. Context synchronization uses the replaced CPU-only scheduler. These inspections rule out those simple explanations, not all graph/thread/runtime differences. Inactive HTP session lifetime, worker behavior, physical/shared-memory residency and CPU per-operation timing need controlled profiling before another design change.

## Separate charging-epoch CPU tuning and interrupted confirmation

Charging epoch `api-recovered-20261004T213622Z-PLUGGED_AC`, same Mistral p2048/g128/c2560/FA ON/F16/b512/repacked settings. Four fresh CPU-only screens completed with finite logits and stable measured scheduling. No inference or build ran concurrently.

| CPU setting | PP tok/s | TG tok/s | Warm response s |
|---|---:|---:|---:|
| 6 threads / cores 2-7 / ub256 | 24.92 | 6.81 | 101.11 |
| 6 threads / cores 2-7 / ub512 | 22.08 | 6.38 | 112.93 |
| 4 threads / cores 4-7 / ub512 | 11.11 | 2.84 | 229.08 |
| 5 threads / mask BC / ub512 | 18.89 | 6.24 | 129.07 |

Six threads/ub256 was the best **screened** configuration by total latency. Its first new confirmation completed with PP26.82, TG1.66 and total153.26s. The difference from its initial 6.81 TG demonstrates substantial variability even within this charging epoch. Do not present the screen winner as a stable optimum or compare charging CPU directly with unplugged NPU. CPU worker/file-residency/operation profiling is now a priority, alongside repeat measurements.

The planned nine-process rotated CPU/NPU/hybrid confirmation stopped at the user's request: one CPU process completed, and the next NPU point was still in admission, with no new NPU inference or timing result. The remaining eight comparison points were not completed. The preserved plan/state distinguishes completed measurements from unstarted points. [Charging CSV/JSON](data/charging/) and [plan/state](data/plans/) retain this evidence. Resume with fresh IDs and a newly measured power epoch; do not treat the partial schedule as a comparison.

## Architecture and actual memory efficiency

These are **CPU canonical/no-repack allocation probes**, FA ON/F16 KV, b512/ub128, two output tokens at mostly empty allocated contexts. All eight completed. They measure real allocations and sampled process RAM, not full 16K ingestion or quality. RSS/PSS, logical buffer sizes and global available memory are different views and must not be summed as independent allocations.

| Model | Stored parameters | Architecture | Weight / file MiB | Blocks | Attention / KV heads | Trained context |
|---|---:|---|---:|---:|---:|---:|
| Qwen2.5-Coder 7B | 7,615,616,512 | Qwen2 GQA | 4460.45 / 4466.13 | 28 | 28 / 4 | 131072 |
| Mistral 7B Uncensored | 7,241,732,096 | LLAMA GQA | 4165.37 / 4166.07 | 32 | 32 / 8 | 32768 |

Both are Q4_K_M with head dimension 128. F16 KV differs by 2.286x because of layer and KV-head counts.

| Allocated context | Qwen KV MiB | Mistral KV MiB | Qwen peak RSS MiB | Mistral peak RSS MiB | Qwen / Mistral compute MiB |
|---|---:|---:|---:|---:|---:|
| 512 | 28 | 64 | 4553 | 4254 | 77.75 / 29.13 |
| 2048 | 112 | 256 | 4636 | 4446 | 77.75 / 29.50 |
| 8192 | 448 | 1024 | 4972 | 5213 | 77.75 / 31.00 |
| 16384 | 896 | 2048 | 5421 | 6238 | 77.75 / 33.00 |

The smaller Mistral consumes more sampled process RAM at 8K/16K despite smaller weights. Earlier measured Qwen3-4B KV reaches 2304 MiB at 16K, versus Qwen7's896 MiB; the recurrent/full-attention Qwen3.5-4B has512 MiB KV plus50.25 MiB recurrent state. Parameter count is not a capacity metric.

Repacked CPU is faster in some workloads but may maintain a large extra weight representation. The no-repack Mistral control reduced peak RSS by about 2.1 GiB compared with the first repacked CPU2048 run. HTP/RPC weights live in shared system DDR, and processRSS can omit driver-managed storage. Low NPU RSS is not extra NPU RAM. No tested accelerator configuration has demonstrated a model/context combination impossible on memory-efficient CPU-only. The 14B model remains safety-admission constrained; no CPU impossibility claim is made.

## Specialized Adreno correction

Two recent FA-on failures omitted the previously qualified `GGML_OPENCL_Q4K_GEMV_WIDE=0` workaround. They do not establish that the narrowed path fails with FA ON. Actual failure/reset logs remain preserved and the original driver fault remains unisolated.

A fresh explicitly narrowed FA-on forced-prefix gate completed 32/32 matching top predictions, finite logits and meanKL0.00004295. A separate untraced tiny cadence gate completed4/4 requests, but TG was only about 8.2-8.6 tok/s and all four missed its one-second offer deadline. Traced quality timings are excluded. This is not a practical speedup, broad stability proof or reason to enable specialized kernels in the installer.

## Next engineering decisions

1. Tune Mistral CPU threads/microbatch at2048 tokens; repeat CPU/full NPU/hybrid candidates in at least three independent, differently ordered blocks.
2. Profile slow hybrid CPU per-operation/worker utilization and retained HTP-session effects. Include file residency/page faults and first decode in handoff cost. Do not attribute all degradation to canonical weights or first-token paging.
3. Profile registration churn in stable foreground conditions; test buffer granularity/window policy and partial offload separately. The attempted2048-vs3200 DSP mapping-budget timing comparison was interrupted by Android scheduling changes and establishes no speed effect.
4. Qualify Qwen7 full NPU separately, then safe larger prompts/occupied contexts. Mistral success does not prove all 7B architecture support.
5. Complete equal-offer genericGPU/hybrid sustained comparisons; compare useful work and latency, not unequal saturated heating. No higher-temperature exemption or thermal security changes.
6. Consider an adaptive Codey-OS policy only after those controls. The current handoff is one-way and releases accelerator weights; repeated-turn accelerator-prefill restoration is not implemented. Spatial `-ngl` partial offload is a distinct strategy and has no confirmed optimal hybrid point yet.

The installed GPU/NPU baselines remain preserved. Experimental patches are opt-in research artifacts; this report does not claim a clean-room installer for the new7B handoff. No proprietary Qualcomm binary, model, SDK or compiler is included.
