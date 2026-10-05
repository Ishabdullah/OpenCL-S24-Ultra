<!-- OCT05_RESUMED_REPORT -->
# October 5 resumed investigation

The authorized three-test run is complete and testing is paused pending another request. Broader characterization remains incomplete. The runs used an API-confirmed unplugged epoch. Historical October 4 checkpoint below remains a record of what was known at that pause; its unresolved hybrid performance finding has new evidence here. Original ~/llama.cpp and preserved generic OpenCL/NPU artifacts remain unchanged.

## Repeated Mistral 7B NPU hybrid advantage

Existing Mistral-7B-Uncensored Q4_K_M, 2048-token synthetic coding prompt, 128 output tokens (127 decode evaluations), actual context2560, F16 KV, FA ON, six CPU threads/core maskFC, batch512/microbatch256, nice10, profilingOFF. Three independent processes per mode in rotated orders CPU/NPU/hybrid, hybrid/CPU/NPU, NPU/hybrid/CPU. Same model, tokenized prompt and settings; CPU uses its optimized repacked weights while the hybrid decodes from mapped canonical GGUF shadows. Full NPU uses opt-in2048MiB registration window, MBUF512MiB, VMEM3200MiB.

| Mode | PP tok/s, mean +/- SD | TG tok/s, mean +/- SD | Warm response seconds, mean +/- SD |
|---|---:|---:|---:|
| CPU | 21.51 +/- 1.44 | 6.26 +/- 0.01 | 116.13 +/- 6.09 |
| full-NPU | 333.65 +/- 2.44 | 3.79 +/- 0.18 | 39.68 +/- 1.58 |
| NPU-prefill-CPU-decode | 335.05 +/- 4.44 | 5.85 +/- 1.44 | 29.17 +/- 6.21 |

Hybrid beat CPU and full NPU in all three matched rounds: 74.9% shorter mean response than CPU and 26.5% shorter than full NPU. All nine processes completed with finite logits, preserved model/cache identities and positions, owned-library paths, and clean shutdown. All128 free-generated output tokens agreed for this repetitive test. This establishes a bounded warm-response benefit, not broad output identity, a generic OpenCL GPU advantage, sustained thermal superiority or lower energy.

Hybrid response varied24.19-36.13seconds, full NPU38.54-41.48, CPU109.09-119.69. Handoff142.68-212.26ms copies320MiB allocated KV with no model weight readback or prompt replay. First CPU decode latency and all initialization/post-load thermal-admission durations are recorded separately in npu-investigation/mistral7-2K-confirmed-oct05.csv/json. Warm response excludes model loading and admission; these results must not be presented as cold application startup.

## Why earlier hybrid decode slowed: measured bounds

Separate per-evaluation diagnostics at p128/g32 and p2048/g64 did not reproduce the persistent ~1tok/s slowdown from October4. At2K, hybrid first CPU decode1.402s recorded718majorfaults,90100minor faults and4366360inputblocks; all later CPU decode evaluations recordedzero major faults/input I/O. Its steady median decode step0.145s versus canonicalCPU0.173 and repackedCPU0.189. These are instrumented diagnostics and not final throughput samples. Initial CPU shadow demand paging explains a measured first-step penalty; it does not explain every earlier persistent slowdown.

Sampled CPU clock caps changed even with stable top-app placement, public thermal NONE and monitored cooling states0. In the confirmation CPU decode, CPU7 ceiling commonly1824MHz; hybrid decode sampled1824-2745.6MHz. Long CPU prefill heated local CPU sensors and reduced caps. Different sensor locations and snapshot clocks cannot identify a universal safe temperature, isolate OEM regulation from all scheduling effects, or prove the causal fraction of latency. Handoff and clock/paging effects must all enter a scheduler decision.

## Memory and remaining work

Atc2560 Mistral allocates320MiB F16 KV. Full NPU retains4510.39MiB converted weights; hybrid releases those after switching and maps4095.05MiB canonical weight shadows. CPU repacking adds4094.04MiB beside the file mapping; observed RSS is recorded but driver buffers may be omitted. All share physical system DDR; no extra accelerator RAM or memory-capacity advantage is established. Earlier actual architectural/context allocation table remains valid, with its empty-context caveat.

Completed foreground registration-cost profiling, the larger-window numerical gate, and partial-offload screens. Still needed: uninstrumented larger-window confirmation; broader prompt/context workloads including8192 when admitted safely; full Qwen7B gate; equal offered-work sustained GPU/hybrid comparisons; best-CPU confirmation for generic OpenCL long-prompt candidates; safe isolated Adreno failure diagnosis. No adaptive bidirectional scheduler or new installer defaults. Raw original logs remain local; machine-readable new evidence resides under performance-investigation/npu-investigation.

## Foreground registration profiling and next experiments

Full Mistral p2048/g32/c2560, same t6FC/FAon/b512ub256, VMEM3200, MBUF512, profiling enabled only for host FastRPC counters. Both2048/3072MiB window profiles completed with finite output, the same output hash, stable sampled foreground/core state and intended runtime library paths. The3072 window also passed32/32 forced-prefix top predictions, finite logits and meanKL0.00000961; a24-layer partial numerical gate passed32/32 and meanKL0.00001169.

| Registration window | New maps / decode step | Mapping-call wall / decode | Eviction-flush wall / decode | Profiled decode tok/s |
|---|---:|---:|---:|---:|
| 2048MiB | 11 | 45.5% | 48.6% | 3.69 |
| 3072MiB | 9 | 40.2% | 54.3% | 4.16 |

Counter intervals use last synchronized cumulative snapshot before phase boundaries. Counts are actual new FastRPC map calls, excluding already-mapped early returns. Mapping/unmapping wall time can overlap pending DSP computation; flush waits include device work. Fractions are not independent pure-overhead slices and must not be added as an attribution. These profiled speeds are diagnostics, not final throughput comparisons. Larger-window speed needs uninstrumented paired repetitions. Existing uninstrumented2048-window confirmation above remains the reported performance baseline.

The first3072 numerical attempt was skipped before model load because its projected4884MiB buffers exceeded4719MiB budget after preserving1536MiB. System MemAvailable later recovered; a fresh-ID retry passed. This is a changing admission condition, not evidence of allocation failure or a backend capacity advantage. Future HTP phase plans explicitly set the device field so the conservative preflight applies as well as the active memory guard. Partial-offload screens now cover requested0/3/6/16/24/99 layers with an optimized repackedCPU control; none is a repeated winner yet.

## Partial NPU offload screen

Mistral p512/g64/c1024, same t6FC/FAon/b512ub256, window3072, profilingOFF. All seven processes passed measured guards, finite outputs and clean shutdown; all64 output tokens matched. Partial cases use canonicalCPU weights to limit duplicate storage; a repacked CPU control is included. Requested layers are not the fraction of compute, memory or power. Single process per point, no replicated optimal-offload claim.

| Requested layers | CPU weight layout | PP tok/s | TG tok/s | Response s | Peak process RSS MiB |
|---|---|---:|---:|---:|---:|
| 0 | canonical | 15.60 | 6.45 | 42.91 | 4372 |
| 3 | canonical | 16.67 | 6.41 | 40.95 | 4184 |
| 6 | canonical | 17.98 | 7.13 | 37.53 | 4183 |
| 16 | canonical | 29.75 | 6.99 | 26.34 | 4182 |
| 24 | canonical | 51.71 | 4.78 | 23.15 | 4183 |
| 99 | canonical | 363.24 | 4.38 | 15.78 | 3892 |
| 0 | repacked | 21.54 | 6.85 | 33.36 | 7763 |

Requested99 produced33/33 actual layer offload. Low/medium partial configurations decoded faster than full NPU in this screen, but full NPU had the shortest total response through its much faster prefill. Partial24 decoded slower than partial6/16; mapping budget and CPU/NPU transitions are candidates, not causal conclusions from this unprofiled screen. CPU explicit worker-placement controls are next, before labeling any baseline best. Raw CSV/JSON: npu-investigation/mistral7-partial-screen-oct05.

## CPU worker placement screen

TermuxOpenMP probe confirmed explicit OMP_PROC_BIND=close / OMP_PLACES={7},{2},{3},{4},{5},{6} assigns six distinct cores. Per-worker active CPUtick deltas and singleton affinity masks were checked in the new screen; base cgroup, available-core, priority, power, thermal, library and memory guards remain. Placement affects CPUparallelgraphs; NPUhost remained2-7 before any such graph. An initial validator incorrectly requiredmain7duringNPUphase, rejected that completed hybrid, and was corrected with fresh IDs. The rejected record remains immutable and excluded from speed claims.

| Mode | CPU OpenMP placement | PP tok/s | TG tok/s | Response s |
|---|---|---:|---:|---:|
| cpu | default | 17.04 | 4.86 | 146.75 |
| cpu | explicit | 20.95 | 4.53 | 126.07 |
| hybrid | explicit | 353.00 | 5.02 | 31.40 |
| full NPU | CPU OpenMP environment (NPU host unbound) | 337.14 | 3.64 | 40.96 |
| cpu | default | 15.94 | 3.72 | 162.98 |

Allfive corrected runs passed finite/state/cleanup and all128outputtokensagree. ExplicitCPU prefill beat both bracketing unboundCPU points, but decode did not beat both, and earlier independently confirmed CPU settings already produced faster totals under earlier thermal conditions. No universal placement improvement or new final fastest configuration follows. A new standalone helper separates context prefill/decode thread counts while preserving model/backend/handoff code; sixprefill/fourdecode nowbeing screened. Original and proven builds unchanged.

## Separate prefill/decode CPU thread counts

The isolated v7 helper sets llama context n_threads_batch=6 while selecting four or six decode threads, with pool capacity=max and the same existing backend/model/handoff. Explicit OpenMP places7,4,5,6,2,3; active worker singleton affinities and thread counts were validated separately by phase. Mistral p2048/g128/c2560/FAon/b512ub256/window2048, profilingOFF, all five valid and all128 output tokens matched.

| Mode | CPU prefill/decode threads | PP tok/s | TG tok/s | Warm response s |
|---|---:|---:|---:|---:|
| CPU | 6/6 | 22.42 | 4.97 | 117.16 |
| CPU | 6/4 | 21.73 | 3.77 | 128.27 |
| NPU-prefill-CPU-decode | 6/4 | 339.12 | 5.67 | 28.75 |
| NPU | 6/4 | 317.76 | 3.40 | 43.83 |
| CPU | 6/4 | 22.50 | 5.22 | 115.60 |

Four-thread CPU decode varied3.77-5.22tok/s versus4.97 on the six-thread opening point. A universal four-thread advantage is not established. Hybrid four-thread decode5.67 and response28.75seconds beats this screen's fullNPU43.83 and CPU115.60-128.27, consistent with the independently repeated hybrid advantage above; this is not an independently repeated optimum for thread placement. Prefill remains six threads in all modes. NPUhost is not pinned by the OpenMP environment before a CPUparallelgraph. Exact plan and CSV/JSON are under npu-investigation/mistral7-dual-threads-20261005T104247Z-schedule.phase-summary.*.

## Charging and scheduling follow-up: incomplete 4B screen

The battery API observed charging at14:59:45UTC, before model evaluation in the next planned unplugged screen. That admission controller was stopped, and fresh charging IDs were used after five-minute settling. Charging results are not pooled with the repeated unplugged Mistral comparison. Qwen3-4B, p512/g128/c1024, six prefill threads and4/5/6 decode threads, FA ON, b512/ub128, repackedCPU:

| CPU decode threads | PP tok/s | TG tok/s | Warm response s |
|---|---:|---:|---:|
| 6 | 59.23 | 8.31 | 24.18 |
| 4 | 57.90 | 10.80 | 20.85 |
| 5 | 55.68 | 11.31 | 20.71 |

These are single screening points, not a confirmed optimum. An earlier four-thread attempt was stopped at an Android cgroup/core transition; its timing is excluded. Full NPU failed the fixed post-load cold-admission requirement after600seconds (CPU<=45/NPU<=43/GPU<=43/battery<=36C held10seconds), exited3 before prompt evaluation, and produced no NPU performance result. The following hybrid/CPU bracket did not run. Admission waiting is an intentional benchmark control, not a mandatory product-startup delay; the failure does not demonstrate inability to infer at those temperatures or a numerical/kernel fault. No limits were raised to obtain a timing.

Source/log A/B falsified an idle-DSP-session hypothesis for CPU mode: driver/registry discovery is eager but session construction is lazy; CPU logs lack the new-session markers that appear in the NPU case. NPU/hybrid initialized sessions still retain SDK MAX/DCVS-off/sleep-disabled requests until lifecycle teardown; idle-vote release remains a source-based engineering lead, not measured energy savings.

Foreground CPU access is required for the pending6-core speed comparison. A proposed deliberately restricted background2-thread/core5-6 diagnostic was skipped before inference because Android restored foreground access; it is not a measured background advantage. Later genericCPU numericalreference ran in background, finite/state/cleanup checks passed, and its timing exclusion was retained. GPU/hybrid forced-prefix numerical checks are separate from the forthcoming8K matched performance screen.


## Generic OpenCL Flash Attention gate and long-prompt comparison

The standalone v7 helper was compiled against the existing generic OpenCL handoff build; no preserved backend or DSP binary was rebuilt. Qwen3-4B Q4_K_M, p128/g32/c1024, FA ON, F16 KV, six CPU threads, b512/ub256: OpenCL and OpenCL-prefill/CPU-decode both passed all32 forced-prefix top predictions with finite logits. Mean KL divergence against CPU was0.00001956 and0.000021899 respectively. Model/cache identity, positions and clean shutdown passed. Numerical qualification does not establish bit-identical logits. These runs had Android core-availability timing exclusions; all their speeds remain excluded, and power epochs are not pooled.

A fresh unplugged p8192/g128/c8704 comparison uses CPU6/core maskFC/FA ON/b512/ub256 and the same generic helper for CPU, OpenCL and OpenCL-prefill/CPU-decode. The first CPU process was aborted at214.8seconds after Android changed top-app to foreground-boost and reset process affinity to0-7. Minimum system MemAvailable was3268.44MiB, above the1536MiB reserve. No complete response or usable timing resulted. This is a scheduling exclusion, not an OpenCL driver fault or a memory-capacity failure. The next fresh-ID plan retains all admission and active guards. The historical8K candidate against a five-thread/FA-OFF CPU remains provisional until this stronger CPU comparison completes.

## Virtual NPU sessions: capability test failed

Pinned upstream documents layer-splitting across virtual sessions on one physical NPU. This could avoid repeated7B weight remapping, so an isolated v8 helper selects HTP0:0 andHTP0:1 with equal layer split and registration-window OFF. It does not alter the known-good backend, DSP kernels or installer defaults. The small1.5B numerical plan failed before model loading completed: session reservation, unsigned-module setup and URI creation reached htp_iface_open, which returned0x200 for `_session=1`. No numerical or performance result exists for two sessions.

A no-model, zero-graph A/B probe then opened session0 alone, session1 alone, and session0 followed by session1 in separate processes. Session0 succeeded; session1 failed with the same0x200 both alone and after session0. All probes completed cleanup. This falsifies an initialization-order explanation and removes model size, tensor loading and GPU/NPU kernel submission from the immediate failure. Local FastRPC headers name0x200 AEE_ERPC, a generic RPC implementation error; it does not identify the exact vendor userspace, kernel, firmware or permission restriction. Two virtual sessions are not demonstrated usable on this phone. Full7B virtual-session benchmarks are withheld. The proven single-session registration-window implementation remains working and unchanged.

Exact small-model plans, artifact hashes, order-probe source/commands and concise errors are recorded in npu-investigation/virtual-session-order-conclusion-oct05.json and the October5 machine-readable evidence. The order probes ran with no model and no graph operations; their subsecond durations are capability diagnostics, not LLM speeds.


## Production scheduler and CPU control caveats

The current one-way handoff replaces the scheduler, rebinds model tensor pointers to CPU shadows, migrates KV storage and, in these measured runs, releases converted accelerator weights. A second switch call explicitly fails. Even retaining accelerator buffers would not by itself restore their tensor bindings, allocator, coherency and backend scheduler. The prototype does not accelerate the next prompt of a resident multi-turn conversation after switching to CPU. Equal offered-work sustained hybrid performance cannot be inferred from a single-response result; a reversible implementation or measured reinitialization cost is required. Existing repeated600second CPU/NPU service tests remain their own evidence. Source-based ownership/lifecycle requirements are in one-way-scheduler-limits-oct05.json.

The CPU control inside the generic OpenCL-enabled helper eagerly probes the vendor device and creates a cl_context, but selects no accelerator model devices, ngl0, op_offloadfalse and offload_kqvfalse. Kernel queues/backend initialization are separate paths; device detection in CPU logs does not demonstrate GPU computation. No effect from an idle OpenCL context on power, clocks or CPU speed is measured. A separate CPU-only-build sensitivity check may be needed if the matched long-prompt screen produces a marginal OpenCL advantage. This is distinct from the earlier falsified idle-DSP-session hypothesis for CPU controls.

A later preservation audit again matched all5 generic source files,11 known-good generic binaries and2 nativeNPU binary hashes. PreservedNPU source remained clean. Original ~/llama.cpp was inspected read-only with optional Git locks disabled; HEAD and its pre-existing Vulkan CMake modification remained the same. This audit did not repeat the earlier entire-tree metadata scan.


### Long-prompt scheduling censoring

The fresh8K CPU retry was also interrupted by a sampled Android top-app to foreground-boost transition, then restricted background access. It produced no complete response or usable CPU timing. Exit-9 followed the harness SIGTERM and8second grace timeout; it is not evidence of spontaneous OOM or a GPU driver crash. Minimum system MemAvailable was2808.18MiB. No cold thresholds, active reserve or affinity checks were relaxed. A new GPU-first order can measure OpenCL/hybrid individually; it still cannot establish crossover without a valid matched CPU baseline. Source-based scheduler conclusions and small numerical checks do not replace that missing measurement.


## Three-test quota completed: generic OpenCL at an occupied 8K prompt

Testing is paused after exactly three authorized inference runs. The 4B candidate was deferred at admission: initial system MemAvailable was 4066.88 MiB; after the 1536 MiB reserve, its conservative 3855.91 MiB estimate exceeded the available budget. The existing 1.5B model fit the 1553.89 MiB estimate. This was a safety-based model selection, not an observed allocation failure or a permanent model-capacity limit. No models were downloaded.

**Method:** Qwen2.5-Coder-1.5B-Instruct Q4_K_M, an 8192-token synthetic coding prompt and 128 generated tokens (127 subsequent decode evaluations), context 8704, six prefill and decode CPU threads, core mask FC, nice 10, batch 512 / microbatch 256, F16 KV, Flash Attention requested ON, profiling OFF, specialized Adreno kernels OFF. One independent process per mode, in OpenCL / hybrid / CPU order, under the same API-confirmed unplugged epoch `api-20261005T180328Z-UNPLUGGED`. All modes used the same v7 helper and owned generic OpenCL build libraries. CPU uses optimized repacked weights; hybrid switches to mapped canonical CPU shadows. The logical workload and options match, but backend weight layouts intentionally differ. No additional repeats were authorized, so standard deviations and a replicated winner are unavailable.

| Mode | Prefill tok/s | Decode tok/s | Time to first token s | Warm response s |
|---|---:|---:|---:|---:|
| CPU | 66.14 | 11.72 | 123.86 | 134.81 |
| OpenCL | 42.01 | 3.53 | 195.02 | 231.07 |
| OpenCL-prefill / CPU-decode | 34.65 | 2.54 | 236.40 | 286.83 |

CPU was fastest in this screen. OpenCL prefill was 36.5% slower and decode 69.9% slower; total response was 71.4% longer than CPU. Hybrid response was 112.8% longer than CPU and 24.1% longer than full OpenCL. All three completed with finite logits, matching 128 generated tokens in this repetitive task, preserved model/cache identity and positions, intended library paths, clean shutdown, and no active scheduling/memory/power guard failures. These are warm resident-model responses, not cold startup, broad output identity, or measurements of the NPU.

### Memory placement and handoff

The model metadata records qwen2 architecture, 1.777088 billion parameters, 28 blocks, 12 attention heads / 2 KV heads, head dimension 128, trained context 32768, GGUF size 1117320768 bytes, and weight bytes 1111370240. Actual benchmark context is 8704.

| Mode | Actual layer offload | CPU mapped / repacked model MiB | OpenCL model MiB | KV placement / MiB | Compute buffers CPU / OpenCL MiB | Peak process RSS / PSS MiB | Minimum system MemAvailable MiB |
|---|---:|---:|---:|---|---:|---:|---:|
| CPU | 0/29 | 877.31 / 934.14 | 0 | CPU / 238.00 | 151.38 / 0 | 2156.38 / 2150.33 | 4161.14 |
| OpenCL | 29/29 | 125.19 / 195.74 | 739.03 | OpenCL / 238.00 | 151.38 / 31.25 | 1355.05 / 1341.93 | 4799.99 |
| Hybrid prefill | 29/29 | 125.19 / 195.74 | 739.03 | OpenCL / 238.00, then CPU | 151.38 / 31.25 before switch | 1820.20 / 1586.44 | 4256.55 |

CPU output buffers are 0.58 MiB in all modes. Hybrid handoff took **342.984 ms**, copied **249561088 bytes (238 MiB)** of KV, copied no weights back from GPU, mapped 774852608 bytes of canonical CPU weight shadows, and released 774924288 bytes of OpenCL weights. CPU compute reservation after switching is 151.38 MiB; repeated reservation log entries are not added as independent simultaneous allocations. The first CPU decode took 156.139 ms, while subsequent average hybrid decode remained slower than the standalone repacked CPU. The 0.343-second switch alone cannot explain that persistent difference. Canonical versus repacked CPU weights, attention fallback/transfers and sampled clock differences remain leads requiring controlled attribution.

RSS/PSS omit some driver allocations. Mapped and repacked logical allocations may overlap physical/cache accounting; global GPU-total snapshots are not process-owned memory. Lower process RSS does not prove extra RAM, lower total physical memory, zero-copy buffers, or that OpenCL can run a model/context that CPU cannot. Both use shared physical DDR.

### Thermal and latency accounting caveats

The fixed cold-admission thresholds and 1536 MiB active reserve passed, but request-start CPU/GPU/battery temperatures were not identical: OpenCL 35.8/32.9/27.4 C, hybrid 44.4/40.8/33.3 C, CPU 42.8/40.4/35.1 C. Public thermal statuses observed were OpenCL NONE; hybrid NONE/LIGHT; CPU NONE/LIGHT/MODERATE. SEVERE was the stop threshold. Sampled CPU7 decode ceilings were 1824 MHz for OpenCL, 1593.6-1939.2 MHz for hybrid, and 1593.6-1939.2 MHz for CPU. These snapshots are not cycle-counter attribution. GPU started colder and CPU still won; one order with accumulating phone heat cannot establish a thermal/energy winner or the exact size of a performance gap under identical thermal conditions.

Initialization was 2.801 / 2.386 / 0.980 seconds for OpenCL / hybrid / CPU. Post-load thermal-admission waits were 10.020 / 20.021 / 104.112 seconds respectively; all are excluded from warm-response totals and retained separately. Benchmark admission waiting is a control, not a product startup requirement.

Exact plan, command/artifact provenance, allocations, CSV/JSON and sampled phase telemetry are under `generic-Qwen1p5-8K-three-20261005T185136Z-schedule.*`; `three-test-outcome-oct05.json` records comparisons and limits. Earlier scheduling-excluded 4B runs remain excluded and cannot be combined with these 1.5B timings.

### Source finding: requested Flash Attention can execute on CPU

Read-only inspection of the actual generic backend identifies a concrete fallback: `ggml-opencl.cpp` classifies Adreno 750 as A7X (lines 295-298), then declines `FLASH_ATTN_EXT` with F32 queries and F16 KV on A7X (lines 9244-9257, repeated later). The comment documents an Adreno 740 compiler E031.41 crash; the guard covers 750 too. Qwen's Flash Attention graph retains F32 queries and uses F16 KV. The measured driver reports compiler E031.45.02.26. Thus requesting FA ON does not mean those attention operations run on GPU, even with all model layers and KV offloaded.

This supports investigating CPU attention fallback and CPU/GPU synchronization/transfers. It does **not** quantify their share of runtime: these unprofiled tests have no per-op timings or measured fallback-transfer byte totals. An isolated mixed-type Flash Attention compile/numerical gate on the current 750 driver is a high-value next experiment, followed by profiling if safe. Removing the guard has not been tested and may crash the driver. The known-good source, guard, generic build and specialized-kernel OFF defaults remain unchanged. No new inference or compilation was performed after completing the three-run quota.


[Compact data and exact plans](data/) | [Helper source](helpers/) | [Remaining work](NEXT_WORK.md) | [Historical report](HISTORICAL_REPORT_THROUGH_OCT04.md) | [October4 experimental source patches](../2026-10-04/patches/README.md)

Paths under npu-investigation in the prose identify original local files; corresponding CSV/JSON and command provenance are curated in data. This checkpoint is research evidence, not a new installer default or complete characterization.
