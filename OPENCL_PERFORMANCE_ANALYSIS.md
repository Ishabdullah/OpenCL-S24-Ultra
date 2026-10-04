# S24 Ultra CPU, Adreno OpenCL, Hexagon NPU and hybrid investigation

**Snapshot: October 3, 2026, America/New_York (exported October 4 UTC). Investigation paused at the user's request.** This is a report of evidence collected so far, not a completed characterization or a new accelerator installer release. All experiment controllers and the battery monitor were stopped before preparing this report. Testing requires an explicit instruction to resume.

## What the evidence establishes

- Real Adreno 750 OpenCL inference works in ordinary Termux. The released generic path is reproducible but substantially slower than CPU for the original 1.5B short workload.
- Real Hexagon v75 HTP inference also works without root through a related `sphal`/vendor-runtime approach. Source-built kernels execute HMX for batched matrix operations and HVX for single-token matrix operations. This is native accelerator execution, not emulated inference.
- Repeated matched NPU tests show strong prompt-processing gains: **7.14x for Qwen2.5-Coder-1.5B and 7.83x for Qwen3-4B** at the tested common settings. Generation improves relative to those particular CPU baselines, but historical better CPU configurations are faster than the corresponding NPU generation averages under different conditions.
- Long-prompt generic OpenCL and OpenCL-prefill/CPU-decode runs offer promising response-latency candidates. They have not completed the required confirmation against the best practical CPU configuration.
- A real one-context accelerator-to-CPU handoff works for plain Qwen2/Qwen3 models without replaying the prompt or reloading the model. It needs explicit state/storage migration; ordinary `-ngl` does not implement this strategy automatically.
- Neither low GPU/NPU process RSS nor unified physical memory establishes additional usable RAM. Architecture has a major effect on KV requirements: the tested full-attention 4B has substantially larger KV storage than the tested 7B.
- CPU and NPU sustained-load pilots reached Android's public SEVERE thermal state and were stopped. **There is no established sustained-performance winner, thermal immunity, or measured adaptive-scheduler benefit.** Sustained GPU trials were pending when work paused.

## Evidence, provenance and completeness

The exported [data directory](reports/2026-10-03/data/) contains **467 result records**, **443 individual llama-bench PP/TG rows**, **125 whole-response/quality records**, buffer allocations, individual repetition timings, numerical checks, exclusions, and sustained request/telemetry records. These are counts of records, not 467 independent matched comparisons or successful performance tests. Some records are diagnostics, failed runs, allocation checks, or safety-censored blocks.

| Recorded mode | Result records | Records passing current run checks |
|---|---:|---:|
| llama-bench | 243 | 228 |
| Whole-response / handoff / quality helper | 132 | 123 |
| Allocation / short completion | 40 | 36 |
| Numerical backend operations | 23 | 17 |
| Forced-prefix logits | 17 | 17 |
| Sustained requests | 8 | 4 |
| Operation timing diagnostic | 4 | 4 |

Passing current checks means no detected exclusion in that record. It does not certify equal temperatures, best settings, a matched comparison, broad numerical equivalence, or a completed confirmation. Four sustained records that pass are short helper gates; none of the long blocks completed the intended 15 minutes.

Authoritative records: [runs.json](reports/2026-10-03/data/runs.json), [llama-bench.csv](reports/2026-10-03/data/llama-bench.csv), [whole-response.csv](reports/2026-10-03/data/whole-response.csv), [allocations.csv](reports/2026-10-03/data/allocations.csv), and [inventory.json](reports/2026-10-03/data/inventory.json). CSV nested cells contain JSON arrays/objects. Exact command argument arrays, settings, logged allocations, binary hashes where captured, library paths, raw repetition times, and current exclusions are retained in runs.json. Paths are normalized to `~` and `$PREFIX`; those strings are documentation placeholders, not automatically expandable JSON commands.

Large original stdout/stderr and per-thread traces remain on the phone under `~/llama-opencl-build/performance-investigation/runs`. Exported hashes identify the original records and text logs. Standalone diagnostics outside the standard result schema are summarized in the supporting evidence files and remain locally preserved. Failed/interrupted evidence was retained, not overwritten to produce favorable results.

### Hardware and source checkpoints

Confirmed device: Samsung Galaxy S24 Ultra, SM-S928U / QTI SM8650, Snapdragon 8 Gen 3, Adreno 750, Android 16, aarch64, F-Droid Termux 0.118.3. Other devices are unverified.

| Item | Revision / status |
|---|---|
| Upstream llama.cpp | `https://github.com/ggml-org/llama.cpp.git` |
| Exact upstream pin | `e358d59178377be4c58ba567925e05faadbccb57` |
| Generic known-good source checkpoint | `7ba662d4cae38e0b2474e9bb0024bb4b6b1fc3ad` |
| Native NPU source checkpoint | `53d539ccc327858498c055909d19468954b90531` |
| Installer baseline | v0.1.0 generic OpenCL; specialized kernels OFF |
| NPU SDK / compiler | Hexagon SDK 6.6.0.0 / QuIC tools 19.0.07 |
| Test period | October 2–3 local time, through October 4 UTC |

The checkpoint hashes identify local experimental commits; they are not promises that those commits exist in upstream or in this repository's Git history. Experimental patch snapshots are included under [reports/2026-10-03/patches](reports/2026-10-03/patches/), with bases and hashes documented there. They are research artifacts, not integrated installer options.

`~/llama.cpp` was treated as read-only throughout. The recorded preservation audit checked 109,483 metadata entries, with no changed/missing/added entries, and unchanged HEAD/status. Its pre-existing Vulkan CMake modification is unrelated and was retained. Five generic patched source hashes and eleven known-good build artifact hashes matched the preserved snapshot. Accelerator tests select their own build libraries, never libraries from the original installation.

## Benchmark methodology and limitations

### Workload definitions

`llama-bench` PP is prompt processing; TG is autoregressive generation. Its separate PP/TG tests can allocate different contexts: e.g. NPU PP512 allocated context512, while standalone TG64 allocated context256. These results are not equivalent to decoding after a 512-token prompt. Actual context values appear in the allocation logs. `n_depth` is also recorded; tests at depth0 must not be represented as long occupied-context decoding.

Whole-response helpers instead tokenize a deterministic repeated Python-code prompt, truncate to the requested prompt length, prefill it, and greedily produce a fixed number of tokens. The first generated token comes from the prefill logits, so 64 generated tokens normally require 63 single-token decode steps. Reported decode throughput uses actual decode steps. This synthetic workload exercises real generation but is not a representative conversational corpus.

TTFT means the first token is ready after logits/sampling. It is not measured screen, network or stdout delivery latency. Model load and retained-weight-map setup are recorded separately from response latency. Handoff, first CPU decode, sampling/check overhead and relevant page faults are included in response measurements. Timing from forced-prefix numerical helpers includes reference/verification work and must not be treated as normal inference latency.

### Matching and repetitions

Common-setting comparisons match model file, quantization, prompt/generation counts, context/depth where applicable, thread/core selection, attention, KV types, batch/microbatch, repacking, profiling and process priority. Most short screens have three internal repetitions per fresh process. Initial OpenCL screens used two order-balanced process rounds. The first NPU comparisons use three alternating CPU/NPU process pairs, three repetitions each. Between-process sample SD is reported separately from internal repetition SD.

A practical comparison may independently tune attention, threads, layout and batches for each backend; it must be labeled separately from an attention-matched comparison. Three fresh, differently ordered confirmations against a tuned CPU remain required before promoting an apparent positive OpenCL/hybrid candidate to a practical speed claim. Do not average measurements from different source variants or power epochs merely because their filenames resemble one another.

### Mobile controls

Charging/unplugged transitions have separate epochs and settling windows. User phone activity and screen-off periods were reported; precise intervals are not always known. API-observed plug state and result metadata are authoritative. Historical IDs containing `unplugged` can contain later charging measurements after a queued plan finally ran.

The harness samples temperatures, CPU/GPU clocks and ceilings, Android cgroups/cpusets, worker priority, process memory, system MemAvailable, and loaded llama libraries. It rejects observed affinity failures, lost required cores, live cgroup changes, failed/parsing-incomplete runs and overlapping power transitions. Zombie-task exit cleanup is distinguished from live scheduling changes. One CPU run overlapping compilation of the thermal probe is excluded, and its repeat has a new ID.

Initial screens used legacy nice0. Later tests use nice+10 and preserve CPUs0/1 as UI headroom with explicit masks: t4/mask9c, t5/maskbc, t6/maskfc. These policies must not be silently pooled. The OpenMP one-thread path ignores the threadpool affinity request in this revision; process-level taskset is required for a controlled single-thread test.

Later standard speed admission uses at least30 seconds between demanding processes, battery <=38C, hottest CPU <=65C and GPU <=47C. Diagnostic/allocation gates have separately recorded limits and are not performance evidence. Sustained cold-start admission is stricter: at least300 seconds without heavy work, battery <=36C, CPU <=45.5C, GPU/NPU <=42.5C, public thermal status NONE, forecast headroom <=0.85, with bounded waiting. These limits are our experimental controls, not device specifications.

The phone's governor and background apps are not fully controlled. Even accepted runs may have different frequency ceilings. Battery current fields had inconsistent scales; no trustworthy joules/token or battery-energy comparison was derived. Android thermal policy, CPU/GPU limits, system/vendor files and security settings were not changed.

## CPU versus generic OpenCL

### Original matched proof-of-concept baseline

Qwen2.5-Coder-1.5B-Instruct Q4_K_M, four threads, PP128/TG64, b/ub128, ordinary attention, F16 KV, profiler OFF:

| Backend | PP tok/s | TG tok/s |
|---|---:|---:|
| CPU | 122.74 +/-1.84 | 28.72 +/-4.08 |
| Generic OpenCL | 82.45 +/-0.91 | 9.50 +/-2.24 |

GPU PP is **32.8% slower**; TG is **66.9% slower**. A separate installer clean-room build reproduced working inference and the same direction of performance: CPU119.75/23.92 versus GPU83.90/8.62 tok/s. [Original benchmark details](docs/BENCHMARKS.md) and [clean-room validation](docs/VALIDATION.md) remain available.

This is real offload: `QUALCOMM Adreno(TM) 750`, approximately5542MiB device-visible memory, 29/29 model layers, about739MiB OpenCL weights plus GPU KV/compute buffers, Q4_K/Q6_K GPU matrix operations and20,539 executed GPU kernels in the separate profiling proof. The original46/46 matrix tests passed. A short32-token deterministic output matched CPU; a128-token GPU completion succeeded. Longer forced-prefix comparison had127/128 identical top tokens, finite logits and meanKL approximately0.000242. None of this implies every operation is GPU-executed or bit-identical logits.

### Early model-size screen

These means are from two fresh processes per backend, three internal repetitions each: PP128/TG32, four threads, b/ub128, ordinary attention, F16 KV, generic OpenCL. Early records lack the later power-epoch/core/priority audit. They are exploratory measurements, not a best-CPU crossover proof.

| Model, Q4_K_M | CPU PP | GPU PP | GPU/CPU PP | CPU TG | GPU TG | GPU/CPU TG |
|---|---:|---:|---:|---:|---:|---:|
| Qwen2.5-Coder1.5B |120.32|81.79|0.68x|30.76|8.35|0.27x|
| Qwen3-4B |29.32|28.55|0.97x|8.41|6.80|0.81x|
| Qwen3.5-4B recurrent |34.84|28.69|0.82x|7.09|5.02|0.71x|
| Qwen2.5-Coder7B |16.40|17.58|1.07x|4.44|4.04|0.91x|
| Mistral7B |11.10|15.85|1.43x|4.63|3.75|0.81x|

Some larger-model PP screens favor GPU over that four-thread CPU, but five/six-thread CPU tuning removes a simple practical crossover claim. Same charging-epoch t5/maskbc CPU screens measured PP/TG: 1.5B151.74/34.38, Qwen3-4B45.80/10.79, Qwen3.5-4B38.74/7.62, Qwen7B19.82/6.66, Mistral7B18.47/6.16. These are CPU screens, not simultaneous matched replacements for every row above.

### Partial GPU layer offload

Yes: the investigation includes approximately10%,20%,25%,50%,75% and all-layer offload. `-ngl` is a layer count, not a utilization percentage, and tensor/output-layer placement makes percentages approximate. The following completed generic screens use t4/mask9c, PP128/TG32, b/ub128, FAOFF/F16 KV, charging-01. Each1.5B/4B cell averages two fresh processes with three internal repetitions. They are short-workload screens, not final optimized recommendations.

| Requested ngl / approximate blocks |1.5B PP|1.5B TG|4B PP|4B TG|
|---|---:|---:|---:|---:|
|0 / CPU|122.31|32.64|39.69|10.16|
|3 /~10%1.5B;4 /~10%4B|120.20|28.07|39.34|11.45|
|6 /~20%1.5B;7 /~20%4B|113.62|22.84|39.70|10.86|
|7 /25%1.5B;9 /25%4B|111.74|20.85|37.33|10.21|
|14 /50%1.5B;18 /50%4B|100.56|15.98|28.44|7.84|
|21 /75%1.5B;27 /75%4B|91.96|9.78|32.50|7.59|
|99 /all practical layers|83.88|8.73|30.20|6.44|

1.5B gets worse as offload rises. Low4B offload is a candidate TG improvement against this t4 CPU anchor, but its11.45tok/s does not beat the best recorded4B CPU whole-response decode15.56 under other settings. Partial7B screens span multiple power epochs; complete rows are exported, and cross-epoch means are deliberately not presented as a controlled result. No globally optimal hybrid layer split is established. The NPU offload sweep is prepared but not completed.

### Batch, context and workload shape

Batch screens test combinations including128/128,256/128,256/256,512/512 and1024/1024. Many are single-process screens and some are interrupted. An accepted1.5B PP1024 pair at b/ub1024 measured CPU73.31 versus generic GPU54.14tok/s; merely increasing the batch did not make that path faster. A4B PP1024 b/ub256 screen measured CPU24.89 versus GPU29.86, a candidate needing tuned-CPU repetition. No universal best batch is established.

Large prompt ingestion changes the balance more than short token generation. Ordinary attention at an occupied long context can itself dominate latency. Batches, microbatches, Flash Attention, CPU layout, and thermal state need separate controls; short standalone TG is not a substitute for occupied8192-token decoding.

## Why generic OpenCL is slow

A separate host-instrumented1.5B32-token decode phase measured:

| Quantity | Measurement |
|---|---:|
| Host benchmark wall |3303.96ms|
| GPU kernel launches |18,912, or591/token|
| Event waits |5,664, or177/token|
| CPU graph groups |1,856, or58/token|
| Kernel enqueue API time |150.33ms,4.55% of wall|
| Kernel-argument API time |38.00ms,1.15%|
| CPU fallback graph time |about309ms,9.36%|
| Event-wait time |2627.20ms, includes GPU execution and queue delay|

Wait time is not pure overhead. Separately profiled device events occupied roughly1.4 seconds of a2.5-second device timeline, leaving roughly1.1seconds of gaps. Those gaps include scheduling/CPU boundaries and cannot all be labeled removable dispatch overhead. The earlier20,539-event proof showed Q4_K/Q6_K matrix kernels consumed about90.4% of kernel execution. A7B PP1024 profile is overwhelmingly kernel-execution dominated; not every workload has the same bottleneck. [Profiling evidence](reports/2026-10-03/data/profiling-results.json) retains per-phase categories.

The loader checks quantized weight support using a512-column dummy input. For1.5B, narrow K/V projection weights are rejected by that shape even though their small-N operations are supported. They become CPU_REPACK tensors and create56 extra CPU graph groups per decoded token. A separate N=1 placement probe reduced CPU graph groups1856->64 and waits5664->288 while launches increased18,912->20,704. Throughput improved in that diagnostic sequence. This falsifies a simple explanation based solely on launch count: **CPU/GPU boundaries and shape-dependent placement matter.** It does not make unsupported large-N skinny matrices valid.

OpenMP waiting also consumes CPU resources. `--poll0` alone does not eliminate its default KMP_BLOCKTIME200ms. KMP_BLOCKTIME0 reduced process occupation in some generic cases to approximately0.32–0.58cores but slowed decode; the placement variant responded differently. These are observer/control findings awaiting cold fixed-affinity confirmation, not a default speed or energy recommendation.

## Specialized Adreno kernels: isolated repair

The released generic build remains unchanged with specialized/binary kernels OFF. The isolated specialized path's Q4_K N=1 wide16-subgroup GEMV returned zeros although enqueue/finish succeeded. Q4_K N=128 and Q6_K controls passed. Cache-enabled/disabled A/B tests both preserved the defect; the narrow4-subgroup launch selected by `GGML_OPENCL_Q4K_GEMV_WIDE=0` passed.

This localizes a numerical defect to a particular launch path, but does not fully explain the original Qualcomm submission failure or prove a specific compiler/hardware bug. An observed SIGKILL cannot be attributed to user phone activity or the driver without evidence.

The isolated narrow candidate passed8/8 small and16/16 supported large matrices; eight other shapes remain unsupported. Forced-prefix model gates for1.5B/4B/7B agreed32/32 top tokens with finite logits. The4B gate had maximum logit difference about10.613, which must remain disclosed.

Three charging-epoch1.5B PP128/TG32 screens, t5/maskbc, FAOFF, profiler OFF:

| Backend |PP across fresh processes|TG across fresh processes|
|---|---:|---:|
|CPU|153.73,152.34,108.68|35.86,33.94,33.26|
|Generic OpenCL|84.38,83.58,85.26|10.61,10.50,13.10|
|Specialized narrow|216.64,216.98,219.56|13.81,16.63,17.22|

Specialized PP is promising and repeatedly above the corresponding ordinary CPU screens. It has not completed three fresh confirmations against the best practical CPU configuration, larger-model performance, full context safety, or sustained GPU tests. Generation remains well below these CPU screens. The conservative installer is not changed to this variant.

## Real OpenCL-prefill / CPU-decode handoff

### Architecture and storage

The scheduler can allocate a new evaluation graph without replacing the whole context. Weight buffers and writable KV tensors constrain backend placement, so changing a graph label alone is insufficient. OpenCL tensors use driver buffers and dummy backend base pointers, not CPU-readable weight/KV pointers. Ordinary buffers are created without an assumed zero-copy mapping and quantized weights have GPU-specific conversion/layouts.

The isolated v2 implementation retains CPU-readable canonical GGUF mappings alongside converted accelerator weights. At handoff it synchronizes, invalidates prior graphs, changes former GPU weights to CPU references, copies accelerator KV buffers to CPU storage, preserves cache cells/sequence positions, installs a CPU-only scheduler, and reserves its compute graph in the same model/context. No model reload or prompt replay occurs. Optional GPU weight release reduces overlapping storage; file-backed CPU mappings remain distinct and reclaimable.

Readback-only weights are expensive: diagnostic1.5B switch2640.6ms versus retained-map/drop99.4ms;4B readback6552.8ms versus retained-map/drop262.7ms. These include correctness checks, not ordinary latency. The14 bounded v2 correctness cases passed. Partial offload and keep/drop storage variants were included.

Former GPU tensors become canonical CPU GGUF tensors, not the fastest CPU_REPACK representation. This can sacrifice CPU decode speed. A CPU-resident/GPU-compute prototype also passed1.5B/4B/7B checks with zero KV handoff bytes, but still transfers tensors during prefill and has not demonstrated a latency win.

### Total response latency

Controlled short-request pair: Qwen3-4B,128prompt+64generated,ctx512,t5/maskbc,b512/ub256,FAOFF/F16KV,nice10,unplugged-04, two reversed-order fresh processes:

|Strategy|Mean response|
|---|---:|
|CPU prefill + CPU decode|7.20068s|
|Generic OpenCL prefill + OpenCL decode|14.09135s|
|Generic OpenCL prefill + CPU decode|9.30604s|

Hybrid is **29.2% slower than CPU** here. Actual handoff260.5–285.8ms copies72MiB of allocated KV with no weight readback.

Long-request candidates,8192prompt+128generated,ctx8704,t5/maskbc,b512/ub256,FAOFF/F16KV,nice10:

|Model / epoch|CPU total|GPU total|Hybrid total|Handoff / KV copied|
|---|---:|---:|---:|---|
|Qwen3-4B / unplugged-04|634.13s, one completed ordinary CPU round|412.37,411.56s|391.13,404.13s|0.976–0.995s /1224MiB|
|Qwen2.5-Coder7B / API charging epoch08:48UTC|865.57,655.34s|562.89,562.97s|558.27,544.46s|0.705–0.729s /476MiB|

These show plausible long-prompt GPU savings greater than switch cost, but the CPU variation and unfinished best-CPU confirmations prevent a final practical ranking. A4B charging Flash-Attention8k result must not be paired with unplugged GPU data. Canonical CPU+FA controls reduce CPU RSS but can slow prefill. Long-request hybrid decode also varies; a one-second switch is not its only cost.

The prototype is one-way, supports plain Qwen2/Qwen3 KV only, and mutates weight references for one live context. Recurrent-state handoff and concurrent contexts are not supported. A production Codey-OS scheduler is not implemented or validated.

## Hexagon NPU access and native inference

### Access and build

Plain vendor CDSP loading from the Termux namespace fails; loading `/vendor/lib64/libcdsprpc.so` through `sphal` succeeds and resolves17 needed runtime functions. Direct device open is denied, but the vendor HIDL FastRPC broker supplies the permitted device path/session. Unsigned user-PD initialization works without root. Capability queries report architecture0x8c75, four128-byte HVX resources and8MiB VTCM. Successful HMX kernels, not a hardcoded SDK capability field, establish tensor-unit use.

An official calculator control succeeded in three sessions. Our own QAIC-generated/source-built arithmetic module passed12/12 checks. The native ARM64 llama host and source-built v75 DSP library subsequently passed8 small plus24 larger quantized matrix checks. Profile traces classified four N128 operations as `hmx-tiled` and four N1 operations as `hvx-tiled`, with12 HMX_COMP start/stop traces. Full1.5B29/29 and4B37/37 layer offload completed32-token generation; recurrent Qwen3.5-4B also completed a full-offload smoke test.

QEMU was required for x86 SDK build tools because the installed Termux compiler lacked the necessary HMX compilation support. **Inference executes natively on ARM64/Hexagon, never inside QEMU.** SDK6.6.0.0/QuIC19.0.07 uses v75/HVX/HMX/Release/LTO builds. DSP build parallelism was limited to two, and an unnecessary repeated install-time LTO rebuild was removed. The only additional package recorded for this path was `qemu-user-x86-64`1:11.0.3; the SDK/build runtime stays in experimental storage.

Source changes are limited to the Hexagon backend CMake, vendor-driver loader and DSP toolchain adaptation: optional sphal loading, corrected64-bit remote-handle close typedef, selected v75 build, QAIC/tool wrappers and bounded nested build/install. No DSP compute kernel or global Android policy was changed. SDK/component licensing is separate from the MIT project; SDK, vendor libraries, proprietary compiler binaries and models are not redistributed here.

### Repeated matched NPU measurements

PP512/TG64, t6/maskFC and process maskFC, b512/ub128, ordinary attention/F16KV, profile0, nice10, same API-observed unplugged epoch. Three alternating process pairs per model, three internal repetitions each. Values are mean +/- between-process sample SD.

|Model|CPU PP|NPU PP|NPU/CPU PP|CPU TG|NPU TG|NPU/CPU TG|
|---|---:|---:|---:|---:|---:|---:|
|Qwen2.5-Coder1.5B|117.54 +/-9.32|839.18 +/-5.60|7.14x|22.41 +/-1.29|32.97 +/-0.09|1.47x|
|Qwen3-4B|30.10 +/-1.96|235.66 +/-2.54|7.83x|7.30 +/-0.96|12.94 +/-0.08|1.77x|

All current power/scheduling/priority checks pass. CPU frequency ceilings still varied. These are strong, repeated gains **at the tested common settings**, especially PP. They are not proof against the best practical CPU, an energy result, or a realistic whole-response comparison. [1.5B trial data](reports/2026-10-03/data/first-matched-results.json), [4B trial data](reports/2026-10-03/data/fourb-matched-results.json).

Highest saved valid4B CPU whole-response decode is **15.555508tok/s**, repeated at15.387650, p128/g64,ctx512,t5/maskbc,b512/ub256,FAON/F16KV,nice10, **charging**. Its ID contains `unplugged-04`, but actual metadata records charging. Prior1.5B CPU screens reach37.19tok/s generation and184.69tok/s PP with differing attention/thread settings. These numbers cannot be substituted into the current NPU pair as if measured under identical conditions. The four latest unplugged1.5B t4/t5 CPU screens at b512/ub256 measured TG21.04–22.84; t6 and larger-model tuning remain pending.

### 7B and numerical quality

There is **no full-offload7B NPU benchmark**. Its first attempt was admission-skipped: estimated5112MiB buffers versus4767MiB available after1536MiB reserve. This is a safety decision, not an observed allocator/driver failure or proof that7B cannot work.

A canonical/no-CPU-repack partial7B smoke test completed32 generated tokens with15/29 layers offloaded,2533.11MiB HTP weights,2146.07MiB CPU weights,14MiB CPU plus14MiB HTP KV. It reported **4.04tok/s**, a verbose unpaired smoke measurement. It is not a repeated7B NPU throughput result. Minimum system MemAvailable was about1708MiB, close to the reserve.

Forced-prefix comparisons use128prompt tokens and32 CPU-selected subsequent tokens, finite-logit checks,ctx1024,t4,b/ub128,F16KV. Different attention/offload gates are listed separately:

|Model / settings|Top-token agreement|RMSE|Max absolute logit difference|Mean KL|
|---|---:|---:|---:|---:|
|1.5B full NPU, FAOFF|32/32|0.06622322|0.52972949|0.00003678|
|4B full NPU, FAOFF|32/32|0.41612397|9.54219949|0.00002693|
|1.5B full NPU, FAON|32/32|0.06854350|0.45895237|0.00001279|
|4B full NPU, FAON|32/32|0.45826948|10.42526603|0.00001168|
|Qwen3.5 recurrent4B, FAOFF|32/32|0.47011936|9.02431917|0.00017560|
|7B partial15 layers, no CPU repack|32/32|0.05038818|0.32320881|0.00000159|

All reported logits are finite. Short top-token agreement does not imply bit-identical logits, long-output equivalence or general model accuracy. A separate4B full-NPU handoff-helper gate had max difference14.83285; its full metrics are also exported. Quantized packing changes scale representation/rounding, so nonzero differences are expected and must be characterized rather than hidden.

### NPU-prefill / CPU-decode prototype

The same isolated handoff architecture works with HTP. The exact source-built DSP library is reused unchanged. Both1.5B and4B hybrid gates agree32/32 CPU-selected top tokens, remain finite and preserve context/cache/position identities. Diagnostic handoff:1.5B64.553ms with14MiB KV;4B111.768ms with72MiB KV, no weight readback. These include verification overhead and are **not uninstrumented response benchmarks**.

HTP uses host-accessible RPC/DMA shared storage, but scheduler/coherency constraints still matter. The prototype copies KV rather than assuming direct reuse is safe. It retains canonical CPU shadows alongside packed HTP weights during prefill and drops packed weights at handoff. Canonical CPU decode is slower than the best repacked CPU representation in some gates. Whether repacking, aliasing or selective KV migration can improve the tradeoff remains unmeasured. No7B NPU handoff latency or NPU hybrid total-response win is established.

## Architecture and memory efficiency

### Actual model metadata

All models were already present; no duplicate model downloads were needed. Nomic embedding is excluded from generation comparisons. Counts below are stored GGUF tensor element counts, which may differ from marketed parameter labels because of stored/tied representation. File sizes are GiB; KV/state allocations use MiB. Detailed metadata and exact bytes are in [model-metadata.csv](reports/2026-10-03/data/model-metadata.csv) and [.json](reports/2026-10-03/data/model-metadata.json).

|Model label|Stored parameters B|Architecture|GGUF GiB|Blocks|Heads / KV heads|Effective K/V dimensions|Trained context|
|---|---:|---|---:|---:|---|---|---:|
|Qwen2.5-0.5B|0.494|qwen2|0.370|24|14 /2|64 /64|32768|
|Qwen2.5-Coder1.5B|1.777|qwen2|1.041|28|12 /2|128 /128|32768|
|Qwen3-4B|4.022|qwen3|2.326|36|32 /8|128 /128|262144|
|Qwen3.5-4B|4.206|qwen35 recurrent + full attention|2.553|32,8 full-attention|16 /4|256 /256|262144|
|Qwen2.5-Coder7B|7.616|qwen2|4.361|28|28 /4|128 /128|131072|
|Mistral7B|7.242|llama|4.069|32|32 /8|128 /128|32768|
|Qwen2.5-Coder14B|14.770|qwen2|8.371|48|40 /8|128 /128|131072|

Quantization is Q4_K_M for these files. Effective dimensions come from explicit metadata where present and documented fallback interpretation elsewhere; do not infer every model's KV dimension from embedding width alone. Trained maximum context is metadata, not a validated phone capacity. The14B large-file case was not forced through an unsafe allocation.

### Measured KV and recurrent-state allocation

Actual llama buffer logs, F16 KV, allocated context followed by a tiny completion:

|Model|ctx512 KV|ctx2048 KV|ctx8192 KV|ctx16384 KV|Additional recurrent state|
|---|---:|---:|---:|---:|---:|
|Qwen1.5B|14|56|224|448|0|
|Qwen3-4B|72|288|1152|2304|0|
|Qwen3.5-4B|16|64|256|512|50.25|
|Qwen2.5-Coder7B|28|112|448|896|0|

Values are MiB; CPU and OpenCL logs show equal logical KV capacities, placed on their selected backend. Some individual runs have exclusions listed in the data; allocation observations are not certified performance comparisons. Mistral ctx512 GPU logged64MiB KV. Its larger-context series is pending;2048MiB at16k is a theoretical F16 projection, not a measured result.

Qwen3.5 has slightly more stored weights than Qwen3, but at16k its KV+recurrent state is562.25MiB versus2304MiB. The7B also needs less KV than the full-attention4B. Architecture, GQA ratio, key/value dimensions and recurrent layers dominate this difference. Empty allocated-context probes establish buffer capacity/allocation, not successful ingestion of a full16k prompt or peak occupied-context throughput.

### Where memory resides

CPU file mappings, CPU repacked anonymous weights, OpenCL converted buffers, HTP rpcmem/DMA buffers, KV, output and compute allocations are recorded separately. A log can describe multiple sequential contexts or backend phases: **do not sum every buffer line as simultaneous physical peak usage.** Logical model buffer sizes include layout/alignment and can overlap mapped pages or measurement views.

Atctx512, full NPU1.5B logs1023.48MiB HTP weights,125.19MiB CPU weights,14MiB HTP KV and74.94MiB HTP compute;4B logs2609.07MiB HTP weights,304.28MiB CPU weights,72MiB KV and75.44MiB compute. NPU process RSS about245/377MiB excludes much shared-DMA storage. Q4_K packing expands144->160 bytes per256 weights, Q6_K210->224 before alignment; VTCM8MiB is scratch, not model RAM. A3200MiB mapped virtual window is address space, not additional memory capacity.

Adreno reports host unified memory and about5542MiB device-visible memory with a roughly1GiB maximum individual allocation. Neither is added to the phone's approximately11GiB system RAM. Normal OpenCL storage is driver-managed and is not directly dereferenceable as a CPU tensor. GPU/RPC counters, RSS/PSS and system MemAvailable describe different, sometimes overlapping views. Zram/swap is not extra physical RAM.

Descriptive pressure observations: Qwen3-4B GPU process RSS can be around700MiB while CPU RSS is4884–5880MiB, yet accepted ctx512/8192 GPU probes had less system MemAvailable (about4788/3555MiB) than CPU (5225/4156MiB). At7B ctx16k CPU minimum MemAvailable was about3010MiB versus GPU1905MiB, despite RSS7339 versus936MiB. Start states and other apps differ, so these are not exact isolated footprint deltas. They do rule out interpreting low accelerator RSS as newly available gigabytes of physical RAM.

Canonical/no-repack CPU+FA small probes reduced4B RSS from about4897 to2532MiB and7B7041 to4518MiB, but those diagnostics use different starting conditions and layout/throughput tradeoffs. No model/context has yet been demonstrated practical on GPU/NPU while impossible on the best memory-efficient CPU path. Full NPU context series, occupied long-context memory, canonical controls and memory-capacity frontier comparisons remain incomplete.

## Sustained load, Android foreground policy and thermal limits

### Foreground/background behavior

Screen-off/background transitions restricted the observed Termux CPU set, including a recorded moderate/background state with only CPUs0,1,5,6. Requested benchmark masks then failed or lost required cores. Such runs are excluded. User phone activity may have contributed to some transitions, but causation is not assumed.

GPU and NPU work still require host submission, synchronization and CPU fallback. CPU decode in a hybrid directly inherits CPU restrictions. No controlled foreground-versus-background GPU/NPU comparison or permanent bypass was established. A wake lock can address sleep behavior but did not by itself restore the required CPU set. There is no non-root guarantee that Termux can override Samsung background/thermal policy, and no such override was attempted. [Android cgroups](https://source.android.com/docs/core/perf/cgroups) describe the platform mechanism.

### Sustained protocol and partial results

One persistent model/context, deterministic512prompt+128generated tokens,ctx1024,t5/maskbc,b512/ub128,FAOFF/F16KV,nice10, unplugged. KV is cleared between requests. A full warmup request precedes timing, followed by10 seconds idle. No cooldown occurs inside the block. Intended duration900seconds; process/cgroup/memory/temperature samples every1second, public thermal manager samples every10seconds.

The public [Android NDK thermal API](https://developer.android.com/ndk/reference/group/thermal) is callable from a small native C helper without root or DUMP access. Status0/1/2/3 means NONE/LIGHT/MODERATE/SEVERE. Headroom1.0 is the severe-throttling threshold, not a percentage of remaining capacity; forecast/history have limitations described in the [ADPF thermal guide](https://developer.android.com/games/optimize/adpf/thermal).

The harness stops at public SEVERE, battery >43C, unsafe memory reserve, changed controlled scheduling, or low-battery hold. These are deliberate censoring rules. A safety stop is not an inference correctness failure, and a censored block is not a completed15-minute benchmark.

|Block|Completed requests|Mean PP|Mean TG|Mean response|Outcome|
|---|---:|---:|---:|---:|---|
|CPU pilot r0|36|74.22|19.01|13.58s|SEVERE at491.70s after first measured request|
|NPU pilot r0|16|822.03|28.13|5.14s|SEVERE at82.46s; warmer start than CPU|
|NPU colder r1|27|815.17|28.04|5.16s|SEVERE at142.41s|
|CPU colder r1|36|75.29|19.30|13.38s|**Invalid scheduling comparison:** live cgroup/core change|

Units PP/TG tok/s. CPU r1 numbers are retained for audit, not ranking. GPU long blocks did not begin before the explicit pause. Pilot NPU started battery38.0C versus CPU35.7C; its earlier cutoff cannot fairly rank backend thermal tolerance. Cold NPU r1 began35.5C/statusNONE, while CPU r1 began34.8C/statusNONE but later failed scheduling controls. There is no fully matched, repeated time-to-limit ranking.

CPU r0 first/last approximately163-second windows: PP82.50->70.49, TG20.98->18.12, response12.26->14.27s. Battery35.6->41.0C; observed CPU7 ceiling2.7456GHz->minimum1.3632GHz. Hardware maximum3.3984GHz is not the observed starting ceiling. NPU r1 first/last windows: TG28.86->27.45, response5.03->5.25s. Similar short-run stability does not prove indefinite sustainable performance; the safety threshold ended the run.

NPU r1's hottest sampled NPU-local sensor reached96.2C while battery reached38.5C. Internal junction and battery temperatures are different sensors; this is not permission to run hotter, proof of a higher safe operating temperature, or a comparison of manufacturer throttle thresholds. Readable kernel cooling states can remain0 while the public API reports throttling and CPU ceilings change. Not all OEM thermal policy is visible.

CPU process occupation was about487.48% (4.87 CPU cores) in CPU r0, versus NPU6.38% of one core in NPU r0 and6.30% in NPU r1. This is measured **freeing of host CPU resources**, not whole-phone power or energy efficiency. Different block starts/durations prevent an isolated thermal/energy ranking. [Sustained summary](reports/2026-10-03/data/sustained-matched-cold-r1-results.json), [requests](reports/2026-10-03/data/sustained-requests.csv), [telemetry](reports/2026-10-03/data/sustained-telemetry.csv).

### Power-policy candidates and scheduler implications

Unmodified HTP source requests maximum bus/core performance corners with DCVS disabled, sleep disabled, and high/maximum HMX corners. This is a concrete candidate for inefficient sustained operation, not a measured attribution of heat. Host event synchronization yields in a polling loop; measured low NPU host CPU usage means it was not a dominant CPU burn in these sustained pilots. More detailed operation/fence profiling remains necessary.

The SDK documents `HAP_power_destroy(ctx)` for power clients. The observed backend stop path frees its context without an explicit call to that wrapper. Whether process death automatically removes all votes on this firmware is unmeasured; a persistent power-vote leak is not established. The current CPU handoff retains backend ownership/session objects, so moving graphs to CPU alone does not prove accelerator votes are released.

An eventual Codey-OS scheduler would need measured admission/thermal signals, conservative hysteresis, model/layout-aware switching costs, explicit accelerator idle/lifecycle handling and per-workload response-latency objectives. Switching between blocks can help only if the shared SoC thermal budget and transition costs permit it. No adaptive CPU/GPU/NPU scheduler, sustained advantage of switching, or higher-temperature operating recommendation has been validated.

## Answers to the requested engineering questions so far

|Question|Evidence-based answer at this pause|
|---|---|
|Is OpenCL faster anywhere?|Some larger PP/long-prompt and specialized PP screens favor GPU over ordinary CPU settings. No final best-practical-CPU win is certified. Original generic1.5B is clearly slower.|
|Model-size crossover?|Relative generic efficiency improves in some larger-model screens, but architecture/CPU tuning/context also change the result. No parameter-count threshold established.|
|PP versus generation?|Accelerators are much more competitive for large batched PP. NPU PP has repeated large gains; generic and specialized GPU generation generally trail good CPU baselines.|
|Partial versus full offload?|Often better than full GPU for short decode. Low4B offload is a candidate improvement over its t4 anchor; no global optimum or tuned-CPU superiority confirmed.|
|Best batch?|Not established. Larger batches alone do not fix generic1.5B. Stable matched screens and long-context confirmations remain.|
|CPU fallback cost?|About9.36% direct CPU graph time in one instrumented1.5B phase; boundary/wait costs are additional and not cleanly additive. Not a universal percentage.|
|Dispatch/synchronization cost?|Enqueue4.55%, args1.15% in that phase; waits include execution. Many CPU/GPU boundaries are important. A single pure-overhead percentage is unsupported.|
|Larger models improve GPU efficiency?|Sometimes relative PP improves;7B kernel time becomes dominant. Not a guarantee, and tuned CPU changes the apparent crossover.|
|Accelerator RAM-pressure benefit?|Low process RSS is real but incomplete accounting. No demonstrated extra physical capacity or consistent system-headroom advantage.|
|Enables otherwise-impractical models?|Not established against canonical/no-repack CPU and appropriate KV/layout controls. Guarded skips are not OOM proof.|
|Fastest recorded CPU?|1.5B standalone TG37.19;4B whole-response decode15.555508, repeated15.387650. Different workloads/power states; no universal optimum.|
|Fastest established accelerator throughput?|NPU repeated means1.5B PP839.18/TG32.97;4B PP235.66/TG12.94 at common settings.7B only partial smoke4.04, not a full-NPU benchmark.|
|Best GPU/hybrid response?|Long4B/7B candidates listed above; short4B hybrid loses29.2%. Final optimal response strategy remains unresolved.|
|Can backends switch without recomputation?|Yes in the isolated one-context Qwen2/Qwen3 prototype. Explicit weights/KV/scheduler handling is required.|
|Must KV/weights be duplicated?|Current full-offload prefill retains canonical CPU mappings plus packed accelerator storage; handoff copies device KV. Zero-copy reuse is not proven. CPU-resident prototype avoids handoff KV copies but still transfers during prefill.|
|Should hybrid enter Codey-OS now?|Research candidate only. Need best-CPU latency confirmation, memory/lifecycle design, recurrent/concurrent-state support and sustained policy measurements.|
|Who wins long-running/thermal operation?|Unresolved. CPU and NPU reach SEVERE in pilots; cold CPU comparison invalid and GPU trials pending. No accelerator is proven immune to phone throttling.|

## Highest-value next work after explicit resume

1. Complete fresh best-CPU versus NPU comparisons with matched power/thermal state, then actual fixed-context total-response tests. The common-setting NPU PP benefit is strong; decode supremacy is still open.
2. Validate graceful shutdown helper v2 before using it for sustained experiments. Its latest edits/recompile were pending when paused; v1 measured artifacts remain preserved. Repeat cold CPU/GPU/NPU blocks with stable cgroup state and balanced ordering; record safety censoring rather than attempting to override thermal limits.
3. Investigate lower/scalable HTP performance corners and explicit idle/backend vote release in an isolated build. Measure response latency, host occupation, temperatures and energy only when power measurements are reliable.
4. Finish NPU partial offload, batch, occupied-context and architectural memory series. The prepared124-configuration screening schedule is a plan, not124 completed tests. Determine safe7B/full-offload feasibility without sacrificing the memory reserve.
5. Profile NPU decode and memory traffic. Packed-weight-size times TG gives about32.95GiB/s for1.5B and32.96GiB/s for4B, a striking bandwidth-bound hypothesis. This is a weight-throughput **proxy**, not a hardware DDR bandwidth measurement or exact bytes touched.
6. Reduce generic GPU K/V placement fallbacks and repeated backend boundaries; preserve supported-shape checks. Consider fused/specialized kernels only after numerical gates. Avoid equating fewer launches with faster execution.
7. Finish isolated specialized-path best-CPU and larger-model confirmations, then sustained GPU controls. Keep the released generic installer baseline until correctness/reproducibility/stability evidence justifies a separate option.
8. Improve hybrid CPU layout and selective occupied-KV migration, quantify page faults/repacking/transition cost, and benchmark total response across128/512/2048/8192 prompts. Production scheduling requires independent ownership/concurrency and accelerator power teardown.

## Reproduction status and preserved runtime rules

The public installer still reproduces the original generic OpenCL proof; it does not install the experimental NPU or hybrid prototypes. Exact released build configuration remains:

```sh
cmake -S "$SOURCE" -B "$BUILD" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER="$PREFIX/bin/cc" \
  -DCMAKE_CXX_COMPILER="$PREFIX/bin/c++" \
  -DGGML_OPENCL=ON -DGGML_OPENCL_USE_SPHAL=ON \
  -DGGML_OPENCL_TARGET_VERSION=300 \
  -DGGML_OPENCL_USE_ADRENO_KERNELS=OFF \
  -DGGML_OPENCL_USE_ADRENO_BIN_KERNELS=OFF \
  -DGGML_OPENCL_EMBED_KERNELS=ON -DGGML_OPENCL_PROFILING=OFF \
  -DGGML_BACKEND_DL=OFF -DLLAMA_BUILD_APP=OFF \
  -DLLAMA_BUILD_SERVER=OFF -DLLAMA_BUILD_TESTS=ON
cmake --build "$BUILD" --target llama-completion llama-bench test-backend-ops -j4
```

Initialization order: load Adreno driver through exported `sphal`, call `InitOpenCLDriver()`, then load the vendor OpenCL ICD and resolve function pointers. The backend has no startup OpenCL DT_NEEDED or OPENCL_* symbol-version requirement. The earlier startup-graph mystery was partly a confounded `LD_LIBRARY_PATH`: adding `$PREFIX/lib` selected Termux's incompatible `libbinder_ndk.so` for a vendor `libCB.so` dependency, causing init-30. A trivial startup dependency worked when that environment trap was absent. Mixing original/installed/experimental llama libraries caused a separate shutdown double-free.

Run with **only the selected build/bin in LD_LIBRARY_PATH**, clear inherited LD_PRELOAD/GGML backend overrides as the harness/launcher does, and do not globally prepend Termux/system/vendor library directories. For the preserved NPU experiment, ADSP_LIBRARY_PATH selects its own source-built DSP directory and profile0 is used for speed; exact configure arguments and ELF hashes are exported in [npu-configure-command.json](reports/2026-10-03/data/npu-configure-command.json) and [npu-elf-validation.json](reports/2026-10-03/data/npu-elf-validation.json).

Upstream [Snapdragon backend documentation](https://github.com/ggml-org/llama.cpp/blob/master/docs/backend/snapdragon/README.md), [Qualcomm FastRPC](https://github.com/qualcomm/fastrpc), and Android's thermal APIs provide the implementation background. Device measurements and checked local source are the authority for the conclusions in this report. llama.cpp MIT attribution is retained; experimental SDK/compiler/vendor components have their own terms and are not included.

**State at publication:** report and existing evidence published; experimental testing explicitly paused; optimized installer integration and final characterization incomplete. No tests should resume until the user asks.
