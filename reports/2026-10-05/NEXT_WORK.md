# Work remaining after the October 5 measurements

Testing is paused after exactly three authorized inference runs. This is a checkpoint, not completed characterization; no tests resume until authorized. Preserve `~/llama.cpp` as read-only, known-good generic OpenCL/NPU source and artifacts, and all excluded records. No installer/kernel defaults changed.

## Established today

- The latest matched single-run 1.5B 8K screen completed in all three modes: CPU 134.81 s, OpenCL 231.07 s, OpenCL-prefill/CPU-decode 286.83 s. All 128 outputs matched and measured guards passed. This negative GPU screen is separate from the repeated NPU benefit below.
- Source inspection finds that mixed F32/F16 Flash Attention is declined on A7X including Adreno 750, causing CPU attention fallback despite full layer offload. No per-op attribution or safe guard bypass has been demonstrated.

- Mistral 7B p2048/g128/c2560, six CPU threads, FA ON, batch512/microbatch256: three independent rotated CPU/full-NPU/NPU-prefill-CPU-decode comparisons. Mean warm response116.13/39.68/29.17seconds. Hybrid beat both in all three rounds; all128 generated tokens agreed in this repetitive synthetic task. Model load/admission excluded from warm response, separately recorded.
- A one-way handoff preserves the same model/context/KV state without prompt replay. It releases converted NPU weights and copies320MiB allocated KV; mapped canonical CPU shadows incur first-decode demand faults. This is not zero-copy, no second context, and no validated bidirectional scheduler.
- Foreground FastRPC profiling finds11/9 new mappings per decoded token for2048/3072MiB windows. Map calls occupy45.5/40.2% decode wall, with overlap/flush-wait caveats. Larger window passes numerical checks, but instrumented timing is not a repeated speed claim.
- Partial offload3/6/16/24/all improves some decode points relative to full NPU; full NPU has shortest response in the p512/g64 screen. No repeated optimum.
- Explicit OpenMP placement and separate CPU prefill/decode thread counts work, verified per worker/phase. CPU speed varies; four decode threads are not a universal improvement.

## Next measurements, in priority order

1. Quantify generic OpenCL CPU attention fallback, transfers/synchronization and canonical-shadow versus repacked CPU decode costs. Preserve the baseline A7X guard; an isolated current-driver mixed FA compile/numerical gate must precede any guard change. The completed 1.5B 8K screen favors CPU. Resume the 4B 8K CPU6/coreFC/FA ON/b512/ub256 comparison only when the conservative memory budget admits it; two earlier CPU attempts were scheduling-censored and a later GPU-first plan never entered inference. Repeat any apparent advantage in rotated orders before a speed claim.
2. Finish the 4B CPU thread screen and independently confirm selected best CPU versus full NPU/hybrid. Charging and unplugged results must remain distinct; reject Android cgroup/core transitions.
3. Uninstrumented rotated2048/3072MiB registration-window comparison, three independent processes per window. Preserve1536MiB system reserve; skip when conservative preflight cannot admit the full7B model. Do not confuse an admission skip with an allocation failure.
4. Full Qwen7B numerical gate in registration-window backend, then matched response comparisons if admitted. Existing model files first; no new model downloads needed.
5. Practical long occupied prompts/contexts including8192 where safe. Existing512/2048/8192/16384 memory probes mostly allocated empty context and are not evidence of full16K ingestion.
6. Confirm generic OpenCL long-prefill candidates against the best practical CPU with FA ON and equal conditions. No final generic OpenCL crossover established.
7. Equal offered-work sustained GPU/hybrid tests in both orders; existing repeated600s equal-work findings cover1.5B CPU/NPU only. Compare response/deadline stability rather than uncontrolled maximum throughput or global battery current as backend energy.
8. Isolate specialized Adreno submission failure after generic characterization. Preserve generic defaults OFF; the narrow Q4_K GEMV workaround passed numerical gates but wider failure/general performance is unresolved.
9. Determine whether F16 NPU KV can be CPU-aliased safely with explicit coherence/lifetime ownership. Source shows CPU-readable rpcmem-backed data, but current handoff truly copies it; this is only an optimization lead. Weight formats still differ.

The virtual-session layer-split alternative is currently blocked at session1 opening. Session0 alone succeeds; session1 fails0x200 both alone and after session0 in zero-model/zero-graph controls. No two-session speed or memory advantage exists in our measurements. Diagnose the exact vendor RPC cause before attempting7B virtual-session benchmarks. Keep the working single-session registration-window path.

## Operational limits

Serialize compile/inference with the existing resource lock, leave cores0/1 for UI, nice10. Battery<=20% hold; recovery>=25% plus fresh five-minute settling. Preserve active1536MiB MemAvailable reserve, battery43C/publicSEVERE stops and stable sampled CPU/cgroup/library checks. Public thermal NONE does not prove unrestricted clocks. Use Termux API voice for notices, do not override OEM security/CPU scheduling. The charger API observed a new charging epoch14:59:45UTC today despite the initial unplugged instruction; fresh charging tests were separated. The completed three-test screen used the later API-confirmed unplugged epoch18:03:28UTC, after settling. Earlier charging and unplugged epochs are not pooled.

## Reproduction and publication

The installer source pins remain unchanged. Experimental patches are in [October4 patch archive](../2026-10-04/patches/README.md), with exact helper build commands/hashes in this checkpoint's data. The combined prototype reused unchanged native/DSP artifacts; no new clean-room installer validation or all-three-backend production executable is implied. Raw full telemetry/text remain local, compact results/source are archived. Repository contains no Qualcomm libraries, SDK/compiler binaries or GGUF models.
