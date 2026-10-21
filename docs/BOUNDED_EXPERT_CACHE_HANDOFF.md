# Bounded CPU Expert Cache — Handoff Document

**Date**: October 15, 2026
**Status**: Experimental prototype — opt-in, not installer default
**Model validated**: Qwen3.6-35B-A3B UD-IQ1_M (40-layer MoE, 256 experts/layer, 10 active/token)
**Hardware**: Samsung Galaxy S24 Ultra, Snapdragon 8 Gen 3, 10 GiB RAM, Termux 0.118.3

## October 20, 2026 — Qwen3-30B-A3B Q4_K HTP Follow-up

This is a separate accelerator-side experiment; the CPU-cache measurements above
remain specific to Qwen3.6-35B-A3B UD-IQ1_M. The unified Qwen3-30B-A3B Q4_K
runner now supports an opt-in bounded HTP expert-cache placement at
`reports/2026-10-19/qwen3-30b-a3b-thread-sweep/run-benchmark.py` and the
prototype source is preserved in
`patches/experimental-htp-bounded-expert-cache.patch`. It keeps selected Q4_K
expert tensors CPU/mmap-backed, stages routed experts into a byte-capped HTP
buffer, remaps route IDs, and synchronizes before slot reuse.

The first one-HTP-block smoke completed two bit-identical passes without a
crash. It used CPU/GPU/HTP placement 32/16/1, 8 HTP cache slots, a 512 MiB HTP
budget, and a 2 GiB live `MemAvailable` floor. Peak RSS was about 3.18 GiB and
minimum `MemAvailable` was 5.28 GiB. However, it did not report any HTP cache
requests, so it is **not** evidence that Q4_K experts executed through the HTP
cache. A second run with Hexagon op fusion disabled also had no cache requests.
Both attempts are retained under the matching `attempt-20261020-bounded-htp-*`
directories. The current diagnostic run adds targeted HTP support-probe and
graph-dispatch logs to identify why those expert ops do not reach the staging
path.

Q4_K GPU support is also generation-specific. This checkout contains OpenCL
Q4_K `MUL_MAT_ID` kernels and a DP4A prefill path, but disables the affected
Adreno A6x/A7x MoE kernels because of a compiler weight-repack miscompile. The
phone's Adreno 750 is in that excluded family; dense Q4_K GPU matmul support
does not imply GPU routed-expert support on this device.

---

## Summary

A bounded expert cache was implemented in the CPU backend (`ggml-cpu`) for MoE `MUL_MAT_ID` operations. It allocates up to **32 slots per canonical expert weight tensor** (gate/up/down separately) with a **1,280 MiB total byte budget** per CPU backend. The cache:

- Pins selected experts against eviction for the duration of each operation
- Fills missing slots from mmap-backed weights on demand
- Evicts unselected slots by recency
- Falls back to full-tensor computation on oversized batches, budget overflow, or tensor identity changes
- Preserves original model tensors, router IDs, and routing probabilities

**Correctness**: All 248,320 vocabulary logits bit-identical at 16 positions (prefill + 15 decode) and 64 positions (4-pass ABBA balanced order, 192 comparisons).

**Performance** (one prompt, one device, one session — **not a universal claim**):
- Evaluation time: 288.2 s → 122.7 s (**2.35×** speedup)
- Storage reads: 219.15 GiB → 36.61 GiB (**83.3% reduction**)
- Decode throughput: 0.224 → 0.541 tok/s (**2.41×**)
- Faster in both baseline-first and cache-first orders

**Thermal**: Battery peak 42.2°C, CPU sensor peak 84.9°C (sampled, not admission-controlled).

---

## Where the Cache Runs

**Critical**: The Qwen3.6-35B-A3B UD-IQ1_M model uses **IQ1_S/IQ2_XXS quantization for routed experts**. The Hexagon backend's `ggml_hexagon_supported_mul_mat_id` **rejects IQ1/IQ2 types** (accepts Q4_0, Q4_1, Q8_0, IQ4_NL, MXFP4, Q4_K, Q5_K, Q6_K). Therefore, routed expert `MUL_MAT_ID` ops **fall back to CPU** even when layers are nominally offloaded to NPU. The cache correctly belongs in the CPU backend's `graph_compute` path. This does not prove feasibility for quantizations whose expert kernels actually execute on Hexagon.

---

## Patch Location & Application

**Patch file**: `patches/experimental-cpu-moe-cache.patch` (relative to repo root)

**Target source**: `.work-npu/llama.cpp` (the pinned commit `e358d59178377be4c58ba567925e05faadbccb57` built by `install-npu.sh`)

```sh
# Apply patch
cd /path/to/OpenCL-S24-Ultra
git -C .work-npu/llama.cpp apply "$PWD/patches/experimental-cpu-moe-cache.patch"

# Rebuild CPU backend only
cmake --build .work-npu/build --target ggml-cpu -j 2
```

**Verify application**:
```sh
# Check patch applies cleanly (dry run)
git -C .work-npu/llama.cpp apply --check "$PWD/patches/experimental-cpu-moe-cache.patch"

# Confirm cache symbols present in built library
nm -D .work-npu/build/bin/libggml-cpu.so | grep -i moe_cache
```

---

## Reproduction Commands

### 1. Correctness Validation Gate (16 positions, bit-exact)

```sh
# Build validation tool
clang++ -std=c++17 -O2 -Wall -Wextra scripts/moe-cache-validate.cpp \
  -I.work-npu/llama.cpp/include -I.work-npu/llama.cpp/ggml/include \
  -L.work-npu/build/bin -Wl,-rpath,"$PWD/.work-npu/build/bin" \
  -lllama -lggml -lggml-base -o .work-npu/build/bin/moe-cache-validate

# Run validation (uses fixed model path in runner)
python reports/2026-10-10/cache/run-validation.py
```

**Expected**: All 16 logit comparisons pass with max absolute error 0. Artifacts: `validation.json`, `cache-stats.json`, `run.json`, `provenance.json`.

### 2. Balanced-Order Benchmark (64 positions, 4-pass ABBA)

```sh
# Build benchmark tool
clang++ -std=c++17 -O2 -Wall -Wextra scripts/moe-cache-benchmark.cpp \
  -I.work-npu/llama.cpp/include -I.work-npu/llama.cpp/ggml/include \
  -L.work-npu/build/bin -Wl,-rpath,"$PWD/.work-npu/build/bin" \
  -lllama -lggml -lggml-base -o .work-npu/build/bin/moe-cache-benchmark

# Run benchmark
python reports/2026-10-10/cache-balanced/run-benchmark.py
python scripts/analyze-moe-cache-benchmark.py reports/2026-10-10/cache-balanced
python tests/test_moe_cache_benchmark_analysis.py
```

**Expected**: 4 passes complete, all 192 logit comparisons bit-exact. Artifacts: `benchmark.jsonl`, `thermal.jsonl`, `cache-stats.json`, `analysis.json`, `run.json`, `provenance.json`.

### 3. Unit & Graph Tests

```sh
# Slot manager unit tests
clang++ -std=c++17 -O2 -Wall -Wextra tests/test_moe_slot_cache.cpp \
  -o .work-npu/build/bin/test-moe-slot-cache
.work-npu/build/bin/test-moe-slot-cache

# Quantized graph tests (Q2_K, IQ2_XXS, IQ1_S, IQ1_M at 1 & 6 threads)
clang++ -std=c++17 -O2 -Wall -Wextra tests/test_moe_cache_graph.cpp \
  -I.work-npu/llama.cpp/ggml/include -L.work-npu/build/bin \
  -Wl,-rpath,"$PWD/.work-npu/build/bin" -lggml-cpu -lggml-base \
  -o .work-npu/build/bin/test-moe-cache-graph
LD_LIBRARY_PATH="$PWD/.work-npu/build/bin" .work-npu/build/bin/test-moe-cache-graph
```

### 4. Ad-hoc Inference with Cache Enabled

```sh
# Enable cache via environment variables
GGML_CPU_MOE_CACHE_SLOTS=32 GGML_CPU_MOE_CACHE_MIB=1280 \
  scripts/run-npu.sh /path/to/model.gguf -lm mmap -t 6 -n 32
```

**Flags**:
- `GGML_CPU_MOE_CACHE_SLOTS` — slots per canonical tensor (1–256, 0 = disabled)
- `GGML_CPU_MOE_CACHE_MIB` — total expert payload budget per CPU backend (1–8192 MiB, default 1280)

---

## Artifact Inventory

| Directory | Purpose | Key Files |
|-----------|---------|-----------|
| `reports/2026-10-10/cache/` | Prototype correctness gate | `validation.json`, `cache-stats.json`, `run.json`, `provenance.json`, `graph-test.txt` |
| `reports/2026-10-10/cache-balanced/` | ABBA benchmark + thermal | `benchmark.jsonl`, `thermal.jsonl`, `cache-stats.json`, `analysis.json`, `run.json`, `provenance.json` |
| `reports/2026-10-10/cache-thermal-matched/` | Incomplete thermal admission run | `run.json` (no speedup claim) |
| `reports/2026-10-10/cache-throughput-sweep/` | Thread/slot sweep (64 slots/4 threads best) | `README.md` with ranked configurations |
| `scripts/` | Source tools | `moe-cache-validate.cpp`, `moe-cache-benchmark.cpp`, `apply-moe-cache.py` |
| `tests/` | Test sources | `test_moe_slot_cache.cpp`, `test_moe_cache_graph.cpp`, `test_moe_cache_benchmark_analysis.py` |
| `patches/` | Experimental patch | `experimental-cpu-moe-cache.patch` |

---

## Known Limitations

1. **Opt-in only** — Not enabled by default; requires env vars. Installer defaults unchanged.
2. **CPU fallback for IQ1/IQ2** — This model's routed experts run on CPU regardless of NPU offload. Cache does not accelerate NPU-executed experts.
3. **No prefetching / grading / persistence** — Pure recency eviction. Phase F profiling showed ~20% consecutive-step overlap; uniform-random routing at this granularity.
4. **No thermal control** — Benchmarks run in whatever thermal state the device happens to be. Observed speedup is session-specific.
5. **Prefill falls back entirely** — All 120 prefill expert ops exceeded slot count in testing; only decode benefits.
6. **Canonical (non-repacked) weights** — Cache uses mapped GGUF shadows, not `CPU_REPACK`-optimized layout. This is the same tradeoff documented for the NPU→CPU handoff.
7. **Single-model validation** — Only tested on Qwen3.6-35B-A3B UD-IQ1_M. Other architectures/quants untested.
8. **Patch not in installers** — `install.sh`/`install-npu.sh` clean-source checks may reject this patch on rerun.

---

## Next Steps (Prioritized)

### Immediate (from Phase F scoped proposal)
1. **NPU-side bounded expert cache** — The profiling in Phase F (`scripts/moe-profile-probe.cpp`) showed catastrophic decode-phase re-reads (~85 GiB for 32 tokens, ~8× model size). The CPU cache helps but the same problem exists for NPU-resident experts (when quantization permits). The first opt-in Hexagon step is a routing-only LRU trace at `.work-npu/llama.cpp/ggml/src/ggml-hexagon/htp/moe-expert-cache-trace.h`, enabled with `GGML_HEXAGON_MOE_CACHE_TRACE_SLOTS`. It estimates per-layer hits from routed IDs for HTP-supported `MUL_MAT_ID`; it does not yet store or rebind weights and introduces a queue sync per traced operation. Use a compatible model/quantization for this trace. A real cache still needs bounded HTP weight storage, ID remapping, safe copy ordering and bit-exact replay before performance measurement.

2. **Per-layer cache sizing from real locality data** — Phase F tighter probe found every layer touches all 256 experts within 63 calls; mean consecutive overlap only 1.66/8 (~20%). Simulated LRU hit rates: 8 slots=15.4%, 32 slots=32.6%, 64 slots=39.0%, 128 slots=43.5%. This data should drive `K` selection, not fixed 32.

### Deferred (per Phase F ordering)
3. **Learned expert grading** — Continuous frequency/recency scoring per expert per layer.
4. **Adaptive prefetching** — Predict next-step experts from router logits/history.
5. **Repeated-context routing memory** — Persist slot maps across sessions for identical prompts.
6. **MTP integration** — Use draft head predictions to prefetch experts speculatively.
7. **Persistence** — Serialize cache state to disk for cold-start benefit.

---

## Related Work (This Repository)

| Work | Location | Relation |
|------|----------|----------|
| Phase F profiling (motivation) | `scripts/moe-profile-probe.cpp`, `reports/2026-10-08/` | Measured catastrophic re-reads, expert selection uniformity |
| NPU→CPU hybrid handoff | `reports/2026-10-03/patches/npu-prefill-cpu-decode.patch`, `reports/2026-10-08/patches/qwen3next-hybrid-handoff.patch` | Proved tensor rebinding works at runtime |
| Qwen3-Coder-Next track | `docs/QWEN3_TRIAD_PLAN.md` | Separate model track; same cache mechanism may apply |
| Expert mapping experiment | `reports/2026-10-10/README.md` | 8-trial routing analysis, 92.37% held-out decode coverage |

---

## Reproduction Checklist

- [ ] Fresh Termux, `pkg update && pkg install git clang cmake ninja python opencl-headers openssl`
- [ ] Clone repo, run `./install-npu.sh --jobs 1` (builds into `.work-npu/`)
- [ ] Apply patch: `git -C .work-npu/llama.cpp apply patches/experimental-cpu-moe-cache.patch`
- [ ] Rebuild: `cmake --build .work-npu/build --target ggml-cpu -j 2`
- [ ] Run unit tests: `test-moe-slot-cache`, `test-moe-cache-graph`
- [ ] Run validation gate: `moe-cache-validate` + `run-validation.py` → 16/16 bit-exact
- [ ] Run balanced benchmark: `moe-cache-benchmark` + `run-benchmark.py` → 4-pass ABBA, 192/192 bit-exact
- [ ] Verify thermal sensors readable: `ls /sys/class/thermal/thermal_zone*/type`

---

## Contacts / Context

This work was done in the `OpenCL-S24-Ultra` repository under the compose-next workflow. The `.codex_state.json` at repo root tracks the full phase history (Phases A–M). The experimental patch is **not** part of either installer and must be reapplied manually after any installer rerun.

For questions about the cache mechanism, see the inline comments in the patch and the validation/benchmark source files. For questions about the profiling motivation, see Phase F in `docs/QWEN3_TRIAD_PLAN.md`.

## 2026-10-20: Bounded HTP Cache Prototype

An opt-in bounded cache prototype now streams selected Q4_K/Q6_K routed expert
weights from the existing CPU/mmap source into HTP slots for an explicitly
selected layer range. The first tested placement is Qwen3-30B-A3B Q4_K_M,
with 32 CPU layers, 16 GPU layers, and block 32 on HTP. It uses eight slots per
expert tensor and a 512 MiB total HTP allocation cap. This remains experimental;
do not infer throughput gains from the two-token validation smoke.

The full evidence is in
`reports/2026-10-19/qwen3-30b-a3b-thread-sweep/attempt-20261020-bounded-htp-1layer-8slots-512mib-staging-verified/`.
Both passes completed with bit-identical replay logits, the observed layer map
matched 32/16/1, and all three block-32 cache tensors emitted per-operation
staging evidence (12 staging events, 79,552,512 bytes transferred in total).
Minimum `MemAvailable` was 6,532,520 KiB, peak process RSS was 2,920,576 KiB,
and both thermal and cooldown monitors exited successfully. The eight-slot
allocations totaled 26,738,688 bytes across gate/up/down. These are cache-stage
and replay checks, not a cross-backend comparison against resident HTP weights;
that numerical equivalence check is still needed before expanding the layer
range or making performance claims.

The fresh staging run confirms the cache verifier now checks actual
per-operation staged-byte logs instead of depending on a backend-session
destructor summary that this runner does not emit before process exit. Tests:
`python -m unittest discover -s tests` (49 passing), and the linked HTP library
was rebuilt from the experimental source. The portable source delta is
`patches/experimental-htp-bounded-expert-cache.patch`.

GPU Q4_K routed-MoE dispatch remains disabled for Adreno A6x/A7x devices because
the current compiler path can corrupt repacked expert weights. Dense Q4_K GPU
support does not establish routed-MoE support on this Adreno 750.

### Resident-HTP numerical control

The first resident control is implemented as `--htp-resident-layers 1` with a
1 GiB pre-allocation cap and tested in
`attempt-20261020-htp-resident-1layer-1gib-smoke/`. It placed block 32's gate,
up, and down matrices (391,643,136 bytes total) in HTP buffers. The process
reported 420.52 MiB of HTP model allocation, minimum `MemAvailable` was
5,615,176 KiB, peak RSS was 3,061,864 KiB, and the two-pass replay passed.

A hash-enabled streamed-cache rerun is in
`attempt-20261020-bounded-htp-1layer-8slots-512mib-logit-hash-control/`; its
side-by-side result is `resident-vs-streamed-comparison.json`. With identical
Qwen3-30B-A3B Q4_K_M, `Hi` prompt, seed, 32/16/1 layer placement, and two
evaluated positions, resident HTP and streamed HTP produced the same token IDs
and identical 64-bit hashes over the full vocabulary logits at both positions:
`dd7c74e907830975` (prefill) and `c93c3c720aa4f106` (decode). Each run also
passed its own full-float `memcmp` replay check. This establishes bitwise
agreement for this short sample only; it does not replace a longer prompt and
decode validation before expanding beyond one HTP block. Speed is intentionally
not compared because the runs had different filesystem/OS cache warmth.

The resident mode, full-float logit hashing, and source patch are experimental.
The exact source hashes used for the device run are recorded as
`source-provenance.json` beside both run reports.

### 16-position replay extension

The resident and streamed-cache controls were then repeated for 16 evaluated
positions (one prefill and 15 decode positions), still on block 32 with the
same prompt, seed, thread count, and 32/16/1 placement. The reports are
`attempt-20261020-htp-resident-1layer-16pos-validation/` and
`attempt-20261020-bounded-htp-1layer-8slots-512mib-16pos-validation/`.
All 16 full-vocabulary logit hashes match across the two paths, the generated
token sequences match, and each run's second pass passed full-float replay.
Resident HTP minimum `MemAvailable` was 5,694,376 KiB with 77,320 KiB peak
swap; streamed-cache minimum was 6,716,768 KiB with 72,656 KiB peak swap.
Both runs completed with no thermal or cooldown failure. The machine-readable
side-by-side evidence is in `resident-vs-streamed-comparison.json` in the
streamed-cache report directory. This increases confidence in block-32
correctness for this prompt but still tests only one short prompt and one HTP
block; use varied, longer prompts before increasing the number of HTP layers.

### 29-token prompt replay

The runner now supports a configurable prompt and sequential one-token
prefill, preserving its batch-size-one safety constraint. A 29-token technical
prompt was tested against resident HTP and the streamed cache, each with 15
decode positions and the same 32/16/1 placement. All 16 full-vocabulary logit
hashes and generated tokens match; both runs passed their independent exact
replay checks. The results are recorded in
`attempt-20261020-bounded-htp-1layer-8slots-512mib-longprompt-16pos/` and its
`resident-vs-streamed-comparison.json`. Minimum `MemAvailable` was 5,990,928
KiB for resident HTP and 6,976,032 KiB for streaming; peak swap stayed under
80 MiB in both. The first attempt is retained separately and documents the
one-token batch harness limitation that sequential prefill fixed. These results
still cover one prompt and one HTP block, so varied prompts and layer counts
remain unvalidated. The repository suite has 52 passing tests after adding
output-placement coverage.

### Alternate-prompt replay

A second, different 30-token prompt was run through the same resident and
streamed controls at 16 evaluated positions with the same 32/16/1 layer
placement and one CPU thread. The reports are
`attempt-20261020-htp-resident-altprompt-16pos/` and
`attempt-20261020-bounded-htp-1layer-8slots-512mib-altprompt-16pos/`.
All 16 full-vocabulary logit hashes and token IDs match, and both two-pass
replay checks passed. The streamed report contains the machine-readable
comparison and source provenance. Minimum `MemAvailable` was 5,798,476 KiB
for resident HTP and 6,327,048 KiB for streaming; peak RSS was 4,116,780 KiB
and 4,930,360 KiB, respectively. Each path reported peak swap below 80 MiB.
Decode rates were 0.357 and 0.534 tokens/s for these individual runs; their
run order and cache warmth differ, so this is not a controlled speed result.
This gives a second-prompt numerical check for block 32, while additional
prompts and longer decoding remain useful before expanding the HTP layer range.

### Two-layer HTP placement probe

The same alternate prompt was then run with the streamed HTP range extended
from block 32 to blocks 32–33 (31 CPU / 16 GPU / 2 HTP layers, output layer on
CPU), using the same 8-slot, 512 MiB cache cap and 16 evaluated positions.
The run staged all six expert tensors across the two HTP blocks, used a
minimum of 6,691,756 KiB `MemAvailable`, and passed its two-pass exact replay.
Its first-position full-vocabulary logits differed from the one-layer run,
and the generated token sequence diverged immediately. This is a placement
numerics change, not a replay instability: each configuration was repeatable
within itself, but moving block 33 from CPU to HTP did not preserve output
bitwise. The detailed comparison and source provenance are in
`attempt-20261020-bounded-htp-2layers-8slots-512mib-altprompt-16pos/`.
The measured 0.433 tokens/s versus 0.534 for one HTP layer is not a controlled
speed comparison because run order and cache warmth differ. Do not treat the
two-layer setting as output-equivalent to the one-layer reference without
investigating the backend numerical difference.

### Four-layer HTP placement probe

The layer range was doubled again to blocks 32–35 with the same prompt,
streamed cache, and 16-position replay. Placement was 29 CPU / 16 GPU / 4 HTP
layers, with the output layer on CPU. All 12 expert tensors were staged, both
passes completed, and the four-layer run passed its independent exact replay.
Minimum `MemAvailable` was 6,535,412 KiB and peak RSS was 4,224,316 KiB.
As with the two-layer placement, the four-layer logits and generated tokens
differed from the smaller HTP placement from the first position; its two
internal passes did match exactly. The comparison and provenance are in
`attempt-20261020-bounded-htp-4layers-8slots-512mib-altprompt-16pos/`.
The measured 0.613 tokens/s is a single-run result and is not a controlled
speed comparison with the two-layer case.

### Eight-layer HTP placement probe

The next doubling covered blocks 32–39 (25 CPU / 16 GPU / 8 HTP layers, plus
the CPU output layer). All 24 expert tensors were staged and both passes
completed with exact internal replay. Minimum `MemAvailable` was 6,628,416
KiB and peak RSS was 4,539,408 KiB. As in the smaller placements, the logits
and generated tokens differed from the four-layer run, while the eight-layer
configuration repeated exactly within itself. The detailed comparison and
source provenance are in
`attempt-20261020-bounded-htp-8layers-8slots-512mib-altprompt-16pos/`.
The measured 0.464 tokens/s is one pass and should not be compared as a speed
ranking across placements.

### Sixteen-layer HTP placement probe

The final bounded-cache range assigns the 48 repeating blocks evenly:
16 CPU, 16 GPU, and 16 HTP blocks (32–47), with the output layer on CPU.
The run staged all 48 expert tensors, completed both passes, and passed its
independent exact replay. Minimum `MemAvailable` was 6,721,592 KiB, peak RSS
was 4,070,184 KiB, and peak swap was 70,676 KiB. Its logits and generated
tokens differed from the eight-layer placement from the first position, but
the 16-layer configuration replayed exactly within itself. The detailed
comparison and source provenance are in
`attempt-20261020-bounded-htp-16layers-8slots-512mib-altprompt-16pos/`.
The 0.448 tokens/s measurement is not a controlled comparison with the other
placements. This completes the CPU-output 16/16/16 layer-share validation; the
GPU-output and HTP-output variants are summarized below.

### Output-layer backend placement comparison

The bounded-cache runner now supports placing the output layer independently
while keeping all 16 routed blocks on HTP. The 30-token prompt was evaluated
for 16 positions with three placements: CPU output (17 CPU / 16 GPU / 16 HTP),
GPU output (16 / 17 / 16), and HTP/NPU output (16 / 16 / 17). All three runs
passed their two-pass replay and verified their layer assignments. CPU-output
and GPU-output logits and tokens matched at all 16 positions. Assigning the
output layer to HTP changed logits and tokens at the first position. This
isolates the output-layer backend effect from the 48-block division for this
prompt; it does not establish which placement gives better task quality.
The decoded CPU/GPU output was identical; the HTP-output sample instead
started its checklist with checking available RAM, while CPU/GPU started with
checking CPU/GPU usage.
The consolidated hashes, tokens, telemetry, and rates are in
`attempt-20261020-triad-output-npu-bounded-16htp-16pos/triad-output-layer-comparison.json`.
Minimum `MemAvailable` was 6,721,592 KiB for CPU output, 6,380,980 KiB for GPU
output, and 6,281,720 KiB for HTP/NPU output. The output-layer run rates are
not controlled comparisons.

### Output-layer speed comparison (2026-10-20)

A controlled six-run comparison followed in mirrored order (CPU, GPU, HTP;
then HTP, GPU, CPU), with two runs per placement. Each run used the same
30-token prompt, 32 evaluated positions, one CPU thread, 16 bounded HTP layers,
an 8-slot / 512-MiB HTP cache, and the cooldown-only policy. All six runs
completed, verified their layer assignments, and passed exact two-pass replay.
The median first-pass decode rates were CPU output **0.399 tok/s**, HTP/NPU
output **0.388 tok/s**, and GPU output **0.380 tok/s**. CPU output is the
provisional winner by this metric (2.9% over HTP and 4.8% over GPU).

The CPU result varied from 0.384 to 0.415 tok/s, HTP from 0.381 to 0.395, and
GPU from 0.377 to 0.383. Those ranges overlap, so two replicates do not
establish a reliable winner; treat CPU output as the lead to retest, rather
than a settled optimum. The per-run rates, replay checks, and memory telemetry
are recorded in
`reports/2026-10-19/qwen3-30b-a3b-thread-sweep/output-layer-speed-comparison.json`.

### CPU thread-count comparison with CPU output layer (2026-10-20)

The CPU-output 17 / GPU 16 / HTP 16 placement was then tested at 2, 3, 4, and
5 CPU threads, using the same prompt, 32 positions, 8-slot / 512-MiB bounded
HTP cache, and exact two-pass replay. All runs passed replay and placement
verification. First-pass decode rates were 0.316, 0.259, 0.212, and 0.226
tok/s, respectively. **Two threads was fastest among the new settings**, but
its rate was still 21% below the prior one-thread CPU-output median of 0.399
tok/s. The replay rates were lower for 2 and 3 threads (0.261 and 0.192 tok/s)
and closer for 4 and 5 (0.219 and 0.218 tok/s), showing substantial run-to-run
variation. The measured results therefore support retaining one thread as the
current fastest setting; two threads is the next candidate if we repeat the
comparison. Detailed rates and telemetry are in
`reports/2026-10-19/qwen3-30b-a3b-thread-sweep/cpu-output-thread-comparison.json`.

### Speculative-decoding research status (2026-10-20)

No speculative-decoding run has been performed on the S24 Ultra with the
unified GPU+HTP build. The stock Qwen3-30B-A3B configuration is the standard
48-layer `qwen3_moe` model; its published config does not identify an MTP
head, and the official GGUF card describes the standard Qwen3 MoE architecture
([config](https://huggingface.co/Qwen/Qwen3-30B-A3B/blob/2a8783f051b79cd7af4e1fb3b5c8e0292c039bbe/config.json),
[GGUF card](https://huggingface.co/Qwen/Qwen3-30B-A3B-GGUF)). Treat the tested
Q4_K_M checkpoint as lacking native single-model MTP.

The closest published direct pairing is Qwen3-0.6B drafting for
Qwen3-30B-A3B on a Strix Halo system: the report measured 27.2 tok/s without a
draft and 10.2 tok/s with it, with 34% draft acceptance (2.35 tokens per
round). This is a different engine and hardware, but it is evidence that a
small same-family draft can slow this MoE target when verification is already
cheap ([report](https://www.bigiron.cc/guides/speculative-decoding-in-a-homelab-when-a-draft-model-makes-you-slower)).
A separate SpecForge/SGLang dashboard reports 1.50–1.82x average throughput
for specialized SpecBundle configurations of Qwen3-30B-A3B-Instruct-2507;
that is a trained drafter/server-stack result, not a measurement of ordinary
Qwen3-0.6B GGUF or this phone ([results](https://sgl-project.github.io/SpecForge/specbundle.html)).

Conclusion: speedup on this phone is unknown and a slowdown is plausible.
Keep the one-thread, CPU-output 17 / GPU 16 / HTP 16 placement as the measured
baseline. Any later trial must first confirm draft-model support in the exact
unified executable, then compare a no-draft control against Qwen3-0.6B draft
with identical prompts and token limits, recording end-to-end decode speed,
acceptance, memory, and replay correctness. Until such a device test is run,
do not claim a speculative-decoding gain.
