# Bounded CPU Expert Cache — Handoff Document

**Date**: October 15, 2026
**Status**: Experimental prototype — opt-in, not installer default
**Model validated**: Qwen3.6-35B-A3B UD-IQ1_M (40-layer MoE, 256 experts/layer, 10 active/token)
**Hardware**: Samsung Galaxy S24 Ultra, Snapdragon 8 Gen 3, 10 GiB RAM, Termux 0.118.3

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
1. **NPU-side bounded expert cache** — The profiling in Phase F (`scripts/moe-profile-probe.cpp`) showed catastrophic decode-phase re-reads (~85 GiB for 32 tokens, ~8× model size). The CPU cache helps but the same problem exists for NPU-resident experts (when quantization permits). The Hexagon backend already has `set_tensor`/`get_tensor` and this repo proved tensor rebinding works (hybrid handoff patch). A per-token slot-swap hook in `MUL_MAT_ID` is structurally feasible but a materially larger patch.

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