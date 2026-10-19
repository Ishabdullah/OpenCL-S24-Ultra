---
feature: bounded-expert-cache-documentation
status: complete
updated: 2026-10-19
branch: ""
commits: ""
---

# Bounded Expert Cache Documentation & Finalization

## Report

## [S1] Problem
The bounded CPU expert cache prototype (32-slot, 1.09 GiB) has been implemented, validated for correctness (bit-exact logit match at 16/64 positions), and benchmarked with promising results (2.35× eval speedup, 83% storage read reduction). This work is currently scattered across multiple report directories and an experimental patch. It needs to be consolidated into the project's main documentation (README.md), the experimental patch needs cleanup/archival, and a clear handoff document needs to be created for future work.

## [S2] Design
**Deliverables:**
1. **README.md update** — Add a dated "Bounded CPU Expert Cache checkpoint: October 15, 2026" section summarizing:
   - Cache mechanism (32-slot per-layer, canonical weight tensors, recency eviction, full-model fallback)
   - Correctness evidence (bit-exact at 16/64 positions, unit/graph tests pass)
   - Performance observations (2.35× eval speedup, 83.3% storage read reduction, balanced ABBA order)
   - Thermal observations (battery 42.2°C peak, CPU 84.9°C peak)
   - Reproduction commands and opt-in flags (`GGML_CPU_MOE_CACHE_SLOTS`, `GGML_CPU_MOE_CACHE_MIB`)
   - Limitations (opt-in, not default; IQ1/IQ2 experts use CPU fallback; not thermally controlled)

2. **Experimental patch handling** — The patch at `patches/experimental-cpu-moe-cache.patch` should be:
   - Kept as-is in `patches/` for reproducibility
   - Documented in README as experimental/opt-in
   - Not integrated into either installer (`install.sh`, `install-npu.sh`)

3. **Handoff document** — Create `docs/BOUNDED_EXPERT_CACHE_HANDOFF.md` with:
   - Complete reproduction steps (apply patch, rebuild, run validation, run benchmark)
   - All artifact locations and what they contain
   - Known limitations and next steps (NPU-side cache, prefetching, MTP integration, per-layer sizing)
   - Link to the Phase F profiling data that motivated this work

## [S3] Out of Scope
- Modifying installer scripts (`install.sh`, `install-npu.sh`) to include the cache
- NPU-side bounded expert cache implementation (separate future work)
- Prefetching, learned grading, repeated-context routing memory (per Phase F scope)
- MTP integration with cache
- Changing any validated installer defaults or pins

## Tasks
- [x] T1.1: Create handoff document `docs/BOUNDED_EXPERT_CACHE_HANDOFF.md` — acceptance: document exists with complete reproduction steps, artifact inventory, limitations, next steps (covers: S2)
- [x] T1.2: Update README.md with cache checkpoint section — acceptance: dated section added following existing hedged-claims convention, reproduction commands included (covers: S2)
- [x] T1.3: Verify experimental patch applies cleanly to current pinned source — acceptance: `git apply --check patches/experimental-cpu-moe-cache.patch` succeeds on `.work-npu/llama.cpp` (covers: S2)
- [x] T1.4: Run validation gate one more time to confirm bit-exact correctness — acceptance: `python reports/2026-10-10/cache/run-validation.py` passes all 16 logit comparisons (covers: S2)
- [x] T1.5: Commit all changes — acceptance: git commit with message summarizing documentation updates (covers: S2)