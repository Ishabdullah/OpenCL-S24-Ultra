# October 8 patch: Qwen3-Coder-Next hybrid NPU-prefill/CPU-decode handoff

`qwen3next-hybrid-handoff.patch` is the full diff (base patch + extension) against
the pinned llama.cpp commit `e358d59178377be4c58ba567925e05faadbccb57` — the
same commit `.work-npu/llama.cpp` already builds. It supersedes
[`../../2026-10-03/patches/npu-prefill-cpu-decode.patch`](../../2026-10-03/patches/npu-prefill-cpu-decode.patch)
for this model: that patch's `llama_perf_switch_to_cpu` handoff was gated to
plain `LLM_ARCH_QWEN2`/`LLM_ARCH_QWEN3` with a bare `llama_kv_cache` (used for
the Mistral 7B investigation). This patch adds:

- `llama_memory_recurrent::perf_migrate_cpu()` — a direct port of the existing
  `llama_kv_cache::perf_migrate_cpu()` (same `ctxs_bufs` structure), migrating
  the recurrent/SSM-GDN state buffer to CPU.
- `llama_context::perf_switch_cpu()` extended to recognize `llama_memory_hybrid`
  (the memory class `qwen3next` actually uses — it wraps a `llama_kv_cache` for
  the periodic full-attention layers and a `llama_memory_recurrent` for the
  linear-attention layers) and migrate both sub-caches via `get_mem_attn()` /
  `get_mem_recr()`.
- `LLM_ARCH_QWEN3NEXT` added to the allowed-architecture check.

This is a **local, experimental, never-upstreamed prototype**, same as the
patch it extends — not submitted to llama.cpp, not meeting that project's
contribution bar (see `.work-npu/llama.cpp/AGENTS.md`). It is not wired into
any CLI flag; exercising it requires a small standalone driver, which is
[`scripts/hybrid-handoff-probe.cpp`](../../../scripts/hybrid-handoff-probe.cpp)
(compiled directly against `.work-npu/build/bin/lib{llama,ggml}.so`, not part
of the CMake build).

**Verified live** (one run, not repeated — see
[`docs/QWEN3_TRIAD_PLAN.md`](../../../docs/QWEN3_TRIAD_PLAN.md) Phase C for
full detail and the three-way comparison this enables): the handoff migrated
both memory types without error (6.00 MiB attention KV, 75.38 MiB recurrent
state) and continued decoding on CPU with no prompt replay. Post-handoff
decode throughput (0.79 tok/s) was markedly slower than a native full-CPU run
at the same thread count (3.30 tok/s), because the handoff installs canonical
(non-repacked) CPU weight shadows rather than the `CPU_REPACK`-optimized
layout a native load uses — the same tradeoff the Oct 3-5 Mistral 7B
investigation already documented, apparently more punishing for this model's
IQ1_S quantization. The technique works for this architecture; it is not a
speed win for this quant.

To apply: `cd .work-npu/llama.cpp && git apply ../../reports/2026-10-08/patches/qwen3next-hybrid-handoff.patch`
(idempotent against a clean checkout of the pinned commit; already applied in
the current `.work-npu/llama.cpp` working tree as of this writing).
