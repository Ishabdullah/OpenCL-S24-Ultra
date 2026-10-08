# Qwen3-Coder-Next MoE Triad Plan

Tracks the pivot to Qwen3-Coder-Next (MoE) on the S24 Ultra, built around three
pillars: **Ultra-Low-Bit Compression (ULBC)**, **Active-Parameter Layer
Streaming (APLS)**, and **Multi-Token Prediction speculative decoding (MTP)**.

This doc is the checkable plan. State and logs live in `.codex_state.json`
and `execution_history.log` (both at repo root; the log is gitignored via the
existing `*.log` rule). Anyone can pick this up by reading those two files
first, then this doc.

**Do not touch the validated GPU/NPU installers or their pins
(`upstream.json`, `npu-upstream.json`, `install.sh`, `install-npu.sh`).** This
track uses a second, separate llama.cpp checkout so the existing validated
installs stay reproducible. See [Decision D1](#decisions).

---

## Phase 0 — Verification gate (premises checked against source, not memory)

Every claim in the original briefing was checked against the actual pinned
llama.cpp source at `.work/llama.cpp` (commit `e358d59178377be4c58ba567925e05faadbccb57`,
the same pin `upstream.json`/`npu-upstream.json` already use) rather than
assumed. Results:

- [x] **Does this llama.cpp pin know about Qwen3-Coder-Next's architecture?**
      Yes. `src/llama-arch.cpp` defines `LLM_ARCH_QWEN3NEXT` ("qwen3next"),
      with a dedicated model graph in `src/models/qwen3next.cpp`.
- [x] **Does the architecture carry native MTP/NextN draft heads?**
      Yes. `qwen3next.cpp` loads `nextn.eh_proj`, `nextn.enorm`, `nextn.hnorm`,
      `nextn.embed_tokens`, `nextn.shared_head_head/norm` tensors when
      `hparams.n_layer_nextn > 0`, and implements a dedicated
      `graph_mtp` (`LLM_GRAPH_TYPE_DECODER_MTP`) decoder path.
- [x] **Does `--spec-type draft-mtp` exist as a real CLI flag?**
      Yes, confirmed in `common/speculative.cpp`/`common/arg.cpp`:
      `COMMON_SPECULATIVE_TYPE_DRAFT_MTP` → `"draft-mtp"`, alongside
      `ngram-simple`, `draft-eagle3`, `draft-dflash`, etc. `--spec-type` is a
      real, documented option (`common/arg.cpp:4249`).
- [x] **Do `--n-cpu-moe` / `-ot "exps=CPU"` exist for MoE expert offload?**
      Yes. `-ncmoe/--n-cpu-moe` and `-cmoe/--cpu-moe` exist
      (`common/arg.cpp:2754-2779`) specifically to keep MoE expert weights off
      the GPU/NPU and on CPU/RAM — this is pillar 2 (APLS) directly.
- [x] **Can the Hexagon NPU backend execute `MUL_MAT_ID` (the MoE routing
      matmul), so attention/dense/shared-expert tensors can stay NPU-resident
      while routed experts stream from CPU?**
      Yes, with a caveat. `ggml/src/ggml-hexagon/ggml-hexagon.cpp` implements
      `GGML_OP_MUL_MAT_ID` (including an `MUL_MAT_ID_NX` fusion path) but gates
      it on a **VTCM size budget** (`"supported MUL_MAT_ID VTCM size needed
      (%d) > budget (%zu)"`). Past that budget it falls back off-NPU. The
      practical split (always-NPU attention/dense/shared-experts vs.
      CPU-streamed routed experts) is achievable, but the VTCM ceiling — not
      the architecture — decides how much routing can stay on-NPU. Measure
      this in Phase C rather than assuming full offload.
- [x] **Is literal 1.58-bit ternary (BitNet b1.58) available for
      Qwen3-Coder-Next?** **No.** BitNet b1.58 requires weights trained
      ternary from scratch; `bitnet.cpp` cannot post-hoc convert a
      pretrained Qwen checkpoint, and no such artifact exists on Hugging Face
      for this model (checked via the HF connector — see below). ULBC must
      use **post-hoc low-bit GGUF quantization** instead: `IQ2_XXS`/`IQ1_S`-class
      or MXFP4-per-expert quants, not true ternary. This is a scope
      correction from the original briefing; log it as [Decision D2](#decisions).
- [x] **Does a real, downloadable Qwen3-Coder-Next GGUF with the MTP head
      exist?** Yes — confirmed via the Hugging Face connector
      (`hub_repo_search`), base model `Qwen/Qwen3-Coder-Next`, with multiple
      community GGUF conversions. Relevant for this RAM-constrained device:
      - `lovedheart/Qwen3-Coder-Next-REAP-{40,48,60}B-A3B-GGUF` — **REAP
        expert-pruned** variants (total params cut from the full MoE down to
        40B/48B/60B while keeping ~3B active params/token). These are the
        realistic size target for a 10GB-class phone.
      - `noctrex/Qwen3-Coder-Next-MXFP4_MOE-GGUF` and
        `noctrex/Qwen3-Coder-Next-REAP-48B-A3B-MXFP4_MOE-GGUF` — 4-bit
        microscaling per-MoE-expert quantization, the closest realistic
        analogue to "ultra-low-bit MoE compression" available for this model.
      - `unsloth/Qwen3-Coder-Next-GGUF`, `bartowski/Qwen_Qwen3-Coder-Next-GGUF`
        — standard IQ/Q-series quants with imatrix, unpruned (full MoE size,
        likely too large to fit).
      - No listing shows MTP tensors stripped or retained explicitly in its
        tags — **must verify per-file** in Phase B by inspecting GGUF
        metadata (`gguf-py` or `llama-gguf-dump`) for `nextn.*` tensors before
        picking a download.
- [x] **What's the actual RAM ceiling?** `free -h` at task start: **10Gi
      total, 4.2Gi available** (other Termux/Android processes already
      resident; swap present at 15Gi but swap-backed LLM inference is not a
      throughput target). Treat **~4-6GB resident for model + KV cache** as
      the realistic working budget until measured otherwise — this is
      tighter than the 8-9GB assumed in the original briefing and should
      drive quant/pruning choice, not the reverse.

**Gate verdict:** Pillars 2 and 3 are real, implemented features at this
llama.cpp pin — the briefing's technical premises for APLS and MTP hold up.
Pillar 1 needs correction (no true ternary; use REAP pruning + MXFP4/IQ2
quantization as the practical ULBC substitute). Proceed to Phase A with this
correction applied.

---

## Decisions

- **D1 — Separate checkout, pin untouched.** This track clones a second,
  unpinned llama.cpp checkout under `.work-qwen3/llama.cpp` (gitignored,
  mirroring the `.work/` pattern) rather than modifying the pinned
  `.work/llama.cpp` the GPU/NPU installers depend on. `upstream.json` and
  `npu-upstream.json` are not touched by this track. If a result here
  eventually warrants re-pinning the main installers, that's a separate,
  explicit decision — not a side effect of this work.
- **D2 — ULBC redefined as REAP pruning + MXFP4/IQ2 quantization**, not
  literal BitNet ternary, because no trained-ternary Qwen3-Coder-Next
  checkpoint exists. This is the closest real substitute for "extreme
  parameter footprint reduction" and is downloadable today.
- **D3 — README stays as-is until Phase C produces numbers.** Per the
  project's existing convention (measured vs. unverified claims kept
  strictly separate), `README.md` only gets a new section once there's a
  benchmark to cite.

---

## Phase A — Research & planning

- [ ] Pick the specific GGUF file(s) to download: compare
      `lovedheart/Qwen3-Coder-Next-REAP-40B-A3B-GGUF` vs. `-48B-A3B` vs.
      `-60B-A3B` vs. `noctrex`'s MXFP4_MOE REAP variant on: file size per
      quant level, presence of `nextn.*` (MTP) tensors, and imatrix
      availability.
- [ ] Dump GGUF metadata for the shortlisted file(s) (`gguf-py` reader or
      `llama-gguf-dump` from the new checkout) to confirm: architecture =
      `qwen3next`, `n_layer_nextn` present and `> 0`, expert count/size,
      quant type actually used per tensor.
- [ ] Record findings in this doc's Phase A notes (below), not a separate
      research doc — keep the plan and the findings in one place.
- [ ] Confirm via `llama-cli --spec-type draft-mtp --help`-equivalent (once
      built) that the new checkout's binary actually lists `draft-mtp` as a
      selectable type for the chosen GGUF's architecture.

**Phase A notes:** _(fill in as work proceeds)_

---

## Phase B — Implementation & setup

- [ ] Clone second checkout: `.work-qwen3/llama.cpp` at a recent upstream
      commit (does not need to match the pinned `e358d591`; record whatever
      commit is used here in `.codex_state.json`, not in `upstream.json`).
- [ ] Build with Hexagon NPU + OpenCL backends enabled (reuse
      `install-npu.sh`'s SDK/toolchain setup logic as reference, but do not
      invoke the installer against this checkout — build manually or via a
      new script under `scripts/`).
- [ ] Download the chosen GGUF(s) from Phase A into `~/models/` (outside the
      repo, consistent with existing convention).
- [ ] Configure APLS: start with `--n-cpu-moe N` sweeps (N = 0, partial, all
      routed experts) to find the VTCM-budget crossover identified in Phase 0,
      keeping attention/dense/shared-experts NPU/GPU-resident.
- [ ] Configure MTP: `--spec-type draft-mtp` against the same model (no
      separate draft model needed — NextN head is embedded), and as a
      fallback/comparison, `--spec-type ngram-simple`.
- [ ] Write a benchmark script (new, under `scripts/`) that runs prefill +
      decode under each config and records prompt tok/s, generation tok/s,
      MTP acceptance rate (from llama.cpp's own speculative stats output),
      and peak RSS — appending structured results to `execution_history.log`
      and summary fields to `.codex_state.json`.

---

## Phase C — Live verification & testing

- [ ] Baseline: CPU-only, no speculation, chosen quant — tok/s + peak RAM.
- [ ] NPU/GPU dense+attention resident, CPU-streamed experts (APLS), no
      speculation — tok/s + peak RAM, compare to baseline.
- [ ] APLS + MTP (`draft-mtp`) — tok/s, MTP acceptance rate, peak RAM.
- [ ] APLS + `ngram-simple` as a speculative-decoding comparison point.
- [ ] Repeat the `--n-cpu-moe` sweep from Phase B with real measurements, not
      just the VTCM-budget estimate, to find the actual best split.
- [ ] Quality spot-check: same prompt set across quant levels (REAP-40B vs
      48B vs 60B, or MXFP4 vs IQ-series) — note any obviously broken output,
      not a formal eval.
- [ ] Log every run (success or failure, exact error text, config) to
      `execution_history.log` immediately; update `.codex_state.json`
      `metrics` after each completed run.

---

## Phase D — Documentation & git discipline

- [ ] Add a new, clearly-dated section to `README.md` summarizing measured
      results only, following the existing hedged-claims convention (measured
      vs. not-yet-measured, one run vs. repeated).
- [ ] Document the final APLS/ULBC/MTP configuration (exact flags) in a new
      `docs/QWEN3_TRIAD_RESULTS.md` or as an addendum to this plan.
- [ ] Commit + push after each phase's completion, per the user's standing
      instruction for this track.

---

## State & recovery

- `.codex_state.json` (repo root, gitignored is **not** required — it's
  small and useful in history, so it is tracked) holds `current_phase`,
  `active_subtask`, `completed_steps`, `failed_attempts`,
  `technique_status`, `hardware_target`, and `metrics`.
- `execution_history.log` (repo root, gitignored via the existing `*.log`
  rule) holds append-only, flushed-per-line command output and results.
- On resume: read both files first. `failed_attempts` records exact error
  text and hardware context so a repeated failure is recognized instead of
  re-attempted blindly.
