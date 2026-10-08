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
      The *code path* does: `qwen3next.cpp` loads `blk.%d.nextn.eh_proj`,
      `nextn.enorm`, `nextn.hnorm`, `nextn.embed_tokens`,
      `nextn.shared_head_head/norm` tensors when `hparams.n_layer_nextn > 0`,
      and implements a dedicated `graph_mtp` (`LLM_GRAPH_TYPE_DECODER_MTP`)
      decoder path, gated by the KV key `qwen3next.nextn_predict_layers`.
      **Corrected in Phase A:** the actual released `Qwen/Qwen3-Coder-Next`
      checkpoint does not ship these weights — see the Phase A finding below.
      The architecture *family* supports MTP; this specific model does not
      carry it.
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
      Yes. `ggml/src/ggml-hexagon/ggml-hexagon.cpp` implements
      `GGML_OP_MUL_MAT_ID` (plus an `MUL_MAT_ID_NX` fusion path).
      **Corrected in Phase A** (see below): the VTCM budget check guards the
      kernel's internal *tile size* for a single op invocation, not how many
      experts/how much weight data can be NPU-resident overall — the code
      auto-shrinks its block size to fit VTCM (down to a floor of 128
      elements/1 row) before giving up. The real ceiling on NPU-resident MoE
      data is `opt_mbuf` (`GGML_HEXAGON_MBUF` env var), the max single
      fastRPC/DMA buffer size, defaulting to **1024 MiB**. This is the number
      to tune/measure in Phase C, not VTCM.
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

**Gate verdict (updated after Phase A):** Pillar 2 (APLS) is real and
implemented, bounded by `opt_mbuf`/`GGML_HEXAGON_MBUF` rather than VTCM.
Pillar 1 (ULBC) needs correction (no true ternary; REAP pruning +
MXFP4/IQ2 quantization instead). Pillar 3 (MTP) needs a bigger correction:
the `draft-mtp` code path is real, but **no released Qwen3-Coder-Next GGUF
carries the NextN weights it requires** — see Phase A. The practical
pillar-3 substitute is `--spec-type ngram-simple` (also a real, confirmed
flag) until/unless an MTP-bearing checkpoint for this model surfaces.

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
- **D4 — MTP (pillar 3) is not available for Qwen3-Coder-Next; fall back to
  `ngram-simple`.** Confirmed by inspecting `Qwen/Qwen3-Coder-Next/config.json`
  (no `num_nextn_predict_layers` field, standard `Qwen3NextForCausalLM`
  config) and by parsing GGUF header metadata of two independent
  conversions (mradermacher's REAP-40B-A3B i1-IQ1_S and bartowski's
  unpruned IQ1_S): both report `qwen3next.expert_count`/`expert_used_count`
  but **no `qwen3next.nextn_predict_layers` KV key and zero `nextn.*`
  tensors**, across 48 blocks / 843 tensors in both files. This isn't a
  conversion artifact — the upstream safetensors release simply doesn't
  include MTP weights. `--spec-type draft-mtp` stays documented in this plan
  as a real flag for *other* qwen3next-family checkpoints that do ship
  NextN heads, but it has no target on this model today. Use
  `--spec-type ngram-simple` for pillar 3 instead, and revisit if Qwen
  releases an MTP-bearing variant.

---

## Phase A — Research & planning

- [x] **Survey candidate GGUF file sizes** via the Hugging Face connector
      (`hf_fs ls --recursive`, no download needed for a file listing):

      | Repo | Quant | Size |
      |---|---|---:|
      | lovedheart REAP-40B-A3B | Q2_K_XL (smallest on that repo) | 18.6 GiB |
      | bartowski (unpruned) | IQ1_S (smallest) | 15.4 GiB |
      | bartowski (unpruned) | IQ1_M | 16.1 GiB |
      | noctrex REAP-48B-A3B | MXFP4_MOE | 25.4 GiB |
      | noctrex (unpruned) | MXFP4_MOE | 42.8 GiB |
      | **mradermacher REAP-40B-A3B i1** | **IQ1_S (smallest found)** | **7.85 GiB** |
      | mradermacher REAP-40B-A3B i1 | IQ1_M | 8.70 GiB |
      | mradermacher REAP-40B-A3B i1 | IQ2_XXS | 10.14 GiB |

      **Finding: nothing fits a literal 4.2 GiB resident-RAM budget as a
      fully-loaded file.** The smallest real candidate, `mradermacher`'s
      REAP-40B-A3B i1-IQ1_S (7.85 GiB), is still ~1.9x available RAM. This
      is expected to run via `mmap` (llama.cpp's default): the OS pages in
      only the tensors actually touched (always-active tensors every token,
      routed-expert tensors only when selected), so *resident* RAM can be
      well under the file size — but with 512 experts and 10 used/token,
      most experts get touched within the first few dozen tokens, so page
      **thrashing against on-device flash** (not swap) is the realistic
      risk, not an outright OOM. This must be measured empirically in Phase
      C, not assumed. **Working candidate: `mradermacher/Qwen3-Coder-Next-REAP-40B-A3B-i1-GGUF`
      → `Qwen3-Coder-Next-REAP-40B-A3B.i1-IQ1_S.gguf`.**
- [x] **Verify GGUF metadata without downloading the full file.** Used an
      HTTP range request (`curl -r 0-33554431`, i.e. just the first 32 MiB —
      GGUF stores all KV metadata and tensor name/shape/offset info before
      the tensor data payload) against the resolve URL, then built a local
      **sparse file** (`os.truncate` to the real file size after writing the
      32 MiB header) so `gguf-py`'s `GGUFReader` — which `mmap`s the whole
      file and validates tensor shapes against it — could parse the real
      header without downloading 7.85 GB. Confirmed for the IQ1_S candidate:
      - `general.architecture` = `qwen3next` ✓
      - `qwen3next.expert_count` = 256 (pruned from the base model's 512 —
        consistent with REAP expert pruning)
      - `qwen3next.expert_used_count` = 10
      - 48 transformer blocks (`blk.0` … `blk.47`), 843 tensors total
      - **No `qwen3next.nextn_predict_layers` KV key. Zero tensors matching
        `nextn` anywhere in the file.**
      Repeated the same probe against `bartowski/Qwen_Qwen3-Coder-Next-GGUF`'s
      unpruned `IQ1_S` (16.58 GB) as a control: same result — 48 blocks, 843
      tensors, zero `nextn.*` tensors. Confirmed via `Qwen/Qwen3-Coder-Next`'s
      own `config.json` (the original safetensors repo, not a GGUF): standard
      `Qwen3NextForCausalLM` config, 48 `num_hidden_layers`, 512
      `num_experts`/10 `num_experts_per_tok` — **no NextN/MTP field at all.**
      This is upstream, not a conversion artifact. Recorded as
      [Decision D4](#decisions): MTP is not usable for this model; pillar 3
      falls back to `--spec-type ngram-simple`.
- [x] **VTCM & NPU routing-capacity check**, reading
      `ggml/src/ggml-hexagon/ggml-hexagon.cpp` directly rather than running
      on-device (no model loaded yet):
      - `sess->vtcm_size` is queried from real hardware via
        `htp_iface_hwinfo()` (falls back to a hardcoded 8 MiB only if that
        query fails) — it is the Hexagon DSP's on-chip scratchpad, a few MiB.
      - The `MUL_MAT_ID` kernel (lines ~3874-3924) treats this as a **tiling
        budget, not a capacity ceiling**: it shrinks `block_elems`/`block_rows`
        until the per-call working set fits, only giving up once the block
        shrinks below a floor (128 elements / 1 row). For this model's shapes
        (hidden 2048, moe_intermediate 512, 10 experts/token), that floor is
        very unlikely to bind — i.e. **the VTCM check is not expected to be
        the thing that blocks NPU-resident MoE routing**, correcting the
        Phase 0 guess.
      - The actual ceiling on how much weight data can be made NPU-resident
        at once is `opt_mbuf`, the max single fastRPC/DMA buffer size,
        **default 1024 MiB**, overridable via the `GGML_HEXAGON_MBUF`
        environment variable (MiB units). This — not VTCM — is the knob to
        sweep in Phase C alongside `--n-cpu-moe`.
- [ ] Confirm via a built `llama-cli --help` (once the Phase B checkout is
      built) that `draft-mtp`/`ngram-simple` both still list as expected for
      this architecture, and that `GGML_HEXAGON_MBUF` is recognized at
      runtime (not just present in source).

**Phase A verdict:** proceed to Phase B with the `mradermacher`
REAP-40B-A3B i1-IQ1_S GGUF as the primary candidate, `ngram-simple` as the
pillar-3 speculative-decoding target (not `draft-mtp`), and `GGML_HEXAGON_MBUF`
as the APLS tuning knob to sweep alongside `--n-cpu-moe`.

---

## Phase B — Implementation & setup

- [ ] Clone second checkout: `.work-qwen3/llama.cpp` at a recent upstream
      commit (does not need to match the pinned `e358d591`; record whatever
      commit is used here in `.codex_state.json`, not in `upstream.json`).
- [ ] Build with Hexagon NPU + OpenCL backends enabled (reuse
      `install-npu.sh`'s SDK/toolchain setup logic as reference, but do not
      invoke the installer against this checkout — build manually or via a
      new script under `scripts/`).
- [ ] Download `mradermacher/Qwen3-Coder-Next-REAP-40B-A3B-i1-GGUF`'s
      `Qwen3-Coder-Next-REAP-40B-A3B.i1-IQ1_S.gguf` (7.85 GiB, Phase A
      candidate) into `~/models/` (outside the repo, consistent with
      existing convention).
- [ ] Configure APLS: `--n-cpu-moe N` sweeps (N = 0, partial, all routed
      experts) crossed with `GGML_HEXAGON_MBUF` sweeps (default 1024 MiB —
      try smaller/larger), keeping attention/dense/shared-experts
      NPU/GPU-resident. This is the real tuning surface per Phase A, not VTCM.
- [ ] Configure pillar 3: `--spec-type ngram-simple` (the model has no MTP
      weights — see Decision D4). Do not spend time on `--spec-type
      draft-mtp` against this checkpoint; it will fail to find the required
      `nextn.*` tensors.
- [ ] Write a benchmark script (new, under `scripts/`) that runs prefill +
      decode under each config and records prompt tok/s, generation tok/s,
      MTP acceptance rate (from llama.cpp's own speculative stats output),
      and peak RSS — appending structured results to `execution_history.log`
      and summary fields to `.codex_state.json`.

---

## Phase C — Live verification & testing

- [ ] Baseline: CPU-only, no speculation, IQ1_S REAP-40B-A3B — tok/s + peak
      RAM, and confirm it loads/generates coherent text at all at this bit
      width before investing in acceleration tuning.
- [ ] NPU/GPU dense+attention resident, CPU-streamed experts (APLS), no
      speculation — tok/s + peak RAM, compare to baseline.
- [ ] APLS + `ngram-simple` speculative decoding — tok/s, acceptance rate,
      peak RAM.
- [ ] Measure page-fault/thrashing behavior directly (not just inferred from
      file-size-vs-RAM): watch RSS and I/O wait during a long generation to
      see whether the 7.85 GiB file mmap'd against ~4.2 GiB available RAM
      actually thrashes, and how badly.
- [ ] Sweep `--n-cpu-moe` × `GGML_HEXAGON_MBUF` with real measurements (per
      Phase A, VTCM is not expected to bind — confirm or refute that here).
- [ ] Quality spot-check at IQ1_S: is output coherent at all? 1-bit-class
      quantization on a REAP-pruned (already lossy) MoE is a real risk of
      unusable output — this is not a formal eval, just a sanity check
      before reporting any speed numbers as meaningful.
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
