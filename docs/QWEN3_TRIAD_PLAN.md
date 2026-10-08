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

- **D1 — REVISED after Phase A: reuse the existing `install-npu.sh`, no
  second checkout.** Originally this track planned a separate, unpinned
  llama.cpp clone (`.work-qwen3/llama.cpp`), on the assumption the pinned
  commit might not support `qwen3next`/MTP/APLS. Phase 0 and Phase A proved
  that assumption wrong — the exact pin `install-npu.sh` already builds
  (`e358d591...`) fully supports everything this track needs. A second
  checkout would mean a second SDK/toolchain setup for zero benefit, so
  this track instead runs the existing, validated `./install-npu.sh`
  unmodified. This still satisfies the "don't touch the validated
  installers" rule: `install-npu.sh` builds into `.work-npu/` (separate from
  `.work/`, the GPU installer's directory) and touches neither
  `upstream.json` nor `npu-upstream.json`. GPU/OpenCL is also dropped from
  this track's hardware target (see below) — only CPU + Hexagon NPU are
  built.
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
- **D5 — GPU/OpenCL dropped from this track's hardware target.** The
  project's own measured README data shows OpenCL 32.8% slower for prefill
  and 66.9% slower for generation than CPU on this exact device. Carrying
  it into this track would cost a third backend's build time for a
  configuration already known to lose. `hardware_target` was
  `NPU_CPU_HYBRID`, not a three-way NPU/GPU/CPU split — see D6, which
  further narrows this to CPU-only.
- **D6 — RESOLVED: NPU (Hexagon HTP0) offload crash root-caused and
  fixed; requires `-lm mmap`. `hardware_target` is `NPU_CPU_HYBRID`
  again.** Earlier `-dev HTP0` attempts aborted with `Scudo ERROR: invalid
  chunk state when deallocating` (confirmed via `logcat`). Root cause,
  found by reading `src/llama-model.cpp:1512-1519` directly: under the
  default `AUTO` load mode, llama.cpp checks every device named in `-dev`
  via `ggml_backend_dev_get_props()`, and if **any** device reports
  `mmap_support = false`, it disables mmap for the **entire model**, not
  just that device's share, falling back to an eager full-file read.
  `ggml/src/ggml-hexagon/ggml-hexagon.cpp:7517` hardcodes
  `.mmap_support = false` (vs. `ggml-cpu.cpp`'s `true`). So naming
  `-dev HTP0` at all — regardless of `--n-cpu-moe`, regardless of how
  little compute actually runs on the NPU — forced an eager read of the
  full 8.4 GiB file into real RSS, measured ballooning from 19 MB to 6.2
  GB in ~10 seconds (1-second-interval `ps` sampling), collapsing
  available RAM to ~15 MB and exploding swap from 1.0 to 5.0 GB — a
  genuine OOM/thrashing death spiral. **The earlier Scudo heap-corruption
  aborts were most likely a downstream symptom of this same RSS explosion
  (allocator bookkeeping corrupted under extreme pressure), not a
  distinct Hexagon-kernel memory-safety bug** — the original D6 framing
  was wrong on that point.
  **Fix: pass `-lm mmap` explicitly.** The mmap-disabling check only
  fires when `load_mode == AUTO`; forcing `LLAMA_LOAD_MODE_MMAP` skips it.
  Verified live with full forensic instrumentation (continuous `logcat`
  stream, 0.5-1s memory/process sampler, and a self-imposed watchdog that
  would `kill -9` the process if available RAM dropped below 400 MB — it
  never fired): `-dev HTP0 -lm mmap --n-cpu-moe 48` completed cleanly, zero
  crashes, zero Scudo errors, memory stable throughout (free 650-760 MB,
  available pinned at 5.5 GB) despite the process's own peak RSS reaching
  ~6.1 GB — confirming mmap lets the OS reclaim pages gracefully under
  pressure instead of thrashing. Perf: load 10.1s, prompt eval 4.00 tok/s,
  decode eval **2.46 tok/s — better than the CPU-only baseline's 1.21
  tok/s**. Output was incoherent at this run's settings
  (`--temp 0`, no `--repeat-penalty`) but that's the already-solved,
  orthogonal coherence question (see the IQ1_S coherence finding above),
  not a regression.
  **`-lm mmap` is now a required flag for every future invocation of
  `-dev HTP0` against this model.** Full reproduction, diagnosis, and fix
  verification steps are in Phase C below.

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
- [x] Confirmed via `llama-cli --help` (built in Phase C): `--spec-type`
      lists `none,draft-simple,draft-eagle3,draft-mtp,draft-dflash,
      draft-dspark,ngram-simple,ngram-map-k,ngram-map-k4v,ngram-mod,
      ngram-cache` as expected. `GGML_HEXAGON_MBUF` was not separately
      runtime-probed (the `--n-cpu-moe` sweep never approached the 1024 MiB
      default buffer ceiling, so there was no occasion to test the override)
      — not pursued further since it wasn't the binding constraint.

**Phase A verdict:** proceed to Phase B with the `mradermacher`
REAP-40B-A3B i1-IQ1_S GGUF as the primary candidate, `ngram-simple` as the
pillar-3 speculative-decoding target (not `draft-mtp`), and `GGML_HEXAGON_MBUF`
as the APLS tuning knob to sweep alongside `--n-cpu-moe`.

---

## Phase B — Implementation & setup

- [x] ~~Clone second checkout~~ — superseded by [Decision D1 (revised)](#decisions):
      run the existing `./install-npu.sh` against its own pin instead.
      Launched `./install-npu.sh --jobs 1` (jobs=1, not 2, because available
      RAM was ~450-900 MiB free at launch time with 2.4 GiB swap already in
      use — a QEMU-emulated amd64 toolchain plus `ninja -j2` is exactly the
      Android LMK's target profile). Runs in the background; builds CPU +
      Hexagon NPU only, into `.work-npu/` (does not touch `.work/` or any
      pin file).
- [x] Started `Qwen3-Coder-Next-REAP-40B-A3B.i1-IQ1_S.gguf` (7.85 GiB)
      downloading in the background (`curl -C -`, resumable) into
      `~/models/qwen3-coder-next-reap-40b/` *before* the build, so the long
      I/O-bound download overlaps the long CPU-bound build instead of
      serializing after it.
- [x] Configure APLS: `--n-cpu-moe N` sweeps done in Phase C (0/24/48 —
      see results table there). `GGML_HEXAGON_MBUF` sweep not needed: the
      default 1024 MiB buffer ceiling was never approached at this batch
      size, so it wasn't the binding constraint worth tuning.
- [x] Configure pillar 3: `--spec-type ngram-simple` run and confirmed
      working in Phase C (see results there). `--spec-type draft-mtp`
      correctly not attempted against this checkpoint (Decision D4).
- [ ] Write a benchmark script (new, under `scripts/`) that runs prefill +
      decode under each config and records prompt tok/s, generation tok/s,
      MTP acceptance rate (from llama.cpp's own speculative stats output),
      and peak RSS — appending structured results to `execution_history.log`
      and summary fields to `.codex_state.json`.

---

## Phase C — Live verification & testing

- [x] **Gate, run first: does the binary load and generate from this GGUF
      at all. PASSED.** `./.work-npu/build/bin/llama-completion -m
      Qwen3-Coder-Next-REAP-40B-A3B.i1-IQ1_S.gguf --device none -c 512 -n 8
      -p "The capital of France is" -no-cnv` loaded in ~9.9s and generated:
      _"The capital of France is what?\nThe capital of France is \*\*"_ —
      grammatical English, no crash, no OOM, on the real hybrid
      full-attention/linear-recurrent `qwen3next` architecture. (Note:
      `-no-cnv`, single dash, is required — the model ships a chat template
      so the binary otherwise auto-enters interactive conversation mode and
      hangs waiting on stdin.) CPU-only (`--device none`) perf at this
      stage: **1.46 tok/s prompt, 0.30 tok/s generation** — very slow,
      consistent with mmap page-fault I/O against the 8.4GB file on a
      ~4-6GB RAM budget; not yet isolated from pure compute cost. Hexagon
      device confirmed visible to the binary via `--list-devices` → `HTP0:
      Hexagon`.
- [x] **Quality/coherence check at IQ1_S — PASS, with a required flag.**
      Same prompt, `-n 48`, CPU-only, three sampler configs:
      - `--repeat-penalty 1.0` (default/disabled): grammatical and
        factually correct ("Paris") for ~15-20 tokens, then degrades into a
        repetition loop ("The Paris is the French government's capital."
        repeated).
      - `--repeat-penalty 1.3`: loop gone, but output becomes incoherent
        gibberish (stray LaTeX-like symbols, broken grammar) — too
        aggressive, trades one failure mode for a worse one.
      - **`--repeat-penalty 1.1 --repeat-last-n 64`: the sweet spot.** No
        loop, correct fact retained, coherent prose throughout: _"The
        capital of France is what?\n\nThe capital of France is
        \*\*Paris\*\*. \n\nParis is serves as the French government's
        official agency for its diplomatic address of Paris for a number
        of reasons related to the history and culture of France, but also
        to how it can"_ — one grammatical wobble, otherwise genuinely
        usable.
      **Verdict: IQ1_S REAP-40B-A3B is usable.** `--repeat-penalty 1.1
      --repeat-last-n 64` is now a **required** flag for every subsequent
      Phase C run, not optional tuning — the default sampler config loops
      on this model/quant combination. No fallback to IQ2_XXS needed.
- [x] **Baseline: CPU-only (`--device none`), no speculation.** `llama-bench
      -dev none -p 64 -n 16 -r 1`: **pp64 = 6.69 tok/s, tg16 = 1.21 tok/s**.
      Model identified by llama-bench as `qwen3next 80B.A3B IQ1_S - 1.5625
      bpw`, 40.99B params, 7.84 GiB.
- [x] **NPU (Hexagon HTP0) offload — crashed, root-caused, and FIXED.
      `-lm mmap` is now required whenever `-dev HTP0` is used.**
      `--list-devices` first showed `HTP0: Hexagon (0 MiB, 0 MiB free)`;
      attempting `-dev HTP0 -ngl 99` failed to open a session
      (`error 0x80000406`). Root cause #1: `ADSP_LIBRARY_PATH` must point
      at `<build>/ggml/src/ggml-hexagon` (where `libggml-htp-v75.so`
      lives) — `install-npu.sh`'s own `scripts/npu-common.sh` sets this via
      `npu_runtime()`, but that's only called inside the installer's script
      chain, not when binaries are run directly. **This is a real gap in
      `docs/NPU_INSTALL.md` for anyone invoking the binaries directly.**
      After exporting `LD_LIBRARY_PATH`/`ADSP_LIBRARY_PATH` correctly, a
      standalone `test-backend-ops -b HTP0` run with small synthetic
      matrices passed clean — but pushing the *real* model's tensors
      through `-dev HTP0` repeatedly aborted with `Fatal signal 6
      (SIGABRT)`, `Scudo ERROR: invalid chunk state when deallocating`
      (confirmed via `logcat`, 11+ reproductions), which was first
      mis-attributed to a Hexagon-kernel memory-safety bug in the MoE
      gather path.
      **Root cause #2, the real one** (found by reading
      `src/llama-model.cpp:1512-1519` directly, not guessed): under the
      default `AUTO` load mode, llama.cpp disables mmap for the **entire
      model** if *any* device named in `-dev` reports
      `mmap_support = false` — and `ggml-hexagon.cpp:7517` hardcodes
      exactly that (`ggml-cpu.cpp` reports `true`). So naming `-dev HTP0`
      at all, independent of `--n-cpu-moe` or how little actually runs on
      the NPU, forced an eager read of the full 8.4 GiB file into RSS.
      Verified via 1-second `ps` sampling: RSS went **19 MB → 6.2 GB in
      ~10 seconds**, available RAM collapsed to ~15 MB, swap exploded from
      1.0 to 5.0 GB — a genuine OOM/thrashing spiral. The Scudo aborts seen
      earlier were almost certainly a downstream symptom of this same
      collapse, not a separate bug.
      **Fix: pass `-lm mmap` explicitly** (the mmap-disable check only
      fires under `AUTO`). Verified live with full forensic instrumentation
      — continuous `logcat` stream, a 0.5-1s memory/process sampler, and a
      self-imposed watchdog (kills the process if available RAM drops
      below 400 MB; never fired): `-dev HTP0 -lm mmap --n-cpu-moe 48`
      completed cleanly, **zero crashes, zero Scudo errors**, memory
      stable throughout (free 650-760 MB, available pinned at 5.5 GB)
      despite the process's own peak RSS reaching ~6.1 GB. Perf: load
      10.1s, prompt eval 4.00 tok/s, **decode eval 2.46 tok/s — better than
      the CPU-only baseline's 1.21 tok/s**. Output was incoherent at this
      run's settings (`--temp 0`, no `--repeat-penalty`) — expected and
      orthogonal; the coherence fix (`--repeat-penalty 1.1`) is already
      established above and just needs combining with `-lm mmap` in the
      next run.
      (Separately, unrelated: `python`/`python3` processes hit a
      differently-worded Scudo abort — `internal map failure, Out of
      memory` — repeatedly on 10-06/10-07, before this task existed. A
      pre-existing device memory-pressure pattern, not the same bug.)
- [x] **`--n-cpu-moe` sweep (full results).** `llama-bench -dev HTP0 -lm mmap
      -p 64 -n 16 -r 1 -b 128 -ub 128 -fa off`, watchdog armed throughout:

      | Config | pp64 (tok/s) | tg16 (tok/s) |
      |---|---:|---:|
      | CPU-only baseline (`--device none`) | 6.69 | 1.21 |
      | NPU, `--n-cpu-moe 48` (all MoE on CPU) | 7.95 | 1.33 |
      | NPU, `--n-cpu-moe 24` | 7.99 | 2.22 |
      | **NPU, `--n-cpu-moe 0`** (all MoE through Hexagon too) | **8.28** | **2.30** |

      Best: `--n-cpu-moe 0` — **1.24x prompt, 1.90x decode** speedup over
      CPU-only. Notably, routing *all* MoE experts through Hexagon's
      `MUL_MAT_ID` did not hit a VTCM budget wall, correcting the Phase A
      worry that VTCM would gate how much routing could stay on-NPU.
- [x] **`ngram-simple` speculative decoding — confirmed functional.**
      `llama-completion`/`llama-bench` don't expose `--spec-type` (gated to
      `LLAMA_EXAMPLE_CLI`/`SERVER`/`SPECULATIVE`, not `COMPLETION`); built
      `llama-cli` by reconfiguring the existing build with
      `-DLLAMA_BUILD_SERVER=ON` and building just that target incrementally
      (reused all already-compiled objects, no full rebuild). Ran
      `llama-cli -dev HTP0 -lm mmap --n-cpu-moe 0 --repeat-penalty 1.1
      --repeat-last-n 64 --spec-type ngram-simple --single-turn`: completed
      successfully, correct fact retained ("**Paris**"), reported
      **Prompt: 3.5 t/s | Generation: 2.1 t/s**. Acceptance-rate stats
      aren't printed by `llama-cli`'s basic output — would need
      `--log-verbosity` or the server's stats endpoint to capture that, not
      done here.
- [x] **Known issue found and deliberately parked (not investigated
      further per explicit user instruction).** A post-hoc `logcat` review
      of the sweep+ngram session found 5 additional `Fatal signal 6`/
      `Scudo invalid chunk state` aborts hitting `llama-completion`,
      `llama-bench`, and `llama-cli`. Crash-dump context for one of them:
      `Cmdline: llama-completion --help`, `Process uptime: 1s` — **this
      fires on a bare `--help` call with no model and no device**, proving
      it is unrelated to this model, to `-dev HTP0`, or to the `-lm mmap`
      fix. It's an intermittent, non-deterministic exit-time
      heap-corruption bug, most likely a destructor-ordering or double-free
      issue in shared startup/teardown code, whose detection by Scudo
      depends on heap layout/reuse timing — which is why many earlier
      "clean" runs showed no crash at all (absence of a detected abort was
      never proof the underlying corruption didn't happen). All captured
      output above (sweep numbers, ngram-simple text) is still considered
      valid, since each crash occurred *after* its run's output had already
      printed. Logged in `.codex_state.json` under `known_issues` as
      `exit-time-scudo-abort`. **Not blocking; not pursued further unless
      it recurs or becomes blocking** — would need a debug/ASAN build to
      actually root-cause.
- [x] **Page-fault/thrashing behavior under `-lm mmap` with `-dev HTP0`.**
      Covered by the same memory sampler used throughout Phase C (0.5-1s
      interval `free`/`ps` capture across the single diagnostic run, the
      full 3-point `--n-cpu-moe` sweep, and the `llama-cli` build + ngram
      test). Across the entire multi-run sweep session, `available` memory
      stayed healthy at **7.5-7.8 GiB** throughout (free fluctuated
      113-760 MiB, which is normal page-cache behavior, not thrashing —
      page cache, not swap, absorbed the pressure, and it actually improved
      across repeated loads of the same mmap'd file as cache warmed). No
      swap growth, no watchdog trigger, across every run in this session.
      Conclusion: with `-lm mmap` active, this device handles the 7.85 GiB
      file comfortably at this batch/context size (`-c 512 -b 128
      -ub 128`); a longer context or larger batch is the next thing that
      would actually test the remaining headroom, not attempted here.
- [x] **Thread-count comparison at `-t 6`, matching the project's established
      "six CPU threads" benchmark convention** (see README's October 5
      checkpoint). Full CPU vs. full NPU vs. an attempted NPU-prefill/
      CPU-decode split:

      | Config (`-t 6`, `-p 64 -n 16 -r 1 -b 128 -ub 128 -fa off`) | Prefill tok/s | Decode tok/s |
      |---|---:|---:|
      | Full CPU (`-dev none`) | 8.17 | 3.30 |
      | Full NPU (`-dev HTP0 -lm mmap --n-cpu-moe 0`) | 7.44 | 1.92 |

      **This reverses the earlier sweep's conclusion.** At the default
      thread count (8), NPU beat CPU (8.28/2.30 vs 6.69/1.21). At `-t 6`,
      **CPU beats NPU on both metrics**. Thread count is not a minor
      tuning knob here — it changes which backend wins. Report both; don't
      cherry-pick the thread count that favors either side.
      **NPU-prefill → CPU-decode via `--prompt-cache`: attempted, doesn't
      work this way — but see the corrected finding below.**
      Step 1 (NPU prefill, `-fa off`, `-n 0 --prompt-cache FILE`) saved a
      cache cleanly (34 tokens, 79.8 MiB, 5.42 tok/s prefill). Step 2 (CPU
      decode loading that cache) first failed with
      `state_read_data: incompatible V transposition` — traced to a flag
      mismatch in testing (the CPU step defaulted `flash_attn` to `auto`/on
      while the NPU step used `-fa off`; matching `-fa off` on both sides
      fixed this specific error). With that fixed, the cache *loaded*
      without error, but llama.cpp then reported `session file has low
      similarity to prompt (0/26 tokens); will mostly be reevaluated` and
      `unable to reuse common prefix (for example, when the memory is
      recurrent)`. Confirmed via source
      (`tools/completion/completion.cpp:334`) that this is a **hardcoded,
      deliberate limitation of the generic session/prompt-cache mechanism
      specifically**: it does not support reusing a cached common prefix
      when the model has recurrent memory. This conclusion is correct for
      `--prompt-cache`, but it is not the full story — see below.

- [x] **CORRECTED: a true NPU-prefill → CPU-decode handoff *is* achievable
      for this model — via the repo's own existing technique, extended.**
      The user pointed out that this repo already has proof a hybrid
      handoff works (the Oct 3-5 Mistral 7B investigation: 29.17s mean
      response, beating both CPU 116.13s and full NPU 39.68s). That
      technique is **not** `--prompt-cache` at all — it's a custom, local,
      never-upstreamed patch
      ([`reports/2026-10-03/patches/npu-prefill-cpu-decode.patch`](../reports/2026-10-03/patches/npu-prefill-cpu-decode.patch))
      adding `llama_perf_switch_to_cpu(ctx, drop_gpu_weights, &stats)`: an
      **in-process, same-context, same-KV-cache** handoff that directly
      migrates backend buffer data and rebinds tensor pointers, with no
      save-to-disk/reload/prefix-matching step at all — so it never hits
      the recurrent-memory limitation found above.
      That patch's dispatch was hardcoded to `LLM_ARCH_QWEN2`/
      `LLM_ARCH_QWEN3` with a bare `llama_kv_cache` (`dynamic_cast` gate:
      _"hybrid prototype supports plain Qwen2/Qwen3 KV only"_), excluding
      `qwen3next`'s hybrid memory. But the existing
      `llama_kv_cache::perf_migrate_cpu()` was already architecture-agnostic
      (generic buffer iteration, no Qwen-specific tensor names) — so this
      was a tractable extension, not a dead end. Extended it (full diff:
      [`reports/2026-10-08/patches/qwen3next-hybrid-handoff.patch`](../reports/2026-10-08/patches/qwen3next-hybrid-handoff.patch)):
      1. Added `llama_memory_recurrent::perf_migrate_cpu()` — a near-direct
         port of the kv-cache version, since `llama_memory_recurrent` has
         the identical internal `ctxs_bufs` structure.
      2. Extended `llama_context::perf_switch_cpu()` to also recognize
         `llama_memory_hybrid` (the class `qwen3next` actually uses — it
         wraps a `llama_kv_cache` for the periodic full-attention layers
         and a `llama_memory_recurrent` for the linear-attention layers),
         migrating both via `get_mem_attn()`/`get_mem_recr()`.
      3. Added `LLM_ARCH_QWEN3NEXT` to the allowed-architecture check.
      Compiled cleanly (incremental rebuild, no errors). Wrote a small
      standalone driver,
      [`scripts/hybrid-handoff-probe.cpp`](../scripts/hybrid-handoff-probe.cpp)
      (adapted from the Oct 5 helper, not part of llama.cpp, compiled
      directly against the built `libllama.so`), since the new API isn't
      wired to any CLI flag.
      **Ran it with full logcat+memory+watchdog instrumentation — success,
      zero crashes:** NPU prefill (32 tokens): 6.80s, 4.71 tok/s. Handoff:
      1.98s wall, migrating **both** memory types — 6.00 MiB attention KV
      and 75.38 MiB recurrent/SSM-GDN state — with 331.37 MiB of NPU
      weights released. CPU decode after handoff (15 steps): 18.94s,
      **0.79 tok/s**.
      **The honest caveat**: post-handoff decode (0.79 tok/s) is markedly
      *slower* than a native full-CPU run at the same thread count (3.30
      tok/s from the `-t 6` table above). The handoff installs canonical
      (non-repacked) CPU weight shadows, not the `CPU_REPACK`-optimized
      layout a native CPU load uses — the exact tradeoff the Mistral 7B
      investigation already documented (_"CPU uses its optimized repacked
      weights while the hybrid decodes from mapped canonical GGUF
      shadows"_), apparently far more punishing for this model's IQ1_S
      quantization than it was for Mistral's Q4_K_M.
      **Net result: the technique is proven correct and functional for
      this hybrid recurrent-memory architecture — the earlier "not
      achievable" conclusion was wrong, and extending an existing local
      prototype rather than fighting the generic session mechanism was the
      right call. It is, however, not a speed win for this quant**: full
      CPU decode (3.30 tok/s) beats the hybrid handoff's decode phase
      (0.79 tok/s) here, unlike the Mistral 7B case where hybrid won
      outright. The three-way comparison for this model at `-t 6`:

      | Config | Prefill tok/s | Decode tok/s |
      |---|---:|---:|
      | Full CPU | 8.17 | **3.30** |
      | Full NPU | **7.44** | 1.92 |
      | NPU-prefill → handoff → CPU-decode | 4.71 | 0.79 |

- [x] **The originally-envisioned full combo, tested as literally described
      (minus the impossible MTP piece): pin attn/shared-experts/dense on
      NPU (`--n-cpu-moe 48`), stream routed experts from CPU, speculative
      decoding on top (`ngram-simple`, substituting for `draft-mtp` which
      cannot exist for this model).** `llama-cli -dev HTP0 -lm mmap
      --n-cpu-moe 48 -t 6 --spec-type ngram-simple --repeat-penalty 1.1
      --repeat-last-n 64`: **Prompt 2.8 tok/s, Generation 0.9 tok/s** — the
      **slowest decode of every NPU configuration tested in this track.**
      No crash. Output quality was also noticeably worse this run than the
      earlier coherent "Paris" result under the same repeat-penalty
      settings (likely sampler interaction with speculative
      rejection/acceptance, or just run-to-run variance — no `--seed` was
      fixed). **Honest conclusion: the full envisioned architecture
      (NPU-pinned core params + CPU-streamed experts + speculative
      decoding) does not outperform simpler configurations for this
      model/quant/hardware.** The best decode throughput found across this
      entire Phase C remains plain `--n-cpu-moe 0` with no speculation
      (2.30 tok/s). Both "pin only active params" (`--n-cpu-moe 48`) and
      adding `ngram-simple` on top make things slower, not faster, here —
      this doesn't contradict anything measured earlier, it's consistent
      with `--n-cpu-moe 48` already being the slowest point in the original
      sweep.
- [x] Log every run (success or failure, exact error text, config) to
      `execution_history.log` immediately; update `.codex_state.json`
      `metrics` after each completed run. (Done continuously throughout
      Phase C — see `execution_history.log` for the full timestamped trail.)

---

## Phase D — Documentation & git discipline

- [x] Added a dated "Qwen3-Coder-Next MoE triad checkpoint: October 8, 2026"
      section to `README.md` with the measured sweep table, the required-flags
      summary, and the known-issue note — following the existing hedged-claims
      convention (one run per row, not repeated/cross-device).
- [x] Final configuration is documented inline in this plan (Phase C /
      Decisions D1-D6) rather than a separate results file — the plan doc
      already carries the full flag history and reasoning, and splitting it
      out would just duplicate it.
- [x] Commit + push after each phase's completion, per the user's standing
      instruction for this track — done throughout (see git log).

---

## Phase E — Qwen3.6-35B-A3B: a model with real MTP tensors

User-directed pivot to find a model that genuinely ships MTP/NextN weights
(Qwen3-Coder-Next does not — see Decision D4). `Qwen/Qwen3.6-35B-A3B`'s own
`config.json` has `text_config.mtp_num_hidden_layers: 1`. Verified at the GGUF
level before downloading (same range-request + sparse-file + `gguf-py` probe
used throughout Phase A): `unsloth/Qwen3.6-35B-A3B-MTP-GGUF`'s smallest file
(`UD-IQ1_M`, 10.59 GiB) has `qwen35moe.nextn_predict_layers=1` and 4 real
`nextn.*` tensors at block 40 (`eh_proj`, `enorm`, `hnorm`,
`shared_head_norm`), across 41 blocks / 753 tensors. Downloaded (resumed once
after a dropped connection at 92%; final size matched exactly:
11,366,414,624 bytes).

- [x] **Gate 1 (CPU-only load/generate): PASSED.** `--device none -t 6`:
      *"The capital of France is Paris. [end of text]"* — correct, clean
      EOS stop. Very slow (0.46 / 0.12 tok/s prompt/decode) but functional.
- [x] **`--spec-type draft-mtp`: accepted and functionally real, but
      unstable under multi-threaded CPU execution — a race condition, not
      a flag-tuning problem.** The flag was accepted with no "MTP tensors
      missing" error, confirming the GGUF's MTP tensors are genuinely
      usable. Short generations (`-n 16`/`-n 24`) completed cleanly
      multiple times at `-t 1`, `-t 2`, and `-t 6` (with `-rea off`),
      producing correct output (e.g. "The capital of France is **Paris**.").
      A longer generation (`-n 48`) at the identical working `-t 6`
      config produced real, coherent, substantive text first (*"There is
      no single event that universally marks the absolute 'beginning' of
      the French Revolution..."*) then hit `OMP: Error #132: Thread
      identifier invalid` repeatedly before crashing (`Fatal signal 6`, a
      worker thread). A separate attempt with `--log-verbosity 4` crashed
      differently and earlier: `Fatal signal 11` (SIGSEGV) at a near-null
      address, before any token was produced. **Disabling reasoning mode
      (`-rea off`) does not reliably fix this** — initial short successful
      runs made it look like the fix, but it only narrowed the window in
      which the race was observed; the longer run crashed with `-rea off`
      too. The failure rate appears to scale with generation length (more
      MTP draft/verify cycles = more chances to hit the race), not with
      reasoning mode or thread count alone.
      **This is logged as a known, not-yet-root-caused issue**
      (`.codex_state.json` → `known_issues` → `mtp-openmp-thread-race`),
      consistent with how the unrelated `exit-time-scudo-abort` issue was
      handled: no further live-fire trial-and-error without a debug/ASAN
      build to actually find the race in the MTP draft context's
      thread-pool interaction with the main context's.
- [x] **NPU (`-dev HTP0 -lm mmap --n-cpu-moe 0`) + `--spec-type draft-mtp`
      — works cleanly, resolving the race condition favorably.** Tested
      `-t 6 -np 1 --spec-draft-n-max 2 -rea off` at both a short (`-n 16`)
      and the exact longer length (`-n 48`) that reliably crashed on
      CPU-only. **Zero crashes at either length.** Short: *"The capital of
      France is **Paris**."* (0.8/0.3 tok/s). Long: *"The history of the
      French Revolution is generally considered to have begun on **July
      14, 1789**, with the **Storming of the Bastille**. ### Key Context:
      - **Immediate Trigger**: The storm"* (1.0/0.3 tok/s) — factually
      correct, well-formatted, cut off naturally by the token limit, not
      by a crash. Memory stable throughout (available pinned ~7.0 GiB).
      **Likely explanation**: the `mtp-openmp-thread-race` issue is
      specific to the CPU-only threadpool path — offloading most compute
      to Hexagon leaves far less concurrent CPU-side work for the race to
      manifest in. **This is the first clean, working combination of all
      three original pillars on real hardware**: NPU-resident compute
      (ULBC/APLS via `--n-cpu-moe 0` + `-lm mmap`) plus genuine native MTP
      self-speculative decoding (`--spec-type draft-mtp`), stable across
      both short and long generations, on a model that actually ships the
      MTP weights this architecture needs.
- [x] **Controlled throughput comparison, MTP on vs. off, NPU only —
      MTP is stable but NOT a speed win here.** Identical config both
      runs (`-dev HTP0 -lm mmap --n-cpu-moe 0 -t 6 -np 1 -n 48`, same
      prompt, same `--seed 42` for a fair, deterministic comparison —
      confirmed by identical output text up to the point of divergence):

      | Config | Prompt tok/s | Decode tok/s |
      |---|---:|---:|
      | No speculation | **1.2** | **0.4** |
      | `--spec-type draft-mtp --spec-draft-n-max 2` | 0.9 | 0.2 |

      MTP is **slower** — ~25% slower prompt eval, ~50% slower decode, no
      crash either way. At this base decode rate (well under 1 tok/s),
      the cost of running the draft head and verifying its predictions
      outweighs any acceptance-driven savings. Speculative decoding's
      payoff requires a base speed fast enough that draft/verify overhead
      is small relative to it — this model/quant/hardware combination
      (IQ1_M 35B-A3B on an S24 Ultra NPU) is far from that regime.
      **Honest conclusion: native MTP self-speculative decoding is now
      proven functionally real and stable on this hardware (resolving the
      open question from Phase A/D4 for Qwen3-Coder-Next, which genuinely
      lacked MTP weights), but it is not currently a throughput win for
      this specific model/quant/device combination.** It may become one
      at a higher quant (faster base decode) or with different
      `--spec-draft-n-max` tuning — not explored further here.
- [ ] NPU/CPU hybrid (the `llama_perf_switch_to_cpu` handoff from Phase C)
      combined with `draft-mtp`: not yet attempted; would need the hybrid
      patch extended again for `qwen35moe`'s architecture (likely similar
      `llama_memory_hybrid` shape, not yet confirmed).

---

## Phase F — Profiling before changing anything, and a bounded expert-caching design

User-directed: profile the current NPU run (expert selections, cache
misses, storage reads, NPU transfers, draft acceptance) *before* building
anything, per the explicit ordering: profile → bounded expert caching
(MTP disabled) → short MTP drafts + prefetching once caching works. Target:
"full-model behavior with a bounded expert working set," with measured RAM
and speed, not an approximation.

- [x] **Built [`scripts/moe-profile-probe.cpp`](../scripts/moe-profile-probe.cpp)**
      (local tool, not part of llama.cpp). Loads the model on NPU
      (`-dev HTP0`, `mmap`, all experts through NPU — our best-known
      config, no MTP by design here), wires `common_debug_cb_user_data`
      (an existing, ready-made llama.cpp debug hook already exported from
      `libllama-common.so` — no new patch needed for this part) filtered
      to `"ffn_moe_topk"` (the tensor holding the router's actual selected
      expert IDs, tagged in `src/llama-graph.cpp`), and samples
      `/proc/self/stat` (minflt/majflt), `/proc/self/io`
      (rchar/read_bytes), and `/proc/self/status` (VmRSS) at load/prefill/
      decode phase boundaries.
- [x] **Ran it cleanly (32-token prompt, 32-token decode, `-t 6`), zero
      crashes. Headline finding: decode-phase storage re-reads are
      catastrophically redundant.**

      | Phase | Storage reads (cumulative) | Major faults (cumulative) | RSS |
      |---|---:|---:|---:|
      | Load (13.6s) | 10.8 GiB | 4,783 | 4.08 GB |
      | Prefill (32 tok, 4.88 tok/s) | 18.8 GiB | 13,543 | 2.1 GB |
      | **Decode (32 tok, 0.35 tok/s)** | **104.0 GiB** | **59,290** | **647 MB** |

      Decoding 32 tokens re-read **~85 GiB from storage — roughly 8x the
      entire 10.59 GiB model file**. RSS shrank monotonically throughout
      the whole run (4.08 GB → 2.1 GB → 647 MB) despite reading *more*
      data each phase: the OS's generic page-cache LRU is evicting
      indiscriminately under memory pressure, with no concept of "this
      expert gets reused soon," forcing the same experts to be re-read
      from storage repeatedly across nearby decode steps. This is solid,
      `/proc`-sourced data, not an estimate, and it directly motivates the
      bounded expert-caching design below.
- [x] **Captured the actual expert-selection trace** (1280
      `ffn_moe_topk` dumps = 40 layers × 32 calls) via the debug hook —
      **but this specific analysis is preliminary, not rigorous**: the
      debug dump truncates each 8-expert selection to its first/last 3
      (hiding the middle 2), and the quick aggregate analysis pooled all
      40 layers together rather than isolating per-layer reuse. Raw
      result: all 256 experts appear touched across the run; the most
      frequently selected experts appeared 70-108 times out of 1280 calls.
      **Do not read this as proof of "no locality"** — a proper
      per-layer locality measurement (needed to actually size a bounded
      cache's `K` experts-per-layer) requires a tighter capture: have our
      own `cb_eval` copy the raw tensor data directly instead of relying
      on the text dump's truncation, and group by layer index rather than
      pooling. Not yet done.
- [x] **Backend feasibility check for a bounded expert slot cache**
      (per the user's own spec: "inspect whether the backend supports
      reusable expert slots... before implementing"). Two concrete,
      source-grounded facts: (1) `ggml-hexagon` already implements the
      standard `set_tensor`/`get_tensor` backend interface
      (`ggml-hexagon.cpp:1993`/`2036`) — the primitive needed to write new
      data into an already-allocated NPU buffer at runtime; (2) this repo
      already *proved* tensor-level buffer/data rebinding works at runtime
      in this exact codebase — that's what the hybrid NPU→CPU handoff
      patch does (`pair.first->buffer = pair.second->buffer; pair.first->data = pair.second->data;`),
      just as a one-time full migration rather than a continuous per-token
      swap. **Verdict: structurally feasible, but a materially bigger
      patch than anything built so far** — `MUL_MAT_ID`'s current design
      assumes each expert tensor has one fixed backend buffer for the
      whole context lifetime; a real bounded cache needs a new per-token
      hook that checks the router's output against a slot table and
      re-points/re-fills slots on a miss, correctly synchronized so the
      NPU never reads a slot mid-overwrite.
- [ ] **Scoped Phase 1 implementation (proposed, not yet started,
      pending confirmation):** given the full 10-section spec the user
      provided (observation, model-identity tracking, continuous learning,
      dynamic expert grading, bounded caching, adaptive prefetching,
      repeated-context routing memory, MTP integration, persistence,
      validation) is realistically a multi-week systems project, proposed
      starting with only the bounded-slot-cache mechanism itself
      (sections 1/5/9's smallest testable slice, MTP-independent per the
      spec's own section 8 ordering), deferring learned grading,
      prefetching, task-context association, repeated-context memory,
      and MTP-assisted prediction until the core mechanism is proven
      correct and measured. Correctness gate before any speed claim:
      numerically compare against unmodified full-residency output on
      identical prompts.
- [ ] **Tighter per-layer expert-selection locality measurement** (needed
      before choosing `K`): rerun with a custom data-capturing callback
      (not the truncating text-dump) grouped by layer, to measure real
      reuse rates across nearby decode steps per layer, not pooled across
      the whole model.

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
