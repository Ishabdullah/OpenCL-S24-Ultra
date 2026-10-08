# OpenCL-S24-Ultra

**Source-built llama.cpp inference on the S24 Ultra: CPU, Adreno OpenCL GPU, and a separate Hexagon v75 NPU installer.**

Builds a pinned llama.cpp revision with Android `sphal` vendor-runtime loading. `install.sh` builds CPU + GPU; `install-npu.sh` separately builds CPU + NPU. Tested on **Samsung Galaxy S24 Ultra, Snapdragon 8 Gen 3 / Adreno 750, Android 16, F-Droid Termux 0.118.3, aarch64**.

**The current generic OpenCL configuration is slower than CPU on the tested phone.** This project proves working GPU computation; it is not optimized production acceleration.

| Original matched test, 4 threads | CPU | OpenCL |
|---|---:|---:|
| Prompt processing, 128 tokens | 122.74 tok/s | 82.45 tok/s |
| Generation, 64 tokens | 28.72 tok/s | 9.50 tok/s |

OpenCL was 32.8% slower for prompt processing and 66.9% slower for generation. Measurements and reproduction details: [Benchmarks](docs/BENCHMARKS.md).

## Performance investigation checkpoint: October 5, 2026

The [CPU/OpenCL/NPU/hybrid report](OPENCL_PERFORMANCE_ANALYSIS.md), [October 5 measurements and helper source](reports/2026-10-05/), and [remaining work](reports/2026-10-05/NEXT_WORK.md) are available. **The requested three tests are complete; testing is paused.** Broader characterization remains incomplete. The [October 4 checkpoint](reports/2026-10-04/) is preserved. Power epochs, scheduling exclusions, profiling and warm versus cold-start latency remain separate.

The latest generic OpenCL screen used **Qwen2.5-Coder-1.5B Q4_K_M, an 8192-token prompt and 128 outputs**, six CPU threads, batch 512 / microbatch 256, F16 KV and Flash Attention requested ON:

| Mode | Prefill tok/s | Decode tok/s | Warm response seconds |
|---|---:|---:|---:|
| CPU | 66.14 | 11.72 | 134.81 |
| OpenCL | 42.01 | 3.53 | 231.07 |
| OpenCL-prefill / CPU-decode | 34.65 | 2.54 | 286.83 |

All three completed, and all 128 generated tokens matched in this repetitive test. **CPU was fastest in this screen.** It is one run per mode in GPU/hybrid/CPU order, with different admitted starting temperatures; no repeated or universal winner is claimed. Source inspection shows that the current backend declines mixed F32/F16 Flash Attention on Adreno 750, causing CPU attention fallback even with 29/29 layers offloaded. Its share of the slowdown is not yet measured. The 4B 8K candidate was deferred by the conservative memory budget, not an observed allocation failure. Installer and kernel defaults are unchanged.

A separate NPU-prefill/CPU-decode prototype beat CPU and full NPU in **three repeated Mistral 7B tests with a 2048-token prompt and 128 outputs**: mean warm response **116.13 seconds CPU, 39.68 full NPU, 29.17 hybrid**. This is a bounded resident-model benefit, distinct from the negative OpenCL screen. It does not establish generation-only, energy or sustained thermal superiority. These experimental 7B changes are **not installer defaults or a new clean-room installer validation**. No final generic OpenCL crossover, extra accelerator RAM or general thermal winner is established. The [NPU guide](docs/NPU_INSTALL.md) covers the separately validated source-build installer.

## Qwen3-Coder-Next MoE triad checkpoint: October 8, 2026

A separate track pivoted to **Qwen3-Coder-Next** (a 48-layer, 256-expert MoE
with hybrid full-attention/linear-recurrent layers), targeting ultra-low-bit
compression, NPU-resident attention with CPU/RAM-streamed MoE experts, and
speculative decoding. Full plan, decisions, and reproduction steps:
[docs/QWEN3_TRIAD_PLAN.md](docs/QWEN3_TRIAD_PLAN.md). This uses the same
`install-npu.sh`-built CPU+Hexagon binaries as above, plus one additional
incrementally-built `llama-cli` target; it does not touch the GPU/NPU
installers' pins.

Model: [`mradermacher/Qwen3-Coder-Next-REAP-40B-A3B-i1-GGUF`](https://huggingface.co/mradermacher/Qwen3-Coder-Next-REAP-40B-A3B-i1-GGUF)
IQ1_S (7.85 GiB, expert-pruned from 512 to 256 experts). No MTP/NextN
weights exist for this model (confirmed against the upstream checkpoint, not
just this GGUF), so speculative decoding used `ngram-simple` rather than the
originally planned `draft-mtp`.

| Config (`llama-bench`, `-p 64 -n 16 -r 1`) | Prefill tok/s | Decode tok/s |
|---|---:|---:|
| CPU only | 6.69 | 1.21 |
| NPU, `--n-cpu-moe 48` (MoE on CPU) | 7.95 | 1.33 |
| NPU, `--n-cpu-moe 24` | 7.99 | 2.22 |
| NPU, `--n-cpu-moe 0` (MoE through NPU too) | 8.28 | 2.30 |

One run per row, same prompt/output length, same device, same session —
not a repeated or cross-device claim. The NPU rows **require** `-lm mmap`:
the Hexagon backend reports no mmap support, and llama.cpp's default `auto`
load mode disables mmap for the *entire* model the instant any `-dev`
device lacks it, forcing an eager full-file read that overflows this
device's RAM in seconds. `-lm mmap` bypasses that. `ngram-simple`
speculative decoding ran successfully on the best config (3.5/2.1 tok/s on
a short prompt, not directly comparable to the table above). At IQ1_S,
`--repeat-penalty 1.1 --repeat-last-n 64` is required for coherent output;
without it, generation degenerates into a repetition loop.

**The table above (default thread count) is not the whole picture — thread
count flips which backend wins.** At `-t 6` specifically: full CPU reached
**8.17 / 3.30 tok/s**, beating full NPU's **7.44 / 1.92 tok/s** at the same
thread count. The earlier table used `llama-bench`'s default thread count,
under which NPU won; at 6 threads, CPU wins instead. Report the thread
count alongside any of these numbers — it is not a minor tuning detail.

A true **NPU-prefill / CPU-decode handoff works for this model** — but not
via llama.cpp's generic `--prompt-cache` session mechanism, which cannot
reuse a cached prompt prefix when the model has recurrent memory (confirmed
hardcoded in `tools/completion/completion.cpp`). This repo already proved a
working in-process handoff for Mistral 7B (see the October 5 checkpoint
above: 29.17s hybrid vs. 116.13s CPU / 39.68s full NPU) via a local,
never-upstreamed patch adding `llama_perf_switch_to_cpu()`. That patch was
gated to plain Qwen2/Qwen3 attention-only KV; extending it to also migrate
`qwen3next`'s recurrent (SSM/GDN) memory state
([`reports/2026-10-08/patches/`](reports/2026-10-08/patches/)) worked on the
first live attempt — zero crashes, both memory types migrated correctly, no
prompt replay.

| `-t 6` | Prefill tok/s | Decode tok/s |
|---|---:|---:|
| Full CPU | 8.17 | **3.30** |
| Full NPU | **7.44** | 1.92 |
| NPU-prefill → handoff → CPU-decode | 4.71 | 0.79 |

Unlike the Mistral 7B case, the hybrid handoff is **not a speed win here**:
post-handoff decode is markedly slower than native full-CPU, because the
handoff uses canonical (non-repacked) CPU weight shadows rather than the
`CPU_REPACK`-optimized layout a native load uses — the same tradeoff
documented for Mistral 7B, apparently much more punishing for this model's
IQ1_S quantization.

**The full originally-envisioned design — pin attention/shared-experts/dense
on NPU, stream routed experts from CPU, speculative decoding on top —
performs worst of everything tested.** `--n-cpu-moe 48` (pin only the
always-active ~3B params on NPU) plus `ngram-simple` speculative decoding
(substituting for `draft-mtp`, which cannot exist for this model — no
MTP/NextN weights were ever shipped for it) reached only **2.8 / 0.9
tok/s**, the slowest decode of any NPU configuration in this track. The
best decode throughput found remains plain `--n-cpu-moe 0` with no
speculation: **2.30 tok/s**.

A separate, pre-existing, intermittent heap-corruption-on-exit affects
`llama-completion`/`llama-bench`/`llama-cli` on this build (reproduces even
on a bare `--help` call with no model) — unrelated to this model or the NPU
work above, not yet root-caused, and not blocking: it fires after a run's
output has already printed. Tracked in `.codex_state.json`.

## NPU installation: fresh Termux to model generation

For the separate CPU + Hexagon v75 NPU build, use:

```sh
pkg update
pkg install git
git clone https://github.com/Ishabdullah/OpenCL-S24-Ultra.git
cd OpenCL-S24-Ultra
./install-npu.sh --download-test-model
```

This installs missing Termux dependencies, prepares checksum-pinned SDK/compiler tools in project storage, patches pinned llama.cpp, source-builds native ARM64/Hexagon binaries, executes numerical DSP tests, and generates 32 tokens from the pinned official Qwen 1.5B GGUF. QEMU emulates compiler tools only; inference is native. No root or system/vendor changes are required. Allow at least 8 GiB free plus model space and tens of minutes for the build.

Use an existing model instead with `./install-npu.sh --model /path/to/model.gguf`, then launch it through `./scripts/run-npu.sh /path/to/model.gguf`. A model-free install verifies DSP matrices but does not test LLM generation. The optional model download is explicit; no model/SDK/compiler/vendor binary is hosted in this repository.

**Start here: [NPU installation, operation, licensing and troubleshooting](docs/NPU_INSTALL.md).** A fresh source/build reproduction passed eight numerical DSP tests and generated text with all 29 Qwen 1.5B layers offloaded: [NPU validation](docs/NPU_VALIDATION.md). NPU inference is experimental: no universal performance/thermal benefit or arbitrary model/device compatibility is claimed. The GPU and NPU installers coexist in separate directories and do not yet provide one validated CPU/GPU/NPU executable.

## GPU requirements and compatibility

- Ordinary, non-root Termux installed from [F-Droid](https://f-droid.org/packages/com.termux/).
- Android aarch64 with readable Qualcomm libraries: `libOpenCL.so`, `libOpenCL_adreno.so`, `libCB.so`, `libgsl.so` in `/vendor/lib64`.
- Network access for Termux packages and pinned upstream source; allow several GB for source/build and additional space for your model. Keep the project in Termux private storage, not shared Android storage.
- Your own GGUF model for the GPU installer. The separate NPU installer offers an explicit pinned test-model download. No GGUF or proprietary Qualcomm library is distributed in this repository.

**Confirmed:** the S24 Ultra configuration above. **Detected but unverified:** other phones whose vendor stack initializes and exposes an Adreno device. **Potentially compatible:** other Snapdragon/Adreno devices with the required namespace APIs and stack; file presence alone is not proof of compatibility.

## GPU installation

In a fresh F-Droid Termux terminal:

```sh
pkg update
pkg install git
git clone https://github.com/Ishabdullah/OpenCL-S24-Ultra.git
cd OpenCL-S24-Ultra
./install.sh
```

The installer installs missing `git clang cmake ninja python opencl-headers openssl` packages. Clang pulls in LLVM (`llvm-readelf`); Python/Termux provide supporting runtime dependencies. Make, pkg-config and an OpenCL ICD loader are not required by this Ninja/source build.

It downloads upstream llama.cpp at **`e358d59178377be4c58ba567925e05faadbccb57`**, checks/applies the maintained patch, builds `llama-completion`, `llama-bench`, and `test-backend-ops`, verifies ELF dependencies, and detects the GPU. It does not modify an existing `~/llama.cpp`, install llama libraries into `$PREFIX`, use root, or modify Android system/vendor files.

Generated source, binaries, cache and logs live under **`.work/` inside this repository**. First kernel compilation and the source build take time; stages print the relevant log path. A successful installer run without a model proves device initialization, **not model inference**.

Rerunning `./install.sh` checks an existing installation. Options:

```sh
./install.sh --help
./install.sh --verify
./install.sh --rebuild             # configure and incrementally rebuild
./install.sh --clean-build         # replace only the generated build directory
./install.sh --jobs 2              # lower build concurrency
```

Local source modifications are rejected rather than silently reset. Upstream is pinned; arbitrary HEAD updates are unsupported.

## Run your GGUF

Put models outside `.work/`, for example under `~/models/`. Obtain them from a source you trust and comply with their licenses. The validated model is Qwen2.5-Coder-1.5B-Instruct Q4_K_M; other models have not been validated here.

```sh
./scripts/run-model.sh /path/to/model.gguf
```

The launcher uses `llama-completion`; models with a chat template normally enter conversation mode. For a deterministic single raw prompt:

```sh
./scripts/run-model.sh /path/to/model.gguf \
  -no-cnv -p "def add(a, b):" -n 32 --temp 0 --seed 1234
```

Defaults are `-ngl 99 -c 512 -b 128 -ub 128 -t 4 -fa off -n 128`. Additional llama arguments pass through as separate arguments and override defaults, for example `-ngl 10` for less offload, `-c 1024`, or `-st -p "Explain this code"` for a single chat turn. Not every model fits GPU-accessible memory. See [Troubleshooting](docs/TROUBLESHOOTING.md).

The script derives paths from its own location and replaces `LD_LIBRARY_PATH` with **only its own build/bin directory**. Do not append `$PREFIX/lib`, `/system/lib64`, or `/vendor/lib64`. This avoids the vendor binder dependency trap and mixed llama libraries.

## Confirm GPU use

```sh
./scripts/check-device.sh
./scripts/verify-opencl.sh
./scripts/verify-opencl.sh --model /path/to/model.gguf
./scripts/verify-opencl.sh --matrix-tests
```

The model-free verifier checks the architecture, vendor files, exact patched source, build configuration, ELF dependencies, successful `InitOpenCLDriver() = 0`, and an Adreno OpenCL device. With a model it also requires **positive layer offload, OpenCL model storage, completed generation evaluations, and exit status 0**.

On the tested model the logs showed `QUALCOMM Adreno(TM) 750`, 5542 MiB, `offloaded 29/29 layers`, and about 739 MiB OpenCL weights. This does not mean every operation executes on GPU: embedding and large output-projection paths have CPU fallbacks. Separate profiling measured 20,539 actual GPU kernel executions.

Logs and machine-readable summaries are saved in `.work/logs/`.

## Compare performance

```sh
./scripts/benchmark.sh /path/to/model.gguf
./scripts/benchmark.sh /path/to/model.gguf \
  --threads 4 --prompt-tokens 128 --generation-tokens 64 --repetitions 3
```

CPU and OpenCL run sequentially against the same model, with matching thread counts, prompt/generation lengths, batch sizes, f16 KV and attention settings. Profiling must be OFF. Raw JSON, commands, standard deviations and a comparison summary are retained. A negative percentage means OpenCL is slower. Avoid concurrent builds/inference; thermal state and phone scheduling affect results.

## Limitations

- Specialized Adreno kernels and binary kernels remain **OFF** in this release. An isolated narrow Q4_K launch passed later numerical gates, but larger performance/stability confirmation and the original submission-failure explanation remain incomplete; see the [investigation report](OPENCL_PERFORMANCE_ANALYSIS.md).
- Generic OpenCL inference currently loses to the tested CPU baseline.
- CPU fallbacks remain; layer assignment is not proof of exclusive GPU execution.
- Short deterministic CPU/GPU output matched exactly, but bit-identical inference is not promised. Identical-prefix logit validation agreed on 127/128 top tokens, all logits were finite, and mean KL divergence was about 0.000242.
- Broad model, context-length and device compatibility are unverified. Private Android linker APIs/vendor entry points may differ after OS updates.
- Flash Attention is disabled in the launch/benchmark defaults; ordinary attention is the baseline.

## Remove generated installation

```sh
./uninstall.sh                     # dry run
./uninstall.sh --yes               # remove only this repository's .work
```

Shared Termux packages and models outside `.work/` are retained. Uninstall protects user GGUFs inside `.work/`. Only unchanged vocabulary fixtures tracked at the pinned upstream commit are treated as generated source data.

## Technical documentation and credits

- [NPU installation and operation](docs/NPU_INSTALL.md)
- [NPU installer clean-room validation](docs/NPU_VALIDATION.md)
- [How GPU loading works](docs/HOW_IT_WORKS.md)
- [Troubleshooting](docs/TROUBLESHOOTING.md)
- [Benchmarks](docs/BENCHMARKS.md)
- [Technical notes and build pin](docs/TECHNICAL_NOTES.md)
- [Development and validation](docs/DEVELOPMENT.md)
- [Clean-room validation](docs/VALIDATION.md)

Built on [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) and Qualcomm's device-provided OpenCL stack. The upstream MIT license is preserved in [docs/LLAMA_CPP_LICENSE.txt](docs/LLAMA_CPP_LICENSE.txt). Original scripts/documentation and patch contributions use [MIT](LICENSE); model and proprietary vendor-library licenses remain separate. No upstream endorsement or universal Qualcomm compatibility is claimed.
