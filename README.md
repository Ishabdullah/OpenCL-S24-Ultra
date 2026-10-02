# OpenCL-S24-Ultra

**v0.1.0: reproducible proof of real Qualcomm Adreno GPU inference in Termux.**

Builds a pinned llama.cpp revision with an internal Android `sphal` OpenCL loader. Tested on **Samsung Galaxy S24 Ultra, Snapdragon 8 Gen 3 / Adreno 750, Android 16, F-Droid Termux 0.118.3, aarch64**.

**The current generic OpenCL configuration is slower than CPU on the tested phone.** This project proves working GPU computation; it is not optimized production acceleration.

| Original matched test, 4 threads | CPU | OpenCL |
|---|---:|---:|
| Prompt processing, 128 tokens | 122.74 tok/s | 82.45 tok/s |
| Generation, 64 tokens | 28.72 tok/s | 9.50 tok/s |

OpenCL was 32.8% slower for prompt processing and 66.9% slower for generation. Measurements and reproduction details: [Benchmarks](docs/BENCHMARKS.md).

## Requirements and compatibility

- Ordinary, non-root Termux installed from [F-Droid](https://f-droid.org/packages/com.termux/).
- Android aarch64 with readable Qualcomm libraries: `libOpenCL.so`, `libOpenCL_adreno.so`, `libCB.so`, `libgsl.so` in `/vendor/lib64`.
- Network access for Termux packages and pinned upstream source; allow several GB for source/build and additional space for your model. Keep the project in Termux private storage, not shared Android storage.
- Your own GGUF model. No model or proprietary Qualcomm library is distributed or downloaded by this installer.

**Confirmed:** the S24 Ultra configuration above. **Detected but unverified:** other phones whose vendor stack initializes and exposes an Adreno device. **Potentially compatible:** other Snapdragon/Adreno devices with the required namespace APIs and stack; file presence alone is not proof of compatibility.

## Install

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

- Specialized Adreno kernels and binary kernels remain **OFF**: the earlier specialized path caused a Qualcomm driver submission failure. Its precise defect is unresolved.
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

- [How it works](docs/HOW_IT_WORKS.md)
- [Troubleshooting](docs/TROUBLESHOOTING.md)
- [Benchmarks](docs/BENCHMARKS.md)
- [Technical notes and build pin](docs/TECHNICAL_NOTES.md)
- [Development and validation](docs/DEVELOPMENT.md)
- [Clean-room validation](docs/VALIDATION.md)

Built on [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) and Qualcomm's device-provided OpenCL stack. The upstream MIT license is preserved in [docs/LLAMA_CPP_LICENSE.txt](docs/LLAMA_CPP_LICENSE.txt). Original scripts/documentation and patch contributions use [MIT](LICENSE); model and proprietary vendor-library licenses remain separate. No upstream endorsement or universal Qualcomm compatibility is claimed.
