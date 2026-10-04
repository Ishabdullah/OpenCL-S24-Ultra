# Install native Hexagon NPU inference in Termux

This repository provides a separate NPU source-build installer, `install-npu.sh`, for the tested Samsung Galaxy S24 Ultra / Snapdragon 8 Gen 3 / Hexagon v75 on Android 16 with aarch64 F-Droid Termux. It builds CPU + NPU; the existing `install.sh` builds CPU + Adreno OpenCL. These are separate build directories, not one validated three-backend executable. Other phones/Android releases are unverified.

## Fresh Termux to an LLM on the NPU

Install Termux from [F-Droid](https://f-droid.org/packages/com.termux/), open it, and run:

```sh
pkg update
pkg install git
git clone https://github.com/Ishabdullah/OpenCL-S24-Ultra.git
cd OpenCL-S24-Ultra
./install-npu.sh --download-test-model
```

The explicit model option downloads the checksum-pinned official [Qwen2.5-Coder-1.5B-Instruct Q4_K_M GGUF](https://huggingface.co/Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF), or reuses the already-validated model at `~/models/qwen2.5-coder-1.5b/` if present. It then completes a deterministic 32-token generation test and requires positive HTP0 model/layer offload. The model is Apache-2.0 according to its upstream card; no model is shipped in this Git repository. Downloaded test models live under `~/models/OpenCL-S24-Ultra/`, outside generated build storage.

To use your own existing model instead:

```sh
./install-npu.sh --model /path/to/model.gguf
```

To install without downloading/running a model:

```sh
./install-npu.sh
```

The model-free path verifies the host backend, device and eight actual source-built Q4_K/Q6_K DSP matrix operations against a CPU reference. It does **not** establish that a particular GGUF fits or generates correctly.

## Requirements and what gets installed

Use ordinary Termux private storage, aarch64 Android, and the device's readable `/vendor/lib64/libcdsprpc.so`. Root, system/vendor modifications, disabling security and a separate Linux/proot installation are not needed. The device vendor must allow the necessary unsigned DSP session through its FastRPC broker; file presence alone does not prove access.

Allow at least **8 GiB free plus model storage**. The SDK download is about 662 MiB and unpacks to approximately 3.2 GiB. Source/build/runtime/download storage is additional. The optional test model is about 1.04 GiB. Keep Termux visible during validation; screen-off/background cpuset restrictions were observed in the investigation. Source compilation can take tens of minutes. Default build concurrency is two with nice+10; use `--jobs 1` to reduce load. Memory/temperature guard failures stop the build and leave logs for retry after cooling/closing other apps.

The installer automatically installs missing Termux packages:

```text
git clang cmake ninja python curl openssl qemu-user-x86-64
```

Termux supplies dpkg and package dependencies; clang supplies LLVM tools used for ELF validation. The installer downloads and SHA-256-verifies the pinned [Hexagon SDK 6.6.0.0 public community mirror archive](https://github.com/snapdragon-toolchain/hexagon-sdk/releases/tag/v6.6.0.0), linked to the [Qualcomm Software Center release](https://softwarecenter.qualcomm.com/catalog/item/Hexagon_SDK?version=6.6.0.0), and a small pinned Debian amd64 runtime library set. It extracts those runtime packages into project storage with `dpkg-deb`; they are **not installed into Android or the global Termux prefix**. Debian pool URLs have a content-addressed official snapshot fallback. Versions, URLs, sizes and checksums are in `npu-upstream.json`.

The SDK/compiler and each bundled component retain their own licenses and notices. The SDK Tools `NOTICE.txt` says use is subject to the Qualcomm license and does not itself grant a new license. Consult the supplied notices and Qualcomm terms applicable to your use. Public mirror availability does not relicense the SDK under this project's MIT terms. No SDK/compiler/vendor library is committed here. If you already have the exact archive, avoid a duplicate download:

```sh
./install-npu.sh --sdk-archive /path/to/hexagon-sdk-v6.6.0.0-amd64-lnx.tar.xz --model /path/to/model.gguf
```

Its checksum is still verified. Arbitrary SDK/compiler versions are not supported by that option.

## Architecture and pinned source

The installer fetches upstream llama.cpp commit `e358d59178377be4c58ba567925e05faadbccb57` and applies `patches/llama.cpp-hexagon-sphal.patch` directly. It checks the complete patch digest and exact four patched-file hashes, and refuses unexpected local changes rather than resetting them.

The patch loads device-provided CDSP/FastRPC through Android's exported `sphal` namespace, fixes the 64-bit remote-handle close signature, selects the v75 DSP target, and adapts QAIC/compiler/archiver/linker discovery for isolated SDK-tool emulation. QEMU runs only x86 **build tools**. The host inference executable is native ARM64; the source-built DSP library is native Hexagon v75 with upstream HVX/HMX compute kernels. It does not run an emulated LLM.

Generated data stays in `.work-npu/`:

```text
.work-npu/
  llama.cpp/       exact upstream source + maintained NPU patch
  toolchain/       SDK, isolated x86 runtime, SDK compiler wrappers
  downloads/       verified SDK/runtime download cache
  build/bin/       native ARM64 inference/benchmark/test binaries
  build/ggml/src/ggml-hexagon/libggml-htp-v75.so
  logs/            toolchain, configure, DSP/host builds and validation evidence
  installed.json  installation pin/version and validation completion timestamp
```

The compiler wrappers convert QAIC IDL inputs to absolute paths: the identical relative-input command stalled under the tested QEMU runtime, while absolute inputs completed in about three seconds. They also preserve SDK target-header/library discovery; this is necessary for reproducible DSP linking. Nested build concurrency is bounded, and SDK installation uses `cmake --install` to avoid an unnecessary repeated LTO rebuild.

## Run, verify and CPU fallback

```sh
./scripts/run-npu.sh /path/to/model.gguf
./scripts/run-npu.sh /path/to/model.gguf \
  -no-cnv -p "def add(a, b):" -n 32 --temp 0 --seed 1234
./scripts/verify-npu.sh --model /path/to/model.gguf
./scripts/verify-npu.sh --device-only
```

Defaults are HTP0, `-ngl99 -c512 -b128 -ub128 -t4 -fa off`; additional llama arguments override defaults. For partial offload or CPU-only execution from this same build:

```sh
./scripts/run-npu.sh /path/to/model.gguf -ngl 15
./scripts/run-npu.sh /path/to/model.gguf -ngl 0 -dev none
./scripts/verify-npu.sh --model /path/to/model.gguf --ngl 15
```

The launcher sets LD_LIBRARY_PATH to **only its own build/bin**, ADSP_LIBRARY_PATH to its own DSP output directory, clears LD_PRELOAD/backend overrides and inherited Hexagon tuning variables, and selects profiling 0. Do not append `$PREFIX/lib`, `/system/lib64` or `/vendor/lib64`; this caused vendor binder dependency collisions during the investigation. CPU fallbacks still occur, and a layer-count log is not a claim that every operation executes on HTP.

The verifier checks pinned source, Release/v75 configuration, native host/DSP ELF architecture, own-build llama dependencies, no startup vendor/OpenCL linkage, device enumeration, and numerical matrices. With a model it additionally requires positive layer offload, HTP weights and successful token-generation evaluations from a process that exits 0. HTP0 reporting 0 MiB device memory is expected in this backend and is not itself an initialization failure. Inference logs may say `layers to GPU` because upstream represents HTP as a GPU-type backend; the **HTP0 weights and runtime** identify NPU execution.

## Rerun and uninstall

```sh
./install-npu.sh --help
./install-npu.sh --verify --model /path/to/model.gguf
./install-npu.sh --rebuild
./install-npu.sh --clean-build --jobs 1
./uninstall-npu.sh              # dry run
./uninstall-npu.sh --yes        # generated NPU storage only
```

Reruns preserve source/model files and reuse verified download/extraction state. `--clean-build` replaces only the managed `.work-npu/build`. Uninstall leaves shared Termux packages, the separate GPU build, and models outside `.work-npu` intact. User-added or modified GGUFs inside generated storage must be moved out before removal.

## Performance and limitations

The [investigation report](../OPENCL_PERFORMANCE_ANALYSIS.md) describes experimental measurements, not guaranteed installer speeds. Repeated common-setting 1.5B NPU PP/TG averaged 839.18/32.97 tok/s; 4B 235.66/12.94. CPU baselines and power/thread/attention details are in that report. The best practical CPU decode can exceed those NPU averages under other settings. No universal speed, energy or sustained thermal advantage is claimed.

Full-offload 1.5B and 4B inference and short forced-prefix checks passed locally. NPU logits are not bit-identical to CPU; some 4B logit differences are substantial even with matching short top tokens. Full 7B NPU was safety-skipped under the recorded memory condition; partial 15-layer 7B completed, but that smoke run is not a full-NPU benchmark. Shared RPC/DMA allocations use system RAM and are incompletely reflected in process RSS; HTP adds no extra physical model RAM. Long-context/model combinations require conservative memory admission.

Thermal pilots reached Android SEVERE and stopped. Maximum upstream HTP power corners and shared SoC thermal limits remain research concerns. This installer preserves the measured compute implementation; it does not introduce experimental hybrid handoff, lower power corners or an adaptive scheduler. The broader performance investigation remains paused.

## Recognizable failures

| Failure | What to check |
|---|---|
| Missing libcdsprpc.so / wrong architecture | `./scripts/check-npu.sh`; this installer needs Android aarch64 with the device's Qualcomm CDSP runtime. |
| Sphal unavailable / driver load failure | Read `.work-npu/logs/device.log` and `matrix.log`; firmware namespace/runtime compatibility is unverified outside the tested phone. |
| FASTRPC_GET_DOMAINS query failed, using static CDSP domains | Observed on the tested device; the backend uses its static domain fallback. Judge access by the subsequent native session and matrix/inference tests, not that query alone. |
| HTP0 detected but DSP module/session fails | `ADSP_LIBRARY_PATH` must point to this build's v75 library; inspect `matrix.log`, DSP ELF validation and vendor broker permissions. Detection alone is insufficient. |
| SDK checksum/layout failure | Use the pinned archive/version; do not bypass checksums. Move an incomplete extraction aside and rerun. |
| Pinned Debian runtime download unavailable | Check network and the URLs/checksums in `npu-upstream.json`; the installer tries Debian's content-addressed snapshot fallback. |
| qemu / x86 interpreter / SDK linker missing | Inspect `toolchain.log`; the isolated x86 runtime and both SDK target symlinks must exist. No global LD_LIBRARY_PATH repair is needed. |
| Patch/local source hash failure | Check the exact source pin and intentional local edits. The installer will not reset your work. |
| Build memory/thermal guard stop | Close other apps/cool the phone and rerun; completed source/build work is retained. |
| Model load/allocation failure | Use a smaller GGUF/context or fewer offloaded layers; parameter count alone does not determine KV/runtime memory. |
| Affinity/background restrictions / phone lag | Keep Termux visible, avoid simultaneous demanding apps, use fewer build jobs; no Android throttling bypass is provided. |
| Mixed library/shutdown error | Use `run-npu.sh`, clear inherited overrides, and inspect `logs/elf.json`; don't invoke an installed llama executable with project libraries. |

The independent installer reproduction result is recorded in [NPU validation](NPU_VALIDATION.md). Its build, matrix and inference evidence is separate from the earlier research report.
