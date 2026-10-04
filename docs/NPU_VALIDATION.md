# NPU installer reproduction: October 4, 2026 UTC

This is a separate installer reproduction, after the October 3 research report. It uses a fresh project-controlled source, SDK extraction, x86 runtime and build directory. No host/DSP llama build artifacts were copied from the research environment or `~/llama.cpp`.

## Scope and environment

- Samsung SM-S928U / Snapdragon SM8650, Android 16, aarch64 F-Droid Termux 0.118.3.
- Native ARM64 host inference and source-built Hexagon v75 DSP kernels. QEMU is used for x86 SDK build tools only.
- Upstream `https://github.com/ggml-org/llama.cpp.git`, pinned commit `e358d59178377be4c58ba567925e05faadbccb57`.
- Direct patch: `patches/llama.cpp-hexagon-sphal.patch`; exact four patched-file hashes and dependencies in `npu-upstream.json`.
- Fresh local test directory: `~/npu-installer-cleanroom-20261004`. All generated data is in its own `.work-npu/`.

This phone already had Termux packages installed. The installer confirmed the eight required packages rather than reinstalling them. Nine pinned Debian amd64 runtime packages were freshly downloaded and privately extracted. The exact SDK archive was reused through `--sdk-archive`, SHA-256 checked and freshly extracted. The official test GGUF was reused after checksum verification to avoid a duplicate 1 GiB download. The public SDK URL was checked and returned HTTP 200 with the pinned size. This is a fresh **source/build** reproduction on the existing phone, not a factory-reset or newly installed Termux test.

## Commands

The independent test used the same installer as users, with the immutable cached SDK archive option:

```sh
cd ~/npu-installer-cleanroom-20261004
./install-npu.sh --sdk-archive /path/to/hexagon-sdk-v6.6.0.0-amd64-lnx.tar.xz \
  --download-test-model
```

Fresh-install users omit `--sdk-archive`:

```sh
pkg update
pkg install git
git clone https://github.com/Ishabdullah/OpenCL-S24-Ultra.git
cd OpenCL-S24-Ultra
./install-npu.sh --download-test-model
```

## Reproduction fixes

Debian's merged-/usr packages do not supply the root filesystem aliases. The toolchain preparer creates private `lib -> usr/lib` and `lib64 -> usr/lib64` aliases for QEMU's ELF interpreter/runtime lookup. These are inside `.work-npu`, not changes to Termux or Android.

QAIC interface generation stalled with a relative `htp_iface.idl` argument. Bounded controls showed timeouts at 15 and 20 seconds; identical absolute-input commands completed in approximately 3 seconds. The SDK wrapper resolves positional IDL inputs to absolute paths without changing include/output options. Three subsequent repetitions of the original build arguments passed in 3.39, 2.47 and 3.45 seconds. This is an observed workaround; the underlying QAIC/QEMU relative-path behavior has not been fully explained.

Compiler wrappers preserve both SDK target-header/library discovery paths. Nested DSP builds are bounded to two jobs and native builds to two jobs by default. The installer retains logs and does not mark an incomplete build successful.

## Results

| Check | Result |
|---|---|
| Source pin and four patched-file hashes | Passed |
| Build | 30 new DSP translation units; native host binaries completed |
| ELF validation | 12 artifacts; ARM64 host + Hexagon v75 DSP; own-build llama dependencies |
| Vendor access | sphal driver loaded; all 17 required symbols found; architecture 0x8c75 |
| Native DSP numerical computation | 8/8 Q4_K/Q6_K matrix tests passed against CPU reference |
| Model | Official Qwen2.5-Coder-1.5B-Instruct Q4_K_M, pinned SHA-256 |
| LLM offload | 29/29 layers, HTP0 weights 1023.48 MiB |
| Generation | 32 tokens requested; 31 subsequent decode evaluations; successful exit 0 |
| Installer rerun | Exit 0; no SDK/source/configure/build repeated; binary hashes unchanged; matrices passed again |
| Launchers | NPU and CPU-only eight-token generation both exited 0 |
| Runtime isolation | Seven loaded llama/ggml libraries all came from the independent build, despite deliberately inherited conflicting paths; profiling was 0 |
| Uninstall protection | Refused removal with a user-added GGUF; dry run preserved storage |

The output began:

```python
def add(a, b):
    return a + b

def subtract(a, b):
    return a - b
```

At context 512, llama.cpp reported HTP0 KV 14.00 MiB, HTP0 compute 74.94 MiB, CPU model 125.19 MiB, CPU compute 2.00 MiB and CPU output 0.58 MiB. These are allocation reports, not a total RSS measurement or extra physical NPU RAM.

The 32-token smoke run reported 95.22 tok/s for its six-token prompt and 29.22 tok/s for 31 decode evaluations. These are **not matched performance benchmark results** and should not replace the research baselines. The reproduced model, source, artifact hashes and allocation values are in [summary.json](npu-validation/summary.json), [elf.json](npu-validation/elf.json) and the raw [matrix](npu-validation/matrix.txt), [inference](npu-validation/inference-stderr.txt) and [output](npu-validation/inference-stdout.txt) records.

## Evidence and limits

The verification evidence is stored beside this document under `npu-validation/`. Text snapshots have trailing whitespace normalized; the original local logs are retained. Artifact SHA-256 hashes identify the tested build; they are not promises of bit-reproducible binaries across compiler versions or paths. The installer verifies native host/DSP architecture, its own llama dependency paths, and absence of startup CDSP/OpenCL linkage and OpenCL symbol-version requirements.

This short test proves installation and native NPU computation for the selected model. It does not establish arbitrary model compatibility, exact CPU/NPU logits, sustained thermal advantage, or a best practical performance configuration. The prior [performance report](../OPENCL_PERFORMANCE_ANALYSIS.md) documents those unresolved research questions.

The original `~/llama.cpp` was read-only throughout. Its Git HEAD/status and all preserved generic OpenCL source/binary and known-good NPU binary hashes were checked against their earlier records. The broad performance investigation remains paused.
