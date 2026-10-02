# v0.1.0 clean-room validation

Validated on 2026-10-02 on Samsung Galaxy S24 Ultra SM-S928U, Snapdragon 8 Gen 3 / QTI SM8650, Adreno 750, Android 16, F-Droid Termux 0.118.3, aarch64. Compiler: Termux clang 21.1.8, default Android 24 target.

## Scope of the clean room

The standalone project started with no managed source, build or kernel cache. `./install.sh` fetched upstream directly from GitHub, checked out `e358d59178377be4c58ba567925e05faadbccb57`, applied the packaged five-file patch, configured and compiled 331 build steps. No experimental build binaries, Git alternates, or compiled kernel-cache artifacts were used. Validation ran against this project's `.work/build/bin` exclusively.

The existing Android/Termux package environment was retained, not factory reset. All seven required packages were already installed. Automatic missing-package installation and installer rerun behavior were additionally tested using isolated command fixtures; package removal/reinstallation was unnecessary. Thus this establishes a fresh project/source/build reproduction on the tested device, not an empirical test of every future mirror or Termux package version.

Patch version 1 SHA-256: `dc21507d755969fbfad95b6493a78ffdb30892a52a5d26cb1f10dc2a7af4a483`. All five resulting source files matched the actual working experiment byte-for-byte. Source hashes are recorded in `upstream.json`.

## Completed checks

| Check | Result |
|---|---|
| Fresh installer / source fetch / patch / configure / build | Exit 0 |
| Subsequent ordinary installer run | Exit 0; reused source/build and verified again |
| `./install.sh --rebuild` | Exit 0; recognized already-applied patch and rebuilt incrementally |
| Runtime ELF/configuration validation | 11 project binaries/libraries checked; no OpenCL DT_NEEDED, OPENCL_* requirement or unresolved cl* APIs |
| Namespace driver initialization | `sphal InitOpenCLDriver() = 0` |
| Device enumeration | `GPUOpenCL: QUALCOMM Adreno(TM) 750 (5542 MiB, 4518 MiB free)` |
| GGUF inference verification | 29/29 layers offloaded, 739.03 MiB OpenCL weights, completed 32-token generation, exit 0 |
| KV and compute storage | 14.00 MiB OpenCL KV, 15.38 MiB OpenCL compute at context 512 |
| Quantized matrix correctness | 46/46 targeted Q4_K/Q6_K MUL_MAT tests passed against CPU |
| Launcher environment/argument handling | Passed with inherited `$PREFIX/lib` and unrelated backend path; caller outside repository |
| Runtime loaded-library inspection | All seven mapped llama/ggml libraries came from this project's build/bin |
| CPU/GPU deterministic raw generation | 32-token stdout matched exactly, both exited 0 |
| Longer GPU run | 128-token requested generation completed, exit 0 |
| Script regression suite | 10 tests passed, including false-success rejection, unsafe ELF linkage, package installation, argument quoting and GGUF protection |
| Shell syntax | All distributed shell scripts passed `bash -n` |
| Uninstall | Real installation dry run passed; isolated destructive tests retained unowned directories/user models and recognized unchanged pinned vocabulary fixtures |
| Original `~/llama.cpp` regression | 109,483 path metadata records, original Git status and HEAD unchanged |

The model was Qwen2.5-Coder-1.5B-Instruct Q4_K_M, GGUF SHA-256 `cc324af070c2ecbfd324a30884d2f951a7ff756aba85cb811a6ec436933bb046`. It remains outside the project and is not distributed.

Inference used `-ngl 99 -dev GPUOpenCL -c 512 -b 128 -ub 128 -t 4 -fa off -n 32 --temp 0 --seed 1234 -no-cnv -p "def add(a, b):"`. The output began:

```python
def add(a, b):
    return a + b

def subtract(a, b):
    return a - b

def multiply(a, b):
    return a * b
```

There were 31 timed decode evaluations for the 32-token request; the first generated token follows prompt evaluation. Device detection and actual model offload remain separate verification stages. The earlier profiling evidence of 20,539 executed kernels is documented in [TECHNICAL_NOTES.md](TECHNICAL_NOTES.md); profiling was OFF for the reproduced baseline and benchmarks.

## Fresh-build matched benchmark

`./scripts/benchmark.sh MODEL` used four threads, pp128, tg64, three repetitions, batch/ubatch 128, f16 KV, Flash Attention OFF, generic kernels, profiling OFF. CPU and GPU ran sequentially with matching metadata and no simultaneous source build or inference. Means +/- sample standard deviations:

| Test | CPU tok/s | OpenCL tok/s | OpenCL vs CPU |
|---|---:|---:|---:|
| pp128 | 119.75 +/- 7.70 | 83.90 +/- 0.52 | 29.9% slower |
| tg64 | 23.92 +/- 2.68 | 8.62 +/- 0.71 | 63.9% slower |

Mobile thermal/clock/scheduler state was not pinned. This independently reproduces functional GPU inference and the observed performance disadvantage, rather than claiming a speed improvement. Earlier measurements remain in [BENCHMARKS.md](BENCHMARKS.md).

## Evidence and limits

A concise machine-readable release record is included in [validation-v0.1.0.json](validation-v0.1.0.json). Full local logs, stdout, build logs, per-repetition benchmark JSON/commands and mapped-library paths are retained under ignored `.work/logs/`. Large/generated logs, binaries, downloaded upstream source, models and proprietary libraries are excluded from Git.

This validates one device/OS/model configuration, targeted numerical tests and short generations. It does not establish arbitrary context/model compatibility or production stability. Specialized Adreno kernels stay disabled, CPU fallbacks remain, and longer CPU/GPU text may diverge from floating-point differences.
