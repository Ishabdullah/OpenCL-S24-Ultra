# Benchmarks and interpretation

Original on-device validation date: 2026-10-02. S24 Ultra SM-S928U, QTI SM8650 / Adreno 750, Android 16, F-Droid Termux 0.118.3. Upstream e358d59178377be4c58ba567925e05faadbccb57 plus patch version 1.

Model: Qwen2.5-Coder-1.5B-Instruct Q4_K_M. GGUF SHA-256:
`cc324af070c2ecbfd324a30884d2f951a7ff756aba85cb811a6ec436933bb046`.

Generic OpenCL, profiler OFF, specialized kernels OFF, f16 KV, ordinary attention, batch/ubatch 128, prompt 128, generation 64, three repetitions. CPU and GPU ran sequentially, without concurrent build/inference. Values are means +/- sample standard deviation.

| Backend | Threads | pp128 tok/s | tg64 tok/s |
|---|---:|---:|---:|
| CPU | 4 | 122.74 +/- 1.84 | 28.72 +/- 4.08 |
| OpenCL | 4 | 82.45 +/- 0.91 | 9.50 +/- 2.24 |
| CPU, additional diagnostic | 8 | 47.84 +/- 3.09 | 7.48 +/- 0.20 |

The matched four-thread GPU run was 32.8% slower for prompt processing, 66.9% slower for generation, or about 3.02x as much generation time. The previous installed CPU observation of 20-22 tok/s was not the matched baseline. This project's original tested source/build achieved 28.72 tok/s CPU under the recorded conditions.

Mobile clock, thermal and scheduler state were not pinned; do not interpret differences between sessions as controlled kernel improvements. Four threads are the conservative tested default, not a universal recommendation for every phone.

## Reproduce

```sh
./scripts/benchmark.sh /path/to/model.gguf
```

Both tests share one generated argument list. The script also compares actual JSON metadata before printing the comparison. CPU uses `-ngl 0 -dev none`; GPU uses `-ngl 99 -dev GPUOpenCL`. The JSON `backends` field can describe compiled/registered backends even in a CPU-only run; use `devices` and `n_gpu_layers` to identify actual test selection.

Results, individual samples, commands and stderr are saved under `.work/logs/benchmark-*/`. Negative percentage means OpenCL is slower. No profiling overhead is included in these benchmarks. Kernel profiling proof is a separate validation run, not a speed comparison.

The project's independent clean-room results are recorded in [VALIDATION.md](VALIDATION.md). They need not numerically match this earlier session.

## Standalone project clean-room benchmark

The new installer's independent build measured CPU pp128 **119.75 +/- 7.70 tok/s**, tg64 **23.92 +/- 2.68 tok/s**; OpenCL pp128 **83.90 +/- 0.52 tok/s**, tg64 **8.62 +/- 0.71 tok/s**. That is **29.9% slower prompt processing** and **63.9% slower generation** with the same four-thread parameters above. Full reproduction scope and checks are in [VALIDATION.md](VALIDATION.md); exact numeric summaries are in [validation-v0.1.0.json](validation-v0.1.0.json).
