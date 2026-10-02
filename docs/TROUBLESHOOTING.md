# Troubleshooting

Start with `./scripts/check-device.sh` and `./install.sh --verify`. Read `.work/logs/configure.log`, `build.log`, `device.log`, and (for model verification) `inference.stderr`. Do not use sudo/root or replace Android/vendor files.

| Symptom | Likely cause and action |
|---|---|
| Wrong architecture | Only Android aarch64 is supported. Check `uname -m`; 32-bit Termux/proot desktop builds are outside this baseline. |
| Missing vendor library | Check the four paths printed by check-device.sh. This is not solved by copying binaries from another phone. |
| `InitOpenCLDriver() = -30` / libCB load failure | Check library-path contamination and initialization order. Use the project launcher. Do not set `LD_LIBRARY_PATH="$PREFIX/lib"`. |
| Android namespace API / sphal unavailable | Vendor/OS namespace configuration differs. This is a compatibility limitation; no root workaround is supplied. |
| Platform/device not found | Read device.log. Successful CPU enumeration is not a GPU pass. Check vendor files, sphal initialization and `clGetPlatformIDs` errors. |
| Patch no longer applies | Confirm upstream.json commit and source HEAD. Do not update arbitrary upstream HEAD or force a half-applied patch. Local modifications are preserved and rejected; use a fresh project clone for a clean install. |
| Scudo double-free / mixed llama versions | Use scripts, not the bare binary. Check `LD_LIBRARY_PATH`, complete project dependencies and `.work/logs/elf.json`. Rebuild with `./install.sh --rebuild`. |
| OpenCL DT_NEEDED or OPENCL_* versions detected | Wrong build configuration/linkage. Run `./install.sh --clean-build`; never link Termux OpenCL or vendor ICD directly. |
| Model does not fit / allocation error | Use a smaller quantized GGUF, lower context `-c` or lower layers `-ngl`. Free memory by closing apps. The reported GPU memory is not a guarantee of usable free allocation capacity. |
| Driver submission failure / exit 137 | Preserve logs. Check specialized kernels are OFF. Exit 137 alone does not distinguish a GPU fault from OS memory/process killing. Consult your own process's logcat entries. |
| Specialized Adreno kernel problems | Both specialization flags must remain OFF for v0.1.0. Kernel optimization is a separate development milestone. |
| TLS segment underaligned | The old build used compiler target overrides. This installer clears imported build flags and uses fresh Termux defaults. Rebuild cleanly; do not reuse old experimental binaries. |
| Package download/build failure | Check network and Termux mirror with `pkg update` / `termux-change-repo`. Retry installer. Lower compile jobs to 2 if needed. Full build logs are retained. |
| Moved project cannot rebuild | CMake embeds source/build paths. Run `./install.sh --clean-build` in its new location. Launcher paths are computed dynamically. |
| Numerical outputs differ | Different arithmetic can change a close argmax and later generation. Deterministic sampling does not imply bit-identical CPU/GPU computation. |
| Very long first startup | The backend compiles many embedded kernels. Subsequent runs use the project-local kernel cache. |

Useful read-only diagnostics after installation:

```sh
git -C .work/llama.cpp rev-parse HEAD
llvm-readelf --dynamic .work/build/bin/libggml-opencl.so.0
llvm-readelf --version-info .work/build/bin/libggml-opencl.so.0
./scripts/verify-opencl.sh --model /path/to/model.gguf
logcat -d -t 300 | rg 'Adreno|vndksupport|libCB|Scudo'
```

`rg` is optional (install `ripgrep` for the last command); logcat access may be limited to your own processes. Do not disable Android security to obtain logs. Project scripts preserve normal Termux LD_PRELOAD behavior; custom preloads are outside the tested environment.

For a bug report, include device/Android/Termux versions, project VERSION, upstream.json, commands, and relevant logs. Remove personal paths, prompts and secrets first.
