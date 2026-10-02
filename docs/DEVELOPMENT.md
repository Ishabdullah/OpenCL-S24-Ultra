# Development

Read upstream.json, the maintained patch and [HOW_IT_WORKS.md](HOW_IT_WORKS.md) before changing the loader. The patch is an exact distribution of the working five-file modification, including new files. Original implementation and packaging were assisted by Codex and validated on the device.

Do not switch to arbitrary upstream HEAD. For an update, use a separate pinned source checkout, deliberately port the patch, regenerate all patched-file hashes and patch SHA-256, then repeat a clean-room install, ELF checks, device detection, actual inference and matched benchmarks. Update patch version and validation documentation.

Do not use another build's binaries or compiled kernel cache during clean-room validation. Do not modify a user's existing llama.cpp. Source modifications are rejected by installer validation; for development use a separate checkout and keep the distributed baseline reproducible.

## Local checks

```sh
for file in install.sh uninstall.sh scripts/*.sh examples/*.sh; do bash -n "$file"; done
python -m unittest discover -s tests -v
./install.sh --verify
./scripts/verify-opencl.sh --model /path/to/model.gguf --matrix-tests
./scripts/benchmark.sh /path/to/model.gguf
git diff --check
git status --short
```

Source/patched-file hashes and ELF checks run during verification. Numerical tests exercise quantized matmul; model verification exercises generation. Negative checks ensure CPU-only enumeration and missing offload logs do not pass.

## Clean-room procedure

Clone this project into a new directory with no .work, then run ./install.sh. The installer obtains upstream from the network and applies the patch itself. Supply a local GGUF to verify-opencl.sh, run the launcher, run benchmarks, then rerun install.sh to verify idempotency. Check no runtime llama/ggml library is mapped from another installation. Existing Termux packages can be retained; resetting or uninstalling the user's environment is unnecessary.

Record command exit codes, device identity, source/patch hashes, model identity, layer/buffer logs, generation output and benchmark parameters. Keep large/raw logs in ignored .work/logs, and publish a concise validation summary.

## Scope and release discipline

v0.1.0 is a proof-of-concept. Specialized Adreno-kernel optimization is a separate milestone; keep both specialization flags OFF in this baseline. No root, system security changes, binary blobs or bundled GGUFs. License notices must accompany distributed upstream-derived patch material.

Review staged files before committing. .work, binaries, models and generated logs are ignored. The project's GitHub remote is Ishabdullah/OpenCL-S24-Ultra; publishing here does not submit changes to upstream llama.cpp.
