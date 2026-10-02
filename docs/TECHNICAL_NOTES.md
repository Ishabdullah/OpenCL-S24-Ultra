# Technical notes

## Reproducibility pin

- Project: v0.1.0; patch: version 1.
- Upstream: https://github.com/ggml-org/llama.cpp.git
- Commit: e358d59178377be4c58ba567925e05faadbccb57.
- Original validation: 2026-10-02.
- Tested compiler: Termux clang 21.1.8, aarch64-unknown-linux-android24.
- Machine-readable metadata, patch SHA-256 and all five patched-file hashes: upstream.json.

The source patch was copied from the checked working implementation, verified against its exact upstream commit, and reproduced from a fresh network fetch. No source modification was reconstructed from memory.

## Build configuration

scripts/build.sh is the authoritative command. It preserves Release, Ninja, Termux cc/c++, OpenCL ON, sphal ON, target version 300, embedded kernels ON, profiling OFF, backend dynamic discovery OFF, specialized Adreno kernels OFF, binary kernels OFF, app/server OFF, tests ON. Imported CFLAGS/CXXFLAGS/CPPFLAGS/LDFLAGS are cleared; no old compiler target overrides are used.

Targets: llama-completion, llama-bench, test-backend-ops. llama-completion handles interactive chat and raw generation; a separate llama-cli build is not needed for this baseline. HTTPS support remains upstream's default ON; openssl is included in dependencies.

Neither an OpenCL runtime development library nor ocl-icd is needed. opencl-headers supplies CL/cl.h. Proprietary libraries are loaded in place from Android/vendor storage.

## Patch files

1. ggml/src/ggml-opencl/CMakeLists.txt: opt-in header discovery and internal sphal sources; no OpenCL link library.
2. ggml/src/ggml-opencl/opencl-sphal.h: 37 typed API pointers with call-site remapping.
3. ggml/src/ggml-opencl/opencl-sphal.cpp: namespace lookup, ordered one-time initialization, ICD loading, checked symbol resolution.
4. ggml/src/ggml-opencl/ggml-opencl.cpp: initialize dispatch before probing and log enumeration failure.
5. ggml/src/ggml-opencl/cl-program-cache.cpp: use the same dispatch for cached-program loading.

Default upstream OpenCL linking remains unchanged when GGML_OPENCL_USE_SPHAL is OFF.

## Runtime isolation

Every launch replaces LD_LIBRARY_PATH with only the selected project's .work/build/bin. Project dependencies are checked for presence there. Kernel cache is .work/kernel-cache, preventing reuse of another project's diagnostic cache during clean-room testing. Nothing references ~/llama.cpp or ~/llama-opencl-build.

The original dependency-order hypothesis was falsified by controlling the global library path. The binder/version dependency failure and mixed llama-library issue are distinct problems.

## Earlier evidence

- Adreno 750 detected with 5542 MiB reported global memory.
- 29/29 layers assigned to OpenCL, approximately 739 MiB weights, GPU KV and compute buffers.
- Profile: 20,539 timed kernel executions including warmup, 4,158 Q4_K and 462 Q6_K matrix-vector operations; total recorded device kernel time 2272.526 ms.
- 46/46 focused Q4_K/Q6_K numerical MUL_MAT tests passed against CPU.
- Deterministic 32-token output matched CPU; 128-token GPU generation completed.
- Identical-prefix comparison: 127/128 equal top tokens, all logits finite, RMSE 0.083248, maximum absolute error 0.832338, mean KL 0.00024201. One near-tie digit changed argmax.
- CPU embedding/output fallbacks retained about 125 MiB CPU and 196 MiB CPU_REPACK storage. Layer assignment does not mean all operations execute on GPU.

These are bounded tests, not a claim of complete operation/model coverage. Debugging binaries, vendor binaries and model files are intentionally excluded from distribution.
