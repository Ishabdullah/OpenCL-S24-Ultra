# Experimental source snapshots

These diffs were generated from the checked local sources after testing stopped. They are preserved for review and future development, not automatically applied by the released installer. Local commit hashes below identify experimental ancestry; they are not upstream commits and are not published Git objects in this repository.

Start from pinned upstream `e358d59178377be4c58ba567925e05faadbccb57` with the project's existing Qualcomm OpenCL patch. That produces the generic known-good source represented locally by `7ba662d4cae38e0b2474e9bb0024bb4b6b1fc3ad`.

| Patch | Base | Purpose |
|---|---|---|
| `hexagon-sphal-v75.patch` | Generic known-good | Native CDSP sphal loader, remote-handle ABI fix, selected v75 DSP/QAIC/tool-wrapper and bounded-build adaptation. Resulting local checkpoint: `53d539ccc327858498c055909d19468954b90531`. |
| `opencl-prefill-cpu-decode.patch` | Generic known-good | One-way retained-weight/KV migration and CPU-only scheduler prototype. |
| `npu-prefill-cpu-decode.patch` | NPU checkpoint | Same handoff prototype on the NPU source; identical patch bytes because the edited context/model files share their base content. |
| `adreno-isolated-instrumentation-and-narrow-launch.patch` | Generic known-good | Actual isolated specialized worktree diff, including opt-in host/operation instrumentation and Q4_K wide/narrow launch control. |

`manifest.json` records exact base/target descriptions, sizes and SHA-256 hashes. Patch applicability was checked against the stated local bases without compiling or running new tests. Passing an apply check does not establish a standalone build: SDK/tool wrappers, experimental configuration, numerical gates and helpers remain necessary.

The handoff supports one live context, one-way transition and plain Qwen2/Qwen3 KV. It is not a concurrent/general scheduler API. The specialized candidate requires its isolated specialized build and `GGML_OPENCL_Q4K_GEMV_WIDE=0`; do not enable it in the conservative installer based only on this report.

No DSP compiler, SDK library, proprietary vendor binary or model is included. Upstream-derived patch context is under llama.cpp's MIT license, preserved in `docs/LLAMA_CPP_LICENSE.txt`; original contributions use this project's MIT license.
