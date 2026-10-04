# Attribution and license scope

Original project scripts, documentation and patch contributions are offered under the MIT License in ../LICENSE. Upstream-derived patch context and modifications retain the upstream llama.cpp MIT terms; the full upstream notice is included in LLAMA_CPP_LICENSE.txt.

Upstream: ggml-org/llama.cpp, copyright (c) 2023-2026 The ggml authors, at the pinned commit in upstream.json. The installer retains upstream LICENSE and third-party notices in the downloaded checkout. This repository does not relicense third-party components.

Qualcomm OpenCL libraries are proprietary device-provided components. They are not copied, distributed, or relicensed here. GGUFs are supplied separately by users and remain subject to their respective licenses.

The performance report also documents experimental Hexagon/FastRPC work. Its upstream-derived patch context retains llama.cpp's MIT notice. Original measurement helpers and the evidence-export script use this project's MIT license. Qualcomm Hexagon SDK/compiler components, FastRPC components, emulation runtime components and their individual licenses are separate; no SDK, compiler executable, vendor library, model or build binary is distributed in this repository. The NPU installer downloads checksum-pinned SDK/runtime dependencies into local project storage, retaining their notices; the optional test model is downloaded from its upstream publisher. See NPU_INSTALL.md for sources and license scope. Public availability of an SDK archive does not relicense it under this project's MIT terms.

The device investigation and project are maintained by Ishabdullah. Codex assisted implementation and packaging. No Qualcomm, Samsung, Termux or llama.cpp upstream endorsement is claimed.
