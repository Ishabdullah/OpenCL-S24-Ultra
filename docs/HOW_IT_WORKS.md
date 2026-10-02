# How the Qualcomm loader works

The process runs with Termux's C++ runtime. Qualcomm vendor libraries need their Android/vendor dependencies. Promoting system/vendor library directories ahead of Termux globally risks ABI collisions, so the backend uses Android's exported **sphal** linker namespace.

Initialization happens once, before device probing:

1. Resolve `__loader_android_get_exported_namespace` and `android_dlopen_ext`.
2. Obtain the `sphal` namespace.
3. Load `/vendor/lib64/libOpenCL_adreno.so` there with `RTLD_NOW | RTLD_LOCAL`.
4. Resolve and call `InitOpenCLDriver()`; require return code 0.
5. Only then load `/vendor/lib64/libOpenCL.so` in the same namespace.
6. Resolve 37 typed OpenCL function pointers and expose the dispatch table only if every lookup succeeds.
7. Probe the platform/device and use the normal ggml OpenCL backend.

Library handles remain loaded for process lifetime. `std::call_once` guards initialization. There is no shim constructor and no startup OpenCL library dependency. Both the backend and its compiled-program cache use the same dispatch.

## Initialization order

The device investigation established that opening both the driver and ICD before driver initialization returns -30, while driver -> initialize -> ICD works. The patch preserves that sequence. These are observed device-specific requirements, not a claim about all OpenCL drivers.

## Why internal dynamic dispatch

Termux's OpenCL loader exports versioned `OPENCL_*` symbols. Qualcomm's ICD does not satisfy those ELF version requirements. Startup linkage to either loader would also undermine control over initialization order. The opt-in `GGML_OPENCL_USE_SPHAL` build therefore discovers headers but does not link an OpenCL library.

All calls are mapped to functions resolved from the vendor ICD handle. The backend contains no OpenCL DT_NEEDED, no OPENCL_* version requirement, and no unresolved cl* API references. Non-sphal upstream builds keep their normal linkage.

## The environment confounder

A trivial extra startup dependency appeared to break driver initialization in earlier tests. Controlled A/B tests disproved that theory: the known-good reference also failed when `LD_LIBRARY_PATH` included `$PREFIX/lib`.

An isolated directory containing only Termux's `libbinder_ndk.so` reproduced the failure. Logcat reported that `libCB.so` could not load through sphal because `/system/lib64/libnativewindow.so` could not satisfy its named binder version dependency. Termux's forwarding binder library was selected rather than the intended Android dependency. Removing the global Termux library path fixed initialization.

An executable RUNPATH and a global LD_LIBRARY_PATH are not interchangeable here. The launchers replace the latter with their own build/bin only. Supporting Termux dependencies remain available through the compiler's normal RUNPATH, while Qualcomm is loaded through sphal.

## Mixed llama libraries

The Termux compiler injects a library RUNPATH before CMake's build path. Existing unversioned tool implementation libraries in the installed Termux prefix were loaded alongside new versioned libraries, producing invalid buffer accounting and Scudo double-free errors.

Using only the project build/bin in LD_LIBRARY_PATH selects one coherent llama build. ELF verification also requires all llama/ggml DT_NEEDED libraries to exist within that directory, so a missing project library cannot quietly fall back to an installed copy.

## Correctness before optimization

The known-good configuration disables specialized and precompiled Adreno kernels. It uses generic OpenCL kernels with ordinary attention and CPU fallbacks. Earlier specialized-path failures reached driver command submission; the exact kernel/layout/lifetime defect was not resolved. Packaging preserves the working baseline.
