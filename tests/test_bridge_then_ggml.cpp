#include <cstdio>
#include <dlfcn.h>

int main() {
    void * bridge = dlopen(
        "/data/data/com.termux/files/home/opencl-sphal-bridge/libOpenCL-unversioned.so",
        RTLD_NOW | RTLD_GLOBAL
    );

    if (!bridge) {
        printf("BRIDGE: FAILED\n%s\n", dlerror());
        return 1;
    }

    printf("BRIDGE: LOADED\n");

    void * ggml = dlopen(
        "/data/data/com.termux/files/home/llama-opencl-build/build-opencl/bin/libggml-opencl.so.0",
        RTLD_NOW | RTLD_GLOBAL
    );

    if (!ggml) {
        printf("GGML: FAILED\n%s\n", dlerror());
        return 1;
    }

    printf("GGML: LOADED\n");

    typedef void *(*init_fn)(void);
    init_fn init = reinterpret_cast<init_fn>(
        dlsym(ggml, "ggml_backend_opencl_init")
    );

    if (!init) {
        printf("INIT_SYMBOL: NOT FOUND\n%s\n", dlerror());
        return 1;
    }

    printf("INIT_SYMBOL: FOUND\n");

    void * backend = init();

    if (backend) {
        printf("OPENCL_BACKEND: INITIALIZED %p\n", backend);
    } else {
        printf("OPENCL_BACKEND: FAILED\n");
    }

    return 0;
}
