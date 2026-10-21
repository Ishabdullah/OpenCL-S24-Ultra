#include <cstdio>
#include <dlfcn.h>

int main() {
    void *h = dlopen(
        "/data/data/com.termux/files/home/llama-opencl-build/build-opencl/bin/libggml-opencl.so.0",
        RTLD_NOW | RTLD_GLOBAL
    );

    if (!h) {
        printf("DLOPEN: FAILED\n%s\n", dlerror());
        return 1;
    }

    printf("GGML_OPENCL: LOADED\n");

    void *sym = dlsym(h, "ggml_backend_opencl_init");
    if (!sym) {
        printf("INIT_SYMBOL: NOT FOUND\n%s\n", dlerror());
        return 1;
    }

    printf("INIT_SYMBOL: FOUND\n");

    typedef void *(*init_fn)(void);
    init_fn init = reinterpret_cast<init_fn>(sym);

    void *backend = init();

    if (backend) {
        printf("OPENCL_BACKEND: INITIALIZED %p\n", backend);
    } else {
        printf("OPENCL_BACKEND: FAILED (NULL)\n");
    }

    return 0;
}
