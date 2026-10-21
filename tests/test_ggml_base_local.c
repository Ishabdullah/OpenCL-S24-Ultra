#include <dlfcn.h>
#include <stdio.h>

int main(void) {
    void *h = dlopen(
        "/data/data/com.termux/files/home/llama-opencl-build/build-opencl/bin/libggml-base.so.0",
        RTLD_NOW | RTLD_LOCAL);

    if (!h) {
        printf("ggml-base FAILED: %s\n", dlerror());
        return 1;
    }

    printf("ggml-base: LOADED\n");

    void *q = dlopen("/vendor/lib64/libOpenCL_adreno.so",
                     RTLD_NOW | RTLD_LOCAL);

    printf("Adreno driver: %s\n", q ? "LOADED" : dlerror());

    if (!q) return 1;

    int (*init_driver)(void) =
        (int (*)(void))dlsym(q, "InitOpenCLDriver");

    printf("InitOpenCLDriver: %s\n",
           init_driver ? "FOUND" : dlerror());

    if (init_driver)
        printf("InitOpenCLDriver() = %d\n", init_driver());

    return 0;
}
