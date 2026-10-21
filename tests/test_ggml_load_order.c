#include <dlfcn.h>
#include <stdio.h>

int main(void) {
    void *h, *q;
    int (*init_driver)(void);

    printf("1. Loading ggml-opencl...\n");
    h = dlopen(
        "/data/data/com.termux/files/home/llama-opencl-build/build-opencl/bin/libggml-opencl.so.0",
        RTLD_NOW | RTLD_GLOBAL);

    if (!h) {
        printf("FAILED: %s\n", dlerror());
        return 1;
    }

    printf("   LOADED\n");

    printf("2. Loading Qualcomm driver...\n");
    q = dlopen("/vendor/lib64/libOpenCL_adreno.so",
               RTLD_NOW | RTLD_GLOBAL);

    if (!q) {
        printf("FAILED: %s\n", dlerror());
        return 1;
    }

    printf("   LOADED\n");

    init_driver = (int (*)(void))dlsym(q, "InitOpenCLDriver");

    printf("3. InitOpenCLDriver: %s\n",
           init_driver ? "FOUND" : dlerror());

    if (init_driver) {
        int rc = init_driver();
        printf("4. InitOpenCLDriver() return = %d\n", rc);
    }

    return 0;
}
