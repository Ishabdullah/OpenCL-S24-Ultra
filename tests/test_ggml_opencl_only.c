#include <dlfcn.h>
#include <stdio.h>

int main(void) {
    void *h;
    void *ocl;
    int (*init_driver)(void);

    h = dlopen(
        "/data/data/com.termux/files/home/llama-opencl-build/build-opencl/bin/libggml-opencl.so.0",
        RTLD_NOW | RTLD_GLOBAL);

    if (!h) {
        printf("ggml-opencl FAILED: %s\n", dlerror());
        return 1;
    }

    printf("ggml-opencl: LOADED\n");

    ocl = dlopen("/data/data/com.termux/files/usr/lib/libOpenCL.so",
                 RTLD_NOW | RTLD_GLOBAL);

    printf("Termux libOpenCL: %s\n", ocl ? "LOADED" : dlerror());

    void *q = dlopen("/vendor/lib64/libOpenCL_adreno.so",
                     RTLD_NOW | RTLD_GLOBAL);

    printf("Adreno driver: %s\n", q ? "LOADED" : dlerror());

    if (!q) return 1;

    init_driver = (int (*)(void))dlsym(q, "InitOpenCLDriver");

    printf("InitOpenCLDriver: %s\n",
           init_driver ? "FOUND" : dlerror());

    if (init_driver)
        printf("InitOpenCLDriver() = %d\n", init_driver());

    return 0;
}
