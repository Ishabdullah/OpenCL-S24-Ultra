#include <dlfcn.h>
#include <stdio.h>
#include <string.h>

int main(void) {
    void *h = dlopen(
        "/data/data/com.termux/files/home/llama-opencl-build/build-opencl/bin/libggml-base.so.0",
        RTLD_NOW | RTLD_GLOBAL);

    if (!h) {
        printf("ggml-base FAILED: %s\n", dlerror());
        return 1;
    }

    printf("ggml-base: LOADED\n");

    void *termux_cpp = dlopen(
        "libc++_shared.so",
        RTLD_NOW | RTLD_GLOBAL);

    printf("Termux libc++_shared.so: %s\n",
           termux_cpp ? "LOADED" : dlerror());

    void *android_cpp = dlopen(
        "/system/lib64/libc++.so",
        RTLD_NOW | RTLD_GLOBAL);

    printf("Android libc++.so: %s\n",
           android_cpp ? "LOADED" : dlerror());

    void *q = dlopen(
        "/vendor/lib64/libOpenCL_adreno.so",
        RTLD_NOW | RTLD_GLOBAL);

    printf("Adreno driver: %s\n",
           q ? "LOADED" : dlerror());

    if (!q)
        return 1;

    int (*init_driver)(void) =
        (int (*)(void))dlsym(q, "InitOpenCLDriver");

    printf("InitOpenCLDriver: %s\n",
           init_driver ? "FOUND" : dlerror());

    if (init_driver)
        printf("InitOpenCLDriver() = %d\n", init_driver());

    return 0;
}
