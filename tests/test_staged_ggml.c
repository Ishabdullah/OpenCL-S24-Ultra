#include <dlfcn.h>
#include <stdio.h>

int main(void) {
    void *cpp = dlopen(
        "/system/lib64/libc++.so",
        RTLD_NOW | RTLD_GLOBAL);

    printf("Android libc++.so: %s\n",
           cpp ? "LOADED" : dlerror());

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

    if (!init_driver)
        return 1;

    printf("Init before ggml = %d\n", init_driver());

    void *termux_cpp = dlopen(
        "/data/data/com.termux/files/usr/lib/libc++_shared.so",
        RTLD_NOW | RTLD_GLOBAL);

    printf("Termux libc++_shared.so: %s\n",
           termux_cpp ? "LOADED" : dlerror());

    void *base = dlopen(
        "/data/data/com.termux/files/home/llama-opencl-build/build-opencl/bin/libggml-base.so.0",
        RTLD_NOW | RTLD_GLOBAL);

    printf("ggml-base: %s\n",
           base ? "LOADED" : dlerror());

    printf("Init after ggml = %d\n", init_driver());

    return 0;
}
