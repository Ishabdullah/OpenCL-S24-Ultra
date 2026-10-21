#include <dlfcn.h>
#include <stdio.h>

int main(void) {
    void *cpp = dlopen(
        "/system/lib64/libc++.so",
        RTLD_NOW | RTLD_GLOBAL);
    printf("Android libc++.so: %s\n", cpp ? "LOADED" : dlerror());

    void *q = dlopen(
        "/vendor/lib64/libOpenCL_adreno.so",
        RTLD_NOW | RTLD_GLOBAL);
    printf("Adreno driver: %s\n", q ? "LOADED" : dlerror());
    if (!q) return 1;

    int (*init_driver)(void) =
        (int (*)(void))dlsym(q, "InitOpenCLDriver");

    printf("InitOpenCLDriver: %s\n",
           init_driver ? "FOUND" : dlerror());
    if (!init_driver) return 1;

    printf("Init before ggml = %d\n", init_driver());

    void *termux_cpp = dlopen(
        "/data/data/com.termux/files/usr/lib/libc++_shared.so",
        RTLD_NOW | RTLD_GLOBAL);
    printf("Termux libc++_shared.so: %s\n",
           termux_cpp ? "LOADED" : dlerror());

    void *base = dlopen(
        "/data/data/com.termux/files/home/llama-opencl-build/build-opencl/bin/libggml-base.so.0",
        RTLD_NOW | RTLD_GLOBAL);
    printf("ggml-base: %s\n", base ? "LOADED" : dlerror());
    if (!base) return 1;

    void *opencl = dlopen(
        "/data/data/com.termux/files/home/llama-opencl-build/build-opencl/bin/libggml-opencl.so.0",
        RTLD_NOW | RTLD_GLOBAL);
    printf("ggml-opencl: %s\n", opencl ? "LOADED" : dlerror());
    if (!opencl) return 1;

    printf("Init after ggml-opencl = %d\n", init_driver());

    typedef int (*get_platforms_t)(
        unsigned int,
        void **,
        unsigned int *);

    get_platforms_t get_platforms =
        (get_platforms_t)dlsym(opencl, "clGetPlatformIDs");

    printf("clGetPlatformIDs: %s\n",
           get_platforms ? "FOUND" : dlerror());

    if (!get_platforms) return 1;

    unsigned int count = 0;
    void *platform = NULL;

    int rc = get_platforms(1, &platform, &count);

    printf("clGetPlatformIDs rc=%d count=%u platform=%p\n",
           rc, count, platform);

    return 0;
}
