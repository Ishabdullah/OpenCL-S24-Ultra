#include <dlfcn.h>
#include <stdio.h>

static void *load(const char *path) {
    void *h = dlopen(path, RTLD_NOW | RTLD_GLOBAL);
    printf("load %-80s -> %s\n", path, h ? "OK" : dlerror());
    return h;
}

int main(void) {
    /*
     * Stage 1: Qualcomm environment only.
     */
    void *cpp = load("/system/lib64/libc++.so");
    if (!cpp) return 1;

    void *q = load("/vendor/lib64/libOpenCL_adreno.so");
    if (!q) return 1;

    int (*init_driver)(void) =
        (int (*)(void))dlsym(q, "InitOpenCLDriver");

    printf("InitOpenCLDriver -> %s\n",
           init_driver ? "FOUND" : dlerror());

    if (!init_driver)
        return 1;

    printf("InitOpenCLDriver() -> %d\n", init_driver());

    /*
     * Stage 2: llama libraries by absolute path.
     */
    void *base = load(
        "/data/data/com.termux/files/home/llama-opencl-build/build-opencl/bin/libggml-base.so.0");

    if (!base) return 1;

    void *opencl = load(
        "/data/data/com.termux/files/home/llama-opencl-build/build-opencl/bin/libggml-opencl.so.0");

    if (!opencl) return 1;

    /*
     * Stage 3: see whether Qualcomm still works.
     */
    printf("InitOpenCLDriver() after llama libs -> %d\n",
           init_driver());

    typedef int (*get_platforms_t)(
        unsigned int,
        void **,
        unsigned int *);

    get_platforms_t get_platforms =
        (get_platforms_t)dlsym(opencl, "clGetPlatformIDs");

    printf("clGetPlatformIDs -> %s\n",
           get_platforms ? "FOUND" : dlerror());

    if (!get_platforms)
        return 1;

    unsigned int count = 0;
    void *platform = NULL;

    int rc = get_platforms(1, &platform, &count);

    printf("clGetPlatformIDs rc=%d count=%u platform=%p\n",
           rc, count, platform);

    return 0;
}
