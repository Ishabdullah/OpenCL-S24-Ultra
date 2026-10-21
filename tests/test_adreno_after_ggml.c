#include <dlfcn.h>
#include <stdio.h>

typedef int (*init_fn)(void);
typedef int (*get_platforms_fn)(unsigned int, void **, unsigned int *);

int main(void) {
    void *cpp = dlopen("/system/lib64/libc++.so", RTLD_NOW | RTLD_GLOBAL);
    printf("Android libc++.so: %s\n", cpp ? "LOADED" : dlerror());

    void *q = dlopen("/vendor/lib64/libOpenCL_adreno.so",
                     RTLD_NOW | RTLD_GLOBAL);
    printf("Adreno driver: %s\n", q ? "LOADED" : dlerror());

    if (!q) return 1;

    init_fn init_driver =
        (init_fn)dlsym(q, "InitOpenCLDriver");

    printf("InitOpenCLDriver: %s\n",
           init_driver ? "FOUND" : dlerror());

    if (!init_driver) return 1;

    printf("Init before ggml = %d\n", init_driver());

    void *ggml = dlopen(
        "/data/data/com.termux/files/usr/lib/libc++_shared.so",
        RTLD_NOW | RTLD_GLOBAL);

    printf("Termux libc++_shared.so: %s\n",
           ggml ? "LOADED" : dlerror());

    void *base = dlopen(
        "/data/data/com.termux/files/home/llama-opencl-build/build-opencl/bin/libggml-base.so.0",
        RTLD_NOW | RTLD_GLOBAL);

    printf("ggml-base: %s\n", base ? "LOADED" : dlerror());

    void *opencl = dlopen(
        "/data/data/com.termux/files/home/llama-opencl-build/build-opencl/bin/libggml-opencl.so.0",
        RTLD_NOW | RTLD_GLOBAL);

    printf("ggml-opencl: %s\n", opencl ? "LOADED" : dlerror());

    printf("Init after ggml = %d\n", init_driver());

    if (!opencl) return 1;

    get_platforms_fn get_platforms =
        (get_platforms_fn)dlsym(opencl, "clGetPlatformIDs");

    printf("clGetPlatformIDs: %s\n",
           get_platforms ? "FOUND" : dlerror());

    if (!get_platforms) return 1;

    void *platforms[4] = {0};
    unsigned int count = 0;

    int rc = get_platforms(4, platforms, &count);

    printf("clGetPlatformIDs rc=%d count=%u platform=%p\n",
           rc, count, platforms[0]);

    return 0;
}
