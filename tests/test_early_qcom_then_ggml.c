#include <stdio.h>
#include <dlfcn.h>
#include <android/dlext.h>

typedef int (*init_fn)(void);

int main(void) {
    void *android_cpp = dlopen("/system/lib64/libc++.so", RTLD_NOW | RTLD_GLOBAL);
    printf("ANDROID_LIBCXX: %s\n", android_cpp ? "LOADED" : "FAILED");

    void *(*get_ns)(const char *) =
        (void *(*)(const char *))
        dlsym(RTLD_DEFAULT, "__loader_android_get_exported_namespace");

    if (!get_ns) {
        printf("GET_NAMESPACE: FAILED\n");
        return 1;
    }

    void *sphal = get_ns("sphal");
    printf("SPHAL_NAMESPACE: %s\n", sphal ? "FOUND" : "FAILED");

    android_dlextinfo ext = {0};
    ext.flags = ANDROID_DLEXT_USE_NAMESPACE;
    ext.library_namespace = sphal;

    void *driver = android_dlopen_ext(
        "/vendor/lib64/libOpenCL_adreno.so",
        RTLD_NOW | RTLD_LOCAL,
        &ext
    );

    printf("ADRENO_DRIVER: %s\n", driver ? "LOADED" : "FAILED");

    if (!driver) {
        printf("%s\n", dlerror());
        return 1;
    }

    init_fn init_driver =
        (init_fn)dlsym(driver, "InitOpenCLDriver");

    printf("INIT_SYMBOL: %s\n", init_driver ? "FOUND" : "MISSING");

    if (!init_driver) return 1;

    int rc = init_driver();
    printf("INIT_OPENCL_DRIVER: %d\n", rc);

    void *ggml = dlopen(
        "/data/data/com.termux/files/home/llama-opencl-build/build-opencl/bin/libggml-opencl.so.0",
        RTLD_NOW | RTLD_GLOBAL
    );

    printf("GGML_OPENCL: %s\n", ggml ? "LOADED" : "FAILED");

    if (!ggml) {
        printf("%s\n", dlerror());
        return 1;
    }

    init_fn backend_init =
        (init_fn)dlsym(ggml, "ggml_backend_opencl_init");

    printf("BACKEND_INIT_SYMBOL: %s\n",
           backend_init ? "FOUND" : "MISSING");

    if (!backend_init) return 1;

    void *backend = ((void *(*)(void))backend_init)();

    printf("OPENCL_BACKEND: %s %p\n",
           backend ? "INITIALIZED" : "FAILED",
           backend);

    return backend ? 0 : 1;
}
