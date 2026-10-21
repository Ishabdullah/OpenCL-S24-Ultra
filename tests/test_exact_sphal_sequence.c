#include <stdio.h>
#include <dlfcn.h>
#include <android/dlext.h>

static void *load_sphal(const char *path) {
    struct android_namespace_t *(*get_ns)(const char *) =
        (struct android_namespace_t *(*)(const char *))
        dlsym(RTLD_DEFAULT, "__loader_android_get_exported_namespace");

    if (!get_ns) {
        printf("GET_NAMESPACE: FAILED\n");
        return NULL;
    }

    struct android_namespace_t *sphal = get_ns("sphal");

    if (!sphal) {
        printf("SPHAL: FAILED\n");
        return NULL;
    }

    android_dlextinfo ext = {0};
    ext.flags = ANDROID_DLEXT_USE_NAMESPACE;
    ext.library_namespace = sphal;

    return android_dlopen_ext(path, RTLD_NOW | RTLD_LOCAL, &ext);
}

int main(void) {
    void *driver = load_sphal("/vendor/lib64/libOpenCL_adreno.so");
    printf("ADRENO_DRIVER: %s\n", driver ? "LOADED" : "FAILED");

    void *loader = load_sphal("/vendor/lib64/libOpenCL.so");
    printf("OPENCL_LOADER: %s\n", loader ? "LOADED" : "FAILED");

    if (!driver) {
        printf("%s\n", dlerror());
        return 1;
    }

    typedef int (*init_fn)(void);

    init_fn init_driver =
        (init_fn)dlsym(driver, "InitOpenCLDriver");

    printf("INIT_SYMBOL: %s\n", init_driver ? "FOUND" : "MISSING");

    if (!init_driver) return 1;

    int rc = init_driver();
    printf("INIT_OPENCL_DRIVER: %d\n", rc);

    return 0;
}
