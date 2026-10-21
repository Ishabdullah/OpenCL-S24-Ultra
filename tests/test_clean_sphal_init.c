#include <stdio.h>
#include <dlfcn.h>
#include <android/dlext.h>

int main(void) {
    void *(*get_ns)(const char *) =
        (void *(*)(const char *))
        dlsym(RTLD_DEFAULT, "__loader_android_get_exported_namespace");

    printf("GET_NAMESPACE: %s\n", get_ns ? "FOUND" : "FAILED");

    if (!get_ns) return 1;

    struct android_namespace_t *sphal = get_ns("sphal");

    printf("SPHAL: %s\n", sphal ? "FOUND" : "FAILED");

    if (!sphal) return 1;

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

    typedef int (*init_fn)(void);

    init_fn init_driver =
        (init_fn)dlsym(driver, "InitOpenCLDriver");

    printf("INIT_SYMBOL: %s\n", init_driver ? "FOUND" : "MISSING");

    if (!init_driver) return 1;

    printf("INIT_OPENCL_DRIVER: %d\n", init_driver());

    return 0;
}
