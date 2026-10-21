#include <stdio.h>
#include <dlfcn.h>
#include <android/dlext.h>

typedef struct android_namespace_t android_namespace_t;
typedef android_namespace_t *(*get_ns_fn)(const char *);
typedef int (*init_fn)(void);

int main(void) {
    void *bridge = dlopen(
        "/data/data/com.termux/files/home/opencl-sphal-bridge/libOpenCL-unversioned.so",
        RTLD_NOW | RTLD_LOCAL);

    printf("BRIDGE: %s\n", bridge ? "LOADED" : "FAILED");

    if (!bridge) {
        printf("BRIDGE_ERROR: %s\n", dlerror());
        return 1;
    }

    get_ns_fn get_ns = (get_ns_fn)dlsym(
        RTLD_DEFAULT,
        "__loader_android_get_exported_namespace");

    printf("NAMESPACE_API: %s\n", get_ns ? "FOUND" : "MISSING");

    if (!get_ns) return 1;

    android_namespace_t *sphal = get_ns("sphal");
    printf("SPHAL: %s\n", sphal ? "FOUND" : "MISSING");

    if (!sphal) return 1;

    android_dlextinfo ext = {0};
    ext.flags = ANDROID_DLEXT_USE_NAMESPACE;
    ext.library_namespace = sphal;

    void *driver = android_dlopen_ext(
        "/vendor/lib64/libOpenCL_adreno.so",
        RTLD_NOW | RTLD_LOCAL,
        &ext);

    printf("DRIVER: %s\n", driver ? "LOADED" : "FAILED");

    if (!driver) {
        printf("DRIVER_ERROR: %s\n", dlerror());
        return 1;
    }

    init_fn init_driver =
        (init_fn)dlsym(driver, "InitOpenCLDriver");

    printf("INIT_SYMBOL: %s\n", init_driver ? "FOUND" : "MISSING");

    if (!init_driver) return 1;

    printf("INIT: %d\n", init_driver());

    return 0;
}
