#include <stdio.h>
#include <dlfcn.h>
#include <android/dlext.h>
#include <CL/cl.h>

typedef struct android_namespace_t android_namespace_t;
typedef android_namespace_t *(*get_ns_fn)(const char *);
typedef int (*init_fn)(void);
typedef cl_int (*get_platforms_fn)(cl_uint, cl_platform_id *, cl_uint *);

int main(void) {
    void *bridge = dlopen(
        "/data/data/com.termux/files/home/opencl-sphal-bridge/libOpenCL-unversioned.so",
        RTLD_NOW | RTLD_LOCAL);

    printf("BRIDGE: %s\n", bridge ? "LOADED" : "FAILED");
    if (!bridge) return 1;

    get_ns_fn get_ns = (get_ns_fn)dlsym(
        RTLD_DEFAULT,
        "__loader_android_get_exported_namespace");

    android_namespace_t *sphal = get_ns("sphal");

    android_dlextinfo ext = {0};
    ext.flags = ANDROID_DLEXT_USE_NAMESPACE;
    ext.library_namespace = sphal;

    void *driver = android_dlopen_ext(
        "/vendor/lib64/libOpenCL_adreno.so",
        RTLD_NOW | RTLD_LOCAL,
        &ext);

    printf("DRIVER: %s\n", driver ? "LOADED" : "FAILED");
    if (!driver) return 1;

    init_fn init_driver =
        (init_fn)dlsym(driver, "InitOpenCLDriver");

    printf("INIT: %d\n", init_driver());

    get_platforms_fn get_platforms =
        (get_platforms_fn)dlsym(bridge, "clGetPlatformIDs");

    printf("BRIDGE_CLGETPLATFORMIDS: %s\n",
           get_platforms ? "FOUND" : "MISSING");

    if (!get_platforms) return 1;

    cl_uint count = 0;
    cl_int rc = get_platforms(0, NULL, &count);

    printf("RESULT: rc=%d count=%u\n", rc, count);

    return 0;
}
