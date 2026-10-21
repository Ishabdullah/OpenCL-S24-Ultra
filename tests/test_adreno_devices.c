#include <stdio.h>
#include <dlfcn.h>

typedef int cl_int;
typedef unsigned int cl_uint;
typedef void *cl_platform_id;
typedef void *cl_device_id;

typedef cl_int (*get_platform_fn)(
    cl_uint,
    cl_platform_id *,
    cl_uint *
);

typedef cl_int (*get_devices_fn)(
    cl_platform_id,
    cl_uint,
    cl_uint,
    cl_device_id *,
    cl_uint *
);

int main(void) {
    void *h = dlopen("/vendor/lib64/libOpenCL_adreno.so", RTLD_NOW | RTLD_LOCAL);

    if (!h) {
        printf("dlopen FAILED: %s\n", dlerror());
        return 1;
    }

    get_platform_fn get_platforms =
        (get_platform_fn)dlsym(h, "qCLDrvAPI_clIcdGetPlatformIDsKHR");

    get_devices_fn get_devices =
        (get_devices_fn)dlsym(h, "qCLDrvAPI_clGetDeviceIDs");

    if (!get_platforms || !get_devices) {
        printf("function lookup failed: %s\n", dlerror());
        return 1;
    }

    cl_platform_id platform = NULL;
    cl_uint platform_count = 0;

    cl_int rc = get_platforms(1, &platform, &platform_count);

    printf("platform rc=%d count=%u handle=%p\n",
           rc, platform_count, platform);

    if (rc != 0 || !platform)
        return 1;

    cl_device_id devices[8] = {0};
    cl_uint device_count = 0;

    rc = get_devices(
        platform,
        0xFFFFFFFF,
        8,
        devices,
        &device_count
    );

    printf("devices rc=%d count=%u\n", rc, device_count);

    for (cl_uint i = 0; i < device_count; ++i)
        printf("device[%u]=%p\n", i, devices[i]);

    return 0;
}
