#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdio.h>
#include <CL/cl.h>

int main(void) {
    void *h = dlopen("/vendor/lib64/libOpenCL_adreno.so",
                     RTLD_NOW | RTLD_GLOBAL);

    if (!h) {
        printf("dlopen failed: %s\n", dlerror());
        return 1;
    }

    int (*InitOpenCLDriver)(void) =
        (int (*)(void))dlsym(h, "InitOpenCLDriver");

    void *(*clGetExtensionFunctionAddress)(const char *) =
        (void *(*)(const char *))dlsym(h, "clGetExtensionFunctionAddress");

    if (!InitOpenCLDriver || !clGetExtensionFunctionAddress) {
        printf("required symbol missing\n");
        return 1;
    }

    printf("InitOpenCLDriver(): %d\n", InitOpenCLDriver());

    void *fn = clGetExtensionFunctionAddress(
        "clIcdGetPlatformIDsKHR");

    printf("clIcdGetPlatformIDsKHR: %s (%p)\n",
           fn ? "FOUND" : "NOT FOUND", fn);

    cl_uint count = 0;
    cl_int rc = clGetPlatformIDs(0, NULL, &count);

    printf("clGetPlatformIDs: rc=%d count=%u\n", rc, count);

    if (rc == CL_SUCCESS && count > 0) {
        cl_platform_id platform = NULL;

        rc = clGetPlatformIDs(1, &platform, NULL);

        printf("platform lookup: rc=%d platform=%p\n",
               rc, (void *)platform);

        if (rc == CL_SUCCESS) {
            char name[256] = {0};
            char version[256] = {0};

            clGetPlatformInfo(platform, CL_PLATFORM_NAME,
                              sizeof(name), name, NULL);
            clGetPlatformInfo(platform, CL_PLATFORM_VERSION,
                              sizeof(version), version, NULL);

            printf("platform: %s\n", name);
            printf("version: %s\n", version);
        }
    }

    return 0;
}
