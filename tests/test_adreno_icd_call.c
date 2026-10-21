#include <stdio.h>
#include <dlfcn.h>

typedef int cl_int;
typedef unsigned int cl_uint;
typedef void *cl_platform_id;

typedef cl_int (*clIcdGetPlatformIDsKHR_fn)(
    cl_uint,
    cl_platform_id *,
    cl_uint *
);

int main(void) {
    void *h = dlopen("/vendor/lib64/libOpenCL_adreno.so", RTLD_NOW | RTLD_LOCAL);

    if (!h) {
        printf("dlopen FAILED: %s\n", dlerror());
        return 1;
    }

    dlerror();
    clIcdGetPlatformIDsKHR_fn fn =
        (clIcdGetPlatformIDsKHR_fn)dlsym(h, "clIcdGetPlatformIDsKHR");

    const char *e = dlerror();

    if (!fn) {
        printf("dlsym FAILED: %s\n", e ? e : "unknown");
        return 1;
    }

    cl_platform_id platforms[8] = {0};
    cl_uint count = 0;

    cl_int rc = fn(8, platforms, &count);

    printf("function: FOUND\n");
    printf("return: %d\n", rc);
    printf("platform count: %u\n", count);

    return 0;
}
