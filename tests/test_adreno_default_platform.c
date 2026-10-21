#include <stdio.h>
#include <dlfcn.h>

typedef int cl_int;
typedef unsigned int cl_uint;
typedef void *cl_platform_id;

typedef cl_int (*fn_t)(
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

    fn_t fn = (fn_t)dlsym(h, "qCLDefaultAPI_clGetPlatformIDs");

    const char *e = dlerror();

    if (!fn) {
        printf("dlsym FAILED: %s\n", e ? e : "unknown");
        return 1;
    }

    cl_platform_id platforms[4] = {0};
    cl_uint count = 0;

    cl_int rc = fn(4, platforms, &count);

    printf("qCLDefaultAPI_clGetPlatformIDs: FOUND\n");
    printf("return: %d\n", rc);
    printf("platform count: %u\n", count);

    for (cl_uint i = 0; i < count; ++i)
        printf("platform[%u]=%p\n", i, platforms[i]);

    return 0;
}
