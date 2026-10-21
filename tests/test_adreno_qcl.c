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

    const char *names[] = {
        "qCLDefaultAPI_clIcdGetPlatformIDsKHR",
        "qCLDrvAPI_clIcdGetPlatformIDsKHR",
        "clIcdGetPlatformIDsKHR"
    };

    for (int i = 0; i < 3; ++i) {
        dlerror();

        fn_t fn = (fn_t)dlsym(h, names[i]);
        const char *e = dlerror();

        printf("%s: %s\n", names[i], fn ? "FOUND" : "NOT FOUND");

        if (e) {
            printf("  error: %s\n", e);
            continue;
        }

        if (fn) {
            cl_platform_id platforms[8] = {0};
            cl_uint count = 0;

            cl_int rc = fn(8, platforms, &count);

            printf("  return: %d\n", rc);
            printf("  count: %u\n", count);
        }
    }

    return 0;
}
