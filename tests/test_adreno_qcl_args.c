#include <stdio.h>
#include <dlfcn.h>

typedef int cl_int;
typedef unsigned int cl_uint;
typedef void *cl_platform_id;

typedef cl_int (*fn_t)(cl_uint, cl_platform_id *, cl_uint *);

int main(void) {
    void *h = dlopen("/vendor/lib64/libOpenCL_adreno.so", RTLD_NOW | RTLD_LOCAL);

    if (!h) {
        printf("dlopen FAILED: %s\n", dlerror());
        return 1;
    }

    dlerror();
    fn_t fn = (fn_t)dlsym(h, "qCLDrvAPI_clIcdGetPlatformIDsKHR");

    if (!fn) {
        printf("dlsym FAILED: %s\n", dlerror());
        return 1;
    }

    cl_uint count;
    cl_platform_id platform;

    count = 0;
    cl_int rc1 = fn(0, NULL, &count);
    printf("case 1: entries=0, platforms=NULL -> return=%d count=%u\n", rc1, count);

    count = 0;
    platform = NULL;
    cl_int rc2 = fn(1, &platform, &count);
    printf("case 2: entries=1, platforms=&platform -> return=%d count=%u platform=%p\n",
           rc2, count, platform);

    count = 0;
    cl_int rc3 = fn(1, NULL, &count);
    printf("case 3: entries=1, platforms=NULL -> return=%d count=%u\n", rc3, count);

    return 0;
}
