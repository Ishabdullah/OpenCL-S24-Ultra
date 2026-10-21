#include <stdio.h>
#include <dlfcn.h>

typedef void *(*clGetExtensionFunctionAddress_fn)(const char *);

int main(void) {
    void *h = dlopen("/vendor/lib64/libOpenCL_adreno.so", RTLD_NOW | RTLD_LOCAL);

    if (!h) {
        printf("dlopen FAILED: %s\n", dlerror());
        return 1;
    }

    dlerror();

    clGetExtensionFunctionAddress_fn getext =
        (clGetExtensionFunctionAddress_fn)
        dlsym(h, "clGetExtensionFunctionAddress");

    const char *e = dlerror();

    printf("clGetExtensionFunctionAddress: %s\n",
           getext ? "FOUND" : "NOT FOUND");

    if (e) {
        printf("dlsym error: %s\n", e);
        return 1;
    }

    if (!getext)
        return 1;

    void *p = getext("clIcdGetPlatformIDsKHR");

    printf("clIcdGetPlatformIDsKHR via extension API: %s\n",
           p ? "FOUND" : "NOT FOUND");

    return 0;
}
