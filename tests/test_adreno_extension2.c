#include <stdio.h>
#include <dlfcn.h>

typedef void *(*getext_fn)(const char *);

int main(void) {
    void *h = dlopen("/vendor/lib64/libOpenCL_adreno.so", RTLD_NOW | RTLD_LOCAL);

    if (!h) {
        printf("dlopen FAILED: %s\n", dlerror());
        return 1;
    }

    dlerror();

    getext_fn getext =
        (getext_fn)dlsym(h, "clGetExtensionFunctionAddress");

    if (!getext) {
        printf("clGetExtensionFunctionAddress lookup FAILED: %s\n",
               dlerror());
        return 1;
    }

    const char *names[] = {
        "clGetDeviceIDs",
        "clGetPlatformIDs",
        "clGetExtensionFunctionAddress",
        "clIcdGetPlatformIDsKHR"
    };

    for (int i = 0; i < 4; ++i) {
        void *p = getext(names[i]);

        printf("%s: %s (%p)\n",
               names[i],
               p ? "FOUND" : "NOT FOUND",
               p);
    }

    return 0;
}
