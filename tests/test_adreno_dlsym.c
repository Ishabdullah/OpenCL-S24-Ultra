#include <stdio.h>
#include <dlfcn.h>

int main(void) {
    void *h = dlopen("/vendor/lib64/libOpenCL_adreno.so", RTLD_NOW | RTLD_LOCAL);
    if (!h) {
        printf("dlopen FAILED: %s\n", dlerror());
        return 1;
    }

    dlerror();
    void *p = dlsym(h, "clIcdGetPlatformIDsKHR");
    const char *e = dlerror();

    printf("handle: %s\n", h ? "LOADED" : "NULL");
    printf("clIcdGetPlatformIDsKHR: %s\n", p ? "FOUND" : "NOT FOUND");
    if (e) printf("dlsym error: %s\n", e);

    return 0;
}
