#include <stdio.h>
#include <dlfcn.h>

int main(void) {
    void *h = dlopen("/vendor/lib64/libOpenCL_adreno.so", RTLD_NOW | RTLD_GLOBAL);
    if (!h) {
        printf("dlopen FAILED: %s\n", dlerror());
        return 1;
    }

    dlerror();
    void *p = dlsym(RTLD_DEFAULT, "clIcdGetPlatformIDsKHR");
    const char *e = dlerror();

    printf("global dlopen: LOADED\n");
    printf("RTLD_DEFAULT lookup: %s\n", p ? "FOUND" : "NOT FOUND");
    if (e) printf("dlsym error: %s\n", e);

    return 0;
}
