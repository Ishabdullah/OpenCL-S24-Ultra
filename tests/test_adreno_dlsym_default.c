#include <stdio.h>
#include <dlfcn.h>

int main(void) {
    void *h = dlopen("/vendor/lib64/libOpenCL_adreno.so", RTLD_NOW | RTLD_LOCAL);
    if (!h) {
        printf("dlopen FAILED: %s\n", dlerror());
        return 1;
    }

    dlerror();
    void *p1 = dlsym(h, "clIcdGetPlatformIDsKHR");
    const char *e1 = dlerror();

    dlerror();
    void *p2 = dlsym(RTLD_DEFAULT, "clIcdGetPlatformIDsKHR");
    const char *e2 = dlerror();

    printf("handle lookup:  %s\n", p1 ? "FOUND" : "NOT FOUND");
    if (e1) printf("handle error: %s\n", e1);

    printf("default lookup: %s\n", p2 ? "FOUND" : "NOT FOUND");
    if (e2) printf("default error: %s\n", e2);

    return 0;
}
