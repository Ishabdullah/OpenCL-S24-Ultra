#include <dlfcn.h>
#include <stdio.h>

int main(void) {
    void *h1, *h2;
    int (*init_driver)(void);

    printf("1. Loading Termux OpenCL loader...\n");
    h1 = dlopen("/data/data/com.termux/files/usr/lib/libOpenCL.so",
                RTLD_NOW | RTLD_GLOBAL);
    printf("   %s\n", h1 ? "LOADED" : dlerror());

    printf("2. Loading Qualcomm Adreno driver...\n");
    h2 = dlopen("/vendor/lib64/libOpenCL_adreno.so",
                RTLD_NOW | RTLD_GLOBAL);
    printf("   %s\n", h2 ? "LOADED" : dlerror());

    if (!h2) return 1;

    printf("3. Looking up InitOpenCLDriver...\n");
    init_driver = (int (*)(void))dlsym(h2, "InitOpenCLDriver");
    printf("   %s\n", init_driver ? "FOUND" : dlerror());

    printf("4. Calling InitOpenCLDriver()...\n");
    if (init_driver) {
        int rc = init_driver();
        printf("   return = %d\n", rc);
    }

    return 0;
}
