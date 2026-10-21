#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdint.h>
#include <stdio.h>

int main(void) {
    void *h = dlopen("/vendor/lib64/libOpenCL_adreno.so",
                     RTLD_NOW | RTLD_GLOBAL);

    if (!h) {
        printf("dlopen failed: %s\n", dlerror());
        return 1;
    }

    int (*InitOpenCLDriver)(void) =
        (int (*)(void))dlsym(h, "InitOpenCLDriver");

    if (!InitOpenCLDriver) {
        printf("InitOpenCLDriver: NOT FOUND\n");
        return 1;
    }

    Dl_info info;
    if (!dladdr((void *)InitOpenCLDriver, &info)) {
        printf("dladdr failed\n");
        return 1;
    }

    uintptr_t base = (uintptr_t)info.dli_fbase;

    printf("library base:       %p\n", (void *)base);
    printf("InitOpenCLDriver:   %p\n", (void *)InitOpenCLDriver);

    printf("g_JumpTableGSL:     %p\n",
           *(void **)(base + 0x2a0e8));

    int rc = InitOpenCLDriver();

    printf("InitOpenCLDriver():  %d\n", rc);

    printf("g_JumpTableGSL:     %p\n",
           *(void **)(base + 0x2a0e8));

    printf("g_gslInitialized:   %u\n",
           *(unsigned char *)(base + 0x2a108));

    printf("g_cbInitialized:    %u\n",
           *(unsigned char *)(base + 0x2a109));

    printf("g_mlbInitialized:   %u\n",
           *(unsigned char *)(base + 0x2a3c8));

    return 0;
}
