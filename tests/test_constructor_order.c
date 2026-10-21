#include <dlfcn.h>
#include <stdio.h>

int main(void) {
    printf("MAIN: starting\n");

    void *cpp = dlopen(
        "/data/data/com.termux/files/usr/lib/libc++_shared.so",
        RTLD_NOW | RTLD_GLOBAL);

    printf("MAIN: Termux libc++ = %s\n",
           cpp ? "LOADED" : dlerror());

    void *q = dlopen(
        "/vendor/lib64/libOpenCL_adreno.so",
        RTLD_NOW | RTLD_GLOBAL);

    printf("MAIN: Qualcomm = %s\n",
           q ? "LOADED" : dlerror());

    if (!q)
        return 1;

    int (*init_driver)(void) =
        (int (*)(void))dlsym(q, "InitOpenCLDriver");

    printf("MAIN: InitOpenCLDriver = %d\n",
           init_driver ? init_driver() : -999);

    return 0;
}
