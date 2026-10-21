#include <dlfcn.h>
#include <stdio.h>

int main(void) {
    void *cpp = dlopen(
        "/data/data/com.termux/files/usr/lib/libc++_shared.so",
        RTLD_NOW | RTLD_GLOBAL);

    printf("Termux libc++_shared: %s\n",
           cpp ? "LOADED" : dlerror());

    void *q = dlopen(
        "/vendor/lib64/libOpenCL_adreno.so",
        RTLD_NOW | RTLD_GLOBAL);

    printf("Adreno driver: %s\n",
           q ? "LOADED" : dlerror());

    if (!q)
        return 1;

    int (*init_driver)(void) =
        (int (*)(void))dlsym(q, "InitOpenCLDriver");

    printf("InitOpenCLDriver: %s\n",
           init_driver ? "FOUND" : dlerror());

    if (init_driver)
        printf("InitOpenCLDriver() = %d\n", init_driver());

    return 0;
}
