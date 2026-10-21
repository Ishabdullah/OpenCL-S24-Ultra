#include <dlfcn.h>
#include <stdio.h>

__attribute__((constructor))
static void shim_init(void) {
    void *android_cpp =
        dlopen("/system/lib64/libc++.so",
               RTLD_NOW | RTLD_GLOBAL);

    printf("shim: Android libc++.so: %s\n",
           android_cpp ? "LOADED" : dlerror());

    void *q =
        dlopen("/vendor/lib64/libOpenCL_adreno.so",
               RTLD_NOW | RTLD_GLOBAL);

    printf("shim: Adreno: %s\n",
           q ? "LOADED" : dlerror());

    if (!q)
        return;

    int (*init_driver)(void) =
        (int (*)(void))dlsym(q, "InitOpenCLDriver");

    printf("shim: InitOpenCLDriver: %s\n",
           init_driver ? "FOUND" : dlerror());

    if (init_driver)
        printf("shim: InitOpenCLDriver() = %d\n",
               init_driver());

    void *real_cpp =
        dlopen("/data/data/com.termux/files/usr/lib/libc++_shared.so.real",
               RTLD_NOW | RTLD_GLOBAL);

    printf("shim: real libc++_shared.so.real: %s\n",
           real_cpp ? "LOADED" : dlerror());
}
