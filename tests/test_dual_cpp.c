#include <dlfcn.h>
#include <stdio.h>

int main(void) {
    void *a = dlopen("/apex/com.android.vndk.v34/lib64/libc++.so", RTLD_NOW | RTLD_LOCAL);
    printf("Android libc++.so: %s\n", a ? "LOADED" : dlerror());

    void *b = dlopen("/data/data/com.termux/files/usr/lib/libc++_shared.so", RTLD_NOW | RTLD_LOCAL);
    printf("Termux libc++_shared.so: %s\n", b ? "LOADED" : dlerror());

    void *c = dlopen("/vendor/lib64/libOpenCL.so", RTLD_NOW | RTLD_LOCAL);
    printf("Vendor libOpenCL.so: %s\n", c ? "LOADED" : dlerror());

    void *d = dlopen("/vendor/lib64/libOpenCL_adreno.so", RTLD_NOW | RTLD_LOCAL);
    printf("Adreno libOpenCL_adreno.so: %s\n", d ? "LOADED" : dlerror());

    return 0;
}
