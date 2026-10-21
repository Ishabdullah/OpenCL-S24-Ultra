#include <dlfcn.h>
#include <stdio.h>

int main(void) {
    void *h = dlopen(
        "/data/data/com.termux/files/home/libc++_shared.so.shim",
        RTLD_NOW | RTLD_GLOBAL);

    printf("shim load: %s\n",
           h ? "SUCCESS" : dlerror());

    return h ? 0 : 1;
}
