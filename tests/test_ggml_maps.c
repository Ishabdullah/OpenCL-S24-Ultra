#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>

int main(void) {
    void *h = dlopen(
        "/data/data/com.termux/files/home/llama-opencl-build/build-opencl/bin/libggml-opencl.so.0",
        RTLD_NOW | RTLD_GLOBAL);

    if (!h) {
        printf("dlopen failed: %s\n", dlerror());
        return 1;
    }

    printf("=== Loaded libraries after libggml-opencl ===\n");

    FILE *f = fopen("/proc/self/maps", "r");
    if (!f) return 1;

    char line[1024];

    while (fgets(line, sizeof(line), f)) {
        if (strstr(line, ".so")) {
            printf("%s", line);
        }
    }

    fclose(f);
    return 0;
}
