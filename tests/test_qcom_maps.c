#include <stdio.h>
#include <dlfcn.h>
#include <android/dlext.h>

int main(void) {
    void *android_cpp = dlopen("/system/lib64/libc++.so", RTLD_NOW | RTLD_GLOBAL);
    printf("ANDROID_LIBCXX: %s\n", android_cpp ? "LOADED" : "FAILED");

    void *(*get_ns)(const char *) =
        (void *(*)(const char *))
        dlsym(RTLD_DEFAULT, "__loader_android_get_exported_namespace");

    void *sphal = get_ns("sphal");

    android_dlextinfo ext = {0};
    ext.flags = ANDROID_DLEXT_USE_NAMESPACE;
    ext.library_namespace = sphal;

    void *driver = android_dlopen_ext(
        "/vendor/lib64/libOpenCL_adreno.so",
        RTLD_NOW | RTLD_LOCAL,
        &ext
    );

    printf("ADRENO_DRIVER: %s\n", driver ? "LOADED" : "FAILED");

    printf("\n=== LOADED C++ / QUALCOMM LIBRARIES ===\n");

    FILE *f = fopen("/proc/self/maps", "r");
    if (!f) return 1;

    char line[1024];

    while (fgets(line, sizeof(line), f)) {
        if (strstr(line, "libc++") ||
            strstr(line, "libCB.so") ||
            strstr(line, "libgsl.so") ||
            strstr(line, "libOpenCL")) {
            printf("%s", line);
        }
    }

    fclose(f);

    return 0;
}
