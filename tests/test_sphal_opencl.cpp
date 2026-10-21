#include <cstdio>
#include <dlfcn.h>
#include <android/dlext.h>

struct android_namespace_t;

using get_ns_fn = android_namespace_t *(*)(const char *);
using init_fn = int (*)();
using get_platforms_fn = int (*)(unsigned int, void **, unsigned int *);

int main() {
    auto get_ns = reinterpret_cast<get_ns_fn>(
        dlsym(RTLD_DEFAULT, "__loader_android_get_exported_namespace"));

    if (!get_ns) {
        std::printf("namespace API: FAILED\n");
        return 1;
    }

    auto *sphal = get_ns("sphal");

    if (!sphal) {
        std::printf("sphal: FAILED\n");
        return 1;
    }

    android_dlextinfo ext{};
    ext.flags = ANDROID_DLEXT_USE_NAMESPACE;
    ext.library_namespace = sphal;

    void *driver = android_dlopen_ext(
        "/vendor/lib64/libOpenCL_adreno.so",
        RTLD_NOW | RTLD_LOCAL,
        &ext
    );

    if (!driver) {
        std::printf("Adreno driver: FAILED\n%s\n", dlerror());
        return 1;
    }

    std::printf("Adreno driver: LOADED through sphal\n");

    auto init = reinterpret_cast<init_fn>(
        dlsym(driver, "InitOpenCLDriver"));

    if (!init) {
        std::printf("InitOpenCLDriver: NOT FOUND\n");
        return 1;
    }

    int rc = init();
    std::printf("InitOpenCLDriver(): %d\n", rc);

    return rc == 0 ? 0 : 1;
}
