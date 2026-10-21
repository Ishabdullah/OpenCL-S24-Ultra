#include <cstdio>
#include <dlfcn.h>
#include <android/dlext.h>

struct android_namespace_t;

using get_ns_fn = android_namespace_t *(*)(const char *);

int main() {
    dlerror();

    auto get_ns = reinterpret_cast<get_ns_fn>(
        dlsym(RTLD_DEFAULT, "__loader_android_get_exported_namespace"));

    if (!get_ns || dlerror()) {
        std::printf("namespace API: FAILED\n");
        return 1;
    }

    auto *sphal = get_ns("sphal");

    if (!sphal) {
        std::printf("sphal namespace: NOT FOUND\n");
        return 1;
    }

    android_dlextinfo ext{};
    ext.flags = ANDROID_DLEXT_USE_NAMESPACE;
    ext.library_namespace = sphal;

    void *handle = android_dlopen_ext(
        "/vendor/lib64/libCB.so",
        RTLD_NOW | RTLD_LOCAL,
        &ext
    );

    if (!handle) {
        std::printf("libCB.so: FAILED\n%s\n", dlerror());
        return 1;
    }

    std::printf("libCB.so: LOADED through sphal\n");
    return 0;
}
