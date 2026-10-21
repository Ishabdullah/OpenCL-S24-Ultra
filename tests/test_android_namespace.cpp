#include <cstdio>
#include <dlfcn.h>

struct android_namespace_t;

using get_ns_fn = android_namespace_t *(*)(const char *);

int main() {
    dlerror();

    auto get_ns = reinterpret_cast<get_ns_fn>(
        dlsym(RTLD_DEFAULT, "__loader_android_get_exported_namespace"));

    const char *err = dlerror();

    if (err) {
        std::printf("get_exported_namespace: NOT FOUND: %s\n", err);
        return 1;
    }

    std::printf("get_exported_namespace: FOUND\n");

    const char *names[] = {
        "default",
        "system",
        "system_server",
        "sphal",
        "vndk"
    };

    for (const char *name : names) {
        void *ns = get_ns(name);
        std::printf("%s: %s (%p)\n",
                    name,
                    ns ? "FOUND" : "NOT FOUND",
                    ns);
    }

    return 0;
}
