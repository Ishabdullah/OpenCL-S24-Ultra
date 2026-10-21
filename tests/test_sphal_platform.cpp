#include <cstdio>
#include <dlfcn.h>
#include <android/dlext.h>

struct android_namespace_t;

using get_ns_fn = android_namespace_t *(*)(const char *);
using init_fn = int (*)();

using cl_int = int;
using cl_uint = unsigned int;
using cl_platform_id = void *;

using clGetPlatformIDs_fn =
    cl_int (*)(cl_uint, cl_platform_id *, cl_uint *);

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

    std::printf("Adreno driver: LOADED\n");

    auto init = reinterpret_cast<init_fn>(
        dlsym(driver, "InitOpenCLDriver"));

    if (!init) {
        std::printf("InitOpenCLDriver: NOT FOUND\n");
        return 1;
    }

    std::printf("InitOpenCLDriver(): %d\n", init());

    void *loader = android_dlopen_ext(
        "/vendor/lib64/libOpenCL.so",
        RTLD_NOW | RTLD_LOCAL,
        &ext
    );

    if (!loader) {
        std::printf("OpenCL ICD loader: FAILED\n%s\n", dlerror());
        return 1;
    }

    std::printf("OpenCL ICD loader: LOADED\n");

    auto get_platforms =
        reinterpret_cast<clGetPlatformIDs_fn>(
            dlsym(loader, "clGetPlatformIDs"));

    if (!get_platforms) {
        std::printf("clGetPlatformIDs: NOT FOUND\n");
        return 1;
    }

    cl_uint count = 0;

    cl_int rc = get_platforms(
        0,
        nullptr,
        &count
    );

    std::printf(
        "clGetPlatformIDs(0): rc=%d count=%u\n",
        rc,
        count
    );

    if (rc != 0 || count == 0)
        return 1;

    cl_platform_id platform = nullptr;

    rc = get_platforms(
        1,
        &platform,
        nullptr
    );

    std::printf(
        "clGetPlatformIDs(1): rc=%d platform=%p\n",
        rc,
        platform
    );

    return (rc == 0 && platform) ? 0 : 1;
}
