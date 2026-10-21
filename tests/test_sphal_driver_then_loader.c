#include <stdio.h>
#include <dlfcn.h>
#include <android/dlext.h>

typedef struct android_namespace_t android_namespace_t;
typedef android_namespace_t *(*get_ns_fn)(const char *);
typedef int (*init_fn)(void);
typedef int (*get_platforms_fn)(unsigned int, void **, unsigned int *);

static void *load_sphal(const char *path) {
    get_ns_fn get_ns = (get_ns_fn)dlsym(
        RTLD_DEFAULT,
        "__loader_android_get_exported_namespace"
    );

    if (!get_ns) {
        printf("GET_NAMESPACE: FAILED\n");
        return NULL;
    }

    android_namespace_t *sphal = get_ns("sphal");

    if (!sphal) {
        printf("SPHAL: FAILED\n");
        return NULL;
    }

    android_dlextinfo ext = {0};
    ext.flags = ANDROID_DLEXT_USE_NAMESPACE;
    ext.library_namespace = sphal;

    return android_dlopen_ext(
        path,
        RTLD_NOW | RTLD_LOCAL,
        &ext
    );
}

int main(void) {
    void *driver = load_sphal("/vendor/lib64/libOpenCL_adreno.so");

    printf("DRIVER: %s\n", driver ? "LOADED" : "FAILED");

    if (!driver) {
        printf("ERROR: %s\n", dlerror());
        return 1;
    }

    init_fn init_driver =
        (init_fn)dlsym(driver, "InitOpenCLDriver");

    printf("INIT_SYMBOL: %s\n", init_driver ? "FOUND" : "MISSING");

    if (!init_driver) return 1;

    int init_rc = init_driver();

    printf("INIT: %d\n", init_rc);

    if (init_rc != 0) return 1;

    void *loader = load_sphal("/vendor/lib64/libOpenCL.so");

    printf("LOADER: %s\n", loader ? "LOADED" : "FAILED");

    if (!loader) {
        printf("ERROR: %s\n", dlerror());
        return 1;
    }

    get_platforms_fn get_platforms =
        (get_platforms_fn)dlsym(loader, "clGetPlatformIDs");

    printf("CLGETPLATFORMIDS: %s\n",
           get_platforms ? "FOUND" : "MISSING");

    if (!get_platforms) return 1;

    unsigned int count = 0;
    int rc = get_platforms(0, NULL, &count);

    printf("CLGETPLATFORMIDS: rc=%d count=%u\n", rc, count);

    return 0;
}
