#include <android/dlext.h>
#include <dlfcn.h>
#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

typedef struct android_namespace_t android_namespace_t;
typedef android_namespace_t *(*namespace_fn)(const char *);
typedef int (*control_fn)(uint32_t, void *, uint32_t);

/* ABI and query IDs: qualcomm/fastrpc inc/remote.h, BSD-3-Clause. */
struct dsp_capability {
    uint32_t domain;
    uint32_t attribute_ID;
    uint32_t capability;
};

int main(int argc, char **argv) {
    setvbuf(stdout, NULL, _IOLBF, 0);
    alarm(15);
    if (argc != 2 || (strcmp(argv[1], "normal") && strcmp(argv[1], "sphal"))) return 2;
    const char *path = "/vendor/lib64/libcdsprpc.so";
    void *driver = NULL;
    if (!strcmp(argv[1], "normal")) {
        driver = dlopen(path, RTLD_NOW | RTLD_LOCAL);
    } else {
        namespace_fn get_ns = (namespace_fn) dlsym(RTLD_DEFAULT, "__loader_android_get_exported_namespace");
        android_namespace_t *ns = get_ns ? get_ns("sphal") : NULL;
        printf("NAMESPACE: %s\n", ns ? "FOUND" : "MISSING");
        if (!ns) return 3;
        android_dlextinfo ext = {0};
        ext.flags = ANDROID_DLEXT_USE_NAMESPACE;
        ext.library_namespace = ns;
        driver = android_dlopen_ext(path, RTLD_NOW | RTLD_LOCAL, &ext);
    }
    printf("DRIVER: %s mode=%s\n", driver ? "LOADED" : "FAILED", argv[1]);
    if (!driver) {
        printf("LOAD_ERROR: %s\n", dlerror());
        return 4;
    }
    const char *symbols[] = {
        "rpcmem_alloc", "rpcmem_free", "rpcmem_to_fd", "fastrpc_mmap", "fastrpc_munmap",
        "dspqueue_create", "dspqueue_close", "dspqueue_export", "dspqueue_write", "dspqueue_read",
        "dspqueue_read_noblock", "remote_handle64_open", "remote_handle64_invoke",
        "remote_handle_control", "remote_handle64_control", "remote_session_control", "remote_handle64_close"
    };
    unsigned missing = 0;
    for (unsigned i = 0; i < sizeof(symbols)/sizeof(symbols[0]); ++i) {
        int found = dlsym(driver, symbols[i]) != NULL;
        printf("SYMBOL: %s %s\n", symbols[i], found ? "FOUND" : "MISSING");
        missing += !found;
    }
    control_fn control = (control_fn) dlsym(driver, "remote_handle_control");
    if (!control) return 5;
    const char *names[] = {
        "DOMAIN_SUPPORT", "UNSIGNED_PD_SUPPORT", "HVX_SUPPORT_64B", "HVX_SUPPORT_128B",
        "VTCM_PAGE", "VTCM_COUNT", "ARCH_VER", "HMX_SUPPORT_DEPTH", "HMX_SUPPORT_SPATIAL"
    };
    int arch_rc = -1;
    uint32_t architecture = 0;
    for (uint32_t id = 0; id < sizeof(names)/sizeof(names[0]); ++id) {
        struct dsp_capability query = {3, id, 0};
        errno = 0;
        int rc = control(2, &query, sizeof(query));
        int saved_errno = errno;
        printf("CAPABILITY: %s rc=%d rc_hex=0x%x value=%u value_hex=0x%x errno=%d\n",
               names[id], rc, (unsigned) rc, query.capability, query.capability, saved_errno);
        if (id == 6) { arch_rc = rc; architecture = query.capability; }
    }
    /* Keep the driver loaded until exit; no DSP program was submitted. */
    if (missing || arch_rc) return 6;
    if ((architecture & 0xff) != 0x75) {
        fprintf(stderr, "This installer builds v75 only; detected architecture 0x%x is unverified/unsupported.\n", architecture);
        return 7;
    }
    return 0;
}
