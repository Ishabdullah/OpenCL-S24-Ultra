#include <android/thermal.h>
#include <dlfcn.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>

int main(int argc, char **argv) {
    int count = argc == 2 ? atoi(argv[1]) : 1;
    if (argc > 2 || count < 1 || count > 1000) return 2;
    void *lib = dlopen("/system/lib64/libandroid.so", RTLD_NOW | RTLD_LOCAL);
    if (!lib) {
        fprintf(stderr, "libandroid load failed: %s\n", dlerror());
        return 77;
    }
    AThermalManager *(*acquire)(void) = dlsym(lib, "AThermal_acquireManager");
    void (*release)(AThermalManager *) = dlsym(lib, "AThermal_releaseManager");
    AThermalStatus (*status)(AThermalManager *) = dlsym(lib, "AThermal_getCurrentThermalStatus");
    float (*headroom)(AThermalManager *, int) = dlsym(lib, "AThermal_getThermalHeadroom");
    if (!acquire || !release || !status) return 77;
    AThermalManager *manager = acquire();
    if (!manager) {
        fprintf(stderr, "Thermal manager unavailable\n");
        return 77;
    }
    for (int i = 0; i < count; ++i) {
        struct timespec ts;
        clock_gettime(CLOCK_MONOTONIC, &ts);
        int level = status(manager);
        float estimate = headroom ? headroom(manager, 30) : NAN;
        printf("{\"monotonic_s\":%.9f,\"thermal_status\":%d,\"forecast_seconds\":30,\"thermal_headroom\":",
               ts.tv_sec + ts.tv_nsec / 1e9, level);
        if (isfinite(estimate)) printf("%.9f", estimate);
        else printf("null");
        printf("}\n");
        fflush(stdout);
        if (i + 1 < count) sleep(10);
    }
    release(manager);
    return 0;
}
