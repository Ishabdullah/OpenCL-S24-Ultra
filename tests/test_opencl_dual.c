#include <CL/cl.h>
#include <stdio.h>

int main(void) {
    cl_uint n = 0;
    cl_int err = clGetPlatformIDs(0, NULL, &n);

    printf("clGetPlatformIDs error: %d\n", err);
    printf("platform count: %u\n", n);

    if (err == CL_SUCCESS && n > 0) {
        cl_platform_id p;
        clGetPlatformIDs(1, &p, NULL);

        char name[256] = {0};
        char version[256] = {0};

        clGetPlatformInfo(p, CL_PLATFORM_NAME, sizeof(name), name, NULL);
        clGetPlatformInfo(p, CL_PLATFORM_VERSION, sizeof(version), version, NULL);

        printf("platform: %s\n", name);
        printf("version: %s\n", version);
    }

    return 0;
}
