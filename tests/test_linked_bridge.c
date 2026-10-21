#include <stdio.h>
#include <CL/cl.h>

int main(void) {
    printf("PROGRAM_START\n");

    cl_uint count = 0;
    cl_int rc = clGetPlatformIDs(0, NULL, &count);

    printf("RESULT: rc=%d count=%u\n", rc, count);

    return 0;
}
