#include <stdio.h>
#include <stdlib.h>
#include <string.h>

__attribute__((constructor))
static void inspect_order(void) {
    FILE *f = fopen("/proc/self/maps", "r");
    if (!f) return;

    char line[512];
    fprintf(stderr, "=== LIBRARIES ALREADY MAPPED IN CONSTRUCTOR ===\n");

    while (fgets(line, sizeof(line), f)) {
        if (strstr(line, "libc++") ||
            strstr(line, "libadreno_init") ||
            strstr(line, "libOpenCL")) {
            fprintf(stderr, "%s", line);
        }
    }

    fclose(f);
    fprintf(stderr, "=== END ===\n");
}
