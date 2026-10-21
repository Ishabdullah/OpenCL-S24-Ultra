#include <dlfcn.h>
#include <stdio.h>

int main(void) {
    const char *libs[] = {
        "/vendor/lib64/libgsl.so",
        "/vendor/lib64/libCB.so"
    };

    for (int i = 0; i < 2; i++) {
        dlerror();
        void *h = dlopen(libs[i], RTLD_NOW | RTLD_GLOBAL);
        const char *err = dlerror();

        printf("%s: %s\n", libs[i], h ? "LOADED" : "FAILED");
        if (err) printf("  %s\n", err);
    }

    return 0;
}
