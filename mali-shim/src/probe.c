// Tiny probe: does the Android hwvulkan loader accept our shim?
// Directly dlopen's /vendor/lib64/hw/vulkan.rk356x.so, dlsym's HMI, and
// calls HMI.common.methods->open(). If the shim loads cleanly we see our
// LOGI line in logcat + "OK: device=%p" here.

#include <dlfcn.h>
#include <stdio.h>
#include <stdint.h>

#define MAX_RESV 32
struct hw_module_methods_t;
typedef struct {
    uint32_t tag;
    uint16_t api1;
    uint16_t api2;
    const char* id;
    const char* name;
    const char* author;
    struct hw_module_methods_t* methods;
    void* dso;
    uint64_t reserved[MAX_RESV - 7];
} hw_module_t;
typedef struct hw_module_methods_t {
    int (*open)(const hw_module_t*, const char*, void**);
} hw_module_methods_t;

int main(void) {
    const char* path = "/vendor/lib64/hw/vulkan.rk356x.so";
    void* lib = dlopen(path, 2 /*RTLD_NOW*/);
    if (!lib) { fprintf(stderr, "dlopen(%s) failed: %s\n", path, dlerror()); return 1; }
    printf("dlopen OK: %p\n", lib);

    hw_module_t* hmi = (hw_module_t*) dlsym(lib, "HMI");
    if (!hmi) { fprintf(stderr, "dlsym(HMI) failed: %s\n", dlerror()); return 2; }
    printf("HMI OK: id=%s name=%s author=%s methods=%p\n",
           hmi->id, hmi->name, hmi->author, (void*)hmi->methods);

    void* device = 0;
    int rc = hmi->methods->open(hmi, "vk0", &device);
    printf("open rc=%d device=%p\n", rc, device);
    return rc;
}
