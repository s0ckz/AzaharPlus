#!/usr/bin/env bash
# Build the Mali Vulkan ICD shim against the Azahar Android toolchain image.
#
# Output: mali-shim/build/vulkan.rk356x_shim.so
#
# The shim is built standalone (not via gradle) because it links only
# against libc/liblog/libdl. We reuse the azahar-android-build image just
# for its NDK install (ndk;27.1.12297006).

set -euo pipefail

SHIM_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SHIM_DIR}/.." && pwd)"
IMAGE="azahar-android-build"

# Build the toolchain image if absent. Mirror build-android.sh so anyone can
# run this script first without having built the main APK.
if ! docker image inspect "${IMAGE}" >/dev/null 2>&1; then
    echo ">>> Building toolchain image ${IMAGE} (first run only)..."
    docker build -t "${IMAGE}" -f "${REPO_ROOT}/docker/android-build/Dockerfile" \
        "${REPO_ROOT}/docker/android-build"
fi

mkdir -p "${SHIM_DIR}/build"

# Use absolute Windows paths for Docker Desktop on Windows. Git Bash otherwise
# mangles /work and /ndk style paths when spawning docker.exe.
MSYS_NO_PATHCONV=1 MSYS2_ARG_CONV_EXCL='*' \
docker run --rm \
    -v "${SHIM_DIR}:/shim" \
    -w /shim \
    "${IMAGE}" \
    bash -eu -c '
        NDK=/opt/android-sdk/ndk/27.1.12297006
        TOOL=$NDK/toolchains/llvm/prebuilt/linux-x86_64/bin
        SYSROOT=$NDK/toolchains/llvm/prebuilt/linux-x86_64/sysroot
        API=28

        # -isystem for the AOSP hardware headers which arent in the NDK sysroot.
        # The hwvulkan.h + hardware.h headers are tiny; we embed our own copy
        # so we dont need the full AOSP tree present in the image.
        mkdir -p build/hw
        cat > build/hw/hardware.h <<EOF
#ifndef SHIM_HARDWARE_H
#define SHIM_HARDWARE_H
#include <stdint.h>
#include <sys/cdefs.h>

#define HARDWARE_MAKE_API_VERSION(maj,min) \
    ((((maj) & 0xff) << 8) | ((min) & 0xff))
#define HARDWARE_MAKE_API_VERSION_2(maj,min,hdr) \
    ((((maj) & 0xff) << 24) | (((min) & 0xff) << 16) | ((hdr) & 0xffff))
#define HARDWARE_HAL_API_VERSION   HARDWARE_MAKE_API_VERSION(1, 0)
#define HARDWARE_MODULE_TAG        (('\''H'\''<<24) | ('\''W'\''<<16) | ('\''M'\''<<8) | '\''T'\'')
#define HARDWARE_DEVICE_TAG        (('\''H'\''<<24) | ('\''W'\''<<16) | ('\''D'\''<<8) | '\''T'\'')

__BEGIN_DECLS
struct hw_module_t;
struct hw_module_methods_t;
struct hw_device_t;

typedef struct hw_module_t {
    uint32_t tag;
    uint16_t module_api_version;
#define version_major module_api_version
    uint16_t hal_api_version;
#define version_minor hal_api_version
    const char* id;
    const char* name;
    const char* author;
    struct hw_module_methods_t* methods;
    void* dso;
#ifdef __LP64__
    uint64_t reserved[32 - 7];
#else
    uint32_t reserved[32 - 7];
#endif
} hw_module_t;

typedef struct hw_module_methods_t {
    int (*open)(const struct hw_module_t* module, const char* id,
                struct hw_device_t** device);
} hw_module_methods_t;

typedef struct hw_device_t {
    uint32_t tag;
    uint32_t version;
    struct hw_module_t* module;
#ifdef __LP64__
    uint64_t reserved[12];
#else
    uint32_t reserved[12];
#endif
    int (*close)(struct hw_device_t* device);
} hw_device_t;
__END_DECLS
#endif
EOF

        cat > build/hw/hwvulkan.h <<EOF
#ifndef SHIM_HWVULKAN_H
#define SHIM_HWVULKAN_H
#include <hardware/hardware.h>
#include <vulkan/vulkan.h>

#define HWVULKAN_HARDWARE_MODULE_ID "vulkan"
#define HWVULKAN_DEVICE_0           "vk0"
#define HWVULKAN_MODULE_API_VERSION_0_1 HARDWARE_MAKE_API_VERSION(0, 1)
#define HWVULKAN_DEVICE_API_VERSION_0_1 HARDWARE_MAKE_API_VERSION_2(0, 1, 0)

typedef struct hwvulkan_dispatch_t {
    uintptr_t magic;
} hwvulkan_dispatch_t;

typedef struct hwvulkan_module_t {
    struct hw_module_t common;
} hwvulkan_module_t;

typedef struct hwvulkan_device_t {
    struct hw_device_t common;
    PFN_vkEnumerateInstanceExtensionProperties EnumerateInstanceExtensionProperties;
    PFN_vkCreateInstance                       CreateInstance;
    PFN_vkGetInstanceProcAddr                  GetInstanceProcAddr;
} hwvulkan_device_t;
#endif
EOF

        ln -sf hw build/hardware

        "$TOOL/aarch64-linux-android${API}-clang++" \
            --sysroot="$SYSROOT" \
            -Ibuild \
            -fPIC -O2 -std=c++17 \
            -fvisibility=hidden \
            -static-libstdc++ \
            -DVK_USE_PLATFORM_ANDROID_KHR \
            -DVK_NO_PROTOTYPES \
            -Wall -Wextra \
            -shared -Wl,-soname,vulkan.rk356x.so \
            -o build/vulkan.rk356x_shim.so \
            src/shim.cpp \
            -llog -ldl

        ls -la build/vulkan.rk356x_shim.so
        "$TOOL/llvm-readelf" -d build/vulkan.rk356x_shim.so | head -20
        "$TOOL/llvm-readelf" --dyn-syms build/vulkan.rk356x_shim.so | grep -E "HMI|GLOBAL" | head -5
    '

echo ""
echo ">>> Built: ${SHIM_DIR}/build/vulkan.rk356x_shim.so"
