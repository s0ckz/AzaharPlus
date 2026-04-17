#!/usr/bin/env bash
# Build the Azahar Mali-G52 serialization Vulkan layer.
#
# Output: build/libVkLayer_azahar_mali_serialize.so  (aarch64-android API 28)
#
# To integrate into Azahar:
#   cp build/libVkLayer_azahar_mali_serialize.so \
#      ../src/android/app/src/main/jniLibs/arm64-v8a/

set -euo pipefail

LAYER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${LAYER_DIR}/.." && pwd)"
IMAGE="azahar-android-build"

if ! docker image inspect "${IMAGE}" >/dev/null 2>&1; then
    docker build -t "${IMAGE}" -f "${REPO_ROOT}/docker/android-build/Dockerfile" \
        "${REPO_ROOT}/docker/android-build"
fi

mkdir -p "${LAYER_DIR}/build"

MSYS_NO_PATHCONV=1 MSYS2_ARG_CONV_EXCL='*' \
docker run --rm \
    -v "${LAYER_DIR}:/layer" \
    -w /layer \
    "${IMAGE}" \
    bash -eu -c '
        NDK=/opt/android-sdk/ndk/27.1.12297006
        TOOL=$NDK/toolchains/llvm/prebuilt/linux-x86_64/bin
        SYSROOT=$NDK/toolchains/llvm/prebuilt/linux-x86_64/sysroot
        API=28

        "$TOOL/aarch64-linux-android${API}-clang++" \
            --sysroot="$SYSROOT" \
            -fPIC -O2 -std=c++17 \
            -fvisibility=hidden \
            -static-libstdc++ \
            -DVK_USE_PLATFORM_ANDROID_KHR \
            -DVK_NO_PROTOTYPES \
            -Wall -Wextra -Wno-unused-variable -Wno-missing-field-initializers \
            -shared -Wl,-soname,libVkLayer_azahar_mali_serialize.so \
            -o build/libVkLayer_azahar_mali_serialize.so \
            src/layer.cpp \
            -llog

        ls -la build/libVkLayer_azahar_mali_serialize.so
        "$TOOL/llvm-readelf" -d build/libVkLayer_azahar_mali_serialize.so | head -12
        echo "--- exported loader entry points ---"
        "$TOOL/llvm-readelf" --dyn-syms build/libVkLayer_azahar_mali_serialize.so \
            | grep -E "vkGetInstanceProcAddr|vkGetDeviceProcAddr|vkNegotiate|vkEnumerateInstance" | head -10
    '

echo ""
echo ">>> Built: ${LAYER_DIR}/build/libVkLayer_azahar_mali_serialize.so"
