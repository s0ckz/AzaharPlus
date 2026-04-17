#!/usr/bin/env bash
# Deploy the Mali Vulkan ICD shim onto the device.
#
# DESTRUCTIVE: this overwrites /vendor/lib64/hw/vulkan.rk356x.so on the
# device. A backup of the original is kept at
# /data/local/tmp/vulkan.rk356x.orig.so on the device and also pulled to
# ./build/vulkan.rk356x.orig.so on the host before the swap.
#
# Restore with:  bash deploy.sh --restore
#
# Requires:
#   - Anbernic RG DS (RK356x, Mali G52) userdebug ROM
#   - `adb root` working
#   - `adb disable-verity` already applied + one reboot (one-time setup)

set -euo pipefail

SHIM_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SHIM_SO="${SHIM_DIR}/build/vulkan.rk356x_shim.so"
TARGET="/vendor/lib64/hw/vulkan.rk356x.so"
BACKUP_DEV="/data/local/tmp/vulkan.rk356x.orig.so"
BACKUP_HOST="${SHIM_DIR}/build/vulkan.rk356x.orig.so"

if [[ "${1:-}" == "--restore" ]]; then
    echo ">>> Restoring original Mali Vulkan HAL..."
    adb root >/dev/null
    adb remount >/dev/null
    if [[ -f "${BACKUP_HOST}" ]]; then
        adb push "${BACKUP_HOST}" "${TARGET}"
    else
        adb shell "cp ${BACKUP_DEV} ${TARGET}"
    fi
    adb shell "chmod 644 ${TARGET} && chcon u:object_r:same_process_hal_file:s0 ${TARGET}"
    echo ">>> Done. Reboot the device to force all processes to reload the driver."
    exit 0
fi

if [[ ! -f "${SHIM_SO}" ]]; then
    echo "!!! Shim not built. Run: bash build.sh" >&2
    exit 1
fi

# 1. Need root + writable /vendor.
echo ">>> Acquiring root + /vendor rw mount..."
adb root >/dev/null
sleep 1
adb remount

# 2. Backup the original HAL if we haven't already.
if ! adb shell "[ -f ${BACKUP_DEV} ]"; then
    echo ">>> Backing up ${TARGET} to ${BACKUP_DEV} (and ${BACKUP_HOST})..."
    adb shell "cp ${TARGET} ${BACKUP_DEV}"
    adb pull "${BACKUP_DEV}" "${BACKUP_HOST}" >/dev/null
else
    echo ">>> Backup already exists at ${BACKUP_DEV} (keeping it)."
fi

# 3. Push the shim to /data/local/tmp first, then copy to /vendor. Direct
#    `adb push` to /vendor sometimes breaks SELinux context even on
#    userdebug; the two-step makes the final mv atomic with chcon.
echo ">>> Deploying shim..."
adb push "${SHIM_SO}" /data/local/tmp/vulkan.rk356x.shim.so >/dev/null
adb shell "cp /data/local/tmp/vulkan.rk356x.shim.so ${TARGET}"
adb shell "chmod 644 ${TARGET} && chcon u:object_r:same_process_hal_file:s0 ${TARGET}"

# 4. Sanity-check: make sure the new HAL still exports HMI.
echo ">>> Verifying HMI symbol on device..."
adb shell "readelf --dyn-syms ${TARGET} 2>/dev/null | grep -w HMI || true"

echo ""
echo ">>> Shim deployed. Launch Azahar — first log line should be:"
echo "    MaliVkShim: Mali Vulkan ICD shim loaded; real driver at 0x..."
echo ""
echo ">>> Watch logcat for the load + any crash:"
echo "    adb logcat -c && adb logcat -s MaliVkShim DEBUG AndroidRuntime"
