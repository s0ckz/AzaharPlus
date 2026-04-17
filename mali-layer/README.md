# Azahar Mali-G52 Serialization Vulkan Layer

Vulkan layer that serialises every hot Mali entry with a global recursive
mutex. Targets the Mali G52 r25p0 (`g25p0-00eac0`) Bifrost driver as shipped
on RK356x Android handhelds (e.g. Anbernic RG DS). No-op pass-through on
every other GPU — detected at `vkCreateDevice` time by reading the physical
device name.

## Why

When Azahar runs with the threaded renderer (`deploy/option-c-targeted`),
the emu thread, GpuWorker and VulkanWorker all enter Vulkan concurrently.
The Mali driver has a TOCTOU race in an internal dispatch helper at offset
`0x1e05a38` (reads `[ctx+0x20]`, dereferences `[NULL+0x60]` → SIGSEGV with
fault addr `0x60`) and a second null-deref at `0x9e5014` (fault addr `0x8`).
This layer puts a `pthread_mutex` around every Vulkan entry point the app
calls so only one thread is inside `libGLES_mali.so` at a time.

## Current status

- Layer correctly intercepts the full dispatch chain (`vkGetInstanceProcAddr`,
  `vkGetDeviceProcAddr`), Mali-G52 gate fires, CreateDevice succeeds.
- Kills the cross-thread TOCTOU: no more crash in the first minute.
- Does **not** cure the deeper issue: the threaded scheduler produces a
  call order inside VulkanWorker's own thread where Mali itself nulls
  `ctx+0x20` in one call and dereferences it in the next. That's a
  sequential Mali bug the mutex can't see. Crashes resume at
  2-4 min of course-preview cycling.

Kept on this branch as the foundation for:
- Any future device that hits the same Mali bug where the fault IS purely
  cross-thread (this layer would fix it outright).
- Reference for reverse engineering the Mali dispatch dance on Android —
  the only open documentation I know of for this driver's layer interface.

## Building

```bash
# 1. Build the layer .so
bash mali-layer/build.sh

# 2. Drop into jniLibs so the APK bundles it
cp mali-layer/build/libVkLayer_azahar_mali_serialize.so \
   src/android/app/src/main/jniLibs/arm64-v8a/

# 3. Build the APK normally
bash docker/android-build/build-android.sh
```

Azahar enables the layer unconditionally on Android via
`src/video_core/renderer_vulkan/vk_platform.cpp` (`ppEnabledLayerNames`).
Runtime gate in the layer (`maybe_arm_mali_gate`) makes it a no-op on
anything that doesn't report `Mali-G52` in `VkPhysicalDeviceProperties`.

## Android vs desktop Vulkan loader quirk

Android's Vulkan loader does NOT insert a `VkLayerDeviceCreateInfo` into
`VkDeviceCreateInfo::pNext` the way the Khronos desktop loader does.
The standard "advance the chain via VK_LAYER_LINK_INFO" dance finds null
and `vkCreateDevice` returns `VK_ERROR_INITIALIZATION_FAILED`.

Workaround in `L_CreateDevice`: use the `vkCreateDevice` and
`vkGetDeviceProcAddr` pointers already resolved through the instance
chain during `L_CreateInstance` — they already point past our layer
because the instance chain is set up normally.

## Companion: mali-shim/

Earlier attempt: replace `/vendor/lib64/hw/vulkan.rk356x.so` with a shim
that `dlopen`s the real Mali HAL and wraps its HMI. Works but requires
root + `adb disable-verity`, and an atomic file replacement (a live `cp`
over the mmapped HAL trashed systemui/launcher/system_server). Kept for
reference on non-Magisk root paths.
