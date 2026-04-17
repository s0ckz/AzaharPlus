# Building Azahar locally

This branch (`deploy/fastmem-only`) is the **stable** Azahar build for
Anbernic RG DS (Rockchip RK3568, Mali G52, Vulkan). It is single-threaded
on the GPU side — all Vulkan work stays on the emu thread + VulkanWorker
(no separate `GpuWorker`). This is intentional: the Mali G52 driver
crashes under concurrent Vulkan API access, and no workaround
we could find reliably prevented that. See `GPU_THREADING_FINDINGS.md`
for the full investigation.

Performance envelope on the RG DS:
- 3DS main menus, light scenes: 60 fps
- SMB3DL heavy scenes: ~40 fps (`rest=15 ms` dynarmic-bound, not GPU-bound)
- MK7 gameplay: 50–60 fps, drops in the most complex scenes

This branch ships a Dockerized Android build so you can produce a signed
sideload-installable APK without installing the Android SDK/NDK on the host.

## Prerequisites

- Docker Desktop (running, with the drive containing this repo enabled under
  Settings → Resources → File sharing)
- `adb` in PATH (only needed for installing the APK)

## Build

```bash
bash docker/android-build/build-android.sh
```

That's it. Default behavior:

- Builds the toolchain image on first run (slow, ~2.5 GB, cached after that)
- Runs `./gradlew assembleVanillaRelease --stacktrace` inside the container
- Builds **arm64-v8a only** (skips x86_64 to halve build time)
- Uses named Docker volumes `azahar-android-ccache` and `azahar-android-gradle`
  so subsequent builds are incremental
- Signs with the persistent keystore at `.local-keystore/keystore.jks` if
  present; otherwise falls back to the Gradle debug keystore

Output:

```
src/android/app/build/outputs/apk/vanilla/release/app-vanilla-release.apk
```

## Install on device

```bash
adb install -r src/android/app/build/outputs/apk/vanilla/release/app-vanilla-release.apk
```
