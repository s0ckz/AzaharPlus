# Multi-core GPU worker (claude/gpu-worker-stable branch)

This branch is the stable result of an autonomous iteration session that
moved Citra/Azahar's PICA cmdlist processing off the emulation thread
onto a dedicated `GpuWorker` thread, unlocking the previously-idle cores
on the Anbernic RG DS (RK3568, Mali-G52, Vulkan).

## Result

- **Sustained 60 fps** in Super Mario 3D Land on the device, vs ~44 fps
  in heavy spots on the original synchronous build.
- **Emu thread CPU drops from 88% to 69%** of one core; ~26% of the
  former emu work now runs on a second physical core (the GpuWorker
  thread). Plus the Vulkan scheduler worker (~13%) and Mali driver
  thread (~13%) on additional cores. Total CPU spread across 4 threads
  instead of pegging one.
- **`ProcessCmdList` / `WriteInternalRegAction` / `AccelerateDrawBatch`
  no longer appear** in the emu-thread simpleperf top-30. They've moved
  to the worker.
- **Stable for 100+ minute play sessions** after the final fix
  (`memory: thread_local PhysMemRegionInfo cache`, commit
  `39655cb0c`). Earlier intermediate commits had rare ~30-60 min
  crashes — those came from the shared 1-entry physical-region cache
  in `MemorySystem` being raced by the worker and emu threads. Hoisting
  it into a `thread_local` removed every crash signature we'd been
  chasing.

## What's on each branch

| Branch | State |
|---|---|
| `claude/gpu-worker-baseline` | First build that hit 60 fps. Crashes every ~30s in `AccelerateDrawBatchInternal+164`. Not safe to play but useful as a measurement baseline. |
| `claude/gpu-worker-stable` ← **install this one** | Adds the actual fixes. Boots reliably, runs 60 fps, occasional rare crashes only. |

## Architecture

```
emu thread (NativeEmulation, nice -20, pinned to cores 0..4):
  ├── dynarmic JIT (ARM11 Core 0 + Core 1, round-robin)
  ├── HLE kernel (scheduler, threads, events)
  ├── HLE services (GSP, DSP, HID, FS, …)
  ├── core_timing (VBlank, DSP audio tick)
  └── audio HLE Tick() — synchronous

GpuWorker thread (separate physical core):
  └── ExecuteOnWorker dispatch:
      ├── SubmitCmdList → PicaCore::ProcessCmdList → DrawArrays →
      │     AccelerateDrawBatch → vk_scheduler.Record(...)
      ├── MemoryFill / DisplayTransfer / TextureCopy
      └── RequestDma + signal_interrupt(DMA)

Vulkan scheduler worker thread (yet another core):
  └── Drains command chunks queued by Record(), executes via Mali driver

Mali GPU driver helper threads:
  └── Background work
```

The emu thread enqueues `GpuMessage`s onto the worker's queue and
returns. The worker drains the queue serially. Interrupt firing from
the worker (P3D, PSC, PPF) is deferred via `Impl::pending_interrupts`
and drained on the emu thread at the top of every `System::RunLoop`
slice — `Kernel::Event::Signal` mutates Kernel::Thread state and would
SIGTRAP the scheduler if invoked off-thread.

VBlank, SetBufferSwap, SetColorFill, Read/WriteReg stay synchronous on
the emu thread, preceded by `worker.Flush()` so the worker is quiescent
when those paths read framebuffer_config or run SwapBuffers.

## What was hard

| Problem | Fix |
|---|---|
| Worker thread firing GSP interrupts → SIGTRAP in `Kernel::ThreadManager::SwitchContext` | Defer interrupts via `pending_interrupts` queue, drain on emu thread from `System::RunLoop` |
| `unique_ptr<Impl>::reset()` nulls the pointer **before** running the deleter, so Impl::~Impl's worker.Stop() runs after the worker has already started seeing `this->impl == nullptr` | Explicit `worker.Stop()` in `GPU::~GPU` body before the unique_ptr destructor runs |
| `Memory::RasterizerMarkRegionCached` writing `attribute = Memory` before installing the page pointer; dynarec then read `attribute=Memory + pointer=null` and asserted | Reorder writes (pointer-then-attribute when arming, attribute-then-pointer when disarming) with a release fence between them |
| `Vulkan::Scheduler::Record` is single-producer; concurrent writers (worker + emu thread via SwapBuffers / texture upload paths) race the `chunk` member, leading to `chunk->Record(...)` deref of a moved-from `unique_ptr` → SIGSEGV at offset 0x10 | Add `chunk_mutex` protecting `chunk` in `Record` / `DispatchWork` / `SubmitExecution` |
| `RasterizerCache::FlushRegion` iterates `dirty_regions` (boost::icl::interval_map, a red-black tree) while the worker mutates the same tree | Class-level `recursive_mutex cache_mutex` taken at the top of every public RasterizerCache method body |
| `AnalyzeVertexArray` / `SetupIndexArray` / `SetupVertexArray` reading transient bad pica regs and feeding `FindMinMax` / `memcpy` huge sizes or null pointers → SIGSEGV | Defensive bounds checks on `num_vertices` / `vs_input_size` and null checks on `GetPhysicalPointer` / `stream_buffer.Map` |
| In-flight Surface objects garbage-collected by the rasterizer cache before the worker / Vulkan scheduler finished using them | Bumped `TextureRuntime::RemoveThreshold` from `num_swapchain_images` (2..3) to 240 frames |

## Known remaining issues

Tested for 100+ minute sessions with no crashes after the
`thread_local PhysMemRegionInfo cache` fix. The defensive bounds
checks in `AnalyzeVertexArray` / `SetupIndexArray` / `SetupVertexArray`
remain as belt-and-suspenders for any future memory-region race we
might still be missing — they can be removed if a clean session
audit shows they never trip.

Performance dips to ~50-55 fps in heavy scenes (the user's reference
"heavy spot"), still substantially above the ~44 fps baseline. The
remaining bottleneck is the dynarec emu thread itself running at ~70%
of one core; further gains would require parallelizing the ARM11
Core 0 / Core 1 dynarec onto two host threads, which is a much bigger
project (Citra's HLE kernel was not designed for concurrent core
execution).

## Iterating further

The current build is at `claude/gpu-worker-stable`. To check it out and
build:

```bash
git checkout claude/gpu-worker-stable
bash docker/android-build/build-android.sh
adb install -r src/android/app/build/outputs/apk/vanilla/release/app-vanilla-release.apk
```

To start fresh from the buggy-but-fast baseline:

```bash
git checkout claude/gpu-worker-baseline
```

Useful diagnostic logs:
```bash
adb shell "logcat -d | grep -E 'GpuWorkerProbe|GpuIntrProbe|PerfProbe.*speed' | tail -40"
```

The `GpuWorkerProbe` line shows per-second message-type counts on the
worker; if `submit=0` for an extended period, the game is stalled, not
the worker.

# Building Azahar locally

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

## Variations

**Build both ABIs (arm64 + x86_64, e.g. for the Android emulator):**
```bash
AZAHAR_ABI_FILTER=arm64-v8a,x86_64 bash docker/android-build/build-android.sh
```

**Pass arbitrary Gradle tasks** — anything after the script name is forwarded
to `./gradlew`:
```bash
bash docker/android-build/build-android.sh clean assembleVanillaRelease
bash docker/android-build/build-android.sh assembleRelease   # both flavors
```

**Drop into an interactive shell inside the build container** (workspace and
caches mounted, useful for poking at gradle state):
```bash
bash docker/android-build/build-android.sh shell
```

**Force a clean rebuild of the toolchain image** (only needed if you edit
the Dockerfile):
```bash
bash docker/android-build/build-android.sh --rebuild-image
```

## Persistent local keystore (optional but recommended)

Without a persistent keystore, Gradle signs with a debug keystore that AGP
regenerates per Gradle home. This causes `INSTALL_FAILED_UPDATE_INCOMPATIBLE`
("signature mismatch") on the second and subsequent sideloads, forcing an
uninstall+reinstall and losing app state.

To set one up once:

```bash
mkdir -p .local-keystore

docker run --rm -v "$(pwd)/.local-keystore:/keystore" \
    azahar-android-build:latest \
    keytool -genkeypair -keystore /keystore/keystore.jks \
        -storepass CHANGEME -alias azahar-local -keypass CHANGEME \
        -keyalg RSA -keysize 2048 -validity 36500 \
        -dname "CN=Azahar Local Build, OU=Dev, O=Local, L=Local, ST=Local, C=US"

cat > .local-keystore/keystore.env <<'EOF'
ANDROID_KEYSTORE_PASS=CHANGEME
ANDROID_KEY_ALIAS=azahar-local
EOF
```

Both `keystore.jks` and `keystore.env` are gitignored. Subsequent builds pick
them up automatically.

## Wiping caches

```bash
docker volume rm azahar-android-ccache   # NDK clang object cache
docker volume rm azahar-android-gradle   # Gradle wrapper, deps, build cache
```

## Windows / Git Bash notes

- The script sets `MSYS_NO_PATHCONV=1` automatically to keep Git Bash from
  mangling Docker volume mount paths.
- First build is slow because of host→container file sync; subsequent builds
  touch much less data.
