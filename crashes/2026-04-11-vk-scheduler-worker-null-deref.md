# 2026-04-11 — VulkanWorker SIGSEGV at offset 0x60

**Status**: Open. Reproduces sporadically (~10–30 min) on `claude/gpu-worker-smooth` after the back-pressure change. **Not investigated yet.**

## Quick facts

| | |
|---|---|
| Branch / commit | `claude/gpu-worker-smooth` @ `e1655b4c0` (tagged `v1`) |
| Game | Super Mario 3D Land |
| Device | Anbernic RG DS (Rockchip RK3568, Cortex-A55, Mali-G52, Vulkan via libGLES_mali) |
| Time of crash | 2026-04-11 13:07:06 (process uptime 651s ≈ 10m 51s of gameplay) |
| Process / TID | pid 15676, **tid 17242 (`VulkanWorker`)** |
| Signal | SIGSEGV / SEGV_MAPERR, fault addr **`0x0000000000000060`** |
| Cause | Null pointer dereference (deref of `nullptr + 0x60`) |
| Crashing PC | `libGLES_mali.so + 0x1e05a38` (Mali userspace driver, frame #00) |

## Stack (top → bottom)

```
#00 libGLES_mali.so + 0x1e05a38                           (Mali driver — null deref)
#01 libGLES_mali.so + 0x9e67a8
#02 libcitra-android.so + 0x2137394                       (inlined helper inside Scheduler::WorkerThread)
#03 libcitra-android.so + 0x21442a0  Vulkan::Scheduler::WorkerThread(std::stop_token) + 748
#04 libcitra-android.so + 0x21447d0                       (jthread thunk)
#05 libc.so __pthread_start + 204
#06 libc.so __start_thread + 64
```

The fault is in the **Mali driver**, not in our code directly — frame #00 PC is inside `libGLES_mali.so`. Frame #02–#03 inside `libcitra-android.so` are the call site that handed Mali a bad argument: `Vulkan::Scheduler::WorkerThread` → `CommandChunk::ExecuteAll` → some recorded command's lambda → vk dispatch into Mali → null deref at offset 0x60 of one of the vk handles passed in.

Plain reading: the WorkerThread popped a chunk from `work_queue`, called `work->ExecuteAll(current_cmdbuf)` (vk_scheduler.cpp:159), and one of the recorded commands either (a) captured a vk handle that has been freed/recycled, or (b) holds a `vk::CommandBuffer` / pipeline / image / etc. whose internal pointer at offset 0x60 is null because the underlying object was destroyed.

## Important: this is the consumer-side crash, not the producer

Earlier in this branch we hit `AccelerateDrawBatchInternal+164` SIGSEGV from concurrent `Record()` calls — fixed with `chunk_mutex` in `vk_scheduler.h`/`.cpp`. **This new crash is downstream of that fix.** `chunk_mutex` protects `chunk` while it's being built up (the producer side in `Record()` and `DispatchWorkLocked()`). Once a chunk is `std::move`d into `work_queue`, the consumer (`WorkerThread`) owns it. The crash is in the consumer dereffing something the recorded lambdas captured.

So the lifetime bug isn't in Scheduler internals — it's in **what the lambdas captured**. Most likely a vk handle whose owning object got destroyed between `Record()` and `WorkerThread::ExecuteAll`.

## Perf data leading into the crash

From `crashes/2026-04-11-perf-around-crash.log`. Last 5 seconds before SIGSEGV (13:07:06):

```
13:07:01  sysFPS=??.?  (heavy spot, fluctuating ~48–55)
13:07:02  sysFPS=48.0  rest=17.71  submit=304/s
13:07:03  sysFPS=49.0  rest=17.69  submit=323/s
13:07:04  sysFPS=49.0  rest=17.27  submit=326/s
13:07:05  sysFPS=54.0  rest=16.03  submit=316/s
13:07:06  sysFPS=32.0  rest=18.74  submit=330/s   ← crash
```

Notable:
- Submit rate is **5×** the typical 60/s (304–330 cmdlists/s) — this is a heavy gameplay scene with lots of draws
- `xfer=128–138/s` (display transfers) — high, suggests lots of texture uploads or DTransfers each frame
- `setbufswap=66–74/s` — also high
- `gpu` bucket ticked up from 0.13ms → 0.45ms in the crashing second
- `psc=0` for many seconds — interesting, suggests no `MemoryFill` events recently
- `pdc0=0 pdc1=0` always — these are fired synchronously on the emu thread (we already moved them off the deferred path), so they don't show in the drain
- `vblank=49–53/s` — emu thread is hitting ~50 frames per wall second under load

The crash window has unusually high `xfer` and `submit` density, consistent with a heavy frame. Likely a transient surface eviction or texture re-upload race during enemy-heavy gameplay.

## Hypotheses (in order of plausibility)

1. **Surface lifetime race**. The `RasterizerCache` evicted a `Surface` whose `vk::Image` / `vk::ImageView` was still referenced by an in-flight `CommandChunk` lambda. We bumped `TextureRuntime::RemoveThreshold` from `num_swapchain_images` (2–3) to **240 frames** for exactly this reason (CLAUDE.md), but 240 frames at variable fps may still not be enough during a heavy spot if the cache churn rate exceeds the eviction guard window.
   - **Where to look**: `src/video_core/renderer_vulkan/vk_texture_runtime.cpp::RemoveThreshold()` — try bumping to a much larger value (e.g. 1024) and see if the crash goes away. If yes, fix the eviction guard to use a **submission tick** (like the master semaphore) instead of a frame counter.
   - Or: make `Scheduler::Record` capture `std::shared_ptr<Surface>` so the vk handles stay alive until the chunk is consumed. Bigger refactor.

2. **`current_cmdbuf` reused before previous submission completed**. In `vk_scheduler.cpp:181`, `AllocateWorkerCommandBuffers` calls `command_pool.Commit()` to get a new cmdbuf, but only AFTER `has_submit` is true on the popped chunk. If a non-submit chunk is being executed and it captured the `current_cmdbuf` from a previous Record(), and meanwhile another thread resets the pool... probably not it (single worker thread), but worth verifying.

3. **DescriptorSet / Pipeline cache eviction**. Pipeline cache evicts an in-use `vk::Pipeline`, the chunk's recorded command holds a stale handle, Mali derefs it at offset 0x60 (some internal ptr in the Mali pipeline object). Less likely because PSO caches in this codebase are append-only AFAIK.

4. **Race in `command_pool.Commit()` itself**. `command_pool` may not be thread-safe. The Scheduler::WorkerThread is the only one calling `AllocateWorkerCommandBuffers`, but `Record()` from the GpuWorker thread might trigger pool ops via the recorded lambdas. Investigate `vk_command_pool.cpp` for thread safety.

5. **Back-pressure interaction**. The new `kPendingExecuteHighWatermark=16` cap might be changing timing such that the worker now sustains pressure for longer windows, exposing a race that was statistically rare before. But the underlying bug is **not caused** by back-pressure — it just made it more reproducible.

## What's already been ruled out

- **Producer-side `chunk` race in `Scheduler::Record`** — fixed by `chunk_mutex` in commit history of this branch.
- **`thread_local PhysMemRegionInfo cache`** — the memory.cpp root-cause fix (commit `39655cb0c`) eliminated the dynarec page-table races. This crash doesn't touch memory.cpp.
- **Worker firing GSP interrupts off-thread** — fixed by `pending_interrupts` deferral.
- **`RasterizerCache::FlushRegion` tree iteration race** — fixed by `recursive_mutex cache_mutex`.

## Files saved alongside this note

- `crashes/2026-04-11-vk-scheduler-worker-crash.log` — full `logcat -b crash` output around the SIGSEGV (the registers + backtrace block).
- `crashes/2026-04-11-perf-around-crash.log` — `PerfProbe`/`GpuWorkerProbe`/`GpuIntrProbe` logcat lines for the ~5 seconds leading up to the crash.
- `crashes/2026-04-11-libcitra-android.so` — the **exact** binary that crashed (38 MB, BuildId `0913b84b3b2ade14c0e6b57463442574c50b61c8`). Pulled from the device's APK install path right after the crash. Use this for `addr2line` / disasm — pc offsets in the stack are valid against THIS binary. **Gitignored** (too big for the repo) — kept locally only. If lost, the same binary lives on the device under `/data/app/~~yFFMiYxpyIfoObxqdros0g==/io.github.lime3ds.android-llGQunlhiRDyclJKRHv74A==/lib/arm64/libcitra-android.so` until the next reinstall, OR can be rebuilt from commit `e1655b4c0` (tag `v1`) which has the matching BuildId.

## How to dig in next session

1. **Confirm BuildId matches**:
   ```bash
   docker run --rm -v "$(pwd):/work" -w /work azahar-android-build:latest \
     /opt/android-sdk/ndk/27.1.12297006/toolchains/llvm/prebuilt/linux-x86_64/bin/llvm-readelf \
     -n crashes/2026-04-11-libcitra-android.so | grep -i build
   # Should print: 0913b84b3b2ade14c0e6b57463442574c50b61c8
   ```

2. **Disasm the crash site** (PC = `Scheduler::WorkerThread + 748` = file offset `0x21442a0`). The `+748` decimal = `+0x2EC` from function start, so the function entry is at `0x21442a0 - 0x2EC = 0x2143FB4`:
   ```bash
   docker run --rm -v "$(pwd):/work" -w /work azahar-android-build:latest \
     /opt/android-sdk/ndk/27.1.12297006/toolchains/llvm/prebuilt/linux-x86_64/bin/llvm-objdump \
     -d --start-address=0x2143fb4 --stop-address=0x21442f0 \
     crashes/2026-04-11-libcitra-android.so | tail -80
   ```
   This is the inner loop of `WorkerThread` — the `work->ExecuteAll(current_cmdbuf)` call site. Look for what register is loaded right before the call into the inlined helper at frame #02 (`+0x2137394`).

3. **Identify the inlined function at `+0x2137394`** — same approach. Likely `CommandChunk::ExecuteAll` or one of the recorded `Command::Execute` virtual calls. The recorded commands live in `src/video_core/renderer_vulkan/vk_scheduler.h::Command` derived classes (search for `command->Execute(cmdbuf)` callsites).

4. **Greppable starting points**:
   - `src/video_core/renderer_vulkan/vk_scheduler.h` — `class Command`, `class CommandChunk`
   - `src/video_core/renderer_vulkan/vk_texture_runtime.cpp::RemoveThreshold` — current value 240
   - `src/video_core/rasterizer_cache/rasterizer_cache.h` — `TickFrame` and surface eviction logic
   - Any `Record(...)` callsite that captures a `Surface*`, `vk::Image`, `vk::ImageView`, or `vk::Pipeline` by value — those are the suspects for stale-handle capture.

5. **Try the cheap fix first**: bump `RemoveThreshold()` to 1024 or even 2048 and see if the crash stops reproducing under heavy load. If it does, the root cause is confirmed as Surface eviction beating the in-flight window, and the proper fix is to track per-Surface-last-used **submission ticks** instead of frame counts.

6. **Capture another fresh tombstone if possible**:
   ```bash
   adb logcat -d -b crash > crashes/<date>-<thread>-crash.log
   adb pull /data/app/.../lib/arm64/libcitra-android.so crashes/<date>-libcitra-android.so
   ```

## TL;DR for future me

VulkanWorker thread crashed at `Scheduler::WorkerThread+748`, fault inside `libGLES_mali.so` derefing a pointer at offset `0x60`. Almost certainly a vk handle (likely an `ImageView` or `Image`) captured by a Recorded lambda whose owning Surface got evicted before the WorkerThread executed the chunk. Cheap fix to test: bump `TextureRuntime::RemoveThreshold()`. Real fix: track per-surface last-used submission tick instead of frame count, and refuse eviction until `master_semaphore->KnownTick() >= surface->last_used_tick`.
