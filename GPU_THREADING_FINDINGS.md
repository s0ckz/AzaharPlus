# GPU Threading Findings — Mali G52 on RK3568

This document captures the investigation into multi-threaded GPU emulation for
Azahar on Anbernic RG DS (Rockchip RK3568 SoC, Mali G52 GPU, Vulkan renderer).
The goal was to push beyond the single-threaded `deploy/fastmem-only` baseline
toward 60fps. What follows is what we learned, so nobody has to rediscover it.

## TL;DR

- **Mali G52 driver corrupts its internal state when two threads are inside
  `libGLES_mali.so` concurrently.** Crash signature: `SIGSEGV fault addr 0x60`
  on whichever thread got preempted mid-call. This is a driver bug. We can
  only work around it.
- **There is no combination of descriptor staging, thread pinning, or a
  global Vulkan mutex that fully prevents the crash** while keeping the
  GpuWorker enabled. Each mitigation reduces crash frequency but never to
  zero. See the table below.
- **Single-threaded Vulkan never crashes.** `deploy/fastmem-only` is the
  reliable baseline (~40 fps SMB3DL heavy scene, 60 fps main menus, never
  crashes in multi-hour sessions).
- **Phase 2 micro-optimizations on single-thread don't help in practice.**
  SyncDrawState dirty tracking and larger command chunks made no measurable
  FPS difference on MK7 or SMB3DL. The bottleneck on heavy scenes is
  dynarmic JIT (`rest=15-16 ms/frame`), not rasterizer CPU work
  (`gpu=8-9 ms/frame`).

## The driver bug, in detail

The crash is always the same:

```
Fatal signal 11 (SIGSEGV), code 1 (SEGV_MAPERR), fault addr 0x60
in tid <N> (VulkanWorker | NativeEmulation | VulkanPresent)
  libGLES_mali.so  (inside vkCmd* / vkQueueSubmit / vkUpdateDescriptorSets)
  Vulkan::Scheduler::WorkerThread+<offset>
```

Offset `0x60` is the same across every crash — it is a field in a per-thread
structure Mali keeps internally. The driver is not thread-safe for
concurrent API calls, even on different VkCommandBuffers, even when those
calls target different queues (there is only one graphics queue on Mali G52
anyway — `queueCount == 1` on the single graphics-capable queue family).

Concrete confirmation that pinning does not fix this:

- `sched_setaffinity` pins `VulkanWorker`, `VulkanPresent`, and `GpuWorker`
  to CPU core 3. Verified via `/proc/<tid>/status`: `Cpus_allowed_list: 3`.
- The OS still preempts threads mid-syscall. When thread A is inside
  `libGLES_mali.so` and the kernel swaps in thread B (also pinned to core
  3) which also enters Mali, the first thread's state is corrupted.
- Crash interval improves from 3–8 min (no pinning) to ~34 min (pinned),
  but never reaches zero.

## What we tried, ranked by stability

Measured on MK7 main menu with course-preview cycling (crash-prone scenario
from prior stability harness):

| Variant | Avg crash interval | Notes |
|---|---|---|
| Single-thread (`deploy/fastmem-only`) | never crashes in 2h+ | ~40 fps heavy scenes |
| v2 GpuWorker, no protection | 3–8 min | ~60 fps |
| + descriptor staging | 12–15 min | staging moves `vkUpdateDescriptorSets` to VulkanWorker only |
| + inline VulkanWorker (no separate thread) | 9–32 min (high variance) | ~55 fps |
| + thread pinning (all Vulkan threads on same core) | ~34 min | ~60 fps |
| + global `mali_mutex` (partial wrap) | ~21 min | made it **worse** — added mutex latency perturbs timing |
| GpuWorker + pinning + `mali_mutex` wide wrap | games hang at boot | deadlock |
| GpuWorker + pinning + `mali_mutex` narrow wrap | 21 min, glitchy frames | mutex breaks submit ordering |

Key observations:

1. **Every added synchronization primitive either deadlocks or perturbs
   timing in a way that makes crashes MORE frequent, not less.** Adding
   locks changes which threads get preempted when, exposing different
   races. We never found a lock layout that was strictly better.
2. **Descriptor staging alone (v25b) gives a clean 2–3× stability
   improvement** (12–15 min vs 3–8 min). It is the one mitigation that
   did not regress. It moves one Vulkan call (`vkUpdateDescriptorSets`)
   from emu thread to VulkanWorker — reducing, but not eliminating, the
   number of threads simultaneously inside Mali.
3. **Deferred draws break the game's interrupt model.** The PICA
   pipeline's interrupt-to-rendering contract assumes draws complete
   before their completion interrupt fires. Deferring draws until
   VBlank and firing interrupts early caused black screens, halved FPS,
   and out-of-order frame composition. Not a viable path.
4. **`use_worker_thread=false` (inline VulkanWorker) had latent bugs.**
   The path existed in the scheduler but was never exercised — chunks
   weren't being executed, `AcquireNewChunk` was skipped at construction,
   and `WaitWorker` was a no-op. Patchable, but didn't help stability.

## Why the "fast" build is not actually fast

With `GpuWorker + pinning + mali_mutex`, SMB3DL heavy scene FPS was 37–46
— roughly the same as pure `fastmem-only` at ~40 fps. The mali_mutex
serializes all Vulkan calls, which eats the parallelism that GpuWorker
was supposed to buy. Net win: zero. Net loss if the app still crashes.

Per-frame breakdown (SMB3DL heavy, fastmem-only, frame ~25 ms → 40 fps):

- `rest = 15–16 ms` — dynarmic JIT + HLE kernel (already using every
  `Unsafe_*` optimization that is safe to enable; this is the true
  bottleneck)
- `gpu = 8–9 ms` — PICA command processing + rasterizer CPU work
- `swap = 0.2 ms` — SwapBuffers + GPU wait (not a bottleneck)
- Everything else < 1 ms

To reach 60 fps (16.7 ms/frame) we would need to eliminate nearly all
GPU-side CPU work AND shave ~2 ms off dynarmic — not realistic without
a working GpuWorker. And a working GpuWorker requires the Mali driver
to be thread-safe, which it is not.

## What's in this branch

`deploy/fastmem-only` contains:

- Dockerized Android build (`docker/android-build/build-android.sh`)
- Persistent local keystore handling
- Fastmem via `memfd_create` + 4 GB host VA reservation for dynarmic
- `sync_fastmem_reservation_with_RasterizerMarkRegionCached` — page-table
  writes are ordered with `std::atomic_thread_fence` so the rasterizer
  cache transitions safely between fastmem-mapped and callback-mapped
- Defensive bounds checks on `AnalyzeVertexArray`, `SetupIndexArray`,
  `SetupVertexArray` (these were added while debugging GpuWorker
  crashes, but the guards are cheap and correct to keep on the
  single-thread path too — they protect against malformed PICA state)

It does **not** contain:

- `GpuWorker` thread (source of the Mali races)
- `chunk_mutex` in `Scheduler::Record` (unnecessary without concurrent
  producers)
- Any `mali_mutex` / affinity / staging changes — those are in
  `deploy/fastmem-optimized` for reference, but that branch crashes
  and is not shipped.

## What to try if you come back to this

If the Mali driver ever gets fixed (new Mali G52 driver version in a
Rockchip vendor tree, a hacked Adreno-style `adrenotools` shim, etc.),
the path to 60 fps is:

1. Cherry-pick the GpuWorker commits from `claude/fastmem` (`98554cea0`
   and up)
2. Keep `mali_mutex` **off** (it hurt more than it helped)
3. Keep thread pinning **off** (same reason)
4. Verify against the `apk-snapshots/` test-automation scripts

Until then, **do not ship anything with `GpuWorker` enabled.** The
stability benefit of single-threaded Vulkan outweighs any FPS gain
from thread offload given how fragile the Mali driver is.

## See also

- `claude/fastmem` — scratch branch with every multi-threading experiment
  preserved commit-by-commit. Use this as a reference, not a base.
- `deploy/fastmem-optimized` — Phase 2 micro-optimizations + all the
  threading experiments layered on top. Did not produce a shippable build.
