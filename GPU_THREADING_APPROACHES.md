# GPU Multi-Threading Approaches — What's Been Tried

## The Goal

Move PICA GPU command processing off the emu thread onto a separate thread to unlock idle CPU cores on the RK3568 (Mali G52, Vulkan). Target: 60fps (up from ~46fps single-threaded baseline).

## Performance Baseline (from perf logs)

| Build | FPS | frame | gpu | rest | notes |
|---|---|---|---|---|---|
| AzaharPlus main (no worker) | ~46 | 22ms | 7ms | 12ms | gpu + rest on same thread |
| v2 (GpuWorker + VulkanWorker) | ~60 | 14ms | 3.2ms | 6.3ms | gpu offloaded, emu thread free |
| Fastmem-only (deploy/fastmem-only) | ~46 | 22ms | 7ms | 12ms | no threading, rock stable |
| Inline execution (deploy/gpu-inline) | VERY SLOW | - | - | - | no pipeline parallelism |

## Architecture

### Original single-threaded (AzaharPlus main)
```
Emu thread: dynarmic JIT → PICA cmdlist → Vulkan Record() → [VulkanWorker executes chunks]
```
Everything serialized on emu thread except VulkanWorker chunk execution.

### v2 three-thread model (fast, crashes)
```
Emu thread:      dynarmic JIT, HLE kernel, services, timing
GpuWorker:       PICA ProcessCmdList → rasterizer → Vulkan object creation + Record()
VulkanWorker:    executes command chunks (vkCmd* calls), submits to GPU
```
Emu thread pushes GpuMessages → GpuWorker drains FIFO → VulkanWorker executes.

### The Mali G52 TLS Problem
Mali's Vulkan driver uses **per-thread TLS** for internal object state. When a VkImage/VkImageView is created on Thread A (GpuWorker) but referenced in vkCmd* calls on Thread B (VulkanWorker), the driver dereferences a null TLS entry → **SIGSEGV at fault addr 0x60**.

This is a **driver bug** (Vulkan spec allows cross-thread object use), but we can't fix the driver.

---

## Approach 1: v2 as-is (3 threads, no crash fixes)

**Branch:** `claude/fastmem` at tag `v2` (commit `601dc292a`)
**Commits:** `98554cea0` through `601dc292a`

**What it does:**
- GpuWorker offloads PICA processing to worker thread
- VulkanWorker (Scheduler's worker_thread) executes recorded chunks
- chunk_mutex protects concurrent Record() calls
- cache_mutex (recursive) on all rasterizer cache public methods
- Deferred interrupt delivery (pending_interrupts queue)
- thread_local PhysMemRegionInfo cache (fixes shared cache race)
- Back-pressure watermark (kPendingExecuteHighWatermark = 16)

**Result:** 60fps. Crashes within ~3-8 minutes during actual gameplay (crashes after finishing 2nd course, pressing to go to 3rd).

**Crash signature (tombstone_17, pure v2):**
```
Thread: VulkanWorker
signal 11 (SIGSEGV), fault addr 0x60
#03 Vulkan::Scheduler::WorkerThread+748
#00-#01 libGLES_mali.so
```
Confirmed: Mali G52 TLS crash. VulkanWorker executes chunk referencing objects created on GpuWorker thread. This is the ROOT CAUSE — not FlushRegion, not surface destruction.

---

## Approach 2: Drain scheduler before destroying surfaces

**Commit:** `66f839d11`

**What it does:**
- Before RunGarbageCollector destroys expired sentenced surfaces, call `runtime.WaitForWorker()` to drain the VulkanWorker's work queue
- Bumped RemoveThreshold from 240 to 4096 frames (~68 seconds at 60fps)
- Added `TextureRuntime::WaitForWorker()` wrapper

**Result:** Still crashed. The crash isn't just use-after-free (destroyed surface handles) — it's the Mali TLS issue where LIVE objects created on GpuWorker crash when used on VulkanWorker.

---

## Approach 3: Sentence Fill surfaces

**Commit:** `946ea5a00`

**What it does:**
- Fill surfaces (SurfaceType::Fill) were the one type immediately destroyed via `slot_surfaces.erase()` in UnregisterSurface
- Changed to sentence ALL surface types (including Fill)

**Result:** Killed fps. Fill surfaces are high-frequency (created+destroyed every color fill op). With RemoveThreshold=4096, the sentenced list bloated → WaitWorker triggered 60x/sec → massive sync overhead.

---

## Approach 4: Short-sentence Fill surfaces (4 frames)

**Commit:** `d5b0d8f76`

**What it does:**
- Fill surfaces sentenced with adjusted tick so they expire after ~4 frames
- Normal surfaces keep full RemoveThreshold (4096)

**Result:** Crash interval improved but still ~30-90 minutes. Commit message explicitly acknowledges: *"the crash still occurs because VkImageView destruction happens on the GpuWorker thread while the VulkanWorker may still be mid-chunk. The proper fix is deferred destruction via scheduler.Record()."*

---

## Approach 5: Proxy Vulkan object creation through VulkanWorker

**Commit:** `d0d4bd8b1`

**What it does:**
- Handle struct gains `Scheduler*` pointer
- When set, `Handle::Create()`, `Surface::ImageView()`, `Surface::Framebuffer()` wrap Vulkan creation calls in `scheduler.Record([&]{...}); scheduler.WaitWorker();`
- Objects are created ON the VulkanWorker thread (same thread that uses them in vkCmd*)

**Result:** Crash interval improved to 76+ minutes (MK7). But `Record+WaitWorker` is **synchronous** — the GpuWorker blocks waiting for the VulkanWorker to create each object. This serializes object creation, destroying the parallelism that gave 60fps. Performance was bad.

---

## Approach 6: Proxy ALL Vulkan creation + descriptor staging

**Commit:** `49fd74054` + uncommitted dirty changes

**What it does:**
- Extended proxy to: vkCreateGraphicsPipeline, vkCreateSampler, vkCreateRenderPass, vkCreateFramebuffer (in vk_render_manager.cpp)
- Added `pre_execute` callback to Scheduler (runs on VulkanWorker before each chunk)
- Added `SwapToStaging()` / `FlushStaging()` to DescriptorUpdateQueue for double-buffered descriptor writes
- PipelineCache's on_dispatch changed from `update_queue.Flush()` to `update_queue.SwapToStaging()`, pre_execute does `FlushStaging()`

**Result:** Introduced deadlocks. Mali driver occasionally deadlocked during command execution, which WaitWorker propagated as a hang. Performance still bad due to serialized creation.

---

## Approach 7: Inline chunk execution (merge VulkanWorker into GpuWorker)

**Branch:** `deploy/gpu-inline` (my Phase 1 attempt)

**What it does:**
- Disable Scheduler's VulkanWorker thread (`SetInlineMode(true)`)
- Chunks execute inline on the calling thread (GpuWorker) instead of being dispatched to VulkanWorker
- Route VBlank/SwapBuffers through GpuWorker (push + Flush)
- Route FlushRegion through GPU::FlushRegion → GpuWorker (push + Flush)
- All Vulkan object creation AND command execution on same thread → no Mali TLS issue

**First deploy:** Crashed. `memory.cpp:401` calls `rasterizer->FlushRegion()` **directly**, bypassing `GPU::FlushRegion()` routing. The emu thread hit the Mali TLS crash in Surface::Download → SubmitExecution → DispatchWorkLocked.

**Second deploy (fixed memory.cpp routing):** VERY SLOW. No speed improvement over single-threaded baseline.

**Why slow:** Inline execution removes the **pipeline parallelism** between GpuWorker (recording) and VulkanWorker (executing). In v2, the VulkanWorker executes chunk N while the GpuWorker records chunk N+1. With inline, the GpuWorker does both — recording AND execution — sequentially. Plus `FlushRegion` routed through the worker adds synchronous round-trips (push + Flush blocks emu thread).

---

## Options To Try (ordered by simplicity)

### Option C: 3-thread + proxy only creation (= DUPLICATE of Approach 5)
**Status: TRIED — ~HALF FPS (~30fps)** (branch: `deploy/option-c`)
**Same as approach 5 (commit d0d4bd8b1).** Re-implemented the same thing.

Keep v2's 3-thread parallelism intact. Add `Record+WaitWorker` ONLY for Vulkan object creation calls:
- `Handle::Create()` — vmaCreateImage + vkCreateImageView
- `Surface::ImageView()` — lazy vkCreateImageView
- `Surface::Framebuffer()` — lazy vkCreateFramebuffer
- `GraphicsPipeline::Build()` — vkCreateGraphicsPipelines
- `RenderManager::CreateRenderPass()` — vkCreateRenderPass
- `Sampler::Sampler()` — vkCreateSampler
- `Framebuffer::Framebuffer()` — vkCreateFramebuffer

Command recording (Record with vkCmd* lambdas) stays on GpuWorker — NOT proxied. Only creation is proxied. This preserves recording/execution parallelism.

Also add: sentence Fill surfaces (approach 4) + route FlushRegion through GPU (from approach 7/inline).

Approach 5 partially did this and got best stability. The difference: approach 5 also proxied descriptor updates and added complex staging buffers. This time, keep it simple — only proxy creation, nothing else.

**Result:** Slower than v2. WaitWorker per creation call stalls the GpuWorker pipeline. The serialization overhead from blocking on every VkImage/VkImageView/VkFramebuffer/VkSampler/VkRenderPass creation is too much. Stability unknown (not tested long enough due to fps regression).

---

### Option C-minimal: v2 + FlushRegion routing + Fill sentencing
**Status: TRIED — SLOW** (branch: `deploy/option-c-minimal`)

Two sub-attempts:

**C-minimal v1 (FlushRegion routing + Fill sentencing):** Slow. Fill sentencing with RemoveThreshold=240 bloats the sentenced list (hundreds of Fill surfaces accumulate). Same issue as approach 3.

**C-minimal v2 (FlushRegion routing only, no Fill sentencing):** Still slow. The `Push+Flush` round-trip for FlushRegion blocks the emu thread waiting for the entire GpuWorker queue to drain. This kills the emu/worker parallelism that gives 60fps. ANY synchronization (worker.Flush()) on the FlushRegion path is too expensive because FlushRegion is called from memory read callbacks.

**Conclusion:** Routing FlushRegion through the GpuWorker is fundamentally incompatible with performance. The sync cost is too high.

---

### Option C-zero: Skip descriptor flush from emu thread
**Status: TRIED — CRASHES AT LAUNCH**

Skipping descriptor flush from emu thread broke Vulkan submission. The VulkanPresent thread hit `Unreachable code!` at `vk_present_window.cpp:483`. The emu thread's SwapBuffers path needs descriptors flushed during SubmitExecution. Skipping the flush caused stale/unflushed descriptor state.

**Conclusion:** on_dispatch descriptor flush cannot be conditionally skipped. Both emu thread and GpuWorker dispatch chunks that need flushed descriptors.

---

### Pure v2 crash analysis
**Status: CONFIRMED** — deployed pure v2, crashed after ~8 min gameplay

The v2 crash is definitively the Mali G52 TLS issue on the VulkanWorker:
```
Thread: VulkanWorker
signal 11 (SIGSEGV), fault addr 0x60
Vulkan::Scheduler::WorkerThread+748 → libGLES_mali.so
```
Objects created on GpuWorker, used in vkCmd* on VulkanWorker → null TLS deref.

This is NOT FlushRegion. NOT surface destruction. The fundamental problem is cross-thread Vulkan object creation/usage.

---

### Option B: Deferred destruction on VulkanWorker
**Status: NOT TRIED**
**Effort: Small** — Record destroy lambdas instead of direct destroy
**Expected fps: same as v2 (~60)** — no overhead added

Instead of destroying VkImage/VkImageView on the GpuWorker or emu thread, wrap destruction in `scheduler.Record([handle]{vkDestroy(handle);})`. The VulkanWorker destroys objects on its own thread after all prior chunks have consumed the handles.

Fixes: use-after-free crashes (surface destroyed while VulkanWorker has in-flight refs).
Does NOT fix: Mali TLS creation crash (objects still created on GpuWorker, used on VulkanWorker).

**Combine with Option C** for full fix: proxy creation + deferred destruction.

---

### Option A: Async proxy creation (no WaitWorker)
**Status: NOT TRIED**
**Effort: Medium** — needs handle indirection or two-phase setup
**Expected fps: ~60** (no synchronous waits)

Record creation lambdas into the command stream WITHOUT blocking. Objects are created on VulkanWorker in-order, before the draw command that references them.

Challenge: GpuWorker needs VkImageView handles immediately to set up descriptor sets. Without WaitWorker, the handle doesn't exist yet when the GpuWorker configures descriptors.

Possible solutions:
- Two-phase: Record creation lambda that writes to a shared slot. Descriptor setup also happens in a Record lambda that reads from the slot. Both execute on VulkanWorker in order.
- Future-style handles: GpuWorker gets a "promise" that resolves when VulkanWorker executes the creation.

Both require refactoring how descriptor sets and framebuffer configs reference Vulkan handles.

---

### Option F: FlushRegion without Vulkan calls
**Status: NOT TRIED**
**Effort: Medium** — persistent mapped staging buffers
**Expected fps: ~60** (removes a sync point)

Make FlushRegion (GPU→CPU data download) work without Vulkan API calls on the emu thread. Use persistently mapped staging buffers so the CPU can read GPU data directly without vkCmdCopyImageToBuffer.

This removes the need to route FlushRegion through the GpuWorker, eliminating a synchronous round-trip.

Combine with Option C for creation fix.

---

### Option D: Pre-create object pools
**Status: NOT TRIED**
**Effort: Large** — pool allocator + format/size bucketing
**Expected fps: ~60** (zero creation overhead per frame)

Pre-allocate VkImage/VkImageView/VkFramebuffer pools on the VulkanWorker thread during init. GpuWorker picks from the pool instead of creating new objects. Return to pool on "destroy" instead of actually destroying.

Challenge: 3DS games create surfaces with varying pixel formats, dimensions, and mip levels. Pool must bucket by format+size or be flexible enough to reuse.

---

### Option E: Lazy migration
**Status: NOT TRIED**
**Effort: Large** — cross-thread detection + recreation logic
**Expected fps: ~58-60** (one-time cost per object)

Create objects on GpuWorker normally. When VulkanWorker first encounters an object from a different thread, re-create it on its own thread using saved creation parameters.

Requires: storing creation params alongside each object, detecting cross-thread origin, and handling the old→new handle swap atomically.

---

### Option G: Hybrid async creation with future-style handles
**Status: NOT TRIED**
**Effort: Very Large** — major refactor of handle ownership model
**Expected fps: ~60** (full parallelism, zero sync waits)

Keep v2's 3-thread parallelism. Use async `Record()` creation WITHOUT WaitWorker. All Vulkan handle references become "futures" that resolve when the VulkanWorker executes the creation lambda. Descriptor setup, framebuffer config, and draw recording all happen inside Record() lambdas, so they naturally execute after creation.

This is the "ideal" solution but requires rearchitecting how the rasterizer interacts with Vulkan handles.

---

---

### Option A: Async creation (staging buffer + pointer indirection)
**Status: ABANDONED DURING IMPLEMENTATION**

Fundamental timing issue: creation lambdas execute INSIDE the chunk, but descriptor flush (FlushStaging) must happen BEFORE chunk execution to provide descriptors to draws. For surfaces created in the same chunk as their first draw, the handle doesn't exist when descriptors are flushed.

Would require separating creation into a prior chunk or restructuring the entire descriptor pipeline. Too complex.

---

### Option C-targeted: Proxy ONLY Handle::Create (not ImageView/Framebuffer/Sampler/RenderPass) ← TESTING
**Status: BUILDING**
**Effort: Tiny** — only Handle::Create proxied, everything else stays v2

Key insight: Option C was slow because it proxied EVERY Vulkan creation call (ImageView, Framebuffer, Sampler, RenderPass — these are called frequently on lazy first-use). Handle::Create (VkImage + VkImageView allocation) is only called on rasterizer cache MISS, which is rare during steady gameplay.

Changes: proxy Handle::Create via Record+WaitWorker + Fill sentencing + guard on_dispatch.

---

## Current Attempt Queue

1. **Option C-targeted** ← NOW (proxy only Handle::Create, minimal WaitWorker calls)
2. **Option B** (deferred destruction if needed)
3. Ship fastmem-only (46fps stable) if nothing works

---

## Key Constraints

1. **Mali G52 TLS:** Objects must be created and used (in vkCmd*) on the same thread. This is a driver bug we cannot fix.
2. **Performance:** The 60fps gain comes from pipeline parallelism between GpuWorker (record) and VulkanWorker (execute). Removing either thread or serializing between them kills the gain.
3. **FlushRegion:** Called from emu thread via memory callbacks. Involves Vulkan API calls (Surface::Download → vkCmdCopyImageToBuffer). Must either run on the Vulkan thread or avoid Vulkan calls.
4. **InvalidateRegion:** Hot path from emu thread. Only cache bookkeeping (no Vulkan calls). Can stay on emu thread.
5. **SwapBuffers:** Creates/uses presentation Vulkan objects. Currently runs on emu thread after worker.Flush(). Needs to be on the Vulkan thread or use emu-thread-created objects only.
