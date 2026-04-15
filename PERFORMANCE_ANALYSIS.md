# Performance Analysis — What the GpuWorker Optimizes

## Baseline Frame Breakdown (no worker, AzaharPlus main + fastmem)

| Component | Time | What it does |
|---|---|---|
| `gpu` | **7ms** | PICA ProcessCmdList + WriteInternalRegAction + DrawArrays + rasterizer state upload + Vulkan Record + vkUpdateDescriptorSets |
| `rest` | **12ms** | Dynarmic JIT (ARM11 emulation) + HLE kernel + memory callbacks |
| `swap` | 0.2ms | SwapBuffers/presentation |
| `tmr` | 0.7ms | Core timing |
| `svc/ipc` | 0.5ms | HLE service calls |
| **Total** | **~22ms** | **~46fps** |

## With GpuWorker Frame Breakdown

| Component | Time | Change |
|---|---|---|
| `gpu` | **3.2ms** | -54% (PICA moved to worker, only VulkanWorker overhead remains) |
| `rest` | **6.3ms** | -47% (emu thread freed from GPU blocking) |
| `swap` | 0.3ms | same |
| `tmr` | 3.3ms | +2.6ms (timing overhead from worker sync) |
| **Total** | **~14ms** | **~60fps** |

## What the GpuWorker Actually Moves Off the Emu Thread

The `gpu` 7ms consists of:

### CPU-side work (~4ms, NO Vulkan API calls):
- `PicaCore::ProcessCmdList` — walk PICA command buffer, decode register writes
- `WriteInternalRegAction` — interpret register writes, trigger draw calls
- `AnalyzeVertexArray` + `SetupVertexArray` + `SetupIndexArray` — vertex/index buffer prep
- Uniform upload to stream buffer (CPU memcpy)
- Pipeline state hash lookup in `PipelineCache`
- Texture surface lookup in `RasterizerCache` (GetSurface, GetTextureSurface)

### Vulkan API calls (~3ms):
- `vkUpdateDescriptorSets` — descriptor writes (via on_dispatch, once per chunk dispatch)
- `vkGetSemaphoreCounterValueKHR` — ResourcePool::Refresh (when pool exhausted)
- `vkWaitSemaphoresKHR` — StreamBuffer::Wait (when buffer wraps, rare)
- `vmaCreateImage` + `vkCreateImageView` — Handle::Create (on cache miss, rare)
- `scheduler.Record()` — stores lambdas in chunk (CPU only, no Vulkan)
- `scheduler.DispatchWork()` — pushes chunk to VulkanWorker queue (CPU only)

## The Mali G52 Constraint

Mali G52's Vulkan driver crashes when ANY two threads call Vulkan APIs simultaneously. This means the GpuWorker cannot make ANY Vulkan calls while the VulkanWorker is executing chunks.

## What We Fixed (descriptor staging, ~13-20 min stability)

Moved `vkUpdateDescriptorSets` from GpuWorker to VulkanWorker via staging buffer. Moved `Refresh` into submit lambda. These were the most frequent concurrent calls.

## What's Still Concurrent (remaining crash source)

| Call | Thread | Frequency | Can be fixed? |
|---|---|---|---|
| `vkGetSemaphoreCounterValueKHR` (ResourcePool::Refresh) | GpuWorker | Every pool exhaust | NO — deadlocks when routed to VulkanWorker (called from BOTH threads) |
| `vkWaitSemaphoresKHR` (StreamBuffer::Wait) | GpuWorker | Rare (buffer wrap) | Spin-wait helps but fallback still concurrent |
| `vkUpdateDescriptorSets` (3 presentation descriptors) | Emu thread | Every VBlank | Can't stage without breaking init |
| `vmaCreateImage` + `vkCreateImageView` | GpuWorker | On cache miss | Proxy deadlocks with priority channel |

## Key Insight: CPU Work is Most of the Gain

The GpuWorker's main benefit is moving **CPU-side PICA processing** (~4ms) off the emu thread. The Vulkan calls (~3ms) are secondary. The `rest` reduction (12→6ms) comes from the emu thread no longer BLOCKING on GPU work — it runs dynarmic while the GpuWorker handles PICA.

## Options for Performance Without Full GPU Parallelism

### Option A: Keep GpuWorker but disable VulkanWorker thread
- GpuWorker does PICA + Record (CPU) + execute chunks inline (Vulkan)
- No concurrent Vulkan access (only one Vulkan thread)
- TESTED: "inline execution" — was slow (~30fps) because recording + execution are sequential
- The pipeline parallelism between Record and Execute is essential for 60fps

### Option B: Optimize the single-thread path (no GpuWorker)
Reduce the 7ms `gpu` and 12ms `rest` on the emu thread:
- **Batch descriptor updates**: reduce vkUpdateDescriptorSets calls
- **Cache pipeline binds**: skip redundant vkCmdBindPipeline
- **Optimize PICA register walks**: skip unchanged registers
- **Reduce rasterizer cache lookups**: cache last-used surfaces
- **Fastmem already done**: saves ~2ms from page table walks
- **Unsafe FP already done**: saves ~1ms from FPCR emulation
- Potential: maybe 3-4ms savings → ~50fps instead of 46fps

### Option C: Partial GpuWorker — move ONLY CPU work to worker
- GpuWorker does: PICA ProcessCmdList + register walks + vertex analysis + uniform uploads
- GpuWorker does NOT call: Record, DispatchWork, descriptor flush, ResourcePool, StreamBuffer
- Emu thread picks up the prepared state and does all Vulkan work synchronously
- This separates CPU prep from Vulkan execution
- MAJOR REFACTOR: requires splitting the rasterizer draw path into "prepare" and "execute" phases
- Potential: ~4ms savings (CPU work on worker) → ~55fps

### Option D: Ship v20 (descriptor staging) as best-effort
- 60fps, crashes every 13-20 min at main menu
- Significantly better than v2 (3-8 min)
- Playable for most game sessions

### Option E: Fastmem-only (no GpuWorker)
- 46fps, never crashes
- Safe fallback
