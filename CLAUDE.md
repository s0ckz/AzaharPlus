# GPU Multi-Threading Fix — Development Guide

## Current Task

Fix the `fault addr 0x60` crash in the Mali G52 Vulkan driver caused by
concurrent Vulkan API calls from multiple threads. Target: 60fps AND stable
(1 hour at MK7 main menu without crash).

## Branch

`deploy/option-c-targeted` — based on v2 (fastmem + GpuWorker)

## Key Documentation Files

| File | Purpose |
|---|---|
| `GPU_THREADING_APPROACHES.md` | Full history of all attempts (v1-v12), root cause analysis, what worked and what didn't. **UPDATE THIS** after every attempt. |
| `TEST_AUTOMATION.md` | How to launch MK7, automate testing, FPS baseline, success criteria. **UPDATE FPS** after each build. |
| `CLAUDE.md` | This file. Development workflow and architecture. |

## Fix Workflow

1. Make code changes
2. Build: `bash docker/android-build/build-android.sh`
3. Deploy + launch MK7 (see TEST_AUTOMATION.md for one-liner)
4. Monitor for crash (30s intervals, check `signal 11` in logcat)
5. **Success = 1 hour at MK7 main menu without crash**
6. If crash: check logs (`VkCrashDump`, `NON-WORKER`, `VkWorkerProbe`), update docs, iterate
7. If fix drops FPS below 50: unacceptable, try different approach

## Architecture

```
Emu thread:
  ├── dynarmic JIT
  ├── HLE kernel/services
  ├── core_timing (VBlank)
  ├── memory callbacks → GPU::FlushRegion (priority channel to GpuWorker)
  └── memory callbacks → rasterizer->InvalidateRegion (direct, no Vulkan)

GpuWorker thread:
  ├── PICA cmdlist processing → rasterizer → Record()
  ├── on_dispatch → vkUpdateDescriptorSets  ← CRASH SOURCE
  ├── Handle::Create proxy → Record+Event → VulkanWorker creates
  ├── VBlank/SwapBuffers (routed from emu thread)
  └── Priority FlushRegion handler (between cmdlists)

VulkanWorker thread:
  └── Executes command chunks (vkCmd* calls)  ← CRASH VICTIM
```

## Root Cause (confirmed)

GpuWorker calls `vkUpdateDescriptorSets` (via `on_dispatch`) WHILE
VulkanWorker executes `vkCmd*`. Both threads inside Mali driver
simultaneously. Mali G52 driver is not thread-safe for this.

In the stable main branch (no GpuWorker), recording and execution are
sequential — never concurrent inside Mali.

## What's Been Tried (summary)

See `GPU_THREADING_APPROACHES.md` for full details.

- **Proxying all Vulkan creation** → didn't fix (crash isn't from creation)
- **Descriptor staging** → fixed the crash BUT caused init crash from Fill sentencing underflow (now fixed)
- **Inline execution** → no crash but killed performance (no pipeline parallelism)
- **VBlank routing through worker** → works but deadlocks with WaitWorker
- **Priority FlushRegion** → removes emu thread from Mali, but GpuWorker+VulkanWorker still concurrent
- **Lightweight Event proxy** → avoids WaitWorker deadlock for Handle::Create

## Next Step

Move `vkUpdateDescriptorSets` from GpuWorker (`on_dispatch`) to
VulkanWorker (`pre_execute`) via descriptor staging:
- `on_dispatch`: SwapToStaging (CPU memcpy, no Vulkan call)
- `pre_execute`: FlushStaging (vkUpdateDescriptorSets on VulkanWorker)

This eliminates the last concurrent Mali access. Previously failed due to
Fill sentencing underflow (unsigned integer wrap when `frame_tick < skip`)
— that bug is now guarded.

## Building

```bash
bash docker/android-build/build-android.sh
```

## Testing (see TEST_AUTOMATION.md)

```bash
# One-liner: launch MK7 and monitor
DEVICE=192.168.1.51:<PORT>
# See TEST_AUTOMATION.md for full command
```

## Diagnostic Logs

```bash
# Crash info
adb shell "logcat -d | grep -iE 'VkCrashDump|signal 11'"
# NON-WORKER descriptor flushes (should be 0)
adb shell "logcat -d | grep 'NON-WORKER' | wc -l"
# Worker stats
adb shell "logcat -d | grep 'VkWorkerProbe' | tail -5"
# FPS
adb shell "logcat -d | grep 'speed=' | tail -5"
```
