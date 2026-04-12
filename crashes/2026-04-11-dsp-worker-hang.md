# 2026-04-11 — DSP HLE worker thread caused guest hang

**Status**: Reverted before committing. Tried, hung, rolled back. Documented so the next attempt doesn't repeat the mistake.

## What I tried

After a simpleperf profile of the slow spot showed ~3.7% of emu thread time in DSP HLE (`AudioInterp::Linear`, `DspHle::Impl::GenerateCurrentFrame`, `FilterFrame`, `DecodeADPCM`, `Mixers::Tick/DownmixAndMix`, `Source::GenerateFrame`, `Source::ParseConfig`), I added a dedicated `std::jthread` audio worker mirroring the GpuWorker pattern:

- `AudioTickCallback` (still scheduled by `core_timing` on the emu thread) bumped a counter and signaled the worker via condvar
- Worker called `Tick()` (which runs `GenerateCurrentFrame()` + `OutputFrame()`) on its own thread
- Audio interrupts fired by the worker were deferred via `pending_interrupts` queue and drained on the emu thread from `System::RunLoop`, same as `GPU::DrainPendingInterrupts`
- Added a `virtual void DrainPendingInterrupts() {}` to `DspInterface` so `DspLle` is unaffected and `DspHle` overrides

## What broke

Game booted fine, ran briefly in the level, then **hung within ~30 seconds**. Visible symptoms:

- Screen frozen mid-frame
- Audio looped the same ~5ms buffer 2-3 times then went silent (sink playing the last buffer it had)
- `adb shell pidof` confirmed the process was still alive — no crash, no tombstone

## Why it deadlocked

Per-thread state at the moment of the hang (`/proc/<pid>/task/<tid>/wchan`):

| TID | Name | wchan | State |
|---|---|---|---|
| 22288 | NativeEmulation (emu) | `__arm64_sys_nanosleep` | R (running) — but only at 54.8% CPU |
| 22301 | NativeEmulation (GpuWorker) | `futex_wait` | S |
| 22325 | NativeEmulation (DSP audio worker) | `futex_wait` | S — only 3.2% CPU |
| 22317 | VulkanWorker | `futex_wait` | S |

PerfProbe at the same moment:

```
sysFPS=60.0  gameFPS=0.0  rest=0.16-1.41ms  tmr=7.87-8.47ms  swap=0.18-0.26ms
GpuWorkerProbe vblank=60/s submit=0 dma=0 fill=0 xfer=0 other=0
```

Read carefully: `sysFPS=60` (the emu wall-clock loop is ticking) but `gameFPS=0` and `submit=0` (the guest is not rendering anything). `tmr=8ms` per frame is 95% of the budget — the emu thread is **literally sleeping in `Core::Timing` waiting for the next event** because dynarec has no work to do. Dynarec has no work because the guest is blocked waiting for an interrupt that never arrives.

**Root cause hypothesis (most likely):** the DSP HLE shared memory race.

`AudioCore::HLE::SharedMemory` (the `dsp_memory` member) is mapped into guest address space. The guest's audio service thread reads/writes its fields *via dynarec on the emu thread*. The HLE `Tick()` also reads/writes the same fields. The protocol is double-buffered with `frame_counter`-based handshake — `ReadRegion()` returns whichever of region_0/region_1 has the higher counter (the "completed" one), `WriteRegion()` returns the other.

The original (pre-worker) design was safe because `AudioTickCallback` ran *between* dynarec slices on the emu thread — guest writes and HLE writes were strictly sequential. With the worker:

- Guest dynarec on emu thread is writing source_configurations / adpcm_coefficients to region X
- Worker thread runs `Tick()`, sees region X is the "completed" region (higher counter), reads from it, **races the in-progress guest write**
- Worker also writes `source_statuses` / `dsp_status` / `final_samples` to whichever region the guest will read next, possibly while the guest is reading it

The guest sees torn data, ends up in a state where it's waiting for a `Pipe::Audio` interrupt that the HLE never decides to fire (because its internal state is now inconsistent), and the dynarec scheduler parks the guest thread on a `Kernel::Event`. With nothing else to run, the emu thread ends up sleeping in `Core::Timing::Advance` waiting for the next core_timing event — which is the audio tick — which signals the worker — which generates another (still-broken) frame. Loop forever.

The interrupt deferral path itself was fine (it's the same well-tested pattern as the GpuWorker). The hang is the dsp_memory data race, not the interrupt routing.

## What I learned

Don't move work off the emu thread when the work touches memory that **is also dynarec'd by the guest on the emu thread**, unless you add an explicit synchronization barrier between the two writers. The GpuWorker gets away with this because:

1. PICA registers (`pica.regs.internal.reg_array`) are touched by HLE GSP service handlers, which we routed through the worker so the worker is the only writer.
2. Guest MMIO writes to `0x1EF00000` (GPU::WriteReg) flush the worker first.
3. The rasterizer cache is touched by the worker exclusively after we added `cache_mutex`.
4. The guest does **not** dynarec PICA registers — it goes through SVCs that we control.

DSP HLE is different: `dsp_memory` is **directly mapped** into the guest's address space (see `Memory::MemorySystem::GetDspMemory(0)` and how it's exposed via the guest's vm_manager). The guest reads and writes it through normal load/store instructions that go through dynarec, not through any service. We can't intercept those accesses without rewriting the memory subsystem.

## Future approaches (if we ever want this win)

Three viable paths for the ~3.7% emu-thread savings, in increasing order of work:

### Option A: Snapshot-and-restore
- Inside `AudioTickCallback` on the emu thread: copy the `read` region into a local buffer (small, ~kilobytes)
- Hand the snapshot to the worker
- Worker generates output into a local `write` buffer
- On the NEXT `AudioTickCallback`, the emu thread copies the worker's local buffer back into `dsp_memory.write_region`
- Total emu-thread work: two `memcpy`s. Worker does the heavy compute on its own snapshot.
- Catch: the snapshot/restore must take the `frame_counter` swap into account so we never copy stale or in-progress data.

### Option B: Lock dsp_memory at the dynarmic memory-callback layer
- Add a `std::shared_mutex dsp_memory_mutex`
- Dynarec memory callbacks (in `arm_dynarmic.cpp` `MemoryRead8/16/32/Write8/16/32`) detect addresses inside the dsp_memory range and take a shared lock
- Worker takes an exclusive lock around `Tick()`
- Pros: guarantees correctness
- Cons: every guest memory access in the audio range pays a lock cost, even though the address range is small. Probably negligible *because* the range is small, but still pollutes the dynarec hot path.

### Option C: Move only `OutputFrame` to a worker
- Keep `GenerateCurrentFrame()` on the emu thread (unchanged)
- Move only the `parent.OutputFrame(...)` call to a worker
- `OutputFrame` writes into the audio sink's ring buffer + drives the time stretcher, which is most of the actual cost in the inner loop because the stretcher does FIR convolution
- Worth profiling first to confirm OutputFrame is actually a chunk of the cost. Looking at the simpleperf output, `Mixers::Tick` (~0.18%) and `Source::GenerateFrame` (~0.17%) are tiny — the bulk of the cost is `AudioInterp::Linear` (~1.11%), `DecodeADPCM` (~0.47%), `FilterFrame` (~0.48%) which all live INSIDE `GenerateCurrentFrame`. So Option C alone wouldn't recover the full 3.7%.

**Recommendation if anyone retries**: start with Option A (snapshot-and-restore). It's the cleanest separation, doesn't pollute dynarec memory paths, and directly works around the race rather than papering over it.

## Files

The reverted implementation lived in:
- `src/audio_core/dsp_interface.h` (added `virtual void DrainPendingInterrupts() {}`)
- `src/audio_core/hle/hle.h` (added `DrainPendingInterrupts() override`)
- `src/audio_core/hle/hle.cpp` (worker thread, condvar, deferred interrupts)
- `src/core/core.cpp` (called `dsp_core->DrainPendingInterrupts()` in `RunLoop`)

All four files were `git checkout`-ed back to their previous state. The full diff is gone but the design is described above in enough detail to recreate it cleanly with one of the three race fixes attached.
