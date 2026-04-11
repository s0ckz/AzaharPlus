// Copyright 2023 Citra Emulator Project
// Copyright 2024 Azahar Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#include <atomic>
#include <mutex>
#include <vector>

#include "common/archives.h"
#include "common/microprofile.h"
#include "core/core.h"
#include "core/core_timing.h"
#include "core/hle/service/gsp/gsp_gpu.h"
#include "core/hle/service/plgldr/plgldr.h"
#include "video_core/debug_utils/debug_utils.h"
#include "video_core/gpu.h"
#include "video_core/gpu_debugger.h"
#include "video_core/gpu_impl.h"
#include "video_core/gpu_worker.h"
#include "video_core/pica/pica_core.h"
#include "video_core/pica/regs_lcd.h"
#include "video_core/renderer_base.h"
#include "video_core/renderer_software/sw_blitter.h"
#include "video_core/right_eye_disabler.h"
#include "video_core/video_core.h"

namespace VideoCore {
struct GPU::Impl {
    Core::Timing& timing;
    Core::System& system;
    Memory::MemorySystem& memory;
    std::shared_ptr<Pica::DebugContext> debug_context;
    Pica::PicaCore pica;
    GraphicsDebugger gpu_debugger;
    std::unique_ptr<RendererBase> renderer;
    RasterizerInterface* rasterizer;
    std::unique_ptr<SwRenderer::SwBlitter> sw_blitter;
    Core::TimingEventType* vblank_event;

    // The "public" signal_interrupt stored by PicaCore / rasterizer /
    // GPU::Execute paths. Installed by GPU::SetInterruptHandler as a thin
    // wrapper over real_signal_interrupt: when invoked on the GpuWorker
    // thread it pushes the interrupt id into pending_interrupts for the
    // emu thread to deliver later; on the emu thread it forwards to the
    // real handler synchronously. This is the fix for the SIGTRAP in
    // ThreadManager::SwitchContext — Kernel::Event::Signal mutates
    // Kernel::Thread state and must run on the emu thread.
    Service::GSP::InterruptHandler signal_interrupt;
    Service::GSP::InterruptHandler real_signal_interrupt;

    // Dedicated GPU worker thread. Owns all mutation of PICA register
    // state, the rasterizer cache, and the Vulkan scheduler's Record()
    // path. Started from GPU::GPU() after the Impl is constructed so
    // gpu_owner is valid before the first Push.
    GpuWorker worker;

    // Interrupt ids the worker has deferred since the last drain. The
    // emu thread drains this list via GPU::DrainPendingInterrupts() at
    // the top of System::RunLoop (every dynarec slice, thousands of
    // times per second) so guest interrupt latency stays sub-ms.
    std::mutex pending_interrupts_mutex;
    std::vector<Service::GSP::InterruptId> pending_interrupts;

    // Diagnostic counter for guest MMIO writes to the 0x1EF00000 PICA
    // register window (GPU::WriteReg path). SMB3DL's HOME menu produces
    // ~68/s; actual gameplay is ~0. Logged 1 Hz from VBlankOnWorker.
    std::atomic<std::uint64_t> mmio_writereg_count{0};

    explicit Impl(Core::System& system, Frontend::EmuWindow& emu_window,
                  Frontend::EmuWindow* secondary_window)
        : timing{system.CoreTiming()}, system{system}, memory{system.Memory()},
          debug_context{Pica::g_debug_context}, pica{memory, debug_context},
          renderer{VideoCore::CreateRenderer(emu_window, secondary_window, pica, system)},
          rasterizer{renderer->Rasterizer()},
          sw_blitter{std::make_unique<SwRenderer::SwBlitter>(memory, rasterizer)} {}

    // Destructor must stop the worker BEFORE other members get destructed
    // so the worker thread doesn't touch dangling state. NOTE: the
    // OWNING GPU is responsible for also calling worker.Stop() BEFORE
    // ~Impl runs, because libc++'s std::unique_ptr::reset sets the
    // stored pointer to null *before* running the deleter — the gap
    // between that null store and Impl::~Impl's worker.Stop() is enough
    // for an in-flight worker message to crash on `this->impl`.
    ~Impl() {
        worker.Stop();
    }
};
} // namespace VideoCore
