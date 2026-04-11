// Copyright Citra Emulator Project / Azahar Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#include <chrono>
#include <cstdint>
#include "common/archives.h"
#include "common/hacks/hack_manager.h"
#include "common/logging/log.h"
#include "common/microprofile.h"
#include "core/core.h"
#include "core/core_timing.h"
#include "core/hle/service/gsp/gsp_gpu.h"
#include "core/hle/service/plgldr/plgldr.h"
#include "core/loader/loader.h"
#include "video_core/debug_utils/debug_utils.h"
#include "video_core/gpu.h"
#include "video_core/gpu_debugger.h"
#include "video_core/gpu_impl.h"
#include "video_core/pica/pica_core.h"
#include "video_core/pica/regs_lcd.h"
#include "video_core/renderer_base.h"
#include "video_core/renderer_software/sw_blitter.h"
#include "video_core/right_eye_disabler.h"
#include "video_core/video_core.h"

namespace VideoCore {

constexpr VAddr VADDR_LCD = 0x1ED02000;
constexpr VAddr VADDR_GPU = 0x1EF00000;

MICROPROFILE_DEFINE(GPU_DisplayTransfer, "GPU", "DisplayTransfer", MP_RGB(100, 100, 255));
MICROPROFILE_DEFINE(GPU_CmdlistProcessing, "GPU", "Cmdlist Processing", MP_RGB(100, 255, 100));

// Worker-side diagnostic counters. Incremented by the *OnWorker methods
// and logged from VBlankOnWorker at 1 Hz. File-scope (not in anonymous
// namespace) so they're reachable from every function in this TU
// without ordering constraints.
namespace {
std::atomic<std::uint64_t> g_worker_exec_submit{0};
std::atomic<std::uint64_t> g_worker_exec_dma{0};
std::atomic<std::uint64_t> g_worker_exec_fill{0};
std::atomic<std::uint64_t> g_worker_exec_xfer{0};
std::atomic<std::uint64_t> g_worker_exec_other{0};
std::atomic<std::uint64_t> g_worker_setbufferswap{0};
std::atomic<std::uint64_t> g_worker_vblank{0};
} // namespace

GPU::GPU(Core::System& system, Frontend::EmuWindow& emu_window,
         Frontend::EmuWindow* secondary_window)
    : right_eye_disabler{std::make_unique<RightEyeDisabler>(*this)},
      impl{std::make_unique<Impl>(system, emu_window, secondary_window)} {
    impl->vblank_event = impl->timing.RegisterEvent(
        "GPU::VBlankCallback",
        [this](uintptr_t user_data, s64 cycles_late) { VBlankCallback(user_data, cycles_late); });
    impl->timing.ScheduleEvent(FRAME_TICKS, impl->vblank_event);

    // Bind the rasterizer to the PICA GPU
    impl->pica.BindRasterizer(impl->rasterizer);

    // Start the GPU worker thread. From this point on, ALL mutation of
    // PICA state, the rasterizer cache, and the Vulkan scheduler must
    // happen on the worker thread via messages pushed through Impl::worker.
    impl->worker.Start(this);
}

GPU::~GPU() {
    // Stop the worker BEFORE the impl unique_ptr destructor runs. libc++'s
    // std::unique_ptr::reset sets its stored pointer to nullptr *before*
    // invoking the deleter, so there is a window between the null store
    // and Impl::~Impl's worker.Stop() where the worker thread reads
    // this->impl as null and SIGSEGVs at the offset of whatever Impl
    // member it was about to touch.
    if (impl) {
        impl->worker.Stop();
        if (impl->vblank_event) {
            impl->timing.UnscheduleEvent(impl->vblank_event, 0);
        }
    }
}

PAddr GPU::VirtualToPhysicalAddress(VAddr addr) {
    if (addr == 0) {
        return 0;
    }

    if (addr >= Memory::VRAM_VADDR && addr <= Memory::VRAM_VADDR_END) {
        return addr - Memory::VRAM_VADDR + Memory::VRAM_PADDR;
    }
    if (addr >= Memory::LINEAR_HEAP_VADDR && addr <= Memory::LINEAR_HEAP_VADDR_END) {
        return addr - Memory::LINEAR_HEAP_VADDR + Memory::FCRAM_PADDR;
    }
    if (addr >= Memory::NEW_LINEAR_HEAP_VADDR && addr <= Memory::NEW_LINEAR_HEAP_VADDR_END) {
        return addr - Memory::NEW_LINEAR_HEAP_VADDR + Memory::FCRAM_PADDR;
    }
    PAddr plg_fb_addr;
    if (addr >= Memory::PLUGIN_3GX_FB_VADDR && addr <= Memory::PLUGIN_3GX_FB_VADDR_END &&
        (plg_fb_addr = impl->system.Memory().Plugin3GXFramebufferAddress())) {
        return addr - Memory::PLUGIN_3GX_FB_VADDR + plg_fb_addr;
    }

    LOG_ERROR(HW_Memory, "Unknown virtual address @ 0x{:08X}", addr);
    return addr;
}

void GPU::SetInterruptHandler(Service::GSP::InterruptHandler handler) {
    // Store the real handler, then install a thin wrapper that routes
    // interrupt delivery away from the worker thread. Kernel::Event::
    // Signal (inside the GSP handler) mutates Kernel::Thread state and
    // races with the emu thread running ThreadManager::SwitchContext
    // → SIGTRAP. The wrapper defers worker-thread signals into
    // impl->pending_interrupts; the emu thread drains them from
    // System::RunLoop (every dynarec slice).
    impl->real_signal_interrupt = handler;
    impl->signal_interrupt = [this](Service::GSP::InterruptId id) {
        if (GpuWorker::IsOnWorkerThread()) {
            std::scoped_lock lock{impl->pending_interrupts_mutex};
            impl->pending_interrupts.push_back(id);
            return;
        }
        if (impl->real_signal_interrupt) {
            impl->real_signal_interrupt(id);
        }
    };
    impl->pica.SetInterruptHandler(impl->signal_interrupt);
}

void GPU::DrainPendingInterrupts() {
    // Called from the emu thread only. Swap the pending list under the
    // lock to keep the critical section tiny, then fire each interrupt
    // outside the lock (Kernel::Event::Signal may wake threads and
    // reschedule — non-trivial work that shouldn't hold the mutex).
    std::vector<Service::GSP::InterruptId> drained;
    {
        std::scoped_lock lock{impl->pending_interrupts_mutex};
        if (impl->pending_interrupts.empty()) {
            return;
        }
        drained.swap(impl->pending_interrupts);
    }
    if (!impl->real_signal_interrupt) {
        return;
    }

    // 1 Hz diagnostic: drain counts per-interrupt-type. Helps diagnose
    // whether the worker is firing interrupts (counts > 0 but game
    // stalls = guest not using those interrupts) or not (counts = 0 =
    // worker stuck or not firing).
    static std::atomic<std::uint64_t> drained_pdc0{0};
    static std::atomic<std::uint64_t> drained_pdc1{0};
    static std::atomic<std::uint64_t> drained_p3d{0};
    static std::atomic<std::uint64_t> drained_psc{0};
    static std::atomic<std::uint64_t> drained_ppf{0};
    static std::atomic<std::uint64_t> drained_dma{0};
    for (const auto id : drained) {
        switch (id) {
        case Service::GSP::InterruptId::PDC0: drained_pdc0++; break;
        case Service::GSP::InterruptId::PDC1: drained_pdc1++; break;
        case Service::GSP::InterruptId::P3D:  drained_p3d++;  break;
        case Service::GSP::InterruptId::PSC0:
        case Service::GSP::InterruptId::PSC1: drained_psc++;  break;
        case Service::GSP::InterruptId::PPF:  drained_ppf++;  break;
        case Service::GSP::InterruptId::DMA:  drained_dma++;  break;
        default: break;
        }
        impl->real_signal_interrupt(id);
    }
    static auto last_log = std::chrono::steady_clock::now();
    const auto now = std::chrono::steady_clock::now();
    if (std::chrono::duration_cast<std::chrono::milliseconds>(now - last_log).count() >= 1000) {
        last_log = now;
        LOG_INFO(HW_GPU,
                 "GpuIntrProbe pdc0={} pdc1={} p3d={} psc={} ppf={} dma={}",
                 drained_pdc0.exchange(0), drained_pdc1.exchange(0),
                 drained_p3d.exchange(0), drained_psc.exchange(0),
                 drained_ppf.exchange(0), drained_dma.exchange(0));
    }
}

void GPU::FlushRegion(PAddr addr, u32 size) {
    impl->rasterizer->FlushRegion(addr, size);
}

void GPU::InvalidateRegion(PAddr addr, u32 size) {
    impl->rasterizer->InvalidateRegion(addr, size);
}

void GPU::ClearAll(bool flush) {
    impl->rasterizer->ClearAll(flush);
}

void GPU::Execute(const Service::GSP::Command& command) {
    // Producer path on the emu thread: push the command to the worker
    // queue and return immediately. On re-entry from the worker itself
    // (via SignalInterruptForThread → GPU::SetBufferSwap etc.) fall
    // through to the synchronous body to avoid self-deadlock on a full
    // queue.
    if (impl->worker.IsRunning() && !GpuWorker::IsOnWorkerThread()) [[likely]] {
        // Drain any worker-fired interrupts BEFORE the guest thread
        // observes the result of this SVC. The guest thread is running
        // in emu-thread context right now and will only see kernel state
        // updates performed on the emu thread.
        DrainPendingInterrupts();
        impl->worker.Push(GpuCmdExecute{command});
        return;
    }
    ExecuteOnWorker(command);
}

void GPU::ExecuteOnWorker(const Service::GSP::Command& command) {
    using Service::GSP::CommandId;
    auto& regs = impl->pica.regs;

    switch (command.id) {
    case CommandId::SubmitCmdList:
        g_worker_exec_submit.fetch_add(1, std::memory_order_relaxed);
        break;
    case CommandId::RequestDma:
        g_worker_exec_dma.fetch_add(1, std::memory_order_relaxed);
        break;
    case CommandId::MemoryFill:
        g_worker_exec_fill.fetch_add(1, std::memory_order_relaxed);
        break;
    case CommandId::DisplayTransfer:
    case CommandId::TextureCopy:
        g_worker_exec_xfer.fetch_add(1, std::memory_order_relaxed);
        break;
    default:
        g_worker_exec_other.fetch_add(1, std::memory_order_relaxed);
        break;
    }

    switch (command.id) {
    case CommandId::RequestDma: {
        impl->system.Memory().RasterizerFlushVirtualRegion(
            command.dma_request.source_address, command.dma_request.size, Memory::FlushMode::Flush);
        impl->system.Memory().RasterizerFlushVirtualRegion(command.dma_request.dest_address,
                                                           command.dma_request.size,
                                                           Memory::FlushMode::Invalidate);

        // TODO(Subv): These memory accesses should not go through the application's memory mapping.
        // They should go through the GSP module's memory mapping.
        const auto process = impl->system.Kernel().GetCurrentProcess();
        impl->memory.CopyBlock(*process, command.dma_request.dest_address,
                               command.dma_request.source_address, command.dma_request.size);
        impl->signal_interrupt(Service::GSP::InterruptId::DMA);
        break;
    }
    case CommandId::SubmitCmdList: {
        auto& params = command.submit_gpu_cmdlist;
        auto& cmdbuffer = regs.internal.pipeline.command_buffer;

        // Write to the command buffer GPU registers
        cmdbuffer.addr[0].Assign(VirtualToPhysicalAddress(params.address) >> 3);
        cmdbuffer.size[0].Assign(params.size >> 3);
        cmdbuffer.trigger[0] = 1;

        // Trigger processing of the command list
        SubmitCmdList(0);
        break;
    }
    case CommandId::MemoryFill: {
        auto& params = command.memory_fill;
        auto& memfill = regs.memory_fill_config;

        // Write to the memory fill GPU registers.
        // If both buffers are set GSP dispatches PSC0 only.
        const bool has_both_bufs = params.start1 != 0 && params.start2 != 0;
        if (params.start1 != 0) {
            memfill[0].address_start = VirtualToPhysicalAddress(params.start1) >> 3;
            memfill[0].address_end = VirtualToPhysicalAddress(params.end1) >> 3;
            memfill[0].value_32bit = params.value1;
            memfill[0].control = params.control1;
            MemoryFill(0, has_both_bufs ? std::numeric_limits<u32>::max() : 0);
        }
        if (params.start2 != 0) {
            memfill[1].address_start = VirtualToPhysicalAddress(params.start2) >> 3;
            memfill[1].address_end = VirtualToPhysicalAddress(params.end2) >> 3;
            memfill[1].value_32bit = params.value2;
            memfill[1].control = params.control2;
            MemoryFill(1, has_both_bufs ? 0 : 1);
        }
        break;
    }
    case CommandId::DisplayTransfer: {
        auto& params = command.display_transfer;
        auto& display_transfer = regs.display_transfer_config;

        // Write to the transfer engine GPU registers.
        display_transfer.input_address = VirtualToPhysicalAddress(params.in_buffer_address) >> 3;
        display_transfer.output_address = VirtualToPhysicalAddress(params.out_buffer_address) >> 3;
        display_transfer.input_size = params.in_buffer_size;
        display_transfer.output_size = params.out_buffer_size;
        display_transfer.flags = params.flags;
        display_transfer.trigger.Assign(1);

        // Trigger the display transfer.
        MemoryTransfer();
        break;
    }
    case CommandId::TextureCopy: {
        auto& params = command.texture_copy;
        auto& texture_copy = regs.display_transfer_config;

        // Write to the transfer engine GPU registers.
        texture_copy.input_address = VirtualToPhysicalAddress(params.in_buffer_address) >> 3;
        texture_copy.output_address = VirtualToPhysicalAddress(params.out_buffer_address) >> 3;
        texture_copy.texture_copy.size = params.size;
        texture_copy.texture_copy.input_size = params.in_width_gap;
        texture_copy.texture_copy.output_size = params.out_width_gap;
        texture_copy.flags = params.flags;
        texture_copy.trigger.Assign(1);

        // Trigger the texture copy.
        MemoryTransfer();
        break;
    }
    case CommandId::CacheFlush: {
        // Rasterizer flushing handled elsewhere in CPU read/write and other GPU handlers
        // Use command.cache_flush.regions to implement this handler
        break;
    }
    default:
        LOG_ERROR(HW_GPU, "Unknown command {:#08X}", command.id.Value());
    }

    // Notify debugger that a GSP command was processed.
    if (impl->debug_context) {
        impl->debug_context->OnEvent(Pica::DebugContext::Event::GSPCommandProcessed, &command);
    }
}

void GPU::SetBufferSwap(u32 screen_id, const Service::GSP::FrameBufferInfo& info) {
    // Keep SetBufferSwap synchronous on the emu thread. This is on the
    // interrupt-delivery path (SignalInterruptForThread → SetBufferSwap)
    // where ordering with PDC interrupt firing matters. Flush any
    // pending worker draws first so framebuffer_config isn't written
    // out of order with a prior cmdlist.
    if (impl->worker.IsRunning() && !GpuWorker::IsOnWorkerThread()) {
        impl->worker.Flush();
    }
    SetBufferSwapOnWorker(screen_id, info);
}

void GPU::SetBufferSwapOnWorker(u32 screen_id, const Service::GSP::FrameBufferInfo& info) {
    g_worker_setbufferswap.fetch_add(1, std::memory_order_relaxed);

    const PAddr phys_address_left = VirtualToPhysicalAddress(info.address_left);
    const PAddr phys_address_right = VirtualToPhysicalAddress(info.address_right);

    // Update framebuffer properties.
    auto& framebuffer = impl->pica.regs.framebuffer_config[screen_id];
    if (info.active_fb == 0) {
        framebuffer.address_left1 = phys_address_left;
        framebuffer.address_right1 = phys_address_right;
    } else {
        framebuffer.address_left2 = phys_address_left;
        framebuffer.address_right2 = phys_address_right;
    }

    framebuffer.stride = info.stride;
    framebuffer.format = info.format;
    framebuffer.active_fb = info.shown_fb;

    // Notify debugger about the buffer swap.
    if (impl->debug_context) {
        impl->debug_context->OnEvent(Pica::DebugContext::Event::BufferSwapped, nullptr);
    }

    if (screen_id == 0) {
        MicroProfileFlip();
        impl->system.perf_stats->EndGameFrame();
        right_eye_disabler->ReportEndFrame();
    }
}

void GPU::SetColorFill(const Pica::ColorFill& fill) {
    if (impl->worker.IsRunning() && !GpuWorker::IsOnWorkerThread()) {
        impl->worker.Flush();
    }
    SetColorFillOnWorker(fill.raw);
}

void GPU::SetColorFillOnWorker(u32 raw) {
    Pica::ColorFill fill{};
    fill.raw = raw;
    impl->pica.regs_lcd.color_fill_top = fill;
    impl->pica.regs_lcd.color_fill_bottom = fill;
}

u32 GPU::ReadReg(VAddr addr) {
    // Guest MMIO reads of PICA registers MUST observe all prior worker
    // writes for read-after-write correctness. Flush the worker before
    // reading. This is rare for commercial 3DS titles (SMB3DL sees zero
    // MMIO reads during gameplay) so the Flush overhead is acceptable.
    if (impl->worker.IsRunning() && !GpuWorker::IsOnWorkerThread()) {
        impl->worker.Flush();
    }
    switch (addr & 0xFFFFF000) {
    case VADDR_LCD: {
        const u32 offset = addr - VADDR_LCD;
        const u32 index = offset / sizeof(u32);
        ASSERT(addr % sizeof(u32) == 0);
        ASSERT(index < Pica::RegsLcd::NumIds());
        return impl->pica.regs_lcd[index];
    }
    case VADDR_GPU:
    case VADDR_GPU + 0x1000: {
        const u32 offset = addr - VADDR_GPU;
        const u32 index = offset / sizeof(u32);
        ASSERT(addr % sizeof(u32) == 0);
        ASSERT(index < Pica::PicaCore::Regs::NUM_REGS);
        return impl->pica.regs.reg_array[index];
    }
    default:
        UNREACHABLE_MSG("Read from unknown GPU address {:#08X}", addr);
    }
}

void GPU::WriteReg(VAddr addr, u32 data) {
    // Track guest MMIO writes. SMB3DL HOME menu hits ~68/s briefly
    // during LLE applet init; actual gameplay is zero. Logged 1 Hz from
    // VBlankOnWorker.
    impl->mmio_writereg_count.fetch_add(1, std::memory_order_relaxed);

    // MMIO writes MUST maintain read-after-write ordering with subsequent
    // ReadRegs AND with worker register writes. Flush the worker first
    // to quiesce it, then apply the write synchronously on the emu
    // thread. This also serializes the trigger-register side effects
    // (MemoryFill/Transfer/SubmitCmdList) with any worker draws.
    if (impl->worker.IsRunning() && !GpuWorker::IsOnWorkerThread()) {
        impl->worker.Flush();
    }

    switch (addr & 0xFFFFF000) {
    case VADDR_LCD: {
        const u32 offset = addr - VADDR_LCD;
        const u32 index = offset / sizeof(u32);
        ASSERT(addr % sizeof(u32) == 0);
        ASSERT(index < Pica::RegsLcd::NumIds());
        impl->pica.regs_lcd[index] = data;
        break;
    }
    case VADDR_GPU:
    case VADDR_GPU + 0x1000: {
        const u32 offset = addr - VADDR_GPU;
        const u32 index = offset / sizeof(u32);

        ASSERT(addr % sizeof(u32) == 0);
        ASSERT(index < Pica::PicaCore::Regs::NUM_REGS);
        impl->pica.regs.reg_array[index] = data;

        // Handle registers that trigger GPU actions. Running these on
        // the emu thread (after Flush) is safe because the worker has
        // been drained and won't touch the rasterizer concurrently.
        switch (index) {
        case GPU_REG_INDEX(memory_fill_config[0].trigger):
            MemoryFill(0, 0);
            break;
        case GPU_REG_INDEX(memory_fill_config[1].trigger):
            MemoryFill(1, 1);
            break;
        case GPU_REG_INDEX(display_transfer_config.trigger):
            MemoryTransfer();
            break;
        case GPU_REG_INDEX(internal.pipeline.command_buffer.trigger[0]):
            SubmitCmdList(0);
            break;
        case GPU_REG_INDEX(internal.pipeline.command_buffer.trigger[1]):
            SubmitCmdList(1);
            break;
        default:
            break;
        }
        break;
    }
    default:
        UNREACHABLE_MSG("Write to unknown GPU address {:#08X}", addr);
    }
}

VideoCore::RendererBase& GPU::Renderer() {
    return *impl->renderer;
}

Pica::PicaCore& GPU::PicaCore() {
    return impl->pica;
}

const Pica::PicaCore& GPU::PicaCore() const {
    return impl->pica;
}

Pica::DebugContext& GPU::DebugContext() {
    return *Pica::g_debug_context;
}

GraphicsDebugger& GPU::Debugger() {
    return impl->gpu_debugger;
}

void GPU::ApplyPerProgramSettings(u64 program_ID) {
    auto hack = Common::Hacks::hack_manager.GetHack(
        Common::Hacks::HackType::ACCURATE_MULTIPLICATION, program_ID);
    bool use_accurate_mul = Settings::values.shaders_accurate_mul.GetValue();
    if (hack) {
        switch (hack->mode) {
        case Common::Hacks::HackAllowMode::DISALLOW:
            use_accurate_mul = false;
            break;
        case Common::Hacks::HackAllowMode::FORCE:
            use_accurate_mul = true;
            break;
        case Common::Hacks::HackAllowMode::ALLOW:
        default:
            break;
        }
    }
    impl->rasterizer->SetAccurateMul(use_accurate_mul);
}

void GPU::SubmitCmdList(u32 index) {
    // Check if a command list was triggered.
    auto& config = impl->pica.regs.internal.pipeline.command_buffer;
    if (!config.trigger[index]) {
        return;
    }

    MICROPROFILE_SCOPE(GPU_CmdlistProcessing);

    // Forward command list processing to the PICA core.
    const PAddr addr = config.GetPhysicalAddress(index);
    const u32 size = config.GetSize(index);
    impl->pica.ProcessCmdList(addr, size,
                              !right_eye_disabler->ShouldAllowCmdQueueTrigger(addr, size));
    config.trigger[index] = 0;
}

void GPU::MemoryFill(u32 index, u32 intr_index) {
    // Check if a memory fill was triggered.
    auto& config = impl->pica.regs.memory_fill_config[index];
    if (!config.trigger) {
        return;
    }

    // Perform memory fill.
    if (!impl->rasterizer->AccelerateFill(config)) {
        impl->sw_blitter->MemoryFill(config);
    }

    // It seems that it won't signal interrupt if "address_start" is zero.
    // TODO: hwtest this
    if (config.GetStartAddress() != 0) {
        if (intr_index == 0) {
            impl->signal_interrupt(Service::GSP::InterruptId::PSC0);
        } else if (intr_index == 1) {
            impl->signal_interrupt(Service::GSP::InterruptId::PSC1);
        }
    }

    // Reset "trigger" flag and set the "finish" flag
    // This was confirmed to happen on hardware even if "address_start" is zero.
    config.trigger.Assign(0);
    config.finished.Assign(1);
}

void GPU::MemoryTransfer() {
    // Check if a transfer was triggered.
    auto& config = impl->pica.regs.display_transfer_config;
    if (!config.trigger.Value()) {
        return;
    }

    MICROPROFILE_SCOPE(GPU_DisplayTransfer);

    // Notify debugger about the display transfer.
    if (impl->debug_context) {
        impl->debug_context->OnEvent(Pica::DebugContext::Event::IncomingDisplayTransfer, nullptr);
    }

    // Perform memory transfer
    if (config.is_texture_copy) {
        if (!impl->rasterizer->AccelerateTextureCopy(config)) {
            impl->sw_blitter->TextureCopy(config);
        }
    } else {
        if (right_eye_disabler->ShouldAllowDisplayTransfer(config.GetPhysicalInputAddress(),
                                                           config.input_height)) {
            if (!impl->rasterizer->AccelerateDisplayTransfer(config)) {
                impl->sw_blitter->DisplayTransfer(config);
            }
        }
    }

    // Complete transfer.
    config.trigger.Assign(0);
    impl->signal_interrupt(Service::GSP::InterruptId::PPF);
}

void GPU::VBlankCallback(std::uintptr_t user_data, s64 cycles_late) {
    // Core timing event dispatch runs on the emu thread. We keep the
    // VBlank/SwapBuffers path SYNCHRONOUS here — routing it through the
    // worker appeared to break guest boot (the guest advanced through
    // DSP init quickly but then stalled with no GSP calls). The guest's
    // PDC interrupt handler and VBlank timing expect PDC0/PDC1 to fire
    // inline with the core_timing event, not asynchronously. Before
    // calling the synchronous body we flush the worker to establish
    // read-after-write ordering — the guest's dynarec will read GSP
    // shared memory state (framebuffer_config, etc.) that any queued
    // worker draw might still be about to mutate.
    if (impl->worker.IsRunning() && !GpuWorker::IsOnWorkerThread()) {
        impl->worker.Flush();
    }
    VBlankOnWorker(cycles_late);
    impl->timing.ScheduleEvent(FRAME_TICKS - cycles_late, impl->vblank_event);
}

void GPU::VBlankOnWorker(s64 cycles_late) {
    g_worker_vblank.fetch_add(1, std::memory_order_relaxed);

    // Present rendered frame.
    impl->renderer->SwapBuffers();

    // Signal to GSP that GPU interrupt has occurred
    impl->signal_interrupt(Service::GSP::InterruptId::PDC0);
    impl->signal_interrupt(Service::GSP::InterruptId::PDC1);

    // 1 Hz diagnostic sample.
    static auto last_log = std::chrono::steady_clock::now();
    const auto now = std::chrono::steady_clock::now();
    if (std::chrono::duration_cast<std::chrono::milliseconds>(now - last_log).count() >= 1000) {
        last_log = now;
        const auto mmio_count = impl->mmio_writereg_count.exchange(0, std::memory_order_relaxed);
        LOG_INFO(HW_GPU,
                 "GpuWorkerProbe vblank={}/s submit={} dma={} fill={} xfer={} other={} "
                 "setbufswap={} mmio={}",
                 g_worker_vblank.exchange(0),
                 g_worker_exec_submit.exchange(0),
                 g_worker_exec_dma.exchange(0),
                 g_worker_exec_fill.exchange(0),
                 g_worker_exec_xfer.exchange(0),
                 g_worker_exec_other.exchange(0),
                 g_worker_setbufferswap.exchange(0),
                 mmio_count);
    }
}

void GPU::RecreateRenderer(Frontend::EmuWindow& emu_window, Frontend::EmuWindow* secondary_window) {
    // Stop the worker so we can destroy the renderer without racing it.
    impl->worker.Stop();

    // Reset the renderer (this will destroy OpenGL resources)
    impl->renderer.reset();

    // Create a new renderer
    impl->renderer =
        VideoCore::CreateRenderer(emu_window, secondary_window, impl->pica, impl->system);
    impl->rasterizer = impl->renderer->Rasterizer();

    // Rebind the rasterizer to the PICA GPU
    impl->pica.BindRasterizer(impl->rasterizer);

    // Update the sw_blitter with the new rasterizer
    impl->sw_blitter = std::make_unique<SwRenderer::SwBlitter>(impl->memory, impl->rasterizer);

    // Re-apply per-game configuration and reload disk shader cache
    u64 program_id{};
    impl->system.GetAppLoader().ReadProgramId(program_id);
    ApplyPerProgramSettings(program_id);
    if (Settings::values.use_disk_shader_cache) {
        impl->renderer->Rasterizer()->LoadDefaultDiskResources(false, nullptr);
    }

    // Mark ALL GPU registers as dirty so current state gets uploaded to new renderer
    impl->pica.dirty_regs.SetAllDirty();

    // Also mark shader setups as dirty so uniforms get re-uploaded and
    // stale pointers to the old rasterizer's JIT cache are cleared.
    impl->pica.vs_setup.uniforms_dirty = true;
    impl->pica.vs_setup.cached_shader = nullptr;
    impl->pica.gs_setup.uniforms_dirty = true;
    impl->pica.gs_setup.cached_shader = nullptr;

    // Mark all cached LUT/table state in pica as dirty
    impl->pica.lighting.lut_dirty = impl->pica.lighting.LutAllDirty;
    impl->pica.fog.lut_dirty = true;
    impl->pica.proctex.table_dirty = impl->pica.proctex.TableAllDirty;

    // Restart the worker for the new renderer.
    impl->worker.Start(this);
}

void GPU::ReleaseRenderer() {
    impl->worker.Stop();

    // Just reset the renderer to release OpenGL resources
    // Don't null out rasterizer pointer as it will become dangling
    impl->renderer.reset();
    impl->sw_blitter.reset();
    LOG_INFO(HW_GPU, "Renderer released for context destroy");
}

template <class Archive>
void GPU::serialize(Archive& ar, const u32 file_version) {
    ar & impl->pica;
}

SERIALIZE_IMPL(GPU)

} // namespace VideoCore
