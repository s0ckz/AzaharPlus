// Copyright Citra Emulator Project / Azahar Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#include <chrono>
#include <thread>
#include <fmt/format.h>
#include "common/archives.h"
#include "common/hacks/hack_manager.h"
#include "common/microprofile.h"
#include "common/settings.h"
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

namespace {
// Diagnostic counters for the GPU transfer paths (texture copies, display transfers,
// memory fills). For each path we track how many went through the rasterizer-accelerated
// fast path vs fell back to the software blitter, plus cumulative wall-time in each.
// The software fallback is catastrophically more expensive than the accelerated path
// because it forces a GPU->CPU readback of the source region, a memcpy, and an invalidate
// of any cached surface overlapping the destination — the PerfProbe 'gpu' bucket in
// SMB3DL is dominated by this work, so knowing the accelerated/software ratio is the
// single most useful data point for deciding whether to chase the rasterizer cache.
//
// No locking: these are updated inside GPU::Execute which already runs on the GSP
// service thread under the same implicit serialization that perf_stats relies on. The
// log output happens from VBlankCallback on the same thread, so the reset is race-free.
struct TexXferCounters {
    // Accelerated-path counts and cumulative ns.
    std::uint64_t accel_tc = 0;
    std::uint64_t accel_tc_ns = 0;
    std::uint64_t accel_dt = 0;
    std::uint64_t accel_dt_ns = 0;
    std::uint64_t accel_mf = 0;
    std::uint64_t accel_mf_ns = 0;
    // Software-fallback counts and cumulative ns (INCLUDES the time wasted in the
    // failed accelerate attempt before bailing out — that's the whole point, the
    // bail-out itself eats time via GetTexCopySurface / GetSurfaceSubRect lookups).
    std::uint64_t sw_tc = 0;
    std::uint64_t sw_tc_ns = 0;
    std::uint64_t sw_dt = 0;
    std::uint64_t sw_dt_ns = 0;
    std::uint64_t sw_mf = 0;
    std::uint64_t sw_mf_ns = 0;
};
TexXferCounters g_tex_xfer;

// Sub-attribution of the GPU::Execute wall time into the three things it actually
// does: SubmitCmdList (PICA command-buffer parsing + WriteInternalReg switch +
// draws), RequestDma (RasterizerFlushVirtualRegion + Memory::CopyBlock), and
// "other" (fill/transfer/cache-flush, which are already accounted for above but
// still incur switch dispatch + interrupt signaling). TexXferProbe confirmed that
// 96% of the 'gpu' PerfProbe bucket in SMB3DL is NOT in fills/transfers — this
// probe pins the remaining work to cmdlist vs dma so we know which one to attack.
struct GpuExecCounters {
    std::uint64_t cmdlist_n = 0;
    std::uint64_t cmdlist_ns = 0;
    // cmdlist calls where skip_draws was true at entry (so ProcessCmdList was
    // short-circuited via ignore_list=true). Counted separately so we can see
    // the savings from the skipped-frame cmdlist bypass in real time.
    std::uint64_t cmdlist_bypass_n = 0;
    std::uint64_t cmdlist_bypass_ns = 0;
    std::uint64_t dma_n = 0;
    std::uint64_t dma_ns = 0;
    std::uint64_t other_n = 0;
    std::uint64_t other_ns = 0;
    // Diagnostic: count SetSkipDraws calls (true vs false) to confirm whether
    // VBlankCallback is actually calling it during the sample window. Compare
    // against cmdlist_bypass_n to debug why the shallow path isn't firing.
    std::uint64_t set_skip_true = 0;
    std::uint64_t set_skip_false = 0;
    // Identity check: addresses of impl->pica seen by VBlankCallback (writer)
    // and by GPU::Execute (reader). If these differ, there are two PicaCore
    // instances and SetSkipDraws/IsSkippingDraws are touching different memory.
    // If they match, the bug is elsewhere (compiler, layout, aliasing).
    const void* writer_pica_addr = nullptr;
    const void* reader_pica_addr = nullptr;
    // ALSO log the address of the skip_draws field itself (not the parent
    // PicaCore object) — if the parent address matches but the field address
    // differs, there's an ODR / layout discrepancy that the parent-address
    // probe can't catch.
    const void* writer_skip_addr = nullptr;
    const void* reader_skip_addr = nullptr;
    // Read-back immediately after SetSkipDraws — if these don't agree with
    // set_skip_true / set_skip_false, the setter itself is broken or the bool
    // is being clobbered between successive instructions.
    std::uint64_t readback_match = 0;
    std::uint64_t readback_mismatch = 0;
    // Thread IDs of writer (VBlankCallback) and reader (Execute) call sites.
    // Hashed via std::hash<std::thread::id>{}() so we can compare in the log.
    // If these differ, GSP::TriggerCmdReqQueue runs on a separate thread from
    // VBlankCallback and the bool needs proper memory ordering — relaxed
    // atomic isn't enough if there's no synchronization edge between them.
    std::size_t writer_tid_hash = 0;
    std::size_t reader_tid_hash = 0;
};
GpuExecCounters g_gpu_exec;

// Small helper so the call-sites stay legible.
inline std::uint64_t NsSince(std::chrono::steady_clock::time_point t0) {
    return static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - t0)
            .count());
}
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
}

GPU::~GPU() = default;

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
    impl->signal_interrupt = handler;
    impl->pica.SetInterruptHandler(handler);
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
    using Service::GSP::CommandId;
    auto& regs = impl->pica.regs;

    // Sub-probe: time each branch of the switch so we can split the PerfProbe 'gpu'
    // bucket into cmdlist / dma / other. The fills and transfers are already
    // separately accounted for in g_tex_xfer, but we ALSO count them as 'other'
    // here so the sum of the sub-probe matches the outer PerfProbe gpu bucket.
    const auto t0 = std::chrono::steady_clock::now();

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
        ++g_gpu_exec.dma_n;
        g_gpu_exec.dma_ns += NsSince(t0);
        break;
    }
    case CommandId::SubmitCmdList: {
        auto& params = command.submit_gpu_cmdlist;
        auto& cmdbuffer = regs.internal.pipeline.command_buffer;

        // Write to the command buffer GPU registers
        cmdbuffer.addr[0].Assign(VirtualToPhysicalAddress(params.address) >> 3);
        cmdbuffer.size[0].Assign(params.size >> 3);
        cmdbuffer.trigger[0] = 1;

        // Track whether the cmdlist is going to be bypassed so the sub-probe can
        // report savings separately. SubmitCmdList() below makes the actual decision.
        const bool bypassing = impl->pica.IsSkippingDraws();
        g_gpu_exec.reader_pica_addr = static_cast<const void*>(&impl->pica);
        g_gpu_exec.reader_skip_addr = impl->pica.DebugSkipDrawsAddr();
        g_gpu_exec.reader_tid_hash = std::hash<std::thread::id>{}(std::this_thread::get_id());

        // Trigger processing of the command list
        SubmitCmdList(0);

        if (bypassing) {
            ++g_gpu_exec.cmdlist_bypass_n;
            g_gpu_exec.cmdlist_bypass_ns += NsSince(t0);
        } else {
            ++g_gpu_exec.cmdlist_n;
            g_gpu_exec.cmdlist_ns += NsSince(t0);
        }
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

    // Attribute everything that wasn't dma/cmdlist to the 'other' sub-bucket. This
    // double-counts the inner fill/transfer timers in g_tex_xfer, but since those
    // are reported separately the sub-probe line still sums cleanly to the PerfProbe
    // gpu total. switch(id) above writes its own g_gpu_exec entry and then break's
    // out of the switch — so by the time we land here, only non-cmdlist / non-dma
    // cases contribute to 'other'.
    switch (command.id) {
    case CommandId::RequestDma:
    case CommandId::SubmitCmdList:
        // Already accounted.
        break;
    default:
        ++g_gpu_exec.other_n;
        g_gpu_exec.other_ns += NsSince(t0);
        break;
    }

    // Notify debugger that a GSP command was processed.
    if (impl->debug_context) {
        impl->debug_context->OnEvent(Pica::DebugContext::Event::GSPCommandProcessed, &command);
    }
}

void GPU::SetBufferSwap(u32 screen_id, const Service::GSP::FrameBufferInfo& info) {
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
    impl->pica.regs_lcd.color_fill_top = fill;
    impl->pica.regs_lcd.color_fill_bottom = fill;
}

u32 GPU::ReadReg(VAddr addr) {
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

        // Handle registers that trigger GPU actions
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
    //
    // Shallow-parse optimization for skipped frames: on frames where
    // IsSkippingDraws()==true, we use ProcessCmdListShallow instead of
    // the full WriteInternalReg path. The shallow parser walks the exact
    // same command stream but:
    //   - updates reg_array[id] directly (preserves register state)
    //   - increments all auto-incrementing offsets (vs/gs program,
    //     swizzle, lighting/fog/proctex LUT indices)
    //   - sets dirty_regs bits (so the next rendered frame knows to
    //     re-upload)
    //   - handles irq_request (game timing) and sub-cmdlist chaining
    //   - skips EVERYTHING else: no DrawArrays, no UpdateProgramCode, no
    //     WriteUniformFloatReg, no LUT data writes, no SubmitImmediate,
    //     no primitive assembler reconfig, no debug callbacks
    //
    // This preserves register state across frames (fixing the blank-screen
    // regression from the previous full-bypass attempt in 7a5fcc40b) while
    // still saving most of the ~300 ms/s wall-time cost that the full
    // parser's side effects eat on skipped frames.
    const PAddr addr = config.GetPhysicalAddress(index);
    const u32 size = config.GetSize(index);

    if (impl->pica.IsSkippingDraws() &&
        right_eye_disabler->ShouldAllowCmdQueueTrigger(addr, size)) {
        impl->pica.ProcessCmdListShallow(addr, size);
    } else {
        impl->pica.ProcessCmdList(addr, size,
                                  !right_eye_disabler->ShouldAllowCmdQueueTrigger(addr, size));
    }
    config.trigger[index] = 0;
}

void GPU::MemoryFill(u32 index, u32 intr_index) {
    // Check if a memory fill was triggered.
    auto& config = impl->pica.regs.memory_fill_config[index];
    if (!config.trigger) {
        return;
    }

    // Perform memory fill, unless the aggressive SkipAllGpu frame-skip mode
    // has elected to drop this frame's GPU work entirely. We still fire the
    // completion interrupt below so the game's GPU command processor keeps
    // advancing normally.
    if (!impl->skip_gpu_transfers) {
        const auto t0 = std::chrono::steady_clock::now();
        if (impl->rasterizer->AccelerateFill(config)) {
            ++g_tex_xfer.accel_mf;
            g_tex_xfer.accel_mf_ns += NsSince(t0);
        } else {
            impl->sw_blitter->MemoryFill(config);
            ++g_tex_xfer.sw_mf;
            g_tex_xfer.sw_mf_ns += NsSince(t0);
        }
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

    // Perform memory transfer.
    //
    // Texture copies are NEVER skipped even in the most aggressive frame-skip
    // mode because they are commonly used for texture-cache / glyph-atlas
    // updates whose results persist across frames. Non-texture-copy display
    // transfers (render target → screen framebuffer copies) are elided on
    // skipped frames in SkipAllGpu mode since we are not going to present
    // that framebuffer anyway.
    if (config.is_texture_copy) {
        const auto t0 = std::chrono::steady_clock::now();
        if (impl->rasterizer->AccelerateTextureCopy(config)) {
            ++g_tex_xfer.accel_tc;
            g_tex_xfer.accel_tc_ns += NsSince(t0);
        } else {
            impl->sw_blitter->TextureCopy(config);
            ++g_tex_xfer.sw_tc;
            g_tex_xfer.sw_tc_ns += NsSince(t0);
        }
    } else if (!impl->skip_gpu_transfers) {
        if (right_eye_disabler->ShouldAllowDisplayTransfer(config.GetPhysicalInputAddress(),
                                                           config.input_height)) {
            const auto t0 = std::chrono::steady_clock::now();
            if (impl->rasterizer->AccelerateDisplayTransfer(config)) {
                ++g_tex_xfer.accel_dt;
                g_tex_xfer.accel_dt_ns += NsSince(t0);
            } else {
                impl->sw_blitter->DisplayTransfer(config);
                ++g_tex_xfer.sw_dt;
                g_tex_xfer.sw_dt_ns += NsSince(t0);
            }
        }
    }

    // Complete transfer.
    config.trigger.Assign(0);
    impl->signal_interrupt(Service::GSP::InterruptId::PPF);
}

void GPU::VBlankCallback(std::uintptr_t user_data, s64 cycles_late) {
    // Frame-skip: emulated timing is always 60Hz (this callback fires every
    // FRAME_TICKS ARM11 cycles regardless) so game logic, audio and physics
    // are untouched. What we manipulate is:
    //
    //   (a) whether the frame that JUST completed is presented to the host,
    //   (b) whether the frame that is ABOUT TO START rendering will run its
    //       PICA draw calls through the host GPU at all.
    //
    // Decision (a) was made at the previous vblank and is stored in
    // `skip_current_present`. Decision (b) is made now for the upcoming frame
    // and is communicated to the PicaCore via SetSkipDraws().

    // (a) Present (or skip-present) the frame that was rendered in the
    // interval before this vblank.
    if (!impl->skip_current_present) {
        impl->renderer->SwapBuffers();
    } else {
        // Keep frame-limiter / input polling / perf counters ticking so the
        // emulator's wall-clock pacing stays on the 60Hz cadence even when we
        // don't draw this frame.
        impl->renderer->EndFrame();
    }

    // Signal to GSP that GPU interrupt has occurred. Always fired — the game
    // relies on this heartbeat to advance its own logic.
    impl->signal_interrupt(Service::GSP::InterruptId::PDC0);
    impl->signal_interrupt(Service::GSP::InterruptId::PDC1);

    // (b) Decide whether the next frame will be skipped and propagate to the
    // PicaCore / GPU transfer paths according to the selected mode.
    impl->vblank_counter++;
    const u32 frame_skip = Settings::values.frame_skip.GetValue();
    const bool next_is_skip =
        (frame_skip > 0) && ((impl->vblank_counter % (frame_skip + 1)) != 0);
    impl->skip_current_present = next_is_skip;

    const auto mode = Settings::values.frame_skip_mode.GetValue();
    // SkipAllGpu is a strict superset of SkipDraws, so both of these enable
    // short-circuiting DrawArrays / DrawImmediate in the PICA core.
    const bool skip_draws = next_is_skip && (mode == Settings::FrameSkipMode::SkipDraws ||
                                              mode == Settings::FrameSkipMode::SkipAllGpu);
    impl->pica.SetSkipDraws(skip_draws);
    // Immediate read-back: if this disagrees with the value we just set, the
    // bool is being clobbered between successive instructions on the SAME thread.
    const bool readback = impl->pica.IsSkippingDraws();
    if (readback == skip_draws) {
        ++g_gpu_exec.readback_match;
    } else {
        ++g_gpu_exec.readback_mismatch;
    }
    g_gpu_exec.writer_pica_addr = static_cast<const void*>(&impl->pica);
    g_gpu_exec.writer_skip_addr = impl->pica.DebugSkipDrawsAddr();
    g_gpu_exec.writer_tid_hash = std::hash<std::thread::id>{}(std::this_thread::get_id());
    if (skip_draws) {
        ++g_gpu_exec.set_skip_true;
    } else {
        ++g_gpu_exec.set_skip_false;
    }
    impl->skip_gpu_transfers =
        next_is_skip && (mode == Settings::FrameSkipMode::SkipAllGpu);

    // Rate-limited diagnostic: dump the effective frame-skip state once a second so we
    // can confirm from logcat whether the setting is actually applied at runtime, and
    // how many frames in each decision-bucket we saw over the last second. PerfProbe's
    // per-frame averages mask this because skipped and unskipped frames both count in
    // system_frames but contribute very different per-bucket costs.
    static u64 skip_log_last_ms = 0;
    static u32 skip_log_skipped = 0;
    static u32 skip_log_total = 0;
    ++skip_log_total;
    if (next_is_skip) {
        ++skip_log_skipped;
    }
    const u64 now_ms = static_cast<u64>(
        std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now().time_since_epoch())
            .count());
    if (now_ms - skip_log_last_ms >= 1000) {
        const char* mode_name = "?";
        switch (mode) {
        case Settings::FrameSkipMode::PresentOnly:
            mode_name = "PresentOnly";
            break;
        case Settings::FrameSkipMode::SkipDraws:
            mode_name = "SkipDraws";
            break;
        case Settings::FrameSkipMode::SkipAllGpu:
            mode_name = "SkipAllGpu";
            break;
        }
        LOG_INFO(HW_GPU,
                 "FrameSkipProbe frame_skip={} mode={} vblanks_last_s={} skipped_last_s={}"
                 " next_is_skip={} skip_draws={} skip_gpu_transfers={}",
                 frame_skip, mode_name, skip_log_total, skip_log_skipped, next_is_skip,
                 skip_draws, impl->skip_gpu_transfers);

        // Dump texture-copy / display-transfer / memory-fill acceleration ratios
        // at the same 1 Hz cadence and reset the counters. The "ms" columns are
        // cumulative wall time over the last ~1 s, in the path's own bucket, so
        // you can compare them directly against the PerfProbe 'gpu' bucket value
        // (which is reported per-frame by PerfProbe, not per-second — divide by
        // the measured vblanks-per-second to correlate).
        LOG_INFO(HW_GPU,
                 "TexXferProbe tc[acc={} sw={} | acc_ms={:.2f} sw_ms={:.2f}] "
                 "dt[acc={} sw={} | acc_ms={:.2f} sw_ms={:.2f}] "
                 "mf[acc={} sw={} | acc_ms={:.2f} sw_ms={:.2f}]",
                 g_tex_xfer.accel_tc, g_tex_xfer.sw_tc,
                 g_tex_xfer.accel_tc_ns / 1.0e6, g_tex_xfer.sw_tc_ns / 1.0e6,
                 g_tex_xfer.accel_dt, g_tex_xfer.sw_dt,
                 g_tex_xfer.accel_dt_ns / 1.0e6, g_tex_xfer.sw_dt_ns / 1.0e6,
                 g_tex_xfer.accel_mf, g_tex_xfer.sw_mf,
                 g_tex_xfer.accel_mf_ns / 1.0e6, g_tex_xfer.sw_mf_ns / 1.0e6);
        g_tex_xfer = TexXferCounters{};

        // Sub-attribution of the PerfProbe 'gpu' bucket across cmdlist/dma/other,
        // plus the bypass bucket so we can see how much wall time the skipped-frame
        // cmdlist optimization is saving. Sum of all four ms columns should roughly
        // equal (PerfProbe gpu per-frame × vblanks_last_s).
        LOG_INFO(HW_GPU,
                 "GpuExecProbe cmdlist[n={} ms={:.2f}] cmdlist_bypass[n={} ms={:.2f}] "
                 "dma[n={} ms={:.2f}] other[n={} ms={:.2f}] set_skip[t={} f={}] "
                 "readback[match={} mismatch={}] "
                 "tids[w={:x} r={:x} match={}] "
                 "pica_addrs[w={} r={} match={}] "
                 "skip_addrs[w={} r={} match={}]",
                 g_gpu_exec.cmdlist_n, g_gpu_exec.cmdlist_ns / 1.0e6,
                 g_gpu_exec.cmdlist_bypass_n, g_gpu_exec.cmdlist_bypass_ns / 1.0e6,
                 g_gpu_exec.dma_n, g_gpu_exec.dma_ns / 1.0e6,
                 g_gpu_exec.other_n, g_gpu_exec.other_ns / 1.0e6,
                 g_gpu_exec.set_skip_true, g_gpu_exec.set_skip_false,
                 g_gpu_exec.readback_match, g_gpu_exec.readback_mismatch,
                 g_gpu_exec.writer_tid_hash, g_gpu_exec.reader_tid_hash,
                 g_gpu_exec.writer_tid_hash == g_gpu_exec.reader_tid_hash,
                 fmt::ptr(g_gpu_exec.writer_pica_addr),
                 fmt::ptr(g_gpu_exec.reader_pica_addr),
                 g_gpu_exec.writer_pica_addr == g_gpu_exec.reader_pica_addr,
                 fmt::ptr(g_gpu_exec.writer_skip_addr),
                 fmt::ptr(g_gpu_exec.reader_skip_addr),
                 g_gpu_exec.writer_skip_addr == g_gpu_exec.reader_skip_addr);
        g_gpu_exec = GpuExecCounters{};

        skip_log_last_ms = now_ms;
        skip_log_total = 0;
        skip_log_skipped = 0;
    }

    // Reschedule recurrent event
    impl->timing.ScheduleEvent(FRAME_TICKS - cycles_late, impl->vblank_event);
}

void GPU::RecreateRenderer(Frontend::EmuWindow& emu_window, Frontend::EmuWindow* secondary_window) {
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
}

void GPU::ReleaseRenderer() {
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
