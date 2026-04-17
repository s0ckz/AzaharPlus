// Copyright Citra Emulator Project / Azahar Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#include "video_core/rasterizer_accelerated.h"
#include "video_core/rasterizer_cache/surface_params.h"
#include "video_core/renderer_vulkan/vk_descriptor_update_queue.h"
#include "video_core/renderer_vulkan/vk_pipeline_cache.h"
#include "video_core/renderer_vulkan/vk_render_manager.h"
#include "video_core/renderer_vulkan/vk_stream_buffer.h"
#include "video_core/renderer_vulkan/vk_texture_runtime.h"

namespace Frontend {
class EmuWindow;
}

namespace VideoCore {
class CustomTexManager;
class RendererBase;
} // namespace VideoCore

namespace Pica {
struct DisplayTransferConfig;
struct MemoryFillConfig;
struct FramebufferConfig;
} // namespace Pica

namespace Vulkan {

struct ScreenInfo;

class Instance;
class Scheduler;
class RenderManager;

class RasterizerVulkan : public VideoCore::RasterizerAccelerated {
public:
    explicit RasterizerVulkan(Memory::MemorySystem& memory, Pica::PicaCore& pica,
                              VideoCore::CustomTexManager& custom_tex_manager,
                              VideoCore::RendererBase& renderer, Frontend::EmuWindow& emu_window,
                              const Instance& instance, Scheduler& scheduler,
                              RenderManager& renderpass_cache, DescriptorUpdateQueue& update_queue,
                              u32 image_count);
    ~RasterizerVulkan() override;

    void TickFrame();
    void LoadDefaultDiskResources(const std::atomic_bool& stop_loading,
                                  const VideoCore::DiskResourceLoadCallback& callback) override;

    void DrawTriangles() override;
    void FlushAll() override;
    void FlushRegion(PAddr addr, u32 size) override;
    void InvalidateRegion(PAddr addr, u32 size) override;
    void FlushAndInvalidateRegion(PAddr addr, u32 size) override;
    void ClearAll(bool flush) override;
    bool AccelerateDisplayTransfer(const Pica::DisplayTransferConfig& config) override;
    bool AccelerateTextureCopy(const Pica::DisplayTransferConfig& config) override;
    bool TryDeferredTextureCopy(const Pica::DisplayTransferConfig& config) override;
    bool AccelerateFill(const Pica::MemoryFillConfig& config) override;
    bool AccelerateDisplay(const Pica::FramebufferConfig& config, PAddr framebuffer_addr,
                           u32 pixel_stride, ScreenInfo& screen_info);
    bool AccelerateDrawBatch(bool is_indexed) override;

    /// Switches the disk resources to the specified title
    void SwitchDiskResources(u64 title_id) override;

private:
    /// Syncs pipeline state from PICA registers
    void SyncDrawState();

    /// Syncs and uploads the lighting, fog and proctex LUTs
    void SyncAndUploadLUTs();
    void SyncAndUploadLUTsLF();

    /// Syncs all enabled PICA texture units
    void SyncTextureUnits(const Framebuffer* framebuffer);

    /// Syncs all utility textures in the fragment shader.
    void SyncUtilityTextures(const Framebuffer* framebuffer);

    /// Binds the PICA shadow cube required for shadow mapping
    void BindShadowCube(const Pica::TexturingRegs::FullTextureConfig& texture,
                        vk::DescriptorSet texture_set);

    /// Binds a texture cube to texture unit 0
    void BindTextureCube(const Pica::TexturingRegs::FullTextureConfig& texture,
                         vk::DescriptorSet texture_set);

    /// Upload the uniform blocks to the uniform buffer object
    void UploadUniforms(bool accelerate_draw);

    /// Generic draw function for DrawTriangles and AccelerateDrawBatch
    bool Draw(bool accelerate, bool is_indexed);

    /// Internal implementation for AccelerateDrawBatch
    bool AccelerateDrawBatchInternal(bool is_indexed);

    /// Setup index array for AccelerateDrawBatch
    void SetupIndexArray();

    /// Setup vertex array for AccelerateDrawBatch
    void SetupVertexArray();

    /// Setup the fixed attribute emulation in vulkan
    void SetupFixedAttribs();

    /// Setup vertex shader for AccelerateDrawBatch
    bool SetupVertexShader();

    /// Setup geometry shader for AccelerateDrawBatch
    bool SetupGeometryShader();

    /// Creates the vertex layout struct used for software shader pipelines
    void MakeSoftwareVertexLayout();

private:
    const Instance& instance;
    Scheduler& scheduler;
    RenderManager& renderpass_cache;
    DescriptorUpdateQueue& update_queue;
    PipelineCache pipeline_cache;
    TextureRuntime runtime;
    RasterizerCache res_cache;

    VertexLayout software_layout;
    std::array<u32, 16> binding_offsets{};
    std::array<bool, 16> enable_attributes{};
    std::array<vk::Buffer, 16> vertex_buffers;
    VertexArrayInfo vertex_info;
    PipelineInfo pipeline_info{};

    StreamBuffer stream_buffer;     ///< Vertex+Index buffer
    StreamBuffer uniform_buffer;    ///< Uniform buffer
    StreamBuffer texture_buffer;    ///< Texture buffer
    StreamBuffer texture_lf_buffer; ///< Texture Light-Fog buffer
    vk::UniqueBufferView texture_lf_view;
    vk::UniqueBufferView texture_rg_view;
    vk::UniqueBufferView texture_rgba_view;
    vk::DeviceSize uniform_buffer_alignment;
    u32 uniform_size_aligned_vs_pica;
    u32 uniform_size_aligned_vs;
    u32 uniform_size_aligned_fs;
    bool async_shaders{false};

    // Draw-state dirty latch (optimization A). False until SyncDrawState has
    // populated pipeline_info once; thereafter a clean rasterizer+framebuffer
    // dirty_regs state lets us skip the rebuild.
    bool draw_state_valid{false};

    // Texture-unit cache (optimization B). Stores the (view, sampler) we wrote
    // to the last Texture descriptor set, plus the scheduler tick when we did.
    // On a cache hit within the same tick we reuse the previously-committed
    // descriptor set (kept alive because its slot is still pinned to the
    // current submission's tick by the resource pool).
    std::array<vk::ImageView, 3> cached_tex_views{};
    std::array<vk::Sampler, 3> cached_tex_samplers{};
    u64 tex_cache_tick{0};

    // Deferred software-TextureCopy queue (Fix B). Each entry records an
    // in-flight readback whose memcpy was postponed; drained at TickFrame or
    // when something touches the destination region.
    struct DeferredSwTc {
        u64 fence_tick;              // scheduler tick to wait on
        u8* staging_mapped;          // host-visible buffer with downloaded data
        u32 staging_size;
        VideoCore::SurfaceParams src_params; // for EncodeTexture re-swizzle
        PAddr flush_start;           // inside src_params surface; EncodeTexture start
        PAddr flush_end;             // EncodeTexture end
        bool needs_conversion;
        PAddr dst_addr;              // memcpy destination
        u32 dst_size;                // contiguous copy size (bytes)
    };
    std::vector<DeferredSwTc> pending_sw_tc;

    // Drain all entries whose [dst_addr, dst_addr+dst_size) overlaps [addr, addr+size).
    // Called from FlushRegion/InvalidateRegion/FlushAndInvalidateRegion before they
    // consult the rasterizer cache so those queries see correct CPU memory.
    void DrainDeferredSwTcOverlapping(PAddr addr, u32 size);
    // Drain everything. Called from TickFrame + ClearAll.
    void DrainAllDeferredSwTc();
    // Worker: perform one drain entry (wait, memcpy, invalidate).
    void PerformDeferredSwTc(DeferredSwTc& entry);

    // Fix A: GPU-to-GPU shifted-tile byte copy. When the source is a cached
    // tiled VkImage that fully contains the byte range, we can do
    // vkCmdCopyImage from source to a newly-created destination VkImage with
    // the same tile layout — no CPU round-trip, no fence Wait. Returns true
    // if handled; caller should try the async CPU memcpy path as a fallback.
    bool TryGpuToGpuShiftedCopy(const Pica::DisplayTransferConfig& config);

    // Fix B: async CPU readback + deferred memcpy. Pulled out from the public
    // TryDeferredTextureCopy so Fix A can be tried first.
    bool TryAsyncMemcpyCopy(const Pica::DisplayTransferConfig& config);
};

} // namespace Vulkan
