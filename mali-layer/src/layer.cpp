// Azahar Mali G52 serialization Vulkan layer.
//
// Sits in the Vulkan loader's dispatch chain between the app and the ICD.
// For every hot entry point the real Mali driver provides, this layer
// acquires a global recursive mutex before forwarding the call — which
// serializes Azahar's emu/GpuWorker/VulkanWorker threads as they enter
// libGLES_mali.so and avoids the TOCTOU races at Mali offsets 0x9e5014
// and 0x9e6730 that produce the "fault addr 0x8" / "fault addr 0x60"
// SIGSEGVs under the threaded renderer.
//
// The layer is a no-op on any device whose VK_PHYSICAL_DEVICE is not
// reported as "Mali-G52 MC1" (or similar Mali-G52 subfamilies). Outside
// that one GPU, all wrappers fall through to the next layer/driver with
// zero lock overhead.
//
// Discovery and activation on Android:
//   - Ship this .so as libVkLayer_azahar_mali_serialize.so inside the
//     APK's jniLibs/arm64-v8a/
//   - Azahar adds the layer name "VK_LAYER_AZAHAR_mali_serialize" to
//     VkInstanceCreateInfo::ppEnabledLayerNames when instantiating the
//     Vulkan renderer on a Mali-G52 device
//
// Layer loader interface implemented:
//   - vkNegotiateLoaderLayerInterfaceVersion (preferred handshake)
//   - vkEnumerateInstanceLayerProperties
//   - vkEnumerateDeviceLayerProperties
//   - vkEnumerateInstanceExtensionProperties
//   - vkEnumerateDeviceExtensionProperties
//   - vkGetInstanceProcAddr / vkGetDeviceProcAddr

#include <cstring>
#include <cstdlib>
#include <cstdint>
#include <unordered_map>
#include <mutex>
#include <pthread.h>

#define VK_USE_PLATFORM_ANDROID_KHR
#define VK_NO_PROTOTYPES
#include <vulkan/vulkan.h>
#include <vulkan/vk_layer.h>

// Vulkan-Headers doesn't always define VK_LAYER_EXPORT. Mirror the canonical
// Khronos definition so our vkGetInstanceProcAddr/etc. are dynamically visible
// regardless of -fvisibility=hidden.
#ifndef VK_LAYER_EXPORT
#define VK_LAYER_EXPORT __attribute__((visibility("default")))
#endif

#include <android/log.h>
#define LOG_TAG "MaliSerializeLayer"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO,  LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

#ifndef LAYER_NAME
#define LAYER_NAME "VK_LAYER_AZAHAR_mali_serialize"
#endif

namespace {

// -------- Global serialization mutex --------
// Recursive so Mali re-entry via its own dispatch can't self-deadlock.
// Initialised lazily on first lock to avoid order-of-static-init issues.
pthread_mutex_t g_mutex;
pthread_once_t g_mutex_once = PTHREAD_ONCE_INIT;
static void mutex_init() {
    pthread_mutexattr_t a;
    pthread_mutexattr_init(&a);
    pthread_mutexattr_settype(&a, PTHREAD_MUTEX_RECURSIVE);
    pthread_mutex_init(&g_mutex, &a);
    pthread_mutexattr_destroy(&a);
}
static inline void lock()   { pthread_once(&g_mutex_once, mutex_init); pthread_mutex_lock(&g_mutex); }
static inline void unlock() { pthread_mutex_unlock(&g_mutex); }

// Whether this physical device is the Mali G52 r25p0 that needs serialisation.
// If false every wrapper below just calls straight into the next chain link,
// so on non-Mali hardware the layer costs nothing beyond a branch.
bool g_active = false;

struct LockGuard {
    bool held;
    LockGuard() : held(g_active) { if (held) lock(); }
    ~LockGuard() { if (held) unlock(); }
};

// -------- Dispatch tables --------
// Per-VkInstance and per-VkDevice, the set of function pointers pointing
// down the chain (next layer or the driver). The loader hands us these via
// VkLayerInstanceCreateInfo / VkLayerDeviceCreateInfo at create time.

struct InstanceDispatch {
    PFN_vkGetInstanceProcAddr                 GetInstanceProcAddr           = nullptr;
    PFN_vkGetDeviceProcAddr                   GetDeviceProcAddr             = nullptr;
    PFN_vkDestroyInstance                     DestroyInstance               = nullptr;
    PFN_vkEnumeratePhysicalDevices            EnumeratePhysicalDevices      = nullptr;
    PFN_vkGetPhysicalDeviceProperties         GetPhysicalDeviceProperties   = nullptr;
    PFN_vkGetPhysicalDeviceFeatures           GetPhysicalDeviceFeatures     = nullptr;
    PFN_vkCreateDevice                        CreateDevice                  = nullptr;
    PFN_vkEnumerateDeviceExtensionProperties  EnumerateDeviceExtensionProperties = nullptr;
};

struct DeviceDispatch {
    PFN_vkGetDeviceProcAddr                   GetDeviceProcAddr             = nullptr;
    PFN_vkDestroyDevice                       DestroyDevice                 = nullptr;

    // Queue submission - hottest crash family
    PFN_vkQueueSubmit                         QueueSubmit                   = nullptr;
    PFN_vkQueueWaitIdle                       QueueWaitIdle                 = nullptr;
    PFN_vkDeviceWaitIdle                      DeviceWaitIdle                = nullptr;
    PFN_vkQueuePresentKHR                     QueuePresentKHR               = nullptr;
    PFN_vkGetDeviceQueue                      GetDeviceQueue                = nullptr;

    // Descriptor sets
    PFN_vkUpdateDescriptorSets                UpdateDescriptorSets          = nullptr;
    PFN_vkAllocateDescriptorSets              AllocateDescriptorSets        = nullptr;
    PFN_vkFreeDescriptorSets                  FreeDescriptorSets            = nullptr;
    PFN_vkResetDescriptorPool                 ResetDescriptorPool           = nullptr;

    // Command buffer lifecycle
    PFN_vkBeginCommandBuffer                  BeginCommandBuffer            = nullptr;
    PFN_vkEndCommandBuffer                    EndCommandBuffer              = nullptr;
    PFN_vkResetCommandBuffer                  ResetCommandBuffer            = nullptr;
    PFN_vkAllocateCommandBuffers              AllocateCommandBuffers        = nullptr;
    PFN_vkFreeCommandBuffers                  FreeCommandBuffers            = nullptr;
    PFN_vkResetCommandPool                    ResetCommandPool              = nullptr;

    // vkCmd* - the crash site
    PFN_vkCmdBindPipeline                     CmdBindPipeline               = nullptr;
    PFN_vkCmdBindDescriptorSets               CmdBindDescriptorSets         = nullptr;
    PFN_vkCmdBindVertexBuffers                CmdBindVertexBuffers          = nullptr;
    PFN_vkCmdBindIndexBuffer                  CmdBindIndexBuffer            = nullptr;
    PFN_vkCmdDraw                             CmdDraw                       = nullptr;
    PFN_vkCmdDrawIndexed                      CmdDrawIndexed                = nullptr;
    PFN_vkCmdDrawIndirect                     CmdDrawIndirect               = nullptr;
    PFN_vkCmdDrawIndexedIndirect              CmdDrawIndexedIndirect        = nullptr;
    PFN_vkCmdDispatch                         CmdDispatch                   = nullptr;
    PFN_vkCmdCopyBuffer                       CmdCopyBuffer                 = nullptr;
    PFN_vkCmdCopyImage                        CmdCopyImage                  = nullptr;
    PFN_vkCmdCopyBufferToImage                CmdCopyBufferToImage          = nullptr;
    PFN_vkCmdCopyImageToBuffer                CmdCopyImageToBuffer          = nullptr;
    PFN_vkCmdBlitImage                        CmdBlitImage                  = nullptr;
    PFN_vkCmdClearColorImage                  CmdClearColorImage            = nullptr;
    PFN_vkCmdClearDepthStencilImage           CmdClearDepthStencilImage     = nullptr;
    PFN_vkCmdClearAttachments                 CmdClearAttachments           = nullptr;
    PFN_vkCmdFillBuffer                       CmdFillBuffer                 = nullptr;
    PFN_vkCmdUpdateBuffer                     CmdUpdateBuffer               = nullptr;
    PFN_vkCmdPipelineBarrier                  CmdPipelineBarrier            = nullptr;
    PFN_vkCmdBeginRenderPass                  CmdBeginRenderPass            = nullptr;
    PFN_vkCmdEndRenderPass                    CmdEndRenderPass              = nullptr;
    PFN_vkCmdNextSubpass                      CmdNextSubpass                = nullptr;
    PFN_vkCmdExecuteCommands                  CmdExecuteCommands            = nullptr;
    PFN_vkCmdSetViewport                      CmdSetViewport                = nullptr;
    PFN_vkCmdSetScissor                       CmdSetScissor                 = nullptr;
    PFN_vkCmdPushConstants                    CmdPushConstants              = nullptr;
    PFN_vkCmdResolveImage                     CmdResolveImage               = nullptr;
    PFN_vkCmdBeginRenderingKHR                CmdBeginRenderingKHR          = nullptr;
    PFN_vkCmdEndRenderingKHR                  CmdEndRenderingKHR            = nullptr;

    // Object create/destroy - GpuWorker calls these concurrent with vkCmd*
    PFN_vkCreateBuffer                        CreateBuffer                  = nullptr;
    PFN_vkDestroyBuffer                       DestroyBuffer                 = nullptr;
    PFN_vkCreateImage                         CreateImage                   = nullptr;
    PFN_vkDestroyImage                        DestroyImage                  = nullptr;
    PFN_vkCreateImageView                     CreateImageView               = nullptr;
    PFN_vkDestroyImageView                    DestroyImageView              = nullptr;
    PFN_vkCreateSampler                       CreateSampler                 = nullptr;
    PFN_vkDestroySampler                      DestroySampler                = nullptr;
    PFN_vkCreateShaderModule                  CreateShaderModule            = nullptr;
    PFN_vkDestroyShaderModule                 DestroyShaderModule           = nullptr;
    PFN_vkCreatePipelineLayout                CreatePipelineLayout          = nullptr;
    PFN_vkDestroyPipelineLayout               DestroyPipelineLayout         = nullptr;
    PFN_vkCreateGraphicsPipelines             CreateGraphicsPipelines       = nullptr;
    PFN_vkCreateComputePipelines              CreateComputePipelines        = nullptr;
    PFN_vkDestroyPipeline                     DestroyPipeline               = nullptr;
    PFN_vkCreateDescriptorSetLayout           CreateDescriptorSetLayout     = nullptr;
    PFN_vkDestroyDescriptorSetLayout          DestroyDescriptorSetLayout    = nullptr;
    PFN_vkCreateDescriptorPool                CreateDescriptorPool          = nullptr;
    PFN_vkDestroyDescriptorPool               DestroyDescriptorPool         = nullptr;
    PFN_vkCreateRenderPass                    CreateRenderPass              = nullptr;
    PFN_vkDestroyRenderPass                   DestroyRenderPass             = nullptr;
    PFN_vkCreateFramebuffer                   CreateFramebuffer             = nullptr;
    PFN_vkDestroyFramebuffer                  DestroyFramebuffer            = nullptr;
    PFN_vkCreateCommandPool                   CreateCommandPool             = nullptr;
    PFN_vkDestroyCommandPool                  DestroyCommandPool            = nullptr;
    PFN_vkCreateFence                         CreateFence                   = nullptr;
    PFN_vkDestroyFence                        DestroyFence                  = nullptr;
    PFN_vkCreateSemaphore                     CreateSemaphore               = nullptr;
    PFN_vkDestroySemaphore                    DestroySemaphore              = nullptr;
    PFN_vkCreateSwapchainKHR                  CreateSwapchainKHR            = nullptr;
    PFN_vkDestroySwapchainKHR                 DestroySwapchainKHR           = nullptr;
    PFN_vkGetSwapchainImagesKHR               GetSwapchainImagesKHR         = nullptr;
    PFN_vkAcquireNextImageKHR                 AcquireNextImageKHR           = nullptr;
    PFN_vkCreatePipelineCache                 CreatePipelineCache           = nullptr;
    PFN_vkDestroyPipelineCache                DestroyPipelineCache          = nullptr;
    PFN_vkMergePipelineCaches                 MergePipelineCaches           = nullptr;
    PFN_vkGetPipelineCacheData                GetPipelineCacheData          = nullptr;

    // Memory
    PFN_vkAllocateMemory                      AllocateMemory                = nullptr;
    PFN_vkFreeMemory                          FreeMemory                    = nullptr;
    PFN_vkMapMemory                           MapMemory                     = nullptr;
    PFN_vkUnmapMemory                         UnmapMemory                   = nullptr;
    PFN_vkFlushMappedMemoryRanges             FlushMappedMemoryRanges       = nullptr;
    PFN_vkInvalidateMappedMemoryRanges        InvalidateMappedMemoryRanges  = nullptr;
    PFN_vkBindBufferMemory                    BindBufferMemory              = nullptr;
    PFN_vkBindImageMemory                     BindImageMemory               = nullptr;
    PFN_vkBindBufferMemory2                   BindBufferMemory2             = nullptr;
    PFN_vkBindImageMemory2                    BindImageMemory2              = nullptr;

    // Fences
    PFN_vkWaitForFences                       WaitForFences                 = nullptr;
    PFN_vkResetFences                         ResetFences                   = nullptr;
    PFN_vkGetFenceStatus                      GetFenceStatus                = nullptr;

    // VMA and other untracked hot queries — VMA calls these concurrently
    // with VulkanWorker's vkCmd* execution and without wrapping they
    // bypass our mutex entirely, reintroducing the Mali +0x60 TOCTOU.
    PFN_vkGetBufferMemoryRequirements         GetBufferMemoryRequirements   = nullptr;
    PFN_vkGetImageMemoryRequirements          GetImageMemoryRequirements    = nullptr;
    PFN_vkGetBufferMemoryRequirements2        GetBufferMemoryRequirements2  = nullptr;
    PFN_vkGetImageMemoryRequirements2         GetImageMemoryRequirements2   = nullptr;
    PFN_vkGetImageSubresourceLayout           GetImageSubresourceLayout     = nullptr;
    PFN_vkGetImageSparseMemoryRequirements    GetImageSparseMemoryRequirements = nullptr;
    PFN_vkGetDeviceMemoryCommitment           GetDeviceMemoryCommitment     = nullptr;
    PFN_vkGetEventStatus                      GetEventStatus                = nullptr;
    PFN_vkSetEvent                            SetEvent                      = nullptr;
    PFN_vkResetEvent                          ResetEvent                    = nullptr;
    PFN_vkGetQueryPoolResults                 GetQueryPoolResults           = nullptr;
};

// Azahar creates exactly one VkInstance and one VkDevice, so we skip the
// full dispatch-table-per-handle pattern and keep a single global of each.
// Keying by `*(void**)handle` would need extra instrumentation because the
// Vulkan loader assigns different dispatch keys per layer, and we'd have
// to trampoline EnumeratePhysicalDevices to track pd->instance. For Azahar
// we don't need any of that.
std::mutex       g_tables_lock;
InstanceDispatch g_inst{};
DeviceDispatch   g_dev{};
bool             g_inst_ready = false;
bool             g_dev_ready  = false;

InstanceDispatch* get_inst(void*) { return g_inst_ready ? &g_inst : nullptr; }
DeviceDispatch*   get_dev(void*)  { return g_dev_ready  ? &g_dev  : nullptr; }

// -------- Chain info extraction --------
// On vkCreateInstance the loader passes a VkLayerInstanceCreateInfo in
// pCreateInfo->pNext. We pluck it out, grab the "next" vkGetInstanceProcAddr
// pointer, advance the chain link, then fetch vkCreateInstance through the
// advanced chain and call it. Equivalent helper for devices.

VkLayerInstanceCreateInfo* get_chain_info(const VkInstanceCreateInfo* ci, VkLayerFunction fn) {
    auto* p = static_cast<const VkLayerInstanceCreateInfo*>(ci->pNext);
    while (p && !(p->sType == VK_STRUCTURE_TYPE_LOADER_INSTANCE_CREATE_INFO && p->function == fn)) {
        p = static_cast<const VkLayerInstanceCreateInfo*>(p->pNext);
    }
    return const_cast<VkLayerInstanceCreateInfo*>(p);
}
VkLayerDeviceCreateInfo* get_chain_info(const VkDeviceCreateInfo* ci, VkLayerFunction fn) {
    auto* p = static_cast<const VkLayerDeviceCreateInfo*>(ci->pNext);
    while (p && !(p->sType == VK_STRUCTURE_TYPE_LOADER_DEVICE_CREATE_INFO && p->function == fn)) {
        p = static_cast<const VkLayerDeviceCreateInfo*>(p->pNext);
    }
    return const_cast<VkLayerDeviceCreateInfo*>(p);
}

// -------- Wrappers --------
// Thin macro: enter -> lock (if active) -> call through dispatch -> unlock.
// All of these look up the per-handle dispatch table on every call;
// that's one hash lookup, no new allocations.

#define DISPATCH_CMD(HANDLE) \
    DeviceDispatch* dd = get_dev(HANDLE); \
    LockGuard _g;

#define DISPATCH_DEV(HANDLE) \
    DeviceDispatch* dd = get_dev(HANDLE); \
    LockGuard _g;

// Command buffer wrappers: key off the dispatchable cmdbuf handle itself;
// it carries the same loader magic as its parent VkDevice.

VKAPI_ATTR VkResult VKAPI_CALL L_QueueSubmit(VkQueue q, uint32_t n, const VkSubmitInfo* p, VkFence f) {
    DeviceDispatch* dd = get_dev(q); LockGuard _g; return dd->QueueSubmit(q, n, p, f);
}
VKAPI_ATTR VkResult VKAPI_CALL L_QueueWaitIdle(VkQueue q) {
    DeviceDispatch* dd = get_dev(q); LockGuard _g; return dd->QueueWaitIdle(q);
}
VKAPI_ATTR VkResult VKAPI_CALL L_DeviceWaitIdle(VkDevice d) {
    DeviceDispatch* dd = get_dev(d); LockGuard _g; return dd->DeviceWaitIdle(d);
}
VKAPI_ATTR VkResult VKAPI_CALL L_QueuePresentKHR(VkQueue q, const VkPresentInfoKHR* p) {
    DeviceDispatch* dd = get_dev(q); LockGuard _g; return dd->QueuePresentKHR(q, p);
}

VKAPI_ATTR void VKAPI_CALL L_UpdateDescriptorSets(VkDevice d, uint32_t wc, const VkWriteDescriptorSet* w, uint32_t cc, const VkCopyDescriptorSet* c) {
    DeviceDispatch* dd = get_dev(d); LockGuard _g; dd->UpdateDescriptorSets(d, wc, w, cc, c);
}
VKAPI_ATTR VkResult VKAPI_CALL L_AllocateDescriptorSets(VkDevice d, const VkDescriptorSetAllocateInfo* a, VkDescriptorSet* s) {
    DeviceDispatch* dd = get_dev(d); LockGuard _g; return dd->AllocateDescriptorSets(d, a, s);
}
VKAPI_ATTR VkResult VKAPI_CALL L_FreeDescriptorSets(VkDevice d, VkDescriptorPool p, uint32_t c, const VkDescriptorSet* s) {
    DeviceDispatch* dd = get_dev(d); LockGuard _g; return dd->FreeDescriptorSets(d, p, c, s);
}
VKAPI_ATTR VkResult VKAPI_CALL L_ResetDescriptorPool(VkDevice d, VkDescriptorPool p, VkDescriptorPoolResetFlags f) {
    DeviceDispatch* dd = get_dev(d); LockGuard _g; return dd->ResetDescriptorPool(d, p, f);
}

VKAPI_ATTR VkResult VKAPI_CALL L_BeginCommandBuffer(VkCommandBuffer c, const VkCommandBufferBeginInfo* b) {
    DeviceDispatch* dd = get_dev(c); LockGuard _g; return dd->BeginCommandBuffer(c, b);
}
VKAPI_ATTR VkResult VKAPI_CALL L_EndCommandBuffer(VkCommandBuffer c) {
    DeviceDispatch* dd = get_dev(c); LockGuard _g; return dd->EndCommandBuffer(c);
}
VKAPI_ATTR VkResult VKAPI_CALL L_ResetCommandBuffer(VkCommandBuffer c, VkCommandBufferResetFlags f) {
    DeviceDispatch* dd = get_dev(c); LockGuard _g; return dd->ResetCommandBuffer(c, f);
}
VKAPI_ATTR VkResult VKAPI_CALL L_AllocateCommandBuffers(VkDevice d, const VkCommandBufferAllocateInfo* a, VkCommandBuffer* b) {
    DeviceDispatch* dd = get_dev(d); LockGuard _g;
    VkResult r = dd->AllocateCommandBuffers(d, a, b);
    // New command buffers inherit the device's dispatch key, so nothing else
    // is required - get_dev(cb) will route to the right DeviceDispatch.
    return r;
}
VKAPI_ATTR void VKAPI_CALL L_FreeCommandBuffers(VkDevice d, VkCommandPool p, uint32_t c, const VkCommandBuffer* b) {
    DeviceDispatch* dd = get_dev(d); LockGuard _g; dd->FreeCommandBuffers(d, p, c, b);
}
VKAPI_ATTR VkResult VKAPI_CALL L_ResetCommandPool(VkDevice d, VkCommandPool p, VkCommandPoolResetFlags f) {
    DeviceDispatch* dd = get_dev(d); LockGuard _g; return dd->ResetCommandPool(d, p, f);
}

#define CMDWRAP_VOID(NAME, PARAMS, ARGS) \
    VKAPI_ATTR void VKAPI_CALL L_##NAME PARAMS { \
        DeviceDispatch* dd = get_dev(c); LockGuard _g; dd->NAME ARGS; \
    }

CMDWRAP_VOID(CmdBindPipeline, (VkCommandBuffer c, VkPipelineBindPoint b, VkPipeline p), (c, b, p))
CMDWRAP_VOID(CmdBindDescriptorSets,
    (VkCommandBuffer c, VkPipelineBindPoint b, VkPipelineLayout l, uint32_t f, uint32_t n, const VkDescriptorSet* s, uint32_t doc, const uint32_t* dod),
    (c, b, l, f, n, s, doc, dod))
CMDWRAP_VOID(CmdBindVertexBuffers,
    (VkCommandBuffer c, uint32_t fb, uint32_t bc, const VkBuffer* b, const VkDeviceSize* o),
    (c, fb, bc, b, o))
CMDWRAP_VOID(CmdBindIndexBuffer,
    (VkCommandBuffer c, VkBuffer b, VkDeviceSize o, VkIndexType t),
    (c, b, o, t))
CMDWRAP_VOID(CmdDraw,
    (VkCommandBuffer c, uint32_t vc, uint32_t ic, uint32_t fv, uint32_t fi),
    (c, vc, ic, fv, fi))
CMDWRAP_VOID(CmdDrawIndexed,
    (VkCommandBuffer c, uint32_t ic, uint32_t inC, uint32_t fi, int32_t vo, uint32_t fins),
    (c, ic, inC, fi, vo, fins))
CMDWRAP_VOID(CmdDrawIndirect,
    (VkCommandBuffer c, VkBuffer b, VkDeviceSize o, uint32_t dc, uint32_t s),
    (c, b, o, dc, s))
CMDWRAP_VOID(CmdDrawIndexedIndirect,
    (VkCommandBuffer c, VkBuffer b, VkDeviceSize o, uint32_t dc, uint32_t s),
    (c, b, o, dc, s))
CMDWRAP_VOID(CmdDispatch,
    (VkCommandBuffer c, uint32_t x, uint32_t y, uint32_t z),
    (c, x, y, z))
CMDWRAP_VOID(CmdCopyBuffer,
    (VkCommandBuffer c, VkBuffer s, VkBuffer d, uint32_t rc, const VkBufferCopy* r),
    (c, s, d, rc, r))
CMDWRAP_VOID(CmdCopyImage,
    (VkCommandBuffer c, VkImage si, VkImageLayout sl, VkImage di, VkImageLayout dl, uint32_t rc, const VkImageCopy* r),
    (c, si, sl, di, dl, rc, r))
CMDWRAP_VOID(CmdCopyBufferToImage,
    (VkCommandBuffer c, VkBuffer s, VkImage di, VkImageLayout dl, uint32_t rc, const VkBufferImageCopy* r),
    (c, s, di, dl, rc, r))
CMDWRAP_VOID(CmdCopyImageToBuffer,
    (VkCommandBuffer c, VkImage si, VkImageLayout sl, VkBuffer d, uint32_t rc, const VkBufferImageCopy* r),
    (c, si, sl, d, rc, r))
CMDWRAP_VOID(CmdBlitImage,
    (VkCommandBuffer c, VkImage si, VkImageLayout sl, VkImage di, VkImageLayout dl, uint32_t rc, const VkImageBlit* r, VkFilter f),
    (c, si, sl, di, dl, rc, r, f))
CMDWRAP_VOID(CmdClearColorImage,
    (VkCommandBuffer c, VkImage i, VkImageLayout l, const VkClearColorValue* cv, uint32_t rc, const VkImageSubresourceRange* r),
    (c, i, l, cv, rc, r))
CMDWRAP_VOID(CmdClearDepthStencilImage,
    (VkCommandBuffer c, VkImage i, VkImageLayout l, const VkClearDepthStencilValue* dv, uint32_t rc, const VkImageSubresourceRange* r),
    (c, i, l, dv, rc, r))
CMDWRAP_VOID(CmdClearAttachments,
    (VkCommandBuffer c, uint32_t ac, const VkClearAttachment* a, uint32_t rc, const VkClearRect* r),
    (c, ac, a, rc, r))
CMDWRAP_VOID(CmdFillBuffer,
    (VkCommandBuffer c, VkBuffer b, VkDeviceSize o, VkDeviceSize s, uint32_t d),
    (c, b, o, s, d))
CMDWRAP_VOID(CmdUpdateBuffer,
    (VkCommandBuffer c, VkBuffer b, VkDeviceSize o, VkDeviceSize s, const void* d),
    (c, b, o, s, d))
CMDWRAP_VOID(CmdPipelineBarrier,
    (VkCommandBuffer c, VkPipelineStageFlags ss, VkPipelineStageFlags ds, VkDependencyFlags df,
     uint32_t mbc, const VkMemoryBarrier* mb,
     uint32_t bbc, const VkBufferMemoryBarrier* bb,
     uint32_t ibc, const VkImageMemoryBarrier* ib),
    (c, ss, ds, df, mbc, mb, bbc, bb, ibc, ib))
CMDWRAP_VOID(CmdBeginRenderPass,
    (VkCommandBuffer c, const VkRenderPassBeginInfo* b, VkSubpassContents s),
    (c, b, s))
CMDWRAP_VOID(CmdEndRenderPass, (VkCommandBuffer c), (c))
CMDWRAP_VOID(CmdNextSubpass, (VkCommandBuffer c, VkSubpassContents s), (c, s))
CMDWRAP_VOID(CmdExecuteCommands, (VkCommandBuffer c, uint32_t n, const VkCommandBuffer* b), (c, n, b))
CMDWRAP_VOID(CmdSetViewport,
    (VkCommandBuffer c, uint32_t fv, uint32_t vc, const VkViewport* v),
    (c, fv, vc, v))
CMDWRAP_VOID(CmdSetScissor,
    (VkCommandBuffer c, uint32_t fs, uint32_t sc, const VkRect2D* s),
    (c, fs, sc, s))
CMDWRAP_VOID(CmdPushConstants,
    (VkCommandBuffer c, VkPipelineLayout l, VkShaderStageFlags s, uint32_t o, uint32_t sz, const void* v),
    (c, l, s, o, sz, v))
CMDWRAP_VOID(CmdResolveImage,
    (VkCommandBuffer c, VkImage si, VkImageLayout sl, VkImage di, VkImageLayout dl, uint32_t rc, const VkImageResolve* r),
    (c, si, sl, di, dl, rc, r))
CMDWRAP_VOID(CmdBeginRenderingKHR,
    (VkCommandBuffer c, const VkRenderingInfo* i), (c, i))
CMDWRAP_VOID(CmdEndRenderingKHR, (VkCommandBuffer c), (c))

#undef CMDWRAP_VOID

// Device-level create/destroy wrappers: look up dispatch by the VkDevice.
#define DEVWRAP_RESULT(NAME, PARAMS, ARGS) \
    VKAPI_ATTR VkResult VKAPI_CALL L_##NAME PARAMS { \
        DeviceDispatch* dd = get_dev(d); LockGuard _g; return dd->NAME ARGS; \
    }
#define DEVWRAP_VOID(NAME, PARAMS, ARGS) \
    VKAPI_ATTR void VKAPI_CALL L_##NAME PARAMS { \
        DeviceDispatch* dd = get_dev(d); LockGuard _g; dd->NAME ARGS; \
    }

DEVWRAP_RESULT(CreateBuffer, (VkDevice d, const VkBufferCreateInfo* ci, const VkAllocationCallbacks* a, VkBuffer* b), (d, ci, a, b))
DEVWRAP_VOID(DestroyBuffer, (VkDevice d, VkBuffer b, const VkAllocationCallbacks* a), (d, b, a))
DEVWRAP_RESULT(CreateImage, (VkDevice d, const VkImageCreateInfo* ci, const VkAllocationCallbacks* a, VkImage* i), (d, ci, a, i))
DEVWRAP_VOID(DestroyImage, (VkDevice d, VkImage i, const VkAllocationCallbacks* a), (d, i, a))
DEVWRAP_RESULT(CreateImageView, (VkDevice d, const VkImageViewCreateInfo* ci, const VkAllocationCallbacks* a, VkImageView* v), (d, ci, a, v))
DEVWRAP_VOID(DestroyImageView, (VkDevice d, VkImageView v, const VkAllocationCallbacks* a), (d, v, a))
DEVWRAP_RESULT(CreateSampler, (VkDevice d, const VkSamplerCreateInfo* ci, const VkAllocationCallbacks* a, VkSampler* s), (d, ci, a, s))
DEVWRAP_VOID(DestroySampler, (VkDevice d, VkSampler s, const VkAllocationCallbacks* a), (d, s, a))
DEVWRAP_RESULT(CreateShaderModule, (VkDevice d, const VkShaderModuleCreateInfo* ci, const VkAllocationCallbacks* a, VkShaderModule* m), (d, ci, a, m))
DEVWRAP_VOID(DestroyShaderModule, (VkDevice d, VkShaderModule m, const VkAllocationCallbacks* a), (d, m, a))
DEVWRAP_RESULT(CreatePipelineLayout, (VkDevice d, const VkPipelineLayoutCreateInfo* ci, const VkAllocationCallbacks* a, VkPipelineLayout* p), (d, ci, a, p))
DEVWRAP_VOID(DestroyPipelineLayout, (VkDevice d, VkPipelineLayout p, const VkAllocationCallbacks* a), (d, p, a))
DEVWRAP_RESULT(CreateGraphicsPipelines, (VkDevice d, VkPipelineCache pc, uint32_t n, const VkGraphicsPipelineCreateInfo* ci, const VkAllocationCallbacks* a, VkPipeline* p), (d, pc, n, ci, a, p))
DEVWRAP_RESULT(CreateComputePipelines, (VkDevice d, VkPipelineCache pc, uint32_t n, const VkComputePipelineCreateInfo* ci, const VkAllocationCallbacks* a, VkPipeline* p), (d, pc, n, ci, a, p))
DEVWRAP_VOID(DestroyPipeline, (VkDevice d, VkPipeline p, const VkAllocationCallbacks* a), (d, p, a))
DEVWRAP_RESULT(CreateDescriptorSetLayout, (VkDevice d, const VkDescriptorSetLayoutCreateInfo* ci, const VkAllocationCallbacks* a, VkDescriptorSetLayout* dsl), (d, ci, a, dsl))
DEVWRAP_VOID(DestroyDescriptorSetLayout, (VkDevice d, VkDescriptorSetLayout dsl, const VkAllocationCallbacks* a), (d, dsl, a))
DEVWRAP_RESULT(CreateDescriptorPool, (VkDevice d, const VkDescriptorPoolCreateInfo* ci, const VkAllocationCallbacks* a, VkDescriptorPool* p), (d, ci, a, p))
DEVWRAP_VOID(DestroyDescriptorPool, (VkDevice d, VkDescriptorPool p, const VkAllocationCallbacks* a), (d, p, a))
// Diagnostic wrapper: prove whether Azahar's cached dispatch actually lands
// here during hangs, and report the returned handle so we can tell if Mali
// is giving us null-but-success (which would cause Azahar to retry).
VKAPI_ATTR VkResult VKAPI_CALL L_CreateRenderPass(
    VkDevice d, const VkRenderPassCreateInfo* ci, const VkAllocationCallbacks* a, VkRenderPass* r) {
    static int calls = 0;
    int n = __atomic_add_fetch(&calls, 1, __ATOMIC_RELAXED);
    DeviceDispatch* dd = get_dev(d);
    VkResult res;
    {
        LockGuard _g;
        res = dd->CreateRenderPass(d, ci, a, r);
    }
    if (n <= 5 || (n & 0xff) == 0) {
        LOGI("L_CreateRenderPass #%d -> res=%d handle=%p (atts=%u subpasses=%u)",
             n, (int)res, r ? (void*)(uintptr_t)*r : nullptr,
             ci ? ci->attachmentCount : 0, ci ? ci->subpassCount : 0);
    }
    return res;
}
DEVWRAP_VOID(DestroyRenderPass, (VkDevice d, VkRenderPass r, const VkAllocationCallbacks* a), (d, r, a))
DEVWRAP_RESULT(CreateFramebuffer, (VkDevice d, const VkFramebufferCreateInfo* ci, const VkAllocationCallbacks* a, VkFramebuffer* f), (d, ci, a, f))
DEVWRAP_VOID(DestroyFramebuffer, (VkDevice d, VkFramebuffer f, const VkAllocationCallbacks* a), (d, f, a))
DEVWRAP_RESULT(CreateCommandPool, (VkDevice d, const VkCommandPoolCreateInfo* ci, const VkAllocationCallbacks* a, VkCommandPool* p), (d, ci, a, p))
DEVWRAP_VOID(DestroyCommandPool, (VkDevice d, VkCommandPool p, const VkAllocationCallbacks* a), (d, p, a))
DEVWRAP_RESULT(CreateFence, (VkDevice d, const VkFenceCreateInfo* ci, const VkAllocationCallbacks* a, VkFence* f), (d, ci, a, f))
DEVWRAP_VOID(DestroyFence, (VkDevice d, VkFence f, const VkAllocationCallbacks* a), (d, f, a))
DEVWRAP_RESULT(CreateSemaphore, (VkDevice d, const VkSemaphoreCreateInfo* ci, const VkAllocationCallbacks* a, VkSemaphore* s), (d, ci, a, s))
DEVWRAP_VOID(DestroySemaphore, (VkDevice d, VkSemaphore s, const VkAllocationCallbacks* a), (d, s, a))
DEVWRAP_RESULT(CreateSwapchainKHR, (VkDevice d, const VkSwapchainCreateInfoKHR* ci, const VkAllocationCallbacks* a, VkSwapchainKHR* s), (d, ci, a, s))
DEVWRAP_VOID(DestroySwapchainKHR, (VkDevice d, VkSwapchainKHR s, const VkAllocationCallbacks* a), (d, s, a))
DEVWRAP_RESULT(GetSwapchainImagesKHR, (VkDevice d, VkSwapchainKHR s, uint32_t* n, VkImage* img), (d, s, n, img))
DEVWRAP_RESULT(AcquireNextImageKHR, (VkDevice d, VkSwapchainKHR s, uint64_t t, VkSemaphore sm, VkFence f, uint32_t* i), (d, s, t, sm, f, i))
DEVWRAP_RESULT(CreatePipelineCache, (VkDevice d, const VkPipelineCacheCreateInfo* ci, const VkAllocationCallbacks* a, VkPipelineCache* p), (d, ci, a, p))
DEVWRAP_VOID(DestroyPipelineCache, (VkDevice d, VkPipelineCache p, const VkAllocationCallbacks* a), (d, p, a))
DEVWRAP_RESULT(MergePipelineCaches, (VkDevice d, VkPipelineCache dst, uint32_t n, const VkPipelineCache* src), (d, dst, n, src))
DEVWRAP_RESULT(GetPipelineCacheData, (VkDevice d, VkPipelineCache p, size_t* sz, void* dp), (d, p, sz, dp))

DEVWRAP_RESULT(AllocateMemory, (VkDevice d, const VkMemoryAllocateInfo* ai, const VkAllocationCallbacks* a, VkDeviceMemory* m), (d, ai, a, m))
DEVWRAP_VOID(FreeMemory, (VkDevice d, VkDeviceMemory m, const VkAllocationCallbacks* a), (d, m, a))
DEVWRAP_RESULT(MapMemory, (VkDevice d, VkDeviceMemory m, VkDeviceSize o, VkDeviceSize sz, VkMemoryMapFlags f, void** pp), (d, m, o, sz, f, pp))
DEVWRAP_VOID(UnmapMemory, (VkDevice d, VkDeviceMemory m), (d, m))
DEVWRAP_RESULT(FlushMappedMemoryRanges, (VkDevice d, uint32_t n, const VkMappedMemoryRange* r), (d, n, r))
DEVWRAP_RESULT(InvalidateMappedMemoryRanges, (VkDevice d, uint32_t n, const VkMappedMemoryRange* r), (d, n, r))
DEVWRAP_RESULT(BindBufferMemory, (VkDevice d, VkBuffer b, VkDeviceMemory m, VkDeviceSize o), (d, b, m, o))
DEVWRAP_RESULT(BindImageMemory, (VkDevice d, VkImage i, VkDeviceMemory m, VkDeviceSize o), (d, i, m, o))
DEVWRAP_RESULT(BindBufferMemory2, (VkDevice d, uint32_t n, const VkBindBufferMemoryInfo* bi), (d, n, bi))
DEVWRAP_RESULT(BindImageMemory2, (VkDevice d, uint32_t n, const VkBindImageMemoryInfo* bi), (d, n, bi))

DEVWRAP_RESULT(WaitForFences, (VkDevice d, uint32_t n, const VkFence* f, VkBool32 all, uint64_t t), (d, n, f, all, t))
DEVWRAP_RESULT(ResetFences, (VkDevice d, uint32_t n, const VkFence* f), (d, n, f))
DEVWRAP_RESULT(GetFenceStatus, (VkDevice d, VkFence f), (d, f))

DEVWRAP_VOID(GetBufferMemoryRequirements, (VkDevice d, VkBuffer b, VkMemoryRequirements* r), (d, b, r))
DEVWRAP_VOID(GetImageMemoryRequirements, (VkDevice d, VkImage i, VkMemoryRequirements* r), (d, i, r))
DEVWRAP_VOID(GetBufferMemoryRequirements2, (VkDevice d, const VkBufferMemoryRequirementsInfo2* info, VkMemoryRequirements2* r), (d, info, r))
DEVWRAP_VOID(GetImageMemoryRequirements2, (VkDevice d, const VkImageMemoryRequirementsInfo2* info, VkMemoryRequirements2* r), (d, info, r))
DEVWRAP_VOID(GetImageSubresourceLayout, (VkDevice d, VkImage i, const VkImageSubresource* sr, VkSubresourceLayout* l), (d, i, sr, l))
DEVWRAP_VOID(GetImageSparseMemoryRequirements, (VkDevice d, VkImage i, uint32_t* n, VkSparseImageMemoryRequirements* r), (d, i, n, r))
DEVWRAP_VOID(GetDeviceMemoryCommitment, (VkDevice d, VkDeviceMemory m, VkDeviceSize* c), (d, m, c))
DEVWRAP_RESULT(GetEventStatus, (VkDevice d, VkEvent e), (d, e))
DEVWRAP_RESULT(SetEvent, (VkDevice d, VkEvent e), (d, e))
DEVWRAP_RESULT(ResetEvent, (VkDevice d, VkEvent e), (d, e))
DEVWRAP_RESULT(GetQueryPoolResults, (VkDevice d, VkQueryPool p, uint32_t fq, uint32_t qc, size_t sz, void* dp, VkDeviceSize st, VkQueryResultFlags f), (d, p, fq, qc, sz, dp, st, f))

#undef DEVWRAP_RESULT
#undef DEVWRAP_VOID

// -------- Create / destroy instance & device --------

// -------- In-process binary patch for libGLES_mali.so --------
//
// The Mali G52 r25p0 driver has two TOCTOU-style null-deref bugs in its
// internal dispatch helpers:
//
//   libGLES_mali.so  +0x1e05a38  ldr x21, [x8, #96]   (x8 was [x0+32], can be NULL)
//                                                     → fault addr 0x60
//   libGLES_mali.so  +0x09e5014  ldr w13, [x23, #8]   (x23 was [x19+72+8], can be NULL)
//                                                     → fault addr 0x8
//
// On our layer's first CreateInstance (when Mali is definitely mapped into
// our process) we locate libGLES_mali.so's base and plant trampolines:
// each crash-site's ldr becomes `b cave`, and the cave we occupy (a
// 224-byte zero region at offset 0x1f62f20 that sits in libGLES_mali's
// .text) does a null check and an early-exit to the function's epilogue
// when the loaded pointer is NULL.
//
// SELinux must be permissive for this to work: `mprotect(PROT_WRITE)` on
// a file-backed executable mapping requires `execmod` permission that
// stock enforcing policy does not grant to untrusted_app. Enforcing mode
// makes the mprotect fail with EACCES; we log the error and leave the
// driver untouched. The mutex-serialisation side of the layer still works.

#include <cstdio>
#include <unistd.h>
#include <sys/mman.h>

namespace {

bool g_mali_patched = false;
std::mutex g_mali_patch_mtx;
uintptr_t g_mali_base = 0;  // set by apply_mali_patches, read by watchdog/signal handler

uintptr_t find_mali_base() {
    FILE* f = std::fopen("/proc/self/maps", "r");
    if (!f) return 0;
    char line[1024];
    uintptr_t result = 0;
    while (std::fgets(line, sizeof(line), f)) {
        // Executable segment of libGLES_mali.so. We want the mapping whose
        // file offset is 0, because the patch offsets we computed are
        // relative to the file start, not the executable segment start.
        if (std::strstr(line, "/libGLES_mali.so")) {
            uintptr_t start = 0, end = 0, off = 0;
            char perms[8] = {0};
            int n = std::sscanf(line, "%lx-%lx %7s %lx", &start, &end, perms, &off);
            if (n >= 4 && off == 0) {
                result = start;
                break;
            }
        }
    }
    std::fclose(f);
    return result;
}

// Write `len` bytes at `dst` across page boundaries, flipping mprotect
// on the affected pages. Returns false if mprotect fails.
bool write_patch(void* dst, const void* src, size_t len) {
    const uintptr_t pg = sysconf(_SC_PAGESIZE);
    uintptr_t lo = reinterpret_cast<uintptr_t>(dst) & ~(pg - 1);
    uintptr_t hi = (reinterpret_cast<uintptr_t>(dst) + len + pg - 1) & ~(pg - 1);
    size_t span = hi - lo;

    if (mprotect(reinterpret_cast<void*>(lo), span, PROT_READ | PROT_WRITE) != 0) {
        LOGE("mprotect RW failed at %p span %zu: %s", (void*)lo, span, std::strerror(errno));
        return false;
    }
    std::memcpy(dst, src, len);
    if (mprotect(reinterpret_cast<void*>(lo), span, PROT_READ | PROT_EXEC) != 0) {
        LOGE("mprotect RX failed at %p span %zu: %s", (void*)lo, span, std::strerror(errno));
        return false;
    }
    __builtin___clear_cache(static_cast<char*>(dst), static_cast<char*>(dst) + len);
    return true;
}

void apply_mali_patches() {
    std::lock_guard<std::mutex> g(g_mali_patch_mtx);
    if (g_mali_patched) return;

    uintptr_t base = find_mali_base();
    if (!base) {
        LOGI("patch: libGLES_mali.so not mapped; skipping (non-Mali device?)");
        g_mali_patched = true;  // don't keep retrying
        return;
    }
    g_mali_base = base;  // publish for the watchdog/signal handler
    LOGI("patch: libGLES_mali.so base=0x%lx", base);

    // Precomputed bytes verified against the on-disk patch + objdump.
    // See mali-re/patch_mali.py for the derivation.
    constexpr uintptr_t OFF_CRASH1 = 0x1e05a38;
    constexpr uintptr_t OFF_CRASH2 = 0x009e5014;
    constexpr uintptr_t OFF_CRASH3 = 0x009e5058;  // ldr x14, [x15, #8], x15 can be NULL
    constexpr uintptr_t OFF_CRASH4 = 0x009e50b4;  // ldr x14, [x23]   , x23 can be NULL
    constexpr uintptr_t OFF_CAVE1  = 0x1f62f20;   // trampoline 1 slot
    constexpr uintptr_t OFF_CAVE2  = 0x1f62f40;   // trampoline 2 slot (+32)
    constexpr uintptr_t OFF_CAVE3  = 0x1f62f60;   // trampoline 3 slot (+64)
    constexpr uintptr_t OFF_CAVE4  = 0x1f62f80;   // trampoline 4 slot (+96)

    constexpr uint32_t ORIG_CRASH1 = 0xf9403115;  // ldr x21, [x8, #96]
    constexpr uint32_t ORIG_CRASH2 = 0xb9400aed;  // ldr w13, [x23, #8]
    constexpr uint32_t ORIG_CRASH3 = 0xf94005ee;  // ldr x14, [x15, #8]
    constexpr uint32_t ORIG_CRASH4 = 0xf94002ee;  // ldr x14, [x23]

    // Trampoline 1 (20 bytes): null-check x8, do the load, then either
    // fall through or early-return via epilogue at 0x1e05ae4 with w22=0.
    const uint32_t tramp1[5] = {
        0xb4000068,   // cbz  x8, +12
        0xf9403115,   // ldr  x21, [x8, #96]
        0x17fa8ac5,   // b    0x1e05a3c (back to insn after patched site)
        0x2a1f03f6,   // mov  w22, wzr
        0x17fa8aed,   // b    0x1e05ae4 (stack-canary check + epilogue)
    };
    // Patch at 0x1e05a38: b 0x1f62f20
    const uint32_t patch1 = 0x1405753a;

    // Trampoline 2 (20 bytes): null-check x23, do the load, then either
    // fall through or early-return via epilogue at 0x9e5258 with w0=0.
    const uint32_t tramp2[5] = {
        0xb4000077,   // cbz  x23, +12
        0xb9400aed,   // ldr  w13, [x23, #8]
        0x17aa0834,   // b    0x9e5018  (back to insn after patched site)
        0x2a1f03e0,   // mov  w0, wzr
        0x17aa0842,   // b    0x9e5258  (stack-canary check + epilogue)
    };
    // Patch at 0x9e5014: b 0x1f62f40
    const uint32_t patch2 = 0x1455f7cb;

    // Trampoline 3 (16 bytes): pointer chase [x23][x12*8] → x15. When x15
    // is NULL, neither "early-return via epilogue" nor "retry loop" work
    // cleanly:
    //   - early-return: makes Mali's own queue-waiter spin at 100% CPU
    //     for completions that never arrive (observed hang at ~26 min).
    //   - retry loop: spins reading [x13,x12*8] which can be freed by
    //     Mali's internal lifecycle thread mid-retry, producing a
    //     "SEGV fault addr garbage" when we re-read past free (~60 min).
    //
    // The right answer is in the function itself: at 0x9e504c there's a
    // `cbz w13, 9e50a0` — if the byte check flags the slot as empty,
    // the function branches to 0x9e50a0 which handles "this slot's empty,
    // try the next index". We mimic that path for a NULL pointer.
    //
    //   cbnz  x15, .have        ; pointer non-null → normal load
    //   b     0x9e50a0          ; null → take Mali's own "skip-slot" branch
    // .have:
    //   ldr   x14, [x15, #8]    ; original load
    //   b     0x9e505c          ; resume
    const uint32_t tramp3[4] = {
        0xb500004f,   // cbnz  x15, +8  (to .have)
        0x17aa084f,   // b     0x9e50a0
        0xf94005ee,   // ldr   x14, [x15, #8]
        0x17aa083c,   // b     0x9e505c
    };
    // Patch at 0x9e5058: b 0x1f62f60
    const uint32_t patch3 = 0x1455f7c2;

    // Trampoline 4 (20 bytes): mirrors patch 2 for the OTHER `ldr *, [x23]`
    // load in the same function. After patch 3 sends execution into the
    // "try next slot" branch at 0x9e50a0, the function reaches 0x9e50b4
    // where it dereferences x23 again - but x23 can be NULL by then
    // (Mali's internal thread frees [x19, #72+8] between the first and
    // second reads). Same null-check + epilogue early-return as patch 2.
    const uint32_t tramp4[5] = {
        0xb4000077,   // cbz   x23, +12
        0xf94002ee,   // ldr   x14, [x23]
        0x17aa084c,   // b     0x9e50b8
        0x2a1f03e0,   // mov   w0, wzr    (tried w0=1 to break caller's
                      //                    while(result==0) loop - hung at
                      //                    7 min instead of 45, confirms
                      //                    caller isn't a success-retry loop)
        0x17aa0832,   // b     0x9e5258
    };
    // Patch at 0x9e50b4: b 0x1f62f80
    const uint32_t patch4 = 0x1455f7b3;

    uint32_t* crash1 = reinterpret_cast<uint32_t*>(base + OFF_CRASH1);
    uint32_t* crash2 = reinterpret_cast<uint32_t*>(base + OFF_CRASH2);
    uint32_t* crash3 = reinterpret_cast<uint32_t*>(base + OFF_CRASH3);
    uint32_t* crash4 = reinterpret_cast<uint32_t*>(base + OFF_CRASH4);
    uint32_t* cave1  = reinterpret_cast<uint32_t*>(base + OFF_CAVE1);
    uint32_t* cave2  = reinterpret_cast<uint32_t*>(base + OFF_CAVE2);
    uint32_t* cave3  = reinterpret_cast<uint32_t*>(base + OFF_CAVE3);
    uint32_t* cave4  = reinterpret_cast<uint32_t*>(base + OFF_CAVE4);

    if (*crash1 != ORIG_CRASH1) {
        LOGE("patch: unexpected insn at crash1: 0x%08x (expected 0x%08x); driver changed?", *crash1, ORIG_CRASH1);
        g_mali_patched = true;
        return;
    }
    if (*crash2 != ORIG_CRASH2) {
        LOGE("patch: unexpected insn at crash2: 0x%08x (expected 0x%08x); driver changed?", *crash2, ORIG_CRASH2);
        g_mali_patched = true;
        return;
    }
    if (*crash3 != ORIG_CRASH3) {
        LOGE("patch: unexpected insn at crash3: 0x%08x (expected 0x%08x); driver changed?", *crash3, ORIG_CRASH3);
        g_mali_patched = true;
        return;
    }
    if (*crash4 != ORIG_CRASH4) {
        LOGE("patch: unexpected insn at crash4: 0x%08x (expected 0x%08x); driver changed?", *crash4, ORIG_CRASH4);
        g_mali_patched = true;
        return;
    }
    // Cave sanity: make sure no one else already patched these slots.
    // Trampolines 1, 2, 4 are 5 insns each; trampoline 3 is 4 insns.
    for (int i = 0; i < 5; ++i) {
        if (cave1[i] != 0) { LOGE("patch: cave1 dirty at +%d: 0x%08x", i, cave1[i]); g_mali_patched = true; return; }
        if (cave2[i] != 0) { LOGE("patch: cave2 dirty at +%d: 0x%08x", i, cave2[i]); g_mali_patched = true; return; }
        if (cave4[i] != 0) { LOGE("patch: cave4 dirty at +%d: 0x%08x", i, cave4[i]); g_mali_patched = true; return; }
    }
    for (int i = 0; i < 4; ++i) {
        if (cave3[i] != 0) { LOGE("patch: cave3 dirty at +%d: 0x%08x", i, cave3[i]); g_mali_patched = true; return; }
    }

    if (!write_patch(cave1, tramp1, sizeof(tramp1))) return;
    if (!write_patch(cave2, tramp2, sizeof(tramp2))) return;
    if (!write_patch(cave3, tramp3, sizeof(tramp3))) return;
    if (!write_patch(cave4, tramp4, sizeof(tramp4))) return;
    if (!write_patch(crash1, &patch1, sizeof(patch1))) return;
    if (!write_patch(crash2, &patch2, sizeof(patch2))) return;
    if (!write_patch(crash3, &patch3, sizeof(patch3))) return;
    if (!write_patch(crash4, &patch4, sizeof(patch4))) return;

    g_mali_patched = true;
    LOGI("patch: Mali G52 null-check trampolines installed (4 sites)");
}

// -------- Hang watchdog --------
//
// After 30-45 min of MK7 course-preview cycling the VulkanWorker thread
// ends up at 100% CPU inside the patch 3 skip-slot -> patch 4 early-return
// spin loop. Mali's internal state is unrecoverable once it enters this
// loop; no amount of additional null-checks has broken it. Rather than
// leave the user with a frozen app requiring force-stop, we detect the
// spin and clean-abort so Android will relaunch the activity cleanly.
//
// Detection: sample the system/user CPU time delta of the VulkanWorker
// thread every 2 seconds. If user_time grows by ~200 ticks (= 100% of
// 2 seconds at 100Hz) for N consecutive samples, call abort().
// Legitimate heavy rendering rarely keeps VulkanWorker pegged at 100%
// for that long (it's frequently in futex_wait between chunks).

#include <pthread.h>
#include <dirent.h>
#include <signal.h>
#include <sys/syscall.h>
#include <ucontext.h>
#include <atomic>
#include <errno.h>

int read_comm(int tid, char* out, size_t n) {
    char path[64];
    std::snprintf(path, sizeof(path), "/proc/self/task/%d/comm", tid);
    FILE* f = std::fopen(path, "r");
    if (!f) return -1;
    if (!std::fgets(out, (int)n, f)) { std::fclose(f); return -1; }
    std::fclose(f);
    size_t len = std::strlen(out);
    if (len && out[len - 1] == '\n') out[len - 1] = 0;
    return 0;
}

long read_utime(int tid) {
    char path[64];
    std::snprintf(path, sizeof(path), "/proc/self/task/%d/stat", tid);
    FILE* f = std::fopen(path, "r");
    if (!f) return -1;
    char buf[512];
    if (!std::fgets(buf, sizeof(buf), f)) { std::fclose(f); return -1; }
    std::fclose(f);
    // field 14 is utime. Need to skip past "(comm)" because comm can have spaces.
    char* p = std::strrchr(buf, ')');
    if (!p) return -1;
    p++;  // past ')'
    // Now we're at field 3 (state), need 14, so skip 11 more fields.
    for (int i = 0; i < 11; ++i) {
        while (*p == ' ') p++;
        while (*p && *p != ' ') p++;
    }
    return std::strtol(p, nullptr, 10);
}

int find_vulkan_worker_tid() {
    DIR* d = opendir("/proc/self/task");
    if (!d) return -1;
    int result = -1;
    struct dirent* e;
    while ((e = readdir(d))) {
        if (e->d_name[0] < '0' || e->d_name[0] > '9') continue;
        int tid = atoi(e->d_name);
        char comm[64];
        if (read_comm(tid, comm, sizeof(comm)) < 0) continue;
        if (std::strcmp(comm, "VulkanWorker") == 0) { result = tid; break; }
    }
    closedir(d);
    return result;
}

// Signal handler run on VulkanWorker when the watchdog tries to unstick
// it. The signal context gives us VulkanWorker's full CPU state; if the
// PC is inside the known Mali spin region (or our trampolines), we
// rewrite it to 0x9e4fa0's epilogue with w0=0 so Mali's call returns
// cleanly and the thread can pick up the next chunk. If the PC is
// somewhere else, we leave it alone - don't want to crash on a
// coincidental signal during unrelated work.
volatile std::atomic<int> g_unstick_count{0};

void unstick_signal_handler(int, siginfo_t*, void* ctx) {
    if (!g_mali_base) return;
    ucontext_t* uctx = static_cast<ucontext_t*>(ctx);
    uint64_t pc = uctx->uc_mcontext.pc;
    uint64_t off = pc - g_mali_base;

    // Spin region: the 0x9e4fa0 function body + our trampolines 3 and 4.
    bool in_spin =
        (off >= 0x009e4fa0 && off <= 0x009e5280) ||   // the function body
        (off >= 0x01f62f60 && off <= 0x01f62fa0);      // tramps 3 + 4

    if (in_spin) {
        // Jump to 0x9e4fa0's epilogue. The canary check in the epilogue
        // will reload x8/x9 from stack and compare; if stack is intact
        // (it should be - we're jumping WITHIN the function body) the
        // epilogue runs ret to the caller.
        uctx->uc_mcontext.pc = g_mali_base + 0x009e5258;
        uctx->uc_mcontext.regs[0] = 0;  // w0 = 0 so caller sees success
        g_unstick_count.fetch_add(1, std::memory_order_relaxed);
    }
}

void* watchdog_main(void*) {
    pthread_setname_np(pthread_self(), "MaliWatchdog");

    // Wait a bit for VulkanWorker to spawn
    sleep(15);

    int tid = find_vulkan_worker_tid();
    if (tid < 0) {
        LOGE("watchdog: could not find VulkanWorker thread; watchdog disabled");
        return nullptr;
    }
    LOGI("watchdog: monitoring VulkanWorker tid=%d", tid);

    // Install the unstick signal handler. We use SIGUSR1 because Azahar
    // doesn't use it for anything else (Android's zygote uses SIGUSR1 for
    // ART GC signals but the signal is process-scoped via pthread_kill
    // when we target a specific tid, so ART isn't affected).
    struct sigaction sa{};
    sa.sa_sigaction = unstick_signal_handler;
    sa.sa_flags = SA_SIGINFO | SA_RESTART;
    sigemptyset(&sa.sa_mask);
    if (sigaction(SIGUSR1, &sa, nullptr) != 0) {
        LOGE("watchdog: sigaction(SIGUSR1) failed: %s", std::strerror(errno));
    }

    // Spin-detection: if utime grows by ~200 ticks over 2 seconds (= 100% CPU
    // for the whole interval) for 5 consecutive checks, we try to unstick.
    // If unstick attempts don't release the thread for another 5 checks, abort.
    const int THRESHOLD_TICKS_PER_2S = 190;
    const int UNSTICK_HITS = 5;
    const int ABORT_HITS = 15;   // 30 seconds of sustained spin = unstick
                                  // didn't work, kill the app for clean
                                  // Android relaunch.

    long prev = read_utime(tid);
    int hits = 0;
    int unstick_attempts = 0;
    int last_unstick_count = 0;
    while (true) {
        sleep(2);
        long cur = read_utime(tid);
        if (cur < 0) {
            LOGE("watchdog: read_utime failed; thread gone?");
            return nullptr;
        }
        long delta = cur - prev;
        prev = cur;
        if (delta >= THRESHOLD_TICKS_PER_2S) {
            hits++;
            if (hits == UNSTICK_HITS) {
                LOGE("watchdog: spin detected, sending SIGUSR1 unstick signal");
                pthread_kill(pthread_self(), 0);  // no-op: ensure tid is valid
                int cnt_before = g_unstick_count.load(std::memory_order_relaxed);
                // Send signal to VulkanWorker (same process). Must use tgkill
                // via syscall (pthread_kill needs a pthread_t, which we don't
                // have for a peer thread - just the TID).
                syscall(__NR_tgkill, getpid(), tid, SIGUSR1);
                unstick_attempts++;
                usleep(50000);
                int cnt_after = g_unstick_count.load(std::memory_order_relaxed);
                if (cnt_after > cnt_before) {
                    LOGI("watchdog: unstick attempt #%d applied (handler fired)",
                         unstick_attempts);
                    last_unstick_count = cnt_after;
                } else {
                    LOGE("watchdog: unstick signal didn't land on spin PC");
                }
            } else if (hits > UNSTICK_HITS && hits % 5 == 0) {
                LOGE("watchdog: still spinning (%ld ticks/2s, hit %d) - retrying unstick",
                     delta, hits);
                syscall(__NR_tgkill, getpid(), tid, SIGUSR1);
                unstick_attempts++;
            }
            if (hits >= ABORT_HITS) {
                LOGE("watchdog: unstick exhausted (%d attempts) - aborting for clean restart",
                     unstick_attempts);
                usleep(100000);
                abort();
            }
        } else {
            if (hits > 0) {
                LOGI("watchdog: VulkanWorker recovered after %d spin samples, %d unsticks (%ld ticks/2s now)",
                     hits, unstick_attempts, delta);
            }
            hits = 0;
            unstick_attempts = 0;
        }
    }
    return nullptr;
}

void start_watchdog_once() {
    static pthread_once_t once = PTHREAD_ONCE_INIT;
    pthread_once(&once, []() {
        pthread_t t;
        if (pthread_create(&t, nullptr, watchdog_main, nullptr) == 0) {
            pthread_detach(t);
            LOGI("watchdog: thread started");
        } else {
            LOGE("watchdog: failed to start thread");
        }
    });
}

} // namespace

VKAPI_ATTR VkResult VKAPI_CALL L_CreateInstance(
    const VkInstanceCreateInfo* pCreateInfo,
    const VkAllocationCallbacks* pAllocator,
    VkInstance* pInstance) {

    auto* chain = get_chain_info(pCreateInfo, VK_LAYER_LINK_INFO);
    if (!chain) return VK_ERROR_INITIALIZATION_FAILED;

    PFN_vkGetInstanceProcAddr nextGIPA = chain->u.pLayerInfo->pfnNextGetInstanceProcAddr;
    // Advance chain so the next layer sees its own link at the head.
    chain->u.pLayerInfo = chain->u.pLayerInfo->pNext;

    auto fpCreateInstance = (PFN_vkCreateInstance) nextGIPA(nullptr, "vkCreateInstance");
    if (!fpCreateInstance) return VK_ERROR_INITIALIZATION_FAILED;

    VkResult r = fpCreateInstance(pCreateInfo, pAllocator, pInstance);
    if (r != VK_SUCCESS) return r;

    InstanceDispatch id{};
    id.GetInstanceProcAddr          = nextGIPA;
    id.GetDeviceProcAddr            = (PFN_vkGetDeviceProcAddr)                  nextGIPA(*pInstance, "vkGetDeviceProcAddr");
    id.DestroyInstance              = (PFN_vkDestroyInstance)                    nextGIPA(*pInstance, "vkDestroyInstance");
    id.EnumeratePhysicalDevices     = (PFN_vkEnumeratePhysicalDevices)           nextGIPA(*pInstance, "vkEnumeratePhysicalDevices");
    id.GetPhysicalDeviceProperties  = (PFN_vkGetPhysicalDeviceProperties)        nextGIPA(*pInstance, "vkGetPhysicalDeviceProperties");
    id.GetPhysicalDeviceFeatures    = (PFN_vkGetPhysicalDeviceFeatures)          nextGIPA(*pInstance, "vkGetPhysicalDeviceFeatures");
    id.CreateDevice                 = (PFN_vkCreateDevice)                       nextGIPA(*pInstance, "vkCreateDevice");
    id.EnumerateDeviceExtensionProperties =
        (PFN_vkEnumerateDeviceExtensionProperties)                               nextGIPA(*pInstance, "vkEnumerateDeviceExtensionProperties");

    {
        std::lock_guard<std::mutex> g(g_tables_lock);
        g_inst = id;
        g_inst_ready = true;
    }
    LOGI("layer loaded; instance=%p", *pInstance);

    // Apply the in-process binary patches now that libGLES_mali.so is
    // guaranteed to be mapped. Safe to call on non-Mali devices - the
    // patcher bails gracefully if the library isn't present.
    apply_mali_patches();

    // Start the hang-watchdog thread (one-shot). Detects VulkanWorker
    // spin-at-100%-CPU and aborts so Android relaunches the activity
    // instead of leaving the user with a frozen app.
    start_watchdog_once();

    return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL L_DestroyInstance(VkInstance inst, const VkAllocationCallbacks* a) {
    PFN_vkDestroyInstance f = nullptr;
    {
        std::lock_guard<std::mutex> g(g_tables_lock);
        f = g_inst.DestroyInstance;
        g_inst_ready = false;
        g_inst = {};
    }
    if (f) f(inst, a);
}

// Decide Mali-G52 based on the physical device's reported name. This is
// the one place we flip `g_active` from false (default) to true.
void maybe_arm_mali_gate(InstanceDispatch* id, VkPhysicalDevice pd) {
    if (!id || !id->GetPhysicalDeviceProperties) return;
    VkPhysicalDeviceProperties props{};
    id->GetPhysicalDeviceProperties(pd, &props);
    LOGI("device: '%s' apiVersion=0x%x driverVersion=0x%x",
         props.deviceName, props.apiVersion, props.driverVersion);
    if (std::strstr(props.deviceName, "Mali-G52") != nullptr) {
        g_active = true;
        LOGI("Mali-G52 detected -> serialisation ACTIVE");
    } else {
        g_active = false;
        LOGI("non Mali-G52 -> serialisation PASS-THROUGH");
    }
}

VKAPI_ATTR VkResult VKAPI_CALL L_CreateDevice(
    VkPhysicalDevice physicalDevice,
    const VkDeviceCreateInfo* pCreateInfo,
    const VkAllocationCallbacks* pAllocator,
    VkDevice* pDevice) {

    InstanceDispatch* id = nullptr;
    {
        std::lock_guard<std::mutex> g(g_tables_lock);
        if (g_inst_ready) id = &g_inst;
    }
    if (!id) { LOGE("CreateDevice: no InstanceDispatch"); return VK_ERROR_INITIALIZATION_FAILED; }

    maybe_arm_mali_gate(id, physicalDevice);

    // Android's Vulkan loader doesn't insert a VkLayerDeviceCreateInfo node
    // in pCreateInfo->pNext the way the desktop Khronos loader does, so the
    // standard "advance the chain via VK_LAYER_LINK_INFO" dance leaves us
    // with a null pointer and CreateDevice returns INITIALIZATION_FAILED.
    // Instead, use the vkCreateDevice we already resolved through the
    // instance chain during L_CreateInstance (next-layer or driver). That
    // function pointer already points past us in the chain, so calling it
    // directly produces the same effect as chain-advancement.
    PFN_vkCreateDevice fpCreateDevice = id->CreateDevice;
    if (!fpCreateDevice) { LOGE("CreateDevice: id->CreateDevice is null"); return VK_ERROR_INITIALIZATION_FAILED; }

    VkResult r = fpCreateDevice(physicalDevice, pCreateInfo, pAllocator, pDevice);
    LOGI("CreateDevice: real CreateDevice returned %d, device=%p", (int)r, (void*)(*pDevice));
    if (r != VK_SUCCESS) return r;

    // Use the vkGetDeviceProcAddr we resolved through the instance chain in
    // L_CreateInstance; that pointer already references the next layer or
    // driver below us, same as the chain-advance version would.
    PFN_vkGetDeviceProcAddr nextGDPA = id->GetDeviceProcAddr;
    if (!nextGDPA) { LOGE("CreateDevice: id->GetDeviceProcAddr is null"); return VK_ERROR_INITIALIZATION_FAILED; }

    DeviceDispatch dd{};
    dd.GetDeviceProcAddr = nextGDPA;

#define RESOLVE(Member, Name) dd.Member = (PFN_vk##Name) nextGDPA(*pDevice, "vk" #Name)

    RESOLVE(DestroyDevice,                  DestroyDevice);
    RESOLVE(QueueSubmit,                    QueueSubmit);
    RESOLVE(QueueWaitIdle,                  QueueWaitIdle);
    RESOLVE(DeviceWaitIdle,                 DeviceWaitIdle);
    RESOLVE(QueuePresentKHR,                QueuePresentKHR);
    RESOLVE(GetDeviceQueue,                 GetDeviceQueue);

    RESOLVE(UpdateDescriptorSets,           UpdateDescriptorSets);
    RESOLVE(AllocateDescriptorSets,         AllocateDescriptorSets);
    RESOLVE(FreeDescriptorSets,             FreeDescriptorSets);
    RESOLVE(ResetDescriptorPool,            ResetDescriptorPool);

    RESOLVE(BeginCommandBuffer,             BeginCommandBuffer);
    RESOLVE(EndCommandBuffer,               EndCommandBuffer);
    RESOLVE(ResetCommandBuffer,             ResetCommandBuffer);
    RESOLVE(AllocateCommandBuffers,         AllocateCommandBuffers);
    RESOLVE(FreeCommandBuffers,             FreeCommandBuffers);
    RESOLVE(ResetCommandPool,               ResetCommandPool);

    RESOLVE(CmdBindPipeline,                CmdBindPipeline);
    RESOLVE(CmdBindDescriptorSets,          CmdBindDescriptorSets);
    RESOLVE(CmdBindVertexBuffers,           CmdBindVertexBuffers);
    RESOLVE(CmdBindIndexBuffer,             CmdBindIndexBuffer);
    RESOLVE(CmdDraw,                        CmdDraw);
    RESOLVE(CmdDrawIndexed,                 CmdDrawIndexed);
    RESOLVE(CmdDrawIndirect,                CmdDrawIndirect);
    RESOLVE(CmdDrawIndexedIndirect,         CmdDrawIndexedIndirect);
    RESOLVE(CmdDispatch,                    CmdDispatch);
    RESOLVE(CmdCopyBuffer,                  CmdCopyBuffer);
    RESOLVE(CmdCopyImage,                   CmdCopyImage);
    RESOLVE(CmdCopyBufferToImage,           CmdCopyBufferToImage);
    RESOLVE(CmdCopyImageToBuffer,           CmdCopyImageToBuffer);
    RESOLVE(CmdBlitImage,                   CmdBlitImage);
    RESOLVE(CmdClearColorImage,             CmdClearColorImage);
    RESOLVE(CmdClearDepthStencilImage,      CmdClearDepthStencilImage);
    RESOLVE(CmdClearAttachments,            CmdClearAttachments);
    RESOLVE(CmdFillBuffer,                  CmdFillBuffer);
    RESOLVE(CmdUpdateBuffer,                CmdUpdateBuffer);
    RESOLVE(CmdPipelineBarrier,             CmdPipelineBarrier);
    RESOLVE(CmdBeginRenderPass,             CmdBeginRenderPass);
    RESOLVE(CmdEndRenderPass,               CmdEndRenderPass);
    RESOLVE(CmdNextSubpass,                 CmdNextSubpass);
    RESOLVE(CmdExecuteCommands,             CmdExecuteCommands);
    RESOLVE(CmdSetViewport,                 CmdSetViewport);
    RESOLVE(CmdSetScissor,                  CmdSetScissor);
    RESOLVE(CmdPushConstants,               CmdPushConstants);
    RESOLVE(CmdResolveImage,                CmdResolveImage);
    RESOLVE(CmdBeginRenderingKHR,           CmdBeginRenderingKHR);
    RESOLVE(CmdEndRenderingKHR,             CmdEndRenderingKHR);

    RESOLVE(CreateBuffer,                   CreateBuffer);
    RESOLVE(DestroyBuffer,                  DestroyBuffer);
    RESOLVE(CreateImage,                    CreateImage);
    RESOLVE(DestroyImage,                   DestroyImage);
    RESOLVE(CreateImageView,                CreateImageView);
    RESOLVE(DestroyImageView,               DestroyImageView);
    RESOLVE(CreateSampler,                  CreateSampler);
    RESOLVE(DestroySampler,                 DestroySampler);
    RESOLVE(CreateShaderModule,             CreateShaderModule);
    RESOLVE(DestroyShaderModule,            DestroyShaderModule);
    RESOLVE(CreatePipelineLayout,           CreatePipelineLayout);
    RESOLVE(DestroyPipelineLayout,          DestroyPipelineLayout);
    RESOLVE(CreateGraphicsPipelines,        CreateGraphicsPipelines);
    RESOLVE(CreateComputePipelines,         CreateComputePipelines);
    RESOLVE(DestroyPipeline,                DestroyPipeline);
    RESOLVE(CreateDescriptorSetLayout,      CreateDescriptorSetLayout);
    RESOLVE(DestroyDescriptorSetLayout,     DestroyDescriptorSetLayout);
    RESOLVE(CreateDescriptorPool,           CreateDescriptorPool);
    RESOLVE(DestroyDescriptorPool,          DestroyDescriptorPool);
    RESOLVE(CreateRenderPass,               CreateRenderPass);
    RESOLVE(DestroyRenderPass,              DestroyRenderPass);
    RESOLVE(CreateFramebuffer,              CreateFramebuffer);
    RESOLVE(DestroyFramebuffer,             DestroyFramebuffer);
    RESOLVE(CreateCommandPool,              CreateCommandPool);
    RESOLVE(DestroyCommandPool,             DestroyCommandPool);
    RESOLVE(CreateFence,                    CreateFence);
    RESOLVE(DestroyFence,                   DestroyFence);
    RESOLVE(CreateSemaphore,                CreateSemaphore);
    RESOLVE(DestroySemaphore,               DestroySemaphore);
    RESOLVE(CreateSwapchainKHR,             CreateSwapchainKHR);
    RESOLVE(DestroySwapchainKHR,            DestroySwapchainKHR);
    RESOLVE(GetSwapchainImagesKHR,          GetSwapchainImagesKHR);
    RESOLVE(AcquireNextImageKHR,            AcquireNextImageKHR);
    RESOLVE(CreatePipelineCache,            CreatePipelineCache);
    RESOLVE(DestroyPipelineCache,           DestroyPipelineCache);
    RESOLVE(MergePipelineCaches,            MergePipelineCaches);
    RESOLVE(GetPipelineCacheData,           GetPipelineCacheData);

    RESOLVE(AllocateMemory,                 AllocateMemory);
    RESOLVE(FreeMemory,                     FreeMemory);
    RESOLVE(MapMemory,                      MapMemory);
    RESOLVE(UnmapMemory,                    UnmapMemory);
    RESOLVE(FlushMappedMemoryRanges,        FlushMappedMemoryRanges);
    RESOLVE(InvalidateMappedMemoryRanges,   InvalidateMappedMemoryRanges);
    RESOLVE(BindBufferMemory,               BindBufferMemory);
    RESOLVE(BindImageMemory,                BindImageMemory);
    RESOLVE(BindBufferMemory2,              BindBufferMemory2);
    RESOLVE(BindImageMemory2,               BindImageMemory2);

    RESOLVE(WaitForFences,                  WaitForFences);
    RESOLVE(ResetFences,                    ResetFences);
    RESOLVE(GetFenceStatus,                 GetFenceStatus);

    RESOLVE(GetBufferMemoryRequirements,       GetBufferMemoryRequirements);
    RESOLVE(GetImageMemoryRequirements,        GetImageMemoryRequirements);
    RESOLVE(GetBufferMemoryRequirements2,      GetBufferMemoryRequirements2);
    RESOLVE(GetImageMemoryRequirements2,       GetImageMemoryRequirements2);
    RESOLVE(GetImageSubresourceLayout,         GetImageSubresourceLayout);
    RESOLVE(GetImageSparseMemoryRequirements,  GetImageSparseMemoryRequirements);
    RESOLVE(GetDeviceMemoryCommitment,         GetDeviceMemoryCommitment);
    RESOLVE(GetEventStatus,                    GetEventStatus);
    RESOLVE(SetEvent,                          SetEvent);
    RESOLVE(ResetEvent,                        ResetEvent);
    RESOLVE(GetQueryPoolResults,               GetQueryPoolResults);

#undef RESOLVE

    {
        std::lock_guard<std::mutex> g(g_tables_lock);
        g_dev = dd;
        g_dev_ready = true;
    }
    return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL L_DestroyDevice(VkDevice d, const VkAllocationCallbacks* a) {
    PFN_vkDestroyDevice f = nullptr;
    {
        std::lock_guard<std::mutex> g(g_tables_lock);
        f = g_dev.DestroyDevice;
        g_dev_ready = false;
        g_dev = {};
    }
    if (f) f(d, a);
}

// -------- Enumerate layer / extension properties --------

constexpr VkLayerProperties kLayerProps = {
    LAYER_NAME,
    VK_MAKE_VERSION(1, 3, 0),
    1,
    "Azahar Mali-G52 serialization"
};

VKAPI_ATTR VkResult VKAPI_CALL L_EnumerateInstanceLayerProperties(uint32_t* pCount, VkLayerProperties* pProps) {
    if (!pProps) { *pCount = 1; return VK_SUCCESS; }
    if (*pCount < 1) return VK_INCOMPLETE;
    *pCount = 1;
    *pProps = kLayerProps;
    return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL L_EnumerateDeviceLayerProperties(VkPhysicalDevice, uint32_t* pCount, VkLayerProperties* pProps) {
    return L_EnumerateInstanceLayerProperties(pCount, pProps);
}

VKAPI_ATTR VkResult VKAPI_CALL L_EnumerateInstanceExtensionProperties(const char* pLayerName, uint32_t* pCount, VkExtensionProperties*) {
    if (pLayerName && std::strcmp(pLayerName, LAYER_NAME) == 0) {
        *pCount = 0;
        return VK_SUCCESS;
    }
    return VK_ERROR_LAYER_NOT_PRESENT;
}

VKAPI_ATTR VkResult VKAPI_CALL L_EnumerateDeviceExtensionProperties(VkPhysicalDevice pd, const char* pLayerName, uint32_t* pCount, VkExtensionProperties* pProps) {
    if (pLayerName && std::strcmp(pLayerName, LAYER_NAME) == 0) {
        *pCount = 0;
        return VK_SUCCESS;
    }
    InstanceDispatch* id = get_inst(pd);
    if (!id || !id->EnumerateDeviceExtensionProperties) return VK_ERROR_LAYER_NOT_PRESENT;
    return id->EnumerateDeviceExtensionProperties(pd, pLayerName, pCount, pProps);
}

// -------- ProcAddr routing --------

struct NameFn { const char* name; PFN_vkVoidFunction fn; };

#define ENTRY(Name, Wrapped) { "vk" #Name, (PFN_vkVoidFunction)L_##Wrapped }

const NameFn g_inst_funcs[] = {
    ENTRY(GetInstanceProcAddr,          /*routed below*/CreateInstance),  // unused sentinel; handled before lookup
    ENTRY(CreateInstance,                   CreateInstance),
    ENTRY(DestroyInstance,                  DestroyInstance),
    ENTRY(CreateDevice,                     CreateDevice),
    ENTRY(EnumerateInstanceLayerProperties, EnumerateInstanceLayerProperties),
    ENTRY(EnumerateDeviceLayerProperties,   EnumerateDeviceLayerProperties),
    ENTRY(EnumerateInstanceExtensionProperties, EnumerateInstanceExtensionProperties),
    ENTRY(EnumerateDeviceExtensionProperties,   EnumerateDeviceExtensionProperties),
};

const NameFn g_dev_funcs[] = {
    ENTRY(DestroyDevice,                    DestroyDevice),
    ENTRY(QueueSubmit,                      QueueSubmit),
    ENTRY(QueueWaitIdle,                    QueueWaitIdle),
    ENTRY(DeviceWaitIdle,                   DeviceWaitIdle),
    ENTRY(QueuePresentKHR,                  QueuePresentKHR),
    ENTRY(UpdateDescriptorSets,             UpdateDescriptorSets),
    ENTRY(AllocateDescriptorSets,           AllocateDescriptorSets),
    ENTRY(FreeDescriptorSets,               FreeDescriptorSets),
    ENTRY(ResetDescriptorPool,              ResetDescriptorPool),
    ENTRY(BeginCommandBuffer,               BeginCommandBuffer),
    ENTRY(EndCommandBuffer,                 EndCommandBuffer),
    ENTRY(ResetCommandBuffer,               ResetCommandBuffer),
    ENTRY(AllocateCommandBuffers,           AllocateCommandBuffers),
    ENTRY(FreeCommandBuffers,               FreeCommandBuffers),
    ENTRY(ResetCommandPool,                 ResetCommandPool),
    ENTRY(CmdBindPipeline,                  CmdBindPipeline),
    ENTRY(CmdBindDescriptorSets,            CmdBindDescriptorSets),
    ENTRY(CmdBindVertexBuffers,             CmdBindVertexBuffers),
    ENTRY(CmdBindIndexBuffer,               CmdBindIndexBuffer),
    ENTRY(CmdDraw,                          CmdDraw),
    ENTRY(CmdDrawIndexed,                   CmdDrawIndexed),
    ENTRY(CmdDrawIndirect,                  CmdDrawIndirect),
    ENTRY(CmdDrawIndexedIndirect,           CmdDrawIndexedIndirect),
    ENTRY(CmdDispatch,                      CmdDispatch),
    ENTRY(CmdCopyBuffer,                    CmdCopyBuffer),
    ENTRY(CmdCopyImage,                     CmdCopyImage),
    ENTRY(CmdCopyBufferToImage,             CmdCopyBufferToImage),
    ENTRY(CmdCopyImageToBuffer,             CmdCopyImageToBuffer),
    ENTRY(CmdBlitImage,                     CmdBlitImage),
    ENTRY(CmdClearColorImage,               CmdClearColorImage),
    ENTRY(CmdClearDepthStencilImage,        CmdClearDepthStencilImage),
    ENTRY(CmdClearAttachments,              CmdClearAttachments),
    ENTRY(CmdFillBuffer,                    CmdFillBuffer),
    ENTRY(CmdUpdateBuffer,                  CmdUpdateBuffer),
    ENTRY(CmdPipelineBarrier,               CmdPipelineBarrier),
    ENTRY(CmdBeginRenderPass,               CmdBeginRenderPass),
    ENTRY(CmdEndRenderPass,                 CmdEndRenderPass),
    ENTRY(CmdNextSubpass,                   CmdNextSubpass),
    ENTRY(CmdExecuteCommands,               CmdExecuteCommands),
    ENTRY(CmdSetViewport,                   CmdSetViewport),
    ENTRY(CmdSetScissor,                    CmdSetScissor),
    ENTRY(CmdPushConstants,                 CmdPushConstants),
    ENTRY(CmdResolveImage,                  CmdResolveImage),
    ENTRY(CmdBeginRenderingKHR,             CmdBeginRenderingKHR),
    ENTRY(CmdEndRenderingKHR,               CmdEndRenderingKHR),
    ENTRY(CreateBuffer,                     CreateBuffer),
    ENTRY(DestroyBuffer,                    DestroyBuffer),
    ENTRY(CreateImage,                      CreateImage),
    ENTRY(DestroyImage,                     DestroyImage),
    ENTRY(CreateImageView,                  CreateImageView),
    ENTRY(DestroyImageView,                 DestroyImageView),
    ENTRY(CreateSampler,                    CreateSampler),
    ENTRY(DestroySampler,                   DestroySampler),
    ENTRY(CreateShaderModule,               CreateShaderModule),
    ENTRY(DestroyShaderModule,              DestroyShaderModule),
    ENTRY(CreatePipelineLayout,             CreatePipelineLayout),
    ENTRY(DestroyPipelineLayout,            DestroyPipelineLayout),
    ENTRY(CreateGraphicsPipelines,          CreateGraphicsPipelines),
    ENTRY(CreateComputePipelines,           CreateComputePipelines),
    ENTRY(DestroyPipeline,                  DestroyPipeline),
    ENTRY(CreateDescriptorSetLayout,        CreateDescriptorSetLayout),
    ENTRY(DestroyDescriptorSetLayout,       DestroyDescriptorSetLayout),
    ENTRY(CreateDescriptorPool,             CreateDescriptorPool),
    ENTRY(DestroyDescriptorPool,            DestroyDescriptorPool),
    ENTRY(CreateRenderPass,                 CreateRenderPass),
    ENTRY(DestroyRenderPass,                DestroyRenderPass),
    ENTRY(CreateFramebuffer,                CreateFramebuffer),
    ENTRY(DestroyFramebuffer,               DestroyFramebuffer),
    ENTRY(CreateCommandPool,                CreateCommandPool),
    ENTRY(DestroyCommandPool,               DestroyCommandPool),
    ENTRY(CreateFence,                      CreateFence),
    ENTRY(DestroyFence,                     DestroyFence),
    ENTRY(CreateSemaphore,                  CreateSemaphore),
    ENTRY(DestroySemaphore,                 DestroySemaphore),
    ENTRY(CreateSwapchainKHR,               CreateSwapchainKHR),
    ENTRY(DestroySwapchainKHR,              DestroySwapchainKHR),
    ENTRY(GetSwapchainImagesKHR,            GetSwapchainImagesKHR),
    ENTRY(AcquireNextImageKHR,              AcquireNextImageKHR),
    ENTRY(CreatePipelineCache,              CreatePipelineCache),
    ENTRY(DestroyPipelineCache,             DestroyPipelineCache),
    ENTRY(MergePipelineCaches,              MergePipelineCaches),
    ENTRY(GetPipelineCacheData,             GetPipelineCacheData),
    ENTRY(AllocateMemory,                   AllocateMemory),
    ENTRY(FreeMemory,                       FreeMemory),
    ENTRY(MapMemory,                        MapMemory),
    ENTRY(UnmapMemory,                      UnmapMemory),
    ENTRY(FlushMappedMemoryRanges,          FlushMappedMemoryRanges),
    ENTRY(InvalidateMappedMemoryRanges,     InvalidateMappedMemoryRanges),
    ENTRY(BindBufferMemory,                 BindBufferMemory),
    ENTRY(BindImageMemory,                  BindImageMemory),
    ENTRY(BindBufferMemory2,                BindBufferMemory2),
    ENTRY(BindImageMemory2,                 BindImageMemory2),
    ENTRY(WaitForFences,                    WaitForFences),
    ENTRY(ResetFences,                      ResetFences),
    ENTRY(GetFenceStatus,                   GetFenceStatus),
    ENTRY(GetBufferMemoryRequirements,      GetBufferMemoryRequirements),
    ENTRY(GetImageMemoryRequirements,       GetImageMemoryRequirements),
    ENTRY(GetBufferMemoryRequirements2,     GetBufferMemoryRequirements2),
    ENTRY(GetImageMemoryRequirements2,      GetImageMemoryRequirements2),
    ENTRY(GetImageSubresourceLayout,        GetImageSubresourceLayout),
    ENTRY(GetImageSparseMemoryRequirements, GetImageSparseMemoryRequirements),
    ENTRY(GetDeviceMemoryCommitment,        GetDeviceMemoryCommitment),
    ENTRY(GetEventStatus,                   GetEventStatus),
    ENTRY(SetEvent,                         SetEvent),
    ENTRY(ResetEvent,                       ResetEvent),
    ENTRY(GetQueryPoolResults,              GetQueryPoolResults),
};
#undef ENTRY

PFN_vkVoidFunction lookup(const NameFn* tbl, size_t n, const char* name) {
    for (size_t i = 0; i < n; ++i) if (std::strcmp(tbl[i].name, name) == 0) return tbl[i].fn;
    return nullptr;
}

} // namespace

// -------- Exported loader-facing symbols --------

extern "C" VK_LAYER_EXPORT VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL
vkGetDeviceProcAddr(VkDevice device, const char* pName) {
    if (std::strcmp(pName, "vkGetDeviceProcAddr") == 0) {
        return (PFN_vkVoidFunction) vkGetDeviceProcAddr;
    }
    if (auto f = lookup(g_dev_funcs, sizeof(g_dev_funcs)/sizeof(g_dev_funcs[0]), pName)) return f;
    DeviceDispatch* dd = get_dev(device);
    if (!dd || !dd->GetDeviceProcAddr) return nullptr;
    return dd->GetDeviceProcAddr(device, pName);
}

extern "C" VK_LAYER_EXPORT VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL
vkGetInstanceProcAddr(VkInstance instance, const char* pName) {
    if (std::strcmp(pName, "vkGetInstanceProcAddr") == 0) {
        return (PFN_vkVoidFunction) vkGetInstanceProcAddr;
    }
    if (std::strcmp(pName, "vkGetDeviceProcAddr") == 0) {
        return (PFN_vkVoidFunction) vkGetDeviceProcAddr;
    }
    if (auto f = lookup(g_inst_funcs, sizeof(g_inst_funcs)/sizeof(g_inst_funcs[0]), pName)) return f;
    if (auto f = lookup(g_dev_funcs, sizeof(g_dev_funcs)/sizeof(g_dev_funcs[0]), pName))  return f;
    if (!instance) return nullptr;
    InstanceDispatch* id = get_inst(instance);
    if (!id || !id->GetInstanceProcAddr) return nullptr;
    return id->GetInstanceProcAddr(instance, pName);
}

extern "C" VK_LAYER_EXPORT VKAPI_ATTR VkResult VKAPI_CALL
vkNegotiateLoaderLayerInterfaceVersion(VkNegotiateLayerInterface* pVersionStruct) {
    if (pVersionStruct->loaderLayerInterfaceVersion >= 2) {
        pVersionStruct->loaderLayerInterfaceVersion = 2;
        pVersionStruct->pfnGetInstanceProcAddr       = vkGetInstanceProcAddr;
        pVersionStruct->pfnGetDeviceProcAddr         = vkGetDeviceProcAddr;
        pVersionStruct->pfnGetPhysicalDeviceProcAddr = nullptr;
    }
    return VK_SUCCESS;
}

extern "C" VK_LAYER_EXPORT VKAPI_ATTR VkResult VKAPI_CALL
vkEnumerateInstanceLayerProperties(uint32_t* pCount, VkLayerProperties* pProps) {
    return L_EnumerateInstanceLayerProperties(pCount, pProps);
}

extern "C" VK_LAYER_EXPORT VKAPI_ATTR VkResult VKAPI_CALL
vkEnumerateInstanceExtensionProperties(const char* pLayerName, uint32_t* pCount, VkExtensionProperties* pProps) {
    return L_EnumerateInstanceExtensionProperties(pLayerName, pCount, pProps);
}
