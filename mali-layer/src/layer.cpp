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
DEVWRAP_RESULT(CreateRenderPass, (VkDevice d, const VkRenderPassCreateInfo* ci, const VkAllocationCallbacks* a, VkRenderPass* r), (d, ci, a, r))
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
    LOGI("patch: libGLES_mali.so base=0x%lx", base);

    // Precomputed bytes verified against the on-disk patch + objdump.
    // See mali-re/patch_mali.py for the derivation.
    constexpr uintptr_t OFF_CRASH1 = 0x1e05a38;
    constexpr uintptr_t OFF_CRASH2 = 0x009e5014;
    constexpr uintptr_t OFF_CRASH3 = 0x009e5058;  // ldr x14, [x15, #8], x15 can be NULL
    constexpr uintptr_t OFF_CAVE1  = 0x1f62f20;   // trampoline 1 slot
    constexpr uintptr_t OFF_CAVE2  = 0x1f62f40;   // trampoline 2 slot (+32)
    constexpr uintptr_t OFF_CAVE3  = 0x1f62f60;   // trampoline 3 slot (+64)

    constexpr uint32_t ORIG_CRASH1 = 0xf9403115;  // ldr x21, [x8, #96]
    constexpr uint32_t ORIG_CRASH2 = 0xb9400aed;  // ldr w13, [x23, #8]
    constexpr uint32_t ORIG_CRASH3 = 0xf94005ee;  // ldr x14, [x15, #8]

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

    // Trampoline 3 (36 bytes): the pointer chase here is `[x23][x12*8]`
    // where the table-slot `x15` can be transiently NULL during another
    // Mali thread's fixup. Just bailing to the epilogue in that case made
    // Mali's own queue-waiter thread spin forever at 100% CPU (it waits
    // for the completion of work our early-return skipped). So: retry
    // loading x15 from its memory source a bounded number of times before
    // giving up — gives the other thread room to finish its write.
    //
    //   mov   w16, #1024                   ; bounded retry counter
    // .retry:
    //   cbnz  x15, .have
    //   ldr   x15, [x13, x12, lsl #3]      ; re-read the table slot
    //   subs  w16, w16, #1
    //   b.ne  .retry
    //   mov   w0, wzr                      ; counter exhausted; fall back
    //   b     0x9e5258
    // .have:
    //   ldr   x14, [x15, #8]               ; original load, now safe
    //   b     0x9e505c                     ; resume
    //
    // w16 is safe scratch at this call site - Mali hasn't assigned x16
    // here yet; first use is at 0x9e5070 (`mov x16, x13`).
    const uint32_t tramp3[9] = {
        0x52808010,   // mov   w16, #1024
        0xb50000cf,   // cbnz  x15, +24  (to .have)
        0xf86c79af,   // ldr   x15, [x13, x12, lsl #3]
        0x71000610,   // subs  w16, w16, #1
        0x54ffffa1,   // b.ne  -12  (back to cbnz)
        0x2a1f03e0,   // mov   w0, wzr
        0x17aa0838,   // b     0x9e5258
        0xf94005ee,   // ldr   x14, [x15, #8]
        0x17aa0837,   // b     0x9e505c
    };
    // Patch at 0x9e5058: b 0x1f62f60
    const uint32_t patch3 = 0x1455f7c2;

    uint32_t* crash1 = reinterpret_cast<uint32_t*>(base + OFF_CRASH1);
    uint32_t* crash2 = reinterpret_cast<uint32_t*>(base + OFF_CRASH2);
    uint32_t* crash3 = reinterpret_cast<uint32_t*>(base + OFF_CRASH3);
    uint32_t* cave1  = reinterpret_cast<uint32_t*>(base + OFF_CAVE1);
    uint32_t* cave2  = reinterpret_cast<uint32_t*>(base + OFF_CAVE2);
    uint32_t* cave3  = reinterpret_cast<uint32_t*>(base + OFF_CAVE3);

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
    // Cave sanity: make sure no one else already patched these slots.
    // Trampolines 1 & 2 are 5 insns each; trampoline 3 (retry loop) is 9.
    for (int i = 0; i < 5; ++i) {
        if (cave1[i] != 0) { LOGE("patch: cave1 dirty at +%d: 0x%08x", i, cave1[i]); g_mali_patched = true; return; }
        if (cave2[i] != 0) { LOGE("patch: cave2 dirty at +%d: 0x%08x", i, cave2[i]); g_mali_patched = true; return; }
    }
    for (int i = 0; i < 9; ++i) {
        if (cave3[i] != 0) { LOGE("patch: cave3 dirty at +%d: 0x%08x", i, cave3[i]); g_mali_patched = true; return; }
    }

    if (!write_patch(cave1, tramp1, sizeof(tramp1))) return;
    if (!write_patch(cave2, tramp2, sizeof(tramp2))) return;
    if (!write_patch(cave3, tramp3, sizeof(tramp3))) return;
    if (!write_patch(crash1, &patch1, sizeof(patch1))) return;
    if (!write_patch(crash2, &patch2, sizeof(patch2))) return;
    if (!write_patch(crash3, &patch3, sizeof(patch3))) return;

    g_mali_patched = true;
    LOGI("patch: Mali G52 null-check trampolines installed (3 sites)");
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
