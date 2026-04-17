// Mali G52 Vulkan ICD serialization shim.
//
// The ARM r25p0 Bifrost Vulkan driver (libGLES_mali.so on RK356x / Mali G52)
// has an internal check-then-use race in a hot vkCmd* helper at libGLES_mali
// offset 0x1e05a00: the function loads a sub-struct pointer from `[ctx+0x20]`
// without any lock, assuming single-threaded access. When two emulator
// threads (emu + GpuWorker) enter Vulkan concurrently, one thread nulls that
// field during internal cache eviction while the other is mid-helper, and
// the subsequent `[NULL+0x60]` load SIGSEGVs.
//
// This shim replaces the vendor HAL `.so` at /vendor/lib64/hw/vulkan.rk356x.so.
// It implements the Android hwvulkan HAL contract:
//   - Exports `HMI` (= HAL_MODULE_INFO_SYM) as a hw_module_t
//   - open() callback returns a hwvulkan_device_t whose CreateInstance and
//     GetInstanceProcAddr are ours
//   - GetInstanceProcAddr hands OUR wrappers for hot Vulkan entry points and
//     the real driver's function pointer for everything else
//   - Each wrapper acquires a global recursive mutex before calling into the
//     real driver
//
// Under this shim every concurrent entry into libGLES_mali for the wrapped
// API is serialized, preventing the TOCTOU at offset 0x1e05a38.

#include <dlfcn.h>
#include <pthread.h>
#include <string.h>
#include <stdlib.h>
#include <stdint.h>
#include <android/log.h>
#include <vulkan/vulkan.h>
#include <hardware/hardware.h>
#include <hardware/hwvulkan.h>

#define LOG_TAG "MaliVkShim"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO,  LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

namespace {

// Global serialization lock covering every wrapped Mali entry. Recursive so
// re-entry via internal callbacks cannot self-deadlock. Fairness doesn't
// matter — contention is low (Azahar enters from at most two threads).
pthread_mutex_t g_mutex;
pthread_once_t g_mutex_once = PTHREAD_ONCE_INIT;
static void mutex_init() {
    pthread_mutexattr_t attr;
    pthread_mutexattr_init(&attr);
    pthread_mutexattr_settype(&attr, PTHREAD_MUTEX_RECURSIVE);
    pthread_mutex_init(&g_mutex, &attr);
    pthread_mutexattr_destroy(&attr);
}
static inline void lock()   { pthread_once(&g_mutex_once, mutex_init); pthread_mutex_lock(&g_mutex); }
static inline void unlock() { pthread_mutex_unlock(&g_mutex); }

struct LockGuard {
    LockGuard()  { lock(); }
    ~LockGuard() { unlock(); }
};

// Real driver handle + HAL device obtained from libGLES_mali.so's HMI symbol.
void*                         g_real_lib    = nullptr;
hwvulkan_module_t*            g_real_module = nullptr;
hwvulkan_device_t*            g_real_device = nullptr;

// Real Vulkan entry points, resolved lazily from the real driver.
struct RealFns {
    PFN_vkGetInstanceProcAddr getInstanceProcAddr = nullptr;
    PFN_vkGetDeviceProcAddr   getDeviceProcAddr   = nullptr;

    // Queue / submission — THE hottest crashing family.
    PFN_vkQueueSubmit          queueSubmit          = nullptr;
    PFN_vkQueueWaitIdle        queueWaitIdle        = nullptr;
    PFN_vkDeviceWaitIdle       deviceWaitIdle       = nullptr;
    PFN_vkQueuePresentKHR      queuePresentKHR      = nullptr;

    // Descriptor set updates — explicitly named concurrent in driver crash log.
    PFN_vkUpdateDescriptorSets updateDescriptorSets = nullptr;
    PFN_vkAllocateDescriptorSets allocateDescriptorSets = nullptr;
    PFN_vkFreeDescriptorSets   freeDescriptorSets   = nullptr;
    PFN_vkResetDescriptorPool  resetDescriptorPool  = nullptr;

    // Command buffer lifecycle.
    PFN_vkBeginCommandBuffer   beginCommandBuffer   = nullptr;
    PFN_vkEndCommandBuffer     endCommandBuffer     = nullptr;
    PFN_vkResetCommandBuffer   resetCommandBuffer   = nullptr;
    PFN_vkAllocateCommandBuffers allocateCommandBuffers = nullptr;
    PFN_vkFreeCommandBuffers   freeCommandBuffers   = nullptr;
    PFN_vkResetCommandPool     resetCommandPool     = nullptr;

    // vkCmd* — the crashing function at Mali offset 0x9e6730 is one of these.
    PFN_vkCmdBindPipeline          cmdBindPipeline          = nullptr;
    PFN_vkCmdBindDescriptorSets    cmdBindDescriptorSets    = nullptr;
    PFN_vkCmdBindVertexBuffers     cmdBindVertexBuffers     = nullptr;
    PFN_vkCmdBindIndexBuffer       cmdBindIndexBuffer       = nullptr;
    PFN_vkCmdDraw                  cmdDraw                  = nullptr;
    PFN_vkCmdDrawIndexed           cmdDrawIndexed           = nullptr;
    PFN_vkCmdDrawIndirect          cmdDrawIndirect          = nullptr;
    PFN_vkCmdDrawIndexedIndirect   cmdDrawIndexedIndirect   = nullptr;
    PFN_vkCmdDispatch              cmdDispatch              = nullptr;
    PFN_vkCmdCopyBuffer            cmdCopyBuffer            = nullptr;
    PFN_vkCmdCopyImage             cmdCopyImage             = nullptr;
    PFN_vkCmdCopyBufferToImage     cmdCopyBufferToImage     = nullptr;
    PFN_vkCmdCopyImageToBuffer     cmdCopyImageToBuffer     = nullptr;
    PFN_vkCmdBlitImage             cmdBlitImage             = nullptr;
    PFN_vkCmdClearColorImage       cmdClearColorImage       = nullptr;
    PFN_vkCmdClearDepthStencilImage cmdClearDepthStencilImage = nullptr;
    PFN_vkCmdClearAttachments      cmdClearAttachments      = nullptr;
    PFN_vkCmdFillBuffer            cmdFillBuffer            = nullptr;
    PFN_vkCmdUpdateBuffer          cmdUpdateBuffer          = nullptr;
    PFN_vkCmdPipelineBarrier       cmdPipelineBarrier       = nullptr;
    PFN_vkCmdBeginRenderPass       cmdBeginRenderPass       = nullptr;
    PFN_vkCmdEndRenderPass         cmdEndRenderPass         = nullptr;
    PFN_vkCmdNextSubpass           cmdNextSubpass           = nullptr;
    PFN_vkCmdExecuteCommands       cmdExecuteCommands       = nullptr;
    PFN_vkCmdSetViewport           cmdSetViewport           = nullptr;
    PFN_vkCmdSetScissor            cmdSetScissor            = nullptr;
    PFN_vkCmdSetLineWidth          cmdSetLineWidth          = nullptr;
    PFN_vkCmdSetDepthBias          cmdSetDepthBias          = nullptr;
    PFN_vkCmdSetBlendConstants     cmdSetBlendConstants     = nullptr;
    PFN_vkCmdSetDepthBounds        cmdSetDepthBounds        = nullptr;
    PFN_vkCmdSetStencilCompareMask cmdSetStencilCompareMask = nullptr;
    PFN_vkCmdSetStencilWriteMask   cmdSetStencilWriteMask   = nullptr;
    PFN_vkCmdSetStencilReference   cmdSetStencilReference   = nullptr;
    PFN_vkCmdPushConstants         cmdPushConstants         = nullptr;
    PFN_vkCmdResolveImage          cmdResolveImage          = nullptr;
    PFN_vkCmdSetEvent              cmdSetEvent              = nullptr;
    PFN_vkCmdResetEvent            cmdResetEvent            = nullptr;
    PFN_vkCmdWaitEvents            cmdWaitEvents            = nullptr;
    PFN_vkCmdBeginQuery            cmdBeginQuery            = nullptr;
    PFN_vkCmdEndQuery              cmdEndQuery              = nullptr;
    PFN_vkCmdResetQueryPool        cmdResetQueryPool        = nullptr;
    PFN_vkCmdCopyQueryPoolResults  cmdCopyQueryPoolResults  = nullptr;
    PFN_vkCmdWriteTimestamp        cmdWriteTimestamp        = nullptr;
    // Dynamic rendering / KHR extensions used by modern Azahar:
    PFN_vkCmdBeginRenderingKHR     cmdBeginRenderingKHR     = nullptr;
    PFN_vkCmdEndRenderingKHR       cmdEndRenderingKHR       = nullptr;

    // Object create/destroy — GpuWorker uses these while VulkanWorker
    // is mid-command-buffer. Every Mali call that touches the shared driver
    // ctx needs the mutex.
    PFN_vkCreateBuffer             createBuffer             = nullptr;
    PFN_vkDestroyBuffer            destroyBuffer            = nullptr;
    PFN_vkCreateBufferView         createBufferView         = nullptr;
    PFN_vkDestroyBufferView        destroyBufferView        = nullptr;
    PFN_vkCreateImage              createImage              = nullptr;
    PFN_vkDestroyImage             destroyImage             = nullptr;
    PFN_vkCreateImageView          createImageView          = nullptr;
    PFN_vkDestroyImageView         destroyImageView         = nullptr;
    PFN_vkCreateSampler            createSampler            = nullptr;
    PFN_vkDestroySampler           destroySampler           = nullptr;
    PFN_vkCreateShaderModule       createShaderModule       = nullptr;
    PFN_vkDestroyShaderModule      destroyShaderModule      = nullptr;
    PFN_vkCreatePipelineLayout     createPipelineLayout     = nullptr;
    PFN_vkDestroyPipelineLayout    destroyPipelineLayout    = nullptr;
    PFN_vkCreatePipelineCache      createPipelineCache      = nullptr;
    PFN_vkDestroyPipelineCache     destroyPipelineCache     = nullptr;
    PFN_vkMergePipelineCaches      mergePipelineCaches      = nullptr;
    PFN_vkGetPipelineCacheData     getPipelineCacheData     = nullptr;
    PFN_vkCreateGraphicsPipelines  createGraphicsPipelines  = nullptr;
    PFN_vkCreateComputePipelines   createComputePipelines   = nullptr;
    PFN_vkDestroyPipeline          destroyPipeline          = nullptr;
    PFN_vkCreateDescriptorSetLayout createDescriptorSetLayout = nullptr;
    PFN_vkDestroyDescriptorSetLayout destroyDescriptorSetLayout = nullptr;
    PFN_vkCreateDescriptorPool     createDescriptorPool     = nullptr;
    PFN_vkDestroyDescriptorPool    destroyDescriptorPool    = nullptr;
    PFN_vkCreateRenderPass         createRenderPass         = nullptr;
    PFN_vkCreateRenderPass2        createRenderPass2        = nullptr;
    PFN_vkDestroyRenderPass        destroyRenderPass        = nullptr;
    PFN_vkCreateFramebuffer        createFramebuffer        = nullptr;
    PFN_vkDestroyFramebuffer       destroyFramebuffer       = nullptr;
    PFN_vkCreateCommandPool        createCommandPool        = nullptr;
    PFN_vkDestroyCommandPool       destroyCommandPool       = nullptr;
    PFN_vkTrimCommandPool          trimCommandPool          = nullptr;
    PFN_vkCreateFence              createFence              = nullptr;
    PFN_vkDestroyFence             destroyFence             = nullptr;
    PFN_vkCreateSemaphore          createSemaphore          = nullptr;
    PFN_vkDestroySemaphore         destroySemaphore         = nullptr;
    PFN_vkCreateEvent              createEvent              = nullptr;
    PFN_vkDestroyEvent             destroyEvent             = nullptr;
    PFN_vkCreateQueryPool          createQueryPool          = nullptr;
    PFN_vkDestroyQueryPool         destroyQueryPool         = nullptr;
    PFN_vkCreateSwapchainKHR       createSwapchainKHR       = nullptr;
    PFN_vkDestroySwapchainKHR      destroySwapchainKHR      = nullptr;
    PFN_vkGetSwapchainImagesKHR    getSwapchainImagesKHR    = nullptr;
    PFN_vkAcquireNextImageKHR      acquireNextImageKHR      = nullptr;

    // Memory management.
    PFN_vkAllocateMemory           allocateMemory           = nullptr;
    PFN_vkFreeMemory               freeMemory               = nullptr;
    PFN_vkMapMemory                mapMemory                = nullptr;
    PFN_vkUnmapMemory              unmapMemory              = nullptr;
    PFN_vkFlushMappedMemoryRanges  flushMappedMemoryRanges  = nullptr;
    PFN_vkInvalidateMappedMemoryRanges invalidateMappedMemoryRanges = nullptr;
    PFN_vkBindBufferMemory         bindBufferMemory         = nullptr;
    PFN_vkBindImageMemory          bindImageMemory          = nullptr;
    PFN_vkBindBufferMemory2        bindBufferMemory2        = nullptr;
    PFN_vkBindImageMemory2         bindImageMemory2         = nullptr;
    PFN_vkGetBufferMemoryRequirements getBufferMemoryRequirements = nullptr;
    PFN_vkGetImageMemoryRequirements  getImageMemoryRequirements  = nullptr;
    PFN_vkGetBufferMemoryRequirements2 getBufferMemoryRequirements2 = nullptr;
    PFN_vkGetImageMemoryRequirements2  getImageMemoryRequirements2  = nullptr;

    // Fence / event / query operations (touch driver state).
    PFN_vkWaitForFences            waitForFences            = nullptr;
    PFN_vkResetFences              resetFences              = nullptr;
    PFN_vkGetFenceStatus           getFenceStatus           = nullptr;
    PFN_vkGetEventStatus           getEventStatus           = nullptr;
    PFN_vkSetEvent                 setEvent                 = nullptr;
    PFN_vkResetEvent               resetEvent               = nullptr;
    PFN_vkGetQueryPoolResults      getQueryPoolResults      = nullptr;
};
RealFns g_real;

// ===================== Wrappers =====================
// Each one takes the mutex, calls the real driver, releases the mutex.
// Kept as a single source of truth — macro to reduce boilerplate.

#define WRAP_VOID(NAME, SIG, ARGS)                              \
    VKAPI_ATTR void VKAPI_CALL w_##NAME SIG {                   \
        LockGuard _g;                                           \
        g_real.NAME ARGS;                                       \
    }
#define WRAP_RESULT(NAME, SIG, ARGS)                            \
    VKAPI_ATTR VkResult VKAPI_CALL w_##NAME SIG {               \
        LockGuard _g;                                           \
        return g_real.NAME ARGS;                                \
    }

WRAP_RESULT(queueSubmit,
    (VkQueue q, uint32_t n, const VkSubmitInfo* p, VkFence f),
    (q, n, p, f))
WRAP_RESULT(queueWaitIdle, (VkQueue q), (q))
WRAP_RESULT(deviceWaitIdle, (VkDevice d), (d))
WRAP_RESULT(queuePresentKHR, (VkQueue q, const VkPresentInfoKHR* p), (q, p))

WRAP_VOID(updateDescriptorSets,
    (VkDevice d, uint32_t wc, const VkWriteDescriptorSet* w, uint32_t cc, const VkCopyDescriptorSet* c),
    (d, wc, w, cc, c))
WRAP_RESULT(allocateDescriptorSets,
    (VkDevice d, const VkDescriptorSetAllocateInfo* a, VkDescriptorSet* s),
    (d, a, s))
WRAP_RESULT(freeDescriptorSets,
    (VkDevice d, VkDescriptorPool p, uint32_t c, const VkDescriptorSet* s),
    (d, p, c, s))
WRAP_RESULT(resetDescriptorPool,
    (VkDevice d, VkDescriptorPool p, VkDescriptorPoolResetFlags f),
    (d, p, f))

WRAP_RESULT(beginCommandBuffer, (VkCommandBuffer c, const VkCommandBufferBeginInfo* b), (c, b))
WRAP_RESULT(endCommandBuffer, (VkCommandBuffer c), (c))
WRAP_RESULT(resetCommandBuffer, (VkCommandBuffer c, VkCommandBufferResetFlags f), (c, f))
WRAP_RESULT(allocateCommandBuffers,
    (VkDevice d, const VkCommandBufferAllocateInfo* a, VkCommandBuffer* b),
    (d, a, b))
WRAP_VOID(freeCommandBuffers,
    (VkDevice d, VkCommandPool p, uint32_t c, const VkCommandBuffer* b),
    (d, p, c, b))
WRAP_RESULT(resetCommandPool, (VkDevice d, VkCommandPool p, VkCommandPoolResetFlags f), (d, p, f))

WRAP_VOID(cmdBindPipeline,
    (VkCommandBuffer c, VkPipelineBindPoint b, VkPipeline p),
    (c, b, p))
WRAP_VOID(cmdBindDescriptorSets,
    (VkCommandBuffer c, VkPipelineBindPoint b, VkPipelineLayout l, uint32_t f, uint32_t n, const VkDescriptorSet* s, uint32_t doc, const uint32_t* dod),
    (c, b, l, f, n, s, doc, dod))
WRAP_VOID(cmdBindVertexBuffers,
    (VkCommandBuffer c, uint32_t fb, uint32_t bc, const VkBuffer* b, const VkDeviceSize* o),
    (c, fb, bc, b, o))
WRAP_VOID(cmdBindIndexBuffer,
    (VkCommandBuffer c, VkBuffer b, VkDeviceSize o, VkIndexType t),
    (c, b, o, t))
WRAP_VOID(cmdDraw,
    (VkCommandBuffer c, uint32_t vc, uint32_t ic, uint32_t fv, uint32_t fi),
    (c, vc, ic, fv, fi))
WRAP_VOID(cmdDrawIndexed,
    (VkCommandBuffer c, uint32_t ic, uint32_t inC, uint32_t fi, int32_t vo, uint32_t fins),
    (c, ic, inC, fi, vo, fins))
WRAP_VOID(cmdDrawIndirect,
    (VkCommandBuffer c, VkBuffer b, VkDeviceSize o, uint32_t dc, uint32_t s),
    (c, b, o, dc, s))
WRAP_VOID(cmdDrawIndexedIndirect,
    (VkCommandBuffer c, VkBuffer b, VkDeviceSize o, uint32_t dc, uint32_t s),
    (c, b, o, dc, s))
WRAP_VOID(cmdDispatch,
    (VkCommandBuffer c, uint32_t x, uint32_t y, uint32_t z),
    (c, x, y, z))
WRAP_VOID(cmdCopyBuffer,
    (VkCommandBuffer c, VkBuffer s, VkBuffer d, uint32_t rc, const VkBufferCopy* r),
    (c, s, d, rc, r))
WRAP_VOID(cmdCopyImage,
    (VkCommandBuffer c, VkImage si, VkImageLayout sl, VkImage di, VkImageLayout dl, uint32_t rc, const VkImageCopy* r),
    (c, si, sl, di, dl, rc, r))
WRAP_VOID(cmdCopyBufferToImage,
    (VkCommandBuffer c, VkBuffer s, VkImage di, VkImageLayout dl, uint32_t rc, const VkBufferImageCopy* r),
    (c, s, di, dl, rc, r))
WRAP_VOID(cmdCopyImageToBuffer,
    (VkCommandBuffer c, VkImage si, VkImageLayout sl, VkBuffer d, uint32_t rc, const VkBufferImageCopy* r),
    (c, si, sl, d, rc, r))
WRAP_VOID(cmdBlitImage,
    (VkCommandBuffer c, VkImage si, VkImageLayout sl, VkImage di, VkImageLayout dl, uint32_t rc, const VkImageBlit* r, VkFilter f),
    (c, si, sl, di, dl, rc, r, f))
WRAP_VOID(cmdClearColorImage,
    (VkCommandBuffer c, VkImage i, VkImageLayout l, const VkClearColorValue* cv, uint32_t rc, const VkImageSubresourceRange* r),
    (c, i, l, cv, rc, r))
WRAP_VOID(cmdClearDepthStencilImage,
    (VkCommandBuffer c, VkImage i, VkImageLayout l, const VkClearDepthStencilValue* dv, uint32_t rc, const VkImageSubresourceRange* r),
    (c, i, l, dv, rc, r))
WRAP_VOID(cmdClearAttachments,
    (VkCommandBuffer c, uint32_t ac, const VkClearAttachment* a, uint32_t rc, const VkClearRect* r),
    (c, ac, a, rc, r))
WRAP_VOID(cmdFillBuffer,
    (VkCommandBuffer c, VkBuffer b, VkDeviceSize o, VkDeviceSize s, uint32_t data),
    (c, b, o, s, data))
WRAP_VOID(cmdUpdateBuffer,
    (VkCommandBuffer c, VkBuffer b, VkDeviceSize o, VkDeviceSize s, const void* d),
    (c, b, o, s, d))
WRAP_VOID(cmdPipelineBarrier,
    (VkCommandBuffer c, VkPipelineStageFlags ss, VkPipelineStageFlags ds, VkDependencyFlags df, uint32_t mbc, const VkMemoryBarrier* mb, uint32_t bbc, const VkBufferMemoryBarrier* bb, uint32_t ibc, const VkImageMemoryBarrier* ib),
    (c, ss, ds, df, mbc, mb, bbc, bb, ibc, ib))
WRAP_VOID(cmdBeginRenderPass,
    (VkCommandBuffer c, const VkRenderPassBeginInfo* b, VkSubpassContents s),
    (c, b, s))
WRAP_VOID(cmdEndRenderPass, (VkCommandBuffer c), (c))
WRAP_VOID(cmdNextSubpass, (VkCommandBuffer c, VkSubpassContents s), (c, s))
WRAP_VOID(cmdExecuteCommands,
    (VkCommandBuffer c, uint32_t n, const VkCommandBuffer* b), (c, n, b))
WRAP_VOID(cmdSetViewport,
    (VkCommandBuffer c, uint32_t fv, uint32_t vc, const VkViewport* v),
    (c, fv, vc, v))
WRAP_VOID(cmdSetScissor,
    (VkCommandBuffer c, uint32_t fs, uint32_t sc, const VkRect2D* s),
    (c, fs, sc, s))
WRAP_VOID(cmdSetLineWidth, (VkCommandBuffer c, float w), (c, w))
WRAP_VOID(cmdSetDepthBias,
    (VkCommandBuffer c, float a, float b, float d), (c, a, b, d))
WRAP_VOID(cmdSetBlendConstants,
    (VkCommandBuffer c, const float b[4]), (c, b))
WRAP_VOID(cmdSetDepthBounds, (VkCommandBuffer c, float mn, float mx), (c, mn, mx))
WRAP_VOID(cmdSetStencilCompareMask,
    (VkCommandBuffer c, VkStencilFaceFlags f, uint32_t m), (c, f, m))
WRAP_VOID(cmdSetStencilWriteMask,
    (VkCommandBuffer c, VkStencilFaceFlags f, uint32_t m), (c, f, m))
WRAP_VOID(cmdSetStencilReference,
    (VkCommandBuffer c, VkStencilFaceFlags f, uint32_t r), (c, f, r))
WRAP_VOID(cmdPushConstants,
    (VkCommandBuffer c, VkPipelineLayout l, VkShaderStageFlags s, uint32_t o, uint32_t sz, const void* v),
    (c, l, s, o, sz, v))
WRAP_VOID(cmdResolveImage,
    (VkCommandBuffer c, VkImage si, VkImageLayout sl, VkImage di, VkImageLayout dl, uint32_t rc, const VkImageResolve* r),
    (c, si, sl, di, dl, rc, r))
WRAP_VOID(cmdSetEvent,
    (VkCommandBuffer c, VkEvent e, VkPipelineStageFlags s), (c, e, s))
WRAP_VOID(cmdResetEvent,
    (VkCommandBuffer c, VkEvent e, VkPipelineStageFlags s), (c, e, s))
WRAP_VOID(cmdWaitEvents,
    (VkCommandBuffer c, uint32_t ec, const VkEvent* e, VkPipelineStageFlags ss, VkPipelineStageFlags ds, uint32_t mbc, const VkMemoryBarrier* mb, uint32_t bbc, const VkBufferMemoryBarrier* bb, uint32_t ibc, const VkImageMemoryBarrier* ib),
    (c, ec, e, ss, ds, mbc, mb, bbc, bb, ibc, ib))
WRAP_VOID(cmdBeginQuery,
    (VkCommandBuffer c, VkQueryPool p, uint32_t q, VkQueryControlFlags f), (c, p, q, f))
WRAP_VOID(cmdEndQuery, (VkCommandBuffer c, VkQueryPool p, uint32_t q), (c, p, q))
WRAP_VOID(cmdResetQueryPool,
    (VkCommandBuffer c, VkQueryPool p, uint32_t fq, uint32_t qc), (c, p, fq, qc))
WRAP_VOID(cmdCopyQueryPoolResults,
    (VkCommandBuffer c, VkQueryPool p, uint32_t fq, uint32_t qc, VkBuffer b, VkDeviceSize o, VkDeviceSize st, VkQueryResultFlags f),
    (c, p, fq, qc, b, o, st, f))
WRAP_VOID(cmdWriteTimestamp,
    (VkCommandBuffer c, VkPipelineStageFlagBits s, VkQueryPool p, uint32_t q),
    (c, s, p, q))
WRAP_VOID(cmdBeginRenderingKHR,
    (VkCommandBuffer c, const VkRenderingInfo* i), (c, i))
WRAP_VOID(cmdEndRenderingKHR, (VkCommandBuffer c), (c))

// Create/destroy/bind/memory — called by the GpuWorker thread concurrent
// with VulkanWorker's vkCmd* execution. Without these wraps, Mali's internal
// object tables mutate mid-read and trigger null-derefs at unrelated PCs.
WRAP_RESULT(createBuffer, (VkDevice d, const VkBufferCreateInfo* ci, const VkAllocationCallbacks* a, VkBuffer* b), (d, ci, a, b))
WRAP_VOID(destroyBuffer, (VkDevice d, VkBuffer b, const VkAllocationCallbacks* a), (d, b, a))
WRAP_RESULT(createBufferView, (VkDevice d, const VkBufferViewCreateInfo* ci, const VkAllocationCallbacks* a, VkBufferView* v), (d, ci, a, v))
WRAP_VOID(destroyBufferView, (VkDevice d, VkBufferView v, const VkAllocationCallbacks* a), (d, v, a))
WRAP_RESULT(createImage, (VkDevice d, const VkImageCreateInfo* ci, const VkAllocationCallbacks* a, VkImage* i), (d, ci, a, i))
WRAP_VOID(destroyImage, (VkDevice d, VkImage i, const VkAllocationCallbacks* a), (d, i, a))
WRAP_RESULT(createImageView, (VkDevice d, const VkImageViewCreateInfo* ci, const VkAllocationCallbacks* a, VkImageView* v), (d, ci, a, v))
WRAP_VOID(destroyImageView, (VkDevice d, VkImageView v, const VkAllocationCallbacks* a), (d, v, a))
WRAP_RESULT(createSampler, (VkDevice d, const VkSamplerCreateInfo* ci, const VkAllocationCallbacks* a, VkSampler* s), (d, ci, a, s))
WRAP_VOID(destroySampler, (VkDevice d, VkSampler s, const VkAllocationCallbacks* a), (d, s, a))
WRAP_RESULT(createShaderModule, (VkDevice d, const VkShaderModuleCreateInfo* ci, const VkAllocationCallbacks* a, VkShaderModule* m), (d, ci, a, m))
WRAP_VOID(destroyShaderModule, (VkDevice d, VkShaderModule m, const VkAllocationCallbacks* a), (d, m, a))
WRAP_RESULT(createPipelineLayout, (VkDevice d, const VkPipelineLayoutCreateInfo* ci, const VkAllocationCallbacks* a, VkPipelineLayout* p), (d, ci, a, p))
WRAP_VOID(destroyPipelineLayout, (VkDevice d, VkPipelineLayout p, const VkAllocationCallbacks* a), (d, p, a))
WRAP_RESULT(createPipelineCache, (VkDevice d, const VkPipelineCacheCreateInfo* ci, const VkAllocationCallbacks* a, VkPipelineCache* p), (d, ci, a, p))
WRAP_VOID(destroyPipelineCache, (VkDevice d, VkPipelineCache p, const VkAllocationCallbacks* a), (d, p, a))
WRAP_RESULT(mergePipelineCaches, (VkDevice d, VkPipelineCache dst, uint32_t n, const VkPipelineCache* src), (d, dst, n, src))
WRAP_RESULT(getPipelineCacheData, (VkDevice d, VkPipelineCache p, size_t* sz, void* dp), (d, p, sz, dp))
WRAP_RESULT(createGraphicsPipelines, (VkDevice d, VkPipelineCache pc, uint32_t n, const VkGraphicsPipelineCreateInfo* ci, const VkAllocationCallbacks* a, VkPipeline* p), (d, pc, n, ci, a, p))
WRAP_RESULT(createComputePipelines, (VkDevice d, VkPipelineCache pc, uint32_t n, const VkComputePipelineCreateInfo* ci, const VkAllocationCallbacks* a, VkPipeline* p), (d, pc, n, ci, a, p))
WRAP_VOID(destroyPipeline, (VkDevice d, VkPipeline p, const VkAllocationCallbacks* a), (d, p, a))
WRAP_RESULT(createDescriptorSetLayout, (VkDevice d, const VkDescriptorSetLayoutCreateInfo* ci, const VkAllocationCallbacks* a, VkDescriptorSetLayout* dsl), (d, ci, a, dsl))
WRAP_VOID(destroyDescriptorSetLayout, (VkDevice d, VkDescriptorSetLayout dsl, const VkAllocationCallbacks* a), (d, dsl, a))
WRAP_RESULT(createDescriptorPool, (VkDevice d, const VkDescriptorPoolCreateInfo* ci, const VkAllocationCallbacks* a, VkDescriptorPool* p), (d, ci, a, p))
WRAP_VOID(destroyDescriptorPool, (VkDevice d, VkDescriptorPool p, const VkAllocationCallbacks* a), (d, p, a))
WRAP_RESULT(createRenderPass, (VkDevice d, const VkRenderPassCreateInfo* ci, const VkAllocationCallbacks* a, VkRenderPass* r), (d, ci, a, r))
WRAP_RESULT(createRenderPass2, (VkDevice d, const VkRenderPassCreateInfo2* ci, const VkAllocationCallbacks* a, VkRenderPass* r), (d, ci, a, r))
WRAP_VOID(destroyRenderPass, (VkDevice d, VkRenderPass r, const VkAllocationCallbacks* a), (d, r, a))
WRAP_RESULT(createFramebuffer, (VkDevice d, const VkFramebufferCreateInfo* ci, const VkAllocationCallbacks* a, VkFramebuffer* f), (d, ci, a, f))
WRAP_VOID(destroyFramebuffer, (VkDevice d, VkFramebuffer f, const VkAllocationCallbacks* a), (d, f, a))
WRAP_RESULT(createCommandPool, (VkDevice d, const VkCommandPoolCreateInfo* ci, const VkAllocationCallbacks* a, VkCommandPool* p), (d, ci, a, p))
WRAP_VOID(destroyCommandPool, (VkDevice d, VkCommandPool p, const VkAllocationCallbacks* a), (d, p, a))
WRAP_VOID(trimCommandPool, (VkDevice d, VkCommandPool p, VkCommandPoolTrimFlags f), (d, p, f))
WRAP_RESULT(createFence, (VkDevice d, const VkFenceCreateInfo* ci, const VkAllocationCallbacks* a, VkFence* f), (d, ci, a, f))
WRAP_VOID(destroyFence, (VkDevice d, VkFence f, const VkAllocationCallbacks* a), (d, f, a))
WRAP_RESULT(createSemaphore, (VkDevice d, const VkSemaphoreCreateInfo* ci, const VkAllocationCallbacks* a, VkSemaphore* s), (d, ci, a, s))
WRAP_VOID(destroySemaphore, (VkDevice d, VkSemaphore s, const VkAllocationCallbacks* a), (d, s, a))
WRAP_RESULT(createEvent, (VkDevice d, const VkEventCreateInfo* ci, const VkAllocationCallbacks* a, VkEvent* e), (d, ci, a, e))
WRAP_VOID(destroyEvent, (VkDevice d, VkEvent e, const VkAllocationCallbacks* a), (d, e, a))
WRAP_RESULT(createQueryPool, (VkDevice d, const VkQueryPoolCreateInfo* ci, const VkAllocationCallbacks* a, VkQueryPool* q), (d, ci, a, q))
WRAP_VOID(destroyQueryPool, (VkDevice d, VkQueryPool q, const VkAllocationCallbacks* a), (d, q, a))
WRAP_RESULT(createSwapchainKHR, (VkDevice d, const VkSwapchainCreateInfoKHR* ci, const VkAllocationCallbacks* a, VkSwapchainKHR* s), (d, ci, a, s))
WRAP_VOID(destroySwapchainKHR, (VkDevice d, VkSwapchainKHR s, const VkAllocationCallbacks* a), (d, s, a))
WRAP_RESULT(getSwapchainImagesKHR, (VkDevice d, VkSwapchainKHR s, uint32_t* n, VkImage* img), (d, s, n, img))
WRAP_RESULT(acquireNextImageKHR, (VkDevice d, VkSwapchainKHR s, uint64_t t, VkSemaphore sm, VkFence f, uint32_t* i), (d, s, t, sm, f, i))

WRAP_RESULT(allocateMemory, (VkDevice d, const VkMemoryAllocateInfo* ai, const VkAllocationCallbacks* a, VkDeviceMemory* m), (d, ai, a, m))
WRAP_VOID(freeMemory, (VkDevice d, VkDeviceMemory m, const VkAllocationCallbacks* a), (d, m, a))
WRAP_RESULT(mapMemory, (VkDevice d, VkDeviceMemory m, VkDeviceSize o, VkDeviceSize sz, VkMemoryMapFlags f, void** pp), (d, m, o, sz, f, pp))
WRAP_VOID(unmapMemory, (VkDevice d, VkDeviceMemory m), (d, m))
WRAP_RESULT(flushMappedMemoryRanges, (VkDevice d, uint32_t n, const VkMappedMemoryRange* r), (d, n, r))
WRAP_RESULT(invalidateMappedMemoryRanges, (VkDevice d, uint32_t n, const VkMappedMemoryRange* r), (d, n, r))
WRAP_RESULT(bindBufferMemory, (VkDevice d, VkBuffer b, VkDeviceMemory m, VkDeviceSize o), (d, b, m, o))
WRAP_RESULT(bindImageMemory, (VkDevice d, VkImage i, VkDeviceMemory m, VkDeviceSize o), (d, i, m, o))
WRAP_RESULT(bindBufferMemory2, (VkDevice d, uint32_t n, const VkBindBufferMemoryInfo* bi), (d, n, bi))
WRAP_RESULT(bindImageMemory2, (VkDevice d, uint32_t n, const VkBindImageMemoryInfo* bi), (d, n, bi))
WRAP_VOID(getBufferMemoryRequirements, (VkDevice d, VkBuffer b, VkMemoryRequirements* r), (d, b, r))
WRAP_VOID(getImageMemoryRequirements, (VkDevice d, VkImage i, VkMemoryRequirements* r), (d, i, r))
WRAP_VOID(getBufferMemoryRequirements2, (VkDevice d, const VkBufferMemoryRequirementsInfo2* info, VkMemoryRequirements2* r), (d, info, r))
WRAP_VOID(getImageMemoryRequirements2, (VkDevice d, const VkImageMemoryRequirementsInfo2* info, VkMemoryRequirements2* r), (d, info, r))

WRAP_RESULT(waitForFences, (VkDevice d, uint32_t n, const VkFence* f, VkBool32 all, uint64_t t), (d, n, f, all, t))
WRAP_RESULT(resetFences, (VkDevice d, uint32_t n, const VkFence* f), (d, n, f))
WRAP_RESULT(getFenceStatus, (VkDevice d, VkFence f), (d, f))
WRAP_RESULT(getEventStatus, (VkDevice d, VkEvent e), (d, e))
WRAP_RESULT(setEvent, (VkDevice d, VkEvent e), (d, e))
WRAP_RESULT(resetEvent, (VkDevice d, VkEvent e), (d, e))
WRAP_RESULT(getQueryPoolResults, (VkDevice d, VkQueryPool p, uint32_t fq, uint32_t qc, size_t sz, void* dp, VkDeviceSize st, VkQueryResultFlags f), (d, p, fq, qc, sz, dp, st, f))

#undef WRAP_VOID
#undef WRAP_RESULT

// Table of functions we intercept. Pointer-to-pointer so we can set the
// real fn by a single lookup, and the wrapper address is known at compile.
struct Intercept {
    const char* name;
    PFN_vkVoidFunction* real_slot;
    PFN_vkVoidFunction  wrap;
};

#define ENTRY(VkName, slot, wrap_name) \
    { #VkName, (PFN_vkVoidFunction*)&g_real.slot, (PFN_vkVoidFunction)wrap_name }

const Intercept g_intercepts[] = {
    ENTRY(vkQueueSubmit,          queueSubmit,          w_queueSubmit),
    ENTRY(vkQueueWaitIdle,        queueWaitIdle,        w_queueWaitIdle),
    ENTRY(vkDeviceWaitIdle,       deviceWaitIdle,       w_deviceWaitIdle),
    ENTRY(vkQueuePresentKHR,      queuePresentKHR,      w_queuePresentKHR),
    ENTRY(vkUpdateDescriptorSets, updateDescriptorSets, w_updateDescriptorSets),
    ENTRY(vkAllocateDescriptorSets, allocateDescriptorSets, w_allocateDescriptorSets),
    ENTRY(vkFreeDescriptorSets,   freeDescriptorSets,   w_freeDescriptorSets),
    ENTRY(vkResetDescriptorPool,  resetDescriptorPool,  w_resetDescriptorPool),
    ENTRY(vkBeginCommandBuffer,   beginCommandBuffer,   w_beginCommandBuffer),
    ENTRY(vkEndCommandBuffer,     endCommandBuffer,     w_endCommandBuffer),
    ENTRY(vkResetCommandBuffer,   resetCommandBuffer,   w_resetCommandBuffer),
    ENTRY(vkAllocateCommandBuffers, allocateCommandBuffers, w_allocateCommandBuffers),
    ENTRY(vkFreeCommandBuffers,   freeCommandBuffers,   w_freeCommandBuffers),
    ENTRY(vkResetCommandPool,     resetCommandPool,     w_resetCommandPool),
    ENTRY(vkCmdBindPipeline,          cmdBindPipeline,          w_cmdBindPipeline),
    ENTRY(vkCmdBindDescriptorSets,    cmdBindDescriptorSets,    w_cmdBindDescriptorSets),
    ENTRY(vkCmdBindVertexBuffers,     cmdBindVertexBuffers,     w_cmdBindVertexBuffers),
    ENTRY(vkCmdBindIndexBuffer,       cmdBindIndexBuffer,       w_cmdBindIndexBuffer),
    ENTRY(vkCmdDraw,                  cmdDraw,                  w_cmdDraw),
    ENTRY(vkCmdDrawIndexed,           cmdDrawIndexed,           w_cmdDrawIndexed),
    ENTRY(vkCmdDrawIndirect,          cmdDrawIndirect,          w_cmdDrawIndirect),
    ENTRY(vkCmdDrawIndexedIndirect,   cmdDrawIndexedIndirect,   w_cmdDrawIndexedIndirect),
    ENTRY(vkCmdDispatch,              cmdDispatch,              w_cmdDispatch),
    ENTRY(vkCmdCopyBuffer,            cmdCopyBuffer,            w_cmdCopyBuffer),
    ENTRY(vkCmdCopyImage,             cmdCopyImage,             w_cmdCopyImage),
    ENTRY(vkCmdCopyBufferToImage,     cmdCopyBufferToImage,     w_cmdCopyBufferToImage),
    ENTRY(vkCmdCopyImageToBuffer,     cmdCopyImageToBuffer,     w_cmdCopyImageToBuffer),
    ENTRY(vkCmdBlitImage,             cmdBlitImage,             w_cmdBlitImage),
    ENTRY(vkCmdClearColorImage,       cmdClearColorImage,       w_cmdClearColorImage),
    ENTRY(vkCmdClearDepthStencilImage,cmdClearDepthStencilImage,w_cmdClearDepthStencilImage),
    ENTRY(vkCmdClearAttachments,      cmdClearAttachments,      w_cmdClearAttachments),
    ENTRY(vkCmdFillBuffer,            cmdFillBuffer,            w_cmdFillBuffer),
    ENTRY(vkCmdUpdateBuffer,          cmdUpdateBuffer,          w_cmdUpdateBuffer),
    ENTRY(vkCmdPipelineBarrier,       cmdPipelineBarrier,       w_cmdPipelineBarrier),
    ENTRY(vkCmdBeginRenderPass,       cmdBeginRenderPass,       w_cmdBeginRenderPass),
    ENTRY(vkCmdEndRenderPass,         cmdEndRenderPass,         w_cmdEndRenderPass),
    ENTRY(vkCmdNextSubpass,           cmdNextSubpass,           w_cmdNextSubpass),
    ENTRY(vkCmdExecuteCommands,       cmdExecuteCommands,       w_cmdExecuteCommands),
    ENTRY(vkCmdSetViewport,           cmdSetViewport,           w_cmdSetViewport),
    ENTRY(vkCmdSetScissor,            cmdSetScissor,            w_cmdSetScissor),
    ENTRY(vkCmdSetLineWidth,          cmdSetLineWidth,          w_cmdSetLineWidth),
    ENTRY(vkCmdSetDepthBias,          cmdSetDepthBias,          w_cmdSetDepthBias),
    ENTRY(vkCmdSetBlendConstants,     cmdSetBlendConstants,     w_cmdSetBlendConstants),
    ENTRY(vkCmdSetDepthBounds,        cmdSetDepthBounds,        w_cmdSetDepthBounds),
    ENTRY(vkCmdSetStencilCompareMask, cmdSetStencilCompareMask, w_cmdSetStencilCompareMask),
    ENTRY(vkCmdSetStencilWriteMask,   cmdSetStencilWriteMask,   w_cmdSetStencilWriteMask),
    ENTRY(vkCmdSetStencilReference,   cmdSetStencilReference,   w_cmdSetStencilReference),
    ENTRY(vkCmdPushConstants,         cmdPushConstants,         w_cmdPushConstants),
    ENTRY(vkCmdResolveImage,          cmdResolveImage,          w_cmdResolveImage),
    ENTRY(vkCmdSetEvent,              cmdSetEvent,              w_cmdSetEvent),
    ENTRY(vkCmdResetEvent,            cmdResetEvent,            w_cmdResetEvent),
    ENTRY(vkCmdWaitEvents,            cmdWaitEvents,            w_cmdWaitEvents),
    ENTRY(vkCmdBeginQuery,            cmdBeginQuery,            w_cmdBeginQuery),
    ENTRY(vkCmdEndQuery,              cmdEndQuery,              w_cmdEndQuery),
    ENTRY(vkCmdResetQueryPool,        cmdResetQueryPool,        w_cmdResetQueryPool),
    ENTRY(vkCmdCopyQueryPoolResults,  cmdCopyQueryPoolResults,  w_cmdCopyQueryPoolResults),
    ENTRY(vkCmdWriteTimestamp,        cmdWriteTimestamp,        w_cmdWriteTimestamp),
    ENTRY(vkCmdBeginRenderingKHR,     cmdBeginRenderingKHR,     w_cmdBeginRenderingKHR),
    ENTRY(vkCmdEndRenderingKHR,       cmdEndRenderingKHR,       w_cmdEndRenderingKHR),

    ENTRY(vkCreateBuffer,            createBuffer,            w_createBuffer),
    ENTRY(vkDestroyBuffer,           destroyBuffer,           w_destroyBuffer),
    ENTRY(vkCreateBufferView,        createBufferView,        w_createBufferView),
    ENTRY(vkDestroyBufferView,       destroyBufferView,       w_destroyBufferView),
    ENTRY(vkCreateImage,             createImage,             w_createImage),
    ENTRY(vkDestroyImage,            destroyImage,            w_destroyImage),
    ENTRY(vkCreateImageView,         createImageView,         w_createImageView),
    ENTRY(vkDestroyImageView,        destroyImageView,        w_destroyImageView),
    ENTRY(vkCreateSampler,           createSampler,           w_createSampler),
    ENTRY(vkDestroySampler,          destroySampler,          w_destroySampler),
    ENTRY(vkCreateShaderModule,      createShaderModule,      w_createShaderModule),
    ENTRY(vkDestroyShaderModule,     destroyShaderModule,     w_destroyShaderModule),
    ENTRY(vkCreatePipelineLayout,    createPipelineLayout,    w_createPipelineLayout),
    ENTRY(vkDestroyPipelineLayout,   destroyPipelineLayout,   w_destroyPipelineLayout),
    ENTRY(vkCreatePipelineCache,     createPipelineCache,     w_createPipelineCache),
    ENTRY(vkDestroyPipelineCache,    destroyPipelineCache,    w_destroyPipelineCache),
    ENTRY(vkMergePipelineCaches,     mergePipelineCaches,     w_mergePipelineCaches),
    ENTRY(vkGetPipelineCacheData,    getPipelineCacheData,    w_getPipelineCacheData),
    ENTRY(vkCreateGraphicsPipelines, createGraphicsPipelines, w_createGraphicsPipelines),
    ENTRY(vkCreateComputePipelines,  createComputePipelines,  w_createComputePipelines),
    ENTRY(vkDestroyPipeline,         destroyPipeline,         w_destroyPipeline),
    ENTRY(vkCreateDescriptorSetLayout, createDescriptorSetLayout, w_createDescriptorSetLayout),
    ENTRY(vkDestroyDescriptorSetLayout, destroyDescriptorSetLayout, w_destroyDescriptorSetLayout),
    ENTRY(vkCreateDescriptorPool,    createDescriptorPool,    w_createDescriptorPool),
    ENTRY(vkDestroyDescriptorPool,   destroyDescriptorPool,   w_destroyDescriptorPool),
    ENTRY(vkCreateRenderPass,        createRenderPass,        w_createRenderPass),
    ENTRY(vkCreateRenderPass2,       createRenderPass2,       w_createRenderPass2),
    ENTRY(vkDestroyRenderPass,       destroyRenderPass,       w_destroyRenderPass),
    ENTRY(vkCreateFramebuffer,       createFramebuffer,       w_createFramebuffer),
    ENTRY(vkDestroyFramebuffer,      destroyFramebuffer,      w_destroyFramebuffer),
    ENTRY(vkCreateCommandPool,       createCommandPool,       w_createCommandPool),
    ENTRY(vkDestroyCommandPool,      destroyCommandPool,      w_destroyCommandPool),
    ENTRY(vkTrimCommandPool,         trimCommandPool,         w_trimCommandPool),
    ENTRY(vkCreateFence,             createFence,             w_createFence),
    ENTRY(vkDestroyFence,            destroyFence,            w_destroyFence),
    ENTRY(vkCreateSemaphore,         createSemaphore,         w_createSemaphore),
    ENTRY(vkDestroySemaphore,        destroySemaphore,        w_destroySemaphore),
    ENTRY(vkCreateEvent,             createEvent,             w_createEvent),
    ENTRY(vkDestroyEvent,            destroyEvent,            w_destroyEvent),
    ENTRY(vkCreateQueryPool,         createQueryPool,         w_createQueryPool),
    ENTRY(vkDestroyQueryPool,        destroyQueryPool,        w_destroyQueryPool),
    ENTRY(vkCreateSwapchainKHR,      createSwapchainKHR,      w_createSwapchainKHR),
    ENTRY(vkDestroySwapchainKHR,     destroySwapchainKHR,     w_destroySwapchainKHR),
    ENTRY(vkGetSwapchainImagesKHR,   getSwapchainImagesKHR,   w_getSwapchainImagesKHR),
    ENTRY(vkAcquireNextImageKHR,     acquireNextImageKHR,     w_acquireNextImageKHR),

    ENTRY(vkAllocateMemory,          allocateMemory,          w_allocateMemory),
    ENTRY(vkFreeMemory,              freeMemory,              w_freeMemory),
    ENTRY(vkMapMemory,               mapMemory,               w_mapMemory),
    ENTRY(vkUnmapMemory,             unmapMemory,             w_unmapMemory),
    ENTRY(vkFlushMappedMemoryRanges, flushMappedMemoryRanges, w_flushMappedMemoryRanges),
    ENTRY(vkInvalidateMappedMemoryRanges, invalidateMappedMemoryRanges, w_invalidateMappedMemoryRanges),
    ENTRY(vkBindBufferMemory,        bindBufferMemory,        w_bindBufferMemory),
    ENTRY(vkBindImageMemory,         bindImageMemory,         w_bindImageMemory),
    ENTRY(vkBindBufferMemory2,       bindBufferMemory2,       w_bindBufferMemory2),
    ENTRY(vkBindImageMemory2,        bindImageMemory2,        w_bindImageMemory2),
    ENTRY(vkGetBufferMemoryRequirements, getBufferMemoryRequirements, w_getBufferMemoryRequirements),
    ENTRY(vkGetImageMemoryRequirements,  getImageMemoryRequirements,  w_getImageMemoryRequirements),
    ENTRY(vkGetBufferMemoryRequirements2, getBufferMemoryRequirements2, w_getBufferMemoryRequirements2),
    ENTRY(vkGetImageMemoryRequirements2,  getImageMemoryRequirements2,  w_getImageMemoryRequirements2),

    ENTRY(vkWaitForFences,           waitForFences,           w_waitForFences),
    ENTRY(vkResetFences,             resetFences,             w_resetFences),
    ENTRY(vkGetFenceStatus,          getFenceStatus,          w_getFenceStatus),
    ENTRY(vkGetEventStatus,          getEventStatus,          w_getEventStatus),
    ENTRY(vkSetEvent,                setEvent,                w_setEvent),
    ENTRY(vkResetEvent,              resetEvent,              w_resetEvent),
    ENTRY(vkGetQueryPoolResults,     getQueryPoolResults,     w_getQueryPoolResults),
};
#undef ENTRY

// Look up a function name in the intercept table.
PFN_vkVoidFunction find_wrap(const char* name, PFN_vkVoidFunction real_fn) {
    for (const auto& e : g_intercepts) {
        if (strcmp(e.name, name) == 0) {
            if (*e.real_slot == nullptr) {
                *e.real_slot = real_fn;
            }
            return e.wrap;
        }
    }
    return real_fn;
}

// ===================== Our GetInstanceProcAddr =====================

VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL shim_GetDeviceProcAddr(VkDevice, const char*);

VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL
shim_GetInstanceProcAddr(VkInstance instance, const char* pName) {
    if (!g_real.getInstanceProcAddr) {
        return nullptr;
    }
    // Return OUR GetDeviceProcAddr so the Vulkan loader builds per-device
    // dispatch tables through our interceptor instead of the raw driver.
    // Without this the loader bypasses the shim for all vkCmd* calls.
    if (strcmp(pName, "vkGetDeviceProcAddr") == 0) {
        if (!g_real.getDeviceProcAddr) {
            g_real.getDeviceProcAddr = (PFN_vkGetDeviceProcAddr)
                g_real.getInstanceProcAddr(instance, pName);
        }
        return (PFN_vkVoidFunction)shim_GetDeviceProcAddr;
    }
    PFN_vkVoidFunction real_fn = g_real.getInstanceProcAddr(instance, pName);
    if (!real_fn) {
        return nullptr;
    }
    return find_wrap(pName, real_fn);
}

VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL
shim_GetDeviceProcAddr(VkDevice device, const char* pName) {
    if (!g_real.getDeviceProcAddr) {
        return nullptr;
    }
    PFN_vkVoidFunction real_fn = g_real.getDeviceProcAddr(device, pName);
    if (!real_fn) {
        return nullptr;
    }
    return find_wrap(pName, real_fn);
}

// ===================== CreateInstance wrapper =====================
// Pass-through; the interesting work happens inside GetInstanceProcAddr.
// We intercept CreateInstance only to seed getDeviceProcAddr after the
// VkInstance exists (some drivers only resolve GetDeviceProcAddr through
// an instance).

VKAPI_ATTR VkResult VKAPI_CALL
shim_CreateInstance(const VkInstanceCreateInfo* ci,
                    const VkAllocationCallbacks* alloc,
                    VkInstance* out) {
    VkResult r;
    {
        LockGuard _g;
        r = g_real_device->CreateInstance(ci, alloc, out);
    }
    if (r == VK_SUCCESS && g_real.getInstanceProcAddr && !g_real.getDeviceProcAddr) {
        g_real.getDeviceProcAddr = (PFN_vkGetDeviceProcAddr)
            g_real.getInstanceProcAddr(*out, "vkGetDeviceProcAddr");
    }
    return r;
}

VKAPI_ATTR VkResult VKAPI_CALL
shim_EnumerateInstanceExtensionProperties(const char* layer,
                                          uint32_t* count,
                                          VkExtensionProperties* props) {
    LockGuard _g;
    return g_real_device->EnumerateInstanceExtensionProperties(layer, count, props);
}

// ===================== HAL plumbing =====================

hwvulkan_device_t g_shim_device;

int shim_hw_close(struct hw_device_t*) {
    // We don't own g_real_device; closing is the real device's responsibility
    // via its own module. For Azahar's use-case, the process dies with the
    // driver loaded; we intentionally skip cleanup.
    return 0;
}

int shim_hw_open(const struct hw_module_t* module, const char* id,
                 struct hw_device_t** device_out) {
    (void)module;
    // Resolve the real libGLES_mali.so and hook its HAL on first open.
    if (!g_real_device) {
        // RTLD_NOLOAD first to see if the loader already mapped it via our
        // NEEDED chain; if not, pull it in by absolute path so we don't race
        // the loader's search order.
        void* lib = dlopen("libGLES_mali.so", RTLD_NOW | RTLD_NOLOAD);
        if (!lib) {
            lib = dlopen("/vendor/lib64/egl/libGLES_mali.so", RTLD_NOW);
        }
        if (!lib) {
            LOGE("failed to dlopen libGLES_mali.so: %s", dlerror());
            return -1;
        }
        g_real_lib = lib;

        hwvulkan_module_t* real_hmi = (hwvulkan_module_t*)dlsym(lib, "HMI");
        if (!real_hmi) {
            LOGE("real libGLES_mali.so has no HMI symbol");
            return -1;
        }
        g_real_module = real_hmi;

        hwvulkan_device_t* real_dev = nullptr;
        int rc = real_hmi->common.methods->open(&real_hmi->common, id,
                                                (struct hw_device_t**)&real_dev);
        if (rc != 0 || !real_dev) {
            LOGE("real Mali open() failed: %d", rc);
            return rc ? rc : -1;
        }
        g_real_device = real_dev;

        // Cache the real function pointers we'll dispatch through.
        g_real.getInstanceProcAddr = real_dev->GetInstanceProcAddr;

        LOGI("Mali Vulkan ICD shim loaded; real driver at %p", real_dev);
    }

    // Initialise our shim device, cloning the real one and overriding the
    // three members Android's Vulkan loader actually dispatches through.
    g_shim_device = *g_real_device;
    g_shim_device.common.module = const_cast<struct hw_module_t*>(module);
    g_shim_device.common.close  = shim_hw_close;
    g_shim_device.EnumerateInstanceExtensionProperties =
        shim_EnumerateInstanceExtensionProperties;
    g_shim_device.CreateInstance      = shim_CreateInstance;
    g_shim_device.GetInstanceProcAddr = shim_GetInstanceProcAddr;

    *device_out = (struct hw_device_t*)&g_shim_device;
    return 0;
}

hw_module_methods_t g_shim_methods = {
    .open = shim_hw_open,
};

} // namespace

// The Android Vulkan loader locates the HAL by dlsym("HMI"). Match the
// expected hw_module_t layout exactly; anything off-by-a-field sends the
// loader into undefined-behaviour land at boot.
extern "C" __attribute__((visibility("default"))) hwvulkan_module_t HMI = {
    .common = {
        .tag                 = HARDWARE_MODULE_TAG,
        .module_api_version  = HWVULKAN_MODULE_API_VERSION_0_1,
        .hal_api_version     = HARDWARE_HAL_API_VERSION,
        .id                  = HWVULKAN_HARDWARE_MODULE_ID,
        .name                = "Azahar Mali Vulkan shim",
        .author              = "azahar-local",
        .methods             = &g_shim_methods,
        .dso                 = nullptr,
        .reserved            = {0},
    },
};
