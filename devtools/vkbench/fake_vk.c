// vkbench (devtools/vkbench): a fake Vulkan driver for the host (macOS) build of the Android replay. The replay's own sources (gfx_vk_core.c, gfx_vk_exec.c, gfx_vk_tex.c,
// gfx_vk_shader.c, gfx_vk_dq.c) are compiled unchanged and load this "driver" through vk_init's dlopen("libvulkan.so"), which the force-included shim sends here (vkb_dlopen).
// The device it reports is the phone's: Adreno 840, Vulkan 1.4, the limits, memory types and extensions of Mario Kart World's run 13 log ("device facts" lines).
// What it costs is a model, kept apart from the replay's own time: every vkCmd* appends a packet to its command buffer (the bytes the call carries, as a driver builds its stream),
// vkUpdateDescriptorSets writes a 64-byte descriptor per element into the set's memory (Adreno's descriptors: an image view's or sampler's words copied from the object, a buffer's
// address and range), vkAllocateDescriptorSets bumps a pool's arena. VKB_CMD_NS / VKB_DESC_NS / VKB_SET_NS (env, nanoseconds of spin per call, per descriptor, per set) add a
// fixed driver cost on top, to put the replay's share against a driver's: 0 by default (the replay's own CPU time only)
#include <dlfcn.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#define VK_NO_PROTOTYPES
#define VK_USE_PLATFORM_ANDROID_KHR
#include <android/hardware_buffer.h>
#include <android/native_window.h>
#include <vulkan/vulkan.h>

static uint64_t g_cmd_ns, g_desc_ns, g_set_ns; static int g_check;   // VKB_CHECK=1: descriptor set lifetimes checked (a set used after its pool's reset, a destroyed view or buffer bound at submit)
_Atomic uint64_t g_fk_errors;
static void fk_err(const char *what, const void *x) { if (atomic_fetch_add(&g_fk_errors, 1) < 20) fprintf(stderr, "[vkbench] driver check: %s (%p)\n", what, x); }
static inline uint64_t fk_now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return (uint64_t)t.tv_sec * 1000000000ull + (uint64_t)t.tv_nsec; }
// the driver cost model's spin: the system counter (24 MHz on Apple silicon, 41.7 ns a tick; clock_gettime's monotonic clock is microseconds there, which made every spin a microsecond)
#include <mach/mach_time.h>
static double g_tick_ns;
static inline void fk_spin(uint64_t ns) { if (!ns) return; if (!g_tick_ns) { mach_timebase_info_data_t tb; mach_timebase_info(&tb); g_tick_ns = (double)tb.numer / tb.denom; }
    uint64_t t0 = mach_absolute_time(), n = (uint64_t)(ns / g_tick_ns + 0.5); while (mach_absolute_time() - t0 < n) {} }
_Atomic uint64_t g_fk_calls, g_fk_cmd_bytes, g_fk_desc_writes, g_fk_sets, g_fk_updates, g_fk_binds, g_fk_draws, g_fk_pipes, g_fk_pushdesc;   // what the driver was asked (the report)

// ---- objects: every handle is a heap object; dispatchable ones start with a word the loader would own ----
typedef struct { void *loader; int kind; } fk_disp;
typedef struct fk_set_s fk_set;
typedef struct { fk_disp h; uint8_t *buf; size_t n, cap; fk_set **sets; uint32_t nsets, capsets; } fk_cb;   // a command buffer: its packets, the sets it bound (VKB_CHECK)
typedef struct fk_view_s { uint8_t hw[64]; int dead; uint32_t gen; struct fk_view_s *next; } fk_view;                 // an image view, a sampler, a buffer view: the descriptor words a set takes from it (dead: destroyed, VKB_CHECK)
typedef struct { uint64_t size; uint8_t *host; } fk_mem;
typedef struct fk_buf_s { uint64_t size, addr; fk_mem *mem; uint64_t off; int dead; uint32_t gen; struct fk_buf_s *next; } fk_buf;
typedef struct { uint32_t n, total; struct { uint32_t binding, type, count, at; } b[96]; } fk_dsl;   // at: the binding's first descriptor in the set's memory (64 bytes each)
typedef struct { uint8_t *arena; size_t used, cap; uint64_t gen; } fk_dpool;   // gen: resets so far (a set of an earlier generation is gone)
struct fk_set_s { const fk_dsl *l; uint8_t *mem; fk_dpool *pool; uint64_t gen; uint32_t nref; void *ref[96]; int refk[96]; uint32_t refgen[96]; };   // ref: the handles its descriptors name (refk 0 view or sampler, 1 buffer)
typedef struct { _Atomic uint64_t v; pthread_mutex_t mu; pthread_cond_t cv; int timeline; } fk_sem;
static fk_disp g_inst = {NULL, 1}, g_pd = {NULL, 2}, g_dev = {NULL, 3}, g_queue = {NULL, 4};
static void *obj(size_t n) { void *p = calloc(1, n); if (!p) abort(); return p; }
static fk_view *g_free_views; static fk_buf *g_free_bufs; static pthread_mutex_t g_free_mu = PTHREAD_MUTEX_INITIALIZER;   // (VKB_CHECK: destroyed handles, given out again)
static fk_view *view_new(void) { fk_view *v = NULL; pthread_mutex_lock(&g_free_mu); if (g_free_views) { v = g_free_views; g_free_views = v->next; } pthread_mutex_unlock(&g_free_mu);
    if (!v) return obj(sizeof(fk_view)); uint32_t g = v->gen; memset(v, 0, sizeof *v); v->gen = g; return v; }
static void view_gone(fk_view *v) { pthread_mutex_lock(&g_free_mu); v->dead = 1; v->gen++; v->next = g_free_views; g_free_views = v; pthread_mutex_unlock(&g_free_mu); }
static fk_buf *buf_new(void) { fk_buf *b = NULL; pthread_mutex_lock(&g_free_mu); if (g_free_bufs) { b = g_free_bufs; g_free_bufs = b->next; } pthread_mutex_unlock(&g_free_mu);
    if (!b) return obj(sizeof(fk_buf)); uint32_t g = b->gen; memset(b, 0, sizeof *b); b->gen = g; return b; }
static void buf_gone(fk_buf *b) { pthread_mutex_lock(&g_free_mu); b->dead = 1; b->gen++; b->next = g_free_bufs; g_free_bufs = b; pthread_mutex_unlock(&g_free_mu); }

// ---- instance and physical device ----
static VKAPI_ATTR VkResult VKAPI_CALL fk_CreateInstance(const VkInstanceCreateInfo *ci, const VkAllocationCallbacks *a, VkInstance *out) { (void)ci; (void)a; *out = (VkInstance)&g_inst; return VK_SUCCESS; }
static VKAPI_ATTR void VKAPI_CALL fk_DestroyInstance(VkInstance i, const VkAllocationCallbacks *a) { (void)i; (void)a; }
static VKAPI_ATTR VkResult VKAPI_CALL fk_EnumeratePhysicalDevices(VkInstance i, uint32_t *n, VkPhysicalDevice *pd) { (void)i; if (pd && *n) pd[0] = (VkPhysicalDevice)&g_pd; *n = 1; return VK_SUCCESS; }
static void fill_props(VkPhysicalDeviceProperties *p) {
    memset(p, 0, sizeof *p);
    p->apiVersion = VK_MAKE_API_VERSION(0, 1, 4, 295); p->driverVersion = 0x8034a013; p->vendorID = 0x5143; p->deviceID = 0x44050a31; p->deviceType = VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU;
    snprintf(p->deviceName, sizeof p->deviceName, "Adreno (TM) 840 (vkbench fake)");
    VkPhysicalDeviceLimits *L = &p->limits;
    L->maxImageDimension1D = L->maxImageDimension2D = 16384; L->maxImageDimension3D = 2048; L->maxImageDimensionCube = 16384; L->maxImageArrayLayers = 2048;
    L->maxUniformBufferRange = 65536; L->maxStorageBufferRange = 2147483647u; L->maxPushConstantsSize = 256; L->maxMemoryAllocationCount = 4294967295u; L->maxSamplerAllocationCount = 65536;
    L->maxBoundDescriptorSets = 7; L->maxPerStageDescriptorSamplers = L->maxPerStageDescriptorUniformBuffers = L->maxPerStageDescriptorStorageBuffers = L->maxPerStageDescriptorSampledImages = L->maxPerStageDescriptorStorageImages = 16777216;
    L->maxPerStageResources = 16777216; L->maxDescriptorSetSamplers = L->maxDescriptorSetUniformBuffers = L->maxDescriptorSetStorageBuffers = L->maxDescriptorSetSampledImages = L->maxDescriptorSetStorageImages = 16777216;
    L->maxDescriptorSetUniformBuffersDynamic = 64; L->maxDescriptorSetStorageBuffersDynamic = 32;   // (the replay takes up to half of the first per stage: gfx_vk_shader.c dynmax)
    L->maxVertexInputAttributes = 32; L->maxVertexInputBindings = 32; L->maxVertexInputAttributeOffset = 2047; L->maxVertexInputBindingStride = 2048; L->maxVertexOutputComponents = 128;
    L->maxFragmentInputComponents = 128; L->maxFragmentOutputAttachments = 8; L->maxColorAttachments = 8; L->maxComputeSharedMemorySize = 32768;
    L->maxComputeWorkGroupCount[0] = L->maxComputeWorkGroupCount[1] = L->maxComputeWorkGroupCount[2] = 65535; L->maxComputeWorkGroupInvocations = 1024;
    L->maxComputeWorkGroupSize[0] = L->maxComputeWorkGroupSize[1] = L->maxComputeWorkGroupSize[2] = 1024; L->maxDrawIndexedIndexValue = 0xffffffffu; L->maxDrawIndirectCount = 0xffffffffu;
    L->maxSamplerLodBias = 15; L->maxSamplerAnisotropy = 16; L->maxViewports = 16; L->maxViewportDimensions[0] = L->maxViewportDimensions[1] = 16384; L->viewportBoundsRange[0] = -32768; L->viewportBoundsRange[1] = 32767;
    L->minMemoryMapAlignment = 64; L->minTexelBufferOffsetAlignment = 64; L->minUniformBufferOffsetAlignment = 64; L->minStorageBufferOffsetAlignment = 64;
    L->maxFramebufferWidth = L->maxFramebufferHeight = 16384; L->maxFramebufferLayers = 2048; L->framebufferColorSampleCounts = L->framebufferDepthSampleCounts = L->framebufferStencilSampleCounts = 0xf;
    L->sampledImageColorSampleCounts = L->sampledImageDepthSampleCounts = L->sampledImageIntegerSampleCounts = L->sampledImageStencilSampleCounts = L->storageImageSampleCounts = 0xf;
    L->timestampComputeAndGraphics = 1; L->timestampPeriod = 52.083f; L->maxClipDistances = 8; L->maxCullDistances = 8; L->pointSizeRange[0] = 1; L->pointSizeRange[1] = 1023; L->lineWidthRange[0] = L->lineWidthRange[1] = 1;
    L->nonCoherentAtomSize = 64; L->optimalBufferCopyOffsetAlignment = 64; L->optimalBufferCopyRowPitchAlignment = 64; L->maxTexelBufferElements = 134217728; L->bufferImageGranularity = 1;
    L->sparseAddressSpaceSize = 3ull << 30;
    p->sparseProperties.residencyStandard2DBlockShape = 1; p->sparseProperties.residencyNonResidentStrict = 1;
}
static VKAPI_ATTR void VKAPI_CALL fk_GetPhysicalDeviceProperties(VkPhysicalDevice pd, VkPhysicalDeviceProperties *p) { (void)pd; fill_props(p); }
static VKAPI_ATTR void VKAPI_CALL fk_GetPhysicalDeviceProperties2(VkPhysicalDevice pd, VkPhysicalDeviceProperties2 *p) {
    (void)pd; fill_props(&p->properties);
    for (VkBaseOutStructure *s = p->pNext; s; s = s->pNext) switch (s->sType) {
    case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES: { VkPhysicalDeviceSubgroupProperties *g = (void *)s; g->subgroupSize = 64; g->supportedStages = 0xf0 | VK_SHADER_STAGE_COMPUTE_BIT | VK_SHADER_STAGE_FRAGMENT_BIT | VK_SHADER_STAGE_VERTEX_BIT; g->supportedOperations = 0x6ff; g->quadOperationsInAllStages = 1; break; }
    case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_PROPERTIES: { VkPhysicalDeviceSubgroupSizeControlProperties *z = (void *)s; z->minSubgroupSize = z->maxSubgroupSize = 64; z->maxComputeWorkgroupSubgroups = 16; z->requiredSubgroupSizeStages = 0xe0; break; }
    case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MESH_SHADER_PROPERTIES_EXT: { VkPhysicalDeviceMeshShaderPropertiesEXT *m = (void *)s;
        m->maxTaskWorkGroupTotalCount = m->maxMeshWorkGroupTotalCount = 4194304; for (int i = 0; i < 3; i++) { m->maxTaskWorkGroupCount[i] = m->maxMeshWorkGroupCount[i] = 65535; m->maxTaskWorkGroupSize[i] = m->maxMeshWorkGroupSize[i] = 128; }
        m->maxTaskWorkGroupInvocations = m->maxMeshWorkGroupInvocations = 128; m->maxTaskPayloadSize = 16384; m->maxTaskSharedMemorySize = m->maxMeshSharedMemorySize = 32768;
        m->maxTaskPayloadAndSharedMemorySize = 49152; m->maxMeshPayloadAndSharedMemorySize = 32768; m->maxMeshOutputVertices = m->maxMeshOutputPrimitives = 256; m->maxMeshOutputMemorySize = 32768; break; }
    case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BLEND_OPERATION_ADVANCED_PROPERTIES_EXT: { VkPhysicalDeviceBlendOperationAdvancedPropertiesEXT *b = (void *)s; b->advancedBlendMaxColorAttachments = 1; b->advancedBlendAllOperations = 1; break; }
    case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRIVER_PROPERTIES: { VkPhysicalDeviceDriverProperties *d = (void *)s; d->driverID = VK_DRIVER_ID_QUALCOMM_PROPRIETARY; snprintf(d->driverName, sizeof d->driverName, "vkbench fake Adreno driver"); snprintf(d->driverInfo, sizeof d->driverInfo, "no GPU"); d->conformanceVersion.major = 1; d->conformanceVersion.minor = 4; break; }
    case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PUSH_DESCRIPTOR_PROPERTIES_KHR: { VkPhysicalDevicePushDescriptorPropertiesKHR *q = (void *)s; q->maxPushDescriptors = 32; break; }
    default: break;
    }
}
static const VkMemoryPropertyFlags MT[11] = {0x1, 0x1, 0x1, 0xb, 0x7, 0x7, 0xf, 0xf, 0x21, 0x11, 0x11};   // run 13's "device facts: memory heap" types
static const uint32_t MH[11] = {0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 2};
static void fill_mem(VkPhysicalDeviceMemoryProperties *m) {
    memset(m, 0, sizeof *m); m->memoryTypeCount = 11; m->memoryHeapCount = 3;
    for (int i = 0; i < 11; i++) { m->memoryTypes[i].propertyFlags = MT[i]; m->memoryTypes[i].heapIndex = MH[i]; }
    m->memoryHeaps[0] = (VkMemoryHeap){15800000000ull, VK_MEMORY_HEAP_DEVICE_LOCAL_BIT}; m->memoryHeaps[1] = (VkMemoryHeap){4ull << 30, VK_MEMORY_HEAP_DEVICE_LOCAL_BIT}; m->memoryHeaps[2] = (VkMemoryHeap){10 << 20, VK_MEMORY_HEAP_DEVICE_LOCAL_BIT};
}
static VKAPI_ATTR void VKAPI_CALL fk_GetPhysicalDeviceMemoryProperties(VkPhysicalDevice pd, VkPhysicalDeviceMemoryProperties *m) { (void)pd; fill_mem(m); }
static VKAPI_ATTR void VKAPI_CALL fk_GetPhysicalDeviceMemoryProperties2(VkPhysicalDevice pd, VkPhysicalDeviceMemoryProperties2 *m) { (void)pd; fill_mem(&m->memoryProperties); }
static VKAPI_ATTR void VKAPI_CALL fk_GetPhysicalDeviceQueueFamilyProperties(VkPhysicalDevice pd, uint32_t *n, VkQueueFamilyProperties *q) {
    (void)pd; if (q && *n) q[0] = (VkQueueFamilyProperties){VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT | VK_QUEUE_TRANSFER_BIT | VK_QUEUE_SPARSE_BINDING_BIT, 1, 48, {1, 1, 1}}; *n = 1; }
static const char *EXTS[] = {"VK_KHR_swapchain", "VK_ANDROID_external_memory_android_hardware_buffer", "VK_EXT_queue_family_foreign", "VK_EXT_custom_border_color", "VK_EXT_robustness2",
    "VK_EXT_depth_clip_control", "VK_EXT_provoking_vertex", "VK_EXT_index_type_uint8", "VK_EXT_transform_feedback", "VK_GOOGLE_display_timing", "VK_EXT_mesh_shader", "VK_EXT_blend_operation_advanced",
    "VK_KHR_push_descriptor", "VK_EXT_extended_dynamic_state", "VK_EXT_extended_dynamic_state2", "VK_EXT_extended_dynamic_state3", "VK_EXT_vertex_input_dynamic_state", "VK_KHR_descriptor_update_template",
    "VK_EXT_subgroup_size_control", "VK_KHR_driver_properties"};
static VKAPI_ATTR VkResult VKAPI_CALL fk_EnumerateDeviceExtensionProperties(VkPhysicalDevice pd, const char *layer, uint32_t *n, VkExtensionProperties *e) {
    (void)pd; (void)layer; uint32_t k = sizeof EXTS / sizeof *EXTS;
    if (e) { if (*n < k) k = *n; for (uint32_t i = 0; i < k; i++) { snprintf(e[i].extensionName, sizeof e[i].extensionName, "%s", EXTS[i]); e[i].specVersion = 1; } }
    *n = k; return VK_SUCCESS; }
static VKAPI_ATTR void VKAPI_CALL fk_GetPhysicalDeviceFeatures2(VkPhysicalDevice pd, VkPhysicalDeviceFeatures2 *f) {
    (void)pd; VkPhysicalDeviceFeatures *c = &f->features; VkBool32 *all = (VkBool32 *)c; for (size_t i = 0; i < sizeof *c / sizeof(VkBool32); i++) all[i] = VK_TRUE;
    c->shaderFloat64 = 0; c->sparseResidencyImage3D = 0; c->sparseResidencyAliased = 0; c->sparseResidency2Samples = c->sparseResidency4Samples = c->sparseResidency8Samples = c->sparseResidency16Samples = 0;
    // every feature struct the replay chains (VkBool32s after sType / pNext): all on, but robustness and a few the device lacks
    for (VkBaseOutStructure *s = f->pNext; s; s = s->pNext) {
        size_t sz = 0;
        switch (s->sType) {
        case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES: sz = sizeof(VkPhysicalDeviceVulkan11Features); break;
        case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES: sz = sizeof(VkPhysicalDeviceVulkan12Features); break;
        case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES: sz = sizeof(VkPhysicalDeviceVulkan13Features); break;
        case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_CUSTOM_BORDER_COLOR_FEATURES_EXT: sz = sizeof(VkPhysicalDeviceCustomBorderColorFeaturesEXT); break;
        case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ROBUSTNESS_2_FEATURES_EXT: sz = sizeof(VkPhysicalDeviceRobustness2FeaturesEXT); break;
        case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DEPTH_CLIP_CONTROL_FEATURES_EXT: sz = sizeof(VkPhysicalDeviceDepthClipControlFeaturesEXT); break;
        case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_INDEX_TYPE_UINT8_FEATURES_EXT: sz = sizeof(VkPhysicalDeviceIndexTypeUint8FeaturesEXT); break;
        case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TRANSFORM_FEEDBACK_FEATURES_EXT: sz = sizeof(VkPhysicalDeviceTransformFeedbackFeaturesEXT); break;
        case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MESH_SHADER_FEATURES_EXT: sz = sizeof(VkPhysicalDeviceMeshShaderFeaturesEXT); break;
        case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BLEND_OPERATION_ADVANCED_FEATURES_EXT: sz = sizeof(VkPhysicalDeviceBlendOperationAdvancedFeaturesEXT); break;
        default: sz = sizeof(VkBaseOutStructure) + 4 * sizeof(VkBool32); break;
        }
        VkBool32 *b = (VkBool32 *)((uint8_t *)s + sizeof(VkBaseOutStructure)); for (size_t i = 0; i < (sz - sizeof(VkBaseOutStructure)) / sizeof(VkBool32); i++) b[i] = VK_TRUE;
        if (s->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES) ((VkPhysicalDeviceVulkan12Features *)s)->shaderFloat16 = VK_TRUE;
        if (s->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MESH_SHADER_FEATURES_EXT) { VkPhysicalDeviceMeshShaderFeaturesEXT *m = (void *)s; m->multiviewMeshShader = m->primitiveFragmentShadingRateMeshShader = 0; }
    }
}
static VKAPI_ATTR void VKAPI_CALL fk_GetPhysicalDeviceFormatProperties(VkPhysicalDevice pd, VkFormat f, VkFormatProperties *p) { (void)pd; (void)f; p->linearTilingFeatures = p->optimalTilingFeatures = 0x7fffffff; p->bufferFeatures = 0x7fffffff; }
static VKAPI_ATTR void VKAPI_CALL fk_GetPhysicalDeviceFormatProperties2(VkPhysicalDevice pd, VkFormat f, VkFormatProperties2 *p) { fk_GetPhysicalDeviceFormatProperties(pd, f, &p->formatProperties); }
static VKAPI_ATTR VkResult VKAPI_CALL fk_GetPhysicalDeviceImageFormatProperties(VkPhysicalDevice pd, VkFormat f, VkImageType t, VkImageTiling ti, VkImageUsageFlags u, VkImageCreateFlags c, VkImageFormatProperties *p) {
    (void)pd; (void)f; (void)t; (void)ti; (void)u; (void)c; p->maxExtent = (VkExtent3D){16384, 16384, 2048}; p->maxMipLevels = 15; p->maxArrayLayers = 2048; p->sampleCounts = 0xf; p->maxResourceSize = 1ull << 34; return VK_SUCCESS; }
static VKAPI_ATTR void VKAPI_CALL fk_GetPhysicalDeviceSparseImageFormatProperties(VkPhysicalDevice pd, VkFormat f, VkImageType t, VkSampleCountFlagBits s, VkImageUsageFlags u, VkImageTiling ti, uint32_t *n, VkSparseImageFormatProperties *p) {
    (void)pd; (void)f; (void)t; (void)s; (void)u; (void)ti; (void)p; *n = 0; }   // (no sparse residency formats: the sparse paths take their degraded way; the benchmark has no sparse pools)
static VKAPI_ATTR VkResult VKAPI_CALL fk_GetPhysicalDeviceSurfaceCapabilitiesKHR(VkPhysicalDevice pd, VkSurfaceKHR s, VkSurfaceCapabilitiesKHR *c) { (void)pd; (void)s; memset(c, 0, sizeof *c); return VK_SUCCESS; }
static VKAPI_ATTR VkResult VKAPI_CALL fk_GetPhysicalDeviceSurfaceFormatsKHR(VkPhysicalDevice pd, VkSurfaceKHR s, uint32_t *n, VkSurfaceFormatKHR *f) { (void)pd; (void)s; (void)f; *n = 0; return VK_SUCCESS; }
static VKAPI_ATTR VkResult VKAPI_CALL fk_GetPhysicalDeviceSurfacePresentModesKHR(VkPhysicalDevice pd, VkSurfaceKHR s, uint32_t *n, VkPresentModeKHR *m) { (void)pd; (void)s; (void)m; *n = 0; return VK_SUCCESS; }
static VKAPI_ATTR VkResult VKAPI_CALL fk_CreateAndroidSurfaceKHR(VkInstance i, const VkAndroidSurfaceCreateInfoKHR *ci, const VkAllocationCallbacks *a, VkSurfaceKHR *s) { (void)i; (void)ci; (void)a; *s = (VkSurfaceKHR)obj(8); return VK_SUCCESS; }
static VKAPI_ATTR void VKAPI_CALL fk_DestroySurfaceKHR(VkInstance i, VkSurfaceKHR s, const VkAllocationCallbacks *a) { (void)i; (void)s; (void)a; }
static VKAPI_ATTR VkResult VKAPI_CALL fk_GetPhysicalDeviceSurfaceSupportKHR(VkPhysicalDevice pd, uint32_t q, VkSurfaceKHR s, VkBool32 *ok) { (void)pd; (void)q; (void)s; *ok = 1; return VK_SUCCESS; }

// ---- device, queue, memory, buffers, images ----
static VKAPI_ATTR VkResult VKAPI_CALL fk_CreateDevice(VkPhysicalDevice pd, const VkDeviceCreateInfo *ci, const VkAllocationCallbacks *a, VkDevice *d) { (void)pd; (void)ci; (void)a; *d = (VkDevice)&g_dev; return VK_SUCCESS; }
static VKAPI_ATTR void VKAPI_CALL fk_GetDeviceQueue(VkDevice d, uint32_t f, uint32_t i, VkQueue *q) { (void)d; (void)f; (void)i; *q = (VkQueue)&g_queue; }
static VKAPI_ATTR VkResult VKAPI_CALL fk_AllocateMemory(VkDevice d, const VkMemoryAllocateInfo *ai, const VkAllocationCallbacks *a, VkDeviceMemory *m) {
    (void)d; (void)a; fk_mem *x = obj(sizeof *x); x->size = ai->allocationSize; int imported = 0;
    for (const VkBaseInStructure *s = ai->pNext; s; s = s->pNext) imported |= s->sType == VK_STRUCTURE_TYPE_IMPORT_ANDROID_HARDWARE_BUFFER_INFO_ANDROID;
    // host-visible memory the replay maps (scratch, the support buffers) is real; device-local and imported memory has no host copy
    if (!imported && ai->memoryTypeIndex < 11 && (MT[ai->memoryTypeIndex] & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)) { x->host = calloc(1, (size_t)(ai->allocationSize ? ai->allocationSize : 1)); if (!x->host) return VK_ERROR_OUT_OF_HOST_MEMORY; }
    *m = (VkDeviceMemory)x; return VK_SUCCESS; }
static VKAPI_ATTR void VKAPI_CALL fk_FreeMemory(VkDevice d, VkDeviceMemory m, const VkAllocationCallbacks *a) { (void)d; (void)a; fk_mem *x = (fk_mem *)m; if (x) { free(x->host); free(x); } }
static VKAPI_ATTR VkResult VKAPI_CALL fk_MapMemory(VkDevice d, VkDeviceMemory m, VkDeviceSize o, VkDeviceSize n, VkMemoryMapFlags f, void **p) { (void)d; (void)n; (void)f; fk_mem *x = (fk_mem *)m; if (!x->host) return VK_ERROR_MEMORY_MAP_FAILED; *p = x->host + o; return VK_SUCCESS; }
static VKAPI_ATTR VkResult VKAPI_CALL fk_CreateBuffer(VkDevice d, const VkBufferCreateInfo *ci, const VkAllocationCallbacks *a, VkBuffer *b) { (void)d; (void)a; fk_buf *x = g_check ? buf_new() : obj(sizeof *x); x->size = ci->size; *b = (VkBuffer)x; return VK_SUCCESS; }
static VKAPI_ATTR void VKAPI_CALL fk_DestroyBuffer(VkDevice d, VkBuffer b, const VkAllocationCallbacks *a) { (void)d; (void)a; if (!b) return; if (g_check) buf_gone((fk_buf *)b); else free((void *)b); }
static VKAPI_ATTR void VKAPI_CALL fk_GetBufferMemoryRequirements(VkDevice d, VkBuffer b, VkMemoryRequirements *r) { (void)d; r->size = ((fk_buf *)b)->size; r->alignment = 256; r->memoryTypeBits = 0x7ff; }
static VKAPI_ATTR VkResult VKAPI_CALL fk_BindBufferMemory(VkDevice d, VkBuffer b, VkDeviceMemory m, VkDeviceSize o) { (void)d; ((fk_buf *)b)->mem = (fk_mem *)m; ((fk_buf *)b)->off = o; ((fk_buf *)b)->addr = 0x100000000ull + (uint64_t)(uintptr_t)b; return VK_SUCCESS; }
typedef struct { uint64_t size; VkFormat fmt; } fk_img;
static VKAPI_ATTR VkResult VKAPI_CALL fk_CreateImage(VkDevice d, const VkImageCreateInfo *ci, const VkAllocationCallbacks *a, VkImage *im) {
    (void)d; (void)a; fk_img *x = obj(sizeof *x); x->fmt = ci->format;
    uint64_t px = (uint64_t)ci->extent.width * ci->extent.height * (ci->extent.depth ? ci->extent.depth : 1) * (ci->arrayLayers ? ci->arrayLayers : 1); x->size = px * 4 * 4 / 3 + 4096; *im = (VkImage)x; return VK_SUCCESS; }
static VKAPI_ATTR void VKAPI_CALL fk_DestroyImage(VkDevice d, VkImage i, const VkAllocationCallbacks *a) { (void)d; (void)a; free((void *)i); }
static VKAPI_ATTR void VKAPI_CALL fk_GetImageMemoryRequirements(VkDevice d, VkImage i, VkMemoryRequirements *r) { (void)d; r->size = (((fk_img *)i)->size + 4095) & ~4095ull; r->alignment = 4096; r->memoryTypeBits = 0x7; }
static VKAPI_ATTR VkResult VKAPI_CALL fk_BindImageMemory(VkDevice d, VkImage i, VkDeviceMemory m, VkDeviceSize o) { (void)d; (void)i; (void)m; (void)o; return VK_SUCCESS; }
static VKAPI_ATTR void VKAPI_CALL fk_GetImageSparseMemoryRequirements(VkDevice d, VkImage i, uint32_t *n, VkSparseImageMemoryRequirements *r) { (void)d; (void)i; (void)r; *n = 0; }
static VKAPI_ATTR VkResult VKAPI_CALL fk_GetAndroidHardwareBufferPropertiesANDROID(VkDevice d, const struct AHardwareBuffer *b, VkAndroidHardwareBufferPropertiesANDROID *p) {
    (void)d; AHardwareBuffer_Desc ds; AHardwareBuffer_describe(b, &ds); p->allocationSize = ds.width; p->memoryTypeBits = 0x7ff; return VK_SUCCESS; }
static uint64_t g_view_serial;
static VKAPI_ATTR VkResult VKAPI_CALL fk_CreateImageView(VkDevice d, const VkImageViewCreateInfo *ci, const VkAllocationCallbacks *a, VkImageView *v) {
    (void)d; (void)a; fk_view *x = g_check ? view_new() : obj(sizeof *x); uint64_t s = ++g_view_serial; memcpy(x->hw, &ci->image, 8); memcpy(x->hw + 8, &ci->format, 4); memcpy(x->hw + 12, &ci->viewType, 4); memcpy(x->hw + 16, &s, 8);
    memcpy(x->hw + 24, &ci->subresourceRange, sizeof ci->subresourceRange); *v = (VkImageView)x; return VK_SUCCESS; }
static VKAPI_ATTR void VKAPI_CALL fk_DestroyImageView(VkDevice d, VkImageView v, const VkAllocationCallbacks *a) { (void)d; (void)a; if (!v) return; if (g_check) view_gone((fk_view *)v); else free((void *)v); }
static VKAPI_ATTR VkResult VKAPI_CALL fk_CreateSampler(VkDevice d, const VkSamplerCreateInfo *ci, const VkAllocationCallbacks *a, VkSampler *s) { (void)d; (void)a; fk_view *x = obj(sizeof *x); memcpy(x->hw, &ci->magFilter, 48); *s = (VkSampler)x; return VK_SUCCESS; }
static VKAPI_ATTR void VKAPI_CALL fk_DestroySampler(VkDevice d, VkSampler s, const VkAllocationCallbacks *a) { (void)d; (void)a; free((void *)s); }
static VKAPI_ATTR VkResult VKAPI_CALL fk_CreateBufferView(VkDevice d, const VkBufferViewCreateInfo *ci, const VkAllocationCallbacks *a, VkBufferView *v) {
    (void)d; (void)a; fk_view *x = obj(sizeof *x); memcpy(x->hw, &ci->buffer, 8); memcpy(x->hw + 8, &ci->offset, 8); memcpy(x->hw + 16, &ci->range, 8); *v = (VkBufferView)x; return VK_SUCCESS; }
static VKAPI_ATTR void VKAPI_CALL fk_DestroyBufferView(VkDevice d, VkBufferView v, const VkAllocationCallbacks *a) { (void)d; (void)a; if (!v) return; if (g_check) ((fk_view *)v)->dead = 1; else free((void *)v); }

// ---- command pools and buffers: packets appended ----
static VKAPI_ATTR VkResult VKAPI_CALL fk_CreateCommandPool(VkDevice d, const VkCommandPoolCreateInfo *ci, const VkAllocationCallbacks *a, VkCommandPool *p) { (void)d; (void)ci; (void)a; *p = (VkCommandPool)obj(8); return VK_SUCCESS; }
static VKAPI_ATTR VkResult VKAPI_CALL fk_ResetCommandPool(VkDevice d, VkCommandPool p, VkCommandPoolResetFlags f) { (void)d; (void)p; (void)f; return VK_SUCCESS; }
static VKAPI_ATTR VkResult VKAPI_CALL fk_AllocateCommandBuffers(VkDevice d, const VkCommandBufferAllocateInfo *ai, VkCommandBuffer *cb) {
    (void)d; for (uint32_t i = 0; i < ai->commandBufferCount; i++) { fk_cb *c = obj(sizeof *c); c->h.kind = 5; c->cap = 1 << 20; c->buf = malloc(c->cap); cb[i] = (VkCommandBuffer)c; } return VK_SUCCESS; }
static VKAPI_ATTR VkResult VKAPI_CALL fk_BeginCommandBuffer(VkCommandBuffer cb, const VkCommandBufferBeginInfo *bi) { (void)bi; ((fk_cb *)cb)->n = 0; ((fk_cb *)cb)->nsets = 0; return VK_SUCCESS; }
static VKAPI_ATTR VkResult VKAPI_CALL fk_EndCommandBuffer(VkCommandBuffer cb) { (void)cb; return VK_SUCCESS; }
static VKAPI_ATTR VkResult VKAPI_CALL fk_ResetCommandBuffer(VkCommandBuffer cb, VkCommandBufferResetFlags f) { (void)f; ((fk_cb *)cb)->n = 0; ((fk_cb *)cb)->nsets = 0; return VK_SUCCESS; }
static inline void pkt(VkCommandBuffer cb, uint32_t op, const void *a, size_t n, const void *b, size_t nb) {   // a packet: op, length, the call's bytes
    fk_cb *c = (fk_cb *)cb; size_t need = 8 + n + nb; atomic_fetch_add_explicit(&g_fk_calls, 1, memory_order_relaxed); atomic_fetch_add_explicit(&g_fk_cmd_bytes, need, memory_order_relaxed);
    if (c->n + need > c->cap) { while (c->n + need > c->cap) c->cap *= 2; c->buf = realloc(c->buf, c->cap); }
    uint32_t h[2] = {op, (uint32_t)need}; memcpy(c->buf + c->n, h, 8); if (n) memcpy(c->buf + c->n + 8, a, n); if (nb) memcpy(c->buf + c->n + 8 + n, b, nb); c->n += need; fk_spin(g_cmd_ns);
}
#define P(op, ...) do { struct { __VA_ARGS__ } a_; memset(&a_, 0, sizeof a_);
static VKAPI_ATTR void VKAPI_CALL fk_CmdPipelineBarrier(VkCommandBuffer cb, VkPipelineStageFlags s, VkPipelineStageFlags d, VkDependencyFlags f, uint32_t nm, const VkMemoryBarrier *m, uint32_t nb, const VkBufferMemoryBarrier *b, uint32_t ni, const VkImageMemoryBarrier *im) {
    uint32_t a[4] = {s, d, f, nm}; (void)b; (void)nb; pkt(cb, 1, a, sizeof a, ni ? (const void *)im : (const void *)m, ni ? ni * sizeof *im : nm * sizeof *m); }
static VKAPI_ATTR void VKAPI_CALL fk_CmdCopyBuffer(VkCommandBuffer cb, VkBuffer s, VkBuffer d, uint32_t n, const VkBufferCopy *r) { VkBuffer a[2] = {s, d}; pkt(cb, 2, a, sizeof a, r, n * sizeof *r); }
static VKAPI_ATTR void VKAPI_CALL fk_CmdCopyBufferToImage(VkCommandBuffer cb, VkBuffer b, VkImage i, VkImageLayout l, uint32_t n, const VkBufferImageCopy *r) { (void)l; void *a[2] = {(void *)b, (void *)i}; pkt(cb, 3, a, sizeof a, r, n * sizeof *r); }
static VKAPI_ATTR void VKAPI_CALL fk_CmdCopyImageToBuffer(VkCommandBuffer cb, VkImage i, VkImageLayout l, VkBuffer b, uint32_t n, const VkBufferImageCopy *r) { (void)l; void *a[2] = {(void *)i, (void *)b}; pkt(cb, 4, a, sizeof a, r, n * sizeof *r); }
static VKAPI_ATTR void VKAPI_CALL fk_CmdCopyImage(VkCommandBuffer cb, VkImage s, VkImageLayout sl, VkImage d, VkImageLayout dl, uint32_t n, const VkImageCopy *r) { (void)sl; (void)dl; void *a[2] = {(void *)s, (void *)d}; pkt(cb, 5, a, sizeof a, r, n * sizeof *r); }
static VKAPI_ATTR void VKAPI_CALL fk_CmdBlitImage(VkCommandBuffer cb, VkImage s, VkImageLayout sl, VkImage d, VkImageLayout dl, uint32_t n, const VkImageBlit *r, VkFilter f) { (void)sl; (void)dl; (void)f; void *a[2] = {(void *)s, (void *)d}; pkt(cb, 6, a, sizeof a, r, n * sizeof *r); }
static VKAPI_ATTR void VKAPI_CALL fk_CmdResolveImage(VkCommandBuffer cb, VkImage s, VkImageLayout sl, VkImage d, VkImageLayout dl, uint32_t n, const VkImageResolve *r) { (void)sl; (void)dl; void *a[2] = {(void *)s, (void *)d}; pkt(cb, 7, a, sizeof a, r, n * sizeof *r); }
static VKAPI_ATTR void VKAPI_CALL fk_CmdFillBuffer(VkCommandBuffer cb, VkBuffer b, VkDeviceSize o, VkDeviceSize n, uint32_t v) { uint64_t a[4] = {(uint64_t)(uintptr_t)b, o, n, v}; pkt(cb, 8, a, sizeof a, NULL, 0); }
static VKAPI_ATTR void VKAPI_CALL fk_CmdUpdateBuffer(VkCommandBuffer cb, VkBuffer b, VkDeviceSize o, VkDeviceSize n, const void *p) { uint64_t a[3] = {(uint64_t)(uintptr_t)b, o, n}; pkt(cb, 9, a, sizeof a, p, (size_t)n); }
static VKAPI_ATTR void VKAPI_CALL fk_CmdClearColorImage(VkCommandBuffer cb, VkImage i, VkImageLayout l, const VkClearColorValue *c, uint32_t n, const VkImageSubresourceRange *r) { (void)l; (void)n; (void)r; void *a[1] = {(void *)i}; pkt(cb, 10, a, sizeof a, c, sizeof *c); }
static VKAPI_ATTR void VKAPI_CALL fk_CmdClearDepthStencilImage(VkCommandBuffer cb, VkImage i, VkImageLayout l, const VkClearDepthStencilValue *c, uint32_t n, const VkImageSubresourceRange *r) { (void)l; (void)n; (void)r; void *a[1] = {(void *)i}; pkt(cb, 11, a, sizeof a, c, sizeof *c); }
static VKAPI_ATTR void VKAPI_CALL fk_CmdClearAttachments(VkCommandBuffer cb, uint32_t na, const VkClearAttachment *at, uint32_t nr, const VkClearRect *rc) { (void)rc; uint32_t a[2] = {na, nr}; pkt(cb, 12, a, sizeof a, at, na * sizeof *at); }
static VKAPI_ATTR void VKAPI_CALL fk_CmdBeginRenderPass(VkCommandBuffer cb, const VkRenderPassBeginInfo *bi, VkSubpassContents c) { (void)c; pkt(cb, 13, bi, sizeof *bi, bi->pClearValues, bi->clearValueCount * sizeof(VkClearValue)); }
static VKAPI_ATTR void VKAPI_CALL fk_CmdEndRenderPass(VkCommandBuffer cb) { pkt(cb, 14, NULL, 0, NULL, 0); }
static VKAPI_ATTR void VKAPI_CALL fk_CmdBeginRendering(VkCommandBuffer cb, const VkRenderingInfo *ri) { pkt(cb, 15, ri, sizeof *ri, ri->pColorAttachments, ri->colorAttachmentCount * sizeof *ri->pColorAttachments); }
static VKAPI_ATTR void VKAPI_CALL fk_CmdEndRendering(VkCommandBuffer cb) { pkt(cb, 16, NULL, 0, NULL, 0); }
static VKAPI_ATTR void VKAPI_CALL fk_CmdBindPipeline(VkCommandBuffer cb, VkPipelineBindPoint bp, VkPipeline p) { void *a[2] = {(void *)(uintptr_t)bp, (void *)p}; pkt(cb, 17, a, sizeof a, NULL, 0); }
static VKAPI_ATTR void VKAPI_CALL fk_CmdBindDescriptorSets(VkCommandBuffer cb, VkPipelineBindPoint bp, VkPipelineLayout l, uint32_t first, uint32_t n, const VkDescriptorSet *s, uint32_t nd, const uint32_t *d) {
    (void)bp; (void)l; uint32_t a[4] = {first, n, nd, 0}; pkt(cb, 18, a, sizeof a, s, n * sizeof *s); if (nd) pkt(cb, 19, d, nd * 4, NULL, 0); atomic_fetch_add_explicit(&g_fk_binds, 1, memory_order_relaxed);
    if (g_check) { fk_cb *c = (fk_cb *)cb; for (uint32_t i = 0; i < n; i++) { fk_set *x = (fk_set *)s[i]; if (!x) continue; if (x->gen != x->pool->gen) fk_err("a set bound after its pool was reset", x);
        if (c->nsets == c->capsets) { c->capsets = c->capsets ? 2 * c->capsets : 1024; c->sets = realloc(c->sets, c->capsets * sizeof *c->sets); } c->sets[c->nsets++] = x; } } }
static VKAPI_ATTR void VKAPI_CALL fk_CmdBindVertexBuffers(VkCommandBuffer cb, uint32_t first, uint32_t n, const VkBuffer *b, const VkDeviceSize *o) { (void)o; uint32_t a[2] = {first, n}; pkt(cb, 20, a, sizeof a, b, n * sizeof *b); }
static VKAPI_ATTR void VKAPI_CALL fk_CmdBindIndexBuffer(VkCommandBuffer cb, VkBuffer b, VkDeviceSize o, VkIndexType t) { uint64_t a[3] = {(uint64_t)(uintptr_t)b, o, t}; pkt(cb, 21, a, sizeof a, NULL, 0); }
static VKAPI_ATTR void VKAPI_CALL fk_CmdSetViewport(VkCommandBuffer cb, uint32_t first, uint32_t n, const VkViewport *v) { (void)first; pkt(cb, 22, v, n * sizeof *v, NULL, 0); }
static VKAPI_ATTR void VKAPI_CALL fk_CmdSetScissor(VkCommandBuffer cb, uint32_t first, uint32_t n, const VkRect2D *r) { (void)first; pkt(cb, 23, r, n * sizeof *r, NULL, 0); }
static VKAPI_ATTR void VKAPI_CALL fk_CmdSetBlendConstants(VkCommandBuffer cb, const float c[4]) { pkt(cb, 24, c, 16, NULL, 0); }
static VKAPI_ATTR void VKAPI_CALL fk_CmdSetStencilReference(VkCommandBuffer cb, VkStencilFaceFlags f, uint32_t v) { uint32_t a[2] = {f, v}; pkt(cb, 25, a, sizeof a, NULL, 0); }
static VKAPI_ATTR void VKAPI_CALL fk_CmdSetStencilCompareMask(VkCommandBuffer cb, VkStencilFaceFlags f, uint32_t v) { uint32_t a[2] = {f, v}; pkt(cb, 26, a, sizeof a, NULL, 0); }
static VKAPI_ATTR void VKAPI_CALL fk_CmdSetStencilWriteMask(VkCommandBuffer cb, VkStencilFaceFlags f, uint32_t v) { uint32_t a[2] = {f, v}; pkt(cb, 27, a, sizeof a, NULL, 0); }
static VKAPI_ATTR void VKAPI_CALL fk_CmdSetDepthBias(VkCommandBuffer cb, float c, float cl, float s) { float a[3] = {c, cl, s}; pkt(cb, 28, a, sizeof a, NULL, 0); }
static VKAPI_ATTR void VKAPI_CALL fk_CmdSetLineWidth(VkCommandBuffer cb, float w) { pkt(cb, 29, &w, 4, NULL, 0); }
static VKAPI_ATTR void VKAPI_CALL fk_CmdDraw(VkCommandBuffer cb, uint32_t vc, uint32_t ic, uint32_t fv, uint32_t fi) { uint32_t a[4] = {vc, ic, fv, fi}; pkt(cb, 30, a, sizeof a, NULL, 0); atomic_fetch_add_explicit(&g_fk_draws, 1, memory_order_relaxed); }
static VKAPI_ATTR void VKAPI_CALL fk_CmdDrawIndexed(VkCommandBuffer cb, uint32_t ic, uint32_t n, uint32_t fi, int32_t vo, uint32_t fin) { uint32_t a[5] = {ic, n, fi, (uint32_t)vo, fin}; pkt(cb, 31, a, sizeof a, NULL, 0); atomic_fetch_add_explicit(&g_fk_draws, 1, memory_order_relaxed); }
static VKAPI_ATTR void VKAPI_CALL fk_CmdPushConstants(VkCommandBuffer cb, VkPipelineLayout l, VkShaderStageFlags st, uint32_t off, uint32_t n, const void *v) { (void)l; uint32_t a[2] = {st, off}; pkt(cb, 32, a, sizeof a, v, n); }
static VKAPI_ATTR void VKAPI_CALL fk_CmdDrawIndirect(VkCommandBuffer cb, VkBuffer b, VkDeviceSize o, uint32_t n, uint32_t s) { uint64_t a[3] = {(uint64_t)(uintptr_t)b, o, (uint64_t)n << 32 | s}; pkt(cb, 33, a, sizeof a, NULL, 0); atomic_fetch_add_explicit(&g_fk_draws, 1, memory_order_relaxed); }
static VKAPI_ATTR void VKAPI_CALL fk_CmdDrawIndexedIndirect(VkCommandBuffer cb, VkBuffer b, VkDeviceSize o, uint32_t n, uint32_t s) { uint64_t a[3] = {(uint64_t)(uintptr_t)b, o, (uint64_t)n << 32 | s}; pkt(cb, 34, a, sizeof a, NULL, 0); atomic_fetch_add_explicit(&g_fk_draws, 1, memory_order_relaxed); }
static VKAPI_ATTR void VKAPI_CALL fk_CmdDrawIndirectCount(VkCommandBuffer cb, VkBuffer b, VkDeviceSize o, VkBuffer c, VkDeviceSize co, uint32_t n, uint32_t s) { uint64_t a[5] = {(uint64_t)(uintptr_t)b, o, (uint64_t)(uintptr_t)c, co, (uint64_t)n << 32 | s}; pkt(cb, 35, a, sizeof a, NULL, 0); }
static VKAPI_ATTR void VKAPI_CALL fk_CmdDrawIndexedIndirectCount(VkCommandBuffer cb, VkBuffer b, VkDeviceSize o, VkBuffer c, VkDeviceSize co, uint32_t n, uint32_t s) { uint64_t a[5] = {(uint64_t)(uintptr_t)b, o, (uint64_t)(uintptr_t)c, co, (uint64_t)n << 32 | s}; pkt(cb, 36, a, sizeof a, NULL, 0); }
static VKAPI_ATTR void VKAPI_CALL fk_CmdDrawMeshTasksEXT(VkCommandBuffer cb, uint32_t x, uint32_t y, uint32_t z) { uint32_t a[3] = {x, y, z}; pkt(cb, 37, a, sizeof a, NULL, 0); atomic_fetch_add_explicit(&g_fk_draws, 1, memory_order_relaxed); }
static VKAPI_ATTR void VKAPI_CALL fk_CmdDispatch(VkCommandBuffer cb, uint32_t x, uint32_t y, uint32_t z) { uint32_t a[3] = {x, y, z}; pkt(cb, 38, a, sizeof a, NULL, 0); }
static VKAPI_ATTR void VKAPI_CALL fk_CmdDispatchBase(VkCommandBuffer cb, uint32_t bx, uint32_t by, uint32_t bz, uint32_t x, uint32_t y, uint32_t z) { uint32_t a[6] = {bx, by, bz, x, y, z}; pkt(cb, 39, a, sizeof a, NULL, 0); }
static VKAPI_ATTR void VKAPI_CALL fk_CmdDispatchIndirect(VkCommandBuffer cb, VkBuffer b, VkDeviceSize o) { uint64_t a[2] = {(uint64_t)(uintptr_t)b, o}; pkt(cb, 40, a, sizeof a, NULL, 0); }
static VKAPI_ATTR void VKAPI_CALL fk_CmdResetQueryPool(VkCommandBuffer cb, VkQueryPool p, uint32_t f, uint32_t n) { uint64_t a[2] = {(uint64_t)(uintptr_t)p, (uint64_t)f << 32 | n}; pkt(cb, 41, a, sizeof a, NULL, 0); }
static VKAPI_ATTR void VKAPI_CALL fk_CmdWriteTimestamp(VkCommandBuffer cb, VkPipelineStageFlagBits s, VkQueryPool p, uint32_t q) { uint64_t a[2] = {(uint64_t)(uintptr_t)p, (uint64_t)s << 32 | q}; pkt(cb, 42, a, sizeof a, NULL, 0); }
static VKAPI_ATTR void VKAPI_CALL fk_CmdBindTransformFeedbackBuffersEXT(VkCommandBuffer cb, uint32_t f, uint32_t n, const VkBuffer *b, const VkDeviceSize *o, const VkDeviceSize *s) { (void)o; (void)s; uint32_t a[2] = {f, n}; pkt(cb, 43, a, sizeof a, b, n * sizeof *b); }
static VKAPI_ATTR void VKAPI_CALL fk_CmdBeginTransformFeedbackEXT(VkCommandBuffer cb, uint32_t f, uint32_t n, const VkBuffer *b, const VkDeviceSize *o) { (void)b; (void)o; uint32_t a[2] = {f, n}; pkt(cb, 44, a, sizeof a, NULL, 0); }
static VKAPI_ATTR void VKAPI_CALL fk_CmdEndTransformFeedbackEXT(VkCommandBuffer cb, uint32_t f, uint32_t n, const VkBuffer *b, const VkDeviceSize *o) { (void)b; (void)o; uint32_t a[2] = {f, n}; pkt(cb, 45, a, sizeof a, NULL, 0); }
// VK_KHR_push_descriptor: the descriptors go into the command buffer, 64 bytes each, as vkUpdateDescriptorSets writes them into a set
static VKAPI_ATTR void VKAPI_CALL fk_CmdPushDescriptorSetKHR(VkCommandBuffer cb, VkPipelineBindPoint bp, VkPipelineLayout l, uint32_t set, uint32_t n, const VkWriteDescriptorSet *w) {
    (void)bp; (void)l; uint8_t d[64]; uint32_t a[2] = {set, n}; pkt(cb, 46, a, sizeof a, NULL, 0); atomic_fetch_add_explicit(&g_fk_pushdesc, 1, memory_order_relaxed);
    for (uint32_t i = 0; i < n; i++) for (uint32_t k = 0; k < w[i].descriptorCount; k++) {
        memset(d, 0, sizeof d);
        if (w[i].pImageInfo) { const VkDescriptorImageInfo *ii = &w[i].pImageInfo[k]; if (ii->imageView) memcpy(d, ((const fk_view *)ii->imageView)->hw, 48); if (ii->sampler) memcpy(d + 48, ((const fk_view *)ii->sampler)->hw, 16); }
        else if (w[i].pBufferInfo) { const VkDescriptorBufferInfo *bi = &w[i].pBufferInfo[k]; uint64_t ad = bi->buffer ? ((const fk_buf *)bi->buffer)->addr + bi->offset : 0; memcpy(d, &ad, 8); memcpy(d + 8, &bi->range, 8); }
        else if (w[i].pTexelBufferView) { if (w[i].pTexelBufferView[k]) memcpy(d, ((const fk_view *)w[i].pTexelBufferView[k])->hw, 32); }
        pkt(cb, 47, d, sizeof d, NULL, 0); atomic_fetch_add_explicit(&g_fk_desc_writes, 1, memory_order_relaxed); fk_spin(g_desc_ns); }
}
// extended dynamic state (VK_EXT_extended_dynamic_state 1-3, VK_EXT_vertex_input_dynamic_state): a packet each
static VKAPI_ATTR void VKAPI_CALL fk_CmdSetU32(VkCommandBuffer cb, uint32_t v) { pkt(cb, 48, &v, 4, NULL, 0); }
static VKAPI_ATTR void VKAPI_CALL fk_CmdSetVertexInputEXT(VkCommandBuffer cb, uint32_t nb, const VkVertexInputBindingDescription2EXT *b, uint32_t na, const VkVertexInputAttributeDescription2EXT *at) {
    uint32_t a[2] = {nb, na}; pkt(cb, 49, a, sizeof a, b, nb * sizeof *b); pkt(cb, 50, at, na * sizeof *at, NULL, 0); }
static VKAPI_ATTR void VKAPI_CALL fk_CmdBindVertexBuffers2(VkCommandBuffer cb, uint32_t first, uint32_t n, const VkBuffer *b, const VkDeviceSize *o, const VkDeviceSize *s, const VkDeviceSize *st) {
    (void)o; (void)s; (void)st; uint32_t a[2] = {first, n}; pkt(cb, 51, a, sizeof a, b, n * sizeof *b); }
static VKAPI_ATTR void VKAPI_CALL fk_CmdSetColorBlendEnableEXT(VkCommandBuffer cb, uint32_t f, uint32_t n, const VkBool32 *e) { (void)f; pkt(cb, 52, e, n * 4, NULL, 0); }
static VKAPI_ATTR void VKAPI_CALL fk_CmdSetColorWriteMaskEXT(VkCommandBuffer cb, uint32_t f, uint32_t n, const VkColorComponentFlags *m) { (void)f; pkt(cb, 53, m, n * 4, NULL, 0); }
static VKAPI_ATTR void VKAPI_CALL fk_CmdSetColorBlendEquationEXT(VkCommandBuffer cb, uint32_t f, uint32_t n, const VkColorBlendEquationEXT *e) { (void)f; pkt(cb, 54, e, n * sizeof *e, NULL, 0); }
static VKAPI_ATTR void VKAPI_CALL fk_CmdSetStencilOp(VkCommandBuffer cb, VkStencilFaceFlags fl, VkStencilOp f, VkStencilOp p, VkStencilOp z, VkCompareOp c) { uint32_t a[5] = {fl, f, p, z, c}; pkt(cb, 55, a, sizeof a, NULL, 0); }

// ---- descriptors ----
static VKAPI_ATTR VkResult VKAPI_CALL fk_CreateDescriptorSetLayout(VkDevice d, const VkDescriptorSetLayoutCreateInfo *ci, const VkAllocationCallbacks *a, VkDescriptorSetLayout *l) {
    (void)d; (void)a; fk_dsl *x = obj(sizeof *x); x->n = ci->bindingCount > 96 ? 96 : ci->bindingCount; uint32_t at = 0;
    for (uint32_t i = 0; i < x->n; i++) { x->b[i].binding = ci->pBindings[i].binding; x->b[i].type = ci->pBindings[i].descriptorType; x->b[i].count = ci->pBindings[i].descriptorCount; x->b[i].at = at; at += ci->pBindings[i].descriptorCount; }
    x->total = at; *l = (VkDescriptorSetLayout)x; return VK_SUCCESS; }
static VKAPI_ATTR VkResult VKAPI_CALL fk_CreatePipelineLayout(VkDevice d, const VkPipelineLayoutCreateInfo *ci, const VkAllocationCallbacks *a, VkPipelineLayout *l) { (void)d; (void)ci; (void)a; *l = (VkPipelineLayout)obj(16); return VK_SUCCESS; }
static VKAPI_ATTR VkResult VKAPI_CALL fk_CreateDescriptorPool(VkDevice d, const VkDescriptorPoolCreateInfo *ci, const VkAllocationCallbacks *a, VkDescriptorPool *p) {
    (void)d; (void)a; fk_dpool *x = obj(sizeof *x); size_t n = 0; for (uint32_t i = 0; i < ci->poolSizeCount; i++) n += ci->pPoolSizes[i].descriptorCount;
    x->cap = n * 64 + (size_t)ci->maxSets * sizeof(fk_set); x->arena = malloc(x->cap); if (!x->arena) return VK_ERROR_OUT_OF_HOST_MEMORY; *p = (VkDescriptorPool)x; return VK_SUCCESS; }
static VKAPI_ATTR VkResult VKAPI_CALL fk_ResetDescriptorPool(VkDevice d, VkDescriptorPool p, VkDescriptorPoolResetFlags f) { (void)d; (void)f; ((fk_dpool *)p)->used = 0; ((fk_dpool *)p)->gen++; return VK_SUCCESS; }
static VKAPI_ATTR VkResult VKAPI_CALL fk_AllocateDescriptorSets(VkDevice d, const VkDescriptorSetAllocateInfo *ai, VkDescriptorSet *s) {
    (void)d; fk_dpool *p = (fk_dpool *)ai->descriptorPool;
    for (uint32_t i = 0; i < ai->descriptorSetCount; i++) {
        const fk_dsl *l = (const fk_dsl *)ai->pSetLayouts[i]; size_t need = sizeof(fk_set) + (size_t)l->total * 64;
        if (p->used + need > p->cap) return VK_ERROR_OUT_OF_POOL_MEMORY;
        fk_set *x = (fk_set *)(p->arena + p->used); p->used += (need + 15) & ~(size_t)15; x->l = l; x->mem = (uint8_t *)(x + 1); x->pool = p; x->gen = p->gen; x->nref = 0; s[i] = (VkDescriptorSet)x;
        atomic_fetch_add_explicit(&g_fk_sets, 1, memory_order_relaxed); fk_spin(g_set_ns);
    }
    return VK_SUCCESS; }
static VKAPI_ATTR void VKAPI_CALL fk_UpdateDescriptorSets(VkDevice d, uint32_t nw, const VkWriteDescriptorSet *w, uint32_t nc, const VkCopyDescriptorSet *c) {
    (void)d; (void)nc; (void)c; atomic_fetch_add_explicit(&g_fk_updates, 1, memory_order_relaxed);
    for (uint32_t i = 0; i < nw; i++) {
        fk_set *s = (fk_set *)w[i].dstSet; const fk_dsl *l = s->l; uint32_t at = 0;
        if (g_check && s->gen != s->pool->gen) fk_err("vkUpdateDescriptorSets on a set whose pool was reset", s);
        for (uint32_t j = 0; j < l->n; j++) if (l->b[j].binding == w[i].dstBinding) { at = l->b[j].at; break; }
        for (uint32_t k = 0; k < w[i].descriptorCount; k++) {
            uint8_t *dd = s->mem + (size_t)(at + w[i].dstArrayElement + k) * 64;
            if (w[i].pImageInfo) { const VkDescriptorImageInfo *ii = &w[i].pImageInfo[k]; if (ii->imageView) memcpy(dd, ((const fk_view *)ii->imageView)->hw, 48); else memset(dd, 0, 48);
                if (ii->sampler) memcpy(dd + 48, ((const fk_view *)ii->sampler)->hw, 16); else memset(dd + 48, 0, 16);
                if (g_check) { if (ii->imageView && ((const fk_view *)ii->imageView)->dead) fk_err("a destroyed image view written into a set", (const void *)ii->imageView);
                    if (s->nref < 96 && ii->imageView) { s->ref[s->nref] = (void *)ii->imageView; s->refgen[s->nref] = ((const fk_view *)ii->imageView)->gen; s->refk[s->nref++] = 0; } } }
            else if (w[i].pBufferInfo) { const VkDescriptorBufferInfo *bi = &w[i].pBufferInfo[k]; uint64_t ad = bi->buffer ? ((const fk_buf *)bi->buffer)->addr + bi->offset : 0; memcpy(dd, &ad, 8); memcpy(dd + 8, &bi->range, 8); memset(dd + 16, 0, 48);
                if (g_check) { if (bi->buffer && ((const fk_buf *)bi->buffer)->dead) fk_err("a destroyed buffer written into a set", (const void *)bi->buffer);
                    if (s->nref < 96 && bi->buffer) { s->ref[s->nref] = (void *)bi->buffer; s->refgen[s->nref] = ((const fk_buf *)bi->buffer)->gen; s->refk[s->nref++] = 1; } } }
            else if (w[i].pTexelBufferView) { if (w[i].pTexelBufferView[k]) memcpy(dd, ((const fk_view *)w[i].pTexelBufferView[k])->hw, 64); else memset(dd, 0, 64); }
            atomic_fetch_add_explicit(&g_fk_desc_writes, 1, memory_order_relaxed); fk_spin(g_desc_ns);
        }
    }
}
// VK_KHR_descriptor_update_template (Vulkan 1.1 core): a set, or a push, filled from a struct by a template (the entries' offsets and strides)
typedef struct { uint32_t n; VkDescriptorUpdateTemplateEntry e[96]; VkDescriptorUpdateTemplateType type; uint32_t set; } fk_tmpl;
static VKAPI_ATTR VkResult VKAPI_CALL fk_CreateDescriptorUpdateTemplate(VkDevice d, const VkDescriptorUpdateTemplateCreateInfo *ci, const VkAllocationCallbacks *a, VkDescriptorUpdateTemplate *t) {
    (void)d; (void)a; fk_tmpl *x = obj(sizeof *x); x->n = ci->descriptorUpdateEntryCount > 96 ? 96 : ci->descriptorUpdateEntryCount; memcpy(x->e, ci->pDescriptorUpdateEntries, x->n * sizeof *x->e); x->type = ci->templateType; x->set = ci->set;
    *t = (VkDescriptorUpdateTemplate)x; return VK_SUCCESS; }
static VKAPI_ATTR void VKAPI_CALL fk_CmdPushDescriptorSetWithTemplateKHR(VkCommandBuffer cb, VkDescriptorUpdateTemplate t, VkPipelineLayout l, uint32_t set, const void *data) {
    (void)l; const fk_tmpl *x = (const fk_tmpl *)t; uint8_t d[64]; uint32_t a[2] = {set, x->n}; pkt(cb, 46, a, sizeof a, NULL, 0); atomic_fetch_add_explicit(&g_fk_pushdesc, 1, memory_order_relaxed);
    for (uint32_t i = 0; i < x->n; i++) for (uint32_t k = 0; k < x->e[i].descriptorCount; k++) {
        const uint8_t *src = (const uint8_t *)data + x->e[i].offset + (size_t)k * x->e[i].stride; memset(d, 0, sizeof d);
        VkDescriptorType ty = x->e[i].descriptorType;
        if (ty == VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER || ty == VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE || ty == VK_DESCRIPTOR_TYPE_STORAGE_IMAGE) { const VkDescriptorImageInfo *ii = (const void *)src;
            if (ii->imageView) memcpy(d, ((const fk_view *)ii->imageView)->hw, 48); if (ii->sampler) memcpy(d + 48, ((const fk_view *)ii->sampler)->hw, 16); }
        else if (ty == VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER || ty == VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER) { VkBufferView v; memcpy(&v, src, sizeof v); if (v) memcpy(d, ((const fk_view *)v)->hw, 32); }
        else { const VkDescriptorBufferInfo *bi = (const void *)src; uint64_t ad = bi->buffer ? ((const fk_buf *)bi->buffer)->addr + bi->offset : 0; memcpy(d, &ad, 8); memcpy(d + 8, &bi->range, 8); }
        pkt(cb, 47, d, sizeof d, NULL, 0); atomic_fetch_add_explicit(&g_fk_desc_writes, 1, memory_order_relaxed); fk_spin(g_desc_ns); }
}
static VKAPI_ATTR void VKAPI_CALL fk_DestroyDescriptorUpdateTemplate(VkDevice d, VkDescriptorUpdateTemplate t, const VkAllocationCallbacks *a) { (void)d; (void)a; free((void *)t); }

// ---- pipelines, shaders, passes ----
static VKAPI_ATTR VkResult VKAPI_CALL fk_CreateShaderModule(VkDevice d, const VkShaderModuleCreateInfo *ci, const VkAllocationCallbacks *a, VkShaderModule *m) { (void)d; (void)ci; (void)a; *m = (VkShaderModule)obj(16); return VK_SUCCESS; }
static VKAPI_ATTR void VKAPI_CALL fk_DestroyShaderModule(VkDevice d, VkShaderModule m, const VkAllocationCallbacks *a) { (void)d; (void)a; free((void *)m); }
static uint64_t g_pipe_build_ns;   // VKB_PIPE_US: a pipeline compile's time (0: none)
static VKAPI_ATTR VkResult VKAPI_CALL fk_CreateGraphicsPipelines(VkDevice d, VkPipelineCache c, uint32_t n, const VkGraphicsPipelineCreateInfo *ci, const VkAllocationCallbacks *a, VkPipeline *p) {
    (void)d; (void)c; (void)ci; (void)a; for (uint32_t i = 0; i < n; i++) { p[i] = (VkPipeline)obj(64); atomic_fetch_add_explicit(&g_fk_pipes, 1, memory_order_relaxed); fk_spin(g_pipe_build_ns); } return VK_SUCCESS; }
static VKAPI_ATTR VkResult VKAPI_CALL fk_CreateComputePipelines(VkDevice d, VkPipelineCache c, uint32_t n, const VkComputePipelineCreateInfo *ci, const VkAllocationCallbacks *a, VkPipeline *p) {
    (void)d; (void)c; (void)ci; (void)a; for (uint32_t i = 0; i < n; i++) { p[i] = (VkPipeline)obj(64); fk_spin(g_pipe_build_ns); } return VK_SUCCESS; }
static VKAPI_ATTR void VKAPI_CALL fk_DestroyPipeline(VkDevice d, VkPipeline p, const VkAllocationCallbacks *a) { (void)d; (void)a; free((void *)p); }
static VKAPI_ATTR VkResult VKAPI_CALL fk_CreatePipelineCache(VkDevice d, const VkPipelineCacheCreateInfo *ci, const VkAllocationCallbacks *a, VkPipelineCache *c) { (void)d; (void)ci; (void)a; *c = (VkPipelineCache)obj(8); return VK_SUCCESS; }
static VKAPI_ATTR VkResult VKAPI_CALL fk_GetPipelineCacheData(VkDevice d, VkPipelineCache c, size_t *n, void *p) { (void)d; (void)c; (void)p; *n = 0; return VK_SUCCESS; }
static VKAPI_ATTR VkResult VKAPI_CALL fk_CreateRenderPass(VkDevice d, const VkRenderPassCreateInfo *ci, const VkAllocationCallbacks *a, VkRenderPass *r) { (void)d; (void)ci; (void)a; *r = (VkRenderPass)obj(8); return VK_SUCCESS; }
static VKAPI_ATTR VkResult VKAPI_CALL fk_CreateRenderPass2(VkDevice d, const VkRenderPassCreateInfo2 *ci, const VkAllocationCallbacks *a, VkRenderPass *r) { (void)d; (void)ci; (void)a; *r = (VkRenderPass)obj(8); return VK_SUCCESS; }
static VKAPI_ATTR void VKAPI_CALL fk_DestroyRenderPass(VkDevice d, VkRenderPass r, const VkAllocationCallbacks *a) { (void)d; (void)a; free((void *)r); }
static VKAPI_ATTR VkResult VKAPI_CALL fk_CreateFramebuffer(VkDevice d, const VkFramebufferCreateInfo *ci, const VkAllocationCallbacks *a, VkFramebuffer *f) { (void)d; (void)ci; (void)a; *f = (VkFramebuffer)obj(8); return VK_SUCCESS; }
static VKAPI_ATTR void VKAPI_CALL fk_DestroyFramebuffer(VkDevice d, VkFramebuffer f, const VkAllocationCallbacks *a) { (void)d; (void)a; free((void *)f); }

// ---- synchronization and submission: the fake GPU completes each submission as it is submitted ----
static fk_sem *sem_new(int timeline) { fk_sem *s = obj(sizeof *s); pthread_mutex_init(&s->mu, NULL); pthread_cond_init(&s->cv, NULL); s->timeline = timeline; return s; }
static void sem_signal(fk_sem *s, uint64_t v) { pthread_mutex_lock(&s->mu); if (v > atomic_load(&s->v)) atomic_store(&s->v, v); pthread_cond_broadcast(&s->cv); pthread_mutex_unlock(&s->mu); }
static VKAPI_ATTR VkResult VKAPI_CALL fk_CreateSemaphore(VkDevice d, const VkSemaphoreCreateInfo *ci, const VkAllocationCallbacks *a, VkSemaphore *s) {
    (void)d; (void)a; int tl = 0; for (const VkBaseInStructure *x = ci->pNext; x; x = x->pNext) if (x->sType == VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO) tl = ((const VkSemaphoreTypeCreateInfo *)x)->semaphoreType == VK_SEMAPHORE_TYPE_TIMELINE;
    *s = (VkSemaphore)sem_new(tl); return VK_SUCCESS; }
static VKAPI_ATTR VkResult VKAPI_CALL fk_WaitSemaphores(VkDevice d, const VkSemaphoreWaitInfo *wi, uint64_t timeout) {
    (void)d; (void)timeout; for (uint32_t i = 0; i < wi->semaphoreCount; i++) { fk_sem *s = (fk_sem *)wi->pSemaphores[i]; pthread_mutex_lock(&s->mu); while (atomic_load(&s->v) < wi->pValues[i]) pthread_cond_wait(&s->cv, &s->mu); pthread_mutex_unlock(&s->mu); }
    return VK_SUCCESS; }
static VKAPI_ATTR VkResult VKAPI_CALL fk_GetSemaphoreCounterValue(VkDevice d, VkSemaphore s, uint64_t *v) { (void)d; *v = atomic_load(&((fk_sem *)s)->v); return VK_SUCCESS; }
static VKAPI_ATTR VkResult VKAPI_CALL fk_CreateFence(VkDevice d, const VkFenceCreateInfo *ci, const VkAllocationCallbacks *a, VkFence *f) { (void)d; (void)a; fk_sem *s = sem_new(0); if (ci->flags & VK_FENCE_CREATE_SIGNALED_BIT) s->v = 1; *f = (VkFence)s; return VK_SUCCESS; }
static VKAPI_ATTR VkResult VKAPI_CALL fk_WaitForFences(VkDevice d, uint32_t n, const VkFence *f, VkBool32 all, uint64_t t) { (void)d; (void)all; (void)t;
    for (uint32_t i = 0; i < n; i++) { fk_sem *s = (fk_sem *)f[i]; pthread_mutex_lock(&s->mu); while (!atomic_load(&s->v)) pthread_cond_wait(&s->cv, &s->mu); pthread_mutex_unlock(&s->mu); } return VK_SUCCESS; }
static VKAPI_ATTR VkResult VKAPI_CALL fk_ResetFences(VkDevice d, uint32_t n, const VkFence *f) { (void)d; for (uint32_t i = 0; i < n; i++) atomic_store(&((fk_sem *)f[i])->v, 0); return VK_SUCCESS; }
static VKAPI_ATTR void VKAPI_CALL fk_DestroyFence(VkDevice d, VkFence f, const VkAllocationCallbacks *a) { (void)d; (void)a; (void)f; }
static VKAPI_ATTR VkResult VKAPI_CALL fk_QueueSubmit(VkQueue q, uint32_t n, const VkSubmitInfo *si, VkFence f) {
    (void)q;
    if (g_check) for (uint32_t i = 0; i < n; i++) for (uint32_t c = 0; c < si[i].commandBufferCount; c++) { fk_cb *cb = (fk_cb *)si[i].pCommandBuffers[c];
        for (uint32_t k = 0; k < cb->nsets; k++) { fk_set *x = cb->sets[k]; if (x->gen != x->pool->gen) { fk_err("a submitted command buffer binds a set whose pool was reset", x); continue; }
            for (uint32_t r = 0; r < x->nref; r++) { int dead = x->refk[r] ? ((fk_buf *)x->ref[r])->dead || ((fk_buf *)x->ref[r])->gen != x->refgen[r] : ((fk_view *)x->ref[r])->dead || ((fk_view *)x->ref[r])->gen != x->refgen[r];
                if (dead) fk_err(x->refk[r] ? "a submitted set names a destroyed buffer (or its handle given to another)" : "a submitted set names a destroyed image view (or its handle given to another)", x->ref[r]); } } }
    for (uint32_t i = 0; i < n; i++) {
        const VkTimelineSemaphoreSubmitInfo *ts = NULL; for (const VkBaseInStructure *x = si[i].pNext; x; x = x->pNext) if (x->sType == VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO) ts = (const void *)x;
        for (uint32_t k = 0; k < si[i].signalSemaphoreCount; k++) sem_signal((fk_sem *)si[i].pSignalSemaphores[k], ts && k < ts->signalSemaphoreValueCount ? ts->pSignalSemaphoreValues[k] : 1);
    }
    if (f) sem_signal((fk_sem *)f, 1);
    return VK_SUCCESS; }
static VKAPI_ATTR VkResult VKAPI_CALL fk_QueueWaitIdle(VkQueue q) { (void)q; return VK_SUCCESS; }
static VKAPI_ATTR VkResult VKAPI_CALL fk_DeviceWaitIdle(VkDevice d) { (void)d; return VK_SUCCESS; }
static VKAPI_ATTR VkResult VKAPI_CALL fk_QueueBindSparse(VkQueue q, uint32_t n, const VkBindSparseInfo *b, VkFence f) { (void)q; (void)n; (void)b; if (f) sem_signal((fk_sem *)f, 1); return VK_SUCCESS; }
static VKAPI_ATTR VkResult VKAPI_CALL fk_CreateQueryPool(VkDevice d, const VkQueryPoolCreateInfo *ci, const VkAllocationCallbacks *a, VkQueryPool *p) { (void)d; (void)ci; (void)a; *p = (VkQueryPool)obj(8); return VK_SUCCESS; }
static VKAPI_ATTR VkResult VKAPI_CALL fk_GetQueryPoolResults(VkDevice d, VkQueryPool p, uint32_t f, uint32_t n, size_t sz, void *data, VkDeviceSize st, VkQueryResultFlags fl) {
    (void)d; (void)p; (void)f; (void)fl; uint64_t t = fk_now(); for (uint32_t i = 0; i < n && (i + 1) * st <= sz; i++) memcpy((uint8_t *)data + i * st, &t, 8); return VK_SUCCESS; }
static VKAPI_ATTR VkResult VKAPI_CALL fk_CreateSwapchainKHR(VkDevice d, const VkSwapchainCreateInfoKHR *ci, const VkAllocationCallbacks *a, VkSwapchainKHR *s) { (void)d; (void)ci; (void)a; *s = (VkSwapchainKHR)obj(8); return VK_SUCCESS; }
static VKAPI_ATTR void VKAPI_CALL fk_DestroySwapchainKHR(VkDevice d, VkSwapchainKHR s, const VkAllocationCallbacks *a) { (void)d; (void)s; (void)a; }
static VKAPI_ATTR VkResult VKAPI_CALL fk_GetSwapchainImagesKHR(VkDevice d, VkSwapchainKHR s, uint32_t *n, VkImage *i) { (void)d; (void)s; (void)i; *n = 0; return VK_SUCCESS; }
static VKAPI_ATTR VkResult VKAPI_CALL fk_AcquireNextImageKHR(VkDevice d, VkSwapchainKHR s, uint64_t t, VkSemaphore se, VkFence f, uint32_t *i) { (void)d; (void)s; (void)t; (void)se; (void)f; *i = 0; return VK_SUCCESS; }
static VKAPI_ATTR VkResult VKAPI_CALL fk_QueuePresentKHR(VkQueue q, const VkPresentInfoKHR *pi) { (void)q; (void)pi; return VK_SUCCESS; }
static VKAPI_ATTR VkResult VKAPI_CALL fk_GetRefreshCycleDurationGOOGLE(VkDevice d, VkSwapchainKHR s, VkRefreshCycleDurationGOOGLE *r) { (void)d; (void)s; r->refreshDuration = 4166666; return VK_SUCCESS; }
static VKAPI_ATTR VkResult VKAPI_CALL fk_GetPastPresentationTimingGOOGLE(VkDevice d, VkSwapchainKHR s, uint32_t *n, VkPastPresentationTimingGOOGLE *t) { (void)d; (void)s; (void)t; *n = 0; return VK_SUCCESS; }

// ---- the loader's entry points ----
static const struct { const char *name; PFN_vkVoidFunction fn; } FNS[] = {
#define F(n) {"vk" #n, (PFN_vkVoidFunction)fk_##n},
    F(CreateInstance) F(DestroyInstance) F(EnumeratePhysicalDevices) F(GetPhysicalDeviceProperties) F(GetPhysicalDeviceProperties2) F(GetPhysicalDeviceFeatures2) F(GetPhysicalDeviceMemoryProperties)
    F(GetPhysicalDeviceMemoryProperties2) F(GetPhysicalDeviceQueueFamilyProperties) F(GetPhysicalDeviceFormatProperties) F(GetPhysicalDeviceFormatProperties2) F(GetPhysicalDeviceImageFormatProperties)
    F(GetPhysicalDeviceSparseImageFormatProperties) F(CreateDevice) F(EnumerateDeviceExtensionProperties) F(GetPhysicalDeviceSurfaceCapabilitiesKHR) F(GetPhysicalDeviceSurfaceFormatsKHR)
    F(GetPhysicalDeviceSurfacePresentModesKHR) F(CreateAndroidSurfaceKHR) F(DestroySurfaceKHR) F(GetPhysicalDeviceSurfaceSupportKHR)
    F(GetDeviceQueue) F(CreateBuffer) F(DestroyBuffer) F(AllocateMemory) F(FreeMemory) F(BindBufferMemory) F(BindImageMemory) F(GetBufferMemoryRequirements) F(GetImageMemoryRequirements)
    F(CreateImage) F(DestroyImage) F(CreateImageView) F(DestroyImageView) F(CreateSampler) F(DestroySampler) F(MapMemory) F(CreateCommandPool) F(ResetCommandPool) F(ResetCommandBuffer)
    F(AllocateCommandBuffers) F(BeginCommandBuffer) F(EndCommandBuffer) F(QueueSubmit) F(QueueWaitIdle) F(DeviceWaitIdle) F(CreateSemaphore) F(WaitSemaphores) F(GetSemaphoreCounterValue)
    F(CreateFence) F(WaitForFences) F(ResetFences) F(DestroyFence) F(CmdPipelineBarrier) F(CmdCopyBuffer) F(CmdCopyBufferToImage) F(CmdCopyImageToBuffer) F(CmdCopyImage) F(CmdBlitImage)
    F(CmdFillBuffer) F(CmdUpdateBuffer) F(CmdClearColorImage) F(CmdClearDepthStencilImage) F(CmdClearAttachments) F(CmdResolveImage) F(CreateRenderPass) F(CreateRenderPass2) F(DestroyRenderPass)
    F(CreateFramebuffer) F(DestroyFramebuffer) F(CmdBeginRenderPass) F(CmdEndRenderPass) F(CmdBeginRendering) F(CmdEndRendering) F(CmdBindPipeline) F(CmdBindDescriptorSets) F(CmdBindVertexBuffers)
    F(CmdBindIndexBuffer) F(CmdSetViewport) F(CmdSetScissor) F(CmdSetBlendConstants) F(CmdSetStencilReference) F(CmdSetStencilCompareMask) F(CmdSetStencilWriteMask) F(CmdSetDepthBias)
    F(CmdSetLineWidth) F(CmdDraw) F(CmdPushConstants) F(CmdDrawIndexed) F(CmdDrawIndirect) F(CmdDrawIndexedIndirect) F(CmdDispatch) F(CmdDispatchIndirect) F(CreateShaderModule) F(DestroyShaderModule)
    F(CreateDescriptorSetLayout) F(CreatePipelineLayout) F(CreateGraphicsPipelines) F(DestroyPipeline) F(CreateComputePipelines) F(CreatePipelineCache) F(GetPipelineCacheData)
    F(CreateDescriptorPool) F(ResetDescriptorPool) F(AllocateDescriptorSets) F(UpdateDescriptorSets) F(CreateBufferView) F(DestroyBufferView) F(GetAndroidHardwareBufferPropertiesANDROID)
    F(CreateQueryPool) F(CmdResetQueryPool) F(CmdWriteTimestamp) F(GetQueryPoolResults) F(CreateSwapchainKHR) F(DestroySwapchainKHR) F(GetSwapchainImagesKHR) F(AcquireNextImageKHR) F(QueuePresentKHR)
    F(CmdBindTransformFeedbackBuffersEXT) F(CmdBeginTransformFeedbackEXT) F(CmdEndTransformFeedbackEXT) F(GetRefreshCycleDurationGOOGLE) F(GetPastPresentationTimingGOOGLE) F(QueueBindSparse)
    F(GetImageSparseMemoryRequirements) F(CmdDrawIndirectCount) F(CmdDrawIndexedIndirectCount) F(CmdDrawMeshTasksEXT) F(CmdDispatchBase)
    F(CmdPushDescriptorSetKHR) F(CmdPushDescriptorSetWithTemplateKHR) F(CreateDescriptorUpdateTemplate) F(DestroyDescriptorUpdateTemplate) F(CmdSetVertexInputEXT)
    {"vkCmdBindVertexBuffers2", (PFN_vkVoidFunction)fk_CmdBindVertexBuffers2}, {"vkCmdBindVertexBuffers2EXT", (PFN_vkVoidFunction)fk_CmdBindVertexBuffers2},
    {"vkCmdSetColorBlendEnableEXT", (PFN_vkVoidFunction)fk_CmdSetColorBlendEnableEXT}, {"vkCmdSetColorWriteMaskEXT", (PFN_vkVoidFunction)fk_CmdSetColorWriteMaskEXT},
    {"vkCmdSetColorBlendEquationEXT", (PFN_vkVoidFunction)fk_CmdSetColorBlendEquationEXT}, {"vkCmdSetStencilOp", (PFN_vkVoidFunction)fk_CmdSetStencilOp}, {"vkCmdSetStencilOpEXT", (PFN_vkVoidFunction)fk_CmdSetStencilOp},
#undef F
};
// the one-word dynamic states (cull mode, front face, topology, depth test / write / compare, stencil test, depth bias / restart enables, rasterizer discard, polygon mode, logic op)
static const char *U32S[] = {"vkCmdSetCullMode", "vkCmdSetFrontFace", "vkCmdSetPrimitiveTopology", "vkCmdSetDepthTestEnable", "vkCmdSetDepthWriteEnable", "vkCmdSetDepthCompareOp", "vkCmdSetDepthBoundsTestEnable",
    "vkCmdSetStencilTestEnable", "vkCmdSetRasterizerDiscardEnable", "vkCmdSetDepthBiasEnable", "vkCmdSetPrimitiveRestartEnable", "vkCmdSetPolygonModeEXT", "vkCmdSetLogicOpEXT", "vkCmdSetDepthClampEnableEXT",
    "vkCmdSetAlphaToCoverageEnableEXT", "vkCmdSetRasterizationSamplesEXT", "vkCmdSetLogicOpEnableEXT"};
static PFN_vkVoidFunction fk_proc(const char *n) {
    if (!n) return NULL;
    for (size_t i = 0; i < sizeof FNS / sizeof *FNS; i++) if (!strcmp(FNS[i].name, n)) return FNS[i].fn;
    for (size_t i = 0; i < sizeof U32S / sizeof *U32S; i++) { size_t l = strlen(U32S[i]); if (!strncmp(U32S[i], n, l) && (!n[l] || !strcmp(n + l, "EXT"))) return (PFN_vkVoidFunction)fk_CmdSetU32; }
    return NULL;
}
static VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL fk_GetDeviceProcAddr(VkDevice d, const char *n) { (void)d; return fk_proc(n); }
VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL vkb_GetInstanceProcAddr(VkInstance i, const char *n) {
    (void)i; if (n && !strcmp(n, "vkGetDeviceProcAddr")) return (PFN_vkVoidFunction)fk_GetDeviceProcAddr;
    if (n && !strcmp(n, "vkGetInstanceProcAddr")) return (PFN_vkVoidFunction)vkb_GetInstanceProcAddr;
    return fk_proc(n);
}
// the shim's dlopen / dlsym (vkbench_force.h): "libvulkan.so" is this driver; anything else goes to the system's
static int g_fk_handle;
void *vkb_dlopen(const char *path, int mode) {
    if (path && strstr(path, "libvulkan")) {
        const char *e; if ((e = getenv("VKB_CMD_NS"))) g_cmd_ns = strtoull(e, NULL, 10); if ((e = getenv("VKB_DESC_NS"))) g_desc_ns = strtoull(e, NULL, 10);
        if ((e = getenv("VKB_SET_NS"))) g_set_ns = strtoull(e, NULL, 10); if ((e = getenv("VKB_PIPE_US"))) g_pipe_build_ns = strtoull(e, NULL, 10) * 1000;
        g_check = (e = getenv("VKB_CHECK")) && atoi(e);
        return &g_fk_handle; }
    if (path && strstr(path, "libadrenotools")) return NULL;
    return dlopen(path, mode);
}
void *vkb_dlsym(void *h, const char *name) {
    if (h == &g_fk_handle) return !strcmp(name, "vkGetInstanceProcAddr") ? (void *)vkb_GetInstanceProcAddr : NULL;
    return dlsym(h, name);
}
