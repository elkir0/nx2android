// Android port, Vulkan backend: device, memory pools, submissions, completion (see gfx_vk.h).
//   Memory pools: NVN gives the GPU the game's own memory (GPU address = CPU address). Each pool's pages are replaced by an AHardwareBuffer mapped at the same
//   address (its dma-buf mmapped MAP_FIXED, contents copied over), and the same AHardwareBuffer is imported as a VkBuffer: the game's pointer is GPU memory,
//   CPU-cached and coherent on Adreno (measured: probes/vkzc.c). A pool the allocator refuses keeps its anonymous pages and gets a GPU copy refreshed before
//   every submission (vk_pools_sync), correct but slow, reported once.
//   Submissions: a ring of slots, each with a command pool (upload + main command buffers), descriptor pools, scratch memory and objects freed once the GPU has
//   passed it. A submission signals a timeline semaphore with its fence; a completion thread waits for it and then applies what the CPU may see of its work
//   (event writes, counter reports), as nvn_mtl.m does in its completion handler.
#define _GNU_SOURCE
#include "gfx_vk.h"
#include "gfx_vk_vt.h"
#include "nvn_hmap.h"
#include "nvn_dirty.h"
#include "nvn_timestamp.h"
#include "gfx_vk_shadow.h"
#include "gfx_vk_budget.h"
#include <android/hardware_buffer.h>
#include <dlfcn.h>
#include <errno.h>
#include <stdarg.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/socket.h>
#include <time.h>
#include <unistd.h>

#define VK_DEF(f) PFN_##f f;
VK_FNS(VK_DEF) VK_DEV_FNS(VK_DEF)
static PFN_vkGetInstanceProcAddr p_vkGetInstanceProcAddr;
vkdev_t V;
pthread_mutex_t g_vk_queue_mu = PTHREAD_MUTEX_INITIALIZER;   // VkQueue access is externally synchronized: submissions and presents (and the sparse binds, gfx_vk_tex.c, on q or sq)
static int g_vk_ready;   // vk_ready
// the live diagnostics' settings (gfx_vk_exec.c x_read): the writers' watched range (vk_gpu_wrote), the reccap line cap (completion_main) and the pass its lines name
extern uint64_t g_x_watch_lo, g_x_watch_hi; extern int g_x_writers_on;
extern uint64_t g_x_reccap_lines; extern uint32_t g_x_reccap_dropped; extern int g_x_reccap_capped;
int vk_sub_pass_now(void);   // the pass being replayed (gfx_vk_exec.c)

uint32_t vk_mem_type(uint32_t bits, VkMemoryPropertyFlags want, VkMemoryPropertyFlags avoid) {
    for (uint32_t i = 0; i < V.mem.memoryTypeCount; i++)
        if ((bits >> i & 1) && (V.mem.memoryTypes[i].propertyFlags & want) == want && !(V.mem.memoryTypes[i].propertyFlags & avoid)) return i;
    for (uint32_t i = 0; i < V.mem.memoryTypeCount; i++) if ((bits >> i & 1) && (V.mem.memoryTypes[i].propertyFlags & want) == want) return i;
    return UINT32_MAX;
}

static int has_ext(const VkExtensionProperties *e, uint32_t n, const char *name) { for (uint32_t i = 0; i < n; i++) if (!strcmp(e[i].extensionName, name)) return 1; return 0; }
// The GPU's clock (PARITY G9): Mario Kart World switches its present interval between 1 and 2 about 1,400 times a race, and the GPU's governor, reading the idle time of the frames at
// interval 2, keeps the clock low through the frames at 1 (the GPU's 34 ms a frame in run 13 was measured so). VK_QCOM_queue_perf_hint asks the driver for a clock on the queue:
// DM_VK_PERF_HINT (live, read at each submission; set only when it changes): max (the highest clock), min, scaled:N (VK_PERF_HINT_TYPE_FREQUENCY_SCALED_QCOM with scale N, the driver's
// unit), default (the driver's own governor again). Unset: never asked (an A/B against DM_GPU_TURBO's KGSL power control, perf_android.c, before it is a default: heat and battery)
#ifndef VK_QCOM_queue_perf_hint   // (the NDK's headers predate it: the registry's definitions, as third_party/vulkan-headers' vulkan_core.h has them)
#define VK_QCOM_queue_perf_hint 1
typedef enum VkPerfHintTypeQCOM { VK_PERF_HINT_TYPE_DEFAULT_QCOM = 0, VK_PERF_HINT_TYPE_FREQUENCY_MIN_QCOM = 1, VK_PERF_HINT_TYPE_FREQUENCY_MAX_QCOM = 2, VK_PERF_HINT_TYPE_FREQUENCY_SCALED_QCOM = 3,
                                  VK_PERF_HINT_TYPE_MAX_ENUM_QCOM = 0x7FFFFFFF } VkPerfHintTypeQCOM;
typedef struct VkPerfHintInfoQCOM { VkStructureType sType; void *pNext; VkPerfHintTypeQCOM type; uint32_t scale; } VkPerfHintInfoQCOM;
typedef struct VkPhysicalDeviceQueuePerfHintFeaturesQCOM { VkStructureType sType; void *pNext; VkBool32 queuePerfHint; } VkPhysicalDeviceQueuePerfHintFeaturesQCOM;
typedef struct VkPhysicalDeviceQueuePerfHintPropertiesQCOM { VkStructureType sType; void *pNext; VkQueueFlags supportedQueues; } VkPhysicalDeviceQueuePerfHintPropertiesQCOM;
typedef VkResult (VKAPI_PTR *PFN_vkQueueSetPerfHintQCOM)(VkQueue queue, const VkPerfHintInfoQCOM *pPerfHintInfo);
#define VK_STRUCTURE_TYPE_PERF_HINT_INFO_QCOM ((VkStructureType)1000302000)
#define VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_QUEUE_PERF_HINT_FEATURES_QCOM ((VkStructureType)1000302001)
#define VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_QUEUE_PERF_HINT_PROPERTIES_QCOM ((VkStructureType)1000302002)
#endif
static int g_has_perf_hint; static PFN_vkQueueSetPerfHintQCOM p_set_perf_hint;
static void perf_hint_apply(void) {   // (mtl_sub_commit, before the queue's lock is taken)
    static char last[24]; static int asked; if (!g_has_perf_hint) return;
    const char *e = getenv("DM_VK_PERF_HINT"); if (!e) { if (!asked) return; e = "default"; }
    if (!strncmp(e, last, sizeof last - 1)) return;
    snprintf(last, sizeof last, "%s", e); asked = 1;
    VkPerfHintInfoQCOM h = {VK_STRUCTURE_TYPE_PERF_HINT_INFO_QCOM, NULL, VK_PERF_HINT_TYPE_DEFAULT_QCOM, 0};
    if (!strncmp(e, "max", 3) || !strcmp(e, "2")) h.type = VK_PERF_HINT_TYPE_FREQUENCY_MAX_QCOM;
    else if (!strncmp(e, "min", 3) || !strcmp(e, "1")) h.type = VK_PERF_HINT_TYPE_FREQUENCY_MIN_QCOM;
    else if (!strncmp(e, "scaled", 6) || !strncmp(e, "3", 1)) { const char *c = strchr(e, ':'); h.type = VK_PERF_HINT_TYPE_FREQUENCY_SCALED_QCOM; h.scale = c ? (uint32_t)strtoul(c + 1, NULL, 10) : 100; }
    pthread_mutex_lock(&g_vk_queue_mu); VkResult r = p_set_perf_hint(V.q, &h); pthread_mutex_unlock(&g_vk_queue_mu);
    VKLOG("%sDM_VK_PERF_HINT=%s: the GPU clock hint %s (type %d, scale %u): %d", r ? "[RED] " : "", e, r ? "refused" : "set", (int)h.type, h.scale, (int)r);
}
static int is_on(const char *const *on, uint32_t n, const char *name) { for (uint32_t i = 0; i < n; i++) if (!strcmp(on[i], name)) return 1; return 0; }
// a feature struct as vk_init queries it (sType, pNext, then VkBool32s; the largest, descriptor indexing, has 20); featx: its VkBool32s, all off if it is not chained
typedef struct { VkBaseOutStructure h; VkBool32 b[24]; } featx_t;
static VkBool32 *featx(featx_t *x, uint32_t n, VkStructureType s) { static VkBool32 off[24]; for (uint32_t i = 0; i < n; i++) if (x[i].h.sType == s) return x[i].b; return off; }   // (off: only 0 is written to it)

static void completion_start(void);
static void slots_init(void);
static VKAPI_ATTR VkBool32 VKAPI_CALL vk_validation_msg(VkDebugUtilsMessageSeverityFlagBitsEXT sev, VkDebugUtilsMessageTypeFlagsEXT type, const VkDebugUtilsMessengerCallbackDataEXT *d, void *u) {
    (void)type; (void)u;   // each distinct message ID logged its first 3 times
    enum { NID = 512 }; static int32_t ids[NID]; static uint8_t cnt[NID]; static int nid; static pthread_mutex_t mu = PTHREAD_MUTEX_INITIALIZER;
    pthread_mutex_lock(&mu); int k = 0; for (; k < nid; k++) if (ids[k] == d->messageIdNumber) break;
    if (k == nid && nid < NID) ids[nid++] = d->messageIdNumber;
    int show = k < NID && cnt[k] < 3; if (k < NID && cnt[k] < 255) cnt[k]++; pthread_mutex_unlock(&mu);
    if (show) VKLOG("[RED] validation %s: %s", sev & VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT ? "error" : "warning", d->pMessage);
    return VK_FALSE;
}

// Whether a driver offers a Vulkan 1.3 GPU (DM_VK_DRIVER: a Turnip built for another Adreno generation loads and then has none)
static int drv_has_gpu(PFN_vkGetInstanceProcAddr g) {
    PFN_vkCreateInstance ci = (PFN_vkCreateInstance)g(NULL, "vkCreateInstance"); VkInstance in; int ok = 0;
    VkApplicationInfo app = {VK_STRUCTURE_TYPE_APPLICATION_INFO, NULL, "dmhost", 1, "nxnative", 1, VK_API_VERSION_1_3};
    VkInstanceCreateInfo ici = {VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, NULL, 0, &app, 0, NULL, 0, NULL};
    if (!ci || ci(&ici, NULL, &in)) return 0;
    PFN_vkEnumeratePhysicalDevices ep = (PFN_vkEnumeratePhysicalDevices)g(in, "vkEnumeratePhysicalDevices");
    PFN_vkGetPhysicalDeviceProperties gp = (PFN_vkGetPhysicalDeviceProperties)g(in, "vkGetPhysicalDeviceProperties");
    PFN_vkDestroyInstance di = (PFN_vkDestroyInstance)g(in, "vkDestroyInstance");
    VkPhysicalDevice pd[8]; uint32_t n = 8;
    if (ep && gp && ep(in, &n, pd) >= 0) for (uint32_t i = 0; i < n && !ok; i++) { VkPhysicalDeviceProperties p; gp(pd[i], &p); ok = p.apiVersion >= VK_API_VERSION_1_3;
        VKLOG("DM_VK_DRIVER: %s, Vulkan %u.%u, driver version 0x%x", p.deviceName, VK_API_VERSION_MAJOR(p.apiVersion), VK_API_VERSION_MINOR(p.apiVersion), p.driverVersion); }
    if (di) di(in, NULL);
    return ok;
}

// The facts of the device a Switch 2 game depends on, logged once at vk_init so that a first run on a phone reads them: Mario Kart World's compute shaders declare 48 KB of shared
// memory, it draws through task and mesh shaders and indirect-count multi-draws, pushes NVN2 constants (0x100 bytes a block), and keeps sparse 8192x8192x68 BC6H and ASTC 8x8
// terrain arrays (the Galaxy Z Fold 8 Ultra has 16 GB, which those arrays made whole would exceed)
static int fcb_nofmt;   // customBorderColorWithoutFormat: gfx_vk_tex.c make_sampler gives its custom border colours no format (VK_FORMAT_UNDEFINED), which takes it
static void device_facts(const VkExtensionProperties *ex, uint32_t ne, int v12, int v13, int sizectl, int fullsg, const VkPhysicalDeviceMeshShaderFeaturesEXT *fms) {
    const VkPhysicalDeviceLimits *L = &V.props.limits; const VkPhysicalDeviceSparseProperties *SP = &V.props.sparseProperties;
    VkPhysicalDeviceSubgroupProperties sg = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES};
    VkPhysicalDeviceSubgroupSizeControlProperties sz = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_PROPERTIES};
    VkPhysicalDeviceMeshShaderPropertiesEXT ms = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MESH_SHADER_PROPERTIES_EXT};
    VkPhysicalDeviceBlendOperationAdvancedPropertiesEXT ba = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BLEND_OPERATION_ADVANCED_PROPERTIES_EXT};
    VkPhysicalDeviceDriverProperties dp = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRIVER_PROPERTIES};
    VkPhysicalDeviceProperties2 p2 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, &sg}; void **tail = &sg.pNext;   // (a struct only where its extension is)
    if (v13 || has_ext(ex, ne, "VK_EXT_subgroup_size_control")) { *tail = &sz; tail = &sz.pNext; }
    if (has_ext(ex, ne, "VK_EXT_mesh_shader")) { *tail = &ms; tail = &ms.pNext; }
    if (has_ext(ex, ne, "VK_EXT_blend_operation_advanced")) { *tail = &ba; tail = &ba.pNext; }
    if (v12 || has_ext(ex, ne, "VK_KHR_driver_properties")) { *tail = &dp; tail = &dp.pNext; }
    vkGetPhysicalDeviceProperties2(V.pd, &p2);
    V.task_max_groups = ms.maxTaskWorkGroupCount[0] < ms.maxTaskWorkGroupTotalCount ? ms.maxTaskWorkGroupCount[0] : ms.maxTaskWorkGroupTotalCount;
    V.mesh_max_groups = ms.maxMeshWorkGroupCount[0] < ms.maxMeshWorkGroupTotalCount ? ms.maxMeshWorkGroupCount[0] : ms.maxMeshWorkGroupTotalCount;
    if (!fms->taskShader) V.task_max_groups = 0;   // (gfx_vk_exec.c mesh_draw: a program with a task shader is then skipped)
    V.sg_stages = sg.supportedStages; V.sg_ops = sg.supportedOperations;   // (gfx_vk_exec.c mesh_need: a task / mesh shader's subgroup operations)
    V.ml.task_smem = ms.maxTaskSharedMemorySize; V.ml.mesh_smem = ms.maxMeshSharedMemorySize; V.ml.task_payload = ms.maxTaskPayloadSize; V.ml.task_pl_smem = ms.maxTaskPayloadAndSharedMemorySize;
    V.ml.mesh_pl_smem = ms.maxMeshPayloadAndSharedMemorySize; V.ml.out_verts = ms.maxMeshOutputVertices; V.ml.out_prims = ms.maxMeshOutputPrimitives; V.ml.task_inv = ms.maxTaskWorkGroupInvocations;
    V.ml.mesh_inv = ms.maxMeshWorkGroupInvocations; memcpy(V.ml.task_size, ms.maxTaskWorkGroupSize, sizeof V.ml.task_size); memcpy(V.ml.mesh_size, ms.maxMeshWorkGroupSize, sizeof V.ml.mesh_size);
    V.adv_npm = ba.advancedBlendNonPremultipliedSrcColor; V.adv_ovl = ba.advancedBlendCorrelatedOverlap; V.adv_all = ba.advancedBlendAllOperations; V.adv_max_att = ba.advancedBlendMaxColorAttachments;
    if (dp.driverName[0]) VKLOG("device facts: driver %s (%s), conformance %u.%u.%u", dp.driverName, dp.driverInfo, dp.conformanceVersion.major, dp.conformanceVersion.minor, dp.conformanceVersion.subminor);
    VKLOG("device facts: compute shared memory %u bytes, workgroup %u invocations (%u x %u x %u), subgroup size %u (stages %#x, operations %#x), subgroup size control %s (%u to %u, required-size stages %#x, "
          "full subgroups %s), push constants %u bytes, descriptor sets %u", L->maxComputeSharedMemorySize, L->maxComputeWorkGroupInvocations, L->maxComputeWorkGroupSize[0], L->maxComputeWorkGroupSize[1],
          L->maxComputeWorkGroupSize[2], sg.subgroupSize, sg.supportedStages, sg.supportedOperations, sizectl ? "yes" : "no", sz.minSubgroupSize, sz.maxSubgroupSize, sz.requiredSubgroupSizeStages,
          fullsg ? "yes" : "no", L->maxPushConstantsSize, L->maxBoundDescriptorSets);
    VKLOG("device facts: per stage %u samplers, %u uniform buffers, %u storage buffers, %u sampled images, %u storage images; uniform range %u, storage range %u; images 2D %u, 3D %u, %u layers; "
          "%u memory allocations, %u samplers", L->maxPerStageDescriptorSamplers, L->maxPerStageDescriptorUniformBuffers, L->maxPerStageDescriptorStorageBuffers, L->maxPerStageDescriptorSampledImages,
          L->maxPerStageDescriptorStorageImages, L->maxUniformBufferRange, L->maxStorageBufferRange, L->maxImageDimension2D, L->maxImageDimension3D, L->maxImageArrayLayers, L->maxMemoryAllocationCount, L->maxSamplerAllocationCount);
    VkPhysicalDeviceFeatures f; memset(&f, 0, sizeof f); { VkPhysicalDeviceFeatures2 f2 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2}; vkGetPhysicalDeviceFeatures2(V.pd, &f2); f = f2.features; }
    VKLOG("device facts: sparse binding %s, residency buffer %s, 2D %s, 3D %s, aliased %s; standard 2D block shape %s, 3D %s, aligned mip size %s, non-resident strict %s; address space %llu GB; "
          "sparse queue family %d%s; sparse textures: %s", f.sparseBinding ? "yes" : "no", f.sparseResidencyBuffer ? "yes" : "no", f.sparseResidencyImage2D ? "yes" : "no", f.sparseResidencyImage3D ? "yes" : "no",
          f.sparseResidencyAliased ? "yes" : "no", SP->residencyStandard2DBlockShape ? "yes" : "no", SP->residencyStandard3DBlockShape ? "yes" : "no", SP->residencyAlignedMipSize ? "yes" : "no",
          SP->residencyNonResidentStrict ? "yes" : "no", (unsigned long long)(L->sparseAddressSpaceSize >> 30), V.sqf == UINT32_MAX ? -1 : (int)V.sqf, V.sqf == V.qf ? " (the graphics queue's)" : "",
          V.has_sparse ? (V.has_sparse_alias ? "sparse-resident, tiles shared by physical memory" : "sparse-resident, a tile per place (no aliasing)") : "made whole under a budget, else not drawn (degraded)");
    static const struct { VkFormat f; const char *n; } SF[] = {{VK_FORMAT_BC6H_UFLOAT_BLOCK, "BC6H"}, {VK_FORMAT_ASTC_8x8_UNORM_BLOCK, "ASTC 8x8"}, {VK_FORMAT_R8G8B8A8_SNORM, "RGBA8 SNORM"},
                                                               {VK_FORMAT_R8G8B8A8_UNORM, "RGBA8"}, {VK_FORMAT_BC7_UNORM_BLOCK, "BC7"}, {VK_FORMAT_R16_UNORM, "R16"}};
    char sb[600]; int so = 0;
    for (unsigned i = 0; i < sizeof SF / sizeof *SF && so < (int)sizeof sb - 80; i++) {
        VkSparseImageFormatProperties sp[4]; uint32_t n = 4;
        vkGetPhysicalDeviceSparseImageFormatProperties(V.pd, SF[i].f, VK_IMAGE_TYPE_2D, VK_SAMPLE_COUNT_1_BIT, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT, VK_IMAGE_TILING_OPTIMAL, &n, sp);
        so += n ? snprintf(sb + so, sizeof sb - (size_t)so, " %s %ux%u (flags %#x);", SF[i].n, sp[0].imageGranularity.width, sp[0].imageGranularity.height, sp[0].flags) : snprintf(sb + so, sizeof sb - (size_t)so, " %s none;", SF[i].n); }
    if (f.sparseResidencyImage2D) VKLOG("device facts: sparse 2D tiles by format:%s", sb);
    // Storage images read with no format in the shader (the bundle's modules declare StorageImageReadWithoutFormat): shaderStorageImageReadWithoutFormat, enabled with every feature
    // the device has (vk_init); without it Vulkan 1.3 / VK_KHR_format_feature_flags2 can still allow it per format (VK_FORMAT_FEATURE_2_STORAGE_READ_WITHOUT_FORMAT_BIT)
    { static const struct { VkFormat f; const char *n; } RF[] = {{VK_FORMAT_R32_UINT, "R32_UINT"}, {VK_FORMAT_R32_SFLOAT, "R32F"}, {VK_FORMAT_R8G8B8A8_UNORM, "RGBA8"}, {VK_FORMAT_R16G16B16A16_SFLOAT, "RGBA16F"},
                                                                  {VK_FORMAT_R32G32B32A32_SFLOAT, "RGBA32F"}, {VK_FORMAT_B10G11R11_UFLOAT_PACK32, "R11G11B10F"}, {VK_FORMAT_R16G16_SFLOAT, "RG16F"}};
      char rb[300]; int ro = 0, f2 = v13 || has_ext(ex, ne, "VK_KHR_format_feature_flags2");
      for (unsigned i = 0; f2 && !f.shaderStorageImageReadWithoutFormat && i < sizeof RF / sizeof *RF; i++) {
          VkFormatProperties3 p3 = {VK_STRUCTURE_TYPE_FORMAT_PROPERTIES_3}; VkFormatProperties2 fp = {VK_STRUCTURE_TYPE_FORMAT_PROPERTIES_2, &p3}; vkGetPhysicalDeviceFormatProperties2(V.pd, RF[i].f, &fp);
          ro += snprintf(rb + ro, sizeof rb - (size_t)ro, " %s %s;", RF[i].n, p3.optimalTilingFeatures & VK_FORMAT_FEATURE_2_STORAGE_READ_WITHOUT_FORMAT_BIT ? "yes" : "no"); }
      if (f.shaderStorageImageReadWithoutFormat) VKLOG("device facts: storage images read without format yes (enabled), written without format %s", f.shaderStorageImageWriteWithoutFormat ? "yes" : "no");
      else LOG("[RED] [vk] device facts: no shaderStorageImageReadWithoutFormat: a shader that reads a storage image (StorageImageReadWithoutFormat) reads it only in the formats that allow it%s%s",
               f2 ? ", per format:" : " (and this device cannot say which: no Vulkan 1.3 / VK_KHR_format_feature_flags2)", f2 ? rb : ""); }
    // Matrix and half-precision arithmetic, for the Ampere tensor instructions (HMMA / LDSM / MOVM) the translator emulates per lane (InstEmitAmpereTensor.cs, HelperFunctionManager
    // dmsh_hmma16816_f16: 64 FP32 FMAs and 64 conversions a lane an HMMA): agl's NDSR upscaler, 30 compute shaders in Mario Kart World 1.8.0, 144-288 HMMAs each. A cooperative matrix
    // (VK_KHR_cooperative_matrix, subgroup scope: 64 lanes here, two of the console's warps) or packed FP16 arithmetic (shaderFloat16) would take most of that work: which the device has
    { VkPhysicalDeviceShaderFloat16Int8Features h = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES};
      VkPhysicalDeviceFeatures2 hf = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, &h}; vkGetPhysicalDeviceFeatures2(V.pd, &hf);
      char qc[512] = ""; size_t ql = 0; for (uint32_t i = 0; i < ne && ql + 64 < sizeof qc; i++) if (!strncmp(ex[i].extensionName, "VK_QCOM_", 8) || strstr(ex[i].extensionName, "cooperative"))
          ql += (size_t)snprintf(qc + ql, sizeof qc - ql, "%s%s", ql ? " " : "", ex[i].extensionName);
      VKLOG("device facts: shaderFloat16 %s, shaderInt8 %s, integer dot product %s; VK_KHR_cooperative_matrix %s; Qualcomm / cooperative extensions: %s", h.shaderFloat16 ? "yes" : "no", h.shaderInt8 ? "yes" : "no",
            has_ext(ex, ne, "VK_KHR_shader_integer_dot_product") || v13 ? "yes" : "no", has_ext(ex, ne, "VK_KHR_cooperative_matrix") ? "yes" : "no", ql ? qc : "none");
      PFN_vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR cm = has_ext(ex, ne, "VK_KHR_cooperative_matrix") ?
          (PFN_vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR)p_vkGetInstanceProcAddr(V.inst, "vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR") : NULL;
      uint32_t ncm = 0; VkCooperativeMatrixPropertiesKHR cp[32];
      if (cm && !cm(V.pd, &ncm, NULL) && ncm) { if (ncm > 32) ncm = 32; for (uint32_t i = 0; i < ncm; i++) cp[i] = (VkCooperativeMatrixPropertiesKHR){VK_STRUCTURE_TYPE_COOPERATIVE_MATRIX_PROPERTIES_KHR};
          if (cm(V.pd, &ncm, cp) >= 0) for (uint32_t i = 0; i < ncm; i++)   // (component types: VkComponentTypeKHR, 0 float16, 1 float32, 3 sint8, 7 uint8 ...; scope 3 subgroup)
              VKLOG("device facts: cooperative matrix %u: M %u N %u K %u, A %d B %d C %d result %d, saturating %u, scope %d", i, cp[i].MSize, cp[i].NSize, cp[i].KSize, cp[i].AType, cp[i].BType, cp[i].CType,
                    cp[i].ResultType, cp[i].saturatingAccumulation, cp[i].scope); } }
    // Depth formats as shadow maps: sampled, filtered linearly (a linear depth-compare sampler is then 2x2 PCF; without it the compare is point-sampled, a shadow's edge in steps of a texel),
    // compared in the sampler (Mario Kart World 1.8.0's kart shadow looked blocky in run 13: gfx_vk_exec.c shadow_check names the format its depth-compare samplers meet)
    { static const struct { VkFormat f; const char *n; } DF[] = {{VK_FORMAT_D16_UNORM, "D16"}, {VK_FORMAT_X8_D24_UNORM_PACK32, "X8_D24"}, {VK_FORMAT_D24_UNORM_S8_UINT, "D24S8"},
                                                                  {VK_FORMAT_D32_SFLOAT, "D32F"}, {VK_FORMAT_D32_SFLOAT_S8_UINT, "D32FS8"}};
      char db[300]; int dn = 0;
      for (unsigned i = 0; i < sizeof DF / sizeof *DF; i++) { VkFormatProperties3 p3 = {VK_STRUCTURE_TYPE_FORMAT_PROPERTIES_3}; VkFormatProperties2 fp = {VK_STRUCTURE_TYPE_FORMAT_PROPERTIES_2, &p3};
          vkGetPhysicalDeviceFormatProperties2(V.pd, DF[i].f, &fp); VkFormatFeatureFlags t = fp.formatProperties.optimalTilingFeatures;
          dn += snprintf(db + dn, sizeof db - (size_t)dn, " %s %s/%s/%s;", DF[i].n, t & VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT ? "yes" : "no", t & VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT ? "yes" : "no",
                         !(v13 || has_ext(ex, ne, "VK_KHR_format_feature_flags2")) ? "?" : p3.optimalTilingFeatures & VK_FORMAT_FEATURE_2_SAMPLED_IMAGE_DEPTH_COMPARISON_BIT ? "yes" : "no"); }
      VKLOG("device facts: depth formats (sampled / filtered linearly / compared in the sampler):%s; custom border colours without a format %s", db, fcb_nofmt ? "yes" : "no (custom borders off)"); }
    VKLOG("device facts: mesh shaders %s (task %s; groups: task %u, mesh %u; mesh outputs %u vertices, %u primitives; task payload %u bytes; shared memory task %u, mesh %u bytes; payload and shared "
          "memory task %u, mesh %u bytes; subgroup operations in task %s, mesh %s; SPIR-V up to 1.%u), draw indirect count %s, "
          "advanced blend %s (all operations %s, non-premultiplied source %s, correlated overlap %s, %u targets), BC %s, ASTC LDR %s, ETC2 %s, geometry shaders %s",
          V.has_mesh ? "yes" : "no", fms->taskShader ? "yes" : "no", V.task_max_groups, V.mesh_max_groups, ms.maxMeshOutputVertices, ms.maxMeshOutputPrimitives, ms.maxTaskPayloadSize, ms.maxTaskSharedMemorySize,
          ms.maxMeshSharedMemorySize, ms.maxTaskPayloadAndSharedMemorySize, ms.maxMeshPayloadAndSharedMemorySize, sg.supportedStages & VK_SHADER_STAGE_TASK_BIT_EXT ? "yes" : "no",
          sg.supportedStages & VK_SHADER_STAGE_MESH_BIT_EXT ? "yes" : "no", V.spv_max >> 8 & 0xff, V.has_dic ? "yes" : "no", V.has_adv_blend ? "yes" : "no", V.adv_all ? "yes" : "no", V.adv_npm ? "yes" : "no", V.adv_ovl ? "yes" : "no", V.adv_max_att, f.textureCompressionBC ? "yes" : "no",
          f.textureCompressionASTC_LDR ? "yes" : "no", f.textureCompressionETC2 ? "yes" : "no", f.geometryShader ? "yes" : "no");
    VkPhysicalDeviceMemoryBudgetPropertiesEXT mb = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_BUDGET_PROPERTIES_EXT};
    VkPhysicalDeviceMemoryProperties2 m2 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PROPERTIES_2, &mb}; int budget = has_ext(ex, ne, "VK_EXT_memory_budget");
    if (budget) vkGetPhysicalDeviceMemoryProperties2(V.pd, &m2);
    for (uint32_t h = 0; h < V.mem.memoryHeapCount && h < VK_MAX_MEMORY_HEAPS; h++) { char ty[200]; int to = 0;
        for (uint32_t k = 0; k < V.mem.memoryTypeCount && to < (int)sizeof ty - 12; k++) if (V.mem.memoryTypes[k].heapIndex == h) to += snprintf(ty + to, sizeof ty - (size_t)to, " %u:%#x", k, V.mem.memoryTypes[k].propertyFlags);
        VKLOG("device facts: memory heap %u: %.2f GB%s%s, budget %.2f GB, used %.2f GB; types (index:flags)%s", h, V.mem.memoryHeaps[h].size / 1073741824.0, V.mem.memoryHeaps[h].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT ? " device-local" : "",
              budget ? "" : " (no VK_EXT_memory_budget)", budget ? mb.heapBudget[h] / 1073741824.0 : 0.0, budget ? mb.heapUsage[h] / 1073741824.0 : 0.0, ty); }
}
static int swap_fault(void *addr, int is_write, void *uc); static void swap_prep(void *addr, size_t len);   // (memory pools, below)
void vk_init(void) {
    static int done; if (done) return; done = 1;
    // the pool swap's hooks, before nvn_dirty's (registered at its first range, which hle_nvn.c adds after the first pool): a write into a range being swapped waits for its new pages
    dm_fault_hook_register(swap_fault); dm_guest_write_prep_register(swap_prep);
    vk_shadow_init();   // (DM_VK_SHADOW: the shadow file's hooks, before nvn_dirty's too; gfx_vk_shadow.c)
    vk_budget_facts();  // (the device's memory as the low-memory killer sees it, once; gfx_vk_budget.c)
    // DM_VK_DRIVER=<library>: a Vulkan driver the APK carries (Mesa's Turnip for Adreno; porter --vulkan-driver), loaded through libadrenotools (BSD-2-Clause), which the
    // APK carries beside it with its hook libraries: a second copy of the system's loader in a linker namespace of its own, whose driver is this library. The loader keeps
    // making the surface and the swapchain, which Turnip on Android has none of (the platform loader makes them over VK_ANDROID_native_buffer). TOTKDroid, a port of Tears
    // of the Kingdom 1.4.3 that also replays NVN on Vulkan, measured its Great Sky Island forest at GPU 16-17 ms a frame on Turnip on an Adreno 830 (docs/FPS-2026-10-31.md),
    // against 22-27 ms here on Qualcomm's driver. A driver that does not load, or offers no Vulkan 1.3 GPU, leaves the system's
    const char *drv = getenv("DM_VK_DRIVER"); void *lib = NULL;
    if (drv && *drv) {
        Dl_info di; char dir[512] = "", dirs[520] = "";   // the APK's extracted libraries (mkapk.py: extractNativeLibs), where adrenotools finds its hooks and the driver
        if (dladdr((void *)vk_init, &di) && di.dli_fname && strrchr(di.dli_fname, '/')) {
            snprintf(dir, sizeof dir, "%.*s", (int)(strrchr(di.dli_fname, '/') - di.dli_fname), di.dli_fname); snprintf(dirs, sizeof dirs, "%s/", dir); }
        void *at = dlopen("libadrenotools.so", RTLD_NOW | RTLD_LOCAL);
        void *(*open_vk)(int, int, const char *, const char *, const char *, const char *, const char *, void **) =
            at ? (void *(*)(int, int, const char *, const char *, const char *, const char *, const char *, void **))dlsym(at, "adrenotools_open_libvulkan") : NULL;
        if (!open_vk || !*dir) VKLOG("[RED] DM_VK_DRIVER %s: no libadrenotools in the APK (%s); the system's driver instead", drv, at ? "no adrenotools_open_libvulkan" : dlerror());
        else if (!(lib = open_vk(RTLD_NOW | RTLD_LOCAL, 1 /* ADRENOTOOLS_DRIVER_CUSTOM */, NULL, dir, dirs, drv, NULL, NULL)))
            VKLOG("[RED] DM_VK_DRIVER %s: adrenotools could not load it from %s; the system's driver instead", drv, dir);
        else if (!(p_vkGetInstanceProcAddr = (PFN_vkGetInstanceProcAddr)dlsym(lib, "vkGetInstanceProcAddr")) || !drv_has_gpu(p_vkGetInstanceProcAddr)) {
            VKLOG("[RED] DM_VK_DRIVER %s: no Vulkan 1.3 GPU through it; the system's driver instead", drv); lib = NULL; }   // (left loaded: a driver does not always unload cleanly)
        else VKLOG("Vulkan driver %s through adrenotools, not the system's (%s)", drv, dir);
    }
    if (!lib) {
        if (!(lib = dlopen("libvulkan.so", RTLD_NOW | RTLD_LOCAL))) die("[vk] libvulkan.so: %s", dlerror());
        p_vkGetInstanceProcAddr = (PFN_vkGetInstanceProcAddr)dlsym(lib, "vkGetInstanceProcAddr");
    }
    vkCreateInstance = (PFN_vkCreateInstance)p_vkGetInstanceProcAddr(NULL, "vkCreateInstance");
    const char *iext[] = {"VK_KHR_surface", "VK_KHR_android_surface", "VK_EXT_debug_utils"};
    VkApplicationInfo app = {VK_STRUCTURE_TYPE_APPLICATION_INFO, NULL, "dmhost", 1, "nxnative", 1, VK_API_VERSION_1_3};
    // DM_VK_VALIDATE=1: Khronos's validation layer, which the APK carries only when built with it (mkapk.py --extra-so); its messages go to the log
    int validate = getenv("DM_VK_VALIDATE") && atoi(getenv("DM_VK_VALIDATE"));
    const char *layers[] = {"VK_LAYER_KHRONOS_validation"};
    VkInstanceCreateInfo ici = {VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, NULL, 0, &app, validate ? 1u : 0u, layers, validate ? 3u : 2u, iext};
    VkResult ir = vkCreateInstance(&ici, NULL, &V.inst);
    if (ir && validate) { VKLOG("[RED] validation layer not available (%d): running without it", ir); validate = 0; ici.enabledLayerCount = 0; ici.enabledExtensionCount = 2; ir = vkCreateInstance(&ici, NULL, &V.inst); }
    VKCHK(ir);
    if (validate) {
        PFN_vkCreateDebugUtilsMessengerEXT mk = (PFN_vkCreateDebugUtilsMessengerEXT)p_vkGetInstanceProcAddr(V.inst, "vkCreateDebugUtilsMessengerEXT");
        VkDebugUtilsMessengerCreateInfoEXT mi = {VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT, NULL, 0,
            VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT,
            VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT, vk_validation_msg, NULL};
        VkDebugUtilsMessengerEXT m; if (mk && !mk(V.inst, &mi, NULL, &m)) VKLOG("validation layer on");
    }
#define VK_LOADI(f) f = (PFN_##f)p_vkGetInstanceProcAddr(V.inst, #f);
    VK_FNS(VK_LOADI)
    uint32_t n = 1; VKCHK(vkEnumeratePhysicalDevices(V.inst, &n, &V.pd)); if (!n) die("[vk] no GPU");
    vkGetPhysicalDeviceProperties(V.pd, &V.props); vkGetPhysicalDeviceMemoryProperties(V.pd, &V.mem);
    // the device names itself before any check below can stop: a Moto g 5G 2024's log (Adreno 619) ended at the timeline check with no GPU or driver named.
    // driver: as Qualcomm encodes it (VK_MAKE_VERSION: 512.530.0), with the raw value for other vendors
    { uint32_t d = V.props.driverVersion;
      VKLOG("GPU %s (vendor %#x, device %#x), Vulkan %u.%u.%u, driver %u.%u.%u (%#x)", V.props.deviceName, V.props.vendorID, V.props.deviceID, VK_API_VERSION_MAJOR(V.props.apiVersion),
            VK_API_VERSION_MINOR(V.props.apiVersion), VK_API_VERSION_PATCH(V.props.apiVersion), d >> 22, d >> 12 & 0x3ff, d & 0xfff, d); }
    uint32_t qn = 0; vkGetPhysicalDeviceQueueFamilyProperties(V.pd, &qn, NULL);
    VkQueueFamilyProperties qp[16]; if (qn > 16) qn = 16; vkGetPhysicalDeviceQueueFamilyProperties(V.pd, &qn, qp);
    V.qf = 0; while (V.qf < qn && (qp[V.qf].queueFlags & (VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT)) != (VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT)) V.qf++;
    if (V.qf == qn) die("[vk] no graphics + compute queue");
    // sparse binding goes on the graphics queue when its family has VK_QUEUE_SPARSE_BINDING_BIT (one queue: the binds and the submissions are ordered by a timeline semaphore
    // alone), else on a queue of a family that has it (gfx_vk_tex.c vk_sparse_flush); none: sparse textures take the degraded path
    V.sqf = qp[V.qf].queueFlags & VK_QUEUE_SPARSE_BINDING_BIT ? V.qf : UINT32_MAX;
    for (uint32_t i = 0; i < qn && V.sqf == UINT32_MAX; i++) if (qp[i].queueFlags & VK_QUEUE_SPARSE_BINDING_BIT) V.sqf = i;
    uint32_t ne = 0; vkEnumerateDeviceExtensionProperties(V.pd, NULL, &ne, NULL);
    VkExtensionProperties *ex = calloc(ne, sizeof *ex); vkEnumerateDeviceExtensionProperties(V.pd, NULL, &ne, ex);
    const char *want[] = {"VK_KHR_swapchain", "VK_ANDROID_external_memory_android_hardware_buffer", "VK_EXT_queue_family_foreign", "VK_EXT_custom_border_color",
                          "VK_EXT_robustness2", "VK_EXT_depth_clip_control", "VK_EXT_provoking_vertex", "VK_EXT_index_type_uint8", "VK_EXT_transform_feedback",
                          "VK_GOOGLE_display_timing", "VK_EXT_mesh_shader", "VK_EXT_blend_operation_advanced", "VK_EXT_memory_budget", "VK_QCOM_queue_perf_hint"};
    const char *on[64]; uint32_t non = 0;
    for (unsigned i = 0; i < sizeof want / sizeof *want; i++) if (has_ext(ex, ne, want[i])) on[non++] = want[i]; else VKLOG("device has no %s", want[i]);
    // A Vulkan 1.1 or 1.2 device (Legion Y700 2023, Adreno 730: its stock driver reports 1.1) has what the replay uses from 1.2 / 1.3 as extensions: they are enabled
    // when present, so the features chained below and the functions loaded by their KHR names are valid there. A 1.3 device needs none of them
    int v12 = V.props.apiVersion >= VK_API_VERSION_1_2, v13 = V.props.apiVersion >= VK_API_VERSION_1_3; uint32_t npromo = 0;
    static const char *promo12[] = {"VK_KHR_create_renderpass2", "VK_KHR_imageless_framebuffer", "VK_KHR_image_format_list", "VK_KHR_depth_stencil_resolve", "VK_KHR_timeline_semaphore",
        "VK_KHR_shader_float16_int8", "VK_KHR_8bit_storage", "VK_EXT_descriptor_indexing", "VK_EXT_scalar_block_layout", "VK_KHR_uniform_buffer_standard_layout",
        "VK_KHR_shader_float_controls", "VK_KHR_spirv_1_4", "VK_KHR_vulkan_memory_model", "VK_KHR_separate_depth_stencil_layouts", "VK_EXT_host_query_reset",
        "VK_KHR_sampler_mirror_clamp_to_edge", "VK_KHR_draw_indirect_count", "VK_KHR_buffer_device_address", "VK_EXT_shader_viewport_index_layer", "VK_EXT_sampler_filter_minmax",
        "VK_KHR_shader_subgroup_extended_types", "VK_KHR_shader_atomic_int64", "VK_KHR_driver_properties"};
    static const char *promo13[] = {"VK_EXT_image_robustness", "VK_KHR_dynamic_rendering", "VK_KHR_synchronization2", "VK_EXT_shader_demote_to_helper_invocation",
        "VK_EXT_subgroup_size_control", "VK_KHR_copy_commands2", "VK_KHR_maintenance4", "VK_KHR_format_feature_flags2", "VK_EXT_extended_dynamic_state",
        "VK_KHR_zero_initialize_workgroup_memory", "VK_KHR_shader_terminate_invocation", "VK_EXT_inline_uniform_block", "VK_EXT_pipeline_creation_cache_control",
        "VK_EXT_texel_buffer_alignment", "VK_EXT_4444_formats", "VK_KHR_shader_integer_dot_product", "VK_EXT_private_data"};
    for (unsigned i = 0; !v12 && i < sizeof promo12 / sizeof *promo12 && non < 64; i++) if (has_ext(ex, ne, promo12[i])) { on[non++] = promo12[i]; npromo++; }
    for (unsigned i = 0; !v13 && i < sizeof promo13 / sizeof *promo13 && non < 64; i++) if (has_ext(ex, ne, promo13[i])) { on[non++] = promo13[i]; npromo++; }
    // the SPIR-V the device takes (the bundle's task and mesh modules are 1.4: gfx_vk_exec.c mesh_need) and vkCmdDispatchBase (Vulkan 1.1: gfx_vk_exec.c spill_dispatch)
    V.spv_max = v13 ? 0x10600 : v12 ? 0x10500 : is_on(on, non, "VK_KHR_spirv_1_4") ? 0x10400 : V.props.apiVersion >= VK_API_VERSION_1_1 ? 0x10300 : 0x10000;
    V.has_dbase = V.props.apiVersion >= VK_API_VERSION_1_1;
    // features: what the replay uses when the device has it. VkPhysicalDeviceVulkan11/12/13Features are valid only on a 1.2 / 1.3 device (a 1.1 driver ignores
    // them and nothing would be enabled); below that, the same features come through Vulkan 1.1's own structs and those of the extensions enabled above (xf)
    static const struct { const char *ext; VkStructureType s; } XF[] = {   // ext NULL: Vulkan 1.1 core, VkPhysicalDeviceVulkan11Features's members
        {NULL, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES}, {NULL, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MULTIVIEW_FEATURES},
        {NULL, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VARIABLE_POINTERS_FEATURES}, {NULL, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROTECTED_MEMORY_FEATURES},
        {NULL, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SAMPLER_YCBCR_CONVERSION_FEATURES}, {NULL, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_DRAW_PARAMETERS_FEATURES},
        {"VK_KHR_imageless_framebuffer", VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_IMAGELESS_FRAMEBUFFER_FEATURES}, {"VK_KHR_timeline_semaphore", VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TIMELINE_SEMAPHORE_FEATURES},
        {"VK_KHR_shader_float16_int8", VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES}, {"VK_KHR_8bit_storage", VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_8BIT_STORAGE_FEATURES},
        {"VK_EXT_descriptor_indexing", VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_INDEXING_FEATURES}, {"VK_EXT_scalar_block_layout", VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SCALAR_BLOCK_LAYOUT_FEATURES},
        {"VK_KHR_uniform_buffer_standard_layout", VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_UNIFORM_BUFFER_STANDARD_LAYOUT_FEATURES}, {"VK_KHR_vulkan_memory_model", VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_MEMORY_MODEL_FEATURES},
        {"VK_KHR_separate_depth_stencil_layouts", VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SEPARATE_DEPTH_STENCIL_LAYOUTS_FEATURES}, {"VK_EXT_host_query_reset", VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_HOST_QUERY_RESET_FEATURES},
        {"VK_KHR_buffer_device_address", VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES}, {"VK_KHR_shader_subgroup_extended_types", VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_SUBGROUP_EXTENDED_TYPES_FEATURES},
        {"VK_KHR_shader_atomic_int64", VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_ATOMIC_INT64_FEATURES},   // (mirror clamp, indirect count, minmax, viewport layer: the extension is the feature)
        {"VK_KHR_dynamic_rendering", VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DYNAMIC_RENDERING_FEATURES}, {"VK_KHR_synchronization2", VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SYNCHRONIZATION_2_FEATURES},
        {"VK_EXT_shader_demote_to_helper_invocation", VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_DEMOTE_TO_HELPER_INVOCATION_FEATURES}, {"VK_EXT_subgroup_size_control", VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_FEATURES},
        {"VK_KHR_maintenance4", VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_4_FEATURES}, {"VK_KHR_zero_initialize_workgroup_memory", VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ZERO_INITIALIZE_WORKGROUP_MEMORY_FEATURES},
        {"VK_KHR_shader_terminate_invocation", VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_TERMINATE_INVOCATION_FEATURES}, {"VK_EXT_inline_uniform_block", VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_INLINE_UNIFORM_BLOCK_FEATURES},
        {"VK_EXT_pipeline_creation_cache_control", VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PIPELINE_CREATION_CACHE_CONTROL_FEATURES}, {"VK_KHR_shader_integer_dot_product", VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_INTEGER_DOT_PRODUCT_FEATURES},
        {"VK_EXT_private_data", VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PRIVATE_DATA_FEATURES}};   // (VK_EXT_image_robustness: robustImageAccess stays off, as on 1.3)
    uint32_t nxf = sizeof XF / sizeof *XF; featx_t xf[sizeof XF / sizeof *XF]; memset(xf, 0, sizeof xf); VkBaseOutStructure *xs = NULL;
    for (int i = (int)nxf - 1; i >= 0; i--) if (XF[i].ext ? is_on(on, non, XF[i].ext) : !v12) { xf[i].h.sType = XF[i].s; xf[i].h.pNext = xs; xs = &xf[i].h; }
    VkPhysicalDeviceVulkan13Features f13 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES, xs};
    VkPhysicalDeviceVulkan12Features f12 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES, v13 ? (void *)&f13 : xs};
    VkPhysicalDeviceVulkan11Features f11 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES, &f12};
    VkPhysicalDeviceFeatures2 ff = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, v12 ? (void *)&f11 : xs};
    vkGetPhysicalDeviceFeatures2(V.pd, &ff);
    // without timeline semaphores (Vulkan 1.1 drivers: the Adreno 618 / 619 of 2021, 512.530) each submission signals a fence of a ring instead (slots_init).
    // DM_VK_X_FENCES=1 (start-up): the fences on a device that has them; with DM_VK_X_BC_CPU=1 below, a device of today stands in for those drivers
    { const char *e = getenv("DM_VK_X_FENCES");
      V.has_timeline = (v12 ? f12.timelineSemaphore : featx(xf, nxf, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TIMELINE_SEMAPHORE_FEATURES)[0]) && !(e && atoi(e)); }
    if (!V.has_timeline) VKLOG("no timeline semaphores (Vulkan 1.2 or VK_KHR_timeline_semaphore): submissions signal fences");
    // dynamic rendering is 1.3 or its extension with the feature on; a loaded pointer does not tell, since the loader hands out a trampoline for core functions the driver lacks
    V.has_dynrender = v13 ? f13.dynamicRendering : featx(xf, nxf, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DYNAMIC_RENDERING_FEATURES)[0];
    // imageless framebuffers need render pass 2 (gfx_vk_exec.c rp_get): core in 1.2, else its extension
    V.has_imageless = v12 ? f12.imagelessFramebuffer : featx(xf, nxf, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_IMAGELESS_FRAMEBUFFER_FEATURES)[0] && is_on(on, non, "VK_KHR_create_renderpass2");
    if (!V.has_imageless && !V.has_dynrender) die("[vk] the device has neither dynamic rendering nor imageless framebuffers with render pass 2: passes cannot be begun");
    if (!v13) VKLOG("Vulkan %u.%u device: %u extensions enabled for the features promoted to 1.2 / 1.3; dynamic rendering %s", VK_API_VERSION_MAJOR(V.props.apiVersion),
                    VK_API_VERSION_MINOR(V.props.apiVersion), npromo, V.has_dynrender ? "yes" : "no (render pass objects only)");
    VkPhysicalDeviceCustomBorderColorFeaturesEXT fcb = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_CUSTOM_BORDER_COLOR_FEATURES_EXT};
    VkPhysicalDeviceRobustness2FeaturesEXT fr2 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ROBUSTNESS_2_FEATURES_EXT, &fcb};
    VkPhysicalDeviceDepthClipControlFeaturesEXT fdc = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DEPTH_CLIP_CONTROL_FEATURES_EXT, &fr2};
    VkPhysicalDeviceIndexTypeUint8FeaturesEXT fu8 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_INDEX_TYPE_UINT8_FEATURES_EXT, &fdc};
    VkPhysicalDeviceTransformFeedbackFeaturesEXT fxf = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TRANSFORM_FEEDBACK_FEATURES_EXT, &fu8};
    VkPhysicalDeviceMeshShaderFeaturesEXT fms = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MESH_SHADER_FEATURES_EXT};
    VkPhysicalDeviceBlendOperationAdvancedFeaturesEXT fba = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BLEND_OPERATION_ADVANCED_FEATURES_EXT};
    VkPhysicalDeviceQueuePerfHintFeaturesQCOM fph = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_QUEUE_PERF_HINT_FEATURES_QCOM};
    if (has_ext(ex, ne, "VK_EXT_mesh_shader")) { fms.pNext = fxf.pNext; fxf.pNext = &fms; }   // (queried only where the extension is: an unknown struct in the chain is not allowed)
    if (has_ext(ex, ne, "VK_EXT_blend_operation_advanced")) { fba.pNext = fxf.pNext; fxf.pNext = &fba; }
    if (has_ext(ex, ne, "VK_QCOM_queue_perf_hint")) { fph.pNext = fxf.pNext; fxf.pNext = &fph; }
    VkPhysicalDeviceFeatures2 fx = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, &fxf};
    vkGetPhysicalDeviceFeatures2(V.pd, &fx);
    fxf.pNext = &fu8;   // (the enabled chain below is built anew)
    // NVN2's DrawMeshTasks (gfx_vk_exec.c mesh_draw): VK_EXT_mesh_shader's mesh stage, and its task stage where the device has one. Only those two features are enabled (multiview,
    // shading-rate and query support stay off). DM_VK_MESH=0 (start-up): none, every DrawMeshTasks skipped
    { const char *e = getenv("DM_VK_MESH"); V.has_mesh = fms.meshShader && has_ext(ex, ne, "VK_EXT_mesh_shader") && !(e && !atoi(e)); }
    fms.multiviewMeshShader = fms.primitiveFragmentShadingRateMeshShader = fms.meshShaderQueries = 0; if (!V.has_mesh) fms.taskShader = 0;
    // advanced blend modes (NVN's SetAdvancedMode with its premultiplied source and overlap, gfx_vk_exec.c adv_blend): VK_EXT_blend_operation_advanced, coherent only (the replay puts no
    // barrier between two draws that blend over the same pixels)
    V.has_adv_blend = fba.advancedBlendCoherentOperations && has_ext(ex, ne, "VK_EXT_blend_operation_advanced");
    g_has_perf_hint = fph.queuePerfHint && has_ext(ex, ne, "VK_QCOM_queue_perf_hint");   // (DM_VK_PERF_HINT, perf_hint_apply)
    // transform feedback (NVN BeginTransformFeedback; gfx_vk_exec.c xfb_*): Xenoblade Chronicles 3 captures particle state with it
    V.has_xfb = fxf.transformFeedback && has_ext(ex, ne, "VK_EXT_transform_feedback"); fxf.geometryStreams = 0;
    fr2.robustBufferAccess2 = 0; fr2.robustImageAccess2 = 0;   // nullDescriptor only: unbound slots read 0 without a dummy
    V.has_custom_border = fcb.customBorderColors && has_ext(ex, ne, "VK_EXT_custom_border_color");
    fcb_nofmt = V.has_custom_border && fcb.customBorderColorWithoutFormat; V.has_custom_border = fcb_nofmt;   // (a border colour with no format is all make_sampler gives: without it, the standard colours)
    V.has_robust2 = fr2.nullDescriptor && has_ext(ex, ne, "VK_EXT_robustness2");
    V.has_depth_clip_ctl = fdc.depthClipControl && has_ext(ex, ne, "VK_EXT_depth_clip_control");
    V.has_dtiming = has_ext(ex, ne, "VK_GOOGLE_display_timing");   // frame pacing on the display (gfx_vk_present.c pace_*)
    V.has_bc = ff.features.textureCompressionBC; V.has_astc = ff.features.textureCompressionASTC_LDR; V.has_gs = ff.features.geometryShader;
    { const char *e = getenv("DM_VK_X_BC_CPU"); if (e && atoi(e)) V.has_bc = 0; }   // (start-up) BC textures decoded on the CPU as on a device without them (gfx_vk_tex.c bc_decode)
    V.max_aniso = V.props.limits.maxSamplerAnisotropy;
    // enable: everything the shaders may use that the device has
    VkPhysicalDeviceFeatures en = ff.features; en.robustBufferAccess = 0; en.shaderFloat64 = 0;
    f12.bufferDeviceAddressCaptureReplay = 0; f12.bufferDeviceAddressMultiDevice = 0;
    featx(xf, nxf, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES)[1] = featx(xf, nxf, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES)[2] = 0;   // (its extension's: the same two)
    f13.robustImageAccess = 0;
    // DM_VK_ROBUST_IMAGE (start-up, an A/B; PARITY G20): 1 robustImageAccess (Vulkan 1.3: an out-of-range texel fetch or image load reads values from the image or zero, never anything),
    // 2 robustImageAccess2 too (VK_EXT_robustness2: zero, as Maxwell reads them, the alpha 0 or 1 by format). Off by default: Maxwell's out-of-range fetches read zero, Vulkan's are undefined
    // without it, and the checks cost the texture unit (to be measured with DM_VK_GPUPROF before it is a default)
    { const char *e = getenv("DM_VK_ROBUST_IMAGE"); int want = e ? atoi(e) : 0;
      if (want >= 1 && v13 && ff.pNext) { VkPhysicalDeviceVulkan13Features q13 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES}; VkPhysicalDeviceFeatures2 q = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, &q13};
          vkGetPhysicalDeviceFeatures2(V.pd, &q); f13.robustImageAccess = q13.robustImageAccess; }
      if (want >= 2 && V.has_robust2) { VkPhysicalDeviceRobustness2FeaturesEXT q2 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ROBUSTNESS_2_FEATURES_EXT}; VkPhysicalDeviceFeatures2 q = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, &q2};
          vkGetPhysicalDeviceFeatures2(V.pd, &q); fr2.robustImageAccess2 = q2.robustImageAccess2; }
      if (want) VKLOG("DM_VK_ROBUST_IMAGE=%d: robustImageAccess %s, robustImageAccess2 %s", want, f13.robustImageAccess ? "on" : v13 ? "not on this device" : "off (a device below Vulkan 1.3)",
                      fr2.robustImageAccess2 ? "on" : want >= 2 ? "not on this device" : "off"); }
    fcb.pNext = NULL; fr2.pNext = V.has_custom_border ? (void *)&fcb : NULL; fdc.pNext = V.has_robust2 ? (void *)&fr2 : (V.has_custom_border ? (void *)&fcb : NULL);
    fu8.pNext = V.has_depth_clip_ctl ? (void *)&fdc : fdc.pNext;
    fxf.pNext = &fu8;
    VkBaseOutStructure *t = ff.pNext; while (t->pNext) t = t->pNext;   // the queried chain (core / promoted features), then the extensions'
    VkBaseOutStructure *xh = V.has_xfb ? (VkBaseOutStructure *)&fxf : (VkBaseOutStructure *)&fu8;
    if (V.has_mesh) { fms.pNext = xh; xh = (VkBaseOutStructure *)&fms; }
    if (V.has_adv_blend) { fba.pNext = xh; xh = (VkBaseOutStructure *)&fba; }
    if (g_has_perf_hint) { fph.pNext = xh; xh = (VkBaseOutStructure *)&fph; }
    t->pNext = xh;
    VkPhysicalDeviceFeatures2 enf = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, ff.pNext, en};
    // Sparse residency images (gfx_vk_tex.c sparse_*: Mario Kart World's terrain arrays, 8192x8192x68, 1.4 to 5.8 GB each made whole). The binds are ordered against the submissions
    // by a timeline semaphore, so a device without them takes the degraded path. DM_VK_SPARSE=0 (start-up): the degraded path on any device (an A/B)
    { const char *e = getenv("DM_VK_SPARSE");
      V.has_sparse = ff.features.sparseBinding && ff.features.sparseResidencyImage2D && V.sqf != UINT32_MAX && V.has_timeline && !(e && !atoi(e));
      V.has_sparse_alias = V.has_sparse && ff.features.sparseResidencyAliased; }
    float prio = 1;
    VkDeviceQueueCreateInfo qci[2] = {{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO, NULL, 0, V.qf, 1, &prio}, {VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO, NULL, 0, V.sqf, 1, &prio}};
    VkDeviceCreateInfo dci = {VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, &enf, 0, V.has_sparse && V.sqf != V.qf ? 2u : 1u, qci, 0, NULL, non, on, NULL};
    VKCHK(vkCreateDevice(V.pd, &dci, NULL, &V.dev));
    vkGetDeviceProcAddr = (PFN_vkGetDeviceProcAddr)p_vkGetInstanceProcAddr(V.inst, "vkGetDeviceProcAddr");
#define VK_LOADD(f) f = (PFN_##f)vkGetDeviceProcAddr(V.dev, #f); if (!f) f = (PFN_##f)vkGetDeviceProcAddr(V.dev, #f "KHR"); if (!f) f = (PFN_##f)p_vkGetInstanceProcAddr(V.inst, #f);   // a promoted function by its extension's name (Vulkan 1.1 / 1.2 devices)
    VK_DEV_FNS(VK_LOADD)
    { void vk_dq_init(void); vk_dq_init(); }   // DM_VK_MT_REPLAY: the vkCmd* calls through the recorder thread (gfx_vk_dq.c)
    { void vk_bt_init(void); vk_bt_init(); }   // full barriers back to back: the later ones left out (DM_VK_BAR_MERGE, gfx_vk_dq.c)
    vkGetDeviceQueue(V.dev, V.qf, 0, &V.q);
    V.sq = V.q; if (V.has_sparse && V.sqf != V.qf) vkGetDeviceQueue(V.dev, V.sqf, 0, &V.sq);
    if (g_has_perf_hint && !(p_set_perf_hint = (PFN_vkQueueSetPerfHintQCOM)vkGetDeviceProcAddr(V.dev, "vkQueueSetPerfHintQCOM"))) g_has_perf_hint = 0;
    if (g_has_perf_hint) { VkPhysicalDeviceQueuePerfHintPropertiesQCOM pp = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_QUEUE_PERF_HINT_PROPERTIES_QCOM}; VkPhysicalDeviceProperties2 p2 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, &pp};
        vkGetPhysicalDeviceProperties2(V.pd, &p2); VKLOG("VK_QCOM_queue_perf_hint: queues %#x take a GPU clock hint (DM_VK_PERF_HINT=max | min | scaled:N | default, live; off by default)", pp.supportedQueues);
        if (!(pp.supportedQueues & (VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT))) g_has_perf_hint = 0; }
    // NVN2's indirect-count multi-draws (gfx_vk_exec.c multi_draw): Vulkan 1.2's drawIndirectCount, or its extension on a 1.1 device (the extension is the feature)
    V.has_dic = v12 ? f12.drawIndirectCount : is_on(on, non, "VK_KHR_draw_indirect_count");
    { const VkBool32 *sc = featx(xf, nxf, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_FEATURES);
      device_facts(ex, ne, v12, v13, v13 ? f13.subgroupSizeControl : sc[0], v13 ? f13.computeFullSubgroups : sc[1], &fms); }
    VkPipelineCacheCreateInfo pci = {VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO};
    char path[1024]; const char *root = getenv("DM_ROOT"); void *data = NULL; size_t dn = 0;
    if (root && !getenv("DM_VK_NOPCACHE")) { snprintf(path, sizeof path, "%s/vk_pipeline_cache.bin", root); FILE *f = fopen(path, "rb");   // DM_VK_NOPCACHE=1: start empty
        if (f) { fseek(f, 0, SEEK_END); dn = (size_t)ftell(f); fseek(f, 0, SEEK_SET); data = malloc(dn); if (fread(data, 1, dn, f) != dn) dn = 0; fclose(f); } }
    pci.initialDataSize = dn; pci.pInitialData = data;
    if (vkCreatePipelineCache(V.dev, &pci, NULL, &V.pcache)) { pci.initialDataSize = 0; pci.pInitialData = NULL; VKCHK(vkCreatePipelineCache(V.dev, &pci, NULL, &V.pcache)); }
    free(data); free(ex);
    VKLOG("%s, Vulkan %u.%u, driver %#x; timeline semaphores %s, BC %s, ASTC %s, geometry shaders %s, nullDescriptor %s, custom border colours %s, depth clip control %s, transform feedback %s; pipeline cache %zu bytes",
          V.props.deviceName, VK_API_VERSION_MAJOR(V.props.apiVersion), VK_API_VERSION_MINOR(V.props.apiVersion), V.props.driverVersion, V.has_timeline ? "yes" : "no (fences)",
          V.has_bc ? "yes" : "no (decoded on the CPU)", V.has_astc ? "yes" : "no", V.has_gs ? "yes" : "no", V.has_robust2 ? "yes" : "no", V.has_custom_border ? "yes" : "no", V.has_depth_clip_ctl ? "yes" : "no", V.has_xfb ? "yes" : "no", dn);
    slots_init();
    completion_start();
    { void crash_handlers_check(const char *when); crash_handlers_check("after the Vulkan device was made"); }   // (again at present 600, gfx_vk_present.c: the shader compiler loads with the first pipelines)
    __atomic_store_n(&g_vk_ready, 1, __ATOMIC_RELEASE);
}
int vk_ready(void) { return __atomic_load_n(&g_vk_ready, __ATOMIC_ACQUIRE); }
void vk_pipeline_cache_save(void) {
    static pthread_mutex_t mu = PTHREAD_MUTEX_INITIALIZER; static size_t last;
    const char *root = getenv("DM_ROOT"); if (!root || !V.pcache) return;
    pthread_mutex_lock(&mu);
    size_t n = 0; if (vkGetPipelineCacheData(V.dev, V.pcache, &n, NULL) || !n || n == last) { pthread_mutex_unlock(&mu); return; }   // unchanged since the last save
    last = n; pthread_mutex_unlock(&mu);
    void *d = malloc(n); if (vkGetPipelineCacheData(V.dev, V.pcache, &n, d)) { free(d); return; }
    char path[1024], tmp[1100]; snprintf(path, sizeof path, "%s/vk_pipeline_cache.bin", root); snprintf(tmp, sizeof tmp, "%s.tmp", path);
    FILE *f = fopen(tmp, "wb"); if (f) { fwrite(d, 1, n, f); fclose(f); rename(tmp, path); }
    free(d);
}

// ---------------- memory pools ----------------
#define ZC_CHUNK 0x10000000ull   // (pool_add_one, below)
// The table of pool entries (chunks, pools placed by parts and their placed parts) has no fixed size: Mario Kart World 1.8.0 died at 28.6 s of its first race (run 5) with "too many
// memory pools", the 4,096 entries the table had (its order kept as uint16_t indices). An entry is allocated once and kept, a removed one reused by a later entry (as the table's slots
// were), so a pointer to one always points at an entry; g_ord holds them sorted by lo (pool_before) and doubles as it fills, an entry put in or taken out by a binary search and a move of
// the pointers after it. Lookups are a binary search (pool_find): a linear scan behind the per-thread caches was 18% of the profiler's samples of Super Mario Bros. Wonder's replay in its
// first level (a draw looks up more pools than the caches hold), and its world map's replay took 4.2 ms a frame with the search instead of 4.6-4.9
static vkpool_t **g_ord; static int g_npool, g_ncap; static vkpool_t *g_pool_spare;   // g_pool_spare: entries removed, free for reuse (linked through spare)
static int g_novl;   // live chunks or pools over memory another holds (vkpool_t ovl): while there are some, a lookup the search does not answer scans the table, as all did before it
static pthread_rwlock_t g_pool_mu = PTHREAD_RWLOCK_INITIALIZER;
static uint32_t g_pool_gone;   // bumped when a pool whose buffer a descriptor set kept across submissions names goes (mtl_pool_del): those sets go (gfx_vk_exec.c kept_check, vk_keep_buf_named)
uint32_t vk_pool_gone_gen(void) { return __atomic_load_n(&g_pool_gone, __ATOMIC_ACQUIRE); }
static uint32_t g_pool_gen;   // bumped under the write lock when a pool or a placed part is added or removed: the per-thread caches of vk_pool_of / vk_pool_gpu hold while it is unchanged
#ifndef POOL_STEP
#define POOL_STEP()   // (tests/test_vk_memory.py counts the searches' steps)
#endif
static int pool_before(const vkpool_t *x, const vkpool_t *y) {   // g_ord's order: by lo, a pool placed by parts before the parts that start where it does, then as made
    if (x->lo != y->lo) return x->lo < y->lo;
    int px = x->state == POOL_PARTS, py = y->state == POOL_PARTS; return px != py ? px : x->serial < y->serial;
}
static int pool_first_ge(uint64_t a) { int lo = 0, hi = g_npool; while (lo < hi) { POOL_STEP(); int m = (lo + hi) / 2; if (g_ord[m]->lo < a) lo = m + 1; else hi = m; } return lo; }   // (g_pool_mu held)
static int pool_last_le(uint64_t a) { int lo = 0, hi = g_npool; while (lo < hi) { POOL_STEP(); int m = (lo + hi) / 2; if (g_ord[m]->lo <= a) lo = m + 1; else hi = m; } return lo - 1; }   // -1: none
static vkpool_t *pool_entry(void) {   // g_pool_mu held for writing: a cleared entry, not in g_ord yet
    vkpool_t *e = g_pool_spare; if (e) g_pool_spare = e->spare; else if (!(e = malloc(sizeof *e))) die("[vk] memory pools: no memory for an entry (%d of them)", g_npool);
    memset(e, 0, sizeof *e); return e;
}
static void pool_put(vkpool_t *e) {   // g_pool_mu held for writing: e into g_ord at its place
    if (g_npool == g_ncap) { int nc = g_ncap ? 2 * g_ncap : 1024; vkpool_t **n = realloc(g_ord, (size_t)nc * sizeof *n); if (!n) die("[vk] memory pools: no memory for a table of %d", nc); g_ord = n; g_ncap = nc; }
    int lo = 0, hi = g_npool; while (lo < hi) { int m = (lo + hi) / 2; if (pool_before(g_ord[m], e)) lo = m + 1; else hi = m; }
    memmove(&g_ord[lo + 1], &g_ord[lo], (size_t)(g_npool - lo) * sizeof *g_ord); g_ord[lo] = e; g_npool++;
}
static void pool_take(int i) {   // g_pool_mu held for writing: entry i out of g_ord, kept for reuse (a copy of it is the caller's to finish with)
    vkpool_t *e = g_ord[i]; memmove(&g_ord[i], &g_ord[i + 1], (size_t)(g_npool - i - 1) * sizeof *g_ord); g_npool--;
    if (e->ovl) g_novl--;
    e->spare = g_pool_spare; g_pool_spare = e;
}
static vkpool_t *pool_parent(const vkpool_t *e) {   // a placed part's pool: the POOL_PARTS entry at its base (first at that lo)
    for (int i = pool_first_ge(e->base); i < g_npool && g_ord[i]->lo == e->base; i++) if (g_ord[i]->state == POOL_PARTS && g_ord[i]->end == e->end) return g_ord[i];
    return NULL;
}
// g_pool_mu held: the entry holding [a, a+n) that want asks for (0: any, the placed part where one holds it; 1: a placed one; 2: one not placed, a chunk or the pool placed by parts
// when no part holds the range). The last entry by lo at or below a holds it, or its pool does when it is a part: chunks and pools do not overlap, and a pool's parts lie inside it,
// after it. *across (when asked): the range starts in a chunk and ends past it, inside its pool (no buffer holds it)
static vkpool_t *pool_find(uint64_t a, uint64_t n, int want, int *across) {
    vkpool_t *r = NULL; int i = pool_last_le(a);
    if (i >= 0) { vkpool_t *e = g_ord[i];
        if (e->part) { if (a + n <= e->hi) r = want == 2 ? NULL : e; else if (want != 1 && (e = pool_parent(e)) && a + n <= e->hi) r = e; }
        else if (e->state == POOL_PARTS) { if (want != 1 && a + n <= e->hi) r = e; }
        else if (a + n <= e->hi) { if (!want || (want == 1) == (e->state == POOL_PLACED)) r = e; }
        else if (across && a < e->hi && a + n <= e->end) *across = 1; }
    if (!r && g_novl) for (int k = 0; k < g_npool && !r; k++) { vkpool_t *e = g_ord[k];   // pools over the same memory: the scan
        if (a >= e->lo && a + n <= e->hi) { if (!want || (want == 1 ? e->state == POOL_PLACED : e->state != POOL_PLACED)) r = e; }
        else if (across && a >= e->lo && a < e->hi && a + n <= e->end) *across = 1; }
    return r;
}
static int recv_fds(int sock, int *fds, int max) {
    char data[4096], ctl[CMSG_SPACE(sizeof(int) * 16)]; struct iovec iov = {data, sizeof data};
    struct msghdr m = {0}; m.msg_iov = &iov; m.msg_iovlen = 1; m.msg_control = ctl; m.msg_controllen = sizeof ctl;
    if (recvmsg(sock, &m, 0) < 0) return -1;
    int n = 0;
    for (struct cmsghdr *c = CMSG_FIRSTHDR(&m); c; c = CMSG_NXTHDR(&m, c))
        if (c->cmsg_level == SOL_SOCKET && c->cmsg_type == SCM_RIGHTS)
            for (size_t i = 0; i < (c->cmsg_len - CMSG_LEN(0)) / sizeof(int) && n < max; i++) fds[n++] = ((int *)CMSG_DATA(c))[i];
    return n;
}
static const VkBufferUsageFlags POOL_USAGE0 = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_UNIFORM_TEXEL_BUFFER_BIT |
    VK_BUFFER_USAGE_STORAGE_TEXEL_BUFFER_BIT | VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT |
    VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT;
#define POOL_USAGE (POOL_USAGE0 | (V.has_xfb ? VK_BUFFER_USAGE_TRANSFORM_FEEDBACK_BUFFER_BIT_EXT : 0))   // any pool range may be a transform feedback buffer
// A hardware buffer of size bytes (a BLOB the CPU and the GPU use) and its dma-buf's descriptors (fds[0] is the one mapped). 0 on success
static int ahb_new(uint64_t size, AHardwareBuffer **out, int fds[16], int *nfd) {
    AHardwareBuffer_Desc d = {(uint32_t)size, 1, 1, AHARDWAREBUFFER_FORMAT_BLOB,
        AHARDWAREBUFFER_USAGE_CPU_READ_OFTEN | AHARDWAREBUFFER_USAGE_CPU_WRITE_OFTEN | AHARDWAREBUFFER_USAGE_GPU_DATA_BUFFER, 0, 0, 0};
    *nfd = 0; if (size > 0xffffffffull) return -1;
    AHardwareBuffer *ahb; if (AHardwareBuffer_allocate(&d, &ahb)) return -1;
    int sv[2], n = -1; if (socketpair(AF_UNIX, SOCK_SEQPACKET | SOCK_CLOEXEC, 0, sv)) { AHardwareBuffer_release(ahb); return -1; }
    if (!AHardwareBuffer_sendHandleToUnixSocket(ahb, sv[0])) n = recv_fds(sv[1], fds, 16);
    close(sv[0]); close(sv[1]);
    if (n < 1) { AHardwareBuffer_release(ahb); return -1; }
    *out = ahb; *nfd = n; return 0;
}
static int ahb_import(vkpool_t *p, AHardwareBuffer *ahb, uint64_t size) {   // the hardware buffer as p's VkBuffer: 0 on success
    VkAndroidHardwareBufferPropertiesANDROID ap = {VK_STRUCTURE_TYPE_ANDROID_HARDWARE_BUFFER_PROPERTIES_ANDROID};
    VkExternalMemoryBufferCreateInfo eb = {VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO, NULL, VK_EXTERNAL_MEMORY_HANDLE_TYPE_ANDROID_HARDWARE_BUFFER_BIT_ANDROID};
    VkBufferCreateInfo bi = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, &eb, 0, size, POOL_USAGE, VK_SHARING_MODE_EXCLUSIVE};
    if (vkGetAndroidHardwareBufferPropertiesANDROID(V.dev, ahb, &ap) || vkCreateBuffer(V.dev, &bi, NULL, &p->buf)) { p->buf = VK_NULL_HANDLE; return -2; }
    VkMemoryDedicatedAllocateInfo ded = {VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO, NULL, VK_NULL_HANDLE, p->buf};   // the driver requires a dedicated allocation for an imported AHardwareBuffer
    VkImportAndroidHardwareBufferInfoANDROID imp = {VK_STRUCTURE_TYPE_IMPORT_ANDROID_HARDWARE_BUFFER_INFO_ANDROID, &ded, ahb};
    VkMemoryAllocateInfo ai = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, &imp, ap.allocationSize, vk_mem_type(ap.memoryTypeBits, 0, 0)};
    if (vkAllocateMemory(V.dev, &ai, NULL, &p->mem) || vkBindBufferMemory(V.dev, p->buf, p->mem, 0)) { vkDestroyBuffer(V.dev, p->buf, NULL); p->buf = VK_NULL_HANDLE; return -2; }
    return 0;
}
// ---- guest memory read without a fault (tests: tests/test_vk_memory.py) ----
// A page of guest memory may be inaccessible: a guest address region reserved and not mapped (the alias region's pages are PROT_NONE until MapPhysicalMemory: Mario Kart World 1.8.0 on the
// phone read a texture whose last 36 KB lay past what the game had mapped), or a virtual pool's range (no CPU alias on Android). mem_read copies through the page tables and fails on such a
// page instead of faulting (process_vm_readv; Mach's vm_read_overwrite on the Mac, for the tests). It also refuses the dma-buf mapping of a placed pool, which is readable: vk_readable
// knows those from the pool table
#ifdef __APPLE__
#include <mach/mach.h>
static int mem_read(void *dst, const void *src, size_t n) { vm_size_t got = 0; return vm_read_overwrite(mach_task_self(), (vm_address_t)(uintptr_t)src, n, (vm_address_t)(uintptr_t)dst, &got) == KERN_SUCCESS && got == n; }
static uintptr_t first_unreadable(uintptr_t a, uintptr_t e, uintptr_t pg, int *err) { *err = 0; for (; a < e; a += pg) { uint8_t b; if (!mem_read(&b, (const void *)a, 1)) return a; } return e; }
#else
#include <sys/uio.h>
static int mem_read(void *dst, const void *src, size_t n) { struct iovec l = {dst, n}, r = {(void *)src, n}; return process_vm_readv(getpid(), &l, 1, &r, 1, 0) == (ssize_t)n; }
static uintptr_t first_unreadable(uintptr_t a, uintptr_t e, uintptr_t pg, int *err) {   // the first page of [a, e) that cannot be read (e: none), a byte of each, 256 pages a call; *err: the call's errno
    enum { IOV = 256 }; struct iovec rv[IOV]; uint8_t buf[IOV]; struct iovec lv = {buf, 0}; *err = 0;
    while (a < e) {
        int k = 0; for (uintptr_t x = a; k < IOV && x < e; k++, x += pg) rv[k] = (struct iovec){(void *)x, 1};
        lv.iov_len = (size_t)k; ssize_t r = process_vm_readv(getpid(), &lv, 1, rv, (unsigned long)k, 0);
        if (r < 0) { *err = errno; return a; }
        if (r < k) return a + (uintptr_t)r * pg;   // (it stops at the first element it cannot read)
        a += (uintptr_t)k * pg;
    }
    return e;
}
#endif
// What can be read of [src, src+len) into dst, unreadable pages as zeros (a megabyte at a time, page by page where that fails): the bytes left as zeros; pg the page size; bad, if given, gets
// a bit per page that could not be read
static uint64_t copy_readable(uint8_t *dst, const uint8_t *src, uint64_t len, uint64_t pg, uint64_t *bad) {
    uint64_t miss = 0;
    for (uint64_t o = 0; o < len;) {
        uint64_t k = len - o < (1u << 20) ? len - o : (1u << 20);
        if (mem_read(dst + o, src + o, k)) { o += k; continue; }
        for (uint64_t e = o + k; o < e; o += pg) { uint64_t m = e - o < pg ? e - o : pg;
            if (!mem_read(dst + o, src + o, m)) { memset(dst + o, 0, m); miss += m; if (bad) bad[(o / pg) >> 6] |= 1ull << ((o / pg) & 63); } }
    }
    return miss;
}
// ---- end of the guest memory read ----
// ---- Lazy placement: the swap ----
// A range of guest memory becomes GPU memory by having its pages replaced by those of a hardware buffer mapped at the same address, its content copied in first. The game's threads may
// write the range meanwhile: it is read-only for the copy, a write into it waits in swap_fault (the host's page fault hooks, registered before nvn_dirty's: vk_init) and a host write into it
// in swap_prep (dm_guest_write_prepare), until the new pages are in place, and is then made over them. A page nvn_dirty let a write through just before the swap began is found by its stamp
// and copied again after a second protection. Pages nvn_dirty tracks come out read-only and tracked (nvn_dirty_protect over the range: the first write after it stamps the range, so a pool's
// textures upload again after its placement); others come out writable. No log line inside a swap: a thread held in its fault hook may hold stderr's lock
static struct { _Atomic uint64_t lo, hi; _Atomic int active; _Atomic long tid; } g_swap;
static _Atomic uint64_t g_swap_waits;   // writes held back until a swap ended
#ifdef __APPLE__
static long swap_tid(void) { uint64_t t = 0; pthread_threadid_np(NULL, &t); return (long)t; }
#else
static long swap_tid(void) { return (long)gettid(); }
#endif
static int swap_fault(void *addr, int is_write, void *uc) {   // (the host's page fault hook: no lock, no allocation)
    (void)is_write; (void)uc; uint64_t a = (uint64_t)(uintptr_t)addr;
    if (!atomic_load(&g_swap.active) || a < atomic_load(&g_swap.lo) || a >= atomic_load(&g_swap.hi) || swap_tid() == atomic_load(&g_swap.tid)) return 0;   // (the swap's own thread: to the crash handler)
    atomic_fetch_add(&g_swap_waits, 1);
    while (atomic_load(&g_swap.active)) { struct timespec ts = {0, 100000}; nanosleep(&ts, NULL); }
    return 1;   // made again, over the new pages
}
static void swap_prep(void *addr, size_t len) {   // a host write into guest memory is about to be made (dm_guest_write_prepare): after any swap of its range
    uint64_t a = (uint64_t)(uintptr_t)addr;
    while (atomic_load(&g_swap.active) && a < atomic_load(&g_swap.hi) && a + len > atomic_load(&g_swap.lo) && swap_tid() != atomic_load(&g_swap.tid)) { struct timespec ts = {0, 100000}; nanosleep(&ts, NULL); }
}
static int prot_runs(uint8_t *mem, uint64_t np, uint64_t pg, const uint64_t *bad, int prot) {   // the runs of pages not in bad given prot: 0, or -1 when an mprotect failed
    for (uint64_t i = 0; i < np; i++) { if (bad[i >> 6] >> (i & 63) & 1) continue; uint64_t j = i; while (j + 1 < np && !(bad[(j + 1) >> 6] >> ((j + 1) & 63) & 1)) j++;
        if (mprotect(mem + i * pg, (j - i + 1) * pg, prot)) return -1; i = j; }
    return 0;
}
// nvn_dirty as the swap uses it (tests give their own): the range's span, its pages written since a stamp, the stamp now, protection
typedef struct { const void *(*span)(const void *, size_t); int (*since)(const void *, const void *, size_t, uint64_t); uint64_t (*now)(void); void (*protect)(const void *, size_t); } swap_dirty;
// [mem, mem+len) swapped for the pages of fd (mapped shared at offset 0): 0 on success; on a failure the range keeps its pages and content
static int swap_pages(uint8_t *mem, uint64_t len, int fd, const swap_dirty *dt) {
    uint8_t *tmp = mmap(NULL, len, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0); if (tmp == MAP_FAILED) return -1;
    const void *span = dt ? dt->span(mem, len) : NULL; uint64_t st0 = span ? dt->now() : 0, pg = (uint64_t)getpagesize();
    atomic_store(&g_swap.lo, (uint64_t)(uintptr_t)mem); atomic_store(&g_swap.hi, (uint64_t)(uintptr_t)mem + len); atomic_store(&g_swap.tid, swap_tid()); atomic_store(&g_swap.active, 1);
    // The game's pages, some of which may be inaccessible (a range it reserved and has not mapped): found first and their protection left alone (making the range read-only would make them
    // readable), copied as zeros, and inaccessible again over the new pages, as they were (bad: a bit a page)
    uint64_t np = len / pg, nb = (np + 63) / 64, *bad = calloc(nb ? nb : 1, 8); int ok = bad != NULL, prot0 = 0;
    for (uintptr_t x = (uintptr_t)mem, e = x + len; ok && x < e;) { int err; uintptr_t f = first_unreadable(x, e, pg, &err); if (f >= e || err) break; uint64_t i = (f - (uintptr_t)mem) / pg; bad[i >> 6] |= 1ull << (i & 63); x = f + pg; }
    ok = ok && !prot_runs(mem, np, pg, bad, PROT_READ); prot0 = ok;
    if (ok) { for (uint64_t i = 0; i < np; i++) { int b = bad[i >> 6] >> (i & 63) & 1; uint64_t j = i; while (j + 1 < np && (int)(bad[(j + 1) >> 6] >> ((j + 1) & 63) & 1) == b) j++;
                  if (b) memset(tmp + i * pg, 0, (j - i + 1) * pg); else copy_readable(tmp + i * pg, mem + i * pg, (j - i + 1) * pg, pg, NULL); i = j; }   // (by runs)
              ok = !prot_runs(mem, np, pg, bad, PROT_READ); }   // (again: a page nvn_dirty made writable meanwhile, for a write that faulted before the swap began)
    if (ok && span) for (uint64_t i = 0; i < np; i++) if (!(bad[i >> 6] >> (i & 63) & 1) && dt->since(span, mem + i * pg, pg, st0)) copy_readable(tmp + i * pg, mem + i * pg, pg, pg, NULL);
    ok = ok && mmap(mem, len, span ? PROT_READ : PROT_READ | PROT_WRITE, MAP_SHARED | MAP_FIXED, fd, 0) == mem;
    if (!ok && prot0) prot_runs(mem, np, pg, bad, PROT_READ | PROT_WRITE);   // (the old pages stay)
    if (span) dt->protect(mem, len);   // read-only and tracked, as nvn_dirty's pages must be (its invariant: a page it holds writable is writable)
    for (uint64_t i = 0; bad && i < np; i++) if (bad[i >> 6] >> (i & 63) & 1) { uint64_t j = i; while (j + 1 < np && bad[(j + 1) >> 6] >> ((j + 1) & 63) & 1) j++;
        mprotect(mem + i * pg, (j - i + 1) * pg, PROT_NONE); i = j; }
    atomic_store(&g_swap.active, 0);
    munmap(tmp, len); free(bad);
    return ok ? 0 : -1;
}
// ---- end of the swap ----
// ---- Lazy placement: parts ----
// A pool of DM_VK_POOL_PART_MB (1024) or more is placed by parts: the granules of DM_VK_POOL_GRANULE_MB (16, counted from the pool's start) that a range the GPU is given touches become one
// placed part (a hardware buffer and a pool entry of its own). Mario Kart World 1.8.0 reserves a 2.37 GB texture heap with no buffer in it; Pokémon Pokopia a 3.4 GB GPU heap with a 3 KB buffer
// 38 MB into it. The granules [*g0, *g1) of [addr, addr+len) in a pool of ngran granules of gran bytes at base: 0 when none of them is placed (placed[]), -1 when one is (the range straddles a
// part placed before: no buffer holds it in one piece), -2 when the range is not in the pool
static int part_span(uint64_t base, uint64_t end, uint64_t gran, const uint8_t *placed, uint64_t addr, uint64_t len, uint32_t *g0, uint32_t *g1) {
    if (!len || addr < base || addr + len > end || addr + len < addr) return -2;
    *g0 = (uint32_t)((addr - base) / gran); *g1 = (uint32_t)((addr + len - 1 - base) / gran + 1);
    for (uint32_t g = *g0; g < *g1; g++) if (placed[g]) return -1;
    return 0;
}
// ---- end of parts ----
static pthread_mutex_t g_place_mu = PTHREAD_MUTEX_INITIALIZER;   // placements, and the removals that would move their entries
static uint64_t g_placed_bytes, g_copied_bytes, g_reserved_bytes, g_place_ns, g_places;   // placed zero copy, copied, every pool's size; placing time, placements
static int pool_lazy(void) { static int v = -1; if (v < 0) { const char *e = getenv("DM_VK_POOL_LAZY"); v = !(e && !atoi(e)); } return v; }
static uint64_t pool_part_min(void) { static uint64_t v = 1; if (v == 1) { const char *e = getenv("DM_VK_POOL_PART_MB"); v = (uint64_t)(e ? atoi(e) : 1024) << 20; } return v; }
static uint64_t pool_gran(void) { static uint64_t v; if (!v) { const char *e = getenv("DM_VK_POOL_GRANULE_MB"); int m = e ? atoi(e) : 16; v = (uint64_t)(m > 0 ? m : 16) << 20; } return v; }
// DM_VK_ZEROCOPY_MAX_MB=<n>: pools bigger than n MB are copied, not zero-copy, and kept whole (a copied pool takes no hardware buffer, so no chunks). The Legion Y700
// 2023's gralloc aborts the process (no error returned) allocating a hardware buffer for Unicorn Overlord's 860 MB pool, while its pools up to 250 MB are fine
static int zc_over(uint64_t size) { static uint64_t m = 1; if (m == 1) { const char *e = getenv("DM_VK_ZEROCOPY_MAX_MB"); m = e && atoi(e) > 0 ? (uint64_t)atoi(e) << 20 : 0; } return m && size > m; }
// [lo, hi) of entry p made GPU memory (g_place_mu held, p not in the table's other users' hands as GPU memory yet): the pages of a hardware buffer, imported as its VkBuffer (zero copy), else
// a GPU copy refreshed before each submission (copied, slow, reported once). 0: zero copy
static int place(vkpool_t *p, uint64_t lo, uint64_t hi) {
    uint64_t size = hi - lo; AHardwareBuffer *ahb = NULL; int fds[16], nfd = 0;
    int zc = (lo & 0xfff) || (size & 0xfff) ? -3 : getenv("DM_VK_NO_ZEROCOPY") ? -4 : zc_over(p->end - p->base) ? -5 : ahb_new(size, &ahb, fds, &nfd);
    if (!zc) { swap_dirty dt = {nvn_dirty_span, nvn_dirty_since_span, nvn_dirty_now, nvn_dirty_protect}; if (swap_pages((uint8_t *)(uintptr_t)lo, size, fds[0], &dt)) zc = -6; }
    if (!zc) vk_shadow_drop(lo, size);   // (its pages are the hardware buffer's now, their bytes copied from the shadow file where it had them: its blocks there go)
    for (int i = 0; i < nfd; i++) close(fds[i]);
    if (!zc && (zc = ahb_import(p, ahb, size))) {   // the pages are the buffer's already: back to anonymous ones with their content
        uint8_t *mem = (uint8_t *)(uintptr_t)lo, *save = malloc(size); if (save) memcpy(save, mem, size);
        mmap(mem, size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED, -1, 0); if (save) { memcpy(mem, save, size); free(save); }
        if (nvn_dirty_tracked(mem, size)) nvn_dirty_protect(mem, size); }
    if (ahb && zc) AHardwareBuffer_release(ahb);
    if (zc) {
        static int said; if (!said++) VKLOG("[RED] memory pool %p +%#llx could not be zero-copy (%d): a GPU copy is refreshed before each submission (slow)", (void *)(uintptr_t)lo, (unsigned long long)size, zc);
        VkBufferCreateInfo bi = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, NULL, 0, size, POOL_USAGE, VK_SHARING_MODE_EXCLUSIVE};
        VKCHK(vkCreateBuffer(V.dev, &bi, NULL, &p->buf));
        VkMemoryRequirements req; vkGetBufferMemoryRequirements(V.dev, p->buf, &req);
        VkMemoryAllocateInfo ai = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, NULL, req.size, vk_mem_type(req.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, 0)};
        VKCHK(vkAllocateMemory(V.dev, &ai, NULL, &p->mem)); VKCHK(vkBindBufferMemory(V.dev, p->buf, p->mem, 0));
        void *m; VKCHK(vkMapMemory(V.dev, p->mem, 0, VK_WHOLE_SIZE, 0, &m)); p->shadow = m; g_copied_bytes += size;
    } else { p->ahb = ahb; p->zero_copy = 1; g_placed_bytes += size; }
    return zc;
}
static uint64_t now_mono(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return (uint64_t)t.tv_sec * 1000000000ull + (uint64_t)t.tv_nsec; }
static void place_said(const vkpool_t *p, uint64_t addr, uint64_t len, uint64_t ns, int zc) {   // (after the swap: no log line inside one)
    static int n; if (n++ < 40 || !(n & (n - 1)))
        VKLOG("memory pool %#llx +%#llx: %#llx +%#llx placed (%s, %.1f ms) when the GPU was given %#llx +%#llx; placed %.0f MB zero-copy, %.0f MB copied, of %.0f MB of pools; %llu writes waited",
              (unsigned long long)p->base, (unsigned long long)(p->end - p->base), (unsigned long long)p->lo, (unsigned long long)(p->hi - p->lo), zc ? "copied" : "zero copy", ns / 1e6,
              (unsigned long long)addr, (unsigned long long)len, g_placed_bytes / 1048576.0, g_copied_bytes / 1048576.0, g_reserved_bytes / 1048576.0, (unsigned long long)atomic_load(&g_swap_waits));
}
static vkpool_t *find_placed(uint64_t addr, uint64_t len) { return pool_find(addr, len, 1, NULL); }   // g_pool_mu held: a placed entry holding the range
// The range given to the GPU: the chunk holding it placed whole, or in a pool placed by parts the granules it touches as a new part. NULL when that cannot be (across two chunks, across a part
// placed before and one not, or a part past ZC_CHUNK)
static vkpool_t *pool_need(uint64_t addr, uint64_t len) {
    pthread_mutex_lock(&g_place_mu);
    pthread_rwlock_rdlock(&g_pool_mu);
    vkpool_t *r = find_placed(addr, len), *u = r ? NULL : pool_find(addr, len, 2, NULL);
    pthread_rwlock_unlock(&g_pool_mu);
    if (!r && u && u->state == POOL_UNPLACED) {   // a chunk: placed whole
        uint64_t t0 = now_mono(); vkpool_t q = *u; int zc = place(&q, q.lo, q.hi); g_place_ns += now_mono() - t0; g_places++;
        pthread_rwlock_wrlock(&g_pool_mu); u->buf = q.buf; u->mem = q.mem; u->ahb = q.ahb; u->zero_copy = q.zero_copy; u->shadow = q.shadow; u->state = POOL_PLACED;
        __atomic_fetch_add(&g_pool_gen, 1, __ATOMIC_RELEASE); pthread_rwlock_unlock(&g_pool_mu);
        r = u; place_said(u, addr, len, now_mono() - t0, zc);
    } else if (!r && u) {   // a pool placed by parts
        uint32_t g0, g1; int k = part_span(u->base, u->end, u->gran, u->placed, addr, len, &g0, &g1);
        uint64_t plo = u->base + (uint64_t)g0 * u->gran, phi = u->base + (uint64_t)g1 * u->gran; if (phi > u->end) phi = u->end;
        if (k == -1) { if (!u->said++) VKLOG("[RED] memory pool %#llx +%#llx: %#llx +%#llx given to the GPU lies across a part placed before and one not: no buffer holds it in one piece (first case of the pool)",
                                            (unsigned long long)u->base, (unsigned long long)(u->end - u->base), (unsigned long long)addr, (unsigned long long)len); }
        else if (k == 0 && phi - plo > ZC_CHUNK) { if (!u->said++) VKLOG("[RED] memory pool %#llx +%#llx: %#llx +%#llx given to the GPU in one piece is past the %llu MB a hardware buffer may take",
                                            (unsigned long long)u->base, (unsigned long long)(u->end - u->base), (unsigned long long)addr, (unsigned long long)len, ZC_CHUNK >> 20); }
        else if (k == 0) {
            uint64_t t0 = now_mono(); vkpool_t q; memset(&q, 0, sizeof q);
            q.lo = plo; q.hi = phi; q.base = u->base; q.end = u->end; q.flags = u->flags; { static uint64_t ser = 1ull << 40; q.serial = ++ser; }
            int zc = place(&q, plo, phi); g_place_ns += now_mono() - t0; g_places++;
            pthread_rwlock_wrlock(&g_pool_mu);
            q.state = POOL_PLACED; q.part = 1; r = pool_entry(); *r = q; for (uint32_t g = g0; g < g1; g++) u->placed[g] = 1;
            pool_put(r); __atomic_fetch_add(&g_pool_gen, 1, __ATOMIC_RELEASE);
            pthread_rwlock_unlock(&g_pool_mu);
            place_said(r, addr, len, now_mono() - t0, zc);
        }
    }
    pthread_mutex_unlock(&g_place_mu);
    return r;
}
// Pools over ZC_CHUNK are made of chunks of at most that size, each a pool entry with an AHardwareBuffer of its own: the Adreno 830's gralloc allocator service
// (vendor.qti.hardware.display.allocator-service, snapalloc) aborts on a BLOB of 812 MB ("ubsan: mul-overflow" in SnapConstraintManager::AlignmentToAlignedConstraints),
// and the whole Android UI restarts with it; 400 MB ones work (13 Sentinels: Aegis Rim's 0x32c00000-byte pool). A range across two chunks has no buffer (vk_pool_of)
// Lazy placement (DM_VK_POOL_LAZY, default on): a pool is GPU memory only once the GPU is given an address in it (vk_pool_gpu: a buffer bound, copied, drawn from), as driver memory is
// committed whole when allocated while games reserve far more pool memory than they fill (Mario Kart World 1.8.0 at its first frames: 4,641 MB of pools, 765 MB of them not zero; placed lazily,
// with the large pools by parts, 558 MB placed, measured with another port on the same phone). Placing costs about 0.7 ms a MB on the thread that asks (the replay's). DM_VK_POOL_LAZY=0: every
// pool placed when it is made, in chunks, as before
static void pool_add_one(const void *mem, uint64_t size, uint64_t base, uint64_t end, uint64_t gran) {
    pthread_rwlock_wrlock(&g_pool_mu);
    vkpool_t *p = pool_entry();
    p->lo = (uint64_t)(uintptr_t)mem; p->hi = p->lo + size; p->base = base; p->end = end; { static uint64_t ser; p->serial = ++ser; }
    if (gran) { p->state = POOL_PARTS; p->gran = gran; p->ngran = (uint32_t)((size + gran - 1) / gran); if (!(p->placed = calloc(p->ngran, 1))) die("[vk] pool parts"); }
    // over memory an entry holds already (a pool made over another's memory): the search no longer answers every lookup alone, the scan takes what it misses while this one lives
    { vkpool_t *o = pool_find(p->lo, 1, 0, NULL); int i = pool_first_ge(p->lo); if (!o && i < g_npool && g_ord[i]->lo < p->hi) o = g_ord[i];
      if (o) { p->ovl = 1; g_novl++; static int n; if (n++ < 8) VKLOG("[RED] memory pool %#llx +%#llx lies over the memory of pool %#llx +%#llx: lookups the search misses scan the table while it lives",
                                                                      (unsigned long long)p->lo, (unsigned long long)size, (unsigned long long)o->base, (unsigned long long)(o->end - o->base)); } }
    pool_put(p); __atomic_fetch_add(&g_pool_gen, 1, __ATOMIC_RELEASE);
    pthread_rwlock_unlock(&g_pool_mu);
}
void mtl_pool_add(const void *mem, uint64_t size) {
    vk_init();
    if (!size) return;
    uint64_t a = (uint64_t)(uintptr_t)mem, c = ((a | size) & 0xfff) || zc_over(size) ? size : ZC_CHUNK;
    int parts = pool_lazy() && pool_part_min() && size >= pool_part_min() && c == ZC_CHUNK;
    pthread_mutex_lock(&g_place_mu); g_reserved_bytes += size; pthread_mutex_unlock(&g_place_mu);
    if (parts) pool_add_one(mem, size, a, a + size, pool_gran());
    else for (uint64_t o = 0; o < size; o += c) { pool_add_one((const uint8_t *)mem + o, size - o < c ? size - o : c, a, a + size, 0); if (!pool_lazy()) pool_need(a + o, 1); }
    static int nl; if (nl++ < 24 || size >= pool_part_min()) VKLOG("memory pool %p +%#llx: %s", mem, (unsigned long long)size, parts ? "placed by parts when the GPU is given them" :
        !pool_lazy() ? "placed now (DM_VK_POOL_LAZY=0)" : size > c ? "chunks placed when the GPU is given them" : "placed when the GPU is given it");
}
// A zero-copy pool's dma-buf mapped again over [mem, mem+len) in one piece per chunk or part, read-only (nvn_dirty_defrag: the mappings its protection split never merge); its pages not placed yet,
// still the game's own, made read-only in one call per run (anonymous mappings of one protection merge). 0 = done
int vk_pool_remap_ro(void *mem, size_t len) {
    int rc = 0; uint64_t done = 0, lo = (uint64_t)(uintptr_t)mem, hi = lo + len;
    pthread_rwlock_rdlock(&g_pool_mu);
    for (int i = g_novl ? 0 : pool_first_ge(lo); i < g_npool && !rc && (g_novl || g_ord[i]->lo < hi); i++) {   // (its entries start in [lo, hi), from lo on)
        vkpool_t *p = g_ord[i]; if (p->base != lo || p->end != hi) continue;
        if (p->state == POOL_UNPLACED) { rc = mprotect((void *)(uintptr_t)p->lo, p->hi - p->lo, PROT_READ); if (!rc) done += p->hi - p->lo; continue; }
        if (p->state == POOL_PARTS) {
            for (uint32_t g = 0; g < p->ngran && !rc; g++) { if (p->placed[g]) continue; uint64_t a = p->base + (uint64_t)g * p->gran, b = a + p->gran > p->end ? p->end : a + p->gran;
                rc = mprotect((void *)(uintptr_t)a, b - a, PROT_READ); if (!rc) done += b - a; }
            continue; }
        if (!p->zero_copy) { if (!p->shadow) { rc = -1; break; } done += p->hi - p->lo; continue; }   // a copied pool: nothing to map again (its protection stays as it is), and done, so defrag stops taking its pages busy
        int sv[2], fds[16], nfd = -1; if (socketpair(AF_UNIX, SOCK_SEQPACKET | SOCK_CLOEXEC, 0, sv)) { rc = -1; break; }
        if (!AHardwareBuffer_sendHandleToUnixSocket(p->ahb, sv[0])) nfd = recv_fds(sv[1], fds, 16);
        close(sv[0]); close(sv[1]);
        void *at = (void *)(uintptr_t)p->lo; rc = nfd >= 1 && mmap(at, p->hi - p->lo, PROT_READ, MAP_SHARED | MAP_FIXED, fds[0], 0) == at ? 0 : -1;
        for (int k = 0; k < nfd; k++) close(fds[k]);
        if (!rc) done += p->hi - p->lo;
    }
    pthread_rwlock_unlock(&g_pool_mu);
    return !rc && done == len ? 0 : -1;
}
static void pool_copy_back(vkpool_t *p) {   // the pages go back to anonymous memory with the buffer's contents (the game may reuse the memory)
    uint64_t size = p->hi - p->lo; uint8_t *mem = (uint8_t *)(uintptr_t)p->lo;
    uint8_t *save = malloc(size); if (save) memcpy(save, mem, size);
    mmap(mem, size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED, -1, 0);
    if (save) { memcpy(mem, save, size); free(save); }
}
void mtl_pool_del(const void *mem) {
    pthread_mutex_lock(&g_place_mu);   // (a placement in progress keeps its entry where it is)
    for (int first = 1;; first = 0) {   // every chunk and part of the pool: the first entry from mem on is one of them while any is left (without pools over the same memory)
        pthread_rwlock_wrlock(&g_pool_mu);
        uint64_t m = (uint64_t)(uintptr_t)mem; int i = pool_first_ge(m);
        if (g_novl) for (i = 0; i < g_npool && g_ord[i]->base != m; i++) {}
        if (i == g_npool || g_ord[i]->base != m) { pthread_rwlock_unlock(&g_pool_mu); break; }
        vkpool_t p = *g_ord[i]; pool_take(i); __atomic_fetch_add(&g_pool_gen, 1, __ATOMIC_RELEASE);
        { int vk_keep_buf_named(VkBuffer b); if (p.buf && vk_keep_buf_named(p.buf)) __atomic_fetch_add(&g_pool_gone, 1, __ATOMIC_RELEASE); }   // (a kept descriptor set names its buffer)
        pthread_rwlock_unlock(&g_pool_mu);
        if (first) g_reserved_bytes -= p.end - p.base;
        if (p.state == POOL_PARTS) { free(p.placed); continue; }
        if (p.state == POOL_UNPLACED) continue;
        if (p.zero_copy) { pool_copy_back(&p); g_placed_bytes -= p.hi - p.lo; } else g_copied_bytes -= p.hi - p.lo;
        vk_defer_free_buffer(p.buf, p.mem);   // the AHardwareBuffer goes with the imported memory's last reference
        if (p.ahb) AHardwareBuffer_release(p.ahb);
    }
    pthread_mutex_unlock(&g_place_mu);
}
// What the CPU knows of the pool holding [addr, addr+len): any entry of it (a chunk or part, placed or not). The GPU's lookups go through vk_pool_gpu
vkpool_t *vk_pool_of(uint64_t addr, uint64_t len) {
    // the last pools this thread found, while none was added or removed: the replay looks up every buffer binding of every draw, and the read lock alone was ~6% of it
    static __thread struct { uint32_t gen; vkpool_t *p[4]; } c;   // gen: g_pool_gen + 1 (0: empty)
    uint32_t g = __atomic_load_n(&g_pool_gen, __ATOMIC_ACQUIRE) + 1;
    if (c.gen == g) for (int i = 0; i < 4; i++) { vkpool_t *p = c.p[i]; if (p && addr >= p->lo && addr + len <= p->hi) return p; }
    int across = 0;
    pthread_rwlock_rdlock(&g_pool_mu); vkpool_t *r = pool_find(addr, len, 0, &across); pthread_rwlock_unlock(&g_pool_mu);
    // (a pool placed by parts is not kept: it holds its parts' ranges too, where the part is the answer, and vk_unreadable_at would take a placed part's pages for the game's own)
    if (r && r->state != POOL_PARTS) { if (c.gen != g) { memset(&c, 0, sizeof c); c.gen = g; } memmove(&c.p[1], &c.p[0], 3 * sizeof *c.p); c.p[0] = r; }
    if (!r && across) { static uint64_t n; uint64_t k = ++n; if (!(k & (k - 1))) LOG("[RED] [vk] %llu ranges across two chunks of a memory pool (ZC_CHUNK; this one %#llx +%#llx): no buffer for them", (unsigned long long)k, (unsigned long long)addr, (unsigned long long)len); }
    return r;
}
vkpool_t *vk_pool_gpu(uint64_t addr, uint64_t len) {
    static __thread struct { uint32_t gen; vkpool_t *p[4]; } c;   // placed entries this thread found (gen: g_pool_gen + 1)
    if (!len) len = 1;
    uint32_t g = __atomic_load_n(&g_pool_gen, __ATOMIC_ACQUIRE) + 1;
    if (c.gen == g) for (int i = 0; i < 4; i++) { vkpool_t *p = c.p[i]; if (p && addr >= p->lo && addr + len <= p->hi) return p; }
    pthread_rwlock_rdlock(&g_pool_mu); vkpool_t *r = find_placed(addr, len); pthread_rwlock_unlock(&g_pool_mu);
    if (!r) { r = pool_need(addr, len); if (!r) { vk_pool_of(addr, len); return NULL; } g = __atomic_load_n(&g_pool_gen, __ATOMIC_ACQUIRE) + 1; }   // (vk_pool_of: the across-chunks report)
    if (c.gen != g) { memset(&c, 0, sizeof c); c.gen = g; } memmove(&c.p[1], &c.p[0], 3 * sizeof *c.p); c.p[0] = r;
    return r;
}
// Virtual pools on Android (hle_nvn.c nvn_virt_xlat): a virtual pool's own range has no CPU alias of the physical pages behind it there, and is never dereferenced. An address in it is
// translated to the physical memory behind it (an aliased pool at its recorded offset, a sparse one where its pages are mapped and consecutive) for the CPU (vk_cpu_ptr) and for a GPU
// buffer (vk_pool_gpu_at, vk_buf_at); textures of a sparse pool are read through its page table (gfx_vk_tex.c)
int nvn_virt_xlat(uint64_t a, uint64_t n, uint64_t *out);
// The offset in [p, p+n) of the first page the CPU cannot read (n: every page readable; UINT64_MAX: it cannot be told, no process_vm_readv). A placed pool's pages are readable (our own
// mapping); the others are tried a byte a page, and the last ranges found readable kept per thread for the submission while the guest's mappings stay as they are (svc_alias_mapped)
uint64_t svc_alias_mapped(void); uint64_t svc_alias_unmapped_at(uint64_t a, uint64_t n); void svc_alias_range(uint8_t **base, size_t *size);   // (hle_svc.c: the alias region's page bitmap)
uint64_t vk_unreadable_at(const void *p, uint64_t n) {
    static __thread struct { uint64_t gen, sub; uintptr_t lo[8], hi[8]; int k; } c;
    if (!n) return 0;
    uintptr_t pg = (uintptr_t)getpagesize(), a = (uintptr_t)p & ~(pg - 1), e = ((uintptr_t)p + n + pg - 1) & ~(pg - 1);
    // the alias region (the game's address regions, where its graphics memory lies): the guest's own page bitmap says, placed pools there included (a page the game had not mapped stays
    // inaccessible over the pool's new pages: swap_pages)
    { uint64_t u = svc_alias_unmapped_at((uint64_t)(uintptr_t)p, n); if (u < n) return u;
      uint8_t *ab; size_t an; svc_alias_range(&ab, &an); if (ab && (uintptr_t)p >= (uintptr_t)ab && (uintptr_t)p + n <= (uintptr_t)ab + an) return n; }
    { vkpool_t *q = vk_pool_of((uint64_t)(uintptr_t)p, n); if (q && q->state == POOL_PLACED && q->zero_copy) return n; }   // (elsewhere a placed pool has no such page)
    uint64_t gen = svc_alias_mapped(), sub = vk_sub_serial();
    if (c.gen != gen || c.sub != sub) { memset(&c, 0, sizeof c); c.gen = gen; c.sub = sub; }
    for (int i = 0; i < 8; i++) if (a >= c.lo[i] && e <= c.hi[i]) return n;
    for (uintptr_t x = a; x < e;) {
        int err; uintptr_t f = first_unreadable(x, e, pg, &err); if (f >= e) break;
        if (err == ENOSYS || err == EPERM) { static int said; if (!said++) VKLOG("[RED] guest memory cannot be checked before it is read here (errno %d): read as it is", err); return UINT64_MAX; }
        vkpool_t *q = vk_pool_of((uint64_t)f, pg); if (!(q && q->state == POOL_PLACED && q->zero_copy)) return f > (uintptr_t)p ? f - (uintptr_t)p : 0;   // (a placed pool's dma-buf page: readable, refused by process_vm_readv)
        x = f + pg;
    }
    c.lo[c.k] = a; c.hi[c.k] = e; c.k = (c.k + 1) & 7;
    return n;
}
int vk_readable(const void *p, uint64_t n) { uint64_t u = vk_unreadable_at(p, n); return u == UINT64_MAX ? -1 : u >= n; }
void *vk_cpu_ptr(uint64_t a, uint64_t n) {   // the host pointer [a, a+n) is read or written through; NULL: a virtual range that cannot be had in one piece, or pages not readable
    if (!n) n = 1;
    if (vk_pool_of(a, n)) return vk_readable((const void *)(uintptr_t)a, n) ? (void *)(uintptr_t)a : NULL;
    uint64_t t; int r = nvn_virt_xlat(a, n, &t);
    return r == 1 ? (vk_readable((const void *)(uintptr_t)t, n) ? (void *)(uintptr_t)t : NULL) : r < 0 ? NULL : (void *)(uintptr_t)a;
}
vkpool_t *vk_pool_gpu_at(uint64_t *a, uint64_t n) {   // vk_pool_gpu, with *a translated when it lies in a virtual pool's range
    if (!n) n = 1;
    vkpool_t *p = vk_pool_gpu(*a, n); if (p) return p;
    uint64_t t; if (nvn_virt_xlat(*a, n, &t) == 1 && (p = vk_pool_gpu(t, n))) *a = t;
    return p;
}
uint32_t vk_pool_gen(void) { return __atomic_load_n(&g_pool_gen, __ATOMIC_ACQUIRE); }   // pools added or removed (gfx_vk_exec.c's descriptor set memo: vk_pool_of's results hold while it is unchanged)
int vk_buf_at(uint64_t addr, uint64_t len, VkBuffer *buf, VkDeviceSize *off) {
    vkpool_t *p = vk_pool_gpu_at(&addr, len ? len : 1);
    if (!p) return 0;
    *buf = p->buf; *off = addr - p->lo; return 1;
}
int mtl_addr(uint64_t addr, uint64_t len, void **buf, uint64_t *off) {
    vkpool_t *p = vk_pool_of(addr, len ? len : 1);
    if (!p) return 0;
    if (buf) *buf = (void *)(uintptr_t)p->lo; if (off) *off = addr - p->lo; return 1;
}
void mtl_alias_drop_pool(const void *pool) { void vk_tex_drop_pool(const void *); vk_tex_drop_pool(pool); }
// The NVN flags of the pool at mem (hle_nvn.c, after mtl_pool_add): a copied pool the CPU maps uncached (CPU_UNCACHED, 0x2) is one the game writes and the GPU reads;
// the game reads GPU results from cached memory, so such a pool is not copied back after each submission
void vk_pool_set_flags(const void *mem, int flags) {
    pthread_rwlock_rdlock(&g_pool_mu); uint64_t m = (uint64_t)(uintptr_t)mem;
    for (int i = g_novl ? 0 : pool_first_ge(m); i < g_npool && (g_novl || g_ord[i]->base == m); i++) if (g_ord[i]->base == m) g_ord[i]->flags = flags;   // (every chunk and part of it)
    pthread_rwlock_unlock(&g_pool_mu);
}
static uint64_t g_sync_bytes, g_sync_full;   // copied pools: bytes copied by the incremental sync, full copies
// Copied pools: the guest's bytes into the GPU copy. The first sync copies the whole pool; later ones copy only the 64 KB blocks with pages the CPU wrote since the last
// sync (nvn_dirty's write-protect tracking: a block is protected again before it is copied, so a write during the copy is caught by the next sync). An 860 MB
// pool copied whole at every submission (Unicorn Overlord on a Legion Y700 2023) ran at a few frames a second
static void vk_pools_sync_(VkCommandBuffer cb);
void vk_pools_sync(VkCommandBuffer cb) { extern int g_prof_on; extern uint64_t g_prof_ns[8]; struct timespec a, b; if (g_prof_on) clock_gettime(CLOCK_MONOTONIC, &a);
    vk_pools_sync_(cb); if (g_prof_on) { clock_gettime(CLOCK_MONOTONIC, &b); g_prof_ns[2] += (uint64_t)((b.tv_sec - a.tv_sec) * 1000000000ll + (b.tv_nsec - a.tv_nsec)); } }
static void vk_pools_sync_(VkCommandBuffer cb) {
    (void)cb;
    enum { BLK = 64 << 10 };
    pthread_rwlock_rdlock(&g_pool_mu);
    for (int i = 0; i < g_npool; i++) {
        vkpool_t *p = g_ord[i]; if (!p->shadow) continue;
        uint8_t *lo = (uint8_t *)(uintptr_t)p->lo; uint64_t size = p->hi - p->lo, st = nvn_dirty_now();
        if (!p->span && (p->span = nvn_dirty_span(lo, size))) p->synced = 0;   // tracked only now (nvnMemoryPoolInitialize registers the range after the pool): copied whole once more, protected
        if (!p->synced || !p->span) {   // untracked: every sync copies all of it, as before
            if (p->span) nvn_dirty_protect(lo, size);
            copy_readable(p->shadow, lo, size, (uint64_t)getpagesize(), NULL); g_sync_bytes += size; g_sync_full++; p->synced = 1; p->stamp0 = p->stamp = st; continue; }
        // judged against the sync before the last: a write whose fault took its generation before the last sync's stamp but stored it after that sync looked
        // at its block would otherwise never count (its page is writable from then on, and clean by every later stamp). A dirty block is copied twice
        for (uint64_t off = 0; off < size; off += BLK) {
            uint64_t n = size - off < BLK ? size - off : BLK;
            if (!nvn_dirty_since_span(p->span, lo + off, n, p->stamp0)) continue;
            nvn_dirty_protect(lo + off, n); copy_readable(p->shadow + off, lo + off, n, (uint64_t)getpagesize(), NULL); g_sync_bytes += n;
        }
        p->stamp0 = p->stamp; p->stamp = st;
    }
    pthread_rwlock_unlock(&g_pool_mu);
    static uint64_t n, last; if (++n % 600 == 0) { VKLOG("copied pools: %.1f MB synced in the last 600 submissions (%.2f MB each), %llu full copies so far", (g_sync_bytes - last) / 1048576.0, (g_sync_bytes - last) / 600.0 / 1048576.0, (unsigned long long)g_sync_full); last = g_sync_bytes; }
}
// Copied pools after a submission: what the GPU wrote, back into the guest's pages. A pool never synced yet is left alone (its copy holds nothing of the game's).
// A CPU-uncached pool (vk_pool_set_flags) takes back only the ranges the submission's GPU writes covered (sub_gpu_wrote):
// copying 860 MB back after every submission held Unicorn Overlord to a few frames a second
struct vksub; static void pools_copy_back(const struct vksub *s);

// ---------------- submission slots ----------------
#define NSLOT 6
#define MAIN_CBS 32   // (allocated as used: DM_VK_PASS_SPLIT cuts inside passes as well)
// GPU time: a timestamp at the start and end of each submission's main command buffer (one queue: their sum is the time the GPU was busy)
static VkQueryPool g_tsq; uint64_t g_gpu_ns;
// Timestamp counters (ReportCounter TIMESTAMP): each writes a GPU timestamp where it is in the command stream (CQ per slot), and at completion it gets the guest
// clock now less how long before the submission's end the GPU passed it. They used to all get the completion time of their submission: a game timing its GPU
// work between two counters read 0 when both were in one submission and the gap between completions (our CPU replay included) when not. Xenoblade Chronicles 3
// sizes its frame by that time (dynamic resolution): its main pass went between 960x544 and 776x440 every few frames while the camera turned; with GPU times
// the size holds steady
enum { CQ = 64 };
static VkQueryPool g_cq;
#define SCRATCH_BLOCK (8u << 20)
typedef struct { VkBuffer b; VkDeviceMemory m; uint8_t *p; VkDeviceSize size, used; uint32_t idle; } scratch_blk;   // idle: the slot's recycles since it was last used (scratch_trim)
static uint64_t g_scratch_bytes, g_scratch_freed;   // scratch blocks made and live, and freed by scratch_trim (vk_mem_log)
typedef struct { int kind; void *a; void *b; } dfree_t;   // 0 buffer+memory, 1 image+memory, 2 view
typedef struct { int kind; volatile uint32_t *val; uint32_t mode, v; uint64_t addr; uint32_t type; int q; void *p; } post_op;   // q: a timestamp counter's query in g_cq, or -1; p: kind 3's capture
struct vksub {
    VkCommandPool pool; VkCommandBuffer up, main; int up_used;
    VkCommandBuffer mains[MAIN_CBS]; int nmain_alloc, nmain_used, cb_err;
    VkCommandBuffer ups[MAIN_CBS]; int nup_alloc, nup_used; int nmain_sent;   // upload command buffers in order (one per part sent early), main buffers already submitted   // the submission's main command buffers in order (vk_cmd_split); main = the one being recorded
    uint64_t fence;   // the fence this slot last committed with (0 = never)
    int ncq;          // timestamp counter queries used in this submission (g_cq)
    scratch_blk *blk; int nblk, capblk;
    VkDescriptorPool *dp; int ndp, capdp, dp_cur;
    uint32_t dp_used[9], dp_sets; int dp_counted; int no_ts;   // what pool dp_cur has given out (vk_alloc_set)
    dfree_t *df; int ndf, capdf;
    post_op *post; int npost, cappost;
    struct { uint64_t lo, hi; } gw[64]; int ngw;   // GPU writes into pool memory in this submission (sub_gpu_wrote), for pools_copy_back: nearby ones merged, and once full each joins the nearest
};
enum { GW_SUB = 64 };
static vksub_t g_slot[NSLOT]; static int g_cur = -1; uint64_t g_cmd_gen;   // g_cmd_gen: main command buffers begun (gfx_vk_exec.c bind cache)
static void pools_copy_back(const struct vksub *s) {
    pthread_rwlock_rdlock(&g_pool_mu);
    for (int i = 0; i < g_npool; i++) { vkpool_t *p = g_ord[i]; if (!p->shadow || !p->synced) continue;
        if (!(p->flags & 0x2)) { memcpy((void *)(uintptr_t)p->lo, p->shadow, p->hi - p->lo); continue; }
        for (int k = 0; k < s->ngw; k++) { uint64_t lo = s->gw[k].lo > p->lo ? s->gw[k].lo : p->lo, hi = s->gw[k].hi < p->hi ? s->gw[k].hi : p->hi;
            if (lo < hi) memcpy((void *)(uintptr_t)lo, p->shadow + (lo - p->lo), hi - lo); } }
    pthread_rwlock_unlock(&g_pool_mu);
}
static VkCommandBuffer g_slot_main_cb(void) { return g_slot[g_cur].main; }
// ---------------- GPU profile (DM_VK_GPUPROF, live; DM_VK_PASS_TIMES=1 at start-up turns it on from the first frame) ----------------
// Run 13 of Mario Kart World 1.8.0 (mesh draws, the alpha test and first stores on) was GPU-bound, 35-38 ms of GPU a frame, against 16 ms at 720p before them: where does it go. Each
// submission's main command buffers are cut into units, a render pass or the stretch between two passes (dispatches, copies, clears, uniform landings, LRZ touches, barriers), and a
// timestamp at each boundary, at the bottom of the pipe, so a unit's span is its own work after all before it and the spans add up to the submission's (what lies outside them: the upload
// command buffers, the gaps between submissions). Each unit carries what it held (vk_prof_t, gfx_vk_exec.c). The completion thread sums them by the unit's place in its frame; every
// DM_VK_GPUPROF_FRAMES presents (live, default 300; DM_VK_GPUPROF=<n> over 1 is that period too) two lines give the totals by kind and the costliest passes and stretches. The timestamps
// cost time of their own and disturb Adreno's binning: compare runs with the profile on
enum { PASSQ = 512 };   // units timed per submission (boundaries: one more)
static VkQueryPool g_pq; static int g_pq_on = -1;   // g_pq_on: DM_VK_PASS_TIMES (start-up)
enum { FPASSQ = 512 }; static int g_fpass;   // units of the frame so far
static vk_prof_t g_pmeta[NSLOT][PASSQ]; static int g_npass[NSLOT], g_sub_prof[NSLOT];   // per slot: its units, and whether its submission is timed
uint64_t g_bar_full, g_bar_merged;
// ---- GPU profile: the sums (tests: tests/test_vk_gpuprof.py) ----
enum { PK_PASS, PK_MESH, PK_ALPHA, PK_FDZ, PK_GAP, PK_DISP, PK_COPY, PK_BAR, PK_N };   // kinds of unit: passes, those with mesh / alpha-tested / depth-variant draws; stretches, with dispatches, copies only, barriers only
enum { PC_DRAWS, PC_MESH, PC_ALPHA, PC_FDZ, PC_FDEPTH, PC_DISP, PC_SPILL, PC_FIRST, PC_COPIES, PC_CLEARS, PC_BARS, PC_MERGED, PC_TOUCHES, PC_UBD,
       PC_DSTORE, PC_DLOAD, PC_DK_END, PC_DK_USED, PC_DK_HIST, PC_N };   // (PC_D*: NVN discards made DONT_CARE stores and loads, stores kept: gfx_vk_exec.c "discards")
typedef struct {
    double ms[FPASSQ]; uint32_t n[FPASSQ]; vk_prof_t last[FPASSQ]; int max;   // by the unit's place in its frame
    double kms[PK_N]; uint64_t kn[PK_N], cnt[PC_N]; uint32_t ends[256];       // by kind; counters summed; passes ended, by the op that ended them
    double sub_ms, units_ms; uint64_t frames, last_frame;                     // the submissions' own spans; frames seen
} prof_sums;
static void prof_add(prof_sums *s, const vk_prof_t *u, double ms, uint64_t frame) {   // one unit of a timed submission, ms its span
    if (frame + 1 != s->last_frame) { s->frames++; s->last_frame = frame + 1; }
    uint32_t f = u->fidx < FPASSQ ? u->fidx : FPASSQ - 1; s->ms[f] += ms; s->n[f]++; s->last[f] = *u; if ((int)f + 1 > s->max) s->max = (int)f + 1; s->units_ms += ms;
    const uint16_t c[PC_N] = {(uint16_t)(u->draws > 65535 ? 65535 : u->draws), u->mesh, u->alpha, u->fdz, u->fdepth, u->disp, u->spill, u->first, u->copies, u->clears, u->bars, u->merged, u->touches, u->ubd,
                              u->dstore, u->dload, u->dk_end, u->dk_used, u->dk_hist};
    for (int i = 0; i < PC_N; i++) s->cnt[i] += c[i];
    int k[4], nk = 0;
    if (u->kind == VK_PU_PASS) { k[nk++] = PK_PASS; if (u->mesh) k[nk++] = PK_MESH; if (u->alpha) k[nk++] = PK_ALPHA; if (u->fdz) k[nk++] = PK_FDZ; s->ends[u->end_op]++; }
    else { k[nk++] = PK_GAP; k[nk++] = u->disp ? PK_DISP : u->copies || u->clears || u->ubd || u->first || u->touches ? PK_COPY : PK_BAR; }
    for (int i = 0; i < nk; i++) { s->kms[k[i]] += ms; s->kn[k[i]]++; }
}
static int prof_cmp_ms(const prof_sums *s, int a, int b) { double x = s->n[a] ? s->ms[a] / s->n[a] : -1, y = s->n[b] ? s->ms[b] / s->n[b] : -1; return x < y ? 1 : x > y ? -1 : a - b; }
static void prof_ops(char *b, int i) { b[0] = "LCDD"[i & 3]; b[1] = 0; }
// the period's two lines into b1 and b2: the totals a frame (and the passes' ends, by op name), then the 15 costliest passes and the 5 costliest stretches between them
static void prof_lines(const prof_sums *s, char *b1, size_t n1, char *b2, size_t n2, const char *(*opname)(int)) {
    double fr = s->frames ? (double)s->frames : 1; const double *km = s->kms; const uint64_t *kn = s->kn, *c = s->cnt;
    int o = snprintf(b1, n1, "GPU profile, %llu frames: timed units %.2f ms a frame (submissions %.2f ms, %.2f outside the units: upload command buffers, gaps); passes %.1f a frame, %.2f ms "
        "(with mesh draws %.1f: %.2f ms; with alpha-tested draws %.1f: %.2f ms; with fragment-depth-variant draws %.1f: %.2f ms); between passes %.1f, %.2f ms (with dispatches %.1f: %.2f ms; "
        "copies, clears, uniform landings, LRZ touches %.1f: %.2f ms; barriers alone %.1f: %.2f ms); a frame: %.0f draws (%.0f mesh, %.0f alpha-tested, %.0f in the fragment-depth variant, "
        "%.0f whose fragment shader writes depth), %.0f dispatches (%.0f spill parts, %.0f first stores), %.0f copies, %.0f clears, %.0f full barriers (%.0f more merged into the one before), "
        "%.0f LRZ touches, %.0f uniform landings; passes ended by", (unsigned long long)s->frames, s->units_ms / fr, s->sub_ms / fr, (s->sub_ms - s->units_ms) / fr,
        kn[PK_PASS] / fr, km[PK_PASS] / fr, kn[PK_MESH] / fr, km[PK_MESH] / fr, kn[PK_ALPHA] / fr, km[PK_ALPHA] / fr, kn[PK_FDZ] / fr, km[PK_FDZ] / fr,
        kn[PK_GAP] / fr, km[PK_GAP] / fr, kn[PK_DISP] / fr, km[PK_DISP] / fr, kn[PK_COPY] / fr, km[PK_COPY] / fr, kn[PK_BAR] / fr, km[PK_BAR] / fr,
        c[PC_DRAWS] / fr, c[PC_MESH] / fr, c[PC_ALPHA] / fr, c[PC_FDZ] / fr, c[PC_FDEPTH] / fr, c[PC_DISP] / fr, c[PC_SPILL] / fr, c[PC_FIRST] / fr, c[PC_COPIES] / fr, c[PC_CLEARS] / fr,
        c[PC_BARS] / fr, c[PC_MERGED] / fr, c[PC_TOUCHES] / fr, c[PC_UBD] / fr);
    uint32_t done[256] = {0};
    for (int k = 0; k < 8 && o < (int)n1 - 48; k++) { int b = -1; for (int i = 0; i < 256; i++) if (s->ends[i] && !done[i] && (b < 0 || s->ends[i] > s->ends[b])) b = i; if (b < 0) break;
        done[b] = 1; o += snprintf(b1 + o, n1 - (size_t)o, "%s %s %.1f", k ? "," : "", opname(b), s->ends[b] / fr); }
    if (o < (int)n1 - 200) o += snprintf(b1 + o, n1 - (size_t)o, "; NVN discards a frame: %.1f stores and %.1f loads made DONT_CARE, stores kept %.1f (the pass's end not certain), %.1f (used again "
                                         "in the submission), %.1f (its image's history: not yet shown re-cleared, or read while discarded)", c[PC_DSTORE] / fr, c[PC_DLOAD] / fr, c[PC_DK_END] / fr, c[PC_DK_USED] / fr, c[PC_DK_HIST] / fr);
    int idx[FPASSQ], m = s->max; for (int i = 0; i < m; i++) idx[i] = i;
    for (int i = 1; i < m; i++) { int v = idx[i], j = i; while (j > 0 && prof_cmp_ms(s, idx[j - 1], v) > 0) { idx[j] = idx[j - 1]; j--; } idx[j] = v; }   // (insertion sort: a few hundred places)
    o = snprintf(b2, n2, "passes by GPU time (the 15 costliest of %d places in a frame):", m);
    for (int i = 0, shown = 0; i < m && shown < 15 && o < (int)n2 - 200; i++) { const vk_prof_t *u = &s->last[idx[i]]; if (!s->n[idx[i]] || u->kind != VK_PU_PASS) continue; shown++;
        char ld[12] = "", st[12] = ""; int a = 0, z = 0; for (uint32_t q = 0; q < u->ncol && q < 4; q++) { prof_ops(ld + a++, u->load >> (2 * q)); st[z++] = u->store >> q & 1 ? 'D' : 'S'; }
        if (u->dfmt) { ld[a++] = '/'; prof_ops(ld + a++, u->load >> 8); st[z++] = '/'; st[z++] = u->store >> 8 & 1 ? 'D' : 'S'; } ld[a] = 0; st[z] = 0;
        o += snprintf(b2 + o, n2 - (size_t)o, " #%d %.2f ms %ux%u colour %u %u %u %u depth %u, load %s store %s, %u draws (%u mesh, %u alpha-tested, %u depth variant), ended by %s;", idx[i],
                      s->ms[idx[i]] / s->n[idx[i]], u->w, u->h, u->cfmt[0], u->cfmt[1], u->cfmt[2], u->cfmt[3], u->dfmt, ld, st, u->draws, u->mesh, u->alpha, u->fdz, opname(u->end_op)); }
    if (o < (int)n2 - 64) o += snprintf(b2 + o, n2 - (size_t)o, " | between passes, the 5 costliest:");
    for (int i = 0, shown = 0; i < m && shown < 5 && o < (int)n2 - 120; i++) { const vk_prof_t *u = &s->last[idx[i]]; if (!s->n[idx[i]] || u->kind != VK_PU_GAP) continue; shown++;
        o += snprintf(b2 + o, n2 - (size_t)o, " #%d %.2f ms: %u dispatches (%u spill parts), %u copies, %u clears, %u barriers (%u merged);", idx[i], s->ms[idx[i]] / s->n[idx[i]], u->disp, u->spill,
                      u->copies, u->clears, u->bars, u->merged); }
}
// ---- end of the GPU profile's sums ----
static prof_sums g_ps; static pthread_mutex_t g_ps_mu = PTHREAD_MUTEX_INITIALIZER;   // the completion thread adds, the present thread prints and clears
static int g_prof_period = 300, g_prof_frames;
static int prof_wanted(void) {   // read at each submission's start (DM_VK_GPUPROF live, DM_VK_PASS_TIMES start-up)
    if (g_pq_on < 0) { const char *e = getenv("DM_VK_PASS_TIMES"); g_pq_on = e && atoi(e); }
    const char *e = getenv("DM_VK_GPUPROF"), *f = getenv("DM_VK_GPUPROF_FRAMES"); int v = e ? atoi(e) : 0;
    g_prof_period = v > 1 ? v : f && atoi(f) > 0 ? atoi(f) : 300;
    return g_pq && (v > 0 || g_pq_on);
}
int vk_prof_on(void) { return g_cur >= 0 && g_sub_prof[g_cur]; }
void vk_prof_unit(const vk_prof_t *u) {
    if (!vk_prof_on()) return; int *n = &g_npass[g_cur]; if (*n >= PASSQ) return;
    vk_prof_t *m = &g_pmeta[g_cur][*n]; *m = *u; m->fidx = (uint32_t)(g_fpass < FPASSQ - 1 ? g_fpass : FPASSQ - 1); (*n)++; g_fpass++;
    vkCmdWriteTimestamp(g_slot_main_cb(), VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, g_pq, (uint32_t)(g_cur * (PASSQ + 1) + *n));   // (boundary n: the end of unit n - 1)
}
static void prof_sub_begin(vksub_t *s) {   // a submission's start: timed or not, its first boundary
    g_npass[g_cur] = 0; g_sub_prof[g_cur] = prof_wanted(); if (!g_sub_prof[g_cur]) return;
    vkCmdResetQueryPool(s->main, g_pq, (uint32_t)(g_cur * (PASSQ + 1)), PASSQ + 1);
    vkCmdWriteTimestamp(s->main, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, g_pq, (uint32_t)(g_cur * (PASSQ + 1)));
}
const char *vk_prof_op_name(int op);   // (gfx_vk_exec.c)
void vk_pass_report(void) {}   // (the profile reports on its own period: vk_pass_frame)
void vk_pass_frame(void) {   // a present: the next unit is the frame's first; the period's lines when it is up
    g_fpass = 0;
    if (!g_pq || (!g_prof_frames && !g_ps.frames) || ++g_prof_frames < g_prof_period) return;
    g_prof_frames = 0; static char b1[3000], b2[4000];
    pthread_mutex_lock(&g_ps_mu); int any = g_ps.frames > 0; if (any) prof_lines(&g_ps, b1, sizeof b1, b2, sizeof b2, vk_prof_op_name); memset(&g_ps, 0, sizeof g_ps); pthread_mutex_unlock(&g_ps_mu);
    if (!any) return;
    { extern uint64_t g_nvn_discards[2]; static uint64_t pd0, pd1; size_t l = strlen(b1); snprintf(b1 + l, sizeof b1 - l, "; NVN discard calls since: colour %llu, depth / stencil %llu (DM_VK_DISCARD: 1 depth / stencil, 2 colour too)",
        (unsigned long long)(g_nvn_discards[0] - pd0), (unsigned long long)(g_nvn_discards[1] - pd1)); pd0 = g_nvn_discards[0]; pd1 = g_nvn_discards[1]; }
    VKLOG("%s", b1); VKLOG("%s", b2);
}
// one submission at a time: the game's (hle_nvn.c holds its own lock around them) and the backend's own (texel readback, present)
static pthread_mutex_t g_sub_lock = PTHREAD_MUTEX_INITIALIZER;
static uint64_t g_fence;   // fence of the last commit
static VkSemaphore g_timeline;   // submission f signals the value f (V.has_timeline), else g_fring[f % NFRING]
#define NFRING 16   // more than the submissions in flight (NSLOT: mtl_sub_begin's slot_recycle waits for the slot's last one)
static VkFence g_fring[NFRING];
static pthread_mutex_t g_done_mu = PTHREAD_MUTEX_INITIALIZER; static pthread_cond_t g_done_cv = PTHREAD_COND_INITIALIZER;
static uint64_t g_done;
static void slots_init(void) {
    VkSemaphoreTypeCreateInfo ti = {VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO, NULL, VK_SEMAPHORE_TYPE_TIMELINE, 0};
    VkSemaphoreCreateInfo si = {VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO, &ti};
    if (V.has_timeline) VKCHK(vkCreateSemaphore(V.dev, &si, NULL, &g_timeline));
    else for (int i = 0; i < NFRING; i++) { VkFenceCreateInfo fi = {VK_STRUCTURE_TYPE_FENCE_CREATE_INFO}; VKCHK(vkCreateFence(V.dev, &fi, NULL, &g_fring[i])); }
    VkQueryPoolCreateInfo qi = {VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO, NULL, 0, VK_QUERY_TYPE_TIMESTAMP, 2 * NSLOT};
    if (V.props.limits.timestampComputeAndGraphics && vkCreateQueryPool(V.dev, &qi, NULL, &g_tsq)) g_tsq = VK_NULL_HANDLE;
    VkQueryPoolCreateInfo cqi = {VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO, NULL, 0, VK_QUERY_TYPE_TIMESTAMP, NSLOT * CQ};
    if (!g_tsq || vkCreateQueryPool(V.dev, &cqi, NULL, &g_cq)) g_cq = VK_NULL_HANDLE;
    if (g_tsq) { VkQueryPoolCreateInfo pq = {VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO, NULL, 0, VK_QUERY_TYPE_TIMESTAMP, NSLOT * (PASSQ + 1)}; if (vkCreateQueryPool(V.dev, &pq, NULL, &g_pq)) g_pq = VK_NULL_HANDLE; }   // (the GPU profile, live: made where timestamps are)
    for (int i = 0; i < NSLOT; i++) {
        // not TRANSIENT, and reset by command buffer (see slot_recycle)
        VkCommandPoolCreateInfo ci = {VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO, NULL, VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT, V.qf};
        VKCHK(vkCreateCommandPool(V.dev, &ci, NULL, &g_slot[i].pool));
        VkCommandBuffer cb[2]; VkCommandBufferAllocateInfo ai = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, NULL, g_slot[i].pool, VK_COMMAND_BUFFER_LEVEL_PRIMARY, 2};
        VKCHK(vkAllocateCommandBuffers(V.dev, &ai, cb)); g_slot[i].up = g_slot[i].ups[0] = cb[0]; g_slot[i].main = g_slot[i].mains[0] = cb[1]; g_slot[i].nmain_alloc = 1; g_slot[i].nup_alloc = 1;
    }
}
uint64_t vk_fence_done(void) { return __atomic_load_n(&g_done, __ATOMIC_ACQUIRE); }   // written under g_done_mu (for the waiters) with a release store
int mtl_wait(uint64_t fence, uint64_t timeout_ns);
// Without timeline semaphores only the completion thread waits on the ring's fences, in submission order (a fence tells nothing of the submissions before
// it); everyone else waits for the completion thread to pass the fence (g_done), so a fence is never waited on while the commit resets it for reuse
static void wait_fence(uint64_t f) {
    if (!f) return;
    if (!V.has_timeline) { mtl_wait(f, ~0ull); return; }
    VkSemaphoreWaitInfo wi = {VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO, NULL, 0, 1, &g_timeline, &f};
    vkWaitSemaphores(V.dev, &wi, UINT64_MAX);
}
// The slot's command buffers are reset on the completion thread once its submission is done (completion_main), not when the replay begins them again: an
// implicit reset at vkBeginCommandBuffer took 7% of Xenoblade Chronicles 3's replay thread at a submission's start and more at every part (its battles
// replay ~3,000 draws a frame on a thread that is the frame's limit). The slot is then taken only after the completion thread has passed it (mtl_wait, g_done)
// rather than as soon as the GPU has. DM_VK_X_NOPRERESET=1 (start-up): reset as they are begun
static int g_prereset = -1;
// The scratch blocks a slot keeps once the GPU is past it: those its last submission used, and up to DM_VK_SCRATCH_KEEP_MB (start-up, default 16) of others; the rest are freed. Each
// slot used to keep every block its largest submission had taken (a course's load stages its texture uploads there): Mario Kart World 1.8.0 held 394 MB of scratch at 28 s of its first
// race (run 5) over the 6 slots, with 1.2 GB of MemAvailable left. Not before the completion thread has passed the slot's fence: the uniform updates in flight read their bytes from
// scratch until then (vk_ubd_read, g_infl)
static void scratch_trim(vksub_t *s) {
    static long long keep = -1; if (keep < 0) { const char *e = getenv("DM_VK_SCRATCH_KEEP_MB"); keep = (long long)(e && atoi(e) >= 0 ? atoi(e) : 16) << 20; }
    int safe = vk_fence_done() >= s->fence, o = 0; VkDeviceSize spare = 0;
    // A spare block goes once it has been unused for SCRATCH_IDLE of the slot's recycles (about two seconds), 4 when memory is critical (at once, run 19 still freed 206 GB in 30 minutes:
    // MemAvailable sat at the critical level, and a block freed is made again by the next submission, which saves nothing): a slot's submissions differ (four
    // a frame over six slots), and one left out by the last submission was needed by the next, freed and made again (Mario Kart World's Grand Prix, run 16: 126 GB of scratch freed
    // in 23 minutes for 48 MB held, the memory pressure of the patch before freeing every spare block at each recycle)
    enum { SCRATCH_IDLE = 32 };
    int pr = vk_mem_pressure(); long long k = keep && pr ? 0 : keep;   // (memory short, gfx_vk_budget.c: the spare blocks go; MemAvailable under DM_VK_PRESSURE_MB, or the swap nearly full)
    for (int i = 0; i < s->nblk; i++) { scratch_blk *b = &s->blk[i];
        if (b->used) b->idle = 0; else if (safe) b->idle++;
        if (b->used || !safe || spare + b->size <= (VkDeviceSize)k || b->idle < (pr >= 2 ? 4u : (uint32_t)SCRATCH_IDLE)) { if (!b->used) spare += b->size; b->used = 0; s->blk[o++] = *b; continue; }
        { int vk_keep_buf_named(VkBuffer bb); if (vk_keep_buf_named(b->b)) __atomic_fetch_add(&g_pool_gone, 1, __ATOMIC_RELEASE); }   // (a kept descriptor set names it: gfx_vk_exec.c kept_check)
        vkDestroyBuffer(V.dev, b->b, NULL); vkFreeMemory(V.dev, b->m, NULL);   // (unmapped with it)
        __atomic_fetch_sub(&g_scratch_bytes, b->size, __ATOMIC_RELAXED); __atomic_fetch_add(&g_scratch_freed, b->size, __ATOMIC_RELAXED); }
    s->nblk = o;
}
static void slot_recycle(vksub_t *s) {   // the GPU has passed this slot's last submission: free what it held, rewind its allocators
    if (g_prereset < 0) { const char *e = getenv("DM_VK_X_NOPRERESET"); g_prereset = !(e && atoi(e)); }
    if (g_prereset) mtl_wait(s->fence, ~0ull); else wait_fence(s->fence);
    for (int i = 0; i < s->ndf; i++) {
        dfree_t *d = &s->df[i];
        if (d->kind == 0) { vkDestroyBuffer(V.dev, (VkBuffer)d->a, NULL); if (d->b) vkFreeMemory(V.dev, (VkDeviceMemory)d->b, NULL); }
        else if (d->kind == 1) { vkDestroyImage(V.dev, (VkImage)d->a, NULL); if (d->b) vkFreeMemory(V.dev, (VkDeviceMemory)d->b, NULL); }
        else if (d->kind == 4) { void img_sub_free_(void *); vkDestroyImage(V.dev, (VkImage)d->a, NULL); img_sub_free_(d->b); }
        else if (d->kind == 2) vkDestroyImageView(V.dev, (VkImageView)d->a, NULL);
        else if (d->kind == 3) vkDestroyBufferView(V.dev, (VkBufferView)d->a, NULL);
    }
    s->ndf = 0;
    scratch_trim(s);
    for (int i = 0; i < s->ndp; i++) vkResetDescriptorPool(V.dev, s->dp[i], 0);
    s->dp_cur = 0; s->dp_counted = -1;
    // The pool is not reset: each command buffer is reset as it is begun again (the pool allows it). A reset of the transient pool made Adreno's driver free the
    // command memory and map it again as the next submission grew (ioctl, mmap and munmap): in Xenoblade Chronicles 3's battles the submission's start took 21%
    // of the replay thread, now 9%, and frames went from about 42 to 38 ms
}
static vksub_t *cur(void) { if (g_cur < 0) die("[vk] no open submission"); return &g_slot[g_cur]; }
int vk_sub_open(void) { return g_cur >= 0; }
// A long submission is recorded into several command buffers, submitted together in order: Adreno's driver refuses a command buffer past some size at
// vkEndCommandBuffer (VK_ERROR_OUT_OF_HOST_MEMORY), which NEO's handheld frames reached at about the 712th draw. The caller has ended the render pass.
static void vk_cmd_split_(void);
void vk_cmd_split(void) { extern int g_prof_on; extern uint64_t g_prof_ns[8]; struct timespec a, b; if (g_prof_on) clock_gettime(CLOCK_MONOTONIC, &a);
    vk_cmd_split_(); if (g_prof_on) { clock_gettime(CLOCK_MONOTONIC, &b); g_prof_ns[6] += (uint64_t)((b.tv_sec - a.tv_sec) * 1000000000ll + (b.tv_nsec - a.tv_nsec)); } }
static void vk_cmd_split_(void) {
    vksub_t *s = cur(); if (s->nmain_used == MAIN_CBS || s->nup_used == MAIN_CBS) return;
    if (vkEndCommandBuffer(s->main) < 0) { s->cb_err = 1; static int n; if (n++ < 3) VKLOG("[RED] a part of a long submission was refused at its end (part %d)", s->nmain_used); }
    // what is recorded so far goes to the GPU now, as a submission of its own (with its uploads first): Adreno refuses a single submission past about
    // 700 draws (NEO's handheld frames: VK_ERROR_OUT_OF_HOST_MEMORY at vkEndCommandBuffer whatever the draws were, and in one command buffer or several)
    const char *se = getenv("DM_VK_SUBMIT_SPLIT"); int send = !(se && !atoi(se));   // on unless DM_VK_SUBMIT_SPLIT=0
    if (send && !s->cb_err) {
        VkCommandBuffer cbs[2]; uint32_t n = 0;
        if (s->up_used) { vk_img_barrier_all(s->up); if (vkEndCommandBuffer(s->up) < 0) s->cb_err = 1; cbs[n++] = s->up; }
        cbs[n++] = s->main;
        if (!s->cb_err) {
            VkSubmitInfo si = {VK_STRUCTURE_TYPE_SUBMIT_INFO, NULL, 0, NULL, NULL, n, cbs, 0, NULL};
            VkSemaphore ws; uint64_t wv; VkPipelineStageFlags wst = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT; VkTimelineSemaphoreSubmitInfo wt = {VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO, NULL, 1, &wv, 0, NULL};
            if (vk_sparse_flush(&ws, &wv)) { si.pNext = &wt; si.waitSemaphoreCount = 1; si.pWaitSemaphores = &ws; si.pWaitDstStageMask = &wst; }   // after the tile binds its uploads and draws read
            pthread_mutex_lock(&g_vk_queue_mu); VkResult r = vkQueueSubmit(V.q, 1, &si, VK_NULL_HANDLE); pthread_mutex_unlock(&g_vk_queue_mu);
            if (r) { void vk_crumbs_report(void); vk_crumbs_report(); die("[vk] vkQueueSubmit (part): %d (device lost?)", r); }
            { static int n; if (n++ < 3) VKLOG("a long submission: part %d sent early", s->nmain_used); }
            s->nmain_sent = s->nmain_used;
            if (s->up_used) {   // the next part records its uploads into a fresh upload buffer
                if (s->nup_used == s->nup_alloc) { VkCommandBufferAllocateInfo ai = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, NULL, s->pool, VK_COMMAND_BUFFER_LEVEL_PRIMARY, 1};
                    VKCHK(vkAllocateCommandBuffers(V.dev, &ai, &s->ups[s->nup_alloc])); s->nup_alloc++; }
                s->up = s->ups[s->nup_used++]; s->up_used = 0;
            }
        }
    }
    if (s->nmain_used == s->nmain_alloc) { VkCommandBufferAllocateInfo ai = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, NULL, s->pool, VK_COMMAND_BUFFER_LEVEL_PRIMARY, 1};
        VKCHK(vkAllocateCommandBuffers(V.dev, &ai, &s->mains[s->nmain_alloc])); s->nmain_alloc++; }
    s->main = s->mains[s->nmain_used++];
    VkCommandBufferBeginInfo bi = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, NULL, VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT}; VKCHK(vkBeginCommandBuffer(s->main, &bi)); g_cmd_gen++;
}   // outside a submission nothing may be recorded (present, a pool finalized between frames)
VkCommandBuffer vk_cmd(void) { return cur()->main; }
int vk_cmd_parts_left(void) { vksub_t *s = cur(); return MAIN_CBS - (s->nmain_used > s->nup_used ? s->nmain_used : s->nup_used); }   // parts vk_cmd_split can still begin (gfx_vk_exec.c pass splits)
uint64_t g_up_gen;   // upload command buffers begun (gfx_vk_tex.c lin_copy: once per upload buffer)
VkCommandBuffer vk_upload_cmd(void) {
    vksub_t *s = cur();
    if (!s->up_used) { VkCommandBufferBeginInfo bi = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, NULL, VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT}; VKCHK(vkBeginCommandBuffer(s->up, &bi)); s->up_used = 1; g_up_gen++; }
    return s->up;
}
void *vk_scratch(VkDeviceSize n, VkDeviceSize align, VkBuffer *buf, VkDeviceSize *off) {
    vksub_t *s = cur(); if (!align) align = 16;
    for (int i = 0; i < s->nblk; i++) {
        scratch_blk *b = &s->blk[i]; VkDeviceSize o = (b->used + align - 1) & ~(align - 1);
        if (o + n <= b->size) { b->used = o + n; *buf = b->b; *off = o; return b->p + o; }
    }
    if (s->nblk == s->capblk) { s->capblk = s->capblk ? 2 * s->capblk : 8; s->blk = realloc(s->blk, s->capblk * sizeof *s->blk); }
    scratch_blk *b = &s->blk[s->nblk++]; memset(b, 0, sizeof *b);
    static VkDeviceSize blk = 0; if (!blk) { const char *e = getenv("DM_VK_SCRATCH_MB"); blk = e && atoi(e) > 0 ? (VkDeviceSize)atoi(e) << 20 : SCRATCH_BLOCK; }
    b->size = n > blk ? (n + 0xffff) & ~0xffffull : blk; __atomic_fetch_add(&g_scratch_bytes, b->size, __ATOMIC_RELAXED);
    VkBufferCreateInfo bi = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, NULL, 0, b->size, POOL_USAGE, VK_SHARING_MODE_EXCLUSIVE};
    VKCHK(vkCreateBuffer(V.dev, &bi, NULL, &b->b));
    VkMemoryRequirements req; vkGetBufferMemoryRequirements(V.dev, b->b, &req);
    VkMemoryAllocateInfo ai = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, NULL, req.size, vk_mem_type(req.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, 0)};
    VKCHK(vkAllocateMemory(V.dev, &ai, NULL, &b->m)); VKCHK(vkBindBufferMemory(V.dev, b->b, b->m, 0));
    void *m; VKCHK(vkMapMemory(V.dev, b->m, 0, VK_WHOLE_SIZE, 0, &m)); b->p = m;
    b->used = n; *buf = b->b; *off = 0; return b->p;
}
static void df_add(int kind, void *a, void *b) {
    if (g_cur < 0) {   // outside a submission (a pool finalized between frames): after everything submitted so far
        wait_fence(g_fence);
        if (kind == 0) { vkDestroyBuffer(V.dev, (VkBuffer)a, NULL); if (b) vkFreeMemory(V.dev, (VkDeviceMemory)b, NULL); }
        else if (kind == 1) { vkDestroyImage(V.dev, (VkImage)a, NULL); if (b) vkFreeMemory(V.dev, (VkDeviceMemory)b, NULL); }
        else if (kind == 4) { void img_sub_free_(void *); vkDestroyImage(V.dev, (VkImage)a, NULL); img_sub_free_(b); }
        else if (kind == 2) vkDestroyImageView(V.dev, (VkImageView)a, NULL);
        else vkDestroyBufferView(V.dev, (VkBufferView)a, NULL);
        return;
    }
    vksub_t *s = cur(); if (s->ndf == s->capdf) { s->capdf = s->capdf ? 2 * s->capdf : 64; s->df = realloc(s->df, s->capdf * sizeof *s->df); }
    s->df[s->ndf++] = (dfree_t){kind, a, b};
}
// Image memory. Small images share blocks: each allocation the driver makes is a mapping of the process (Adreno maps them all), and a game with thousands of textures
// ran towards the kernel's per-process mapping limit (Xenoblade Chronicles 3: about 9,400 driver mappings in its first area, 2,216 of them 172 KB). Images up to
// IMG_SLOT_MAX take a slot of a block (64 slots, 2 MB to IMG_BLOCK bytes), in size classes 2^(k/2) x 4 KB apart (a slot is at most 41% larger than the image) whose size is a
// multiple of the image's alignment (slots lie at multiples of it). Larger images take a range of a large block (DM_VK_IMG_BLOCK_MB, default 64; first fit), as many in one allocation as
// fit: the stock Adreno 840 driver failed image allocations (-2) at about 2,300 images with memory of their own each, beside 4.1 GB of pools (measured with another port). Images past half
// a large block get memory of their own (a few render targets)
enum { IMG_CLASSES = 18, IMG_BLOCK = 32 << 20 };
#define IMG_SLOT_MAX ((VkDeviceSize)2 << 20)
typedef struct imgblk { VkDeviceMemory m; uint32_t type, cls, nslot, nfree; uint64_t bits[128]; struct imgblk *next; } imgblk;   // bits: 1 = slot free
typedef struct lblk { VkDeviceMemory m; uint32_t type, live, nfr, capfr; VkDeviceSize size; struct { VkDeviceSize off, len; } *fr; struct lblk *next; } lblk;   // fr: free ranges by offset
typedef struct { imgblk *b; uint32_t slot; lblk *lb; VkDeviceSize off, len; VkDeviceMemory own; } imgsub;   // a slot, a large block's range, or memory of its own
static imgblk *g_imgblk[IMG_CLASSES]; static lblk *g_lblk; static pthread_mutex_t g_imgblk_mu = PTHREAD_MUTEX_INITIALIZER;
static uint64_t g_imgblk_bytes, g_lblk_bytes, g_img_own_bytes, g_img_own;   // (vk_mem_log)
static VkDeviceSize img_class_size(int c) { return (c & 1 ? 6144ull : 4096ull) << (c >> 1); }   // 4K, 6K, 8K, 12K, 16K, 24K, ... 2M
// ---- large image blocks: first fit (tests: tests/test_vk_switch2.py) ----
static int lblk_take(lblk *b, VkDeviceSize size, VkDeviceSize al, VkDeviceSize *off) {   // a range of size at a multiple of al: 1 and its offset, 0 when none fits
    if (!al) al = 1;
    for (uint32_t i = 0; i < b->nfr; i++) {
        VkDeviceSize o = (b->fr[i].off + al - 1) / al * al, e = b->fr[i].off + b->fr[i].len; if (o + size > e) continue;
        VkDeviceSize lead = o - b->fr[i].off, tail = e - (o + size);
        if (lead && tail) {   // the range splits in two
            if (b->nfr == b->capfr) { b->capfr = b->capfr ? 2 * b->capfr : 8; void *n = realloc(b->fr, b->capfr * sizeof *b->fr); if (!n) return 0; b->fr = n; }
            memmove(&b->fr[i + 2], &b->fr[i + 1], (b->nfr - i - 1) * sizeof *b->fr); b->nfr++; b->fr[i].len = lead; b->fr[i + 1].off = o + size; b->fr[i + 1].len = tail; }
        else if (lead) b->fr[i].len = lead;
        else if (tail) { b->fr[i].off = o + size; b->fr[i].len = tail; }
        else { memmove(&b->fr[i], &b->fr[i + 1], (b->nfr - i - 1) * sizeof *b->fr); b->nfr--; }
        *off = o; b->live++; return 1;
    }
    return 0;
}
static void lblk_give(lblk *b, VkDeviceSize off, VkDeviceSize len) {   // the range free again, joined with its free neighbours
    uint32_t i = 0; while (i < b->nfr && b->fr[i].off < off) i++;
    int prev = i > 0 && b->fr[i - 1].off + b->fr[i - 1].len == off, next = i < b->nfr && off + len == b->fr[i].off;
    if (prev && next) { b->fr[i - 1].len += len + b->fr[i].len; memmove(&b->fr[i], &b->fr[i + 1], (b->nfr - i - 1) * sizeof *b->fr); b->nfr--; }
    else if (prev) b->fr[i - 1].len += len;
    else if (next) { b->fr[i].off = off; b->fr[i].len += len; }
    else { if (b->nfr == b->capfr) { b->capfr = b->capfr ? 2 * b->capfr : 8; void *n = realloc(b->fr, b->capfr * sizeof *b->fr); if (!n) { b->live--; return; } b->fr = n; }   // (no room: the range is lost)
        memmove(&b->fr[i + 1], &b->fr[i], (b->nfr - i) * sizeof *b->fr); b->nfr++; b->fr[i].off = off; b->fr[i].len = len; }
    b->live--;
}
// ---- end of large image blocks ----
static VkDeviceSize img_lblock(void) { static VkDeviceSize v; if (!v) { const char *e = getenv("DM_VK_IMG_BLOCK_MB"); v = (VkDeviceSize)(e && atoi(e) > 0 ? atoi(e) : 64) << 20; } return v; }
VkDeviceMemory vk_image_memory(const VkMemoryRequirements *req, uint32_t type, VkDeviceSize *off, void **sub) {
    *off = 0; *sub = NULL;
    VkDeviceSize al = req->alignment ? req->alignment : 1;
    int c = 0; while (c < IMG_CLASSES && (img_class_size(c) < req->size || img_class_size(c) % al)) c++;
    if (c == IMG_CLASSES || req->size > IMG_SLOT_MAX) {
        if (req->size > img_lblock() / 2) {   // memory of its own
            VkMemoryAllocateInfo ai = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, NULL, req->size, type}; VkDeviceMemory m;
            if (vkAllocateMemory(V.dev, &ai, NULL, &m)) return VK_NULL_HANDLE;
            pthread_mutex_lock(&g_imgblk_mu); g_img_own_bytes += req->size; g_img_own++; pthread_mutex_unlock(&g_imgblk_mu);
            imgsub *s = calloc(1, sizeof *s); s->len = req->size; s->own = m; *sub = s;   // (its own memory, freed with the image)
            return m;
        }
        pthread_mutex_lock(&g_imgblk_mu);
        lblk *b = NULL; VkDeviceSize o = 0;
        { lblk *best = NULL; VkDeviceSize bfree = 0;   // the block with the least free memory that has a range the image fits (the emptier ones drain and go)
          for (lblk *x = g_lblk; x; x = x->next) { if (x->type != type) continue; VkDeviceSize fr = 0; int fits = 0;
              for (uint32_t i = 0; i < x->nfr; i++) { fr += x->fr[i].len; VkDeviceSize oo = (x->fr[i].off + al - 1) / al * al; if (oo + req->size <= x->fr[i].off + x->fr[i].len) fits = 1; }
              if (fits && (!best || fr < bfree)) { best = x; bfree = fr; } }
          if (best && lblk_take(best, req->size, al, &o)) b = best; }
        if (!b) {
            if (!(b = calloc(1, sizeof *b))) { pthread_mutex_unlock(&g_imgblk_mu); return VK_NULL_HANDLE; }
            b->type = type; b->size = img_lblock(); VkMemoryAllocateInfo ai = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, NULL, b->size, type};
            if (vkAllocateMemory(V.dev, &ai, NULL, &b->m)) { pthread_mutex_unlock(&g_imgblk_mu); free(b); return VK_NULL_HANDLE; }
            b->capfr = 8; b->fr = calloc(b->capfr, sizeof *b->fr); b->fr[0].off = 0; b->fr[0].len = b->size; b->nfr = 1;
            b->next = g_lblk; g_lblk = b; g_lblk_bytes += b->size;
            if (!lblk_take(b, req->size, al, &o)) { pthread_mutex_unlock(&g_imgblk_mu); return VK_NULL_HANDLE; }
        }
        pthread_mutex_unlock(&g_imgblk_mu);
        imgsub *s = calloc(1, sizeof *s); s->lb = b; s->off = o; s->len = req->size; *sub = s; *off = o;
        return b->m;
    }
    pthread_mutex_lock(&g_imgblk_mu);
    // the fullest block with a free slot: the others drain and go (img_sub_free). First free slot in the newest block (as before) left every block partly full once textures came and
    // went (Mario Kart World's Grand Prix with idle textures evicted, run 16: 2,015 MB of images in 3,080 MB of blocks)
    imgblk *b = NULL; for (imgblk *x = g_imgblk[c]; x; x = x->next) if (x->type == type && x->nfree && (!b || x->nfree < b->nfree)) b = x;
    if (!b) {
        b = calloc(1, sizeof *b); b->type = type; b->cls = (uint32_t)c; VkDeviceSize bs = img_class_size(c) * 64; if (bs < (2 << 20)) bs = 2 << 20; if (bs > IMG_BLOCK) bs = IMG_BLOCK; b->nslot = (uint32_t)(bs / img_class_size(c));   // 64 slots, 2 MB to 32 MB
        VkMemoryAllocateInfo ai = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, NULL, (VkDeviceSize)b->nslot * img_class_size(c), type};
        if (vkAllocateMemory(V.dev, &ai, NULL, &b->m)) { pthread_mutex_unlock(&g_imgblk_mu); free(b); return VK_NULL_HANDLE; }
        for (uint32_t i = 0; i < b->nslot; i++) b->bits[i >> 6] |= 1ull << (i & 63);
        b->nfree = b->nslot; b->next = g_imgblk[c]; g_imgblk[c] = b; g_imgblk_bytes += ai.allocationSize;
    }
    uint32_t i = 0; while (!b->bits[i >> 6]) i += 64; i += (uint32_t)__builtin_ctzll(b->bits[i >> 6]);
    b->bits[i >> 6] &= ~(1ull << (i & 63)); b->nfree--;
    pthread_mutex_unlock(&g_imgblk_mu);
    imgsub *s = calloc(1, sizeof *s); s->b = b; s->slot = i; *sub = s; *off = (VkDeviceSize)i * img_class_size(c);
    return b->m;
}
uint64_t vk_image_sub_size(const void *sub, int *own) {   // the memory vk_image_memory gave an image (its slot, range or own allocation), and whether it is its own
    const imgsub *s = sub; if (own) *own = s && s->own; return !s ? 0 : s->b ? img_class_size((int)s->b->cls) : s->len;
}
// The slot or range is free again, or the image's own memory freed. A block left empty is freed too when another of its kind could take the next image (a small block: another of its
// class and memory type a quarter free; a large one: another of its type with a free range of half a block, which any image of a large block fits): blocks were all kept, and a game
// that changes courses or areas leaves its old images' blocks behind (run 5 of Mario Kart World: 1,004 MB of blocks under 1,561 MB of images with 838 MB of their own)
static uint64_t g_imgblk_freed;   // (vk_mem_log)
static int g_img_press;   // memory short (gfx_vk_budget.c vk_mem_pressure, read by vk_tex_sweep's caller every 300 presents: vk_img_pressure): empty blocks go without waiting for another's room
void vk_img_pressure(int level) { __atomic_store_n(&g_img_press, level > 0, __ATOMIC_RELAXED); }
static void img_sub_free(imgsub *s) {
    VkDeviceMemory gone = VK_NULL_HANDLE;
    pthread_mutex_lock(&g_imgblk_mu);
    if (s->b) { imgblk *b = s->b; b->bits[s->slot >> 6] |= 1ull << (s->slot & 63); b->nfree++;
        if (b->nfree == b->nslot) { imgblk **pp = &g_imgblk[b->cls], *o = NULL; for (imgblk *x = *pp; x && !o; x = x->next) if (x != b && x->type == b->type && (x->nfree * 4 >= x->nslot || g_img_press)) o = x;
            while (o && *pp && *pp != b) pp = &(*pp)->next;
            if (o && *pp == b) { *pp = b->next; gone = b->m; VkDeviceSize n = (VkDeviceSize)b->nslot * img_class_size((int)b->cls); g_imgblk_bytes -= n; g_imgblk_freed += n; free(b); } } }
    else if (s->lb) { lblk *b = s->lb; lblk_give(b, s->off, s->len);
        if (!b->live) { lblk **pp = &g_lblk, *o = NULL;
            for (lblk *x = g_lblk; x && !o; x = x->next) if (x != b && x->type == b->type) for (uint32_t i = 0; i < x->nfr; i++) if (x->fr[i].len >= img_lblock() / 2 || g_img_press) { o = x; break; }
            if (!o && g_img_press) o = b;   // (memory short: an empty large block goes even when no other has room, the next image makes a block again)
            while (o && *pp && *pp != b) pp = &(*pp)->next;
            if (o && *pp == b) { *pp = b->next; gone = b->m; g_lblk_bytes -= b->size; g_imgblk_freed += b->size; free(b->fr); free(b); } } }
    else { g_img_own_bytes -= s->len; g_img_own--; }
    pthread_mutex_unlock(&g_imgblk_mu);
    if (gone) vkFreeMemory(V.dev, gone, NULL);   // (no image is bound to it: each was destroyed before its slot or range came back)
    if (s->own) vkFreeMemory(V.dev, s->own, NULL);
    free(s);
}
// DM_VK_MEMLOG=<seconds> (start-up): a line of the memory the replay holds, at most that often (the first device run of a game shows its budget): pools reserved and placed, images (dense
// images of sparse textures apart), the blocks they lie in, sparse tiles bound and the sparse address space, scratch, and the device's MemAvailable (the low-memory killer ends a game near a
// few hundred MB of it). Logged at a submission's end, never inside a pool's swap
// The GPU memory the kgsl driver counts for this process (/sys/class/kgsl/kgsl/proc/<pid>/: its kernel and user allocations, the buffers it imported, what of them the CPU maps or not) and for
// every process (page_alloc), as far as the system lets an app read them; the files are tried once, those readable read at each line
static void kgsl_mem(char *out, size_t n) {
    static const char *const nm[6] = {"kernel", "user", "imported_mem", "gpumem_mapped", "gpumem_unmapped", "page_alloc"}; static int ok = -1;   // ok: the files readable (bits), -1 not tried yet
    int o = 0, got = 0, seen = 0; out[0] = 0;
    for (int i = 0; i < 6; i++) {
        if (ok >= 0 && !(ok >> i & 1)) continue;
        char p[96]; if (i < 5) snprintf(p, sizeof p, "/sys/class/kgsl/kgsl/proc/%d/%s", (int)getpid(), nm[i]); else snprintf(p, sizeof p, "/sys/class/kgsl/kgsl/%s", nm[i]);
        FILE *f = fopen(p, "r"); long long v = -1; if (f) { if (fscanf(f, "%lld", &v) != 1) v = -1; fclose(f); }
        if (v < 0) continue;
        seen |= 1 << i; if (o < (int)n) o += snprintf(out + o, n - (size_t)o, "%s%s %.0f MB", got++ ? ", " : "kgsl ", i < 5 ? nm[i] : "all processes", v / 1048576.0);
    }
    if (ok < 0) ok = seen;
    if (!got) snprintf(out, n, "kgsl not readable");
}
void vk_mem_log(void) {
    static int every = -1; static uint64_t last; if (every < 0) { const char *e = getenv("DM_VK_MEMLOG"); every = e ? atoi(e) : 0; }
    if (every <= 0) return;
    uint64_t now = now_mono(); if (last && now - last < (uint64_t)every * 1000000000ull) return; last = now;
    void vk_tex_mem(uint64_t o[8]); uint64_t t[8] = {0}; vk_tex_mem(t);   // images, their bytes, dense images, their bytes, sparse tiles bound, tile blocks, sparse address space, sparse images
    long avail = -1, sfree = -1; FILE *f = fopen("/proc/meminfo", "r"); char line[128];
    if (f) { while (fgets(line, sizeof line, f)) { if (!strncmp(line, "MemAvailable:", 13)) avail = atol(line + 13) / 1024; else if (!strncmp(line, "SwapFree:", 9)) sfree = atol(line + 9) / 1024; } fclose(f); }
    // the process as the kernel counts it (resident, of it anonymous, and swapped out) and the GPU memory the driver counts: the rest of what the low-memory killer weighs (run 11 of Mario Kart
    // World was killed with 6.06 GB of swap, "min watermark is breached and swap is low")
    long rss = -1, anon = -1, swp = -1; FILE *ps = fopen("/proc/self/status", "r");
    if (ps) { while (fgets(line, sizeof line, ps)) { if (!strncmp(line, "VmRSS:", 6)) rss = atol(line + 6) / 1024; else if (!strncmp(line, "RssAnon:", 8)) anon = atol(line + 8) / 1024;
                                                     else if (!strncmp(line, "VmSwap:", 7)) swp = atol(line + 7) / 1024; } fclose(ps); }
    char kg[200]; kgsl_mem(kg, sizeof kg);
    pthread_mutex_lock(&g_place_mu); uint64_t res = g_reserved_bytes, zc = g_placed_bytes, cp = g_copied_bytes, np = g_places, pns = g_place_ns; pthread_mutex_unlock(&g_place_mu);
    pthread_mutex_lock(&g_imgblk_mu); uint64_t sb = g_imgblk_bytes, lb = g_lblk_bytes, ob = g_img_own_bytes, on = g_img_own, bf = g_imgblk_freed; pthread_mutex_unlock(&g_imgblk_mu);
    VKLOG("memory: pools %.0f MB reserved, %.0f MB placed zero-copy + %.0f MB copied (%llu placements, %.0f ms, %llu writes waited); images %llu, %.0f MB (%llu dense from a base, %.0f MB) in %.0f MB "
          "of small blocks, %.0f MB of large ones (%.0f MB of empty ones freed so far), %.0f MB of their own (%llu); sparse tiles %llu bound in %llu blocks (%.0f MB), %llu sparse images, address space %.0f MB; scratch %.0f MB (%.0f MB freed so far); process VmRSS %ld MB (RssAnon %ld MB), VmSwap %ld MB; %s; MemAvailable %ld MB, SwapFree %ld MB",
          res / 1048576.0, zc / 1048576.0, cp / 1048576.0, (unsigned long long)np, pns / 1e6, (unsigned long long)atomic_load(&g_swap_waits), (unsigned long long)t[0], t[1] / 1048576.0,
          (unsigned long long)t[2], t[3] / 1048576.0, sb / 1048576.0, lb / 1048576.0, bf / 1048576.0, ob / 1048576.0, (unsigned long long)on, (unsigned long long)t[4], (unsigned long long)t[5], (t[5] << 26) / 1048576.0,
          (unsigned long long)t[7], t[6] / 1048576.0, __atomic_load_n(&g_scratch_bytes, __ATOMIC_RELAXED) / 1048576.0, __atomic_load_n(&g_scratch_freed, __ATOMIC_RELAXED) / 1048576.0, rss, anon, swp, kg, avail, sfree);
    // what of it could go (gfx_vk_tex.c vk_tex_census): the textures' storage the shadow file could hold (DM_VK_SHADOW, gfx_vk_shadow.c), the GPU's images nothing has written, idle
    // images, the images' memory against their storage, sparse places against the physical tiles behind them, resolution families, the shaders' SPIR-V kept (gfx_vk_shader.c vk_shader_mem)
    { void vk_tex_census(uint64_t o[32]); void vk_shader_mem(uint64_t o[4]); uint64_t c[32], sm[4] = {0}; vk_tex_census(c); vk_shader_mem(sm); const double M = 1048576.0;
      VKLOG("memory census: textures uploaded from guest memory %llu, %.0f MB of storage: shadow candidates now %llu (%.0f MB), in placed pools %llu (%.0f MB), protection released %llu (%.0f MB), "
            "others %llu (%.0f MB: under the size floor %llu, %.0f MB; written since their upload %llu, %.0f MB; the rest linear, GPU-written or untracked); shaders %llu (%.0f MB of SPIR-V kept, "
            "%llu variants)", (unsigned long long)c[0], c[1] / M, (unsigned long long)c[2], c[3] / M, (unsigned long long)c[4], c[5] / M, (unsigned long long)c[6], c[7] / M, (unsigned long long)c[8],
            c[9] / M, (unsigned long long)c[27], c[28] / M, (unsigned long long)c[29], c[30] / M, (unsigned long long)sm[0], sm[1] / M, (unsigned long long)sm[2]);
      VKLOG("memory census: images not sparse %llu, %.0f MB of memory for %.0f MB of NVN storage; the GPU's never written %llu (%.0f MB; of 64 MB or more %llu, %.0f MB); not looked up for 1800 "
            "presents %llu (%.0f MB), the GPU's for 300 %llu (%.0f MB); resolution families %llu of %llu images, %.0f MB over one allocation each; sparse places bound %llu slots (%.0f MB) over %llu "
            "physical tiles (%.0f MB)", (unsigned long long)c[16], c[17] / M, c[18] / M, (unsigned long long)c[10], c[11] / M, (unsigned long long)c[14], c[15] / M, (unsigned long long)c[12],
            c[13] / M, (unsigned long long)c[25], c[26] / M, (unsigned long long)c[22], (unsigned long long)c[23], c[24] / M, (unsigned long long)c[20], c[20] * 65536.0 / M,
            (unsigned long long)c[21], c[21] * 65536.0 / M); }
    vk_shadow_log(); vk_budget_log();   // (the shadow file's line, DM_VK_SHADOW; the smaps census, DM_VK_SMAPS: a thread of its own)
}
void vk_defer_free_buffer(VkBuffer b, VkDeviceMemory m) { df_add(0, (void *)b, (void *)m); }
void vk_defer_free_image_sub(VkImage i, void *sub) { df_add(4, (void *)i, sub); }
void img_sub_free_(void *s) { img_sub_free((imgsub *)s); }
void vk_defer_free_image(VkImage i, VkDeviceMemory m) { df_add(1, (void *)i, (void *)m); }
void vk_defer_free_view(VkImageView v) { df_add(2, (void *)v, NULL); }
// Descriptor pools: the allocator counts what each set takes and moves to a new pool before any count runs out. Adreno's driver does not refuse an
// allocation past a pool's sizes: it succeeds and the submission then fails as a whole at vkEndCommandBuffer (VK_ERROR_OUT_OF_HOST_MEMORY; NEO's
// handheld frames with 32-slot texture arrays crossed the old 8192 combined-sampler limit at about the 712th draw)
enum { DP_SETS = 4096 };
static const uint32_t DP_CAP[9] = {[VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC] = 16384, [VK_DESCRIPTOR_TYPE_SAMPLER] = 1024, [VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER] = 65536, [VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE] = 1024,
    [VK_DESCRIPTOR_TYPE_STORAGE_IMAGE] = 2048, [VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER] = 4096, [VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER] = 2048,
    [VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER] = 32768, [VK_DESCRIPTOR_TYPE_STORAGE_BUFFER] = 4096};
static nvn_hmap g_dsl_need = {.klen = sizeof(VkDescriptorSetLayout)}; static pthread_mutex_t g_dsl_mu = PTHREAD_MUTEX_INITIALIZER;
void vk_dsl_register(VkDescriptorSetLayout l, const VkDescriptorSetLayoutBinding *b, uint32_t n) {   // what one set of this layout takes, by type
    uint32_t *need = calloc(9, sizeof *need);
    for (uint32_t i = 0; i < n; i++) if ((uint32_t)b[i].descriptorType < 9) need[b[i].descriptorType] += b[i].descriptorCount;
    pthread_mutex_lock(&g_dsl_mu); nvn_hm_put(&g_dsl_need, &l, need); pthread_mutex_unlock(&g_dsl_mu);
}
VkDescriptorSet vk_alloc_set(VkDescriptorSetLayout l) {
    vksub_t *s = cur();
    pthread_mutex_lock(&g_dsl_mu); const uint32_t *need = nvn_hm_get(&g_dsl_need, &l); pthread_mutex_unlock(&g_dsl_mu);
    for (;;) {
        if (s->dp_cur == s->ndp) {
            VkDescriptorPoolSize sz[9]; uint32_t nz = 0;
            for (int t = 0; t < 9; t++) if (DP_CAP[t]) sz[nz++] = (VkDescriptorPoolSize){(VkDescriptorType)t, DP_CAP[t]};
            VkDescriptorPoolCreateInfo pi = {VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO, NULL, 0, DP_SETS, nz, sz};
            if (s->ndp == s->capdp) { s->capdp = s->capdp ? 2 * s->capdp : 4; s->dp = realloc(s->dp, s->capdp * sizeof *s->dp); }
            VKCHK(vkCreateDescriptorPool(V.dev, &pi, NULL, &s->dp[s->ndp++]));
        }
        if (s->dp_cur != s->dp_counted) { memset(s->dp_used, 0, sizeof s->dp_used); s->dp_sets = 0; s->dp_counted = s->dp_cur; }
        int fits = s->dp_sets < DP_SETS;
        if (need) for (int t = 0; t < 9 && fits; t++) if (need[t] && s->dp_used[t] + need[t] > DP_CAP[t]) fits = 0;
        if (!fits) { s->dp_cur++; continue; }
        VkDescriptorSetAllocateInfo ai = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, NULL, s->dp[s->dp_cur], 1, &l};
        VkDescriptorSet set; VkResult r = vkAllocateDescriptorSets(V.dev, &ai, &set);
        if (r == VK_SUCCESS) { s->dp_sets++; if (need) for (int t = 0; t < 9; t++) s->dp_used[t] += need[t]; return set; }
        if (r != VK_ERROR_OUT_OF_POOL_MEMORY && r != VK_ERROR_FRAGMENTED_POOL) die("[vk] vkAllocateDescriptorSets: %d", r);
        s->dp_cur++;
    }
}
// Kept descriptor sets (gfx_vk_exec.c stage_set: sets kept by their content across submissions): pools of their own, which no submission resets. vk_keep_flush retires the pools in
// use at the work recorded so far (the open submission's included), and a retired pool is reset and used again once the GPU has passed it. NULL when every pool is full or still
// retired: the caller takes a submission's set (vk_alloc_set) meanwhile, and flushes
enum { KEEP_POOLS = 8 };
static struct { VkDescriptorPool p; uint32_t used[9], sets; uint64_t fence; int state; } g_kp[KEEP_POOLS];   // state 0 free, 1 in use, 2 retired at fence
static int g_kp_cur = -1;
VkDescriptorSet vk_alloc_set_kept(VkDescriptorSetLayout l) {
    pthread_mutex_lock(&g_dsl_mu); const uint32_t *need = nvn_hm_get(&g_dsl_need, &l); pthread_mutex_unlock(&g_dsl_mu);
    for (int tries = 0; tries <= KEEP_POOLS; tries++) {
        if (g_kp_cur < 0) {   // a free pool, or a retired one the GPU has passed (reset), or a new one
            uint64_t done = vk_fence_done();
            for (int i = 0; i < KEEP_POOLS && g_kp_cur < 0; i++) if (g_kp[i].p && (g_kp[i].state == 0 || (g_kp[i].state == 2 && g_kp[i].fence <= done))) {
                if (g_kp[i].state == 2) vkResetDescriptorPool(V.dev, g_kp[i].p, 0);
                g_kp[i].state = 1; memset(g_kp[i].used, 0, sizeof g_kp[i].used); g_kp[i].sets = 0; g_kp_cur = i; }
            for (int i = 0; i < KEEP_POOLS && g_kp_cur < 0; i++) if (!g_kp[i].p) {
                VkDescriptorPoolSize sz[9]; uint32_t nz = 0; for (int t = 0; t < 9; t++) if (DP_CAP[t]) sz[nz++] = (VkDescriptorPoolSize){(VkDescriptorType)t, DP_CAP[t]};
                VkDescriptorPoolCreateInfo pi = {VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO, NULL, 0, DP_SETS, nz, sz};
                if (vkCreateDescriptorPool(V.dev, &pi, NULL, &g_kp[i].p)) { g_kp[i].p = VK_NULL_HANDLE; break; }
                g_kp[i].state = 1; memset(g_kp[i].used, 0, sizeof g_kp[i].used); g_kp[i].sets = 0; g_kp_cur = i; }
            if (g_kp_cur < 0) return VK_NULL_HANDLE;
        }
        typeof(g_kp[0]) *k = &g_kp[g_kp_cur];
        int fits = k->sets < DP_SETS; if (need) for (int t = 0; t < 9 && fits; t++) if (need[t] && k->used[t] + need[t] > DP_CAP[t]) fits = 0;   // (counted: Adreno does not refuse past a pool's sizes, vk_alloc_set)
        if (fits) { VkDescriptorSetAllocateInfo ai = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, NULL, k->p, 1, &l}; VkDescriptorSet set;
            if (vkAllocateDescriptorSets(V.dev, &ai, &set) == VK_SUCCESS) { k->sets++; if (need) for (int t = 0; t < 9; t++) k->used[t] += need[t]; return set; } }
        k->state = 3; g_kp_cur = -1;   // full: kept in use until the next flush (state 3: full and in use)
    }
    return VK_NULL_HANDLE;
}
int vk_keep_live(void) { int n = 0; for (int i = 0; i < KEEP_POOLS; i++) n += g_kp[i].state == 1 || g_kp[i].state == 3; return n; }   // pools of the current keep generation (gfx_vk_exec.c kept_new)
void vk_keep_flush(void) {   // every kept set is dropped (the caller forgets them): their pools retire at the work recorded so far
    uint64_t f = g_fence + (vk_sub_open() ? 1 : 0);
    for (int i = 0; i < KEEP_POOLS; i++) if (g_kp[i].state == 1 || g_kp[i].state == 3) { g_kp[i].state = 2; g_kp[i].fence = f; }
    g_kp_cur = -1;
}

// ---------------- submissions: begin, CPU-visible effects, commit ----------------
static void post(post_op o) { vksub_t *s = cur(); if (s->npost == s->cappost) { s->cappost = s->cappost ? 2 * s->cappost : 64; s->post = realloc(s->post, s->cappost * sizeof *s->post); } s->post[s->npost++] = o; }
// DM_VK_X_RECCAP (gfx_vk_exec.c): one range of indirect-draw records copied off the GPU at a pass's end, queued for the completion thread to log. The scratch
// pointer is read there and only there, before this submission's g_done advances: the slot is not recycled until the completion thread has passed its fence
// (slot_recycle waits, then scratch_trim frees), so between the copy and the read no later work can overwrite the bytes
typedef struct { uint64_t present; int pass; uint64_t key, addr, cnt; uint32_t n, stride, rs; int has_count; const uint32_t *p; } vk_reccap_t;
void vk_post_reccap(uint64_t present, int pass, uint64_t key, uint64_t addr, uint64_t cnt, uint32_t n, uint32_t stride, uint32_t rs, int has_count, const uint32_t *p) {
    vk_reccap_t *d = malloc(sizeof *d); if (!d) return;
    *d = (vk_reccap_t){present, pass, key, addr, cnt, n, stride, rs, has_count, p};
    post((post_op){3, NULL, 0, 0, 0, 0, -1, d});
}
void mtl_init(void) { vk_init(); }
// The first submission's prepare step, outside the graphics gate (hle_nvn.c nvnQueueSubmitCommands; nvn_mtl.m mtl_prepare: the prewarm and the pipeline prebuild): the device is
// made here, which takes the driver a while, so a suspend meanwhile does not wait in the gate for it. The pipeline prewarm itself already runs on background threads (gfx_vk_exec.c
// prewarm_start, at the first submission)
void mtl_prepare(void) { vk_init(); }
// The game's SyncWait on a sync WindowAcquireTexture handed out: Metal's backend gives it its own line in a wait ledger (DM_NVN_WAITS), which this one has not; the same wait
int mtl_wait_acq(uint64_t fence, uint64_t timeout_ns) { return mtl_wait(fence, timeout_ns); }
// The Metal device for the GL and Vulkan layers of the Mac host (dm/gfx/vk/hle_vk.m, dm/gfx/gl) and the presents of their textures: no such layers run here (a game that renders with
// Vulkan or OpenGL is refused by the port, docs/ARCHITECTURE.md), so there is no device to give and nothing to present
void *mtl_device(void) { return NULL; }
void *mtl_device_classic(void) { return NULL; }
uint64_t mtl_ext_tex(void *tex) { (void)tex; return 0; }
uint64_t mtl_present_ext(uint64_t serial, double period, void *ev, uint64_t ev_val, const int crop[4]) { (void)serial; (void)period; (void)ev; (void)ev_val; (void)crop; VKRED_ONCE(120, "mtl_present_ext: an external layer's present, which the Android host has none of"); return 0; }
static int g_shadow_pools;
static uint64_t g_sub_serial;   // submissions begun (vk_sub_serial)
uint64_t vk_sub_serial(void) { return __atomic_load_n(&g_sub_serial, __ATOMIC_ACQUIRE); }
void mtl_sub_begin(void) {
    vk_init();
    pthread_mutex_lock(&g_sub_lock); __atomic_add_fetch(&g_sub_serial, 1, __ATOMIC_RELEASE);
    // the next slot of the ring (g_cur is -1 between submissions, so it cannot say where the ring is: every submission took slot 0 and waited in slot_recycle
    // for the one before to finish on the GPU, which kept the replay of a frame from overlapping the GPU work of the last)
    static int next; g_cur = next; next = (next + 1) % NSLOT; vksub_t *s = &g_slot[g_cur];
    slot_recycle(s);
    s->npost = 0; s->up_used = 0; s->ngw = 0;
    VkCommandBufferBeginInfo bi = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, NULL, VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT};
    s->main = s->mains[0]; s->nmain_used = 1; s->cb_err = 0; s->up = s->ups[0]; s->nup_used = 1; s->nmain_sent = 0;
    VKCHK(vkBeginCommandBuffer(s->main, &bi)); g_cmd_gen++;
    // copied pools: what the game wrote before submitting goes into the GPU copy before any part of the submission can reach the GPU (DM_VK_CB_DRAWS sends
    // parts early; synced at the commit, those parts read the copy as the last submission left it), and only once no earlier submission still reads it
    pthread_rwlock_rdlock(&g_pool_mu); int shadow = 0; for (int i = 0; i < g_npool; i++) shadow |= g_ord[i]->shadow != NULL; pthread_rwlock_unlock(&g_pool_mu);
    g_shadow_pools = shadow;
    if (shadow) { mtl_wait(g_fence, ~0ull); vk_pools_sync(NULL); }   // (after the completion thread's copy-back of the last submission, not only its GPU work)
    if (g_tsq) { vkCmdResetQueryPool(s->main, g_tsq, 2 * (uint32_t)g_cur, 2); vkCmdWriteTimestamp(s->main, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, g_tsq, 2 * (uint32_t)g_cur); }
    if (g_cq) vkCmdResetQueryPool(s->main, g_cq, (uint32_t)g_cur * CQ, CQ);
    s->ncq = 0;
    prof_sub_begin(s);   // (DM_VK_GPUPROF: timed or not, and its first boundary)
    vk_exec_reset();
}
void mtl_sub_event(volatile uint32_t *val, uint32_t mode, uint32_t v) { post((post_op){1, val, mode, v, 0, 0, -1}); }
// Counter census (Mario Kart World's trees flicker: does it cull by SAMPLES_PASSED reports, and does it clear them before our completion write?); replay thread only
uint64_t g_ctr_n[18], g_ctr_reset[18], g_ctr_zero, g_ctr_seen; int g_ctr_early, g_x_subidle, g_ts_cal, g_ts_flat;   // g_ts_cal: DM_VK_TS_CAL, g_ts_flat: DM_VK_TS_FLAT (completion_main)   // DM_VK_CTR_EARLY, DM_VK_X_SUBIDLE (live, gfx_vk_exec.c x_read)
static uint64_t gpu_ticks(void);
void mtl_sub_counter(uint64_t addr, uint32_t type) {
    vksub_t *s = cur(); int q = -1; uint32_t ti = type < 17 ? type : 17; uint64_t tot = 0;
    if (++g_ctr_n[ti] == 1) VKLOG("counter: type %#x first reported (%llu resets of it so far)", type, (unsigned long long)g_ctr_reset[ti]);
    for (int i = 0; i < 18; i++) tot += g_ctr_n[i];
    if (tot % 4096 == 0) VKLOG("counter: %llu reports (timestamp %llu, samples passed %llu, top %llu, other %llu), %llu samples resets; samples records found zero at completion %llu of %llu; early write %d",
                               (unsigned long long)tot, (unsigned long long)g_ctr_n[0], (unsigned long long)g_ctr_n[1], (unsigned long long)g_ctr_n[16], (unsigned long long)(tot - g_ctr_n[0] - g_ctr_n[1] - g_ctr_n[16]),
                               (unsigned long long)g_ctr_reset[1], (unsigned long long)g_ctr_zero, (unsigned long long)g_ctr_seen, g_ctr_early);
    // DM_VK_CTR_EARLY=1: a SAMPLES_PASSED report is also written now, at replay, not only once the GPU has finished: a game that reads it sooner sees "visible", not 0
    if (g_ctr_early && type == NVN_COUNTER_TYPE_SAMPLES_PASSED) { uint64_t rec[2] = {1u << 20, gpu_ticks()}; void *at = vk_cpu_ptr(addr, sizeof rec); if (at) memcpy(at, rec, sizeof rec); }
    if (g_cq && (type == 0 || type == 0x10) && s->ncq < CQ) { q = g_cur * CQ + s->ncq++; vkCmdWriteTimestamp(vk_cmd(), VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, g_cq, (uint32_t)q); }
    post((post_op){2, NULL, 0, 0, addr, type, q});
}
static VkMemoryBarrier2 full_barrier(void) {
    return (VkMemoryBarrier2){VK_STRUCTURE_TYPE_MEMORY_BARRIER_2, NULL, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                              VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT};
}
void vk_img_barrier_all(VkCommandBuffer cb) {
    int vk_bar_redundant(VkCommandBuffer cb); void vk_bar_recorded(VkCommandBuffer cb);
    if (vk_bar_redundant(cb)) { g_bar_merged++; return; }   // right after another, only state between: it adds nothing (DM_VK_BAR_MERGE, gfx_vk_dq.c)
    VkMemoryBarrier b = {VK_STRUCTURE_TYPE_MEMORY_BARRIER, NULL, VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT, VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT};
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 1, &b, 0, NULL, 0, NULL);
    vk_bar_recorded(cb); g_bar_full++;
}
void vk_gpu_wrote(uint64_t a, uint64_t n);
void vk_wrote_by(const char *fmt, ...);   // (defined by vk_gpu_wrote: names the work whose writes follow)
void mtl_sub_copy(uint64_t src, uint64_t dst, uint64_t n) {
    VkBuffer sb, db; VkDeviceSize so, dof;
    if (!n) return;
    if (!vk_buf_at(src, n, &sb, &so) || !vk_buf_at(dst, n, &db, &dof)) { VKRED_ONCE(1, "buffer copy outside the memory pools (%#llx → %#llx, %llu bytes)", (unsigned long long)src, (unsigned long long)dst, (unsigned long long)n); return; }
    vk_end_pass_land();
    VkCommandBuffer cb = vk_cmd(); vk_img_barrier_all(cb);
    VkBufferCopy r = {so, dof, n}; vkCmdCopyBuffer(cb, sb, db, 1, &r);
    vk_img_barrier_all(cb); if (g_x_writers_on) vk_wrote_by("buffer copy from %#llx", (unsigned long long)src); vk_gpu_wrote(dst, n);   // (DM_VK_X_WRITERS: the copy's kind)
}
void mtl_sub_fill(uint64_t dst, uint64_t n, uint32_t value) {
    VkBuffer db; VkDeviceSize dof;
    if (!n) return;
    if (!vk_buf_at(dst, n, &db, &dof)) { VKRED_ONCE(2, "buffer fill outside the memory pools"); return; }
    vk_end_pass_land();
    VkCommandBuffer cb = vk_cmd(); vk_img_barrier_all(cb);
    vkCmdFillBuffer(cb, db, dof, n & ~3ull, value);
    vk_img_barrier_all(cb); if (g_x_writers_on) vk_wrote_by("buffer fill %#x", value); vk_gpu_wrote(dst, n);   // (DM_VK_X_WRITERS: the fill's kind)
}
// an inline uniform update (UpdateUniformBuffer): the bytes are taken at record time, copied in GPU order from scratch memory
// Inline uniform updates (UpdateUniformBuffer) are deferred instead of ending the render pass (as nvn_mtl.m's ubd_*): a copy cannot run inside a pass, and
// ending it stores every attachment and loads it again in the next pass, which on a tiler is a round trip of the whole tile memory per update (Unity updates
// before nearly every draw: NEO had about 145 passes for 148 draws a frame). NVN applies the update in command order, so a draw after it whose uniform
// binding overlaps it reads a shadow: memory with the deferred updates applied in order (vk_ubd_shadow). The real copies are made in order when a pass
// ends anyway, and at the end of the submission (vk_ubd_land), so later passes, copies, dispatches and the CPU see them as before.
typedef struct { uint64_t lo, hi; VkBuffer sb; VkDeviceSize so; const uint8_t *p; } ubd_t;
enum { UBD_MAX = 512 };
static ubd_t g_ubd[UBD_MAX]; static int g_nubd; uint64_t g_ubd_gen = 1, g_ubd_landed, g_ubd_shadows, g_ubd_bytes, g_ns_submit, g_ubd_superseded;
int vk_ubd_room(void) { return UBD_MAX - g_nubd; }   // updates the list takes before the next one lands it (vk_sub_update; gfx_vk_exec.c dsc_scan: a pass's end then)
void vk_end_pass_land(void);
void vk_sub_update(uint64_t dst, const void *data, uint64_t n) {
    VkBuffer db, sb; VkDeviceSize dof, so;
    if (!n) return;
    if (!vk_buf_at(dst, n, &db, &dof)) { VKRED_ONCE(3, "uniform update outside the memory pools"); return; }
    // an update of exactly the range of a deferred one supersedes it: the older one goes (Unity rewrites the same few constant buffers before every draw,
    // so the list keeps about one entry per buffer region, which keeps the landing to a few copies and the shadows cheap to build)
    for (int i = g_nubd - 1; i >= 0; i--) if (g_ubd[i].lo == dst && g_ubd[i].hi == dst + n) { memmove(&g_ubd[i], &g_ubd[i + 1], sizeof *g_ubd * (size_t)(g_nubd - i - 1)); g_nubd--; g_ubd_superseded++; break; }
    if (g_nubd == UBD_MAX) vk_end_pass_land();   // too many distinct ranges: they land now
    uint8_t *p = vk_scratch(n, 4, &sb, &so); memcpy(p, data, n);
    g_ubd[g_nubd++] = (ubd_t){dst, dst + n, sb, so, p}; g_ubd_gen++;
}
// the draw bisection (gfx_vk_exec.c draw_allowed): the outcome of this submission
extern int g_bis_on, g_bis_lo, g_bis_hi, g_bis_trace, g_bis_var; void vk_bisect_reset_count(void);
static void vk_bisect_result(int ok) {
    static int done, draws_seen; extern uint32_t g_sub_ops[64]; int nd = (int)(g_sub_ops[25] + g_sub_ops[26]);
    if (done) {   // the found draw: one change per submission, then the draw is no longer traced
        if (g_bis_trace >= 0) {
            static const char *what[] = {"as recorded", "depth bias values 0", "no depth bias", "dummy textures", "vertex offset 0", "zero vertex buffer", "in a new pass", "skipped alone (the rest drawn)", "without their fragment shader", "with zero uniform buffers", "with their quad draws skipped", "with all but their quad draws skipped"};
            VKLOG("bisection: draws from %d on %s: %s", g_bis_trace, what[g_bis_var], ok ? "ACCEPTED" : "refused");
            if (++g_bis_var > 11) { g_bis_trace = -1; g_bis_var = 0; }
        }
        vk_bisect_reset_count(); return; }
    if (!g_bis_on) { if (!ok) { g_bis_on = 1; g_bis_lo = 0; g_bis_hi = nd + 1; draws_seen = nd; VKLOG("bisecting the refused submission's %d draws", nd); } vk_bisect_reset_count(); return; }
    int mid = (g_bis_lo + g_bis_hi) / 2;
    if (ok) g_bis_lo = mid; else g_bis_hi = mid;
    if (g_bis_hi - g_bis_lo <= 1) { done = 1; g_bis_on = 0; g_bis_trace = g_bis_lo; VKLOG("bisection: draw %d of %d is the first the driver refuses (traced in the next submission)", g_bis_lo, draws_seen); }
    vk_bisect_reset_count();
}
void vk_ubd_land_drop(void) { g_nubd = 0; g_ubd_gen++; }
// Landed updates still in flight: their copies are queued but the GPU may not have run them, so memory may not hold them yet; a shadow built on the
// CPU (vk_ubd_read) applies them over memory until their submission completes. Without this a shadow mixed the newest update with the frame before's
// memory (shadows detached from moving characters, far pedestrians flickering).
typedef struct { uint64_t lo, hi, fence; const uint8_t *data; } ubd_infl_t;   // data: the update's bytes in its submission's scratch memory, which is not reused before that submission completes
static ubd_infl_t *g_infl; static int g_ninfl, g_capinfl;
static void infl_prune(void) {
    uint64_t done = vk_fence_done(); int o = 0;
    for (int i = 0; i < g_ninfl; i++) if (g_infl[i].fence > done) g_infl[o++] = g_infl[i];
    g_ninfl = o;
}
static void sub_gpu_wrote(uint64_t a, uint64_t n);
void vk_ubd_land(VkCommandBuffer cb) {   // outside a render pass
    if (!g_nubd) return;
    infl_prune();
    for (int i = 0; i < g_nubd; i++) {
        if (g_ninfl == g_capinfl) { g_capinfl = g_capinfl ? 2 * g_capinfl : 256; g_infl = realloc(g_infl, sizeof *g_infl * (size_t)g_capinfl); }
        // (the bytes used to be copied into the heap here: a malloc, memcpy and free of up to 64 KB per update, for bytes the scratch memory keeps anyway)
        g_infl[g_ninfl++] = (ubd_infl_t){g_ubd[i].lo, g_ubd[i].hi, g_fence + 1, g_ubd[i].p};   // the fence of the submission being recorded
    }
    vk_img_barrier_all(cb);
    for (int i = 0; i < g_nubd; i++) { VkBuffer db; VkDeviceSize dof; if (vk_buf_at(g_ubd[i].lo, g_ubd[i].hi - g_ubd[i].lo, &db, &dof)) {
        VkBufferCopy r = {g_ubd[i].so, dof, g_ubd[i].hi - g_ubd[i].lo}; vkCmdCopyBuffer(cb, g_ubd[i].sb, db, 1, &r); sub_gpu_wrote(g_ubd[i].lo, g_ubd[i].hi - g_ubd[i].lo); } }
    vk_img_barrier_all(cb);
    g_ubd_landed += (uint64_t)g_nubd; g_nubd = 0; g_ubd_gen++;
}
int vk_ubd_overlaps(uint64_t a, uint64_t n) { for (int i = 0; i < g_nubd; i++) if (g_ubd[i].lo < a + n && a < g_ubd[i].hi) return 1; return 0; }
// GPU writes to pool memory (buffer copies and fills, texture → buffer copies, storage buffers of dispatches) whose submission may not have completed:
// until it has, a CPU read of that memory (vk_ubd_read: a shadow, a handle in a uniform buffer) may see the bytes from before them or after them
typedef struct { uint64_t lo, hi, fence; } gpuw_t; enum { GW_MAX = 1024 };
static gpuw_t g_gw[GW_MAX]; static int g_ngw; uint64_t g_ubd_racy;
static void sub_gpu_wrote(uint64_t a, uint64_t n) {   // into the open submission's list (pools_copy_back)
    if (g_cur < 0 || !n) return;
    vksub_t *s = &g_slot[g_cur]; uint64_t lo = a, hi = a + n, bd = ~0ull; int best = -1;
    for (int k = 0; k < s->ngw; k++) { uint64_t d = lo > s->gw[k].hi ? lo - s->gw[k].hi : s->gw[k].lo > hi ? s->gw[k].lo - hi : 0; if (d < bd) { bd = d; best = k; } }
    if (best >= 0 && (bd <= 65536 || s->ngw == GW_SUB)) { if (lo < s->gw[best].lo) s->gw[best].lo = lo; if (hi > s->gw[best].hi) s->gw[best].hi = hi; }
    else { s->gw[s->ngw].lo = lo; s->gw[s->ngw].hi = hi; s->ngw++; }
}
// DM_VK_X_WRITERS=<guest address hex>[+<length hex>] (live, debugging; parsed in gfx_vk_exec.c x_read): the GPU work about to call vk_gpu_wrote names itself here
// first (a dispatch's compute shader, storage register and size; the kind of a copy or fill), and each distinct name that writes over the watched range is logged
// once. Graphics stages that write storage buffers are not tracked today, so such a writer is not logged
char g_vk_writer[112];
void vk_wrote_by(const char *fmt, ...) { va_list ap; va_start(ap, fmt); vsnprintf(g_vk_writer, sizeof g_vk_writer, fmt, ap); va_end(ap); }
enum { WR_MAX = 64 };
static struct { char who[112]; } g_wr[WR_MAX]; static int g_nwr;
void vk_writers_reset(void) { g_nwr = 0; }
void vk_gpu_wrote(uint64_t a, uint64_t n) {
    // DM_VK_X_WRITERS: the check runs before the range merge below, which would fold a second writer's write of the same range into the first's and hide it
    if (g_x_writers_on && n && a < g_x_watch_hi && g_x_watch_lo < a + n) {
        int seen = 0; for (int i = 0; i < g_nwr; i++) seen |= !strcmp(g_wr[i].who, g_vk_writer);
        if (!seen && g_nwr < WR_MAX) { snprintf(g_wr[g_nwr++].who, sizeof g_wr[0].who, "%s", g_vk_writer);
            extern uint64_t g_tex_frame;
            VKLOG("writers: %s writes %#llx +%llu over the watched %#llx +%llu (present %llu, pass %d)", g_vk_writer, (unsigned long long)a, (unsigned long long)n,
                  (unsigned long long)g_x_watch_lo, (unsigned long long)(g_x_watch_hi - g_x_watch_lo), (unsigned long long)g_tex_frame, vk_sub_pass_now()); }
        else if (!seen) VKRED_ONCE(185, "writers: 64 distinct writers logged, later ones not (set DM_VK_X_WRITERS again for a new list)"); }
    uint64_t f = g_fence + 1;
    for (int i = g_ngw - 1; i >= 0 && g_gw[i].fence == f; i--) if (g_gw[i].lo <= a && a + n <= g_gw[i].hi) return;
    if (g_ngw == GW_MAX) { uint64_t done = vk_fence_done(); int o = 0; for (int i = 0; i < g_ngw; i++) if (g_gw[i].fence > done) g_gw[o++] = g_gw[i]; g_ngw = o; }
    if (g_ngw == GW_MAX) { memmove(g_gw, g_gw + 1, sizeof *g_gw * (GW_MAX - 1)); g_ngw--; }
    g_gw[g_ngw++] = (gpuw_t){a, a + n, f};
    sub_gpu_wrote(a, n);
}
int vk_gpu_pending(uint64_t a, uint64_t n) {
    if (!g_ngw) return 0;
    uint64_t done = vk_fence_done();
    for (int i = g_ngw - 1; i >= 0; i--) if (g_gw[i].fence > done && g_gw[i].lo < a + n && a < g_gw[i].hi) return 1;
    return 0;
}
// what a GPU read of [a, a + n) sees at this point of the command stream, into dst (memory, then the deferred updates in order)
void vk_ubd_read(uint64_t a, uint64_t n, uint8_t *dst) {
    if (vk_gpu_pending(a, n)) { static int k; g_ubd_racy++; if (k < 20) { k++; int cov = 0; for (int i = 0; i < g_nubd; i++) if (g_ubd[i].lo <= a && a + n <= g_ubd[i].hi) cov = 1;
        VKLOG("[RED] a CPU read of %#llx +%llu (%s) overlaps a GPU write not known to be complete", (unsigned long long)a, (unsigned long long)n, cov ? "covered by a deferred update" : "partly from memory"); } }
    { const void *src = vk_cpu_ptr(a, n); if (src) memcpy(dst, src, n); else memset(dst, 0, n); }   // (a virtual pool's range: through its physical memory)
    if (g_ninfl) { uint64_t done = vk_fence_done();
        for (int i = 0; i < g_ninfl; i++) { if (g_infl[i].fence <= done) continue; uint64_t lo = g_infl[i].lo > a ? g_infl[i].lo : a, hi = g_infl[i].hi < a + n ? g_infl[i].hi : a + n;
            if (lo < hi) memcpy(dst + (lo - a), g_infl[i].data + (lo - g_infl[i].lo), hi - lo); } }
    for (int i = 0; i < g_nubd; i++) { uint64_t lo = g_ubd[i].lo > a ? g_ubd[i].lo : a, hi = g_ubd[i].hi < a + n ? g_ubd[i].hi : a + n;
        if (lo < hi) memcpy(dst + (lo - a), g_ubd[i].p + (lo - g_ubd[i].lo), hi - lo); }
}
// a uniform binding [a, a + n) that deferred updates overlap: its shadow (0 when none overlaps). Shadows are kept per (address, size) until the next update.
int vk_ubd_shadow(uint64_t a, uint64_t n, VkBuffer *buf, VkDeviceSize *off) {
    if (!g_nubd || !vk_ubd_overlaps(a, n)) return 0;   // (a binding no deferred update touches reads memory on the GPU, after the in-flight copies: no shadow needed)
    enum { NC = 64 }; static struct { uint64_t a, n, gen; VkBuffer b; VkDeviceSize o; } c[NC];
    unsigned k = (unsigned)((a ^ n * 0x9E3779B97F4A7C15ull) >> 6) % NC;
    extern int g_x_noshc; if (c[k].a == a && c[k].n == n && c[k].gen == g_ubd_gen && !g_x_noshc) { *buf = c[k].b; *off = c[k].o; return 1; }
    uint8_t *d = vk_scratch(n, V.props.limits.minUniformBufferOffsetAlignment, buf, off); vk_ubd_read(a, n, d); g_ubd_bytes += n;
    c[k].a = a; c[k].n = n; c[k].gen = g_ubd_gen; c[k].b = *buf; c[k].o = *off; g_ubd_shadows++;
    return 1;
}
void mtl_sub_index(const uint64_t *h, int n) { (void)h; (void)n; }   // (mtl_exec_rest: gfx_vk_exec.c, the discards' lookahead)

// completion: a queue of committed submissions, each with its CPU-visible effects
typedef struct cpend { uint64_t fence, frame; post_op *ops; int n; int slot, rslot; int npass; vk_prof_t *pmeta; struct cpend *next; } cpend;   // pmeta: a timed submission's units
static cpend *g_ch, *g_ct; static pthread_mutex_t g_cmu = PTHREAD_MUTEX_INITIALIZER; static pthread_cond_t g_ccv = PTHREAD_COND_INITIALIZER;
uint64_t guest_ns(void);
// NVN counter timestamps are the GPU's 614.4 MHz ticks (nvn_timestamp.h), which hle_nvn.c converts back with nvn_timestamp_to_ns: written at the CPU's 19.2 MHz, every time a game took from them came out 32 times short
static uint64_t gpu_ticks(void) { uint64_t g = guest_ns(); if (!g) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); g = (uint64_t)t.tv_sec * 1000000000ull + (uint64_t)t.tv_nsec; } return nvn_timestamp_from_ns(g); }
// GPU time per NVN frame (the sum of its submissions' timestamp spans), for the timing line's GPU column: the frame a submission belongs to is the one being replayed
static uint64_t g_cur_frame, g_frame_gpu[64], g_frame_gpu_of[64];
void mtl_set_frame(uint64_t f) { __atomic_store_n(&g_cur_frame, f, __ATOMIC_RELAXED); }
double mtl_gpu_ms(uint64_t f) { return g_frame_gpu_of[f % 64] == f + 1 ? g_frame_gpu[f % 64] / 1e6 : -1; }
double vk_gpu_ms_recent(void) {   // the latest finished frame's GPU time among the last 8 (-1: none known)
    uint64_t f = __atomic_load_n(&g_cur_frame, __ATOMIC_RELAXED);
    for (uint64_t k = 2; k < 8 && k <= f; k++) { double g = mtl_gpu_ms(f - k); if (g >= 0) return g; }
    return -1;
}
// DM_VK_PASS_SPIKE=<ms> (read at start-up, with DM_VK_PASS_TIMES; default 25): a frame whose timed passes and dispatches, or whose submissions, take more than that lists its costliest
// ones, to find what a one-frame GPU spike drew (Tears of the Kingdom: 25-180 ms frames among 10-13 ms ones while fighting). Completion thread only
static uint64_t g_sf_frame = ~0ull; static struct { double ms; vk_prof_t m; } g_sf[PASSQ * 4]; static int g_nsf; static double g_sf_tot;
static void pass_spike_add(double ms, const vk_prof_t *m) { g_sf_tot += ms; if (g_nsf < PASSQ * 4) { g_sf[g_nsf].ms = ms; g_sf[g_nsf].m = *m; g_nsf++; } }
static void pass_spike_flush(void) {
    static double lim = -1; if (lim < 0) { const char *e = getenv("DM_VK_PASS_SPIKE"); lim = e && atof(e) > 0 ? atof(e) : 25; }
    double fg = g_sf_frame != ~0ull ? mtl_gpu_ms(g_sf_frame) : -1;   // the frame's submissions end to end: a spike outside the timed units shows as the difference
    if (g_sf_frame != ~0ull && (g_sf_tot > lim || fg > lim)) {
        for (int i = 0; i < g_nsf && i < 6; i++) for (int j = i + 1; j < g_nsf; j++) if (g_sf[j].ms > g_sf[i].ms) { __typeof__(g_sf[0]) x = g_sf[i]; g_sf[i] = g_sf[j]; g_sf[j] = x; }
        char b[1200]; int o = snprintf(b, sizeof b, "GPU spike: frame %llu, %.2f ms in %d passes and stretches between them (frame GPU %.2f ms, %.2f outside them):", (unsigned long long)g_sf_frame, g_sf_tot, g_nsf, fg, fg - g_sf_tot);
        for (int i = 0; i < g_nsf && i < 6 && o < (int)sizeof b - 100; i++) { const vk_prof_t *m = &g_sf[i].m;
            if (m->kind == VK_PU_GAP) o += snprintf(b + o, sizeof b - (size_t)o, " #%u %.2f ms (between passes: %u dispatches, %u copies, %u barriers);", m->fidx, g_sf[i].ms, m->disp, m->copies, m->bars);
            else o += snprintf(b + o, sizeof b - (size_t)o, " #%u %.2f ms (%ux%u colour %u depth %u, %u draws);", m->fidx, g_sf[i].ms, m->w, m->h, m->cfmt[0], m->dfmt, m->draws); }
        VKLOG("%s", b);
    }
    g_nsf = 0; g_sf_tot = 0;
}
static void *completion_main(void *a) {
    (void)a; pthread_setname_np(pthread_self(), "dm-vk-complete");
    for (;;) {
        pthread_mutex_lock(&g_cmu);
        while (!g_ch) pthread_cond_wait(&g_ccv, &g_cmu);
        cpend *c = g_ch; pthread_mutex_unlock(&g_cmu);
        if (V.has_timeline) wait_fence(c->fence); else vkWaitForFences(V.dev, 1, &g_fring[c->fence % NFRING], VK_TRUE, UINT64_MAX);
        if (g_shadow_pools && c->rslot >= 0) pools_copy_back(&g_slot[c->rslot]);
        uint64_t sub_ns = 0;   // (the submission's own span, for the GPU profile)
        if (g_tsq && c->slot >= 0) { uint64_t t[2]; if (vkGetQueryPoolResults(V.dev, g_tsq, 2 * (uint32_t)c->slot, 2, sizeof t, t, 8, VK_QUERY_RESULT_64_BIT) == VK_SUCCESS && t[1] > t[0])
            { uint64_t ns = (uint64_t)((double)(t[1] - t[0]) * V.props.limits.timestampPeriod); g_gpu_ns += ns; sub_ns = ns; int k = (int)(c->frame % 64);
              if (g_frame_gpu_of[k] != c->frame + 1) { g_frame_gpu[k] = 0; g_frame_gpu_of[k] = c->frame + 1; } g_frame_gpu[k] += ns; } }
        if (g_pq && c->npass && c->slot >= 0) {   // a timed submission's units: the span of each between its boundaries (gfx_vk_core.c "GPU profile")
            int np = c->npass; static uint64_t t[PASSQ + 1];
            if (c->frame != g_sf_frame) { pass_spike_flush(); g_sf_frame = c->frame; }
            if (vkGetQueryPoolResults(V.dev, g_pq, (uint32_t)(c->slot * (PASSQ + 1)), (uint32_t)np + 1, sizeof t, t, 8, VK_QUERY_RESULT_64_BIT) == VK_SUCCESS) {
                pthread_mutex_lock(&g_ps_mu);
                for (int i = 0; i < np; i++) { double ms = t[i + 1] > t[i] ? (double)(t[i + 1] - t[i]) * V.props.limits.timestampPeriod / 1e6 : 0; pass_spike_add(ms, &c->pmeta[i]); prof_add(&g_ps, &c->pmeta[i], ms, c->frame); }
                if (sub_ns) g_ps.sub_ms += sub_ns / 1e6;
                pthread_mutex_unlock(&g_ps_mu); }
        }
        uint64_t now = gpu_ticks(), tend = 0;
        if (g_tsq && c->slot >= 0) { uint64_t t[2]; if (vkGetQueryPoolResults(V.dev, g_tsq, 2 * (uint32_t)c->slot, 2, sizeof t, t, 8, VK_QUERY_RESULT_64_BIT) == VK_SUCCESS) tend = t[1]; }
        // A report's time was this thread's clock when it got here, less the GPU time from the report to the submission's end: its lateness (the fence wait's wake-up, the copy-back,
        // the submissions queued before) differs from one submission to the next by milliseconds, so reports of two submissions disagreed by that much, and a game that takes a
        // frame's GPU time from reports in two submissions saw it jump frame to frame. DM_VK_TS_CAL=1 (live, A/B): the GPU's clock plus the smallest lateness seen over the last 256
        // submissions (the closest to none, kept current against drift), the same offset for every report
        static int64_t cal_min[256]; static unsigned cal_n; int64_t cal_off = INT64_MAX;
        if (tend) { cal_min[cal_n++ & 255] = (int64_t)nvn_timestamp_from_ns((uint64_t)((double)tend * V.props.limits.timestampPeriod)) - (int64_t)now;   // (GPU − CPU, in NVN ticks: the largest is the least late)
            if (g_ts_cal) { cal_off = INT64_MIN; for (unsigned k = 0; k < (cal_n < 256 ? cal_n : 256); k++) if (cal_min[k] > cal_off) cal_off = cal_min[k]; } }
        for (int i = 0; i < c->n; i++) {
            post_op *o = &c->ops[i];
            if (o->kind == 1) { volatile uint32_t *val = vk_cpu_ptr((uint64_t)(uintptr_t)o->val, 4); if (val) { if (o->mode == 1) *val += o->v; else *val = o->v; } continue; }
            if (o->kind == 3) {   // DM_VK_X_RECCAP's capture (vk_post_reccap): logged here, after the fence, while the slot's scratch still holds the copy
                vk_reccap_t *d = o->p; const uint32_t *w = d->p;
                if (d->has_count && g_x_reccap_lines < 4096) { g_x_reccap_lines++;
                    VKLOG("reccap: present %llu pass %d vs %016llx count %#llx: %u", (unsigned long long)d->present, d->pass, (unsigned long long)d->key,
                          (unsigned long long)d->cnt, w[(d->stride * (d->n - 1) + d->rs) / 4]); }
                for (uint32_t k = 0; k < d->n && g_x_reccap_lines < 4096; k++) { g_x_reccap_lines++; const uint32_t *r = (const uint32_t *)((const uint8_t *)w + (size_t)k * d->stride);
                    if (d->rs == 20) VKLOG("reccap: present %llu pass %d vs %016llx record %#llx: count %u instances %u first index %u vertex offset %d first instance %u",
                                           (unsigned long long)d->present, d->pass, (unsigned long long)d->key, (unsigned long long)(d->addr + (uint64_t)k * d->stride), r[0], r[1], r[2], (int32_t)r[3], r[4]);
                    else VKLOG("reccap: present %llu pass %d vs %016llx record %#llx: count %u instances %u first vertex %u first instance %u",
                               (unsigned long long)d->present, d->pass, (unsigned long long)d->key, (unsigned long long)(d->addr + (uint64_t)k * d->stride), r[0], r[1], r[2], r[3]); }
                if (g_x_reccap_lines >= 4096 && !g_x_reccap_capped) { g_x_reccap_capped = 1;
                    VKLOG("reccap: 4096 lines reached, no more logged this setting%s", g_x_reccap_dropped ? " (draws' record ranges past 128 a pass dropped too)" : ""); }
                free(d); continue; }
            uint64_t ts = now, tq;
            // DM_VK_TS_FLAT=1 (live, debugging): every report of a submission at its completion time, so a game measures no GPU time between two of them
            if (!g_ts_flat && o->q >= 0 && tend && vkGetQueryPoolResults(V.dev, g_cq, (uint32_t)o->q, 1, 8, &tq, 8, VK_QUERY_RESULT_64_BIT) == VK_SUCCESS && tq <= tend) {
                if (cal_off != INT64_MAX) ts = (uint64_t)((int64_t)nvn_timestamp_from_ns((uint64_t)((double)tq * V.props.limits.timestampPeriod)) - cal_off);   // (DM_VK_TS_CAL)
                else { uint64_t back = nvn_timestamp_from_ns((uint64_t)((double)(tend - tq) * V.props.limits.timestampPeriod)); if (back < now) ts = now - back; } }
            else if (!g_ts_flat && cal_off != INT64_MAX) ts = (uint64_t)((int64_t)nvn_timestamp_from_ns((uint64_t)((double)tend * V.props.limits.timestampPeriod)) - cal_off);   // (no query of its own: the submission's end)
            // render_cond: a timestamp report's counter field is 0, as on the GPU (gfx_vk_exec.c render_cond evaluates conditions on it)
            uint64_t rec[2] = {o->type == 0 || o->type == 0x10 ? (DM_GAME_RENDER_COND ? 0 : ts) : o->type == NVN_COUNTER_TYPE_SAMPLES_PASSED ? (1u << 20) : 0, ts};
            void *at = vk_cpu_ptr(o->addr, sizeof rec);
            if (at && o->type == NVN_COUNTER_TYPE_SAMPLES_PASSED) { uint64_t old; memcpy(&old, at, 8); g_ctr_seen++; g_ctr_zero += old == 0; }   // census: the game cleared it (or it was never written)
            if (at) memcpy(at, rec, sizeof rec);
        }
        pthread_mutex_lock(&g_cmu); g_ch = c->next; if (!g_ch) g_ct = NULL; pthread_mutex_unlock(&g_cmu);
        if (g_prereset > 0 && c->rslot >= 0) { vksub_t *s = &g_slot[c->rslot];   // its command buffers reset here (slot_recycle); the slot is not taken before g_done passes this fence
            if (s->fence == c->fence) { for (int i = 0; i < s->nmain_alloc; i++) vkResetCommandBuffer(s->mains[i], 0); for (int i = 0; i < s->nup_alloc; i++) vkResetCommandBuffer(s->ups[i], 0); } }
        pthread_mutex_lock(&g_done_mu); if (c->fence > g_done) __atomic_store_n(&g_done, c->fence, __ATOMIC_RELEASE); pthread_cond_broadcast(&g_done_cv); pthread_mutex_unlock(&g_done_mu);
        free(c->ops); free(c->pmeta); free(c);
    }
    return NULL;
}
static void completion_start(void) { pthread_t t; if (pthread_create(&t, NULL, completion_main, NULL)) die("[vk] completion thread"); pthread_detach(t); }
uint64_t mtl_sub_commit(void) {
    { void vk_prof_end(void); vk_prof_end(); }
    vksub_t *s = cur();
    { void vk_prof_end_sub(void); vk_prof_end_sub(); }   // the pass still open ends (vk_end_pass_land), then the GPU profile's last stretch
    if (g_tsq) vkCmdWriteTimestamp(s->main, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, g_tsq, 2 * (uint32_t)g_cur + 1);
    vk_vt_sub_end(s->main);   // (gfx_vk_vt.c: the shaders' tile requests visible to the host once the fence has passed)
    VkResult er = vkEndCommandBuffer(s->main); if (s->cb_err && er >= 0) er = VK_ERROR_OUT_OF_HOST_MEMORY;
    if (er < 0) {   // the driver refused something recorded in it: logged with what the submission held, then it is dropped and the game goes on
        extern uint32_t g_sub_ops[64]; char ops[600]; int o = 0;
        for (int i = 0; i < 64 && o < (int)sizeof ops - 16; i++) if (g_sub_ops[i]) o += snprintf(ops + o, sizeof ops - (size_t)o, " %d:%u", i, g_sub_ops[i]);
        static int n; if (n++ < 3) VKLOG("[RED] vkEndCommandBuffer failed (%d): submission dropped; record ops (op:count)%s", er, ops);
        vk_bisect_result(0);
        { void vk_sparse_lost(void); vk_sparse_lost(); } vk_vt_lost();   // tile uploads of the dropped command buffers: the sparse images upload their mapped tiles again at their next look,
                                                                         // and the VT pool's (gfx_vk_vt.c: its slots go back, its page table is written again)
        if (s->nmain_sent) {   // parts already sent may still run: nothing more is submitted (the submission below only signals its fence)
            s->nmain_sent = s->nmain_used; s->up_used = 0; s->cb_err = 0; s->no_ts = 1; vk_ubd_land_drop();
        } else {
            vkResetCommandPool(V.dev, s->pool, 0); VkCommandBufferBeginInfo bi = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, NULL, VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT};
            s->main = s->mains[0]; s->nmain_used = 1; s->cb_err = 0; s->no_ts = 1;   // no timestamps: the slot's queries still hold an older submission's
            if (s->nup_used > 1) { s->up = s->ups[0]; s->nup_used = 1; }
            VKCHK(vkBeginCommandBuffer(s->main, &bi)); g_cmd_gen++; s->up_used = 0; vk_ubd_land_drop();
            VKCHK(vkEndCommandBuffer(s->main));
        }
    }
    else vk_bisect_result(1);
    { extern uint32_t g_sub_ops[64]; memset(g_sub_ops, 0, sizeof g_sub_ops); }
    if (s->up_used) { vk_img_barrier_all(s->up); VKCHK(vkEndCommandBuffer(s->up)); }
    uint64_t f = ++g_fence;
    VkCommandBuffer cbs[1 + MAIN_CBS]; uint32_t ncb = 0; if (s->up_used) cbs[ncb++] = s->up; for (int i = s->nmain_sent; i < s->nmain_used; i++) cbs[ncb++] = s->mains[i];
    VkTimelineSemaphoreSubmitInfo ts = {VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO, NULL, 0, NULL, 1, &f};
    VkSubmitInfo si = {VK_STRUCTURE_TYPE_SUBMIT_INFO, &ts, 0, NULL, NULL, ncb, cbs, 1, &g_timeline};
    VkSemaphore ws; uint64_t wv; VkPipelineStageFlags wst = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
    if (V.has_timeline && vk_sparse_flush(&ws, &wv)) { ts.waitSemaphoreValueCount = 1; ts.pWaitSemaphoreValues = &wv; si.waitSemaphoreCount = 1; si.pWaitSemaphores = &ws; si.pWaitDstStageMask = &wst; }   // (as vk_cmd_split)
    VkFence fe = VK_NULL_HANDLE;
    if (!V.has_timeline) {   // the ring's fence for f: free once the completion thread has passed f - NFRING (already so: slot_recycle waited for f - NSLOT)
        if (f > NFRING) mtl_wait(f - NFRING, ~0ull);
        fe = g_fring[f % NFRING]; VKCHK(vkResetFences(V.dev, 1, &fe)); si.pNext = NULL; si.signalSemaphoreCount = 0;
    }
    perf_hint_apply();   // (DM_VK_PERF_HINT, live)
    struct timespec t0, t1; clock_gettime(CLOCK_MONOTONIC, &t0); pthread_mutex_lock(&g_vk_queue_mu); VkResult r = vkQueueSubmit(V.q, 1, &si, fe); pthread_mutex_unlock(&g_vk_queue_mu); clock_gettime(CLOCK_MONOTONIC, &t1); g_ns_submit += (uint64_t)((t1.tv_sec - t0.tv_sec) * 1000000000ll + (t1.tv_nsec - t0.tv_nsec));
    if (r) { void vk_crumbs_report(void); vk_crumbs_report(); die("[vk] vkQueueSubmit: %d (device lost?)", r); }
    s->fence = f;
    cpend *c = calloc(1, sizeof *c); c->fence = f; c->frame = __atomic_load_n(&g_cur_frame, __ATOMIC_RELAXED); c->n = s->npost; c->rslot = g_cur; c->slot = s->no_ts ? -1 : g_cur; s->no_ts = 0;
    c->npass = g_sub_prof[g_cur] ? g_npass[g_cur] : 0; if (c->npass && (c->pmeta = malloc(sizeof(vk_prof_t) * (size_t)c->npass))) memcpy(c->pmeta, g_pmeta[g_cur], sizeof(vk_prof_t) * (size_t)c->npass); else c->npass = 0;
    if (s->npost) { c->ops = malloc(s->npost * sizeof *c->ops); memcpy(c->ops, s->post, s->npost * sizeof *c->ops); }
    pthread_mutex_lock(&g_cmu); if (g_ct) g_ct->next = c; else g_ch = c; g_ct = c; pthread_cond_signal(&g_ccv); pthread_mutex_unlock(&g_cmu);
    s->npost = 0; g_cur = -1;
    pthread_mutex_unlock(&g_sub_lock);
    vk_mem_log();   // (DM_VK_MEMLOG: rate-limited by itself)
    if (g_x_subidle) mtl_wait(f, 2000000000ull);   // DM_VK_X_SUBIDLE=1 (live, slow): the submitting thread waits for the GPU to finish each submission, so no CPU work overlaps GPU work
    return f;
}
uint64_t mtl_last_fence(void) { return g_fence; }
int mtl_wait(uint64_t fence, uint64_t timeout_ns) {
    struct timespec dl; clock_gettime(CLOCK_REALTIME, &dl);
    uint64_t ns = (uint64_t)dl.tv_nsec + (timeout_ns > 3600ull * 1000000000ull ? 3600ull * 1000000000ull : timeout_ns);
    dl.tv_sec += (time_t)(ns / 1000000000ull); dl.tv_nsec = (long)(ns % 1000000000ull);
    pthread_mutex_lock(&g_done_mu);
    while (g_done < fence) if (pthread_cond_timedwait(&g_done_cv, &g_done_mu, &dl)) break;
    int ok = g_done >= fence; pthread_mutex_unlock(&g_done_mu);
    return ok;
}
void mtl_finish(void) { mtl_wait(g_fence, ~0ull); }
void mtl_drain(void) { mtl_finish(); vk_pipeline_cache_save(); }
