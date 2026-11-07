// Android port: the Vulkan backend behind the replay interface of nvn_mtl.h (what nvn_mtl.m implements on Metal).
// Files: gfx_vk_core.c (device, memory pools, submissions, completion), gfx_vk_tex.c (formats, textures, samplers, registration tables),
// gfx_vk_shader.c (shader bundle, modules, layouts), gfx_vk_exec.c (command replay: passes, pipelines, descriptors, draws, copies), gfx_vk_present.c (window, swapchain),
// gfx_vk_fsr.c (the present's upscaling passes, FSR 1: its own header gfx_vk_fsr.h).
// Version 1 is correct-first: whole-pipeline barriers between passes, full pipelines with a small dynamic state set, descriptor sets per draw.
#pragma once
#define VK_NO_PROTOTYPES
#define VK_USE_PLATFORM_ANDROID_KHR
#include <android/hardware_buffer.h>
#include <android/native_window.h>
#include <vulkan/vulkan.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdint.h>
#include "dm.h"
#include "nvn_obj.h"
#include "nvn_mtl.h"

#define VKLOG(...) LOG("[vk] " __VA_ARGS__)
#define VKRED_ONCE(id, ...) do { static int once_##id; if (!once_##id++) LOG("[RED] [vk] " __VA_ARGS__); } while (0)
#define VKCHK(x) do { VkResult r_ = (x); if (r_ < 0) die("[vk] %s failed: %d", #x, r_); } while (0)

// ---- Vulkan entry points (loaded from libvulkan.so) ----
#define VK_FNS(X) \
    X(vkCreateInstance) X(vkEnumeratePhysicalDevices) X(vkGetPhysicalDeviceProperties) X(vkGetPhysicalDeviceProperties2) X(vkGetPhysicalDeviceFeatures2) \
    X(vkGetPhysicalDeviceMemoryProperties) X(vkGetPhysicalDeviceQueueFamilyProperties) X(vkGetPhysicalDeviceFormatProperties) X(vkCreateDevice) X(vkGetDeviceProcAddr) \
    X(vkEnumerateDeviceExtensionProperties) X(vkGetPhysicalDeviceSurfaceCapabilitiesKHR) X(vkGetPhysicalDeviceSurfaceFormatsKHR) X(vkGetPhysicalDeviceSurfacePresentModesKHR) \
    X(vkCreateAndroidSurfaceKHR) X(vkDestroySurfaceKHR) X(vkGetPhysicalDeviceSurfaceSupportKHR) \
    X(vkGetPhysicalDeviceSparseImageFormatProperties) X(vkGetPhysicalDeviceImageFormatProperties) X(vkGetPhysicalDeviceMemoryProperties2) X(vkGetPhysicalDeviceFormatProperties2)
#define VK_DEV_FNS(X) \
    X(vkGetDeviceQueue) X(vkCreateBuffer) X(vkDestroyBuffer) X(vkAllocateMemory) X(vkFreeMemory) X(vkBindBufferMemory) X(vkBindImageMemory) X(vkGetBufferMemoryRequirements) \
    X(vkGetImageMemoryRequirements) X(vkCreateImage) X(vkDestroyImage) X(vkCreateImageView) X(vkDestroyImageView) X(vkCreateSampler) X(vkDestroySampler) X(vkMapMemory) \
    X(vkCreateCommandPool) X(vkResetCommandPool) X(vkResetCommandBuffer) X(vkAllocateCommandBuffers) X(vkBeginCommandBuffer) X(vkEndCommandBuffer) X(vkQueueSubmit) X(vkQueueWaitIdle) X(vkDeviceWaitIdle) \
    X(vkCreateSemaphore) X(vkWaitSemaphores) X(vkGetSemaphoreCounterValue) X(vkCreateFence) X(vkWaitForFences) X(vkResetFences) X(vkDestroyFence) \
    X(vkCmdPipelineBarrier) X(vkCmdCopyBuffer) X(vkCmdCopyBufferToImage) X(vkCmdCopyImageToBuffer) X(vkCmdCopyImage) X(vkCmdBlitImage) X(vkCmdFillBuffer) X(vkCmdUpdateBuffer) \
    X(vkCmdClearColorImage) X(vkCmdClearDepthStencilImage) X(vkCmdClearAttachments) X(vkCmdResolveImage) \
    X(vkCreateRenderPass) X(vkCreateRenderPass2) X(vkDestroyRenderPass) X(vkCreateFramebuffer) X(vkDestroyFramebuffer) X(vkCmdBeginRenderPass) X(vkCmdEndRenderPass) X(vkCmdBeginRendering) X(vkCmdEndRendering) X(vkCmdBindPipeline) X(vkCmdBindDescriptorSets) X(vkCmdBindVertexBuffers) X(vkCmdBindIndexBuffer) \
    X(vkCmdSetViewport) X(vkCmdSetScissor) X(vkCmdSetBlendConstants) X(vkCmdSetStencilReference) X(vkCmdSetStencilCompareMask) X(vkCmdSetStencilWriteMask) \
    X(vkCmdSetDepthBias) X(vkCmdSetLineWidth) X(vkCmdDraw) X(vkCmdPushConstants) X(vkCmdDrawIndexed) X(vkCmdDrawIndirect) X(vkCmdDrawIndexedIndirect) X(vkCmdDispatch) X(vkCmdDispatchIndirect) \
    X(vkCreateShaderModule) X(vkDestroyShaderModule) X(vkCreateDescriptorSetLayout) X(vkCreatePipelineLayout) X(vkCreateGraphicsPipelines) X(vkDestroyPipeline) X(vkCreateComputePipelines) X(vkCreatePipelineCache) \
    X(vkGetPipelineCacheData) X(vkCreateDescriptorPool) X(vkResetDescriptorPool) X(vkAllocateDescriptorSets) X(vkUpdateDescriptorSets) \
    X(vkCreateBufferView) X(vkDestroyBufferView) X(vkGetAndroidHardwareBufferPropertiesANDROID) \
    X(vkCreateQueryPool) X(vkCmdResetQueryPool) X(vkCmdWriteTimestamp) X(vkGetQueryPoolResults) \
    X(vkCreateSwapchainKHR) X(vkDestroySwapchainKHR) X(vkGetSwapchainImagesKHR) X(vkAcquireNextImageKHR) X(vkQueuePresentKHR) \
    X(vkCmdBindTransformFeedbackBuffersEXT) X(vkCmdBeginTransformFeedbackEXT) X(vkCmdEndTransformFeedbackEXT) \
    X(vkGetRefreshCycleDurationGOOGLE) X(vkGetPastPresentationTimingGOOGLE) \
    X(vkQueueBindSparse) X(vkGetImageSparseMemoryRequirements) X(vkCmdDrawIndirectCount) X(vkCmdDrawIndexedIndirectCount) X(vkCmdDrawMeshTasksEXT) X(vkCmdDispatchBase)
// (a loaded pointer does not say a device has the function: the loader hands out trampolines. V.has_* decide: has_dic, has_mesh, has_sparse, has_dbase)
#define VK_DECL(f) extern PFN_##f f;
VK_FNS(VK_DECL) VK_DEV_FNS(VK_DECL)

// ---- device ----
typedef struct {
    VkInstance inst; VkPhysicalDevice pd; VkDevice dev; VkQueue q; uint32_t qf;
    VkPhysicalDeviceProperties props; VkPhysicalDeviceMemoryProperties mem;
    VkPipelineCache pcache;
    int has_bc, has_astc, has_gs, has_depth_clip_ctl, has_custom_border, has_robust2, has_xfb, has_imageless, has_dtiming, has_dynrender, has_timeline;
    float max_aniso;
    // Switch 2 (gfx_vk_core.c vk_init): sparse residency images (sparseBinding + sparseResidencyImage2D, a queue with VK_QUEUE_SPARSE_BINDING_BIT: sq, the graphics queue q when its family has it),
    // sparse aliasing (one physical tile at several places: sparseResidencyAliased), VK_EXT_mesh_shader's task and mesh stages, vkCmdDraw(Indexed)IndirectCount, VK_EXT_blend_operation_advanced
    int has_sparse, has_sparse_alias, has_mesh, has_dic, has_adv_blend, adv_npm, adv_ovl, adv_all; VkQueue sq; uint32_t sqf; uint32_t task_max_groups, mesh_max_groups, adv_max_att;
    // what a bundle's module may ask beyond its stage (gfx_vk_exec.c mesh_need, spill_dispatch): the SPIR-V version the device takes (0x10400: 1.4), the stages and operations of subgroup
    // operations (VkPhysicalDeviceSubgroupProperties), VK_EXT_mesh_shader's limits, vkCmdDispatchBase (Vulkan 1.1)
    uint32_t spv_max, sg_stages, sg_ops; int has_dbase;
    struct { uint32_t task_smem, mesh_smem, task_payload, task_pl_smem, mesh_pl_smem, out_verts, out_prims, task_inv, mesh_inv, task_size[3], mesh_size[3]; } ml;
} vkdev_t;
extern vkdev_t V;
uint32_t vk_mem_type(uint32_t bits, VkMemoryPropertyFlags want, VkMemoryPropertyFlags avoid);
void vk_init(void);   // once, from mtl_init or the first use
int vk_ready(void);   // vk_init has made the device and loaded its entry points (a background thread waits for it rather than calling vk_init)

// ---- memory pools (gfx_vk_core.c): the guest's pool memory is GPU memory (AHardwareBuffer mapped over it) ----
typedef struct { uint64_t lo, hi; VkBuffer buf; VkDeviceMemory mem; void *ahb; int zero_copy; uint8_t *shadow; uint64_t serial, base, end; int flags, synced; uint64_t stamp, stamp0; const void *span;
                 int state; uint32_t ngran; uint64_t gran; uint8_t *placed; int said; int part, ovl; void *spare; } vkpool_t;
// serial: unique per pool made (caches of views over a pool's buffer key on it); [base, end): the game's pool, of which [lo, hi) is one chunk (ZC_CHUNK) or one placed part; flags: the NVN pool's
// (vk_pool_set_flags); synced, stamp, stamp0, span: a copied pool's incremental sync. state (gfx_vk_core.c "Lazy placement"): POOL_UNPLACED, the game's own pages, no buffer yet; POOL_PLACED,
// GPU memory (zero_copy, or a copy: shadow); POOL_PARTS, a pool of DM_VK_POOL_PART_MB or more whose granules (gran bytes, ngran of them, placed[] 1 when placed) become placed parts of their own
// (part 1). ovl: a chunk or pool over memory another one holds (gfx_vk_core.c pool_find); spare: the table's list of entries free for reuse
enum { POOL_UNPLACED, POOL_PLACED, POOL_PARTS };
vkpool_t *vk_pool_of(uint64_t addr, uint64_t len);    // the pool containing [addr, addr+len), NULL if none: what the CPU knows of it (it may have no GPU buffer yet)
vkpool_t *vk_pool_gpu(uint64_t addr, uint64_t len);   // the same, made GPU memory first if it is not (the GPU is given the range): NULL if none, or across two placed parts
int vk_buf_at(uint64_t addr, uint64_t len, VkBuffer *buf, VkDeviceSize *off);   // GPU buffer + offset of a pool address range (vk_pool_gpu_at)
vkpool_t *vk_pool_gpu_at(uint64_t *a, uint64_t n);   // vk_pool_gpu, *a translated to the physical memory when it lies in a virtual pool's range (Android: hle_nvn.c nvn_virt_xlat)
void *vk_cpu_ptr(uint64_t a, uint64_t n);            // the host pointer the CPU reads or writes guest range [a, a+n) through (translated as above); NULL: never dereference it
int vk_readable(const void *p, uint64_t n);          // every page of [p, p+n) readable (not a guest range reserved and unmapped): 1, 0, or -1 when it cannot be told (gfx_vk_core.c)
uint64_t vk_unreadable_at(const void *p, uint64_t n);   // the offset of its first page that cannot be read (n: none; UINT64_MAX: cannot be told)
void vk_mem_log(void);   // DM_VK_MEMLOG=<seconds>: a memory line at most that often (pools placed, images, sparse tiles, scratch, MemAvailable)
// Pools that could not be made zero-copy keep a GPU copy refreshed before each submission (vk_pools_sync, called from mtl_sub_commit)
void vk_pools_sync(VkCommandBuffer cb);

// ---- submissions (gfx_vk_core.c) ----
typedef struct vksub vksub_t;
VkCommandBuffer vk_cmd(void);          // the current submission's main command buffer (recording)
int vk_sub_open(void);                 // a submission is being recorded
VkCommandBuffer vk_upload_cmd(void);   // the current submission's upload command buffer (runs before the main one)
void *vk_scratch(VkDeviceSize n, VkDeviceSize align, VkBuffer *buf, VkDeviceSize *off);   // host-visible memory that lives until this submission completes
void vk_defer_free_buffer(VkBuffer b, VkDeviceMemory m);   // destroyed once the GPU has passed the current submission
void vk_defer_free_image(VkImage i, VkDeviceMemory m);
VkDeviceMemory vk_image_memory(const VkMemoryRequirements *req, uint32_t type, VkDeviceSize *off, void **sub);   // NULL on failure; *sub: a slot of a shared block, or NULL (memory of its own)
uint64_t vk_image_sub_size(const void *sub, int *own);   // the bytes vk_image_memory gave (*sub), *own 1 for memory of its own
void vk_defer_free_image_sub(VkImage i, void *sub);
void vk_defer_free_view(VkImageView v);
VkDescriptorSet vk_alloc_set(VkDescriptorSetLayout l);    // from this submission's descriptor pools
uint64_t vk_fence_done(void);   // the highest submission fence the GPU has completed

// ---- formats and textures (gfx_vk_tex.c) ----
// dec: a BC format the device lacks, decoded on the CPU (gfx_vk_tex.c bc_decode) into vk, dbpp bytes a texel; bw, bh and bpb stay the guest's blocks
typedef struct { VkFormat vk; VkComponentMapping swz; uint8_t kind; uint8_t depth, stencil, compressed, is_int, is_sint, bw, bh, bpb, dec, dbpp; } vkfmt_t;
const vkfmt_t *vk_fmt(int nvn_format);   // NULL when the format has no Vulkan counterpart on this device
void vk_bc_decode(const vkfmt_t *f, const uint8_t *src, size_t pitch, uint32_t w, uint32_t h, uint32_t z, uint8_t *dst);   // (f->dec) gfx_vk_tex.c bc_decode
typedef struct vkimg vkimg_t;
struct vkimg {
    VkImageUsageFlags usage; VkImageCreateFlags cflags;   // as created (an imageless framebuffer names them)
    VkImage img; VkDeviceMemory mem; void *sub; VkFormat fmt; VkImageViewType vtype; VkImageAspectFlags aspect;
    uint32_t w, h, d, layers, levels, samples; int target, nvn_format;
    VkImageLayout layout;           // version 1 keeps every image in GENERAL after its first use
    int packaged;                   // filled from guest memory (deswizzled upload), re-uploaded when its pages change
    uint64_t pkg_stamp; const void *pkg_span; uint64_t pkg_seen;
    uint64_t pkg_hash;              // hash of the guest bytes it was last uploaded from: pages written around it with its own bytes unchanged keep what the GPU wrote (gfx_vk_tex.c vk_tex)
    const uint8_t *pkg_mem; size_t pkg_len; uint64_t pkg_used; int pkg_released;   // its storage, the present it was last drawn at, protection given up (vk_tex_sweep)
    uint64_t used;                  // the present it was last looked up at (vk_tex: evict_overlaps)
    uint64_t rt_last, rt_frames;     // as an attachment: 1 + the present it was last drawn in, and in how many presents in a row up to it (gfx_vk_exec.c async_pipe)
    uint64_t fdz_frame;             // depth images: 1 + the present at which a fragment shader that replaces depth last wrote it (gfx_vk_exec.c fdz)
    int gpu_w;                      // the GPU wrote it: an attachment, a storage image, a copy or clear destination (gfx_vk_exec.c); lin_copy keeps that until the CPU writes its storage
    uint64_t lin_stamp; const void *lin_span; const uint8_t *lin_mem; size_t lin_len;   // lin_copy: the storage watched since the GPU wrote it
    uint64_t lin_gen;               // linear textures: the upload command buffer (g_up_gen) that last copied it from its pool (vk_tex: lin_copy)
    uint64_t depth_sub;   // the submission (gfx_vk_exec.c g_set_epoch) whose passes last used it as their depth attachment
    uint64_t depth_cut;   // and the command buffer within it (gfx_vk_exec.c g_cuts)
    int fixed_fmt;   // created with a format list (DM_VK_UBWC): views only in its format or its sRGB twin
    VkFormat fmt2;   // with fixed_fmt: the other format of its list (its sRGB twin), VK_FORMAT_UNDEFINED when the list is its format alone
    VkImageView full;               // a view of the whole image as created
    VkBufferView tbuf;              // buffer textures: a texel buffer view of the pool
    uint64_t last_use;              // submission fence of the last use
    uint64_t pool; int64_t off, size; int flags;   // its storage (pool, offset, layout size) and NVN texture flags: images over the same memory find each other
    struct vkimg *ovl[8]; int novl;   // the other images over some of the same memory (a texture of another description over it: on the Switch it sees these texels)
    uint64_t wseq, rseen;             // when it was last rendered to (gfx_vk_tex.c g_wseq); the newest overlapping write already reported
    struct vksparse *sp;              // a sparse-resident image (gfx_vk_tex.c sparse_*): its tiles follow the game's mappings of a sparse virtual pool, NULL for any other image
    uint32_t sp_vbase;                // a texture of a sparse pool uploaded whole: its first level whose pages are all mapped (sampled from there, as nvn_mtl.m sparse_view)
    uint64_t sp_sub, sp_gen;          // and the submission (vk_sub_serial) of its last look, the pool mappings' generation (nvn_sparse_gen) at its upload
    uint32_t lv0;                     // the texture level of its level 0 (a dense image of a sparse texture, or a sparse one past the sparse address space, starts at a coarser level): views and
                                      // targets of a finer level start at its level 0 (gfx_vk_tex.c vk_tex_view); copies into a level it lacks are left out
    struct vkdense *dn;               // a dense image of a sparse texture that cannot be sparse-resident (gfx_vk_tex.c dense_*), NULL for any other image
    uint8_t promo;                    // a sparse image made in its texture format's exact wider twin (R16 for R8: gfx_vk_tex.c "promoted formats"), 1 + its table entry; 0 for any other
    // its NVN discards' history (gfx_vk_exec.c "discards"), by aspect: discarded at a pass's end and not written since (pend), its store left out there (drop), read while
    // discarded (veto: its store is never left out again), and the times a discard was followed by a whole write (a clear or a discard as a pass's load) before any read (ok)
    uint8_t dsc_pend, dsc_drop, dsc_veto, dsc_ok[3];   // (aspect bits: gfx_vk_exec.c VK_DSC_COLOR, _DEPTH, _STENCIL)
    uint64_t kept_gen;                // a view of it is in a descriptor set kept across submissions since that keep generation (gfx_vk_exec.c g_kgen): dropping its views then drops them (g_tex_kgen)
    uint64_t msub;                    // 1 + the submission (vk_sub_serial) whose main command buffers last wrote it or took it out of UNDEFINED (vk_img_wrote, vk_tex_prepare): an upgrade's
                                      // copy (gfx_vk_tex.c img_upgrade), in the upload command buffer that runs before them, would miss what they did
};
void vk_dsc_read_(vkimg_t *im, const char *how);   // its contents read other than as an attachment (sampled, a storage image, a copy's source, read back, presented): gfx_vk_exec.c
static inline void vk_dsc_read(vkimg_t *im, const char *how) { if (im && im->dsc_pend) vk_dsc_read_(im, how); }
static inline void vk_dsc_wrote(vkimg_t *im) { if (im) im->dsc_pend = im->dsc_drop = 0; }   // written other than as an attachment (a copy's destination, a clear, a storage image)
// A promoted image (vkimg_t promo): its texture's own format, that format's bytes a texel (0: not promoted), and texels between the two (the wider twice the bytes)
VkFormat vk_promo_from(const vkimg_t *im);
uint32_t vk_promo_bpt(const vkimg_t *im);
void vk_promo_widen(const vkimg_t *im, const uint8_t *src, size_t ntexels, uint8_t *dst);
void vk_promo_narrow(const vkimg_t *im, const uint8_t *src, size_t ntexels, uint8_t *dst);   // (rounded as drawing into the narrower format would)
vkimg_t *vk_tex(const tex_t *t);                                 // the image of a texture object (created, uploaded or refreshed as needed)
void vk_lrz_forget(vkimg_t *im);                                 // an image is being dropped: the pass code forgets it (gfx_vk_exec.c)
int vk_window_bottom_up(void);                                   // the game draws in lower-left origin mode: its window image is stored bottom-up (gfx_vk_exec.c)
VkImageView vk_tex_view(const tex_t *t, const view_t *v, int dim, int arrayed, int as_attachment, vkimg_t **img_out);   // a view for sampling / storage / rendering
void vk_watch(const vkimg_t *im, const char *what);   // DM_VK_WATCH (gfx_vk_exec.c)
vkimg_t *vk_tex_by_id(int tid, const view_t **view_out, const tex_t **tex_out);   // registered texture ID → image (NULL if none)
VkSampler vk_samp_by_id(int sid);
extern __thread VkFormat vk_view_fmt; extern __thread uint32_t vk_view_layers;   // the format and layer count of the view vk_tex_view last returned on this thread
VkImageView vk_dummy_view(int dim, int arrayed, int ms, int storage, int is_int);
VkSampler vk_dummy_sampler(void);
VkBufferView vk_dummy_tbuf(int storage);
void vk_end_pass_land(void);   // vk_end_pass, then the deferred uniform updates land (before copies, dispatches, the submission's end)
void vk_img_barrier_all(VkCommandBuffer cb);   // version 1: one global memory barrier (all images stay in GENERAL)
void vk_tex_prepare(VkCommandBuffer cb, vkimg_t *im);   // first use: transition from UNDEFINED to GENERAL (and upload a packaged texture)
// Sparse binding (gfx_vk_tex.c): the tile maps and unmaps the replay asked for since the last queue submission, sent with vkQueueBindSparse before it (gfx_vk_core.c, every
// vkQueueSubmit of a submission's parts). 1 when binds were sent: the submission waits for *sem to reach *val (a timeline semaphore the binds signal)
int vk_sparse_flush(VkSemaphore *sem, uint64_t *val);
void vk_sparse_gpu_wrote(vkimg_t *im);   // the GPU writes a sparse image (an attachment, a storage image, a copy or clear destination): its shared tiles hold what the memory does not
uint64_t vk_sub_serial(void);            // submissions begun so far (gfx_vk_core.c): a check made once per submission keys on it

// ---- shaders (gfx_vk_shader.c) ----
enum { RK_SUPPORT, RK_CBUF, RK_C1, RK_DRV, RK_SBUF, RK_UBO_PTR, RK_SBUF_UBO, RK_TEX, RK_TBUF, RK_IMG, RK_IMG_TBUF, RK_TEX_C2A, RK_TEX_UBO, RK_TBUF_UBO, RK_IMG_UBO, RK_SPILL, RK_UNKNOWN = 255 };
// RK_SPILL (shader_vk.py SHARED_MAX): a compute shader's shared memory in a storage buffer, off words a workgroup in samp_arg slices (workgroup w of the dispatch takes slice w mod samp_arg):
// bind vk_spill_buffer(off * 4 * samp_arg) (gfx_vk_shader.c), whole. Workgroups samp_arg apart must not run together: a larger dispatch is cut into parts (gfx_vk_exec.c spill_dispatch)
// RK_DRV's reg is the constant bank (shader_vk.py classify: <stage>_c<n> outside c[1] and the uniform slots c[3..16]): 0 / 2 the driver segments c0 / c2, DMSH_PUSH_BANK the NVN2 push
// constants (c[24], a 0x100-byte block: gfx_vk_exec.c push_desc), DMSH_MESH_FIRST_SLOT a task or mesh shader's first workgroup (16 bytes, word 0: the draw's first, as nvn_geometry.inc
// mtk_encode binds it). A uniform slot past 15 (RK_CBUF reg DMSH_PUSH_BANK, or DMSH_PUSH_BANK - 3 counted as c[3 + reg]) is taken as the push block too
enum { RD_UBO, RD_SSBO, RD_SAMPLED, RD_STORAGE, RD_TBUF, RD_STORAGE_TBUF, RD_IMAGE, RD_SAMPLER };
enum { RS_NONE, RS_TEX, RS_SLOT, RS_ID, RS_UBO };
typedef struct __attribute__((packed)) { uint8_t kind, desc, dim, flags; uint16_t binding, count; int16_t reg; uint8_t samp, fmt; int32_t off, samp_arg; } dmsv_bind;
_Static_assert(sizeof(dmsv_bind) == 20, "dmsv_bind");
typedef struct {
    uint64_t key; int stage;          // NVN stage: 0 vertex 1 fragment 2 geometry 3 tess control 4 tess eval 5 compute
    VkShaderModule mod;
    VkDescriptorSetLayout dsl; int set;   // the set this stage's bindings were moved to (vertex 0, fragment 1, geometry 2, compute 0)
    int nbind; const dmsv_bind *b;
    const uint8_t *c1; uint32_t c1_len; VkBuffer c1_buf; VkDeviceSize c1_off;
    uint32_t local[3];
    uint8_t dyn[96]; int ndyn;          // bindings of uniform buffers bound with a dynamic offset (gfx_vk_exec.c stage_set), and their count
    int xfb; uint32_t xfb_stride[4];    // a vertex shader that captures transform feedback (its SPIR-V has the Xfb execution mode), and each buffer's stride
    uint32_t *spv, nspv;                // its SPIR-V as the module was made from it: NULL once the module is made unless kept (gfx_vk_shader.c spv_kept_for_readers, DM_VK_SPV_KEEP=1), the variants then rebuild it; nspv its length either way
    struct { uint64_t um, im; uint32_t xo; int fd; VkShaderModule mod; uint64_t lm; } var[16]; int nvar;   // variants whose sampled bindings (bit = index in b) of an integer view read uint (um) / int (im) texels,
                                                                                    // that declare the vec4 outputs xo (bit = location) the next stage reads, and (fd) that write gl_FragDepth = gl_FragCoord.z
    int depth_repl;                     // a fragment shader that writes gl_FragDepth (DepthReplacing / DepthGreater / DepthLess)
    uint32_t in_locs, out_locs;         // the vec4 (or per-vertex vec4) inputs and outputs it declares, by location
    int setmemo;                        // gfx_vk_exec.c stage_set: 1 its set may be memoized per stage, 2 not (it reads handles or addresses from uniform buffer memory), 0 not known yet
    uint32_t prim_locs;                 // a mesh shader's per-primitive outputs (PerPrimitiveEXT), by location: a fragment shader drawn after it takes vk_shader_module's xo = prim_locs & its in_locs
    int dev_ok;                         // gfx_vk_exec.c mesh_need: 1 the device takes this task / mesh shader (SPIR-V version, subgroup operations, memory, outputs), 2 not, 0 not looked at yet
    int col0;                           // a fragment shader's colour 0 (VK_COL0_*): the alpha test reads a float one's alpha (gfx_vk_shader.c spv_alpha_test)
    uint64_t lod_ops;                   // its sampled bindings (bit = index in b) that take level numbers: query LOD / levels, explicit LOD, fetch levels, size by level (vk_lodshift_mask)
    int nfirst; struct { int32_t reg; uint32_t word, value; } first[4];   // a compute shader's first-invocation stores, taken out of it (shader_vk.py first_store): before each dispatch the
                                                                           // replay writes value at word of the storage buffer in slot reg (gfx_vk_exec.c first_stores)
    int ordered;                            // a compute shader that allocates through a storage-buffer atomic (shader_vk.py ordered_alloc): a small direct dispatch runs its workgroups in
                                            // order (gfx_vk_exec.c dispatch, DM_VK_ORDERED), as NVIDIA's GPUs start them
} vkshader_t;
enum { VK_COL0_NONE, VK_COL0_FLOAT, VK_COL0_INT, VK_COL0_OTHER };   // no output at location 0; a vec4 of 32-bit floats; integers; another float type
// NVN's alpha test (gfx_vk_shader.c spv_alpha_test, gfx_vk_exec.c prepare_draw): its function (NVN_ALPHA_FUNC_*) and reference (float) are each draw's push constants, at this offset of
// every graphics pipeline layout's fragment range (vk_pipeline_layout_gfx)
#define VK_ALPHA_PUSH_OFF 0
#define VK_ALPHA_PUSH_SIZE 8
VkBuffer vk_spill_buffer(VkDeviceSize size);   // RK_SPILL's buffer, at least size bytes (gfx_vk_shader.c)
vkshader_t *vk_shader(uint64_t key, int stage);   // NULL when the bundle has no such shader
VkShaderModule vk_shader_module(vkshader_t *s, uint64_t um, uint64_t im, uint32_t xo, int fd);   // the module, or its variant (integer views; extra outputs; fd: 1 fragment depth written, 2 dual source, 4 alpha test; a mesh shader's 8 payload made Private, 16 gl_Position in a block)
// LOD shifts (PARITY G1; gfx_vk_shader.c spv_lod_shift): a binding whose view's level 0 is its texture's level B > 0 (gfx_vk_exec.c tex_desc_for, g_lod_shift by stage and binding number) takes its
// level numbers through a variant that reads B as data, 4 bits a binding in a push constant word of its stage's (vk_lodshift_off: vertex, mesh and compute 8, fragment 12, geometry and task 16;
// every pipeline layout has those ranges while DM_VK_LODSHIFT is on, start-up, default on). lm: the variant's bindings (bit = index in b, at most 8: vk_lodshift_mask); its draw or dispatch pushes
// vk_lodshift_word for them. Tessellation stages have no range
int vk_lodshift_on(void);
uint32_t vk_lodshift_off(int stage);
uint64_t vk_lodshift_mask(const vkshader_t *s, const uint8_t shift[64]);   // shift: B by binding number (& 63); 0 when no binding of the shader both is shifted and takes level numbers
uint32_t vk_lodshift_word(const vkshader_t *s, uint64_t lm, const uint8_t shift[64]);
VkShaderModule vk_shader_module_lod(vkshader_t *s, uint64_t um, uint64_t im, uint32_t xo, int fd, uint64_t lm);   // vk_shader_module with the LOD-shift variant for lm too
VkPipelineLayout vk_pipeline_layout(VkDescriptorSetLayout sets[5], int nsets);       // compute: the set layouts only
VkPipelineLayout vk_pipeline_layout_gfx(VkDescriptorSetLayout sets[5], int nsets);   // graphics: the same, with the alpha test's push constant range (VK_ALPHA_PUSH_OFF)
void vk_shaders_open(const char *path);
VkBuffer vk_support_buffer(void);
VkBuffer vk_zero_buffer(void);

// ---- replay (gfx_vk_exec.c) ----
void vk_exec_reset(void);   // per submission
void vk_end_pass(void);     // close an open render pass

// ---- presentation (gfx_vk_present.c) ----
void vk_present_init(void);
uint64_t vk_present(const tex_t *t, double period, const int crop[4]);
// ---- GPU profile (gfx_vk_core.c "GPU profile", DM_VK_GPUPROF): a unit of a submission's main command buffers, a render pass or the stretch between two passes, as gfx_vk_exec.c hands it over ----
enum { VK_PU_PASS = 1, VK_PU_GAP = 2, VK_PU_END_SUB = 255 };   // kind; end_op VK_PU_END_SUB: the pass ended at the submission's end
typedef struct {
    uint32_t w, h, draws, fidx;      // a pass's size and draws; fidx: the unit's place in its frame (gfx_vk_core.c)
    uint8_t kind, ncol, end_op, pad; // end_op: the record op (gfx/nvn/nvn_obj.h) whose replay ended the pass
    uint32_t cfmt[4], dfmt;          // VkFormats: the first four colour attachments, the depth / stencil one
    uint16_t load, store;            // 2 bits an attachment (colour i at 2i, depth at 8, stencil at 10): 0 load, 1 clear, 2 don't care; store: 1 bit (colour i, depth 8, stencil 9), 1 = don't care
    uint16_t mesh, alpha, fdz, fdepth;   // in a pass: draws that are mesh tasks, alpha-tested, in the fragment-depth variant, whose fragment shader writes depth
    uint16_t disp, spill, first, copies, clears, bars, merged, touches, ubd;   // between passes: dispatches, spill parts, first stores, copies, clears, full barriers recorded and merged away,
                                                                              // LRZ touches, uniform landings (bars, merged and ubd count in passes too: the ones recorded where it began)
    uint16_t dstore, dload, dk_end, dk_used, dk_hist;   // a pass's NVN discards made store / load DONT_CARE, and the stores kept: the pass's end not certain, the attachment used again
                                                        // in the submission, its image not a transient by its history (gfx_vk_exec.c "discards")
} vk_prof_t;
int vk_prof_on(void);                   // this submission is timed
void vk_prof_unit(const vk_prof_t *u);  // the unit u ends here: a timestamp at the boundary
extern uint64_t g_bar_full, g_bar_merged;   // full barriers recorded, and merged into the one just before (vk_img_barrier_all)

