// Android port, Vulkan backend: the present's upscaling passes (gfx_vk_fsr.h; the shaders are shaders/fsr/fsr1.glsl, AMD FidelityFX FSR 1, MIT: shaders/fsr/LICENSE).
// A present draws the game's crop of its window texture into the swapchain image in one or two fragment passes:
//   fsr        EASU from the crop into an intermediate image of the picture's size (A2B10G10R10: RCAS amplifies the intermediate's rounding up to 2.7x in flat areas, an
//              8-bit one would show as ±1 step of noise in the sky; the same 4 bytes a texel), then RCAS from it into the swapchain image;
//   fsr, RCAS off   EASU straight into the swapchain image;
//   bilinear   one bilinear draw into the swapchain image (with DM_VK_RCAS=<stops>: bilinear into the intermediate, then RCAS);
// the last pass covers the whole swapchain image (black outside the picture: no clear of the whole image first, as the blit had) and its rendering scope stays open
// for the gamepad overlay (gfx_vk_present.c ov_render), so the overlay costs no pass, load or barrier of its own. Fragment and not compute passes: on Adreno a colour
// attachment keeps its framebuffer compression (UBWC), which the display reads too, where a storage image may not have it; an sRGB swapchain cannot be a storage image
// at all; and Qualcomm's own spatial upscaler (Snapdragon GSR) is a fragment shader. The swapchain's supported usages are logged (gfx_vk_present.c swap_make).
// Cost (estimated, measured on the device by the present's timer, DM_VK_FSR_TIME): EASU is 385 floating-point operations a pixel in SPIR-V (363 of them in 16 bits in
// the FP16 variant) and 12 texel fetches; RCAS about 32 operations and 5 fetches. At 2504x1408 (the Fold's inner screen, 3.53 Mpixels) that is 1.36 G operations, about
// 0.4 ms at the Adreno 840's rate (~1.8 T 32-bit instructions a second, FP16 twice that) if nothing waited, call it 0.6 ms; RCAS 0.1 ms; the outer screen's 1920x1080
// 60 % of that. Measured on an M4 through MoltenVK, eight presents in flight: 0.76 ms p50 at 2504x1408 (EASU 0.49), 0.84 in 32 bits, EASU alone 0.63, the bilinear pass 0.10. Memory: the intermediate written and read (14 MB each way before compression) where the blit's whole-image clear (22.6 MB) and the overlay's own pass go.
// Every vkCmd* here goes through the backend's wrappers (gfx_vk_dq.c: the two-stage replay's queue, the barrier merge's marks), as the present's blit did.
#include "gfx_vk.h"
#include "gfx_vk_fsr.h"
#include "fsr_vert.spv.h"
#include "fsr_easu_frag.spv.h"
#include "fsr_easu16_frag.spv.h"
#include "fsr_rcas_frag.spv.h"
#include "fsr_rcas16_frag.spv.h"
#include "fsr_blit_frag.spv.h"
#include <stdio.h>

enum { SLOTS = 8, NPIPE = 4 };   // (gfx_vk_present.c's present slots, g_pi % 8)
static struct {
    int init;                         // 1 made, -1 failed for good (the blit draws every picture)
    int fp16;                         // the shaders' 16-bit variants (shaderFloat16; DM_VK_FSR_FP16=0: the 32-bit ones)
    VkDescriptorSetLayout dsl; VkPipelineLayout layout; VkDescriptorPool pool; VkSampler samp;
    VkShaderModule vs, m_easu, m_rcas, m_blit;
    VkDescriptorSet src_set[SLOTS], mid_set[SLOTS]; VkImageView srcv[SLOTS];   // per present slot: the window image's view, made at each present and dropped at the slot's next
    struct { VkFormat fmt; VkPipeline easu, rcas, blit; } pp[NPIPE]; int npp;  // per colour format (the swapchain's, the intermediate's)
    VkImage mid; VkDeviceMemory midm; VkImageView midv; VkFormat midf; uint32_t mw, mh;   // the intermediate image
    int mode; float rcas; uint64_t polls;   // DM_VK_UPSCALE, DM_VK_RCAS (read every 30 presents)
    struct { int path, two, fp16, cw, ch, pw, ph, px, py, W, H, rot; float rcas; } said;   // the configuration last logged
} F = {.mode = VKFSR_FSR, .rcas = VKFSR_RCAS_AUTO};

static void fsr_switches(void) {
    if (F.polls++ % 30) return;
    const char *e = getenv("DM_VK_UPSCALE"); int m = vkfsr_parse_mode(e);
    if (m < 0) { VKRED_ONCE(120, "DM_VK_UPSCALE=%s: not fsr, bilinear or off; fsr", e); m = VKFSR_FSR; }
    F.mode = m; F.rcas = vkfsr_parse_rcas(getenv("DM_VK_RCAS"), m);
}
static VkFormat unorm_of(VkFormat f) {   // the window image read as its bytes: an sRGB format's UNORM twin
    switch (f) {
    case VK_FORMAT_R8G8B8A8_SRGB: return VK_FORMAT_R8G8B8A8_UNORM;
    case VK_FORMAT_B8G8R8A8_SRGB: return VK_FORMAT_B8G8R8A8_UNORM;
    case VK_FORMAT_A8B8G8R8_SRGB_PACK32: return VK_FORMAT_A8B8G8R8_UNORM_PACK32;
    default: return f;
    }
}
static int src_ok(const vkimg_t *im) {   // a window image the passes can read through a view in its UNORM twin: single-sampled colour, linear filtering (the bilinear pass)
    VkFormat vf = unorm_of(im->fmt);
    if (im->samples > 1 || im->aspect != VK_IMAGE_ASPECT_COLOR_BIT || !(im->usage & VK_IMAGE_USAGE_SAMPLED_BIT)) return 0;
    if (vf != im->fmt && (!(im->cflags & VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT) || (im->fixed_fmt && im->fmt2 != vf))) return 0;   // (gfx_vk_tex.c img_new: its format list)
    static VkFormat known[4]; static int ok[4], nk;
    for (int i = 0; i < nk; i++) if (known[i] == vf) return ok[i];
    VkFormatProperties fp; vkGetPhysicalDeviceFormatProperties(V.pd, vf, &fp);
    int r = (fp.optimalTilingFeatures & (VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT | VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT)) == (VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT | VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT);
    if (!r) VKLOG("[RED] upscale: window format %d cannot be sampled with a filter: the blit draws it", vf);
    if (nk < 4) { known[nk] = vf; ok[nk++] = r; }
    return r;
}
int vk_fsr_path(const vkimg_t *im, const int crop[4], int W, int H) {
    fsr_switches();
    if (F.mode == VKFSR_OFF || !im || F.init < 0 || W <= 0 || H <= 0) return VKFSR_OFF;
    if (!V.has_dynrender) { VKRED_ONCE(121, "upscale: no dynamic rendering: the blit draws the picture (DM_VK_UPSCALE has no effect)"); return VKFSR_OFF; }
    if (!src_ok(im)) return VKFSR_OFF;
    int c[4]; vkfsr_crop(crop, (int)im->w, (int)im->h, c); vkfsr_rect pic = vkfsr_fit(c[2], c[3], W, H);
    if (pic.w <= c[2] && pic.h <= c[3]) return VKFSR_BILINEAR;   // not made larger: EASU has nothing to do
    return F.mode;
}

static VkShaderModule fsr_module(const uint32_t *code, size_t n) {
    VkShaderModuleCreateInfo ci = {VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, NULL, 0, n, code}; VkShaderModule m = VK_NULL_HANDLE;
    return vkCreateShaderModule(V.dev, &ci, NULL, &m) ? VK_NULL_HANDLE : m;
}
static int fsr_init(void) {
    if (F.init) return F.init > 0;
    F.init = -1;
    // 16-bit arithmetic where the device has shaderFloat16 (Vulkan 1.2's feature, which vk_init enables with every other the device has: gfx_vk_core.c); the Adreno 840 has it
    if (V.props.apiVersion >= VK_API_VERSION_1_2) {
        VkPhysicalDeviceShaderFloat16Int8Features h = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES};
        VkPhysicalDeviceFeatures2 f2 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, &h}; vkGetPhysicalDeviceFeatures2(V.pd, &f2);
        const char *e = getenv("DM_VK_FSR_FP16"); F.fp16 = h.shaderFloat16 && !(e && !atoi(e));
    }
    VkDescriptorSetLayoutBinding b = {0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, NULL};
    VkDescriptorSetLayoutCreateInfo li = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, NULL, 0, 1, &b};
    if (vkCreateDescriptorSetLayout(V.dev, &li, NULL, &F.dsl)) goto fail;
    VkPushConstantRange pr = {VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(vkfsr_push)};
    VkPipelineLayoutCreateInfo pl = {VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, NULL, 0, 1, &F.dsl, 1, &pr};
    if (vkCreatePipelineLayout(V.dev, &pl, NULL, &F.layout)) goto fail;
    VkDescriptorPoolSize ps = {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 2 * SLOTS};
    VkDescriptorPoolCreateInfo pi = {VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO, NULL, 0, 2 * SLOTS, 1, &ps};
    if (vkCreateDescriptorPool(V.dev, &pi, NULL, &F.pool)) goto fail;
    VkDescriptorSetLayout ls[2 * SLOTS]; for (int i = 0; i < 2 * SLOTS; i++) ls[i] = F.dsl;
    VkDescriptorSet sets[2 * SLOTS]; VkDescriptorSetAllocateInfo ai = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, NULL, F.pool, 2 * SLOTS, ls};
    if (vkAllocateDescriptorSets(V.dev, &ai, sets)) goto fail;
    for (int i = 0; i < SLOTS; i++) { F.src_set[i] = sets[i]; F.mid_set[i] = sets[SLOTS + i]; }
    VkSamplerCreateInfo si = {VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO, NULL, 0, VK_FILTER_LINEAR, VK_FILTER_LINEAR, VK_SAMPLER_MIPMAP_MODE_NEAREST,
        VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE, VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE, VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE, 0, 0, 1, 0, VK_COMPARE_OP_ALWAYS, 0, 0};
    if (vkCreateSampler(V.dev, &si, NULL, &F.samp)) goto fail;
    F.vs = fsr_module(fsr_vert_spv, sizeof fsr_vert_spv);
    F.m_easu = F.fp16 ? fsr_module(fsr_easu16_frag_spv, sizeof fsr_easu16_frag_spv) : fsr_module(fsr_easu_frag_spv, sizeof fsr_easu_frag_spv);
    F.m_rcas = F.fp16 ? fsr_module(fsr_rcas16_frag_spv, sizeof fsr_rcas16_frag_spv) : fsr_module(fsr_rcas_frag_spv, sizeof fsr_rcas_frag_spv);
    F.m_blit = fsr_module(fsr_blit_frag_spv, sizeof fsr_blit_frag_spv);
    if (!F.vs || !F.m_easu || !F.m_rcas || !F.m_blit) goto fail;
    // the intermediate's format: 10 bits a channel where the device draws into and samples it (a mandatory format), else RGBA8
    VkFormatProperties fp; vkGetPhysicalDeviceFormatProperties(V.pd, VK_FORMAT_A2B10G10R10_UNORM_PACK32, &fp);
    VkFormatFeatureFlags need = VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT | VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT;
    F.midf = (fp.optimalTilingFeatures & need) == need ? VK_FORMAT_A2B10G10R10_UNORM_PACK32 : VK_FORMAT_R8G8B8A8_UNORM;
    F.init = 1;
    return 1;
fail:
    VKLOG("[RED] upscale: its pipeline objects could not be made: the blit draws every picture");
    return 0;
}
static VkPipeline fsr_pipe1(VkShaderModule fs, VkFormat fmt) {
    VkPipelineShaderStageCreateInfo ss[2] = {{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, NULL, 0, VK_SHADER_STAGE_VERTEX_BIT, F.vs, "main", NULL},
                                            {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, NULL, 0, VK_SHADER_STAGE_FRAGMENT_BIT, fs, "main", NULL}};
    VkPipelineVertexInputStateCreateInfo vi = {VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
    VkPipelineInputAssemblyStateCreateInfo ia = {VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO, NULL, 0, VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST, 0};
    VkPipelineViewportStateCreateInfo vs = {VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO, NULL, 0, 1, NULL, 1, NULL};
    VkPipelineRasterizationStateCreateInfo rs = {VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO, NULL, 0, 0, 0, VK_POLYGON_MODE_FILL, VK_CULL_MODE_NONE, VK_FRONT_FACE_COUNTER_CLOCKWISE, 0, 0, 0, 0, 1};
    VkPipelineMultisampleStateCreateInfo ms = {VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO, NULL, 0, VK_SAMPLE_COUNT_1_BIT};
    VkPipelineColorBlendAttachmentState ba = {0, VK_BLEND_FACTOR_ONE, VK_BLEND_FACTOR_ZERO, VK_BLEND_OP_ADD, VK_BLEND_FACTOR_ONE, VK_BLEND_FACTOR_ZERO, VK_BLEND_OP_ADD, 0xf};
    VkPipelineColorBlendStateCreateInfo cbs = {VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO, NULL, 0, 0, 0, 1, &ba};
    VkDynamicState dyn[] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR}; VkPipelineDynamicStateCreateInfo dy = {VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO, NULL, 0, 2, dyn};
    VkPipelineRenderingCreateInfo rci = {VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO, NULL, 0, 1, &fmt};
    VkGraphicsPipelineCreateInfo ci = {VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO, &rci, 0, 2, ss, &vi, &ia, NULL, &vs, &rs, &ms, NULL, &cbs, &dy, F.layout, VK_NULL_HANDLE};
    VkPipeline p = VK_NULL_HANDLE; VkResult r = vkCreateGraphicsPipelines(V.dev, V.pcache, 1, &ci, NULL, &p);
    if (r) { VKLOG("[RED] upscale: pipeline for format %d failed: %d", fmt, r); return VK_NULL_HANDLE; }
    return p;
}
static int fsr_pipes(VkFormat fmt, int *slot) {   // the three pipelines for a colour format, made at its first use
    for (int i = 0; i < F.npp; i++) if (F.pp[i].fmt == fmt) { *slot = i; return F.pp[i].easu && F.pp[i].rcas && F.pp[i].blit; }
    if (F.npp == NPIPE) { VKRED_ONCE(122, "upscale: more than %d target formats", NPIPE); return 0; }
    int i = F.npp++; F.pp[i].fmt = fmt; *slot = i;
    F.pp[i].easu = fsr_pipe1(F.m_easu, fmt); F.pp[i].rcas = fsr_pipe1(F.m_rcas, fmt); F.pp[i].blit = fsr_pipe1(F.m_blit, fmt);
    return F.pp[i].easu && F.pp[i].rcas && F.pp[i].blit;
}
static void fsr_mid_drop(void) {
    if (F.midv) vkDestroyImageView(V.dev, F.midv, NULL);
    if (F.mid) vkDestroyImage(V.dev, F.mid, NULL);
    if (F.midm) vkFreeMemory(V.dev, F.midm, NULL);
    F.midv = VK_NULL_HANDLE; F.mid = VK_NULL_HANDLE; F.midm = VK_NULL_HANDLE; F.mw = F.mh = 0;
}
static int fsr_mid(uint32_t w, uint32_t h, const VkFence *busy, uint32_t nbusy) {   // the intermediate at the picture's size: made again when that changes (a window or crop of another shape)
    if (F.mid && F.mw == w && F.mh == h) return 1;
    if (F.mid) {   // the presents that read it have run (a slot whose submission failed keeps its fence unsignalled: 2 s at most, then this present goes without RCAS)
        if (nbusy && vkWaitForFences(V.dev, nbusy, busy, VK_TRUE, 2000000000ull) != VK_SUCCESS) { VKRED_ONCE(126, "upscale: the presents before did not complete: the intermediate is kept"); return 0; }
        fsr_mid_drop();
    }
    VkImageCreateInfo ic = {VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, NULL, 0, VK_IMAGE_TYPE_2D, F.midf, {w, h, 1}, 1, 1, VK_SAMPLE_COUNT_1_BIT, VK_IMAGE_TILING_OPTIMAL,
        VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, VK_SHARING_MODE_EXCLUSIVE, 0, NULL, VK_IMAGE_LAYOUT_UNDEFINED};   // (no other usage, no MUTABLE_FORMAT: UBWC)
    if (vkCreateImage(V.dev, &ic, NULL, &F.mid)) { F.mid = VK_NULL_HANDLE; return 0; }
    VkMemoryRequirements rq; vkGetImageMemoryRequirements(V.dev, F.mid, &rq);
    VkMemoryAllocateInfo ai = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, NULL, rq.size, vk_mem_type(rq.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 0)};
    if (vkAllocateMemory(V.dev, &ai, NULL, &F.midm) || vkBindImageMemory(V.dev, F.mid, F.midm, 0)) { fsr_mid_drop(); return 0; }
    VkImageViewCreateInfo vi = {VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, NULL, 0, F.mid, VK_IMAGE_VIEW_TYPE_2D, F.midf, {0}, {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
    if (vkCreateImageView(V.dev, &vi, NULL, &F.midv)) { F.midv = VK_NULL_HANDLE; fsr_mid_drop(); return 0; }
    F.mw = w; F.mh = h;
    VKLOG("upscale: intermediate image %ux%u format %d, %.1f MB", w, h, F.midf, rq.size / 1048576.0);
    return 1;
}
static void fsr_img_barrier(VkCommandBuffer cb, VkImage img, VkImageLayout from, VkImageLayout to, VkPipelineStageFlags ss, VkAccessFlags sa, VkPipelineStageFlags ds, VkAccessFlags da) {
    VkImageMemoryBarrier b = {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER, NULL, sa, da, from, to, VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED, img, {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
    vkCmdPipelineBarrier(cb, ss, ds, 0, 0, NULL, 0, NULL, 1, &b);
}
static void fsr_set(VkDescriptorSet s, VkImageView v, VkImageLayout l) {
    VkDescriptorImageInfo ii = {F.samp, v, l}; VkWriteDescriptorSet w = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, NULL, s, 0, 0, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &ii};
    vkUpdateDescriptorSets(V.dev, 1, &w, 0, NULL);
}
static void fsr_draw(VkCommandBuffer cb, VkImageView view, uint32_t w, uint32_t h, VkPipeline p, VkDescriptorSet s, const vkfsr_push *pc) {   // one triangle over a w x h target
    VkRenderingAttachmentInfo ca = {VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO, NULL, view, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, 0, NULL, VK_IMAGE_LAYOUT_UNDEFINED,
                                    VK_ATTACHMENT_LOAD_OP_DONT_CARE, VK_ATTACHMENT_STORE_OP_STORE};   // (every pixel is written)
    VkRenderingInfo ri = {VK_STRUCTURE_TYPE_RENDERING_INFO, NULL, 0, {{0, 0}, {w, h}}, 1, 0, 1, &ca, NULL, NULL};
    vkCmdBeginRendering(cb, &ri);
    VkViewport vp = {0, 0, (float)w, (float)h, 0, 1}; VkRect2D sc = {{0, 0}, {w, h}};
    vkCmdSetViewport(cb, 0, 1, &vp); vkCmdSetScissor(cb, 0, 1, &sc);
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, p);
    vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, F.layout, 0, 1, &s, 0, NULL);
    vkCmdPushConstants(cb, F.layout, VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof *pc, pc);
    vkCmdDraw(cb, 3, 1, 0, 0);
}
static void fsr_time_mid(VkCommandBuffer cb, uint32_t slot);
static const char *const PATH_NAME[3] = {"blit", "bilinear", "fsr"};
int vk_fsr_begin(VkCommandBuffer cb, uint32_t slot, int path, const VkFence *busy, uint32_t nbusy, vkimg_t *im, const int crop[4], int bottom_up,
                 VkImage target, VkImageView view, VkFormat fmt, int W, int H, int rot) {
    if (path <= VKFSR_OFF || path > VKFSR_FSR || !im || !fsr_init()) return 0;
    slot %= SLOTS;
    int c[4]; vkfsr_crop(crop, (int)im->w, (int)im->h, c); vkfsr_rect pic = vkfsr_fit(c[2], c[3], W, H);
    float rcas = F.rcas; int two = rcas >= 0;   // a first pass into the intermediate, then RCAS
    int pw, ph; vkfsr_target_size(rot, W, H, &pw, &ph);
    int ti, mi = -1; if (!fsr_pipes(fmt, &ti) || (two && !fsr_pipes(F.midf, &mi))) { VKRED_ONCE(123, "upscale: no pipelines: the blit draws the picture"); return 0; }
    if (F.srcv[slot]) { vkDestroyImageView(V.dev, F.srcv[slot], NULL); F.srcv[slot] = VK_NULL_HANDLE; }   // (the slot's last present has completed)
    VkImageViewCreateInfo vi = {VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, NULL, 0, im->img, VK_IMAGE_VIEW_TYPE_2D, unorm_of(im->fmt), {0}, {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
    if (vkCreateImageView(V.dev, &vi, NULL, &F.srcv[slot])) { F.srcv[slot] = VK_NULL_HANDLE; VKRED_ONCE(124, "upscale: no view of the window image: the blit draws it"); return 0; }
    if (two && !fsr_mid((uint32_t)pic.w, (uint32_t)pic.h, busy, nbusy)) { VKRED_ONCE(125, "upscale: no intermediate image: no RCAS"); two = 0; }
    fsr_set(F.src_set[slot], F.srcv[slot], VK_IMAGE_LAYOUT_GENERAL);   // (every image the replay draws stays in GENERAL)
    int easu = path == VKFSR_FSR;
    if (memcmp(&F.said, &(typeof(F.said)){path, two, F.fp16, c[2], c[3], pic.w, pic.h, pic.x, pic.y, W, H, rot, two ? rcas : -1}, sizeof F.said)) {
        F.said = (typeof(F.said)){path, two, F.fp16, c[2], c[3], pic.w, pic.h, pic.x, pic.y, W, H, rot, two ? rcas : -1};
        char r[48]; if (two) snprintf(r, sizeof r, " + RCAS %.2f stops", rcas); else r[0] = 0;
        VKLOG("upscale: %s (%s%s%s), crop %dx%d at %d,%d → %dx%d at %d,%d of the %dx%d window%s", PATH_NAME[path], easu ? "EASU" : "bilinear", r, easu ? (F.fp16 ? ", 16-bit" : ", 32-bit") : "",
              c[2], c[3], c[0], c[1], pic.w, pic.h, pic.x, pic.y, W, H, rot ? (rot == VKFSR_ROT90 ? ", turned 90" : rot == VKFSR_ROT180 ? ", turned 180" : ", turned 270") : "");
    }
    vkfsr_push pc; memset(&pc, 0, sizeof pc);
    if (two) {   // the first pass: the picture, unturned, into the intermediate (its last reader, the slot before's RCAS, came earlier on the queue)
        fsr_set(F.mid_set[slot], F.midv, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        vkfsr_xform_id(pic.w, pic.h, &pc); vkfsr_map(c, pic, bottom_up, &pc);
        pc.k[0] = 1.0f / (float)im->w; pc.k[1] = 1.0f / (float)im->h;
        fsr_img_barrier(cb, F.mid, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0,
                        VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT);
        fsr_draw(cb, F.midv, (uint32_t)pic.w, (uint32_t)pic.h, easu ? F.pp[mi].easu : F.pp[mi].blit, F.src_set[slot], &pc);
        vkCmdEndRendering(cb);
        fsr_img_barrier(cb, F.mid, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                        VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT);
        fsr_time_mid(cb, slot);
    }
    // the last pass: the whole swapchain image (black around the picture), turned as the swapchain is; its scope stays open for the overlay
    memset(&pc, 0, sizeof pc); vkfsr_xform(rot, W, H, pic, &pc);
    if (two) { pc.clip[2] = pic.w - 1; pc.clip[3] = pic.h - 1; pc.k[0] = vkfsr_rcas_k(rcas); }
    else { vkfsr_map(c, pic, bottom_up, &pc); pc.k[0] = 1.0f / (float)im->w; pc.k[1] = 1.0f / (float)im->h; }
    fsr_img_barrier(cb, target, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, 0,
                    VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT);   // (after the acquire's semaphore, waited at this stage)
    fsr_draw(cb, view, (uint32_t)pw, (uint32_t)ph, two ? F.pp[ti].rcas : easu ? F.pp[ti].easu : F.pp[ti].blit, two ? F.mid_set[slot] : F.src_set[slot], &pc);
    return 1;
}
void vk_fsr_end(VkCommandBuffer cb) { vkCmdEndRendering(cb); }

// ---- the present's GPU time (DM_VK_FSR_TIME, start-up, default on): three timestamps a present (after its first barrier, after the intermediate pass, after the picture and
// the overlay), read when the slot comes round, the median and 90th percentile of the last 600 presents of each path logged. Outside any render pass, so they do not
// disturb Adreno's binning (DM_VK_PASS_TIMES' caveat is about stamps inside passes). What it shows: the present's own GPU work, the acquire's wait (at the last pass's
// stage) included when the display holds the image back
enum { TQ = 3, TN = 600 };
static struct { int init; VkQueryPool qp; double tick_ns; uint64_t mask; int pend[SLOTS], mid[SLOTS], path[SLOTS]; float tot[3][TN], first[3][TN]; int n[3], nf[3]; uint64_t seen; } T;
static int time_init(void) {
    if (T.init) return T.init > 0;
    T.init = -1;
    const char *e = getenv("DM_VK_FSR_TIME"); if (e && !atoi(e)) return 0;
    uint32_t nq = 0; vkGetPhysicalDeviceQueueFamilyProperties(V.pd, &nq, NULL); VkQueueFamilyProperties qp[16]; if (nq > 16) nq = 16; vkGetPhysicalDeviceQueueFamilyProperties(V.pd, &nq, qp);
    uint32_t bits = V.qf < nq ? qp[V.qf].timestampValidBits : 0;
    if (!bits || V.props.limits.timestampPeriod <= 0) { VKLOG("upscale: the queue has no timestamps: no present GPU times"); return 0; }
    VkQueryPoolCreateInfo ci = {VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO, NULL, 0, VK_QUERY_TYPE_TIMESTAMP, SLOTS * TQ, 0};
    if (vkCreateQueryPool(V.dev, &ci, NULL, &T.qp)) return 0;
    T.tick_ns = V.props.limits.timestampPeriod; T.mask = bits >= 64 ? ~0ull : (1ull << bits) - 1;
    T.init = 1; return 1;
}
void vk_fsr_time_start(VkCommandBuffer cb, uint32_t slot) {
    slot %= SLOTS; T.pend[slot] = 0; T.mid[slot] = 0;
    if (!time_init()) return;
    vkCmdResetQueryPool(cb, T.qp, slot * TQ, TQ);
    vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, T.qp, slot * TQ);   // (after the present's full barrier: when the frame's own work is done)
}
static void fsr_time_mid(VkCommandBuffer cb, uint32_t slot) {
    if (T.init <= 0) return;
    vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, T.qp, slot * TQ + 1); T.mid[slot] = 1;
}
void vk_fsr_time_end(VkCommandBuffer cb, uint32_t slot) {   // the caller then submits; vk_fsr_time_collect reads the slot only if that submission was made (T.pend, set below)
    slot %= SLOTS;
    if (T.init <= 0) return;
    if (!T.mid[slot]) vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, T.qp, slot * TQ + 1);   // (one pass: the stamp in the middle is the last's twin)
    vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, T.qp, slot * TQ + 2);
}
void vk_fsr_time_submitted(uint32_t slot, int path) { slot %= SLOTS; if (T.init > 0) { T.pend[slot] = 1; T.path[slot] = path; } }
static int fcmp(const void *a, const void *b) { float x = *(const float *)a, y = *(const float *)b; return x < y ? -1 : x > y; }
static float pct(float *v, int n, float q) { qsort(v, (size_t)n, sizeof *v, fcmp); int i = (int)(q * (n - 1) + 0.5f); return v[i < 0 ? 0 : i >= n ? n - 1 : i]; }
void vk_fsr_time_collect(uint32_t slot) {
    slot %= SLOTS;
    if (T.init <= 0 || !T.pend[slot]) return;
    T.pend[slot] = 0; uint64_t t[TQ];
    if (vkGetQueryPoolResults(V.dev, T.qp, slot * TQ, TQ, sizeof t, t, sizeof t[0], VK_QUERY_RESULT_64_BIT) != VK_SUCCESS) return;
    int p = T.path[slot] < 0 || T.path[slot] > 2 ? 0 : T.path[slot];
    double tot = (double)((t[2] - t[0]) & T.mask) * T.tick_ns / 1e6, first = (double)((t[1] - t[0]) & T.mask) * T.tick_ns / 1e6;
    if (tot >= 0 && tot < 1000 && T.n[p] < TN) T.tot[p][T.n[p]++] = (float)tot;
    if (T.mid[slot] && first >= 0 && first < 1000 && T.nf[p] < TN) T.first[p][T.nf[p]++] = (float)first;
    if (++T.seen % TN) return;
    for (int i = 0; i < 3; i++) if (T.n[i]) {
        char f[64] = ""; if (T.nf[i] > 0) snprintf(f, sizeof f, "; first pass p50 %.2f ms", pct(T.first[i], T.nf[i], 0.5f));
        VKLOG("[perf] present GPU (%s): p50 %.2f ms, p90 %.2f, max %.2f over %d presents%s", PATH_NAME[i], pct(T.tot[i], T.n[i], 0.5f), pct(T.tot[i], T.n[i], 0.9f), pct(T.tot[i], T.n[i], 1.0f), T.n[i], f);
        T.n[i] = T.nf[i] = 0;
    }
}

// ---- the swap of a game's own upscaler (DM_VK_NDSR, live: fsr, the default, or game) ----
// Mario Kart World, Donkey Kong Bananza and Splatoon Raiders upscale their 3D picture with agl's NDSR network: one compute dispatch of one of the shaders the shader map found in
// agl_ndsr.pck (DM_GAME_UPSCALER_KEYS, game.py: Mario Kart World's 30), reading the scene colour (one texture) and writing the output picture (one image) between its rectangles
// (vkndsr_rects), after which the game draws its HUD over it. The network is written for the Switch 2's tensor cores: on this backend its HMMA.16816 products run as
// a helper called 163-322 times a thread (dmsh_hmma16816_f16, 1,220 SPIR-V lines of scratch loads and FMAs), its LDSM / MOVM gathers as 289-433 subgroup shuffles; another
// project measured the same family (Donkey Kong Bananza's, 192-240 calls a thread) at 28-40 ms a frame on this Adreno 840, 3.7-4.6 ms after rewriting that lowering. Each
// variant's pipeline (430-814 KB of SPIR-V) is also made on the replay thread at its first dispatch (Mario Kart World's 727 compute shaders are past compute_warm's cap of 256).
// With the swap the dispatch is drawn by FSR 1's EASU (fsr1.glsl, FSR_CS; 16-bit where the device has it) from the input rectangle into the output rectangle, in the variant's
// colour (vkndsr_shape, from its code: the taps sRGB-encoded as they are read when its input is linear, as the network encodes it; its output encoded, or decoded and clamped
// as the network does); no RCAS here, the present's (DM_VK_RCAS) sharpens the picture once, at the end of the chain. An indirect dispatch, a variant of another colour shape,
// rectangles outside their textures, or a dispatch that does not make the picture larger (the network at 1:1: not an upscale, its job is not EASU's) run the game's shader.
// Why fsr by default: the look changes (AMD's spatial upscaler instead of the game's network; upstream offers MetalFX the same way as the player's choice), but at the cost
// measured on the family and the Fold's frame of 35-38 ms of GPU the network's tens of ms outweigh it; this pass took 0.30-0.41 ms on an M4 through MoltenVK (960x540 → 1280x720,
// its barriers and the taps' encoding included; the GPU's output within 1.2 % of the CPU reference's, fsr1_ref.c ndsr). Whether the game runs the network
// in a race on this port is the first line to read (`NDSR swap: first dispatch …` and the counts every 1800); DM_VK_NDSR=game (live) gives the network back for an A/B
#include "fsr_ndsr_comp.spv.h"
#include "fsr_ndsr16_comp.spv.h"
void vk_dsl_register(VkDescriptorSetLayout l, const VkDescriptorSetLayoutBinding *b, uint32_t n);   // gfx_vk_core.c: what one set of a layout takes (the submission's pools)
static struct { int init, fp16, on; uint64_t polls; VkDescriptorSetLayout dsl; VkPipelineLayout layout; VkPipeline pipe; VkSampler samp; uint64_t keys[64]; int nkeys;
                struct { uint64_t key; int shape; } sh[64]; int nsh; uint64_t n, kept_shape, kept_rect, kept_same; } N = {.on = 1};
static const char *const SHAPE_NAME[5] = {"neither of the colour shapes the swap draws: the network runs", "linear input, encoded output (clamped to 0-1)",
                                          "linear input and output (clamped to 0-1)", "linear input and output (at most 128)", "encoded input, linear output (clamped to 0-1)"};
static int u64cmp(const void *a, const void *b) { uint64_t x = *(const uint64_t *)a, y = *(const uint64_t *)b; return x < y ? -1 : x > y; }
static int ndsr_named(uint64_t key) {
    static int parsed;
    if (!parsed) { parsed = 1;
#ifdef DM_GAME_UPSCALER_KEYS
        for (const char *k = DM_GAME_UPSCALER_KEYS; *k && N.nkeys < 64; k += k[16] == ',' ? 17 : 16) { char b[17]; memcpy(b, k, 16); b[16] = 0; N.keys[N.nkeys++] = strtoull(b, NULL, 16); if (!k[16]) break; }
#endif
        qsort(N.keys, (size_t)N.nkeys, sizeof *N.keys, u64cmp);
    }
    return N.nkeys && bsearch(&key, N.keys, (size_t)N.nkeys, sizeof *N.keys, u64cmp) != NULL;
}
int vk_ndsr_wanted(uint64_t key) {   // (every dispatch asks: a binary search, the switch read every 64 asks)
    if (N.polls++ % 64 == 0) {
        const char *e = getenv("DM_VK_NDSR"); int on = !(e && !strcmp(e, "game"));
        if (e && *e && strcmp(e, "game") && strcmp(e, "fsr")) VKRED_ONCE(127, "DM_VK_NDSR=%s: fsr or game; fsr", e);
        if (on != N.on && N.nkeys) VKLOG("NDSR swap: %s", on ? "on (DM_VK_NDSR=fsr): the game's upscaler shaders are drawn by EASU" : "off (DM_VK_NDSR=game): the game's network draws");
        N.on = on;
    }
    return N.on && ndsr_named(key);
}
static int ndsr_init(void) {
    if (N.init) return N.init > 0;
    N.init = -1;
    if (V.props.apiVersion >= VK_API_VERSION_1_2) {
        VkPhysicalDeviceShaderFloat16Int8Features h = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES};
        VkPhysicalDeviceFeatures2 f2 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, &h}; vkGetPhysicalDeviceFeatures2(V.pd, &f2);
        const char *e = getenv("DM_VK_FSR_FP16"); N.fp16 = h.shaderFloat16 && !(e && !atoi(e));
    }
    VkPhysicalDeviceFeatures f; VkPhysicalDeviceFeatures2 f2 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2}; vkGetPhysicalDeviceFeatures2(V.pd, &f2); f = f2.features;
    if (!f.shaderStorageImageWriteWithoutFormat) { VKLOG("[RED] NDSR swap: the device writes no storage image without a format: the network draws"); return 0; }
    VkDescriptorSetLayoutBinding b[2] = {{0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL}, {1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL}};
    VkDescriptorSetLayoutCreateInfo li = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, NULL, 0, 2, b};
    if (vkCreateDescriptorSetLayout(V.dev, &li, NULL, &N.dsl)) return 0;
    vk_dsl_register(N.dsl, b, 2);
    VkPushConstantRange pr = {VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(vkfsr_push)};
    VkPipelineLayoutCreateInfo pl = {VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, NULL, 0, 1, &N.dsl, 1, &pr};
    if (vkCreatePipelineLayout(V.dev, &pl, NULL, &N.layout)) return 0;
    VkSamplerCreateInfo si = {VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO, NULL, 0, VK_FILTER_NEAREST, VK_FILTER_NEAREST, VK_SAMPLER_MIPMAP_MODE_NEAREST,
        VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE, VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE, VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE, 0, 0, 1, 0, VK_COMPARE_OP_ALWAYS, 0, 0};   // (texelFetch only)
    if (vkCreateSampler(V.dev, &si, NULL, &N.samp)) return 0;
    VkShaderModule m = N.fp16 ? fsr_module(fsr_ndsr16_comp_spv, sizeof fsr_ndsr16_comp_spv) : fsr_module(fsr_ndsr_comp_spv, sizeof fsr_ndsr_comp_spv);
    if (!m) return 0;
    VkComputePipelineCreateInfo ci = {VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO, NULL, 0, {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, NULL, 0, VK_SHADER_STAGE_COMPUTE_BIT, m, "main", NULL},
                                      N.layout, VK_NULL_HANDLE, -1};
    VkResult r = vkCreateComputePipelines(V.dev, V.pcache, 1, &ci, NULL, &N.pipe); vkDestroyShaderModule(V.dev, m, NULL);
    if (r) { VKLOG("[RED] NDSR swap: its pipeline failed (%d): the network draws", r); return 0; }
    N.init = 1; return 1;
}
static int ndsr_shape_of(const vkshader_t *cs) {
    for (int i = 0; i < N.nsh; i++) if (N.sh[i].key == cs->key) return N.sh[i].shape;
    int s = cs->spv && cs->nspv ? vkndsr_shape(cs->spv, cs->nspv) : VKNDSR_KEEP;
    if (N.nsh < 64) { N.sh[N.nsh].key = cs->key; N.sh[N.nsh++].shape = s; }
    VKLOG("NDSR swap: upscaler shader %016llx: %s", (unsigned long long)cs->key, SHAPE_NAME[s]);
    return s;
}
int vk_ndsr_swap(VkCommandBuffer cb, const vkshader_t *cs, const vkfsr_tex *in, const vkfsr_tex *out, const uint32_t push[8]) {
    int shape = ndsr_shape_of(cs);
    if (shape == VKNDSR_KEEP) { N.kept_shape++; return 0; }
    int ir[4], orr[4];
    if (!in->view || !out->view || !vkndsr_rects(push, in->w, in->h, out->w, out->h, ir, orr)) {
        if (!N.kept_rect++) VKLOG("[RED] NDSR swap: %016llx's rectangles (push %u %u %u %u / %u %u %u %u) do not fit its textures (%ux%u → %ux%u): the network draws", (unsigned long long)cs->key,
                                   push[0], push[1], push[2], push[3], push[4], push[5], push[6], push[7], in->w, in->h, out->w, out->h);
        return 0;
    }
    if (orr[2] < ir[2] || orr[3] < ir[3] || (orr[2] == ir[2] && orr[3] == ir[3])) {   // not made larger: the network's other job, not EASU's
        if (!N.kept_same++) VKLOG("NDSR swap: %016llx at %dx%d → %dx%d is no upscale: the network draws it (counted)", (unsigned long long)cs->key, ir[2], ir[3], orr[2], orr[3]);
        return 0;
    }
    if (!ndsr_init()) { N.kept_shape++; return 0; }
    VkDescriptorSet set = vk_alloc_set(N.dsl);
    VkDescriptorImageInfo ii[2] = {{N.samp, in->view, VK_IMAGE_LAYOUT_GENERAL}, {VK_NULL_HANDLE, out->view, VK_IMAGE_LAYOUT_GENERAL}};   // (every image the replay draws stays in GENERAL)
    VkWriteDescriptorSet wr[2] = {{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, NULL, set, 0, 0, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &ii[0]},
                                  {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, NULL, set, 1, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &ii[1]}};
    vkUpdateDescriptorSets(V.dev, 2, wr, 0, NULL);
    vkfsr_push pc; memset(&pc, 0, sizeof pc);
    float sx = (float)ir[2] / (float)orr[2], sy = (float)ir[3] / (float)orr[3];   // (as vkfsr_map: an output pixel's centre in the input's texels)
    pc.xb[0] = orr[0]; pc.xb[1] = orr[1]; pc.xb[2] = orr[2]; pc.xb[3] = orr[3];
    pc.map[0] = sx; pc.map[1] = sy; pc.map[2] = 0.5f * sx - 0.5f + (float)ir[0]; pc.map[3] = 0.5f * sy - 0.5f + (float)ir[1];
    pc.clip[0] = ir[0]; pc.clip[1] = ir[1]; pc.clip[2] = ir[0] + ir[2] - 1; pc.clip[3] = ir[1] + ir[3] - 1;
    pc.k[1] = shape != VKNDSR_SRGB_LIN1; pc.k[2] = shape == VKNDSR_ENC ? 0 : shape == VKNDSR_LIN128 ? 2 : 1;
    vk_img_barrier_all(cb);   // (as the replay's dispatch: after everything before, before everything after)
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, N.pipe);
    vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, N.layout, 0, 1, &set, 0, NULL);
    vkCmdPushConstants(cb, N.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof pc, &pc);
    vkCmdDispatch(cb, ((uint32_t)orr[2] + 7) / 8, ((uint32_t)orr[3] + 7) / 8, 1);
    vk_img_barrier_all(cb);
    if (!N.n++) VKLOG("NDSR swap: first dispatch drawn by EASU (%s): %016llx, %dx%d at %d,%d of %ux%u → %dx%d at %d,%d of %ux%u, %s", N.fp16 ? "16-bit" : "32-bit", (unsigned long long)cs->key,
                      ir[2], ir[3], ir[0], ir[1], in->w, in->h, orr[2], orr[3], orr[0], orr[1], out->w, out->h, SHAPE_NAME[shape]);
    if (N.n % 1800 == 0) VKLOG("NDSR swap: %llu dispatches drawn by EASU; left to the network: %llu (colour shape), %llu (rectangles), %llu (not an upscale)", (unsigned long long)N.n,
                               (unsigned long long)N.kept_shape, (unsigned long long)N.kept_rect, (unsigned long long)N.kept_same);
    return 1;
}
