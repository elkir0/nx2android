// Android port, Vulkan backend: formats, textures, views, samplers and the texture / sampler ID tables (see gfx_vk.h).
//   A texture object's image is found by its storage: textures over the same pool, offset and description share one image (as nvn_mtl.m's storage-sharing
//   table), which is how a render target and the texture a later pass samples it through meet. A packaged texture (packaged layout, CPU-written) is filled
//   from the guest's block-linear bytes (nvn_deswizzle on the CPU into scratch memory, copied in by the upload command buffer) and filled again when the
//   dirty-page tracker saw a write to its pages; any other texture is the GPU's (render targets, images), filled only by GPU work.
//   Version 1 keeps every image in VK_IMAGE_LAYOUT_GENERAL after its first use, ordered by whole-pipeline barriers.
#include "gfx_vk.h"
#include "gfx_vk_vt.h"
#include "nvn_tex.h"
#include "../gfx/tex/nvn_deswizzle.h"
#include "nvn_hmap.h"
#include "gfx_vk_shadow.h"
#include "gfx_vk_budget.h"
#include <string.h>
#define BCDEC_STATIC
#define BCDEC_BC4BC5_PRECISE
#define BCDEC_IMPLEMENTATION
#include "../third_party/bcdec/bcdec.h"

int nvn_dirty_since_span(const void *span, const void *addr, size_t len, uint64_t stamp);
const void *nvn_dirty_span(const void *addr, size_t len);
void nvn_dirty_protect(const void *addr, size_t len); void nvn_dirty_release(const void *addr, size_t len); uint64_t nvn_dirty_released(void); int nvn_dirty_runs(void);
uint64_t nvn_dirty_now(void);
int nvn_dirty_since(const void *addr, size_t len, uint64_t stamp); void nvn_dirty_protect_sparse(const void *addr, size_t len); uint64_t nvn_dirty_sparse_writes(void);   // (nvn_dirty.h)

// ---------------- formats ----------------
enum { K_NONE, K_FLOAT, K_INT, K_SINT };
typedef struct { uint16_t vk; uint8_t noalpha, rb, kind; } fent;   // vk: VkFormat (16 bits suffice for the core formats; ASTC / packed extras computed)
#define F(v) {v, 0, 0, K_FLOAT}
#define FX(v) {v, 1, 0, K_FLOAT}
#define U(v) {v, 0, 0, K_INT}
#define UX(v) {v, 1, 0, K_INT}
#define I(v) {v, 0, 0, K_SINT}
#define IX(v) {v, 1, 0, K_SINT}
static const fent FT[0x9D] = {
    [0x01] = F(VK_FORMAT_R8_UNORM), F(VK_FORMAT_R8_SNORM), U(VK_FORMAT_R8_UINT), I(VK_FORMAT_R8_SINT),
    [0x05] = F(VK_FORMAT_R16_SFLOAT), F(VK_FORMAT_R16_UNORM), F(VK_FORMAT_R16_SNORM), U(VK_FORMAT_R16_UINT), I(VK_FORMAT_R16_SINT),
    [0x0A] = F(VK_FORMAT_R32_SFLOAT), U(VK_FORMAT_R32_UINT), I(VK_FORMAT_R32_SINT),
    [0x0D] = F(VK_FORMAT_R8G8_UNORM), F(VK_FORMAT_R8G8_SNORM), U(VK_FORMAT_R8G8_UINT), I(VK_FORMAT_R8G8_SINT),
    [0x11] = F(VK_FORMAT_R16G16_SFLOAT), F(VK_FORMAT_R16G16_UNORM), F(VK_FORMAT_R16G16_SNORM), U(VK_FORMAT_R16G16_UINT), I(VK_FORMAT_R16G16_SINT),
    [0x16] = F(VK_FORMAT_R32G32_SFLOAT), U(VK_FORMAT_R32G32_UINT), I(VK_FORMAT_R32G32_SINT),
    // 0x19-0x24 three-component formats: not sampleable on mobile GPUs (vertex input only, see vk_vertex_format)
    [0x25] = F(VK_FORMAT_R8G8B8A8_UNORM), F(VK_FORMAT_R8G8B8A8_SNORM), U(VK_FORMAT_R8G8B8A8_UINT), I(VK_FORMAT_R8G8B8A8_SINT),
    [0x29] = F(VK_FORMAT_R16G16B16A16_SFLOAT), F(VK_FORMAT_R16G16B16A16_UNORM), F(VK_FORMAT_R16G16B16A16_SNORM), U(VK_FORMAT_R16G16B16A16_UINT), I(VK_FORMAT_R16G16B16A16_SINT),
    [0x2E] = F(VK_FORMAT_R32G32B32A32_SFLOAT), U(VK_FORMAT_R32G32B32A32_UINT), I(VK_FORMAT_R32G32B32A32_SINT),
    [0x31] = U(VK_FORMAT_S8_UINT), F(VK_FORMAT_D16_UNORM), F(VK_FORMAT_X8_D24_UNORM_PACK32), F(VK_FORMAT_D32_SFLOAT), F(VK_FORMAT_D24_UNORM_S8_UINT), F(VK_FORMAT_D32_SFLOAT_S8_UINT),
    [0x37] = FX(VK_FORMAT_R8G8B8A8_SRGB), F(VK_FORMAT_R8G8B8A8_SRGB),
    // NVN packed names run from low bits to high bits; Vulkan PACK16 names
    // run from high bits to low bits. RGB565 has R in bits 0..4, B in 11..15.
    [0x39] = F(VK_FORMAT_A4B4G4R4_UNORM_PACK16), FX(VK_FORMAT_A1B5G5R5_UNORM_PACK16_KHR), F(VK_FORMAT_A1B5G5R5_UNORM_PACK16_KHR), F(VK_FORMAT_B5G6R5_UNORM_PACK16),
    [0x3D] = F(VK_FORMAT_A2B10G10R10_UNORM_PACK32), U(VK_FORMAT_A2B10G10R10_UINT_PACK32), F(VK_FORMAT_B10G11R11_UFLOAT_PACK32), F(VK_FORMAT_E5B9G9R9_UFLOAT_PACK32),
    [0x41] = F(VK_FORMAT_BC1_RGB_UNORM_BLOCK), F(VK_FORMAT_BC1_RGBA_UNORM_BLOCK), F(VK_FORMAT_BC2_UNORM_BLOCK), F(VK_FORMAT_BC3_UNORM_BLOCK),
             F(VK_FORMAT_BC1_RGB_SRGB_BLOCK), F(VK_FORMAT_BC1_RGBA_SRGB_BLOCK), F(VK_FORMAT_BC2_SRGB_BLOCK), F(VK_FORMAT_BC3_SRGB_BLOCK),
    [0x49] = F(VK_FORMAT_BC4_UNORM_BLOCK), F(VK_FORMAT_BC4_SNORM_BLOCK), F(VK_FORMAT_BC5_UNORM_BLOCK), F(VK_FORMAT_BC5_SNORM_BLOCK),
             F(VK_FORMAT_BC7_UNORM_BLOCK), F(VK_FORMAT_BC7_SRGB_BLOCK), F(VK_FORMAT_BC6H_SFLOAT_BLOCK), F(VK_FORMAT_BC6H_UFLOAT_BLOCK),
    // *_UI2F / *_I2F: integer storage read as float without normalization = Vulkan's scaled formats
    [0x51] = F(VK_FORMAT_R8_USCALED), F(VK_FORMAT_R8_SSCALED), F(VK_FORMAT_R16_USCALED), F(VK_FORMAT_R16_SSCALED), U(VK_FORMAT_R32_UINT), I(VK_FORMAT_R32_SINT),
    [0x57] = F(VK_FORMAT_R8G8_USCALED), F(VK_FORMAT_R8G8_SSCALED), F(VK_FORMAT_R16G16_USCALED), F(VK_FORMAT_R16G16_SSCALED), U(VK_FORMAT_R32G32_UINT), I(VK_FORMAT_R32G32_SINT),
    [0x63] = F(VK_FORMAT_R8G8B8A8_USCALED), F(VK_FORMAT_R8G8B8A8_SSCALED), F(VK_FORMAT_R16G16B16A16_USCALED), F(VK_FORMAT_R16G16B16A16_SSCALED), U(VK_FORMAT_R32G32B32A32_UINT), I(VK_FORMAT_R32G32B32A32_SINT),
    [0x69] = F(VK_FORMAT_A2B10G10R10_SNORM_PACK32), I(VK_FORMAT_A2B10G10R10_SINT_PACK32), F(VK_FORMAT_A2B10G10R10_USCALED_PACK32), F(VK_FORMAT_A2B10G10R10_SSCALED_PACK32),
    [0x6D] = FX(VK_FORMAT_R8G8B8A8_UNORM), FX(VK_FORMAT_R8G8B8A8_SNORM), UX(VK_FORMAT_R8G8B8A8_UINT), IX(VK_FORMAT_R8G8B8A8_SINT),
    [0x71] = FX(VK_FORMAT_R16G16B16A16_SFLOAT), FX(VK_FORMAT_R16G16B16A16_UNORM), FX(VK_FORMAT_R16G16B16A16_SNORM), UX(VK_FORMAT_R16G16B16A16_UINT), IX(VK_FORMAT_R16G16B16A16_SINT),
    [0x76] = FX(VK_FORMAT_R32G32B32A32_SFLOAT), UX(VK_FORMAT_R32G32B32A32_UINT), IX(VK_FORMAT_R32G32B32A32_SINT),
    [0x95] = F(VK_FORMAT_R5G6B5_UNORM_PACK16), FX(VK_FORMAT_A1R5G5B5_UNORM_PACK16), F(VK_FORMAT_A1R5G5B5_UNORM_PACK16), F(VK_FORMAT_R5G5B5A1_UNORM_PACK16),
    [0x99] = FX(VK_FORMAT_B8G8R8A8_UNORM), F(VK_FORMAT_B8G8R8A8_UNORM), FX(VK_FORMAT_B8G8R8A8_SRGB), F(VK_FORMAT_B8G8R8A8_SRGB),
};
static vkfmt_t g_fmt[0x9D]; static uint8_t g_fmt_done[0x9D];
// BC decoded on the CPU: what each BC format decodes into (dec 1 BC1 RGB, 2 BC1, 3 BC2, 4 BC3, 5 / 6 BC4 unsigned / signed, 7 / 8 BC5, 9 / 10 BC6H unsigned / signed, 11 BC7)
static void bc_target(vkfmt_t *o) {
    int srgb = o->vk == VK_FORMAT_BC1_RGB_SRGB_BLOCK || o->vk == VK_FORMAT_BC1_RGBA_SRGB_BLOCK || o->vk == VK_FORMAT_BC2_SRGB_BLOCK || o->vk == VK_FORMAT_BC3_SRGB_BLOCK || o->vk == VK_FORMAT_BC7_SRGB_BLOCK;
    VkFormat rgba = srgb ? VK_FORMAT_R8G8B8A8_SRGB : VK_FORMAT_R8G8B8A8_UNORM;
    switch (o->vk) {
    case VK_FORMAT_BC1_RGB_UNORM_BLOCK: case VK_FORMAT_BC1_RGB_SRGB_BLOCK: o->dec = 1; o->vk = rgba; o->dbpp = 4; break;
    case VK_FORMAT_BC1_RGBA_UNORM_BLOCK: case VK_FORMAT_BC1_RGBA_SRGB_BLOCK: o->dec = 2; o->vk = rgba; o->dbpp = 4; break;
    case VK_FORMAT_BC2_UNORM_BLOCK: case VK_FORMAT_BC2_SRGB_BLOCK: o->dec = 3; o->vk = rgba; o->dbpp = 4; break;
    case VK_FORMAT_BC3_UNORM_BLOCK: case VK_FORMAT_BC3_SRGB_BLOCK: o->dec = 4; o->vk = rgba; o->dbpp = 4; break;
    case VK_FORMAT_BC4_UNORM_BLOCK: o->dec = 5; o->vk = VK_FORMAT_R8_UNORM; o->dbpp = 1; break;
    case VK_FORMAT_BC4_SNORM_BLOCK: o->dec = 6; o->vk = VK_FORMAT_R8_SNORM; o->dbpp = 1; break;
    case VK_FORMAT_BC5_UNORM_BLOCK: o->dec = 7; o->vk = VK_FORMAT_R8G8_UNORM; o->dbpp = 2; break;
    case VK_FORMAT_BC5_SNORM_BLOCK: o->dec = 8; o->vk = VK_FORMAT_R8G8_SNORM; o->dbpp = 2; break;
    case VK_FORMAT_BC6H_UFLOAT_BLOCK: o->dec = 9; o->vk = VK_FORMAT_R16G16B16A16_SFLOAT; o->dbpp = 8; break;
    case VK_FORMAT_BC6H_SFLOAT_BLOCK: o->dec = 10; o->vk = VK_FORMAT_R16G16B16A16_SFLOAT; o->dbpp = 8; break;
    default: o->dec = 11; o->vk = rgba; o->dbpp = 4; break;   // BC7
    }
}
// Decodes BC blocks (rows of whole blocks pitch bytes apart, (h + 3) / 4 rows a slice, z slices) into w x h x z texels of f->vk, rows of w * f->dbpp bytes
static void bc_decode(const vkfmt_t *f, const uint8_t *src, size_t pitch, uint32_t w, uint32_t h, uint32_t z, uint8_t *dst) {
    uint32_t bx = (w + 3) / 4, by = (h + 3) / 4; size_t row = (size_t)w * f->dbpp;
    uint8_t t[4 * 4 * 8];   // one decoded block: 4 rows of 4 texels, up to 8 bytes each (BC6H decodes to RGB16F, widened to RGBA16F)
    for (uint32_t s = 0; s < z; s++) for (uint32_t y = 0; y < by; y++) for (uint32_t x = 0; x < bx; x++) {
        const uint8_t *b = src + ((size_t)s * by + y) * pitch + (size_t)x * f->bpb;
        switch (f->dec) {
        case 1: case 2: bcdec_bc1(b, t, 16); if (f->dec == 1) for (int i = 0; i < 16; i++) t[i * 4 + 3] = 255; break;   // BC1 RGB: no alpha
        case 3: bcdec_bc2(b, t, 16); break;
        case 4: bcdec_bc3(b, t, 16); break;
        case 5: case 6: bcdec_bc4(b, t, 4, f->dec == 6); break;
        case 7: case 8: bcdec_bc5(b, t, 8, f->dec == 8); break;
        case 9: case 10: {
            uint16_t h3[16 * 3]; bcdec_bc6h_half(b, h3, 12, f->dec == 10);
            for (int i = 0; i < 16; i++) { uint16_t px[4] = {h3[i * 3], h3[i * 3 + 1], h3[i * 3 + 2], 0x3c00}; memcpy(t + i * 8, px, 8); }   // alpha 1.0
            break; }
        default: bcdec_bc7(b, t, 16); break;
        }
        uint32_t x0 = x * 4, y0 = y * 4, cw = w - x0 < 4 ? w - x0 : 4, ch = h - y0 < 4 ? h - y0 : 4;
        for (uint32_t r = 0; r < ch; r++) memcpy(dst + ((size_t)s * h + y0 + r) * row + (size_t)x0 * f->dbpp, t + r * 4 * f->dbpp, (size_t)cw * f->dbpp);
    }
}
void vk_bc_decode(const vkfmt_t *f, const uint8_t *src, size_t pitch, uint32_t w, uint32_t h, uint32_t z, uint8_t *dst) { bc_decode(f, src, pitch, w, h, z, dst); }
static pthread_mutex_t g_fmt_mu = PTHREAD_MUTEX_INITIALIZER;
const vkfmt_t *vk_fmt(int f) {
    if (f <= 0 || f >= 0x9D) return NULL;
    if (__atomic_load_n(&g_fmt_done[f], __ATOMIC_ACQUIRE)) return g_fmt[f].kind ? &g_fmt[f] : NULL;   // made already: no lock (the replay looks formats up for every texture it binds)
    pthread_mutex_lock(&g_fmt_mu);
    if (!g_fmt_done[f]) {
        vkfmt_t *o = &g_fmt[f]; memset(o, 0, sizeof *o);
        fent e = FT[f];
        if (f >= 0x79 && f <= 0x94) { int k = (f - 0x79) % 14; e = (fent){(uint16_t)(VK_FORMAT_ASTC_4x4_UNORM_BLOCK + 2 * k + (f >= 0x87)), 0, 0, K_FLOAT}; }
        if (e.vk) {
            int bw, bh; o->bpb = (uint8_t)nvn_fmt_bpb(f, &bw, &bh); o->bw = (uint8_t)bw; o->bh = (uint8_t)bh;
            o->vk = (VkFormat)(e.vk == (uint16_t)VK_FORMAT_A1B5G5R5_UNORM_PACK16_KHR ? VK_FORMAT_A1B5G5R5_UNORM_PACK16_KHR :   // (the extension formats past 16 bits)
                               e.vk == (uint16_t)VK_FORMAT_A4B4G4R4_UNORM_PACK16 ? VK_FORMAT_A4B4G4R4_UNORM_PACK16 : e.vk);   // (RGBA4 read as format 64033 before: no image)
            if (f == 0x33) {   // DEPTH24: X8_D24 where the device renders to it, else D32
                VkFormatProperties fp; vkGetPhysicalDeviceFormatProperties(V.pd, VK_FORMAT_X8_D24_UNORM_PACK32, &fp);
                if (!(fp.optimalTilingFeatures & VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT)) o->vk = VK_FORMAT_D32_SFLOAT;
            }
            if (f == 0x31) {   // STENCIL8 alone: a combined format where the device has no S8 attachments
                VkFormatProperties fp; vkGetPhysicalDeviceFormatProperties(V.pd, VK_FORMAT_S8_UINT, &fp);
                if (!(fp.optimalTilingFeatures & VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT)) o->vk = VK_FORMAT_D24_UNORM_S8_UINT;
            }
            o->depth = f >= 0x32 && f <= 0x36; o->stencil = f == 0x31 || f == 0x35 || f == 0x36 || (f == 0x31);
            o->compressed = o->bw > 1 || o->bh > 1;
            o->is_int = e.kind == K_INT || e.kind == K_SINT; o->is_sint = e.kind == K_SINT; o->kind = 1;
            o->swz = (VkComponentMapping){VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY, e.noalpha ? VK_COMPONENT_SWIZZLE_ONE : VK_COMPONENT_SWIZZLE_IDENTITY};
            if (o->compressed && o->vk >= VK_FORMAT_BC1_RGB_UNORM_BLOCK && o->vk <= VK_FORMAT_BC7_SRGB_BLOCK && !V.has_bc) {
                // no BC on the device (the Adreno 618 / 619 drivers of 2021, 512.530): an uncompressed image the CPU decodes into (bc_decode), 4 to 8 times the memory
                VkFormat bc = o->vk; bc_target(o); o->compressed = 0;
                VKLOG("texture format %#x (VkFormat %d): the device has no BC textures, decoded on the CPU into VkFormat %d", f, bc, o->vk);
            }
            if (o->vk >= VK_FORMAT_ASTC_4x4_UNORM_BLOCK && o->vk <= VK_FORMAT_ASTC_12x12_SRGB_BLOCK && !V.has_astc) {
                o->kind = 0;   // said once per format: these textures went missing with nothing in the log
                VKLOG("[RED] texture format %#x (VkFormat %d) needs ASTC (textureCompressionASTC_LDR), which the device lacks: textures in it are not drawn", f, o->vk);
            }
        }
        __atomic_store_n(&g_fmt_done[f], 1, __ATOMIC_RELEASE);   // published once complete
    }
    pthread_mutex_unlock(&g_fmt_mu);
    return g_fmt[f].kind ? &g_fmt[f] : NULL;
}
static VkImageAspectFlags aspect_of(const vkfmt_t *f) {
    if (f->vk == VK_FORMAT_S8_UINT) return VK_IMAGE_ASPECT_STENCIL_BIT;
    if (f->depth || f->stencil) return VK_IMAGE_ASPECT_DEPTH_BIT | (f->vk == VK_FORMAT_D24_UNORM_S8_UINT || f->vk == VK_FORMAT_D32_SFLOAT_S8_UINT ? VK_IMAGE_ASPECT_STENCIL_BIT : 0);
    return VK_IMAGE_ASPECT_COLOR_BIT;
}

// ---------------- images ----------------
static unsigned layers_of(const nvn_texdesc *d) { return d->target == 4 || d->target == 9 ? (d->depth ? d->depth : 1) : d->target == 8 ? 6 : d->target == 3 ? (d->height ? d->height : 1) : 1; }
static VkImageViewType vtype_of(int target) {
    switch (target) {
    case 0: return VK_IMAGE_VIEW_TYPE_1D; case 3: return VK_IMAGE_VIEW_TYPE_1D_ARRAY; case 2: return VK_IMAGE_VIEW_TYPE_3D;
    case 4: case 6: return VK_IMAGE_VIEW_TYPE_2D_ARRAY; case 8: return VK_IMAGE_VIEW_TYPE_CUBE; case 9: return VK_IMAGE_VIEW_TYPE_CUBE_ARRAY;
    default: return VK_IMAGE_VIEW_TYPE_2D;
    }
}
// storage key: textures with the same pool, offset and description share one image
typedef struct { uint64_t pool; int64_t off; nvn_texdesc d; } akey;
static nvn_hmap g_by_key = {.klen = sizeof(akey)};     // akey → vkimg_t*
static nvn_hmap g_by_serial = {.klen = sizeof(uint64_t)};  // serial → vkimg_t* (images with no storage key: no pool memory)
static pthread_mutex_t g_img_mu = PTHREAD_MUTEX_INITIALIZER;
// The sRGB / UNORM twin of a format (the one reinterpretation games make of uncompressed colour targets), VK_FORMAT_UNDEFINED when it has none
static VkFormat srgb_twin(VkFormat f) {
    switch (f) {
    case VK_FORMAT_R8G8B8A8_UNORM: return VK_FORMAT_R8G8B8A8_SRGB; case VK_FORMAT_R8G8B8A8_SRGB: return VK_FORMAT_R8G8B8A8_UNORM;
    case VK_FORMAT_B8G8R8A8_UNORM: return VK_FORMAT_B8G8R8A8_SRGB; case VK_FORMAT_B8G8R8A8_SRGB: return VK_FORMAT_B8G8R8A8_UNORM;
    case VK_FORMAT_A8B8G8R8_UNORM_PACK32: return VK_FORMAT_A8B8G8R8_SRGB_PACK32; case VK_FORMAT_A8B8G8R8_SRGB_PACK32: return VK_FORMAT_A8B8G8R8_UNORM_PACK32;
    default: return VK_FORMAT_UNDEFINED;
    }
}
// DM_VK_UBWC (start-up, bits): 1, uncompressed colour images get a format list (their format and its sRGB twin) instead of a free MUTABLE_FORMAT, which Adreno
// needs to keep UBWC framebuffer compression; views in another format then use the image's own (vk_tex_view). 2, depth / stencil images are made without
// MUTABLE_FORMAT and EXTENDED_USAGE (their views always take the image's own format). Unicorn Overlord on an Adreno 730: 1 cut GPU time by 35-40%. Both are the
// default (3): Super Mario Bros. Wonder's world map on the Adreno 830 took 8.7-8.9 ms of GPU a frame instead of 11.8, the same picture (1 alone: 9.5-10.0 ms).
// Xenoblade Chronicles 3's field took 22.9-23.0 ms against 23.0-23.6 (its own shading dominates), with clouds in every frame of a sky turn and no idle flicker;
// Graveyard Keeper 2, Advance Wars and DKCR drew the same pictures. A colour view in another format than the image's and its twin is logged [RED] (vk_tex_view):
// such a game wants DM_VK_UBWC=2 in its profile
static int ubwc_mode(void) { static int v = -1; if (v < 0) { const char *e = getenv("DM_VK_UBWC"); v = e ? atoi(e) : 3; } return v; }
static vkimg_t *img_new(const tex_t *t, const vkfmt_t *f) {
    vkimg_t *im = calloc(1, sizeof *im);
    const nvn_texdesc *d = &t->d;
    im->w = d->width ? d->width : 1; im->h = d->target == 0 || d->target == 3 ? 1 : (d->height ? d->height : 1);
    im->d = d->target == 2 ? (d->depth ? d->depth : 1) : 1; im->layers = layers_of(d);
    im->levels = d->levels ? d->levels : 1; im->samples = d->samples > 1 ? d->samples : 1; im->target = d->target; im->nvn_format = d->format;
    im->fmt = f->vk; im->vtype = vtype_of(d->target); im->aspect = aspect_of(f);
    VkFormatProperties fp; vkGetPhysicalDeviceFormatProperties(V.pd, f->vk, &fp);
    VkImageUsageFlags u = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    if (fp.optimalTilingFeatures & VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT) u |= VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
    if (fp.optimalTilingFeatures & VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT) u |= VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
    if ((d->flags & NVN_TEXTURE_FLAGS_IMAGE) && (fp.optimalTilingFeatures & VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT)) u |= VK_IMAGE_USAGE_STORAGE_BIT;
    // a block-compressed texture can be drawn into through a view in the uncompressed format of its block size (Mario Kart 8 Deluxe writes BC3 blocks into its
    // reflection cubemaps through RGBA32UI views): with EXTENDED_USAGE the image may carry a usage its own format lacks, and an attachment view needs it on the image
    if (f->compressed && !f->dec && (f->bpb == 16 || f->bpb == 8)) { VkFormatProperties up; vkGetPhysicalDeviceFormatProperties(V.pd, f->bpb == 16 ? VK_FORMAT_R32G32B32A32_UINT : VK_FORMAT_R32G32_UINT, &up);
        if (up.optimalTilingFeatures & VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT) u |= VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT; }
    VkImageCreateInfo ci = {VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, NULL,
        // a square 2D array of 6n layers too: shaders read one as a cube (Pokémon Legends: Arceus's reflection probes; the view was refused by the validation layer without the flag)
        VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT | VK_IMAGE_CREATE_EXTENDED_USAGE_BIT | (d->target == 8 || d->target == 9 || (d->target == 4 && im->layers >= 6 && im->layers % 6 == 0 && im->w == im->h && im->samples <= 1) ? VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT : 0) |
        (d->target == 2 ? VK_IMAGE_CREATE_2D_ARRAY_COMPATIBLE_BIT : 0) | (f->compressed ? VK_IMAGE_CREATE_BLOCK_TEXEL_VIEW_COMPATIBLE_BIT : 0),
        d->target == 2 ? VK_IMAGE_TYPE_3D : d->target == 0 || d->target == 3 ? VK_IMAGE_TYPE_1D : VK_IMAGE_TYPE_2D, f->vk,
        {im->w, im->h, im->d}, im->levels, im->layers, (VkSampleCountFlagBits)im->samples, VK_IMAGE_TILING_OPTIMAL, u, VK_SHARING_MODE_EXCLUSIVE, 0, NULL, VK_IMAGE_LAYOUT_UNDEFINED};
    if (im->samples > 1) ci.flags &= ~(VkImageCreateFlags)VK_IMAGE_CREATE_2D_ARRAY_COMPATIBLE_BIT;
    VkFormat fl[2] = {f->vk, srgb_twin(f->vk)}; VkImageFormatListCreateInfo fli = {VK_STRUCTURE_TYPE_IMAGE_FORMAT_LIST_CREATE_INFO, NULL, fl[1] ? 2u : 1u, fl};
    if (!f->compressed && (ubwc_mode() & (im->aspect == VK_IMAGE_ASPECT_COLOR_BIT ? 1 : 2))) {   // (a depth format has no twin: a list of itself, no MUTABLE_FORMAT)
        im->fixed_fmt = 1; im->fmt2 = fl[1]; ci.pNext = &fli;
        ci.flags &= ~(VkImageCreateFlags)VK_IMAGE_CREATE_EXTENDED_USAGE_BIT; if (!fl[1]) ci.flags &= ~(VkImageCreateFlags)VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT;
    }
    im->usage = ci.usage; im->cflags = ci.flags;
    VkResult r = vkCreateImage(V.dev, &ci, NULL, &im->img);
    if (r && f->compressed && (ci.usage & VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT)) {   // the driver refused the attachment usage on a compressed image: as before, without it
        VKRED_ONCE(96, "compressed image format %#x refused colour-attachment usage (%d): created without it", d->format, r);
        ci.usage &= ~(VkImageUsageFlags)VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT; im->usage = ci.usage; r = vkCreateImage(V.dev, &ci, NULL, &im->img); }
    if (r) { VKRED_ONCE(10, "vkCreateImage failed (%d): %ux%ux%u format %#x (vk %d) target %d levels %u layers %u samples %u", r, im->w, im->h, im->d, d->format, f->vk, d->target, im->levels, im->layers, im->samples); free(im); return NULL; }
    VkMemoryRequirements req; vkGetImageMemoryRequirements(V.dev, im->img, &req);
    if ((ci.usage & (VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT)) && im->w >= 256) { static int n; if (n++ < 16)   // UBWC check: a compressed target takes room for its metadata
        VKLOG("target %ux%u format %d%s: %llu bytes", im->w, im->h, f->vk, im->fixed_fmt ? " (format-listed)" : "", (unsigned long long)req.size); }
    VkDeviceSize moff;
    if (!(im->mem = vk_image_memory(&req, vk_mem_type(req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT), &moff, &im->sub))) {
        VKRED_ONCE(11, "image memory allocation failed (%llu bytes)", (unsigned long long)req.size); vkDestroyImage(V.dev, im->img, NULL); free(im); return NULL; }
    VKCHK(vkBindImageMemory(V.dev, im->img, im->mem, moff));
    im->layout = VK_IMAGE_LAYOUT_UNDEFINED;
    return im;
}
// ---- promoted formats: a format without sparse tiles made in its exact wider twin (tests: tests/test_vk_promo.py) ----
// The stock Adreno 840 driver gives no sparse residency to its 8-bit formats: Mario Kart World 1.8.0's four 16384x16384 R8 textures of 6 levels in sparse pools, which the GPU writes, were made
// whole at 341 MB each (1,365 MB of memory of their own, run 10) where the game maps 8-26 MB of each. A wider twin holds every value of the format exactly: R8 UNORM k/255 is R16 UNORM
// k*257/65535, integers widen as they are, a half is a float. Sampled it reads the same values; drawn into, a value is rounded once, to the wider format (more precision, the same blending).
// NVN lays the texture out in 64 KB tiles of the narrower format's standard shape (256x256 texels at 8 bits), the device's tiles of the wider one have their own (256x128 at 16 bits, the
// Adreno 840's "R16 256x128"): one NVN tile is two device tiles, nx across and ny down (promo_split), each bound to a slot of its own behind the page the game mapped (sp_tile_wide), and the
// texels a copy moves widened or narrowed on the way (promo_widen, promo_narrow). DM_VK_SPARSE_PROMO=0 (start-up): none, made whole as before
enum { PROMO_UNORM8 = 1, PROMO_UINT8, PROMO_SINT8, PROMO_HALF, PROMO_UINT16, PROMO_SINT16 };
typedef struct { VkFormat from, to; uint8_t kind, ch, bpt; } promo_t;   // ch: channels; bpt: the narrower format's bytes a texel (the wider's twice that)
static const promo_t PROMO[] = {
    {VK_FORMAT_R8_UNORM, VK_FORMAT_R16_UNORM, PROMO_UNORM8, 1, 1}, {VK_FORMAT_R8G8_UNORM, VK_FORMAT_R16G16_UNORM, PROMO_UNORM8, 2, 2}, {VK_FORMAT_R8G8B8A8_UNORM, VK_FORMAT_R16G16B16A16_UNORM, PROMO_UNORM8, 4, 4},
    {VK_FORMAT_R8_UINT, VK_FORMAT_R16_UINT, PROMO_UINT8, 1, 1}, {VK_FORMAT_R8G8_UINT, VK_FORMAT_R16G16_UINT, PROMO_UINT8, 2, 2}, {VK_FORMAT_R8G8B8A8_UINT, VK_FORMAT_R16G16B16A16_UINT, PROMO_UINT8, 4, 4},
    {VK_FORMAT_R8_SINT, VK_FORMAT_R16_SINT, PROMO_SINT8, 1, 1}, {VK_FORMAT_R8G8_SINT, VK_FORMAT_R16G16_SINT, PROMO_SINT8, 2, 2}, {VK_FORMAT_R8G8B8A8_SINT, VK_FORMAT_R16G16B16A16_SINT, PROMO_SINT8, 4, 4},
    {VK_FORMAT_R16_SFLOAT, VK_FORMAT_R32_SFLOAT, PROMO_HALF, 1, 2}, {VK_FORMAT_R16G16_SFLOAT, VK_FORMAT_R32G32_SFLOAT, PROMO_HALF, 2, 4}, {VK_FORMAT_R16G16B16A16_SFLOAT, VK_FORMAT_R32G32B32A32_SFLOAT, PROMO_HALF, 4, 8},
    {VK_FORMAT_R16_UINT, VK_FORMAT_R32_UINT, PROMO_UINT16, 1, 2}, {VK_FORMAT_R16G16_UINT, VK_FORMAT_R32G32_UINT, PROMO_UINT16, 2, 4}, {VK_FORMAT_R16G16B16A16_UINT, VK_FORMAT_R32G32B32A32_UINT, PROMO_UINT16, 4, 8},
    {VK_FORMAT_R16_SINT, VK_FORMAT_R32_SINT, PROMO_SINT16, 1, 2}, {VK_FORMAT_R16G16_SINT, VK_FORMAT_R32G32_SINT, PROMO_SINT16, 2, 4}, {VK_FORMAT_R16G16B16A16_SINT, VK_FORMAT_R32G32B32A32_SINT, PROMO_SINT16, 4, 8},
};   // (no SNORM: k/127 is not m/32767; no sRGB: there is no 16-bit one; no 16-bit UNORM: there is no 32-bit one)
static const promo_t *promo_of(VkFormat f) { for (size_t i = 0; i < sizeof PROMO / sizeof *PROMO; i++) if (PROMO[i].from == f) return &PROMO[i]; return NULL; }
static float promo_h2f(uint16_t h) {   // a half's float, exactly
    uint32_t s = (uint32_t)(h >> 15) << 31, e = (h >> 10) & 31, m = h & 1023, x;
    if (!e) { if (!m) x = s; else { e = 1; while (!(m & 1024)) { m <<= 1; e--; } m &= 1023; x = s | ((e + 112) << 23) | (m << 13); } }
    else if (e == 31) x = s | 0x7f800000 | (m << 13); else x = s | ((e + 112) << 23) | (m << 13);
    float f; memcpy(&f, &x, 4); return f;
}
static uint16_t promo_f2h(float f) {   // a float's half, rounded to the nearest (ties to even); past the largest half, infinity
    uint32_t x; memcpy(&x, &f, 4); uint32_t s = (x >> 16) & 0x8000, e = (x >> 23) & 0xff, m = x & 0x7fffff; int ee = (int)e - 112;
    if (e == 0xff) return (uint16_t)(s | 0x7c00 | (m ? 0x200 | (m >> 13) : 0));   // infinity, NaN (quiet)
    if (ee >= 31) return (uint16_t)(s | 0x7c00);
    if (ee <= 0) { if (ee < -10) return (uint16_t)s;   // a half subnormal, or zero
        uint32_t mm = m | 0x800000, sh = (uint32_t)(14 - ee), hm = mm >> sh, rem = mm & ((1u << sh) - 1), half = 1u << (sh - 1);
        if (rem > half || (rem == half && (hm & 1))) hm++; return (uint16_t)(s | hm); }   // (rounding up to the smallest normal is right)
    uint32_t r = (uint32_t)ee << 10 | m >> 13, rem = m & 0x1fff; if (rem > 0x1000 || (rem == 0x1000 && (r & 1))) r++;   // (a carry into the exponent is right, up to infinity)
    return (uint16_t)(s | r);
}
static void promo_widen(const promo_t *p, const uint8_t *src, size_t n, uint8_t *dst) {   // n texels of the narrower format into the wider
    size_t v = n * p->ch;
    for (size_t i = 0; i < v; i++)
        switch (p->kind) {
        case PROMO_UNORM8: { uint16_t x = (uint16_t)(src[i] * 257u); memcpy(dst + 2 * i, &x, 2); break; }
        case PROMO_UINT8: { uint16_t x = src[i]; memcpy(dst + 2 * i, &x, 2); break; }
        case PROMO_SINT8: { int16_t x = (int8_t)src[i]; memcpy(dst + 2 * i, &x, 2); break; }
        case PROMO_HALF: { uint16_t h; memcpy(&h, src + 2 * i, 2); float f = promo_h2f(h); memcpy(dst + 4 * i, &f, 4); break; }
        case PROMO_UINT16: { uint16_t h; memcpy(&h, src + 2 * i, 2); uint32_t x = h; memcpy(dst + 4 * i, &x, 4); break; }
        default: { int16_t h; memcpy(&h, src + 2 * i, 2); int32_t x = h; memcpy(dst + 4 * i, &x, 4); break; }
        }
}
static void promo_narrow(const promo_t *p, const uint8_t *src, size_t n, uint8_t *dst) {   // back: the wider values as drawing into the narrower format would have stored them
    size_t v = n * p->ch;
    for (size_t i = 0; i < v; i++)
        switch (p->kind) {
        case PROMO_UNORM8: { uint16_t x; memcpy(&x, src + 2 * i, 2); dst[i] = (uint8_t)((x * 255u + 32767u) / 65535u); break; }
        case PROMO_UINT8: case PROMO_SINT8: { uint16_t x; memcpy(&x, src + 2 * i, 2); dst[i] = (uint8_t)x; break; }   // (the low bits, as an 8-bit integer target keeps)
        case PROMO_HALF: { float f; memcpy(&f, src + 4 * i, 4); uint16_t h = promo_f2h(f); memcpy(dst + 2 * i, &h, 2); break; }
        default: { uint32_t x; memcpy(&x, src + 4 * i, 4); uint16_t h = (uint16_t)x; memcpy(dst + 2 * i, &h, 2); break; }
        }
}
// NVN's tile of tw x th texels split into the device's of dtw x dth for a format twice as wide: 1, with nx across and ny down, when they tile it exactly, two of them
static int promo_split(uint32_t tw, uint32_t th, uint32_t dtw, uint32_t dth, uint32_t *nx, uint32_t *ny) {
    if (!dtw || !dth || tw % dtw || th % dth) return 0;
    *nx = tw / dtw; *ny = th / dth; return *nx * *ny == 2;
}
// Device tile k (across, then down) of NVN tile (tx, ty) of a level lw x lh: its texels, 0 when it lies past the level's edge (a bind's extent is the granularity or reaches the edge)
static int promo_sub(uint32_t tx, uint32_t ty, uint32_t tw, uint32_t th, uint32_t nx, uint32_t dtw, uint32_t dth, uint32_t k, uint32_t lw, uint32_t lh, uint32_t *x, uint32_t *y, uint32_t *w, uint32_t *h) {
    *x = tx * tw + (k % nx) * dtw; *y = ty * th + (k / nx) * dth; if (*x >= lw || *y >= lh) return 0;
    *w = lw - *x < dtw ? lw - *x : dtw; *h = lh - *y < dth ? lh - *y : dth; return 1;
}
// ---- end of promoted formats ----
VkFormat vk_promo_from(const vkimg_t *im) { return im && im->promo ? PROMO[im->promo - 1].from : VK_FORMAT_UNDEFINED; }
uint32_t vk_promo_bpt(const vkimg_t *im) { return im && im->promo ? PROMO[im->promo - 1].bpt : 0; }
void vk_promo_widen(const vkimg_t *im, const uint8_t *src, size_t n, uint8_t *dst) { promo_widen(&PROMO[im->promo - 1], src, n, dst); }
void vk_promo_narrow(const vkimg_t *im, const uint8_t *src, size_t n, uint8_t *dst) { promo_narrow(&PROMO[im->promo - 1], src, n, dst); }
// The wider twin of format `from` that can stand for it in a sparse image of usage u whose NVN tile is tw x th: it does what the texture's format does there (sampled, filtered, drawn into
// and blended, stored to, read and written without a declared format, copied), and its device tiles are the standard shape, two to an NVN tile. NULL when there is none
static const promo_t *promo_for(VkFormat from, VkImageUsageFlags u, uint32_t tw, uint32_t th, uint32_t *nx, uint32_t *ny, uint32_t *dtw, uint32_t *dth) {
    static int on = -1; if (on < 0) { const char *e = getenv("DM_VK_SPARSE_PROMO"); on = !(e && !atoi(e)); }
    const promo_t *p = on ? promo_of(from) : NULL; if (!p) return NULL;
    VkFormatProperties3 a3 = {VK_STRUCTURE_TYPE_FORMAT_PROPERTIES_3}, b3 = {VK_STRUCTURE_TYPE_FORMAT_PROPERTIES_3};
    VkFormatProperties2 a = {VK_STRUCTURE_TYPE_FORMAT_PROPERTIES_2, &a3}, b = {VK_STRUCTURE_TYPE_FORMAT_PROPERTIES_2, &b3};
    vkGetPhysicalDeviceFormatProperties2(V.pd, from, &a); vkGetPhysicalDeviceFormatProperties2(V.pd, p->to, &b);
    VkFormatFeatureFlags need = a.formatProperties.optimalTilingFeatures & (VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT | VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT | VK_FORMAT_FEATURE_TRANSFER_SRC_BIT |
        VK_FORMAT_FEATURE_TRANSFER_DST_BIT | (u & VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT ? VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT | VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BLEND_BIT : 0) |
        (u & VK_IMAGE_USAGE_STORAGE_BIT ? VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT : 0));   // (what the image's usage takes; blits: checked by the copies that need them)
    VkFormatFeatureFlags2 need3 = u & VK_IMAGE_USAGE_STORAGE_BIT ? a3.optimalTilingFeatures & (VK_FORMAT_FEATURE_2_STORAGE_READ_WITHOUT_FORMAT_BIT | VK_FORMAT_FEATURE_2_STORAGE_WRITE_WITHOUT_FORMAT_BIT) : 0;
    if ((b.formatProperties.optimalTilingFeatures & need) != need || (b3.optimalTilingFeatures & need3) != need3) return NULL;
    VkSparseImageFormatProperties sf[4]; uint32_t nsf = 4, k = 4; vkGetPhysicalDeviceSparseImageFormatProperties(V.pd, p->to, VK_IMAGE_TYPE_2D, VK_SAMPLE_COUNT_1_BIT, u, VK_IMAGE_TILING_OPTIMAL, &nsf, sf);
    for (uint32_t i = 0; i < nsf; i++) if (sf[i].aspectMask & VK_IMAGE_ASPECT_COLOR_BIT) k = i;
    if (k == 4 || sf[k].imageGranularity.depth != 1 || (sf[k].flags & VK_SPARSE_IMAGE_FORMAT_NONSTANDARD_BLOCK_SIZE_BIT)) return NULL;
    *dtw = sf[k].imageGranularity.width; *dth = sf[k].imageGranularity.height;
    return promo_split(tw, th, *dtw, *dth, nx, ny) ? p : NULL;
}

// ---------------- sparse textures over sparse virtual pools ----------------
// A sparse texture in a sparse virtual pool (hle_nvn.c struct vpool: MapVirtual maps physical pages into it) holds what the game mapped into it, tile by tile. Mario Kart World keeps its
// terrain in six 8192x8192x68 arrays of 12 levels (two BC6H of 5,803 MB each made whole, four ASTC 8x8 of 1,451 MB) and maps a small part of them (by its flyover's first minute 21,796
// tiles, 1.36 GB, where made whole they take 8.7 GB: more than the phone's 16 GB leave the GPU). NVN lays such a texture out in 64 KB tiles of the standard sparse block shape (256x256
// texels of 16-byte blocks, 512x512 of ASTC 8x8, 128x128 of RGBA8), its mip tail from the first level smaller than a tile, which is Vulkan's standard shape too
// (residencyStandard2DBlockShape). So it is a sparse-resident image (sparse_make, as nvn_mtl.m's placement sparse textures): no memory until a tile the game mapped in full is bound to a
// 64 KB slot of a block of device memory and its 64 KB decoded into it (nvn_deswizzle_tile); a tile the game unmaps is unbound and its slot reused once the GPU is past. Each layer's mip
// tail is backed at creation (one block, the device's tail size a layer) and uploaded when the game has mapped all of its storage (nvn_deswizzle_from). The pool's mapping changes reach the
// image through the pool's log of changed ranges (nvn_sparse_log_read): a frame without any costs one comparison. Its binds go to the queue before the submission whose uploads and draws
// read them (vk_sparse_flush, a timeline semaphore). Only a 2D texture or 2D array of SP_MIN or more made whole, whose tiles and tail are the device's, goes this way: the others are made
// whole and uploaded through the page table (vk_tex_: the degraded path, under a budget, which is also what a device without sparse residency gets)
int nvn_tex_sparse_tiles(const nvn_texdesc *d, uint32_t *tw, uint32_t *th, uint32_t *td);   // (nvn_tex.h)
extern uint64_t g_tex_frame;   // presents so far (vk_tex_sweep)
#define SP_TILE 0x10000u
#define SP_MIN (32ull << 20)   // (sampled RGBA8 / RGBA16F arrays of 32 MB and more resident on the Adreno 840: 448 MB of tile blocks in Mario Kart World, measured with another port)
enum { SP_BLOCK_TILES = 1024 };   // 64 MB blocks of device memory (nvn_mtl.m's placement heaps are 256 MB: smaller blocks follow a phone's memory more closely)
static pthread_mutex_t g_sp_mu = PTHREAD_MUTEX_INITIALIZER;
static VkDeviceMemory *g_sp_blk; static uint32_t g_sp_nblk, g_sp_capblk, g_sp_type = UINT32_MAX;   // the slots' blocks (a slot is block × SP_BLOCK_TILES + tile), all of one memory type
static uint32_t *g_sp_free; static size_t g_sp_nfree, g_sp_capfree;
static struct sp_late { uint32_t slot; uint64_t fence; } *g_sp_late; static size_t g_sp_nlate, g_sp_caplate;   // unbound slots, free once the GPU is past the fence
static uint64_t g_sp_live, g_sp_peak, g_sp_shared, g_sp_lost = 1, g_sp_ups, g_sp_vaddr, g_sp_nimg;   // g_sp_vaddr: the sparse images' address space (sparse_make)
// Shared tiles (nvn_mtl.m sp_tl, upstream 1110473c): a game maps one physical page at several places at once (Mario Kart World's race: 976 MB of memory behind 2,959 MB of mappings,
// always in textures of one format), and on hardware every place is that memory. A tile whose 16 pages are consecutive physical memory (nvn_sparse_tile_phys) takes the slot of that memory
// and format, so every place mapping it binds the same 64 KB (sparseResidencyAliased; without it every place gets a slot of its own, as before that change). Until the GPU writes
// through a texture mapping it (vk_sparse_gpu_wrote) a slot is a copy of the memory: each new place uploads it again (the memory may have changed in ways no fault shows), and it goes back
// with its last place. A drawn slot holds what the memory does not: it is uploaded only after a CPU write and stays while its physical pool lives (mtl_sp_phys_gone). The tail, tiles of
// scattered pages and tiles cut by a level's edge keep a slot of their own. p: the memory a shared slot stands for (0: one place's own); stamp: its content's dirty-page stamp
typedef struct { uintptr_t p, ph; uint64_t stamp; uint32_t rc; int fmt; uint8_t drawn; } sp_tl;   // ph: the physical memory of its one place, for the counts below (no aliasing)
static sp_tl *g_sp_tl; static size_t g_sp_ntl;
static uint32_t *g_sp_ph; static size_t g_sp_phcap, g_sp_phn;   // physical memory and format → slot + 1, linear probing
struct vksparse {
    VkImage img; int gpu, synced, undrawn; uint32_t tw, th, ntiled, levels, layers, per_layer, tail_at, bw, bh, bpb, w, h, first; uint64_t vsize;   // first: the texture level of its level 0; vsize: its sparse address space
    uint32_t tiles_x[16], tiles_y[16], loff_t[16]; uint64_t loff[16], tail_off, layer_size;   // tiles of a layer: those of the tiled levels from loff_t[l], then its tail (at tail_at)
    uint32_t *slot, *bseq;           // per tile: its slot + 1 (0: not bound) for each of its nsub device tiles, and the bind batch (g_sp_batch) it was last queued in
    const promo_t *pr; uint32_t nx, nsub, dtw, dth;   // made in its format's wider twin (promo_for): an NVN tile is nsub device tiles of dtw x dth, nx across (not promoted: 1 of tw x th)
    uint8_t *tup, *tq;               // per layer: its tail holds the game's bytes (uploaded since it was mapped); per tile: it is in todo
    VkDeviceMemory tail_mem, meta_mem;   // the mip tails' memory (bound whole at creation) and the metadata's, if the device asks for any
    uint64_t gen, stamp, seq, wseen, wframe, lost, up_gen;   // the page table's generation and the dirty-page stamp at the last look; the pool's log handled up to seq; sparse writes seen
    uint32_t *todo, ntodo, captodo;  // tiles left to the next look: the submission's upload budget ran out (SP_BUDGET)
};
static sp_tl *sp_tl_of(uint32_t slot) {   // caller holds g_sp_mu
    if (slot >= g_sp_ntl) { size_t n = ((size_t)slot / SP_BLOCK_TILES + 1) * SP_BLOCK_TILES; sp_tl *g = realloc(g_sp_tl, n * sizeof *g); if (!g) die("[vk] sparse tile table");
        memset(g + g_sp_ntl, 0, (n - g_sp_ntl) * sizeof *g); g_sp_tl = g; g_sp_ntl = n; }
    return &g_sp_tl[slot];
}
static int sp_slot(uint32_t *out) {   // a free slot (its block made when none is left), caller holds g_sp_mu; 0 when the device has no memory for another block
    uint64_t done = vk_fence_done(); size_t k = 0;
    for (size_t i = 0; i < g_sp_nlate; i++) {
        if (g_sp_late[i].fence > done) { g_sp_late[k++] = g_sp_late[i]; continue; }
        if (g_sp_nfree == g_sp_capfree) { g_sp_capfree = g_sp_capfree ? 2 * g_sp_capfree : 4096; if (!(g_sp_free = realloc(g_sp_free, g_sp_capfree * sizeof *g_sp_free))) die("[vk] sparse slot list"); }
        g_sp_free[g_sp_nfree++] = g_sp_late[i].slot;
    }
    g_sp_nlate = k;
    if (!g_sp_nfree) {
        VkMemoryAllocateInfo ai = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, NULL, (VkDeviceSize)SP_BLOCK_TILES * SP_TILE, g_sp_type}; VkDeviceMemory m;
        if (vkAllocateMemory(V.dev, &ai, NULL, &m)) { VKRED_ONCE(150, "sparse tiles: no memory for another 64 MB block (%llu tiles bound): tiles left unbound", (unsigned long long)g_sp_live); return 0; }
        if (g_sp_nblk == g_sp_capblk) { g_sp_capblk = g_sp_capblk ? 2 * g_sp_capblk : 16; if (!(g_sp_blk = realloc(g_sp_blk, g_sp_capblk * sizeof *g_sp_blk))) die("[vk] sparse block list"); }
        uint32_t base = g_sp_nblk * SP_BLOCK_TILES; g_sp_blk[g_sp_nblk++] = m;
        if (g_sp_capfree < g_sp_nfree + SP_BLOCK_TILES) { g_sp_capfree = g_sp_nfree + SP_BLOCK_TILES; if (!(g_sp_free = realloc(g_sp_free, g_sp_capfree * sizeof *g_sp_free))) die("[vk] sparse slot list"); }
        for (uint32_t i = SP_BLOCK_TILES; i-- > 0;) g_sp_free[g_sp_nfree++] = base + i;   // taken from the front of the block first
        VKLOG("sparse tiles: memory block %u (64 MB), %llu tiles bound", g_sp_nblk, (unsigned long long)g_sp_live);
    }
    *out = g_sp_free[--g_sp_nfree]; if (++g_sp_live > g_sp_peak) g_sp_peak = g_sp_live;
    *sp_tl_of(*out) = (sp_tl){.rc = 1};
    return 1;
}
static void sp_slot_free(uint32_t slot) {   // caller holds g_sp_mu; the GPU may still read it through the work submitted so far and the submission being recorded
    if (g_sp_nlate == g_sp_caplate) { g_sp_caplate = g_sp_caplate ? 2 * g_sp_caplate : 1024; if (!(g_sp_late = realloc(g_sp_late, g_sp_caplate * sizeof *g_sp_late))) die("[vk] sparse slot list"); }
    g_sp_late[g_sp_nlate++] = (struct sp_late){slot, mtl_last_fence() + 1}; g_sp_live--;
}
static size_t sp_ph_at(uintptr_t p, int fmt) { return (size_t)((((uint64_t)p >> 16) ^ (uint64_t)fmt * 0x9E3779B97F4A7C15ull) * 0x9E3779B97F4A7C15ull >> 20) & (g_sp_phcap - 1); }
static uint32_t sp_ph_find(uintptr_t p, int fmt) {   // caller holds g_sp_mu; the slot, or ~0u
    if (!g_sp_phcap) return ~0u;
    for (size_t i = sp_ph_at(p, fmt);; i = (i + 1) & (g_sp_phcap - 1)) { uint32_t e = g_sp_ph[i]; if (!e) return ~0u; if (g_sp_tl[e - 1].p == p && g_sp_tl[e - 1].fmt == fmt) return e - 1; }
}
static void sp_ph_put(uint32_t slot) {   // caller holds g_sp_mu
    if (2 * (g_sp_phn + 1) > g_sp_phcap) {
        uint32_t *old = g_sp_ph; size_t oc = g_sp_phcap; g_sp_phcap = oc ? 2 * oc : 4096;
        if (!(g_sp_ph = calloc(g_sp_phcap, sizeof *g_sp_ph))) die("[vk] shared sparse tile table");
        for (size_t i = 0; i < oc; i++) if (old[i]) { size_t j = sp_ph_at(g_sp_tl[old[i] - 1].p, g_sp_tl[old[i] - 1].fmt); while (g_sp_ph[j]) j = (j + 1) & (g_sp_phcap - 1); g_sp_ph[j] = old[i]; }
        free(old);
    }
    size_t j = sp_ph_at(g_sp_tl[slot].p, g_sp_tl[slot].fmt); while (g_sp_ph[j]) j = (j + 1) & (g_sp_phcap - 1);
    g_sp_ph[j] = slot + 1; g_sp_phn++; g_sp_shared++;
}
static void sp_ph_del(uint32_t slot) {   // caller holds g_sp_mu; backward-shift deletion
    if (!g_sp_phcap) return;
    size_t i = sp_ph_at(g_sp_tl[slot].p, g_sp_tl[slot].fmt); while (g_sp_ph[i] && g_sp_ph[i] != slot + 1) i = (i + 1) & (g_sp_phcap - 1);
    if (!g_sp_ph[i]) return;
    g_sp_ph[i] = 0; g_sp_phn--; g_sp_shared--;
    for (size_t j = (i + 1) & (g_sp_phcap - 1); g_sp_ph[j]; j = (j + 1) & (g_sp_phcap - 1)) {
        size_t h = sp_ph_at(g_sp_tl[g_sp_ph[j] - 1].p, g_sp_tl[g_sp_ph[j] - 1].fmt);
        if ((j > i && (h <= i || h > j)) || (j < i && h <= i && h > j)) { g_sp_ph[i] = g_sp_ph[j]; g_sp_ph[j] = 0; i = j; }
    }
}
// PARITY G3: without sparseResidencyAliased (the Adreno 840 has none: "aliased no") a physical page the game maps at several places of its sparse textures gets a tile of its own at each,
// so what the GPU draws at one place is not seen at the others (upstream measured about 3 places a page in a race), and a drawn tile whose place moves to another page keeps its old
// content (sp_tile: a GPU-written image's bound tile is not uploaded again). Counted for the sparse lines before a fix (move-on-remap, copy-on-map) is chosen: pages bound at two places
// or more now, binds of a GPU-written texture's tile over a page bound elsewhere, and drawn tiles whose page changed. sp_pc: physical page → places bound, linear probing (keys stay)
static struct sp_pc { uintptr_t p; uint32_t n; } *g_sp_pc; static size_t g_sp_pccap, g_sp_pcused; static uint64_t g_sp_multi, g_sp_multi_gpu, g_sp_moved;
static uint32_t *sp_pc_at(uintptr_t p, int add) {   // caller holds g_sp_mu; NULL: not there (or no room)
    if (add && 2 * (g_sp_pcused + 1) > g_sp_pccap) {   // rebuilt with the live keys only
        struct sp_pc *o = g_sp_pc; size_t oc = g_sp_pccap, live = 0; for (size_t i = 0; i < oc; i++) live += o[i].n > 0;
        size_t nc = oc ? oc : 4096; while (4 * (live + 1) > nc) nc *= 2; struct sp_pc *g = calloc(nc, sizeof *g); if (!g) return NULL;
        g_sp_pc = g; g_sp_pccap = nc; g_sp_pcused = 0;
        for (size_t i = 0; i < oc; i++) if (o[i].n) { size_t j = (size_t)(((uint64_t)o[i].p >> 16) * 0x9E3779B97F4A7C15ull >> 20) & (nc - 1); while (g[j].p) j = (j + 1) & (nc - 1); g[j] = o[i]; g_sp_pcused++; }
        free(o); }
    if (!g_sp_pccap) return NULL;
    for (size_t i = (size_t)(((uint64_t)p >> 16) * 0x9E3779B97F4A7C15ull >> 20) & (g_sp_pccap - 1);; i = (i + 1) & (g_sp_pccap - 1)) {
        if (g_sp_pc[i].p == p) return &g_sp_pc[i].n;
        if (!g_sp_pc[i].p) { if (!add) return NULL; g_sp_pc[i].p = p; g_sp_pcused++; return &g_sp_pc[i].n; } }
}
static const char *sp_g3_text(void) {   // (the sparse lines: G3's counts, no aliasing)
    static char b[200]; if (V.has_sparse_alias) return "";
    snprintf(b, sizeof b, "; without aliasing: %llu pages bound at 2 places or more, %llu binds of a drawn texture's tile over such a page, %llu drawn tiles whose page moved (old content kept)",
             (unsigned long long)g_sp_multi, (unsigned long long)g_sp_multi_gpu, (unsigned long long)g_sp_moved); return b;
}
static void sp_pc_add(sp_tl *e, uintptr_t ph, int gpu) {   // caller holds g_sp_mu: the slot's one place is over physical memory ph now (0: not a whole tile of consecutive pages)
    if (e->ph == ph) return;
    if (e->ph) { uint32_t *n = sp_pc_at(e->ph, 0); if (n && *n && (*n)-- == 2) g_sp_multi--; }
    e->ph = ph; if (!ph) return;
    uint32_t *n = sp_pc_at(ph, 1); if (!n) return;
    if (++*n == 2) g_sp_multi++;
    if (*n >= 2 && gpu) g_sp_multi_gpu++;
}
static void sp_release(uint32_t slot) {   // caller holds g_sp_mu: one place stops binding the slot
    sp_tl *e = sp_tl_of(slot); if (e->ph) sp_pc_add(e, 0, 0);
    if (e->rc && --e->rc) return;
    if (e->p && e->drawn) return;   // the GPU's content, which only this slot holds: it stays for the next place that maps that memory
    if (e->p) sp_ph_del(slot);
    *e = (sp_tl){0}; sp_slot_free(slot);
}
// A physical pool's Finalize (hle_nvn.c): the shared slots of its memory leave the table at once, so a new pool over the same memory never finds one; a place still binding one moves off at its next look
void mtl_sp_phys_gone(const void *mem, uint64_t size) {
    uintptr_t lo = (uintptr_t)mem, hi = lo + size;
    pthread_mutex_lock(&g_sp_mu);
    for (size_t i = 0; g_sp_phn && i < g_sp_phcap; i++) {
        uint32_t e = g_sp_ph[i]; if (!e) continue;
        sp_tl *t = &g_sp_tl[e - 1]; if (t->p < lo || t->p >= hi) continue;
        sp_ph_del(e - 1); i--;   // the deletion may shift a later entry into slot i
        if (t->rc) { t->p = 0; t->drawn = 0; } else { *t = (sp_tl){0}; sp_slot_free(e - 1); }
    }
    pthread_mutex_unlock(&g_sp_mu);
}
// The binds gathered since the last queue submission (vk_sparse_flush sends them as one batch). A tile bound twice in one batch is not allowed: the batch is sent first (sp_bind)
typedef struct { VkImage img; int opaque; VkSparseImageMemoryBind b; VkSparseMemoryBind o; } sp_bind_t;
static sp_bind_t *g_spb; static size_t g_nspb, g_capspb; static uint32_t g_sp_batch = 1;
static VkSemaphore g_sp_sem; static uint64_t g_sp_val;   // the binds' timeline semaphore: batch n signals n, each waiting for the one before (batches may complete out of order)
static int spb_cmp(const void *a, const void *b) { const sp_bind_t *x = a, *y = b; return x->img < y->img ? -1 : x->img > y->img ? 1 : x->opaque - y->opaque; }
static void sp_send(void) {   // caller holds g_sp_mu
    if (!g_nspb) return;
    if (!g_sp_sem) { VkSemaphoreTypeCreateInfo ti = {VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO, NULL, VK_SEMAPHORE_TYPE_TIMELINE, 0}; VkSemaphoreCreateInfo si = {VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO, &ti};
        VKCHK(vkCreateSemaphore(V.dev, &si, NULL, &g_sp_sem)); }
    qsort(g_spb, g_nspb, sizeof *g_spb, spb_cmp);   // each image's binds together (one VkSparseImageMemoryBindInfo, one VkSparseImageOpaqueMemoryBindInfo)
    VkSparseImageMemoryBind *ib = malloc(g_nspb * sizeof *ib); VkSparseMemoryBind *ob = malloc(g_nspb * sizeof *ob);
    VkSparseImageMemoryBindInfo *ii = malloc(g_nspb * sizeof *ii); VkSparseImageOpaqueMemoryBindInfo *oi = malloc(g_nspb * sizeof *oi); uint32_t nib = 0, nob = 0, nii = 0, noi = 0;
    if (!ib || !ob || !ii || !oi) die("[vk] sparse bind batch");
    for (size_t i = 0; i < g_nspb; i++) {
        const sp_bind_t *e = &g_spb[i];
        if (e->opaque) { if (!noi || oi[noi - 1].image != e->img) oi[noi++] = (VkSparseImageOpaqueMemoryBindInfo){e->img, 0, ob + nob}; ob[nob++] = e->o; oi[noi - 1].bindCount++; }
        else { if (!nii || ii[nii - 1].image != e->img) ii[nii++] = (VkSparseImageMemoryBindInfo){e->img, 0, ib + nib}; ib[nib++] = e->b; ii[nii - 1].bindCount++; }
    }
    uint64_t wv = g_sp_val, sv = g_sp_val + 1;
    VkTimelineSemaphoreSubmitInfo ts = {VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO, NULL, wv ? 1u : 0u, &wv, 1, &sv};
    VkBindSparseInfo bi = {VK_STRUCTURE_TYPE_BIND_SPARSE_INFO, &ts, wv ? 1u : 0u, &g_sp_sem, 0, NULL, noi, oi, nii, ii, 1, &g_sp_sem};
    extern pthread_mutex_t g_vk_queue_mu;
    pthread_mutex_lock(&g_vk_queue_mu); VkResult r = vkQueueBindSparse(V.sq, 1, &bi, VK_NULL_HANDLE); pthread_mutex_unlock(&g_vk_queue_mu);
    if (r == VK_ERROR_DEVICE_LOST) die("[vk] vkQueueBindSparse: device lost");
    if (r) VKRED_ONCE(151, "vkQueueBindSparse failed (%d): %u tile binds and %u tail binds of this batch did not happen", r, nib, nob);
    else g_sp_val = sv;
    { static uint64_t nb; if (!nb++) VKLOG("sparse binds: first batch, %u tile binds and %u opaque (tail) binds over %u + %u images", nib, nob, nii, noi); }
    free(ib); free(ob); free(ii); free(oi); g_nspb = 0; g_sp_batch++;
}
static void sp_bind(struct vksparse *s, uint32_t tile, const sp_bind_t *b) {   // caller holds g_sp_mu; tile: its index (y × per_layer + i), ~0u for an opaque bind
    if (tile != ~0u) { if (s->bseq[tile] == g_sp_batch) sp_send(); s->bseq[tile] = g_sp_batch; }
    if (g_nspb == g_capspb) { g_capspb = g_capspb ? 2 * g_capspb : 1024; if (!(g_spb = realloc(g_spb, g_capspb * sizeof *g_spb))) die("[vk] sparse bind list"); }
    g_spb[g_nspb++] = *b;
}
static void sp_purge(VkImage img) { size_t k = 0; for (size_t i = 0; i < g_nspb; i++) if (g_spb[i].img != img) g_spb[k++] = g_spb[i]; g_nspb = k; }   // caller holds g_sp_mu: an image dropped before its binds went
int vk_sparse_flush(VkSemaphore *sem, uint64_t *val) {
    if (!V.has_sparse) return 0;
    pthread_mutex_lock(&g_sp_mu); sp_send(); int r = g_sp_val != 0; if (r) { *sem = g_sp_sem; *val = g_sp_val; } pthread_mutex_unlock(&g_sp_mu);
    return r;   // (every submission after the first batch waits for the newest value: a wait already met costs nothing, and none may run ahead of a bind its draws read)
}
void vk_sparse_lost(void) { pthread_mutex_lock(&g_sp_mu); g_sp_lost++; pthread_mutex_unlock(&g_sp_mu); }   // (gfx_vk_core.c: a dropped submission's tile uploads; each image uploads its tiles again at its next look)
// The GPU writes t's image (an attachment, a written image, a copy or clear destination): the shared slots it binds now hold what the memory does not. Only the first write after a mapping change walks the tiles
void vk_sparse_gpu_wrote(vkimg_t *im) {
    struct vksparse *s = im ? im->sp : NULL; if (!s || !s->undrawn) return;
    pthread_mutex_lock(&g_sp_mu); s->undrawn = 0;
    for (size_t i = 0, n = (size_t)s->layers * s->per_layer * s->nsub; i < n; i++) if (s->slot[i]) { sp_tl *e = sp_tl_of(s->slot[i] - 1); if (e->p) e->drawn = 1; }
    pthread_mutex_unlock(&g_sp_mu);
}
static unsigned layers_of(const nvn_texdesc *d);
// The sparse-resident image of t, or NULL when it does not qualify: a 2D texture or 2D array in a sparse pool, laid out in tiles whose shape is the device's (vkGetPhysicalDeviceSparseImage-
// FormatProperties), each tiled level whole tiles, the device's tail from NVN's first tail level, and SP_MIN or more made whole. gpu: a texture the GPU writes (render target, image): its
// tiles get no upload, and it takes its attachment and storage usages up front, as nvn_mtl.m's do. DM_VK_SPARSE_TILES=0 (start-up): none (the degraded path)
static vkimg_t *sparse_make(const tex_t *t, const vkfmt_t *f, int gpu) {
    static int on = -1; if (on < 0) { const char *e = getenv("DM_VK_SPARSE_TILES"); on = !e || atoi(e); }
    if (!on || !V.has_sparse || !t->pool || !t->mem || !nvn_sparse_pool(t->pool) || (t->d.target != NVN_TEXTURE_TARGET_2D && t->d.target != 4) || t->d.samples > 1 || f->dec || f->depth || f->stencil) return NULL;
    uint32_t tw, th, td; int n = nvn_tex_sparse_tiles(&t->d, &tw, &th, &td);
    if (n <= 0 || n > 15 || td != 1 || t->d.levels > 16 || t->L.size < SP_MIN || !f->bpb) return NULL;
    uint32_t levels = t->d.levels ? t->d.levels : 1, layers = layers_of(&t->d), w = t->d.width ? t->d.width : 1, h = t->d.height ? t->d.height : 1;
    VkFormatProperties fp; vkGetPhysicalDeviceFormatProperties(V.pd, f->vk, &fp);
    VkImageUsageFlags u = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    if (!f->compressed && (fp.optimalTilingFeatures & VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT)) u |= VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
    if ((t->d.flags & NVN_TEXTURE_FLAGS_IMAGE) && (fp.optimalTilingFeatures & VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT)) u |= VK_IMAGE_USAGE_STORAGE_BIT;
    VkSparseImageFormatProperties sf[4]; uint32_t nsf = 4; vkGetPhysicalDeviceSparseImageFormatProperties(V.pd, f->vk, VK_IMAGE_TYPE_2D, VK_SAMPLE_COUNT_1_BIT, u, VK_IMAGE_TILING_OPTIMAL, &nsf, sf);
    int k = -1; for (uint32_t i = 0; i < nsf; i++) if (sf[i].aspectMask & VK_IMAGE_ASPECT_COLOR_BIT) k = (int)i;
    const promo_t *pr = NULL; VkFormat vf = f->vk; uint32_t nx = 1, ny = 1, dtw = tw, dth = th;
    if ((k < 0 || sf[k].imageGranularity.width != tw || sf[k].imageGranularity.height != th || sf[k].imageGranularity.depth != 1 || (sf[k].flags & VK_SPARSE_IMAGE_FORMAT_NONSTANDARD_BLOCK_SIZE_BIT)) &&
        (pr = promo_for(f->vk, u, tw, th, &nx, &ny, &dtw, &dth))) {   // in its exact wider twin ("promoted formats" above)
        vf = pr->to; static int nl; if (nl++ < 16) VKLOG("sparse texture %ux%ux%u nvn format %#x (vk %d, usage %#x%s): no sparse tiles of NVN's %ux%u for its format here; sparse-resident in format %d, its exact wider twin: "
            "each 64 KB tile the game maps is %ux%u device tiles of %ux%u (twice the memory of what is mapped, and of sparse address space)", w, h, layers, t->d.format, f->vk, u, gpu ? ", GPU-written" : "", tw, th, vf, nx, ny, dtw, dth); }
    else if (k < 0 || sf[k].imageGranularity.width != tw || sf[k].imageGranularity.height != th || sf[k].imageGranularity.depth != 1 || (sf[k].flags & VK_SPARSE_IMAGE_FORMAT_NONSTANDARD_BLOCK_SIZE_BIT)) {
        // none at all for the format with this usage (Mario Kart World's 16384x16384 R8 textures, GPU-written, made whole at 256 MB each in run 5): what sampling and copies alone would get
        static int nl; if (nl++ < 16) { char alt[96] = "";
            if (k < 0) { VkSparseImageFormatProperties s2[4]; uint32_t n2 = 4, j = 4; vkGetPhysicalDeviceSparseImageFormatProperties(V.pd, f->vk, VK_IMAGE_TYPE_2D, VK_SAMPLE_COUNT_1_BIT,
                             VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT, VK_IMAGE_TILING_OPTIMAL, &n2, s2);
                         for (uint32_t i = 0; i < n2; i++) if (s2[i].aspectMask & VK_IMAGE_ASPECT_COLOR_BIT) j = i;
                         if (j < 4) snprintf(alt, sizeof alt, "; sampled and copied alone it would get %ux%u", s2[j].imageGranularity.width, s2[j].imageGranularity.height);
                         else snprintf(alt, sizeof alt, "; none sampled and copied alone either"); }
            VKLOG("sparse texture %ux%ux%u nvn format %#x (vk %d, usage %#x%s): the device's tile (%ux%u) is not NVN's %ux%u, made whole%s", w, h, layers, t->d.format, f->vk, u, gpu ? ", GPU-written" : "",
                  k >= 0 ? sf[k].imageGranularity.width : 0, k >= 0 ? sf[k].imageGranularity.height : 0, tw, th, alt); }
        return NULL; }
    // The sparse address space (sparseAddressSpaceSize, 4 GB on the Adreno 840) holds every sparse image at its full size: a sampled one that does not fit starts at a coarser level (first;
    // views of finer levels start at its level 0, vkimg_t lv0), a GPU-written one needs its level 0 and is not made
    VkImageCreateInfo ci; VkImage img = VK_NULL_HANDLE; VkMemoryRequirements req; uint32_t first = 0;
    for (;; first++) {
        uint32_t fw = w >> first ? w >> first : 1, fh = h >> first ? h >> first : 1;
        ci = (VkImageCreateInfo){VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, NULL, VK_IMAGE_CREATE_SPARSE_BINDING_BIT | VK_IMAGE_CREATE_SPARSE_RESIDENCY_BIT | (V.has_sparse_alias ? VK_IMAGE_CREATE_SPARSE_ALIASED_BIT : 0) |
            VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT | (f->compressed ? VK_IMAGE_CREATE_BLOCK_TEXEL_VIEW_COMPATIBLE_BIT | VK_IMAGE_CREATE_EXTENDED_USAGE_BIT : 0) | (layers >= 6 && layers % 6 == 0 && fw == fh ? VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT : 0),
            VK_IMAGE_TYPE_2D, vf, {fw, fh, 1}, levels - first, layers, VK_SAMPLE_COUNT_1_BIT, VK_IMAGE_TILING_OPTIMAL, u, VK_SHARING_MODE_EXCLUSIVE, 0, NULL, VK_IMAGE_LAYOUT_UNDEFINED};
        VkResult r = vkCreateImage(V.dev, &ci, NULL, &img);
        if (r) { VKRED_ONCE(152, "sparse image %ux%ux%u format %d refused (%d): made whole", fw, fh, layers, vf, r); return NULL; }
        vkGetImageMemoryRequirements(V.dev, img, &req);
        // an image in its format's wider twin takes twice the space for the memory it saves (16384x16384 R8: 682 MB, where an 8192x8192 RG16 GPU-written one takes 341 MB and saves about as much):
        // it leaves DM_VK_SPARSE_PROMO_RESERVE_MB (start-up, default 1024) to the images resident in their own format. Run 11 of Mario Kart World had 2,183 MB of them, of 4,095
        // (gfx_vk_budget.c vk_sparse_fits: DM_VK_SPARSE_SPACE_MB, start-up, sets the space instead of the device's, to see whether the driver takes more; tests/test_vk_budget.py replays run 13)
        uint64_t space = vk_sparse_space(V.props.limits.sparseAddressSpaceSize);
        if (vk_sparse_fits(g_sp_vaddr, req.size, pr != NULL, V.props.limits.sparseAddressSpaceSize)) break;
        vkDestroyImage(V.dev, img, NULL); img = VK_NULL_HANDLE;
        if (gpu || first + 1 >= (uint32_t)n) { static int nl; if (nl++ < 8) VKLOG("[RED] sparse texture %ux%ux%u nvn format %#x%s: past the sparse address space (%llu of %llu MB taken%s), %s - a platform limit", w, h, layers, t->d.format,
            gpu ? " (GPU-written)" : "", (unsigned long long)(g_sp_vaddr >> 20), (unsigned long long)(space >> 20), pr ? ", its wider twin leaving DM_VK_SPARSE_PROMO_RESERVE_MB to the others" : "",
            gpu ? "not sparse-resident" : "no coarser tiled level fits"); return NULL; }
    }
    if (g_sp_type == UINT32_MAX) g_sp_type = vk_mem_type(req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
    VkSparseImageMemoryRequirements mr[4]; uint32_t nmr = 4; vkGetImageSparseMemoryRequirements(V.dev, img, &nmr, mr);
    int kc = -1, km = -1; for (uint32_t i = 0; i < nmr; i++) { if (mr[i].formatProperties.aspectMask & VK_IMAGE_ASPECT_COLOR_BIT) kc = (int)i; if (mr[i].formatProperties.aspectMask & VK_IMAGE_ASPECT_METADATA_BIT) km = (int)i; }
    // the slots are 64 KB at 64 KB: the device's alignment must divide that, its memory type be the blocks' one, and its tail begin where NVN's does (with no tail, past the last level)
    const char *why = g_sp_type == UINT32_MAX || !(req.memoryTypeBits >> g_sp_type & 1) ? "its memory types" : req.alignment > SP_TILE || SP_TILE % req.alignment ? "its alignment" : kc < 0 ? "no colour requirement" :
                      (uint32_t)n < levels ? (mr[kc].imageMipTailFirstLod != (uint32_t)n - first ? "its mip tail's first level" : NULL) : (mr[kc].imageMipTailFirstLod < levels - first ? "its mip tail's first level" : NULL);
    struct vksparse *s = why ? NULL : calloc(1, sizeof *s);
    if (s) {   // NVN's tiled levels must be whole tiles, as the device's are (nvn_mtl.m sparse_make)
        uint32_t kk = 0; s->layer_size = t->L.layer_size; s->ntiled = (uint32_t)n; s->levels = levels; s->layers = layers; s->first = first;
        for (uint32_t l = 0; l < first; l++) s->loff[l] = nvn_tex_level_offset(&t->d, l);   // (levels finer than its first: their offsets only, sp_range skips them)
        for (uint32_t l = first; l < s->ntiled && !why; l++) {
            uint32_t lw = w >> l ? w >> l : 1, lh = h >> l ? h >> l : 1;
            s->tiles_x[l] = (lw + tw - 1) / tw; s->tiles_y[l] = (lh + th - 1) / th; s->loff_t[l] = kk; kk += s->tiles_x[l] * s->tiles_y[l];
            s->loff[l] = nvn_tex_level_offset(&t->d, l);
            uint64_t next = l + 1 < levels ? nvn_tex_level_offset(&t->d, l + 1) : s->layer_size;
            if (next - s->loff[l] != (uint64_t)s->tiles_x[l] * s->tiles_y[l] * SP_TILE) why = "a level that is not whole tiles";
        }
        s->tail_at = kk; s->per_layer = kk + (s->ntiled < levels); s->tail_off = s->ntiled < levels ? nvn_tex_level_offset(&t->d, s->ntiled) : s->layer_size;
    }
    if (why) { free(s); vkDestroyImage(V.dev, img, NULL); static int nl; if (nl++ < 4) VKLOG("sparse texture %ux%ux%u nvn format %#x: %s differ from what the tiles need, made whole", w, h, layers, t->d.format, why); return NULL; }
    s->img = img; s->gpu = gpu; s->tw = tw; s->th = th; s->w = w; s->h = h; int bw, bh; s->bpb = (uint32_t)nvn_fmt_bpb(t->d.format, &bw, &bh); s->bw = (uint32_t)bw; s->bh = (uint32_t)bh;
    s->pr = pr; s->nx = nx; s->nsub = nx * ny; s->dtw = dtw; s->dth = dth;
    if (!(s->slot = calloc((size_t)layers * s->per_layer * s->nsub, sizeof *s->slot)) || !(s->bseq = calloc((size_t)layers * s->per_layer, sizeof *s->bseq)) || !(s->tup = calloc(layers, 1)) || !(s->tq = calloc((size_t)layers * s->per_layer, 1))) die("[vk] sparse tile table");
    pthread_mutex_lock(&g_sp_mu);
    // each layer's tail (and the metadata, should the device ask for any) bound whole now, to memory of its own: the tail is uploaded when the game maps its storage
    for (int pass = 0; pass < 2; pass++) { int q = pass ? km : (s->ntiled < levels ? kc : -1); if (q < 0) continue;
        int single = (mr[q].formatProperties.flags & VK_SPARSE_IMAGE_FORMAT_SINGLE_MIPTAIL_BIT) || pass; VkDeviceSize al = (mr[q].imageMipTailSize + req.alignment - 1) / req.alignment * req.alignment;
        VkMemoryAllocateInfo ai = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, NULL, al * (single ? 1 : layers), g_sp_type}; VkDeviceMemory m;
        if (!al || vkAllocateMemory(V.dev, &ai, NULL, &m)) { VKRED_ONCE(153, "sparse image: no memory for its %s (%llu bytes)", pass ? "metadata" : "mip tails", (unsigned long long)ai.allocationSize); continue; }
        if (pass) s->meta_mem = m; else s->tail_mem = m;
        for (uint32_t y = 0; y < (single ? 1u : layers); y++) { sp_bind_t b = {img, 1}; b.o = (VkSparseMemoryBind){mr[q].imageMipTailOffset + y * mr[q].imageMipTailStride, mr[q].imageMipTailSize, m, y * al, pass ? VK_SPARSE_MEMORY_BIND_METADATA_BIT : 0};
            sp_bind(s, ~0u, &b); } }
    pthread_mutex_unlock(&g_sp_mu);
    pthread_mutex_lock(&g_sp_mu); s->vsize = req.size; g_sp_vaddr += req.size; g_sp_nimg++; pthread_mutex_unlock(&g_sp_mu);
    if (gpu && !V.has_sparse_alias) { static int said; if (!said++) VKLOG("sparse textures the GPU writes, without sparseResidencyAliased: a physical page mapped at several places gets a tile at each, so what the GPU draws "
        "at one place is not seen at the others (they cannot stay coherent) - a platform limit"); }
    vkimg_t *im = calloc(1, sizeof *im);
    im->w = w >> first ? w >> first : 1; im->h = h >> first ? h >> first : 1; im->d = 1; im->layers = layers; im->levels = levels - first; im->lv0 = first; im->samples = 1; im->target = t->d.target; im->nvn_format = t->d.format;
    im->fmt = vf; im->vtype = t->d.target == 4 ? VK_IMAGE_VIEW_TYPE_2D_ARRAY : VK_IMAGE_VIEW_TYPE_2D; im->aspect = VK_IMAGE_ASPECT_COLOR_BIT; im->usage = u; im->cflags = ci.flags;
    im->promo = pr ? (uint8_t)(pr - PROMO + 1) : 0;
    im->img = img; im->layout = VK_IMAGE_LAYOUT_UNDEFINED; im->sp = s;
    static int nl; if (nl++ < 20) VKLOG("sparse texture %ux%ux%u nvn format %#x levels %u: sparse-resident%s%s from level %u, %u tiles a layer, tail from level %u (%llu MB made whole; sparse address space %llu of %llu MB)", w, h, layers,
                                        t->d.format, levels, gpu ? " (GPU-written)" : "", pr ? " in its wider twin" : "", first, s->per_layer, s->ntiled, (unsigned long long)(t->L.size >> 20), (unsigned long long)(g_sp_vaddr >> 20),
                                        (unsigned long long)(V.props.limits.sparseAddressSpaceSize >> 20));
    return im;
}
static void sparse_drop(vkimg_t *im) {   // the image is dropped (img_drop, g_img_mu held): its slots go back once the GPU is past, its tails' memory with the image
    struct vksparse *s = im->sp;
    pthread_mutex_lock(&g_sp_mu); sp_purge(s->img);
    for (size_t i = 0, n = (size_t)s->layers * s->per_layer * s->nsub; i < n; i++) if (s->slot[i]) sp_release(s->slot[i] - 1);
    pthread_mutex_unlock(&g_sp_mu);
    pthread_mutex_lock(&g_sp_mu); g_sp_vaddr -= s->vsize; g_sp_nimg--; pthread_mutex_unlock(&g_sp_mu);
    if (s->tail_mem) vk_defer_free_buffer(VK_NULL_HANDLE, s->tail_mem); if (s->meta_mem) vk_defer_free_buffer(VK_NULL_HANDLE, s->meta_mem);
    free(s->slot); free(s->bseq); free(s->tup); free(s->tq); free(s->todo); free(s); im->sp = NULL;
}
// One look at an image's tiles: the binds, the uploads into the upload command buffer, what remains for the next look
enum { SP_BUDGET = 256 };   // tiles uploaded in one submission (16 MB of its scratch memory): a look that would take more leaves the rest to the next ones (DM_VK_SPARSE_UPLOADS)
static uint64_t g_sp_budget_sub; static uint32_t g_sp_budget;
typedef struct { vkimg_t *im; struct vksparse *s; const tex_t *t; uint8_t *tile, *tail, *nar; uint64_t up, unbound, stamp; int force; VkCommandBuffer cb; } sp_look;   // nar: a promoted tile's narrower texels
static int sp_budget(void) {   // caller holds g_sp_mu: one more tile upload may be made in this submission
    static uint32_t cap; if (!cap) { const char *e = getenv("DM_VK_SPARSE_UPLOADS"); cap = e && atoi(e) > 0 ? (uint32_t)atoi(e) : SP_BUDGET; }
    if (g_sp_budget_sub != vk_sub_serial()) { g_sp_budget_sub = vk_sub_serial(); g_sp_budget = cap; }
    if (!g_sp_budget) return 0; g_sp_budget--; return 1;
}
static void sp_todo(struct vksparse *s, uint32_t tile) {
    if (s->tq[tile]) return; s->tq[tile] = 1;
    if (s->ntodo == s->captodo) { s->captodo = s->captodo ? 2 * s->captodo : 256; if (!(s->todo = realloc(s->todo, s->captodo * sizeof *s->todo))) die("[vk] sparse tile list"); }
    s->todo[s->ntodo++] = tile;
}
static VkCommandBuffer sp_cb(sp_look *c) {   // the upload command buffer, ordered after the reads of earlier submissions once per upload buffer, the image in GENERAL
    extern uint64_t g_up_gen; VkCommandBuffer cb = vk_upload_cmd();
    if (c->s->up_gen != g_up_gen) { c->s->up_gen = g_up_gen; vk_img_barrier_all(cb); }
    if (c->im->layout != VK_IMAGE_LAYOUT_GENERAL) vk_tex_prepare(cb, c->im);
    return c->cb = cb;
}
// sp_tile for an image in its format's wider twin (vksparse nsub > 1, "promoted formats" above): tile i of layer y, (tx, ty) of level l, at guest address a, its mapping state st. Its nsub
// device tiles are bound together, in one batch, each to a slot of its own (none shared: sharing takes sparseResidencyAliased, which the Adreno 840 lacks), and its content deswizzled in
// the narrower format and widened into the copy. Caller holds g_sp_mu
static void sp_tile_wide(sp_look *c, uint32_t y, uint32_t i, uint32_t l, uint32_t tx, uint32_t ty, uint64_t a, int st) {
    struct vksparse *s = c->s; const tex_t *t = c->t; uint32_t ti = y * s->per_layer + i, *sl = &s->slot[(size_t)ti * s->nsub];
    uint32_t lw = s->w >> l ? s->w >> l : 1, lh = s->h >> l ? s->h >> l : 1, x0 = tx * s->tw, y0 = ty * s->th, ew = s->tw < lw - x0 ? s->tw : lw - x0, eh = s->th < lh - y0 ? s->th : lh - y0;
    sp_bind_t bd[16]; uint32_t in[16], nin = 0;   // the device tiles inside the level
    for (uint32_t k = 0; k < s->nsub && k < 16; k++) { uint32_t x, yy, w, h; if (!promo_sub(tx, ty, s->tw, s->th, s->nx, s->dtw, s->dth, k, lw, lh, &x, &yy, &w, &h)) continue;
        bd[nin] = (sp_bind_t){s->img, 0}; bd[nin].b = (VkSparseImageMemoryBind){{VK_IMAGE_ASPECT_COLOR_BIT, l - s->first, y}, {(int32_t)x, (int32_t)yy, 0}, {w, h, 1}, VK_NULL_HANDLE, 0, 0}; in[nin++] = k; }
    if (!(st & 1)) {   // unmapped: unbound, its slots free once the GPU is past
        if (sl[0]) { for (uint32_t j = 0; j < nin; j++) if (sl[in[j]]) { sp_release(sl[in[j]] - 1); sl[in[j]] = 0; } for (uint32_t j = 0; j < nin; j++) sp_bind(s, j ? ~0u : ti, &bd[j]); c->unbound++; }
        return; }
    if (sl[0]) {   // bound: the GPU's own content, or the game's unchanged since its upload
        if (s->gpu || (!c->force && !(st & 2))) return;
        if (!sp_budget()) { sp_todo(s, ti); return; }
    } else {
        if (!s->gpu && !sp_budget()) { sp_todo(s, ti); return; }
        for (uint32_t j = 0; j < nin; j++) { uint32_t slot; if (!sp_slot(&slot)) { while (j--) { sp_release(sl[in[j]] - 1); sl[in[j]] = 0; } return; }
            sl[in[j]] = slot + 1; bd[j].b.memory = g_sp_blk[slot / SP_BLOCK_TILES]; bd[j].b.memoryOffset = (VkDeviceSize)(slot % SP_BLOCK_TILES) * SP_TILE; }
        for (uint32_t j = 0; j < nin; j++) sp_bind(s, j ? ~0u : ti, &bd[j]);   // (one check of the batch for the NVN tile: its device tiles are distinct)
        s->undrawn = 1;
        if (s->gpu) { c->up++; return; }   // bound for the GPU to write
    }
    nvn_sparse_protect_tile(t->pool, a, SP_TILE); nvn_sparse_copy(t->pool, c->tile, a, SP_TILE);   // (protected before it is read, as sp_tile's)
    vk_shadow_note_sparse(t->pool, a, SP_TILE, c->stamp);
    if (!c->nar && !(c->nar = malloc(SP_TILE))) return;
    uint32_t wb, rows; size_t tpitch = (size_t)(s->tw / s->bw) * s->bpb;
    if (nvn_deswizzle_tile(&t->d, l, tx, ty, c->tile, c->nar, tpitch, &wb, &rows)) { VKRED_ONCE(155, "sparse tile of nvn format %#x not decoded", t->d.format); return; }
    VkBuffer sb; VkDeviceSize so; uint8_t *dst = vk_scratch((VkDeviceSize)SP_TILE * 2, 16, &sb, &so); promo_widen(s->pr, c->nar, (size_t)s->tw * s->th, dst);
    if (wb && rows) { VkBufferImageCopy r = {so, s->tw, 0, {VK_IMAGE_ASPECT_COLOR_BIT, l - s->first, y, 1}, {(int32_t)x0, (int32_t)y0, 0}, {ew, eh, 1}}; vkCmdCopyBufferToImage(sp_cb(c), sb, s->img, VK_IMAGE_LAYOUT_GENERAL, 1, &r); }
    c->up++; g_sp_ups++;
}
// Tile i of layer y (the tail at s->tail_at): the game has mapped all its pages or not, and the image's binding follows; its content is uploaded when it is newly bound or its pages were written.
// Caller holds g_sp_mu
static void sp_tile(sp_look *c, uint32_t y, uint32_t i) {
    struct vksparse *s = c->s; const tex_t *t = c->t; uint32_t ti = y * s->per_layer + i;
    int is_tail = i == s->tail_at; uint32_t l = 0, tx = 0, ty = 0;
    if (!is_tail) { l = s->first; while (l + 1 < s->ntiled && i >= s->loff_t[l + 1]) l++; uint32_t j = i - s->loff_t[l]; tx = j % s->tiles_x[l]; ty = j / s->tiles_x[l]; }
    uint64_t a = (uint64_t)(uintptr_t)t->mem + y * s->layer_size + (is_tail ? s->tail_off : s->loff[l] + (uint64_t)(i - s->loff_t[l]) * SP_TILE), n = is_tail ? s->layer_size - s->tail_off : SP_TILE;
    int st = nvn_sparse_tile_state(t->pool, a, n, s->gen, s->stamp); if (st < 0) st = 0;
    if (is_tail) {   // bound for good (sparse_make): uploaded when the game has mapped all of it and it is new or changed
        if (!(st & 1) || s->gpu || (!(st & 2) && !c->force && s->tup[y])) return;
        if (!sp_budget()) { sp_todo(s, ti); return; }
        nvn_sparse_protect_tile(t->pool, a, n); uint8_t *tb = realloc(c->tail, n); if (!tb) return; c->tail = tb;
        nvn_sparse_copy(t->pool, c->tail, a, n); VkCommandBuffer cb = sp_cb(c);
        for (uint32_t tl = s->ntiled; tl < s->levels; tl++) {
            uint32_t lw = s->w >> tl ? s->w >> tl : 1, lh = s->h >> tl ? s->h >> tl : 1; size_t pitch = (size_t)((lw + s->bw - 1) / s->bw) * s->bpb, rows = (lh + s->bh - 1) / s->bh;
            VkBuffer sb; VkDeviceSize so; uint8_t *dst = vk_scratch(pitch * rows * (s->pr ? 2 : 1), 16, &sb, &so), *nar = s->pr ? malloc(pitch * rows) : dst;   // (promoted: widened into the copy)
            if (!nar || nvn_deswizzle_from(&t->d, tl, y, c->tail, y * s->layer_size + s->tail_off, nar, pitch)) { if (nar != dst) free(nar); VKRED_ONCE(154, "sparse tail level %u of nvn format %#x not decoded", tl, t->d.format); break; }
            if (nar != dst) { promo_widen(s->pr, nar, (size_t)lw * lh, dst); free(nar); }
            VkBufferImageCopy r = {so, 0, 0, {VK_IMAGE_ASPECT_COLOR_BIT, tl - s->first, y, 1}, {0, 0, 0}, {lw, lh, 1}}; vkCmdCopyBufferToImage(cb, sb, s->img, VK_IMAGE_LAYOUT_GENERAL, 1, &r); }
        s->tup[y] = 1; c->up++; return;
    }
    if (s->nsub > 1) { sp_tile_wide(c, y, i, l, tx, ty, a, st); return; }   // (a promoted format's NVN tile: several device tiles)
    uint32_t lw = s->w >> l ? s->w >> l : 1, lh = s->h >> l ? s->h >> l : 1, x0 = tx * s->tw, y0 = ty * s->th, ew = s->tw < lw - x0 ? s->tw : lw - x0, eh = s->th < lh - y0 ? s->th : lh - y0;
    sp_bind_t bd = {s->img, 0}; bd.b = (VkSparseImageMemoryBind){{VK_IMAGE_ASPECT_COLOR_BIT, l - s->first, y}, {(int32_t)x0, (int32_t)y0, 0}, {ew, eh, 1}, VK_NULL_HANDLE, 0, 0};
    uint32_t *sl = &s->slot[ti];
    if (!(st & 1)) { if (*sl) { sp_bind(s, ti, &bd); sp_release(*sl - 1); *sl = 0; c->unbound++; } return; }   // unmapped: unbound (its slot free once the GPU is past)
    const uint8_t *p = V.has_sparse_alias && x0 + s->tw <= lw && y0 + s->th <= lh ? nvn_sparse_tile_phys(t->pool, a) : NULL;   // a tile of consecutive physical pages: shared (see sp_tl)
    int rebind = 0;
    if (*sl) {
        const sp_tl *e = sp_tl_of(*sl - 1);
        if (e->p != (uintptr_t)p || (p && e->fmt != t->d.format)) { sp_release(*sl - 1); *sl = 0; rebind = 1; }   // other memory behind the tile now: bound again below (one bind replaces the other)
        else if (s->gpu) { if (!V.has_sparse_alias && e->ph) { uintptr_t ph = (uintptr_t)nvn_sparse_tile_phys(t->pool, a); if (ph != e->ph) { g_sp_moved++; sp_pc_add(sp_tl_of(*sl - 1), ph, 1); } }   // (G3: counted)
            return; }   // the GPU's own writes are in it: a CPU write to its pages is not looked at
        else if (e->drawn ? !nvn_dirty_since(p, SP_TILE, e->stamp) : !c->force && !(p ? nvn_dirty_since(p, SP_TILE, e->stamp) : (st & 2))) return;   // unchanged since its content was taken
    }
    if (!*sl) {
        if (!s->gpu && !sp_budget()) { if (rebind) { sp_bind(s, ti, &bd); c->unbound++; } sp_todo(s, ti); return; }   // (a tile moved off its slot waits unbound for its upload)
        uint32_t slot = p ? sp_ph_find((uintptr_t)p, t->d.format) : ~0u; int fresh = slot == ~0u;
        if (fresh) { if (!sp_slot(&slot)) { if (rebind) { sp_bind(s, ti, &bd); c->unbound++; } return; } }
        else sp_tl_of(slot)->rc++;
        bd.b.memory = g_sp_blk[slot / SP_BLOCK_TILES]; bd.b.memoryOffset = (VkDeviceSize)(slot % SP_BLOCK_TILES) * SP_TILE; sp_bind(s, ti, &bd);
        *sl = slot + 1; s->undrawn = 1;
        if (!V.has_sparse_alias && x0 + s->tw <= lw && y0 + s->th <= lh) sp_pc_add(sp_tl_of(slot), (uintptr_t)nvn_sparse_tile_phys(t->pool, a), s->gpu);   // (G3: counted)
        if (fresh && p) { sp_tl *e = sp_tl_of(slot); e->p = (uintptr_t)p; e->fmt = t->d.format; e->stamp = nvn_dirty_now(); sp_ph_put(slot); }
        if (s->gpu) { if (fresh && p) nvn_dirty_protect_sparse(p, SP_TILE); c->up++; return; }   // bound for the GPU to write (a later CPU write to the memory shows at a sampled place)
        if (!fresh && sp_tl_of(slot)->drawn && !nvn_dirty_since(p, SP_TILE, sp_tl_of(slot)->stamp)) return;   // the GPU's content, newer than the memory's
    } else if (!sp_budget()) { sp_todo(s, ti); return; }
    // the content, protected before it is read as any packaged texture's, so a later CPU write shows (nvn_dirty_sparse_writes, sparse_look). A shared slot's memory is read at p, not
    // through the page table: the game may remap the place on its own thread meanwhile, and the slot must never take another memory's bytes under this one's key
    if (p) nvn_dirty_protect_sparse(p, SP_TILE); else nvn_sparse_protect_tile(t->pool, a, n);
    if (p) memcpy(c->tile, p, SP_TILE); else nvn_sparse_copy(t->pool, c->tile, a, SP_TILE);
    vk_shadow_note_sparse(t->pool, a, SP_TILE, c->stamp);   // (the physical pages behind it, protected: DM_VK_SHADOW)
    VkBuffer sb; VkDeviceSize so; uint8_t *dst = vk_scratch(SP_TILE, 16, &sb, &so); uint32_t wb, rows; size_t tpitch = (size_t)(s->tw / s->bw) * s->bpb;   // a tile's row of blocks
    if (nvn_deswizzle_tile(&t->d, l, tx, ty, c->tile, dst, tpitch, &wb, &rows)) { VKRED_ONCE(155, "sparse tile of nvn format %#x not decoded", t->d.format); return; }
    if (wb && rows) { VkBufferImageCopy r = {so, s->tw, 0, {VK_IMAGE_ASPECT_COLOR_BIT, l - s->first, y, 1}, {(int32_t)x0, (int32_t)y0, 0}, {ew, eh, 1}}; vkCmdCopyBufferToImage(sp_cb(c), sb, s->img, VK_IMAGE_LAYOUT_GENERAL, 1, &r); }
    if (p) sp_tl_of(*sl - 1)->stamp = c->stamp;   // taken before the protection above
    c->up++; g_sp_ups++;
}
static void sp_range(sp_look *c, uint64_t lo, uint64_t hi) {   // the tiles of the guest range [lo, hi)
    struct vksparse *s = c->s; uint64_t base = (uint64_t)(uintptr_t)c->t->mem, end = base + (uint64_t)s->layers * s->layer_size;
    if (lo < base) lo = base;
    if (hi > end) hi = end;
    for (uint64_t o = (lo - base) & ~(uint64_t)(SP_TILE - 1); lo < hi && base + o < hi;) {
        uint32_t y = (uint32_t)(o / s->layer_size); uint64_t in = o % s->layer_size;
        if (in >= s->tail_off) { if (s->ntiled < s->levels) sp_tile(c, y, s->tail_at); o += s->layer_size - in; continue; }   // the tail once, whatever part of it the range touches
        uint32_t l = 0; while (l + 1 < s->ntiled && in >= s->loff[l + 1]) l++;
        if (l >= s->first) sp_tile(c, y, s->loff_t[l] + (uint32_t)((in - s->loff[l]) / SP_TILE));   // (a level finer than the image's first: not in it)
        o += SP_TILE;
    }
}
static int sp_cmp(const void *a, const void *b) { uint64_t x = *(const uint64_t *)a, y = *(const uint64_t *)b; return x < y ? -1 : x > y; }
// The game's mappings into the image since the last look: tiles mapped now get a slot and their content, tiles unmapped give theirs back. All the tiles at the first look or when the pool's
// log has moved on past it, else those of the requests since (sorted and joined: a tile is handled once). whole 2: only the tiles bound now, for one a write to its pages has changed;
// 3: all of those again, their uploads having been dropped (vk_sparse_lost)
static void sparse_sync(vkimg_t *im, const tex_t *t, int whole) {
    struct vksparse *s = im->sp; enum { RANGES = 4096 }; uint64_t (*ranges)[2] = NULL; int nr = -1;
    uint64_t gen = nvn_sparse_gen(), stamp = nvn_dirty_now();
    if (whole == 1) s->seq = nvn_sparse_log_seq(t->pool);
    else if (whole >= 2) { s->wseen = nvn_dirty_sparse_writes(); s->wframe = g_tex_frame; }
    else { if (!(ranges = malloc(RANGES * sizeof *ranges))) die("[vk] sparse mapping ranges"); nr = nvn_sparse_log_read(t->pool, &s->seq, ranges, RANGES); if (!nr) { free(ranges); return; } }
    sp_look c = {.im = im, .s = s, .t = t, .force = whole == 3, .stamp = stamp}; if (!(c.tile = malloc(SP_TILE))) die("[vk] sparse tile copy");
    pthread_mutex_lock(&g_sp_mu);
    if (nr < 0) for (uint32_t y = 0; y < s->layers; y++) for (uint32_t i = 0; i < s->per_layer; i++) { if (whole < 2 || s->slot[((size_t)y * s->per_layer + i) * s->nsub] || (i == s->tail_at && s->tup[y])) sp_tile(&c, y, i); }
    else {
        qsort(ranges, (size_t)nr, sizeof ranges[0], sp_cmp);
        uint64_t lo = ranges[0][0], hi = ranges[0][1];
        for (int k = 1; k < nr; k++) { if (ranges[k][0] <= hi) { if (ranges[k][1] > hi) hi = ranges[k][1]; continue; } sp_range(&c, lo, hi); lo = ranges[k][0]; hi = ranges[k][1]; }
        sp_range(&c, lo, hi);
    }
    pthread_mutex_unlock(&g_sp_mu);
    free(c.tile); free(c.tail); free(c.nar); free(ranges);
    s->gen = gen; s->stamp = stamp; s->synced = 1;
    static uint64_t looks; if (c.up || c.unbound) { looks++; if (looks <= 20 || looks % 500 == 0) VKLOG("sparse texture %ux%ux%u nvn format %#x: %llu tiles bound or uploaded, %llu unbound%s; %llu tiles bound in all (%llu of them shared by physical memory), peak %llu (%llu MB)%s",
        s->w, s->h, s->layers, t->d.format, (unsigned long long)c.up, (unsigned long long)c.unbound, s->ntodo ? ", the rest at the next looks (upload budget)" : "", (unsigned long long)g_sp_live, (unsigned long long)g_sp_shared,
        (unsigned long long)g_sp_peak, (unsigned long long)(g_sp_peak * SP_TILE >> 20), sp_g3_text()); }
}
// A look when the image is looked up in a submission: what the budget left first, then a new mapping request in its pool, or (at most once a second, as it goes over every bound tile) a
// CPU write to a page behind a tile, which no game seen does to a tile it keeps mapped (none of 42,072 in Mario Kart World's first race, nvn_mtl.m sparse_look)
static void sparse_look(vkimg_t *im, const tex_t *t) {
    struct vksparse *s = im->sp;
    if (!s->synced) { if (im->layout != VK_IMAGE_LAYOUT_GENERAL) vk_tex_prepare(vk_upload_cmd(), im); sparse_sync(im, t, 1); s->lost = g_sp_lost; return; }
    if (s->ntodo) { sp_look c = {.im = im, .s = s, .t = t, .stamp = nvn_dirty_now()}; if (!(c.tile = malloc(SP_TILE))) die("[vk] sparse tile copy");
        pthread_mutex_lock(&g_sp_mu); uint32_t n = s->ntodo; uint32_t *td = s->todo; s->todo = NULL; s->ntodo = s->captodo = 0;
        for (uint32_t k = 0; k < n; k++) s->tq[td[k]] = 0;
        for (uint32_t k = 0; k < n; k++) sp_tile(&c, td[k] / s->per_layer, td[k] % s->per_layer);
        pthread_mutex_unlock(&g_sp_mu); free(td); free(c.tile); free(c.tail); free(c.nar); }
    if (s->lost != g_sp_lost) { s->lost = g_sp_lost; if (!s->gpu) { sparse_sync(im, t, 3); return; } }
    if (nvn_sparse_log_seq(t->pool) != s->seq) sparse_sync(im, t, 0);
    if (!s->gpu && nvn_dirty_sparse_writes() != s->wseen && g_tex_frame - s->wframe >= 60) sparse_sync(im, t, 2);
}
static void vk_tex_view_drop_all(vkimg_t *im); static uint64_t g_img_n, g_img_bytes, g_img_evicted;   // (below)
// ---------------- dense images of sparse textures that cannot be sparse-resident ----------------
// A sampled texture of a sparse pool that cannot be sparse-resident and is larger than DM_VK_SPARSE_DENSE_MB (default 32) made whole: the stock Adreno 840 driver (05/2026) gives no sparse
// tiles for any BC, ETC2 or ASTC format, and Mario Kart World keeps its terrain in 8192x8192x68 BC6H arrays (6.08 GB made whole) and ASTC 8x8 ones (1.45 GB), of which it maps 3-12 %. Such a
// texture gets a dense image from its finest level with a mapped page (its base), but no finer than the cap allows (its levels take at most DM_VK_SPARSE_DENSE_MB: 128 MB still ran that phone
// out of memory, 32 MB gave the first frames, measured with another port). Each level and layer is uploaded from the page table, unmapped pages as zero texels, when its mappings or pages
// change; the image is made again when a finer level gets mapped, and loses its finer levels when the finest mapped one is 2 or more levels coarser (dense_first). Views and targets of a
// level left out start at the image's level 0 (vkimg_t lv0): sampling with implicit LOD works unchanged, its coordinates being normalized. A platform limit, logged as one
// ---- dense images: the level choice (tests: tests/test_vk_switch2.py) ----
struct vkdense {
    uint32_t tl, layers, ntiled, per_layer, tail_at; uint64_t layer_size, tail_off, seq, sub, bytes;
    uint32_t loff_t[16]; uint64_t loff[16];   // per tiled level: its first unit in a layer, its offset in a layer (64 KB units, NVN's sparse tiles)
    uint8_t *mapped; uint32_t *cnt;           // per unit (a tile, or a layer's mip tail): every page mapped; per level and layer (level × layers + layer): its mapped units
    uint64_t *gen, *stamp; uint8_t *up;       // per level and layer: the page table generation and dirty stamp its upload saw, uploaded at all
};
static uint32_t dense_level_of(const struct vkdense *dn, uint32_t i) {   // the first level unit i of a layer holds (a tail: its first level)
    if (i == dn->tail_at && dn->ntiled < dn->tl) return dn->ntiled;
    uint32_t l = 0; while (l + 1 < dn->ntiled && i >= dn->loff_t[l + 1]) l++; return l;
}
static void dense_unit_range(const struct vkdense *dn, uint32_t u, uint64_t *o, uint64_t *n) {   // unit u's offset in the texture and its bytes
    uint32_t y = u / dn->per_layer, i = u % dn->per_layer, l = dense_level_of(dn, i);
    if (i == dn->tail_at && dn->ntiled < dn->tl) { *o = (uint64_t)y * dn->layer_size + dn->tail_off; *n = dn->layer_size - dn->tail_off; return; }
    uint64_t end = l + 1 < dn->ntiled ? dn->loff[l + 1] : dn->tail_off, a = dn->loff[l] + (uint64_t)(i - dn->loff_t[l]) * SP_TILE;
    *o = (uint64_t)y * dn->layer_size + a; *n = end - a < SP_TILE ? end - a : SP_TILE;
}
static void dense_set(struct vkdense *dn, uint32_t u, int m) {   // unit u found mapped (m) or not: the counts of its levels follow
    if (dn->mapped[u] == (uint8_t)m) return; dn->mapped[u] = (uint8_t)m;
    uint32_t y = u / dn->per_layer, i = u % dn->per_layer, l = dense_level_of(dn, i), l1 = i == dn->tail_at && dn->ntiled < dn->tl ? dn->tl : l + 1;
    for (; l < l1; l++) { if (m) dn->cnt[l * dn->layers + y]++; else dn->cnt[l * dn->layers + y]--; }
}
// The units of the texture's range [lo, hi) (texture offsets) looked at again through mapped(a, n): every page of the guest range [mem + a, +n) has memory behind it
static void dense_scan(struct vkdense *dn, uint64_t lo, uint64_t hi, int (*mapped)(const void *, uint64_t, uint64_t), const void *pool, uint64_t mem) {
    uint64_t total = (uint64_t)dn->layers * dn->layer_size; if (hi > total) hi = total;
    for (uint32_t u = 0, nu = dn->layers * dn->per_layer; u < nu; u++) { uint64_t o, n; dense_unit_range(dn, u, &o, &n);
        if (o + n > lo && o < hi) dense_set(dn, u, mapped(pool, mem + o, n)); }
}
static uint32_t dense_base(const struct vkdense *dn) {   // the finest level with a mapped unit in any layer (tl: none)
    for (uint32_t l = 0; l < dn->tl; l++) for (uint32_t y = 0; y < dn->layers; y++) if (dn->cnt[l * dn->layers + y]) return l;
    return dn->tl;
}
// The first level of the dense image: from the base (none mapped: the smallest level), no finer than levels c.. take at most cap bytes (lvbytes: a level's bytes, all layers; the smallest
// level always), and an image from cur kept while the base is no finer than it and less than 2 levels coarser (cur past the levels: none yet)
static uint32_t dense_first(const uint64_t *lvbytes, uint32_t levels, uint32_t base, uint64_t cap, uint32_t cur) {
    uint32_t c = levels - 1; uint64_t sum = lvbytes[c];
    while (c > 0 && sum + lvbytes[c - 1] <= cap) sum += lvbytes[--c];
    uint32_t want = base > c ? base : c; if (want > levels - 1) want = levels - 1;
    if (cur >= levels || cur < c || want < cur || want >= cur + 2) return want;
    return cur;
}
// ---- end of the dense level choice ----
static uint64_t dense_cap(void) { static uint64_t v; if (!v) { const char *e = getenv("DM_VK_SPARSE_DENSE_MB"); v = (uint64_t)(e && atoi(e) > 0 ? atoi(e) : 32) << 20; } return v; }
static uint64_t g_dense_bytes; static uint32_t g_dense_n;
static void dense_lvbytes(const tex_t *t, const vkfmt_t *f, uint32_t layers, uint64_t lv[16]) {
    uint32_t W = t->d.width ? t->d.width : 1, H = t->d.height ? t->d.height : 1, L = t->d.levels ? t->d.levels : 1, bw = f->bw ? f->bw : 1, bh = f->bh ? f->bh : 1;
    for (uint32_t l = 0; l < L && l < 16; l++) { uint32_t w = W >> l ? W >> l : 1, h = H >> l ? H >> l : 1;
        lv[l] = (uint64_t)layers * (f->dec ? (uint64_t)w * h * f->dbpp : (uint64_t)((w + bw - 1) / bw) * ((h + bh - 1) / bh) * f->bpb); }
}
static vkimg_t *dense_image(const tex_t *t, const vkfmt_t *f, uint32_t first) {   // the image of t's levels first..
    tex_t tt = *t; uint32_t W = t->d.width ? t->d.width : 1, H = t->d.height ? t->d.height : 1;
    tt.d.width = W >> first ? W >> first : 1; tt.d.height = H >> first ? H >> first : 1; tt.d.levels = (t->d.levels ? t->d.levels : 1) - first;
    vkimg_t *im = img_new(&tt, f); if (im) im->lv0 = first;
    return im;
}
static int dense_mapped(const void *pool, uint64_t a, uint64_t n) { return nvn_sparse_mapped(pool, a, n); }
static vkimg_t *dense_make(const tex_t *t, const vkfmt_t *f) {
    uint32_t tw, th, td; int n = nvn_tex_sparse_tiles(&t->d, &tw, &th, &td);
    if (!t->pool || !t->mem || (t->d.target != NVN_TEXTURE_TARGET_2D && t->d.target != 4) || t->d.samples > 1 || t->d.levels > 16 || !f->bpb || n <= 0 || n > 16) return NULL;
    struct vkdense *dn = calloc(1, sizeof *dn); if (!dn) return NULL;
    dn->tl = t->d.levels ? t->d.levels : 1; dn->layers = layers_of(&t->d); dn->ntiled = (uint32_t)n < dn->tl ? (uint32_t)n : dn->tl; dn->layer_size = t->L.layer_size ? t->L.layer_size : t->L.size;
    uint32_t k = 0;
    for (uint32_t l = 0; l < dn->ntiled; l++) { dn->loff[l] = nvn_tex_level_offset(&t->d, l); dn->loff_t[l] = k;
        uint64_t next = l + 1 < dn->tl ? nvn_tex_level_offset(&t->d, l + 1) : dn->layer_size; k += (uint32_t)((next - dn->loff[l] + SP_TILE - 1) / SP_TILE); }
    dn->tail_at = k; dn->per_layer = k + (dn->ntiled < dn->tl); dn->tail_off = dn->ntiled < dn->tl ? nvn_tex_level_offset(&t->d, dn->ntiled) : dn->layer_size;
    size_t nu = (size_t)dn->layers * dn->per_layer, nl = (size_t)dn->tl * dn->layers;
    if (!(dn->mapped = calloc(nu, 1)) || !(dn->cnt = calloc(nl, sizeof *dn->cnt)) || !(dn->gen = calloc(nl, 8)) || !(dn->stamp = calloc(nl, 8)) || !(dn->up = calloc(nl, 1))) die("[vk] dense image tables");
    dn->seq = nvn_sparse_log_seq(t->pool);   // (before the scan: a mapping meanwhile is looked at again)
    dense_scan(dn, 0, ~0ull, dense_mapped, t->pool, (uint64_t)(uintptr_t)t->mem);
    uint64_t lv[16]; dense_lvbytes(t, f, dn->layers, lv); uint32_t base = dense_base(dn), first = dense_first(lv, dn->tl, base, dense_cap(), ~0u);
    vkimg_t *im = dense_image(t, f, first);
    if (!im) { free(dn->mapped); free(dn->cnt); free(dn->gen); free(dn->stamp); free(dn->up); free(dn); return NULL; }
    im->dn = dn; for (uint32_t l = first; l < dn->tl; l++) dn->bytes += lv[l]; g_dense_bytes += dn->bytes; g_dense_n++;
    vk_vt_attach(im, t, f, dn->mapped, dn->per_layer, dn->tail_at);   // DM_VK_VT=1: its levels finer than the image's through a pool of tiles (gfx_vk_vt.c)
    static int nlog; if (nlog++ < 12) VKLOG("sparse texture %ux%ux%u nvn format %#x (%llu MB made whole): %s; a dense image from level %u of %u (the finest mapped %d, at most %llu MB: %.1f MB) - a platform limit",
        t->d.width, t->d.height, dn->layers, t->d.format, (unsigned long long)(t->L.size >> 20), V.has_sparse ? "no sparse residency for its format here" : "no sparse residency on this device", first, dn->tl,
        base < dn->tl ? (int)base : -1, (unsigned long long)(dense_cap() >> 20), dn->bytes / 1048576.0);
    return im;
}
static void dense_remake(vkimg_t *im, const tex_t *t, const vkfmt_t *f, uint32_t first) {   // the image again from level first (the replay thread, in a submission)
    vkimg_t *n = dense_image(t, f, first); if (!n) return;   // (the old one stays)
    struct vkdense *dn = im->dn; vk_tex_view_drop_all(im);
    if (im->sub) vk_defer_free_image_sub(im->img, im->sub); else vk_defer_free_image(im->img, im->mem);
    im->img = n->img; im->mem = n->mem; im->sub = n->sub; im->w = n->w; im->h = n->h; im->levels = n->levels; im->layout = n->layout; im->usage = n->usage; im->cflags = n->cflags;
    im->fixed_fmt = n->fixed_fmt; im->fmt2 = n->fmt2; uint32_t was = im->lv0; im->lv0 = first; free(n);
    uint64_t lv[16]; dense_lvbytes(t, f, dn->layers, lv); uint64_t b = 0; for (uint32_t l = first; l < dn->tl; l++) b += lv[l];
    pthread_mutex_lock(&g_img_mu); g_dense_bytes += b - dn->bytes; g_img_bytes += b - dn->bytes; pthread_mutex_unlock(&g_img_mu); dn->bytes = b;
    memset(dn->up, 0, (size_t)dn->tl * dn->layers);
    static int nl; if (nl++ < 24) VKLOG("sparse texture %ux%ux%u nvn format %#x: its dense image from level %u now (was %u; %.1f MB)", t->d.width, t->d.height, dn->layers, t->d.format, first, was, b / 1048576.0);
}
// An unmapped page of a sparse texture reads as zero texels on the Switch (and on Metal's placement sparse textures, and through a sparse-resident image here). Gathered through the page table
// it is zero bytes, which ASTC decodes as its error colour: an all-zero block's mode is reserved (opaque magenta on LDR formats). So the ASTC blocks of a level copied from the page table that
// are all zero become a void-extent block of colour 0 (block mode 0x1fc, LDR, no extent, RGBA 0): what the unmapped tile reads. BC formats decode zero blocks to zero colours already.
// Mario Kart World 1.8.0: its 8192x8192x68 ASTC 8x8 terrain arrays are dense images from their level 3 (no sparse residency for ASTC on the Adreno 840), uploaded level by level from the page table
static void astc_zero_void(const vkfmt_t *f, uint8_t *p, size_t n) {
    if (f->dec || f->bpb != 16 || f->vk < VK_FORMAT_ASTC_4x4_UNORM_BLOCK || f->vk > VK_FORMAT_ASTC_12x12_SRGB_BLOCK) return;
    static const uint8_t ve[16] = {0xfc, 0xfd, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0, 0, 0, 0, 0, 0, 0, 0};
    for (size_t i = 0; i + 16 <= n; i += 16) { uint64_t a, b; memcpy(&a, p + i, 8); memcpy(&b, p + i + 8, 8); if (!(a | b)) memcpy(p + i, ve, 16); }
}
static void dense_upload(vkimg_t *im, const tex_t *t, const vkfmt_t *f, uint32_t l, uint32_t y) {   // level l of layer y through the page table, unmapped pages as zeros
    struct vkdense *dn = im->dn; size_t idx = (size_t)l * dn->layers + y;
    uint64_t lo = nvn_tex_level_offset(&t->d, l), hi = l + 1 < dn->tl ? nvn_tex_level_offset(&t->d, l + 1) : dn->layer_size, n = hi - lo, a = (uint64_t)(uintptr_t)t->mem + (uint64_t)y * dn->layer_size + lo;
    if (hi <= lo) return;
    uint8_t *buf = malloc(n); if (!buf) { VKRED_ONCE(160, "no memory for a dense level's copy (%llu bytes)", (unsigned long long)n); return; }
    uint64_t stamp = nvn_dirty_now(), gen = nvn_sparse_gen(); nvn_sparse_protect(t->pool, a, n); nvn_sparse_copy(t->pool, buf, a, n);   // (stamps first: a write after them is seen)
    vk_shadow_note_sparse(t->pool, a, n, stamp);
    uint32_t W = t->d.width ? t->d.width : 1, H = t->d.height ? t->d.height : 1, w = W >> l ? W >> l : 1, h = H >> l ? H >> l : 1, bw = f->bw ? f->bw : 1, bh = f->bh ? f->bh : 1;
    size_t pitch = (size_t)((w + bw - 1) / bw) * f->bpb, rows = (h + bh - 1) / bh; VkBuffer sb; VkDeviceSize so;
    uint8_t *blocks = f->dec ? malloc(pitch * rows) : NULL, *dst = blocks ? blocks : vk_scratch(pitch * rows, 16, &sb, &so);
    if (dst && !nvn_deswizzle_from(&t->d, l, y, buf, (uint64_t)y * dn->layer_size + lo, dst, pitch)) {
        if (blocks) { dst = vk_scratch((size_t)w * h * f->dbpp, 16, &sb, &so); bc_decode(f, blocks, pitch, w, h, 1, dst); }
        else astc_zero_void(f, dst, pitch * rows);   // (unmapped pages: zero texels, not ASTC's error colour)
        VkCommandBuffer cb = vk_upload_cmd(); if (im->layout != VK_IMAGE_LAYOUT_GENERAL) vk_tex_prepare(cb, im);
        VkBufferImageCopy r = {so, 0, 0, {VK_IMAGE_ASPECT_COLOR_BIT, l - im->lv0, y, 1}, {0, 0, 0}, {w, h, 1}}; vkCmdCopyBufferToImage(cb, sb, im->img, VK_IMAGE_LAYOUT_GENERAL, 1, &r);
        dn->gen[idx] = gen; dn->stamp[idx] = stamp; dn->up[idx] = 1;
    } else VKRED_ONCE(161, "dense level %u of sparse nvn format %#x not decoded", l, t->d.format);
    free(blocks); free(buf);
}
// Once a submission: the units of the ranges the game mapped or unmapped since the last look (all of them when the pool's log has moved past it), a new first level when the base moved
// (dense_first), then each level and layer whose mappings or pages changed since its upload
static void dense_look(vkimg_t *im, const tex_t *t, const vkfmt_t *f) {
    struct vkdense *dn = im->dn; if (dn->sub == vk_sub_serial()) return; dn->sub = vk_sub_serial();
    if (nvn_sparse_log_seq(t->pool) != dn->seq) {
        enum { RANGES = 4096 }; uint64_t (*r)[2] = malloc(RANGES * sizeof *r), mem = (uint64_t)(uintptr_t)t->mem;
        int nr = r ? nvn_sparse_log_read(t->pool, &dn->seq, r, RANGES) : -1;
        if (nr < 0) { dn->seq = nvn_sparse_log_seq(t->pool); dense_scan(dn, 0, ~0ull, dense_mapped, t->pool, mem); vk_vt_rescan(im, dn->mapped); }
        else { for (int k = 0; k < nr; k++) if (r[k][1] > mem && r[k][0] < mem + (uint64_t)dn->layers * dn->layer_size)
            dense_scan(dn, r[k][0] > mem ? r[k][0] - mem : 0, r[k][1] - mem, dense_mapped, t->pool, mem);
            vk_vt_ranges(im, dn->mapped, (const uint64_t (*)[2])r, nr); }   // (gfx_vk_vt.c: a tile with a slot in a range the game mapped or unmapped holds the old content)
        free(r);
        uint64_t lv[16]; dense_lvbytes(t, f, dn->layers, lv); uint32_t first = dense_first(lv, dn->tl, dense_base(dn), dense_cap(), im->lv0);
        if (first != im->lv0) dense_remake(im, t, f, first);
    }
    for (uint32_t l = im->lv0; l < dn->tl; l++) for (uint32_t y = 0; y < dn->layers; y++) {
        size_t idx = (size_t)l * dn->layers + y; uint64_t lo = nvn_tex_level_offset(&t->d, l), hi = l + 1 < dn->tl ? nvn_tex_level_offset(&t->d, l + 1) : dn->layer_size;
        if (dn->up[idx] && !nvn_sparse_since(t->pool, (uint64_t)(uintptr_t)t->mem + (uint64_t)y * dn->layer_size + lo, hi - lo, dn->gen[idx], dn->stamp[idx])) continue;
        dense_upload(im, t, f, l, y);
    }
    vk_vt_look(im);   // (gfx_vk_vt.c, once a submission: the tiles the shaders asked for, the page table)
}
void vk_tex_mem(uint64_t o[8]) {   // (gfx_vk_core.c vk_mem_log) images, their bytes, dense images, their bytes, sparse tiles bound, tile blocks, sparse address space, sparse images
    pthread_mutex_lock(&g_img_mu); o[0] = g_img_n; o[1] = g_img_bytes; o[2] = g_dense_n; o[3] = g_dense_bytes; pthread_mutex_unlock(&g_img_mu);
    pthread_mutex_lock(&g_sp_mu); o[4] = g_sp_live; o[5] = g_sp_nblk; o[6] = g_sp_vaddr; o[7] = g_sp_nimg; pthread_mutex_unlock(&g_sp_mu);
}
static void dense_drop(vkimg_t *im) { vk_vt_detach(im); struct vkdense *dn = im->dn; g_dense_bytes -= dn->bytes; g_dense_n--; free(dn->mapped); free(dn->cnt); free(dn->gen); free(dn->stamp); free(dn->up); free(dn); im->dn = NULL; }
// The degraded path (no sparse residency, or a texture whose tiles are not the device's), for what dense_make does not take (a texture the GPU writes, or one not in NVN's sparse tiles):
// a sparse pool's texture is made whole and uploaded through the page table, as before tiles, while it fits DM_VK_SPARSE_WHOLE_MB (default 512); a larger one is not made (it reads as the
// dummy, zeros), with a [RED] once
static int sp_whole_ok(const tex_t *t) {
    static uint64_t lim; if (!lim) { const char *e = getenv("DM_VK_SPARSE_WHOLE_MB"); lim = (uint64_t)(e && atoi(e) > 0 ? atoi(e) : 512) << 20; }
    if (t->L.size <= lim) return 1;
    static int n; if (!n++) VKLOG("[RED] sparse texture %ux%ux%u nvn format %#x (%llu MB made whole) in a sparse pool: %s, and it is past the %llu MB a whole one may take; not drawn (the dummy), later ones not logged",
        t->d.width, t->d.height, t->d.depth, t->d.format, (unsigned long long)(t->L.size >> 20), V.has_sparse ? "its tiles are not the device's" : "the device has no sparse residency", (unsigned long long)(lim >> 20));
    return 0;
}
// Each image over 32 MB once, as it is made: its size and formats, NVN flags, usage, whether the GPU writes it, and how its memory is had (sparse-resident, dense from a base, memory of
// its own, a large block's). Run 5 of Mario Kart World 1.8.0 had 838 MB in 4 images of their own memory at 28 s (DM_VK_MEMLOG) and nothing named them
static void img_big_said(const vkimg_t *im, const tex_t *t, const vkfmt_t *f, int sp_pool, int gpu) {
    int own = 0; uint64_t here = im->sp ? 0 : im->dn ? im->dn->bytes : vk_image_sub_size(im->sub, &own);
    if (t->L.size <= (32ull << 20) && here <= (32ull << 20)) return;
    static int n; if (n++ >= 64 && (n & (n - 1))) return;
    char mf[48] = ""; if (im->fmt != f->vk) snprintf(mf, sizeof mf, ", made in vk %d%s", im->fmt, im->promo ? " (its wider twin)" : "");
    VKLOG("image over 32 MB: %ux%ux%u, %u layers, %u levels, %u samples, nvn format %#x (vk %d%s), target %d, flags %#x, usage %#x%s%s: %.1f MB made whole; %s %.1f MB (%d so far)",
          t->d.width, t->d.height, t->d.depth, im->layers, im->levels, im->samples, t->d.format, f->vk, mf, t->d.target, t->d.flags, im->usage, sp_pool ? ", in a sparse pool" : "",
          gpu ? ", GPU-written" : ", sampled", t->L.size / 1048576.0, im->sp ? "sparse-resident: tiles bound where the game maps its pool, not" : im->dn ? "a dense image from a base level:" :
          own ? "memory of its own:" : "in a large block:", here / 1048576.0, n);
}
void vk_tex_prepare(VkCommandBuffer cb, vkimg_t *im) {
    if (im->layout == VK_IMAGE_LAYOUT_GENERAL) return;
    VkImageMemoryBarrier b = {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER, NULL, 0, VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
        VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED, im->img, {im->aspect, 0, VK_REMAINING_MIP_LEVELS, 0, VK_REMAINING_ARRAY_LAYERS}};
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, NULL, 0, NULL, 1, &b);
    im->layout = VK_IMAGE_LAYOUT_GENERAL;
}
// Upload a packaged texture from guest memory: each level and layer deswizzled into scratch memory, copied in by the upload command buffer
uint64_t g_pkg_n, g_pkg_bytes, g_pkg_ns;   // packaged texture uploads: count, bytes, CPU time (the replay's per-frame line)
static void pkg_upload_(vkimg_t *im, const tex_t *t, const vkfmt_t *f);
static uint64_t pkg_hash(const uint8_t *m, size_t n) {   // FNV-1a over 8-byte words
    uint64_t h = 1469598103934665603ull; size_t i = 0;
    for (; i + 8 <= n; i += 8) { uint64_t w; memcpy(&w, m + i, 8); h = (h ^ w) * 1099511628211ull; }
    for (; i < n; i++) h = (h ^ m[i]) * 1099511628211ull;
    return h;
}
static void pkg_upload(vkimg_t *im, const tex_t *t, const vkfmt_t *f) {
    struct timespec a, b; clock_gettime(CLOCK_MONOTONIC, &a);
    pkg_upload_(im, t, f);
    clock_gettime(CLOCK_MONOTONIC, &b); g_pkg_ns += (uint64_t)((b.tv_sec - a.tv_sec) * 1000000000ll + (b.tv_nsec - a.tv_nsec)); g_pkg_n++; g_pkg_bytes += t->L.size;
    vk_shadow_note_tex(im);   // (its storage protected, the stamp after it: the shadow file may take its pages once they stay unwritten, DM_VK_SHADOW)
}
// A texture whose storage is not wholly readable (a guest range reserved and not mapped, past what the game has mapped: tex_storage, vk_readable): its bytes are not read. Its image keeps what
// it holds, or starts as zeros at its first use; a [RED] line names it (logged for the first cases and then at powers of two)
static void img_clear(vkimg_t *im, const vkfmt_t *f) {   // the image's content as zeros, recorded in the upload command buffer
    VkCommandBuffer cb = vk_upload_cmd(); if (im->layout != VK_IMAGE_LAYOUT_GENERAL) vk_tex_prepare(cb, im);
    VkImageSubresourceRange r = {im->aspect, 0, VK_REMAINING_MIP_LEVELS, 0, VK_REMAINING_ARRAY_LAYERS};
    if (im->aspect & (VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT)) { VkClearDepthStencilValue z = {0, 0}; vkCmdClearDepthStencilImage(cb, im->img, VK_IMAGE_LAYOUT_GENERAL, &z, 1, &r); return; }
    if (!f->compressed || f->dec) { VkClearColorValue z = {{0}}; vkCmdClearColorImage(cb, im->img, VK_IMAGE_LAYOUT_GENERAL, &z, 1, &r); return; }
    unsigned bw = f->bw ? f->bw : 1, bh = f->bh ? f->bh : 1;   // compressed: blocks of zeros copied in (clears do not take compressed images)
    for (unsigned lv = 0; lv < im->levels; lv++) { uint32_t w = im->w >> lv ? im->w >> lv : 1, h = im->h >> lv ? im->h >> lv : 1; size_t slab = (size_t)((w + bw - 1) / bw) * ((h + bh - 1) / bh) * f->bpb;
        for (unsigned ly = 0; ly < im->layers; ly++) { VkBuffer sb; VkDeviceSize so; uint8_t *z = vk_scratch(slab, 16, &sb, &so); memset(z, 0, slab);
            VkBufferImageCopy c = {so, 0, 0, {VK_IMAGE_ASPECT_COLOR_BIT, lv, ly, 1}, {0, 0, 0}, {w, h, 1}}; vkCmdCopyBufferToImage(cb, sb, im->img, VK_IMAGE_LAYOUT_GENERAL, 1, &c); } }
}
static void tex_unreadable(vkimg_t *im, const tex_t *t, const vkfmt_t *f, uint64_t at, const char *what) {
    static uint64_t n; n++;
    const pool_t *pl = t->pool;
    if (n <= 8 || !(n & (n - 1))) VKLOG("[RED] texture %ux%ux%u nvn format %#x target %d levels %u at %p +%#llx (pool %p flags %#x, its storage %p +%#llx, the texture at offset %#llx%s): its bytes from +%#llx are "
        "not readable (a guest range reserved and not mapped): %s not read, the image %s (%llu so far)", t->d.width, t->d.height, t->d.depth, t->d.format, t->d.target, t->d.levels, (const void *)t->mem,
        (unsigned long long)t->L.size, (const void *)pl, pl ? pl->flags : 0, pl ? (const void *)pl->mem : NULL, pl ? (unsigned long long)pl->size : 0ull, (unsigned long long)t->off,
        pl && (uint64_t)t->off + t->L.size > pl->size ? ", past the pool's end" : "", (unsigned long long)at, what, im->packaged <= 1 ? "cleared" : "kept as it is", (unsigned long long)n);
    if (im->packaged <= 1) img_clear(im, f);
}
static void pkg_upload_(vkimg_t *im, const tex_t *t, const vkfmt_t *f) {
    { static int n; const char *e = getenv("DM_VK_LOG_UPLOADS"); if (e && atoi(e) && n < 60) { n++;   // which packaged textures are (re-)uploaded
        VKLOG("upload: serial %llu %ux%ux%u nvn format %#x (vk %d) target %d levels %u layers %u flags %#x stride %u size %llu mem %p, tileH*100+pkg_block_h %d%s",
              (unsigned long long)t->serial, t->d.width, t->d.height, t->d.depth, t->d.format, f->vk, t->d.target, im->levels, im->layers, t->d.flags, t->d.stride, (unsigned long long)t->L.size,
              (const void *)t->mem, (int)t->L.tileH * 100 + t->d.pkg_block_h, im->packaged > 1 ? " (again)" : ""); } }
    if (im->packaged < 100) im->packaged++;
    VkCommandBuffer cb = vk_upload_cmd();
    if (im->layout != VK_IMAGE_LAYOUT_GENERAL) vk_tex_prepare(cb, im);
    // A sparse pool's texture is read through a copy gathered from its page table, unmapped pages as 0 (nvn_mtl.m pkg_upload_: its own address range is only partly mapped, and a mip tail at
    // a 4 KB offset has no alias at all: reading the alias there stopped the host with SIGBUS); its pages are protected through the page table, and its re-upload is judged there (vk_tex_)
    const uint8_t *src = t->mem; uint8_t *part = NULL; int sparse = t->pool && nvn_sparse_pool(t->pool);
    if (sparse) { im->sp_gen = nvn_sparse_gen(); if (!(part = malloc(t->L.size ? t->L.size : 1))) { VKRED_ONCE(156, "no memory for a sparse texture's copy (%llu bytes)", (unsigned long long)t->L.size); return; }
        nvn_sparse_protect(t->pool, (uint64_t)(uintptr_t)t->mem, t->L.size); nvn_sparse_copy(t->pool, part, (uint64_t)(uintptr_t)t->mem, t->L.size); src = part; }
    else { uint64_t u = vk_unreadable_at(t->mem, t->L.size); if (u < t->L.size) { tex_unreadable(im, t, f, u, "its upload"); return; }   // (never read past what is mapped)
        nvn_dirty_protect(t->mem, t->L.size); }
    im->pkg_stamp = nvn_dirty_now(); im->pkg_span = sparse ? NULL : nvn_dirty_span(t->mem, t->L.size); im->pkg_seen = im->pkg_stamp;
    im->pkg_mem = sparse ? NULL : t->mem; im->pkg_len = t->L.size; im->pkg_released = 0; im->pkg_hash = pkg_hash(src, t->L.size);   // (pkg_mem NULL: vk_tex_sweep leaves a sparse one's protection to the page table)
    if (sparse && t->d.target == NVN_TEXTURE_TARGET_2D && im->levels > 1) {   // sampled from its first level whose pages, and those of every smaller level, are mapped (nvn_mtl.m sparse_view)
        uint32_t r = im->levels;
        while (r > 0) { uint64_t lo = nvn_tex_level_offset(&t->d, r - 1), hi = r < im->levels ? nvn_tex_level_offset(&t->d, r) : t->L.size; if (!nvn_sparse_mapped(t->pool, (uint64_t)(uintptr_t)t->mem + lo, hi - lo)) break; r--; }
        if (r == im->levels) r = im->levels - 1;   // nothing mapped: the smallest level (reads 0)
        if (r != im->sp_vbase) { static int nl; if (nl++ < 6) VKLOG("sparse texture %ux%u nvn format %#x: sampled from level %u of %u (finer levels not mapped)", t->d.width, t->d.height, t->d.format, r, im->levels);
            im->sp_vbase = r; extern uint64_t g_tex_gen, g_tex_dgen; g_tex_gen++; g_tex_dgen++; }   // (the cached views by ID are made again)
    }
    int linear = (t->d.flags & (NVN_TEXTURE_FLAGS_LINEAR | NVN_TEXTURE_FLAGS_LINEAR_RENDER_TARGET)) != 0;
    unsigned bw = f->bw ? f->bw : 1, bh = f->bh ? f->bh : 1, bpb = f->bpb;
    if (!bpb) { free(part); VKRED_ONCE(12, "packaged texture format %#x has no block size", t->d.format); return; }
    for (unsigned lv = 0; lv < im->levels; lv++) {
        uint32_t w = im->w >> lv ? im->w >> lv : 1, h = im->h >> lv ? im->h >> lv : 1, z = im->d >> lv ? im->d >> lv : 1;
        uint32_t bx = (w + bw - 1) / bw, by = (h + bh - 1) / bh; size_t pitch = (size_t)bx * bpb, slab = pitch * by * z;
        for (unsigned ly = 0; ly < im->layers; ly++) {
            VkBuffer sb; VkDeviceSize so; uint8_t *blocks = f->dec ? malloc(slab) : NULL, *dst = blocks ? blocks : vk_scratch(slab, 16, &sb, &so);
            if (f->dec && !blocks) { free(part); return; }
            if (linear) {   // linear storage: rows of the texture's stride, one level, one layer
                size_t stride = t->d.stride ? t->d.stride : pitch;
                for (uint32_t y = 0; y < by * z; y++) memcpy(dst + y * pitch, src + (size_t)y * stride, pitch);
            } else if (nvn_deswizzle(&t->d, lv, ly, src, dst, pitch)) { free(blocks); free(part); VKRED_ONCE(13, "deswizzle refused texture format %#x target %d", t->d.format, t->d.target); return; }
            if (sparse && !blocks) astc_zero_void(f, dst, slab);   // (read through the page table: an unmapped page is zero texels, dense_upload)
            if (blocks) { dst = vk_scratch((size_t)w * h * z * f->dbpp, 16, &sb, &so); bc_decode(f, blocks, pitch, w, h, z, dst); free(blocks); }   // (BC the device lacks)
            VkBufferImageCopy r = {so, 0, 0, {im->aspect & VK_IMAGE_ASPECT_COLOR_BIT ? VK_IMAGE_ASPECT_COLOR_BIT : VK_IMAGE_ASPECT_DEPTH_BIT, lv, ly, 1}, {0, 0, 0}, {w, h, z}};
            vkCmdCopyBufferToImage(cb, sb, im->img, VK_IMAGE_LAYOUT_GENERAL, 1, &r);
        }
    }
    free(part);
}
static int packaged_tex(const tex_t *t);
static int lin_copy(vkimg_t *im, const tex_t *t, const vkfmt_t *f);
static void vk_tex_view_drop_all(vkimg_t *im);
// Images over the same memory with different descriptions (the storage key differs) are separate Vulkan images: on the Switch a texture sees what another
// one over its memory rendered, here it does not. They are linked both ways at creation (up to 8 each), and a sampled image older than an overlapping
// image's last rendering is logged ([alias] stale read), to find what a game reinterprets. Every GPU write counts as a rendering (vk_img_wrote: attachments, storage images,
// copy and clear destinations): with attachments alone, an image a compute shader writes read as older than a render target drawn over the same memory, and Mario Kart World's
// run 13 spent the 200 lines on such pairs (a 52x29 R8UI image beside 32x32 R8 and 13x7 RGBA16F targets of its transient memory) before anything else could be seen
static uint64_t g_wseq;
void vk_img_wrote(vkimg_t *im) { im->wseq = ++g_wseq; im->gpu_w = 1; if (im->sp) vk_sparse_gpu_wrote(im); }   // the GPU writes im (gfx_vk_exec.c: storage images, copies, clears)
static void alias_link(vkimg_t *im) {
    for (uint32_t i = 0; i < g_by_key.cap; i++) {
        vkimg_t *o = g_by_key.vals[i]; if (!o || o == im || o->pool != im->pool || o->off >= im->off + im->size || im->off >= o->off + o->size) continue;
        if (im->novl < 8 && o->novl < 8) { im->ovl[im->novl++] = o; o->ovl[o->novl++] = im; }
    }
}
static void alias_unlink(vkimg_t *im) {
    for (int i = 0; i < im->novl; i++) { vkimg_t *o = im->ovl[i]; for (int j = 0; j < o->novl; j++) if (o->ovl[j] == im) { o->ovl[j] = o->ovl[--o->novl]; break; } }
    im->novl = 0;
}
static void alias_check(vkimg_t *im) {   // im is about to be sampled
    for (int i = 0; i < im->novl; i++) {
        vkimg_t *o = im->ovl[i]; if (o->wseq <= im->wseq || o->wseq <= im->rseen) continue;
        im->rseen = o->wseq;
        extern uint64_t g_tex_frame; static uint64_t n; n++;   // the first 64, then one in 1024 (with the count): a pattern that repeats every frame no longer hides the rest
        if (n <= 64 || !(n & 1023)) VKLOG("[alias] stale read #%llu (present %llu): %ux%u fmt %#x levels %u at %+lld..%+lld sampled, but %ux%u fmt %#x levels %u at %+lld..%+lld written over its memory later (%llu GPU writes ago)",
            (unsigned long long)n, (unsigned long long)g_tex_frame, im->w, im->h, im->nvn_format, im->levels, (long long)im->off, (long long)(im->off + im->size), o->w, o->h, o->nvn_format, o->levels,
            (long long)o->off, (long long)(o->off + o->size), (unsigned long long)(g_wseq - o->wseq));
    }
}
extern uint64_t g_tex_frame;   // presents so far, for the packaged textures' last use (vk_tex_sweep)
// Images are kept per storage key until their pool goes, so a texture the game streams into memory another texture had before gets an image of its own and the old one
// stayed: Xenoblade Chronicles 3 streams its field textures through pools that live for good, the device's GPU memory grew by about 470 MB a minute while walking (6.3 GB
// after 20 minutes) and Android's low-memory killer ended the game. A new image takes over: the images of the same pool over its memory that were not looked up in the
// last EVICT_KEEP presents go (their memory now holds another texture; one drawn again is made and filled again). Called with g_img_mu held
enum { EVICT_KEEP = 60 };
static uint64_t g_img_n, g_img_bytes, g_img_evicted;   // live images and their bytes, images evicted so far (vk_tex_sweep's line)
static void dense_drop(vkimg_t *im);
static void img_drop(vkimg_t *im) { vk_lrz_forget(im); uint64_t bytes = im->sp ? 0 : im->dn ? im->dn->bytes : im->size <= 0 ? 0 : (uint64_t)im->size; if (im->sp) sparse_drop(im); if (im->dn) dense_drop(im);   // (a sparse image's bytes are its slots: not counted)
    if (im->lin_mem) nvn_dirty_release(im->lin_mem, im->lin_len); if (im->packaged && !im->pkg_released && im->pkg_mem) nvn_dirty_release(im->pkg_mem, im->pkg_len); g_img_n--; g_img_bytes -= bytes; alias_unlink(im); vk_tex_view_drop_all(im); if (im->sub) vk_defer_free_image_sub(im->img, im->sub); else vk_defer_free_image(im->img, im->mem); free(im); }
static void evict_overlaps(uint64_t pool, int64_t off, int64_t size) {
    akey ks[64]; int n = 0;
    for (uint32_t i = 0; i < g_by_key.cap && n < 64; i++) {
        vkimg_t *o = g_by_key.vals[i]; if (!o || o->pool != pool || o->off >= off + size || off >= o->off + o->size || g_tex_frame - o->used < EVICT_KEEP) continue;
        memcpy(&ks[n++], g_by_key.keys + (size_t)i * g_by_key.klen, sizeof(akey));
    }
    for (int j = 0; j < n; j++) { vkimg_t *o = nvn_hm_get(&g_by_key, &ks[j]); if (!o) continue; nvn_hm_del(&g_by_key, &ks[j]); img_drop(o); g_img_evicted++; }
}
static vkimg_t *vk_tex_(const tex_t *t);
// A texture of a virtual pool that is not sparse (Android: an aliased pool, whose own range has no CPU alias, hle_nvn.c nvn_virt_xlat): its storage at the physical memory behind it, the
// texture otherwise as it is (its image key is its pool and offset). One whose range cannot be had in one piece has no storage the replay may read: it starts cleared, [RED] once. A sparse
// pool's texture keeps its own address, read through the pool's page table (sparse_*, dense_*, pkg_upload_)
int nvn_virt_xlat(uint64_t a, uint64_t n, uint64_t *out);
static const tex_t *tex_storage(const tex_t *t, tex_t *tt) {
    if (!t || !t->mem || !t->pool || !(t->pool->flags & NVN_MEMORY_POOL_FLAGS_VIRTUAL) || nvn_sparse_pool(t->pool)) return t;
    uint64_t ph; int r = nvn_virt_xlat((uint64_t)(uintptr_t)t->mem, t->L.size ? t->L.size : 1, &ph); if (!r) return t;
    *tt = *t; tt->mem = r == 1 ? (uint8_t *)(uintptr_t)ph : NULL;
    if (r < 0) VKRED_ONCE(165, "texture %ux%u nvn format %#x in a virtual pool's range the CPU cannot read in one piece: no storage read, it starts cleared", t->d.width, t->d.height, t->d.format);
    return tt;
}
vkimg_t *vk_tex(const tex_t *t) { extern int g_prof_on; extern uint64_t g_prof_ns[8]; struct timespec a, b; if (g_prof_on) clock_gettime(CLOCK_MONOTONIC, &a);
    tex_t tt; t = tex_storage(t, &tt);
    vkimg_t *im = vk_tex_(t); if (g_prof_on) { clock_gettime(CLOCK_MONOTONIC, &b); g_prof_ns[1] += (uint64_t)((b.tv_sec - a.tv_sec) * 1000000000ll + (b.tv_nsec - a.tv_nsec)); }
    if (im && im->dsc_pend && !vk_sub_open()) vk_dsc_read_(im, "presented or read outside a submission");   // (the discards' history: gfx_vk_present.c takes the window texture so)
    return im; }
static int tex_cpu_content(const tex_t *t, const vkfmt_t *f); static void cpu_unc_drop(void); static int img_use(uint64_t serial); static const uint8_t *nvn_twins(int f);
uint64_t g_pkg_edge_skips;   // pages written beside a packaged texture, its own bytes unchanged: protected again instead of uploaded (mtl_pkg_edge_skips)
static vkimg_t *vk_tex_(const tex_t *t) {
    if (!t || !t->serial || t->magic != MAGIC('T', 'E', 'X', 0)) return NULL;
    // a texture of a memory pool that is gone (nvnMemoryPoolFinalize clears its magic, after its images were dropped, vk_tex_drop_pool): an ID or a record may name it past that, and an image
    // made now would read memory the game has freed (nvn_mtl.m tex_mtl_: garbage in a finalized pool's page table pointer sent an upload through a table that was never there)
    if (t->pool && t->pool->magic != MAGIC('P', 'O', 'O', 'L')) { static int n; if (n++ < 4) VKLOG("texture of a memory pool that is gone %ux%u nvn format %#x: not made (the dummy)", t->d.width, t->d.height, t->d.format); return NULL; }
    const vkfmt_t *f = vk_fmt(t->d.format);
    if (!f) { VKRED_ONCE(14, "texture format %#x has no Vulkan counterpart here (first case %ux%u target %d)", t->d.format, t->d.width, t->d.height, t->d.target); return NULL; }
    if (t->d.target == NVN_TEXTURE_TARGET_BUFFER) return NULL;   // buffer textures: texel buffer views of the pool (vk_tex_view)
    if (vk_sub_open()) cpu_unc_drop();   // (the submit thread, as tex_cpu_content: a pool's first GPU write turns its CPU-content textures into GPU ones)
    akey k; memset(&k, 0, sizeof k); k.pool = (uint64_t)(uintptr_t)t->pool; k.off = t->off; k.d = t->d;
    // flags that do not change the storage's layout are not part of the key: on the Switch two textures over the same memory that differ only in them see the
    // same texels. Xenoblade Chronicles 3 renders its shadow maps and depth-stencil buffer through textures with COMPRESSIBLE and samples them through others with
    // IMAGE added; as separate images those were never written, the shadows were total and the stencil's material tags 0, and its lit scene black. IMAGE stays
    // in the key for colour formats (it gives the image storage usage), not for depth and stencil, which cannot be storage images
    k.d.flags &= NVN_TEXTURE_FLAGS_LINEAR | NVN_TEXTURE_FLAGS_SPARSE | NVN_TEXTURE_FLAGS_LINEAR_RENDER_TARGET | NVN_TEXTURE_FLAGS_MINIMAL_LAYOUT | (f->depth || f->stencil ? 0 : NVN_TEXTURE_FLAGS_IMAGE);
    // registered as an image (TexturePoolRegisterImage, mtl_reg_image) though its builder omitted IMAGE: made with storage use, as the image it is bound as (nvn_mtl.m RegisterImage)
    tex_t ti; const tex_t *tc = t;
    if (!(t->d.flags & NVN_TEXTURE_FLAGS_IMAGE) && !(f->depth || f->stencil) && img_use(t->serial)) { ti = *t; ti.d.flags |= NVN_TEXTURE_FLAGS_IMAGE; tc = &ti; k.d.flags |= NVN_TEXTURE_FLAGS_IMAGE; }
    // A sparse pool's texture (hle_nvn.c struct vpool): sampled ones hold what the game mapped into the pool and are read from it (nvn_mtl.m tex_from_storage), render targets and images
    // are the GPU's; either is a sparse-resident image when it qualifies (sparse_make), else made whole under a budget (sp_whole_ok)
    int sp_pool = t->pool && t->mem && nvn_sparse_pool(t->pool);
    int from_storage = sp_pool && !(t->d.flags & (NVN_TEXTURE_FLAGS_LINEAR | NVN_TEXTURE_FLAGS_LINEAR_RENDER_TARGET | NVN_TEXTURE_FLAGS_COMPRESSIBLE | NVN_TEXTURE_FLAGS_IMAGE));
    int keyed = t->pool != NULL;
    pthread_mutex_lock(&g_img_mu);
    vkimg_t *im = keyed ? nvn_hm_get(&g_by_key, &k) : nvn_hm_get(&g_by_serial, &t->serial);
    int fresh = 0;
    // A texture over the storage of its sRGB / linear twin (the key differs only by the other format of the pair) is a view of that image: what one writes, the other reads. Mario Kart
    // World's Grand Prix movie draws its YUV pass into an RGBA8 target and composites it through an RGBA8_SRGB texture: as an image of its own, a frozen frame (upstream 92f223a7, #1110).
    // The image must take views in the twin format: mutable, or format-listed with it (DM_VK_UBWC: fixed_fmt, fmt2)
    if (!im && keyed) { const uint8_t *tw = nvn_twins(t->d.format);
        for (int j = 0; j < 3 && tw[j] && !im; j++) { akey kt = k; kt.d.format = tw[j]; vkimg_t *o = nvn_hm_get(&g_by_key, &kt);
            if (o && !o->sp && !o->dn && !o->lv0 && (o->cflags & VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT) && (!o->fixed_fmt || o->fmt2 == f->vk)) { im = o; static int n; if (n++ < 8) VKLOG("texture %ux%u nvn format %#x: a view of its sRGB / linear twin's image (format %#x) over the same storage", t->d.width, t->d.height, t->d.format, tw[j]); } } }
    if (!im) {
        // A colour texture's key keeps IMAGE (its image's storage use), so two texture objects over the same storage that differ only by it are two images: a compute shader's
        // writes through the IMAGE one are not seen through the other, where on the Switch they are the same texels (upstream left IMAGE out of the key: XC3's material tags,
        // nvn_mtl.m tex_upgrade). Said for the first cases, to tell whether a game does it ([RED]: what one writes the other never reads)
        if (keyed && !(f->depth || f->stencil)) { akey ki = k; ki.d.flags ^= NVN_TEXTURE_FLAGS_IMAGE; vkimg_t *o = nvn_hm_get(&g_by_key, &ki);
            static int ns; if (o && ns++ < 8) VKLOG("[RED] texture %ux%u nvn format %#x at %+lld: a second image beside the one %s IMAGE over the same storage (%llu so far): writes through one are not seen through the other",
                t->d.width, t->d.height, t->d.format, (long long)t->off, (k.d.flags & NVN_TEXTURE_FLAGS_IMAGE) ? "without" : "with", (unsigned long long)ns); }
        int gpu = !(from_storage || packaged_tex(t));
        if (sp_pool) im = sparse_make(tc, f, gpu);
        if (!im && sp_pool && !gpu && t->L.size > dense_cap()) im = dense_make(tc, f);   // (sampled, too large to be made whole: a dense image from its base)
        if (!im && sp_pool && !sp_whole_ok(t)) { pthread_mutex_unlock(&g_img_mu); return NULL; }
        if (!im && keyed) evict_overlaps(k.pool, t->off, (int64_t)t->L.size);
        if (!im) im = img_new(tc, f);
        if (!im) { pthread_mutex_unlock(&g_img_mu); return NULL; }
        im->pool = k.pool; im->off = t->off; im->size = (int64_t)t->L.size; im->flags = tc->d.flags; g_img_n++; g_img_bytes += im->sp ? 0 : im->dn ? im->dn->bytes : t->L.size;
        if (keyed) { alias_link(im); nvn_hm_put(&g_by_key, &k, im); } else nvn_hm_put(&g_by_serial, &t->serial, im);
        fresh = 1; img_big_said(im, tc, f, sp_pool, gpu);
    }
    im->used = g_tex_frame;
    pthread_mutex_unlock(&g_img_mu);
    if (!vk_sub_open()) return fresh ? NULL : im;   // looked up outside a submission (present): nothing can be recorded, a new image has no contents yet
    if (im->sp) { sparse_look(im, t); return im; }   // its tiles follow the game's mappings (the first look binds and uploads them all, within the submission's budget)
    if (im->dn) { dense_look(im, t, f); return im; }   // its levels from the base follow the game's mappings (dense_look)
    // a texture's first use: written by the CPU since it was registered (profile tex_dirty_upload), it is CPU-written from now on, as at nvn_mtl.m's first resolution;
    // otherwise only a later nvn_ft_poll pass finds it, and its first draws read zeros
    int nvn_ft_written(const tex_t *);
    if (t->mem && lin_copy(im, t, f)) {}
    else if (t->mem && (packaged_tex(t) || from_storage || (fresh && (nvn_ft_written(t) || (tex_cpu_content(t, f) && (mtl_tex_cpu_written(t), 1)))))) {
        im->pkg_used = g_tex_frame;
        if (!im->packaged) { im->packaged = 1; pkg_upload(im, t, f); im->sp_sub = vk_sub_serial(); }
        else if (sp_pool) {   // through the page table, once a submission: a mapping in its range newer than the upload, or a physical page behind it written since (nvn_mtl.m, upstream 229fc22c)
            if (im->sp_sub != vk_sub_serial()) { im->sp_sub = vk_sub_serial(); if (nvn_sparse_since(t->pool, (uint64_t)(uintptr_t)t->mem, t->L.size, im->sp_gen, im->pkg_stamp)) pkg_upload(im, t, f); }
        } else {
            uint64_t now = nvn_dirty_now();
            if (now != im->pkg_seen && nvn_dirty_since_span(im->pkg_span, t->mem, t->L.size, im->pkg_stamp)) {   // pages written since the upload
                // Written around it on shared pages, its own bytes as uploaded (a texture of its own pool with the pool object in its first page, as Pokémon Sword keeps its GFPAK
                // textures, nvn_mtl.m pkg_edges_clean): protected again, not uploaded. That also keeps what the GPU wrote into it since (a compute shader's image stores, a copy, a render),
                // which an upload would put the CPU's bytes over: Monster Hunter Rise's spot light profiles (512x1x64 R16F, written by a compute shader into memory the game had zeroed)
                // were uploaded again as zeros at presents 60 and 1079, every spot light's profile read 0 and the faces lit only by those lights (the character maker, cutscenes) came out black
                uint64_t un = vk_unreadable_at(t->mem, t->L.size);
                if (un < t->L.size) tex_unreadable(im, t, f, un, "its new bytes");
                else if (pkg_hash(t->mem, t->L.size) == im->pkg_hash) { nvn_dirty_protect(t->mem, t->L.size); im->pkg_stamp = nvn_dirty_now(); im->pkg_released = 0; g_pkg_edge_skips++;
                    static int nl; if (nl++ < 16) VKLOG("texture %ux%u nvn format %#x at %p: its pages were written, its bytes are as uploaded%s: kept", t->d.width, t->d.height, t->d.format, (const void *)t->mem, im->gpu_w ? " and the GPU wrote it" : ""); }
                else pkg_upload(im, t, f);
            }
            im->pkg_seen = now;
        }
    } else if (fresh) vk_tex_prepare(vk_upload_cmd(), im);
    return im;
}
// The per-submission part of vk_tex for the image an earlier lookup gave this texture (gfx_vk_exec.c tex_desc_for keeps an ID's view across submissions while the ID is not registered
// again and no image or view went): what vk_tex_ and vk_tex_view do once they have found the image (a packaged texture's re-upload check, a linear one's copy, the aliasing check),
// without finding it again (the registry's copy, the image key's hash under g_img_mu, the view's). NULL where it does not apply (outside a submission, a pool that is gone, sparse and
// dense images, sparse pools' textures, buffer textures): the caller looks the texture up as before
static int linear_tex(const tex_t *t);
vkimg_t *vk_tex_again(vkimg_t *im, const tex_t *t0) {
    if (!im || im->sp || im->dn || !vk_sub_open() || !t0 || !t0->serial || t0->magic != MAGIC('T', 'E', 'X', 0) || (t0->pool && t0->pool->magic != MAGIC('P', 'O', 'O', 'L'))) return NULL;
    tex_t tt; const tex_t *t = tex_storage(t0, &tt);
    const vkfmt_t *f = vk_fmt(t->d.format); if (!f || t->d.target == NVN_TEXTURE_TARGET_BUFFER || (t->pool && t->mem && nvn_sparse_pool(t->pool))) return NULL;
    int pk = packaged_tex(t);   // (a packaged layout or a linear texture: no lock; else the CPU-written tables)
    cpu_unc_drop();
    im->used = g_tex_frame;   // (vk_tex_ writes it under g_img_mu, where vk_tex_sweep reads it: a present count, which a race leaves at most one present late)
    if (im->novl) alias_check(im);   // (vk_tex_view's part: an overlapping image written since, Mario Kart World's aliased pools)
    vk_watch(im, "sampled or image");
    if (t->mem && lin_copy(im, t, f)) {}
    else if (t->mem && pk) {   // as vk_tex_ for an image that exists (fresh 0), not of a sparse pool (from_storage 0)
        im->pkg_used = g_tex_frame;
        if (!im->packaged) { im->packaged = 1; pkg_upload(im, t, f); im->sp_sub = vk_sub_serial(); }
        else { uint64_t now = nvn_dirty_now();
            if (now != im->pkg_seen && nvn_dirty_since_span(im->pkg_span, t->mem, t->L.size, im->pkg_stamp)) {   // pages written since the upload: as vk_tex_
                uint64_t un = vk_unreadable_at(t->mem, t->L.size);
                if (un < t->L.size) tex_unreadable(im, t, f, un, "its new bytes");
                else if (pkg_hash(t->mem, t->L.size) == im->pkg_hash) { nvn_dirty_protect(t->mem, t->L.size); im->pkg_stamp = nvn_dirty_now(); im->pkg_released = 0; g_pkg_edge_skips++; }
                else pkg_upload(im, t, f); }
            im->pkg_seen = now; }
    }
    return im;
}
// Packaged textures give up their write protection (nvn_dirty_release; drawn again, they are uploaded and protected again) when not drawn for PKG_IDLE presents, and the
// least recently drawn ones beyond that while the protected runs, each splitting its pool's mapping, are more than PKG_RUNS (textures stay protected until the game writes them).
// Textures drawn in the last PKG_KEEP presents keep theirs (released, they would be uploaded again at once). Swept every 300 presents, with the pools whose mapping the
// protection has split most mapped again in one piece (nvn_dirty_defrag: Xenoblade Chronicles 3 went from 43,000 mappings after 25 minutes of walking, toward the kernel's
// limit of 65,530, to about 17,500)
enum { PKG_IDLE = 1800, PKG_KEEP = 60, PKG_RUNS = 8000 };   // about 60 s and 2 s at 30 fps; at most about 16,000 mappings for the protection
uint64_t g_tex_frame;
typedef struct { uint64_t used; vkimg_t *im; } pkg_age;
static int pkg_age_cmp(const void *a, const void *b) { uint64_t x = ((const pkg_age *)a)->used, y = ((const pkg_age *)b)->used; return x < y ? -1 : x > y; }
void vk_tex_sweep(void) {
    vk_shadow_tick();   // (each present: the swaps the shadow file's worker has made ready, DM_VK_SHADOW)
    if (++g_tex_frame % 300) return;
    uint64_t idle = 0, lru = 0, before = nvn_dirty_released();
    pthread_mutex_lock(&g_img_mu);
    pkg_age *v = NULL; size_t nv = 0, cap = 0;
    for (int m = 0; m < 2; m++) {
        nvn_hmap *h = m ? &g_by_serial : &g_by_key;
        for (uint32_t i = 0; i < h->cap; i++) {
            vkimg_t *im = h->vals[i];
            if (!im || !im->packaged || im->pkg_released || !im->pkg_mem) continue;
            if (g_tex_frame - im->pkg_used >= PKG_IDLE) { nvn_dirty_release(im->pkg_mem, im->pkg_len); im->pkg_released = 1; idle++; continue; }
            if (g_tex_frame - im->pkg_used < PKG_KEEP) continue;
            if (nv == cap) { cap = cap ? cap * 2 : 4096; pkg_age *nw = realloc(v, cap * sizeof *v); if (!nw) break; v = nw; }
            v[nv++] = (pkg_age){im->pkg_used, im};
        }
    }
    // Textures from guest memory not looked up for a while go (DM_VK_IDLE_EVICT=<presents>, start-up, default 1800: about a minute at 30 fps; 0 never): their bytes stay in guest memory,
    // and one looked up again is made and uploaded again, as a texture evicted by a newer one over its memory is (evict_overlaps). Mario Kart World 1.8.0's Grand Prix (run 15): images
    // grew from 3,504 (2.1 GB) to 12,129 (3.7 GB) over four tracks, 6,793 of them (1.9 GB) not looked up for 1,800 presents, the GPU's memory from 4.2 to 7.0 GB, and the low-memory killer
    // ended the game after 16 minutes. The GPU's images (render targets, storage images, copy destinations: their content is the GPU's), sparse and dense ones stay. Under memory pressure
    // (gfx_vk_budget.c vk_mem_pressure) critical, the idle time is 600 presents. Not when moderate: Mario Kart World's races keep MemAvailable under DM_VK_PRESSURE_MB the whole time, and
    // shorter idle times there evicted the far half of the track each lap (runs 16 and 17: 35,000 to 41,000 textures, 7-8 GB, in 20 minutes, uploaded again a lap later)
    uint64_t idle_ev = 0, idle_ev_mb = 0;
    { static long idle_n = -1; if (idle_n < 0) { const char *e = getenv("DM_VK_IDLE_EVICT"); idle_n = e && *e ? atol(e) : 1800; }
      int vk_mem_pressure(void); void vk_img_pressure(int level); int pr = vk_mem_pressure(); vk_img_pressure(pr); uint64_t lim = !idle_n ? 0 : pr >= 2 ? (idle_n < 600 ? (uint64_t)idle_n : 600) : (uint64_t)idle_n;
      if (lim) { akey ks[512]; int nk = 0;
          for (uint32_t i = 0; i < g_by_key.cap && nk < 512; i++) { vkimg_t *im = g_by_key.vals[i];
              if (!im || !im->packaged || im->gpu_w || im->sp || im->dn || g_tex_frame - im->used < lim) continue;
              memcpy(&ks[nk++], g_by_key.keys + (size_t)i * g_by_key.klen, sizeof(akey)); }
          for (int j = 0; j < nk; j++) { vkimg_t *o = nvn_hm_get(&g_by_key, &ks[j]); if (!o) continue; idle_ev_mb += o->size > 0 ? (uint64_t)o->size : 0; nvn_hm_del(&g_by_key, &ks[j]); img_drop(o); idle_ev++; g_img_evicted++; } } }
    { int vk_pool_remap_ro(void *mem, size_t len); int nvn_dirty_defrag(int (*)(void *, size_t)); struct timespec t0, t1; clock_gettime(CLOCK_MONOTONIC, &t0); int d = nvn_dirty_defrag(vk_pool_remap_ro); clock_gettime(CLOCK_MONOTONIC, &t1); static uint64_t nd;
        if (d && (nd += d) <= 8) VKLOG("dirty-page tracking: %d pool(s) mapped again in one piece in %.1f ms (%llu so far)", d, (t1.tv_sec - t0.tv_sec) * 1e3 + (t1.tv_nsec - t0.tv_nsec) / 1e6, (unsigned long long)nd); }
    int runs = nvn_dirty_runs();
    if (runs > PKG_RUNS && nv) {   // the least recently drawn first, in batches, until the runs are well under the budget
        qsort(v, nv, sizeof *v, pkg_age_cmp);
        for (size_t i = 0; i < nv && runs > PKG_RUNS * 3 / 4; ) {
            for (size_t e = i + 256; i < nv && i < e; i++) { nvn_dirty_release(v[i].im->pkg_mem, v[i].im->pkg_len); v[i].im->pkg_released = 1; lru++; }
            runs = nvn_dirty_runs();
        }
    }
    pthread_mutex_unlock(&g_img_mu);
    free(v);
    static uint64_t total; total += idle + lru;
    if (g_tex_frame % 3000 == 0) VKLOG("images: %llu live, %.0f MB of textures, %llu evicted so far (a newer texture over their memory)", (unsigned long long)g_img_n, g_img_bytes / 1048576.0, (unsigned long long)g_img_evicted);
    { static uint64_t ev_n, ev_mb; ev_n += idle_ev; ev_mb += idle_ev_mb; if (idle_ev && (ev_n == idle_ev || g_tex_frame % 3000 == 0)) VKLOG("idle textures evicted: %llu (%.0f MB) now, %llu (%.0f MB) so far (DM_VK_IDLE_EVICT)", (unsigned long long)idle_ev, idle_ev_mb / 1048576.0, (unsigned long long)ev_n, ev_mb / 1048576.0); }
    if ((idle || lru) && (g_tex_frame % 3000 == 0 || lru)) VKLOG("packaged textures: %llu not drawn for %d presents and %llu least recently drawn released their write protection (%llu pages; %llu so far); %d protected runs",
                    (unsigned long long)idle, PKG_IDLE, (unsigned long long)lru, (unsigned long long)(nvn_dirty_released() - before), (unsigned long long)total, runs);
}
// The census of the memory line (gfx_vk_core.c vk_mem_log, DM_VK_MEMLOG). o[0..9]: textures uploaded from guest memory (not sparse: their storage's bytes), of them what the shadow file
// could hold now (gfx_vk_shadow.c vk_shadow_tex_class: their pages' bytes are their images' as well), in placed pools (a hardware buffer's pages: GPU memory), released, and the others.
// o[10..15]: the GPU's images nothing has written yet (not uploaded, not linear: render targets, storage images, copy and clear destinations, whose memory holds only zeros until then:
// another port measured Mario Kart World 1.8.0's four 16384x16384 R8 terrain targets, 1,365 MB, never written in its scenes), those of 64 MB or more apart, and images not looked up for
// 1800 presents (about a minute). o[16..18]: images not sparse, the memory they were given, their NVN storage (the blocks' slack is the memory line's blocks less that memory).
// o[20..21]: sparse tiles, the places bound (slots: a promoted tile has two) and the distinct physical tiles behind them (nvn_sparse_tile_phys): without sparseResidencyAliased every place
// of a physical tile takes a slot of its own (upstream measured about 3 places a page in a race). o[22..24]: resolution families, the GPU's images over one pool offset in one format with
// different sizes (dynamic resolution: run 13's images went from 3,188 to 6,966 in its race), the images in them, and the memory one allocation per family (the largest) would leave over.
// o[25..26]: the GPU's images not looked up for 300 presents (transient targets the game moved to other offsets). o[27..30]: of the textures not candidates, those under the size
// floor (DM_VK_SHADOW_MIN_KB) and those written since their upload. Under g_img_mu, at the memory line's period
typedef struct { uint64_t pool; int64_t off; int fmt; uint64_t mem; } rfam;
static int rfam_cmp(const void *a, const void *b) { const rfam *x = a, *y = b; return x->pool != y->pool ? (x->pool < y->pool ? -1 : 1) : x->off != y->off ? (x->off < y->off ? -1 : 1) : x->fmt - y->fmt; }
void vk_tex_census(uint64_t o[32]) {
    memset(o, 0, 32 * sizeof *o);
    pthread_mutex_lock(&g_img_mu);
    size_t cap = (size_t)g_img_n + 16, nf = 0; rfam *fam = malloc(cap * sizeof *fam); size_t hcap = 1024; uintptr_t *ph = NULL;
    for (int m = 0; m < 2; m++) { nvn_hmap *h = m ? &g_by_serial : &g_by_key;
        for (uint32_t i = 0; i < h->cap; i++) { vkimg_t *im = h->vals[i]; if (!im) continue;
            int own = 0; uint64_t mem = im->sp ? 0 : im->dn ? im->dn->bytes : vk_image_sub_size(im->sub, &own);
            int gpu_class = !im->packaged && !im->lin_mem && !im->sp && !im->dn && !(im->flags & (NVN_TEXTURE_FLAGS_LINEAR | NVN_TEXTURE_FLAGS_LINEAR_RENDER_TARGET));
            if (g_tex_frame - im->used >= 1800) { o[12]++; o[13] += mem; }
            if (gpu_class && g_tex_frame - im->used >= 300) { o[25]++; o[26] += mem; }
            if (!im->sp) { o[16]++; o[17] += mem; o[18] += im->dn ? im->dn->bytes : im->size > 0 ? (uint64_t)im->size : 0; }
            if (im->packaged && im->pkg_mem && !im->sp && !im->dn) { int c = vk_shadow_tex_class(im), k = c == VK_SHC_OK ? 2 : c == VK_SHC_PLACED ? 4 : c == VK_SHC_RELEASED ? 6 : 8;
                o[0]++; o[1] += im->pkg_len; o[k]++; o[k + 1] += im->pkg_len; if (c == VK_SHC_SMALL || c == VK_SHC_WRITTEN) { int q = c == VK_SHC_SMALL ? 27 : 29; o[q]++; o[q + 1] += im->pkg_len; } }
            else if (gpu_class && !im->gpu_w) { o[10]++; o[11] += mem; if (mem >= (64ull << 20)) { o[14]++; o[15] += mem; } }
            if (gpu_class && !m && fam && nf < cap) fam[nf++] = (rfam){im->pool, im->off, im->nvn_format, mem};
            if (im->sp) {   // its bound places and the physical tiles behind them
                struct vksparse *s = im->sp; const pool_t *pl = (const pool_t *)(uintptr_t)im->pool; uint64_t base = pl && pl->mem ? (uint64_t)(uintptr_t)pl->mem + (uint64_t)im->off : 0;
                for (uint32_t y = 0; base && y < s->layers; y++) for (uint32_t ti = 0; ti < s->per_layer; ti++) {
                    if (ti == s->tail_at || !s->slot[((size_t)y * s->per_layer + ti) * s->nsub]) continue;
                    uint32_t l = s->first; while (l + 1 < s->ntiled && ti >= s->loff_t[l + 1]) l++;
                    uint64_t a = base + (uint64_t)y * s->layer_size + s->loff[l] + (uint64_t)(ti - s->loff_t[l]) * SP_TILE; const uint8_t *p = nvn_sparse_tile_phys(pl, a);
                    uintptr_t key = p ? (uintptr_t)p : (uintptr_t)a | 1; o[20] += s->nsub;
                    if (!ph || 2 * (o[21] + 1) > hcap) { size_t nc = ph ? 2 * hcap : hcap; uintptr_t *n2 = calloc(nc, sizeof *n2); if (!n2) continue;
                        for (size_t q = 0; ph && q < hcap; q++) if (ph[q]) { size_t j = (ph[q] >> 12) * 0x9E3779B97F4A7C15ull % nc; while (n2[j]) j = (j + 1) % nc; n2[j] = ph[q]; }
                        free(ph); ph = n2; hcap = nc; }
                    size_t j = (key >> 12) * 0x9E3779B97F4A7C15ull % hcap; while (ph[j] && ph[j] != key) j = (j + 1) % hcap;
                    if (!ph[j]) { ph[j] = key; o[21]++; } }
            }
        } }
    pthread_mutex_unlock(&g_img_mu);
    if (fam) { qsort(fam, nf, sizeof *fam, rfam_cmp);
        for (size_t i = 0; i < nf;) { size_t j = i; uint64_t sum = 0, mx = 0; while (j < nf && !rfam_cmp(&fam[i], &fam[j])) { sum += fam[j].mem; if (fam[j].mem > mx) mx = fam[j].mem; j++; }
            if (j - i > 1) { o[22]++; o[23] += j - i; o[24] += sum - mx; } i = j; }
        free(fam); }
    free(ph);
}
void vk_tex_drop_pool(const void *pool) {   // the pool is finalized: its textures' images go once the GPU is done with them
    pthread_mutex_lock(&g_img_mu);
    for (uint32_t i = 0; i < g_by_key.cap; i++) {
        if (!g_by_key.vals[i]) continue;
        akey k; memcpy(&k, g_by_key.keys + (size_t)i * g_by_key.klen, sizeof k);
        if (k.pool != (uint64_t)(uintptr_t)pool) continue;
        vkimg_t *im = g_by_key.vals[i]; nvn_hm_del(&g_by_key, &k); i = (uint32_t)-1;   // the table was rehashed by the delete: rescan
        img_drop(im);
    }
    pthread_mutex_unlock(&g_img_mu);
}
// CPU-written textures (WriteTexels / a dirty first use): packaged from now on. A finalized texture's serial moves to the retired set (mtl_tex_release): a registration or a record may
// still name it, and its content still comes from storage, but it no longer counts as live (mtl_cpu_tex_count, as nvn_mtl.m's g_cpu_tex_retired)
static nvn_hmap g_cpu_written = {.klen = sizeof(uint64_t)}, g_cpu_retired = {.klen = sizeof(uint64_t)}; static pthread_mutex_t g_cw_mu = PTHREAD_MUTEX_INITIALIZER;
void mtl_tex_cpu_written(const tex_t *t) { if (!t->serial) return; pthread_mutex_lock(&g_cw_mu); nvn_hm_put(&g_cpu_written, &t->serial, (void *)1); pthread_mutex_unlock(&g_cw_mu); }
unsigned mtl_cpu_tex_count(void) { pthread_mutex_lock(&g_cw_mu); unsigned n = g_cpu_written.n; pthread_mutex_unlock(&g_cw_mu); return n; }
uint64_t mtl_pkg_edge_skips(void) { return __atomic_load_n(&g_pkg_edge_skips, __ATOMIC_RELAXED); }   // (vk_tex_: written pages, the texture's own bytes unchanged)
// Storage the CPU wrote before the texture's first use (nvn_mtl.m tex_cpu_content). A block-compressed texture the GPU cannot render into holds what the CPU wrote, flushed or not, before or
// after nvnTextureInitialize (The Great Ace Attorney Chronicles decodes its court-record pictures, BC1 in a CPU_CACHED pool with no flush, into storage it reuses: black, or the previous
// item's picture; GitHub issue #568). An uncompressed 2D texture cannot be told from a render target by its flags, but a pool the GPU has not written (pool_t.gpu_written: a render target,
// an image, a texture clear or a copy into a texture recorded in it, hle_nvn.c mtl_pool_gpu_written) holds only what the CPU wrote: Mario Kart World writes its terrain tiles (260x260 R16
// height, RG8, RGBA8, R11G11B10F, 68x68 RGBA16) into a CPU_CACHED pool with no flush, and as GPU textures of zeros every tile was a flat plane at its base height (upstream 75e5ddd8). At the
// pool's first GPU write those become GPU textures with the content they have (cpu_unc_drop). An uncompressed 3D texture with none of the flags below is taken from storage in any pool
// (Xenoblade Chronicles 3's particle steering fields). DM_NVN_STORAGE_FIRST_USE=0 (start-up): off
static nvn_hmap g_cpu_unc = {.klen = sizeof(uint64_t)};   // serial → its pool: uncompressed textures taken from storage in a pool the GPU had not written (cpu_unc_drop), under g_cw_mu
static _Atomic int g_pool_gpuw_new;   // a pool has had its first GPU write since cpu_unc_drop last looked
static int mem_any(const uint8_t *p, uint64_t n) { uint64_t i = 0, v; for (; i + 8 <= n; i += 8) { memcpy(&v, p + i, 8); if (v) return 1; } for (; i < n; i++) if (p[i]) return 1; return 0; }   // (stops at the first non-zero byte)
static int tex_cpu_content(const tex_t *t, const vkfmt_t *f) {
    static int on = -1; if (on < 0) { const char *e = getenv("DM_NVN_STORAGE_FIRST_USE"); on = !e || atoi(e) != 0; }
    if (!on || !t->mem || !t->pool || t->has_layout || t->d.samples > 1 || t->d.target == NVN_TEXTURE_TARGET_BUFFER) return 0;
    int comp = f->bw > 1 || f->bh > 1, unc = !comp && t->d.target != NVN_TEXTURE_TARGET_3D;   // (block-compressed storage, a BC format decoded on the CPU included)
    if (unc && (atomic_load_explicit(&t->pool->gpu_written, memory_order_relaxed) || f->depth || f->stencil || (t->d.flags & (NVN_TEXTURE_FLAGS_DISPLAY | NVN_TEXTURE_FLAGS_VIDEO_DECODE)))) return 0;
    if (t->pool->flags & (NVN_MEMORY_POOL_FLAGS_CPU_NO_ACCESS | NVN_MEMORY_POOL_FLAGS_VIRTUAL)) return 0;   // the CPU cannot write it (hle_nvn.c fills it once at Initialize) / sparse (read through the page table)
    // IMAGE only matters for an uncompressed texture, as shaders cannot write a block-compressed one (Pokopia gives nearly every texture IMAGE, its icons BC1 / BC7)
    if ((t->d.flags & (NVN_TEXTURE_FLAGS_COMPRESSIBLE | NVN_TEXTURE_FLAGS_LINEAR | NVN_TEXTURE_FLAGS_LINEAR_RENDER_TARGET | NVN_TEXTURE_FLAGS_SPARSE)) || ((t->d.flags & NVN_TEXTURE_FLAGS_IMAGE) && !comp)) return 0;
    if (!vk_readable(t->mem, t->L.size) || !mem_any(t->mem, t->L.size)) return 0;   // (storage not wholly mapped: no content to take)
    if (unc) { uint64_t sr = t->serial; pthread_mutex_lock(&g_cw_mu); nvn_hm_put(&g_cpu_unc, &sr, (void *)t->pool); pthread_mutex_unlock(&g_cw_mu); }
    static uint64_t n[2]; if (n[unc]++ < 4) VKLOG("texture written by the CPU before its first use %ux%ux%u nvn format %#x%s, pool flags %#x: uploaded from its storage, again when it changes (%llu so far)",
        t->d.width, t->d.height, t->d.depth, t->d.format, unc ? " (uncompressed, in a pool the GPU has not written)" : "", t->pool->flags, (unsigned long long)n[unc]);
    return 1;
}
void mtl_pool_gpu_written(const void *pool) {   // hle_nvn.c, when it records a GPU write into the pool (see tex_cpu_content)
    pool_t *p = (pool_t *)pool;
    if (p && !atomic_load_explicit(&p->gpu_written, memory_order_relaxed)) { atomic_store(&p->gpu_written, 1); atomic_store(&g_pool_gpuw_new, 1); }
}
// A pool's first GPU write: its storage no longer holds the whole truth, so the uncompressed textures taken from it become GPU textures with the content they have (no more uploads)
static void cpu_unc_drop(void) {
    if (!atomic_load_explicit(&g_pool_gpuw_new, memory_order_relaxed) || !atomic_exchange(&g_pool_gpuw_new, 0)) return;
    uint32_t n = 0;
    pthread_mutex_lock(&g_cw_mu);
    for (uint32_t i = 0; i < g_cpu_unc.cap;) {
        const pool_t *p = g_cpu_unc.vals[i];
        if (!p || !atomic_load_explicit(&p->gpu_written, memory_order_relaxed)) { i++; continue; }
        uint64_t sr; memcpy(&sr, g_cpu_unc.keys + (size_t)i * 8, 8); nvn_hm_del(&g_cpu_unc, &sr); nvn_hm_del(&g_cpu_written, &sr); nvn_hm_del(&g_cpu_retired, &sr); n++;   // (the deletion may shift a later entry into slot i)
    }
    pthread_mutex_unlock(&g_cw_mu);
    if (n) { static int nl; if (nl++ < 8) VKLOG("a pool's first GPU write: %u uncompressed textures taken from its storage are GPU textures from now on", n); }
}
// Textures registered as images (mtl_reg_image) whose builder omitted IMAGE: their image is made with storage use (vk_tex_)
static nvn_hmap g_img_use = {.klen = sizeof(uint64_t)};
static int img_use(uint64_t serial) { if (!__atomic_load_n(&g_img_use.n, __ATOMIC_RELAXED)) return 0; pthread_mutex_lock(&g_cw_mu); int r = nvn_hm_get(&g_img_use, &serial) != NULL; pthread_mutex_unlock(&g_cw_mu); return r; }
// The NVN formats whose Vulkan format is the sRGB / linear twin of f's (srgb_twin): an RGBA8 texture has RGBA8_SRGB and RGBX8_SRGB (vk_tex_'s twin views). Only the formats srgb_twin pairs
static const uint8_t *nvn_twins(int f) {   // (g_img_mu held)
    static uint8_t tw[0x9D][4]; static int done;
    if (!done) { done = 1; static const uint8_t cand[] = {0x25, 0x37, 0x38, 0x6D, 0x99, 0x9A, 0x9B, 0x9C};
        for (unsigned a = 0; a < sizeof cand; a++) { const vkfmt_t *fa = vk_fmt(cand[a]); VkFormat t2 = fa ? srgb_twin(fa->vk) : VK_FORMAT_UNDEFINED; int n = 0;
            for (unsigned b = 0; t2 && b < sizeof cand && n < 3; b++) { const vkfmt_t *fb = vk_fmt(cand[b]); if (b != a && fb && fb->vk == t2) tw[cand[a]][n++] = cand[b]; } } }
    return f > 0 && f < 0x9D ? tw[f] : tw[0];
}
// Linear textures (LINEAR) in a pool are their storage on the Switch: the CPU writes the pool memory and the GPU reads it (Metal's tex_linear makes them on the pool buffer).
// Hades II rasterizes its glyphs on the CPU into a 1024x1216 R8 IMAGE|LINEAR atlas (pool CPU_UNCACHED|GPU_CACHED, no flush, no WriteTexels) and decodes its Bink movies
// into linear planes: as GPU-only images of zeros all text, the title art and the movies were blank. A Vulkan image cannot be bound to the pool's memory (a dedicated
// AHardwareBuffer import, bound to its buffer), so each upload command buffer that a submission's draws need it in copies it from the pool buffer on the GPU (lin_copy).
// The packaged path (CPU upload, re-upload on written pages) is the fallback; as the only path the page faults of the CPU's writes cost Hades II about 120 a frame and
// 2 million while loading. Not LINEAR_RENDER_TARGET: the GPU draws into those, and a copy from storage would replace what it drew
static int linear_tex(const tex_t *t) { return t->pool && (t->d.flags & (NVN_TEXTURE_FLAGS_LINEAR | NVN_TEXTURE_FLAGS_LINEAR_RENDER_TARGET)) == NVN_TEXTURE_FLAGS_LINEAR; }
// lin_copy takes LINEAR_RENDER_TARGET textures too: Super Mario Bros. Wonder's CPU writes its sky's four 64x1 RGBA16F gradients straight into such textures (NVN flags 0x40)
// that nothing draws into, and as GPU-only images of zeros the sky drew diagonal bands of saturated colour over every level. What the GPU draws into one (gpu_w) stays, as for a
// cleared or copied linear texture, until the CPU writes its storage again. The packaged path (linear_tex) still leaves them out: a re-upload there would replace a drawing
static int linear_storage(const tex_t *t) { return t->pool && (t->d.flags & (NVN_TEXTURE_FLAGS_LINEAR | NVN_TEXTURE_FLAGS_LINEAR_RENDER_TARGET)); }
extern uint64_t g_up_gen;
static int lin_copy(vkimg_t *im, const tex_t *t, const vkfmt_t *f) {   // 1 = the image holds its storage as of this upload buffer
    if (!linear_storage(t) || im->levels != 1 || im->d != 1 || im->layers != 1 || f->bw > 1 || f->bh > 1 || !f->bpb || im->aspect != VK_IMAGE_ASPECT_COLOR_BIT) return 0;
    size_t stride = t->d.stride ? t->d.stride : (size_t)im->w * f->bpb; VkBuffer b; VkDeviceSize o;
    if (stride % f->bpb || stride < (size_t)im->w * f->bpb || !vk_buf_at((uint64_t)(uintptr_t)t->mem, (uint64_t)stride * (im->h - 1) + (uint64_t)im->w * f->bpb, &b, &o) || o % f->bpb) return 0;
    if (!vk_readable(t->mem, (uint64_t)stride * (im->h - 1) + (uint64_t)im->w * f->bpb)) return 0;   // (its pages tracked below: not over a range the game has not mapped)
    // the GPU wrote it (a clear, a copy, a storage write): it holds that until the CPU writes its storage again, which its pages, set read-only, tell
    size_t len = stride * (im->h - 1) + (size_t)im->w * f->bpb;
    if (im->gpu_w && !im->lin_mem) { nvn_dirty_protect(t->mem, len); im->lin_stamp = nvn_dirty_now(); im->lin_span = nvn_dirty_span(t->mem, len); im->lin_mem = t->mem; im->lin_len = len; return 1; }
    if (im->gpu_w) { if (!nvn_dirty_since_span(im->lin_span, t->mem, len, im->lin_stamp)) return 1; nvn_dirty_release(im->lin_mem, im->lin_len); im->lin_mem = NULL; im->gpu_w = 0; im->lin_gen = 0; }
    VkCommandBuffer cb = vk_upload_cmd();
    if (im->lin_gen == g_up_gen) return 1;
    im->lin_gen = g_up_gen;
    vk_img_barrier_all(cb);   // after the reads of earlier submissions
    if (im->layout != VK_IMAGE_LAYOUT_GENERAL) vk_tex_prepare(cb, im);
    VkBufferImageCopy r = {o, (uint32_t)(stride / f->bpb), 0, {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}, {0, 0, 0}, {im->w, im->h, 1}};
    vkCmdCopyBufferToImage(cb, b, im->img, VK_IMAGE_LAYOUT_GENERAL, 1, &r);
    return 1;
}
static int packaged_tex(const tex_t *t) {
    if (t->has_layout || linear_tex(t)) return 1;
    pthread_mutex_lock(&g_cw_mu); int r = nvn_hm_get(&g_cpu_written, &t->serial) != NULL || nvn_hm_get(&g_cpu_retired, &t->serial) != NULL; pthread_mutex_unlock(&g_cw_mu);
    return r;
}
void mtl_tex_init(const tex_t *t) { (void)t; }
void mtl_tex_release(tex_t *t) {   // images belong to their storage key, dropped with the pool; a CPU-written serial is retired (no longer live, still from storage)
    if (!t || !t->serial) return; uint64_t sr = t->serial;
    pthread_mutex_lock(&g_cw_mu); if (nvn_hm_del(&g_cpu_written, &sr)) nvn_hm_put(&g_cpu_retired, &sr, (void *)1); nvn_hm_del(&g_img_use, &sr); pthread_mutex_unlock(&g_cw_mu);
}
int mtl_tex_gpu_owned(uint64_t serial) { (void)serial; return 0; }

// ---------------- views ----------------
typedef struct vview { VkImageView v; VkImageViewType type; VkFormat fmt; uint32_t lv0, nlv, ly0, nly; VkComponentMapping swz; VkImageAspectFlags aspect; int attach; struct vview *next; } vview;   // attach: an attachment view (its usage differs from a sampling view's)
static nvn_hmap g_views = {.klen = sizeof(vkimg_t *)};   // vkimg_t* → vview list
static pthread_mutex_t g_view_mu = PTHREAD_MUTEX_INITIALIZER;
extern uint64_t g_tex_gen;
extern uint64_t g_tex_dgen; uint64_t g_tex_kgen = 1;   // kgen: views went of an image a kept descriptor set names (vkimg_t kept_gen)
static void vk_tex_view_drop_all(vkimg_t *im) {
    pthread_mutex_lock(&g_view_mu);
    vview *v = nvn_hm_get(&g_views, &im); g_tex_gen++; g_tex_dgen++;
    { extern uint64_t g_kgen; uint64_t kg = __atomic_exchange_n(&im->kept_gen, 0, __ATOMIC_ACQ_REL);   // (a set kept since the last flush names one of these views: gfx_vk_exec.c kept_check)
      if (kg && kg == __atomic_load_n(&g_kgen, __ATOMIC_ACQUIRE)) __atomic_fetch_add(&g_tex_kgen, 1, __ATOMIC_RELEASE); }
    while (v) { vview *n = v->next; vk_defer_free_view(v->v); free(v); v = n; }
    nvn_hm_del(&g_views, &im);
    pthread_mutex_unlock(&g_view_mu);
}
static VkComponentSwizzle swz_comp(int nvn) {   // NVN swizzle enum: 0 zero 1 one 2 R 3 G 4 B 5 A
    switch (nvn) { case 0: return VK_COMPONENT_SWIZZLE_ZERO; case 1: return VK_COMPONENT_SWIZZLE_ONE; case 2: return VK_COMPONENT_SWIZZLE_R; case 3: return VK_COMPONENT_SWIZZLE_G; case 4: return VK_COMPONENT_SWIZZLE_B; default: return VK_COMPONENT_SWIZZLE_A; }
}
static VkComponentSwizzle apply(VkComponentSwizzle outer, const VkComponentMapping *inner, VkComponentSwizzle self) {   // outer picks a channel of what inner produces
    if (outer == VK_COMPONENT_SWIZZLE_IDENTITY) outer = self;
    switch (outer) {
    case VK_COMPONENT_SWIZZLE_R: return inner->r == VK_COMPONENT_SWIZZLE_IDENTITY ? VK_COMPONENT_SWIZZLE_R : inner->r;
    case VK_COMPONENT_SWIZZLE_G: return inner->g == VK_COMPONENT_SWIZZLE_IDENTITY ? VK_COMPONENT_SWIZZLE_G : inner->g;
    case VK_COMPONENT_SWIZZLE_B: return inner->b == VK_COMPONENT_SWIZZLE_IDENTITY ? VK_COMPONENT_SWIZZLE_B : inner->b;
    case VK_COMPONENT_SWIZZLE_A: return inner->a == VK_COMPONENT_SWIZZLE_IDENTITY ? VK_COMPONENT_SWIZZLE_A : inner->a;
    default: return outer;
    }
}
static VkComponentMapping sample_swizzle(const VkComponentMapping *format, const tex_t *t, const view_t *v, int attachment) {
    if (attachment) return (VkComponentMapping){0};
    // NVN view selectors override texture selectors; both refer to the NVN
    // format's channels. Applying the texture first composes two selectors
    // and can turn a requested mask channel into a different component.
    int sw[4];
    if (v && v->set_swizzle) for (int c=0;c<4;c++) sw[c]=v->swz[c];
    else if (t->set_swz) for (int c=0;c<4;c++) sw[c]=(t->swz>>(3*c))&7;
    else return *format;
    return (VkComponentMapping){
        apply(swz_comp(sw[0]),format,VK_COMPONENT_SWIZZLE_R),
        apply(swz_comp(sw[1]),format,VK_COMPONENT_SWIZZLE_G),
        apply(swz_comp(sw[2]),format,VK_COMPONENT_SWIZZLE_B),
        apply(swz_comp(sw[3]),format,VK_COMPONENT_SWIZZLE_A)};
}
__thread VkFormat vk_view_fmt; __thread uint32_t vk_view_layers;
VkImageView vk_tex_view(const tex_t *t, const view_t *vw, int dim, int arrayed, int as_attachment, vkimg_t **img_out) {
    vkimg_t *im = vk_tex(t); if (img_out) *img_out = im;
    if (!im) return VK_NULL_HANDLE;
    if (as_attachment) vk_img_wrote(im); else if (im->novl) alias_check(im);
    vk_watch(im, as_attachment ? "attachment" : "sampled or image");
    if (!as_attachment) { void vk_texdump_note(vkimg_t *im); vk_texdump_note(im); vk_dsc_read(im, "sampled or bound as an image"); }
    const vkfmt_t *f = vk_fmt(vw && vw->set_format ? vw->format : t->d.format); if (!f) f = vk_fmt(t->d.format);
    uint32_t lv0 = vw && vw->set_levels ? vw->base_level : 0, nlv = vw && vw->set_levels ? vw->nlevels : im->levels + im->lv0 - (lv0 < im->levels + im->lv0 ? lv0 : 0);
    if (im->lv0) { uint32_t skip = lv0 < im->lv0 ? im->lv0 - lv0 : 0; lv0 = lv0 > im->lv0 ? lv0 - im->lv0 : 0; nlv = nlv > skip ? nlv - skip : 1; }   // the texture's levels → the image's (lv0): one it lacks starts at its level 0
    if (lv0 >= im->levels) lv0 = im->levels - 1; if (!nlv || lv0 + nlv > im->levels) nlv = im->levels - lv0;
    // a sparse pool's texture uploaded whole is sampled from its first mapped level (pkg_upload_: the translator drops TEX's LOD clamp and residency predicate, so Monster Hunter Rise's
    // streamed textures were sampled at levels not mapped yet, which read 0: ASTC's error colour, magenta clothes on its title screen; nvn_mtl.m sparse_view)
    if (!as_attachment && im->sp_vbase > lv0 && im->sp_vbase < lv0 + nlv) { nlv -= im->sp_vbase - lv0; lv0 = im->sp_vbase; }
    // a 3D image drawn into: the view's layers are depth slices of its level (the image is 2D-array compatible). Xenoblade Chronicles 3 renders its 16x16x16 colour grading
    // LUT one slice per pass through 2D views at layers 0-15; clamped to the image's one layer, every slice went to slice 0, and the graded title sky came out flat blue
    uint32_t nlay = as_attachment && im->vtype == VK_IMAGE_VIEW_TYPE_3D ? (im->d >> lv0 ? im->d >> lv0 : 1) : im->layers;
    uint32_t ly0 = vw && vw->set_layers ? vw->min_layer : 0, nly = vw && vw->set_layers ? vw->nlayers : nlay - ly0;
    if (ly0 >= nlay) ly0 = nlay - 1; if (!nly || ly0 + nly > nlay) nly = nlay - ly0;
    int target = vw && vw->set_target ? vw->target : im->target;
    VkImageViewType type = vtype_of(target);
    // what the shader declares wins over the object's target (a 2D render target sampled as an array, a cube face read as 2D)
    if (dim != 0xff && !as_attachment) {
        VkImageViewType want = dim == 0 ? (arrayed ? VK_IMAGE_VIEW_TYPE_1D_ARRAY : VK_IMAGE_VIEW_TYPE_1D) : dim == 1 ? (arrayed ? VK_IMAGE_VIEW_TYPE_2D_ARRAY : VK_IMAGE_VIEW_TYPE_2D) :
                               dim == 2 ? VK_IMAGE_VIEW_TYPE_3D : dim == 3 ? (arrayed ? VK_IMAGE_VIEW_TYPE_CUBE_ARRAY : VK_IMAGE_VIEW_TYPE_CUBE) : type;
        if (want == VK_IMAGE_VIEW_TYPE_2D && type == VK_IMAGE_VIEW_TYPE_2D_ARRAY) nly = 1;
        if ((want == VK_IMAGE_VIEW_TYPE_CUBE || want == VK_IMAGE_VIEW_TYPE_CUBE_ARRAY) && (im->layers < 6 || nly % 6 || !(im->cflags & VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT))) want = type;   // not a cube: keep it as it is
                                                                                                    // (a cube view of an image made without CUBE_COMPATIBLE is invalid: gfx_vk_exec.c tex_desc_for binds the zero cube)
        if (want == VK_IMAGE_VIEW_TYPE_CUBE) nly = 6;
        if (want == VK_IMAGE_VIEW_TYPE_3D && im->d == 1) want = type;
        if ((want == VK_IMAGE_VIEW_TYPE_1D || want == VK_IMAGE_VIEW_TYPE_1D_ARRAY) != (type == VK_IMAGE_VIEW_TYPE_1D || type == VK_IMAGE_VIEW_TYPE_1D_ARRAY)) want = type;
        type = want;
    }
    if (as_attachment) { type = im->layers > 1 && nly > 1 ? VK_IMAGE_VIEW_TYPE_2D_ARRAY : (im->vtype == VK_IMAGE_VIEW_TYPE_3D ? VK_IMAGE_VIEW_TYPE_2D_ARRAY : VK_IMAGE_VIEW_TYPE_2D); nlv = 1; }
    if (type == VK_IMAGE_VIEW_TYPE_2D || type == VK_IMAGE_VIEW_TYPE_1D || type == VK_IMAGE_VIEW_TYPE_3D) nly = 1;
    VkImageAspectFlags aspect = im->aspect;
    if (aspect == (VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT) && !as_attachment)
        aspect = (!f->depth || t->stencil || (vw && vw->set_dsmode && vw->dsmode == 1)) ? VK_IMAGE_ASPECT_STENCIL_BIT : VK_IMAGE_ASPECT_DEPTH_BIT;   // (STENCIL8 alone may be stored combined)
    VkComponentMapping swz = sample_swizzle(&f->swz,t,vw,as_attachment);
    VkFormat fmt = f->vk;
    const vkfmt_t *imf = vk_fmt(im->nvn_format);   // (a BC format decoded on the CPU counts as compressed: its view keeps the decoded image's format)
    if ((im->aspect & ~VK_IMAGE_ASPECT_COLOR_BIT) || (f->compressed || f->dec) != (imf->compressed || imf->dec) || (f->dec && f->dbpp != imf->dbpp)) fmt = im->fmt;   // depth views and block-size changes keep the image's format
    if (im->fixed_fmt && fmt != im->fmt && fmt != srgb_twin(im->fmt)) { fmt = im->fmt; static int n; if (n++ < 8) VKLOG("[RED] view format %d of a format-listed image (%d, DM_VK_UBWC): the image's own used", f->vk, im->fmt); }
    // an image in its texture format's wider twin ("promoted formats"): every view in that twin, so attachments, pipelines (vk_view_fmt) and render passes take it too. A view in another
    // format (an integer or sRGB reinterpretation) has no twin holding the same values: the image's own, [RED]
    if (im->promo && fmt != im->fmt) {
        if (fmt != PROMO[im->promo - 1].from) { static int n; if (n++ < 8) VKLOG("[RED] a view in format %d of a sparse texture made in %d, the wider twin of its format %d: the image's own used", fmt, im->fmt, PROMO[im->promo - 1].from); }
        fmt = im->fmt; }
    pthread_mutex_lock(&g_view_mu);
    vview *head = nvn_hm_get(&g_views, &im);
    for (vview *v = head; v; v = v->next)
        if (v->type == type && v->fmt == fmt && v->lv0 == lv0 && v->nlv == nlv && v->ly0 == ly0 && v->nly == nly && v->aspect == aspect && v->attach == (as_attachment != 0) && !memcmp(&v->swz, &swz, sizeof swz)) { pthread_mutex_unlock(&g_view_mu); vk_view_fmt = fmt; vk_view_layers = nly; return v->v; }
    VkImageViewUsageCreateInfo uci = {VK_STRUCTURE_TYPE_IMAGE_VIEW_USAGE_CREATE_INFO, NULL, 0};
    VkFormatProperties fp; vkGetPhysicalDeviceFormatProperties(V.pd, fmt, &fp);
    uci.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    if (as_attachment) uci.usage = (fp.optimalTilingFeatures & VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT ? VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT : 0) | (fp.optimalTilingFeatures & VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT ? VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT : 0);
    else if ((im->usage & VK_IMAGE_USAGE_STORAGE_BIT) && (fp.optimalTilingFeatures & VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT)) uci.usage |= VK_IMAGE_USAGE_STORAGE_BIT;   // (the image's: IMAGE, or registered as an image, mtl_reg_image)
    VkImageViewCreateInfo ci = {VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, &uci, 0, im->img, type, fmt, swz, {aspect, lv0, nlv, ly0, nly}};
    VkImageView view; VkResult r = vkCreateImageView(V.dev, &ci, NULL, &view);
    if (r) { pthread_mutex_unlock(&g_view_mu); VKRED_ONCE(15, "vkCreateImageView failed (%d): type %d format %d levels %u+%u layers %u+%u of image %ux%ux%u", r, type, fmt, lv0, nlv, ly0, nly, im->w, im->h, im->layers); return VK_NULL_HANDLE; }
    vview *nv = calloc(1, sizeof *nv); *nv = (vview){view, type, fmt, lv0, nlv, ly0, nly, swz, aspect, as_attachment != 0, head};
    nvn_hm_put(&g_views, &im, nv);
    pthread_mutex_unlock(&g_view_mu);
    vk_view_fmt = fmt; vk_view_layers = nly;
    return view;
}

// ---------------- texture / sampler ID tables ----------------
// A registration as the replay of each submission sees it. NVN's texture pool is memory the GPU reads as it runs, and a game rewrites an ID once the work
// that used it is done; with the replay on its own thread (DM_NVN_ASYNC) a submission is replayed after the game has gone on, and Xenoblade Chronicles 3
// rewrites about 90 IDs a second with other textures: a frame replayed late drew with the next frame's textures (whole-screen garbage while the camera
// turned). So each registration keeps the number of replay jobs queued when it was made, and a job sees the newest one made before it was queued
// (nvn_jobs_queued / nvn_replaying_job, gfx/nvn/hle_nvn.c); without a replay thread that is always the newest. TREG_OLD older ones per ID, kept for IDs rewritten
enum { TREG_OLD = 3 };
typedef struct { tex_t t; view_t v; uint8_t ok, has_v, gone; uint64_t seq; } treg_ver;   // gone: its memory pool finalized (mtl_reg_tex_gone): it reads as the dummy
typedef struct { treg_ver cur; treg_ver *old; } treg_t;   // old: TREG_OLD earlier versions, newest first (allocated at the first rewrite)
static treg_t *g_treg; static int g_ntreg;
uint64_t g_tex_gen = 1;   // changes whenever a texture ID is registered again or views go (gfx_vk_exec.c's per-submission view cache)
uint64_t g_samp_gen = 1;  // changes whenever a sampler ID is registered with other parameters (gfx_vk_exec.c samp_cached and the descriptor set memo)
static samp_par *g_sreg; static uint8_t *g_sreg_ok; static VkSampler *g_samp; static int g_nsreg;
// Samplers as a bind saw them. Super Mario Odyssey hands out sampler IDs from a cache of slots and registers a slot again with other parameters while a command
// buffer it has not submitted yet still binds that ID: the first pass of its bloom (a 400x228 chain, the level read chosen by the sampler's LOD range 0..0) bound a
// sampler that was then registered again as LOD 1..1 to 3..3, and read the chain's previous upsampled level. The bloom came out 2.7 times too strong for that frame:
// the glow around the moon flashed while the camera turned, in about 1% of frames (3 of 332 first passes; issue #42). Every registration that changes an ID's
// parameters counts (g_sregs), each bind records the count when it was made (mtl_samp_seq, hle_nvn.c), and a bind resolves its sampler to the newest version
// registered before it (vk_samp_at): SREG_OLD earlier versions are kept per ID. DM_NVN_SAMP_SNAP=0 (live, gfx_vk_exec.c): the newest, as before
enum { SREG_OLD = 3 };
typedef struct { samp_par p; uint64_t seq; } sreg_ver;
static uint64_t *g_sseq; static sreg_ver *g_sold; static uint64_t g_sregs;   // the current version's count, SREG_OLD older ones per ID (newest first)
uint64_t mtl_samp_seq(void) { return __atomic_load_n(&g_sregs, __ATOMIC_ACQUIRE); }
static pthread_mutex_t g_rmu = PTHREAD_MUTEX_INITIALIZER;
__attribute__((weak)) uint64_t nvn_jobs_queued(void) { return 0; }
__attribute__((weak)) uint64_t nvn_replaying_job(void) { return 0; }
// Each texture ID's registrations that changed it (gfx_vk_exec.c tex_desc_for keeps an ID's view while its count holds, g_tex_gen being any ID's), and the images and views dropped and
// the sparse views moved (g_tex_dgen): what else makes a view kept by ID stale. IDs past TVER_N count in TVER_N - 1 for all of them
enum { TVER_N = 1 << 20 };   // (texture handles carry 20 bits of ID)
_Atomic uint32_t g_tver[TVER_N]; uint64_t g_tex_dgen = 1;
// The same registration again (the same texture object and view at the ID it holds: an engine refreshing its texture table, Mario Kart World ~87 registrations a frame in a race, the
// Mac's race1.frames.tsv) changes nothing, and drops nothing kept by registration (the views by ID, the descriptor set memo: g_tex_gen). DM_VK_REG_SAME=0 (start-up): every one counts
static int reg_same_on(void) { static int v = -1; if (v < 0) { const char *e = getenv("DM_VK_REG_SAME"); v = !(e && !atoi(e)); } return v; }
void mtl_reg_tex(int tid, const tex_t *t, const view_t *v) {
    if (tid < 0) return;
    uint64_t seq = nvn_jobs_queued();
    pthread_mutex_lock(&g_rmu);
    if (tid >= g_ntreg) { int n = tid + 1024; g_treg = realloc(g_treg, n * sizeof *g_treg); memset(g_treg + g_ntreg, 0, (n - g_ntreg) * sizeof *g_treg); g_ntreg = n; }
    treg_t *r = &g_treg[tid];
    if (reg_same_on() && r->cur.ok && !r->cur.gone && t && !memcmp(&r->cur.t, t, sizeof *t) && r->cur.has_v == (v != NULL) && (!v || !memcmp(&r->cur.v, v, sizeof *v))) { pthread_mutex_unlock(&g_rmu); return; }
    g_tex_gen++; atomic_fetch_add_explicit(&g_tver[tid < TVER_N ? tid : TVER_N - 1], 1, memory_order_release);
    if (r->cur.ok && seq) {   // keep the one replaced for the jobs queued before this registration
        if (!r->old) r->old = calloc(TREG_OLD, sizeof *r->old);
        if (r->old) { memmove(r->old + 1, r->old, sizeof *r->old * (TREG_OLD - 1)); r->old[0] = r->cur; }
    }
    r->cur.ok = t != NULL; if (t) r->cur.t = *t; r->cur.has_v = v != NULL; if (v) r->cur.v = *v; r->cur.seq = seq; r->cur.gone = 0;
    pthread_mutex_unlock(&g_rmu);
}
// TexturePoolRegisterImage: an image descriptor at the ID (images share the texture pool's IDs, hle_nvn.c), which declares image use even when the texture's builder omitted IMAGE
// (nvn_mtl.m mtl_reg_image): the texture's image is then made with storage use (vk_tex_, img_use), so a storage view of it is valid
void mtl_reg_image(int tid, const tex_t *t, const view_t *v) {
    if (t && t->serial && !(t->d.flags & NVN_TEXTURE_FLAGS_IMAGE)) { uint64_t sr = t->serial; pthread_mutex_lock(&g_cw_mu); nvn_hm_put(&g_img_use, &sr, (void *)1); pthread_mutex_unlock(&g_cw_mu); }
    mtl_reg_tex(tid, t, v);
}
// The ID stays registered, its memory pool finalized (nvn_mtl.m: a captured registry entry as the live run left it): the pool's images went with it (vk_tex_drop_pool), so it reads as the dummy
void mtl_reg_tex_gone(int tid) { pthread_mutex_lock(&g_rmu); if (tid > 0 && tid < g_ntreg) { g_treg[tid].cur.gone = 1; g_tex_gen++; atomic_fetch_add_explicit(&g_tver[tid < TVER_N ? tid : TVER_N - 1], 1, memory_order_release); } pthread_mutex_unlock(&g_rmu); }
void mtl_set_tex_pool_n(int n) { (void)n; }
static __thread treg_ver t_treg;   // the version a lookup found, copied (the table may be rewritten or grow meanwhile)
vkimg_t *vk_tex_by_id(int tid, const view_t **view_out, const tex_t **tex_out) {
    uint64_t job = nvn_replaying_job();
    pthread_mutex_lock(&g_rmu);
    const treg_ver *r = tid > 0 && tid < g_ntreg ? &g_treg[tid].cur : NULL;
    if (r && job && r->seq >= job && g_treg[tid].old)   // registered after this job was queued: the newest earlier one
        for (int k = 0; k < TREG_OLD; k++) { const treg_ver *o = &g_treg[tid].old[k]; if (!o->ok) break; r = o; if (o->seq < job) break; }
    int ok = r && r->ok && !r->gone; if (ok) t_treg = *r;
    pthread_mutex_unlock(&g_rmu);
    if (!ok) return NULL;
    if (view_out) *view_out = t_treg.has_v ? &t_treg.v : NULL; if (tex_out) *tex_out = &t_treg.t;
    return (vkimg_t *)1;   // presence only: the caller makes the view with vk_tex_view
}
static VkSamplerAddressMode wrap(int w) {
    switch (w) { case 1: return VK_SAMPLER_ADDRESS_MODE_REPEAT; case 6: return VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT; case 2: case 3: return VK_SAMPLER_ADDRESS_MODE_MIRROR_CLAMP_TO_EDGE;
                 case 4: case 5: return VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER; default: return VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE; }
}
static VkCompareOp cmp_op(int f) { return f >= 1 && f <= 8 ? (VkCompareOp)(f - 1) : VK_COMPARE_OP_ALWAYS; }   // NVN NEVER..ALWAYS = 1..8, Vulkan 0..7 in the same order
static VkSampler g_cmp_lin[64]; static volatile int g_ncmp_lin;   // depth-compare samplers with linear filtering (make_sampler: under g_rmu; read without it, the list only grows)
// Samplers with a linear minification or magnification filter, for gfx_vk_exec.c's check of the formats they are used with (vk_samp_linear): an open-addressed set of handles, filled by
// make_sampler and read without a lock (a lookup racing an insert may miss it: the check is a diagnostic). VkSamplers are never destroyed, and few parameter sets exist
enum { LINSET = 2048 }; static VkSampler g_lin_s[LINSET];
static void lin_note(VkSampler s) { uint64_t h = (uint64_t)(uintptr_t)s * 0x9E3779B97F4A7C15ull; for (unsigned k = 0, i = (unsigned)(h >> 53) % LINSET; k < LINSET; k++, i = (i + 1) % LINSET) {
    VkSampler e = __atomic_load_n(&g_lin_s[i], __ATOMIC_ACQUIRE); if (e == s) return; if (!e) { __atomic_store_n(&g_lin_s[i], s, __ATOMIC_RELEASE); return; } } }
int vk_samp_linear(VkSampler s) { if (!s) return 0; uint64_t h = (uint64_t)(uintptr_t)s * 0x9E3779B97F4A7C15ull;
    for (unsigned k = 0, i = (unsigned)(h >> 53) % LINSET; k < LINSET; k++, i = (i + 1) % LINSET) { VkSampler e = __atomic_load_n(&g_lin_s[i], __ATOMIC_ACQUIRE); if (e == s) return 1; if (!e) return 0; } return 0; }
static VkSampler make_sampler(const samp_par *p) {
    VkSamplerCustomBorderColorCreateInfoEXT cb = {VK_STRUCTURE_TYPE_SAMPLER_CUSTOM_BORDER_COLOR_CREATE_INFO_EXT};
    VkSamplerReductionModeCreateInfo rd = {VK_STRUCTURE_TYPE_SAMPLER_REDUCTION_MODE_CREATE_INFO, NULL, p->reduction == 1 ? VK_SAMPLER_REDUCTION_MODE_MIN : p->reduction == 2 ? VK_SAMPLER_REDUCTION_MODE_MAX : VK_SAMPLER_REDUCTION_MODE_WEIGHTED_AVERAGE};
    VkSamplerCreateInfo ci = {VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO, p->reduction ? &rd : NULL};
    ci.magFilter = p->magf ? VK_FILTER_LINEAR : VK_FILTER_NEAREST;
    ci.minFilter = p->minf == 1 || p->minf == 3 || p->minf == 5 ? VK_FILTER_LINEAR : VK_FILTER_NEAREST;
    ci.mipmapMode = p->minf == 4 || p->minf == 5 ? VK_SAMPLER_MIPMAP_MODE_LINEAR : VK_SAMPLER_MIPMAP_MODE_NEAREST;
    ci.addressModeU = wrap(p->ws); ci.addressModeV = wrap(p->wt); ci.addressModeW = wrap(p->wr);
    ci.mipLodBias = p->lod_bias;
    ci.anisotropyEnable = p->aniso > 1.0f; ci.maxAnisotropy = p->aniso > V.max_aniso ? V.max_aniso : p->aniso;
    ci.compareEnable = p->cmp_mode == 1; ci.compareOp = cmp_op(p->cmp_func);
    // no mipmap filter: level 0 only (GL's rule; Maxwell's sampler with no mip filter reads the view's base level whatever its LOD clamp). A clamp whose minimum is past that, or past its own
    // maximum, made an invalid VkSamplerCreateInfo (maxLod below minLod), which drivers may take any way
    ci.minLod = p->minf <= 1 ? 0.0f : p->lod_min; ci.maxLod = p->minf <= 1 ? 0.25f : p->lod_max; if (ci.maxLod < ci.minLod) ci.maxLod = ci.minLod;
    int border = ci.addressModeU == VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER || ci.addressModeV == VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER || ci.addressModeW == VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
    ci.borderColor = VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK;
    if (border) {
        const float *b = p->border;
        if (b[0] == 0 && b[1] == 0 && b[2] == 0 && b[3] == 1) ci.borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_BLACK;
        else if (b[0] == 1 && b[1] == 1 && b[2] == 1 && b[3] == 1) ci.borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE;
        else if ((b[0] || b[1] || b[2] || b[3]) && V.has_custom_border) {
            ci.borderColor = VK_BORDER_COLOR_FLOAT_CUSTOM_EXT; memcpy(cb.customBorderColor.float32, b, 16); cb.format = VK_FORMAT_UNDEFINED;
            cb.pNext = (void *)ci.pNext; ci.pNext = &cb;
        }
    }
    VkSampler s; if (vkCreateSampler(V.dev, &ci, NULL, &s)) return VK_NULL_HANDLE;
    if (ci.magFilter == VK_FILTER_LINEAR || ci.minFilter == VK_FILTER_LINEAR) lin_note(s);   // (vk_samp_linear)
    // depth-compare samplers (shadow maps): each new one named, and the linear ones kept for gfx_vk_exec.c shadow_check (Mario Kart World's kart shadow looked blocky, run 13)
    if (ci.compareEnable) { int lin = ci.magFilter == VK_FILTER_LINEAR || ci.minFilter == VK_FILTER_LINEAR; if (lin && g_ncmp_lin < 64) g_cmp_lin[g_ncmp_lin++] = s;
        static int n; if (n++ < 24) VKLOG("depth-compare sampler: filters min %s mag %s mip %s (%s), anisotropy %g, compare op %d, wrap %d %d %d, border %s (%g %g %g %g), lod %g..%g bias %g",
            ci.minFilter == VK_FILTER_LINEAR ? "linear" : "nearest", ci.magFilter == VK_FILTER_LINEAR ? "linear" : "nearest", ci.mipmapMode == VK_SAMPLER_MIPMAP_MODE_LINEAR ? "linear" : "nearest",
            lin ? "2x2 PCF where the format filters linearly" : "point-sampled compare: a shadow edge in texel steps", ci.anisotropyEnable ? ci.maxAnisotropy : 1.0f, ci.compareOp, p->ws, p->wt, p->wr,
            border ? (ci.borderColor == VK_BORDER_COLOR_FLOAT_CUSTOM_EXT ? "custom" : ci.borderColor == VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE ? "white" : ci.borderColor == VK_BORDER_COLOR_FLOAT_OPAQUE_BLACK ? "black" : "transparent black") : "none",
            p->border[0], p->border[1], p->border[2], p->border[3], ci.minLod, ci.maxLod, ci.mipLodBias); }
    return s;
}
int vk_samp_cmp_linear(VkSampler s) { for (int i = 0; i < g_ncmp_lin; i++) if (g_cmp_lin[i] == s) return 1; return 0; }   // a depth-compare sampler with linear filtering (make_sampler)
volatile int g_vk_samp_log;   // DM_VK_SAMP_LOG=<n> (live, gfx_vk_exec.c x_read): the next n sampler registrations, in time order with the replay's lines
void mtl_reg_samp(int sid, const samp_par *sp) {
    if (sid < 0) return;
    if (g_vk_samp_log > 0) { extern uint64_t g_tex_frame; g_vk_samp_log--; LOG("[vk] sampler registered: id %d lod %g..%g min %d (present %llu, thread %d)", sid, sp->lod_min, sp->lod_max, sp->minf, (unsigned long long)g_tex_frame, (int)gettid()); }
    pthread_mutex_lock(&g_rmu);
    if (sid >= g_nsreg) { int n = sid + 256; g_sreg = realloc(g_sreg, n * sizeof *g_sreg); g_sreg_ok = realloc(g_sreg_ok, n); g_samp = realloc(g_samp, n * sizeof *g_samp);
        g_sseq = realloc(g_sseq, n * sizeof *g_sseq); g_sold = realloc(g_sold, (size_t)n * SREG_OLD * sizeof *g_sold);
        memset(g_sreg_ok + g_nsreg, 0, n - g_nsreg); memset(g_samp + g_nsreg, 0, (n - g_nsreg) * sizeof *g_samp); memset(g_sseq + g_nsreg, 0, (n - g_nsreg) * sizeof *g_sseq);
        memset(g_sold + (size_t)g_nsreg * SREG_OLD, 0, (size_t)(n - g_nsreg) * SREG_OLD * sizeof *g_sold); g_nsreg = n; }
    if (!g_sreg_ok[sid] || memcmp(&g_sreg[sid], sp, sizeof *sp)) {
        sreg_ver *o = g_sold + (size_t)sid * SREG_OLD;
        if (g_sreg_ok[sid]) { memmove(o + 1, o, sizeof *o * (SREG_OLD - 1)); o[0] = (sreg_ver){g_sreg[sid], g_sseq[sid]}; }
        g_sreg[sid] = *sp; g_sreg_ok[sid] = 1; g_samp[sid] = VK_NULL_HANDLE; g_sseq[sid] = __atomic_add_fetch(&g_sregs, 1, __ATOMIC_RELEASE); __atomic_fetch_add(&g_samp_gen, 1, __ATOMIC_RELEASE); }   // samplers are cached by parameters below
    pthread_mutex_unlock(&g_rmu);
}
static nvn_hmap g_samp_cache = {.klen = sizeof(samp_par)};   // samp_par → VkSampler (VkSamplers are never destroyed: few distinct parameter sets)
static const samp_par *samp_at(int sid, uint64_t seq);
void vk_samp_describe_at(int sid, uint64_t seq, char *out, size_t n) {   // the registered parameters a bind made after seq registrations sees (traces; 0: the newest)
    pthread_mutex_lock(&g_rmu);
    if (sid > 0 && sid < g_nsreg && g_sreg_ok[sid]) { const samp_par *p = samp_at(sid, seq); snprintf(out, n, "sampler %d: wrap %d %d %d min %d mag %d lod %g..%g bias %g aniso %g cmp %d/%d red %d border %g %g %g %g", sid, p->ws, p->wt, p->wr,
        p->minf, p->magf, p->lod_min, p->lod_max, p->lod_bias, p->aniso, p->cmp_mode, p->cmp_func, p->reduction, p->border[0], p->border[1], p->border[2], p->border[3]); }
    else snprintf(out, n, "sampler %d: not registered (dummy)", sid);
    pthread_mutex_unlock(&g_rmu);
}
void vk_samp_describe(int sid, char *out, size_t n) { vk_samp_describe_at(sid, 0, out, n); }
static const samp_par *samp_at(int sid, uint64_t seq) {   // under g_rmu: the version a bind made after seq registrations sees; NULL when not registered
    if (sid <= 0 || sid >= g_nsreg || !g_sreg_ok[sid]) return NULL;
    if (!seq || g_sseq[sid] <= seq) return &g_sreg[sid];
    const sreg_ver *o = g_sold + (size_t)sid * SREG_OLD;
    for (int k = 0; k < SREG_OLD; k++) if (o[k].seq && o[k].seq <= seq) return &o[k].p;
    return &g_sreg[sid];   // older than what is kept: the newest
}
int vk_samp_par_at(int sid, uint64_t seq, samp_par *out) {   // the parameters a bind made after seq registrations sees (gfx_vk_vt.h): 0 when not registered
    pthread_mutex_lock(&g_rmu); const samp_par *q = samp_at(sid, seq); if (q) *out = *q; pthread_mutex_unlock(&g_rmu); return q != NULL;
}
VkSampler vk_samp_at(int sid, uint64_t seq, int nearest) {   // the sampler a bind made after seq registrations sees (gfx_vk_exec.c samp_for)
    pthread_mutex_lock(&g_rmu);
    VkSampler s = VK_NULL_HANDLE; const samp_par *q = samp_at(sid, seq);
    if (q) { samp_par p = *q;
        if (nearest) { p.magf = 0; p.minf = p.minf == 1 ? 0 : p.minf == 3 || p.minf == 4 || p.minf == 5 ? 2 : p.minf; p.aniso = 1; p.reduction = 0; }
        if (!(s = nvn_hm_get(&g_samp_cache, &p)) && (s = make_sampler(&p))) nvn_hm_put(&g_samp_cache, &p, s); }
    pthread_mutex_unlock(&g_rmu);
    return s;
}
// A bind's sampler (as vk_samp_at gives it, seq 0 the newest; nearest its unfiltered twin) with its LOD clamps lowered by b levels, for a view whose level 0 is its texture's level b
// (an image made from a coarser level: a dense image from its base, dense_*, or a sparse one past the sparse address space; vkimg_t lv0). Vulkan clamps the LOD in the view's levels and NVN
// in the texture's, so a clamp of m read m levels too coarse there (PARITY G1: Mario Kart World's 8192x8192x68 terrain arrays from their level 3 or 4). The shader's side (explicit LODs,
// fetch levels, size queries in the texture's levels) is gfx_vk_shader.c's LOD-shift variant; gfx_vk_exec.c tex_desc_for keeps each binding's shift (vk_lod_shift)
VkSampler vk_samp_shift(int sid, uint64_t seq, int nearest, uint32_t b) {
    pthread_mutex_lock(&g_rmu);
    VkSampler s = VK_NULL_HANDLE; const samp_par *q = samp_at(sid, seq);
    if (q) { samp_par p = *q;
        if (nearest) { p.magf = 0; p.minf = p.minf == 1 ? 0 : p.minf == 3 || p.minf == 4 || p.minf == 5 ? 2 : p.minf; p.aniso = 1; p.reduction = 0; }
        p.lod_min = p.lod_min > (float)b ? p.lod_min - (float)b : 0.0f; p.lod_max = p.lod_max > (float)b ? p.lod_max - (float)b : 0.0f;
        if (!(s = nvn_hm_get(&g_samp_cache, &p)) && (s = make_sampler(&p))) nvn_hm_put(&g_samp_cache, &p, s); }
    pthread_mutex_unlock(&g_rmu);
    return s;
}
VkSampler vk_samp_by_id(int sid) {
    pthread_mutex_lock(&g_rmu);
    VkSampler s = VK_NULL_HANDLE;
    if (sid > 0 && sid < g_nsreg && g_sreg_ok[sid]) {
        s = g_samp[sid];
        if (!s) { s = nvn_hm_get(&g_samp_cache, &g_sreg[sid]); if (!s) { s = make_sampler(&g_sreg[sid]); if (s) nvn_hm_put(&g_samp_cache, &g_sreg[sid], s); } g_samp[sid] = s; }
    }
    pthread_mutex_unlock(&g_rmu);
    return s;
}
// The same sampler without filtering, for integer views: Maxwell reads integer texels unfiltered whatever the sampler says, while Vulkan forbids linear filtering of
// formats without SAMPLED_IMAGE_FILTER_LINEAR (Pokémon Legends: Arceus samples R8_UINT and R32_UINT views with trilinear samplers; the validation layer named them)
VkSampler vk_samp_by_id_nearest(int sid) {   // cached by its parameters, as the samplers themselves
    pthread_mutex_lock(&g_rmu);
    VkSampler s = VK_NULL_HANDLE;
    if (sid > 0 && sid < g_nsreg && g_sreg_ok[sid]) {
        samp_par p = g_sreg[sid]; p.magf = 0; p.minf = p.minf == 1 ? 0 : p.minf == 3 || p.minf == 4 || p.minf == 5 ? 2 : p.minf; p.aniso = 1; p.reduction = 0;
        if (!(s = nvn_hm_get(&g_samp_cache, &p)) && (s = make_sampler(&p))) nvn_hm_put(&g_samp_cache, &p, s);
    }
    pthread_mutex_unlock(&g_rmu);
    return s;
}
VkSampler vk_dummy_sampler(void) {
    static VkSampler s; if (s) return s;
    samp_par p = {0}; p.magf = 1; p.minf = 1; p.lod_max = 1000; p.aniso = 1; s = make_sampler(&p); return s;
}
// ---------------- dummies: what an unbound slot reads (zeros) ----------------
VkImageView vk_dummy_view(int dim, int arrayed, int ms, int storage, int is_int) {
    static VkImageView cache[4][2][2][2][2]; static vkimg_t *imgs[4][2][2][2]; static pthread_mutex_t mu = PTHREAD_MUTEX_INITIALIZER;
    int d = dim == 0 ? 0 : dim == 2 ? 2 : dim == 3 ? 3 : 1; arrayed = !!arrayed; ms = !!ms && d == 1; storage = !!storage; is_int = !!is_int;
    pthread_mutex_lock(&mu);
    VkImageView v = cache[d][arrayed][ms][storage][is_int];
    if (!v) {
        vkimg_t **pi = &imgs[d][ms][storage][is_int];
        if (!*pi) {
            tex_t t; memset(&t, 0, sizeof t); t.magic = MAGIC('T', 'E', 'X', 0); t.serial = 1;
            t.d.format = is_int ? 0x27 : 0x25; t.d.width = t.d.height = 1; t.d.depth = d == 3 ? 12 : 6; t.d.levels = 1; t.d.samples = ms ? 4 : 0;
            t.d.target = d == 0 ? 3 : d == 2 ? 2 : d == 3 ? 9 : ms ? 6 : 4; t.d.flags = storage ? NVN_TEXTURE_FLAGS_IMAGE : 0;
            if (d == 2) t.d.depth = 1;
            *pi = img_new(&t, vk_fmt(t.d.format));
            VkCommandBuffer cb = vk_upload_cmd(); vk_tex_prepare(cb, *pi);
            VkClearColorValue z = {{0}}; VkImageSubresourceRange rg = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, VK_REMAINING_ARRAY_LAYERS};
            if (!ms) vkCmdClearColorImage(cb, (*pi)->img, VK_IMAGE_LAYOUT_GENERAL, &z, 1, &rg);
        }
        vkimg_t *im = *pi;
        VkImageViewType type = d == 0 ? (arrayed ? VK_IMAGE_VIEW_TYPE_1D_ARRAY : VK_IMAGE_VIEW_TYPE_1D) : d == 2 ? VK_IMAGE_VIEW_TYPE_3D : d == 3 ? (arrayed ? VK_IMAGE_VIEW_TYPE_CUBE_ARRAY : VK_IMAGE_VIEW_TYPE_CUBE) : (arrayed ? VK_IMAGE_VIEW_TYPE_2D_ARRAY : VK_IMAGE_VIEW_TYPE_2D);
        uint32_t nly = type == VK_IMAGE_VIEW_TYPE_CUBE ? 6 : type == VK_IMAGE_VIEW_TYPE_CUBE_ARRAY ? 12 : arrayed ? im->layers : 1;
        VkImageViewCreateInfo ci = {VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, NULL, 0, im->img, type, im->fmt, {0}, {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, nly}};
        VKCHK(vkCreateImageView(V.dev, &ci, NULL, &v)); cache[d][arrayed][ms][storage][is_int] = v;
    }
    pthread_mutex_unlock(&mu);
    return v;
}
VkBufferView vk_dummy_tbuf(int storage) {
    static VkBufferView v[2]; static VkBuffer b; static VkDeviceMemory m; storage = !!storage;
    if (v[storage]) return v[storage];
    if (!b) {
        VkBufferCreateInfo bi = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, NULL, 0, 65536, VK_BUFFER_USAGE_UNIFORM_TEXEL_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_TEXEL_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_SHARING_MODE_EXCLUSIVE};
        VKCHK(vkCreateBuffer(V.dev, &bi, NULL, &b)); VkMemoryRequirements req; vkGetBufferMemoryRequirements(V.dev, b, &req);
        VkMemoryAllocateInfo ai = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, NULL, req.size, vk_mem_type(req.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, 0)};
        VKCHK(vkAllocateMemory(V.dev, &ai, NULL, &m)); VKCHK(vkBindBufferMemory(V.dev, b, m, 0));
        void *p; VKCHK(vkMapMemory(V.dev, m, 0, VK_WHOLE_SIZE, 0, &p)); memset(p, 0, 65536);
    }
    VkBufferViewCreateInfo ci = {VK_STRUCTURE_TYPE_BUFFER_VIEW_CREATE_INFO, NULL, 0, b, VK_FORMAT_R32_UINT, 0, 65536};
    VKCHK(vkCreateBufferView(V.dev, &ci, NULL, &v[storage]));
    return v[storage];
}
// a buffer texture's texel buffer view over its pool (cached by texture contents: the registered copy)
VkBufferView vk_tbuf_view(const tex_t *t) {
    static nvn_hmap cache = {.klen = 24}; static pthread_mutex_t mu = PTHREAD_MUTEX_INITIALIZER;
    // keyed by the pool too: a pool finalized and another made at the same address has a new VkBuffer, and the old one's memory is freed once the GPU is past it. Keyed by address
    // alone, the view over the old buffer was handed out again and shaders read and wrote freed memory, which the kernel gives to newer buffers (Xenoblade Chronicles 3: the
    // allocator kept inside one of its memory pools had its tree bins overwritten during the opening movie, and malloc crashed at the next scene)
    vkpool_t *pl = t->mem ? vk_pool_gpu((uint64_t)(uintptr_t)t->mem, 1) : NULL;   // (the placed part's serial: its buffer)
    struct { uint64_t mem, pool; uint32_t fmt, n; } k = {(uint64_t)(uintptr_t)t->mem, pl ? pl->serial : 0, (uint32_t)t->d.format, t->d.width};
    pthread_mutex_lock(&mu);
    VkBufferView v = nvn_hm_get(&cache, &k);
    if (!v) {
        const vkfmt_t *f = vk_fmt(t->d.format); VkBuffer b; VkDeviceSize off;
        if (f && t->mem && vk_buf_at((uint64_t)(uintptr_t)t->mem, (uint64_t)t->d.width * f->bpb, &b, &off) && off % V.props.limits.minTexelBufferOffsetAlignment == 0) {
            VkBufferViewCreateInfo ci = {VK_STRUCTURE_TYPE_BUFFER_VIEW_CREATE_INFO, NULL, 0, b, f->vk, off, (VkDeviceSize)t->d.width * f->bpb};
            if (vkCreateBufferView(V.dev, &ci, NULL, &v)) v = VK_NULL_HANDLE; else nvn_hm_put(&cache, &k, v);
        } else VKRED_ONCE(16, "buffer texture not in a pool or misaligned (format %#x, %u elements)", t->d.format, t->d.width);
    }
    pthread_mutex_unlock(&mu);
    return v;
}
// CPU readback of texels (ReadTexels): the image copied to scratch memory by a submission of its own, waited for
void mtl_read_texels(const tex_t *t, unsigned level, unsigned layer0, const NVNcopyRegion *r, uint8_t *dst, size_t row, size_t img) {
    void mtl_sub_begin(void); uint64_t mtl_sub_commit(void); int mtl_wait(uint64_t, uint64_t);
    mtl_sub_begin();
    vkimg_t *im = vk_tex(t); const vkfmt_t *f = vk_fmt(t->d.format); vk_dsc_read(im, "read back (ReadTexels)");
    size_t depth = r->depth ? r->depth : 1;
    if (!im || !f || level < im->lv0) { mtl_sub_commit(); for (size_t z = 0; z < depth; z++) for (int y = 0; y < r->height; y++) memset(dst + z * img + (size_t)y * row, 0, row); return; }   // (lv0: a level its image lacks reads 0)
    level -= im->lv0;
    if (f->dec) { VKRED_ONCE(17, "ReadTexels of a BC texture decoded on the CPU: zeros returned"); mtl_sub_commit(); for (size_t z = 0; z < depth; z++) for (int y = 0; y < (r->height + 3) / 4; y++) memset(dst + z * img + (size_t)y * row, 0, row); return; }
    unsigned bw = f->bw ? f->bw : 1, bh = f->bh ? f->bh : 1;
    size_t pitch = (size_t)((r->width + bw - 1) / bw) * f->bpb, rows = (r->height + bh - 1) / bh, wp = im->promo ? pitch * 2 : pitch;   // (wp: an image in its wider twin's texels)
    VkBuffer sb; VkDeviceSize so; uint8_t *p = vk_scratch(wp * rows * depth, 16, &sb, &so), *nar = im->promo ? malloc(pitch) : NULL;
    VkCommandBuffer cb = vk_cmd(); vk_tex_prepare(cb, im); vk_img_barrier_all(cb);
    int is3d = im->d > 1;
    VkBufferImageCopy c = {so, 0, 0, {im->aspect & VK_IMAGE_ASPECT_COLOR_BIT ? VK_IMAGE_ASPECT_COLOR_BIT : VK_IMAGE_ASPECT_DEPTH_BIT, level, is3d ? 0 : layer0 + (uint32_t)r->zoffset, is3d ? 1 : (uint32_t)depth},
                           {r->xoffset, r->yoffset, is3d ? r->zoffset : 0}, {(uint32_t)r->width, (uint32_t)r->height, is3d ? (uint32_t)depth : 1}};
    vkCmdCopyImageToBuffer(cb, im->img, VK_IMAGE_LAYOUT_GENERAL, sb, 1, &c);
    mtl_wait(mtl_sub_commit(), ~0ull);
    for (size_t z = 0; z < depth; z++) for (size_t y = 0; y < rows; y++) { const uint8_t *q = p + (z * rows + y) * wp;
        if (im->promo) { if (!nar) break; promo_narrow(&PROMO[im->promo - 1], q, (size_t)r->width, nar); q = nar; }   // (narrowed as drawing into the texture's format would)
        memcpy(dst + z * img + y * row, q, pitch < row ? pitch : row); }
    free(nar);
}
