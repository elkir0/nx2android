// Android port, Vulkan backend: the present's upscaling (gfx_vk_fsr.c, shaders/fsr/fsr1.glsl): the game's crop of its window texture drawn into the swapchain image,
// FSR 1 (EASU, then RCAS) or a bilinear scale, black around the picture, the gamepad overlay drawn after it in the same pass (gfx_vk_present.c).
// The first part is plain arithmetic, included alone by tests/test_vk_fsr.py; the Vulkan part follows when gfx_vk.h was included first.
#pragma once
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

// ---- the picture's geometry ----
// The picture of a cw x ch crop in a W x H window, the aspect kept, centred (black borders: on the Galaxy Z Fold 8's inner 2504x2256 screen 1280x720 is 2504x1408 at
// y 424, on its outer 2520x1080 one 1920x1080 at x 300). In integers: the blit's double arithmetic (s = min(W / cw, H / ch), (int)(cw * s)) can land a hair under a whole
// pixel and lose it, a black column or row along the picture's edge: 3 of 273 pairs of common crops and phone windows (1440x810 in 2960x1848 came out 2959x1664 where
// 2960x1665 is exact, 1600x900 in 1856x2160 1855 wide; tests/test_vk_fsr.py). The Fold's two screens with a 1280x720 crop were not among them
typedef struct { int x, y, w, h; } vkfsr_rect;
static inline vkfsr_rect vkfsr_fit(int cw, int ch, int W, int H) {
    vkfsr_rect r = {0, 0, W, H};
    if (cw <= 0 || ch <= 0 || W <= 0 || H <= 0) return r;
    if ((int64_t)W * ch <= (int64_t)H * cw) r.h = (int)((int64_t)ch * W / cw); else r.w = (int)((int64_t)cw * H / ch);   // width-bound, else height-bound
    if (r.w < 1) r.w = 1;
    if (r.h < 1) r.h = 1;
    r.x = (W - r.w) / 2; r.y = (H - r.h) / 2;
    return r;
}
// The swapchain's pre-rotation (VkSurfaceTransformFlagBitsKHR ROTATE_*): the image holds the window's picture turned 90 degrees clockwise (VKFSR_ROT90: the window's top
// row is the image's right column), 180 or 270. A target pixel (px, py) of the swapchain image is window pixel ROT0 (px, py), ROT90 (py, H-1-px), ROT180 (W-1-px, H-1-py),
// ROT270 (W-1-py, px): as Android's pre-rotation guide turns clip space (x, y) into (-y, x) for ROTATE_90. The image is W x H, or H x W for 90 and 270
enum { VKFSR_ROT0, VKFSR_ROT90, VKFSR_ROT180, VKFSR_ROT270 };
static inline void vkfsr_target_size(int rot, int W, int H, int *pw, int *ph) { int t = rot == VKFSR_ROT90 || rot == VKFSR_ROT270; *pw = t ? H : W; *ph = t ? W : H; }
// The shaders' push constants (shaders/fsr/fsr1.glsl FsrPush): xa, xb a target pixel t as a picture pixel l = (xa0 t.x + xa1 t.y, xa2 t.x + xa3 t.y) + (xb0, xb1), inside
// when 0 <= l < (xb2, xb3); map l to the input position (texel centres at integers); clip the input texels a pass may read (first, last); k a pass's own (RCAS: k0 its
// sharpness, 2^-stops; bilinear: k0 k1 1 / the input texture's size)
typedef struct { int32_t xa[4], xb[4]; float map[4]; int32_t clip[4]; float k[4]; } vkfsr_push;
static inline void vkfsr_xform(int rot, int W, int H, vkfsr_rect pic, vkfsr_push *p) {
    static const int32_t A[4][4] = {{1, 0, 0, 1}, {0, 1, -1, 0}, {-1, 0, 0, -1}, {0, -1, 1, 0}};
    const int32_t bx[4] = {0, 0, W - 1, W - 1}, by[4] = {0, H - 1, H - 1, 0};
    rot &= 3; memcpy(p->xa, A[rot], sizeof p->xa);
    p->xb[0] = bx[rot] - pic.x; p->xb[1] = by[rot] - pic.y; p->xb[2] = pic.w; p->xb[3] = pic.h;
}
static inline void vkfsr_xform_id(int w, int h, vkfsr_push *p) {   // an intermediate image: the picture is the whole target, unturned
    vkfsr_rect r = {0, 0, w, h}; vkfsr_xform(VKFSR_ROT0, w, h, r, p);
}
// The picture pixel l's input position, as vkCmdBlitImage maps a destination pixel's centre: (l + 0.5) * crop / picture - 0.5 from the crop's origin. A window image stored
// bottom-up (lower-left origin: gfx_vk_exec.c vk_window_bottom_up) is read from the crop's bottom row up, as the blit's swapped source offsets did. crop: x, y, w, h
static inline void vkfsr_map(const int crop[4], vkfsr_rect pic, int bottom_up, vkfsr_push *p) {
    float sx = (float)crop[2] / (float)pic.w, sy = (float)crop[3] / (float)pic.h;
    p->map[0] = sx; p->map[2] = 0.5f * sx - 0.5f + (float)crop[0];
    if (!bottom_up) { p->map[1] = sy; p->map[3] = 0.5f * sy - 0.5f + (float)crop[1]; }
    else { p->map[1] = -sy; p->map[3] = (float)(crop[1] + crop[3]) - 0.5f * sy - 0.5f; }
    p->clip[0] = crop[0]; p->clip[1] = crop[1]; p->clip[2] = crop[0] + crop[2] - 1; p->clip[3] = crop[1] + crop[3] - 1;
}
// The crop a present shows, inside the w x h image (a crop past its edge is cut to it; none, or nothing left: the whole image)
static inline void vkfsr_crop(const int *crop, int w, int h, int out[4]) {
    int x = crop ? crop[0] : 0, y = crop ? crop[1] : 0, cw = crop && crop[2] ? crop[2] : w, ch = crop && crop[3] ? crop[3] : h;
    if (x < 0) { cw += x; x = 0; }
    if (y < 0) { ch += y; y = 0; }
    if (x + cw > w) cw = w - x;
    if (y + ch > h) ch = h - y;
    if (cw <= 0 || ch <= 0) { x = y = 0; cw = w; ch = h; }
    out[0] = x; out[1] = y; out[2] = cw; out[3] = ch;
}

// ---- switches ----
// DM_VK_UPSCALE (live): fsr (the default), bilinear, off. fsr: EASU into an intermediate image of the picture's size, then RCAS into the swapchain image. bilinear: one
// bilinear draw into the swapchain image (the blit's picture without its crop-edge bleed and its full-screen clear). off: the blit as before (a clear of the whole image,
// vkCmdBlitImage, the overlay in a pass of its own): the reference for an A/B, and the path of a device without dynamic rendering. A picture that is not made larger
// than its crop in both directions is drawn bilinear whatever the switch (EASU is an upscaler).
// Why fsr by default: Mario Kart World shows its 1280x720 crop (the game's own upscaler's output, its HUD drawn at 720p over it) 1.96x larger on the inner screen; a bilinear
// stretch of that soft picture is what the player called soft and aliased. FSR 1 is AMD's answer to exactly this case (a 1.5-2x upscale of an anti-aliased, tone-mapped
// frame), and its cost is a fraction of a frame that already takes 35-38 ms of GPU: about 0.6 ms at 2504x1408 by its instruction count (gfx_vk_fsr.c "Cost"), measured
// on the device by the present's own timer (DM_VK_FSR_TIME). off brings the old path back as it was.
// DM_VK_RCAS (live): auto (the default), off, or the sharpness in stops, 0 (the most) to 2 (FSR's scale: each stop halves it). auto: 0.25 with fsr (the FidelityFX FSR 1
// sample's own default, rcasAttenuation 0.25), none with bilinear. A number with bilinear sharpens the bilinear picture (bilinear into the intermediate, then RCAS)
enum { VKFSR_OFF, VKFSR_BILINEAR, VKFSR_FSR };
static inline int vkfsr_parse_mode(const char *e) {   // -1: not a value (the caller keeps the default and says so)
    if (!e || !*e || !strcmp(e, "fsr") || !strcmp(e, "1") || !strcmp(e, "easu")) return VKFSR_FSR;
    if (!strcmp(e, "bilinear") || !strcmp(e, "linear")) return VKFSR_BILINEAR;
    if (!strcmp(e, "off") || !strcmp(e, "0") || !strcmp(e, "blit")) return VKFSR_OFF;
    return -1;
}
#define VKFSR_RCAS_AUTO 0.25f
static inline float vkfsr_parse_rcas(const char *e, int mode) {   // stops, < 0: no RCAS
    if (!e || !*e || !strcmp(e, "auto")) return mode == VKFSR_FSR ? VKFSR_RCAS_AUTO : -1.0f;
    if (!strcmp(e, "off") || !strcmp(e, "none") || !strcmp(e, "-1")) return -1.0f;
    char *end; float v = strtof(e, &end);
    if (end == e || *end) return mode == VKFSR_FSR ? VKFSR_RCAS_AUTO : -1.0f;
    return v < 0 ? 0 : v > 2 ? 2 : v;
}
static inline float vkfsr_rcas_k(float stops) { return exp2f(-stops); }   // FsrRcasCon: stops to the linear factor

// ---- the swap of a game's own upscaler (agl's NDSR network: gfx_vk_fsr.c "NDSR swap", DM_VK_NDSR) ----
// What a variant of the network reads and writes, from its SPIR-V (the translator keeps the sRGB curves' constants of its code as OpConstants): upstream's rule (nvn_mtl.m
// up_mfx_color, over the Ampere code), counted here as the instructions that use each constant, 8 at least. Linear input encoded by the shader itself (0.0031308 and 1/2.4 in
// 32 bits), then: no decode, the output stays encoded (VKNDSR_ENC); a decode (0.04045, 2.4) clamped to 0-1 (VKNDSR_LIN1) or capped at 128 (VKNDSR_LIN128: an FMin / NMin /
// clamp with 128). Input encoded already, decoded at the end (0.04045, 2.4 and 1/12.92 as halves: Splatoon Raiders' pack): VKNDSR_SRGB_LIN1. Anything else: VKNDSR_KEEP, the
// game's shader runs. Mario Kart World 1.8.0's 30 compute variants (agl_ndsr.pck): 10 ENC, 10 LIN1, 10 LIN128 (tests/test_vk_fsr.py, with the bundle at hand)
enum { VKNDSR_KEEP, VKNDSR_ENC, VKNDSR_LIN1, VKNDSR_LIN128, VKNDSR_SRGB_LIN1 };
static inline float vkfsr_h2f(uint32_t h) {   // an IEEE half's value
    uint32_t s = h >> 15 & 1, e = h >> 10 & 31, m = h & 1023; float v = e ? ldexpf((float)(m | 1024), (int)e - 25) : ldexpf((float)m, -24);
    return e == 31 ? INFINITY : s ? -v : v;
}
static inline int vkndsr_shape(const uint32_t *w, uint32_t n) {
    enum { MAXC = 4096 }; uint32_t ids[MAXC]; uint8_t what[MAXC]; uint32_t nc = 0, ft[64][2], nft = 0, glsl = 0, uses[16] = {0}; int cap128 = 0;
    // what: 1-4 f32 0.0031308, 1/2.4, 0.04045, 2.4; 5-9 f16 0.04045, 2.4, 1/12.92, 1/2.4 (two roundings: 9 and 10); 15 the number 128
    for (uint32_t i = 5; i < n;) {
        uint32_t len = w[i] >> 16, op = w[i] & 0xffff; if (!len || i + len > n) break;
        const uint32_t *a = w + i + 1;
        if (op == 11 && len >= 3 && !memcmp(a + 1, "GLSL.std.450", 12)) glsl = a[0];   // OpExtInstImport
        else if (op == 22 && len >= 3 && nft < 64) { ft[nft][0] = a[0]; ft[nft++][1] = a[1]; }   // OpTypeFloat
        else if (op == 43 && len >= 4 && nc < MAXC) {   // OpConstant: a float of 32 or 16 bits
            uint32_t bits = 0; for (uint32_t k = 0; k < nft; k++) if (ft[k][0] == a[0]) bits = ft[k][1];
            float v = 0; if (bits == 32) memcpy(&v, &a[2], 4); else if (bits == 16) v = vkfsr_h2f(a[2] & 0xffff); else { i += len; continue; }
            uint8_t k = 0;
            if (bits == 32) k = fabsf(v - 0.0031308f) < 1e-7f ? 1 : fabsf(v - 1.0f / 2.4f) < 4e-6f ? 2 : fabsf(v - 0.04045f) < 1e-6f ? 3 : fabsf(v - 2.4f) < 4e-6f ? 4 : 0;
            else k = v == 0.040435791015625f ? 5 : v == 2.3984375f ? 6 : v == 0.07745361328125f ? 7 : v == 0.41650390625f ? 9 : v == 0.416748046875f ? 10 : 0;
            if (!k && v == 128.0f) k = 15;
            if (k) { ids[nc] = a[1]; what[nc++] = k; }
        } else if (op == 44 && len >= 4 && nc < MAXC) {   // OpConstantComposite: a vector of one of them (glslang's vec3(0.0031308)) stands for it, its uses for its uses
            for (uint32_t o = 2; o < len - 1; o++) for (uint32_t c = 0; c < nc; c++) if (ids[c] == a[o]) { ids[nc] = a[1]; what[nc] = what[c]; nc++; o = len; break; }
        } else if (op == 12 && len >= 5 && glsl && a[2] == glsl) {   // OpExtInst: a min or clamp with 128
            uint32_t inst = a[3];
            if (inst == 37 || inst == 79 || inst == 43 || inst == 81) for (uint32_t o = 4; o < len - 1; o++) for (uint32_t c = 0; c < nc; c++) if (what[c] == 15 && ids[c] == a[o]) cap128 = 1;
            for (uint32_t o = 4; o < len - 1; o++) for (uint32_t c = 0; c < nc; c++) if (ids[c] == a[o]) uses[what[c]]++;
        } else if (op != 43 && op != 5 && op != 6 && op != 71 && op != 72) {   // any other instruction (names and decorations carry literals): its operands that are those constants
            for (uint32_t o = 0; o < len - 1; o++) for (uint32_t c = 0; c < nc; c++) if (ids[c] == a[o]) uses[what[c]]++;
        }
        i += len;
    }
    int enc32 = uses[1] >= 8 && uses[2] >= 8, dec32 = uses[3] >= 8 && uses[4] >= 8, enc16 = uses[9] >= 8 || uses[10] >= 8, dec16 = uses[5] >= 8 && uses[6] >= 8 && uses[7] >= 8;
    if (enc32 && !enc16 && !dec16) return dec32 ? (cap128 ? VKNDSR_LIN128 : VKNDSR_LIN1) : VKNDSR_ENC;
    if (dec16 && !enc16 && !enc32 && !dec32) return VKNDSR_SRGB_LIN1;
    return VKNDSR_KEEP;
}
// The network's two rectangles (its push constants, bank 24, words 0-7): the input's origin x, y and its last texel x, y relative to that origin (inclusive), then the
// output's; the Mario Kart World variants read exactly those 8 words as integers. Both inside their textures (iw x ih, ow x oh at the bound levels): 1, the rectangles as x, y, w, h
static inline int vkndsr_rects(const uint32_t v[8], uint32_t iw, uint32_t ih, uint32_t ow, uint32_t oh, int in[4], int out[4]) {
    for (int r = 0; r < 2; r++) {
        uint64_t x = v[4 * r], y = v[4 * r + 1], w = (uint64_t)v[4 * r + 2] + 1, h = (uint64_t)v[4 * r + 3] + 1, W = r ? ow : iw, H = r ? oh : ih;
        if (x + w > W || y + h > H) return 0;
        int *o = r ? out : in; o[0] = (int)x; o[1] = (int)y; o[2] = (int)w; o[3] = (int)h;
    }
    return 1;
}

#ifdef VK_VERSION_1_0
// ---- the passes (gfx_vk_fsr.c) ----
// vk_fsr_path: what a present of window image im will draw with (VKFSR_* after DM_VK_UPSCALE, the device and the image: VKFSR_OFF when the blit must do it). The swapchain
// for any other path is UNORM whatever the image's sRGB-ness: the passes read its bytes through a UNORM view (gamma-encoded, what FSR expects) and write them as they are.
int vk_fsr_path(const vkimg_t *im, const int crop[4], int W, int H);
// vk_fsr_begin: the passes of one present into command buffer cb (present slot `slot`, whose earlier use has completed; busy: the fences of the presents that may still be in
// flight, waited on only when an image they use must go). The window is W x H, the swapchain image (view, image, format) is it turned by rot. On 1 the target's rendering
// scope is open after the picture's pass, for the overlay (gfx_vk_present.c), and vk_fsr_end closes it, leaving the image in COLOR_ATTACHMENT_OPTIMAL. On 0 nothing was
// recorded and the blit draws the picture
int vk_fsr_begin(VkCommandBuffer cb, uint32_t slot, int path, const VkFence *busy, uint32_t nbusy, vkimg_t *im, const int crop[4], int bottom_up,
                 VkImage target, VkImageView view, VkFormat fmt, int W, int H, int rot);   // path: vk_fsr_path's answer for this present
void vk_fsr_end(VkCommandBuffer cb);
// The present's GPU time (DM_VK_FSR_TIME, start-up, on by default): timestamps around the present's own work in its command buffer, read back when its slot comes round
void vk_fsr_time_start(VkCommandBuffer cb, uint32_t slot);   // after the present's first barrier (before any of its passes or its blit)
void vk_fsr_time_end(VkCommandBuffer cb, uint32_t slot);     // after the picture and the overlay, before vkEndCommandBuffer
void vk_fsr_time_submitted(uint32_t slot, int path);         // the command buffer was submitted (its stamps will be written): path, what drew it
void vk_fsr_time_collect(uint32_t slot);                     // the slot's fence has signalled: its times go into the report (every 600 presents)
// The NDSR swap, called from the replay's dispatch (gfx_vk_exec.c dispatch): vk_ndsr_wanted, whether a compute shader is one of the game's upscaler shaders (DM_GAME_UPSCALER_KEYS)
// and DM_VK_NDSR is fsr; vk_ndsr_swap, the dispatch drawn by EASU into its output instead (1), or left to the shader (0: an indirect dispatch is the caller's to refuse).
// in, out: its texture and image bindings as the dispatch would bind them (the sampled and the storage view, each one's size at its level); push: bank 24's first 8 words
typedef struct { VkImageView view; uint32_t w, h; } vkfsr_tex;
int vk_ndsr_wanted(uint64_t key);
int vk_ndsr_swap(VkCommandBuffer cb, const vkshader_t *cs, const vkfsr_tex *in, const vkfsr_tex *out, const uint32_t push[8]);
#endif
