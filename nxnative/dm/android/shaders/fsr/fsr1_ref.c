// CPU reference of fsr1.glsl (the present's upscaling passes, gfx_vk_fsr.c): the same operations in the same order, in _Float16 when built with -DFSR_H=1 as the shaders'
// 16-bit variants compute (the direction's normalization and the kernel's shape in 32 bits in both, as the shaders do), and the passes' geometry from gfx_vk_fsr.h itself.
// It draws a whole present: the first pass into an intermediate image (A2B10G10R10 or RGBA8, rounded as the GPU stores it), the last into the target, black around the
// picture, turned as the swapchain is. tests/test_vk_fsr.py builds and checks it; the image comparisons and the GPU check (the shaders run through MoltenVK on a Mac)
// take it as the reference.
//   cc -O2 -std=gnu11 -ffp-contract=off [-DFSR_H=1] fsr1_ref.c -o fsr1_ref -lm
//   fsr1_ref <in.rgba> <w> <h> <crop x> <y> <w> <h> <window W> <H> <rot 0-3> <path: fsr|bilinear> <rcas stops, -1 none> <bottom_up 0|1> <intermediate bits 10|8> <out.rgba>
// in.rgba: w x h RGBA8 texels; out.rgba: the target, W x H (H x W turned 90 or 270), RGBA8. A path the present would not take is not refused (a picture that is not made
// larger still gets EASU with fsr): the reference draws what it is told.
// The NDSR swap (gfx_vk_fsr.c vk_ndsr_swap, FSR_CS): path ndsr0 to ndsr3 (the variant's shape: VKNDSR_ENC, LIN1, LIN128, SRGB_LIN1 less 1), in.rgba RGBA16F texels (w x h), the crop
// the input rectangle, W x H the output rectangle (at 0,0 of a W x H output), out.rgba RGBA16F; rot, rcas, bottom_up and bits are ignored
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../../gfx_vk_fsr.h"

#ifndef FSR_H
#define FSR_H 0
#endif
#if FSR_H
typedef _Float16 F1;
#else
typedef float F1;
#endif
typedef struct { F1 x, y; } F2;
typedef struct { F1 r, g, b; } F3;
#define FSR_RCAS_LIMIT (0.25 - (1.0 / 16.0))
#define FSR_EPS (1.0 / 16384.0)

// GLSL's built-ins as the shaders use them (no NaN reaches them)
static F1 h_max(F1 a, F1 b) { return a > b ? a : b; }
static F1 h_min(F1 a, F1 b) { return a < b ? a : b; }
static F1 h_abs(F1 a) { return a < (F1)0 ? -a : a; }
static F1 h_clamp(F1 x, F1 lo, F1 hi) { return h_min(h_max(x, lo), hi); }
static int i_clamp(int x, int lo, int hi) { return x < lo ? lo : x > hi ? hi : x; }

// an image the passes read: 4 channels a texel as the GPU returns them (an 8-bit or 10-bit UNORM texel's exact value), w x h
typedef struct { const float *px; int w, h; } img_t;
static F3 fetch(const img_t *s, int x, int y) { const float *p = s->px + 4 * ((size_t)y * s->w + x); F3 c = {(F1)p[0], (F1)p[1], (F1)p[2]}; return c; }
static F1 luma2(F3 c) { return c.b * (F1)0.5 + (c.r * (F1)0.5 + c.g); }

static void easu_set(F2 *dir, F1 *len, F1 w, F1 lA, F1 lB, F1 lC, F1 lD, F1 lE) {
    F1 dc = lD - lC, cb = lC - lB;
    F1 lenX = (F1)1.0 / h_max(h_max(h_abs(dc), h_abs(cb)), (F1)FSR_EPS);
    F1 dirX = lD - lB;
    dir->x += dirX * w;
    lenX = h_clamp(h_abs(dirX) * lenX, (F1)0.0, (F1)1.0);
    lenX *= lenX;
    *len += lenX * w;
    F1 ec = lE - lC, ca = lC - lA;
    F1 lenY = (F1)1.0 / h_max(h_max(h_abs(ec), h_abs(ca)), (F1)FSR_EPS);
    F1 dirY = lE - lA;
    dir->y += dirY * w;
    lenY = h_clamp(h_abs(dirY) * lenY, (F1)0.0, (F1)1.0);
    lenY *= lenY;
    *len += lenY * w;
}
static void easu_tap(F3 *aC, F1 *aW, F1 ox, F1 oy, F2 pp, F2 dir, F2 len, F1 lob, F1 clp, F3 c) {
    F2 off = {ox - pp.x, oy - pp.y};
    F2 v = {off.x * dir.x + off.y * dir.y, off.x * (-dir.y) + off.y * dir.x};
    v.x *= len.x; v.y *= len.y;
    F1 d2 = h_min(v.x * v.x + v.y * v.y, clp);
    F1 wB = (F1)(2.0 / 5.0) * d2 + (F1)(-1.0);
    F1 wA = lob * d2 + (F1)(-1.0);
    wB *= wB;
    wA *= wA;
    wB = (F1)(25.0 / 16.0) * wB + (F1)(-(25.0 / 16.0 - 1.0));
    F1 w = wB * wA;
    aC->r += c.r * w; aC->g += c.g * w; aC->b += c.b * w;
    *aW += w;
}
static void easu(const img_t *s, const int32_t clip[4], float px, float py, float out[3]) {
    float flx = floorf(px), fly = floorf(py);
    F2 pp = {(F1)(px - flx), (F1)(py - fly)};
    int qx = (int)flx, qy = (int)fly;
    int x0 = i_clamp(qx - 1, clip[0], clip[2]), x1 = i_clamp(qx, clip[0], clip[2]), x2 = i_clamp(qx + 1, clip[0], clip[2]), x3 = i_clamp(qx + 2, clip[0], clip[2]);
    int y0 = i_clamp(qy - 1, clip[1], clip[3]), y1 = i_clamp(qy, clip[1], clip[3]), y2 = i_clamp(qy + 1, clip[1], clip[3]), y3 = i_clamp(qy + 2, clip[1], clip[3]);
    F3 b = fetch(s, x1, y0), c = fetch(s, x2, y0);
    F3 e = fetch(s, x0, y1), f = fetch(s, x1, y1), g = fetch(s, x2, y1), h = fetch(s, x3, y1);
    F3 i = fetch(s, x0, y2), j = fetch(s, x1, y2), k = fetch(s, x2, y2), l = fetch(s, x3, y2);
    F3 n = fetch(s, x1, y3), o = fetch(s, x2, y3);
    F1 bL = luma2(b), cL = luma2(c), eL = luma2(e), fL = luma2(f), gL = luma2(g), hL = luma2(h);
    F1 iL = luma2(i), jL = luma2(j), kL = luma2(k), lL = luma2(l), nL = luma2(n), oL = luma2(o);
    F2 dir = {0, 0}; F1 len = 0;
    F1 ix = (F1)1.0 - pp.x, iy = (F1)1.0 - pp.y;
    easu_set(&dir, &len, ix * iy, bL, eL, fL, gL, jL);
    easu_set(&dir, &len, pp.x * iy, cL, fL, gL, hL, kL);
    easu_set(&dir, &len, ix * pp.y, fL, iL, jL, kL, nL);
    easu_set(&dir, &len, pp.x * pp.y, gL, jL, kL, lL, oL);
    float dx = (float)dir.x, dy = (float)dir.y, dirR = dx * dx + dy * dy; int zro = dirR < (float)(1.0 / 32768.0);
    dirR = zro ? 1.0f : 1.0f / sqrtf(dirR); dx = zro ? 1.0f : dx; dx *= dirR; dy *= dirR;
    float ln = (float)len * 0.5f; ln *= ln;
    float stretch = (dx * dx + dy * dy) / fmaxf(fabsf(dx), fabsf(dy));
    float l2x = 1.0f + (stretch - 1.0f) * ln, l2y = 1.0f + -0.5f * ln;
    float lobf = 0.5f + (float)((1.0 / 4.0 - 0.04) - 0.5) * ln;
    F2 dh = {(F1)dx, (F1)dy}, len2 = {(F1)l2x, (F1)l2y}; F1 lob = (F1)lobf, clp = (F1)(1.0f / lobf);
    F3 mn = {h_min(h_min(f.r, g.r), h_min(j.r, k.r)), h_min(h_min(f.g, g.g), h_min(j.g, k.g)), h_min(h_min(f.b, g.b), h_min(j.b, k.b))};
    F3 mx = {h_max(h_max(f.r, g.r), h_max(j.r, k.r)), h_max(h_max(f.g, g.g), h_max(j.g, k.g)), h_max(h_max(f.b, g.b), h_max(j.b, k.b))};
    F3 aC = {0, 0, 0}; F1 aW = 0;
    easu_tap(&aC, &aW,  0, -1, pp, dh, len2, lob, clp, b);
    easu_tap(&aC, &aW,  1, -1, pp, dh, len2, lob, clp, c);
    easu_tap(&aC, &aW, -1,  1, pp, dh, len2, lob, clp, i);
    easu_tap(&aC, &aW,  0,  1, pp, dh, len2, lob, clp, j);
    easu_tap(&aC, &aW,  0,  0, pp, dh, len2, lob, clp, f);
    easu_tap(&aC, &aW, -1,  0, pp, dh, len2, lob, clp, e);
    easu_tap(&aC, &aW,  1,  1, pp, dh, len2, lob, clp, k);
    easu_tap(&aC, &aW,  2,  1, pp, dh, len2, lob, clp, l);
    easu_tap(&aC, &aW,  2,  0, pp, dh, len2, lob, clp, h);
    easu_tap(&aC, &aW,  1,  0, pp, dh, len2, lob, clp, g);
    easu_tap(&aC, &aW,  1,  2, pp, dh, len2, lob, clp, o);
    easu_tap(&aC, &aW,  0,  2, pp, dh, len2, lob, clp, n);
    F1 r = aC.r / aW, gg = aC.g / aW, bb = aC.b / aW;
    out[0] = (float)h_min(mx.r, h_max(mn.r, r)); out[1] = (float)h_min(mx.g, h_max(mn.g, gg)); out[2] = (float)h_min(mx.b, h_max(mn.b, bb));
}
static F3 rfetch(const img_t *s, const int32_t clip[4], int x, int y) { return fetch(s, i_clamp(x, clip[0], clip[2]), i_clamp(y, clip[1], clip[3])); }
static F1 sel(int c, F1 a, F1 b) { return c ? a : b; }
static void rcas(const img_t *s, const int32_t clip[4], float kk, int x, int y, float out[3]) {
    F3 b = rfetch(s, clip, x, y - 1), d = rfetch(s, clip, x - 1, y), e = rfetch(s, clip, x, y), f = rfetch(s, clip, x + 1, y), h = rfetch(s, clip, x, y + 1);
    F1 B[3] = {b.r, b.g, b.b}, D[3] = {d.r, d.g, d.b}, E[3] = {e.r, e.g, e.b}, Fv[3] = {f.r, f.g, f.b}, H[3] = {h.r, h.g, h.b}, lobe3[3];
    for (int ch = 0; ch < 3; ch++) {
        F1 mn4 = h_min(h_min(B[ch], D[ch]), h_min(Fv[ch], H[ch])), mx4 = h_max(h_max(B[ch], D[ch]), h_max(Fv[ch], H[ch]));
        F1 hitMin = sel(mx4 > (F1)0.0, h_min(mn4, E[ch]) / ((F1)4.0 * mx4), (F1)1.0);
        F1 hitMax = sel(mn4 < (F1)1.0, ((F1)1.0 - h_max(mx4, E[ch])) / ((F1)4.0 * mn4 + (F1)(-4.0)), (F1)(-1.0));
        lobe3[ch] = h_max(-hitMin, hitMax);
    }
    F1 lobe = h_max((F1)(-FSR_RCAS_LIMIT), h_min(h_max(lobe3[0], h_max(lobe3[1], lobe3[2])), (F1)0.0)) * (F1)kk;
    F1 rcpL = (F1)1.0 / ((F1)4.0 * lobe + (F1)1.0);
    for (int ch = 0; ch < 3; ch++) out[ch] = (float)((lobe * B[ch] + lobe * D[ch] + lobe * H[ch] + lobe * Fv[ch] + E[ch]) * rcpL);
}
static void blit(const img_t *s, const vkfsr_push *p, int lx, int ly, float out[3]) {   // the sampler's bilinear filter, exact (a GPU's has 8 bits of sub-texel weights)
    float x = (float)lx * p->map[0] + p->map[2], y = (float)ly * p->map[1] + p->map[3];
    x = fminf(fmaxf(x, (float)p->clip[0]), (float)p->clip[2]); y = fminf(fmaxf(y, (float)p->clip[1]), (float)p->clip[3]);
    float fx = floorf(x), fy = floorf(y), ax = x - fx, ay = y - fy; int x0 = (int)fx, y0 = (int)fy, x1 = x0 + 1, y1 = y0 + 1;
    x0 = i_clamp(x0, 0, s->w - 1); x1 = i_clamp(x1, 0, s->w - 1); y0 = i_clamp(y0, 0, s->h - 1); y1 = i_clamp(y1, 0, s->h - 1);   // CLAMP_TO_EDGE
    for (int ch = 0; ch < 3; ch++) {
        const float *q = s->px + ch;
        float t0 = q[4 * ((size_t)y0 * s->w + x0)] * (1 - ax) + q[4 * ((size_t)y0 * s->w + x1)] * ax, t1 = q[4 * ((size_t)y1 * s->w + x0)] * (1 - ax) + q[4 * ((size_t)y1 * s->w + x1)] * ax;
        out[ch] = t0 * (1 - ay) + t1 * ay;
    }
}
static float srgb_enc(float x) { x = x > 0 ? x : 0; return x > 0.0031308f ? 1.055f * powf(x, 1.0f / 2.4f) - 0.055f : x * 12.92f; }   // fsr1.glsl fsr_srgb_enc / _dec
static float srgb_dec(float y) { y = y > 0 ? y : 0; return y > 0.04045f ? powf((y + 0.055f) * (1.0f / 1.055f), 2.4f) : y * (1.0f / 12.92f); }
static int ndsr_main(char **argv) {   // the swap's compute pass: the taps encoded as read (a per-texel function: the same as encoding the input first), EASU, the output's shape
    int w = atoi(argv[2]), h = atoi(argv[3]), ir[4] = {atoi(argv[4]), atoi(argv[5]), atoi(argv[6]), atoi(argv[7])}, ow = atoi(argv[8]), oh = atoi(argv[9]), shape = atoi(argv[11] + 4) + 1;
    FILE *f = fopen(argv[1], "rb"); if (!f) return 3;
    uint16_t *in = malloc((size_t)w * h * 8); if (fread(in, 8, (size_t)w * h, f) != (size_t)w * h) return 3; fclose(f);
    float *src = malloc((size_t)w * h * 16);
    for (size_t i = 0; i < (size_t)w * h * 4; i++) { float v = vkfsr_h2f(in[i]); src[i] = shape == VKNDSR_SRGB_LIN1 || i % 4 == 3 ? v : srgb_enc(v); }
    img_t S = {src, w, h}; int32_t clip[4] = {ir[0], ir[1], ir[0] + ir[2] - 1, ir[1] + ir[3] - 1};
    float sx = (float)ir[2] / (float)ow, sy = (float)ir[3] / (float)oh, mx = 0.5f * sx - 0.5f + (float)ir[0], my = 0.5f * sy - 0.5f + (float)ir[1];
    _Float16 *out = malloc((size_t)ow * oh * 8);
    for (int y = 0; y < oh; y++) for (int x = 0; x < ow; x++) {
        float o[3]; easu(&S, clip, (float)x * sx + mx, (float)y * sy + my, o);
        for (int c = 0; c < 3; c++) { float v = o[c];
            v = shape == VKNDSR_ENC ? fminf(fmaxf(v, 0), 1) : shape == VKNDSR_LIN128 ? fminf(srgb_dec(v), 128.0f) : srgb_dec(fminf(fmaxf(v, 0), 1));
            out[4 * ((size_t)y * ow + x) + c] = (_Float16)v; }
        out[4 * ((size_t)y * ow + x) + 3] = (_Float16)1.0f;
    }
    f = fopen(argv[15], "wb"); if (!f) return 4; fwrite(out, 8, (size_t)ow * oh, f); fclose(f);
    return 0;
}
static float unorm(float v, int bits) { float m = (float)((1 << bits) - 1); v = v < 0 ? 0 : v > 1 ? 1 : v; return floorf(v * m + 0.5f) / m; }   // a colour attachment's store, read back

int main(int argc, char **argv) {
    if (argc != 16) { fprintf(stderr, "usage: fsr1_ref in.rgba w h cx cy cw ch W H rot fsr|bilinear|ndsr0-3 stops bottom_up bits out.rgba\n"); return 2; }
    if (!strncmp(argv[11], "ndsr", 4)) return ndsr_main(argv);
    int w = atoi(argv[2]), hh = atoi(argv[3]), crop[4] = {atoi(argv[4]), atoi(argv[5]), atoi(argv[6]), atoi(argv[7])}, W = atoi(argv[8]), H = atoi(argv[9]), rot = atoi(argv[10]) & 3;
    int easu_path = !strcmp(argv[11], "fsr"); float stops = (float)atof(argv[12]); int bu = atoi(argv[13]), bits = atoi(argv[14]) == 8 ? 8 : 10;
    FILE *f = fopen(argv[1], "rb"); if (!f) return 3;
    uint8_t *in = malloc((size_t)w * hh * 4); if (fread(in, 4, (size_t)w * hh, f) != (size_t)w * hh) return 3; fclose(f);
    float *src = malloc((size_t)w * hh * 4 * sizeof(float)); for (size_t i = 0; i < (size_t)w * hh * 4; i++) src[i] = (float)in[i] / 255.0f;
    img_t S = {src, w, hh};
    int c[4]; vkfsr_crop(crop, w, hh, c); vkfsr_rect pic = vkfsr_fit(c[2], c[3], W, H);
    int two = stops >= 0, pw, ph; vkfsr_target_size(rot, W, H, &pw, &ph);
    float *mid = NULL; vkfsr_push pc;
    if (two) {   // the first pass: the picture, unturned, into the intermediate
        mid = malloc((size_t)pic.w * pic.h * 4 * sizeof(float)); memset(&pc, 0, sizeof pc); vkfsr_xform_id(pic.w, pic.h, &pc); vkfsr_map(c, pic, bu, &pc);
        for (int y = 0; y < pic.h; y++) for (int x = 0; x < pic.w; x++) {
            float o[3]; if (easu_path) easu(&S, pc.clip, (float)x * pc.map[0] + pc.map[2], (float)y * pc.map[1] + pc.map[3], o); else blit(&S, &pc, x, y, o);
            float *m = mid + 4 * ((size_t)y * pic.w + x); for (int ch = 0; ch < 3; ch++) m[ch] = unorm(o[ch], bits); m[3] = 1;
        }
    }
    img_t M = {mid, pic.w, pic.h};
    uint8_t *out = calloc((size_t)pw * ph, 4);
    memset(&pc, 0, sizeof pc); vkfsr_xform(rot, W, H, pic, &pc);
    if (two) { pc.clip[2] = pic.w - 1; pc.clip[3] = pic.h - 1; pc.k[0] = vkfsr_rcas_k(stops); } else vkfsr_map(c, pic, bu, &pc);
    for (int ty = 0; ty < ph; ty++) for (int tx = 0; tx < pw; tx++) {
        int lx = pc.xa[0] * tx + pc.xa[1] * ty + pc.xb[0], ly = pc.xa[2] * tx + pc.xa[3] * ty + pc.xb[1]; float o[3] = {0, 0, 0};
        if (lx >= 0 && ly >= 0 && lx < pc.xb[2] && ly < pc.xb[3]) {
            if (two) rcas(&M, pc.clip, pc.k[0], lx, ly, o);
            else if (easu_path) easu(&S, pc.clip, (float)lx * pc.map[0] + pc.map[2], (float)ly * pc.map[1] + pc.map[3], o);
            else blit(&S, &pc, lx, ly, o);
        }
        uint8_t *q = out + 4 * ((size_t)ty * pw + tx); for (int ch = 0; ch < 3; ch++) q[ch] = (uint8_t)(unorm(o[ch], 8) * 255.0f + 0.5f); q[3] = 255;
    }
    f = fopen(argv[15], "wb"); if (!f) return 4; fwrite(out, 4, (size_t)pw * ph, f); fclose(f);
    return 0;
}
