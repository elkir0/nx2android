// Copyright (c) 2021 Advanced Micro Devices, Inc. All rights reserved. MIT license: LICENSE in this directory.
//
// Android port: the present's upscaling passes (dm/android/gfx_vk_fsr.c), included by the stage files beside this directory (fsr_easu*.frag, fsr_rcas*.frag,
// fsr_blit.frag) with FSR_H 1 (float16_t arithmetic, shaderFloat16) or 0 (32-bit). AMD FidelityFX FSR 1's EASU (edge-adaptive spatial upsampling: 12 taps around the
// 2x2 quad, an analysis of the luma gradient into a direction and an edge amount, a Lanczos-2 approximation stretched along the edge, the result clamped to the 4
// nearest texels) and RCAS (robust contrast-adaptive sharpening over a + of 5 texels, its negative lobe limited from the ring's min and max so that no channel would
// clip at 0 or 1: it may overshoot the ring's range inside that, as AMD's does), as ffx_fsr1.h's
// FsrEasuF and FsrRcasF compute them, in the same order. Changes from ffx_fsr1.h, all for this port (tests/test_vk_fsr.py runs the C reference that mirrors this file):
//  - fragment shaders (one triangle over the target) instead of compute: the targets stay colour attachments, which keeps Adreno's framebuffer compression on the
//    swapchain and the intermediate image, the gamepad overlay is drawn in the same pass, and an sRGB swapchain needs no storage view (gfx_vk_fsr.c);
//  - texelFetch of the 12 taps instead of 4 gathers of each channel (the same 12 texture instructions), each coordinate clamped to the game's crop: the window
//    texture is larger than what the game drew (Mario Kart World: 1280x720 at the top left of 3840x2160), and nothing outside the crop may be read;
//  - AMD's approximate reciprocals (APrxLoRcpF1 / APrxLoRsqF1, integer tricks on the float bits) are the GPU's own; lenX / lenY divide by max(x, 2^-14) instead
//    (AMD's approximation is finite at 0, a division is not, and 0 * inf would be NaN); the direction's normalization and the kernel's shape are computed in 32 bits
//    in both variants: AMD's zero test 1/32768 is an FP16 subnormal, which a GPU that flushes FP16 subnormals would make 0, and then a flat area divides by 0;
//  - RCAS: a channel whose ring is all 0 (all 1) sets no lower (upper) limit; AMD's 0 * (1/0) is NaN there, which its max() drops on hardware with IEEE maxNum;
//    here the limit is a select, the same result without relying on NaN handling;
//  - the output pixel's input position (and a target's pre-rotation) comes from push constants P: a picture pixel l maps to input texel position
//    l * P.map.xy + P.map.zw (texel centres at integers), so a bottom-up window image is a negative y scale and the crop's origin is in the offset.
layout(set = 0, binding = 0) uniform sampler2D fsr_src;
// xa, xb: the target's pixel t as a pixel l of the picture (its top left at 0, in the window's orientation): l = (xa.x t.x + xa.y t.y, xa.z t.x + xa.w t.y) + xb.xy,
// inside the picture when 0 <= l < xb.zw (outside it the pass writes black: the letterbox). map: l to the input position (EASU, bilinear). clip: the input texels the
// pass may read, first and last, x0 y0 x1 y1. k: RCAS k.x its sharpness (linear: 2^-stops); bilinear k.xy 1 / the input texture's size
layout(push_constant) uniform FsrPush { ivec4 xa; ivec4 xb; vec4 map; ivec4 clip; vec4 k; } P;
#ifndef FSR_CS
layout(location = 0) out vec4 fsr_out;
#else
// FSR_CS: the swap of a game's own upscaler (gfx_vk_fsr.c "NDSR swap"): a compute pass into the image the game's dispatch would have written, through the view it would have
// written it with. One invocation an output pixel l of the output rectangle (xb.xy its origin in the image, xb.zw its size); k.y 1: the input is linear and each tap is
// sRGB-encoded as it is read (the network encodes its input so before it works); k.z the network's output: 0 encoded, clamped to 0-1; 1 linear, clamped to 0-1; 2 linear, at most 128
layout(local_size_x = 8, local_size_y = 8) in;
layout(set = 0, binding = 1) uniform writeonly image2D fsr_dst;
vec3 fsr_srgb_enc(vec3 x) { x = max(x, vec3(0.0)); return mix(x * 12.92, 1.055 * pow(x, vec3(1.0 / 2.4)) - 0.055, greaterThan(x, vec3(0.0031308))); }   // (above 1 the curve goes on)
vec3 fsr_srgb_dec(vec3 y) { y = max(y, vec3(0.0)); return mix(y * (1.0 / 12.92), pow((y + 0.055) * (1.0 / 1.055), vec3(2.4)), greaterThan(y, vec3(0.04045))); }
#endif

#if FSR_H
#define F1 float16_t
#define F2 f16vec2
#define F3 f16vec3
#else
#define F1 float
#define F2 vec2
#define F3 vec3
#endif
#define FSR_RCAS_LIMIT (0.25 - (1.0 / 16.0))
#define FSR_EPS (1.0 / 16384.0)

#ifndef FSR_CS
bool fsr_pixel(out ivec2 l) {
    ivec2 t = ivec2(gl_FragCoord.xy);
    l = ivec2(P.xa.x * t.x + P.xa.y * t.y, P.xa.z * t.x + P.xa.w * t.y) + P.xb.xy;
    return all(greaterThanEqual(l, ivec2(0))) && all(lessThan(l, P.xb.zw));
}
F3 fsr_fetch(int x, int y) { return F3(texelFetch(fsr_src, ivec2(x, y), 0).rgb); }
#else
F3 fsr_fetch(int x, int y) { vec3 c = texelFetch(fsr_src, ivec2(x, y), 0).rgb; return F3(P.k.y > 0.5 ? fsr_srgb_enc(c) : c); }   // (a uniform branch)
#endif
F1 fsr_luma2(F3 c) { return c.b * F1(0.5) + (c.r * F1(0.5) + c.g); }   // luma times 2, in 2 FMAs (AMD's)

// ---- EASU ----
// The gradient around one texel of the 2x2 quad, weighted by its bilinear weight w:   a
// the '+' differences give the direction, and their reversal the edge amount        b c d
// (0 where the luma turns back, 1 where it keeps going)                               e
void fsr_easu_set(inout F2 dir, inout F1 len, F1 w, F1 lA, F1 lB, F1 lC, F1 lD, F1 lE) {
    F1 dc = lD - lC, cb = lC - lB;
    F1 lenX = F1(1.0) / max(max(abs(dc), abs(cb)), F1(FSR_EPS));
    F1 dirX = lD - lB;
    dir.x += dirX * w;
    lenX = clamp(abs(dirX) * lenX, F1(0.0), F1(1.0));
    lenX *= lenX;
    len += lenX * w;
    F1 ec = lE - lC, ca = lC - lA;
    F1 lenY = F1(1.0) / max(max(abs(ec), abs(ca)), F1(FSR_EPS));
    F1 dirY = lE - lA;
    dir.y += dirY * w;
    lenY = clamp(abs(dirY) * lenY, F1(0.0), F1(1.0));
    lenY *= lenY;
    len += lenY * w;
}
// One tap: its offset rotated into the edge's frame and stretched (len), then (25/16 (2/5 x^2 - 1)^2 - (25/16 - 1)) (lob x^2 - 1)^2, a Lanczos-2 approximation
// without sin, rcp or sqrt, whose window (lob) narrows from sqrt(2) to a little over 2 with the edge amount; d2 is clipped to the window's end (clp)
void fsr_easu_tap(inout F3 aC, inout F1 aW, F2 off, F2 dir, F2 len, F1 lob, F1 clp, F3 c) {
    F2 v = F2(off.x * dir.x + off.y * dir.y, off.x * (-dir.y) + off.y * dir.x);
    v *= len;
    F1 d2 = min(v.x * v.x + v.y * v.y, clp);
    F1 wB = F1(2.0 / 5.0) * d2 + F1(-1.0);
    F1 wA = lob * d2 + F1(-1.0);
    wB *= wB;
    wA *= wA;
    wB = F1(25.0 / 16.0) * wB + F1(-(25.0 / 16.0 - 1.0));
    F1 w = wB * wA;
    aC += c * w;
    aW += w;
}
// pos: the output pixel's centre in input texels (texel centres at integers). The taps, f the quad's top left:    b c
//                                                                                                              e f g h
//                                                                                                              i j k l
//                                                                                                                n o
vec3 fsr_easu(vec2 pos) {
    vec2 fl = floor(pos);
    F2 pp = F2(pos - fl);
    ivec2 q = ivec2(fl);
    int x0 = clamp(q.x - 1, P.clip.x, P.clip.z), x1 = clamp(q.x, P.clip.x, P.clip.z), x2 = clamp(q.x + 1, P.clip.x, P.clip.z), x3 = clamp(q.x + 2, P.clip.x, P.clip.z);
    int y0 = clamp(q.y - 1, P.clip.y, P.clip.w), y1 = clamp(q.y, P.clip.y, P.clip.w), y2 = clamp(q.y + 1, P.clip.y, P.clip.w), y3 = clamp(q.y + 2, P.clip.y, P.clip.w);
    F3 b = fsr_fetch(x1, y0), c = fsr_fetch(x2, y0);
    F3 e = fsr_fetch(x0, y1), f = fsr_fetch(x1, y1), g = fsr_fetch(x2, y1), h = fsr_fetch(x3, y1);
    F3 i = fsr_fetch(x0, y2), j = fsr_fetch(x1, y2), k = fsr_fetch(x2, y2), l = fsr_fetch(x3, y2);
    F3 n = fsr_fetch(x1, y3), o = fsr_fetch(x2, y3);
    F1 bL = fsr_luma2(b), cL = fsr_luma2(c), eL = fsr_luma2(e), fL = fsr_luma2(f), gL = fsr_luma2(g), hL = fsr_luma2(h);
    F1 iL = fsr_luma2(i), jL = fsr_luma2(j), kL = fsr_luma2(k), lL = fsr_luma2(l), nL = fsr_luma2(n), oL = fsr_luma2(o);
    F2 dir = F2(0.0); F1 len = F1(0.0);
    F1 ix = F1(1.0) - pp.x, iy = F1(1.0) - pp.y;
    fsr_easu_set(dir, len, ix * iy, bL, eL, fL, gL, jL);       // s (f)
    fsr_easu_set(dir, len, pp.x * iy, cL, fL, gL, hL, kL);     // t (g)
    fsr_easu_set(dir, len, ix * pp.y, fL, iL, jL, kL, nL);     // u (j)
    fsr_easu_set(dir, len, pp.x * pp.y, gL, jL, kL, lL, oL);   // v (k)
    // the direction normalized (none: along x), the edge amount from {0 to 2} to {0 to 1} and squared, the kernel's stretch ({1 across x or y, to sqrt(2) on a
    // diagonal}) along the edge and squeeze across it, and its window: 32 bits (see the top)
    vec2 d = vec2(dir); float dirR = d.x * d.x + d.y * d.y; bool zro = dirR < 1.0 / 32768.0;
    dirR = zro ? 1.0 : inversesqrt(dirR); d.x = zro ? 1.0 : d.x; d *= dirR;
    float ln = float(len) * 0.5; ln *= ln;
    float stretch = (d.x * d.x + d.y * d.y) / max(abs(d.x), abs(d.y));
    vec2 ln2 = vec2(1.0 + (stretch - 1.0) * ln, 1.0 + -0.5 * ln);
    float lobf = 0.5 + ((1.0 / 4.0 - 0.04) - 0.5) * ln;
    F2 dh = F2(d), len2 = F2(ln2); F1 lob = F1(lobf), clp = F1(1.0 / lobf);
    F3 min4 = min(min(f, g), min(j, k)), max4 = max(max(f, g), max(j, k));   // the dering: the 4 nearest texels' range
    F3 aC = F3(0.0); F1 aW = F1(0.0);
    fsr_easu_tap(aC, aW, F2( 0.0, -1.0) - pp, dh, len2, lob, clp, b);
    fsr_easu_tap(aC, aW, F2( 1.0, -1.0) - pp, dh, len2, lob, clp, c);
    fsr_easu_tap(aC, aW, F2(-1.0,  1.0) - pp, dh, len2, lob, clp, i);
    fsr_easu_tap(aC, aW, F2( 0.0,  1.0) - pp, dh, len2, lob, clp, j);
    fsr_easu_tap(aC, aW, F2( 0.0,  0.0) - pp, dh, len2, lob, clp, f);
    fsr_easu_tap(aC, aW, F2(-1.0,  0.0) - pp, dh, len2, lob, clp, e);
    fsr_easu_tap(aC, aW, F2( 1.0,  1.0) - pp, dh, len2, lob, clp, k);
    fsr_easu_tap(aC, aW, F2( 2.0,  1.0) - pp, dh, len2, lob, clp, l);
    fsr_easu_tap(aC, aW, F2( 2.0,  0.0) - pp, dh, len2, lob, clp, h);
    fsr_easu_tap(aC, aW, F2( 1.0,  0.0) - pp, dh, len2, lob, clp, g);
    fsr_easu_tap(aC, aW, F2( 1.0,  2.0) - pp, dh, len2, lob, clp, o);
    fsr_easu_tap(aC, aW, F2( 0.0,  2.0) - pp, dh, len2, lob, clp, n);
    return vec3(min(max4, max(min4, aC / aW)));
}
#ifndef FSR_CS
void fsr_easu_main() {
    ivec2 l; if (!fsr_pixel(l)) { fsr_out = vec4(0.0, 0.0, 0.0, 1.0); return; }
    fsr_out = vec4(fsr_easu(vec2(l) * P.map.xy + P.map.zw), 1.0);
}
#else
void fsr_ndsr_main() {   // EASU in the network's encoded space, then its output's space and range; alpha 1, as the network writes it
    ivec2 l = ivec2(gl_GlobalInvocationID.xy);
    if (any(greaterThanEqual(l, P.xb.zw))) return;
    vec3 c = fsr_easu(vec2(l) * P.map.xy + P.map.zw);
    int shape = int(P.k.z + 0.5);
    c = shape == 0 ? clamp(c, 0.0, 1.0) : shape == 1 ? fsr_srgb_dec(clamp(c, 0.0, 1.0)) : min(fsr_srgb_dec(c), vec3(128.0));
    imageStore(fsr_dst, l + P.xb.xy, vec4(c, 1.0));
}
#endif

#ifndef FSR_CS
// ---- RCAS ----
// The +:  b    sharpened as (lobe (b + d + f + h) + e) / (4 lobe + 1), lobe <= 0 the largest negative lobe that keeps every channel inside [0, 1] given its ring's
//       d e f  min and max (hitMin, hitMax), at most FSR_RCAS_LIMIT, times the sharpness (P.k.x = 2^-stops). Read from the intermediate image at the picture's own
//         h    pixel, clamped to it (P.clip)
F3 fsr_rcas_fetch(ivec2 q) { q = clamp(q, P.clip.xy, P.clip.zw); return F3(texelFetch(fsr_src, q, 0).rgb); }
vec3 fsr_rcas(ivec2 q) {
    F3 b = fsr_rcas_fetch(q + ivec2(0, -1)), d = fsr_rcas_fetch(q + ivec2(-1, 0)), e = fsr_rcas_fetch(q), f = fsr_rcas_fetch(q + ivec2(1, 0)), h = fsr_rcas_fetch(q + ivec2(0, 1));
    F3 mn4 = min(min(b, d), min(f, h)), mx4 = max(max(b, d), max(f, h));
    F3 hitMin = mix(F3(1.0), min(mn4, e) / (F1(4.0) * mx4), greaterThan(mx4, F3(0.0)));
    F3 hitMax = mix(F3(-1.0), (F3(1.0) - max(mx4, e)) / (F1(4.0) * mn4 + F1(-4.0)), lessThan(mn4, F3(1.0)));
    F3 lobeRGB = max(-hitMin, hitMax);
    F1 lobe = max(F1(-FSR_RCAS_LIMIT), min(max(lobeRGB.r, max(lobeRGB.g, lobeRGB.b)), F1(0.0))) * F1(P.k.x);
    F1 rcpL = F1(1.0) / (F1(4.0) * lobe + F1(1.0));
    return vec3((lobe * b + lobe * d + lobe * h + lobe * f + e) * rcpL);
}
void fsr_rcas_main() {
    ivec2 l; if (!fsr_pixel(l)) { fsr_out = vec4(0.0, 0.0, 0.0, 1.0); return; }
    fsr_out = vec4(fsr_rcas(l), 1.0);
}

// ---- bilinear (DM_VK_UPSCALE=bilinear, and any picture that is not made larger): the blit's mapping, its position clamped to the crop's texel centres, so the
// filter never reaches a texel outside the crop (vkCmdBlitImage clamps at the image's edge, not the region's) ----
void fsr_blit_main() {
    ivec2 l; if (!fsr_pixel(l)) { fsr_out = vec4(0.0, 0.0, 0.0, 1.0); return; }
    vec2 pos = clamp(vec2(l) * P.map.xy + P.map.zw, vec2(P.clip.xy), vec2(P.clip.zw));
    fsr_out = vec4(textureLod(fsr_src, (pos + 0.5) * P.k.xy, 0.0).rgb, 1.0);
}
#endif
