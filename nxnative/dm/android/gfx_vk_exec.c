// Android port, Vulkan backend: the command replay (mtl_exec, see gfx_vk.h and nvn_obj.h's record format).
// State records update a shadow of the NVN state; a draw or dispatch turns it into a pipeline (cached by everything that shapes it), descriptor sets (one per
// stage, filled from the shader's reflection: uniform slots, the shader's c1, the driver's c0 / c2 segments, textures and samplers by handle, images, storage
// buffers) and the draw. Render passes open lazily at the first draw or clear after SetRenderTargets and close before anything that is not drawing.
// Version 1 orders everything with whole-pipeline barriers around passes, copies and dispatches.
#include "gfx_vk.h"
#include "gfx_vk_vt.h"
#include "gfx_vk_fsr.h"   // the NDSR swap (dispatch)
#include "nvn_hmap.h"
#include <math.h>
#include <string.h>
#include <sys/resource.h>
#include <unistd.h>

void vk_shader_c1(vkshader_t *s, VkBuffer *buf, VkDeviceSize *off, VkDeviceSize *len);
void vk_img_wrote(vkimg_t *im);   // gfx_vk_tex.c: the GPU writes this image (an attachment, a storage image, a copy or clear destination)
// a GPU buffer for [a, a+n), placed as one range (vk_buf_at), else for its first chunk or part (cut at its end, as when ranges were looked up by their first byte)
static int buf_at_n(uint64_t a, uint64_t n, VkBuffer *b, VkDeviceSize *o) { return vk_buf_at(a, n ? n : 1, b, o) || (n > 1 && vk_buf_at(a, 1, b, o)); }
VkBufferView vk_tbuf_view(const tex_t *t);
void vk_sub_update(uint64_t dst, const void *data, uint64_t n);
int prog_key(const prog_t *p, unsigned st, uint8_t key[8]);

// ---------------- NVN state shadow ----------------
static struct {
    const prog_t *prog; vkshader_t *sh[6]; int prog_ok;
    p_bind_buf ubo[6][16], ssbo[6][16];
    uint64_t tex[6][256], samp[6][128], img[6][32], tex_sseq[6][256], samp_sseq[6][128];   // *_sseq: the sampler registrations made before the bind (gfx_vk_tex.c vk_samp_at)
    blend_st blend[8]; color_st color; chmask_st chmask; ms_st ms; poly_st poly; ds_st ds;
    uint32_t vattr[32]; vs_st vstream[16]; NVNbufferRange vbuf[16];   // vattr: 32 on NVN2 (BindVertexAttribState's n, hle_nvn.c), 16 on NVN; 16 streams on both
    uint8_t push[2][DMSH_PUSH_SIZE]; float alpha_ref;   // NVN2 push constants of the graphics stages and of compute (OP_PUSH_CONST, bound as c[24]: push_shared); SetAlphaRef (OP_ALPHA_REF)
    float vp[16][4]; float dnear, dfar; p_scissor sc; int sc_set; uint8_t vp_sw[16][4];   // vp_sw: viewport swizzles XOR the identity (p_vp_swizzle)
    int sref[2], smask[2], swmask[2];
    p_poly_offset pofs; float blendc[4]; int depth_clamp, rast_discard, prim_restart;
    tex_t rt_col[8], rt_dep; view_t rt_cview[8], rt_dview; uint8_t rt_has_col[8], rt_has_cview[8], rt_has_dep, rt_has_dview; int rt_n; uint32_t rt_bad[8];   // rt_bad: the first word of a colour slot that is not a texture object (traces)
    int64_t row_stride, image_stride;
    // the open render pass
    int pass_open; uint32_t pass_w, pass_h; VkFormat cfmt[8], dfmt; VkImageAspectFlags daspect; uint32_t samples; int ncol;
    VkPipeline bound_pipe; int rp_pass;   // rp_pass: the open pass began with vkCmdBeginRenderPass
} S;
static p_tf_bind g_tf[4]; static uint64_t g_tf_off[4]; static int g_tf_on, g_tf_in_pass; static uint32_t g_tf_verts;   // transform feedback: buffers, bytes written into each, capture on, a capture in the open pass (xfb_begin), vertices captured since Begin
static int g_depth_zero = 1, g_lower_left, g_queue_ll = -1;
void mtl_set_depth_zero(int on) { g_depth_zero = on; }
void mtl_set_origin_lower_left(int on) { g_lower_left = on; }
// The raster flip NVN captures when a queue is initialized (hle_nvn.c sets it under the submit lock, before each submission): the winding follows it, the viewport the device's
// origin now (nvn_mtl.m g_queue_origin_ll, upstream's "raster flip captured at NVN queue creation"). -1 until a host that sets it runs: the device's origin, as before
void mtl_set_queue_origin_lower_left(int on) { g_queue_ll = on != 0; }
// NVN's upper-left mode has clip space y up (as Metal and D3D): it is drawn with a flipped viewport over the same top-left rectangle, lower-left mode without
// one; the scissor is then never flipped and the winding follows nvn_mtl.m's. Drawn the other way, every blit flipped the picture (NEO docked: upside down)
// and the winding had to be inverted. DM_VK_FLIPY=0 brings the old way back
#include <dlfcn.h>
static int g_flipy;
// Lower-left mode then stores the window image bottom-up (GL's rows), so the present flips it as nvn_mtl.m does (Celeste's MonoGame keeps NVN's default lower-left origin)
int vk_window_bottom_up(void) { return g_lower_left && g_flipy; }
static void state_reset(void) {
    memset(&S, 0, sizeof S);
    S.dnear = 0; S.dfar = 1;
    S.chmask.mask = 0xffffffffu;
    S.color.logic_op = 3;
    S.poly.front = 1; S.poly.mode = 2;
    S.ds.dw = 1; S.ds.df = 2; S.ds.sf_f = S.ds.sf_b = 8; S.ds.fail_f = S.ds.zfail_f = S.ds.zpass_f = S.ds.fail_b = S.ds.zfail_b = S.ds.zpass_b = 1;
    S.smask[0] = S.smask[1] = S.swmask[0] = S.swmask[1] = 0xff;
    for (int i = 0; i < 8; i++) { S.blend[i].target = i; S.blend[i].src_rgb = S.blend[i].src_a = 2; S.blend[i].dst_rgb = S.blend[i].dst_a = 1; S.blend[i].eq_rgb = S.blend[i].eq_a = 1; }
}
static void prewarm_start(void); static void compute_warm_start(void); static void build_dump(void);
static uint64_t g_set_epoch = 1;   // the submission the descriptor set cache belongs to (stage_set)
static int32_t g_pass_bb[4];   // bounding box of the scissors the open pass's draws used (x0 y0 x1 y1)
static int g_cb_draws;   // draws in the command buffer being recorded (vk_cmd_split after DM_VK_CB_DRAWS draws, 0 = never)
static int g_psplit, g_psplit_depth, g_psplit_now, g_ord_parts, g_setsplit;   // DM_VK_PASS_SPLIT / _DEPTH (prepare_draw); g_psplit_now: the pass being begun goes on with the one just ended (begin_pass cuts before it);
                                                                    // g_ord_parts: ordinary DM_VK_CB_DRAWS cuts in this submission, kept to the old 15 (MAIN_CBS was 16): the extra parts are for pass splits
static int g_var_now;   // the bisection's change for the draw being recorded (see g_bis_var)
static int g_tracing;   // the draw being recorded is the one the bisection found (its descriptors are logged)
// replay counters, printed by gfx_vk_present.c every 600 presents (draws done and skipped by reason, passes, clears folded into load ops, dispatches)
struct { uint64_t draws, skip_prog, skip_rt, skip_pipe, passes, folded, dispatches, pipes_built, rt_kept, ns_exec, ns_desc, ns_pipe, max_pass_draws; } g_vkst; static uint64_t g_pass_draws;   // ns_*: time in the replay, its descriptor sets, its pipeline keys and lookups
static inline uint64_t tnow(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return (uint64_t)t.tv_sec * 1000000000ull + (uint64_t)t.tv_nsec; }
// The per-draw timers (about 10 a draw, ~3,000 draws a frame): the arm64 counter register read directly, a few ns against clock_gettime's ~20-40, in ns
static inline uint64_t tick(void) {
#if defined(__aarch64__)
    static uint64_t q; if (!q) { uint64_t f; __asm__ volatile("mrs %0, cntfrq_el0" : "=r"(f)); q = (1000000000ull << 32) / f; }
    uint64_t c; __asm__ volatile("mrs %0, cntvct_el0" : "=r"(c)); return (uint64_t)(((unsigned __int128)c * q) >> 32);
#else
    return tnow();
#endif
}
// Measurement switches, read again at every submission so DM_CTL "env" lines change them while the game runs (never for normal play: they draw wrong):
// DM_VK_X_NOBAR=1 no barrier before a pass, DM_VK_X_DEPTH=1 depth neither loaded nor stored, DM_VK_X_COLOR=1 colour not loaded unless cleared
static int g_x_nobar, g_x_depth, g_x_color, g_x_cbdraws, g_x_tracepass = -1, g_sub_pass;   // DM_VK_TRACE_PASS=n: log every draw of the submission's pass n (once)
static int g_x_listpasses;
uint64_t vk_replay_ns(void) { return g_vkst.ns_exec; }   // the replay time so far (gfx_vk_present.c: a frame's cost for the adaptive rate)
// flicker experiments (DM_CTL env lines, read every submission): DM_VK_X_NOSETC no descriptor set reuse, DM_VK_X_NODRVC no driver segment reuse, DM_VK_X_LANDCUT deferred
// uniform updates land at each cut, DM_VK_X_NOKEEP no pass kept for the same targets, DM_VK_X_NOSHC no shadow reuse
static int g_fdz = 1;  // DM_VK_FDZ (live): fragment-depth variants for draws on depth written by fragment shaders (pipeline key fdz); 0 none; 2 only for draws that write no depth (A/B: the
                       // variant takes early depth and LRZ from every draw it is on, and Mario Kart World's main D24S8 has it, run 13; Hades II's lights, its first case, wrote no depth)
// Task / mesh pipelines, A/B switches to bisect a driver compile (live: read each submission, taken into the key of the pipelines made from then on). Run 4 of Mario Kart World 1.8.0
// crashed in the Adreno 840's compiler (libllvm-qgl.so +0x7be0ac, address 0x10) building its first one: mesh 30df8b09fea2f84b, task 0bf222be05e595d3, fragment 4f296d872429b144 with
// its depth variant, FILL. DM_VK_MESH_NOFDZ=1 no fragment depth variant; DM_VK_MESH_NOTASK=1 the pipeline without its task stage (the mesh shader's payload made Private), its draws
// not made; DM_VK_MESH_FILL=1 polygon mode FILL whatever the state; DM_VK_MESH_POSBLOCK=1 the mesh shader's gl_Position in a per-vertex block as glslang declares it (the bundle's
// is a standalone BuiltIn Position array); DM_VK_MESH_VISTATE=1 the vertex input and input assembly states given as before (by default none: ignored for mesh pipelines, which
// the specification lets be NULL, as most mesh shader programs pass them). No subgroup-free mesh SPIR-V: the translator emulates 32-lane warps with subgroup operations and has
// no other path (DM_VK_SPV_DIR with DM_VK_SPV_ALT=1 takes a module made apart)
static int g_mesh_nofdz, g_mesh_notask, g_mesh_fill, g_mesh_posblock, g_mesh_vistate;
// What the integrated build of Mario Kart World added, each with a live A/B switch (read each submission; run 13: 35-38 ms of GPU a frame against 16 without them): DM_VK_MESH=0 leaves out the
// DrawMeshTasks draws (at start-up it also keeps VK_EXT_mesh_shader off, as before); DM_VK_ALPHA=0 draws without the alpha-test variant, as before it came (alpha-tested pixels then kept);
// DM_VK_FIRST_STORE=0 leaves out the first-invocation stores the replay makes before a dispatch (a cost measurement only: the translator took them out of the shaders, so the lists they
// reset are not reset and draw wrong)
static int g_mesh_live = 1, g_alpha_on = 1, g_first_store = 1;
static int g_disc = 1;   // DM_VK_DISCARD (live; "discards" below): 1 NVN's depth / stencil discards become DONT_CARE loads and stores, 2 colour ones too, 0 none
static int g_x_csalt = -1; static uint32_t g_x_csalt_m, g_x_csalt_r;   // DM_VK_X_CSALT (csalt_of)
static int g_x_csserial; static uint32_t g_x_csserial_m, g_x_csserial_r;   // DM_VK_X_CSSERIAL (serial_dispatch)
static int g_spill_cpuind = 1; static uint64_t g_spill_cpuind_n, g_spill_gpuind_n;   // DM_VK_X_SPILL_CPUIND (dispatch), and the indirect spilled dispatches that took each path
static int g_x_indlist; static unsigned g_x_indlist_gen;   // DM_VK_X_INDLIST (indlist_note)
static int g_ind_c0 = 1;   // DM_VK_IND_C0 (dispatch: an indirect dispatch's group counts copied into c0)
static int g_x_nosetc, g_x_nodrvc, g_x_landcut, g_x_nokeep, g_x_nobc, g_x_cutany, g_lrz_touch, g_xfb_on = 1, g_x_nogs, g_spv_alt, g_x_nodispbar, g_spill_split = 1; int g_x_noshc;   // g_xfb_on: DM_VK_XFB=0 captures no transform feedback (A/B)
static int g_cbc_on = 1;   // DM_VK_CBUF_CACHE (see cbuf_desc)
static int g_tvc_keep_env = 1;   // DM_VK_TVC_KEEP (see tex_desc_for)
static int g_keep_on = 1;   // DM_VK_SET_KEEP (see kept_get)
static int g_keep_fine = 1;   // DM_VK_KEEP_FINE (see kept_check)
static int g_x_nosetmemo, g_x_setverify;   // DM_VK_X_NOSETMEMO=1 (live, A/B): no per-stage set memo; DM_VK_X_SETVERIFY=1 (live): one memo hit in 64 checked (see stage_set)
static int g_vbound = 1;   // DM_VK_VBOUND (see vbuf_bound)
// DM_VK_PROF_SLOW=1 (live): a submission whose replay took over 40 ms logs where it went: pipelines built on the replay thread, textures (vk_tex: images made and
// packaged uploads, the latter also apart), the copied pools' sync before it, shader modules, and the rest
int g_prof_on; uint64_t g_prof_ns[8]; static uint64_t g_prof_t0, g_prof_pkg0, g_prof_npkg0, g_prof_pipes0;
static int g_async_pipe = 1;   // DM_VK_ASYNC_PIPE (see async_pipe)
static int g_lin_back = 1;   // DM_VK_LIN_BACK (see lin_back)
static int g_samp_snap = 1;   // DM_NVN_SAMP_SNAP=0 (live): samplers as registered now, not as the bind saw them (gfx_vk_tex.c vk_samp_at)
static uint64_t g_x_logfs;   // DM_VK_LOG_FS=<fragment shader key> (live): one line per draw with it, with its samplers as the draw sees them
static uint64_t g_x_skipvs_m, g_x_skipvs_r; static unsigned g_x_skipvs_gen;   // DM_VK_X_SKIPVS_MOD (prepare_draw)
static int g_rd_touch = 1, g_x_rdkeep, g_x_nskipvs, g_x_nskipcs, g_x_mdiall; static uint64_t g_x_skipvs[8], g_x_skipcs[8];   // + DM_VK_X_SKIPCS, DM_VK_X_MDI_ALL (diagnostics, live)
   // DM_VK_RD_TOUCH, DM_VK_X_RDKEEP, DM_VK_X_SKIPVS (see prepare_draw)
static char g_x_tracesize[40], g_x_traceframe[40]; static int g_x_traceframe_on;
// DM_VK_WATCH=<pool offset, hex>[#n] (live): log the next n (20) uses of images at that offset of their pool (attachment, sampled, copy), with the pass and present
// numbers, to find what fills a texture a pass reads
static long long g_watch_off = -1; static int g_watch_n; extern uint64_t g_tex_frame;
// DM_VK_UBO_LOG=<shader key, hex>[#n] (live): log the next n (40) uniform buffers that shader reads (present, stage, register, address, a hash and the first 16 words)
static uint64_t g_ubolog_key; static int g_ubolog_n;
static void ubo_log(const vkshader_t *s, int st, int reg, uint64_t a, uint64_t n) {
    if (!g_ubolog_key || s->key != g_ubolog_key || g_ubolog_n <= 0) return;
    g_ubolog_n--; uint8_t buf[4096]; if (n > sizeof buf) n = sizeof buf; void vk_ubd_read(uint64_t a, uint64_t n, uint8_t *dst); vk_ubd_read(a, n, buf);
    uint64_t h = 1469598103934665603ull; for (uint64_t i = 0; i < n; i++) h = (h ^ buf[i]) * 1099511628211ull; float f[16] = {0}; memcpy(f, buf, n < 64 ? n : 64);
    char t[4096]; int k = 0; uint32_t nw = (uint32_t)(n < 512 ? n : 512) / 4; (void)f;   // every word up to 512 bytes, as floats
    for (uint32_t i = 0; i < nw && k < (int)sizeof t - 24; i++) { float v; memcpy(&v, buf + 4 * i, 4); k += snprintf(t + k, sizeof t - k, "%s%g", i ? (i % 4 ? " " : " | ") : "", v); }
    if (st == 0 && reg == 0 && S.vbuf[0].address && vk_cpu_ptr(S.vbuf[0].address, S.vbuf[0].size > sizeof buf ? sizeof buf : S.vbuf[0].size)) {   // the vertex stream 0 the draw reads, hashed the same way (CPU view at replay)
        uint64_t vn = S.vbuf[0].size > sizeof buf ? sizeof buf : S.vbuf[0].size, vh = 1469598103934665603ull; memcpy(buf, vk_cpu_ptr(S.vbuf[0].address, vn), vn);
        for (uint64_t i = 0; i < vn; i++) vh = (vh ^ buf[i]) * 1099511628211ull;
        float v[6] = {0}; memcpy(v, buf, vn < 24 ? vn : 24);
        if (S.rt_has_dep) VKLOG("ubo: present %llu pass %d depth target pool %#llx+%#llx serial %llu; depth test %d func %d", (unsigned long long)g_tex_frame, g_sub_pass - 1,
                                (unsigned long long)(uintptr_t)S.rt_dep.pool, (unsigned long long)S.rt_dep.off, (unsigned long long)S.rt_dep.serial, S.ds.dt, S.ds.df);
        VKLOG("ubo: present %llu vertex stream 0 %#llx +%llu hash %016llx: %g %g %g %g %g %g", (unsigned long long)g_tex_frame, (unsigned long long)S.vbuf[0].address, (unsigned long long)S.vbuf[0].size,
              (unsigned long long)vh, v[0], v[1], v[2], v[3], v[4], v[5]);
        void vk_ubd_read(uint64_t a, uint64_t n, uint8_t *dst); vk_ubd_read(a, n, buf); }
    VKLOG("ubo: present %llu pass %d stage %d c%d %#llx +%llu hash %016llx: %s", (unsigned long long)g_tex_frame, g_sub_pass - 1, st, reg, (unsigned long long)a, (unsigned long long)n, (unsigned long long)h, t);
}
// DM_VK_TEXDUMP=<pool hex>+<offset hex> (read when a pass dump starts; the trace prints both): during the dump, the first sampled image there is blitted (level 0, up to 6 layers)
// into RGBA16F images and dumped with the passes, so a compressed texture (BC6H cube maps...) can be looked at decoded by the GPU
static long long g_td_off = -1; static unsigned long long g_td_pool; static vkimg_t *g_td_im; static int g_td_done;
void vk_texdump_note(vkimg_t *im) { if (g_td_off >= 0 && !g_td_done && !g_td_im && im && im->off == g_td_off && im->pool == g_td_pool) g_td_im = im; }
void vk_watch(const vkimg_t *im, const char *what) {
    if (g_watch_off < 0 || !im || im->off != g_watch_off || g_watch_n <= 0) return;
    g_watch_n--; VKLOG("watch: present %llu pass %d (%ux%u): %s %ux%u fmt %d pool %#llx+%#llx (image %p, nvn flags %#x, %d overlapping)", (unsigned long long)g_tex_frame, g_sub_pass - 1, S.pass_w, S.pass_h, what, im->w, im->h, im->fmt,
                       (unsigned long long)im->pool, (unsigned long long)im->off, (const void *)im, im->flags, im->novl);
}
static void x_read(void) { const char *c = getenv("DM_VK_CB_DRAWS"); g_x_cbdraws = c ? atoi(c) : 128; g_sub_pass = 0;
    const char *fy = getenv("DM_VK_FLIPY"); g_flipy = !(fy && !atoi(fy));   // on unless DM_VK_FLIPY=0
    if (g_x_listpasses) { g_x_listpasses = 0; setenv("DM_VK_TRACE_PASS", "-1", 1); }
    const char *tp = getenv("DM_VK_TRACE_PASS"); g_x_tracepass = tp ? atoi(tp) : -1;
    { const char *w = getenv("DM_VK_WATCH"); static char last[40]; if (!w || !*w || !strcmp(w, "0")) g_watch_off = -1;
      else if (strcmp(w, last)) { g_watch_off = strtoll(w, NULL, 16); const char *h = strchr(w, '#'); g_watch_n = h ? atoi(h + 1) : 20; }
      snprintf(last, sizeof last, "%s", w ? w : ""); }
    { const char *w = getenv("DM_VK_UBO_LOG"); static char last[40]; if (!w || !*w || !strcmp(w, "0")) g_ubolog_key = 0;
      else if (strcmp(w, last)) { g_ubolog_key = strtoull(w, NULL, 16); const char *h = strchr(w, '#'); g_ubolog_n = h ? atoi(h + 1) : 40; }
      snprintf(last, sizeof last, "%s", w ? w : ""); }
    if (g_x_traceframe_on) { g_x_traceframe_on = 0; setenv("DM_VK_TRACE_FRAME", "", 1); }   // one submission traced
    { const char *tf = getenv("DM_VK_TRACE_FRAME"); snprintf(g_x_traceframe, sizeof g_x_traceframe, "%s", tf && strcmp(tf, "0") ? tf : ""); }
    { const char *tz = getenv("DM_VK_TRACE_SIZE"); snprintf(g_x_tracesize, sizeof g_x_tracesize, "%s", tz ? tz : ""); } const char *e; g_x_nobar = (e = getenv("DM_VK_X_NOBAR")) && atoi(e); g_x_depth = (e = getenv("DM_VK_X_DEPTH")) && atoi(e); g_x_color = (e = getenv("DM_VK_X_COLOR")) && atoi(e);
    g_x_nosetc = (e = getenv("DM_VK_X_NOSETC")) && atoi(e); g_x_nodrvc = (e = getenv("DM_VK_X_NODRVC")) && atoi(e); g_x_landcut = (e = getenv("DM_VK_X_LANDCUT")) && atoi(e);
    g_x_nodispbar = (e = getenv("DM_VK_X_NODISPBAR")) && atoi(e); g_spill_split = !((e = getenv("DM_VK_SPILL_SPLIT")) && !atoi(e));
    g_ind_c0 = !((e = getenv("DM_VK_IND_C0")) && !atoi(e));
    { int il = (e = getenv("DM_VK_X_INDLIST")) && atoi(e); if (il && !g_x_indlist) g_x_indlist_gen++; g_x_indlist = il; }
    g_spill_cpuind = !((e = getenv("DM_VK_X_SPILL_CPUIND")) && !atoi(e));
    g_x_csserial = (e = getenv("DM_VK_X_CSSERIAL")) && atoi(e); g_x_csserial_m = g_x_csserial_r = 0; if (g_x_csserial && (e = strchr(e, ':'))) { g_x_csserial_m = (uint32_t)atoi(e + 1); e = strchr(e + 1, ':'); g_x_csserial_r = e ? (uint32_t)atoi(e + 1) : 0; }
    g_x_csalt = (e = getenv("DM_VK_X_CSALT")) && *e ? atoi(e) : -1; g_x_csalt_m = 0; g_x_csalt_r = 0; if (e && (e = strchr(e, ':'))) { g_x_csalt_m = (uint32_t)atoi(e + 1); e = strchr(e + 1, ':'); g_x_csalt_r = e ? (uint32_t)atoi(e + 1) : 0; }
    g_x_nokeep = (e = getenv("DM_VK_X_NOKEEP")) && atoi(e); g_x_nobc = (e = getenv("DM_VK_X_NOBC")) && atoi(e); g_x_noshc = (e = getenv("DM_VK_X_NOSHC")) && atoi(e); g_x_cutany = (e = getenv("DM_VK_X_CUTANY")) && atoi(e); g_x_nogs = (e = getenv("DM_VK_X_NOGS")) && atoi(e); g_spv_alt = (e = getenv("DM_VK_SPV_ALT")) && atoi(e); g_rd_touch = !((e = getenv("DM_VK_RD_TOUCH")) && !atoi(e));
    g_lrz_touch = !((e = getenv("DM_VK_LRZ_TOUCH")) && !atoi(e)); g_fdz = (e = getenv("DM_VK_FDZ")) && *e ? atoi(e) : 1; g_xfb_on = !((e = getenv("DM_VK_XFB")) && !atoi(e));
    g_mesh_nofdz = (e = getenv("DM_VK_MESH_NOFDZ")) && atoi(e); g_mesh_notask = (e = getenv("DM_VK_MESH_NOTASK")) && atoi(e); g_mesh_fill = (e = getenv("DM_VK_MESH_FILL")) && atoi(e);
    g_mesh_posblock = (e = getenv("DM_VK_MESH_POSBLOCK")) && atoi(e); g_mesh_vistate = (e = getenv("DM_VK_MESH_VISTATE")) && atoi(e);
    { extern int g_ctr_early, g_ts_cal, g_ts_flat; g_ctr_early = (e = getenv("DM_VK_CTR_EARLY")) && atoi(e); g_ts_cal = (e = getenv("DM_VK_TS_CAL")) && atoi(e); g_ts_flat = (e = getenv("DM_VK_TS_FLAT")) && atoi(e); } g_mesh_live = !((e = getenv("DM_VK_MESH")) && !atoi(e)); g_alpha_on = !((e = getenv("DM_VK_ALPHA")) && !atoi(e)); g_first_store = !((e = getenv("DM_VK_FIRST_STORE")) && !atoi(e)); g_disc = (e = getenv("DM_VK_DISCARD")) ? atoi(e) : 1;
    { void vk_bar_merge_set(int on); vk_bar_merge_set(!((e = getenv("DM_VK_BAR_MERGE")) && !atoi(e))); }   // (gfx_vk_dq.c: full barriers back to back)
    g_psplit = (e = getenv("DM_VK_PASS_SPLIT")) ? atoi(e) : 0; g_setsplit = (e = getenv("DM_VK_SETSPLIT")) && atoi(e); g_psplit_depth = (e = getenv("DM_VK_PASS_SPLIT_DEPTH")) && atoi(e);
    g_x_rdkeep = (e = getenv("DM_VK_X_RDKEEP")) && atoi(e); g_vbound = !((e = getenv("DM_VK_VBOUND")) && !atoi(e)); g_x_logfs = (e = getenv("DM_VK_LOG_FS")) ? strtoull(e, NULL, 16) : 0; g_samp_snap = !((e = getenv("DM_NVN_SAMP_SNAP")) && !atoi(e)); g_lin_back = !((e = getenv("DM_VK_LIN_BACK")) && !atoi(e)); g_async_pipe = !((e = getenv("DM_VK_ASYNC_PIPE")) && !atoi(e)); g_prof_on = (e = getenv("DM_VK_PROF_SLOW")) && atoi(e); { extern volatile int g_vk_samp_log; if ((e = getenv("DM_VK_SAMP_LOG")) && atoi(e) > 0) { g_vk_samp_log = atoi(e); setenv("DM_VK_SAMP_LOG", "0", 1); } } g_x_nosetmemo = (e = getenv("DM_VK_X_NOSETMEMO")) && atoi(e); g_cbc_on = !((e = getenv("DM_VK_CBUF_CACHE")) && !atoi(e)); g_tvc_keep_env = !((e = getenv("DM_VK_TVC_KEEP")) && !atoi(e)); g_keep_on = !((e = getenv("DM_VK_SET_KEEP")) && !atoi(e)); g_keep_fine = !((e = getenv("DM_VK_KEEP_FINE")) && !atoi(e)); g_x_setverify = (e = getenv("DM_VK_X_SETVERIFY")) && atoi(e);
    g_x_nskipcs = 0; for (const char *q = getenv("DM_VK_X_SKIPCS"); q && *q && g_x_nskipcs < 8;) { char *n; uint64_t v = strtoull(q, &n, 16); if (n == q) break; if (v) g_x_skipcs[g_x_nskipcs++] = v; q = *n ? n + 1 : n; }
    g_x_mdiall = (e = getenv("DM_VK_X_MDI_ALL")) && atoi(e); { extern int g_x_subidle; g_x_subidle = (e = getenv("DM_VK_X_SUBIDLE")) && atoi(e); }
    { const char *q = getenv("DM_VK_X_SKIPVS_MOD"); uint64_t m = 0, r = 0; if (q && *q) { m = strtoull(q, NULL, 10); const char *c = strchr(q, ':'); r = c ? strtoull(c + 1, NULL, 10) : 0; }
      if (m != g_x_skipvs_m || r != g_x_skipvs_r) { g_x_skipvs_m = m; g_x_skipvs_r = r; g_x_skipvs_gen++; } }
    g_x_nskipvs = 0; for (const char *q = getenv("DM_VK_X_SKIPVS"); q && *q && g_x_nskipvs < 8;) { char *n; uint64_t v = strtoull(q, &n, 16); if (n == q) break; if (v) g_x_skipvs[g_x_nskipvs++] = v; q = *n ? n + 1 : n; } }
// ---------------- conditional rendering (profile render_cond) ----------------
// As nvn_mtl.m (OP_RENDER_COND): SetRenderEnableConditional draws while the 64-bit counters of the reports at addr and addr + 16 are equal / differ, until SetRenderEnable(true).
// Evaluated on the CPU at replay from the records in memory: a report replayed earlier in this submission is not written yet (that happens when the GPU completes), so a timestamp
// one counts with its known counter 0 (vk_sub_complete writes 0 there under the key) and any other one is unknown, which keeps drawing. Inside a block evaluated false, draws and
// clears are dropped. Mario vs. Donkey Kong (gsys, as Tears of the Kingdom and Animal Crossing) clears its HDR scene target under such a condition after lighting it: drawn
// anyway, its levels were black under the HUD
static int g_rc_skip;   // a render-enable block evaluated false is active
#define RC_PEND 256
static struct { uint64_t addr; int kind; } g_rc_pend[RC_PEND]; static int g_rc_npend;   // reports replayed in this submission: kind 1 = timestamp (counter 0), 2 = another counter (unknown)
static uint64_t g_rc_occ[4096];   // addresses ever written by a non-timestamp report (open addressing): conditions on them keep drawing
static void rc_occ_add(uint64_t a) { for (uint32_t i = (uint32_t)(a >> 4) & 4095, k = 0; k < 4096; k++, i = (i + 1) & 4095) { if (g_rc_occ[i] == a) return; if (!g_rc_occ[i]) { g_rc_occ[i] = a; return; } } }
static int rc_occ_has(uint64_t a) { for (uint32_t i = (uint32_t)(a >> 4) & 4095, k = 0; k < 4096; k++, i = (i + 1) & 4095) { if (g_rc_occ[i] == a) return 1; if (!g_rc_occ[i]) return 0; } return 0; }
static void rc_note(uint64_t a, int kind) {
    if (kind == 2) rc_occ_add(a);
    for (int i = 0; i < g_rc_npend; i++) if (g_rc_pend[i].addr == a) { g_rc_pend[i].kind = kind; return; }
    if (g_rc_npend < RC_PEND) { g_rc_pend[g_rc_npend].addr = a; g_rc_pend[g_rc_npend++].kind = kind; }
}
static int rc_value(uint64_t a, uint64_t *v) {   // 0 = unknown at replay time
    if (rc_occ_has(a)) return 0;
    for (int i = g_rc_npend - 1; i >= 0; i--) if (g_rc_pend[i].addr == a) { if (g_rc_pend[i].kind != 1) return 0; *v = 0; return 1; }
    *v = *(const volatile uint64_t *)(uintptr_t)a; return 1;
}
static void render_cond(const p_render_cond *rc) {
    if (rc->mode >= 2) { g_rc_skip = rc->mode == 3; return; }
    uint64_t x = 0, y = 0; int known = rc_value(rc->addr, &x) & rc_value(rc->addr + 16, &y);
    g_rc_skip = known && (rc->mode ? x == y : x != y);
}

static void pu_reset(void);   // (GPU profile units, below)
void vk_exec_reset(void) {   // a new submission: NVN state does not carry over between command buffers, the replay starts from the defaults
    static int inited; if (!inited) { inited = 1; state_reset(); crash_add_note_hook(build_dump); prewarm_start(); compute_warm_start(); }
    g_rc_npend = 0; g_rc_skip = 0; pu_reset();
    S.pass_open = 0; S.bound_pipe = VK_NULL_HANDLE; g_cb_draws = 0; g_ord_parts = 0; g_psplit_now = 0; g_set_epoch++; x_read();
    if (g_prof_on) { extern uint64_t g_pkg_ns, g_pkg_n; memset(g_prof_ns, 0, sizeof g_prof_ns); g_prof_t0 = tick(); g_prof_pkg0 = g_pkg_ns; g_prof_npkg0 = g_pkg_n; g_prof_pipes0 = g_vkst.pipes_built; }
    { extern volatile int g_vk_samp_log; if (g_vk_samp_log > 0) VKLOG("submission begins (present %llu, thread %d)", (unsigned long long)g_tex_frame, (int)gettid()); }
}

// ---------------- render passes ----------------
// A pass is begun lazily: begin_pass() only sets it up (pass_open 2), and the first command that must run inside it begins it (pass_commit, pass_open 1).
// Until then a clear of a whole attachment becomes that attachment's load op, so a tiler clears in tile memory instead of loading the
// attachment from memory and clearing it with a draw; a pass that is set up and never used is dropped.
static VkRenderingAttachmentInfo P_ca[8], P_da, P_sa; static VkRenderingInfo P_ri; static int P_cleared;
// Render passes. Adreno's driver spent about 40 µs in each vkCmdBeginRendering (a quarter of the replay thread in Xenoblade Chronicles 3's battles, 128 passes a
// frame); a VkRenderPass is made once per set of formats and load operations, and an imageless framebuffer once per set of attachment descriptions, and passes
// then begin with vkCmdBeginRenderPass (the battle's replay went from about 20 to 17.5 ms a frame). Pipelines are made against a compatible render pass (the same
// formats, every load operation LOAD). DM_VK_RP=0 (start-up): dynamic rendering
typedef struct { uint32_t ncol, cfmt[8], cload[8], cstore[8], dfmt, dload, sload, dstore, sstore, samples; } rpkey;   // (store ops: the discards, dsc_commit; 0 is STORE)
typedef struct { VkRenderPass rp; uint32_t w, h, layers, n; struct { uint32_t usage, flags, w, h, layers, fmt, fmt2; } a[9]; } fbkey;   // fmt2: a format-listed image's second format (an imageless framebuffer names the image's whole list)
static nvn_hmap g_rps = {.klen = sizeof(rpkey)}, g_fbs = {.klen = sizeof(fbkey)}; static pthread_mutex_t g_rp_mu = PTHREAD_MUTEX_INITIALIZER;
// Without dynamic rendering (a Vulkan 1.1 / 1.2 device, V.has_dynrender) passes must be render pass objects, whatever DM_VK_RP says (vk_init refuses a device with neither)
static int rp_on(void) { static int on = -1; if (on < 0) { const char *e = getenv("DM_VK_RP"); on = V.has_imageless && (!V.has_dynrender || !(e && !atoi(e))); VKLOG("passes: %s", on ? "render pass objects with imageless framebuffers" : "dynamic rendering"); } return on; }
static int fmt_has_depth(uint32_t f) { return f && f != VK_FORMAT_S8_UINT; }
static int fmt_has_stencil(uint32_t f) { return f == VK_FORMAT_D24_UNORM_S8_UINT || f == VK_FORMAT_D32_SFLOAT_S8_UINT || f == VK_FORMAT_S8_UINT || f == VK_FORMAT_D16_UNORM_S8_UINT; }
static VkRenderPass rp_get(const rpkey *k) {
    pthread_mutex_lock(&g_rp_mu); VkRenderPass rp = nvn_hm_get(&g_rps, k);
    if (!rp) {
        VkAttachmentDescription2 ad[9]; VkAttachmentReference2 cr[8], dr; uint32_t na = 0;
        for (uint32_t i = 0; i < k->ncol; i++) {
            if (!k->cfmt[i]) { cr[i] = (VkAttachmentReference2){VK_STRUCTURE_TYPE_ATTACHMENT_REFERENCE_2, NULL, VK_ATTACHMENT_UNUSED, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_ASPECT_COLOR_BIT}; continue; }
            ad[na] = (VkAttachmentDescription2){VK_STRUCTURE_TYPE_ATTACHMENT_DESCRIPTION_2, NULL, 0, (VkFormat)k->cfmt[i], (VkSampleCountFlagBits)k->samples, (VkAttachmentLoadOp)k->cload[i], (VkAttachmentStoreOp)k->cstore[i],
                VK_ATTACHMENT_LOAD_OP_DONT_CARE, VK_ATTACHMENT_STORE_OP_DONT_CARE, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL};
            cr[i] = (VkAttachmentReference2){VK_STRUCTURE_TYPE_ATTACHMENT_REFERENCE_2, NULL, na++, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_ASPECT_COLOR_BIT};
        }
        if (k->dfmt) {
            int hd = fmt_has_depth(k->dfmt), hs = fmt_has_stencil(k->dfmt);
            ad[na] = (VkAttachmentDescription2){VK_STRUCTURE_TYPE_ATTACHMENT_DESCRIPTION_2, NULL, 0, (VkFormat)k->dfmt, (VkSampleCountFlagBits)k->samples,
                hd ? (VkAttachmentLoadOp)k->dload : VK_ATTACHMENT_LOAD_OP_DONT_CARE, hd ? (VkAttachmentStoreOp)k->dstore : VK_ATTACHMENT_STORE_OP_DONT_CARE,
                hs ? (VkAttachmentLoadOp)k->sload : VK_ATTACHMENT_LOAD_OP_DONT_CARE, hs ? (VkAttachmentStoreOp)k->sstore : VK_ATTACHMENT_STORE_OP_DONT_CARE, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL};
            dr = (VkAttachmentReference2){VK_STRUCTURE_TYPE_ATTACHMENT_REFERENCE_2, NULL, na++, VK_IMAGE_LAYOUT_GENERAL, (hd ? VK_IMAGE_ASPECT_DEPTH_BIT : 0) | (hs ? VK_IMAGE_ASPECT_STENCIL_BIT : 0)};
        }
        VkSubpassDescription2 sd = {VK_STRUCTURE_TYPE_SUBPASS_DESCRIPTION_2, NULL, 0, VK_PIPELINE_BIND_POINT_GRAPHICS, 0, 0, NULL, k->ncol, cr, NULL, k->dfmt ? &dr : NULL, 0, NULL};
        VkRenderPassCreateInfo2 ci = {VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO_2, NULL, 0, na, ad, 1, &sd, 0, NULL, 0, NULL};
        if (vkCreateRenderPass2(V.dev, &ci, NULL, &rp)) { VKRED_ONCE(70, "vkCreateRenderPass2 failed"); rp = VK_NULL_HANDLE; } else nvn_hm_put(&g_rps, k, rp);
    }
    pthread_mutex_unlock(&g_rp_mu); return rp;
}
static VkRenderPass rp_compat(uint32_t ncol, const uint32_t *cfmt, uint32_t dfmt, uint32_t samples) {   // for pipelines: every load operation LOAD
    rpkey k; memset(&k, 0, sizeof k); k.ncol = ncol; for (uint32_t i = 0; i < ncol && i < 8; i++) { k.cfmt[i] = cfmt[i]; k.cload[i] = VK_ATTACHMENT_LOAD_OP_LOAD; }
    k.dfmt = dfmt; k.dload = k.sload = VK_ATTACHMENT_LOAD_OP_LOAD; k.dstore = VK_ATTACHMENT_STORE_OP_STORE; k.samples = samples ? samples : 1; return rp_get(&k);
}
static VkFramebuffer fb_get(const fbkey *k) {
    pthread_mutex_lock(&g_rp_mu); VkFramebuffer fb = nvn_hm_get(&g_fbs, k);
    if (!fb) {
        VkFramebufferAttachmentImageInfo ai[9]; VkFormat vf[9][2];
        for (uint32_t i = 0; i < k->n; i++) { vf[i][0] = (VkFormat)k->a[i].fmt; vf[i][1] = (VkFormat)k->a[i].fmt2;
            ai[i] = (VkFramebufferAttachmentImageInfo){VK_STRUCTURE_TYPE_FRAMEBUFFER_ATTACHMENT_IMAGE_INFO, NULL, k->a[i].flags, k->a[i].usage, k->a[i].w, k->a[i].h, k->a[i].layers, k->a[i].fmt2 ? 2u : 1u, vf[i]}; }
        VkFramebufferAttachmentsCreateInfo aci = {VK_STRUCTURE_TYPE_FRAMEBUFFER_ATTACHMENTS_CREATE_INFO, NULL, k->n, ai};
        VkFramebufferCreateInfo ci = {VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO, &aci, VK_FRAMEBUFFER_CREATE_IMAGELESS_BIT, k->rp, k->n, NULL, k->w, k->h, k->layers};
        if (vkCreateFramebuffer(V.dev, &ci, NULL, &fb)) { VKRED_ONCE(71, "vkCreateFramebuffer failed"); fb = VK_NULL_HANDLE; } else nvn_hm_put(&g_fbs, k, fb);
    }
    pthread_mutex_unlock(&g_rp_mu); return fb;
}
// the attachments begin_pass found, for the render pass begin: each colour slot's (and the depth's) image, view, view format and layers, size at the view's level
static struct { vkimg_t *im; VkImageView v; uint32_t fmt, layers, w, h; } P_att[9];

// GPU profile units (gfx_vk_core.c "GPU profile", DM_VK_GPUPROF): what the unit being recorded holds, handed over where it ends (a stretch between passes where a pass begins, pass_commit;
// a pass at its end, vk_end_pass; the last stretch at the submission's end, vk_prof_end_sub). Its barrier and uniform-landing counts are the counters' moves since it began
static vk_prof_t g_pu; static uint64_t g_pu_bar0, g_pu_merged0, g_pu_ubd0; static int g_cur_op = -1;   // g_cur_op: the record op being replayed (exec_ops), for a pass's end
extern uint64_t g_ubd_landed;
static void pu_end(int kind) {
    g_pu.kind = (uint8_t)kind; g_pu.bars = (uint16_t)(g_bar_full - g_pu_bar0); g_pu.merged = (uint16_t)(g_bar_merged - g_pu_merged0); g_pu.ubd = (uint16_t)(g_ubd_landed - g_pu_ubd0);
    vk_prof_unit(&g_pu);
    memset(&g_pu, 0, sizeof g_pu); g_pu_bar0 = g_bar_full; g_pu_merged0 = g_bar_merged; g_pu_ubd0 = g_ubd_landed;
}
static void pu_reset(void) { memset(&g_pu, 0, sizeof g_pu); g_pu_bar0 = g_bar_full; g_pu_merged0 = g_bar_merged; g_pu_ubd0 = g_ubd_landed; }   // (a submission's start)
static uint16_t pu_add(uint16_t a, uint32_t n) { return (uint16_t)(a + n > 65535 ? 65535 : a + n); }
// Before a pass that goes on with a depth buffer (loads it), one texel of its depth is copied out: a transfer read makes Adreno's driver drop the buffer's
// low-resolution depth (LRZ), which it otherwise kept from earlier passes even within one command buffer. NEO's menu → field wipe drew whole 32x32 squares of
// its old frame brown for two frames (the overlay's tiles rejected); with the read, never. Copying the colour target out instead, or a barrier alone, did not help.
// Depth-only buffers too: 13 Sentinels: Aegis Rim draws its frame (206 sprites, depth test off) and then copies it to the window in one quad, both passes on one
// D32; the quad's second triangle came out as a checkerboard of stale bins in every frame (either diagonal; a command-buffer cut or this read cleared it, the
// depth attachment not loaded did not). write: the texel is also written back (unchanged), a transfer write. DM_VK_LRZ_TOUCH=0 (a DM_CTL "env" line) leaves both touches out
static void lrz_touch(vkimg_t *im, int write) {
    static VkBuffer b; static VkDeviceMemory m;
    if (!im || !(im->aspect & VK_IMAGE_ASPECT_DEPTH_BIT) || im->samples > 1) return;   // (multisampled images cannot be copied to a buffer)
    g_pu.touches = pu_add(g_pu.touches, 1);   // (DM_VK_GPUPROF)
    if (!b) { VkBufferCreateInfo bi = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, NULL, 0, 256, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_SHARING_MODE_EXCLUSIVE};
        VKCHK(vkCreateBuffer(V.dev, &bi, NULL, &b)); VkMemoryRequirements rq; vkGetBufferMemoryRequirements(V.dev, b, &rq);
        VkMemoryAllocateInfo ai = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, NULL, rq.size, vk_mem_type(rq.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 0)};
        VKCHK(vkAllocateMemory(V.dev, &ai, NULL, &m)); VKCHK(vkBindBufferMemory(V.dev, b, m, 0)); }
    VkCommandBuffer cb = vk_cmd();
    VkBufferImageCopy c = {0, 0, 0, {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 0, 1}, {0, 0, 0}, {1, 1, 1}};
    vkCmdCopyImageToBuffer(cb, im->img, VK_IMAGE_LAYOUT_GENERAL, b, 1, &c); vk_img_barrier_all(cb);   // (begin_pass's barrier already ordered it after the writes)
    if (write) { vkCmdCopyBufferToImage(cb, b, im->img, VK_IMAGE_LAYOUT_GENERAL, 1, &c); vk_img_barrier_all(cb); }
}
// The depth image of the last pass that had one. When a pass draws with another, that one gets the texel written back: Adreno otherwise applied its LRZ in later
// passes on other depth buffers and rejected whole triangles there. Xenoblade Chronicles 3's cloud layer (one 720-index dome drawn into a 200x112 target, its D32
// depth downsampled from the scene's 776x440 D24S8 by a shader writing gl_FragDepth) lost whole triangles while the camera turned and kept losing them with the
// camera still at some angles: a primitive-ID dump showed whole triangles missing, triangles reaching past the screen edge kept, the same with the dome's depth
// forced to a constant; the dome drawn with gl_FragDepth written (no LRZ for it) lost none in 47 frames. Reading the dome's depth buffer before its pass (the
// read above), or writing it back, did not help; writing back every depth buffer before every pass that loads one did (0 broken frames of 53 against 17 of 46,
// GPU +1 ms), as did writing back only the one a pass switches away from (0 genuine drops in about 140 frames, GPU +0.6-0.8 ms of 13-18)
static vkimg_t *g_lrz_prev;
void vk_lrz_forget(vkimg_t *im) { if (g_lrz_prev == im) g_lrz_prev = NULL; }   // an image dropped (gfx_vk_tex.c img_drop)
const char *vk_prof_op_name(int op) {   // the record op that ended a pass, for the profile's lines
    switch (op) { case VK_PU_END_SUB: return "the submission's end"; case OP_SET_RT: return "SetRenderTargets"; case OP_DISPATCH: return "Dispatch"; case OP_BARRIER: return "Barrier";
                  case OP_COPY_B2B: return "CopyBufferToBuffer"; case OP_COPY_B2T: return "CopyBufferToTexture"; case OP_COPY_T2B: return "CopyTextureToBuffer"; case OP_COPY_T2T: return "CopyTextureToTexture";
                  case OP_CLEAR_TEX: return "ClearTexture"; case OP_CLEAR_BUF: return "ClearBuffer"; case OP_DRAW: case OP_DRAW_IDX: case OP_MULTI_DRAW: return "a draw (rasterizer discard or a split)";
                  case OP_UPDATE_UB: return "UpdateUniformBuffer"; case OP_REPORT_COUNTER: return "ReportCounter"; case OP_DRAW_TEX: return "DrawTexture"; case OP_RENDER_COND: return "RenderEnable";
                  case OP_CLEAR_COLOR: return "ClearColor"; case OP_CLEAR_DS: return "ClearDepthStencil"; case OP_DLSS: return "DLSS";
                  default: { static char b[8][16]; static int k; char *s = b[k++ & 7]; snprintf(s, 16, "op %d", op); return s; } }
}
void vk_end_pass_land(void);
void vk_prof_end_sub(void) {   // the submission's end (mtl_sub_commit): the pass still open ends there, then the last stretch
    g_cur_op = VK_PU_END_SUB; vk_end_pass_land(); if (vk_prof_on()) pu_end(VK_PU_GAP);
}
// ---------------- discards (NVN DiscardColor / DiscardDepthStencil: OP_DISCARD; DM_VK_DISCARD, live) ----------------
// On a tiler a pass's store of an attachment resolves tile memory into memory and its load fills tile memory from it: on Adreno the store of an attachment the game discards right after
// costs GMEM resolve bandwidth for nothing, and so does the load of one it discarded before. NVN's discards say a bound target's contents are undefined from there. One before a pass's
// first draw makes that attachment's load DONT_CARE (disc_op, begin_pass; a clear folded into the load stays a clear). One after its last draw makes its store DONT_CARE, but a pass's
// store op is fixed where it begins (its render pass), so that side looks ahead from there (dsc_commit): dsc_scan finds the discards after the pass's last draw and before the record
// that ends it for certain, where a record that may cut the pass on the replay's own terms first (a rasterizer-discard switch, transform feedback's end, the deferred uniform updates'
// list filling up) keeps the store: a cut pass's discard goes to the part that really ends there, whose own scan finds it. A discarded attachment whose storage the rest of the
// submission uses again (a target, a copy or clear, a texture or image bind: the walk dsc_memo_build makes once a submission) keeps its store. The replay cannot see past its
// submission nor reads it cannot name from records (handles in uniform buffers): there the image's own history decides (vkimg_t dsc_*: dsc_hist_*): a store is left out only after
// DSC_EVIDENCE discards of that aspect each followed by a whole write (a clear, or a discard as the next pass's load) before any read, and never after one read while discarded
// (vk_dsc_read_: [RED] when its store had been left out, and vk_end_pass checks that the pass ended where the scan said). Depth and stencil are decided apart.
// DM_VK_DISCARD (live): 1 (the default) depth and stencil, 2 colour too, 0 none. Colour is not the default: a game that draws over part of a colour target it discarded and shows the
// rest would show tile garbage where the console kept the old pixels (its discard is a hint there); depth and stencil are never shown, and they are what NVN games discard when done
// ---- discards: the lookahead (tests: tests/test_vk_discard.py) ----
enum { VK_DSC_COLOR = 1, VK_DSC_DEPTH = 2, VK_DSC_STENCIL = 4 };   // an image's aspects in its history (vkimg_t dsc_*)
enum { DSC_D = 1u << 8, DSC_S = 1u << 9, DSC_EVIDENCE = 3 };       // attachment bits: colour target i at bit i, the depth, the stencil; discards followed by a whole write before a store goes
enum { DSC_SURE, DSC_UNSURE_OP, DSC_UNSURE_UBD, DSC_UNSURE_LONG };  // dsc_scan's why: the end is certain; a record that may cut the pass; uniform updates past the list's room; too long or lost
typedef struct { const uint8_t *p, *ret[4]; int nret; const uint64_t *rest; int nrest; } dsc_cur;   // a place in the submission's records: the record there, calls to return from, command handles left
typedef struct {
    int (*same_rt)(const p_set_rt *r);                                  // a SetRenderTargets that keeps the pass's targets (set_rt's rt_same)
    int (*handle_range)(uint64_t h, uintptr_t *lo, uintptr_t *hi);      // the storage of a bound texture or image handle (0: none)
    int ubd_room, max_ops;                                              // uniform updates the deferred list takes before it lands them (a pass's end); records looked at, at most
} dsc_ctx;
static int dsc_tex_range(const tex_t *t, uintptr_t *lo, uintptr_t *hi) {   // a texture object's storage (0: none known)
    if (!t || !t->mem) return 0;
    *lo = (uintptr_t)t->mem; *hi = *lo + (t->L.size ? (uintptr_t)t->L.size : 1); return 1;
}
static int dsc_next(dsc_cur *c, op_hdr *h, const uint8_t **a) {   // the next record, jumps, calls and stream ends followed as exec_ops follows them: 1, 0 at the submission's end, -1 lost
    for (int guard = 0; guard < 4096; guard++) {
        if (!c->p) return 0;
        memcpy(h, c->p, sizeof *h); *a = c->p + 8;
        if (h->op == OP_JUMP) { memcpy(&c->p, *a, 8); continue; }
        if (h->op == OP_CALL) { uint64_t t; memcpy(&t, *a, 8); if (c->nret == 4) return -1; c->ret[c->nret++] = c->p + h->size; c->p = (const uint8_t *)(uintptr_t)t; continue; }
        if (h->op == OP_END) { if (c->nret) { c->p = c->ret[--c->nret]; continue; } if (c->nrest > 0) { c->p = (const uint8_t *)(uintptr_t)*c->rest++; c->nrest--; continue; } c->p = NULL; return 0; }
        if (h->size < 8) return -1;
        c->p += h->size; return 1;
    }
    return -1;
}
// From the record at c (the one that begins the pass) on: the attachments discarded after the pass's last draw and before the record that ends it for certain, *end that record (NULL:
// the submission's end) and *at the place just before it. *why tells whether that end is certain; when it is not, the bits are still the ones found up to it (for the counts), if
// it comes within DSC_PAST records of the first record that made it uncertain (a pass the replay cuts often is scanned again from each part: that bounds the cost)
enum { DSC_PAST = 4096 };
static uint32_t dsc_scan(dsc_cur c, const dsc_ctx *x, const uint8_t **end, dsc_cur *at, int *why) {
    uint32_t bits = 0; int n = 0, ubd = 0, past = DSC_PAST; op_hdr h; const uint8_t *a = NULL; *end = NULL; *why = DSC_SURE; *at = c;
    for (;;) {
        dsc_cur here = c; int r = dsc_next(&c, &h, &a);
        if (r < 0 || ++n > x->max_ops || (*why != DSC_SURE && --past < 0)) { if (*why == DSC_SURE) *why = DSC_UNSURE_LONG; return 0; }
        if (!r) { *at = c; return bits; }   // the submission's end: the pass ends there
        switch (h.op) {
        case OP_DRAW: case OP_DRAW_IDX: case OP_MULTI_DRAW: case OP_DRAW_TEX: case OP_DRAW_TF: bits = 0; continue;   // drawn again (and read: blending, the depth test)
        case OP_CLEAR_COLOR: { p_clear_color k; memcpy(&k, a, sizeof k); if (k.index >= 0 && k.index < 8) bits &= ~(1u << k.index); continue; }
        case OP_CLEAR_DS: { p_clear_ds k; memcpy(&k, a, sizeof k); if (k.depth_mask) bits &= ~DSC_D; if (k.stencil_mask) bits &= ~DSC_S; continue; }
        case OP_DISCARD: { p_discard k; memcpy(&k, a, sizeof k); bits |= k.index < 0 ? DSC_D | DSC_S : k.index < 8 ? 1u << k.index : 0; continue; }
        case OP_SET_RT: if (x->same_rt((const p_set_rt *)a)) continue; break;   // other targets: the end
        case OP_DISPATCH: case OP_COPY_B2B: case OP_COPY_B2T: case OP_COPY_T2B: case OP_COPY_T2T: case OP_CLEAR_TEX: case OP_CLEAR_BUF: case OP_BARRIER: break;   // (each ends the pass first)
        case OP_UPDATE_UB: if (++ubd > x->ubd_room && *why == DSC_SURE) *why = DSC_UNSURE_UBD; continue;   // (vk_sub_update lands a full list: an update supersedes at most one)
        case OP_RAST_DISCARD: case OP_TF_STATE: case OP_DLSS: if (*why == DSC_SURE) *why = DSC_UNSURE_OP; continue;
        default: continue;   // state, binds, counters, events
        }
        *end = a - 8; *at = here; return bits;
    }
}
// The rest of the submission's uses of storage, from the record that ended the first pass needing them (dsc_memo_build, once a submission): each range's last use (the walk's ordinal),
// and the ordinal of every record that may end a pass. A pass ended by record e leaves its attachment's storage unused when no range over it is used at e or later
enum { DSC_NUSE = 4096, DSC_NEND = 8192 };
typedef struct { uintptr_t lo, hi; uint32_t last, gen; } dsc_use;
typedef struct { const uint8_t *rec; uint32_t ord, gen; } dsc_endrec;   // ord 0: walked twice (a record replayed twice in the submission: no answer for it)
typedef struct { uint32_t gen; int nuse, nend, full; dsc_use use[DSC_NUSE]; dsc_endrec end[DSC_NEND]; } dsc_memo;
static uint32_t dsc_hash(uintptr_t a, uintptr_t b) { uint64_t k = (uint64_t)a * 0x9e3779b97f4a7c15ull ^ (uint64_t)b * 0xc2b2ae3d27d4eb4full; return (uint32_t)(k >> 40); }
static void dsc_memo_use(dsc_memo *m, uintptr_t lo, uintptr_t hi, uint32_t ord) {
    if (lo >= hi || m->full) return;
    for (uint32_t i = dsc_hash(lo, hi) & (DSC_NUSE - 1), k = 0; k < DSC_NUSE; k++, i = (i + 1) & (DSC_NUSE - 1)) { dsc_use *u = &m->use[i];
        if (u->gen != m->gen) { if (m->nuse >= DSC_NUSE * 3 / 4) break; *u = (dsc_use){lo, hi, ord, m->gen}; m->nuse++; return; }
        if (u->lo == lo && u->hi == hi) { u->last = ord; return; } }
    m->full = 1;
}
static void dsc_memo_end(dsc_memo *m, const uint8_t *rec, uint32_t ord) {
    if (m->full) return;
    for (uint32_t i = dsc_hash((uintptr_t)rec, 0) & (DSC_NEND - 1), k = 0; k < DSC_NEND; k++, i = (i + 1) & (DSC_NEND - 1)) { dsc_endrec *e = &m->end[i];
        if (e->gen != m->gen) { if (m->nend >= DSC_NEND * 3 / 4) break; *e = (dsc_endrec){rec, ord, m->gen}; m->nend++; return; }
        if (e->rec == rec) { e->ord = 0; return; } }
    m->full = 1;
}
static void dsc_memo_tex(dsc_memo *m, const tex_t *t, uint32_t ord) { uintptr_t lo, hi; if (dsc_tex_range(t, &lo, &hi)) dsc_memo_use(m, lo, hi, ord); }
static void dsc_memo_build(dsc_memo *m, dsc_cur c, const dsc_ctx *x) {   // from the place c (just before a pass's end record) to the submission's end
    if (!++m->gen) { memset(m, 0, sizeof *m); m->gen = 1; }
    m->nuse = m->nend = m->full = 0;
    op_hdr h; const uint8_t *a; uintptr_t lo, hi;
    for (uint32_t ord = 1;; ord++) {
        int r = dsc_next(&c, &h, &a);
        if (r < 0 || ord > (uint32_t)x->max_ops) { m->full = 1; return; }   // (not walked to the end: no answers)
        if (!r) return;
        switch (h.op) {
        case OP_SET_RT: { const p_set_rt *s = (const p_set_rt *)a; for (int i = 0; i < s->n && i < 8; i++) dsc_memo_tex(m, s->color[i], ord); dsc_memo_tex(m, s->depth, ord); break; }
        case OP_COPY_B2T: case OP_COPY_T2B: dsc_memo_tex(m, ((const p_copy_b2t *)a)->tex, ord); break;
        case OP_COPY_T2T: dsc_memo_tex(m, ((const p_copy_t2t *)a)->src, ord); dsc_memo_tex(m, ((const p_copy_t2t *)a)->dst, ord); break;
        case OP_CLEAR_TEX: dsc_memo_tex(m, ((const p_clear_tex *)a)->tex, ord); break;
        case OP_DLSS: for (int i = 0; i < DLSS_TEXTURES; i++) dsc_memo_tex(m, ((const p_dlss *)a)->tex[i], ord); break;
        case OP_COPY_B2B: { uint64_t s, d, n; memcpy(&s, a, 8); memcpy(&d, a + 8, 8); memcpy(&n, a + 16, 8); dsc_memo_use(m, (uintptr_t)s, (uintptr_t)(s + n), ord); dsc_memo_use(m, (uintptr_t)d, (uintptr_t)(d + n), ord); break; }
        case OP_CLEAR_BUF: { const p_clear_buf *k = (const p_clear_buf *)a; dsc_memo_use(m, (uintptr_t)k->addr, (uintptr_t)(k->addr + k->size), ord); break; }
        case OP_BIND_TEX: case OP_BIND_TEX1: case OP_BIND_IMG: { p_bind_handles t; const p_bind_handles *b = bind_rec(h.op, a, &t);
            for (int i = 0; i < b->n && i < 32; i++) if (b->h[i] && x->handle_range(b->h[i], &lo, &hi)) dsc_memo_use(m, lo, hi, ord); break; }
        case OP_DRAW_TEX: if (x->handle_range(((const p_draw_tex *)a)->h, &lo, &hi)) dsc_memo_use(m, lo, hi, ord); break;
        default: break;
        }
        switch (h.op) { case OP_SET_RT: case OP_DISPATCH: case OP_COPY_B2B: case OP_COPY_B2T: case OP_COPY_T2B: case OP_COPY_T2T: case OP_CLEAR_TEX: case OP_CLEAR_BUF: case OP_BARRIER: case OP_DLSS:
                        dsc_memo_end(m, a - 8, ord); break; default: break; }
    }
}
static int dsc_memo_used(const dsc_memo *m, const uint8_t *e, uintptr_t lo, uintptr_t hi) {   // [lo, hi) used again from the pass's end record e on (1 also when the walk has no answer)
    if (m->full) return 1;
    uint32_t oe = 0;
    for (uint32_t i = dsc_hash((uintptr_t)e, 0) & (DSC_NEND - 1), k = 0; k < DSC_NEND; k++, i = (i + 1) & (DSC_NEND - 1)) { const dsc_endrec *r = &m->end[i];
        if (r->gen != m->gen) break; if (r->rec == e) { oe = r->ord; break; } }
    if (!oe) return 1;
    for (int i = 0; i < DSC_NUSE; i++) { const dsc_use *u = &m->use[i]; if (u->gen == m->gen && u->lo < hi && lo < u->hi && u->last >= oe) return 1; }
    return 0;
}
// An image's discard history, by aspect a (one VK_DSC_* bit)
static int dsc_ix(int a) { return a == VK_DSC_COLOR ? 0 : a == VK_DSC_DEPTH ? 1 : 2; }
static int dsc_hist_ok(const vkimg_t *im, int a) { return !(im->dsc_veto & a) && im->dsc_ok[dsc_ix(a)] >= DSC_EVIDENCE; }   // its store may be left out
static void dsc_hist_end(vkimg_t *im, int a, int dropped) { im->dsc_pend |= (uint8_t)a; if (dropped) im->dsc_drop |= (uint8_t)a; else im->dsc_drop &= (uint8_t)~a; }   // discarded at a pass's end
// its use as a pass's attachment: loaded (a read), or wholly written (a clear, or DONT_CARE: discarded before the pass's first draw). 1: what a store left out was loaded
static int dsc_hist_use(vkimg_t *im, int a, int loaded) {
    if (!(im->dsc_pend & a)) return 0;
    int bad = loaded && (im->dsc_drop & a);
    if (loaded) im->dsc_veto |= (uint8_t)a; else if (im->dsc_ok[dsc_ix(a)] < 255) im->dsc_ok[dsc_ix(a)]++;
    im->dsc_pend &= (uint8_t)~a; im->dsc_drop &= (uint8_t)~a; return bad;
}
static int dsc_hist_read(vkimg_t *im) { int bad = im->dsc_drop != 0; im->dsc_veto |= im->dsc_pend; im->dsc_pend = im->dsc_drop = 0; return bad; }   // read otherwise: every discarded aspect
// ---- end of the discards' lookahead ----
static uint32_t g_disc_pend; static uint16_t P_dload;   // discards met before the targets' pass was set up (begin_pass takes them); the loads they left out in the pass set up
static const uint64_t *g_rest; static int g_nrest;   // the submission's command handles after the one being replayed (hle_nvn.c submit_now)
void mtl_exec_rest(const uint64_t *h, int n) { g_rest = h; g_nrest = n; }
static const uint8_t *g_op_cur, *g_call_ret[4]; static int g_ncall;   // the record being replayed (exec_ops; NULL between command handles), and where each call being replayed returns
static uint64_t g_skip_ops = ~0ull;   // DM_VK_SKIP_OPS (exec_ops, debugging): records the replay leaves out (~0: not read yet)
static dsc_memo g_dm; static uint64_t g_dm_epoch;   // the submission's walk (g_set_epoch: built for it)
static const uint8_t *P_dsc_end; static uint32_t P_dsc_bits, P_dsc_drop;   // the open pass: the record its scan said ends it, its attachments discarded after its last draw, and left out
static int rt_same(const p_set_rt *r);
static int dsc_same_rt(const p_set_rt *r) { return !g_x_nokeep && rt_same(r); }
static struct { uint64_t epoch, gen; int tid; uintptr_t lo, hi; } g_dsc_hc[1024];   // handles' storage this submission (dsc_handle_range)
static int dsc_handle_range(uint64_t h, uintptr_t *lo, uintptr_t *hi) {
    extern uint64_t g_tex_gen; int tid = (int)(h & 0xfffff); if (!tid) return 0;
    unsigned k = ((unsigned)tid * 2654435761u) >> 22;
    if (g_dsc_hc[k].epoch != g_set_epoch || g_dsc_hc[k].gen != g_tex_gen || g_dsc_hc[k].tid != tid) {
        const view_t *v = NULL; const tex_t *t = NULL; uintptr_t l = 0, u = 0; if (vk_tex_by_id(tid, &v, &t) && t) dsc_tex_range(t, &l, &u);
        g_dsc_hc[k].epoch = g_set_epoch; g_dsc_hc[k].gen = g_tex_gen; g_dsc_hc[k].tid = tid; g_dsc_hc[k].lo = l; g_dsc_hc[k].hi = u; }
    *lo = g_dsc_hc[k].lo; *hi = g_dsc_hc[k].hi; return *hi > *lo;
}
void vk_dsc_read_(vkimg_t *im, const char *how) {   // (gfx_vk.h vk_dsc_read: an image read other than as an attachment while its history has a discard pending)
    if (dsc_hist_read(im)) VKRED_ONCE(179, "discards: an image (%ux%u, format %d) whose store a pass left out was %s before anything wrote it: the scan missed that read, its store stays from now on", im->w, im->h, im->fmt, how);
}
static void dsc_load(uint32_t bits) {   // a pass set up and not begun: its loads of the attachments in bits DONT_CARE (a clear folded into the load stays)
    for (int i = 0; i < S.ncol && i < 8; i++) if (bits >> i & 1 && P_ca[i].imageView && P_ca[i].loadOp == VK_ATTACHMENT_LOAD_OP_LOAD) { P_ca[i].loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE; P_dload++; }
    if ((bits & DSC_D) && (S.daspect & VK_IMAGE_ASPECT_DEPTH_BIT) && P_da.loadOp == VK_ATTACHMENT_LOAD_OP_LOAD) { P_da.loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE; P_dload++; }
    if ((bits & DSC_S) && (S.daspect & VK_IMAGE_ASPECT_STENCIL_BIT) && P_sa.loadOp == VK_ATTACHMENT_LOAD_OP_LOAD) { P_sa.loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE; P_dload++; }
}
static void disc_op(const p_discard *d) {   // OP_DISCARD
    uint32_t bit = d->index < 0 ? DSC_D | DSC_S : d->index < 8 ? 1u << d->index : 0;
    if (!(g_disc & 2)) bit &= DSC_D | DSC_S;   // (colour: DM_VK_DISCARD=2)
    if (!g_disc || !bit) return;
    if (S.pass_open == 2) dsc_load(bit);   // before the pass's first draw
    else if (!S.pass_open) g_disc_pend |= bit;   // before its targets' pass is set up (begin_pass; any pass end before it drops them)
    // (an open pass: its store was looked ahead where it began, dsc_commit)
}
static int pd_dumping(void);
// The pass about to begin (pass_commit, its load ops final): its attachments' history first, then each one the game discards after the pass's last draw stores DONT_CARE when
// the rest of the submission does not use it and its history allows it
static void dsc_commit(void) {
    P_dsc_bits = P_dsc_drop = 0; P_dsc_end = NULL;
    for (int i = 0; i < S.ncol && i < 8; i++) { vkimg_t *im = P_att[i].im; if (im && im->dsc_pend && P_ca[i].imageView && dsc_hist_use(im, VK_DSC_COLOR, P_ca[i].loadOp == VK_ATTACHMENT_LOAD_OP_LOAD))
        VKRED_ONCE(180, "discards: a colour target (%ux%u, format %d) whose store a pass left out was loaded before anything wrote it: its store stays from now on", im->w, im->h, im->fmt); }
    vkimg_t *dim = S.daspect ? P_att[8].im : NULL; int bad = 0;
    if (dim && dim->dsc_pend && (S.daspect & VK_IMAGE_ASPECT_DEPTH_BIT)) bad |= dsc_hist_use(dim, VK_DSC_DEPTH, P_da.loadOp == VK_ATTACHMENT_LOAD_OP_LOAD);
    if (dim && dim->dsc_pend && (S.daspect & VK_IMAGE_ASPECT_STENCIL_BIT)) bad |= dsc_hist_use(dim, VK_DSC_STENCIL, P_sa.loadOp == VK_ATTACHMENT_LOAD_OP_LOAD);
    if (bad) VKRED_ONCE(181, "discards: a depth / stencil target (%ux%u, format %d) whose store a pass left out was loaded before anything wrote it: its store stays from now on", dim->w, dim->h, dim->fmt);
    extern int g_bis_trace;   // (not while the replay cuts passes itself or something reads every pass: a split, a bisection, records left out, traces, pass dumps)
    if (!g_disc || !g_op_cur || g_ncall > 4 || g_psplit > 0 || g_bis_trace >= 0 || g_skip_ops || g_x_tracepass >= 0 || g_x_traceframe[0] || g_x_tracesize[0] || pd_dumping()) return;
    int vk_ubd_room(void); dsc_ctx x = {dsc_same_rt, dsc_handle_range, vk_ubd_room(), 400000};
    dsc_cur c = {g_op_cur, {NULL}, g_ncall, g_rest, g_nrest}; memcpy(c.ret, g_call_ret, sizeof c.ret);
    const uint8_t *end; dsc_cur at; int why; uint32_t bits = dsc_scan(c, &x, &end, &at, &why);
    if (!(g_disc & 2)) bits &= DSC_D | DSC_S;
    if (!bits) return;
    if (why != DSC_SURE) { g_pu.dk_end = pu_add(g_pu.dk_end, (uint32_t)__builtin_popcount(bits)); return; }
    if (end && g_dm_epoch != g_set_epoch) { dsc_memo_build(&g_dm, at, &x); g_dm_epoch = g_set_epoch; }   // (the walk starts at the first such end: every later pass ends after it)
    uintptr_t lo, hi;
    for (int i = 0; i < 10; i++) { uint32_t b = i < 8 ? 1u << i : i == 8 ? DSC_D : DSC_S; if (!(bits & b)) continue;
        const tex_t *t = i < 8 ? (S.rt_has_col[i] ? &S.rt_col[i] : NULL) : (S.rt_has_dep ? &S.rt_dep : NULL); vkimg_t *im = P_att[i < 8 ? i : 8].im;
        VkRenderingAttachmentInfo *at_ = i < 8 ? &P_ca[i] : i == 8 ? &P_da : &P_sa; int a = i < 8 ? VK_DSC_COLOR : i == 8 ? VK_DSC_DEPTH : VK_DSC_STENCIL;
        if (!t || !im || !at_->imageView || (i == 8 && !(S.daspect & VK_IMAGE_ASPECT_DEPTH_BIT)) || (i == 9 && !(S.daspect & VK_IMAGE_ASPECT_STENCIL_BIT))) continue;
        if (!dsc_tex_range(t, &lo, &hi) || (end && dsc_memo_used(&g_dm, end, lo, hi)) || (i < 8 && g_lin_back && (t->d.flags & NVN_TEXTURE_FLAGS_LINEAR_RENDER_TARGET))) {   // (lin_back reads it)
            g_pu.dk_used = pu_add(g_pu.dk_used, 1); continue; }
        P_dsc_bits |= b;
        if (!dsc_hist_ok(im, a)) { g_pu.dk_hist = pu_add(g_pu.dk_hist, 1); continue; }
        at_->storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE; P_dsc_drop |= b; g_pu.dstore = pu_add(g_pu.dstore, 1); }
    if (P_dsc_bits) P_dsc_end = end;
}
static void dsc_pass_end(void) {   // the open pass ends (vk_end_pass): its discarded attachments go into their images' history, if it ended where its scan said
    if (!P_dsc_bits) return;
    int where = g_op_cur == P_dsc_end;
    if (!where && P_dsc_drop) VKRED_ONCE(182, "discards: a pass whose stores were left out ended at op %d, not where its scan said: those images keep their stores from now on", g_cur_op);
    for (int i = 0; i < 10; i++) { uint32_t b = i < 8 ? 1u << i : i == 8 ? DSC_D : DSC_S; vkimg_t *im = P_att[i < 8 ? i : 8].im; if (!(P_dsc_bits & b) || !im) continue;
        int a = i < 8 ? VK_DSC_COLOR : i == 8 ? VK_DSC_DEPTH : VK_DSC_STENCIL;
        if (where) dsc_hist_end(im, a, (P_dsc_drop & b) != 0); else if (P_dsc_drop & b) im->dsc_veto |= (uint8_t)a; }
    P_dsc_bits = P_dsc_drop = 0; P_dsc_end = NULL;
}
// 0: no pass could be begun (no render pass or framebuffer, and no dynamic rendering to fall back on): the caller records nothing that needs one
static int pass_commit(void) {
    if (S.pass_open != 2) return S.pass_open == 1;
    vkimg_t *dim = S.daspect ? P_att[8].im : NULL;
    if (g_lrz_touch && dim && g_lrz_prev && g_lrz_prev != dim) lrz_touch(g_lrz_prev, 1);
    if (dim) g_lrz_prev = dim;
    if (g_lrz_touch && dim && P_da.loadOp == VK_ATTACHMENT_LOAD_OP_LOAD) lrz_touch(dim, 0);
    if (vk_prof_on()) pu_end(VK_PU_GAP);   // (DM_VK_GPUPROF: the stretch before this pass ends here, its barrier and touches with it)
    g_pu.dload = pu_add(g_pu.dload, P_dload); P_dload = 0; dsc_commit();   // the loads its discards left out (decided while it was set up), then its stores (before its render pass is made)
    if (rp_on()) {
        rpkey k; memset(&k, 0, sizeof k); k.ncol = P_ri.colorAttachmentCount; k.samples = S.samples ? S.samples : 1;
        fbkey fk; memset(&fk, 0, sizeof fk); VkImageView views[9]; VkClearValue cv[9]; uint32_t na = 0;
        for (uint32_t i = 0; i < k.ncol; i++) if (P_ca[i].imageView) {
            k.cfmt[i] = P_att[i].fmt; k.cload[i] = P_ca[i].loadOp; k.cstore[i] = P_ca[i].storeOp; cv[na] = P_ca[i].clearValue; views[na] = P_ca[i].imageView;
            fk.a[na].usage = P_att[i].im->usage; fk.a[na].flags = P_att[i].im->cflags; fk.a[na].w = P_att[i].w; fk.a[na].h = P_att[i].h; fk.a[na].layers = P_att[i].layers;
            fk.a[na].fmt = P_att[i].im->fmt2 ? (uint32_t)P_att[i].im->fmt : P_att[i].fmt; fk.a[na].fmt2 = (uint32_t)P_att[i].im->fmt2; na++; }   // (a format-listed image (DM_VK_UBWC): its whole list, as imageless framebuffers require)
        if (P_ri.pDepthAttachment || P_ri.pStencilAttachment) {
            k.dfmt = P_att[8].fmt; k.dload = P_da.loadOp; k.sload = P_sa.imageView ? P_sa.loadOp : VK_ATTACHMENT_LOAD_OP_DONT_CARE; k.dstore = P_da.storeOp; k.sstore = P_sa.imageView ? P_sa.storeOp : VK_ATTACHMENT_STORE_OP_STORE;
            cv[na] = P_da.loadOp == VK_ATTACHMENT_LOAD_OP_CLEAR ? P_da.clearValue : P_sa.clearValue; if (P_sa.loadOp == VK_ATTACHMENT_LOAD_OP_CLEAR) cv[na].depthStencil.stencil = P_sa.clearValue.depthStencil.stencil;
            views[na] = P_da.imageView;
            fk.a[na].usage = P_att[8].im->usage; fk.a[na].flags = P_att[8].im->cflags; fk.a[na].w = P_att[8].w; fk.a[na].h = P_att[8].h; fk.a[na].layers = P_att[8].layers; fk.a[na].fmt = P_att[8].fmt; na++; }
        VkRenderPass rp = rp_get(&k);
        fk.rp = rp_compat(k.ncol, k.cfmt, k.dfmt, k.samples); fk.w = P_ri.renderArea.extent.width; fk.h = P_ri.renderArea.extent.height; fk.layers = P_ri.layerCount; fk.n = na;
        VkFramebuffer fb = rp && fk.rp ? fb_get(&fk) : VK_NULL_HANDLE;
        if (fb) { VkRenderPassAttachmentBeginInfo abi = {VK_STRUCTURE_TYPE_RENDER_PASS_ATTACHMENT_BEGIN_INFO, NULL, na, views};
            VkRenderPassBeginInfo rbi = {VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO, &abi, rp, fb, P_ri.renderArea, na, cv};
            vkCmdBeginRenderPass(vk_cmd(), &rbi, VK_SUBPASS_CONTENTS_INLINE); S.rp_pass = 1; }
        else if (V.has_dynrender) { vkCmdBeginRendering(vk_cmd(), &P_ri); S.rp_pass = 0; }
        else { VKRED_ONCE(76, "render pass or framebuffer failed without dynamic rendering: pass skipped"); return 0; }   // (still pass_open 2: vk_end_pass drops it)
    } else { vkCmdBeginRendering(vk_cmd(), &P_ri); S.rp_pass = 0; }
    S.pass_open = 1; g_vkst.passes++; g_sub_pass++;
    for (int i = 0; i < 9; i++) { vkimg_t *a = P_att[i].im; if (a && a->rt_last != g_tex_frame + 1) { a->rt_frames = a->rt_last == g_tex_frame ? a->rt_frames + 1 : 1; a->rt_last = g_tex_frame + 1; } }   // (async_pipe)
    { extern uint64_t g_cmd_gen; g_cmd_gen++; }   // the bind cache starts empty in each pass: Adreno does not keep the state a pass's draws set for the next pass (XC3 drew its lighting black)
    if (g_x_tracepass == -2) { g_x_listpasses = 1; VKLOG("--- pass %d: %ux%u, colour formats %d %d, depth format %d", g_sub_pass - 1, S.pass_w, S.pass_h, S.cfmt[0], S.cfmt[1], S.dfmt); }
    if (g_x_traceframe[0]) {   // DM_VK_TRACE_FRAME=WxH[:fmt[:fmt]] (live): trace every pass of the next submission whose targets match (a whole effect chain in one log)
        char b[48]; snprintf(b, sizeof b, "%ux%u:%d:%d", S.pass_w, S.pass_h, S.cfmt[0], S.cfmt[1]);
        g_x_tracepass = !strncmp(b, g_x_traceframe, strlen(g_x_traceframe)) ? g_sub_pass - 1 : -1; g_x_traceframe_on = 1; }
    if (g_x_tracesize[0] && g_x_tracepass < 0) {   // DM_VK_TRACE_SIZE=WxH[:format[:format]] (live): trace the next pass whose first colour target is that (passes are numbered anew each frame)
        char b[48]; snprintf(b, sizeof b, "%ux%u:%d:%d", S.pass_w, S.pass_h, S.cfmt[0], S.cfmt[1]);
        const char *h = strchr(g_x_tracesize, '#'); size_t l = h ? (size_t)(h - g_x_tracesize) : strlen(g_x_tracesize);   // #n: the (n+1)th such pass
        if (!strncmp(b, g_x_tracesize, l)) { if (h && atoi(h + 1) > 0) { snprintf(g_x_tracesize + l, sizeof g_x_tracesize - l, "#%d", atoi(h + 1) - 1); setenv("DM_VK_TRACE_SIZE", g_x_tracesize, 1); }
            else { g_x_tracepass = g_sub_pass - 1; g_x_tracesize[0] = 0; setenv("DM_VK_TRACE_SIZE", "", 1); } } }
    if (g_x_tracepass >= 0 && g_sub_pass - 1 == g_x_tracepass) VKLOG("=== traced pass %d: %ux%u, colour formats %d %d, depth format %d", g_x_tracepass, S.pass_w, S.pass_h, S.cfmt[0], S.cfmt[1], S.dfmt); g_pass_draws = 0; g_pass_bb[0] = g_pass_bb[1] = INT32_MAX; g_pass_bb[2] = g_pass_bb[3] = INT32_MIN;
    return 1;
}
void vk_ubd_land(VkCommandBuffer cb); int vk_ubd_shadow(uint64_t a, uint64_t n, VkBuffer *buf, VkDeviceSize *off); void vk_ubd_read(uint64_t a, uint64_t n, uint8_t *dst);
// DM_VK_PASSDUMP=1 (a DM_CTL "env" line; =N waits N - 1 presents first): the next submissions' passes each leave their first colour target in a buffer (up to 48 passes), written as
// DM_ROOT/pass_NN.ppm when the GPU is done (RGBA8 as is, RGBA16F clamped to 0..1, with its NaN / infinite pixel count logged)
// DM_VK_PASSDUMP_SKIP=n (read when a dump starts): leave out each submission's first n passes, to reach a frame's later passes (Xenoblade Chronicles 3's title frame has about 200);
// =WxH: start at the first pass of that size (pass numbers move by tens from frame to frame)
// DM_VK_PASSDUMP_SIZE=WxH[,WxH...] (read when a dump starts): only passes whose first colour target has one of these sizes, over every submission of the frame (no depth),
// to follow one effect's targets when the pass numbers move from frame to frame (Xenoblade Chronicles 3's 384x216 clouds)
// DM_VK_PASSDUMP_FRAMES=N (read when a dump starts): keep dumping for N presents (up to 96 dumps in all), to catch an effect that is wrong in single frames
typedef struct { VkBuffer b; VkDeviceMemory m; void *p; uint32_t w, h; VkFormat f; uint64_t fence; int pass; uint64_t present; } pdump_t;
enum { PD_MAX = 96 };
static pdump_t g_pd[PD_MAX]; static int g_npd, g_pd_state, g_pd_skip, g_pd_frames; static char g_pd_size[128], g_pd_from[32];   // state 0 off, 1 dumping this submission's passes, 2 waiting to write
static int pd_dumping(void) { return g_pd_state == 1; }   // this submission's passes are being dumped: the discards keep every store
static void pass_dump_depth(void);
static void texdump_flush(void);
static void pass_dump(void) {
    texdump_flush();
    if (g_pd_state != 1) return;   // (no dump in progress: nothing below acts, and its getenv calls cost ~3 a pass, ~350 a frame of Mario Kart World's 116 passes; vkbench)
    if (g_pd_state == 1 && g_sub_pass - 1 < g_pd_skip) return;   // (a submission with fewer passes dumps nothing, and the next one is tried)
    { const char *pp = getenv("DM_VK_PASSDUMP_PASS"); if (g_pd_state == 1 && pp && *pp) {   // DM_VK_PASSDUMP_PASS=n[,n...] (live): only these pass numbers of each submission,
        int hit = 0; for (const char *q = pp; *q;) { char *e; long v = strtol(q, &e, 10); if (e == q) break; if (v == g_sub_pass - 1) hit = 1; q = *e ? e + 1 : e; }   // to follow a few passes
        if (!hit) return; } }                                                                                                                                       // over many frames
    { const char *pf = getenv("DM_VK_PASSDUMP_FMT"); int a, b;   // DM_VK_PASSDUMP_FMT=a/b (live): only passes whose first two colour formats are VkFormat a and b (pass numbers move)
      if (g_pd_state == 1 && pf && sscanf(pf, "%d/%d", &a, &b) == 2 && ((int)S.cfmt[0] != a || (int)S.cfmt[1] != b)) return; }
    if (g_pd_state == 1 && g_pd_from[0]) { char wh[32]; snprintf(wh, sizeof wh, "%ux%u", S.pass_w, S.pass_h); if (strcmp(wh, g_pd_from)) return; g_pd_from[0] = 0; }
    if (!g_pd_size[0]) pass_dump_depth();
    int ci = 0; while (ci < S.ncol && ci < 8 && !S.rt_has_col[ci]) ci++;   // the first colour target bound (Pokémon Legends: Arceus's G-buffer leaves colour 0 unbound)
    const char *pt = getenv("DM_VK_PASSDUMP_TARGET"); if (pt && *pt && atoi(pt) >= 0 && atoi(pt) < 8 && S.rt_has_col[atoi(pt)]) ci = atoi(pt);   // =n: colour target n where the pass has one
    if (g_pd_state != 1 || g_npd >= PD_MAX || !S.ncol || ci >= S.ncol || ci >= 8) return;
    vkimg_t *im = vk_tex(&S.rt_col[ci]);
    if (im && g_pd_size[0]) { char wh[32]; snprintf(wh, sizeof wh, "%ux%u", im->w, im->h); const char *f = strstr(g_pd_size, wh); size_t l = strlen(wh);
        if (!f || (f != g_pd_size && f[-1] != ',') || (f[l] && f[l] != ',')) return;
        const char *dd = getenv("DM_VK_PASSDUMP_DEPTH"); if (dd && atoi(dd)) pass_dump_depth(); }   // DM_VK_PASSDUMP_DEPTH=1: the size-filtered passes' depth too
    if (!im || (im->fmt != VK_FORMAT_R8G8B8A8_UNORM && im->fmt != VK_FORMAT_R8G8B8A8_SRGB && im->fmt != VK_FORMAT_R16G16B16A16_SFLOAT && im->fmt != VK_FORMAT_B8G8R8A8_UNORM &&
                                                    im->fmt != VK_FORMAT_B10G11R11_UFLOAT_PACK32 && im->fmt != VK_FORMAT_R16_SFLOAT && im->fmt != VK_FORMAT_R8_UNORM && im->fmt != VK_FORMAT_A2B10G10R10_UNORM_PACK32 && im->fmt != VK_FORMAT_R16G16_SFLOAT)) return;
    const view_t *cv = S.rt_has_cview[ci] ? &S.rt_cview[ci] : NULL;   // the view's level and layer (a cube face of an array, a mip)
    uint32_t lv = cv && cv->set_levels ? cv->base_level : 0, ly = cv && cv->set_layers ? cv->min_layer : 0, w = im->w >> lv ? im->w >> lv : 1, h = im->h >> lv ? im->h >> lv : 1;
    if (lv >= im->levels || ly >= im->layers) lv = ly = 0, w = im->w, h = im->h;
    uint32_t bpp = im->fmt == VK_FORMAT_R16G16B16A16_SFLOAT ? 8 : im->fmt == VK_FORMAT_R16_SFLOAT ? 2 : im->fmt == VK_FORMAT_R8_UNORM ? 1 : 4; VkDeviceSize n = (VkDeviceSize)w * h * bpp;
    pdump_t *d = &g_pd[g_npd++]; d->w = w; d->h = h; d->f = im->fmt; d->pass = g_sub_pass - 1; d->present = g_tex_frame;
    VkBufferCreateInfo bi = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, NULL, 0, n, VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_SHARING_MODE_EXCLUSIVE};
    VKCHK(vkCreateBuffer(V.dev, &bi, NULL, &d->b)); VkMemoryRequirements rq; vkGetBufferMemoryRequirements(V.dev, d->b, &rq);
    VkMemoryAllocateInfo ai = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, NULL, rq.size, vk_mem_type(rq.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_CACHED_BIT, 0)};
    VKCHK(vkAllocateMemory(V.dev, &ai, NULL, &d->m)); VKCHK(vkBindBufferMemory(V.dev, d->b, d->m, 0)); VKCHK(vkMapMemory(V.dev, d->m, 0, VK_WHOLE_SIZE, 0, &d->p));
    VkCommandBuffer cb = vk_cmd(); vk_img_barrier_all(cb);
    VkBufferImageCopy c = {0, 0, 0, {VK_IMAGE_ASPECT_COLOR_BIT, lv, ly, 1}, {0, 0, 0}, {w, h, 1}};
    vkCmdCopyImageToBuffer(cb, im->img, VK_IMAGE_LAYOUT_GENERAL, d->b, 1, &c); vk_img_barrier_all(cb);
}
static void texdump_flush(void) {   // DM_VK_TEXDUMP: outside a render pass (pass_dump runs after its end)
    vkimg_t *im = g_td_im; if (!im || g_pd_state != 1) return; g_td_im = NULL; g_td_done = 1;
    VkFormatProperties fp; vkGetPhysicalDeviceFormatProperties(V.pd, im->fmt, &fp);
    if (!(fp.optimalTilingFeatures & VK_FORMAT_FEATURE_BLIT_SRC_BIT)) { VKLOG("texdump: format %d cannot be blitted", im->fmt); return; }
    const char *tl = getenv("DM_VK_TEXDUMP_LEVEL"); uint32_t lv = tl && *tl ? (uint32_t)atoi(tl) : 0; if (lv >= im->levels) lv = im->levels - 1;   // =n: that mip level instead of level 0
    uint32_t w = im->w >> lv ? im->w >> lv : 1, h = im->h >> lv ? im->h >> lv : 1;
    VkCommandBuffer cb = vk_cmd(); vk_img_barrier_all(cb);
    for (uint32_t ly = 0; ly < im->layers && ly < 6 && g_npd < PD_MAX; ly++) {
        VkImageCreateInfo ic = {VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, NULL, 0, VK_IMAGE_TYPE_2D, VK_FORMAT_R16G16B16A16_SFLOAT, {w, h, 1}, 1, 1, VK_SAMPLE_COUNT_1_BIT,
            VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT, VK_SHARING_MODE_EXCLUSIVE, 0, NULL, VK_IMAGE_LAYOUT_UNDEFINED};
        VkImage ti; VKCHK(vkCreateImage(V.dev, &ic, NULL, &ti)); VkMemoryRequirements rq; vkGetImageMemoryRequirements(V.dev, ti, &rq);
        VkMemoryAllocateInfo ma = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, NULL, rq.size, vk_mem_type(rq.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 0)};
        VkDeviceMemory tm; VKCHK(vkAllocateMemory(V.dev, &ma, NULL, &tm)); VKCHK(vkBindImageMemory(V.dev, ti, tm, 0));
        VkImageMemoryBarrier b = {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER, NULL, 0, VK_ACCESS_TRANSFER_WRITE_BIT, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL, VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED, ti, {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
        vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, NULL, 0, NULL, 1, &b);
        VkImageBlit r = {{VK_IMAGE_ASPECT_COLOR_BIT, lv, ly, 1}, {{0, 0, 0}, {(int32_t)w, (int32_t)h, 1}}, {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}, {{0, 0, 0}, {(int32_t)w, (int32_t)h, 1}}};
        vkCmdBlitImage(cb, im->img, VK_IMAGE_LAYOUT_GENERAL, ti, VK_IMAGE_LAYOUT_GENERAL, 1, &r, VK_FILTER_NEAREST); vk_img_barrier_all(cb);
        pdump_t *d = &g_pd[g_npd++]; d->w = w; d->h = h; d->f = VK_FORMAT_R16G16B16A16_SFLOAT; d->pass = -100 - (int)ly; d->present = g_tex_frame;
        VkBufferCreateInfo bi = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, NULL, 0, (VkDeviceSize)w * h * 8, VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_SHARING_MODE_EXCLUSIVE};
        VKCHK(vkCreateBuffer(V.dev, &bi, NULL, &d->b)); vkGetBufferMemoryRequirements(V.dev, d->b, &rq);
        VkMemoryAllocateInfo ai = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, NULL, rq.size, vk_mem_type(rq.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_CACHED_BIT, 0)};
        VKCHK(vkAllocateMemory(V.dev, &ai, NULL, &d->m)); VKCHK(vkBindBufferMemory(V.dev, d->b, d->m, 0)); VKCHK(vkMapMemory(V.dev, d->m, 0, VK_WHOLE_SIZE, 0, &d->p));
        VkBufferImageCopy c = {0, 0, 0, {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}, {0, 0, 0}, {w, h, 1}};
        vkCmdCopyImageToBuffer(cb, ti, VK_IMAGE_LAYOUT_GENERAL, d->b, 1, &c); vk_img_barrier_all(cb);
        vk_defer_free_image(ti, tm);
    }
    VKLOG("texdump: %ux%u fmt %d layers %u levels %u (level %u dumped) at pool %#llx+%#llx: dumps with pass -100 - layer", im->w, im->h, im->fmt, im->layers, im->levels, lv, (unsigned long long)im->pool, (unsigned long long)im->off);
}
static void pass_dump_depth(void) {   // the pass's depth (D24S8 / D32 / D16) as a pdump_t with f = the depth format; written as a grayscale stretched between its min and max
    if (g_pd_state != 1 || g_npd >= PD_MAX || !S.rt_has_dep) return;
    vkimg_t *im = vk_tex(&S.rt_dep); if (!im || !(im->aspect & VK_IMAGE_ASPECT_DEPTH_BIT)) return;
    uint32_t bpp = im->fmt == VK_FORMAT_D16_UNORM ? 2 : 4; VkDeviceSize n = (VkDeviceSize)im->w * im->h * bpp;
    pdump_t *d = &g_pd[g_npd++]; d->w = im->w; d->h = im->h; d->f = im->fmt; d->pass = g_sub_pass - 1; d->present = g_tex_frame;
    VkBufferCreateInfo bi = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, NULL, 0, n, VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_SHARING_MODE_EXCLUSIVE};
    VKCHK(vkCreateBuffer(V.dev, &bi, NULL, &d->b)); VkMemoryRequirements rq; vkGetBufferMemoryRequirements(V.dev, d->b, &rq);
    VkMemoryAllocateInfo ai = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, NULL, rq.size, vk_mem_type(rq.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_CACHED_BIT, 0)};
    VKCHK(vkAllocateMemory(V.dev, &ai, NULL, &d->m)); VKCHK(vkBindBufferMemory(V.dev, d->b, d->m, 0)); VKCHK(vkMapMemory(V.dev, d->m, 0, VK_WHOLE_SIZE, 0, &d->p));
    VkCommandBuffer cb = vk_cmd(); vk_img_barrier_all(cb);
    VkBufferImageCopy c = {0, 0, 0, {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 0, 1}, {0, 0, 0}, {im->w, im->h, 1}};
    vkCmdCopyImageToBuffer(cb, im->img, VK_IMAGE_LAYOUT_GENERAL, d->b, 1, &c); vk_img_barrier_all(cb);
}
static inline float half2f(uint16_t h) { uint32_t s = (uint32_t)(h >> 15) << 31, e = (h >> 10) & 31, m = h & 1023, f;
    if (!e) { if (!m) f = s; else { e = 1; while (!(m & 1024)) { m <<= 1; e--; } m &= 1023; f = s | ((e + 112) << 23) | (m << 13); } } else if (e == 31) f = s | 0x7f800000 | (m << 13); else f = s | ((e + 112) << 23) | (m << 13);
    float r; memcpy(&r, &f, 4); return r; }
void vk_pass_dump_poll(uint64_t done_fence) {   // gfx_vk_present.c, every present: start on the switch, write once the GPU finished
    const char *e = getenv("DM_VK_PASSDUMP");
    if (g_pd_state == 0 && e && atoi(e) > 1) { char n[16]; snprintf(n, sizeof n, "%d", atoi(e) - 1); setenv("DM_VK_PASSDUMP", n, 1); return; }   // =N: the frame after N - 1 more presents
    if (g_pd_state == 0 && e && atoi(e)) { const char *sk = getenv("DM_VK_PASSDUMP_SKIP"); g_pd_skip = sk && !strchr(sk, 'x') ? atoi(sk) : 0;
        snprintf(g_pd_from, sizeof g_pd_from, "%s", sk && strchr(sk, 'x') ? sk : "");
        const char *sz = getenv("DM_VK_PASSDUMP_SIZE"); snprintf(g_pd_size, sizeof g_pd_size, "%s", sz && strcmp(sz, "0") ? sz : "");
        { const char *td = getenv("DM_VK_TEXDUMP"); const char *pl = td ? strchr(td, '+') : NULL; g_td_pool = pl ? strtoull(td, NULL, 16) : 0; g_td_off = pl ? strtoll(pl + 1, NULL, 16) : -1; g_td_im = NULL; g_td_done = 0; }
        const char *fr = getenv("DM_VK_PASSDUMP_FRAMES"); g_pd_frames = fr && atoi(fr) > 1 ? atoi(fr) : 1;
        g_pd_state = 1; g_npd = 0; setenv("DM_VK_PASSDUMP", "0", 1); VKLOG("pass dump: the next submission's passes from pass %d", g_pd_skip); return; }
    if (g_pd_state == 1 && g_npd && --g_pd_frames > 0 && g_npd < PD_MAX - 8) return;   // DM_VK_PASSDUMP_FRAMES=N: N presents in a row (with DM_VK_PASSDUMP_SIZE)
    if (g_pd_state == 1 && g_npd) { g_pd_state = 2; for (int i = 0; i < g_npd; i++) g_pd[i].fence = mtl_last_fence() + 1; return; }
    if (g_pd_state != 2 || !g_npd || done_fence < g_pd[g_npd - 1].fence) return;
    const char *root = getenv("DM_ROOT");
    for (int i = 0; i < g_npd; i++) {
        pdump_t *d = &g_pd[i]; char path[1024]; snprintf(path, sizeof path, "%s/pass_%02d.ppm", root ? root : ".", i);
        VKLOG("pass dump %02d is the submission's pass %d (present %llu)", i, d->pass, (unsigned long long)d->present);
        FILE *f = fopen(path, "wb"); if (!f) continue; fprintf(f, "P6\n%u %u\n255\n", d->w, d->h);
        uint8_t *row = malloc((size_t)d->w * 3); uint64_t bad = 0; float mx = 0, mn = 0; double sum = 0;
        if (d->f == VK_FORMAT_D24_UNORM_S8_UINT || d->f == VK_FORMAT_D32_SFLOAT || d->f == VK_FORMAT_D32_SFLOAT_S8_UINT || d->f == VK_FORMAT_D16_UNORM || d->f == VK_FORMAT_X8_D24_UNORM_PACK32) {
            size_t np = (size_t)d->w * d->h; float lo = 1e30f, hi = -1e30f; float *v = malloc(np * sizeof *v);
            for (size_t k = 0; k < np; k++) { float x;
                if (d->f == VK_FORMAT_D16_UNORM) x = ((const uint16_t *)d->p)[k] / 65535.f;
                else if (d->f == VK_FORMAT_D32_SFLOAT || d->f == VK_FORMAT_D32_SFLOAT_S8_UINT) memcpy(&x, (const uint8_t *)d->p + 4 * k, 4);
                else x = (((const uint32_t *)d->p)[k] & 0xffffff) / 16777215.f;
                v[k] = x; if (x < lo) lo = x; if (x > hi) hi = x; }
            for (uint32_t y = 0; y < d->h; y++) { for (uint32_t x = 0; x < d->w; x++) { float t = hi > lo ? (v[(size_t)y * d->w + x] - lo) / (hi - lo) : 0; row[3 * x] = row[3 * x + 1] = row[3 * x + 2] = (uint8_t)(t * 255); } fwrite(row, 1, (size_t)d->w * 3, f); }
            { char rp[1100]; snprintf(rp, sizeof rp, "%s.f32", path); FILE *rf = fopen(rp, "wb"); if (rf) { fwrite(v, sizeof *v, np, rf); fclose(rf); } }   // the values themselves
            free(v); free(row); fclose(f); VKLOG("pass dump %02d: depth %ux%u format %d, values %.6f .. %.9g → %s", i, d->w, d->h, d->f, lo, hi, path);
            vkDestroyBuffer(V.dev, d->b, NULL); vkFreeMemory(V.dev, d->m, NULL); continue;
        }
        if (d->f == VK_FORMAT_R16G16_SFLOAT) {   // signed two-channel fields (velocities): red and green around grey, 1/16 a step; the values themselves in .f32 (r, g a texel)
            size_t np = (size_t)d->w * d->h; float *v = malloc(np * 2 * sizeof *v), lo = 0, hi = 0; uint64_t nz = 0;
            for (size_t k = 0; k < np * 2; k++) { v[k] = half2f(((const uint16_t *)d->p)[k]); if (v[k] < lo) lo = v[k]; if (v[k] > hi) hi = v[k]; }
            for (uint32_t y = 0; y < d->h; y++) { for (uint32_t x = 0; x < d->w; x++) { const float *t = v + 2 * ((size_t)y * d->w + x); int on = t[0] != 0 || t[1] != 0; nz += on;
                for (int c = 0; c < 2; c++) { float q = 128 + t[c] * 16; row[3 * x + c] = (uint8_t)(q < 0 ? 0 : q > 255 ? 255 : q); } row[3 * x + 2] = on ? 255 : 128; }
                fwrite(row, 1, (size_t)d->w * 3, f); }
            { char rp[1100]; snprintf(rp, sizeof rp, "%s.f32", path); FILE *rf = fopen(rp, "wb"); if (rf) { fwrite(v, sizeof *v, np * 2, rf); fclose(rf); } }
            free(v); free(row); fclose(f); VKLOG("pass dump %02d: %ux%u RG16F, values %.3f .. %.3f, nonzero texels %llu → %s", i, d->w, d->h, lo, hi, (unsigned long long)nz, path);
            vkDestroyBuffer(V.dev, d->b, NULL); vkFreeMemory(V.dev, d->m, NULL); continue;
        }
        for (uint32_t y = 0; y < d->h; y++) { for (uint32_t x = 0; x < d->w; x++) {
            if (d->f == VK_FORMAT_R16G16B16A16_SFLOAT || d->f == VK_FORMAT_B10G11R11_UFLOAT_PACK32 || d->f == VK_FORMAT_R16_SFLOAT) { float v3[3];
                if (d->f == VK_FORMAT_R16G16B16A16_SFLOAT) { const uint16_t *p = (const uint16_t *)d->p + ((size_t)y * d->w + x) * 4; for (int c = 0; c < 3; c++) v3[c] = half2f(p[c]); }
                else if (d->f == VK_FORMAT_R16_SFLOAT) v3[0] = v3[1] = v3[2] = half2f(((const uint16_t *)d->p)[(size_t)y * d->w + x]);
                else { uint32_t w = ((const uint32_t *)d->p)[(size_t)y * d->w + x];   // 11-bit and 10-bit floats: a half's exponent and top mantissa bits
                    v3[0] = half2f((uint16_t)((w & 0x7ff) << 4)); v3[1] = half2f((uint16_t)((w >> 11 & 0x7ff) << 4)); v3[2] = half2f((uint16_t)((w >> 22 & 0x3ff) << 5)); }
                for (int c = 0; c < 3; c++) { float v = v3[c]; if (v != v || v > 65000.f || v < -65000.f) { bad++; v = 1; } if (v > mx) mx = v; if (v < mn) mn = v; sum += v; row[3 * x + c] = (uint8_t)(v <= 0 ? 0 : v >= 1 ? 255 : v * 255 + .5f); } }
            else if (d->f == VK_FORMAT_R8_UNORM) row[3 * x] = row[3 * x + 1] = row[3 * x + 2] = ((const uint8_t *)d->p)[(size_t)y * d->w + x];
            else if (d->f == VK_FORMAT_A2B10G10R10_UNORM_PACK32) { uint32_t w = ((const uint32_t *)d->p)[(size_t)y * d->w + x]; for (int c = 0; c < 3; c++) row[3 * x + c] = (uint8_t)((w >> (10 * c) & 0x3ff) >> 2); }
            else { const uint8_t *p = (const uint8_t *)d->p + ((size_t)y * d->w + x) * 4; int bgr = d->f == VK_FORMAT_B8G8R8A8_UNORM; row[3 * x] = p[bgr ? 2 : 0]; row[3 * x + 1] = p[1]; row[3 * x + 2] = p[bgr ? 0 : 2]; } }
            fwrite(row, 1, (size_t)d->w * 3, f); }
        free(row); fclose(f);
        VKLOG("pass dump %02d: %ux%u format %d → %s%s", i, d->w, d->h, d->f, path, d->f == VK_FORMAT_R16G16B16A16_SFLOAT ? "" : "");
        if (d->f == VK_FORMAT_R16G16B16A16_SFLOAT || d->f == VK_FORMAT_B10G11R11_UFLOAT_PACK32 || d->f == VK_FORMAT_R16_SFLOAT) VKLOG("   NaN/inf channels %llu, finite values %.3f .. %.3f, mean %.5f", (unsigned long long)bad, mn, mx, sum / (3.0 * d->w * d->h));
        if (d->f == VK_FORMAT_B10G11R11_UFLOAT_PACK32 && d->w * d->h <= 4) { char t[200]; int k = 0;   // a tiny target (an exposure value): every texel
            for (uint32_t q = 0; q < d->w * d->h; q++) { uint32_t w = ((const uint32_t *)d->p)[q];
                k += snprintf(t + k, sizeof t - k, " (%g %g %g)", half2f((uint16_t)((w & 0x7ff) << 4)), half2f((uint16_t)((w >> 11 & 0x7ff) << 4)), half2f((uint16_t)((w >> 22 & 0x3ff) << 5))); }
            VKLOG("   texels:%s", t); }
        vkDestroyBuffer(V.dev, d->b, NULL); vkFreeMemory(V.dev, d->m, NULL);
    }
    g_npd = 0; g_pd_state = 0;
}
// DM_VK_CRUMBS=1 (start-up), for a device lost (a GPU hang): every pass end and dispatch writes {submission, pass} into host-visible memory after a full barrier, and the host
// keeps the shaders each pass drew with; when the device is lost (vk_crumbs_report), the last work the GPU finished and the passes after it are logged
static int g_crumbs = -1; static VkBuffer g_crumb_buf; static volatile uint32_t *g_crumb;
typedef struct { uint32_t sub; int pass, n; uint64_t k[8][3]; } crumb_t; static crumb_t g_cring[512]; static unsigned g_cn;
static int crumbs_on(void) {
    if (g_crumbs < 0) { const char *e = getenv("DM_VK_CRUMBS"); g_crumbs = e && atoi(e);
        if (g_crumbs) { VkBufferCreateInfo bi = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, NULL, 0, 256, VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_SHARING_MODE_EXCLUSIVE};
            VKCHK(vkCreateBuffer(V.dev, &bi, NULL, &g_crumb_buf)); VkMemoryRequirements rq; vkGetBufferMemoryRequirements(V.dev, g_crumb_buf, &rq);
            VkMemoryAllocateInfo ai = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, NULL, rq.size, vk_mem_type(rq.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, 0)};
            VkDeviceMemory m; VKCHK(vkAllocateMemory(V.dev, &ai, NULL, &m)); VKCHK(vkBindBufferMemory(V.dev, g_crumb_buf, m, 0)); void *p; VKCHK(vkMapMemory(V.dev, m, 0, VK_WHOLE_SIZE, 0, &p)); g_crumb = p; g_crumb[0] = 0; } }
    return g_crumbs;
}
static crumb_t *crumb_cur(void) { crumb_t *c = &g_cring[(g_cn ? g_cn - 1 : 0) % 512]; uint32_t sub = (uint32_t)mtl_last_fence() + 1;
    if (!g_cn || c->sub != sub || c->pass != g_sub_pass - 1) { c = &g_cring[g_cn++ % 512]; memset(c, 0, sizeof *c); c->sub = sub; c->pass = g_sub_pass - 1; } return c; }
static void crumb_draw(uint64_t a, uint64_t b, uint64_t c3) { if (!crumbs_on()) return; crumb_t *c = crumb_cur();
    for (int i = 0; i < c->n; i++) if (c->k[i][0] == a && c->k[i][1] == b && c->k[i][2] == c3) return; if (c->n < 8) { c->k[c->n][0] = a; c->k[c->n][1] = b; c->k[c->n][2] = c3; c->n++; } }
static void crumb_mark(int dispatch) { if (!crumbs_on()) return; VkCommandBuffer cb = vk_cmd(); vk_img_barrier_all(cb);
    vkCmdFillBuffer(cb, g_crumb_buf, 0, 4, ((uint32_t)(mtl_last_fence() + 1) & 0xffff) << 16 | (dispatch ? 0x8000u : 0) | ((uint32_t)(g_sub_pass - 1) & 0x7fff)); }
void vk_crumbs_report(void) {
    if (g_crumbs <= 0 || !g_crumb) return;
    uint32_t v = g_crumb[0], sub = v >> 16, pass = v & 0x7fff;
    VKLOG("[crumbs] the GPU last finished submission %u (low 16 bits) %s %u", sub, v & 0x8000 ? "a dispatch after pass" : "pass", pass);
    int shown = 0;
    for (unsigned i = g_cn > 512 ? g_cn - 512 : 0; i < g_cn && shown < 12; i++) { crumb_t *c = &g_cring[i % 512];
        if ((c->sub & 0xffff) < sub || ((c->sub & 0xffff) == sub && c->pass <= (int)pass)) continue;
        shown++; VKLOG("[crumbs] then submission %u pass %d:", c->sub, c->pass);
        for (int k = 0; k < c->n; k++) VKLOG("[crumbs]   shaders %016llx %016llx %016llx", (unsigned long long)c->k[k][0], (unsigned long long)c->k[k][1], (unsigned long long)c->k[k][2]); }
}
void vk_prof_end(void) {
    if (!g_prof_on || !g_prof_t0) return;
    extern uint64_t g_pkg_ns, g_pkg_n; uint64_t tot = tick() - g_prof_t0; g_prof_t0 = 0;
    if (tot < 40000000ull) return;
    double ms = 1e-6; uint64_t known = g_prof_ns[0] + g_prof_ns[1] + g_prof_ns[2] + g_prof_ns[3] + g_prof_ns[4] + g_prof_ns[5] + g_prof_ns[6] + g_prof_ns[7];
    VKLOG("slow replay (present %llu): %.1f ms; pipelines built here %llu in %.1f ms, textures %.1f ms (%llu packaged uploads, %.1f ms of it), pools sync %.1f ms, shader modules %.1f ms, pipeline binds %.1f ms, descriptor sets %.1f ms, cuts %.1f ms, compute pipelines %.1f ms, other %.1f ms",
          (unsigned long long)g_tex_frame, tot * ms, (unsigned long long)(g_vkst.pipes_built - g_prof_pipes0), g_prof_ns[0] * ms, g_prof_ns[1] * ms, (unsigned long long)(g_pkg_n - g_prof_npkg0),
          (g_pkg_ns - g_prof_pkg0) * ms, g_prof_ns[2] * ms, g_prof_ns[3] * ms, g_prof_ns[4] * ms, g_prof_ns[5] * ms, g_prof_ns[6] * ms, g_prof_ns[7] * ms, (tot > known ? tot - known : 0) * ms);
}
// Linear render targets (NVN LINEAR_RENDER_TARGET) are their storage on the Switch: the CPU can read what the GPU drew straight from the pool. Here a Vulkan image holds
// the drawing and the pool memory kept what it held before. Super Mario Odyssey's fluid surface (al::FluidSimulateWave: the Cap Kingdom ground fog's wave solver ends in a
// 128x128 RGBA16F linear target that the fog's vertex shader samples, and a 256x128 R32F one) is such a target. So a pass that drew into one copies each such colour
// target back to its storage, at its stride, marked as written by the GPU (the pools' copy-back carries it to the CPU). DM_VK_LIN_BACK=0 (live, A/B): not
static void lin_back(void) {
    for (int i = 0; i < S.ncol && i < 8; i++) {
        vkimg_t *im = P_att[i].im; const tex_t *t = S.rt_has_col[i] ? &S.rt_col[i] : NULL; const vkfmt_t *f = t ? vk_fmt(t->d.format) : NULL;
        if (!im || !t || !t->mem || !(t->d.flags & NVN_TEXTURE_FLAGS_LINEAR_RENDER_TARGET) || !f || !f->bpb || f->bw > 1 || im->levels != 1 || im->layers != 1 || im->d != 1) continue;
        size_t stride = t->d.stride ? t->d.stride : (size_t)im->w * f->bpb, len = stride * (im->h - 1) + (size_t)im->w * f->bpb; VkBuffer b; VkDeviceSize o;
        if (stride % f->bpb || stride < (size_t)im->w * f->bpb || !vk_buf_at((uint64_t)(uintptr_t)t->mem, len, &b, &o) || o % f->bpb) continue;
        VkCommandBuffer cb = vk_cmd(); vk_img_barrier_all(cb);
        VkBufferImageCopy r = {o, (uint32_t)(stride / f->bpb), 0, {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}, {0, 0, 0}, {im->w, im->h, 1}};
        vkCmdCopyImageToBuffer(cb, im->img, VK_IMAGE_LAYOUT_GENERAL, b, 1, &r); vk_img_barrier_all(cb);
        void vk_gpu_wrote(uint64_t a, uint64_t n); vk_gpu_wrote((uint64_t)(uintptr_t)t->mem, len);
        { static int n; if (n++ < 4) VKLOG("linear render target %ux%u (nvn format %#x, stride %zu) copied back to its storage %p", im->w, im->h, t->d.format, stride, (void *)t->mem); }
    }
}
void vk_end_pass(void) {
    if (g_x_listpasses && S.pass_open == 1) { Dl_info di; void *ra = __builtin_return_address(0); uintptr_t base = dladdr((void *)vk_end_pass, &di) ? (uintptr_t)di.dli_fbase : 0; VKLOG("    ended from %#llx (%llu draws)", (unsigned long long)((uintptr_t)ra - base), (unsigned long long)g_pass_draws); }   // (DM_VK_TRACE_PASS=-2: who ends each pass)
    if (S.pass_open == 2 && P_cleared) (void)pass_commit();   // only clears: the pass still runs, for its load ops
    if (S.pass_open == 1) { if (S.rp_pass) vkCmdEndRenderPass(vk_cmd()); else vkCmdEndRendering(vk_cmd());
        if (vk_prof_on()) {   // (DM_VK_GPUPROF: the pass as a unit: its attachments, load and store ops, draws, and the op that ended it)
            g_pu.w = S.pass_w; g_pu.h = S.pass_h; g_pu.draws = (uint32_t)g_pass_draws; g_pu.ncol = (uint8_t)S.ncol; g_pu.end_op = (uint8_t)(g_cur_op >= 0 && g_cur_op < 256 ? g_cur_op : 254);
            for (int i = 0; i < S.ncol && i < 8; i++) { int lo = P_ca[i].loadOp == VK_ATTACHMENT_LOAD_OP_CLEAR ? 1 : P_ca[i].loadOp == VK_ATTACHMENT_LOAD_OP_DONT_CARE ? 2 : 0;
                if (i < 4) { g_pu.cfmt[i] = (uint32_t)S.cfmt[i]; g_pu.load |= (uint16_t)(lo << (2 * i)); } if (P_ca[i].storeOp == VK_ATTACHMENT_STORE_OP_DONT_CARE) g_pu.store |= (uint16_t)(1u << i); }
            if (S.daspect) { g_pu.dfmt = (uint32_t)S.dfmt; const VkRenderingAttachmentInfo *d = (S.daspect & VK_IMAGE_ASPECT_DEPTH_BIT) ? &P_da : &P_sa;
                g_pu.load |= (uint16_t)((d->loadOp == VK_ATTACHMENT_LOAD_OP_CLEAR ? 1 : d->loadOp == VK_ATTACHMENT_LOAD_OP_DONT_CARE ? 2 : 0) << 8);
                if (S.daspect & VK_IMAGE_ASPECT_STENCIL_BIT) g_pu.load |= (uint16_t)((P_sa.loadOp == VK_ATTACHMENT_LOAD_OP_CLEAR ? 1 : P_sa.loadOp == VK_ATTACHMENT_LOAD_OP_DONT_CARE ? 2 : 0) << 10);
                if (d->storeOp == VK_ATTACHMENT_STORE_OP_DONT_CARE) g_pu.store |= 1u << 8; if ((S.daspect & VK_IMAGE_ASPECT_STENCIL_BIT) && P_sa.storeOp == VK_ATTACHMENT_STORE_OP_DONT_CARE) g_pu.store |= 1u << 9; }
            pu_end(VK_PU_PASS); }
        dsc_pass_end();
        pass_dump(); if (g_x_tracepass >= 0 && g_sub_pass - 1 == g_x_tracepass) { setenv("DM_VK_TRACE_PASS", "-1", 1); g_x_tracepass = -1; } }
    if (S.pass_open == 1 && g_lin_back && g_pass_draws) lin_back();
    if (S.pass_open == 1) crumb_mark(0);
    S.pass_open = 0; S.bound_pipe = VK_NULL_HANDLE; g_tf_in_pass = 0; g_disc_pend = 0;   // (a discard before the next pass is set up holds only up to the next pass end: a copy may write the target)
}
// a pass end before something that reads or writes pool memory other than through a uniform binding (copies, fills, dispatches, buffer ↔ texture
// copies, the submission's end): the deferred uniform updates land first (draws read them through their shadows, so ordinary pass ends keep them)
void vk_end_pass_land(void) { vk_end_pass(); if (vk_sub_open()) vk_ubd_land(vk_cmd()); }
static int pass_whole(VkRect2D r) { return !r.offset.x && !r.offset.y && r.extent.width >= S.pass_w && r.extent.height >= S.pass_h; }
static int begin_pass(void) {
    if (S.pass_open) return 1;
    P_dload = 0;
    int psplit = g_psplit_now; g_psplit_now = 0;   // a pass split (prepare_draw): taken before any early return, so the flag never forces a later pass's cut
    VkCommandBuffer cb = vk_cmd();
    VkRenderingAttachmentInfo *ca = P_ca; memset(ca, 0, sizeof P_ca);
    for (int i = 0; i < 9; i++) P_att[i].im = NULL;
    VkRenderingAttachmentInfo da = {0}, sa = {0};
    uint32_t w = UINT32_MAX, h = UINT32_MAX, layers = UINT32_MAX, samples = 1; int ncol = 0, have = 0;
    for (int i = 0; i < S.rt_n; i++) {
        S.cfmt[i] = VK_FORMAT_UNDEFINED;
        ca[i].sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO; ca[i].imageLayout = VK_IMAGE_LAYOUT_GENERAL; ca[i].loadOp = VK_ATTACHMENT_LOAD_OP_LOAD; ca[i].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        if (!S.rt_has_col[i]) continue;
        vkimg_t *im; VkImageView v = vk_tex_view(&S.rt_col[i], S.rt_has_cview[i] ? &S.rt_cview[i] : NULL, 0xff, 0, 1, &im);
        if (!v) continue;
        vk_tex_prepare(cb, im);
        uint32_t lv = S.rt_has_cview[i] && S.rt_cview[i].set_levels ? S.rt_cview[i].base_level : 0;
        uint32_t iw = im->w >> lv ? im->w >> lv : 1, ih = im->h >> lv ? im->h >> lv : 1;
        // a block-compressed texture drawn into through an uncompressed view (RGBA32UI over BC3: one texel per 4x4 block, Mario Kart 8 Deluxe compressing its
        // reflection cubemaps on the GPU) is that view's size in blocks: with the texel size the render area was 16 times the attachment, which Vulkan does not allow (the render area must lie within every attachment view)
        { const vkfmt_t *tf = vk_fmt(S.rt_col[i].d.format), *vf = S.rt_has_cview[i] && S.rt_cview[i].set_format ? vk_fmt(S.rt_cview[i].format) : tf;
          if (tf && vf && tf->compressed && !tf->dec && !vf->compressed && tf->bw && tf->bh) { iw = (iw + tf->bw - 1) / tf->bw; ih = (ih + tf->bh - 1) / tf->bh;
              static int said; if (!said++) VKLOG("colour target: block-compressed format %#x drawn through view format %#x: pass in blocks, %ux%u", S.rt_col[i].d.format, S.rt_cview[i].format, iw, ih); } }
        if (iw < w) w = iw; if (ih < h) h = ih; samples = im->samples; have = 1;
        ca[i].imageView = v; S.cfmt[i] = vk_view_fmt; ncol = i + 1; P_att[i].im = im;   // (the view's format: a render pass's attachment and its pipelines take it)
        P_att[i].fmt = vk_view_fmt; P_att[i].layers = vk_view_layers ? vk_view_layers : 1; P_att[i].w = iw; P_att[i].h = ih;
        if (P_att[i].layers < layers) layers = P_att[i].layers;
    }
    S.dfmt = VK_FORMAT_UNDEFINED; S.daspect = 0;
    if (S.rt_has_dep) {
        vkimg_t *im; VkImageView v = vk_tex_view(&S.rt_dep, S.rt_has_dview ? &S.rt_dview : NULL, 0xff, 0, 1, &im);
        if (v) {
            vk_tex_prepare(cb, im);
            uint32_t lv = S.rt_has_dview && S.rt_dview.set_levels ? S.rt_dview.base_level : 0;
            uint32_t iw = im->w >> lv ? im->w >> lv : 1, ih = im->h >> lv ? im->h >> lv : 1;
            if (iw < w) w = iw; if (ih < h) h = ih; samples = im->samples; have = 1;
            da = (VkRenderingAttachmentInfo){VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO, NULL, v, VK_IMAGE_LAYOUT_GENERAL, 0, NULL, VK_IMAGE_LAYOUT_UNDEFINED, VK_ATTACHMENT_LOAD_OP_LOAD, VK_ATTACHMENT_STORE_OP_STORE};
            S.dfmt = im->fmt; S.daspect = im->aspect; P_att[8].im = im; P_att[8].fmt = vk_view_fmt; P_att[8].layers = vk_view_layers ? vk_view_layers : 1; P_att[8].w = iw; P_att[8].h = ih;
            if (P_att[8].layers < layers) layers = P_att[8].layers;
            if (im->aspect & VK_IMAGE_ASPECT_STENCIL_BIT) sa = da;
        }
    }
    if (!have) { VKRED_ONCE(30, "draw with no usable render target: skipped"); return 0; }
    // The cut (DM_VK_CB_DRAWS): only before a pass whose depth attachment no earlier pass of this submission drew with. Adreno keeps a low-resolution copy of a depth
    // attachment (LRZ) that it trusts within a command buffer; a pass that goes on with a depth buffer begun in an earlier command buffer had whole 8x8 blocks of its
    // nearest objects rejected in some frames (NEO: bushes, trees and the phone booth near the camera flickered light blue). DM_VK_X_CUTANY=1 cuts before any pass.
    // The rule looks only at the pass at the cut: a later pass may still go on with a depth buffer drawn before it. Hades II cuts before its depth-less passes and then
    // goes on with the 1280x720 D24S8 of the passes before the cut: its lights went out for single frames while Melinoe walked near them (one-frame dark spikes in
    // 6-23 of 600 recorded frames; 0 with DM_VK_CB_DRAWS=0, also 2-15 with the parts submitted together, 135 without the LRZ touches). Such a pass first gets a texel of its
    // depth written back (lrz_touch), the transfer write that made Adreno drop a stale LRZ for Xenoblade Chronicles 3 (DM_VK_LRZ_TOUCH=0 leaves it out)
    static uint64_t g_cuts; int cut_touch = 0;
    vkimg_t *dim = da.imageView ? P_att[8].im : NULL;   // (the image vk_tex_view gave the depth attachment above: looking it up again cost a registry, image and view lookup per pass)
    if (psplit || (g_x_cbdraws > 0 && g_cb_draws >= g_x_cbdraws && (!dim || dim->depth_sub != g_set_epoch || g_x_cutany))) {   // (a pass split: prepare_draw checked the depth rule and the part budget)
        void vk_cmd_split(void); if (g_x_landcut) vk_ubd_land(vk_cmd());
        if (psplit) vk_cmd_split(); else if (g_ord_parts < 15) { vk_cmd_split(); g_ord_parts++; }   // ordinary cuts as before MAIN_CBS 32 (refused past 15; g_cuts still counts, as it did)
        g_cb_draws = 0; cb = vk_cmd(); g_cuts++;   // no pass is open here; deferred uniform updates stay deferred (draws read their shadows)
    }
    if (dim) { cut_touch = g_lrz_touch && dim->depth_sub == g_set_epoch && dim->depth_cut != g_cuts; dim->depth_sub = g_set_epoch; dim->depth_cut = g_cuts; }
    { static int n; uint32_t cs[9] = {0}; int any = 0, mis = 0; uint32_t s0 = 0;   // multisampled passes: every attachment's sample count (they must agree)
      for (int i = 0; i < ncol; i++) if (ca[i].imageView) { vkimg_t *im2 = P_att[i].im; cs[i] = im2 ? im2->samples : 0; if (cs[i] > 1) any = 1; if (!s0) s0 = cs[i]; else if (cs[i] != s0) mis = 1; }   // (the attachments' images found above)
      if (da.imageView) { vkimg_t *im2 = P_att[8].im; cs[8] = im2 ? im2->samples : 0; if (cs[8] > 1) any = 1; if (s0 && cs[8] != s0) mis = 1; }
      if ((any || mis) && n < 10) { n++; VKLOG("pass %ux%u samples: colour %u %u %u %u, depth %u%s", w, h, cs[0], cs[1], cs[2], cs[3], cs[8], mis ? " MISMATCH" : ""); } }
    if (!g_x_nobar) vk_img_barrier_all(cb);
    if (cut_touch) lrz_touch(dim, 1);   // (after the barrier: the earlier command buffer's depth writes come first)
    if (g_x_depth) { da.loadOp = sa.loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE; da.storeOp = sa.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE; }
    if (g_x_color) for (int i = 0; i < ncol; i++) ca[i].loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    P_da = da; P_sa = sa; P_cleared = 0;
    P_ri = (VkRenderingInfo){VK_STRUCTURE_TYPE_RENDERING_INFO, NULL, 0, {{0, 0}, {w, h}}, layers, 0, (uint32_t)ncol, P_ca,
                             (S.daspect & VK_IMAGE_ASPECT_DEPTH_BIT) ? &P_da : NULL, (S.daspect & VK_IMAGE_ASPECT_STENCIL_BIT) ? &P_sa : NULL};
    S.pass_open = 2; S.pass_w = w; S.pass_h = h; S.samples = samples; S.ncol = ncol; S.bound_pipe = VK_NULL_HANDLE;
    if (g_disc_pend) { uint32_t b = g_disc_pend; g_disc_pend = 0; dsc_load(b); }   // discards made after these targets were set and before this pass (disc_op)
    return 1;
}
static VkRect2D scissor_rect(void) {
    int32_t x = 0, y = 0; int64_t x1 = S.pass_w, y1 = S.pass_h;
    if (S.sc_set) {
        int64_t sx = S.sc.x, sy = S.sc.y, sx1 = (int64_t)S.sc.x + S.sc.w, sy1 = (int64_t)S.sc.y + S.sc.h;
        if (g_lower_left && !g_flipy) { int64_t t = (int64_t)S.pass_h - sy1; sy1 = (int64_t)S.pass_h - sy; sy = t; }
        if (sx > x) x = (int32_t)sx; if (sy > y) y = (int32_t)sy; if (sx1 < x1) x1 = sx1; if (sy1 < y1) y1 = sy1;
    }
    if (x1 < x) x1 = x; if (y1 < y) y1 = y;
    return (VkRect2D){{x, y}, {(uint32_t)(x1 - x), (uint32_t)(y1 - y)}};
}
// A clear of some channels only (NVN's ClearColor mask): vkCmdClearAttachments writes every channel, so it is a draw of dm/android/shaders/maskclear.* with the mask as
// the colour write mask, cut to the clear's rectangle by the scissor. Paper Mario: The Thousand-Year Door clears alpha alone (mask 0x8) over a rectangle of its scene
// target when Mario comes out of the water: cleared in full, the floor and water there went black for a frame. One pipeline per pass layout and mask, for attachment 0
#include "maskclear_vert.spv.h"
#include "maskclear_frag.spv.h"
typedef struct { uint32_t ncol, cfmt[8], dfmt, samples, mask; } mckey;
static struct { VkPipelineLayout pl; VkShaderModule vs, fs; int bad; mckey k[32]; VkPipeline p[32]; int n; } g_mc;
static VkPipeline mc_pipeline(uint32_t mask) {
    mckey k; memset(&k, 0, sizeof k); k.ncol = (uint32_t)S.ncol; for (int i = 0; i < S.ncol && i < 8; i++) k.cfmt[i] = (uint32_t)S.cfmt[i];
    k.dfmt = (uint32_t)S.dfmt; k.samples = S.samples ? S.samples : 1; k.mask = mask;
    for (int i = 0; i < g_mc.n; i++) if (!memcmp(&g_mc.k[i], &k, sizeof k)) return g_mc.p[i];
    if (g_mc.bad || g_mc.n == 32) return VK_NULL_HANDLE;
    if (!g_mc.pl) {
        VkPushConstantRange pr = {VK_SHADER_STAGE_FRAGMENT_BIT, 0, 16};
        VkPipelineLayoutCreateInfo li = {VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, NULL, 0, 0, NULL, 1, &pr};
        VkShaderModuleCreateInfo vi = {VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, NULL, 0, sizeof maskclear_vert_spv, maskclear_vert_spv}, fi = vi;
        fi.codeSize = sizeof maskclear_frag_spv; fi.pCode = maskclear_frag_spv;
        if (vkCreatePipelineLayout(V.dev, &li, NULL, &g_mc.pl) || vkCreateShaderModule(V.dev, &vi, NULL, &g_mc.vs) || vkCreateShaderModule(V.dev, &fi, NULL, &g_mc.fs)) { g_mc.bad = 1; return VK_NULL_HANDLE; }
    }
    VkPipelineShaderStageCreateInfo ss[2] = {{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, NULL, 0, VK_SHADER_STAGE_VERTEX_BIT, g_mc.vs, "main", NULL},
                                             {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, NULL, 0, VK_SHADER_STAGE_FRAGMENT_BIT, g_mc.fs, "main", NULL}};
    VkPipelineVertexInputStateCreateInfo vin = {VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
    VkPipelineInputAssemblyStateCreateInfo ia = {VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO, NULL, 0, VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST, 0};
    VkPipelineViewportStateCreateInfo vs = {VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO, NULL, 0, 1, NULL, 1, NULL};
    VkPipelineRasterizationStateCreateInfo rs = {VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO, NULL, 0, 0, 0, VK_POLYGON_MODE_FILL, VK_CULL_MODE_NONE, VK_FRONT_FACE_COUNTER_CLOCKWISE, 0, 0, 0, 0, 1.0f};
    VkPipelineMultisampleStateCreateInfo ms = {VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO, NULL, 0, (VkSampleCountFlagBits)k.samples, 0, 0, NULL, 0, 0};
    VkPipelineDepthStencilStateCreateInfo ds = {VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
    VkPipelineColorBlendAttachmentState cba[8]; memset(cba, 0, sizeof cba);
    cba[0].colorWriteMask = mask & 0xf;   // NVN's mask bits R G B A are Vulkan's
    VkPipelineColorBlendStateCreateInfo cb = {VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO, NULL, 0, 0, 0, k.ncol, cba, {0, 0, 0, 0}};
    VkDynamicState dyn[] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
    VkPipelineDynamicStateCreateInfo dy = {VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO, NULL, 0, 2, dyn};
    VkFormat cf[8]; for (uint32_t i = 0; i < k.ncol; i++) cf[i] = (VkFormat)k.cfmt[i];
    VkFormat df = (VkFormat)k.dfmt; int has_stencil = df == VK_FORMAT_D24_UNORM_S8_UINT || df == VK_FORMAT_D32_SFLOAT_S8_UINT || df == VK_FORMAT_S8_UINT;
    VkPipelineRenderingCreateInfo rci = {VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO, NULL, 0, k.ncol, cf, df == VK_FORMAT_S8_UINT ? VK_FORMAT_UNDEFINED : df, has_stencil ? df : VK_FORMAT_UNDEFINED};
    VkRenderPass prp = rp_on() ? rp_compat(k.ncol, k.cfmt, k.dfmt, k.samples) : VK_NULL_HANDLE;
    VkGraphicsPipelineCreateInfo ci = {VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO, prp ? NULL : &rci, 0, 2, ss, &vin, &ia, NULL, &vs, &rs, &ms, &ds, &cb, &dy, g_mc.pl, prp, 0, VK_NULL_HANDLE, -1};
    VkPipeline p = VK_NULL_HANDLE;
    if ((!prp && !V.has_dynrender) || vkCreateGraphicsPipelines(V.dev, V.pcache, 1, &ci, NULL, &p)) { VKRED_ONCE(32, "masked clear pipeline failed: cleared in full"); p = VK_NULL_HANDLE; }
    g_mc.k[g_mc.n] = k; g_mc.p[g_mc.n++] = p; return p;
}
static int g_bc_stale;
static int clear_masked(const p_clear_color *c, VkRect2D r) {   // 1 = done
    const vkfmt_t *f = vk_fmt(S.rt_col[c->index].d.format);
    if (c->index != 0 || (f && f->is_int)) return 0;
    VkPipeline p = mc_pipeline(c->mask); if (!p) return 0;
    if (!pass_commit()) return 1;
    VkCommandBuffer cb = vk_cmd();
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, p); S.bound_pipe = VK_NULL_HANDLE;
    VkViewport vp = {0, 0, (float)S.pass_w, (float)S.pass_h, 0, 1}; vkCmdSetViewport(cb, 0, 1, &vp); vkCmdSetScissor(cb, 0, 1, &r);
    vkCmdPushConstants(cb, g_mc.pl, VK_SHADER_STAGE_FRAGMENT_BIT, 0, 16, c->rgba);
    vkCmdDraw(cb, 3, 1, 0, 0);
    g_bc_stale = 1;   // the next draw sets its viewport, scissor and bindings again (bc_fresh)
    return 1;
}
static void clear_color(const p_clear_color *c) {
    if (!begin_pass()) return;
    if (c->index < 0 || c->index >= S.ncol || S.cfmt[c->index] == VK_FORMAT_UNDEFINED) return;
    if (!c->mask) return;
    if ((c->mask & 0xf) != 0xf) { VkRect2D r = scissor_rect(); if (!r.extent.width || !r.extent.height) return;
        if (clear_masked(c, r)) return; VKRED_ONCE(31, "partial channel clear mask %#x cleared in full", c->mask); }
    VkClearAttachment a = {VK_IMAGE_ASPECT_COLOR_BIT, (uint32_t)c->index, {{{c->rgba[0], c->rgba[1], c->rgba[2], c->rgba[3]}}}};
    const vkfmt_t *f = vk_fmt(S.rt_col[c->index].d.format);
    if (f && f->is_int) for (int i = 0; i < 4; i++) { if (f->is_sint) a.clearValue.color.int32[i] = (int32_t)c->rgba[i]; else a.clearValue.color.uint32[i] = c->rgba[i] > 0 ? (uint32_t)c->rgba[i] : 0; }
    VkClearRect r = {scissor_rect(), 0, 1};
    if (S.pass_open == 2 && (c->mask & 0xf) == 0xf && pass_whole(r.rect)) {   // the whole attachment before any draw: its load op
        P_ca[c->index].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR; P_ca[c->index].clearValue.color = a.clearValue.color; P_cleared = 1; g_vkst.folded++; return;
    }
    if (!pass_commit()) return;
    if (r.rect.extent.width && r.rect.extent.height) vkCmdClearAttachments(vk_cmd(), 1, &a, 1, &r);
}
static void clear_ds(const p_clear_ds *c) {
    if (!begin_pass() || !S.daspect) return;
    VkImageAspectFlags as = (c->depth_mask ? VK_IMAGE_ASPECT_DEPTH_BIT : 0) | (c->stencil_mask && (S.daspect & VK_IMAGE_ASPECT_STENCIL_BIT) ? VK_IMAGE_ASPECT_STENCIL_BIT : 0);
    as &= S.daspect; if (!as) return;
    VkClearAttachment a = {as, 0, {.depthStencil = {c->depth, (uint32_t)c->stencil}}};
    VkClearRect r = {scissor_rect(), 0, 1};
    if (S.pass_open == 2 && pass_whole(r.rect) && (!(as & VK_IMAGE_ASPECT_STENCIL_BIT) || (c->stencil_mask & 0xff) == 0xff)) {   // into the load ops
        if (as & VK_IMAGE_ASPECT_DEPTH_BIT) { P_da.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR; P_da.clearValue.depthStencil = a.clearValue.depthStencil; }
        if (as & VK_IMAGE_ASPECT_STENCIL_BIT) { P_sa.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR; P_sa.clearValue.depthStencil = a.clearValue.depthStencil; }
        P_cleared = 1; g_vkst.folded++; return;
    }
    if (!pass_commit()) return;
    if (r.rect.extent.width && r.rect.extent.height) vkCmdClearAttachments(vk_cmd(), 1, &a, 1, &r);
}

// ---------------- formats and enums ----------------
static VkFormat vertex_format(int f, int want_int, int want_sint) {
    static const uint16_t rgb[] = {VK_FORMAT_R8G8B8_UNORM, VK_FORMAT_R8G8B8_SNORM, VK_FORMAT_R8G8B8_UINT, VK_FORMAT_R8G8B8_SINT, VK_FORMAT_R16G16B16_SFLOAT, VK_FORMAT_R16G16B16_UNORM,
        VK_FORMAT_R16G16B16_SNORM, VK_FORMAT_R16G16B16_UINT, VK_FORMAT_R16G16B16_SINT, VK_FORMAT_R32G32B32_SFLOAT, VK_FORMAT_R32G32B32_UINT, VK_FORMAT_R32G32B32_SINT};
    static const uint16_t rgb_scaled[] = {VK_FORMAT_R8G8B8_USCALED, VK_FORMAT_R8G8B8_SSCALED, VK_FORMAT_R16G16B16_USCALED, VK_FORMAT_R16G16B16_SSCALED, VK_FORMAT_R32G32B32_UINT, VK_FORMAT_R32G32B32_SINT};
    VkFormat v;
    if (f >= 0x19 && f <= 0x24) v = (VkFormat)rgb[f - 0x19];
    else if (f >= 0x5D && f <= 0x62) v = (VkFormat)rgb_scaled[f - 0x5D];
    else { const vkfmt_t *e = vk_fmt(f); v = e ? e->vk : VK_FORMAT_UNDEFINED; }
    static uint8_t checked[0x100];   // a format the device cannot read as a vertex attribute gives undefined values: named once
    if (v != VK_FORMAT_UNDEFINED && f > 0 && f < 0x100 && !checked[f]) { checked[f] = 1; VkFormatProperties fp; vkGetPhysicalDeviceFormatProperties(V.pd, v, &fp);
        if (!(fp.bufferFeatures & VK_FORMAT_FEATURE_VERTEX_BUFFER_BIT)) VKLOG("[RED] vertex attribute format %#x (vk %d): the device cannot read it as a vertex attribute", f, v); }
    if (want_int) {   // the shader declared an integer input: the attribute must be an integer format of the same layout
        switch (v) {
        case VK_FORMAT_R32_SFLOAT: return want_sint ? VK_FORMAT_R32_SINT : VK_FORMAT_R32_UINT;
        case VK_FORMAT_R32G32_SFLOAT: return want_sint ? VK_FORMAT_R32G32_SINT : VK_FORMAT_R32G32_UINT;
        case VK_FORMAT_R32G32B32_SFLOAT: return want_sint ? VK_FORMAT_R32G32B32_SINT : VK_FORMAT_R32G32B32_UINT;
        case VK_FORMAT_R32G32B32A32_SFLOAT: return want_sint ? VK_FORMAT_R32G32B32A32_SINT : VK_FORMAT_R32G32B32A32_UINT;
        case VK_FORMAT_R8G8B8A8_UNORM: case VK_FORMAT_R8G8B8A8_USCALED: return want_sint ? VK_FORMAT_R8G8B8A8_SINT : VK_FORMAT_R8G8B8A8_UINT;
        case VK_FORMAT_R16G16_UNORM: case VK_FORMAT_R16G16_USCALED: return want_sint ? VK_FORMAT_R16G16_SINT : VK_FORMAT_R16G16_UINT;
        case VK_FORMAT_R16G16B16A16_UNORM: case VK_FORMAT_R16G16B16A16_USCALED: return want_sint ? VK_FORMAT_R16G16B16A16_SINT : VK_FORMAT_R16G16B16A16_UINT;
        default: return v;
        }
    }
    return v;
}
static const VkBlendFactor BF[0x65] = {
    [1] = VK_BLEND_FACTOR_ZERO, [2] = VK_BLEND_FACTOR_ONE, [3] = VK_BLEND_FACTOR_SRC_COLOR, [4] = VK_BLEND_FACTOR_ONE_MINUS_SRC_COLOR, [5] = VK_BLEND_FACTOR_SRC_ALPHA,
    [6] = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA, [7] = VK_BLEND_FACTOR_DST_ALPHA, [8] = VK_BLEND_FACTOR_ONE_MINUS_DST_ALPHA, [9] = VK_BLEND_FACTOR_DST_COLOR,
    [10] = VK_BLEND_FACTOR_ONE_MINUS_DST_COLOR, [11] = VK_BLEND_FACTOR_SRC_ALPHA_SATURATE, [0x10] = VK_BLEND_FACTOR_SRC1_COLOR, [0x11] = VK_BLEND_FACTOR_ONE_MINUS_SRC1_COLOR,
    [0x12] = VK_BLEND_FACTOR_SRC1_ALPHA, [0x13] = VK_BLEND_FACTOR_ONE_MINUS_SRC1_ALPHA, [0x61] = VK_BLEND_FACTOR_CONSTANT_COLOR, [0x62] = VK_BLEND_FACTOR_ONE_MINUS_CONSTANT_COLOR,
    [0x63] = VK_BLEND_FACTOR_CONSTANT_ALPHA, [0x64] = VK_BLEND_FACTOR_ONE_MINUS_CONSTANT_ALPHA};
static VkBlendFactor bfac(unsigned f) { return f < 0x65 && (f == 1 || BF[f]) ? BF[f] : VK_BLEND_FACTOR_ONE; }
static VkBlendOp bop(unsigned e) { return e >= 1 && e <= 5 ? (VkBlendOp)(e - 1) : VK_BLEND_OP_ADD; }
static VkCompareOp cmpop(unsigned f) { return f >= 1 && f <= 8 ? (VkCompareOp)(f - 1) : VK_COMPARE_OP_ALWAYS; }
static VkStencilOp sop(unsigned o) { return o >= 1 && o <= 8 ? (VkStencilOp)(o - 1) : VK_STENCIL_OP_KEEP; }
static VkPrimitiveTopology topo(int p) {
    switch (p) { case 0: return VK_PRIMITIVE_TOPOLOGY_POINT_LIST; case 1: return VK_PRIMITIVE_TOPOLOGY_LINE_LIST; case 2: case 3: return VK_PRIMITIVE_TOPOLOGY_LINE_STRIP;
                 case 5: case 8: return VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP; case 6: case 9: return VK_PRIMITIVE_TOPOLOGY_TRIANGLE_FAN;
                 case 0xA: return VK_PRIMITIVE_TOPOLOGY_LINE_LIST_WITH_ADJACENCY; case 0xB: return VK_PRIMITIVE_TOPOLOGY_LINE_STRIP_WITH_ADJACENCY;
                 case 0xC: return VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST_WITH_ADJACENCY; case 0xD: return VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP_WITH_ADJACENCY; case 0xE: return VK_PRIMITIVE_TOPOLOGY_PATCH_LIST;
                 default: return VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST; }
}

// ---------------- advanced blend modes ----------------
// NVN's SetAdvancedMode (NV_blend_equation_advanced: multiply, screen, overlay, ...) with its premultiplied source (adv_npm: SetAdvancedPremultipliedSrc(false)) and overlap (adv_ovl), kept
// in the blend state as upstream's hle_nvn.c records it (blend_st.adv; this fork's resolved the expressible modes at bind and cleared it, which reaches here as no mode). Per colour target,
// the first of these that holds:
//  · a Porter-Duff mode with uncorrelated overlap, a sum of source and destination weighted by 1, 0, As, Ad and their complements: exact with fixed-function factors and ADD, a source that
//    is not premultiplied taken through SRC_ALPHA where its factor is ONE (nvn_mtl.m adv_blend, the same table);
//  · VK_EXT_blend_operation_advanced (coherent operations), where the device takes the mode, the source's premultiplication, the overlap and the target (V.adv_all, adv_npm, adv_ovl,
//    adv_max_att): every mode as NVN draws it;
//  · MULTIPLY and SCREEN of a premultiplied source with uncorrelated overlap, exact over an opaque destination (Ad = 1, a colour target's usual alpha) with plain functions: Viractal's
//    mode 21, as hle_nvn.c blend_adv resolves it;
//  · else the state's own functions, a [RED] line per mode
enum { ADV_KEY = 0x10000 };   // pkey bl[].eq: an advanced operation, its NVN mode in the low byte; bl[].eqa: the source not premultiplied (bit 0), the overlap (bits 1-2)
static VkBlendOp adv_vk_op(int m) {
    switch (m) {
    case NVN_BLEND_ADVANCED_MODE_ZERO: return VK_BLEND_OP_ZERO_EXT; case NVN_BLEND_ADVANCED_MODE_SRC: return VK_BLEND_OP_SRC_EXT; case NVN_BLEND_ADVANCED_MODE_DST: return VK_BLEND_OP_DST_EXT;
    case NVN_BLEND_ADVANCED_MODE_SRC_OVER: return VK_BLEND_OP_SRC_OVER_EXT; case NVN_BLEND_ADVANCED_MODE_DST_OVER: return VK_BLEND_OP_DST_OVER_EXT; case NVN_BLEND_ADVANCED_MODE_SRC_IN: return VK_BLEND_OP_SRC_IN_EXT;
    case NVN_BLEND_ADVANCED_MODE_DST_IN: return VK_BLEND_OP_DST_IN_EXT; case NVN_BLEND_ADVANCED_MODE_SRC_OUT: return VK_BLEND_OP_SRC_OUT_EXT; case NVN_BLEND_ADVANCED_MODE_DST_OUT: return VK_BLEND_OP_DST_OUT_EXT;
    case NVN_BLEND_ADVANCED_MODE_SRC_ATOP: return VK_BLEND_OP_SRC_ATOP_EXT; case NVN_BLEND_ADVANCED_MODE_DST_ATOP: return VK_BLEND_OP_DST_ATOP_EXT; case NVN_BLEND_ADVANCED_MODE_XOR: return VK_BLEND_OP_XOR_EXT;
    case NVN_BLEND_ADVANCED_MODE_PLUS: return VK_BLEND_OP_PLUS_EXT; case NVN_BLEND_ADVANCED_MODE_PLUS_CLAMPED: return VK_BLEND_OP_PLUS_CLAMPED_EXT; case NVN_BLEND_ADVANCED_MODE_PLUS_CLAMPED_ALPHA: return VK_BLEND_OP_PLUS_CLAMPED_ALPHA_EXT;
    case NVN_BLEND_ADVANCED_MODE_PLUS_DARKER: return VK_BLEND_OP_PLUS_DARKER_EXT; case NVN_BLEND_ADVANCED_MODE_MULTIPLY: return VK_BLEND_OP_MULTIPLY_EXT; case NVN_BLEND_ADVANCED_MODE_SCREEN: return VK_BLEND_OP_SCREEN_EXT;
    case NVN_BLEND_ADVANCED_MODE_OVERLAY: return VK_BLEND_OP_OVERLAY_EXT; case NVN_BLEND_ADVANCED_MODE_DARKEN: return VK_BLEND_OP_DARKEN_EXT; case NVN_BLEND_ADVANCED_MODE_LIGHTEN: return VK_BLEND_OP_LIGHTEN_EXT;
    case NVN_BLEND_ADVANCED_MODE_COLORDODGE: return VK_BLEND_OP_COLORDODGE_EXT; case NVN_BLEND_ADVANCED_MODE_COLORBURN: return VK_BLEND_OP_COLORBURN_EXT; case NVN_BLEND_ADVANCED_MODE_HARDLIGHT: return VK_BLEND_OP_HARDLIGHT_EXT;
    case NVN_BLEND_ADVANCED_MODE_SOFTLIGHT: return VK_BLEND_OP_SOFTLIGHT_EXT; case NVN_BLEND_ADVANCED_MODE_DIFFERENCE: return VK_BLEND_OP_DIFFERENCE_EXT; case NVN_BLEND_ADVANCED_MODE_MINUS: return VK_BLEND_OP_MINUS_EXT;
    case NVN_BLEND_ADVANCED_MODE_MINUS_CLAMPED: return VK_BLEND_OP_MINUS_CLAMPED_EXT; case NVN_BLEND_ADVANCED_MODE_EXCLUSION: return VK_BLEND_OP_EXCLUSION_EXT; case NVN_BLEND_ADVANCED_MODE_CONTRAST: return VK_BLEND_OP_CONTRAST_EXT;
    case NVN_BLEND_ADVANCED_MODE_INVERT: return VK_BLEND_OP_INVERT_EXT; case NVN_BLEND_ADVANCED_MODE_INVERT_RGB: return VK_BLEND_OP_INVERT_RGB_EXT; case NVN_BLEND_ADVANCED_MODE_INVERT_OVG: return VK_BLEND_OP_INVERT_OVG_EXT;
    case NVN_BLEND_ADVANCED_MODE_LINEARDODGE: return VK_BLEND_OP_LINEARDODGE_EXT; case NVN_BLEND_ADVANCED_MODE_LINEARBURN: return VK_BLEND_OP_LINEARBURN_EXT; case NVN_BLEND_ADVANCED_MODE_VIVIDLIGHT: return VK_BLEND_OP_VIVIDLIGHT_EXT;
    case NVN_BLEND_ADVANCED_MODE_LINEARLIGHT: return VK_BLEND_OP_LINEARLIGHT_EXT; case NVN_BLEND_ADVANCED_MODE_PINLIGHT: return VK_BLEND_OP_PINLIGHT_EXT; case NVN_BLEND_ADVANCED_MODE_HARDMIX: return VK_BLEND_OP_HARDMIX_EXT;
    case NVN_BLEND_ADVANCED_MODE_RED: return VK_BLEND_OP_RED_EXT; case NVN_BLEND_ADVANCED_MODE_GREEN: return VK_BLEND_OP_GREEN_EXT; case NVN_BLEND_ADVANCED_MODE_BLUE: return VK_BLEND_OP_BLUE_EXT;
    case NVN_BLEND_ADVANCED_MODE_HSL_HUE: return VK_BLEND_OP_HSL_HUE_EXT; case NVN_BLEND_ADVANCED_MODE_HSL_SATURATION: return VK_BLEND_OP_HSL_SATURATION_EXT; case NVN_BLEND_ADVANCED_MODE_HSL_COLOR: return VK_BLEND_OP_HSL_COLOR_EXT;
    case NVN_BLEND_ADVANCED_MODE_HSL_LUMINOSITY: return VK_BLEND_OP_HSL_LUMINOSITY_EXT;
    default: return VK_BLEND_OP_MAX_ENUM;
    }
}
static int adv_basic(int m) { return (m >= NVN_BLEND_ADVANCED_MODE_MULTIPLY && m <= NVN_BLEND_ADVANCED_MODE_DIFFERENCE) || m == NVN_BLEND_ADVANCED_MODE_EXCLUSION || (m >= NVN_BLEND_ADVANCED_MODE_HSL_HUE && m <= NVN_BLEND_ADVANCED_MODE_HSL_LUMINOSITY); }   // KHR_blend_equation_advanced's set: a device without advancedBlendAllOperations takes only these
// target i's blend with b's advanced mode resolved into o (src, dst, srca, dsta, eq, eqa as pkey bl[]); 0 when nothing expresses it (o keeps the state's functions)
static int adv_blend(const blend_st *b, int i, uint32_t o[6]) {
    enum { Z = NVN_BLEND_FUNC_ZERO, O = NVN_BLEND_FUNC_ONE, SA = NVN_BLEND_FUNC_SRC_ALPHA, ISA = NVN_BLEND_FUNC_ONE_MINUS_SRC_ALPHA, DA = NVN_BLEND_FUNC_DST_ALPHA, IDA = NVN_BLEND_FUNC_ONE_MINUS_DST_ALPHA,
           DC = NVN_BLEND_FUNC_DST_COLOR, IDC = NVN_BLEND_FUNC_ONE_MINUS_DST_COLOR };
    static const uint8_t pd[0x14][4] = {   // {source factor, destination factor} for colour, then for alpha (nvn_mtl.m adv_blend)
        [NVN_BLEND_ADVANCED_MODE_ZERO] = {Z, Z, Z, Z}, [NVN_BLEND_ADVANCED_MODE_SRC] = {O, Z, O, Z}, [NVN_BLEND_ADVANCED_MODE_DST] = {Z, O, Z, O}, [NVN_BLEND_ADVANCED_MODE_SRC_OVER] = {O, ISA, O, ISA},
        [NVN_BLEND_ADVANCED_MODE_DST_OVER] = {IDA, O, IDA, O}, [NVN_BLEND_ADVANCED_MODE_SRC_IN] = {DA, Z, DA, Z}, [NVN_BLEND_ADVANCED_MODE_DST_IN] = {Z, SA, Z, SA}, [NVN_BLEND_ADVANCED_MODE_SRC_OUT] = {IDA, Z, IDA, Z},
        [NVN_BLEND_ADVANCED_MODE_DST_OUT] = {Z, ISA, Z, ISA}, [NVN_BLEND_ADVANCED_MODE_SRC_ATOP] = {DA, ISA, Z, O}, [NVN_BLEND_ADVANCED_MODE_DST_ATOP] = {IDA, SA, O, Z}, [NVN_BLEND_ADVANCED_MODE_XOR] = {IDA, ISA, IDA, ISA},
        [NVN_BLEND_ADVANCED_MODE_PLUS] = {O, O, O, O}, [NVN_BLEND_ADVANCED_MODE_PLUS_CLAMPED] = {O, O, O, O}, [NVN_BLEND_ADVANCED_MODE_PLUS_CLAMPED_ALPHA] = {O, O, O, O}};
    int m = (int)b->adv, npm = b->adv_npm, ovl = (int)b->adv_ovl;
    const uint8_t *f = m < 0x14 && pd[m][0] ? pd[m] : NULL;
    if (f && !ovl && (!npm || f[0] == O || f[0] == Z)) { o[0] = npm && f[0] == O ? SA : f[0]; o[1] = f[1]; o[2] = f[2]; o[3] = f[3]; o[4] = o[5] = NVN_BLEND_EQUATION_ADD; return 1; }
    if (V.has_adv_blend && adv_vk_op(m) != VK_BLEND_OP_MAX_ENUM && (V.adv_all || adv_basic(m)) && (!npm || V.adv_npm) && (!ovl || V.adv_ovl) && (uint32_t)i < V.adv_max_att) {
        o[4] = ADV_KEY | (uint32_t)m; o[5] = (uint32_t)npm | (uint32_t)ovl << 1; o[0] = o[1] = o[2] = o[3] = O; return 1; }
    if ((m == NVN_BLEND_ADVANCED_MODE_MULTIPLY || m == NVN_BLEND_ADVANCED_MODE_SCREEN) && !npm && !ovl) {
        o[0] = m == NVN_BLEND_ADVANCED_MODE_MULTIPLY ? DC : IDC; o[1] = m == NVN_BLEND_ADVANCED_MODE_MULTIPLY ? ISA : O; o[2] = O; o[3] = ISA; o[4] = o[5] = NVN_BLEND_EQUATION_ADD; return 1; }
    static uint64_t seen; if (!(seen >> (m & 63) & 1)) { seen |= 1ull << (m & 63);
        LOG("[RED] [vk] advanced blend mode %#x (overlap %d, %spremultiplied source, target %d) has no Vulkan equivalent here%s: the state's blend functions are used", m, ovl, npm ? "not " : "", i,
            V.has_adv_blend ? "" : " (no VK_EXT_blend_operation_advanced)"); }
    return 0;
}

// ---------------- programs and pipelines ----------------
static uint64_t key64(const uint8_t k[8]) { uint64_t v = 0; for (int i = 0; i < 8; i++) v = v << 8 | k[i]; return v; }
// DM_NVN_SHDUMP (debugging; hle_nvn.c then keeps each program's control sections): a shader missing from the bundle is written to <files>/shader_<key>.bin as its 0x880-byte
// control section followed by its code area (control +0x704 bytes), so it can be looked for in the game files (Xenoblade Chronicles 3's title had one such program)
static void shader_dump(const prog_t *p, unsigned i, uint64_t kk) {
    static int on = -1; if (on < 0) on = getenv("DM_NVN_SHDUMP") != NULL;
    if (!on || i >= p->nstage || !p->sd[i].control || !p->sd[i].data) return;
    uint32_t ncode; memcpy(&ncode, (const uint8_t *)p->sd[i].control + 0x704, 4);
    const char *dir = getenv("DM_ROOT"); char path[1024]; snprintf(path, sizeof path, "%s/shader_%016llx.bin", dir ? dir : ".", (unsigned long long)kk);
    FILE *f = fopen(path, "wb"); if (!f) return;
    fwrite(p->sd[i].control, 1, 0x880, f); if (ncode < (1u << 24)) fwrite((const void *)(uintptr_t)p->sd[i].data, 1, ncode, f); fclose(f);
    VKLOG("shader %016llx written to %s", (unsigned long long)kk, path);
}
static void bind_program(const p_bind_program *bp) {
    // NVN binds by stage mask (vertex 1, fragment 2, geometry 4, tessellation 8 / 0x10, compute 0x20): only the masked stages change, so a game that keeps
    // its vertex and fragment shaders in programs of their own (NEO, AI: THE SOMNIUM FILES) binds them one after the other and draws with both.
    // prog_key's index is the position in the program's shader array (as the game passed them to SetShaders), not the stage: the stage is the bundle's.
    // NVN2's task and mesh stages (bits 0x40, 0x80) take the tessellation stages' places, 3 and 4, as their bindings do (hle_nvn.c st_host; nvn_mtl.m gfx_stage_bits): a program has
    // mesh shaders or tessellation, never both. A bundle that files them as stages 6 / 7 gets the same places
    int vk_shader_stage(uint64_t key);
    uint32_t m = bp->stages ? (uint32_t)bp->stages : 0xff, mask = (m & 0x3f) | (m & 0x40 ? 8 : 0) | (m & 0x80 ? 16 : 0);
    for (int st = 0; st < 6; st++) if (mask >> st & 1) S.sh[st] = NULL;
    S.prog = bp->prog;
    // What a bind gives, by program, serial and stage mask: a program's shaders are fixed until it is given others (SetShaders gives it a new serial), and each bind looked its
    // keys up again (hle_nvn.c prog_key: a lock and a dictionary per stage), the bundle (vk_shader_stage, a binary search over its 11,603 entries) and the shader table (a lock
    // and a hash) per stage: 5% of the replay thread at Mario Kart World's ~800 program binds a frame (vkbench). DM_VK_PROG_CACHE=0 (start-up): looked up at every bind
    enum { PGC = 1024 }; static struct { const prog_t *p; uint64_t serial; uint32_t mask; vkshader_t *sh[6]; } pgc[PGC]; static int pgc_on = -1;
    if (pgc_on < 0) { const char *e = getenv("DM_VK_PROG_CACHE"); pgc_on = !(e && !atoi(e)); }
    unsigned pk = S.prog ? (unsigned)((((uintptr_t)S.prog >> 4) ^ S.prog->serial * 0x9E3779B97F4A7C15ull ^ mask) >> 7) % PGC : 0;
    if (S.prog && pgc_on && pgc[pk].p == S.prog && pgc[pk].serial == S.prog->serial && pgc[pk].mask == mask) {
        for (int st = 0; st < 6; st++) if (mask >> st & 1) S.sh[st] = pgc[pk].sh[st];
        S.prog_ok = S.sh[0] || S.sh[5]; return; }
    if (S.prog) for (unsigned i = 0; i < 6; i++) {
        uint8_t k[8]; if (!prog_key(S.prog, i, k)) break;
        uint64_t kk = key64(k); int st = kk ? vk_shader_stage(kk) : -1;
        if (kk && st < 0) {   // the draw or dispatch is skipped. Logged once per shader (the first 16 binds were 4 shaders of Xenoblade Chronicles 3 over and over, which hid others)
            static uint64_t seen[256]; static int nseen; int known = 0; for (int j = 0; j < nseen; j++) if (seen[j] == kk) { known = 1; break; }
            if (!known && nseen < 256) { seen[nseen++] = kk; void vk_shader_missing(uint64_t key); vk_shader_missing(kk); shader_dump(S.prog, i, kk); } }
        int pl = st == 6 ? 3 : st == 7 ? 4 : st;
        if (pl < 0 || pl > 5 || !(mask >> pl & 1)) continue;
        S.sh[pl] = vk_shader(kk, st);
    }
    S.prog_ok = S.sh[0] || S.sh[5];   // a vertex-only program draws depth only (prepare_draw)
    if (S.prog && pgc_on) { pgc[pk].p = S.prog; pgc[pk].serial = S.prog->serial; pgc[pk].mask = mask; memcpy(pgc[pk].sh, S.sh, sizeof S.sh); }
}
typedef struct {
    uint64_t key[3]; uint32_t topo, restart, raster_discard, cull, front, mode, depth_clamp, bias_en;
    uint32_t dt, dw, df, st, sf[2], sfail[2], szfail[2], szpass[2];
    uint32_t nattr; struct { uint32_t loc, binding, fmt, off; } attr[32];   // 32: NVN2's attributes (a changed key starts a new prewarm file, PIPE_MAGIC's header)
    uint32_t nbind; struct { uint32_t binding, stride, rate, divisor; } bind[17];
    uint32_t ncol, cfmt[8], dfmt, samples, a2c, blend_en, logic_en, logic_op, wmask[8];
    struct { uint32_t src, dst, srca, dsta, eq, eqa; } bl[8];
    uint32_t clip_neg1, alt, fdz, dual;   // fdz: the fragment shader's variant that writes gl_FragDepth = gl_FragCoord.z (fdz below); dual: its dual-source variant (below)
    uint64_t um[3], im[3];   // each stage's sampled bindings given integer views (vk_shader_module)
    uint32_t mesh, atest;    // mesh: a task / mesh pipeline (mesh_draw): key[0] the mesh shader, key[2] the task shader (0: none), no vertex input; bits 2, 4, 8 its A/B variants (DM_VK_MESH_NOTASK,
                             // _POSBLOCK, _VISTATE: g_mesh_nofdz above). atest: the fragment shader's alpha test variant (prepare_draw; a prewarm row from before it reads as 0, the test off)
    uint64_t lm[3];          // each stage's bindings whose view starts B levels into its texture and whose level numbers it reads (vk_lodshift_mask; gfx_vk_shader.c spv_lod_shift: B as data)
} pkey;
// The stages a draw is made of, by pipeline place (pkey key[]: vertex, fragment, geometry), and the NVN stage whose bindings each one reads. A DrawMeshTasks being prepared (g_mesh,
// mesh_draw) puts its mesh shader in the vertex place and its task shader in the geometry place: their bindings are stages 4 and 3 (hle_nvn.c st_host)
static struct { int on, notask; vkshader_t *ts, *ms; uint32_t first, count; } g_mesh;   // notask: the program's task shader left out (DM_VK_MESH_NOTASK)
static inline vkshader_t *dsh(int i) { return g_mesh.on ? (i == 0 ? g_mesh.ms : i == 2 ? g_mesh.ts : S.sh[1]) : S.sh[i]; }
static inline int dst_of(int i) { return g_mesh.on ? (i == 0 ? 4 : i == 2 ? 3 : 1) : i; }
static nvn_hmap g_pipes = {.klen = sizeof(pkey)}; static pthread_mutex_t g_pipe_mu = PTHREAD_MUTEX_INITIALIZER;   // the replay thread and the prewarm threads
static VkPipeline build_pipeline(const pkey *k, VkPipelineLayout layout, vkshader_t *const stg[3]);
static void pipe_record(const pkey *k);
// The last key and its pipeline (the replay thread's only; pipelines are never dropped): runs of draws with one pipeline skip the 992-byte hash and the lock
static pkey g_last_k; static VkPipeline g_last_p;
// Pipelines built in the background. A pipeline met for the first time costs the driver's compile on the replay thread, 20-35 ms each on Adreno 830 (no pipeline
// libraries there): in Super Mario Odyssey the first cap throw at a breakable object, the first swim, made 2-7 new pipelines a frame (175 ms frames, seen as 40, 30
// fps or lower for a moment), then never again in that run; the prewarm (below) covers what an earlier run met. A draw whose pipeline does not exist yet is queued for
// a background thread and left out until it is built, a few frames, when every target of its pass was drawn in each of the two presents before: a target drawn every
// frame shows it a few frames late. Not two presents at any time: an image kept from the title screen and drawn once at a scene's load lost SMO's fog for the run. A target drawn for the first time (an environment map made once) builds at once, as a draw left out there would stay missing, and so
// do rasterizer-discard and transform-feedback draws (they write memory later draws read). Age goes by the image: counted by size and format instead, SMO's first scene
// after Resume lost its coin counter and tutorial movie for good and showed the Cap Kingdom fog unparted (targets drawn once, in classes the title screen had aged).
// So a capture's new full-screen targets still build at once (SMO's first frog capture: 33 pipelines, two 300 ms frames, once). So do passes into a linear render target:
// the CPU reads it (lin_back), and SMO's fog wave solver feeds what it reads into the next step, so one step left out lost the Cap Kingdom fog for the run.
// DM_VK_ASYNC_PIPE=0 (live): every pipeline is built at once
#define PIPE_PENDING ((VkPipeline)(uintptr_t)2)
enum { APQ = 256 };
static pkey g_apq[APQ]; static int g_apq_h, g_apq_t, g_apq_on;   // queued keys (a draw left out counts as skipped for no pipeline)
static pthread_mutex_t g_apq_mu = PTHREAD_MUTEX_INITIALIZER; static pthread_cond_t g_apq_cv = PTHREAD_COND_INITIALIZER;
static int pk_stage(const pkey *k, int s) { int vk_shader_stage(uint64_t key); return k->mesh && s != 1 ? vk_shader_stage(k->key[s]) : s; }   // the stage a key's shader was made with (bind_program: the bundle's)
static void *async_pipe_thread(void *arg) {
    (void)arg; pthread_setname_np(pthread_self(), "vk async pipe");
    for (;;) {
        pthread_mutex_lock(&g_apq_mu); while (g_apq_h == g_apq_t) pthread_cond_wait(&g_apq_cv, &g_apq_mu);
        pkey k = g_apq[g_apq_t % APQ]; g_apq_t++; pthread_mutex_unlock(&g_apq_mu);
        vkshader_t *stg[3] = {0}; VkPipeline p = (VkPipeline)1;
        for (int s = 0; s < 3; s++) if (k.key[s] && !(s == 1 && k.raster_discard)) stg[s] = vk_shader(k.key[s], pk_stage(&k, s));
        if (stg[0]) { VkDescriptorSetLayout dsl[5] = {0}; int nsets = 0;
            for (int s = 0; s < 3; s++) if (stg[s]) { dsl[stg[s]->set] = stg[s]->dsl; if (stg[s]->set + 1 > nsets) nsets = stg[s]->set + 1; }
            uint64_t t0 = tick(); p = build_pipeline(&k, vk_pipeline_layout_gfx(dsl, nsets), stg);
            { static int n; if (n++ < 6) VKLOG("pipeline built in the background in %.1f ms (shaders %016llx %016llx)", (tick() - t0) / 1e6, (unsigned long long)k.key[0], (unsigned long long)k.key[1]); } }
        pthread_mutex_lock(&g_pipe_mu); nvn_hm_put(&g_pipes, &k, p); if (p != (void *)1) pipe_record(&k); pthread_mutex_unlock(&g_pipe_mu);
    }
    return NULL;
}
static int async_pipe(const pkey *k) {   // 1: queued (the draw is left out); under nothing (takes g_pipe_mu)
    if (!g_async_pipe || k->raster_discard || k->mesh || (S.sh[0] && S.sh[0]->xfb)) return 0;   // (task / mesh pipelines: at once, they are few)
    for (int i = 0; i < S.ncol && i < 8; i++) if (S.rt_has_col[i] && (S.rt_col[i].d.flags & NVN_TEXTURE_FLAGS_LINEAR_RENDER_TARGET)) return 0;   // (the CPU reads it: lin_back)
    int natt = 0; for (int i = 0; i < 9; i++) { const vkimg_t *a = P_att[i].im; if (!a) continue; natt++; if (a->rt_frames < 3) {
            if (g_prof_on) { static int n; if (n++ < 300) VKLOG("pipeline built here: attachment %d %ux%u nvn format %#x drawn in %llu presents in a row (shaders %016llx %016llx)", i, a->w, a->h, a->nvn_format, (unsigned long long)a->rt_frames, (unsigned long long)k->key[0], (unsigned long long)k->key[1]); }
            return 0; } }   // (counted in this present too)
    if (!natt) return 0;
    pthread_mutex_lock(&g_apq_mu);
    if (!g_apq_on) { g_apq_on = 1; for (int i = 0; i < 2; i++) { pthread_t t; pthread_attr_t a; pthread_attr_init(&a); pthread_attr_setdetachstate(&a, PTHREAD_CREATE_DETACHED); pthread_create(&t, &a, async_pipe_thread, NULL); pthread_attr_destroy(&a); } }
    int ok = g_apq_h - g_apq_t < APQ; if (ok) { g_apq[g_apq_h % APQ] = *k; g_apq_h++; pthread_cond_signal(&g_apq_cv); }
    pthread_mutex_unlock(&g_apq_mu);
    if (ok) { pthread_mutex_lock(&g_pipe_mu); if (!nvn_hm_get(&g_pipes, k)) nvn_hm_put(&g_pipes, k, PIPE_PENDING); pthread_mutex_unlock(&g_pipe_mu); }
    return ok;
}
static VkPipeline graphics_pipeline(const pkey *k, VkPipelineLayout layout) {
    if (g_last_p && !memcmp(k, &g_last_k, sizeof *k)) return g_last_p == (void *)1 ? VK_NULL_HANDLE : g_last_p;
    pthread_mutex_lock(&g_pipe_mu); VkPipeline p = nvn_hm_get(&g_pipes, k); pthread_mutex_unlock(&g_pipe_mu);
    if (p == PIPE_PENDING) return VK_NULL_HANDLE;
    if (p) { g_last_k = *k; g_last_p = p; return p == (void *)1 ? VK_NULL_HANDLE : p; }
    if (async_pipe(k)) return VK_NULL_HANDLE;
    vkshader_t *stg[3] = {dsh(0), dsh(1), dsh(2)};
    uint64_t tb = g_prof_on ? tick() : 0; p = build_pipeline(k, layout, stg); if (g_prof_on) g_prof_ns[0] += tick() - tb; g_vkst.pipes_built++;
    pthread_mutex_lock(&g_pipe_mu);
    VkPipeline had = nvn_hm_get(&g_pipes, k);
    if (had) { if (p != (void *)1) vkDestroyPipeline(V.dev, p, NULL); p = had; }   // a prewarm thread made it meanwhile
    else { nvn_hm_put(&g_pipes, k, p); if (p != (void *)1) pipe_record(k); }
    pthread_mutex_unlock(&g_pipe_mu);
    return p == (void *)1 ? VK_NULL_HANDLE : p;
}
// ---- pipeline prewarm: every pipeline the game needed is recorded (DM_ROOT/vk_pipelines.bin) and made again at the next boot on background
// threads, before the game reaches it, so a scene seen once never stalls a frame on a pipeline compile again (the driver's compile, not the
// pipeline cache's lookup, is the cost: tens of milliseconds per pipeline on Adreno). DM_VK_NOPREWARM=1 turns it off.
#define PIPE_MAGIC 0x31455050u   // "PPE1"; the header also holds sizeof(pkey), so a changed key layout starts a new file
static FILE *g_prec;
// The records are written by a thread of their own: a flush to the app's files folder (FUSE) took tens of milliseconds, and made under g_pipe_mu it held up the replay's
// pipeline lookups (a 66 ms frame with no pipeline built in it)
enum { PREC_Q = 64 };
static pkey g_precq[PREC_Q]; static int g_precq_n, g_precw_on; static pthread_mutex_t g_precq_mu = PTHREAD_MUTEX_INITIALIZER; static pthread_cond_t g_precq_cv = PTHREAD_COND_INITIALIZER;
static void *pipe_record_thread(void *arg) {
    (void)arg; pthread_setname_np(pthread_self(), "vk pipe record"); static pkey b[PREC_Q];
    for (;;) {
        pthread_mutex_lock(&g_precq_mu); while (!g_precq_n) pthread_cond_wait(&g_precq_cv, &g_precq_mu);
        int n = g_precq_n; memcpy(b, g_precq, sizeof *b * (size_t)n); g_precq_n = 0; pthread_mutex_unlock(&g_precq_mu);
        fwrite(b, sizeof *b, (size_t)n, g_prec); fflush(g_prec);
    }
    return NULL;
}
static void pipe_record(const pkey *k) {   // (under g_pipe_mu: only queued here)
    if (!g_prec || k->alt || (k->mesh & ~1u)) return;   // DM_VK_SPV_ALT's pipelines are a debugging run's own, as the task / mesh A/B variants are
    pthread_mutex_lock(&g_precq_mu);
    if (!g_precw_on) { g_precw_on = 1; pthread_t t; pthread_attr_t a; pthread_attr_init(&a); pthread_attr_setdetachstate(&a, PTHREAD_CREATE_DETACHED); pthread_create(&t, &a, pipe_record_thread, NULL); pthread_attr_destroy(&a); }
    if (g_precq_n < PREC_Q) g_precq[g_precq_n++] = *k;   // (a full queue drops the record: that pipeline is only not prewarmed next time)
    pthread_cond_signal(&g_precq_cv); pthread_mutex_unlock(&g_precq_mu);
}
static pkey *g_pw; static size_t g_pw_n; static volatile size_t g_pw_next, g_pw_done;
static void *prewarm_thread(void *arg) {
    (void)arg; pthread_setname_np(pthread_self(), "vk prewarm");
    setpriority(PRIO_PROCESS, 0, 10);   // below the game's threads: prewarm uses idle cores
    for (;;) {
        size_t i = __atomic_fetch_add(&g_pw_next, 1, __ATOMIC_RELAXED); if (i >= g_pw_n) break;
        const pkey *k = &g_pw[i];
        pthread_mutex_lock(&g_pipe_mu); int have = nvn_hm_get(&g_pipes, k) != NULL; pthread_mutex_unlock(&g_pipe_mu);
        if (have) continue;
        vkshader_t *stg[3] = {0}; int ok = 1;
        for (int s = 0; s < 3; s++) if (k->key[s] && !(s == 1 && k->raster_discard)) { stg[s] = vk_shader(k->key[s], pk_stage(k, s)); if (!stg[s] || stg[s]->set >= 5) ok = 0; }
        if (!ok || !stg[0]) continue;
        VkDescriptorSetLayout dsl[5] = {0}; int nsets = 0;
        for (int s = 0; s < 3; s++) if (stg[s]) { dsl[stg[s]->set] = stg[s]->dsl; if (stg[s]->set + 1 > nsets) nsets = stg[s]->set + 1; }
        VkPipeline p = build_pipeline(k, vk_pipeline_layout_gfx(dsl, nsets), stg);
        pthread_mutex_lock(&g_pipe_mu);
        if (!nvn_hm_get(&g_pipes, k)) nvn_hm_put(&g_pipes, k, p); else if (p != (void *)1) vkDestroyPipeline(V.dev, p, NULL);
        pthread_mutex_unlock(&g_pipe_mu);
        size_t d = __atomic_add_fetch(&g_pw_done, 1, __ATOMIC_RELAXED);
        if (d == g_pw_n || d % 500 == 0) VKLOG("prewarm: %zu of %zu pipelines", d, g_pw_n);
    }
    return NULL;
}
static void prewarm_start(void) {
    const char *root = getenv("DM_ROOT"); if (!root) return;
    char path[1024]; snprintf(path, sizeof path, "%s/vk_pipelines.bin", root);
    uint32_t hdr[2] = {PIPE_MAGIC, (uint32_t)sizeof(pkey)}, fh[2] = {0};
    FILE *f = fopen(path, "rb");
    if (f && fread(fh, sizeof fh, 1, f) == 1 && fh[0] == hdr[0] && fh[1] == hdr[1]) {
        fseek(f, 0, SEEK_END); long n = (ftell(f) - (long)sizeof fh) / (long)sizeof(pkey); fseek(f, sizeof fh, SEEK_SET);
        if (n > 0) { g_pw = malloc((size_t)n * sizeof(pkey)); g_pw_n = fread(g_pw, sizeof(pkey), (size_t)n, f); }
    }
    if (f) fclose(f);
    // rewrite the file with the records read (dropping a torn last record), then append to it
    g_prec = fopen(path, "wb"); if (g_prec) { fwrite(hdr, sizeof hdr, 1, g_prec); if (g_pw_n) fwrite(g_pw, sizeof(pkey), g_pw_n, g_prec); fflush(g_prec); }
    if (!g_pw_n || getenv("DM_VK_NOPREWARM")) { VKLOG("prewarm: %zu pipelines recorded%s", g_pw_n, g_pw_n ? ", off (DM_VK_NOPREWARM)" : ""); return; }
    long nc = sysconf(_SC_NPROCESSORS_ONLN); int nt = nc >= 8 ? 3 : nc >= 4 ? 2 : 1;
    VKLOG("prewarm: %zu recorded pipelines on %d threads", g_pw_n, nt);
    for (int i = 0; i < nt; i++) { pthread_t t; pthread_attr_t a; pthread_attr_init(&a); pthread_attr_setdetachstate(&a, PTHREAD_CREATE_DETACHED); pthread_create(&t, &a, prewarm_thread, NULL); pthread_attr_destroy(&a); }
}
// ---- what each thread is building, for the crash dump. Issue #64: Super Mario Bros. Wonder crashed in the Adreno 750's shader compiler (libllvm-qgl.so, driver 512.762)
// at its title scene, with no log line to say which pipeline or which thread; and a pipeline that crashes is never recorded for the prewarm, so every run met it again.
// A slot per thread in a build: the crash dump prints each (the crashing thread's marked), with the shaders and the state that pick the driver's compile
enum { NBUILD = 16 };
static struct { _Atomic int tid; volatile int phase; char what[512]; } g_build[NBUILD];   // phase 0: claimed, 1: making shader modules, 2: in the driver's pipeline compile
static int build_begin(void) {   // the slot, or -1 when all are taken (nothing noted)
    int me = gettid(); for (int i = 0; i < NBUILD; i++) { int z = 0; if (atomic_compare_exchange_strong(&g_build[i].tid, &z, me)) { g_build[i].phase = 0; return i; } }
    return -1;
}
static void build_end(int i) { if (i >= 0) { g_build[i].phase = 0; atomic_store(&g_build[i].tid, 0); } }
static void build_dump(void) {   // (the crash handler: no locks, no allocation)
    int me = gettid(), n = 0;
    for (int i = 0; i < NBUILD; i++) { int t = atomic_load(&g_build[i].tid); if (!t || !g_build[i].phase) continue; n++;
        char nm[64]; crash_thread_name((unsigned)t, nm, sizeof nm);
        fprintf(stderr, "[dm] [vk] %s thread %#x '%s' was %s: %s\n", t == me ? "this (the crashing)" : "another", (unsigned)t, nm,
                g_build[i].phase == 1 ? "making its shader modules" : "in the driver's pipeline compile", g_build[i].what); }
    if (!n) fprintf(stderr, "[dm] [vk] no pipeline was being built\n");
}
static VkPipeline build_pipeline(const pkey *k, VkPipelineLayout layout, vkshader_t *const stg[3]) {
    VkPipeline p;
    int bs = build_begin();
    // the outputs each stage must declare for the next (vk_shader_module): vertex → geometry or fragment, geometry → fragment (a mesh shader's per-vertex outputs are left as they are)
    const vkshader_t *fs = stg[1] && !k->raster_discard ? stg[1] : NULL; uint32_t xo[3] = {0};
    if (stg[0] && !k->mesh) xo[0] = (stg[2] ? stg[2]->in_locs : fs ? fs->in_locs : 0) & ~stg[0]->out_locs;
    if (stg[2] && !k->mesh) xo[2] = (fs ? fs->in_locs : 0) & ~stg[2]->out_locs;
    if (k->mesh && stg[0] && fs) xo[1] = stg[0]->prim_locs & fs->in_locs;   // the fragment shader's inputs a mesh shader writes per primitive (gfx_vk.h prim_locs: vk_shader_module's variant)
    // what is built, for the crash dump (build_dump), and a line of its own before a task / mesh pipeline's compile (run 4 crashed in the driver's compile of the first one)
    char wl[512]; char *what = bs >= 0 ? g_build[bs].what : wl; size_t wn = bs >= 0 ? sizeof g_build[bs].what : sizeof wl;
    if (k->mesh) snprintf(what, wn, "task / mesh pipeline, mesh %016llx (payload private %u, position block %u), task %016llx, fragment %016llx (depth variant %u, per-primitive inputs %#x, "
            "dual source %u, alpha test %u); polygon mode %u, samples %u, %u colour targets (formats %u %u %u %u), depth format %u; raster discard %u, vertex input and assembly states %s, "
            "integer-texel views %#llx/%#llx %#llx/%#llx, alternative code %u", (unsigned long long)k->key[0], k->mesh >> 1 & 1, k->mesh >> 2 & 1, (unsigned long long)k->key[2],
            (unsigned long long)k->key[1], k->fdz, xo[1], k->dual, k->atest, k->mode, k->samples, k->ncol, k->cfmt[0], k->cfmt[1], k->cfmt[2], k->cfmt[3], k->dfmt, k->raster_discard,
            k->mesh & 8 ? "given" : "none", (unsigned long long)k->um[0], (unsigned long long)k->im[0], (unsigned long long)k->um[1], (unsigned long long)k->im[1], k->alt);
    else if (bs >= 0) snprintf(what, wn, "graphics pipeline, shaders %016llx %016llx %016llx; topology %u, polygon mode %u, samples %u, "
            "%u colour targets (formats %u %u %u %u), depth format %u; raster discard %u, fragment depth variant %u, dual source %u, transform feedback %d, logic op %u, "
            "vertex attributes %u in %u bindings, integer-texel views %#llx/%#llx %#llx/%#llx, alternative code %u, alpha test %u",
            (unsigned long long)k->key[0], (unsigned long long)k->key[1], (unsigned long long)k->key[2], k->topo, k->mode, k->samples, k->ncol, k->cfmt[0], k->cfmt[1], k->cfmt[2], k->cfmt[3],
            k->dfmt, k->raster_discard, k->fdz, k->dual, stg[0] ? stg[0]->xfb : 0, k->logic_en ? k->logic_op : 0, k->nattr, k->nbind,
            (unsigned long long)k->um[0], (unsigned long long)k->im[0], (unsigned long long)k->um[1], (unsigned long long)k->im[1], k->alt, k->atest);
    if (bs >= 0) g_build[bs].phase = 1;
    VkPipelineShaderStageCreateInfo ss[3]; uint32_t ns = 0;
    VkShaderStageFlagBits fl[3] = {VK_SHADER_STAGE_VERTEX_BIT, VK_SHADER_STAGE_FRAGMENT_BIT, VK_SHADER_STAGE_GEOMETRY_BIT};
    if (k->mesh) { fl[0] = VK_SHADER_STAGE_MESH_BIT_EXT; fl[2] = VK_SHADER_STAGE_TASK_BIT_EXT; }   // (mesh_draw: the mesh shader in the vertex place, the task shader in the geometry place)
    // a stage's variant (vk_shader_module fd): the fragment shader's depth, dual-source and alpha-test variants; the mesh shader's payload made Private and gl_Position in a block (A/B)
    for (int i = 0; i < 3; i++) if (stg[i] && !(i == 1 && k->raster_discard)) ss[ns++] = (VkPipelineShaderStageCreateInfo){VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, NULL, 0, fl[i],
        vk_shader_module_lod(stg[i], k->um[i], k->im[i], k->alt && !k->um[i] && !k->im[i] && !xo[i] ? 1u << 31 : xo[i], i == 1 ? (k->fdz ? 1 : 0) | (k->dual ? 2 : 0) | (k->atest ? 4 : 0) :
                             i == 0 && k->mesh ? (k->mesh & 2 ? 8 : 0) | (k->mesh & 4 ? 16 : 0) : 0, k->lm[i]), "main", NULL};
    VkVertexInputAttributeDescription ad[32]; VkVertexInputBindingDescription bd[17]; VkVertexInputBindingDivisorDescriptionEXT dd[17]; uint32_t ndd = 0;
    for (uint32_t i = 0; i < k->nattr; i++) ad[i] = (VkVertexInputAttributeDescription){k->attr[i].loc, k->attr[i].binding, (VkFormat)k->attr[i].fmt, k->attr[i].off};
    for (uint32_t i = 0; i < k->nbind; i++) { bd[i] = (VkVertexInputBindingDescription){k->bind[i].binding, k->bind[i].stride, (VkVertexInputRate)k->bind[i].rate};
        if (k->bind[i].rate && k->bind[i].divisor > 1) dd[ndd++] = (VkVertexInputBindingDivisorDescriptionEXT){k->bind[i].binding, k->bind[i].divisor}; }
    VkPipelineVertexInputDivisorStateCreateInfoEXT dvs = {VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_DIVISOR_STATE_CREATE_INFO_EXT, NULL, ndd, dd};
    VkPipelineVertexInputStateCreateInfo vi = {VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO, ndd ? &dvs : NULL, 0, k->nbind, bd, k->nattr, ad};
    VkPipelineInputAssemblyStateCreateInfo ia = {VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO, NULL, 0, (VkPrimitiveTopology)k->topo, k->restart};
    VkPipelineViewportDepthClipControlCreateInfoEXT dcc = {VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_DEPTH_CLIP_CONTROL_CREATE_INFO_EXT, NULL, k->clip_neg1};
    VkPipelineViewportStateCreateInfo vs = {VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO, k->clip_neg1 && V.has_depth_clip_ctl ? &dcc : NULL, 0, 1, NULL, 1, NULL};
    VkPipelineRasterizationStateCreateInfo rs = {VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO, NULL, 0, k->depth_clamp, k->raster_discard,
        k->mode == 0 ? VK_POLYGON_MODE_POINT : k->mode == 1 ? VK_POLYGON_MODE_LINE : VK_POLYGON_MODE_FILL, k->cull, k->front ? VK_FRONT_FACE_COUNTER_CLOCKWISE : VK_FRONT_FACE_CLOCKWISE,
        k->bias_en, 0, 0, 0, 1.0f};
    VkPipelineMultisampleStateCreateInfo ms = {VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO, NULL, 0, (VkSampleCountFlagBits)k->samples, 0, 0, NULL, k->a2c, 0};
    VkStencilOpState sf = {sop(k->sfail[0]), sop(k->szpass[0]), sop(k->szfail[0]), cmpop(k->sf[0]), 0xff, 0xff, 0}, sb = {sop(k->sfail[1]), sop(k->szpass[1]), sop(k->szfail[1]), cmpop(k->sf[1]), 0xff, 0xff, 0};
    VkPipelineDepthStencilStateCreateInfo ds = {VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO, NULL, 0, k->dt, k->dt && k->dw, cmpop(k->df), 0, k->st, sf, sb, 0, 1};
    VkPipelineColorBlendAttachmentState cba[8];
    VkPipelineColorBlendAdvancedStateCreateInfoEXT cadv = {VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_ADVANCED_STATE_CREATE_INFO_EXT, NULL, VK_TRUE, VK_TRUE, VK_BLEND_OVERLAP_UNCORRELATED_EXT}; int nadv = 0;
    for (uint32_t i = 0; i < k->ncol; i++) {
        cba[i] = (VkPipelineColorBlendAttachmentState){(k->blend_en >> i) & 1, bfac(k->bl[i].src), bfac(k->bl[i].dst), bop(k->bl[i].eq), bfac(k->bl[i].srca), bfac(k->bl[i].dsta), bop(k->bl[i].eqa), k->wmask[i]};
        if (k->bl[i].eq & ADV_KEY) {   // an advanced blend operation (adv_blend): colour and alpha take it; premultiplied source and overlap are the pipeline's (from its first such target)
            cba[i].colorBlendOp = cba[i].alphaBlendOp = adv_vk_op((int)(k->bl[i].eq & 0xff));
            if (!nadv++) { cadv.srcPremultiplied = !(k->bl[i].eqa & 1); cadv.blendOverlap = (k->bl[i].eqa >> 1) == 1 ? VK_BLEND_OVERLAP_DISJOINT_EXT : (k->bl[i].eqa >> 1) == 2 ? VK_BLEND_OVERLAP_CONJOINT_EXT : VK_BLEND_OVERLAP_UNCORRELATED_EXT; } }
    }
    VkPipelineColorBlendStateCreateInfo cb = {VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO, nadv ? &cadv : NULL, 0, k->logic_en, (VkLogicOp)k->logic_op, k->ncol, cba, {0, 0, 0, 0}};
    VkDynamicState dyn[] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR, VK_DYNAMIC_STATE_BLEND_CONSTANTS, VK_DYNAMIC_STATE_STENCIL_COMPARE_MASK,
                            VK_DYNAMIC_STATE_STENCIL_WRITE_MASK, VK_DYNAMIC_STATE_STENCIL_REFERENCE, VK_DYNAMIC_STATE_DEPTH_BIAS, VK_DYNAMIC_STATE_LINE_WIDTH};
    VkPipelineDynamicStateCreateInfo dy = {VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO, NULL, 0, sizeof dyn / sizeof *dyn, dyn};
    VkFormat cf[8]; for (uint32_t i = 0; i < k->ncol; i++) cf[i] = (VkFormat)k->cfmt[i];
    VkFormat df = (VkFormat)k->dfmt;
    int has_stencil = df == VK_FORMAT_D24_UNORM_S8_UINT || df == VK_FORMAT_D32_SFLOAT_S8_UINT || df == VK_FORMAT_S8_UINT;
    VkPipelineRenderingCreateInfo rci = {VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO, NULL, 0, k->ncol, cf, df == VK_FORMAT_S8_UINT ? VK_FORMAT_UNDEFINED : df, has_stencil ? df : VK_FORMAT_UNDEFINED};
    VkRenderPass prp = rp_on() ? rp_compat(k->ncol, k->cfmt, k->dfmt, k->samples) : VK_NULL_HANDLE;   // with render passes, pipelines are made against a compatible one
    if (!prp && !V.has_dynrender) { build_end(bs); return (VkPipeline)1; }   // no compatible render pass and no dynamic rendering: no valid pipeline (pass_commit skips such passes)
    VkGraphicsPipelineCreateInfo ci = {VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO, prp ? NULL : &rci, 0, ns, ss, &vi, &ia, NULL, &vs, &rs, &ms, &ds, &cb, &dy, layout, prp, 0, VK_NULL_HANDLE, -1};
    if (k->mesh && !(k->mesh & 8)) ci.pVertexInputState = NULL, ci.pInputAssemblyState = NULL;   // (ignored with a mesh stage, and may be NULL: DM_VK_MESH_VISTATE=1 gives them)
    if (k->mesh) VKLOG("%s: into the driver's compile", what);
    if (bs >= 0) g_build[bs].phase = 2;
    VkResult r = vkCreateGraphicsPipelines(V.dev, V.pcache, 1, &ci, NULL, &p);
    build_end(bs);
    static uint64_t n; uint64_t c = __atomic_add_fetch(&n, 1, __ATOMIC_RELAXED);
    if (r) { LOG("[RED] [vk] graphics pipeline failed (%d): shaders %016llx %016llx %016llx", r, (unsigned long long)k->key[0], (unsigned long long)k->key[1], (unsigned long long)k->key[2]); p = (VkPipeline)1; }
    else if (k->mesh) VKLOG("task / mesh pipeline, mesh %016llx task %016llx fragment %016llx: compiled", (unsigned long long)k->key[0], (unsigned long long)k->key[2], (unsigned long long)k->key[1]);
    else if (c % 200 == 0) VKLOG("%llu graphics pipelines", (unsigned long long)c);
    __atomic_add_fetch(&g_vkst.pipes_built, 1, __ATOMIC_RELAXED);
    return p;   // (VkPipeline)1 = failed
}

// ---------------- descriptors ----------------
static uint64_t c2_handle(int st, int off) {   // the driver's handle table c2, as dmsh_build.py lays it out
    if (off >= 0x20 && off < 0x120) return S.tex[st][(off - 0x20) / 8];
    if (off >= 0x120 && off < 0x160) return S.img[st][(off - 0x120) / 8];
    if (off >= 0x168 && off < 0x568) return S.tex[st][128 + (off - 0x168) / 8];
    if (off >= 0x568 && off < 0x968) return S.samp[st][(off - 0x568) / 8];
    return 0;
}
static uint32_t ubo_word(int st, int reg, int off) {
    const p_bind_buf *u = reg >= 0 && reg < 16 ? &S.ubo[st][reg] : NULL; uint32_t v = 0;
    if (u && u->addr && off >= 0 && (uint64_t)off + 4 <= (u->size ? u->size : 4) && vk_cpu_ptr(u->addr + off, 4)) vk_ubd_read(u->addr + off, 4, (uint8_t *)&v);   // with the deferred updates
    return v;
}
static uint32_t g_dispatch[3], g_draw_binst;
// the compute stage's c0 driver segment as stage_set last bound it (dispatch: an indirect dispatch's group counts go to its + 0x200), and DM_VK_IND_C0 (live, default 1; 0: left at 0,
// as before, the A/B); the indirect dispatches that took their counts so (logged at the first, then every 65536th)
static VkBuffer g_c0cs_b; static VkDeviceSize g_c0cs_o; static uint64_t g_ind_c0_n;   // (g_ind_c0 is with the other switches, above)
#define SB_ALIGN 256   // storage buffers are bound from this boundary below their address (dm/android/tools/shader_vk.py SB_ALIGN, which the shaders were translated for)
// The stage's storage buffer table in c0 (dmsh_abi.h; nvn_mtl.m c0_ssbo_base): vertex, pixel, geometry (NieR:Automata), compute, and NVN2's task and mesh stages in the tessellation
// stages' places (Mario Kart World's task shaders read c0 0x670 and 0x680, its mesh shaders 0x770 to 0x7b0, the game binding their buffers from slot 0: upstream 5ae0d9ae, the sky panels)
static int c0_ssbo_base(int st) { return st == 0 ? DMSH_C0_SSBO_VS : st == 1 ? DMSH_C0_SSBO_PS : st == 2 ? DMSH_C0_SSBO_GS : st == 5 ? DMSH_C0_SSBO_CS : st == 3 ? DMSH_C0_SSBO_TS : DMSH_C0_SSBO_MS; }
static int nvn2(void) { static int v = -1; if (v < 0) v = dm_sdk_ounce() != 0; return v; }   // the Switch 2's NVN2 (dm.h dm_sdk_ounce: the sdk's "spec: Ounce", loader.c)
static void drv_segment(int st, int bank, uint8_t *d) {
    memset(d, 0, 0x1000);
    if (bank == 0) {
        int base = c0_ssbo_base(st);
        for (int r = 0; r < DMSH_C0_SSBO_SLOTS; r++) { uint64_t a = S.ssbo[st][r].addr; uint32_t n = (uint32_t)S.ssbo[st][r].size; memcpy(d + base + DMSH_C0_ROW * r, &a, 8); memcpy(d + base + DMSH_C0_ROW * r + 8, &n, 4); }
        if (st != 5) { const float mp = 2047.0f; memcpy(d + DMSH_C0_POINT_SIZE, &mp, 4); memcpy(d + DMSH_C0_BASE_INSTANCE, &g_draw_binst, 4); }
        else { memcpy(d + DMSH_C0_DISPATCH, g_dispatch, 12);
            // NVN2 compute also keeps the group counts below its per-shader layout (c0 + 4: a light culling shader's cluster is (z * c0[8] + y) * c0[4] + x, nvn_mtl.m); its uniform and storage
            // entries of c0 + 0x10 on are read at NVN's compute tables, as the translator maps them (Ryujinx InstEmitAmpere.cs AMapC0: uniform slot r at 0x210 + 0x10 r, storage s at 0x310 + 0x10 s)
            if (nvn2()) memcpy(d + DMSH_C0_DISPATCH_NVN2, g_dispatch, 12);
            for (int r = 0; r < DMSH_C0_CS_UBO_SLOTS; r++) { uint64_t a = S.ubo[5][r].addr; uint32_t n = (uint32_t)S.ubo[5][r].size; memcpy(d + DMSH_C0_CS_UBO + DMSH_C0_ROW * r, &a, 8); memcpy(d + DMSH_C0_CS_UBO + DMSH_C0_ROW * r + 8, &n, 4); } }
    } else for (int off = 0x20; off < 0x968; off += 8) { uint64_t h = c2_handle(st, off); memcpy(d + off, &h, 8); }
}
_Static_assert(DMSH_C0_SSBO_MS + DMSH_C0_ROW * DMSH_C0_SSBO_SLOTS <= 0x1000 && DMSH_DRV_SIZE <= 0x1000, "the driver segment buffer (drv_segment) holds the NVN2 task and mesh tables");
// vk_samp_by_id without the registry lock while no sampler ID was registered with other parameters (g_samp_gen): a 32-slot texture array took the lock
// per slot, a lock the game thread takes for every texture it registers (DM_NVN_ASYNC). VkSamplers are never destroyed, so a kept handle stays valid
extern uint64_t g_samp_gen;
static VkSampler samp_cached(int sid) {
    enum { SC = 1024 }; static struct { uint64_t gen; int sid; VkSampler s; } c[SC]; unsigned k = (unsigned)sid % SC;
    uint64_t g = __atomic_load_n(&g_samp_gen, __ATOMIC_ACQUIRE);   // read before the lookup: a registration racing it leaves the entry outdated, never wrong and current
    if (c[k].gen == g && c[k].sid == sid) return c[k].s;
    VkSampler s = vk_samp_by_id(sid); c[k].gen = g; c[k].sid = sid; c[k].s = s; return s;
}
static uint64_t g_tex_sseq;   // stage_set: the bind count of the texture slot being collected (0 for handles read from memory)
static int samp_sid(int st, const dmsv_bind *b, uint64_t texh, uint64_t *seq_out) {   // the sampler ID a binding reads and the registrations its bind saw, as samp_for takes them
    int sid = 0; uint64_t seq = 0;
    switch (b->samp) {
    case RS_TEX: sid = (int)(texh >> 20); seq = g_tex_sseq; break;
    case RS_SLOT: sid = b->samp_arg >= 0 && b->samp_arg < 128 ? (int)(S.samp[st][b->samp_arg] >> 20) : 0; seq = sid ? S.samp_sseq[st][b->samp_arg] : 0; break;
    case RS_ID: sid = b->samp_arg; break;
    case RS_UBO: sid = (int)(ubo_word(st, b->samp_arg & 0xff, b->samp_arg >> 8) >> 20); break;
    default: break;
    }
    uint64_t mtl_samp_seq(void); if (!g_samp_snap || seq >= mtl_samp_seq()) seq = 0;
    *seq_out = seq; return sid;
}
static VkSampler samp_for(int st, const dmsv_bind *b, uint64_t texh, int nearest, uint32_t shift) {   // nearest: its unfiltered twin (integer views); shift: LOD clamps lowered that much (vk_samp_shift)
    uint64_t seq; int sid = samp_sid(st, b, texh, &seq);   // (seq 0 when no registration came since the bind: the newest, the cached lookups)
    VkSampler vk_samp_by_id_nearest(int sid), vk_samp_at(int sid, uint64_t seq, int nearest), vk_samp_shift(int sid, uint64_t seq, int nearest, uint32_t b);
    VkSampler s = !sid ? VK_NULL_HANDLE : shift ? vk_samp_shift(sid, seq, nearest, shift) : seq ? vk_samp_at(sid, seq, nearest) : nearest ? vk_samp_by_id_nearest(sid) : samp_cached(sid);
    if (g_tracing || (g_x_logfs && S.sh[1] && S.sh[1]->key == g_x_logfs)) { char d[256]; void vk_samp_describe_at(int sid, uint64_t seq, char *out, size_t n); vk_samp_describe_at(sid, seq, d, sizeof d); VKLOG("  %sbinding %d: %s (texture handle %#llx, present %llu, pass %d, thread %d)", g_tracing ? "" : "logfs ", b->binding, d, (unsigned long long)texh, (unsigned long long)g_tex_frame, g_sub_pass - 1, (int)gettid()); }
    return s ? s : vk_dummy_sampler();
}
// texture views by ID within a submission: a draw's texture arrays (32 slots) looked every slot up again, each through the registry's and the view table's
// locks (NEO: about 2 ms a frame). An entry holds while no texture ID is registered again and no view is dropped (g_tex_gen)
// Kept across submissions (DM_VK_TVC_KEEP, live, default on): an entry holds while its ID is not registered again with another texture (gfx_vk_tex.c g_tver, the ID's own count: g_tex_gen
// is any ID's, and a frame's ~87 registrations dropped every entry) and no image or view went (g_tex_dgen); at an entry's first use in a submission its image's per-submission work is
// done (vk_tex_again: a packaged texture's re-upload check, a linear one's copy, the aliasing check) without the lookups. Mario Kart World's race frame is 4 submissions of ~2,600
// textures: kept for one submission, ~3,600 views a frame were looked up again, ~17% of the replay thread (vkbench). Not for storage images (their first use in a submission marks them
// GPU-written) nor under asynchronous replay (DM_NVN_ASYNC: a job sees the registrations made before it was queued, which an entry made by an earlier job may not be). =0: one submission
enum { TVC = 8192 };
static struct { uint64_t epoch, gen, dgen; uint32_t ver; int tid, dim, arr, num; uint32_t shift; VkImageView v; vkimg_t *im; VkFormat vf; uint16_t ly0, nly; } g_tvc[TVC];   // shift: the view's LOD shift (tex_desc_for)
static tex_t g_tvc_t[TVC];   // the registration each kept entry's view was made from (vk_tex_again's look)
#define g_tvc_keep g_tvc_keep_env
extern _Atomic uint32_t g_tver[]; extern uint64_t g_tex_dgen; vkimg_t *vk_tex_again(vkimg_t *im, const tex_t *t); uint64_t nvn_replaying_job(void);
static inline uint32_t tex_ver(int tid) { return atomic_load_explicit(&g_tver[tid < (1 << 20) ? tid : (1 << 20) - 1], memory_order_acquire); }
// what a view's texels are, as gfx_vk_shader.c's integer-texel variants need it: 0 float, 1 uint (unsigned integer formats, a stencil), 2 int
static int view_num(const tex_t *t, const view_t *v) {
    const vkfmt_t *f = vk_fmt(v && v->set_format ? v->format : t->d.format); if (!f) return 0;
    if (f->depth || f->stencil) return f->stencil && (!f->depth || t->stencil || (v && v->set_dsmode && v->dsmode == 1));   // vk_tex_view's aspect
    return f->is_int ? (f->is_sint ? 2 : 1) : 0;
}
static uint64_t g_num_um, g_num_im;   // stage_set: the sampled bindings of the stage being built that got uint / int views (bit = binding index)
extern uint64_t g_tex_gen;
// A depth image sampled with a linear depth-compare sampler (a shadow map's PCF): whether the device filters its format linearly, said once a format (without, the compare is point-sampled
// and a shadow's edge goes in steps of a texel: Mario Kart World's kart shadow looked so in run 13); the sampler's parameters are in gfx_vk_tex.c make_sampler's line
static void shadow_check(const vkimg_t *im) {
    static VkFormat seen[16]; static int ns; for (int i = 0; i < ns; i++) if (seen[i] == im->fmt) return; if (ns == 16) return; seen[ns++] = im->fmt;
    VkFormatProperties fp; vkGetPhysicalDeviceFormatProperties(V.pd, im->fmt, &fp);
    if (fp.optimalTilingFeatures & VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT) VKLOG("shadow map format %d (%ux%u, %u layers): sampled with a linear depth-compare sampler, filtered linearly (2x2 PCF)", im->fmt, im->w, im->h, im->layers);
    else VKLOG("[RED] shadow map format %d (%ux%u, %u layers): sampled with a linear depth-compare sampler, but the device does not filter that format linearly: point-sampled, edges in texel steps",
               im->fmt, im->w, im->h, im->layers);
}
// A sampled view whose sampler filters linearly, in a format the device does not filter linearly (no SAMPLED_IMAGE_FILTER_LINEAR in its optimal tiling features): Vulkan leaves the result
// undefined, and a driver that point-samples it draws a texture, a shadow map's moments or a depth pyramid in blocks a texel wide (a 32-bit float format, which Vulkan does not require
// any device to filter, is the likely case). Said once a format, with the first image; integer views take their sampler's unfiltered twin already (samp_for). gfx_vk_tex.c vk_samp_linear
static void filter_check(VkFormat fmt, VkSampler s, const vkimg_t *im) {
    static uint8_t st[256];   // per VkFormat: 0 not looked at, 1 filtered linearly, 2 not, 3 not and said
    if ((unsigned)fmt >= 256 || !im) return;
    if (!st[fmt]) { VkFormatProperties fp; vkGetPhysicalDeviceFormatProperties(V.pd, fmt, &fp); st[fmt] = fp.optimalTilingFeatures & VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT ? 1 : 2; }
    int vk_samp_linear(VkSampler s);
    if (st[fmt] == 2 && vk_samp_linear(s)) { st[fmt] = 3; VKLOG("[RED] format %d (an image %ux%u, %u layers, nvn format %#x) sampled with a linear filter the device does not have for it: undefined (point-sampled at best)",
        fmt, im->w, im->h, im->layers, im->nvn_format); }
}
// Same-pass feedback: a draw that samples (or loads as an image) the image of an attachment of its own open pass. Vulkan leaves such a feedback loop undefined without
// VK_EXT_attachment_feedback_loop_layout (every image here stays in GENERAL), and a tiler keeps the attachment in tile memory while the texture unit reads the image's memory:
// what the draw reads is the content from before the pass, or anything when the pass runs in system memory (UBWC metadata in flux). On the Switch it reads the content from before
// the draw. Upstream found such draws in agl's post-effect programs (Super Mario 3D World, Captain Toad, Wonder: 16x16 blocks on Apple GPUs) and in decals reading the G-buffer
// they write (nvn_mtl.m fb_prepare: a snapshot of the attachment, or a read from the tile). Counted and named here to tell whether a game does it (DM_VK_TRACE_PASS then shows the pass)
uint64_t g_fb_loops;   // draws found sampling an attachment of their open pass (a descriptor set built for them; a memoized set is not looked at again)
static void feedback_check(int st, const dmsv_bind *b, const vkimg_t *im, int storage) {
    if (!im || S.pass_open != 1 || st == 5) return;
    for (int i = 0; i < 9; i++) if (P_att[i].im == im) {
        g_fb_loops++; const vkshader_t *sh = st < 3 ? S.sh[st] : NULL;   // (a task or mesh stage: no key here)
        static int n; if (n++ < 16) VKLOG("[RED] same-pass feedback #%llu: stage %d shader %016llx %s binding %d the image of its pass's %s %d (%ux%u, nvn format %#x): undefined in Vulkan, the content from before the draw on the Switch",
            (unsigned long long)g_fb_loops, st, (unsigned long long)(sh ? sh->key : 0), storage ? "loads as an image" : "samples", b->binding, i < 8 ? "colour target" : "depth target", i < 8 ? i : 0, im->w, im->h, im->nvn_format);
        return;
    }
}
// A binding's LOD shift (PARITY G1): its view's level 0 is its texture's level (view base + shift), the image having been made from a coarser level (vkimg_t lv0: a dense image from its
// base). The sampler's clamps are lowered by it (samp_for, gfx_vk_tex.c vk_samp_shift); the shader's explicit LODs, fetch levels and size queries need it as data (gfx_vk_shader.c's LOD-shift
// variant): g_lod_shift holds it by stage and binding for the draw or dispatch whose descriptors are being made, g_lod_shifted counts the bindings made with one
uint8_t g_lod_shift[6][64]; uint64_t g_lod_shifted;
static void lod_shift_note(int st, const dmsv_bind *b, uint32_t shift) {
    if (st < 0 || st > 5) return; uint8_t v = (uint8_t)(shift > 15 ? 15 : shift); g_lod_shift[st][b->binding & 63] = v;
    if (v) { g_lod_shifted++; static int n; if (n++ < 6) VKLOG("stage %d binding %d: a view that starts %u levels coarser than its texture's base (its image made from a coarser level): its sampler's LOD clamps lowered by as many", st, b->binding, v); }
}
static __thread const vkimg_t *t_desc_im; static __thread int t_desc_dummy; static __thread VkFormat t_desc_vf;   // what tex_desc_for last resolved: the image and the view's format (stage_set's VT bindings)
static __thread uint32_t t_desc_shift, t_desc_ly0, t_desc_nly;   // and its LOD shift, the view's first layer and its layers (gfx_vk_vt.h vt_bound_t)
static int tex_desc_for(int st, const dmsv_bind *b, uint64_t h, VkDescriptorImageInfo *ii, int storage) {   // the view's view_num
    int tid = (int)(h & 0xfffff); const view_t *v = NULL; const tex_t *t = NULL; t_desc_im = NULL; t_desc_dummy = 0; t_desc_vf = VK_FORMAT_UNDEFINED; t_desc_shift = t_desc_ly0 = t_desc_nly = 0;
    ii->imageLayout = VK_IMAGE_LAYOUT_GENERAL; ii->sampler = storage ? VK_NULL_HANDLE : samp_for(st, b, h, 0, 0); ii->imageView = VK_NULL_HANDLE;
    unsigned ck = ((unsigned)tid * 2654435761u ^ (unsigned)b->dim * 97u ^ (unsigned)(b->flags & 1)) % TVC;
    int keep = g_tvc_keep && !storage && !nvn_replaying_job();
    if (tid && g_var_now != 3 && !g_tracing && g_tvc[ck].tid == tid && g_tvc[ck].dim == b->dim && g_tvc[ck].arr == (b->flags & 1) &&
        (keep ? g_tvc[ck].ver == tex_ver(tid) && g_tvc[ck].dgen == g_tex_dgen && (g_tvc[ck].epoch == g_set_epoch || (vk_tex_again(g_tvc[ck].im, &g_tvc_t[ck]) && (g_tvc[ck].epoch = g_set_epoch, 1)))
              : g_tvc[ck].epoch == g_set_epoch && g_tvc[ck].gen == g_tex_gen)) {
        ii->imageView = g_tvc[ck].v; t_desc_im = g_tvc[ck].im; t_desc_vf = g_tvc[ck].vf; t_desc_shift = g_tvc[ck].shift; t_desc_ly0 = g_tvc[ck].ly0; t_desc_nly = g_tvc[ck].nly; if (g_tvc[ck].im) { g_tvc[ck].im->last_use = mtl_last_fence() + 1; vk_dsc_read(g_tvc[ck].im, "sampled or bound as an image"); } vk_texdump_note(g_tvc[ck].im);
        feedback_check(st, b, g_tvc[ck].im, storage); lod_shift_note(st, b, g_tvc[ck].shift);
        if ((g_tvc[ck].num || g_tvc[ck].shift) && !storage) ii->sampler = samp_for(st, b, h, g_tvc[ck].num != 0, g_tvc[ck].shift);
        if (!storage && g_tvc[ck].im && (g_tvc[ck].im->aspect & VK_IMAGE_ASPECT_DEPTH_BIT)) { int vk_samp_cmp_linear(VkSampler s); if (vk_samp_cmp_linear(ii->sampler)) shadow_check(g_tvc[ck].im); }
        return g_tvc[ck].num;
    }
    int num = 0; uint32_t shift = 0, ver0 = tid ? tex_ver(tid) : 0; uint64_t dgen0 = g_tex_dgen;   // (before the lookup: a registration racing it leaves the entry outdated, never wrong)
    if (tid && g_var_now != 3 && vk_tex_by_id(tid, &v, &t) && t) {
        vkimg_t *im = NULL; ii->imageView = vk_tex_view(t, v, b->dim, b->flags & 1, 0, &im); if (storage && im) vk_img_wrote(im);   // (gfx_vk_tex.c: every GPU write counts for the [alias] lines)
        if (g_tracing) VKLOG("  texture handle %#llx id %d: %ux%u nvn format %#x target %d serial %llu pool %p+%#llx size %#llx → image %p (%ux%u fmt %d layers %u levels %u) view %p", (unsigned long long)h, tid, t->d.width, t->d.height, t->d.format, t->d.target,
                             (unsigned long long)t->serial, (const void *)t->pool, (unsigned long long)t->off, (unsigned long long)t->L.size, (void *)im, im ? im->w : 0, im ? im->h : 0, im ? im->fmt : 0, im ? im->layers : 0, im ? im->levels : 0, (void *)ii->imageView);
        if (im) im->last_use = mtl_last_fence() + 1;
        // A texture whose image has no view of the type the shader declares (a 2D texture or a 2D array that is not cube compatible in a cube slot, a 2D one in a 3D slot): vk_tex_view keeps
        // the image's own type, which does not match the shader's, and Vulkan leaves such reads undefined. Upstream binds the zero cube / the zero 3D texture there, a defined read (nvn_mtl.m
        // tex_fit: Pokémon Legends: Z-A and Mario Kart 8 Deluxe's slot 10 placeholders, Batman: Arkham Knight's 2D targets in 3D slots); the dummy below is that zero texture
        if (ii->imageView && im && !storage && ((b->dim == 3 && !(im->cflags & VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT)) || (b->dim == 2 && im->vtype != VK_IMAGE_VIEW_TYPE_3D))) {
            static int n; if (n++ < 4) VKLOG("texture %ux%u (%u layers, nvn format %#x) in a %s slot (binding %d): no view of that type, the zero %s bound instead", im->w, im->h, im->layers, im->nvn_format,
                                             b->dim == 3 ? "cube" : "3D", b->binding, b->dim == 3 ? "cube" : "3D texture");
            ii->imageView = VK_NULL_HANDLE; im = NULL;
        }
        t_desc_im = ii->imageView ? im : NULL; t_desc_vf = vk_view_fmt;   // (after the slot-type check, which may have bound the zero texture)
        if (ii->imageView) num = view_num(t, v);
        if (ii->imageView && im && im->lv0) { uint32_t base = v && v->set_levels ? v->base_level : 0; shift = base < im->lv0 ? im->lv0 - base : 0; }   // (the view's level 0 is its texture's level base + shift)
        lod_shift_note(st, b, shift);
        if (t_desc_im) { uint32_t ly = v && v->set_layers ? v->min_layer : 0; t_desc_shift = shift; t_desc_ly0 = ly < im->layers ? ly : im->layers - 1; t_desc_nly = vk_view_layers; }   // (vk_tex_view's)
        if (!storage && !num && im && ii->imageView) filter_check(vk_view_fmt, ii->sampler, im);   // (a linear sampler on a format without linear filtering)
        feedback_check(st, b, im, storage);   // (an attachment of the open pass read by this draw)
        if (!storage && !num && im && ii->imageView && (im->aspect & VK_IMAGE_ASPECT_DEPTH_BIT)) { int vk_samp_cmp_linear(VkSampler s); if (vk_samp_cmp_linear(ii->sampler)) shadow_check(im); }   // (a shadow map)
        if (ii->imageView) { g_tvc[ck].epoch = g_set_epoch; g_tvc[ck].gen = g_tex_gen; g_tvc[ck].tid = tid; g_tvc[ck].dim = b->dim; g_tvc[ck].arr = b->flags & 1; g_tvc[ck].v = ii->imageView; g_tvc[ck].vf = vk_view_fmt; g_tvc[ck].ly0 = (uint16_t)t_desc_ly0; g_tvc[ck].nly = (uint16_t)t_desc_nly; g_tvc[ck].im = im; g_tvc[ck].num = num; g_tvc[ck].shift = shift;
            g_tvc[ck].ver = ver0; g_tvc[ck].dgen = keep ? dgen0 : 0; if (keep) g_tvc_t[ck] = *t; }   // (dgen 0, never current: an entry made without keeping is not kept, its g_tvc_t not written)
    }
    if ((num || shift) && !storage) ii->sampler = samp_for(st, b, h, num != 0, shift);
    if (!ii->imageView) {
        const vkfmt_t *f = t ? vk_fmt(t->d.format) : NULL;
        ii->imageView = vk_dummy_view(b->dim == 0xff ? 1 : b->dim, b->flags & 1, (b->flags & 2) != 0, storage, f && f->is_int); t_desc_dummy = 1; num = f && f->is_int ? (f->is_sint ? 2 : 1) : 0;
    }
    return num;
}
// Reuse within a submission (the cache is emptied when it begins, vk_exec_reset): descriptor sets by their whole content (layout, and every buffer range,
// view, sampler and texel buffer view written), and driver segments by their bytes. Unity draws many times with the same bindings, and every new set
// is an allocation and a vkUpdateDescriptorSets; the driver segment's copy is shared so that a set using it can match.
enum { SETC = 8192, DRVC = 1024 };
static struct { uint64_t h, epoch; VkDescriptorSetLayout l; VkDescriptorSet set; } g_setc[SETC];
static struct { uint64_t h, epoch; VkBuffer b; VkDeviceSize o; } g_drvc[DRVC];
uint64_t g_set_hits, g_set_new, g_ns_set_build, g_ns_set_hash, g_ns_set_new;   // descriptor sets: reused, new; time collecting their descriptors, hashing them, allocating and writing new ones
// content hash of descriptor sets and driver segments: 8 bytes a step (a set with 32-slot texture arrays is several KB, which byte-wise FNV took milliseconds a frame over)
static inline uint64_t fnv_w(uint64_t h, uint64_t w) { h = (h ^ w) * 0x9E3779B97F4A7C15ull; return h ^ (h >> 29); }   // fnv's word step, for one word
static inline uint64_t fnv(uint64_t h, const void *p, size_t n) {
    const uint8_t *b = p; size_t i = 0;
    for (; i + 8 <= n; i += 8) { uint64_t w; memcpy(&w, b + i, 8); h = (h ^ w) * 0x9E3779B97F4A7C15ull; h ^= h >> 29; }
    for (; i < n; i++) { h ^= b[i]; h *= 0x100000001b3ull; }
    return h;
}
static void drv_shared(int st, int bank, VkBuffer *buf, VkDeviceSize *off) {   // the driver segment: built, then one copy per distinct content in the submission
    // c0 (bank 0) is made of the stage's storage buffer table, its base instance (graphics) or its dispatch size and uniform table (compute): the stage's last segment is kept while
    // those are the same, and a new one is looked up by them, not by its 2 KB (vkbench: building and hashing those was ~7% of the replay thread). DM_VK_DRV_INPUTS=0 (live): by the bytes
    static int inp_on = -1; static unsigned inp_k; if (!(inp_k++ & 255)) { const char *e = getenv("DM_VK_DRV_INPUTS"); inp_on = !(e && !atoi(e)); }
    if (bank == 0 && inp_on && st >= 0 && st < 6) {
        static struct { uint64_t epoch; uint32_t in[2 * DMSH_C0_SSBO_SLOTS * 3 + 8]; VkBuffer b; VkDeviceSize o; } last[6];
        uint32_t in[2 * DMSH_C0_SSBO_SLOTS * 3 + 8]; int n = 0;
        for (int r = 0; r < DMSH_C0_SSBO_SLOTS; r++) { uint64_t a = S.ssbo[st][r].addr; in[n++] = (uint32_t)a; in[n++] = (uint32_t)(a >> 32); in[n++] = (uint32_t)S.ssbo[st][r].size; }
        if (st != 5) in[n++] = g_draw_binst;
        else { in[n++] = g_dispatch[0]; in[n++] = g_dispatch[1]; in[n++] = g_dispatch[2];
            for (int r = 0; r < DMSH_C0_CS_UBO_SLOTS; r++) { uint64_t a = S.ubo[5][r].addr; in[n++] = (uint32_t)a; in[n++] = (uint32_t)(a >> 32); in[n++] = (uint32_t)S.ubo[5][r].size; } }
        in[n++] = (uint32_t)st;
        if (last[st].epoch == g_set_epoch && !g_x_nodrvc && !memcmp(last[st].in, in, sizeof *in * (size_t)n)) { *buf = last[st].b; *off = last[st].o; return; }
        uint64_t h = fnv(0x6a09e667f3bcc909ull, in, sizeof *in * (size_t)n); unsigned k = (unsigned)(h % DRVC);
        if (!(g_drvc[k].epoch == g_set_epoch && g_drvc[k].h == h && !g_x_nodrvc)) {
            uint8_t d[0x1000]; drv_segment(st, bank, d);
            uint8_t *p = vk_scratch(0x1000, V.props.limits.minUniformBufferOffsetAlignment, buf, off); memcpy(p, d, sizeof d);
            g_drvc[k].h = h; g_drvc[k].epoch = g_set_epoch; g_drvc[k].b = *buf; g_drvc[k].o = *off; }
        *buf = g_drvc[k].b; *off = g_drvc[k].o; last[st].epoch = g_set_epoch; memcpy(last[st].in, in, sizeof *in * (size_t)n); last[st].b = *buf; last[st].o = *off;
        return;
    }
    uint8_t d[0x1000]; drv_segment(st, bank, d);
    // only the bytes drv_segment can write are hashed (bank 0 ends with the mesh stage's table at 0x870, the c2 table at 0x968; the rest is always zero): 4 KB was 512 hash steps a segment
    uint64_t h = fnv(0xcbf29ce484222325ull, d, bank ? 0x968 : DMSH_C0_SSBO_MS + DMSH_C0_ROW * DMSH_C0_SSBO_SLOTS); unsigned k = (unsigned)(h % DRVC);
    if (g_drvc[k].epoch == g_set_epoch && g_drvc[k].h == h && !g_x_nodrvc) { *buf = g_drvc[k].b; *off = g_drvc[k].o; return; }
    uint8_t *p = vk_scratch(0x1000, V.props.limits.minUniformBufferOffsetAlignment, buf, off); memcpy(p, d, sizeof d);
    g_drvc[k].h = h; g_drvc[k].epoch = g_set_epoch; g_drvc[k].b = *buf; g_drvc[k].o = *off;
}
// NVN2 push constants (OP_PUSH_CONST: nvnCommandBufferPush{Graphics,Compute}Constants, Donkey Kong Bananza's upscaler (1 / w, 1 / h, w, h), Mario Kart World): the bytes the game pushed,
// one DMSH_PUSH_SIZE block for the graphics stages and one for compute, which the shaders read as c[24] (the translator maps NVN2 compute's push region of c0 there, Ryujinx AMapC0).
// Bound as a uniform buffer: the block's shadow copied at each draw or dispatch that reads it, one copy per distinct content in the submission (as drv_shared), as nvn_mtl.m binds NVN_RB_PUSH
static void push_shared(int compute, VkBuffer *buf, VkDeviceSize *off) {
    enum { PC = 256 }; static struct { uint64_t h, epoch; VkBuffer b; VkDeviceSize o; } c[PC];
    const uint8_t *d = S.push[compute != 0]; uint64_t h = fnv(0x84222325cbf29ce4ull ^ (uint64_t)(compute != 0), d, DMSH_PUSH_SIZE); unsigned k = (unsigned)(h % PC);
    if (c[k].epoch == g_set_epoch && c[k].h == h && !g_x_nodrvc) { *buf = c[k].b; *off = c[k].o; return; }
    uint8_t *p = vk_scratch(DMSH_PUSH_SIZE, V.props.limits.minUniformBufferOffsetAlignment, buf, off); memcpy(p, d, DMSH_PUSH_SIZE);
    c[k].h = h; c[k].epoch = g_set_epoch; c[k].b = *buf; c[k].o = *off;
}
static int push_bank(const dmsv_bind *b) { return (b->kind == RK_DRV && b->reg == DMSH_PUSH_BANK) || (b->kind == RK_CBUF && (b->reg == DMSH_PUSH_BANK || b->reg == DMSH_PUSH_BANK - 3)); }   // (gfx_vk.h RK_DRV)
// a constant buffer binding's descriptor (stage_set, set_memo_get): its pool's buffer from the alignment boundary below it, or the shadow of the deferred updates over it
// The last descriptor each stage's uniform slot gave, by the binding's address and size, in this submission while no pool came or went and no deferred update changed (its shadow):
// a draw looks up each of its constant buffers again (Mario Kart World's race: ~9,700 a frame, a pool lookup each), most of them bound to the same view and material constants as the
// draw before (vkbench). DM_VK_CBUF_CACHE=0 (live): looked up every time
extern uint64_t g_ubd_gen; uint32_t vk_pool_gen(void);
static struct { uint64_t addr, size, epoch, ugen; uint32_t pgen, scratch; VkDescriptorBufferInfo bi; } g_cbc[6][16];
static int g_cb_scratch;   // the last cbuf_desc gave a descriptor of the submission's own memory (a shadow, the push block): not a set to keep (kept_get)
static void cbuf_desc_(int st, const vkshader_t *s, const dmsv_bind *b, VkDescriptorBufferInfo *bi);
static void cbuf_desc(int st, const vkshader_t *s, const dmsv_bind *b, VkDescriptorBufferInfo *bi) {
    if (!g_cbc_on || push_bank(b) || b->reg < 0 || b->reg >= 16 || st < 0 || st > 5 || g_var_now || g_ubolog_key) { cbuf_desc_(st, s, b, bi); return; }
    const p_bind_buf *u = &S.ubo[st][b->reg]; typeof(g_cbc[0][0]) *c = &g_cbc[st][b->reg]; uint32_t pg = vk_pool_gen();
    if (c->epoch == g_set_epoch && c->addr == u->addr && c->size == u->size && c->pgen == pg && c->ugen == g_ubd_gen) { *bi = c->bi; g_cb_scratch = (int)c->scratch; return; }
    cbuf_desc_(st, s, b, bi); c->epoch = g_set_epoch; c->addr = u->addr; c->size = u->size; c->pgen = pg; c->ugen = g_ubd_gen; c->bi = *bi; c->scratch = (uint32_t)g_cb_scratch;
}
static void cbuf_desc_(int st, const vkshader_t *s, const dmsv_bind *b, VkDescriptorBufferInfo *bi) {
    g_cb_scratch = 0;
    if (push_bank(b)) { VkBuffer pb; VkDeviceSize po; push_shared(st == 5, &pb, &po); *bi = (VkDescriptorBufferInfo){pb, po, DMSH_PUSH_SIZE}; g_cb_scratch = 1; return; }   // c[24]: the push block
    const p_bind_buf *u = b->reg >= 0 && b->reg < 16 ? &S.ubo[st][b->reg] : NULL;
    uint64_t n = u && u->size ? (u->size > 65536 ? 65536 : u->size) : 65536; VkBuffer buf; VkDeviceSize off;
    uint64_t ua = u ? u->addr : 0;   // (a virtual pool's range: translated to its physical memory, vk_pool_gpu_at)
    vkpool_t *p = g_var_now != 9 && ua ? vk_pool_gpu_at(&ua, n) : NULL;   // (the GPU is given it: placed if it is not yet)
    if (!p && g_var_now != 9 && ua) p = vk_pool_gpu_at(&ua, 1);   // (across two chunks or parts: the first one's, cut at its end)
    if (!p) { *bi = (VkDescriptorBufferInfo){vk_zero_buffer(), 0, 65536}; return; }
    buf = p->buf; off = ua - p->lo; if (off + n > p->hi - p->lo) n = p->hi - p->lo - off;
    ubo_log(s, st, b->reg, u->addr, n);
    if (vk_ubd_shadow(u->addr, n, &buf, &off)) { *bi = (VkDescriptorBufferInfo){buf, off, n}; g_cb_scratch = 1; return; }   // deferred updates in it: their shadow
    VkDeviceSize al = V.props.limits.minUniformBufferOffsetAlignment; if (off % al) VKRED_ONCE(40, "uniform buffer at an offset not aligned to %llu", (unsigned long long)al);
    *bi = (VkDescriptorBufferInfo){buf, off - off % al, n + off % al};
}
// The set a stage's last draw got, and what it was built from. Between two draws most stages change no texture, sampler, image or storage buffer, only
// the offsets of their constant buffers, which go to the bind (dynamic uniform buffers). Then nothing is collected or hashed: the constant buffers alone are
// looked at again, and their buffer and range must be the same (the offset too for one that is not dynamic). It holds within one submission (the sets and
// the driver segments' copies are its own) while the stage's shader and bindings (g_bind_gen, g_ubo_gen; only real changes count), the texture IDs (g_tex_gen),
// sampler IDs (g_samp_gen), memory pools (vk_pool_gen) and the base instance (the driver segment's c0 + 0x644) are unchanged. Images keep the last_use the
// first build gave them: the fence it names is the same for the whole submission. Not for compute (its driver segment holds the dispatch size and its storage
// buffers are marked written), nor for a shader that reads handles or addresses from uniform buffer memory (that memory changes under the same bindings).
// DM_VK_X_NOSETMEMO=1 (live): always collected. DM_VK_X_SETVERIFY=1 (live): one hit in 64 is collected anyway and compared, [RED] on a difference
enum { MEMO_CB = 16 };
typedef struct { vkshader_t *s; uint64_t epoch, bind, ubo, ubd, tex, samp, h, um, im; uint32_t pool, binst, dofs[96]; int ncb; uint8_t ci[MEMO_CB], cpos[MEMO_CB]; VkDescriptorBufferInfo cd[MEMO_CB]; VkDescriptorSet set; uint8_t ls[64]; } set_memo_t;   // ls: the LOD shifts its build noted (g_lod_shift)
static set_memo_t g_smemo[3]; static uint64_t g_bind_gen[6], g_ubo_gen[6];   // a stage's handles or storage buffers changed (bind_handles_rec, OP_BIND_SSBO), its uniform buffers (OP_BIND_UBO)
uint64_t g_set_memo, g_set_memo_bad;   // memo hits, and differences the verify mode found
extern uint64_t g_ubd_gen; uint32_t vk_pool_gen(void);
static int set_memo_ok(vkshader_t *s) {
    if (!s->setmemo) { int ok = 1, ncb = 0;
        for (int i = 0; i < s->nbind; i++) { const dmsv_bind *b = &s->b[i]; ncb += b->kind == RK_CBUF;
            ok &= b->kind != RK_SBUF_UBO && b->kind != RK_UBO_PTR && b->kind != RK_TEX_UBO && b->kind != RK_IMG_UBO && b->kind != RK_TBUF_UBO && b->samp != RS_UBO && i < 256; }
        s->setmemo = ok && ncb <= MEMO_CB ? 1 : 2; }
    return s->setmemo == 1;
}
static int set_memo_get(int st, vkshader_t *s, uint32_t *dofs) {
    set_memo_t *m = &g_smemo[st];
    if (m->s != s || m->epoch != g_set_epoch || m->bind != g_bind_gen[st] || m->tex != g_tex_gen || m->samp != __atomic_load_n(&g_samp_gen, __ATOMIC_ACQUIRE) || m->pool != vk_pool_gen() || m->binst != g_draw_binst) return 0;
    if (m->ubo != g_ubo_gen[st] || m->ubd != g_ubd_gen) {   // the constant buffers again: most draws move only their offsets
        for (int c = 0; c < m->ncb; c++) { VkDescriptorBufferInfo d; cbuf_desc(st, s, &s->b[m->ci[c]], &d);
            if (d.buffer != m->cd[c].buffer || d.range != m->cd[c].range || (m->cpos[c] == 0xff && d.offset != m->cd[c].offset)) return 0;
            if (m->cpos[c] != 0xff) m->dofs[m->cpos[c]] = (uint32_t)d.offset; }
        m->ubo = g_ubo_gen[st]; m->ubd = g_ubd_gen; }
    memcpy(dofs, m->dofs, sizeof *dofs * (size_t)s->ndyn); g_num_um = m->um; g_num_im = m->im; memcpy(g_lod_shift[st], m->ls, sizeof m->ls);   // (no tex_desc_for on a hit: the shifts of its build)
    return 1;
}
// Sets kept by their content across submissions (DM_VK_SET_KEEP, live, default on). A set names image views, samplers and buffers, and the same content comes back every frame:
// Mario Kart World's race frame makes ~1,370 new sets (vkAllocateDescriptorSets and vkUpdateDescriptorSets: run 13's "new ones", 4.3 ms a frame), 99% of them the content of a set an
// earlier submission made (vkbench census), and the submission's own cache (g_setc) ends with it. A set whose every descriptor outlives the submission (not a driver segment, a push
// block or a shadow in its scratch memory, nor the spill buffer or a texel buffer view) is made from kept pools (gfx_vk_core.c vk_alloc_set_kept) and found again by its content in
// later submissions. All of them are dropped (vk_keep_flush) when a view or an image goes (g_tex_dgen) or a pool comes or goes (vk_pool_gen: a buffer a set names may be gone, and a
// handle may come back naming another object); also when the kept pools are full. =0: each submission's own sets, as before
enum { KSET = 1 << 15 };
static struct { uint64_t h, gen; VkDescriptorSetLayout l; VkDescriptorSet set; } g_kset[KSET];
uint64_t g_kgen = 1; static uint64_t g_k_dgen; static uint32_t g_k_pgen; static int g_k_init;
uint64_t g_set_kept, g_set_kmade, g_kflush, g_set_kraced;   // kept sets found, made; flushes; not kept as a view went while they were collected
VkDescriptorSet vk_alloc_set_kept(VkDescriptorSetLayout l); void vk_keep_flush(void); extern uint64_t g_tex_dgen; uint32_t vk_pool_gen(void);
static void kept_flush(void) { vk_keep_flush(); __atomic_fetch_add(&g_kgen, 1, __ATOMIC_ACQ_REL); g_kflush++; }
// What makes them go (DM_VK_KEEP_FINE, live, default on): only a freed object a kept set may name, a pool gone (its buffers) or the views of an image one names (kept_gen). Additions
// never do: a set's key is the hash of the descriptors collected now (stage_set), so a new pool or view gives another key. Mario Kart World's free roam (run 14, the generations of any
// pool or view change): 5,403 flushes for 131,072 sets made in a minute, the sets kept no longer than a few draws and the replay's descriptor sets 12 ms a frame. =0: any change, as before
uint64_t g_kflush_why[3];   // flushes: pools gone, views of a kept set's image gone, kept pools full (kept_new)
uint32_t vk_pool_gone_gen(void); extern uint64_t g_tex_kgen;
static void kept_check(void) {
    uint32_t pg = g_keep_fine ? vk_pool_gone_gen() : vk_pool_gen(); uint64_t dg = __atomic_load_n(g_keep_fine ? &g_tex_kgen : &g_tex_dgen, __ATOMIC_ACQUIRE);
    if (!g_k_init || dg != g_k_dgen || pg != g_k_pgen) { if (g_k_init) { kept_flush(); g_kflush_why[pg != g_k_pgen ? 0 : 1]++; } g_k_init = 1; g_k_dgen = dg; g_k_pgen = pg; }
}
static VkDescriptorSet kept_get(uint64_t h, VkDescriptorSetLayout l) {
    kept_check(); unsigned i = (unsigned)(h >> 17) & (KSET - 1);
    for (int p = 0; p < 8; p++, i = (i + 1) & (KSET - 1)) { if (g_kset[i].gen != g_kgen) return VK_NULL_HANDLE; if (g_kset[i].h == h && g_kset[i].l == l) return g_kset[i].set; }
    return VK_NULL_HANDLE;
}
static VkDescriptorSet kept_new(VkDescriptorSetLayout l) {
    VkDescriptorSet set = vk_alloc_set_kept(l); if (set) { g_set_kmade++; return set; }
    int vk_keep_live(void); if (vk_keep_live()) { kept_flush(); g_kflush_why[2]++; }   // every pool retired (a flush just now, the GPU not past it yet): none is full, no flush again (run 14: 168 of 284)
    return VK_NULL_HANDLE;   // (the kept pools are full: start over once the GPU is past them; this one from the submission's pools)
}
// The buffers the kept sets of the current keep generation name: a pool that goes flushes them only when its buffer is one (run 14's free roam: 115 of 284 flushes for pools gone,
// the game freeing pools about three times a second). mtl_pool_del asks with the replay lock held (hle_nvn.c g_sub_mu, which the replay thread holds per job): no set is kept meanwhile
enum { KBUF = 4096 };
static struct { VkBuffer b; uint64_t gen; } g_kbuf[KBUF]; static uint64_t g_kbuf_over;   // over: a buffer did not fit this generation (any pool that goes flushes)
static inline unsigned kbuf_home(VkBuffer b) { return (unsigned)(((uint64_t)(uintptr_t)b * 0x9E3779B97F4A7C15ull) >> 52) & (KBUF - 1); }
static void kept_bufs(const VkWriteDescriptorSet *w, uint32_t nw) {
    for (uint32_t i = 0; i < nw; i++) { if (!w[i].pBufferInfo || !w[i].pBufferInfo->buffer) continue; VkBuffer b = w[i].pBufferInfo->buffer; unsigned k = kbuf_home(b); int p;
        for (p = 0; p < 16; p++, k = (k + 1) & (KBUF - 1)) { if (g_kbuf[k].gen != g_kgen) { g_kbuf[k].b = b; g_kbuf[k].gen = g_kgen; break; } if (g_kbuf[k].b == b) break; }
        if (p == 16) g_kbuf_over = g_kgen; }
}
int vk_keep_buf_named(VkBuffer b) {
    uint64_t kg = __atomic_load_n(&g_kgen, __ATOMIC_ACQUIRE); if (!g_keep_fine || g_kbuf_over == kg) return 1; unsigned k = kbuf_home(b);
    for (int p = 0; p < 16; p++, k = (k + 1) & (KBUF - 1)) { if (g_kbuf[k].gen != kg) return 0; if (g_kbuf[k].b == b) return 1; }
    return 1;
}
static void kept_put(uint64_t h, VkDescriptorSetLayout l, VkDescriptorSet set) {
    unsigned i = (unsigned)(h >> 17) & (KSET - 1);
    for (int p = 0; p < 8; p++, i = (i + 1) & (KSET - 1)) if (g_kset[i].gen != g_kgen) break;   // (the eighth probe is taken over when all are live)
    g_kset[i].h = h; g_kset[i].gen = g_kgen; g_kset[i].l = l; g_kset[i].set = set;
    if (!(g_set_kmade & (g_set_kmade - 1)) && g_set_kmade >= 256) VKLOG("descriptor sets kept across submissions: %llu found again, %llu made, %llu flushes (%llu pools gone, %llu kept images' views gone, %llu kept pools full; fine %d), %llu not kept (a view went while collected)", (unsigned long long)g_set_kept, (unsigned long long)g_set_kmade, (unsigned long long)g_kflush, (unsigned long long)g_kflush_why[0], (unsigned long long)g_kflush_why[1], (unsigned long long)g_kflush_why[2], g_keep_fine, (unsigned long long)g_set_kraced);
}
// dofs (at least s->ndyn entries): the dynamic offsets of the set's dynamic uniform buffers, in binding order as vkCmdBindDescriptorSets takes them
static VkDescriptorSet stage_set(int st, vkshader_t *s, uint32_t *dofs) {
    int memo = st < 3 && !g_x_nosetmemo && !g_tracing && !g_var_now && !g_x_nosetc && !g_x_nodrvc && !g_ubolog_key && g_td_off < 0 && set_memo_ok(s), verify = 0;
    uint32_t vd[96]; uint64_t vum = 0, vim = 0;   // verify: what the memo gave
    if (memo && set_memo_get(st, s, dofs)) { static unsigned nv; if (!g_x_setverify || ++nv % 64) { g_set_memo++; return g_smemo[st].set; }
        verify = 1; memcpy(vd, dofs, sizeof *dofs * (size_t)s->ndyn); vum = g_num_um; vim = g_num_im; }
    uint64_t gtex = g_tex_gen, gsamp = __atomic_load_n(&g_samp_gen, __ATOMIC_ACQUIRE); uint32_t gpool = vk_pool_gen();   // before collecting: a change racing it (game thread) leaves the memo outdated
    VkDescriptorSet set = VK_NULL_HANDLE; struct { uint32_t b, o; int i; } dp[96]; int nd = 0; uint64_t tb0 = tick();
    VkWriteDescriptorSet w[96]; VkDescriptorBufferInfo bi[96]; VkDescriptorImageInfo ii[96 * 4]; VkBufferView bv[96]; uint32_t nw = 0, nii = 0;
    int ncb = 0; uint8_t mci[MEMO_CB]; VkDescriptorBufferInfo mcd[MEMO_CB];   // the constant buffers, for the memo
    vkimg_t *kim[96]; int nkim = 0; uint64_t kdg0 = __atomic_load_n(&g_tex_dgen, __ATOMIC_ACQUIRE);   // the images a kept set names (kept_gen), and any view going meanwhile
    int keepable = 1;   // no descriptor of this submission's own memory (a driver segment or push block, a shadow) nor of a buffer that may be replaced (the spill buffer, texel buffer views): kept_set
    g_num_um = g_num_im = 0; memset(g_lod_shift[st], 0, sizeof g_lod_shift[st]);   // (tex_desc_for notes this set's shifts: none left from another set's)
    // shader-side virtual texturing (gfx_vk_vt.c, spv_vt.py): a shader with VT bindings has each candidate's texture recorded by its binding (vtb), its VT bindings written after
    int hasvt = 0; for (int i = 0; i < s->nbind; i++) hasvt |= s->b[i].kind >= RK_VT_POOL && s->b[i].kind <= RK_VT_FB;
    vt_bound_t vtb[64]; uint64_t vtb_set = 0;
    for (int i = 0; i < s->nbind && nw < 96; i++) {
        const dmsv_bind *b = &s->b[i];
        // (the kinds whose descriptors name only objects that outlive the submission: pool buffers, the support, zero and c1 buffers, views, samplers; any other kind keeps the set out of
        // the kept ones: driver segments and push blocks in scratch memory, the spill buffer, which grows, texel buffer views, virtual texturing's, kinds added later)
        keepable &= b->kind == RK_SUPPORT || b->kind == RK_C1 || b->kind == RK_CBUF || b->kind == RK_SBUF || b->kind == RK_UBO_PTR || b->kind == RK_SBUF_UBO || b->kind == RK_TEX ||
                    b->kind == RK_TEX_UBO || b->kind == RK_TEX_C2A || b->kind == RK_IMG || b->kind == RK_IMG_UBO || (b->kind == RK_DRV && s->dyn[i] && g_keep_fine);
        // (a driver segment or push block bound as a dynamic uniform buffer: the set names the slot's scratch block at offset 0, the segment's offset goes to the bind, so the set holds
        // while the block lives; one the slot frees flushes the kept sets naming it, gfx_vk_core.c scratch_trim. Nearly every shader has c0: as the slot's own memory those sets were
        // never kept, and vkUpdateDescriptorSets was 8% of the replay thread in Mario Kart World's race, run 17's simpleperf)
        VkWriteDescriptorSet *wr = &w[nw]; memset(wr, 0, sizeof *wr);
        wr->sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET; wr->dstSet = set; wr->dstBinding = b->binding; wr->descriptorCount = 1;
        VkBuffer buf; VkDeviceSize off, len;
        switch (b->kind) {
        case RK_SUPPORT: bi[nw] = (VkDescriptorBufferInfo){vk_support_buffer(), 0, 65536}; break;
        // a compute shader's shared memory past the device's limit, in a storage buffer (gfx_vk.h RK_SPILL: off words a workgroup, samp_arg slices; the 48 KB compute shaders of Mario Kart World)
        case RK_SPILL: bi[nw] = (VkDescriptorBufferInfo){vk_spill_buffer((VkDeviceSize)(b->off > 0 ? b->off : 1) * 4 * (VkDeviceSize)(b->samp_arg > 0 ? b->samp_arg : 1)), 0, VK_WHOLE_SIZE}; break;
        case RK_C1: vk_shader_c1(s, &buf, &off, &len); bi[nw] = (VkDescriptorBufferInfo){buf, off, len}; break;
        case RK_CBUF: cbuf_desc(st, s, b, &bi[nw]); keepable &= !g_cb_scratch || (s->dyn[i] && g_keep_fine); if (ncb < MEMO_CB) { mci[ncb] = (uint8_t)i; mcd[ncb++] = bi[nw]; } break;
        case RK_DRV: { VkBuffer sb; VkDeviceSize so;
            if (b->reg == DMSH_PUSH_BANK) { push_shared(st == 5, &sb, &so); bi[nw] = (VkDescriptorBufferInfo){sb, so, DMSH_PUSH_SIZE}; break; }   // c[24]: the NVN2 push block (push_shared)
            if (b->reg == DMSH_MESH_FIRST_SLOT) { uint32_t *q = vk_scratch(16, V.props.limits.minUniformBufferOffsetAlignment, &sb, &so); q[0] = g_mesh.first; q[1] = q[2] = q[3] = 0;   // a task or mesh shader's first workgroup (mesh_draw)
                bi[nw] = (VkDescriptorBufferInfo){sb, so, 16}; break; }
            drv_shared(st, b->reg, &sb, &so); bi[nw] = (VkDescriptorBufferInfo){sb, so, 0x1000}; if (st == 5 && b->reg == 0) { g_c0cs_b = sb; g_c0cs_o = so; } break; }
        case RK_SBUF: case RK_UBO_PTR: case RK_SBUF_UBO: {
            uint64_t a = 0, n = 0;
            if (b->kind == RK_SBUF && b->reg >= 0 && b->reg < 16) { a = S.ssbo[st][b->reg].addr; n = S.ssbo[st][b->reg].size; }
            else if (b->kind == RK_UBO_PTR && b->reg >= 0 && b->reg < 16) { a = S.ubo[st][b->reg].addr; n = S.ubo[st][b->reg].size; }
            else if (b->kind == RK_SBUF_UBO) { a = (uint64_t)ubo_word(st, b->reg, b->off) | (uint64_t)ubo_word(st, b->reg, b->off + 4) << 32; }
            uint64_t ga = a; vkpool_t *p = a ? vk_pool_gpu_at(&a, n ? n : 1) : NULL; if (!p && a && n > 1) p = vk_pool_gpu_at(&a, 1);   // (across two chunks or parts: cut at the first one's end)
            if (p) { buf = p->buf; off = a - p->lo; if (!n || off + n > p->hi - p->lo) n = p->hi - p->lo - off;
                // bound from the 256-byte boundary below it, the boundary the translator offsets every access from (shader_vk.py SB_ALIGN; a multiple of any device's minimum alignment)
                const VkDeviceSize al = SB_ALIGN; if (V.props.limits.minStorageBufferOffsetAlignment > al) VKRED_ONCE(41, "storage buffer alignment %llu above %llu", (unsigned long long)V.props.limits.minStorageBufferOffsetAlignment, (unsigned long long)al);
                bi[nw] = (VkDescriptorBufferInfo){buf, off - off % al, n + off % al};
                if (st == 5 && b->kind != RK_UBO_PTR) { void vk_gpu_wrote(uint64_t a, uint64_t n); vk_gpu_wrote(ga, n); }   // a dispatch's storage buffers: taken as written (at the address the game gave)
            } else bi[nw] = (VkDescriptorBufferInfo){vk_zero_buffer(), 0, 65536};
            break; }
        case RK_TEX: case RK_TEX_UBO: case RK_TEX_C2A: {
            int cnt = b->count ? b->count : 32; if (nii + (uint32_t)cnt > sizeof ii / sizeof *ii) cnt = 1;
            for (int k = 0; k < cnt; k++) {
                uint64_t h = b->kind == RK_TEX ? (b->reg >= 0 && b->reg < 256 ? S.tex[st][b->reg] : 0) :
                             b->kind == RK_TEX_UBO ? ubo_word(st, b->reg, b->off + 8 * k) : (b->off >= 0 ? c2_handle(st, b->off + 8 * k) : 0);
                g_tex_sseq = b->kind == RK_TEX && b->reg >= 0 && b->reg < 256 ? S.tex_sseq[st][b->reg] : 0;
                int num = tex_desc_for(st, b, h, &ii[nii + k], 0);
                if (t_desc_im) { if (nkim < 96) kim[nkim++] = (vkimg_t *)t_desc_im; else keepable = 0; } else if (!t_desc_dummy) keepable = 0;   // (a view of no known image: never kept)
                if (hasvt && cnt == 1 && b->binding < 64) { uint64_t sq; int sid = samp_sid(st, b, h, &sq); vtb[b->binding] = (vt_bound_t){t_desc_im, sid, sq, t_desc_vf, t_desc_shift, t_desc_ly0, t_desc_nly}; vtb_set |= 1ull << b->binding; }
                if (num && i < 64) { if (num == 1) g_num_um |= 1ull << i; else g_num_im |= 1ull << i; }
            }
            wr->pImageInfo = &ii[nii]; nii += (uint32_t)cnt; wr->descriptorCount = (uint32_t)cnt; wr->descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; nw++;
            continue; }
        case RK_VT_POOL: case RK_VT_CTL: case RK_VT_PT: case RK_VT_FB: continue;   // (after every texture binding: below)
        case RK_IMG: case RK_IMG_UBO: {
            uint64_t h = b->kind == RK_IMG ? (b->reg >= 0 && b->reg < 32 ? S.img[st][b->reg] : 0) : ubo_word(st, b->reg, b->off);
            tex_desc_for(st, b, h, &ii[nii], 1); wr->pImageInfo = &ii[nii++]; wr->descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE; nw++;
            if (t_desc_im) { if (nkim < 96) kim[nkim++] = (vkimg_t *)t_desc_im; else keepable = 0; } else if (!t_desc_dummy) keepable = 0;
            continue; }
        case RK_TBUF: case RK_TBUF_UBO: case RK_IMG_TBUF: {
            uint64_t h = b->kind == RK_TBUF ? (b->reg >= 0 && b->reg < 256 ? S.tex[st][b->reg] : 0) : b->kind == RK_IMG_TBUF ? (b->reg >= 0 && b->reg < 32 ? S.img[st][b->reg] : 0) : ubo_word(st, b->reg, b->off);
            const view_t *v; const tex_t *t = NULL; int tid = (int)(h & 0xfffff);
            bv[nw] = tid && vk_tex_by_id(tid, &v, &t) && t && t->d.target == NVN_TEXTURE_TARGET_BUFFER ? vk_tbuf_view(t) : VK_NULL_HANDLE;
            if (!bv[nw]) bv[nw] = vk_dummy_tbuf(b->kind == RK_IMG_TBUF);
            else if (b->kind != RK_IMG_TBUF && i < 64) { int num = view_num(t, NULL); if (num == 1) g_num_um |= 1ull << i; else if (num == 2) g_num_im |= 1ull << i; }   // texel buffers read their words alike
            wr->pTexelBufferView = &bv[nw]; wr->descriptorType = b->kind == RK_IMG_TBUF ? VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER : VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER; nw++;
            continue; }
        default: {   // unknown: the descriptor type it was declared with, filled with what reads zero
            if (b->desc == RD_UBO || b->desc == RD_SSBO) bi[nw] = (VkDescriptorBufferInfo){vk_zero_buffer(), 0, 65536};
            else if (b->desc == RD_SAMPLED || b->desc == RD_STORAGE) {
                ii[nii] = (VkDescriptorImageInfo){b->desc == RD_SAMPLED ? vk_dummy_sampler() : VK_NULL_HANDLE, vk_dummy_view(b->dim == 0xff ? 1 : b->dim, b->flags & 1, (b->flags & 2) != 0, b->desc == RD_STORAGE, 0), VK_IMAGE_LAYOUT_GENERAL};
                int cnt = b->count ? b->count : 32; for (int k = 1; k < cnt && nii + k < sizeof ii / sizeof *ii; k++) ii[nii + k] = ii[nii];
                wr->pImageInfo = &ii[nii]; nii += (uint32_t)cnt; wr->descriptorCount = (uint32_t)cnt;
                wr->descriptorType = b->desc == RD_SAMPLED ? VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER : VK_DESCRIPTOR_TYPE_STORAGE_IMAGE; nw++; continue;
            } else if (b->desc == RD_TBUF || b->desc == RD_STORAGE_TBUF) { bv[nw] = vk_dummy_tbuf(b->desc == RD_STORAGE_TBUF); wr->pTexelBufferView = &bv[nw];
                wr->descriptorType = b->desc == RD_TBUF ? VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER : VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER; nw++; continue; }
            else continue;
        }
        }
        wr->pBufferInfo = &bi[nw]; wr->descriptorType = b->desc == RD_SSBO ? VK_DESCRIPTOR_TYPE_STORAGE_BUFFER : VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        if (s->dyn[i]) {   // the offset goes to the bind, so the set's content (and its reuse) does not depend on it
            if (bi[nw].offset > UINT32_MAX) VKRED_ONCE(42, "dynamic uniform buffer offset past 4 GB");
            int j = nd++; while (j > 0 && dp[j - 1].b > b->binding) { dp[j] = dp[j - 1]; j--; }
            dp[j].b = b->binding; dp[j].o = (uint32_t)bi[nw].offset; dp[j].i = i; bi[nw].offset = 0; wr->descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC;
        }
        nw++;
    }
    if (hasvt) {   // a candidate's pool (reg: its texture's binding, off: k), then the control block that says what each candidate's texture is, the page table and the requests
        vt_bound_t cand[VT_NC]; int ncand = 0; memset(cand, 0, sizeof cand);
        for (int pass = 0; pass < 2; pass++) for (int i = 0; i < s->nbind && nw < 96; i++) {
            const dmsv_bind *b = &s->b[i];
            if (pass ? b->kind != RK_VT_CTL && b->kind != RK_VT_PT && b->kind != RK_VT_FB : b->kind != RK_VT_POOL) continue;
            VkWriteDescriptorSet *wr = &w[nw]; memset(wr, 0, sizeof *wr);
            wr->sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET; wr->dstSet = set; wr->dstBinding = b->binding; wr->descriptorCount = 1;
            if (!pass) { const vt_bound_t *c = b->reg >= 0 && b->reg < 64 && (vtb_set >> b->reg & 1) ? &vtb[b->reg] : NULL;
                if (b->off >= 0 && b->off < VT_NC) { if (c) cand[b->off] = *c; if (b->off + 1 > ncand) ncand = b->off + 1; }
                vk_vt_pool_desc(c, &ii[nii]); wr->pImageInfo = &ii[nii++]; wr->descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; nw++; continue; }
            if (b->kind == RK_VT_CTL) { vk_vt_ctl_desc(cand, ncand, &bi[nw]); wr->descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER; }
            else { if (b->kind == RK_VT_PT) vk_vt_pt_desc(&bi[nw]); else vk_vt_fb_desc(&bi[nw]); wr->descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; }
            wr->pBufferInfo = &bi[nw]; nw++;
        }
    }
    for (int j = 0; j < nd; j++) dofs[j] = dp[j].o;
    if (g_tracing) for (int j = 0; j < nd; j++) VKLOG("  stage %d binding %u: dynamic offset %u", st, dp[j].b, dp[j].o);
    if (g_tracing) for (uint32_t i = 0; i < nw; i++) {   // the refused draw (bisection): every descriptor it is given
        const VkWriteDescriptorSet *x = &w[i];
        if (x->pBufferInfo) VKLOG("  stage %d binding %u type %d: buffer %p +%llu size %llu", st, x->dstBinding, x->descriptorType, (void *)x->pBufferInfo->buffer, (unsigned long long)x->pBufferInfo->offset, (unsigned long long)x->pBufferInfo->range);
        else if (x->pImageInfo) for (uint32_t k = 0; k < x->descriptorCount && k < 4; k++) VKLOG("  stage %d binding %u[%u] type %d: view %p sampler %p", st, x->dstBinding, k, x->descriptorType, (void *)x->pImageInfo[k].imageView, (void *)x->pImageInfo[k].sampler);
        else if (x->pTexelBufferView) VKLOG("  stage %d binding %u type %d: texel buffer view %p", st, x->dstBinding, x->descriptorType, (void *)x->pTexelBufferView[0]);
    }
    // the set's content, then an earlier identical set of this submission or a new one
    uint64_t tb1 = tick(); g_ns_set_build += tb1 - tb0;
    // (a word per field, fnv's word step: hashed as bytes, the three 4-byte fields of each write took fnv's byte-by-byte tail, 12 steps where one does, and an image info's padding
    // was hashed with it, stack garbage that could tell identical sets apart; vkbench: hashing was ~6% of the replay thread)
    uint64_t h = fnv_w(0xcbf29ce484222325ull, (uint64_t)(uintptr_t)s->dsl);
    for (uint32_t i = 0; i < nw; i++) {
        const VkWriteDescriptorSet *x = &w[i]; h = fnv_w(h, (uint64_t)x->dstBinding | (uint64_t)x->descriptorCount << 24 | (uint64_t)x->descriptorType << 40);
        if (x->pBufferInfo) { h = fnv_w(h, (uint64_t)(uintptr_t)x->pBufferInfo->buffer); h = fnv_w(h, x->pBufferInfo->offset); h = fnv_w(h, x->pBufferInfo->range); }
        else if (x->pImageInfo) for (uint32_t k = 0; k < x->descriptorCount; k++) { const VkDescriptorImageInfo *q = &x->pImageInfo[k];
            h = fnv_w(h, (uint64_t)(uintptr_t)q->sampler); h = fnv_w(h, (uint64_t)(uintptr_t)q->imageView ^ (uint64_t)q->imageLayout << 60); }
        else if (x->pTexelBufferView) h = fnv_w(h, (uint64_t)(uintptr_t)x->pTexelBufferView[0]);
    }
    unsigned k = (unsigned)(h % SETC); uint64_t tb2 = tick(); g_ns_set_hash += tb2 - tb1;
    if (g_setc[k].epoch == g_set_epoch && g_setc[k].h == h && g_setc[k].l == s->dsl && !g_tracing && !g_x_nosetc) { g_set_hits++; set = g_setc[k].set; }
    else if (keepable && g_keep_on && !g_tracing && !g_x_nosetc && (set = kept_get(h, s->dsl))) { g_set_hits++; g_set_kept++;   // an earlier submission's set of this content
        g_setc[k].h = h; g_setc[k].epoch = g_set_epoch; g_setc[k].l = s->dsl; g_setc[k].set = set; }
    else { int kept = keepable && g_keep_on && !g_tracing && !g_x_nosetc;
        if (kept) {   // its images marked first, then no view gone since its collection began: a view that goes later bumps g_tex_kgen (vk_tex_view_drop_all), one that went meanwhile keeps it out
            uint64_t kg = __atomic_load_n(&g_kgen, __ATOMIC_ACQUIRE); for (int j = 0; j < nkim; j++) __atomic_store_n(&kim[j]->kept_gen, kg, __ATOMIC_RELEASE);
            __atomic_thread_fence(__ATOMIC_SEQ_CST); if (__atomic_load_n(&g_tex_dgen, __ATOMIC_ACQUIRE) != kdg0) { kept = 0; g_set_kraced++; } }
        kept = kept && (set = kept_new(s->dsl)) != VK_NULL_HANDLE; if (!kept) set = vk_alloc_set(s->dsl); g_set_new++;
        for (uint32_t i = 0; i < nw; i++) w[i].dstSet = set;
        vkUpdateDescriptorSets(V.dev, nw, w, 0, NULL);
        if (kept) { kept_bufs(w, nw); kept_put(h, s->dsl, set); }
        g_setc[k].h = h; g_setc[k].epoch = g_set_epoch; g_setc[k].l = s->dsl; g_setc[k].set = set; g_ns_set_new += tick() - tb2; }
    // verify: the same content (hash), integer-texel bindings and dynamic offsets as the memo gave. A lone difference in dofs alone can be a driver segment
    // copied again after its g_drvc entry was taken by another (the memo's copy is as good); anything repeated is a dependency the memo misses
    if (verify && (h != g_smemo[st].h || g_num_um != vum || g_num_im != vim || memcmp(vd, dofs, sizeof *dofs * (size_t)s->ndyn))) { g_set_memo_bad++; static int n;
        if (n++ < 20) VKLOG("[RED] descriptor set memo: stage %d shader %016llx: a fresh set differs from the memo's (hash %016llx / %016llx, um %#llx / %#llx, im %#llx / %#llx, dynamic offsets %s)", st, (unsigned long long)s->key,
                            (unsigned long long)h, (unsigned long long)g_smemo[st].h, (unsigned long long)g_num_um, (unsigned long long)vum, (unsigned long long)g_num_im, (unsigned long long)vim, memcmp(vd, dofs, sizeof *dofs * (size_t)s->ndyn) ? "differ" : "same"); }
    if (memo) { set_memo_t *m = &g_smemo[st];   // (set_memo_ok: at most MEMO_CB constant buffers)
        m->s = s; m->epoch = g_set_epoch; m->bind = g_bind_gen[st]; m->ubo = g_ubo_gen[st]; m->ubd = g_ubd_gen; m->tex = gtex; m->samp = gsamp; m->pool = gpool; m->binst = g_draw_binst;
        m->h = h; m->um = g_num_um; m->im = g_num_im; m->set = set; memcpy(m->ls, g_lod_shift[st], sizeof m->ls); memcpy(m->dofs, dofs, sizeof *dofs * (size_t)s->ndyn); m->ncb = ncb; memcpy(m->ci, mci, (size_t)ncb); memcpy(m->cd, mcd, sizeof *mcd * (size_t)ncb);
        for (int c = 0; c < ncb; c++) { m->cpos[c] = 0xff; for (int j = 0; j < nd; j++) if (dp[j].i == mci[c]) m->cpos[c] = (uint8_t)j; } }
    return set;
}

// ---------------- draws ----------------
// What the draws of the command buffer being recorded set last: a draw that would set the same again records nothing. Vulkan keeps dynamic state and
// bindings for the whole command buffer, across render passes and across pipelines that declare the same dynamic state (every one of ours, build_pipeline);
// a new command buffer (g_cmd_gen, gfx_vk_core.c) starts empty. Xenoblade Chronicles 3 docked: ~1,900 draws a frame, each with 10 dynamic-state calls and
// its binds, most the same as the draw before; the driver's time grows with what is recorded (vkEndCommandBuffer too). DM_VK_X_NOBC=1: everything every draw
extern uint64_t g_cmd_gen;
static struct { uint64_t gen; VkViewport vp; VkRect2D sc; float bc[4]; int sm[2], sw[2], sr[2], line; p_poly_offset po; VkPipelineLayout lay; uint32_t nsets, ndofs;
                VkDescriptorSet sets[5]; uint32_t dofs[3 * 96]; VkBuffer vb[17]; VkDeviceSize vo[17]; VkBuffer ib; VkDeviceSize io; VkIndexType it; } g_bc;

static int bc_fresh(void) { if (g_bc.gen != g_cmd_gen || g_x_nobc || g_bc_stale) { g_bc_stale = 0; memset(&g_bc, 0, sizeof g_bc); g_bc.gen = g_cmd_gen; g_bc.it = (VkIndexType)-1; return 1; } return 0; }
#define BC_SAME(field, val) (!fresh && !memcmp(&g_bc.field, &(val), sizeof g_bc.field))
static void bc_index(VkCommandBuffer cb, VkBuffer b, VkDeviceSize o, VkIndexType t) {
    int fresh = bc_fresh(); if (!fresh && g_bc.ib == b && g_bc.io == o && g_bc.it == t) return;
    vkCmdBindIndexBuffer(cb, b, o, t); g_bc.ib = b; g_bc.io = o; g_bc.it = t;
}
static void set_dynamic(VkCommandBuffer cb) {
    int fresh = bc_fresh();
    float x = S.vp[0][0], y = S.vp[0][1], w = S.vp[0][2], h = S.vp[0][3];
    if (w == 0 && h == 0) { w = (float)S.pass_w; h = (float)S.pass_h; }
    VkViewport vp = {x, y, w, h, S.dnear, S.dfar};
    if (g_lower_left && !g_flipy) { vp.y = (float)S.pass_h - y; vp.height = -h; }   // lower-left viewports count y from the bottom
    else if (!g_lower_left && g_flipy) { vp.y = y + h; vp.height = -h; }        // clip space y up over the same top-left rectangle
    if (!g_depth_zero && !V.has_depth_clip_ctl) { /* GL depth convention without clip control: left as is (rare) */ }
    // Viewport swizzle (Absolum's FNA3D sets NEGATIVE_Y every frame; Advance Wars 1+2's Unity sets it too): NEGATIVE_Y negates clip y, the same as the viewport flipped about
    // its centre; facing is decided after the viewport transform in Vulkan as on NVN, so the winding stays. Other swizzles are drawn unswizzled (nvn_mtl.m does the same)
    { uint32_t sw; memcpy(&sw, S.vp_sw[0], 4);
      if (sw == 0x100) { vp.y += vp.height; vp.height = -vp.height; }
      else if (sw) VKRED_ONCE(81, "viewport swizzle %d,%d,%d,%d not implemented (drawn unswizzled)", S.vp_sw[0][0], S.vp_sw[0][1] ^ 2, S.vp_sw[0][2] ^ 4, S.vp_sw[0][3] ^ 6); }
    if (vp.minDepth < 0) vp.minDepth = 0; if (vp.maxDepth > 1) vp.maxDepth = 1;
    if (!BC_SAME(vp, vp)) { vkCmdSetViewport(cb, 0, 1, &vp); g_bc.vp = vp; }
    VkRect2D sc = scissor_rect(); if (!BC_SAME(sc, sc)) { vkCmdSetScissor(cb, 0, 1, &sc); g_bc.sc = sc; }
    { int32_t vx0 = (int32_t)fmaxf(vp.x, 0), vy0 = (int32_t)fmaxf(fminf(vp.y, vp.y + vp.height), 0), vx1 = (int32_t)(vp.x + vp.width), vy1 = (int32_t)fmaxf(vp.y, vp.y + vp.height);   // what this draw can touch
      int32_t x0 = sc.offset.x > vx0 ? sc.offset.x : vx0, y0 = sc.offset.y > vy0 ? sc.offset.y : vy0, x1 = sc.offset.x + (int32_t)sc.extent.width < vx1 ? sc.offset.x + (int32_t)sc.extent.width : vx1, y1 = sc.offset.y + (int32_t)sc.extent.height < vy1 ? sc.offset.y + (int32_t)sc.extent.height : vy1;
      if (x0 < g_pass_bb[0]) g_pass_bb[0] = x0; if (y0 < g_pass_bb[1]) g_pass_bb[1] = y0; if (x1 > g_pass_bb[2]) g_pass_bb[2] = x1; if (y1 > g_pass_bb[3]) g_pass_bb[3] = y1; }
    if (!BC_SAME(bc, S.blendc)) { vkCmdSetBlendConstants(cb, S.blendc); memcpy(g_bc.bc, S.blendc, sizeof g_bc.bc); }
    if (!BC_SAME(sm, S.smask)) { vkCmdSetStencilCompareMask(cb, VK_STENCIL_FACE_FRONT_BIT, (uint32_t)S.smask[0]); vkCmdSetStencilCompareMask(cb, VK_STENCIL_FACE_BACK_BIT, (uint32_t)S.smask[1]); memcpy(g_bc.sm, S.smask, sizeof g_bc.sm); }
    if (!BC_SAME(sw, S.swmask)) { vkCmdSetStencilWriteMask(cb, VK_STENCIL_FACE_FRONT_BIT, (uint32_t)S.swmask[0]); vkCmdSetStencilWriteMask(cb, VK_STENCIL_FACE_BACK_BIT, (uint32_t)S.swmask[1]); memcpy(g_bc.sw, S.swmask, sizeof g_bc.sw); }
    if (!BC_SAME(sr, S.sref)) { vkCmdSetStencilReference(cb, VK_STENCIL_FACE_FRONT_BIT, (uint32_t)S.sref[0]); vkCmdSetStencilReference(cb, VK_STENCIL_FACE_BACK_BIT, (uint32_t)S.sref[1]); memcpy(g_bc.sr, S.sref, sizeof g_bc.sr); }
    if (!BC_SAME(po, S.pofs)) { vkCmdSetDepthBias(cb, S.pofs.units, S.pofs.clamp, S.pofs.factor); g_bc.po = S.pofs; }
    if (fresh || !g_bc.line) { vkCmdSetLineWidth(cb, 1.0f); g_bc.line = 1; }
}
static int g_no_cull = -1;
// When the driver refuses a submission (vkEndCommandBuffer), the next ones find the draw it refuses: only draws below a limit are recorded, the limit
// bisected between a submission that passed and one that failed; the draw found is logged with its whole state (vk_bisect_result, gfx_vk_core.c)
int g_bis_on, g_bis_lo, g_bis_hi, g_bis_trace = -1, g_bis_var; static int g_draw_i;   // g_bis_var: the change tried on the found draw (1 depth bias values 0, 2 no depth bias, 3 dummy textures, 4 vertex offset 0, 5 zero vertex buffer, 6 a new pass)
static int draw_allowed(void) { int i = g_draw_i++; if (i == g_bis_trace) return g_bis_var == 7 ? 0 : 2;
    return !g_bis_on || i < (g_bis_lo + g_bis_hi) / 2; }
void vk_bisect_reset_count(void) { g_draw_i = 0; }
static int dbg_draw(void) { static int v = -1; if (v < 0) { const char *e = getenv("DM_VK_DBG_DRAW"); v = e ? atoi(e) : 0; } return v; }   // debugging: 1 no draw calls, 2 no vertex buffers, 4 no descriptor sets
// a transfer read of 4 bytes of each buffer the addresses are in (outside a render pass, between barriers): see prepare_draw's runs of rasterizer discard
static void rd_read(const uint64_t *a, int n) {
    VkCommandBuffer cb = vk_cmd(); VkBuffer seen[16]; int ns = 0;
    for (int i = 0; i < n; i++) { VkBuffer b, db; VkDeviceSize o, dof; if (!vk_buf_at(a[i], 4, &b, &o)) continue;
        int dup = 0; for (int j = 0; j < ns; j++) dup |= seen[j] == b; if (dup) continue; seen[ns++] = b;
        vk_scratch(4, 4, &db, &dof); VkBufferCopy r = {o, dof, 4}; vk_img_barrier_all(cb); vkCmdCopyBuffer(cb, b, db, 1, &r); vk_img_barrier_all(cb); }
}
// everything a draw needs bound; 0 when the draw cannot be done (logged once per reason)
static int prepare_draw(int prim, int indexed) {
    // DM_VK_CB_DRAWS (read every submission, default 128, 0 = never): the replay is cut every that many draws and what is recorded goes to the GPU at once
    // (vk_cmd_split), so the GPU works on the frame while the rest is replayed instead of after all of it: NEO's Scramble Crossing 36.7 → 15.7 ms a frame
    g_cb_draws++;   // the cut itself happens between passes (begin_pass); inside one only with DM_VK_PASS_SPLIT (below): a pass cut in two stores and reloads its attachments
    g_tracing = g_draw_i == g_bis_trace && g_bis_trace >= 0;
    // the change being tried applies to the found draw and every later one (the later draws are refused too: changing one draw alone proves nothing)
    int var = g_bis_trace >= 0 && g_draw_i >= g_bis_trace ? g_bis_var : 0; if (g_bis_var) g_tracing = 0;   // full logs only the first time
    g_var_now = 0;
    if (var == 6) vk_end_pass();   // a new pass for this draw
    // A program may have no fragment stage (a depth-only draw: NVN binds by stage mask, and Metroid Dread draws about nine a frame so in its intro; nvn_mtl.m makes them vertex-only pipelines):
    // drawn with no fragment shader and no colour written (Vulkan leaves colour outputs undefined without one), as before only rasterizer-discard draws were
    if (!dsh(0)) { g_vkst.skip_prog++; VKRED_ONCE(50, "draw with a program whose shaders are not in the bundle: skipped"); return 0; }
    if (g_x_nogs && S.sh[2] && !g_mesh.on) return 0;   // DM_VK_X_NOGS=1 (live, A/B): draws with a geometry shader are left out
    for (int i = 0; i < g_x_nskipvs; i++) if (dsh(0)->key == g_x_skipvs[i]) return 0;   // DM_VK_X_SKIPVS=<key>[,<key>...] (live, up to 8): draws with these vertex shaders are left out
    // DM_VK_X_SKIPVS_MOD=<m>:<r> (live): also the draws whose vertex (or mesh) shader key mod m is r, in passes with a depth target, a bisection by halves; from m 64 on, the first
    // 48 keys it leaves out are logged
    // (in passes with a depth target only: a full-screen pass's quad left out blacked the picture)
    if (g_x_skipvs_m && S.rt_has_dep && dsh(0)->key % g_x_skipvs_m == g_x_skipvs_r) { static uint64_t seen[48]; static int nseen; static unsigned gen;
        if (gen != g_x_skipvs_gen) { gen = g_x_skipvs_gen; nseen = 0; }
        if (g_x_skipvs_m >= 64 && nseen < 48) { int k = 0; while (k < nseen && seen[k] != dsh(0)->key) k++;
            if (k == nseen) { seen[nseen++] = dsh(0)->key; VKLOG("DM_VK_X_SKIPVS_MOD %llu:%llu leaves out vertex shader %016llx (%s)", (unsigned long long)g_x_skipvs_m, (unsigned long long)g_x_skipvs_r,
                                                                 (unsigned long long)dsh(0)->key, g_mesh.on ? "mesh" : "vertex"); } }
        return 0; }
    // A run of draws with rasterizer discard gets a pass of its own: the pass ends where such a run begins and where it ends (and begin_pass puts a barrier
    // before the next). They are vertex shaders that write memory the later draws read: Xenoblade Chronicles 3 transforms its meshes with point draws
    // (724ddce6 then 37d04447, cf66cef4), which share passes with its G-buffer draws. NVN's GPU runs draws in order; Adreno, a tiled GPU, runs the vertex
    // shaders of a pass's draws in its binning pass, and draws read vertices not yet written: thin black slivers over the cliffs and grass in 17-20 of
    // 361 frames, 0 with this (0 also with the draws of 37d04447 and cf66cef4 left out). A pass for each such draw instead, splitting the runs, lost
    // Noah's hair (724ddce6's draws come in pairs in one pass); the run kept together draws it. About +1 ms of replay, +0.1 ms of GPU a frame, 33.2 ms
    // frames held. DM_VK_X_RDKEEP=1 (live, A/B): they stay in the pass
    // Where a run ends, the memory its shaders wrote is read by a transfer (rd_read): with barriers alone, the draws of the next passes still met vertices the
    // run had not written yet. Xenoblade Chronicles 3 sways its grass with point draws (5f6491f9 for its depth pre-pass, c0882609 for its G-buffer) whose records
    // the grass cards draw from, the G-buffer's with depth EQUAL against the pre-pass: whole blades failed the test and stayed black, a different few each frame
    // (the grass flicker, stable on the Switch). Grass blades the G-buffer never wrote: 900-2,300 a frame, 0 with this (8 dumps); flicker on screen 4.2-7.7
    // spiking blocks a frame, 0.0-0.5. Everything the two passes read matched (records over 60 frames, instances, index data, shader code: the G-buffer's
    // vertex shader drawing the pre-pass too changed nothing), and a full GPU wait after each submission, explicit barrier bits or a barrier to vertex input
    // did not help; drawing from a copy of the records did, as did a command buffer cut or a transfer read of the written buffer at either end of the run.
    // About +0.45 ms of GPU a frame, 33.2 ms frames held. DM_VK_RD_TOUCH=0 (live, A/B): no read
    if (!g_mesh.on) { static int prev; static uint64_t wa[16]; static int nwa; int rd = S.rast_discard; if (!g_x_rdkeep && rd != prev && S.pass_open) vk_end_pass();
      if (rd) { if (!prev) nwa = 0; for (int r = 0; r < 16 && nwa < 16; r++) { uint64_t x = S.ssbo[0][r].addr; int have = 0; for (int i = 0; i < nwa; i++) have |= wa[i] == x; if (x && !have) wa[nwa++] = x; } }
      else if (prev && g_rd_touch && !S.pass_open) rd_read(wa, nwa);
      prev = rd; }
    // DM_VK_PASS_SPLIT=n (live, 0 = off): a pass is ended after every n of its draws and begun again on the same attachments (every load op LOAD: clears folded into
    // the first part's load ops are not run again), with a cut before it (begin_pass, g_psplit_now), so its first parts reach the GPU while the replay records the rest:
    // DM_VK_CB_DRAWS cuts only between passes, and Unicorn Overlord's battle map draws ~2,030 of its ~3,546 draws in one 1280x720 pass (13.3 ms of GPU, all after its
    // replay: the 15.5 ms GPU tail). Not inside a run of rasterizer discard (it stays in one pass, see above), and only a pass without depth: a pass that goes on with a
    // depth buffer begun in an earlier command buffer meets Adreno's stale LRZ (begin_pass). DM_VK_PASS_SPLIT_DEPTH=1 (live, A/B) splits those too, with begin_pass's
    // write-back of a depth texel (lrz_touch) before the second part. The pipeline, dynamic state, sets and buffers are bound again (vk_end_pass, g_cmd_gen); deferred
    // uniform updates stay deferred (the draws read their shadows), as at any cut. Only while vk_cmd_split can still begin a part: otherwise the pass would be cut for nothing
    if (g_psplit > 0 && g_x_cbdraws > 0 && S.pass_open == 1 && g_pass_draws >= (uint64_t)g_psplit && !S.rast_discard && (!S.daspect || g_psplit_depth)) {
        int vk_cmd_parts_left(void); if (vk_cmd_parts_left() > 0) { { static int n; if (n++ < 3) VKLOG("pass split: pass %d (%ux%u) after %llu draws, depth %d", g_sub_pass - 1, S.pass_w, S.pass_h, (unsigned long long)g_pass_draws, S.dfmt); }
            vk_end_pass(); g_psplit_now = 1; } }
    else if (g_psplit > 0 && S.pass_open == 1 && g_pass_draws == (uint64_t)g_psplit && S.daspect && !g_psplit_depth) { static int n; if (n++ < 3) VKLOG("pass split: pass %d (%ux%u) has depth (format %d): not split", g_sub_pass - 1, S.pass_w, S.pass_h, S.dfmt); }
    if (!begin_pass() || !pass_commit()) { g_vkst.skip_rt++; return 0; }
    if (g_x_tracepass >= 0 && g_sub_pass - 1 == g_x_tracepass) g_tracing = 1;
    uint64_t t0 = tick();
    VkCommandBuffer cb = vk_cmd();
    if (g_no_cull < 0) g_no_cull = getenv("DM_VK_NOCULL") != NULL;
    pkey k; memset(&k, 0, sizeof k);
    for (int i = 0; i < 3; i++) k.key[i] = dsh(i) ? dsh(i)->key : 0;
    k.mesh = g_mesh.on ? 1u | (g_mesh.notask ? 2u : 0) | (g_mesh_posblock ? 4u : 0) | (g_mesh_vistate ? 8u : 0) : 0;   // (the A/B variants: g_mesh_nofdz above)
    k.topo = topo(prim); k.restart = S.prim_restart && (k.topo == VK_PRIMITIVE_TOPOLOGY_LINE_STRIP || k.topo == VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP || k.topo == VK_PRIMITIVE_TOPOLOGY_TRIANGLE_FAN) && indexed;
    k.raster_discard = S.rast_discard; k.alt = (uint32_t)g_spv_alt; k.cull = g_no_cull ? 0 : S.poly.cull;
    // counter-clockwise front faces in Vulkan's framebuffer: Vulkan's clip space has y down, Metal's up (nvn_mtl.m: CCW when (front == CCW) != lower left), and the
    // lower-left origin mode flips the viewport again, so CCW when (front == CCW) == lower left. Taken as given, upper-left games (NEO) culled their ground and walls
    k.front = ((S.poly.front == NVN_FRONT_FACE_CCW) == ((g_queue_ll >= 0 ? g_queue_ll : g_lower_left) != 0)) != (g_flipy != 0); k.mode = S.poly.mode; k.depth_clamp = S.depth_clamp;   // (the queue's raster flip)
    k.bias_en = (S.poly.offset_en & (S.poly.mode == 0 ? 1 : S.poly.mode == 1 ? 2 : 4)) != 0;
    if (g_mesh.on && g_mesh_fill) { k.mode = 2; k.bias_en = (S.poly.offset_en & 4) != 0; }   // DM_VK_MESH_FILL (NVN_POLYGON_MODE_FILL)
    if (var == 2) k.bias_en = 0;
    if (var == 8) k.raster_discard = 1;   // no fragment shader in the pipeline
    int has_depth = (S.daspect & VK_IMAGE_ASPECT_DEPTH_BIT) != 0, has_stencil = (S.daspect & VK_IMAGE_ASPECT_STENCIL_BIT) != 0;
    k.dt = has_depth && S.ds.dt; k.dw = S.ds.dw; k.df = S.ds.df; k.st = has_stencil && S.ds.st;
    if (g_x_logfs && S.sh[1] && S.sh[1]->key == g_x_logfs) VKLOG("logfs: present %llu pass %d (%ux%u, %llu draws before): depth attachment %d test %d (state %d) func %u write %u; depth range %.9g %.9g; viewport %.1f %.1f %.1f %.1f",
        (unsigned long long)g_tex_frame, g_sub_pass - 1, S.pass_w, S.pass_h, (unsigned long long)g_pass_draws, has_depth, k.dt, S.ds.dt, k.df, k.dw, S.dnear, S.dfar, S.vp[0][0], S.vp[0][1], S.vp[0][2], S.vp[0][3]);
    // A depth buffer that a fragment shader wrote through gl_FragDepth: Adreno's early culling (LRZ) then rejects whole triangles of later draws that test it, in some frames
    // and not others, with the same draws, state and depth contents. Hades II downsamples its scene depth into a 640x360 D24S8 with such a shader (Blit2DDepth), then
    // draws its lights into it with the depth test on and no depth write: the lights went out in turn every 24-48 frames (whole halves of light quads missing in pass dumps),
    // and Xenoblade Chronicles 3's cloud dome (its depth downsampled the same way) lost whole triangles. The light shader rewritten to write gl_FragDepth = gl_FragCoord.z
    // (the same depth, tested after the shader) brought the pulses to 0 in 5 recordings, against 8-21; neither writing the depth image back by transfer (one texel or the
    // whole image) nor waiting for the GPU after each submission changed them. So draws that test such a buffer (written in the last two presents) use that variant.
    // Not with a depth bias (gl_FragCoord.z need not carry it). DM_VK_FDZ=0 (live): never
    { vkimg_t *dim = has_depth ? P_att[8].im : NULL; vkshader_t *fsh = k.raster_discard ? NULL : S.sh[1];
      if (dim && fsh && fsh->depth_repl && k.dt && k.dw) dim->fdz_frame = g_tex_frame + 1;
      k.fdz = g_fdz && dim && fsh && !fsh->depth_repl && k.dt && !k.bias_en && dim->fdz_frame && g_tex_frame + 1 - dim->fdz_frame <= 2 && (g_fdz != 2 || !k.dw);
      if (g_mesh.on && g_mesh_nofdz) k.fdz = 0; }   // DM_VK_MESH_NOFDZ
    k.sf[0] = S.ds.sf_f; k.sf[1] = S.ds.sf_b; k.sfail[0] = S.ds.fail_f; k.szfail[0] = S.ds.zfail_f; k.szpass[0] = S.ds.zpass_f; k.sfail[1] = S.ds.fail_b; k.szfail[1] = S.ds.zfail_b; k.szpass[1] = S.ds.zpass_b;
    // vertex input: every location the vertex shader reads
    vkshader_t *vsh = g_mesh.on ? NULL : S.sh[0]; uint32_t in_mask = vsh ? vsh->local[0] : 0, int_mask = vsh ? vsh->local[1] | vsh->local[2] : 0, sint_mask = vsh ? vsh->local[1] : 0;
    uint32_t used_streams = 0; int need_zero = 0;
    for (int loc = 0; loc < 32; loc++) {
        if (!(in_mask >> loc & 1)) continue;
        va_st a; memcpy(&a, &S.vattr[loc], 4);
        VkFormat f = a.fmt ? vertex_format(a.fmt, int_mask >> loc & 1, sint_mask >> loc & 1) : VK_FORMAT_UNDEFINED;
        if (f == VK_FORMAT_UNDEFINED || a.stream >= 16 || !S.vbuf[a.stream].address) { k.attr[k.nattr++] = (typeof(k.attr[0])){(uint32_t)loc, 16, int_mask >> loc & 1 ? VK_FORMAT_R32G32B32A32_UINT : VK_FORMAT_R32G32B32A32_SFLOAT, 0}; need_zero = 1; continue; }
        k.attr[k.nattr++] = (typeof(k.attr[0])){(uint32_t)loc, a.stream, f, a.offset};
        used_streams |= 1u << a.stream;
    }
    for (int s = 0; s < 16; s++) if (used_streams >> s & 1) k.bind[k.nbind++] = (typeof(k.bind[0])){(uint32_t)s, S.vstream[s].stride, S.vstream[s].divisor ? 1u : 0u, S.vstream[s].divisor};
    if (need_zero) k.bind[k.nbind++] = (typeof(k.bind[0])){16, 0, 0, 0};
    // attachments and blending
    k.ncol = (uint32_t)S.ncol; k.dfmt = S.dfmt; k.samples = S.samples ? S.samples : 1; k.a2c = S.ms.a2c;
    k.logic_en = S.color.logic_op != 3 && S.color.logic_op != 0 && 0;   // logic ops: not on mobile render targets in version 1
    if (S.color.logic_op != 3 && S.color.logic_op != 0) { static int n; if (n++ < 4) VKLOG("[RED] logic operation %u asked (NVN ColorStateSetLogicOp): left out, the draw writes its colours as COPY (upstream draws it, dmsh_logic.py)", S.color.logic_op); }
    k.logic_op = S.color.logic_op;
    for (int i = 0; i < S.ncol; i++) {
        k.cfmt[i] = S.cfmt[i];
        // no fragment shader: no colour written. A colour target the fragment shader declares no output for keeps its texels: Maxwell writes only the targets in the shader's output map,
        // while Vulkan leaves a target without an output undefined unless its write mask is 0 (upstream "No writes to colour outputs the shader does not write": Super Mario 3D World's
        // volume draws write outputs 0 and 2 with every channel of targets 1 and 3 enabled). Mario Kart World 1.8.0: 4,888 of its 7,301 pixel shaders write one output, 2,248 two
        k.wmask[i] = S.cfmt[i] && S.sh[1] && (S.sh[1]->out_locs >> i & 1) ? (S.chmask.mask >> (4 * i)) & 0xf : 0;
        const vkfmt_t *f = S.rt_has_col[i] ? vk_fmt(S.rt_col[i].d.format) : NULL;
        if (((S.color.blend_en >> i) & 1) && f && !f->is_int) {
            k.blend_en |= 1u << i;
            k.bl[i].src = S.blend[i].src_rgb; k.bl[i].dst = S.blend[i].dst_rgb; k.bl[i].srca = S.blend[i].src_a; k.bl[i].dsta = S.blend[i].dst_a; k.bl[i].eq = S.blend[i].eq_rgb; k.bl[i].eqa = S.blend[i].eq_a;
            if (S.blend[i].adv) { uint32_t o[6]; if (adv_blend(&S.blend[i], i, o)) { k.bl[i].src = o[0]; k.bl[i].dst = o[1]; k.bl[i].srca = o[2]; k.bl[i].dsta = o[3]; k.bl[i].eq = o[4]; k.bl[i].eqa = o[5]; } }   // (adv_blend)
        }
    }
    // NVN's alpha test (ColorStateSetAlphaTest, its reference SetAlphaRef: S.alpha_ref): the fragment shader's alpha test variant (gfx_vk_shader.c spv_alpha_test), which reads the function and
    // the reference as push constants (below, after the pipeline), so one pipeline serves them all (upstream ee369f09 atest_key). No test without a fragment shader, nor into an integer colour 0
    // target (as OpenGL's rule and upstream); a shader whose colour 0 is no float vec4 is drawn without it, a [RED] once
    k.atest = 0;
    if (g_alpha_on && S.color.alpha_func && S.color.alpha_func != NVN_ALPHA_FUNC_ALWAYS && dsh(1) && !k.raster_discard) {   // (DM_VK_ALPHA=0: as before the test came)
        const vkfmt_t *f0 = S.ncol > 0 && S.rt_has_col[0] ? vk_fmt(S.rt_col[0].d.format) : NULL;
        if (f0 && f0->is_int) k.atest = 0;   // (an integer target: no test)
        else if (dsh(1)->col0 == VK_COL0_FLOAT) k.atest = 1;
        else VKRED_ONCE(159, "alpha test (function %u, reference %g) with fragment shader %016llx, whose colour 0 %s: drawn without the test", (unsigned)S.color.alpha_func, S.alpha_ref,
                        (unsigned long long)dsh(1)->key, dsh(1)->col0 == VK_COL0_NONE ? "is not written" : "has no float alpha");
    }
    // Dual-source blending: NVN blends colour target 0 with the shader's output 1 as its second source (factors SRC1_*, 0x10-0x13), the translator declares that output at
    // location 1, and Vulkan takes the second source only from location 0, index 1 (with location 1 the blend reads undefined values: Pokémon Legends: Arceus's post effects,
    // shader ha_post_effects variation 4, washed the whole picture out; the validation layer named it). Such a pipeline takes the fragment shader's dual-source variant
    #define SRC1F(x) ((x) >= 0x10 && (x) <= 0x13)
    k.dual = (k.blend_en & 1) && S.sh[1] && !k.raster_discard && (SRC1F(k.bl[0].src) || SRC1F(k.bl[0].dst) || SRC1F(k.bl[0].srca) || SRC1F(k.bl[0].dsta));
    #undef SRC1F
    k.clip_neg1 = !g_depth_zero;
    // layout and pipeline
    VkDescriptorSetLayout dsl[5] = {0}; int nsets = 0;
    for (int st = 0; st < 3; st++) if (dsh(st) && !(st == 1 && k.raster_discard)) { dsl[dsh(st)->set] = dsh(st)->dsl; if (dsh(st)->set + 1 > nsets) nsets = dsh(st)->set + 1; }
    // the last draw's set layouts and what vk_pipeline_layout made of them (it fills absent stages with the empty layout): its lock and hash took ~10% of the
    // replay thread on Unicorn Overlord's battle map. Pipeline layouts are never destroyed
    static VkDescriptorSetLayout l_in[5], l_out[5]; static int l_n = -1; static VkPipelineLayout l_lay; VkPipelineLayout layout;
    if (nsets == l_n && !memcmp(dsl, l_in, sizeof l_in)) { memcpy(dsl, l_out, sizeof l_out); layout = l_lay; }
    else { memcpy(l_in, dsl, sizeof l_in); l_n = nsets; layout = l_lay = vk_pipeline_layout_gfx(dsl, nsets); memcpy(l_out, dsl, sizeof l_out); }
    // descriptor sets first: the views they are given decide which shader variants the pipeline takes (integer texels, vk_shader_module)
    g_var_now = var;
    VkDescriptorSet sets[5] = {0};
    uint32_t dofs[5][96], ndofs = 0, dall[3 * 96];
    uint64_t t1d = tick();
    for (int st = 0; st < 3; st++) if (dsh(st) && !(st == 1 && k.raster_discard)) { sets[dsh(st)->set] = stage_set(dst_of(st), dsh(st), dofs[dsh(st)->set]); k.um[st] = g_num_um; k.im[st] = g_num_im;
        k.lm[st] = vk_lodshift_mask(dsh(st), g_lod_shift[dst_of(st)]); }   // (its stage set's tex_desc_for noted each binding's shift)
    uint64_t tdesc = tick() - t1d; g_vkst.ns_desc += tdesc; if (g_prof_on) g_prof_ns[5] += tdesc;
    if (g_tracing) {
        VKLOG("  pipeline: topology %u restart %u discard %u cull %u front %u mode %u clamp %u bias %u; depth test %u write %u func %u stencil %u; attributes %u bindings %u; colour %u (format %u) samples %u a2c %u; blend enable %#x logic %u/%u; fragment depth variant %u (depth written by a shader at present %llu, now %llu)",
              k.topo, k.restart, k.raster_discard, k.cull, k.front, k.mode, k.depth_clamp, k.bias_en, k.dt, k.dw, k.df, k.st, k.nattr, k.nbind, k.ncol, k.cfmt[0], k.samples, k.a2c, k.blend_en, k.logic_en, k.logic_op,
              k.fdz, has_depth && P_att[8].im ? (unsigned long long)P_att[8].im->fdz_frame : 0ull, (unsigned long long)g_tex_frame + 1);
        if (k.st) VKLOG("  stencil: func %u/%u fail %u/%u zfail %u/%u pass %u/%u; compare mask %#x/%#x write mask %#x/%#x reference %#x/%#x", k.sf[0], k.sf[1], k.sfail[0], k.sfail[1], k.szfail[0], k.szfail[1], k.szpass[0], k.szpass[1],
                        (unsigned)S.smask[0], (unsigned)S.smask[1], (unsigned)S.swmask[0], (unsigned)S.swmask[1], (unsigned)S.sref[0], (unsigned)S.sref[1]);
        for (uint32_t i = 0; i < k.ncol; i++) VKLOG("  blend %u: src %#x dst %#x eq %u, alpha src %#x dst %#x eq %u, write mask %#x", i, k.bl[i].src, k.bl[i].dst, k.bl[i].eq, k.bl[i].srca, k.bl[i].dsta, k.bl[i].eqa, k.wmask[i]);
        for (uint32_t i = 0; i < k.nattr; i++) VKLOG("  attribute location %u binding %u format %u offset %u", k.attr[i].loc, k.attr[i].binding, k.attr[i].fmt, k.attr[i].off);
        for (uint32_t i = 0; i < k.nbind; i++) VKLOG("  binding %u stride %u rate %u divisor %u", k.bind[i].binding, k.bind[i].stride, k.bind[i].rate, k.bind[i].divisor);
    }
    crumb_draw(k.key[0], k.key[1], k.key[2]);
    VkPipeline p = graphics_pipeline(&k, layout);
    uint64_t t1 = tick(); g_vkst.ns_pipe += t1 - t0 - tdesc;
    if (!p) { g_vkst.skip_pipe++; return 0; }
    if (g_ubolog_key && g_ubolog_n > 0 && S.sh[0] && S.sh[0]->key == g_ubolog_key) {   // DM_VK_UBO_LOG: the pipeline the draw got, and its key's hash
        uint64_t kh = fnv(1469598103934665603ull, &k, sizeof k); VKLOG("ubo: present %llu pipeline %p key %016llx um %llx/%llx im %llx/%llx", (unsigned long long)g_tex_frame, (void *)p, (unsigned long long)kh,
              (unsigned long long)k.um[0], (unsigned long long)k.um[1], (unsigned long long)k.im[0], (unsigned long long)k.im[1]); }
    g_vkst.draws++; if (++g_pass_draws > g_vkst.max_pass_draws) g_vkst.max_pass_draws = g_pass_draws;
    if (p != S.bound_pipe) { uint64_t tb = g_prof_on ? tick() : 0; vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, p); S.bound_pipe = p; if (g_prof_on) g_prof_ns[4] += tick() - tb; }
    set_dynamic(cb);
    if (k.atest) { uint32_t pc[2] = {S.color.alpha_func, 0}; memcpy(&pc[1], &S.alpha_ref, 4); vkCmdPushConstants(cb, layout, VK_SHADER_STAGE_FRAGMENT_BIT, VK_ALPHA_PUSH_OFF, VK_ALPHA_PUSH_SIZE, pc); }   // (k.atest)
    for (int st = 0; st < 3; st++) if (k.lm[st]) {   // each stage's LOD shifts, 4 bits a binding (gfx_vk_shader.c vk_lodshift_word), at its stage's word (vk_lodshift_off)
        static const VkShaderStageFlags fl[2][3] = {{VK_SHADER_STAGE_VERTEX_BIT, VK_SHADER_STAGE_FRAGMENT_BIT, VK_SHADER_STAGE_GEOMETRY_BIT}, {VK_SHADER_STAGE_MESH_BIT_EXT, VK_SHADER_STAGE_FRAGMENT_BIT, VK_SHADER_STAGE_TASK_BIT_EXT}};
        uint32_t wv = vk_lodshift_word(dsh(st), k.lm[st], g_lod_shift[dst_of(st)]);
        vkCmdPushConstants(cb, layout, fl[k.mesh != 0][st], vk_lodshift_off(dsh(st)->stage), 4, &wv); }
    if (var == 1) { vkCmdSetDepthBias(cb, 0, 0, 0); memset(&g_bc.po, 0xff, sizeof g_bc.po); }
    uint32_t at[6] = {0};   // where each set's dynamic offsets start in dall
    for (int i = 0; i < nsets; i++) { at[i] = ndofs; for (int st = 0; st < 3; st++)   // the dynamic offsets: by set, then binding
        if (dsh(st) && !(st == 1 && k.raster_discard) && dsh(st)->set == i) { memcpy(dall + ndofs, dofs[i], sizeof(uint32_t) * (size_t)dsh(st)->ndyn); ndofs += (uint32_t)dsh(st)->ndyn; } }
    at[nsets] = ndofs;
    for (int i = 0; i < nsets; i++) if (!sets[i]) sets[i] = vk_alloc_set(dsl[i]);   // an absent stage's set: from the empty layout vk_pipeline_layout filled in
    // DM_VK_SETSPLIT=1 (live): under the same layout, only the run of sets whose handle or dynamic offsets changed is bound again. Adreno's vkCmdBindDescriptorSets
    // took ~22% of the replay thread on Unicorn Overlord's battle map (one load in it), and a draw that moves only its vertex stage's offsets sent the fragment set too
    if (!(dbg_draw() & 4) && g_setsplit && !bc_fresh() && g_bc.lay == layout && g_bc.nsets == (uint32_t)nsets && g_bc.ndofs == ndofs) {
        int lo = -1, hi = -1;
        for (int i = 0; i < nsets; i++) if (g_bc.sets[i] != sets[i] || memcmp(g_bc.dofs + at[i], dall + at[i], sizeof *dall * (at[i + 1] - at[i]))) { if (lo < 0) lo = i; hi = i; }
        if (lo >= 0) { vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, layout, (uint32_t)lo, (uint32_t)(hi - lo + 1), sets + lo, at[hi + 1] - at[lo], dall + at[lo]);
            memcpy(g_bc.sets + lo, sets + lo, sizeof *sets * (size_t)(hi - lo + 1)); memcpy(g_bc.dofs + at[lo], dall + at[lo], sizeof *dall * (at[hi + 1] - at[lo])); } }
    else if (!(dbg_draw() & 4)) { int fresh = bc_fresh();
        if (fresh || g_bc.lay != layout || g_bc.nsets != (uint32_t)nsets || g_bc.ndofs != ndofs || memcmp(g_bc.sets, sets, sizeof *sets * (size_t)nsets) || memcmp(g_bc.dofs, dall, sizeof *dall * ndofs)) {
            vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, layout, 0, (uint32_t)nsets, sets, ndofs, dall);
            g_bc.lay = layout; g_bc.nsets = (uint32_t)nsets; g_bc.ndofs = ndofs; memcpy(g_bc.sets, sets, sizeof *sets * (size_t)nsets); memcpy(g_bc.dofs, dall, sizeof *dall * ndofs); } }
    // vertex buffers
    for (uint32_t i = 0; i < k.nbind; i++) {
        uint32_t b = k.bind[i].binding; VkBuffer buf; VkDeviceSize off;
        if (b == 16) { buf = vk_zero_buffer(); off = 0; }
        else if (!buf_at_n(S.vbuf[b].address, S.vbuf[b].size, &buf, &off)) { buf = vk_zero_buffer(); off = 0; VKRED_ONCE(51, "vertex buffer outside the memory pools"); }
        if (g_var_now == 5) { buf = vk_zero_buffer(); off = 0; }
        if (!(dbg_draw() & 2) && b < 17) { int fresh = bc_fresh(); if (fresh || g_bc.vb[b] != buf || g_bc.vo[b] != off) { vkCmdBindVertexBuffers(cb, b, 1, &buf, &off); g_bc.vb[b] = buf; g_bc.vo[b] = off; } }
    }
    (void)indexed;
    if (vk_prof_on()) { if (k.atest) g_pu.alpha = pu_add(g_pu.alpha, 1); if (k.fdz) g_pu.fdz = pu_add(g_pu.fdz, 1);   // (DM_VK_GPUPROF: the pass's draws by kind)
        if (!k.raster_discard && S.sh[1] && S.sh[1]->depth_repl && k.dt && k.dw) g_pu.fdepth = pu_add(g_pu.fdepth, 1); }
    return 1;
}
// quads without an index buffer: one buffer of 0 1 2 0 2 3, 4 5 6 4 6 7, ... for up to QUAD_MAX quads (more are drawn as the first QUAD_MAX)
#define QUAD_MAX 65536u
static VkBuffer quad_index_buffer(uint32_t *nq) {
    static VkBuffer b; static pthread_mutex_t mu = PTHREAD_MUTEX_INITIALIZER;
    if (*nq > QUAD_MAX) { VKRED_ONCE(58, "quad draw of %u quads: the first %u drawn", *nq, QUAD_MAX); *nq = QUAD_MAX; }
    pthread_mutex_lock(&mu);
    if (!b) {
        VkBufferCreateInfo bi = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, NULL, 0, QUAD_MAX * 6 * 4, VK_BUFFER_USAGE_INDEX_BUFFER_BIT, VK_SHARING_MODE_EXCLUSIVE};
        VKCHK(vkCreateBuffer(V.dev, &bi, NULL, &b)); VkMemoryRequirements req; vkGetBufferMemoryRequirements(V.dev, b, &req);
        VkMemoryAllocateInfo ai = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, NULL, req.size, vk_mem_type(req.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, 0)};
        VkDeviceMemory m; VKCHK(vkAllocateMemory(V.dev, &ai, NULL, &m)); VKCHK(vkBindBufferMemory(V.dev, b, m, 0));
        uint32_t *p; VKCHK(vkMapMemory(V.dev, m, 0, VK_WHOLE_SIZE, 0, (void **)&p));
        static const uint32_t pat[6] = {0, 1, 2, 0, 2, 3};
        for (uint32_t q = 0; q < QUAD_MAX; q++) for (int k = 0; k < 6; k++) p[q * 6 + k] = q * 4 + pat[k];
    }
    pthread_mutex_unlock(&mu);
    return b;
}
// quads: a triangle list of indices made on the CPU (from the game's index buffer for indexed draws)
static int quad_indices(uint32_t count, const uint8_t *src, int itype, int32_t first, VkBuffer *buf, VkDeviceSize *off) {
    uint32_t nq = count / 4; if (!nq) return 0;
    uint32_t *d = vk_scratch((VkDeviceSize)nq * 6 * 4, 4, buf, off);
    for (uint32_t q = 0; q < nq; q++) for (int k = 0; k < 6; k++) {
        static const int pat[6] = {0, 1, 2, 0, 2, 3}; uint32_t i = q * 4 + (uint32_t)pat[k];
        d[q * 6 + k] = !src ? (uint32_t)first + i : itype == 0 ? src[i] : itype == 1 ? ((const uint16_t *)src)[i] : ((const uint32_t *)src)[i];
    }
    return (int)(nq * 6);
}
// ---------------- transform feedback ----------------
// NVN's capture state (OP_TF_BIND / OP_TF_STATE). Each capturing draw (feedback on, a vertex shader translated with Xfb: shader_vk.py xfb_layout) is a Vulkan capture of its own:
// the buffers bound at the bytes written so far, Begin, the draw, End. So no pipeline is bound and no pass ends while a capture is active, and no counter buffer is read back:
// the host counts the bytes, as nvn_mtl.m tf_draw does (vertices the primitives emit × the buffer's stride). Xenoblade Chronicles 3 steps its GPU particles this way (8 shaders,
// up to 9 outputs interleaved in one buffer); before, those draws wrote nothing and the particles kept stale or zero state
static uint32_t xfb_verts(int prim, uint32_t n) {   // vertices a capture writes for n vertices of this primitive (strips and fans are written as lists)
    switch (topo(prim)) {
    case VK_PRIMITIVE_TOPOLOGY_LINE_LIST: return n & ~1u;
    case VK_PRIMITIVE_TOPOLOGY_LINE_STRIP: return n >= 2 ? (n - 1) * 2 : 0;
    case VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST: return n / 3 * 3;
    case VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP: case VK_PRIMITIVE_TOPOLOGY_TRIANGLE_FAN: return n >= 3 ? (n - 2) * 3 : 0;
    default: return n;
    }
}
static int xfb_begin(VkCommandBuffer cb) {   // inside the pass, after the pipeline is bound; 1 when a capture began
    vkshader_t *vs = S.sh[0];
    if (!g_tf_on || !V.has_xfb || !vs || !vs->xfb || !g_xfb_on) return 0;
    VkBuffer b[4]; VkDeviceSize o[4], n[4];
    for (int k = 0; k < 4; k++) {
        uint64_t xa = g_tf[k].addr + g_tf_off[k]; vkpool_t *p = g_tf[k].addr ? vk_pool_gpu_at(&xa, 4) : NULL;   // (a virtual pool's range translated)
        if (vs->xfb_stride[k] && p && g_tf_off[k] < g_tf[k].size && vk_buf_at(xa, 4, &b[k], &o[k])) {
            n[k] = g_tf[k].size - g_tf_off[k]; if (o[k] + n[k] > p->hi - p->lo) n[k] = p->hi - p->lo - o[k];
        } else { vk_scratch(64, 64, &b[k], &o[k]); n[k] = 64; if (vs->xfb_stride[k]) VKRED_ONCE(83, "transform feedback buffer %d unbound, full or outside the memory pools: its capture is dropped", k); }
    }
    vkCmdBindTransformFeedbackBuffersEXT(cb, 0, 4, b, o, n);
    vkCmdBeginTransformFeedbackEXT(cb, 0, 0, NULL, NULL);
    return 1;
}
static void xfb_end(VkCommandBuffer cb, int prim, uint32_t count, uint32_t inst) {   // after the draw: the capture ends and the bytes it wrote are counted
    vkCmdEndTransformFeedbackEXT(cb, 0, 0, NULL, NULL);
    uint32_t v = xfb_verts(prim, count) * (inst ? inst : 1); g_tf_in_pass = 1; g_tf_verts += v;
    for (int k = 0; k < 4; k++) {
        uint64_t m = (uint64_t)v * S.sh[0]->xfb_stride[k]; if (!m || !g_tf[k].addr) continue;
        if (g_tf_off[k] + m > g_tf[k].size) m = g_tf[k].size > g_tf_off[k] ? g_tf[k].size - g_tf_off[k] : 0;
        void vk_gpu_wrote(uint64_t a, uint64_t n); if (m) vk_gpu_wrote(g_tf[k].addr + g_tf_off[k], m);
        g_tf_off[k] += m;
    }
}
// DrawTransformFeedback draws the vertices the captures ended into one control buffer wrote: counted since Begin (g_tf_verts) and kept by control buffer address at End / Pause
// (a table by address: Xenoblade Chronicles 3 gives every particle emitter a control buffer of its own, hundreds of them, 32 bytes apart)
enum { TFC = 4096 }; static struct { uint64_t ctl; uint32_t verts; } g_tf_ctl[TFC];
static int tf_ctl_slot(uint64_t ctl, int add) {   // the entry of a control buffer; with add, a new one (the oldest in its probe range is replaced when they are all taken)
    unsigned h = (unsigned)((ctl >> 5) * 2654435761u) % TFC;
    for (int k = 0; k < 16; k++) { unsigned i = (h + (unsigned)k) % TFC; if (g_tf_ctl[i].ctl == ctl) return (int)i; if (!g_tf_ctl[i].ctl) return add ? (int)i : -1; }
    return add ? (int)h : -1;
}
static void tf_state(const p_tf_state *t) {
    switch (t->what & 3) {
    case 0: memset(g_tf_off, 0, sizeof g_tf_off); g_tf_verts = 0; g_tf_on = 1; break;   // Begin: written from the start of each buffer
    case 3: g_tf_on = 1; break;                                          // Resume: on from the bytes written
    default: g_tf_on = 0;                                                // End / Pause: what was captured is read by later draws (as vertices), after the barrier the next pass begins with
        if (g_tf_in_pass) { vk_end_pass(); g_tf_in_pass = 0; }
        int i = tf_ctl_slot(t->ctl, 1); g_tf_ctl[i].ctl = t->ctl; g_tf_ctl[i].verts = g_tf_verts;
    }
}
static void draw(const p_draw *d);
static void draw_tf(const p_draw_tf *t) {
    int i = tf_ctl_slot(t->ctl, 0);
    if (i < 0) { VKRED_ONCE(86, "DrawTransformFeedback with a control buffer no capture ended into (%#llx): nothing drawn", (unsigned long long)t->ctl); return; }
    p_draw d = {t->prim, 0, (int32_t)g_tf_ctl[i].verts, 1, 0, 0, 0, 0, 0};
    if (d.count) draw(&d);
}
static void trace_draw(const char *kind, int prim, uint32_t count, uint32_t inst, int itype) {
    VKLOG("draw: %s prim %d count %u instances %u itype %d; shaders %016llx %016llx %016llx; targets %d (formats %d %d %d %d) depth format %d samples %u, pass %ux%u; viewport %.0f %.0f %.0f %.0f; streams:",
          kind, prim, count, inst, itype, (unsigned long long)(S.sh[0] ? S.sh[0]->key : 0), (unsigned long long)(S.sh[1] ? S.sh[1]->key : 0), (unsigned long long)(S.sh[2] ? S.sh[2]->key : 0),
          S.ncol, S.cfmt[0], S.cfmt[1], S.cfmt[2], S.cfmt[3], S.dfmt, S.samples, S.pass_w, S.pass_h, S.vp[0][0], S.vp[0][1], S.vp[0][2], S.vp[0][3]);
    for (int i = 0; i < S.rt_n; i++) if (S.rt_bad[i]) VKLOG("  colour target %d: not a texture object (first word %#x): left out", i, S.rt_bad[i]);
    for (int i = 0; i < S.rt_n; i++) if (S.rt_has_col[i]) { vkimg_t *im = vk_tex(&S.rt_col[i]); const tex_t *t = &S.rt_col[i];
        VKLOG("  colour target %d: serial %llu → image %p: %ux%ux%u layers %u levels %u samples %u fmt %d vtype %d; nvn target %d format %#x flags %#x stride %u size %llu pool %p+%#llx",
              i, (unsigned long long)t->serial, (void *)im, im ? im->w : 0, im ? im->h : 0, im ? im->d : 0, im ? im->layers : 0, im ? im->levels : 0, im ? im->samples : 0, im ? im->fmt : 0, im ? im->vtype : 0,
              t->d.target, t->d.format, t->d.flags, t->d.stride, (unsigned long long)t->L.size, (const void *)t->pool, (unsigned long long)t->off);
        if (S.rt_has_cview[i]) { const view_t *v = &S.rt_cview[i]; VKLOG("  its view: levels %d (%u+%u) layers %d (%u+%u) target %d (%u) format %d (%#x)",
              v->set_levels, v->base_level, v->nlevels, v->set_layers, v->min_layer, v->nlayers, v->set_target, v->target, v->set_format, v->format); } }
    if (S.rt_has_dep) { vkimg_t *im = vk_tex(&S.rt_dep); const tex_t *t = &S.rt_dep;
        VKLOG("  depth target: serial %llu → image %p: %ux%u fmt %d; nvn format %#x flags %#x size %llu pool %p+%#llx", (unsigned long long)t->serial, (void *)im, im ? im->w : 0, im ? im->h : 0,
              im ? im->fmt : 0, t->d.format, t->d.flags, (unsigned long long)t->L.size, (const void *)t->pool, (unsigned long long)t->off); }
    VKLOG("  scissor set %d: %d %d %d %d; depth range %.3f %.3f; viewport count first %.1f %.1f %.1f %.1f", S.sc_set, S.sc.x, S.sc.y, S.sc.w, S.sc.h, S.dnear, S.dfar, S.vp[0][0], S.vp[0][1], S.vp[0][2], S.vp[0][3]);
    vkshader_t *vsh = S.sh[0]; uint32_t in_mask = vsh ? vsh->local[0] : 0;
    for (int l = 0; l < 32; l++) if (in_mask >> l & 1) { uint32_t a = S.vattr[l]; VKLOG("  location %d: attrib state %#x (stream %u) stride %d divisor %d buffer %#llx +%llu", l, a, (a >> 16) & 0x1f,
        S.vstream[(a >> 16) & 0xf].stride, S.vstream[(a >> 16) & 0xf].divisor, (unsigned long long)S.vbuf[(a >> 16) & 0xf].address, (unsigned long long)S.vbuf[(a >> 16) & 0xf].size); }
}
// Vertex fetches past a stream's bound size. NVN binds each stream with a size (BindVertexBuffers), and Maxwell's fetch reads zeros past it; Vulkan's binding carries no
// size here (robustBufferAccess is off), so Adreno read on into whatever followed the stream in the pool. agl draws full-screen passes past their streams: Super Mario
// Odyssey's alNoiseShader 9 vertices from a 36-byte stream (one triangle) and an alProgramShader pass 18 vertices from 72 bytes, every frame in Cap Kingdom; the extra
// vertices made triangles of the pool's next bytes (on Metal the hard-edged mist slabs of the Cascade Kingdom, seedds/nxnative issue #558; issue #42 here, seams on
// volumetrics). For a direct draw, a stream whose last
// element read (the last vertex, the last instance of a per-instance stream, element 0 of a stride-0 stream) reaches past its bound size is bound as a copy of its bound
// bytes followed by zeros. Indexed and indirect draws are not bounded (their last element is not known without reading the indices or the arguments).
// DM_VK_VBOUND=0 (live): bound as before (A/B)
static uint32_t vfmt_bytes(VkFormat f) {   // a vertex attribute's bytes (the formats vertex_format returns)
    if (f >= VK_FORMAT_R8_UNORM && f <= VK_FORMAT_R8_SRGB) return 1;
    if (f >= VK_FORMAT_R8G8_UNORM && f <= VK_FORMAT_R8G8_SRGB) return 2;
    if (f >= VK_FORMAT_R8G8B8_UNORM && f <= VK_FORMAT_B8G8R8_SRGB) return 3;
    if (f >= VK_FORMAT_R8G8B8A8_UNORM && f <= VK_FORMAT_A2B10G10R10_SINT_PACK32) return 4;
    if (f >= VK_FORMAT_R16_UNORM && f <= VK_FORMAT_R16_SFLOAT) return 2;
    if (f >= VK_FORMAT_R16G16_UNORM && f <= VK_FORMAT_R16G16_SFLOAT) return 4;
    if (f >= VK_FORMAT_R16G16B16_UNORM && f <= VK_FORMAT_R16G16B16_SFLOAT) return 6;
    if (f >= VK_FORMAT_R16G16B16A16_UNORM && f <= VK_FORMAT_R16G16B16A16_SFLOAT) return 8;
    if (f >= VK_FORMAT_R32_UINT && f <= VK_FORMAT_R32_SFLOAT) return 4;
    if (f >= VK_FORMAT_R32G32_UINT && f <= VK_FORMAT_R32G32_SFLOAT) return 8;
    if (f >= VK_FORMAT_R32G32B32_UINT && f <= VK_FORMAT_R32G32B32_SFLOAT) return 12;
    if (f == VK_FORMAT_B10G11R11_UFLOAT_PACK32 || f == VK_FORMAT_E5B9G9R9_UFLOAT_PACK32) return 4;
    return 16;
}
static uint64_t g_vbound_n;
static void vbuf_bound(VkCommandBuffer cb, uint32_t first, uint32_t nvert, uint32_t binst, uint32_t inst) {
    vkshader_t *vsh = S.sh[0]; if (!g_vbound || !vsh || !nvert || (dbg_draw() & 2) || g_var_now == 5) return;
    uint32_t in_mask = vsh->local[0], int_mask = vsh->local[1] | vsh->local[2], sint_mask = vsh->local[1]; uint64_t need[16] = {0};
    for (int loc = 0; loc < 32; loc++) {
        if (!(in_mask >> loc & 1)) continue;
        va_st a; memcpy(&a, &S.vattr[loc], 4); if (!a.fmt || a.stream >= 16 || !S.vbuf[a.stream].address) continue;
        VkFormat f = vertex_format(a.fmt, int_mask >> loc & 1, sint_mask >> loc & 1); if (f == VK_FORMAT_UNDEFINED) continue;
        uint32_t s = a.stream, st = S.vstream[s].stride, dv = S.vstream[s].divisor;
        uint64_t last = !st ? 0 : dv ? binst + (inst - 1) / dv : (uint64_t)first + nvert - 1, e = last * st + a.offset + vfmt_bytes(f);
        if (e > need[s]) need[s] = e;
    }
    for (uint32_t s = 0; s < 16; s++) {
        uint64_t n = need[s], sz = S.vbuf[s].size; if (n <= sz || !sz) continue;
        if (n > (16u << 20)) { VKRED_ONCE(87, "vertex stream %u: a draw reads %llu bytes of a %llu-byte binding: not bounded", s, (unsigned long long)n, (unsigned long long)sz); continue; }
        VkBuffer b; VkDeviceSize o; uint8_t *p = vk_scratch(n, 16, &b, &o);
        vk_ubd_read(S.vbuf[s].address, sz, p); memset(p + sz, 0, (size_t)(n - sz));
        vkCmdBindVertexBuffers(cb, s, 1, &b, &o); g_bc.vb[s] = b; g_bc.vo[s] = o;
        if (g_vbound_n++ < 8) VKLOG("vertex stream %u: draw of %u vertices from %u (shader %016llx) reads %llu bytes past its bound %llu: bound as a copy padded with zeros",
                                    s, nvert, first, (unsigned long long)vsh->key, (unsigned long long)(n - sz), (unsigned long long)sz);
    }
}
static void ubolog_draw(const char *what, int32_t count, int32_t inst, int32_t binst, int32_t bvert, uint64_t ind, uint64_t mdi_cnt, int32_t mdi_i);
// DM_VK_X_INDLIST=1 (live, debugging): each vertex shader drawn with GPU-written draw arguments (an indirect draw, a multi-draw and how it was drawn) logged once, up to 96:
// the candidates for DM_VK_X_SKIPVS when what flickers is GPU-culled
static void indlist_note(const char *how, uint32_t first, uint32_t n) {
    static uint64_t seen[96]; static int nseen; static unsigned gen; if (!g_x_indlist || !S.sh[0] && !g_mesh.on) return; if (gen != g_x_indlist_gen) { gen = g_x_indlist_gen; nseen = 0; }
    uint64_t k = dsh(0) ? dsh(0)->key : 0; for (int i = 0; i < nseen; i++) if (seen[i] == k) return; if (nseen == 96) return; seen[nseen++] = k;
    VKLOG("indlist: vertex shader %016llx %s (first record %u, records %u), pass %d, depth target %d", (unsigned long long)k, how, first, n, g_sub_pass - 1, S.rt_has_dep);
}
static void mesh_draw(const p_draw *d); static void multi_draw(int prim, int itype, uint64_t ibuf, uint64_t ind, uint64_t cnt, uint32_t first, uint32_t n, uint32_t stride);
static void draw(const p_draw *d) {
    if (d->prim == DM_PRIM_MESH_TASKS) { mesh_draw(d); return; }   // nvnCommandBufferDrawMeshTasks
    if (d->mdi_cnt) { multi_draw(d->prim, -1, 0, d->ind, d->mdi_cnt, (uint32_t)d->mdi_i, 1, 0); return; }   // a multi-draw's record mdi_run did not take with its call
    int prim = d->prim;
    // before prepare_draw, whose driver segment carries it (c0 + 0x644): set after it, each draw's segment had the draw before's. An indirect draw's is in GPU memory: 0
    g_draw_binst = d->ind ? 0 : (uint32_t)d->base_instance;
    if (!prepare_draw(prim == 7 ? 4 : prim, prim == 7)) return;
    VkCommandBuffer cb = vk_cmd();
    int g_ok = draw_allowed(); if (g_ok == 2 || (g_tracing && g_x_tracepass >= 0)) trace_draw("draw", d->prim, (uint32_t)d->count, (uint32_t)d->instances, -1);
    if (g_tracing && g_x_tracepass >= 0 && !d->ind && S.vbuf[0].address) {   // the first vertices stream 0 gives the draw (with the deferred updates), every word as a float
        uint32_t st = S.vstream[0].stride ? S.vstream[0].stride : 16, nw = st / 4 < 16 ? st / 4 : 16;
        for (uint32_t v = 0; v < (uint32_t)d->count && v < 4; v++) { uint64_t o = (uint64_t)((uint32_t)d->first + v) * st; if (o + nw * 4 > S.vbuf[0].size) break;
            float w[16]; vk_ubd_read(S.vbuf[0].address + o, nw * 4, (uint8_t *)w); char t[400]; int k = 0; for (uint32_t i = 0; i < nw; i++) k += snprintf(t + k, sizeof t - k, " %g", w[i]);
            VKLOG("  vertex %u of stream 0 (+%llu):%s", (uint32_t)d->first + v, (unsigned long long)o, t); } }
    if (g_var_now == 10 && prim == 7) g_ok = 0; if (g_var_now == 11 && prim != 7) g_ok = 0;   // bisection: quads from the found draw on, or the rest
    { static int nq, nn; if (g_var_now) { if (prim == 7) nq++; else nn++; if ((nq + nn) % 50 == 1) VKLOG("draws from the found one on: %d quads, %d others", nq, nn); } }
    uint32_t inst = d->instances ? (uint32_t)d->instances : 1;
    { static int n; if (n < 12 && (d->ind || (uint32_t)d->count > (1u << 20) || inst > (1u << 16))) { n++;
        VKLOG("unusual draw: count %u instances %u indirect %#llx prim %d first %d", (unsigned)d->count, inst, (unsigned long long)d->ind, d->prim, d->first); } }
    if (d->ind) {
        VkBuffer b; VkDeviceSize o;
        if (g_tf_on && S.sh[0] && S.sh[0]->xfb) VKRED_ONCE(84, "indirect draw under transform feedback: not captured");
        if (vk_buf_at(d->ind, 16, &b, &o)) if (!(dbg_draw() & 1) && g_ok) vkCmdDrawIndirect(cb, b, o, 1, 16); else VKRED_ONCE(53, "indirect draw arguments outside the memory pools");
        return;
    }
    if (prim == 7) {   // quads: the triangle list indices of one static buffer (built once), the first vertex as the vertex offset
        uint32_t nq = (uint32_t)d->count / 4; if (!nq) return;
        vbuf_bound(cb, (uint32_t)d->first, nq * 4, (uint32_t)d->base_instance, inst);
        if (g_tf_on && S.sh[0] && S.sh[0]->xfb) VKRED_ONCE(85, "quads under transform feedback: not captured");
        VkBuffer qb = quad_index_buffer(&nq);
        bc_index(cb, qb, 0, VK_INDEX_TYPE_UINT32); if (!(dbg_draw() & 1) && g_ok) vkCmdDrawIndexed(cb, nq * 6, inst, 0, g_var_now == 4 ? 0 : d->first, (uint32_t)d->base_instance); return; }
    if (prim == 2) VKRED_ONCE(54, "line loops drawn as line strips");
    vbuf_bound(cb, (uint32_t)d->first, (uint32_t)d->count, (uint32_t)d->base_instance, inst);
    if (!(dbg_draw() & 1) && g_ok) { int xf = xfb_begin(cb); vkCmdDraw(cb, (uint32_t)d->count, inst, (uint32_t)d->first, (uint32_t)d->base_instance); if (xf) xfb_end(cb, prim, (uint32_t)d->count, inst); }
}
static void draw_idx(const p_draw_idx *d) {
    if (d->mdi_cnt) { multi_draw(d->prim, d->itype, d->ibuf, d->ind, d->mdi_cnt, (uint32_t)d->mdi_i, 1, 0); return; }   // (as in draw)
    int prim = d->prim;
    g_draw_binst = d->ind ? 0 : (uint32_t)d->base_instance;   // before prepare_draw (see draw)
    if (!prepare_draw(prim == 7 ? 4 : prim, 1)) return;
    VkCommandBuffer cb = vk_cmd();
    int g_ok = draw_allowed(); if (g_ok == 2 || (g_tracing && g_x_tracepass >= 0)) trace_draw("indexed draw", d->prim, (uint32_t)d->count, (uint32_t)d->instances, d->itype);
    if (g_var_now == 11) g_ok = 0;
    VkIndexType it = d->itype == 0 ? VK_INDEX_TYPE_UINT8_EXT : d->itype == 1 ? VK_INDEX_TYPE_UINT16 : VK_INDEX_TYPE_UINT32;
    uint32_t isz = d->itype == 0 ? 1 : d->itype == 1 ? 2 : 4;
    VkBuffer ib; VkDeviceSize io;
    if (!buf_at_n(d->ibuf, d->ind ? 1 : (uint64_t)(d->count > 0 ? d->count : 1) * isz, &ib, &io)) { VKRED_ONCE(56, "index buffer outside the memory pools"); return; }
    if (g_ubolog_key && S.sh[0] && S.sh[0]->key == g_ubolog_key && g_ubolog_n > 0) ubolog_draw("indexed draw", d->count, d->instances, d->base_instance, d->base_vertex, d->ind, 0, 0);
    if (d->ind) indlist_note("indexed indirect draw", 0, 1);
    if (g_ubolog_key && S.sh[0] && S.sh[0]->key == g_ubolog_key && g_ubolog_n > 0 && vk_cpu_ptr(d->ibuf, (uint64_t)(d->count > 0 ? d->count : 1) * isz) == (void *)(uintptr_t)d->ibuf &&
        (!S.vbuf[0].address || vk_cpu_ptr(S.vbuf[0].address, S.vbuf[0].size ? S.vbuf[0].size : 1) == (void *)(uintptr_t)S.vbuf[0].address)) {   // DM_VK_UBO_LOG: the index data the draw reads (CPU view at replay; not in a virtual pool's range)
        const uint8_t *ix = (const uint8_t *)(uintptr_t)d->ibuf; uint64_t h = 1469598103934665603ull; uint32_t mn = ~0u, mx = 0;
        for (uint32_t i = 0; i < (uint32_t)d->count; i++) { uint32_t v = isz == 1 ? ix[i] : isz == 2 ? ((const uint16_t *)ix)[i] : ((const uint32_t *)ix)[i]; h = (h ^ v) * 1099511628211ull; if (v < mn) mn = v; if (v > mx) mx = v; }
        VKLOG("ubo: present %llu index buffer %#llx count %d base vertex %d indirect %#llx hash %016llx range %u..%u", (unsigned long long)g_tex_frame, (unsigned long long)d->ibuf, d->count, d->base_vertex,
              (unsigned long long)d->ind, (unsigned long long)h, mn, mx);
        if (S.vbuf[0].address && mn != ~0u) { uint32_t st = S.vstream[0].stride ? S.vstream[0].stride : 12; uint64_t v0 = (uint64_t)(int64_t)d->base_vertex + mn;   // the first vertex this draw reads, every word
            if ((v0 + 1) * st <= S.vbuf[0].size) { const float *vf = (const float *)(uintptr_t)(S.vbuf[0].address + v0 * st); char t[512]; int k = 0;
                for (uint32_t i = 0; i < st / 4 && i < 24; i++) k += snprintf(t + k, sizeof t - k, " %g", vf[i]);
                VKLOG("ubo: present %llu first vertex read (%llu):%s", (unsigned long long)g_tex_frame, (unsigned long long)v0, t); } }
        static uint64_t dumped; if (dumped != h && S.vbuf[0].address && d->count <= 4096) { dumped = h; char t[1024]; int k = 0; uint32_t i0 = 0;   // once per index data: every index,
            for (uint32_t i = 0; i < (uint32_t)d->count; i++) { uint32_t v = isz == 1 ? ix[i] : isz == 2 ? ((const uint16_t *)ix)[i] : ((const uint32_t *)ix)[i];   // then every word of stream 0's vertices
                k += snprintf(t + k, sizeof t - k, " %u", v); if (k > 900 || i + 1 == (uint32_t)d->count) { VKLOG("ubo: indices from %u:%s", i0, t); k = 0; i0 = i + 1; } }
            uint32_t st = S.vstream[0].stride ? S.vstream[0].stride : 12, nv = (uint32_t)(S.vbuf[0].size / st);
            for (uint32_t i = 0; i < nv && i <= mx; i++) { const float *vf = (const float *)(uintptr_t)(S.vbuf[0].address + (uint64_t)i * st);   // every word of stream 0's vertex (up to 8)
                for (uint32_t w = 0; w < st / 4 && w < 8; w++) k += snprintf(t + k, sizeof t - k, "%s%g", w ? "," : " ", vf[w]);
                if (k > 800 || i + 1 == nv || i == mx) { VKLOG("ubo: vertices to %u:%s", i, t); k = 0; } } } }
    uint32_t inst = d->instances ? (uint32_t)d->instances : 1;
    { static int n; if (n < 12 && (d->itype == 0 || d->ind || (uint32_t)d->count > (1u << 20) || inst > (1u << 16))) { n++;
        VKLOG("unusual indexed draw: itype %d count %u instances %u indirect %#llx prim %d base vertex %d", d->itype, (unsigned)d->count, inst, (unsigned long long)d->ind, d->prim, d->base_vertex); } }
    if (d->ind) { VkBuffer b; VkDeviceSize o; bc_index(cb, ib, io, it); if (vk_buf_at(d->ind, 20, &b, &o)) if (!(dbg_draw() & 1) && g_ok) vkCmdDrawIndexedIndirect(cb, b, o, 1, 20); return; }
    if (prim == 7) { const uint8_t *ix = vk_cpu_ptr(d->ibuf, (uint64_t)(d->count > 0 ? d->count : 1) * isz); if (!ix) { VKRED_ONCE(164, "quad draw whose indices lie in a virtual pool's range the CPU cannot read in one piece: skipped"); return; }
        VkBuffer b; VkDeviceSize o; int n = quad_indices((uint32_t)d->count, ix, d->itype, 0, &b, &o); if (!n) return;
        bc_index(cb, b, o, VK_INDEX_TYPE_UINT32); if (!(dbg_draw() & 1) && g_ok) vkCmdDrawIndexed(cb, (uint32_t)n, inst, 0, d->base_vertex, (uint32_t)d->base_instance); return; }
    if (io % isz) { VKRED_ONCE(57, "index buffer offset not aligned to its index size"); return; }
    bc_index(cb, ib, io, it);
    if (!(dbg_draw() & 1) && g_ok) { int xf = xfb_begin(cb); vkCmdDrawIndexed(cb, (uint32_t)d->count, inst, 0, d->base_vertex, (uint32_t)d->base_instance); if (xf) xfb_end(cb, prim, (uint32_t)d->count, inst); }
}
// ---------------- indirect-count multi-draws ----------------
// NVN2's MultiDraw{Elements,Arrays}IndirectCount: up to max draws from records stride apart, as many as the u32 at cnt says, which the GPU writes (Monster Hunter Rise's GPU-driven scene:
// 1-49 draws a call, stride 20, records and count in its sparse virtual pool, written by compute culling). hle_nvn.c records a large call once (OP_MULTI_DRAW) and a call of up to
// MDI_INLINE draws as that many indirect draws carrying the count's address and their index (p_draw / p_draw_idx mdi_cnt, mdi_i), which nvn_mtl.m masks on the GPU (mdx_run: zeros past
// the count). Here a call is one vkCmdDraw(Indexed)IndirectCount (Vulkan 1.2's drawIndirectCount, or VK_KHR_draw_indirect_count), the records of a small one taken together (mdi_run).
// Without it (or for a record alone past index 0) the draws follow the count when the CPU knows it (no GPU write to it in flight), else all of them are made, with a [RED] once
int vk_gpu_pending(uint64_t a, uint64_t n);   // gfx_vk_core.c: a GPU write of [a, a+n) not known to be complete
// DM_VK_UBO_LOG's draw line: the draw's own numbers (count, instances, first instance, an indirect or multi-draw source and, where the CPU can see them, its arguments as
// they are at replay) and the vertex stage's storage buffers, each with a hash of its first 256 bytes: what changes from one frame to the next for a shader's draws
static void ubolog_draw(const char *what, int32_t count, int32_t inst, int32_t binst, int32_t bvert, uint64_t ind, uint64_t mdi_cnt, int32_t mdi_i) {
    char t[700]; int k = snprintf(t, sizeof t, "ubo: present %llu pass %d %s count %d instances %d first instance %d base vertex %d", (unsigned long long)g_tex_frame, g_sub_pass - 1, what, count, inst, binst, bvert);
    if (ind) { const uint32_t *a = vk_cpu_ptr(ind, 20); k += snprintf(t + k, sizeof t - (size_t)k, "; indirect %#llx", (unsigned long long)ind);
        if (a) k += snprintf(t + k, sizeof t - (size_t)k, " args %u %u %u %u %u", a[0], a[1], a[2], a[3], a[4]); }
    if (mdi_cnt) { const uint32_t *c = vk_cpu_ptr(mdi_cnt, 4); k += snprintf(t + k, sizeof t - (size_t)k, "; draw count at %#llx (record %d) = %d%s", (unsigned long long)mdi_cnt, mdi_i, c ? (int)*c : -1, vk_gpu_pending(mdi_cnt, 4) ? " (GPU write pending)" : ""); }
    for (int r = 0; r < 16 && k < (int)sizeof t - 60; r++) if (S.ssbo[0][r].addr) { uint64_t a = S.ssbo[0][r].addr, n = S.ssbo[0][r].size; const uint8_t *m = vk_cpu_ptr(a, n < 256 ? n : 256);
        k += snprintf(t + k, sizeof t - (size_t)k, "; s%d %#llx +%llu %016llx", r, (unsigned long long)a, (unsigned long long)n, m ? (unsigned long long)fnv(1469598103934665603ull, m, n < 256 ? n : 256) : 0ull); }
    VKLOG("%s", t);
}
static void multi_draw(int prim, int itype, uint64_t ibuf, uint64_t ind, uint64_t cnt, uint32_t first, uint32_t n, uint32_t stride) {
    int indexed = itype >= 0; uint32_t rs = indexed ? 20 : 16; if (!stride) stride = rs;
    if (!n) return;
    if (prim == 7) { VKRED_ONCE(130, "indirect-count multi-draw of quads: not drawn (their triangles' indices would have to be made from GPU-written records)"); return; }
    g_draw_binst = 0;   // (an indirect draw's base instance is in GPU memory: c0 + 0x644 reads 0, as in draw)
    if (!prepare_draw(prim, indexed)) return;
    indlist_note(V.has_dic && !first && !g_x_mdiall ? "multi-draw (one indirect-count draw)" : "multi-draw (records drawn one by one)", first, n);
    if (g_ubolog_key && S.sh[0] && S.sh[0]->key == g_ubolog_key && g_ubolog_n > 0) ubolog_draw(itype >= 0 ? "indexed multi-draw" : "multi-draw", (int32_t)n, -1, (int32_t)first, 0, ind + (uint64_t)first * (stride ? stride : (itype >= 0 ? 20u : 16u)), cnt, (int32_t)first);
    VkCommandBuffer cb = vk_cmd(); int ok = draw_allowed(); if (!ok || (dbg_draw() & 1)) return;
    if (g_tf_on && S.sh[0] && S.sh[0]->xfb) VKRED_ONCE(131, "indirect-count multi-draw under transform feedback: not captured");
    VkBuffer b, cbuf; VkDeviceSize o, co;
    if (!vk_buf_at(ind, (uint64_t)stride * (n - 1) + rs, &b, &o) || (o & 3) || (stride & 3) || stride < rs) {
        VKRED_ONCE(132, "indirect-count multi-draw records outside the memory pools or misaligned (%#llx, %u of stride %u): not drawn", (unsigned long long)ind, n, stride); return; }
    if (indexed) { VkIndexType it = itype == 0 ? VK_INDEX_TYPE_UINT8_EXT : itype == 1 ? VK_INDEX_TYPE_UINT16 : VK_INDEX_TYPE_UINT32; uint32_t isz = itype == 0 ? 1 : itype == 1 ? 2 : 4; VkBuffer ib; VkDeviceSize io;
        if (!vk_buf_at(ibuf, 1, &ib, &io) || io % isz) { VKRED_ONCE(133, "indirect-count multi-draw: index buffer outside the memory pools or misaligned"); return; }
        bc_index(cb, ib, io, it); }
    if (g_x_mdiall) { for (uint32_t i = 0; i < n; i++) { if (indexed) vkCmdDrawIndexedIndirect(cb, b, o + (VkDeviceSize)i * stride, 1, stride); else vkCmdDrawIndirect(cb, b, o + (VkDeviceSize)i * stride, 1, stride); } return; }   // DM_VK_X_MDI_ALL=1 (live, may draw wrong): every record, the GPU-written count left aside
    if (V.has_dic && !first && vk_buf_at(cnt, 4, &cbuf, &co) && !(co & 3)) {
        if (indexed) vkCmdDrawIndexedIndirectCount(cb, b, o, cbuf, co, n, stride); else vkCmdDrawIndirectCount(cb, b, o, cbuf, co, n, stride);
        { static int nl; if (!nl++) VKLOG("indirect-count multi-draw: first one, %u records of stride %u (%s), the count at %#llx", n, stride, indexed ? "indexed" : "arrays", (unsigned long long)cnt); }
        return; }
    uint32_t c = first + n;
    if (vk_cpu_ptr(cnt, 4) && !vk_gpu_pending(cnt, 4)) vk_ubd_read(cnt, 4, (uint8_t *)&c);   // written by the CPU, or by GPU work already done
    else VKRED_ONCE(134, "indirect-count multi-draw%s: its count is written by GPU work in flight, every record drawn", V.has_dic ? " (a record alone)" : " without drawIndirectCount");
    uint32_t m = c > first ? c - first : 0; if (m > n) m = n;
    for (uint32_t i = 0; i < m; i++) { if (indexed) vkCmdDrawIndexedIndirect(cb, b, o + (VkDeviceSize)i * stride, 1, stride); else vkCmdDrawIndirect(cb, b, o + (VkDeviceSize)i * stride, 1, stride); }
}
// A small call's records (hle_nvn.c: draws mdi_i 0, 1, ... with one count address and index buffer, their records stride apart, recorded one after the other; a command memory segment's
// JUMP may come between): drawn as one (multi_draw). Returns the record after the last one taken
extern uint32_t g_sub_ops[64];
static const uint8_t *mdi_run(const uint8_t *p) {
    op_hdr h; memcpy(&h, p, sizeof h); int idx = h.op == OP_DRAW_IDX; p_draw_idx x; p_draw d;
    if (idx) { p_draw_idx t; x = *draw_idx_rec(p + 8, h.size, &t); } else { p_draw t; d = *draw_rec(p + 8, h.size, &t); }
    uint64_t cnt = idx ? x.mdi_cnt : d.mdi_cnt, ind0 = idx ? x.ind : d.ind, last = ind0; int32_t i0 = idx ? x.mdi_i : d.mdi_i, prim = idx ? x.prim : d.prim;
    const uint8_t *q = p + h.size, *end = q; uint32_t n = 1; int64_t stride = 0;
    for (;;) {
        op_hdr g; memcpy(&g, q, sizeof g);
        if (g.op == OP_JUMP) { const uint8_t *t; memcpy(&t, q + 8, 8); q = t; continue; }
        if (g.op != h.op || n >= 65536) break;
        uint64_t rc, ri; int32_t rx, rp;
        if (idx) { p_draw_idx t; const p_draw_idx *r = draw_idx_rec(q + 8, g.size, &t); if (r->ibuf != x.ibuf || r->itype != x.itype) break; rc = r->mdi_cnt; ri = r->ind; rx = r->mdi_i; rp = r->prim; }
        else { p_draw t; const p_draw *r = draw_rec(q + 8, g.size, &t); rc = r->mdi_cnt; ri = r->ind; rx = r->mdi_i; rp = r->prim; }
        if (rc != cnt || rx != i0 + (int32_t)n || rp != prim) break;
        int64_t s = (int64_t)(ri - last); if (n == 1) stride = s; else if (s != stride) break;
        if (stride <= 0) break;
        last = ri; n++; q += g.size; end = q;
    }
    g_sub_ops[h.op & 63] += n - 1;
    multi_draw(prim, idx ? x.itype : -1, idx ? x.ibuf : 0, ind0, cnt, (uint32_t)i0, n, n > 1 ? (uint32_t)stride : 0);
    return end;
}

// ---------------- NVN2 task and mesh shaders ----------------
// nvnCommandBufferDrawMeshTasks (p_draw.prim DM_PRIM_MESH_TASKS: the first task or mesh workgroup, the count): a program's task and mesh shaders, in the tessellation stages' places
// (bind_program; nvn_mtl.m sh[3], sh[4]). Mario Kart World draws the bushes and grass tufts beside its tracks with them (7 task and 18 mesh shaders). With VK_EXT_mesh_shader they are a
// pipeline of the task (where the program has one), mesh and fragment stages (prepare_draw with g_mesh: the mesh shader in the vertex place, the task shader in the geometry place, their
// bindings stages 4 and 3, the storage buffer tables of c0 0x670 / 0x770), drawn with vkCmdDrawMeshTasksEXT(count): task workgroups, or mesh workgroups without a task shader. The draw's first
// workgroup reaches the shaders as nvn_geometry.inc mtk_encode binds it (an RK_DRV binding of bank DMSH_MESH_FIRST_SLOT, gfx_vk.h). Without the extension, or without the program's
// task / mesh SPIR-V (execution models TaskEXT / MeshEXT: what XLATE's translation gives), the draw is skipped with a [RED] once
static uint32_t spv_model(const vkshader_t *s) {   // the execution model of a shader's entry point (~0u: none found)
    if (!s || !s->spv) return ~0u;
    for (uint32_t i = 5; i < s->nspv;) { uint32_t n = s->spv[i] >> 16, op = s->spv[i] & 0xffff; if (!n) break; if (op == 15 && n >= 3) return s->spv[i + 1]; if (op == 54) break; i += n; }
    return ~0u;
}
enum { SPV_TASK_EXT = 5364, SPV_MESH_EXT = 5365 };
static uint32_t spv_group_op(uint32_t op) {   // the VK_SUBGROUP_FEATURE_* bit an instruction of subgroup scope needs (0: none of those)
    return op == 333 ? VK_SUBGROUP_FEATURE_BASIC_BIT : (op >= 334 && op <= 336) || (op >= 4428 && op <= 4430) ? VK_SUBGROUP_FEATURE_VOTE_BIT :   // Elect; All, Any, AllEqual (and SPV_KHR_subgroup_vote's)
           (op >= 337 && op <= 344) || op == 4421 || op == 4422 || op == 4432 ? VK_SUBGROUP_FEATURE_BALLOT_BIT :                               // Broadcast ... BallotFindMSB (and SPV_KHR_shader_ballot's)
           op == 345 || op == 346 ? VK_SUBGROUP_FEATURE_SHUFFLE_BIT : op == 347 || op == 348 ? VK_SUBGROUP_FEATURE_SHUFFLE_RELATIVE_BIT :
           op >= 349 && op <= 364 ? VK_SUBGROUP_FEATURE_ARITHMETIC_BIT : op == 365 || op == 366 ? VK_SUBGROUP_FEATURE_QUAD_BIT : 0;
}
// What a task or mesh shader asks of the device beyond VK_EXT_mesh_shader (XLATE's bundle: SPIR-V 1.4, subgroup operations in its stage, Workgroup memory (Mario Kart World's mesh shaders up to
// 26,452 bytes), a task payload, the workgroup size and mesh outputs of its execution modes), read from its module once (vkshader_t dev_ok) against the device's limits (vk_init device_facts).
// NULL when the device takes it; else why not, and a [RED] line once per shader
static const char *mesh_need(vkshader_t *s, int task) {
    if (s->dev_ok) return s->dev_ok == 1 ? NULL : task ? "the device cannot run its task shader (said where first met)" : "the device cannot run its mesh shader (said where first met)";
    static char why[240]; why[0] = 0;
    const uint32_t *w = s->spv; uint32_t nw = s->spv ? s->nspv : 0, bound = nw >= 5 ? w[3] : 0, ops = 0, lx = 1, ly = 1, lz = 1, ov = 0, opr = 0; uint64_t wg = 0, pl = 0;
    uint32_t lid[3] = {0}, wgs = 0, wgv[3] = {0};   // LocalSizeId's constants (SPIR-V 1.6 modules; resolved after the walk: execution modes come before constants), the WorkgroupSize built-in's
    uint64_t *sz = bound && bound < (1u << 22) ? calloc(bound, sizeof *sz) : NULL; uint32_t *pt = sz ? calloc(bound, sizeof *pt) : NULL, *cv = pt ? calloc(bound, sizeof *cv) : NULL;
    if (!cv) snprintf(why, sizeof why, "its SPIR-V is not at hand");
    else if (w[1] > V.spv_max) snprintf(why, sizeof why, "its SPIR-V is version 1.%u, the device takes up to 1.%u", w[1] >> 8 & 0xff, V.spv_max >> 8 & 0xff);
    else for (uint32_t i = 5; i < nw;) {
        uint32_t n = w[i] >> 16, op = w[i] & 0xffff; if (!n || i + n > nw) break;
        const uint32_t *a = w + i + 1;   // the operands
#define SPV_ID(k) (a[k] < bound ? a[k] : 0)
        switch (op) {
        case 16: if (n >= 6 && a[1] == 17) { lx = a[2]; ly = a[3]; lz = a[4]; } else if (n >= 4 && a[1] == 26) ov = a[2]; else if (n >= 4 && a[1] == 5270) opr = a[2]; break;   // OpExecutionMode LocalSize, OutputVertices, OutputPrimitivesEXT
        case 331: if (n >= 6 && a[1] == 38) { lid[0] = SPV_ID(2); lid[1] = SPV_ID(3); lid[2] = SPV_ID(4); } break;   // OpExecutionModeId LocalSizeId
        case 71: if (n >= 4 && a[1] == 11 && a[2] == 25) wgs = SPV_ID(0); break;       // OpDecorate BuiltIn WorkgroupSize (it overrides the execution mode)
        case 44: if (n >= 6 && wgs && a[1] == wgs) { wgv[0] = cv[SPV_ID(2)]; wgv[1] = cv[SPV_ID(3)]; wgv[2] = cv[SPV_ID(4)]; } break;   // OpConstantComposite: that built-in's value
        case 20: if (n >= 2) sz[SPV_ID(0)] = 4; break;                                  // OpTypeBool (as a 32-bit value)
        case 21: case 22: if (n >= 3) sz[SPV_ID(0)] = a[1] / 8; break;                  // OpTypeInt, OpTypeFloat
        case 23: case 24: if (n >= 4) sz[SPV_ID(0)] = sz[SPV_ID(1)] * a[2]; break;      // OpTypeVector, OpTypeMatrix
        case 28: if (n >= 4) sz[SPV_ID(0)] = sz[SPV_ID(1)] * cv[SPV_ID(2)]; break;      // OpTypeArray (its length a constant)
        case 30: if (n >= 2) { uint64_t t = 0; for (uint32_t k = 1; k + 1 < n; k++) t += sz[SPV_ID(k)]; sz[SPV_ID(0)] = t; } break;   // OpTypeStruct
        case 32: if (n >= 4) pt[SPV_ID(0)] = SPV_ID(2); break;                          // OpTypePointer: its pointee
        case 43: case 50: if (n >= 4) cv[SPV_ID(1)] = a[2]; break;                      // OpConstant, OpSpecConstant (the low word)
        case 59: if (n >= 4) { uint64_t b = sz[pt[SPV_ID(0)]]; if (a[2] == 4) wg += b; else if (a[2] == 5402) pl += b; } break;   // OpVariable: Workgroup, TaskPayloadWorkgroupEXT
        default: ops |= spv_group_op(op); break;
        }
#undef SPV_ID
        i += n;
    }
    if (wgv[0]) { lx = wgv[0]; ly = wgv[1]; lz = wgv[2]; } else if (cv && lid[0]) { lx = cv[lid[0]]; ly = cv[lid[1]]; lz = cv[lid[2]]; }
    uint32_t inv = lx * ly * lz, smem = task ? V.ml.task_smem : V.ml.mesh_smem, both = task ? V.ml.task_pl_smem : V.ml.mesh_pl_smem, mi = task ? V.ml.task_inv : V.ml.mesh_inv;
    const uint32_t *ms3 = task ? V.ml.task_size : V.ml.mesh_size; const char *sn = task ? "task" : "mesh";
    if (why[0]) {}
    else if (ops && !(V.sg_stages & (task ? VK_SHADER_STAGE_TASK_BIT_EXT : VK_SHADER_STAGE_MESH_BIT_EXT))) snprintf(why, sizeof why, "it uses subgroup operations (%#x) and the device has none in %s shaders (stages %#x)", ops, sn, V.sg_stages);
    else if (ops & ~V.sg_ops) snprintf(why, sizeof why, "it uses subgroup operations %#x the device lacks (it has %#x)", ops & ~V.sg_ops, V.sg_ops);
    else if (wg > smem) snprintf(why, sizeof why, "its %llu bytes of shared memory are past the device's %u", (unsigned long long)wg, smem);
    else if (task && pl > V.ml.task_payload) snprintf(why, sizeof why, "its %llu-byte payload is past the device's %u", (unsigned long long)pl, V.ml.task_payload);
    else if (wg + pl > both) snprintf(why, sizeof why, "its payload and shared memory (%llu bytes) are past the device's %u", (unsigned long long)(wg + pl), both);
    else if (inv > mi || lx > ms3[0] || ly > ms3[1] || lz > ms3[2]) snprintf(why, sizeof why, "its %ux%ux%u workgroup is past the device's (%u invocations, %ux%ux%u)", lx, ly, lz, mi, ms3[0], ms3[1], ms3[2]);
    else if (!task && (ov > V.ml.out_verts || opr > V.ml.out_prims)) snprintf(why, sizeof why, "its %u vertices and %u primitives are past the device's %u and %u", ov, opr, V.ml.out_verts, V.ml.out_prims);
    free(sz); free(pt); free(cv);
    s->dev_ok = why[0] ? 2 : 1;
    if (why[0]) LOG("[RED] [vk] %s shader %016llx: %s; the draws with it are skipped", sn, (unsigned long long)s->key, why);
    else VKLOG("%s shader %016llx: SPIR-V 1.%u, shared memory %llu bytes, payload %llu bytes, workgroup %ux%ux%u%s, subgroup operations %#x: the device takes it", sn, (unsigned long long)s->key,
               w[1] >> 8 & 0xff, (unsigned long long)wg, (unsigned long long)pl, lx, ly, lz, task ? "" : (snprintf(why, sizeof why, ", %u vertices and %u primitives", ov, opr), why), ops);
    return s->dev_ok == 1 ? NULL : why;
}
static uint64_t g_mesh_draws, g_mesh_skips;
static void mesh_draw(const p_draw *d) {
    if (d->count <= 0) return;
    if (!g_mesh_live) { g_mesh_skips++; return; }   // DM_VK_MESH=0 (live): left out
    vkshader_t *ts = S.sh[3], *ms = S.sh[4]; int notask = ts && g_mesh_notask; if (notask) ts = NULL;   // DM_VK_MESH_NOTASK: its pipeline without the task stage, the draw not made
    int tok = !ts || spv_model(ts) == SPV_TASK_EXT, mok = ms && spv_model(ms) == SPV_MESH_EXT;
    const char *why = !V.has_mesh ? "the device has no VK_EXT_mesh_shader" : !mok ? "the program has no mesh shader the bundle gives as MeshEXT" : !tok ? "its task shader is not TaskEXT SPIR-V" :
                      ts && !V.task_max_groups ? "the device has no task shaders" : ms->set >= 5 || (ts && ts->set >= 5) ? "its stages' descriptor sets are past the fifth" : NULL;
    if (!why) why = mesh_need(ms, 0); if (!why && ts) why = mesh_need(ts, 1);   // (SPIR-V version, subgroup operations, memory, outputs)
    if (why) { g_mesh_skips++; VKRED_ONCE(140, "DrawMeshTasks (task %016llx mesh %016llx): %s; skipped (Mario Kart World's bushes and grass go missing)", (unsigned long long)(ts ? ts->key : 0), (unsigned long long)(ms ? ms->key : 0), why); return; }
    uint32_t n = (uint32_t)d->count, max = ts ? V.task_max_groups : V.mesh_max_groups;
    if (max && n > max) { VKRED_ONCE(141, "DrawMeshTasks of %u workgroups: the device takes %u, the rest left out", n, max); n = max; }
    g_mesh.on = 1; g_mesh.notask = notask; g_mesh.ts = ts; g_mesh.ms = ms; g_mesh.first = (uint32_t)d->first; g_mesh.count = n;
    g_draw_binst = 0;
    int ok = prepare_draw(4, 0) && draw_allowed() && !(dbg_draw() & 1);
    if (ok && notask) { g_mesh_skips++; static int nl; if (!nl++) VKLOG("task / mesh: DM_VK_MESH_NOTASK=1, pipelines made without their task stage and their draws left out (first: task %016llx mesh %016llx)",
                                                                      (unsigned long long)S.sh[3]->key, (unsigned long long)ms->key); }
    else if (ok) { vkCmdDrawMeshTasksEXT(vk_cmd(), n, 1, 1); g_mesh_draws++; g_pu.mesh = pu_add(g_pu.mesh, 1);
        if (g_mesh_draws == 1) VKLOG("task / mesh: first draw, task %016llx mesh %016llx fragment %016llx, %u workgroups from %u", (unsigned long long)(ts ? ts->key : 0), (unsigned long long)ms->key, (unsigned long long)(S.sh[1] ? S.sh[1]->key : 0), n, (uint32_t)d->first); }
    g_mesh.on = 0; g_mesh.notask = 0;
}

// ---------------- DrawTexture ----------------
// nvnCommandBufferDrawTexture: the source region of a texture (texels, through the handle's sampler) drawn into the destination region of every bound colour target (pixels, the targets'
// own rows: the window origin does not turn them), cut by the scissor and each target's channel mask, with no blending (a [RED] once when it is on: as nvn_mtl.m draw_tex, which Streets of
// Rage 4's movie frames go through). Here a textured triangle of dm/android/shaders/drawtex.* inside the pass in progress, the targets' own render pass, with a pipeline per pass layout and mask
#include "drawtex_vert.spv.h"
#include "drawtex_frag.spv.h"
typedef struct { uint32_t ncol, cfmt[8], dfmt, samples, wmask[8]; } dtkey;
static struct { VkPipelineLayout pl; VkDescriptorSetLayout dl; VkShaderModule vs, fs; int bad; dtkey k[32]; VkPipeline p[32]; int n; } g_dt;
static VkPipeline dt_pipeline(const dtkey *k) {
    for (int i = 0; i < g_dt.n; i++) if (!memcmp(&g_dt.k[i], k, sizeof *k)) return g_dt.p[i];
    if (g_dt.bad || g_dt.n == 32) return VK_NULL_HANDLE;
    if (!g_dt.pl) {
        VkDescriptorSetLayoutBinding lb = {0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, NULL};
        VkDescriptorSetLayoutCreateInfo li = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, NULL, 0, 1, &lb};
        VkPushConstantRange pr = {VK_SHADER_STAGE_FRAGMENT_BIT, 0, 32};
        VkShaderModuleCreateInfo vi = {VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, NULL, 0, sizeof drawtex_vert_spv, drawtex_vert_spv}, fi = vi; fi.codeSize = sizeof drawtex_frag_spv; fi.pCode = drawtex_frag_spv;
        if (vkCreateDescriptorSetLayout(V.dev, &li, NULL, &g_dt.dl) || vkCreateShaderModule(V.dev, &vi, NULL, &g_dt.vs) || vkCreateShaderModule(V.dev, &fi, NULL, &g_dt.fs)) { g_dt.bad = 1; return VK_NULL_HANDLE; }
        void vk_dsl_register(VkDescriptorSetLayout l, const VkDescriptorSetLayoutBinding *b, uint32_t n); vk_dsl_register(g_dt.dl, &lb, 1);
        VkPipelineLayoutCreateInfo pli = {VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, NULL, 0, 1, &g_dt.dl, 1, &pr};
        if (vkCreatePipelineLayout(V.dev, &pli, NULL, &g_dt.pl)) { g_dt.bad = 1; return VK_NULL_HANDLE; }
    }
    VkPipelineShaderStageCreateInfo ss[2] = {{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, NULL, 0, VK_SHADER_STAGE_VERTEX_BIT, g_dt.vs, "main", NULL},
                                             {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, NULL, 0, VK_SHADER_STAGE_FRAGMENT_BIT, g_dt.fs, "main", NULL}};
    VkPipelineVertexInputStateCreateInfo vin = {VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
    VkPipelineInputAssemblyStateCreateInfo ia = {VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO, NULL, 0, VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST, 0};
    VkPipelineViewportStateCreateInfo vs = {VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO, NULL, 0, 1, NULL, 1, NULL};
    VkPipelineRasterizationStateCreateInfo rs = {VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO, NULL, 0, 0, 0, VK_POLYGON_MODE_FILL, VK_CULL_MODE_NONE, VK_FRONT_FACE_COUNTER_CLOCKWISE, 0, 0, 0, 0, 1.0f};
    VkPipelineMultisampleStateCreateInfo msi = {VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO, NULL, 0, (VkSampleCountFlagBits)k->samples, 0, 0, NULL, 0, 0};
    VkPipelineDepthStencilStateCreateInfo ds = {VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
    VkPipelineColorBlendAttachmentState cba[8]; memset(cba, 0, sizeof cba); for (uint32_t i = 0; i < k->ncol; i++) cba[i].colorWriteMask = k->wmask[i];
    VkPipelineColorBlendStateCreateInfo cbs = {VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO, NULL, 0, 0, 0, k->ncol, cba, {0, 0, 0, 0}};
    VkDynamicState dyn[] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
    VkPipelineDynamicStateCreateInfo dy = {VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO, NULL, 0, 2, dyn};
    VkFormat cf[8]; for (uint32_t i = 0; i < k->ncol; i++) cf[i] = (VkFormat)k->cfmt[i];
    VkFormat df = (VkFormat)k->dfmt; int hs = df == VK_FORMAT_D24_UNORM_S8_UINT || df == VK_FORMAT_D32_SFLOAT_S8_UINT || df == VK_FORMAT_S8_UINT;
    VkPipelineRenderingCreateInfo rci = {VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO, NULL, 0, k->ncol, cf, df == VK_FORMAT_S8_UINT ? VK_FORMAT_UNDEFINED : df, hs ? df : VK_FORMAT_UNDEFINED};
    VkRenderPass prp = rp_on() ? rp_compat(k->ncol, k->cfmt, k->dfmt, k->samples) : VK_NULL_HANDLE;
    VkGraphicsPipelineCreateInfo ci = {VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO, prp ? NULL : &rci, 0, 2, ss, &vin, &ia, NULL, &vs, &rs, &msi, &ds, &cbs, &dy, g_dt.pl, prp, 0, VK_NULL_HANDLE, -1};
    VkPipeline p = VK_NULL_HANDLE;
    if ((!prp && !V.has_dynrender) || vkCreateGraphicsPipelines(V.dev, V.pcache, 1, &ci, NULL, &p)) { VKRED_ONCE(142, "DrawTexture: its pipeline could not be made"); p = VK_NULL_HANDLE; }
    g_dt.k[g_dt.n] = *k; g_dt.p[g_dt.n++] = p; return p;
}
static void draw_tex(const p_draw_tex *c) {
    float dw = c->d.x1 - c->d.x0, dh = c->d.y1 - c->d.y0;
    if (dw == 0 || dh == 0 || !begin_pass()) return;
    const view_t *v = NULL; const tex_t *t = NULL; int tid = (int)(c->h & 0xfffff), sid = (int)(c->h >> 20);   // (the host table's IDs, hle_nvn.c: the sampler's may pass 12 bits)
    vkimg_t *im = NULL; VkImageView view = tid && vk_tex_by_id(tid, &v, &t) && t ? vk_tex_view(t, v, 1, 0, 0, &im) : VK_NULL_HANDLE;
    const vkfmt_t *f = t ? vk_fmt(v && v->set_format ? v->format : t->d.format) : NULL; VkSampler smp = sid ? vk_samp_by_id(sid) : VK_NULL_HANDLE;
    if (!view || !im || !smp || !f) { VKRED_ONCE(143, "DrawTexture: texture handle %#llx has no texture or sampler: not drawn", (unsigned long long)c->h); return; }
    if (f->is_int || f->depth || f->stencil || im->vtype != VK_IMAGE_VIEW_TYPE_2D) { VKRED_ONCE(144, "DrawTexture: source format %#x target %d not drawn (2D float or unorm only)", t->d.format, t->d.target); return; }
    if (S.color.blend_en) VKRED_ONCE(145, "DrawTexture with blending on (targets %#x): drawn without it", (unsigned)S.color.blend_en);
    dtkey k; memset(&k, 0, sizeof k); k.ncol = (uint32_t)S.ncol; k.dfmt = (uint32_t)S.dfmt; k.samples = S.samples ? S.samples : 1; int any = 0;
    for (int i = 0; i < S.ncol && i < 8; i++) { const vkfmt_t *tf = S.rt_has_col[i] ? vk_fmt(S.rt_col[i].d.format) : NULL; k.cfmt[i] = (uint32_t)S.cfmt[i];
        k.wmask[i] = S.cfmt[i] && tf && !tf->is_int ? (S.chmask.mask >> (4 * i)) & 0xf : 0; any |= k.wmask[i] != 0; }
    if (!any) return;
    VkPipeline p = dt_pipeline(&k); if (!p || !pass_commit()) return;
    // the destination (either corner order), within the scissor and the pass: the pixels whose centres are inside
    double x0 = fmin(c->d.x0, c->d.x1), x1 = fmax(c->d.x0, c->d.x1), y0 = fmin(c->d.y0, c->d.y1), y1 = fmax(c->d.y0, c->d.y1); VkRect2D sc = scissor_rect();
    x0 = fmax(x0, sc.offset.x); y0 = fmax(y0, sc.offset.y); x1 = fmin(x1, (double)sc.offset.x + sc.extent.width); y1 = fmin(y1, (double)sc.offset.y + sc.extent.height);
    int32_t sx = (int32_t)ceil(x0 - 0.5), sy = (int32_t)ceil(y0 - 0.5), ex = (int32_t)ceil(x1 - 0.5), ey = (int32_t)ceil(y1 - 0.5);
    if (ex <= sx || ey <= sy) return;
    VkDescriptorSet set = vk_alloc_set(g_dt.dl); VkDescriptorImageInfo ii = {smp, view, VK_IMAGE_LAYOUT_GENERAL};
    VkWriteDescriptorSet w = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, NULL, set, 0, 0, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &ii, NULL, NULL}; vkUpdateDescriptorSets(V.dev, 1, &w, 0, NULL);
    uint32_t lv = v && v->set_levels ? v->base_level : 0; float pc[8] = {c->d.x0, c->d.y0, (c->s.x1 - c->s.x0) / dw, (c->s.y1 - c->s.y0) / dh, c->s.x0, c->s.y0, (float)(im->w >> lv ? im->w >> lv : 1), (float)(im->h >> lv ? im->h >> lv : 1)};
    VkCommandBuffer cb = vk_cmd(); im->last_use = mtl_last_fence() + 1;
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, p); S.bound_pipe = VK_NULL_HANDLE;
    VkViewport vp = {0, 0, (float)S.pass_w, (float)S.pass_h, 0, 1}; VkRect2D r = {{sx, sy}, {(uint32_t)(ex - sx), (uint32_t)(ey - sy)}};
    vkCmdSetViewport(cb, 0, 1, &vp); vkCmdSetScissor(cb, 0, 1, &r);
    vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, g_dt.pl, 0, 1, &set, 0, NULL);
    vkCmdPushConstants(cb, g_dt.pl, VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof pc, pc);
    vkCmdDraw(cb, 3, 1, 0, 0); g_pass_draws++;
    g_bc_stale = 1;   // the next draw sets its viewport, scissor and bindings again (bc_fresh)
    { static int once; if (!once++) VKLOG("DrawTexture: first case %gx%g texels of a %ux%u texture (nvn format %#x) into %gx%g pixels of the pass's targets (%ux%u)", c->s.x1 - c->s.x0, c->s.y1 - c->s.y0, im->w, im->h, t->d.format, dw, dh, S.pass_w, S.pass_h); }
}

// ---------------- DLSS ----------------
static void copy_t2t(const p_copy_t2t *c);
// NVN2's nvnCommandBufferEvaluateDlss (p_dlss: the colour, depth, motion and output textures, sizes and rectangles): there is no DLSS here. The colour input's rectangle is scaled bilinearly into
// the output's, as nvn_mtl.m does when MetalFX makes no temporal scaler (its stand-in, copy_t2t_draw): a blit, outside the pass, with copy_t2t's checks
static void dlss_eval(const p_dlss *c) {
    if (!c->tex[DLSS_COLOR] || !c->tex[DLSS_OUTPUT]) { VKRED_ONCE(146, "DLSS evaluated without %s", c->tex[DLSS_COLOR] ? "an output" : "an input colour"); return; }
    const tex_t *ct = c->tex[DLSS_COLOR], *ot = c->tex[DLSS_OUTPUT];
    int iw = c->iw > 0 ? c->iw : (int)ct->d.width, ih = c->ih > 0 ? c->ih : (int)ct->d.height, ow = c->ow > 0 ? c->ow : (int)ot->d.width, oh = c->oh > 0 ? c->oh : (int)ot->d.height;
    VKRED_ONCE(147, "DLSS (network %#llx, %dx%d to %dx%d, flags %#x): no DLSS on this device, the colour input is scaled bilinearly into the output", (unsigned long long)c->net, iw, ih, ow, oh, c->flags);
    p_copy_t2t k; memset(&k, 0, sizeof k); k.src = ct; k.dst = ot; k.flags = 1;   // (bit 0: linear filtering, as a scaled copy's)
    k.sr = (NVNcopyRegion){c->base[DLSS_COLOR][0], c->base[DLSS_COLOR][1], 0, iw, ih, 1}; k.dr = (NVNcopyRegion){c->base[DLSS_OUTPUT][0], c->base[DLSS_OUTPUT][1], 0, ow, oh, 1};
    copy_t2t(&k);
}

// ---- compute shared memory in a buffer (gfx_vk.h RK_SPILL; XLATE's SharedMemorySpill.cs: Mario Kart World's 168 compute shaders of 48 KB) ----
// Workgroup ((z × count y) + y) × count x + x of a dispatch, its count NumWorkgroups, takes slice (that number mod the binding's samp_arg) of the spill buffer. Two dispatches never use it at
// once: every dispatch comes after a full barrier (dispatch, vk_img_barrier_all), and those of every other command too. Inside one dispatch, workgroups samp_arg apart share a slice, and nothing
// on the device keeps them from running together (in-order starts, but a slow workgroup stays resident while hundreds after it start and finish). So a dispatch of more workgroups than slices
// is cut into parts of at most that many consecutive ones (vkCmdDispatchBase: WorkgroupId keeps its place in the whole dispatch and NumWorkgroups is the part's, so the part's workgroups count
// a run of consecutive numbers from the slice formula and take distinct slices), with a compute-to-compute barrier between parts; c0's group counts stay the whole dispatch's (drv_segment).
// DM_VK_SPILL_SPLIT=0 (live): one dispatch, as nvn_mtl.m does (the A/B of the parts' cost). An indirect dispatch whose counts GPU work writes cannot be cut: [RED] once
static uint64_t g_spill_parts, g_spill_whole;   // dispatches cut into parts, and those left whole that would have needed it
static uint32_t spill_slices(const vkshader_t *cs) { for (int i = 0; cs && i < cs->nbind; i++) if (cs->b[i].kind == RK_SPILL) return cs->b[i].samp_arg > 0 ? (uint32_t)cs->b[i].samp_arg : 1; return 0; }
static void spill_dispatch(VkCommandBuffer cb, const vkshader_t *cs, uint32_t x, uint32_t y, uint32_t z, uint32_t sl) {
    uint64_t n = (uint64_t)x * y * z, parts = !n || n <= sl ? 1 : (uint64_t)x * y <= sl ? (z + sl / (x * y) - 1) / (sl / (x * y)) : x <= sl ? (uint64_t)z * ((y + sl / x - 1) / (sl / x)) : (uint64_t)z * y * ((x + sl - 1) / sl);
    if (parts == 1) { vkCmdDispatch(cb, x, y, z); return; }
    if (!g_spill_split || !V.has_dbase || parts > 4096) { g_spill_whole++; vkCmdDispatch(cb, x, y, z);
        VKRED_ONCE(157, "compute shader %016llx with its shared memory in a buffer: a dispatch of %ux%ux%u workgroups over %u slices drawn whole (%s): workgroups %u apart may share a slice",
                   (unsigned long long)cs->key, x, y, z, sl, !g_spill_split ? "DM_VK_SPILL_SPLIT=0" : !V.has_dbase ? "no vkCmdDispatchBase, a Vulkan 1.0 device" : "it would take over 4096 parts", sl); return; }
    VkMemoryBarrier mb = {VK_STRUCTURE_TYPE_MEMORY_BARRIER, NULL, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT}; uint32_t k = 0;
#define SPILL_PART(bx, by, bz, cx, cy, cz) do { if (k++) vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb, 0, NULL, 0, NULL); \
                                                vkCmdDispatchBase(cb, bx, by, bz, cx, cy, cz); } while (0)
    if ((uint64_t)x * y <= sl) { uint32_t c = sl / (x * y); for (uint32_t b = 0; b < z; b += c) SPILL_PART(0, 0, b, x, y, c < z - b ? c : z - b); }   // whole planes
    else if (x <= sl) { uint32_t c = sl / x; for (uint32_t zz = 0; zz < z; zz++) for (uint32_t b = 0; b < y; b += c) SPILL_PART(0, b, zz, x, c < y - b ? c : y - b, 1); }   // whole rows
    else for (uint32_t zz = 0; zz < z; zz++) for (uint32_t yy = 0; yy < y; yy++) for (uint32_t b = 0; b < x; b += sl) SPILL_PART(b, yy, zz, sl < x - b ? sl : x - b, 1, 1);   // runs of a row
#undef SPILL_PART
    g_spill_parts++; g_pu.spill = pu_add(g_pu.spill, k);   // (DM_VK_GPUPROF: the parts of the dispatch)
    if (g_spill_parts == 1) VKLOG("compute shader %016llx with its shared memory in a buffer: a dispatch of %ux%ux%u workgroups cut into %u parts of at most %u (slices), a barrier between parts",
                                  (unsigned long long)cs->key, x, y, z, k, sl);
}
// DM_VK_X_CSALT=<n>[:<m>:<r>] (live, debugging): compute pipelines made from now on take the bundle's code (0), DM_VK_SPV_DIR's (1) or DM_VK_SPV_DIR2's (2) where the file exists
// (gfx_vk_shader.c alt_module), in place of the start-up DM_VK_SPV_ALT; with m, only shaders whose key mod m is r (a bisection). Two translations A/B'd in the same scene, as a
// restart cannot promise (Mario Kart World's Free Roam starts at another place each time). Each choice's pipelines are kept apart (ck[3]'s top bits)
static int csalt_of(const vkshader_t *cs) { return g_x_csalt < 0 ? -1 : g_x_csalt_m && cs->key % g_x_csalt_m != g_x_csalt_r ? 0 : g_x_csalt; }
// DM_VK_X_CSSERIAL=1 (live, debugging): every dispatch of at most 4096 workgroups runs one workgroup at a time, in order (x fastest), a compute-to-compute barrier between them
// (vkCmdDispatchBase, as spill_dispatch's parts): whether an artifact comes from workgroups that run together or out of order on the device, as Maxwell and Ampere would not
static uint64_t g_serial_n;
static int serial_dispatch(VkCommandBuffer cb, uint32_t x, uint32_t y, uint32_t z) {
    if ((uint64_t)x * y * z > 4096 || (uint64_t)x * y * z < 2) return 0;
    VkMemoryBarrier mb = {VK_STRUCTURE_TYPE_MEMORY_BARRIER, NULL, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT};
    for (uint32_t k = 0; k < z; k++) for (uint32_t j = 0; j < y; j++) for (uint32_t i = 0; i < x; i++) {
        if (i | j | k) vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb, 0, NULL, 0, NULL);
        vkCmdDispatchBase(cb, i, j, k, 1, 1, 1); }
    if (!g_serial_n++) VKLOG("DM_VK_X_CSSERIAL: dispatches run one workgroup at a time (first: %ux%ux%u)", x, y, z);
    return 1;
}
static nvn_hmap g_cpipes = {.klen = 32};   // {shader, its integer-texel bindings, its LOD-shifted bindings (vk_lodshift_mask)}
static pthread_mutex_t g_cpipe_mu = PTHREAD_MUTEX_INITIALIZER;   // the replay thread and compute_warm
static VkPipeline compute_build(vkshader_t *cs, uint64_t um, uint64_t im, uint64_t lm, int csalt, int serial) {   // csalt: DM_VK_X_CSALT's choice for it, -1: the start-up DM_VK_SPV_ALT's   // (VkPipeline)1 when it fails
    VkDescriptorSetLayout dsl[5] = {cs->dsl}; VkPipeline p;
    int bs = build_begin();   // (the crash dump's note: build_pipeline)
    if (bs >= 0) { snprintf(g_build[bs].what, sizeof g_build[bs].what, "compute pipeline, shader %016llx, integer-texel views %#llx/%#llx", (unsigned long long)cs->key, (unsigned long long)um, (unsigned long long)im); g_build[bs].phase = 1; }
    // DM_VK_SPV_ALT=1 in env.txt (with DM_VK_SPV_DIR): compute shaders take the alternative code too, from the start (compute_warm builds them all then, and they are kept)
    static int alt = -1; if (alt < 0) { const char *e = getenv("DM_VK_SPV_ALT"); alt = e && atoi(e); }
    int a = csalt >= 0 ? csalt : alt; uint32_t xo = a && !um && !im ? (a == 2 ? 1u << 31 | 1u << 30 : 1u << 31) : 0;
    VkComputePipelineCreateInfo ci = {VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO, NULL, (spill_slices(cs) || serial) && V.has_dbase ? VK_PIPELINE_CREATE_DISPATCH_BASE_BIT : 0,   // (spill_dispatch's parts, serial_dispatch's)
        {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, NULL, 0, VK_SHADER_STAGE_COMPUTE_BIT, vk_shader_module_lod(cs, um, im, xo, 0, lm), "main", NULL}, vk_pipeline_layout(dsl, 1), VK_NULL_HANDLE, -1};
    if (bs >= 0) g_build[bs].phase = 2;
    VkResult r = vkCreateComputePipelines(V.dev, V.pcache, 1, &ci, NULL, &p); build_end(bs);
    if (r) { LOG("[RED] [vk] compute pipeline failed: %016llx", (unsigned long long)cs->key); p = (VkPipeline)1; }
    return p;
}
// Compute pipelines made ahead. One depends only on its shader (and integer-texel variants, rare), and building one costs Adreno 45-105 ms: Super Mario Odyssey's
// particle emitters run on compute shaders (EffectPtcl), and the first cap throw at a breakable object built one on the replay thread (a 65 ms frame), the first scenes
// several. A game has few (Odyssey 41), so at the first submission a background thread builds every compute shader in the bundle with no integer-texel variant (up to a cap, below)
static void *compute_warm(void *arg) {
    (void)arg; pthread_setname_np(pthread_self(), "vk compute warm"); setpriority(PRIO_PROCESS, 0, 10);
    int vk_shader_keys(uint32_t stage, uint64_t *out, int max); static uint64_t keys[4096]; int n = vk_shader_keys(5, keys, 4096), made = 0; uint64_t t0 = tick();
    // A bundle with many compute shaders is left to the replay: Monster Hunter Rise's (4096 or more) held ~550 MB more from the title on, and a Retroid Pocket 5
    // (Adreno 650, 8 GB) ran out of GPU memory loading the first area (vkAllocateMemory -2 at a 4 MB pool). DM_VK_COMPUTE_WARM_MAX (start-up, default 256): the most made ahead
    { const char *e = getenv("DM_VK_COMPUTE_WARM_MAX"); int cap = e ? atoi(e) : 256;
      if (n > cap) { VKLOG("compute pipelines made ahead: none, the bundle has %d%s compute shaders (more than %d, DM_VK_COMPUTE_WARM_MAX)", n, n == 4096 ? " or more" : "", cap); return NULL; } }
    for (int i = 0; i < n; i++) {
        uint64_t ck[4] = {keys[i], 0, 0, 0};
        pthread_mutex_lock(&g_cpipe_mu); int have = nvn_hm_get(&g_cpipes, ck) != NULL; pthread_mutex_unlock(&g_cpipe_mu);
        vkshader_t *cs = have ? NULL : vk_shader(keys[i], 5); if (!cs) continue;
        VkPipeline p = compute_build(cs, 0, 0, 0, -1, 0); made++;
        pthread_mutex_lock(&g_cpipe_mu); if (!nvn_hm_get(&g_cpipes, ck)) nvn_hm_put(&g_cpipes, ck, p); else if (p != (void *)1) vkDestroyPipeline(V.dev, p, NULL); pthread_mutex_unlock(&g_cpipe_mu);
    }
    VKLOG("compute pipelines made ahead: %d of %d compute shaders in %.0f ms", made, n, (tick() - t0) / 1e6);
    return NULL;
}
static void compute_warm_start(void) {
    if (getenv("DM_VK_NOPREWARM")) return;
    pthread_t t; pthread_attr_t a; pthread_attr_init(&a); pthread_attr_setdetachstate(&a, PTHREAD_CREATE_DETACHED); pthread_create(&t, &a, compute_warm, NULL); pthread_attr_destroy(&a);
}
// A compute shader's stores that only the dispatch's first invocation made (shader_vk.py first_store, upstream e07cba1c; nvn_mtl.m makes them the same way): taken out of the shader
// and made here, in order, after the barrier before the dispatch, so they land before every workgroup's work. Mario Kart World's culling lists reset their counters so (2ce9071c2f22d74f,
// 5fea9b7ddb30e8a8, 0823b08c2027a38d): made by the shader, a workgroup that started before the first one appended entries the reset then wiped, and the desert's cacti came and went in groups
static void first_stores(VkCommandBuffer cb, const vkshader_t *cs) {
    int n = 0;
    for (int i = 0; i < cs->nfirst; i++) {
        int r = cs->first[i].reg; uint64_t a = r >= 0 && r < 16 ? S.ssbo[5][r].addr : 0, at = a + 4ull * cs->first[i].word; VkBuffer b; VkDeviceSize o;
        if (!a || !vk_buf_at(at, 4, &b, &o)) { VKRED_ONCE(166, "compute shader %016llx: a first-invocation store to storage slot %d has no buffer there: not made", (unsigned long long)cs->key, r); continue; }
        vkCmdFillBuffer(cb, b, o, 4, cs->first[i].value); n++;
    }
    if (n) { VkMemoryBarrier mb = {VK_STRUCTURE_TYPE_MEMORY_BARRIER, NULL, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT};
             vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb, 0, NULL, 0, NULL); }
}
// The NDSR swap (gfx_vk_fsr.c vk_ndsr_swap, DM_VK_NDSR): a game's own upscaler dispatch (agl's NDSR network) drawn by EASU. Its two bindings as stage_set would make them,
// through tex_desc_for (so the image is taken as GPU-written, as the dispatch would leave it): its one texture as a sampled view, its one image as a storage view, each one's size
// at its view's level. 0 when the shader binds another number of either, an array, a texel buffer, a texture that is not 2D, or an integer, depth or compressed format
static int ndsr_tex(const dmsv_bind *b, int storage, vkfsr_tex *o) {
    const int st = 5;
    uint64_t h = storage ? (b->kind == RK_IMG ? (b->reg >= 0 && b->reg < 32 ? S.img[st][b->reg] : 0) : ubo_word(st, b->reg, b->off))
                         : b->kind == RK_TEX ? (b->reg >= 0 && b->reg < 256 ? S.tex[st][b->reg] : 0) : b->kind == RK_TEX_UBO ? ubo_word(st, b->reg, b->off) : (b->off >= 0 ? c2_handle(st, b->off) : 0);
    int tid = (int)(h & 0xfffff); const view_t *v = NULL; const tex_t *t = NULL;
    if (!tid || !vk_tex_by_id(tid, &v, &t) || !t || t->d.target != NVN_TEXTURE_TARGET_2D) return 0;
    const vkfmt_t *f = vk_fmt(v && v->set_format ? v->format : t->d.format);
    if (!f || f->is_int || f->depth || f->stencil || f->compressed) return 0;
    int lv = v && v->set_levels ? v->base_level : 0;
    VkDescriptorImageInfo ii; tex_desc_for(st, b, h, &ii, storage);
    if (!ii.imageView) return 0;
    o->view = ii.imageView; o->w = t->d.width >> lv ? t->d.width >> lv : 1; o->h = t->d.height >> lv ? t->d.height >> lv : 1;
    return 1;
}
static int ndsr_binds(const vkshader_t *cs, vkfsr_tex *in, vkfsr_tex *out) {
    const dmsv_bind *tb = NULL, *ib = NULL;
    for (int i = 0; i < cs->nbind; i++) {
        const dmsv_bind *b = &cs->b[i];
        if (b->kind == RK_TEX || b->kind == RK_TEX_UBO || b->kind == RK_TEX_C2A) { if (tb) return 0; tb = b; }
        else if (b->kind == RK_IMG || b->kind == RK_IMG_UBO) { if (ib) return 0; ib = b; }
        else if (b->kind == RK_TBUF || b->kind == RK_TBUF_UBO || b->kind == RK_IMG_TBUF) return 0;
    }
    return tb && ib && tb->count <= 1 && ib->count <= 1 && ndsr_tex(tb, 0, in) && ndsr_tex(ib, 1, out);
}
static void dispatch(const p_dispatch *d) {
    vkshader_t *cs = S.sh[5];
    vk_end_pass_land();   // (first, even for a dispatch left out: the discards' lookahead takes every dispatch as a pass's end)
    if (!cs) { VKRED_ONCE(60, "dispatch with no compute shader in the bundle: skipped"); return; }
    g_vkst.dispatches++;
    VkCommandBuffer cb = vk_cmd();
    // a game's own upscaler (agl's NDSR network: DM_GAME_UPSCALER_KEYS) drawn by EASU instead, before its pipeline is made (gfx_vk_fsr.c "NDSR swap"; DM_VK_NDSR=game keeps it)
    if (!d->ind && vk_ndsr_wanted(cs->key)) { vkfsr_tex ti, to; uint32_t pv[8]; memcpy(pv, S.push[1], sizeof pv);
        if (ndsr_binds(cs, &ti, &to) && vk_ndsr_swap(cb, cs, &ti, &to, pv)) { g_pu.disp = pu_add(g_pu.disp, 1); return; } }
    VkDescriptorSetLayout dsl[5] = {cs->dsl}; VkPipelineLayout layout = vk_pipeline_layout(dsl, 1);
    // the group counts before the descriptors: the driver segment stage_set builds carries them at c0 + 0x200 (drv_segment), as nvn_mtl.m's does. Set after it, every
    // dispatch read the previous one's: Link's Awakening's light culling (tile = workgroup y × c0[0x200] + workgroup x) ran after a 2x1x1 dispatch, so of its 30x17
    // tiles it wrote only 0-61. An indirect dispatch's counts are on the GPU only (0 here; nvn_mtl.m copies them in)
    g_dispatch[0] = (uint32_t)d->x; g_dispatch[1] = (uint32_t)d->y; g_dispatch[2] = (uint32_t)d->z;
    uint32_t dofs[96]; g_c0cs_b = VK_NULL_HANDLE; VkDescriptorSet set = stage_set(5, cs, dofs);
    int ca = csalt_of(cs); uint64_t lmk = vk_lodshift_mask(cs, g_lod_shift[5]);   // (stage_set's tex_desc_for noted each binding's LOD shift)
    int ser = g_x_csserial && V.has_dbase && (!g_x_csserial_m || cs->key % g_x_csserial_m == g_x_csserial_r);   // (with m: only the shaders whose key mod m is r, a bisection)
    uint64_t ck[4] = {cs->key, g_num_um, g_num_im, lmk | (ca >= 0 ? (uint64_t)(ca + 1) << 62 : 0) | (uint64_t)ser << 61};   // (DM_VK_X_CSALT's choice in the top bits: a LOD-shift mask has 64 bindings at most, 8 set)
    pthread_mutex_lock(&g_cpipe_mu); VkPipeline p = nvn_hm_get(&g_cpipes, ck); pthread_mutex_unlock(&g_cpipe_mu);
    if (!p) {
        uint64_t tb = g_prof_on ? tick() : 0; p = compute_build(cs, ck[1], ck[2], lmk, ca, ser); if (g_prof_on) g_prof_ns[7] += tick() - tb;
        pthread_mutex_lock(&g_cpipe_mu); VkPipeline had = nvn_hm_get(&g_cpipes, ck);
        if (had) { if (p != (void *)1) vkDestroyPipeline(V.dev, p, NULL); p = had; } else nvn_hm_put(&g_cpipes, ck, p);   // (compute_warm made it meanwhile)
        pthread_mutex_unlock(&g_cpipe_mu);
    }
    if (p == (void *)1) return;
    for (int i = 0; i < g_x_nskipcs; i++) if (cs->key == g_x_skipcs[i]) return;   // DM_VK_X_SKIPCS=<key>[,<key>...] (live, up to 8): dispatches of these compute shaders are left out (bisect what an artifact needs)
    vk_img_barrier_all(cb);
    // An indirect dispatch's group counts are written by the GPU, usually earlier in the frame, and the shader reads them at c0 + 0x200 (NVN's driver constants: Maxwell and
    // Ampere code reads its grid size there, not from a register), where the CPU put 0 (hle_nvn.c records 0x0x0): copied in first, as nvn_mtl.m does. Left at 0, a shader that
    // numbers its workgroups y × count x + x gives every row the numbers of the first. Mario Kart World makes such dispatches; the A/B on the Fold (DM_VK_IND_C0=0, live) showed
    // no change in what it draws (its flickering trees were the base instance of indirect draws: shader_vk.py DRAW_PARAMS)
    if (d->ind && g_ind_c0 && g_c0cs_b) { VkBuffer ib; VkDeviceSize io;
        if (vk_buf_at(d->ind, 12, &ib, &io)) { VkBufferCopy r = {io, g_c0cs_o + 0x200, 12}; vkCmdCopyBuffer(cb, ib, g_c0cs_b, 1, &r);
            if (!(g_ind_c0_n++ & 0xffff)) VKLOG("indirect dispatches with their group counts copied into c0 + 0x200: %llu (shader %016llx)", (unsigned long long)g_ind_c0_n, (unsigned long long)cs->key);
            VkMemoryBarrier mb = {VK_STRUCTURE_TYPE_MEMORY_BARRIER, NULL, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_UNIFORM_READ_BIT | VK_ACCESS_SHADER_READ_BIT};
            vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb, 0, NULL, 0, NULL); } }
    if (cs->nfirst && g_first_store) { first_stores(cb, cs); g_pu.first = pu_add(g_pu.first, (uint32_t)cs->nfirst); }   // the first invocation's stores, ahead of every workgroup (first_stores)
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, p);
    vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, layout, 0, 1, &set, (uint32_t)cs->ndyn, dofs);
    if (lmk) { uint32_t wv = vk_lodshift_word(cs, lmk, g_lod_shift[5]); vkCmdPushConstants(cb, layout, VK_SHADER_STAGE_COMPUTE_BIT, vk_lodshift_off(5), 4, &wv); }   // (its LOD shifts)
    g_pu.disp = pu_add(g_pu.disp, 1);   // (DM_VK_GPUPROF: in the stretch between passes)
    uint32_t sl = spill_slices(cs);   // (RK_SPILL: spill_dispatch)
    if (d->ind) { VkBuffer b; VkDeviceSize o; uint32_t gc[3];
        // DM_VK_X_SPILL_CPUIND=0 (live, debugging): never from the CPU's copy of the counts, which is right only when no GPU write in flight is missed (vk_gpu_pending knows
        // dispatches' bound storage buffers, copies and fills, not global-memory stores through a pointer nor graphics stages' storage writes)
        if ((sl || ser) && (g_spill_cpuind || ser) && vk_cpu_ptr(d->ind, 12) && !vk_gpu_pending(d->ind, 12)) { vk_ubd_read(d->ind, 12, (uint8_t *)gc);   // counts the CPU wrote
            if (sl && !(g_spill_cpuind_n++ & 0x3fff)) VKLOG("spilled compute shaders' indirect dispatches: %llu with the CPU's counts (this one %ux%ux%u, shader %016llx), %llu from the GPU's", (unsigned long long)g_spill_cpuind_n, gc[0], gc[1], gc[2], (unsigned long long)cs->key, (unsigned long long)g_spill_gpuind_n);
            if (!(ser && serial_dispatch(cb, gc[0], gc[1], gc[2]))) { if (sl) spill_dispatch(cb, cs, gc[0], gc[1], gc[2], sl); else vkCmdDispatch(cb, gc[0], gc[1], gc[2]); } }
        else if (vk_buf_at(d->ind, 12, &b, &o)) { vkCmdDispatchIndirect(cb, b, o);
            if (sl) { g_spill_whole++; g_spill_gpuind_n++; VKRED_ONCE(158, "compute shader %016llx with its shared memory in a buffer: an indirect dispatch whose counts GPU work writes, not cut into parts: workgroups %u apart may share a slice", (unsigned long long)cs->key, sl); } } }
    else if (ser && serial_dispatch(cb, (uint32_t)d->x, (uint32_t)d->y, (uint32_t)d->z)) {}
    else if (sl) spill_dispatch(cb, cs, (uint32_t)d->x, (uint32_t)d->y, (uint32_t)d->z, sl);
    else vkCmdDispatch(cb, (uint32_t)d->x, (uint32_t)d->y, (uint32_t)d->z);
    if (!g_x_nodispbar) vk_img_barrier_all(cb);   // DM_VK_X_NODISPBAR=1 (live, a cost measurement that may draw wrong): no barrier after the dispatch
    if (crumbs_on()) { crumb_t *c = crumb_cur(); if (c->n < 8) { c->k[c->n][0] = 0; c->k[c->n][1] = 0; c->k[c->n][2] = cs->key; c->n++; } crumb_mark(1); }
    if (g_x_tracepass == -2) {   // (DM_VK_TRACE_PASS=-2 lists the dispatches too, with their storage buffers: the addresses for a DM_CTL memdump)
        char sb[512]; int o = 0; for (int r = 0; r < 16 && o < (int)sizeof sb - 40; r++) if (S.ssbo[5][r].addr) o += snprintf(sb + o, sizeof sb - (size_t)o, " s%d %#llx+%#llx", r, (unsigned long long)S.ssbo[5][r].addr, (unsigned long long)S.ssbo[5][r].size);
        VKLOG("--- dispatch %016llx %dx%dx%d%s after pass %d;%s", (unsigned long long)cs->key, d->x, d->y, d->z, d->ind ? " (indirect)" : "", g_sub_pass - 1, o ? sb : " no storage buffers");
    }
}

// ---------------- copies and clears ----------------
// a level of the texture the image lacks (vkimg_t lv0: a dense image of a sparse texture, or a sparse one past the sparse address space, starts at a coarser level): a copy or clear of it is left out
static int lv_missing(const vkimg_t *im, const view_t *v, int has_v) {
    uint32_t lv = has_v && v->set_levels ? v->base_level : 0; if (lv >= im->lv0) return 0;
    VKRED_ONCE(162, "a copy or clear of level %u of a texture whose image starts at its level %u (a sparse texture's dense image, or a sparse one past the address space): left out", lv, im->lv0); return 1;
}
static VkImageSubresourceLayers sub_of(vkimg_t *im, const view_t *v, int has_v, int z) {
    uint32_t lv = has_v && v->set_levels ? v->base_level : 0, ly = has_v && v->set_layers ? v->min_layer : 0; lv = lv >= im->lv0 ? lv - im->lv0 : 0;   // (the image's levels: lv_missing)
    int layered = im->layers > 1;
    return (VkImageSubresourceLayers){im->aspect & VK_IMAGE_ASPECT_COLOR_BIT ? VK_IMAGE_ASPECT_COLOR_BIT : (im->aspect & VK_IMAGE_ASPECT_DEPTH_BIT ? VK_IMAGE_ASPECT_DEPTH_BIT : VK_IMAGE_ASPECT_STENCIL_BIT), lv, ly + (layered ? (uint32_t)z : 0), 1};
}
// A region of an image made in its texture format's wider twin (gfx_vk_tex.c "promoted formats") into buffer b in the texture's own format: blitted to an image of that format (the conversion
// rounds as drawing into it would), copied out of that. 0 when the formats cannot be blitted ([RED] once)
static int promo_t2b(VkCommandBuffer cb, vkimg_t *im, const VkBufferImageCopy *r, VkBuffer b) {
    VkFormat nf = vk_promo_from(im); VkFormatProperties sp, dp; vkGetPhysicalDeviceFormatProperties(V.pd, im->fmt, &sp); vkGetPhysicalDeviceFormatProperties(V.pd, nf, &dp);
    if (!(sp.optimalTilingFeatures & VK_FORMAT_FEATURE_BLIT_SRC_BIT) || !(dp.optimalTilingFeatures & VK_FORMAT_FEATURE_BLIT_DST_BIT)) {
        VKRED_ONCE(176, "copy of a sparse texture made in format %d, the wider twin of its format %d, into a buffer: they cannot be blitted, skipped", im->fmt, nf); return 0; }
    uint32_t w = r->imageExtent.width, h = r->imageExtent.height, nl = r->imageSubresource.layerCount ? r->imageSubresource.layerCount : 1;
    VkImageCreateInfo ic = {VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, NULL, 0, VK_IMAGE_TYPE_2D, nf, {w, h, 1}, 1, nl, VK_SAMPLE_COUNT_1_BIT, VK_IMAGE_TILING_OPTIMAL,
                            VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT, VK_SHARING_MODE_EXCLUSIVE, 0, NULL, VK_IMAGE_LAYOUT_UNDEFINED};
    VkImage ti; if (vkCreateImage(V.dev, &ic, NULL, &ti)) return 0;
    VkMemoryRequirements rq; vkGetImageMemoryRequirements(V.dev, ti, &rq); VkDeviceMemory tm;
    VkMemoryAllocateInfo ma = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, NULL, rq.size, vk_mem_type(rq.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 0)};
    if (vkAllocateMemory(V.dev, &ma, NULL, &tm)) { vkDestroyImage(V.dev, ti, NULL); return 0; }
    VKCHK(vkBindImageMemory(V.dev, ti, tm, 0));
    VkImageMemoryBarrier ib = {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER, NULL, 0, VK_ACCESS_TRANSFER_WRITE_BIT, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL, VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED, ti, {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, nl}};
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, NULL, 0, NULL, 1, &ib);
    VkImageBlit bl = {r->imageSubresource, {{r->imageOffset.x, r->imageOffset.y, 0}, {r->imageOffset.x + (int32_t)w, r->imageOffset.y + (int32_t)h, 1}}, {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, nl}, {{0, 0, 0}, {(int32_t)w, (int32_t)h, 1}}};
    vkCmdBlitImage(cb, im->img, VK_IMAGE_LAYOUT_GENERAL, ti, VK_IMAGE_LAYOUT_GENERAL, 1, &bl, VK_FILTER_NEAREST); vk_img_barrier_all(cb);
    VkBufferImageCopy c2 = *r; c2.imageSubresource = (VkImageSubresourceLayers){VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, nl}; c2.imageOffset = (VkOffset3D){0, 0, 0};
    vkCmdCopyImageToBuffer(cb, ti, VK_IMAGE_LAYOUT_GENERAL, b, 1, &c2);
    vk_defer_free_image(ti, tm);
    return 1;
}
static void copy_b2t(const p_copy_b2t *c, int to_texture) {
    vk_end_pass_land();
    vkimg_t *im = vk_tex(c->tex); const vkfmt_t *f = vk_fmt(c->tex->d.format); if (to_texture) vk_dsc_wrote(im); else vk_dsc_read(im, "copied to a buffer");
    if (!im || !f || lv_missing(im, &c->view, c->has_view)) return;
    VkBuffer b; VkDeviceSize o;
    if (!vk_buf_at(c->src, 1, &b, &o)) { VKRED_ONCE(70, "buffer of a buffer ↔ texture copy outside the memory pools"); return; }
    VkCommandBuffer cb = vk_cmd(); vk_tex_prepare(cb, im); vk_img_barrier_all(cb);
    unsigned bw = f->bw ? f->bw : 1, bh = f->bh ? f->bh : 1;
    uint32_t rowlen = S.row_stride > 0 && f->bpb ? (uint32_t)(S.row_stride / f->bpb) * bw : 0;
    uint32_t imgh = S.image_stride > 0 && S.row_stride > 0 ? (uint32_t)(S.image_stride / S.row_stride) * bh : 0;
    int layered = im->layers > 1, depth = c->r.depth ? c->r.depth : 1;
    VkBufferImageCopy r = {o, rowlen, imgh, sub_of(im, &c->view, c->has_view, c->r.zoffset), {c->r.xoffset, c->r.yoffset, layered ? 0 : c->r.zoffset},
                           {(uint32_t)c->r.width, (uint32_t)(c->r.height ? c->r.height : 1), layered ? 1 : (uint32_t)depth}};
    if (layered) r.imageSubresource.layerCount = (uint32_t)depth;
    if (f->dec) {   // a BC format the device lacks (gfx_vk_tex.c bc_decode): the blocks decoded on the CPU from the buffer's memory as the replay reaches the copy
        if (!to_texture) { VKRED_ONCE(71, "copy of a BC texture decoded on the CPU into a buffer: skipped"); return; }
        uint32_t w = (uint32_t)c->r.width, h = (uint32_t)(c->r.height ? c->r.height : 1), nz = (uint32_t)depth;
        size_t pitch = S.row_stride > 0 ? (size_t)S.row_stride : (size_t)((w + 3) / 4) * f->bpb, slice = S.image_stride > 0 ? (size_t)S.image_stride : pitch * ((h + 3) / 4), out = (size_t)w * h * f->dbpp;
        const uint8_t *src = vk_cpu_ptr(c->src, slice * nz); if (!src) { VKRED_ONCE(163, "BC copy from a virtual pool's range the CPU cannot read in one piece: skipped"); return; }
        VkBuffer sb; VkDeviceSize so; uint8_t *p = vk_scratch(out * nz, 16, &sb, &so);
        for (uint32_t z = 0; z < nz; z++) vk_bc_decode(f, src + z * slice, pitch, w, h, 1, p + z * out);
        b = sb; r.bufferOffset = so; r.bufferRowLength = 0; r.bufferImageHeight = 0;
    }
    // a sparse texture made in its format's wider twin (gfx_vk_tex.c "promoted formats"): into it, the buffer's texels widened on the CPU as the replay reaches the copy (as a BC format decoded
    // on the CPU is, from the buffer's memory); out of it, through an image of the texture's own format (promo_t2b)
    if (im->promo && to_texture) {
        uint32_t w = (uint32_t)c->r.width, h = (uint32_t)(c->r.height ? c->r.height : 1), nz = (uint32_t)depth; size_t row = (size_t)w * vk_promo_bpt(im);
        size_t pitch = S.row_stride > 0 ? (size_t)S.row_stride : row, slice = S.image_stride > 0 ? (size_t)S.image_stride : pitch * h, n = slice * (nz - 1) + pitch * (h - 1) + row;
        const uint8_t *src = vk_cpu_ptr(c->src, n); if (!src) { VKRED_ONCE(177, "copy into a sparse texture made in its format's wider twin, from a range the CPU cannot read in one piece: skipped"); return; }
        int vk_gpu_pending(uint64_t a, uint64_t n); if (vk_gpu_pending(c->src, n)) VKRED_ONCE(178, "copy into a sparse texture made in its format's wider twin, from memory the GPU writes in work not complete: read as it is");
        VkBuffer sb; VkDeviceSize so; uint8_t *p = vk_scratch((VkDeviceSize)row * 2 * h * nz, 16, &sb, &so);
        for (uint32_t z = 0; z < nz; z++) for (uint32_t yy = 0; yy < h; yy++) vk_promo_widen(im, src + z * slice + yy * pitch, w, p + ((size_t)z * h + yy) * row * 2);
        b = sb; r.bufferOffset = so; r.bufferRowLength = 0; r.bufferImageHeight = 0;
    } else if (im->promo) {
        void vk_gpu_wrote(uint64_t a, uint64_t n); if (promo_t2b(cb, im, &r, b)) { vkpool_t *p = vk_pool_of(c->src, 1); vk_gpu_wrote(c->src, p ? p->hi - c->src : 1); }
        vk_img_barrier_all(cb); return;
    }
    if (g_watch_off >= 0) { char w[96]; snprintf(w, sizeof w, "buffer copy %s (buffer %#llx, %dx%d)", to_texture ? "destination" : "source", (unsigned long long)c->src, c->r.width, c->r.height); vk_watch(im, w); }
    if (to_texture) { vkCmdCopyBufferToImage(cb, b, im->img, VK_IMAGE_LAYOUT_GENERAL, 1, &r); vk_img_wrote(im); }
    else { vkCmdCopyImageToBuffer(cb, im->img, VK_IMAGE_LAYOUT_GENERAL, b, 1, &r);
        void vk_gpu_wrote(uint64_t a, uint64_t n); vkpool_t *p = vk_pool_of(c->src, 1); vk_gpu_wrote(c->src, p ? p->hi - c->src : 1); }   // (to the pool's end: the copy's extent in memory is not worked out here)
    vk_img_barrier_all(cb);
}
// Scaled copies of depth and stencil: Adreno blits no depth-stencil format (Pokémon Legends: Arceus copies its 1920x1080 D24S8 depth to a 480x270 one, which stayed unwritten), so each
// aspect goes image → buffer, through dm/android/shaders/depthscale.comp (the nearest texel), buffer → image. Two device buffers kept from copy to copy, grown when a region needs more
#include "depthscale_comp.spv.h"
static struct { VkPipeline p; VkPipelineLayout pl; VkDescriptorSetLayout dl; VkDescriptorPool dp; VkBuffer b[2]; VkDeviceMemory m[2]; VkDeviceSize n[2]; VkDescriptorSet set; int bad; } g_dsc;
static int dsc_init(void) {
    if (g_dsc.p || g_dsc.bad) return !g_dsc.bad;
    VkDescriptorSetLayoutBinding lb[2] = {{0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL}, {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL}};
    VkDescriptorSetLayoutCreateInfo li = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, NULL, 0, 2, lb};
    VkPushConstantRange pr = {VK_SHADER_STAGE_COMPUTE_BIT, 0, 20};
    VkDescriptorPoolSize ps = {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 128};
    VkDescriptorPoolCreateInfo pi = {VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO, NULL, 0, 64, 1, &ps};
    VkShaderModuleCreateInfo mi = {VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, NULL, 0, sizeof depthscale_comp_spv, depthscale_comp_spv}; VkShaderModule sm = VK_NULL_HANDLE;
    if (vkCreateDescriptorSetLayout(V.dev, &li, NULL, &g_dsc.dl) || vkCreateDescriptorPool(V.dev, &pi, NULL, &g_dsc.dp) || vkCreateShaderModule(V.dev, &mi, NULL, &sm)) { g_dsc.bad = 1; return 0; }
    VkPipelineLayoutCreateInfo pli = {VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, NULL, 0, 1, &g_dsc.dl, 1, &pr};
    VkComputePipelineCreateInfo ci = {VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO, NULL, 0, {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, NULL, 0, VK_SHADER_STAGE_COMPUTE_BIT, sm, "main", NULL}};
    if (vkCreatePipelineLayout(V.dev, &pli, NULL, &g_dsc.pl) || (ci.layout = g_dsc.pl, vkCreateComputePipelines(V.dev, VK_NULL_HANDLE, 1, &ci, NULL, &g_dsc.p))) g_dsc.bad = 1;
    vkDestroyShaderModule(V.dev, sm, NULL);
    if (g_dsc.bad) VKLOG("[RED] the scaled depth copy's compute pipeline could not be made");
    return !g_dsc.bad;
}
static int dsc_buf(int i, VkDeviceSize n) {   // buffer i holds at least n bytes
    if (g_dsc.n[i] >= n) return 1;
    if (g_dsc.b[i]) vk_defer_free_buffer(g_dsc.b[i], g_dsc.m[i]);
    g_dsc.b[i] = VK_NULL_HANDLE; g_dsc.n[i] = 0; g_dsc.set = VK_NULL_HANDLE; n = n < (1u << 20) ? 1u << 20 : n + n / 2;
    VkBufferCreateInfo bi = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, NULL, 0, n, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_SHARING_MODE_EXCLUSIVE};
    if (vkCreateBuffer(V.dev, &bi, NULL, &g_dsc.b[i])) return 0;
    VkMemoryRequirements rq; vkGetBufferMemoryRequirements(V.dev, g_dsc.b[i], &rq);
    VkMemoryAllocateInfo ai = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, NULL, rq.size, vk_mem_type(rq.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 0)};
    if (vkAllocateMemory(V.dev, &ai, NULL, &g_dsc.m[i])) { vkDestroyBuffer(V.dev, g_dsc.b[i], NULL); g_dsc.b[i] = VK_NULL_HANDLE; return 0; }
    vkBindBufferMemory(V.dev, g_dsc.b[i], g_dsc.m[i], 0); g_dsc.n[i] = n;
    return 1;
}
static int depth_scale(VkCommandBuffer cb, vkimg_t *s, vkimg_t *d, VkImageSubresourceLayers ss, VkImageSubresourceLayers ds, const p_copy_t2t *k) {   // 1: copied
    if (s->fmt != d->fmt || !(s->aspect & (VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT)) || s->aspect != d->aspect || !dsc_init()) return 0;
    uint32_t sw = (uint32_t)k->sr.width, sh = (uint32_t)k->sr.height, dw = (uint32_t)k->dr.width, dh = (uint32_t)k->dr.height;
    for (int a = 0; a < 2; a++) {
        VkImageAspectFlags as = a ? VK_IMAGE_ASPECT_STENCIL_BIT : VK_IMAGE_ASPECT_DEPTH_BIT; if (!(s->aspect & as)) continue;
        uint32_t bytes = a ? 1 : s->fmt == VK_FORMAT_D16_UNORM ? 0 : 4; if (!bytes) return 0;   // (16-bit depth: two texels a word, not handled)
        if (!dsc_buf(0, ((VkDeviceSize)sw * sh * bytes + 3) & ~3ull) || !dsc_buf(1, ((VkDeviceSize)dw * dh * bytes + 3) & ~3ull)) return 0;
        if (!g_dsc.set) {
            VkDescriptorSetAllocateInfo sa = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, NULL, g_dsc.dp, 1, &g_dsc.dl};
            if (vkAllocateDescriptorSets(V.dev, &sa, &g_dsc.set)) { g_dsc.set = VK_NULL_HANDLE; VKRED_ONCE(76, "scaled depth copy: no descriptor set left"); return 0; }
            VkDescriptorBufferInfo bi[2] = {{g_dsc.b[0], 0, VK_WHOLE_SIZE}, {g_dsc.b[1], 0, VK_WHOLE_SIZE}};
            VkWriteDescriptorSet w[2] = {{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, NULL, g_dsc.set, 0, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, NULL, &bi[0], NULL},
                                         {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, NULL, g_dsc.set, 1, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, NULL, &bi[1], NULL}};
            vkUpdateDescriptorSets(V.dev, 2, w, 0, NULL);
        }
        VkImageSubresourceLayers sl = ss, dl = ds; sl.aspectMask = as; dl.aspectMask = as;
        VkBufferImageCopy r = {0, 0, 0, sl, {k->sr.xoffset, k->sr.yoffset, 0}, {sw, sh, 1}};
        vkCmdCopyImageToBuffer(cb, s->img, VK_IMAGE_LAYOUT_GENERAL, g_dsc.b[0], 1, &r); vk_img_barrier_all(cb);
        uint32_t pc[5] = {sw, sh, dw, dh, bytes}, n = bytes == 4 ? dw * dh : (dw * dh + 3) / 4, groups = (n + 63) / 64, gx = groups < 65535 ? groups : 65535, gy = (groups + gx - 1) / gx;
        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, g_dsc.p);
        vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, g_dsc.pl, 0, 1, &g_dsc.set, 0, NULL);
        vkCmdPushConstants(cb, g_dsc.pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof pc, pc);
        vkCmdDispatch(cb, gx, gy, 1); vk_img_barrier_all(cb);
        VkBufferImageCopy w = {0, 0, 0, dl, {k->dr.xoffset, k->dr.yoffset, 0}, {dw, dh, 1}};
        vkCmdCopyBufferToImage(cb, g_dsc.b[1], d->img, VK_IMAGE_LAYOUT_GENERAL, 1, &w); vk_img_barrier_all(cb);
    }
    S.bound_pipe = VK_NULL_HANDLE;   // the compute pipeline took the bind point the replay's dispatches use
    return 1;
}
static void copy_t2t(const p_copy_t2t *c) {
    vk_end_pass();
    vkimg_t *s = vk_tex(c->src), *d = vk_tex(c->dst); vk_dsc_read(s, "a copy's source"); vk_dsc_wrote(d);
    if (!s || !d || lv_missing(s, &c->sv, c->has_sv) || lv_missing(d, &c->dv, c->has_dv)) return;
    vk_watch(s, "copy source"); vk_watch(d, "copy destination");
    VkCommandBuffer cb = vk_cmd(); vk_tex_prepare(cb, s); vk_tex_prepare(cb, d); vk_img_barrier_all(cb);
    VkImageSubresourceLayers ss = sub_of(s, &c->sv, c->has_sv, c->sr.zoffset), ds = sub_of(d, &c->dv, c->has_dv, c->dr.zoffset);
    // Vulkan refuses (vkEndCommandBuffer fails for the whole submission) a region outside either image or a blit the format cannot do: regions are clipped
    // to both images' levels, a copy with nothing left is skipped
    int32_t sw = (int32_t)(s->w >> ss.mipLevel ? s->w >> ss.mipLevel : 1), sh = (int32_t)(s->h >> ss.mipLevel ? s->h >> ss.mipLevel : 1);
    int32_t dw = (int32_t)(d->w >> ds.mipLevel ? d->w >> ds.mipLevel : 1), dh = (int32_t)(d->h >> ds.mipLevel ? d->h >> ds.mipLevel : 1);
    p_copy_t2t k = *c; c = &k; vk_img_wrote(d);
    if (k.sr.height <= 0) k.sr.height = 1; if (k.dr.height <= 0) k.dr.height = 1;
    if (ss.mipLevel >= s->levels || ds.mipLevel >= d->levels || k.sr.xoffset < 0 || k.sr.yoffset < 0 || k.dr.xoffset < 0 || k.dr.yoffset < 0 || k.sr.xoffset >= sw || k.sr.yoffset >= sh || k.dr.xoffset >= dw || k.dr.yoffset >= dh) {
        VKRED_ONCE(72, "texture copy outside its images (src %dx%d @%d,%d of %dx%d level %u, dst %dx%d @%d,%d of %dx%d level %u): skipped", c->sr.width, c->sr.height, c->sr.xoffset, c->sr.yoffset, sw, sh, ss.mipLevel,
                   c->dr.width, c->dr.height, c->dr.xoffset, c->dr.yoffset, dw, dh, ds.mipLevel); vk_img_barrier_all(cb); return; }
    int same = k.sr.width == k.dr.width && k.sr.height == k.dr.height;
    if (k.sr.xoffset + k.sr.width > sw) { VKRED_ONCE(73, "texture copy source region clipped to its image"); k.sr.width = sw - k.sr.xoffset; if (same) k.dr.width = k.sr.width; }
    if (k.sr.yoffset + k.sr.height > sh) { VKRED_ONCE(73, "texture copy source region clipped to its image"); k.sr.height = sh - k.sr.yoffset; if (same) k.dr.height = k.sr.height; }
    if (k.dr.xoffset + k.dr.width > dw) { VKRED_ONCE(74, "texture copy destination region clipped to its image"); k.dr.width = dw - k.dr.xoffset; if (same) k.sr.width = k.dr.width; }
    if (k.dr.yoffset + k.dr.height > dh) { VKRED_ONCE(74, "texture copy destination region clipped to its image"); k.dr.height = dh - k.dr.yoffset; if (same) k.sr.height = k.dr.height; }
    if (k.sr.width <= 0 || k.sr.height <= 0 || k.dr.width <= 0 || k.dr.height <= 0) { vk_img_barrier_all(cb); return; }
    if (!(c->flags & DM_COPY_RESOLVE) && !(k.sr.width == k.dr.width && k.sr.height == k.dr.height && s->fmt == d->fmt)) {   // a blit: both formats must allow it (and linear filtering)
        VkFormatProperties sp, dp; vkGetPhysicalDeviceFormatProperties(V.pd, s->fmt, &sp); vkGetPhysicalDeviceFormatProperties(V.pd, d->fmt, &dp);
        if (!(sp.optimalTilingFeatures & VK_FORMAT_FEATURE_BLIT_SRC_BIT) || !(dp.optimalTilingFeatures & VK_FORMAT_FEATURE_BLIT_DST_BIT) || (s->aspect != d->aspect)) {
            if (depth_scale(cb, s, d, ss, ds, &k)) return;
            VKRED_ONCE(75, "texture blit between formats %d → %d that cannot be blitted (%dx%d → %dx%d, present %llu): skipped", s->fmt, d->fmt, k.sr.width, k.sr.height, k.dr.width, k.dr.height, (unsigned long long)g_tex_frame); vk_img_barrier_all(cb); return; }
        if (!(sp.optimalTilingFeatures & VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT) || (s->aspect & ~VK_IMAGE_ASPECT_COLOR_BIT)) k.flags &= ~1u;   // nearest where linear is not allowed
    }
    if (s == d && !memcmp(&ss, &ds, sizeof ss) && k.sr.xoffset == k.dr.xoffset && k.sr.yoffset == k.dr.yoffset && k.sr.width == k.dr.width && k.sr.height == k.dr.height && k.sr.zoffset == k.dr.zoffset) {
        vk_img_barrier_all(cb); return; }   // a copy of a region onto itself (two textures over the same storage): nothing to do, and Vulkan forbids overlapping copies
    if (c->flags & DM_COPY_RESOLVE) {
        VkImageResolve r = {ss, {c->sr.xoffset, c->sr.yoffset, 0}, ds, {c->dr.xoffset, c->dr.yoffset, 0}, {(uint32_t)c->sr.width, (uint32_t)c->sr.height, 1}};
        vkCmdResolveImage(cb, s->img, VK_IMAGE_LAYOUT_GENERAL, d->img, VK_IMAGE_LAYOUT_GENERAL, 1, &r);
    } else if (c->sr.width == c->dr.width && c->sr.height == c->dr.height && s->fmt == d->fmt) {
        VkImageCopy r = {ss, {c->sr.xoffset, c->sr.yoffset, s->layers > 1 ? 0 : c->sr.zoffset}, ds, {c->dr.xoffset, c->dr.yoffset, d->layers > 1 ? 0 : c->dr.zoffset},
                         {(uint32_t)c->sr.width, (uint32_t)(c->sr.height ? c->sr.height : 1), s->layers > 1 ? 1 : (uint32_t)(c->sr.depth ? c->sr.depth : 1)}};
        vkCmdCopyImage(cb, s->img, VK_IMAGE_LAYOUT_GENERAL, d->img, VK_IMAGE_LAYOUT_GENERAL, 1, &r);
    } else {
        VkImageBlit r = {ss, {{c->sr.xoffset, c->sr.yoffset, 0}, {c->sr.xoffset + c->sr.width, c->sr.yoffset + (c->sr.height ? c->sr.height : 1), 1}},
                         ds, {{c->dr.xoffset, c->dr.yoffset, 0}, {c->dr.xoffset + c->dr.width, c->dr.yoffset + (c->dr.height ? c->dr.height : 1), 1}}};
        vkCmdBlitImage(cb, s->img, VK_IMAGE_LAYOUT_GENERAL, d->img, VK_IMAGE_LAYOUT_GENERAL, 1, &r, (c->flags & 1) ? VK_FILTER_LINEAR : VK_FILTER_NEAREST);
    }
    vk_img_barrier_all(cb);
}
static void clear_tex(const p_clear_tex *c) {
    vk_end_pass();
    vkimg_t *im = vk_tex(c->tex); vk_dsc_wrote(im); if (!im || lv_missing(im, &c->view, c->has_view)) return;
    VkCommandBuffer cb = vk_cmd(); vk_tex_prepare(cb, im); vk_img_barrier_all(cb); vk_img_wrote(im);
    VkImageSubresourceLayers l = sub_of(im, &c->view, c->has_view, c->r.zoffset);
    VkImageSubresourceRange rg = {l.aspectMask, l.mipLevel, 1, l.baseArrayLayer, im->layers > 1 ? (uint32_t)(c->r.depth ? c->r.depth : 1) : 1};
    if (c->r.xoffset || c->r.yoffset || (uint32_t)c->r.width < (im->w >> l.mipLevel) || (uint32_t)c->r.height < (im->h >> l.mipLevel)) VKRED_ONCE(71, "partial texture clear: the whole level is cleared");
    if (im->aspect & VK_IMAGE_ASPECT_COLOR_BIT) { VkClearColorValue v; memcpy(&v, c->c.u, 16); vkCmdClearColorImage(cb, im->img, VK_IMAGE_LAYOUT_GENERAL, &v, 1, &rg); }
    else { VkClearDepthStencilValue v = {c->c.f[0], c->c.u[1]}; rg.aspectMask = im->aspect; vkCmdClearDepthStencilImage(cb, im->img, VK_IMAGE_LAYOUT_GENERAL, &v, 1, &rg); }
    vk_img_barrier_all(cb);
}

// ---------------- queues and programs ----------------
// NVN keeps the GPU's bindings per queue, across submissions (nvn_mtl.m queue_state: Final Fantasy X/X-2 destroys its launcher's queue before it creates a game's, and Fire Emblem: Three
// Houses submits on two queues): each queue's state shadow (S, and its transform feedback bindings) is saved when another queue's submission comes and restored at its own
// (mtl_queue_activate, hle_nvn.c, under the submit lock before mtl_sub_begin). The first queue takes the state as it is; another new one starts from the defaults
typedef struct qstate { uint64_t serial; typeof(S) s; p_tf_bind tf[4]; int tf_on; struct qstate *next; } qstate;
static qstate *g_qs, *g_qcur;
void mtl_queue_activate(uint64_t serial) {
    if (g_qcur && g_qcur->serial == serial) return;
    if (g_qcur) { g_qcur->s = S; memcpy(g_qcur->tf, g_tf, sizeof g_tf); g_qcur->tf_on = g_tf_on; }
    qstate *q = g_qs; while (q && q->serial != serial) q = q->next;
    if (q) { S = q->s; memcpy(g_tf, q->tf, sizeof g_tf); g_tf_on = q->tf_on; }
    else { if (!(q = calloc(1, sizeof *q))) die("[vk] queue state allocation failed"); q->serial = serial; q->next = g_qs;
        if (g_qs) { state_reset(); memset(g_tf, 0, sizeof g_tf); g_tf_on = 0; } g_qs = q;
        static int n; if (n++ < 8) VKLOG("queue %llu: %s", (unsigned long long)serial, g_qcur ? "a second queue, its bindings kept apart" : "the first queue"); }
    g_qcur = q; for (int st = 0; st < 6; st++) { g_bind_gen[st]++; g_ubo_gen[st]++; } g_last_p = VK_NULL_HANDLE;   // (the memos and the last pipeline belong to the other queue's state)
}
void mtl_queue_forget(uint64_t serial) {   // Finalize, submit lock held: its state, guest object pointers included, goes
    qstate **pp = &g_qs; while (*pp && (*pp)->serial != serial) pp = &(*pp)->next;
    if (!*pp) return;
    qstate *q = *pp; *pp = q->next;
    if (q == g_qcur) { g_qcur = NULL; state_reset(); memset(g_tf, 0, sizeof g_tf); g_tf_on = 0; for (int st = 0; st < 6; st++) { g_bind_gen[st]++; g_ubo_gen[st]++; } }
    free(q);
}
// A finalized program leaves every queue's bindings (NVN reads a program only when it is bound, and a game may finalize one a queue still has bound and reuse its memory: Fire Emblem
// Fortune's Weave at its title, nvn_mtl.m mtl_prog_forget). The stages it resolved to stay (S.sh: the backend's shaders, not the game's memory); only the pointer goes
void mtl_prog_forget(const prog_t *p) {
    if (S.prog == p) S.prog = NULL;
    for (qstate *q = g_qs; q; q = q->next) if (q != g_qcur && q->s.prog == p) q->s.prog = NULL;
}
// A new program's shaders made in the background (hle_nvn.c SetShaders; nvn_mtl.m loads its shader libraries there): the module, set layout and reflection (gfx_vk_shader.c vk_shader)
// are ready by its first draw instead of being made on the replay thread then. The keys are taken now (the program is the game's memory), the modules once the device is made.
// DM_VK_PREFETCH=0 (start-up): at the first draw, as before
enum { PFQ = 8192 }; static uint64_t g_pfq[PFQ]; static unsigned g_pfq_h, g_pfq_t; static int g_pfq_on; static uint64_t g_pfq_made;
static pthread_mutex_t g_pfq_mu = PTHREAD_MUTEX_INITIALIZER; static pthread_cond_t g_pfq_cv = PTHREAD_COND_INITIALIZER;
static void *prefetch_thread(void *arg) {
    (void)arg; pthread_setname_np(pthread_self(), "vk prefetch"); setpriority(PRIO_PROCESS, 0, 10);
    while (!vk_ready()) usleep(20000);
    for (;;) {
        pthread_mutex_lock(&g_pfq_mu); while (g_pfq_h == g_pfq_t) pthread_cond_wait(&g_pfq_cv, &g_pfq_mu); uint64_t k = g_pfq[g_pfq_t++ % PFQ]; pthread_mutex_unlock(&g_pfq_mu);
        int vk_shader_stage(uint64_t key); int st = vk_shader_stage(k); if (st >= 0 && vk_shader(k, st)) g_pfq_made++;
    }
    return NULL;
}
void mtl_prog_prefetch(const prog_t *p) {
    static int on = -1; if (on < 0) { const char *e = getenv("DM_VK_PREFETCH"); on = !(e && !atoi(e)); }
    if (!on || !p) return;
    for (unsigned i = 0; i < 6; i++) { uint8_t k[8]; if (!prog_key(p, i, k)) break; uint64_t kk = key64(k); if (!kk) continue;
        pthread_mutex_lock(&g_pfq_mu);
        if (!g_pfq_on) { g_pfq_on = 1; pthread_t t; pthread_attr_t a; pthread_attr_init(&a); pthread_attr_setdetachstate(&a, PTHREAD_CREATE_DETACHED); pthread_create(&t, &a, prefetch_thread, NULL); pthread_attr_destroy(&a); }
        if (g_pfq_h - g_pfq_t < PFQ) { g_pfq[g_pfq_h++ % PFQ] = kk; pthread_cond_signal(&g_pfq_cv); }   // (a full queue drops the key: that shader is made at its first draw)
        pthread_mutex_unlock(&g_pfq_mu); }
}
// The pipeline prewarm (prewarm_start, recorded pipelines built on background threads), for a front end that waits on it (nvn_mtl.m mtl_warm_wait): 0 when none runs, else the ms waited + 1
int mtl_warm_wait(int timeout_ms) {
    if (!g_pw_n || __atomic_load_n(&g_pw_done, __ATOMIC_RELAXED) >= g_pw_n) return 0;
    uint64_t t0 = tnow();
    while (__atomic_load_n(&g_pw_next, __ATOMIC_RELAXED) < g_pw_n && (tnow() - t0) / 1000000ull < (uint64_t)timeout_ms) usleep(2000);
    return (int)((tnow() - t0) / 1000000ull) + 1;
}
// The pipelines line of the host's periodic report (hle_nvn.c "pipelines up to frame")
void mtl_pso_line(char *o, size_t n) {
    snprintf(o, n, "%llu graphics pipelines built, %zu of %zu recorded ones prewarmed, %llu draws left out while theirs were built in the background, %llu shaders made ahead of their first draw",
             (unsigned long long)g_vkst.pipes_built, (size_t)__atomic_load_n(&g_pw_done, __ATOMIC_RELAXED), g_pw_n, (unsigned long long)g_vkst.skip_pipe, (unsigned long long)g_pfq_made);
}

// ---------------- the record walker ----------------
static int rt_same(const p_set_rt *r) {   // the targets the open pass draws into already
    int n = r->n > 8 ? 8 : r->n; if (n != S.rt_n) return 0;
    for (int i = 0; i < n; i++) {
        int has = r->color[i] != NULL && r->color[i]->magic == MAGIC('T', 'E', 'X', 0);
        if (has != S.rt_has_col[i] || (has && memcmp(r->color[i], &S.rt_col[i], sizeof S.rt_col[i]))) return 0;
        if (r->has_cview[i] != S.rt_has_cview[i] || (r->has_cview[i] && memcmp(&r->cview[i], &S.rt_cview[i], sizeof S.rt_cview[i]))) return 0;
    }
    int hd = r->depth != NULL && r->depth->magic == MAGIC('T', 'E', 'X', 0);
    if (hd != S.rt_has_dep || (hd && memcmp(r->depth, &S.rt_dep, sizeof S.rt_dep))) return 0;
    if (r->has_dview != S.rt_has_dview || (r->has_dview && memcmp(&r->dview, &S.rt_dview, sizeof S.rt_dview))) return 0;
    return 1;
}
static void set_rt(const p_set_rt *r) {
    if (S.pass_open && !g_x_nokeep && rt_same(r)) { g_vkst.rt_kept++; return; }   // the same targets again (Unity sets them before many draws): the pass goes on
    vk_end_pass();
    S.rt_n = r->n > 8 ? 8 : r->n;
    for (int i = 0; i < S.rt_n; i++) {
        S.rt_has_col[i] = r->color[i] != NULL && r->color[i]->magic == MAGIC('T', 'E', 'X', 0); if (S.rt_has_col[i]) S.rt_col[i] = *r->color[i];
        S.rt_bad[i] = r->color[i] && !S.rt_has_col[i] ? r->color[i]->magic | 1u : 0;
        S.rt_has_cview[i] = r->has_cview[i]; if (r->has_cview[i]) S.rt_cview[i] = r->cview[i];
    }
    S.rt_has_dep = r->depth != NULL && r->depth->magic == MAGIC('T', 'E', 'X', 0); if (S.rt_has_dep) S.rt_dep = *r->depth;
    S.rt_has_dview = r->has_dview; if (r->has_dview) S.rt_dview = r->dview;
}
static void bind_handles_rec(uint32_t op, const void *a) {
    p_bind_handles tmp; const p_bind_handles *b = bind_rec(op, a, &tmp);
    if (b->defer && b->src && b->n > 0 && b->n <= 32) { const void *src = vk_cpu_ptr(b->src, 8 * (uint64_t)b->n); if (src) memcpy((void *)b->h, src, 8 * (size_t)b->n); else memset((void *)b->h, 0, 8 * (size_t)b->n); }   // (a virtual pool's range: vk_cpu_ptr)
    int st = b->stage; if (st < 0 || st > 5) return;
    for (int i = 0; i < b->n; i++) {
        int slot = b->first + i;
        uint64_t *d = NULL;
        uint64_t *q = NULL;
        if (op == OP_BIND_TEX || op == OP_BIND_TEX1) { if (slot >= 0 && slot < 256) { d = &S.tex[st][slot]; q = &S.tex_sseq[st][slot]; } }
        else if (op == OP_BIND_SAMP) { if (slot >= 0 && slot < 128) { d = &S.samp[st][slot]; q = &S.samp_sseq[st][slot]; } }
        else if (op == OP_BIND_IMG) { if (slot >= 0 && slot < 32) d = &S.img[st][slot]; }
        if (d && (*d != b->h[i] || (q && *q != b->sseq))) { *d = b->h[i]; if (q) *q = b->sseq; g_bind_gen[st]++; }   // only a change counts: games bind the same handles again before many draws (stage_set's memo)
    }
}
uint32_t g_sub_ops[64];   // record ops replayed in the open submission, by op (logged when the driver refuses it)
static void exec_ops(const uint8_t *p);
// DM_VK_PROF=1 (start-up): a 1 ms CPU-time timer on the replay thread samples where it is; every 20 s the most frequent places are logged: offsets in this library,
// time in other libraries charged to the first caller of ours ("via:"), or with =2 the place itself (libc, the driver). Resolve offsets with llvm-symbolizer
// on the debug build's libdmhost.so. Simpleperf is refused on retail devices; this found the command pool resets and render pass begins that took half the replay
#include <signal.h>
#include <sys/syscall.h>
static _Atomic uint64_t g_prof_pc[16384], g_prof_leaf[16384], g_prof_lr[16384], g_prof_st[16384][6]; static _Atomic uint32_t g_prof_n; static uintptr_t g_prof_lo, g_prof_hi;   // this library's text, for the first frame of ours
static void prof_sig(int sig, siginfo_t *si, void *ucv) { (void)sig; (void)si; ucontext_t *u = ucv; uint64_t pc = u->uc_mcontext.pc;
    if (pc < g_prof_lo || pc >= g_prof_hi) {   // outside this library: charge it to our first caller on the frame chain (marked with bit 63)
        uint64_t lr = u->uc_mcontext.regs[30] & 0x7fffffffffull, fp = u->uc_mcontext.regs[29], sp = u->uc_mcontext.sp, c = 0;
        if (lr >= g_prof_lo && lr < g_prof_hi) c = lr;
        for (int k = 0; k < 12 && !c && fp && !(fp & 7) && fp >= sp && fp - sp < 0x400000; k++) { uint64_t r = ((const uint64_t *)(uintptr_t)fp)[1] & 0x7fffffffffull; if (r >= g_prof_lo && r < g_prof_hi) c = r; fp = ((const uint64_t *)(uintptr_t)fp)[0]; }
        if (c) pc = c | (1ull << 63); }
    uint32_t i = atomic_fetch_add(&g_prof_n, 1); if (i < 16384) { atomic_store(&g_prof_pc[i], pc); atomic_store(&g_prof_leaf[i], u->uc_mcontext.pc); atomic_store(&g_prof_lr[i], u->uc_mcontext.regs[30]);
        uint64_t fp = u->uc_mcontext.regs[29], sp = u->uc_mcontext.sp;   // the frame chain's return addresses (DM_VK_PROF_OUT)
        for (int k = 0; k < 6; k++) { uint64_t r = 0; if (fp && !(fp & 7) && fp >= sp && fp - sp < 0x400000) { r = ((const uint64_t *)(uintptr_t)fp)[1] & 0x7fffffffffull; fp = ((const uint64_t *)(uintptr_t)fp)[0]; } else fp = 0; atomic_store(&g_prof_st[i][k], r); } } }
static void prof_tick(void) {
    // armed by the first DM_VK_PROF=1 it sees, read every 64 frames: a ctl.txt env line can start it after loading (Xenoblade Chronicles 3 crashed loading a save with it on from start-up)
    static int on; static unsigned chk; static uint64_t last; if (!on && !(chk++ & 63)) { const char *e = getenv("DM_VK_PROF"); on = e && atoi(e);
        if (on) { Dl_info d0; if (dladdr((void *)prof_tick, &d0)) { g_prof_lo = (uintptr_t)d0.dli_fbase; g_prof_hi = g_prof_lo + 0x4000000; }
            struct sigaction sa; memset(&sa, 0, sizeof sa); sa.sa_sigaction = prof_sig; sa.sa_flags = SA_SIGINFO | SA_RESTART; sigaction(SIGPROF, &sa, NULL);
            // DM_VK_PROF_TID=<tid>: sample that thread of the process instead (its CPU-time clock; e.g. the game's main thread), and DM_VK_PROF_OUT=<file>: append the raw samples (leaf pc, lr) there
            const char *tt = getenv("DM_VK_PROF_TID"); pid_t tid = tt && atoi(tt) > 0 ? (pid_t)atoi(tt) : (pid_t)syscall(SYS_gettid);
            clockid_t clk = tt && atoi(tt) > 0 ? (clockid_t)((~(unsigned)tid << 3) | 6) : CLOCK_THREAD_CPUTIME_ID;   // Linux MAKE_THREAD_CPUCLOCK(tid, CPUCLOCK_SCHED)
            struct sigevent ev; memset(&ev, 0, sizeof ev); ev.sigev_notify = SIGEV_THREAD_ID; ev.sigev_signo = SIGPROF; ev._sigev_un._tid = tid;
            timer_t tm; if (!timer_create(clk, &ev, &tm)) { struct itimerspec it = {{0, 1000000}, {0, 1000000}}; timer_settime(tm, 0, &it, NULL); VKLOG("[prof] sampling thread %d every 1 ms of its CPU time", (int)tid); } else VKLOG("[RED] [prof] timer_create for thread %d failed", (int)tid); } }
    if (!on) return; uint64_t now = tnow(); if (!last) last = now; if (now - last < 20000000000ull) return; last = now;
    uint32_t n = atomic_exchange(&g_prof_n, 0); if (n > 16384) n = 16384;
    { const char *po = getenv("DM_VK_PROF_OUT"); FILE *pf = po ? fopen(po, "ab") : NULL;
      if (pf) { for (uint32_t i = 0; i < n; i++) { uint64_t v[8] = {atomic_load(&g_prof_leaf[i]), atomic_load(&g_prof_lr[i])}; for (int k = 0; k < 6; k++) v[2 + k] = atomic_load(&g_prof_st[i][k]); fwrite(v, sizeof v, 1, pf); } fclose(pf); } }
    Dl_info di; uintptr_t base = dladdr((void *)prof_tick, &di) ? (uintptr_t)di.dli_fbase : 0;
    static uint64_t pcs[16384]; static uint32_t cnt[16384]; int np = 0;
    static int leaves = -1; if (leaves < 0) { const char *e = getenv("DM_VK_PROF"); leaves = e && atoi(e) == 2; }   // DM_VK_PROF=2: where the time is, not who called
    for (uint32_t i = 0; i < n; i++) { uint64_t pc = atomic_load(leaves ? &g_prof_leaf[i] : &g_prof_pc[i]); int k = 0; for (; k < np; k++) if (pcs[k] == pc) break; if (k == np) { pcs[np] = pc; cnt[np++] = 0; } cnt[k]++; }
    char line[4000]; int o = snprintf(line, sizeof line, "[prof] %u samples, top pcs (library offset:count):", n);
    for (int t = 0; t < 40 && o < (int)sizeof line - 40; t++) { int b = -1; for (int k = 0; k < np; k++) if (cnt[k] && (b < 0 || cnt[k] > cnt[b])) b = k; if (b < 0) break;
        uint64_t pc = pcs[b]; Dl_info d2;
        if (pc >> 63) o += snprintf(line + o, sizeof line - o, " via:%llx:%u", (unsigned long long)((pc & ~(1ull << 63)) - base), cnt[b]);
        else if (pc - base < 0x10000000) o += snprintf(line + o, sizeof line - o, " %llx:%u", (unsigned long long)(pc - base), cnt[b]);
        else if (dladdr((void *)(uintptr_t)pc, &d2) && d2.dli_fname) o += snprintf(line + o, sizeof line - o, " %s(%s)+%llx:%u", strrchr(d2.dli_fname, '/') ? strrchr(d2.dli_fname, '/') + 1 : d2.dli_fname, d2.dli_sname ? d2.dli_sname : "?", (unsigned long long)(pc - (uintptr_t)(d2.dli_saddr ? d2.dli_saddr : d2.dli_fbase)), cnt[b]);
        else o += snprintf(line + o, sizeof line - o, " ?%llx:%u", (unsigned long long)pc, cnt[b]);
        cnt[b] = 0; }
    VKLOG("%s", line);
}
void mtl_exec(const uint8_t *p) {   // timed at the top level only (OP_CALL nests)
    static int depth; uint64_t t0 = depth ? 0 : tnow(); depth++; if (depth == 1) prof_tick();
    exec_ops(p);
    if (!--depth) { g_vkst.ns_exec += tnow() - t0; g_op_cur = NULL; g_ncall = 0; }   // (between command handles no record is being replayed: the discards' lookahead)
}
static void exec_ops(const uint8_t *p) {
    for (;;) {
        op_hdr h; memcpy(&h, p, sizeof h);
        if (h.op < 64) g_sub_ops[h.op]++;
        if (g_skip_ops == ~0ull) { g_skip_ops = 0; const char *e = getenv("DM_VK_SKIP_OPS"); for (const char *q = e; q && *q;) { char *n; long v = strtol(q, &n, 10); if (n == q) break; if (v > 0 && v < 64) g_skip_ops |= 1ull << v; q = *n ? n + 1 : n; } }   // debugging: record ops not replayed
        if (h.op < 64 && h.op > 3 && (g_skip_ops >> h.op & 1)) { p += h.size; continue; }
        const uint8_t *a = p + 8;
        g_cur_op = (int)h.op; g_op_cur = p;
        switch (h.op) {
        case OP_END: return;
        case OP_BEGIN: break;
        case OP_JUMP: { const uint8_t *t; memcpy(&t, a, 8); p = t; continue; }
        case OP_CALL: { uint64_t t; memcpy(&t, a, 8); if (g_ncall < 4) g_call_ret[g_ncall] = p + h.size; g_ncall++; exec_ops((const uint8_t *)(uintptr_t)t); g_ncall--; break; }   // (the return: the discards' lookahead)
        case OP_SIGNAL_EVENT: { volatile uint32_t *val; uint32_t m, v; memcpy(&val, a, 8); memcpy(&m, a + 8, 4); memcpy(&v, a + 12, 4); mtl_sub_event(val, m, v); break; }
        case OP_REPORT_COUNTER: { uint64_t t, ad; memcpy(&t, a, 8); memcpy(&ad, a + 8, 8); if (DM_GAME_RENDER_COND) rc_note(ad, t == 0 || t == 0x10 ? 1 : 2); mtl_sub_counter(ad, (uint32_t)t); break; }
        case OP_RESET_COUNTER: { extern uint64_t g_ctr_reset[18]; uint64_t t; memcpy(&t, a, 8); g_ctr_reset[t < 17 ? t : 17]++; break; }   // counted only (gfx_vk_core.c counter census)
        case OP_FENCE_SYNC: { sync_t *s; memcpy(&s, a, 8); s->fence = mtl_last_fence() + 1; s->fenced = 1; break; }
        // (DM_VK_TRACE_PASS=-2 lists the buffer writes too: copies, clears and uniform updates, by address)
        case OP_COPY_B2B: { vk_end_pass_land(); g_pu.copies = pu_add(g_pu.copies, 1); uint64_t s_, d, n; memcpy(&s_, a, 8); memcpy(&d, a + 8, 8); memcpy(&n, a + 16, 8); mtl_sub_copy(s_, d, n);
            if (g_x_tracepass == -2) VKLOG("--- copy %#llx → %#llx, %llu bytes, after pass %d", (unsigned long long)s_, (unsigned long long)d, (unsigned long long)n, g_sub_pass - 1); break; }
        case OP_CLEAR_BUF: { vk_end_pass_land(); g_pu.clears = pu_add(g_pu.clears, 1); const p_clear_buf *c = (const p_clear_buf *)a; mtl_sub_fill(c->addr, c->size, c->value);
            if (g_x_tracepass == -2) VKLOG("--- buffer clear %#llx, %llu bytes to %#x, after pass %d", (unsigned long long)c->addr, (unsigned long long)c->size, c->value, g_sub_pass - 1); break; }
        case OP_UPDATE_UB: { uint64_t d, n; memcpy(&d, a, 8); memcpy(&n, a + 8, 8); vk_sub_update(d, a + 16, n);
            if (g_x_tracepass == -2) VKLOG("--- uniform update %#llx, %llu bytes, after pass %d", (unsigned long long)d, (unsigned long long)n, g_sub_pass - 1); break; }
        case OP_SET_RT: set_rt((const p_set_rt *)a); break;
        case OP_CLEAR_COLOR: if (!g_rc_skip) clear_color((const p_clear_color *)a); break;
        case OP_CLEAR_DS: if (!g_rc_skip) clear_ds((const p_clear_ds *)a); break;
        case OP_VIEWPORTS: { const p_viewports *v = (const p_viewports *)a; for (int i = 0; i < v->n && v->first + i < 16; i++) memcpy(S.vp[v->first + i], v->v[i], 16); break; }
        case OP_SCISSOR: S.sc = *(const p_scissor *)a; S.sc_set = 1; break;
        case OP_DEPTH_RANGE: { const p_depth_range *d = (const p_depth_range *)a; S.dnear = d->n; S.dfar = d->f; break; }
        case OP_BIND_PROGRAM: bind_program((const p_bind_program *)a); break;
        case OP_BIND_STATE: { const p_bind_state *s = (const p_bind_state *)a;
            switch (s->kind) {
            case 0: { blend_st b; memcpy(&b, s->bytes, sizeof b); if (b.target < 8) S.blend[b.target] = b; break; }
            case 1: memcpy(&S.color, s->bytes, sizeof S.color); break;
            case 2: memcpy(&S.chmask, s->bytes, sizeof S.chmask); break;
            case 3: memcpy(&S.ms, s->bytes, sizeof S.ms); break;
            case 4: memcpy(&S.poly, s->bytes, sizeof S.poly); break;
            case 5: memcpy(&S.ds, s->bytes, sizeof S.ds); break;
            }
            break; }
        case OP_BIND_VATTR: { const p_bind_vattr *v = (const p_bind_vattr *)a; memset(S.vattr, 0, sizeof S.vattr); for (int i = 0; i < v->n && v->first + i < 32; i++) S.vattr[v->first + i] = v->st[i]; break; }   // (up to 32: NVN2)
        case OP_BIND_VSTREAM: { const p_bind_vstream *v = (const p_bind_vstream *)a; for (int i = 0; i < v->n && v->first + i < 16; i++) S.vstream[v->first + i] = v->st[i]; break; }
        // the compact records (24 bytes with the header: one vertex buffer, one uniform buffer; games size command memory by GetCommandMemoryUsed, nvn_obj.h) read as the full ones
        case OP_BIND_VBUF: case OP_BIND_VBUF1: { p_bind_vbuf t; const p_bind_vbuf *v = vbuf_rec(h.op, a, &t); for (int i = 0; i < v->n && v->first + i < 16; i++) S.vbuf[v->first + i] = v->r[i]; break; }
        case OP_BIND_UBO: case OP_BIND_UBO_SMALL: { p_bind_buf t; const p_bind_buf *b = bind_buf_rec(h.op, a, &t);
            if (b->stage >= 0 && b->stage < 6 && b->index >= 0 && b->index < 16 && memcmp(&S.ubo[b->stage][b->index], b, sizeof *b)) { S.ubo[b->stage][b->index] = *b; g_ubo_gen[b->stage]++; } break; }
        case OP_BIND_SSBO: { const p_bind_buf *b = (const p_bind_buf *)a; if (b->stage >= 0 && b->stage < 6 && b->index >= 0 && b->index < 16 && memcmp(&S.ssbo[b->stage][b->index], b, sizeof *b)) { S.ssbo[b->stage][b->index] = *b; g_bind_gen[b->stage]++; } break; }
        case OP_BIND_TEX: case OP_BIND_SAMP: case OP_BIND_IMG: case OP_BIND_TEX1: bind_handles_rec(h.op, a); break;
        case OP_STENCIL: { const p_stencil *s = (const p_stencil *)a;
            for (int f = 0; f < 2; f++) if (s->face & (1 << f)) { if (s->which == 0) S.smask[f] = s->value_mask; else if (s->which == 1) S.swmask[f] = s->mask; else S.sref[f] = s->ref; }
            break; }
        case OP_POLY_OFFSET: S.pofs = *(const p_poly_offset *)a; break;
        case OP_BLEND_COLOR: memcpy(S.blendc, ((const p_blend_color *)a)->rgba, 16); break;
        case OP_DEPTH_CLAMP: S.depth_clamp = ((const p_depth_clamp *)a)->on; break;
        case OP_RAST_DISCARD: S.rast_discard = ((const p_rast_discard *)a)->on; break;
        case OP_PRIM_RESTART: S.prim_restart = ((const p_prim_restart *)a)->on; break;
        case OP_DRAW: { if (g_rc_skip) break; p_draw t; const p_draw *d = draw_rec(a, h.size, &t); if (d->mdi_cnt && d->prim != DM_PRIM_MESH_TASKS) { p = mdi_run(p); continue; } draw(d); break; }
        case OP_DRAW_IDX: { if (g_rc_skip) break; p_draw_idx t; const p_draw_idx *d = draw_idx_rec(a, h.size, &t); if (d->mdi_cnt) { p = mdi_run(p); continue; } draw_idx(d); break; }   // (mdi_run: a small multi-draw's records)
        case OP_MULTI_DRAW: { if (g_rc_skip) break; const p_multi_draw *m = (const p_multi_draw *)a; if (m->max > 0) multi_draw(m->prim, m->itype, m->ibuf, m->ind, m->cnt, 0, (uint32_t)m->max, (uint32_t)m->stride); break; }
        case OP_DRAW_TEX: if (!g_rc_skip) draw_tex((const p_draw_tex *)a); break;
        case OP_DLSS: dlss_eval((const p_dlss *)a); break;
        case OP_ALPHA_REF: S.alpha_ref = ((const p_alpha_ref *)a)->ref; break;   // (the alpha test's reference: prepare_draw)
        case OP_PUSH_CONST: { const p_push_const *b = (const p_push_const *)a; int c = b->compute != 0;
            if ((uint32_t)b->offset + b->size <= DMSH_PUSH_SIZE && memcmp(S.push[c] + b->offset, b->data, b->size)) {   // (hle_nvn.c drops a push past the block)
                memcpy(S.push[c] + b->offset, b->data, b->size); if (c) g_bind_gen[5]++; else for (int st = 0; st < 5; st++) g_bind_gen[st]++; }   // a stage set made from the old block is not kept (stage_set's memo)
            break; }
        case OP_DISPATCH: dispatch((const p_dispatch *)a); break;
        case OP_BARRIER:   // the pass ends at every barrier: kept across XC3's fragment- and primitive-order barriers (0x1 0x2 0x4) its characters drew torn in some frames
            if (g_x_listpasses) VKLOG("    barrier %#x", *(const int32_t *)a);
            vk_end_pass(); vk_img_barrier_all(vk_cmd()); break;
        case OP_SET_TEX_POOL: case OP_SET_SAMP_POOL: break;   // the handles were rebased when recorded (hle_nvn.c bind_handles)
        case OP_COPY_STRIDE: { const p_copy_stride *c = (const p_copy_stride *)a; if (c->image) S.image_stride = c->stride; else S.row_stride = c->stride; break; }
        case OP_COPY_B2T: g_pu.copies = pu_add(g_pu.copies, 1); copy_b2t((const p_copy_b2t *)a, 1); break;
        case OP_COPY_T2B: g_pu.copies = pu_add(g_pu.copies, 1); copy_b2t((const p_copy_b2t *)a, 0); break;
        case OP_COPY_T2T: g_pu.copies = pu_add(g_pu.copies, 1); copy_t2t((const p_copy_t2t *)a); break;
        case OP_CLEAR_TEX: g_pu.clears = pu_add(g_pu.clears, 1); clear_tex((const p_clear_tex *)a); break;
        case OP_RENDER_COND: render_cond((const p_render_cond *)a); break;   // recorded only with the profile key render_cond (hle_nvn.c)
        case OP_DRAW_TF: if (!g_rc_skip) draw_tf((const p_draw_tf *)a); break;
        case OP_TF_BIND: { const p_tf_bind *b = (const p_tf_bind *)a; if (b->index >= 0 && b->index < 4) g_tf[b->index] = *b; break; }
        case OP_TF_STATE: if (!V.has_xfb) VKRED_ONCE(80, "transform feedback: the device has none, its draws write nothing"); tf_state((const p_tf_state *)a); break;
        case OP_VP_SWIZZLE: { const p_vp_swizzle *b = (const p_vp_swizzle *)a; memcpy(S.vp_sw[b->first], b->sw, 4 * (size_t)b->n); break; }
        case OP_DISCARD: disc_op((const p_discard *)a); break;
        default: VKRED_ONCE(82, "unknown record op %u", h.op); break;
        }
        p += h.size;
    }
}
