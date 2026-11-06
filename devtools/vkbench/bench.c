// vkbench (devtools/vkbench): a CPU benchmark of the Android replay's per-draw path, on the Mac. The replay's own sources (dm/android/gfx_vk_*.c) run unchanged over a fake
// Vulkan driver (fake_vk.c) and the host's stand-ins (host_stubs.c); this file plays the game: it builds a world like Mario Kart World 1.8.0's in a race and replays its
// frames through mtl_sub_begin / mtl_exec / mtl_sub_commit, as hle_nvn.c's nvnQueueSubmitCommands does, timing the replay with its own counters.
// Where the numbers come from (each a VKB_* switch, the default the measured value):
// - the shaders are the game's own: the port's shader bundle (shaders.dmsv, VKB_BUNDLE), each program a vertex / fragment pair the shader split found (shader_pairs.tsv,
//   VKB_PAIRS), so every descriptor set has the bindings the game's sets have (census: vertex shaders ~5 uniform buffers and ~2 textures, fragment ~3 and ~5.5, up to 19);
// - per frame (the Mac's race1.frames.tsv, frames 40000-40600 of a race, 60 fps; the phone's run 13 log): 4 submissions, ~1,750 draws in ~120 passes (run 13: 1,600-1,800 a
//   frame, 118 passes), ~110 dispatches, ~130 barriers, ~2,150 CallCommands, program binds ~0.4 a draw, texture binds ~2.3 a draw, uniform buffer binds ~1.4 a draw, 87 texture
//   registrations (nvnTexturePoolRegisterTexture), no sampler registration;
// - memory: more than 4,096 pool entries (run 5 died at 4,096: "too many memory pools"), mesh data in pools of its own, ~7,000 images (run 13: "images 6961").
// Output: the replay's per-frame line as the phone logs it (gfx_vk_present.c), plus the fake driver's call counts; VKB_JSON=<file> appends one JSON object per run
#define _GNU_SOURCE
#include <math.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <time.h>
#include <unistd.h>
#include "gfx_vk.h"
#include <pthread/qos.h>

// ---- what the replay exports (gfx_vk_exec.c, gfx_vk_core.c, gfx_vk_tex.c) ----
extern struct { uint64_t draws, skip_prog, skip_rt, skip_pipe, passes, folded, dispatches, pipes_built, rt_kept, ns_exec, ns_desc, ns_pipe, max_pass_draws; } g_vkst;
extern uint64_t g_set_hits, g_set_new, g_ns_set_build, g_ns_set_hash, g_ns_set_new, g_set_memo, g_set_memo_bad, g_ns_submit;
void mtl_sub_begin(void); uint64_t mtl_sub_commit(void); void mtl_exec(const uint8_t *p); void mtl_init(void); void mtl_pool_add(const void *mem, uint64_t size); void vk_pool_set_flags(const void *mem, int flags);
void mtl_reg_tex(int tid, const tex_t *t, const view_t *v); void mtl_reg_samp(int sid, const samp_par *sp); void vk_tex_sweep(void); void vk_pass_frame(void); int mtl_wait(uint64_t fence, uint64_t timeout_ns);
void vk_shaders_open(const char *path); void mtl_set_frame(uint64_t f); void mtl_pool_del(const void *mem); void mtl_alias_drop_pool(const void *pool);
extern _Atomic uint64_t g_fk_calls, g_fk_cmd_bytes, g_fk_desc_writes, g_fk_sets, g_fk_updates, g_fk_binds, g_fk_draws, g_fk_pipes, g_fk_pushdesc;
extern uint8_t *g_vkb_alias_base; extern size_t g_vkb_alias_size; extern _Atomic uint64_t g_vkb_dirty_now; extern _Atomic int g_vkb_dirty_edge;
void vkb_valias_add(const void *pool, const uint8_t *res, uint64_t len, uint64_t size, const uint8_t *phys); void vkb_vpool_add(uint8_t *base, uint64_t len);
typedef struct { prog_t p; uint64_t key[6]; } vkb_prog;
void nvn_tex_layout(const nvn_texdesc *d, nvn_texlayout *o);

uint64_t g_cz[32] __attribute__((weak));   // (a census build of the replay counts its calls here: vkbench, analysis only)
static int envi(const char *k, int d) { const char *e = getenv(k); return e && *e ? atoi(e) : d; }
static double envf(const char *k, double d) { const char *e = getenv(k); return e && *e ? atof(e) : d; }
static uint64_t cpu_now(void) { struct timespec t; clock_gettime(CLOCK_THREAD_CPUTIME_ID, &t); return (uint64_t)t.tv_sec * 1000000000ull + (uint64_t)t.tv_nsec; }   // this thread's CPU time: what preemption by other work leaves out
// the replay thread's instructions retired and cycles (XNU's per-thread counters, thread_selfcounts: what the Mac's other load and its core and clock changes leave alone, the
// instructions at least; the cycles still differ between a performance and an efficiency core)
extern int thread_selfcounts(int type, void *buf, size_t nbytes);
static void ic_now(uint64_t c[2]) { c[0] = c[1] = 0; thread_selfcounts(1, c, sizeof(uint64_t) * 2); }
static int cmp_d(const void *a, const void *b) { double x = *(const double *)a, y = *(const double *)b; return x < y ? -1 : x > y; }
static uint64_t vkb_now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return (uint64_t)t.tv_sec * 1000000000ull + (uint64_t)t.tv_nsec; }
static uint64_t g_rng = 0x9E3779B97F4A7C15ull;
static uint32_t rnd(void) { g_rng ^= g_rng << 13; g_rng ^= g_rng >> 7; g_rng ^= g_rng << 17; return (uint32_t)(g_rng >> 16); }
static uint32_t rndn(uint32_t n) { return n ? rnd() % n : 0; }
static void fatal(const char *m) { fprintf(stderr, "vkbench: %s\n", m); exit(2); }

// ---- the bundle's reflection (gfx_vk.h dmsv_bind, gfx_vk_shader.c bent) ----
typedef struct __attribute__((packed)) { uint64_t key; uint32_t stage, spv_off, spv_len, refl_off, refl_len, pad; } bent_t;
typedef struct { uint64_t key; int stage, nbind; uint32_t local[3]; const dmsv_bind *b; int ntex, ncb, nsb, nimg; } shinfo;
static shinfo *g_sh; static uint32_t g_nsh; static const uint8_t *g_bundle;
static shinfo *sh_find(uint64_t key) { uint32_t lo = 0, hi = g_nsh; while (lo < hi) { uint32_t m = (lo + hi) / 2; if (g_sh[m].key < key) lo = m + 1; else hi = m; } return lo < g_nsh && g_sh[lo].key == key ? &g_sh[lo] : NULL; }
static void bundle_load(const char *path) {
    FILE *f = fopen(path, "rb"); if (!f) fatal("no shader bundle (VKB_BUNDLE)"); fseek(f, 0, SEEK_END); long n = ftell(f); fclose(f);
    int fd = open(path, 0); g_bundle = mmap(NULL, (size_t)n, PROT_READ, MAP_PRIVATE, fd, 0); close(fd); if (g_bundle == MAP_FAILED) fatal("bundle mmap");
    uint32_t ne; memcpy(&ne, g_bundle + 8, 4); g_sh = calloc(ne, sizeof *g_sh); g_nsh = ne;
    for (uint32_t i = 0; i < ne; i++) { bent_t e; memcpy(&e, g_bundle + 12 + 32 * (size_t)i, 32); shinfo *s = &g_sh[i]; s->key = e.key; s->stage = (int)e.stage;
        const uint8_t *r = g_bundle + e.refl_off; uint32_t h[5]; memcpy(h, r, 20); s->nbind = (int)h[0]; memcpy(s->local, h + 2, 12); s->b = (const dmsv_bind *)(r + 20);
        for (int k = 0; k < s->nbind; k++) { int kd = s->b[k].kind; s->ntex += kd == RK_TEX; s->ncb += kd == RK_CBUF; s->nsb += kd == RK_SBUF; s->nimg += kd == RK_IMG; } }
    // (the bundle is sorted by key: gfx_vk_shader.c find)
}

// ---- guest memory: one arena, pools carved from it at 16 KB (the Mac's page; zero copy needs whole pages) ----
static uint8_t *g_arena; static uint64_t g_arena_size, g_arena_used;
typedef struct { uint8_t *mem; uint64_t size, used; pool_t *obj; } vpool;
static uint8_t *carve(uint64_t n) { n = (n + 0x3fff) & ~0x3fffull; if (g_arena_used + n > g_arena_size) fatal("arena full (VKB_ARENA_MB)"); uint8_t *p = g_arena + g_arena_used; g_arena_used += n; return p; }
static vpool pool_new(uint64_t size, int flags) {
    vpool p; memset(&p, 0, sizeof p); p.size = (size + 0x3fff) & ~0x3fffull; p.mem = carve(p.size);
    p.obj = calloc(1, sizeof *p.obj); p.obj->magic = MAGIC('P', 'O', 'O', 'L'); p.obj->flags = (uint32_t)flags; p.obj->mem = p.mem; p.obj->size = p.size;
    mtl_pool_add(p.mem, p.size); vk_pool_set_flags(p.mem, flags); return p;
}
static uint64_t sub(vpool *p, uint64_t n, uint64_t al) { uint64_t o = (p->used + al - 1) & ~(al - 1); if (o + n > p->size) o = 0; p->used = o + n; return (uint64_t)(uintptr_t)(p->mem + o); }

// ---- textures, samplers, programs ----
enum { NVN_RGBA8 = 0x25, NVN_RGBA16F = 0x29, NVN_R11G11B10F = 0x3F, NVN_D24S8 = 0x35, NVN_D32F = 0x34, NVN_DXT1 = 0x42, NVN_DXT5 = 0x44, NVN_BPTC = 0x4D, NVN_RGTC2 = 0x4B, NVN_ASTC4 = 0x79 };
static tex_t *g_tex; static view_t *g_view; static int *g_has_view; static int g_ntex;   // registered sampled textures, IDs 1..g_ntex
static tex_t g_rt[64]; static int g_nrt;                                            // render targets (not registered; SetRenderTargets names them)
static int g_nsamp;
static vkb_prog *g_prog; static int g_nprog; static vkb_prog *g_cprog; static int g_ncprog;   // graphics (vertex + fragment), compute
static uint64_t g_serial = 1;
static tex_t mk_tex(vpool *p, int target, int fmt, uint32_t w, uint32_t h, uint32_t d, uint32_t levels, int flags, int packaged) {
    tex_t t; memset(&t, 0, sizeof t); t.magic = MAGIC('T', 'E', 'X', 0); t.d.target = target; t.d.format = fmt; t.d.flags = flags; t.d.width = w; t.d.height = h; t.d.depth = d; t.d.levels = levels;
    nvn_tex_layout(&t.d, &t.L); uint64_t a = sub(p, t.L.size, t.L.align ? t.L.align : 512); t.mem = (uint8_t *)(uintptr_t)a; t.pool = p->obj; t.off = (int64_t)(a - (uint64_t)(uintptr_t)p->mem);
    t.has_layout = (uint8_t)packaged; t.serial = g_serial++; return t;
}

// ---- the record stream (nvn_obj.h): hle_nvn.c's encoding ----
typedef struct { uint8_t *p; size_t n, cap; } recbuf;
static void *rec(recbuf *r, uint32_t op, size_t n) {
    size_t sz = (8 + n + 7) & ~(size_t)7; if (r->n + sz + 16 > r->cap) { r->cap = r->cap ? r->cap * 2 : 1 << 20; r->p = realloc(r->p, r->cap); }
    op_hdr h = {op, (uint32_t)sz}; memcpy(r->p + r->n, &h, 8); void *a = r->p + r->n + 8; memset(a, 0, sz - 8); r->n += sz; return a;
}
static void r_end(recbuf *r) { rec(r, OP_END, 0); }
static void r_call(recbuf *r, const uint8_t *seg) { uint64_t t = (uint64_t)(uintptr_t)seg; memcpy(rec(r, OP_CALL, 8), &t, 8); }
static void r_prog(recbuf *r, const vkb_prog *p, int stages) { p_bind_program *b = rec(r, OP_BIND_PROGRAM, sizeof *b); b->prog = &p->p; b->stages = stages; }
static void r_ubo(recbuf *r, int st, int idx, uint64_t a, uint64_t n) { p_bind_buf *b = rec(r, OP_BIND_UBO, sizeof *b); *b = (p_bind_buf){st, idx, a, n}; }
static void r_ssbo(recbuf *r, int st, int idx, uint64_t a, uint64_t n) { p_bind_buf *b = rec(r, OP_BIND_SSBO, sizeof *b); *b = (p_bind_buf){st, idx, a, n}; }
static void r_tex(recbuf *r, int st, int slot, uint32_t tid, uint32_t sid) { p_bind_tex1 *b = rec(r, OP_BIND_TEX1, sizeof *b); b->stage = st | BIND_TEX1_NOBASE; b->slot = slot; b->h = (uint64_t)(tid | sid << 20); }
static void r_img(recbuf *r, int st, int slot, uint32_t tid) { p_bind_handles *b = rec(r, OP_BIND_IMG, sizeof *b); b->stage = st; b->first = slot; b->n = 1; b->h[0] = tid; }
static void r_state(recbuf *r, int kind, const void *s, size_t n) { p_bind_state *b = rec(r, OP_BIND_STATE, sizeof *b); b->kind = kind; memcpy(b->bytes, s, n); }
static void r_vp(recbuf *r, float w, float h) { p_viewports *v = rec(r, OP_VIEWPORTS, 8 + 24); v->first = 0; v->n = 1; float x[6] = {0, 0, w, h, 0, 1}; memcpy(v->v[0], x, 24); }
static void r_sc(recbuf *r, int w, int h) { p_scissor *s = rec(r, OP_SCISSOR, sizeof *s); *s = (p_scissor){0, 0, w, h}; }
static void r_rt(recbuf *r, int n, const int *col, int dep) {
    p_set_rt *s = rec(r, OP_SET_RT, sizeof *s); s->n = n; for (int i = 0; i < n; i++) s->color[i] = col[i] >= 0 ? &g_rt[col[i]] : NULL; s->depth = dep >= 0 ? &g_rt[dep] : NULL; }
static void r_clear(recbuf *r, int i) { p_clear_color *c = rec(r, OP_CLEAR_COLOR, sizeof *c); c->index = i; c->mask = 0xf; }
static void r_clear_ds(recbuf *r) { p_clear_ds *c = rec(r, OP_CLEAR_DS, sizeof *c); c->depth = 1; c->depth_mask = 1; c->stencil = 0; c->stencil_mask = 0xff; }
static void r_barrier(recbuf *r) { int32_t *b = rec(r, OP_BARRIER, 8); *b = 0x1f; }
static void r_vattr(recbuf *r, const uint32_t *st, int n) { p_bind_vattr *v = rec(r, OP_BIND_VATTR, sizeof *v); v->first = 0; v->n = n; memcpy(v->st, st, 4 * (size_t)n); }
static void r_vstream(recbuf *r, uint32_t s0, uint32_t s1) { p_bind_vstream *v = rec(r, OP_BIND_VSTREAM, sizeof *v); v->first = 0; v->n = 2; v->st[0].stride = s0; v->st[1].stride = s1; }
static void r_vbuf(recbuf *r, int idx, uint64_t a, uint32_t n) { p_bind_vbuf1 *v = rec(r, OP_BIND_VBUF1, sizeof *v); v->index = (uint32_t)idx; v->size = n; v->addr = a; }
static void r_drawi(recbuf *r, uint64_t ib, int count, int inst) { p_draw_idx *d = rec(r, OP_DRAW_IDX, 40); d->prim = 4; d->itype = 1; d->count = count; d->instances = inst; d->ibuf = ib; }
static void r_draw(recbuf *r, int count) { p_draw *d = rec(r, OP_DRAW, 20); d->prim = 4; d->count = count; d->instances = 1; }
static void r_dispatch(recbuf *r, int x, int y, int z) { p_dispatch *d = rec(r, OP_DISPATCH, sizeof *d); d->x = x; d->y = y; d->z = z; }

// ---- the world ----
typedef struct { int prog, sprog; uint32_t tid[32], sid[32]; uint64_t ubo; int blend; } material;   // sprog: its shadow casters' program (depth only: light)
typedef struct { int mat; uint64_t vb, ib, ubo; int count; uint32_t vb_size; uint8_t *seg, *sseg; } object;
static material *g_mat; static int g_nmat; static object *g_obj; static int g_nobj;
static vpool g_pobj, g_pmat, g_pview, g_pring, g_pssbo[16], *g_pmesh, g_ptex[512], g_prtp; static int g_nptex, g_npmesh;
static uint64_t g_view_ubo[16], g_ssbo_addr[16]; static int g_virt_pct;
static uint8_t *g_virt_base; static uint64_t g_virt_len;   // an aliased virtual pool's own range, translated to the object constants (nvn_virt_xlat)
static uint64_t virt(uint64_t a) {   // an address the game gives through a virtual pool's range instead (VKB_VIRT_PCT of the per-object ones)
    if (!g_virt_base || (uint32_t)rndn(100) >= (uint32_t)g_virt_pct) return a;
    uint64_t lo = (uint64_t)(uintptr_t)g_pobj.mem; if (a < lo || a >= lo + g_pobj.size) return a; return (uint64_t)(uintptr_t)g_virt_base + (a - lo); }
static int g_cat[4][4096], g_ncat[4];   // programs by fragment texture count: 0 none, 1 light (1-2), 2 medium (3-7), 3 heavy (8+)
static int pick_prog(int cat) { for (int c = cat; c >= 0; c--) if (g_ncat[c]) return g_cat[c][rndn((uint32_t)g_ncat[c])]; for (int c = cat; c < 4; c++) if (g_ncat[c]) return g_cat[c][rndn((uint32_t)g_ncat[c])]; fatal("no programs"); return 0; }
static const shinfo *psh(int prog, int st) { return sh_find(g_prog[prog].key[st]); }

static void world(void) {
    // pools: the race's buffers, then filler pools to the measured entry count (VKB_POOLS)
    int npools = envi("VKB_POOLS", 4600), nobj = envi("VKB_OBJECTS", 720), nmat = envi("VKB_MATERIALS", 260);
    g_pobj = pool_new((uint64_t)nobj * 512 + (1 << 16), 0x22); g_pmat = pool_new((uint64_t)nmat * 512 + (1 << 16), 0x22); g_pview = pool_new(1 << 16, 0x22); g_pring = pool_new(4 << 20, 0x22);
    for (int i = 0; i < 16; i++) g_pssbo[i] = pool_new(1 << 20, 0x22);
    g_npmesh = nobj; g_pmesh = calloc((size_t)g_npmesh, sizeof *g_pmesh);
    for (int i = 0; i < g_npmesh; i++) g_pmesh[i] = pool_new(48 << 10, 0x22);   // a model's vertices and indices in a pool of its own
    g_nptex = envi("VKB_TEX_POOLS", 400); if (g_nptex > 512) g_nptex = 512;
    for (int i = 0; i < g_nptex; i++) g_ptex[i] = pool_new(1 << 20, 0x21);   // texture archives (CPU_NO_ACCESS | GPU_CACHED)
    g_prtp = pool_new(96 << 20, 0x29);
    int made = 4 + 16 + g_npmesh + g_nptex + 1; for (; made < npools; made++) pool_new(16 << 10, 0x22);   // (the rest of the game's pools: in the table, never drawn from)
    for (int i = 0; i < 16; i++) { g_view_ubo[i] = sub(&g_pview, 1024, 256); g_ssbo_addr[i] = (uint64_t)(uintptr_t)g_pssbo[i].mem; }
    // an aliased virtual pool over the object constants (Android: its own range has no CPU alias, addresses in it are translated: nvn_virt_xlat), and the race's other virtual pools in the table
    g_virt_pct = envi("VKB_VIRT_PCT", 0);
    { int nva = envi("VKB_VALIASES", 65), nvs = envi("VKB_VSPARSE", 30); uint64_t span = 1ull << 30; uint8_t *res = mmap(NULL, span * (uint64_t)(nva + nvs + 1), PROT_NONE, MAP_ANON | MAP_PRIVATE, -1, 0);
      if (res == MAP_FAILED) fatal("virtual ranges");
      for (int i = 0; i < nva; i++) { static pool_t vp[128]; vkb_valias_add(&vp[i % 128], res + span * (uint64_t)i, span, g_pobj.size, g_pobj.mem); }   // (the others name the same memory: the scan's length counts)
      g_virt_base = res; g_virt_len = g_pobj.size;
      for (int i = 0; i < nvs; i++) vkb_vpool_add(res + span * (uint64_t)(nva + i), span); }
    // render targets: 720p G-buffer and scene, shadow cascades, half-resolution chain, post targets (MKW at DM_GRAPHICS_SCALE=2: 1280x720)
    static const int RTF[] = {NVN_RGBA8, NVN_RGBA8, NVN_RGBA16F, NVN_R11G11B10F};
    for (int i = 0; i < 4; i++) g_rt[g_nrt++] = mk_tex(&g_prtp, 1, RTF[i], 1280, 720, 1, 1, 0x8, 0);            // 0-3 G-buffer
    g_rt[g_nrt++] = mk_tex(&g_prtp, 1, NVN_D24S8, 1280, 720, 1, 1, 0x8, 0);                                     // 4 depth
    for (int i = 0; i < 3; i++) g_rt[g_nrt++] = mk_tex(&g_prtp, 1, NVN_D32F, 2048, 2048, 1, 1, 0x8, 0);          // 5-7 shadow cascades
    g_rt[g_nrt++] = mk_tex(&g_prtp, 1, NVN_RGBA16F, 1280, 720, 1, 1, 0x8, 0);                                   // 8 scene HDR
    for (int i = 0; i < 12; i++) g_rt[g_nrt++] = mk_tex(&g_prtp, 1, i & 1 ? NVN_R11G11B10F : NVN_RGBA16F, 640 >> (i / 2), 360 >> (i / 2), 1, 1, 0x8, 0);   // 9-20 post chain
    g_rt[g_nrt++] = mk_tex(&g_prtp, 1, NVN_RGBA8, 1280, 720, 1, 1, 0x8, 0);                                     // 21 UI / final
    for (int i = 0; i < 8; i++) g_rt[g_nrt++] = mk_tex(&g_prtp, 1, NVN_RGBA16F, 256, 256, 1, 1, 0x8, 0);        // 22-29 probes and small targets
    // sampled textures: BC / ASTC / RGBA8 of 64 to 1024 texels, mips, a third with a texture view; registered at IDs 1.. as the game's pool holds them (31,778 registrations at frame 0)
    g_ntex = envi("VKB_TEXTURES", 6000); g_tex = calloc((size_t)g_ntex + 1, sizeof *g_tex); g_view = calloc((size_t)g_ntex + 1, sizeof *g_view); g_has_view = calloc((size_t)g_ntex + 1, sizeof *g_has_view);
    static const int TF[] = {NVN_DXT1, NVN_DXT5, NVN_BPTC, NVN_RGTC2, NVN_ASTC4, NVN_RGBA8};
    for (int i = 1; i <= g_ntex; i++) {
        uint32_t w = 64u << rndn(4), h = w >> rndn(2); int f = TF[rndn(6)], tg = rndn(20) == 0 ? 8 : 1; uint32_t lv = 1; while ((w >> lv) >= 4 && lv < 8) lv++;
        g_tex[i] = mk_tex(&g_ptex[rndn((uint32_t)g_nptex)], tg, f, w, tg == 8 ? w : h, tg == 8 ? 6 : 1, lv, 0, 1);
        if (rndn(3) == 0) { g_has_view[i] = 1; g_view[i].set_levels = 1; g_view[i].base_level = 0; g_view[i].nlevels = (uint8_t)lv; }
        mtl_reg_tex(i, &g_tex[i], g_has_view[i] ? &g_view[i] : NULL);
    }
    // render targets are textures the later passes sample: registered after the sampled ones (IDs g_ntex + 1 ..)
    for (int i = 0; i < g_nrt; i++) mtl_reg_tex(g_ntex + 1 + i, &g_rt[i], NULL);
    // samplers (181 registered at frame 0, none per frame in the race)
    g_nsamp = 181; for (int i = 1; i <= g_nsamp; i++) { samp_par s; memset(&s, 0, sizeof s); s.minf = (uint8_t)(i % 3 ? 5 : 1); s.magf = 1; s.ws = s.wt = s.wr = (uint8_t)(i % 4 == 0 ? 2 : 1); s.lod_max = 1000; s.aniso = i % 5 ? 8 : 1; mtl_reg_samp(i, &s); }
    // programs: vertex / fragment pairs of the game (shader_pairs.tsv) whose shaders are in the bundle, by how many textures the fragment shader reads
    { const char *pp = getenv("VKB_PAIRS"); FILE *f = pp ? fopen(pp, "r") : NULL; if (!f) fatal("no shader pairs (VKB_PAIRS)");
      int cap = 8192; g_prog = calloc((size_t)cap, sizeof *g_prog); char line[256];
      while (fgets(line, sizeof line, f) && g_nprog < cap) { if (line[0] == '#') continue; uint64_t vk, fk; if (sscanf(line, "%llx %llx", (unsigned long long *)&vk, (unsigned long long *)&fk) != 2) continue;
          shinfo *v = sh_find(vk), *fs = sh_find(fk); if (!v || !fs || v->stage != 0 || fs->stage != 1) continue;
          vkb_prog *p = &g_prog[g_nprog]; p->p.magic = MAGIC('P', 'R', 'O', 'G'); p->p.nstage = 2; p->p.serial = g_serial++; p->key[0] = vk; p->key[1] = fk;
          int c = fs->ntex == 0 ? 0 : fs->ntex <= 2 ? 1 : fs->ntex <= 7 ? 2 : 3; if (g_ncat[c] < 4096) g_cat[c][g_ncat[c]++] = g_nprog; g_nprog++; }
      fclose(f); }
    // compute programs: the bundle's compute shaders (stage 5)
    g_cprog = calloc(1024, sizeof *g_cprog); for (uint32_t i = 0; i < g_nsh && g_ncprog < 1024; i++) if (g_sh[i].stage == 5) { vkb_prog *p = &g_cprog[g_ncprog++]; p->p.magic = MAGIC('P', 'R', 'O', 'G'); p->p.nstage = 1; p->p.serial = g_serial++; p->key[0] = g_sh[i].key; }
    // materials: a program (G-buffer: medium and heavy) and a texture for each of its fragment slots; objects: a mesh in its own pool, a material, a constant buffer
    g_nmat = nmat; g_mat = calloc((size_t)nmat, sizeof *g_mat);
    int nshp = envi("VKB_SHADOW_PROGS", 12), *shp = calloc((size_t)nshp, sizeof *shp); for (int i = 0; i < nshp; i++) shp[i] = pick_prog(i % 3 ? 1 : 0);   // the few depth-only programs casters share
    for (int i = 0; i < nmat; i++) { material *m = &g_mat[i]; m->prog = pick_prog(i % 5 == 0 ? 2 : 3); m->sprog = shp[rndn((uint32_t)nshp)]; m->ubo = sub(&g_pmat, 256, 256); m->blend = 0;
        for (int k = 0; k < 32; k++) { m->tid[k] = 1 + rndn((uint32_t)g_ntex); m->sid[k] = 1 + rndn((uint32_t)g_nsamp); } }
    g_nobj = nobj; g_obj = calloc((size_t)nobj, sizeof *g_obj);
    for (int i = 0; i < nobj; i++) { object *o = &g_obj[i]; o->mat = rndn((uint32_t)nmat); vpool *mp = &g_pmesh[i]; o->vb_size = 32 << 10; o->vb = sub(mp, o->vb_size, 256); o->ib = sub(mp, 12 << 10, 256); o->ubo = sub(&g_pobj, 256, 256); o->count = 300 + (int)rndn(3000); }
}

// ---- a frame's records: 4 submissions, the passes of a race frame ----
typedef struct { int prog, mat; uint32_t tex[2][32]; uint64_t ubo[2][16], ssbo[2][16]; uint32_t vattr_sig; int bound_vb_obj; int blend, ds; } bstate;   // what the stream has bound (a game's own state tracking)
static recbuf g_sub[4]; static recbuf g_segs;   // the submissions, and the objects' prerecorded draw segments (CallCommands)
static bstate B;
static void bind_prog(recbuf *r, int prog) { if (B.prog == prog) return; r_prog(r, &g_prog[prog], 0x3); B.prog = prog; }
static void bind_view(recbuf *r, int prog) {   // the per-view buffers the program's shaders read: uniform slots outside the object's and the material's, the light lists
    for (int st = 0; st < 2; st++) { const shinfo *s = psh(prog, st); if (!s) continue;
        for (int k = 0; k < s->nbind; k++) { const dmsv_bind *b = &s->b[k]; if (b->reg < 0 || b->reg >= 16) continue;
            if (b->kind == RK_CBUF && !(st == 0 && (b->reg == 5 || b->reg == 7)) && !(st == 1 && (b->reg == 8 || b->reg == 10))) { uint64_t a = g_view_ubo[b->reg]; if (B.ubo[st][b->reg] != a) { r_ubo(r, st, b->reg, a, 1024); B.ubo[st][b->reg] = a; } }
            if (b->kind == RK_SBUF) { uint64_t a = g_ssbo_addr[b->reg]; if (B.ssbo[st][b->reg] != a) { r_ssbo(r, st, b->reg, a, 1 << 20); B.ssbo[st][b->reg] = a; } } } }
}
static void bind_mat(recbuf *r, const material *m, int prog, int vtex) {   // the material's textures for the fragment slots the program reads (and the vertex shader's), its constant buffer
    for (int st = 0; st < 2; st++) { const shinfo *s = psh(prog, st); if (!s) continue;
        for (int k = 0; k < s->nbind; k++) { const dmsv_bind *b = &s->b[k]; if (b->kind == RK_TEX && b->reg >= 0 && b->reg < 32 && (st == 1 || vtex)) { uint32_t h = m->tid[b->reg] | m->sid[b->reg] << 20;
                if (B.tex[st][b->reg] != h) { r_tex(r, st, b->reg, m->tid[b->reg], m->sid[b->reg]); B.tex[st][b->reg] = h; } }
            if (st == 1 && b->kind == RK_CBUF && (b->reg == 8 || b->reg == 10) && B.ubo[1][b->reg] != m->ubo) { r_ubo(r, 1, b->reg, m->ubo, 256); B.ubo[1][b->reg] = m->ubo; } } }
}
static void bind_vertex(recbuf *r, int prog) {   // the vertex format the vertex shader reads (its input locations), when it changes
    const shinfo *v = psh(prog, 0); uint32_t in = v ? v->local[0] : 1, sig = in; if (B.vattr_sig == sig) return; B.vattr_sig = sig;
    uint32_t st[32]; int n = 0; for (int loc = 0; loc < 32; loc++) { va_st a; memset(&a, 0, sizeof a); if (in >> loc & 1) { a.fmt = loc == 0 ? 0x22 : (loc & 1) ? 0x29 : 0x11; a.stream = loc < 3 ? 0 : 1; a.offset = loc < 3 ? 12 * (uint32_t)loc : 8 * (uint32_t)(loc - 3); } memcpy(&st[loc], &a, 4); if (in >> loc & 1) n = loc + 1; }
    r_vattr(r, st, n ? n : 1); r_vstream(r, 36, 64);
}
static size_t obj_segment(object *o, int prog) {   // the object's prerecorded draw: its constant buffer, its vertices, its draw (the game calls it from each pass)
    size_t at = g_segs.n; const shinfo *v = psh(prog, 0);
    if (v) for (int k = 0; k < v->nbind; k++) if (v->b[k].kind == RK_CBUF && (v->b[k].reg == 5 || v->b[k].reg == 7)) r_ubo(&g_segs, 0, v->b[k].reg, virt(o->ubo), 256);
    r_vbuf(&g_segs, 0, o->vb, o->vb_size / 2); r_vbuf(&g_segs, 1, o->vb + o->vb_size / 2, o->vb_size / 2);
    r_drawi(&g_segs, o->ib, o->count, 1); r_end(&g_segs);
    return at;   // (an offset until the segment buffer stops moving: fixed up in frame_build)
}
static int g_frame_draws, g_frame_passes, g_frame_disp, g_frame_calls, g_frame_progbinds, g_frame_texbinds, g_frame_ubobinds;
static void pass_begin(recbuf *r, int n, const int *col, int dep, int clear, float w, float h) {
    r_rt(r, n, col, dep); r_vp(r, w, h); r_sc(r, (int)w, (int)h); if (clear) { for (int i = 0; i < n; i++) r_clear(r, i); if (dep >= 0) r_clear_ds(r); }
    memset(&B, 0, sizeof B); B.prog = -1; B.mat = -1; g_frame_passes++;   // (the game rebinds at a pass's start as an engine does; the replay's caches start over at each pass too)
}
static void compute(recbuf *r, int n) {   // n dispatches: a compute program, its buffers, textures and images, between barriers
    for (int i = 0; i < n; i++) { const vkb_prog *p = &g_cprog[rndn((uint32_t)g_ncprog)]; const shinfo *s = sh_find(p->key[0]); if (!s) continue;
        r_barrier(r); r_prog(r, p, 0x20);
        for (int k = 0; k < s->nbind; k++) { const dmsv_bind *b = &s->b[k]; if (b->reg < 0) continue;
            if ((b->kind == RK_CBUF || b->kind == RK_UBO_PTR) && b->reg < 16) r_ubo(r, 5, b->reg, g_view_ubo[b->reg], 1024);
            if (b->kind == RK_SBUF && b->reg < 16) r_ssbo(r, 5, b->reg, g_ssbo_addr[b->reg], 1 << 20);
            if (b->kind == RK_TEX && b->reg < 32) r_tex(r, 5, b->reg, 1 + rndn((uint32_t)g_ntex), 1 + rndn((uint32_t)g_nsamp));
            if (b->kind == RK_IMG && b->reg < 8) r_img(r, 5, b->reg, (uint32_t)(1 + rndn((uint32_t)g_ntex))); }
        r_dispatch(r, 16 + (int)rndn(64), 8 + (int)rndn(32), 1); g_frame_disp++; }
    r_barrier(r);
}
static void draw_objects(recbuf *r, const int *ids, int n, int vtex, int shadow) {   // objects sorted by material, each called from its prerecorded segment (a shadow pass: the casters' program)
    for (int i = 0; i < n; i++) { object *o = &g_obj[ids[i]]; const material *m = &g_mat[o->mat]; int prog = shadow ? m->sprog : m->prog;
        int newp = B.prog != prog; bind_prog(r, prog); if (newp) { bind_view(r, prog); bind_vertex(r, prog); g_frame_progbinds++; }
        if (B.mat != o->mat || newp) { bind_mat(r, m, prog, vtex); B.mat = o->mat; }
        r_call(r, shadow ? o->sseg : o->seg); g_frame_calls++; g_frame_draws++; }
}
static int cmp_mat(const void *a, const void *b) { const object *x = &g_obj[*(const int *)a], *y = &g_obj[*(const int *)b]; return x->mat != y->mat ? x->mat - y->mat : *(const int *)a - *(const int *)b; }
static int g_set_ui = 48, g_set_fx = 96;   // the UI's atlas pages and the effects' sheets: the textures quads draw with (VKB_UI_TEX, VKB_FX_TEX)
static uint32_t quad_tex(int cat, int k) {
    if (cat == 1 && k == 0 && rndn(3) == 0) return (uint32_t)(g_ntex + 1 + 8 + rndn(14));   // a render target (the post chain, the scene)
    return 1 + (cat == 1 ? rndn((uint32_t)g_set_ui) : 64 + rndn((uint32_t)g_set_fx)); }
static void quads(recbuf *r, int n, int cat, int ring) {   // n small draws (effects, UI): a program run every few draws, a texture or two each, a constant buffer from the ring
    int prog = -1;
    for (int i = 0; i < n; i++) {
        if (prog < 0 || rndn(4) == 0) { prog = pick_prog(cat); r_prog(r, &g_prog[prog], 0x3); B.prog = prog; bind_view(r, prog); bind_vertex(r, prog); g_frame_progbinds++; }
        const shinfo *fs = psh(prog, 1);
        if (fs) for (int k = 0; k < fs->nbind; k++) { const dmsv_bind *b = &fs->b[k]; if (b->kind == RK_TEX && b->reg >= 0 && b->reg < 32 && (k == 0 || rndn(2))) { uint32_t t = quad_tex(cat, k), s = 1 + (uint32_t)(b->reg % 8);
            if (B.tex[1][b->reg] != (t | s << 20)) { r_tex(r, 1, b->reg, t, s); B.tex[1][b->reg] = t | s << 20; g_frame_texbinds++; } } }
        const shinfo *vs = psh(prog, 0);
        if (vs && ring) for (int k = 0; k < vs->nbind; k++) if (vs->b[k].kind == RK_CBUF && (vs->b[k].reg == 5 || vs->b[k].reg == 7)) { r_ubo(r, 0, vs->b[k].reg, sub(&g_pring, 256, 256), 256); g_frame_ubobinds++; }
        r_vbuf(r, 0, sub(&g_pring, 4096, 256), 4096); r_draw(r, 6); g_frame_draws++; }
}
static void frame_build(void) {
    int *ids = malloc(sizeof(int) * (size_t)g_nobj); for (int i = 0; i < g_nobj; i++) ids[i] = i;
    size_t *so = calloc((size_t)g_nobj * 2, sizeof *so);
    for (int i = 0; i < g_nobj; i++) { so[2 * i] = obj_segment(&g_obj[i], g_mat[g_obj[i].mat].prog); so[2 * i + 1] = obj_segment(&g_obj[i], g_mat[g_obj[i].mat].sprog); }
    for (int i = 0; i < g_nobj; i++) { g_obj[i].seg = g_segs.p + so[2 * i]; g_obj[i].sseg = g_segs.p + so[2 * i + 1]; }   // (the segment buffer no longer moves)
    free(so);
    int shadow = envi("VKB_SHADOW_DRAWS", 140), gbuf = envi("VKB_GBUF_DRAWS", 640), fx = envi("VKB_FX_DRAWS", 210), ui = envi("VKB_UI_DRAWS", 250), post = envi("VKB_POST_PASSES", 40), small = envi("VKB_SMALL_PASSES", 70);
    int disp = envi("VKB_DISPATCHES", 110);
    // submission 0: compute (culling, skinning), the three shadow cascades
    recbuf *r = &g_sub[0]; compute(r, disp / 4);
    // the cascades draw mostly the same casters (each culls a few of the others in): their draws repeat the sets of the first cascade's (the set cache's hits within a submission)
    int *cast = malloc(sizeof(int) * (size_t)shadow); for (int i = 0; i < shadow; i++) cast[i] = (int)rndn((uint32_t)g_nobj);
    for (int c = 0; c < 3; c++) { int col[1] = {-1}; pass_begin(r, 0, col, 5 + c, 1, 2048, 2048);
        for (int i = 0; i < shadow; i++) ids[i] = rndn(10) < 8 ? cast[i] : (int)rndn((uint32_t)g_nobj); qsort(ids, (size_t)shadow, sizeof *ids, cmp_mat); draw_objects(r, ids, shadow, 0, 1); }
    free(cast);
    r_end(r);
    // submission 1: the G-buffer, decals, the probes and small passes
    r = &g_sub[1]; compute(r, disp / 4);
    { int col[4] = {0, 1, 2, 3}; pass_begin(r, 4, col, 4, 1, 1280, 720); for (int i = 0; i < gbuf; i++) ids[i] = i % g_nobj; qsort(ids, (size_t)gbuf, sizeof *ids, cmp_mat); draw_objects(r, ids, gbuf, 1, 0); }
    for (int p = 0; p < small; p++) { int col[1] = {22 + p % 8}; pass_begin(r, 1, col, -1, p % 3 == 0, 256, 256); quads(r, 1 + (int)rndn(3), 2, 0); if (p % 10 == 9) compute(r, 1); }
    r_end(r);
    // submission 2: lighting (compute), effects and transparent draws into the scene, the post chain
    r = &g_sub[2]; compute(r, disp / 4);
    { int col[1] = {8}; pass_begin(r, 1, col, 4, 0, 1280, 720); quads(r, fx, 2, 1); }
    for (int p = 0; p < post; p++) { int col[1] = {9 + p % 12}; pass_begin(r, 1, col, -1, 0, 640, 360); quads(r, 1, 1, 0); }
    r_end(r);
    // submission 3: the HUD into the final target, the remaining dispatches
    r = &g_sub[3]; compute(r, disp - 3 * (disp / 4));
    { int col[1] = {21}; pass_begin(r, 1, col, -1, 0, 1280, 720); quads(r, ui, 1, 1); }
    r_end(r);
    free(ids);
}

// ---- texture registrations during the frame (VKB_REG_PER_FRAME, VKB_REG_MODE: 0 between submissions, 1 from a thread of their own while the replay runs) ----
static int g_reg_pf, g_reg_mode; static _Atomic int g_reg_run, g_reg_quit;
static void reg_some(int n) { for (int i = 0; i < n; i++) { if (rndn(2)) { int k = (int)rndn((uint32_t)g_nrt); mtl_reg_tex(g_ntex + 1 + k, &g_rt[k], NULL); continue; }
    int id = 1 + (int)rndn((uint32_t)g_ntex); mtl_reg_tex(id, &g_tex[id], g_has_view[id] ? &g_view[id] : NULL); } }
static void *reg_thread(void *a) { (void)a; uint64_t ns = g_reg_pf ? 16666667ull / (uint64_t)g_reg_pf : 0; while (!atomic_load(&g_reg_quit)) { if (atomic_load(&g_reg_run) && ns) { reg_some(1); struct timespec t = {0, (long)ns}; nanosleep(&t, NULL); } else usleep(1000); } return NULL; }

static int g_frame_no;
static void run_frame(void) {
    mtl_set_frame((uint64_t)g_frame_no);
    // the dirty tracker's stamp moves with the game's writes (run 13: ~1,100 write-protection faults a second), and a few times a frame pages beside a packaged texture were written
    // (run 13: "writes beside a texture only" ~0.15 a frame: its bytes hashed, pkg_hash)
    atomic_fetch_add(&g_vkb_dirty_now, (uint64_t)envi("VKB_DIRTY_STEPS", 18));
    { static double acc; acc += envf("VKB_PKG_EDGE", 0.15); while (acc >= 1) { atomic_fetch_add(&g_vkb_dirty_edge, 1); acc -= 1; } }
    // VKB_CHURN=n: every n frames a texture archive's images are dropped (a pool finalized: hle_nvn.c mtl_alias_drop_pool, its views go) and a storage buffer pool goes and comes back at
    // its address (a buffer the sets name goes): what the caches keyed on views and buffers must see (VKB_CHECK=1 in the fake driver catches a set that outlives them)
    { static int churn = -1; if (churn < 0) churn = envi("VKB_CHURN", 0);
      if (churn > 0 && g_frame_no && g_frame_no % churn == 0) { mtl_alias_drop_pool(g_ptex[rndn((uint32_t)g_nptex)].obj); vpool *sp = &g_pssbo[rndn(16)]; mtl_pool_del(sp->mem); mtl_pool_add(sp->mem, sp->size); vk_pool_set_flags(sp->mem, 0x22); } }
    atomic_store(&g_reg_run, 1);
    for (int s = 0; s < 4; s++) { if (g_reg_mode == 0) reg_some(g_reg_pf / 4); mtl_sub_begin(); mtl_exec(g_sub[s].p); mtl_sub_commit(); }
    atomic_store(&g_reg_run, 0);
    vk_tex_sweep(); vk_pass_frame(); g_frame_no++;
}

int main(int argc, char **argv) {
    (void)argc; (void)argv;
    setenv("DM_VK_SHADERS", getenv("VKB_BUNDLE") ? getenv("VKB_BUNDLE") : "", 0);
    { const char *e = getenv("VKB_LOG"); if (e && *e && !freopen(e, "w", stderr)) return 2; }
    pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);   // (the replay thread on a performance core, as the game's render thread is meant to be)
    g_reg_pf = envi("VKB_REG_PER_FRAME", 87); g_reg_mode = envi("VKB_REG_MODE", 0);
    g_arena_size = (uint64_t)envi("VKB_ARENA_MB", 1024) << 20;
    g_arena = mmap(NULL, g_arena_size, PROT_READ | PROT_WRITE, MAP_ANON | MAP_PRIVATE, -1, 0); if (g_arena == MAP_FAILED) fatal("arena");
    g_vkb_alias_base = g_arena; g_vkb_alias_size = g_arena_size;
    bundle_load(getenv("VKB_BUNDLE"));
    mtl_init(); vk_shaders_open(getenv("VKB_BUNDLE"));
    world(); frame_build();
    int warm = envi("VKB_WARM", 40), frames = envi("VKB_FRAMES", 300); g_set_ui = envi("VKB_UI_TEX", 48); g_set_fx = envi("VKB_FX_TEX", 96);
    pthread_t rt; if (g_reg_mode == 1) pthread_create(&rt, NULL, reg_thread, NULL);
    for (int i = 0; i < warm; i++) run_frame();
    mtl_wait(~0ull >> 1, 0);
    // timed frames: the replay's own counters (gfx_vk_present.c's per-frame line), and the wall time of the frame's submissions on this thread
    uint64_t d0 = g_vkst.draws, e0 = g_vkst.ns_exec, ds0 = g_vkst.ns_desc, p0 = g_vkst.ns_pipe, h0 = g_set_hits, n0 = g_set_new, b0 = g_ns_set_build, hh0 = g_ns_set_hash, nn0 = g_ns_set_new, m0 = g_set_memo, sk0 = g_vkst.skip_pipe + g_vkst.skip_rt + g_vkst.skip_prog;
    uint64_t pa0 = g_vkst.passes, di0 = g_vkst.dispatches, sb0 = g_ns_submit, c0 = atomic_load(&g_fk_calls), w0 = atomic_load(&g_fk_desc_writes), st0 = atomic_load(&g_fk_sets), up0 = atomic_load(&g_fk_updates), bi0 = atomic_load(&g_fk_binds), pd0 = atomic_load(&g_fk_pushdesc), by0 = atomic_load(&g_fk_cmd_bytes);
    double *ft = calloc((size_t)frames, sizeof *ft), *fc = calloc((size_t)frames, sizeof *fc); uint64_t t0 = vkb_now();
    double *fi = calloc((size_t)frames, sizeof *fi), *fy = calloc((size_t)frames, sizeof *fy);
    for (int i = 0; i < frames; i++) { uint64_t a = vkb_now(), c = cpu_now(), q0[2], q1[2]; ic_now(q0); run_frame(); ic_now(q1); ft[i] = (vkb_now() - a) / 1e6; fc[i] = (cpu_now() - c) / 1e6; fi[i] = (double)(q1[0] - q0[0]); fy[i] = (double)(q1[1] - q0[1]); }
    qsort(fi, (size_t)frames, sizeof *fi, cmp_d); qsort(fy, (size_t)frames, sizeof *fy, cmp_d);
    qsort(fc, (size_t)frames, sizeof *fc, cmp_d);
    uint64_t wall = vkb_now() - t0; atomic_store(&g_reg_quit, 1);
    double F = frames; int nf = frames;
    { if ((g_cz[10] || g_cz[0]) && getenv("VKB_CENSUS")) { static const char *nm0[] = {"tex_desc_for", "tvc miss", "graphics_pipeline", "last_k hit", "stage_set", "memo hit", "cbuf_desc", "drv_shared", "drv hit", "bind_program", "prepare_draw", "vk_pool_gpu", "pool miss", "vk_tex"}; static const char *nm1[] = {"new sets", "new seen earlier", "of those with scratch", "new with scratch", "stage sets", "-", "-", "-", "-", "-", "-", "-", "-", "-"}; const char **nm = getenv("VKB_CENSUS")[0] == '2' ? nm1 : nm0;
        printf("census (cumulative, warm-up included; per frame):"); for (int i = 0; i < 14; i++) printf(" %s %.0f;", nm[i], g_cz[i] / (double)(frames + warm)); printf("\n"); } } for (int i = 1; i < nf; i++) for (int j = i; j > 0 && ft[j] < ft[j - 1]; j--) { double x = ft[j]; ft[j] = ft[j - 1]; ft[j - 1] = x; }
    printf("vkbench: %d frames: %.0f draws, %.0f passes, %.0f dispatches a frame (stream: %d draws, %d passes, %d dispatches, %d calls, %d program binds; %d graphics programs, %d compute)\n",
           frames, (g_vkst.draws - d0) / F, (g_vkst.passes - pa0) / F, (g_vkst.dispatches - di0) / F, g_frame_draws, g_frame_passes, g_frame_disp, g_frame_calls, g_frame_progbinds, g_nprog, g_ncprog);
    printf("replay per frame: replay %.2f ms (descriptor sets %.2f, pipeline keys %.2f), queue submit %.2f ms; descriptor sets reused %.0f, new %.0f (collecting %.2f ms, hashing %.2f, new ones %.2f), kept per stage %.0f a frame; skipped %.1f\n",
           (g_vkst.ns_exec - e0) / F / 1e6, (g_vkst.ns_desc - ds0) / F / 1e6, (g_vkst.ns_pipe - p0) / F / 1e6, (g_ns_submit - sb0) / F / 1e6, (g_set_hits - h0) / F, (g_set_new - n0) / F,
           (g_ns_set_build - b0) / F / 1e6, (g_ns_set_hash - hh0) / F / 1e6, (g_ns_set_new - nn0) / F / 1e6, (g_set_memo - m0) / F, (g_vkst.skip_pipe + g_vkst.skip_rt + g_vkst.skip_prog - sk0) / F);
    printf("frame wall p50 %.2f / p90 %.2f ms (submit + replay + commit on this thread; %.2f ms average); its CPU time p10 %.3f / p50 %.3f ms; per draw %.2f us of replay\n", ft[nf / 2], ft[nf * 9 / 10], wall / F / 1e6, fc[nf / 10], fc[nf / 2], (g_vkst.ns_exec - e0) / 1e3 / (double)(g_vkst.draws - d0 ? g_vkst.draws - d0 : 1));
    printf("replay thread per frame: %.3f M instructions (p50; p10 %.3f), %.3f M cycles (p10; p50 %.3f): %.0f instructions a draw\n", fi[nf / 2] / 1e6, fi[nf / 10] / 1e6, fy[nf / 10] / 1e6, fy[nf / 2] / 1e6, fi[nf / 2] / ((g_vkst.draws - d0) / F));
    { extern _Atomic uint64_t g_fk_errors; if (atomic_load(&g_fk_errors)) printf("DRIVER CHECK FAILED: %llu errors (VKB_CHECK)\n", (unsigned long long)atomic_load(&g_fk_errors)); }
    printf("driver per frame: %.0f calls, %.0f KB of commands, %.0f descriptor sets allocated, %.0f vkUpdateDescriptorSets, %.0f descriptors written, %.0f set binds, %.0f pushes\n",
           (atomic_load(&g_fk_calls) - c0) / F, (atomic_load(&g_fk_cmd_bytes) - by0) / F / 1024, (atomic_load(&g_fk_sets) - st0) / F, (atomic_load(&g_fk_updates) - up0) / F, (atomic_load(&g_fk_desc_writes) - w0) / F, (atomic_load(&g_fk_binds) - bi0) / F, (atomic_load(&g_fk_pushdesc) - pd0) / F);
    const char *js = getenv("VKB_JSON");
    if (js && *js) { FILE *f = fopen(js, "a"); if (f) { fprintf(f, "{\"label\": \"%s\", \"frames\": %d, \"draws\": %.1f, \"replay_ms\": %.4f, \"desc_ms\": %.4f, \"pipe_ms\": %.4f, \"collect_ms\": %.4f, \"hash_ms\": %.4f, \"new_ms\": %.4f, "
        "\"reused\": %.1f, \"new\": %.1f, \"memo\": %.1f, \"wall_p50\": %.4f, \"wall_avg\": %.4f, \"cpu_p10\": %.4f, \"cpu_p50\": %.4f, \"minstr\": %.4f, \"mcycles\": %.4f, \"sets\": %.1f, \"updates\": %.1f, \"desc_writes\": %.1f, \"binds\": %.1f, \"pushes\": %.1f, \"calls\": %.1f}\n",
        getenv("VKB_LABEL") ? getenv("VKB_LABEL") : "", frames, (g_vkst.draws - d0) / F, (g_vkst.ns_exec - e0) / F / 1e6, (g_vkst.ns_desc - ds0) / F / 1e6, (g_vkst.ns_pipe - p0) / F / 1e6, (g_ns_set_build - b0) / F / 1e6,
        (g_ns_set_hash - hh0) / F / 1e6, (g_ns_set_new - nn0) / F / 1e6, (g_set_hits - h0) / F, (g_set_new - n0) / F, (g_set_memo - m0) / F, ft[nf / 2], wall / F / 1e6, fc[nf / 10], fc[nf / 2], fi[nf / 2] / 1e6, fy[nf / 10] / 1e6, (atomic_load(&g_fk_sets) - st0) / F,
        (atomic_load(&g_fk_updates) - up0) / F, (atomic_load(&g_fk_desc_writes) - w0) / F, (atomic_load(&g_fk_binds) - bi0) / F, (atomic_load(&g_fk_pushdesc) - pd0) / F, (atomic_load(&g_fk_calls) - c0) / F); fclose(f); } }
    fflush(stdout); _exit(0);   // (the replay's threads never end)
}
