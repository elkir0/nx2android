// Android port, Vulkan backend: shader-side virtual texturing of sparse textures that cannot be sparse-resident (gfx_vk_vt.h; the shader side tools/spv_vt.py).
// Why: the stock Adreno 840 driver gives sparse residency to no block-compressed format, and its sparse address space is 3-4 GB, so Mario Kart World 1.8.0's terrain arrays
// (8192x8192x68, two BC6H of 5.8 GB and four ASTC 8x8 of 1.45 GB made whole, 12 levels, 3-12 % mapped in 64 KB tiles) are dense images from level 4 (BC6H) and 3 (ASTC) under
// DM_VK_SPARSE_DENSE_MB (run 13: "a dense image from level 3 of 12 (the finest mapped 0 ...)"): the terrain is drawn 8 to 16 times coarser than the game maps it. The game maps
// 12,392 terrain tiles (774 MB) by its flyover on the Mac (race1.log, nvn_mtl.m placement sparse), far more than a phone with 2.1-2.3 GB available in a race can hold; what a
// 720p frame samples is a small part of it. So the levels finer than the dense image's first come from a pool of slots filled on demand: a shader that finds a tile it needs
// without a slot sets its request bit, the request region of a submission is read once its fence has passed, and the tiles asked for that the game maps are uploaded (64 KB
// deswizzled, nvn_deswizzle_tile, and a border of one block of each neighbour, so the pool's bilinear filter is the texture's own; whole blocks: no re-encoding). Slots go back
// when the game unmaps their tile, and by a CLOCK: every present the hand puts a few slots on probation (a page table bit), a shader sampling a slot on probation asks for it
// (which takes it off), and a slot still on probation 12 submissions later is taken back when the free slots run low. A slot taken back is reused only once the GPU is past the
// submission that last could read it (its page table word is rewritten in the same upload command buffer as the new tile, after it). DM_VK_VT (start-up) 0 / 1 / 2: gfx_vk_vt.h;
// DM_VK_VT_SLOTS (start-up, default 1024): slots a pool (one a format and tile shape, 68 KB each with its border); DM_VK_VT_UPLOADS (start-up, default 48): tiles uploaded a
// submission; DM_VK_VT_PT_MB (start-up, default 4): the page table
#include "gfx_vk_vt.h"
#include "nvn_tex.h"
#include "nvn_deswizzle.h"
#include "nvn_dirty.h"
#include <stdlib.h>
#include <string.h>
#include <time.h>

// ---- VT logic: geometry, the control block, a slot's content, the slots' CLOCK (tests: tests/test_vk_vt.py); no Vulkan ----
#define VT_TILE 0x10000u
enum { VT_PROB_SUBS = 12 };   // submissions a slot stays on probation before it may be taken back (requests come back 2-4 submissions later: NSLOT is 6)
typedef struct {
    uint32_t W, H, layers, ntiled, tw, th, tws, ths, bw, bh, bpb, nbx, nby;   // level 0 texels; tiled levels; a tile's texels (2^tws x 2^ths) and blocks (nbx x nby)
    uint32_t nx[16], ny[16], loff_t[16], per_layer;                          // a level's tiles across and down, its first tile in a layer; tiles a layer
    uint64_t loff[16], layer_size;                                           // a level's byte offset in a layer (NVN's, nvn_tex_level_offset)
} vt_geom;
static uint32_t vt_log2(uint32_t v) { uint32_t n = 0; while (v > 1) { v >>= 1; n++; } return n; }
// 0, or why the texture is not one VT takes: its tiled levels whole power-of-two tiles of whole square blocks, each level at its NVN offset, at most 16 levels
static const char *vt_geom_init(vt_geom *g, uint32_t W, uint32_t H, uint32_t layers, uint32_t ntiled, uint32_t tw, uint32_t th, uint32_t bw, uint32_t bh, uint32_t bpb, const uint64_t *loff,
                                uint64_t layer_size) {
    memset(g, 0, sizeof *g);
    if (!ntiled || ntiled > 16 || !tw || !th || tw & (tw - 1) || th & (th - 1) || !bw || bw != bh || tw % bw || th % bh || !bpb || W > 65535 || H > 65535) return "its tile or block shape";
    g->W = W; g->H = H; g->layers = layers ? layers : 1; g->ntiled = ntiled; g->tw = tw; g->th = th; g->tws = vt_log2(tw); g->ths = vt_log2(th); g->bw = bw; g->bh = bh; g->bpb = bpb;
    g->nbx = tw / bw; g->nby = th / bh; g->layer_size = layer_size;
    if ((uint64_t)g->nbx * g->nby * bpb != VT_TILE) return "a tile that is not 64 KB of blocks";
    uint32_t k = 0;
    for (uint32_t l = 0; l < ntiled; l++) {
        uint32_t lw = W >> l, lh = H >> l;
        if (!lw || !lh || lw % tw || lh % th) return "a tiled level that is not whole tiles";
        g->nx[l] = lw / tw; g->ny[l] = lh / th; g->loff_t[l] = k; g->loff[l] = loff[l]; k += g->nx[l] * g->ny[l];
        uint64_t next = l + 1 < ntiled ? loff[l + 1] : 0;
        if (l + 1 < ntiled && next - loff[l] != (uint64_t)g->nx[l] * g->ny[l] * VT_TILE) return "a level not laid out as whole tiles";
    }
    g->per_layer = k;
    return NULL;
}
static uint32_t vt_place(const vt_geom *g, uint32_t y, uint32_t l, uint32_t tx, uint32_t ty) { return y * g->per_layer + g->loff_t[l] + ty * g->nx[l] + tx; }
static void vt_place_of(const vt_geom *g, uint32_t p, uint32_t *y, uint32_t *l, uint32_t *tx, uint32_t *ty) {
    *y = p / g->per_layer; uint32_t i = p % g->per_layer, k = 0; while (k + 1 < g->ntiled && i >= g->loff_t[k + 1]) k++;
    *l = k; i -= g->loff_t[k]; *tx = i % g->nx[k]; *ty = i / g->nx[k];
}
static uint64_t vt_tile_off(const vt_geom *g, uint32_t y, uint32_t l, uint32_t tx, uint32_t ty) { return (uint64_t)y * g->layer_size + g->loff[l] + (uint64_t)(ty * g->nx[l] + tx) * VT_TILE; }
// NVN's address mode (gfx_vk_tex.c wrap) → the template's: 0 clamp to edge (and to border: a texel's footprint never leaves the texture), 1 repeat, 2 mirrored repeat, 3 mirror clamp
static int vt_wrap_code(int ws) { return ws == 1 ? 1 : ws == 6 ? 2 : ws == 2 || ws == 3 ? 3 : 0; }
// A candidate's three uvec4 of vt_ctl (spv_vt.py TEMPLATE_GLSL): e[0] mode and filters, first, page table base, layers; e[1] level 0 size, tile shape and border, tiled levels, tiles
// a layer; e[2] the sampler's lowest and highest LOD in the texture's levels (floats), which vt_qlod clamps the whole texture's LOD to
static void vt_ctl_fill(uint32_t e[12], int mode, uint32_t first, int linear, int wu, int wv, uint32_t pt_base, const vt_geom *g, float lod_min, float lod_max) {
    memset(e, 0, 12 * sizeof *e);
    e[0] = (uint32_t)mode | (linear ? VT_CTL_LINEAR : 0) | VT_CTL_WRAPU(wu) | VT_CTL_WRAPV(wv); e[1] = first;
    memcpy(&e[8], &lod_min, 4); memcpy(&e[9], &lod_max, 4);
    if (mode != VT_MODE_VT || !g) return;
    e[2] = pt_base; e[3] = g->layers; e[4] = g->W | g->H << 16; e[5] = g->tws | g->ths << 8 | g->bw << 16; e[6] = g->ntiled; e[7] = g->per_layer;
}
// The block for the view the shader reads: its level 0 the texture's level base, its layer 0 the texture's layer ly0, nly layers. The shader's level numbers are the view's, so
// first is the view's shift (its levels below the dense image's); the page table is read from the view's base level and layer on (the view's geometry: the texture's from base).
// VT only where every level of the view below the dense image is a tiled level of the texture, else the dense image's LOD fix alone
static void vt_ctl_view(uint32_t e[12], int mode, const vt_geom *g, uint32_t pt_base, uint32_t base, uint32_t ly0, uint32_t nly, uint32_t shift, int linear, int wu, int wv, float lo, float hi) {
    if (mode == VT_MODE_VT && (!g || base >= g->ntiled || shift > g->ntiled - base || ly0 >= g->layers)) mode = VT_MODE_DENSE;
    if (mode != VT_MODE_VT) { vt_ctl_fill(e, mode, shift, linear, wu, wv, 0, NULL, lo, hi); return; }
    vt_geom v = *g; v.W = g->W >> base; v.H = g->H >> base; v.ntiled = g->ntiled - base; v.layers = nly && nly <= g->layers - ly0 ? nly : g->layers - ly0;
    for (uint32_t l = 0; l < 16; l++) { int in = l + base < g->ntiled; v.nx[l] = in ? g->nx[l + base] : 0; v.ny[l] = in ? g->ny[l + base] : 0; v.loff_t[l] = in ? g->loff_t[l + base] - g->loff_t[base] : 0; v.loff[l] = in ? g->loff[l + base] : 0; }
    vt_ctl_fill(e, VT_MODE_VT, shift, linear, wu, wv, pt_base + ly0 * g->per_layer + g->loff_t[base], &v, lo, hi);
}
// A slot's content: (nbx + 2) x (nby + 2) blocks, rows of pitch bytes: tile (tx, ty) of level l, layer y at (1, 1) and around it one block of each of its 8 neighbours (wrapped at
// the level's edges, so a repeating sampler filters across them as the texture does; a clamping one never reaches them), or of zero blocks where the game does not map one (NVN
// reads zeros there). pieces: bit 0 the tile itself, bits 1-8 the left, right, top and bottom strips and the top-left, top-right, bottom-left and bottom-right corners. rd reads n
// bytes of the texture's storage at off (64 KB of the tile, else whole 512-byte GOBs); mapped says whether the game maps all of a tile. A neighbour's blocks are at their
// block-linear offsets in its own 64 KB (nvn_bl_offset, the layout nvn_deswizzle_tile reads: a tile is one block of GOBs, nby / 8 GOBs high). 0, or -1 when the tile is not decoded
typedef void (*vt_read_fn)(void *ctx, uint64_t off, void *dst, uint32_t n);
typedef int (*vt_mapped_fn)(void *ctx, uint32_t place);
static void vt_blocks(const vt_geom *g, uint32_t y, uint32_t l, uint32_t tx, uint32_t ty, uint32_t bx0, uint32_t by0, uint32_t nbxs, uint32_t nbys, uint8_t *dst, size_t pitch,
                      const uint8_t *zero, vt_read_fn rd, vt_mapped_fn mp, void *ctx) {   // blocks [bx0, +nbxs) x [by0, +nbys) of a tile into dst
    if (!mp(ctx, vt_place(g, y, l, tx, ty))) { for (uint32_t j = 0; j < nbys; j++) for (uint32_t i = 0; i < nbxs; i++) memcpy(dst + j * pitch + i * g->bpb, zero, g->bpb); return; }
    uint64_t base = vt_tile_off(g, y, l, tx, ty); uint8_t gob[512]; uint64_t at = ~0ull; unsigned thg = vt_log2(g->nby / 8);
    for (uint32_t j = 0; j < nbys; j++) for (uint32_t i = 0; i < nbxs; i++) {
        uint64_t o = nvn_bl_offset((bx0 + i) * g->bpb, by0 + j, 0, g->nbx * g->bpb, g->nby, thg, 0);
        if ((o & ~511ull) != at) { at = o & ~511ull; rd(ctx, base + at, gob, 512); }
        memcpy(dst + j * pitch + i * g->bpb, gob + (o & 511), g->bpb);
    }
}
static int vt_slot_build(const vt_geom *g, const nvn_texdesc *d, uint32_t y, uint32_t l, uint32_t tx, uint32_t ty, uint32_t pieces, const uint8_t *zero, vt_read_fn rd, vt_mapped_fn mp,
                         void *ctx, uint8_t *tile, uint8_t *dst) {
    size_t pitch = (size_t)(g->nbx + 2) * g->bpb; uint32_t nx = g->nx[l], ny = g->ny[l], L = (tx + nx - 1) % nx, R = (tx + 1) % nx, T = (ty + ny - 1) % ny, B = (ty + 1) % ny;
    uint32_t X = g->nbx, Y = g->nby;
    if (pieces & 1) {
        if (!mp(ctx, vt_place(g, y, l, tx, ty))) vt_blocks(g, y, l, tx, ty, 0, 0, X, Y, dst + pitch + g->bpb, pitch, zero, rd, mp, ctx);
        else { rd(ctx, vt_tile_off(g, y, l, tx, ty), tile, VT_TILE); uint32_t wb, rows;
               if (nvn_deswizzle_tile(d, l, tx, ty, tile, dst + pitch + g->bpb, pitch, &wb, &rows) || wb != X * g->bpb || rows != Y) return -1; }
    }
    if (pieces & 2) vt_blocks(g, y, l, L, ty, X - 1, 0, 1, Y, dst + pitch, pitch, zero, rd, mp, ctx);                                   // left strip: the left tile's last column
    if (pieces & 4) vt_blocks(g, y, l, R, ty, 0, 0, 1, Y, dst + pitch + (X + 1) * g->bpb, pitch, zero, rd, mp, ctx);                   // right: the right tile's first column
    if (pieces & 8) vt_blocks(g, y, l, tx, T, 0, Y - 1, X, 1, dst + g->bpb, pitch, zero, rd, mp, ctx);                                // top: the upper tile's last row
    if (pieces & 16) vt_blocks(g, y, l, tx, B, 0, 0, X, 1, dst + (Y + 1) * pitch + g->bpb, pitch, zero, rd, mp, ctx);                  // bottom: the lower tile's first row
    if (pieces & 32) vt_blocks(g, y, l, L, T, X - 1, Y - 1, 1, 1, dst, pitch, zero, rd, mp, ctx);
    if (pieces & 64) vt_blocks(g, y, l, R, T, 0, Y - 1, 1, 1, dst + (X + 1) * g->bpb, pitch, zero, rd, mp, ctx);
    if (pieces & 128) vt_blocks(g, y, l, L, B, X - 1, 0, 1, 1, dst + (Y + 1) * pitch, pitch, zero, rd, mp, ctx);
    if (pieces & 256) vt_blocks(g, y, l, R, B, 0, 0, 1, 1, dst + (Y + 1) * pitch + (X + 1) * g->bpb, pitch, zero, rd, mp, ctx);
    return 0;
}
// The pieces of neighbour (dx, dy) of a tile (-1..1 each) that face it: the ones its slot copies from the tile (bit numbers as vt_slot_build's)
static uint32_t vt_facing(int dx, int dy) {   // the neighbour at (dx, dy) holds, in its own border, the piece on the opposite side
    static const uint32_t P[3][3] = {{256, 16, 128}, {4, 0, 2}, {64, 8, 32}};   // [dy + 1][dx + 1]: e.g. the tile to the left (dx -1) keeps this tile's first column as its right strip (4)
    return P[dy + 1][dx + 1];
}
// The slots of one pool. place: the page table word it holds (VT_FREE none); prob: the submission its probation began (0 none); free_at: a slot taken back is free once the
// fence of the submission that took it back has passed
#define VT_FREE 0xffffffffu
typedef struct { uint32_t place; uint64_t prob, last, free_at, gen, stamp; } vt_slot;
typedef struct { vt_slot *s; uint32_t n, hand, *free, nfree, *late, nlate; } vt_slots;
static int vt_slots_init(vt_slots *v, uint32_t n) {
    memset(v, 0, sizeof *v); v->n = n; v->s = calloc(n, sizeof *v->s); v->free = malloc(n * sizeof *v->free); v->late = malloc(n * sizeof *v->late);
    if (!v->s || !v->free || !v->late) return -1;
    for (uint32_t i = 0; i < n; i++) { v->s[i].place = VT_FREE; v->free[v->nfree++] = n - 1 - i; }
    return 0;
}
__attribute__((unused)) static void vt_slots_free(vt_slots *v) { free(v->s); free(v->free); free(v->late); memset(v, 0, sizeof *v); }
static void vt_slot_give(vt_slots *v, uint32_t i, uint64_t fence) { v->s[i].place = VT_FREE; v->s[i].prob = 0; v->s[i].free_at = fence; v->late[v->nlate++] = i; }   // free once fence has passed
static int vt_slot_take(vt_slots *v, uint64_t done, uint32_t place, uint64_t sub) {   // a free slot for place (-1: none now)
    for (uint32_t k = 0; k < v->nlate;) { uint32_t i = v->late[k]; if (v->s[i].free_at <= done) { v->free[v->nfree++] = i; v->late[k] = v->late[--v->nlate]; } else k++; }
    if (!v->nfree) return -1;
    uint32_t i = v->free[--v->nfree]; v->s[i] = (vt_slot){place, 0, sub, 0, 0, 0}; return (int)i;
}
// The hand's step at a present: `step` resident slots looked at; one off probation goes on it (set_prob), one on probation for VT_PROB_SUBS submissions or more is taken back
// (evict: its page table word loses the slot) while fewer than `low` slots are free or on their way back. Returns the slots taken back
static uint32_t vt_clock(vt_slots *v, uint32_t step, uint32_t low, uint64_t sub, uint64_t fence, void (*set_prob)(void *, uint32_t place), void (*evict)(void *, uint32_t place), void *ctx) {
    uint32_t gone = 0;
    for (uint32_t k = 0, seen = 0; k < v->n && seen < step; k++) {
        uint32_t i = v->hand; v->hand = (v->hand + 1) % v->n; vt_slot *s = &v->s[i];
        if (s->place == VT_FREE) continue;
        seen++;
        if (!s->prob) { s->prob = sub ? sub : 1; set_prob(ctx, s->place); continue; }
        if (sub - s->prob >= VT_PROB_SUBS && v->nfree + v->nlate < low) { evict(ctx, s->place); vt_slot_give(v, i, fence); gone++; }
    }
    return gone;
}
// A piece's rectangle in its slot's blocks (x0, y0, x1, y1): the whole slot for 0x1ff, else one piece (bit as vt_slot_build's)
static void vt_piece_rect(uint32_t pieces, uint32_t X, uint32_t Y, uint32_t r[4]) {
    switch (pieces) {
    case 2: r[0] = 0; r[1] = 1; r[2] = 1; r[3] = Y + 1; return;              case 4: r[0] = X + 1; r[1] = 1; r[2] = X + 2; r[3] = Y + 1; return;
    case 8: r[0] = 1; r[1] = 0; r[2] = X + 1; r[3] = 1; return;              case 16: r[0] = 1; r[1] = Y + 1; r[2] = X + 1; r[3] = Y + 2; return;
    case 32: r[0] = 0; r[1] = 0; r[2] = 1; r[3] = 1; return;                 case 64: r[0] = X + 1; r[1] = 0; r[2] = X + 2; r[3] = 1; return;
    case 128: r[0] = 0; r[1] = Y + 1; r[2] = 1; r[3] = Y + 2; return;        case 256: r[0] = X + 1; r[1] = Y + 1; r[2] = X + 2; r[3] = Y + 2; return;
    default: r[0] = 0; r[1] = 0; r[2] = X + 2; r[3] = Y + 2; return;
    }
}
// The runs of set bits of a bitmap of n bits (64 a word) from bit `from`: at most max runs (start, length) into out, the bits taken cleared. Returns the runs; *next where to go on
static uint32_t vt_runs(uint64_t *bm, uint32_t n, uint32_t from, uint32_t (*out)[2], uint32_t max, uint32_t *next) {
    uint32_t k = 0, i = from;
    while (i < n && k < max) {
        uint64_t w = bm[i >> 6] >> (i & 63);
        if (!w) { i = (i | 63) + 1; continue; }
        i += (uint32_t)__builtin_ctzll(w); uint32_t j = i;
        while (j < n && bm[j >> 6] >> (j & 63) & 1) { bm[j >> 6] &= ~(1ull << (j & 63)); j++; }
        out[k][0] = i; out[k][1] = j - i; k++; i = j;
    }
    *next = i; return k;
}
// ---- end of the VT logic ----


// ---------------- the backend's state ----------------
enum { NFB = 8 };   // request regions (more than the submissions in flight: NSLOT is 6)
typedef struct {
    VkFormat fmt; uint32_t bw, bh, bpb, nbx, nby, sw, sh, n;   // the tile shape it holds, a slot's texels (sw x sh, its border included), its slots
    VkImage img; VkDeviceMemory mem; int ready, failed; VkImageView view[4]; VkFormat vfmt[4]; int nview; vt_slots sl;
} vt_pool;
typedef struct vt_tex {
    vkimg_t *im; vt_geom g; nvn_texdesc d; const void *pool; uint64_t mem; int pi; uint8_t zero[16]; uint32_t pt_base, pt_n, dn_per_layer, tail_at;
} vt_tex;
typedef struct { uint32_t i, pieces; } vt_piece_job;   // a page table word's slot, the border pieces of it to build again
static struct {
    pthread_mutex_t mu; int mode, init, cleared;
    uint32_t *pt, ptcap, pt_hi; uint64_t *dirty; uint32_t ndirty, dnext; VkBuffer ptbuf; VkDeviceMemory ptmem; uint8_t *queued;   // the page table's shadow, its words to copy, the GPU's copy
    VkBuffer fbbuf; VkDeviceMemory fbmem; uint32_t *fb, rw; uint64_t regf[NFB]; int reg; uint64_t sub, frame;   // requests: rw words a region, the fence of each region's submission
    vt_tex *tex[64]; int ntex; vt_pool pools[8]; int npool; VkSampler samp[4]; uint8_t *tile;
    uint32_t *q, nq, capq; vt_piece_job *pj; uint32_t npj, cappj; uint32_t slots_n, uploads;
    uint64_t n_up, n_req, n_touch, n_evict, n_nofree, n_unmap, n_piece, n_pt, n_cpu, wseen, wframe; double t_log; uint8_t *slotbuf;
} g_vt = {.mu = PTHREAD_MUTEX_INITIALIZER, .reg = -1};
// (gfx_vk_tex.c gives the sampler parameters once its VT hooks are in: until then no candidate is anything but mode 0)
__attribute__((weak)) int vk_samp_par_at(int sid, uint64_t seq, samp_par *out) { (void)sid; (void)seq; (void)out; return 0; }
static int env_int(const char *k, int d) { const char *e = getenv(k); return e && *e ? atoi(e) : d; }
int vk_vt_mode(void) { if (!g_vt.init) { g_vt.mode = env_int("DM_VK_VT", 0); g_vt.init = 1; } return g_vt.mode; }
static uint32_t pt_get(uint32_t i) { return i < g_vt.ptcap ? g_vt.pt[i] : 0; }
static void pt_set(uint32_t i, uint32_t v) {
    if (i >= g_vt.ptcap || g_vt.pt[i] == v) return;
    g_vt.pt[i] = v; uint64_t b = 1ull << (i & 63); if (!(g_vt.dirty[i >> 6] & b)) { g_vt.dirty[i >> 6] |= b; g_vt.ndirty++; }
}
static vt_tex *owner(uint32_t i) { for (int k = 0; k < g_vt.ntex; k++) { vt_tex *v = g_vt.tex[k]; if (i >= v->pt_base && i < v->pt_base + v->pt_n) return v; } return NULL; }
static uint64_t now_fence(void) { return mtl_last_fence() + 1; }   // the fence the submission being recorded will get (gfx_vk_core.c mtl_sub_commit)
static int buf_make(VkDeviceSize n, VkBufferUsageFlags u, VkMemoryPropertyFlags want, VkMemoryPropertyFlags pref, VkBuffer *b, VkDeviceMemory *m, void **map) {
    VkBufferCreateInfo bi = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, NULL, 0, n, u, VK_SHARING_MODE_EXCLUSIVE, 0, NULL};
    if (vkCreateBuffer(V.dev, &bi, NULL, b)) { *b = VK_NULL_HANDLE; return -1; }
    VkMemoryRequirements r; vkGetBufferMemoryRequirements(V.dev, *b, &r);
    uint32_t t = vk_mem_type(r.memoryTypeBits, want | pref, 0); if (t == UINT32_MAX) t = vk_mem_type(r.memoryTypeBits, want, 0);
    VkMemoryAllocateInfo ai = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, NULL, r.size, t};
    if (t == UINT32_MAX || vkAllocateMemory(V.dev, &ai, NULL, m) || vkBindBufferMemory(V.dev, *b, *m, 0) || (map && vkMapMemory(V.dev, *m, 0, VK_WHOLE_SIZE, 0, map))) {
        vkDestroyBuffer(V.dev, *b, NULL); *b = VK_NULL_HANDLE; return -1; }
    return 0;
}
// The page table (DM_VK_VT_PT_MB of device memory, written only by copies in upload command buffers) and the request regions (host-visible), at the first VT texture
static int vt_init(void) {
    if (g_vt.ptbuf) return 0;
    uint32_t mb = (uint32_t)env_int("DM_VK_VT_PT_MB", 4); if (mb < 1) mb = 1; if (mb > 64) mb = 64;
    g_vt.ptcap = mb << 18; g_vt.rw = g_vt.ptcap / 32; g_vt.slots_n = (uint32_t)env_int("DM_VK_VT_SLOTS", 1024); g_vt.uploads = (uint32_t)env_int("DM_VK_VT_UPLOADS", 48);
    if (g_vt.slots_n < 64) g_vt.slots_n = 64;
    if (g_vt.slots_n > V.props.limits.maxImageArrayLayers) g_vt.slots_n = V.props.limits.maxImageArrayLayers;
    if (g_vt.slots_n > VT_PT_SLOT) g_vt.slots_n = VT_PT_SLOT;
    if (!g_vt.uploads) g_vt.uploads = 1;
    if (!(g_vt.pt = calloc(g_vt.ptcap, 4)) || !(g_vt.dirty = calloc(g_vt.ptcap / 64, 8)) || !(g_vt.queued = calloc(g_vt.ptcap / 8, 1)) || !(g_vt.tile = malloc(VT_TILE))) die("[vk] VT page table shadow");
    if (buf_make((VkDeviceSize)g_vt.ptcap * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 0, &g_vt.ptbuf, &g_vt.ptmem, NULL) ||
        buf_make((VkDeviceSize)g_vt.rw * 4 * NFB, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, VK_MEMORY_PROPERTY_HOST_CACHED_BIT,
                 &g_vt.fbbuf, &g_vt.fbmem, (void **)&g_vt.fb)) {
        if (g_vt.ptbuf) { vkDestroyBuffer(V.dev, g_vt.ptbuf, NULL); vkFreeMemory(V.dev, g_vt.ptmem, NULL); g_vt.ptbuf = VK_NULL_HANDLE; }
        VKLOG("[RED] VT: no memory for its page table (%u MB) or its requests: the dense images' LOD fix alone (DM_VK_VT=2)", mb); g_vt.mode = 2; return -1; }
    memset(g_vt.fb, 0, (size_t)g_vt.rw * 4 * NFB);
    VKLOG("VT: page table %u MB (%u tiles), request regions %u x %u KB, pools of %u slots, %u tiles uploaded a submission at most", mb, g_vt.ptcap, NFB, g_vt.rw / 256, g_vt.slots_n, g_vt.uploads);
    return 0;
}
// A pool for f's tiles of nbx x nby blocks: a 2D array image, a slot a layer (the tile and a border of one block), made in f with its views in f's twins (MUTABLE_FORMAT).
// Its entry at attach, its image at the first draw that binds it (pool_ready): a bundle without the transform (spv_vt.py) never makes one
static int pool_for(const vkfmt_t *f, uint32_t nbx, uint32_t nby) {
    for (int i = 0; i < g_vt.npool; i++) if (g_vt.pools[i].fmt == f->vk && g_vt.pools[i].nbx == nbx && g_vt.pools[i].nby == nby) return g_vt.pools[i].failed ? -1 : i;
    if (g_vt.npool == 8) return -1;
    vt_pool *p = &g_vt.pools[g_vt.npool++]; memset(p, 0, sizeof *p);
    p->fmt = f->vk; p->bw = f->bw; p->bh = f->bh; p->bpb = f->bpb; p->nbx = nbx; p->nby = nby; p->sw = (nbx + 2) * f->bw; p->sh = (nby + 2) * f->bh; p->n = g_vt.slots_n;
    return g_vt.npool - 1;
}
static int pool_ready(vt_pool *p) {   // its image made (1), or refused (0: its textures keep their dense images alone)
    if (p->img) return 1;
    if (p->failed) return 0;
    VkImageCreateInfo ci = {VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, NULL, VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT, VK_IMAGE_TYPE_2D, p->fmt, {p->sw, p->sh, 1}, 1, p->n, VK_SAMPLE_COUNT_1_BIT,
                            VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT, VK_SHARING_MODE_EXCLUSIVE, 0, NULL, VK_IMAGE_LAYOUT_UNDEFINED};
    VkMemoryRequirements r = {0}; VkResult e = vkCreateImage(V.dev, &ci, NULL, &p->img);
    if (!e) { vkGetImageMemoryRequirements(V.dev, p->img, &r);
              VkMemoryAllocateInfo ai = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, NULL, r.size, vk_mem_type(r.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 0)};
              e = ai.memoryTypeIndex == UINT32_MAX ? VK_ERROR_OUT_OF_DEVICE_MEMORY : vkAllocateMemory(V.dev, &ai, NULL, &p->mem);
              if (!e) e = vkBindImageMemory(V.dev, p->img, p->mem, 0); }
    else p->img = VK_NULL_HANDLE;
    if (e || vt_slots_init(&p->sl, p->n)) {
        VKLOG("[RED] VT: the pool for format %d (%u slots of %ux%u) was refused (%d): its textures keep their dense images alone", p->fmt, p->n, p->sw, p->sh, e);
        if (p->img) vkDestroyImage(V.dev, p->img, NULL);
        if (p->mem) vkFreeMemory(V.dev, p->mem, NULL);
        p->img = VK_NULL_HANDLE; p->mem = VK_NULL_HANDLE; p->failed = 1; return 0; }
    VKLOG("VT: pool for format %d: %u slots of %ux%u texels (a tile of %u x %u blocks and a border of one), %.1f MB", p->fmt, p->n, p->sw, p->sh, p->nbx, p->nby, r.size / 1048576.0);
    return 1;
}
static VkImageView pool_view(vt_pool *p, VkFormat vf) {
    if (!vf) vf = p->fmt;
    for (int i = 0; i < p->nview; i++) if (p->vfmt[i] == vf) return p->view[i];
    if (p->nview == 4) return p->view[0];
    VkImageViewCreateInfo vi = {VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, NULL, 0, p->img, VK_IMAGE_VIEW_TYPE_2D_ARRAY, vf, {0, 0, 0, 0}, {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, p->n}};
    VkImageView v;
    if (vkCreateImageView(V.dev, &vi, NULL, &v)) { VKRED_ONCE(170, "VT: a view in format %d of the pool made in %d was refused: the pool's own format", vf, p->fmt); return p->nview ? p->view[0] : VK_NULL_HANDLE; }
    p->view[p->nview] = v; p->vfmt[p->nview++] = vf; return v;
}

// ---------------- textures ----------------
static int is_astc(VkFormat f) { return f >= VK_FORMAT_ASTC_4x4_UNORM_BLOCK && f <= VK_FORMAT_ASTC_12x12_SRGB_BLOCK; }
int vk_vt_attach(vkimg_t *im, const tex_t *t, const vkfmt_t *f, const uint8_t *mapped, uint32_t per_layer, uint32_t tail_at) {
    if (vk_vt_mode() != 1 || !im || !t || !f || f->dec || !f->compressed || (f->bpb != 16 && f->bpb != 8) || !im->lv0) return 0;
    uint32_t tw, th, td; int n = nvn_tex_sparse_tiles(&t->d, &tw, &th, &td); if (n <= 0 || td != 1) return 0;
    if (n > 16) n = 16;
    uint32_t layers = im->layers ? im->layers : 1; uint64_t loff[16];
    for (int l = 0; l < n; l++) loff[l] = nvn_tex_level_offset(&t->d, (unsigned)l);
    vt_tex *v = calloc(1, sizeof *v); if (!v) return 0;
    const char *why = vt_geom_init(&v->g, t->d.width, t->d.height, layers, (uint32_t)n, tw, th, f->bw, f->bh, f->bpb, loff, t->L.layer_size ? t->L.layer_size : t->L.size);
    if (!why && v->g.per_layer != tail_at) why = "its tiles counted otherwise than its dense image's";
    pthread_mutex_lock(&g_vt.mu);
    if (!why && vt_init()) why = "no page table";
    uint32_t need = layers * v->g.per_layer, base = 0;
    if (!why) {   // the first gap of the page table that fits (textures are few)
        for (int k = 0; k <= g_vt.ntex; k++) { uint32_t end = k < g_vt.ntex ? g_vt.tex[k]->pt_base : g_vt.ptcap; if (end >= base + need) break; if (k < g_vt.ntex) base = g_vt.tex[k]->pt_base + g_vt.tex[k]->pt_n; }
        if ((uint64_t)base + need > g_vt.ptcap || g_vt.ntex == 64) why = "no room in the page table (DM_VK_VT_PT_MB)";
    }
    if (!why && (v->pi = pool_for(f, v->g.nbx, v->g.nby)) < 0) why = "no pool for its format";
    if (why) { pthread_mutex_unlock(&g_vt.mu); static int nl; if (nl++ < 8) VKLOG("VT: sparse texture %ux%ux%u nvn format %#x keeps its dense image alone: %s", t->d.width, t->d.height, layers, t->d.format, why); free(v); return 0; }
    v->im = im; v->d = t->d; v->pool = t->pool; v->mem = (uint64_t)(uintptr_t)t->mem; v->pt_base = base; v->pt_n = need; v->dn_per_layer = per_layer; v->tail_at = tail_at;
    if (is_astc(f->vk)) { static const uint8_t ve[16] = {0xfc, 0xfd, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff}; memcpy(v->zero, ve, 16); }   // ASTC: a void-extent block of colour 0 (zero bytes decode as its error colour)
    int at = 0; while (at < g_vt.ntex && g_vt.tex[at]->pt_base < base) at++;
    memmove(&g_vt.tex[at + 1], &g_vt.tex[at], (size_t)(g_vt.ntex - at) * sizeof *g_vt.tex); g_vt.tex[at] = v; g_vt.ntex++;
    if (base + need > g_vt.pt_hi) g_vt.pt_hi = base + need;
    uint32_t nm = 0;
    for (uint32_t y = 0; y < layers; y++) for (uint32_t i = 0; i < tail_at; i++) if (mapped && mapped[(size_t)y * per_layer + i]) { pt_set(base + y * v->g.per_layer + i, VT_PT_MAPPED); nm++; }
    pthread_mutex_unlock(&g_vt.mu);
    extern uint64_t g_tex_gen; __atomic_add_fetch(&g_tex_gen, 1, __ATOMIC_RELEASE);   // (gfx_vk_exec.c: the stage sets memoized before named no pool)
    VKLOG("VT: sparse texture %ux%ux%u nvn format %#x: levels 0-%u through the pool of format %d (%u tiles a layer, page table words %u-%u), %u of %u tiles mapped; its dense image from level %u",
          t->d.width, t->d.height, layers, t->d.format, v->g.ntiled - 1, f->vk, v->g.per_layer, base, base + need - 1, nm, need, im->lv0);
    return 1;
}
static void slot_drop(vt_tex *v, uint32_t i) {   // word i's slot back (free once the GPU is past this submission), the word keeping its mapping bit
    uint32_t e = pt_get(i), s = e & VT_PT_SLOT; if (!s || !g_vt.pools[v->pi].img) return;
    vt_slot_give(&g_vt.pools[v->pi].sl, s - 1, now_fence()); pt_set(i, e & VT_PT_MAPPED);
}
static void enqueue(uint32_t i) {   // a tile asked for (once until it is taken)
    if (g_vt.queued[i >> 3] & 1u << (i & 7)) return;
    if (g_vt.nq == g_vt.capq) { uint32_t n = g_vt.capq ? 2 * g_vt.capq : 1024; if (n > 65536) return; uint32_t *q = realloc(g_vt.q, n * sizeof *q); if (!q) return; g_vt.q = q; g_vt.capq = n; }
    g_vt.q[g_vt.nq++] = i; g_vt.queued[i >> 3] |= (uint8_t)(1u << (i & 7));
}
static void piece_job(uint32_t i, uint32_t pieces) {   // a slot's border pieces to build again at the next look (they read the guest memory as it is then)
    for (uint32_t k = 0; k < g_vt.npj; k++) if (g_vt.pj[k].i == i) { g_vt.pj[k].pieces |= pieces; return; }
    if (g_vt.npj == g_vt.cappj) { uint32_t n = g_vt.cappj ? 2 * g_vt.cappj : 256; vt_piece_job *q = realloc(g_vt.pj, n * sizeof *q); if (!q) return; g_vt.pj = q; g_vt.cappj = n; }
    g_vt.pj[g_vt.npj++] = (vt_piece_job){i, pieces};
}
// A tile's mapping changed (changed 1: it lies in a range the game mapped or unmapped; 0: a rescan, which says only its state): a slot it has holds what the game mapped there
// before, so it goes back and the tile is asked for again where it is still mapped; the neighbours' borders that face it are built again. u: the dense image's unit (g_vt.mu held)
static void vt_unit(vt_tex *v, uint32_t u, int m, int changed) {
    uint32_t y = u / v->dn_per_layer, k = u % v->dn_per_layer; if (k >= v->tail_at || y >= v->g.layers) return;   // (the tail stays the dense image's)
    uint32_t i = v->pt_base + y * v->g.per_layer + k, e = pt_get(i), was = (e & VT_PT_MAPPED) != 0;
    if (!changed && was == (uint32_t)(m != 0)) return;
    if (e & VT_PT_SLOT) { slot_drop(v, i); if (m) enqueue(i); else g_vt.n_unmap++; }
    pt_set(i, (pt_get(i) & ~VT_PT_MAPPED) | (m ? VT_PT_MAPPED : 0));
    uint32_t ly, l, tx, ty; vt_place_of(&v->g, y * v->g.per_layer + k, &ly, &l, &tx, &ty);
    for (int dy = -1; dy <= 1; dy++) for (int dx = -1; dx <= 1; dx++) {
        if (!dx && !dy) continue;
        uint32_t nx = (uint32_t)(((int)tx + dx + (int)v->g.nx[l]) % (int)v->g.nx[l]), ny = (uint32_t)(((int)ty + dy + (int)v->g.ny[l]) % (int)v->g.ny[l]);
        if (nx == tx && ny == ty) continue;   // (a level one tile across or down: its own border, built with it)
        uint32_t j = v->pt_base + vt_place(&v->g, ly, l, nx, ny); if (pt_get(j) & VT_PT_SLOT) piece_job(j, vt_facing(dx, dy));
    }
}
static vt_tex *vt_of(const vkimg_t *im) { if (!im) return NULL; for (int k = 0; k < g_vt.ntex; k++) if (g_vt.tex[k]->im == im) return g_vt.tex[k]; return NULL; }
void vk_vt_ranges(const vkimg_t *im, const uint8_t *mapped, const uint64_t (*r)[2], int n) {   // the tiles of the guest ranges r, as gfx_vk_tex.c sp_range walks them
    pthread_mutex_lock(&g_vt.mu); vt_tex *v = vt_of(im);
    for (int k = 0; v && mapped && k < n; k++) {
        uint64_t end = v->mem + (uint64_t)v->g.layers * v->g.layer_size, lo = r[k][0] > v->mem ? r[k][0] : v->mem, hi = r[k][1] < end ? r[k][1] : end;
        for (uint64_t o = (lo - v->mem) & ~(uint64_t)(VT_TILE - 1); lo < hi && v->mem + o < hi; o += VT_TILE) {
            uint32_t y = (uint32_t)(o / v->g.layer_size); uint64_t in = o % v->g.layer_size; uint32_t l = 0;
            if (in < v->g.loff[0]) continue;
            while (l + 1 < v->g.ntiled && in >= v->g.loff[l + 1]) l++;
            uint64_t t = (in - v->g.loff[l]) / VT_TILE; if (t >= (uint64_t)v->g.nx[l] * v->g.ny[l]) continue;   // (the tail, past the tiled levels)
            uint32_t u = y * v->dn_per_layer + v->g.loff_t[l] + (uint32_t)t; vt_unit(v, u, mapped[u], 1);
        }
    }
    pthread_mutex_unlock(&g_vt.mu);
}
void vk_vt_rescan(const vkimg_t *im, const uint8_t *mapped) {   // every unit's state again (the pool's log ran past the last look)
    pthread_mutex_lock(&g_vt.mu); vt_tex *v = vt_of(im);
    for (uint32_t y = 0; v && mapped && y < v->g.layers; y++) for (uint32_t i = 0; i < v->tail_at; i++) vt_unit(v, y * v->dn_per_layer + i, mapped[(size_t)y * v->dn_per_layer + i], 0);
    pthread_mutex_unlock(&g_vt.mu);
}
void vk_vt_detach(const vkimg_t *im) {
    pthread_mutex_lock(&g_vt.mu); vt_tex *v = vt_of(im); if (!v) { pthread_mutex_unlock(&g_vt.mu); return; }
    for (uint32_t i = v->pt_base; i < v->pt_base + v->pt_n; i++) { slot_drop(v, i); pt_set(i, 0); }
    int k = 0; while (k < g_vt.ntex && g_vt.tex[k] != v) k++;
    if (k < g_vt.ntex) { memmove(&g_vt.tex[k], &g_vt.tex[k + 1], (size_t)(g_vt.ntex - k - 1) * sizeof *g_vt.tex); g_vt.ntex--; }
    pthread_mutex_unlock(&g_vt.mu);
    free(v);
}
void vk_vt_lost(void) {   // the dropped command buffers' uploads and page table copies did not happen: every slot goes back, every word is written again
    pthread_mutex_lock(&g_vt.mu);
    for (int k = 0; k < g_vt.ntex; k++) { vt_tex *v = g_vt.tex[k]; for (uint32_t i = v->pt_base; i < v->pt_base + v->pt_n; i++) slot_drop(v, i); }
    for (uint32_t i = 0; g_vt.dirty && i < g_vt.pt_hi; i++) if (!(g_vt.dirty[i >> 6] >> (i & 63) & 1)) { g_vt.dirty[i >> 6] |= 1ull << (i & 63); g_vt.ndirty++; }
    g_vt.cleared = 0; pthread_mutex_unlock(&g_vt.mu);
}

// ---------------- uploads (in the upload command buffer: before every draw of the submission) ----------------
typedef struct { vt_tex *v; } rd_ctx;
static void rd_guest(void *c, uint64_t off, void *dst, uint32_t n) { vt_tex *v = ((rd_ctx *)c)->v; nvn_sparse_copy(v->pool, dst, v->mem + off, n); }
static int mp_shadow(void *c, uint32_t place) { vt_tex *v = ((rd_ctx *)c)->v; return (pt_get(v->pt_base + place) & VT_PT_MAPPED) != 0; }
static void slot_copy(vt_tex *v, uint32_t slot, uint32_t pieces, uint32_t place) {   // pieces of a slot built from the guest memory and copied into the pool
    vt_pool *p = &g_vt.pools[v->pi]; size_t pitch = (size_t)(p->nbx + 2) * p->bpb, sz = pitch * (p->nby + 2); VkBuffer sb; VkDeviceSize so;
    uint32_t y, l, tx, ty, r[4]; vt_place_of(&v->g, place, &y, &l, &tx, &ty); vt_piece_rect(pieces, p->nbx, p->nby, r);
    int whole = r[0] == 0 && r[1] == 0 && r[2] == p->nbx + 2 && r[3] == p->nby + 2; rd_ctx c = {v};
    if (!whole && !g_vt.slotbuf && !(g_vt.slotbuf = malloc(sz))) return;
    uint8_t *dst = whole ? vk_scratch(sz, 16, &sb, &so) : g_vt.slotbuf;   // a piece: built in the slot's layout, then its rectangle alone copied to the scratch memory
    if (pieces & 1) nvn_sparse_protect_tile(v->pool, v->mem + vt_tile_off(&v->g, y, l, tx, ty), VT_TILE);   // (protected before it is read, as gfx_vk_tex.c sp_tile's: a later CPU write is seen)
    if (vt_slot_build(&v->g, &v->d, y, l, tx, ty, pieces, v->zero, rd_guest, mp_shadow, &c, g_vt.tile, dst)) { VKRED_ONCE(171, "VT: a tile of nvn format %#x not decoded", v->d.format); return; }
    size_t rw = (size_t)(r[2] - r[0]) * p->bpb;
    if (!whole) { uint8_t *q = vk_scratch(rw * (r[3] - r[1]), 16, &sb, &so); for (uint32_t j = r[1]; j < r[3]; j++) memcpy(q + (j - r[1]) * rw, dst + j * pitch + (size_t)r[0] * p->bpb, rw); }
    VkCommandBuffer cb = vk_upload_cmd();
    if (!p->ready) {   // the pool's first use: GENERAL, as every image the backend keeps
        VkImageMemoryBarrier b = {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER, NULL, 0, VK_ACCESS_TRANSFER_WRITE_BIT, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL, VK_QUEUE_FAMILY_IGNORED,
                                  VK_QUEUE_FAMILY_IGNORED, p->img, {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, p->n}};
        vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, NULL, 0, NULL, 1, &b); p->ready = 1; }
    VkBufferImageCopy rg = {so, whole ? (p->nbx + 2) * p->bw : (r[2] - r[0]) * p->bw, (r[3] - r[1]) * p->bh, {VK_IMAGE_ASPECT_COLOR_BIT, 0, slot, 1},
                            {(int32_t)(r[0] * p->bw), (int32_t)(r[1] * p->bh), 0}, {(r[2] - r[0]) * p->bw, (r[3] - r[1]) * p->bh, 1}};
    vkCmdCopyBufferToImage(cb, sb, p->img, VK_IMAGE_LAYOUT_GENERAL, 1, &rg);
}
static void upload(uint32_t i) {   // word i asked for: a slot, the tile and its border
    vt_tex *v = owner(i); uint32_t e = pt_get(i);
    if (!v || (e & VT_PT_SLOT) || !(e & VT_PT_MAPPED) || !g_vt.pools[v->pi].img) return;
    vt_slots *sl = &g_vt.pools[v->pi].sl; int s = vt_slot_take(sl, vk_fence_done(), i, g_vt.sub);
    if (s < 0) { g_vt.n_nofree++; enqueue(i); return; }
    sl->s[s].gen = nvn_sparse_gen(); sl->s[s].stamp = nvn_dirty_now();   // (before the tile is read: a write after it is seen)
    slot_copy(v, (uint32_t)s, 0x1ff, i - v->pt_base); pt_set(i, (uint32_t)(s + 1) | VT_PT_MAPPED); g_vt.n_up++;
}
static void pt_flush(void) {   // the words changed since the last flush, copied after the tiles they name
    if (!g_vt.ptbuf) return;
    if (!g_vt.cleared) { vkCmdFillBuffer(vk_upload_cmd(), g_vt.ptbuf, 0, VK_WHOLE_SIZE, 0); g_vt.cleared = 1; }   // (the GPU's copy starts as zeros: no word, no slot, before any draw reads it)
    if (!g_vt.ndirty) return;
    VkCommandBuffer cb = vk_upload_cmd();
    VkMemoryBarrier mb = {VK_STRUCTURE_TYPE_MEMORY_BARRIER, NULL, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_TRANSFER_WRITE_BIT};
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &mb, 0, NULL, 0, NULL);   // (a slot's upload lands before a word names it)
    enum { RUNS = 1024 }; static uint32_t run[RUNS][2]; static VkBufferCopy rg[RUNS];
    uint32_t next, nr = vt_runs(g_vt.dirty, g_vt.ptcap, g_vt.dnext < g_vt.ptcap ? g_vt.dnext : 0, run, RUNS, &next), words = 0;
    if (nr < RUNS) { uint32_t more = vt_runs(g_vt.dirty, g_vt.ptcap, 0, run + nr, RUNS - nr, &next); nr += more; }   // (from the start again: the runs left behind by an earlier full flush)
    g_vt.dnext = next;
    for (uint32_t k = 0; k < nr; k++) words += run[k][1];
    if (!nr) { g_vt.ndirty = 0; return; }
    VkBuffer sb; VkDeviceSize so; uint32_t *d = vk_scratch((VkDeviceSize)words * 4, 4, &sb, &so), at = 0;
    for (uint32_t k = 0; k < nr; k++) { memcpy(d + at, &g_vt.pt[run[k][0]], (size_t)run[k][1] * 4); rg[k] = (VkBufferCopy){so + (VkDeviceSize)at * 4, (VkDeviceSize)run[k][0] * 4, (VkDeviceSize)run[k][1] * 4}; at += run[k][1]; }
    vkCmdCopyBuffer(cb, sb, g_vt.ptbuf, nr, rg);
    g_vt.ndirty = g_vt.ndirty > words ? g_vt.ndirty - words : 0; g_vt.n_pt += words;
}
// The CLOCK's callbacks (vt_clock): a slot's word goes on probation, or loses its slot
static void cb_prob(void *c, uint32_t i) { (void)c; pt_set(i, pt_get(i) | VT_PT_PROB); }
static void cb_evict(void *c, uint32_t i) { (void)c; pt_set(i, pt_get(i) & VT_PT_MAPPED); g_vt.n_evict++; }
// Once a submission, at its first VT look: the requests of the submissions the GPU is done with, the CLOCK at a new present, this submission's request region, the uploads
static void sub_begin(void) {
    extern uint64_t g_tex_frame; uint64_t done = vk_fence_done(), f = now_fence(); uint32_t words = (g_vt.pt_hi + 31) / 32;
    g_vt.sub = vk_sub_serial();
    for (int r = 0; r < NFB; r++) {
        if (!g_vt.regf[r] || g_vt.regf[r] > done) continue;
        uint32_t *w = g_vt.fb + (size_t)r * g_vt.rw;
        for (uint32_t k = 0; k < words; k++) { uint32_t m = w[k]; if (!m) continue; w[k] = 0;
            for (; m; m &= m - 1) { uint32_t i = k * 32 + (uint32_t)__builtin_ctz(m), e = pt_get(i); vt_tex *v;
                if (e & VT_PT_SLOT) { if ((v = owner(i))) { vt_slot *s = &g_vt.pools[v->pi].sl.s[(e & VT_PT_SLOT) - 1]; s->prob = 0; s->last = g_vt.sub; } pt_set(i, e & ~VT_PT_PROB); g_vt.n_touch++; }
                else if (e & VT_PT_MAPPED) { enqueue(i); g_vt.n_req++; } } }
        g_vt.regf[r] = 0;
    }
    if (g_tex_frame != g_vt.frame) { g_vt.frame = g_tex_frame;
        for (int k = 0; k < g_vt.npool; k++) { vt_pool *p = &g_vt.pools[k]; if (p->img) vt_clock(&p->sl, p->n / 32 ? p->n / 32 : 1, p->n / 16 + g_vt.uploads, g_vt.sub, f, cb_prob, cb_evict, NULL); } }
    if (nvn_dirty_sparse_writes() != g_vt.wseen && g_tex_frame - g_vt.wframe >= 60) {   // a page of a sparse pool written since: a slot whose tile it is holds old texels
        g_vt.wseen = nvn_dirty_sparse_writes(); g_vt.wframe = g_tex_frame;
        for (int k = 0; k < g_vt.ntex; k++) { vt_tex *v = g_vt.tex[k]; vt_pool *p = &g_vt.pools[v->pi]; if (!p->img) continue;
            for (uint32_t j = 0; j < p->n; j++) { vt_slot *sl = &p->sl.s[j]; if (sl->place == VT_FREE || sl->place < v->pt_base || sl->place >= v->pt_base + v->pt_n) continue;
                uint32_t y, l, tx, ty, i = sl->place; vt_place_of(&v->g, i - v->pt_base, &y, &l, &tx, &ty);
                if (nvn_sparse_since(v->pool, v->mem + vt_tile_off(&v->g, y, l, tx, ty), VT_TILE, sl->gen, sl->stamp)) { slot_drop(v, i); enqueue(i); g_vt.n_cpu++; } } } }
    int r = (int)(f % NFB); g_vt.reg = g_vt.regf[r] ? -1 : r; if (g_vt.reg >= 0) g_vt.regf[r] = f;   // (-1: no region free, this submission's draws ask for nothing)
    for (uint32_t k = 0; k < g_vt.npj; k++) { vt_tex *v = owner(g_vt.pj[k].i); uint32_t e = pt_get(g_vt.pj[k].i);   // border pieces first: their slots are in use
        if (v && (e & VT_PT_SLOT)) for (uint32_t b = 2; b <= 256; b <<= 1) if (g_vt.pj[k].pieces & b) { slot_copy(v, (e & VT_PT_SLOT) - 1, b, g_vt.pj[k].i - v->pt_base); g_vt.n_piece++; } }
    g_vt.npj = 0;
    uint32_t n = g_vt.nq < g_vt.uploads ? g_vt.nq : g_vt.uploads;
    if (n) { uint32_t take[256], m = n < 256 ? n : 256; memcpy(take, g_vt.q, m * sizeof *take); memmove(g_vt.q, g_vt.q + m, (g_vt.nq - m) * sizeof *g_vt.q); g_vt.nq -= m;
        for (uint32_t k = 0; k < m; k++) g_vt.queued[take[k] >> 3] &= (uint8_t)~(1u << (take[k] & 7));
        for (uint32_t k = 0; k < m; k++) upload(take[k]); }
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts); double t = (double)ts.tv_sec + ts.tv_nsec * 1e-9;
    if (t - g_vt.t_log >= 10) { g_vt.t_log = t; char b[256]; int o = 0;
        for (int k = 0; k < g_vt.npool && o < 200; k++) { vt_pool *p = &g_vt.pools[k]; o += snprintf(b + o, sizeof b - (size_t)o, " format %d %u of %u", p->fmt, p->n - p->sl.nfree - p->sl.nlate, p->n); }
        VKLOG("VT: %d textures, slots used%s; tiles uploaded %llu, asked for %llu, kept by a touch %llu, taken back %llu, unmapped %llu, without a free slot %llu, border pieces %llu, "
              "page table words copied %llu, written by the CPU %llu; %u waiting", g_vt.ntex, b, (unsigned long long)g_vt.n_up, (unsigned long long)g_vt.n_req, (unsigned long long)g_vt.n_touch,
              (unsigned long long)g_vt.n_evict, (unsigned long long)g_vt.n_unmap, (unsigned long long)g_vt.n_nofree, (unsigned long long)g_vt.n_piece, (unsigned long long)g_vt.n_pt,
              (unsigned long long)g_vt.n_cpu, g_vt.nq); }
}
void vk_vt_look(const vkimg_t *im) {   // gfx_vk_tex.c dense_look, once a submission for each texture with a dense image looked up in it
    if (!g_vt.ntex || !vk_sub_open()) return;
    pthread_mutex_lock(&g_vt.mu);
    if (!vt_of(im)) { pthread_mutex_unlock(&g_vt.mu); return; }
    if (g_vt.sub != vk_sub_serial()) sub_begin();
    pt_flush();
    pthread_mutex_unlock(&g_vt.mu);
}
void vk_vt_sub_end(VkCommandBuffer cb) {   // the request writes of the submission's shaders visible to the host once its fence has passed
    if (!g_vt.fbbuf || g_vt.reg < 0 || g_vt.sub != vk_sub_serial()) return;
    VkMemoryBarrier mb = {VK_STRUCTURE_TYPE_MEMORY_BARRIER, NULL, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_HOST_READ_BIT};
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &mb, 0, NULL, 0, NULL);
}

// ---------------- descriptors (gfx_vk_exec.c stage_set) ----------------
static VkSampler pool_sampler(int magf, int minlin) {   // the game's filters on one level of a slot, clamped to it (its border holds what the filter reaches past the tile)
    int k = (magf ? 1 : 0) | (minlin ? 2 : 0); if (g_vt.samp[k]) return g_vt.samp[k];
    VkSamplerCreateInfo ci = {VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO, NULL, 0, magf ? VK_FILTER_LINEAR : VK_FILTER_NEAREST, minlin ? VK_FILTER_LINEAR : VK_FILTER_NEAREST, VK_SAMPLER_MIPMAP_MODE_NEAREST,
                              VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE, VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE, VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE, 0, VK_FALSE, 1, VK_FALSE, VK_COMPARE_OP_ALWAYS, 0, 0.25f,
                              VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK, VK_FALSE};
    if (vkCreateSampler(V.dev, &ci, NULL, &g_vt.samp[k])) g_vt.samp[k] = VK_NULL_HANDLE;
    return g_vt.samp[k];
}
void vk_vt_pool_desc(const vt_bound_t *c, VkDescriptorImageInfo *ii) {
    *ii = (VkDescriptorImageInfo){VK_NULL_HANDLE, VK_NULL_HANDLE, VK_IMAGE_LAYOUT_GENERAL}; samp_par sp;
    if (vk_vt_mode() == 1 && c && c->im && c->shift && vk_samp_par_at(c->sid, c->sseq, &sp)) {
        pthread_mutex_lock(&g_vt.mu); vt_tex *v = vt_of(c->im);
        if (v && pool_ready(&g_vt.pools[v->pi])) { ii->imageView = pool_view(&g_vt.pools[v->pi], c->vfmt); ii->sampler = pool_sampler(sp.magf, sp.minf == 1 || sp.minf == 3 || sp.minf == 5); }
        pthread_mutex_unlock(&g_vt.mu); }
    if (!ii->imageView || !ii->sampler) { ii->imageView = vk_dummy_view(1, 1, 0, 0, 0); ii->sampler = vk_dummy_sampler(); }
}
void vk_vt_ctl_desc(const vt_bound_t *c, int n, VkDescriptorBufferInfo *bi) {
    uint32_t e[VT_CTL_BYTES / 4]; memset(e, 0, sizeof e); int mode = vk_vt_mode(); samp_par sp[VT_NC]; int ok[VT_NC];
    for (int k = 0; k < n && k < VT_NC; k++) ok[k] = mode && c[k].im && c[k].im->dn && c[k].shift && c[k].shift <= c[k].im->lv0 && vk_samp_par_at(c[k].sid, c[k].sseq, &sp[k]);   // (the registry's lock, before ours)
    pthread_mutex_lock(&g_vt.mu);
    if (g_vt.reg >= 0 && g_vt.sub == vk_sub_serial()) { e[0] = (uint32_t)g_vt.reg * g_vt.rw; e[1] = (g_vt.pt_hi + 31) / 32; }
    for (int k = 0; k < n && k < VT_NC; k++) {
        if (!ok[k]) continue;
        const vkimg_t *im = c[k].im; const samp_par *q = &sp[k];
        vt_tex *v = mode == 1 ? vt_of(im) : NULL; uint32_t base = im->lv0 - c[k].shift, lv = im->levels + im->lv0 - base;   // (the view's levels: from the texture's level base on)
        if (v && !g_vt.pools[v->pi].img) v = NULL;   // (its pool not made: the dense image's LOD fix alone)
        float lo = q->lod_min > 0 ? q->lod_min : 0, hi = q->minf <= 1 ? 0.25f : q->lod_max; if (hi > (float)(lv - 1)) hi = (float)(lv - 1); if (lo > hi) lo = hi;   // (the view's levels, as the sampler's)
        vt_ctl_view(&e[4 + 12 * k], v ? VT_MODE_VT : VT_MODE_DENSE, v ? &v->g : NULL, v ? v->pt_base : 0, base, c[k].ly0, c[k].nly, c[k].shift, q->minf == 4 || q->minf == 5, vt_wrap_code(q->ws),
                    vt_wrap_code(q->wt), lo, hi);
    }
    pthread_mutex_unlock(&g_vt.mu);
    // one copy per distinct block in the submission (as gfx_vk_exec.c drv_shared), so the descriptor sets naming it can be the same
    enum { CC = 256 }; static struct { uint64_t h, sub; VkBuffer b; VkDeviceSize o; } cc[CC];
    uint64_t h = 0xcbf29ce484222325ull; for (size_t i = 0; i < sizeof e / 4; i++) { h ^= e[i]; h *= 0x100000001b3ull; }
    unsigned k = (unsigned)(h % CC); uint64_t sub = vk_sub_serial();
    if (cc[k].sub == sub && cc[k].h == h) { *bi = (VkDescriptorBufferInfo){cc[k].b, cc[k].o, VT_CTL_BYTES}; return; }
    VkBuffer b; VkDeviceSize o; uint8_t *p = vk_scratch(VT_CTL_BYTES, V.props.limits.minUniformBufferOffsetAlignment, &b, &o); memcpy(p, e, VT_CTL_BYTES);
    cc[k].h = h; cc[k].sub = sub; cc[k].b = b; cc[k].o = o; *bi = (VkDescriptorBufferInfo){b, o, VT_CTL_BYTES};
}
void vk_vt_pt_desc(VkDescriptorBufferInfo *bi) { *bi = g_vt.ptbuf ? (VkDescriptorBufferInfo){g_vt.ptbuf, 0, VK_WHOLE_SIZE} : (VkDescriptorBufferInfo){vk_zero_buffer(), 0, 65536}; }
void vk_vt_fb_desc(VkDescriptorBufferInfo *bi) { *bi = g_vt.fbbuf ? (VkDescriptorBufferInfo){g_vt.fbbuf, 0, VK_WHOLE_SIZE} : (VkDescriptorBufferInfo){vk_zero_buffer(), 0, 65536}; }
