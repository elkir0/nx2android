// Android port, stage 1: a null graphics backend behind the replay interface of nvn_mtl.h (the Metal backend's functions).
// Nothing is drawn. Everything the CPU can observe of the GPU's work is honoured, so a game's own logic runs as on the device:
//   · a submission completes at once: events are written / added, counters get {value, timestamp} (19.2 MHz guest ticks), FenceSync objects take the
//     submission's fence and every fence counts as passed;
//   · buffer copies, fills and inline uniform updates are done in order with plain memory operations (pool memory is the guest's own memory);
//   · samples-passed counters report many samples (everything visible) rather than 0 (everything culled).
// Selected with DM_GFX=null (the default until the Vulkan backend exists); the Vulkan backend (gfx_vk.c) implements the same functions.
#include "dm.h"
#include "nvn_obj.h"
#include "nvn_mtl.h"
#include "nvn_timestamp.h"
#include <pthread.h>
#include <stdatomic.h>
#include <string.h>
#include <time.h>

uint64_t guest_ns(void);
static uint64_t gpu_ticks(void) { uint64_t g = guest_ns(); if (!g) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); g = (uint64_t)t.tv_sec * 1000000000ull + (uint64_t)t.tv_nsec; } return nvn_timestamp_from_ns(g); }   // the GPU's 614.4 MHz (nvn_timestamp.h)

// ---------------- memory pools: GPU address = CPU address ----------------
typedef struct { uint64_t lo, hi; } pool_ent;
static pool_ent *g_pools; static int g_npool, g_poolcap;
static pthread_rwlock_t g_pool_mu = PTHREAD_RWLOCK_INITIALIZER;
void mtl_pool_add(const void *mem, uint64_t size) {
    pthread_rwlock_wrlock(&g_pool_mu);
    if (g_npool == g_poolcap) { g_poolcap = g_poolcap ? 2 * g_poolcap : 256; g_pools = realloc(g_pools, g_poolcap * sizeof *g_pools); }
    g_pools[g_npool++] = (pool_ent){(uint64_t)(uintptr_t)mem, (uint64_t)(uintptr_t)mem + size};
    pthread_rwlock_unlock(&g_pool_mu);
}
void mtl_pool_del(const void *mem) {
    pthread_rwlock_wrlock(&g_pool_mu);
    for (int i = 0; i < g_npool; i++) if (g_pools[i].lo == (uint64_t)(uintptr_t)mem) { g_pools[i] = g_pools[--g_npool]; break; }
    pthread_rwlock_unlock(&g_pool_mu);
}
int mtl_addr(uint64_t addr, uint64_t len, void **buf, uint64_t *off) {
    int ok = 0;
    pthread_rwlock_rdlock(&g_pool_mu);
    for (int i = 0; i < g_npool && !ok; i++) if (addr >= g_pools[i].lo && addr + len <= g_pools[i].hi) { if (buf) *buf = (void *)(uintptr_t)g_pools[i].lo; if (off) *off = addr - g_pools[i].lo; ok = 1; }
    pthread_rwlock_unlock(&g_pool_mu);
    return ok;
}
void mtl_alias_drop_pool(const void *pool) { (void)pool; }
// a virtual pool's range (Android: no CPU alias of it, hle_nvn.c nvn_virt_xlat): read and written at the physical memory behind it; NULL: left alone
int nvn_virt_xlat(uint64_t a, uint64_t n, uint64_t *out);
static void *nptr(uint64_t a, uint64_t n) { uint64_t t; int r = nvn_virt_xlat(a, n, &t); return r == 1 ? (void *)(uintptr_t)t : r ? NULL : (void *)(uintptr_t)a; }

// ---------------- submissions ----------------
typedef struct { int kind; volatile uint32_t *val; uint32_t mode, v; uint64_t addr; uint32_t type; } post_op;   // kind 1 event, 2 counter
static post_op *g_post; static int g_npost, g_postcap;
static uint64_t g_fence;
static void post(post_op o) { if (g_npost == g_postcap) { g_postcap = g_postcap ? 2 * g_postcap : 64; g_post = realloc(g_post, g_postcap * sizeof *g_post); } g_post[g_npost++] = o; }

void mtl_init(void) {}
void mtl_sub_begin(void) { g_npost = 0; }
void mtl_sub_index(const uint64_t *h, int n) { (void)h; (void)n; }
void mtl_exec_rest(const uint64_t *h, int n) { (void)h; (void)n; }
void mtl_sub_copy(uint64_t src, uint64_t dst, uint64_t n) { void *d = n ? nptr(dst, n) : NULL, *s = n ? nptr(src, n) : NULL; if (d && s) memmove(d, s, n); }
void mtl_sub_fill(uint64_t dst, uint64_t n, uint32_t value) { uint8_t *d = n ? nptr(dst, n) : NULL; for (uint64_t i = 0; d && i + 4 <= n; i += 4) memcpy(d + i, &value, 4); }
void mtl_sub_event(volatile uint32_t *val, uint32_t mode, uint32_t v) { post((post_op){1, val, mode, v, 0, 0}); }
void mtl_sub_counter(uint64_t addr, uint32_t type) { post((post_op){2, NULL, 0, 0, addr, type}); }

void mtl_exec(const uint8_t *p) {
    for (;;) {
        op_hdr h; memcpy(&h, p, sizeof h);
        const uint8_t *a = p + 8;
        switch (h.op) {
        case OP_END: return;
        case OP_JUMP: { const uint8_t *t; memcpy(&t, a, 8); p = t; continue; }
        case OP_CALL: { uint64_t t; memcpy(&t, a, 8); mtl_exec((const uint8_t *)(uintptr_t)t); break; }
        case OP_SIGNAL_EVENT: { volatile uint32_t *val; uint32_t m, v; memcpy(&val, a, 8); memcpy(&m, a + 8, 4); memcpy(&v, a + 12, 4); mtl_sub_event(val, m, v); break; }
        case OP_REPORT_COUNTER: { uint64_t t, ad; memcpy(&t, a, 8); memcpy(&ad, a + 8, 8); mtl_sub_counter(ad, (uint32_t)t); break; }
        case OP_FENCE_SYNC: { sync_t *s; memcpy(&s, a, 8); s->fence = g_fence + 1; s->fenced = 1; break; }
        case OP_CLEAR_BUF: { const p_clear_buf *cb = (const p_clear_buf *)a; mtl_sub_fill(cb->addr, cb->size, cb->value); break; }
        case OP_COPY_B2B: { uint64_t s_, d, n; memcpy(&s_, a, 8); memcpy(&d, a + 8, 8); memcpy(&n, a + 16, 8); mtl_sub_copy(s_, d, n); break; }
        case OP_UPDATE_UB: { uint64_t d, n; memcpy(&d, a, 8); memcpy(&n, a + 8, 8); void *at = n ? nptr(d, n) : NULL; if (at) memcpy(at, a + 16, n); break; }
        default: break;   // drawing, binding, texture copies and clears: no visible effect without a GPU
        }
        p += h.size;
    }
}
uint64_t mtl_sub_commit(void) {
    uint64_t now = gpu_ticks();
    for (int i = 0; i < g_npost; i++) {
        post_op *o = &g_post[i];
        if (o->kind == 1) { volatile uint32_t *val = nptr((uint64_t)(uintptr_t)o->val, 4); if (val) { if (o->mode == 1) *val += o->v; else *val = o->v; } }
        else { uint64_t rec[2] = {o->type == 0 || o->type == 0x10 ? now : o->type == NVN_COUNTER_TYPE_SAMPLES_PASSED ? (1u << 20) : 0, now}; void *at = nptr(o->addr, sizeof rec); if (at) memcpy(at, rec, sizeof rec); }
    }
    g_npost = 0;
    return ++g_fence;
}
uint64_t mtl_last_fence(void) { return g_fence; }
int mtl_wait(uint64_t fence, uint64_t timeout_ns) { (void)fence; (void)timeout_ns; return 1; }
void mtl_finish(void) {}
void mtl_drain(void) {}

// ---------------- presentation: frames are counted, not shown ----------------
static uint64_t g_presents; static double g_pres_t[64];
uint64_t mtl_present(const tex_t *t, double period, const int crop[4]) {
    (void)t; (void)period; (void)crop;
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts); g_pres_t[g_presents % 64] = ts.tv_sec + ts.tv_nsec / 1e9;
    return ++g_presents;
}
void mtl_present_wait(uint64_t tk) { (void)tk; }
double mtl_presented_time(uint64_t n) { return n && n <= g_presents && g_presents - n < 64 ? g_pres_t[n % 64] : 0; }
uint32_t mtl_present_qos(void) { return 0; }
int mtl_link_active(void) { return 0; }
int mtl_link_wait_tick(int vsyncs, uint64_t timeout_ns) { (void)vsyncs; (void)timeout_ns; return 0; }
int mtl_has_window(void) { return 0; }
int mtl_display_hz(void) { return 60; }
int mtl_fps_target(void) { return DM_GAME_FPS_TARGET > 0 ? DM_GAME_FPS_TARGET : 60; }
void mtl_xv_window_tex(uint64_t serial) { (void)serial; }
void mtl_set_frame(uint64_t f) { (void)f; }
void mtl_set_depth_zero(int on) { (void)on; }
void mtl_set_origin_lower_left(int on) { (void)on; }

// ---------------- resources ----------------
void mtl_tex_init(const tex_t *t) { (void)t; }
void mtl_tex_release(tex_t *t) { (void)t; }
void mtl_tex_cpu_written(const tex_t *t) { (void)t; }
void mtl_reg_tex(int tid, const tex_t *t, const view_t *v) { (void)tid; (void)t; (void)v; }
void mtl_reg_samp(int sid, const samp_par *sp) { (void)sid; (void)sp; }
void mtl_set_tex_pool_n(int n) { (void)n; }
void mtl_prog_release(prog_t *p) { (void)p; }
// The Switch 2 port's entry points (nvn_mtl.h, nvn_obj.h): nothing is drawn here, so none has an effect beyond what the CPU can see
void mtl_prepare(void) {}
int mtl_wait_acq(uint64_t fence, uint64_t timeout_ns) { return mtl_wait(fence, timeout_ns); }
void mtl_queue_activate(uint64_t serial) { (void)serial; }
void mtl_queue_forget(uint64_t serial) { (void)serial; }
void mtl_set_queue_origin_lower_left(int on) { (void)on; }
void mtl_pool_gpu_written(const void *pool) { pool_t *p = (pool_t *)pool; if (p) atomic_store(&p->gpu_written, 1); }
void mtl_sp_phys_gone(const void *mem, uint64_t size) { (void)mem; (void)size; }
void mtl_reg_tex_gone(int id) { (void)id; }
void mtl_reg_image(int id, const tex_t *t, const view_t *v) { (void)id; (void)t; (void)v; }
void mtl_prog_prefetch(const prog_t *p) { (void)p; }
void mtl_prog_forget(const prog_t *p) { (void)p; }
unsigned mtl_cpu_tex_count(void) { return 0; }
uint64_t mtl_pkg_edge_skips(void) { return 0; }
int mtl_tex_gpu_owned(uint64_t serial) { (void)serial; return 0; }
uint64_t mtl_samp_seq(void) { return 0; }
void mtl_present_flush(void) {}
void mtl_present_test_stall(uint64_t ns) { (void)ns; }
int mtl_warm_wait(int timeout_ms) { (void)timeout_ms; return 0; }
void mtl_pso_line(char *o, size_t n) { if (n) snprintf(o, n, "none (the null device)"); }
void *mtl_device(void) { return NULL; }
void *mtl_device_classic(void) { return NULL; }
uint64_t mtl_ext_tex(void *tex) { (void)tex; return 0; }
uint64_t mtl_present_ext(uint64_t serial, double period, void *ev, uint64_t ev_val, const int crop[4]) { (void)serial; (void)period; (void)ev; (void)ev_val; (void)crop; return 0; }
void mtl_read_texels(const tex_t *t, unsigned level, unsigned layer0, const NVNcopyRegion *r, uint8_t *dst, size_t row, size_t img) {
    (void)t; (void)level; (void)layer0; (void)img;
    if (r && dst) for (int z = 0; z < (r->depth ? r->depth : 1); z++) for (int y = 0; y < r->height; y++) memset(dst + (size_t)z * img + (size_t)y * row, 0, row);
}

