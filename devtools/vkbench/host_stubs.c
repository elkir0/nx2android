// vkbench (devtools/vkbench): the host's side of the replay (dm/host, dm/gfx/nvn/hle_nvn.c) as far as the Vulkan backend calls it, for the macOS build of the benchmark.
// Most are stand-ins (no guest, no crash handler); two are the host's own code: the aliased virtual pools' translation (nvn_virt_xlat, cut from hle_nvn.c by vkbench.py
// into vkb_vpool.inc, as tests/test_vk_vpool.py does) and the texture layout (nvn_tex.c, compiled as it is). The dirty tracker's stamp moves as the bench says (bench.c),
// which is what decides how often the replay looks at a packaged texture's pages again
#define _GNU_SOURCE
#include <errno.h>
#include <fcntl.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>
#include "dm.h"
#include "nvn_obj.h"
#include <android/hardware_buffer.h>
#include "gfx_vk.h"

// ---- the log ----
// LOG (dm.h) writes stderr; the bench keeps it to a file of its own (VKB_LOG, bench.c), so this file needs nothing

// ---- hardware buffers: POSIX shared memory, whose descriptor goes over the socket as a dma-buf's does ----
struct AHardwareBuffer { int fd; uint64_t size; };
int AHardwareBuffer_allocate(const AHardwareBuffer_Desc *d, AHardwareBuffer **out) {
    static _Atomic int n; char name[64]; snprintf(name, sizeof name, "/vkb.%d.%d", (int)getpid(), atomic_fetch_add(&n, 1));
    int fd = shm_open(name, O_RDWR | O_CREAT | O_EXCL, 0600); if (fd < 0) return -1; shm_unlink(name);
    if (ftruncate(fd, (off_t)d->width)) { close(fd); return -1; }
    AHardwareBuffer *a = calloc(1, sizeof *a); a->fd = fd; a->size = d->width; *out = a; return 0;
}
void AHardwareBuffer_release(AHardwareBuffer *a) { if (a) { close(a->fd); free(a); } }
void AHardwareBuffer_acquire(AHardwareBuffer *a) { (void)a; }
void AHardwareBuffer_describe(const AHardwareBuffer *a, AHardwareBuffer_Desc *d) { memset(d, 0, sizeof *d); d->width = (uint32_t)a->size; d->height = d->layers = 1; d->format = AHARDWAREBUFFER_FORMAT_BLOB; }
int AHardwareBuffer_sendHandleToUnixSocket(const AHardwareBuffer *a, int sock) {
    char data = 0, ctl[CMSG_SPACE(sizeof(int))]; struct iovec iov = {&data, 1}; struct msghdr m; memset(&m, 0, sizeof m); memset(ctl, 0, sizeof ctl);
    m.msg_iov = &iov; m.msg_iovlen = 1; m.msg_control = ctl; m.msg_controllen = sizeof ctl;
    struct cmsghdr *c = CMSG_FIRSTHDR(&m); c->cmsg_level = SOL_SOCKET; c->cmsg_type = SCM_RIGHTS; c->cmsg_len = CMSG_LEN(sizeof(int)); memcpy(CMSG_DATA(c), &a->fd, sizeof(int));
    return sendmsg(sock, &m, 0) == 1 ? 0 : -1;
}

// ---- the crash handler, faults, the loader ----
void crash_add_note_hook(void (*fn)(void)) { (void)fn; }
void crash_handlers_check(const char *when) { (void)when; }
void crash_thread_name(unsigned t, char *b, size_t n) { (void)t; snprintf(b, n, "?"); }
void dm_fault_hook_register(int (*fn)(void *addr, int is_write, void *uc)) { (void)fn; }
void dm_guest_write_prep_register(void (*fn)(void *addr, size_t len)) { (void)fn; }
int dm_sdk_ounce(void) { return 1; }   // Mario Kart World: NVN2
uint64_t g_nvn_discards[2];
void (*g_nvn_on_shader)(const uint8_t *, const void *, uint64_t, uint32_t);
uint64_t guest_ns(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return (uint64_t)t.tv_sec * 1000000000ull + (uint64_t)t.tv_nsec; }
void nvn_ctl_lens(const void *control, uint32_t *nctl, uint32_t *ncode) { (void)control; *nctl = 0x880; *ncode = 0; }
int nvn_ft_written(const tex_t *t) { (void)t; return 0; }

// ---- the guest's alias region: where the bench's guest memory lies (vk_unreadable_at answers from it at once, as on the phone for the game's graphics memory) ----
uint8_t *g_vkb_alias_base; size_t g_vkb_alias_size;
uint64_t svc_alias_mapped(void) { return 1; }
uint64_t svc_alias_unmapped_at(uint64_t a, uint64_t n) { (void)a; return n; }
void svc_alias_range(uint8_t **base, size_t *size) { *base = g_vkb_alias_base; *size = g_vkb_alias_size; }

// ---- the dirty tracker (dm/gfx/nvn/nvn_dirty.c): its stamp, moved by the bench (the game's writes to protected pages); edge: the next nvn_dirty_since_span answers "written" ----
_Atomic uint64_t g_vkb_dirty_now = 1; _Atomic int g_vkb_dirty_edge;
uint64_t nvn_dirty_now(void) { return atomic_load_explicit(&g_vkb_dirty_now, memory_order_relaxed); }
void nvn_dirty_protect(const void *addr, size_t len) { (void)addr; (void)len; }
void nvn_dirty_protect_sparse(const void *addr, size_t len) { (void)addr; (void)len; }
uint64_t nvn_dirty_sparse_writes(void) { return 0; }
void nvn_dirty_release(const void *addr, size_t len) { (void)addr; (void)len; }
uint64_t nvn_dirty_released(void) { return 0; }
int nvn_dirty_runs(void) { return 0; }
int nvn_dirty_defrag(int (*remap)(void *mem, size_t len)) { (void)remap; return 0; }
int nvn_dirty_since(const void *addr, size_t len, uint64_t stamp) { (void)addr; (void)len; (void)stamp; return 0; }
static int g_span_dummy;
const void *nvn_dirty_span(const void *addr, size_t len) { (void)addr; (void)len; return &g_span_dummy; }
int nvn_dirty_since_span(const void *span, const void *addr, size_t len, uint64_t stamp) { (void)span; (void)addr; (void)len; (void)stamp;
    int e = atomic_load_explicit(&g_vkb_dirty_edge, memory_order_relaxed); if (e > 0) { atomic_fetch_sub(&g_vkb_dirty_edge, 1); return 1; } return 0; }
int nvn_dirty_tracked(const void *addr, size_t len) { (void)addr; (void)len; return 0; }

// ---- sparse pools: none in the benchmark ----
int nvn_sparse_pool(const void *pool) { (void)pool; return 0; }
uint64_t nvn_sparse_gen(void) { return 0; }
uint64_t nvn_sparse_copy(const void *pool, void *dst, uint64_t a, uint64_t n) { (void)pool; (void)a; memset(dst, 0, (size_t)n); return n; }
int nvn_sparse_since(const void *pool, uint64_t a, uint64_t n, uint64_t gen, uint64_t stamp) { (void)pool; (void)a; (void)n; (void)gen; (void)stamp; return 0; }
void nvn_sparse_protect(const void *pool, uint64_t a, uint64_t n) { (void)pool; (void)a; (void)n; }
int nvn_sparse_mapped(const void *pool, uint64_t a, uint64_t n) { (void)pool; (void)a; (void)n; return 1; }
int nvn_sparse_tile_state(const void *pool, uint64_t a, uint64_t n, uint64_t gen, uint64_t stamp) { (void)pool; (void)a; (void)n; (void)gen; (void)stamp; return 0; }
const uint8_t *nvn_sparse_tile_phys(const void *pool, uint64_t a) { (void)pool; (void)a; return NULL; }
uint64_t nvn_sparse_log_seq(const void *pool) { (void)pool; return 0; }
int nvn_sparse_log_read(const void *pool, uint64_t *seq, uint64_t (*out)[2], int max) { (void)pool; (void)seq; (void)out; (void)max; return 0; }
void nvn_sparse_protect_tile(const void *pool, uint64_t a, uint64_t n) { (void)pool; (void)a; (void)n; }

// ---- programs: the bench's prog_t carries its stage keys after it (hle_nvn.c keeps them in a table by program: prog_keys) ----
typedef struct { prog_t p; uint64_t key[6]; } vkb_prog;
int prog_key(const prog_t *p, unsigned st, uint8_t key[8]) {
    if (st >= p->nstage) return 0;
    // hle_nvn.c pk_get: the program's keys behind a mutex and a hash map, per stage asked
    static pthread_mutex_t mu = PTHREAD_MUTEX_INITIALIZER; pthread_mutex_lock(&mu);
    uint64_t k = ((const vkb_prog *)p)->key[st]; pthread_mutex_unlock(&mu);
    for (int i = 0; i < 8; i++) key[i] = (uint8_t)(k >> (56 - 8 * i));   // (gfx_vk_exec.c key64 reads it big-endian)
    return 1;
}

// ---- aliased virtual pools: hle_nvn.c's own code (vkb_vpool.inc), and the bench's way in ----
#define VP_ALIAS 0
#include "vkb_vpool.inc"
void vkb_valias_add(const void *pool, const uint8_t *res, uint64_t len, uint64_t size, const uint8_t *phys) { pthread_mutex_lock(&g_vp_mu); valias_add(pool, res, len, size, phys); pthread_mutex_unlock(&g_vp_mu); }
// a virtual pool with a page table (the sparse kind), counted in nvn_virt_xlat's scan as MKW's ~30 are
void vkb_vpool_add(uint8_t *base, uint64_t len) {
    struct vpool *v = calloc(1, sizeof *v); v->base = base; v->len = len; v->size = len; v->pt = calloc((len + (2u << 20) - 1) >> 21, sizeof *v->pt);
    pthread_mutex_lock(&g_vp_mu); if (g_nvp == g_cvp) g_vp = realloc(g_vp, (size_t)(g_cvp = g_cvp ? 2 * g_cvp : 64) * sizeof *g_vp); g_vp[g_nvp++] = v; atomic_store(&g_vp_any, 1); pthread_mutex_unlock(&g_vp_mu);
}

// ---- the backend's other files (gfx_vk_vt.c: shader-side virtual texturing; gfx_vk_fsr.c: the NDSR swap), which the bench does not build: weak stand-ins, off (no VT bindings in
// the bench's bundle, no upscaler keys), so a variant whose sources call them links; a variant that builds them would give its own ----
typedef struct { const vkimg_t *im; int sid; uint64_t sseq; VkFormat vfmt; uint32_t shift, ly0, nly; } vkb_vt_bound;
#define VKB_WEAK __attribute__((weak))
VKB_WEAK int vk_vt_attach(vkimg_t *im, const tex_t *t, const vkfmt_t *f, const uint8_t *mapped, uint32_t per_layer, uint32_t tail_at) { (void)im; (void)t; (void)f; (void)mapped; (void)per_layer; (void)tail_at; return 0; }
VKB_WEAK void vk_vt_ranges(const vkimg_t *im, const uint8_t *mapped, const uint64_t (*r)[2], int n) { (void)im; (void)mapped; (void)r; (void)n; }
VKB_WEAK void vk_vt_rescan(const vkimg_t *im, const uint8_t *mapped) { (void)im; (void)mapped; }
VKB_WEAK void vk_vt_look(const vkimg_t *im) { (void)im; }
VKB_WEAK void vk_vt_detach(const vkimg_t *im) { (void)im; }
VKB_WEAK void vk_vt_lost(void) {}
VKB_WEAK void vk_vt_sub_end(VkCommandBuffer cb) { (void)cb; }
VKB_WEAK void vk_vt_pool_desc(const vkb_vt_bound *c, VkDescriptorImageInfo *ii) { (void)c; memset(ii, 0, sizeof *ii); }
VKB_WEAK void vk_vt_ctl_desc(const vkb_vt_bound *c, int n, VkDescriptorBufferInfo *bi) { (void)c; (void)n; memset(bi, 0, sizeof *bi); }
VKB_WEAK void vk_vt_pt_desc(VkDescriptorBufferInfo *bi) { memset(bi, 0, sizeof *bi); }
VKB_WEAK void vk_vt_fb_desc(VkDescriptorBufferInfo *bi) { memset(bi, 0, sizeof *bi); }
VKB_WEAK int vk_ndsr_wanted(uint64_t key) { (void)key; return 0; }
VKB_WEAK int vk_ndsr_swap(VkCommandBuffer cb, const void *cs, const void *in, const void *out, const uint32_t push[8]) { (void)cb; (void)cs; (void)in; (void)out; (void)push; return 0; }
