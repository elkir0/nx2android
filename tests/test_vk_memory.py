"""Memory of the Vulkan backend on a phone, without one: the lazy placement of memory pools (gfx_vk_core.c "memory pools": chunks placed
when the GPU is first given an address in them, large pools by parts, the swap that keeps concurrent CPU writes), the first level of a
dense image of a sparse texture (gfx_vk_tex.c dense_*: base, cap, rebase on synthetic page tables) and the large image blocks
(lblk_take / lblk_give). The pool code is gfx_vk_core.c's own, built on the host with the Android and Vulkan calls it makes stubbed:
a file stands in for each hardware buffer, so the swap maps real pages over real memory while threads write it."""
from pathlib import Path
import os
import shutil
import subprocess
import tempfile
import unittest

DM = Path(__file__).resolve().parents[1] / 'nxnative/dm'
CORE = DM / 'android/gfx_vk_core.c'
TEX = DM / 'android/gfx_vk_tex.c'
HDR = DM / 'android/gfx_vk.h'
VKH = DM / 'third_party/vulkan-headers/include'


def region(src, start, end):
    a = src.index(start)
    return src[a:src.index(end, a)]


POOL_STUBS = r'''
#define _GNU_SOURCE
#include <assert.h>
#include <errno.h>
#include <pthread.h>
#include <signal.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/socket.h>
#include <time.h>
#include <unistd.h>
#define VK_NO_PROTOTYPES
#include <vulkan/vulkan_core.h>
#ifndef SOCK_CLOEXEC
#define SOCK_CLOEXEC 0
#endif
#undef SOCK_SEQPACKET
#define SOCK_SEQPACKET SOCK_STREAM
#define die(...) (fprintf(stderr, __VA_ARGS__), abort())
static long g_steps;   // the pool table's search steps (gfx_vk_core.c POOL_STEP)
#define POOL_STEP() (g_steps++)
static char g_log[1 << 16]; static size_t g_nlog;
#define LOG(...) (g_nlog += (size_t)snprintf(g_log + g_nlog, sizeof g_log - g_nlog > 1 ? sizeof g_log - g_nlog : 1, __VA_ARGS__), g_nlog >= sizeof g_log - 2 ? (g_nlog = 0) : 0, g_log[g_nlog++] = '\n', g_log[g_nlog] = 0)
#define VKLOG(...) LOG(__VA_ARGS__)
#define VKCHK(x) do { if ((x) < 0) die("%s", #x); } while (0)
static struct { VkDevice dev; int has_xfb; } V;
static void vk_init(void) {}
static uint32_t vk_mem_type(uint32_t bits, VkMemoryPropertyFlags want, VkMemoryPropertyFlags avoid) { (void)bits; (void)want; (void)avoid; return 0; }
static int g_freed; static void vk_defer_free_buffer(VkBuffer b, VkDeviceMemory m) { (void)b; (void)m; g_freed++; }
// hardware buffers: a file each (its descriptor sent over the socket as the dma-buf's)
typedef struct AHardwareBuffer { int fd; uint64_t size; } AHardwareBuffer;
typedef struct { uint32_t width, height, layers, format; uint64_t usage; uint32_t stride, rfu0; uint64_t rfu1; } AHardwareBuffer_Desc;
enum { AHARDWAREBUFFER_FORMAT_BLOB = 0x21 };
#define AHARDWAREBUFFER_USAGE_CPU_READ_OFTEN 3ull
#define AHARDWAREBUFFER_USAGE_CPU_WRITE_OFTEN (3ull << 4)
#define AHARDWAREBUFFER_USAGE_GPU_DATA_BUFFER (1ull << 24)
static int g_ahb_fail, g_ahb_live; static AHardwareBuffer *g_last_ahb;
static int AHardwareBuffer_allocate(const AHardwareBuffer_Desc *d, AHardwareBuffer **out) {
    if (g_ahb_fail) return -1;
    char path[] = "/tmp/nx2a_ahb_XXXXXX"; int fd = mkstemp(path); if (fd < 0) return -1; unlink(path);
    if (ftruncate(fd, d->width)) { close(fd); return -1; }
    AHardwareBuffer *a = calloc(1, sizeof *a); a->fd = fd; a->size = d->width; *out = g_last_ahb = a; g_ahb_live++; return 0;
}
static int AHardwareBuffer_sendHandleToUnixSocket(const AHardwareBuffer *a, int sock) {
    char data = 0, ctl[CMSG_SPACE(sizeof(int))]; struct iovec iov = {&data, 1}; struct msghdr m; memset(&m, 0, sizeof m); memset(ctl, 0, sizeof ctl);
    m.msg_iov = &iov; m.msg_iovlen = 1; m.msg_control = ctl; m.msg_controllen = sizeof ctl;
    struct cmsghdr *c = CMSG_FIRSTHDR(&m); c->cmsg_level = SOL_SOCKET; c->cmsg_type = SCM_RIGHTS; c->cmsg_len = CMSG_LEN(sizeof(int)); memcpy(CMSG_DATA(c), &a->fd, sizeof(int));
    return sendmsg(sock, &m, 0) == 1 ? 0 : -1;
}
static void AHardwareBuffer_release(AHardwareBuffer *a) { if (a) { close(a->fd); free(a); g_ahb_live--; } }
typedef struct { VkStructureType sType; void *pNext; VkDeviceSize allocationSize; uint32_t memoryTypeBits; } VkAndroidHardwareBufferPropertiesANDROID;
typedef struct { VkStructureType sType; const void *pNext; struct AHardwareBuffer *buffer; } VkImportAndroidHardwareBufferInfoANDROID;
// Vulkan: handles counted, mapped memory of the size allocated
static uint64_t g_handles = 0x1000; static int g_import_fail;
typedef struct { VkDeviceSize size; void *p; } fake_mem;
static VkResult vkGetAndroidHardwareBufferPropertiesANDROID(VkDevice d, const AHardwareBuffer *a, VkAndroidHardwareBufferPropertiesANDROID *p) { (void)d; p->allocationSize = a->size; p->memoryTypeBits = 1; return g_import_fail ? VK_ERROR_INVALID_EXTERNAL_HANDLE : VK_SUCCESS; }
static VkResult vkCreateBuffer(VkDevice d, const VkBufferCreateInfo *ci, const VkAllocationCallbacks *a, VkBuffer *b) { (void)d; (void)ci; (void)a; *b = (VkBuffer)(uintptr_t)++g_handles; return VK_SUCCESS; }
static void vkDestroyBuffer(VkDevice d, VkBuffer b, const VkAllocationCallbacks *a) { (void)d; (void)b; (void)a; }
static void vkGetBufferMemoryRequirements(VkDevice d, VkBuffer b, VkMemoryRequirements *r) { (void)d; (void)b; r->size = 1 << 20; r->alignment = 256; r->memoryTypeBits = 1; }
static VkResult vkAllocateMemory(VkDevice d, const VkMemoryAllocateInfo *ai, const VkAllocationCallbacks *a, VkDeviceMemory *m) { (void)d; (void)a; fake_mem *f = calloc(1, sizeof *f); f->size = ai->allocationSize; *m = (VkDeviceMemory)(uintptr_t)f; return VK_SUCCESS; }
static VkResult vkBindBufferMemory(VkDevice d, VkBuffer b, VkDeviceMemory m, VkDeviceSize o) { (void)d; (void)b; (void)m; (void)o; return VK_SUCCESS; }
static VkResult vkMapMemory(VkDevice d, VkDeviceMemory m, VkDeviceSize o, VkDeviceSize n, VkMemoryMapFlags f, void **p) { (void)d; (void)o; (void)n; (void)f; fake_mem *x = (fake_mem *)(uintptr_t)m; if (!x->p) x->p = calloc(1, x->size); *p = x->p; return VK_SUCCESS; }
// nvn_dirty: a range the test says is tracked; pages it protects come out read-only, and the fault handler below lets a write through as nvn_dirty does
static uint8_t *g_tr_lo, *g_tr_hi; static int g_protects, g_since_calls;
static const void *nvn_dirty_span(const void *a, size_t n) { return (const uint8_t *)a >= g_tr_lo && (const uint8_t *)a + n <= g_tr_hi ? (const void *)g_tr_lo : NULL; }
static int nvn_dirty_tracked(const void *a, size_t n) { return nvn_dirty_span(a, n) != NULL; }
static int nvn_dirty_since_span(const void *s, const void *a, size_t n, uint64_t st) { (void)s; (void)a; (void)n; (void)st; g_since_calls++; return 0; }
static uint64_t nvn_dirty_now(void) { return 1; }
static void nvn_dirty_protect(const void *a, size_t n) { g_protects++; mprotect((void *)a, n, PROT_READ); }
static void vk_shadow_drop(uint64_t lo, uint64_t n) { (void)lo; (void)n; }   // (gfx_vk_shadow.c: a range placed is no longer the shadow file's)
int vk_keep_buf_named(VkBuffer b) { (void)b; return 1; }   // (gfx_vk_exec.c: a kept descriptor set names the buffer of a pool that goes)
uint64_t vk_sub_serial(void);
'''

POOL_MAIN = r'''
// the host's alias region bitmap (hle_svc.c): a page the test says the game has not mapped
static uint64_t g_alias_hole;
uint64_t svc_alias_mapped(void) { return 0; }
uint64_t svc_alias_unmapped_at(uint64_t a, uint64_t n) { return g_alias_hole && g_alias_hole >= a && g_alias_hole < a + n ? g_alias_hole - a : n; }
void svc_alias_range(uint8_t **base, size_t *size) { *base = NULL; *size = 0; }
uint64_t vk_sub_serial(void) { return 1; }
static uint64_t g_xv_lo, g_xv_len, g_xv_to;   // a virtual pool's range the stub translates (hle_nvn.c nvn_virt_xlat: Android has no CPU alias of it)
int nvn_virt_xlat(uint64_t a, uint64_t n, uint64_t *out) {
    if (!g_xv_len || a < g_xv_lo || a >= g_xv_lo + g_xv_len) return 0;
    if (a + n > g_xv_lo + g_xv_len) return -1;
    *out = g_xv_to + (a - g_xv_lo); return 1;
}
static void on_fault(int sig, siginfo_t *si, void *uc) {
    if (swap_fault(si->si_addr, 1, uc)) return;   // held back by a swap, made again over the new pages
    uint8_t *a = si->si_addr; long pg = getpagesize();
    if (a >= g_tr_lo && a < g_tr_hi) { mprotect((void *)((uintptr_t)a & ~(uintptr_t)(pg - 1)), (size_t)pg, PROT_READ | PROT_WRITE); return; }   // (nvn_dirty's release of a tracked page)
    signal(sig, SIG_DFL); raise(sig);
}
static uint8_t *pool_mem(size_t n) { uint8_t *p = mmap(NULL, n, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0); assert(p != MAP_FAILED); return p; }
static void fill(uint8_t *p, size_t n, uint32_t seed) { for (size_t i = 0; i < n; i += 4) { uint32_t v = (uint32_t)i * 2654435761u ^ seed; memcpy(p + i, &v, 4); } }
static int same_at(const uint8_t *p, size_t off, size_t n, uint32_t seed) { for (size_t i = off; i < off + n; i += 4) { uint32_t v = (uint32_t)i * 2654435761u ^ seed, w; memcpy(&w, p + i, 4); if (v != w) return 0; } return 1; }   // (offsets from the pool's start, as fill)
static int same(const uint8_t *p, size_t n, uint32_t seed) { return same_at(p, 0, n, seed); }
static int zero_copy_of(const vkpool_t *e, const uint8_t *at) {   // the bytes at `at` are the entry's hardware buffer's (written through the pointer, read back through its file)
    AHardwareBuffer *a = e->ahb; uint32_t v = 0x5eed1234, r = 0; memcpy((void *)at, &v, 4);
    return pread(a->fd, &r, 4, (off_t)((uint64_t)(uintptr_t)at - e->lo)) == 4 && r == v;
}
enum { NW = 4, WWORDS = (2 << 20) / 4 };
static uint8_t *g_wbase; static _Atomic int g_stop; static uint32_t g_nw[NW];
static void *writer(void *arg) {   // each writer writes the words of its own 2 MB once each, in order and slowly (the writes span the swap), word j = j + 1: a write lost leaves a 0
    int t = (int)(intptr_t)arg; volatile uint32_t *w = (volatile uint32_t *)(g_wbase + (size_t)t * (4u << 20));
    uint32_t j = 0; for (; j < WWORDS && !atomic_load(&g_stop); j++) { w[j] = j + 1; for (volatile int d = 0; d < 300; d++) {} }
    g_nw[t] = j; return NULL;
}
// The table past the 4,096 entries it once had (Mario Kart World's run 5 died there, "too many memory pools"): 6,000 pools put in out of order, each found by one search of about
// log2(n) steps with the gaps between them found empty, an entry's address kept as the table grows, a third of them removed and their entries reused, a pool placed by parts among
// them (a part, and the pool where no part holds the range: two searches), and a pool over another's memory, which the scan covers while it lives
static int lg2(int n) { int k = 0; while ((1 << k) < n + 1) k++; return k; }
static int many(void) {
    enum { NP = 6000 }; const uint64_t KB = 1024, MB = 1 << 20, ST = 128 * KB;
    uint8_t *ar = pool_mem((size_t)(NP * ST)); uint64_t A = (uint64_t)(uintptr_t)ar; vkpool_t *first = NULL;
    for (int k = 0; k < NP; k++) { int i = (int)((k * 7919L) % NP); mtl_pool_add(ar + (size_t)i * ST, 64 * KB); if (k == 999) first = vk_pool_of(A + (uint64_t)((999 * 7919L) % NP) * ST + 8, 8); }
    assert(g_npool == NP && g_npool > 4096 && first);
    assert(vk_pool_of(first->lo + 8, 8) == first && first->hi == first->lo + 64 * KB);   // (made before the table grew twice: the same entry)
    long worst = 0;
    for (int i = 0; i < NP; i++) { uint64_t lo = A + (uint64_t)i * ST; long s0 = g_steps;
        vkpool_t *e = vk_pool_of(lo + 100 + (uint64_t)(i % 7) * 4096, 16); assert(e && e->lo == lo && e->hi == lo + 64 * KB && e->state == POOL_UNPLACED);
        assert(!vk_pool_of(lo + 64 * KB + 8, 16) && !vk_pool_of(lo + 64 * KB - 8, 16)); if (g_steps - s0 > worst) worst = g_steps - s0; }   // (a gap; across the pool's end)
    printf("many: %d entries, three lookups took at most %ld search steps\n", g_npool, worst); assert(worst <= 3 * (lg2(g_npool) + 1));
    for (int i = 0; i < NP; i += 3) mtl_pool_del(ar + (size_t)i * ST);
    assert(g_npool == NP - (NP + 2) / 3 && g_pool_spare);
    for (int i = 0; i < NP; i++) { uint64_t lo = A + (uint64_t)i * ST; vkpool_t *e = vk_pool_of(lo + 200, 8); assert(i % 3 ? e && e->lo == lo : !e); }
    for (int i = 0; i < NP; i += 3) mtl_pool_add(ar + (size_t)i * ST + 32 * KB, 32 * KB);   // made again over half the memory: the removed entries reused
    assert(g_npool == NP && !g_pool_spare);
    for (int i = 0; i < NP; i++) { uint64_t lo = A + (uint64_t)i * ST; vkpool_t *e = vk_pool_of(lo + 32 * KB + 200, 8); assert(e && e->lo == (i % 3 ? lo : lo + 32 * KB));
        if (i % 3 == 0) assert(!vk_pool_of(lo + 200, 8)); }
    // a pool placed by parts among them: its parts, and the pool itself where no part holds the range
    uint8_t *q = pool_mem(128 * MB); mtl_pool_add(q, 128 * MB); uint64_t Q = (uint64_t)(uintptr_t)q;
    vkpool_t *e = vk_pool_gpu(Q + 5 * MB + 10, 16), *f = vk_pool_gpu(Q, 16), *g = vk_pool_gpu(Q + 127 * MB + 5, 16);
    assert(e && e->part && e->lo == Q + 5 * MB && f && f->part && f->lo == Q && g && g->part && g->lo == Q + 127 * MB && g_ahb_live == 3);
    long s0 = g_steps; vkpool_t *pp = vk_pool_of(Q + 70 * MB, 4); long s = g_steps - s0;
    assert(pp && pp->state == POOL_PARTS && pp->lo == Q && s <= 2 * (lg2(g_npool) + 1));
    assert(vk_pool_of(Q + 5 * MB + 100, 4) == e && vk_pool_of(Q + 100, 4) == f && vk_pool_of(Q + 6 * MB - 2, 4) == pp);   // (the part at the pool's start sorts after the pool; across a part's end: the pool)
    assert(vk_pool_gpu(Q + 5 * MB + 100, 4) == e && !vk_pool_gpu(Q + 6 * MB - 2, 4) && g_ahb_live == 3);   // (across a part placed and a granule not: no buffer)
    // a pool over another's memory: the search lands on it where it does not hold the range, and the scan finds the other
    mtl_pool_add(ar + ST + 16 * KB, 16 * KB); assert(g_novl == 1 && strstr(g_log, "lies over the memory of pool"));
    vkpool_t *o = vk_pool_of(A + ST + 48 * KB, 4); assert(o && o->lo == A + ST && vk_pool_of(A + ST + 20 * KB, 4));
    mtl_pool_del(ar + ST + 16 * KB); assert(g_novl == 0 && vk_pool_of(A + ST + 48 * KB, 4) == o);
    mtl_pool_del(q); assert(!vk_pool_of(Q + 70 * MB, 4) && g_ahb_live == 0 && g_npool == NP);
    puts("many ok"); return 0;
}
int main(int argc, char **argv) {
    int eager = argc > 1 && !strcmp(argv[1], "eager");
    setenv("DM_VK_POOL_PART_MB", "64", 1); setenv("DM_VK_POOL_GRANULE_MB", "1", 1); if (eager) setenv("DM_VK_POOL_LAZY", "0", 1);
    struct sigaction sa; memset(&sa, 0, sizeof sa); sa.sa_sigaction = on_fault; sa.sa_flags = SA_SIGINFO; sigaction(SIGSEGV, &sa, NULL); sigaction(SIGBUS, &sa, NULL);
    if (argc > 1 && !strcmp(argv[1], "many")) return many();
    const size_t MB = 1 << 20; VkBuffer b; VkDeviceSize o;
    if (eager) {   // DM_VK_POOL_LAZY=0: placed when made, as before
        uint8_t *p = pool_mem(32 * MB); fill(p, 32 * MB, 7); mtl_pool_add(p, 32 * MB);
        vkpool_t *e = vk_pool_of((uint64_t)(uintptr_t)p, 1); assert(e && e->state == POOL_PLACED && e->zero_copy && same(p, 32 * MB, 7));
        uint8_t *q = pool_mem(96 * MB); mtl_pool_add(q, 96 * MB);   // (no parts: every chunk placed)
        e = vk_pool_of((uint64_t)(uintptr_t)q + 90 * MB, 1); assert(e && e->state == POOL_PLACED && e->hi - e->lo == 96 * MB);
        puts("eager ok"); return 0;
    }
    // a pool under the part size: one chunk, placed whole when the GPU is first given an address in it, its content kept
    uint8_t *p = pool_mem(32 * MB); fill(p, 32 * MB, 1); mtl_pool_add(p, 32 * MB);
    vkpool_t *e = vk_pool_of((uint64_t)(uintptr_t)p + 100, 16); assert(e && e->state == POOL_UNPLACED && !e->buf && g_ahb_live == 0);
    void *mb; uint64_t mo; assert(mtl_addr((uint64_t)(uintptr_t)p + 5, 8, &mb, &mo) && mo == 5 && e->state == POOL_UNPLACED);   // (a CPU lookup places nothing)
    assert(vk_buf_at((uint64_t)(uintptr_t)p + 100, 16, &b, &o) && b && o == 100 && g_ahb_live == 1);
    e = vk_pool_of((uint64_t)(uintptr_t)p + 100, 16); assert(e->state == POOL_PLACED && e->zero_copy && e->lo == (uint64_t)(uintptr_t)p && e->hi - e->lo == 32 * MB);
    assert(same(p, 32 * MB, 1) && zero_copy_of(e, p + 4096));
    VkBuffer b2; assert(vk_buf_at((uint64_t)(uintptr_t)p + 31 * MB, 4, &b2, &o) && b2 == b && o == 31 * MB && g_ahb_live == 1);   // the same chunk: no second placement
    // a pool of the part size or more: granules of 1 MB, the ones a range touches placed as one part
    uint8_t *q = pool_mem(128 * MB); fill(q, 128 * MB, 2); mtl_pool_add(q, 128 * MB); uint64_t Q = (uint64_t)(uintptr_t)q;
    e = vk_pool_of(Q + 70 * MB, 1); assert(e && e->state == POOL_PARTS && e->ngran == 128 && g_ahb_live == 1);
    assert(vk_buf_at(Q + 5 * MB + MB / 2, 3 * MB, &b, &o) && o == MB / 2 && g_ahb_live == 2);
    e = vk_pool_gpu(Q + 5 * MB + MB / 2, 3 * MB); assert(e && e->state == POOL_PLACED && e->lo == Q + 5 * MB && e->hi == Q + 9 * MB && e->buf == b && g_ahb_live == 2);   // granules 5-8, placed once
    assert(vk_buf_at(Q + 6 * MB, MB, &b2, &o) && b2 == e->buf && o == MB && g_ahb_live == 2);   // inside it: found, nothing placed
    assert(same(q, 128 * MB, 2) && zero_copy_of(e, q + 7 * MB));
    assert(!vk_pool_gpu(Q + 8 * MB + MB / 2, MB) && g_ahb_live == 2);   // across the part and a granule not placed: no buffer holds it, reported
    assert(strstr(g_log, "lies across a part placed before and one not"));
    vkpool_t *e2 = vk_pool_gpu(Q + 9 * MB + 100, 100); assert(e2 && e2->lo == Q + 9 * MB && e2->hi == Q + 10 * MB && g_ahb_live == 3);   // the next granule: a part of its own
    assert(vk_pool_gpu(Q + 127 * MB, MB) && g_ahb_live == 4);   // the pool's last granule
    e = vk_pool_of(Q + 70 * MB, 4); assert(e && e->state == POOL_PARTS && e->placed[5] && e->placed[8] && e->placed[9] && !e->placed[10] && e->placed[127]);
    assert(!vk_pool_gpu(Q + 20 * MB, 0x10000001ull) && g_ahb_live == 4);   // past the pool: nothing
    assert(!vk_pool_gpu(Q + 120 * MB, 7 * MB + 1) && g_ahb_live == 4);   // (touches the placed last granule)
    // a hardware buffer refused: a copy instead (a shadow the submissions sync)
    uint8_t *c = pool_mem(4 * MB); fill(c, 4 * MB, 3); mtl_pool_add(c, 4 * MB); g_ahb_fail = 1;
    e = vk_pool_gpu((uint64_t)(uintptr_t)c, 64); g_ahb_fail = 0; assert(e && e->state == POOL_PLACED && !e->zero_copy && e->shadow && same(c, 4 * MB, 3));
    // the swap while threads write the range (a chunk under the part size, placed whole): no write is lost, some were held back
    uint8_t *w = pool_mem(32 * MB); mtl_pool_add(w, 32 * MB); g_wbase = w; pthread_t th[NW];
    for (int t = 0; t < NW; t++) pthread_create(&th[t], NULL, writer, (void *)(intptr_t)t);
    usleep(20000); uint64_t waits0 = atomic_load(&g_swap_waits);
    assert(vk_buf_at((uint64_t)(uintptr_t)w, 1, &b, &o));
    usleep(20000); atomic_store(&g_stop, 1); for (int t = 0; t < NW; t++) pthread_join(th[t], NULL);
    long lost = 0, words = 0;
    for (int t = 0; t < NW; t++) { const uint32_t *v = (const uint32_t *)(w + (size_t)t * (4u << 20)); words += g_nw[t]; for (uint32_t j = 0; j < g_nw[t]; j++) lost += v[j] != j + 1; }
    printf("swap: %ld words written by %d threads around it, %ld lost\n", words, NW, lost);
    e = vk_pool_of((uint64_t)(uintptr_t)w, 1); assert(!lost && e->state == POOL_PLACED && e->zero_copy && e->hi - e->lo == 32 * MB && zero_copy_of(e, w + 20 * MB) && zero_copy_of(e, w + 4096));
    printf("swap: %llu writes held back\n", (unsigned long long)(atomic_load(&g_swap_waits) - waits0)); assert(atomic_load(&g_swap_waits) > waits0);
    // a range nvn_dirty tracks: its pages come out protected through it, and its fault path lets a write through
    uint8_t *tr = pool_mem(8 * MB); fill(tr, 8 * MB, 4); mtl_pool_add(tr, 8 * MB); g_tr_lo = tr; g_tr_hi = tr + 8 * MB; int pr = g_protects;
    assert(vk_buf_at((uint64_t)(uintptr_t)tr, 1, &b, &o) && g_protects == pr + 1 && g_since_calls > 0 && same(tr, 8 * MB, 4));
    tr[12345] = 0x77; assert(tr[12345] == 0x77);   // (faulted, released by "nvn_dirty")
    // a virtual pool's range (never dereferenced): a GPU lookup lands on the physical pool behind it, the CPU pointer is the physical memory; what cannot be had in one piece is NULL
    g_xv_lo = 0x7000000000ull; g_xv_len = 4 * MB; g_xv_to = (uint64_t)(uintptr_t)p + 8 * MB;
    assert(vk_buf_at(g_xv_lo + 4096, 256, &b2, &o) && b2 == vk_pool_of((uint64_t)(uintptr_t)p, 1)->buf && o == 8 * MB + 4096);
    assert(vk_cpu_ptr(g_xv_lo + 100, 4) == p + 8 * MB + 100 && !vk_cpu_ptr(g_xv_lo + 4 * MB - 2, 4) && vk_cpu_ptr((uint64_t)(uintptr_t)p + 5, 4) == p + 5);
    uint64_t xa = g_xv_lo + 64; assert(vk_pool_gpu_at(&xa, 16) && xa == (uint64_t)(uintptr_t)p + 8 * MB + 64);
    g_xv_len = 0;
    // the guard: a page of a pool the game has not mapped (PROT_NONE, as the alias region's are), found before anything reads it
    { uint8_t *h = pool_mem(8 * MB); fill(h, 8 * MB, 5); long hp = getpagesize(); assert(!mprotect(h + 3 * MB, (size_t)hp, PROT_NONE)); mtl_pool_add(h, 8 * MB); uint64_t H = (uint64_t)(uintptr_t)h;
      assert(vk_readable(h, 3 * MB) == 1 && vk_readable(h, 8 * MB) == 0 && vk_unreadable_at(h, 8 * MB) == 3 * MB && vk_unreadable_at(h + MB, 4 * MB) == 2 * MB);
      assert(vk_unreadable_at(h + 3 * MB + 100, 8) == 0 && vk_readable(h + 3 * MB + hp, MB) == 1);
      assert(!vk_cpu_ptr(H + 3 * MB + 8, 4) && vk_cpu_ptr(H + 8, 4) == h + 8);
      g_alias_hole = H + 5 * MB; assert(vk_unreadable_at(h + 4 * MB, 2 * MB) == MB && !vk_cpu_ptr(H + 5 * MB, 16)); g_alias_hole = 0;   // (the alias region: the host's bitmap says)
      // placed over the hole: the other pages' bytes kept, the hole's as zeros in the buffer, and the hole still inaccessible over the new pages
      assert(vk_buf_at(H, 1, &b, &o)); e = vk_pool_of(H, 1); assert(e && e->state == POOL_PLACED && e->zero_copy);
      assert(same_at(h, 0, 3 * MB, 5) && same_at(h, 3 * MB + (size_t)hp, 5 * MB - (size_t)hp, 5));
      { uint8_t x; assert(!mem_read(&x, h + 3 * MB, 1)); }
      { AHardwareBuffer *ah = e->ahb; uint8_t z[64]; assert(pread(ah->fd, z, 64, 3 * MB) == 64); for (int i = 0; i < 64; i++) assert(!z[i]); }
      g_alias_hole = H + 3 * MB; assert(vk_unreadable_at(h, 8 * MB) == 3 * MB); g_alias_hole = 0;   // (placed in the alias region: the bitmap still says it)
      uint8_t *buf = malloc(MB); memset(buf, 0xee, MB); assert(copy_readable(buf, h + 3 * MB - MB / 2, MB, (uint64_t)hp, NULL) == (uint64_t)hp);   // a read across it: the hole as zeros
      assert(same_at(buf - (3 * MB - MB / 2), 3 * MB - MB / 2, MB / 2, 5) && !buf[MB / 2] && !buf[MB / 2 + hp - 1]); free(buf); }
    // read-only again in one piece per chunk (nvn_dirty_defrag): the zero-copy chunks mapped again, the granules not placed made read-only
    assert(vk_pool_remap_ro(q, 128 * MB) == 0 && vk_pool_remap_ro(p, 32 * MB) == 0);
    mprotect(q, 128 * MB, PROT_READ | PROT_WRITE); mprotect(p, 32 * MB, PROT_READ | PROT_WRITE);
    // removed: every chunk and part, the content back in the game's own pages
    int live = g_ahb_live; mtl_pool_del(q); assert(!vk_pool_of(Q + 70 * MB, 1) && !vk_pool_of(Q + 6 * MB, 1) && g_ahb_live == live - 3);   // (its three parts)
    assert(same_at(q, 5 * MB, 2 * MB, 2) && same_at(q, 7 * MB + 4096, MB, 2) && same_at(q, 20 * MB, MB, 2) && same_at(q, 127 * MB, MB, 2));   // (7 MB: zero_copy_of's word)
    mtl_pool_del(p); assert(!vk_pool_of((uint64_t)(uintptr_t)p, 1) && g_ahb_live == live - 4);
    puts("lazy ok");
    return 0;
}
'''

DENSE_MAIN = r'''
#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define SP_TILE 0x10000u
'''


class Memory(unittest.TestCase):
    def cc(self):
        cc = shutil.which('cc') or shutil.which('clang')
        if not cc:
            self.skipTest('a C compiler is needed')
        return cc

    def build_run(self, code, name, args=(), extra=()):
        with tempfile.TemporaryDirectory(prefix='nx2a_vk_mem_') as tmp:
            c, exe = Path(tmp) / f'{name}.c', Path(tmp) / name
            c.write_text(code, encoding='utf-8')
            r = subprocess.run([self.cc(), '-std=gnu11', '-O1', '-Wall', '-Werror', '-Wno-unused-function', '-Wno-unused-variable', '-Wno-unused-but-set-variable', '-pthread',
                                *extra, str(c), '-o', str(exe)], capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
            outs = []
            for a in args or [()]:
                r = subprocess.run([str(exe), *a], capture_output=True, text=True, timeout=120)
                self.assertEqual(r.returncode, 0, f'{a}: {r.stdout}{r.stderr}')
                outs.append(r.stdout)
            return outs

    def test_lazy_pools(self):
        if not (VKH / 'vulkan/vulkan_core.h').exists():
            self.skipTest('the vendored Vulkan headers are needed')
        src, hdr = CORE.read_text(encoding='utf-8'), HDR.read_text(encoding='utf-8')
        pool_t = region(hdr, 'typedef struct { uint64_t lo, hi; VkBuffer buf;', 'vkpool_t *vk_pool_of(')
        body = region(src, '// ---------------- memory pools ----------------', 'void mtl_alias_drop_pool(')
        code = POOL_STUBS + pool_t + body + POOL_MAIN
        outs = self.build_run(code, 'pools', args=[(), ('eager',), ('many',)], extra=['-I' + str(VKH)])
        self.assertIn('lazy ok', outs[0])
        self.assertIn('eager ok', outs[1])
        self.assertIn('many ok', outs[2])   # the table past 4,096 entries (no fixed size), searched in about log2(n) steps

    def test_part_span(self):
        body = region(CORE.read_text(encoding='utf-8'), '// ---- Lazy placement: parts ----', '// ---- end of parts ----')
        code = '#include <assert.h>\n#include <stdint.h>\n' + body + r'''
int main(void) {
    uint8_t placed[8] = {0}; uint32_t g0, g1; const uint64_t G = 16;
    assert(part_span(1000, 1128, G, placed, 1000, 1, &g0, &g1) == 0 && g0 == 0 && g1 == 1);
    assert(part_span(1000, 1128, G, placed, 1015, 2, &g0, &g1) == 0 && g0 == 0 && g1 == 2);   // across a granule boundary: both
    assert(part_span(1000, 1128, G, placed, 1016, 16, &g0, &g1) == 0 && g0 == 1 && g1 == 2);   // exactly one
    assert(part_span(1000, 1128, G, placed, 1127, 1, &g0, &g1) == 0 && g0 == 7 && g1 == 8);
    placed[3] = 1;
    assert(part_span(1000, 1128, G, placed, 1040, 20, &g0, &g1) == -1 && part_span(1000, 1128, G, placed, 1047, 2, &g0, &g1) == -1);   // touching a placed granule
    assert(part_span(1000, 1128, G, placed, 1063, 2, &g0, &g1) == -1 && part_span(1000, 1128, G, placed, 1064, 2, &g0, &g1) == 0 && g0 == 4);
    assert(part_span(1000, 1128, G, placed, 999, 2, &g0, &g1) == -2 && part_span(1000, 1128, G, placed, 1127, 2, &g0, &g1) == -2 && part_span(1000, 1128, G, placed, 1000, 0, &g0, &g1) == -2);
    assert(part_span(1000, 1124, G, placed, 1120, 4, &g0, &g1) == 0 && g0 == 7 && g1 == 8);   // a last granule cut by the pool's end
    return 0;
}
'''
        self.build_run(code, 'parts')

    def test_dense_level_choice(self):
        body = region(TEX.read_text(encoding='utf-8'), '// ---- dense images: the level choice', '// ---- end of the dense level choice ----')
        code = DENSE_MAIN + body + r'''
// A synthetic sparse texture: 2048x2048 of 16-byte blocks (BC6H-like: a 64 KB tile holds 256x256 texels), 8 levels, 6 layers, tiles for levels 0-3 (8x8, 4x4, 2x2, 1x1 tiles), the mip tail
// (levels 4-7) in 64 KB; the page table a page bitmap of the texture's offsets
enum { L = 8, Y = 6, NT = 4 }; static const uint32_t TILES[NT] = {64, 16, 4, 1};
static uint8_t pages[(Y * (64 + 16 + 4 + 1 + 1) * 16) + 64];
static int mapped(const void *pool, uint64_t a, uint64_t n) { (void)pool; for (uint64_t p = a >> 12; p < (a + n + 4095) >> 12; p++) if (!pages[p]) return 0; return 1; }
static void map_unit(uint64_t off, int m) { for (int k = 0; k < 16; k++) pages[(off >> 12) + k] = (uint8_t)m; }
static uint64_t lsize, loffs[L];
int main(void) {
    struct vkdense d; memset(&d, 0, sizeof d); d.tl = L; d.layers = Y; d.ntiled = NT; uint32_t k = 0; uint64_t o = 0;
    for (uint32_t l = 0; l < NT; l++) { d.loff[l] = o; loffs[l] = o; d.loff_t[l] = k; k += TILES[l]; o += (uint64_t)TILES[l] * SP_TILE; }
    d.tail_off = o; d.tail_at = k; d.per_layer = k + 1; lsize = o + SP_TILE; d.layer_size = lsize;
    d.mapped = calloc((size_t)Y * d.per_layer, 1); d.cnt = calloc((size_t)L * Y, sizeof *d.cnt);
    // units: their levels and ranges
    uint64_t uo, un; dense_unit_range(&d, 0, &uo, &un); assert(uo == 0 && un == SP_TILE && dense_level_of(&d, 0) == 0 && dense_level_of(&d, 64) == 1 && dense_level_of(&d, 84) == 3);
    dense_unit_range(&d, d.per_layer + d.tail_at, &uo, &un); assert(uo == lsize + d.tail_off && un == SP_TILE && dense_level_of(&d, d.tail_at) == NT);
    // nothing mapped: no base; the first level the cap allows from the smallest up
    dense_scan(&d, 0, ~0ull, mapped, NULL, 0); assert(dense_base(&d) == L);
    uint64_t lv[L]; for (uint32_t l = 0; l < L; l++) { uint32_t w = 2048 >> l; lv[l] = (uint64_t)Y * ((w + 3) / 4) * ((w + 3) / 4) * 16; }   // 24 MB, 6 MB, 1.5 MB, ...
    assert(dense_first(lv, L, L, 64ull << 20, ~0u) == L - 1);   // none mapped: the smallest level
    // the tail of layer 2 mapped: levels 4.. have a page; the base is level 4
    map_unit(2 * lsize + d.tail_off, 1); dense_scan(&d, 2 * lsize + d.tail_off, 2 * lsize + lsize, mapped, NULL, 0);
    assert(dense_base(&d) == NT && d.cnt[NT * Y + 2] == 1 && d.cnt[(L - 1) * Y + 2] == 1 && d.cnt[(NT - 1) * Y + 2] == 0);
    uint32_t cur = dense_first(lv, L, dense_base(&d), 64ull << 20, ~0u); assert(cur == NT);
    // a tile of level 1 in layer 5 mapped (an incremental scan of its range): a finer base, and the cap allows it (levels 1.. take 8 MB): rebase finer
    uint64_t t1 = 5 * lsize + loffs[1] + 3 * SP_TILE; map_unit(t1, 1); dense_scan(&d, t1, t1 + SP_TILE, mapped, NULL, 0);
    assert(dense_base(&d) == 1 && d.cnt[1 * Y + 5] == 1);
    cur = dense_first(lv, L, dense_base(&d), 64ull << 20, cur); assert(cur == 1);
    // level 0 mapped, but the cap (8 MB) keeps the image from level 1: levels 0.. would take 32 MB
    uint64_t t0 = 0 * lsize + loffs[0]; map_unit(t0, 1); dense_scan(&d, t0, t0 + SP_TILE, mapped, NULL, 0);
    assert(dense_base(&d) == 0 && dense_first(lv, L, 0, 8ull << 20, cur) == 1 && dense_first(lv, L, 0, 64ull << 20, cur) == 0);
    // the cap shrinks below the current image: from the level it allows
    assert(dense_first(lv, L, 0, 2ull << 20, 1) == 2);
    // unmapped again: the base coarser by one keeps the image (hysteresis), by two drops its finer levels
    map_unit(t0, 0); dense_scan(&d, 0, lsize, mapped, NULL, 0); assert(dense_base(&d) == 1);
    map_unit(t1, 0); dense_scan(&d, 5 * lsize, 6 * lsize, mapped, NULL, 0); assert(dense_base(&d) == NT && d.cnt[1 * Y + 5] == 0);
    assert(dense_first(lv, L, NT, 64ull << 20, NT - 1) == NT - 1);   // one level coarser: kept
    assert(dense_first(lv, L, NT, 64ull << 20, 1) == NT);           // three: rebased
    assert(dense_first(lv, L, 2, 64ull << 20, 1) == 1 && dense_first(lv, L, 3, 64ull << 20, 1) == 3);
    // a full scan finds what the incremental ones did
    struct vkdense e = d; e.mapped = calloc((size_t)Y * d.per_layer, 1); e.cnt = calloc((size_t)L * Y, sizeof *e.cnt); dense_scan(&e, 0, ~0ull, mapped, NULL, 0);
    assert(!memcmp(e.cnt, d.cnt, (size_t)L * Y * sizeof *d.cnt) && !memcmp(e.mapped, d.mapped, (size_t)Y * d.per_layer));
    return 0;
}
'''
        self.build_run(code, 'dense')

    def test_large_image_blocks(self):
        if not (VKH / 'vulkan/vulkan_core.h').exists():
            self.skipTest('the vendored Vulkan headers are needed')
        src = CORE.read_text(encoding='utf-8')
        decl = region(src, 'typedef struct lblk {', 'typedef struct { imgblk *b;')
        body = region(src, '// ---- large image blocks: first fit', '// ---- end of large image blocks ----')
        code = '#include <assert.h>\n#include <stdint.h>\n#include <stdlib.h>\n#include <string.h>\n#define VK_NO_PROTOTYPES\n#include <vulkan/vulkan_core.h>\n' + decl + body + r'''
int main(void) {
    lblk b; memset(&b, 0, sizeof b); b.size = 1000; b.capfr = 1; b.fr = calloc(1, sizeof *b.fr); b.fr[0].len = 1000; b.nfr = 1;
    VkDeviceSize a, c, d, e;
    assert(lblk_take(&b, 100, 1, &a) && a == 0 && b.live == 1 && b.nfr == 1 && b.fr[0].off == 100);
    assert(lblk_take(&b, 50, 64, &c) && c == 128 && b.nfr == 2 && b.fr[0].off == 100 && b.fr[0].len == 28 && b.fr[1].off == 178);   // aligned: the gap before it stays free
    assert(lblk_take(&b, 28, 4, &d) && d == 100 && b.nfr == 1);   // first fit: the gap
    assert(!lblk_take(&b, 900, 1, &e));   // too large for what is left
    assert(lblk_take(&b, 822, 1, &e) && e == 178 && b.nfr == 0 && b.live == 4);   // the rest, exactly
    lblk_give(&b, c, 50); assert(b.nfr == 1 && b.fr[0].off == 128 && b.fr[0].len == 50);
    lblk_give(&b, a, 100); assert(b.nfr == 2 && b.fr[0].off == 0);
    lblk_give(&b, d, 28); assert(b.nfr == 1 && b.fr[0].off == 0 && b.fr[0].len == 178);   // joined with both neighbours
    lblk_give(&b, e, 822); assert(b.nfr == 1 && b.fr[0].len == 1000 && b.live == 0);
    // many images, freed in a shuffled order: the block whole again
    VkDeviceSize off[40]; for (int i = 0; i < 40; i++) assert(lblk_take(&b, 24, 8, &off[i]) && off[i] == (VkDeviceSize)i * 24);
    assert(!lblk_take(&b, 41, 1, &a) && lblk_take(&b, 40, 1, &a) && a == 960 && b.nfr == 0);
    for (int i = 0; i < 40; i++) { int j = (i * 17) % 40; lblk_give(&b, off[j], 24); }   // (17 is prime to 40: every one, shuffled)
    lblk_give(&b, a, 40);
    assert(b.nfr == 1 && b.fr[0].off == 0 && b.fr[0].len == 1000 && b.live == 0);
    return 0;
}
'''
        self.build_run(code, 'lblk', extra=['-I' + str(VKH)])


    def test_image_block_release(self):
        """A small or large image block left empty is freed when another of its kind could take the next image, kept otherwise (no churn); no block is freed while an image holds part
        of it; memory of its own goes with its image (gfx_vk_core.c vk_image_memory, img_sub_free)."""
        if not (VKH / 'vulkan/vulkan_core.h').exists():
            self.skipTest('the vendored Vulkan headers are needed')
        body = region(CORE.read_text(encoding='utf-8'), '// Image memory. Small images share blocks', '// DM_VK_MEMLOG=<seconds>')
        code = r"""
#include <assert.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define VK_NO_PROTOTYPES
#include <vulkan/vulkan_core.h>
static struct { VkDevice dev; } V;
typedef struct { VkDeviceSize size; int freed; } fmem;
static int g_live;
static VkResult vkAllocateMemory(VkDevice d, const VkMemoryAllocateInfo *ai, const VkAllocationCallbacks *a, VkDeviceMemory *m) { (void)d; (void)a; fmem *f = calloc(1, sizeof *f); f->size = ai->allocationSize; *m = (VkDeviceMemory)(uintptr_t)f; g_live++; return VK_SUCCESS; }
static void vkFreeMemory(VkDevice d, VkDeviceMemory m, const VkAllocationCallbacks *a) { (void)d; (void)a; fmem *f = (fmem *)(uintptr_t)m; assert(!f->freed); f->freed = 1; g_live--; }
""" + body + r"""
static void *img(VkDeviceSize n, VkDeviceMemory *m, VkDeviceSize *off) { VkMemoryRequirements r = {n, 4096, 1}; void *s = NULL; *m = vk_image_memory(&r, 0, off, &s); assert(*m && s); return s; }
int main(void) {
    const VkDeviceSize MB = 1 << 20; VkDeviceMemory m, ma, mb; VkDeviceSize o; int own;
    // small: 1 MB slots, 32 to a block
    void *a[40]; for (int i = 0; i < 32; i++) a[i] = img(MB, &ma, &o);
    assert(g_live == 1 && vk_image_sub_size(a[0], &own) == MB && !own);
    void *b = img(MB, &mb, &o), *b2; assert(g_live == 2 && mb != ma);
    img_sub_free(b); assert(g_live == 2);   // empty, the other block full: kept (the next image goes there)
    b = img(MB, &m, &o); assert(m == mb);   // (the first block full: the second)
    for (int i = 0; i < 9; i++) img_sub_free(a[i]);   // the first block more than a quarter free
    b2 = img(MB, &m, &o); assert(m == ma);   // the fullest block with a free slot (the other drains)
    img_sub_free(b); assert(g_live == 1 && g_imgblk_bytes == 32 * MB && g_imgblk_freed == 32 * MB);   // empty, the first can take the next: freed
    a[0] = b2; for (int i = 1; i < 9; i++) { a[i] = img(MB, &m, &o); assert(m == ma); }   // (the list still sound)
    for (int i = 0; i < 32; i++) img_sub_free(a[i]); assert(g_live == 1);   // the last of its kind: kept
    // large: 64 MB blocks, images over 2 MB
    void *l[4]; for (int i = 0; i < 3; i++) { l[i] = img(20 * MB, &m, &o); assert(i == 0 ? (ma = m) != 0 : m == ma); }
    void *x = img(20 * MB, &mb, &o); assert(mb != ma && g_live == 3 && vk_image_sub_size(x, &own) == 20 * MB && !own);
    img_sub_free(x); assert(g_live == 3);   // empty, the other has 4 MB free: kept
    x = img(20 * MB, &m, &o); assert(m == mb);   // (only the second has room)
    img_sub_free(l[1]); void *y = img(20 * MB, &m, &o); assert(m == ma);   // best fit: the block with less free memory (24 MB) before the emptier one (44 MB)
    img_sub_free(l[0]); img_sub_free(y); assert(g_live == 3);   // (an image still in the second: kept)
    img_sub_free(x); assert(g_live == 2 && g_lblk_bytes == 64 * MB);   // empty, 40 MB free in the other: freed
    x = img(20 * MB, &m, &o); assert(m == ma && o == 0); img_sub_free(x); img_sub_free(l[2]); assert(g_live == 2);   // (the last large block: kept, whole again)
    // memory of its own: with its image
    x = img(40 * MB, &m, &o); assert(g_live == 3 && vk_image_sub_size(x, &own) == 40 * MB && own); img_sub_free(x); assert(g_live == 2);
    puts("blocks ok"); return 0;
}
"""
        outs = self.build_run(code, 'imgblk', extra=['-I' + str(VKH)])
        self.assertIn('blocks ok', outs[0])


    def test_scratch_trim(self):
        """A slot keeps the scratch blocks its last submission used and DM_VK_SCRATCH_KEEP_MB of others, frees the rest once unused for SCRATCH_IDLE recycles (at once when memory
        is critical), and frees nothing before the completion thread has passed its fence (the uniform updates in flight read scratch until then) (gfx_vk_core.c scratch_trim)."""
        if not (VKH / 'vulkan/vulkan_core.h').exists():
            self.skipTest('the vendored Vulkan headers are needed')
        src = CORE.read_text(encoding='utf-8')
        blk = region(src, 'typedef struct { VkBuffer b; VkDeviceMemory m; uint8_t *p; VkDeviceSize size, used; uint32_t idle; } scratch_blk;', 'typedef struct { int kind; void *a; void *b; } dfree_t;')
        sub = region(src, 'typedef struct { int kind; void *a; void *b; } dfree_t;', 'enum { GW_SUB = 64 };')
        trim = region(src, '// The scratch blocks a slot keeps', 'static void slot_recycle(')
        code = r"""
#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define VK_NO_PROTOTYPES
#include <vulkan/vulkan_core.h>
static struct { VkDevice dev; } V;
static int g_freed_b, g_freed_m; static uint64_t g_done;
static void vkDestroyBuffer(VkDevice d, VkBuffer b, const VkAllocationCallbacks *a) { (void)d; (void)b; (void)a; g_freed_b++; }
static void vkFreeMemory(VkDevice d, VkDeviceMemory m, const VkAllocationCallbacks *a) { (void)d; (void)m; (void)a; g_freed_m++; }
static uint64_t vk_fence_done(void) { return g_done; }
static int g_pressure; static int vk_mem_pressure(void) { return g_pressure; }   // (gfx_vk_budget.c)
static uint32_t g_pool_gone; static int vk_keep_buf_named(VkBuffer b) { (void)b; return 0; }   // (gfx_vk_exec.c: no kept set names it)
typedef struct vksub vksub_t;
#define MAIN_CBS 32
""" + blk + sub + ";\n" + trim + r"""
int main(void) {
    const VkDeviceSize MB = 1 << 20; static vksub_t s; s.blk = calloc(16, sizeof *s.blk); s.capblk = 16;
    VkDeviceSize sz[8] = {8 * MB, 8 * MB, 8 * MB, 8 * MB, 40 * MB, 8 * MB, 8 * MB, 8 * MB}, used[8] = {1, 0, 5, 0, 0, 0, 7, 0};
    for (int i = 0; i < 8; i++) { s.blk[i].size = sz[i]; s.blk[i].used = used[i]; s.blk[i].b = (VkBuffer)(uintptr_t)(i + 1); } s.nblk = 8; g_scratch_bytes = 96 * MB; s.fence = 10;
    g_done = 9; scratch_trim(&s); assert(s.nblk == 8 && !g_freed_m && s.blk[0].used == 0);   // its fence not passed yet: nothing freed, the blocks rewound
    for (int i = 0; i < 8; i++) s.blk[i].used = used[i];
    g_done = 10; scratch_trim(&s); assert(s.nblk == 8 && !g_freed_m);   // unused once: kept (the next submission of the slot may need it)
    for (int r = 1; r < 32; r++) { s.blk[0].used = 1; s.blk[2].used = 5; s.blk[6].used = 7; scratch_trim(&s); if (r < 31) assert(s.nblk == 8 && !g_freed_m); }
    // unused for 32 recycles: used ones kept (0, 2, 6), 16 MB of the others (1, 3), the rest freed (the 40 MB one, 5, 7)
    assert(s.nblk == 5 && g_freed_m == 3 && g_freed_b == 3 && g_scratch_bytes == 40 * MB && g_scratch_freed == 56 * MB);
    uintptr_t kept[5] = {1, 2, 3, 4, 7}; for (int i = 0; i < 5; i++) assert((uintptr_t)s.blk[i].b == kept[i] && !s.blk[i].used);
    for (int r = 0; r < 32; r++) scratch_trim(&s); assert(s.nblk == 2 && g_freed_m == 6);   // nothing used since: 16 MB kept
    g_pressure = 1; s.blk[0].used = 1; scratch_trim(&s); assert(s.nblk == 1 && g_freed_m == 7 && (uintptr_t)s.blk[0].b == 1);   // memory short: no spare kept (the long unused one goes)
    s.blk[1] = s.blk[0]; s.blk[1].b = (VkBuffer)(uintptr_t)9; s.blk[1].idle = 0; s.blk[1].used = 0; s.nblk = 2; g_scratch_bytes += 8 * MB;   // a block unused once
    s.blk[0].used = 1; scratch_trim(&s); assert(s.nblk == 2 && g_freed_m == 7);   // memory short: a recent one stays
    g_pressure = 2; for (int r = 0; r < 2; r++) { s.blk[0].used = 1; scratch_trim(&s); assert(s.nblk == 2 && g_freed_m == 7); }   // memory critical: kept until unused 4 recycles (made again at once otherwise)
    s.blk[0].used = 1; scratch_trim(&s); assert(s.nblk == 1 && g_freed_m == 8 && (uintptr_t)s.blk[0].b == 1);   // then it goes
    puts("scratch ok"); return 0;
}
"""
        outs = self.build_run(code, 'scratch', extra=['-I' + str(VKH)])
        self.assertIn('scratch ok', outs[0])


if __name__ == '__main__':
    unittest.main()
