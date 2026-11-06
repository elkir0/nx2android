// Android port, Vulkan backend: the shadow file (see gfx_vk_shadow.h).
//   Why. A packaged texture is uploaded once from guest memory into an image of its own (gfx_vk_tex.c pkg_upload) and its storage is then write-protected (nvn_dirty): from then on its
//   bytes live twice, in the GPU image and in the game's anonymous pages, which nothing reads again. Those pages are what the kernel swaps: run 13 of Mario Kart World 1.8.0 on the Z Fold 8
//   Ultra held 4.0 GB resident (RssAnon 3.7 GB) and 1.2 GB swapped at its race start, 2.0 GB resident and 2.9 GB swapped minutes on (4.9 GB anonymous throughout), beside 4.7 GB of GPU
//   memory and 2.1-2.3 GB MemAvailable; run 11 was ended by the low-memory killer ("oom_score_adj 0 to free 758232kB rss, 249728kB anon rss, 6060360kB swap; reason: min watermark is
//   breached and swap is low"). zram cannot shrink BC / ASTC blocks, so swapping them saves little RAM and fills the swap. As clean file pages they cost neither: the kernel drops them under
//   pressure and reads them back from flash if they are ever read again.
//   How. One unlinked file (DM_VK_SHADOW_DIR, default $DM_TEMP: the app's internal storage) spans the guest block, each guest page at the offset of its address (base: the alias region less
//   256 MB, hle_svc.c mem_init), so neighbours mapped one after the other merge into one mapping. The replay notes a texture right after its upload (its pages protected, the stamp taken
//   after that, gfx_vk_tex.c pkg_upload_) and the physical pages behind each sparse tile it uploads. A worker thread writes the whole pages of a note that is DM_VK_SHADOW_AGE_MS old and
//   not written since to the file (O_DIRECT where the file takes it: no page cache on the way), and the replay thread swaps them at its next present (vk_shadow_tick), as gfx_vk_core.c's
//   pool swap does: inside a window in which a game write into the range waits in this file's page fault hook and a host write in its pre-write hook (registered before the dirty-page
//   tracker's), it checks that no page was written or unprotected since the note (nvn_dirty_since: each page then stayed protected, so its bytes are the ones the worker read), makes the
//   range read-only, checks again (a write that faulted just before the window), maps the file over it MAP_PRIVATE | MAP_FIXED read-only, and protects it through the tracker (a page a
//   write in flight unprotected meanwhile is protected again, as the tracker's state must say). A later write faults as before, the tracker lets it through, and the kernel copies the page
//   to anonymous memory: the game never writes the file.
//   Risks: storage full -> O_DIRECT reports it at the write (else fallocate first), and that range is never mapped (no SIGBUS: a mapped page has its blocks); flash writes (about each new
//   course's textures) -> a page whose bytes the file already holds at its offset (the same 64-bit hash) is not written again; re-reads of a dropped page (about 50 us a 64 KB read) ->
//   only when the game or an upload reads a texture it no longer draws; the mapping limit (65,530) -> runs merge, and no swap past DM_VK_SHADOW_MAPS mappings; the game unmapping memory
//   (hle_svc.c guest_release -> dm_guest_released) -> the windows are waited out and the file's blocks there punched. Never shadowed: placed pools (a hardware buffer's pages), textures the
//   GPU wrote, linear ones, ones not protected since their upload.
#ifndef VK_SHADOW_TEST
#include "gfx_vk.h"
#endif
#include "gfx_vk_shadow.h"
#include "nvn_dirty.h"
#include "dm.h"
#include <errno.h>
#include <fcntl.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <sys/statvfs.h>
#include <time.h>
#include <unistd.h>
#if defined(__linux__) || defined(__ANDROID__)
#include <linux/falloc.h>
#endif

// ---- the shadow core (tests: tests/test_vk_shadow.py) ----
typedef struct {   // what the core asks of its caller (the glue below, or a test)
    int (*unplaced)(uint64_t lo, uint64_t n);            // [lo, lo+n) is guest memory no GPU buffer lies over (a pool chunk or granules not placed)
    uint64_t (*unreadable)(const void *p, uint64_t n);   // the offset of its first page the CPU cannot read (n: none; UINT64_MAX: cannot be told)
    int (*pressure)(void);                               // the device is short of memory (gfx_vk_budget.c vk_mem_pressure): notes are written without waiting out their age
} sh_env;
enum { SH_NONE, SH_FILE, SH_MAPPED };   // a page's state: nothing in the file; the file holds its bytes of hash h; and the guest page is the file's (since its swap; a write moves it back)
enum { SH_CH = 4096 };                  // pages in a chunk of the page table
typedef struct { uint64_t h[SH_CH]; uint8_t st[SH_CH]; } sh_chunk;
typedef struct { uint64_t lo, hi, stamp, t_ns; const void *span; } sh_cand;   // a range noted (the tracker's stamp after its protection, its range handle), when
typedef struct { uint64_t lo, hi, stamp, gen, due; const void *span; int tries; } sh_run;   // a range written to the file, for the replay thread to swap (not before due; put off tries times)
static struct {
    int on, fd, dio; uint64_t base, size, pg; sh_env env; char dir[512];
    uint64_t min, age_ns, bps, max_bytes, free_min, quiet_ns; int tick_runs, maps_max;
} g_sh = {.fd = -1};
static sh_chunk **g_pt; static size_t g_npt; static pthread_mutex_t g_pt_mu = PTHREAD_MUTEX_INITIALIZER;
static sh_cand *g_cand; static size_t g_ncand, g_capcand; static sh_run *g_ready; static size_t g_nready, g_capready;
static pthread_mutex_t g_q_mu = PTHREAD_MUTEX_INITIALIZER; static pthread_cond_t g_q_cv = PTHREAD_COND_INITIALIZER;
static struct { _Atomic uint64_t lo, hi; _Atomic int on; _Atomic long tid; } g_win, g_rd;   // the swap's window (replay thread), the worker's read of a range
static _Atomic uint64_t g_relgen;   // ranges the game gave back so far (a run written before one is checked again)
static _Atomic int g_maps;          // the process's mappings at the worker's last count (/proc/self/maps)
// counters (vk_shadow_log): bytes noted, written to the file, not written (the file held them), mapped now; swaps done and given up (written meanwhile, placed, given back or
// unreadable, mmap failed, the mapping limit), writes refused (no room, an I/O error), notes skipped (in flux, placed, too small), held writes, punched bytes
static _Atomic uint64_t c_noted, c_written, c_reused, c_mapped, c_swaps, c_ab_written, c_ab_placed, c_ab_gone, c_ab_map, c_ab_maps, c_full, c_ioerr, c_flux, c_skip_placed,
    c_held, c_punched, c_worker_ns, c_tick_ns, c_postponed;
static int sh_pgsh(void) { return __builtin_ctzll(g_sh.pg); }
static uint64_t sh_now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return (uint64_t)t.tv_sec * 1000000000ull + (uint64_t)t.tv_nsec; }
static void sh_nap(long ns) { struct timespec t = {0, ns}; nanosleep(&t, NULL); }
#ifdef __APPLE__
static long sh_tid(void) { uint64_t t = 0; pthread_threadid_np(NULL, &t); return (long)t; }
#else
static long sh_tid(void) { return (long)gettid(); }
#endif
// 64 bits of a page's bytes (n a multiple of 32): four lanes of xxHash64's round over 32-byte blocks, then its merge and avalanche. Only equal pages may skip their write: a false match
// would map stale bytes, at 2^-64 per page compared
static inline uint64_t sh_rotl(uint64_t x, int r) { return x << r | x >> (64 - r); }
static uint64_t sh_hash(const uint8_t *p, size_t n) {
    const uint64_t P1 = 0x9E3779B185EBCA87ull, P2 = 0xC2B2AE3D27D4EB4Full, P3 = 0x165667B19E3779F9ull, P4 = 0x85EBCA77C2B2AE63ull;
    uint64_t v[4] = {P1 + P2, P2, 0, (uint64_t)0 - P1};
    for (size_t i = 0; i + 32 <= n; i += 32) for (int k = 0; k < 4; k++) { uint64_t w; memcpy(&w, p + i + 8 * k, 8); v[k] = sh_rotl(v[k] + w * P2, 31) * P1; }
    uint64_t h = sh_rotl(v[0], 1) + sh_rotl(v[1], 7) + sh_rotl(v[2], 12) + sh_rotl(v[3], 18);
    for (int k = 0; k < 4; k++) { h ^= sh_rotl(v[k] * P2, 31) * P1; h = h * P1 + P4; }
    h += n; h ^= h >> 33; h *= P2; h ^= h >> 29; h *= P3; h ^= h >> 32;
    return h;
}
// The page table: a chunk of SH_CH pages made at its first use (g_pt_mu held)
static sh_chunk *pt_get(uint64_t pi, int make) {
    size_t c = (size_t)(pi / SH_CH); if (c >= g_npt) return NULL;
    if (!g_pt[c] && make && !(g_pt[c] = calloc(1, sizeof **g_pt))) return NULL;
    return g_pt[c];
}
static uint64_t sh_off(uint64_t a) { return a - g_sh.base; }
static int sh_inside(uint64_t lo, uint64_t hi) { return lo >= g_sh.base && hi <= g_sh.base + g_sh.size && lo < hi; }
// ---- writers' announcements. A window holds a writer only from when it opens: a store that faulted just before, whose release the tracker has in flight (its page made writable,
// not stamped yet), or a host write prepared just before, could land in the old page after the swap's last check and before the file is mapped over it, and be lost (tests/test_vk_shadow.py's
// stress lost one write in about five runs that way; gfx_vk_core.c's pool swap has the same opening). So a write fault and a host write mark the 256 KB granules they touch with the
// time, before they look at the window (one atomic store a granule: signal-safe), and a swap whose window is open gives up its range when a granule of it was marked in the last
// DM_VK_SHADOW_QUIET_MS (it is tried again later): either the mark comes first and the swap sees it, or the window does and the writer waits in it. Pool finalize marks its range too
// (dm_guest_unshadow: its pages are made writable without a fault, nvn_dirty_remove_range) ----
enum { SH_GRAN_SHIFT = 18 };
static _Atomic uint64_t *g_wt; static size_t g_nwt;   // per granule of the file's span: the last mark's time (CLOCK_MONOTONIC ns)
static void wt_mark(uint64_t lo, uint64_t hi) {
    uint64_t b = g_sh.base, e = b + g_sh.size; if (!g_wt || hi <= lo || hi <= b || lo >= e) return;
    if (lo < b) lo = b; if (hi > e) hi = e;
    uint64_t now = sh_now(); for (uint64_t g = (lo - b) >> SH_GRAN_SHIFT, z = (hi - 1 - b) >> SH_GRAN_SHIFT; g <= z && g < g_nwt; g++) atomic_store(&g_wt[g], now);
}
static int wt_recent(uint64_t lo, uint64_t hi, uint64_t quiet) {   // a granule of [lo, hi) marked less than quiet ago
    if (!g_wt || !quiet) return 0;
    uint64_t now = sh_now(), b = g_sh.base;
    for (uint64_t g = (lo - b) >> SH_GRAN_SHIFT, z = (hi - 1 - b) >> SH_GRAN_SHIFT; g <= z && g < g_nwt; g++) { uint64_t t = atomic_load(&g_wt[g]); if (t && now - t < quiet) return 1; }
    return 0;
}
// ---- the windows: a game write into a range being swapped waits for the new pages, a host write too (signal context: atomics and nanosleep only) ----
static int win_hit(const typeof(g_win) *w, uint64_t lo, uint64_t hi) { return atomic_load(&w->on) && lo < atomic_load(&w->hi) && hi > atomic_load(&w->lo); }
static int sh_fault(void *addr, int is_write, void *uc) {   // the host's page fault hook (crash.c): 1 = made again
    (void)is_write; (void)uc; uint64_t a = (uint64_t)(uintptr_t)addr;
    wt_mark(a, a + 1);   // (first: a window opened after it sees it)
    if (!atomic_load(&g_win.on) || a < atomic_load(&g_win.lo) || a >= atomic_load(&g_win.hi) || sh_tid() == atomic_load(&g_win.tid)) return 0;   // (the swap's own thread: to the crash handler)
    atomic_fetch_add(&c_held, 1);
    while (atomic_load(&g_win.on)) sh_nap(100000);
    return 1;
}
static void sh_prep(void *addr, size_t len) {   // a host write into guest memory is about to be made (dm_guest_write_prepare): after any swap of its range
    uint64_t a = (uint64_t)(uintptr_t)addr;
    wt_mark(a, a + len);
    if (win_hit(&g_win, a, a + len) && sh_tid() != atomic_load(&g_win.tid)) { atomic_fetch_add(&c_held, 1); while (win_hit(&g_win, a, a + len)) sh_nap(100000); }
}
static void win_open(typeof(g_win) *w, uint64_t lo, uint64_t hi) { atomic_store(&w->lo, lo); atomic_store(&w->hi, hi); atomic_store(&w->tid, sh_tid()); atomic_store(&w->on, 1); }
static void win_close(typeof(g_win) *w) { atomic_store(&w->on, 0); }
// ---- the file ----
static int sh_write_all(const uint8_t *b, uint64_t n, uint64_t off) {   // 0, or -1 (errno: ENOSPC, EIO...)
    while (n) { ssize_t w = pwrite(g_sh.fd, b, (size_t)n, (off_t)off); if (w < 0 && errno == EINTR) continue; if (w <= 0) { if (!w) errno = EIO; return -1; } b += w; n -= (uint64_t)w; off += (uint64_t)w; }
    return 0;
}
static int sh_reserve(uint64_t off, uint64_t n) {   // the blocks of [off, off+n) before a buffered write (a write-back that finds no room later would lose the page): 0 or -1
#if defined(__linux__) || defined(__ANDROID__)
    if (g_sh.dio) return 0;   // (a direct write reports it itself)
    return fallocate(g_sh.fd, 0, (off_t)off, (off_t)n) && errno != EOPNOTSUPP ? -1 : 0;
#else
    (void)off; (void)n; return 0;
#endif
}
// [off, off+n) of the file from b: its blocks first (buffered), then the bytes. A file system that opens O_DIRECT but refuses direct writes (EINVAL: an encrypted file without inline
// encryption, for one) gets buffered writes from then on. 0, or -1 with errno
static int sh_put(const uint8_t *b, uint64_t n, uint64_t off) {
    if (!sh_reserve(off, n) && !sh_write_all(b, n, off)) return 0;
#if defined(O_DIRECT)
    if (g_sh.dio && errno == EINVAL) { int fl = fcntl(g_sh.fd, F_GETFL); if (fl >= 0 && !fcntl(g_sh.fd, F_SETFL, fl & ~O_DIRECT)) { g_sh.dio = 0; LOG("[vk] shadow: the file refused a direct write (EINVAL): buffered writes from now on"); return sh_reserve(off, n) || sh_write_all(b, n, off) ? -1 : 0; } }
#endif
    return -1;
}
static void sh_punch_file(uint64_t off, uint64_t n) {
#if defined(__APPLE__)
    struct fpunchhole ph = {0, 0, (off_t)off, (off_t)n}; fcntl(g_sh.fd, F_PUNCHHOLE, &ph);
#else
    fallocate(g_sh.fd, FALLOC_FL_PUNCH_HOLE | FALLOC_FL_KEEP_SIZE, (off_t)off, (off_t)n);
#endif
}
static uint64_t sh_file_bytes(void) { struct stat st; return g_sh.fd >= 0 && !fstat(g_sh.fd, &st) ? (uint64_t)st.st_blocks * 512 : 0; }
static uint64_t sh_free_bytes(void) { struct statvfs v; return g_sh.fd >= 0 && !fstatvfs(g_sh.fd, &v) ? (uint64_t)v.f_bavail * v.f_frsize : 0; }
// The file's pages of [lo, hi) given back: blocks punched, the page table's entries cleared (g_pt_mu held)
static void sh_punch(uint64_t lo, uint64_t hi) {
    uint64_t pg = g_sh.pg; lo = lo & ~(pg - 1); hi = (hi + pg - 1) & ~(pg - 1); if (lo < g_sh.base) lo = g_sh.base; if (hi > g_sh.base + g_sh.size) hi = g_sh.base + g_sh.size;
    uint64_t run = 0, nrun = 0;
    for (uint64_t a = lo; a < hi; a += pg) {
        uint64_t pi = sh_off(a) >> sh_pgsh(); sh_chunk *c = pt_get(pi, 0); int st = c ? c->st[pi % SH_CH] : SH_NONE;
        if (st != SH_NONE) { if (st == SH_MAPPED) atomic_fetch_sub(&c_mapped, pg); c->st[pi % SH_CH] = SH_NONE; c->h[pi % SH_CH] = 0; if (!nrun) run = a; nrun += pg; continue; }
        if (!c) a = g_sh.base + (((pi / SH_CH) + 1) * SH_CH << sh_pgsh()) - pg;   // (a chunk never made: skip it)
        if (nrun) { sh_punch_file(sh_off(run), nrun); atomic_fetch_add(&c_punched, nrun); nrun = 0; }
    }
    if (nrun) { sh_punch_file(sh_off(run), nrun); atomic_fetch_add(&c_punched, nrun); }
}
// ---- notes (the replay thread) and the worker ----
static void sh_note(uint64_t lo, uint64_t hi, uint64_t stamp, const void *span) {   // (the size floor is the texture's, vk_shadow_tex_class: a sparse tile is 64 KB, joined with its neighbours)
    uint64_t pg = g_sh.pg; lo = (lo + pg - 1) & ~(pg - 1); hi &= ~(pg - 1);   // whole pages only: a page shared with another object keeps its anonymous memory
    if (!g_sh.on || hi <= lo || !sh_inside(lo, hi) || !span) return;
    pthread_mutex_lock(&g_q_mu);
    if (g_ncand == g_capcand) { size_t nc = g_capcand ? 2 * g_capcand : 1024; sh_cand *n = realloc(g_cand, nc * sizeof *n); if (!n) { pthread_mutex_unlock(&g_q_mu); return; } g_cand = n; g_capcand = nc; }
    g_cand[g_ncand++] = (sh_cand){lo, hi, stamp, sh_now(), span}; atomic_fetch_add(&c_noted, hi - lo);
    pthread_cond_signal(&g_q_cv); pthread_mutex_unlock(&g_q_mu);
}
static int cand_cmp(const void *a, const void *b) { const sh_cand *x = a, *y = b; return x->lo < y->lo ? -1 : x->lo > y->lo ? 1 : 0; }
// Notes of one range of the tracker that touch or overlap joined (the older stamp kept: a page unprotected after it is seen), in place; the count left
static size_t cand_merge(sh_cand *c, size_t n) {
    if (!n) return 0;
    qsort(c, n, sizeof *c, cand_cmp); size_t k = 0;
    for (size_t i = 1; i < n; i++) {
        if (c[i].lo <= c[k].hi && c[i].span == c[k].span) { if (c[i].hi > c[k].hi) c[k].hi = c[i].hi; if (c[i].stamp < c[k].stamp) c[k].stamp = c[i].stamp; if (c[i].t_ns > c[k].t_ns) c[k].t_ns = c[i].t_ns; continue; }
        c[++k] = c[i];
    }
    return k + 1;
}
static void ready_push(const sh_run *r) {
    pthread_mutex_lock(&g_q_mu);
    if (g_nready == g_capready) { size_t nc = g_capready ? 2 * g_capready : 256; sh_run *n = realloc(g_ready, nc * sizeof *n); if (!n) { pthread_mutex_unlock(&g_q_mu); return; } g_ready = n; g_capready = nc; }
    g_ready[g_nready++] = *r; pthread_mutex_unlock(&g_q_mu);
}
enum { SH_IO = 1 << 20 };   // the worker reads and writes a range a megabyte at a time (the window a release waits for is one such read)
static uint8_t *g_iobuf;
// One note: its pages read (inside the read window: a release waits for it), hashed, the ones the file does not hold written; then a run for the replay thread. 0, or why not
static int sh_write(const sh_cand *c) {
    uint64_t lo = c->lo, hi = c->hi, n = hi - lo, pg = g_sh.pg; int sh = sh_pgsh();
    if (!g_sh.env.unplaced(lo, n)) { atomic_fetch_add(&c_skip_placed, 1); return 1; }
    if (nvn_dirty_since_span(c->span, (const void *)(uintptr_t)lo, n, c->stamp)) { atomic_fetch_add(&c_flux, 1); return 2; }   // written or unprotected since its upload: in flux
    uint64_t have = sh_file_bytes(); if (have + n > g_sh.max_bytes || sh_free_bytes() < g_sh.free_min + n) { atomic_fetch_add(&c_full, 1); return 3; }
    uint64_t gen = atomic_load(&g_relgen);
    for (uint64_t a = lo; a < hi;) {
        uint64_t k = hi - a < SH_IO ? hi - a : SH_IO;
        win_open(&g_rd, a, a + k);
        uint64_t u = g_sh.env.unreadable((const void *)(uintptr_t)a, k);
        if (u < k || u == UINT64_MAX) { win_close(&g_rd); atomic_fetch_add(&c_ab_gone, 1); return 4; }   // (a page the game has not mapped: never read)
        memcpy(g_iobuf, (const void *)(uintptr_t)a, k);
        win_close(&g_rd);
        // the pages whose bytes the file does not hold, written a run at a time
        uint64_t hs[SH_IO / 4096], np = k >> sh; int need[SH_IO / 4096];
        pthread_mutex_lock(&g_pt_mu);
        for (uint64_t i = 0; i < np; i++) { uint64_t pi = sh_off(a + (i << sh)) >> sh; sh_chunk *ch = pt_get(pi, 1); hs[i] = sh_hash(g_iobuf + (i << sh), pg);
            need[i] = !ch || ch->st[pi % SH_CH] == SH_NONE || ch->h[pi % SH_CH] != hs[i]; if (!ch) { pthread_mutex_unlock(&g_pt_mu); return 5; } }
        pthread_mutex_unlock(&g_pt_mu);
        for (uint64_t i = 0; i < np;) {
            if (!need[i]) { atomic_fetch_add(&c_reused, pg); i++; continue; }
            uint64_t j = i; while (j < np && need[j]) j++;
            uint64_t off = sh_off(a + (i << sh)), len = (j - i) << sh;
            // the pages' entries cleared first: the file holds neither the old bytes nor (until the write is whole) the new ones
            pthread_mutex_lock(&g_pt_mu); for (uint64_t q = i; q < j; q++) { uint64_t pi = sh_off(a + (q << sh)) >> sh; sh_chunk *ch = pt_get(pi, 0); if (ch->st[pi % SH_CH] == SH_MAPPED) atomic_fetch_sub(&c_mapped, pg); ch->st[pi % SH_CH] = SH_NONE; }
            pthread_mutex_unlock(&g_pt_mu);
            if (sh_put(g_iobuf + (i << sh), len, off)) {
                int e = errno; static int said; if (!said++) LOG("[RED] [vk] shadow: the file refused %llu KB at %#llx (errno %d): nothing of it is mapped; shadowing paused for 60 s", (unsigned long long)(len >> 10), (unsigned long long)off, e);
                if (e == ENOSPC || e == EDQUOT) atomic_fetch_add(&c_full, 1); else atomic_fetch_add(&c_ioerr, 1);
                sh_punch_file(off, len); return 6; }
            atomic_fetch_add(&c_written, len);
            pthread_mutex_lock(&g_pt_mu); for (uint64_t q = i; q < j; q++) { uint64_t pi = sh_off(a + (q << sh)) >> sh; sh_chunk *ch = pt_get(pi, 0); ch->st[pi % SH_CH] = SH_FILE; ch->h[pi % SH_CH] = hs[q]; }
            pthread_mutex_unlock(&g_pt_mu);
            i = j;
        }
        a += k;
    }
#if defined(__linux__) || defined(__ANDROID__)
    if (!g_sh.dio) sync_file_range(g_sh.fd, (off_t)sh_off(lo), (off_t)n, SYNC_FILE_RANGE_WRITE);   // (the write-back started now: clean, droppable pages sooner)
#endif
    sh_run r = {lo, hi, c->stamp, gen, 0, c->span, 0}; ready_push(&r);
    return 0;
}
static int sh_count_maps(void) {   // the process's mappings (each a line of /proc/self/maps), -1 when it cannot be read
#if defined(__APPLE__)
    return -1;
#else
    int fd = open("/proc/self/maps", O_RDONLY | O_CLOEXEC); if (fd < 0) return -1;
    static char b[1 << 16]; int n = 0; ssize_t r; while ((r = read(fd, b, sizeof b)) > 0) for (ssize_t i = 0; i < r; i++) n += b[i] == '\n';
    close(fd); return n;
#endif
}
static _Atomic int g_stop;   // (tests: the worker ends)
static void *sh_worker(void *arg) {
    (void)arg; pthread_setname_np("vk shadow");
#if !defined(__APPLE__)
    setpriority(PRIO_PROCESS, 0, 10);   // below the game's threads, as the prewarm (gfx_vk_exec.c prewarm_thread)
#endif
    uint64_t t_maps = 0, paused_until = 0, budget_t = sh_now(); double budget = 0;
    while (!atomic_load(&g_stop)) {
        pthread_mutex_lock(&g_q_mu);
        if (!g_ncand) { struct timespec dl; clock_gettime(CLOCK_REALTIME, &dl); dl.tv_nsec += 200000000; if (dl.tv_nsec >= 1000000000) { dl.tv_sec++; dl.tv_nsec -= 1000000000; } pthread_cond_timedwait(&g_q_cv, &g_q_mu, &dl); }
        uint64_t now = sh_now(), age = g_sh.env.pressure && g_sh.env.pressure() ? 0 : g_sh.age_ns; sh_cand *take = NULL; size_t nt = 0, keep = 0;
        if (g_ncand && now >= paused_until && (take = malloc(g_ncand * sizeof *take))) {
            for (size_t i = 0; i < g_ncand; i++) { if (now - g_cand[i].t_ns >= age) take[nt++] = g_cand[i]; else g_cand[keep++] = g_cand[i]; }   // (the young ones wait, unless memory is short)
            g_ncand = keep; }
        pthread_mutex_unlock(&g_q_mu);
        if (now - t_maps > 5000000000ull) { t_maps = now; atomic_store(&g_maps, sh_count_maps()); }
        nt = cand_merge(take, nt);
        uint64_t t0 = sh_now();
        for (size_t i = 0; i < nt && !atomic_load(&g_stop); i++) {
            budget += (sh_now() - budget_t) / 1e9 * (double)g_sh.bps; budget_t = sh_now(); if (budget > (double)g_sh.bps) budget = (double)g_sh.bps;   // (a second's worth at most)
            while (budget < 0 && !atomic_load(&g_stop)) { sh_nap(20000000); budget += (sh_now() - budget_t) / 1e9 * (double)g_sh.bps; budget_t = sh_now(); }
            int why = sh_write(&take[i]); if (!why) budget -= (double)(take[i].hi - take[i].lo);
            if (why == 3 || why == 6) { paused_until = sh_now() + 60000000000ull; break; }   // (no room, or the file refused a write: a minute's pause, the notes dropped)
        }
        atomic_fetch_add(&c_worker_ns, sh_now() - t0);
        free(take);
    }
    return NULL;
}
// ---- the swap (the replay thread, at a present: the thread that places pools, so no placement runs meanwhile) ----
#ifdef VK_SHADOW_TEST
static void (*g_test_in_window)(void);   // (tests: called inside the window, the range read-only, before the file is mapped over it)
#endif
static int sh_swap(const sh_run *r) {   // 0 = the range is the file's now
    uint64_t lo = r->lo, n = r->hi - r->lo; uint8_t *mem = (uint8_t *)(uintptr_t)lo; int pg = (int)g_sh.pg, why = 0;
    if (!g_sh.env.unplaced(lo, n)) { atomic_fetch_add(&c_ab_placed, 1); return 1; }
    if (g_sh.maps_max && atomic_load(&g_maps) >= g_sh.maps_max) { atomic_fetch_add(&c_ab_maps, 1); return 2; }
    win_open(&g_win, lo, r->hi);
    if (atomic_load(&g_relgen) != r->gen) why = 3;                                  // (memory given back or a pool finalized since the write: given up)
    else if (wt_recent(lo, r->hi, g_sh.quiet_ns)) why = 7;                          // a writer announced lately (see wt_mark): tried again later
    else if (nvn_dirty_since_span(r->span, mem, n, r->stamp)) why = 4;              // written or unprotected since the note (before any protection change of ours)
    else if (mprotect(mem, n, PROT_READ)) why = 5;                                  // (read-only already: protected since the note)
    else if (nvn_dirty_since_span(r->span, mem, n, r->stamp)) { why = 4; nvn_dirty_protect(mem, n); }   // a write that faulted just before the window: its page protected again, as its state says
#ifdef VK_SHADOW_TEST
    else if (g_test_in_window && (g_test_in_window(), 0)) {}
#endif
    else if (mmap(mem, n, PROT_READ, MAP_PRIVATE | MAP_FIXED, g_sh.fd, (off_t)sh_off(lo)) != (void *)mem) { why = 6; nvn_dirty_protect(mem, n); }
    else nvn_dirty_protect(mem, n);   // (every page protected already, but one a write in flight unprotected while the window opened)
    win_close(&g_win);
    if (why == 7) { atomic_fetch_add(&c_postponed, 1); return why; }
    if (why) { atomic_fetch_add(why == 3 ? &c_ab_gone : why == 4 ? &c_ab_written : &c_ab_map, 1); return why; }
    pthread_mutex_lock(&g_pt_mu);
    for (uint64_t a = lo; a < r->hi; a += (uint64_t)pg) { uint64_t pi = sh_off(a) >> sh_pgsh(); sh_chunk *c = pt_get(pi, 0);
        if (c && c->st[pi % SH_CH] == SH_FILE) { c->st[pi % SH_CH] = SH_MAPPED; atomic_fetch_add(&c_mapped, (uint64_t)pg); } }
    pthread_mutex_unlock(&g_pt_mu);
    atomic_fetch_add(&c_swaps, 1);
    return 0;
}
static void sh_tick_(void) {
    uint64_t now = sh_now(); size_t n = 0;
    pthread_mutex_lock(&g_q_mu); sh_run *take = g_nready ? malloc((size_t)g_sh.tick_runs * sizeof *take) : NULL; size_t k = 0;
    for (size_t i = 0; i < g_nready; i++) { if (take && n < (size_t)g_sh.tick_runs && g_ready[i].due <= now) take[n++] = g_ready[i]; else g_ready[k++] = g_ready[i]; }   // (the due ones, in order)
    g_nready = k; pthread_mutex_unlock(&g_q_mu);
    for (size_t i = 0; i < n; i++) if (sh_swap(&take[i]) == 7 && take[i].tries < 16) { sh_run r = take[i]; r.due = sh_now() + g_sh.quiet_ns; r.tries++; ready_push(&r); }   // (a range written near all the time: dropped)
    if (n) atomic_fetch_add(&c_tick_ns, sh_now() - now);
    free(take);
}
// [lo, hi) is about to be made writable without a fault (a pool finalized: nvn_dirty_remove_range) or given back (punch): marked, the runs written before given up, the windows over it
// waited out (the read windows too when the blocks go)
static void sh_unshadow(uint64_t lo, uint64_t hi, int punch) {
    if (!g_sh.on || hi <= lo) return;
    wt_mark(lo, hi); atomic_fetch_add(&g_relgen, 1);
    while (win_hit(&g_win, lo, hi) || (punch && win_hit(&g_rd, lo, hi))) sh_nap(100000);
    if (punch) { pthread_mutex_lock(&g_pt_mu); sh_punch(lo, hi); pthread_mutex_unlock(&g_pt_mu); }
}
static void sh_released(uint64_t lo, uint64_t hi) { sh_unshadow(lo, hi, 1); }   // the game gave [lo, hi) back
// The file over [base, base+size) in dir, its page table, the worker; the hooks are the caller's to register first. 0 on success
static int sh_open(uint64_t base, uint64_t size, const char *dir, int dio) {
    uint64_t pg = (uint64_t)getpagesize(); g_sh.pg = pg; g_sh.base = base & ~(pg - 1); g_sh.size = size & ~(pg - 1); snprintf(g_sh.dir, sizeof g_sh.dir, "%s", dir);
    int fd = -1;
#if defined(O_TMPFILE)
    fd = open(dir, O_TMPFILE | O_RDWR | O_CLOEXEC | (dio ? O_DIRECT : 0), 0600);
    if (fd < 0 && dio) fd = open(dir, O_TMPFILE | O_RDWR | O_CLOEXEC, 0600), dio = 0;
#endif
    if (fd < 0) { char p[600]; snprintf(p, sizeof p, "%s/.vk_shadow.XXXXXX", dir); fd = mkstemp(p); if (fd >= 0) unlink(p); dio = 0;   // (unlinked: nothing to delete, ever)
#if defined(O_DIRECT)
        if (fd >= 0 && dio) { int fl = fcntl(fd, F_GETFL); if (fl >= 0 && !fcntl(fd, F_SETFL, fl | O_DIRECT)) dio = 1; }
#endif
    }
    if (fd < 0) return -1;
    if (ftruncate(fd, (off_t)g_sh.size)) { close(fd); return -2; }
    g_npt = (size_t)((g_sh.size / pg + SH_CH - 1) / SH_CH); if (!(g_pt = calloc(g_npt ? g_npt : 1, sizeof *g_pt))) { close(fd); return -3; }
    g_nwt = (size_t)((g_sh.size >> SH_GRAN_SHIFT) + 1); if (!(g_wt = calloc(g_nwt, sizeof *g_wt))) { close(fd); return -3; }
    if (posix_memalign((void **)&g_iobuf, pg > 4096 ? pg : 4096, SH_IO)) { close(fd); return -3; }
    g_sh.fd = fd; g_sh.dio = dio;
    pthread_t t; pthread_attr_t at; pthread_attr_init(&at); pthread_attr_setdetachstate(&at, PTHREAD_CREATE_DETACHED);
    if (pthread_create(&t, &at, sh_worker, NULL)) { pthread_attr_destroy(&at); return -4; }
    pthread_attr_destroy(&at);
    g_sh.on = 1; return 0;
}
static void sh_config(void) {   // the switches (start-up)
    const char *e;
    g_sh.min = (uint64_t)((e = getenv("DM_VK_SHADOW_MIN_KB")) && atoi(e) > 0 ? atoi(e) : 256) << 10;
    g_sh.age_ns = (uint64_t)((e = getenv("DM_VK_SHADOW_AGE_MS")) && atoi(e) >= 0 ? atoi(e) : 2000) * 1000000ull;
    g_sh.bps = (uint64_t)((e = getenv("DM_VK_SHADOW_MBPS")) && atoi(e) > 0 ? atoi(e) : 256) << 20;
    g_sh.max_bytes = (uint64_t)((e = getenv("DM_VK_SHADOW_MAX_MB")) && atoi(e) > 0 ? atoi(e) : 6144) << 20;
    g_sh.free_min = (uint64_t)((e = getenv("DM_VK_SHADOW_FREE_MB")) && atoi(e) >= 0 ? atoi(e) : 4096) << 20;
    g_sh.tick_runs = (e = getenv("DM_VK_SHADOW_TICK_RUNS")) && atoi(e) > 0 ? atoi(e) : 64;
    g_sh.maps_max = (e = getenv("DM_VK_SHADOW_MAPS")) && atoi(e) >= 0 ? atoi(e) : 40000;
    g_sh.quiet_ns = (uint64_t)((e = getenv("DM_VK_SHADOW_QUIET_MS")) && atoi(e) >= 0 ? atoi(e) : 500) * 1000000ull;
}
static void sh_line(char *o, size_t n) {
    snprintf(o, n, "shadow file: %.0f MB mapped from it now (%llu swaps, %llu put off: written near lately; given up: %llu written meanwhile, %llu placed, %llu given back or finalized, %llu mmap, %llu at the mapping limit), "
        "%.0f MB noted, %.0f MB written (%.0f MB not written: the file held them), %.0f MB punched; notes skipped %llu in flux, %llu placed; refused %llu no room, %llu I/O errors; "
        "%llu writes held; file %.0f MB on disk, %.0f MB free there%s; %d mappings; worker %.1f s, swaps %.1f ms",
        atomic_load(&c_mapped) / 1048576.0, (unsigned long long)atomic_load(&c_swaps), (unsigned long long)atomic_load(&c_postponed), (unsigned long long)atomic_load(&c_ab_written), (unsigned long long)atomic_load(&c_ab_placed),
        (unsigned long long)atomic_load(&c_ab_gone), (unsigned long long)atomic_load(&c_ab_map), (unsigned long long)atomic_load(&c_ab_maps), atomic_load(&c_noted) / 1048576.0,
        atomic_load(&c_written) / 1048576.0, atomic_load(&c_reused) / 1048576.0, atomic_load(&c_punched) / 1048576.0, (unsigned long long)atomic_load(&c_flux),
        (unsigned long long)atomic_load(&c_skip_placed), (unsigned long long)atomic_load(&c_full), (unsigned long long)atomic_load(&c_ioerr), (unsigned long long)atomic_load(&c_held),
        sh_file_bytes() / 1048576.0, sh_free_bytes() / 1048576.0, g_sh.dio ? ", direct I/O" : "", atomic_load(&g_maps), atomic_load(&c_worker_ns) / 1e9, atomic_load(&c_tick_ns) / 1e6);
}
// ---- end of the shadow core ----

#ifndef VK_SHADOW_TEST
// ---- the glue: the replay's textures and pools ----
void svc_alias_range(uint8_t **base, size_t *size);   // (hle_svc.c)
static int gl_unplaced(uint64_t lo, uint64_t n) {   // no GPU buffer over [lo, lo+n): a chunk not placed, or granules of a pool placed by parts none of which is placed
    vkpool_t *p = vk_pool_of(lo, n);
    if (!p || p->state == POOL_PLACED || p->ovl || p->part) return 0;
    if (p->state == POOL_PARTS) { if (!p->gran || !p->placed) return 0; for (uint64_t g = (lo - p->base) / p->gran; g <= (lo + n - 1 - p->base) / p->gran && g < p->ngran; g++) if (p->placed[g]) return 0; }
    return 1;
}
static uint64_t gl_unreadable(const void *p, uint64_t n) { return vk_unreadable_at(p, n); }
void vk_shadow_init(void) {
    const char *e = getenv("DM_VK_SHADOW"); if (!e || !atoi(e)) return;
    int live, used; uint64_t adds; nvn_dirty_slot_stats(&live, &used, &adds);
    if (adds) { LOG("[RED] [vk] shadow (DM_VK_SHADOW): the dirty-page tracker has its hooks already, ahead of the shadow's: off"); return; }
    uint8_t *ab; size_t an; svc_alias_range(&ab, &an); uint64_t hb = 0; size_t hn = 0; dm_guest_heap_range(&hb, &hn);
    if (!ab) { LOG("[RED] [vk] shadow (DM_VK_SHADOW): no guest block yet: off"); return; }
    uint64_t base = (uint64_t)(uintptr_t)ab >= (256ull << 20) ? (uint64_t)(uintptr_t)ab - (256ull << 20) : (uint64_t)(uintptr_t)ab;   // (the guest block's start: hle_svc.c mem_init)
    uint64_t top = (hb > base ? hb : (uint64_t)(uintptr_t)ab + an) + (32ull << 30);   // past the heap region's end (the file is sparse: its size costs nothing)
    int vk_mem_pressure(void); sh_config(); g_sh.env = (sh_env){gl_unplaced, gl_unreadable, vk_mem_pressure};
    const char *dir = getenv("DM_VK_SHADOW_DIR"); if (!dir || !*dir) dir = getenv("DM_TEMP"); if (!dir || !*dir) dir = getenv("TMPDIR"); if (!dir || !*dir) dir = ".";
    const char *d = getenv("DM_VK_SHADOW_DIO"); int dio = !(d && !atoi(d));
    dm_fault_hook_register(sh_fault); dm_guest_write_prep_register(sh_prep);   // (before nvn_dirty's: vk_init runs before the first pool)
    int r = sh_open(base, top - base, dir, dio);
    if (r) { LOG("[RED] [vk] shadow (DM_VK_SHADOW): no file in %s (%d, errno %d): off", dir, r, errno); return; }
    LOG("[vk] shadow file in %s (unlinked, %.1f GB of guest addresses from %#llx, %s, pages of %llu KB): textures of %llu KB or more, %llu ms after their upload, at most %llu MB/s, %llu MB on disk, "
        "%llu MB left free; %d swaps a present, none past %d mappings", dir, g_sh.size / 1073741824.0, (unsigned long long)g_sh.base, g_sh.dio ? "direct I/O" : "buffered I/O", (unsigned long long)(g_sh.pg >> 10),
        (unsigned long long)(g_sh.min >> 10), (unsigned long long)(g_sh.age_ns / 1000000), (unsigned long long)(g_sh.bps >> 20), (unsigned long long)(g_sh.max_bytes >> 20),
        (unsigned long long)(g_sh.free_min >> 20), g_sh.tick_runs, g_sh.maps_max);
}
int vk_shadow_tex_class(const vkimg_t *im) {
    static uint64_t min; if (!min) { const char *e = getenv("DM_VK_SHADOW_MIN_KB"); min = (uint64_t)(e && atoi(e) > 0 ? atoi(e) : 256) << 10; }
    if (!im || !im->packaged || !im->pkg_mem || im->sp || im->dn) return VK_SHC_NOT_PACKAGED;
    if (im->pkg_len < min) return VK_SHC_SMALL;
    if (im->flags & (NVN_TEXTURE_FLAGS_LINEAR | NVN_TEXTURE_FLAGS_LINEAR_RENDER_TARGET)) return VK_SHC_LINEAR;
    if (im->gpu_w) return VK_SHC_GPU_WRITTEN;   // (its image holds what its storage does not: a later upload is not this one's business)
    if (!im->pkg_span) return VK_SHC_UNTRACKED;
    if (!gl_unplaced((uint64_t)(uintptr_t)im->pkg_mem, im->pkg_len)) return VK_SHC_PLACED;
    if (im->pkg_released) return VK_SHC_RELEASED;
    if (nvn_dirty_since_span(im->pkg_span, im->pkg_mem, im->pkg_len, im->pkg_stamp)) return VK_SHC_WRITTEN;
    return VK_SHC_OK;
}
void vk_shadow_note_tex(const vkimg_t *im) {
    if (!g_sh.on || vk_shadow_tex_class(im) != VK_SHC_OK) return;
    sh_note((uint64_t)(uintptr_t)im->pkg_mem, (uint64_t)(uintptr_t)im->pkg_mem + im->pkg_len, im->pkg_stamp, im->pkg_span);
}
// The physical pages behind a sparse texture's range just uploaded: each 64 KB tile whose 16 pages are consecutive memory (nvn_sparse_tile_phys), a note of its own (the worker joins
// neighbours). DM_VK_SHADOW_SPARSE=0: none
void vk_shadow_note_sparse(const void *pool, uint64_t a, uint64_t n, uint64_t stamp) {
    static int on = -1; if (on < 0) { const char *e = getenv("DM_VK_SHADOW_SPARSE"); on = !(e && !atoi(e)); }
    if (!g_sh.on || !on || !pool || !n) return;
    for (uint64_t t = a & ~0xFFFFull; t < a + n; t += 0x10000) {
        const uint8_t *p = nvn_sparse_tile_phys(pool, t); if (!p) continue;
        const void *span = nvn_dirty_span(p, 0x10000); if (span) sh_note((uint64_t)(uintptr_t)p, (uint64_t)(uintptr_t)p + 0x10000, stamp, span);
    }
}
void vk_shadow_tick(void) { if (g_sh.on) sh_tick_(); }
void vk_shadow_drop(uint64_t lo, uint64_t n) { if (g_sh.on && n) { pthread_mutex_lock(&g_pt_mu); sh_punch(lo, lo + n); pthread_mutex_unlock(&g_pt_mu); } }
void dm_guest_released(uintptr_t lo, uintptr_t hi) { sh_released((uint64_t)lo, (uint64_t)hi); }
void dm_guest_unshadow(uintptr_t lo, uintptr_t hi) { sh_unshadow((uint64_t)lo, (uint64_t)hi, 0); }
void vk_shadow_log(void) { if (!g_sh.on) return; char b[1200]; sh_line(b, sizeof b); VKLOG("%s", b); }
#endif
