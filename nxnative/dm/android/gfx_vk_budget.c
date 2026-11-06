// Android port, Vulkan backend: the memory budget (see gfx_vk_budget.h).
//   The memory line (gfx_vk_core.c vk_mem_log) says what the replay allocated and what the kernel counts for the process (VmRSS, RssAnon, VmSwap, kgsl), not where the resident and
//   swapped memory is: run 13 of Mario Kart World 1.8.0 had 1,953 MB anonymous before its first image, and 4.9 GB anonymous (resident + swapped) at its race beside 2.6 GB of images.
//   DM_VK_SMAPS=<seconds> reads /proc/self/smaps on a thread of its own at most that often and sums each mapping's Rss and Swap by what it is: the guest block (heap, alias region, stack
//   region; file pages there are the shadow file's, gfx_vk_shadow.c), the game's modules, GPU memory the driver maps (/dev/kgsl-3d0), dma-bufs (placed pools), the shader bundle and
//   the RomFS image (file pages: the kernel drops them), host malloc (scudo: the replay's tables, the driver's own allocations, SPIR-V), thread stacks, libraries, the rest; and counts
//   the mappings (the kernel's limit is 65,530). vk_budget_facts logs once what the low-memory killer works with (ro.lmk.*, swap and zram where readable, oom_score_adj).
//   The sparse address space (gfx_vk_tex.c sparse_make) is here as a policy: vk_sparse_fits.
#ifndef VK_BUDGET_TEST
#include "gfx_vk.h"
#include <sys/system_properties.h>
#endif
#include "gfx_vk_budget.h"
#include "dm.h"
#include <ctype.h>
#include <fcntl.h>
#include <pthread.h>
#include <stdlib.h>
#include <sys/resource.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

// ---- the census and the policy (tests: tests/test_vk_budget.py) ----
enum { MC_HEAP, MC_SHADOW, MC_ALIAS, MC_GSTACK, MC_MODULES, MC_KGSL, MC_DMABUF, MC_BUNDLE, MC_ROMFS, MC_MALLOC, MC_TSTACK, MC_LIBS, MC_ANON, MC_FILE, MC_N };
static const char *const MC_NAME[MC_N] = {"guest heap", "shadow file (in the guest block)", "guest alias region", "guest stack region", "game modules", "GPU memory mapped (kgsl)", "dma-bufs (placed pools)",
    "shader bundle", "RomFS image", "host malloc", "thread stacks", "libraries", "other anonymous", "other files"};
typedef struct { uint64_t block_lo, block_hi, heap_lo, heap_hi, alias_lo, alias_hi; int (*module)(uint64_t a); } mc_layout;
typedef struct { uint64_t rss[MC_N], swap[MC_N]; uint32_t n[MC_N]; uint32_t vmas; } mc_census;   // kB, as smaps gives them
static int mc_ends(const char *s, const char *t) { size_t a = strlen(s), b = strlen(t); return a >= b && !strcmp(s + a - b, t); }
static int mc_class(uint64_t lo, uint64_t hi, unsigned long inode, const char *path, const mc_layout *L) {
    int file = inode != 0 || path[0] == '/';
    if (L->block_hi && lo >= L->block_lo && hi <= L->block_hi) {
        if (file) return MC_SHADOW;   // (a file mapped in the guest block: only the shadow file is)
        if (lo >= L->heap_lo && hi <= L->heap_hi) return MC_HEAP;
        if (lo >= L->alias_lo && hi <= L->alias_hi) return MC_ALIAS;
        return MC_GSTACK;   // (the rest of the block: its stack region, and the first 256 MB)
    }
    if (strstr(path, "kgsl")) return MC_KGSL;
    if (strstr(path, "dmabuf") || strstr(path, "dma_buf") || strstr(path, "dma-buf")) return MC_DMABUF;
    if (mc_ends(path, ".dmsv")) return MC_BUNDLE;
    if (mc_ends(path, "RomFS.img") || strstr(path, "/RomFS/")) return MC_ROMFS;
    if (!strncmp(path, "[anon:scudo", 11) || !strncmp(path, "[anon:libc_malloc", 17) || !strncmp(path, "[anon:jemalloc", 14) || !strcmp(path, "[heap]") || !strncmp(path, "[anon:GWP-ASan", 14)) return MC_MALLOC;
    if (!strncmp(path, "[anon:stack_and_tls", 19) || !strcmp(path, "[stack]") || !strncmp(path, "[anon:thread signal stack", 25) || !strncmp(path, "[anon:bionic TLS", 16)) return MC_TSTACK;
    if (L->module && L->module(lo)) return MC_MODULES;
    if (file && (mc_ends(path, ".so") || strstr(path, ".so ") || strstr(path, ".apk"))) return MC_LIBS;
    return file ? MC_FILE : MC_ANON;
}
static uint64_t mc_hex(const char **p) { uint64_t v = 0; for (; isxdigit((unsigned char)**p); (*p)++) v = v * 16 + (uint64_t)(isdigit((unsigned char)**p) ? **p - '0' : (tolower((unsigned char)**p) - 'a' + 10)); return v; }
static uint64_t mc_kb(const char *p) { while (*p == ' ' || *p == '\t') p++; return strtoull(p, NULL, 10); }
// One line of smaps into the census: a mapping's header ("lo-hi perms offset dev inode path") starts a mapping, its "Rss:" and "Swap:" lines add to it. *cur: the mapping's class
static void mc_line(const char *l, const mc_layout *L, mc_census *c, int *cur) {
    if (!strncmp(l, "Rss:", 4)) { if (*cur >= 0) c->rss[*cur] += mc_kb(l + 4); return; }
    if (!strncmp(l, "Swap:", 5)) { if (*cur >= 0) c->swap[*cur] += mc_kb(l + 5); return; }
    if (!isxdigit((unsigned char)l[0])) return;
    const char *p = l; uint64_t lo = mc_hex(&p); if (*p != '-') return; p++; uint64_t hi = mc_hex(&p); if (*p != ' ') return;
    for (int f = 0; f < 3; f++) { while (*p == ' ') p++; while (*p && *p != ' ') p++; }   // perms, offset, dev
    while (*p == ' ') p++; unsigned long inode = strtoul(p, (char **)&p, 10); while (*p == ' ') p++;
    char path[256]; size_t k = 0; while (p[k] && p[k] != '\n' && k < sizeof path - 1) { path[k] = p[k]; k++; } path[k] = 0;
    *cur = mc_class(lo, hi, inode, path, L); c->n[*cur]++; c->vmas++;
}
// The census of a whole smaps file read through fd, a chunk at a time
static int mc_read(int fd, const mc_layout *L, mc_census *c) {
    static char b[1 << 18]; size_t have = 0; int cur = -1; ssize_t r; memset(c, 0, sizeof *c);
    while ((r = read(fd, b + have, sizeof b - 1 - have)) > 0) {
        have += (size_t)r; b[have] = 0; char *s = b, *e;
        while ((e = memchr(s, '\n', have - (size_t)(s - b)))) { *e = 0; mc_line(s, L, c, &cur); s = e + 1; }
        have -= (size_t)(s - b); memmove(b, s, have);
        if (have == sizeof b - 1) have = 0;   // (a line longer than the buffer: dropped)
    }
    return r < 0 ? -1 : 0;
}
static void mc_text(const mc_census *c, char *o, size_t n) {
    int k = snprintf(o, n, "%u mappings:", c->vmas);
    for (int i = 0; i < MC_N && k < (int)n - 96; i++) { if (!c->rss[i] && !c->swap[i]) continue;
        k += snprintf(o + k, n - (size_t)k, "%s %s %.0f MB", k > 14 ? "," : "", MC_NAME[i], c->rss[i] / 1024.0);
        if (c->swap[i]) k += snprintf(o + k, n - (size_t)k, " + %.0f swapped", c->swap[i] / 1024.0);
        if (i == MC_KGSL || i == MC_SHADOW || i == MC_DMABUF) k += snprintf(o + k, n - (size_t)k, " (%u)", c->n[i]); }
}
// The pressure level from /proc/meminfo's text: 2 when MemAvailable is under crit_mb or less than swap_pct / 2 % of the swap is free, 1 under warn_mb or swap_pct %, else 0
static int mp_level(const char *mi, uint64_t warn_mb, uint64_t crit_mb, int swap_pct, uint64_t *avail_mb) {
    long long av = -1, st = -1, sf = -1; const char *p;
    if ((p = strstr(mi, "MemAvailable:"))) av = strtoll(p + 13, NULL, 10) / 1024;
    if ((p = strstr(mi, "SwapTotal:"))) st = strtoll(p + 10, NULL, 10) / 1024;
    if ((p = strstr(mi, "SwapFree:"))) sf = strtoll(p + 9, NULL, 10) / 1024;
    if (avail_mb) *avail_mb = av < 0 ? 0 : (uint64_t)av;
    if (av < 0) return 0;
    int sw = st > 0 && sf >= 0 ? (int)(sf * 100 / st) : 100;
    if ((uint64_t)av < crit_mb || sw * 2 < swap_pct) return 2;
    if ((uint64_t)av < warn_mb || sw < swap_pct) return 1;
    return 0;
}
// The sparse address space: a sparse image of size bytes may be made with used taken of space, the ones in a wider twin (promoted) only past reserve.
// What an image is worth there: the memory it saves against being made whole (its size less its slots: on this device every place a physical tile is mapped at takes a slot of its own,
// no sparseResidencyAliased, and upstream measured about 3 places a page in a race) for the address space it takes. Run 13's GPU-written RG16 / R16 targets map 4-16 tiles: about 1 MB saved a MB;
// its R8 targets in R16 take twice their size: 0.5; its sampled RGBA8 arrays, about half mapped (places, not pages): about 0.5 too. With every image in its own format fitting, the reserve only
// decides the R8 targets: in run 13's order (tests/test_vk_budget.py) none would leave one R8 and one R16 target whole (511 MB) where 1,024 MB leaves two R8 (682 MB), as offline packing would
static int sp_fits(uint64_t used, uint64_t size, int promoted, uint64_t space, uint64_t reserve) {
    uint64_t room = !promoted ? space : space > reserve ? space - reserve : 0;
    return used + size <= room;
}
// ---- end of the census and the policy ----

static uint64_t env_mb(const char *k, long dflt) { const char *e = getenv(k); return (uint64_t)(e && atol(e) >= 0 ? atol(e) : dflt) << 20; }
uint64_t vk_sparse_space(uint64_t device) {
    static uint64_t v = 1; if (v == 1) { const char *e = getenv("DM_VK_SPARSE_SPACE_MB"); v = e && atol(e) > 0 ? (uint64_t)atol(e) << 20 : 0; }
    return v ? v : device;
}
int vk_sparse_fits(uint64_t used, uint64_t size, int promoted, uint64_t device) {
    static uint64_t resv = UINT64_MAX; if (resv == UINT64_MAX) resv = env_mb("DM_VK_SPARSE_PROMO_RESERVE_MB", 1024);
    return sp_fits(used, size, promoted, vk_sparse_space(device), resv);
}
static int read_small(const char *path, char *o, size_t n) {   // a small file's text (0 on success)
    int fd = open(path, O_RDONLY | O_CLOEXEC); if (fd < 0) return -1;
    ssize_t r = read(fd, o, n - 1); close(fd); if (r < 0) return -1; o[r] = 0; return 0;
}
int vk_mem_pressure(void) {
    static uint64_t last, warn, crit; static int lvl, pct;
    if (!warn) { warn = env_mb("DM_VK_PRESSURE_MB", 1536); crit = warn / 2; const char *e = getenv("DM_VK_PRESSURE_SWAP_PCT"); pct = e && atoi(e) >= 0 ? atoi(e) : 25; }
    struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); uint64_t now = (uint64_t)t.tv_sec * 1000 + (uint64_t)t.tv_nsec / 1000000;
    if (last && now - last < 500) return lvl;
    last = now; char mi[4096]; if (read_small("/proc/meminfo", mi, sizeof mi)) return lvl = 0;
    return lvl = mp_level(mi, warn >> 20, crit >> 20, pct, NULL);
}

#ifndef VK_BUDGET_TEST
// ---- the glue: the guest's layout, the census thread, the device's facts ----
void svc_alias_range(uint8_t **base, size_t *size);   // (hle_svc.c)
static int gl_module(uint64_t a) { return module_of((const void *)(uintptr_t)a) != NULL; }
static mc_layout gl_layout(void) {
    mc_layout L = {0}; uint8_t *ab; size_t an; svc_alias_range(&ab, &an); uint64_t hb = 0; size_t hn = 0; dm_guest_heap_range(&hb, &hn);
    if (ab) { L.alias_lo = (uint64_t)(uintptr_t)ab; L.alias_hi = L.alias_lo + an; L.block_lo = L.alias_lo >= (256ull << 20) ? L.alias_lo - (256ull << 20) : L.alias_lo;
              L.heap_lo = hb; L.heap_hi = hb + hn; L.block_hi = L.heap_hi > L.alias_hi ? L.heap_hi : L.alias_hi; }
    L.module = gl_module; return L;
}
static void *census_main(void *arg) {
    (void)arg; pthread_setname_np("vk mem census");
    setpriority(PRIO_PROCESS, 0, 10);   // (below the game's threads, as the prewarm)
    const char *e = getenv("DM_VK_SMAPS"); unsigned every = e && atoi(e) > 0 ? (unsigned)atoi(e) : 30;
    for (;;) {
        struct timespec t0, t1; clock_gettime(CLOCK_MONOTONIC, &t0);
        int fd = open("/proc/self/smaps", O_RDONLY | O_CLOEXEC); mc_census c; mc_layout L = gl_layout();
        if (fd < 0) { LOG("[RED] [vk] memory map: /proc/self/smaps not readable (errno %d): no census", errno); return NULL; }
        int r = mc_read(fd, &L, &c); close(fd); clock_gettime(CLOCK_MONOTONIC, &t1);
        char line[1600]; mc_text(&c, line, sizeof line);
        uint64_t rss = 0, swp = 0; for (int i = 0; i < MC_N; i++) { rss += c.rss[i]; swp += c.swap[i]; }
        LOG("[vk] memory map (smaps%s, %.0f ms): resident %.0f MB, swapped %.0f MB; %s; pressure %d", r ? ", cut short" : "", (t1.tv_sec - t0.tv_sec) * 1e3 + (t1.tv_nsec - t0.tv_nsec) / 1e6,
            rss / 1024.0, swp / 1024.0, line, vk_mem_pressure());
        sleep(every);
    }
    return NULL;
}
void vk_budget_log(void) {
    static int started; const char *e = getenv("DM_VK_SMAPS"); if (started || !e || atoi(e) <= 0) return;
    started = 1; pthread_t t; pthread_attr_t a; pthread_attr_init(&a); pthread_attr_setdetachstate(&a, PTHREAD_CREATE_DETACHED);
    if (pthread_create(&t, &a, census_main, NULL)) LOG("[RED] [vk] memory map: no census thread"); pthread_attr_destroy(&a);
}
static char g_props[1200]; static int g_nprops;
static void prop_val(void *cookie, const char *k, const char *v, uint32_t serial) {
    (void)cookie; (void)serial;
    int want = strstr(k, "lmk") || strstr(k, "nandswap") || strstr(k, "zram") || strstr(k, "swap") || !strcmp(k, "ro.config.low_ram") || strstr(k, "ram_plus") || strstr(k, "memory_ext");
    if (want && g_nprops < (int)sizeof g_props - 80) g_nprops += snprintf(g_props + g_nprops, sizeof g_props - (size_t)g_nprops, "%s%s=%s", g_nprops ? " " : "", k, v);
}
static void prop_one(const prop_info *pi, void *cookie) { __system_property_read_callback(pi, prop_val, cookie); }
void vk_budget_facts(void) {
    static int done; if (done++) return;
    char mi[4096] = "", b[256]; long long mt = -1, st = -1, sf = -1, av = -1; const char *p;
    if (!read_small("/proc/meminfo", mi, sizeof mi)) { if ((p = strstr(mi, "MemTotal:"))) mt = strtoll(p + 9, NULL, 10) >> 10; if ((p = strstr(mi, "SwapTotal:"))) st = strtoll(p + 10, NULL, 10) >> 10;
        if ((p = strstr(mi, "SwapFree:"))) sf = strtoll(p + 9, NULL, 10) >> 10; if ((p = strstr(mi, "MemAvailable:"))) av = strtoll(p + 13, NULL, 10) >> 10; }
    char adj[32] = "?", swp[32] = "?", zd[64] = "not readable", zm[128] = "", zb[96] = "";
    if (!read_small("/proc/self/oom_score_adj", b, sizeof b)) snprintf(adj, sizeof adj, "%d", atoi(b));
    if (!read_small("/proc/sys/vm/swappiness", b, sizeof b)) snprintf(swp, sizeof swp, "%d", atoi(b));
    if (!read_small("/sys/block/zram0/disksize", b, sizeof b)) snprintf(zd, sizeof zd, "%.0f MB", atof(b) / 1048576.0);
    if (!read_small("/sys/block/zram0/mm_stat", b, sizeof b)) { b[strcspn(b, "\n")] = 0; snprintf(zm, sizeof zm, ", mm_stat %s", b); }
    if (!read_small("/sys/block/zram0/backing_dev", b, sizeof b)) { b[strcspn(b, "\n")] = 0; snprintf(zb, sizeof zb, ", backing device %s", b); }
    g_nprops = 0; g_props[0] = 0; __system_property_foreach(prop_one, NULL);
    VKLOG("memory facts: MemTotal %lld MB, MemAvailable %lld MB, SwapTotal %lld MB (free %lld), page %d KB; oom_score_adj %s, swappiness %s; zram0 %s%s%s; properties: %s", mt, av, st, sf,
          getpagesize() >> 10, adj, swp, zd, zm, zb, g_nprops ? g_props : "none readable");
}
#endif
