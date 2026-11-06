"""The shadow file of the Vulkan backend (nxnative/dm/android/gfx_vk_shadow.c, DM_VK_SHADOW): guest pages of uploaded textures written once to an unlinked file and the file
mapped over them MAP_PRIVATE, so that the kernel can drop them as clean page cache. Built on the host from the real code: the shadow core (gfx_vk_shadow.c with VK_SHADOW_TEST, which
leaves out the glue to the replay's textures and pools), the real dirty-page tracker (gfx/nvn/nvn_dirty.c) and the host's real page fault dispatch (host/crash.c: the hooks in their
registration order, the shadow's before the tracker's). The checks: the range reads the same bytes and is the file's (mincore: not anonymous), writers held by the swap's window lose no
write, a write after the swap moves only its page back to anonymous memory and an upload after it reads the right bytes, a second note writes only what changed (the file held the
rest), a range the game gives back is punched and reads as new memory, placed or written ranges are left alone, and a full disk maps nothing."""
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

DM = Path(__file__).resolve().parents[1] / 'nxnative/dm'

MAIN = r'''
#include <assert.h>
#include <signal.h>
#include <stdio.h>
#include <sys/mman.h>
#include <mach/mach.h>
#include "SHADOW_C"
// the real tracker, its mprotect interposed: a writer's release (its page taken, its mprotect to writable not landed yet) can be held in flight for the race below
static _Atomic int g_race_arm, g_race_inflight, g_race_go, g_race_stored; static _Atomic long g_race_tid;
int t_mprotect(void *a, size_t n, int prot);
#define mprotect t_mprotect
#include "DIRTY_C"
#undef mprotect
int t_mprotect(void *a, size_t n, int prot) {
    if (prot == (PROT_READ | PROT_WRITE) && atomic_load(&g_race_arm) && sh_tid() == atomic_load(&g_race_tid)) {
        atomic_store(&g_race_arm, 0); atomic_store(&g_race_inflight, 1);
        while (!atomic_load(&g_race_go)) sh_nap(100000);   // (in the fault handler: atomics and nanosleep)
    }
    return mprotect(a, n, prot);
}
module_t *g_mod; int g_nmod; module_t *module_of(const void *p) { (void)p; return NULL; }
void crash_install(void);
static int g_placed; static uint8_t *g_hole;   // the test's pool state: placed (no shadow), a page the game has not mapped
static int t_unplaced(uint64_t lo, uint64_t n) { (void)lo; (void)n; return !g_placed; }
static uint64_t t_unreadable(const void *p, uint64_t n) {   // (as gfx_vk_core.c's Apple path: a byte of each page read without a fault)
    uint64_t pg = g_sh.pg;
    for (uint64_t o = 0; o < n; o += pg) { uint8_t b; vm_size_t got = 0; if (vm_read_overwrite(mach_task_self(), (vm_address_t)((uintptr_t)p + o), 1, (vm_address_t)(uintptr_t)&b, &got) != KERN_SUCCESS) return o; }
    return n;
}
static void fill(uint8_t *p, size_t n, uint32_t seed) { for (size_t i = 0; i < n; i += 4) { uint32_t v = (uint32_t)i * 2654435761u ^ seed; memcpy(p + i, &v, 4); } }
static int same(const uint8_t *p, size_t off, size_t n, uint32_t seed) { for (size_t i = off; i < off + n; i += 4) { uint32_t v = (uint32_t)i * 2654435761u ^ seed, w; memcpy(&w, p + i, 4); if (v != w) return 0; } return 1; }
// pages of [p, p+n) that are anonymous memory now (mincore: copied or anonymous), the others are the file's
static size_t anon_pages(const uint8_t *p, size_t n) { size_t np = n / g_sh.pg, k = 0; char *v = calloc(np, 1); assert(!mincore((void *)p, n, v)); for (size_t i = 0; i < np; i++) k += (v[i] & (MINCORE_ANONYMOUS | MINCORE_COPIED)) != 0; free(v); return k; }
static void wait_ready(size_t want) { for (int i = 0; i < 4000; i++) { pthread_mutex_lock(&g_q_mu); size_t r = g_nready, c = g_ncand; pthread_mutex_unlock(&g_q_mu); if (r >= want && !c) return; usleep(1000); } }
static uint64_t file_blocks(void) { struct stat st; fstat(g_sh.fd, &st); return (uint64_t)st.st_blocks * 512; }
static uint8_t *g_arena; static size_t ARENA;
static const void *protect_note(uint8_t *p, size_t n) {   // as an upload: protected, the stamp after it, noted
    nvn_dirty_protect(p, n); uint64_t st = nvn_dirty_now(); const void *span = nvn_dirty_span(p, n); assert(span);
    sh_note((uint64_t)(uintptr_t)p, (uint64_t)(uintptr_t)p + n, st, span); return span;
}
// writers held by the window: each writes the words of its own 256 KB once, in order, when the window is open (SH_TEST_IN_WINDOW), so every store meets the swap
enum { NW = 4, WREG = 256 << 10 };
static uint8_t *g_wbase; static _Atomic int g_go, g_wdone;
static void *writer(void *arg) {
    int t = (int)(intptr_t)arg; volatile uint32_t *w = (volatile uint32_t *)(g_wbase + (size_t)t * WREG);
    while (!atomic_load(&g_go)) usleep(100);
    for (uint32_t j = 0; j < WREG / 4; j += 61) w[j] = 0xA0000000u | j;   // (a store every 244 bytes: every page of the region)
    atomic_fetch_add(&g_wdone, 1); return NULL;
}
static void in_window(void) { atomic_store(&g_go, 1); usleep(50000); }
// The race the writers' announcements close (wt_mark): a store faults before the window opens, the tracker takes its page, and its release lands inside the window after the swap's last
// check; the store then goes into the old page, which the file is mapped over. Without announcements (quiet 0) it is lost; with them the swap is put off, then given up (written)
static uint8_t *g_race_page;
static void *race_writer(void *arg) { (void)arg; atomic_store(&g_race_tid, sh_tid()); atomic_store(&g_race_arm, 1); g_race_page[4] = 0x77; atomic_store(&g_race_stored, 1); return NULL; }
static void race_hook(void) { atomic_store(&g_race_go, 1); while (!atomic_load(&g_race_stored)) sh_nap(100000); }   // (inside the window: the release lands, the store with it)
static int race(uint64_t quiet_ms) {
    long pg = getpagesize(); g_sh.quiet_ns = quiet_ms * 1000000ull; uint8_t *t = g_arena + (56 << 20); const size_t N = 256 << 10; fill(t, N, 11);
    protect_note(t, N); wait_ready(1); g_race_page = t + pg; uint8_t was = g_race_page[4];
    pthread_t w; pthread_create(&w, NULL, race_writer, NULL); while (!atomic_load(&g_race_inflight)) usleep(100);   // (its release held in flight, its page taken)
    uint64_t sw0 = atomic_load(&c_swaps), po0 = atomic_load(&c_postponed);
    g_test_in_window = race_hook; sh_tick_(); g_test_in_window = NULL;
    atomic_store(&g_race_go, 1); pthread_join(w, NULL);   // (put off: the release lands now, outside any window)
    int lost = g_race_page[4] != 0x77;
    printf("race (quiet %llu ms): %s, %llu swapped, %llu put off; the byte %#x (was %#x)\n", (unsigned long long)quiet_ms, lost ? "the store LOST" : "the store kept",
           (unsigned long long)(atomic_load(&c_swaps) - sw0), (unsigned long long)(atomic_load(&c_postponed) - po0), g_race_page[4], was);
    if (quiet_ms) { usleep((useconds_t)(quiet_ms * 1000 + 50000)); uint64_t ab0 = atomic_load(&c_ab_written); sh_tick_();   // (quiet again: the run is tried, and given up: written)
        assert(atomic_load(&c_ab_written) == ab0 + 1 && g_race_page[4] == 0x77); }
    return lost;
}
// The stress: threads write slots of a range at random, in bursts, while the range is noted, written to the file and swapped over and over (and the host writes into it through
// dm_guest_write_prepare). Slot k of thread t holds the last value t wrote there: any lost write shows
enum { SW = 3, SSLOTS = (1 << 20) / 4 };   // a megabyte of 4-byte slots a thread
static uint32_t g_last[SW][SSLOTS]; static _Atomic int g_sstop;
static void *swriter(void *arg) {
    int t = (int)(intptr_t)arg; volatile uint32_t *w = (volatile uint32_t *)(g_wbase + (size_t)t * (SSLOTS * 4)); uint32_t x = 0x9e3779b9u * (uint32_t)(t + 1), gen = 1;
    while (!atomic_load(&g_sstop)) {
        int burst = 1 + (int)(x % 64);
        for (int i = 0; i < burst; i++) { x ^= x << 13; x ^= x >> 17; x ^= x << 5; uint32_t k = x % SSLOTS; uint32_t v = (gen++ << 8) | (uint32_t)t;
            if (t == 2 && (i & 7) == 0) { dm_guest_write_prepare((void *)&w[k], 4); memcpy((void *)&w[k], &v, 4); } else w[k] = v;   // (thread 2 also as the host: prepared writes)
            g_last[t][k] = v; }
        usleep(2000 + x % 60000);   // (bursts a few ms to 60 ms apart: some swaps meet a writer in their window, some none)
    }
    return NULL;
}
static int stress(void) {
    long pg = getpagesize(); g_wbase = g_arena + (8 << 20); memset(g_wbase, 0, SW * SSLOTS * 4); g_sh.quiet_ns = 20000000ull;   // (20 ms: writers announced lately put a swap off)
    pthread_t th[SW]; for (int t = 0; t < SW; t++) pthread_create(&th[t], NULL, swriter, (void *)(intptr_t)t);
    for (int round = 0; round < 200; round++) {
        protect_note(g_wbase, SW * SSLOTS * 4); wait_ready(1); usleep(500 + (round % 7) * 300); sh_tick_();
        if (round % 50 == 49) { uint64_t lo = (uint64_t)(uintptr_t)g_wbase; nvn_dirty_protect(g_wbase, SW * SSLOTS * 4); sh_note(lo, lo + (uint64_t)(pg * 8), nvn_dirty_now(), nvn_dirty_span(g_wbase, SW * SSLOTS * 4)); }
    }
    atomic_store(&g_sstop, 1); for (int t = 0; t < SW; t++) pthread_join(th[t], NULL);
    sh_tick_();
    long lost = 0, n = 0;
    for (int t = 0; t < SW; t++) { const uint32_t *w = (const uint32_t *)(g_wbase + (size_t)t * (SSLOTS * 4)); for (uint32_t k = 0; k < SSLOTS; k++) { n += g_last[t][k] != 0; lost += w[k] != g_last[t][k]; } }
    char line[1200]; sh_line(line, sizeof line);
    printf("stress: %llu swaps, %llu put off, %llu given up (written meanwhile), %llu writes held, %ld slots written, %ld lost\n%s\n", (unsigned long long)atomic_load(&c_swaps),
           (unsigned long long)atomic_load(&c_postponed), (unsigned long long)atomic_load(&c_ab_written), (unsigned long long)atomic_load(&c_held), n, lost, line);
    if (lost || atomic_load(&c_swaps) < 10) return 1;
    atomic_store(&g_stop, 1); puts("stress ok"); return 0;
}
int main(int argc, char **argv) {
    const char *dir = argv[1];
    crash_install();
    dm_fault_hook_register(sh_fault); dm_guest_write_prep_register(sh_prep);   // the shadow's hooks first (vk_init), the tracker's at its first range
    long pg = getpagesize(); ARENA = (size_t)64 << 20;
    g_arena = mmap(NULL, ARENA, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON, -1, 0); assert(g_arena != MAP_FAILED);
    nvn_dirty_add_range(g_arena, ARENA);
    setenv("DM_VK_SHADOW_AGE_MS", "0", 1); setenv("DM_VK_SHADOW_FREE_MB", "0", 1); setenv("DM_VK_SHADOW_MBPS", "4096", 1); setenv("DM_VK_SHADOW_QUIET_MS", "0", 1); sh_config();
    g_sh.env = (sh_env){t_unplaced, t_unreadable};
    assert(!sh_open((uint64_t)(uintptr_t)g_arena - (16u << 20), ARENA + (32u << 20), dir, 0));   // (the file's span starts below the arena, as the guest block's does)
    if (argc > 2 && !strcmp(argv[2], "stress")) return stress();
    if (argc > 2 && !strcmp(argv[2], "race")) { int lost = race((uint64_t)atoi(argv[3])); puts(lost ? "race lost" : "race kept"); return 0; }
    // 1. a texture of 8 MB at 1 MB: swapped, the same bytes, the file's (not anonymous), its blocks on disk
    uint8_t *t1 = g_arena + (1 << 20); const size_t N1 = 8 << 20; fill(t1, N1, 1);
    assert(anon_pages(t1, N1) == N1 / (size_t)pg);
    const void *sp1 = protect_note(t1, N1); wait_ready(1); sh_tick_();
    assert(atomic_load(&c_swaps) == 1 && atomic_load(&c_mapped) == N1 && atomic_load(&c_written) == N1);
    assert(same(t1, 0, N1, 1) && anon_pages(t1, N1) == 0 && file_blocks() >= N1);
    { uint32_t w, v = (uint32_t)(4096 * 7) * 2654435761u ^ 1; assert(pread(g_sh.fd, &w, 4, (off_t)sh_off((uint64_t)(uintptr_t)t1 + 4096 * 7)) == 4 && w == v); }
    printf("swap: %zu MB the file's, the same bytes\n", N1 >> 20);
    // 2. a write after the swap: faults, the tracker lets it through, only its page is copied; the texture's next upload reads the new and the old bytes
    uint64_t sw = nvn_dirty_now(); t1[3 * pg + 8] = 0x5a; assert(t1[3 * pg + 8] == 0x5a && anon_pages(t1, N1) == 1);
    assert(nvn_dirty_since_span(sp1, t1, N1, sw) && nvn_dirty_since(t1 + 3 * pg, (size_t)pg, sw));   // (the tracker saw it: the texture is uploaded again)
    { uint8_t *up = malloc(N1); memcpy(up, t1, N1);   // (the upload's read of its storage: the new byte and the old ones)
      assert(up[3 * pg + 8] == 0x5a && same(up, 0, 3 * (size_t)pg, 1) && same(up, 4 * (size_t)pg, N1 - 4 * (size_t)pg, 1)); free(up); }
    // noted again (the re-upload): only the written page goes to the file again, the rest it holds
    uint64_t w0 = atomic_load(&c_written), r0 = atomic_load(&c_reused);
    protect_note(t1, N1); wait_ready(1); sh_tick_();
    assert(atomic_load(&c_swaps) == 2 && atomic_load(&c_written) - w0 == (uint64_t)pg && atomic_load(&c_reused) - r0 == N1 - (uint64_t)pg);
    assert(anon_pages(t1, N1) == 0 && t1[3 * pg + 8] == 0x5a && same(t1, 0, 3 * (size_t)pg, 1));
    printf("re-upload: 1 page written again, %zu MB held by the file\n", (N1 - (size_t)pg) >> 20);
    // 3. writers held by the window: their stores wait in the shadow's hook, then go through the tracker onto the new pages; none is lost
    g_wbase = g_arena + (16 << 20); fill(g_wbase, NW * WREG, 3);
    pthread_t th[NW]; for (int t = 0; t < NW; t++) pthread_create(&th[t], NULL, writer, (void *)(intptr_t)t);
    protect_note(g_wbase, NW * WREG); wait_ready(1);
    g_test_in_window = in_window; uint64_t held0 = atomic_load(&c_held);
    sh_tick_(); g_test_in_window = NULL;
    for (int t = 0; t < NW; t++) pthread_join(th[t], NULL);
    assert(atomic_load(&c_swaps) == 3 && atomic_load(&c_held) > held0);
    long lost = 0, kept = 0;
    for (int t = 0; t < NW; t++) { const uint32_t *w = (const uint32_t *)(g_wbase + (size_t)t * WREG);
        for (uint32_t j = 0; j < WREG / 4; j++) { uint32_t want = j % 61 == 0 ? 0xA0000000u | j : ((uint32_t)(t * WREG + 4 * j) * 2654435761u ^ 3); if (w[j] != want) lost++; else kept++; } }
    printf("window: %llu writes held, %ld words checked, %ld lost\n", (unsigned long long)(atomic_load(&c_held) - held0), kept + lost, lost); assert(!lost);
    // 4. writes before the swap (the note is stale): not swapped, the range stays as it is
    uint8_t *t4 = g_arena + (24 << 20); const size_t N4 = 2 << 20; fill(t4, N4, 4);
    protect_note(t4, N4); wait_ready(1); ((volatile uint8_t *)t4)[100] ^= 1; uint64_t ab0 = atomic_load(&c_ab_written);
    sh_tick_(); assert(atomic_load(&c_ab_written) == ab0 + 1 && anon_pages(t4, N4) == N4 / (size_t)pg); ((volatile uint8_t *)t4)[100] ^= 1; assert(same(t4, 0, N4, 4));
    // written before the worker took it: skipped as in flux, nothing written
    uint64_t fl0 = atomic_load(&c_flux); nvn_dirty_protect(t4, N4); uint64_t st4 = nvn_dirty_now(); ((volatile uint8_t *)t4)[200] ^= 1; ((volatile uint8_t *)t4)[200] ^= 1;   // (volatile: the two stores must happen)
    sh_note((uint64_t)(uintptr_t)t4, (uint64_t)(uintptr_t)t4 + N4, st4, nvn_dirty_span(t4, N4)); wait_ready(0); usleep(300000);
    assert(atomic_load(&c_flux) == fl0 + 1);
    // 5. a placed pool: written maybe, never swapped
    uint8_t *t5 = g_arena + (28 << 20); const size_t N5 = 1 << 20; fill(t5, N5, 5);
    protect_note(t5, N5); wait_ready(1); g_placed = 1; uint64_t pl0 = atomic_load(&c_ab_placed); sh_tick_(); g_placed = 0;
    assert(atomic_load(&c_ab_placed) == pl0 + 1 && anon_pages(t5, N5) == N5 / (size_t)pg);
    // 6. a page the game has not mapped: the range is never read (no fault in the worker), nothing written
    uint8_t *t6 = g_arena + (30 << 20); const size_t N6 = 1 << 20; fill(t6, N6, 6); g_hole = t6 + 2 * pg;
    nvn_dirty_protect(t6, N6); uint64_t st6 = nvn_dirty_now(); assert(!mprotect(g_hole, (size_t)pg, PROT_NONE)); uint64_t gone0 = atomic_load(&c_ab_gone);
    sh_note((uint64_t)(uintptr_t)t6, (uint64_t)(uintptr_t)t6 + N6, st6, nvn_dirty_span(t6, N6)); wait_ready(0); usleep(300000);
    assert(atomic_load(&c_ab_gone) == gone0 + 1); assert(!mprotect(g_hole, (size_t)pg, PROT_READ));
    // 7. given back by the game (guest_release): the windows waited out, the blocks punched, the range new memory again
    uint64_t b0 = file_blocks(), m0 = atomic_load(&c_mapped);
    sh_released((uint64_t)(uintptr_t)t1, (uint64_t)(uintptr_t)t1 + N1);
    assert(mmap(t1, N1, PROT_NONE, MAP_PRIVATE | MAP_ANON | MAP_FIXED, -1, 0) == t1);   // (hle_svc.c guest_release)
    assert(atomic_load(&c_mapped) == m0 - N1 && atomic_load(&c_punched) >= N1);
    { uint8_t z[4096]; assert(pread(g_sh.fd, z, sizeof z, (off_t)sh_off((uint64_t)(uintptr_t)t1 + (2 << 20))) == (ssize_t)sizeof z); for (size_t i = 0; i < sizeof z; i++) assert(!z[i]); }   // (a hole: its blocks freed, APFS counts them after the write-back)
    (void)b0;
    assert(!mprotect(t1, N1, PROT_READ | PROT_WRITE)); for (size_t i = 0; i < N1; i += 4096) assert(!t1[i]);
    printf("given back: %llu MB punched, the file %llu MB on disk\n", (unsigned long long)(atomic_load(&c_punched) >> 20), (unsigned long long)(file_blocks() >> 20));
    // a run written before the release and swapped after it: the range is checked again (here unreadable: given up)
    uint8_t *t7 = g_arena + (40 << 20); const size_t N7 = 1 << 20; fill(t7, N7, 7);
    protect_note(t7, N7); wait_ready(1);
    sh_released((uint64_t)(uintptr_t)t7, (uint64_t)(uintptr_t)t7 + N7); assert(mmap(t7, N7, PROT_NONE, MAP_PRIVATE | MAP_ANON | MAP_FIXED, -1, 0) == t7);
    gone0 = atomic_load(&c_ab_gone); sh_tick_(); assert(atomic_load(&c_ab_gone) == gone0 + 1);
    // 10. a write announced in a granule of the range (a fault on the page before it, which is not the run's): the swap is put off, and made once the granule has been quiet
    { uint8_t *g = g_arena + (50 << 20); const size_t N = 128 << 10; fill(g, 256 << 10, 10); g_sh.quiet_ns = 300000000ull;
      uint8_t *t10 = g + (64 << 10); nvn_dirty_protect(g, 64 << 10); nvn_dirty_protect(t10, N); uint64_t st = nvn_dirty_now(); const void *span = nvn_dirty_span(t10, N);   // (two protections: a write before it releases its own)
      sh_note((uint64_t)(uintptr_t)t10, (uint64_t)(uintptr_t)t10 + N, st, span); wait_ready(1);
      uint64_t sw0 = atomic_load(&c_swaps), po0 = atomic_load(&c_postponed); ((volatile uint8_t *)g)[8] ^= 1;   // (the granule's first page: released, not the run's)
      sh_tick_(); assert(atomic_load(&c_postponed) == po0 + 1 && atomic_load(&c_swaps) == sw0 && anon_pages(t10, N) == N / (size_t)pg);
      sh_tick_(); assert(atomic_load(&c_swaps) == sw0);   // (not due yet)
      usleep(350000); sh_tick_(); assert(atomic_load(&c_swaps) == sw0 + 1 && anon_pages(t10, N) == 0 && same(g, 64 << 10, N, 10));
      g_sh.quiet_ns = 0; printf("announced: put off once, swapped %d ms later\n", 350); }
    // 8. no room (the file's cap): nothing written, nothing mapped
    uint8_t *t8 = g_arena + (44 << 20); const size_t N8 = 4 << 20; fill(t8, N8, 8); g_sh.max_bytes = file_blocks() + (1 << 20);
    uint64_t fu0 = atomic_load(&c_full); protect_note(t8, N8); wait_ready(0); usleep(300000);
    assert(atomic_load(&c_full) == fu0 + 1 && anon_pages(t8, N8) == N8 / (size_t)pg);
    // 9. the hash: equal pages equal, a bit apart different
    { uint8_t a[4096], b[4096]; fill(a, sizeof a, 9); memcpy(b, a, sizeof b); assert(sh_hash(a, sizeof a) == sh_hash(b, sizeof b)); b[4000] ^= 0x10; assert(sh_hash(a, sizeof a) != sh_hash(b, sizeof b)); }
    char line[1200]; sh_line(line, sizeof line); printf("%s\n", line);
    atomic_store(&g_stop, 1);
    puts("shadow ok");
    return 0;
}
'''


class Shadow(unittest.TestCase):
    def test_shadow_core(self):
        cc = shutil.which('clang') or shutil.which('cc')
        if not cc:
            self.skipTest('a C compiler is needed')
        import sys
        if sys.platform != 'darwin':
            self.skipTest('the host build of these tests is the Mac one (mincore flags, vm_read_overwrite)')
        with tempfile.TemporaryDirectory(prefix='nx2a_vk_shadow_') as tmp:
            c, exe = Path(tmp) / 't_shadow.c', Path(tmp) / 't_shadow'
            c.write_text(MAIN.replace('SHADOW_C', str(DM / 'android/gfx_vk_shadow.c')).replace('DIRTY_C', str(DM / 'gfx/nvn/nvn_dirty.c')), encoding='utf-8')
            r = subprocess.run([cc, '-std=gnu11', '-O1', '-g', '-Wall', '-Werror', '-Wno-unused-function', '-Wno-unused-variable', '-Wno-deprecated-declarations', '-pthread',
                                '-DVK_SHADOW_TEST', f'-I{DM / "host/tests/memory_profile"}', f'-I{DM / "host"}', f'-I{DM / "gfx/nvn"}', f'-I{DM / "android"}',
                                str(c), str(DM / 'host/crash.c'), '-o', str(exe)], capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
            work = Path(tmp) / 'files'; work.mkdir()
            r = subprocess.run([str(exe), str(work)], capture_output=True, text=True, timeout=120)
            self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
            self.assertIn('shadow ok', r.stdout)
            self.assertIn('0 lost', r.stdout)
            self.assertEqual(list(work.iterdir()), [])   # the file was unlinked at once
            # the race the announcements close, made to happen: lost without them (quiet 0), kept with them
            r = subprocess.run([str(exe), str(work), 'race', '0'], capture_output=True, text=True, timeout=60)
            self.assertEqual(r.returncode, 0, r.stdout + r.stderr); self.assertIn('race lost', r.stdout)
            r = subprocess.run([str(exe), str(work), 'race', '300'], capture_output=True, text=True, timeout=60)
            self.assertEqual(r.returncode, 0, r.stdout + r.stderr); self.assertIn('race kept', r.stdout)
            for _ in range(3):   # (a lost write showed in about one run of five before the writers' announcements)
                r = subprocess.run([str(exe), str(work), 'stress'], capture_output=True, text=True, timeout=300)
                self.assertEqual(r.returncode, 0, r.stdout + r.stderr)   # writers through hundreds of notes and swaps: no write lost, some swaps done
                self.assertIn('stress ok', r.stdout)


if __name__ == '__main__':
    unittest.main()
