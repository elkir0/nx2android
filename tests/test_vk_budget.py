"""The memory budget of the Vulkan backend (nxnative/dm/android/gfx_vk_budget.c), built on the host from its portable part (VK_BUDGET_TEST): the census of /proc/self/smaps by what
each mapping is (a synthetic smaps file), the pressure level read from /proc/meminfo's text, and the sparse address space policy, replayed over the sparse textures Mario Kart World
1.8.0 made in run 13 on the Z Fold 8 Ultra (sizes from its log, in the order it made them): the 1,024 MB reserve for images in their own format leaves two 16384x16384 R8 targets
made whole (682 MB), none leaves one R8 and one R16 target whole (511 MB), and a space past the device's 4,095 MB (DM_VK_SPARSE_SPACE_MB, if the driver takes it) none."""
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

DM = Path(__file__).resolve().parents[1] / 'nxnative/dm'

MAIN = r'''
#include <assert.h>
#include <stdio.h>
#include "BUDGET_C"
static int t_module(uint64_t a) { return a >= 0x7340000000ull && a < 0x7344000000ull; }
static const char *SMAPS =
    "6d70000000-6d80000000 ---p 00000000 00:00 0 \n" "Rss:                   0 kB\n" "Swap:                  0 kB\n"
    "7090000000-70a0000000 rw-p 00000000 00:00 0 \n" "Size:   262144 kB\n" "Rss:              102400 kB\n" "Anonymous:        102400 kB\n" "Swap:              51200 kB\n" "VmFlags: rd wr mr mw me ac\n"
    "70a0000000-70a1000000 r--p 1030000000 fd:2e 4242   /data/data/org.nx2a.mkw.test/files/temp/#12 (deleted)\n" "Rss:               10240 kB\n" "Swap:                  0 kB\n"
    "70a1000000-70b0000000 rw-p 00000000 00:00 0 \n" "Rss:               20480 kB\n" "Swap:               1024 kB\n"
    "6d80000000-6d90000000 rw-p 00000000 00:00 0 \n" "Rss:                3072 kB\n" "Swap:                 512 kB\n"
    "7080000000-7081000000 rw-p 00000000 00:00 0 \n" "Rss:                2048 kB\n" "Swap:                  0 kB\n"
    "7340e68000-7344800000 r-xp 00000000 00:00 0 \n" "Rss:               57344 kB\n" "Swap:                128 kB\n"
    "7400000000-7402000000 rw-s 12340000 00:0e 9000   /dev/kgsl-3d0\n" "Rss:               32768 kB\n" "Swap:                  0 kB\n"
    "7402000000-7403000000 rw-s 00000000 00:0f 9001   /dmabuf:qcom,system\n" "Rss:                   0 kB\n" "Swap:                  0 kB\n"
    "7500000000-7520000000 r--s 00000000 fd:30 777   /storage/emulated/0/Android/data/org.nx2a.mkw.test/files/shaders.dmsv\n" "Rss:              153600 kB\n" "Swap:                  0 kB\n"
    "7520000000-7521000000 r--s 0 fd:30 778   /storage/emulated/0/Android/data/org.nx2a.mkw.test/files/RomFS.img\n" "Rss:                4096 kB\n" "Swap:                  0 kB\n"
    "7600000000-7610000000 rw-p 00000000 00:00 0   [anon:scudo:primary]\n" "Rss:              409600 kB\n" "Swap:             102400 kB\n"
    "7610000000-7611000000 rw-p 00000000 00:00 0   [anon:scudo:secondary]\n" "Rss:               51200 kB\n" "Swap:                  0 kB\n"
    "7620000000-7620100000 rw-p 00000000 00:00 0   [anon:stack_and_tls:1234]\n" "Rss:                1024 kB\n" "Swap:                  0 kB\n"
    "7630000000-7630400000 r-xp 00000000 fd:01 55   /vendor/lib64/hw/vulkan.adreno.so\n" "Rss:               20480 kB\n" "Swap:                  0 kB\n"
    "7640000000-7640100000 rw-p 00000000 00:00 0 \n" "Rss:                 512 kB\n" "Swap:                 256 kB\n"
    "7650000000-7650100000 r--p 00000000 fd:01 56   /system/fonts/Roboto.ttf\n" "Rss:                 256 kB\n" "Swap:                  0 kB\n";
static double mb(uint64_t kb) { return kb / 1024.0; }
// run 13's sparse textures (MB of sparse address space; promoted: the R8 targets in R16, twice their 341.2 MB), in the order the game made them
typedef struct { const char *what; double space, whole; int promoted; } cand;
static const cand RUN13[] = {{"6144x6144x2 RGBA8 sampled", 382.5, 382.5, 0}, {"8192x8192 RG16", 341.0, 341.0, 0}, {"8192x8192 RG16", 341.0, 341.0, 0}, {"16384x16384 R8", 682.4, 341.2, 1},
    {"16384x16384 R8", 682.4, 341.2, 1}, {"8192x8192 RG16", 341.0, 341.0, 0}, {"8192x8192 RG16", 341.0, 341.0, 0}, {"3072x3072x2 RGBA8 sampled", 95.6, 95.6, 0}, {"16384x16384 R8", 682.4, 341.2, 1},
    {"8192x8192 R16", 170.5, 170.5, 0}, {"16384x16384 R8", 682.4, 341.2, 1}, {"8192x8192 R16", 170.5, 170.5, 0}};
static double replay(double space_mb, double reserve_mb, int *nwhole) {   // the memory made whole (MB): as sparse_make, each in turn, none coarser (they are GPU-written, or fit)
    uint64_t used = 0, space = (uint64_t)(space_mb * 1048576), resv = (uint64_t)(reserve_mb * 1048576); double whole = 0; *nwhole = 0;
    for (size_t i = 0; i < sizeof RUN13 / sizeof *RUN13; i++) { uint64_t s = (uint64_t)(RUN13[i].space * 1048576);
        if (sp_fits(used, s, RUN13[i].promoted, space, resv)) used += s; else { whole += RUN13[i].whole; (*nwhole)++; } }
    return whole;
}
int main(int argc, char **argv) {
    // the census of a synthetic smaps
    char path[] = "/tmp/nx2a_smaps_XXXXXX"; int fd = mkstemp(path); assert(fd >= 0); unlink(path); assert(write(fd, SMAPS, strlen(SMAPS)) == (ssize_t)strlen(SMAPS)); lseek(fd, 0, SEEK_SET);
    mc_layout L = {0x6d70000000ull, 0x7290000000ull, 0x7090000000ull, 0x7290000000ull, 0x6d80000000ull, 0x7080000000ull, t_module};
    mc_census c; assert(!mc_read(fd, &L, &c)); close(fd);
    assert(c.vmas == 17);
    assert(c.rss[MC_HEAP] == 102400 + 20480 && c.swap[MC_HEAP] == 51200 + 1024 && c.rss[MC_SHADOW] == 10240 && c.n[MC_SHADOW] == 1);
    assert(c.rss[MC_ALIAS] == 3072 && c.swap[MC_ALIAS] == 512 && c.rss[MC_GSTACK] == 2048 && c.n[MC_GSTACK] == 2);   // (the block's first 256 MB too: inaccessible, nothing resident)
    assert(c.rss[MC_MODULES] == 57344 && c.rss[MC_KGSL] == 32768 && c.n[MC_DMABUF] == 1 && c.rss[MC_BUNDLE] == 153600 && c.rss[MC_ROMFS] == 4096);
    assert(c.rss[MC_MALLOC] == 409600 + 51200 && c.swap[MC_MALLOC] == 102400 && c.rss[MC_TSTACK] == 1024 && c.rss[MC_LIBS] == 20480 && c.rss[MC_ANON] == 512 && c.rss[MC_FILE] == 256);
    char line[1600]; mc_text(&c, line, sizeof line); printf("census: %s\n", line);
    assert(strstr(line, "17 mappings:") && strstr(line, "guest heap 120 MB + 51 swapped") && strstr(line, "shadow file (in the guest block) 10 MB (1)") && strstr(line, "host malloc 450 MB + 100 swapped"));
    // the pressure level
    assert(mp_level("MemTotal: 15400000 kB\nMemAvailable: 6000000 kB\nSwapTotal: 8000000 kB\nSwapFree: 7000000 kB\n", 1536, 768, 25, NULL) == 0);
    assert(mp_level("MemAvailable: 1400000 kB\nSwapTotal: 8000000 kB\nSwapFree: 7000000 kB\n", 1536, 768, 25, NULL) == 1);
    assert(mp_level("MemAvailable: 600000 kB\nSwapTotal: 8000000 kB\nSwapFree: 7000000 kB\n", 1536, 768, 25, NULL) == 2);
    assert(mp_level("MemAvailable: 3000000 kB\nSwapTotal: 8000000 kB\nSwapFree: 1500000 kB\n", 1536, 768, 25, NULL) == 1);   // (18 % of the swap free: run 11's kill said "swap is low")
    assert(mp_level("MemAvailable: 3000000 kB\nSwapTotal: 8000000 kB\nSwapFree: 800000 kB\n", 1536, 768, 25, NULL) == 2);
    assert(mp_level("MemAvailable: 3000000 kB\nSwapTotal: 0 kB\nSwapFree: 0 kB\n", 1536, 768, 25, NULL) == 0);   // (no swap: the memory alone)
    // the sparse address space over run 13's textures
    int nw; double w = replay(4095, 1024, &nw); printf("sparse, 4,095 MB, reserve 1,024 MB: %d made whole, %.0f MB\n", nw, w); assert(nw == 2 && w > 682 && w < 683);   // (as measured: 838 MB of own memory, 682 of it these two)
    w = replay(4095, 0, &nw); printf("sparse, 4,095 MB, no reserve: %d made whole, %.0f MB\n", nw, w); assert(nw == 2 && w > 511 && w < 512);
    w = replay(4095, 200, &nw); assert(nw == 2 && w > 511 && w < 512);
    w = replay(4095, 300, &nw); assert(nw == 2 && w > 682);   // (from about 210 MB of reserve on, the third R8 target is refused)
    w = replay(6144, 1024, &nw); printf("sparse, 6,144 MB, reserve 1,024 MB: %d made whole, %.0f MB\n", nw, w); assert(nw == 0);
    // switches: the space override, the reserve
    assert(vk_sparse_space(4095ull << 20) == 4095ull << 20);
    puts("budget ok");
    (void)argc; (void)argv; return 0;
}
'''


class Budget(unittest.TestCase):
    def test_budget(self):
        cc = shutil.which('clang') or shutil.which('cc')
        if not cc:
            self.skipTest('a C compiler is needed')
        with tempfile.TemporaryDirectory(prefix='nx2a_vk_budget_') as tmp:
            c, exe = Path(tmp) / 't_budget.c', Path(tmp) / 't_budget'
            c.write_text(MAIN.replace('BUDGET_C', str(DM / 'android/gfx_vk_budget.c')), encoding='utf-8')
            r = subprocess.run([cc, '-std=gnu11', '-O1', '-Wall', '-Werror', '-Wno-unused-function', '-pthread', '-DVK_BUDGET_TEST', f'-I{DM / "host/tests/memory_profile"}', f'-I{DM / "host"}',
                                f'-I{DM / "android"}', str(c), '-o', str(exe)], capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
            r = subprocess.run([str(exe)], capture_output=True, text=True, timeout=60)
            self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
            self.assertIn('budget ok', r.stdout)


if __name__ == '__main__':
    unittest.main()
