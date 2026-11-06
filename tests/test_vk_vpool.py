"""Virtual memory pools on Android, without a device (hle_nvn.c, the Android branch): a virtual pool's own range has no CPU alias of
the physical pages behind it there, so its mappings live in the page table only, and an address in the range is translated to the
physical memory behind it (nvn_virt_xlat): an aliased pool's (vmode 1: mapped whole at one offset of a physical pool) at its recorded
offset, a sparse pool's where every page of the range is mapped and consecutive. Textures of a sparse pool are gathered through the
page table (nvn_sparse_copy: unmapped pages as zeros). The code is hle_nvn.c's own, built on the host on a synthetic pool whose range
is reserved and inaccessible, as on the phone: a dereference of it would end the test."""
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

DM = Path(__file__).resolve().parents[1] / 'nxnative/dm'
HLE = DM / 'gfx/nvn/hle_nvn.c'
INC = ['-I' + str(DM / 'gfx/nvn'), '-I' + str(DM / 'gfx/nvn/include'), '-I' + str(DM / 'gfx/tex')]


def region(src, start, end):
    a = src.index(start)
    return src[a:src.index(end, a)]


PRELUDE = r'''
#include <assert.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>
#include "nvn_obj.h"
#define VP_ALIAS 0   // (Android)
#define die(...) (fprintf(stderr, __VA_ARGS__), abort())
#define LOG(...) ((void)0)
'''

MAIN = r'''
int main(void) {
    const uint64_t MB = 1 << 20, KB = 1 << 10;
    // a sparse pool of 64 MB: its range reserved and inaccessible (the test would fault on a read of it), physical memory of 8 MB with a pattern
    pool_t pool; memset(&pool, 0, sizeof pool); pool.flags = 0x200; pool.size = 64 * MB;
    uint8_t *res = mmap(NULL, 64 * MB, PROT_NONE, MAP_ANON | MAP_PRIVATE, -1, 0); assert(res != MAP_FAILED);
    uint8_t *phys = malloc(8 * MB); for (uint64_t i = 0; i < 8 * MB; i++) phys[i] = (uint8_t)(i * 7 + i / 4096);
    struct vpool *v = calloc(1, sizeof *v); v->pool = &pool; v->base = res; v->len = 64 * MB; v->size = 64 * MB; v->pt = calloc((v->size + (2u << 20) - 1) >> 21, sizeof *v->pt);
    pool.vp = v; g_vp = calloc(4, sizeof *g_vp); g_cvp = 4; g_vp[g_nvp++] = v; atomic_store(&g_vp_any, 1);
    uint64_t B = (uint64_t)(uintptr_t)res, out;
    vpt_set(v, 0, 4 * MB, phys);                       // [0, 4 MB) → phys + 0
    vpt_set(v, 8 * MB, 64 * KB, phys + 6 * MB);        // a tile elsewhere in physical memory
    vpt_set(v, 8 * MB + 64 * KB, 64 * KB, phys + MB);  // the next virtual tile: not the next physical one
    // translations: within a mapped run, across its end, across physical pages that do not follow, unmapped, outside every virtual range
    assert(nvn_virt_xlat(B + 100, 1000, &out) == 1 && out == (uint64_t)(uintptr_t)phys + 100);
    assert(nvn_virt_xlat(B + 4 * MB - 4096, 4096, &out) == 1 && out == (uint64_t)(uintptr_t)phys + 4 * MB - 4096);
    assert(nvn_virt_xlat(B + 4 * MB - 16, 32, &out) == -1);
    assert(nvn_virt_xlat(B + 8 * MB + 4, 64 * KB - 8, &out) == 1 && out == (uint64_t)(uintptr_t)phys + 6 * MB + 4);
    assert(nvn_virt_xlat(B + 8 * MB + 60 * KB, 8 * KB, &out) == -1);
    assert(nvn_virt_xlat(B + 20 * MB, 4, &out) == -1 && nvn_virt_xlat(B + 64 * MB - 4, 8, &out) == -1);
    assert(nvn_virt_xlat(0x1234, 4, &out) == 0 && nvn_virt_xlat((uint64_t)(uintptr_t)phys, 64, &out) == 0);
    // the gather a sparse texture's upload makes: mapped pages from physical memory, unmapped ones as zeros; the bytes not mapped counted
    uint8_t *dst = malloc(16 * KB); memset(dst, 0xee, 16 * KB);
    assert(nvn_sparse_copy(&pool, dst, B + 4 * MB - 8 * KB, 16 * KB) == 8 * KB);
    assert(!memcmp(dst, phys + 4 * MB - 8 * KB, 8 * KB)); for (uint64_t i = 8 * KB; i < 16 * KB; i++) assert(dst[i] == 0);
    assert(nvn_sparse_copy(&pool, dst, B + 8 * MB + 60 * KB, 8 * KB) == 0 && !memcmp(dst, phys + 6 * MB + 60 * KB, 4 * KB) && !memcmp(dst + 4 * KB, phys + MB, 4 * KB));
    assert(nvn_sparse_mapped(&pool, B + 8 * MB, 128 * KB) == 1 && nvn_sparse_mapped(&pool, B + 8 * MB, 132 * KB) == 0 && nvn_sparse_mapped(&pool, B + 4 * MB - 4 * KB, 8 * KB) == 0);
    // an unmapping: the pages read as zeros again, and do not translate
    vpt_set(v, 0, 64 * KB, NULL);
    assert(nvn_virt_xlat(B + 100, 10, &out) == -1 && nvn_virt_xlat(B + 64 * KB, 10, &out) == 1 && out == (uint64_t)(uintptr_t)phys + 64 * KB);
    assert(nvn_sparse_copy(&pool, dst, B, 4 * KB) == 4 * KB && !dst[0] && !dst[4095]);
    // an aliased pool (vmode 1): its own range (2 MB reserved, a 1 MB pool) at its physical pool's memory, until it is finalized
    pool_t pool2; memset(&pool2, 0, sizeof pool2); pool2.flags = 0x200; pool2.size = MB;
    uint8_t *res2 = mmap(NULL, 2 * MB, PROT_NONE, MAP_ANON | MAP_PRIVATE, -1, 0); assert(res2 != MAP_FAILED); uint64_t R = (uint64_t)(uintptr_t)res2;
    uint8_t *phys2 = malloc(MB);
    pthread_mutex_lock(&g_vp_mu); valias_add(&pool2, res2, 2 * MB, MB, phys2); pthread_mutex_unlock(&g_vp_mu);
    assert(nvn_virt_xlat(R + 123, 10, &out) == 1 && out == (uint64_t)(uintptr_t)phys2 + 123);
    assert(nvn_virt_xlat(R, MB, &out) == 1 && out == (uint64_t)(uintptr_t)phys2);
    assert(nvn_virt_xlat(R + MB - 4, 8, &out) == -1 && nvn_virt_xlat(R + MB + MB / 2, 4, &out) == -1);   // past the pool, in its reservation
    assert(nvn_virt_xlat(B + 64 * KB, 10, &out) == 1);   // (the sparse pool beside it unchanged)
    valias_release(&pool2);
    assert(nvn_virt_xlat(R + 123, 10, &out) == 0 && g_nva == 0);   // finalized: its range given back
    puts("vpool ok");
    return 0;
}
'''


class VirtualPools(unittest.TestCase):
    def test_translation_and_gather(self):
        cc = shutil.which('cc') or shutil.which('clang')
        if not cc:
            self.skipTest('a C compiler is needed')
        src = HLE.read_text(encoding='utf-8')
        decl = region(src, 'enum { NVRUN = 1024, VPT_LEAF = 512 };', 'static struct vpool *vpool_of(const pool_t *o)')
        table = region(src, '// Page table of a sparse pool (see above). Caller holds g_vp_mu', 'static int vpool_map(struct vpool *v, const map_req *r)')
        reads = region(src, 'static struct vpool *vpool_at(const void *pool, uint64_t a) {', 'static int vpt_walk(')
        mapped = region(src, 'int nvn_sparse_mapped(const void *pool, uint64_t a, uint64_t n) {', 'void nvn_sparse_protect(const void *pool')
        code = PRELUDE + decl + table + reads + mapped + MAIN
        with tempfile.TemporaryDirectory(prefix='nx2a_vk_vpool_') as tmp:
            c, exe = Path(tmp) / 'vpool.c', Path(tmp) / 'vpool'
            c.write_text(code, encoding='utf-8')
            r = subprocess.run([cc, '-std=gnu11', '-O1', '-Wall', '-Werror', '-Wno-unused-function', '-Wno-unused-variable', '-pthread', *INC, str(c), '-o', str(exe)], capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
            r = subprocess.run([str(exe)], capture_output=True, text=True, timeout=60)
            self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
            self.assertIn('vpool ok', r.stdout)


if __name__ == '__main__':
    unittest.main()
