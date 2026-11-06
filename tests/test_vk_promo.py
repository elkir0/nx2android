"""Sparse textures in their format's exact wider twin (gfx_vk_tex.c "promoted formats"): the Adreno 840 gives no sparse residency to its 8-bit
formats, so an R8 texture of a sparse pool is made sparse-resident in R16 (k/255 is k*257/65535), each 64 KB NVN tile the game maps being two
device tiles of the wider format. The region is gfx_vk_tex.c's own, built on the host: the table, the widening and narrowing of texels (every
value of each kind, a half's rounding against Python's), and the tile mapping (the split, and the device tiles of every NVN tile of Mario Kart
World's 16384x16384 R8 texture of 6 levels and of levels that are not whole tiles: inside their level, at the granularity, covering each texel once)."""
from pathlib import Path
import random
import shutil
import struct
import subprocess
import tempfile
import unittest

DM = Path(__file__).resolve().parents[1] / 'nxnative/dm'
TEX = DM / 'android/gfx_vk_tex.c'
VKH = DM / 'third_party/vulkan-headers/include'


def region(src, start, end):
    a = src.index(start)
    return src[a:src.index(end, a)]


MAIN = r'''
static void tiles(uint32_t w, uint32_t h, uint32_t levels, uint32_t tw, uint32_t th, uint32_t dtw, uint32_t dth, long *nbind, long *ntexel) {
    // every NVN tile of every level: its device tiles inside the level, at a multiple of the granularity, of the granularity or reaching the level's edge, covering the tile's texels once
    uint32_t nx, ny; assert(promo_split(tw, th, dtw, dth, &nx, &ny));
    for (uint32_t l = 0; l < levels; l++) {
        uint32_t lw = w >> l ? w >> l : 1, lh = h >> l ? h >> l : 1, ntx = (lw + tw - 1) / tw, nty = (lh + th - 1) / th;
        for (uint32_t ty = 0; ty < nty; ty++) for (uint32_t tx = 0; tx < ntx; tx++) {
            uint64_t cover = 0, want = (uint64_t)(lw - tx * tw < tw ? lw - tx * tw : tw) * (lh - ty * th < th ? lh - ty * th : th); int first = 0;
            for (uint32_t k = 0; k < nx * ny; k++) { uint32_t x, y, bw, bh; if (!promo_sub(tx, ty, tw, th, nx, dtw, dth, k, lw, lh, &x, &y, &bw, &bh)) continue;
                if (k == 0) first = 1;
                assert(x % dtw == 0 && y % dth == 0 && x + bw <= lw && y + bh <= lh && (bw == dtw || x + bw == lw) && (bh == dth || y + bh == lh));
                assert(x >= tx * tw && x + bw <= tx * tw + tw && y >= ty * th && y + bh <= ty * th + th);
                cover += (uint64_t)bw * bh; (*nbind)++; }
            assert(first && cover == want); *ntexel += (long)cover;
        }
    }
}
int main(int argc, char **argv) {
    if (argc > 1 && !strcmp(argv[1], "f2h")) {   // floats (hex bits, one a line) to halves, for Python's rounding to compare with
        char line[64]; while (fgets(line, sizeof line, stdin)) { uint32_t x = (uint32_t)strtoul(line, NULL, 16); float f; memcpy(&f, &x, 4); printf("%04x\n", promo_f2h(f)); }
        return 0;
    }
    // the table: R8 UNORM's twin is R16 UNORM; no SNORM, sRGB, 16-bit UNORM or compressed format has one
    const promo_t *p = promo_of(VK_FORMAT_R8_UNORM); assert(p && p->to == VK_FORMAT_R16_UNORM && p->kind == PROMO_UNORM8 && p->ch == 1 && p->bpt == 1);
    assert(!promo_of(VK_FORMAT_R8_SNORM) && !promo_of(VK_FORMAT_R8_SRGB) && !promo_of(VK_FORMAT_R16_UNORM) && !promo_of(VK_FORMAT_BC1_RGB_UNORM_BLOCK) && !promo_of(VK_FORMAT_D16_UNORM));
    for (size_t i = 0; i < sizeof PROMO / sizeof *PROMO; i++) assert(PROMO[i].bpt % PROMO[i].ch == 0 && promo_of(PROMO[i].from) == &PROMO[i]);
    // UNORM8: k/255 is exactly k*257/65535, and back to k; a 16-bit value drawn into the twin narrows to the nearest 8-bit one
    for (int k = 0; k < 256; k++) { uint8_t b = (uint8_t)k, w[2], o; promo_widen(p, &b, 1, w); uint16_t v; memcpy(&v, w, 2);
        assert((uint64_t)v * 255 == (uint64_t)k * 65535); promo_narrow(p, w, 1, &o); assert(o == k); }
    for (uint32_t v = 0; v < 65536; v++) { uint16_t x = (uint16_t)v; uint8_t o, w[2]; memcpy(w, &x, 2); promo_narrow(p, w, 1, &o); double e = v * 255.0 / 65535.0; assert(o - e <= 0.5 && e - o <= 0.5); }
    // integers: widened as they are (zero- or sign-extended), back to their low bits
    const promo_t *u8 = promo_of(VK_FORMAT_R8G8_UINT), *s8 = promo_of(VK_FORMAT_R8G8B8A8_SINT), *u16 = promo_of(VK_FORMAT_R16_UINT), *s16 = promo_of(VK_FORMAT_R16G16_SINT);
    assert(u8 && s8 && u16 && s16 && u8->to == VK_FORMAT_R16G16_UINT && s8->to == VK_FORMAT_R16G16B16A16_SINT && u16->to == VK_FORMAT_R32_UINT && s16->to == VK_FORMAT_R32G32_SINT);
    for (int k = 0; k < 256; k++) { uint8_t b[2] = {(uint8_t)k, (uint8_t)(255 - k)}, w[4], o[2]; uint16_t v[2];
        promo_widen(u8, b, 1, w); memcpy(v, w, 4); assert(v[0] == k && v[1] == 255 - k); promo_narrow(u8, w, 1, o); assert(o[0] == b[0] && o[1] == b[1]);
        uint8_t c[4] = {(uint8_t)k, 0x80, 0x7f, 0xff}, x[8], q[4]; int16_t sv[4]; promo_widen(s8, c, 1, x); memcpy(sv, x, 8);
        assert(sv[0] == (int8_t)k && sv[1] == -128 && sv[2] == 127 && sv[3] == -1); promo_narrow(s8, x, 1, q); assert(!memcmp(q, c, 4)); }
    for (uint32_t k = 0; k < 65536; k += 7) { uint16_t a = (uint16_t)k; uint8_t w[4], o[2]; uint32_t v; promo_widen(u16, (uint8_t *)&a, 1, w); memcpy(&v, w, 4); assert(v == k);
        promo_narrow(u16, w, 1, o); assert(!memcmp(o, &a, 2));
        int16_t sa[2] = {(int16_t)k, (int16_t)(k ^ 0x8000)}; uint8_t sw[8], so[4]; int32_t sv[2]; promo_widen(s16, (uint8_t *)sa, 1, sw); memcpy(sv, sw, 8);
        assert(sv[0] == sa[0] && sv[1] == sa[1]); promo_narrow(s16, sw, 1, so); assert(!memcmp(so, sa, 4)); }
    // halves: every one exactly as a float, and back the same (a NaN stays a NaN)
    const promo_t *hf = promo_of(VK_FORMAT_R16G16B16A16_SFLOAT); assert(hf && hf->to == VK_FORMAT_R32G32B32A32_SFLOAT && hf->bpt == 8);
    for (uint32_t k = 0; k < 65536; k++) { uint16_t h = (uint16_t)k, back; float f = promo_h2f(h); uint8_t w[4]; memcpy(w, &f, 4); promo_narrow(promo_of(VK_FORMAT_R16_SFLOAT), w, 1, (uint8_t *)&back);
        if ((h & 0x7c00) == 0x7c00 && (h & 0x3ff)) { assert(f != f && (back & 0x7c00) == 0x7c00 && (back & 0x3ff)); continue; }
        assert(back == h);
        int e = (h >> 10) & 31, m = h & 1023; double v = e ? ldexp(1.0 + m / 1024.0, e - 15) : ldexp(m / 1024.0, -14); if (h >> 15) v = -v;
        if (e != 31) assert((double)f == v); }
    // the split: an 8-bit NVN tile of 256x256 is two 16-bit device tiles of 256x128 (1 across, 2 down); RGBA8's 128x128 two of RGBA16's 128x64; a 16-bit 256x128 two 32-bit 128x128 (2 across)
    uint32_t nx, ny; assert(promo_split(256, 256, 256, 128, &nx, &ny) && nx == 1 && ny == 2); assert(promo_split(128, 128, 128, 64, &nx, &ny) && nx == 1 && ny == 2);
    assert(promo_split(256, 128, 128, 128, &nx, &ny) && nx == 2 && ny == 1);
    assert(!promo_split(256, 256, 128, 128, &nx, &ny) && !promo_split(256, 256, 256, 256, &nx, &ny) && !promo_split(256, 256, 96, 128, &nx, &ny) && !promo_split(256, 256, 0, 128, &nx, &ny));
    // Mario Kart World's 16384x16384 R8 texture of 6 levels: every NVN tile (5,460 of them, 341 MB) two device tiles (682 MB were all of it mapped)
    long nb = 0, nt = 0; tiles(16384, 16384, 6, 256, 256, 256, 128, &nb, &nt);
    long want = 0; for (int l = 0; l < 6; l++) want += (16384L >> l) * (16384L >> l);
    assert(nt == want && nb == 2 * 5460); printf("16384x16384 R8, 6 levels: %ld NVN tiles, %ld device tiles\n", nb / 2, nb);
    // levels that are not whole tiles: an edge NVN tile keeps the device tiles inside the level (a 300x200 level: tile (1, 0) is 44 texels wide, its lower device tile 72 rows high)
    nb = nt = 0; tiles(300, 200, 3, 256, 256, 256, 128, &nb, &nt); assert(nt == 300 * 200 + 150 * 100 + 75 * 50);
    { uint32_t x, y, w, h; assert(promo_sub(1, 0, 256, 256, 1, 256, 128, 1, 300, 200, &x, &y, &w, &h) && x == 256 && y == 128 && w == 44 && h == 72);
      assert(!promo_sub(0, 0, 256, 256, 1, 256, 128, 1, 300, 100, &x, &y, &w, &h)); }   // (a level 100 rows high: no lower device tile)
    nb = nt = 0; tiles(1000, 700, 4, 256, 128, 128, 128, &nb, &nt); assert(nt == 1000 * 700 + 500 * 350 + 250 * 175 + 125 * 87);   // (2 across: a 16-bit format's twin)
    // a row of every kind there and back: the identity on the narrower texels
    for (size_t i = 0; i < sizeof PROMO / sizeof *PROMO; i++) { const promo_t *q = &PROMO[i]; uint8_t a[64 * 8], w[64 * 16], b[64 * 8];
        for (size_t j = 0; j < sizeof a; j++) a[j] = (uint8_t)(j * 37 + i);
        if (q->kind == PROMO_HALF) for (size_t j = 0; j < sizeof a; j += 2) a[j + 1] &= 0x7b;   // (no NaN halves: their payload is not kept)
        promo_widen(q, a, 64, w); promo_narrow(q, w, 64, b); assert(!memcmp(a, b, 64 * q->bpt)); }
    puts("promo ok");
    return 0;
}
'''


class PromotedFormats(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory(prefix='nx2a_vk_promo_')
        cls.exe = None
        cc = shutil.which('cc') or shutil.which('clang')
        if not cc or not (VKH / 'vulkan/vulkan_core.h').exists():
            return
        body = region(TEX.read_text(encoding='utf-8'), '// ---- promoted formats', '// ---- end of promoted formats ----')
        code = '#include <assert.h>\n#include <math.h>\n#include <stdint.h>\n#include <stdio.h>\n#include <stdlib.h>\n#include <string.h>\n#define VK_NO_PROTOTYPES\n#include <vulkan/vulkan_core.h>\n' + body + MAIN
        c, exe = Path(cls.tmp.name) / 'promo.c', Path(cls.tmp.name) / 'promo'
        c.write_text(code, encoding='utf-8')
        r = subprocess.run([cc, '-std=gnu11', '-O1', '-Wall', '-Werror', '-Wno-unused-function', '-I' + str(VKH), str(c), '-o', str(exe), '-lm'], capture_output=True, text=True)
        if r.returncode:
            raise AssertionError(r.stdout + r.stderr)
        cls.exe = exe

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def need(self):
        if not self.exe:
            self.skipTest('a C compiler and the vendored Vulkan headers are needed')

    def test_mapping_and_values(self):
        self.need()
        r = subprocess.run([str(self.exe)], capture_output=True, text=True, timeout=120)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn('promo ok', r.stdout)
        self.assertIn('5460 NVN tiles, 10920 device tiles', r.stdout)

    def test_half_rounding(self):
        """promo_f2h (a copy out of an R32F twin into R16F) against Python's half packing: nearest, ties to even, subnormals, overflow to infinity."""
        self.need()
        rnd = random.Random(1234)
        fl = [0.0, -0.0, 1.0, -2.5, 65504.0, 65519.99, 65520.0, 1e9, -1e9, 2.0 ** -24, 2.0 ** -25, 2.0 ** -25 * 1.0001, 2.0 ** -26, 6.1e-5, 1.0 + 2.0 ** -11, 1.0 + 3 * 2.0 ** -11,
              float('inf'), float('-inf')]
        fl += [struct.unpack('<f', struct.pack('<I', rnd.getrandbits(32)))[0] for _ in range(20000)]
        fl += [rnd.uniform(-70000, 70000) for _ in range(5000)] + [rnd.uniform(-1e-4, 1e-4) for _ in range(5000)]
        fl = [x for x in fl if x == x]   # (NaNs: checked in C)
        bits = [struct.unpack('<I', struct.pack('<f', x))[0] for x in fl]
        r = subprocess.run([str(self.exe), 'f2h'], input=''.join(f'{b:08x}\n' for b in bits), capture_output=True, text=True, timeout=120)
        self.assertEqual(r.returncode, 0, r.stderr)
        got = [int(x, 16) for x in r.stdout.split()]
        self.assertEqual(len(got), len(bits))
        for b, g in zip(bits, got):
            x = struct.unpack('<f', struct.pack('<I', b))[0]
            try:
                want = struct.unpack('<H', struct.pack('<e', x))[0]
            except OverflowError:
                want = 0xfc00 if x < 0 else 0x7c00
            self.assertEqual(g, want, f'{x!r} (float bits {b:08x}): {g:04x}, Python {want:04x}')


if __name__ == '__main__':
    unittest.main()
