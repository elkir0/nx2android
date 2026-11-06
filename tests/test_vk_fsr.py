"""The present's upscaling (dm/android/gfx_vk_fsr.c, gfx_vk_fsr.h, shaders/fsr/fsr1.glsl): the picture's geometry (the letterbox of each Fold screen, the integer fit, the
pre-rotation's convention, the shaders' input mapping) compiled from gfx_vk_fsr.h itself, the switches' parsing, the shaders in the build (each stage file compiles with glslc
and passes spirv-val, and the 16-bit variants really compute in 16 bits), the module wired into the present, and the CPU reference of the shaders (shaders/fsr/fsr1_ref.c:
a flat picture stays flat, no picture spills into the letterbox, the four turns are the same picture, a crop never reads outside itself)."""
from pathlib import Path
import glob
import os
import shutil
import subprocess
import tempfile
import unittest

DM = Path(__file__).resolve().parents[1] / 'nxnative/dm'
AND = DM / 'android'
HDR = AND / 'gfx_vk_fsr.h'
SHADERS = AND / 'shaders'
REF = SHADERS / 'fsr/fsr1_ref.c'


def tool(name):
    """a tool on PATH, else from ANDROID_NDK_HOME, the repository's tools/ NDK or the NDK the porter installs (None when none)."""
    if shutil.which(name):
        return shutil.which(name)
    roots = [os.environ.get('ANDROID_NDK_HOME', ''), str(DM.parents[1] / 'tools/android-ndk-r27c')] + \
            sorted(glob.glob(os.path.expanduser('~/Library/Application Support/nx2android/toolchains/android-ndk-*')))
    for r in roots:
        for g in glob.glob(os.path.join(r, f'shader-tools/*/{name}')) if r else []:
            return g
    return None


def cc():
    return shutil.which('cc') or shutil.which('clang') or shutil.which('gcc')


class Geometry(unittest.TestCase):
    """gfx_vk_fsr.h's arithmetic, built into a small program with the host compiler."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory(prefix='nx2a_fsr_geo_')
        if not cc():
            raise unittest.SkipTest('a C compiler is needed')
        prog = r'''
#include <stdio.h>
#include "gfx_vk_fsr.h"
int main(void) {
    char op[16]; int a[8]; float f;
    while (scanf("%15s", op) == 1) {
        if (!strcmp(op, "fit")) { for (int i = 0; i < 4; i++) scanf("%d", &a[i]); vkfsr_rect r = vkfsr_fit(a[0], a[1], a[2], a[3]); printf("%d %d %d %d\n", r.x, r.y, r.w, r.h); }
        else if (!strcmp(op, "xf")) {   // rot W H x y w h tx ty: the target pixel's picture pixel, and whether it is inside
            for (int i = 0; i < 7; i++) scanf("%d", &a[i]); int tx, ty; scanf("%d %d", &tx, &ty);
            vkfsr_rect r = {a[3], a[4], a[5], a[6]}; vkfsr_push p; memset(&p, 0, sizeof p); vkfsr_xform(a[0], a[1], a[2], r, &p);
            int lx = p.xa[0] * tx + p.xa[1] * ty + p.xb[0], ly = p.xa[2] * tx + p.xa[3] * ty + p.xb[1];
            printf("%d %d %d\n", lx, ly, lx >= 0 && ly >= 0 && lx < p.xb[2] && ly < p.xb[3]); }
        else if (!strcmp(op, "size")) { for (int i = 0; i < 3; i++) scanf("%d", &a[i]); int pw, ph; vkfsr_target_size(a[0], a[1], a[2], &pw, &ph); printf("%d %d\n", pw, ph); }
        else if (!strcmp(op, "map")) {   // cx cy cw ch pw ph bottom_up lx ly: the input position
            for (int i = 0; i < 7; i++) scanf("%d", &a[i]); int lx, ly; scanf("%d %d", &lx, &ly);
            int crop[4] = {a[0], a[1], a[2], a[3]}; vkfsr_rect r = {0, 0, a[4], a[5]}; vkfsr_push p; vkfsr_map(crop, r, a[6], &p);
            printf("%.6f %.6f %d %d %d %d\n", lx * p.map[0] + p.map[2], ly * p.map[1] + p.map[3], p.clip[0], p.clip[1], p.clip[2], p.clip[3]); }
        else if (!strcmp(op, "crop")) { for (int i = 0; i < 6; i++) scanf("%d", &a[i]); int o[4]; vkfsr_crop(a, a[4], a[5], o); printf("%d %d %d %d\n", o[0], o[1], o[2], o[3]); }
        else if (!strcmp(op, "mode")) { char s[32]; scanf("%31s", s); printf("%d\n", vkfsr_parse_mode(!strcmp(s, "-") ? NULL : s)); }
        else if (!strcmp(op, "rcas")) { char s[32]; int m; scanf("%31s %d", s, &m); printf("%.4f\n", vkfsr_parse_rcas(!strcmp(s, "-") ? NULL : s, m)); }
        else if (!strcmp(op, "k")) { scanf("%f", &f); printf("%.6f\n", vkfsr_rcas_k(f)); }
    }
    return 0;
}
'''
        c = Path(cls.tmp.name) / 'geo.c'; c.write_text(prog, encoding='utf-8'); cls.exe = Path(cls.tmp.name) / 'geo'
        r = subprocess.run([cc(), '-std=gnu11', '-Wall', '-Wextra', '-Werror', f'-I{AND}', str(c), '-o', str(cls.exe), '-lm'], capture_output=True, text=True)
        if r.returncode:
            raise AssertionError(r.stdout + r.stderr)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def ask(self, *lines):
        r = subprocess.run([str(self.exe)], input='\n'.join(lines) + '\n', capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr)
        return [l.split() for l in r.stdout.splitlines()]

    def test_fold_screens(self):
        inner, outer = self.ask('fit 1280 720 2504 2256', 'fit 1280 720 2520 1080')
        self.assertEqual(inner, ['0', '424', '2504', '1408'])   # the run 12 screenshot's band
        self.assertEqual(outer, ['300', '0', '1920', '1080'])
        self.assertEqual(self.ask('fit 1920 1080 2504 2256')[0], ['0', '424', '2504', '1408'])
        self.assertEqual(self.ask('fit 1280 720 1280 720')[0], ['0', '0', '1280', '720'])

    def test_fit_is_exact_where_doubles_lost_a_pixel(self):
        # the blit's s = min(W / cw, H / ch), (int)(cw * s) gave 2959x1664, 1855x1044 and 2800x1574 for these (gfx_vk_fsr.h)
        self.assertEqual(self.ask('fit 1440 810 2960 1848')[0], ['0', '91', '2960', '1665'])
        self.assertEqual(self.ask('fit 1600 900 1856 2160')[0], ['0', '558', '1856', '1044'])
        self.assertEqual(self.ask('fit 1152 648 2800 1752')[0], ['0', '88', '2800', '1575'])
        for cw, ch, W, H in [(1280, 720, 2504, 2256), (1706, 960, 2520, 1080), (1, 1, 7, 3), (3840, 2160, 1080, 2520)]:
            x, y, w, h = map(int, self.ask(f'fit {cw} {ch} {W} {H}')[0])
            self.assertTrue(w == W or h == H, (cw, ch, W, H))   # one side fills the window
            self.assertTrue(0 <= x and 0 <= y and x + w <= W and y + h <= H)
            self.assertLessEqual(abs(w / h - cw / ch), max(1 / h, 1 / w) * cw / ch + 1e-9)   # the aspect kept to a pixel

    def test_rotation_convention(self):
        # VK_SURFACE_TRANSFORM_ROTATE_90_BIT_KHR: the content turned 90 degrees clockwise, the window's top row the image's right column (Android's guide: clip (x, y) -> (-y, x))
        W, H = 2504, 2256
        self.assertEqual(self.ask(f'size 1 {W} {H}')[0], [str(H), str(W)])
        self.assertEqual(self.ask(f'size 2 {W} {H}')[0], [str(W), str(H)])
        full = f'{W} {H} 0 0 {W} {H}'
        corners = {0: [(0, 0, 0, 0), (W - 1, 0, W - 1, 0)], 1: [(H - 1, 0, 0, 0), (H - 1, W - 1, W - 1, 0), (0, 0, 0, H - 1)],
                   2: [(W - 1, H - 1, 0, 0), (0, 0, W - 1, H - 1)], 3: [(0, W - 1, 0, 0), (0, 0, W - 1, 0), (H - 1, W - 1, 0, H - 1)]}
        for rot, cases in corners.items():
            for tx, ty, wx, wy in cases:   # target pixel (tx, ty) shows window pixel (wx, wy)
                self.assertEqual(self.ask(f'xf {rot} {full} {tx} {ty}')[0], [str(wx), str(wy), '1'], (rot, tx, ty))
        # the letterbox: the picture's band on the inner screen, turned 90; the row above it is outside
        self.assertEqual(self.ask(f'xf 1 {W} {H} 0 424 2504 1408 {H - 1 - 424} 0')[0], ['0', '0', '1'])
        self.assertEqual(self.ask(f'xf 1 {W} {H} 0 424 2504 1408 {H - 1 - 423} 0')[0][2], '0')

    def test_rotation_is_a_bijection(self):
        W, H = 7, 5
        for rot in range(4):
            pw, ph = (H, W) if rot in (1, 3) else (W, H)
            got = [tuple(map(int, l[:2])) for l in self.ask(*[f'xf {rot} {W} {H} 0 0 {W} {H} {tx} {ty}' for ty in range(ph) for tx in range(pw)])]
            self.assertEqual(sorted(got), sorted((x, y) for y in range(H) for x in range(W)), rot)

    def test_input_mapping(self):
        # a destination pixel's centre as vkCmdBlitImage maps it: (l + 0.5) * crop / picture - 0.5 from the crop's origin; bottom-up: from the crop's bottom
        x, y, *clip = self.ask('map 0 0 1280 720 2504 1408 0 0 0')[0]
        self.assertAlmostEqual(float(x), 0.5 * 1280 / 2504 - 0.5, places=5); self.assertAlmostEqual(float(y), 0.5 * 720 / 1408 - 0.5, places=5)
        self.assertEqual(clip, ['0', '0', '1279', '719'])
        x, y, *_ = self.ask('map 0 0 1280 720 2504 1408 0 2503 1407')[0]
        self.assertAlmostEqual(float(x), 1279.5 - 0.5 * 1280 / 2504, places=3); self.assertAlmostEqual(float(y), 719.5 - 0.5 * 720 / 1408, places=3)
        x, y, *_ = self.ask('map 100 60 1280 720 2504 1408 1 0 0')[0]
        self.assertAlmostEqual(float(x), 100 + 0.5 * 1280 / 2504 - 0.5, places=4); self.assertAlmostEqual(float(y), 60 + 720 - 0.5 * 720 / 1408 - 0.5, places=3)

    def test_crop_cut_to_the_image(self):
        self.assertEqual(self.ask('crop 0 0 0 0 3840 2160')[0], ['0', '0', '3840', '2160'])        # none: the whole image
        self.assertEqual(self.ask('crop 0 0 1280 720 3840 2160')[0], ['0', '0', '1280', '720'])
        self.assertEqual(self.ask('crop 3000 0 1280 720 3840 2160')[0], ['3000', '0', '840', '720'])
        self.assertEqual(self.ask('crop 5000 0 1280 720 3840 2160')[0], ['0', '0', '3840', '2160'])   # nothing left: the whole image

    def test_switches(self):
        modes = {'-': 2, 'fsr': 2, 'easu': 2, '1': 2, 'bilinear': 1, 'linear': 1, 'off': 0, 'blit': 0, '0': 0, 'fsr2': -1}
        for s, m in modes.items():
            self.assertEqual(int(self.ask(f'mode {s}')[0][0]), m, s)
        self.assertAlmostEqual(float(self.ask('rcas - 2')[0][0]), 0.25)       # auto with fsr: AMD's sample default
        self.assertAlmostEqual(float(self.ask('rcas auto 1')[0][0]), -1)      # auto with bilinear: none
        self.assertAlmostEqual(float(self.ask('rcas off 2')[0][0]), -1)
        self.assertAlmostEqual(float(self.ask('rcas 0.5 1')[0][0]), 0.5)
        self.assertAlmostEqual(float(self.ask('rcas 9 2')[0][0]), 2)          # FSR's range is 0 to 2 stops
        self.assertAlmostEqual(float(self.ask('rcas x 2')[0][0]), 0.25)
        self.assertAlmostEqual(float(self.ask('k 0.25')[0][0]), 2 ** -0.25, places=5)


class Shaders(unittest.TestCase):
    """the stage files build.py compiles (every shaders/*.* file, includes from shaders/fsr/): each compiles and validates, the 16-bit ones compute in 16 bits."""

    STAGES = ['fsr.vert', 'fsr_easu.frag', 'fsr_easu16.frag', 'fsr_rcas.frag', 'fsr_rcas16.frag', 'fsr_blit.frag', 'fsr_ndsr.comp', 'fsr_ndsr16.comp']

    def test_present_in_the_build(self):
        for s in self.STAGES:
            self.assertTrue((SHADERS / s).is_file(), s)
        build = (AND / 'build.py').read_text(encoding='utf-8')
        self.assertIn("(DM / 'android/shaders').glob('*.*')", build)   # every file there is a stage, so the shared code lives in a subdirectory
        self.assertEqual([p.name for p in SHADERS.glob('*.*') if p.suffix not in ('.vert', '.frag', '.comp')], [])
        src = (AND / 'gfx_vk_fsr.c').read_text(encoding='utf-8')
        for s in self.STAGES:
            stem, st = s.rsplit('.', 1)
            self.assertIn(f'#include "{stem}_{st}.spv.h"', src)
            self.assertIn(f'{stem}_{st}_spv', src)
        self.assertTrue((SHADERS / 'fsr/LICENSE').is_file())
        self.assertIn('Advanced Micro Devices', (SHADERS / 'fsr/fsr1.glsl').read_text(encoding='utf-8').splitlines()[0])

    def test_compile_and_validate(self):
        glslc, val, dis = tool('glslc'), tool('spirv-val'), tool('spirv-dis')
        if not glslc or not val:
            self.skipTest('glslc and spirv-val are needed (PATH, ANDROID_NDK_HOME or the toolchain the porter installs)')
        with tempfile.TemporaryDirectory(prefix='nx2a_fsr_spv_') as tmp:
            for s in self.STAGES:
                o = Path(tmp) / (s + '.spv')
                r = subprocess.run([glslc, '--target-env=vulkan1.1', '-O', str(SHADERS / s), '-o', str(o)], capture_output=True, text=True)   # build.py's command
                self.assertEqual(r.returncode, 0, s + r.stderr)
                r = subprocess.run([val, '--target-env', 'vulkan1.1', str(o)], capture_output=True, text=True)
                self.assertEqual(r.returncode, 0, s + r.stdout + r.stderr)
                if dis and s.endswith(('16.frag', '16.comp')):
                    text = subprocess.run([dis, str(o)], capture_output=True, text=True).stdout
                    self.assertIn('OpCapability Float16', text)
                    lines = text.splitlines(); half = {l.split()[0] for l in lines if 'OpTypeFloat 16' in l}
                    half |= {l.split()[0] for l in lines if 'OpTypeVector' in l and l.split()[3] in half}   # its vectors too
                    n16 = sum(1 for l in lines if len(l.split()) > 3 and l.split()[1] == '=' and (l.split()[2].startswith('OpF') or l.split()[2] == 'OpExtInst') and l.split()[3] in half)
                    self.assertGreater(n16, 25, s)   # (EASU: 363 of its 385 float operations, RCAS 31 of 32, when written)

    def test_overlay_turns_as_the_picture(self):
        # gfx_vk_present.c's clip-space turn for the overlay (g_ov_rot, applied by shaders/overlay.vert as mat2(rot.xy, rot.zw)) moves each window pixel's centre to the
        # centre of the target pixel where the picture's passes put that window pixel (gfx_vk_fsr.h: ROT90 (H-1-y, x), ROT180 (W-1-x, H-1-y), ROT270 (y, W-1-x))
        p = (AND / 'gfx_vk_present.c').read_text(encoding='utf-8')
        line = next(l for l in p.splitlines() if 'static const float R[4][4]' in l)
        rows = [[float(v) for v in r.split(',')] for r in line.split('{{', 1)[1].split('}};', 1)[0].split('}, {')]
        W, H = 9, 5
        for rot, (a, b, c, d) in enumerate(rows):
            pw, ph = (H, W) if rot in (1, 3) else (W, H)
            for wy in range(H):
                for wx in range(W):
                    lx, ly = 2 * (wx + 0.5) / W - 1, 2 * (wy + 0.5) / H - 1
                    px, py = a * lx + c * ly, b * lx + d * ly   # mat2(col0 = (a, b), col1 = (c, d))
                    tx, ty = {0: (wx, wy), 1: (H - 1 - wy, wx), 2: (W - 1 - wx, H - 1 - wy), 3: (wy, W - 1 - wx)}[rot]
                    self.assertAlmostEqual(px, 2 * (tx + 0.5) / pw - 1, places=6, msg=(rot, wx, wy))
                    self.assertAlmostEqual(py, 2 * (ty + 0.5) / ph - 1, places=6, msg=(rot, wx, wy))
        vert = (SHADERS / 'overlay.vert').read_text(encoding='utf-8')
        self.assertIn('{ vec4 pos; vec4 uv; float alpha; vec4 rot; }', vert)   # rot at offset 48, as ov_quad's
        self.assertIn('float pos[4], uv[4], alpha, pad[3], rot[4]; } ov_quad;', p)

    def test_present_uses_the_passes(self):
        p = (AND / 'gfx_vk_present.c').read_text(encoding='utf-8')
        for f in ('vk_fsr_path(', 'vk_fsr_begin(', 'vk_fsr_end(', 'vk_fsr_time_start(', 'vk_fsr_time_end(', 'vk_fsr_time_submitted(', 'vk_fsr_time_collect('):
            self.assertIn(f, p)
        self.assertIn('vkCmdBlitImage', p)   # the blit stays: DM_VK_UPSCALE=off, devices without dynamic rendering


BUNDLE = DM.parents[1] / '.gamedump/MarioKartWorld/0400C3F00006E000_build/android/shaders.dmsv'   # (the port's bundle: present on the machine that ported the game)
MKW_NDSR = {   # Mario Kart World 1.8.0's agl_ndsr.pck compute variants by colour shape, read from their SPIR-V when this was written (VKNDSR_ENC 1, LIN1 2, LIN128 3)
    1: '0e351bd8ef324466 131d7dcc054f92b9 446761e211ec30d8 472f4f2b47df0ffa 639fc8644430c165 8b82a01f1fd62cc5 9e3b5c40cf35dbf3 b401e181d9dc0a50 eddc4da7235e31b8 fb0ca0be3aff5d62',
    2: '27ed4aca06572efb 2ebeef7ec37fdecf 2fa9e5020f13a5a5 672650f11b4e2a82 81d3813d8ab8756c aacfad2f403d993a e0c196a9a2aa75bb ee2fca64102e59e7 f0c619522053c2af f50d631b1f382605',
    3: '051422a635fafeee 1aadf4b1fe140cc6 26a45d0d86eeb07f 6269703847d1b0b8 6dbfff7d7a7fe685 91719bec44edd803 a84041e37fade1d8 b35f14364dbad423 d901a683786fec6c fc93a64eafaee3fb'}


class NdsrSwap(unittest.TestCase):
    """gfx_vk_fsr.h's rule for a game's own upscaler (vkndsr_shape over SPIR-V, vkndsr_rects), built into a host tool."""

    @classmethod
    def setUpClass(cls):
        if not cc():
            raise unittest.SkipTest('a C compiler is needed')
        cls.tmp = tempfile.TemporaryDirectory(prefix='nx2a_ndsr_')
        prog = r'''
#include <stdio.h>
#include "gfx_vk_fsr.h"
int main(int argc, char **argv) {
    if (argc == 2) {   // a SPIR-V file: its shape
        FILE *f = fopen(argv[1], "rb"); if (!f) return 3; fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
        uint32_t *w = malloc((size_t)n); if (fread(w, 1, (size_t)n, f) != (size_t)n) return 3; fclose(f);
        printf("%d\n", vkndsr_shape(w, (uint32_t)(n / 4))); return 0;
    }
    uint32_t v[8]; int in[4], out[4]; for (int i = 0; i < 8; i++) v[i] = (uint32_t)strtoul(argv[1 + i], NULL, 0);
    int ok = vkndsr_rects(v, (uint32_t)atoi(argv[9]), (uint32_t)atoi(argv[10]), (uint32_t)atoi(argv[11]), (uint32_t)atoi(argv[12]), in, out);
    printf("%d %d %d %d %d %d %d %d %d\n", ok, in[0], in[1], in[2], in[3], out[0], out[1], out[2], out[3]); return 0;
}
'''
        c = Path(cls.tmp.name) / 'ndsr.c'; c.write_text(prog, encoding='utf-8'); cls.exe = Path(cls.tmp.name) / 'ndsr'
        r = subprocess.run([cc(), '-std=gnu11', '-Wall', '-Wextra', '-Werror', '-Wno-unused-parameter', f'-I{AND}', str(c), '-o', str(cls.exe), '-lm'], capture_output=True, text=True)
        if r.returncode:
            raise AssertionError(r.stdout + r.stderr)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def shape(self, spv):
        r = subprocess.run([str(self.exe), str(spv)], capture_output=True, text=True); self.assertEqual(r.returncode, 0, r.stderr)
        return int(r.stdout)

    def test_rectangles(self):
        ask = lambda *a: subprocess.run([str(self.exe), *map(str, a)], capture_output=True, text=True).stdout.split()
        self.assertEqual(ask(0, 0, 959, 539, 0, 0, 1279, 719, 1280, 720, 1280, 720), ['1', '0', '0', '960', '540', '0', '0', '1280', '720'])   # last texel inclusive
        self.assertEqual(ask(32, 16, 959, 539, 8, 4, 1271, 715, 1280, 720, 1280, 720), ['1', '32', '16', '960', '540', '8', '4', '1272', '716'])
        self.assertEqual(ask(0, 0, 1280, 539, 0, 0, 1279, 719, 1280, 720, 1280, 720)[0], '0')   # one past the input's edge
        self.assertEqual(ask(0, 0, 959, 539, 1, 0, 1279, 719, 1280, 720, 1280, 720)[0], '0')    # the output's origin pushes it past the edge
        self.assertEqual(ask(0, 0, 0xffffffff, 539, 0, 0, 1279, 719, 1280, 720, 1280, 720)[0], '0')   # (no wrap)

    def test_shapes_of_synthetic_shaders(self):
        glslc = tool('glslc')
        if not glslc:
            self.skipTest('glslc is needed')
        # each curve's constants used 8 times and more, as a network's per-channel, per-tile code does; the cap 128 in a min
        enc = 'c = mix(c * 12.92, 1.055 * pow(c, vec3(1.0 / 2.4)) - 0.055, greaterThan(c, vec3(0.0031308)));'
        dec = 'c = mix(c * (1.0 / 12.92), pow((c + 0.055) * (1.0 / 1.055), vec3(2.4)), greaterThan(c, vec3(0.04045)));'
        bodies = {1: enc + 'c = clamp(c, 0.0, 1.0);', 2: enc + dec + 'c = clamp(c, 0.0, 1.0);', 3: enc + dec + 'c = min(c, vec3(128.0));', 0: 'c = c * 2.0;'}
        for want, body in bodies.items():
            src = ('#version 450\nlayout(local_size_x = 8) in;\nlayout(binding = 0) uniform sampler2D t;\nlayout(binding = 1) uniform writeonly image2D o;\n'
                   'layout(push_constant) uniform P { vec4 k[4]; } p;\nvoid main() {\n')
            for i in range(9):   # 9 taps: each curve's constants used 9 times (glslang makes them vectors, used once a tap; the rule takes 8)
                src += f'  {{ vec3 c = texelFetch(t, ivec2(gl_GlobalInvocationID.xy) + ivec2({i}, 0), 0).rgb * p.k[{i % 4}].xyz; {body} imageStore(o, ivec2(gl_GlobalInvocationID.xy) + ivec2({i}, 0), vec4(c, 1.0)); }}\n'
            src += '}\n'
            g, o = Path(self.tmp.name) / f's{want}.comp', Path(self.tmp.name) / f's{want}.spv'
            g.write_text(src, encoding='utf-8')
            r = subprocess.run([glslc, '--target-env=vulkan1.1', str(g), '-o', str(o)], capture_output=True, text=True)   # (unoptimized: one use per channel stays)
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertEqual(self.shape(o), want, body)

    def test_mario_kart_world_variants(self):
        if not BUNDLE.is_file():
            self.skipTest('the Mario Kart World port\'s shader bundle is not on this machine')
        import struct
        want = {int(k, 16): s for s, keys in MKW_NDSR.items() for k in keys.split()}
        got = {}
        with open(BUNDLE, 'rb') as f:
            magic, ver, n = struct.unpack('<4sII', f.read(12)); tab = f.read(32 * n)
            for i in range(n):
                key, st, so, sl, ro, rl, _ = struct.unpack_from('<QIIIIII', tab, 32 * i)
                if key in want:
                    f.seek(so); spv = Path(self.tmp.name) / f'{key:016x}.spv'; spv.write_bytes(f.read(sl)); got[key] = self.shape(spv)
        self.assertEqual(len(got), 30)
        self.assertEqual(got, want)


class Reference(unittest.TestCase):
    """shaders/fsr/fsr1_ref.c, the shaders' CPU mirror, in both precisions."""

    @classmethod
    def setUpClass(cls):
        if not cc():
            raise unittest.SkipTest('a C compiler is needed')
        cls.tmp = tempfile.TemporaryDirectory(prefix='nx2a_fsr_ref_'); cls.exe = {}
        for h in (0, 1):
            exe = Path(cls.tmp.name) / f'ref{h}'
            r = subprocess.run([cc(), '-O2', '-std=gnu11', '-Wall', '-Wextra', '-ffp-contract=off', f'-DFSR_H={h}', str(REF), '-o', str(exe), '-lm'], capture_output=True, text=True)
            if r.returncode and h:   # a host compiler without _Float16: the 32-bit reference alone
                continue
            if r.returncode:
                raise AssertionError(r.stdout + r.stderr)
            cls.exe[h] = exe

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def draw(self, h, px, w, hh, crop, W, H, rot, path, stops, bu=0):
        src, dst = Path(self.tmp.name) / 'in.rgba', Path(self.tmp.name) / 'out.rgba'
        src.write_bytes(px)
        r = subprocess.run([str(self.exe[h]), str(src), str(w), str(hh), *map(str, crop), str(W), str(H), str(rot), path, str(stops), str(bu), '10', str(dst)], capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr)
        return dst.read_bytes()

    @staticmethod
    def pattern(w, h):   # edges, a diagonal, a gradient and noise-like detail
        out = bytearray()
        for y in range(h):
            for x in range(w):
                out += bytes([255 if (x // 7 + y // 5) % 2 else 20, (x * 255) // (w - 1), (37 * x * x + 11 * y * y + 5 * x * y) % 251 if x > y else 90, 255])
        return bytes(out)

    def test_flat_stays_flat(self):
        w, h = 64, 36; px = bytes([200, 100, 30, 255]) * (w * h)
        for hb in self.exe:
            for path, st in (('fsr', 0.25), ('fsr', -1), ('bilinear', -1), ('bilinear', 0.5)):
                o = self.draw(hb, px, w, h, (0, 0, w, h), 125, 113, 0, path, st)
                colours = {o[i:i + 4] for i in range(0, len(o), 4)}
                self.assertEqual(colours, {bytes([200, 100, 30, 255]), bytes([0, 0, 0, 255])}, (hb, path, st))

    def test_letterbox_black_and_turns_agree(self):
        w, h = 48, 27; px = self.pattern(w, h); W, H = 101, 77
        x0, y0, pw, ph = 0, 10, 101, 56   # vkfsr_fit(48, 27, 101, 77)
        for hb in self.exe:
            base = self.draw(hb, px, w, h, (0, 0, w, h), W, H, 0, 'fsr', 0.25)
            for y in range(H):
                for x in range(W):
                    inside = x0 <= x < x0 + pw and y0 <= y < y0 + ph
                    if not inside:
                        self.assertEqual(base[4 * (y * W + x):4 * (y * W + x) + 4], bytes([0, 0, 0, 255]), (x, y))
            for rot in (1, 2, 3):
                o = self.draw(hb, px, w, h, (0, 0, w, h), W, H, rot, 'fsr', 0.25); tw = H if rot in (1, 3) else W
                for y in range(0, H, 3):
                    for x in range(0, W, 3):   # window pixel (x, y) where each turn puts it (gfx_vk_fsr.h)
                        tx, ty = {1: (H - 1 - y, x), 2: (W - 1 - x, H - 1 - y), 3: (y, W - 1 - x)}[rot]
                        self.assertEqual(o[4 * (ty * tw + tx):4 * (ty * tw + tx) + 4], base[4 * (y * W + x):4 * (y * W + x) + 4], (rot, x, y))

    def test_crop_reads_nothing_outside(self):
        w, h, cx, cy, cw, ch = 80, 50, 9, 7, 48, 27
        inner = self.pattern(cw, ch); px = bytearray(bytes([255, 0, 255, 255]) * (w * h))   # magenta around the crop
        for y in range(ch):
            px[4 * ((cy + y) * w + cx):4 * ((cy + y) * w + cx + cw)] = inner[4 * y * cw:4 * (y + 1) * cw]
        for hb in self.exe:
            for path, st in (('fsr', 0.25), ('fsr', -1), ('bilinear', -1)):
                a = self.draw(hb, bytes(px), w, h, (cx, cy, cw, ch), 101, 77, 0, path, st)
                b = self.draw(hb, inner, cw, ch, (0, 0, cw, ch), 101, 77, 0, path, st)
                self.assertEqual(a, b, (hb, path, st))   # the same picture as the crop alone

    def test_bottom_up_is_the_flip(self):
        w, h = 48, 27; px = self.pattern(w, h)
        flipped = b''.join(px[4 * w * y:4 * w * (y + 1)] for y in reversed(range(h)))
        for hb in self.exe:
            for path, st in (('fsr', 0.25), ('bilinear', -1)):
                a = self.draw(hb, px, w, h, (0, 0, w, h), 101, 57, 0, path, st, bu=1)
                b = self.draw(hb, flipped, w, h, (0, 0, w, h), 101, 57, 0, path, st, bu=0)
                # mirrored, pp becomes 1 - pp and the taps add up in another order: in 32 bits a step on under 1 % of the values; in 16 bits RCAS (up to 2.7x
                # in flat areas) makes the rounding 2 steps at most, on 6.4 % of this noisy pattern's values (the mapping itself is the 32-bit check)
                d = [abs(x - y) for x, y in zip(a, b)]
                self.assertLessEqual(max(d), 2 if hb else 1, (hb, path, st)); self.assertLess(sum(1 for v in d if v) / len(d), 0.10 if hb else 0.01, (hb, path, st))

    def test_ndsr_swap_shapes(self):
        import struct
        w, h = 24, 14; lin = [0.02, 0.2, 2.5]   # a flat linear colour, one channel past 1
        px = struct.pack(f'<{w * h * 4}e', *(lin + [1.0]) * (w * h))
        def run(hb, shape):
            src, dst = Path(self.tmp.name) / 'n.in', Path(self.tmp.name) / 'n.out'; src.write_bytes(px)
            r = subprocess.run([str(self.exe[hb]), str(src), str(w), str(h), '2', '1', '20', '12', '40', '24', '0', f'ndsr{shape}', '-1', '0', '10', str(dst)], capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stderr)
            v = struct.unpack(f'<{40 * 24 * 4}e', dst.read_bytes()); return {tuple(round(x, 2) for x in v[i:i + 4]) for i in range(0, len(v), 4)}
        enc = lambda x: 1.055 * x ** (1 / 2.4) - 0.055 if x > 0.0031308 else 12.92 * x
        for hb in self.exe:
            self.assertEqual(run(hb, 0), {tuple(round(min(enc(x), 1.0), 2) for x in lin) + (1.0,)})   # encoded, clamped
            self.assertEqual(run(hb, 1), {(0.02, 0.2, 1.0, 1.0)})                                      # linear again, clamped to 1
            self.assertEqual(run(hb, 2), {(0.02, 0.2, 2.5, 1.0)})                                      # linear again, its highlight kept

    def test_sharper_than_bilinear_on_an_edge(self):
        w, h = 32, 18; px = b''.join(bytes([230, 230, 230, 255]) if x * 18 > y * 32 + 100 else bytes([25, 25, 25, 255]) for y in range(h) for x in range(w))
        for hb in self.exe:
            lin = self.draw(hb, px, w, h, (0, 0, w, h), 64, 36, 0, 'bilinear', -1)
            easu = self.draw(hb, px, w, h, (0, 0, w, h), 64, 36, 0, 'fsr', -1)
            fsr = self.draw(hb, px, w, h, (0, 0, w, h), 64, 36, 0, 'fsr', 0.25)
            mid = lambda o: sum(1 for i in range(0, len(o), 4) if 60 < o[i] < 195)   # pixels caught halfway across the edge
            self.assertLess(mid(easu), mid(lin), hb); self.assertLess(mid(fsr), mid(easu), hb)
            self.assertTrue(all(25 <= easu[i] <= 230 for i in range(0, len(easu), 4)), hb)   # EASU: inside its 4 nearest texels' range (the dering)
            self.assertTrue(all(0 < fsr[i] < 255 for i in range(0, len(fsr), 4)), hb)        # RCAS: sharper, and never clipped at 0 or 1 (its limiter)


if __name__ == '__main__':
    unittest.main()
