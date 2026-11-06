"""Shader-side virtual texturing without a device (nxnative/dm/android/gfx_vk_vt.c, tools/spv_vt.py): the SPIR-V transform on synthetic fragment shaders made the way the
translator writes Mario Kart World's terrain materials (textureLod at max(clamp, textureQueryLod)), every output valid (spirv-val); its reflection through shader_vk.py's own
classify / refl_bin; gfx_vk_vt.c's logic built on the host (the tile geometry against nvn_tex.c's sparse layout, each slot's tile and border blocks against nvn_deswizzle.c's
whole-level decode, the pieces a neighbour's change rebuilds, the CLOCK over the slots, the page table's runs); and the template's address math (a Python port of TEMPLATE_GLSL's
VT_LEVEL, in float32) against the C side's page table words and slots. DM_VT_BUNDLE=<shaders.dmsv> also transforms and validates every fragment shader of a bundle (game files:
never in the repository)."""
import ast, base64, collections, glob, hashlib, json, os, random, re, shutil, struct, subprocess, sys, tempfile, unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DM = REPO / 'nxnative/dm'
TOOLS = DM / 'android/tools'
VT_C, VT_H = DM / 'android/gfx_vk_vt.c', DM / 'android/gfx_vk_vt.h'
SHADER_VK = TOOLS / 'shader_vk.py'
sys.path.insert(0, str(TOOLS))
import spv_vt   # noqa: E402

INC = ['-I' + str(DM / 'gfx/nvn'), '-I' + str(DM / 'gfx/nvn/include'), '-I' + str(DM / 'gfx/tex')]


def tool(name):
    """A SPIR-V tool or shader compiler: on PATH, else the NDK's shader-tools (ANDROID_NDK_HOME, the repository's tools/ NDK, the porter's)"""
    if shutil.which(name): return shutil.which(name)
    roots = [os.environ.get('ANDROID_NDK_HOME', ''), str(REPO / 'tools/android-ndk-r27c')] + sorted(glob.glob(os.path.expanduser('~/Library/Application Support/nx2android/toolchains/android-ndk-*')))
    for r in roots:
        for g in glob.glob(os.path.join(r, f'shader-tools/*/{name}')) if r else []: return g
    return None


def glsl(src, stage='frag'):
    """SPIR-V 1.3 (the translator's version) of a GLSL shader, or None without a compiler"""
    comp = tool('glslangValidator') or tool('glslc')
    if not comp: return None
    with tempfile.TemporaryDirectory() as d:
        p, o = os.path.join(d, f's.{stage}'), os.path.join(d, 's.spv'); Path(p).write_text(src)
        cmd = [comp, '-V', '--target-env', 'vulkan1.1', p, '-o', o] if 'glslang' in os.path.basename(comp) else [comp, '--target-env=vulkan1.1', p, '-o', o]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode: raise AssertionError(r.stdout + r.stderr)
        return Path(o).read_bytes()


def spirv_val(spv):
    v = tool('spirv-val')
    if not v: return None
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, 'm.spv'); Path(p).write_bytes(spv)
        r = subprocess.run([v, '--target-env', 'vulkan1.1', p], capture_output=True, text=True)
        return r.returncode, r.stdout + r.stderr


def ops(spv):
    """(opcode, words) of a module's instructions"""
    w = struct.unpack(f'<{len(spv) // 4}I', spv); i, out = 5, []
    while i < len(w): n = w[i] >> 16; out.append((w[i] & 0xffff, w[i:i + n])); i += n
    return out


# The game's form (InstEmitAmpereMem.cs ATex / ATmml: TMML's level in 8.8 fixed point, then TEX.LL at the larger of it and the clamps from its uniforms)
TERRAIN = r'''#version 450
layout(set = 0, binding = 0) uniform sampler2DArray tA;   // a candidate
layout(set = 0, binding = 1) uniform sampler2DArray tB;   // a candidate, queried twice
layout(set = 0, binding = 2) uniform sampler2DArray tC;   // sampled with an implicit LOD too: not a candidate
layout(set = 0, binding = 3) uniform sampler2D tD;        // 2D: not a candidate
layout(set = 0, binding = 4, std140) uniform U { vec4 c[4]; } u;
layout(location = 0) in vec4 uv;
layout(location = 0) out vec4 o;
#define q(t, p) (float(uint(int(textureQueryLod(t, p).x * 256.0))) * 0.00390625)   // (inline, as the translator writes it: no sampler is passed to a function)
void main() {
    float la = max(u.c[0].x, q(tA, uv.xy));
    vec4 a = textureLod(tA, vec3(uv.xy, roundEven(u.c[0].y)), la);
    vec4 b = textureLod(tB, vec3(uv.zw, 2.0), max(u.c[1].x, q(tB, uv.zw))) + textureLod(tB, vec3(uv.wz, 1.0), max(u.c[1].y, q(tB, uv.wz)));
    vec4 c = textureLod(tC, vec3(uv.xy, 0.0), 1.0) + texture(tC, vec3(uv.zw, 0.0));
    o = a + b + c + texture(tD, uv.xy)%s;
}
'''


class Template(unittest.TestCase):
    def test_embedded_template_is_the_source(self):
        self.assertEqual(spv_vt.TEMPLATE_SHA, hashlib.sha256(spv_vt.TEMPLATE_GLSL.encode()).hexdigest(), 'TEMPLATE_GLSL changed: run spv_vt.py --regen')
        for fb in (0, 1):
            m = spv_vt.template(fb); names = set(m.names().values())
            self.assertTrue({'vt_ctl', 'vt_pt'} <= names and ('vt_fb' in names) == bool(fb))
            r = spirv_val(base64.b64decode(spv_vt.TEMPLATE_SPV[fb]))
            if r is not None: self.assertEqual(r[0], 0, r[1])

    def test_template_layout_matches_the_backend(self):   # vt_ctl's size and the page table word's bits as gfx_vk_vt.h has them
        h = VT_H.read_text(encoding='utf-8')
        self.assertIn(f'enum {{ VT_NC = {spv_vt.TEMPLATE_NC} }};', h)
        self.assertIn(f'uvec4 e[{1 + 3 * spv_vt.TEMPLATE_NC}]', spv_vt.TEMPLATE_GLSL)
        for bit, name in ((0xffff, 'VT_PT_SLOT'), (0x10000, 'VT_PT_MAPPED'), (0x20000, 'VT_PT_PROB')):
            self.assertRegex(h, rf'#define {name} {bit:#x}u')
            self.assertIn(f'{bit:#x}u', spv_vt.TEMPLATE_GLSL)
        kinds = dict(re.findall(r'RK_VT_(\w+) = (\d+)', h))
        sv = SHADER_VK.read_text(encoding='utf-8')
        if "'vt_pool'" in sv:   # (shader_vk.py's hook: patches/VT/01-shader_vk-vt-hook.patch)
            for k, v in kinds.items(): self.assertIn(f"'vt_{k.lower()}': {v}", sv)


class Transform(unittest.TestCase):
    def setUp(self):
        if not (tool('glslangValidator') or tool('glslc')): self.skipTest('glslangValidator or glslc is needed')

    def check(self, spv):
        r = spirv_val(spv)
        if r is None: self.skipTest('spirv-val is needed')
        self.assertEqual(r[0], 0, r[1])

    def test_terrain_candidates(self):
        spv = glsl(TERRAIN % '')
        out, info = spv_vt.transform(spv)
        self.assertIsNotNone(info); self.check(out)
        self.assertEqual([b for b, _ in info['cands']], [0, 1])   # tC (sampled with an implicit LOD) and tD (2D) are not candidates
        self.assertEqual((info['queries'], info['samples']), (3, 3))
        self.assertTrue(info['early'] and info['fb'] is not None)
        self.assertEqual(info['pools'], [5, 6]); self.assertEqual((info['ctl'], info['pt'], info['fb']), (7, 8, 9))
        o = ops(out)
        main_end = next(i for i, (op, w) in enumerate(o) if op == 56)   # the shader's own function (the template's follow it)
        self.assertEqual(sum(1 for op, _ in o[:main_end] if op == 57), 6)   # three queries and three samples became calls
        self.assertTrue(any(op == 16 and w[2] == 9 for op, w in o))   # EarlyFragmentTests
        names = {w[1]: spv_vt.str_of(list(w[2:])) for op, w in o if op == 5}
        self.assertTrue({'vt_pool_0', 'vt_pool_1', 'vt_ctl', 'vt_pt', 'vt_fb'} <= set(names.values()))
        # what was sampled through a candidate is now a call, the rest untouched: tC keeps its two samples, tD its own
        self.assertEqual(sum(1 for op, _ in o[:main_end] if op in (87, 88)), sum(1 for op, _ in ops(spv) if op in (87, 88)) - 3)
        again, info2 = spv_vt.transform(out)   # the transform's output has no candidate left (its calls are not loads sampled)
        self.assertIsNone(info2)

    def test_discard_takes_the_variant_without_requests(self):
        out, info = spv_vt.transform(glsl(TERRAIN % '; if (uv.x < 0.0) discard'))
        self.check(out)
        self.assertIsNone(info['fb']); self.assertFalse(info['early'])
        self.assertFalse(any(op == 16 and w[2] == 9 for op, w in ops(out)))
        self.assertFalse(any(op == 234 for op, w in ops(out)))   # no OpAtomicOr

    def test_own_stores_take_the_variant_without_requests(self):
        src = TERRAIN.replace('layout(location = 0) out vec4 o;', 'layout(location = 0) out vec4 o;\nlayout(set = 0, binding = 5, std430) buffer B { uint n; } sb;') % '; atomicAdd(sb.n, 1u)'
        out, info = spv_vt.transform(glsl(src))
        self.check(out)
        self.assertIsNone(info['fb']); self.assertFalse(info['early'])

    def test_unchanged(self):
        vs = glsl('#version 450\nlayout(location = 0) out vec4 o;\nvoid main() { gl_Position = vec4(0.0); o = vec4(1.0); }\n', 'vert')
        self.assertIsNone(spv_vt.transform(vs)[1])   # a vertex shader
        fs = glsl('#version 450\nlayout(binding = 0) uniform sampler2DArray t;\nlayout(location = 0) in vec3 p;\nlayout(location = 0) out vec4 o;\nvoid main() { o = texture(t, p); }\n')
        self.assertIsNone(spv_vt.transform(fs)[1])   # an implicit-LOD sample only

    def test_reflection_through_shader_vk(self):
        tree = ast.parse(SHADER_VK.read_text(encoding='utf-8'))
        if "'vt_pool'" not in SHADER_VK.read_text(encoding='utf-8'): self.skipTest("shader_vk.py's VT hook is not applied yet (patches/VT/01-shader_vk-vt-hook.patch)")
        def names(t): return [t.id] if isinstance(t, ast.Name) else [n for e in t.elts for n in names(e)] if isinstance(t, ast.Tuple) else ['?']
        body = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in ('classify', 'refl_bin', 'spirv_bindings') or isinstance(n, ast.Assign)
                and all(x.isupper() and x not in {'HERE', 'TOOLS', 'GAME', 'FACTS', 'MAP', 'XLATE_DLL'} for t in n.targets for x in names(t))]
        SV = {'struct': struct, 're': re, 'os': os, 'hashlib': hashlib, 'collections': collections, 'FACTS': {}}
        exec(compile(ast.Module(body=body, type_ignores=[]), str(SHADER_VK), 'exec'), SV)
        out, info = spv_vt.transform(glsl(TERRAIN % ''))
        binds, _, io = SV['spirv_bindings'](out)
        cl = [SV['classify']('ps', b, {}) for b in binds]
        for e in cl:
            if e['kind'] == 'vt_pool': e['reg'] = dict((k, b) for b, k in info['cands']).get(e['off'])
        r = SV['refl_bin'](cl, b'', io[3]); n = struct.unpack_from('<I', r, 0)[0]
        got = {}
        for i in range(n):
            kind, desc, dim, fl, bnd, cnt, reg, samp, fmt, off, sarg = struct.unpack_from('<BBBBHHhBBii', r, 20 + 20 * i); got[bnd] = (kind, desc, reg, off)
        self.assertEqual(got[5], (16, 2, 0, 0)); self.assertEqual(got[6], (16, 2, 1, 1))   # vt_pool_k: kind 16, combined sampler, its candidate's binding, k
        self.assertEqual((got[7][:2], got[8][:2], got[9][:2]), ((17, 0), (18, 1), (19, 1)))   # vt_ctl a uniform buffer, vt_pt and vt_fb storage buffers

    def test_bundle(self):
        path = os.environ.get('DM_VT_BUNDLE')
        if not path: self.skipTest('DM_VT_BUNDLE=<shaders.dmsv> transforms and validates a bundle (game files are never in the repository)')
        data = Path(path).read_bytes(); n = struct.unpack_from('<I', data, 8)[0]; ents = [struct.unpack_from('<QIIIIII', data, 12 + 32 * i) for i in range(n)]
        bad, done = [], 0
        for key, st, so, sl, *_ in ents:
            if st != 1: continue
            out, info = spv_vt.transform(data[so:so + sl])
            if info is None: continue
            done += 1; r = spirv_val(out)
            if r and r[0]: bad.append(f'{key:016x}: {r[1][:200]}')
        self.assertFalse(bad, '\n'.join(bad[:5])); self.assertGreater(done, 0)


# ---- gfx_vk_vt.c's logic on the host ----
def region(src, start, end):
    a = src.index(start); return src[a:src.index(end, a) + len(end)]


C_MAIN = r'''
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
static uint8_t *g_store; static const uint8_t *g_map; static vt_geom *g_g; static uint64_t g_reads;
static void rd(void *c, uint64_t off, void *dst, uint32_t n) { (void)c; memcpy(dst, g_store + off, n); g_reads += n; }
static int mp(void *c, uint32_t place) { (void)c; return g_map[place]; }
static uint32_t prob_seen[64], evict_seen[64], nprob, nevict;
static void on_prob(void *c, uint32_t p) { (void)c; prob_seen[nprob++ % 64] = p; }
static void on_evict(void *c, uint32_t p) { (void)c; evict_seen[nevict++ % 64] = p; }
// One sparse texture: its geometry against nvn_tex.c's, every tile's slot against a whole-level decode (nvn_deswizzle) at the wrapped neighbours' blocks, zeros where not mapped
static void check_tex(int format, uint32_t w, uint32_t layers, uint32_t levels, int nexp, uint32_t texp, const uint8_t *zero, FILE *js) {
    nvn_texdesc d = {4, format, 0x20, 1, w, w, layers, levels, 0, 0}; nvn_texlayout L; nvn_tex_layout(&d, &L);
    uint32_t tw, th, td; int n = nvn_tex_sparse_tiles(&d, &tw, &th, &td); assert(n == nexp && tw == texp && th == texp && td == 1);
    int bw, bh, bpb = nvn_fmt_bpb(format, &bw, &bh); uint64_t loff[16]; for (int l = 0; l < n; l++) loff[l] = nvn_tex_level_offset(&d, (unsigned)l);
    vt_geom g; const char *why = vt_geom_init(&g, w, w, layers, (uint32_t)n, tw, th, (uint32_t)bw, (uint32_t)bh, (uint32_t)bpb, loff, L.layer_size); assert(!why);
    assert(g.nbx == 64 && g.nby == 64 && g.per_layer > 0);
    for (int l = 0; l < n; l++) { assert(g.nx[l] == (w >> l) / tw && g.loff[l] == loff[l]); unsigned th_, td_; nvn_tex_level_tiles(&d, (unsigned)l, &th_, &td_); assert(8u << th_ == g.nby); }
    if (n < (int)levels) assert(nvn_tex_level_offset(&d, (unsigned)n) == loff[n - 1] + (uint64_t)g.nx[n - 1] * g.ny[n - 1] * VT_TILE);   // the tail right after the last tiled level
    g_store = malloc(L.size); for (uint64_t i = 0; i < L.size; i++) g_store[i] = (uint8_t)(rand() >> 7);
    uint32_t np = layers * g.per_layer; uint8_t *map = calloc(np, 1); for (uint32_t i = 0; i < np; i++) map[i] = (rand() % 4) != 0; g_map = map; g_g = &g;
    size_t pitch = (size_t)(g.nbx + 2) * g.bpb, ssz = pitch * (g.nby + 2); uint8_t *slot = malloc(ssz), *part = malloc(ssz), *tile = malloc(VT_TILE);
    fprintf(js, "{\"format\": %d, \"W\": %u, \"layers\": %u, \"per_layer\": %u, \"tws\": %u, \"places\": [", format, w, layers, g.per_layer, g.tws); int first = 1;
    for (uint32_t y = 0; y < layers; y++) for (int l = 0; l < n; l++) {
        uint32_t lw = (w >> l) / (uint32_t)bw, lb = lw * (uint32_t)bpb; uint8_t *ref = malloc((size_t)lb * lw);
        assert(!nvn_deswizzle(&d, (unsigned)l, y, g_store, ref, lb));
        for (uint32_t ty = 0; ty < g.ny[l]; ty++) for (uint32_t tx = 0; tx < g.nx[l]; tx++) {
            uint32_t p = vt_place(&g, y, (uint32_t)l, tx, ty), py, pl, ptx, pty; vt_place_of(&g, p, &py, &pl, &ptx, &pty); assert(py == y && pl == (uint32_t)l && ptx == tx && pty == ty);
            assert(vt_tile_off(&g, y, (uint32_t)l, tx, ty) == (uint64_t)y * L.layer_size + loff[l] + (uint64_t)(ty * g.nx[l] + tx) * VT_TILE);
            if (rand() % 3 == 0 || (l == n - 1)) { fprintf(js, "%s[%u, %d, %u, %u, %u]", first ? "" : ", ", y, l, tx, ty, p); first = 0; }
            if (!map[p] && rand() % 4) continue;   // (most unmapped tiles are never uploaded: a few are, for their zero interior)
            memset(slot, 0xa5, ssz); assert(!vt_slot_build(&g, &d, y, (uint32_t)l, tx, ty, 0x1ff, zero, rd, mp, NULL, tile, slot));
            for (uint32_t sy = 0; sy < g.nby + 2; sy++) for (uint32_t sx = 0; sx < g.nbx + 2; sx++) {
                uint32_t gx = (tx * g.nbx + sx + lw - 1) % lw, gy = (ty * g.nby + sy + lw - 1) % lw;   // the wrapped level block it holds
                uint32_t q = vt_place(&g, y, (uint32_t)l, gx / g.nbx, gy / g.nby);
                const uint8_t *want = map[q] ? ref + (size_t)gy * lb + (size_t)gx * (uint32_t)bpb : zero;
                if (memcmp(slot + sy * pitch + sx * (uint32_t)bpb, want, (size_t)bpb)) { fprintf(stderr, "slot of tile (%u, %d, %u, %u): block (%u, %u) differs\n", y, l, tx, ty, sx, sy); exit(1); }
            }
            for (uint32_t b = 2; b <= 256; b <<= 1) {   // each piece on its own builds what the whole slot holds in its rectangle, and nothing else
                uint32_t r[4]; vt_piece_rect(b, g.nbx, g.nby, r); memset(part, 0x5a, ssz); assert(!vt_slot_build(&g, &d, y, (uint32_t)l, tx, ty, b, zero, rd, mp, NULL, tile, part));
                for (uint32_t sy = 0; sy < g.nby + 2; sy++) for (uint32_t sx = 0; sx < g.nbx + 2; sx++) {
                    int in = sx >= r[0] && sx < r[2] && sy >= r[1] && sy < r[3]; const uint8_t *a = part + sy * pitch + sx * (uint32_t)bpb;
                    if (in) assert(!memcmp(a, slot + sy * pitch + sx * (uint32_t)bpb, (size_t)bpb)); else for (int k = 0; k < bpb; k++) assert(a[k] == 0x5a);
                }
            }
        }
        free(ref);
    }
    uint32_t e[12]; vt_ctl_fill(e, VT_MODE_VT, 3, 1, 1, 0, 1000, &g, 0.0f, 11.0f);
    assert((e[0] & 3) == VT_MODE_VT && (e[0] & VT_CTL_LINEAR) && (e[0] >> 3 & 3) == 1 && (e[0] >> 5 & 3) == 0 && e[1] == 3 && e[2] == 1000 && e[3] == layers);
    assert(e[4] == (w | w << 16) && (e[5] & 31) == g.tws && (e[5] >> 8 & 31) == g.ths && (e[5] >> 16) == g.bw && e[6] == (uint32_t)n && e[7] == g.per_layer);
    fprintf(js, "], \"ctl\": [%u, %u, %u, %u, %u, %u, %u, %u, %u, %u, %u, %u]", e[0], e[1], e[2], e[3], e[4], e[5], e[6], e[7], e[8], e[9], e[10], e[11]);
    // a view from level 1 and the last layer, one layer, whose image starts two levels down (shift 2: the dense image from the texture's level 3): the page table read from the
    // view's base level and layer, its level numbers the view's; and the shapes VT leaves to the dense image (a view past the tiled levels, a shift past them, a layer past the texture)
    uint32_t ve[12]; vt_ctl_view(ve, VT_MODE_VT, &g, 1000, 1, layers - 1, 1, 2, 1, 1, 1, 0.0f, 9.0f);
    assert((ve[0] & 3) == VT_MODE_VT && ve[1] == 2 && ve[2] == 1000 + (layers - 1) * g.per_layer + g.loff_t[1] && ve[3] == 1 && ve[4] == ((w >> 1) | (w >> 1) << 16) && ve[6] == (uint32_t)n - 1 && ve[7] == g.per_layer);
    fprintf(js, ", \"view\": {\"base\": 1, \"ly0\": %u, \"ctl\": [%u, %u, %u, %u, %u, %u, %u, %u, %u, %u, %u, %u], \"places\": [", layers - 1, ve[0], ve[1], ve[2], ve[3], ve[4], ve[5], ve[6], ve[7], ve[8], ve[9], ve[10], ve[11]);
    for (int vl = 0, f1 = 1; vl + 1 < n; vl++) for (uint32_t ty = 0; ty < g.ny[vl + 1]; ty++) for (uint32_t tx = 0; tx < g.nx[vl + 1]; tx++) {
        fprintf(js, "%s[0, %d, %u, %u, %u]", f1 ? "" : ", ", vl, tx, ty, vt_place(&g, layers - 1, (uint32_t)vl + 1, tx, ty)); f1 = 0; }
    fprintf(js, "]}}");
    vt_ctl_view(ve, VT_MODE_VT, &g, 1000, (uint32_t)n, 0, 1, 1, 1, 0, 0, 0.0f, 9.0f); assert((ve[0] & 3) == VT_MODE_DENSE && ve[1] == 1 && ve[2] == 0);   // the view starts past the tiled levels
    vt_ctl_view(ve, VT_MODE_VT, &g, 1000, 1, 0, 1, (uint32_t)n, 1, 0, 0, 0.0f, 9.0f); assert((ve[0] & 3) == VT_MODE_DENSE);   // levels below the image past the tiled ones
    vt_ctl_view(ve, VT_MODE_VT, &g, 1000, 0, layers, 1, 2, 1, 0, 0, 0.0f, 9.0f); assert((ve[0] & 3) == VT_MODE_DENSE);   // a first layer past the texture's
    free(g_store); free(map); free(slot); free(part); free(tile);
}
int main(int argc, char **argv) {
    srand(7); nvn_gob_switch2 = atoi(argv[1]);
    static const uint8_t z16[16] = {0}, ve[16] = {0xfc, 0xfd, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff};
    FILE *js = fopen(argv[2], "w"); fprintf(js, "[");
    check_tex(0x50, 1024, 2, 11, 3, 256, z16, js); fprintf(js, ", ");   // BC6H: tiles of 256x256 texels for levels 0-2 (4x4, 2x2, 1x1 tiles), the tail from level 3
    check_tex(0x8e, 2048, 2, 12, 3, 512, ve, js); fprintf(js, "]");      // ASTC 8x8: tiles of 512x512 texels for levels 0-2, ASTC's zero a void-extent block
    fclose(js);
    // MKW's arrays: their shape as nvn_tex.c lays them out (8192x8192x68, 12 levels): 6 and 5 tiled levels, 1,365 and 341 tiles a layer
    { nvn_texdesc d = {4, 0x50, 0x20, 1, 8192, 8192, 68, 12, 0, 0}; uint32_t tw, th, td; assert(nvn_tex_sparse_tiles(&d, &tw, &th, &td) == 6 && tw == 256);
      uint64_t lo[16]; for (int l = 0; l < 6; l++) lo[l] = nvn_tex_level_offset(&d, (unsigned)l); nvn_texlayout L; nvn_tex_layout(&d, &L);
      vt_geom g; assert(!vt_geom_init(&g, 8192, 8192, 68, 6, 256, 256, 4, 4, 16, lo, L.layer_size) && g.per_layer == 1365); }
    { nvn_texdesc d = {4, 0x8e, 0x20, 1, 8192, 8192, 68, 12, 0, 0}; uint32_t tw, th, td; assert(nvn_tex_sparse_tiles(&d, &tw, &th, &td) == 5 && tw == 512);
      uint64_t lo[16]; for (int l = 0; l < 5; l++) lo[l] = nvn_tex_level_offset(&d, (unsigned)l); nvn_texlayout L; nvn_tex_layout(&d, &L);
      vt_geom g; assert(!vt_geom_init(&g, 8192, 8192, 68, 5, 512, 512, 8, 8, 16, lo, L.layer_size) && g.per_layer == 341); }
    // shapes VT does not take
    { vt_geom g; uint64_t lo[2] = {0, 0}; assert(vt_geom_init(&g, 1000, 1000, 1, 1, 256, 256, 4, 4, 16, lo, 0) != NULL); assert(vt_geom_init(&g, 1024, 1024, 1, 1, 256, 256, 8, 5, 16, lo, 0) != NULL); }
    // the faces a neighbour keeps of this tile: opposite sides, opposite corners
    assert(vt_facing(-1, 0) == 4 && vt_facing(1, 0) == 2 && vt_facing(0, -1) == 16 && vt_facing(0, 1) == 8 && vt_facing(-1, -1) == 256 && vt_facing(1, 1) == 32 && vt_facing(1, -1) == 128 && vt_facing(-1, 1) == 64);
    // the CLOCK: 8 slots; taken, then put on probation by the hand; a touch takes one off; taken back only on probation VT_PROB_SUBS submissions and while few are free; reused after the fence
    vt_slots s; assert(!vt_slots_init(&s, 8));
    for (uint32_t i = 0; i < 8; i++) assert(vt_slot_take(&s, 0, 100 + i, 1) >= 0);
    assert(vt_slot_take(&s, 0, 200, 1) < 0);
    assert(vt_clock(&s, 8, 4, 2, 10, on_prob, on_evict, NULL) == 0 && nprob == 8);   // all on probation now
    for (uint32_t i = 0; i < 8; i++) if (s.s[i].place == 103) s.s[i].prob = 0;          // a shader touched slot of 103
    assert(vt_clock(&s, 8, 4, 2 + VT_PROB_SUBS - 1, 10, on_prob, on_evict, NULL) == 0);   // too early: nothing taken back (103 back on probation)
    uint32_t gone = vt_clock(&s, 8, 4, 2 + VT_PROB_SUBS, 11, on_prob, on_evict, NULL);
    assert(gone == 4 && nevict == 4);                                                 // until 4 are free or on their way back
    for (uint32_t k = 0; k < 4; k++) assert(evict_seen[k] != 103);
    assert(vt_slot_take(&s, 10, 300, 20) < 0);                                        // their fence (11) not passed: none yet
    int t = vt_slot_take(&s, 11, 300, 20); assert(t >= 0 && s.s[t].place == 300 && s.nlate == 0 && s.nfree == 3);   // (all four back at once)
    // the page table's dirty runs: cut at max, gone from the bitmap, resumed
    uint64_t bm[4] = {0}; for (int i = 3; i < 9; i++) bm[0] |= 1ull << i; bm[1] |= 1ull << 63; bm[2] |= 3; bm[3] |= 1ull << 40;
    uint32_t runs[8][2], next; uint32_t nr = vt_runs(bm, 256, 0, runs, 2, &next);
    assert(nr == 2 && runs[0][0] == 3 && runs[0][1] == 6 && runs[1][0] == 127 && runs[1][1] == 3 && next == 130);
    nr = vt_runs(bm, 256, next, runs, 8, &next); assert(nr == 1 && runs[0][0] == 232 && runs[0][1] == 1 && !bm[0] && !bm[1] && !bm[2] && !bm[3]);
    puts("vt logic ok");
    return 0;
}
'''


class Logic(unittest.TestCase):
    def run_c(self, gob):
        cc = shutil.which('cc') or shutil.which('clang')
        if not cc: self.skipTest('a C compiler is needed')
        src = VT_C.read_text(encoding='utf-8'); h = VT_H.read_text(encoding='utf-8')
        body = region(src, '// ---- VT logic', '// ---- end of the VT logic ----')
        consts = '\n'.join(l for l in h.splitlines() if l.startswith('#define VT_') or l.startswith('enum { VT_MODE'))
        code = '#include <stdint.h>\n#include <stdlib.h>\n#include <string.h>\n#include "nvn_tex.h"\n#include "nvn_deswizzle.h"\n' + consts + '\n' + body + C_MAIN
        with tempfile.TemporaryDirectory(prefix='nx2a_vk_vt_') as d:
            c, exe, js = Path(d) / 'vt.c', Path(d) / 'vt', Path(d) / 'places.json'
            c.write_text(code, encoding='utf-8')
            r = subprocess.run([cc, '-std=gnu11', '-O1', '-Wall', '-Werror', '-Wno-unused-function', *INC, str(c), str(DM / 'gfx/nvn/nvn_tex.c'), str(DM / 'gfx/tex/nvn_deswizzle.c'), '-o', str(exe)],
                               capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
            r = subprocess.run([str(exe), str(gob), str(js)], capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
            return json.loads(js.read_text())

    def test_slots_switch2_gobs(self): self.shader_math(self.run_c(1))
    def test_slots_switch_gobs(self): self.run_c(0)

    def shader_math(self, texs):
        """TEMPLATE_GLSL's VT_LEVEL in float32 on the C side's control block: a texel of a tile at its level finds that tile's page table word, and its slot coordinate is the
        texel's place in the tile plus the border; the bilinear footprint stays in the slot"""
        f32 = lambda x: struct.unpack('<f', struct.pack('<f', x))[0]
        def wrap(u, m):
            if m == 1: return f32(u - __import__('math').floor(u))
            if m == 2: t = f32(u % 2.0); return f32(2.0 - t) if t > 1.0 else t
            if m == 3: return min(abs(u), 1.0)
            return min(max(u, 0.0), 1.0)
        def level(e, l, uv, layer, mu, mv):
            W, H, tws, ths, bd = e[4] & 0xffff, e[4] >> 16, e[5] & 31, e[5] >> 8 & 31, e[5] >> 16 & 255
            wl, hl = max(W >> l, 1), max(H >> l, 1); tx_, ty_ = f32(uv[0] * wl), f32(uv[1] * hl)
            if mu != 1: tx_ = min(max(tx_, 0.5), wl - 0.5)
            if mv != 1: ty_ = min(max(ty_, 0.5), hl - 0.5)
            nx, ny = max(wl >> tws, 1), max(hl >> ths, 1); tx, ty = min(int(tx_) >> tws, nx - 1), min(int(ty_) >> ths, ny - 1)
            lo = sum(max((W >> j) >> tws, 1) * max((H >> j) >> ths, 1) for j in range(l))
            ix = e[2] + layer * e[7] + lo + ty * nx + tx
            q = (f32(tx_ - (tx << tws) + bd), f32(ty_ - (ty << ths) + bd)); ss = ((1 << tws) + 2 * bd, (1 << ths) + 2 * bd)
            return ix, q, ss, (tx_, ty_)
        rnd = random.Random(3)
        for t in texs:
            e = t['ctl']; self.assertEqual(e[1], 3)
            for y, l, tx, ty, p in t['places']:
                ts = 1 << t['tws']
                for _ in range(24):
                    edge = rnd.random() < 0.5   # half of them within a texel of the tile's edges
                    fx = rnd.choice([rnd.uniform(0, 1), rnd.uniform(ts - 1, ts)]) if edge else rnd.uniform(0, ts)
                    fy = rnd.choice([rnd.uniform(0, 1), rnd.uniform(ts - 1, ts)]) if edge else rnd.uniform(0, ts)
                    wl = t['W'] >> l; uv = ((tx * ts + fx) / wl, (ty * ts + fy) / wl)
                    for mu in (0, 1):
                        u2 = (wrap(uv[0], mu), wrap(uv[1], mu)); ix, q, ss, tt = level(e, l, u2, y, mu, mu)
                        if mu == 1 or (0.5 <= tt[0] <= wl - 0.5 and 0.5 <= tt[1] <= wl - 0.5): self.assertEqual(ix, e[2] + p, (y, l, tx, ty, uv))   # (clamping may move an edge texel's tile)
                        bd = e[5] >> 16 & 255
                        for a in (0, 1):   # the sample within half a texel of the tile, so the bilinear footprint (one texel) stays in its border; the slot texel is the level's
                            self.assertTrue(bd - 0.5 - 1e-3 <= q[a] <= ss[a] - bd + 0.5 + 1e-3, (q, ss, uv, mu))
                            self.assertAlmostEqual(q[a] - bd + ((int(tt[a]) >> t['tws']) << t['tws']), tt[a], places=2)
            # the view from level base and layer ly0 (shift 2): view level l, view layer 0 reads the texture's level base + l, layer ly0
            vw = t['view']; ve = vw['ctl']; self.assertEqual((ve[0] & 3, ve[1]), (2, 2))
            for vy, vl, tx, ty, p in vw['places']:
                ts = 1 << t['tws']; wl = t['W'] >> (vw['base'] + vl)
                for _ in range(8):
                    uv = ((tx * ts + rnd.uniform(0.5, ts - 0.5)) / wl, (ty * ts + rnd.uniform(0.5, ts - 0.5)) / wl)
                    ix, q, ss, tt = level(ve, vl, uv, vy, 1, 1)
                    self.assertEqual(ix, 1000 + p, (vl, tx, ty, uv))
            # wrapped coordinates (repeat) of the same texel a whole period away find the same word
            ix1 = level(e, 0, (wrap(1.25, 1), wrap(-0.75, 1)), 0, 1, 1)[0]; ix2 = level(e, 0, (0.25, 0.25), 0, 1, 1)[0]; self.assertEqual(ix1, ix2)


if __name__ == '__main__':
    unittest.main()
