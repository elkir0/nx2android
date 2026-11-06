"""NVN's alpha test on Vulkan (gfx_vk_shader.c spv_alpha_test): the fragment shader variant whose new entry point calls the
shader's own main, then compares colour 0's alpha with the draw's function and reference (push constants) and kills the fragment
when the comparison fails. The transform is built from gfx_vk_shader.c's own functions into a host tool and run on modules glslc
makes (and on a sample of Mario Kart World's bundle when it is at hand); spirv-val checks every variant, and a small interpreter
runs the new entry point for each NVN function over chosen alphas and references against upstream's dm_alpha_pass."""
from pathlib import Path
import glob
import math
import os
import shutil
import struct
import subprocess
import tempfile
import unittest

DM = Path(__file__).resolve().parents[1] / 'nxnative/dm'
SHADER = DM / 'android/gfx_vk_shader.c'
HDR = DM / 'android/gfx_vk.h'
BUNDLE = DM.parents[1] / '.gamedump/MarioKartWorld/xlate-check/final'   # (the translator's check run: present on the machine that ported the game)


def ndk_tool(name):
    """A SPIR-V tool on PATH, else from ANDROID_NDK_HOME, the repository's tools/ NDK or the NDK the porter installs (None when none)."""
    if shutil.which(name):
        return shutil.which(name)
    roots = [os.environ.get('ANDROID_NDK_HOME', ''), str(DM.parents[1] / 'tools/android-ndk-r27c')] + \
            sorted(glob.glob(os.path.expanduser('~/Library/Application Support/nx2android/toolchains/android-ndk-*')))
    for r in roots:
        for g in glob.glob(os.path.join(r, f'shader-tools/*/{name}')) if r else []:
            return g
    return None


def tool_source():
    """spvx <steps> <in.spv> <out.spv>: gfx_vk_shader.c's transforms applied in order (f fragment depth, p per-primitive inputs, a alpha test, d dual source)."""
    src, hdr = SHADER.read_text(encoding='utf-8'), HDR.read_text(encoding='utf-8')

    def region(start, end):
        a = src.index(start)
        return src[a:src.index(end, a)]
    consts = '\n'.join(l for l in hdr.splitlines() if l.startswith('enum { VK_COL0_NONE') or l.startswith('#define VK_ALPHA_PUSH_'))
    body = region('static int spv_is_vec4(', 'static nvn_hmap g_shaders') + \
        region('typedef struct { const uint32_t *w; uint32_t fn, bound, *at, *nt, nnt;', '// DM_VK_SPV_DIR=<dir>') + \
        region("// A fragment shader's dual-source variant", 'static VkShaderModule vk_shader_module_(vkshader_t *s')
    return '#include <stdint.h>\n#include <stdio.h>\n#include <stdlib.h>\n#include <string.h>\n' + consts + '\n' + body + r'''
int main(int argc, char **argv) {
    if (argc != 4) return 3;
    FILE *f = fopen(argv[2], "rb"); if (!f) return 3; fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
    uint32_t nw = (uint32_t)(n / 4), *w = malloc((size_t)n); if (fread(w, 1, (size_t)n, f) != (size_t)n) return 3; fclose(f);
    for (const char *s = argv[1]; *s; s++) {
        uint32_t n2 = 0, *w2 = NULL, in = 0, out = 0; const char *why = NULL;
        if (*s == 'f') w2 = spv_frag_depth(w, nw, &n2);
        else if (*s == 'p') { spv_io_locs(w, nw, &in, &out); w2 = in ? spv_per_primitive(w, nw, in, &n2) : NULL; if (!in) why = "no vec4 inputs"; }
        else if (*s == 'a') w2 = spv_alpha_test(w, nw, &n2, &why);
        else if (*s == 'd') w2 = spv_dual(w, nw, &n2);
        else if (*s == 'm') w2 = spv_payload_private(w, nw, &n2);   // (a mesh shader's A/B variants: gfx_vk_exec.c DM_VK_MESH_NOTASK, DM_VK_MESH_POSBLOCK)
        else if (*s == 'b') w2 = spv_pos_block(w, nw, &n2);
        else if (*s == 'c') { printf("%d\n", spv_col0(w, nw, NULL, NULL, NULL)); continue; }
        if (!w2) { printf("%c: %s\n", *s, why ? why : "no variant"); return 2; }
        free(w); w = w2; nw = n2;
    }
    f = fopen(argv[3], "wb"); if (!f || fwrite(w, 4, nw, f) != nw) return 3; fclose(f);
    return 0;
}
'''


def words(b):
    return list(struct.unpack(f'<{len(b) // 4}I', b))


def insts(w):
    i = 5
    while i < len(w):
        n = w[i] >> 16
        if not n:
            raise AssertionError('a zero-length instruction')
        yield w[i] & 0xffff, w[i + 1:i + n]
        i += n


def f32(x):   # x rounded to a 32-bit float
    return struct.unpack('<f', struct.pack('<f', x))[0]


def ordered(x, y):
    return not (math.isnan(x) or math.isnan(y))


CMP = {180: lambda x, y: ordered(x, y) and x == y, 181: lambda x, y: not ordered(x, y) or x == y,
       182: lambda x, y: ordered(x, y) and x != y, 183: lambda x, y: not ordered(x, y) or x != y,
       184: lambda x, y: ordered(x, y) and x < y, 185: lambda x, y: not ordered(x, y) or x < y,
       186: lambda x, y: ordered(x, y) and x > y, 187: lambda x, y: not ordered(x, y) or x > y,
       188: lambda x, y: ordered(x, y) and x <= y, 189: lambda x, y: not ordered(x, y) or x <= y,
       190: lambda x, y: ordered(x, y) and x >= y, 191: lambda x, y: not ordered(x, y) or x >= y}


def expected(func, a, r):   # upstream's dmsh_alpha.py dm_alpha_pass (C++ comparisons: a NaN fails all but !=)
    return {1: False, 2: a < r, 3: a == r, 4: a <= r, 5: a > r, 6: a != r, 7: a >= r}.get(func, True)


def run_entry(w, func, ref, alpha):
    """The variant's entry point run with colour 0 = (0, 0, 0, alpha) and push constants {func, ref}: True when the fragment is kept."""
    ins = list(insts(w))
    entry = next(a[1] for op, a in ins if op == 15)
    floats = {a[0] for op, a in ins if op == 22}
    env = {}
    for op, a in ins:
        if op == 43:
            env[a[1]] = struct.unpack('<f', struct.pack('<I', a[2]))[0] if a[0] in floats else a[2]
        elif op in (41, 42):
            env[a[1]] = op == 41
    pcvar = next(a[1] for op, a in ins if op == 59 and a[2] == 9)
    loc0 = {a[0] for op, a in ins if op == 71 and len(a) == 3 and a[1] == 30 and a[2] == 0}
    colvar = next(a[1] for op, a in ins if op == 59 and a[2] == 3 and a[1] in loc0)
    start = next(k for k, (op, a) in enumerate(ins) if op == 54 and a[1] == entry)
    body = []
    for op, a in ins[start + 1:]:
        if op == 56:
            break
        body.append((op, a))
    labels = {a[0]: k for k, (op, a) in enumerate(body) if op == 248}
    k, steps = 0, 0
    while True:
        op, a = body[k]
        k += 1
        steps += 1
        assert steps < 1000
        if op in (248, 247, 57):   # OpLabel, OpSelectionMerge, OpFunctionCall (the shader's own main: colour 0 is the one given)
            continue
        if op == 65:   # OpAccessChain: into the push constants, or (a fragment depth variant made after this one) gl_FragCoord
            env[a[1]] = ('pc', env[a[3]]) if a[2] == pcvar else ('other', None)
        elif op == 61:
            if a[2] == colvar:
                env[a[1]] = [0.0, 0.0, 0.0, f32(alpha)]
            elif env.get(a[2], ('other',))[0] == 'pc':
                env[a[1]] = func if env[a[2]][1] == 0 else f32(ref)
            else:
                env[a[1]] = None
        elif op == 62:   # OpStore (gl_FragDepth)
            continue
        elif op == 81:
            env[a[1]] = env[a[2]][a[3]]
        elif op in CMP:
            env[a[1]] = CMP[op](env[a[2]], env[a[3]])
        elif op == 170:
            env[a[1]] = env[a[2]] == env[a[3]]
        elif op == 169:
            env[a[1]] = env[a[3]] if env[a[2]] else env[a[4]]
        elif op == 250:
            k = labels[a[1] if env[a[0]] else a[2]]
        elif op == 249:
            k = labels[a[0]]
        elif op in (252, 4416):
            return False
        elif op == 253:
            return True
        else:
            raise AssertionError(f'instruction {op} in the entry point')


FRAG = r'''#version 450
layout(location = 0) in vec4 v;
layout(location = 1) in vec4 u;
layout(location = 0) out vec4 o;
void main() {
    if (v.x < -1.0) discard;
    o = v * u;
    if (v.y > 2.0) return;
    o.w = v.w;
}
'''
FRAG_DUAL = r'''#version 450
layout(location = 0) in vec4 v;
layout(location = 0) out vec4 o;
layout(location = 1) out vec4 o1;
void main() { o = v; o1 = v.wzyx; }
'''
FRAG_INT = r'''#version 450
layout(location = 0) flat in uvec4 v;
layout(location = 0) out uvec4 o;
void main() { o = v; }
'''
FRAG_NONE = r'''#version 450
layout(location = 0) in vec4 v;
layout(location = 1) out vec4 o1;
void main() { o1 = v; }
'''
FRAG_PUSH = r'''#version 450
layout(push_constant) uniform P { vec4 k; } p;
layout(location = 0) out vec4 o;
void main() { o = p.k; }
'''
ALPHAS = [0.0, -0.0, 0.1, 0.25, 0.5, 0.75, 1.0, -1.0, float('inf'), float('nan')]
REFS = [0.0, 0.1, 0.5, 1.0, float('nan')]


# A mesh shader as the translator writes them (Mario Kart World 1.8.0's, XLATE): gl_Position a standalone Output array decorated BuiltIn Position, written through access chains
# (one with a component index), a Location 0 output, the triangle indices and a task payload. The A/B variants of gfx_vk_exec.c (DM_VK_MESH_NOTASK: the payload made Private;
# DM_VK_MESH_POSBLOCK: gl_Position in a Block, as glslang declares gl_MeshVerticesEXT) must stay valid; a whole-array store has no block variant
MESH_ASM = """
OpCapability Shader
OpCapability MeshShadingEXT
OpExtension "SPV_EXT_mesh_shader"
OpMemoryModel Logical GLSL450
OpEntryPoint MeshEXT %main "main" %pos %col %idx %pl %lid
OpExecutionMode %main LocalSize 32 1 1
OpExecutionMode %main OutputVertices 64
OpExecutionMode %main OutputPrimitivesEXT 32
OpExecutionMode %main OutputTrianglesEXT
OpDecorate %pos BuiltIn Position
OpDecorate %col Location 0
OpDecorate %idx BuiltIn PrimitiveTriangleIndicesEXT
OpDecorate %lid BuiltIn LocalInvocationIndex
%void = OpTypeVoid
%fn = OpTypeFunction %void
%float = OpTypeFloat 32
%uint = OpTypeInt 32 0
%v4 = OpTypeVector %float 4
%v3u = OpTypeVector %uint 3
%u64 = OpConstant %uint 64
%u32 = OpConstant %uint 32
%u2 = OpConstant %uint 2
%f1 = OpConstant %float 1
%av4 = OpTypeArray %v4 %u64
%av3 = OpTypeArray %v3u %u32
%apl = OpTypeArray %uint %u32
%pav4 = OpTypePointer Output %av4
%pav3 = OpTypePointer Output %av3
%ppl = OpTypePointer TaskPayloadWorkgroupEXT %apl
%pv4 = OpTypePointer Output %v4
%pf = OpTypePointer Output %float
%pv3 = OpTypePointer Output %v3u
%pplu = OpTypePointer TaskPayloadWorkgroupEXT %uint
%piu = OpTypePointer Input %uint
%pos = OpVariable %pav4 Output
%col = OpVariable %pav4 Output
%idx = OpVariable %pav3 Output
%pl = OpVariable %ppl TaskPayloadWorkgroupEXT
%lid = OpVariable %piu Input
%main = OpFunction %void None %fn
%l = OpLabel
%i = OpLoad %uint %lid
%a = OpAccessChain %pplu %pl %i
%x = OpLoad %uint %a
%xf = OpConvertUToF %float %x
OpSetMeshOutputsEXT %u64 %u32
%v = OpCompositeConstruct %v4 %xf %xf %f1 %f1
%pp = OpAccessChain %pv4 %pos %i
OpStore %pp %v
%pz = OpAccessChain %pf %pos %i %u2
OpStore %pz %f1
%pc = OpAccessChain %pv4 %col %i
OpStore %pc %v
%t = OpCompositeConstruct %v3u %i %i %i
%pi = OpAccessChain %pv3 %idx %i
OpStore %pi %t
OpReturn
OpFunctionEnd
"""
MESH_BUNDLE = DM.parents[1] / '.gamedump/MarioKartWorld/MarioKartWorld-data/shaders.dmsv'   # (the ported game's bundle: present on the machine that ported it)


class AlphaTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory(prefix='nx2a_vk_alpha_')
        cls.dir = Path(cls.tmp.name)
        cls.glslc, cls.val = ndk_tool('glslc'), ndk_tool('spirv-val')
        cc = shutil.which('cc') or shutil.which('clang')
        cls.tool = None
        if cc:
            c = cls.dir / 'spvx.c'
            c.write_text(tool_source(), encoding='utf-8')
            r = subprocess.run([cc, '-std=gnu11', '-O1', '-Wall', '-Werror', '-Wno-unused-function', '-Wno-unused-variable', str(c), '-o', str(cls.dir / 'spvx')], capture_output=True, text=True)
            if r.returncode:
                raise AssertionError(r.stdout + r.stderr)
            cls.tool = cls.dir / 'spvx'

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def need(self, glslc=True):
        if not self.tool:
            self.skipTest('a C compiler is needed')
        if glslc and not self.glslc:
            self.skipTest('glslc is needed (PATH, ANDROID_NDK_HOME or the toolchain the porter installs)')

    def compile(self, name, text, env='vulkan1.1'):
        g, o = self.dir / f'{name}.frag', self.dir / f'{name}.spv'
        g.write_text(text, encoding='utf-8')
        r = subprocess.run([self.glslc, f'--target-env={env}', str(g), '-o', str(o)], capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        return o

    def transform(self, steps, src, ok=True):
        out = self.dir / f'{src.stem}.{steps}.spv'
        r = subprocess.run([str(self.tool), steps, str(src), str(out)], capture_output=True, text=True)
        if ok:
            self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
            return out
        self.assertEqual(r.returncode, 2, r.stdout + r.stderr)
        return r.stdout.strip()

    def validate(self, spv):
        if not self.val:
            return
        r = subprocess.run([self.val, '--target-env', 'vulkan1.3', str(spv)], capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, f'{spv.name}: {r.stdout}{r.stderr}')

    def check_functions(self, spv):
        w = words(spv.read_bytes())
        for func in range(0, 10):
            for a in ALPHAS:
                for r in REFS:
                    self.assertEqual(run_entry(w, func, r, a), expected(func, f32(a), f32(r)), f'function {func}, alpha {a}, reference {r}')

    def test_every_function(self):
        self.need()
        base = self.compile('frag', FRAG)
        v = self.transform('a', base)
        self.validate(v)
        self.check_functions(v)
        w, b = words(v.read_bytes()), words(base.read_bytes())
        ins = list(insts(w))
        self.assertEqual(w[1], 0x10300)
        ep = [a for op, a in ins if op == 15]
        self.assertEqual(len(ep), 1)
        fns = [a[1] for op, a in ins if op == 54]
        self.assertEqual(len(fns), len([1 for op, a in insts(b) if op == 54]) + 1)   # the new entry point beside the shader's own main
        self.assertIn(252, [op for op, a in ins])   # OpKill in SPIR-V 1.3
        self.assertNotIn(next(a[1] for op, a in ins if op == 59 and a[2] == 9), ep[0][3:])   # a 1.3 interface lists only inputs and outputs
        self.assertTrue(any(op == 71 and a[1] == 2 for op, a in ins))   # Block
        offs = sorted(a[3] for op, a in ins if op == 72 and len(a) == 4 and a[2] == 35 and a[0] == next(x[0] for o2, x in ins if o2 == 71 and len(x) == 2 and x[1] == 2))
        self.assertEqual(offs, [0, 4])

    def test_spirv_1_6(self):
        self.need()
        v = self.transform('a', self.compile('frag16', FRAG, 'vulkan1.3'))
        self.validate(v)
        w = words(v.read_bytes())
        ins = list(insts(w))
        self.assertEqual(w[1], 0x10600)
        entry = next(a[1] for op, a in ins if op == 15)
        start = next(k for k, (op, a) in enumerate(ins) if op == 54 and a[1] == entry)
        tail = [op for op, a in ins[start:]]
        self.assertIn(4416, tail[:tail.index(56)])   # OpTerminateInvocation, not the deprecated OpKill
        pcvar = next(a[1] for op, a in ins if op == 59 and a[2] == 9)
        self.assertIn(pcvar, next(a for op, a in ins if op == 15)[3:])   # SPIR-V 1.4 lists the push constants in the interface
        self.check_functions(v)

    def test_with_the_other_variants(self):
        self.need()
        base = self.compile('frag2', FRAG)
        for steps in ('fa', 'af', 'pa', 'ap'):   # fragment depth written; per-primitive inputs (after a mesh shader)
            v = self.transform(steps, base)
            self.validate(v)
            self.check_functions(v)
        v = self.transform('ad', self.compile('dual', FRAG_DUAL))   # dual source after it: location 1 becomes location 0, index 1
        self.validate(v)
        self.check_functions(v)

    def test_left_out(self):
        self.need()
        self.assertEqual(self.transform('a', self.compile('int', FRAG_INT), ok=False), 'a: colour 0 has no float alpha')
        self.assertEqual(self.transform('a', self.compile('none', FRAG_NONE), ok=False), 'a: no colour 0 output')
        self.assertEqual(self.transform('a', self.compile('push', FRAG_PUSH), ok=False), 'a: a push constant block of its own')
        r = subprocess.run([str(self.tool), 'c', str(self.compile('int2', FRAG_INT)), str(self.dir / 'x.spv')], capture_output=True, text=True)
        self.assertEqual(r.stdout.split()[0], '2')   # VK_COL0_INT, what vk_shader records for gfx_vk_exec.c

    def test_bundle_sample(self):
        """Fragment shaders of the game's bundle, among them some that write their depth: the alpha test variant alone and with the others."""
        self.need(glslc=False)
        if not BUNDLE.is_dir() or not self.val:
            self.skipTest('the translated bundle sample and spirv-val are needed')
        frag, depth = [], []
        for p in sorted(BUNDLE.glob('*.spv')):
            ins = list(insts(words(p.read_bytes())))
            if next((a[0] for op, a in ins if op == 15), None) != 4:
                continue
            (depth if any(op == 71 and len(a) == 3 and a[1] == 11 and a[2] == 22 for op, a in ins) else frag).append(p)
            if len(frag) >= 40 and len(depth) >= 8:
                break
        for p in frag[:40] + depth[:8]:
            for steps in (('a', 'pa', 'fa') if p in frag[:40] else ('a', 'pa')):
                out = self.dir / f'{p.stem}.{steps}.spv'
                r = subprocess.run([str(self.tool), steps, str(p), str(out)], capture_output=True, text=True)
                if r.returncode == 2 and r.stdout.startswith(('a: no colour 0 output', 'p: no vec4 inputs')):
                    continue
                self.assertEqual(r.returncode, 0, f'{p.name} {steps}: {r.stdout}{r.stderr}')
                self.validate(out)
                self.check_functions(out) if steps == 'a' and p == frag[0] else None
                out.unlink()


    def mesh_checks(self, out, steps):
        ins = list(insts(words(out.read_bytes())))
        self.assertFalse(any(op in (32, 59) and len(a) >= 3 and a[1 if op == 32 else 2] == 5402 for op, a in ins) and 'm' in steps)   # no TaskPayloadWorkgroupEXT left
        if 'b' in steps:
            blk = [a[0] for op, a in ins if op == 72 and a[1:] == [0, 11, 0]]   # OpMemberDecorate %blk 0 BuiltIn Position
            self.assertEqual(len(blk), 1)
            self.assertIn((71, [blk[0], 2]), [(op, a) for op, a in ins])   # Block
            self.assertFalse(any(op == 71 and a[1:] == [11, 0] for op, a in ins))   # no standalone BuiltIn Position

    def test_mesh_variants(self):
        """The task / mesh A/B variants (gfx_vk_exec.c DM_VK_MESH_NOTASK, DM_VK_MESH_POSBLOCK) of a mesh shader written as the translator writes them, and of the game's when its bundle is here."""
        self.need(glslc=False)
        asm = ndk_tool('spirv-as')
        if not asm or not self.val:
            self.skipTest('spirv-as and spirv-val are needed')
        src, base = self.dir / 'mesh.spvasm', self.dir / 'mesh.spv'
        src.write_text(MESH_ASM, encoding='utf-8')
        r = subprocess.run([asm, '--target-env', 'spv1.4', str(src), '-o', str(base)], capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.validate(base)
        for steps in ('m', 'b', 'mb'):
            out = self.transform(steps, base)
            self.validate(out)
            self.mesh_checks(out, steps)
        chains = [a for op, a in insts(words(self.transform('b', base).read_bytes())) if op == 65 and len(a) >= 4 and a[3] != a[2]]
        self.assertIn(5, {len(a) for a in chains})   # (the component store: vertex index, member 0, component)
        whole = self.dir / 'mesh_whole.spvasm'
        whole.write_text(MESH_ASM.replace('%pz = OpAccessChain %pf %pos %i %u2\nOpStore %pz %f1', '%w = OpLoad %av4 %col\nOpStore %pos %w'), encoding='utf-8')
        r = subprocess.run([asm, '--target-env', 'spv1.4', str(whole), '-o', str(self.dir / 'mesh_whole.spv')], capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertEqual(self.transform('b', self.dir / 'mesh_whole.spv', ok=False), 'b: no variant')
        if not MESH_BUNDLE.is_file():
            return
        data = MESH_BUNDLE.read_bytes(); n = struct.unpack_from('<I', data, 8)[0]; done = 0
        for e in range(n):
            key, stage, off, ln = struct.unpack_from('<QIII', data, 12 + 32 * e)
            if stage != 7:
                continue
            p = self.dir / f'{key:016x}.spv'; p.write_bytes(data[off:off + ln])
            for steps in ('m', 'b', 'mb'):
                out = self.transform(steps, p)
                self.validate(out)
                self.mesh_checks(out, steps)
            done += 1
        self.assertGreater(done, 0)


if __name__ == '__main__':
    unittest.main()
