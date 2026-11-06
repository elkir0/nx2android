"""Exercise the Vulkan backend's Switch 2 logic without a device: the coalescing of a small indirect-count
multi-draw's records (gfx_vk_exec.c mdi_run), the resolution of NVN's advanced blend modes (adv_blend), the parts a
dispatch of a compute shader with its shared memory in a buffer is cut into (spill_dispatch) and the shared sparse
tile slots (gfx_vk_tex.c sp_*: Mario Kart World maps one physical page at several places)."""
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

DM = Path(__file__).resolve().parents[1] / 'nxnative/dm'
EXEC = DM / 'android/gfx_vk_exec.c'
TEX = DM / 'android/gfx_vk_tex.c'
VKH = DM / 'third_party/vulkan-headers/include'   # the vendored Vulkan headers (vulkan_core.h; the NDK gives vulkan.h)
INC = ['-I' + str(DM / 'gfx/nvn'), '-I' + str(DM / 'gfx/nvn/include'), '-I' + str(DM / 'gfx/tex'), '-I' + str(VKH)]


def find_glslc():
    """glslc: on PATH, in ANDROID_NDK_HOME, the repository's tools/ NDK or the NDK the porter installs (None when there is none)."""
    import glob, os
    if shutil.which('glslc'):
        return shutil.which('glslc')
    roots = [os.environ.get('ANDROID_NDK_HOME', ''), str(DM.parents[1] / 'tools/android-ndk-r27c')] + \
            sorted(glob.glob(os.path.expanduser('~/Library/Application Support/nx2android/toolchains/android-ndk-*')))
    for r in roots:
        for g in glob.glob(os.path.join(r, 'shader-tools/*/glslc')) if r else []:
            return g
    return None


def cut(src, start, end_marker, after=0):
    """The source from start up to and including the first end_marker found after it (after: skip that many matches)."""
    a = src.index(start)
    b = a
    for _ in range(after + 1):
        b = src.index(end_marker, b) + len(end_marker)
    return src[a:b]


class Switch2Logic(unittest.TestCase):
    def run_c(self, code, name):
        cc = shutil.which('cc') or shutil.which('clang')
        if not cc:
            self.skipTest('a C compiler is needed')
        if 'vulkan_core.h' in code and not (VKH / 'vulkan/vulkan_core.h').exists():
            self.skipTest('the vendored Vulkan headers are needed (nxnative/dm/third_party/vulkan-headers)')
        with tempfile.TemporaryDirectory(prefix='nx2a_vk_switch2_') as tmp:
            c, exe = Path(tmp) / f'{name}.c', Path(tmp) / name
            c.write_text(code, encoding='utf-8')
            r = subprocess.run([cc, '-std=gnu11', '-Wall', '-Werror', '-Wno-unused-function', '-Wno-unused-variable', '-pthread', *INC, str(c), '-o', str(exe)],
                               capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
            r = subprocess.run([str(exe)], capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stdout + r.stderr)

    def test_multi_draw_runs(self):
        src = EXEC.read_text(encoding='utf-8')
        body = cut(src, 'extern uint32_t g_sub_ops[64];\nstatic const uint8_t *mdi_run', '    return end;\n}\n')
        code = r'''
#include <assert.h>
#include <stdint.h>
#include <string.h>
#include "nvn_obj.h"
uint32_t g_sub_ops[64];
static struct { int prim, itype, calls; uint64_t ibuf, ind, cnt; uint32_t first, n, stride; } got;
static void multi_draw(int prim, int itype, uint64_t ibuf, uint64_t ind, uint64_t cnt, uint32_t first, uint32_t n, uint32_t stride) {
    got.prim = prim; got.itype = itype; got.ibuf = ibuf; got.ind = ind; got.cnt = cnt; got.first = first; got.n = n; got.stride = stride; got.calls++;
}
''' + body + r'''
static uint8_t m1[4096] __attribute__((aligned(8))), m2[4096] __attribute__((aligned(8)));
static uint8_t *idx(uint8_t *p, int i, uint64_t ind, uint64_t cnt, uint64_t ibuf) {
    p_draw_idx d = {4, 1, 0, 0, 0, 1, ibuf, ind, cnt, i, 0}; op_hdr h = {OP_DRAW_IDX, (uint32_t)((8 + sizeof d + 7) & ~7u)};
    memcpy(p, &h, 8); memcpy(p + 8, &d, sizeof d); return p + h.size; }
static uint8_t *arr(uint8_t *p, int i, uint64_t ind, uint64_t cnt) {
    p_draw d = {4, 0, 0, 1, 0, ind, cnt, i, 0}; op_hdr h = {OP_DRAW, (uint32_t)((8 + sizeof d + 7) & ~7u)};
    memcpy(p, &h, 8); memcpy(p + 8, &d, sizeof d); return p + h.size; }
static uint8_t *op(uint8_t *p, uint32_t o) { op_hdr h = {o, 8}; memcpy(p, &h, 8); return p + 8; }
static uint8_t *jump(uint8_t *p, const uint8_t *to) { op_hdr h = {OP_JUMP, 16}; memcpy(p, &h, 8); memcpy(p + 8, &to, 8); return p + 16; }
int main(void) {
    // five records of one call, stride 20, a segment's JUMP after the third: one multi-draw, the walk ends at the record after the last
    uint8_t *p = m1; for (int i = 0; i < 3; i++) p = idx(p, i, 0x1000 + 20 * i, 0x9000, 0x5000);
    uint8_t *q = m2; for (int i = 3; i < 5; i++) q = idx(q, i, 0x1000 + 20 * i, 0x9000, 0x5000);
    uint8_t *end = q; op(q, OP_END); jump(p, m2);
    const uint8_t *r = mdi_run(m1);
    assert(got.calls == 1 && got.n == 5 && got.stride == 20 && got.first == 0 && got.ind == 0x1000 && got.cnt == 0x9000 && got.ibuf == 0x5000 && got.itype == 1 && r == end);
    // a stride that changes ends the run: two records, then the third starts its own
    memset(&got, 0, sizeof got); p = m1; p = idx(p, 0, 0x1000, 0x9000, 0x5000); p = idx(p, 1, 0x1014, 0x9000, 0x5000); uint8_t *third = p; p = idx(p, 2, 0x1030, 0x9000, 0x5000); op(p, OP_END);
    r = mdi_run(m1); assert(got.calls == 1 && got.n == 2 && got.stride == 20 && r == third);
    r = mdi_run(third); assert(got.calls == 2 && got.n == 1 && got.first == 2 && got.stride == 0);
    // another count address, or another index buffer, is another call
    memset(&got, 0, sizeof got); p = m1; p = idx(p, 0, 0x1000, 0x9000, 0x5000); uint8_t *second = p; p = idx(p, 1, 0x1014, 0x9100, 0x5000); op(p, OP_END);
    r = mdi_run(m1); assert(got.n == 1 && r == second);
    memset(&got, 0, sizeof got); p = m1; p = idx(p, 0, 0x1000, 0x9000, 0x5000); second = p; p = idx(p, 1, 0x1014, 0x9000, 0x6000); op(p, OP_END);
    r = mdi_run(m1); assert(got.n == 1 && r == second);
    // the draws of DrawArrays: no index buffer (itype -1), stride 16
    memset(&got, 0, sizeof got); p = m1; for (int i = 0; i < 4; i++) p = arr(p, i, 0x2000 + 16 * i, 0x9000); end = p; op(p, OP_BARRIER);
    r = mdi_run(m1); assert(got.calls == 1 && got.n == 4 && got.stride == 16 && got.itype == -1 && got.ibuf == 0 && r == end);
    return 0;
}
'''
        self.run_c(code, 'mdi')

    def test_advanced_blend(self):
        src = EXEC.read_text(encoding='utf-8')
        body = cut(src, 'enum { ADV_KEY = 0x10000 };', '    return 0;\n}\n')
        code = r'''
#include <assert.h>
#include <stdint.h>
#include <string.h>
#define VK_NO_PROTOTYPES
#include <vulkan/vulkan_core.h>
#include "nvn_obj.h"
static struct { int has_adv_blend, adv_npm, adv_ovl, adv_all; uint32_t adv_max_att; } V;
#define LOG(...) ((void)0)
''' + body + r'''
static int res(int mode, int npm, int ovl, int target, uint32_t o[6]) { blend_st b; memset(&b, 0, sizeof b); b.adv = (unsigned)mode; b.adv_npm = (unsigned)npm; b.adv_ovl = (unsigned)ovl; memset(o, 0xee, 24); return adv_blend(&b, target, o); }
int main(void) {
    enum { Z = NVN_BLEND_FUNC_ZERO, O = NVN_BLEND_FUNC_ONE, SA = NVN_BLEND_FUNC_SRC_ALPHA, ISA = NVN_BLEND_FUNC_ONE_MINUS_SRC_ALPHA, DA = NVN_BLEND_FUNC_DST_ALPHA, DC = NVN_BLEND_FUNC_DST_COLOR };
    uint32_t o[6];
    // Porter-Duff modes: exact with factors, a source not premultiplied through SRC_ALPHA where its factor is ONE
    assert(res(NVN_BLEND_ADVANCED_MODE_SRC_OVER, 0, 0, 0, o) && o[0] == O && o[1] == ISA && o[2] == O && o[3] == ISA && o[4] == NVN_BLEND_EQUATION_ADD && o[5] == NVN_BLEND_EQUATION_ADD);
    assert(res(NVN_BLEND_ADVANCED_MODE_SRC_OVER, 1, 0, 0, o) && o[0] == SA && o[1] == ISA);
    assert(res(NVN_BLEND_ADVANCED_MODE_DST_IN, 1, 0, 0, o) && o[0] == Z && o[1] == SA);
    assert(res(NVN_BLEND_ADVANCED_MODE_SRC_ATOP, 0, 0, 0, o) && o[0] == DA && o[1] == ISA && o[2] == Z && o[3] == O);
    // what factors cannot say, with no VK_EXT_blend_operation_advanced: the state's own functions (0)
    assert(!res(NVN_BLEND_ADVANCED_MODE_SRC_IN, 1, 0, 0, o));        // DA × a source not premultiplied
    assert(!res(NVN_BLEND_ADVANCED_MODE_SRC_OVER, 0, 1, 0, o));      // disjoint overlap
    assert(!res(NVN_BLEND_ADVANCED_MODE_OVERLAY, 0, 0, 0, o));
    // multiply and screen over an opaque destination (this fork's Viractal rule), premultiplied and uncorrelated only
    assert(res(NVN_BLEND_ADVANCED_MODE_MULTIPLY, 0, 0, 0, o) && o[0] == DC && o[1] == ISA && o[2] == O && o[3] == ISA);
    assert(res(NVN_BLEND_ADVANCED_MODE_SCREEN, 0, 0, 0, o) && o[0] == NVN_BLEND_FUNC_ONE_MINUS_DST_COLOR && o[1] == O);
    assert(!res(NVN_BLEND_ADVANCED_MODE_MULTIPLY, 1, 0, 0, o));
    // the extension: KHR_blend_equation_advanced's modes, the rest with advancedBlendAllOperations; npm and overlap where the device takes them; within its targets
    V.has_adv_blend = 1; V.adv_max_att = 1;
    assert(res(NVN_BLEND_ADVANCED_MODE_OVERLAY, 0, 0, 0, o) && o[4] == (ADV_KEY | NVN_BLEND_ADVANCED_MODE_OVERLAY) && o[5] == 0 && adv_vk_op(NVN_BLEND_ADVANCED_MODE_OVERLAY) == VK_BLEND_OP_OVERLAY_EXT);
    assert(!res(NVN_BLEND_ADVANCED_MODE_OVERLAY, 0, 0, 1, o));       // target 1: past advancedBlendMaxColorAttachments
    assert(!res(NVN_BLEND_ADVANCED_MODE_OVERLAY, 1, 0, 0, o));       // no advancedBlendNonPremultipliedSrcColor
    V.adv_npm = 1; V.adv_ovl = 1;
    assert(res(NVN_BLEND_ADVANCED_MODE_HARDLIGHT, 1, 2, 0, o) && o[4] == (ADV_KEY | NVN_BLEND_ADVANCED_MODE_HARDLIGHT) && o[5] == (1 | 2 << 1));
    assert(res(NVN_BLEND_ADVANCED_MODE_SRC_OVER, 0, 0, 0, o) && o[4] == NVN_BLEND_EQUATION_ADD);   // factors first when they are exact
    assert(!res(NVN_BLEND_ADVANCED_MODE_PINLIGHT, 0, 0, 0, o));     // not in the basic set
    V.adv_all = 1; assert(res(NVN_BLEND_ADVANCED_MODE_PINLIGHT, 0, 0, 0, o) && adv_vk_op(NVN_BLEND_ADVANCED_MODE_PINLIGHT) == VK_BLEND_OP_PINLIGHT_EXT);
    for (int m = NVN_BLEND_ADVANCED_MODE_ZERO; m <= NVN_BLEND_ADVANCED_MODE_HSL_LUMINOSITY; m++) assert(adv_vk_op(m) != VK_BLEND_OP_MAX_ENUM);   // every NVN mode has its Vulkan op
    return 0;
}
'''
        self.run_c(code, 'adv')

    def test_spill_dispatch_parts(self):
        src = EXEC.read_text(encoding='utf-8')
        body = cut(src, 'static uint64_t g_spill_parts, g_spill_whole;', 'k, sl);\n}\n')
        code = r'''
#include <assert.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#define VK_NO_PROTOTYPES
#include <vulkan/vulkan_core.h>
enum { RK_SPILL = 15 };
typedef struct { uint8_t kind; int32_t samp_arg; } bind_t;
typedef struct { uint64_t key; int nbind; const bind_t *b; } vkshader_t;
static struct { int has_dbase; } V = {1};
static int g_spill_split = 1, reds;
static struct { uint16_t spill; } g_pu; static uint16_t pu_add(uint16_t a, uint32_t n) { return (uint16_t)(a + n > 65535 ? 65535 : a + n); }   // (the GPU profile's unit: gfx_vk_exec.c)
#define VKLOG(...) ((void)0)
#define VKRED_ONCE(id, ...) (reds++)
enum { MAXP = 1 << 16 };
static struct { uint32_t b[3], c[3]; int bars; } P[MAXP]; static int np, nbar;
static void vkCmdDispatch(VkCommandBuffer cb, uint32_t x, uint32_t y, uint32_t z) { (void)cb; assert(np < MAXP); P[np].b[0] = P[np].b[1] = P[np].b[2] = 0; P[np].c[0] = x; P[np].c[1] = y; P[np].c[2] = z; P[np++].bars = nbar; }
static void vkCmdDispatchBase(VkCommandBuffer cb, uint32_t bx, uint32_t by, uint32_t bz, uint32_t x, uint32_t y, uint32_t z) { (void)cb; assert(np < MAXP);
    P[np].b[0] = bx; P[np].b[1] = by; P[np].b[2] = bz; P[np].c[0] = x; P[np].c[1] = y; P[np].c[2] = z; P[np++].bars = nbar; }
static void vkCmdPipelineBarrier(VkCommandBuffer cb, VkPipelineStageFlags s, VkPipelineStageFlags d, VkDependencyFlags f, uint32_t nm, const VkMemoryBarrier *m, uint32_t nb,
                                 const VkBufferMemoryBarrier *b, uint32_t ni, const VkImageMemoryBarrier *im) {
    (void)cb; (void)f; (void)nb; (void)b; (void)ni; (void)im;
    assert(s == VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT && d == VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT && nm == 1 && (m->srcAccessMask & VK_ACCESS_SHADER_WRITE_BIT) && (m->dstAccessMask & VK_ACCESS_SHADER_WRITE_BIT)); nbar++; }
''' + body + r'''
// one dispatch through spill_dispatch: every workgroup exactly once, and within a part the slices XLATE's formula gives (WorkgroupId with the part's base, NumWorkgroups the part's counts) distinct
static int check(uint32_t x, uint32_t y, uint32_t z, uint32_t sl) {
    static const bind_t bs[2] = {{3, 0}, {RK_SPILL, 0}}; bind_t b2[2] = {bs[0], bs[1]}; b2[1].samp_arg = (int32_t)sl; vkshader_t cs = {0x1234, 2, b2};
    assert(spill_slices(&cs) == sl);
    np = nbar = 0; spill_dispatch(NULL, &cs, x, y, z, spill_slices(&cs));
    uint64_t n = (uint64_t)x * y * z; uint8_t *seen = calloc(n ? n : 1, 1), *used = calloc(sl, 1);
    for (int i = 0; i < np; i++) {
        assert(P[i].bars == i);   // a barrier before every part but the first
        memset(used, 0, sl);
        for (uint32_t lz = 0; lz < P[i].c[2]; lz++) for (uint32_t ly = 0; ly < P[i].c[1]; ly++) for (uint32_t lx = 0; lx < P[i].c[0]; lx++) {
            uint32_t gx = P[i].b[0] + lx, gy = P[i].b[1] + ly, gz = P[i].b[2] + lz; assert(gx < x && gy < y && gz < z);
            uint64_t w = ((uint64_t)gz * y + gy) * x + gx; assert(!seen[w]); seen[w] = 1;
            uint32_t lin = (gz * P[i].c[1] + gy) * P[i].c[0] + gx, s = lin % sl;   // SharedMemorySpill.cs EmitBase (u32 arithmetic)
            if (np > 1 || n <= sl) { assert(!used[s]); used[s] = 1; }
        }
    }
    for (uint64_t w = 0; w < n; w++) assert(seen[w]);
    free(seen); free(used);
    return np;
}
int main(void) {
    assert(check(1, 1, 1, 512) == 1 && check(512, 1, 1, 512) == 1 && check(8, 8, 8, 512) == 1);   // within the slices: one dispatch, no barrier
    assert(check(513, 1, 1, 512) == 2 && check(1000, 3, 2, 512) == 12);   // a row past the slices: runs of it
    assert(check(120, 68, 1, 512) == 17);   // rows: 4 of 120 a part (Mario Kart World's size of a screen pass of 16x16 tiles)
    assert(check(16, 16, 4, 512) == 2 && check(1, 1, 5000, 512) == 10 && check(7, 7, 100, 512) == 10);   // whole planes
    assert(check(3, 1000, 1, 512) == 6 && check(600, 2, 2, 512) == 8 && check(5, 3, 2, 7) == 6 && check(2, 2, 9, 7) == 9);
    assert(check(0, 4, 4, 512) == 1 && reds == 0);
    // left whole, with a [RED] once: more than 4096 parts, no vkCmdDispatchBase, DM_VK_SPILL_SPLIT=0
    np = 0; static const bind_t sb[1] = {{RK_SPILL, 512}}; vkshader_t cs = {1, 1, sb};
    spill_dispatch(NULL, &cs, 1024, 4096, 1, 512); assert(np == 1 && P[0].c[0] == 1024 && P[0].c[1] == 4096 && reds == 1 && g_spill_whole == 1);
    V.has_dbase = 0; np = 0; spill_dispatch(NULL, &cs, 1024, 1, 1, 512); assert(np == 1 && P[0].c[0] == 1024 && g_spill_whole == 2); V.has_dbase = 1;
    g_spill_split = 0; np = 0; spill_dispatch(NULL, &cs, 1024, 1, 1, 512); assert(np == 1 && g_spill_whole == 3);
    vkshader_t none = {2, 0, NULL}; assert(spill_slices(&none) == 0 && spill_slices(NULL) == 0);
    return 0;
}
'''
        self.run_c(code, 'spill')

    def test_mesh_needs(self):
        glslc = find_glslc()
        if not glslc:
            self.skipTest('glslc is needed (PATH, ANDROID_NDK_HOME or the toolchain the porter installs)')
        mesh = r'''#version 460
#extension GL_EXT_mesh_shader : require
#extension GL_KHR_shader_subgroup_ballot : require
layout(local_size_x = 32) in;
layout(triangles, max_vertices = 64, max_primitives = 126) out;
struct P { uint a[8]; };
taskPayloadSharedEXT P pay;
shared uint s_data[1000];
shared vec4 s_v[10];
void main() {
    uvec4 b = subgroupBallot(gl_LocalInvocationIndex < 7u);
    s_data[gl_LocalInvocationIndex] = b.x + pay.a[gl_LocalInvocationIndex & 7u];
    s_v[gl_LocalInvocationIndex % 10u] = vec4(1.0);
    barrier();
    SetMeshOutputsEXT(3u, 1u);
    if (gl_LocalInvocationIndex < 3u) gl_MeshVerticesEXT[gl_LocalInvocationIndex].gl_Position = vec4(float(s_data[0]), s_v[0].y, 0.0, 1.0);
    if (gl_LocalInvocationIndex == 0u) gl_PrimitiveTriangleIndicesEXT[0] = uvec3(0u, 1u, 2u);
}
'''
        task = r'''#version 460
#extension GL_EXT_mesh_shader : require
layout(local_size_x = 16, local_size_y = 2) in;
struct P { uint a[8]; };
taskPayloadSharedEXT P pay;
shared uint t[300];
void main() {
    t[gl_LocalInvocationIndex] = gl_LocalInvocationIndex;
    barrier();
    if (gl_LocalInvocationIndex < 8u) pay.a[gl_LocalInvocationIndex] = t[gl_LocalInvocationIndex + 1u];
    EmitMeshTasksEXT(1u, 1u, 1u);
}
'''
        words = {}
        with tempfile.TemporaryDirectory(prefix='nx2a_vk_mesh_') as tmp:
            for name, stage, text, spv in (('mesh', 'mesh', mesh, []), ('task', 'task', task, []), ('task14', 'task', task, ['--target-spv=spv1.4'])):   # (1.4: XLATE's, LocalSize)
                g, o = Path(tmp) / f'{name}.glsl', Path(tmp) / f'{name}.spv'
                g.write_text(text, encoding='utf-8')
                r = subprocess.run([glslc, f'-fshader-stage={stage}', '--target-env=vulkan1.3', *spv, str(g), '-o', str(o)], capture_output=True, text=True)
                self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
                words[name] = ', '.join(str(int.from_bytes(o.read_bytes()[i:i + 4], 'little')) + 'u' for i in range(0, len(o.read_bytes()), 4))
        src = EXEC.read_text(encoding='utf-8')
        body = cut(src, 'static uint32_t spv_group_op(uint32_t op) {', '    return s->dev_ok == 1 ? NULL : why;\n}\n')
        code = r'''
#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define VK_NO_PROTOTYPES
#include <vulkan/vulkan_core.h>
typedef struct { uint64_t key; uint32_t *spv, nspv; int dev_ok; } vkshader_t;
static struct { uint32_t spv_max, sg_stages, sg_ops; struct { uint32_t task_smem, mesh_smem, task_payload, task_pl_smem, mesh_pl_smem, out_verts, out_prims, task_inv, mesh_inv, task_size[3], mesh_size[3]; } ml; } V;
static char said[600];
#define LOG(...) snprintf(said, sizeof said, __VA_ARGS__)
#define VKLOG(...) snprintf(said, sizeof said, __VA_ARGS__)
static uint32_t MS[] = {''' + words['mesh'] + r'''}, TS[] = {''' + words['task'] + r'''}, T14[] = {''' + words['task14'] + r'''};
''' + body + r'''
static void limits(void) {
    V.spv_max = 0x10600; V.sg_stages = VK_SHADER_STAGE_TASK_BIT_EXT | VK_SHADER_STAGE_MESH_BIT_EXT | VK_SHADER_STAGE_COMPUTE_BIT; V.sg_ops = 0xff;
    V.ml.task_smem = V.ml.mesh_smem = 28672; V.ml.task_payload = 16384; V.ml.task_pl_smem = V.ml.mesh_pl_smem = 32768; V.ml.out_verts = V.ml.out_prims = 256; V.ml.task_inv = V.ml.mesh_inv = 128;
    for (int i = 0; i < 3; i++) V.ml.task_size[i] = V.ml.mesh_size[i] = 128;
}
static vkshader_t ms = {0xa1, MS, sizeof MS / 4, 0}, ts = {0xb2, TS, sizeof TS / 4, 0};
static const char *again(vkshader_t *s, int task) { s->dev_ok = 0; said[0] = 0; return mesh_need(s, task); }
int main(void) {
    limits();
    // what the modules ask: Workgroup memory 1000 x 4 + 10 x 16 bytes, the 32-byte payload, the execution modes, the ballot (VK_SUBGROUP_FEATURE_BALLOT_BIT)
    assert(!mesh_need(&ms, 0) && ms.dev_ok == 1);
    assert(strstr(said, "mesh shader 00000000000000a1: SPIR-V 1.6, shared memory 4160 bytes, payload 32 bytes, workgroup 32x1x1, 64 vertices and 126 primitives, subgroup operations 0x8: the device takes it"));
    assert(!mesh_need(&ts, 1) && ts.dev_ok == 1 && strstr(said, "shared memory 1200 bytes, payload 32 bytes, workgroup 16x2x1, subgroup operations 0:"));
    said[0] = 0; assert(!mesh_need(&ms, 0) && !said[0]);   // looked at once
    vkshader_t t14 = {0xb3, T14, sizeof T14 / 4, 0}; assert(!mesh_need(&t14, 1) && strstr(said, "SPIR-V 1.4, shared memory 1200 bytes, payload 32 bytes, workgroup 16x2x1,"));
    // each limit in turn
    V.ml.mesh_smem = 4159; const char *w = again(&ms, 0); assert(w && strstr(w, "its 4160 bytes of shared memory are past the device's 4159") && ms.dev_ok == 2 && strstr(said, "[RED]"));
    assert(strstr(mesh_need(&ms, 0), "cannot run its mesh shader")); limits();
    V.sg_stages = VK_SHADER_STAGE_TASK_BIT_EXT; assert(strstr(again(&ms, 0), "subgroup operations (0x8) and the device has none in mesh shaders")); assert(!again(&ts, 1)); limits();
    V.sg_ops = 0x7; assert(strstr(again(&ms, 0), "subgroup operations 0x8 the device lacks (it has 0x7)")); limits();
    V.ml.out_prims = 100; assert(strstr(again(&ms, 0), "its 64 vertices and 126 primitives are past the device's 256 and 100")); limits();
    V.ml.mesh_inv = 16; assert(strstr(again(&ms, 0), "its 32x1x1 workgroup is past the device's (16 invocations")); limits();
    V.ml.mesh_pl_smem = 4191; assert(strstr(again(&ms, 0), "payload and shared memory (4192 bytes) are past the device's 4191")); limits();
    V.spv_max = 0x10500; assert(strstr(again(&ms, 0), "its SPIR-V is version 1.6, the device takes up to 1.5")); limits();
    V.ml.task_payload = 16; assert(strstr(again(&ts, 1), "its 32-byte payload is past the device's 16")); limits();
    V.ml.task_smem = 1000; assert(strstr(again(&ts, 1), "its 1200 bytes of shared memory are past the device's 1000")); limits();
    V.ml.task_size[1] = 1; assert(strstr(again(&ts, 1), "its 16x2x1 workgroup is past")); limits();
    vkshader_t none = {0xc3, NULL, 0, 0}; assert(strstr(mesh_need(&none, 0), "its SPIR-V is not at hand") && none.dev_ok == 2);
    return 0;
}
'''
        self.run_c(code, 'mesh')

    def test_sparse_shared_slots(self):
        src = TEX.read_text(encoding='utf-8')
        body = cut(src, '#define SP_TILE 0x10000u', '    pthread_mutex_unlock(&g_sp_mu);\n}\n')
        self.assertIn('void mtl_sp_phys_gone', body)
        code = r'''
#include <assert.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define VK_NO_PROTOTYPES
#include <vulkan/vulkan_core.h>
#include "nvn_tex.h"
static struct { VkDevice dev; int has_sparse_alias; } V;
static unsigned nalloc; static int fail_alloc;
static VkResult alloc_mem(VkDevice d, const VkMemoryAllocateInfo *ai, const VkAllocationCallbacks *cb, VkDeviceMemory *m) { (void)d; (void)ai; (void)cb; if (fail_alloc) return VK_ERROR_OUT_OF_DEVICE_MEMORY; *m = (VkDeviceMemory)(uintptr_t)++nalloc; return VK_SUCCESS; }
#define vkAllocateMemory alloc_mem
typedef struct promo_t promo_t;   // (vksparse's promoted format: gfx_vk_tex.c "promoted formats", tests/test_vk_promo.py)
#define VKLOG(...) ((void)0)
#define VKRED_ONCE(id, ...) ((void)0)
static void die(const char *f, ...) { (void)f; abort(); }
static uint64_t done_fence, last_fence; static uint64_t vk_fence_done(void) { return done_fence; } static uint64_t mtl_last_fence(void) { return last_fence; }
uint64_t g_tex_frame;
''' + body.replace('extern uint64_t g_tex_frame;', '') + r'''
int main(void) {
    g_sp_type = 0; uint32_t a, b, c;
    assert(sp_slot(&a) && a == 0 && nalloc == 1 && g_sp_live == 1);   // a block's slots from its front
    assert(sp_slot(&b) && b == 1);
    // a slot shared by physical memory and format: found by both, by no other format
    sp_tl *e = sp_tl_of(a); e->p = 0x7000000; e->fmt = 0x4f; sp_ph_put(a);
    assert(sp_ph_find(0x7000000, 0x4f) == a && sp_ph_find(0x7000000, 0x50) == ~0u && sp_ph_find(0x7010000, 0x4f) == ~0u);
    sp_tl_of(a)->rc++;                       // a second place maps it
    sp_release(a); assert(sp_ph_find(0x7000000, 0x4f) == a && g_sp_live == 2);
    sp_release(a); assert(sp_ph_find(0x7000000, 0x4f) == ~0u && g_sp_live == 1 && g_sp_nlate == 1);   // its last place: gone, free once the GPU is past
    // the GPU drew into a shared slot: it stays with no place, until its physical pool's Finalize
    assert(sp_slot(&c) && c == 2); e = sp_tl_of(c); e->p = 0x8000000; e->fmt = 0x4f; sp_ph_put(c); e->drawn = 1;
    sp_release(c); assert(sp_ph_find(0x8000000, 0x4f) == c && g_sp_live == 2);
    mtl_sp_phys_gone((const void *)0x8000000, 0x10000); assert(sp_ph_find(0x8000000, 0x4f) == ~0u && g_sp_live == 1 && g_sp_nlate == 2);
    // a Finalize while a place still maps the slot: it leaves the table, the place keeps it as its own
    uint32_t d; assert(sp_slot(&d)); e = sp_tl_of(d); e->p = 0x9000000; e->fmt = 0x4f; sp_ph_put(d);
    mtl_sp_phys_gone((const void *)0x9000000, 0x10000); assert(sp_ph_find(0x9000000, 0x4f) == ~0u && sp_tl_of(d)->rc == 1 && sp_tl_of(d)->p == 0 && g_sp_live == 2);
    // unbound slots come back once the GPU is past the submission that last read them
    last_fence = 7; sp_release(d); assert(g_sp_nlate == 3);   // a and c wait for fence 1, d for fence 8
    uint32_t f, g, h; assert(sp_slot(&f) && f != a && f != c && f != d && g_sp_nlate == 3);   // the GPU is past none: a fresh slot
    done_fence = 1; assert(sp_slot(&g) && (g == a || g == c) && g_sp_nlate == 1);
    done_fence = 8; assert(sp_slot(&h) && h == d && g_sp_nlate == 0);
    // the table through rehashes and backward-shift deletions
    enum { N = 6000 }; static uint32_t s[N];
    for (int i = 0; i < N; i++) { assert(sp_slot(&s[i])); e = sp_tl_of(s[i]); e->p = 0x100000000ull + (uint64_t)i * 0x10000; e->fmt = 0x80 + i % 3; sp_ph_put(s[i]); }
    assert(nalloc >= 6);
    for (int i = 0; i < N; i += 2) sp_release(s[i]);
    for (int i = 0; i < N; i++) assert(sp_ph_find(0x100000000ull + (uint64_t)i * 0x10000, 0x80 + i % 3) == (i % 2 ? s[i] : ~0u));
    mtl_sp_phys_gone((const void *)0x100000000ull, (uint64_t)N * 0x10000); assert(g_sp_phn == 0 && g_sp_shared == 0);
    for (int i = 1; i < N; i += 2) assert(sp_tl_of(s[i])->p == 0 && sp_tl_of(s[i])->rc == 1);   // (still mapped by their places)
    // PARITY G3's counts (no aliasing: each place its own slot): a page bound at two places, a drawn texture's tile bound over it, a page moving under a place, places going
    uint32_t p1, p2, p3; assert(sp_slot(&p1) && sp_slot(&p2) && sp_slot(&p3)); uint64_t m0 = g_sp_multi, g0 = g_sp_multi_gpu;
    sp_pc_add(sp_tl_of(p1), 0x7a0000000ull, 0); assert(g_sp_multi == m0 && g_sp_multi_gpu == g0);
    sp_pc_add(sp_tl_of(p2), 0x7a0000000ull, 1); assert(g_sp_multi == m0 + 1 && g_sp_multi_gpu == g0 + 1);   // (the second place, in a GPU-written texture)
    sp_pc_add(sp_tl_of(p3), 0x7a0000000ull, 0); assert(g_sp_multi == m0 + 1 && g_sp_multi_gpu == g0 + 1);   // (a third: still one page)
    sp_pc_add(sp_tl_of(p3), 0x7b0000000ull, 0); assert(g_sp_multi == m0 + 1);   // (p3 moved to another page alone there)
    sp_release(p2); assert(g_sp_multi == m0 && sp_tl_of(p2)->ph == 0);   // (only p1 left on the first page)
    sp_release(p1); assert(g_sp_multi == m0);
    sp_pc_add(sp_tl_of(p3), 0x7a0000000ull, 0); assert(g_sp_multi == m0);   // (alone again on the first page)
    sp_release(p3); assert(g_sp_multi == m0 && strstr(sp_g3_text(), "pages bound at 2 places or more"));
    for (int i = 0; i < 9000; i++) { uint32_t q; if (!sp_slot(&q)) break; sp_pc_add(sp_tl_of(q), 0x200000000ull + (uint64_t)(i / 2) * 0x10000, 0); }   // (the table through rebuilds)
    assert(g_sp_multi == m0 + 4500);
    fail_alloc = 1; uint32_t x; while (g_sp_nfree || g_sp_nlate) { if (!sp_slot(&x)) break; } assert(!sp_slot(&x));   // no memory for another block: no slot
    return 0;
}
'''
        self.run_c(code, 'sparse')


if __name__ == '__main__':
    unittest.main()
