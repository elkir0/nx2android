"""The GPU pass's logic, built on the host from the backend's own code: the merge of full barriers back to back (gfx_vk_dq.c, DM_VK_BAR_MERGE:
a barrier right after another, with only binds, dynamic state and push constants between them, is left out; any command that records work,
another command buffer, or a command buffer begun again keeps the next one; an entry point the merge does not know turns it off for good),
and the GPU profile's sums (gfx_vk_core.c, DM_VK_GPUPROF: units summed by kind and by place in the frame, the costliest passes first, a
pass's attachments and load / store ops, the ops that ended passes)."""
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import unittest

DM = Path(__file__).resolve().parents[1] / 'nxnative/dm'
DQ = DM / 'android/gfx_vk_dq.c'
CORE = DM / 'android/gfx_vk_core.c'
HDR = DM / 'android/gfx_vk.h'
VKH = DM / 'third_party/vulkan-headers/include'


def region(src, start, end):
    a = src.index(start)
    return src[a:src.index(end, a)]


STATE = ['vkCmdBindPipeline', 'vkCmdBindDescriptorSets', 'vkCmdBindVertexBuffers', 'vkCmdBindIndexBuffer', 'vkCmdSetViewport', 'vkCmdSetScissor', 'vkCmdSetBlendConstants',
         'vkCmdSetStencilReference', 'vkCmdSetStencilCompareMask', 'vkCmdSetStencilWriteMask', 'vkCmdSetDepthBias', 'vkCmdSetLineWidth', 'vkCmdPushConstants',
         'vkCmdBindTransformFeedbackBuffersEXT']

BARRIER_MAIN = r'''
static int g_real_bar, g_real_other;   // the "driver": barriers recorded, other work recorded
static VKAPI_ATTR void VKAPI_CALL real_bar(VkCommandBuffer cb, VkPipelineStageFlags s, VkPipelineStageFlags d, VkDependencyFlags f, uint32_t nm, const VkMemoryBarrier *m, uint32_t nb,
                                           const VkBufferMemoryBarrier *b, uint32_t ni, const VkImageMemoryBarrier *im) { (void)cb; (void)s; (void)d; (void)f; (void)nm; (void)m; (void)nb; (void)b; (void)ni; (void)im; g_real_bar++; }
static VKAPI_ATTR void VKAPI_CALL real_disp(VkCommandBuffer cb, uint32_t x, uint32_t y, uint32_t z) { (void)cb; (void)x; (void)y; (void)z; g_real_other++; }
static VKAPI_ATTR void VKAPI_CALL real_fill(VkCommandBuffer cb, VkBuffer d, VkDeviceSize o, VkDeviceSize n, uint32_t v) { (void)cb; (void)d; (void)o; (void)n; (void)v; g_real_other++; }
static VKAPI_ATTR void VKAPI_CALL real_ts(VkCommandBuffer cb, VkPipelineStageFlagBits s, VkQueryPool q, uint32_t i) { (void)cb; (void)s; (void)q; (void)i; g_real_other++; }
static VKAPI_ATTR void VKAPI_CALL real_endrp(VkCommandBuffer cb) { (void)cb; g_real_other++; }
static VKAPI_ATTR VkResult VKAPI_CALL real_begin(VkCommandBuffer cb, const VkCommandBufferBeginInfo *bi) { (void)cb; (void)bi; return VK_SUCCESS; }
static VKAPI_ATTR void VKAPI_CALL real_bind(VkCommandBuffer cb, VkPipelineBindPoint p, VkPipeline pl) { (void)cb; (void)p; (void)pl; }
static VKAPI_ATTR void VKAPI_CALL real_push(VkCommandBuffer cb, VkPipelineLayout l, VkShaderStageFlags s, uint32_t o, uint32_t n, const void *v) { (void)cb; (void)l; (void)s; (void)o; (void)n; (void)v; }
static VKAPI_ATTR void VKAPI_CALL real_vp(VkCommandBuffer cb, uint32_t f, uint32_t n, const VkViewport *v) { (void)cb; (void)f; (void)n; (void)v; }
#define CHECK(cond) do { if (!(cond)) { printf("failed at line %d: %s (barriers %d, merged %llu)\n", __LINE__, #cond, g_real_bar, (unsigned long long)g_bar_merged); return 1; } } while (0)
int main(int argc, char **argv) {
    vkCmdPipelineBarrier = real_bar; vkCmdDispatch = real_disp; vkCmdFillBuffer = real_fill; vkCmdWriteTimestamp = real_ts; vkCmdEndRenderPass = real_endrp; vkBeginCommandBuffer = real_begin;
    vkCmdBindPipeline = real_bind; vkCmdPushConstants = real_push; vkCmdSetViewport = real_vp;
    vk_bt_init();
    if (argc > 1) { vk_bar_merge_set(1); vk_img_barrier_all(A); vk_img_barrier_all(A); CHECK(g_real_bar == 2 && g_bar_merged == 0); puts("unknown ok"); return 0; }   // (an unknown entry point: off for good)
    VkMemoryBarrier mb = {VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    // back to back, or with only state between: one barrier
    vk_img_barrier_all(A); vk_img_barrier_all(A); CHECK(g_real_bar == 1 && g_bar_merged == 1);
    vkCmdBindPipeline(A, VK_PIPELINE_BIND_POINT_COMPUTE, VK_NULL_HANDLE); vkCmdPushConstants(A, VK_NULL_HANDLE, 0, 0, 4, &mb); vkCmdSetViewport(A, 0, 0, NULL);
    vk_img_barrier_all(A); CHECK(g_real_bar == 1 && g_bar_merged == 2);
    // work between: both kept (a dispatch, a fill, a timestamp, a pass's end, a barrier of another kind)
    vkCmdDispatch(A, 1, 1, 1); vk_img_barrier_all(A); CHECK(g_real_bar == 2);
    vkCmdFillBuffer(A, VK_NULL_HANDLE, 0, 4, 0); vk_img_barrier_all(A); CHECK(g_real_bar == 3);
    vkCmdWriteTimestamp(A, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, VK_NULL_HANDLE, 0); vk_img_barrier_all(A); CHECK(g_real_bar == 4);
    vkCmdEndRenderPass(A); vk_img_barrier_all(A); CHECK(g_real_bar == 5);
    vkCmdPipelineBarrier(A, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb, 0, NULL, 0, NULL); CHECK(g_real_bar == 6);   // (first_stores' own)
    vk_img_barrier_all(A); CHECK(g_real_bar == 7 && g_bar_merged == 2);
    // another command buffer: its own first barrier; one begun again starts empty
    vk_img_barrier_all(B); CHECK(g_real_bar == 8); vk_img_barrier_all(B); CHECK(g_real_bar == 8 && g_bar_merged == 3);
    vk_img_barrier_all(A); CHECK(g_real_bar == 9);
    VkCommandBufferBeginInfo bi = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO}; vkBeginCommandBuffer(A, &bi); vk_img_barrier_all(A); CHECK(g_real_bar == 10);
    vkBeginCommandBuffer(B, &bi); vk_img_barrier_all(A); CHECK(g_real_bar == 10 && g_bar_merged == 4);   // (B begun again leaves A's mark)
    // a run of dispatches as the replay makes it: barrier, the first stores' fill and its barrier, bind, dispatch, barrier; NVN's Barrier; the next dispatch's barrier before it
    int r0 = g_real_bar; uint64_t m0 = g_bar_merged;
    for (int i = 0; i < 4; i++) { vk_img_barrier_all(A); vkCmdFillBuffer(A, VK_NULL_HANDLE, 0, 4, 0); vkCmdPipelineBarrier(A, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb, 0, NULL, 0, NULL);
        vkCmdBindPipeline(A, VK_PIPELINE_BIND_POINT_COMPUTE, VK_NULL_HANDLE); vkCmdDispatch(A, 8, 1, 1); vk_img_barrier_all(A); vk_img_barrier_all(A); }
    printf("dispatch run: %d barriers recorded, %llu merged\n", g_real_bar - r0, (unsigned long long)(g_bar_merged - m0));
    CHECK(g_real_bar - r0 == 4 * 2 && g_bar_merged - m0 == 4 * 2);   // (each: its first-store barrier and the one after it recorded; the one before it and NVN's merged, the first's too: A ended with a barrier)
    // off (DM_VK_BAR_MERGE=0): every one recorded; on again
    vk_bar_merge_set(0); r0 = g_real_bar; vk_img_barrier_all(A); vk_img_barrier_all(A); CHECK(g_real_bar == r0 + 2);
    vk_bar_merge_set(1); vk_img_barrier_all(A); CHECK(g_real_bar == r0 + 2);
    puts("barriers ok"); return 0;
}
'''

PROF_MAIN = r'''
static const char *opname(int op) { return op == 255 ? "end" : op == 8 ? "SetRenderTargets" : op == 26 ? "Dispatch" : "other"; }
static vk_prof_t pass(uint32_t w, uint32_t h, uint32_t draws, uint16_t mesh, uint16_t alpha, uint16_t fdz, uint32_t c0, uint32_t d, uint16_t load, uint16_t store, int end) {
    vk_prof_t u; memset(&u, 0, sizeof u); u.kind = VK_PU_PASS; u.w = w; u.h = h; u.draws = draws; u.mesh = mesh; u.alpha = alpha; u.fdz = fdz; u.ncol = c0 ? 1 : 0; u.cfmt[0] = c0; u.dfmt = d;
    u.load = load; u.store = store; u.end_op = (uint8_t)end; return u; }
static vk_prof_t gap(uint16_t disp, uint16_t copies, uint16_t bars, uint16_t merged) { vk_prof_t u; memset(&u, 0, sizeof u); u.kind = VK_PU_GAP; u.disp = disp; u.copies = copies; u.bars = bars; u.merged = merged; return u; }
int main(void) {
    static prof_sums s; static char b1[3000], b2[4000];
    for (uint64_t f = 0; f < 3; f++) {   // three frames of the same shape: a dispatch stretch, a mesh pass, a copy, an alpha-tested pass in the fragment-depth variant, barriers alone
        vk_prof_t u[6] = {gap(2, 0, 3, 2), pass(1280, 720, 100, 5, 0, 0, 97, 129, 1 | 1 << 8, 1 << 8, 8), gap(0, 1, 2, 0), pass(640, 360, 50, 0, 3, 2, 37, 0, 0, 0, 26), gap(0, 0, 1, 1),
                          pass(64, 64, 1, 0, 0, 0, 37, 0, 2, 0, 255)};
        double ms[6] = {1.0, 5.0, 0.5, 3.0, 0.2, 0.1};
        for (int i = 0; i < 6; i++) { u[i].fidx = (uint32_t)i; prof_add(&s, &u[i], ms[i], f); }
        s.sub_ms += 10.0;
    }
    assert(s.frames == 3 && s.max == 6);
    assert(s.kn[PK_PASS] == 9 && s.kn[PK_MESH] == 3 && s.kn[PK_ALPHA] == 3 && s.kn[PK_FDZ] == 3 && s.kn[PK_GAP] == 9 && s.kn[PK_DISP] == 3 && s.kn[PK_COPY] == 3 && s.kn[PK_BAR] == 3);
    assert(s.kms[PK_PASS] > 24.29 && s.kms[PK_PASS] < 24.31 && s.kms[PK_MESH] == 15 && s.kms[PK_ALPHA] == 9 && s.kms[PK_FDZ] == 9);
    assert(s.kms[PK_GAP] > 5.09 && s.kms[PK_GAP] < 5.11 && s.kms[PK_DISP] == 3 && s.kms[PK_COPY] == 1.5 && s.kms[PK_BAR] > 0.59 && s.kms[PK_BAR] < 0.61);
    assert(s.cnt[PC_DRAWS] == 3 * 151 && s.cnt[PC_MESH] == 15 && s.cnt[PC_ALPHA] == 9 && s.cnt[PC_FDZ] == 6 && s.cnt[PC_DISP] == 6 && s.cnt[PC_BARS] == 18 && s.cnt[PC_MERGED] == 9);
    assert(s.ends[8] == 3 && s.ends[26] == 3 && s.ends[255] == 3);
    prof_lines(&s, b1, sizeof b1, b2, sizeof b2, opname);
    puts(b1); puts(b2);
    assert(strstr(b1, "3 frames: timed units 9.80 ms a frame (submissions 10.00 ms, 0.20 outside the units"));
    assert(strstr(b1, "passes 3.0 a frame, 8.10 ms (with mesh draws 1.0: 5.00 ms; with alpha-tested draws 1.0: 3.00 ms; with fragment-depth-variant draws 1.0: 3.00 ms)"));
    assert(strstr(b1, "between passes 3.0, 1.70 ms (with dispatches 1.0: 1.00 ms; copies, clears, uniform landings, LRZ touches 1.0: 0.50 ms; barriers alone 1.0: 0.20 ms)"));
    assert(strstr(b1, "151 draws (5 mesh, 3 alpha-tested, 2 in the fragment-depth variant") && strstr(b1, "6 full barriers (3 more merged into the one before)"));
    assert(strstr(b1, "ended by SetRenderTargets 1.0, Dispatch 1.0, end 1.0") || strstr(b1, "ended by Dispatch 1.0") || strstr(b1, "ended by end 1.0"));
    const char *p1 = strstr(b2, "#1 5.00 ms 1280x720 colour 97 0 0 0 depth 129, load C/C store S/D, 100 draws (5 mesh, 0 alpha-tested, 0 depth variant), ended by SetRenderTargets;");
    const char *p3 = strstr(b2, "#3 3.00 ms 640x360 colour 37 0 0 0 depth 0, load L store S, 50 draws (0 mesh, 3 alpha-tested, 2 depth variant), ended by Dispatch;");
    const char *p5 = strstr(b2, "#5 0.10 ms 64x64 colour 37 0 0 0 depth 0, load D store S");
    assert(p1 && p3 && p5 && p1 < p3 && p3 < p5);   // (the costliest first)
    const char *g0 = strstr(b2, "#0 1.00 ms: 2 dispatches (0 spill parts), 0 copies, 0 clears, 3 barriers (2 merged);"), *g2 = strstr(b2, "#2 0.50 ms: 0 dispatches"), *gb = strstr(b2, "between passes, the 5 costliest:");
    assert(g0 && g2 && gb && gb < g0 && g0 < g2);
    puts("prof ok"); return 0;
}
'''


class GpuPass(unittest.TestCase):
    def cc(self):
        cc = shutil.which('cc') or shutil.which('clang')
        if not cc or not (VKH / 'vulkan/vulkan_core.h').exists():
            self.skipTest('a C compiler and the vendored Vulkan headers are needed')
        return cc

    def run_c(self, code, name, args=()):
        with tempfile.TemporaryDirectory(prefix='nx2a_vk_gpuprof_') as tmp:
            c, exe = Path(tmp) / f'{name}.c', Path(tmp) / name
            c.write_text(code, encoding='utf-8')
            r = subprocess.run([self.cc(), '-std=gnu11', '-O1', '-Wall', '-Werror', '-Wno-unused-function', '-Wno-unused-variable', '-I' + str(VKH), str(c), '-o', str(exe)], capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
            r = subprocess.run([str(exe), *args], capture_output=True, text=True, timeout=60)
            self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
            return r.stdout

    def barrier_code(self, extra=''):
        dq, core = DQ.read_text(encoding='utf-8'), CORE.read_text(encoding='utf-8')
        body = region(dq, '// ---- full barriers back to back', '// ---- end of full barriers back to back ----')
        bar = region(core, 'void vk_img_barrier_all(VkCommandBuffer cb) {', 'void vk_gpu_wrote(uint64_t a, uint64_t n);')
        d = body[body.index('#define BT_CMDS(X)'):]; d = d[:d.index('\n', d.index('X(vkBeginCommandBuffer)'))]   # (the macro, its continuation lines)
        names = re.findall(r'X\((vk\w+)\)', d)
        allf = names + STATE + ([extra] if extra else [])
        decls = '\n'.join(f'static PFN_{n} {n};' for n in names + STATE)
        if extra:
            decls += f'\nstatic void *{extra};'
        return ('#include <stdatomic.h>\n#include <stdint.h>\n#include <stdio.h>\n#include <stdlib.h>\n#include <string.h>\n#define VK_NO_PROTOTYPES\n#include <vulkan/vulkan_core.h>\n'
                '#define VKLOG(...) (printf("[vk] " __VA_ARGS__), putchar(10))\n' + decls + '\n#define VK_DEV_FNS(X) ' + ' '.join(f'X({n})' for n in allf) + '\n'
                'static uint64_t g_bar_full, g_bar_merged;\nstatic VkCommandBuffer A = (VkCommandBuffer)(uintptr_t)0x100, B = (VkCommandBuffer)(uintptr_t)0x200;\n'
                + body + bar + BARRIER_MAIN)

    def test_barrier_merge(self):
        out = self.run_c(self.barrier_code(), 'bar')
        self.assertIn('barriers ok', out)
        self.assertIn('dispatch run: 8 barriers recorded, 8 merged', out)

    def test_barrier_merge_unknown_entry_point(self):
        out = self.run_c(self.barrier_code('vkCmdCopyImage2'), 'bar2', args=('unknown',))
        self.assertIn('vkCmdCopyImage2 is not known to the barrier merge', out)
        self.assertIn('unknown ok', out)

    def test_profile_sums(self):
        core, hdr = CORE.read_text(encoding='utf-8'), HDR.read_text(encoding='utf-8')
        prof_t = region(hdr, 'enum { VK_PU_PASS = 1, VK_PU_GAP = 2, VK_PU_END_SUB = 255 };', 'int vk_prof_on(void);')
        body = region(core, '// ---- GPU profile: the sums', "// ---- end of the GPU profile's sums ----")
        code = '#include <assert.h>\n#include <stdint.h>\n#include <stdio.h>\n#include <string.h>\nenum { FPASSQ = 512 };\n' + prof_t + body + PROF_MAIN
        out = self.run_c(code, 'prof')
        self.assertIn('prof ok', out)


if __name__ == '__main__':
    unittest.main()
