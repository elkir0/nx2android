"""An indirect dispatch's group counts in c0 + 0x200 (gfx_vk_exec.c dispatch, DM_VK_IND_C0): NVN code reads its grid size from the driver constants there, and
hle_nvn.c records an indirect dispatch as 0x0x0, so the backend copies the GPU-written counts in before the dispatch, as nvn_mtl.m does. The checks read
gfx_vk_exec.c: stage_set notes the compute stage's c0 segment, dispatch forgets it before stage_set and copies the 12 bytes at the indirect address to its
+ 0x200 between the barrier before the dispatch and the dispatch itself, the live switch is read with the others, and the switches list documents it."""
from pathlib import Path
import re
import unittest

ROOT = Path(__file__).resolve().parents[1]
EXEC = ROOT / 'nxnative/dm/android/gfx_vk_exec.c'
HLE = ROOT / 'nxnative/dm/gfx/nvn/hle_nvn.c'
SWITCHES = ROOT / '.claude/skills/switch-android-port/references/switches.md'


def region(src, start, end):
    a = src.index(start)
    return src[a:src.index(end, a)]


class IndirectDispatchCounts(unittest.TestCase):
    def setUp(self):
        self.src = EXEC.read_text()
        self.disp = region(self.src, 'static void dispatch(const p_dispatch *d) {', '\n}\n')

    def test_hle_records_zero_counts(self):
        # the premise: an indirect dispatch reaches the backend with no counts of its own
        body = region(HLE.read_text(), 'NVN(nvnCommandBufferDispatchComputeIndirect', '\n}')
        self.assertIn('(p_dispatch){0, 0, 0,', body)

    def test_stage_set_notes_the_c0_segment(self):
        body = region(self.src, 'static VkDescriptorSet stage_set(', '\n}\n')
        self.assertRegex(body, r'drv_shared\(st, b->reg, &sb, &so\);[^\n]*if \(st == 5 && b->reg == 0\) \{ g_c0cs_b = sb; g_c0cs_o = so; \}')

    def test_segment_forgotten_before_stage_set(self):
        d = self.disp
        self.assertLess(d.index('g_c0cs_b = VK_NULL_HANDLE;'), d.index('stage_set(5, cs, dofs)'))

    def test_copy_between_barrier_and_dispatch(self):
        d = self.disp
        copy = d.index('VkBufferCopy r = {io, g_c0cs_o + 0x200, 12}; vkCmdCopyBuffer(cb, ib, g_c0cs_b, 1, &r);')
        self.assertIn('if (d->ind && g_ind_c0 && g_c0cs_b)', d[:copy])
        self.assertLess(d.index('vk_img_barrier_all(cb);'), copy)              # after the GPU work that wrote the counts
        after = d[copy:]
        self.assertRegex(after[:after.index('vkCmdBindPipeline')],             # visible to the shader's uniform reads
                         r'VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_UNIFORM_READ_BIT[^;]*;\s*vkCmdPipelineBarrier\(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT')
        self.assertLess(copy, d.index('vkCmdDispatchIndirect(cb, b, o)'))

    def test_live_switch(self):
        self.assertRegex(self.src, r'g_ind_c0 = !\(\(e = getenv\("DM_VK_IND_C0"\)\) && !atoi\(e\)\);')
        self.assertRegex(self.src, r'static int g_ind_c0 = 1;')
        self.assertIn('`DM_VK_IND_C0=0`', SWITCHES.read_text())


if __name__ == '__main__':
    unittest.main()
