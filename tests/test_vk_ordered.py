"""Ordered allocations (gfx_vk_exec.c dispatch / gfx_vk_shader.c): a compute shader the bundle marks as allocating
through a storage-buffer atomic (reflection section "ORDR", shader_vk.py ordered_alloc) runs a small direct dispatch's
workgroups in index order, through the same serial_dispatch as DM_VK_X_CSSERIAL, emulating NVIDIA's in-order workgroup
start. The checks read the sources: both switches are parsed in x_read, the rule is gated on a direct dispatch and on
the workgroup maximum before ser feeds the pipeline key, and the reader takes ORDR after the FSTR section."""
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
EXEC = ROOT / 'nxnative/dm/android/gfx_vk_exec.c'
SHADER = ROOT / 'nxnative/dm/android/gfx_vk_shader.c'
SWITCHES = ROOT / '.claude/skills/switch-android-port/references/switches.md'


def region(src, start, end):
    a = src.index(start)
    return src[a:src.index(end, a)]


class Ordered(unittest.TestCase):
    def setUp(self):
        self.exec = EXEC.read_text()
        self.shader = SHADER.read_text()

    def test_switches_parsed_in_x_read(self):
        body = region(self.exec, 'static void x_read(void) {', '\n}\n')
        self.assertIn('getenv("DM_VK_ORDERED")', body)
        self.assertIn('getenv("DM_VK_ORDERED_MAX")', body)

    def test_rule_gated_in_dispatch(self):
        d = region(self.exec, 'static void dispatch(const p_dispatch *d) {', '\n}\n')
        gate = d[d.index('int ord ='):].split(';')[0]
        for part in ('cs->ordered', 'g_ordered', '!d->ind', 'V.has_dbase', 'wn >= 2', 'wn <= g_ordered_max', '!spill_slices(cs)'): self.assertIn(part, gate)   # (spilled: spill_dispatch's parts)
        self.assertIn('|| ord;', d)   # ser feeds the pipeline key: the ordered pipeline takes DISPATCH_BASE
        ind = d.index('if (d->ind)')
        self.assertIn('ord', d[:ind])   # the gate is computed before the dispatch itself, from !d->ind

    def test_the_indirect_path_cannot_pick_it_up(self):
        d = region(self.exec, 'static void dispatch(const p_dispatch *d) {', '\n}\n')
        gate = d.index('int ord =')
        self.assertIn('!d->ind', d[gate:gate + 200])   # an indirect dispatch's counts are on the GPU: left unordered

    def test_shader_reader_takes_ordr_after_fstr(self):
        body = region(self.shader, 'vkshader_t *vk_shader(uint64_t key, int stage) {', '\n}\n')
        fstr = body.index('"FSTR"')
        ordr = body.index('"ORDR"')
        walk = body.index('at += 8 + 12 * n')   # FSTR's stores walked past before ORDR is tried
        self.assertLess(fstr, walk)
        self.assertIn('e->refl_len >= at + 8 && !memcmp(r + at, "ORDR"', body)   # the tag bounds-checked against refl_len, as FSTR's
        self.assertIn('s->ordered = n == 1', body)                                # the flag set where the tag is

    def test_switches_doc(self):
        doc = SWITCHES.read_text()
        self.assertRegex(doc, r'`DM_VK_ORDERED=0`')
        self.assertRegex(doc, r'`DM_VK_ORDERED_MAX=<n>`')


if __name__ == '__main__':
    unittest.main()
