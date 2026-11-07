"""Two live diagnostics for GPU-culled indirect draws (gfx_vk_exec.c / gfx_vk_core.c): DM_VK_X_RECCAP captures on the GPU the indirect-draw records chosen vertex
shaders read in each render pass (a CPU read at replay is worthless: the records are GPU-written that frame), and DM_VK_X_WRITERS logs which tracked GPU work
writes a watched guest address range. The checks read the sources: both switches are parsed in x_read, the capture copies happen in vk_end_pass after the render
pass ends with a HOST_READ barrier after them, the completion thread handles the capture post-op, vk_gpu_wrote checks the watched range, and switches.md
documents both."""
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
EXEC = ROOT / 'nxnative/dm/android/gfx_vk_exec.c'
CORE = ROOT / 'nxnative/dm/android/gfx_vk_core.c'
SWITCHES = ROOT / '.claude/skills/switch-android-port/references/switches.md'


def region(src, start, end):
    a = src.index(start)
    return src[a:src.index(end, a)]


class Reccap(unittest.TestCase):
    def setUp(self):
        self.exec = EXEC.read_text()
        self.core = CORE.read_text()

    def test_switch_parsed_in_x_read(self):
        body = region(self.exec, 'static void x_read(void) {', '\n}\n')
        self.assertIn('getenv("DM_VK_X_RECCAP")', body)
        self.assertIn('getenv("DM_VK_X_WRITERS")', body)

    def test_capture_copy_at_pass_end(self):
        body = region(self.exec, 'void vk_end_pass(void) {', '\n}\n')
        ended = body.index('vkCmdEndRenderPass')   # the render pass is ended first, then the records copied outside it
        copy = body.index('reccap_copy()')
        self.assertLess(ended, copy)
        self.assertIn('S.pass_open == 1', body[:copy])

    def test_host_read_barrier_after_copy(self):
        body = region(self.exec, 'static void reccap_copy(void) {', '\n}\n')
        copy = body.index('vkCmdCopyBuffer')
        barrier = body.index('VK_ACCESS_HOST_READ_BIT')
        self.assertLess(copy, barrier)
        self.assertRegex(body[barrier:], r'VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT')

    def test_completion_thread_handles_the_capture(self):
        body = region(self.core, 'static void *completion_main(void *a) {', '\n    return NULL;')
        self.assertIn('o->kind == 3', body)
        self.assertIn('reccap: present %llu pass %d vs %016llx record %#llx: count %u instances %u first index %u vertex offset %d first instance %u', body)

    def test_records_noted_where_drawn(self):
        d = region(self.exec, 'static void draw_idx(const p_draw_idx *d) {', '\n}\n')
        line = d.index('vkCmdDrawIndexedIndirect(cb, b, o, 1, 20)')
        self.assertLess(line, d.index('reccap_note('))          # noted only where the draw is recorded
        self.assertIn('d->ind, 1, 20', d)                       # one 20-byte record at the draw's indirect address
        m = region(self.exec, 'static void multi_draw(int prim, int itype, uint64_t ibuf, uint64_t ind, uint64_t cnt, uint32_t first, uint32_t n, uint32_t stride) {', '\n}\n')
        self.assertIn('reccap_note(ind, n, stride, rs, cnt)', m)

    def test_gpu_wrote_checks_watched_range(self):
        body = region(self.core, 'void vk_gpu_wrote(uint64_t a, uint64_t n) {', '\n}\n')
        self.assertIn('a < g_x_watch_hi && g_x_watch_lo < a + n', body)
        self.assertIn('writers: %s writes %#llx +%llu over the watched %#llx +%llu (present %llu, pass %d)', body)

    def test_switches_doc(self):
        doc = SWITCHES.read_text()
        self.assertRegex(doc, r'`DM_VK_X_RECCAP=<vs key hex>\[,<key>...\]\[#<presents>\]`')
        self.assertRegex(doc, r'`DM_VK_X_WRITERS=<guest address hex>\[\+<length hex>\]`')


if __name__ == '__main__':
    unittest.main()
