"""NVN's discards as Vulkan DONT_CARE loads and stores (gfx_vk_exec.c "discards", DM_VK_DISCARD), built on the host from the backend's own code: the lookahead
from a pass's first draw to the record that ends it for certain (dsc_scan: discards after the last draw, draws and clears after a discard, targets kept, jumps,
calls and the submission's later command handles, records that may cut the pass first, a cut pass's discard found by the part that really ends there), the
walk of the rest of the submission that finds a discarded attachment used again (dsc_memo_*: binds, copies, targets, aliasing ranges, uses before the end,
a walk built once and asked from later ends, a record walked twice, overflow), and the image history that decides across submissions (dsc_hist_*: no store
left out before DSC_EVIDENCE discards each followed by a whole write, never after a read while discarded, depth and stencil apart). The record op itself
(nvn_obj.h OP_DISCARD after the existing ones, hle_nvn.c recording it) and the replay's wiring are checked in the sources."""
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
DM = ROOT / 'nxnative/dm'
EXEC = DM / 'android/gfx_vk_exec.c'
CORE = DM / 'android/gfx_vk_core.c'
HLE = DM / 'gfx/nvn/hle_nvn.c'
OBJ = DM / 'gfx/nvn/nvn_obj.h'
VKH = DM / 'third_party/vulkan-headers/include'
INC = ['-I' + str(DM / 'gfx/nvn'), '-I' + str(DM / 'gfx/nvn/include'), '-I' + str(DM / 'gfx/tex'), '-I' + str(VKH)]


def region(src, start, end):
    a = src.index(start)
    return src[a:src.index(end, a)]


HEAD = r'''
#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include "nvn_obj.h"
_Static_assert(OP_DISCARD == OP_MULTI_DRAW + 1 && OP_COUNT_ == OP_DISCARD + 1, "OP_DISCARD is appended after the existing record ops");
typedef struct vkimg { uint8_t dsc_pend, dsc_drop, dsc_veto, dsc_ok[3]; } vkimg_t;   // (the backend's image: its discard history only)
'''

MAIN = r'''
// ---- synthetic records, laid out as hle_nvn.c records them (op_hdr, payload, 8-byte steps) ----
static uint64_t g_mem[1 << 17]; static size_t g_used;
static void *rec(uint32_t op, size_t payload) {
    size_t sz = 8 + ((payload + 7) & ~(size_t)7); uint8_t *p = (uint8_t *)g_mem + g_used; g_used += sz; assert(g_used <= sizeof g_mem);
    op_hdr h = {op, (uint32_t)sz}; memcpy(p, &h, 8); memset(p + 8, 0, sz - 8); return p + 8;
}
#define REC(op, T) ((T *)rec(op, sizeof(T)))
static const uint8_t *here(void) { return (const uint8_t *)g_mem + g_used; }   // where the next record goes
static void end_(void) { rec(OP_END, 0); }
static void draw(void) { REC(OP_DRAW, p_draw)->count = 3; }
static void discard(int i) { REC(OP_DISCARD, p_discard)->index = i; }
static void set_rt(const tex_t *c0, const tex_t *d) { p_set_rt *r = REC(OP_SET_RT, p_set_rt); r->n = c0 ? 1 : 0; r->color[0] = c0; r->depth = d; }
static void clear_ds(int dm, int sm) { p_clear_ds *k = REC(OP_CLEAR_DS, p_clear_ds); k->depth_mask = dm; k->stencil_mask = sm; }
static void clear_color(int i) { p_clear_color *k = REC(OP_CLEAR_COLOR, p_clear_color); k->index = i; k->mask = 0xf; }
static void op(uint32_t o) { rec(o, 16); }   // a record whose payload the lookahead does not read
static void bind_tex(uint64_t h) { p_bind_tex1 *b = REC(OP_BIND_TEX1, p_bind_tex1); b->stage = 1 | BIND_TEX1_NOBASE; b->h = h; }
static void bind_imgs(uint64_t h0, uint64_t h1) { p_bind_handles *b = REC(OP_BIND_IMG, p_bind_handles); b->stage = 5; b->n = 2; b->h[0] = h0; b->h[1] = h1; }
static void copy_t2t(const tex_t *s, const tex_t *d) { p_copy_t2t *k = REC(OP_COPY_T2T, p_copy_t2t); k->src = s; k->dst = d; }
static void copy_b2b(uint64_t s, uint64_t d, uint64_t n) { uint64_t *k = rec(OP_COPY_B2B, 24); k[0] = s; k[1] = d; k[2] = n; }
static void jump(const uint8_t *to) { void *k = rec(OP_JUMP, 8); memcpy(k, &to, 8); }
static void call(const uint8_t *to) { void *k = rec(OP_CALL, 8); uint64_t t = (uint64_t)(uintptr_t)to; memcpy(k, &t, 8); }

// ---- the replay's side: textures, the pass's targets, handles ----
static tex_t T[6];   // T[0] the depth, T[1] colour A, T[2] colour B, T[3] another texture, T[4] a texture over part of T[0]'s storage, T[5] another depth
static const tex_t *g_c0, *g_d;   // the open pass's targets
static int same_rt(const p_set_rt *r) { return r->n == (g_c0 ? 1 : 0) && r->color[0] == g_c0 && r->depth == g_d; }
static int handle_range(uint64_t h, uintptr_t *lo, uintptr_t *hi) {
    int tid = (int)(h & 0xfffff);
    if (tid >= 1 && tid <= 6) return dsc_tex_range(&T[tid - 1], lo, hi);
    if (tid >= 1000) { *lo = 0x70000000u + (uintptr_t)tid * 0x1000; *hi = *lo + 0x100; return 1; }
    return 0;
}
static dsc_ctx X = {same_rt, handle_range, 512, 100000};
static uint32_t scan_from(const uint8_t *s, const uint64_t *rest, int nrest, const uint8_t **end, dsc_cur *at, int *why) { dsc_cur c = {s, {NULL}, 0, rest, nrest}; return dsc_scan(c, &X, end, at, why); }
static uint32_t scan(const uint8_t *s, const uint8_t **end, int *why) { dsc_cur at; return scan_from(s, NULL, 0, end, &at, why); }
static dsc_memo M;
static int used_after(const uint8_t *s, const tex_t *t) {   // the pass that begins at s, its end, the memo built from there, then: t used again from that end on
    const uint8_t *e; dsc_cur at; int why; uint32_t b = scan_from(s, NULL, 0, &e, &at, &why); assert(b && why == DSC_SURE && e);
    dsc_memo_build(&M, at, &X); uintptr_t lo, hi; assert(dsc_tex_range(t, &lo, &hi)); return dsc_memo_used(&M, e, lo, hi);
}
#define CHECK(c) do { if (!(c)) { printf("failed at line %d: %s\n", __LINE__, #c); return 1; } } while (0)
int main(void) {
    for (int i = 0; i < 6; i++) { T[i].magic = MAGIC('T', 'E', 'X', 0); T[i].mem = (uint8_t *)(uintptr_t)(0x10000000u + 0x100000u * (unsigned)i); T[i].L.size = 0x80000; }
    T[4].mem = T[0].mem + 0x1000; T[4].L.size = 0x1000;
    const uint8_t *s, *e, *e2; int why; uint32_t b;
    g_c0 = &T[1]; g_d = &T[0];

    // a discard after the last draw: the depth and the stencil, the record that ends the pass, certain
    s = here(); draw(); draw(); discard(-1); e = here(); set_rt(&T[2], NULL); end_();
    b = scan(s, &e2, &why); CHECK(b == (DSC_D | DSC_S) && e2 == e && why == DSC_SURE);
    // drawn after it: nothing
    s = here(); draw(); discard(-1); draw(); set_rt(&T[2], NULL); end_();
    b = scan(s, &e2, &why); CHECK(b == 0 && why == DSC_SURE);
    // cleared after it: the depth cleared keeps the stencil's, and the other way
    s = here(); draw(); discard(-1); clear_ds(1, 0); set_rt(&T[2], NULL); end_(); CHECK(scan(s, &e2, &why) == DSC_S);
    s = here(); draw(); discard(-1); clear_ds(0, 1); set_rt(&T[2], NULL); end_(); CHECK(scan(s, &e2, &why) == DSC_D);
    // colour targets by index; a colour clear after one takes only its own back
    s = here(); draw(); discard(0); discard(2); clear_color(2); discard(5); op(OP_BARRIER); end_();
    b = scan(s, &e2, &why); CHECK(b == (1u | 1u << 5) && why == DSC_SURE);
    // the same targets set again: the pass goes on (a discard before them counts only if no draw follows)
    s = here(); draw(); discard(-1); set_rt(g_c0, g_d); draw(); set_rt(&T[2], NULL); end_(); CHECK(scan(s, &e2, &why) == 0);
    s = here(); draw(); set_rt(g_c0, g_d); discard(-1); e = here(); op(OP_BARRIER); end_();
    b = scan(s, &e2, &why); CHECK(b == (DSC_D | DSC_S) && e2 == e);
    // every pass end the replay makes: dispatches, copies, clears of textures and buffers, barriers
    const uint32_t ends[] = {OP_DISPATCH, OP_COPY_B2B, OP_COPY_B2T, OP_COPY_T2B, OP_COPY_T2T, OP_CLEAR_TEX, OP_CLEAR_BUF, OP_BARRIER};
    for (unsigned k = 0; k < sizeof ends / sizeof *ends; k++) { s = here(); draw(); discard(-1); e = here(); op(ends[k]); draw(); end_();
        b = scan(s, &e2, &why); CHECK(b == (DSC_D | DSC_S) && e2 == e && why == DSC_SURE); }
    // state, binds, counters between: no end
    s = here(); draw(); discard(-1); op(OP_BIND_PROGRAM); bind_tex(4); op(OP_SCISSOR); op(OP_REPORT_COUNTER); draw(); op(OP_BARRIER); end_(); CHECK(scan(s, &e2, &why) == 0);

    // the submission's end: the pass ends there (no end record), also when the discard is in a later command handle of the submission
    s = here(); draw(); discard(-1); end_();
    b = scan(s, &e2, &why); CHECK(b == (DSC_D | DSC_S) && e2 == NULL && why == DSC_SURE);
    const uint8_t *h1 = here(); draw(); end_(); const uint8_t *h2 = here(); discard(-1); end_(); const uint8_t *h3 = here(); draw(); end_();
    uint64_t rest[2] = {(uint64_t)(uintptr_t)h2, (uint64_t)(uintptr_t)h3}; dsc_cur at;
    b = scan_from(h1, rest, 1, &e2, &at, &why); CHECK(b == (DSC_D | DSC_S) && e2 == NULL);
    b = scan_from(h1, rest, 2, &e2, &at, &why); CHECK(b == 0);   // (the third handle draws again)
    // jumps and calls followed as the replay follows them, also from inside a call
    const uint8_t *sub = here(); discard(-1); end_();
    s = here(); draw(); call(sub); e = here(); set_rt(&T[2], NULL); end_();
    b = scan(s, &e2, &why); CHECK(b == (DSC_D | DSC_S) && e2 == e);
    const uint8_t *chunk = here(); discard(-1); e = here(); op(OP_BARRIER); end_();
    s = here(); draw(); jump(chunk);
    b = scan(s, &e2, &why); CHECK(b == (DSC_D | DSC_S) && e2 == e);
    const uint8_t *sub2 = here(); draw(); end_();
    s = here(); call(sub2); const uint8_t *ret = here(); discard(-1); e = here(); op(OP_DISPATCH); end_();
    { dsc_cur c = {sub2, {ret}, 1, NULL, 0}; b = dsc_scan(c, &X, &e2, &at, &why); CHECK(b == (DSC_D | DSC_S) && e2 == e); }   // (the replay inside sub2's draw)

    // cut passes: a record that may cut the pass first makes its end uncertain (the bits still counted); the part that really ends there finds the discard
    s = here(); draw(); REC(OP_RAST_DISCARD, p_rast_discard)->on = 1; draw(); REC(OP_RAST_DISCARD, p_rast_discard)->on = 0; const uint8_t *s1 = here(); draw(); discard(-1); e = here(); set_rt(&T[2], NULL); end_();
    b = scan(s, &e2, &why); CHECK(b == (DSC_D | DSC_S) && why == DSC_UNSURE_OP);
    b = scan(s1, &e2, &why); CHECK(b == (DSC_D | DSC_S) && why == DSC_SURE && e2 == e);
    s = here(); draw(); REC(OP_TF_STATE, p_tf_state)->what = 1; discard(-1); op(OP_BARRIER); end_(); scan(s, &e2, &why); CHECK(why == DSC_UNSURE_OP);
    // uniform updates past the deferred list's room: it lands them, a pass end
    X.ubd_room = 2;
    s = here(); draw(); op(OP_UPDATE_UB); op(OP_UPDATE_UB); draw(); discard(-1); op(OP_BARRIER); end_(); scan(s, &e2, &why); CHECK(why == DSC_SURE);
    s = here(); draw(); op(OP_UPDATE_UB); op(OP_UPDATE_UB); op(OP_UPDATE_UB); draw(); discard(-1); op(OP_BARRIER); end_(); b = scan(s, &e2, &why); CHECK(why == DSC_UNSURE_UBD && b == (DSC_D | DSC_S));
    X.ubd_room = 512;
    // too long, a call stack too deep, or the certain end too far past an uncertain record: no answer
    X.max_ops = 3; s = here(); draw(); draw(); draw(); discard(-1); op(OP_BARRIER); end_(); b = scan(s, &e2, &why); CHECK(b == 0 && why == DSC_UNSURE_LONG); X.max_ops = 100000;
    { const uint8_t *d = here(); discard(-1); end_(); for (int k = 0; k < 5; k++) { const uint8_t *n = here(); call(d); end_(); d = n; }
      b = scan(d, &e2, &why); CHECK(b == 0 && why == DSC_UNSURE_LONG); }
    s = here(); draw(); op(OP_TF_STATE); for (int k = 0; k < DSC_PAST + 8; k++) op(OP_SCISSOR); discard(-1); op(OP_BARRIER); end_();
    b = scan(s, &e2, &why); CHECK(b == 0 && why == DSC_UNSURE_OP);

    // used again in the submission: bound as a texture or an image, a copy's source, a buffer copy over its memory, set as a target again, an aliasing texture
    s = here(); draw(); discard(-1); set_rt(&T[2], NULL); draw(); bind_tex(1); draw(); end_(); CHECK(used_after(s, &T[0]) == 1);
    s = here(); draw(); discard(-1); set_rt(&T[2], NULL); draw(); bind_imgs(1004, 1); end_(); CHECK(used_after(s, &T[0]) == 1);
    s = here(); draw(); discard(-1); set_rt(&T[2], NULL); draw(); copy_t2t(&T[0], &T[3]); end_(); CHECK(used_after(s, &T[0]) == 1);
    s = here(); draw(); discard(-1); set_rt(&T[2], NULL); copy_b2b((uint64_t)(uintptr_t)T[0].mem + 0x100, 0x50000000u, 64); end_(); CHECK(used_after(s, &T[0]) == 1);
    s = here(); draw(); discard(-1); set_rt(&T[2], NULL); draw(); set_rt(&T[1], &T[0]); draw(); end_(); CHECK(used_after(s, &T[0]) == 1);
    s = here(); draw(); discard(-1); set_rt(&T[2], NULL); draw(); bind_tex(5); draw(); end_(); CHECK(used_after(s, &T[0]) == 1);   // (T[4] lies inside T[0])
    s = here(); draw(); discard(-1); set_rt(&T[2], &T[0]); end_(); CHECK(used_after(s, &T[0]) == 1);   // (the end record itself binds it again)
    // not used again: other textures, or uses before the end
    s = here(); draw(); discard(-1); set_rt(&T[2], NULL); draw(); bind_tex(4); bind_imgs(1004, 1005); copy_t2t(&T[3], &T[2]); draw(); end_(); CHECK(used_after(s, &T[0]) == 0);
    s = here(); draw(); bind_tex(1); draw(); discard(-1); set_rt(&T[2], NULL); draw(); end_(); CHECK(used_after(s, &T[0]) == 0);
    // the walk built at the first end and asked from a later one: only uses from that end on count
    s = here(); draw(); discard(-1); e = here(); set_rt(&T[1], &T[5]); draw(); bind_tex(1); draw(); discard(-1); e2 = here(); op(OP_BARRIER); draw(); end_();
    { const uint8_t *e3; dsc_cur a2; b = scan_from(s, NULL, 0, &e3, &a2, &why); CHECK(e3 == e); dsc_memo_build(&M, a2, &X); uintptr_t lo, hi;
      dsc_tex_range(&T[0], &lo, &hi); CHECK(dsc_memo_used(&M, e, lo, hi) == 1 && dsc_memo_used(&M, e2, lo, hi) == 0);
      dsc_tex_range(&T[5], &lo, &hi); CHECK(dsc_memo_used(&M, e2, lo, hi) == 0);
      CHECK(dsc_memo_used(&M, here(), lo, hi) == 1); }   // (an end the walk never met: no answer)
    // a record walked twice (a command buffer called twice in the submission): no answer for its end
    { const uint8_t *body = here(); draw(); discard(-1); const uint8_t *eb = here(); set_rt(&T[2], NULL); end_();
      const uint8_t *m0 = here(); call(body); call(body); end_(); dsc_cur c = {m0, {NULL}, 0, NULL, 0}; dsc_memo_build(&M, c, &X);
      uintptr_t lo, hi; dsc_tex_range(&T[3], &lo, &hi); CHECK(dsc_memo_used(&M, eb, lo, hi) == 1); }
    // a walk too big to keep: every answer "used"
    { s = here(); draw(); discard(-1); set_rt(&T[2], NULL); for (int k = 0; k < DSC_NUSE; k++) bind_tex(1000 + (uint64_t)k); end_(); CHECK(used_after(s, &T[0]) == 1 && M.full); }

    // the image's history across submissions: no store left out before DSC_EVIDENCE discards each followed by a whole write before any read
    vkimg_t im; memset(&im, 0, sizeof im);
    for (int k = 0; k < DSC_EVIDENCE; k++) { CHECK(!dsc_hist_ok(&im, VK_DSC_DEPTH)); dsc_hist_end(&im, VK_DSC_DEPTH, 0); CHECK(dsc_hist_use(&im, VK_DSC_DEPTH, 0) == 0); }
    CHECK(dsc_hist_ok(&im, VK_DSC_DEPTH) && !dsc_hist_ok(&im, VK_DSC_STENCIL) && !dsc_hist_ok(&im, VK_DSC_COLOR));   // (depth and stencil apart)
    // a read while discarded, after a store was left out: reported, and never again
    dsc_hist_end(&im, VK_DSC_DEPTH, 1); CHECK(im.dsc_pend == VK_DSC_DEPTH && im.dsc_drop == VK_DSC_DEPTH);
    CHECK(dsc_hist_read(&im) == 1 && !dsc_hist_ok(&im, VK_DSC_DEPTH) && !im.dsc_pend);
    for (int k = 0; k < 10; k++) { dsc_hist_end(&im, VK_DSC_DEPTH, 0); dsc_hist_use(&im, VK_DSC_DEPTH, 0); } CHECK(!dsc_hist_ok(&im, VK_DSC_DEPTH));
    // loaded by the next pass while discarded: the same; a load with nothing pending changes nothing; a read only vetoes the aspects discarded
    vkimg_t j; memset(&j, 0, sizeof j); j.dsc_ok[1] = j.dsc_ok[2] = DSC_EVIDENCE;
    CHECK(dsc_hist_use(&j, VK_DSC_DEPTH, 1) == 0 && dsc_hist_ok(&j, VK_DSC_DEPTH));
    dsc_hist_end(&j, VK_DSC_STENCIL, 1); CHECK(dsc_hist_use(&j, VK_DSC_STENCIL, 1) == 1 && !dsc_hist_ok(&j, VK_DSC_STENCIL) && dsc_hist_ok(&j, VK_DSC_DEPTH));
    dsc_hist_end(&j, VK_DSC_DEPTH, 0); CHECK(dsc_hist_read(&j) == 0 && !dsc_hist_ok(&j, VK_DSC_DEPTH));   // (kept, so not reported: a read during the evidence)
    puts("discards ok"); return 0;
}
'''


class Discards(unittest.TestCase):
    def test_lookahead_memo_history(self):
        cc = shutil.which('cc') or shutil.which('clang')
        if not cc:
            self.skipTest('a C compiler is needed')
        src = EXEC.read_text(encoding='utf-8')
        body = region(src, '// ---- discards: the lookahead', "// ---- end of the discards' lookahead ----")
        with tempfile.TemporaryDirectory(prefix='nx2a_vk_discard_') as tmp:
            c, exe = Path(tmp) / 'dsc.c', Path(tmp) / 'dsc'
            c.write_text(HEAD + body + MAIN, encoding='utf-8')
            r = subprocess.run([cc, '-std=gnu11', '-O1', '-g', '-Wall', '-Werror', '-Wno-unused-function', '-Wno-unused-variable', '-fsanitize=address,undefined', *INC, str(c), '-o', str(exe)],
                               capture_output=True, text=True)
            if r.returncode and 'sanitize' in r.stderr:   # (a compiler without the sanitizers)
                r = subprocess.run([cc, '-std=gnu11', '-O1', '-Wall', '-Werror', '-Wno-unused-function', '-Wno-unused-variable', *INC, str(c), '-o', str(exe)], capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
            r = subprocess.run([str(exe)], capture_output=True, text=True, timeout=120)
            self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
            self.assertIn('discards ok', r.stdout)

    def test_record_and_wiring(self):
        obj, hle, ex, core = (p.read_text(encoding='utf-8') for p in (OBJ, HLE, EXEC, CORE))
        enum = re.search(r'enum\s*\{([^}]*OP_COUNT_[^}]*)\}', obj).group(1)
        names = [n.strip() for n in enum.split(',') if n.strip()]
        names = [re.sub(r'//.*', '', n).strip() for n in names]
        self.assertEqual(names[-2:], ['OP_DISCARD', 'OP_COUNT_'])   # appended after the existing ones: the others keep their numbers
        self.assertRegex(obj, r'typedef struct \{ int32_t index, pad; \} p_discard;')
        for fn, idx in (('nvnCommandBufferDiscardColor', r'= i;'), ('nvnCommandBufferDiscardDepthStencil', r'= -1;')):
            line = next(l for l in hle.splitlines() if l.startswith(f'NVN({fn},'))
            self.assertIn('REC(c, OP_DISCARD, p_discard)->index ' + idx, line)
        self.assertIn('case OP_DISCARD: disc_op((const p_discard *)a); break;', ex)   # (replayed, not the unknown-op [RED])
        self.assertIn('(VkAttachmentStoreOp)k->cstore[i]', ex)
        self.assertIn('(VkAttachmentStoreOp)k->sstore', ex)
        self.assertIn('g_disc = (e = getenv("DM_VK_DISCARD")) ? atoi(e) : 1;', ex)   # (live: x_read, each submission; default 1)
        self.assertEqual(len(re.findall(r'^void mtl_exec_rest\(', ex + core, re.M)), 1)   # (the Vulkan backend's lookahead keeps the handles left)
        self.assertIn('PC_DSTORE, PC_DLOAD, PC_DK_END, PC_DK_USED, PC_DK_HIST', core)   # (each decision counted in the GPU profile)


if __name__ == '__main__':
    unittest.main()
