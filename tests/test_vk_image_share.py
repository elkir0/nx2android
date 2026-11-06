"""One Vulkan image for NVN textures over the same storage that differ only by IMAGE (gfx_vk_tex.c "image sharing", DM_VK_IMAGE_SHARE): the decision a
texture that needs storage use makes about the image it found (use it, upgrade it, or an image of its own as before), the reasons a [RED] line names, the
start-up switch, and the upgrade itself (img_upgrade) with its Vulkan calls stubbed: the new image's create info (the old one's plus storage use, its format
list), its memory, the copy of every level and layer in the upload command buffer between barriers, the old image's views dropped and its image and memory
freed once the GPU is past, and the handles swapped into the same vkimg_t. The regions are gfx_vk_tex.c's own, built on the host."""
from pathlib import Path
import os
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
DM = ROOT / 'nxnative/dm'
TEX = DM / 'android/gfx_vk_tex.c'
VKH = DM / 'third_party/vulkan-headers/include'
SWITCHES = ROOT / '.claude/skills/switch-android-port/references/switches.md'

REUSE, UPGRADE, SEPARATE = 0, 1, 2
STORAGE = 0x8   # VK_IMAGE_USAGE_STORAGE_BIT


def region(src, start, end):
    a = src.index(start)
    return src[a:src.index(end, a)]


def cut(src, start, end_marker):
    """The source from start up to and including the first end_marker after it."""
    a = src.index(start)
    return src[a:src.index(end_marker, a) + len(end_marker)]


HEAD = '#undef NDEBUG\n#include <assert.h>\n#include <stdint.h>\n#include <stdio.h>\n#include <stdlib.h>\n#include <string.h>\n#define VK_NO_PROTOTYPES\n#include <vulkan/vulkan_core.h>\n'

DECIDE_MAIN = r'''
int main(int argc, char **argv) {
    if (argc > 1 && !strcmp(argv[1], "share")) { printf("%d\n", share_on()); return 0; }   // DM_VK_IMAGE_SHARE as the process was started with
    if (argc > 1 && !strcmp(argv[1], "text")) {   // shx_text of a mask into a buffer of n bytes: the text, then a line with its length
        unsigned x = (unsigned)strtoul(argv[2], NULL, 0); size_t n = (size_t)strtoul(argv[3], NULL, 0); char buf[1024]; memset(buf, 'Z', sizeof buf);
        shx_text(x, buf, n); if (n) { assert(memchr(buf, 0, n)); printf("%s\n%zu\n", buf, strlen(buf)); } else assert(buf[0] == 'Z');
        for (size_t i = n; i < sizeof buf; i++) assert(buf[i] == 'Z');   // nothing written past the buffer
        return 0;
    }
    const VkImageUsageFlags haves[] = {0, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
                                       VK_IMAGE_USAGE_STORAGE_BIT, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT};
    const int excls[] = {0, SHX_BUSY, SHX_SPARSE | SHX_TWIN, -1};
    for (unsigned h = 0; h < sizeof haves / sizeof *haves; h++) for (int need = 0; need < 2; need++) for (int fok = 0; fok < 2; fok++) for (unsigned e = 0; e < sizeof excls / sizeof *excls; e++)
        printf("%u %d %d %d %d\n", (unsigned)haves[h], need, fok, excls[e], img_share_decide(haves[h], need, fok, excls[e]));
    assert(SHARE_REUSE == 0 && SHARE_UPGRADE == 1 && SHARE_SEPARATE == 2);
    return 0;
}
'''

# The parts of gfx_vk.h, gfx_vk_core.c and gfx_vk_tex.c that img_upgrade calls, stubbed: each call is logged in order
UPGRADE_STUBS = r'''
typedef struct vkimg { VkImageUsageFlags usage; VkImageCreateFlags cflags; VkImage img; VkDeviceMemory mem; void *sub; VkFormat fmt; VkImageAspectFlags aspect;
                       uint32_t w, h, d, layers, levels, samples; int target; VkImageLayout layout; int fixed_fmt; VkFormat fmt2; } vkimg_t;
typedef struct { struct { int format; } d; int64_t off; } tex_t;
static struct { VkDevice dev; } V;
static int g_nlog, g_nred;
#define VKLOG(...) (g_nlog++, printf(__VA_ARGS__), printf("\n"))
#define VKRED_ONCE(id, ...) (g_nred++)
#define VKCHK(x) do { if ((x) < 0) abort(); } while (0)
enum { C_CREATE, C_REQ, C_MEM, C_BIND, C_UPCMD, C_BARALL, C_PBAR, C_COPY, C_DROPV, C_FREESUB, C_FREEMEM, C_DESTROY };
static struct { int k; uint64_t a, b; VkImageCopy c; VkImageMemoryBarrier ib; VkImageCreateInfo ci; uint32_t nfl; VkFormat fl[2]; } L[64]; static int NL;
static VkResult g_create_r; static int g_mem_fail;
#define H(x) ((uint64_t)(uintptr_t)(x))
#define NEW_IMG 0xbeef0
#define NEW_MEM 0x3e30
#define NEW_SUB 0x5ab0
#define CB 0xc0b0
static VkResult vkCreateImage(VkDevice d, const VkImageCreateInfo *ci, const VkAllocationCallbacks *a, VkImage *out) {
    (void)d; (void)a; L[NL].k = C_CREATE; L[NL].ci = *ci; L[NL].nfl = 0;
    if (ci->pNext) { const VkImageFormatListCreateInfo *f = ci->pNext; assert(f->sType == VK_STRUCTURE_TYPE_IMAGE_FORMAT_LIST_CREATE_INFO && f->viewFormatCount <= 2 && !f->pNext);
                     L[NL].nfl = f->viewFormatCount; memcpy(L[NL].fl, f->pViewFormats, f->viewFormatCount * sizeof(VkFormat)); }
    NL++; if (g_create_r) return g_create_r; *out = (VkImage)(uintptr_t)NEW_IMG; return VK_SUCCESS; }
static void vkGetImageMemoryRequirements(VkDevice d, VkImage i, VkMemoryRequirements *r) { (void)d; L[NL].k = C_REQ; L[NL++].a = H(i); r->size = 4096; r->alignment = 256; r->memoryTypeBits = 1; }
static uint32_t vk_mem_type(uint32_t bits, VkMemoryPropertyFlags want, VkMemoryPropertyFlags avoid) {   // (img_new's choice)
    assert(bits == 1 && want == VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT && avoid == VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT); return 7; }
static VkDeviceMemory vk_image_memory(const VkMemoryRequirements *req, uint32_t type, VkDeviceSize *off, void **sub) {
    assert(type == 7 && req->size == 4096); L[NL].k = C_MEM; L[NL++].a = req->size;
    if (g_mem_fail) return VK_NULL_HANDLE; *off = 512; *sub = (void *)(uintptr_t)NEW_SUB; return (VkDeviceMemory)(uintptr_t)NEW_MEM; }
static VkResult vkBindImageMemory(VkDevice d, VkImage i, VkDeviceMemory m, VkDeviceSize o) { (void)d; assert(H(m) == NEW_MEM); L[NL].k = C_BIND; L[NL].a = H(i); L[NL++].b = o; return VK_SUCCESS; }
static void vkDestroyImage(VkDevice d, VkImage i, const VkAllocationCallbacks *a) { (void)d; (void)a; L[NL].k = C_DESTROY; L[NL++].a = H(i); }
static VkCommandBuffer vk_upload_cmd(void) { L[NL++].k = C_UPCMD; return (VkCommandBuffer)(uintptr_t)CB; }
static void vk_img_barrier_all(VkCommandBuffer cb) { L[NL].k = C_BARALL; L[NL++].a = H(cb); }
static void vkCmdPipelineBarrier(VkCommandBuffer cb, VkPipelineStageFlags s, VkPipelineStageFlags d, VkDependencyFlags f, uint32_t nm, const VkMemoryBarrier *m, uint32_t nb,
                                 const VkBufferMemoryBarrier *b, uint32_t ni, const VkImageMemoryBarrier *ib) {
    (void)f; (void)m; (void)b; assert(H(cb) == CB && s == VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT && d == VK_PIPELINE_STAGE_ALL_COMMANDS_BIT && !nm && !nb && ni == 1);
    L[NL].k = C_PBAR; L[NL++].ib = ib[0]; }
static void vkCmdCopyImage(VkCommandBuffer cb, VkImage s, VkImageLayout sl, VkImage d, VkImageLayout dl, uint32_t n, const VkImageCopy *r) {
    assert(H(cb) == CB && sl == VK_IMAGE_LAYOUT_GENERAL && dl == VK_IMAGE_LAYOUT_GENERAL && n == 1); L[NL].k = C_COPY; L[NL].a = H(s); L[NL].b = H(d); L[NL++].c = r[0]; }
static void vk_tex_view_drop_all(vkimg_t *im) { L[NL].k = C_DROPV; L[NL++].a = H(im); }
static void vk_defer_free_image_sub(VkImage i, void *sub) { L[NL].k = C_FREESUB; L[NL].a = H(i); L[NL++].b = H(sub); }
static void vk_defer_free_image(VkImage i, VkDeviceMemory m) { L[NL].k = C_FREEMEM; L[NL].a = H(i); L[NL++].b = H(m); }
'''

UPGRADE_MAIN = r'''
#define OLD_IMG 0x01d0
#define OLD_MEM 0x01e0
#define OLD_SUB 0x01f0
static const VkImageUsageFlags U0 = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
static vkimg_t mk(int target, uint32_t w, uint32_t h, uint32_t d, uint32_t layers, uint32_t levels) {
    vkimg_t im; memset(&im, 0, sizeof im); im.target = target; im.w = w; im.h = h; im.d = d; im.layers = layers; im.levels = levels; im.samples = 1;
    im.fmt = VK_FORMAT_R8G8B8A8_UNORM; im.aspect = VK_IMAGE_ASPECT_COLOR_BIT; im.usage = U0; im.cflags = VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT | VK_IMAGE_CREATE_EXTENDED_USAGE_BIT;
    im.layout = VK_IMAGE_LAYOUT_GENERAL; im.img = (VkImage)(uintptr_t)OLD_IMG; im.mem = (VkDeviceMemory)(uintptr_t)OLD_MEM; im.sub = (void *)(uintptr_t)OLD_SUB; return im;
}
static uint32_t lv(uint32_t x, uint32_t l) { return x >> l ? x >> l : 1; }
// One upgrade that succeeds: the calls in order, then the image's fields
static void ok(vkimg_t *im, VkImageType type, int copies) {
    vkimg_t was = *im; tex_t t = {{0x25}, 0x1000}; NL = 0;
    int r = img_upgrade(im, &t); assert(r == 1);
    int i = 0;
    const VkImageCreateInfo *ci = &L[i].ci; assert(L[i].k == C_CREATE); i++;   // the old create info plus storage use
    assert(ci->sType == VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO && ci->flags == was.cflags && ci->imageType == type && ci->format == was.fmt);
    assert(ci->extent.width == was.w && ci->extent.height == was.h && ci->extent.depth == was.d && ci->mipLevels == was.levels && ci->arrayLayers == was.layers);
    assert(ci->samples == (VkSampleCountFlagBits)was.samples && ci->tiling == VK_IMAGE_TILING_OPTIMAL && ci->usage == (was.usage | VK_IMAGE_USAGE_STORAGE_BIT));
    assert(ci->sharingMode == VK_SHARING_MODE_EXCLUSIVE && ci->initialLayout == VK_IMAGE_LAYOUT_UNDEFINED);
    if (was.fixed_fmt) { assert(L[0].nfl == (was.fmt2 ? 2u : 1u) && L[0].fl[0] == was.fmt && (!was.fmt2 || L[0].fl[1] == was.fmt2)); } else assert(!L[0].nfl);
    assert(L[i].k == C_REQ && L[i].a == NEW_IMG); i++;
    assert(L[i].k == C_MEM); i++;
    assert(L[i].k == C_BIND && L[i].a == NEW_IMG && L[i].b == 512); i++;
    assert(L[i].k == C_UPCMD); i++;
    assert(L[i].k == C_BARALL && L[i].a == CB); i++;   // after every earlier access of the old image
    const VkImageMemoryBarrier *b = &L[i].ib; assert(L[i].k == C_PBAR); i++;   // the new image from UNDEFINED into GENERAL, whole
    assert(b->image == (VkImage)(uintptr_t)NEW_IMG && b->oldLayout == VK_IMAGE_LAYOUT_UNDEFINED && b->newLayout == VK_IMAGE_LAYOUT_GENERAL && !b->srcAccessMask);
    assert(b->dstAccessMask == (VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT) && b->srcQueueFamilyIndex == VK_QUEUE_FAMILY_IGNORED && b->dstQueueFamilyIndex == VK_QUEUE_FAMILY_IGNORED);
    assert(b->subresourceRange.aspectMask == was.aspect && !b->subresourceRange.baseMipLevel && b->subresourceRange.levelCount == VK_REMAINING_MIP_LEVELS);
    assert(!b->subresourceRange.baseArrayLayer && b->subresourceRange.layerCount == VK_REMAINING_ARRAY_LAYERS);
    for (int l = 0; l < copies; l++, i++) {   // every level, all its layers, old into new
        const VkImageCopy *c = &L[i].c; assert(L[i].k == C_COPY && L[i].a == OLD_IMG && L[i].b == NEW_IMG);
        assert(c->srcSubresource.aspectMask == was.aspect && c->srcSubresource.mipLevel == (uint32_t)l && !c->srcSubresource.baseArrayLayer && c->srcSubresource.layerCount == was.layers);
        assert(!memcmp(&c->srcSubresource, &c->dstSubresource, sizeof c->srcSubresource) && !c->srcOffset.x && !c->srcOffset.y && !c->srcOffset.z && !c->dstOffset.x && !c->dstOffset.y && !c->dstOffset.z);
        assert(c->extent.width == lv(was.w, l) && c->extent.height == lv(was.h, l) && c->extent.depth == lv(was.d, l));
    }
    assert(L[i].k == C_BARALL && L[i].a == CB); i++;   // before anything later reads or writes the new image
    assert(L[i].k == C_DROPV && L[i].a == H(im)); i++;   // the old image's views
    if (was.sub) { assert(L[i].k == C_FREESUB && L[i].a == OLD_IMG && L[i].b == OLD_SUB); } else { assert(L[i].k == C_FREEMEM && L[i].a == OLD_IMG && L[i].b == OLD_MEM); } i++;
    assert(i == NL);   // nothing else: no main command buffer, nothing destroyed now
    assert(im->img == (VkImage)(uintptr_t)NEW_IMG && im->mem == (VkDeviceMemory)(uintptr_t)NEW_MEM && im->sub == (void *)(uintptr_t)NEW_SUB);
    assert(im->usage == (was.usage | VK_IMAGE_USAGE_STORAGE_BIT) && im->layout == VK_IMAGE_LAYOUT_GENERAL && im->cflags == was.cflags && im->fmt == was.fmt && im->fmt2 == was.fmt2);
    assert(im->w == was.w && im->h == was.h && im->d == was.d && im->layers == was.layers && im->levels == was.levels && im->samples == was.samples && im->target == was.target);
}
int main(void) {
    int ups = 0;
    // a 2D colour target of 4 levels, format-listed with its sRGB twin (DM_VK_UBWC)
    vkimg_t a = mk(1, 64, 32, 1, 1, 4); a.fixed_fmt = 1; a.fmt2 = VK_FORMAT_R8G8B8A8_SRGB; a.cflags = VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT; ok(&a, VK_IMAGE_TYPE_2D, 4); ups++;
    vkimg_t a1 = mk(1, 64, 64, 1, 1, 1); a1.fixed_fmt = 1; a1.fmt = VK_FORMAT_R16G16B16A16_SFLOAT; a1.cflags = 0; ok(&a1, VK_IMAGE_TYPE_2D, 1); ups++;   // a list of its format alone
    vkimg_t b = mk(2, 16, 16, 8, 1, 3); b.cflags |= VK_IMAGE_CREATE_2D_ARRAY_COMPATIBLE_BIT; ok(&b, VK_IMAGE_TYPE_3D, 3); ups++;   // 3D: one layer, the slices in the extent
    vkimg_t c = mk(4, 32, 32, 1, 6, 2); c.cflags |= VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT; ok(&c, VK_IMAGE_TYPE_2D, 2); ups++;   // a cube-compatible 2D array: all 6 layers a copy
    vkimg_t d = mk(3, 128, 1, 1, 4, 1); ok(&d, VK_IMAGE_TYPE_1D, 1); ups++;   // a 1D array
    vkimg_t e = mk(1, 5, 3, 1, 1, 3); ok(&e, VK_IMAGE_TYPE_2D, 3); ups++;   // odd sizes: 5x3, 2x1, 1x1
    vkimg_t f = mk(1, 32, 32, 1, 1, 2); f.layout = VK_IMAGE_LAYOUT_UNDEFINED; ok(&f, VK_IMAGE_TYPE_2D, 0); ups++;   // never used: nothing copied, the new one still put in GENERAL
    vkimg_t g = mk(0, 256, 1, 1, 1, 1); g.sub = NULL; ok(&g, VK_IMAGE_TYPE_1D, 1); ups++;   // memory of its own without a slot: vk_defer_free_image
    // refused: the image is left as it was, nothing freed
    vkimg_t h = mk(1, 32, 32, 1, 1, 1), h0 = h; tex_t t = {{0x25}, 0}; g_create_r = VK_ERROR_FORMAT_NOT_SUPPORTED; NL = 0; int red = g_nred;
    int r = img_upgrade(&h, &t); assert(r == 0 && NL == 1 && L[0].k == C_CREATE && !memcmp(&h, &h0, sizeof h) && g_nred == red + 1); g_create_r = VK_SUCCESS;
    g_mem_fail = 1; NL = 0; r = img_upgrade(&h, &t); assert(r == 0 && !memcmp(&h, &h0, sizeof h) && g_nred == red + 2);
    assert(NL == 4 && L[0].k == C_CREATE && L[1].k == C_REQ && L[2].k == C_MEM && L[3].k == C_DESTROY && L[3].a == NEW_IMG); g_mem_fail = 0;   // the new image destroyed at once (never used)
    // the log: the first 8 upgrades each, then a count every 64
    assert(g_nlog == 8);
    while (ups < 130) { vkimg_t x = mk(1, 8, 8, 1, 1, 1); ok(&x, VK_IMAGE_TYPE_2D, 1); ups++; }
    assert(g_nlog == 8 + 2);
    puts("upgrade ok");
    return 0;
}
'''


class ImageShare(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory(prefix='nx2a_vk_imgshare_')
        cls.decide = cls.upgrade = None
        cls.cc = shutil.which('cc') or shutil.which('clang')
        if not cls.cc or not (VKH / 'vulkan/vulkan_core.h').exists():
            return
        src = TEX.read_text(encoding='utf-8')
        cls.decide = cls.build('decide', HEAD + region(src, '// ---- image sharing: the decision', '// ---- end of the image sharing decision ----') + DECIDE_MAIN)
        cls.upgrade = cls.build('upgrade', HEAD + UPGRADE_STUBS + cut(src, 'static uint64_t g_img_upgrades;', '    return 1;\n}\n') + UPGRADE_MAIN)

    @classmethod
    def build(cls, name, code):
        c, exe = Path(cls.tmp.name) / f'{name}.c', Path(cls.tmp.name) / name
        c.write_text(code, encoding='utf-8')
        r = subprocess.run([cls.cc, '-std=gnu11', '-O1', '-Wall', '-Werror', '-Wno-unused-function', '-I' + str(VKH), str(c), '-o', str(exe)], capture_output=True, text=True)
        if r.returncode:
            raise AssertionError(r.stdout + r.stderr)
        return exe

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def need(self):
        if not self.decide:
            self.skipTest('a C compiler and the vendored Vulkan headers are needed')

    def run_exe(self, exe, *args, env=None):
        r = subprocess.run([str(exe), *args], capture_output=True, text=True, timeout=60, env=env)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        return r.stdout

    def test_decision(self):
        """Requirement 1: no storage needed or already there: use it (b); needed, its format takes storage images and nothing excludes the image: upgrade (a); else an image of its own (c)."""
        self.need()
        rows = [tuple(int(x) for x in line.split()) for line in self.run_exe(self.decide).splitlines()]
        self.assertEqual(len(rows), 4 * 2 * 2 * 4)
        for have, need, fok, excl, got in rows:
            want = REUSE if not need or have & STORAGE or not fok else UPGRADE if not excl else SEPARATE
            self.assertEqual(got, want, f'have {have:#x} need {need} format takes storage {fok} excluded {excl}')
        d = {r[:4]: r[4] for r in rows}
        sampled = 0x1 | 0x2 | 0x4 | 0x10
        self.assertEqual(d[(sampled, 1, 1, 0)], UPGRADE)                 # a render target the IMAGE texture finds: made again with storage use
        self.assertEqual(d[(sampled, 0, 1, 0)], REUSE)                   # the texture without IMAGE finds the storage one, or its own
        self.assertEqual(d[(0x4 | STORAGE | 0x10, 1, 1, 64)], REUSE)     # storage use already: nothing keeps it from being shared
        self.assertEqual(d[(sampled, 1, 1, 64)], SEPARATE)               # written in this submission already: an image of its own (SHX_BUSY)
        self.assertEqual(d[(sampled, 1, 0, 0)], REUSE)                   # its format takes no storage images (ASTC, BC): an image of its own would have none either
        self.assertEqual(d[(sampled, 1, 1, -1)], SEPARATE)               # any reason counts

    def test_reasons_text(self):
        self.need()
        def text(mask, n=1024):
            out = self.run_exe(self.decide, 'text', str(mask), str(n)).split('\n')
            self.assertEqual(len(out[0]), int(out[1]))
            return out[0]
        self.assertEqual(text(0), '')
        self.assertEqual(text(64 | 128), 'written in this submission already (a copy before its command buffers would miss it), its format takes no storage images')
        full = text(511)
        parts = full.split(', ')
        self.assertEqual(parts[0], 'sparse')
        self.assertEqual(parts[-1], 'the upgrade failed')
        self.assertEqual(len(parts), 9)
        self.assertEqual(text(4 | 8 | 16 | 32), 'packaged (CPU content), an sRGB / linear twin\'s image, linear, multisampled')
        for n in (1, 2, 7, 8, 30, len(full), len(full) + 1, len(full) + 2):   # cut to the buffer, always terminated, never past it
            got = text(511, n)
            self.assertTrue(full.startswith(got) and len(got) < n, (n, got))
            self.assertEqual(got, full[:n - 1] if n <= len(full) else full)
        self.run_exe(self.decide, 'text', '511', '0')   # an empty buffer: untouched

    def test_switch(self):
        """Requirement 3: DM_VK_IMAGE_SHARE, read once at start-up, on unless 0."""
        self.need()
        env = {k: v for k, v in os.environ.items() if k != 'DM_VK_IMAGE_SHARE'}
        self.assertEqual(self.run_exe(self.decide, 'share', env=env).strip(), '1')
        for v, want in (('0', '0'), ('1', '1'), ('2', '1')):
            self.assertEqual(self.run_exe(self.decide, 'share', env={**env, 'DM_VK_IMAGE_SHARE': v}).strip(), want, v)

    def test_upgrade(self):
        """Requirement 2: the new image as the old was created plus storage use, its memory, the copy of every level and layer between barriers in the upload command
        buffer, the old views dropped and the old image and memory freed later, the handles swapped in place; refused or out of memory: nothing changed, nothing freed."""
        self.need()
        self.assertIn('upgrade ok', self.run_exe(self.upgrade))

    def test_attachment_marked_busy(self):
        # begin_pass (gfx_vk_exec.c) prepares every attachment it renders into, most already in GENERAL, and nothing else marks attachments as written: the busy mark (msub) must come
        # before vk_tex_prepare's GENERAL return, or an upgrade later in the submission copies the image without this submission's rendering (the copy runs before the main command buffers)
        body = region(TEX.read_text(), 'void vk_tex_prepare(', '\n}\n')
        self.assertLess(body.index('im->msub = vk_sub_serial() + 1'), body.index('if (im->layout == VK_IMAGE_LAYOUT_GENERAL) return;'))

    def test_documented(self):
        doc = SWITCHES.read_text(encoding='utf-8')
        line = next((ln for ln in doc.splitlines() if ln.startswith('| `DM_VK_IMAGE_SHARE')), None)
        self.assertIsNotNone(line, 'DM_VK_IMAGE_SHARE has no entry in switches.md')
        self.assertIn('start-up', line)
        self.assertIn('gfx_vk_tex.c', line)


if __name__ == '__main__':
    unittest.main()
