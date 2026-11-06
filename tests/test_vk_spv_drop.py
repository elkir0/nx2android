"""The shaders' SPIR-V is not kept in host memory (gfx_vk_shader.c shader_words, vk_shader, vk_shader_module_): a shader's words are rewritten from the
mapped bundle, its module made, and the words freed; a variant rebuilds them with the same rewrite (Mario Kart World kept 286 MB of them for 4,813
shaders on the phone). gfx_vk_shader.c's own code (the bundle, the rewrite, vk_shader, the variant maker, the census) is built on the host with
vkCreateShaderModule stubbed to record each module's words, over a small bundle made here: words rebuilt with first=0 must equal the first build's,
byte for byte, and leave the shader untouched; every variant must be the one the old behaviour (DM_VK_SPV_KEEP=1, the words kept) makes; the
shaders other files read the words of (task / mesh, the game's upscaler) keep them; the alternative code (DM_VK_SPV_DIR) of a dropped shader still loads."""
from pathlib import Path
import glob
import os
import shutil
import struct
import subprocess
import tempfile
import unittest

DM = Path(__file__).resolve().parents[1] / 'nxnative/dm'
SHADER = DM / 'android/gfx_vk_shader.c'
HDR = DM / 'android/gfx_vk.h'
VKH = DM / 'third_party/vulkan-headers/include'
UPSCALER = '0123456789abcdef,4444000000000004'   # (DM_GAME_UPSCALER_KEYS: the second one is the bundle's C1)


def spirv_val():
    """spirv-val on PATH or in an NDK's shader-tools (None when none)."""
    if shutil.which('spirv-val'):
        return shutil.which('spirv-val')
    roots = [os.environ.get('ANDROID_NDK_HOME', ''), str(DM.parents[1] / 'tools/android-ndk-r27c')] + \
            sorted(glob.glob(os.path.expanduser('~/Library/Application Support/nx2android/toolchains/android-ndk-*')))
    for r in roots:
        for g in glob.glob(os.path.join(r, 'shader-tools/*/spirv-val')) if r else []:
            return g
    return None


# ---- SPIR-V written out as words ----
def lit(text):   # a literal string: UTF-8, NUL-terminated, padded to whole words
    b = text.encode() + b'\0'
    b += b'\0' * (-len(b) % 4)
    return list(struct.unpack(f'<{len(b) // 4}I', b))


class Ids:
    def __init__(self, names):
        self.__dict__.update({n: i + 1 for i, n in enumerate(names.split())})
        self.bound = len(names.split()) + 1


def module(version, n, insts):
    out = [0x07230203, version, 0, n.bound, 0]
    for op, *args in insts:
        ws = [w for a in args for w in (lit(a) if isinstance(a, str) else [a])]
        out += [(len(ws) + 1) << 16 | op] + ws
    return out


def frag():
    """A fragment shader as the translator writes them: Float64 and fragment shader interlock declared but unused, its sampled image in set 0 (binding 3)
    with its levels queried (a LOD-shift site), vec4 inputs at locations 0 and 1, float colour outputs at locations 0 and 1."""
    n = Ids('main void fn float v4 int img simg pimg pin pout tex v u o o1 l a b c t i q qf d')
    return module(0x10300, n, [
        (17, 1), (17, 10), (17, 50), (17, 5378), (10, 'SPV_EXT_fragment_shader_interlock'), (14, 0, 1),
        (15, 4, n.main, 'main', n.v, n.u, n.o, n.o1), (16, n.main, 7),
        (71, n.v, 30, 0), (71, n.u, 30, 1), (71, n.o, 30, 0), (71, n.o1, 30, 1), (71, n.tex, 34, 0), (71, n.tex, 33, 3),
        (19, n.void), (33, n.fn, n.void), (22, n.float, 32), (23, n.v4, n.float, 4), (21, n.int, 32, 1),
        (25, n.img, n.float, 1, 0, 0, 0, 1, 0), (27, n.simg, n.img), (32, n.pimg, 0, n.simg), (32, n.pin, 1, n.v4), (32, n.pout, 3, n.v4),
        (59, n.pimg, n.tex, 0), (59, n.pin, n.v, 1), (59, n.pin, n.u, 1), (59, n.pout, n.o, 3), (59, n.pout, n.o1, 3),
        (54, n.void, n.main, 0, n.fn), (248, n.l),
        (61, n.v4, n.a, n.v), (61, n.v4, n.b, n.u), (133, n.v4, n.c, n.a, n.b),
        (61, n.simg, n.t, n.tex), (100, n.img, n.i, n.t), (106, n.int, n.q, n.i), (111, n.float, n.qf, n.q), (142, n.v4, n.d, n.c, n.qf),
        (62, n.o, n.d), (62, n.o1, n.a), (253,), (56,)])


def vert():
    """A vertex shader that captures transform feedback (Xfb, its output in buffer 1 of stride 16), a uniform buffer in set 7, gl_Position and one vec4 output."""
    n = Ids('main void fn float v4 int i0 blk pu pv4u pout pin pos out0 in0 ubo l a p k x')
    return module(0x10300, n, [
        (17, 1), (17, 10), (17, 53), (14, 0, 1),
        (15, 0, n.main, 'main', n.pos, n.out0, n.in0), (16, n.main, 11),
        (71, n.pos, 11, 0), (71, n.out0, 30, 0), (71, n.out0, 36, 1), (71, n.out0, 37, 16), (71, n.in0, 30, 0),
        (71, n.ubo, 34, 7), (71, n.ubo, 33, 0), (71, n.blk, 2), (72, n.blk, 0, 35, 0),
        (19, n.void), (33, n.fn, n.void), (22, n.float, 32), (23, n.v4, n.float, 4), (21, n.int, 32, 1), (43, n.int, n.i0, 0),
        (30, n.blk, n.v4), (32, n.pu, 2, n.blk), (32, n.pv4u, 2, n.v4), (32, n.pout, 3, n.v4), (32, n.pin, 1, n.v4),
        (59, n.pout, n.pos, 3), (59, n.pout, n.out0, 3), (59, n.pin, n.in0, 1), (59, n.pu, n.ubo, 2),
        (54, n.void, n.main, 0, n.fn), (248, n.l),
        (61, n.v4, n.a, n.in0), (65, n.pv4u, n.p, n.ubo, n.i0), (61, n.v4, n.k, n.p), (129, n.v4, n.x, n.a, n.k),
        (62, n.pos, n.x), (62, n.out0, n.a), (253,), (56,)])


def mesh():
    """A mesh shader (SPIR-V 1.4) with a per-vertex output at location 0 and a per-primitive one at location 1."""
    n = Ids('main void fn float v4 uint u3 u1 a3 a1 pa3 pa1 col prim l')
    return module(0x10400, n, [
        (17, 5283), (10, 'SPV_EXT_mesh_shader'), (14, 0, 1),
        (15, 5365, n.main, 'main', n.col, n.prim), (16, n.main, 17, 1, 1, 1), (16, n.main, 26, 3), (16, n.main, 5270, 1), (16, n.main, 5298),
        (71, n.col, 30, 0), (71, n.prim, 30, 1), (71, n.prim, 5271),
        (19, n.void), (33, n.fn, n.void), (22, n.float, 32), (23, n.v4, n.float, 4), (21, n.uint, 32, 0),
        (43, n.uint, n.u3, 3), (43, n.uint, n.u1, 1), (28, n.a3, n.v4, n.u3), (28, n.a1, n.v4, n.u1),
        (32, n.pa3, 3, n.a3), (32, n.pa1, 3, n.a1), (59, n.pa3, n.col, 3), (59, n.pa1, n.prim, 3),
        (54, n.void, n.main, 0, n.fn), (248, n.l), (5295, n.u3, n.u1), (253,), (56,)])


def comp(local, dset):
    """A compute shader reading a uniform buffer in set dset."""
    n = Ids('main void fn float v4 int i0 blk pu pv4u ubo l p k')
    return module(0x10300, n, [
        (17, 1), (14, 0, 1), (15, 5, n.main, 'main'), (16, n.main, 17, local, local, 1),
        (71, n.ubo, 34, dset), (71, n.ubo, 33, 0), (71, n.blk, 2), (72, n.blk, 0, 35, 0),
        (19, n.void), (33, n.fn, n.void), (22, n.float, 32), (23, n.v4, n.float, 4), (21, n.int, 32, 1), (43, n.int, n.i0, 0),
        (30, n.blk, n.v4), (32, n.pu, 2, n.blk), (32, n.pv4u, 2, n.v4), (59, n.pu, n.ubo, 2),
        (54, n.void, n.main, 0, n.fn), (248, n.l), (65, n.pv4u, n.p, n.ubo, n.i0), (61, n.v4, n.k, n.p), (253,), (56,)])


RK_CBUF, RK_TEX, RD_UBO, RD_SAMPLED = 1, 7, 0, 2   # (gfx_vk.h)
FRAG, VERT, MESH, C1, C2 = 0x1111000000000001, 0x2222000000000002, 0x3333000000000003, 0x4444000000000004, 0x5555000000000005
SHADERS = {   # key: (bundle stage, words, bindings {kind, desc, binding})
    FRAG: (1, frag(), [(RK_TEX, RD_SAMPLED, 3)]),
    VERT: (0, vert(), [(RK_CBUF, RD_UBO, 0)]),
    MESH: (7, mesh(), []),
    C1: (5, comp(8, 2), [(RK_CBUF, RD_UBO, 0)]),
    C2: (5, comp(4, 2), [(RK_CBUF, RD_UBO, 0)]),
}


def bundle():
    """shader_vk.py's bundle: "DMSV", a version, the entry count, the entries sorted by key {u64 key, stage, spv_off, spv_len, refl_off, refl_len, pad},
    then each one's SPIR-V and reflection (u32 nbind, c1_len, local[3], 20-byte bindings, c1)."""
    keys = sorted(SHADERS)
    head = 12 + 32 * len(keys)
    blobs, ents = b'', []
    for k in keys:
        st, w, binds = SHADERS[k]
        spv = struct.pack(f'<{len(w)}I', *w)
        refl = struct.pack('<5I', len(binds), 16, 8, 8, 1) + b''.join(struct.pack('<BBBBHHhBBii', kind, desc, 0, 0, b, 1, 0, 0, 0, 0, 0) for kind, desc, b in binds) + bytes(16)
        so = head + len(blobs); blobs += spv
        ro = head + len(blobs); blobs += refl + bytes(-len(refl) % 4)
        ents.append(struct.pack('<QIIIIII', k, st, so, len(spv), ro, len(refl), 0))
    return b'DMSV' + struct.pack('<II', 1, len(keys)) + b''.join(ents) + blobs


def words(b):
    return list(struct.unpack(f'<{len(b) // 4}I', b))


def insts(w):
    i = 5
    while i < len(w):
        n = w[i] >> 16
        assert n, 'a zero-length instruction'
        yield w[i] & 0xffff, w[i + 1:i + n]
        i += n


# ---- the harness: gfx_vk_shader.c's bundle, rewrite, vk_shader, variant maker and census, with Vulkan stubbed ----
STUBS = r'''
#include <fcntl.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>
#define VK_NO_PROTOTYPES
#include <vulkan/vulkan_core.h>
#include "nvn_hmap.h"
#define LOG(...) (fprintf(stderr, __VA_ARGS__), fputc('\n', stderr))
#define die(...) (fprintf(stderr, __VA_ARGS__), abort())
'''
STUBS2 = r'''
static struct { VkDevice dev; int has_xfb; VkPhysicalDeviceProperties props; } V;
int g_prof_on; uint64_t g_prof_ns[8];
static uint32_t *g_last, g_nlast; static unsigned g_nmods;   // the words the last module was made from, the modules made
static VkResult vkCreateShaderModule(VkDevice d, const VkShaderModuleCreateInfo *ci, const VkAllocationCallbacks *a, VkShaderModule *m) {
    (void)d; (void)a; free(g_last); g_nlast = (uint32_t)(ci->codeSize / 4); g_last = malloc(ci->codeSize); memcpy(g_last, ci->pCode, ci->codeSize);
    *m = (VkShaderModule)(uintptr_t)++g_nmods; return VK_SUCCESS;
}
static unsigned g_ndsl;
static VkResult vkCreateDescriptorSetLayout(VkDevice d, const VkDescriptorSetLayoutCreateInfo *ci, const VkAllocationCallbacks *a, VkDescriptorSetLayout *l) {
    (void)d; (void)ci; (void)a; *l = (VkDescriptorSetLayout)(uintptr_t)++g_ndsl; return VK_SUCCESS;
}
void vk_dsl_register(VkDescriptorSetLayout l, const VkDescriptorSetLayoutBinding *b, uint32_t n) { (void)l; (void)b; (void)n; }
'''
MAIN = r'''
static void save(const char *dir, uint64_t key, const char *what, const uint32_t *w, uint32_t n) {
    char p[1024]; snprintf(p, sizeof p, "%s/%016llx.%s.spv", dir, (unsigned long long)key, what); FILE *f = fopen(p, "wb");
    if (!f || fwrite(w, 4, n, f) != n) { fprintf(stderr, "cannot write %s\n", p); exit(3); } fclose(f);
}
// words <out>: each bundle shader's words made with first = 1, then rebuilt with first = 0 (three times, and on a shader whose other fields hold a pattern)
static int words_mode(const char *out) {
    bundle_auto(); if (!g_bundle) return 3;
    for (uint32_t i = 0; i < g_nent; i++) {
        const bent *e = &g_ent[i]; uint64_t key; uint32_t st; memcpy(&key, &e->key, 8); memcpy(&st, &e->stage, 4);
        vkshader_t *s = calloc(1, sizeof *s); s->key = key; s->stage = (int)st; s->set = set_of((int)st);   // (as vk_shader)
        const uint8_t *r = g_bundle + e->refl_off; uint32_t hdr[5]; memcpy(hdr, r, 20); s->nbind = (int)hdr[0]; s->b = (const dmsv_bind *)(r + 20);
        uint32_t n1 = 0, *w1 = shader_words(s, e, &n1, 1); if (!w1) return 3;
        vkshader_t snap; memcpy(&snap, s, sizeof snap); int same = 1, untouched = 1;
        for (int k = 0; k < 3; k++) { uint32_t n2 = 0, *w2 = shader_words(s, e, &n2, 0); same &= w2 && n2 == n1 && !memcmp(w1, w2, (size_t)n1 * 4); untouched &= !memcmp(&snap, s, sizeof snap); free(w2); }
        vkshader_t *p = malloc(sizeof *p); memset(p, 0xa5, sizeof *p); p->key = s->key; p->stage = s->stage; p->set = s->set; p->nbind = s->nbind; p->b = s->b;
        vkshader_t pat; memcpy(&pat, p, sizeof pat); uint32_t n3 = 0, *w3 = shader_words(p, e, &n3, 0);
        same &= w3 && n3 == n1 && !memcmp(w1, w3, (size_t)n1 * 4); untouched &= !memcmp(&pat, p, sizeof pat);
        printf("words %016llx stage %d nw %u raw %u xfb %d stride %u,%u,%u,%u depth %d in %#x out %#x prim %#x col0 %d lod %#llx same %d untouched %d\n", (unsigned long long)key, s->stage, n1,
               e->spv_len / 4, s->xfb, s->xfb_stride[0], s->xfb_stride[1], s->xfb_stride[2], s->xfb_stride[3], s->depth_repl, s->in_locs, s->out_locs, s->prim_locs, s->col0,
               (unsigned long long)s->lod_ops, same, untouched);
        save(out, key, "base", w1, n1); free(w1); free(w3); free(p); free(s);
    }
    return 0;
}
// drop <out>: vk_shader for each bundle shader, then its variants (each asked twice: the second from the cache), the alternative code of the compute shaders
// (DM_VK_SPV_DIR), the census
typedef struct { uint64_t um, im; uint32_t xo; int fd; uint64_t lm; } req_t;
static int drop_mode(const char *out) {
    bundle_auto(); if (!g_bundle) return 3;
    for (uint32_t i = 0; i < g_nent; i++) {
        uint64_t key; uint32_t st; memcpy(&key, &g_ent[i].key, 8); memcpy(&st, &g_ent[i].stage, 4);
        unsigned before = g_nmods; vkshader_t *s = vk_shader(key, (int)st); if (!s || g_nmods != before + 1) return 4;
        printf("shader %016llx stage %d kept %d nspv %u xfb %d stride %u,%u,%u,%u depth %d in %#x out %#x prim %#x col0 %d lod %#llx\n", (unsigned long long)key, s->stage, s->spv != NULL, s->nspv,
               s->xfb, s->xfb_stride[0], s->xfb_stride[1], s->xfb_stride[2], s->xfb_stride[3], s->depth_repl, s->in_locs, s->out_locs, s->prim_locs, s->col0, (unsigned long long)s->lod_ops);
        save(out, key, "mod", g_last, g_nlast);
        req_t rq[8]; int nr = 0;
        if (s->stage == 1) { rq[nr++] = (req_t){1, 0, 0, 0, 0}; rq[nr++] = (req_t){0, 0, 0, 1, 0}; rq[nr++] = (req_t){0, 0, 0, 4, 0}; rq[nr++] = (req_t){0, 0, 0, 2, 0};
                             rq[nr++] = (req_t){0, 0, s->in_locs, 0, 0}; rq[nr++] = (req_t){0, 0, 0, 0, s->lod_ops}; rq[nr++] = (req_t){1, 0, 0, 5, s->lod_ops}; }
        if (s->stage == 0) rq[nr++] = (req_t){0, 0, 4, 0, 0};
        if (s->stage == 7) { rq[nr++] = (req_t){0, 0, 0, 8, 0}; rq[nr++] = (req_t){0, 0, 0, 16, 0}; }
        for (int k = 0; k < nr; k++) {
            unsigned b = g_nmods; VkShaderModule m = vk_shader_module_lod(s, rq[k].um, rq[k].im, rq[k].xo, rq[k].fd, rq[k].lm); int made = g_nmods != b;
            if (made) { char what[16]; snprintf(what, sizeof what, "v%d", k); save(out, key, what, g_last, g_nlast); }
            unsigned b2 = g_nmods; VkShaderModule m2 = vk_shader_module_lod(s, rq[k].um, rq[k].im, rq[k].xo, rq[k].fd, rq[k].lm);
            printf("variant %016llx %d made %d cached %d\n", (unsigned long long)key, k, made, m2 == m && g_nmods == b2);
        }
        if (s->stage == 5 && getenv("DM_VK_SPV_DIR")) { unsigned b = g_nmods; VkShaderModule m = vk_shader_module_lod(s, 0, 0, 1u << 31, 0, 0);
            printf("alt %016llx made %d\n", (unsigned long long)key, g_nmods != b && m != s->mod); if (g_nmods != b) save(out, key, "alt", g_last, g_nlast); }
    }
    uint64_t o[4] = {0}; vk_shader_mem(o);
    printf("mem %llu %llu %llu %llu\n", (unsigned long long)o[0], (unsigned long long)o[1], (unsigned long long)o[2], (unsigned long long)o[3]);
    return 0;
}
int main(int argc, char **argv) {
    const char *x = getenv("TEST_HAS_XFB"); V.has_xfb = x && atoi(x);
    if (argc == 3 && !strcmp(argv[1], "words")) return words_mode(argv[2]);
    if (argc == 3 && !strcmp(argv[1], "drop")) return drop_mode(argv[2]);
    return 2;
}
'''


def harness_source(src, hdr, with_words=True):
    """The harness's C source from gfx_vk_shader.c (src) and gfx_vk.h (hdr): the bundle, then everything from the stage tables to the variant maker
    (the capture's code and the static buffers left out). with_words=False leaves the words mode out: the drop mode alone also builds against a
    gfx_vk_shader.c from before shader_words (git show), whose modules the working tree's must equal."""
    def region(text, start, end):
        a = text.index(start)
        return text[a:text.index(end, a)]
    picks = '\n'.join(l for l in hdr.splitlines() if l.startswith(('#define VKLOG(', '#define VKRED_ONCE(', '#define VKCHK(', 'enum { RK_SUPPORT', 'enum { RD_UBO',
                                                                   'enum { VK_COL0_NONE', '#define VK_ALPHA_PUSH_')))
    types = region(hdr, 'typedef struct __attribute__((packed)) { uint8_t kind, desc', '} vkshader_t;') + '} vkshader_t;\n'
    body = region(src, 'typedef struct __attribute__((packed)) { uint64_t key;', '// Run-time capture') + \
        region(src, "// NVN2's task and mesh shaders are the bundle's stages 6 and 7", '// static buffers:')
    main = MAIN if with_words else MAIN.replace('if (argc == 3 && !strcmp(argv[1], "words")) return words_mode(argv[2]);', '')
    if not with_words:
        a = main.index('// words <out>'); main = main[:a] + main[main.index('// drop <out>'):]
    return STUBS + picks + '\n' + types + STUBS2 + body + main


def build_harness(src_text, hdr_text, out_dir, with_words=True):
    """Compiles the harness into out_dir/h; (path, None) or (None, the compiler's messages)."""
    cc = shutil.which('cc') or shutil.which('clang')
    c = Path(out_dir) / 'h.c'
    c.write_text(harness_source(src_text, hdr_text, with_words), encoding='utf-8')
    r = subprocess.run([cc, '-std=gnu11', '-O1', '-Wall', '-Werror', '-Wno-unused-function', '-Wno-unused-variable', '-Wno-unused-but-set-variable', '-D_GNU_SOURCE',
                        f'-DDM_GAME_UPSCALER_KEYS="{UPSCALER}"', f'-I{VKH}', f'-I{DM / "gfx/nvn"}', str(c), '-o', str(Path(out_dir) / 'h'), '-lpthread'],
                       capture_output=True, text=True)
    return (Path(out_dir) / 'h', None) if not r.returncode else (None, r.stdout + r.stderr)


class SpvDrop(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory(prefix='nx2a_vk_spv_drop_')
        cls.dir = Path(cls.tmp.name)
        cls.bundle = cls.dir / 'shaders.dmsv'
        cls.bundle.write_bytes(bundle())
        cls.exe = None
        if shutil.which('cc') or shutil.which('clang'):
            cls.exe, err = build_harness(SHADER.read_text(encoding='utf-8'), HDR.read_text(encoding='utf-8'), cls.dir)
            if err:
                raise AssertionError(err)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def run_mode(self, mode, name, **env):
        if not self.exe:
            self.skipTest('a C compiler is needed')
        out = self.dir / name
        out.mkdir(exist_ok=True)
        e = {k: v for k, v in os.environ.items() if not k.startswith(('DM_VK_SPV', 'DM_VK_LODSHIFT'))}
        e.update({'DM_VK_SHADERS': str(self.bundle), 'TEST_HAS_XFB': '0'}, **env)
        r = subprocess.run([str(self.exe), mode, str(out)], capture_output=True, text=True, env=e)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        return out, r.stdout.splitlines(), r.stderr

    @staticmethod
    def fields(line):
        t = line.split()
        return {t[i]: t[i + 1] for i in range(2, len(t) - 1, 2)}

    def test_rebuilt_words_are_the_first_ones(self):
        """shader_words(first = 0) gives the first build's words byte for byte and writes nothing of the shader; first = 1 records the facts as vk_shader did."""
        for xfb in ('0', '1'):
            out, lines, _ = self.run_mode('words', f'words{xfb}', TEST_HAS_XFB=xfb)
            got = {int(l.split()[1], 16): self.fields(l) for l in lines if l.startswith('words ')}
            self.assertEqual(set(got), set(SHADERS))
            for k, f in got.items():
                self.assertEqual((f['same'], f['untouched']), ('1', '1'), f'{k:016x}: {f}')
            f, v, m = got[FRAG], got[VERT], got[MESH]
            self.assertEqual((f['col0'], f['in'], f['out'], f['lod'], f['depth']), ('1', '0x3', '0x3', '0x1', '0'))   # VK_COL0_FLOAT; binding index 0 takes level numbers
            self.assertEqual((v['xfb'], v['stride'], v['in'], v['out']), (xfb, '0,16,0,0', '0x1', '0x1'))
            self.assertEqual((m['prim'], m['out']), ('0x2', '0x3'))
            for k, (st, raw, _) in SHADERS.items():
                w = words((out / f'{k:016x}.base.spv').read_bytes())
                ins = list(insts(w))
                want = {1: 1, 0: 0, 7: 0, 5: 0}[st]   # (set_of)
                self.assertTrue(all(a[2] == want for op, a in ins if op == 71 and len(a) == 3 and a[1] == 34), f'{k:016x}: DescriptorSet not {want}')
                self.assertNotIn((17, [10]), ins)   # Float64 unused: a second Shader
                self.assertNotIn((17, [5378]), ins)  # interlock unused: dropped, its extension too
                self.assertFalse(any(op == 10 and bytes(struct.pack(f'<{len(a)}I', *a)).startswith(b'SPV_EXT_fragment_shader_interlock') for op, a in ins))
                if st == 0:
                    dropped = xfb == '0'
                    self.assertEqual((17, [53]) not in ins, dropped)
                    self.assertEqual(not any(op == 16 and a[1] == 11 for op, a in ins), dropped)
                    self.assertEqual(not any(op == 71 and a[1] in (36, 37) for op, a in ins), dropped)
                    self.assertEqual(sum(op == 62 for op, a in ins), sum(op == 62 for op, a in insts(raw)) + 1)   # spv_vs_init: out0 starts as (0, 0, 0, 1)
                if st != 7:   # (the mesh shader has nothing to rewrite)
                    self.assertNotEqual(w, raw)

    def test_variants_as_with_the_words_kept(self):
        """Every module, base and variant, is the one the old behaviour (DM_VK_SPV_KEEP=1) makes; the words are dropped but where other files read them; each variant is made once."""
        keep, kl, kerr = self.run_mode('drop', 'keep', DM_VK_SPV_KEEP='1')
        drop, dl, derr = self.run_mode('drop', 'drop')
        kf, df = sorted(p.name for p in keep.iterdir()), sorted(p.name for p in drop.iterdir())
        self.assertEqual(kf, df)
        for name in kf:
            self.assertEqual((keep / name).read_bytes(), (drop / name).read_bytes(), name)
        made = [l for l in dl if l.startswith('variant ')]
        self.assertTrue(made and all(l.endswith('cached 1') for l in made), made)
        self.assertGreaterEqual(sum(l.split()[4] == '1' for l in made if int(l.split()[1], 16) == FRAG), 6)   # (the fragment shader's variants made, not left as the module)
        self.assertTrue(any(l.split()[4] == '1' for l in made if int(l.split()[1], 16) == VERT))
        kept = {int(l.split()[1], 16): (l.split()[5], int(l.split()[7])) for l in dl if l.startswith('shader ')}
        self.assertEqual({k: v[0] for k, v in kept.items()}, {FRAG: '0', VERT: '0', MESH: '1', C1: '1', C2: '0'})   # task / mesh, the upscaler: kept
        self.assertTrue(all(l.split()[5] == '1' for l in kl if l.startswith('shader ')))
        mods = {k: len(words((drop / f'{k:016x}.mod.spv').read_bytes())) for k in SHADERS}
        self.assertEqual({k: v[1] for k, v in kept.items()}, mods)   # nspv: the module's length, kept or not
        ko, do = [list(map(int, next(l for l in x if l.startswith('mem ')).split()[1:])) for x in (kl, dl)]
        nvar = sum(1 for l in made)
        rebuilt = sum(1 for l in made if kept[int(l.split()[1], 16)][0] == '0')
        self.assertEqual(ko, [len(SHADERS), 4 * sum(mods.values()), nvar, 0])
        self.assertEqual(do, [len(SHADERS), 4 * (mods[MESH] + mods[C1]), nvar, rebuilt])
        self.assertIn(f'rebuilt from the bundle for {rebuilt} variants', derr)
        _, wl, _ = self.run_mode('words', 'facts')
        facts = {int(l.split()[1], 16): {k: v for k, v in self.fields(l).items() if k not in ('nw', 'raw', 'same', 'untouched')} for l in wl if l.startswith('words ')}
        for l in dl:
            if l.startswith('shader '):
                f = self.fields(l)
                self.assertEqual({k: v for k, v in f.items() if k not in ('kept', 'nspv')}, facts[int(l.split()[1], 16)], l)
        self.assertNotIn('could not be rebuilt', derr)
        self.assertNotIn('rebuilt for a variant is', derr)   # (the rebuilt words' length is the module's)

    def test_alternative_code_of_a_dropped_shader(self):
        """DM_VK_SPV_DIR's code is read from its file, with the stage's set, whether the bundle's words were kept or not."""
        alt = self.dir / 'alt'
        alt.mkdir(exist_ok=True)
        w = comp(16, 3)
        for k in (C1, C2):
            (alt / f'{k:016x}.spv').write_bytes(struct.pack(f'<{len(w)}I', *w))
        out, lines, _ = self.run_mode('drop', 'altrun', DM_VK_SPV_DIR=str(alt))
        self.assertTrue(any(l.startswith(f'shader {C2:016x} stage 5 kept 0 nspv {len(w)} ') for l in lines), lines)
        want = [x if not (i >= 3 and w[i - 3] == (4 << 16 | 71) and w[i - 1] == 34) else 0 for i, x in enumerate(w)]   # DescriptorSet 3 → 0
        self.assertNotEqual(want, w)
        for k in (C1, C2):
            self.assertIn(f'alt {k:016x} made 1', lines)
            self.assertEqual(words((out / f'{k:016x}.alt.spv').read_bytes()), want)

    def test_modules_valid(self):
        """The rewritten words the base modules are made from pass spirv-val (when it is at hand)."""
        val = spirv_val()
        if not val:
            self.skipTest('spirv-val is needed')
        out, _, _ = self.run_mode('words', 'val')
        for k, (st, _, _) in SHADERS.items():
            p = out / f'{k:016x}.base.spv'
            r = subprocess.run([val, '--target-env', 'vulkan1.3', str(p)], capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, f'{p.name}: {r.stdout}{r.stderr}')


if __name__ == '__main__':
    unittest.main()
