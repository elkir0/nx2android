// Android port, Vulkan backend: the game's shaders (see gfx_vk.h).
// The bundle (dm/android/tools/shader_vk.py) is mapped read-only: entries sorted by key, each the SPIR-V of one Switch shader and its binary reflection.
// The translator numbers every stage's bindings from 0 in set 0; at module creation each stage's bindings move to a set of its own (vertex 0, fragment 1,
// geometry 2, tessellation 3 / 4; compute 0) by rewriting the DescriptorSet decorations, so a program's stages never collide.
#include "gfx_vk.h"
#include "nvn_hmap.h"
#include <CommonCrypto/CommonDigest.h>
#include <fcntl.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

typedef struct __attribute__((packed)) { uint64_t key; uint32_t stage, spv_off, spv_len, refl_off, refl_len, pad; } bent;
static const uint8_t *g_bundle; static uint32_t g_nent; static const bent *g_ent;
void vk_shaders_open(const char *path) {
    int fd = open(path, O_RDONLY | O_CLOEXEC); struct stat st;
    if (fd < 0 || fstat(fd, &st)) { VKLOG("[RED] no shader bundle at %s: nothing will be drawn", path); if (fd >= 0) close(fd); return; }
    g_bundle = mmap(NULL, (size_t)st.st_size, PROT_READ, MAP_SHARED, fd, 0); close(fd);
    if (g_bundle == MAP_FAILED || memcmp(g_bundle, "DMSV", 4)) { VKLOG("[RED] %s is not a shader bundle", path); g_bundle = NULL; return; }
    memcpy(&g_nent, g_bundle + 8, 4); g_ent = (const bent *)(g_bundle + 12);
    VKLOG("shader bundle %s: %u shaders, %.1f MB", path, g_nent, st.st_size / 1048576.0);
}
static void bundle_open_once(void) {
    const char *p = getenv("DM_VK_SHADERS"); char buf[1024];
    if (!p) { const char *r = getenv("DM_ROOT"); snprintf(buf, sizeof buf, "%s/shaders.dmsv", r ? r : "."); p = buf; }
    vk_shaders_open(p);
}
// once, and every caller waits until it is open: the replay and the prewarm threads both look shaders up at start, and a lookup that ran before the bundle
// was open cached its shader as missing for good (every draw skipped as "no program" in some runs)
static void bundle_auto(void) { static pthread_once_t once = PTHREAD_ONCE_INIT; pthread_once(&once, bundle_open_once); }
int vk_shader_keys(uint32_t stage, uint64_t *out, int max) {   // the bundle's shaders of one stage (gfx_vk_exec.c compute_warm)
    bundle_auto(); int n = 0;
    for (uint32_t i = 0; i < g_nent && n < max; i++) { uint32_t st; memcpy(&st, &g_ent[i].stage, 4); if (st == stage) memcpy(&out[n++], &g_ent[i].key, 8); }
    return n;
}
static const bent *find(uint64_t key) {
    uint32_t lo = 0, hi = g_nent;
    while (lo < hi) { uint32_t m = (lo + hi) / 2; uint64_t k; memcpy(&k, &g_ent[m].key, 8); if (k < key) lo = m + 1; else hi = m; }
    if (lo < g_nent) { uint64_t k; memcpy(&k, &g_ent[lo].key, 8); if (k == key) return &g_ent[lo]; }
    return NULL;
}
int vk_shader_stage(uint64_t key) { bundle_auto(); const bent *e = g_bundle ? find(key) : NULL; return e ? (int)e->stage : -1; }   // the NVN stage the bundle has this shader as

// Run-time capture: a shader a draw needs that the bundle does not have is saved as DM_ROOT/captured_shaders/<key>.bin ("DMSC", u32 control length, u32 code length, the control
// section as its key hashed it (an NVN2 one with the bytes after its key's, below), the code area), and the porter translates the folder into the bundle at the next port (porter/pipeline.py, shader_vk.py --capture).
// Xenoblade Chronicles 3 sets hundreds of programs whose code is in no game file (GLSLC outputs of its Firehawk effects, and vertex / pixel shaders of models: a machine near
// the camera in Everlight Plain drew as a black silhouette, its depth pre-pass drawn and its G-buffer draw skipped); 51 of them in the opening and the first field.
// The game may free a control section once SetShaders returns, so SetShaders keeps a copy of each one the bundle lacks (shader_set); the file is written only when a draw misses
// the key (vk_shader_missing) and the code area then still hashes to it.
// The key's control bytes are hle_nvn.c's (nvn_ctl_lens): NVN's section up to its own length (0x880 at most), NVN2's (Switch 2, version 0xc9) first 0x48 bytes with +0x30 as zero. An
// NVN2 section is kept longer than its key, as the translation reads it (shader_vk.py nvn2_job: compute's layout to +0xbb, a task or mesh shader's facts to +0x68): 0xd0 bytes, the
// shortest NVN2 section seen (0xd8 in BNSH files, 0xd0 in effects; its length is 0 in memory, so it cannot be read there)
typedef struct { uint64_t key, data; uint32_t nctl, ncode, nkey; uint8_t *ctl; } pend_t;
enum { NPEND = 4096, NVN2_SAVE_CTL = 0xd0 };
static pend_t g_pend[NPEND]; static unsigned g_npend; static pthread_mutex_t g_pend_mu = PTHREAD_MUTEX_INITIALIZER;
static int nvn2_section(const uint8_t *c) { uint32_t m[2]; memcpy(m, c, 8); return m[0] == 0x98761234u && m[1] == 0xc9; }
static void shader_set(const uint8_t k[8], const void *control, uint64_t data, uint32_t ncode) {
    uint64_t key = 0; for (int i = 0; i < 8; i++) key = key << 8 | k[i];
    if (!key || !control || !data || !ncode || ncode >= (1u << 24) || vk_shader_stage(key) >= 0) return;
    uint32_t nkey, nc; nvn_ctl_lens(control, &nkey, &nc);   // as hle_nvn.c keys_hash
    uint32_t nctl = nvn2_section(control) ? NVN2_SAVE_CTL : nkey;
    uint8_t *c = malloc(nctl); if (!c) return; memcpy(c, control, nctl);
    pthread_mutex_lock(&g_pend_mu);
    pend_t *p = &g_pend[g_npend++ % NPEND]; free(p->ctl); *p = (pend_t){key, data, nctl, ncode, nkey, c};   // the oldest goes: a draw needs a program soon after it is set
    pthread_mutex_unlock(&g_pend_mu);
}
__attribute__((constructor)) static void shader_set_on(void) { extern void (*g_nvn_on_shader)(const uint8_t *, const void *, uint64_t, uint32_t); g_nvn_on_shader = shader_set; }
void vk_shader_missing(uint64_t key) {
    pend_t p = {0};
    pthread_mutex_lock(&g_pend_mu);
    for (unsigned i = 0; i < NPEND && i < g_npend; i++) if (g_pend[i].key == key && g_pend[i].ctl) { p = g_pend[i]; p.ctl = malloc(p.nctl); if (p.ctl) memcpy(p.ctl, g_pend[i].ctl, p.nctl); break; }
    pthread_mutex_unlock(&g_pend_mu);
    if (!p.ctl) { VKLOG("[RED] shader %016llx is not in the bundle (and was not set while the capture could see it)", (unsigned long long)key); return; }
    uint8_t md[CC_SHA256_DIGEST_LENGTH], kc[0x880]; CC_SHA256_CTX hc; CC_SHA256_Init(&hc);
    memcpy(kc, p.ctl, p.nkey); if (nvn2_section(p.ctl)) memset(kc + 0x30, 0, 4);   // the key's control bytes (hle_nvn.c keys_hash)
    CC_SHA256_Update(&hc, kc, p.nkey); CC_SHA256_Update(&hc, (const void *)(uintptr_t)p.data, p.ncode); CC_SHA256_Final(md, &hc);
    uint64_t now = 0; for (int i = 0; i < 8; i++) now = now << 8 | md[i];
    if (now != key) { VKLOG("[RED] shader %016llx is not in the bundle; its code area changed since SetShaders, not saved", (unsigned long long)key); free(p.ctl); return; }
    const char *r = getenv("DM_ROOT"); char dir[1024], path[1100], tmp[1110]; snprintf(dir, sizeof dir, "%s/captured_shaders", r ? r : "."); mkdir(dir, 0755);
    snprintf(path, sizeof path, "%s/%016llx.bin", dir, (unsigned long long)key); snprintf(tmp, sizeof tmp, "%s.part", path);
    struct stat st; if (!stat(path, &st)) { VKLOG("[RED] shader %016llx is not in the bundle: captured in %s by an earlier run, the next port translates it", (unsigned long long)key, path); free(p.ctl); return; }
    FILE *f = fopen(tmp, "wb"); uint32_t hdr[3] = {0x43534d44, p.nctl, p.ncode};   // "DMSC"
    int ok = f && fwrite(hdr, 1, 12, f) == 12 && fwrite(p.ctl, 1, p.nctl, f) == p.nctl && fwrite((const void *)(uintptr_t)p.data, 1, p.ncode, f) == p.ncode;
    if (f && fclose(f)) ok = 0;
    if (!ok || rename(tmp, path)) { VKLOG("[RED] shader %016llx is not in the bundle, and %s could not be written", (unsigned long long)key, path); unlink(tmp); }
    else VKLOG("[RED] shader %016llx is not in the bundle: saved to %s, the next port translates it", (unsigned long long)key, path);
    free(p.ctl);
}
// NVN2's task and mesh shaders are the bundle's stages 6 and 7 (shader_vk.py STAGES; the program binds them in the tessellation stages' places, gfx_vk_exec.c bind_program). A mesh
// pipeline draws the mesh shader in the vertex place and the task shader in the geometry place (gfx_vk_exec.c mesh_draw), so their bindings take those places' sets, 0 and 2:
// three sets at most, as for a vertex pipeline (Adreno's maxBoundDescriptorSets is small)
static VkShaderStageFlags stage_flag(int st) {
    switch (st) { case 0: return VK_SHADER_STAGE_VERTEX_BIT; case 1: return VK_SHADER_STAGE_FRAGMENT_BIT; case 2: return VK_SHADER_STAGE_GEOMETRY_BIT;
                  case 3: return VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT; case 4: return VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT;
                  case 6: return VK_SHADER_STAGE_TASK_BIT_EXT; case 7: return VK_SHADER_STAGE_MESH_BIT_EXT; default: return VK_SHADER_STAGE_COMPUTE_BIT; }
}
static int set_of(int st) { return st == 5 ? 0 : st == 0 ? 0 : st == 1 ? 1 : st == 2 ? 2 : st == 6 ? 2 : st == 7 ? 0 : st; }
static VkDescriptorType desc_type(int d) {
    switch (d) { case RD_UBO: return VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER; case RD_SSBO: return VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; case RD_SAMPLED: return VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
                 case RD_STORAGE: return VK_DESCRIPTOR_TYPE_STORAGE_IMAGE; case RD_TBUF: return VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER; case RD_STORAGE_TBUF: return VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER;
                 case RD_IMAGE: return VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE; default: return VK_DESCRIPTOR_TYPE_SAMPLER; }
}
// Stage interfaces. Vulkan requires every input location of a stage to be declared as an output of the stage before it (VUID-RuntimeSpirv-OpEntryPoint-08743);
// the translator writes each stage apart, so a stage may read locations the one before never declares. Xenoblade Chronicles 3's passthrough geometry shaders
// (written out as ordinary ones, forwarding all 16 attributes) follow vertex shaders with two outputs, and Adreno refused those pipelines (VK_ERROR_UNKNOWN).
// The earlier stage then gets a variant that declares the missing vec4 outputs and leaves them unwritten; a vertex shader's variant then stores (0, 0, 0, 1) to them
// as to its own outputs (spv_vs_init), the values Maxwell gives unwritten outputs, where Vulkan leaves them undefined.
static int spv_is_vec4(const uint32_t *w, uint32_t fn, uint32_t t) {   // a vec4 of 32-bit floats, or an array of them (per-vertex inputs)
    for (int depth = 0; depth < 3; depth++) { uint32_t i = 5, found = 0;
        while (i < fn) { uint32_t n = w[i] >> 16; if (!n) return 0; if (w[i + 1] == t && ((w[i] & 0xffff) == 23 || (w[i] & 0xffff) == 28 || (w[i] & 0xffff) == 32)) { found = i; break; } i += n; }
        if (!found) return 0; uint32_t op = w[found] & 0xffff;
        if (op == 32) { t = w[found + 3]; continue; }   // OpTypePointer
        if (op == 28) { t = w[found + 2]; continue; }   // OpTypeArray
        if (w[found + 3] != 4) return 0; for (i = 5; i < fn;) { if ((w[i] & 0xffff) == 22 && w[i + 1] == w[found + 2]) return w[i + 2] == 32; i += w[i] >> 16; } return 0; }
    return 0;
}
static void spv_io_locs(const uint32_t *w, uint32_t nw, uint32_t *in, uint32_t *out) {
    *in = *out = 0; uint32_t fn = nw;
    for (uint32_t i = 5; i < nw;) { uint32_t n = w[i] >> 16; if (!n) return; if ((w[i] & 0xffff) == 54) { fn = i; break; } i += n; }
    for (uint32_t i = 5; i < fn;) { uint32_t n = w[i] >> 16;
        if ((w[i] & 0xffff) == 71 && n == 4 && w[i + 2] == 30 && w[i + 3] < 32) {   // OpDecorate Location
            uint32_t v = w[i + 1];
            for (uint32_t j = 5; j < fn;) { uint32_t m = w[j] >> 16; if ((w[j] & 0xffff) == 59 && w[j + 2] == v) {
                    if (spv_is_vec4(w, fn, w[j + 1])) { if (w[j + 3] == 1) *in |= 1u << w[i + 3]; else if (w[j + 3] == 3) *out |= 1u << w[i + 3]; }
                    else if (w[j + 3] == 3) *out |= 1u << w[i + 3];   // an output of another type still takes its location
                    break; }
                j += m; } }
        i += n; }
}
// A fragment shader's colour 0 (the alpha test reads its alpha: spv_alpha_test): the Output variable at Location 0 without an Index past 0. Returns VK_COL0_* (gfx_vk.h) and, for a vec4 of
// 32-bit floats, the variable, its vector type and its float type
static int spv_col0(const uint32_t *w, uint32_t nw, uint32_t *var, uint32_t *v4, uint32_t *f32) {
    uint32_t bound = w[3], fn = nw, cv = 0; int kind = VK_COL0_NONE; uint8_t *loc0 = calloc(bound, 1); if (!loc0) return VK_COL0_NONE;
    for (uint32_t i = 5; i < nw;) { uint32_t n = w[i] >> 16, op = w[i] & 0xffff; if (!n || i + n > nw) { free(loc0); return VK_COL0_NONE; } if (op == 54) { fn = i; break; }
        if (op == 71 && n == 4 && w[i + 1] < bound && ((w[i + 2] == 30 && w[i + 3] == 0) || (w[i + 2] == 32 && w[i + 3] > 0)))
            loc0[w[i + 1]] |= w[i + 2] == 30 ? 1 : 2;   // Location 0 (1), an Index past 0 (2: a dual-source second output)
        i += n; }
    for (uint32_t i = 5; i < fn && !cv;) { uint32_t n = w[i] >> 16; if ((w[i] & 0xffff) == 59 && n >= 4 && w[i + 3] == 3 && w[i + 2] < bound && loc0[w[i + 2]] == 1) cv = i; i += n; }   // OpVariable ... Output
    free(loc0);
    if (!cv) return VK_COL0_NONE;
    uint32_t t = 0, c = 0, cnt = 0, comp = 0, width = 0;
    for (uint32_t i = 5; i < fn;) { uint32_t n = w[i] >> 16; if ((w[i] & 0xffff) == 32 && n == 4 && w[i + 1] == w[cv + 1]) t = w[i + 3]; i += n; }   // its pointer type's pointee
    for (uint32_t i = 5; i < fn && t;) { uint32_t n = w[i] >> 16; if ((w[i] & 0xffff) == 23 && n == 4 && w[i + 1] == t) { c = w[i + 2]; cnt = w[i + 3]; } i += n; }   // OpTypeVector
    for (uint32_t i = 5; i < fn && c;) { uint32_t n = w[i] >> 16, op = w[i] & 0xffff; if ((op == 21 || op == 22) && w[i + 1] == c) { comp = op; width = w[i + 2]; } i += n; }
    kind = comp == 21 ? VK_COL0_INT : comp == 22 && width == 32 && cnt == 4 ? VK_COL0_FLOAT : VK_COL0_OTHER;
    if (kind == VK_COL0_FLOAT) { if (var) *var = w[cv + 2]; if (v4) *v4 = t; if (f32) *f32 = c; }
    return kind;
}
// A mesh shader's per-primitive outputs, by location: variables decorated PerPrimitiveEXT (5271) and Location (the translator's MeshStage.cs writes the NVN2 mesh shader's second SPH so)
static uint32_t spv_prim_locs(const uint32_t *w, uint32_t nw) {
    uint32_t bound = w[3], m = 0; uint8_t *pp = calloc(bound, 1); if (!pp) return 0;
    for (uint32_t i = 5; i < nw;) { uint32_t n = w[i] >> 16; if (!n || (w[i] & 0xffff) == 54) break; if ((w[i] & 0xffff) == 71 && n >= 3 && w[i + 2] == 5271 && w[i + 1] < bound) pp[w[i + 1]] = 1; i += n; }
    for (uint32_t i = 5; i < nw;) { uint32_t n = w[i] >> 16; if (!n || (w[i] & 0xffff) == 54) break; if ((w[i] & 0xffff) == 71 && n == 4 && w[i + 2] == 30 && w[i + 1] < bound && pp[w[i + 1]] && w[i + 3] < 32) m |= 1u << w[i + 3]; i += n; }
    free(pp); return m;
}
// The binding numbers (bit = binding & 63) whose images the module takes level numbers of: spv_lod_shift's sites (vkshader_t lod_ops)
static uint64_t spv_lod_bindings(const uint32_t *w, uint32_t nw) {
    uint64_t m = 0; uint8_t *bn = calloc(w[3], 1); if (!bn) return 0;   // 1 + the binding & 63 of a variable, and of the ids taken from it
    for (uint32_t i = 5; i < nw;) { uint32_t n = w[i] >> 16, op = w[i] & 0xffff; if (!n || i + n > nw) break;
        if (op == 71 && n >= 4 && w[i + 2] == 33 && w[i + 1] < w[3]) bn[w[i + 1]] = (uint8_t)(1 + (w[i + 3] & 63));   // OpDecorate Binding
        if ((op == 61 || op == 65 || op == 66 || op == 83 || op == 100 || op == 86) && n >= 4 && w[i + 2] < w[3] && w[i + 3] < w[3] && bn[w[i + 3]]) bn[w[i + 2]] = bn[w[i + 3]];
        if ((op == 105 || op == 106 || op == 103 || (op == 88 && n > 5 && w[i + 5] & 2) || (op == 90 && n > 6 && w[i + 6] & 2) || (op == 95 && n > 5 && w[i + 5] & 2)) && n >= 4 && w[i + 3] < w[3] && bn[w[i + 3]])
            m |= 1ull << (bn[w[i + 3]] - 1);
        i += n; }
    free(bn); return m;
}
// A fragment shader drawn after a mesh shader with per-primitive outputs: Vulkan matches such an output only with an input decorated PerPrimitiveEXT, and that decoration
// belongs to SPV_EXT_mesh_shader, which takes SPIR-V 1.4. The translator writes SPIR-V 1.3, so the variant becomes 1.4 (version, the capability MeshShadingEXT and the extension,
// every global variable in the entry point's interface, its storage buffers moved from Uniform + BufferBlock, which 1.4 has no more, to StorageBuffer + Block, and the access
// chains into them retyped) and decorates the inputs at the locations in mask PerPrimitiveEXT, without interpolation decorations (a primitive's value is one for all its
// fragments). Mario Kart World 1.8.0: 11 of its 19 mesh shaders write attributes per primitive (up to 4). NULL when the module is not as expected
static uint32_t *spv_per_primitive(const uint32_t *w, uint32_t nw, uint32_t mask, uint32_t *onw) {
    uint32_t bound = w[3], fn = 0, ep = 0; enum { NEW = 256 };
    uint8_t *bb = calloc(bound, 1), *sbptr = calloc(bound, 1), *sbvar = calloc(bound, 1), *glob = calloc(bound, 1), *ppvar = calloc(bound, 1), *inp = calloc(bound, 1);
    uint32_t *ptr_pointee = calloc(bound, 4), *ptr_class = calloc(bound, 4), *sbof = calloc(bound, 4), *loc = calloc(bound, 4), newt[NEW][2], nnew = 0;   // newt: {pointee, id} of StorageBuffer pointers made
    uint32_t *o = NULL;
    if (!bb || !sbptr || !sbvar || !glob || !ppvar || !inp || !ptr_pointee || !ptr_class || !sbof || !loc) goto done;
    for (uint32_t i = 5; i < nw;) { uint32_t n = w[i] >> 16, op = w[i] & 0xffff; if (!n || i + n > nw) goto done;
        if (op == 54) { fn = i; break; }
        if (op == 15) { if (ep) goto done; ep = i; }
        if (op == 71 && n >= 3 && w[i + 2] == 3 && w[i + 1] < bound) bb[w[i + 1]] = 1;   // BufferBlock
        if (op == 71 && n == 4 && w[i + 2] == 30 && w[i + 1] < bound) loc[w[i + 1]] = w[i + 3] + 1;   // Location (+1: 0 = none)
        if (op == 32 && n == 4 && w[i + 1] < bound) { ptr_class[w[i + 1]] = w[i + 2]; ptr_pointee[w[i + 1]] = w[i + 3]; }
        if (op == 59 && n >= 4 && w[i + 2] < bound && w[i + 3] != 7) { glob[w[i + 2]] = 1; if (w[i + 3] == 1) inp[w[i + 2]] = 1; }
        i += n; }
    if (!fn || !ep) goto done;
    for (uint32_t t = 1; t < bound; t++) if (ptr_class[t] == 2 && ptr_pointee[t] < bound && bb[ptr_pointee[t]]) sbptr[t] = 1;   // Uniform pointers to BufferBlock structures
    for (uint32_t i = 5; i < fn;) { uint32_t n = w[i] >> 16; if ((w[i] & 0xffff) == 59 && w[i + 1] < bound && sbptr[w[i + 1]] && w[i + 2] < bound) sbvar[w[i + 2]] = 1; i += n; }
    for (uint32_t v = 1; v < bound; v++) if (inp[v] && loc[v] && loc[v] <= 32 && (mask >> (loc[v] - 1) & 1)) ppvar[v] = 1;
    // the access chains into the storage buffers: their pointer types in StorageBuffer, made once per pointee (an existing one is taken)
    for (uint32_t i = fn; i < nw;) { uint32_t n = w[i] >> 16, op = w[i] & 0xffff; if (!n) goto done;
        if ((op == 65 || op == 66) && n >= 4 && w[i + 3] < bound && w[i + 2] < bound && (sbvar[w[i + 3]] || sbof[w[i + 3]])) {
            uint32_t pointee = ptr_pointee[w[i + 1]], id = 0;
            for (uint32_t t = 1; t < bound && !id; t++) if (ptr_class[t] == 12 && ptr_pointee[t] == pointee) id = t;
            for (uint32_t k = 0; k < nnew && !id; k++) if (newt[k][0] == pointee) id = newt[k][1];
            if (!id) { if (nnew == NEW) goto done; id = bound + nnew; newt[nnew][0] = pointee; newt[nnew][1] = id; nnew++; }
            sbof[w[i + 2]] = id; }
        i += n; }
    uint32_t nglob = 0, nppv = 0; for (uint32_t v = 1; v < bound; v++) { nglob += glob[v]; nppv += ppvar[v]; }
    o = malloc(((size_t)nw + 16 + nglob + 4 * nnew + 3 * nppv) * 4); if (!o) goto done;
    uint32_t no = 5; memcpy(o, w, 20); o[1] = 0x00010400; o[3] = bound + nnew;
    int capped = 0;
    for (uint32_t i = 5; i < nw;) { uint32_t n = w[i] >> 16, op = w[i] & 0xffff;
        if (i == fn) { for (uint32_t k = 0; k < nnew; k++) { o[no++] = 4u << 16 | 32; o[no++] = newt[k][1]; o[no++] = 12; o[no++] = newt[k][0]; } }   // OpTypePointer StorageBuffer
        if (op != 17 && !capped) { o[no++] = 2u << 16 | 17; o[no++] = 5283;   // OpCapability MeshShadingEXT, OpExtension "SPV_EXT_mesh_shader", after the capabilities
            static const char ext[20] = "SPV_EXT_mesh_shader"; o[no++] = 6u << 16 | 10; memcpy(&o[no], ext, 20); no += 5; capped = 1; }
        if (i == ep) {   // the entry point: its interface and every other global variable
            uint32_t at = no; memcpy(&o[no], &w[i], n * 4); no += n; uint8_t *in = calloc(bound, 1); if (!in) { free(o); o = NULL; goto done; }
            uint32_t s = 3; while (s < n && (w[i + s] & 0xff000000u) && (w[i + s] & 0xff) && (w[i + s] & 0xff00) && (w[i + s] & 0xff0000)) s++;   // past the name's words
            for (s++; s < n; s++) if (w[i + s] < bound) in[w[i + s]] = 1;
            for (uint32_t v = 1; v < bound; v++) if (glob[v] && !in[v]) o[no++] = v;
            o[at] = (no - at) << 16 | 15; free(in); i += n; continue; }
        if (op == 71 && n >= 3 && w[i + 2] == 3) { memcpy(&o[no], &w[i], n * 4); o[no + 2] = 2; no += n; i += n; continue; }   // BufferBlock → Block
        if (op == 71 && n >= 3 && w[i + 1] < bound && ppvar[w[i + 1]] && (w[i + 2] == 13 || w[i + 2] == 16 || w[i + 2] == 17)) { i += n; continue; }   // NoPerspective, Centroid, Sample
        if (op == 71 && n == 4 && w[i + 2] == 30 && w[i + 1] < bound && ppvar[w[i + 1]]) { memcpy(&o[no], &w[i], 16); no += 4; o[no++] = 3u << 16 | 71; o[no++] = w[i + 1]; o[no++] = 5271; i += n; continue; }   // + PerPrimitiveEXT
        if (op == 32 && n == 4 && w[i + 1] < bound && sbptr[w[i + 1]]) { memcpy(&o[no], &w[i], 16); o[no + 2] = 12; no += 4; i += n; continue; }   // Uniform → StorageBuffer
        if (op == 59 && n >= 4 && w[i + 2] < bound && sbvar[w[i + 2]]) { memcpy(&o[no], &w[i], n * 4); o[no + 3] = 12; no += n; i += n; continue; }
        if ((op == 65 || op == 66) && n >= 4 && w[i + 2] < bound && sbof[w[i + 2]]) { memcpy(&o[no], &w[i], n * 4); o[no + 1] = sbof[w[i + 2]]; no += n; i += n; continue; }
        memcpy(&o[no], &w[i], n * 4); no += n; i += n; }
    *onw = no;
done:
    free(bb); free(sbptr); free(sbvar); free(glob); free(ppvar); free(inp); free(ptr_pointee); free(ptr_class); free(sbof); free(loc);
    return o;
}
// A vertex shader's outputs start as (0, 0, 0, 1), as on Maxwell: the translator stores only the components the shader writes and Vulkan leaves the rest
// undefined (dmsh_build.py vs_init does the same for the Apple chain, found on Link's Awakening's lighting). Pokémon Legends: Arceus's deferred lighting draws
// with vertex shader 1327af7c2b78070e, which writes out0.xy only, and its pixel shaders divide by in0.w: every lit pixel was wrong (a washed-out picture,
// dark hair and cloth turned white). Each vec4 output with a Location gets that value stored at the entry point's start. NULL when the module is not as expected
static uint32_t *spv_vs_init(const uint32_t *w, uint32_t nw, uint32_t *onw) {
    uint32_t bound = w[3], fn = 0, ep = 0, f32 = 0, v4 = 0, c0 = 0, c1 = 0; uint8_t *ptr = calloc(bound, 1), *loc = calloc(bound, 1); uint32_t vars[32], nv = 0;
    for (uint32_t i = 5; i < nw;) { uint32_t n = w[i] >> 16, op = w[i] & 0xffff; if (!n || i + n > nw) goto fail;
        if (op == 54) { fn = i; break; }
        if (op == 15 && n >= 3 && w[i + 1] == 0) ep = w[i + 2];   // OpEntryPoint Vertex
        if (op == 22 && n == 3 && w[i + 2] == 32) f32 = w[i + 1];
        if (op == 23 && n == 4 && w[i + 2] == f32 && w[i + 3] == 4 && f32) v4 = w[i + 1];
        if (op == 32 && n == 4 && w[i + 2] == 3 && v4 && w[i + 3] == v4 && w[i + 1] < bound) ptr[w[i + 1]] = 1;   // OpTypePointer Output vec4
        if (op == 43 && n == 4 && w[i + 1] == f32 && f32) { if (w[i + 3] == 0) c0 = w[i + 2]; else if (w[i + 3] == 0x3f800000) c1 = w[i + 2]; }
        if (op == 71 && n == 4 && w[i + 2] == 30 && w[i + 1] < bound) loc[w[i + 1]] = 1;   // Location
        i += n; }
    if (!fn || !ep || !v4) goto fail;
    for (uint32_t i = 5; i < fn;) { uint32_t n = w[i] >> 16; if ((w[i] & 0xffff) == 59 && n >= 4 && w[i + 3] == 3 && w[i + 1] < bound && ptr[w[i + 1]] && w[i + 2] < bound && loc[w[i + 2]] && nv < 32) vars[nv++] = w[i + 2]; i += n; }
    if (!nv) goto fail;
    uint32_t at = 0;   // the entry point's first block, after its OpVariables
    for (uint32_t i = fn; i < nw;) { uint32_t n = w[i] >> 16, op = w[i] & 0xffff; if (!n) goto fail;
        if (op == 54 && w[i + 2] == ep) { uint32_t j = i + n; while (j < nw && (w[j] & 0xffff) != 248) j += w[j] >> 16; j += 2;
            while (j < nw && ((w[j] & 0xffff) == 59 || (w[j] & 0xffff) == 8 || (w[j] & 0xffff) == 317)) j += w[j] >> 16; at = j; break; }
        i += n; }
    if (!at || at >= nw) goto fail;
    uint32_t dec[16], nd = 0, id = bound;
    if (!c0) { c0 = id++; dec[nd++] = 4u << 16 | 43; dec[nd++] = f32; dec[nd++] = c0; dec[nd++] = 0; }
    if (!c1) { c1 = id++; dec[nd++] = 4u << 16 | 43; dec[nd++] = f32; dec[nd++] = c1; dec[nd++] = 0x3f800000; }
    uint32_t cv = id++; dec[nd++] = 7u << 16 | 44; dec[nd++] = v4; dec[nd++] = cv; dec[nd++] = c0; dec[nd++] = c0; dec[nd++] = c0; dec[nd++] = c1;   // OpConstantComposite
    uint32_t *o = malloc(((size_t)nw + nd + 3 * nv) * 4), no = 0;
    memcpy(o, w, (size_t)fn * 4); no = fn; o[3] = id;
    memcpy(&o[no], dec, nd * 4); no += nd;
    memcpy(&o[no], &w[fn], (size_t)(at - fn) * 4); no += at - fn;
    for (uint32_t k = 0; k < nv; k++) { o[no++] = 3u << 16 | 62; o[no++] = vars[k]; o[no++] = cv; }   // OpStore
    memcpy(&o[no], &w[at], (size_t)(nw - at) * 4); no += nw - at;
    free(ptr); free(loc); *onw = no; return o;
fail:
    free(ptr); free(loc); return NULL;
}
static uint32_t *spv_add_outputs(const uint32_t *w, uint32_t nw, uint32_t mask, uint32_t *onw) {
    uint32_t fn = nw, ep = 0, lastdec = 0, bound = w[3], f32 = 0, v4 = 0, pv4 = 0;
    for (uint32_t i = 5; i < nw;) { uint32_t n = w[i] >> 16, op = w[i] & 0xffff; if (!n) return NULL; if (op == 54) { fn = i; break; }
        if (op == 15 && !ep) ep = i; if (op == 71 || op == 72) lastdec = i + n;
        if (op == 22 && w[i + 2] == 32) f32 = w[i + 1];
        i += n; }
    if (!ep || !lastdec || !f32) return NULL;
    for (uint32_t i = 5; i < fn;) { if ((w[i] & 0xffff) == 23 && w[i + 2] == f32 && w[i + 3] == 4) v4 = w[i + 1]; i += w[i] >> 16; }
    if (v4) for (uint32_t i = 5; i < fn;) { if ((w[i] & 0xffff) == 32 && w[i + 2] == 3 && w[i + 3] == v4) pv4 = w[i + 1]; i += w[i] >> 16; }
    int k = __builtin_popcount(mask); uint32_t ids[32], nid = 0;
    uint32_t *o = malloc(((size_t)nw + 16 * k + 16) * 4), no = 5; memcpy(o, w, 20);
    for (int l = 0; l < 32; l++) if (mask >> l & 1) ids[nid++] = bound++;
    uint32_t tv4 = v4 ? v4 : bound++, tpv4 = pv4 ? pv4 : bound++;
    for (uint32_t i = 5; i < fn;) { uint32_t n = w[i] >> 16;
        if (i == ep) { o[no++] = (n + nid) << 16 | 15; memcpy(&o[no], &w[i + 1], (n - 1) * 4); no += n - 1; memcpy(&o[no], ids, nid * 4); no += nid; }   // the entry point lists them
        else { memcpy(&o[no], &w[i], n * 4); no += n; }
        i += n;
        if (i == lastdec) { uint32_t q = 0; for (int l = 0; l < 32; l++) if (mask >> l & 1) { o[no++] = 4u << 16 | 71; o[no++] = ids[q++]; o[no++] = 30; o[no++] = (uint32_t)l; } }   // Location
    }
    if (!v4) { o[no++] = 4u << 16 | 23; o[no++] = tv4; o[no++] = f32; o[no++] = 4; }
    if (!pv4) { o[no++] = 4u << 16 | 32; o[no++] = tpv4; o[no++] = 3; o[no++] = tv4; }
    for (uint32_t q = 0; q < nid; q++) { o[no++] = 4u << 16 | 59; o[no++] = tpv4; o[no++] = ids[q]; o[no++] = 3; }   // OpVariable Output
    memcpy(&o[no], &w[fn], (nw - fn) * 4); no += nw - fn;
    o[3] = bound; *onw = no; return o;
}
static nvn_hmap g_shaders = {.klen = 16};   // {key, stage} → vkshader_t*
static pthread_mutex_t g_sh_mu = PTHREAD_MUTEX_INITIALIZER;
// A shader's SPIR-V is not kept once its module is made (s->spv NULL, s->nspv its length): a variant (vk_shader_module_: integer-texel views, outputs added, fragment depth, alpha test,
// dual source, the mesh A/B, LOD shift) rebuilds the words from the mapped bundle (shader_words: the same words every time), makes its module from them and frees them. Kept, each
// shader's rewritten copy was host heap for the whole run, swapped with the rest: Mario Kart World on the phone (16 GB, near the low-memory killer) kept 286 MB of it for 4,813 shaders,
// where a variant is made once per shader and variant (the cache below). Kept still: every shader's with DM_VK_SPV_KEEP=1 (start-up, the old behaviour), and those another file reads
// (spv_kept_for_readers)
static uint64_t g_spv_rebuilds;   // variants made from rebuilt words (under g_sh_mu)
static int spv_keep_all(void) { static int on = -1; if (on < 0) { const char *e = getenv("DM_VK_SPV_KEEP"); on = e && atoi(e); } return on; }
// (gfx_vk_core.c vk_mem_log, the census) the shaders made so far, the SPIR-V they keep (s->spv, rewritten for its set; the driver's module holds another copy), their variants; o[3] and a
// line of its own (the census line of gfx_vk_core.c prints o[0..2]): the shaders whose words were dropped, and how many variants rebuilt them
void vk_shader_mem(uint64_t o[4]) {
    uint64_t kept = 0, kb = 0, dropped = 0, db = 0, rb;
    pthread_mutex_lock(&g_sh_mu);
    for (uint32_t i = 0; i < g_shaders.cap; i++) { vkshader_t *s = g_shaders.vals[i]; if (!s || s == (void *)1) continue; o[0]++; o[2] += (uint64_t)s->nvar;
        if (s->spv) { kept++; kb += (uint64_t)s->nspv * 4; } else { dropped++; db += (uint64_t)s->nspv * 4; } }
    o[1] += kb; rb = g_spv_rebuilds; o[3] += rb;
    pthread_mutex_unlock(&g_sh_mu);
    VKLOG("memory census: shaders' SPIR-V kept for %llu (%.1f MB), dropped once their module was made for %llu (%.1f MB not held; DM_VK_SPV_KEEP=1 keeps them), rebuilt from the bundle for %llu variants",
          (unsigned long long)kept, kb / 1048576.0, (unsigned long long)dropped, db / 1048576.0, (unsigned long long)rb);
}
// Shared layouts keep pipeline layouts and descriptor bindings reusable across shaders; g_sh_mu protects this cache too.
static VkDescriptorSetLayout shader_layout(const VkDescriptorSetLayoutBinding *b, uint32_t n) {
    struct { uint32_t n, b[96][4]; } k = {.n = n};
    static nvn_hmap cache = {.klen = sizeof k};
    for (uint32_t i = 0; i < n; i++) {
        uint32_t j = i;
        while (j && k.b[j - 1][0] > b[i].binding) { memcpy(k.b[j], k.b[j - 1], sizeof k.b[j]); j--; }
        k.b[j][0] = b[i].binding; k.b[j][1] = b[i].descriptorType; k.b[j][2] = b[i].descriptorCount; k.b[j][3] = b[i].stageFlags;
    }
    VkDescriptorSetLayout layout = nvn_hm_get(&cache, &k);
    if (!layout) {
        VkDescriptorSetLayoutCreateInfo li = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, NULL, 0, n, b};
        VKCHK(vkCreateDescriptorSetLayout(V.dev, &li, NULL, &layout));
        void vk_dsl_register(VkDescriptorSetLayout l, const VkDescriptorSetLayoutBinding *b, uint32_t n); vk_dsl_register(layout, b, n);
        nvn_hm_put(&cache, &k, layout);
    }
    return layout;
}
// A shader's SPIR-V as its module is made from it: the bundle's words copied and rewritten for the device. The same words at every call: only the bundle (mapped read-only, never
// unmapped), the shader's stage and set and the device's facts decide them. first (vk_shader): also the facts read from them into s (xfb and its strides, depth_repl, in_locs / out_locs,
// prim_locs, col0, lod_ops) and the NDSR line; else (a variant's rebuild, vk_shader_module_) s is not written. Under g_sh_mu either way. NULL without memory; *onw its length in words
static uint32_t *shader_words(vkshader_t *s, const bent *e, uint32_t *onw, int first) {
    const int stage = s->stage; uint32_t nw = e->spv_len / 4; uint32_t *w = malloc(e->spv_len); if (!w) return NULL; memcpy(w, g_bundle + e->spv_off, e->spv_len);
    int f64_used = 0, xfb = 0, depth_repl = 0; uint32_t xb = 4, stride[4] = {0};
    for (uint32_t i = 5; i < nw;) { uint32_t n = w[i] >> 16, op = w[i] & 0xffff; if (!n) break; if (op == 71 && n >= 4 && w[i + 2] == 34) w[i + 3] = (uint32_t)s->set;   // OpDecorate DescriptorSet
        if (op == 22 && n == 3 && w[i + 2] == 64) f64_used = 1;   // OpTypeFloat 64
        if (op == 16 && n >= 3 && w[i + 2] == 11) xfb = 1;        // OpExecutionMode Xfb: a vertex shader that captures transform feedback (shader_vk.py xfb_layout)
        if (op == 16 && n >= 3 && (w[i + 2] == 12 || w[i + 2] == 14 || w[i + 2] == 15)) depth_repl = 1;   // OpExecutionMode DepthReplacing / DepthGreater / DepthLess
        // each buffer's stride: the translator decorates an output XfbBuffer b, then XfbStride s (OpDecorate, or OpMemberDecorate for gl_Position's block)
        { uint32_t d = op == 71 && n >= 4 ? w[i + 2] : op == 72 && n >= 5 ? w[i + 3] : 0, v = op == 71 ? w[i + 3] : op == 72 && n >= 5 ? w[i + 4] : 0;
          if (d == 36) xb = v; else if (d == 37 && xb < 4) stride[xb] = v; }
        i += n; }
    // without VK_EXT_transform_feedback the capture declarations would make the module invalid: they are dropped (the capture's draws then write nothing, as before)
    if (xfb && !V.has_xfb) { uint32_t o = 5;
        for (uint32_t i = 5; i < nw;) { uint32_t n = w[i] >> 16, op = w[i] & 0xffff; if (!n || i + n > nw) { while (i < nw) w[o++] = w[i++]; break; }
            int drop = (op == 17 && n == 2 && w[i + 1] == 53) || (op == 16 && n >= 3 && w[i + 2] == 11) ||   // OpCapability TransformFeedback, OpExecutionMode Xfb
                       (op == 71 && n >= 3 && (w[i + 2] == 36 || w[i + 2] == 37)) || (op == 72 && n >= 4 && (w[i + 3] == 36 || w[i + 3] == 37));   // XfbBuffer / XfbStride
            if (drop) i += n; else for (uint32_t k = 0; k < n; k++) w[o++] = w[i++]; }
        nw = o; xfb = 0; if (first) VKRED_ONCE(90, "the device has no transform feedback: shaders that capture it run without"); }
    // the translator declares Float64 in every shader; the device has shaderFloat64 off (Adreno has none), which makes the declaration invalid: when the
    // shader has no 64-bit float type it becomes a second Shader declaration
    if (!f64_used) for (uint32_t i = 5; i < nw;) { uint32_t n = w[i] >> 16, op = w[i] & 0xffff; if (!n || op != 17) break; if (n == 2 && w[i + 1] == 10) w[i + 1] = 1; i += n; }
    // the translator declares fragment shader interlock in every pixel shader; the device has no VK_EXT_fragment_shader_interlock: when the shader uses no
    // interlock instruction, its OpCapability and OpExtension are dropped
    { int used = 0; for (uint32_t i = 5; i < nw;) { uint32_t n = w[i] >> 16, op = w[i] & 0xffff; if (!n) break; if (op == 5364 || op == 5365) used = 1; i += n; }   // OpBegin/EndInvocationInterlockEXT
      if (!used) { uint32_t o = 5;
        for (uint32_t i = 5; i < nw;) { uint32_t n = w[i] >> 16, op = w[i] & 0xffff; if (!n || i + n > nw) { while (i < nw) w[o++] = w[i++]; break; }
            int drop = (op == 17 && n == 2 && (w[i + 1] == 5378 || w[i + 1] == 5363 || w[i + 1] == 5372)) ||   // FragmentShader{Pixel,Sample,ShadingRate}InterlockEXT
                       (op == 10 && !memcmp(&w[i + 1], "SPV_EXT_fragment_shader_interlock", 33));
            if (drop) i += n; else for (uint32_t k = 0; k < n; k++) w[o++] = w[i++]; }
        nw = o; } }
    if (stage == 0) { uint32_t n2 = 0, *w2 = spv_vs_init(w, nw, &n2); if (w2) { free(w); w = w2; nw = n2; } }
    *onw = nw;
    if (!first) return w;
    s->xfb = xfb; memcpy(s->xfb_stride, stride, sizeof stride); s->depth_repl = depth_repl;
    spv_io_locs(w, nw, &s->in_locs, &s->out_locs);
    if (stage == 7) s->prim_locs = spv_prim_locs(w, nw);
    if (stage == 1) s->col0 = spv_col0(w, nw, NULL, NULL, NULL);   // (the alpha test: gfx_vk_exec.c pkey atest)
    { uint64_t lb = spv_lod_bindings(w, nw); s->lod_ops = 0; for (int i = 0; i < s->nbind && i < 64; i++) if (lb >> (s->b[i].binding & 63) & 1) s->lod_ops |= 1ull << i; }   // (LOD shifts: vk_lodshift_mask)
    // A compute shader that runs Ampere's tensor instructions per lane (the translator's HMMA helper dmsh_hmma16816_f16: 64 FP32 FMAs and 64 conversions a lane an HMMA; LDSM / MOVM
    // through subgroup shuffles): agl's NDSR upscaler network, whose 30 shaders Mario Kart World 1.8.0 loads at boot and the game engages as its render size falls (PARITY G4). Said
    // once a shader, at its load: its first dispatch, the bundle having more compute shaders than compute_warm makes ahead
    if (stage == 5) { uint32_t hf = 0, calls = 0, shuf = 0;
        for (uint32_t i = 5; i < nw;) { uint32_t n = w[i] >> 16, op = w[i] & 0xffff; if (!n || i + n > nw) break;
            if (op == 5 && n >= 7 && !memcmp(&w[i + 2], "dmsh_hmma16816_f16", 18)) hf = w[i + 1];   // OpName
            if (op == 57 && n >= 4 && hf && w[i + 3] == hf) calls++; if (op == 345) shuf++;   // OpFunctionCall, OpGroupNonUniformShuffle
            i += n; }
        if (calls) VKLOG("compute shader %016llx: Ampere tensor instructions emulated per lane (%u HMMA, %u subgroup shuffles): agl's NDSR upscaler network, in use (DM_VK_GPUPROF: the stretch its dispatch falls in)",
                         (unsigned long long)s->key, calls, shuf); }
    return w;
}
// The words other files read from s->spv stay, their readers taking NULL as no code: a task or mesh shader's (gfx_vk_exec.c spv_model, its execution model, and mesh_need, what the device
// must take: Mario Kart World's 7 task and 18 mesh shaders), a game upscaler shader's (gfx_vk_fsr.c ndsr_shape_of, the NDSR swap's shape read from its code: DM_GAME_UPSCALER_KEYS, its 30)
static int spv_kept_for_readers(const vkshader_t *s, const uint32_t *w, uint32_t nw) {
    uint32_t model = ~0u;
    for (uint32_t i = 5; i < nw;) { uint32_t n = w[i] >> 16, op = w[i] & 0xffff; if (!n || op == 54) break; if (op == 15 && n >= 3) { model = w[i + 1]; break; } i += n; }   // (as spv_model)
    if (s->stage == 6 || s->stage == 7 || model == 5364 || model == 5365) return 1;   // TaskEXT, MeshEXT
#ifdef DM_GAME_UPSCALER_KEYS
    for (const char *k = DM_GAME_UPSCALER_KEYS; strnlen(k, 16) == 16; k += 17) { char b[17]; memcpy(b, k, 16); b[16] = 0; if (strtoull(b, NULL, 16) == s->key) return 1; if (k[16] != ',') break; }   // (as ndsr_named)
#endif
    return 0;
}
vkshader_t *vk_shader(uint64_t key, int stage) {
    bundle_auto();
    struct { uint64_t key; uint64_t stage; } hk = {key, (uint64_t)stage};
    pthread_mutex_lock(&g_sh_mu);
    vkshader_t *s = nvn_hm_get(&g_shaders, &hk);
    if (s) { pthread_mutex_unlock(&g_sh_mu); return s == (void *)1 ? NULL : s; }
    const bent *e = g_bundle ? find(key) : NULL;
    if (!e) { nvn_hm_put(&g_shaders, &hk, (void *)1); pthread_mutex_unlock(&g_sh_mu);
              static int n; if (n++ < 16) VKLOG("[RED] shader %016llx (stage %d) is not in the bundle", (unsigned long long)key, stage); return NULL; }
    // a task or mesh shader takes the bundle's stage (6, 7), also when asked by the stage its bindings are recorded at (hle_nvn.c st_host: 3, 4)
    if (e->stage == 6 || e->stage == 7) stage = (int)e->stage;
    s = calloc(1, sizeof *s); s->key = key; s->stage = stage; s->set = set_of(stage);
    // reflection
    const uint8_t *r = g_bundle + e->refl_off; uint32_t hdr[5]; memcpy(hdr, r, 20);
    s->nbind = (int)hdr[0]; s->c1_len = hdr[1]; memcpy(s->local, hdr + 2, 12);
    s->b = (const dmsv_bind *)(r + 20); s->c1 = r + 20 + 20 * s->nbind;
    // after c1, "FSTR": the stores only a dispatch's first invocation made, taken out of the shader (shader_vk.py first_store), {i32 slot, u32 word, u32 value} each: the replay makes them;
    // then "ORDR": the shader allocates through a storage-buffer atomic (shader_vk.py ordered_alloc), so a small dispatch of it runs its workgroups in order (gfx_vk_exec.c dispatch)
    { uint32_t at = 20 + 20 * (uint32_t)s->nbind + s->c1_len, n = 0;
      if (e->refl_len >= at + 8 && !memcmp(r + at, "FSTR", 4)) { memcpy(&n, r + at + 4, 4);
          for (uint32_t i = 0; i < n && i < 4 && at + 8 + 12 * (i + 1) <= e->refl_len; i++) { memcpy(&s->first[i], r + at + 8 + 12 * i, 12); s->nfirst = (int)i + 1; }
          at += 8 + 12 * n; }
      if (e->refl_len >= at + 8 && !memcmp(r + at, "ORDR", 4)) { memcpy(&n, r + at + 4, 4); s->ordered = n == 1; } }
    // SPIR-V with the stage's set, and the facts read from it (shader_words)
    uint32_t nw = 0, *w = shader_words(s, e, &nw, 1);
    VkShaderModuleCreateInfo ci = {VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, NULL, 0, (size_t)nw * 4, w};
    VkResult vr = w ? vkCreateShaderModule(V.dev, &ci, NULL, &s->mod) : VK_ERROR_OUT_OF_HOST_MEMORY;
    if (vr) { VKLOG("[RED] shader %016llx: vkCreateShaderModule %d", (unsigned long long)key, vr); free(w); free(s); nvn_hm_put(&g_shaders, &hk, (void *)1); pthread_mutex_unlock(&g_sh_mu); return NULL; }
    // the words go now that the driver has its module: a variant rebuilds them (vk_shader_module_); DM_VK_SPV_KEEP=1, and the shaders other files read the words of, keep them
    s->nspv = nw; if (spv_keep_all() || spv_kept_for_readers(s, w, nw)) s->spv = w; else free(w);
    // descriptor set layout: one entry per binding
    // The game's constant buffers and the driver segment are dynamic uniform buffers: their offset is given when the set is bound, so a set matches any draw's
    // with the same buffers (Unity updates a constant buffer before nearly every draw, and each update is a shadow at a new offset: NEO made a new set for
    // 94% of its draws, 5 ms a frame). Up to half the device's limit per stage, as the vertex and fragment sets share a pipeline layout. DM_VK_DYN_UBO=0: none.
    static int dyn_on = -1; if (dyn_on < 0) { const char *e = getenv("DM_VK_DYN_UBO"); dyn_on = !(e && !atoi(e)); }
    uint32_t dynmax = dyn_on ? V.props.limits.maxDescriptorSetUniformBuffersDynamic / 2 : 0;
    VkDescriptorSetLayoutBinding lb[96]; uint32_t nl = 0;
    for (int i = 0; i < s->nbind && nl < 96; i++) {
        const dmsv_bind *b = &s->b[i];
        int dyn = b->desc == RD_UBO && (b->kind == RK_CBUF || b->kind == RK_DRV) && b->count == 1 && (uint32_t)s->ndyn < dynmax;
        s->dyn[i] = (uint8_t)dyn; s->ndyn += dyn;
        lb[nl++] = (VkDescriptorSetLayoutBinding){b->binding, dyn ? VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC : desc_type(b->desc), b->count ? b->count : 32, stage_flag(stage), NULL};
    }
    s->dsl = shader_layout(lb, nl);
    nvn_hm_put(&g_shaders, &hk, s);
    pthread_mutex_unlock(&g_sh_mu);
    return s;
}
// Integer texels. Maxwell's texture instructions return the texel's 32-bit words as they are, and the shader decides what they mean; the translator declares
// every sampled image as float, which matches only float views. A view of an integer format, or the stencil of a depth-stencil texture (DepthStencilMode
// STENCIL), must be read through a uint / int image in Vulkan, and Adreno converts the value otherwise. Xenoblade Chronicles 3 tags each pixel's material
// for its lighting from the stencil's bit 3 (shader 094f4afb9c876459 tests the raw word & 8): read as float, every tag was 0 and the lit scene was black.
// A variant of the module retypes those bindings' images to uint / int and bitcasts each texel read back to float, which gives the shader the raw words.
typedef struct { const uint32_t *w; uint32_t fn, bound, *at, *nt, nnt; struct { uint32_t old, kind, id; } map[64]; int nmap; } spvi;
static uint32_t spvi_find(const uint32_t *a, uint32_t lo, uint32_t hi, uint32_t op, const uint32_t *ops, uint32_t nops) {   // a declaration op with these operands (after the result id)
    for (uint32_t j = lo; j < hi;) { uint32_t m = a[j] >> 16; if (!m) break; if ((a[j] & 0xffff) == op && m == nops + 2 && !memcmp(&a[j + 2], ops, nops * 4)) return a[j + 1]; j += m; }
    return 0;
}
static uint32_t spvi_decl(spvi *c, uint32_t op, const uint32_t *ops, uint32_t nops, int reuse) {   // an existing declaration, or a new one
    uint32_t id = reuse ? spvi_find(c->w, 5, c->fn, op, ops, nops) : 0; if (!id && reuse) id = spvi_find(c->nt, 0, c->nnt, op, ops, nops);
    if (id || c->nnt + nops + 2 > 4096) return id;
    id = c->bound++; c->nt[c->nnt++] = (nops + 2) << 16 | op; c->nt[c->nnt++] = id; memcpy(&c->nt[c->nnt], ops, nops * 4); c->nnt += nops; return id;
}
static uint32_t spvi_scalar(spvi *c, uint32_t k) { uint32_t o[2] = {32, k - 1}; return spvi_decl(c, 21, o, 2, 1); }   // OpTypeInt 32, unsigned (kind 1) / signed (2)
static uint32_t spvi_vec4(spvi *c, uint32_t k) { uint32_t o[2] = {spvi_scalar(c, k), 4}; return spvi_decl(c, 23, o, 2, 1); }
// the type for old with its image's sampled type made uint / int: images, sampled images, arrays of them, pointers to those
static uint32_t spvi_type(spvi *c, uint32_t old, uint32_t k) {
    for (int j = 0; j < c->nmap; j++) if (c->map[j].old == old && c->map[j].kind == k) return c->map[j].id;
    uint32_t i = old < c->bound ? c->at[old] : 0; if (!i) return 0;
    uint32_t n = c->w[i] >> 16, op = c->w[i] & 0xffff, d[12]; if (n > 12 || n < 3) return 0; memcpy(d, &c->w[i + 2], (n - 2) * 4);
    if (op == 25) d[0] = spvi_scalar(c, k);                                                                          // OpTypeImage: the sampled type
    else if (op == 27 || op == 28 || op == 29) { if (!(d[0] = spvi_type(c, d[0], k))) return 0; }                  // OpTypeSampledImage, OpTypeArray, OpTypeRuntimeArray
    else if (op == 32 && n == 4) { if (!(d[1] = spvi_type(c, d[1], k))) return 0; }                                // OpTypePointer
    else return 0;
    uint32_t id = spvi_decl(c, op, d, n - 2, op != 28 && op != 29); if (!id) return 0;
    if (c->nmap < 64) c->map[c->nmap++] = (typeof(c->map[0])){old, k, id};
    return id;
}
static uint32_t *spv_int_images(const uint32_t *w, uint32_t nw, const uint16_t *bind, const uint8_t *kind, int nb, uint32_t *onw) {
    spvi c = {.w = w, .fn = nw, .bound = w[3]};
    for (uint32_t i = 5; i < nw;) { uint32_t n = w[i] >> 16; if (!n) return NULL; if ((w[i] & 0xffff) == 54) { c.fn = i; break; } i += n; }   // the first OpFunction
    uint32_t base = c.bound; c.at = calloc(base, sizeof *c.at); c.nt = malloc(4096 * 4);
    uint8_t *taint = calloc(base, 1);   // ids that are (pointers to) the retyped images: their kind
    for (uint32_t i = 5; i < c.fn;) { uint32_t n = w[i] >> 16, op = w[i] & 0xffff;
        if (op >= 19 && op <= 39 && w[i + 1] < base) c.at[w[i + 1]] = i;   // OpType*
        if (op == 71 && n >= 4 && w[i + 2] == 33) for (int k = 0; k < nb; k++) if (w[i + 3] == bind[k] && w[i + 1] < base) taint[w[i + 1]] = kind[k];   // OpDecorate Binding
        i += n; }
    // the functions, rewritten apart: every new type they need is known before the declarations are written
    uint32_t *fb = malloc(((size_t)(nw - c.fn) * 2 + 16) * 4), nf = 0, *mv = malloc(1024 * 4), nmv = 0; int ok = 1;
    for (uint32_t i = 5; i < c.fn && ok;) { uint32_t n = w[i] >> 16;   // the retyped variables, with their new pointer types
        if ((w[i] & 0xffff) == 59 && w[i + 2] < base && taint[w[i + 2]]) { uint32_t t = spvi_type(&c, w[i + 1], taint[w[i + 2]]); if (!t || nmv + n > 1024) ok = 0; else { memcpy(&mv[nmv], &w[i], n * 4); mv[nmv + 1] = t; nmv += n; } }
        i += n; }
    for (uint32_t i = c.fn; i < nw && ok;) { uint32_t n = w[i] >> 16, op = w[i] & 0xffff; if (!n) { ok = 0; break; }
        uint32_t *x = &fb[nf]; memcpy(x, &w[i], n * 4); nf += n;
        uint32_t src = n >= 4 ? w[i + 3] : 0, k = src && src < base ? taint[src] : 0;
        if (k && (op == 61 || op == 65 || op == 66 || op == 83 || op == 100 || op == 86)) {   // Load, AccessChain, InBoundsAccessChain, CopyObject, Image, SampledImage
            uint32_t t = spvi_type(&c, x[1], k); if (!t) { ok = 0; break; } x[1] = t; if (x[2] < base) taint[x[2]] = (uint8_t)k;
        } else if (k && (op == 87 || op == 88 || op == 91 || op == 92 || op == 95 || op == 96)) {   // Sample{Implicit,Explicit}Lod, SampleProj*, Fetch, Gather: a vec4 of the integer type, bitcast back
            uint32_t ft = x[1], r = x[2], r2 = c.bound++; x[1] = spvi_vec4(&c, k); x[2] = r2;
            fb[nf++] = 4u << 16 | 124; fb[nf++] = ft; fb[nf++] = r; fb[nf++] = r2;   // OpBitcast
        }
        i += n; }
    uint32_t *o = NULL, no = 5;
    if (ok && c.nnt < 4096) {
        o = malloc(((size_t)c.fn + c.nnt + nmv + nf + 8) * 4); memcpy(o, w, 20); o[3] = c.bound;
        for (uint32_t i = 5; i < c.fn;) { uint32_t n = w[i] >> 16; if (!((w[i] & 0xffff) == 59 && w[i + 2] < base && taint[w[i + 2]])) { memcpy(&o[no], &w[i], n * 4); no += n; } i += n; }
        memcpy(&o[no], c.nt, c.nnt * 4); no += c.nnt; memcpy(&o[no], mv, nmv * 4); no += nmv; memcpy(&o[no], fb, nf * 4); no += nf;
        *onw = no;
    }
    free(c.at); free(c.nt); free(taint); free(fb); free(mv);
    return o;
}
// A variant of a fragment shader that writes gl_FragDepth = gl_FragCoord.z before each return of its entry point (DepthReplacing): the same depth, tested after the
// shader instead of by Adreno's early culling (gfx_vk_exec.c fdz says when). NULL when the shader writes its depth already or the module is not as expected
static uint32_t *spv_frag_depth(const uint32_t *w, uint32_t nw, uint32_t *onw) {
    spvi c = {.w = w, .fn = nw, .bound = w[3]};
    uint32_t ep = 0, epfn = 0, firsttype = 0, coord = 0, c2 = 0;
    for (uint32_t i = 5; i < nw;) { uint32_t n = w[i] >> 16, op = w[i] & 0xffff; if (!n || i + n > nw) return NULL;
        if (op == 15) { if (ep || n < 3 || w[i + 1] != 4) return NULL; ep = i; epfn = w[i + 2]; }   // one entry point, a Fragment one
        if (op == 71 && n >= 4 && w[i + 2] == 11 && w[i + 3] == 22) return NULL;   // BuiltIn FragDepth: written already
        if (op == 71 && n >= 4 && w[i + 2] == 11 && w[i + 3] == 15) coord = w[i + 1];   // BuiltIn FragCoord
        if (!firsttype && ((op >= 19 && op <= 39) || (op >= 41 && op <= 52) || op == 59)) firsttype = i;   // the first type, constant or variable: the annotations end here
        if (op == 54) { c.fn = i; break; }   // the first OpFunction
        i += n; }
    if (!ep || !firsttype || c.fn == nw) return NULL;
    c.nt = malloc(4096 * 4);
    uint32_t o32[1] = {32}; uint32_t f32 = spvi_decl(&c, 22, o32, 1, 1), i32 = spvi_scalar(&c, 2);
    uint32_t ov4[2] = {f32, 4}; uint32_t v4 = spvi_decl(&c, 23, ov4, 2, 1);
    uint32_t op1[2] = {1, v4}, op2[2] = {1, f32}, op3[2] = {3, f32}; uint32_t pin4 = spvi_decl(&c, 32, op1, 2, 1), pinf = spvi_decl(&c, 32, op2, 2, 1), poutf = spvi_decl(&c, 32, op3, 2, 1);
    if (!f32 || !i32 || !v4 || !pin4 || !pinf || !poutf || c.nnt + 16 > 4096) { free(c.nt); return NULL; }
    for (uint32_t i = 5; i < c.fn;) { uint32_t n = w[i] >> 16; if ((w[i] & 0xffff) == 43 && n == 4 && w[i + 1] == i32 && w[i + 3] == 2) c2 = w[i + 2]; i += n; }   // OpConstant int 2
    if (!c2) { c2 = c.bound++; c.nt[c.nnt++] = 4u << 16 | 43; c.nt[c.nnt++] = i32; c.nt[c.nnt++] = c2; c.nt[c.nnt++] = 2; }
    int newcoord = !coord; if (newcoord) { coord = c.bound++; c.nt[c.nnt++] = 4u << 16 | 59; c.nt[c.nnt++] = pin4; c.nt[c.nnt++] = coord; c.nt[c.nnt++] = 1; }   // OpVariable Input
    uint32_t dv = c.bound++; c.nt[c.nnt++] = 4u << 16 | 59; c.nt[c.nnt++] = poutf; c.nt[c.nnt++] = dv; c.nt[c.nnt++] = 3;   // OpVariable Output
    uint32_t nret = 0; for (uint32_t i = c.fn; i < nw;) { uint32_t n = w[i] >> 16; if (!n) { free(c.nt); return NULL; } nret += (w[i] & 0xffff) == 253; i += n; }
    uint32_t *o = malloc(((size_t)nw + c.nnt + 16 + 12 * nret) * 4), no = 5; memcpy(o, w, 20);
    for (uint32_t i = 5; i < c.fn;) { uint32_t n = w[i] >> 16;
        if (i == firsttype) { uint32_t d[8] = {4u << 16 | 71, dv, 11, 22, 4u << 16 | 71, coord, 11, 15}; memcpy(&o[no], d, (newcoord ? 8 : 4) * 4); no += newcoord ? 8 : 4; }   // BuiltIn FragDepth (FragCoord)
        if (i == ep) { o[no++] = (n + 1 + newcoord) << 16 | 15; memcpy(&o[no], &w[i + 1], (n - 1) * 4); no += n - 1; o[no++] = dv; if (newcoord) o[no++] = coord;   // the entry point lists them
                       o[no++] = 3u << 16 | 16; o[no++] = epfn; o[no++] = 12; }   // OpExecutionMode DepthReplacing
        else { memcpy(&o[no], &w[i], n * 4); no += n; }
        i += n; }
    memcpy(&o[no], c.nt, c.nnt * 4); no += c.nnt;
    int in_ep = 0;
    for (uint32_t i = c.fn; i < nw;) { uint32_t n = w[i] >> 16, op = w[i] & 0xffff;
        if (op == 54) in_ep = w[i + 2] == epfn; else if (op == 56) in_ep = 0;   // OpFunction / OpFunctionEnd
        if (op == 253 && in_ep) { uint32_t a = c.bound++, b = c.bound++, x[12] = {5u << 16 | 65, pinf, a, coord, c2, 4u << 16 | 61, f32, b, a, 3u << 16 | 62, dv, b};   // AccessChain .z, Load, Store
                                  memcpy(&o[no], x, sizeof x); no += 12; }
        memcpy(&o[no], &w[i], n * 4); no += n; i += n; }
    o[3] = c.bound; *onw = no; free(c.nt); return o;
}
// DM_VK_SPV_DIR=<dir> with DM_VK_SPV_ALT=1 (live, debugging): pipelines made while it is on use <dir>/<key>.spv in place of the bundle's code where the file
// exists (same bindings), so two compilations of a shader can be compared in one run: the translator's module through spirv-opt, its GLSL through glslc...
// Its words are read from the file each time its module is made (once: vk_shader_module_ keeps it as a variant), never kept in s->spv: nothing of them is lost when the
// bundle's words are dropped, and the shader's other variants are made from the bundle's words, rebuilt, as without it
static VkShaderModule alt_module(vkshader_t *s, int which) {   // which 2: DM_VK_SPV_DIR2 (gfx_vk_exec.c DM_VK_X_CSALT=2)
    const char *dir = getenv(which == 2 ? "DM_VK_SPV_DIR2" : "DM_VK_SPV_DIR"); char path[512]; if (!dir) return s->mod;
    snprintf(path, sizeof path, "%s/%016llx.spv", dir, (unsigned long long)s->key); FILE *f = fopen(path, "rb"); if (!f) return s->mod;
    fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET); uint32_t *w = n > 20 && !(n & 3) ? malloc((size_t)n) : NULL; VkShaderModule m = s->mod;
    if (w && fread(w, 1, (size_t)n, f) == (size_t)n) { VkShaderModuleCreateInfo ci = {VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, NULL, 0, (size_t)n, w};
        for (uint32_t i = 5, nw = (uint32_t)(n / 4); i < nw;) { uint32_t k = w[i] >> 16; if (!k) break; if ((w[i] & 0xffff) == 71 && k >= 4 && w[i + 2] == 34) w[i + 3] = (uint32_t)s->set; i += k; }   // the stage's set, as at load
        if (vkCreateShaderModule(V.dev, &ci, NULL, &m)) m = s->mod; else VKLOG("shader %016llx: alternative code from %s", (unsigned long long)s->key, path); }
    free(w); fclose(f); return m;
}
// A fragment shader's dual-source variant (gfx_vk_exec.c pkey dual): its output at location 1 becomes location 0, index 1, the second source of target 0's blend.
// NULL when it has no output at location 1, or outputs past it (one dual-source attachment at most: maxFragmentDualSrcAttachments)
static uint32_t *spv_dual(const uint32_t *w, uint32_t nw, uint32_t *onw) {
    uint32_t bound = w[3], fn = nw, at = 0, v1 = 0; uint8_t *out = calloc(bound, 1); int past = 0;
    for (uint32_t i = 5; i < nw;) { uint32_t n = w[i] >> 16, op = w[i] & 0xffff; if (!n || i + n > nw) { free(out); return NULL; }
        if (op == 54) { fn = i; break; }
        if (op == 59 && n >= 4 && w[i + 3] == 3 && w[i + 2] < bound) out[w[i + 2]] = 1;   // OpVariable ... Output
        i += n; }
    for (uint32_t i = 5; i < fn;) { uint32_t n = w[i] >> 16, op = w[i] & 0xffff;
        if (op == 71 && n == 4 && w[i + 2] == 30 && w[i + 1] < bound && out[w[i + 1]]) { if (w[i + 3] == 1) { at = i; v1 = w[i + 1]; } else if (w[i + 3] > 1) past = 1; }   // Location
        if (op == 71 && n == 4 && w[i + 2] == 32) past = 1;   // an Index already: not a shader this is for
        i += n; }
    free(out);
    if (!at || past) return NULL;
    uint32_t *o = malloc(((size_t)nw + 4) * 4); memcpy(o, w, (size_t)(at + 4) * 4); o[at + 3] = 0;   // Location 1 → 0
    o[at + 4] = 4u << 16 | 71; o[at + 5] = v1; o[at + 6] = 32; o[at + 7] = 1;   // OpDecorate ... Index 1
    memcpy(&o[at + 8], &w[at + 4], (size_t)(nw - at - 4) * 4); *onw = nw + 4;
    return o;
}
// Two variants of a mesh shader to bisect a driver compile (gfx_vk_exec.c DM_VK_MESH_NOTASK / DM_VK_MESH_POSBLOCK; run 4 of Mario Kart World 1.8.0 crashed in the Adreno 840's compiler
// building its first task / mesh pipeline). spv_payload_private: the task payload's pointer types and variables take the Private storage class, for a pipeline made without the task
// stage (the variable stays in the interface, where SPIR-V 1.4 lists every global); the module unchanged when it has none
static uint32_t *spv_payload_private(const uint32_t *w, uint32_t nw, uint32_t *onw) {
    uint32_t *o = malloc((size_t)nw * 4); if (!o) return NULL; memcpy(o, w, (size_t)nw * 4);
    for (uint32_t i = 5; i < nw;) { uint32_t n = w[i] >> 16, op = w[i] & 0xffff; if (!n || i + n > nw) { free(o); return NULL; }
        if (op == 54) break;   // (the first function: the declarations end)
        if (op == 32 && n == 4 && w[i + 2] == 5402) o[i + 2] = 6;   // OpTypePointer TaskPayloadWorkgroupEXT → Private
        if (op == 59 && n >= 4 && w[i + 3] == 5402) o[i + 3] = 6;   // OpVariable
        i += n; }
    *onw = nw; return o;
}
// spv_pos_block: gl_Position in a per-vertex block, as glslang declares gl_MeshVerticesEXT (the translator writes a standalone vec4 array decorated BuiltIn Position, which drivers meet
// less): the variable takes an array of a Block struct whose one vec4 member is BuiltIn Position, its access chains member 0 after the vertex index. NULL when the module is not as
// expected (no such Output array, a use of it other than an access chain, an initializer, no 32-bit integer type)
static uint32_t *spv_pos_block(const uint32_t *w, uint32_t nw, uint32_t *onw) {
    uint32_t bound = w[3], fn = nw, pv = 0, dec = 0, var = 0, pt = 0, arr = 0, v4 = 0, len = 0, i32 = 0, c0 = 0, nac = 0;
    for (uint32_t i = 5; i < nw;) { uint32_t n = w[i] >> 16, op = w[i] & 0xffff; if (!n || i + n > nw) return NULL; if (op == 54) { fn = i; break; }
        if (op == 71 && n == 4 && w[i + 2] == 11 && w[i + 3] == 0) { pv = w[i + 1]; dec = i; }   // OpDecorate BuiltIn Position
        if (op == 21 && n == 4 && w[i + 2] == 32 && !i32) i32 = w[i + 1];   // OpTypeInt 32
        i += n; }
    if (!pv || fn == nw) return NULL;
    for (uint32_t i = 5; i < fn;) { uint32_t n = w[i] >> 16, op = w[i] & 0xffff;
        if (op == 59 && w[i + 2] == pv) { if (n != 4 || w[i + 3] != 3) return NULL; var = i; pt = w[i + 1]; }   // OpVariable Output, no initializer
        if (op == 43 && n == 4 && w[i + 3] == 0 && !c0) for (uint32_t j = 5; j < fn; j += w[j] >> 16) if ((w[j] & 0xffff) == 21 && w[j + 1] == w[i + 1] && w[j + 2] == 32) { c0 = w[i + 2]; break; }   // OpConstant 0 of an int32
        i += n; }
    for (uint32_t i = 5; var && i < fn;) { uint32_t n = w[i] >> 16, op = w[i] & 0xffff;
        if (op == 32 && w[i + 1] == pt && w[i + 2] == 3) arr = w[i + 3];   // OpTypePointer Output
        i += n; }
    for (uint32_t i = 5; arr && i < fn;) { uint32_t n = w[i] >> 16, op = w[i] & 0xffff; if (op == 28 && w[i + 1] == arr) { v4 = w[i + 2]; len = w[i + 3]; } i += n; }   // OpTypeArray
    { int ok = 0; for (uint32_t i = 5; v4 && i < fn; i += w[i] >> 16) if ((w[i] & 0xffff) == 23 && w[i + 1] == v4 && w[i + 3] == 4) ok = 1; if (!ok || !len || (!c0 && !i32)) return NULL; }   // OpTypeVector of 4
    for (uint32_t i = fn; i < nw;) { uint32_t n = w[i] >> 16, op = w[i] & 0xffff, uses = 0; if (!n || i + n > nw) return NULL;   // its uses: access chains with a vertex index, nothing else
        for (uint32_t k = 1; k < n; k++) uses += w[i + k] == pv;
        if (uses && !((op == 65 || op == 66) && n >= 5 && w[i + 3] == pv && uses == 1)) return NULL;
        nac += uses; i += n; }
    uint32_t blk = bound, arr2 = bound + 1, pt2 = bound + 2, nc = c0 ? 0 : 4; if (!c0) c0 = bound + 3;
    uint32_t *o = malloc(((size_t)nw + 24 + nac) * 4), no = 5; if (!o) return NULL;   // (+4 decorations, +11 types, +4 a constant, a word a chain)
    memcpy(o, w, 20); o[3] = bound + 3 + (nc ? 1 : 0);
    for (uint32_t i = 5; i < nw;) { uint32_t n = w[i] >> 16, op = w[i] & 0xffff;
        if (i == dec) { uint32_t d[8] = {3u << 16 | 71, blk, 2, 5u << 16 | 72, blk, 0, 11, 0}; memcpy(&o[no], d, sizeof d); no += 8; i += n; continue; }   // Block; member 0 BuiltIn Position
        if (i == var) { uint32_t d[15] = {3u << 16 | 30, blk, v4, 4u << 16 | 28, arr2, blk, len, 4u << 16 | 32, pt2, 3, arr2, 4u << 16 | 59, pt2, pv, 3}; memcpy(&o[no], d, sizeof d); no += 15; i += n; continue; }
        if (i == fn && nc) { o[no++] = 4u << 16 | 43; o[no++] = i32; o[no++] = c0; o[no++] = 0; }   // OpConstant 0 (before the functions: after every type)
        if ((op == 65 || op == 66) && i >= fn && w[i + 3] == pv) { o[no++] = (n + 1) << 16 | op; memcpy(&o[no], &w[i + 1], 4 * 4); no += 4; o[no++] = c0; memcpy(&o[no], &w[i + 5], (n - 5) * 4); no += n - 5; i += n; continue; }
        memcpy(&o[no], &w[i], n * 4); no += n; i += n; }
    *onw = no; return o;
}
// NVN's alpha test (ColorStateSetAlphaTest, SetAlphaRef: Maxwell's fixed-function test of the alpha a pixel shader writes to colour 0, before blending; upstream ee369f09, dmsh_alpha.py's
// <key>_ba libraries for Metal): Vulkan has none. A fragment shader's alpha test variant (gfx_vk_exec.c pkey atest) gets a new entry point, under the old one's id (its execution modes and
// name stay with it), that calls the shader's own main and then compares colour 0's alpha with the function and the reference in the push constant range VK_ALPHA_PUSH_OFF (gfx_vk.h; data
// of each draw, so one pipeline serves every function and reference), and kills the fragment when the comparison fails: after every return of the shader, while a discard of the shader's
// own ends it before. NVN's functions as dmsh_alpha.py dm_alpha_pass has them: NEVER, LESS a < ref, EQUAL a == ref, LEQUAL a <= ref, GREATER a > ref, NOTEQUAL a != ref (true for a NaN),
// GEQUAL a >= ref; any other value passes (ALWAYS: drawn without the variant). Only a float colour 0 (spv_col0): an integer one has no test, as upstream. NULL, with why, when the module is
// not as expected
static uint32_t spvi_const(spvi *c, uint32_t op, uint32_t type, uint32_t value) {   // an OpConstant (43, value) or OpConstantTrue / OpConstantFalse (41 / 42) of type: found, or added to c->nt
    uint32_t n = op == 43 ? 4 : 3;
    for (int pass = 0; pass < 2; pass++) { const uint32_t *a = pass ? c->nt : c->w; uint32_t lo = pass ? 0 : 5, hi = pass ? c->nnt : c->fn;
        for (uint32_t j = lo; j < hi;) { uint32_t m = a[j] >> 16; if (!m) break; if ((a[j] & 0xffff) == op && m == n && a[j + 1] == type && (op != 43 || a[j + 3] == value)) return a[j + 2]; j += m; } }
    if (c->nnt + n > 4096) return 0;
    uint32_t id = c->bound++; c->nt[c->nnt++] = n << 16 | op; c->nt[c->nnt++] = type; c->nt[c->nnt++] = id; if (op == 43) c->nt[c->nnt++] = value; return id;
}
static uint32_t *spv_alpha_test(const uint32_t *w, uint32_t nw, uint32_t *onw, const char **why) {
    spvi c = {.w = w, .fn = nw, .bound = w[3]}; uint32_t ep = 0, epn = 0, epfn = 0, firsttype = 0, colv = 0, v4 = 0, f32 = 0, fnat = 0, none[1] = {0};
    for (uint32_t i = 5; i < nw;) { uint32_t n = w[i] >> 16, op = w[i] & 0xffff; if (!n || i + n > nw) { *why = "a malformed module"; return NULL; }
        if (op == 15) { if (ep || n < 3 || w[i + 1] != 4) { *why = "not one fragment entry point"; return NULL; } ep = i; epn = n; epfn = w[i + 2]; }
        if (op == 59 && n >= 4 && w[i + 3] == 9) { *why = "a push constant block of its own"; return NULL; }   // (one an entry point at most)
        if (!firsttype && ((op >= 19 && op <= 39) || (op >= 41 && op <= 52) || op == 59)) firsttype = i;   // the first type, constant or variable: the annotations end here
        if (op == 54) { c.fn = i; break; }   // the first OpFunction
        i += n; }
    if (!ep || !firsttype || c.fn == nw) { *why = "no entry point or no function"; return NULL; }
    int k0 = spv_col0(w, nw, &colv, &v4, &f32); if (k0 != VK_COL0_FLOAT) { *why = k0 == VK_COL0_NONE ? "no colour 0 output" : "colour 0 has no float alpha"; return NULL; }
    for (uint32_t i = c.fn; i < nw;) { uint32_t n = w[i] >> 16, op = w[i] & 0xffff; if (!n || i + n > nw) { *why = "a malformed module"; return NULL; }
        if (op == 54 && n == 5 && w[i + 2] == epfn) fnat = i;
        if (op == 57 && n >= 4 && w[i + 3] == epfn) { *why = "its entry point is called"; return NULL; }   // OpFunctionCall
        i += n; }
    if (!fnat) { *why = "no entry point function"; return NULL; }
    if (!(c.nt = malloc(4096 * 4))) { *why = "no memory"; return NULL; }
    uint32_t vd = w[fnat + 1], fty = w[fnat + 4], u32 = spvi_scalar(&c, 1), bl = spvi_decl(&c, 20, none, 0, 1), k[8], tru, fls;
    for (uint32_t v = 0; v < 8; v++) k[v] = spvi_const(&c, 43, u32, v);
    tru = spvi_const(&c, 41, bl, 0); fls = spvi_const(&c, 42, bl, 0);
    uint32_t st = c.bound++; if (c.nnt + 4 <= 4096) { c.nt[c.nnt++] = 4u << 16 | 30; c.nt[c.nnt++] = st; c.nt[c.nnt++] = u32; c.nt[c.nnt++] = f32; }   // OpTypeStruct {uint func; float ref;}
    uint32_t o1[2] = {9, st}, o2[2] = {9, u32}, o3[2] = {9, f32}; uint32_t pps = spvi_decl(&c, 32, o1, 2, 0), ppu = spvi_decl(&c, 32, o2, 2, 1), ppf = spvi_decl(&c, 32, o3, 2, 1);   // PushConstant pointers
    uint32_t pcv = c.bound++; if (c.nnt + 4 <= 4096) { c.nt[c.nnt++] = 4u << 16 | 59; c.nt[c.nnt++] = pps; c.nt[c.nnt++] = pcv; c.nt[c.nnt++] = 9; }   // OpVariable PushConstant
    int ok = u32 && bl && tru && fls && pps && ppu && ppf && c.nnt + 8 <= 4096; for (int v = 0; v < 8; v++) ok &= k[v] != 0;
    if (!ok) { free(c.nt); *why = "no room for its declarations"; return NULL; }
    uint32_t orig = c.bound++, iface = w[1] >= 0x10400;   // the shader's own main, under a new id; SPIR-V 1.4 lists every global the entry point uses in its interface
    uint32_t *o = malloc(((size_t)nw + c.nnt + 256) * 4), no = 5; if (!o) { free(c.nt); *why = "no memory"; return NULL; }
    memcpy(o, w, 20);
    for (uint32_t i = 5; i < c.fn;) { uint32_t n = w[i] >> 16;
        if (i == firsttype) { uint32_t d[13] = {3u << 16 | 71, st, 2, 5u << 16 | 72, st, 0, 35, VK_ALPHA_PUSH_OFF, 5u << 16 | 72, st, 1, 35, VK_ALPHA_PUSH_OFF + 4};   // Block, member Offsets
                              memcpy(&o[no], d, sizeof d); no += 13; }
        if (i == ep && iface) { o[no++] = (epn + 1) << 16 | 15; memcpy(&o[no], &w[i + 1], (epn - 1) * 4); no += epn - 1; o[no++] = pcv; }
        else { memcpy(&o[no], &w[i], n * 4); no += n; }
        i += n; }
    memcpy(&o[no], c.nt, c.nnt * 4); no += c.nnt;
    memcpy(&o[no], &w[c.fn], (size_t)(nw - c.fn) * 4); o[no + (fnat - c.fn) + 2] = orig; no += nw - c.fn;   // the functions, the entry point's renamed
    uint32_t l0 = c.bound++, lk = c.bound++, lm = c.bound++, call = c.bound++, p0 = c.bound++, fv = c.bound++, p1 = c.bound++, ref = c.bound++, col = c.bound++, a = c.bound++;
    uint32_t x[] = {5u << 16 | 54, vd, epfn, 0, fty, 2u << 16 | 248, l0, 4u << 16 | 57, vd, call, orig,   // OpFunction (the entry point's id), OpLabel, OpFunctionCall main
                    5u << 16 | 65, ppu, p0, pcv, k[0], 4u << 16 | 61, u32, fv, p0, 5u << 16 | 65, ppf, p1, pcv, k[1], 4u << 16 | 61, f32, ref, p1,   // the function, the reference
                    4u << 16 | 61, v4, col, colv, 5u << 16 | 81, f32, a, col, 3};   // colour 0, its alpha
    memcpy(&o[no], x, sizeof x); no += sizeof x / 4;
    static const uint16_t CMP[6] = {184, 180, 188, 186, 183, 190};   // LESS FOrdLessThan, EQUAL FOrdEqual, LEQUAL FOrdLessThanEqual, GREATER FOrdGreaterThan, NOTEQUAL FUnordNotEqual, GEQUAL FOrdGreaterThanEqual
    uint32_t cmp[6]; for (int j = 0; j < 6; j++) { cmp[j] = c.bound++; o[no++] = 5u << 16 | CMP[j]; o[no++] = bl; o[no++] = cmp[j]; o[no++] = a; o[no++] = ref; }
    uint32_t r = tru;   // pass = func == 1 ? false : func == 2 ? a < ref : ... func == 7 ? a >= ref : true
    for (uint32_t f = 7; f >= 1; f--) { uint32_t e = c.bound++, s = c.bound++;
        o[no++] = 5u << 16 | 170; o[no++] = bl; o[no++] = e; o[no++] = fv; o[no++] = k[f];   // OpIEqual
        o[no++] = 6u << 16 | 169; o[no++] = bl; o[no++] = s; o[no++] = e; o[no++] = f == 1 ? fls : cmp[f - 2]; o[no++] = r; r = s; }   // OpSelect
    uint32_t y[] = {3u << 16 | 247, lm, 0, 4u << 16 | 250, r, lm, lk, 2u << 16 | 248, lk, w[1] >= 0x10600 ? 1u << 16 | 4416 : 1u << 16 | 252,   // if (!pass) OpKill (OpTerminateInvocation in SPIR-V 1.6)
                    2u << 16 | 248, lm, 1u << 16 | 253, 1u << 16 | 56};   // OpReturn, OpFunctionEnd
    memcpy(&o[no], y, sizeof y); no += sizeof y / 4;
    o[3] = c.bound; *onw = no; free(c.nt); return o;
}
// Images that start at a coarser level of their texture (vkimg_t lv0: a dense image of a sparse texture whose format the device cannot make sparse-resident, a sparse one past
// the sparse address space; a view moved to its first mapped level, sp_vbase): the shader works in the texture's level numbers and its view's level 0 is the texture's level B.
// Mario Kart World 1.8.0: 1,118 pixel shaders sample the BC6H / ASTC terrain and object arrays (dense from level 3 or 4 on the Adreno 840) with
// textureLod(t, uv, max(min LOD, quantize(textureQueryLod(t, uv).x))): the query answers in the image's levels and the minimum is in the texture's, so the level read was the
// minimum's plus B, up to 16 times blurrier than the screen needs. The variant adds B to OpImageQueryLod's and OpImageQueryLevels' results, and takes it from explicit Lod
// operands and from OpImageFetch / OpImageQuerySizeLod levels: a fetch below level B reads the image's level 0 at the texel that covers it (spatial coordinates shifted down), a
// size below it is level 0's scaled up. Implicit-LOD sampling, gradients and gathers need nothing (the hardware's LOD is the image's own). For the images of up to 8 bindings
// (bind, nb: binding numbers in that order); B is data, 4 bits a binding (binding i at bits 4i) in the push constant word at off (vk_lodshift_off), so one variant serves every draw
static uint32_t *spv_lod_shift(const uint32_t *w, uint32_t nw, const uint16_t *bind, int nb, uint32_t off, uint32_t *onw, const char **why) {
    if (nb < 1 || nb > 8) { *why = "not 1 to 8 bindings"; return NULL; }
    spvi c = {.w = w, .fn = nw, .bound = w[3]}; uint32_t ep = 0, epn = 0, firsttype = 0, pcv = 0, pst = 0, psti = 0, pstn = 0;
    uint8_t *nib = calloc(w[3], 1), *dims = calloc(w[3], 1); uint32_t *ity = calloc(w[3], 4), *rty = calloc(w[3], 4);   // nib: 1 + the binding's place; dims: spatial dimensions | 4 arrayed
    #define LS_FAIL(s) do { free(nib); free(dims); free(ity); free(rty); free(c.nt); *why = s; return NULL; } while (0)
    if (!nib || !dims || !ity || !rty) LS_FAIL("no memory");
    for (uint32_t i = 5; i < nw;) { uint32_t n = w[i] >> 16, op = w[i] & 0xffff; if (!n || i + n > nw) LS_FAIL("a malformed module");
        if (op == 15) { if (ep) LS_FAIL("more than one entry point"); ep = i; epn = n; }
        if (op == 71 && n >= 4 && w[i + 2] == 33 && w[i + 1] < w[3]) for (int j = 0; j < nb; j++) if (w[i + 3] == bind[j]) nib[w[i + 1]] = (uint8_t)(j + 1);   // OpDecorate Binding
        if (op == 25 && n >= 9 && w[i + 1] < w[3]) { uint32_t d = w[i + 3]; dims[w[i + 1]] = (uint8_t)((d == 0 ? 1 : d == 2 ? 3 : d == 1 || d == 3 ? 2 : 0) | (w[i + 5] ? 4 : 0)); }   // OpTypeImage: 1D, 2D / Cube, 3D
        if (op == 27 && n >= 3 && w[i + 1] < w[3]) ity[w[i + 1]] = w[i + 2];   // OpTypeSampledImage: its image type
        if ((op == 43 || op == 44 || op == 46 || op == 1) && n >= 3 && w[i + 2] < w[3]) rty[w[i + 2]] = w[i + 1];   // a constant's (or undef's) type
        if (op == 59 && n >= 4 && w[i + 3] == 9) { if (pcv) LS_FAIL("two push constant blocks"); pcv = w[i + 2];
            for (uint32_t j = 5; j < i;) { uint32_t m = w[j] >> 16; if (!m) break; if ((w[j] & 0xffff) == 32 && m == 4 && w[j + 1] == w[i + 1]) pst = w[j + 3]; j += m; } }   // its struct
        if (!firsttype && ((op >= 19 && op <= 39) || (op >= 41 && op <= 52) || op == 59)) firsttype = i;   // the first type, constant or variable: the annotations end here
        if (op == 54) { c.fn = i; break; }   // the first OpFunction
        i += n; }
    if (!ep || !firsttype || c.fn == nw) LS_FAIL("no entry point or no function");
    if (pcv) { for (uint32_t i = 5; i < c.fn;) { uint32_t n = w[i] >> 16; if ((w[i] & 0xffff) == 30 && w[i + 1] == pst) { psti = i; pstn = n - 2; } i += n; }   // OpTypeStruct of the block: n - 2 members
               if (!psti) LS_FAIL("a push constant block whose type is not a struct"); }
    // the images of those bindings as the function bodies take them (OpLoad of the variable or of an element of it, OpSampledImage, OpImage, OpCopyObject), and the sites
    uint32_t sites = 0;
    for (uint32_t i = c.fn; i < nw;) { uint32_t n = w[i] >> 16, op = w[i] & 0xffff; if (!n || i + n > nw) LS_FAIL("a malformed module");
        uint32_t r = n >= 3 ? w[i + 2] : 0; if (r >= w[3]) LS_FAIL("an id past the bound");
        if (op != 248 && op != 62 && op != 247 && op != 246 && op != 249 && op != 250 && op != 251 && op != 253 && op != 254 && n >= 3) rty[r] = w[i + 1];   // a value's type
        if ((op == 61 || op == 65 || op == 66 || op == 83 || op == 100) && n >= 4 && w[i + 3] < w[3] && nib[w[i + 3]]) nib[r] = nib[w[i + 3]];   // Load / AccessChain / CopyObject / Image
        if (op == 86 && n >= 5 && w[i + 3] < w[3] && nib[w[i + 3]]) nib[r] = nib[w[i + 3]];   // OpSampledImage (image, sampler)
        if ((op == 105 || op == 88 || op == 90 || op == 95 || op == 103 || op == 106) && n >= 4 && w[i + 3] < w[3] && nib[w[i + 3]]) sites++;
        i += n; }
    if (!sites) LS_FAIL("no level operation on those bindings");
    if (!(c.nt = malloc(4096 * 4))) LS_FAIL("no memory");
    uint32_t none[1] = {0}, o32[1] = {32};
    uint32_t u32 = spvi_scalar(&c, 1), i32 = spvi_scalar(&c, 2), f32 = spvi_decl(&c, 22, o32, 1, 1), bl = spvi_decl(&c, 20, none, 0, 1), k15 = spvi_const(&c, 43, u32, 15), z = spvi_const(&c, 43, i32, 0);
    uint32_t ksh[8] = {0}; for (int j = 0; j < nb; j++) ksh[j] = spvi_const(&c, 43, u32, 4u * (uint32_t)j);
    uint32_t mem = pcv ? pstn : 0, kmem = spvi_const(&c, 43, u32, mem), ppu_d[2] = {9, u32}, ppu = spvi_decl(&c, 32, ppu_d, 2, 1), st = pst, nv = 0;
    if (pcv && u32 >= w[3]) LS_FAIL("a push constant block and no 32-bit uint before it");   // (its struct takes one more member, of a type declared before it)
    if (!pcv) { uint32_t pps = c.bound++; st = c.bound++; nv = c.bound++;   // a block of its own: struct {uint}, its pointer and variable
                if (c.nnt + 11 > 4096) LS_FAIL("no room for its declarations");
                uint32_t d[11] = {3u << 16 | 30, st, u32, 4u << 16 | 32, pps, 9, st, 4u << 16 | 59, pps, nv, 9}; memcpy(&c.nt[c.nnt], d, sizeof d); c.nnt += 11; }
    uint32_t iv[3]; for (uint32_t q = 0; q < 3; q++) { uint32_t vd[2] = {i32, q + 2}; iv[q] = spvi_decl(&c, 23, vd, 2, 1); }   // ivec2..4: a vector's level shifts
    int ok = u32 && i32 && f32 && bl && k15 && z && kmem && ppu && iv[0] && iv[1] && iv[2]; for (int j = 0; j < nb; j++) ok &= ksh[j] != 0;
    if (!ok || c.nnt + 16 > 4096) LS_FAIL("no room for its declarations");
    uint32_t pvar = pcv ? pcv : nv, iface = w[1] >= 0x10400 && !pcv;   // SPIR-V 1.4 lists every global the entry point uses
    uint32_t *o = malloc(((size_t)nw + c.nnt + 64 + 72 * (size_t)sites) * 4), no = 5; if (!o) LS_FAIL("no memory");
    memcpy(o, w, 20);
    for (uint32_t i = 5; i < c.fn;) { uint32_t n = w[i] >> 16;
        if (i == firsttype) { if (!pcv) { o[no++] = 3u << 16 | 71; o[no++] = st; o[no++] = 2; }   // Block
                              o[no++] = 5u << 16 | 72; o[no++] = st; o[no++] = mem; o[no++] = 35; o[no++] = off; }   // OpMemberDecorate Offset
        if (i == ep && iface) { o[no++] = (epn + 1) << 16 | 15; memcpy(&o[no], &w[i + 1], (epn - 1) * 4); no += epn - 1; o[no++] = nv; }
        else if (i == psti) { o[no++] = (n + 1) << 16 | 30; memcpy(&o[no], &w[i + 1], (n - 1) * 4); no += n - 1; o[no++] = u32; }   // the block's struct, a member more
        else { memcpy(&o[no], &w[i], n * 4); no += n; }
        i += n; }
    memcpy(&o[no], c.nt, c.nnt * 4); no += c.nnt;
    for (uint32_t i = c.fn; i < nw;) { uint32_t n = w[i] >> 16, op = w[i] & 0xffff;
        int site = (op == 105 || op == 88 || op == 90 || op == 95 || op == 103 || op == 106) && n >= 4 && nib[w[i + 3]];
        if (!site) { memcpy(&o[no], &w[i], n * 4); no += n; i += n; continue; }
        uint32_t j = nib[w[i + 3]] - 1u, rt = w[i + 1], r = w[i + 2], img = w[i + 3], t = ity[rty[img]] ? ity[rty[img]] : rty[img], dm = t < w[3] ? dims[t] : 0;
        // B = (word >> 4j) & 15, as uint, int and float
        uint32_t p = c.bound++, wd = c.bound++, sh = c.bound++, bu = c.bound++, bi = c.bound++, bf = c.bound++;
        uint32_t x[] = {5u << 16 | 65, ppu, p, pvar, kmem, 4u << 16 | 61, u32, wd, p, 5u << 16 | 194, u32, sh, wd, ksh[j], 5u << 16 | 199, u32, bu, sh, k15,   // AccessChain, Load, >>, &
                        4u << 16 | 124, i32, bi, bu, 4u << 16 | 112, f32, bf, bu};   // OpBitcast int, OpConvertUToF
        memcpy(&o[no], x, sizeof x); no += sizeof x / 4;
        if (op == 105 || op == 106) {   // OpImageQueryLod: + B to both components; OpImageQueryLevels: + B
            uint32_t r0 = c.bound++; memcpy(&o[no], &w[i], n * 4); o[no + 2] = r0; no += n;
            if (op == 105) { uint32_t bv = c.bound++; uint32_t y[] = {5u << 16 | 80, rt, bv, bf, bf, 5u << 16 | 129, rt, r, r0, bv}; memcpy(&o[no], y, sizeof y); no += sizeof y / 4; }
            else { uint32_t y[] = {5u << 16 | 128, rt, r, r0, rt == u32 ? bu : bi}; memcpy(&o[no], y, sizeof y); no += sizeof y / 4; }
        } else if (op == 88 || op == 90) {   // explicit Lod - B (the sampler's minimum LOD, lowered by B, keeps it from going below the image's level 0)
            uint32_t mi = op == 88 ? 5 : 6, mask = n > mi ? w[i + mi] : 0, li = mi + 1 + (mask & 1);
            if (!(mask & 2) || li >= n) { memcpy(&o[no], &w[i], n * 4); no += n; i += n; continue; }
            uint32_t l2 = c.bound++, y[] = {5u << 16 | 131, f32, l2, w[i + li], bf}; memcpy(&o[no], y, sizeof y); no += sizeof y / 4;   // OpFSub
            memcpy(&o[no], &w[i], n * 4); o[no + li] = l2; no += n;
        } else {   // OpImageFetch (its Lod operand) / OpImageQuerySizeLod (its level): d = level - B; the level max(d, 0), and below level B, s = B - level spatial texel shifts
            uint32_t mask = op == 95 && n > 5 ? w[i + 5] : 0, li = op == 95 ? 6 + (mask & 1) : 4;
            if ((op == 95 && !(mask & 2)) || li >= n) { memcpy(&o[no], &w[i], n * 4); no += n; i += n; continue; }
            uint32_t d = c.bound++, neg = c.bound++, l2 = c.bound++, nd = c.bound++, s = c.bound++;
            uint32_t y[] = {5u << 16 | 130, i32, d, w[i + li], bi, 5u << 16 | 177, bl, neg, d, z, 6u << 16 | 169, i32, l2, neg, z, d,   // OpISub, OpSLessThan, OpSelect
                            4u << 16 | 126, i32, nd, d, 6u << 16 | 169, i32, s, neg, nd, z};   // OpSNegate, OpSelect
            memcpy(&o[no], y, sizeof y); no += sizeof y / 4;
            uint32_t vt = op == 95 ? rty[w[i + 4]] : rt, ncomp = 0;   // the value the shifts apply to: the fetch's coordinates, the query's result
            for (uint32_t q = 5; q < c.fn;) { uint32_t m = w[q] >> 16, qo = w[q] & 0xffff; if (w[q + 1] == vt) ncomp = qo == 23 ? w[q + 3] : qo == 21 ? 1 : ncomp; q += m; }
            if (!ncomp || ncomp > 4) { free(o); LS_FAIL("a level operation on a value whose type is not an integer scalar or vector"); }
            uint32_t sp = dm & 3, sv = s;
            if (ncomp > 1) { sv = c.bound++; o[no++] = (3 + ncomp) << 16 | 80; o[no++] = iv[ncomp - 2]; o[no++] = sv; for (uint32_t q = 0; q < ncomp; q++) o[no++] = q < sp ? s : z; }   // (s, s, 0): not the layer
            if (op == 95) { uint32_t c2 = c.bound++; o[no++] = 5u << 16 | 195; o[no++] = vt; o[no++] = c2; o[no++] = w[i + 4]; o[no++] = sv;   // OpShiftRightArithmetic
                            memcpy(&o[no], &w[i], n * 4); o[no + 4] = c2; o[no + li] = l2; no += n; }
            else { uint32_t r0 = c.bound++; memcpy(&o[no], &w[i], n * 4); o[no + 2] = r0; o[no + li] = l2; no += n;
                   o[no++] = 5u << 16 | 196; o[no++] = rt; o[no++] = r; o[no++] = r0; o[no++] = sv; }   // OpShiftLeftLogical
        }
        i += n; }
    free(nib); free(dims); free(ity); free(rty);
    o[3] = c.bound; *onw = no; free(c.nt); return o;
    #undef LS_FAIL
}
static VkShaderModule vk_shader_module_(vkshader_t *s, uint64_t um, uint64_t im, uint32_t xo, int fd, uint64_t lm);
// DM_VK_LODSHIFT=0 (start-up): no LOD-shift variant and no push constant words for it, as before
int vk_lodshift_on(void) { static int on = -1; if (on < 0) { const char *e = getenv("DM_VK_LODSHIFT"); on = !(e && !atoi(e)); } return on; }
uint32_t vk_lodshift_off(int stage) { return 8u + 4u * (uint32_t)set_of(stage); }
uint64_t vk_lodshift_mask(const vkshader_t *s, const uint8_t shift[64]) {
    uint64_t m = 0; int n = 0; if (!s || !vk_lodshift_on()) return 0;
    for (int i = 0; i < s->nbind && i < 64; i++) if (s->lod_ops >> i & 1 && shift[s->b[i].binding & 63]) { if (n++ == 8) { VKRED_ONCE(170, "shader %016llx: more than 8 bindings with a LOD shift, the 9th on read in the image's level numbers", (unsigned long long)s->key); break; } m |= 1ull << i; }
    return m;
}
uint32_t vk_lodshift_word(const vkshader_t *s, uint64_t lm, const uint8_t shift[64]) {
    uint32_t v = 0, j = 0; for (int i = 0; i < s->nbind && i < 64 && j < 8; i++) if (lm >> i & 1) { uint32_t b = shift[s->b[i].binding & 63]; v |= (b > 15 ? 15u : b) << (4 * j++); }
    return v;
}
VkShaderModule vk_shader_module_lod(vkshader_t *s, uint64_t um, uint64_t im, uint32_t xo, int fd, uint64_t lm) { extern int g_prof_on; extern uint64_t g_prof_ns[8]; struct timespec a, b; if (g_prof_on) clock_gettime(CLOCK_MONOTONIC, &a);
    VkShaderModule m = vk_shader_module_(s, um, im, xo, fd, lm); if (g_prof_on) { clock_gettime(CLOCK_MONOTONIC, &b); g_prof_ns[3] += (uint64_t)((b.tv_sec - a.tv_sec) * 1000000000ll + (b.tv_nsec - a.tv_nsec)); } return m; }
VkShaderModule vk_shader_module(vkshader_t *s, uint64_t um, uint64_t im, uint32_t xo, int fd) { return vk_shader_module_lod(s, um, im, xo, fd, 0); }
static VkShaderModule vk_shader_module_(vkshader_t *s, uint64_t um, uint64_t im, uint32_t xo, int fd, uint64_t lm) {
    int dual = fd >> 1 & 1, at = fd >> 2 & 1, mp = fd >> 3 & 1, mb = fd >> 4 & 1; fd &= 1;   // fd bit 1: the dual-source variant; bit 2: the alpha test's (spv_alpha_test); bits 3, 4: a mesh
    if (fd && (s->stage != 1 || s->depth_repl)) fd = 0;                                       // shader's payload made Private, its gl_Position in a block (spv_payload_private, spv_pos_block)
    if (dual && s->stage != 1) dual = 0;
    if (at && (s->stage != 1 || s->col0 != VK_COL0_FLOAT)) at = 0;
    if (s->stage != 7) mp = mb = 0;
    int fv = fd | dual << 1 | at << 2 | mp << 3 | mb << 4;   // (the variant's key)
    lm &= vk_lodshift_on() ? s->lod_ops : 0;   // (the bindings that take level numbers: vk_lodshift_mask)
    if (!(um | im | xo | (uint32_t)fv | lm)) return s->mod;   // (the alternative code below is read from its file, the variants from the words: kept or rebuilt)
    if (xo == 1u << 31 || xo == (1u << 31 | 1u << 30)) {   // DM_VK_SPV_ALT; bit 30 too: DM_VK_X_CSALT=2's second folder
        pthread_mutex_lock(&g_sh_mu); for (int i = 0; i < s->nvar; i++) if (s->var[i].um == 0 && s->var[i].im == 0 && s->var[i].xo == xo) { VkShaderModule m = s->var[i].mod; pthread_mutex_unlock(&g_sh_mu); return m; }
        VkShaderModule m = alt_module(s, xo >> 30 & 1 ? 2 : 1); if (s->nvar < 16) s->var[s->nvar++] = (typeof(s->var[0])){0, 0, xo, 0, m}; pthread_mutex_unlock(&g_sh_mu); return m; }
    xo &= ~(1u << 31);
    pthread_mutex_lock(&g_sh_mu);
    for (int i = 0; i < s->nvar; i++) if (s->var[i].um == um && s->var[i].im == im && s->var[i].xo == xo && s->var[i].fd == fv && s->var[i].lm == lm) { VkShaderModule m = s->var[i].mod; pthread_mutex_unlock(&g_sh_mu); return m; }
    uint16_t bind[64]; uint8_t kind[64]; int nb = 0;
    for (int i = 0; i < s->nbind && i < 64; i++) if ((um | im) >> i & 1) { bind[nb] = s->b[i].binding; kind[nb++] = um >> i & 1 ? 1 : 2; }
    // the words the variant starts from: the shader's kept ones (DM_VK_SPV_KEEP=1, spv_kept_for_readers), or rebuilt from the bundle as vk_shader made them (shader_words), under
    // g_sh_mu as the variant is made, and freed once its module is
    uint32_t nbase = s->nspv, *own = NULL; const uint32_t *base = s->spv;
    if (!base) { const bent *e = g_bundle ? find(s->key) : NULL; base = own = e ? shader_words(s, e, &nbase, 0) : NULL;
        if (!own) { pthread_mutex_unlock(&g_sh_mu); static int nfail; if (nfail++ < 16) VKLOG("[RED] shader %016llx: its SPIR-V could not be rebuilt from the bundle (%s): no variant this time, the module as it is",
                                                                                         (unsigned long long)s->key, e ? "no memory" : "not found"); return s->mod; }
        if (nbase != s->nspv) VKRED_ONCE(171, "shader %016llx: its SPIR-V rebuilt for a variant is %u words, %u when its module was made", (unsigned long long)s->key, nbase, s->nspv);
        g_spv_rebuilds++; }
    uint32_t nw = nbase, *w = nb ? spv_int_images(base, nbase, bind, kind, nb, &nw) : NULL; VkShaderModule m = s->mod;
    // a fragment shader's xo: the inputs a mesh shader writes per primitive (vkshader_t prim_locs), decorated PerPrimitiveEXT (spv_per_primitive); another stage's: outputs to declare
    if (xo && (w || !nb) && s->stage == 1) { uint32_t n2 = 0, *w2 = spv_per_primitive(w ? w : base, nw, xo, &n2); free(w); w = w2; nw = n2; }
    else if (xo && (w || !nb)) { uint32_t n2 = 0, *w2 = spv_add_outputs(w ? w : base, nw, xo, &n2); free(w); w = w2; nw = n2;
        if (w && s->stage == 0) { uint32_t n3 = 0, *w3 = spv_vs_init(w, nw, &n3); if (w3) { free(w); w = w3; nw = n3; } } }   // the added outputs start as (0, 0, 0, 1) too
    if (fd && (w || !(nb || xo))) { uint32_t n2 = 0, *w2 = spv_frag_depth(w ? w : base, nw, &n2); free(w); w = w2; nw = n2; }
    if (at && (w || !(nb || xo || fd))) { const char *why = NULL; uint32_t n2 = 0, *w2 = spv_alpha_test(w ? w : base, nw, &n2, &why);   // (before spv_dual: colour 0 is location 0 alone)
        if (w2) { free(w); w = w2; nw = n2; } else VKLOG("[RED] shader %016llx: no alpha test variant (%s): its alpha-tested draws go without the test", (unsigned long long)s->key, why ? why : "?"); }
    if (dual && (w || !(nb || xo || fd))) { uint32_t n2 = 0, *w2 = spv_dual(w ? w : base, nw, &n2); free(w); w = w2; nw = n2; }
    if (mp && (w || !(nb || xo))) { uint32_t n2 = 0, *w2 = spv_payload_private(w ? w : base, nw, &n2); free(w); w = w2; nw = n2; }
    if (mb && (w || !(nb || xo || mp))) { uint32_t n2 = 0, *w2 = spv_pos_block(w ? w : base, nw, &n2); free(w); w = w2; nw = n2; }
    if (lm && (w || !(nb || xo || fv))) { uint16_t lb[8]; int nl = 0; const char *why = NULL; uint32_t n2 = 0;   // (last: it adds to a push constant block the alpha test made)
        for (int i = 0; i < s->nbind && i < 64 && nl < 8; i++) if (lm >> i & 1) lb[nl++] = s->b[i].binding;
        uint32_t *w2 = spv_lod_shift(w ? w : base, nw, lb, nl, vk_lodshift_off(s->stage), &n2, &why);
        if (w2) { free(w); w = w2; nw = n2; } else VKLOG("[RED] shader %016llx: no LOD-shift variant (%s): its level numbers stay the image's", (unsigned long long)s->key, why ? why : "?"); }
    if (w) { VkShaderModuleCreateInfo ci = {VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, NULL, 0, (size_t)nw * 4, w};
        if (vkCreateShaderModule(V.dev, &ci, NULL, &m)) { m = s->mod; VKLOG("[RED] shader %016llx: its variant was refused (bindings %#llx / %#llx, outputs %#x, fragment depth %d, dual source %d, alpha test %d, payload private %d, position block %d, LOD shift %#llx)", (unsigned long long)s->key, (unsigned long long)um, (unsigned long long)im, xo, fd, dual, at, mp, mb, (unsigned long long)lm); }
        else VKLOG("shader %016llx: variant for bindings %#llx (uint texels) %#llx (int texels), %s %#x, fragment depth written %d, dual source %d, alpha test %d, LOD shift %#llx%s", (unsigned long long)s->key, (unsigned long long)um, (unsigned long long)im,
                   s->stage == 1 ? "per-primitive inputs" : "outputs added", xo, fd, dual, at, (unsigned long long)lm, s->stage == 7 ? (mp ? (mb ? ", payload private, position block" : ", payload private") : mb ? ", position block" : "") : "");
        free(w); }
    else VKLOG("[RED] shader %016llx: no variant could be made (bindings %#llx / %#llx, outputs %#x, fragment depth %d, dual source %d, alpha test %d, payload private %d, position block %d, LOD shift %#llx)", (unsigned long long)s->key, (unsigned long long)um, (unsigned long long)im, xo, fd, dual, at, mp, mb, (unsigned long long)lm);
    free(own);
    if (s->nvar < 16) s->var[s->nvar++] = (typeof(s->var[0])){um, im, xo, fv, m, lm};
    pthread_mutex_unlock(&g_sh_mu);
    return m;
}
// static buffers: Ryujinx's support buffer (render scales 1.0, nothing else set: as nvn_mtl.m support_buf) and 64 KB of zeros (unbound uniform slots, driver segments nobody reads)
static VkBuffer static_buf(int which) {
    static VkBuffer b[2]; static pthread_mutex_t mu = PTHREAD_MUTEX_INITIALIZER;
    { VkBuffer r = __atomic_load_n(&b[which], __ATOMIC_ACQUIRE); if (r) return r; }   // made already: no lock (every draw binds them)
    pthread_mutex_lock(&mu);
    if (!b[which]) {
        VkBuffer nb; VkBufferCreateInfo bi = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, NULL, 0, 65536, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, VK_SHARING_MODE_EXCLUSIVE};
        VKCHK(vkCreateBuffer(V.dev, &bi, NULL, &nb)); VkMemoryRequirements req; vkGetBufferMemoryRequirements(V.dev, nb, &req);
        VkMemoryAllocateInfo ai = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, NULL, req.size, vk_mem_type(req.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, 0)};
        VkDeviceMemory m; VKCHK(vkAllocateMemory(V.dev, &ai, NULL, &m)); VKCHK(vkBindBufferMemory(V.dev, nb, m, 0));
        void *p; VKCHK(vkMapMemory(V.dev, m, 0, VK_WHOLE_SIZE, 0, &p)); memset(p, 0, 65536);
        if (which == 0) for (int i = 0; i < 73; i++) *(float *)((uint8_t *)p + 192 + 16 * i) = 1.0f;   // render_scale[i].x (SupportBuffer.cs layout)
        __atomic_store_n(&b[which], nb, __ATOMIC_RELEASE);   // published once filled
    }
    pthread_mutex_unlock(&mu);
    return b[which];
}
VkBuffer vk_support_buffer(void) { return static_buf(0); }
VkBuffer vk_zero_buffer(void) { return static_buf(1); }
// RK_SPILL: the storage buffer that holds the shared memory of compute shaders that declare more than the device gives a workgroup (shader_vk.py SHARED_MAX; the translator's
// SharedMemorySpill.cs), a slice per workgroup. One buffer for all of them, made as large as the largest asks (48 KB x 512 slices for Mario Kart World's DrivePassShaderDefault,
// 24 MB); a smaller one is freed once the GPU is past it. Shared memory starts undefined on the console too, so its content between dispatches does not matter, but two
// dispatches that run together must not share it: the replay orders a dispatch after the work before it (gfx_vk_exec.c)
VkBuffer vk_spill_buffer(VkDeviceSize size) {
    static VkBuffer buf; static VkDeviceMemory mem; static VkDeviceSize have; static pthread_mutex_t mu = PTHREAD_MUTEX_INITIALIZER;
    pthread_mutex_lock(&mu);
    if (size > have) {
        VkBuffer nb; VkDeviceMemory m; VkMemoryRequirements req;
        VkBufferCreateInfo bi = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, NULL, 0, size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_SHARING_MODE_EXCLUSIVE};
        VKCHK(vkCreateBuffer(V.dev, &bi, NULL, &nb)); vkGetBufferMemoryRequirements(V.dev, nb, &req);
        VkMemoryAllocateInfo ai = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, NULL, req.size, vk_mem_type(req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 0)};
        VKCHK(vkAllocateMemory(V.dev, &ai, NULL, &m)); VKCHK(vkBindBufferMemory(V.dev, nb, m, 0));
        if (buf) vk_defer_free_buffer(buf, mem);
        buf = nb; mem = m; have = size;
        VKLOG("compute shared memory in a buffer: %.1f MB", size / 1048576.0);
    }
    VkBuffer b = buf; pthread_mutex_unlock(&mu);
    return b;
}
// a shader's c1 constants: copied once into a buffer of their own (the bundle bytes do not change)
void vk_shader_c1(vkshader_t *s, VkBuffer *buf, VkDeviceSize *off, VkDeviceSize *len) {
    if (!s->c1_len) { *buf = vk_zero_buffer(); *off = 0; *len = 65536; return; }
    if (!s->c1_buf) {
        VkBufferCreateInfo bi = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, NULL, 0, (s->c1_len + 255) & ~255u, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VK_SHARING_MODE_EXCLUSIVE};
        VKCHK(vkCreateBuffer(V.dev, &bi, NULL, &s->c1_buf)); VkMemoryRequirements req; vkGetBufferMemoryRequirements(V.dev, s->c1_buf, &req);
        VkMemoryAllocateInfo ai = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, NULL, req.size, vk_mem_type(req.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, 0)};
        VkDeviceMemory m; VKCHK(vkAllocateMemory(V.dev, &ai, NULL, &m)); VKCHK(vkBindBufferMemory(V.dev, s->c1_buf, m, 0));
        void *p; VKCHK(vkMapMemory(V.dev, m, 0, VK_WHOLE_SIZE, 0, &p)); memset(p, 0, bi.size); memcpy(p, s->c1, s->c1_len);
    }
    *buf = s->c1_buf; *off = 0; *len = (s->c1_len + 255) & ~255u;
}
// pipeline layouts by the set layouts of their stages (an empty layout fills the sets of absent stages); a graphics one also has the alpha test's push constant range for the fragment
// stage (VK_ALPHA_PUSH_OFF), so every graphics layout is compatible with every other one of the same sets for binding them, whether a pipeline takes the alpha test variant or not.
// With DM_VK_LODSHIFT, every layout has a word for each stage's LOD shifts (vk_lodshift_off): the fragment range runs on to it, vertex / mesh and geometry / task get one each, compute one
static VkPipelineLayout layout_(VkDescriptorSetLayout sets[5], int nsets, int gfx) {
    static VkDescriptorSetLayout empty; static nvn_hmap cache = {.klen = 6 * sizeof(VkDescriptorSetLayout)}; static pthread_mutex_t mu = PTHREAD_MUTEX_INITIALIZER;
    pthread_mutex_lock(&mu);
    if (!empty) { VkDescriptorSetLayoutCreateInfo li = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO}; VKCHK(vkCreateDescriptorSetLayout(V.dev, &li, NULL, &empty)); }
    VkDescriptorSetLayout k[6]; for (int i = 0; i < 5; i++) { k[i] = i < nsets && sets[i] ? sets[i] : (i < nsets ? empty : VK_NULL_HANDLE); if (i < nsets) sets[i] = k[i]; }   // absent stages get the empty layout (the caller allocates their sets from it)
    k[5] = (VkDescriptorSetLayout)(uintptr_t)gfx;   // (the cache key's last word)
    VkPipelineLayout pl = nvn_hm_get(&cache, k);
    if (!pl) {
        VkPushConstantRange pr[3]; uint32_t np = 0; int ls = vk_lodshift_on();
        if (gfx) { pr[np++] = (VkPushConstantRange){VK_SHADER_STAGE_FRAGMENT_BIT, VK_ALPHA_PUSH_OFF, ls ? vk_lodshift_off(1) + 4 - VK_ALPHA_PUSH_OFF : VK_ALPHA_PUSH_SIZE};
                   if (ls) pr[np++] = (VkPushConstantRange){VK_SHADER_STAGE_VERTEX_BIT | (V.has_mesh ? VK_SHADER_STAGE_MESH_BIT_EXT : 0), vk_lodshift_off(0), 4};
                   if (ls && (V.has_gs || V.has_mesh)) pr[np++] = (VkPushConstantRange){(V.has_gs ? VK_SHADER_STAGE_GEOMETRY_BIT : 0) | (V.has_mesh ? VK_SHADER_STAGE_TASK_BIT_EXT : 0), vk_lodshift_off(2), 4}; }
        else if (ls) pr[np++] = (VkPushConstantRange){VK_SHADER_STAGE_COMPUTE_BIT, vk_lodshift_off(5), 4};
        VkPipelineLayoutCreateInfo ci = {VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, NULL, 0, (uint32_t)nsets, k, np, np ? pr : NULL};
        VKCHK(vkCreatePipelineLayout(V.dev, &ci, NULL, &pl)); nvn_hm_put(&cache, k, pl);
    }
    pthread_mutex_unlock(&mu);
    return pl;
}
VkPipelineLayout vk_pipeline_layout(VkDescriptorSetLayout sets[5], int nsets) { return layout_(sets, nsets, 0); }
VkPipelineLayout vk_pipeline_layout_gfx(VkDescriptorSetLayout sets[5], int nsets) { return layout_(sets, nsets, 1); }
