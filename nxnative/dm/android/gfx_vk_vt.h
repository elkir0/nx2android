// Shader-side virtual texturing of sparse textures without sparse residency (gfx_vk_vt.c; the shader side: tools/spv_vt.py; the design: .gamedump/<App>/vt/DESIGN.md).
// A dense image of a sparse texture (gfx_vk_tex.c dense_*) holds its levels from `first` on; VT gives the shaders the finer ones through a pool of 64 KB tiles in the texture's own
// compressed format (a 2D array image, a slot a layer: the tile with a border of one block of its neighbours) and a page table (a word a tile). The shaders ask for the tiles
// they find without a slot (a request bit a page table word, a region of the request buffer a submission), the backend uploads those the game maps, and a CLOCK over the slots
// takes back the ones no shader touched. DM_VK_VT (start-up): 0 off (default: a VT bundle draws as before), 1 on, 2 only the dense images' LOD fix (no pool)
#pragma once
#include "gfx_vk.h"

enum { VT_NC = 8 };                                     // candidates a shader may have (spv_vt.py TEMPLATE_NC): vt_ctl holds 1 + 3 VT_NC uvec4
enum { VT_CTL_BYTES = 16 * (1 + 3 * VT_NC) };
enum { RK_VT_POOL = 16, RK_VT_CTL = 17, RK_VT_PT = 18, RK_VT_FB = 19 };   // reflection kinds (shader_vk.py KIND): a candidate's pool (reg: its texture's binding, off: k), the control block, page table, requests
// a page table word (spv_vt.py TEMPLATE_GLSL)
#define VT_PT_SLOT 0xffffu      // slot + 1 (0: none)
#define VT_PT_MAPPED 0x10000u   // the game maps all 16 pages of the tile
#define VT_PT_PROB 0x20000u     // on probation: a shader that samples it asks for it, which keeps it
// a candidate's e[1 + 3k].x
enum { VT_MODE_OFF = 0, VT_MODE_DENSE = 1, VT_MODE_VT = 2 };
#define VT_CTL_LINEAR 4u        // linear mipmaps (else nearest)
#define VT_CTL_WRAPU(m) ((uint32_t)(m) << 3)   // the template's wrap codes: 0 clamp (to edge, to border), 1 repeat, 2 mirrored repeat, 3 mirror clamp to edge
#define VT_CTL_WRAPV(m) ((uint32_t)(m) << 5)

int vk_vt_mode(void);   // DM_VK_VT
// gfx_vk_tex.c: a dense image of a sparse texture was made (dense_make: mapped[u] its units' state, u = layer × per_layer + the tile's index in NVN's order, its tail at tail_at),
// the game mapped or unmapped ranges of it (dense_look, after its own scan of them: guest address ranges), its units' state was scanned again whole (the pool's log ran past),
// it is looked at once a submission (dense_look's end) or dropped (dense_drop). 1 / 0: the texture takes VT or not
int vk_vt_attach(vkimg_t *im, const tex_t *t, const vkfmt_t *f, const uint8_t *mapped, uint32_t per_layer, uint32_t tail_at);
void vk_vt_ranges(const vkimg_t *im, const uint8_t *mapped, const uint64_t (*r)[2], int n);
void vk_vt_rescan(const vkimg_t *im, const uint8_t *mapped);
void vk_vt_look(const vkimg_t *im);
void vk_vt_detach(const vkimg_t *im);
void vk_vt_lost(void);                    // gfx_vk_core.c: a submission's command buffers were dropped (its uploads and page table writes with them)
void vk_vt_sub_end(VkCommandBuffer cb);   // gfx_vk_core.c: the submission's last main command buffer ends (its request writes made visible to the host)
// gfx_vk_exec.c stage_set: what a candidate's texture binding resolved to, then the VT bindings' descriptors
// shift: the levels between the view's level 0 and its image's first (gfx_vk_exec.c tex_desc_for, its sampler's LOD shift): the shader's level numbers are the view's, so B is that,
// and the view's level 0 is the texture's level im->lv0 - shift; ly0, nly: the view's first layer and its layers (the shader's layer numbers are the view's too)
typedef struct { const vkimg_t *im; int sid; uint64_t sseq; VkFormat vfmt; uint32_t shift, ly0, nly; } vt_bound_t;
void vk_vt_pool_desc(const vt_bound_t *c, VkDescriptorImageInfo *ii);
void vk_vt_ctl_desc(const vt_bound_t *c, int n, VkDescriptorBufferInfo *bi);   // c[k] for candidate k (im NULL: none)
void vk_vt_pt_desc(VkDescriptorBufferInfo *bi);
void vk_vt_fb_desc(VkDescriptorBufferInfo *bi);
// gfx_vk_tex.c: the parameters of sampler sid as a bind after seq registrations sees them (0 when not registered)
int vk_samp_par_at(int sid, uint64_t seq, samp_par *out);
