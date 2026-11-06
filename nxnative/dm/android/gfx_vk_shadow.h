// Android port, Vulkan backend: the guest pages of uploaded static textures moved onto a file mapping (gfx_vk_shadow.c, DM_VK_SHADOW=1; off by default until measured).
// Their bytes are written once to an unlinked file and the file is mapped over them MAP_PRIVATE: clean page cache the kernel drops and reads again from flash, where they were
// anonymous memory that only swap could take (and zram cannot shrink: the textures are compressed formats). A write by the game copies its page back to anonymous memory.
#pragma once
#include <stddef.h>
#include <stdint.h>

struct vkimg;
void vk_shadow_init(void);                     // gfx_vk_core.c vk_init, before the dirty-page tracker's first range: its page fault and host pre-write hooks, the file, the worker
int vk_shadow_tex_class(const struct vkimg *im);   // what a packaged texture is to the shadow (VK_SHC_*): VK_SHC_OK when its pages qualify now (also for the memory census, DM_VK_MEMLOG)
enum { VK_SHC_OK, VK_SHC_NOT_PACKAGED, VK_SHC_SMALL, VK_SHC_LINEAR, VK_SHC_GPU_WRITTEN, VK_SHC_UNTRACKED, VK_SHC_PLACED, VK_SHC_RELEASED, VK_SHC_WRITTEN, VK_SHC_N };
void vk_shadow_note_tex(const struct vkimg *im);   // gfx_vk_tex.c pkg_upload: a texture just uploaded from its storage, which is protected (its stamp taken after the protection)
void vk_shadow_note_sparse(const void *pool, uint64_t a, uint64_t n, uint64_t stamp);   // gfx_vk_tex.c sp_tile, dense_upload: the physical pages behind [a, a+n) of a sparse pool, just uploaded
void vk_shadow_tick(void);                     // gfx_vk_tex.c vk_tex_sweep, at every present on the replay thread: the swaps the worker has prepared, within a budget
void vk_shadow_drop(uint64_t lo, uint64_t n);  // gfx_vk_core.c place(): [lo, lo+n) becomes GPU memory (its pages a hardware buffer's): no longer the file's
void vk_shadow_log(void);                      // gfx_vk_core.c vk_mem_log: one line (DM_VK_MEMLOG)
void dm_guest_released(uintptr_t lo, uintptr_t hi);   // hle_svc.c guest_release (weak there): the game gave the range back; its blocks in the file are freed
void dm_guest_unshadow(uintptr_t lo, uintptr_t hi);   // hle_nvn.c nvnMemoryPoolFinalize (weak there), before nvn_dirty_remove_range makes the pool's pages writable without a fault
