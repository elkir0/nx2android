// Android port, Vulkan backend: the memory budget (gfx_vk_budget.c): where the process's memory is (a census of /proc/self/smaps, DM_VK_SMAPS), the device's limits as its low-memory
// killer sees them, a pressure level the replay's caches can answer, and the sparse address space policy (which sparse image may take how much of it)
#pragma once
#include <stdint.h>

void vk_budget_facts(void);   // once, from gfx_vk_core.c vk_init: memory and swap totals, the low-memory killer's settings (ro.lmk.*), oom_score_adj, zram where readable
void vk_budget_log(void);     // gfx_vk_core.c vk_mem_log: the census line (DM_VK_SMAPS=<seconds>: made on a thread of its own at most that often, off by default)
int vk_mem_pressure(void);    // 0 none, 1 moderate, 2 critical: MemAvailable and SwapFree against DM_VK_PRESSURE_MB / DM_VK_PRESSURE_SWAP_PCT (read at most every 500 ms)
// The sparse address space (gfx_vk_tex.c sparse_make): what the replay may take of it (DM_VK_SPARSE_SPACE_MB, else the device's sparseAddressSpaceSize), and whether a sparse image of
// size bytes may be made with used bytes taken: in its texture's format, any room; in its wider twin (promoted: twice the space for the memory it saves), room past what
// DM_VK_SPARSE_PROMO_RESERVE_MB leaves to images in their own format
uint64_t vk_sparse_space(uint64_t device);
int vk_sparse_fits(uint64_t used, uint64_t size, int promoted, uint64_t device);
