// Android port: what both graphics backends (gfx_null.c, gfx_vk_*.c) report for the host's statistics and diagnostic hooks of the Metal backend
// (nvn_mtl.m) that have no counterpart yet: zeros.
#include "dm.h"
#include "nvn_obj.h"
#include "nvn_mtl.h"
#include <string.h>

// ---------------- statistics and diagnostics: nothing to report ----------------
void mtl_vbuf_stats(uint64_t *a, uint64_t *b, uint64_t *c, uint64_t *d, uint64_t *e, uint64_t *f) { *a = *b = *c = *d = *e = *f = 0; }
void mtl_res_counts(uint32_t tick, const char *why) { (void)tick; (void)why; }
void mtl_res_snapshot(mtl_res_count *out) { memset(out, 0, sizeof *out); }
void mtl_alloc_stats(uint64_t *w, uint64_t *g, int *n) { *w = *g = 0; *n = 0; }
void mtl_args_stats(uint64_t *e, uint64_t *r) { *e = *r = 0; }
void mtl_barrier_stats(uint64_t *s, uint64_t *f) { *s = *f = 0; }
uint64_t mtl_blit_splits(void) { return 0; }
void mtl_bw(uint64_t *l, uint64_t *s, uint64_t *f) { *l = *s = *f = 0; }
uint64_t mtl_copy_cut_count(void) { return 0; }
uint64_t mtl_cut_count(void) { return 0; }
void mtl_create_ns(uint64_t *l, uint64_t *p) { *l = *p = 0; }
void mtl_create_stats(uint64_t *l, uint64_t *p) { *l = *p = 0; }
uint64_t mtl_drawable_ns_take(void) { return 0; }
uint64_t mtl_ds_omit(void) { return 0; }
__attribute__((weak)) double mtl_gpu_ms(uint64_t f) { (void)f; return 0; }   // gfx_vk_core.c has the real one
int mtl_gpu_span(uint64_t f, uint64_t *t0, uint64_t *t1) { (void)f; *t0 = *t1 = 0; return 0; }
void mtl_lc_stats(uint64_t *t, uint64_t *l) { *t = *l = 0; }
void mtl_mem_stat(unsigned long long frame, unsigned long long rss_mb) { (void)frame; (void)rss_mb; }
uint64_t mtl_pkg_miss(void) { return 0; }
void mtl_pkg_stats(uint64_t *u, uint64_t *r) { *u = *r = 0; }
void mtl_res_stats(char *out, size_t n) { if (n) out[0] = 0; }
void mtl_stats(uint64_t *p, uint64_t *d) { *p = *d = 0; }
void mtl_ub_present(double iv) { (void)iv; }
void mtl_ubd_stats(uint64_t *d, uint64_t *c) { *d = *c = 0; }
void mtl_ubp_stats(uint64_t *m, uint64_t *r, uint64_t *f) { *m = *r = *f = 0; }
void mtl_ubp_why(uint64_t w[5]) { memset(w, 0, 5 * sizeof *w); }
uint64_t mtl_udc_saved_bytes(void) { return 0; }
uint64_t mtl_upload_bytes(void) { return 0; }
uint64_t mtl_upload_gpu_ns_take(void) { return 0; }
uint64_t mtl_upload_ns_take(void) { return 0; }
void mtl_use_stats(uint64_t *c, uint64_t *s) { *c = *s = 0; }
// Statistics of Metal-only machinery that hle_nvn.c's periodic lines read (nvn_mtl.m): compute-shader hoisting, the geometry lowering's hoist and mesh paths, the hybrid draw path and the
// shared-memory spill. None exists here: zeros
void mtl_cshoist_stats(uint64_t *moved, uint64_t *depth, uint64_t *stops) { *moved = *depth = 0; memset(stops, 0, 8 * sizeof *stops); }
void mtl_geohoist_stats(uint64_t *moved, uint64_t *runs, uint64_t *stops) { *moved = *runs = 0; memset(stops, 0, 6 * sizeof *stops); }
void mtl_geomesh_stats(uint64_t *n, uint64_t *wait, uint64_t *run_draws, uint64_t *runs, uint64_t ends[4]) { *n = *wait = *run_draws = *runs = 0; memset(ends, 0, 4 * sizeof *ends); }
void mtl_hybrid_stats(uint64_t *draws, uint64_t *made, uint64_t *none) { *draws = *made = *none = 0; }
void mtl_sm_stats(uint64_t *np, uint64_t *nc, uint64_t *nw) { *np = *nc = *nw = 0; }
// DM_AB's alternating phases (nvn_mtl.m ab_frame / ab_report: an A/B of a switch measured on the Mac's GPU timeline): the Android backends have live switches read per submission
// instead (gfx_vk_exec.c x_read), so the per-frame GPU times hle_nvn.c reports are not collected
void mtl_ab_gpu(uint64_t f, double gpu_ms, double frame_ms) { (void)f; (void)gpu_ms; (void)frame_ms; }
// The frame step's probe keeps its grace while its frames get cheaper (hle_nvn.c g_fs.ramp; nvn_mtl.m: macOS raises the GPU clock over seconds for a shorter deadline). Mobile GPU clocks ramp
// over seconds too (perf_android.c: the governor judges a rate after 4 s): on, as there. DM_FSTEP_PROBE_RAMP=0 (start-up): off
int mtl_probe_ramp_on(void) { static int v = -1; if (v < 0) { const char *e = getenv("DM_FSTEP_PROBE_RAMP"); v = !(e && !atoi(e)); } return v; }
// The display link (CAMetalDisplayLink: the Mac's and the iPhone's vsync callback, its 120 / 60 / 30 Hz modes, its stalls in the background) and the present snapshots taken from it
// (DM_PRESENT_SNAP) are nvn_mtl.m's: Android paces presents itself (perf_android.c, the swapchain, VK_GOOGLE_display_timing). With no link hle_nvn.c paces by its own clock, as before:
// every caller asks mtl_link_active first (0 in each backend), or takes 0 as no tick
int mtl_link_wait_ticks(int ticks, uint64_t timeout_ns) { (void)ticks; (void)timeout_ns; return 0; }
int mtl_link_wait_vsync(uint64_t *seen, uint64_t timeout_ns) { (void)seen; (void)timeout_ns; return 0; }   // (hle_sdk16.c's vsync event keeps its own clock)
int mtl_link_rate(void) { return 60; }
void mtl_link_mode_set(int mode) { (void)mode; }
int mtl_link_tick(uint64_t n, double *target, double *lead_ms, int *ticks, double *gap_ms) { (void)n; (void)target; (void)lead_ms; (void)ticks; (void)gap_ms; return 0; }
int mtl_link_stalled(void) { return 0; }
int mtl_snap_on(void) { return 0; }
uint64_t mtl_snap_acquire_wait(void) { return 0; }
// The Metal backend's own test and diagnostic hooks (nvn_mtl.m): the residency validator's counts and the blend-function wish list
void xv_test_counts(int *dc, int *viol) { if (dc) *dc = 0; if (viol) *viol = 0; }
void bf_want_note(const char *key) { (void)key; }
void dm_gfx_passdump_next(long n) { (void)n; }
void dm_gfx_shot_at(long k, const char *path) { (void)k; (void)path; }
void dm_gfx_texdump_next(const char *path) { (void)path; }
void dm_gfx_texlist(int target) { (void)target; }
void dm_gfx_texraw(uint64_t serial, const char *path) { (void)serial; (void)path; }
