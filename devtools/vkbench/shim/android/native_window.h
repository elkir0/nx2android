// vkbench: the NDK's ANativeWindow declarations the Vulkan backend names (host build, no window)
#pragma once
#include <stdint.h>
typedef struct ANativeWindow ANativeWindow;
enum { ANATIVEWINDOW_FRAME_RATE_COMPATIBILITY_DEFAULT = 0, ANATIVEWINDOW_FRAME_RATE_COMPATIBILITY_FIXED_SOURCE = 1 };
int32_t ANativeWindow_getWidth(ANativeWindow *w); int32_t ANativeWindow_getHeight(ANativeWindow *w);
void ANativeWindow_acquire(ANativeWindow *w); void ANativeWindow_release(ANativeWindow *w);
int32_t ANativeWindow_setFrameRate(ANativeWindow *w, float rate, int8_t compat);
