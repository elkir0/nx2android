// vkbench (devtools/vkbench): the NDK's AHardwareBuffer declarations the Vulkan backend names, for a host (macOS) build with a fake device
#pragma once
#include <stdint.h>
typedef struct AHardwareBuffer AHardwareBuffer;
typedef struct { uint32_t width, height, layers, format; uint64_t usage; uint32_t stride, rfu0; uint64_t rfu1; } AHardwareBuffer_Desc;
enum { AHARDWAREBUFFER_FORMAT_BLOB = 0x21 };
#define AHARDWAREBUFFER_USAGE_CPU_READ_OFTEN 3ull
#define AHARDWAREBUFFER_USAGE_CPU_WRITE_OFTEN (3ull << 4)
#define AHARDWAREBUFFER_USAGE_GPU_DATA_BUFFER (1ull << 24)
int AHardwareBuffer_allocate(const AHardwareBuffer_Desc *d, AHardwareBuffer **out);
void AHardwareBuffer_release(AHardwareBuffer *a);
void AHardwareBuffer_acquire(AHardwareBuffer *a);
void AHardwareBuffer_describe(const AHardwareBuffer *a, AHardwareBuffer_Desc *d);
int AHardwareBuffer_lock(AHardwareBuffer *a, uint64_t usage, int32_t fence, const void *rect, void **out);
int AHardwareBuffer_unlock(AHardwareBuffer *a, int32_t *fence);
int AHardwareBuffer_sendHandleToUnixSocket(const AHardwareBuffer *a, int sock);
