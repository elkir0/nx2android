// vkbench (devtools/vkbench): force-included before every backend source in the host (macOS) build. What Linux / bionic gives the Android build and macOS lacks, mapped onto
// macOS, so the replay's own sources compile unchanged. Nothing here runs on the device
#pragma once
#include <signal.h>
#include <sys/ucontext.h>
#include <pthread.h>
#include <stdatomic.h>
#include <sched.h>
#include <stdint.h>
#include <unistd.h>
// Linux names the thread it is given; macOS only the calling one
#define pthread_setname_np(t, n) pthread_setname_np(n)
#include <sys/socket.h>
#ifndef SOCK_CLOEXEC
#define SOCK_CLOEXEC 0
#endif
// macOS has no SOCK_SEQPACKET for AF_UNIX: the dma-buf descriptor of a hardware buffer (gfx_vk_core.c ahb_new) goes over a stream socket, one message, as tests/test_vk_memory.py does
#undef SOCK_SEQPACKET
#define SOCK_SEQPACKET SOCK_STREAM
// the replay's SIGPROF sampler (gfx_vk_exec.c prof_sig) reads Linux arm64's mcontext: a stand-in of that layout (the sampler is never armed in the benchmark)
typedef struct { struct { uint64_t fault_address, regs[31], sp, pc, pstate; } uc_mcontext; } vkb_ucontext_t;
#define ucontext_t vkb_ucontext_t
// bionic's spinlocks (gfx_vk_dq.c's producers' lock): a test-and-set lock that yields, as bionic's spins
typedef struct { _Atomic int v; } pthread_spinlock_t;
static inline int pthread_spin_init(pthread_spinlock_t *l, int s) { (void)s; atomic_store(&l->v, 0); return 0; }
static inline int pthread_spin_lock(pthread_spinlock_t *l) { while (atomic_exchange_explicit(&l->v, 1, memory_order_acquire)) while (atomic_load_explicit(&l->v, memory_order_relaxed)) {} return 0; }
static inline int pthread_spin_unlock(pthread_spinlock_t *l) { atomic_store_explicit(&l->v, 0, memory_order_release); return 0; }
static inline int pthread_spin_destroy(pthread_spinlock_t *l) { (void)l; return 0; }
#ifndef PTHREAD_PROCESS_PRIVATE
#define PTHREAD_PROCESS_PRIVATE 2
#endif
// Linux thread ids
static inline int vkb_gettid(void) { uint64_t t = 0; pthread_threadid_np(NULL, &t); return (int)t; }
#define gettid() vkb_gettid()
// POSIX per-thread CPU-time timers (the same sampler's): macOS has none; the stand-ins fail, so the sampler would say it could not arm
struct vkb_sigevent { int sigev_notify, sigev_signo; union { int _tid; void *ptr; } _sigev_un; };
#define sigevent vkb_sigevent
#ifndef SIGEV_THREAD_ID
#define SIGEV_THREAD_ID 4
#endif
typedef int timer_t;
struct itimerspec { struct timespec it_interval, it_value; };
static inline int timer_create(clockid_t c, struct vkb_sigevent *e, timer_t *t) { (void)c; (void)e; (void)t; return -1; }
static inline int timer_settime(timer_t t, int f, const struct itimerspec *n, struct itimerspec *o) { (void)t; (void)f; (void)n; (void)o; return -1; }
// the driver: vk_init's dlopen("libvulkan.so") / dlsym go to the fake driver (fake_vk.c vkb_dlopen), everything else to the system's
#include <dlfcn.h>
void *vkb_dlopen(const char *path, int mode); void *vkb_dlsym(void *h, const char *name);
#define dlopen vkb_dlopen
#define dlsym vkb_dlsym
