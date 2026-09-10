#ifndef LOOM_RUNTIME_COMPUTATION_H
#define LOOM_RUNTIME_COMPUTATION_H

/* Source-owned performance boundaries, shared by the CPU and accelerator
 * executions. Prepare inputs before begin; publish all outputs before end.
 * Initialization, warmup, and result checking belong outside these calls.
 * Native source execution treats the calls as compiler memory barriers. The
 * application image builder supplies the target observation implementation.
 */
#ifdef __cplusplus
extern "C" {
#endif

__attribute__((weak, noinline)) void loom_computation_begin(void) {
  __asm__ volatile("" ::: "memory");
}

__attribute__((weak, noinline)) void loom_computation_end(void) {
  __asm__ volatile("" ::: "memory");
}

#ifdef __cplusplus
}
#endif

#endif
