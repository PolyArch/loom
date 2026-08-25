#if defined(LOOM_APPLICATION_HOST_EXECUTION)
#include <stdio.h>
#endif

#if !defined(LOOM_VECADD_VECTOR_LENGTH)
#define LOOM_VECADD_VECTOR_LENGTH 64
#endif

#if !defined(LOOM_VECADD_MULTIPLIER)
#define LOOM_VECADD_MULTIPLIER 2
#endif

#if !defined(LOOM_VECADD_BIAS)
#define LOOM_VECADD_BIAS 0
#endif

enum { VECTOR_LENGTH = LOOM_VECADD_VECTOR_LENGTH };

__attribute__((noinline)) static void
vecadd_memory(const int *restrict lhs, const int *restrict rhs,
              int *restrict result, int length) {
  for (int index = 0; index < length; ++index)
    result[index] = lhs[index] + rhs[index];
}

int main(void) {
  int lhs[VECTOR_LENGTH];
  int rhs[VECTOR_LENGTH];
  int result[VECTOR_LENGTH];
  for (int index = 0; index < VECTOR_LENGTH; ++index) {
    lhs[index] = index;
    rhs[index] = index * LOOM_VECADD_MULTIPLIER + LOOM_VECADD_BIAS;
  }
  vecadd_memory(lhs, rhs, result, VECTOR_LENGTH);
  long long checksum = 0;
  for (int index = 0; index < VECTOR_LENGTH; ++index)
    if (result[index] !=
        index * (LOOM_VECADD_MULTIPLIER + 1) + LOOM_VECADD_BIAS)
      return 1;
    else
      checksum += result[index];
#if defined(LOOM_APPLICATION_HOST_EXECUTION)
  printf("vecadd checksum: %lld\n", checksum);
#else
  (void)checksum;
#endif
  return 0;
}
