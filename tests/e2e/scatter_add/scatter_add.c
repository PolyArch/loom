#include <stdint.h>

__attribute__((noinline)) void scatter_add(
    const uint32_t *restrict src, const uint32_t *restrict indices,
    uint32_t *restrict dst, uint32_t n, uint32_t dst_size) {
#pragma clang loop vectorize(disable) interleave(disable)
  for (uint32_t i = 0u; i < n; ++i) {
    uint32_t idx = indices[i];
    if (idx < dst_size)
      dst[idx] = dst[idx] + src[i];
  }
}

int main(void) {
  uint32_t src[8] = {15u, 27u, 8u, 42u, 19u, 33u, 11u, 25u};
  uint32_t indices[8] = {0u, 3u, 1u, 3u, 5u, 2u, 1u, 4u};
  uint32_t dst[6] = {5u, 10u, 7u, 3u, 12u, 8u};

  scatter_add(src, indices, dst, 8u, 6u);

  return (dst[0] == 20u && dst[1] == 29u && dst[2] == 40u && dst[3] == 72u &&
          dst[4] == 37u && dst[5] == 27u)
             ? 0
             : 1;
}
