#include <stdint.h>
#include <stdio.h>

enum { EXTENT = 4, CELL_COUNT = EXTENT * EXTENT * EXTENT };

int main(void) {
  uint32_t input[CELL_COUNT];
  uint32_t output[CELL_COUNT];
  uint64_t checksum = 0;

  for (uint32_t cell = 0; cell < CELL_COUNT; ++cell)
    input[cell] = cell + 1u;
  for (uint32_t z = 0; z < EXTENT; ++z) {
    for (uint32_t y = 0; y < EXTENT; ++y) {
      for (uint32_t x = 0; x < EXTENT; ++x) {
        const uint32_t cell = (z * EXTENT + y) * EXTENT + x;
        if (z == 0 || y == 0 || x == 0 || z + 1 == EXTENT || y + 1 == EXTENT ||
            x + 1 == EXTENT) {
          output[cell] = input[cell];
        } else {
          output[cell] = 2u * input[cell] + input[cell - EXTENT * EXTENT] +
                         input[cell + EXTENT * EXTENT] + input[cell - EXTENT] +
                         input[cell + EXTENT] + input[cell - 1u] +
                         input[cell + 1u];
        }
      }
    }
  }

  for (uint32_t cell = 0; cell < CELL_COUNT; ++cell)
    checksum += (uint64_t)(cell + 1u) * output[cell];
  if (checksum != UINT64_C(152412)) {
    printf("FAILED\n");
    return 1;
  }

  printf("stencil3d checksum: %llu\n", (unsigned long long)checksum);
  printf("PASSED\n");
  return 0;
}
