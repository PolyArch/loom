#include <stdint.h>
#include <stdio.h>

enum { EXTENT = 4, CELL_COUNT = EXTENT * EXTENT * EXTENT };

static uint32_t offset(uint32_t z, uint32_t y, uint32_t x) {
  return (z * EXTENT + y) * EXTENT + x;
}

__attribute__((noinline)) static void stencil3d_kernel(const uint32_t *input,
                                                       uint32_t *output) {
  for (uint32_t z = 0; z < EXTENT; ++z) {
    for (uint32_t y = 0; y < EXTENT; ++y) {
      for (uint32_t x = 0; x < EXTENT; ++x) {
        const uint32_t center = offset(z, y, x);
        if (z == 0 || y == 0 || x == 0 || z + 1 == EXTENT || y + 1 == EXTENT ||
            x + 1 == EXTENT) {
          output[center] = input[center];
          continue;
        }
        output[center] =
            2u * input[center] + input[offset(z - 1u, y, x)] +
            input[offset(z + 1u, y, x)] + input[offset(z, y - 1u, x)] +
            input[offset(z, y + 1u, x)] + input[offset(z, y, x - 1u)] +
            input[offset(z, y, x + 1u)];
      }
    }
  }
}

static void stencil3d_reference(const uint32_t *input, uint32_t *output) {
  for (uint32_t cell = 0; cell < CELL_COUNT; ++cell) {
    const uint32_t z = cell / (EXTENT * EXTENT);
    const uint32_t plane_cell = cell % (EXTENT * EXTENT);
    const uint32_t y = plane_cell / EXTENT;
    const uint32_t x = plane_cell % EXTENT;
    if (z == 0 || y == 0 || x == 0 || z + 1 == EXTENT || y + 1 == EXTENT ||
        x + 1 == EXTENT) {
      output[cell] = input[cell];
    } else {
      output[cell] = 2u * input[cell] + input[cell - EXTENT * EXTENT] +
                     input[cell + EXTENT * EXTENT] + input[cell - EXTENT] +
                     input[cell + EXTENT] + input[cell - 1u] + input[cell + 1u];
    }
  }
}

int main(void) {
  uint32_t input[CELL_COUNT];
  uint32_t reference[CELL_COUNT];
  uint32_t candidate[CELL_COUNT];
  uint64_t checksum = 0;

  for (uint32_t cell = 0; cell < CELL_COUNT; ++cell)
    input[cell] = cell + 1u;
  stencil3d_reference(input, reference);
  stencil3d_kernel(input, candidate);

  for (uint32_t cell = 0; cell < CELL_COUNT; ++cell) {
    if (candidate[cell] != reference[cell]) {
      printf("FAILED\n");
      return 1;
    }
    checksum += (uint64_t)(cell + 1u) * candidate[cell];
  }

  printf("stencil3d checksum: %llu\n", (unsigned long long)checksum);
  printf("PASSED\n");
  return 0;
}
