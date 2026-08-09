
#include <stdint.h>
#include <stdio.h>

namespace {

constexpr uint32_t kNodes = 8;
constexpr uint32_t kEdges = 16;
constexpr uint32_t kRowPtr[kNodes + 1u] = {0, 2, 4, 7, 10, 12, 14, 15, 16};
constexpr uint32_t kCols[kEdges] = {1, 2, 0, 3, 0, 4, 5, 1,
                                    2, 6, 3, 7, 4, 6, 7, 5};

void initialize_weights(uint32_t *weights) {
  for (uint32_t i = 0; i < kEdges; ++i) {
    weights[i] = i + 1u;
  }
}

void edge_update_ref(const uint32_t *row_ptr, const uint32_t *cols,
                     const uint32_t *input, uint32_t *output, uint32_t src,
                     uint32_t dst, uint32_t new_weight, uint32_t nodes,
                     uint32_t edges) {
  for (uint32_t i = 0; i < edges; ++i) {
    output[i] = input[i];
  }
  if (src >= nodes) {
    return;
  }
  for (uint32_t i = row_ptr[src]; i < row_ptr[src + 1u]; ++i) {
    if (cols[i] == dst) {
      output[i] = new_weight;
      return;
    }
  }
}

uint64_t checksum(const uint32_t *weights) {
  uint64_t sum = 0;
  for (uint32_t i = 0; i < kEdges; ++i) {
    sum += static_cast<uint64_t>(i + 1u) * weights[i];
  }
  return sum;
}

} // namespace

extern "C" __attribute__((noinline)) void
edge_update_kernel(const uint32_t *row_ptr, const uint32_t *cols,
                   const uint32_t *input, uint32_t *output, uint32_t src,
                   uint32_t dst, uint32_t new_weight, uint32_t nodes,
                   uint32_t edges) {
  for (uint32_t i = 0; i < edges; ++i) {
    output[i] = input[i];
  }
  if (src >= nodes) {
    return;
  }
  for (uint32_t i = row_ptr[src]; i < row_ptr[src + 1u]; ++i) {
    if (cols[i] == dst) {
      output[i] = new_weight;
      return;
    }
  }
}

int main() {
  uint32_t input[kEdges] = {};
  uint32_t reference[kEdges] = {};
  uint32_t candidate[kEdges] = {};

  initialize_weights(input);
  edge_update_ref(kRowPtr, kCols, input, reference, 2, 4, 100, kNodes, kEdges);
  edge_update_kernel(kRowPtr, kCols, input, candidate, 2, 4, 100, kNodes,
                     kEdges);

  for (uint32_t i = 0; i < kEdges; ++i) {
    if (reference[i] != candidate[i]) {
      printf("FAILED\n");
      return 1;
    }
  }

  printf("edge_update checksum: %llu\n",
         static_cast<unsigned long long>(checksum(candidate)));
  printf("PASSED\n");
  return 0;
}
