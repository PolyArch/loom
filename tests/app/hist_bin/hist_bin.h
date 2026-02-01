// Loom kernel: hist_bin
#ifndef HIST_BIN_H
#define HIST_BIN_H

#include <cstdint>
#include <cstddef>

void hist_bin_cpu(const float* __restrict__ input, uint32_t* __restrict__ output, const uint32_t N, const uint32_t num_bins, const float min_val, const float max_val);

void hist_bin_dsa(const float* __restrict__ input, uint32_t* __restrict__ output, const uint32_t N, const uint32_t num_bins, const float min_val, const float max_val);

#endif // HIST_BIN_H
