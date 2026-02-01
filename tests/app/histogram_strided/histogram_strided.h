// Loom kernel: histogram_strided
#ifndef HISTOGRAM_STRIDED_H
#define HISTOGRAM_STRIDED_H

#include <cstdint>
#include <cstddef>

void histogram_strided_cpu(const uint32_t* __restrict__ input, uint32_t* __restrict__ hist, const uint32_t N, const uint32_t num_bins, const uint32_t stride);

void histogram_strided_dsa(const uint32_t* __restrict__ input, uint32_t* __restrict__ hist, const uint32_t N, const uint32_t num_bins, const uint32_t stride);

#endif // HISTOGRAM_STRIDED_H
