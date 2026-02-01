// Loom kernel: histogram
#ifndef HISTOGRAM_H
#define HISTOGRAM_H

#include <cstdint>
#include <cstddef>

void histogram_cpu(const uint32_t* __restrict__ input, uint32_t* __restrict__ hist, const uint32_t N, const uint32_t num_bins);

void histogram_dsa(const uint32_t* __restrict__ input, uint32_t* __restrict__ hist, const uint32_t N, const uint32_t num_bins);

#endif // HISTOGRAM_H
