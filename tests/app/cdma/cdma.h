// Loom kernel: cdma
#ifndef CDMA_H
#define CDMA_H

#include <cstdint>
#include <cstddef>

void cdma_cpu(const uint32_t* __restrict__ SRC, uint32_t* __restrict__ DST, const size_t N);

void cdma_dsa(const uint32_t* __restrict__ SRC, uint32_t* __restrict__ DST, const size_t N);

#endif // CDMA_H
