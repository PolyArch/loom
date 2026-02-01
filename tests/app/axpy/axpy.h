// AXPY kernel header
// AXPY: output_y[i] = alpha * input_x[i] + input_y[i]
#ifndef AXPY_H
#define AXPY_H

#include <cstdint>

// CPU reference implementation
void axpy_cpu(const uint32_t* __restrict input_x,
              const uint32_t* __restrict input_y,
              uint32_t* __restrict output_y,
              uint32_t alpha, uint32_t N);

// DSA accelerated implementation with Loom pragmas
void axpy_dsa(const uint32_t* __restrict input_x,
              const uint32_t* __restrict input_y,
              uint32_t* __restrict output_y,
              uint32_t alpha, uint32_t N);

#endif // AXPY_H
