// Loom kernel: line_intersect
#ifndef LINE_INTERSECT_H
#define LINE_INTERSECT_H

#include <cstdint>
#include <cstddef>

void line_intersect_cpu(const float* __restrict__ input_line_a, const float* __restrict__ input_line_b, uint32_t* __restrict__ output_intersect, const uint32_t N);

void line_intersect_dsa(const float* __restrict__ input_line_a, const float* __restrict__ input_line_b, uint32_t* __restrict__ output_intersect, const uint32_t N);

#endif // LINE_INTERSECT_H
