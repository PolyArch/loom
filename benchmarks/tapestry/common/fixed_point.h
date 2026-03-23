#ifndef FIXED_POINT_H
#define FIXED_POINT_H

#include <stdint.h>

/*
 * Fixed-point arithmetic helpers for INT8 quantized kernels.
 * Uses Q7.0 format (plain int8) with int32 accumulators and
 * a configurable fractional-bit shift for dequantization.
 */

#define FP_FRAC_BITS 7

/* Multiply two int8 values, accumulate into int32 */
static inline int32_t fp_mul_acc(int32_t acc, int8_t a, int8_t b) {
    return acc + (int32_t)a * (int32_t)b;
}

/* Saturating cast from int32 accumulator back to int8 with right-shift */
static inline int8_t fp_saturate(int32_t val, int shift) {
    val = val >> shift;
    if (val > 127) return 127;
    if (val < -128) return -128;
    return (int8_t)val;
}

/* Convert float to fixed-point int8 (Q0.7 style) */
static inline int8_t fp_from_float(float f) {
    int val = (int)(f * (1 << FP_FRAC_BITS));
    if (val > 127) return 127;
    if (val < -128) return -128;
    return (int8_t)val;
}

/* Convert fixed-point int8 back to float */
static inline float fp_to_float(int8_t v) {
    return (float)v / (float)(1 << FP_FRAC_BITS);
}

#endif /* FIXED_POINT_H */
