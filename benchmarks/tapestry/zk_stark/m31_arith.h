#ifndef M31_ARITH_H
#define M31_ARITH_H

/*
 * M31 field arithmetic: operations over the Mersenne prime p = 2^31 - 1.
 * Provides add, subtract, multiply, power, and inverse.
 * All values are canonical: in range [0, p-1] where p = 0x7FFFFFFF.
 */

#include <stdint.h>

#define M31_P 0x7FFFFFFFU  /* 2^31 - 1 = 2147483647 */

typedef uint32_t m31_t;

/* Reduce value to canonical form [0, p-1] */
static inline m31_t m31_reduce(uint32_t x) {
    /* If x == p, return 0; if x > p that means overflow handled elsewhere */
    return (x >= M31_P) ? (x - M31_P) : x;
}

/* Addition: (a + b) mod p */
static inline m31_t m31_add(m31_t a, m31_t b) {
    uint32_t sum = a + b;
    return m31_reduce(sum);
}

/* Subtraction: (a - b) mod p */
static inline m31_t m31_sub(m31_t a, m31_t b) {
    /* If a >= b, result is a-b; otherwise a-b+p */
    return (a >= b) ? (a - b) : (a + M31_P - b);
}

/* Multiplication: (a * b) mod p using Mersenne reduction */
static inline m31_t m31_mul(m31_t a, m31_t b) {
    uint64_t prod = (uint64_t)a * (uint64_t)b;
    /* Split product: prod = hi * 2^31 + lo */
    uint32_t lo = (uint32_t)(prod & M31_P);
    uint32_t hi = (uint32_t)(prod >> 31);
    /* (hi * 2^31 + lo) mod (2^31-1) = hi + lo mod p */
    return m31_add(lo, hi);
}

/* Negation: (-a) mod p */
static inline m31_t m31_neg(m31_t a) {
    return (a == 0) ? 0 : (M31_P - a);
}

/* Exponentiation by squaring: a^exp mod p */
static inline m31_t m31_pow(m31_t base, uint32_t exp) {
    m31_t result = 1;
    m31_t b = base;
    while (exp > 0) {
        if (exp & 1) {
            result = m31_mul(result, b);
        }
        b = m31_mul(b, b);
        exp >>= 1;
    }
    return result;
}

/* Multiplicative inverse: a^(-1) mod p = a^(p-2) mod p (Fermat) */
static inline m31_t m31_inv(m31_t a) {
    return m31_pow(a, M31_P - 2);
}

#endif /* M31_ARITH_H */
