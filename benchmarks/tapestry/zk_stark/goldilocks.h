#ifndef GOLDILOCKS_H
#define GOLDILOCKS_H

/*
 * Goldilocks-64 field arithmetic: operations over p = 2^64 - 2^32 + 1.
 * This prime has the special form that enables efficient reduction.
 * All values are canonical: in range [0, p-1].
 */

#include <stdint.h>

#define GL64_P 0xFFFFFFFF00000001ULL

typedef uint64_t gl64_t;

/* Reduce to canonical form [0, p-1] */
static inline gl64_t gl64_reduce(gl64_t x) {
    return (x >= GL64_P) ? (x - GL64_P) : x;
}

/* Addition: (a + b) mod p */
static inline gl64_t gl64_add(gl64_t a, gl64_t b) {
    gl64_t sum = a + b;
    /* Detect overflow: if sum < a, we wrapped around 2^64 */
    if (sum < a || sum >= GL64_P) {
        sum -= GL64_P;
    }
    return sum;
}

/* Subtraction: (a - b) mod p */
static inline gl64_t gl64_sub(gl64_t a, gl64_t b) {
    if (a >= b) {
        return a - b;
    } else {
        return a + GL64_P - b;
    }
}

/*
 * Multiplication: (a * b) mod p
 * Uses 128-bit product and reduction specific to the Goldilocks prime.
 * p = 2^64 - 2^32 + 1, so 2^64 = 2^32 - 1 (mod p)
 */
static inline gl64_t gl64_mul(gl64_t a, gl64_t b) {
    /* Compute 128-bit product a*b = hi:lo */
#ifdef __SIZEOF_INT128__
    __uint128_t prod = (__uint128_t)a * (__uint128_t)b;
    uint64_t lo = (uint64_t)prod;
    uint64_t hi = (uint64_t)(prod >> 64);
#else
    /* Fallback: split into 32-bit parts */
    uint64_t a_lo = a & 0xFFFFFFFFULL;
    uint64_t a_hi = a >> 32;
    uint64_t b_lo = b & 0xFFFFFFFFULL;
    uint64_t b_hi = b >> 32;

    uint64_t p0 = a_lo * b_lo;
    uint64_t p1 = a_lo * b_hi;
    uint64_t p2 = a_hi * b_lo;
    uint64_t p3 = a_hi * b_hi;

    uint64_t mid = p1 + (p0 >> 32);
    mid += p2;
    /* carry from mid overflow */
    uint64_t carry = (mid < p2) ? 1ULL : 0ULL;

    uint64_t lo = (p0 & 0xFFFFFFFFULL) | ((mid & 0xFFFFFFFFULL) << 32);
    uint64_t hi = p3 + (mid >> 32) + (carry << 32);
#endif

    /*
     * Reduce hi:lo mod p where p = 2^64 - 2^32 + 1.
     * hi * 2^64 + lo = hi * (2^32 - 1) + lo (mod p)
     *
     * Let r = lo + hi * (2^32 - 1) = lo + (hi << 32) - hi
     * This may overflow, so handle carefully.
     */
    uint64_t hi_shifted = hi << 32;
    uint64_t t0 = lo - hi;
    uint64_t borrow = (lo < hi) ? 1ULL : 0ULL;
    uint64_t t1 = t0 + hi_shifted;
    uint64_t carry2 = (t1 < t0) ? 1ULL : 0ULL;

    /* Adjust for borrow and carry */
    /* Net adjustment = (carry2 - borrow) * 2^64 mod p
     * = (carry2 - borrow) * (2^32 - 1) mod p */
    if (carry2 > borrow) {
        /* Add (2^32 - 1) */
        uint64_t adj = 0xFFFFFFFFULL;
        t1 += adj;
        if (t1 < adj) {
            t1 += 0xFFFFFFFFULL; /* double overflow, very rare */
        }
    } else if (borrow > carry2) {
        /* Subtract (2^32 - 1), i.e. add p - (2^32-1) = 2^64 - 2*2^32 + 2 */
        /* Equivalently, subtract 0xFFFFFFFF and if underflow, add p */
        if (t1 >= 0xFFFFFFFFULL) {
            t1 -= 0xFFFFFFFFULL;
        } else {
            t1 = t1 + GL64_P - 0xFFFFFFFFULL;
        }
    }

    /* Final reduction */
    if (t1 >= GL64_P) {
        t1 -= GL64_P;
    }
    return t1;
}

/* Negation: (-a) mod p */
static inline gl64_t gl64_neg(gl64_t a) {
    return (a == 0) ? 0 : (GL64_P - a);
}

/* Exponentiation by squaring: a^exp mod p */
static inline gl64_t gl64_pow(gl64_t base, uint64_t exp) {
    gl64_t result = 1;
    gl64_t b = base;
    while (exp > 0) {
        if (exp & 1) {
            result = gl64_mul(result, b);
        }
        b = gl64_mul(b, b);
        exp >>= 1;
    }
    return result;
}

/* Multiplicative inverse: a^(-1) mod p = a^(p-2) mod p (Fermat) */
static inline gl64_t gl64_inv(gl64_t a) {
    return gl64_pow(a, GL64_P - 2);
}

#endif /* GOLDILOCKS_H */
