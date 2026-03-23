#ifndef COMPLEX_OPS_H
#define COMPLEX_OPS_H

/*
 * Lightweight complex number operations for DSP kernels.
 * Uses a simple struct with real/imag float components.
 * Avoids C99 <complex.h> for maximum portability and
 * CGRA-compiler friendliness.
 */

typedef struct {
    float re;
    float im;
} cmplx_t;

static inline cmplx_t cmplx_make(float re, float im) {
    cmplx_t c;
    c.re = re;
    c.im = im;
    return c;
}

static inline cmplx_t cmplx_add(cmplx_t a, cmplx_t b) {
    cmplx_t c;
    c.re = a.re + b.re;
    c.im = a.im + b.im;
    return c;
}

static inline cmplx_t cmplx_sub(cmplx_t a, cmplx_t b) {
    cmplx_t c;
    c.re = a.re - b.re;
    c.im = a.im - b.im;
    return c;
}

static inline cmplx_t cmplx_mul(cmplx_t a, cmplx_t b) {
    cmplx_t c;
    c.re = a.re * b.re - a.im * b.im;
    c.im = a.re * b.im + a.im * b.re;
    return c;
}

static inline cmplx_t cmplx_conj(cmplx_t a) {
    cmplx_t c;
    c.re = a.re;
    c.im = -a.im;
    return c;
}

/* Complex division: a / b = a * conj(b) / |b|^2 */
static inline cmplx_t cmplx_div(cmplx_t a, cmplx_t b) {
    float denom = b.re * b.re + b.im * b.im;
    cmplx_t c;
    if (denom < 1e-12f && denom > -1e-12f) {
        c.re = 0.0f;
        c.im = 0.0f;
    } else {
        c.re = (a.re * b.re + a.im * b.im) / denom;
        c.im = (a.im * b.re - a.re * b.im) / denom;
    }
    return c;
}

static inline float cmplx_mag_sq(cmplx_t a) {
    return a.re * a.re + a.im * a.im;
}

static inline cmplx_t cmplx_scale(cmplx_t a, float s) {
    cmplx_t c;
    c.re = a.re * s;
    c.im = a.im * s;
    return c;
}

#endif /* COMPLEX_OPS_H */
