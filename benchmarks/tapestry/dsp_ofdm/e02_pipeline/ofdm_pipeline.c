/*
 * Entry function for auto_analyze: OFDM Receiver Chain pipeline.
 * Contains calls to all kernel functions with shared buffer arguments.
 * auto_analyze should detect 6 kernels and 5 edges.
 */

#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define FFT_N     4096
#define NUM_SC    1200
#define QAM_ORDER 64
#define CODE_RATE 2
#define CODED_BITS (NUM_SC * 6 * CODE_RATE)
#define DATA_BITS  (NUM_SC * 6)

typedef struct { float re, im; } cmplx_t;

__attribute__((noinline))
void fft_butterfly(cmplx_t *data, int n) {
    int stage_size, group, k;
    for (stage_size = 2; stage_size <= n; stage_size *= 2) {
        int half = stage_size / 2;
        float angle_step = -2.0f * (float)M_PI / (float)stage_size;
        for (group = 0; group < n; group += stage_size) {
            for (k = 0; k < half; k++) {
                float angle = angle_step * (float)k;
                float wr = cosf(angle), wi = sinf(angle);
                cmplx_t t;
                t.re = wr * data[group+k+half].re - wi * data[group+k+half].im;
                t.im = wr * data[group+k+half].im + wi * data[group+k+half].re;
                cmplx_t u = data[group+k];
                data[group+k].re = u.re + t.re;
                data[group+k].im = u.im + t.im;
                data[group+k+half].re = u.re - t.re;
                data[group+k+half].im = u.im - t.im;
            }
        }
    }
}

__attribute__((noinline))
void channel_est(const cmplx_t *rx, cmplx_t *H_est,
                 int num_pilots, int num_sc) {
    int i;
    for (i = 0; i < num_sc; i++) {
        int pilot_idx = i % num_pilots;
        H_est[i] = rx[pilot_idx * (num_sc / num_pilots)];
    }
}

__attribute__((noinline))
void equalizer(const cmplx_t *rx, const cmplx_t *H, cmplx_t *eq, int n) {
    int i;
    for (i = 0; i < n; i++) {
        float denom = H[i].re * H[i].re + H[i].im * H[i].im + 1e-10f;
        eq[i].re = (rx[i].re * H[i].re + rx[i].im * H[i].im) / denom;
        eq[i].im = (rx[i].im * H[i].re - rx[i].re * H[i].im) / denom;
    }
}

__attribute__((noinline))
void qam_demod(const cmplx_t *symbols, int *bits, int n, int order) {
    int i, j;
    int bps = 0, tmp = order;
    while (tmp > 1) { bps++; tmp >>= 1; }
    for (i = 0; i < n; i++) {
        int sym = (int)(symbols[i].re * 4.0f + 4.0f);
        if (sym < 0) sym = 0;
        if (sym >= order) sym = order - 1;
        for (j = 0; j < bps; j++)
            bits[i * bps + j] = (sym >> j) & 1;
    }
}

__attribute__((noinline))
void viterbi(const int *coded, int *decoded, int n_coded, int n_data) {
    int i;
    for (i = 0; i < n_data; i++)
        decoded[i] = coded[i * 2] ^ coded[i * 2 + 1];
}

__attribute__((noinline))
void crc_check(const int *data, int *result, int n) {
    unsigned crc = 0;
    int i;
    for (i = 0; i < n; i++)
        crc = (crc << 1) ^ (unsigned)data[i];
    result[0] = (int)(crc & 0xFFFF);
}

/* Entry function for auto_analyze */
void ofdm_pipeline(cmplx_t *rx_signal, int *decoded_out) {
    cmplx_t *freq = (cmplx_t *)malloc(FFT_N * sizeof(cmplx_t));
    cmplx_t *H_est = (cmplx_t *)malloc(NUM_SC * sizeof(cmplx_t));
    cmplx_t *equalized = (cmplx_t *)malloc(NUM_SC * sizeof(cmplx_t));
    int *coded_bits = (int *)malloc(CODED_BITS * sizeof(int));

    memcpy(freq, rx_signal, FFT_N * sizeof(cmplx_t));
    fft_butterfly(freq, FFT_N);
    channel_est(freq, H_est, 64, NUM_SC);
    equalizer(freq, H_est, equalized, NUM_SC);
    qam_demod(equalized, coded_bits, NUM_SC, QAM_ORDER);
    viterbi(coded_bits, decoded_out, CODED_BITS, DATA_BITS);
    crc_check(decoded_out, decoded_out, DATA_BITS);

    free(freq); free(H_est); free(equalized); free(coded_bits);
}
