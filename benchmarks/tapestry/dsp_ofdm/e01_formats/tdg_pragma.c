/* Pragma-annotated C -- OFDM Receiver Chain (DSP domain)
 * E01 Productivity Comparison: pragma-based baseline format
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

typedef struct { float re, im; } cmplx_t;

#define FFT_N       4096
#define NUM_SC      1200
#define QAM_ORDER   64
#define CODED_BITS  7200
#define DATA_BITS   3600

#pragma tapestry graph(ofdm_receiver)

#pragma tapestry kernel(fft_butterfly, target=CGRA, source="fft_butterfly.c")
void fft_butterfly(cmplx_t *data, int n);

#pragma tapestry kernel(channel_est, target=CGRA, source="channel_est.c")
void channel_est(const cmplx_t *rx_pilots, cmplx_t *H_est,
                 int num_pilots, int num_sc);

#pragma tapestry kernel(equalizer, target=CGRA, source="equalizer.c")
void equalizer(const cmplx_t *rx, const cmplx_t *H, cmplx_t *eq, int n);

#pragma tapestry kernel(qam_demod, target=CGRA, source="qam_demod.c")
void qam_demod(const cmplx_t *symbols, int *bits, int n, int order);

#pragma tapestry kernel(viterbi, target=CGRA, source="viterbi.c")
void viterbi(const int *coded, int *decoded, int n_coded, int n_data);

#pragma tapestry kernel(crc_check, target=CGRA, source="crc_check.c")
void crc_check(const int *data, int *result, int n);

#pragma tapestry connect(fft_butterfly, channel_est, \
    ordering=FIFO, data_type=complex64, rate=4096, \
    tile_shape=[4096], visibility=LOCAL_SPM, \
    double_buffering=true, backpressure=BLOCK, \
    may_fuse=true, may_replicate=true, may_pipeline=true, \
    may_reorder=false, may_retile=true)

#pragma tapestry connect(channel_est, equalizer, \
    ordering=FIFO, data_type=complex64, rate=1200, \
    tile_shape=[1200], visibility=LOCAL_SPM, \
    double_buffering=false, backpressure=BLOCK, \
    may_fuse=true, may_replicate=true, may_pipeline=true, \
    may_reorder=false, may_retile=true)

#pragma tapestry connect(equalizer, qam_demod, \
    ordering=FIFO, data_type=complex64, rate=1200, \
    tile_shape=[128], visibility=LOCAL_SPM, \
    double_buffering=false, backpressure=BLOCK, \
    may_fuse=true, may_replicate=true, may_pipeline=true, \
    may_reorder=false, may_retile=true)

#pragma tapestry connect(qam_demod, viterbi, \
    ordering=FIFO, data_type=i32, rate=7200, \
    tile_shape=[7200], visibility=LOCAL_SPM, \
    double_buffering=true, backpressure=BLOCK, \
    may_fuse=true, may_replicate=true, may_pipeline=true, \
    may_reorder=false, may_retile=true)

#pragma tapestry connect(viterbi, crc_check, \
    ordering=FIFO, data_type=i32, rate=1800, \
    tile_shape=[3600], visibility=LOCAL_SPM, \
    double_buffering=false, backpressure=BLOCK, \
    may_fuse=true, may_replicate=true, may_pipeline=true, \
    may_reorder=false, may_retile=true)

void ofdm_pipeline(cmplx_t *rx_signal, int *decoded) {
    cmplx_t *freq_domain = (cmplx_t *)malloc(FFT_N * sizeof(cmplx_t));
    cmplx_t *H_est = (cmplx_t *)malloc(NUM_SC * sizeof(cmplx_t));
    cmplx_t *equalized = (cmplx_t *)malloc(NUM_SC * sizeof(cmplx_t));
    int *coded_bits = (int *)malloc(CODED_BITS * sizeof(int));

    memcpy(freq_domain, rx_signal, FFT_N * sizeof(cmplx_t));
    fft_butterfly(freq_domain, FFT_N);
    channel_est(freq_domain, H_est, 64, NUM_SC);
    equalizer(freq_domain, H_est, equalized, NUM_SC);
    qam_demod(equalized, coded_bits, NUM_SC, QAM_ORDER);
    viterbi(coded_bits, decoded, CODED_BITS, DATA_BITS);
    crc_check(decoded, decoded, DATA_BITS);

    free(freq_domain); free(H_est); free(equalized); free(coded_bits);
}
