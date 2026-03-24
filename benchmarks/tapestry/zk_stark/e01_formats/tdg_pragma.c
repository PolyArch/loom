/* Pragma-annotated C -- STARK Proof pipeline (ZK domain)
 * E01 Productivity Comparison: pragma-based baseline format
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef unsigned int m31_t;

#define NTT_N       1024
#define NTT_LOG_N   10
#define NUM_QUERIES 256
#define HASH_RATE   4
#define MSM_SIZE    1024

#pragma tapestry graph(stark_proof)

#pragma tapestry kernel(ntt, target=CGRA, source="ntt.c")
void ntt_forward_tiled(m31_t *data, int n, int log_n, const m31_t *tw);

#pragma tapestry kernel(msm, target=CGRA, source="msm.c")
void msm(const m31_t *scalars, const m31_t *bases, m31_t *result, int n);

#pragma tapestry kernel(poseidon_hash, target=CGRA, source="poseidon_hash.c")
void poseidon_hash(const m31_t *input, m31_t *output,
                   int n_blocks, int rate);

#pragma tapestry kernel(poly_eval, target=CGRA, source="poly_eval.c")
void poly_eval(const m31_t *coeffs, const m31_t *points,
               m31_t *results, int degree, int n_points);

#pragma tapestry kernel(proof_compose, target=CGRA, source="proof_compose.c")
void proof_compose(const m31_t *poly_vals, const m31_t *hash_out,
                   const m31_t *msm_out, m31_t *proof, int n);

#pragma tapestry connect(ntt, poly_eval, \
    ordering=FIFO, data_type=u32, rate=1024, \
    tile_shape=[1024], visibility=LOCAL_SPM, \
    double_buffering=true, backpressure=BLOCK, \
    may_fuse=true, may_replicate=true, may_pipeline=true, \
    may_reorder=false, may_retile=true)

#pragma tapestry connect(poly_eval, proof_compose, \
    ordering=FIFO, data_type=u32, rate=256, \
    tile_shape=[256], visibility=LOCAL_SPM, \
    double_buffering=false, backpressure=BLOCK, \
    may_fuse=true, may_replicate=true, may_pipeline=true, \
    may_reorder=false, may_retile=true)

#pragma tapestry connect(poseidon_hash, proof_compose, \
    ordering=FIFO, data_type=u32, rate=4, \
    tile_shape=[4], visibility=LOCAL_SPM, \
    double_buffering=false, backpressure=BLOCK, \
    may_fuse=true, may_replicate=true, may_pipeline=true, \
    may_reorder=false, may_retile=true)

#pragma tapestry connect(msm, proof_compose, \
    ordering=FIFO, data_type=u32, rate=3, \
    tile_shape=[3], visibility=LOCAL_SPM, \
    double_buffering=false, backpressure=BLOCK, \
    may_fuse=true, may_replicate=true, may_pipeline=true, \
    may_reorder=false, may_retile=true)

#pragma tapestry connect(ntt, poseidon_hash, \
    ordering=FIFO, data_type=u32, rate=8, \
    tile_shape=[8], visibility=LOCAL_SPM, \
    double_buffering=false, backpressure=BLOCK, \
    may_fuse=true, may_replicate=true, may_pipeline=true, \
    may_reorder=false, may_retile=true)

void stark_pipeline(m31_t *trace, m31_t *proof) {
    m31_t twiddles[NTT_N / 2];
    m31_t ntt_out[NTT_N];
    m31_t poly_vals[NUM_QUERIES];
    m31_t hash_out[HASH_RATE];
    m31_t msm_result[3];
    m31_t scalars[MSM_SIZE], bases[MSM_SIZE * 3];
    m31_t query_pts[NUM_QUERIES];

    memcpy(ntt_out, trace, NTT_N * sizeof(m31_t));
    ntt_forward_tiled(ntt_out, NTT_N, NTT_LOG_N, twiddles);
    msm(scalars, bases, msm_result, MSM_SIZE);
    poseidon_hash(ntt_out, hash_out, NTT_N / HASH_RATE, HASH_RATE);
    poly_eval(ntt_out, query_pts, poly_vals, NTT_N, NUM_QUERIES);
    proof_compose(poly_vals, hash_out, msm_result, proof, NUM_QUERIES);
}
