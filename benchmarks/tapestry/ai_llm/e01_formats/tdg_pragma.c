/* Pragma-annotated C -- Transformer Layer pipeline (AI/LLM domain)
 * E01 Productivity Comparison: pragma-based baseline format
 *
 * Each kernel function and each inter-kernel data edge must be
 * manually annotated with pragmas. The compiler extracts the TDG
 * from these annotations.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define SEQ_LEN   128
#define D_MODEL   512
#define D_FF      2048
#define NUM_HEADS 8
#define D_HEAD    (D_MODEL / NUM_HEADS)

#pragma tapestry graph(transformer_layer)

#pragma tapestry kernel(qkv_proj, target=CGRA, source="qkv_proj.c")
void qkv_proj(const float *A, const float *B, float *C,
              int M, int N, int K);

#pragma tapestry kernel(attn_score, target=CGRA, source="attn_score.c")
void attn_score(const float *Q, const float *K, float *S,
                int M, int N, int K_dim);

#pragma tapestry kernel(softmax, target=CGRA, source="softmax.c")
void softmax(float *data, int H, int N);

#pragma tapestry kernel(attn_output, target=CGRA, source="attn_output.c")
void attn_output(const float *S, const float *V, float *O,
                 int M, int N, int K);

#pragma tapestry kernel(ffn1, target=CGRA, source="ffn1.c")
void ffn1(const float *A, const float *B, float *C,
           int M, int N, int K);

#pragma tapestry kernel(gelu, target=CGRA, source="gelu.c")
void gelu(float *data, int n);

#pragma tapestry kernel(ffn2, target=CGRA, source="ffn2.c")
void ffn2(const float *A, const float *B, float *C,
           int M, int N, int K);

#pragma tapestry kernel(layernorm, target=CGRA, source="layernorm.c")
void layernorm(float *data, int rows, int cols);

#pragma tapestry connect(qkv_proj, attn_score, \
    ordering=FIFO, data_type=f32, rate=2048, \
    tile_shape=[32,64], visibility=LOCAL_SPM, \
    double_buffering=false, backpressure=BLOCK, \
    may_fuse=true, may_replicate=true, may_pipeline=true, \
    may_reorder=false, may_retile=true)

#pragma tapestry connect(attn_score, softmax, \
    ordering=FIFO, data_type=f32, rate=4096, \
    tile_shape=[32,128], visibility=LOCAL_SPM, \
    double_buffering=false, backpressure=BLOCK, \
    may_fuse=true, may_replicate=true, may_pipeline=true, \
    may_reorder=false, may_retile=true)

#pragma tapestry connect(softmax, attn_output, \
    ordering=FIFO, data_type=f32, rate=4096, \
    tile_shape=[32,128], visibility=LOCAL_SPM, \
    double_buffering=false, backpressure=BLOCK, \
    may_fuse=true, may_replicate=true, may_pipeline=true, \
    may_reorder=false, may_retile=true)

#pragma tapestry connect(attn_output, ffn1, \
    ordering=FIFO, data_type=f32, rate=16384, \
    tile_shape=[32,512], visibility=LOCAL_SPM, \
    double_buffering=true, backpressure=BLOCK, \
    may_fuse=true, may_replicate=true, may_pipeline=true, \
    may_reorder=false, may_retile=true)

#pragma tapestry connect(ffn1, gelu, \
    ordering=FIFO, data_type=f32, rate=65536, \
    tile_shape=[32,2048], visibility=LOCAL_SPM, \
    double_buffering=false, backpressure=BLOCK, \
    may_fuse=true, may_replicate=true, may_pipeline=true, \
    may_reorder=false, may_retile=true)

#pragma tapestry connect(gelu, ffn2, \
    ordering=FIFO, data_type=f32, rate=65536, \
    tile_shape=[32,2048], visibility=LOCAL_SPM, \
    double_buffering=false, backpressure=BLOCK, \
    may_fuse=true, may_replicate=true, may_pipeline=true, \
    may_reorder=false, may_retile=true)

#pragma tapestry connect(ffn2, layernorm, \
    ordering=FIFO, data_type=f32, rate=16384, \
    tile_shape=[32,512], visibility=LOCAL_SPM, \
    double_buffering=false, backpressure=BLOCK, \
    may_fuse=true, may_replicate=true, may_pipeline=true, \
    may_reorder=false, may_retile=true)

void transformer_pipeline(float *input, float *output) {
    float *qkv_out = (float *)malloc(SEQ_LEN * 3 * D_MODEL * sizeof(float));
    float *attn_scores = (float *)malloc(NUM_HEADS * SEQ_LEN * SEQ_LEN * sizeof(float));
    float *attn_out = (float *)malloc(SEQ_LEN * D_MODEL * sizeof(float));
    float *ffn_hidden = (float *)malloc(SEQ_LEN * D_FF * sizeof(float));
    float *Wqkv = (float *)malloc(D_MODEL * 3 * D_MODEL * sizeof(float));
    float *Wffn1 = (float *)malloc(D_MODEL * D_FF * sizeof(float));
    float *Wffn2 = (float *)malloc(D_FF * D_MODEL * sizeof(float));

    qkv_proj(input, Wqkv, qkv_out, SEQ_LEN, 3 * D_MODEL, D_MODEL);
    attn_score(qkv_out, qkv_out + SEQ_LEN * D_MODEL,
               attn_scores, SEQ_LEN, SEQ_LEN, D_HEAD);
    softmax(attn_scores, NUM_HEADS, SEQ_LEN);
    attn_output(attn_scores, qkv_out + 2 * SEQ_LEN * D_MODEL,
                attn_out, SEQ_LEN, D_MODEL, SEQ_LEN);
    ffn1(attn_out, Wffn1, ffn_hidden, SEQ_LEN, D_FF, D_MODEL);
    gelu(ffn_hidden, SEQ_LEN * D_FF);
    ffn2(ffn_hidden, Wffn2, output, SEQ_LEN, D_MODEL, D_FF);
    layernorm(output, SEQ_LEN, D_MODEL);

    free(qkv_out); free(attn_scores); free(attn_out);
    free(ffn_hidden); free(Wqkv); free(Wffn1); free(Wffn2);
}
