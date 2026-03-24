/*
 * Entry function for auto_analyze: Transformer Layer pipeline.
 * Contains calls to all kernel functions with shared buffer arguments.
 * auto_analyze should detect 8 kernels and 7 edges.
 */

#include <stdlib.h>
#include <string.h>

#define SEQ_LEN   128
#define D_MODEL   512
#define D_FF      2048
#define NUM_HEADS 8
#define D_HEAD    (D_MODEL / NUM_HEADS)

/* Kernel function declarations (noinline to preserve call boundaries) */
__attribute__((noinline))
void qkv_proj(const float *A, const float *B, float *C,
              int M, int N, int K) {
    int i, j, k;
    memset(C, 0, (size_t)M * N * sizeof(float));
    for (i = 0; i < M; i++)
        for (j = 0; j < N; j++)
            for (k = 0; k < K; k++)
                C[i * N + j] += A[i * K + k] * B[k * N + j];
}

__attribute__((noinline))
void attn_score(const float *Q, const float *K_mat, float *S,
                int M, int N, int K) {
    int i, j, k;
    memset(S, 0, (size_t)M * N * sizeof(float));
    for (i = 0; i < M; i++)
        for (j = 0; j < N; j++)
            for (k = 0; k < K; k++)
                S[i * N + j] += Q[i * K + k] * K_mat[j * K + k];
}

__attribute__((noinline))
void softmax(float *data, int H, int N) {
    int h, i, j;
    for (h = 0; h < H; h++) {
        float *head = data + (size_t)h * N * N;
        for (i = 0; i < N; i++) {
            float *row = head + (size_t)i * N;
            float mx = row[0];
            for (j = 1; j < N; j++)
                if (row[j] > mx) mx = row[j];
            float sum = 0.0f;
            for (j = 0; j < N; j++) {
                row[j] = row[j] - mx;
                sum += row[j];
            }
            float inv = 1.0f / sum;
            for (j = 0; j < N; j++)
                row[j] *= inv;
        }
    }
}

__attribute__((noinline))
void attn_output(const float *S, const float *V, float *O,
                 int M, int N, int K) {
    int i, j, k;
    memset(O, 0, (size_t)M * N * sizeof(float));
    for (i = 0; i < M; i++)
        for (j = 0; j < N; j++)
            for (k = 0; k < K; k++)
                O[i * N + j] += S[i * K + k] * V[k * N + j];
}

__attribute__((noinline))
void ffn1(const float *A, const float *W, float *C,
           int M, int N, int K) {
    int i, j, k;
    memset(C, 0, (size_t)M * N * sizeof(float));
    for (i = 0; i < M; i++)
        for (j = 0; j < N; j++)
            for (k = 0; k < K; k++)
                C[i * N + j] += A[i * K + k] * W[k * N + j];
}

__attribute__((noinline))
void gelu(float *data, int n) {
    int i;
    for (i = 0; i < n; i++) {
        float x = data[i];
        data[i] = 0.5f * x * (1.0f + x * (0.7978845608f +
                  x * x * 0.0356774f));
    }
}

__attribute__((noinline))
void ffn2(const float *A, const float *W, float *C,
           int M, int N, int K) {
    int i, j, k;
    memset(C, 0, (size_t)M * N * sizeof(float));
    for (i = 0; i < M; i++)
        for (j = 0; j < N; j++)
            for (k = 0; k < K; k++)
                C[i * N + j] += A[i * K + k] * W[k * N + j];
}

__attribute__((noinline))
void layernorm(float *data, int rows, int cols) {
    int i, j;
    for (i = 0; i < rows; i++) {
        float *row = data + (size_t)i * cols;
        float mean = 0.0f;
        for (j = 0; j < cols; j++) mean += row[j];
        mean /= (float)cols;
        float var = 0.0f;
        for (j = 0; j < cols; j++) {
            float d = row[j] - mean;
            var += d * d;
        }
        var /= (float)cols;
        float inv_std = 1.0f / (var + 1e-5f);
        for (j = 0; j < cols; j++)
            row[j] = (row[j] - mean) * inv_std;
    }
}

/* Entry function for auto_analyze */
void transformer_pipeline(float *input, float *output,
                          const float *Wqkv, const float *Wffn1,
                          const float *Wffn2) {
    float *qkv_out = (float *)malloc(SEQ_LEN * 3 * D_MODEL * sizeof(float));
    float *attn_scores = (float *)malloc(
        (size_t)NUM_HEADS * SEQ_LEN * SEQ_LEN * sizeof(float));
    float *attn_out = (float *)malloc(SEQ_LEN * D_MODEL * sizeof(float));
    float *ffn_hidden = (float *)malloc(SEQ_LEN * D_FF * sizeof(float));

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

    free(qkv_out);
    free(attn_scores);
    free(attn_out);
    free(ffn_hidden);
}
