// Self-attention CPU baseline.
// Implements: Attention(Q, K, V) = softmax(Q * K^T / sqrt(d_k)) * V
// Uses OpenMP for parallelism across heads.

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "../common/timer.h"

#ifdef _OPENMP
#include <omp.h>
#endif

// -----------------------------------------------------------------------
// Row-wise softmax
// -----------------------------------------------------------------------

static void softmax_rows(float *data, int rows, int cols) {
  for (int r = 0; r < rows; ++r) {
    float *row = data + r * cols;

    // Max for numerical stability.
    float max_val = row[0];
    for (int c = 1; c < cols; ++c) {
      if (row[c] > max_val)
        max_val = row[c];
    }

    // Exponentiate and sum.
    float sum_exp = 0.0f;
    for (int c = 0; c < cols; ++c) {
      row[c] = std::exp(row[c] - max_val);
      sum_exp += row[c];
    }

    // Normalize.
    float inv_sum = 1.0f / sum_exp;
    for (int c = 0; c < cols; ++c) {
      row[c] *= inv_sum;
    }
  }
}

// -----------------------------------------------------------------------
// Dense matmul helpers (small, for per-head operations)
// -----------------------------------------------------------------------

// C = alpha * A * B + beta * C
static void matmul(const float *A, const float *B, float *C, int M, int N,
                   int K, float alpha = 1.0f, float beta = 0.0f) {
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      float sum = 0.0f;
      for (int k = 0; k < K; ++k) {
        sum += A[i * K + k] * B[k * N + j];
      }
      C[i * N + j] = alpha * sum + beta * C[i * N + j];
    }
  }
}

// C = alpha * A * B^T + beta * C
static void matmul_abt(const float *A, const float *B, float *C, int M, int N,
                       int K, float alpha = 1.0f, float beta = 0.0f) {
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      float sum = 0.0f;
      for (int k = 0; k < K; ++k) {
        sum += A[i * K + k] * B[j * K + k];
      }
      C[i * N + j] = alpha * sum + beta * C[i * N + j];
    }
  }
}

// -----------------------------------------------------------------------
// Single-head attention
// -----------------------------------------------------------------------

static void attention_single_head(const float *Q, const float *K,
                                  const float *V, float *out, float *scores,
                                  int seq_len, int d_k) {
  float scale = 1.0f / std::sqrt(static_cast<float>(d_k));

  // scores = Q * K^T * scale  (seq_len x seq_len)
  matmul_abt(Q, K, scores, seq_len, seq_len, d_k, scale, 0.0f);

  // softmax(scores) row-wise
  softmax_rows(scores, seq_len, seq_len);

  // out = scores * V  (seq_len x d_k)
  matmul(scores, V, out, seq_len, d_k, seq_len);
}

// -----------------------------------------------------------------------
// Main
// -----------------------------------------------------------------------

int main(int argc, char **argv) {
  int batch = 1, heads = 8, seq_len = 512, d_k = 64;
  if (argc >= 5) {
    batch = std::atoi(argv[1]);
    heads = std::atoi(argv[2]);
    seq_len = std::atoi(argv[3]);
    d_k = std::atoi(argv[4]);
  }

  int total_heads = batch * heads;

  // FLOPs: per head: 2 matmuls + softmax.
  double total_ops = total_heads * (2.0 * 2.0 * seq_len * seq_len * d_k +
                                     5.0 * seq_len * seq_len);

  // Allocate Q, K, V for all heads.
  size_t head_qkv_size = static_cast<size_t>(seq_len) * d_k;
  size_t head_score_size = static_cast<size_t>(seq_len) * seq_len;

  std::vector<float> Q(total_heads * head_qkv_size);
  std::vector<float> K(total_heads * head_qkv_size);
  std::vector<float> V(total_heads * head_qkv_size);
  std::vector<float> out(total_heads * head_qkv_size);
  std::vector<float> scores(total_heads * head_score_size);

  for (size_t i = 0; i < Q.size(); ++i) {
    Q[i] = static_cast<float>(rand()) / RAND_MAX * 0.1f;
    K[i] = static_cast<float>(rand()) / RAND_MAX * 0.1f;
    V[i] = static_cast<float>(rand()) / RAND_MAX * 0.1f;
  }

  auto attn_fn = [&]() {
#pragma omp parallel for schedule(dynamic)
    for (int h = 0; h < total_heads; ++h) {
      attention_single_head(Q.data() + h * head_qkv_size,
                            K.data() + h * head_qkv_size,
                            V.data() + h * head_qkv_size,
                            out.data() + h * head_qkv_size,
                            scores.data() + h * head_score_size, seq_len, d_k);
    }
  };

  auto timer = baselines::benchmark(attn_fn);

  std::printf("RESULT kernel=attention_cpu mean_ms=%.4f stddev_ms=%.4f "
              "min_ms=%.4f total_ops=%.0f batch=%d heads=%d seq_len=%d d_k=%d\n",
              timer.mean_ms(), timer.stddev_ms(), timer.min_ms(), total_ops,
              batch, heads, seq_len, d_k);

  return 0;
}
