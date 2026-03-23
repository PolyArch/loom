// Self-attention baseline on GPU.
// Implements: Attention(Q, K, V) = softmax(Q * K^T / sqrt(d_k)) * V
// Uses cuBLAS for matmul portions, custom kernel for softmax.

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include <cublas_v2.h>
#include <cuda_runtime.h>

// -----------------------------------------------------------------------
// Error checking
// -----------------------------------------------------------------------

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err = (call);                                                  \
    if (err != cudaSuccess) {                                                  \
      std::fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,   \
                   cudaGetErrorString(err));                                    \
      std::exit(1);                                                            \
    }                                                                          \
  } while (0)

#define CUBLAS_CHECK(call)                                                     \
  do {                                                                         \
    cublasStatus_t st = (call);                                                \
    if (st != CUBLAS_STATUS_SUCCESS) {                                         \
      std::fprintf(stderr, "cuBLAS error at %s:%d: %d\n", __FILE__, __LINE__, \
                   static_cast<int>(st));                                      \
      std::exit(1);                                                            \
    }                                                                          \
  } while (0)

// -----------------------------------------------------------------------
// Row-wise softmax kernel with warp-level reduction
// -----------------------------------------------------------------------

__global__ void softmax_kernel(float *data, int rows, int cols) {
  int row = blockIdx.x;
  if (row >= rows)
    return;

  float *row_data = data + row * cols;

  // Find max for numerical stability.
  float max_val = -1e30f;
  for (int c = threadIdx.x; c < cols; c += blockDim.x) {
    if (row_data[c] > max_val)
      max_val = row_data[c];
  }

  // Warp reduction for max.
  for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
    float other = __shfl_down_sync(0xffffffff, max_val, offset);
    if (other > max_val)
      max_val = other;
  }
  // Cross-warp via shared memory.
  __shared__ float shared_max[32];
  int lane = threadIdx.x % warpSize;
  int warp_id = threadIdx.x / warpSize;
  if (lane == 0)
    shared_max[warp_id] = max_val;
  __syncthreads();
  if (threadIdx.x < blockDim.x / warpSize) {
    max_val = shared_max[threadIdx.x];
  } else {
    max_val = -1e30f;
  }
  for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
    float other = __shfl_down_sync(0xffffffff, max_val, offset);
    if (other > max_val)
      max_val = other;
  }
  max_val = __shfl_sync(0xffffffff, max_val, 0);

  // Exponentiate and sum.
  float sum_exp = 0.0f;
  for (int c = threadIdx.x; c < cols; c += blockDim.x) {
    float e = expf(row_data[c] - max_val);
    row_data[c] = e;
    sum_exp += e;
  }

  // Warp reduction for sum.
  for (int offset = warpSize / 2; offset > 0; offset >>= 1)
    sum_exp += __shfl_down_sync(0xffffffff, sum_exp, offset);
  __shared__ float shared_sum[32];
  if (lane == 0)
    shared_sum[warp_id] = sum_exp;
  __syncthreads();
  if (threadIdx.x < blockDim.x / warpSize)
    sum_exp = shared_sum[threadIdx.x];
  else
    sum_exp = 0.0f;
  for (int offset = warpSize / 2; offset > 0; offset >>= 1)
    sum_exp += __shfl_down_sync(0xffffffff, sum_exp, offset);
  sum_exp = __shfl_sync(0xffffffff, sum_exp, 0);

  // Normalize.
  float inv_sum = 1.0f / sum_exp;
  for (int c = threadIdx.x; c < cols; c += blockDim.x) {
    row_data[c] *= inv_sum;
  }
}

// -----------------------------------------------------------------------
// Timing helper
// -----------------------------------------------------------------------

struct GpuTimer {
  cudaEvent_t start, stop;
  GpuTimer() {
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
  }
  ~GpuTimer() {
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
  }
  void begin() { CUDA_CHECK(cudaEventRecord(start, 0)); }
  float end_ms() {
    CUDA_CHECK(cudaEventRecord(stop, 0));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    return ms;
  }
};

struct BenchResult {
  double mean_ms;
  double stddev_ms;
  double min_ms;
};

template <typename Fn>
BenchResult run_benchmark(Fn &&fn, int warmup = 3, int trials = 10) {
  GpuTimer timer;
  std::vector<double> times;

  for (int i = 0; i < warmup; ++i) {
    timer.begin();
    fn();
    timer.end_ms();
  }

  for (int i = 0; i < trials; ++i) {
    timer.begin();
    fn();
    double ms = timer.end_ms();
    times.push_back(ms);
  }

  double sum = 0.0, mn = 1e30;
  for (double t : times) {
    sum += t;
    if (t < mn) mn = t;
  }
  double mean = sum / times.size();
  double sq = 0.0;
  for (double t : times)
    sq += (t - mean) * (t - mean);
  double sd = std::sqrt(sq / (times.size() - 1));
  return {mean, sd, mn};
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

  // FLOPs: 2 matmuls (QK^T and attn*V) + softmax.
  // QK^T: 2*seq*seq*d_k per head.
  // attn*V: 2*seq*d_k*seq per head.
  // Softmax: ~5*seq*seq per head (exp, sum, div).
  double total_ops = total_heads * (2.0 * 2.0 * seq_len * seq_len * d_k +
                                     5.0 * seq_len * seq_len);

  // Allocations for Q, K, V, scores, output (per head).
  size_t qkv_bytes = (size_t)total_heads * seq_len * d_k * sizeof(float);
  size_t score_bytes = (size_t)total_heads * seq_len * seq_len * sizeof(float);

  std::vector<float> h_q(total_heads * seq_len * d_k);
  std::vector<float> h_k(total_heads * seq_len * d_k);
  std::vector<float> h_v(total_heads * seq_len * d_k);

  for (size_t i = 0; i < h_q.size(); ++i) {
    h_q[i] = static_cast<float>(rand()) / RAND_MAX * 0.1f;
    h_k[i] = static_cast<float>(rand()) / RAND_MAX * 0.1f;
    h_v[i] = static_cast<float>(rand()) / RAND_MAX * 0.1f;
  }

  float *d_q, *d_k_dev, *d_v, *d_scores, *d_out;
  CUDA_CHECK(cudaMalloc(&d_q, qkv_bytes));
  CUDA_CHECK(cudaMalloc(&d_k_dev, qkv_bytes));
  CUDA_CHECK(cudaMalloc(&d_v, qkv_bytes));
  CUDA_CHECK(cudaMalloc(&d_scores, score_bytes));
  CUDA_CHECK(cudaMalloc(&d_out, qkv_bytes));

  CUDA_CHECK(cudaMemcpy(d_q, h_q.data(), qkv_bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_k_dev, h_k.data(), qkv_bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_v, h_v.data(), qkv_bytes, cudaMemcpyHostToDevice));

  cublasHandle_t handle;
  CUBLAS_CHECK(cublasCreate(&handle));

  float scale = 1.0f / std::sqrt(static_cast<float>(d_k));
  float zero = 0.0f;
  float one = 1.0f;

  int softmax_threads = 256;

  auto attn_fn = [&]() {
    // Batched: Q * K^T -> scores for each head.
    CUBLAS_CHECK(cublasSgemmStridedBatched(
        handle, CUBLAS_OP_T, CUBLAS_OP_N, seq_len, seq_len, d_k, &scale,
        d_k_dev, d_k, seq_len * d_k, d_q, d_k, seq_len * d_k, &zero,
        d_scores, seq_len, seq_len * seq_len, total_heads));

    // Softmax over scores rows.
    softmax_kernel<<<total_heads * seq_len, softmax_threads>>>(
        d_scores, total_heads * seq_len, seq_len);

    // scores * V -> output.
    CUBLAS_CHECK(cublasSgemmStridedBatched(
        handle, CUBLAS_OP_N, CUBLAS_OP_N, d_k, seq_len, seq_len, &one,
        d_v, d_k, seq_len * d_k, d_scores, seq_len, seq_len * seq_len, &zero,
        d_out, d_k, seq_len * d_k, total_heads));
  };

  BenchResult res = run_benchmark(attn_fn);

  std::printf("RESULT kernel=attention_gpu mean_ms=%.4f stddev_ms=%.4f "
              "min_ms=%.4f total_ops=%.0f batch=%d heads=%d seq_len=%d d_k=%d\n",
              res.mean_ms, res.stddev_ms, res.min_ms, total_ops, batch, heads,
              seq_len, d_k);

  CUBLAS_CHECK(cublasDestroy(handle));
  CUDA_CHECK(cudaFree(d_q));
  CUDA_CHECK(cudaFree(d_k_dev));
  CUDA_CHECK(cudaFree(d_v));
  CUDA_CHECK(cudaFree(d_scores));
  CUDA_CHECK(cudaFree(d_out));

  return 0;
}
