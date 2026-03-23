// Matrix multiply baseline: cuBLAS SGEMM + naive CUDA kernel.
// Reports timing in the RESULT format consumed by run_gpu_baselines.py.

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
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
// Naive CUDA matmul kernel (for comparison)
// -----------------------------------------------------------------------

__global__ void matmul_naive(const float *A, const float *B, float *C, int M,
                             int N, int K) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row < M && col < N) {
    float sum = 0.0f;
    for (int k = 0; k < K; ++k) {
      sum += A[row * K + k] * B[k * N + col];
    }
    C[row * N + col] = sum;
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

// -----------------------------------------------------------------------
// Benchmark driver
// -----------------------------------------------------------------------

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

  double sum = 0.0;
  double mn = 1e30;
  for (double t : times) {
    sum += t;
    if (t < mn)
      mn = t;
  }
  double mean = sum / times.size();
  double sq = 0.0;
  for (double t : times) {
    sq += (t - mean) * (t - mean);
  }
  double sd = std::sqrt(sq / (times.size() - 1));

  return {mean, sd, mn};
}

// -----------------------------------------------------------------------
// Main
// -----------------------------------------------------------------------

static void print_result(const char *name, int M, int N, int K,
                         const BenchResult &r) {
  double total_ops = 2.0 * M * N * K; // FLOPs for matmul
  std::printf("RESULT kernel=%s mean_ms=%.4f stddev_ms=%.4f min_ms=%.4f "
              "total_ops=%.0f M=%d N=%d K=%d\n",
              name, r.mean_ms, r.stddev_ms, r.min_ms, total_ops, M, N, K);
}

int main(int argc, char **argv) {
  // Default sizes; can override via command line.
  int M = 1024, N = 1024, K = 1024;
  if (argc >= 4) {
    M = std::atoi(argv[1]);
    N = std::atoi(argv[2]);
    K = std::atoi(argv[3]);
  }

  size_t sizeA = (size_t)M * K * sizeof(float);
  size_t sizeB = (size_t)K * N * sizeof(float);
  size_t sizeC = (size_t)M * N * sizeof(float);

  // Host allocation and initialization.
  std::vector<float> hA(M * K), hB(K * N), hC(M * N);
  for (size_t i = 0; i < hA.size(); ++i)
    hA[i] = static_cast<float>(rand()) / RAND_MAX;
  for (size_t i = 0; i < hB.size(); ++i)
    hB[i] = static_cast<float>(rand()) / RAND_MAX;

  // Device allocation.
  float *dA, *dB, *dC;
  CUDA_CHECK(cudaMalloc(&dA, sizeA));
  CUDA_CHECK(cudaMalloc(&dB, sizeB));
  CUDA_CHECK(cudaMalloc(&dC, sizeC));
  CUDA_CHECK(cudaMemcpy(dA, hA.data(), sizeA, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dB, hB.data(), sizeB, cudaMemcpyHostToDevice));

  // -- cuBLAS benchmark --
  cublasHandle_t handle;
  CUBLAS_CHECK(cublasCreate(&handle));
  float alpha = 1.0f, beta = 0.0f;

  auto cublas_fn = [&]() {
    CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha,
                             dB, N, dA, K, &beta, dC, N));
  };

  BenchResult cublas_res = run_benchmark(cublas_fn);
  print_result("matmul_cublas", M, N, K, cublas_res);

  // -- Naive CUDA kernel benchmark --
  dim3 block(16, 16);
  dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);

  auto naive_fn = [&]() { matmul_naive<<<grid, block>>>(dA, dB, dC, M, N, K); };

  BenchResult naive_res = run_benchmark(naive_fn);
  print_result("matmul_naive_cuda", M, N, K, naive_res);

  // Cleanup.
  CUBLAS_CHECK(cublasDestroy(handle));
  CUDA_CHECK(cudaFree(dA));
  CUDA_CHECK(cudaFree(dB));
  CUDA_CHECK(cudaFree(dC));

  return 0;
}
