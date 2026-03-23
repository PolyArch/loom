// FFT baseline using cuFFT.
// Reports timing in the RESULT format consumed by run_gpu_baselines.py.

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include <cuda_runtime.h>
#include <cufft.h>

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

#define CUFFT_CHECK(call)                                                      \
  do {                                                                         \
    cufftResult res = (call);                                                  \
    if (res != CUFFT_SUCCESS) {                                                \
      std::fprintf(stderr, "cuFFT error at %s:%d: %d\n", __FILE__, __LINE__,  \
                   static_cast<int>(res));                                      \
      std::exit(1);                                                            \
    }                                                                          \
  } while (0)

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
    if (t < mn)
      mn = t;
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
  int N = 1 << 20; // 1M-point FFT by default.
  if (argc >= 2)
    N = std::atoi(argv[1]);

  // Total ops: 5 * N * log2(N) (standard FFT complexity estimate).
  double total_ops = 5.0 * N * std::log2(static_cast<double>(N));

  size_t bytes = sizeof(cufftComplex) * N;

  // Host data.
  std::vector<cufftComplex> h_data(N);
  for (int i = 0; i < N; ++i) {
    h_data[i].x = static_cast<float>(rand()) / RAND_MAX;
    h_data[i].y = 0.0f;
  }

  // Device data.
  cufftComplex *d_data;
  CUDA_CHECK(cudaMalloc(&d_data, bytes));
  CUDA_CHECK(cudaMemcpy(d_data, h_data.data(), bytes, cudaMemcpyHostToDevice));

  // Plan.
  cufftHandle plan;
  CUFFT_CHECK(cufftPlan1d(&plan, N, CUFFT_C2C, 1));

  // Benchmark forward FFT.
  auto fft_fn = [&]() { CUFFT_CHECK(cufftExecC2C(plan, d_data, d_data, CUFFT_FORWARD)); };

  BenchResult res = run_benchmark(fft_fn);

  std::printf("RESULT kernel=fft_cufft mean_ms=%.4f stddev_ms=%.4f "
              "min_ms=%.4f total_ops=%.0f N=%d\n",
              res.mean_ms, res.stddev_ms, res.min_ms, total_ops, N);

  // Cleanup.
  CUFFT_CHECK(cufftDestroy(plan));
  CUDA_CHECK(cudaFree(d_data));

  return 0;
}
