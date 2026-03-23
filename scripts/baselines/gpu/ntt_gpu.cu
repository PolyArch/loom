// NTT (Number Theoretic Transform) baseline over M31 field (p = 2^31 - 1).
// Custom CUDA kernel implementing butterfly operations in the M31 prime field.

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

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

// -----------------------------------------------------------------------
// M31 field arithmetic
// -----------------------------------------------------------------------

static constexpr uint32_t M31_P = (1U << 31) - 1;

// Primitive root of unity for M31. Using g=3 as a generator.
static constexpr uint32_t M31_G = 3;

__device__ __forceinline__ uint32_t m31_add(uint32_t a, uint32_t b) {
  uint32_t s = a + b;
  // Reduce: if s >= P, subtract P.
  return s >= M31_P ? s - M31_P : s;
}

__device__ __forceinline__ uint32_t m31_sub(uint32_t a, uint32_t b) {
  return a >= b ? a - b : a + M31_P - b;
}

__device__ __forceinline__ uint32_t m31_mul(uint32_t a, uint32_t b) {
  uint64_t prod = static_cast<uint64_t>(a) * b;
  // Barrett-style reduction for M31.
  uint32_t lo = static_cast<uint32_t>(prod & M31_P);
  uint32_t hi = static_cast<uint32_t>(prod >> 31);
  uint32_t s = lo + hi;
  return s >= M31_P ? s - M31_P : s;
}

// Modular exponentiation on host for twiddle factor precomputation.
static uint32_t m31_pow_host(uint32_t base, uint32_t exp) {
  uint64_t result = 1;
  uint64_t b = base;
  while (exp > 0) {
    if (exp & 1) {
      result = (result * b) % M31_P;
    }
    b = (b * b) % M31_P;
    exp >>= 1;
  }
  return static_cast<uint32_t>(result);
}

// -----------------------------------------------------------------------
// NTT butterfly kernel
// -----------------------------------------------------------------------

__global__ void ntt_butterfly(uint32_t *data, const uint32_t *twiddles,
                              int half_len, int step) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= half_len)
    return;

  // Determine which butterfly pair this thread handles.
  int group = tid / (step / 2);
  int pos = tid % (step / 2);
  int idx0 = group * step + pos;
  int idx1 = idx0 + step / 2;

  uint32_t w = twiddles[pos * (half_len / (step / 2))];
  uint32_t u = data[idx0];
  uint32_t v = m31_mul(data[idx1], w);

  data[idx0] = m31_add(u, v);
  data[idx1] = m31_sub(u, v);
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
  int log_n = 20; // 1M-point NTT by default.
  if (argc >= 2)
    log_n = std::atoi(argv[1]);

  int N = 1 << log_n;
  int half = N / 2;

  // Total ops estimate: N * log2(N) butterfly operations, each ~3 M31 ops.
  double total_ops = 3.0 * N * log_n;

  // Precompute twiddle factors on host.
  // omega = g^((P-1)/N) mod P
  uint32_t omega = m31_pow_host(M31_G, (M31_P - 1) / N);
  std::vector<uint32_t> twiddles(half);
  twiddles[0] = 1;
  for (int i = 1; i < half; ++i) {
    twiddles[i] = static_cast<uint32_t>(
        (static_cast<uint64_t>(twiddles[i - 1]) * omega) % M31_P);
  }

  // Random input data in [0, P).
  std::vector<uint32_t> h_data(N);
  for (int i = 0; i < N; ++i)
    h_data[i] = static_cast<uint32_t>(rand()) % M31_P;

  // Device allocations.
  uint32_t *d_data, *d_twiddles;
  CUDA_CHECK(cudaMalloc(&d_data, N * sizeof(uint32_t)));
  CUDA_CHECK(cudaMalloc(&d_twiddles, half * sizeof(uint32_t)));
  CUDA_CHECK(cudaMemcpy(d_twiddles, twiddles.data(), half * sizeof(uint32_t),
                         cudaMemcpyHostToDevice));

  int threads_per_block = 256;
  int blocks = (half + threads_per_block - 1) / threads_per_block;

  auto ntt_fn = [&]() {
    // Copy fresh data each iteration.
    CUDA_CHECK(cudaMemcpy(d_data, h_data.data(), N * sizeof(uint32_t),
                           cudaMemcpyHostToDevice));
    // Iterative NTT: log_n stages of butterflies.
    for (int step = N; step >= 2; step >>= 1) {
      ntt_butterfly<<<blocks, threads_per_block>>>(d_data, d_twiddles, half,
                                                    step);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
  };

  BenchResult res = run_benchmark(ntt_fn);

  std::printf("RESULT kernel=ntt_m31_gpu mean_ms=%.4f stddev_ms=%.4f "
              "min_ms=%.4f total_ops=%.0f N=%d\n",
              res.mean_ms, res.stddev_ms, res.min_ms, total_ops, N);

  CUDA_CHECK(cudaFree(d_data));
  CUDA_CHECK(cudaFree(d_twiddles));

  return 0;
}
