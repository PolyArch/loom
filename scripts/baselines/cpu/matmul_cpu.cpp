// Matrix multiply CPU baseline: naive and OpenMP-tiled implementations.
// Reports timing in the RESULT format consumed by run_cpu_baselines.py.

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
// Naive matmul (single-threaded, cache-unfriendly)
// -----------------------------------------------------------------------

static void matmul_naive(const float *A, const float *B, float *C, int M,
                         int N, int K) {
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      float sum = 0.0f;
      for (int k = 0; k < K; ++k) {
        sum += A[i * K + k] * B[k * N + j];
      }
      C[i * N + j] = sum;
    }
  }
}

// -----------------------------------------------------------------------
// Tiled matmul with OpenMP parallelism
// -----------------------------------------------------------------------

static constexpr int TILE_SIZE = 64;

static void matmul_tiled_omp(const float *A, const float *B, float *C, int M,
                             int N, int K) {
  std::memset(C, 0, sizeof(float) * M * N);

#pragma omp parallel for collapse(2) schedule(dynamic)
  for (int ii = 0; ii < M; ii += TILE_SIZE) {
    for (int jj = 0; jj < N; jj += TILE_SIZE) {
      for (int kk = 0; kk < K; kk += TILE_SIZE) {
        int i_end = (ii + TILE_SIZE < M) ? ii + TILE_SIZE : M;
        int j_end = (jj + TILE_SIZE < N) ? jj + TILE_SIZE : N;
        int k_end = (kk + TILE_SIZE < K) ? kk + TILE_SIZE : K;

        for (int i = ii; i < i_end; ++i) {
          for (int k = kk; k < k_end; ++k) {
            float a_ik = A[i * K + k];
            for (int j = jj; j < j_end; ++j) {
              C[i * N + j] += a_ik * B[k * N + j];
            }
          }
        }
      }
    }
  }
}

// -----------------------------------------------------------------------
// Main
// -----------------------------------------------------------------------

static void print_result(const char *name, int M, int N, int K,
                         const baselines::Timer &timer) {
  double total_ops = 2.0 * M * N * K;
  std::printf("RESULT kernel=%s mean_ms=%.4f stddev_ms=%.4f min_ms=%.4f "
              "total_ops=%.0f M=%d N=%d K=%d\n",
              name, timer.mean_ms(), timer.stddev_ms(), timer.min_ms(),
              total_ops, M, N, K);
}

int main(int argc, char **argv) {
  int M = 512, N = 512, K = 512;
  bool run_naive = true;
  if (argc >= 4) {
    M = std::atoi(argv[1]);
    N = std::atoi(argv[2]);
    K = std::atoi(argv[3]);
  }
  if (argc >= 5 && std::string(argv[4]) == "--skip-naive") {
    run_naive = false;
  }

  std::vector<float> A(M * K), B(K * N), C(M * N);
  for (size_t i = 0; i < A.size(); ++i)
    A[i] = static_cast<float>(rand()) / RAND_MAX;
  for (size_t i = 0; i < B.size(); ++i)
    B[i] = static_cast<float>(rand()) / RAND_MAX;

  // Naive benchmark (only for small sizes to avoid excessive runtime).
  if (run_naive && M <= 1024) {
    auto naive_timer = baselines::benchmark(
        [&]() { matmul_naive(A.data(), B.data(), C.data(), M, N, K); });
    print_result("matmul_naive_cpu", M, N, K, naive_timer);
  }

  // Tiled + OpenMP benchmark.
  auto tiled_timer = baselines::benchmark(
      [&]() { matmul_tiled_omp(A.data(), B.data(), C.data(), M, N, K); });
  print_result("matmul_tiled_omp", M, N, K, tiled_timer);

  return 0;
}
