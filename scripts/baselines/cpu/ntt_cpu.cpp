// NTT (Number Theoretic Transform) CPU baseline over M31 field (p = 2^31 - 1).
// Single-threaded iterative butterfly implementation.

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "../common/timer.h"

// -----------------------------------------------------------------------
// M31 field arithmetic
// -----------------------------------------------------------------------

static constexpr uint32_t M31_P = (1U << 31) - 1;
static constexpr uint32_t M31_G = 3; // Generator.

static inline uint32_t m31_add(uint32_t a, uint32_t b) {
  uint32_t s = a + b;
  return s >= M31_P ? s - M31_P : s;
}

static inline uint32_t m31_sub(uint32_t a, uint32_t b) {
  return a >= b ? a - b : a + M31_P - b;
}

static inline uint32_t m31_mul(uint32_t a, uint32_t b) {
  uint64_t prod = static_cast<uint64_t>(a) * b;
  uint32_t lo = static_cast<uint32_t>(prod & M31_P);
  uint32_t hi = static_cast<uint32_t>(prod >> 31);
  uint32_t s = lo + hi;
  return s >= M31_P ? s - M31_P : s;
}

static uint32_t m31_pow(uint32_t base, uint32_t exp) {
  uint64_t result = 1;
  uint64_t b = base;
  while (exp > 0) {
    if (exp & 1)
      result = (result * b) % M31_P;
    b = (b * b) % M31_P;
    exp >>= 1;
  }
  return static_cast<uint32_t>(result);
}

// -----------------------------------------------------------------------
// Bit-reversal permutation
// -----------------------------------------------------------------------

static unsigned bit_reverse(unsigned x, int log_n) {
  unsigned result = 0;
  for (int i = 0; i < log_n; ++i) {
    result = (result << 1) | (x & 1);
    x >>= 1;
  }
  return result;
}

static void bit_reverse_permute(uint32_t *data, int N, int log_n) {
  for (int i = 0; i < N; ++i) {
    unsigned j = bit_reverse(static_cast<unsigned>(i), log_n);
    if (i < static_cast<int>(j)) {
      uint32_t tmp = data[i];
      data[i] = data[j];
      data[j] = tmp;
    }
  }
}

// -----------------------------------------------------------------------
// Iterative NTT (DIT butterfly)
// -----------------------------------------------------------------------

static void ntt_m31(uint32_t *data, int N, int log_n,
                    const uint32_t *twiddles) {
  bit_reverse_permute(data, N, log_n);

  for (int s = 1; s <= log_n; ++s) {
    int m = 1 << s;
    int half_m = m >> 1;
    int tw_stride = N / m;

    for (int k = 0; k < N; k += m) {
      for (int j = 0; j < half_m; ++j) {
        uint32_t w = twiddles[j * tw_stride];
        uint32_t u = data[k + j];
        uint32_t v = m31_mul(data[k + j + half_m], w);
        data[k + j] = m31_add(u, v);
        data[k + j + half_m] = m31_sub(u, v);
      }
    }
  }
}

// -----------------------------------------------------------------------
// Main
// -----------------------------------------------------------------------

int main(int argc, char **argv) {
  int log_n = 20;
  if (argc >= 2)
    log_n = std::atoi(argv[1]);

  int N = 1 << log_n;
  int half = N / 2;

  double total_ops = 3.0 * N * log_n;

  // Precompute twiddle factors.
  uint32_t omega = m31_pow(M31_G, (M31_P - 1) / N);
  std::vector<uint32_t> twiddles(half);
  twiddles[0] = 1;
  for (int i = 1; i < half; ++i) {
    twiddles[i] = m31_mul(twiddles[i - 1], omega);
  }

  // Random input.
  std::vector<uint32_t> data(N);
  std::vector<uint32_t> data_copy(N);
  for (int i = 0; i < N; ++i)
    data[i] = static_cast<uint32_t>(rand()) % M31_P;

  auto ntt_fn = [&]() {
    std::copy(data.begin(), data.end(), data_copy.begin());
    ntt_m31(data_copy.data(), N, log_n, twiddles.data());
  };

  auto timer = baselines::benchmark(ntt_fn);

  std::printf("RESULT kernel=ntt_m31_cpu mean_ms=%.4f stddev_ms=%.4f "
              "min_ms=%.4f total_ops=%.0f N=%d\n",
              timer.mean_ms(), timer.stddev_ms(), timer.min_ms(), total_ops, N);

  return 0;
}
