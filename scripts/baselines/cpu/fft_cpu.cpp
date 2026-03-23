// FFT CPU baseline: iterative Cooley-Tukey radix-2 DIT.
// Reports timing in the RESULT format consumed by run_cpu_baselines.py.

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <complex>
#include <vector>

#include "../common/timer.h"

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

static void bit_reverse_permute(std::complex<float> *data, int N, int log_n) {
  for (int i = 0; i < N; ++i) {
    unsigned j = bit_reverse(static_cast<unsigned>(i), log_n);
    if (i < static_cast<int>(j)) {
      std::swap(data[i], data[j]);
    }
  }
}

// -----------------------------------------------------------------------
// Iterative Cooley-Tukey FFT (radix-2 DIT)
// -----------------------------------------------------------------------

static void fft_cooley_tukey(std::complex<float> *data, int N, int log_n) {
  bit_reverse_permute(data, N, log_n);

  for (int s = 1; s <= log_n; ++s) {
    int m = 1 << s;
    int half_m = m >> 1;
    std::complex<float> wm = std::polar(1.0f,
        -2.0f * static_cast<float>(M_PI) / static_cast<float>(m));

    for (int k = 0; k < N; k += m) {
      std::complex<float> w(1.0f, 0.0f);
      for (int j = 0; j < half_m; ++j) {
        std::complex<float> t = w * data[k + j + half_m];
        std::complex<float> u = data[k + j];
        data[k + j] = u + t;
        data[k + j + half_m] = u - t;
        w *= wm;
      }
    }
  }
}

// -----------------------------------------------------------------------
// Main
// -----------------------------------------------------------------------

int main(int argc, char **argv) {
  int log_n = 20; // 1M-point FFT by default.
  if (argc >= 2)
    log_n = std::atoi(argv[1]);

  int N = 1 << log_n;
  double total_ops = 5.0 * N * log_n;

  // Input data.
  std::vector<std::complex<float>> data(N);
  std::vector<std::complex<float>> data_copy(N);
  for (int i = 0; i < N; ++i) {
    data[i] = std::complex<float>(
        static_cast<float>(rand()) / RAND_MAX, 0.0f);
  }

  auto fft_fn = [&]() {
    std::copy(data.begin(), data.end(), data_copy.begin());
    fft_cooley_tukey(data_copy.data(), N, log_n);
  };

  auto timer = baselines::benchmark(fft_fn);

  std::printf("RESULT kernel=fft_cooley_tukey mean_ms=%.4f stddev_ms=%.4f "
              "min_ms=%.4f total_ops=%.0f N=%d\n",
              timer.mean_ms(), timer.stddev_ms(), timer.min_ms(), total_ops, N);

  return 0;
}
