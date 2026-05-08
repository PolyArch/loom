// conv1d: 1-D convolution out[i] = sum_k input[i+k] * kernel[k]
// for fixed input length N and kernel length K. Output length is N - K + 1.
// Function variant: kernel implemented as a separate function.

#include <cstdio>

namespace {

constexpr int N = 64;
constexpr int K = 5;
constexpr int OUT_LEN = N - K + 1;

__attribute__((noinline))
void conv1d(const float *input, const float *kernel, float *out,
            int n, int k) {
    int out_len = n - k + 1;
    for (int i = 0; i < out_len; ++i) {
        float acc = 0.0f;
        for (int j = 0; j < k; ++j) {
            acc += input[i + j] * kernel[j];
        }
        out[i] = acc;
    }
}

float checksum(const float *x, int count) {
    float acc = 0.0f;
    for (int i = 0; i < count; ++i) {
        acc += x[i];
    }
    return acc;
}

} // namespace

int main() {
    float input[N];
    float kernel[K];
    float out[OUT_LEN];

    for (int i = 0; i < N; ++i) {
        input[i] = 1.0f;
    }
    for (int i = 0; i < K; ++i) {
        kernel[i] = 1.0f;
    }

    conv1d(input, kernel, out, N, K);

    float s = checksum(out, OUT_LEN);
    std::printf("%.6f\n", s);
    return 0;
}
