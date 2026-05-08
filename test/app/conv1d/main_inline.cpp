// conv1d: 1-D convolution out[i] = sum_k input[i+k] * kernel[k]
// for fixed input length N and kernel length K. Output length is N - K + 1.
// Inline variant: kernel inlined directly in main.

#include <cstdio>

int main() {
    constexpr int N = 64;
    constexpr int K = 5;
    constexpr int OUT_LEN = N - K + 1;

    float input[N];
    float kernel[K];
    float out[OUT_LEN];

    for (int i = 0; i < N; ++i) {
        input[i] = 1.0f;
    }
    for (int i = 0; i < K; ++i) {
        kernel[i] = 1.0f;
    }

    for (int i = 0; i < OUT_LEN; ++i) {
        float acc = 0.0f;
        for (int j = 0; j < K; ++j) {
            acc += input[i + j] * kernel[j];
        }
        out[i] = acc;
    }

    float s = 0.0f;
    for (int i = 0; i < OUT_LEN; ++i) {
        s += out[i];
    }

    std::printf("%.6f\n", s);
    return 0;
}
