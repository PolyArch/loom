// gemm: small dense matrix multiply C = A * B for fixed M, N, K.
// Inline variant: kernel inlined directly in main.

#include <cstdio>

int main() {
    constexpr int M = 8;
    constexpr int N = 8;
    constexpr int K = 8;

    float A[M * K];
    float B[K * N];
    float C[M * N];

    for (int i = 0; i < M; ++i) {
        for (int p = 0; p < K; ++p) {
            A[i * K + p] = static_cast<float>(i + 1);
        }
    }
    for (int p = 0; p < K; ++p) {
        for (int j = 0; j < N; ++j) {
            B[p * N + j] = static_cast<float>(j + 1);
        }
    }

    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float acc = 0.0f;
            for (int p = 0; p < K; ++p) {
                acc += A[i * K + p] * B[p * N + j];
            }
            C[i * N + j] = acc;
        }
    }

    float s = 0.0f;
    for (int i = 0; i < M * N; ++i) {
        s += C[i];
    }

    std::printf("%.6f\n", s);
    return 0;
}
