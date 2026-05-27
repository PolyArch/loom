// gemm: small dense matrix multiply C = A * B for fixed M, N, K.
// Function variant: kernel implemented as a separate function.

#include <cstdio>

namespace {

constexpr int M = 8;
constexpr int N = 8;
constexpr int K = 8;

__attribute__((noinline))
void gemm(const float *A, const float *B, float *C, int m, int n, int k) {
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            float acc = 0.0f;
            for (int p = 0; p < k; ++p) {
                acc += A[i * k + p] * B[p * n + j];
            }
            C[i * n + j] = acc;
        }
    }
}

float checksum(const float *X, int count) {
    float acc = 0.0f;
    for (int i = 0; i < count; ++i) {
        acc += X[i];
    }
    return acc;
}

} // namespace

int main() {
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

    gemm(A, B, C, M, N, K);

    float s = checksum(C, M * N);
    std::printf("%.6f\n", s);
    return 0;
}
