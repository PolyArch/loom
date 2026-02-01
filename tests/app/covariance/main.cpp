#include <cstdio>

#include "covariance.h"
#include <cmath>

int main() {
    const uint32_t N = 1024;

    // Allocate and initialize inputs
    float X[N];
    float Y[N];
    for (uint32_t i = 0; i < N; i++) {
        X[i] = static_cast<float>(i % 100);
        Y[i] = static_cast<float>((i * 2) % 100) + 0.5f;
    }

    // Compute expected result with CPU version
    float expect_result = covariance_cpu(X, Y, N);

    // Compute result with DSA version
    float calculated_result = covariance_dsa(X, Y, N);

    // Compare results with tolerance
    if (fabsf(expect_result - calculated_result) > 1e-3f) {
        printf("covariance: FAILED\n");
        return 1;
    }

    printf("covariance: PASSED\n");
    return 0;
}

