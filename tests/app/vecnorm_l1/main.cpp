#include <cstdio>

#include "vecnorm_l1.h"

int main() {
    const uint32_t N = 1024;

    // Input array
    uint32_t A[N];

    // Initialize input array
    for (uint32_t i = 0; i < N; i++) {
        A[i] = i % 10;
    }

    // Test L1 norm
    uint32_t expect = vecnorm_l1_cpu(A, N);
    uint32_t calculated = vecnorm_l1_dsa(A, N);

    if (expect != calculated) {
        printf("vecnorm_l1: FAILED\n");
        return 1;
    }

    printf("vecnorm_l1: PASSED\n");
    return 0;
}

