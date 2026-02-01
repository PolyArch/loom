// Loom app test driver: vecsum-while
#include "vecsum-while.h"
#include <cstdio>
#include <cmath>
#include <cstdlib>

int main() {
    const uint32_t N = 16;
    uint32_t input[N];
    for (uint32_t i = 0; i < N; ++i) {
        input[i] = i;
    }
    uint32_t cpu_result = vecsum_cpu(input, 0, N);
    uint32_t dsa_result = vecsum_dsa(input, 0, N);
    if (cpu_result != dsa_result) {
        printf("vecsum-while: FAILED\n");
        return 1;
    }
    printf("vecsum-while: PASSED\n");
    return 0;
}
