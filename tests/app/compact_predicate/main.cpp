#include <cstdio>

#include "compact_predicate.h"

int main() {
    const uint32_t N = 1024;

    // Input array
    uint32_t input[N];

    // Predicate array
    uint32_t predicate[N];

    // Output arrays (large enough to hold all elements)
    uint32_t expect_output[N];
    uint32_t calculated_output[N];

    // Initialize input and predicate
    for (uint32_t i = 0; i < N; i++) {
        input[i] = i;
        predicate[i] = (i % 3 != 0) ? 1 : 0;
    }

    // Test predicated compaction
    uint32_t expect_count = compact_predicate_cpu(input, predicate, expect_output, N);
    uint32_t calculated_count = compact_predicate_dsa(input, predicate, calculated_output, N);

    if (expect_count != calculated_count) {
        printf("compact_predicate: FAILED\n");
        return 1;
    }

    for (uint32_t i = 0; i < expect_count; i++) {
        if (expect_output[i] != calculated_output[i]) {
            printf("compact_predicate: FAILED\n");
            return 1;
        }
    }

    printf("compact_predicate: PASSED\n");
    return 0;
}

