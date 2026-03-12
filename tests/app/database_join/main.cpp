#include <cstdio>

#include "database_join.h"

int main() {
    // Test: A_ids=[1,2,3], B_ids=[2,3,4], A_vals=[10,20,30], B_vals=[200,300,400]
    // Expected: Matches on 2 and 3 -> output_ids=[2,3], output_a_values=[20,30], output_b_values=[200,300]
    const uint32_t size_a = 3;
    const uint32_t size_b = 3;
    const uint32_t max_output = 5; // Maximum possible matches

    int32_t a_ids[3] = {1, 2, 3};
    int32_t b_ids[3] = {2, 3, 4};
    int32_t a_values[3] = {10, 20, 30};
    int32_t b_values[3] = {200, 300, 400};

    // Output arrays
    int32_t expect_ids[5];
    int32_t expect_a_values[5];
    int32_t expect_b_values[5];
    int32_t calculated_ids[5];
    int32_t calculated_a_values[5];
    int32_t calculated_b_values[5];

    // Compute expected result with CPU version
    uint32_t expect_count = database_join_cpu(a_ids, b_ids, a_values, b_values,
                                              expect_ids, expect_a_values, expect_b_values, size_a, size_b);

    // Compute result with DSA version
    uint32_t calculated_count = database_join_dsa(a_ids, b_ids, a_values, b_values,
                                                  calculated_ids, calculated_a_values, calculated_b_values, size_a, size_b);

    // Compare counts
    if (expect_count != calculated_count) {
        printf("database_join: FAILED (count mismatch: expected %u, got %u)\n",
               expect_count, calculated_count);
        return 1;
    }

    // Compare results
    for (uint32_t i = 0; i < expect_count; i++) {
        if (expect_ids[i] != calculated_ids[i] ||
            expect_a_values[i] != calculated_a_values[i] ||
            expect_b_values[i] != calculated_b_values[i]) {
            printf("database_join: FAILED (mismatch at index %u)\n", i);
            return 1;
        }
    }

    printf("database_join: PASSED\n");
    return 0;
}
