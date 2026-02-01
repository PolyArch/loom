#include <cstdio>

#include "kmp_table.h"

int main() {
    const uint32_t M = 16;
    
    // Pattern: "ABABCABABA"
    uint32_t pattern[M] = {'A', 'B', 'A', 'B', 'C', 'A', 'B', 'A', 'B', 'A',
                           'A', 'B', 'A', 'B', 'C', 'D'};
    
    // Output failure function tables
    uint32_t expect_table[M];
    uint32_t calculated_table[M];
    
    // Compute expected result with CPU version
    kmp_table_cpu(pattern, expect_table, M);
    
    // Compute result with accelerator version
    kmp_table_dsa(pattern, calculated_table, M);
    
    // Compare results
    for (uint32_t i = 0; i < M; i++) {
        if (expect_table[i] != calculated_table[i]) {
            printf("kmp_table: FAILED\n");
            return 1;
        }
    }
    
    printf("kmp_table: PASSED\n");
    return 0;
}

