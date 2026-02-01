#include <cstdio>

#include "wildcard_match.h"

int main() {
    const uint32_t N = 64;
    const uint32_t M = 8;

    // Test case 1: Match with wildcards
    uint32_t text_1[N];
    uint32_t pattern_1[M] = {'A', 'B', '?', 'D', 'E', '?', 'G', 'H'};

    for (uint32_t i = 0; i < N; i++) {
        text_1[i] = 'X';
    }
    // Place matching substring at position 10
    text_1[10] = 'A';
    text_1[11] = 'B';
    text_1[12] = 'C'; // Matches '?'
    text_1[13] = 'D';
    text_1[14] = 'E';
    text_1[15] = 'F'; // Matches '?'
    text_1[16] = 'G';
    text_1[17] = 'H';

    uint32_t result_cpu_1, result_dsa_1;
    wildcard_match_cpu(text_1, pattern_1, &result_cpu_1, N, M);
    wildcard_match_dsa(text_1, pattern_1, &result_dsa_1, N, M);

    if (result_cpu_1 != result_dsa_1 || result_cpu_1 != 1) {
        printf("wildcard_match: FAILED\n");
        return 1;
    }

    // Test case 2: No match
    uint32_t text_2[N];
    uint32_t pattern_2[M] = {'Z', 'Z', 'Z', 'Z', 'Z', 'Z', 'Z', 'Z'};

    for (uint32_t i = 0; i < N; i++) {
        text_2[i] = 'A';
    }

    uint32_t result_cpu_2, result_dsa_2;
    wildcard_match_cpu(text_2, pattern_2, &result_cpu_2, N, M);
    wildcard_match_dsa(text_2, pattern_2, &result_dsa_2, N, M);

    if (result_cpu_2 != result_dsa_2 || result_cpu_2 != 0) {
        printf("wildcard_match: FAILED\n");
        return 1;
    }

    // Test case 3: All wildcards
    uint32_t text_3[N];
    uint32_t pattern_3[M] = {'?', '?', '?', '?', '?', '?', '?', '?'};

    for (uint32_t i = 0; i < N; i++) {
        text_3[i] = 'A' + (i % 26);
    }

    uint32_t result_cpu_3, result_dsa_3;
    wildcard_match_cpu(text_3, pattern_3, &result_cpu_3, N, M);
    wildcard_match_dsa(text_3, pattern_3, &result_dsa_3, N, M);

    if (result_cpu_3 != result_dsa_3 || result_cpu_3 != 1) {
        printf("wildcard_match: FAILED\n");
        return 1;
    }

    printf("wildcard_match: PASSED\n");
    return 0;
}

