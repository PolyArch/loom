#include <cstdio>

#include "string_compare.h"

int main() {
    const uint32_t N = 128;

    // Test case 1: Equal strings
    uint32_t str_a_1[N];
    uint32_t str_b_1[N];
    for (uint32_t i = 0; i < N; i++) {
        str_a_1[i] = 'a' + (i % 26);
        str_b_1[i] = 'a' + (i % 26);
    }

    uint32_t result_cpu_1, result_dsa_1;
    string_compare_cpu(str_a_1, str_b_1, &result_cpu_1, N);
    string_compare_dsa(str_a_1, str_b_1, &result_dsa_1, N);

    if (result_cpu_1 != result_dsa_1 || result_cpu_1 != 0) {
        printf("string_compare: FAILED\n");
        return 1;
    }

    // Test case 2: str_a < str_b
    uint32_t str_a_2[N];
    uint32_t str_b_2[N];
    for (uint32_t i = 0; i < N; i++) {
        str_a_2[i] = 'a';
        str_b_2[i] = 'b';
    }

    uint32_t result_cpu_2, result_dsa_2;
    string_compare_cpu(str_a_2, str_b_2, &result_cpu_2, N);
    string_compare_dsa(str_a_2, str_b_2, &result_dsa_2, N);

    if (result_cpu_2 != result_dsa_2 || result_cpu_2 != 0xFFFFFFFF) {
        printf("string_compare: FAILED\n");
        return 1;
    }

    // Test case 3: str_a > str_b
    uint32_t str_a_3[N];
    uint32_t str_b_3[N];
    for (uint32_t i = 0; i < N; i++) {
        str_a_3[i] = 'z';
        str_b_3[i] = 'a';
    }

    uint32_t result_cpu_3, result_dsa_3;
    string_compare_cpu(str_a_3, str_b_3, &result_cpu_3, N);
    string_compare_dsa(str_a_3, str_b_3, &result_dsa_3, N);

    if (result_cpu_3 != result_dsa_3 || result_cpu_3 != 1) {
        printf("string_compare: FAILED\n");
        return 1;
    }

    printf("string_compare: PASSED\n");
    return 0;
}

