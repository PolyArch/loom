#include <cstdio>

#include "crc32.h"

int main() {
    const uint32_t N = 256;

    // Input data
    uint32_t input[N];
    for (uint32_t i = 0; i < N; i++) {
        input[i] = i * 0x12345678;
    }

    // Output checksums
    uint32_t expect_checksum;
    uint32_t calculated_checksum;

    // Compute expected result with CPU version
    crc32_cpu(input, &expect_checksum, N);

    // Compute result with accelerator version
    crc32_dsa(input, &calculated_checksum, N);

    // Compare results
    if (expect_checksum != calculated_checksum) {
        printf("crc32: FAILED\n");
        return 1;
    }

    printf("crc32: PASSED\n");
    return 0;
}

