/* matvec: unsigned matrix-vector multiplication.
 * Inline variant: kernel loop written directly in main. */

#include <stdio.h>

#define M 4
#define N 5

static unsigned checksum(const unsigned *values, unsigned count) {
    unsigned sum = 0;
    for (unsigned i = 0; i < count; ++i) {
        sum += values[i];
    }
    return sum;
}

int main(void) {
    unsigned matrix[M * N];
    unsigned vector[N];
    unsigned output[M];

    for (unsigned i = 0; i < M * N; ++i) {
        matrix[i] = (i % 10) + 1;
    }
    for (unsigned i = 0; i < N; ++i) {
        vector[i] = i + 1;
    }

    for (unsigned i = 0; i < M; ++i) {
        unsigned sum = 0;
        for (unsigned j = 0; j < N; ++j) {
            sum += matrix[i * N + j] * vector[j];
        }
        output[i] = sum;
    }

    unsigned result = checksum(output, M);
    if (result != 370) {
        puts("FAILED");
        return 1;
    }

    printf("matvec checksum: %u\n", result);
    puts("PASSED");
    return 0;
}
