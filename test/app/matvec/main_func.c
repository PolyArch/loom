/* matvec: unsigned matrix-vector multiplication.
 * Function variant: kernel implemented as a separate function. */

#include <stdio.h>

#define M 4
#define N 5

__attribute__((noinline))
static void matvec_kernel(const unsigned *matrix, const unsigned *vector,
                          unsigned *output, unsigned rows, unsigned cols) {
    for (unsigned i = 0; i < rows; ++i) {
        unsigned sum = 0;
        for (unsigned j = 0; j < cols; ++j) {
            sum += matrix[i * cols + j] * vector[j];
        }
        output[i] = sum;
    }
}

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

    matvec_kernel(matrix, vector, output, M, N);
    unsigned result = checksum(output, M);
    if (result != 370) {
        puts("FAILED");
        return 1;
    }

    printf("matvec checksum: %u\n", result);
    puts("PASSED");
    return 0;
}
