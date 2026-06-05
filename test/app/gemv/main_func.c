/* gemv: unsigned y = alpha * A * x + beta * y0.
 * Function variant: kernel implemented as a separate function. */

#include <stdio.h>

#define M 4
#define N 5
#define ALPHA 2u
#define BETA 3u

__attribute__((noinline))
static void gemv_kernel(unsigned alpha, const unsigned *matrix,
                        const unsigned *vector, unsigned beta,
                        const unsigned *input_y, unsigned *output,
                        unsigned rows, unsigned cols) {
    for (unsigned i = 0; i < rows; ++i) {
        unsigned sum = 0;
        for (unsigned j = 0; j < cols; ++j) {
            sum += matrix[i * cols + j] * vector[j];
        }
        output[i] = alpha * sum + beta * input_y[i];
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
    unsigned input_y[M];
    unsigned output[M];

    for (unsigned i = 0; i < M * N; ++i) {
        matrix[i] = (i % 10) + 1;
    }
    for (unsigned i = 0; i < N; ++i) {
        vector[i] = i + 1;
    }
    for (unsigned i = 0; i < M; ++i) {
        input_y[i] = i % 5;
    }

    gemv_kernel(ALPHA, matrix, vector, BETA, input_y, output, M, N);
    unsigned result = checksum(output, M);
    if (result != 758) {
        puts("FAILED");
        return 1;
    }

    printf("gemv checksum: %u\n", result);
    puts("PASSED");
    return 0;
}
