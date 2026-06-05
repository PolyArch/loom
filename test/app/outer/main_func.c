/* outer: unsigned outer product C[i,j] = a[i] * b[j].
 * Function variant: kernel implemented as a separate function. */

#include <stdio.h>

#define M 3
#define N 4

__attribute__((noinline))
static void outer_kernel(const unsigned *lhs, const unsigned *rhs,
                         unsigned *out, unsigned rows, unsigned cols) {
    for (unsigned i = 0; i < rows; ++i) {
        for (unsigned j = 0; j < cols; ++j) {
            out[i * cols + j] = lhs[i] * rhs[j];
        }
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
    unsigned lhs[M];
    unsigned rhs[N];
    unsigned out[M * N];

    for (unsigned i = 0; i < M; ++i) {
        lhs[i] = i + 1;
    }
    for (unsigned i = 0; i < N; ++i) {
        rhs[i] = 2 * i + 1;
    }

    outer_kernel(lhs, rhs, out, M, N);
    unsigned result = checksum(out, M * N);
    if (result != 96) {
        puts("FAILED");
        return 1;
    }

    printf("outer checksum: %u\n", result);
    puts("PASSED");
    return 0;
}
