/* outer: unsigned outer product C[i,j] = a[i] * b[j].
 * Inline variant: kernel loop written directly in main. */

#include <stdio.h>

#define M 3
#define N 4

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

    for (unsigned i = 0; i < M; ++i) {
        for (unsigned j = 0; j < N; ++j) {
            out[i * N + j] = lhs[i] * rhs[j];
        }
    }

    unsigned result = checksum(out, M * N);
    if (result != 96) {
        puts("FAILED");
        return 1;
    }

    printf("outer checksum: %u\n", result);
    puts("PASSED");
    return 0;
}
