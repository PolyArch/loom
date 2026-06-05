/* vecnorm_l1: signed integer L1 norm.
 * Function variant: kernel implemented as a separate function. */

#include <stdio.h>

#define N 64
#define EXPECTED_NORM 171u

__attribute__((noinline))
static unsigned vecnorm_l1(const int *input, unsigned n) {
    unsigned norm = 0;
    for (unsigned i = 0; i < n; ++i) {
        int value = input[i];
        norm += (unsigned)(value < 0 ? -value : value);
    }
    return norm;
}

int main(void) {
    int input[N];

    for (int i = 0; i < N; ++i) {
        input[i] = (i % 11) - 5;
    }

    unsigned result = vecnorm_l1(input, N);
    printf("vecnorm_l1 norm: %u\n", result);
    if (result != EXPECTED_NORM) {
        printf("FAILED\n");
        return 1;
    }
    printf("PASSED\n");
    return 0;
}
