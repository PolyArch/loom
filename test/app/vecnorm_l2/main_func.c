/* vecnorm_l2: signed integer L2 norm squared.
 * Function variant: kernel implemented as a separate function. */

#include <stdio.h>

#define N 64
#define EXPECTED_NORM_SQ 619u

__attribute__((noinline))
static unsigned vecnorm_l2(const int *input, unsigned n) {
    unsigned norm_sq = 0;
    for (unsigned i = 0; i < n; ++i) {
        int value = input[i];
        norm_sq += (unsigned)(value * value);
    }
    return norm_sq;
}

int main(void) {
    int input[N];

    for (int i = 0; i < N; ++i) {
        input[i] = (i % 11) - 5;
    }

    unsigned result = vecnorm_l2(input, N);
    printf("vecnorm_l2 norm_sq: %u\n", result);
    if (result != EXPECTED_NORM_SQ) {
        printf("FAILED\n");
        return 1;
    }
    printf("PASSED\n");
    return 0;
}
