/* vecnorm_l1: signed integer L1 norm.
 * Inline variant: kernel loop lives directly in main. */

#include <stdio.h>

#define N 64
#define EXPECTED_NORM 171u

int main(void) {
    int input[N];

    for (int i = 0; i < N; ++i) {
        input[i] = (i % 11) - 5;
    }

    unsigned result = 0;
    for (unsigned i = 0; i < N; ++i) {
        int value = input[i];
        result += (unsigned)(value < 0 ? -value : value);
    }

    printf("vecnorm_l1 norm: %u\n", result);
    if (result != EXPECTED_NORM) {
        printf("FAILED\n");
        return 1;
    }
    printf("PASSED\n");
    return 0;
}
