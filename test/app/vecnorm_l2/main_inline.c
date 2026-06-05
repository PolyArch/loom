/* vecnorm_l2: signed integer L2 norm squared.
 * Inline variant: kernel loop lives directly in main. */

#include <stdio.h>

#define N 64
#define EXPECTED_NORM_SQ 619u

int main(void) {
    int input[N];

    for (int i = 0; i < N; ++i) {
        input[i] = (i % 11) - 5;
    }

    unsigned result = 0;
    for (unsigned i = 0; i < N; ++i) {
        int value = input[i];
        result += (unsigned)(value * value);
    }

    printf("vecnorm_l2 norm_sq: %u\n", result);
    if (result != EXPECTED_NORM_SQ) {
        printf("FAILED\n");
        return 1;
    }
    printf("PASSED\n");
    return 0;
}
