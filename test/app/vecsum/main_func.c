/* vecsum: unsigned integer vector sum with an initial value.
 * Function variant: kernel implemented as a separate function. */

#include <stdio.h>

#define N 64
#define INIT_VALUE 100u

__attribute__((noinline))
static unsigned vecsum(const unsigned *input, unsigned init_value, unsigned n) {
    unsigned sum = init_value;
    for (unsigned i = 0; i < n; ++i) {
        sum += input[i];
    }
    return sum;
}

int main(void) {
    unsigned input[N];

    for (unsigned i = 0; i < N; ++i) {
        input[i] = i;
    }

    unsigned result = vecsum(input, INIT_VALUE, N);
    printf("%u\n", result);
    return 0;
}
