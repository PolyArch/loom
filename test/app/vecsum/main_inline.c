/* vecsum: unsigned integer vector sum with an initial value.
 * Inline variant: kernel loop written directly in main. */

#include <stdio.h>

#define N 64
#define INIT_VALUE 100u

int main(void) {
    unsigned input[N];

    for (unsigned i = 0; i < N; ++i) {
        input[i] = i;
    }

    unsigned result = INIT_VALUE;
    for (unsigned i = 0; i < N; ++i) {
        result += input[i];
    }

    printf("%u\n", result);
    return 0;
}
