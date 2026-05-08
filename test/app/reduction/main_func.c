/* reduction: integer sum reduction acc = sum_i a[i].
 * Function variant: kernel implemented as a separate function. */

#include <stdio.h>

#define N 128

__attribute__((noinline))
static int reduce_sum(const int *a, int n) {
    int acc = 0;
    for (int i = 0; i < n; ++i) {
        acc += a[i];
    }
    return acc;
}

int main(void) {
    int a[N];

    for (int i = 0; i < N; ++i) {
        a[i] = i;
    }

    int acc = reduce_sum(a, N);
    printf("%d\n", acc);
    return 0;
}
