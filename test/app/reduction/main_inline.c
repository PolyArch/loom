/* reduction: integer sum reduction acc = sum_i a[i].
 * Inline variant: kernel inlined directly in main. */

#include <stdio.h>

#define N 128

int main(void) {
    int a[N];

    for (int i = 0; i < N; ++i) {
        a[i] = i;
    }

    int acc = 0;
    for (int i = 0; i < N; ++i) {
        acc += a[i];
    }

    printf("%d\n", acc);
    return 0;
}
