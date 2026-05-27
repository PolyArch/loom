/* prefix_sum: in-place prefix sum out[i] = sum_{j <= i} in[j].
 * Inline variant: kernel inlined directly in main. */

#include <stdio.h>

#define N 32

int main(void) {
    int in[N];
    int out[N];

    for (int i = 0; i < N; ++i) {
        in[i] = 1;
    }

    int acc = 0;
    for (int i = 0; i < N; ++i) {
        acc += in[i];
        out[i] = acc;
    }

    printf("%d %d\n", out[N - 1], acc);
    return 0;
}
