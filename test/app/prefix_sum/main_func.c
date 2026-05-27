/* prefix_sum: in-place prefix sum out[i] = sum_{j <= i} in[j].
 * Function variant: kernel implemented as a separate function. */

#include <stdio.h>

#define N 32

__attribute__((noinline))
static int prefix_sum(const int *in, int *out, int n) {
    int acc = 0;
    for (int i = 0; i < n; ++i) {
        acc += in[i];
        out[i] = acc;
    }
    return acc;
}

int main(void) {
    int in[N];
    int out[N];

    for (int i = 0; i < N; ++i) {
        in[i] = 1;
    }

    int total = prefix_sum(in, out, N);
    printf("%d %d\n", out[N - 1], total);
    return 0;
}
