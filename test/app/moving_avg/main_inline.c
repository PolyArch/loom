/* moving_avg: sliding-window average with prefix windows.
 * Inline variant: kernel loop written directly in main. */

#include <math.h>
#include <stdio.h>

#define N 16
#define WINDOW 5

static float checksum(const float *values, unsigned n) {
    float sum = 0.0f;
    for (unsigned i = 0; i < n; ++i) {
        sum += values[i];
    }
    return sum;
}

int main(void) {
    float input[N];
    float output[N];

    for (unsigned i = 0; i < N; ++i) {
        input[i] = (float)(i % 10);
    }

    for (unsigned i = 0; i < N; ++i) {
        unsigned start = (i + 1 >= WINDOW) ? (i + 1 - WINDOW) : 0;
        unsigned actual_window = i - start + 1;
        float sum = 0.0f;
        for (unsigned j = start; j <= i; ++j) {
            sum += input[j];
        }
        output[i] = sum / (float)actual_window;
    }

    float result = checksum(output, N);
    if (fabsf(result - 53.0f) > 1e-5f) {
        puts("FAILED");
        return 1;
    }

    printf("moving_avg checksum: %.6f\n", result);
    puts("PASSED");
    return 0;
}
