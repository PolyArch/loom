/* delta_encode: first element plus adjacent differences.
 * Function variant: kernel implemented as a separate function. */

#include <stdint.h>
#include <stdio.h>

#define N 10
#define EXPECTED_CHECKSUM 439u

__attribute__((noinline))
static void delta_encode(const uint32_t *input, uint32_t *output, unsigned n) {
    output[0] = input[0];
    for (unsigned i = 1; i < n; ++i) {
        output[i] = input[i] - input[i - 1u];
    }
}

int main(void) {
    uint32_t input[N] = {100u, 102u, 105u, 110u, 115u, 122u, 130u, 135u, 142u, 150u};
    uint32_t output[N];

    delta_encode(input, output, N);

    uint32_t checksum = 0;
    for (unsigned i = 0; i < N; ++i) {
        checksum += (i + 1u) * output[i];
    }

    printf("delta_encode checksum: %u\n", checksum);
    if (checksum != EXPECTED_CHECKSUM) {
        printf("FAILED\n");
        return 1;
    }
    printf("PASSED\n");
    return 0;
}
