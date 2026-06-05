/* delta_encode: first element plus adjacent differences.
 * Inline variant: kernel body inlined into main. */

#include <stdint.h>
#include <stdio.h>

#define N 10
#define EXPECTED_CHECKSUM 439u

int main(void) {
    uint32_t input[N] = {100u, 102u, 105u, 110u, 115u, 122u, 130u, 135u, 142u, 150u};
    uint32_t output[N];

    output[0] = input[0];
    for (unsigned i = 1; i < N; ++i) {
        output[i] = input[i] - input[i - 1u];
    }

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
