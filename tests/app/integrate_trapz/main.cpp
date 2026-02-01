#include <cstdio>

#include "integrate_trapz.h"
#include <cmath>

int main() {
    const uint32_t N = 128;
    
    // Input coordinates
    float input_x[N];
    float input_y[N];
    
    // Initialize inputs (integrate y = x from 0 to 1, should give 0.5)
    for (uint32_t i = 0; i < N; i++) {
        input_x[i] = (float)i / (float)(N - 1);
        input_y[i] = input_x[i];  // y = x
    }
    
    // Compute expected result with CPU version
    float expect_integral = integrate_trapz_cpu(input_x, input_y, N);
    
    // Compute result with accelerator version
    float calculated_integral = integrate_trapz_dsa(input_x, input_y, N);
    
    // Compare results with tolerance
    if (fabsf(expect_integral - calculated_integral) > 1e-5f) {
        printf("integrate_trapz: FAILED\n");
        return 1;
    }
    
    printf("integrate_trapz: PASSED\n");
    return 0;
}

