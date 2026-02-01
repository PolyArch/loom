#include <cstdio>

#include "jacobi_stencil_7pt.h"
#include <cmath>

int main() {
    const uint32_t L = 4;  // Depth
    const uint32_t M = 4;  // Rows
    const uint32_t N = 4;  // Columns

    // Input and output grids
    float expect_grid[L * M * N];
    float calculated_grid[L * M * N];

    // Initialize input grid
    // for (uint32_t k = 0; k < L; k++) {
    //     for (uint32_t i = 0; i < M; i++) {
    //         for (uint32_t j = 0; j < N; j++) {
    //             input_grid[k * M * N + i * N + j] = (float)(k + i + j);
    //         }
    //     }
    // }
    float input_grid[64] = {
        3.929384f, -4.277213f,-5.462971f,1.026295f,
        4.389380f,-1.537871f,9.615284f,3.696595f,
        -3.813620e-01,-2.157650f,-3.136440f,4.580994f,
        -1.228555f,-8.806442f,-2.039115f,4.759908f,
        -6.350165f,-6.490965f,6.310275e-01,6.365517e-01,
        2.688019f,6.988636f,4.489107f,2.220470f,
        4.448868f,-3.540822f,-2.764227f,-5.434735f,
        -4.125719f,2.619523f,-8.157901f,-1.325976f,
        -1.382745f,-1.262980e-01,-1.483394f,-3.754776f,
        -1.472974f,7.867783f,8.883201f,3.673352e-02,
        2.479059f,-7.687632f,-3.654290f,-1.703476f,
        7.326183f,-4.990893f,-3.393147e-01,9.711196f,
        3.897024e-01,2.257890f,-7.587427f,6.526816f,
        2.061203f,9.013602e-01,-3.144723f,-3.917584f,
        -1.659556f,3.626015f,7.509137f,2.084468e-01,
        3.386276f,1.718731f,2.498070f,3.493781f
    };

    // Compute expected result with CPU version
    jacobi_stencil_7pt_cpu(input_grid, expect_grid, L, M, N);

    // Compute result with accelerator version
    jacobi_stencil_7pt_dsa(input_grid, calculated_grid, L, M, N);

    // Compare results with tolerance
    for (uint32_t i = 0; i < L * M * N; i++) {
        if (fabsf(expect_grid[i] - calculated_grid[i]) > 1e-5f) {
            printf("jacobi_stencil_7pt: FAILED\n");
            return 1;
        }
    }

    printf("jacobi_stencil_7pt: PASSED\n");
    return 0;
}

