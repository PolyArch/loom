#include <algorithm>
#include <cstdio>

#include "spmv.h"

int main() {
    const uint32_t M = 64;  // Number of rows
    const uint32_t N = 64;  // Number of columns
    
    // Create a sparse matrix with approximately 20% density
    // We'll create a simple pattern: diagonal + some random non-zeros
    const uint32_t nnz = 256;  // Number of non-zeros
    
    uint32_t values[nnz];
    uint32_t col_indices[nnz];
    uint32_t row_ptr[M + 1];
    
    // Input vector (dense)
    uint32_t x[N];
    
    // Output vectors
    uint32_t expect_y[M];
    uint32_t calculated_y[M];
    
    // Initialize input vector
    for (uint32_t i = 0; i < N; i++) {
        x[i] = i % 10 + 1;
    }
    
    // Build CSR matrix (simple pattern: approximately 4 non-zeros per row)
    uint32_t nnz_count = 0;
    row_ptr[0] = 0;
    
    for (uint32_t i = 0; i < M; i++) {
        uint32_t nnz_per_row = std::min(4u, nnz - nnz_count);
        
        // Add diagonal element
        if (nnz_count < nnz && i < N) {
            values[nnz_count] = i % 10 + 1;
            col_indices[nnz_count] = i;
            nnz_count++;
        }
        
        // Add a few more elements per row
        for (uint32_t k = 1; k < nnz_per_row && nnz_count < nnz; k++) {
            uint32_t col = (i + k * 16) % N;
            values[nnz_count] = (i + k) % 10 + 1;
            col_indices[nnz_count] = col;
            nnz_count++;
        }
        
        row_ptr[i + 1] = nnz_count;
    }
    
    // Compute expected result with CPU version
    spmv_cpu(values, col_indices, row_ptr, x, expect_y, M, N);
    
    // Compute result with accelerator version
    spmv_dsa(values, col_indices, row_ptr, x, calculated_y, M, N);
    
    // Compare results
    for (uint32_t i = 0; i < M; i++) {
        if (expect_y[i] != calculated_y[i]) {
            printf("spmv: FAILED\n");
            return 1;
        }
    }
    
    printf("spmv: PASSED\n");
    return 0;
}
