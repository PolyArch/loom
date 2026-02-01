// Loom kernel implementation: gemv
#include "gemv.h"
#include "loom/loom.h"
#include <cmath>
#include <cstdlib>


// Full pipeline test from C++ source: GEMV (General Matrix-Vector multiply)
// Tests complete compilation chain with nested loops and scalar-vector operations
// Test: y = 2*[[1,2],[3,4]]*[1,2] + 1*[1,1] → [11,23]






// CPU implementation of GEMV (General Matrix-Vector multiply)
// output_y = alpha * A * x + beta * input_y
// A: M x N matrix (row-major)
// x: input vector of length N
// input_y: input vector of length M (read-only)
// output_y: output vector of length M (write-only)
// alpha, beta: scalar multipliers
void gemv_cpu(const uint32_t alpha, 
              const uint32_t* __restrict__ A, 
              const uint32_t* __restrict__ x, 
              const uint32_t beta, 
              const uint32_t* __restrict__ input_y,
              uint32_t* __restrict__ output_y, 
              const uint32_t M, 
              const uint32_t N) {
    for (uint32_t i = 0; i < M; i++) {
        uint32_t sum = 0;
        for (uint32_t j = 0; j < N; j++) {
            sum += A[i * N + j] * x[j];
        }
        output_y[i] = alpha * sum + beta * input_y[i];
    }
}

// GEMV: output_y[i] = alpha * sum_j(A[i,j] * x[j]) + beta * input_y[i]
// Accelerator implementation of GEMV
LOOM_ACCEL()
void gemv_dsa(const uint32_t alpha, 
              LOOM_MEMORY_BANK(4, block) LOOM_STREAM const uint32_t* __restrict__ A, 
              LOOM_STREAM const uint32_t* __restrict__ x, 
              const uint32_t beta, 
              const uint32_t* __restrict__ input_y,
              uint32_t* __restrict__ output_y, 
              const uint32_t M, 
              const uint32_t N) {
    for (uint32_t i = 0; i < M; i++) {
        uint32_t sum = 0;
        for (uint32_t j = 0; j < N; j++) {
            sum += A[i * N + j] * x[j];
        }
        output_y[i] = alpha * sum + beta * input_y[i];
    }
}



// Step 1: Read A matrix using 2D linear read (M rows, each of length N, with stride=N)
// This produces M logical dataflows (one per row), each of length N
// Physical dataflow has M LAST markers separating the rows

// Step 2: Read x vector using 2D linear read with stride=0 (repeat M times)
// This produces M copies of the x vector
// Physical dataflow has M LAST markers, same positions as A_rows

// Step 3: Element-wise multiply: A[i,j] * x[j] for all i,j
// Produces M logical dataflows (one per row), each of length N

// Step 4: Reduce each row to compute row sums: sum_j(A[i,j] * x[j]) for each i
// LAST markers trigger reduction output and state reset for each row
// Input: M logical dataflows of length N (physical length M*N)
// Output: M scalars as a single dataflow of length M

// Step 5: Multiply sums by alpha: alpha * sum_i

// Step 6: Read input_y vector linearly (length M)

// Step 7: Multiply input_y by beta: beta * input_y[i]

// Step 8: Add the two terms: alpha * sum_i + beta * input_y[i]

// Step 9: Write result to output_y linearly




