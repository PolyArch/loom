// Loom kernel implementation: sort_quick
#include "sort_quick.h"
#include "loom/loom.h"
#include <cmath>
#include <cstdlib>


// Full pipeline test from C++ source: Quick sort (iterative)
// Tests complete compilation chain with explicit stack and partitioning
// Test: [3.0, 1.0, 4.0, 2.0] → [1.0, 2.0, 3.0, 4.0]






// CPU implementation of quick sort
void sort_quick_cpu(const float* __restrict__ input,
                    float* __restrict__ output,
                    const uint32_t N) {
    // Copy input to output
    for (uint32_t i = 0; i < N; i++) {
        output[i] = input[i];
    }
    
    // Quick sort implementation using iterative approach with explicit stack
    if (N <= 1) return;
    
    // Stack to store partition ranges
    uint32_t stack[64];  // Support up to 2^64 elements (more than enough)
    int32_t top = -1;
    
    // Push initial range
    stack[++top] = 0;
    stack[++top] = N - 1;
    
    while (top >= 0) {
        // Pop range
        uint32_t high = stack[top--];
        uint32_t low = stack[top--];
        
        if (low >= high) continue;
        
        // Partition
        float pivot = output[high];
        uint32_t i = low;
        
        for (uint32_t j = low; j < high; j++) {
            if (output[j] <= pivot) {
                float temp = output[i];
                output[i] = output[j];
                output[j] = temp;
                i++;
            }
        }
        
        float temp = output[i];
        output[i] = output[high];
        output[high] = temp;
        
        uint32_t pivot_idx = i;
        
        // Push left partition
        if (pivot_idx > low) {
            stack[++top] = low;
            stack[++top] = pivot_idx - 1;
        }
        
        // Push right partition
        if (pivot_idx < high) {
            stack[++top] = pivot_idx + 1;
            stack[++top] = high;
        }
    }
}

// Accelerator implementation of quick sort
LOOM_ACCEL()
void sort_quick_dsa(const float* __restrict__ input,
                    float* __restrict__ output,
                    const uint32_t N) {
    // Copy input to output
    LOOM_NO_PARALLEL
    LOOM_NO_UNROLL
    for (uint32_t i = 0; i < N; i++) {
        output[i] = input[i];
    }
    
    // Quick sort implementation using iterative approach with explicit stack
    if (N <= 1) return;
    
    // Stack to store partition ranges
    uint32_t stack[64];  // Support up to 2^64 elements (more than enough)
    int32_t top = -1;
    
    // Push initial range
    stack[++top] = 0;
    stack[++top] = N - 1;
    
    while (top >= 0) {
        // Pop range
        uint32_t high = stack[top--];
        uint32_t low = stack[top--];
        
        if (low >= high) continue;
        
        // Partition
        float pivot = output[high];
        uint32_t i = low;
        
        for (uint32_t j = low; j < high; j++) {
            if (output[j] <= pivot) {
                float temp = output[i];
                output[i] = output[j];
                output[j] = temp;
                i++;
            }
        }
        
        float temp = output[i];
        output[i] = output[high];
        output[high] = temp;
        
        uint32_t pivot_idx = i;
        
        // Push left partition
        if (pivot_idx > low) {
            stack[++top] = low;
            stack[++top] = pivot_idx - 1;
        }
        
        // Push right partition
        if (pivot_idx < high) {
            stack[++top] = pivot_idx + 1;
            stack[++top] = high;
        }
    }
}



