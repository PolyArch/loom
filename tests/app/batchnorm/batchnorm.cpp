// Loom kernel implementation: batchnorm
#include "batchnorm.h"
#include "loom/loom.h"
#include <cmath>
#include <cstdlib>


// Full pipeline test from C++ source: Batch normalization
// Tests complete compilation chain with nested loops and transcendental functions (sqrt)
// Test: input=[1,2,3,4,5,6,7,8], mean=[2.5,6.5], var=[1.25,1.25], C=2, H=2, W=2, eps=0.001







// CPU implementation of batch normalization
// output[i] = gamma * (input[i] - mean) / sqrt(variance + epsilon) + beta
// Applied per channel across spatial dimensions
// Input: C x H x W image (channel-major)
// Output: C x H x W normalized image
// mean, variance, gamma, beta: C-length arrays (per channel)
void batchnorm_cpu(const float* __restrict__ input,
                   const float* __restrict__ mean,
                   const float* __restrict__ variance,
                   const float* __restrict__ gamma,
                   const float* __restrict__ beta,
                   float* __restrict__ output,
                   const uint32_t C,
                   const uint32_t H,
                   const uint32_t W,
                   const float epsilon) {
    for (uint32_t c = 0; c < C; c++) {
        float inv_std = 1.0f / sqrtf(variance[c] + epsilon);
        
        for (uint32_t h = 0; h < H; h++) {
            for (uint32_t w = 0; w < W; w++) {
                uint32_t idx = c * (H * W) + h * W + w;
                float normalized = (input[idx] - mean[c]) * inv_std;
                output[idx] = gamma[c] * normalized + beta[c];
            }
        }
    }
}

// Accelerator implementation of batch normalization
LOOM_ACCEL()
void batchnorm_dsa(const float* __restrict__ input,
                   const float* __restrict__ mean,
                   const float* __restrict__ variance,
                   const float* __restrict__ gamma,
                   const float* __restrict__ beta,
                   float* __restrict__ output,
                   const uint32_t C,
                   const uint32_t H,
                   const uint32_t W,
                   const float epsilon) {
    LOOM_PARALLEL()
    LOOM_UNROLL()
    for (uint32_t c = 0; c < C; c++) {
        float inv_std = 1.0f / sqrtf(variance[c] + epsilon);
        
        for (uint32_t h = 0; h < H; h++) {
            for (uint32_t w = 0; w < W; w++) {
                uint32_t idx = c * (H * W) + h * W + w;
                float normalized = (input[idx] - mean[c]) * inv_std;
                output[idx] = gamma[c] * normalized + beta[c];
            }
        }
    }
}



// Batch Normalization: output = gamma * (input - mean) / sqrt(variance + epsilon) + beta
// Applied per channel across spatial dimensions (C channels, H×W pixels each)
// Despite 3-layer nested loops, this is fundamentally a 1D operation on C×H×W elements
// The 3D structure (C,H,W) is a logical abstraction - memory layout is linear

// Stage 1: Compute normalization factors for each channel (C elements)
// Step 1.1: Read variance array (C elements)

// Step 1.2: Add epsilon to variance for numerical stability

// Step 1.3: Compute square root (standard deviation)

// Step 1.4: Compute reciprocal (1.0 / sqrt) = inverse std deviation
// Scalar rdiv with scalar=1.0 gives reciprocal of each element

// Stage 2: Broadcast channel parameters to all pixels (C → C×H×W)
// Each channel's parameters (inv_std, mean, gamma, beta) repeat H×W times

// Step 2.1: Repeat inv_std from C to C×H×W elements
// Each inv_std[c] repeats H×W times for all pixels in that channel

// Step 2.2: Read mean (C elements) and repeat to C×H×W

// Step 2.3: Read gamma (C elements) and repeat to C×H×W

// Step 2.4: Read beta (C elements) and repeat to C×H×W

// Stage 3: Normalization computation (C×H×W element-wise operations)
// All operations are 1D element-wise on the linearized C×H×W array

// Step 3.1: Read input image (C×H×W elements in channel-major layout)

// Step 3.2: Subtract mean (input - mean)

// Step 3.3: Multiply by inverse std ((input - mean) * inv_std)
// This normalizes to zero mean and unit variance

// Step 3.4: Scale by gamma (gamma * normalized)
// Learnable scale parameter

// Step 3.5: Add beta (result + beta)
// Learnable shift parameter

// Stage 4: Write normalized output (C×H×W elements)




