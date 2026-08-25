#ifndef LOOM_TEST_APPLICATIONS_LLAMA2C_KERNELS_SMOKE_H
#define LOOM_TEST_APPLICATIONS_LLAMA2C_KERNELS_SMOKE_H

#include <stddef.h>
#include <stdint.h>

#if defined(LOOM_APPLICATION_HOST_EXECUTION)
#include <stdio.h>
#endif

#if !defined(LOOM_LLAMA_INPUT_VARIANT)
#define LOOM_LLAMA_INPUT_VARIANT 0
#endif

__attribute__((noinline)) void rmsnorm(float *output, float *input,
                                       float *weight, int size);
__attribute__((noinline)) void softmax(float *values, int size);
__attribute__((noinline)) void matmul(float *output, float *input,
                                      float *weight, int columns, int rows);

static float llama2cAbs(float value) { return value < 0.0f ? -value : value; }

int main(void) {
#if LOOM_LLAMA_INPUT_VARIANT == 1
  float input[4] = {2.0f, 1.0f, 3.0f, 3.0f};
#else
  float input[4] = {1.0f, 2.0f, 3.0f, 4.0f};
#endif
  float weight[4] = {1.0f, 1.0f, 1.0f, 1.0f};
  float normalized[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  rmsnorm(normalized, input, weight, 4);
  float squareMean = 0.0f;
  for (int index = 0; index < 4; ++index)
    squareMean += normalized[index] * normalized[index];
  squareMean *= 0.25f;
  if (llama2cAbs(squareMean - 1.0f) > 2.0e-5f)
    return 1;

  float probabilities[4] = {1.0f, 2.0f, 3.0f, 4.0f};
  softmax(probabilities, 4);
  float probabilitySum = 0.0f;
  for (int index = 0; index < 4; ++index)
    probabilitySum += probabilities[index];
  if (llama2cAbs(probabilitySum - 1.0f) > 1.0e-6f ||
      !(probabilities[0] < probabilities[1] &&
        probabilities[1] < probabilities[2] &&
        probabilities[2] < probabilities[3]))
    return 2;

  float matrix[8] = {1.0f, 0.0f, 0.0f, 1.0f, 0.5f, 0.5f, 0.5f, 0.5f};
  float product[2] = {0.0f, 0.0f};
  matmul(product, input, matrix, 4, 2);
#if LOOM_LLAMA_INPUT_VARIANT == 1
  const float expectedFirst = 5.0f;
  const float expectedSecond = 4.5f;
#else
  const float expectedFirst = 5.0f;
  const float expectedSecond = 5.0f;
#endif
  if (llama2cAbs(product[0] - expectedFirst) > 1.0e-6f ||
      llama2cAbs(product[1] - expectedSecond) > 1.0e-6f)
    return 3;
#if defined(LOOM_APPLICATION_HOST_EXECUTION)
  printf("llama kernels variant: %d\n", LOOM_LLAMA_INPUT_VARIANT);
#endif
  return 0;
}

#define main llama2c_upstream_main

#endif
