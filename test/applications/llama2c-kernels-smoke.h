#ifndef LOOM_TEST_APPLICATIONS_LLAMA2C_KERNELS_SMOKE_H
#define LOOM_TEST_APPLICATIONS_LLAMA2C_KERNELS_SMOKE_H

#include "../../include/Runtime/Computation.h"

#include <stddef.h>
#include <stdint.h>

#if defined(LOOM_APPLICATION_HOST_EXECUTION)
#include <stdio.h>
#endif

#if !defined(LOOM_LLAMA_INPUT_VARIANT)
#define LOOM_LLAMA_INPUT_VARIANT 0
#endif

/* Kernel extent. The smoke input keeps the smallest shape that still exercises
   every upstream kernel. A qualified input scales the same shapes until the
   measured computation dominates the System's fixed per-launch cost, so the
   saturation target it declares is reachable at all. Every expectation below
   is derived from these extents; none of them is a transcribed constant. */
#if !defined(LOOM_LLAMA_KERNEL_WIDTH)
#define LOOM_LLAMA_KERNEL_WIDTH 4
#endif
#if !defined(LOOM_LLAMA_KERNEL_ROWS)
#define LOOM_LLAMA_KERNEL_ROWS 2
#endif

enum {
  LLAMA2C_KERNEL_WIDTH = LOOM_LLAMA_KERNEL_WIDTH,
  LLAMA2C_KERNEL_ROWS = LOOM_LLAMA_KERNEL_ROWS,
};

_Static_assert(LLAMA2C_KERNEL_WIDTH >= 4 && LLAMA2C_KERNEL_WIDTH % 2 == 0,
               "the kernel width carries an even matmul selector pair");
_Static_assert(LLAMA2C_KERNEL_ROWS >= 2, "matmul needs at least two rows");

__attribute__((noinline)) void rmsnorm(float *output, float *values,
                                       float *scale, int size);
__attribute__((noinline)) void softmax(float *values, int size);
__attribute__((noinline)) void matmul(float *output, float *values,
                                      float *weights, int columns, int rows);

static float llama2cAbs(float value) { return value < 0.0f ? -value : value; }

/* A float32 reduction over the kernel width admits about one rounding of
   relative size 2^-24 per term. Both kernel invariants below compare against
   an exact value, so their bounds carry that width-proportional slack above
   the small fixed bound the smallest shape needs. */
static float llama2cAccumulationSlack(void) {
  return 1.2e-7f * (float)LLAMA2C_KERNEL_WIDTH;
}

/* The matmul weight selects one positive and one negated input element per
   row, so every row's exact product is a single float32 subtraction of two
   input elements: every other product is an exact zero and adds nothing. */
static int llama2cPositiveSelector(int row) {
  return row % LLAMA2C_KERNEL_WIDTH;
}

static int llama2cNegativeSelector(int row) {
  return (row + LLAMA2C_KERNEL_WIDTH / 2) % LLAMA2C_KERNEL_WIDTH;
}

static float llama2cValues[LLAMA2C_KERNEL_WIDTH];
static float llama2cScale[LLAMA2C_KERNEL_WIDTH];
static float llama2cNormalized[LLAMA2C_KERNEL_WIDTH];
static float llama2cProbabilities[LLAMA2C_KERNEL_WIDTH];
static float llama2cWeights[LLAMA2C_KERNEL_ROWS * LLAMA2C_KERNEL_WIDTH];
static float llama2cProduct[LLAMA2C_KERNEL_ROWS];

int main(void) {
  for (int index = 0; index < LLAMA2C_KERNEL_WIDTH; ++index) {
    llama2cValues[index] = 1.0f + (float)((index + LOOM_LLAMA_INPUT_VARIANT) % 4);
    llama2cScale[index] = 1.0f;
    llama2cNormalized[index] = 0.0f;
    /* Strictly increasing across the whole width, so the softmax result stays
       strictly increasing at every extent and never underflows. */
    llama2cProbabilities[index] =
        1.0f + (float)index * (3.0f / (float)(LLAMA2C_KERNEL_WIDTH - 1));
  }
  for (int row = 0; row < LLAMA2C_KERNEL_ROWS; ++row) {
    float *weightRow = llama2cWeights + row * LLAMA2C_KERNEL_WIDTH;
    for (int column = 0; column < LLAMA2C_KERNEL_WIDTH; ++column)
      weightRow[column] = 0.0f;
    weightRow[llama2cPositiveSelector(row)] = 1.0f;
    weightRow[llama2cNegativeSelector(row)] = -1.0f;
    llama2cProduct[row] = 0.0f;
  }

  loom_computation_begin();
  rmsnorm(llama2cNormalized, llama2cValues, llama2cScale, LLAMA2C_KERNEL_WIDTH);
  softmax(llama2cProbabilities, LLAMA2C_KERNEL_WIDTH);
  matmul(llama2cProduct, llama2cValues, llama2cWeights, LLAMA2C_KERNEL_WIDTH,
         LLAMA2C_KERNEL_ROWS);
  loom_computation_end();

  double squareSum = 0.0;
  for (int index = 0; index < LLAMA2C_KERNEL_WIDTH; ++index)
    squareSum +=
        (double)llama2cNormalized[index] * (double)llama2cNormalized[index];
  const float squareMean = (float)(squareSum / (double)LLAMA2C_KERNEL_WIDTH);
  if (llama2cAbs(squareMean - 1.0f) > 2.0e-5f + llama2cAccumulationSlack())
    return 1;

  double probabilitySum = 0.0;
  int strictlyIncreasing = 1;
  for (int index = 0; index < LLAMA2C_KERNEL_WIDTH; ++index) {
    probabilitySum += (double)llama2cProbabilities[index];
    if (index != 0 &&
        !(llama2cProbabilities[index - 1] < llama2cProbabilities[index]))
      strictlyIncreasing = 0;
  }
  if (llama2cAbs((float)probabilitySum - 1.0f) >
          1.0e-6f + llama2cAccumulationSlack() ||
      !strictlyIncreasing)
    return 2;

  for (int row = 0; row < LLAMA2C_KERNEL_ROWS; ++row) {
    const float expected = llama2cValues[llama2cPositiveSelector(row)] -
                           llama2cValues[llama2cNegativeSelector(row)];
    if (llama2cAbs(llama2cProduct[row] - expected) > 1.0e-6f)
      return 3;
  }
#if defined(LOOM_APPLICATION_HOST_EXECUTION)
  printf("llama kernels variant: %d\n", LOOM_LLAMA_INPUT_VARIANT);
#endif
  return 0;
}

#define main llama2c_upstream_main

#endif
