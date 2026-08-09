#include <math.h>
#include <stdint.h>
#include <stdio.h>

enum { TOKENS = 3, WIDTH = 2 };

static float absolute(float value) { return value < 0.0f ? -value : value; }

__attribute__((noinline)) static void
attention_kernel(const float q[TOKENS][WIDTH], const float k[TOKENS][WIDTH],
                 const float v[TOKENS][WIDTH], float output[TOKENS][WIDTH]) {
  float scores[TOKENS][TOKENS];
  float probabilities[TOKENS][TOKENS];

  for (uint32_t row = 0; row < TOKENS; ++row) {
    for (uint32_t column = 0; column < TOKENS; ++column) {
      float dot = 0.0f;
      for (uint32_t lane = 0; lane < WIDTH; ++lane)
        dot += q[row][lane] * k[column][lane];
      scores[row][column] = dot * 0.5f;
    }
  }

  for (uint32_t row = 0; row < TOKENS; ++row) {
    float maximum = scores[row][0];
    for (uint32_t column = 1; column < TOKENS; ++column)
      if (scores[row][column] > maximum)
        maximum = scores[row][column];
    float denominator = 0.0f;
    for (uint32_t column = 0; column < TOKENS; ++column) {
      probabilities[row][column] = expf(scores[row][column] - maximum);
      denominator += probabilities[row][column];
    }
    for (uint32_t column = 0; column < TOKENS; ++column)
      probabilities[row][column] /= denominator;
  }

  for (uint32_t row = 0; row < TOKENS; ++row) {
    for (uint32_t lane = 0; lane < WIDTH; ++lane) {
      float sum = 0.0f;
      for (uint32_t column = 0; column < TOKENS; ++column)
        sum += probabilities[row][column] * v[column][lane];
      output[row][lane] = sum;
    }
  }
}

int main(void) {
  const float q[TOKENS][WIDTH] = {{1.0f, 1.0f}, {2.0f, 2.0f}, {-1.0f, -1.0f}};
  const float k[TOKENS][WIDTH] = {{1.0f, 0.0f}, {0.0f, 1.0f}, {2.0f, -1.0f}};
  const float v[TOKENS][WIDTH] = {{1.0f, 2.0f}, {3.0f, 4.0f}, {5.0f, 6.0f}};
  const float expected[TOKENS][WIDTH] = {
      {3.0f, 4.0f}, {3.0f, 4.0f}, {3.0f, 4.0f}};
  float output[TOKENS][WIDTH];
  float checksum = 0.0f;

  attention_kernel(q, k, v, output);
  for (uint32_t row = 0; row < TOKENS; ++row) {
    for (uint32_t lane = 0; lane < WIDTH; ++lane) {
      if (absolute(output[row][lane] - expected[row][lane]) > 1.0e-6f) {
        printf("FAILED\n");
        return 1;
      }
      checksum += (float)(row * WIDTH + lane + 1u) * output[row][lane];
    }
  }

  printf("attention checksum: %.6f\n", checksum);
  printf("PASSED\n");
  return 0;
}
