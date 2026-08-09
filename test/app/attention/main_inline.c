#include <math.h>
#include <stdint.h>
#include <stdio.h>

enum { TOKENS = 3, WIDTH = 2 };

int main(void) {
  const float q[TOKENS][WIDTH] = {{1.0f, 1.0f}, {2.0f, 2.0f}, {-1.0f, -1.0f}};
  const float k[TOKENS][WIDTH] = {{1.0f, 0.0f}, {0.0f, 1.0f}, {2.0f, -1.0f}};
  const float v[TOKENS][WIDTH] = {{1.0f, 2.0f}, {3.0f, 4.0f}, {5.0f, 6.0f}};
  float scores[TOKENS][TOKENS];
  float probabilities[TOKENS][TOKENS];
  float output[TOKENS][WIDTH];
  float checksum = 0.0f;

  for (uint32_t row = 0; row < TOKENS; ++row)
    for (uint32_t column = 0; column < TOKENS; ++column) {
      float dot = 0.0f;
      for (uint32_t lane = 0; lane < WIDTH; ++lane)
        dot += q[row][lane] * k[column][lane];
      scores[row][column] = dot * 0.5f;
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
  for (uint32_t row = 0; row < TOKENS; ++row)
    for (uint32_t lane = 0; lane < WIDTH; ++lane) {
      float sum = 0.0f;
      for (uint32_t column = 0; column < TOKENS; ++column)
        sum += probabilities[row][column] * v[column][lane];
      output[row][lane] = sum;
      checksum += (float)(row * WIDTH + lane + 1u) * output[row][lane];
    }

  if (checksum < 74.99999f || checksum > 75.00001f) {
    printf("FAILED\n");
    return 1;
  }
  printf("attention checksum: %.6f\n", checksum);
  printf("PASSED\n");
  return 0;
}
