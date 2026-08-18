#if !defined(LOOM_ATTENTION_PRODUCT_EXECUTION)
#include <stdio.h>
#endif

enum {
  TOKEN_COUNT = 4,
  SENSOR_LANE_COUNT = 3,
  FEATURE_LANE_COUNT = 4,
  STATISTIC_COUNT = 4,
};

#if defined(__clang__) && !defined(LOOM_ATTENTION_RETAIN_LOOPS)
#define LOOM_UNROLL_FULL _Pragma("clang loop unroll(full)")
#else
#define LOOM_UNROLL_FULL
#endif

static float absolute_value(float value) {
  return value < 0.0f ? -value : value;
}

static int outside_tolerance(float actual, float expected) {
  return absolute_value(actual - expected) > 1.0e-4f;
}

__attribute__((weak)) void
project_audio(const float input[restrict TOKEN_COUNT][SENSOR_LANE_COUNT],
              float output[restrict TOKEN_COUNT][FEATURE_LANE_COUNT]) {
  LOOM_UNROLL_FULL
  for (unsigned int token = 0; token < TOKEN_COUNT; ++token) {
    const float x = input[token][0];
    const float y = input[token][1];
    const float z = input[token][2];
    output[token][0] = x;
    output[token][1] = y;
    output[token][2] = z;
    output[token][3] = x + y;
  }
}

__attribute__((weak)) void
project_imu(const float input[restrict TOKEN_COUNT][SENSOR_LANE_COUNT],
            float output[restrict TOKEN_COUNT][FEATURE_LANE_COUNT]) {
  LOOM_UNROLL_FULL
  for (unsigned int token = 0; token < TOKEN_COUNT; ++token) {
    const float x = input[token][0];
    const float y = input[token][1];
    const float z = input[token][2];
    output[token][0] = x;
    output[token][1] = y;
    output[token][2] = z;
    output[token][3] = x - y;
  }
}

__attribute__((weak)) void
fuse_attention(const float query[restrict TOKEN_COUNT][FEATURE_LANE_COUNT],
               const float key_value[restrict TOKEN_COUNT][FEATURE_LANE_COUNT],
               float output[restrict TOKEN_COUNT][FEATURE_LANE_COUNT]) {
  float query_local[TOKEN_COUNT][FEATURE_LANE_COUNT];
  float key_value_local[TOKEN_COUNT][FEATURE_LANE_COUNT];
  float scores[TOKEN_COUNT];
  float probabilities[TOKEN_COUNT];

  LOOM_UNROLL_FULL
  for (unsigned int token = 0; token < TOKEN_COUNT; ++token) {
    LOOM_UNROLL_FULL
    for (unsigned int lane = 0; lane < FEATURE_LANE_COUNT; ++lane) {
      query_local[token][lane] = query[token][lane];
      key_value_local[token][lane] = key_value[token][lane];
    }
  }

  LOOM_UNROLL_FULL
  for (unsigned int row = 0; row < TOKEN_COUNT; ++row) {
    float maximum = 0.0f;
    LOOM_UNROLL_FULL
    for (unsigned int column = 0; column < TOKEN_COUNT; ++column) {
      float dot = 0.0f;
      LOOM_UNROLL_FULL
      for (unsigned int lane = 0; lane < FEATURE_LANE_COUNT; ++lane)
        dot += query_local[row][lane] * key_value_local[column][lane];
      scores[column] = dot * 0.5f;
      if (column == 0 || scores[column] > maximum)
        maximum = scores[column];
    }

    float denominator = 0.0f;
    LOOM_UNROLL_FULL
    for (unsigned int column = 0; column < TOKEN_COUNT; ++column) {
      probabilities[column] = __builtin_expf(scores[column] - maximum);
      denominator += probabilities[column];
    }

    LOOM_UNROLL_FULL
    for (unsigned int lane = 0; lane < FEATURE_LANE_COUNT; ++lane) {
      float value = 0.0f;
      LOOM_UNROLL_FULL
      for (unsigned int column = 0; column < TOKEN_COUNT; ++column)
        value += probabilities[column] * key_value_local[column][lane];
      output[row][lane] = value / denominator;
    }
  }
}

__attribute__((weak)) void reduce_statistics(
    const float projected_audio[restrict TOKEN_COUNT][FEATURE_LANE_COUNT],
    const float attention[restrict TOKEN_COUNT][FEATURE_LANE_COUNT],
    float output[restrict TOKEN_COUNT][FEATURE_LANE_COUNT],
    float statistics[restrict STATISTIC_COUNT]) {
  float projection_energy = 0.0f;
  float attention_sum = 0.0f;
  float weighted_sum = 0.0f;
  float maximum_magnitude = 0.0f;

  LOOM_UNROLL_FULL
  for (unsigned int token = 0; token < TOKEN_COUNT; ++token) {
    LOOM_UNROLL_FULL
    for (unsigned int lane = 0; lane < FEATURE_LANE_COUNT; ++lane) {
      const float projection = projected_audio[token][lane];
      const float value = attention[token][lane];
      const float magnitude = value < 0.0f ? -value : value;
      projection_energy += projection * projection;
      attention_sum += value;
      weighted_sum += (float)(token * FEATURE_LANE_COUNT + lane + 1u) * value;
      if (magnitude > maximum_magnitude)
        maximum_magnitude = magnitude;
      output[token][lane] = value + projection * 0.125f;
    }
  }

  statistics[0] = projection_energy;
  statistics[1] = attention_sum;
  statistics[2] = weighted_sum;
  statistics[3] = maximum_magnitude;
}

__attribute__((noinline)) void loom_multisensor_attention(
    const float audio[restrict TOKEN_COUNT][SENSOR_LANE_COUNT],
    const float imu[restrict TOKEN_COUNT][SENSOR_LANE_COUNT],
    float output[restrict TOKEN_COUNT][FEATURE_LANE_COUNT],
    float statistics[restrict STATISTIC_COUNT]) {
  float projected_audio[TOKEN_COUNT][FEATURE_LANE_COUNT];
  float projected_imu[TOKEN_COUNT][FEATURE_LANE_COUNT];
  float attention[TOKEN_COUNT][FEATURE_LANE_COUNT];

  project_audio(audio, projected_audio);
  project_imu(imu, projected_imu);
  fuse_attention(projected_audio, projected_imu, attention);
  reduce_statistics(projected_audio, attention, output, statistics);
}

int main(void) {
  static const float audio[TOKEN_COUNT][SENSOR_LANE_COUNT] = {
      {1.0f, 0.0f, 0.5f},
      {0.5f, 1.0f, -0.5f},
      {-1.0f, 0.5f, 1.0f},
      {0.25f, -0.75f, 0.5f},
  };
  static const float imu[TOKEN_COUNT][SENSOR_LANE_COUNT] = {
      {0.5f, 1.0f, 0.0f},
      {-0.5f, 0.25f, 1.0f},
      {1.0f, -0.5f, 0.5f},
      {0.0f, 0.75f, -1.0f},
  };
  float output[TOKEN_COUNT][FEATURE_LANE_COUNT];
  float statistics[STATISTIC_COUNT];

  loom_multisensor_attention(audio, imu, output, statistics);
  const float combined = statistics[2] + 3.0f * statistics[0] +
                         7.0f * statistics[1] + 11.0f * statistics[3];
  if (outside_tolerance(statistics[0], 9.62500f) ||
      outside_tolerance(statistics[1], 3.93196f) ||
      outside_tolerance(statistics[2], 24.58165f) ||
      outside_tolerance(statistics[3], 0.73098f) ||
      outside_tolerance(combined, 89.02110f))
    return 1;
#if defined(LOOM_ATTENTION_PRODUCT_EXECUTION)
  return 0;
#else
  printf("attention checksum: %.5f\n", statistics[2]);
  printf("projection energy: %.5f\n", statistics[0]);
  printf("attention sum: %.5f\n", statistics[1]);
  printf("attention max: %.5f\n", statistics[3]);
  printf("combined checksum: %.5f\n", combined);
  printf("PASSED\n");
  return 0;
#endif
}
