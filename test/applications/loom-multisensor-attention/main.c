#include "../../../include/Runtime/Computation.h"
#if !defined(LOOM_ATTENTION_PRODUCT_EXECUTION) || \
    defined(LOOM_APPLICATION_HOST_EXECUTION)
#include <stdio.h>
#endif

/* Token extent. The smoke input keeps the smallest sequence that still
   exercises every projection, the attention fusion, and the reduction. A
   qualified input scales the same sequence until the measured computation
   dominates the System's fixed per-launch cost, so the saturation target it
   declares is reachable at all. */
#if !defined(LOOM_ATTENTION_TOKEN_COUNT)
#define LOOM_ATTENTION_TOKEN_COUNT 4
#endif

enum {
  TOKEN_COUNT = LOOM_ATTENTION_TOKEN_COUNT,
  SENSOR_LANE_COUNT = 3,
  FEATURE_LANE_COUNT = 4,
  STATISTIC_COUNT = 4,
};

#if !defined(LOOM_ATTENTION_INPUT_VARIANT)
#define LOOM_ATTENTION_INPUT_VARIANT 0
#endif

#if LOOM_ATTENTION_INPUT_VARIANT == 1
#define LOOM_ATTENTION_INPUT_SCALE 0.5f
#elif LOOM_ATTENTION_INPUT_VARIANT == 2
#define LOOM_ATTENTION_INPUT_SCALE 1.5f
#else
#define LOOM_ATTENTION_INPUT_SCALE 1.0f
#endif

#define LOOM_ATTENTION_SCALED(value) ((value) * LOOM_ATTENTION_INPUT_SCALE)

#if defined(__clang__) && !defined(LOOM_ATTENTION_RETAIN_LOOPS)
#define LOOM_UNROLL_FULL _Pragma("clang loop unroll(full)")
#else
#define LOOM_UNROLL_FULL
#endif

static float absolute_value(float value) {
  return value < 0.0f ? -value : value;
}

/* One relative bound covers every extent: a float32 reduction over the whole
   sequence admits about one rounding of relative size 2^-24 per term, and the
   accelerator may retire the same reduction in another order. */
static int outside_tolerance(float actual, float expected) {
  return absolute_value(actual - expected) >
         1.0e-4f * (1.0f + absolute_value(expected));
}

/* A bounded deterministic sensor pattern. Every sample is an exact multiple of
   1/128 in (-1, 1); the prime modulus and its coprime stride keep successive
   tokens distinct across the whole sequence. */
static float sensor_sample(unsigned int ordinal, unsigned int phase) {
  const unsigned int mixed = (ordinal * 37u + phase * 101u) % 251u;
  return LOOM_ATTENTION_SCALED((float)((int)mixed - 125) * (1.0f / 128.0f));
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

static float attention_audio[TOKEN_COUNT][SENSOR_LANE_COUNT];
static float attention_imu[TOKEN_COUNT][SENSOR_LANE_COUNT];
static float reference_query[TOKEN_COUNT][FEATURE_LANE_COUNT];
static float reference_key_value[TOKEN_COUNT][FEATURE_LANE_COUNT];
static float reference_attention[TOKEN_COUNT][FEATURE_LANE_COUNT];
static float reference_statistics[STATISTIC_COUNT];
static float reference_scores[TOKEN_COUNT];
static float reference_probabilities[TOKEN_COUNT];

/* An independent host implementation of the same fused attention, evaluated
   outside the measured computation interval. It replaces the statistic
   constants this program used to transcribe for one fixed token count, so
   every extent keeps an exact oracle. */
__attribute__((noinline)) static void reference_attention_statistics(void) {
  for (unsigned int token = 0; token < TOKEN_COUNT; ++token) {
    const float audio_x = attention_audio[token][0];
    const float audio_y = attention_audio[token][1];
    const float imu_x = attention_imu[token][0];
    const float imu_y = attention_imu[token][1];
    reference_query[token][0] = audio_x;
    reference_query[token][1] = audio_y;
    reference_query[token][2] = attention_audio[token][2];
    reference_query[token][3] = audio_x + audio_y;
    reference_key_value[token][0] = imu_x;
    reference_key_value[token][1] = imu_y;
    reference_key_value[token][2] = attention_imu[token][2];
    reference_key_value[token][3] = imu_x - imu_y;
  }

  for (unsigned int row = 0; row < TOKEN_COUNT; ++row) {
    float maximum = 0.0f;
    for (unsigned int column = 0; column < TOKEN_COUNT; ++column) {
      float dot = 0.0f;
      for (unsigned int lane = 0; lane < FEATURE_LANE_COUNT; ++lane)
        dot += reference_query[row][lane] * reference_key_value[column][lane];
      reference_scores[column] = dot * 0.5f;
      if (column == 0 || reference_scores[column] > maximum)
        maximum = reference_scores[column];
    }
    float denominator = 0.0f;
    for (unsigned int column = 0; column < TOKEN_COUNT; ++column) {
      reference_probabilities[column] =
          __builtin_expf(reference_scores[column] - maximum);
      denominator += reference_probabilities[column];
    }
    for (unsigned int lane = 0; lane < FEATURE_LANE_COUNT; ++lane) {
      float value = 0.0f;
      for (unsigned int column = 0; column < TOKEN_COUNT; ++column)
        value +=
            reference_probabilities[column] * reference_key_value[column][lane];
      reference_attention[row][lane] = value / denominator;
    }
  }

  float projection_energy = 0.0f;
  float attention_sum = 0.0f;
  float weighted_sum = 0.0f;
  float maximum_magnitude = 0.0f;
  for (unsigned int token = 0; token < TOKEN_COUNT; ++token)
    for (unsigned int lane = 0; lane < FEATURE_LANE_COUNT; ++lane) {
      const float projection = reference_query[token][lane];
      const float value = reference_attention[token][lane];
      const float magnitude = absolute_value(value);
      projection_energy += projection * projection;
      attention_sum += value;
      weighted_sum += (float)(token * FEATURE_LANE_COUNT + lane + 1u) * value;
      if (magnitude > maximum_magnitude)
        maximum_magnitude = magnitude;
    }
  reference_statistics[0] = projection_energy;
  reference_statistics[1] = attention_sum;
  reference_statistics[2] = weighted_sum;
  reference_statistics[3] = maximum_magnitude;
}

int main(void) {
  for (unsigned int token = 0; token < TOKEN_COUNT; ++token)
    for (unsigned int lane = 0; lane < SENSOR_LANE_COUNT; ++lane) {
      const unsigned int ordinal = token * SENSOR_LANE_COUNT + lane;
      attention_audio[token][lane] = sensor_sample(ordinal, 0u);
      attention_imu[token][lane] = sensor_sample(ordinal, 1u);
    }
  float output[TOKEN_COUNT][FEATURE_LANE_COUNT];
  float statistics[STATISTIC_COUNT];

  loom_computation_begin();
  loom_multisensor_attention(
      (const float(*)[SENSOR_LANE_COUNT])attention_audio,
      (const float(*)[SENSOR_LANE_COUNT])attention_imu, output, statistics);
  loom_computation_end();
  reference_attention_statistics();
  const float combined = statistics[2] + 3.0f * statistics[0] +
                         7.0f * statistics[1] + 11.0f * statistics[3];
  const float expectedCombined =
      reference_statistics[2] + 3.0f * reference_statistics[0] +
      7.0f * reference_statistics[1] + 11.0f * reference_statistics[3];
  for (unsigned int ordinal = 0; ordinal < STATISTIC_COUNT; ++ordinal)
    if (outside_tolerance(statistics[ordinal], reference_statistics[ordinal]))
      return 1;
  if (outside_tolerance(combined, expectedCombined))
    return 1;
#if defined(LOOM_APPLICATION_HOST_EXECUTION)
  printf("attention tokens: %d\n", (int)TOKEN_COUNT);
  printf("attention checksum: %.5f\n", statistics[2]);
  printf("projection energy: %.5f\n", statistics[0]);
  printf("attention sum: %.5f\n", statistics[1]);
  printf("attention max: %.5f\n", statistics[3]);
  printf("combined checksum: %.5f\n", combined);
  printf("PASSED\n");
#endif
#if defined(LOOM_ATTENTION_PRODUCT_EXECUTION)
  return 0;
#else
#if !defined(LOOM_APPLICATION_HOST_EXECUTION)
  printf("attention tokens: %d\n", (int)TOKEN_COUNT);
  printf("attention checksum: %.5f\n", statistics[2]);
  printf("projection energy: %.5f\n", statistics[0]);
  printf("attention sum: %.5f\n", statistics[1]);
  printf("attention max: %.5f\n", statistics[3]);
  printf("combined checksum: %.5f\n", combined);
  printf("PASSED\n");
#endif
  return 0;
#endif
}
