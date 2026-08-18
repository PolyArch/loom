#ifndef LOOM_TEST_APPLICATIONS_MLPERF_TINY_ANOMALY_SMOKE_H
#define LOOM_TEST_APPLICATIONS_MLPERF_TINY_ANOMALY_SMOKE_H

#include <stdarg.h>
#include <stddef.h>
#include <stdint.h>

#define MLPERF_TINY_V0_1_API_SUBMITTER_IMPLEMENTED_H_
#define MLPERF_TINY_V0_1_API_INTERNALLY_IMPLEMENTED_H_
#define TENSORFLOW_LITE_MICRO_EXAMPLES_ANOMALY_DETECTION_MICRO_FEATURES_MICRO_MODEL_SETTINGS_H_
#define EE_CFG_ENERGY_MODE 0
#define EE_MSG_TIMESTAMP ""
#define EE_MSG_TIMESTAMP_MODE ""
#define TFLITE_SCHEMA_VERSION 3
#define kTfLiteInt8 1
#define TF_LITE_REPORT_ERROR(...)

inline constexpr int kFeatureSliceSize = 4;
inline constexpr int kFeatureSliceCount = 4;
inline constexpr int kFeatureElementCount =
    kFeatureSliceSize * kFeatureSliceCount;
inline constexpr int kSpectrogramSliceCount = 16;
inline constexpr int kInputSize = kFeatureElementCount;

enum TfLiteStatus { kTfLiteOk = 0, kTfLiteError = 1 };

struct TfLiteIntArray final {
  int size = 2;
  int data[2] = {1, kInputSize};
};

struct TfLiteQuantizationParams final {
  float scale = 0.25f;
  int zero_point = 0;
};

union TfLiteTensorData {
  int8_t *int8;
  uint8_t *uint8;
};

struct TfLiteTensor final {
  TfLiteTensorData data{};
  TfLiteIntArray *dims = nullptr;
  int type = kTfLiteInt8;
  TfLiteQuantizationParams params{};
};

namespace mlperf_tiny_smoke {

inline int8_t input[kInputSize]{};
inline int8_t output[kInputSize]{};
inline TfLiteIntArray dimensions;
inline TfLiteTensor inputTensor{{input}, &dimensions, kTfLiteInt8, {0.25f, 0}};
inline TfLiteTensor outputTensor{
    {output}, &dimensions, kTfLiteInt8, {0.25f, 0}};

} // namespace mlperf_tiny_smoke

namespace tflite {

class ErrorReporter {};
class MicroErrorReporter final : public ErrorReporter {};

class Model final {
public:
  int version() const { return TFLITE_SCHEMA_VERSION; }
};

inline const Model *GetModel(const unsigned char *) {
  static Model model;
  return &model;
}

template <int OperationCount> class MicroMutableOpResolver final {
public:
  explicit MicroMutableOpResolver(ErrorReporter * = nullptr) {}
  TfLiteStatus AddFullyConnected() { return kTfLiteOk; }
  TfLiteStatus AddQuantize() { return kTfLiteOk; }
  TfLiteStatus AddDequantize() { return kTfLiteOk; }
};

class MicroInterpreter final {
public:
  template <typename Resolver>
  MicroInterpreter(const Model *, Resolver &, uint8_t *, size_t,
                   ErrorReporter *) {}

  TfLiteStatus AllocateTensors() { return kTfLiteOk; }
  TfLiteTensor *input(int) { return &mlperf_tiny_smoke::inputTensor; }
  TfLiteTensor *output(int) { return &mlperf_tiny_smoke::outputTensor; }
  TfLiteStatus Invoke() {
    for (size_t index = 0; index < kInputSize; ++index) {
      int value = mlperf_tiny_smoke::input[index];
      if ((index & 3U) == 0)
        ++value;
      mlperf_tiny_smoke::output[index] = static_cast<int8_t>(value);
    }
    return kTfLiteOk;
  }
};

} // namespace tflite

class UnbufferedSerial final {
public:
  UnbufferedSerial(int, int) {}
  void baud(int) {}
};

class DigitalOut final {
public:
  explicit DigitalOut(int) {}
  DigitalOut &operator=(int) { return *this; }
};

inline constexpr int USBTX = 0;
inline constexpr int USBRX = 1;
inline constexpr int D7 = 7;
inline const unsigned char g_model[] = {0};

inline size_t ee_get_buffer(uint8_t *destination, size_t bytes) {
  float *values = reinterpret_cast<float *>(destination);
  const size_t count = bytes / sizeof(float);
  for (size_t index = 0; index < count; ++index)
    values[index] =
        static_cast<float>(static_cast<int>(index % 17) - 8) * 0.25f;
  return bytes;
}

inline void ee_serial_command_parser_callback(char *) {}
inline unsigned long us_ticker_read() { return 0; }

inline float DequantizeInt8ToFloat(int8_t value, float scale, int zeroPoint);

extern "C" __attribute__((noinline)) float
mlperf_tiny_anomaly_mse_kernel(const int8_t *output, const float *input,
                               size_t count, float scale, int zeroPoint) {
  float differenceSum = 0.0f;
  for (size_t index = 0; index < count; ++index) {
    const float converted =
        DequantizeInt8ToFloat(output[index], scale, zeroPoint);
    const float difference = converted - input[index];
    differenceSum += difference * difference;
  }
  return differenceSum / static_cast<float>(count);
}

void th_printf(const char *, ...);

static float mlperfTinyAbs(float value) {
  return value < 0.0f ? -value : value;
}

int main() {
  int8_t output[kInputSize]{};
  float input[kInputSize]{};
  for (size_t index = 0; index < kInputSize; ++index) {
    const int value = static_cast<int>(index % 17) - 8;
    input[index] = static_cast<float>(value) * 0.25f;
    output[index] = static_cast<int8_t>(value + ((index & 3U) == 0));
  }
  const float result =
      mlperf_tiny_anomaly_mse_kernel(output, input, kInputSize, 0.25f, 0);
  return mlperfTinyAbs(result - 0.015625f) <= 1.0e-6f ? 0 : 1;
}

#endif
