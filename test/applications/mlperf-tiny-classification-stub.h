#ifndef LOOM_TEST_APPLICATIONS_MLPERF_TINY_CLASSIFICATION_STUB_H
#define LOOM_TEST_APPLICATIONS_MLPERF_TINY_CLASSIFICATION_STUB_H

#include <stdarg.h>
#include <stddef.h>
#include <stdint.h>

#define MLPERF_TINY_V0_1_API_SUBMITTER_IMPLEMENTED_H_
#define MLPERF_TINY_V0_1_API_INTERNALLY_IMPLEMENTED_H_
#define EE_CFG_ENERGY_MODE 0
#define EE_MSG_TIMESTAMP ""
#define EE_MSG_TIMESTAMP_MODE ""
#define kTfLiteOk 0

namespace tflite {

template <int OperationCount> class MicroMutableOpResolver final {
public:
  void AddAdd() {}
  void AddFullyConnected() {}
  void AddConv2D() {}
  void AddDepthwiseConv2D() {}
  void AddReshape() {}
  void AddSoftmax() {}
  void AddAveragePool2D() {}
};

template <typename Input, typename Output, int OperationCount>
class MicroModelRunner final {
public:
  template <typename Resolver>
  MicroModelRunner(const unsigned char *, Resolver &, uint8_t *, size_t) {}

  void SetInput(const Input *) {}
  void Invoke() {}
  Output *GetOutput() { return output_; }
  float output_scale() const { return 0.125f; }
  int output_zero_point() const { return 0; }

private:
  Output output_[16]{};
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

inline size_t ee_get_buffer(uint8_t *destination, size_t bytes) {
  for (size_t index = 0; index < bytes; ++index)
    destination[index] = static_cast<uint8_t>(index * 13U + 7U);
  return bytes;
}

inline void ee_serial_command_parser_callback(char *) {}
inline unsigned long us_ticker_read() { return 0; }
void th_printf(const char *, ...);

inline float DequantizeInt8ToFloat(int8_t value, float scale, int zeroPoint);

#endif
