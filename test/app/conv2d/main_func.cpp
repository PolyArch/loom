// Direct 2D convolution function variant.

#include <stdint.h>
#include <stdio.h>

namespace {

constexpr uint32_t kInputChannels = 1;
constexpr uint32_t kOutputChannels = 2;
constexpr uint32_t kHeight = 4;
constexpr uint32_t kWidth = 4;
constexpr uint32_t kKernelH = 2;
constexpr uint32_t kKernelW = 2;
constexpr uint32_t kStrideH = 1;
constexpr uint32_t kStrideW = 1;
constexpr uint32_t kOutH = 3;
constexpr uint32_t kOutW = 3;
constexpr float kTolerance = 1e-5f;
constexpr float kInput[kInputChannels * kHeight * kWidth] = {
    1.0f, 2.0f,  3.0f,  4.0f,  5.0f,  6.0f,  7.0f,  8.0f,
    9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f};
constexpr float kKernel[kOutputChannels * kInputChannels * kKernelH *
                        kKernelW] = {1.0f,  0.0f, 0.5f,  -1.0f,
                                     -0.5f, 1.0f, 0.25f, 0.75f};
constexpr float kExpected[kOutputChannels * kOutH * kOutW] = {
    -2.5f, -2.0f, -1.5f,  -0.5f,  0.0f,   0.5f,   1.5f,   2.0f,   2.5f,
    7.25f, 8.75f, 10.25f, 13.25f, 14.75f, 16.25f, 19.25f, 20.75f, 22.25f};

float absolute(float value) { return value < 0.0f ? -value : value; }

double checksum(const float *values) {
  double sum = 0.0;
  for (uint32_t i = 0; i < kOutputChannels * kOutH * kOutW; ++i) {
    sum += static_cast<double>(i + 1u) * values[i];
  }
  return sum;
}

} // namespace

extern "C" __attribute__((noinline)) void
conv2d_kernel(const float *input, const float *kernel, float *output,
              uint32_t in_channels, uint32_t out_channels, uint32_t height,
              uint32_t width, uint32_t kernel_h, uint32_t kernel_w,
              uint32_t stride_h, uint32_t stride_w) {
  const uint32_t out_h = (height - kernel_h) / stride_h + 1u;
  const uint32_t out_w = (width - kernel_w) / stride_w + 1u;
  for (uint32_t co = 0; co < out_channels; ++co) {
    for (uint32_t oh = 0; oh < out_h; ++oh) {
      for (uint32_t ow = 0; ow < out_w; ++ow) {
        float sum = 0.0f;
        for (uint32_t ci = 0; ci < in_channels; ++ci) {
          for (uint32_t kh = 0; kh < kernel_h; ++kh) {
            for (uint32_t kw = 0; kw < kernel_w; ++kw) {
              const uint32_t h = oh * stride_h + kh;
              const uint32_t w = ow * stride_w + kw;
              const float input_value =
                  input[ci * (height * width) + h * width + w];
              const float kernel_value =
                  kernel[co * (in_channels * kernel_h * kernel_w) +
                         ci * (kernel_h * kernel_w) + kh * kernel_w + kw];
              sum += input_value * kernel_value;
            }
          }
        }
        output[co * (out_h * out_w) + oh * out_w + ow] = sum;
      }
    }
  }
}

int main() {
  float candidate[kOutputChannels * kOutH * kOutW] = {};

  conv2d_kernel(kInput, kKernel, candidate, kInputChannels, kOutputChannels,
                kHeight, kWidth, kKernelH, kKernelW, kStrideH, kStrideW);

  for (uint32_t i = 0; i < kOutputChannels * kOutH * kOutW; ++i) {
    if (absolute(candidate[i] - kExpected[i]) > kTolerance) {
      printf("FAILED\n");
      return 1;
    }
  }

  printf("conv2d checksum: %.3f\n", checksum(candidate));
  printf("PASSED\n");
  return 0;
}
