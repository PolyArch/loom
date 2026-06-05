// Direct 2D convolution inline variant.

#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>

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
constexpr std::array<float, kInputChannels * kHeight * kWidth> kInput = {
    1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f,
    9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f};
constexpr std::array<float, kOutputChannels * kInputChannels * kKernelH * kKernelW> kKernel = {
    1.0f, 0.0f, 0.5f, -1.0f,
    -0.5f, 1.0f, 0.25f, 0.75f};
constexpr std::array<float, kOutputChannels * kOutH * kOutW> kExpected = {
    -2.5f, -2.0f, -1.5f, -0.5f, 0.0f, 0.5f, 1.5f, 2.0f, 2.5f,
    7.25f, 8.75f, 10.25f, 13.25f, 14.75f, 16.25f, 19.25f, 20.75f, 22.25f};

double checksum(const std::array<float, kOutputChannels * kOutH * kOutW> &values) {
    double sum = 0.0;
    for (uint32_t i = 0; i < values.size(); ++i) {
        sum += static_cast<double>(i + 1u) * values[i];
    }
    return sum;
}

} // namespace

int main() {
    std::array<float, kOutputChannels * kOutH * kOutW> output = {};

    for (uint32_t co = 0; co < kOutputChannels; ++co) {
        for (uint32_t oh = 0; oh < kOutH; ++oh) {
            for (uint32_t ow = 0; ow < kOutW; ++ow) {
                float sum = 0.0f;
                for (uint32_t ci = 0; ci < kInputChannels; ++ci) {
                    for (uint32_t kh = 0; kh < kKernelH; ++kh) {
                        for (uint32_t kw = 0; kw < kKernelW; ++kw) {
                            const uint32_t h = oh * kStrideH + kh;
                            const uint32_t w = ow * kStrideW + kw;
                            const float input_value =
                                kInput[ci * (kHeight * kWidth) + h * kWidth + w];
                            const float kernel_value =
                                kKernel[co * (kInputChannels * kKernelH * kKernelW) +
                                        ci * (kKernelH * kKernelW) +
                                        kh * kKernelW + kw];
                            sum += input_value * kernel_value;
                        }
                    }
                }
                output[co * (kOutH * kOutW) + oh * kOutW + ow] = sum;
            }
        }
    }

    for (uint32_t i = 0; i < output.size(); ++i) {
        if (std::fabs(output[i] - kExpected[i]) > kTolerance) {
            std::puts("FAILED");
            return 1;
        }
    }

    std::printf("conv2d checksum: %.3f\n", checksum(output));
    std::puts("PASSED");
    return 0;
}
