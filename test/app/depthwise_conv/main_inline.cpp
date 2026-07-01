// Depthwise convolution inline variant.

#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>

namespace {

constexpr uint32_t kChannels = 4;
constexpr uint32_t kHeight = 8;
constexpr uint32_t kWidth = 8;
constexpr uint32_t kKernelH = 3;
constexpr uint32_t kKernelW = 3;
constexpr uint32_t kOutH = 6;
constexpr uint32_t kOutW = 6;
constexpr uint32_t kOutputCount = kChannels * kOutH * kOutW;
constexpr float kTolerance = 1.0e-4f;

void initialize(std::array<float, kChannels * kHeight * kWidth> &input,
                std::array<float, kChannels * kKernelH * kKernelW> &kernel) {
    for (uint32_t i = 0; i < input.size(); ++i) {
        input[i] = static_cast<float>(i % 10u);
    }
    for (uint32_t i = 0; i < kernel.size(); ++i) {
        kernel[i] = (static_cast<float>(i % 5u) - 2.0f) / 10.0f;
    }
}

void depthwise_ref(const float *input, const float *kernel, float *output) {
    for (uint32_t c = 0; c < kChannels; ++c) {
        for (uint32_t oh = 0; oh < kOutH; ++oh) {
            for (uint32_t ow = 0; ow < kOutW; ++ow) {
                float sum = 0.0f;
                for (uint32_t kh = 0; kh < kKernelH; ++kh) {
                    for (uint32_t kw = 0; kw < kKernelW; ++kw) {
                        const uint32_t h = oh + kh;
                        const uint32_t w = ow + kw;
                        const float input_value =
                            input[c * (kHeight * kWidth) + h * kWidth + w];
                        const float kernel_value =
                            kernel[c * (kKernelH * kKernelW) + kh * kKernelW + kw];
                        sum += input_value * kernel_value;
                    }
                }
                output[c * (kOutH * kOutW) + oh * kOutW + ow] = sum;
            }
        }
    }
}

double checksum(const std::array<float, kOutputCount> &values) {
    double total = 0.0;
    for (uint32_t i = 0; i < values.size(); ++i) {
        total += static_cast<double>(i + 1u) * values[i];
    }
    return total;
}

} // namespace

int main() {
    std::array<float, kChannels * kHeight * kWidth> input = {};
    std::array<float, kChannels * kKernelH * kKernelW> kernel = {};
    std::array<float, kOutputCount> expected = {};
    std::array<float, kOutputCount> actual = {};
    initialize(input, kernel);
    depthwise_ref(input.data(), kernel.data(), expected.data());

    for (uint32_t flat = 0; flat < kOutputCount; ++flat) {
        const uint32_t c = flat / (kOutH * kOutW);
        const uint32_t rem = flat - c * (kOutH * kOutW);
        const uint32_t oh = rem / kOutW;
        const uint32_t ow = rem - oh * kOutW;
        float sum = 0.0f;
        for (uint32_t kh = 0; kh < kKernelH; ++kh) {
            for (uint32_t kw = 0; kw < kKernelW; ++kw) {
                const uint32_t h = oh + kh;
                const uint32_t w = ow + kw;
                const float input_value =
                    input[c * (kHeight * kWidth) + h * kWidth + w];
                const float kernel_value =
                    kernel[c * (kKernelH * kKernelW) + kh * kKernelW + kw];
                sum += input_value * kernel_value;
            }
        }
        actual[flat] = sum;
    }

    for (uint32_t i = 0; i < kOutputCount; ++i) {
        if (std::fabs(expected[i] - actual[i]) > kTolerance) {
            std::puts("FAILED");
            return 1;
        }
    }

    std::printf("depthwise_conv checksum: %.3f\n", checksum(actual));
    std::puts("PASSED");
    return 0;
}
