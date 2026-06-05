// Column-to-image overlap accumulation inline variant.

#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>

namespace {

constexpr uint32_t kChannels = 1;
constexpr uint32_t kHeight = 4;
constexpr uint32_t kWidth = 4;
constexpr uint32_t kKernelH = 2;
constexpr uint32_t kKernelW = 2;
constexpr uint32_t kStrideH = 1;
constexpr uint32_t kStrideW = 1;
constexpr uint32_t kOutH = 3;
constexpr uint32_t kOutW = 3;
constexpr uint32_t kRows = kChannels * kKernelH * kKernelW;
constexpr uint32_t kCols = kOutH * kOutW;
constexpr float kTolerance = 1e-5f;
constexpr std::array<float, kRows * kCols> kInput = {
    1.0f, 2.0f, 3.0f, 5.0f, 6.0f, 7.0f, 9.0f, 10.0f, 11.0f,
    2.0f, 3.0f, 4.0f, 6.0f, 7.0f, 8.0f, 10.0f, 11.0f, 12.0f,
    5.0f, 6.0f, 7.0f, 9.0f, 10.0f, 11.0f, 13.0f, 14.0f, 15.0f,
    6.0f, 7.0f, 8.0f, 10.0f, 11.0f, 12.0f, 14.0f, 15.0f, 16.0f};
constexpr std::array<float, kChannels * kHeight * kWidth> kExpected = {
    1.0f, 4.0f, 6.0f, 4.0f, 10.0f, 24.0f, 28.0f, 16.0f,
    18.0f, 40.0f, 44.0f, 24.0f, 13.0f, 28.0f, 30.0f, 16.0f};

double checksum(const std::array<float, kChannels * kHeight * kWidth> &values) {
    double sum = 0.0;
    for (uint32_t i = 0; i < values.size(); ++i) {
        sum += static_cast<double>(i + 1u) * values[i];
    }
    return sum;
}

} // namespace

int main() {
    std::array<float, kChannels * kHeight * kWidth> output = {};

    for (uint32_t i = 0; i < output.size(); ++i) {
        output[i] = 0.0f;
    }
    for (uint32_t c = 0; c < kChannels; ++c) {
        for (uint32_t kh = 0; kh < kKernelH; ++kh) {
            for (uint32_t kw = 0; kw < kKernelW; ++kw) {
                const uint32_t row = c * (kKernelH * kKernelW) + kh * kKernelW + kw;
                for (uint32_t oh = 0; oh < kOutH; ++oh) {
                    for (uint32_t ow = 0; ow < kOutW; ++ow) {
                        const uint32_t h = oh * kStrideH + kh;
                        const uint32_t w = ow * kStrideW + kw;
                        const uint32_t col = oh * kOutW + ow;
                        output[c * (kHeight * kWidth) + h * kWidth + w] +=
                            kInput[row * (kOutH * kOutW) + col];
                    }
                }
            }
        }
    }

    for (uint32_t i = 0; i < output.size(); ++i) {
        if (std::fabs(output[i] - kExpected[i]) > kTolerance) {
            std::puts("FAILED");
            return 1;
        }
    }

    std::printf("col2im checksum: %.3f\n", checksum(output));
    std::puts("PASSED");
    return 0;
}
