
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>

namespace {

constexpr uint32_t kHeight = 4;
constexpr uint32_t kWidth = 4;
constexpr uint32_t kPoolH = 2;
constexpr uint32_t kPoolW = 2;
constexpr uint32_t kStrideH = 2;
constexpr uint32_t kStrideW = 2;
constexpr uint32_t kOutH = 2;
constexpr uint32_t kOutW = 2;
constexpr float kInitialMax = -1.0e30f;
constexpr float kTolerance = 1.0e-5f;
constexpr std::array<float, kHeight * kWidth> kInput = {
    1.0f, 2.0f, 3.0f, 4.0f,
    5.0f, 6.0f, 7.0f, 8.0f,
    9.0f, 10.0f, 11.0f, 12.0f,
    13.0f, 14.0f, 15.0f, 16.0f};
constexpr std::array<float, kOutH * kOutW> kExpected = {
    6.0f, 8.0f, 14.0f, 16.0f};

double checksum(const std::array<float, kOutH * kOutW> &values) {
    double sum = 0.0;
    for (uint32_t i = 0; i < values.size(); ++i) {
        sum += static_cast<double>(i + 1u) * values[i];
    }
    return sum;
}

} // namespace

extern "C" __attribute__((noinline))
void pool_max_kernel(const float *input, float *output,
                     uint32_t height, uint32_t width,
                     uint32_t pool_h, uint32_t pool_w,
                     uint32_t stride_h, uint32_t stride_w) {
    const uint32_t out_h = (height - pool_h) / stride_h + 1u;
    const uint32_t out_w = (width - pool_w) / stride_w + 1u;
    for (uint32_t oh = 0; oh < out_h; ++oh) {
        for (uint32_t ow = 0; ow < out_w; ++ow) {
            float max_val = kInitialMax;
            for (uint32_t ph = 0; ph < pool_h; ++ph) {
                for (uint32_t pw = 0; pw < pool_w; ++pw) {
                    const uint32_t h = oh * stride_h + ph;
                    const uint32_t w = ow * stride_w + pw;
                    const float candidate = input[h * width + w];
                    max_val = candidate > max_val ? candidate : max_val;
                }
            }
            output[oh * out_w + ow] = max_val;
        }
    }
}

int main() {
    std::array<float, kOutH * kOutW> candidate = {};

    pool_max_kernel(kInput.data(), candidate.data(),
                    kHeight, kWidth, kPoolH, kPoolW, kStrideH, kStrideW);

    for (uint32_t i = 0; i < candidate.size(); ++i) {
        if (std::fabs(candidate[i] - kExpected[i]) > kTolerance) {
            std::puts("FAILED");
            return 1;
        }
    }

    std::printf("pool_max checksum: %.6f\n", checksum(candidate));
    std::puts("PASSED");
    return 0;
}
