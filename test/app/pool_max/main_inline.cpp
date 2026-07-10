
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

int main() {
    std::array<float, kOutH * kOutW> candidate = {};

    for (uint32_t oh = 0; oh < kOutH; ++oh) {
        for (uint32_t ow = 0; ow < kOutW; ++ow) {
            float max_val = kInitialMax;
            for (uint32_t ph = 0; ph < kPoolH; ++ph) {
                for (uint32_t pw = 0; pw < kPoolW; ++pw) {
                    const uint32_t h = oh * kStrideH + ph;
                    const uint32_t w = ow * kStrideW + pw;
                    const float value = kInput[h * kWidth + w];
                    max_val = value > max_val ? value : max_val;
                }
            }
            candidate[oh * kOutW + ow] = max_val;
        }
    }

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
