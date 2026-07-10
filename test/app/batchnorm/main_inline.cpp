
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>

namespace {

constexpr uint32_t kChannels = 4;
constexpr uint32_t kHeight = 8;
constexpr uint32_t kWidth = 8;
constexpr uint32_t kElementCount = kChannels * kHeight * kWidth;
constexpr float kEpsilon = 1.0e-5f;
constexpr float kTolerance = 1.0e-4f;

void initialize_inputs(std::array<float, kElementCount> &input,
                       std::array<float, kChannels> &mean,
                       std::array<float, kChannels> &variance,
                       std::array<float, kChannels> &gamma,
                       std::array<float, kChannels> &beta) {
    for (uint32_t i = 0; i < kElementCount; ++i) {
        input[i] = static_cast<float>(i % 100) - 50.0f;
    }
    for (uint32_t c = 0; c < kChannels; ++c) {
        mean[c] = static_cast<float>(c * 10);
        variance[c] = static_cast<float>(c + 1) * 2.0f;
        gamma[c] = 1.0f;
        beta[c] = 0.0f;
    }
}

float checksum(const std::array<float, kElementCount> &values) {
    float sum = 0.0f;
    for (float value : values) {
        sum += value;
    }
    return sum;
}

} // namespace

int main() {
    std::array<float, kElementCount> input = {};
    std::array<float, kChannels> mean = {};
    std::array<float, kChannels> variance = {};
    std::array<float, kChannels> gamma = {};
    std::array<float, kChannels> beta = {};
    std::array<float, kElementCount> expected = {};
    std::array<float, kElementCount> candidate = {};
    initialize_inputs(input, mean, variance, gamma, beta);

    for (uint32_t c = 0; c < kChannels; ++c) {
        float inv_std = 1.0f / sqrtf(variance[c] + kEpsilon);
        for (uint32_t h = 0; h < kHeight; ++h) {
            for (uint32_t w = 0; w < kWidth; ++w) {
                uint32_t idx = c * (kHeight * kWidth) + h * kWidth + w;
                float normalized = (input[idx] - mean[c]) * inv_std;
                expected[idx] = gamma[c] * normalized + beta[c];
            }
        }
    }

    for (uint32_t c = 0; c < kChannels; ++c) {
        float inv_std = 1.0f / sqrtf(variance[c] + kEpsilon);
        for (uint32_t h = 0; h < kHeight; ++h) {
            for (uint32_t w = 0; w < kWidth; ++w) {
                uint32_t idx = c * (kHeight * kWidth) + h * kWidth + w;
                float normalized = (input[idx] - mean[c]) * inv_std;
                candidate[idx] = gamma[c] * normalized + beta[c];
            }
        }
    }

    for (uint32_t i = 0; i < kElementCount; ++i) {
        if (std::fabs(expected[i] - candidate[i]) > kTolerance) {
            std::puts("FAILED");
            return 1;
        }
    }

    std::printf("batchnorm checksum: %.6f\n", checksum(candidate));
    std::puts("PASSED");
    return 0;
}
