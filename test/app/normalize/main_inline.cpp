
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>

namespace {

constexpr uint32_t kSize = 8;
constexpr std::array<float, kSize> kInput = {
    1.0f,
    2.0f,
    3.0f,
    4.0f,
    5.0f,
    6.0f,
    7.0f,
    8.0f,
};
constexpr float kTolerance = 1.0e-5f;

void normalize_ref(const float *input, float *output, uint32_t size) {
    float sum = 0.0f;
    for (uint32_t i = 0; i < size; ++i) {
        sum += input[i];
    }
    const float scale = (sum > 0.0f) ? (1.0f / sum) : 1.0f;
    for (uint32_t i = 0; i < size; ++i) {
        output[i] = input[i] * scale;
    }
}

float checksum(const std::array<float, kSize> &values) {
    float total = 0.0f;
    for (uint32_t i = 0; i < kSize; ++i) {
        total += static_cast<float>(i + 1u) * values[i];
    }
    return total;
}

} // namespace

int main() {
    std::array<float, kSize> input = kInput;
    std::array<float, kSize> reference = {};
    std::array<float, kSize> candidate = {};

    normalize_ref(input.data(), reference.data(), kSize);

    float sum_result = 0.0f;
    float max_result = input[0];
    for (uint32_t i = 0; i < kSize; ++i) {
        sum_result += input[i];
    }
    for (uint32_t i = 1; i < kSize; ++i) {
        if (input[i] > max_result) {
            max_result = input[i];
        }
    }
    const float scale = (sum_result > 0.0f) ? (1.0f / sum_result) : 1.0f;
    for (uint32_t i = 0; i < kSize; ++i) {
        candidate[i] = input[i] * scale;
    }
    (void)max_result;

    for (uint32_t i = 0; i < kSize; ++i) {
        if (std::fabs(reference[i] - candidate[i]) > kTolerance) {
            std::puts("FAILED");
            return 1;
        }
    }

    std::printf("normalize checksum: %.6f\n", checksum(candidate));
    std::puts("PASSED");
    return 0;
}
