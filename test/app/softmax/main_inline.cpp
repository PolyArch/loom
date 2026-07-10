
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>

namespace {

constexpr uint32_t kSize = 128;
constexpr float kTolerance = 1.0e-5f;

void initialize_input(std::array<float, kSize> &input) {
    for (uint32_t i = 0; i < kSize; ++i) {
        input[i] = static_cast<float>(i % 20) - 10.0f;
    }
}

float checksum(const std::array<float, kSize> &values) {
    float sum = 0.0f;
    for (float value : values) {
        sum += value;
    }
    return sum;
}

} // namespace

int main() {
    std::array<float, kSize> input = {};
    std::array<float, kSize> expected = {};
    std::array<float, kSize> candidate = {};
    initialize_input(input);

    float expected_max = input[0];
    for (uint32_t i = 1; i < kSize; ++i) {
        if (input[i] > expected_max) {
            expected_max = input[i];
        }
    }
    float expected_sum = 0.0f;
    for (uint32_t i = 0; i < kSize; ++i) {
        expected[i] = expf(input[i] - expected_max);
        expected_sum += expected[i];
    }
    for (uint32_t i = 0; i < kSize; ++i) {
        expected[i] = expected[i] / expected_sum;
    }

    float candidate_max = input[0];
    for (uint32_t i = 1; i < kSize; ++i) {
        if (input[i] > candidate_max) {
            candidate_max = input[i];
        }
    }
    float candidate_sum = 0.0f;
    for (uint32_t i = 0; i < kSize; ++i) {
        candidate[i] = expf(input[i] - candidate_max);
        candidate_sum += candidate[i];
    }
    for (uint32_t i = 0; i < kSize; ++i) {
        candidate[i] = candidate[i] / candidate_sum;
    }

    for (uint32_t i = 0; i < kSize; ++i) {
        if (std::fabs(expected[i] - candidate[i]) > kTolerance) {
            std::puts("FAILED");
            return 1;
        }
    }
    if (std::fabs(checksum(candidate) - 1.0f) > 1.0e-4f) {
        std::puts("FAILED");
        return 1;
    }

    std::printf("softmax checksum: %.6f\n", checksum(candidate));
    std::puts("PASSED");
    return 0;
}
