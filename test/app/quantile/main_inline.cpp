// Quantile inline variant migrated from the legacy app corpus.

#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>

namespace {

constexpr uint32_t kSize = 1024;
constexpr float kQ = 0.5f;
constexpr float kTolerance = 1.0e-5f;

void initialize_input(std::array<float, kSize> &input) {
    for (uint32_t i = 0; i < kSize; ++i) {
        input[i] = static_cast<float>(i);
    }
}

} // namespace

int main() {
    std::array<float, kSize> input = {};
    initialize_input(input);

    float expected_pos = kQ * static_cast<float>(kSize - 1);
    uint32_t expected_lower = static_cast<uint32_t>(expected_pos);
    uint32_t expected_upper = expected_lower + 1;
    float expected = input[kSize - 1];
    if (expected_upper < kSize) {
        float frac = expected_pos - static_cast<float>(expected_lower);
        expected = input[expected_lower] * (1.0f - frac) +
                   input[expected_upper] * frac;
    }

    float candidate_pos = kQ * static_cast<float>(kSize - 1);
    uint32_t candidate_lower = static_cast<uint32_t>(candidate_pos);
    uint32_t candidate_upper = candidate_lower + 1;
    float candidate = input[kSize - 1];
    if (candidate_upper < kSize) {
        float frac = candidate_pos - static_cast<float>(candidate_lower);
        candidate = input[candidate_lower] * (1.0f - frac) +
                    input[candidate_upper] * frac;
    }

    if (std::fabs(expected - candidate) > kTolerance) {
        std::puts("FAILED");
        return 1;
    }
    if (std::fabs(expected - 511.5f) > 1.0e-3f) {
        std::puts("FAILED");
        return 1;
    }

    std::printf("quantile result: %.3f\n", candidate);
    std::puts("PASSED");
    return 0;
}
