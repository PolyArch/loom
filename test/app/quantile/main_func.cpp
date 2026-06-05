// Quantile function variant migrated from the legacy app corpus.

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

float quantile_ref(const float *sorted_input, uint32_t size, float q) {
    float pos = q * static_cast<float>(size - 1);
    uint32_t lower = static_cast<uint32_t>(pos);
    uint32_t upper = lower + 1;
    if (upper >= size) {
        return sorted_input[size - 1];
    }

    float frac = pos - static_cast<float>(lower);
    return sorted_input[lower] * (1.0f - frac) + sorted_input[upper] * frac;
}

extern "C" __attribute__((noinline))
float quantile_kernel(const float *sorted_input, uint32_t size, float q) {
    float pos = q * static_cast<float>(size - 1);
    uint32_t lower = static_cast<uint32_t>(pos);
    uint32_t upper = lower + 1;
    if (upper >= size) {
        return sorted_input[size - 1];
    }

    float frac = pos - static_cast<float>(lower);
    return sorted_input[lower] * (1.0f - frac) + sorted_input[upper] * frac;
}

} // namespace

int main() {
    std::array<float, kSize> input = {};
    initialize_input(input);

    float expected = quantile_ref(input.data(), kSize, kQ);
    float candidate = quantile_kernel(input.data(), kSize, kQ);
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
