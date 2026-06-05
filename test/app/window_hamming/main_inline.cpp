// Hamming window inline variant migrated from the legacy app corpus.

#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>

namespace {

constexpr uint32_t kSize = 256;
constexpr float kInputPi = 3.14159f;
constexpr float kWindowPi = 3.14159265358979323846f;
constexpr float kTolerance = 1e-6f;

void initialize_input(std::array<float, kSize> &input) {
    for (uint32_t i = 0; i < kSize; ++i) {
        input[i] = std::sin(2.0f * kInputPi * static_cast<float>(i) / 32.0f);
    }
}

void window_hamming_ref(const float *input, float *output, uint32_t size) {
    for (uint32_t n = 0; n < size; ++n) {
        const float angle =
            2.0f * kWindowPi * static_cast<float>(n) / static_cast<float>(size - 1u);
        const float window = 0.54f - 0.46f * std::cos(angle);
        output[n] = input[n] * window;
    }
}

float checksum(const std::array<float, kSize> &values) {
    float sum = 0.0f;
    for (uint32_t i = 0; i < kSize; ++i) {
        sum += static_cast<float>(i + 1u) * values[i];
    }
    return sum;
}

} // namespace

int main() {
    std::array<float, kSize> input = {};
    std::array<float, kSize> reference = {};
    std::array<float, kSize> candidate = {};

    initialize_input(input);
    window_hamming_ref(input.data(), reference.data(), kSize);

    for (uint32_t n = 0; n < kSize; ++n) {
        const float angle =
            2.0f * kWindowPi * static_cast<float>(n) / static_cast<float>(kSize - 1u);
        const float window = 0.54f - 0.46f * std::cos(angle);
        candidate[n] = input[n] * window;
    }

    for (uint32_t i = 0; i < kSize; ++i) {
        if (std::fabs(reference[i] - candidate[i]) > kTolerance) {
            std::puts("FAILED");
            return 1;
        }
    }

    std::printf("window_hamming checksum: %.3f\n", checksum(candidate));
    std::puts("PASSED");
    return 0;
}
