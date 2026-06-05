// Bubble-sort inline variant migrated from the legacy app corpus.

#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>

namespace {

constexpr uint32_t kSize = 12;
constexpr float kTolerance = 1e-5f;
constexpr std::array<float, kSize> kInput = {
    9.0f, 1.5f, 4.0f, 4.0f, -2.0f, 7.25f,
    0.0f, 3.5f, 8.0f, -1.0f, 2.0f, 6.0f};
constexpr std::array<float, kSize> kExpected = {
    -2.0f, -1.0f, 0.0f, 1.5f, 2.0f, 3.5f,
    4.0f, 4.0f, 6.0f, 7.25f, 8.0f, 9.0f};

double checksum(const std::array<float, kSize> &values) {
    double sum = 0.0;
    for (uint32_t i = 0; i < kSize; ++i) {
        sum += static_cast<double>(i + 1u) * values[i];
    }
    return sum;
}

} // namespace

int main() {
    std::array<float, kSize> output = {};
    for (uint32_t i = 0; i < kSize; ++i) {
        output[i] = kInput[i];
    }

    for (uint32_t i = 0; i + 1u < kSize; ++i) {
        for (uint32_t j = 0; j + i + 1u < kSize; ++j) {
            if (output[j] > output[j + 1u]) {
                const float temp = output[j];
                output[j] = output[j + 1u];
                output[j + 1u] = temp;
            }
        }
    }

    for (uint32_t i = 0; i < kSize; ++i) {
        if (std::fabs(output[i] - kExpected[i]) > kTolerance) {
            std::puts("FAILED");
            return 1;
        }
    }

    std::printf("sort_bubble checksum: %.3f\n", checksum(output));
    std::puts("PASSED");
    return 0;
}
