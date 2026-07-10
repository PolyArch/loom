
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>

namespace {

constexpr uint32_t kSize = 1024;
constexpr float kTolerance = 1.0e-6f;

void initialize_input(std::array<float, kSize> &input) {
    for (uint32_t i = 0; i < kSize; ++i) {
        input[i] = (static_cast<float>(i) / static_cast<float>(kSize) - 0.5f) *
                   10.0f;
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

    for (uint32_t i = 0; i < kSize; ++i) {
        expected[i] = 1.0f / (1.0f + expf(-input[i]));
    }
    for (uint32_t i = 0; i < kSize; ++i) {
        candidate[i] = 1.0f / (1.0f + expf(-input[i]));
    }

    for (uint32_t i = 0; i < kSize; ++i) {
        if (std::fabs(expected[i] - candidate[i]) > kTolerance) {
            std::puts("FAILED");
            return 1;
        }
    }

    std::printf("sigmoid checksum: %.6f\n", checksum(candidate));
    std::puts("PASSED");
    return 0;
}
