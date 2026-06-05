// Newton iteration inline variant migrated from the legacy app corpus.

#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>

namespace {

constexpr uint32_t kSize = 64;
constexpr float kTolerance = 1.0e-5f;

void initialize_inputs(std::array<float, kSize> &input_x,
                       std::array<float, kSize> &input_f,
                       std::array<float, kSize> &input_df) {
    for (uint32_t i = 0; i < kSize; ++i) {
        float c = static_cast<float>(i + 1);
        input_x[i] = c;
        input_f[i] = input_x[i] * input_x[i] - c;
        input_df[i] = 2.0f * input_x[i];
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
    std::array<float, kSize> input_x = {};
    std::array<float, kSize> input_f = {};
    std::array<float, kSize> input_df = {};
    std::array<float, kSize> expected = {};
    std::array<float, kSize> candidate = {};
    initialize_inputs(input_x, input_f, input_df);

    for (uint32_t i = 0; i < kSize; ++i) {
        expected[i] = input_x[i] - input_f[i] / input_df[i];
    }
    for (uint32_t i = 0; i < kSize; ++i) {
        candidate[i] = input_x[i] - input_f[i] / input_df[i];
    }

    for (uint32_t i = 0; i < kSize; ++i) {
        if (std::fabs(expected[i] - candidate[i]) > kTolerance) {
            std::puts("FAILED");
            return 1;
        }
    }

    std::printf("newton_iter checksum: %.6f\n", checksum(candidate));
    std::puts("PASSED");
    return 0;
}
