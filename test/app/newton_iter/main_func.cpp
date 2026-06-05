// Newton iteration function variant migrated from the legacy app corpus.

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

void newton_iter_ref(const float *input_x, const float *input_f,
                     const float *input_df, float *output_x, uint32_t size) {
    for (uint32_t i = 0; i < size; ++i) {
        output_x[i] = input_x[i] - input_f[i] / input_df[i];
    }
}

extern "C" __attribute__((noinline))
void newton_iter_kernel(const float *input_x, const float *input_f,
                        const float *input_df, float *output_x,
                        uint32_t size) {
    for (uint32_t i = 0; i < size; ++i) {
        output_x[i] = input_x[i] - input_f[i] / input_df[i];
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

    newton_iter_ref(input_x.data(), input_f.data(), input_df.data(),
                    expected.data(), kSize);
    newton_iter_kernel(input_x.data(), input_f.data(), input_df.data(),
                       candidate.data(), kSize);

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
