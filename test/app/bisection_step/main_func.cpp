// Bisection-step function variant migrated from the legacy app corpus.

#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>

namespace {

constexpr uint32_t kSize = 64;
constexpr float kTolerance = 1.0e-5f;

void initialize_inputs(std::array<float, kSize> &input_a,
                       std::array<float, kSize> &input_b,
                       std::array<float, kSize> &input_fa,
                       std::array<float, kSize> &input_fc) {
    for (uint32_t i = 0; i < kSize; ++i) {
        if (i % 3 == 0) {
            input_a[i] = 0.0f;
            input_b[i] = 2.0f;
            input_fa[i] = -1.0f;
            input_fc[i] = 0.25f;
        } else if (i % 3 == 1) {
            input_a[i] = 1.0f;
            input_b[i] = 5.0f;
            input_fa[i] = -2.0f;
            input_fc[i] = -0.5f;
        } else {
            input_a[i] = 2.0f;
            input_b[i] = 6.0f;
            input_fa[i] = 4.0f;
            input_fc[i] = 5.0f;
        }
    }
}

void bisection_step_ref(const float *input_a, const float *input_b,
                        const float *input_fa, const float *input_fc,
                        float *output_a, float *output_b, uint32_t size) {
    for (uint32_t i = 0; i < size; ++i) {
        float c = (input_a[i] + input_b[i]) * 0.5f;
        if (input_fa[i] * input_fc[i] < 0.0f) {
            output_a[i] = input_a[i];
            output_b[i] = c;
        } else {
            output_a[i] = c;
            output_b[i] = input_b[i];
        }
    }
}

extern "C" __attribute__((noinline))
void bisection_step_kernel(const float *input_a, const float *input_b,
                           const float *input_fa, const float *input_fc,
                           float *output_a, float *output_b, uint32_t size) {
    for (uint32_t i = 0; i < size; ++i) {
        float c = (input_a[i] + input_b[i]) * 0.5f;
        if (input_fa[i] * input_fc[i] < 0.0f) {
            output_a[i] = input_a[i];
            output_b[i] = c;
        } else {
            output_a[i] = c;
            output_b[i] = input_b[i];
        }
    }
}

float checksum(const std::array<float, kSize> &a,
               const std::array<float, kSize> &b) {
    float sum = 0.0f;
    for (uint32_t i = 0; i < kSize; ++i) {
        sum += a[i] + b[i];
    }
    return sum;
}

} // namespace

int main() {
    std::array<float, kSize> input_a = {};
    std::array<float, kSize> input_b = {};
    std::array<float, kSize> input_fa = {};
    std::array<float, kSize> input_fc = {};
    std::array<float, kSize> expected_a = {};
    std::array<float, kSize> expected_b = {};
    std::array<float, kSize> candidate_a = {};
    std::array<float, kSize> candidate_b = {};
    initialize_inputs(input_a, input_b, input_fa, input_fc);

    bisection_step_ref(input_a.data(), input_b.data(), input_fa.data(),
                       input_fc.data(), expected_a.data(), expected_b.data(),
                       kSize);
    bisection_step_kernel(input_a.data(), input_b.data(), input_fa.data(),
                          input_fc.data(), candidate_a.data(),
                          candidate_b.data(), kSize);

    for (uint32_t i = 0; i < kSize; ++i) {
        if (std::fabs(expected_a[i] - candidate_a[i]) > kTolerance ||
            std::fabs(expected_b[i] - candidate_b[i]) > kTolerance) {
            std::puts("FAILED");
            return 1;
        }
    }

    std::printf("bisection_step checksum: %.6f\n",
                checksum(candidate_a, candidate_b));
    std::puts("PASSED");
    return 0;
}
