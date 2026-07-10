
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>

namespace {

constexpr uint32_t kSize = 64;
constexpr float kStep = 0.1f;
constexpr float kTolerance = 1.0e-5f;

void initialize_inputs(std::array<float, kSize> &input_y,
                       std::array<float, kSize> &input_k1,
                       std::array<float, kSize> &input_k2,
                       std::array<float, kSize> &input_k3,
                       std::array<float, kSize> &input_k4) {
    for (uint32_t i = 0; i < kSize; ++i) {
        input_y[i] = static_cast<float>(i);
        input_k1[i] = 1.0f + 0.1f * static_cast<float>(i);
        input_k2[i] = 1.1f + 0.1f * static_cast<float>(i);
        input_k3[i] = 1.2f + 0.1f * static_cast<float>(i);
        input_k4[i] = 1.3f + 0.1f * static_cast<float>(i);
    }
}

void runge_kutta_step_ref(const float *input_y, const float *input_k1,
                          const float *input_k2, const float *input_k3,
                          const float *input_k4, float *output_y, float step,
                          uint32_t size) {
    float weight = step / 6.0f;
    for (uint32_t i = 0; i < size; ++i) {
        float slope = input_k1[i] + 2.0f * input_k2[i] +
                      2.0f * input_k3[i] + input_k4[i];
        output_y[i] = input_y[i] + weight * slope;
    }
}

extern "C" __attribute__((noinline))
void runge_kutta_step_kernel(const float *input_y, const float *input_k1,
                             const float *input_k2, const float *input_k3,
                             const float *input_k4, float *output_y,
                             float step, uint32_t size) {
    float weight = step / 6.0f;
    for (uint32_t i = 0; i < size; ++i) {
        float slope = input_k1[i] + 2.0f * input_k2[i] +
                      2.0f * input_k3[i] + input_k4[i];
        output_y[i] = input_y[i] + weight * slope;
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
    std::array<float, kSize> input_y = {};
    std::array<float, kSize> input_k1 = {};
    std::array<float, kSize> input_k2 = {};
    std::array<float, kSize> input_k3 = {};
    std::array<float, kSize> input_k4 = {};
    std::array<float, kSize> expected = {};
    std::array<float, kSize> candidate = {};
    initialize_inputs(input_y, input_k1, input_k2, input_k3, input_k4);

    runge_kutta_step_ref(input_y.data(), input_k1.data(), input_k2.data(),
                         input_k3.data(), input_k4.data(), expected.data(),
                         kStep, kSize);
    runge_kutta_step_kernel(input_y.data(), input_k1.data(), input_k2.data(),
                            input_k3.data(), input_k4.data(),
                            candidate.data(), kStep, kSize);

    for (uint32_t i = 0; i < kSize; ++i) {
        if (std::fabs(expected[i] - candidate[i]) > kTolerance) {
            std::puts("FAILED");
            return 1;
        }
    }

    std::printf("runge_kutta_step checksum: %.6f\n", checksum(candidate));
    std::puts("PASSED");
    return 0;
}
