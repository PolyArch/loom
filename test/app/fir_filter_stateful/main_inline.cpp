
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>

namespace {

constexpr uint32_t kInputSize = 16;
constexpr uint32_t kTaps = 5;
constexpr float kTolerance = 1e-5f;
constexpr std::array<float, kInputSize> kExpectedOutput = {
    1.25f, 3.0f, 2.25f, 1.0f, 2.25f, 3.0f, 3.75f, 4.5f,
    5.25f, 6.0f, 6.75f, 7.5f, 8.25f, 9.0f, 9.75f, 10.5f};
constexpr std::array<float, kTaps - 1u> kExpectedState = {
    16.0f, 15.0f, 14.0f, 13.0f};

double checksum(const std::array<float, kInputSize> &output,
                const std::array<float, kTaps - 1u> &state) {
    double sum = 0.0;
    for (uint32_t i = 0; i < kInputSize; ++i) {
        sum += static_cast<double>(i + 1u) * output[i];
    }
    for (uint32_t i = 0; i < kTaps - 1u; ++i) {
        sum += static_cast<double>(i + 1u) * 100.0 * state[i];
    }
    return sum;
}

} // namespace

int main() {
    const std::array<float, kInputSize> input = {
        1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f,
        9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f};
    const std::array<float, kTaps> coeffs = {
        0.25f, -0.125f, 0.5f, 0.375f, -0.25f};
    const std::array<float, kTaps - 1u> input_state = {4.0f, 3.0f, 2.0f, 1.0f};
    std::array<float, kInputSize> output = {};
    std::array<float, kTaps - 1u> output_state = {};
    std::array<float, kTaps - 1u> current_state = {};

    for (uint32_t k = 0; k < kTaps - 1u; ++k) {
        current_state[k] = input_state[k];
    }

    for (uint32_t n = 0; n < kInputSize; ++n) {
        float sum = coeffs[0] * input[n];
        for (uint32_t k = 1; k < kTaps; ++k) {
            sum += coeffs[k] * current_state[kTaps - 1u - k];
        }
        output[n] = sum;

        for (uint32_t k = kTaps - 1u; k > 1u; --k) {
            current_state[k - 1u] = current_state[k - 2u];
        }
        current_state[0] = input[n];
    }

    for (uint32_t k = 0; k < kTaps - 1u; ++k) {
        output_state[k] = current_state[k];
    }

    for (uint32_t i = 0; i < kInputSize; ++i) {
        if (std::fabs(output[i] - kExpectedOutput[i]) > kTolerance) {
            std::puts("FAILED");
            return 1;
        }
    }
    for (uint32_t i = 0; i < kTaps - 1u; ++i) {
        if (std::fabs(output_state[i] - kExpectedState[i]) > kTolerance) {
            std::puts("FAILED");
            return 1;
        }
    }

    std::printf("fir_filter_stateful checksum: %.3f\n",
                checksum(output, output_state));
    std::puts("PASSED");
    return 0;
}
