
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>

namespace {

constexpr uint32_t kInputSize = 128;
constexpr uint32_t kKernelSize = 7;
constexpr float kInputPi = 3.14159f;
constexpr float kTolerance = 1e-5f;

void initialize_input(std::array<float, kInputSize> &input) {
    for (uint32_t i = 0; i < kInputSize; ++i) {
        input[i] = std::sin(2.0f * kInputPi * static_cast<float>(i) / 32.0f);
    }
}

void initialize_kernel(std::array<float, kKernelSize> &kernel) {
    for (uint32_t i = 0; i < kKernelSize; ++i) {
        kernel[i] = 1.0f / static_cast<float>(kKernelSize);
    }
}

void convolve_1d_same_ref(const float *input, const float *kernel, float *output,
                          uint32_t input_size, uint32_t kernel_size) {
    const int32_t pad = static_cast<int32_t>(kernel_size / 2u);
    for (uint32_t n = 0; n < input_size; ++n) {
        float sum = 0.0f;
        for (uint32_t k = 0; k < kernel_size; ++k) {
            const int32_t idx = static_cast<int32_t>(n) - pad + static_cast<int32_t>(k);
            if (idx >= 0 && idx < static_cast<int32_t>(input_size)) {
                sum += input[idx] * kernel[k];
            }
        }
        output[n] = sum;
    }
}

float checksum(const std::array<float, kInputSize> &values) {
    float sum = 0.0f;
    for (uint32_t i = 0; i < kInputSize; ++i) {
        sum += static_cast<float>(i + 1u) * values[i];
    }
    return sum;
}

} // namespace

int main() {
    std::array<float, kInputSize> input = {};
    std::array<float, kKernelSize> kernel = {};
    std::array<float, kInputSize> reference = {};
    std::array<float, kInputSize> candidate = {};

    initialize_input(input);
    initialize_kernel(kernel);
    convolve_1d_same_ref(input.data(), kernel.data(), reference.data(),
                         kInputSize, kKernelSize);

    constexpr int32_t kPad = static_cast<int32_t>(kKernelSize / 2u);
    for (uint32_t n = 0; n < kInputSize; ++n) {
        float sum = 0.0f;
        for (uint32_t k = 0; k < kKernelSize; ++k) {
            const int32_t idx = static_cast<int32_t>(n) - kPad + static_cast<int32_t>(k);
            if (idx >= 0 && idx < static_cast<int32_t>(kInputSize)) {
                sum += input[idx] * kernel[k];
            }
        }
        candidate[n] = sum;
    }

    for (uint32_t i = 0; i < kInputSize; ++i) {
        if (std::fabs(reference[i] - candidate[i]) > kTolerance) {
            std::puts("FAILED");
            return 1;
        }
    }

    std::printf("convolve_1d_same checksum: %.3f\n", checksum(candidate));
    std::puts("PASSED");
    return 0;
}
