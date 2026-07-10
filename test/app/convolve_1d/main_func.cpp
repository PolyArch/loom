
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>

namespace {

constexpr uint32_t kInputSize = 128;
constexpr uint32_t kKernelSize = 7;
constexpr uint32_t kOutputSize = kInputSize - kKernelSize + 1;
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

void convolve_1d_ref(const float *input, const float *kernel, float *output,
                     uint32_t input_size, uint32_t kernel_size) {
    const uint32_t output_size = input_size - kernel_size + 1u;
    for (uint32_t n = 0; n < output_size; ++n) {
        float sum = 0.0f;
        for (uint32_t k = 0; k < kernel_size; ++k) {
            sum += input[n + k] * kernel[k];
        }
        output[n] = sum;
    }
}

float checksum(const std::array<float, kOutputSize> &values) {
    float sum = 0.0f;
    for (uint32_t i = 0; i < kOutputSize; ++i) {
        sum += static_cast<float>(i + 1u) * values[i];
    }
    return sum;
}

} // namespace

extern "C" __attribute__((noinline))
void convolve_1d_kernel(const float *input, const float *kernel, float *output,
                        uint32_t input_size, uint32_t kernel_size) {
    const uint32_t output_size = input_size - kernel_size + 1u;
    for (uint32_t n = 0; n < output_size; ++n) {
        float sum = 0.0f;
        for (uint32_t k = 0; k < kernel_size; ++k) {
            sum += input[n + k] * kernel[k];
        }
        output[n] = sum;
    }
}

int main() {
    std::array<float, kInputSize> input = {};
    std::array<float, kKernelSize> kernel = {};
    std::array<float, kOutputSize> reference = {};
    std::array<float, kOutputSize> candidate = {};

    initialize_input(input);
    initialize_kernel(kernel);
    convolve_1d_ref(input.data(), kernel.data(), reference.data(), kInputSize, kKernelSize);
    convolve_1d_kernel(input.data(), kernel.data(), candidate.data(), kInputSize, kKernelSize);

    for (uint32_t i = 0; i < kOutputSize; ++i) {
        if (std::fabs(reference[i] - candidate[i]) > kTolerance) {
            std::puts("FAILED");
            return 1;
        }
    }

    std::printf("convolve_1d checksum: %.3f\n", checksum(candidate));
    std::puts("PASSED");
    return 0;
}
