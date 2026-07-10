
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>

namespace {

constexpr uint32_t kSize = 1024;
constexpr float kTolerance = 1e-3f;

void initialize_input(std::array<float, kSize> &input) {
    for (uint32_t i = 0; i < kSize; ++i) {
        input[i] = static_cast<float>(i % 10u) + 1.0f;
    }
}

void cumsum_ref(const float *input, float *output, uint32_t size) {
    float sum = 0.0f;
    for (uint32_t i = 0; i < size; ++i) {
        sum += input[i];
        output[i] = sum;
    }
}

double checksum(const std::array<float, kSize> &values) {
    double sum = 0.0;
    for (uint32_t i = 0; i < kSize; ++i) {
        sum += static_cast<double>(i + 1u) * static_cast<double>(values[i]);
    }
    return sum;
}

} // namespace

extern "C" __attribute__((noinline))
void cumsum_kernel(const float *input, float *output, uint32_t size) {
    float sum = 0.0f;
    for (uint32_t i = 0; i < size; ++i) {
        sum += input[i];
        output[i] = sum;
    }
}

int main() {
    std::array<float, kSize> input = {};
    std::array<float, kSize> reference = {};
    std::array<float, kSize> candidate = {};

    initialize_input(input);
    cumsum_ref(input.data(), reference.data(), kSize);
    cumsum_kernel(input.data(), candidate.data(), kSize);

    for (uint32_t i = 0; i < kSize; ++i) {
        if (std::fabs(reference[i] - candidate[i]) > kTolerance) {
            std::puts("FAILED");
            return 1;
        }
    }

    std::printf("cumsum checksum: %.3f\n", checksum(candidate));
    std::puts("PASSED");
    return 0;
}
