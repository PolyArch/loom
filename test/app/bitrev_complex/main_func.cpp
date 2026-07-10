
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>

namespace {

constexpr uint32_t kSize = 128;
constexpr float kTolerance = 1e-6f;

uint32_t bit_reverse_index(uint32_t index, uint32_t size) {
    uint32_t reversed = 0;
    uint32_t current = index;
    uint32_t mask = size >> 1;

    while (mask > 0) {
        reversed = (reversed << 1) | (current & 1u);
        current >>= 1;
        mask >>= 1;
    }
    return reversed;
}

void initialize_input(std::array<float, kSize> &real,
                      std::array<float, kSize> &imag) {
    for (uint32_t i = 0; i < kSize; ++i) {
        real[i] = static_cast<float>(i);
        imag[i] = static_cast<float>(kSize - i);
    }
}

void bitrev_complex_ref(const float *input_real, const float *input_imag,
                        float *output_real, float *output_imag, uint32_t size) {
    for (uint32_t i = 0; i < size; ++i) {
        const uint32_t j = bit_reverse_index(i, size);
        output_real[j] = input_real[i];
        output_imag[j] = input_imag[i];
    }
}

float checksum(const std::array<float, kSize> &real,
               const std::array<float, kSize> &imag) {
    float sum = 0.0f;
    for (uint32_t i = 0; i < kSize; ++i) {
        const float weight = static_cast<float>(i + 1u);
        sum += weight * real[i] + (weight + 0.5f) * imag[i];
    }
    return sum;
}

} // namespace

extern "C" __attribute__((noinline))
void bitrev_complex_kernel(const float *input_real, const float *input_imag,
                           float *output_real, float *output_imag, uint32_t size) {
    for (uint32_t i = 0; i < size; ++i) {
        uint32_t j = 0;
        uint32_t k = i;
        uint32_t m = size >> 1;

        while (m > 0) {
            j = (j << 1) | (k & 1u);
            k >>= 1;
            m >>= 1;
        }

        output_real[j] = input_real[i];
        output_imag[j] = input_imag[i];
    }
}

int main() {
    std::array<float, kSize> input_real = {};
    std::array<float, kSize> input_imag = {};
    std::array<float, kSize> reference_real = {};
    std::array<float, kSize> reference_imag = {};
    std::array<float, kSize> candidate_real = {};
    std::array<float, kSize> candidate_imag = {};

    initialize_input(input_real, input_imag);
    bitrev_complex_ref(input_real.data(), input_imag.data(), reference_real.data(),
                       reference_imag.data(), kSize);
    bitrev_complex_kernel(input_real.data(), input_imag.data(), candidate_real.data(),
                          candidate_imag.data(), kSize);

    for (uint32_t i = 0; i < kSize; ++i) {
        if (std::fabs(reference_real[i] - candidate_real[i]) > kTolerance ||
            std::fabs(reference_imag[i] - candidate_imag[i]) > kTolerance) {
            std::puts("FAILED");
            return 1;
        }
    }

    std::printf("bitrev_complex checksum: %.3f\n",
                checksum(candidate_real, candidate_imag));
    std::puts("PASSED");
    return 0;
}
