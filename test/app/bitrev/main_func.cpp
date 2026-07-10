
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

void initialize_input(std::array<float, kSize> &input) {
    for (uint32_t i = 0; i < kSize; ++i) {
        input[i] = static_cast<float>(i);
    }
}

void bitrev_ref(const float *input, float *output, uint32_t size) {
    for (uint32_t i = 0; i < size; ++i) {
        output[bit_reverse_index(i, size)] = input[i];
    }
}

float checksum(const std::array<float, kSize> &values) {
    float sum = 0.0f;
    for (uint32_t i = 0; i < kSize; ++i) {
        sum += static_cast<float>(i + 1u) * values[i];
    }
    return sum;
}

} // namespace

extern "C" __attribute__((noinline))
void bitrev_kernel(const float *input, float *output, uint32_t size) {
    for (uint32_t i = 0; i < size; ++i) {
        uint32_t j = 0;
        uint32_t k = i;
        uint32_t m = size >> 1;

        while (m > 0) {
            j = (j << 1) | (k & 1u);
            k >>= 1;
            m >>= 1;
        }

        output[j] = input[i];
    }
}

int main() {
    std::array<float, kSize> input = {};
    std::array<float, kSize> reference = {};
    std::array<float, kSize> candidate = {};

    initialize_input(input);
    bitrev_ref(input.data(), reference.data(), kSize);
    bitrev_kernel(input.data(), candidate.data(), kSize);

    for (uint32_t i = 0; i < kSize; ++i) {
        if (std::fabs(reference[i] - candidate[i]) > kTolerance) {
            std::puts("FAILED");
            return 1;
        }
    }

    std::printf("bitrev checksum: %.3f\n", checksum(candidate));
    std::puts("PASSED");
    return 0;
}
