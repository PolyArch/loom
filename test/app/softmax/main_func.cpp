// Softmax function variant migrated from the legacy app corpus.

#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>

namespace {

constexpr uint32_t kSize = 128;
constexpr float kTolerance = 1.0e-5f;

void initialize_input(std::array<float, kSize> &input) {
    for (uint32_t i = 0; i < kSize; ++i) {
        input[i] = static_cast<float>(i % 20) - 10.0f;
    }
}

void softmax_ref(const float *input, float *output, uint32_t size) {
    float max_value = input[0];
    for (uint32_t i = 1; i < size; ++i) {
        if (input[i] > max_value) {
            max_value = input[i];
        }
    }

    float sum = 0.0f;
    for (uint32_t i = 0; i < size; ++i) {
        output[i] = expf(input[i] - max_value);
        sum += output[i];
    }

    for (uint32_t i = 0; i < size; ++i) {
        output[i] = output[i] / sum;
    }
}

extern "C" __attribute__((noinline))
void softmax_kernel(const float *input, float *output, uint32_t size) {
    float max_value = input[0];
    for (uint32_t i = 1; i < size; ++i) {
        if (input[i] > max_value) {
            max_value = input[i];
        }
    }

    float sum = 0.0f;
    for (uint32_t i = 0; i < size; ++i) {
        output[i] = expf(input[i] - max_value);
        sum += output[i];
    }

    for (uint32_t i = 0; i < size; ++i) {
        output[i] = output[i] / sum;
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
    std::array<float, kSize> input = {};
    std::array<float, kSize> expected = {};
    std::array<float, kSize> candidate = {};
    initialize_input(input);

    softmax_ref(input.data(), expected.data(), kSize);
    softmax_kernel(input.data(), candidate.data(), kSize);

    for (uint32_t i = 0; i < kSize; ++i) {
        if (std::fabs(expected[i] - candidate[i]) > kTolerance) {
            std::puts("FAILED");
            return 1;
        }
    }
    if (std::fabs(checksum(candidate) - 1.0f) > 1.0e-4f) {
        std::puts("FAILED");
        return 1;
    }

    std::printf("softmax checksum: %.6f\n", checksum(candidate));
    std::puts("PASSED");
    return 0;
}
