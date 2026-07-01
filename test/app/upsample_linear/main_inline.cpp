// Linear-interpolation upsampling inline variant migrated from the legacy app corpus.

#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>

namespace {

constexpr uint32_t kInputSize = 4;
constexpr uint32_t kFactor = 4;
constexpr uint32_t kOutputSize = kInputSize * kFactor;
constexpr float kTolerance = 1.0e-5f;
constexpr uint32_t kExpectedChecksum = 102405u;

void upsample_linear_ref(const float *input, float *output,
                         uint32_t input_size, uint32_t factor) {
    if (input_size == 0) {
        return;
    }
    for (uint32_t i = 0; i < input_size - 1; ++i) {
        output[i * factor] = input[i];
        for (uint32_t j = 1; j < factor; ++j) {
            const float alpha =
                static_cast<float>(j) / static_cast<float>(factor);
            output[i * factor + j] =
                (1.0f - alpha) * input[i] + alpha * input[i + 1];
        }
    }
    output[(input_size - 1) * factor] = input[input_size - 1];
    for (uint32_t j = 1; j < factor; ++j) {
        output[(input_size - 1) * factor + j] = input[input_size - 1];
    }
}

void initialize_input(std::array<float, kInputSize> &input) {
    for (uint32_t i = 0; i < kInputSize; ++i) {
        input[i] = std::sinf(2.0f * 3.14159f * static_cast<float>(i) / 16.0f);
    }
}

uint32_t weighted_milli_checksum(const std::array<float, kOutputSize> &values) {
    uint32_t sum = 0;
    for (uint32_t i = 0; i < kOutputSize; ++i) {
        sum += (i + 1u) * static_cast<uint32_t>(values[i] * 1000.0f + 0.5f);
    }
    return sum;
}

} // namespace

int main() {
    std::array<float, kInputSize> input = {};
    std::array<float, kOutputSize> expected = {};
    std::array<float, kOutputSize> candidate = {};

    initialize_input(input);
    upsample_linear_ref(input.data(), expected.data(), kInputSize, kFactor);

    constexpr uint32_t factor = kFactor;
    for (uint32_t out = 0; out < kInputSize * factor; ++out) {
        const uint32_t base = out / factor;
        const uint32_t offset = out - base * factor;
        if (base >= kInputSize - 1) {
            candidate[out] = input[kInputSize - 1];
        } else if (offset == 0) {
            candidate[out] = input[base];
        } else {
            const float alpha =
                static_cast<float>(offset) / static_cast<float>(factor);
            candidate[out] = (1.0f - alpha) * input[base] +
                             alpha * input[base + 1];
        }
    }

    for (uint32_t i = 0; i < kOutputSize; ++i) {
        if (std::fabs(expected[i] - candidate[i]) > kTolerance) {
            std::puts("FAILED");
            return 1;
        }
    }

    const uint32_t checksum = weighted_milli_checksum(candidate);
    std::printf("upsample_linear checksum: %u\n", checksum);
    if (checksum != kExpectedChecksum) {
        std::puts("FAILED");
        return 1;
    }
    std::puts("PASSED");
    return 0;
}
