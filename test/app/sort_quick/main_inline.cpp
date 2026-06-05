// Iterative quick-sort inline variant migrated from the legacy app corpus.

#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>

namespace {

constexpr uint32_t kSize = 12;
constexpr uint32_t kStackCapacity = 64;
constexpr float kTolerance = 1e-5f;
constexpr std::array<float, kSize> kInput = {
    9.0f, 1.5f, 4.0f, 4.0f, -2.0f, 7.25f,
    0.0f, 3.5f, 8.0f, -1.0f, 2.0f, 6.0f};
constexpr std::array<float, kSize> kExpected = {
    -2.0f, -1.0f, 0.0f, 1.5f, 2.0f, 3.5f,
    4.0f, 4.0f, 6.0f, 7.25f, 8.0f, 9.0f};

double checksum(const std::array<float, kSize> &values) {
    double sum = 0.0;
    for (uint32_t i = 0; i < kSize; ++i) {
        sum += static_cast<double>(i + 1u) * values[i];
    }
    return sum;
}

} // namespace

int main() {
    std::array<float, kSize> output = {};
    for (uint32_t i = 0; i < kSize; ++i) {
        output[i] = kInput[i];
    }

    uint32_t stack[kStackCapacity] = {};
    int32_t top = -1;
    stack[++top] = 0u;
    stack[++top] = kSize - 1u;

    while (top >= 0) {
        const uint32_t high = stack[top--];
        const uint32_t low = stack[top--];
        if (low >= high) {
            continue;
        }

        const float pivot = output[high];
        uint32_t i = low;
        for (uint32_t j = low; j < high; ++j) {
            if (output[j] <= pivot) {
                const float temp = output[i];
                output[i] = output[j];
                output[j] = temp;
                ++i;
            }
        }

        const float temp = output[i];
        output[i] = output[high];
        output[high] = temp;

        if (i > low) {
            stack[++top] = low;
            stack[++top] = i - 1u;
        }
        if (i < high) {
            stack[++top] = i + 1u;
            stack[++top] = high;
        }
    }

    for (uint32_t i = 0; i < kSize; ++i) {
        if (std::fabs(output[i] - kExpected[i]) > kTolerance) {
            std::puts("FAILED");
            return 1;
        }
    }

    std::printf("sort_quick checksum: %.3f\n", checksum(output));
    std::puts("PASSED");
    return 0;
}
