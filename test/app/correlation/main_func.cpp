
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>

namespace {

constexpr uint32_t kXSize = 128;
constexpr uint32_t kYSize = 16;
constexpr uint32_t kOutputSize = kXSize - kYSize + 1;
constexpr float kInputPi = 3.14159f;
constexpr float kTolerance = 1e-4f;

void initialize_x(std::array<float, kXSize> &x) {
    for (uint32_t i = 0; i < kXSize; ++i) {
        x[i] = std::sin(2.0f * kInputPi * static_cast<float>(i) / 16.0f);
    }
}

void initialize_y(std::array<float, kYSize> &y) {
    for (uint32_t i = 0; i < kYSize; ++i) {
        y[i] = std::cos(2.0f * kInputPi * static_cast<float>(i) / 8.0f);
    }
}

void correlation_ref(const float *x, const float *y, float *output,
                     uint32_t x_size, uint32_t y_size) {
    const uint32_t output_size = x_size - y_size + 1u;
    for (uint32_t lag = 0; lag < output_size; ++lag) {
        float sum = 0.0f;
        for (uint32_t i = 0; i < y_size; ++i) {
            sum += x[lag + i] * y[i];
        }
        output[lag] = sum;
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
void correlation_kernel(const float *x, const float *y, float *output,
                        uint32_t x_size, uint32_t y_size) {
    const uint32_t output_size = x_size - y_size + 1u;
    for (uint32_t lag = 0; lag < output_size; ++lag) {
        float sum = 0.0f;
        for (uint32_t i = 0; i < y_size; ++i) {
            sum += x[lag + i] * y[i];
        }
        output[lag] = sum;
    }
}

int main() {
    std::array<float, kXSize> x = {};
    std::array<float, kYSize> y = {};
    std::array<float, kOutputSize> reference = {};
    std::array<float, kOutputSize> candidate = {};

    initialize_x(x);
    initialize_y(y);
    correlation_ref(x.data(), y.data(), reference.data(), kXSize, kYSize);
    correlation_kernel(x.data(), y.data(), candidate.data(), kXSize, kYSize);

    for (uint32_t i = 0; i < kOutputSize; ++i) {
        if (std::fabs(reference[i] - candidate[i]) > kTolerance) {
            std::puts("FAILED");
            return 1;
        }
    }

    std::printf("correlation checksum: %.3f\n", checksum(candidate));
    std::puts("PASSED");
    return 0;
}
