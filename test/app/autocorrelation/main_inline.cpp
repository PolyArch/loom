
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>

namespace {

constexpr uint32_t kXSize = 128;
constexpr uint32_t kMaxLag = 32;
constexpr float kInputPi = 3.14159f;
constexpr float kTolerance = 1e-4f;

void initialize_x(std::array<float, kXSize> &x) {
    for (uint32_t i = 0; i < kXSize; ++i) {
        x[i] = std::sin(2.0f * kInputPi * static_cast<float>(i) / 16.0f);
    }
}

void autocorrelation_ref(const float *x, float *output,
                         uint32_t x_size, uint32_t max_lag) {
    for (uint32_t lag = 0; lag < max_lag; ++lag) {
        float sum = 0.0f;
        for (uint32_t i = 0; i < x_size - lag; ++i) {
            sum += x[i] * x[i + lag];
        }
        output[lag] = sum;
    }
}

float checksum(const std::array<float, kMaxLag> &values) {
    float sum = 0.0f;
    for (uint32_t i = 0; i < kMaxLag; ++i) {
        sum += static_cast<float>(i + 1u) * values[i];
    }
    return sum;
}

} // namespace

int main() {
    std::array<float, kXSize> x = {};
    std::array<float, kMaxLag> reference = {};
    std::array<float, kMaxLag> candidate = {};

    initialize_x(x);
    autocorrelation_ref(x.data(), reference.data(), kXSize, kMaxLag);

    for (uint32_t lag = 0; lag < kMaxLag; ++lag) {
        float sum = 0.0f;
        for (uint32_t i = 0; i < kXSize - lag; ++i) {
            sum += x[i] * x[i + lag];
        }
        candidate[lag] = sum;
    }

    for (uint32_t i = 0; i < kMaxLag; ++i) {
        if (std::fabs(reference[i] - candidate[i]) > kTolerance) {
            std::puts("FAILED");
            return 1;
        }
    }

    std::printf("autocorrelation checksum: %.3f\n", checksum(candidate));
    std::puts("PASSED");
    return 0;
}
