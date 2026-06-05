// 3x3 matrix-multiply inline variant migrated from the legacy app corpus.

#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>

namespace {

constexpr uint32_t kMatrices = 8;
constexpr uint32_t kElems = kMatrices * 9u;
constexpr double kExpectedChecksum = 909.4296875;
constexpr double kTolerance = 1e-3;

double checksum(const std::array<float, kElems> &values) {
    double sum = 0.0;
    for (uint32_t i = 0; i < kElems; ++i) {
        sum += static_cast<double>(i + 1u) * values[i];
    }
    return sum;
}

} // namespace

int main() {
    std::array<float, kElems> a = {};
    std::array<float, kElems> b = {};
    std::array<float, kElems> c = {};

    for (uint32_t i = 0; i < kElems; ++i) {
        a[i] = 1.0f + static_cast<float>((i * 7u) % 23u) * 0.125f;
        b[i] = -0.5f + static_cast<float>((i * 5u) % 19u) * 0.0625f;
    }

    for (uint32_t n = 0; n < kMatrices; ++n) {
        const float *mat_a = &a[n * 9u];
        const float *mat_b = &b[n * 9u];
        float *mat_c = &c[n * 9u];
        for (uint32_t row = 0; row < 3u; ++row) {
            for (uint32_t col = 0; col < 3u; ++col) {
                float sum = 0.0f;
                for (uint32_t k = 0; k < 3u; ++k) {
                    sum += mat_a[row * 3u + k] * mat_b[k * 3u + col];
                }
                mat_c[row * 3u + col] = sum;
            }
        }
    }

    const double actual = checksum(c);
    if (std::fabs(actual - kExpectedChecksum) > kTolerance) {
        std::puts("FAILED");
        return 1;
    }

    std::printf("mat3x3_mult checksum: %.3f\n", actual);
    std::puts("PASSED");
    return 0;
}
