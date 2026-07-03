// Static-size Gauss-Seidel update migrated from the legacy app corpus.

#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>

namespace {

constexpr uint32_t kSize = 8;
constexpr float kTolerance = 1.0e-5f;
constexpr float kExpectedChecksum = 15.735688f;

float checksum(const std::array<float, kSize> &values) {
    float sum = 0.0f;
    for (uint32_t i = 0; i < kSize; ++i) {
        sum += static_cast<float>(i + 1u) * values[i];
    }
    return sum;
}

} // namespace

int main() {
    std::array<float, kSize * kSize> matrix = {};
    std::array<float, kSize> rhs = {};
    std::array<float, kSize> current = {};
    std::array<float, kSize> expected = {};
    std::array<float, kSize> actual = {};

    for (uint32_t i = 0; i < kSize; ++i) {
        for (uint32_t j = 0; j < kSize; ++j) {
            matrix[i * kSize + j] = i == j ? 10.0f : 1.0f;
        }
        rhs[i] = static_cast<float>(i + 1u);
        current[i] = 0.0f;
    }

    for (uint32_t row = 0; row < kSize; ++row) {
        float sigma = 0.0f;
        for (uint32_t col = 0; col < kSize; ++col) {
            const float value =
                col == row ? 0.0f : (col < row ? expected[col] : current[col]);
            sigma += matrix[row * kSize + col] * value;
        }
        expected[row] = (rhs[row] - sigma) / matrix[row * kSize + row];
    }

    for (uint32_t row = 0; row < kSize; ++row) {
        float sigma = 0.0f;
        for (uint32_t col = 0; col < kSize; ++col) {
            const float value =
                col == row ? 0.0f : (col < row ? actual[col] : current[col]);
            sigma += matrix[row * kSize + col] * value;
        }
        actual[row] = (rhs[row] - sigma) / matrix[row * kSize + row];
    }

    for (uint32_t i = 0; i < kSize; ++i) {
        if (std::fabs(expected[i] - actual[i]) > kTolerance) {
            std::puts("FAILED");
            return 1;
        }
    }

    const float actual_checksum = checksum(actual);
    if (std::fabs(actual_checksum - kExpectedChecksum) > kTolerance) {
        std::puts("FAILED");
        return 1;
    }

    std::printf("gauss_seidel_step checksum: %.4f\n",
                static_cast<double>(actual_checksum));
    std::puts("PASSED");
    return 0;
}
