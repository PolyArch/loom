
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>

namespace {

constexpr uint32_t kSize = 8;
constexpr float kTolerance = 1.0e-5f;
constexpr float kExpectedChecksum = 15.735688f;

void initialize_a(std::array<float, kSize * kSize> &matrix) {
    for (uint32_t i = 0; i < kSize; ++i) {
        for (uint32_t j = 0; j < kSize; ++j) {
            matrix[i * kSize + j] = i == j ? 10.0f : 1.0f;
        }
    }
}

void initialize_b(std::array<float, kSize> &rhs) {
    for (uint32_t i = 0; i < kSize; ++i) {
        rhs[i] = static_cast<float>(i + 1u);
    }
}

void initialize_x(std::array<float, kSize> &current) {
    for (float &value : current) {
        value = 0.0f;
    }
}

void gauss_seidel_step_ref(const float *matrix, const float *rhs,
                           const float *current, float *updated) {
    for (uint32_t row = 0; row < kSize; ++row) {
        float sigma = 0.0f;
        for (uint32_t col = 0; col < kSize; ++col) {
            const float value =
                col == row ? 0.0f : (col < row ? updated[col] : current[col]);
            sigma += matrix[row * kSize + col] * value;
        }
        updated[row] = (rhs[row] - sigma) / matrix[row * kSize + row];
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
void gauss_seidel_step_kernel(const float *matrix, const float *rhs,
                              const float *current, float *updated) {
    for (uint32_t row = 0; row < kSize; ++row) {
        float sigma = 0.0f;
        for (uint32_t col = 0; col < kSize; ++col) {
            const float value =
                col == row ? 0.0f : (col < row ? updated[col] : current[col]);
            sigma += matrix[row * kSize + col] * value;
        }
        updated[row] = (rhs[row] - sigma) / matrix[row * kSize + row];
    }
}

int main() {
    std::array<float, kSize * kSize> matrix = {};
    std::array<float, kSize> rhs = {};
    std::array<float, kSize> current = {};
    std::array<float, kSize> expected = {};
    std::array<float, kSize> actual = {};

    initialize_a(matrix);
    initialize_b(rhs);
    initialize_x(current);
    gauss_seidel_step_ref(matrix.data(), rhs.data(), current.data(), expected.data());
    gauss_seidel_step_kernel(matrix.data(), rhs.data(), current.data(), actual.data());

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
