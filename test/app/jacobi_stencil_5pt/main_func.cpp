
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>

namespace {

constexpr uint32_t kRows = 4;
constexpr uint32_t kCols = 4;
constexpr uint32_t kSize = kRows * kCols;
constexpr uint32_t kInteriorCount = (kRows - 2u) * (kCols - 2u);
constexpr float kTolerance = 1.0e-5f;
constexpr std::array<float, kSize> kInput = {
    1.0f, 2.0f, 4.0f, 8.0f,
    16.0f, 32.0f, 64.0f, 128.0f,
    3.0f, 9.0f, 27.0f, 81.0f,
    5.0f, 25.0f, 125.0f, 625.0f,
};

void jacobi_stencil_5pt_ref(const float *input, float *output,
                            uint32_t rows, uint32_t cols) {
    for (uint32_t idx = 0; idx < rows * cols; ++idx) {
        output[idx] = input[idx];
    }
    for (uint32_t r = 1; r < rows - 1; ++r) {
        for (uint32_t c = 1; c < cols - 1; ++c) {
            const uint32_t idx = r * cols + c;
            const float north = input[idx - cols];
            const float south = input[idx + cols];
            const float west = input[idx - 1u];
            const float east = input[idx + 1u];
            output[idx] = (north + south + west + east) * 0.25f;
        }
    }
}

void scatter_interior(const float *interior, float *output) {
    output[5] = interior[0];
    output[6] = interior[1];
    output[9] = interior[2];
    output[10] = interior[3];
}

double checksum(const std::array<float, kSize> &values) {
    double sum = 0.0;
    for (uint32_t i = 0; i < kSize; ++i) {
        sum += static_cast<double>(i + 1u) * values[i];
    }
    return sum;
}

} // namespace

extern "C" __attribute__((noinline))
void jacobi_stencil_5pt_kernel(const float *input, float *interior,
                               uint32_t count) {
    for (uint32_t inner = 0; inner < count; ++inner) {
        const uint32_t r = (inner >> 1) + 1u;
        const uint32_t c = (inner & 1u) + 1u;
        const uint32_t idx = r * kCols + c;
        const float north = input[idx - kCols];
        const float south = input[idx + kCols];
        const float west = input[idx - 1u];
        const float east = input[idx + 1u];
        interior[inner] = (north + south + west + east) * 0.25f;
    }
}

int main() {
    std::array<float, kSize> expected = {};
    std::array<float, kSize> candidate = kInput;
    std::array<float, kInteriorCount> interior = {};

    jacobi_stencil_5pt_ref(kInput.data(), expected.data(), kRows, kCols);
    jacobi_stencil_5pt_kernel(kInput.data(), interior.data(), kInteriorCount);
    scatter_interior(interior.data(), candidate.data());

    for (uint32_t i = 0; i < kSize; ++i) {
        if (std::fabs(expected[i] - candidate[i]) > kTolerance) {
            std::puts("FAILED");
            return 1;
        }
    }

    std::printf("jacobi_stencil_5pt checksum: %.6f\n", checksum(candidate));
    std::puts("PASSED");
    return 0;
}
