
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>

namespace {

constexpr uint32_t kDepth = 4;
constexpr uint32_t kRows = 4;
constexpr uint32_t kCols = 4;
constexpr uint32_t kPlane = kRows * kCols;
constexpr uint32_t kSize = kDepth * kPlane;
constexpr uint32_t kInteriorCount = (kDepth - 2u) * (kRows - 2u) * (kCols - 2u);
constexpr float kTolerance = 1.0e-5f;
constexpr std::array<float, kSize> kInput = {
    3.929384f, -4.277213f, -5.462971f, 1.026295f,
    4.389380f, -1.537871f, 9.615284f, 3.696595f,
    -0.3813620f, -2.157650f, -3.136440f, 4.580994f,
    -1.228555f, -8.806442f, -2.039115f, 4.759908f,
    -6.350165f, -6.490965f, 0.6310275f, 0.6365517f,
    2.688019f, 6.988636f, 4.489107f, 2.220470f,
    4.448868f, -3.540822f, -2.764227f, -5.434735f,
    -4.125719f, 2.619523f, -8.157901f, -1.325976f,
    -1.382745f, -0.1262980f, -1.483394f, -3.754776f,
    -1.472974f, 7.867783f, 8.883201f, 0.03673352f,
    2.479059f, -7.687632f, -3.654290f, -1.703476f,
    7.326183f, -4.990893f, -0.3393147f, 9.711196f,
    0.3897024f, 2.257890f, -7.587427f, 6.526816f,
    2.061203f, 0.9013602f, -3.144723f, -3.917584f,
    -1.659556f, 3.626015f, 7.509137f, 0.2084468f,
    3.386276f, 1.718731f, 2.498070f, 3.493781f,
};

void jacobi_stencil_7pt_ref(const float *input, float *output,
                            uint32_t depth, uint32_t rows, uint32_t cols) {
    const uint32_t plane = rows * cols;
    for (uint32_t idx = 0; idx < depth * plane; ++idx) {
        output[idx] = input[idx];
    }
    for (uint32_t z = 1; z < depth - 1; ++z) {
        for (uint32_t r = 1; r < rows - 1; ++r) {
            for (uint32_t c = 1; c < cols - 1; ++c) {
                const uint32_t idx = z * plane + r * cols + c;
                const float front = input[idx - plane];
                const float back = input[idx + plane];
                const float north = input[idx - cols];
                const float south = input[idx + cols];
                const float west = input[idx - 1u];
                const float east = input[idx + 1u];
                output[idx] = (front + back + north + south + west + east) * (1.0f / 6.0f);
            }
        }
    }
}

void scatter_interior(const float *interior, float *output) {
    uint32_t out = 0;
    for (uint32_t z = 1; z < kDepth - 1u; ++z) {
        for (uint32_t r = 1; r < kRows - 1u; ++r) {
            for (uint32_t c = 1; c < kCols - 1u; ++c) {
                output[z * kPlane + r * kCols + c] = interior[out++];
            }
        }
    }
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
void jacobi_stencil_7pt_kernel(const float *input, float *interior,
                               uint32_t count) {
    for (uint32_t inner = 0; inner < count; ++inner) {
        const uint32_t z = (inner >> 2) + 1u;
        const uint32_t rem = inner & 3u;
        const uint32_t r = (rem >> 1) + 1u;
        const uint32_t c = (rem & 1u) + 1u;
        const uint32_t idx = z * kPlane + r * kCols + c;
        const float front = input[idx - kPlane];
        const float back = input[idx + kPlane];
        const float north = input[idx - kCols];
        const float south = input[idx + kCols];
        const float west = input[idx - 1u];
        const float east = input[idx + 1u];
        interior[inner] =
            (front + back + north + south + west + east) * (1.0f / 6.0f);
    }
}

int main() {
    std::array<float, kSize> expected = {};
    std::array<float, kSize> candidate = kInput;
    std::array<float, kInteriorCount> interior = {};

    jacobi_stencil_7pt_ref(kInput.data(), expected.data(), kDepth, kRows, kCols);
    jacobi_stencil_7pt_kernel(kInput.data(), interior.data(), kInteriorCount);
    scatter_interior(interior.data(), candidate.data());

    for (uint32_t i = 0; i < kSize; ++i) {
        if (std::fabs(expected[i] - candidate[i]) > kTolerance) {
            std::puts("FAILED");
            return 1;
        }
    }

    std::printf("jacobi_stencil_7pt checksum: %.6f\n", checksum(candidate));
    std::puts("PASSED");
    return 0;
}
