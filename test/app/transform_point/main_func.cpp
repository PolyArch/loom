
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>

namespace {

constexpr uint32_t kSize = 64;
constexpr uint32_t kWidth = kSize * 3;

void initialize_points(std::array<float, kWidth> &points) {
    for (uint32_t i = 0; i < kSize; ++i) {
        points[i * 3 + 0] = 1.0f + static_cast<float>(i) * 0.1f;
        points[i * 3 + 1] = 2.0f + static_cast<float>(i) * 0.2f;
        points[i * 3 + 2] = 3.0f + static_cast<float>(i) * 0.3f;
    }
}

void initialize_transform(std::array<float, 9> &matrix,
                          std::array<float, 3> &translation) {
    matrix = {2.0f, 0.0f, 0.0f,
              0.0f, 2.0f, 0.0f,
              0.0f, 0.0f, 2.0f};
    translation = {1.0f, 2.0f, 3.0f};
}

void transform_point_ref(const float *input_points, const float *input_matrix,
                         const float *input_translation, float *output_points,
                         uint32_t size) {
    const float m00 = input_matrix[0], m01 = input_matrix[1], m02 = input_matrix[2];
    const float m10 = input_matrix[3], m11 = input_matrix[4], m12 = input_matrix[5];
    const float m20 = input_matrix[6], m21 = input_matrix[7], m22 = input_matrix[8];
    const float tx = input_translation[0];
    const float ty = input_translation[1];
    const float tz = input_translation[2];

    for (uint32_t i = 0; i < size; ++i) {
        const float px = input_points[i * 3 + 0];
        const float py = input_points[i * 3 + 1];
        const float pz = input_points[i * 3 + 2];

        output_points[i * 3 + 0] = m00 * px + m01 * py + m02 * pz + tx;
        output_points[i * 3 + 1] = m10 * px + m11 * py + m12 * pz + ty;
        output_points[i * 3 + 2] = m20 * px + m21 * py + m22 * pz + tz;
    }
}

float checksum(const std::array<float, kWidth> &values) {
    float sum = 0.0f;
    for (uint32_t i = 0; i < kWidth; ++i) {
        sum += static_cast<float>(i + 1u) * values[i];
    }
    return sum;
}

} // namespace

extern "C" __attribute__((noinline))
void transform_point_kernel(const float *input_points, const float *input_matrix,
                            const float *input_translation, float *output_points,
                            uint32_t size) {
    const float m00 = input_matrix[0], m01 = input_matrix[1], m02 = input_matrix[2];
    const float m10 = input_matrix[3], m11 = input_matrix[4], m12 = input_matrix[5];
    const float m20 = input_matrix[6], m21 = input_matrix[7], m22 = input_matrix[8];
    const float tx = input_translation[0];
    const float ty = input_translation[1];
    const float tz = input_translation[2];

    for (uint32_t i = 0; i < size; ++i) {
        const float px = input_points[i * 3 + 0];
        const float py = input_points[i * 3 + 1];
        const float pz = input_points[i * 3 + 2];

        output_points[i * 3 + 0] = m00 * px + m01 * py + m02 * pz + tx;
        output_points[i * 3 + 1] = m10 * px + m11 * py + m12 * pz + ty;
        output_points[i * 3 + 2] = m20 * px + m21 * py + m22 * pz + tz;
    }
}

int main() {
    std::array<float, kWidth> input_points = {};
    std::array<float, 9> matrix = {};
    std::array<float, 3> translation = {};
    std::array<float, kWidth> reference = {};
    std::array<float, kWidth> candidate = {};

    initialize_points(input_points);
    initialize_transform(matrix, translation);
    transform_point_ref(input_points.data(), matrix.data(), translation.data(),
                        reference.data(), kSize);
    transform_point_kernel(input_points.data(), matrix.data(), translation.data(),
                           candidate.data(), kSize);

    for (uint32_t i = 0; i < kWidth; ++i) {
        if (std::fabs(reference[i] - candidate[i]) > 1e-5f) {
            std::puts("FAILED");
            return 1;
        }
    }

    std::printf("transform_point checksum: %.3f\n", checksum(candidate));
    std::puts("PASSED");
    return 0;
}
