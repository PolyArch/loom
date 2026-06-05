// 3D cross-product inline variant migrated from the legacy app corpus.

#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>

namespace {

constexpr uint32_t kSize = 64;
constexpr uint32_t kWidth = kSize * 3;

void initialize_inputs(std::array<float, kWidth> &vec_a,
                       std::array<float, kWidth> &vec_b) {
    for (uint32_t i = 0; i < kSize; ++i) {
        vec_a[i * 3 + 0] = 1.0f + static_cast<float>(i) * 0.1f;
        vec_a[i * 3 + 1] = 0.0f;
        vec_a[i * 3 + 2] = 0.0f;

        vec_b[i * 3 + 0] = 0.0f;
        vec_b[i * 3 + 1] = 1.0f + static_cast<float>(i) * 0.1f;
        vec_b[i * 3 + 2] = 0.0f;
    }
}

void cross_product_ref(const float *input_vec_a, const float *input_vec_b,
                       float *output_result, uint32_t size) {
    for (uint32_t i = 0; i < size; ++i) {
        const float ax = input_vec_a[i * 3 + 0];
        const float ay = input_vec_a[i * 3 + 1];
        const float az = input_vec_a[i * 3 + 2];
        const float bx = input_vec_b[i * 3 + 0];
        const float by = input_vec_b[i * 3 + 1];
        const float bz = input_vec_b[i * 3 + 2];

        output_result[i * 3 + 0] = ay * bz - az * by;
        output_result[i * 3 + 1] = az * bx - ax * bz;
        output_result[i * 3 + 2] = ax * by - ay * bx;
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

int main() {
    std::array<float, kWidth> vec_a = {};
    std::array<float, kWidth> vec_b = {};
    std::array<float, kWidth> reference = {};
    std::array<float, kWidth> candidate = {};

    initialize_inputs(vec_a, vec_b);
    cross_product_ref(vec_a.data(), vec_b.data(), reference.data(), kSize);

    for (uint32_t i = 0; i < kSize; ++i) {
        const float ax = vec_a[i * 3 + 0];
        const float ay = vec_a[i * 3 + 1];
        const float az = vec_a[i * 3 + 2];
        const float bx = vec_b[i * 3 + 0];
        const float by = vec_b[i * 3 + 1];
        const float bz = vec_b[i * 3 + 2];

        candidate[i * 3 + 0] = ay * bz - az * by;
        candidate[i * 3 + 1] = az * bx - ax * bz;
        candidate[i * 3 + 2] = ax * by - ay * bx;
    }

    for (uint32_t i = 0; i < kWidth; ++i) {
        if (std::fabs(reference[i] - candidate[i]) > 1e-5f) {
            std::puts("FAILED");
            return 1;
        }
    }

    std::printf("cross_product checksum: %.3f\n", checksum(candidate));
    std::puts("PASSED");
    return 0;
}
