// 3D vector-normalization function variant migrated from the legacy app corpus.

#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>

namespace {

constexpr uint32_t kSize = 64;
constexpr uint32_t kWidth = kSize * 3;
constexpr float kEpsilon = 1e-8f;

void initialize_input(std::array<float, kWidth> &input) {
    for (uint32_t i = 0; i < kSize; ++i) {
        input[i * 3 + 0] = 3.0f + static_cast<float>(i) * 0.1f;
        input[i * 3 + 1] = 4.0f + static_cast<float>(i) * 0.2f;
        input[i * 3 + 2] = 0.0f;
    }
    input[0] = 0.0f;
    input[1] = 0.0f;
    input[2] = 0.0f;
}

void normalize_vec3_ref(const float *input_vec, float *output_normalized,
                        uint32_t size) {
    for (uint32_t i = 0; i < size; ++i) {
        const float x = input_vec[i * 3 + 0];
        const float y = input_vec[i * 3 + 1];
        const float z = input_vec[i * 3 + 2];
        const float length = std::sqrt(x * x + y * y + z * z);

        if (length > kEpsilon) {
            output_normalized[i * 3 + 0] = x / length;
            output_normalized[i * 3 + 1] = y / length;
            output_normalized[i * 3 + 2] = z / length;
        } else {
            output_normalized[i * 3 + 0] = 0.0f;
            output_normalized[i * 3 + 1] = 0.0f;
            output_normalized[i * 3 + 2] = 0.0f;
        }
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
void normalize_vec3_kernel(const float *input_vec, float *output_normalized,
                           uint32_t size) {
    for (uint32_t i = 0; i < size; ++i) {
        const float x = input_vec[i * 3 + 0];
        const float y = input_vec[i * 3 + 1];
        const float z = input_vec[i * 3 + 2];
        const float length = std::sqrt(x * x + y * y + z * z);

        if (length > kEpsilon) {
            output_normalized[i * 3 + 0] = x / length;
            output_normalized[i * 3 + 1] = y / length;
            output_normalized[i * 3 + 2] = z / length;
        } else {
            output_normalized[i * 3 + 0] = 0.0f;
            output_normalized[i * 3 + 1] = 0.0f;
            output_normalized[i * 3 + 2] = 0.0f;
        }
    }
}

int main() {
    std::array<float, kWidth> input = {};
    std::array<float, kWidth> reference = {};
    std::array<float, kWidth> candidate = {};

    initialize_input(input);
    normalize_vec3_ref(input.data(), reference.data(), kSize);
    normalize_vec3_kernel(input.data(), candidate.data(), kSize);

    for (uint32_t i = 0; i < kWidth; ++i) {
        if (std::fabs(reference[i] - candidate[i]) > 1e-5f) {
            std::puts("FAILED");
            return 1;
        }
    }

    std::printf("normalize_vec3 checksum: %.3f\n", checksum(candidate));
    std::puts("PASSED");
    return 0;
}
