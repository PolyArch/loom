
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>

namespace {

constexpr uint32_t kSize = 8;
constexpr std::array<float, kSize> kInput = {
    1.0f,
    2.0f,
    3.0f,
    4.0f,
    5.0f,
    6.0f,
    7.0f,
    8.0f,
};
constexpr float kTolerance = 1.0e-5f;

void normalize_ref(const float *input, float *output, uint32_t size) {
    float sum = 0.0f;
    for (uint32_t i = 0; i < size; ++i) {
        sum += input[i];
    }
    const float scale = (sum > 0.0f) ? (1.0f / sum) : 1.0f;
    for (uint32_t i = 0; i < size; ++i) {
        output[i] = input[i] * scale;
    }
}

float checksum(const std::array<float, kSize> &values) {
    float total = 0.0f;
    for (uint32_t i = 0; i < kSize; ++i) {
        total += static_cast<float>(i + 1u) * values[i];
    }
    return total;
}

} // namespace

extern "C" __attribute__((noinline))
void normalize_sum_kernel(const float *__restrict__ data,
                          float *__restrict__ result, int n) {
    float sum = 0.0f;
    for (int i = 0; i < n; ++i) {
        sum += data[i];
    }
    *result = sum;
}

extern "C" __attribute__((noinline))
void normalize_max_kernel(const float *__restrict__ data,
                          float *__restrict__ result, int n) {
    float max_value = data[0];
    for (int i = 1; i < n; ++i) {
        if (data[i] > max_value) {
            max_value = data[i];
        }
    }
    *result = max_value;
}

extern "C" __attribute__((noinline))
void normalize_scale_kernel(const float *__restrict__ input, float sum,
                            float *__restrict__ output, int n) {
    const float scale = (sum > 0.0f) ? (1.0f / sum) : 1.0f;
    for (int i = 0; i < n; ++i) {
        output[i] = input[i] * scale;
    }
}

extern "C" __attribute__((noinline))
void normalize_kernel(const float *__restrict__ input,
                      float *__restrict__ output, int n) {
    float sum_result = 0.0f;
    float max_result = 0.0f;

    normalize_sum_kernel(input, &sum_result, n);
    normalize_max_kernel(input, &max_result, n);
    normalize_scale_kernel(input, sum_result, output, n);

    (void)max_result;
}

int main() {
    std::array<float, kSize> input = kInput;
    std::array<float, kSize> reference = {};
    std::array<float, kSize> candidate = {};

    normalize_ref(input.data(), reference.data(), kSize);
    normalize_kernel(input.data(), candidate.data(), static_cast<int>(kSize));

    for (uint32_t i = 0; i < kSize; ++i) {
        if (std::fabs(reference[i] - candidate[i]) > kTolerance) {
            std::puts("FAILED");
            return 1;
        }
    }

    std::printf("normalize checksum: %.6f\n", checksum(candidate));
    std::puts("PASSED");
    return 0;
}
