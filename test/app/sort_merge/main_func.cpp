// Bottom-up merge-sort function variant migrated from the legacy app corpus.

#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>

namespace {

constexpr uint32_t kSize = 8;
constexpr float kTolerance = 1e-6f;
constexpr std::array<float, kSize> kInput = {
    3.0f, 1.0f, 4.0f, 1.0f, 5.0f, 9.0f, 2.0f, 6.0f};

uint32_t min_u32(uint32_t a, uint32_t b) {
    return a < b ? a : b;
}

void sort_merge_ref(const float *input, float *output, float *temp,
                    uint32_t size) {
    for (uint32_t i = 0; i < size; ++i) {
        output[i] = input[i];
    }
    for (uint32_t width = 1; width < size; width *= 2u) {
        for (uint32_t left = 0; left < size; left += 2u * width) {
            const uint32_t mid = min_u32(left + width, size);
            const uint32_t right = min_u32(left + 2u * width, size);
            uint32_t i = left;
            uint32_t j = mid;
            uint32_t k = left;

            while (i < mid && j < right) {
                if (output[i] <= output[j]) {
                    temp[k] = output[i];
                    ++i;
                } else {
                    temp[k] = output[j];
                    ++j;
                }
                ++k;
            }
            while (i < mid) {
                temp[k] = output[i];
                ++i;
                ++k;
            }
            while (j < right) {
                temp[k] = output[j];
                ++j;
                ++k;
            }
            for (uint32_t idx = left; idx < right; ++idx) {
                output[idx] = temp[idx];
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
void sort_merge_kernel(const float *input, float *output, float *temp,
                       uint32_t size) {
    for (uint32_t i = 0; i < size; ++i) {
        output[i] = input[i];
    }
    for (uint32_t width = 1; width < size; width *= 2u) {
        for (uint32_t left = 0; left < size; left += 2u * width) {
            const uint32_t mid = min_u32(left + width, size);
            const uint32_t right = min_u32(left + 2u * width, size);
            uint32_t i = left;
            uint32_t j = mid;
            uint32_t k = left;

            while (i < mid && j < right) {
                if (output[i] <= output[j]) {
                    temp[k] = output[i];
                    ++i;
                } else {
                    temp[k] = output[j];
                    ++j;
                }
                ++k;
            }
            while (i < mid) {
                temp[k] = output[i];
                ++i;
                ++k;
            }
            while (j < right) {
                temp[k] = output[j];
                ++j;
                ++k;
            }
            for (uint32_t idx = left; idx < right; ++idx) {
                output[idx] = temp[idx];
            }
        }
    }
}

int main() {
    std::array<float, kSize> reference = {};
    std::array<float, kSize> ref_temp = {};
    std::array<float, kSize> candidate = {};
    std::array<float, kSize> cand_temp = {};

    sort_merge_ref(kInput.data(), reference.data(), ref_temp.data(), kSize);
    sort_merge_kernel(kInput.data(), candidate.data(), cand_temp.data(), kSize);

    for (uint32_t i = 0; i < kSize; ++i) {
        if (std::fabs(reference[i] - candidate[i]) > kTolerance) {
            std::puts("FAILED");
            return 1;
        }
    }

    std::printf("sort_merge checksum: %.3f\n", checksum(candidate));
    std::puts("PASSED");
    return 0;
}
