// Bottom-up merge-sort inline variant migrated from the legacy app corpus.

#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>

namespace {

constexpr uint32_t kSize = 8;
constexpr float kTolerance = 1e-6f;
constexpr std::array<float, kSize> kInput = {
    3.0f, 1.0f, 4.0f, 1.0f, 5.0f, 9.0f, 2.0f, 6.0f};
constexpr std::array<float, kSize> kExpected = {
    1.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 9.0f};

uint32_t min_u32(uint32_t a, uint32_t b) {
    return a < b ? a : b;
}

double checksum(const std::array<float, kSize> &values) {
    double sum = 0.0;
    for (uint32_t i = 0; i < kSize; ++i) {
        sum += static_cast<double>(i + 1u) * values[i];
    }
    return sum;
}

} // namespace

int main() {
    std::array<float, kSize> output = {};
    std::array<float, kSize> merge_buffer = {};
    for (uint32_t i = 0; i < kSize; ++i) {
        output[i] = kInput[i];
    }

    for (uint32_t width = 1; width < kSize; width *= 2u) {
        for (uint32_t left = 0; left < kSize; left += 2u * width) {
            const uint32_t mid = min_u32(left + width, kSize);
            const uint32_t right = min_u32(left + 2u * width, kSize);
            uint32_t i = left;
            uint32_t j = mid;
            uint32_t k = left;

            while (i < mid && j < right) {
                if (output[i] <= output[j]) {
                    merge_buffer[k] = output[i];
                    ++i;
                } else {
                    merge_buffer[k] = output[j];
                    ++j;
                }
                ++k;
            }
            while (i < mid) {
                merge_buffer[k] = output[i];
                ++i;
                ++k;
            }
            while (j < right) {
                merge_buffer[k] = output[j];
                ++j;
                ++k;
            }
            for (uint32_t idx = left; idx < right; ++idx) {
                output[idx] = merge_buffer[idx];
            }
        }
    }

    for (uint32_t i = 0; i < kSize; ++i) {
        if (std::fabs(output[i] - kExpected[i]) > kTolerance) {
            std::puts("FAILED");
            return 1;
        }
    }

    std::printf("sort_merge checksum: %.3f\n", checksum(output));
    std::puts("PASSED");
    return 0;
}
