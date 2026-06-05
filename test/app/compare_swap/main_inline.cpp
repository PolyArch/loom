// Compare-swap inline variant migrated from the legacy app corpus.

#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>

namespace {

constexpr uint32_t kSize = 16;

float checksum(const std::array<float, kSize> &values) {
    float sum = 0.0f;
    for (float value : values) {
        sum += value;
    }
    return sum;
}

} // namespace

int main() {
    const std::array<float, kSize> a = {
        5.0f, 2.0f, 8.0f, 1.0f, 9.0f, 3.0f, 7.0f, 4.0f,
        6.0f, 10.0f, 15.0f, 12.0f, 11.0f, 14.0f, 13.0f, 16.0f,
    };
    const std::array<float, kSize> b = {
        3.0f, 7.0f, 1.0f, 9.0f, 2.0f, 8.0f, 4.0f, 6.0f,
        10.0f, 5.0f, 12.0f, 15.0f, 14.0f, 11.0f, 16.0f, 13.0f,
    };
    std::array<float, kSize> ref_min = {};
    std::array<float, kSize> ref_max = {};
    std::array<float, kSize> cand_min = {};
    std::array<float, kSize> cand_max = {};

    for (uint32_t i = 0; i < kSize; ++i) {
        if (a[i] <= b[i]) {
            ref_min[i] = a[i];
            ref_max[i] = b[i];
        } else {
            ref_min[i] = b[i];
            ref_max[i] = a[i];
        }
    }
    for (uint32_t i = 0; i < kSize; ++i) {
        if (a[i] <= b[i]) {
            cand_min[i] = a[i];
            cand_max[i] = b[i];
        } else {
            cand_min[i] = b[i];
            cand_max[i] = a[i];
        }
    }
    for (uint32_t i = 0; i < kSize; ++i) {
        if (std::fabs(ref_min[i] - cand_min[i]) > 1e-5f ||
            std::fabs(ref_max[i] - cand_max[i]) > 1e-5f) {
            std::puts("FAILED");
            return 1;
        }
    }

    std::printf("compare_swap checksums: %.6f %.6f\n",
                checksum(cand_min), checksum(cand_max));
    std::puts("PASSED");
    return 0;
}
