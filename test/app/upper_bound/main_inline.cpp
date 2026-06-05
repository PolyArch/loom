// Upper-bound inline variant migrated from the legacy app corpus.

#include <array>
#include <cstdint>
#include <cstdio>

namespace {

constexpr uint32_t kInputSize = 10;
constexpr uint32_t kTargetCount = 8;

uint32_t checksum(const std::array<uint32_t, kTargetCount> &values) {
    uint32_t sum = 0;
    for (uint32_t value : values) {
        sum += value;
    }
    return sum;
}

} // namespace

int main() {
    const std::array<float, kInputSize> sorted = {
        1.0f, 3.0f, 3.0f, 5.0f, 7.0f, 9.0f, 11.0f, 13.0f, 15.0f, 17.0f,
    };
    const std::array<float, kTargetCount> targets = {
        3.0f, 0.0f, 8.0f, 20.0f, 5.0f, 11.0f, 17.0f, 18.0f,
    };
    std::array<uint32_t, kTargetCount> reference = {};
    std::array<uint32_t, kTargetCount> candidate = {};

    for (uint32_t t = 0; t < kTargetCount; ++t) {
        float target = targets[t];
        uint32_t left = 0;
        uint32_t right = kInputSize;
        while (left < right) {
            uint32_t mid = left + (right - left) / 2;
            if (sorted[mid] <= target) {
                left = mid + 1;
            } else {
                right = mid;
            }
        }
        reference[t] = left;
    }
    for (uint32_t t = 0; t < kTargetCount; ++t) {
        float target = targets[t];
        uint32_t left = 0;
        uint32_t right = kInputSize;
        while (left < right) {
            uint32_t mid = left + (right - left) / 2;
            if (sorted[mid] <= target) {
                left = mid + 1;
            } else {
                right = mid;
            }
        }
        candidate[t] = left;
    }
    for (uint32_t i = 0; i < kTargetCount; ++i) {
        if (reference[i] != candidate[i]) {
            std::puts("FAILED");
            return 1;
        }
    }

    std::printf("upper_bound checksum: %u\n", checksum(candidate));
    std::puts("PASSED");
    return 0;
}
