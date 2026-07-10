
#include <array>
#include <cstdint>
#include <cstdio>

namespace {

constexpr uint32_t kInputSize = 10;
constexpr uint32_t kTargetCount = 5;
constexpr uint32_t kNotFound = 0xffffffffu;

uint64_t checksum(const std::array<uint32_t, kTargetCount> &values) {
    uint64_t sum = 0;
    for (uint32_t value : values) {
        sum += value;
    }
    return sum;
}

} // namespace

int main() {
    const std::array<float, kInputSize> sorted = {
        1.0f, 3.0f, 5.0f, 7.0f, 9.0f, 11.0f, 13.0f, 15.0f, 17.0f, 19.0f,
    };
    const std::array<float, kTargetCount> targets = {
        7.0f, 2.0f, 15.0f, 20.0f, 1.0f,
    };
    std::array<uint32_t, kTargetCount> reference = {};
    std::array<uint32_t, kTargetCount> candidate = {};

    for (uint32_t t = 0; t < kTargetCount; ++t) {
        int32_t found = -1;
        for (uint32_t i = 0; i < kInputSize; ++i) {
            if (sorted[i] == targets[t]) {
                found = static_cast<int32_t>(i);
            }
        }
        reference[t] = found < 0 ? kNotFound : static_cast<uint32_t>(found);
    }

    for (uint32_t t = 0; t < kTargetCount; ++t) {
        float target = targets[t];
        int32_t left = 0;
        int32_t right = static_cast<int32_t>(kInputSize) - 1;
        int32_t found = -1;
        while (left <= right) {
            int32_t mid = left + (right - left) / 2;
            if (sorted[mid] == target) {
                found = mid;
                break;
            }
            if (sorted[mid] < target) {
                left = mid + 1;
            } else {
                right = mid - 1;
            }
        }
        candidate[t] = found < 0 ? kNotFound : static_cast<uint32_t>(found);
    }

    for (uint32_t i = 0; i < kTargetCount; ++i) {
        if (reference[i] != candidate[i]) {
            std::puts("FAILED");
            return 1;
        }
    }

    std::printf("binary_search checksum: %llu\n",
                static_cast<unsigned long long>(checksum(candidate)));
    std::puts("PASSED");
    return 0;
}
