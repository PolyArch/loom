// Upper-bound function variant migrated from the legacy app corpus.

#include <array>
#include <cstdint>
#include <cstdio>

namespace {

constexpr uint32_t kInputSize = 10;
constexpr uint32_t kTargetCount = 8;

void upper_bound_ref(const float *sorted, const float *targets,
                     uint32_t *indices, uint32_t input_size,
                     uint32_t target_count) {
    for (uint32_t t = 0; t < target_count; ++t) {
        float target = targets[t];
        uint32_t left = 0;
        uint32_t right = input_size;
        while (left < right) {
            uint32_t mid = left + (right - left) / 2;
            if (sorted[mid] <= target) {
                left = mid + 1;
            } else {
                right = mid;
            }
        }
        indices[t] = left;
    }
}

__attribute__((noinline))
void upper_bound_candidate(const float *sorted, const float *targets,
                           uint32_t *indices, uint32_t input_size,
                           uint32_t target_count) {
    for (uint32_t t = 0; t < target_count; ++t) {
        float target = targets[t];
        uint32_t left = 0;
        uint32_t right = input_size;
        while (left < right) {
            uint32_t mid = left + (right - left) / 2;
            if (sorted[mid] <= target) {
                left = mid + 1;
            } else {
                right = mid;
            }
        }
        indices[t] = left;
    }
}

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

    upper_bound_ref(sorted.data(), targets.data(), reference.data(),
                    kInputSize, kTargetCount);
    upper_bound_candidate(sorted.data(), targets.data(), candidate.data(),
                          kInputSize, kTargetCount);

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
