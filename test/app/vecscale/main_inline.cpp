// Vector-scale inline variant migrated from the legacy app corpus.

#include <array>
#include <cstdint>
#include <cstdio>

namespace {

constexpr uint32_t kSize = 32;
constexpr uint32_t kAlpha = 7;

uint32_t checksum(const std::array<uint32_t, kSize> &values) {
    uint32_t sum = 0;
    for (uint32_t value : values) {
        sum += value;
    }
    return sum;
}

} // namespace

int main() {
    std::array<uint32_t, kSize> input = {};
    std::array<uint32_t, kSize> reference = {};
    std::array<uint32_t, kSize> candidate = {};

    for (uint32_t i = 0; i < kSize; ++i) {
        input[i] = i % 100;
    }

    for (uint32_t i = 0; i < kSize; ++i) {
        reference[i] = kAlpha * input[i];
    }
    for (uint32_t i = 0; i < kSize; ++i) {
        candidate[i] = kAlpha * input[i];
    }
    for (uint32_t i = 0; i < kSize; ++i) {
        if (reference[i] != candidate[i]) {
            std::puts("FAILED");
            return 1;
        }
    }

    std::printf("vecscale checksum: %u\n", checksum(candidate));
    std::puts("PASSED");
    return 0;
}
