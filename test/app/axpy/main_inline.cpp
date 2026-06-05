// AXPY inline variant migrated from the legacy app corpus.

#include <array>
#include <cstdint>
#include <cstdio>

namespace {

constexpr uint32_t kSize = 8;
constexpr uint32_t kAlpha = 3;

uint32_t checksum(const std::array<uint32_t, kSize> &values) {
    uint32_t sum = 0;
    for (uint32_t value : values) {
        sum += value;
    }
    return sum;
}

} // namespace

int main() {
    const std::array<uint32_t, kSize> x = {1, 2, 3, 4, 5, 6, 7, 8};
    const std::array<uint32_t, kSize> y = {10, 20, 30, 40, 50, 60, 70, 80};
    std::array<uint32_t, kSize> reference = {};
    std::array<uint32_t, kSize> candidate = {};

    for (uint32_t i = 0; i < kSize; ++i) {
        reference[i] = kAlpha * x[i] + y[i];
    }
    for (uint32_t i = 0; i < kSize; ++i) {
        candidate[i] = kAlpha * x[i] + y[i];
    }
    for (uint32_t i = 0; i < kSize; ++i) {
        if (reference[i] != candidate[i]) {
            std::puts("FAILED");
            return 1;
        }
    }

    std::printf("AXPY checksum: %u\n", checksum(candidate));
    std::puts("PASSED");
    return 0;
}
