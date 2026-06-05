// Population-count inline variant migrated from the legacy app corpus.

#include <array>
#include <cstdint>
#include <cstdio>

namespace {

constexpr uint32_t kSize = 32;

uint32_t popcount32(uint32_t value) {
    uint32_t count = 0;
    while (value != 0) {
        count += value & 1u;
        value >>= 1;
    }
    return count;
}

uint32_t checksum(const std::array<uint32_t, kSize> &values) {
    uint32_t sum = 0;
    for (uint32_t value : values) {
        sum += value;
    }
    return sum;
}

} // namespace

int main() {
    std::array<uint32_t, kSize> input = {
        0x00000000u,
        0x00000001u,
        0x00000002u,
        0x00000003u,
        0x00000007u,
        0x0000000fu,
        0xffffffffu,
        0x80000000u,
    };
    std::array<uint32_t, kSize> reference = {};
    std::array<uint32_t, kSize> candidate = {};

    for (uint32_t i = 8; i < kSize; ++i) {
        input[i] = i * 0x12345678u + (i << 16);
    }

    for (uint32_t i = 0; i < kSize; ++i) {
        reference[i] = popcount32(input[i]);
    }
    for (uint32_t i = 0; i < kSize; ++i) {
        candidate[i] = popcount32(input[i]);
    }
    for (uint32_t i = 0; i < kSize; ++i) {
        if (reference[i] != candidate[i]) {
            std::puts("FAILED");
            return 1;
        }
    }

    std::printf("popcount checksum: %u\n", checksum(candidate));
    std::puts("PASSED");
    return 0;
}
