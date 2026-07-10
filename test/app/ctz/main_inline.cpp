
#include <array>
#include <cstdint>
#include <cstdio>

namespace {

constexpr uint32_t kSize = 32;

uint32_t ctz32(uint32_t value) {
    if (value == 0) {
        return 32;
    }

    uint32_t count = 0;
    while ((value & 1u) == 0) {
        ++count;
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
        0x80000000u,
        0xffffffffu,
        0x00010000u,
        0x01000000u,
        0x00000008u,
    };
    std::array<uint32_t, kSize> reference = {};
    std::array<uint32_t, kSize> candidate = {};

    for (uint32_t i = 8; i < kSize; ++i) {
        input[i] = i * 0x00005678u;
    }

    for (uint32_t i = 0; i < kSize; ++i) {
        reference[i] = ctz32(input[i]);
    }
    for (uint32_t i = 0; i < kSize; ++i) {
        candidate[i] = ctz32(input[i]);
    }
    for (uint32_t i = 0; i < kSize; ++i) {
        if (reference[i] != candidate[i]) {
            std::puts("FAILED");
            return 1;
        }
    }

    std::printf("ctz checksum: %u\n", checksum(candidate));
    std::puts("PASSED");
    return 0;
}
