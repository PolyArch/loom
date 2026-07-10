
#include <array>
#include <cstdint>
#include <cstdio>

namespace {

constexpr uint32_t kSize = 32;

uint32_t clz32(uint32_t value) {
    if (value == 0) {
        return 32;
    }

    uint32_t count = 0;
    uint32_t mask = 0x80000000u;
    while ((value & mask) == 0) {
        ++count;
        mask >>= 1;
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
        0x80000000u,
        0x40000000u,
        0x20000000u,
        0x00000001u,
        0xffffffffu,
        0x00ff00ffu,
        0x01000000u,
    };
    std::array<uint32_t, kSize> reference = {};
    std::array<uint32_t, kSize> candidate = {};

    for (uint32_t i = 8; i < kSize; ++i) {
        input[i] = i * 0x0012345u;
    }

    for (uint32_t i = 0; i < kSize; ++i) {
        reference[i] = clz32(input[i]);
    }
    for (uint32_t i = 0; i < kSize; ++i) {
        candidate[i] = clz32(input[i]);
    }
    for (uint32_t i = 0; i < kSize; ++i) {
        if (reference[i] != candidate[i]) {
            std::puts("FAILED");
            return 1;
        }
    }

    std::printf("clz checksum: %u\n", checksum(candidate));
    std::puts("PASSED");
    return 0;
}
