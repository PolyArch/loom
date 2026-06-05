// Count-leading-zeros function variant migrated from the legacy app corpus.

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

void clz_ref(const uint32_t *input, uint32_t *output, uint32_t size) {
    for (uint32_t i = 0; i < size; ++i) {
        output[i] = clz32(input[i]);
    }
}

__attribute__((noinline))
void clz_candidate(const uint32_t *input, uint32_t *output, uint32_t size) {
    for (uint32_t i = 0; i < size; ++i) {
        output[i] = clz32(input[i]);
    }
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

    clz_ref(input.data(), reference.data(), kSize);
    clz_candidate(input.data(), candidate.data(), kSize);

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
