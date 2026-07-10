
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

void ctz_ref(const uint32_t *input, uint32_t *output, uint32_t size) {
    for (uint32_t i = 0; i < size; ++i) {
        output[i] = ctz32(input[i]);
    }
}

__attribute__((noinline))
void ctz_candidate(const uint32_t *input, uint32_t *output, uint32_t size) {
    for (uint32_t i = 0; i < size; ++i) {
        output[i] = ctz32(input[i]);
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

    ctz_ref(input.data(), reference.data(), kSize);
    ctz_candidate(input.data(), candidate.data(), kSize);

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
