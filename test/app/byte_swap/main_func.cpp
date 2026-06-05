// Byte-swap function variant migrated from the legacy app corpus.

#include <array>
#include <cstdint>
#include <cstdio>

namespace {

constexpr uint32_t kSize = 8;

uint32_t bswap32(uint32_t value) {
    uint32_t byte0 = (value >> 0) & 0xffu;
    uint32_t byte1 = (value >> 8) & 0xffu;
    uint32_t byte2 = (value >> 16) & 0xffu;
    uint32_t byte3 = (value >> 24) & 0xffu;
    return (byte0 << 24) | (byte1 << 16) | (byte2 << 8) | byte3;
}

void byte_swap_ref(const uint32_t *input, uint32_t *output, uint32_t size) {
    for (uint32_t i = 0; i < size; ++i) {
        output[i] = bswap32(input[i]);
    }
}

__attribute__((noinline))
void byte_swap_candidate(const uint32_t *input, uint32_t *output, uint32_t size) {
    for (uint32_t i = 0; i < size; ++i) {
        output[i] = bswap32(input[i]);
    }
}

uint64_t checksum(const std::array<uint32_t, kSize> &values) {
    uint64_t sum = 0;
    for (uint32_t value : values) {
        sum += value;
    }
    return sum;
}

} // namespace

int main() {
    const std::array<uint32_t, kSize> input = {
        0x00000000u,
        0xffffffffu,
        0x12345678u,
        0x11223344u,
        0xff000000u,
        0x000000ffu,
        0xabcdef01u,
        0x01020304u,
    };
    std::array<uint32_t, kSize> reference = {};
    std::array<uint32_t, kSize> candidate = {};

    byte_swap_ref(input.data(), reference.data(), kSize);
    byte_swap_candidate(input.data(), candidate.data(), kSize);

    for (uint32_t i = 0; i < kSize; ++i) {
        if (reference[i] != candidate[i]) {
            std::puts("FAILED");
            return 1;
        }
    }

    std::printf("byte_swap checksum: %llu\n",
                static_cast<unsigned long long>(checksum(candidate)));
    std::puts("PASSED");
    return 0;
}
