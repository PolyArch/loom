// Fixed-size nested stream accumulation inline variant.

#include <array>
#include <cstdint>
#include <cstdio>

namespace {

constexpr uint32_t kSize = 16;
constexpr uint32_t kExpected = 835;

} // namespace

int main() {
    std::array<uint32_t, kSize> input = {};
    for (uint32_t i = 0; i < kSize; ++i) {
        input[i] = i + 1u;
    }

    uint32_t acc = 0;
    for (uint32_t i = 1; i < kSize; i <<= 1u) {
        for (uint32_t j = kSize; j != 0; j >>= 1u) {
            for (uint32_t k = 1; k <= kSize; k <<= 1u) {
                const uint32_t idx = (i + j + k) % kSize;
                acc += input[idx];
            }
        }
    }

    if (acc != kExpected) {
        std::puts("FAILED");
        return 1;
    }

    std::printf("stream_nested checksum: %u\n", acc);
    std::puts("PASSED");
    return 0;
}
