// Indexed nested-accumulation inline variant migrated from the legacy app corpus.

#include <array>
#include <cstdint>
#include <cstdio>

namespace {

constexpr uint32_t kSize = 32;
constexpr uint32_t kStep = 3;
constexpr uint32_t kExpected = 1976;

} // namespace

int main() {
    std::array<uint32_t, kSize> input = {};
    for (uint32_t i = 0; i < kSize; ++i) {
        input[i] = (i + 1u) * 2u;
    }

    uint32_t acc = 0;
    for (uint32_t i = 0; i + kStep < kSize; i += kStep) {
        for (uint32_t j = kSize; j > 0; j >>= 1u) {
            const uint32_t idx = (i + j) % kSize;
            acc += input[idx];
        }
    }

    if (acc != kExpected) {
        std::puts("FAILED");
        return 1;
    }

    std::printf("stream_update checksum: %u\n", acc);
    std::puts("PASSED");
    return 0;
}
