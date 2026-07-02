// Fixed-size nested stream accumulation migrated from the legacy app corpus.

#include <array>
#include <cstdint>
#include <cstdio>

namespace {

constexpr uint32_t kSize = 16;
constexpr uint32_t kLogSize = 4;
constexpr uint32_t kExpected = 835;

void initialize_input(std::array<uint32_t, kSize> &input) {
    for (uint32_t i = 0; i < kSize; ++i) {
        input[i] = i + 1u;
    }
}

uint32_t stream_nested_ref(const uint32_t *input) {
    uint32_t acc = 0;
    for (uint32_t i = 1; i < kSize; i <<= 1u) {
        for (uint32_t j = kSize; j != 0; j >>= 1u) {
            for (uint32_t k = 1; k <= kSize; k <<= 1u) {
                const uint32_t idx = (i + j + k) % kSize;
                acc += input[idx];
            }
        }
    }
    return acc;
}

} // namespace

extern "C" __attribute__((noinline))
void stream_nested_kernel(const uint32_t *input, uint32_t *output) {
    uint32_t acc = 0;
    for (uint32_t outer = 0; outer < kLogSize; ++outer) {
        const uint32_t i = 1u << outer;
        for (uint32_t middle = 0; middle <= kLogSize; ++middle) {
            const uint32_t j = kSize >> middle;
            for (uint32_t inner = 0; inner <= kLogSize; ++inner) {
                const uint32_t k = 1u << inner;
                const uint32_t idx = (i + j + k) % kSize;
                acc += input[idx];
            }
        }
    }
    output[0] = acc;
}

int main() {
    std::array<uint32_t, kSize> input = {};
    std::array<uint32_t, 1> output = {};

    initialize_input(input);
    const uint32_t expected = stream_nested_ref(input.data());
    stream_nested_kernel(input.data(), output.data());

    if (expected != kExpected || output[0] != expected) {
        std::puts("FAILED");
        return 1;
    }

    std::printf("stream_nested checksum: %u\n", output[0]);
    std::puts("PASSED");
    return 0;
}
