
#include <array>
#include <cstdint>
#include <cstdio>

namespace {

constexpr uint32_t kSize = 32;
constexpr uint32_t kStep = 3;

void initialize_input(std::array<uint32_t, kSize> &input) {
    for (uint32_t i = 0; i < kSize; ++i) {
        input[i] = (i + 1u) * 2u;
    }
}

uint32_t stream_update_ref(const uint32_t *input, uint32_t size,
                           uint32_t step) {
    uint32_t acc = 0;
    for (uint32_t i = 0; i + step < size; i += step) {
        for (uint32_t j = size; j > 0; j >>= 1u) {
            const uint32_t idx = (i + j) % size;
            acc += input[idx];
        }
    }
    return acc;
}

} // namespace

extern "C" __attribute__((noinline))
void stream_update_kernel(const uint32_t *input, uint32_t *output,
                          uint32_t size, uint32_t step) {
    uint32_t acc = 0;
    for (uint32_t i = 0; i + step < size; i += step) {
        for (uint32_t j = size; j > 0; j >>= 1u) {
            const uint32_t idx = (i + j) % size;
            acc += input[idx];
        }
    }
    output[0] = acc;
}

int main() {
    std::array<uint32_t, kSize> input = {};
    std::array<uint32_t, 1> output = {};

    initialize_input(input);
    const uint32_t expected = stream_update_ref(input.data(), kSize, kStep);
    stream_update_kernel(input.data(), output.data(), kSize, kStep);

    if (output[0] != expected) {
        std::puts("FAILED");
        return 1;
    }

    std::printf("stream_update checksum: %u\n", output[0]);
    std::puts("PASSED");
    return 0;
}
