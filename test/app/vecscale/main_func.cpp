
#include <array>
#include <cstdint>
#include <cstdio>

namespace {

constexpr uint32_t kSize = 32;
constexpr uint32_t kAlpha = 7;

void vecscale_ref(const uint32_t *input, uint32_t alpha, uint32_t *output,
                  uint32_t size) {
    for (uint32_t i = 0; i < size; ++i) {
        output[i] = alpha * input[i];
    }
}

__attribute__((noinline))
void vecscale_candidate(const uint32_t *input, uint32_t alpha,
                        uint32_t *output, uint32_t size) {
    for (uint32_t i = 0; i < size; ++i) {
        output[i] = alpha * input[i];
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
    std::array<uint32_t, kSize> input = {};
    std::array<uint32_t, kSize> reference = {};
    std::array<uint32_t, kSize> candidate = {};

    for (uint32_t i = 0; i < kSize; ++i) {
        input[i] = i % 100;
    }

    vecscale_ref(input.data(), kAlpha, reference.data(), kSize);
    vecscale_candidate(input.data(), kAlpha, candidate.data(), kSize);

    for (uint32_t i = 0; i < kSize; ++i) {
        if (reference[i] != candidate[i]) {
            std::puts("FAILED");
            return 1;
        }
    }

    std::printf("vecscale checksum: %u\n", checksum(candidate));
    std::puts("PASSED");
    return 0;
}
