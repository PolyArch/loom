// Delta-decode function variant migrated from the legacy app corpus.

#include <array>
#include <cstdint>
#include <cstdio>

namespace {

constexpr uint32_t kSize = 10;

void delta_decode_ref(const uint32_t *input, uint32_t *output, uint32_t size) {
    if (size == 0) {
        return;
    }
    output[0] = input[0];
    for (uint32_t i = 1; i < size; ++i) {
        output[i] = output[i - 1u] + input[i];
    }
}

uint64_t checksum(const std::array<uint32_t, kSize> &values) {
    uint64_t sum = 0;
    for (uint32_t i = 0; i < kSize; ++i) {
        sum += static_cast<uint64_t>(i + 1u) * values[i];
    }
    return sum;
}

} // namespace

extern "C" __attribute__((noinline))
void delta_decode_kernel(const uint32_t *input, uint32_t *output,
                         uint32_t size) {
    if (size == 0) {
        return;
    }
    output[0] = input[0];
    for (uint32_t i = 1; i < size; ++i) {
        output[i] = output[i - 1u] + input[i];
    }
}

int main() {
    const std::array<uint32_t, kSize> input = {100, 2, 3, 5, 5,
                                               7,   8, 5, 7, 8};
    std::array<uint32_t, kSize> reference = {};
    std::array<uint32_t, kSize> candidate = {};

    delta_decode_ref(input.data(), reference.data(), kSize);
    delta_decode_kernel(input.data(), candidate.data(), kSize);

    for (uint32_t i = 0; i < kSize; ++i) {
        if (reference[i] != candidate[i]) {
            std::puts("FAILED");
            return 1;
        }
    }

    std::printf("delta_decode checksum: %llu\n",
                static_cast<unsigned long long>(checksum(candidate)));
    std::puts("PASSED");
    return 0;
}
