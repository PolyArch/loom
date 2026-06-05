// Find-first-set function variant migrated from the legacy app corpus.

#include <array>
#include <cstdint>
#include <cstdio>

namespace {

constexpr uint32_t kSize = 32;

uint32_t find_first_set32(uint32_t value) {
    if (value == 0) {
        return 0;
    }

    uint32_t position = 1;
    while ((value & 1u) == 0) {
        ++position;
        value >>= 1;
    }
    return position;
}

void find_first_set_ref(const uint32_t *input, uint32_t *output, uint32_t size) {
    for (uint32_t i = 0; i < size; ++i) {
        output[i] = find_first_set32(input[i]);
    }
}

__attribute__((noinline))
void find_first_set_candidate(const uint32_t *input, uint32_t *output,
                              uint32_t size) {
    for (uint32_t i = 0; i < size; ++i) {
        output[i] = find_first_set32(input[i]);
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
        0x00000004u,
        0x80000000u,
        0xffffffffu,
        0xfffffff0u,
        0x00000100u,
    };
    std::array<uint32_t, kSize> reference = {};
    std::array<uint32_t, kSize> candidate = {};

    for (uint32_t i = 8; i < kSize; ++i) {
        input[i] = i * 0x00008765u;
    }

    find_first_set_ref(input.data(), reference.data(), kSize);
    find_first_set_candidate(input.data(), candidate.data(), kSize);

    for (uint32_t i = 0; i < kSize; ++i) {
        if (reference[i] != candidate[i]) {
            std::puts("FAILED");
            return 1;
        }
    }

    std::printf("find_first_set checksum: %u\n", checksum(candidate));
    std::puts("PASSED");
    return 0;
}
