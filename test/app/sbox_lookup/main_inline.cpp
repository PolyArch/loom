// S-box lookup inline variant migrated from the legacy app corpus.

#include <array>
#include <cstdint>
#include <cstdio>

namespace {

constexpr uint32_t kSize = 64;
constexpr uint32_t kSboxSize = 256;

void initialize_sbox(std::array<uint32_t, kSboxSize> &sbox) {
    for (uint32_t i = 0; i < kSboxSize; ++i) {
        sbox[i] = (i * 7u + 31u) & 0xffu;
    }
}

void initialize_input(std::array<uint32_t, kSize> &input) {
    for (uint32_t i = 0; i < kSize; ++i) {
        input[i] = (i * 13u + 17u) & 0xffu;
    }
}

void sbox_lookup_ref(const uint32_t *input_data, const uint32_t *input_sbox,
                     uint32_t *output_result, uint32_t size) {
    for (uint32_t i = 0; i < size; ++i) {
        const uint32_t index = input_data[i] & 0xffu;
        output_result[i] = input_sbox[index];
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

int main() {
    std::array<uint32_t, kSboxSize> sbox = {};
    std::array<uint32_t, kSize> input = {};
    std::array<uint32_t, kSize> reference = {};
    std::array<uint32_t, kSize> candidate = {};

    initialize_sbox(sbox);
    initialize_input(input);
    sbox_lookup_ref(input.data(), sbox.data(), reference.data(), kSize);

    for (uint32_t i = 0; i < kSize; ++i) {
        const uint32_t index = input[i] & 0xffu;
        candidate[i] = sbox[index];
    }

    for (uint32_t i = 0; i < kSize; ++i) {
        if (reference[i] != candidate[i]) {
            std::puts("FAILED");
            return 1;
        }
    }

    std::printf("sbox_lookup checksum: %llu\n",
                static_cast<unsigned long long>(checksum(candidate)));
    std::puts("PASSED");
    return 0;
}
