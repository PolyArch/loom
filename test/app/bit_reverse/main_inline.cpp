
#include <array>
#include <cstdint>
#include <cstdio>

namespace {

constexpr uint32_t kSize = 256;

uint32_t reverse_bits32(uint32_t value) {
    uint32_t result = 0;
    for (uint32_t bit = 0; bit < 32; ++bit) {
        result = (result << 1) | (value & 1u);
        value >>= 1;
    }
    return result;
}

void initialize_input(std::array<uint32_t, kSize> &input) {
    input[0] = 0x00000000u;
    input[1] = 0xffffffffu;
    input[2] = 0x80000000u;
    input[3] = 0x00000001u;
    input[4] = 0xf0f0f0f0u;
    input[5] = 0x12345678u;

    for (uint32_t i = 6; i < kSize; ++i) {
        input[i] = i * 0xabcd1234u;
    }
}

void bit_reverse_ref(const uint32_t *input, uint32_t *output, uint32_t size) {
    for (uint32_t i = 0; i < size; ++i) {
        output[i] = reverse_bits32(input[i]);
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
    std::array<uint32_t, kSize> input = {};
    std::array<uint32_t, kSize> reference = {};
    std::array<uint32_t, kSize> candidate = {};

    initialize_input(input);
    bit_reverse_ref(input.data(), reference.data(), kSize);

    for (uint32_t i = 0; i < kSize; ++i) {
        uint32_t value = input[i];
        uint32_t result = 0;

        for (uint32_t bit = 0; bit < 32; ++bit) {
            result = (result << 1) | (value & 1u);
            value >>= 1;
        }

        candidate[i] = result;
    }

    for (uint32_t i = 0; i < kSize; ++i) {
        if (reference[i] != candidate[i]) {
            std::puts("FAILED");
            return 1;
        }
    }

    std::printf("bit_reverse checksum: %llu\n",
                static_cast<unsigned long long>(checksum(candidate)));
    std::puts("PASSED");
    return 0;
}
