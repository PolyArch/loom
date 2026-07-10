
#include <array>
#include <cstdint>
#include <cstdio>

namespace {

constexpr uint32_t kNumBits = 100;
constexpr uint32_t kBitsPerWord = 32;
constexpr uint32_t kNumWords = (kNumBits + kBitsPerWord - 1) / kBitsPerWord;

void initialize_input(std::array<uint32_t, kNumWords> &input) {
    input[0] = 0xaaaaaaaau;
    input[1] = 0x13579bdfu;
    input[2] = 0x80000001u;
    input[3] = 0x0000000fu;
}

void unpack_bits_ref(const uint32_t *input_packed, uint32_t *output_bits,
                     uint32_t num_bits) {
    const uint32_t num_words = (num_bits + kBitsPerWord - 1) / kBitsPerWord;

    for (uint32_t word_idx = 0; word_idx < num_words; ++word_idx) {
        const uint32_t packed_word = input_packed[word_idx];
        const uint32_t start_bit = word_idx * kBitsPerWord;
        uint32_t end_bit = start_bit + kBitsPerWord;
        if (end_bit > num_bits) {
            end_bit = num_bits;
        }

        for (uint32_t bit_idx = start_bit; bit_idx < end_bit; ++bit_idx) {
            const uint32_t bit_position = bit_idx - start_bit;
            output_bits[bit_idx] = (packed_word >> bit_position) & 1u;
        }
    }
}

uint32_t checksum(const std::array<uint32_t, kNumBits> &values) {
    uint32_t sum = 0;
    for (uint32_t i = 0; i < kNumBits; ++i) {
        sum += (i + 1u) * values[i];
    }
    return sum;
}

} // namespace

extern "C" __attribute__((noinline))
void unpack_bits_kernel(const uint32_t *input_packed, uint32_t *output_bits,
                        uint32_t num_bits) {
    const uint32_t num_words = (num_bits + kBitsPerWord - 1) / kBitsPerWord;

    for (uint32_t word_idx = 0; word_idx < num_words; ++word_idx) {
        const uint32_t packed_word = input_packed[word_idx];
        const uint32_t start_bit = word_idx * kBitsPerWord;
        uint32_t end_bit = start_bit + kBitsPerWord;
        if (end_bit > num_bits) {
            end_bit = num_bits;
        }

        for (uint32_t bit_idx = start_bit; bit_idx < end_bit; ++bit_idx) {
            const uint32_t bit_position = bit_idx - start_bit;
            output_bits[bit_idx] = (packed_word >> bit_position) & 1u;
        }
    }
}

int main() {
    std::array<uint32_t, kNumWords> input = {};
    std::array<uint32_t, kNumBits> reference = {};
    std::array<uint32_t, kNumBits> candidate = {};

    initialize_input(input);
    unpack_bits_ref(input.data(), reference.data(), kNumBits);
    unpack_bits_kernel(input.data(), candidate.data(), kNumBits);

    for (uint32_t i = 0; i < kNumBits; ++i) {
        if (reference[i] != candidate[i]) {
            std::puts("FAILED");
            return 1;
        }
    }

    std::printf("unpack_bits checksum: %u\n", checksum(candidate));
    std::puts("PASSED");
    return 0;
}
