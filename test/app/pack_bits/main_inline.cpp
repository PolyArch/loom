// Bit-packing inline variant migrated from the legacy app corpus.

#include <array>
#include <cstdint>
#include <cstdio>

namespace {

constexpr uint32_t kNumBits = 100;
constexpr uint32_t kBitsPerWord = 32;
constexpr uint32_t kNumWords = (kNumBits + kBitsPerWord - 1) / kBitsPerWord;

void initialize_input(std::array<uint32_t, kNumBits> &input) {
    for (uint32_t i = 0; i < kNumBits; ++i) {
        input[i] = (i % 3u == 0u) ? 1u : 0u;
    }
}

void pack_bits_ref(const uint32_t *input_bits, uint32_t *output_packed,
                   uint32_t num_bits) {
    const uint32_t num_words = (num_bits + kBitsPerWord - 1) / kBitsPerWord;

    for (uint32_t word_idx = 0; word_idx < num_words; ++word_idx) {
        uint32_t packed_word = 0;
        const uint32_t start_bit = word_idx * kBitsPerWord;
        uint32_t end_bit = start_bit + kBitsPerWord;
        if (end_bit > num_bits) {
            end_bit = num_bits;
        }

        for (uint32_t bit_idx = start_bit; bit_idx < end_bit; ++bit_idx) {
            const uint32_t bit_position = bit_idx - start_bit;
            if ((input_bits[bit_idx] & 1u) != 0u) {
                packed_word |= 1u << bit_position;
            }
        }

        output_packed[word_idx] = packed_word;
    }
}

uint64_t checksum(const std::array<uint32_t, kNumWords> &values) {
    uint64_t sum = 0;
    for (uint32_t i = 0; i < kNumWords; ++i) {
        sum += static_cast<uint64_t>(i + 1u) * values[i];
    }
    return sum;
}

} // namespace

int main() {
    std::array<uint32_t, kNumBits> input = {};
    std::array<uint32_t, kNumWords> reference = {};
    std::array<uint32_t, kNumWords> candidate = {};

    initialize_input(input);
    pack_bits_ref(input.data(), reference.data(), kNumBits);

    const uint32_t num_words = (kNumBits + kBitsPerWord - 1) / kBitsPerWord;
    for (uint32_t word_idx = 0; word_idx < num_words; ++word_idx) {
        uint32_t packed_word = 0;
        const uint32_t start_bit = word_idx * kBitsPerWord;
        uint32_t end_bit = start_bit + kBitsPerWord;
        if (end_bit > kNumBits) {
            end_bit = kNumBits;
        }

        for (uint32_t bit_idx = start_bit; bit_idx < end_bit; ++bit_idx) {
            const uint32_t bit_position = bit_idx - start_bit;
            if ((input[bit_idx] & 1u) != 0u) {
                packed_word |= 1u << bit_position;
            }
        }

        candidate[word_idx] = packed_word;
    }

    for (uint32_t i = 0; i < kNumWords; ++i) {
        if (reference[i] != candidate[i]) {
            std::puts("FAILED");
            return 1;
        }
    }

    std::printf("pack_bits checksum: %llu\n",
                static_cast<unsigned long long>(checksum(candidate)));
    std::puts("PASSED");
    return 0;
}
