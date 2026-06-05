// Hash-mixing inline variant migrated from the legacy app corpus.

#include <array>
#include <cstdint>
#include <cstdio>

namespace {

constexpr uint32_t kSize = 64;
constexpr uint32_t kMixConstant = 0x5bd1e995u;

uint32_t rotl32(uint32_t value, uint32_t amount) {
    return (value << amount) | (value >> (32u - amount));
}

uint32_t mix_one(uint32_t state, uint32_t data) {
    uint32_t mixed = state + data;
    mixed = rotl32(mixed, 7);
    mixed ^= data;
    mixed *= kMixConstant;
    mixed = rotl32(mixed, 13);
    return mixed;
}

void initialize_inputs(std::array<uint32_t, kSize> &states,
                       std::array<uint32_t, kSize> &data) {
    for (uint32_t i = 0; i < kSize; ++i) {
        states[i] = 0x67452301u + i;
        data[i] = 0xefcdab89u + i * 13u;
    }
}

void hash_mix_ref(const uint32_t *input_state, const uint32_t *input_data,
                  uint32_t *output_state, uint32_t size) {
    for (uint32_t i = 0; i < size; ++i) {
        output_state[i] = mix_one(input_state[i], input_data[i]);
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
    std::array<uint32_t, kSize> states = {};
    std::array<uint32_t, kSize> data = {};
    std::array<uint32_t, kSize> reference = {};
    std::array<uint32_t, kSize> candidate = {};

    initialize_inputs(states, data);
    hash_mix_ref(states.data(), data.data(), reference.data(), kSize);

    for (uint32_t i = 0; i < kSize; ++i) {
        uint32_t mixed = states[i] + data[i];
        mixed = (mixed << 7) | (mixed >> 25);
        mixed ^= data[i];
        mixed *= kMixConstant;
        mixed = (mixed << 13) | (mixed >> 19);
        candidate[i] = mixed;
    }

    for (uint32_t i = 0; i < kSize; ++i) {
        if (reference[i] != candidate[i]) {
            std::puts("FAILED");
            return 1;
        }
    }

    std::printf("hash_mix checksum: %llu\n",
                static_cast<unsigned long long>(checksum(candidate)));
    std::puts("PASSED");
    return 0;
}
