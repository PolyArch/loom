// Rolling-hash function variant migrated from the legacy app corpus.

#include <array>
#include <cstdint>
#include <cstdio>

namespace {

constexpr uint32_t kSize = 128;
constexpr uint32_t kWindowSize = 8;
constexpr uint32_t kWindowCount = kSize - kWindowSize + 1;
constexpr uint32_t kBase = 256;
constexpr uint32_t kModulus = 101;

void initialize_input(std::array<uint32_t, kSize> &input) {
    for (uint32_t i = 0; i < kSize; ++i) {
        input[i] = static_cast<uint32_t>('a') + (i % 26);
    }
}

void string_hash_ref(const uint32_t *input, uint32_t *output_hashes,
                     uint32_t size, uint32_t window_size) {
    if (window_size > size) {
        return;
    }

    uint32_t h = 1;
    for (uint32_t i = 0; i < window_size - 1; ++i) {
        h = (h * kBase) % kModulus;
    }

    uint32_t hash_value = 0;
    for (uint32_t i = 0; i < window_size; ++i) {
        hash_value = (hash_value * kBase + input[i]) % kModulus;
    }
    output_hashes[0] = hash_value;

    for (uint32_t i = 1; i <= size - window_size; ++i) {
        hash_value = (hash_value + kModulus - (input[i - 1] * h) % kModulus) %
                     kModulus;
        hash_value =
            (hash_value * kBase + input[i + window_size - 1]) % kModulus;
        output_hashes[i] = hash_value;
    }
}

extern "C" __attribute__((noinline))
void string_hash_kernel(const uint32_t *input, uint32_t *output_hashes,
                        uint32_t size, uint32_t window_size) {
    if (window_size > size) {
        return;
    }

    uint32_t h = 1;
    for (uint32_t i = 0; i < window_size - 1; ++i) {
        h = (h * kBase) % kModulus;
    }

    uint32_t hash_value = 0;
    for (uint32_t i = 0; i < window_size; ++i) {
        hash_value = (hash_value * kBase + input[i]) % kModulus;
    }
    output_hashes[0] = hash_value;

    for (uint32_t i = 1; i <= size - window_size; ++i) {
        hash_value = (hash_value + kModulus - (input[i - 1] * h) % kModulus) %
                     kModulus;
        hash_value =
            (hash_value * kBase + input[i + window_size - 1]) % kModulus;
        output_hashes[i] = hash_value;
    }
}

uint64_t checksum(const std::array<uint32_t, kWindowCount> &values) {
    uint64_t sum = 0;
    for (uint32_t value : values) {
        sum += value;
    }
    return sum;
}

} // namespace

int main() {
    std::array<uint32_t, kSize> input = {};
    std::array<uint32_t, kWindowCount> expected = {};
    std::array<uint32_t, kWindowCount> candidate = {};
    initialize_input(input);

    string_hash_ref(input.data(), expected.data(), kSize, kWindowSize);
    string_hash_kernel(input.data(), candidate.data(), kSize, kWindowSize);

    for (uint32_t i = 0; i < kWindowCount; ++i) {
        if (expected[i] != candidate[i]) {
            std::puts("FAILED");
            return 1;
        }
    }

    std::printf("string_hash checksum: %llu\n",
                static_cast<unsigned long long>(checksum(candidate)));
    std::puts("PASSED");
    return 0;
}
