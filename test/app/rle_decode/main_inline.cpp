// Run-length decode inline variant migrated from the legacy app corpus.

#include <array>
#include <cstdint>
#include <cstdio>

namespace {

constexpr uint32_t kEncodedLength = 7;
constexpr uint32_t kOutputSize = 20;

void initialize_inputs(std::array<uint32_t, kEncodedLength> &values,
                       std::array<uint32_t, kEncodedLength> &counts) {
    values = {1, 2, 3, 4, 5, 6, 7};
    counts = {3, 2, 4, 5, 1, 3, 2};
}

uint64_t checksum(const std::array<uint32_t, kOutputSize> &values) {
    uint64_t sum = 0;
    for (uint32_t value : values) {
        sum += value;
    }
    return sum;
}

} // namespace

int main() {
    std::array<uint32_t, kEncodedLength> values = {};
    std::array<uint32_t, kEncodedLength> counts = {};
    std::array<uint32_t, kOutputSize> expected = {};
    std::array<uint32_t, kOutputSize> candidate = {};
    initialize_inputs(values, counts);

    uint32_t expected_write = 0;
    for (uint32_t i = 0; i < kEncodedLength; ++i) {
        for (uint32_t j = 0; j < counts[i]; ++j) {
            expected[expected_write] = values[i];
            ++expected_write;
        }
    }

    uint32_t candidate_write = 0;
    for (uint32_t i = 0; i < kEncodedLength; ++i) {
        for (uint32_t j = 0; j < counts[i]; ++j) {
            candidate[candidate_write] = values[i];
            ++candidate_write;
        }
    }

    for (uint32_t i = 0; i < kOutputSize; ++i) {
        if (expected[i] != candidate[i]) {
            std::puts("FAILED");
            return 1;
        }
    }

    std::printf("rle_decode checksum: %llu\n",
                static_cast<unsigned long long>(checksum(candidate)));
    std::puts("PASSED");
    return 0;
}
