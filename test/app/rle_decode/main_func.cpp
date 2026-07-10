
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

void rle_decode_ref(const uint32_t *values, const uint32_t *counts,
                    uint32_t *output, uint32_t encoded_length) {
    uint32_t write_idx = 0;
    for (uint32_t i = 0; i < encoded_length; ++i) {
        uint32_t value = values[i];
        uint32_t count = counts[i];
        for (uint32_t j = 0; j < count; ++j) {
            output[write_idx] = value;
            ++write_idx;
        }
    }
}

extern "C" __attribute__((noinline))
void rle_decode_kernel(const uint32_t *values, const uint32_t *counts,
                       uint32_t *output, uint32_t encoded_length) {
    uint32_t write_idx = 0;
    for (uint32_t i = 0; i < encoded_length; ++i) {
        uint32_t value = values[i];
        uint32_t count = counts[i];
        for (uint32_t j = 0; j < count; ++j) {
            output[write_idx] = value;
            ++write_idx;
        }
    }
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

    rle_decode_ref(values.data(), counts.data(), expected.data(),
                   kEncodedLength);
    rle_decode_kernel(values.data(), counts.data(), candidate.data(),
                      kEncodedLength);

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
