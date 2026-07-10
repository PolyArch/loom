
#include <array>
#include <cstdint>
#include <cstdio>

namespace {

constexpr uint32_t kSize = 20;

void rle_encode_ref(const uint32_t *input, uint32_t *values, uint32_t *counts,
                    uint32_t *length, uint32_t size) {
    if (size == 0) {
        *length = 0;
        return;
    }

    uint32_t write = 0;
    uint32_t current = input[0];
    uint32_t count = 1;
    for (uint32_t i = 1; i < size; ++i) {
        if (input[i] == current) {
            ++count;
        } else {
            values[write] = current;
            counts[write] = count;
            ++write;
            current = input[i];
            count = 1;
        }
    }

    values[write] = current;
    counts[write] = count;
    *length = write + 1u;
}

uint64_t checksum(const std::array<uint32_t, kSize> &values,
                  const std::array<uint32_t, kSize> &counts,
                  uint32_t length) {
    uint64_t sum = static_cast<uint64_t>(length) * 1009u;
    for (uint32_t i = 0; i < length; ++i) {
        sum += static_cast<uint64_t>(i + 1u) *
               (static_cast<uint64_t>(values[i]) * 131u + counts[i] * 17u);
    }
    return sum;
}

} // namespace

extern "C" __attribute__((noinline))
void rle_encode_kernel(const uint32_t *input, uint32_t *values,
                       uint32_t *counts, uint32_t *length, uint32_t size) {
    if (size == 0) {
        *length = 0;
        return;
    }

    uint32_t write = 0;
    uint32_t current = input[0];
    uint32_t count = 1;
    for (uint32_t i = 1; i < size; ++i) {
        if (input[i] == current) {
            ++count;
        } else {
            values[write] = current;
            counts[write] = count;
            ++write;
            current = input[i];
            count = 1;
        }
    }

    values[write] = current;
    counts[write] = count;
    *length = write + 1u;
}

int main() {
    const std::array<uint32_t, kSize> input = {1, 1, 1, 2, 2, 3, 3,
                                               3, 3, 4, 4, 4, 4, 4,
                                               5, 6, 6, 6, 7, 7};
    std::array<uint32_t, kSize> ref_values = {};
    std::array<uint32_t, kSize> ref_counts = {};
    std::array<uint32_t, kSize> cand_values = {};
    std::array<uint32_t, kSize> cand_counts = {};
    uint32_t ref_length = 0;
    uint32_t cand_length = 0;

    rle_encode_ref(input.data(), ref_values.data(), ref_counts.data(),
                   &ref_length, kSize);
    rle_encode_kernel(input.data(), cand_values.data(), cand_counts.data(),
                      &cand_length, kSize);

    if (ref_length != cand_length) {
        std::puts("FAILED");
        return 1;
    }
    for (uint32_t i = 0; i < ref_length; ++i) {
        if (ref_values[i] != cand_values[i] || ref_counts[i] != cand_counts[i]) {
            std::puts("FAILED");
            return 1;
        }
    }

    std::printf("rle_encode checksum: %llu\n",
                static_cast<unsigned long long>(
                    checksum(cand_values, cand_counts, cand_length)));
    std::puts("PASSED");
    return 0;
}
