// KMP prefix-table function variant migrated from the legacy app corpus.

#include <array>
#include <cstdint>
#include <cstdio>

namespace {

constexpr uint32_t kSize = 16;
constexpr std::array<uint32_t, kSize> kPattern = {
    'A', 'B', 'A', 'B', 'C', 'A', 'B', 'A',
    'B', 'A', 'A', 'B', 'A', 'B', 'C', 'D'};
constexpr std::array<uint32_t, kSize> kExpected = {
    0u, 0u, 1u, 2u, 0u, 1u, 2u, 3u,
    4u, 3u, 1u, 2u, 3u, 4u, 5u, 0u};

uint64_t checksum(const std::array<uint32_t, kSize> &values) {
    uint64_t sum = 0;
    for (uint32_t i = 0; i < kSize; ++i) {
        sum += static_cast<uint64_t>(i + 1u) * values[i];
    }
    return sum;
}

} // namespace

extern "C" __attribute__((noinline))
void kmp_table_kernel(const uint32_t *input_pattern,
                      uint32_t *output_table, uint32_t size) {
    output_table[0] = 0;
    uint32_t j = 0;

    for (uint32_t i = 1; i < size; ++i) {
        while (j > 0 && input_pattern[i] != input_pattern[j]) {
            j = output_table[j - 1u];
        }

        if (input_pattern[i] == input_pattern[j]) {
            ++j;
        }

        output_table[i] = j;
    }
}

int main() {
    std::array<uint32_t, kSize> candidate = {};

    kmp_table_kernel(kPattern.data(), candidate.data(), kSize);

    for (uint32_t i = 0; i < kSize; ++i) {
        if (candidate[i] != kExpected[i]) {
            std::puts("FAILED");
            return 1;
        }
    }

    std::printf("kmp_table checksum: %llu\n",
                static_cast<unsigned long long>(checksum(candidate)));
    std::puts("PASSED");
    return 0;
}
