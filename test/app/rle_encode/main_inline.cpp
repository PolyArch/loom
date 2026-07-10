
#include <array>
#include <cstdint>
#include <cstdio>

namespace {

constexpr uint32_t kSize = 20;
constexpr uint32_t kRuns = 7;
constexpr std::array<uint32_t, kRuns> kExpectedValues = {1, 2, 3, 4, 5, 6, 7};
constexpr std::array<uint32_t, kRuns> kExpectedCounts = {3, 2, 4, 5, 1, 3, 2};

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

int main() {
    const std::array<uint32_t, kSize> input = {1, 1, 1, 2, 2, 3, 3,
                                               3, 3, 4, 4, 4, 4, 4,
                                               5, 6, 6, 6, 7, 7};
    std::array<uint32_t, kSize> values = {};
    std::array<uint32_t, kSize> counts = {};

    uint32_t write = 0;
    uint32_t current = input[0];
    uint32_t count = 1;
    for (uint32_t i = 1; i < kSize; ++i) {
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
    const uint32_t length = write + 1u;

    if (length != kRuns) {
        std::puts("FAILED");
        return 1;
    }
    for (uint32_t i = 0; i < kRuns; ++i) {
        if (values[i] != kExpectedValues[i] || counts[i] != kExpectedCounts[i]) {
            std::puts("FAILED");
            return 1;
        }
    }

    std::printf("rle_encode checksum: %llu\n",
                static_cast<unsigned long long>(checksum(values, counts, length)));
    std::puts("PASSED");
    return 0;
}
