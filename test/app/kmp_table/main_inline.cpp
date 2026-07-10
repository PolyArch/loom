
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

int main() {
    std::array<uint32_t, kSize> candidate = {};

    candidate[0] = 0;
    uint32_t j = 0;
    for (uint32_t i = 1; i < kSize; ++i) {
        while (j > 0 && kPattern[i] != kPattern[j]) {
            j = candidate[j - 1u];
        }

        if (kPattern[i] == kPattern[j]) {
            ++j;
        }

        candidate[i] = j;
    }

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
