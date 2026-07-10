
#include <array>
#include <cstdint>
#include <cstdio>

namespace {

constexpr uint32_t kSize = 10;
constexpr std::array<uint32_t, kSize> kExpected = {
    100, 102, 105, 110, 115, 122, 130, 135, 142, 150};

uint64_t checksum(const std::array<uint32_t, kSize> &values) {
    uint64_t sum = 0;
    for (uint32_t i = 0; i < kSize; ++i) {
        sum += static_cast<uint64_t>(i + 1u) * values[i];
    }
    return sum;
}

} // namespace

int main() {
    const std::array<uint32_t, kSize> input = {100, 2, 3, 5, 5,
                                               7,   8, 5, 7, 8};
    std::array<uint32_t, kSize> output = {};

    output[0] = input[0];
    for (uint32_t i = 1; i < kSize; ++i) {
        output[i] = output[i - 1u] + input[i];
    }

    for (uint32_t i = 0; i < kSize; ++i) {
        if (output[i] != kExpected[i]) {
            std::puts("FAILED");
            return 1;
        }
    }

    std::printf("delta_decode checksum: %llu\n",
                static_cast<unsigned long long>(checksum(output)));
    std::puts("PASSED");
    return 0;
}
