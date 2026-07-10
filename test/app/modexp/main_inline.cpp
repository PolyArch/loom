
#include <array>
#include <cstdint>
#include <cstdio>

namespace {

constexpr uint32_t kSize = 8;
constexpr uint32_t kModulus = 1000000007u;
constexpr std::array<uint32_t, kSize> kBase = {
    2u, 3u, 5u, 123u, 65535u, 1000000006u, 314159u, 271828u};
constexpr std::array<uint32_t, kSize> kExponent = {
    3u, 4u, 2u, 7u, 11u, 5u, 13u, 17u};
constexpr std::array<uint32_t, kSize> kExpected = {
    8u, 81u, 25u, 593996258u, 586778098u,
    1000000006u, 154996558u, 89848317u};

uint64_t checksum(const std::array<uint32_t, kSize> &values) {
    uint64_t sum = 0;
    for (uint32_t i = 0; i < kSize; ++i) {
        sum += static_cast<uint64_t>(i + 1u) * values[i];
    }
    return sum;
}

} // namespace

int main() {
    std::array<uint32_t, kSize> output = {};

    for (uint32_t i = 0; i < kSize; ++i) {
        uint64_t result = 1;
        uint64_t base = kBase[i] % kModulus;
        uint32_t exponent = kExponent[i];

        while (exponent > 0u) {
            if ((exponent & 1u) != 0u) {
                result = (result * base) % kModulus;
            }
            base = (base * base) % kModulus;
            exponent >>= 1u;
        }

        output[i] = static_cast<uint32_t>(result);
    }

    for (uint32_t i = 0; i < kSize; ++i) {
        if (output[i] != kExpected[i]) {
            std::puts("FAILED");
            return 1;
        }
    }

    std::printf("modexp checksum: %llu\n",
                static_cast<unsigned long long>(checksum(output)));
    std::puts("PASSED");
    return 0;
}
