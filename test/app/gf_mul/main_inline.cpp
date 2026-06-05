// GF(2^8) multiplication inline variant migrated from the legacy app corpus.

#include <array>
#include <cstdint>
#include <cstdio>

namespace {

constexpr uint32_t kSize = 8;
constexpr std::array<uint32_t, kSize> kInputA = {
    0x57u, 0x83u, 0x01u, 0xffu, 0x53u, 0xcau, 0x10u, 0xaeu};
constexpr std::array<uint32_t, kSize> kInputB = {
    0x83u, 0x57u, 0x05u, 0x13u, 0xcau, 0x53u, 0x20u, 0x02u};
constexpr std::array<uint32_t, kSize> kExpected = {
    193u, 193u, 5u, 115u, 1u, 1u, 54u, 71u};

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
        uint32_t a = kInputA[i] & 0xffu;
        uint32_t b = kInputB[i] & 0xffu;
        uint32_t product = 0;

        for (uint32_t bit = 0; bit < 8u; ++bit) {
            if ((b & 1u) != 0u) {
                product ^= a;
            }
            const uint32_t high_bit = a & 0x80u;
            a <<= 1u;
            if (high_bit != 0u) {
                a ^= 0x1bu;
            }
            b >>= 1u;
        }

        output[i] = product & 0xffu;
    }

    for (uint32_t i = 0; i < kSize; ++i) {
        if (output[i] != kExpected[i]) {
            std::puts("FAILED");
            return 1;
        }
    }

    std::printf("gf_mul checksum: %llu\n",
                static_cast<unsigned long long>(checksum(output)));
    std::puts("PASSED");
    return 0;
}
