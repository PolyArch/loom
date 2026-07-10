
#include <array>
#include <cstdint>
#include <cstdio>

namespace {

constexpr uint32_t kSize = 8;
constexpr uint32_t kModulus = 1000000007u;
constexpr std::array<uint32_t, kSize> kInputA = {
    12345u, 24690u, 987654321u, 42u, 65535u, 1000000006u, 314159u, 271828u};
constexpr std::array<uint32_t, kSize> kInputB = {
    67890u, 13579u, 123456789u, 99u, 65537u, 1000000006u, 271828u, 314159u};
constexpr std::array<uint32_t, kSize> kExpected = {
    838102050u, 335265510u, 259106859u, 4158u,
    294967267u, 1u, 397212057u, 397212057u};

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
        const uint64_t a = kInputA[i];
        const uint64_t b = kInputB[i];
        output[i] = static_cast<uint32_t>((a * b) % kModulus);
    }

    for (uint32_t i = 0; i < kSize; ++i) {
        if (output[i] != kExpected[i]) {
            std::puts("FAILED");
            return 1;
        }
    }

    std::printf("modmul checksum: %llu\n",
                static_cast<unsigned long long>(checksum(output)));
    std::puts("PASSED");
    return 0;
}
