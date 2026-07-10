
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

extern "C" __attribute__((noinline))
void modmul_kernel(const uint32_t *input_a, const uint32_t *input_b,
                   uint32_t *output, uint32_t modulus, uint32_t size) {
    for (uint32_t i = 0; i < size; ++i) {
        const uint64_t a = input_a[i];
        const uint64_t b = input_b[i];
        output[i] = static_cast<uint32_t>((a * b) % modulus);
    }
}

int main() {
    std::array<uint32_t, kSize> candidate = {};

    modmul_kernel(kInputA.data(), kInputB.data(), candidate.data(), kModulus, kSize);

    for (uint32_t i = 0; i < kSize; ++i) {
        if (candidate[i] != kExpected[i]) {
            std::puts("FAILED");
            return 1;
        }
    }

    std::printf("modmul checksum: %llu\n",
                static_cast<unsigned long long>(checksum(candidate)));
    std::puts("PASSED");
    return 0;
}
