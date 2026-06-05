// Modular exponentiation function variant migrated from the legacy app corpus.

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

extern "C" __attribute__((noinline))
void modexp_kernel(const uint32_t *base_input, const uint32_t *exp_input,
                   uint32_t *output, uint32_t modulus, uint32_t size) {
    for (uint32_t i = 0; i < size; ++i) {
        uint64_t result = 1;
        uint64_t base = base_input[i] % modulus;
        uint32_t exponent = exp_input[i];

        while (exponent > 0u) {
            if ((exponent & 1u) != 0u) {
                result = (result * base) % modulus;
            }
            base = (base * base) % modulus;
            exponent >>= 1u;
        }

        output[i] = static_cast<uint32_t>(result);
    }
}

int main() {
    std::array<uint32_t, kSize> candidate = {};

    modexp_kernel(kBase.data(), kExponent.data(), candidate.data(), kModulus, kSize);

    for (uint32_t i = 0; i < kSize; ++i) {
        if (candidate[i] != kExpected[i]) {
            std::puts("FAILED");
            return 1;
        }
    }

    std::printf("modexp checksum: %llu\n",
                static_cast<unsigned long long>(checksum(candidate)));
    std::puts("PASSED");
    return 0;
}
