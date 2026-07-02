// Fixed-size lower triangular solve inline variant.

#include <array>
#include <cstdint>
#include <cstdio>

namespace {

constexpr uint32_t kSize = 16;
constexpr uint32_t kExpectedTail = 2751;
constexpr std::array<uint32_t, kSize> kExpectedVector = {
    1u,
    0u,
    3u,
    3u,
    static_cast<uint32_t>(-6),
    6u,
    15u,
    static_cast<uint32_t>(-39),
    36u,
    81u,
    static_cast<uint32_t>(-234),
    201u,
    471u,
    static_cast<uint32_t>(-1374),
    1176u,
    2751u,
};

} // namespace

int main() {
    std::array<uint32_t, kSize * kSize> matrix = {};
    std::array<uint32_t, kSize> rhs = {};
    std::array<uint32_t, kSize> actual = {};

    for (uint32_t i = 0; i < kSize; ++i) {
        matrix[i * kSize + i] = 1;
        rhs[i] = i + 1u;
        for (uint32_t j = 0; j < i; ++j) {
            matrix[i * kSize + j] = (i + j + 1) % 3;
        }
    }

    for (uint32_t i = 0; i < kSize; ++i) {
        uint32_t sum = rhs[i];
        for (uint32_t j = 0; j < i; ++j) {
            sum -= matrix[i * kSize + j] * actual[j];
        }
        actual[i] = sum / matrix[i * kSize + i];
    }

    for (uint32_t i = 0; i < kSize; ++i) {
        if (actual[i] != kExpectedVector[i]) {
            std::puts("FAILED");
            return 1;
        }
    }

    if (actual[kSize - 1] != kExpectedTail) {
        std::puts("FAILED");
        return 1;
    }

    std::printf("trsv_lower tail: %u\n", actual[kSize - 1]);
    std::puts("PASSED");
    return 0;
}
