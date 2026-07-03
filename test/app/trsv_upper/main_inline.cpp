// Fixed-size upper triangular solve inline variant.

#include <array>
#include <cstdint>
#include <cstdio>

namespace {

constexpr uint32_t kSize = 16;
constexpr uint32_t kExpectedHead = 4451;
constexpr std::array<uint32_t, kSize> kExpectedVector = {
    4451u,
    static_cast<uint32_t>(-2224),
    326u,
    761u,
    static_cast<uint32_t>(-379),
    56u,
    131u,
    static_cast<uint32_t>(-64),
    11u,
    23u,
    static_cast<uint32_t>(-10),
    2u,
    5u,
    static_cast<uint32_t>(-1),
    2u,
    1u,
};

} // namespace

int main() {
    std::array<uint32_t, kSize * kSize> matrix = {};
    std::array<uint32_t, kSize> rhs = {};
    std::array<uint32_t, kSize> actual = {};

    for (uint32_t i = 0; i < kSize; ++i) {
        matrix[i * kSize + i] = 1;
        rhs[i] = kSize - i;
        for (uint32_t j = i + 1; j < kSize; ++j) {
            matrix[i * kSize + j] = (i + j + 1) % 3;
        }
    }

    for (uint32_t rev = 0; rev < kSize; ++rev) {
        uint32_t row = (kSize - 1u) - rev;
        uint32_t sum = rhs[row];
        for (uint32_t k = 0; k < rev; ++k) {
            uint32_t col = row + 1u + k;
            sum -= matrix[row * kSize + col] * actual[col];
        }
        actual[row] = sum / matrix[row * kSize + row];
    }

    for (uint32_t i = 0; i < kSize; ++i) {
        if (actual[i] != kExpectedVector[i]) {
            std::puts("FAILED");
            return 1;
        }
    }

    if (actual[0] != kExpectedHead) {
        std::puts("FAILED");
        return 1;
    }

    std::printf("trsv_upper head: %u\n", actual[0]);
    std::puts("PASSED");
    return 0;
}
