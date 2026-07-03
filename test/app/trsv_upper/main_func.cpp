// Fixed-size upper triangular solve migrated from the legacy app corpus.

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

void initialize_u(std::array<uint32_t, kSize * kSize> &matrix) {
    for (uint32_t &value : matrix) {
        value = 0;
    }
    for (uint32_t i = 0; i < kSize; ++i) {
        matrix[i * kSize + i] = 1;
        for (uint32_t j = i + 1; j < kSize; ++j) {
            matrix[i * kSize + j] = (i + j + 1) % 3;
        }
    }
}

void initialize_b(std::array<uint32_t, kSize> &rhs) {
    for (uint32_t i = 0; i < kSize; ++i) {
        rhs[i] = kSize - i;
    }
}

void trsv_upper_ref(const uint32_t *matrix, const uint32_t *rhs, uint32_t *out) {
    for (uint32_t rev = 0; rev < kSize; ++rev) {
        uint32_t row = (kSize - 1u) - rev;
        uint32_t sum = rhs[row];
        for (uint32_t k = 0; k < rev; ++k) {
            uint32_t col = row + 1u + k;
            sum -= matrix[row * kSize + col] * out[col];
        }
        out[row] = sum / matrix[row * kSize + row];
    }
}

} // namespace

extern "C" __attribute__((noinline))
void trsv_upper_kernel(const uint32_t *matrix, const uint32_t *rhs, uint32_t *out) {
    for (uint32_t rev = 0; rev < kSize; ++rev) {
        uint32_t row = (kSize - 1u) - rev;
        uint32_t sum = rhs[row];
        for (uint32_t k = 0; k < rev; ++k) {
            uint32_t col = row + 1u + k;
            sum -= matrix[row * kSize + col] * out[col];
        }
        out[row] = sum / matrix[row * kSize + row];
    }
}

int main() {
    std::array<uint32_t, kSize * kSize> matrix = {};
    std::array<uint32_t, kSize> rhs = {};
    std::array<uint32_t, kSize> expected = {};
    std::array<uint32_t, kSize> actual = {};

    initialize_u(matrix);
    initialize_b(rhs);
    trsv_upper_ref(matrix.data(), rhs.data(), expected.data());
    trsv_upper_kernel(matrix.data(), rhs.data(), actual.data());

    for (uint32_t i = 0; i < kSize; ++i) {
        if (expected[i] != actual[i] || actual[i] != kExpectedVector[i]) {
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
