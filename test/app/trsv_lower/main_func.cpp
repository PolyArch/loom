
#include <array>
#include <cstdint>
#include <cstdio>

namespace {

constexpr uint32_t kSize = 16;
constexpr uint32_t kExpectedTail = 2751;

void initialize_l(std::array<uint32_t, kSize * kSize> &matrix) {
    for (uint32_t &value : matrix) {
        value = 0;
    }
    for (uint32_t i = 0; i < kSize; ++i) {
        matrix[i * kSize + i] = 1;
        for (uint32_t j = 0; j < i; ++j) {
            matrix[i * kSize + j] = (i + j + 1) % 3;
        }
    }
}

void initialize_b(std::array<uint32_t, kSize> &rhs) {
    for (uint32_t i = 0; i < kSize; ++i) {
        rhs[i] = i + 1u;
    }
}

void trsv_lower_ref(const uint32_t *matrix, const uint32_t *rhs, uint32_t *out) {
    for (uint32_t i = 0; i < kSize; ++i) {
        uint32_t sum = rhs[i];
        for (uint32_t j = 0; j < i; ++j) {
            sum -= matrix[i * kSize + j] * out[j];
        }
        out[i] = sum / matrix[i * kSize + i];
    }
}

} // namespace

extern "C" __attribute__((noinline))
void trsv_lower_kernel(const uint32_t *matrix, const uint32_t *rhs, uint32_t *out) {
    for (uint32_t i = 0; i < kSize; ++i) {
        uint32_t sum = rhs[i];
        for (uint32_t j = 0; j < i; ++j) {
            sum -= matrix[i * kSize + j] * out[j];
        }
        out[i] = sum / matrix[i * kSize + i];
    }
}

int main() {
    std::array<uint32_t, kSize * kSize> matrix = {};
    std::array<uint32_t, kSize> rhs = {};
    std::array<uint32_t, kSize> expected = {};
    std::array<uint32_t, kSize> actual = {};

    initialize_l(matrix);
    initialize_b(rhs);
    trsv_lower_ref(matrix.data(), rhs.data(), expected.data());
    trsv_lower_kernel(matrix.data(), rhs.data(), actual.data());

    for (uint32_t i = 0; i < kSize; ++i) {
        if (expected[i] != actual[i]) {
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
