// Dense matrix multiplication function variant.

#include <array>
#include <cstdint>
#include <cstdio>

namespace {

constexpr uint32_t kRows = 2;
constexpr uint32_t kInner = 3;
constexpr uint32_t kCols = 2;
constexpr std::array<uint32_t, kRows * kInner> kInputA = {
    1u, 2u, 3u,
    4u, 5u, 6u};
constexpr std::array<uint32_t, kInner * kCols> kInputB = {
    7u, 8u,
    9u, 10u,
    11u, 12u};
constexpr std::array<uint32_t, kRows * kCols> kExpected = {
    58u, 64u, 139u, 154u};

uint64_t checksum(const std::array<uint32_t, kRows * kCols> &values) {
    uint64_t sum = 0;
    for (uint32_t i = 0; i < values.size(); ++i) {
        sum += static_cast<uint64_t>(i + 1u) * values[i];
    }
    return sum;
}

} // namespace

extern "C" __attribute__((noinline))
void matmul_kernel(const uint32_t *a, const uint32_t *b, uint32_t *c,
                   uint32_t rows, uint32_t inner, uint32_t cols) {
    for (uint32_t i = 0; i < rows; ++i) {
        for (uint32_t j = 0; j < cols; ++j) {
            uint32_t sum = 0;
            for (uint32_t k = 0; k < inner; ++k) {
                sum += a[i * inner + k] * b[k * cols + j];
            }
            c[i * cols + j] = sum;
        }
    }
}

int main() {
    std::array<uint32_t, kRows * kCols> candidate = {};

    matmul_kernel(kInputA.data(), kInputB.data(), candidate.data(),
                  kRows, kInner, kCols);

    for (uint32_t i = 0; i < candidate.size(); ++i) {
        if (candidate[i] != kExpected[i]) {
            std::puts("FAILED");
            return 1;
        }
    }

    std::printf("matmul checksum: %llu\n",
                static_cast<unsigned long long>(checksum(candidate)));
    std::puts("PASSED");
    return 0;
}
