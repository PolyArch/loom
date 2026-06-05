// Sparse matrix times sparse matrix inline variant.

#include <array>
#include <cstdint>
#include <cstdio>

namespace {

constexpr uint32_t kRows = 3;
constexpr uint32_t kInner = 4;
constexpr uint32_t kCols = 3;
constexpr uint32_t kANnz = 6;
constexpr uint32_t kBNnz = 6;
constexpr uint32_t kMaxC = kRows * kCols;
constexpr std::array<uint32_t, kANnz> kAValues = {
    2u, 1u, 3u, 2u, 1u, 4u};
constexpr std::array<uint32_t, kANnz> kAColIndices = {
    0u, 2u, 1u, 3u, 0u, 3u};
constexpr std::array<uint32_t, kRows + 1> kARowPtr = {
    0u, 2u, 4u, 6u};
constexpr std::array<uint32_t, kBNnz> kBValues = {
    1u, 2u, 5u, 3u, 1u, 2u};
constexpr std::array<uint32_t, kBNnz> kBColIndices = {
    0u, 2u, 1u, 0u, 2u, 1u};
constexpr std::array<uint32_t, kInner + 1> kBRowPtr = {
    0u, 2u, 3u, 5u, 6u};
constexpr std::array<uint32_t, 6> kExpectedValues = {
    5u, 5u, 19u, 1u, 8u, 2u};
constexpr std::array<uint32_t, 6> kExpectedColIndices = {
    0u, 2u, 1u, 0u, 1u, 2u};
constexpr std::array<uint32_t, kRows + 1> kExpectedRowPtr = {
    0u, 2u, 3u, 6u};

template <size_t N>
uint64_t checksum(const std::array<uint32_t, N> &values) {
    uint64_t sum = 0;
    for (uint32_t i = 0; i < N; ++i) {
        sum += static_cast<uint64_t>(i + 1u) * values[i];
    }
    return sum;
}

} // namespace

int main() {
    std::array<uint32_t, kMaxC> c_values = {};
    std::array<uint32_t, kMaxC> c_col_indices = {};
    std::array<uint32_t, kRows + 1> c_row_ptr = {};
    std::array<uint32_t, kCols> temp_row = {};
    uint32_t nnz = 0;
    c_row_ptr[0] = 0;

    for (uint32_t row = 0; row < kRows; ++row) {
        for (uint32_t col = 0; col < kCols; ++col) {
            temp_row[col] = 0;
        }

        const uint32_t a_begin = kARowPtr[row];
        const uint32_t a_end = kARowPtr[row + 1u];
        for (uint32_t a_idx = a_begin; a_idx < a_end; ++a_idx) {
            const uint32_t a_value = kAValues[a_idx];
            const uint32_t a_col = kAColIndices[a_idx];
            const uint32_t b_begin = kBRowPtr[a_col];
            const uint32_t b_end = kBRowPtr[a_col + 1u];
            for (uint32_t b_idx = b_begin; b_idx < b_end; ++b_idx) {
                const uint32_t b_col = kBColIndices[b_idx];
                temp_row[b_col] += a_value * kBValues[b_idx];
            }
        }

        for (uint32_t col = 0; col < kCols; ++col) {
            if (temp_row[col] != 0u) {
                c_values[nnz] = temp_row[col];
                c_col_indices[nnz] = col;
                ++nnz;
            }
        }
        c_row_ptr[row + 1u] = nnz;
    }

    for (uint32_t i = 0; i < kExpectedValues.size(); ++i) {
        if (c_values[i] != kExpectedValues[i] ||
            c_col_indices[i] != kExpectedColIndices[i]) {
            std::puts("FAILED");
            return 1;
        }
    }
    for (uint32_t i = 0; i < kRows + 1u; ++i) {
        if (c_row_ptr[i] != kExpectedRowPtr[i]) {
            std::puts("FAILED");
            return 1;
        }
    }

    std::printf("spmspm checksums: %llu %llu %llu\n",
                static_cast<unsigned long long>(checksum(kExpectedValues)),
                static_cast<unsigned long long>(checksum(kExpectedColIndices)),
                static_cast<unsigned long long>(checksum(c_row_ptr)));
    std::puts("PASSED");
    return 0;
}
