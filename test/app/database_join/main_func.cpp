// Integer nested-loop join function variant.

#include <array>
#include <cstdint>
#include <cstdio>

namespace {

constexpr uint32_t kSizeA = 3;
constexpr uint32_t kSizeB = 3;
constexpr uint32_t kMaxOutput = 5;

uint32_t database_join_ref(const int32_t *a_ids, const int32_t *b_ids,
                           const int32_t *a_values, const int32_t *b_values,
                           int32_t *output_ids, int32_t *output_a_values,
                           int32_t *output_b_values, uint32_t size_a,
                           uint32_t size_b) {
    uint32_t out_idx = 0;
    for (uint32_t i = 0; i < size_a; ++i) {
        for (uint32_t j = 0; j < size_b; ++j) {
            if (a_ids[i] == b_ids[j]) {
                output_ids[out_idx] = a_ids[i];
                output_a_values[out_idx] = a_values[i];
                output_b_values[out_idx] = b_values[j];
                ++out_idx;
            }
        }
    }
    return out_idx;
}

} // namespace

extern "C" __attribute__((noinline))
uint32_t database_join_kernel(const int32_t *a_ids, const int32_t *b_ids,
                              const int32_t *a_values, const int32_t *b_values,
                              int32_t *output_ids, int32_t *output_a_values,
                              int32_t *output_b_values, uint32_t size_a,
                              uint32_t size_b) {
    uint32_t out_idx = 0;
    for (uint32_t i = 0; i < size_a; ++i) {
        for (uint32_t j = 0; j < size_b; ++j) {
            if (a_ids[i] == b_ids[j]) {
                output_ids[out_idx] = a_ids[i];
                output_a_values[out_idx] = a_values[i];
                output_b_values[out_idx] = b_values[j];
                ++out_idx;
            }
        }
    }
    return out_idx;
}

int main() {
    std::array<int32_t, kSizeA> a_ids = {1, 2, 3};
    std::array<int32_t, kSizeB> b_ids = {2, 3, 4};
    std::array<int32_t, kSizeA> a_values = {10, 20, 30};
    std::array<int32_t, kSizeB> b_values = {200, 300, 400};
    std::array<int32_t, kMaxOutput> expect_ids = {};
    std::array<int32_t, kMaxOutput> expect_a_values = {};
    std::array<int32_t, kMaxOutput> expect_b_values = {};
    std::array<int32_t, kMaxOutput> actual_ids = {};
    std::array<int32_t, kMaxOutput> actual_a_values = {};
    std::array<int32_t, kMaxOutput> actual_b_values = {};

    uint32_t expect_count = database_join_ref(
        a_ids.data(), b_ids.data(), a_values.data(), b_values.data(),
        expect_ids.data(), expect_a_values.data(), expect_b_values.data(),
        kSizeA, kSizeB);
    uint32_t actual_count = database_join_kernel(
        a_ids.data(), b_ids.data(), a_values.data(), b_values.data(),
        actual_ids.data(), actual_a_values.data(), actual_b_values.data(),
        kSizeA, kSizeB);
    if (expect_count != actual_count) {
        std::puts("FAILED");
        return 1;
    }
    for (uint32_t i = 0; i < expect_count; ++i) {
        if (expect_ids[i] != actual_ids[i] ||
            expect_a_values[i] != actual_a_values[i] ||
            expect_b_values[i] != actual_b_values[i]) {
            std::puts("FAILED");
            return 1;
        }
    }

    std::printf("database_join count: %u\n", actual_count);
    std::puts("PASSED");
    return 0;
}
