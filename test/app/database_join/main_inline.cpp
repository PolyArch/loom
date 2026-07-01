// Integer nested-loop join inline variant.

#include <array>
#include <cstdint>
#include <cstdio>

namespace {

constexpr uint32_t kSizeA = 3;
constexpr uint32_t kSizeB = 3;
constexpr uint32_t kMaxOutput = 5;

} // namespace

int main() {
    std::array<int32_t, kSizeA> a_ids = {1, 2, 3};
    std::array<int32_t, kSizeB> b_ids = {2, 3, 4};
    std::array<int32_t, kSizeA> a_values = {10, 20, 30};
    std::array<int32_t, kSizeB> b_values = {200, 300, 400};
    std::array<int32_t, kMaxOutput> output_ids = {};
    std::array<int32_t, kMaxOutput> output_a_values = {};
    std::array<int32_t, kMaxOutput> output_b_values = {};

    uint32_t out_idx = 0;
    for (uint32_t i = 0; i < kSizeA; ++i) {
        for (uint32_t j = 0; j < kSizeB; ++j) {
            if (a_ids[i] == b_ids[j]) {
                output_ids[out_idx] = a_ids[i];
                output_a_values[out_idx] = a_values[i];
                output_b_values[out_idx] = b_values[j];
                ++out_idx;
            }
        }
    }

    constexpr std::array<int32_t, kMaxOutput> kExpectedIds = {2, 3, 0, 0, 0};
    constexpr std::array<int32_t, kMaxOutput> kExpectedAValues = {20, 30, 0, 0, 0};
    constexpr std::array<int32_t, kMaxOutput> kExpectedBValues = {200, 300, 0, 0, 0};
    if (out_idx != 2) {
        std::puts("FAILED");
        return 1;
    }
    for (uint32_t i = 0; i < kMaxOutput; ++i) {
        if (output_ids[i] != kExpectedIds[i] ||
            output_a_values[i] != kExpectedAValues[i] ||
            output_b_values[i] != kExpectedBValues[i]) {
            std::puts("FAILED");
            return 1;
        }
    }

    std::printf("database_join count: %u\n", out_idx);
    std::puts("PASSED");
    return 0;
}
