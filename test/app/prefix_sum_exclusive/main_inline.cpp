// Exclusive prefix-sum inline variant migrated from the legacy app corpus.

#include <array>
#include <cstdint>
#include <cstdio>

namespace {

constexpr uint32_t kSize = 1024;

void initialize_input(std::array<uint32_t, kSize> &input) {
    for (uint32_t i = 0; i < kSize; ++i) {
        input[i] = (i % 10u) + 1u;
    }
}

void prefix_sum_exclusive_ref(const uint32_t *input, uint32_t *output, uint32_t size) {
    if (size == 0) {
        return;
    }

    output[0] = 0;
    for (uint32_t i = 1; i < size; ++i) {
        output[i] = output[i - 1] + input[i - 1];
    }
}

uint64_t checksum(const std::array<uint32_t, kSize> &values) {
    uint64_t sum = 0;
    for (uint32_t i = 0; i < kSize; ++i) {
        sum += static_cast<uint64_t>(i + 1u) * static_cast<uint64_t>(values[i]);
    }
    return sum;
}

} // namespace

int main() {
    std::array<uint32_t, kSize> input = {};
    std::array<uint32_t, kSize> reference = {};
    std::array<uint32_t, kSize> candidate = {};

    initialize_input(input);
    prefix_sum_exclusive_ref(input.data(), reference.data(), kSize);

    candidate[0] = 0;
    for (uint32_t i = 1; i < kSize; ++i) {
        candidate[i] = candidate[i - 1] + input[i - 1];
    }

    for (uint32_t i = 0; i < kSize; ++i) {
        if (reference[i] != candidate[i]) {
            std::puts("FAILED");
            return 1;
        }
    }

    std::printf("prefix_sum_exclusive checksum: %llu\n",
                static_cast<unsigned long long>(checksum(candidate)));
    std::puts("PASSED");
    return 0;
}
