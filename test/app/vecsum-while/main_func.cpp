
#include <array>
#include <cstdint>
#include <cstdio>

namespace {

constexpr uint32_t kSize = 16;
constexpr uint32_t kInitial = 0;

void initialize_input(std::array<uint32_t, kSize> &input) {
    for (uint32_t i = 0; i < kSize; ++i) {
        input[i] = i;
    }
}

uint32_t vecsum_ref(const uint32_t *input, uint32_t initial, uint32_t size) {
    uint32_t sum = initial;
    for (uint32_t i = 0; i < size; ++i) {
        sum += input[i];
    }
    return sum;
}

} // namespace

extern "C" __attribute__((noinline))
uint32_t vecsum_while_kernel(const uint32_t *input, uint32_t initial, uint32_t size) {
    uint32_t sum = initial;
    for (uint32_t i = 0; i < size; ++i) {
        sum += input[i];
    }
    return sum;
}

int main() {
    std::array<uint32_t, kSize> input = {};
    initialize_input(input);

    const uint32_t reference = vecsum_ref(input.data(), kInitial, kSize);
    const uint32_t candidate = vecsum_while_kernel(input.data(), kInitial, kSize);
    if (reference != candidate) {
        std::puts("FAILED");
        return 1;
    }

    std::printf("vecsum-while result: %u\n", candidate);
    std::puts("PASSED");
    return 0;
}
