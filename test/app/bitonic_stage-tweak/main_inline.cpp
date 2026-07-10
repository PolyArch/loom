
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>

namespace {

constexpr uint32_t kSize = 8;
constexpr float kTolerance = 1e-5f;
constexpr std::array<float, kSize> kInput = {
    3.0f, 1.0f, 4.0f, 2.0f, 8.0f, 6.0f, 7.0f, 5.0f};
constexpr std::array<float, kSize> kExpected = {
    1.0f, 2.0f, 2.0f, 3.0f, 8.0f, 5.0f, 7.0f, 4.0f};

double checksum(const std::array<float, kSize> &values) {
    double sum = 0.0;
    for (uint32_t i = 0; i < kSize; ++i) {
        sum += static_cast<double>(i + 1u) * values[i];
    }
    return sum;
}

} // namespace

int main() {
    std::array<float, kSize> inplace = kInput;
    const uint32_t distance = 1u;
    const uint32_t block_size = 4u;

    for (uint32_t i = 0; i < kSize; ++i) {
        const uint32_t block_idx = i / block_size;
        const uint32_t idx_in_block = i % block_size;
        const uint32_t ascending = (block_idx % 2u) == 0u;

        if ((idx_in_block & distance) == 0u) {
            const uint32_t partner = i + distance;
            if (partner < kSize) {
                const float left = inplace[i];
                const float right = inplace[partner];
                const uint32_t should_swap =
                    ascending ? (left > right) : (left < right);
                if (should_swap != 0u) {
                    inplace[i] = right;
                    inplace[partner] = left;
                }
            }
            inplace[i] += 1.0f;
        }
        inplace[i] -= 1.0f;
    }

    for (uint32_t i = 0; i < kSize; ++i) {
        if (std::fabs(inplace[i] - kExpected[i]) > kTolerance) {
            std::puts("FAILED");
            return 1;
        }
    }

    std::printf("bitonic_stage-tweak checksum: %.3f\n", checksum(inplace));
    std::puts("PASSED");
    return 0;
}
