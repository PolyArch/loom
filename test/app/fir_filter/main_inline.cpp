// FIR-filter inline variant migrated from the legacy app corpus.

#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>

namespace {

constexpr uint32_t kInputSize = 8;
constexpr uint32_t kTapCount = 3;

float checksum(const std::array<float, kInputSize> &values) {
    float sum = 0.0f;
    for (float value : values) {
        sum += value;
    }
    return sum;
}

bool same(const std::array<float, kInputSize> &lhs,
          const std::array<float, kInputSize> &rhs) {
    for (uint32_t i = 0; i < kInputSize; ++i) {
        if (std::fabs(lhs[i] - rhs[i]) > 1e-5f) {
            return false;
        }
    }
    return true;
}

} // namespace

int main() {
    const std::array<float, kInputSize> input = {
        1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f,
    };
    const std::array<float, kTapCount> coeffs = {0.25f, 0.5f, 0.25f};
    std::array<float, kInputSize> reference = {};
    std::array<float, kInputSize> candidate = {};

    for (uint32_t n = 0; n < kInputSize; ++n) {
        float sum = 0.0f;
        for (uint32_t k = 0; k < kTapCount; ++k) {
            int32_t idx = static_cast<int32_t>(n) - static_cast<int32_t>(k);
            if (idx >= 0) {
                sum += coeffs[k] * input[idx];
            }
        }
        reference[n] = sum;
    }
    for (uint32_t n = 0; n < kInputSize; ++n) {
        float sum = 0.0f;
        for (uint32_t k = 0; k < kTapCount; ++k) {
            int32_t idx = static_cast<int32_t>(n) - static_cast<int32_t>(k);
            if (idx >= 0) {
                sum += coeffs[k] * input[idx];
            }
        }
        candidate[n] = sum;
    }

    if (!same(reference, candidate)) {
        std::puts("FAILED");
        return 1;
    }

    std::printf("fir_filter checksum: %.6f\n", checksum(candidate));
    std::puts("PASSED");
    return 0;
}
