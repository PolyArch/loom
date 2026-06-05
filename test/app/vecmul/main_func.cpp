// Vector-multiply function variant migrated from the legacy app corpus.

#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>

namespace {

constexpr uint32_t kSize = 16;

void vecmul_ref(const float *lhs, const float *rhs, float *out,
                uint32_t size) {
    for (uint32_t i = 0; i < size; ++i) {
        out[i] = lhs[i] * rhs[i];
    }
}

__attribute__((noinline))
void vecmul_candidate(const float *lhs, const float *rhs, float *out,
                      uint32_t size) {
    for (uint32_t i = 0; i < size; ++i) {
        out[i] = lhs[i] * rhs[i];
    }
}

float checksum(const std::array<float, kSize> &values) {
    float sum = 0.0f;
    for (float value : values) {
        sum += value;
    }
    return sum;
}

bool same(const std::array<float, kSize> &lhs,
          const std::array<float, kSize> &rhs) {
    for (uint32_t i = 0; i < kSize; ++i) {
        if (std::fabs(lhs[i] - rhs[i]) > 1e-5f) {
            return false;
        }
    }
    return true;
}

} // namespace

int main() {
    std::array<float, kSize> lhs = {};
    std::array<float, kSize> rhs = {};
    std::array<float, kSize> reference = {};
    std::array<float, kSize> candidate = {};

    for (uint32_t i = 0; i < kSize; ++i) {
        lhs[i] = static_cast<float>(i + 1);
        rhs[i] = 0.5f * static_cast<float>(i + 1);
    }

    vecmul_ref(lhs.data(), rhs.data(), reference.data(), kSize);
    vecmul_candidate(lhs.data(), rhs.data(), candidate.data(), kSize);

    if (!same(reference, candidate)) {
        std::puts("FAILED");
        return 1;
    }

    std::printf("vecmul checksum: %.6f\n", checksum(candidate));
    std::puts("PASSED");
    return 0;
}
