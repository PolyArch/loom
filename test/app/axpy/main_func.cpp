// AXPY function variant migrated from the legacy app corpus.

#include <array>
#include <cstdint>
#include <cstdio>

namespace {

constexpr uint32_t kSize = 8;
constexpr uint32_t kAlpha = 3;

void axpy_ref(const uint32_t *x, const uint32_t *y, uint32_t *out,
              uint32_t alpha, uint32_t size) {
    for (uint32_t i = 0; i < size; ++i) {
        out[i] = alpha * x[i] + y[i];
    }
}

__attribute__((noinline))
void axpy_candidate(const uint32_t *x, const uint32_t *y, uint32_t *out,
                    uint32_t alpha, uint32_t size) {
    for (uint32_t i = 0; i < size; ++i) {
        out[i] = alpha * x[i] + y[i];
    }
}

bool same(const std::array<uint32_t, kSize> &lhs,
          const std::array<uint32_t, kSize> &rhs) {
    for (uint32_t i = 0; i < kSize; ++i) {
        if (lhs[i] != rhs[i]) {
            return false;
        }
    }
    return true;
}

uint32_t checksum(const std::array<uint32_t, kSize> &values) {
    uint32_t sum = 0;
    for (uint32_t value : values) {
        sum += value;
    }
    return sum;
}

} // namespace

int main() {
    const std::array<uint32_t, kSize> x = {1, 2, 3, 4, 5, 6, 7, 8};
    const std::array<uint32_t, kSize> y = {10, 20, 30, 40, 50, 60, 70, 80};
    std::array<uint32_t, kSize> reference = {};
    std::array<uint32_t, kSize> candidate = {};

    axpy_ref(x.data(), y.data(), reference.data(), kAlpha, kSize);
    axpy_candidate(x.data(), y.data(), candidate.data(), kAlpha, kSize);

    if (!same(reference, candidate)) {
        std::puts("FAILED");
        return 1;
    }

    std::printf("AXPY checksum: %u\n", checksum(candidate));
    std::puts("PASSED");
    return 0;
}

