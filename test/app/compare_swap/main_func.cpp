// Compare-swap function variant migrated from the legacy app corpus.

#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>

namespace {

constexpr uint32_t kSize = 16;

void compare_swap_ref(const float *a, const float *b, float *min_out,
                      float *max_out, uint32_t size) {
    for (uint32_t i = 0; i < size; ++i) {
        if (a[i] <= b[i]) {
            min_out[i] = a[i];
            max_out[i] = b[i];
        } else {
            min_out[i] = b[i];
            max_out[i] = a[i];
        }
    }
}

__attribute__((noinline))
void compare_swap_candidate(const float *a, const float *b, float *min_out,
                            float *max_out, uint32_t size) {
    for (uint32_t i = 0; i < size; ++i) {
        if (a[i] <= b[i]) {
            min_out[i] = a[i];
            max_out[i] = b[i];
        } else {
            min_out[i] = b[i];
            max_out[i] = a[i];
        }
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
    const std::array<float, kSize> a = {
        5.0f, 2.0f, 8.0f, 1.0f, 9.0f, 3.0f, 7.0f, 4.0f,
        6.0f, 10.0f, 15.0f, 12.0f, 11.0f, 14.0f, 13.0f, 16.0f,
    };
    const std::array<float, kSize> b = {
        3.0f, 7.0f, 1.0f, 9.0f, 2.0f, 8.0f, 4.0f, 6.0f,
        10.0f, 5.0f, 12.0f, 15.0f, 14.0f, 11.0f, 16.0f, 13.0f,
    };
    std::array<float, kSize> ref_min = {};
    std::array<float, kSize> ref_max = {};
    std::array<float, kSize> cand_min = {};
    std::array<float, kSize> cand_max = {};

    compare_swap_ref(a.data(), b.data(), ref_min.data(), ref_max.data(), kSize);
    compare_swap_candidate(a.data(), b.data(), cand_min.data(), cand_max.data(), kSize);

    if (!same(ref_min, cand_min) || !same(ref_max, cand_max)) {
        std::puts("FAILED");
        return 1;
    }

    std::printf("compare_swap checksums: %.6f %.6f\n",
                checksum(cand_min), checksum(cand_max));
    std::puts("PASSED");
    return 0;
}
