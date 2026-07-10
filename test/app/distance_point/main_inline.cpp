
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>

namespace {

constexpr uint32_t kCount = 16;
constexpr uint32_t kPointElems = kCount * 3u;
constexpr double kTolerance = 1e-3;

void distance_ref(const float *a, const float *b, float *out, uint32_t count) {
    for (uint32_t i = 0; i < count; ++i) {
        const float dx = a[i * 3u + 0u] - b[i * 3u + 0u];
        const float dy = a[i * 3u + 1u] - b[i * 3u + 1u];
        const float dz = a[i * 3u + 2u] - b[i * 3u + 2u];
        out[i] = std::sqrt(dx * dx + dy * dy + dz * dz);
    }
}

double checksum(const std::array<float, kCount> &values) {
    double sum = 0.0;
    for (uint32_t i = 0; i < kCount; ++i) {
        sum += static_cast<double>(i + 1u) * values[i];
    }
    return sum;
}

} // namespace

int main() {
    std::array<float, kPointElems> a = {};
    std::array<float, kPointElems> b = {};
    std::array<float, kCount> reference = {};
    std::array<float, kCount> candidate = {};

    for (uint32_t i = 0; i < kCount; ++i) {
        a[i * 3u + 0u] = 0.25f * static_cast<float>(i);
        a[i * 3u + 1u] = -0.5f + 0.125f * static_cast<float>(i);
        a[i * 3u + 2u] = 1.0f + 0.0625f * static_cast<float>(i);

        b[i * 3u + 0u] = 3.0f + 0.1875f * static_cast<float>(i);
        b[i * 3u + 1u] = 4.0f - 0.25f * static_cast<float>(i);
        b[i * 3u + 2u] = -2.0f + 0.3125f * static_cast<float>(i);
    }

    distance_ref(a.data(), b.data(), reference.data(), kCount);
    for (uint32_t i = 0; i < kCount; ++i) {
        const float dx = a[i * 3u + 0u] - b[i * 3u + 0u];
        const float dy = a[i * 3u + 1u] - b[i * 3u + 1u];
        const float dz = a[i * 3u + 2u] - b[i * 3u + 2u];
        candidate[i] = std::sqrt(dx * dx + dy * dy + dz * dz);
    }

    for (uint32_t i = 0; i < kCount; ++i) {
        if (std::fabs(reference[i] - candidate[i]) > kTolerance) {
            std::puts("FAILED");
            return 1;
        }
    }

    const double actual = checksum(candidate);
    std::printf("distance_point checksum: %.3f\n", actual);
    std::puts("PASSED");
    return 0;
}
