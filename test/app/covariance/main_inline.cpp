// Covariance inline variant migrated from the legacy app corpus.

#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>

namespace {

constexpr uint32_t kSize = 1024;
constexpr float kTolerance = 1e-3f;

void initialize_inputs(std::array<float, kSize> &x, std::array<float, kSize> &y) {
    for (uint32_t i = 0; i < kSize; ++i) {
        x[i] = static_cast<float>(i % 100u);
        y[i] = static_cast<float>((i * 2u) % 100u) + 0.5f;
    }
}

float covariance_ref(const float *x, const float *y, uint32_t size) {
    float sum_x = 0.0f;
    float sum_y = 0.0f;
    for (uint32_t i = 0; i < size; ++i) {
        sum_x += x[i];
        sum_y += y[i];
    }
    const float mean_x = sum_x / static_cast<float>(size);
    const float mean_y = sum_y / static_cast<float>(size);

    float cov = 0.0f;
    for (uint32_t i = 0; i < size; ++i) {
        cov += (x[i] - mean_x) * (y[i] - mean_y);
    }
    return cov / static_cast<float>(size);
}

} // namespace

int main() {
    std::array<float, kSize> x = {};
    std::array<float, kSize> y = {};

    initialize_inputs(x, y);
    const float reference = covariance_ref(x.data(), y.data(), kSize);

    float sum_x = 0.0f;
    float sum_y = 0.0f;
    for (uint32_t i = 0; i < kSize; ++i) {
        sum_x += x[i];
        sum_y += y[i];
    }
    const float mean_x = sum_x / static_cast<float>(kSize);
    const float mean_y = sum_y / static_cast<float>(kSize);

    float cov = 0.0f;
    for (uint32_t i = 0; i < kSize; ++i) {
        cov += (x[i] - mean_x) * (y[i] - mean_y);
    }
    const float candidate = cov / static_cast<float>(kSize);

    if (std::fabs(reference - candidate) > kTolerance) {
        std::puts("FAILED");
        return 1;
    }

    std::printf("covariance result: %.3f\n", candidate);
    std::puts("PASSED");
    return 0;
}
