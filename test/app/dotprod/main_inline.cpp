// Two-stage dot product inline variant.

#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>

namespace {

constexpr uint32_t kSize = 8;
constexpr float kTolerance = 1e-5f;
constexpr std::array<float, kSize> kInputA = {
    1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
constexpr std::array<float, kSize> kInputB = {
    0.5f, 1.0f, 1.5f, 2.0f, 2.5f, 3.0f, 3.5f, 4.0f};

double product_checksum(const std::array<float, kSize> &values) {
    double sum = 0.0;
    for (uint32_t i = 0; i < kSize; ++i) {
        sum += static_cast<double>(i + 1u) * values[i];
    }
    return sum;
}

} // namespace

int main() {
    std::array<float, kSize> products = {};

    for (uint32_t i = 0; i < kSize; ++i) {
        products[i] = kInputA[i] * kInputB[i];
    }

    float result = 0.0f;
    for (uint32_t i = 0; i < kSize; ++i) {
        result += products[i];
    }

    if (std::fabs(result - 102.0f) > kTolerance ||
        std::fabs(product_checksum(products) - 648.0) > kTolerance) {
        std::puts("FAILED");
        return 1;
    }

    std::printf("dotprod result: %.3f products: %.3f\n",
                static_cast<double>(result), product_checksum(products));
    std::puts("PASSED");
    return 0;
}
