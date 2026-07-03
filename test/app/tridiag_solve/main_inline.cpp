#include <array>
#include <cmath>
#include <cstdio>
#include <cstdint>

namespace {

constexpr uint32_t kSize = 8;
constexpr std::array<float, kSize> kInputA = {
    0.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f};
constexpr std::array<float, kSize> kInputB = {
    4.0f, 4.0f, 4.0f, 4.0f, 4.0f, 4.0f, 4.0f, 4.0f};
constexpr std::array<float, kSize> kInputC = {
    -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, 0.0f};
constexpr std::array<float, kSize> kInputD = {
    1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
constexpr std::array<float, kSize> kExpectedX = {
    0.499889012f, 0.999556049f, 1.498335183f, 1.993784684f,
    2.476803552f, 2.913429523f, 3.176914539f, 2.794228635f};
constexpr std::array<float, kSize> kExpectedCPrime = {
    -0.250000000f, -0.266666667f, -0.267857143f, -0.267942584f,
    -0.267948718f, -0.267949158f, -0.267949190f, 0.000000000f};
constexpr std::array<float, kSize> kExpectedDPrime = {
    0.250000000f, 0.600000000f, 0.964285714f, 1.330143541f,
    1.696153846f, 2.062177946f, 2.428203240f, 2.794228635f};

bool close(float lhs, float rhs) {
    return std::fabs(lhs - rhs) <= 1.0e-5f;
}

bool equal_array(const std::array<float, kSize> &lhs,
                 const std::array<float, kSize> &rhs) {
    for (uint32_t i = 0; i < kSize; ++i) {
        if (!close(lhs[i], rhs[i])) {
            return false;
        }
    }
    return true;
}

float checksum(const std::array<float, kSize> &values) {
    float sum = 0.0f;
    for (uint32_t i = 0; i < kSize; ++i) {
        sum += static_cast<float>(i + 1u) * values[i];
    }
    return sum;
}

} // namespace

int main() {
    std::array<float, kSize> candidate_x = {};
    std::array<float, kSize> candidate_c_prime = {};
    std::array<float, kSize> candidate_d_prime = {};

    candidate_c_prime[0] = kInputC[0] / kInputB[0];
    candidate_d_prime[0] = kInputD[0] / kInputB[0];

    for (uint32_t i = 1; i < kSize; ++i) {
        float denominator =
            kInputB[i] - kInputA[i] * candidate_c_prime[i - 1u];
        candidate_c_prime[i] = kInputC[i] / denominator;
        candidate_d_prime[i] =
            (kInputD[i] - kInputA[i] * candidate_d_prime[i - 1u]) /
            denominator;
    }

    candidate_x[kSize - 1u] = candidate_d_prime[kSize - 1u];
    for (uint32_t i = kSize - 1u; i > 0; --i) {
        candidate_x[i - 1u] =
            candidate_d_prime[i - 1u] - candidate_c_prime[i - 1u] * candidate_x[i];
    }

    if (!equal_array(candidate_x, kExpectedX) ||
        !equal_array(candidate_c_prime, kExpectedCPrime) ||
        !equal_array(candidate_d_prime, kExpectedDPrime)) {
        std::puts("FAILED");
        return 1;
    }

    std::printf("tridiag_solve checksum: %.3f\n", checksum(candidate_x));
    std::puts("PASSED");
    return 0;
}
