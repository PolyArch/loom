
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>

namespace {

constexpr uint32_t kCount = 16;
constexpr uint32_t kElems = kCount * 4u;
constexpr float kTolerance = 1e-5f;

void initialize_quaternions(std::array<float, kElems> &q1,
                            std::array<float, kElems> &q2) {
    for (uint32_t i = 0; i < kCount; ++i) {
        q1[i * 4u + 0u] = 1.0f + static_cast<float>(i) * 0.01f;
        q1[i * 4u + 1u] = 0.1f + static_cast<float>(i) * 0.03f;
        q1[i * 4u + 2u] = -0.2f + static_cast<float>(i) * 0.02f;
        q1[i * 4u + 3u] = 0.05f + static_cast<float>(i) * 0.025f;

        q2[i * 4u + 0u] = 0.8f - static_cast<float>(i) * 0.005f;
        q2[i * 4u + 1u] = -0.1f + static_cast<float>(i) * 0.01f;
        q2[i * 4u + 2u] = 0.2f + static_cast<float>(i) * 0.015f;
        q2[i * 4u + 3u] = -0.3f + static_cast<float>(i) * 0.02f;
    }
}

void quat_ref(const float *q1, const float *q2, float *out, uint32_t count) {
    for (uint32_t i = 0; i < count; ++i) {
        const float w1 = q1[i * 4u + 0u];
        const float x1 = q1[i * 4u + 1u];
        const float y1 = q1[i * 4u + 2u];
        const float z1 = q1[i * 4u + 3u];
        const float w2 = q2[i * 4u + 0u];
        const float x2 = q2[i * 4u + 1u];
        const float y2 = q2[i * 4u + 2u];
        const float z2 = q2[i * 4u + 3u];

        out[i * 4u + 0u] = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2;
        out[i * 4u + 1u] = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2;
        out[i * 4u + 2u] = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2;
        out[i * 4u + 3u] = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2;
    }
}

double checksum(const std::array<float, kElems> &values) {
    double sum = 0.0;
    for (uint32_t i = 0; i < kElems; ++i) {
        sum += static_cast<double>(i + 1u) * values[i];
    }
    return sum;
}

} // namespace

extern "C" __attribute__((noinline))
void quat_mult_kernel(const float *q1, const float *q2, float *out,
                      uint32_t count) {
    for (uint32_t i = 0; i < count; ++i) {
        const float w1 = q1[i * 4u + 0u];
        const float x1 = q1[i * 4u + 1u];
        const float y1 = q1[i * 4u + 2u];
        const float z1 = q1[i * 4u + 3u];
        const float w2 = q2[i * 4u + 0u];
        const float x2 = q2[i * 4u + 1u];
        const float y2 = q2[i * 4u + 2u];
        const float z2 = q2[i * 4u + 3u];

        out[i * 4u + 0u] = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2;
        out[i * 4u + 1u] = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2;
        out[i * 4u + 2u] = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2;
        out[i * 4u + 3u] = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2;
    }
}

int main() {
    std::array<float, kElems> q1 = {};
    std::array<float, kElems> q2 = {};
    std::array<float, kElems> reference = {};
    std::array<float, kElems> candidate = {};

    initialize_quaternions(q1, q2);
    quat_ref(q1.data(), q2.data(), reference.data(), kCount);
    quat_mult_kernel(q1.data(), q2.data(), candidate.data(), kCount);

    for (uint32_t i = 0; i < kElems; ++i) {
        if (std::fabs(reference[i] - candidate[i]) > kTolerance) {
            std::puts("FAILED");
            return 1;
        }
    }

    std::printf("quat_mult checksum: %.3f\n", checksum(candidate));
    std::puts("PASSED");
    return 0;
}
