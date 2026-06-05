// Linear-interpolation inline variant migrated from the legacy app corpus.

#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>

namespace {

constexpr uint32_t kDataCount = 32;
constexpr uint32_t kQueryCount = 63;
constexpr float kTolerance = 1.0e-4f;

void initialize_inputs(std::array<float, kDataCount> &input_x,
                       std::array<float, kDataCount> &input_y,
                       std::array<float, kQueryCount> &input_xq) {
    for (uint32_t i = 0; i < kDataCount; ++i) {
        input_x[i] = static_cast<float>(i);
        input_y[i] = static_cast<float>(i * i);
    }
    for (uint32_t i = 0; i < kQueryCount; ++i) {
        input_xq[i] = static_cast<float>(i) * 0.5f;
    }
}

float checksum(const std::array<float, kQueryCount> &values) {
    float sum = 0.0f;
    for (float value : values) {
        sum += value;
    }
    return sum;
}

} // namespace

int main() {
    std::array<float, kDataCount> input_x = {};
    std::array<float, kDataCount> input_y = {};
    std::array<float, kQueryCount> input_xq = {};
    std::array<float, kQueryCount> expected = {};
    std::array<float, kQueryCount> candidate = {};
    initialize_inputs(input_x, input_y, input_xq);

    for (uint32_t q = 0; q < kQueryCount; ++q) {
        float xq = input_xq[q];
        uint32_t interval = 0;
        for (uint32_t k = 0; k < kDataCount - 1; ++k) {
            if (xq >= input_x[k] && xq <= input_x[k + 1]) {
                interval = k;
                break;
            }
        }
        float x0 = input_x[interval];
        float x1 = input_x[interval + 1];
        float y0 = input_y[interval];
        float y1 = input_y[interval + 1];
        float t = (xq - x0) / (x1 - x0);
        expected[q] = y0 + t * (y1 - y0);
    }

    for (uint32_t q = 0; q < kQueryCount; ++q) {
        float xq = input_xq[q];
        uint32_t interval = 0;
        for (uint32_t k = 0; k < kDataCount - 1; ++k) {
            if (xq >= input_x[k] && xq <= input_x[k + 1]) {
                interval = k;
                break;
            }
        }
        float x0 = input_x[interval];
        float x1 = input_x[interval + 1];
        float y0 = input_y[interval];
        float y1 = input_y[interval + 1];
        float t = (xq - x0) / (x1 - x0);
        candidate[q] = y0 + t * (y1 - y0);
    }

    for (uint32_t i = 0; i < kQueryCount; ++i) {
        if (std::fabs(expected[i] - candidate[i]) > kTolerance) {
            std::puts("FAILED");
            return 1;
        }
    }

    std::printf("interpolate_linear checksum: %.6f\n", checksum(candidate));
    std::puts("PASSED");
    return 0;
}
