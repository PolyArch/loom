// Linear-interpolation function variant migrated from the legacy app corpus.

#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>

namespace {

constexpr uint32_t kDataCount = 32;
constexpr uint32_t kQueryCount = 64;
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

void interpolate_linear_ref(const float *input_x, const float *input_y,
                            const float *input_xq, float *output_yq,
                            uint32_t data_count, uint32_t query_count) {
    for (uint32_t q = 0; q < query_count; ++q) {
        float xq = input_xq[q];
        uint32_t interval = 0;
        for (uint32_t k = 0; k < data_count - 1; ++k) {
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
        output_yq[q] = y0 + t * (y1 - y0);
    }
}

extern "C" __attribute__((noinline))
void interpolate_linear_kernel(const float *input_x, const float *input_y,
                               const float *input_xq, float *output_yq,
                               uint32_t data_count, uint32_t query_count) {
    for (uint32_t q = 0; q < query_count; ++q) {
        float xq = input_xq[q];
        uint32_t interval = 0;
        for (uint32_t k = 0; k < data_count - 1; ++k) {
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
        output_yq[q] = y0 + t * (y1 - y0);
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

    interpolate_linear_ref(input_x.data(), input_y.data(), input_xq.data(),
                           expected.data(), kDataCount, kQueryCount);
    interpolate_linear_kernel(input_x.data(), input_y.data(), input_xq.data(),
                              candidate.data(), kDataCount, kQueryCount);

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
