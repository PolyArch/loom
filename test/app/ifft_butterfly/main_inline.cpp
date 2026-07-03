// IFFT butterfly inline variant migrated from the legacy app corpus.

#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>

namespace {

constexpr uint32_t kSize = 16;
constexpr float kPi = 3.14159265358979323846f;
constexpr float kTolerance = 1e-4f;

const std::array<float, kSize> kInputReal = {
    -3.981271f, 11.75819f, 1.067285f, -24.85930f,
    19.57854f, 2.130239f, 4.313792f, -6.751454f,
    1.824865f, -3.408288f, 0.6815608f, 4.723578f,
    8.814138f, -27.89848f, -6.278558f, -1.788430f,
};

const std::array<float, kSize> kInputImag = {
    -16.65886f, 5.712252f, 4.835214f, -10.16193f,
    -7.473514f, -2.492356f, -16.83604f, 4.521151f,
    -4.100621f, 6.368324f, 9.526210f, 12.11642f,
    10.31990f, -23.74220f, 6.707603f, -9.962789f,
};

void initialize_input(std::array<float, kSize> &real,
                      std::array<float, kSize> &imag) {
    real = kInputReal;
    imag = kInputImag;
}

uint32_t fft_stage_count(uint32_t size) {
    uint32_t stages = 0;
    for (uint32_t n = size; n > 1; n >>= 1) {
        ++stages;
    }
    return stages;
}

void ifft_butterfly_ref(const float *input_real, const float *input_imag,
                        float *output_real, float *output_imag, uint32_t size) {
    for (uint32_t i = 0; i < size; ++i) {
        output_real[i] = input_real[i];
        output_imag[i] = input_imag[i];
    }

    const uint32_t stage_count = fft_stage_count(size);
    for (uint32_t s = 1; s <= stage_count; ++s) {
        const uint32_t m = 1u << s;
        const float wm_r = std::cos(-2.0f * kPi / static_cast<float>(m));
        const float wm_i = std::sin(-2.0f * kPi / static_cast<float>(m));

        for (uint32_t k = 0; k < size; k += m) {
            float w_r = 1.0f;
            float w_i = 0.0f;
            for (uint32_t j = 0; j < m / 2u; ++j) {
                const uint32_t lo = k + j;
                const uint32_t hi = lo + m / 2u;
                const float t_r = w_r * output_real[hi] - w_i * output_imag[hi];
                const float t_i = w_r * output_imag[hi] + w_i * output_real[hi];
                const float u_r = output_real[lo];
                const float u_i = output_imag[lo];

                output_real[lo] = u_r + t_r;
                output_imag[lo] = u_i + t_i;
                output_real[hi] = u_r - t_r;
                output_imag[hi] = u_i - t_i;

                const float next_w_r = w_r * wm_r - w_i * wm_i;
                const float next_w_i = w_r * wm_i + w_i * wm_r;
                w_r = next_w_r;
                w_i = next_w_i;
            }
        }
    }

    const float scale = 1.0f / static_cast<float>(size);
    for (uint32_t i = 0; i < size; ++i) {
        output_real[i] *= scale;
        output_imag[i] = -output_imag[i] * scale;
    }
}

float checksum(const std::array<float, kSize> &real,
               const std::array<float, kSize> &imag) {
    float sum = 0.0f;
    for (uint32_t i = 0; i < kSize; ++i) {
        const float weight = static_cast<float>(i + 1u);
        sum += weight * real[i] + (weight + 0.25f) * imag[i];
    }
    return sum;
}

} // namespace

int main() {
    std::array<float, kSize> input_real = {};
    std::array<float, kSize> input_imag = {};
    std::array<float, kSize> reference_real = {};
    std::array<float, kSize> reference_imag = {};
    std::array<float, kSize> candidate_real = {};
    std::array<float, kSize> candidate_imag = {};

    initialize_input(input_real, input_imag);
    ifft_butterfly_ref(input_real.data(), input_imag.data(), reference_real.data(),
                       reference_imag.data(), kSize);

    for (uint32_t i = 0; i < kSize; ++i) {
        candidate_real[i] = input_real[i];
        candidate_imag[i] = input_imag[i];
    }

    const uint32_t stage_count = fft_stage_count(kSize);
    for (uint32_t s = 1; s <= stage_count; ++s) {
        const uint32_t m = 1u << s;
        const float wm_r = std::cos(-2.0f * kPi / static_cast<float>(m));
        const float wm_i = std::sin(-2.0f * kPi / static_cast<float>(m));

        for (uint32_t k = 0; k < kSize; k += m) {
            float w_r = 1.0f;
            float w_i = 0.0f;
            for (uint32_t j = 0; j < m / 2u; ++j) {
                const uint32_t lo = k + j;
                const uint32_t hi = lo + m / 2u;
                const float t_r = w_r * candidate_real[hi] - w_i * candidate_imag[hi];
                const float t_i = w_r * candidate_imag[hi] + w_i * candidate_real[hi];
                const float u_r = candidate_real[lo];
                const float u_i = candidate_imag[lo];

                candidate_real[lo] = u_r + t_r;
                candidate_imag[lo] = u_i + t_i;
                candidate_real[hi] = u_r - t_r;
                candidate_imag[hi] = u_i - t_i;

                const float next_w_r = w_r * wm_r - w_i * wm_i;
                const float next_w_i = w_r * wm_i + w_i * wm_r;
                w_r = next_w_r;
                w_i = next_w_i;
            }
        }
    }

    const float scale = 1.0f / static_cast<float>(kSize);
    for (uint32_t i = 0; i < kSize; ++i) {
        candidate_real[i] *= scale;
        candidate_imag[i] = -candidate_imag[i] * scale;
    }

    for (uint32_t i = 0; i < kSize; ++i) {
        if (std::fabs(reference_real[i] - candidate_real[i]) > kTolerance ||
            std::fabs(reference_imag[i] - candidate_imag[i]) > kTolerance) {
            std::puts("FAILED");
            return 1;
        }
    }

    std::printf("ifft_butterfly checksum: %.3f\n",
                checksum(candidate_real, candidate_imag));
    std::puts("PASSED");
    return 0;
}
