
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>

namespace {

constexpr uint32_t kSize = 16;
constexpr float kPi = 3.14159265358979323846f;
constexpr float kTolerance = 1e-4f;

const std::array<float, kSize> kInputReal = {
    -1.254599f, 4.507143f, 2.319939f, 0.986585f,
    -3.439814f, -3.440055f, -4.419164f, 3.661762f,
    1.011150f, 2.080726f, -4.794155f, 4.699099f,
    3.324426f, -2.876609f, -3.181750f, -3.165955f,
};

const std::array<float, kSize> kInputImag = {
    -1.957578f, 0.247564f, -0.680550f, -2.087709f,
    1.118529f, -3.605061f, -2.078553f, -1.336382f,
    -0.439300f, 2.851760f, -3.003262f, 0.142344f,
    0.924146f, -4.535496f, 1.075449f, -3.294759f,
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

void fft_butterfly_ref(const float *input_real, const float *input_imag,
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
    fft_butterfly_ref(input_real.data(), input_imag.data(), reference_real.data(),
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

    for (uint32_t i = 0; i < kSize; ++i) {
        if (std::fabs(reference_real[i] - candidate_real[i]) > kTolerance ||
            std::fabs(reference_imag[i] - candidate_imag[i]) > kTolerance) {
            std::puts("FAILED");
            return 1;
        }
    }

    std::printf("fft_butterfly checksum: %.3f\n",
                checksum(candidate_real, candidate_imag));
    std::puts("PASSED");
    return 0;
}
