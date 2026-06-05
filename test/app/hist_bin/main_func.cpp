// Histogram-binning function variant migrated from the legacy app corpus.

#include <array>
#include <cstdint>
#include <cstdio>

namespace {

constexpr uint32_t kSize = 55;
constexpr uint32_t kBins = 10;
constexpr float kMin = 0.0f;
constexpr float kMax = 100.0f;

void initialize_input(std::array<float, kSize> &input) {
    uint32_t write = 0;
    for (uint32_t bin = 0; bin < kBins; ++bin) {
        for (uint32_t count = 0; count <= bin; ++count) {
            input[write] = static_cast<float>(bin * 10u + 1u);
            ++write;
        }
    }
}

void hist_bin_ref(const float *input, uint32_t *output, uint32_t size,
                  uint32_t bins, float min_value, float max_value) {
    for (uint32_t i = 0; i < bins; ++i) {
        output[i] = 0;
    }

    const float width = (max_value - min_value) / static_cast<float>(bins);
    for (uint32_t i = 0; i < size; ++i) {
        const float value = input[i];
        if (value < min_value || value >= max_value) {
            continue;
        }
        uint32_t bin = static_cast<uint32_t>((value - min_value) / width);
        if (bin >= bins) {
            bin = bins - 1u;
        }
        ++output[bin];
    }
}

uint64_t checksum(const std::array<uint32_t, kBins> &bins) {
    uint64_t sum = 0;
    for (uint32_t i = 0; i < kBins; ++i) {
        sum += static_cast<uint64_t>(i + 1u) * bins[i];
    }
    return sum;
}

} // namespace

extern "C" __attribute__((noinline))
void hist_bin_kernel(const float *input, uint32_t *output, uint32_t size,
                     uint32_t bins, float min_value, float max_value) {
    for (uint32_t i = 0; i < bins; ++i) {
        output[i] = 0;
    }

    const float width = (max_value - min_value) / static_cast<float>(bins);
    for (uint32_t i = 0; i < size; ++i) {
        const float value = input[i];
        if (value < min_value || value >= max_value) {
            continue;
        }
        uint32_t bin = static_cast<uint32_t>((value - min_value) / width);
        if (bin >= bins) {
            bin = bins - 1u;
        }
        ++output[bin];
    }
}

int main() {
    std::array<float, kSize> input = {};
    std::array<uint32_t, kBins> reference = {};
    std::array<uint32_t, kBins> candidate = {};

    initialize_input(input);
    hist_bin_ref(input.data(), reference.data(), kSize, kBins, kMin, kMax);
    hist_bin_kernel(input.data(), candidate.data(), kSize, kBins, kMin, kMax);

    for (uint32_t i = 0; i < kBins; ++i) {
        if (reference[i] != candidate[i]) {
            std::puts("FAILED");
            return 1;
        }
    }

    std::printf("hist_bin checksum: %llu\n",
                static_cast<unsigned long long>(checksum(candidate)));
    std::puts("PASSED");
    return 0;
}
