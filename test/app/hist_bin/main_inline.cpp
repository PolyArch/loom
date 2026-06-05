// Histogram-binning inline variant migrated from the legacy app corpus.

#include <array>
#include <cstdint>
#include <cstdio>

namespace {

constexpr uint32_t kSize = 55;
constexpr uint32_t kBins = 10;
constexpr float kMin = 0.0f;
constexpr float kMax = 100.0f;
constexpr std::array<uint32_t, kBins> kExpected = {1, 2, 3, 4, 5,
                                                   6, 7, 8, 9, 10};

uint64_t checksum(const std::array<uint32_t, kBins> &bins) {
    uint64_t sum = 0;
    for (uint32_t i = 0; i < kBins; ++i) {
        sum += static_cast<uint64_t>(i + 1u) * bins[i];
    }
    return sum;
}

} // namespace

int main() {
    std::array<float, kSize> input = {};
    std::array<uint32_t, kBins> output = {};

    uint32_t write = 0;
    for (uint32_t bin = 0; bin < kBins; ++bin) {
        for (uint32_t count = 0; count <= bin; ++count) {
            input[write] = static_cast<float>(bin * 10u + 1u);
            ++write;
        }
    }
    for (uint32_t i = 0; i < kBins; ++i) {
        output[i] = 0;
    }

    const float width = (kMax - kMin) / static_cast<float>(kBins);
    for (uint32_t i = 0; i < kSize; ++i) {
        const float value = input[i];
        if (value < kMin || value >= kMax) {
            continue;
        }
        uint32_t bin = static_cast<uint32_t>((value - kMin) / width);
        if (bin >= kBins) {
            bin = kBins - 1u;
        }
        ++output[bin];
    }

    for (uint32_t i = 0; i < kBins; ++i) {
        if (output[i] != kExpected[i]) {
            std::puts("FAILED");
            return 1;
        }
    }

    std::printf("hist_bin checksum: %llu\n",
                static_cast<unsigned long long>(checksum(output)));
    std::puts("PASSED");
    return 0;
}
