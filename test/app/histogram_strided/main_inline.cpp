
#include <array>
#include <cstdint>
#include <cstdio>

namespace {

constexpr uint32_t kBins = 16;
constexpr uint32_t kStride = 7;
constexpr uint32_t kSize = (kBins * (kBins + 3u)) / 2u;
constexpr std::array<uint32_t, kBins> kExpected = {
    2, 3, 4, 5, 6, 7, 8, 9,
    10, 11, 12, 13, 14, 15, 16, 17};

uint64_t checksum(const std::array<uint32_t, kBins> &hist) {
    uint64_t sum = 0;
    for (uint32_t i = 0; i < kBins; ++i) {
        sum += static_cast<uint64_t>(i + 1u) * hist[i];
    }
    return sum;
}

} // namespace

int main() {
    std::array<uint32_t, kSize> input = {};
    std::array<uint32_t, kBins> hist = {};

    uint32_t write = 0;
    for (uint32_t bin = 0; bin < kBins; ++bin) {
        for (uint32_t count = 0; count < bin + 2u; ++count) {
            input[write] = bin * kStride + (bin % kStride);
            ++write;
        }
    }

    for (uint32_t i = 0; i < kBins; ++i) {
        hist[i] = 0;
    }
    for (uint32_t i = 0; i < kSize; ++i) {
        const uint32_t bin = input[i] / kStride;
        if (bin < kBins) {
            ++hist[bin];
        }
    }

    for (uint32_t i = 0; i < kBins; ++i) {
        if (hist[i] != kExpected[i]) {
            std::puts("FAILED");
            return 1;
        }
    }

    std::printf("histogram_strided checksum: %llu\n",
                static_cast<unsigned long long>(checksum(hist)));
    std::puts("PASSED");
    return 0;
}
