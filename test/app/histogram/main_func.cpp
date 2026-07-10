
#include <array>
#include <cstdint>
#include <cstdio>

namespace {

constexpr uint32_t kBins = 16;
constexpr uint32_t kSize = (kBins * (kBins + 1u)) / 2u;

void initialize_input(std::array<uint32_t, kSize> &input) {
    uint32_t write = 0;
    for (uint32_t bin = 0; bin < kBins; ++bin) {
        for (uint32_t count = 0; count <= bin; ++count) {
            input[write] = bin;
            ++write;
        }
    }
}

void histogram_ref(const uint32_t *input, uint32_t *hist, uint32_t size,
                   uint32_t bins) {
    for (uint32_t i = 0; i < bins; ++i) {
        hist[i] = 0;
    }
    for (uint32_t i = 0; i < size; ++i) {
        const uint32_t value = input[i];
        if (value < bins) {
            ++hist[value];
        }
    }
}

uint64_t checksum(const std::array<uint32_t, kBins> &hist) {
    uint64_t sum = 0;
    for (uint32_t i = 0; i < kBins; ++i) {
        sum += static_cast<uint64_t>(i + 1u) * hist[i];
    }
    return sum;
}

} // namespace

extern "C" __attribute__((noinline))
void histogram_kernel(const uint32_t *input, uint32_t *hist, uint32_t size,
                      uint32_t bins) {
    for (uint32_t i = 0; i < bins; ++i) {
        hist[i] = 0;
    }
    for (uint32_t i = 0; i < size; ++i) {
        const uint32_t value = input[i];
        if (value < bins) {
            ++hist[value];
        }
    }
}

int main() {
    std::array<uint32_t, kSize> input = {};
    std::array<uint32_t, kBins> reference = {};
    std::array<uint32_t, kBins> candidate = {};

    initialize_input(input);
    histogram_ref(input.data(), reference.data(), kSize, kBins);
    histogram_kernel(input.data(), candidate.data(), kSize, kBins);

    for (uint32_t i = 0; i < kBins; ++i) {
        if (reference[i] != candidate[i]) {
            std::puts("FAILED");
            return 1;
        }
    }

    std::printf("histogram checksum: %llu\n",
                static_cast<unsigned long long>(checksum(candidate)));
    std::puts("PASSED");
    return 0;
}
