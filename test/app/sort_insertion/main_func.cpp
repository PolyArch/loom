// Insertion-sort function variant migrated from the legacy app corpus.

#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>

namespace {

constexpr uint32_t kSize = 12;
constexpr float kTolerance = 1e-5f;
constexpr std::array<float, kSize> kInput = {
    9.0f, 1.5f, 4.0f, 4.0f, -2.0f, 7.25f,
    0.0f, 3.5f, 8.0f, -1.0f, 2.0f, 6.0f};

void sort_insertion_ref(const float *input, float *output, uint32_t size) {
    for (uint32_t i = 0; i < size; ++i) {
        output[i] = input[i];
    }
    for (uint32_t i = 1; i < size; ++i) {
        const float key = output[i];
        int32_t j = static_cast<int32_t>(i) - 1;
        while (j >= 0 && output[static_cast<uint32_t>(j)] > key) {
            output[static_cast<uint32_t>(j + 1)] = output[static_cast<uint32_t>(j)];
            --j;
        }
        output[static_cast<uint32_t>(j + 1)] = key;
    }
}

double checksum(const std::array<float, kSize> &values) {
    double sum = 0.0;
    for (uint32_t i = 0; i < kSize; ++i) {
        sum += static_cast<double>(i + 1u) * values[i];
    }
    return sum;
}

} // namespace

extern "C" __attribute__((noinline))
void sort_insertion_kernel(const float *input, float *output, uint32_t size) {
    for (uint32_t i = 0; i < size; ++i) {
        output[i] = input[i];
    }
    for (uint32_t i = 1; i < size; ++i) {
        const float key = output[i];
        int32_t j = static_cast<int32_t>(i) - 1;
        while (j >= 0 && output[static_cast<uint32_t>(j)] > key) {
            output[static_cast<uint32_t>(j + 1)] = output[static_cast<uint32_t>(j)];
            --j;
        }
        output[static_cast<uint32_t>(j + 1)] = key;
    }
}

int main() {
    std::array<float, kSize> reference = {};
    std::array<float, kSize> candidate = {};

    sort_insertion_ref(kInput.data(), reference.data(), kSize);
    sort_insertion_kernel(kInput.data(), candidate.data(), kSize);

    for (uint32_t i = 0; i < kSize; ++i) {
        if (std::fabs(reference[i] - candidate[i]) > kTolerance) {
            std::puts("FAILED");
            return 1;
        }
    }

    std::printf("sort_insertion checksum: %.3f\n", checksum(candidate));
    std::puts("PASSED");
    return 0;
}
