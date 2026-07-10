
#include <array>
#include <cstdint>
#include <cstdio>

namespace {

constexpr uint32_t kSize = 128;
constexpr uint32_t kWindowSize = 8;
constexpr uint32_t kWindowCount = kSize - kWindowSize + 1;
constexpr uint32_t kBase = 256;
constexpr uint32_t kModulus = 101;

void initialize_input(std::array<uint32_t, kSize> &input) {
    for (uint32_t i = 0; i < kSize; ++i) {
        input[i] = static_cast<uint32_t>('a') + (i % 26);
    }
}

uint64_t checksum(const std::array<uint32_t, kWindowCount> &values) {
    uint64_t sum = 0;
    for (uint32_t value : values) {
        sum += value;
    }
    return sum;
}

} // namespace

int main() {
    std::array<uint32_t, kSize> input = {};
    std::array<uint32_t, kWindowCount> expected = {};
    std::array<uint32_t, kWindowCount> candidate = {};
    initialize_input(input);

    uint32_t h = 1;
    for (uint32_t i = 0; i < kWindowSize - 1; ++i) {
        h = (h * kBase) % kModulus;
    }
    uint32_t expected_hash = 0;
    for (uint32_t i = 0; i < kWindowSize; ++i) {
        expected_hash = (expected_hash * kBase + input[i]) % kModulus;
    }
    expected[0] = expected_hash;
    for (uint32_t i = 1; i <= kSize - kWindowSize; ++i) {
        expected_hash =
            (expected_hash + kModulus - (input[i - 1] * h) % kModulus) %
            kModulus;
        expected_hash =
            (expected_hash * kBase + input[i + kWindowSize - 1]) % kModulus;
        expected[i] = expected_hash;
    }

    uint32_t candidate_hash = 0;
    for (uint32_t i = 0; i < kWindowSize; ++i) {
        candidate_hash = (candidate_hash * kBase + input[i]) % kModulus;
    }
    candidate[0] = candidate_hash;
    for (uint32_t i = 1; i <= kSize - kWindowSize; ++i) {
        candidate_hash =
            (candidate_hash + kModulus - (input[i - 1] * h) % kModulus) %
            kModulus;
        candidate_hash =
            (candidate_hash * kBase + input[i + kWindowSize - 1]) % kModulus;
        candidate[i] = candidate_hash;
    }

    for (uint32_t i = 0; i < kWindowCount; ++i) {
        if (expected[i] != candidate[i]) {
            std::puts("FAILED");
            return 1;
        }
    }

    std::printf("string_hash checksum: %llu\n",
                static_cast<unsigned long long>(checksum(candidate)));
    std::puts("PASSED");
    return 0;
}
