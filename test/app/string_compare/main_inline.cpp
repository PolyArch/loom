// String-compare inline variant migrated from the legacy app corpus.

#include <array>
#include <cstdint>
#include <cstdio>

namespace {

constexpr uint32_t kSize = 128;
constexpr uint32_t kLess = 0xffffffffu;

void fill_case(uint32_t kind, std::array<uint32_t, kSize> &a,
               std::array<uint32_t, kSize> &b) {
    for (uint32_t i = 0; i < kSize; ++i) {
        if (kind == 0) {
            a[i] = static_cast<uint32_t>('a') + (i % 26);
            b[i] = static_cast<uint32_t>('a') + (i % 26);
        } else if (kind == 1) {
            a[i] = static_cast<uint32_t>('a');
            b[i] = static_cast<uint32_t>('b');
        } else {
            a[i] = static_cast<uint32_t>('z');
            b[i] = static_cast<uint32_t>('a');
        }
    }
}

uint32_t compare_inline(const std::array<uint32_t, kSize> &a,
                        const std::array<uint32_t, kSize> &b) {
    for (uint32_t i = 0; i < kSize; ++i) {
        if (a[i] < b[i]) {
            return kLess;
        }
        if (a[i] > b[i]) {
            return 1;
        }
    }
    return 0;
}

uint32_t expected_result(uint32_t kind) {
    if (kind == 1) {
        return kLess;
    }
    if (kind == 2) {
        return 1;
    }
    return 0;
}

} // namespace

int main() {
    uint64_t sum = 0;
    for (uint32_t kind = 0; kind < 3; ++kind) {
        std::array<uint32_t, kSize> a = {};
        std::array<uint32_t, kSize> b = {};
        fill_case(kind, a, b);
        uint32_t expected = expected_result(kind);
        uint32_t candidate = compare_inline(a, b);
        if (expected != candidate) {
            std::puts("FAILED");
            return 1;
        }
        sum += static_cast<uint64_t>(candidate) * (kind + 1);
    }

    std::printf("string_compare checksum: %llu\n",
                static_cast<unsigned long long>(sum));
    std::puts("PASSED");
    return 0;
}
