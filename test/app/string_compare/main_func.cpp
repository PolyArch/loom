
#include <array>
#include <cstdint>
#include <cstdio>

namespace {

constexpr uint32_t kSize = 128;
constexpr uint32_t kLess = 0xffffffffu;

void fill_equal(std::array<uint32_t, kSize> &a,
                std::array<uint32_t, kSize> &b) {
    for (uint32_t i = 0; i < kSize; ++i) {
        a[i] = static_cast<uint32_t>('a') + (i % 26);
        b[i] = static_cast<uint32_t>('a') + (i % 26);
    }
}

void fill_less(std::array<uint32_t, kSize> &a,
               std::array<uint32_t, kSize> &b) {
    for (uint32_t i = 0; i < kSize; ++i) {
        a[i] = static_cast<uint32_t>('a');
        b[i] = static_cast<uint32_t>('b');
    }
}

void fill_greater(std::array<uint32_t, kSize> &a,
                  std::array<uint32_t, kSize> &b) {
    for (uint32_t i = 0; i < kSize; ++i) {
        a[i] = static_cast<uint32_t>('z');
        b[i] = static_cast<uint32_t>('a');
    }
}

void string_compare_ref(const uint32_t *a, const uint32_t *b,
                        uint32_t *result, uint32_t size) {
    for (uint32_t i = 0; i < size; ++i) {
        if (a[i] < b[i]) {
            *result = kLess;
            return;
        }
        if (a[i] > b[i]) {
            *result = 1;
            return;
        }
    }
    *result = 0;
}

extern "C" __attribute__((noinline))
void string_compare_kernel(const uint32_t *a, const uint32_t *b,
                           uint32_t *result, uint32_t size) {
    for (uint32_t i = 0; i < size; ++i) {
        if (a[i] < b[i]) {
            *result = kLess;
            return;
        }
        if (a[i] > b[i]) {
            *result = 1;
            return;
        }
    }
    *result = 0;
}

bool check_case(void (*fill)(std::array<uint32_t, kSize> &,
                             std::array<uint32_t, kSize> &),
                uint32_t weight, uint64_t *sum) {
    std::array<uint32_t, kSize> a = {};
    std::array<uint32_t, kSize> b = {};
    uint32_t expected = 0;
    uint32_t candidate = 0;
    fill(a, b);
    string_compare_ref(a.data(), b.data(), &expected, kSize);
    string_compare_kernel(a.data(), b.data(), &candidate, kSize);
    *sum += static_cast<uint64_t>(candidate) * weight;
    return expected == candidate;
}

} // namespace

int main() {
    uint64_t sum = 0;
    if (!check_case(fill_equal, 1, &sum) ||
        !check_case(fill_less, 2, &sum) ||
        !check_case(fill_greater, 3, &sum)) {
        std::puts("FAILED");
        return 1;
    }

    std::printf("string_compare checksum: %llu\n",
                static_cast<unsigned long long>(sum));
    std::puts("PASSED");
    return 0;
}
