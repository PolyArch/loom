
#include <array>
#include <cstdint>
#include <cstdio>

namespace {

constexpr uint32_t kTextSize = 64;
constexpr uint32_t kPatternSize = 8;
constexpr uint32_t kWildcard = '?';

using Text = std::array<uint32_t, kTextSize>;
using Pattern = std::array<uint32_t, kPatternSize>;

void fill_match_case(Text &text, Pattern &pattern) {
    for (uint32_t i = 0; i < kTextSize; ++i) {
        text[i] = 'X';
    }
    pattern = {'A', 'B', '?', 'D', 'E', '?', 'G', 'H'};

    text[10] = 'A';
    text[11] = 'B';
    text[12] = 'C';
    text[13] = 'D';
    text[14] = 'E';
    text[15] = 'F';
    text[16] = 'G';
    text[17] = 'H';
}

void fill_no_match_case(Text &text, Pattern &pattern) {
    for (uint32_t i = 0; i < kTextSize; ++i) {
        text[i] = 'A';
    }
    for (uint32_t i = 0; i < kPatternSize; ++i) {
        pattern[i] = 'Z';
    }
}

void fill_all_wildcards_case(Text &text, Pattern &pattern) {
    for (uint32_t i = 0; i < kTextSize; ++i) {
        text[i] = static_cast<uint32_t>('A') + (i % 26);
    }
    for (uint32_t i = 0; i < kPatternSize; ++i) {
        pattern[i] = kWildcard;
    }
}

void wildcard_match_ref(const uint32_t *text, const uint32_t *pattern,
                        uint32_t *match, uint32_t text_size,
                        uint32_t pattern_size) {
    if (pattern_size > text_size) {
        *match = 0;
        return;
    }

    for (uint32_t i = 0; i <= text_size - pattern_size; ++i) {
        uint32_t candidate = 1;
        for (uint32_t j = 0; j < pattern_size; ++j) {
            const uint32_t pattern_value = pattern[j];
            if (pattern_value != kWildcard && text[i + j] != pattern_value) {
                candidate = 0;
                break;
            }
        }
        if (candidate) {
            *match = 1;
            return;
        }
    }

    *match = 0;
}

extern "C" __attribute__((noinline))
void wildcard_match_kernel(const uint32_t *text, const uint32_t *pattern,
                           uint32_t *match, uint32_t text_size,
                           uint32_t pattern_size) {
    if (pattern_size > text_size) {
        *match = 0;
        return;
    }

    for (uint32_t i = 0; i <= text_size - pattern_size; ++i) {
        uint32_t candidate = 1;
        for (uint32_t j = 0; j < pattern_size; ++j) {
            const uint32_t pattern_value = pattern[j];
            if (pattern_value != kWildcard && text[i + j] != pattern_value) {
                candidate = 0;
                break;
            }
        }
        if (candidate) {
            *match = 1;
            return;
        }
    }

    *match = 0;
}

bool check_case(void (*fill)(Text &, Pattern &), uint32_t weight,
                uint64_t *checksum) {
    Text text = {};
    Pattern pattern = {};
    uint32_t expected = 0;
    uint32_t candidate = 0;

    fill(text, pattern);
    wildcard_match_ref(text.data(), pattern.data(), &expected, kTextSize,
                       kPatternSize);
    wildcard_match_kernel(text.data(), pattern.data(), &candidate, kTextSize,
                          kPatternSize);

    *checksum += static_cast<uint64_t>(candidate) * weight;
    return expected == candidate;
}

} // namespace

int main() {
    uint64_t checksum = 0;
    if (!check_case(fill_match_case, 1, &checksum) ||
        !check_case(fill_no_match_case, 2, &checksum) ||
        !check_case(fill_all_wildcards_case, 3, &checksum)) {
        std::puts("FAILED");
        return 1;
    }

    std::printf("wildcard_match checksum: %llu\n",
                static_cast<unsigned long long>(checksum));
    std::puts("PASSED");
    return 0;
}
