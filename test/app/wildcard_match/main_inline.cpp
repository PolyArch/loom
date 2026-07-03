// Wildcard string match inline variant migrated from the legacy app corpus.

#include <array>
#include <cstdint>
#include <cstdio>

namespace {

constexpr uint32_t kTextSize = 64;
constexpr uint32_t kPatternSize = 8;
constexpr uint32_t kWildcard = '?';

using Text = std::array<uint32_t, kTextSize>;
using Pattern = std::array<uint32_t, kPatternSize>;

void fill_case(uint32_t kind, Text &text, Pattern &pattern) {
    if (kind == 0) {
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
    } else if (kind == 1) {
        for (uint32_t i = 0; i < kTextSize; ++i) {
            text[i] = 'A';
        }
        for (uint32_t i = 0; i < kPatternSize; ++i) {
            pattern[i] = 'Z';
        }
    } else {
        for (uint32_t i = 0; i < kTextSize; ++i) {
            text[i] = static_cast<uint32_t>('A') + (i % 26);
        }
        for (uint32_t i = 0; i < kPatternSize; ++i) {
            pattern[i] = kWildcard;
        }
    }
}

uint32_t wildcard_match_inline(const Text &text, const Pattern &pattern) {
    for (uint32_t i = 0; i <= kTextSize - kPatternSize; ++i) {
        uint32_t candidate = 1;
        for (uint32_t j = 0; j < kPatternSize; ++j) {
            const uint32_t pattern_value = pattern[j];
            if (pattern_value != kWildcard && text[i + j] != pattern_value) {
                candidate = 0;
                break;
            }
        }
        if (candidate) {
            return 1;
        }
    }
    return 0;
}

uint32_t expected_result(uint32_t kind) {
    return kind == 1 ? 0 : 1;
}

} // namespace

int main() {
    uint64_t checksum = 0;
    for (uint32_t kind = 0; kind < 3; ++kind) {
        Text text = {};
        Pattern pattern = {};
        fill_case(kind, text, pattern);

        const uint32_t expected = expected_result(kind);
        const uint32_t candidate = wildcard_match_inline(text, pattern);
        if (expected != candidate) {
            std::puts("FAILED");
            return 1;
        }
        checksum += static_cast<uint64_t>(candidate) * (kind + 1u);
    }

    std::printf("wildcard_match checksum: %llu\n",
                static_cast<unsigned long long>(checksum));
    std::puts("PASSED");
    return 0;
}
