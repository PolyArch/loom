// Loom kernel implementation: wildcard_match
#include "wildcard_match.h"
#include "loom/loom.h"
#include <cmath>
#include <cstdlib>

// CPU implementation of single character wildcard matching
// Matches text against pattern with '?' as wildcard (matches any single character)
// Output: 1 if match found, 0 otherwise
void wildcard_match_cpu(const uint32_t* __restrict__ input_text,
                        const uint32_t* __restrict__ input_pattern,
                        uint32_t* __restrict__ output_match,
                        const uint32_t N,
                        const uint32_t M) {
    const uint32_t wildcard = '?';

    if (M > N) {
        *output_match = 0;
        return;
    }

    // Try matching pattern at each position in text
    for (uint32_t i = 0; i <= N - M; i++) {
        uint32_t match = 1;

        for (uint32_t j = 0; j < M; j++) {
            if (input_pattern[j] != wildcard && 
                input_text[i + j] != input_pattern[j]) {
                match = 0;
                break;
            }
        }

        if (match) {
            *output_match = 1;
            return;
        }
    }

    *output_match = 0;
}

// Accelerator implementation of single character wildcard matching
// NOTE: No LOOM_* pragmas intentionally - this is a no-pragma baseline test
void wildcard_match_dsa(const uint32_t* __restrict__ input_text,
                        const uint32_t* __restrict__ input_pattern,
                        uint32_t* __restrict__ output_match,
                        const uint32_t N,
                        const uint32_t M) {
    const uint32_t wildcard = '?';

    if (M > N) {
        *output_match = 0;
        return;
    }

    // Try matching pattern at each position in text
    for (uint32_t i = 0; i <= N - M; i++) {
        uint32_t match = 1;

        for (uint32_t j = 0; j < M; j++) {
            if (input_pattern[j] != wildcard && 
                input_text[i + j] != input_pattern[j]) {
                match = 0;
                break;
            }
        }

        if (match) {
            *output_match = 1;
            return;
        }
    }

    *output_match = 0;
}

