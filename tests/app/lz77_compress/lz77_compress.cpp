// Loom kernel implementation: lz77_compress
#include "lz77_compress.h"
#include "loom/loom.h"
#include <cmath>
#include <cstdlib>

// Full pipeline test from C++ source: LZ77 sliding-window compression
// Tests complete compilation chain with nested loops and data-dependent control flow
// At each position, searches a lookback window for the longest match.
// Outputs (offset, length, literal) triples.
// Test: [1,2,3,1,2,3,4,5,4,5,4,5,6,7,8,6,7,8,9,9] with window_size=10

// CPU implementation of LZ77 compression
void lz77_compress_cpu(const uint32_t* __restrict__ input_data,
                       uint32_t* __restrict__ output_offsets,
                       uint32_t* __restrict__ output_lengths,
                       uint32_t* __restrict__ output_literals,
                       uint32_t* __restrict__ output_count,
                       const uint32_t N,
                       const uint32_t window_size) {
    if (N == 0) {
        *output_count = 0;
        return;
    }

    uint32_t pos = 0;
    uint32_t write_idx = 0;

    while (pos < N) {
        uint32_t best_offset = 0;
        uint32_t best_length = 0;

        // Determine search window start
        uint32_t search_start = (pos > window_size) ? pos - window_size : 0;

        // Search for longest match in the lookback window
        for (uint32_t s = search_start; s < pos; s++) {
            uint32_t match_len = 0;

            // Extend match as far as possible (cap at 255)
            while (pos + match_len < N &&
                   input_data[s + match_len] == input_data[pos + match_len] &&
                   match_len < 255) {
                match_len++;
            }

            if (match_len > best_length) {
                best_length = match_len;
                best_offset = pos - s;
            }
        }

        // Emit the triple
        output_offsets[write_idx] = best_offset;
        output_lengths[write_idx] = best_length;

        uint32_t next_pos = pos + best_length;
        output_literals[write_idx] = (next_pos < N) ? input_data[next_pos] : 0;

        write_idx++;
        pos = next_pos + 1;
    }

    *output_count = write_idx;
}

// Accelerator implementation of LZ77 compression
LOOM_ACCEL()
void lz77_compress_dsa(LOOM_MEMORY_BANK(8) const uint32_t* __restrict__ input_data,
                       LOOM_STREAM uint32_t* __restrict__ output_offsets,
                       LOOM_STREAM uint32_t* __restrict__ output_lengths,
                       LOOM_STREAM uint32_t* __restrict__ output_literals,
                       uint32_t* __restrict__ output_count,
                       const uint32_t N,
                       const uint32_t window_size) {
    if (N == 0) {
        *output_count = 0;
        return;
    }

    uint32_t pos = 0;
    uint32_t write_idx = 0;

    while (pos < N) {
        uint32_t best_offset = 0;
        uint32_t best_length = 0;

        // Determine search window start
        uint32_t search_start = (pos > window_size) ? pos - window_size : 0;

        // Search for longest match in the lookback window
        for (uint32_t s = search_start; s < pos; s++) {
            uint32_t match_len = 0;

            // Extend match as far as possible (cap at 255)
            while (pos + match_len < N &&
                   input_data[s + match_len] == input_data[pos + match_len] &&
                   match_len < 255) {
                match_len++;
            }

            if (match_len > best_length) {
                best_length = match_len;
                best_offset = pos - s;
            }
        }

        // Emit the triple
        output_offsets[write_idx] = best_offset;
        output_lengths[write_idx] = best_length;

        uint32_t next_pos = pos + best_length;
        output_literals[write_idx] = (next_pos < N) ? input_data[next_pos] : 0;

        write_idx++;
        pos = next_pos + 1;
    }

    *output_count = write_idx;
}
