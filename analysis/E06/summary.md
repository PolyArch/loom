# E06: ADG Size vs Mapping Quality -- Summary

## Overview

Swept 3 PE array sizes (4x4/16 PEs, 6x6/36 PEs, 8x8/64 PEs) for 10
representative kernels across 5 domains. Used existing ADG library
(ctrl_core, dsp_core, ai_core). The 10x10 and 12x12 ADGs were not available.

- **Total runs**: 30 (10 kernels x 3 sizes)
- **Successful mappings**: 0

## Results

All 30 mapping attempts failed. The failures are consistent with E04:
the representative kernels selected for the size sweep are complex
multi-loop C programs that hit the same frontend and techmap limitations.

## Compilation Time vs Array Size

Despite no successful mappings, the compilation time data reveals a clear
scaling pattern:

| Kernel | 4x4 (ms) | 6x6 (ms) | 8x8 (ms) | Trend |
|--------|----------|----------|----------|-------|
| qkv_proj | 2,666 | 9,845 | 69,237 | 26x |
| softmax | 657 | 60,821 | 60,893 | 93x |
| fft_butterfly | 6,532 | 8,329 | 11,860 | 1.8x |
| viterbi | 38 | 34 | 34 | 1x (compile fail) |
| sad_matching | 3,015 | 22,147 | 26,704 | 8.9x |
| image_warp | 110 | 8,209 | 12,100 | 110x |
| imu_integration | 79 | 144 | 211 | 2.7x |
| fast_detect | 39 | 38 | 37 | 1x (compile fail) |
| bfs_traversal | 87 | 147 | 220 | 2.5x |
| pagerank_spmv | 542 | 49,688 | 58,313 | 108x |

### Observations

1. **Compile-fail kernels**: viterbi, fast_detect fail instantly (<40ms)
   regardless of ADG size -- the frontend error occurs before ADG loading.

2. **Small kernels**: imu_integration, bfs_traversal show modest scaling
   (2-3x from 4x4 to 8x8) because they fail early in the pipeline.

3. **Large kernels**: qkv_proj, softmax, pagerank_spmv show dramatic scaling
   (26-108x) because they reach the techmap/mapper stage where candidate
   enumeration scales with PE count.

4. **DFG-size gated**: The kernels that show large time scaling are the ones
   where the DFG is large enough to trigger full techmap candidate search.
   Smaller DFGs terminate quickly even on larger ADGs.

## Limitations

- Only 3 of 5 planned array sizes were tested (no 10x10, 12x12 ADGs)
- No successful mappings means we cannot analyze II vs array size trends
- The dsp_core FU set was not uniformly applied (4x4 uses ctrl_core FUs)

## Thesis Evaluation

The original thesis ("II improves with PE array size, with diminishing
returns") cannot be evaluated because no kernels successfully map. However,
the compilation time scaling data supports a related finding: **mapper
resource consumption grows superlinearly with ADG size**, suggesting that
larger arrays have diminishing returns in mapper efficiency even before
considering II improvements.

## Provenance

- Git: dd849d3
- Data: out/experiments/E06/size_sweep.csv
- ADGs: out/adg_library/ (ctrl_core, dsp_core, ai_core)
