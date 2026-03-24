# E05: Failure Analysis -- Summary

## Overview

Post-processed all 131 failed mapping attempts from E04 to classify failure
causes and identify systematic patterns.

## Failure Stage Distribution

| Stage | Count | Fraction | Description |
|-------|-------|----------|-------------|
| TECHMAP_FAIL | 83 | 63.4% | DFG ops cannot be matched to ADG FUs |
| COMPILE_FAIL | 45 | 34.4% | C frontend or DFG conversion error |
| TIMEOUT | 3 | 2.3% | Mapper exceeded 60s budget |

## COMPILE_FAIL Analysis (45 failures, 10 kernels)

Kernels that fail at the C-to-DFG frontend across ALL core types:

| Kernel | Domain | Root Cause |
|--------|--------|------------|
| bfs_traversal | graph_analytics | Complex pointer-based graph traversal |
| fast_detect | robotics_vio | 2D array access patterns |
| harris_corner | arvr_stereo | 2D convolution with runtime bounds |
| imu_integration | robotics_vio | Struct-heavy computation |
| msm | zk_stark | Multi-scalar multiplication with custom types |
| orb_descriptor | robotics_vio | Bit manipulation with 2D indexing |
| poseidon_hash | zk_stark | Lookup tables and modular arithmetic |
| proof_compose | zk_stark | Multi-function composition |
| qam_demod | dsp_ofdm | Complex number operations via structs |
| viterbi | dsp_ofdm | Dynamic programming with 2D state arrays |

Common frontend failure patterns:
- LLVM IR dominance errors from complex loop nest optimization
- Unsupported memcpy/memset intrinsics
- Struct-typed operations not lowered to scalar
- Missing header file resolution for domain-specific types

## TECHMAP_FAIL Analysis (83 failures, 20 kernels)

These kernels pass the frontend but produce DFGs with operations the ADG
cannot support. Typical missing operations include:
- `math.fma` (fused multiply-add) on integer-only cores
- `arith.sitofp`/`arith.fptosi` type conversions
- Complex memory access patterns requiring multiple extmem ports
- DFG candidates too large for available PE count

| Pattern | Approximate Count |
|---------|------------------|
| Missing FP ops on ctrl | 20 |
| DFG too large for array | 25 |
| Missing math intrinsics | 18 |
| Memory port exhaustion | 10 |
| Coverage < 0.6 | 10 |

## TIMEOUT Analysis (3 failures)

All 3 timeouts occurred for `feature_match` on gp/dsp/ai cores:
- The kernel produces a large DFG (estimated ~40 nodes)
- Tech mapping + placement exceeds the 60s budget
- The ctrl version fails at techmap (too few PEs)

## Mixed-Stage Kernels

Three kernels show different failure modes depending on core type:

| Kernel | ctrl | gp | dsp | ai |
|--------|------|-----|-----|-----|
| equalizer | COMPILE | TECHMAP | TECHMAP | TECHMAP |
| image_warp | COMPILE | TECHMAP | TECHMAP | TECHMAP |
| pose_estimate | TECHMAP | COMPILE | COMPILE | COMPILE |

The ctrl core's smaller ADG causes earlier termination (either faster failure
or different DFG candidate selection), leading to different failure modes.

## Actionable Insights

1. **Frontend improvements needed**: The C-to-DFG pipeline needs better
   handling of struct types, memcpy lowering, and 2D array access patterns.
   This would rescue 10 additional kernels (30% of the benchmark suite).

2. **Broader FU coverage**: Adding FP+math intrinsics to the ctrl_core would
   not help (too few PEs). But reducing DFG candidate sizes via loop tiling
   in the frontend could help kernels map to smaller arrays.

3. **Mapper budget**: The 60s budget is sufficient for most cases. Only
   feature_match needs more time.

## Near-Miss Analysis

Several kernel/core combinations achieve high techmap coverage (>90%) but
still fail at routing. These represent the closest-to-success cases:

| Kernel | Core | Coverage | Unrouted Edges |
|--------|------|----------|----------------|
| ntt | ai | 96.9% | 126 |
| softmax | dsp | 96.2% | 5 |
| softmax | ai | 96.2% | 5 |
| qkv_proj | ai | 95.4% | 42 |
| attn_output | ai | 95.3% | 30 |
| equalizer | ai | 95.2% | 134 |
| fft_butterfly | ai | 94.9% | 123 |
| triangle_count | gp/dsp/ai | 93.6% | 5 |
| label_prop | gp/dsp/ai | 92.1% | 8-16 |

Key observations:
- The ai_core (8x8) achieves highest coverage for most kernels, confirming
  that the larger FU repertoire (int+float+math) provides better DFG coverage.
- Even with >95% techmap coverage, routing can fail with many unrouted edges
  when the DFG is very large relative to the PE array.
- `softmax` and `triangle_count` on ai/dsp are within 5 unrouted edges of
  success -- small FU additions or DFG simplifications could rescue these.

## Provenance

- Git: dd849d3
- Data: out/experiments/E05/failure_analysis.csv
- Source: Post-processing of E04 results
