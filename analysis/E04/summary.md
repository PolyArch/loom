# E04: Per-Kernel Mapping Panorama -- Summary

## Overview

Mapped 33 benchmark kernels across 6 domains onto 4 core types (ctrl 4x4,
gp 6x6, dsp 6x6, ai 8x8) using the Loom single-kernel compilation pipeline.

- **Total runs**: 132 (33 kernels x 4 core types)
- **Successful mappings**: 1 (0.76%)
- **Failed**: 131

## Successful Mapping

| Kernel | Domain | Core | II | Nodes | PE Util | Coverage |
|--------|--------|------|----|-------|---------|----------|
| label_prop | graph_analytics | ctrl | 2 | 12 | 68.75% | -- |

The `label_prop` kernel (graph label propagation) on the ctrl_core (4x4, 16 PEs)
is the only successful end-to-end mapping. This kernel uses simple integer
operations (store, carry, gate, cond_br, index_cast) which are all available
in the ctrl_core's baseline FU set.

## Failure Distribution

| Category | Count | Fraction |
|----------|-------|----------|
| TECHMAP_FAIL | 83 | 63.4% |
| COMPILE_FAIL | 45 | 34.4% |
| TIMEOUT | 3 | 2.3% |

## Mapping Success by Core Type

| Core | Array | Mapped | Rate |
|------|-------|--------|------|
| ctrl | 4x4 | 1/33 | 3.0% |
| gp | 6x6 | 0/33 | 0.0% |
| dsp | 6x6 | 0/33 | 0.0% |
| ai | 8x8 | 0/33 | 0.0% |

## Key Findings

1. **Frontend limitation is the primary bottleneck**: 10 of 33 kernels (30%)
   fail at the C-to-DFG compilation stage across all core types. These kernels
   use constructs (multi-dimensional arrays, complex control flow, function
   calls to stdlib) that the current LLVM-to-SCF-to-DFG pipeline cannot lower.

2. **Technology mapping is the secondary bottleneck**: 20 kernels pass frontend
   compilation but fail at technology mapping. The benchmarks produce DFG
   candidates with operations (memcpy, complex memory access patterns, nested
   loop structures) that cannot be matched to the ADG's FU repertoire.

3. **No single core type dominates**: label_prop succeeds on ctrl (smallest
   core) but fails on gp/dsp/ai due to techmap timeouts on larger ADGs. This
   suggests the larger ADG search space actually makes mapping harder for
   simple kernels.

4. **Compilation time scales with ADG size**: ctrl runs complete in 0.5-7s,
   while ai runs take 12-70s for the same kernels, due to the 4x larger
   candidate search space (8x8 = 64 PEs vs 4x4 = 16 PEs).

## Provenance

- Git: dd849d3
- Binary: build/bin/loom
- ADG library: out/adg_library/
- Data: out/experiments/E04/mapping_matrix.csv
