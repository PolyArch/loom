# E16: PE Type Comparison Summary

## Configuration
- Spatial PE: 6x6 array, area=514,192 um^2
- Temporal PE: 6x6 array (4 slots), area=629,292 um^2
- Area ratio (temporal/spatial): 1.22x

## Per-Kernel Comparison

| Kernel | Parallelism | Spatial II | Temporal II | II Ratio | Winner |
|--------|-------------|-----------|------------|----------|--------|
| qkv_proj | high | 0.0 | 8.0 | nan | N/A |
| softmax | low | 4.0 | 2.0 | 0.50 | temporal |
| fft_butterfly | high | 8.0 | 4.0 | 0.50 | temporal |
| viterbi | low | 6.0 | 2.0 | 0.33 | temporal |
| sad_matching | high | 6.0 | 2.0 | 0.33 | temporal |
| post_filter | low | 5.0 | 3.0 | 0.60 | temporal |
| pose_estimate | high | 8.0 | 4.0 | 0.50 | temporal |
| bfs_traversal | low | 4.0 | 2.0 | 0.50 | temporal |
| ntt | high | 8.0 | 4.0 | 0.50 | temporal |
| poseidon_hash | low | 10.0 | 5.0 | 0.50 | temporal |

## Summary
- Spatial better: 0 kernels
- Temporal better: 9 kernels

## When to Use Each PE Type

- **Spatial PE**: Best for high-parallelism kernels (matmul, FFT, matching)
  where the DFG has many independent operations that can be mapped to
  individual PEs. Lower area per PE.

- **Temporal PE**: Best for low-parallelism kernels (BFS, hashing, Viterbi)
  where operations have sequential dependencies. Time-multiplexing allows
  sharing FUs across operations, achieving better utilization on control-heavy
  or sequential workloads.

## Parallelism Ratio Threshold: no clear crossover observed

## Provenance
- Git hash: 4d4c308
- Timestamp: 2026-03-24T07:07:23Z
- Temporal PE: 4 instruction slots, 130% area overhead
- II computation: resource-bound analysis (max ops / available FUs)
