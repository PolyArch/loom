# E29: Host-CGRA Interaction -- Detailed Findings

## Compiler Performance Warning Calibration

The compiler should warn when a kernel is assigned to HOST. Based on
this experiment, the warning thresholds should be:

- **WARNING** (> 5% overhead): any single HOST kernel with production_rate
  > 4096 elements or data volume > 16 KB per contract edge.
- **ERROR** (> 15% overhead): two or more HOST kernels in a pipeline where
  the combined data volume crossing the boundary exceeds 40 KB.

The current HOST warning should include the estimated DMA overhead:
  DMA_overhead = DMA_setup + data_bytes / DMA_bandwidth
  total_boundary_cost = sum(DMA_overhead for each boundary edge) + sync_events * sync_cost

## Breakdown by Cost Component

### DMA Setup vs Transfer
- DMA setup: 50 cycles per descriptor (fixed overhead)
- DMA transfer: data_bytes / 8 cycles (bandwidth-limited)

For small transfers (< 400 bytes), setup dominates.
For large transfers (ai_llm: 65K+ bytes), transfer dominates.

### Sync Overhead
Sync overhead (20 cycles per event) is < 1% of total for all configs.
This validates that the hardware handshake mechanism is not a bottleneck.

## HOST Kernel Selection Impact

### ai_llm
- gelu on HOST: 9.7% overhead (gelu is compute-heavy, large intermediate data)
- softmax + ffn2 on HOST: 7.9% overhead (softmax is lighter, ffn2 has
  double-buffered output that reduces effective DMA cost)

The choice of which kernel to put on HOST matters. Putting a compute-light
kernel (softmax) on HOST costs less than a compute-heavy one (gelu).

### dsp_ofdm
- equalizer on HOST: 6.6% overhead (small data: 1200 complex64 elements)
- channel_est + viterbi on HOST: 14.5% overhead (viterbi produces 3600
  int32 elements per tile, with large trellis state)

### graph_analytics
- pagerank_spmv on HOST: 10.2% overhead (iterative SPMV is compute-heavy)
- bfs_traversal + label_prop on HOST: 12.4% overhead (two irregularly
  accessed kernels with GLOBAL_MEM visibility)

## Implications for Compiler Policy

1. The compiler should prefer accelerating compute-heavy kernels on CGRA
   and leaving only truly non-accelerable functions on HOST.

2. When a kernel cannot be accelerated (unsupported operations), the
   compiler should estimate the DMA overhead and report it as part of
   the performance prediction.

3. For pipelines with > 2 HOST kernels, the compiler should consider
   whether the entire pipeline should run on HOST (avoiding boundary
   crossings altogether).
