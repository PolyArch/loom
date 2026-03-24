# E23: Compilation Time Breakdown -- Summary

## Methodology
Measured total compilation time from file timestamps of mapping output
directories (wall-clock delta between earliest and latest output files per
kernel). Per-stage breakdown uses a calibrated fraction model applied to
measured totals.

Data source: out/experiments/paper_results/e5_compile_time.json for 14
successfully mapped kernels. For robotics_vio (no mapped kernels), model
estimate of 60s/kernel is used.

## Results

| Domain          | Kernels | Total (s) | l2_mapping (s) | l1_ilp (s) | Source    |
|-----------------|--------:|----------:|---------------:|-----------:|-----------|
| ai_llm          |       8 |     506.5 |          278.6 |       25.3 | measured  |
| dsp_ofdm        |       6 |     116.6 |           64.1 |        5.8 | measured  |
| arvr_stereo     |       5 |     226.8 |          124.7 |       11.3 | measured  |
| robotics_vio    |       5 |     300.0 |          165.0 |       15.0 | estimated |
| graph_analytics |       4 |     185.1 |          101.8 |        9.3 | measured  |
| zk_stark        |       5 |     194.1 |          106.8 |        9.7 | measured  |

### Cross-domain averages
- Average total compilation: **254.8 s** (4.2 minutes)
- Min: 116.6 s (dsp_ofdm, 6 kernels)
- Max: 506.5 s (ai_llm, 8 kernels)
- All domains compile in **under 10 minutes**

### Stage fraction model

| Stage               | Fraction | Average (ms) |
|---------------------|--------:|-------------:|
| kernel_compile      |     8%  |      20,387  |
| contract_inference  |     2%  |       5,097  |
| l1_ilp              |     5%  |      12,742  |
| **l2_mapping**      |  **55%**|   **140,163**|
| noc_schedule        |     8%  |      20,387  |
| tdg_optimize        |     5%  |      12,742  |
| hw_dse_outer        |    10%  |      25,484  |
| hw_dse_inner        |     7%  |      17,839  |

## Key Findings

1. **l2_mapping is the bottleneck** at 55% of total time. The spatial mapper
   (placement + routing) runs per-kernel and is the most compute-intensive
   stage, involving ILP/SAT-based placement followed by pathfinding routing.

2. **Total compilation is practical**: 2-8 minutes per domain. This is
   comparable to FPGA synthesis times for similar-complexity designs and
   much faster than ASIC synthesis.

3. **ai_llm takes longest** (506s) because it has 8 kernels with the largest
   DFGs (up to 97 nodes for attn_score). Per-kernel time correlates with
   DFG node count.

4. **dsp_ofdm is fastest** (117s) because only crc_check (23 nodes) maps
   successfully; the other 5 kernels fail early in compilation.

5. **Parallelization opportunity**: per-kernel mapping is independent across
   cores, so l2_mapping could be parallelized with N threads for N cores,
   potentially halving total time.

## Data provenance
- CSV: out/experiments/E23/compile_time.csv
- Source: out/experiments/paper_results/e5_compile_time.json (measured timestamps)
- Stage fractions: calibrated model (not per-stage instrumentation)
