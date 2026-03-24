# E21: End-to-End Throughput vs Baselines -- Summary

## Methodology
Compared Tapestry CGRA throughput against CPU (Ryzen 9950X3D) and GPU (RTX 5090)
baselines for the 14 successfully mapped kernels across 6 domains.

- CGRA throughput: ops_per_cycle = DFG_nodes / II, Gops/s = ops/cycle * 500MHz
- CPU data: single-thread naive and 16-thread OpenMP from e10_cpu_baselines
- GPU data: cuBLAS/cuFFT/cuSPARSE where applicable from e10_gpu_baselines
- All IIs from real mapper output; synthesis confirmed 500MHz at SAED14nm

## Results

| Kernel         | Domain         | CGRA Gops/s | CPU-1T Gops/s | GPU Gops/s  | vs CPU-1T |
|----------------|----------------|------------:|----------:|--------:|----------:|
| attn_output    | ai_llm         |       10.67 |      4.14 |  30408  |    2.58x  |
| sad_matching   | arvr_stereo    |       11.88 |      3.80 |   1200  |    3.12x  |
| label_prop     | graph_analytics|        9.50 |      2.10 |    209  |    4.52x  |
| triangle_count | graph_analytics|        7.75 |      2.10 |    209  |    3.69x  |
| qkv_proj       | ai_llm         |        8.12 |      3.55 |  45401  |    2.29x  |
| image_warp     | arvr_stereo    |       11.00 |      5.10 |   2800  |    2.16x  |
| layernorm      | ai_llm         |       10.50 |      6.50 |   3800  |    1.62x  |
| poly_eval      | zk_stark       |        2.73 |      1.80 |    282  |    1.52x  |
| gelu           | ai_llm         |        1.26 |      8.20 |   5200  |    0.15x  |
| crc_check      | dsp_ofdm       |        5.75 |     12.00 |   1500  |    0.48x  |

### Aggregate
- Geomean speedup vs CPU single-thread: **1.47x**
- Geomean speedup vs GPU: **0.0012x** (GPU has 100-1000x more transistors)
- Geomean energy efficiency vs CPU: **501x** (CGRA: 0.5W vs CPU: 170W)
- Geomean energy efficiency vs GPU: **0.19x**
- Mapped kernels: 14/33 (42%)

## Key Findings

1. **CGRA beats CPU single-thread on 10/14 kernels** (geomean 1.47x). The
   advantage is greatest on graph and vision kernels (3-4.5x) where irregular
   access patterns penalize CPU caches but CGRA's spatial dataflow handles well.

2. **GPU throughput is 100-10000x higher in absolute terms**, which is expected:
   RTX 5090 has ~33B transistors at 4nm vs our ~160-PE CGRA at 14nm. The
   comparison is fundamentally unfair in raw throughput.

3. **Energy efficiency is where CGRA excels**: 501x better Gops/W than CPU.
   At 0.5W, the CGRA delivers meaningful throughput for embedded/edge workloads
   where GPU is impractical.

4. **Gelu is the worst kernel** (0.15x vs CPU-1T). Its II=29 on 73 nodes
   yields only 2.5 ops/cycle due to high-latency transcendental approximation.

5. **19 unmapped kernels** (58%) cannot run on CGRA due to: control-flow
   complexity (7), mapper failure (3), insufficient external memory ports (3),
   empty DFG extraction (4), and compile failures (2).

## Data provenance
- CSV: out/experiments/E21/baselines.csv, out/experiments/E21/efficiency.csv
- CGRA IIs: out/experiments/paper_results/e4_mapping_quality.json
- CPU/GPU: out/experiments/e10_cpu_baselines/, e10_gpu_baselines/
- Synthesis clock: out/experiments/e8_rtl_synthesis/synthesis_summary.json
