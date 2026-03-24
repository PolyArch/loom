# E24: Scalability -- Summary

## Methodology
Two scalability sweeps using real per-kernel compile times and mapping data:

1. **Kernel count sweep**: 2, 4, 8, 16, 33 kernels on fixed 4-core architecture
2. **Core count sweep**: 2, 4, 8, 16 cores with fixed AI/LLM 8-kernel workload

Compile times derived from measured per-kernel times (e5_compile_time.json)
with overhead model for ILP core assignment, NoC scheduling, and DSE.
Throughput from real mapping IIs at 500MHz.

## Results

### Kernel Count Sweep

| Kernels | Compile (s) | Mapped | Throughput (Gops/s) | Benders Iter |
|--------:|------------:|-------:|--------------------:|-------------:|
|       2 |       136.9 |    2/2 |               14.19 |            2 |
|       4 |       265.3 |    4/4 |               28.10 |            2 |
|       8 |       527.0 |    8/8 |               28.10 |            2 |
|      16 |       852.2 |  11/16 |               28.10 |            3 |
|      33 |      1482.6 |  14/33 |               28.10 |            3 |

### Core Count Sweep (AI/LLM, 8 kernels)

| Cores | Compile (s) | Throughput (Gops/s) | Scaling Eff |
|------:|------------:|--------------------:|------------:|
|     2 |       571.6 |               14.19 |         n/a |
|     4 |       629.1 |               28.10 |        0.99 |
|     8 |       689.3 |               48.99 |        0.86 |
|    16 |       753.4 |               48.99 |        0.43 |

## Key Findings

1. **Compile time scales sub-linearly** with kernel count (exponent = 0.85).
   Going from 2 to 33 kernels increases compile time by 10.8x, not 16.5x.
   This is because per-kernel mapping is the dominant cost and it is
   independent across kernels.

2. **Throughput saturates at 8 kernels** for a 4-core system. Beyond 8
   kernels, additional kernels either fail to map or are assigned to already-
   occupied cores, providing no additional throughput.

3. **Core count scaling is efficient up to 8 cores** (86% efficiency at 8
   cores). Beyond 8 cores, throughput saturates because there are only 8
   AI/LLM kernels to distribute. Scaling efficiency at 16 cores drops to
   43%.

4. **Compile time grows slowly with core count**: 2-core to 16-core increases
   compile time by only 1.32x, because the dominant cost (per-kernel mapping)
   is independent of core count. Only ILP and NoC scheduling grow.

5. **Benders iterations remain low** (2-3) even at 33 kernels, confirming
   that the decomposition converges quickly. The infeasibility cuts from
   failed mappings effectively prune the search space.

6. **Mapping success rate degrades** beyond the initially supported kernels:
   100% for the first 8 (all ai_llm), dropping to 42% at 33 kernels.
   The unmapped kernels are primarily from domains with control-flow
   complexity (robotics, DSP).

## Data provenance
- CSV: out/experiments/E24/scalability.csv
- Per-kernel times: out/experiments/paper_results/e5_compile_time.json
- Mapping IIs: out/experiments/paper_results/e4_mapping_quality.json
