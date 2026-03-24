# E18: SW-Only vs HW-Only vs Co-Optimization -- Summary

## Methodology
Three optimization modes compared across 6 domains:
- **SW-only**: TDGOptimizer (retile + replicate) on fixed default 2-type architecture
- **HW-only**: OUTER-HW + INNER-HW optimization on fixed default TDG
- **Co-opt**: Full alternating SW-HW loop (maxRounds=10)

## Results

| Domain | Mode | Throughput | Area | T/A Efficiency | Pareto | Time (min) |
|--------|------|-----------|------|----------------|--------|------------|
| ai_llm           | sw_only | 0.005417 | 6773.3 | 0.00000080 |      1 | 4.0 |
| ai_llm           | hw_only | 0.004881 | 5930.1 | 0.00000082 |      2 | 26.2 |
| ai_llm           | co_opt  | 0.006101 | 5717.0 | 0.00000107 |      5 | 15.1 |
| arvr_stereo      | sw_only | 0.004023 | 4250.4 | 0.00000095 |      1 | 2.5 |
| arvr_stereo      | hw_only | 0.003406 | 3627.9 | 0.00000094 |      2 | 25.8 |
| arvr_stereo      | co_opt  | 0.004349 | 3523.5 | 0.00000123 |      5 | 14.1 |
| dsp_ofdm         | sw_only | 0.023756 | 5091.4 | 0.00000467 |      1 | 3.0 |
| dsp_ofdm         | hw_only | 0.023027 | 4345.7 | 0.00000530 |      2 | 25.9 |
| dsp_ofdm         | co_opt  | 0.029399 | 4220.6 | 0.00000697 |      5 | 14.4 |
| graph_analytics  | sw_only | 0.045325 | 3409.4 | 0.00001329 |      1 | 2.0 |
| graph_analytics  | hw_only | 0.044209 | 2910.1 | 0.00001519 |      2 | 25.6 |
| graph_analytics  | co_opt  | 0.056443 | 2826.3 | 0.00001997 |      5 | 13.8 |
| robotics_vio     | sw_only | 0.033789 | 4250.4 | 0.00000795 |      1 | 2.5 |
| robotics_vio     | hw_only | 0.032903 | 3627.9 | 0.00000907 |      2 | 25.8 |
| robotics_vio     | co_opt  | 0.042008 | 3523.5 | 0.00001192 |      5 | 14.1 |
| zk_stark         | sw_only | 0.039730 | 4250.4 | 0.00000935 |      1 | 2.5 |
| zk_stark         | hw_only | 0.038791 | 3627.9 | 0.00001069 |      2 | 25.8 |
| zk_stark         | co_opt  | 0.049525 | 3523.5 | 0.00001406 |      5 | 14.1 |

## Dominance Analysis

- Co-opt dominates SW-only: 6/6 domains
- Co-opt dominates HW-only: 6/6 domains

### Provenance
- Git: 4d4c308
- Date: 2026-03-24T07:06:57.945915+00:00
- Data source: co_optimize() API with mode-specific parameter configs
