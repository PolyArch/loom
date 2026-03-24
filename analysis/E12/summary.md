# E12: TDC Pruning Effectiveness Summary

## Overall Results
- Average pruning fraction: 91.0%
- Most restrictive constraint: noc_bandwidth

## Per-Domain Pruning

| Domain | Feasible | Pruned | Pruned % | Dominant Constraint |
|--------|----------|--------|----------|-------------------|
| ai_llm | 0 | 1200 | 100.0% | l2_size_kb |
| dsp_ofdm | 108 | 1092 | 91.0% | noc_bandwidth |
| arvr_stereo | 0 | 1200 | 100.0% | l2_size_kb |
| robotics_vio | 150 | 1050 | 87.5% | noc_bandwidth |
| graph_analytics | 210 | 990 | 82.5% | noc_bandwidth |
| zk_stark | 180 | 1020 | 85.0% | noc_bandwidth |

## Constraint Ranking (aggregate across domains)

- noc_bandwidth: 5400 total eliminations
- total_cores: 2940 total eliminations
- l2_size_kb: 2640 total eliminations
- core_type_count: 1440 total eliminations

## Provenance
- Git hash: 4d4c308
- Timestamp: 2026-03-24T07:06:49Z
- Method: Exhaustive enumeration of 1200 candidates per domain
- TDC bounds computed from contract bandwidth, L2 volume, FU diversity, kernel count
