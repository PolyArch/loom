# E11: INNER-HW Area Optimization Summary

## Area Reduction per Core Type

| Core Type | Tier-1 Area (um^2) | Tier-2 Best Area (um^2) | Reduction (%) | Feasible T2 Candidates |
|-----------|-------------------|------------------------|---------------|----------------------|
| ctrl | 491,596 | 486,048 | 1.1 | 29 |
| gp | 506,192 | 482,548 | 4.7 | 50 |
| dsp | 516,692 | 486,096 | 5.9 | 35 |
| ai | 884,384 | 865,432 | 2.1 | 40 |

## Provenance
- Git hash: 4d4c308
- Timestamp: 2026-03-24T07:06:15Z
- Method: BO-guided per-core parameter search (50 iterations/type)
- Area formula: PE_AREA=12000.0, FU_AREA={'alu': 2000.0, 'mul': 4500.0, 'fp': 8000.0, 'mem': 3500.0}
