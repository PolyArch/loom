# E17: Co-Optimization Convergence -- Summary

## Methodology
Co-optimization loop with maxRounds=10, threshold=0.01.
Alternating SW (TDGOptimizer) and HW (OUTER+INNER) optimization.
TDC contracts carry achieved rates between steps.

## Results

| Domain | Rounds | Round-1 T | Final T | T Gain | Round-1 A | Final A | A Reduction |
|--------|--------|-----------|---------|--------|-----------|---------|-------------|
| ai_llm           |     10 | 0.005336 | 0.006638 | +24.4% | 6231.4 | 5330.8 | +14.5% |
| arvr_stereo      |     10 | 0.003774 | 0.004828 | +27.9% | 3915.7 | 3358.0 | +14.2% |
| dsp_ofdm         |     10 | 0.024736 | 0.029679 | +20.0% | 4671.3 | 3976.5 | +14.9% |
| graph_analytics  |     10 | 0.047297 | 0.056271 | +19.0% | 3153.7 | 2724.7 | +13.6% |
| robotics_vio     |     10 | 0.035239 | 0.042017 | +19.2% | 3915.7 | 3358.0 | +14.2% |
| zk_stark         |     10 | 0.041472 | 0.049276 | +18.8% | 3915.7 | 3358.0 | +14.2% |

### Convergence Distribution
- Average rounds to convergence: 10.0
- Domains converged by round 3: 0/6
- Throughput improvement range: 18.8% to 27.9%
- Area reduction range: 13.6% to 14.9%

### Provenance
- Git: 4d4c308
- Date: 2026-03-24T07:07:42.277352+00:00
- Max rounds: 10
- Threshold: 0.01
- Mapper budget: 10.0s
- Data source: tapestry_coopt_experiment --mode=convergence (co_optimize() API)
