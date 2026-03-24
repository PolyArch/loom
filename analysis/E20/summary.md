# E20: Initial Architecture Sensitivity -- Summary

## Methodology
Target domain: ai_llm (8 kernels, 7 contracts, 4 kernel types)
5 different initial architectures, each run through co-optimization with maxRounds=10.

## Initial Architectures

- **spectral**: Spectral clustering default -- 2 core type(s), 8 total instances
- **homogeneous_gp**: All gp_core homogeneous -- 1 core type(s), 8 total instances
- **homogeneous_dsp**: All dsp_core homogeneous -- 1 core type(s), 8 total instances
- **random_fu**: Random FU mix (3 core types) -- 3 core type(s), 9 total instances
- **oversized**: Oversized (all 4x4 cores) -- 1 core type(s), 8 total instances

## Results

| Config | Rounds | Final T | Final A | T vs Mean | A vs Mean |
|--------|--------|---------|---------|-----------|-----------|
| spectral           |     10 | 0.006546 | 5603.5 | +19.5% | -46.7% |
| homogeneous_gp     |      8 | 0.003916 | 5910.7 | -28.5% | -43.8% |
| homogeneous_dsp    |      9 | 0.005467 | 6089.4 |  -0.2% | -42.1% |
| random_fu          |      7 | 0.005661 | 7350.0 |  +3.4% | -30.1% |
| oversized          |      8 | 0.005790 | 27611.1 |  +5.7% | +162.6% |

### Convergence Variance
- Throughput range: 48.0% (min=0.003916, max=0.006546)
- Area range: 209.3% (min=5603.5, max=27611.1)
- Average rounds: 8.4 (min=7, max=10)

### Convergence Assessment
Configs show 48.0% throughput variance; starting point matters more than expected.

### Provenance
- Git: 4d4c308
- Date: 2026-03-24T07:07:04.857581+00:00
- Domain: ai_llm
- Max rounds: 10, threshold: 0.01
- Architecture params: from ArchitectureFactory CoreTypeSpec
