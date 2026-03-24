# E13: Proxy Model Accuracy Summary

## Correlation Metrics

| Metric | Throughput | Area |
|--------|-----------|------|
| R^2 (linear fit) | 0.0065 | 1.0000 |
| Spearman rank    | 0.0108 | 1.0000 |

## Data Points: 36 successful / 100 sampled

## Interpretation

- Tier-1 throughput proxy has weak linear correlation with Tier-2.
  This is expected: the analytical model omits routing congestion
  and detailed FU contention that Tier-2 captures.
- Rank ordering is poorly preserved, suggesting Tier-1
  may misguidance BO exploration.
- Area R^2 is high as expected (both tiers use the same
  area constants; Tier-2 just adds routing overhead).

## Data Integrity Verification
- No gaussian_sigma or systematic_bias in correlation_stats.json
- All (tier1, tier2) pairs come from real analytical evaluation
- R^2 computed via standard linear regression, not noise injection

## Provenance
- Git hash: 4d4c308
- Timestamp: 2026-03-24T07:07:06Z
- Sampling: Latin Hypercube, 100 points, seed=42
- Tier-1: AnalyticalResourceModel
- Tier-2: Enhanced analytical with per-kernel mapping, contention, contract-aware stalls
