# E20: Initial Architecture Sensitivity -- Findings

## Key Findings

1. **Best final throughput**: spectral (0.006546), converged in 10 rounds.
2. **Fastest convergence**: random_fu (7 rounds).
3. Starting 'close' (spectral clustering) helps: fewer rounds and slightly better final quality.
4. Oversized architecture converges to comparable quality but takes more rounds (area reduction dominates early).
5. Homogeneous architectures must discover heterogeneity through the HW optimization step, adding 1-2 extra rounds.
