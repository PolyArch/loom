# E28: FU Body Structure Impact -- Summary

## Methodology
Analyzed 10 representative kernels from 5 domains under three FU body variants:
- **single_op**: Each FU implements one arith/math operation
- **fused_dag**: Compound FU bodies (mul+add -> fma, cmp+select -> cmp_select)
- **configurable**: fabric.mux-based FUs that share hardware between compatible ops

Metrics: PE count, FU count, mapping success, II, area (um^2 at 32nm).
Operation counts derived from kernel C source files using pattern matching.

## Per-Variant Averages (across 10 kernels)

| Variant       | Avg FU Count | Avg Area (um^2) | All Mapped |
|--------------|-------------|-----------------|------------|
| single_op    | 174.9       | 70,162          | 10/10      |
| fused_dag    | 134.2       | 88,691          | 10/10      |
| configurable | 174.9       | 103,805         | 10/10      |

## Key Findings

1. **Fused DAG reduces FU count by 23%** (174.9 -> 134.2 average) by merging
   multiply-add pairs into fma and compare-select pairs into compound FUs.
   This directly reduces PE count and routing pressure.

2. **Fused DAG increases per-FU area by 53%** because each fused FU is larger
   (fma = 750 um^2 vs separate mul=450 + add=120 = 570 um^2, plus 15%
   internal routing overhead). The net total area is 26% higher than single_op.

3. **Configurable FUs have the highest total area** (+48% vs single_op)
   because fabric.mux overhead adds 25% to each FU while not reducing FU count
   (the same number of FU instances is needed; they just share hardware).

4. **Area savings from fused DAG depend on fusion opportunities.**
   - High fusion: qkv_proj has 50 mul + 27 add operations; fusing 27 into fma
     reduces FU count from 112 to 84 (25% reduction)
   - Low fusion: ntt has many shift and bitwise ops that cannot be fused

5. **The most beneficial fused pattern is fma** (multiply-accumulate), which
   appears in 8/10 kernels. The cmp_select fusion has marginal benefit
   because comparison-selection pairs are rare (1-6 instances per kernel).

6. **Configurable FUs do not improve mapping success** in this experiment
   because all kernels already map successfully with single_op FUs on
   sufficiently large PE arrays. The flexibility advantage would emerge
   on smaller, resource-constrained PE arrays.

## Area Savings Ranking (single_op as baseline)

| Kernel        | Fused vs Single | Configurable vs Single |
|--------------|----------------|------------------------|
| qkv_proj     | +28.2%         | +51.2%                 |
| softmax      | +30.3%         | +55.1%                 |
| gelu         | +24.3%         | +43.3%                 |
| fft_butterfly| +28.9%         | +52.3%                 |
| equalizer    | +21.8%         | +38.2%                 |
| sad_matching | +36.4%         | +74.9%                 |
| pagerank_spmv| +27.7%         | +49.7%                 |
| ntt          | +23.6%         | +41.9%                 |
| pose_estimate| +24.9%         | +44.5%                 |
| layernorm    | +27.6%         | +49.9%                 |

Note: fused_dag and configurable both have higher total area than single_op
in this analysis because the overhead factors (routing, mux) outweigh the
FU count reduction. The FU count reduction is real but the area model
charges realistic overhead for internal interconnect.

## Data Provenance
- CSV: out/experiments/E28/fu_body.csv (30 rows)
- Git: 4d4c308
- Area model: 32nm SRAM density estimates from spec-fabric-function_unit-ops.md
- Op counts: from actual kernel source files in benchmarks/tapestry/
