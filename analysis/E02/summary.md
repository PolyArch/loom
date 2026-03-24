# E02: Auto-Analyze Accuracy -- Summary

## Methodology
For each domain, the entry function was analyzed using `tapestry_compile
--auto-tdg <source.c> --entry <func> --analyze-only`, which performs LLVM IR
level call graph extraction and pointer alias analysis. Results were compared
against manually constructed reference TDGs.

The analyzer detects: (1) noinline CGRA-targeted function calls as kernel
candidates (filtering out HOST functions like malloc/free), (2) data flow
edges between kernels based on shared pointer arguments.

Metrics reported per domain:
- **Recall** = matched / reference (false negative rate)
- **Precision** = matched / predicted (false positive rate)
- **F1** = harmonic mean of recall and precision

## Results

| Domain         | Kernels (auto/ref) | Recall | Precision |  F1  | Edges (auto/ref) | Recall | Precision |  F1  | Extra | Missing |
|----------------|--------------------:|-------:|----------:|-----:|---------:|-------:|----------:|-----:|------:|--------:|
| ai_llm         |             8 /  8 |  100%  |    100%   | 100% |  10 / 7  |  100%  |     70%   |  82% |     3 |       0 |
| dsp_ofdm       |             6 /  6 |  100%  |    100%   | 100% |   6 / 5  |  100%  |     83%   |  91% |     1 |       0 |
| arvr_stereo    |             4 /  5 |   80%  |    100%   |  89% |   5 / 4  |   75%  |     60%   |  67% |     2 |       1 |
| robotics_vio   |             5 /  5 |  100%  |    100%   | 100% |   4 / 4  |  100%  |    100%   | 100% |     0 |       0 |
| graph_analytics|             2 /  4 |   50%  |    100%   |  67% |   1 / 3  |    0%  |      0%   |   0% |     1 |       3 |
| zk_stark       |             5 /  5 |  100%  |    100%   | 100% |   6 / 5  |  100%  |     83%   |  91% |     1 |       0 |

### Aggregate
- **Kernel recall**: 30/33 (90.9%), precision: 30/30 (100%), F1: 95.2%
- **Edge recall**: 24/28 (85.7%), precision: 24/32 (75.0%), F1: 80.0%
- **Extra edges (false positives)**: 8 total
- **Missing edges (false negatives)**: 4 total (3 in graph_analytics, 1 in arvr_stereo)

## Key Findings

1. **Kernel precision is perfect** (100%) -- every detected kernel is a real
   kernel. Recall is 90.9% -- graph_analytics misses 2 kernels where the LLVM
   optimizer may inline or eliminate calls.

2. **Edge recall is high for linear pipelines** (100% for ai_llm, dsp_ofdm,
   robotics_vio, zk_stark). Pipelines with non-linear data flow (graph_analytics)
   show lower recall.

3. **False positives are moderate** -- the analyzer uses conservative pointer
   analysis, so buffer reuse (e.g., ai_llm qkv_proj output shared across
   attn_score and attn_output) produces extra edges. The real LLVM IR analysis
   is more precise than the old structural C-source fallback.

4. **graph_analytics performs worst** because its sparse-graph kernels use
   indirect memory access patterns that are harder for static analysis to
   track.

5. **robotics_vio achieves perfect scores** (100% on all metrics) because its
   pipeline is strictly linear with no buffer reuse.

## Data provenance
- CSV: out/experiments/E02/auto_vs_manual.csv
- Method: `tapestry_compile --auto-tdg --analyze-only` (binary mode)
