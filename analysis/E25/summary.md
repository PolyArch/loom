# E25: Multi-Kernel Application Pipelines -- Summary

## Methodology
Evaluated pipeline parallelism on 6 application TDGs using real per-kernel
mapping IIs. Each domain defines a directed pipeline graph where kernels
execute on separate cores with inter-core NoC communication.

Pipeline model:
- Pipeline throughput = 1 / (bottleneck_II + NoC_excess)
- Sequential throughput = 1 / sum(all_IIs)
- Speedup = sequential / pipeline
- NoC: 256-bit links, 3-cycle hop latency, 2-hop average on 2x2 mesh

## Results

| Domain          | Stages | Mapped | Bottleneck    |  Speedup | NoC OH% | Util% |
|-----------------|-------:|-------:|:--------------|--------:|--------:|------:|
| ai_llm          |      8 |    8/8 | gelu (II=29)  |   2.28x |    0.0% | 28.4% |
| dsp_ofdm        |      6 |    1/6 | crc_check (2) |   1.00x |    0.0% |100.0% |
| arvr_stereo     |      5 |    2/5 | sad_match (4) |   0.32x |   81.8% | 87.5% |
| robotics_vio    |      5 |    0/5 | (none)        |    n/a  |    n/a  |  n/a  |
| graph_analytics |      4 |    2/4 | label_prop(2) |   0.18x |   90.9% |100.0% |
| zk_stark        |      5 |    1/5 | poly_eval(15) |   1.00x |    0.0% |100.0% |

### Aggregate (5 analyzable pipelines)
- Average pipeline speedup: **0.96x**
- Average NoC overhead: **34.5%**
- Average core utilization: **83.2%**
- Best pipeline speedup: **2.28x** (ai_llm)

## Key Findings

1. **AI/LLM achieves 2.28x pipeline speedup** -- the only domain with all
   kernels mapped. The 8-stage pipeline (sequential sum=66 cycles) is
   bottlenecked by gelu (II=29), which has the highest II due to
   transcendental function approximation. Without gelu, the bottleneck
   would be attn_score/ffn1/ffn2 at II=8, yielding 7.25x speedup.

2. **Low core utilization in ai_llm** (28.4%) is caused by the gelu
   bottleneck: stages with II=2-4 are idle 85-93% of the time waiting
   for gelu to complete each iteration. This motivates kernel-level
   optimization of gelu or splitting it across multiple PEs.

3. **Partially-mapped pipelines suffer from NoC overhead**. When only 2
   of 5 stages are mapped (arvr_stereo, graph_analytics), the NoC transfer
   latency (~22 cycles per hop) exceeds the small IIs (2-4), making NoC
   the bottleneck rather than computation. This is only an issue when the
   bottleneck II is small (<22 cycles).

4. **Robotics_vio has zero mapped kernels**, preventing any pipeline
   analysis. All 5 kernels fail due to control-flow complexity or
   compilation failures.

5. **Pipeline speedup is bounded by stage balance**: the theoretical maximum
   is N (number of stages) but only achievable when all stages have equal
   II. ai_llm's 2.28x out of a theoretical 8x reflects the severe
   imbalance (II range: 2-29, ratio 14.5:1).

6. **Full pipeline execution requires solving the mapping gap**: 19/33
   kernels are unmapped. Improving DFG extraction for control-heavy kernels
   and adding more external memory ports would unlock pipeline parallelism
   for DSP, AR/VR, and robotics domains.

## Data provenance
- CSV: out/experiments/E25/pipeline_results.csv, stage_details.csv
- Per-kernel IIs: out/experiments/paper_results/e4_mapping_quality.json
- TDG pipeline definitions: benchmarks/tapestry/*/tdg_*.py
- NoC model: 256-bit links, 500MHz, 2x2 mesh topology
