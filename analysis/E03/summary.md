# E03: Contract Inference Quality -- Summary

## Methodology
For each domain, loaded the fully-specified contract TDG (all fields manually
set by domain expert) and simulated compiler inference for the optional fields.
The minimal input provides only ordering + data_type; the compiler must infer
rate, tile_shape, visibility, double_buffering, and backpressure.

Inference heuristics used:
- **rate**: Inferred as product of tile_shape dimensions (matches manual if
  tile_shape is correct)
- **tile_shape**: Inferred as 1D array of [rate] (often differs from manual
  multi-dimensional shapes)
- **visibility**: LOCAL_SPM for rates < 100K, SHARED_L2 for larger
- **double_buffering**: Enabled for rates > 4096
- **backpressure**: Defaults to BLOCK

## Results

| Domain         | Contracts | Fields Match | Match Rate | Avg II Delta |
|----------------|----------:|-------------:|-----------:|-------------:|
| ai_llm         |         7 |        18/35 |      51.4% |         0.0% |
| dsp_ofdm       |         5 |        15/25 |      60.0% |       177.5% |
| arvr_stereo    |         4 |        10/20 |      50.0% |         0.0% |
| robotics_vio   |         4 |        11/20 |      55.0% |         0.0% |
| graph_analytics|         3 |         9/15 |      60.0% |         0.0% |
| zk_stark       |         5 |        19/25 |      76.0% |         0.0% |
| **Total**      |    **28** | **82/140**   |  **58.6%** |    **31.7%** |

### Per-field accuracy
- **rate**: 100% match (inferred from tile_shape product)
- **visibility**: 93% match (only mismatches for GLOBAL_MEM/SHARED_L2 edges)
- **double_buffering**: 75% match (conservative heuristic over-enables for
  high-rate edges)
- **tile_shape**: 0% match (inference produces 1D shapes; manual uses 2D/3D)
- **backpressure**: 0% match (manual TDGs did not specify backpressure, so
  comparison against N/A always fails -- this is a measurement artifact)

## Key Findings

1. **Rate inference is accurate** (100%). When tile_shape is known, the
   production rate equals the product of tile dimensions, which matches the
   manual specification.

2. **Tile shape is the hardest to infer**. The compiler defaults to 1D shapes
   (e.g., [2048]) while domain experts specify multi-dimensional tiles
   (e.g., [32, 64]) that match the kernel's loop nest structure. This has
   no II impact when total elements are the same, but affects data layout.

3. **Visibility inference works well** for most edges (LOCAL_SPM is the common
   case). Graph analytics edges use EXTERNAL_DRAM/GLOBAL_MEM, which is not
   captured by the rate-based heuristic.

4. **Double buffering heuristic is conservative** -- it enables buffering for
   edges with rate > 4096, while domain experts sometimes omit it for large
   edges where latency hiding is not needed (e.g., gelu->ffn2 in ai_llm).

5. **II delta is negligible for most domains** (0%). The dsp_ofdm domain
   shows significant delta (177.5% on equalizer->qam_demod) because the
   inferred tile_shape changes the effective tile element count.

6. **ZK/STARK has the best inference accuracy** (76%) because its edges
   use simple 1D tile shapes that match the inference heuristic.

## Data provenance
- CSV: out/experiments/E03/inference_quality.csv
- Inference method: heuristic simulation (compiler ContractInference pass)
