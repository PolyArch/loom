# E03: Contract Inference Quality -- Summary

## Methodology
For each domain, loaded the fully-specified contract TDG (all fields manually
set by domain expert) and simulated compiler inference for the optional fields.
The minimal input provides only ordering + data_type; the compiler must infer
rate, tile_shape, visibility, double_buffering, and backpressure.

The inference uses fixed defaults for all inferable fields (no ground-truth
leakage). This represents what a user would get when providing only the
required fields and relying entirely on compiler defaults.

Inference defaults used:
- **rate**: 1024 (fixed default)
- **tile_shape**: [1024] (fixed 1D default)
- **visibility**: LOCAL_SPM
- **double_buffering**: false
- **backpressure**: BLOCK

Note: The ContractInferencePass binary was not used for this run because the
`--dump-inferred-contracts` flag is not yet supported. Results are marked as
"simulated" in the CSV. When the compiler pass is available, it should produce
better results for rate and tile_shape inference.

## Results

| Domain         | Contracts | Fields Match | Match Rate | Avg II Delta |
|----------------|----------:|-------------:|-----------:|-------------:|
| ai_llm         |         7 |        13/35 |      37.1% |        83.5% |
| dsp_ofdm       |         5 |         8/25 |      32.0% |        46.6% |
| arvr_stereo    |         4 |         6/20 |      30.0% |        81.2% |
| robotics_vio   |         4 |         7/20 |      35.0% |        75.9% |
| graph_analytics|         3 |         9/15 |      60.0% |         0.0% |
| zk_stark       |         5 |        11/25 |      44.0% |     14506.7% |
| **Total**      |    **28** |   **54/140** |  **38.6%** |   **2642.1%** |

### Per-field accuracy (from CSV data)
- **rate**: 14.3% match (4/28) -- default 1024 only matches graph_analytics
  and zk_stark::ntt edges
- **tile_shape**: 14.3% match (4/28) -- default [1024] only matches the same
  edges as rate
- **visibility**: 89.3% match (25/28) -- LOCAL_SPM default is correct for most
  edges; mismatches for GLOBAL_MEM in graph_analytics
- **double_buffering**: 75.0% match (21/28) -- false default is correct for
  most edges; mismatches where manual enables buffering
- **backpressure**: 0.0% match (0/28) -- manual TDGs do not specify
  backpressure (N/A), so comparison against BLOCK always fails

## Key Findings

1. **Rate and tile_shape are the hardest to infer from defaults alone**. The
   fixed default of 1024 only matches 4 of 28 edges. Real compiler inference
   (when available) would analyze kernel loop bounds to produce better values.

2. **Visibility inference works well** (89.3%) because LOCAL_SPM is the
   dominant pattern. Only graph_analytics uses GLOBAL_MEM, accounting for all
   3 mismatches.

3. **Backpressure shows 0% match** due to a measurement artifact: the manual
   TDGs do not explicitly specify backpressure, so the field reads as N/A.
   The comparison against the BLOCK default always fails. This does not
   indicate a real inference problem.

4. **II delta is very high for zk_stark** because small-rate edges (rate=3, 4,
   8) get a default of 1024, inflating the tile element count dramatically.
   This would not occur with real compiler inference that reads kernel bounds.

5. **graph_analytics has the best match rate** (60%) because its edges happen
   to use rate=1024 and tile_shape=[1024], matching the defaults.

## Data provenance
- CSV: out/experiments/E03/inference_quality.csv
- Inference method: simulated (fixed defaults, no compiler binary used)
- All data marked as "simulated" in the method column
