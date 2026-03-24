# E15: Core Type Specialization Summary

## Per-Domain FU Demand Profile

| Domain | ALU | MUL | FP | MEM | Control |
|--------|-----|-----|-----|-----|---------|
| ai_llm | 0.0% | 0.0% | 71.9% | 27.5% | 0.7% |
| dsp_ofdm | 21.8% | 1.1% | 36.8% | 27.6% | 12.6% |
| arvr_stereo | 18.7% | 0.0% | 46.7% | 26.7% | 8.0% |
| robotics_vio | 23.0% | 0.0% | 30.2% | 26.6% | 20.1% |
| graph_analytics | 17.7% | 0.0% | 14.2% | 32.7% | 35.4% |
| zk_stark | 38.5% | 36.5% | 0.0% | 22.9% | 2.1% |

## Key Findings

- **ai_llm**: Dominated by fp (72%)
- **dsp_ofdm**: Dominated by fp (37%)
- **arvr_stereo**: Dominated by fp (47%)
- **robotics_vio**: Dominated by fp (30%)
- **graph_analytics**: Dominated by control (35%)
- **zk_stark**: Dominated by alu (39%)

## Core Type Role Assignment

- **ctrl** (8 kernels): qam_demod, viterbi, fast_detect, orb_descriptor, feature_match, bfs_traversal, triangle_count, label_prop
- **gp** (2 kernels): crc_check, sad_matching
- **dsp** (5 kernels): ntt, msm, poseidon_hash, poly_eval, proof_compose
- **ai** (18 kernels): qkv_proj, attn_score, softmax, attn_output, ffn1, gelu, ffn2, layernorm, fft_butterfly, channel_est, equalizer, harris_corner, stereo_disparity, image_warp, post_filter, imu_integration, pose_estimate, pagerank_spmv

## Specialization Axis Analysis

The primary axes of specialization are:

1. **FP-intensive**: ai_llm, dsp_ofdm, arvr_stereo, robotics_vio
2. **Control-heavy**: robotics_vio, graph_analytics
3. **MUL-intensive**: zk_stark
4. **Balanced**: domains not in above categories

## Provenance
- Git hash: 4d4c308
- Timestamp: 2026-03-24T07:07:18Z
- Op histograms from kernel profile database (derived from C source structure)
- FU counts computed with target II=4 per kernel
