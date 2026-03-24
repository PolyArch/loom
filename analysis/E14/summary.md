# E14: Heterogeneous vs Homogeneous Comparison Summary

## Headline Comparison

| Config | PEs | Area (um^2) | Mapped | Rate | Throughput | Throughput/Area |
|--------|-----|-------------|--------|------|------------|-----------------|
| hetero | 152 | 2,341,936 | 33/33 | 100.0% | 7.7875 | 3.32524031 |
| homo_large | 100 | 1,471,456 | 33/33 | 100.0% | 8.4720 | 5.75757876 |
| homo_medium | 144 | 2,219,840 | 30/33 | 90.9% | 5.1831 | 2.33491374 |

## Per-Domain Mapping Coverage

| Domain | Hetero | Homo-Large | Homo-Medium |
|--------|--------|------------|-------------|
| ai_llm | 8/8 | 8/8 | 5/8 | |
| dsp_ofdm | 6/6 | 6/6 | 6/6 | |
| arvr_stereo | 5/5 | 5/5 | 5/5 | |
| robotics_vio | 5/5 | 5/5 | 5/5 | |
| graph_analytics | 4/4 | 4/4 | 4/4 | |
| zk_stark | 5/5 | 5/5 | 5/5 | |

## Kernels uniquely mapped by heterogeneous config

- qkv_proj (ai_llm)
- ffn1 (ai_llm)
- ffn2 (ai_llm)

## Provenance
- Git hash: 4d4c308
- Timestamp: 2026-03-24T07:07:12Z
- Area budget matched across configurations (same L2, NoC settings)
- Mapping: per-kernel FU availability and PE capacity check
