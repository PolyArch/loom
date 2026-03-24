# E19: Cross-Domain Hardware Portability -- Summary

## Methodology
6x6 cross-compilation matrix: each domain's TDG compiled on each domain's co-optimized architecture.
Diagonal entries are native (100%). Off-diagonal entries show throughput degradation.

## Portability Matrix (throughput_vs_native_pct)

|              | ai_llm | dsp_of | arvr_s | roboti | graph_ | zk_sta |
|--------------|--------|--------|--------|--------|--------|--------|
| ai_llm       | 100.0% |  39.0% |  28.6% |  27.9% |  13.3% |  32.5% |
| dsp_ofdm     |  33.2% | 100.0% |  29.4% |  39.9% |  18.2% |  25.4% |
| arvr_stereo  |  24.3% |  29.4% | 100.0% |  30.0% |  20.3% |  20.9% |
| robotics_vio |  23.7% |  39.9% |  30.0% | 100.0% |  20.9% |  27.6% |
| graph_analyt |  18.9% |  30.3% |  33.8% |  34.8% | 100.0% |  23.6% |
| zk_stark     |  32.5% |  29.9% |  24.5% |  32.5% |  16.7% | 100.0% |

## Mapping Success Rate Matrix (%)

|              | ai_llm | dsp_of | arvr_s | roboti | graph_ | zk_sta |
|--------------|--------|--------|--------|--------|--------|--------|
| ai_llm       | 100.0% | 100.0% |  60.0% |  60.0% |  12.5% |  60.0% |
| dsp_ofdm     | 100.0% | 100.0% |  60.0% |  60.0% |  33.3% |  60.0% |
| arvr_stereo  |  60.0% |  60.0% | 100.0% |  60.0% |  60.0% |  60.0% |
| robotics_vio |  60.0% |  60.0% |  60.0% | 100.0% |  20.0% |  60.0% |
| graph_analyt |  60.0% |  60.0% |  60.0% |  60.0% | 100.0% |  60.0% |
| zk_stark     |  60.0% |  60.0% |  60.0% |  60.0% |  20.0% | 100.0% |

## Domain Clustering by Hardware Compatibility

### Most Compatible Domain Pairs
- dsp_ofdm on robotics_vio hardware: 39.9% native
- robotics_vio on dsp_ofdm hardware: 39.9% native
- ai_llm on dsp_ofdm hardware: 39.0% native
- graph_analytics on robotics_vio hardware: 34.8% native
- graph_analytics on arvr_stereo hardware: 33.8% native

### Least Compatible Domain Pairs
- arvr_stereo on graph_analytics hardware: 20.3% native
- graph_analytics on ai_llm hardware: 18.9% native
- dsp_ofdm on graph_analytics hardware: 18.2% native
- zk_stark on graph_analytics hardware: 16.7% native
- ai_llm on graph_analytics hardware: 13.3% native

### Provenance
- Git: 4d4c308
- Date: 2026-03-24T07:07:01.398143+00:00
- Model: BendersDriver cost model + Tier-A area model
