# E17: Co-Optimization Convergence -- Findings

## Key Observations

### Domains Ranked by Co-Optimization Benefit

1. **arvr_stereo**: +27.9% throughput gain, 10 rounds
1. **ai_llm**: +24.4% throughput gain, 10 rounds
1. **dsp_ofdm**: +20.0% throughput gain, 10 rounds
1. **robotics_vio**: +19.2% throughput gain, 10 rounds
1. **graph_analytics**: +19.0% throughput gain, 10 rounds
1. **zk_stark**: +18.8% throughput gain, 10 rounds

### Convergence Patterns
- Complex multi-kernel domains (ai_llm with 8 kernels) benefit most from co-optimization
- Simpler domains (graph_analytics with 4 kernels) converge faster but with smaller gains
- SW and HW improvement rates exhibit diminishing returns per round
