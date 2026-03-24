# E18: SW-Only vs HW-Only vs Co-Optimization -- Findings

## Key Findings

1. Co-optimization produces Pareto fronts that dominate or equal both single-sided optimization modes across all 6 domains.
2. SW-only optimization achieves higher throughput gains on fixed hardware, especially for domains with high data volume (arvr_stereo: 274K elements).
3. HW-only optimization achieves better area reduction, especially for heterogeneous domains where core type specialization has more room.
4. The compounding effect of co-optimization is largest for complex domains (ai_llm: 8 kernels, 4 kernel types) where SW and HW bottlenecks are jointly coupled.
5. Simpler domains (graph_analytics: 4 kernels) show smaller co-optimization benefit because the design space is less entangled.
