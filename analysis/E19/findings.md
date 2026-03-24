# E19: Cross-Domain Hardware Portability -- Findings

## Key Findings

1. Similar-category domains (compute-heavy: ai_llm/zk_stark, memory-heavy: arvr_stereo/graph_analytics) cross-port well (>70% native throughput).
2. Dissimilar categories (graph_analytics on ai_llm hardware) show significant degradation (<60% native).
3. FU mismatch is the primary degradation factor: domains needing multipliers cannot efficiently use architectures optimized for comparison-heavy workloads.
4. A 2-3 type core library (compute-heavy, memory-heavy, balanced) would cover all 6 domains with >75% native throughput.
5. The diagonal (native) entries confirm that co-optimization effectively specializes architectures for their target domain.
