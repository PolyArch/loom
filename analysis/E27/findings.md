# E27: Memory Hierarchy -- Detailed Findings

## TDC Visibility Constraint Validation

The TDC (Tile Data Contract) visibility field determines the minimum
required memory level:

- LOCAL_SPM: tile must fit in per-core scratchpad
- SHARED_L2: tile shared through L2 cache
- GLOBAL_MEM: tile accessed through off-chip memory (L2 as cache)

### Does visibility correctly predict minimum SPM/L2?

**ai_llm (LOCAL_SPM):** The largest contract tile is 32x2048 = 65536
float32 elements = 256 KB. This single tile exceeds any SPM size in the
sweep (max 64 KB), explaining why mapping fails for all SPM points.
The visibility constraint correctly flags this as a LOCAL_SPM requirement
that the current SPM budget cannot satisfy.

**graph_analytics (GLOBAL_MEM):** Tiles are 1024 int32 = 4 KB each.
The GLOBAL_MEM visibility means L2 caching is the primary strategy.
The SPM sweep still shows benefit because the mapper uses SPM for
intermediate results even when the contract is GLOBAL_MEM.

### Recommendation
For ai_llm, the compiler should either:
1. Auto-reduce tile size when SPM is insufficient (at the cost of higher II)
2. Promote the visibility to SHARED_L2 when tile exceeds SPM

## Diminishing Returns Analysis

### SPM (graph_analytics)
- 4 KB -> 8 KB: throughput 0.33 -> 0.50 (+50%)
- 8 KB -> 16 KB: throughput 0.50 -> 1.00 (+100%)
- 16 KB -> 32 KB: throughput 1.00 -> 1.00 (+0%)
- 32 KB -> 64 KB: throughput 1.00 -> 1.00 (+0%)

Clear knee at 16 KB. Investing beyond the knee wastes area.

### L2 (ai_llm)
- 64 KB -> 128 KB: throughput 0.80 -> 1.00 (+25%)
- 128 KB -> 256 KB: throughput 1.00 -> 1.00 (+0%)

Clear knee at 128 KB for L2 spill benefit.

## Area-Performance Trade-off

Optimal configuration per domain:
- ai_llm: 16 KB SPM + 128 KB L2 (total SRAM area: 537 K um^2)
- graph_analytics: 16 KB SPM + 64 KB L2 (total SRAM area: 358 K um^2)

Going beyond these points provides zero throughput improvement but
continues to increase area linearly.
