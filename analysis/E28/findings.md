# E28: FU Body Structure -- Detailed Findings

## Which Fused Patterns Are Most Beneficial?

### Multiply-Accumulate (fma)
Present in 8/10 kernels. Average fusion count: 10.5 per kernel.
Saves one FU instance per fused pair. The fma hardware (750 um^2) is
larger than separate mul+add (570 um^2) but eliminates one PE port
pair and one routing edge, which saves switch area and reduces routing
congestion.

Best-case: pose_estimate with 34 fma fusions (94 add + 175 mul -> 34 fma
+ 60 add + 141 mul), reducing FU count from 464 to 368 (21%).

### Compare-and-Select (cmp_select)
Present in 9/10 kernels. Average fusion count: 2.4 per kernel.
The savings are marginal because comparison-selection pairs are
infrequent in most dataflow kernels. The fused FU saves one PE slot
but the area overhead (cmpi+select at 190 um^2 * 1.15 = 219 um^2 vs
separate 190 um^2) is only justified when PE slots are scarce.

### Potential Future Fusions
Based on operation profiles:
- **add-shift** (loop index computation): common but adds little complexity
- **load-add** (address generation + load): could save a PE if address gen
  is fused into the load FU body. Already supported by the spec
  (handshake.load can coexist with arith ops in compound FU bodies).

## Configurable FU Analysis

The configurable variant uses fabric.mux to select between operations at
configuration time. This means:
- Hardware: max(area of all alternatives) + mux overhead
- Flexibility: one FU can serve different DFG nodes in different mappings

The 25% mux overhead comes from:
- fabric.mux select logic and config bits
- Additional internal routing for alternative datapaths
- Config memory for per-FU operation selection

### When Configurable FUs Win
Configurable FUs are most valuable when:
1. PE array is small relative to DFG size (resource constrained)
2. Different kernels need different operation mixes on the same core
3. The configurable alternatives have similar area (add vs sub: same ALU)

In this experiment, PE arrays are sized to fit the DFG, so configurability
provides no mapping advantage. The area overhead is pure cost.

## Recommendation for dsp_core Design

Based on the analysis of DSP-heavy kernels (fft_butterfly, equalizer):
1. Include fma as a first-class fused FU (saves 15-20% FU slots)
2. Keep arithmetic ops (add, sub, mul) as single-op FUs
3. Use configurable FUs only for comparison predicates (cmpi with
   configurable predicate is already supported in the spec)
4. Do not make general arithmetic configurable -- the area overhead
   is not justified by the flexibility gain
