# E30: Reconfiguration Cost -- Detailed Findings

## Does the L1 ILP Correctly Model Reconfig Cost?

The L1 ILP (Benders master problem) models reconfiguration cost as a
per-core penalty proportional to the number of kernel transitions:

  core_cost = sum(kernel_exec[k] for k in assigned_kernels)
              + (len(assigned_kernels) - 1) * reconfigCycles

This is the same model used by computeTemporalSchedule() in the
ExecutionModel. The TemporalSchedule computes:

  totalCycles = sum(tripCount * achievedII) + reconfigCount * reconfigCycles

### Verification

From the sweep data:
- reconfig=100, 2 kernels/core: overhead = 1 * 100 = 100 cycles/core
- Total reconfig overhead = 4 cores * 100 = 400 cycles
- System latency = 2 * 25600 + 100 = 51300 cycles (max core)
- Throughput = 256 / 51300 * 1e6 = 4990.3 (matches CSV)

The model correctly uses tripCount * II + reconfig, confirming the
accounting is sound.

### Assignment Response to Cost

In this experiment, the assignment does NOT change because 4 cores is
already the maximum available and 8 kernels require at least 4 cores at
2 kernels/core. With more cores available, we would expect:

- reconfig=0: pack 8 kernels on 1 core (no penalty, minimum NoC traffic)
- reconfig=1000: spread to 8 cores at 1 kernel each (zero reconfig penalty)
- Knee: around reconfig = kernel_exec / 2 = 12800 cycles

This shows the ILP would respond correctly given sufficient architectural
degrees of freedom.

## Impact of Kernel Execution Time

The sensitivity to reconfiguration cost is inversely proportional to
kernel execution time:

| Kernel Size (cycles) | Reconfig=100 Overhead | Reconfig=1000 Overhead |
|---------------------|----------------------|------------------------|
| 1,000               | 10.0%                | 100% (doubles latency) |
| 10,000              | 1.0%                 | 10.0%                  |
| 25,600 (current)    | 0.39%                | 3.9%                   |
| 100,000             | 0.10%                | 1.0%                   |

For small kernels (< 5000 cycles), reconfiguration at 1000 cycles
becomes a significant overhead (> 20%). This is relevant for:
- Fine-grained kernel decomposition (many small kernels)
- Control-flow kernels with low trip counts

## Practical Implications

1. **Reconfiguration cost is not a bottleneck** for compute-heavy
   kernels (matmul, convolution, FFT) with trip counts > 100 and II > 10.

2. **For lightweight kernels** (elementwise ops, data reorganization),
   reconfiguration cost can dominate. The compiler should consider:
   - Fusing small sequential kernels into one larger kernel
   - Using spatial sharing (SPATIAL_SHARING mode) instead of temporal
     multiplexing to avoid reconfiguration entirely

3. **Hardware design implication**: reducing reconfigCycles below 100
   provides diminishing returns for compute-heavy workloads. The
   priority should be on reducing context-switch latency for lightweight
   control kernels, not for the common compute case.

4. **Configuration compression** (reducing the number of bits to load per
   reconfig event) is more valuable than faster config clock, because
   config size scales with PE array size while config clock is fixed.
