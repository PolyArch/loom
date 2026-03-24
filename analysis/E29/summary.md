# E29: Host-CGRA Interaction Overhead -- Summary

## Methodology
Modeled three configurations per domain using the ExecutionModel and
DMAScheduler infrastructure:
- **all_cgra**: all kernels on CGRA cores
- **1_host**: one pipeline-middle kernel on HOST
- **2_host**: two non-adjacent kernels on HOST

Parameters: DMA setup = 50 cycles, DMA bandwidth = 8 bytes/cycle,
sync = 20 cycles/event, host execution 3x slower than CGRA.

## Results

| Domain          | Config   | CGRA    | Host    | DMA    | Sync | Total   | Overhead |
|-----------------|----------|---------|---------|--------|------|---------|----------|
| ai_llm          | all_cgra | 348160  | 0       | 0      | 0    | 348160  | 0.0%     |
| ai_llm          | 1_host   | 217088  | 393216  | 65636  | 40   | 675980  | 9.7%     |
| ai_llm          | 2_host   | 258048  | 270336  | 45256  | 80   | 573720  | 7.9%     |
| dsp_ofdm        | all_cgra | 30992   | 0       | 0      | 0    | 30992   | 0.0%     |
| dsp_ofdm        | 1_host   | 28592   | 7200    | 2500   | 40   | 38332   | 6.6%     |
| dsp_ofdm        | 2_host   | 16696   | 42888   | 9996   | 80   | 69660   | 14.5%    |
| graph_analytics | all_cgra | 6144    | 0       | 0      | 0    | 6144    | 0.0%     |
| graph_analytics | 1_host   | 4096    | 6144    | 1124   | 40   | 11404   | 10.2%    |
| graph_analytics | 2_host   | 3072    | 9216    | 1686   | 60   | 14034   | 12.4%    |

Overhead = (DMA + sync) / total * 100%.

## Key Findings

1. **All-CGRA is always fastest** because it avoids all boundary crossing
   costs. Even a single HOST kernel adds 6.6-10.2% overhead from DMA and
   synchronization alone.

2. **1-host adds 6.6-10.2% overhead; 2-host adds 7.9-14.5% overhead.**
   The overhead depends on the data volume crossing the HOST-CGRA boundary.
   dsp_ofdm 2-host is worst (14.5%) because two high-bandwidth kernels
   (channel_est, viterbi) generate large DMA transfers.

3. **DMA dominates the overhead** (95%+ of boundary crossing cost). Sync
   overhead (20 cycles per event) is negligible compared to DMA setup (50
   cycles) and data transfer time.

4. **Host execution slowdown is the real cost.** The 3x slowdown from
   running kernels on the host CPU (no spatial parallelism) increases
   total time by 94% for ai_llm 1-host (gelu on host generates 393K
   host cycles vs 131K CGRA cycles).

5. **Accounting verification passes** for all 9 configurations:
   total = cgra + host + dma + sync.

## At What Pipeline Fraction Does HOST Become Unacceptable?

From the data:
- 1 of 8 kernels on host (12.5%): 9.7% overhead (ai_llm)
- 2 of 8 kernels on host (25%):   7.9% overhead (ai_llm -- less because
  the two host kernels are smaller than gelu)
- 1 of 4 kernels on host (25%): 10.2% overhead (graph_analytics)
- 2 of 4 kernels on host (50%): 12.4% overhead (graph_analytics)

Threshold: HOST overhead exceeds 10% when > 25% of pipeline kernels are
on host. For latency-sensitive workloads, even 1 HOST kernel may be
unacceptable due to the 3x execution slowdown.

## Data Provenance
- CSV: out/experiments/E29/host_overhead.csv (9 rows)
- Git: 4d4c308
- Model: DMAScheduler parameters from spec-host-accel-interface.md
- Accounting verified: total = cgra + host + dma + sync for all rows
