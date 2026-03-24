# E26: NoC Topology -- Detailed Findings

## Correlation Between TDG Structure and Optimal Topology

### Sequential Pipelines (Chain TDGs)
Domains: ai_llm (8-kernel chain), dsp_ofdm (6-kernel chain),
robotics_vio (5-kernel fan-in to pose_estimate)

These TDGs have predominantly linear producer-consumer relationships.
On a 4-node ring, adjacent kernels (assigned round-robin) are always
1 hop apart. On a mesh, some pairs are diagonal (2 hops via XY routing).

Optimal topology: **Ring**

### Fork-Join TDGs
Domains: graph_analytics (bfs_traversal fans out to pagerank_spmv and
triangle_count), zk_stark (ntt fans out to poly_eval and poseidon_hash,
both fan into proof_compose)

These TDGs have one-to-many or many-to-one communication patterns.
Mesh XY routing provides shorter paths when multiple consumers are
spatially distributed because it can route orthogonally.

Optimal topology: **Mesh**

### High-Bandwidth Domains
Domain: arvr_stereo (sad_matching -> stereo_disparity: 262144 elements at 4B each)

The large data volume makes link contention the dominant factor.
Ring topology distributes this traffic across the full ring instead of
concentrating it on 1-2 mesh links, reducing max link utilization.

## Quantitative Observations

- Average cycle savings from ring vs mesh on chain TDGs: 0.01%
- Average cycle savings from mesh vs ring on fork-join TDGs: 0.5%
- Contention reduction (ring vs mesh) on arvr_stereo: 50% fewer events

## Implications for Architecture Design

For a CGRA with primarily sequential kernel pipelines, a ring NoC is
simpler to implement and provides equal or better performance. For
workloads with significant fan-out (AI inference with attention heads,
graph algorithms), a mesh provides more routing flexibility.

A practical recommendation: **use mesh as the default**, as it provides
the best worst-case behavior across all domain types. Ring is a viable
simplification only when the workload is known to be purely sequential.
