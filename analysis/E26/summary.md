# E26: NoC Topology Comparison -- Summary

## Methodology
Compared three NoC topologies on a 2x2 core grid across 6 benchmark domains:
- **Mesh**: XY dimension-ordered routing
- **Ring**: Unidirectional ring (clockwise)
- **Hierarchical**: 2 clusters of 2 cores, intra-cluster 1-hop, inter-cluster 2-hop

Metrics: total NoC transfer cycles, maximum link utilization, average hop count,
contention events (links with utilization > 80%).

Kernel assignment: round-robin across 4 cores. Same assignment for all topologies.

## Best Topology per Domain

| Domain          | Best Topology | Cycles | Reasoning                                |
|-----------------|--------------|--------|------------------------------------------|
| ai_llm          | ring         | 43527  | Sequential pipeline: ring has fewer hops |
| dsp_ofdm        | ring         | 5503   | Sequential chain: ring wins by 2 cycles  |
| arvr_stereo     | ring         | 68612  | Fewest contention events in ring (1 vs 2)|
| robotics_vio    | ring         | 1353   | Sequential pipeline with 1.0 avg hops    |
| graph_analytics | mesh         | 771    | Fork-join pattern: mesh shortest paths   |
| zk_stark        | mesh         | 331    | Fan-out to multiple consumers: mesh best |

## Key Findings

1. **Ring topology wins for sequential pipeline workloads** (ai_llm, dsp_ofdm,
   arvr_stereo, robotics_vio). These domains have mostly linear producer-consumer
   chains where ring routing achieves avg_hop_count = 1.0 on a 4-node ring.

2. **Mesh topology wins for fork-join and fan-out workloads** (graph_analytics,
   zk_stark). These have one producer sending to multiple consumers; mesh
   provides shorter average paths for such patterns.

3. **Hierarchical has no advantage over mesh** for a 2x2 grid because the
   cluster structure does not match any domain's communication pattern better
   than mesh XY routing.

4. **Contention is significant for arvr_stereo** (stereo_disparity produces
   262144 elements). Ring reduces contention from 2 to 1 events by distributing
   traffic across the unidirectional ring.

5. **The absolute cycle difference between topologies is small** (< 1%) for
   most domains because the 2x2 grid limits maximum hop count to 2. Larger
   grids would show more topology sensitivity.

## Data Provenance
- CSV: out/experiments/E26/noc_topology.csv (18 rows)
- Git: 4d4c308
- Topology model: analytical NoCScheduler-compatible (XY DOR, ring, hierarchical)
- Assignment: round-robin, same for all topologies
