# Network-on-Chip Model Specification

## Overview

This document specifies the Network-on-Chip (NoC) model used in the
heterogeneous multi-core CGRA framework. The NoC provides inter-core
communication through links and routers, supporting both stream and MMIO
communication modes.

The NoC model is analytical: it defines topology, bandwidth, latency, and
contention semantics for cost estimation, not for hardware generation or
cycle-accurate simulation.

## NoC Topology

A NoC topology is a directed graph of endpoints (cores and routers)
connected by links. The topology is defined by the `system.link` and
`system.router` operations within a `system.design`.

### Topology Graph

```
NoC Graph G = (V, E)
  V = { system.core instances } U { system.router instances }
  E = { system.link instances }, where each link is a directed edge
```

A topology may take any form: mesh, torus, ring, tree, fat-tree,
crossbar, or irregular. The system dialect does not impose structural
constraints beyond basic connectivity rules.

### Common Topologies

While any topology is representable, the following patterns are expected
to be common:

| Topology | Description |
|----------|-------------|
| Mesh | Cores at grid positions, routers at intersections. Manhattan routing. |
| Ring | Cores connected in a ring via routers. Unidirectional or bidirectional. |
| Crossbar | All cores connected through a central router. Single-hop. |
| Hierarchical | Clusters of cores with intra-cluster and inter-cluster links. |

The framework may provide builder utilities for common topologies, but the
IR representation is topology-agnostic.

## Link Semantics

A `system.link` carries data tokens between two NoC endpoints. Links are
the fundamental communication primitive.

### Bandwidth Model

Link bandwidth is specified in tokens per cycle:

```
sustained_throughput = bandwidth * token_bitwidth * link_frequency
```

where:
- `bandwidth`: tokens/cycle (integer, from link attribute)
- `token_bitwidth`: bits per token (from link type attribute)
- `link_frequency`: the operating frequency of this link in GHz,
  computed as `min(src.frequency, dst.frequency)` for links between
  cores. For links involving routers, the frequency is inherited from
  the nearest connected core. Different cores may have different
  frequencies for analytical modeling. This does not imply multi-clock
  hardware generation; see [spec-hetero.md](./spec-hetero.md).

When multiple data flows share a link, the available bandwidth is divided
among them. See the contention model below.

### Latency Model

Link latency is the minimum traversal time in cycles:

```
link_latency = latency attribute (cycles)
```

For a multi-hop path, total path latency is the sum of all link latencies
plus all intermediate router latencies:

```
path_latency = sum(link.latency for link in path)
             + sum(router.latency for router in intermediate_routers)
```

### Buffering

Each link has a `buffer_depth` attribute specifying the FIFO buffer
capacity at the destination end (in tokens). Buffering affects:

- **Backpressure propagation**: a full buffer stalls the upstream
  producer. In the analytical model, this is captured as potential
  throughput degradation when buffer depth is insufficient for
  bursty traffic.
- **Pipeline overlap**: deeper buffers allow more tokens to be in-flight,
  improving pipeline efficiency for latency-tolerant flows.

A `buffer_depth` of 0 means combinational pass-through (no buffering).

## Router Semantics

A `system.router` forwards data between its incident links without
computation. It acts as a crossbar: any input link can forward to any
output link.

### Router Bandwidth

A router has an aggregate `bandwidth` attribute representing the total
forwarding capacity across all ports. When multiple flows traverse the
same router, the aggregate bandwidth is shared.

### Router Latency

Each router adds a per-hop `latency` (in cycles) to paths traversing it.

## Multi-Hop Routing

Inter-core communication may traverse multiple links and routers when
cores are not directly connected.

### Path Definition

A **routing path** from core A to core B is an alternating sequence of
links and routers:

```
path = link_0, router_0, link_1, router_1, ..., link_n
```

where:
- `link_0.src = core_A` (or a router connected to core_A)
- `link_n.dst = core_B` (or a router connected to core_B)
- Each consecutive (link_i, router_i, link_{i+1}) is connected

### Path Bandwidth

The effective bandwidth of a multi-hop path is limited by the minimum
bandwidth along the path (bottleneck link or router):

```
path_bandwidth = min(
  min(link.bandwidth for link in path),
  min(router.bandwidth for router in intermediate_routers)
)
```

When flows share path segments, the shared segment bandwidth is further
divided among flows (see contention model).

### Path Latency

See the latency model above. Path latency is additive.

## Deadlock Avoidance

Multi-hop routing with cyclic dependencies among buffered links can
create deadlocks. The NoC model requires a deadlock-free routing
strategy.

### Dimension-Ordered Routing (Default)

For mesh and grid topologies, dimension-ordered routing (also known as
XY routing) is the default strategy:

1. Route along the X dimension first (horizontal links)
2. Then route along the Y dimension (vertical links)

This strategy is provably deadlock-free for mesh topologies because
it eliminates cyclic buffer dependencies. For torus topologies,
dimension-ordered routing alone is not sufficient; virtual channels or
dateline techniques are required for deadlock freedom.

### General Deadlock Avoidance

For non-mesh topologies, the system scheduler must verify that the set of
allocated routing paths does not create a cyclic dependency in the
**channel dependency graph** (CDG):

```
CDG: nodes = links, edges = (link_i -> link_j) if a flow uses link_i
     immediately before link_j through an intermediate router
```

A routing allocation is deadlock-free if and only if the CDG is acyclic.

The scheduler may use virtual channels, turn restrictions, or other
standard deadlock avoidance techniques. The choice of strategy is
implementation-defined.

### Deadlock Detection

The scheduler must check for potential deadlocks after route allocation.
If a deadlock is detected (cycle in the CDG), the scheduler must:

1. Report the conflicting flows and links
2. Attempt re-routing with a different strategy
3. If no deadlock-free allocation exists, report failure

## Contention Model

When multiple data flows share NoC resources, contention reduces per-flow
bandwidth. The analytical contention model uses fair bandwidth sharing.

### Link Contention

For a link with `n` flows sharing it:

```
per_flow_bandwidth(link) = link.bandwidth / n
```

If any flow's required bandwidth exceeds its fair share, the cost model
reports a bandwidth warning and uses the fair share as the effective
bandwidth.

### Router Contention

For a router with total aggregate bandwidth `B` and `n` flows
traversing it:

```
per_flow_bandwidth(router) = B / n
```

### End-to-End Flow Bandwidth

The effective bandwidth of a flow is the minimum of per-flow bandwidths
across all links and routers in its path:

```
flow_bandwidth = min(per_flow_bandwidth(resource)
                     for resource in flow.path)
```

## Communication Modes

### Stream Mode

Stream communication carries dataflow-typed tokens
(`!dataflow.tagged<!dataflow.bits<N>, iK>`, `!dataflow.bits<N>`, or `none`).
Properties:

- In-order delivery per link (FIFO semantics)
- Backpressure via valid/ready handshake
- Token-level granularity
- Tags identify logical channels when multiple logical flows share a
  physical link (tag-based multiplexing)

### MMIO Mode

MMIO communication carries memory access requests and responses.
Properties:

- Request: address + data (for writes) or address only (for reads)
- Response: read data
- Request-response pairing per transaction
- Ordering: requests from the same source are in-order; requests from
  different sources may interleave

MMIO mode uses the same physical NoC infrastructure as stream mode. The
distinction is in payload interpretation and the request-response pattern.

For analytical purposes, MMIO token_bitwidth is derived from the
`memref` element type bitwidth. Each request or response is modeled as
one token. The scheduler allocates two NoC paths for each MMIO
dependency: a forward path (request, src_core to dst_core) and a
reverse path (response, dst_core to src_core). The effective bandwidth
accounts for both directions: each MMIO access consumes bandwidth on
both paths.

## NoC Metrics

The NoC model produces the following metrics for the cost model:

| Metric | Definition | Unit |
|--------|-----------|------|
| `path_latency` | Sum of link and router latencies along a path | cycles |
| `path_bandwidth` | Minimum bandwidth along a path (pre-contention) | tokens/cycle |
| `flow_bandwidth` | Effective bandwidth after contention | tokens/cycle |
| `link_utilization` | sum(flow.bandwidth_req) / link.bandwidth | ratio [0, inf) |
| `router_utilization` | sum(flow.bandwidth_req) / router.bandwidth | ratio [0, inf) |

Utilization > 1.0 indicates oversubscription. The cost model treats this
as a soft warning, not a hard error.

## Related Documents

- [spec-hetero.md](./spec-hetero.md)
- [spec-hetero-system.md](./spec-hetero-system.md)
- [spec-hetero-scheduler.md](./spec-hetero-scheduler.md)
- [spec-hetero-cost.md](./spec-hetero-cost.md)
- [spec-fabric-switch.md](./spec-fabric-switch.md)
