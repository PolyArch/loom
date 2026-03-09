# Cache Effect Model Specification

## Overview

This document specifies the analytical cache effect model used in the
heterogeneous multi-core CGRA framework. The model estimates cache hit
rates and effective memory latencies based on kernel memory access patterns,
cache parameters, and spatial relationships on the NoC.

The cache model is purely analytical. It does not define hardware cache
semantics (no fill, evict, coherence, or replacement policy implementation).
It provides performance estimates for design space exploration.

## Cache Node Role

A `system.cache` node (defined in
[spec-hetero-system.md](./spec-hetero-system.md)) represents a cache in the
memory hierarchy between CGRA cores and external memory. When a kernel's
memory traffic passes through a cache node on the NoC, a portion of that
traffic may be served from the cache (hits) rather than from external
memory (misses).

The cache effect model determines what fraction of traffic is served
locally and how this affects effective memory latency.

## Hit Rate Estimation

Hit rate estimation uses the kernel's memory access pattern characterization
and the cache's physical parameters.

### Access Pattern Models

Each kernel declares a `mem_pattern` attribute. The hit rate model uses
different estimators per pattern:

#### Streaming Pattern (`"streaming"`)

```
hit_rate = 0.0
```

Streaming access has no temporal reuse. Each data element is accessed
exactly once and never revisited. The cache provides no benefit.

Spatial locality (prefetching adjacent cache lines) is not modeled in the
baseline. An implementation may add a spatial_prefetch_factor if needed.

#### Reuse-Heavy Pattern (`"reuse_heavy"`)

```
hit_rate = min(1.0, cache.capacity / kernel.mem_footprint)
```

The working set is assumed to be accessed uniformly within its bounds. If
the cache can hold the entire working set, hit rate approaches 1.0. If the
working set exceeds cache capacity, the hit rate degrades proportionally.

This is a simplified model that assumes uniform access within the working
set. For more refined estimation, the reuse distance model (below) may be
used.

#### Random Pattern (`"random"`)

```
hit_rate = min(1.0, cache.capacity / total_address_space)
```

where `total_address_space` is the kernel's `mem_footprint` (the total
address range that the kernel touches). For random access, each access
is equally likely to hit any cache line in the working set. The
probability of a cache hit equals the fraction of the working set that
fits in the cache. When `mem_footprint` greatly exceeds `cache.capacity`,
this is effectively zero.

#### Mixed Pattern (`"mixed"`)

```
hit_rate = reuse_fraction * hit_rate_reuse + (1 - reuse_fraction) * hit_rate_random
```

where:
- `hit_rate_reuse` is computed using the reuse-heavy formula above
- `hit_rate_random` is computed using the random formula above
- `reuse_fraction` is estimated from the kernel's memory operation
  profile (ratio of operations exhibiting temporal locality to total
  memory operations). This can be derived from the DFG structure:
  operations inside loop-carried dependency chains are classified as
  reuse, while operations on unique addresses are classified as random.
  If not derivable, default `reuse_fraction = 0.5`.

### Reuse Distance Refinement

For more precise hit rate estimation, the model supports reuse distance
analysis. Reuse distance is the number of distinct cache lines accessed
between two consecutive accesses to the same cache line.

```
For each memory access pattern with reuse distance distribution D:
  hit_rate = P(reuse_distance < cache.num_sets * cache.associativity)
```

where `cache.num_sets = cache.capacity / (cache.line_size * cache.associativity)`.

The reuse distance distribution can be:
- Derived analytically from known access patterns (e.g., nested loop
  iteration spaces)
- Approximated from DFG structure (loop bounds, array dimensions)
- Provided as a kernel attribute (user-specified histogram)

Reuse distance refinement is optional. The basic pattern-based model is
the minimum requirement.

## External Memory Selection

When a system contains no `system.extmem` nodes, the cache effect model
is skipped entirely: no effective latency or bandwidth calculations are
performed, and the cost model emits `WARN_NO_EXTMEM`.

When a system contains one or more `system.extmem` nodes, each kernel
uses the **nearest** external memory endpoint:

```
effective_extmem(kernel) = argmin over all system.extmem nodes:
    noc_distance(kernel.assigned_core, extmem.position)
```

The selected endpoint's latency and bandwidth are used for all miss-path
calculations for that kernel. If two endpoints are equidistant, the one
with higher bandwidth is preferred.

## Effective Memory Latency

The cache effect model produces an **effective memory latency** for each
(kernel, cache, extmem) triple:

```
effective_latency =
    hit_rate * (cache.access_latency + noc_distance(core, cache.position))
  + (1 - hit_rate) * (extmem.latency + noc_distance(core, extmem.position))
```

where:
- `cache.access_latency`: cache hit access time (from `system.cache` attribute)
- `noc_distance(A, B)`: shortest-path NoC latency from node A to node B,
  computed as the sum of link latencies and router latencies along the path
  (see Path Length Calculation below)
- `extmem`: the selected `system.extmem` for this kernel (see External
  Memory Selection above)
- `core`: the kernel's assigned core
- `cache.position`: the NoC node where the cache is located
- `extmem.position`: the NoC node where the external memory interface
  is located (from `system.extmem`)

### Path Length Calculation

NoC path lengths are computed using shortest-path algorithms on the NoC
topology graph. The distance metric is the sum of link latencies and
router latencies along the path:

```
noc_distance(A, B) = min over all paths P from A to B:
    sum(link.latency for link in P) + sum(router.latency for router in P)
```

## Spatial Interaction

The **spatial interaction** between kernel placement, cache location, and
memory access patterns is a critical modeling dimension. The key insight is
that these three factors interact multiplicatively:

1. **Kernel placement**: which core a kernel runs on determines its
   physical position in the NoC.

2. **Cache location**: where the cache node sits in the NoC determines
   the distance from each core to the cache.

3. **Memory access pattern**: the kernel's access pattern determines the
   cache hit rate, which determines what fraction of traffic benefits
   from the cache's proximity.

### Spatial Interaction Formula

The performance benefit of a cache for a given kernel depends on all
three factors:

```
cache_benefit(kernel, cache) =
    hit_rate(kernel, cache) * latency_savings(kernel, cache)

latency_savings(kernel, cache) =
    (extmem.latency + noc_distance(core, extmem.position))
  - (cache.access_latency + noc_distance(core, cache.position))
```

A cache provides positive benefit only when:
- The hit rate is non-zero (the kernel has reusable accesses), AND
- The cache is closer (in NoC latency) than external memory

Moving a kernel to a core closer to a cache increases `latency_savings`
(if the cache is nearer than memory from the new core). Moving a kernel
further from the cache decreases the benefit, potentially to zero or
negative.

### Multi-Cache Interaction

**Baseline model (required):** When multiple caches exist at the same
hierarchy level, the model considers the **nearest cache** to each
kernel's assigned core:

```
effective_cache(kernel) = argmin over all caches:
    noc_distance(kernel.assigned_core, cache.position)
```

The `effective_latency` formula from the Effective Memory Latency
section uses this single nearest cache.

`system.cache` represents shared caches only (e.g., a shared L2
between multiple cores). Core-private data storage is modeled as
scratchpad (`fabric.memory` within each core's `fabric.module`), not
as a cache. Multi-level cache hierarchy (L1/L2) is not modeled.

## Data Reuse Across Kernels

When multiple kernels access overlapping memory regions, data cached by
one kernel's execution may benefit subsequent kernels. This **inter-kernel
reuse** is modeled as:

```
For kernels K1 and K2 sharing data region R:
  If K1 executes before K2 on the same core (or a core sharing the same cache):
    K2.effective_hit_rate += overlap_fraction * K1.cache_residue_probability
```

where:
- `overlap_fraction`: fraction of K2's accesses that target region R
- `cache_residue_probability`: probability that K1's cached data for R is
  still in the cache when K2 runs (depends on intermediate cache pollution
  and cache capacity relative to R)

Inter-kernel reuse is an optional refinement. The baseline model treats
each kernel's cache behavior independently.

## Effective Bandwidth

The cache model also affects effective memory bandwidth:

```
effective_mem_bandwidth(kernel) =
    hit_rate * cache_bandwidth + (1 - hit_rate) * external_mem_bandwidth
```

where:
- `cache_bandwidth`: bandwidth of the cache access path (limited by
  the bottleneck link bandwidth on the NoC path from core to cache)
- `external_mem_bandwidth`: `min(extmem.bandwidth, bottleneck_link_bw)`
  where `bottleneck_link_bw` is the minimum link bandwidth on the NoC
  path from the core to `extmem.position`

This effective bandwidth is an informational metric reported for design
space exploration. The cost model's throughput adjustment uses the
latency-based `memory_stall_fraction` (see
[spec-hetero-cost.md](./spec-hetero-cost.md)), not this bandwidth term.

## Parameters

### System-Level Parameters

| Parameter | Source | Description |
|-----------|--------|-------------|
| `extmem.latency` | `system.extmem` attribute | External memory access latency (cycles) |
| `extmem.position` | `system.extmem` attribute | NoC node where external memory interface is located |
| `extmem.bandwidth` | `system.extmem` attribute | External memory bandwidth (tokens/cycle) |
| `noc_distance(A, B)` | Computed from topology | Sum of link + router latencies on shortest path |

### Per-Cache Parameters

Defined in `system.cache` attributes. See
[spec-hetero-system.md](./spec-hetero-system.md).

### Per-Kernel Parameters

Defined in `system.kernel` attributes. See
[spec-hetero-kernel.md](./spec-hetero-kernel.md).

## Related Documents

- [spec-hetero.md](./spec-hetero.md)
- [spec-hetero-system.md](./spec-hetero-system.md)
- [spec-hetero-kernel.md](./spec-hetero-kernel.md)
- [spec-hetero-cost.md](./spec-hetero-cost.md)
- [spec-fabric-mem.md](./spec-fabric-mem.md)
