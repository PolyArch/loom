# Analytical Cost Model Specification

## Overview

This document specifies the analytical cost model for the heterogeneous
multi-core CGRA framework. The cost model evaluates a scheduled multi-kernel
system and produces performance, utilization, and efficiency metrics.

The cost model operates on the outputs of the system scheduler
([spec-hetero-scheduler.md](./spec-hetero-scheduler.md)): partition map,
route allocation, and epoch schedule. It does not modify any inputs.

Hard constraints are enforced by the scheduler, not the cost model. The cost
model evaluates quality and produces warnings for soft-limit violations.

## Score Structure

The cost model produces a multi-dimensional metric vector:

```
CostResult = {
  throughput          : f64,    // tokens per cycle (system-level)
  total_latency       : f64,    // cycles (end-to-end for acyclic;
                                //   pipeline fill + one iteration for cyclic)
  per_core_utilization: map<core, f64>,   // ratio [0, inf), mean across epochs
  per_link_utilization: map<link, f64>,   // ratio [0, inf)
  cache_hit_rates     : map<(kernel,cache), f64>,  // ratio [0, 1]
  reconfig_overhead   : f64,    // fraction of total time spent reconfiguring
  estimated_power     : f64,    // relative power estimate
  perf_per_watt       : f64,    // throughput / estimated_power
  warnings            : list<string>
}
```

Lower latency and reconfig_overhead are better. Higher throughput,
utilization, and perf_per_watt are better.

## Core Metric Families

### Compute Utilization

Measures how effectively each core's compute resources are used.

Utilization is computed per-epoch (at most one kernel runs per core per
epoch):

```
utilization(core, epoch) = kernel_compute_demand[k] / core_compute_capacity
```

where `k` is the kernel assigned to `core` in `epoch` (0 if idle), and:
- `kernel_compute_demand[k]`: estimated compute cycles per output token,
  derived from kernel's DFG critical path depth and operation count
- `core_compute_capacity`: available compute cycles per epoch
  (`pe_count * epoch_duration`)

The reported `per_core_utilization` in `CostResult` is the average
across all epochs: `mean(utilization(core, epoch) for epoch in all_epochs)`.

Ideal utilization is 1.0. Below 1.0 means the core has spare capacity
in that epoch. Above 1.0 means the kernel is too large for the core's
spatial resources (the mapper may still succeed via temporal execution).

### NoC Bandwidth Utilization

Measures how heavily each NoC link is loaded.

```
link_utilization(link) = sum(flow.bandwidth_req for flow in flows_on(link))
                       / link.bandwidth
```

See [spec-hetero-noc.md](./spec-hetero-noc.md) for the contention model.

Utilization > 1.0 is a soft warning. The cost model reports the over-
subscription. The throughput adjustment for NoC contention is applied
in the system-level throughput formula (see below):

```
max_noc_utilization = max(max_link_utilization, max_router_utilization)
noc_contention_factor = min(1.0, 1.0 / max_noc_utilization)
```

### Cache Effect

Measures the performance impact of caches on memory-intensive kernels.

For each (kernel, cache) pair, the cache model produces:
- `hit_rate`: fraction of memory accesses served by the cache
- `effective_latency`: weighted average of cache hit and miss latencies
- `cache_benefit`: latency savings due to caching

See [spec-hetero-cache.md](./spec-hetero-cache.md) for the detailed model.

The aggregate cache effect on throughput:

```
memory_stall_fraction(kernel) =
    kernel.memory_op_count / kernel.op_count
    * cache_effective_latency(kernel) / no_cache_latency(kernel)

throughput_adjustment(kernel) = 1.0 / (1.0 + memory_stall_fraction(kernel))
```

where:
- `cache_effective_latency(kernel)`: the effective memory latency for
  this kernel, computed as
  `effective_latency(kernel, effective_cache(kernel), effective_extmem(kernel))`
  using the nearest cache and nearest extmem as defined in
  [spec-hetero-cache.md](./spec-hetero-cache.md).
  Special cases:
  - If no `system.extmem` exists: `throughput_adjustment(kernel) = 1.0`
    (cache/memory penalty skipped; `WARN_NO_EXTMEM` emitted).
  - If no `system.cache` exists but `system.extmem` exists:
    `cache_effective_latency(kernel) = no_cache_latency(kernel)` (all
    accesses go to external memory with no cache benefit).
- `no_cache_latency(kernel)`: the memory latency without any cache
  (all accesses go directly to external memory):
  `extmem.latency + noc_distance(kernel.assigned_core, extmem.position)`

### Reconfiguration Overhead

Measures the fraction of total execution time spent on reconfiguration
between epochs.

```
reconfig_time(core, epoch_i, epoch_i+1) =
    config_size(kernel_at(core, epoch_i+1)) / config_bus_bandwidth

total_reconfig_time = sum over all epoch transitions:
    max(reconfig_time(core, i, i+1) for core in reconfigured_cores)

total_execution_time = total_latency + total_reconfig_time
reconfig_overhead = total_reconfig_time / total_execution_time
```

where `total_latency` is the sum of epoch durations (excluding
reconfiguration), as defined in the System-Level Throughput section.

Configuration size is a static property of the target core's
`fabric.module`, derived from the `config_mem` depth attribute (total
configuration words). This is read directly from the fabric definition,
not from mapper output. See
[spec-fabric-config_mem.md](./spec-fabric-config_mem.md).

### System-Level Throughput

Throughput is the reciprocal of the system's bottleneck epoch duration:

```
For acyclic kernel DAGs:
  total_latency = sum(epoch_duration[i] for i in all_epochs)
  raw_throughput = num_output_tokens / total_latency

For cyclic kernel DAGs (steady-state):
  bottleneck_epoch = max(epoch_duration[i] for i in cycle_epochs)
  raw_throughput = 1.0 / bottleneck_epoch
  total_latency = fill_latency + bottleneck_epoch
    where fill_latency = sum(epoch_duration[i] for i in fill_epochs)
```

where:
- `num_output_tokens`: total output tokens produced by the DAG's sink
  kernels (kernels with no outgoing forward edges), estimated as
  `sum(kernel.token_rate * epoch_duration)` for each sink.
- `cycle_epochs`: the set of epochs that form the repeating steady-state
  cycle in a cyclic kernel DAG.
- `fill_epochs`: the set of epochs before steady state is reached
  (pipeline fill phase).

`raw_throughput` is then adjusted for NoC contention, cache effects, and
reconfiguration overhead:

```
adjusted_throughput = raw_throughput
    * noc_contention_factor
    * product(throughput_adjustment(k) for k in active_kernels)
    * (1.0 - reconfig_overhead)
```

where:
- `noc_contention_factor`: see NoC Bandwidth Utilization above
  (accounts for both link and router contention)
- `active_kernels`: the set of all kernels that execute during the
  evaluated epoch(s). For acyclic DAGs, this is all kernels. For cyclic
  DAGs, this is the set of kernels executing in steady-state epochs.

`CostResult.throughput` stores the `adjusted_throughput` value.
`CostResult.total_latency` stores the raw `total_latency` before
throughput adjustments.

### Power Estimation

Power is estimated analytically from resource utilization:

```
estimated_power = sum over all cores:
    core.pe_count * pe_power_factor * utilization(core)
  + sum over all links:
    link.bandwidth * link_power_factor * link_utilization(link)
  + sum over all routers:
    router.bandwidth * router_power_factor * router_utilization(router)
  + sum over all caches:
    cache.capacity * cache_power_factor
  + static_power
```

Power factors are technology-dependent parameters. Default values:

| Parameter | Default | Unit |
|-----------|---------|------|
| `pe_power_factor` | 1.0 | relative |
| `link_power_factor` | 0.1 | relative per token/cycle |
| `router_power_factor` | 0.2 | relative per token/cycle |
| `cache_power_factor` | 0.001 | relative per byte |
| `static_power` | 0.5 | relative |

These defaults produce relative power numbers suitable for comparative
analysis (design A vs. design B), not absolute power in watts.

### Performance per Watt

```
perf_per_watt = adjusted_throughput / estimated_power
```

This metric enables comparison across designs with different
performance/power tradeoffs. It is the primary metric when absolute
performance does not exceed a GPU baseline but efficiency may be superior.

## Sensitivity Analysis Support

The cost model supports parameter sweeps for design space exploration.
Sweepable parameters include:

| Parameter | Sweep dimension |
|-----------|----------------|
| Core count | 1, 2, 4, 8, 16, ... |
| NoC link bandwidth | 1, 2, 4, 8, ... tokens/cycle |
| Cache capacity | 0, 16KB, 64KB, 256KB, 1MB |
| Reconfiguration bandwidth | 1, 2, 4, 8, ... words/cycle |
| Heterogeneity degree | homogeneous, 2 types, all different |

For each parameter setting, the scheduler re-runs and the cost model
re-evaluates. Results are collected into a sensitivity table:

```
| parameter | value | throughput | latency | perf_per_watt | bottleneck |
|-----------|-------|-----------|---------|---------------|------------|
```

## Bottleneck Classification

For each evaluated configuration, the cost model classifies the
performance bottleneck:

| Bottleneck class | Condition |
|-----------------|-----------|
| `compute_bound` | max(utilization(core)) > 0.8 and max(link_utilization) < 0.5 |
| `bandwidth_bound` | max(link_utilization) > 0.8 |
| `cache_bound` | max(memory_stall_fraction) > 0.3 |
| `reconfig_bound` | reconfig_overhead > 0.2 |
| `balanced` | no single dimension dominates |

Multiple bottleneck classes may apply simultaneously.

## Warnings

The cost model produces warnings (not errors) for soft-limit violations:

| Warning | Condition |
|---------|-----------|
| `WARN_LINK_OVERSUBSCRIBED` | link_utilization > 1.0 |
| `WARN_ROUTER_OVERSUBSCRIBED` | router_utilization > 1.0 |
| `WARN_CORE_IDLE` | utilization(core) == 0.0 |
| `WARN_HIGH_RECONFIG` | reconfig_overhead > 0.5 |
| `WARN_NO_CACHE_BENEFIT` | all hit_rates == 0.0 but caches exist |
| `WARN_BANDWIDTH_ESTIMATE_WEAK` | token_rate is derived, not measured |
| `WARN_NO_EXTMEM` | no `system.extmem` defined; memory latency calculations skipped |

## Comparison Baseline

The single-core sequential baseline is defined as:

```
sequential_baseline_time = sum(kernel_execution_time[k] for k in all_kernels)
                         + sum(reconfig_time between consecutive kernels)
sequential_baseline_throughput = num_output_tokens / sequential_baseline_time
```

This represents executing all kernels one at a time on a single core,
reconfiguring between each kernel. The baseline uses the largest core
(most capable) as the execution target.

Speedup is throughput-based:

```
speedup = CostResult.throughput / sequential_baseline_throughput
```

Using throughput (which incorporates NoC contention, cache effects,
and reconfiguration adjustments) ensures a consistent comparison.

Speedup may be < 1.0 when multi-core overhead (NoC latency,
reconfiguration) exceeds parallelism benefit.

## Related Documents

- [spec-hetero.md](./spec-hetero.md)
- [spec-hetero-scheduler.md](./spec-hetero-scheduler.md)
- [spec-hetero-noc.md](./spec-hetero-noc.md)
- [spec-hetero-cache.md](./spec-hetero-cache.md)
- [spec-mapper-cost.md](./spec-mapper-cost.md)
- [spec-fabric-config_mem.md](./spec-fabric-config_mem.md)
