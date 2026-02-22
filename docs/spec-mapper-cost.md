# Loom Mapper Cost Model Specification

## Overview

This document defines how mapper solutions are scored after hard constraints are
satisfied.

Hard constraints are never traded for better score. Any hard-constraint
violation is an invalid mapping regardless of cost.

Constraint definitions are in [spec-mapper-model.md](./spec-mapper-model.md).

## Score Structure

The mapper uses a weighted sum objective:

`total_cost = sum(weight_i * metric_i)`

Lower cost is better.

Implementations may add target-specific metrics, but the core metric families
below should be supported.

The formulas in this document are reference formulas for comparability. Exact
implementations may refine coefficients and normalization details if they
preserve metric intent and deterministic behavior.

## Core Metric Families

### Placement Pressure

Measures concentration of mapped software nodes onto limited hardware
resources.

A **tile class** groups all hardware nodes that share the same PE body
pattern (identical operation set and internal connectivity). Two
`fabric.pe` nodes belong to the same tile class if and only if their body
subgraphs are structurally identical (same operations, same wiring). Two
`fabric.temporal_pe` FU nodes belong to the same tile class if their parent
temporal PEs have identical body patterns. The temporal PE virtual node
itself is not a tile class member (it is not a placement target). Memory
nodes form separate tile classes per type (`fabric.memory` vs
`fabric.extmemory`). Routing nodes and boundary sentinel nodes do not
participate in tile class computation.

For each tile class:

- `cap(tile)` = total number of hardware nodes in the class.
- `occ(tile)` = number of hardware nodes in the class that have at least
  one mapped software operation.

Reference formula:

`placement_pressure = sum_tile (occ(tile) / cap(tile))^2`

The squaring penalizes unbalanced placement: concentrating operations on a
few nodes of a tile class is more expensive than spreading them evenly.

Examples:

- Overuse risk of high-fanout switch neighborhoods
- Occupancy skew across equivalent compute tiles

### Routing Cost

Measures path expense of mapped software edges.

Reference formula:

`routing_cost = sum_hwedge (hop_weight(hwedge) * usage(hwedge))`

where `hop_weight` can encode topology preference (for example, direct < switch
hop) and `usage` is routed software-edge multiplicity on that hardware edge.

Examples:

- Total hop count
- Weighted hop count (switch hop > direct hop)
- Congestion penalty for heavily shared links

### Temporal Cost

Measures quality of temporal assignments.

Reference formula:

`temporal_cost = alpha * slot_util + beta * reg_pressure + gamma * tag_pressure`

with implementation-defined coefficients `alpha`, `beta`, and `gamma`.

Examples:

- Slot utilization efficiency
- Register usage pressure
- Tag-range pressure relative to tag width

### Throughput and Latency Proxies

Predicts performance impact before backend timing is available.

Reference formula:

`perf_proxy_cost = critical_path_est + ii_pressure + queue_pressure`

where each term is a normalized proxy (not a signoff timing number).

Examples:

- Critical mapped path estimate
- Initiation interval pressure proxy
- Queue-depth pressure proxy on load/store adapters

### Configuration Footprint

Measures runtime configuration size and programming overhead.

Reference formula:

`config_footprint = non_default_words / total_config_words`

optionally extended with weighted terms for sparse updates or table occupancy.

Examples:

- Number of configured nodes
- Number of non-default config words
- Temporal table occupancy

## Normalization

Each metric should be normalized before weighting, so weights are stable across
different graph sizes.

Recommended normalization:

- Count-like metrics: divide by relevant graph size
- Delay-like metrics: divide by technology reference delay
- Footprint metrics: divide by total available config capacity

## Weight Profiles

Named weight profiles define weight vectors for the core metric
families. The vector format is:

`[placement_pressure, routing_cost, temporal_cost, perf_proxy, config_footprint]`

| Profile | Weights | Goal |
|---------|---------|------|
| `balanced` | [1.0, 1.0, 0.5, 0.5, 0.1] | Even tradeoff across all metrics |
| `cpsat_full` | [1.0, 1.0, 0.5, 0.5, 0.1] | Same as balanced (CP-SAT mode selection, not weight change) |
| `heuristic_only` | [1.0, 1.0, 0.5, 0.5, 0.1] | Same as balanced (heuristic mode selection, not weight change) |
| `throughput_first` | [0.3, 0.5, 0.3, 2.0, 0.1] | Prioritize performance proxies (critical path, II) |
| `area_power_first` | [2.0, 0.5, 0.5, 0.2, 1.0] | Prioritize compact placement and minimal config |
| `deterministic_debug` | [1.0, 1.0, 0.5, 0.0, 0.0] | Disable noisy proxies; maximize reproducibility |

Notes:

- `cpsat_full` and `heuristic_only` select the solver mode (see
  [spec-mapper-algorithm.md](./spec-mapper-algorithm.md)) but use
  the same cost weights as `balanced`. They affect search strategy,
  not objective function.
- `deterministic_debug` zeroes performance and config metrics to
  eliminate sources of non-determinism in cost comparison.
- Weight values are tunable parameters. The values above are initial
  defaults; implementations should expose these as inspectable and
  overridable configuration.
- All weights are non-negative. A weight of 0.0 disables that metric.

## Invalid Mapping Penalty

Invalid mappings are not scored as normal candidates. Recommended policy:

- Hard reject invalid mapping from candidate pool, or
- Assign an explicit infinite cost sentinel

Both policies are acceptable if behavior is deterministic and documented.

## Cost and Search Integration

Algorithms may use cost in different ways:

- Greedy local selection
- Global iterative improvement
- Temperature-based acceptance
- Learned policy ranking

All methods must preserve hard constraints from
[spec-mapper-model.md](./spec-mapper-model.md).

## Related Documents

- [spec-mapper.md](./spec-mapper.md)
- [spec-mapper-model.md](./spec-mapper-model.md)
- [spec-mapper-algorithm.md](./spec-mapper-algorithm.md)
