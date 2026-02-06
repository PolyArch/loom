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

## Core Metric Families

### Placement Pressure

Measures concentration of mapped software nodes onto limited hardware
resources.

Examples:

- Overuse risk of high-fanout switch neighborhoods
- Occupancy skew across equivalent compute tiles

### Routing Cost

Measures path expense of mapped software edges.

Examples:

- Total hop count
- Weighted hop count (switch hop > direct hop)
- Congestion penalty for heavily shared links

### Temporal Cost

Measures quality of temporal assignments.

Examples:

- Slot utilization efficiency
- Register usage pressure
- Tag-range pressure relative to tag width

### Throughput and Latency Proxies

Predicts performance impact before backend timing is available.

Examples:

- Critical mapped path estimate
- Initiation interval pressure proxy
- Queue-depth pressure proxy on load/store adapters

### Configuration Footprint

Measures runtime configuration size and programming overhead.

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

Implementations should expose named weight profiles for common goals:

- `balanced`: mixed tradeoff
- `throughput_first`: prioritize throughput proxies
- `area_power_first`: prioritize compact/resource-light mapping
- `deterministic_debug`: prioritize stable deterministic tie-break behavior

Profile contents are implementation-defined but should be inspectable.

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
