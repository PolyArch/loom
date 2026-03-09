# Heterogeneous Multi-Core CGRA Specification

## Overview

The heterogeneous multi-core CGRA framework extends Loom from single-kernel,
single-core accelerator compilation to multi-kernel, multi-core system-level
design space exploration. It introduces a `system` dialect that sits above the
existing `fabric` dialect, enabling analytical evaluation of heterogeneous CGRA
architectures with multi-hop NoC interconnect, cache effects, and
reconfiguration-aware scheduling.

This document is the top-level specification. Detailed definitions are
delegated to specialized documents:

- System dialect operations: [spec-hetero-system.md](./spec-hetero-system.md)
- Multi-kernel application model: [spec-hetero-kernel.md](./spec-hetero-kernel.md)
- Network-on-Chip model: [spec-hetero-noc.md](./spec-hetero-noc.md)
- Cache effect model: [spec-hetero-cache.md](./spec-hetero-cache.md)
- System scheduler: [spec-hetero-scheduler.md](./spec-hetero-scheduler.md)
- Analytical cost model: [spec-hetero-cost.md](./spec-hetero-cost.md)

## Pipeline Position

The hetero framework introduces a new stage that wraps the existing Loom
pipeline. The existing stages (A through E) remain unchanged.

```
Stage F: Multi-Core System Design and Evaluation

  F.1  System Description
       Input:  Multiple fabric.module definitions (from Stage B)
               Multi-kernel application (multiple handshake.funcs from Stage A)
               System topology description (system dialect)
       Output: system.design with cores, NoC links, routers, caches,
               external memory endpoints, and embedded kernel_dag

  F.2  System Scheduling
       Input:  system.design (contains embedded system.kernel_dag)
       Output: Kernel-to-core assignment, NoC route allocation, epoch schedule

  F.3  Per-Core Validation (optional)
       Input:  Each (kernel, core) pair from F.2
       Output: Mapper feasibility confirmation via existing Stage C mapper

  F.4  Analytical Evaluation
       Input:  Schedule from F.2, system topology, kernel characteristics
       Output: Throughput, latency, bandwidth utilization, cache hit rates,
               reconfiguration overhead, Perf/Watt estimates
```

The existing single-core flow remains the workhorse for per-core mapping.
The system scheduler orchestrates multiple invocations of the existing mapper
as a feasibility oracle, but does not modify mapper internals.

## Design Principles

### Layered Extension

The `system` dialect sits strictly above the `fabric` dialect. It references
`fabric.module` definitions but never modifies them. The fabric dialect
remains purely hardware-structural; the system dialect adds
scheduling/orchestration semantics.

```
system.design                     <-- NEW: multi-core topology
  |-- system.core -> fabric.module    (reference, not modification)
  |-- system.link                     (inter-core communication)
  |-- system.router                   (NoC routing nodes)
  |-- system.cache                    (analytical cache nodes)
  |-- system.extmem                   (external memory endpoint)
  |-- system.kernel_dag               (application kernel graph)
```

### Existing Mapper as Oracle

The system scheduler does not duplicate mapper logic. When it needs to verify
that a kernel can actually map to a specific core's hardware, it invokes the
existing Stage C mapper on the (kernel.func_ref, core.module_ref) pair. The mapper
result is used as a feasibility signal only; the system scheduler does not
consume detailed mapping state.

### Analytical Evaluation

The framework produces analytical estimates, not cycle-accurate simulation
results. This design choice enables rapid design space exploration without
requiring RTL generation or simulation infrastructure. All cost metrics are
analytical models parameterized by hardware attributes and kernel
characteristics.

### Backward Compatibility

Existing single-core Loom workflows are completely unaffected. A standalone
`fabric.module` without a wrapping `system.design` compiles and maps
identically to the current flow. The system dialect is an optional addition.

### Static Scheduling

The system scheduler produces compile-time-static assignments. There is no
runtime scheduler or dynamic kernel migration. Scheduling decisions are fixed
at design time and represent one point in the design space.

## Terminology

| Term | Definition |
|------|-----------|
| **Core** | A single CGRA accelerator, represented by one `fabric.module` with a specific hardware configuration. |
| **System** | A collection of heterogeneous cores connected by a NoC. Represented by `system.design`. |
| **Kernel** | A single dataflow computation graph (`handshake.func`) that can be mapped to one core. |
| **Kernel DAG** | A directed graph of kernels with data dependency edges, optionally containing feedback cycles. |
| **Epoch** | A time interval during which a fixed set of kernels is executing on their assigned cores. Epoch boundaries involve reconfiguration. |
| **NoC** | Network-on-Chip: the inter-core communication infrastructure consisting of links and routers. |
| **Link** | A point-to-point connection segment between two NoC endpoints (core ports or router ports) with defined bandwidth and latency. |
| **Router** | An intermediate NoC node that forwards data between links without computation. |
| **Cache node** | An analytical abstraction representing a cache in the memory hierarchy, used for hit rate and latency estimation. |
| **External memory** | An analytical abstraction (`system.extmem`) representing the external memory interface (e.g., DRAM controller). Attached to a NoC position (a core or router) via its `position` attribute; it is not a topology vertex itself. |
| **Capability summary** | A compact description of a core's hardware resources extracted from its `fabric.module`: supported ops, PE count, memory capacity, temporal slots, frequency. |
| **Stream communication** | Inter-core data transfer using dataflow types (`!dataflow.tagged<...>`, `!dataflow.bits<N>`, or `none`), analogous to AXI-Stream. |
| **MMIO communication** | Inter-core memory-mapped data transfer using memref-style semantics, analogous to AXI-MM. |

## Component Summary

| Component | Operations / Concepts | Spec |
|-----------|----------------------|------|
| System topology | `system.design`, `system.core`, `system.link`, `system.router`, `system.cache`, `system.extmem` | [spec-hetero-system.md](./spec-hetero-system.md) |
| Application model | `system.kernel_dag`, kernel nodes, dependency edges, feedback cycles | [spec-hetero-kernel.md](./spec-hetero-kernel.md) |
| NoC model | Link semantics, router semantics, multi-hop routing, deadlock avoidance, bandwidth/latency | [spec-hetero-noc.md](./spec-hetero-noc.md) |
| Cache model | Cache node semantics, reuse distance, hit rate estimation, spatial interaction | [spec-hetero-cache.md](./spec-hetero-cache.md) |
| Scheduler | Kernel-to-core partitioning, NoC route allocation, epoch scheduling | [spec-hetero-scheduler.md](./spec-hetero-scheduler.md) |
| Cost model | Compute utilization, bandwidth utilization, cache effects, reconfiguration, Perf/Watt | [spec-hetero-cost.md](./spec-hetero-cost.md) |

## Heterogeneity Dimensions

The framework models four dimensions of core heterogeneity:

1. **Compute capability**: Different cores support different operation sets
   (e.g., one core specialized for ML ops, another for general arithmetic).
   Represented by the capability summary's `supported_ops` field.

2. **Memory capacity and bandwidth**: Different cores have different amounts
   of on-chip scratchpad and different numbers of memory ports. Represented
   by `memory_capacity`, `memory_ports`, and related fields.

3. **NoC bandwidth and latency**: Different inter-core links have different
   bandwidth and latency characteristics. Represented by `system.link`
   attributes.

4. **Temporal capacity**: Different cores have different numbers of temporal
   PE instruction slots, affecting how many operations each PE can
   time-multiplex within a single kernel mapping. Cores with more
   temporal slots can accommodate larger kernels. Represented by
   `temporal_slots` in the capability summary.

A critical cross-dimensional interaction is the **spatial cache effect**:
the physical placement of kernels relative to cache nodes on the NoC
determines effective memory latency. This three-way interaction (kernel
placement x cache location x memory access pattern) is modeled explicitly
in the cost model. See [spec-hetero-cache.md](./spec-hetero-cache.md).

## Communication Modes

Two inter-core communication modes are supported analytically:

### Stream Mode

Stream communication uses dataflow dialect types:
`!dataflow.tagged<!dataflow.bits<N>, iK>`, `!dataflow.bits<N>`, or `none`.
This mode is analogous to AXI-Stream on Versal FPGA NoC.

- Data flows as a sequence of tokens
- Backpressure uses standard valid/ready handshake
- In-order delivery per link (FIFO semantics)
- Suited for pipeline-style inter-kernel communication

### MMIO Mode

MMIO communication uses memref-style semantics for memory-mapped access
across cores. This mode is analogous to AXI-MM on Versal FPGA NoC.

- One core issues load/store requests through the NoC to a remote core's
  exported memory
- Requests carry address and data payloads
- Response carries read data
- Suited for shared-data inter-kernel communication

Both modes use the same physical NoC infrastructure (links, routers) with
different payload types.

## Interaction with Existing Specifications

The hetero framework consumes but does not redefine existing Loom
specifications:

| Existing spec | Interaction |
|---------------|------------|
| [spec-fabric.md](./spec-fabric.md) | `system.core` references `fabric.module` definitions. Fabric semantics unchanged. |
| [spec-mapper.md](./spec-mapper.md) | System scheduler invokes mapper as feasibility oracle. Mapper semantics unchanged. |
| [spec-adg.md](./spec-adg.md) | ADG builder constructs `fabric.module` definitions consumed by system cores. ADG API unchanged. |
| [spec-dataflow.md](./spec-dataflow.md) | Kernel DAG kernels reference `handshake.func` definitions. Dataflow semantics unchanged. |
| [spec-cli.md](./spec-cli.md) | CLI extensions for system-level commands (TBD). |

## Non-Goals

The hetero specification does not standardize:

- Dynamic/runtime kernel scheduling or migration
- Hardware-coherent cache protocols (fill, evict, consistency)
- Multi-clock-domain hardware generation (CDC, async FIFOs). Per-core
  `frequency` attributes are for analytical bandwidth normalization
  only. This does not imply generated hardware supports multiple clock
  domains.
- RTL or SystemC generation for multi-core systems
- Cycle-accurate simulation of multi-core execution
- Host runtime or device driver for multi-core systems

## Related Documents

- [spec-loom.md](./spec-loom.md)
- [spec-hetero-system.md](./spec-hetero-system.md)
- [spec-hetero-kernel.md](./spec-hetero-kernel.md)
- [spec-hetero-noc.md](./spec-hetero-noc.md)
- [spec-hetero-cache.md](./spec-hetero-cache.md)
- [spec-hetero-scheduler.md](./spec-hetero-scheduler.md)
- [spec-hetero-cost.md](./spec-hetero-cost.md)
- [spec-fabric.md](./spec-fabric.md)
- [spec-mapper.md](./spec-mapper.md)
- [spec-adg.md](./spec-adg.md)
- [spec-dataflow.md](./spec-dataflow.md)
