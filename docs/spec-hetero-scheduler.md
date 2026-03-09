# System Scheduler Specification

## Overview

The system scheduler assigns multi-kernel applications to heterogeneous
multi-core CGRA systems. It performs three functions:

1. **Partitioning**: assign each kernel to a compatible core
2. **NoC routing**: allocate inter-core communication paths for kernel
   dependencies
3. **Epoch scheduling**: determine the execution order and reconfiguration
   points

The scheduler produces compile-time-static assignments. There is no runtime
scheduler or dynamic kernel migration.

This document defines scheduler scope, interfaces, and constraint classes.
It is the system-level analog of [spec-mapper.md](./spec-mapper.md), which
handles single-kernel-to-single-core mapping.

## Scheduler Inputs and Outputs

### Inputs

- A `system.design` containing heterogeneous cores, NoC topology, cache
  nodes, and external memory endpoints. Core capability summaries are
  derived from referenced `fabric.module` definitions.
- A `system.kernel_dag` containing kernel nodes with annotations and
  dependency edges (possibly with feedback cycles).
- Optional scheduler policy parameters (optimization weights, search
  budget, deterministic seed).

### Outputs

- **Partition map**: `kernel -> core` assignment for every kernel.
- **Route allocation**: `dependency_edge -> NoC_path` for every
  inter-core dependency.
- **Epoch schedule**: per-core kernel execution ordering with
  reconfiguration points and epoch boundaries.
- **Diagnostics**: when no valid assignment exists, actionable failure
  messages identifying the constraint violation.

## Functional Responsibilities

The scheduler must perform all of the following:

1. Extract capability summaries from all cores.
2. Extract kernel characteristics from all kernel nodes.
3. Build the NoC topology graph from links and routers.
4. Assign kernels to compatible cores (partitioning).
5. Allocate NoC paths for inter-core dependencies (routing).
6. Compute epoch assignments respecting data dependencies and
   reconfiguration constraints (scheduling).
7. Optionally invoke the existing mapper as a feasibility oracle.
8. Report failures with actionable diagnostics.

The scheduler must not modify the `system.design`, `system.kernel_dag`,
or any referenced `fabric.module`. All inputs are read-only.

## Partitioning

### Two-Phase Validation

Partitioning uses a two-phase approach:

**Phase 1 (Fast)**: Check kernel requirements against core capability
summaries. A kernel is compatible with a core if:

- `kernel.required_ops` is a subset of `core.supported_ops`
- `kernel.op_count <= core.pe_count * max(1, core.temporal_slots)`
  (optimistic capacity upper bound)
- `kernel.mem_footprint <= core.memory_capacity` (if scratchpad-bound)
- `kernel.memory_op_count <= core.memory_ports`

The capacity check is an optimistic upper bound that admits candidates
for Phase 2 validation. It assumes all functional units can use temporal
slots. For cores where some PEs are spatial-only and others temporal,
the actual feasibility is confirmed by the Phase 2 mapper oracle. Phase
1 errs on the side of inclusion (may admit pairs that Phase 2 rejects),
not exclusion.

Phase 1 is fast (set intersection and comparisons) and produces a
**candidate core set** for each kernel.

**Phase 2 (Oracle, Optional)**: For each (kernel, core) pair in the
candidate set, invoke the existing Loom mapper
([spec-mapper.md](./spec-mapper.md)) on (kernel.func_ref,
core.module_ref). If the mapper succeeds, the assignment is confirmed
feasible. If the mapper fails, the pair is removed from the candidate set.

Phase 2 is expensive (full place-and-route per pair) and is invoked
selectively:
- Always after final partition selection (validation)
- Optionally during search to prune infeasible candidates early
- Skippable when using fast analytical mode only

### Heterogeneity Matching

Partitioning considers all four heterogeneity dimensions:

| Dimension | Matching Rule |
|-----------|--------------|
| Compute | `kernel.required_ops` subset of `core.supported_ops` |
| Capacity | `kernel.op_count <= core.pe_count * max(1, core.temporal_slots)` (optimistic upper bound; Phase 2 validates exact feasibility) |
| Memory | `kernel.mem_footprint <= core.memory_capacity` and `kernel.memory_op_count <= core.memory_ports` |
| NoC | Communication cost depends on core position in NoC topology |

The partitioner optimizes a combined objective across all dimensions
(see [spec-hetero-cost.md](./spec-hetero-cost.md)).

### Partition Constraints

- **(P1)** Every kernel must be assigned to exactly one core.
- **(P2)** A kernel's required ops must be a subset of its assigned core's
  supported ops.
- **(P3)** Multiple kernels may be assigned to the same core (they execute
  sequentially across epochs on that core).
- **(P4)** A core may have zero assigned kernels (idle core).

## NoC Routing

### Route Allocation

For each inter-core dependency edge (where `src_kernel` and `dst_kernel`
are assigned to different cores), the scheduler allocates NoC routing
path(s):

```
For stream dependencies:
  path = shortest_path(src_core, dst_core, noc_topology)

For MMIO dependencies (request + response):
  forward_path = shortest_path(src_core, dst_core, noc_topology)
  reverse_path = shortest_path(dst_core, src_core, noc_topology)
```

The routing algorithm must produce deadlock-free routes. See
[spec-hetero-noc.md](./spec-hetero-noc.md) for deadlock avoidance
strategies.

### Intra-Core Dependencies

When `src_kernel` and `dst_kernel` are assigned to the same core, their
dependency does not use the NoC. The data transfer is handled through the
core's internal memory (scratchpad) or registers, and the dependency
becomes a sequencing constraint in the epoch schedule.

### Route Constraints

- **(R1)** Every inter-core dependency must have an allocated route.
- **(R2)** Routes must follow physically connected links and routers.
- **(R3)** The set of all allocated routes must be deadlock-free (acyclic
  channel dependency graph).
- **(R4)** Bandwidth oversubscription is a soft warning, not a hard error.
  The cost model degrades throughput estimates accordingly.

## Epoch Scheduling

### Epoch Model

An **epoch** is a time interval during which a fixed set of kernels
executes on their assigned cores. Between epochs, cores may be
reconfigured to run different kernels.

```
Epoch i:
  core_0 runs kernel_a
  core_1 runs kernel_b
  core_2 idle
  inter-core: kernel_a -> kernel_b via NoC path P

Reconfiguration (core_0: kernel_a -> kernel_c)

Epoch i+1:
  core_0 runs kernel_c
  core_1 runs kernel_b (continues)
  core_2 runs kernel_d
```

### Epoch Duration

Analytical epoch duration:

```
epoch_duration = max(kernel_execution_time[core] for core in active_cores)
              + max(link_drain_time[link] for link in active_links)
```

Reconfiguration time between epochs is accounted for separately by the
cost model (see `reconfig_overhead` in
[spec-hetero-cost.md](./spec-hetero-cost.md)), not included in
`epoch_duration`. This avoids double-counting.

where:
- `kernel_execution_time`: estimated from kernel's DFG depth, PE count,
  and token rate
- `link_drain_time`: time for in-flight tokens to propagate through
  buffered links (estimated as `buffer_depth / bandwidth`)

### Acyclic Kernel DAG Scheduling

For acyclic kernel DAGs, epoch assignment uses topological ordering:

```
for each kernel in topological_order(kernel_dag):
  epoch[kernel] = max(epoch[pred] + 1 for pred in predecessors)
                  if kernel shares a core with any predecessor,
                  else max(epoch[pred] for pred in predecessors)
```

The "+1" is required when a kernel shares a core with its predecessor
because they cannot execute simultaneously (same core, sequential
epochs).

### Cyclic Kernel DAG Scheduling

For kernel DAGs with feedback cycles, the scheduler uses **steady-state
scheduling**:

1. **Cycle breaking**: identify back-edges (edges with
   `initial_tokens >= 1`) and temporarily remove them, producing an
   acyclic DAG.

2. **Baseline scheduling**: schedule the acyclic DAG using topological
   ordering (above).

3. **Pipeline overlap**: the back-edges represent cross-iteration
   dependencies. In steady state, different iterations of the cyclic
   subgraph overlap across epochs:

   ```
   Epoch 0: iteration 0, kernel_A
   Epoch 1: iteration 0, kernel_B  |  iteration 1, kernel_A
   Epoch 2: iteration 0, kernel_C  |  iteration 1, kernel_B  |  iteration 2, kernel_A
   ...
   ```

4. **Steady-state throughput**: after the pipeline fills, the throughput
   is determined by the bottleneck epoch (longest epoch in the cycle):

   ```
   steady_state_throughput = 1 / max(epoch_duration[epoch] for epoch in cycle)
   ```

5. **Initial tokens**: the `initial_tokens` count on back-edges determines
   how many iterations can be in-flight simultaneously during fill.

### Epoch Constraints

- **(E1)** A kernel can execute only after all its forward-edge
  predecessors have completed (or, for steady-state, have produced their
  output tokens for this iteration).
- **(E2)** At most one kernel per core per epoch (no time-sharing within
  an epoch).
- **(E3)** Reconfiguration is required when a core switches from one
  kernel to another between epochs.
- **(E4)** Epoch ordering must be consistent with the kernel DAG's
  dependency structure (no epoch assignment where a kernel's input
  depends on output from a later epoch, unless pipeline overlap through
  back-edges is valid).

## Scheduler-Mapper Contract

The system scheduler and the existing mapper have a well-defined contract:

| Scheduler responsibility | Mapper responsibility |
|-------------------------|----------------------|
| Kernel-to-core assignment | Kernel-to-hardware-node assignment within a core |
| Inter-core NoC routing | Intra-core switch/FIFO routing |
| Epoch-level scheduling | N/A (mapper is epoch-unaware) |
| System-level cost estimation | Per-core mapping quality |

The scheduler invokes the mapper as a black-box feasibility oracle:

```
mapper_result = mapper.run(kernel.func_ref, core.module_ref)
if mapper_result.success:
  // assignment is feasible
else:
  // assignment is infeasible; remove from candidate set
```

The scheduler does not inspect `MappingState` internals. It only uses the
success/failure signal.

Note: `config_size` needed by the cost model's reconfiguration overhead
calculation is a static property of the core's `fabric.module`
(specifically the `config_mem` depth attribute), not a mapper output.
The scheduler reads `config_size` directly from the `fabric.module`
definition without invoking the mapper. See
[spec-fabric-config_mem.md](./spec-fabric-config_mem.md).

## Failure Modes

Scheduler failure is expected when no feasible assignment exists under
hard constraints.

Failure diagnostics should include:

| Failure class | Diagnostic content |
|--------------|-------------------|
| No compatible core | Kernel name, required ops, available core ops |
| Resource exhaustion | Core name, resource type, demand vs. capacity |
| No NoC path | Source core, destination core, topology info |
| Deadlock | Conflicting flows, cycle in channel dependency graph |
| Infeasible epoch | Dependency edge, epoch conflict details |
| Mapper rejection | Kernel name, core name, mapper failure summary |

## Determinism and Reproducibility

Given identical inputs and deterministic policy settings, scheduler output
must be reproducible. Sources of non-determinism (randomized search, tie
breaks) must be controlled by explicit seed parameters.

## Related Documents

- [spec-hetero.md](./spec-hetero.md)
- [spec-hetero-system.md](./spec-hetero-system.md)
- [spec-hetero-kernel.md](./spec-hetero-kernel.md)
- [spec-hetero-noc.md](./spec-hetero-noc.md)
- [spec-hetero-cost.md](./spec-hetero-cost.md)
- [spec-mapper.md](./spec-mapper.md)
- [spec-mapper-model.md](./spec-mapper-model.md)
