# Multi-Kernel Application Model Specification

## Overview

This document specifies the multi-kernel application model used by the
heterogeneous multi-core CGRA framework. A multi-kernel application is
represented as a `system.kernel_dag`: a directed graph of kernels with
typed dependency edges, where each kernel corresponds to a `handshake.func`
that can be mapped to a single CGRA core.

The kernel DAG supports feedback cycles with explicit initial-token
semantics, enabling iterative and pipelined multi-kernel applications.

## Operation: `system.kernel_dag`

A `system.kernel_dag` defines the application-level structure of a
multi-kernel workload. It is nested inside a `system.design`.

### Syntax

```mlir
system.kernel_dag @app_name {
  system.kernel @k0 {
    func_ref = @handshake_func_a,
    token_rate = 1.0 : f64,
    mem_footprint = 4096 : i64,
    mem_pattern = "streaming"
  }
  system.kernel @k1 {
    func_ref = @handshake_func_b,
    token_rate = 0.8 : f64,
    mem_footprint = 8192 : i64,
    mem_pattern = "reuse_heavy"
  }
  system.dep {
    src = @k0, src_port = 0 : i64,
    dst = @k1, dst_port = 0 : i64,
    type = !dataflow.tagged<!dataflow.bits<32>, i4>,
    bandwidth_req = 2.0 : f64
  }
  system.dep {
    src = @k1, src_port = 0 : i64,
    dst = @k0, dst_port = 1 : i64,
    type = !dataflow.tagged<!dataflow.bits<32>, i4>,
    bandwidth_req = 1.0 : f64,
    initial_tokens = 1 : i64
  }
}
```

### Nested Operations

A `system.kernel_dag` contains two kinds of nested operations:

- `system.kernel`: a single kernel node
- `system.dep`: a data dependency edge between two kernels

## Operation: `system.kernel`

A kernel node represents a single dataflow computation graph that can be
mapped to one CGRA core.

### Attributes

- `func_ref` (symbol reference): references a `handshake.func` that defines
  this kernel's computation. The referenced function must exist.

- `token_rate` (f64): estimated sustained token production/consumption rate
  in tokens per cycle. This is a soft analytical estimate derived from DFG
  structure (number of operations, critical path depth, pipeline initiation
  interval). Used as input to the cost model, not as a hard scheduling
  constraint.

- `mem_footprint` (i64): estimated working set size in bytes. Represents the
  total memory that this kernel actively accesses during execution.

- `mem_pattern` (string): memory access pattern characterization. Valid
  values:
  - `"streaming"`: sequential or strided access with no temporal reuse.
    Data is used once and not revisited.
  - `"reuse_heavy"`: data is accessed multiple times with significant
    temporal locality. Working set fits (partially or fully) in cache.
  - `"random"`: unpredictable access pattern with minimal spatial or
    temporal locality.
  - `"mixed"`: combination of patterns. Cost model uses `mem_footprint` as
    the primary predictor.

- `required_ops` (list of string, derived): the set of MLIR operation names
  required by this kernel's `handshake.func`. Derived automatically from
  the referenced function by walking all operations in the function body.
  Used for compute-capability matching during partitioning.

### Constraints

- `func_ref` must reference an existing `handshake.func`.
- `token_rate` must be positive.
- `mem_footprint` must be >= 0.
- `mem_pattern` must be one of the valid values listed above.
- Kernel names must be unique within a `system.kernel_dag`.

## Operation: `system.dep`

A dependency edge represents a data flow between two kernels.

### Attributes

- `src` (symbol reference): source kernel.
- `src_port` (i64): output port index on the source kernel.
- `dst` (symbol reference): destination kernel.
- `dst_port` (i64): input port index on the destination kernel.
- `type`: the payload type carried by this dependency. For stream mode:
  `!dataflow.bits<N>`, `!dataflow.tagged<...>`, or `none`. For MMIO
  mode: `memref<...>`. Determines the per-token payload size.
- `bandwidth_req` (f64): required bandwidth in tokens per cycle. This is
  a soft analytical estimate. The cost model produces warnings when total
  required bandwidth on a NoC link exceeds its capacity.
- `initial_tokens` (i64, optional): number of initial tokens present in
  this edge before execution begins. Required for feedback edges (back-edges
  in cycles). Default: 0 (forward edge).
- `comm_mode` (string, optional): communication mode. Valid values:
  - `"stream"`: streaming data transfer (default).
  - `"mmio"`: memory-mapped access (one kernel reads/writes another's
    exported memory).

### Constraints

- `src` and `dst` must reference existing kernels within the same
  `system.kernel_dag`.
- Port indices must be valid for the referenced kernels.
- `bandwidth_req` must be >= 0.
- `initial_tokens` must be >= 0.
- `comm_mode` must be one of the valid values.

## Feedback Cycles

The kernel DAG permits cycles, representing iterative multi-kernel
computations. Cycles must satisfy the **initial-token rule**:

**Every cycle in the kernel DAG must contain at least one edge with
`initial_tokens >= 1`.**

This rule ensures that the cyclic computation can begin execution: initial
tokens provide the data necessary to break the circular dependency at
startup. This is analogous to the initial-token requirement in Kahn process
networks and synchronous dataflow graphs.

### Cycle Detection and Validation

The verifier performs cycle detection on the kernel DAG. For each cycle
found, it checks that at least one edge in the cycle has
`initial_tokens >= 1`. Violations produce a verifier error.

A DAG with no cycles (purely acyclic) is trivially valid.

### Steady-State Semantics

For cyclic kernel DAGs, execution follows a pipeline-fill / steady-state /
drain model:

1. **Fill**: Initial tokens flow through the first iteration. Kernels that
   receive initial tokens can begin execution immediately. Other kernels in
   the cycle wait for their input dependencies.

2. **Steady state**: After the pipeline fills, kernels in the cycle
   execute on different data iterations with pipeline overlap. Kernels
   assigned to different cores execute concurrently; kernels sharing a
   core execute sequentially across epochs (see epoch constraints in
   [spec-hetero-scheduler.md](./spec-hetero-scheduler.md)).

3. **Drain**: When input data is exhausted, the pipeline drains as
   remaining tokens propagate through the cycle.

The scheduler models steady-state throughput for cyclic kernel DAGs.
See [spec-hetero-scheduler.md](./spec-hetero-scheduler.md).

## Kernel Port Model

Each kernel has input and output ports derived from its `handshake.func`:

- **Input ports**: correspond to `handshake.func` block arguments
  (function parameters). These receive data from other kernels or from
  system-level I/O.

- **Output ports**: correspond to `handshake.return` operands (function
  results). These produce data consumed by other kernels or by
  system-level I/O.

Port types follow the `handshake.func` signature. The system dialect
does not introduce new port types; it references existing dataflow
dialect types and standard MLIR types (memref for MMIO).

### System I/O Ports

Kernel ports that are not connected to any `system.dep` edge are
**system I/O ports**: they receive data from or produce data to the
external environment (host, off-chip memory, other subsystems). The
scheduler treats system I/O ports as external bandwidth sources/sinks.

## Kernel Characterization

Beyond the explicit attributes on `system.kernel`, the framework derives
additional kernel characteristics from the referenced `handshake.func`:

| Characteristic | Source | Usage |
|---------------|--------|-------|
| `required_ops` | Walk `handshake.func` body ops | Partitioning compatibility |
| `op_count` | Count operations in `handshake.func` | Resource demand estimation |
| `critical_path_depth` | Longest path in DFG | Latency estimation |
| `memory_op_count` | Count load/store operations | Memory bandwidth demand |
| `fan_out_degree` | Max SSA value fan-out | Routing pressure estimate |

These derived characteristics are read-only metadata consumed by the
scheduler and cost model. They do not modify the kernel or its
`handshake.func`.

## Related Documents

- [spec-hetero.md](./spec-hetero.md)
- [spec-hetero-system.md](./spec-hetero-system.md)
- [spec-hetero-scheduler.md](./spec-hetero-scheduler.md)
- [spec-dataflow.md](./spec-dataflow.md)
- [spec-mapper-model.md](./spec-mapper-model.md)
