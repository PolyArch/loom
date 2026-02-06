# Loom Co-Simulation Architecture Specification

## Overview

This document is the authoritative architecture specification for Loom `cosim`.
It defines runtime components, lifecycle states, and the concurrency model that
connects host software to one accelerator simulation thread.

Transport message layouts are defined in
[spec-cosim-protocol.md](./spec-cosim-protocol.md). Host API details are
defined in [spec-cosim-runtime.md](./spec-cosim-runtime.md).

## Terminology

- **Simulation thread**: the single thread that owns simulation time
  progression and device-side communication.
- **Host worker thread**: any host thread that submits requests to `cosim`
  (configuration, invocation, I/O, trace queries).
- **Dispatcher**: host runtime component that serializes requests from multiple
  host worker threads onto one simulation-thread-facing execution stream.
- **Session**: one connected runtime instance from `connect` to `disconnect`.
- **Epoch**: one configuration generation boundary; starts after successful
  `config_mem` programming and ends before reconfiguration.

## Component Model

A conforming `cosim` system contains the following components.

1. **Compile-time artifact provider**
   - Supplies Stage A-D outputs:
     - interface/manifest metadata
     - mapper `config_mem` image
     - backend executable model

2. **Host runtime front-end**
   - Exposes thread-safe user-facing APIs.
   - Receives requests from host worker threads.

3. **Host dispatcher (single owner of device connection)**
   - Owns the ESI accelerator connection object.
   - Serializes control-plane operations and ordering-sensitive data-plane
     operations.
   - Enforces deterministic ordering when deterministic mode is enabled.

4. **Backend adapter**
   - Bridges dispatcher operations to concrete backend transport.
   - Supports at least one of:
     - SystemC backend
     - RTL backend

5. **Simulation thread and kernel**
   - Owns global cycle progression.
   - Services ESI endpoints.
   - Exposes event hooks for trace/perf emission.

6. **Result oracle (CPU reference path)**
   - Computes expected outputs from the same host input vectors.
   - Produces compare verdicts.

## Execution Planes

`cosim` behavior is organized into three planes.

### Control Plane

Used for session-level operations:

- connect/disconnect
- manifest discovery
- `config_mem` programming
- start/stop and epoch boundaries
- error and status reporting

### Data Plane

Used for accelerator payload exchange:

- input payload transfer from host to accelerator
- output payload transfer from accelerator to host
- optional host memory service traffic

### Observability Plane

Used for instrumentation:

- trace event stream
- performance statistics
- cycle/frequency reporting

## Lifecycle State Machine

A session follows the state machine below.

| State | Meaning | Entered by | Exit condition |
|------|---------|------------|----------------|
| `Created` | Session object exists but not connected | runtime constructor | `connect()` |
| `Connected` | Transport connected, manifest not yet validated | successful backend connect | manifest parsed and validated |
| `Ready` | Manifest validated, channels discoverable | manifest+service checks | config upload request |
| `Configured` | Full `config_mem` image loaded for current epoch | config upload complete | start request |
| `Running` | Accelerator invocation in progress | start/invoke accepted | done signal or fatal error |
| `Draining` | Output and trace buffers being drained | run complete | all expected outputs consumed |
| `Verified` | Output comparison completed | compare complete | next epoch or disconnect |
| `Closed` | Session disconnected and resources released | disconnect | terminal |

Transitions that skip mandatory checks are invalid.

## Host Multi-Threading Contract

The runtime must accept concurrent calls from multiple host worker threads,
while preserving single-owner communication with the accelerator backend.

Required behavior:

- user APIs are thread-safe at the front-end boundary
- dispatcher performs serialized execution for ordering-sensitive operations
- each request returns a completion object (future/promise or equivalent)
- completion ordering is deterministic when deterministic mode is enabled

### Required request classes

- session control (`connect`, `disconnect`, `reset`)
- configuration (`load_config`, `read_config` optional)
- invocation control (`invoke`, `wait_done`, `cancel` optional)
- payload transfer (`send_input`, `recv_output`)
- observability (`start_trace`, `stop_trace`, `read_perf`)

## Simulation Thread Contract

The simulation thread is authoritative for cycle order and event order.

Required properties:

- monotonic global cycle counter
- no parallel advancement of simulation time
- backend message servicing integrated with cycle progression
- deterministic event timestamping relative to the global cycle

`cosim` may use helper polling/service threads from the underlying runtime, but
those threads must not violate the single simulation timeline.

## Multi-Core Heterogeneous Accelerator Model

`cosim` must support hardware models where one simulation thread represents
multiple heterogeneous accelerator cores.

Required modeling rules:

- each execution event is attributed to `core_id`
- each hardware activity event is attributed to `hw_node_id`
- trace/perf reports include both `core_id` and `hw_node_id` dimensions where
  applicable
- host runtime may dispatch work from multiple host worker threads to different
  logical cores, but backend execution still follows one global cycle timeline

Core assignment policies are runtime-configurable and do not alter mapper
legality rules.

## Backend Portability Layer

To preserve future real-hardware support, `cosim` architecture defines a
backend-neutral interface boundary:

- session connect/disconnect
- manifest retrieval
- service/channel binding
- control/data transfer operations
- trace/perf collection operations

Backend-specific behavior is constrained to adapters specified in:

- [spec-cosim-backend-systemc.md](./spec-cosim-backend-systemc.md)
- [spec-cosim-backend-rtl.md](./spec-cosim-backend-rtl.md)

The same host runtime front-end and dispatcher contract must remain valid for
future physical backends (for example, FPGA runtime backends).

## Failure Containment

Failure handling must preserve process safety and diagnosability.

Required rules:

- protocol/transport failures transition session to `Closed` or
  implementation-defined terminal error state
- runtime must release channel and backend resources on terminal failure
- partial output comparison must be marked invalid when run termination is not
  clean
- fatal backend error must include enough context to identify lifecycle state,
  epoch, and last successfully completed operation class

Detailed error taxonomy is defined in
[spec-cosim-protocol.md](./spec-cosim-protocol.md) and
[spec-cosim-validation.md](./spec-cosim-validation.md).

## Minimal Reference Architecture

A minimal conforming architecture is:

- one process hosting:
  - N host worker threads
  - one dispatcher thread
  - one optional runtime service/poll thread
- one backend process or in-process kernel that runs one simulation thread
- ESI transport between dispatcher and backend adapter

This reference architecture is sufficient for undergraduate-level implementation
and validation.

## Related Documents

- [spec-cosim.md](./spec-cosim.md)
- [spec-cosim-protocol.md](./spec-cosim-protocol.md)
- [spec-cosim-runtime.md](./spec-cosim-runtime.md)
- [spec-cosim-backend-systemc.md](./spec-cosim-backend-systemc.md)
- [spec-cosim-backend-rtl.md](./spec-cosim-backend-rtl.md)
- [spec-cosim-trace.md](./spec-cosim-trace.md)
- [spec-cosim-validation.md](./spec-cosim-validation.md)
