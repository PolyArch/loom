# Loom Co-Simulation SystemC Backend Specification

## Overview

This document defines how Loom `cosim` binds to a SystemC backend generated
from ADG/Fabric outputs. It is authoritative for SystemC-specific runtime
binding behavior.

General protocol and runtime contracts are defined in:

- [spec-cosim-protocol.md](./spec-cosim-protocol.md)
- [spec-cosim-runtime.md](./spec-cosim-runtime.md)

Simulation/waveform tool priority follows
[spec-adg-tools.md](./spec-adg-tools.md): Synopsys VCS/Verdi is the primary
flow, and Verilator/GTKWave is the secondary open-source flow.

## Backend Role

The SystemC backend must provide:

- one simulation thread with global cycle progression
- ESI-compatible service/channel endpoints for host communication
- `config_mem` programming path
- application data channel path
- optional trace/perf channel path

## Required Inputs

SystemC backend consumes:

- generated SystemC model from `exportSysC()` (see
  [spec-adg-sysc.md](./spec-adg-sysc.md))
- manifest metadata for service and channel discovery
- mapper-produced `config_mem` words

## Reference Integration Architecture

A minimal conforming integration can run in one process:

- SystemC kernel thread (authoritative simulation timeline)
- in-process ESI cosim RPC server (`esi::cosim::RpcServer`)
- channel adapters between SystemC stream signals and ESI ports
- MMIO adapter between ESI MMIO requests and `cfg_socket` transactions

### Channel adapter directions

- **From host to SystemC**: ESI read port -> stream input signals
- **From SystemC to host**: stream output signals -> ESI write port

Adapters must preserve valid/ready semantics and ordering.

## Simulation Thread Contract

SystemC backend must keep one authoritative simulation loop.

Required loop semantics:

1. process inbound host requests that are ready at current cycle boundary
2. advance SystemC by one logical cycle step
3. sample outbound events/messages for current cycle
4. update cycle counter and instrumentation state

A typical implementation pattern is:

```cpp
while (!stopRequested) {
  bridge.pumpInbound();
  sc_core::sc_start(cyclePeriod);
  bridge.pumpOutbound();
  ++cycleCount;
}
```

No parallel `sc_start` advancement is allowed.

## MMIO Binding for config_mem

SystemC backend must expose MMIO control service compatible with host runtime
configuration upload.

Required behavior:

- decode MMIO read/write command messages from control channel/service
- map command address to generated config module address space
- execute read/write via `b_transport` on `cfg_socket` (or equivalent)
- return response in request order

Address semantics remain authoritative in
[spec-fabric-config_mem.md](./spec-fabric-config_mem.md).

## Manifest and Service Discovery

Before accepting host invocations, backend must:

1. set manifest payload and manifest version in runtime service
2. register all required service/channel endpoints
3. signal readiness (for example by creating `cosim.cfg` equivalent metadata)

Missing mandatory service registration is a startup failure.

## Required Startup and Shutdown Semantics

### Startup

- initialize SystemC modules and reset state
- initialize cosim RPC server
- register channels/services
- publish manifest
- enter run loop

### Shutdown

- stop run loop
- drain and close endpoint queues
- stop RPC server cleanly
- release SystemC objects

## Trace and Performance Hooks

SystemC backend must provide instrumentation hook points:

- per-cycle callback
- per-node activity callback
- invocation boundary callback

These hooks feed schemas defined in [spec-cosim-trace.md](./spec-cosim-trace.md).
Instrumentation must not modify functional behavior.

## Error Handling Requirements

SystemC backend must report at least:

- startup failures (manifest/service/channel registration)
- MMIO/configuration access errors
- channel protocol violations (type/size/order)
- simulation runtime fatal errors

Errors must map into protocol/runtime error categories defined by
[spec-cosim-protocol.md](./spec-cosim-protocol.md).

## Practical Build Notes

Generated SystemC model requirements are defined in
[spec-adg-sysc.md](./spec-adg-sysc.md). A practical environment typically
includes SystemC 3.0.x and C++17 toolchain support. When multiple simulators
are available, implementations should default to VCS-based execution; Verilator
remains a supported fallback according to
[spec-adg-tools.md](./spec-adg-tools.md).

## Related Documents

- [spec-cosim.md](./spec-cosim.md)
- [spec-cosim-protocol.md](./spec-cosim-protocol.md)
- [spec-cosim-runtime.md](./spec-cosim-runtime.md)
- [spec-cosim-trace.md](./spec-cosim-trace.md)
- [spec-cosim-validation.md](./spec-cosim-validation.md)
- [spec-adg-sysc.md](./spec-adg-sysc.md)
- [spec-adg-tools.md](./spec-adg-tools.md)
- [spec-fabric-config_mem.md](./spec-fabric-config_mem.md)
