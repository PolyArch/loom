# Loom Co-Simulation RTL Backend Specification

## Overview

This document defines how Loom `cosim` binds to RTL simulators
(SystemVerilog-based) through CIRCT ESI cosim infrastructure. It is
authoritative for RTL backend integration behavior.

General protocol and runtime contracts are defined in:

- [spec-cosim-protocol.md](./spec-cosim-protocol.md)
- [spec-cosim-runtime.md](./spec-cosim-runtime.md)

Simulation/waveform tool priority follows
[spec-adg-tools.md](./spec-adg-tools.md): Synopsys VCS/Verdi is the primary
flow, and Verilator/GTKWave is the secondary open-source flow.

## Required CIRCT Collateral

A conforming RTL backend integration must include CIRCT ESI cosim collateral:

- `Cosim_DpiPkg.sv`
- `Cosim_Endpoint.sv`
- `Cosim_Manifest.sv`
- `Cosim_CycleCount.sv` (recommended; optional if cycle stats unsupported)
- DPI shared library implementing cosim server entry points

These files are provided by CIRCT runtime packaging.

## Mandatory Module-Level Integration Rules

### Manifest injection

Instantiate exactly one manifest provider module for the simulated design and
publish compressed manifest bytes before host discovery.

### Endpoint binding

For each host-visible application channel, instantiate and connect endpoint
logic that preserves valid/ready handshake semantics:

- `Cosim_Endpoint_FromHost` for host-to-device traffic
- `Cosim_Endpoint_ToHost` for device-to-host traffic

### Cycle counter service

If cycle/perf stats are supported, instantiate one cycle counter service
provider (`Cosim_CycleCount` or equivalent).

## Runtime Startup Contract

RTL backend startup must:

1. initialize DPI cosim server
2. register all channels/endpoints
3. publish manifest
4. publish listening port metadata (for example `cosim.cfg`)
5. enter simulation run loop

If startup fails before endpoint registration is complete, backend must report
fatal initialization error and exit.

## Environment and Connection Behavior

Common environment/control variables:

- `COSIM_PORT`: requested server port (optional; dynamic port allowed)
- `COSIM_DEBUG_FILE`: optional debug log path
- `ESI_COSIM_HOST`: host override for client connection
- `ESI_COSIM_PORT`: port override for client connection

Dynamic port mode must provide discovery metadata usable by host runtime
(typically `cosim.cfg` containing the selected port).

## MMIO and config_mem Binding

RTL backend must expose MMIO-compatible channel/service endpoints used by host
runtime to program `config_mem`.

Required behavior:

- preserve request ordering for MMIO commands
- map MMIO addresses to generated config controller address space
- return read/write completion results deterministically

Address and word-layout authority remains in
[spec-fabric-config_mem.md](./spec-fabric-config_mem.md).

## Simulation Thread Semantics

Even if simulator internally uses multiple host threads, the modeled hardware
time progression must be equivalent to one global simulation timeline.

Required properties:

- one authoritative cycle counter
- deterministic reset behavior
- deterministic ordering of host-visible message events for deterministic mode

## Simulator Support

A conforming implementation may target one or more simulators. Simulator-
specific scripts are allowed, but they must preserve the same
channel/service contract.

Priority policy:

1. Synopsys VCS for RTL simulation and Synopsys Verdi for waveform/debug.
2. Verilator with GTKWave as the secondary open-source fallback.

Simulator selection must not change:

- manifest semantics
- channel names and types
- lifecycle ordering at protocol level

## Trace and Performance Support

RTL backend must provide hook points to generate telemetry events in schema
defined by [spec-cosim-trace.md](./spec-cosim-trace.md).

Trace collection may use channel streaming and/or sidecar files. If both are
present, channel stream is authoritative for online host feedback.

## Error Handling Requirements

RTL backend must classify and report:

- endpoint registration failures
- manifest publication failures
- channel protocol violations
- MMIO/configuration failures
- runtime fatal simulation errors

These errors map to categories in
[spec-cosim-protocol.md](./spec-cosim-protocol.md).

## Related Documents

- [spec-cosim.md](./spec-cosim.md)
- [spec-cosim-protocol.md](./spec-cosim-protocol.md)
- [spec-cosim-runtime.md](./spec-cosim-runtime.md)
- [spec-cosim-trace.md](./spec-cosim-trace.md)
- [spec-cosim-validation.md](./spec-cosim-validation.md)
- [spec-adg-sv.md](./spec-adg-sv.md)
- [spec-adg-tools.md](./spec-adg-tools.md)
- [spec-fabric-config_mem.md](./spec-fabric-config_mem.md)
