# Loom Co-Simulation Protocol Specification

## Overview

This document is the authoritative protocol specification for Loom `cosim`.
It defines transport layering, required service contracts, channel naming
rules, and session-level sequencing for configuration and execution.

Architecture and threading behavior are defined in
[spec-cosim-architecture.md](./spec-cosim-architecture.md) and
[spec-cosim-runtime.md](./spec-cosim-runtime.md).

## Protocol Layers

Loom `cosim` uses a three-layer protocol stack.

### Layer P0: Transport RPC (CIRCT cosim)

P0 uses CIRCT ESI cosim RPC service (`esi.cosim.ChannelServer`) with methods:

- `GetManifest`
- `ListChannels`
- `SendToServer`
- `ConnectToClientChannel`

This RPC schema is authoritative in:
`externals/circt/lib/Dialect/ESI/runtime/cosim.proto`.

### Layer P1: ESI Service and Channel Contract

P1 defines typed channels and services discovered through the manifest and/or
channel list. Types are interpreted by the ESI runtime and must match manifest
metadata.

### Layer P2: Loom Session Semantics

P2 defines Loom-specific sequencing rules:

- connect and manifest validation
- epoch-scoped configuration loading (`config_mem`)
- invocation and completion
- output drain and verification
- trace/perf collection

P2 never redefines P0 wire encoding or ESI type rules.

## Required Service Contract

A conforming Loom `cosim` session must expose the following logical services.

| Service | Requirement | Purpose |
|---------|-------------|---------|
| `SysInfo` | Required | manifest/version retrieval; optional cycle/frequency |
| `MMIO` | Required | `config_mem` programming and optional status reads |
| Application data channels (typed ESI channels) | Required | accelerator input/output payload exchange |
| `HostMem` | Optional | high-bandwidth host memory transactions |
| Trace service channels | Optional | streaming trace and performance telemetry |

Service discovery and type interpretation are manifest-driven.

## Reserved Channel/Endpoint Namespace

The prefix `__cosim_` is reserved for backend/runtime infrastructure channels.
Loom application channels must not use this prefix.

Known runtime channels in CIRCT cosim implementations include:

- `__cosim_cycle_count.arg`
- `__cosim_cycle_count.result`
- `__cosim_mmio_read_write.arg`
- `__cosim_mmio_read_write.result`
- `__cosim_hostmem_read_req.data`
- `__cosim_hostmem_read_resp.data`
- `__cosim_hostmem_write.arg`
- `__cosim_hostmem_write.result`

Additional helper channels may be added by backend implementations, but they
must remain in reserved namespace.

## Application Channel Naming

Application channels are identified by ESI AppID path and channel name.
Channel identity is the tuple:

- `appid_path`
- `bundle_name`
- `channel_name`

Implementations may materialize that identity into backend-specific channel
strings. Mapping from logical identity to physical channel name must be
manifest-derived and deterministic.

## Configuration Transfer Protocol

Mapper-produced `config_mem` words are transferred using ESI-accessible control
paths (typically MMIO service).

### Required sequence per epoch

1. Session in `Ready` state.
2. Optional accelerator reset assertion (backend-specific).
3. Write all `config_mem` words in authoritative address order from
   [spec-fabric-config_mem.md](./spec-fabric-config_mem.md).
4. Optional readback verification of selected/all words.
5. Optional reset de-assertion.
6. Transition to `Configured`.

Partial in-run reconfiguration is out of scope for baseline `cosim`.

### Address and data authority

- Address allocation is authoritative in `spec-fabric-config_mem.md`.
- Word payload values are authoritative from mapper output contract in
  `spec-mapper-model.md`.

This protocol only defines transfer sequencing.

## Invocation Protocol

After configuration, accelerator invocation follows P2 ordering rules.

### Required invocation phases

1. **Input stage**: host sends required input payloads on typed channels.
2. **Start stage**: host issues invocation start action (implicit or explicit,
   backend dependent).
3. **Run stage**: simulation thread advances until completion condition.
4. **Output stage**: host drains all expected output payloads.
5. **Done stage**: session records terminal status for this invocation.

If the accelerator interface is function-like (`arg/result` bundle), start may
be implicit in first argument message. If a separate start control channel
exists, start must be explicit.

## Message Typing Rules

- Channel payload bytes must conform to manifest type definitions.
- Field order and packing are manifest/runtime-defined; host code must not
  infer layout by ad-hoc C struct assumptions outside runtime type utilities.
- Type mismatch is a protocol error.

## Trace and Performance Channels

When trace/perf is enabled, telemetry travels through dedicated typed channels
or equivalent backend service ports. Event schema and metric schema are defined
in [spec-cosim-trace.md](./spec-cosim-trace.md).

Trace/perf traffic must not perturb functional message ordering.

## Error Classification

`cosim` protocol errors are classified as:

- `TransportError`: RPC connection or streaming failure (P0)
- `DiscoveryError`: manifest/channel/service resolution failure (P1)
- `TypeError`: payload type mismatch or decode/encode failure (P1)
- `SessionOrderError`: invalid P2 state transition (for example invoke before
  configuration)
- `DeviceRuntimeError`: backend-reported runtime fault

Fabric compile/config/runtime error symbols remain authoritative in
[spec-fabric-error.md](./spec-fabric-error.md). This document only defines
protocol-layer categories.

## Determinism Rules

In deterministic mode:

- request serialization order is fixed by dispatcher submission order
- channel bind order is deterministic
- configuration write order is deterministic
- run completion and output drain follow deterministic policy for tie cases

Randomized scheduling for stress testing is allowed only when explicitly
requested and must log seed/configuration.

## Protocol Compatibility

A Loom `cosim` endpoint is compatible when:

- P0 RPC schema is implemented or equivalent adapter is provided
- manifest version is accepted by host runtime policy
- required services for selected run mode are present
- channel types for required application interfaces are resolvable

Compatibility failures must be reported before entering `Configured` state.

## Related Documents

- [spec-cosim.md](./spec-cosim.md)
- [spec-cosim-architecture.md](./spec-cosim-architecture.md)
- [spec-cosim-runtime.md](./spec-cosim-runtime.md)
- [spec-cosim-trace.md](./spec-cosim-trace.md)
- [spec-fabric-config_mem.md](./spec-fabric-config_mem.md)
- [spec-mapper-model.md](./spec-mapper-model.md)
- [spec-fabric-error.md](./spec-fabric-error.md)
