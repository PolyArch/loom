# Loom Co-Simulation Runtime Specification

## Overview

This document is the authoritative runtime contract for Loom `cosim` host-side
software. It defines the required API surface, threading model, request
serialization rules, and deterministic execution behavior.

Transport details are defined in [spec-cosim-protocol.md](./spec-cosim-protocol.md).
Backend-specific binding is defined in backend sub-specs.

## Runtime Responsibilities

A conforming runtime must:

- connect to a selected backend (`cosim` SystemC or RTL adapter)
- validate manifest and required services
- load mapper-produced `config_mem` words
- accept concurrent host requests from multiple host threads
- serialize device communication through one dispatcher owner
- invoke accelerator and collect outputs
- run CPU reference computation and compare results
- collect optional trace/performance data

## Required API Surface

A conforming implementation must expose equivalent capabilities to the
following logical API. Names may differ, semantics must match.

```cpp
struct CosimConnectOptions {
  std::string backend;            // "cosim" for CIRCT cosim backend
  std::string connection;         // host:port, /path/to/cosim.cfg, or env mode
  bool deterministic = true;
  bool enableTrace = false;
  bool enablePerf = false;
};

struct ConfigImage {
  std::vector<uint32_t> words;    // authoritative order from config_mem spec
};

struct InvocationRequest {
  std::string kernel;
  std::vector<uint8_t> inputPayload;
  uint64_t invocationId;
  std::optional<uint32_t> coreHint;
};

struct InvocationResult {
  bool success;
  std::vector<uint8_t> outputPayload;
  uint64_t cycleStart;
  uint64_t cycleEnd;
  std::optional<uint16_t> deviceErrorCode;
};

class CosimSession {
public:
  static CosimSession connect(const CosimConnectOptions &);
  void loadConfig(const ConfigImage &);
  std::future<InvocationResult> invokeAsync(const InvocationRequest &);
  InvocationResult invoke(const InvocationRequest &);
  void enableTrace(bool on);
  void enablePerf(bool on);
  void disconnect();
};
```

## Threading Model

### Design constraints

Underlying CIRCT runtime documents that `AcceleratorConnection` methods are not
thread-safe. Loom runtime must therefore enforce single-owner access to device
connection objects.

### Required host concurrency model

- public APIs are callable from multiple host worker threads
- all device operations are submitted into a multi-producer queue
- one dispatcher thread consumes the queue and performs device interactions
- each request has a completion future/promise

A runtime may additionally use background polling/service threads offered by
ESI runtime, but that does not replace the dispatcher ownership rule.

## Dispatcher Operation Classes

Dispatcher must support at least these operation classes.

| Class | Examples | Ordering requirement |
|------|----------|----------------------|
| Session control | connect, disconnect, reset | strict serial |
| Configuration | load config, optional readback | strict serial |
| Invocation control | invoke, wait done | strict serial per kernel/epoch |
| Data transfer | send input, receive output | serial within invocation |
| Observability | start/stop trace, read perf | serial or snapshot-safe |

## Epoch Model

Each successful `loadConfig` opens a new runtime epoch. Invocation requests are
bound to exactly one epoch.

Rules:

- invocation in `Ready` state without config is invalid
- reconfiguration during active invocation is invalid in baseline mode
- trace/perf reports must include epoch identifier

## Deterministic Mode

When `deterministic=true`, runtime must guarantee:

- stable queue submission order for same program order
- stable invocation ordering for same input order
- stable tie-breaking policy for completion processing
- stable compare and report ordering

If deterministic mode is disabled, runtime may allow throughput-oriented
reordering but must log scheduling policy and seed.

## Configuration Upload Algorithm

Required algorithm:

1. verify session in `Ready`
2. check word count against backend-reported address space bounds
3. write all words in ascending address order
4. optionally read back selected words
5. mark session `Configured`

Address mapping authority remains in
[spec-fabric-config_mem.md](./spec-fabric-config_mem.md).

## Invocation Algorithm

Required high-level algorithm per request:

1. validate `Configured` state and kernel/interface availability
2. enqueue input payload on required channels
3. trigger execution start (implicit or explicit)
4. wait for completion condition (`done`/result contract)
5. drain output payload and package `InvocationResult`
6. attach cycle statistics when available

## CPU Reference Comparison Contract

For end-to-end verdict, runtime must support a CPU reference path.

Required behavior:

- consume same logical input vectors as accelerator invocation
- compute expected outputs using CPU implementation
- compare expected vs accelerator outputs with deterministic report format
- mark mismatch as validation failure

Comparison policy (exact/epsilon/tolerance) is selected by test specification,
not by runtime defaults.

## Host Memory Service Usage

If `HostMem` service is available, runtime may use it for high-bandwidth
payload exchange.

Minimum requirements:

- explicit map/unmap lifetime handling
- buffer flush semantics honored when backend requires it
- fallback path to channel-based payload transfer when `HostMem` unavailable

## Error Handling Contract

Runtime must surface errors with category and context:

- category (transport/discovery/type/session/device)
- lifecycle state
- epoch id
- invocation id if applicable
- backend identifier

Fatal errors must force session to terminal state and reject new invocations
until reconnect.

## Minimal Implementation Strategy

A minimal practical implementation can be built in these layers:

1. `SessionCore`: lifecycle state machine and dispatcher queue
2. `BackendAdapter`: wraps ESI connection and typed channel access
3. `ConfigLoader`: MMIO-based `config_mem` uploader
4. `Invoker`: kernel data/control exchange
5. `OracleComparator`: CPU reference compare and report
6. `TraceCollector`: optional telemetry sink

This decomposition is sufficient for undergraduate implementation with clear
module boundaries.

## Related Documents

- [spec-cosim.md](./spec-cosim.md)
- [spec-cosim-architecture.md](./spec-cosim-architecture.md)
- [spec-cosim-protocol.md](./spec-cosim-protocol.md)
- [spec-cosim-validation.md](./spec-cosim-validation.md)
- [spec-fabric-config_mem.md](./spec-fabric-config_mem.md)
