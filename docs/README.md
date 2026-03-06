# Loom Documentation Index

Loom is a full-stack framework for domain-specific accelerators, from C++ source
code to hardware backend. This directory contains the normative specification
documents for each major subsystem.

## Pipeline Overview

| Spec | Description |
|------|-------------|
| [spec-loom.md](./spec-loom.md) | End-to-end pipeline overview (C++ to hardware) |
| [spec-cli.md](./spec-cli.md) | `loom` CLI interface, invocation modes, option reference |
| [spec-pragma.md](./spec-pragma.md) | C++ pragma system for compiler hints and constraints |

## Software IR (Dataflow)

| Spec | Description |
|------|-------------|
| [spec-dataflow.md](./spec-dataflow.md) | Dataflow dialect: carry, invariant, stream, gate, tagged types |
| [spec-dataflow-error.md](./spec-dataflow-error.md) | Compile-time error codes for the software IR pipeline |

## Hardware Architecture (ADG)

| Spec | Description |
|------|-------------|
| [spec-adg.md](./spec-adg.md) | Architecture Description Graph: C++ API for CGRA hardware descriptions |
| [spec-adg-api.md](./spec-adg-api.md) | ADGBuilder C++ API reference (`<loom/adg.h>`) |
| [spec-adg-sv.md](./spec-adg-sv.md) | SystemVerilog RTL output generation |
| [spec-adg-sysc.md](./spec-adg-sysc.md) | SystemC simulation model generation |
| [spec-adg-tools.md](./spec-adg-tools.md) | Simulation tools: VCS, Verilator, waveform viewers |

## Hardware Fabric Dialect

| Spec | Description |
|------|-------------|
| [spec-fabric.md](./spec-fabric.md) | Fabric dialect overview: module structure, type system, operations |
| [spec-fabric-pe.md](./spec-fabric-pe.md) | Processing element (PE) specification |
| [spec-fabric-pe-ops.md](./spec-fabric-pe-ops.md) | PE operation semantics |
| [spec-fabric-switch.md](./spec-fabric-switch.md) | Configurable routing switch |
| [spec-fabric-temporal_pe.md](./spec-fabric-temporal_pe.md) | Time-multiplexed PE with instruction slots |
| [spec-fabric-temporal_sw.md](./spec-fabric-temporal_sw.md) | Tag-aware routing switch |
| [spec-fabric-fifo.md](./spec-fabric-fifo.md) | Pipeline buffer (FIFO) |
| [spec-fabric-tag.md](./spec-fabric-tag.md) | Tag boundary operations: add_tag, map_tag, del_tag |
| [spec-fabric-mem.md](./spec-fabric-mem.md) | Memory operations: on-chip and external memory |
| [spec-fabric-config_mem.md](./spec-fabric-config_mem.md) | Configuration memory structure and semantics |
| [spec-fabric-error.md](./spec-fabric-error.md) | Compile-time, configuration, and runtime error codes |

## Mapper (Place and Route)

| Spec | Description |
|------|-------------|
| [spec-mapper.md](./spec-mapper.md) | Mapper scope, inputs/outputs, DFG preconditions |
| [spec-mapper-model.md](./spec-mapper-model.md) | Graph data model, tech-mapping, constraints C1-C6 |
| [spec-mapper-algorithm.md](./spec-mapper-algorithm.md) | PnR pipeline, repair strategies, CP-SAT integration |
| [spec-mapper-cost.md](./spec-mapper-cost.md) | Cost metrics and weight profiles |
| [spec-mapper-output.md](./spec-mapper-output.md) | Output formats: mapping.json, config.bin, addr.h |

## Visualization

| Spec | Description |
|------|-------------|
| [spec-viz.md](./spec-viz.md) | Visualization overview: DFG, ADG, Mapped views |
| [spec-viz-dfg.md](./spec-viz-dfg.md) | DFG (software dataflow) DOT conventions |
| [spec-viz-adg.md](./spec-viz-adg.md) | ADG (hardware architecture) DOT conventions |
| [spec-viz-mapped.md](./spec-viz-mapped.md) | Mapped visualization: overlay/side-by-side, route coloring |
| [spec-viz-gui.md](./spec-viz-gui.md) | Browser-based HTML viewer specification |

## Co-Simulation

| Spec | Description |
|------|-------------|
| [spec-cosim.md](./spec-cosim.md) | Co-simulation overview and authority map |
| [spec-cosim-architecture.md](./spec-cosim-architecture.md) | Simulation thread, dispatcher, sessions, epochs |
| [spec-cosim-protocol.md](./spec-cosim-protocol.md) | Three-layer protocol stack (P0/P1/P2) |
| [spec-cosim-backend-rtl.md](./spec-cosim-backend-rtl.md) | RTL backend simulation behavior |
| [spec-cosim-backend-systemc.md](./spec-cosim-backend-systemc.md) | SystemC backend simulation behavior |
| [spec-cosim-runtime.md](./spec-cosim-runtime.md) | Host API and runtime behavior |
| [spec-cosim-trace.md](./spec-cosim-trace.md) | Trace data collection and format |
| [spec-cosim-validation.md](./spec-cosim-validation.md) | Validation and verification requirements |

## Test Targets

| Target | Command | Description |
|--------|---------|-------------|
| `check-loom` | `ninja -C build check-loom` | Master test suite (all tests) |
| `check-loom-app` | `ninja -C build check-loom-app` | Application compilation roundtrips |
| `check-loom-fabric-tdd` | `ninja -C build check-loom-fabric-tdd` | Fabric dialect TDD tests (lit) |
| `check-loom-mapper` | `ninja -C build check-loom-mapper` | Mapper unit tests |
| `check-loom-adg` | `ninja -C build check-loom-adg` | ADG construction tests |
| `check-loom-sv` | `ninja -C build check-loom-sv` | SystemVerilog backend tests |
| `check-loom-handshake` | `ninja -C build check-loom-handshake` | Handshake dialect validation |
| `check-loom-spec` | `ninja -C build check-loom-spec` | Spec consistency checks |

## Build

```bash
ninja -C build clean-loom   # Clean loom build artifacts
ninja -C build loom          # Build the loom binary
ninja -C build check-loom    # Run all tests
```
