# Loom Co-Simulation Specification

## Overview

This document defines the final execution stage of Loom: co-simulation
(`cosim`). In this stage, host software executes together with a simulated
accelerator backend (SystemC or RTL), uses ESI channels to configure and drive
that backend, and verifies accelerator outputs against CPU reference outputs.

This document is the top-level `cosim` contract. Detailed behavior is delegated
to `spec-cosim-*.md` documents listed below.

## Scope

`cosim` is responsible for runtime integration of artifacts produced by earlier
stages:

- software graph behavior from Stage A (Handshake/Dataflow)
- hardware graph behavior from Stage B (Fabric + ADG backend generation)
- mapping and configuration from Stage C (`MappingState` and `config_mem` image)
- executable backend model from Stage D (SystemC executable or RTL simulator)

`cosim` does not define new Fabric operation semantics, mapper legality rules,
or ADG construction APIs.

## Stage Position in the Loom Pipeline

`cosim` is Stage E of the end-to-end flow:

1. Stage A: frontend lowering (`spec-cli.md`, `spec-dataflow.md`)
2. Stage B: ADG hardware graph construction (`spec-adg.md`)
3. Stage C: mapper place-and-route (`spec-mapper.md`)
4. Stage D: backend realization (`spec-adg-sysc.md`, `spec-adg-sv.md`)
5. Stage E: co-simulation and end-to-end verification (this spec family)

## Inputs and Outputs

### Inputs

- Handshake/Dataflow accelerator interface contract (from Stage A)
- Fabric hardware topology and backend model (from Stages B and D)
- Mapper result (`MappingState`) and concrete `config_mem` words (from Stage C)
- Host test vectors and CPU reference implementation

### Outputs

- Pass/fail verdict for each test case
- Accelerator output payloads captured from ESI channels
- Optional execution traces and performance statistics
- Optional replay bundle for deterministic reproduction of one run

## Normative Runtime Requirements

A conforming Loom `cosim` implementation must provide all of the following:

- One simulation thread owning device communication and simulation progression
- Host-side multi-thread request capability
- ESI-based configuration transfer for mapper-produced `config_mem`
- ESI-based input/output transfer for accelerator execution
- End-of-run host comparison against CPU reference outputs
- Error reporting with stage-aware classification

Detailed thread, protocol, and backend contracts are specified in the
sub-documents below.

## Single-Source Boundaries

The following documents are authoritative for each `cosim` topic. Definitions
must not be duplicated outside the listed authority.

| Topic | Authoritative document |
|------|-------------------------|
| Architecture and lifecycle model | [spec-cosim-architecture.md](./spec-cosim-architecture.md) |
| Transport and channel/session protocol | [spec-cosim-protocol.md](./spec-cosim-protocol.md) |
| Host runtime API and threading contract | [spec-cosim-runtime.md](./spec-cosim-runtime.md) |
| SystemC backend binding | [spec-cosim-backend-systemc.md](./spec-cosim-backend-systemc.md) |
| RTL backend binding | [spec-cosim-backend-rtl.md](./spec-cosim-backend-rtl.md) |
| Trace and performance data model | [spec-cosim-trace.md](./spec-cosim-trace.md) |
| End-to-end validation and acceptance | [spec-cosim-validation.md](./spec-cosim-validation.md) |

## Relationship to Existing Specs

- `config_mem` layout and bit packing remain authoritative in
  [spec-fabric-config_mem.md](./spec-fabric-config_mem.md).
- Mapper validity and configuration derivation remain authoritative in
  [spec-mapper-model.md](./spec-mapper-model.md).
- SystemC and SystemVerilog generation details remain authoritative in
  [spec-adg-sysc.md](./spec-adg-sysc.md) and
  [spec-adg-sv.md](./spec-adg-sv.md).
- Handshake/Dataflow operation semantics remain authoritative in
  [spec-dataflow.md](./spec-dataflow.md).
- Simulation tool and waveform priority remain authoritative in
  [spec-adg-tools.md](./spec-adg-tools.md), including default preference for
  Synopsys VCS/Verdi and secondary open-source Verilator/GTKWave flow.

`cosim` consumes these contracts. It does not redefine them.

## Non-Goals

The `cosim` spec family does not standardize:

- mapper search heuristics
- Fabric lowering internals
- a mandatory GUI or waveform viewer
- deployment-specific FPGA driver behavior

## Related Documents

- [spec-loom.md](./spec-loom.md)
- [spec-cli.md](./spec-cli.md)
- [spec-dataflow.md](./spec-dataflow.md)
- [spec-fabric.md](./spec-fabric.md)
- [spec-fabric-config_mem.md](./spec-fabric-config_mem.md)
- [spec-adg.md](./spec-adg.md)
- [spec-adg-sysc.md](./spec-adg-sysc.md)
- [spec-adg-sv.md](./spec-adg-sv.md)
- [spec-adg-tools.md](./spec-adg-tools.md)
- [spec-mapper.md](./spec-mapper.md)
- [spec-mapper-model.md](./spec-mapper-model.md)
- [spec-cosim-architecture.md](./spec-cosim-architecture.md)
- [spec-cosim-protocol.md](./spec-cosim-protocol.md)
- [spec-cosim-runtime.md](./spec-cosim-runtime.md)
- [spec-cosim-backend-systemc.md](./spec-cosim-backend-systemc.md)
- [spec-cosim-backend-rtl.md](./spec-cosim-backend-rtl.md)
- [spec-cosim-trace.md](./spec-cosim-trace.md)
- [spec-cosim-validation.md](./spec-cosim-validation.md)
