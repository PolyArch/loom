# Loom Compiler Full-Pipeline Specification

## Overview

Loom is a compiler and hardware generation framework that connects three
different representations:

- C/C++ source code (software intent)
- Handshake/Dataflow MLIR (software execution graph)
- Fabric MLIR (hardware resource graph)

The complete flow is:

1. Compile annotated C/C++ into LLVM IR and MLIR.
2. Lower control/data semantics into Handshake/Dataflow MLIR.
3. Build a target hardware graph as Fabric MLIR using ADG.
4. Run mapper place-and-route from software graph to hardware graph.
5. Emit runtime configuration (`config_mem`) and backend artifacts
   (SystemC/SystemVerilog).

This document is the top-level picture. Detailed definitions are delegated to
specialized specs:

- CLI and frontend lowering: [spec-cli.md](./spec-cli.md)
- Pragma semantics: [spec-pragma.md](./spec-pragma.md)
- Dataflow dialect and tagged types: [spec-dataflow.md](./spec-dataflow.md)
- Fabric dialect and hardware operations: [spec-fabric.md](./spec-fabric.md)
- ADG hardware construction: [spec-adg.md](./spec-adg.md)
- Mapper model and algorithms: [spec-mapper.md](./spec-mapper.md)

## Pipeline Stages

### Stage A: Frontend Compilation

Input:
- C/C++ source files, optionally using Loom pragmas.

Output:
- LLVM IR (`*.llvm.ll`)
- LLVM dialect MLIR (`*.llvm.mlir`)
- SCF MLIR (`*.scf.mlir`)
- Handshake MLIR (`*.handshake.mlir`)

This stage is specified in [spec-cli.md](./spec-cli.md). Pragma semantics that
influence lowering are specified in [spec-pragma.md](./spec-pragma.md).

### Stage B: Hardware Architecture Definition

Input:
- ADGBuilder C++ program (compiled with `loom --as-clang`).

Output:
- Fabric MLIR hardware graph
- DOT visualization
- SystemC template output
- SystemVerilog template output

ADG describes hardware structure only. Runtime configuration is left empty by
design. See [spec-adg.md](./spec-adg.md).

### Stage C: Mapper Place-and-Route (Bridge Stage)

Input:
- Software graph: Handshake/Dataflow MLIR from Stage A
- Hardware graph: Fabric MLIR from Stage B

Output:
- Placement: software operations mapped to hardware compute resources
- Routing: software edges mapped to hardware paths
- Temporal assignments: instruction slot/tag/opcode/register decisions
- Runtime configuration image for all configurable Fabric operations

This stage is the bridge between software and hardware graphs. It is specified
in [spec-mapper.md](./spec-mapper.md) and its companion docs.

### Stage D: Backend Realization

Input:
- Fabric MLIR hardware graph
- Mapper-produced runtime configuration

Output:
- Executable/simulatable accelerator model
- Programmed `config_mem` image for runtime behavior

Backend structure and generated artifacts are specified in:

- [spec-adg-sysc.md](./spec-adg-sysc.md)
- [spec-adg-sv.md](./spec-adg-sv.md)
- [spec-fabric-config_mem.md](./spec-fabric-config_mem.md)

## Graph Boundary Definitions

To avoid ambiguity, Loom uses these terms consistently:

- **Software graph**: operations and edges in Handshake/Dataflow MLIR that
  represent program behavior.
- **Hardware graph**: operations and edges in Fabric MLIR that represent
  available hardware resources.
- **Mapping**: a relation that binds software graph nodes/edges to hardware
  graph nodes/paths while satisfying hardware constraints.

Authoritative mapper data structures and constraints are defined only in
[spec-mapper-model.md](./spec-mapper-model.md).

## Runtime Configuration Ownership

Runtime configuration is not authored in Stage A or Stage B:

- Stage A builds software semantics.
- Stage B builds hardware capacity.
- Stage C (mapper) produces workload-specific configuration values.

Configuration layout itself is defined by Fabric operation specs and the unified
config memory specification:
[spec-fabric-config_mem.md](./spec-fabric-config_mem.md).

## Error Responsibility by Stage

- Frontend/lowering errors: diagnosed in Stage A.
- ADG structural errors: diagnosed in Stage B.
- Mapping feasibility and assignment errors: diagnosed in Stage C.
- Runtime configuration and execution errors: reported by Fabric hardware using
  symbols from [spec-fabric-error.md](./spec-fabric-error.md).

The mapper must preserve the Fabric error model and emit constraints that are
compatible with these symbols.

## Related Documents

- [spec-cli.md](./spec-cli.md)
- [spec-pragma.md](./spec-pragma.md)
- [spec-dataflow.md](./spec-dataflow.md)
- [spec-fabric.md](./spec-fabric.md)
- [spec-adg.md](./spec-adg.md)
- [spec-mapper.md](./spec-mapper.md)
