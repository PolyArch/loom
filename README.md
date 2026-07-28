# Loom

Loom is a full-stack compiler and architecture-exploration framework for
multi-core heterogeneous spatial accelerators. It accepts software through an
LLVM IR boundary, initially from C and C++ via `loom-cc` and `loom-c++`, and
hardware as Fabric MLIR produced by the typed C++ ADG Builder, a built-in
target, or an imported design.

Loom connects structured compiler optimization, canonical Dataflow programs,
Tech/Spatial/System Mapping, multi-fidelity evaluation and simulation,
configuration generation, and RTL/EDA implementation. Its target machine is a
system of heterogeneous AccCores, each combining an InstructionCore with an
arbitrary-topology CGRA SpatialCore.

## Documentation

Start with the [design documentation](docs/README.md). The
[full-stack architecture](docs/spec-loom-stack.md) defines the product and
links to the normative subsystem specifications. The
[architecture rationales](docs/rationales/README.md) explain the decisions
behind those contracts without duplicating their authority.
