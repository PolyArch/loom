# FCC ADG Builder Specification

## Overview

The ADG Builder is FCC's programmatic user interface for constructing
architecture descriptions before mapping.

Its intended role is two-layered:

- a high-level, topology-oriented API for quickly sketching regular ADGs
- a low-level, typed API for users who want precise control over emitted
  `fabric.mlir` without hand-writing the entire file

The builder accumulates an MLIR-independent internal model first and only
materializes MLIR during export.

## Design Goals

The builder should let users:

- define reusable `fabric.function_unit`
- group them into `fabric.spatial_pe` or `fabric.temporal_pe`
- define `fabric.spatial_sw`, `fabric.temporal_sw`, `fabric.memory`,
  `fabric.extmemory`, and FIFO resources
- instantiate those components repeatedly
- wire them using simple graph operations
- use topology helpers for regular fabrics
- specialize selected locations after a regular topology has been created
- attach explicit visualization metadata when topology geometry is already
  known

The builder must also preserve an advanced escape hatch for users who need
exact control over a compound `function_unit` body.

## API Layers

### High-Level Convenience Layer

The convenience layer is intended for the common case:

- `FunctionUnitSpec`
- `SpatialPESpec`
- `SpatialSWSpec`
- `ChessMeshOptions`
- `LatticeMeshOptions`
- `CubeOptions`
- `defineSingleFUSpatialPE`
- `defineSingleFUTemporalPE`
- `defineConstantFU`
- `defineUnaryFU`
- `defineBinaryFU`
- `instantiatePEArray`
- `instantiatePEGrid`
- `instantiateSWArray`
- `instantiateSWGrid`
- `instantiateMemoryArray`
- `instantiateExtMemArray`
- `addScalarInputs`
- `addScalarOutputs`
- `addMemrefInputs`
- `addInputs`
- `addOutputs`
- `connectInputVectorToInstance`
- `connectInstanceToOutputVector`
- `SwitchBankDomainSpec`
- `SwitchBankDomainResult`
- `buildSwitchBankDomain`
- `connectPEBankToSwitch`
- `associateExtMemBankWithSW`
- `associateMemoryBankWithSW`
- `defineCmpiFU`
- `defineCmpfFU`
- `defineStreamFU`
- `defineIndexCastFU`
- `defineSelectFU`
- `defineGateFU`
- `defineCarryFU`
- `defineInvariantFU`
- `defineCondBrFU`
- `defineMuxFU`
- `defineJoinFU`
- `defineLoadFU`
- `defineStoreFU`
- `defineFullCrossbarSpatialSW`
- `buildMesh`
- `buildLatticeMesh`
- `buildTorusMesh`
- `buildRing`
- `buildChessMesh`
- `buildCube`

This layer favors:

- compact declarations
- automatic full-crossbar connectivity when appropriate
- automatic handling of edge and corner switch degrees in helper topologies
- explicit visualization sidecar emission for known layouts
- one-line construction of single-function-unit spatial PEs
- one-line construction of single-function-unit temporal PEs
- bulk instantiation of repeated PE templates without hand-written loops
- bulk creation of repeated boundary ports and repeated memory-family instances
- cursor-based attachment of PE and memory banks to a shared switch

### Typed Low-Level Layer

The typed layer is intended for advanced users:

- `defineSpatialPE(name, inputTypes, outputTypes, fus)`
- `defineTemporalPE(...)`
- `defineSpatialSW(name, inputTypes, outputTypes, connectivity, ...)`
- `defineTemporalSW(...)`
- `createAddTag(...)`
- `createAddTagBank(...)`
- `createMapTag(...)`
- `createDelTag(...)`
- `createDelTagBank(...)`
- `defineMemory(...)`
- `defineExtMemory(...)`
- `addInput(name, typeStr)`
- `addOutput(name, typeStr)`
- `addInputs(prefix, typeStrs)`
- `addOutputs(prefix, typeStrs)`
- `connect(PortRef, PortRef)`
- `connectInputToPort(...)`
- `connectPortToOutput(...)`

This layer gives direct control over:

- per-port types
- spatial versus temporal component kind
- temporal PE hardware parameters
- temporal switch route-table capacity
- memory-family hardware parameters

The convenience `function_unit` helpers are intended to cover the common
configurable-op cases without forcing raw MLIR bodies:

- `defineConstantFU(...)`
- `defineUnaryFU(...)`
- `defineBinaryFU(...)`
- `defineCmpiFU(...)`
- `defineCmpfFU(...)`
- `defineStreamFU(...)`
- `defineIndexCastFU(...)`
- `defineSelectFU(...)`
- `defineGateFU(...)`
- `defineCarryFU(...)`
- `defineInvariantFU(...)`
- `defineCondBrFU(...)`
- `defineMuxFU(...)`
- `defineJoinFU(...)`
- `defineLoadFU(...)`
- `defineStoreFU(...)`

These helpers still emit ordinary `fabric.function_unit` definitions and feed
the same mapper/config generation pipeline as hand-authored `fabric.mlir`.

`defineUnaryFU(...)` and `defineBinaryFU(...)` are the preferred helpers for
the common "one software op per function_unit" case. More specialized helpers
such as `defineConstantFU(...)`, `defineCmpiFU(...)`, or `defineStreamFU(...)`
exist for ops whose configurable payload cannot be expressed only by input and
result types.

`defineJoinFU(name, inputCount, ...)` defines the maximum hardware join fan-in
supported by that `fabric.function_unit`. The current Builder accepts
`1..64` inputs for one hardware join. Mapper tech-mapping may later bind a
smaller software `handshake.join` onto that FU by emitting a `join_mask`
runtime-config field, but it may not exceed the hardware fan-in declared by
the Builder-generated FU body.

`defineSingleFUSpatialPE(...)` is the preferred helper when a user wants one
`fabric.spatial_pe` that wraps exactly one `fabric.function_unit`. This is the
common pattern for quick sketches, chess-like meshes, and small builder-based
unit tests.

`defineSingleFUTemporalPE(...)` plays the same role for `fabric.temporal_pe`
when a design needs a small tagged temporal container without manually spelling
out a full `TemporalPESpec`.

`instantiatePEArray(...)` and `instantiatePEGrid(...)` are intended for the
"replicate this PE many times" case. They do not impose connectivity, but they
remove repeated boilerplate instance naming and make it easier to build custom
topologies on top of a regular replicated substrate.

`instantiateSWArray(...)` and `instantiateSWGrid(...)` play the same role for
`fabric.spatial_sw` and `fabric.temporal_sw` instances. They are intended for
custom topologies that are more structured than ad-hoc instantiation but do not
fit one of the dedicated topology helpers.

`instantiateMemoryArray(...)` and `instantiateExtMemArray(...)` serve the same
role for memory-family components.

`addScalarInputs(...)`, `addScalarOutputs(...)`, and `addMemrefInputs(...)`
exist for the common "declare N homogeneous boundary ports" case. They return
the created boundary indices in builder-defined order so users can still wire
them individually.

`connectInputVectorToInstance(...)` and `connectInstanceToOutputVector(...)`
cover the common case where a user wants to bind a consecutive run of builder
boundary ports to a consecutive run of instance ports.

For central-switch topologies, the builder exposes:

- `SwitchPortCursor`
- `connectPEBankToSwitch(...)`
- `associateExtMemBankWithSW(...)`
- `associateMemoryBankWithSW(...)`

These helpers are intended for star-style fabrics where the user wants one
switch to host a bank of PEs, then append one or more memory banks, then append
scalar boundary ports, all without hand-maintaining cursor arithmetic.

For the even more common "one central switch plus PE bank plus memory bank"
pattern, the builder also exposes:

- `SwitchBankDomainSpec`
- `SwitchBankDomainResult`
- `buildSwitchBankDomain(...)`

This helper instantiates the switch, PE bank, optional memory-family banks,
optional boundary memref inputs for `fabric.extmemory`, and typed scalar
boundaries in one step. It is intended to remove boilerplate from small domain
ADGs and builder-based examples while preserving full access to the returned
instance handles and remaining switch cursor for later customization.

`buildSwitchBankDomain(...)` auto-binds module memref inputs only for
`fabric.extmemory`. For non-private `fabric.memory`, it can optionally append
host-visible memref outputs after the ordinary scalar outputs. This preserves
the common expectation that software-visible scalar outputs appear first in the
module result list, while the memory-mapped slave interfaces appear afterward.

When wiring `fabric.memory` to a shared switch, only the runtime load/store
ports participate in switch routing. A non-private memory's host-visible memref
output is not part of the switch-facing port budget and must remain a module
boundary connection.

### Advanced Escape Hatch

For compound `function_unit` design, the builder provides:

- `defineFUWithBody(...)`

This accepts the `fabric.function_unit` body as raw MLIR text and preserves
the rest of the ADG in the builder's structured model.

This is the intended way to expose precise `function_unit` internals without
forcing users to hand-write the entire `fabric.module`.

## Supported Component Families

The builder currently supports structured construction of:

- `fabric.function_unit`
- `fabric.spatial_pe`
- `fabric.temporal_pe`
- `fabric.spatial_sw`
- `fabric.temporal_sw`
- `fabric.add_tag`
- `fabric.map_tag`
- `fabric.del_tag`
- `fabric.memory`
- `fabric.extmemory`
- `fabric.fifo`

At export time, the builder may emit:

- component definitions
- named instances
- module boundary ports
- inter-instance connections
- layout sidecar metadata

## Boundary API

The builder supports two boundary styles:

- convenience bit-vector boundaries through `addScalarInput` and
  `addScalarOutput`
- explicit typed boundaries through `addInput` and `addOutput`

Typed boundaries are the preferred advanced interface when the ADG uses
non-default port types, including tagged ports.

For repeated typed boundaries, the builder also provides:

- `addInputs(prefix, typeStrs)`
- `addOutputs(prefix, typeStrs)`

These helpers create a consecutive run of typed boundary ports while preserving
the same explicit type control as repeated `addInput(...)` and `addOutput(...)`
calls.

The builder also exposes a lightweight boundary attachment handle:

- `PortRef { InstanceHandle instance; unsigned port; }`

`PortRef` is intended for user-facing graph construction. It avoids forcing
callers to carry a separate instance handle and integer port index through
topology code.

The corresponding convenience wiring helpers are:

- `connect(PortRef src, PortRef dst)`
- `connectInputToPort(unsigned inputIdx, PortRef dst)`
- `connectPortToOutput(PortRef src, unsigned outputIdx)`

These are functionally equivalent to the lower-level instance/port overloads,
but are better suited for topology helpers that already know which concrete
attachment point a user should target.

For repeated inline tag stages, the builder exposes:

- `createAddTagBank(...)`
- `createDelTagBank(...)`

These helpers are intended for temporal-style designs that often need several
parallel `add_tag` or `del_tag` nodes with one uniform type pair.

## Topology Helpers

The topology helpers currently target spatial fabrics:

- `buildMesh`
- `buildLatticeMesh`
- `buildTorusMesh`
- `buildRing`
- `buildChessMesh`
- `buildCube`

These helpers:

- instantiate the requested grid or ring
- wire regular neighbor connections
- handle boundary-specific switch degrees automatically
- attach explicit visualization coordinates
- expose user-facing boundary attachment points for designated ingress and
  egress locations

`buildLatticeMesh`, `buildChessMesh`, and `buildCube` synthesize
boundary-aware `spatial_sw` templates so that corner, edge, face, and interior
positions do not require the user to hand-count switch degrees.

`LatticeMeshOptions`, `ChessMeshOptions`, and `CubeOptions` allow
module-boundary abuse cases such as attaching extra inputs at one designated
boundary corner or extra outputs at the opposite boundary corner without
changing the reusable PE template.

When a topology helper synthesizes designated boundary ingress or egress
attachment points, the returned `MeshResult` or `CubeResult` publishes them
through:

- `ingressPorts`
- `egressPorts`

These vectors contain `PortRef` entries in builder-defined order. Users can
feed them directly into `connectInputToPort(...)` and `connectPortToOutput(...)`
instead of manually reconstructing switch instance names or counting corner
port offsets.

`buildMesh`, `buildTorusMesh`, and `buildRing` accept either:

- one PE template reused everywhere
- a selector callback that chooses the PE template per location

`buildChessMesh` and `buildCube` also support per-cell specialization through
selector callbacks.

This is the intended mechanism for patterns such as:

- "mostly one PE kind, but one cell is specialized"
- "different rows use different PE variants"
- "a regular topology with a few manually chosen hot spots"

The topology helpers require spatial component templates and will reject
temporal ones.

`buildMesh` is the legacy torus-style helper that reuses one switch template
everywhere. Users who want non-wrapping edge behavior should prefer
`buildLatticeMesh`.

`buildCube` projects depth layers into the sidecar layout using staggered
depth offsets. The projection is builder-defined and deterministic; the
renderer consumes the emitted coordinates instead of inferring a 3-D layout.

## Visualization Sidecar Contract

When a topology helper already knows the intended geometry, the builder should
emit a sidecar JSON file and bind it from `fabric.module` using
`attributes {viz_file = "...json"}`.

The sidecar contains:

- component identities
- component centers
- optional grid coordinates
- precomputed module-level routes

This allows the renderer to reuse builder-provided geometry instead of
recomputing layout and routing on the fly.

For Builder-produced ADGs that do not come from a topology helper with explicit
geometry, export should still precompute a non-overlapping sidecar placement
offline. A graph-layout pass comparable to Graphviz `neato` is the intended
fallback, and its result should be serialized before HTML generation.

## Validation Contract

Builder-emitted ADGs and hand-authored ADGs consumed by the mapper are subject
to the same `verifyFabricModule` contract.

Before export, the builder also performs its own structural checks, including:

- no dangling instance inputs
- no dangling instance outputs
- no implicit self-loop repair

If an ADG is incomplete, the builder must fail instead of synthesizing hidden
connections.

## Relationship to Other Specs

- [spec-fabric.md](./spec-fabric.md)
- [spec-fabric-function_unit.md](./spec-fabric-function_unit.md)
- [spec-fabric-spatial_pe.md](./spec-fabric-spatial_pe.md)
- [spec-fabric-temporal_pe.md](./spec-fabric-temporal_pe.md)
- [spec-fabric-spatial_sw.md](./spec-fabric-spatial_sw.md)
- [spec-fabric-temporal_sw.md](./spec-fabric-temporal_sw.md)
- [spec-fabric-memory-interface.md](./spec-fabric-memory-interface.md)
- [spec-mapper-model.md](./spec-mapper-model.md)
- [spec-viz.md](./spec-viz.md)
