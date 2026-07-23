# Fabric To RTL

This document defines the hardware-generation boundary from one exact Fabric
Hardware Description to an immutable RTL `HardwareImplementation`.

## Input And Output

Fabric-to-RTL consumes:

* one fully elaborated Fabric Hardware Description;
* one exact `ConfigurationABI` for that Fabric;
* exact Fabric-owned Interconnect and implementation refinements;
* exact external implementation/library bindings required by selected Fabric
  resources.

It produces one `HardwareImplementation` containing or content-addressing the
SystemVerilog sources, packages, interfaces, constraints, black-box contracts,
activity-name map, and implementation manifest needed by downstream tools.

The lowering does not consume Dataflow or Mapping and does not create a
workload-specific RTL design. Workload execution combines the reusable
`HardwareImplementation` with an exact `Deployment`, configuration images, and
runtime inputs.

## Semantic Ownership

Fabric is the hardware semantic SSOT. RTL lowering implements, but never
extends, these Fabric facts:

* occurrences, ports, directed connectivity, and module/system boundaries;
* compute, switch, memory, FIFO, boundary, and transport capabilities;
* spatial/temporal organization, ResourceState, UsePattern, and GrantPolicy;
* latency, initiation interval, capacity, buffering, ordering, backpressure,
  reset, and progress behavior;
* clock, reset, power, address, memory, coherence, and protection domains;
* Transport Architecture and exact Interconnect Implementation refinement; and
* Mapping-visible semantic and physical configuration domains.

Selector logic, comparator structure, gate decomposition, and naming may vary
when they preserve this contract. A pipeline stage, arbitration order, buffer
visibility, or other cycle-observable difference requires a different Fabric
capability or Mapping-selected Fabric refinement. The backend cannot hide such
a choice in RTL.

Visualization metadata is stripped before Fabric identity and has no effect on
hardware generation.

## Operation Provider Registry

Operation lowering is dispatched by the same closed
`ImplementationFamilyId` used by the normative Hardware Sharing Group
registry:

```text
ImplementationFamilyId -> RTL provider callback
```

Before emission, the backend mechanically constructs one
`ResolvedFabricOpCapabilityView` for each concrete `fabric.op`. A provider
consumes that exact view and the exact `ConfigurationABI`. It may own emitter
code, behavioral or external-IP implementation availability, and typed
external dependencies. It does not own family membership, operation types,
HSG legality, timing semantics, or configuration encoding.

Operation-name string classification, backend-local exact-mode enumeration,
and global `(operation name, variant)` selection are forbidden as semantic or
dispatch authorities. A behavioral or golden provider uses the same family ID
and exact Fabric contract rather than a second operation-name support table.
Missing provider support is typed `Unsupported`; a partially lowered or
semantically substituted implementation is never produced.

The provider must distinguish the exact resource contract from the selected
operation semantics. The initial scalar `CoreAluFu` and arithmetic `MacFu`
resources lower as the Fabric-declared one-stage registered elastic resources.
That implementation rule is not a default wrapper for every `fabric.op`.
Stateful operations such as `dataflow.stream`, `dataflow.carry`,
`dataflow.invariant`, and `dataflow.gate` require providers that implement
their registered actor transitions together with their exact Fabric-owned
state capacity, atomic use patterns, transition timing, result holding, and
backpressure contract. A generic shell may not consume an inactive operand,
publish an inactive result, advance logical state while blocked, or convert
an operation-specific state machine into a stateless pipeline.

## Implementation Recipes

Implementation choices are classified by their first observable difference:

* A choice that changes exact operation semantics, numeric accuracy, or the
  accepted actor domain is a different Fabric capability or `hw_params`
  contract.
* A choice that changes latency, initiation interval, state, buffering,
  capacity, or progress is a different Fabric contract. It is a Mapping
  physical refinement only when Fabric declares that exact runtime-selectable,
  semantic-preserving domain.
* A choice that preserves all Fabric-observable semantics, timing, capacity,
  progress, and `ConfigurationABI` may be a backend implementation recipe,
  such as two gate decompositions with different PPA.

The exact hardware `ResolvedCandidateGeneratorBinding` selects backend recipes
per occurrence:

```text
FabricEntityRef -> typed BackendRecipeKey
```

Recipe selection is not global by operation name. It is recorded in
`HardwareImplementation` derivation and identity while leaving Fabric identity
unchanged. Accuracy, timing, or other Fabric-visible differences cannot be
hidden behind a recipe key. A provider may report an unavailable recipe or
external dependency, but it may not silently choose another contract.

## Structural Lowering

Every Fabric connection lowers to explicit RTL connectivity. Replication,
fan-in, arbitration, temporal sharing, tag-domain transformation, and protocol
conversion appear only when represented by the corresponding Fabric primitive
or selected refinement. Same-kind endpoint width normalization is not a
resource or refinement; RTL derives it directly from the two endpoint types.

Fabric's port rule remains low-bit aligned: a wider source is truncated at the
high end and a narrower source is zero-extended at the high end. The rule
applies independently to payload and tag fields of same-kind `bits_tag`
connections. `bits` and `bits_tag` never convert into one another implicitly.
RTL emits the required slice or zero-fill wiring without an adapter node,
configuration field, or route hop. Hardware modules must preserve each
endpoint's declared tag width and temporal tag behavior.

Unrealizable or unsupported resources fail lowering. They are not replaced by
a similar primitive or silently emitted as behaviorally different logic.
`fabric.fifo` lowering preserves the capability and selected-mode contract in
`docs/spec-fabric-fifo.md`; implementation structure cannot change buffered
visibility, bypass backpressure, or inactive-state semantics.

An unbound or inactive operation input is not an implicit sink. A provider
must not assert readiness merely to drain and discard tokens unless the exact
Fabric capability explicitly defines that consumption and backpressure
behavior. FU-local selection remains the explicit `fabric.mux` and
`fabric.demux` topology owned by Fabric.

## Clocks, Reset, And Quiescence

RTL exposes the exact Fabric clock/reset domains and only their declared
crossings. Stateful resources start in their canonical initial state and, for a
legal completed invocation, satisfy Fabric's self-reset/quiescence contract
before the same physical slot is reused.

Power, clock-gating, reset synchronization, and backend constraints are emitted
only from explicit Fabric implementation facts. Missing implementation support
is a typed failure, not permission to omit required behavior.

## Memory And System Interfaces

`fabric.mem` lowers its operation engine, internal dependency forwarding,
configurable service dispatch, optional local storage, and manager/subordinate
interfaces without adding storage semantics absent from Fabric. System
interconnect lowers the selected exact implementation protocol while preserving
the architecture service, multicast, ordering, capacity, and progress contract.

The backend implements the Fabric-owned memory operation-port inventory,
capability alternatives, parameterized access domains, mask endpoints, and
declared use patterns. A complete element, contiguous, or indexed
address/data/mask token enters one operation endpoint. A selected use pattern
may decompose that firing across several service transactions or beats and
must implement inactive-lane suppression, masked-load zero fill, row-major
result assembly, and one logical retirement event. Endpoint payload width and
service beat width are independent facts; the backend cannot infer
decomposition from their ratio or reinterpret Physical Tags as vector lanes.

The leaf-channel shape is mechanical:

```text
!fabric.bits<W>       -> data[W] when W > 0, valid, ready
!fabric.bits_tag<W,T> -> data[W] when W > 0, tag[T], valid, ready
```

`!fabric.bits<0>` therefore emits only valid/ready, while
`!fabric.bits_tag<0,T>` emits tag plus valid/ready. RTL must not create a
zero-width data vector. Spatial memory uses the untagged form. Temporal memory
implements the configured per-role input `(endpoint, tag)` matches and output
`(endpoint, tag)` writes; it must not replace them with one common row tag or
use the operation kind as a runtime match key.

Manager and subordinate `memref` capabilities remain typed internal service
interfaces. Their AXI, TileLink, CXL, or custom physical pinout is selected by
the exact HardwareImplementation and is not inferred from the `fabric.mem`
operation-channel schedule.

Behavioral memory models and black boxes are legal only when Fabric or its
implementation binding explicitly declares that realization. The
`HardwareImplementation` records their contracts and unresolved external
dependencies.

## Configuration ABI

Fabric-to-RTL implements an exact `ConfigurationABI` for every exposed
Programming Unit. The ABI, not RTL source order or backend-local structs, owns
bit positions, codebooks, padding, programming visibility, and image loading.

Backend-local configuration signal names are implementation details. Every
configuration input and decoder relation must be mechanically derived from the
exact `ConfigurationABI`; an independently designed `cfg_*` interface is not a
public or semantic authority.

Mapped RTL execution must program the implementation through decoded
`HardwareConfigurationImage` artifacts from the exact `Deployment`. Reading a
Mapping directly in a testbench and bypassing the physical programming path is
invalid.

## Activity And Observability

The implementation provides a deterministic activity-name map from emitted
hierarchy/signals to canonical Fabric entity references. Mapping can then
derive actor correlation without making emitted names semantic identities.

Waveforms, toggle files, testbench logs, and vendor-native products are raw
detailed bundle material. An architecture-evaluation descriptor owns an
`implementation` subject slot. A mapped-RTL simulator descriptor owns
`implementation` and `deployment` subject slots. Requests bind exact
`HardwareImplementation` and, where applicable, exact `Deployment` artifacts;
the models produce Evidence and optional raw material, while mapped execution
also produces `SimulationExecution`. The RTL
simulator alone owns HDL event time; Loom numbers cycles only from explicit
clock-domain edges. Architecture-only lint, elaboration, reset, ABI, or formal
checks produce Evidence without an empty execution artifact.

## Determinism

Identical Fabric, implementation inputs, resolved semantic configuration, and
producer identity must yield byte-identical canonical HardwareImplementation
content. Emitted labels derive deterministically from canonical structural
references but remain presentation details.

## Anchor Verification

Stable anchors cover:

* one regular and one arbitrary-topology Fabric lowering;
* exact replication/arbitration and width/tag behavior;
* dispatch of one operation schema through two implementation families and
  typed rejection of a missing provider;
* temporal context and memory-operation behavior;
* vector element, contiguous, indexed, and masked memory operation lowering,
  including one declared narrower-beat realization and one logical retirement;
* Spatial and Temporal element-only, vector-only, and shared-hybrid memory
  ports, including distinct per-role Temporal tags and zero-payload control
  channels;
* clock/reset domain and self-reset closure;
* ConfigurationABI programming through one mapped workload; and
* rejection of an unsupported or behavior-changing hidden refinement,
  including implicit input draining.

Tests do not preserve whole RTL text, vendor command lines, hierarchy names, or
per-family exhaustive matrices. Syntax, elaboration, formal, simulation, and
physical observations are ordinary Evaluations of the finalized implementation.
