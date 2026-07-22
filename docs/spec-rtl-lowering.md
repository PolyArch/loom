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

Behavioral memory models and black boxes are legal only when Fabric or its
implementation binding explicitly declares that realization. The
`HardwareImplementation` records their contracts and unresolved external
dependencies.

## Configuration ABI

Fabric-to-RTL implements an exact `ConfigurationABI` for every exposed
Programming Unit. The ABI, not RTL source order or backend-local structs, owns
bit positions, codebooks, padding, programming visibility, and image loading.

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
* temporal context and memory-operation behavior;
* clock/reset domain and self-reset closure;
* ConfigurationABI programming through one mapped workload; and
* rejection of an unsupported or behavior-changing hidden refinement.

Tests do not preserve whole RTL text, vendor command lines, hierarchy names, or
per-primitive fixture matrices. Syntax, elaboration, formal, simulation, and
physical observations are ordinary Evaluations of the finalized implementation.
