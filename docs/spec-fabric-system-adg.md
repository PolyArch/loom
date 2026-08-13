# Fabric System ADG

## Purpose

`fabric.system` is the fully elaborated architecture-level system description
for a heterogeneous Loom target. It owns typed physical occurrences, explicit
directed connectivity, architecture-level services and guarantees, the exact
Transport Architecture, and the attachments that connect SpatialCore
templates to concrete AccCore occurrences. Protocol-specific Interconnect
Implementations are independent sibling objects in the Fabric Hardware
Description family; they are not members of the `fabric.system` semantic
identity.

It does not own software execution semantics, selected workload Mapping,
runtime remapping, simulator observations, or DSE choices.

`docs/spec-fabric-identity.md` is the sole owner of the persistent Fabric
entity and structural-reference catalog. This document owns the
system entities, endpoint inventories, service regions, transport patterns,
resource states, use patterns, and other hardware semantics addressed by
those references.

## Hardware Objects

The Fabric Hardware Description family distinguishes:

* HostCore and AccCore occurrences;
* node-local immutable InstructionCore descriptions;
* SpatialCore occurrences referencing exact `fabric.module` templates;
* typed module-to-occurrence endpoint attachments;
* physical memory services and regions;
* typed service endpoints, service-leg carrier attachments, and service
  transforms;
* one Transport Architecture owned by the `fabric.system` root;
* zero or more independent Interconnect Implementation objects refining that
  architecture;
* external boundaries;
* address, clock, reset, power, and memory-consistency domains; and
* optional non-semantic visualization metadata.

These are typed Fabric concepts. A generic node kind, open dictionary, protocol
string, or placeholder record is not an alternative system schema.

The host/accelerator organization partitions these existing objects into a
HostCore domain and a weakly coupled accelerator domain containing
heterogeneous AccCores, the accelerator Transport Architecture, and
accelerator memory or services. Communication uses the same typed service
endpoints, transport resources, and directed connections as the rest of the
System graph. The partition is derived from those objects and connections; it
is not a new persistent entity. PCIe, CXL, an SoC interconnect, or a custom
link may implement the interface, but none is a topology assumption or a field
in `fabric.system`.

Every referenced module is selected through the root's canonical
`ImportedModule` dependency table from `docs/spec-fabric-artifact.md`. A
module target inside that dependency is encoded as the dependency ordinal plus
the exact Module-owned local reference. The System root never stores an
`ArtifactReference<T>` in place of the root dependency or treats a digest as a
module-local target.

## Canonical Service Schema

The Canonical Service Schema is the sole owner of logical operation semantics
shared by software obligations, Fabric capabilities, Mapping, simulation, and
implementation refinement. Canonical Service Schema 2.0 has exactly six
parameterized kinds:

```text
message_transfer<Payload>
  arguments: payload : Payload
  results: completion : none
  effects: none
  legs:
    0 message(source -> sink, payload)
  completion: sink accepts leg 0

memory_read<Access>
  arguments: address : AddressOf(Access), mask? : MaskOf(Access),
             control : none
  results: data : DataOf(Access), completion : none
  effects: read(logical memory service)
  legs:
    0 request(manager -> provider, address, mask?, control)
    1 response(provider -> manager, data, completion)
  completion: response leg 1 is accepted

memory_write<Access>
  arguments: address : AddressOf(Access), data : DataOf(Access),
             mask? : MaskOf(Access), control : none
  results: completion : none
  effects: write(logical memory service)
  legs:
    0 request(manager -> provider, address, data, mask?, control)
    1 response(provider -> manager, completion)
  completion: response leg 1 is accepted

memory_atomic_rmw<Access>
  arguments: address : AddressOf(Access), update : DataOf(Access),
             mask? : MaskOf(Access), control : none
  results: old : DataOf(Access), completion : none
  effects: read_modify_write(logical memory service)
  legs:
    0 request(manager -> provider, address, update, mask?, control)
    1 response(provider -> manager, old, completion)
  completion: response leg 1 is accepted

memory_compare_exchange<Access>
  arguments: address : AddressOf(Access), expected : DataOf(Access),
             desired : DataOf(Access), mask? : MaskOf(Access), control : none
  results: old : DataOf(Access), success : SuccessOf(Access),
           completion : none
  effects: compare_exchange(logical memory service)
  legs:
    0 request(manager -> provider, address, expected, desired, mask?, control)
    1 response(provider -> manager, old, success, completion)
  completion: response leg 1 is accepted

memory_fence<FenceContract>
  arguments: control : none
  results: completion : none
  effects: order(memory consistency domain)
  legs:
    0 request(manager -> consistency provider, control)
    1 response(consistency provider -> manager, completion)
  completion: response leg 1 is accepted
```

`Payload` is an exact supported type, not a byte-count hint. For every
addressed memory kind, `Access` is the nonpersistent
`CanonicalMemoryAccessView` mechanically derived from the exact Dataflow
actor. `AddressOf`, `DataOf`, optional `MaskOf`, and `SuccessOf` are projections
of that view. `MaskOf` is present exactly for a dynamic-mask actor;
`SuccessOf` is `i1` for scalar or whole-payload compare-exchange and the exact
row-major `i1` access shape for per-lane compare-exchange. The actor remains
the only persistent owner of its type, shape, mask, and access contract. A
service obligation references that actor, and every consumer derives the same
view rather than serializing a copy.

An endpoint capability binds the kind, declares a closed accepted access or
contract domain, and owns its visibility, alignment, width, rate, outstanding
capacity, and progress guarantees. The kind owns argument/result order,
effects, leg direction and ordinal, and completion event once. Endpoint and
Mapping records reference that capability; they cannot copy or weaken it.

Plain and atomic read or write share one natural operation shape. A capability
must explicitly admit the selected `Plain` or `Atomic` contract; atomic support
does not implicitly admit plain mode, and plain support does not imply
atomicity. Volatile is an actor contract value, not another service kind.
Coherence and MMIO are physical service or region properties, not operation
names. There is no generic service name, property bag, callback, or operation
DSL. The three-kind plain-only memory model from Canonical Service Schema 1.0
is superseded because it cannot express these actor contracts without
duplicated operation kinds.

## AccCore And SpatialCore Attachment

The architecture uses:

```text
AccCore = InstructionCore + SpatialCore
```

Both HostCore and AccCore InstructionCore use one closed Architectural
Contract. `loom.fabric 4.1` has one ISA variant, `RiscV`; adding another ISA is
a schema change rather than an open string or opaque payload:

```text
InstructionCoreArchitecturalContract = RiscV {
  xlen                        : X32 | X64
  base                        : I | E
  extensions                  : canonical set<RiscVExtension>
  endianness                  : Little | Big
  physical_address_width_bits : positive uint32, <= xlen
  privilege_modes             : canonical non-empty set<User | Supervisor | Machine>
  abi_capabilities            : canonical non-empty set<RiscVAbi>
  memory_ordering              : Rvwmo | Ztso
  sync_scopes                 : canonical non-empty set<SingleThread | Hart | System>
  code_models                 : canonical non-empty set<MediumLow | MediumAny>
  relocation_models           : canonical non-empty set<Static | PositionIndependent>
  runtime_services            : canonical set<RuntimeService>
}

RiscVExtension =
    M | A | F | D | C | V | Zicsr | Zifencei | Zba | Zbb | Zbs | Ztso

RiscVAbi =
    Ilp32 | Ilp32e | Ilp32f | Ilp32d | Lp64 | Lp64f | Lp64d

RuntimeService =
    ThreadDispatch | SpatialLaunch | MemoryAllocation | AtomicRuntime
```

A root-complete `fabric.system` contains exactly one HostCore and one or more
AccCores. For `N` AccCores, these are the `N + 1` stored-program engines of the
System. The HostCore remains a distinct `HostCoreOccurrenceRef`; it is not an
AccCore occurrence and owns no SpatialCore occurrence binding or endpoint
attachment.

All `N + 1` Architectural Contracts must form one executable ISA and ABI
cohort: they have the same XLEN and endianness, and the intersection of their
declared ABI-capability sets is nonempty. Extensions, privilege support,
physical-address width, code and relocation models, runtime services, and
Microarchitectural Realizations remain exact per-core capabilities. Later
Compiler Target Binding and Deployment validation select values admitted by
each exact core and cannot widen this Fabric-owned cohort.

The canonical set order is the closed wire-enum order shown above. Duplicate,
unknown, or noncanonical set entries are invalid. `E` requires `X32`; `D`
requires `F`; an `*f` or `*d` ABI requires the corresponding extension; an
`lp64*` ABI requires `X64`; an `ilp32*` ABI requires `X32`; and `Ilp32e`
requires base `E`. Every other ABI requires base `I`. The selected ABI fixes
the pointer and C data model, while `xlen` fixes the integer register width, so
neither fact is serialized a second time. The supported privilege set must
include `Machine`. The runtime-service set may be empty for a HostCore that is
not an accelerator dispatcher; every AccCore requires `ThreadDispatch` and
`SpatialLaunch`. `Ztso` memory ordering requires the `Ztso` extension.

These are hardware architecture facts. Compiler triple, CPU spelling,
DataLayout, runtime library selection, gem5 model names, cache sizes,
speculation policy, and pipeline organization are not fields of this contract.
The exact contract has one domain-separated architecture fingerprint used by
`CompilerTargetBinding`, but the digest is an index and never replaces the
typed contract. The fingerprint is the Common digest of the canonical
Architectural Contract record bytes under the domain
`loom.fabric.instruction_core_architecture.1.0`.

An AccCore is one physical occurrence and contains exactly one InstructionCore
plus exactly one SpatialCore occurrence binding. The InstructionCore
description has an
Architectural Contract for binary compatibility and a Microarchitectural
Realization for execution structure, timing, and capacity. Simulator model
names and compiler target spellings are bindings over that description, not
Fabric hardware facts.

The Microarchitectural Realization is one closed sum:

```text
InstructionCoreMicroarchitecturalRealization =
    InOrder {
      common
      fetch_width, decode_width, issue_width, commit_width
      memory_issue_width, memory_commit_width
      max_outstanding_memory_operations
      store_buffer_entries
    }
  | OutOfOrder {
      common
      fetch_width, decode_width, rename_width, dispatch_width
      issue_width, writeback_width, commit_width
      reorder_buffer_entries, issue_queue_entries
      load_queue_entries, store_queue_entries
      physical_integer_registers
      physical_float_registers
      physical_vector_registers
    }

common = {
  hardware_thread_count : positive uint32
  execution_units       : canonical non-empty sequence<ExecutionUnitRecord>
  resource_contract     : ResourceContract
}

ExecutionUnitRecord = {
  operation_class       : InstructionOperationClass
  count                 : positive uint32
  latency_cycles        : positive uint32
  initiation_interval   : positive uint32
}

InstructionOperationClass =
    IntegerAlu | IntegerMultiply | IntegerDivide | Branch
  | LoadStore | FloatingPointAlu | FloatingPointMultiply
  | FloatingPointDivide | VectorAlu | VectorMultiply | System
```

Every width and capacity field is positive. Execution-unit records are sorted
lexicographically by `(operation_class, latency_cycles,
initiation_interval)`; two records with the same tuple are merged by checked
addition of `count`, and an unrepresentable sum is invalid. The exact variant
and all fields enter Fabric identity. Cache hierarchy, branch-predictor shape,
pipeline stage names, rename maps, dynamic queues, speculative state, and
provider-private scheduling remain implementation or simulation state unless
an observable shared capacity is deliberately exposed through the one
`resource_contract`.

The architecture and microarchitecture codecs are separate. Changing only the
Microarchitectural Realization preserves binary compatibility but changes the
Fabric artifact identity and performance model. Changing the Architectural
Contract changes both compatibility and Fabric identity.

Both record codecs use unsigned big-endian fields. Closed variants, enum
values, sequence counts, widths, capacities, and execution-unit fields are
`u32be`. The embedded canonical `ResourceContract` is framed by a `u64be` byte
count followed by its exact production record bytes. Sequence elements appear
in their canonical order and no padding, unknown field, or trailing byte is
admitted. Strict import reconstructs the typed record, re-encodes it through
the same production codec, and requires byte equality.

For a selected AccCore, its exact InstructionCore Architectural Contract
mechanically selects and validates the Compiler Target Binding owned by
`docs/spec-executable-closure.md`.
Neither that binding nor its target-specific binary enters `fabric.system`
identity or SystemMapping identity.

Its SpatialCore references one exact `fabric.module` template. Multiple
AccCores may reference the same template while remaining distinct physical
resources. Their internal physical targets and symbolic domain slots are
qualified by their exact `SpatialCoreOccurrenceRef` as defined by
`docs/spec-fabric-identity.md`; importing the Module never clones or rebinds
its definition-local identifiers.

Because the InstructionCore cardinality is one, its Mapping reference is
derived rather than allocated. `docs/spec-fabric-identity.md` solely owns the
persistent `InstructionCoreContextRef` framing; this specification owns the
rule that every AccCore has exactly one such context at its fixed ordinal.

This reference is distinct from the instruction-context domain of a temporal
PE. Thread binding selects the AccCore only; it never makes a second
InstructionCore target choice.

The InstructionCore Microarchitectural Realization is also the unique owner of
its mapping-visible execution-resource contract. That contract contains a
closed typed `ResourceState` set, the canonical all-free initial state, typed
capacity dimensions, atomic `UsePattern` values, a stable typed requester
order, and an exact `GrantPolicy` or closed exact refinement domain. A pattern
may claim the instruction context and several declared shared capacities as
one atomic use. SystemMapping binds workload-specific demand, activation,
release, and sharing values; it cannot split the pattern or define another
scheduler.

The initial schema has one mapping-visible admission requester, the derived
`InstructionCoreContextRef`, so its requester order and grant are structural
and have no selectable policy field. Internal pipeline stages, registers,
caches, speculation, and gem5 state remain implementation or simulation
details unless the Fabric contract deliberately exposes one as a
Mapping-visible shared capacity. Dynamic occupancy, instruction progress, and
grant state are transient and never persist in Fabric or Mapping.

`docs/spec-fabric-resource-contract.md` owns the shared meanings of
`ResourceState`, atomic `UsePattern`, requester ordering, and `GrantPolicy`.
The Microarchitectural Realization owns the concrete closed values that use
those atoms.

Every fully elaborated occurrence has one `fabric.system.spatial_attachment`
row for each imported Module boundary endpoint. Each row contains the Module
boundary endpoint and the corresponding occurrence-qualified AccCore-local
SpatialCore endpoint. A Transport row contains exactly that structural pair.
A Memory row additionally contains the exact `SystemServiceEndpointRef` that
continues the occurrence endpoint into the System service topology. Direction,
type, and role are derived from the Module and occurrence endpoints; the exact
service capability set remains owned by the referenced System endpoint. The
attachment copies neither that set nor a workload capability requirement.

An endpoint attachment is not a route and cannot hide conversion, buffering,
arbitration, clock-domain crossing, or any other stateful behavior. Such
behavior requires an explicit Fabric resource or transfer pattern. Every
Module boundary endpoint has exactly one endpoint attachment. Its effective
Clock and Reset are derived from the endpoint's Module slot and that occurrence
slot's System domain membership; neither the AccCore nor SpatialCore parent
supplies an inherited domain.

Value, stream, control, completion, and other token transfers remain typed
transport contracts across the attachment. A memory endpoint remains a typed
operation-service capability. It is never flattened into an untyped data plane
or inferred from a token edge.

## Endpoint And Service Model

Fabric endpoints expose operation-relative capability. The Canonical Service
Schema above defines the exact initial message-transfer and plain memory
operations. Each operation owns typed arguments and results, logical effects,
ordering and completion semantics, and abstract transfer legs.

An endpoint declares whether it can initiate or serve each operation, together
with typed constraints such as address range, width, alignment, burst,
outstanding-request capacity, issue or accept rate, ordering domain, and
coherence membership.

Manager/requester and subordinate/provider are endpoint-relative memory roles:

* a manager endpoint can initiate a memory operation;
* a subordinate endpoint can accept and serve a memory operation.

An endpoint is not a software address space or physical storage identity. One
endpoint may serve several mapped logical memories, and one logical memory may
be reachable through several endpoints. Mapping records own those relations.

### Service-Leg Carrier Attachment

A memory-plane endpoint does not itself expose a transport carrier. The exact
`fabric.system` root therefore owns one structural relation from each admitted
memory-service leg to its non-empty physical carrier domain:

```text
ServiceLegCarrierAttachmentKey = {
  endpoint: FabricMemoryEndpointRef
  kind: ServiceKind
  leg_ordinal: dataflow::StructuralOrdinal
}

ServiceLegCarrierAttachment = {
  key: ServiceLegCarrierAttachmentKey
  carriers: canonical non-empty sorted-unique
            Set<FabricTransportEndpointRef>
}
```

The key is a structural lookup key, not an entity or identity. The relation
receives no `EntityId`, local reference kind, capability ordinal, or Mapping
record. It admits exactly two memory-endpoint cases:

* a memory-plane `SystemServiceEndpointRef` at ordinal zero uses its own exact
  capability set;
* an occurrence-qualified SpatialCore memory endpoint uses the exact
  `SystemServiceEndpointRef` in its unique memory `spatial_attachment` as its
  capability authority.

No other memory-endpoint owner is admitted. In the second case the occurrence
endpoint remains a Module-derived structural endpoint: it gains no service
entity, capability set, or independent operation-admission authority. Its
manager or subordinate role is the complement of the bound System endpoint's
role, while the bound System endpoint remains the sole owner of the accepted
operation domain. The capability set admits at most one record for each
`(ServiceKind, EndpointRole)` pair. For a System endpoint, that endpoint and
kind select one exact capability. For an occurrence endpoint, its unique
memory spatial attachment first selects the exact System endpoint, after which
the kind selects one exact capability. Neither case needs a capability
ordinal. The schema-local leg ordinal then selects one leg of that kind.

The leg ordinal uses the existing Dataflow-owned
`dataflow::StructuralOrdinal` semantic domain. Its persistent field reuses the
canonical unsigned 64-bit big-endian framing already used by
`CanonicalServiceLegKey`; Fabric defines no local ordinal type or second
ordinal codec. Finalization validates only that the ordinal is less than
`CanonicalService(kind).legCount()` before deriving the leg's direction and
payload semantics from that owner.

The [Canonical Service Schema](#canonical-service-schema) remains the sole
owner of the number, ordinal, direction, payload roles, and payload types of
the legs. Fabric owns only which physical transport endpoints can carry each
leg. For an occurrence endpoint, the effective `Initiate | Serve` role is
derived from its manager or subordinate role. Carrier direction is derived
mechanically from that effective endpoint role and canonical leg direction:

| effective service role | `InitiatorToServer` leg | `ServerToInitiator` leg |
|---|---|---|
| `Initiate` | transport output | transport input |
| `Serve` | transport input | transport output |

For every memory or fence capability and every leg of its exact service kind,
finalization derives the canonical leg payload over the capability domain. A
leg is one ordered collection of independent semantic `ServiceValue` tokens
under one transaction and one shared logical `RouteTree`; it is not a packed
tuple or a new Fabric payload type. Its nonpersistent required payload-width
envelope is exactly:

```text
max(flattened transport width of each active ServiceValue role
    at every point admitted by the capability domain)
```

An optional role participates when any admitted point activates it. A
control or completion value of type `none` has width zero and creates no
semantic bit. The payload field of `!fabric.bits_tag<W,T>` has capacity `W`;
the tag field `T` never contributes payload capacity. For example, a
`memory_write` request with a 64-bit address, 128-bit data, a 16-bit mask, and
zero-bit control has required width 128, not 208.

Every independent value must fit without splitting or serialization in every
selected carrier endpoint and traversal under the canonical low-bit-aligned
transport rule. Taking the maximum does not permit a 128-bit value on a
64-bit segment. Widths are never summed, and this projection defines no tuple
layout, field offset, byte order, role-specific attachment, role-specific
route, persistent envelope field, schema field, configuration field, or codec.
Memory beat decomposition remains owned by the selected Memory Service
Contract and its declared use patterns. Protocol channels, request/response
subchannels, packets, flits, headers, physical serialization or parallelism,
occupancy, and timing remain Interconnect Implementation concerns.

Finalization retains only carriers whose derived direction matches and whose
payload data-field width is at least that entire envelope under canonical
low-bit-aligned transport. The relation does not copy a workload actor,
payload, payload width, accepted domain, endpoint role, or protocol name.
Compatibility is always recomputed from the pair's exact System endpoint
capability, Canonical Service Schema, selected memory endpoint role, and
transport endpoint inventory.

Every admitted memory or fence capability leg has exactly one attachment row
for its System service memory endpoint and at least one compatible carrier.
For every memory `spatial_attachment`, each such leg of the row's bound System
endpoint also has exactly one attachment row for the occurrence-qualified
SpatialCore memory endpoint and at least one compatible carrier. Reusing one
System endpoint across several occurrences reuses its one System-endpoint row
while retaining one row for each distinct occurrence endpoint. All memory and
transport endpoint references resolve inside the same exact Fabric root.
`MessageTransfer` already belongs to the transport plane and must not have an
attachment row. One transport endpoint may appear in several rows; sharing,
capacity, contention, and occupancy remain owned by transfer patterns,
`ResourceUse`, and their selected routes.

Authoring rows with the same key are coalesced by set union. Finalization sorts
rows by the complete key and sorts and deduplicates each carrier set. The
canonical form contains one row per key and no empty set. Strict import
re-encodes and rejects persisted duplicate, unsorted, incomplete, or otherwise
noncanonical relation data.

## Physical Memory And Services

Fabric owns physical memory services, address spaces, service regions,
operation capability, ordering, visibility, coherence, latency, bandwidth,
capacity, and typed use patterns. `fabric.mem` may combine an optional
Operation Engine, an optional Local Memory Service, and manager or subordinate
endpoints as described by `spec-fabric-mem.md`.

Every physical service region has one closed behavior kind:

```text
MemoryRegionBehavior =
    Storage
  | Mmio {
      accepted access domain
      NonTrapping
      AtMostOnceLogicalOperation provider observation
    }
```

The region kind is a physical service fact, not a Dataflow operation name or
ordering mode. A SpatialCore mapping may use an `Mmio` region only when its
declared range is non-trapping for the selected accesses. A capability that admits a
volatile actor must preserve one at-most-once provider-observable logical
operation. Internal beats and retries may occur only below that observation
boundary and may not expose duplicate, merged, speculative, or replayed
operations.

Cache, proxy, address translation, hashing, sharding, replication, and
coherence are typed service transforms. They are not inferred from hierarchy,
endpoint names, or the presence of both manager and subordinate ports.

SpatialMapping binds graph-local memory operations to a local service or an
explicit boundary proxy. SystemMapping extends only proxy obligations to a
system provider. Runtime supplies invocation-specific allocations,
authorization, and initial contents within the already selected service and
range envelope.

## Transport Architecture

Transport Architecture is technology-neutral. It owns the explicit directed
routable graph and all Mapping-visible hardware facts needed for topology,
routing, multicast, contention, capacity, deadlock, buffering, latency,
bandwidth, ordering, visibility, and resource-time verification.

Its core concepts are:

```text
typed endpoint
transport resource
directed connection
atomic transfer pattern
typed capacity and service contract
```

An endpoint-pair cost matrix, coordinates, or viewer layout cannot replace
this graph because each would hide shared resources and contention. A link,
lane, FIFO, queue, or other object with independent state, capacity,
configuration, or parallel identity is an explicit typed resource.

Each stateful transport-resource schema uniquely owns its closed typed
`ResourceState` set, canonical initial state, capacity dimensions, stable typed
requester order, atomic `UsePattern` domain, and exact `GrantPolicy` or closed
exact refinement domain. One pattern may atomically claim several real states;
Mapping may select a declared refinement and bind workload values but cannot
split that claim or construct a parallel generic resource/arbiter graph.
Queue contents, occupancy, cursors, credits, and other execution state are
transient and never persist in Fabric or Mapping.

A programmable transport occurrence, when selected into a configured view,
uses the same closed shape as other Fabric resources:

```text
TransportResourceConfiguration =
    Disabled
  | Active { selected_pattern_controls, physical_refinements }
```

`Disabled` carries no route, Tag, selector, reservation, or refinement.
`ConfigurationABI` alone owns its physical inactive encoding. Fixed,
non-programmable connectivity has no artificial configuration record.

### Atomic Transfer Patterns

Each transfer pattern has exactly one ingress and a non-empty egress set. One
egress is forwarding; several egresses are physical replication of one
message. A pattern never has several ingresses. Arbitration is represented by
several single-ingress patterns competing for shared capacity, not by an
implicit software merge or reduction.

Each pattern declares its typed resource-use vector and architecture-visible
latency, initiation interval or service rate, ordering, and progress
guarantees. Concurrent legality follows from aggregate use against declared
capacities. There is no second pairwise-conflict authority.

### Eligibility And Grant

Pattern eligibility is a closed typed relation over declared configuration,
context, route or flow class, Mapping-assigned controls, and ingress readiness.
It is not a free-form predicate program.

Grant chooses a capacity-feasible subset of eligible requests. Each contended
resource's typed Fabric contract owns the exact cycle-visible grant and
state-update behavior, or exposes a closed exact refinement domain selected by
Mapping. `docs/spec-fabric-resource-contract.md` solely owns the shared
`FixedPriority` and `RoundRobin` grant-policy atoms, including requester order,
reset cursor, and successful-grant advancement. Each System resource owns only
its exact requester inventory and any specialized state or refinement domain.

No policy is needed when complete Mapping proves at most one simultaneous
requester. Reachable contention without an exact Fabric/refinement policy is
unsupported for deterministic exact simulation. Ordering, fairness, maximum
wait, guaranteed service rate, priority, QoS, and reservation controls are
derived from or declared alongside that same contract; runtime and Mapping do
not invent a policy. The implementation owns the arbiter circuit and transient
state that execute the selected contract.

## Interconnect Implementation

Interconnect Implementation owns concrete mechanisms: AXI, TileLink, CXL or a
custom protocol; protocol endpoints and bundles; request or response subchannels;
packet, flit, and header formats; credit or ready-valid flow control; concrete
virtual-channel encoding; router pipelines; adapters; arbiters; RTL or IP
blocks; and configuration circuitry. `ConfigurationABI` alone owns the
physical bit/address encoding and programming contract for every exposed
semantic configuration field.

A service-leg carrier attachment is only an architecture-level candidate
relation between existing memory and transport endpoints. It does not create
a request channel, response channel, packet, flit, header, adapter, or physical
encoding. The exact Interconnect Implementation refines the Mapping-selected
transport path into those protocol-specific mechanisms.

Transport Architecture and Interconnect Implementation are independent,
immutable, content-addressed objects in the Fabric Hardware Description
family, even when serialized in one MLIR file. An implementation references
one exact architecture and owns a complete Refinement Relation. It does not
enter the referenced `fabric.system` root's canonical bytes or identity. The
relation covers architecture endpoints, transfer patterns, Mapping-visible
controls, and their concrete implementation correspondences. It is not a
third artifact family or a stored `valid` claim.

SystemMapping references only the exact Transport Architecture and selects
architecture-visible resources and controls. Deployment later combines that
immutable Mapping with one verified Interconnect Implementation. Protocol or
implementation identity never enters SystemMapping.

Refinement must prove that the implementation satisfies architecture service,
connectivity, ordering, backpressure, multicast, capacity, progress, and
performance bounds. An implementation may not hide a shared bottleneck or
restriction absent from the architecture contract. Such a mismatch requires a
new Transport Architecture identity and revalidation of Mapping.

For exact architecture `A`, Mapping `M`, implementation `I`, and refinement
`R`, the semantic implementation configuration projection is the deterministic
result of:

```text
lower(A, M, I, R)
```

Lowering performs no routing, packetization DSE, hidden Mapping choice, or
physical encoding. The exact `ConfigurationABI` then encodes this semantic
projection for each selected Programming Unit. Equivalent physical encodings
collapse to the one canonical encoding exposed by that ABI.

## gem5 Execution Boundary

Gem5 executes concrete system and interconnect implementation microstate:
dynamic arbitration, queues, credits, protocol state, cache and memory-system
state, InstructionCore execution, and the whole-system event queue. Its grant
sequence must follow the exact Fabric/Mapping-selected refinement contract.
Gem5 does not own Fabric topology, cycle-visible policy, Mapping choices, or
the Interconnect Implementation definition.

The gem5 execution contains the HostCore side and accelerator-side
InstructionCores, NoC, caches, coherence, and accelerator external memory.
Loom's bridge supplies only the SpatialCore participant at each exact Spatial
Launch boundary. A DFG-sim host-retargeted oracle module is never executable
input to this system path.

A workload-independent Gem5 Simulation Binding maps exact Fabric and
Interconnect Implementation objects to gem5 models, SimObjects, parameters,
and the Bridge ABI. It is a simulator binding, not hardware truth. Every modeled
InstructionCore must validate all three authorities: the exact InstructionCore
Architectural Contract; the exact InstructionCore Microarchitectural
Realization, including execution structure, timing, capacity, and
mapping-visible resources; and the compatible Compiler Target Binding used by
its target-specific binary. The system-simulator descriptor references the
shared system-simulation case signature with ordered `deployment` and
`system_model` roles; an ordinary `EvaluationRequest` binds their exact
subjects. Exact workload and runtime data use
`SimulationWorkload` and `SimulationRuntimeInput` references, and remaining
simulator parameters use `ResolvedModelBinding`. There is no separate
system-simulation request family.

The exact persistent `Gem5SimulationBinding` root and total correspondence
table are owned by `docs/spec-runtime-abi.md` section `Gem5 Simulation
Binding`. This architecture view does not copy that schema.

The generated gem5 projection is derived configuration. Handwritten simulator
configuration cannot become another topology, memory, route, timing, or
protocol authority.

## SystemMapping Boundary

SystemMapping binds an exact Canonical Dataflow Program, the architecture-only
`fabric.system`, and its exact Transport Architecture. Its only persistent
coverage root is a canonical non-empty set of root thread launches. The exact
imported SpatialMapping set is the finite, deduplicated value range of
normalized graph execution bindings over all reachable static graph launches
and their legal may-domains. There is no separate editable selected-set field
and no fixed cardinality.

Compiler Target Binding, target-specific binary, Interconnect Implementation,
HardwareImplementation, Gem5 Simulation Binding, and Deployment identities are
outside SystemMapping semantic identity. Those consumers resolve and validate
their own exact bindings from the Mapping-selected AccCores and architecture.

SystemMapping has exactly three persistent record families:

```text
ExecutionBinding
ServiceRealization
ResourceUse
```

`ExecutionBinding` contains typed Thread and Graph variants and owns only
where computation executes. `ServiceRealization` owns selected target services,
system route trees, buffers, and Mapping-visible physical refinements.
`ResourceUse` owns event-relative activation, occupancy, typed capacity demand,
Physical Tags, and sharing assignments for the selected structures. A Physical
Tag is only a local interpretation key where may-overlap incompatible uses of
a Fabric match domain require distinction; it is not global firing, iteration,
invocation, or logical-token identity. Fabric remains the sole owner of raw
capacity and use-pattern shape.

`docs/spec-mapping-identity.md` owns the closed typed
`SystemServiceObligationKey` wire. A transfer key stores only its exact
producer-terminal anchor; exact Dataflow scope derives the complete non-empty
sink set, including multicast. An operation-service key stores only its
logical-memory root-or-view or fence-family anchor; exact Dataflow derives the
complete contextual addressed-operation or fence-member set. For a logical
memory owner, Dataflow separately derives the complete memory-exposure set.

This document's Canonical Service Schema remains the sole owner of each
member's local request, data, response, completion, direction, payload, and
ordinal semantics. Mapping combines the obligation key, exact
`ServiceMemberRef`, and schema-local leg ordinal as one
`CanonicalServiceLegKey`, then realizes each required leg explicitly over the
Transport Architecture. Protocol channels are derived only by Interconnect
refinement.

A memory exposure is a provided capability boundary, not an operation member.
It has no Canonical Service leg. A Mapping service-target binding selects its
provider endpoint and region using the Dataflow-owned `MemoryExposureRef`;
request and response legs arise only from addressed actors that use that
capability. Fabric does not add an exposure service kind or a zero-leg
operation to compensate.

End-to-end continuity composes, in order:

```text
source SpatialMapping
source module-to-AccCore attachment
SystemMapping ServiceRealization
destination attachment
destination SpatialMapping or system provider
```

No layer copies its neighbor's endpoint or route authority.

## Runtime And Deployment Boundary

Runtime evaluates one immutable, preverified SystemMapping. It may wait,
backpressure, and atomically admit a compiled activation set. It cannot change
an AccCore, SpatialMapping, route, tag, context, service, or configuration.

`docs/spec-configuration-deployment.md` is the sole owner of runtime-image
child membership, identity status, schemas, canonical child keys and ordering,
and presence conditions. This Fabric contract contributes the exact
architecture, Mapping-selected hardware facts, and implementation refinements
that Deployment validates; it does not write runtime-image, binary, or
implementation facts back into SystemMapping.

For an InstructionCore-only SystemMapping, runtime and Deployment consumers do
not invent Spatial execution. Exact runtime-image and configuration-image
presence follows the Deployment owner, and every transport or other
Programming Unit selected by that SystemMapping remains a closure obligation.

## Domains And External Boundaries

Clock, reset, power, address, and memory-consistency domains are explicit typed
system facts. Domain crossings require explicit resources and verified
semantics. Hierarchy or visualization grouping does not imply membership.

The finalized `fabric.system` root is the sole persistent authority for the
complete domain and crossing relation. Clock/reset validation consumes only a
root-complete `FabricSystemRootView`, the typed refinement defined by
`docs/spec-fabric-artifact.md`:

```text
validateClockReset(FabricSystemRootView) -> ValidatedClockResetView
```

The input view enumerates every point connection, spatial attachment,
hardware-domain declaration and member, system transport resource, transfer
pattern, and optional crossing contract in canonical order. The validator does
not accept a `ClockResetStructure`, connection vector, domain vector, crossing
vector, or any other caller-supplied shadow catalog. `FabricImportBinding`
checks exact artifact scope only and is not a completeness proof.

`ValidatedClockResetView` is a removable derived index over the same immutable
root storage. It may cache exact member-to-domain and carrier-to-crossing
lookups, but it owns no copied declarations and cannot outlive or disagree
with its `FabricSystemRootView`.

Concrete System-domain membership uses the one closed
`FabricHardwareDomainMemberRef` wire defined by
`docs/spec-fabric-identity.md`.

`DirectSystemOwner` retains the existing kind-owned membership semantics for
System objects. A Clock or Reset domain admits only the identity spec's exact
`FabricClockResetDirectOwnerRef` subset. It rejects `AccCoreOccurrenceRef` and
`SpatialCoreOccurrenceRef`: neither is an inheritance proxy for its
InstructionCore, Module boundary, or Module internals. An InstructionCore uses
its exact independently typed context reference. An imported Module's Clock or
Reset role uses only `SpatialCoreSlot`.

Every direct Clock/Reset physical owner is associated with exactly one Clock
and one Reset domain. A direct owner with a nonempty canonical ResourceState
inventory consumes both signals and defines every state's reset value; an
owner with no state is combinational and consumes neither signal. The
association still closes ordinary connectivity for a combinational owner. The
only exception is an asynchronous crossing transport resource, whose two-sided
Clock and Reset associations are owned by its crossing contract and which has
no ordinary single-domain membership.

For every SpatialCore occurrence, every slot declared by its exact imported
Module is projected to one `SpatialCoreDomainSlotOccurrenceRef` and appears in
exactly one same-kind System domain. The domain membership row itself is the
binding; there is no second slot-binding table. A slot cannot bind by name,
ordinal alone, a different imported Module, or an AccCore-wide default.

The effective domain of a Module boundary or internal target is derived by:

```text
exact Module-local target
  -> exact Module-local physical owner
  -> ModuleDomainAssignment
  -> SpatialCoreDomainSlotOccurrenceRef
  -> exact System HardwareDomainRef
```

The resulting physical target is occurrence-qualified. The expanded
target-to-domain relation is never persisted in the System member list, and a
consumer cannot supply or override it. Reusing one Module in two AccCores may
bind equal or different concrete domains, but the two effective internal
targets remain distinct.

The typed lookup is a function because the requested domain kind is explicit:

```text
effectiveHardwareDomain(
    Direct(FabricClockResetDirectOwnerRef)
      | SpatialCore(SpatialCorePhysicalDomainTargetRef),
    Clock | Reset)
  -> exact HardwareDomainRef
```

For a SpatialCore token or memory boundary, `spatial_attachment` first recovers
the exact Module boundary face and its same-kind slot assignment. For an
internal target, the target-to-physical-owner projection recovers the
assignment. The lookup never treats Clock and Reset as one value and never
persists an expanded target-membership table.

A transfer between different clock domains is legal only through a transport
resource carrying one exact crossing contract:

```text
ClockCrossingContract = AsyncFifo {
  transfer_pattern_ref
  source_clock_domain_ref
  destination_clock_domain_ref
  source_reset_domain_ref
  destination_reset_domain_ref
  depth
  synchronizer_stages
  preserves_order = true
  lossless = true
  backpressure = true
  reset_behavior = FlushToEmptyAfterBothDomainsReleased
}
```

The contract is a typed variant on an existing transport resource and use
pattern, not a hidden connection behavior or a new generic crossing graph.
The current `ClockCrossingContract` admits only the `AsyncFifo` variant. A
direct cross-domain
connection, an attachment that hides crossing state, or a backend-invented
synchronizer is invalid.

One transport-resource occurrence has zero or one crossing contract as one
field of that resource's canonical record. A separate crossing list is
forbidden. Consequently one carrier cannot have duplicate contracts and
lookup never uses first-match behavior. The carrier is recovered from
structural ownership and is not copied inside the contract. The selected
transfer pattern must belong to that exact carrier.

The source and destination Reset domains cover the exact ingress and egress
faces, respectively. When a Reset contract names `synchronous_to`, it must name
the corresponding source or destination Clock domain. The fixed reset behavior
therefore has two exact release authorities rather than an inferred reset from
the carrier owner.

The fixed reset behavior discards no accepted token during ordinary operation.
If either Reset domain is asserted, the crossing cannot accept or publish
traffic until both Reset domains have completed their declared release latency,
after which its observable state is empty. A design that must preserve in-flight
traffic across reset requires a future explicit contract rather than another
policy field.

Coherence is an explicit typed service transform, not a parallel domain kind.
Runtime protection remains an invocation fact rather than permanent Fabric
topology.

External endpoints expose complete typed transfer or operation-service
capability. Partial system closure may terminate only at such an explicit
external provider. Missing adapters, routes, services, or crossings are errors
or declared unsupported scope, never implicit defaults.

Runtime `ProtectionDomain` and memory authorization are invocation facts over
Fabric-owned capability. They do not change permanent hardware topology or
Mapping selection.

## ADG Builder Output

The C++ ADG Builder is an ergonomic producer of Fabric Hardware Descriptions.
Builder-only objects are not downstream authority. Built-in templates and
external descriptions elaborate to the same typed, explicit model before
Mapping, simulation, or backend use.

Helpers may construct regular or irregular networks, memories, AccCores, and
service hierarchies. Their finalized output must contain the actual endpoints,
resources, attachments, directed connections, contracts, and refinement
objects. It must not preserve a generic builder dictionary as a parallel
schema.

## Persistent Declarative Schema

`fabric.system` is the single architecture root. Its body is one declarative
block with no block arguments, SSA values, CFG successors, symbol table, or
runtime terminator. The exact child operation catalog is:

```text
fabric.system.host_core
fabric.system.acc_core
fabric.system.memory_service
fabric.system.service_endpoint
fabric.system.service_leg_carrier_attachment
fabric.system.service_transform
fabric.system.transport_resource
fabric.system.transfer_pattern
fabric.system.connection
fabric.system.spatial_attachment
fabric.system.hardware_domain
fabric.system.external_boundary
```

No generic `node`, `edge`, `kind`, or property operation is accepted. A child
that represents an independently meaningful physical resource receives one
Artifact-global `EntityId`. Structural relationships do not receive IDs.
Ports, Canonical Service legs, service regions, transfer-pattern ingress and
egress, and other owner-local structures use `(owner EntityId, typed ordinal)`.
They never receive a second global identity.

The operations have these semantic fields:

```text
fabric.system.host_core
  EntityId
  InstructionCoreArchitecturalContract
  InstructionCoreMicroarchitecturalRealization

fabric.system.acc_core
  EntityId
  InstructionCoreArchitecturalContract
  InstructionCoreMicroarchitecturalRealization
  exact (ImportedModule dependency ordinal, FabricModuleTemplateRef)
    spatial_core

fabric.system.memory_service
  EntityId
  exact MemoryServiceContractRecord

fabric.system.service_endpoint
  EntityId
  exact owner reference
  canonical non-empty CanonicalServiceCapability set
  absent | exact message carrier type

fabric.system.service_leg_carrier_attachment
  FabricMemoryEndpointRef
  ServiceKind
  dataflow::StructuralOrdinal schema-local leg ordinal
  canonical non-empty sorted-unique FabricTransportEndpointRef set

fabric.system.service_transform
  EntityId
  exact input and output endpoint references
  closed ServiceTransformContract

fabric.system.transport_resource
  EntityId
  typed ports
  ResourceState, capacity, timing, progress, and grant contracts
  optional ClockCrossingContract

fabric.system.transfer_pattern
  SystemTransportResourceRef owner
  owner-local pattern ordinal
  one ingress port
  canonical non-empty egress-port sequence
  atomic typed resource-use vector
  timing, ordering, progress, eligibility, and semantic controls

fabric.system.connection
  Transport {
    exact source transport endpoint or output-port reference
    exact destination transport endpoint or input-port reference
  }
  | MemoryService {
    exact source System service manager endpoint
    exact destination System service subordinate endpoint
  }

fabric.system.spatial_attachment
  Transport {
    exact (ImportedModule dependency ordinal,
           FabricModuleBoundaryEndpointRef)
    exact AccCore-local SpatialCore transport endpoint reference
  }
  | Memory {
    exact (ImportedModule dependency ordinal,
           FabricModuleBoundaryEndpointRef)
    exact AccCore-local SpatialCore memory endpoint reference
    exact SystemServiceEndpointRef
  }

fabric.system.hardware_domain
  EntityId
  closed domain kind
  canonical FabricHardwareDomainMemberRef sequence
  kind-owned typed contract

fabric.system.external_boundary
  EntityId
```

The attachment variant is derived from the Module and occurrence endpoint
pair; it is not a stored plane discriminant. A transport attachment has no
System service endpoint. A memory attachment has exactly one, and that
endpoint is the sole
Fabric-owned continuation of the occurrence-qualified SpatialCore memory
endpoint into the System service topology. Its plane must be Memory and its
role must complement the SpatialCore endpoint. The Module boundary owns no
second capability-domain catalog against which Fabric root finalization could
compare the endpoint. SystemMapping domain construction and base verification
instead test each selected Dataflow memory or fence member against this bound
endpoint's exact capability set. An incompatible selected member is
infeasible; capability equality with another endpoint, owner identity, entity
order, or a unique candidate observed by one consumer cannot substitute for
the explicit reference.

`CanonicalServiceCapability` binds one exact Canonical Service kind, one
operation-relative `Initiate | Serve` role, and a closed accepted access or
actor-contract domain, plus address/range, alignment, issue/accept rate,
outstanding capacity, consistency-domain reference, visibility, and progress.
For an addressed memory kind, the domain is tested against the exact derived
`CanonicalMemoryAccessView`; it does not copy actor-owned fields. The
consistency-domain reference is present exactly when required by the selected
service kind or accepted actor contract. Fields not owned by that service kind
are absent rather than populated with defaults.

The `MessageTransfer` domain owns a non-empty canonical set of exact payload
types and may additionally own one fixed-vector family and a finite exact
`PointerFormatRelation`. The vector family contains a non-empty canonical set
of exact scalar integer or floating element types, a positive maximum
flattened payload width, and a positive maximum fixed rank bounded by the
canonical type codec. Admission compares exact canonical element types and
checked row-major flattened width; it does not enumerate shapes, collapse
equal-width element types, or admit scalable vectors. A pointer payload is
admitted only when its address space and the application module's exact
DataLayout-derived representation width, address width, and stable-integral
layout kind match one listed pointer format. `!llvm.ptr<AS>` alone supplies no
width and no fallback target layout is legal. The exact payload set remains
authoritative for non-pointer scalars and for any deliberately listed vector
outside the family.

`fabric.system.service_endpoint` is the sole System-level physical owner of an
operation-service endpoint. Its owner is exactly one `HostCoreOccurrenceRef`,
`AccCoreOccurrenceRef`, `SystemMemoryServiceRef`,
`SystemServiceTransformRef`, or `ExternalBoundaryRef`. Those entities own the
endpoint entity but do not expose a second endpoint inventory. Multiple
physical ports are represented by multiple endpoint entities, never by a
nonzero endpoint ordinal.

All capabilities in one endpoint have one common role and belong to one common
plane. `MessageTransfer` selects the token-transport plane; every memory read,
write, atomic, compare-exchange, and fence kind selects the memory-service
plane. Mixing planes or roles in one endpoint is invalid. The selected plane
has exactly one endpoint at ordinal zero. On the transport plane, `Initiate`
is an output and `Serve` is an input. On the memory plane, `Initiate` is a
manager endpoint and `Serve` is a subordinate endpoint.

A message endpoint has exactly one physical carrier type, either
`!fabric.bits<W>` or `!fabric.bits_tag<W,T>`. The carrier must represent every
admitted payload type and pointer representation under the canonical
low-bit-aligned transport rule. A
System service memory endpoint has no carrier type because its beat width and
accepted operation domain are already owned by its capabilities. An attached
occurrence memory endpoint likewise remains on the memory plane and reuses the
pair's System capability authority. Both endpoints' service legs use only the
root-owned `ServiceLegCarrierAttachment` relation above. No owner, connection,
or Mapping record may override these derived plane, direction, role, or type
facts.

`fabric.system.external_boundary` is only the identity of one external
interface grouping. Its complete outward contract is the canonical non-empty
set of `fabric.system.service_endpoint` entities whose owner refers to that
boundary. The boundary operation stores no endpoint list, capability copy, or
generic external-contract bag. A boundary with no owned endpoint is invalid at
root-complete finalization.

`MemoryServiceContractRecord` is defined once by
`docs/spec-fabric-mem.md`. A System memory service imports that exact record in
the `System` owner context: region addresses are absolute, `SystemDomain` is
admitted, and `LocalProvider` is rejected. It does not repeat service regions,
actor or access admission, ResourceState, UsePattern, capacity, timing,
progress, or grant fields as sibling System properties.

Every service transform input is an exact ordered sequence of Memory-plane
`Serve`/subordinate endpoints owned by that transform, and every output is an
exact ordered sequence of Memory-plane `Initiate`/manager endpoints owned by
the same transform. The input sequence is where upstream requests enter the
transform; the output sequence is where transformed requests continue toward
downstream providers. A transform endpoint cannot appear in both sequences,
and every Memory-plane endpoint owned by the transform appears exactly once in
the applicable sequence. Capability compatibility does not connect endpoints.

`ServiceTransformContract` is a closed sum. The current contract admits
`AddressOffset`, `AddressMaskXor`, `StaticInterleave`, and `CoherentMemory`;
each variant owns its exact typed parameters and total input-to-output
relation:

```text
AddressOffset { address_width, signed_offset }
  one input and one output; out = in + signed_offset
  the declared input range must make the result representable

AddressMaskXor { address_width, and_mask, xor_mask }
  one input and one output; out = (in & and_mask) ^ xor_mask

StaticInterleave { granule_bytes, output_count }
  one input and exactly output_count outputs
  granule_bytes and output_count are positive
  q = address / granule_bytes; r = address % granule_bytes
  output_ordinal = q % output_count
  output_address = (q / output_count) * granule_bytes + r

CoherentMemory {
  MemoryConsistencyDomainRef consistency_domain
  canonical non-empty region correspondence:
    input_region -> output_region
}
  every input and output region reference occurs at most once
  paired regions have the same nonzero size
  an address a in input_region maps to
    output_region.base + (a - input_region.base)
  the input region is a physical copy or proxy of its paired output region
  under the exact domain contract
```

All non-address arguments and results pass unchanged, and the output endpoint
capabilities must accept the transformed range and exact service signature.
The `CoherentMemory` correspondence is a canonical partial bijection over
Fabric-owned service-region references. All ordered input endpoints are ingress
ports for that one relation. An ordered output endpoint can realize a
correspondence only when its explicit downstream MemoryService closure reaches
the correspondence's output region. The endpoint order and explicit
connections therefore remain the topology owner; the correspondence remains
the storage-identity and region-relative address owner. No input/output ordinal
is copied into a correspondence.

For one incoming address, every matching correspondence is a legal coherent
provider alternative. A selected Mapping plan chooses a subset whose matching
input-region domains cover each source address exactly once. Unlike
`StaticInterleave`, unused `CoherentMemory` output alternatives are not missing
collective branches. Multiple output endpoints that reach the same output
region and have the same composed transform path form one derived branch group.
Overlapping input regions therefore create alternatives that Mapping must
disambiguate; they do not authorize duplicated requests.

An identity transform is represented by a direct MemoryService connection and
has no operation. `CoherentMemory` is the only authority that permits overlapping
physical service regions to represent one coherent service identity. Domain
membership alone never implies storage identity, replication, or coherence.
Cache behavior beyond this architecture-level coherence relation, arbitrary
hashing, programmable callbacks, and opaque custom transforms require a future
closed variant rather than an open extension bag.

The current `hardware_domain` contract uses the closed kinds `Clock`, `Reset`,
`Power`, `Address`, and `MemoryConsistency`. Their contracts are exactly:

```text
Clock { period_fs: positive uint64, phase_fs: uint64 where phase_fs < period_fs }
Reset { polarity: ActiveHigh | ActiveLow,
        assertion: Synchronous | Asynchronous,
        deassertion: Synchronous | Asynchronous,
        initial_state: Asserted | Deasserted,
        synchronous_to: absent | ClockDomainRef,
        release_latency_cycles: uint32 }
Power { nominal_voltage_uv: positive uint64 }
Address { address_width: positive uint32,
          canonical disjoint half-open unsigned ranges }
MemoryConsistency {
  canonical non-empty participant service or provider references
  release_visibility_point : AtLinearization | ByRetirement
  progress : BoundedCompletion {
               progress_clock: ClockDomainRef,
               max_issue_to_retire_ticks: positive uint64
             }
           | FairEventual
  closed ResourceState and atomic UsePattern domains
}
```

`MemoryConsistency` has one fixed linearization and completion contract rather
than configurable booleans that could weaken software semantics:

* every admitted atomic addressed action and fence has exactly one logical
  linearization after issue and before retirement;
* modifying actions on one exact `AtomicObjectKey` form one total modification
  order, RMW is one indivisible read/write action, and every admitted
  sequentially-consistent action participates in the domain's one compatible
  total sequentially-consistent order;
* acquire visibility required by the exact Dataflow contract is imported
  before the actor's result and done retirement packet can publish;
* an addressed actor retires only after its linearization and required
  visibility obligations hold, while a fence retires only after its exact
  ordering and visibility closure holds; and
* all results and done for one actor retire atomically according to the
  Canonical Service Schema.

`release_visibility_point` is the only current visibility timing choice. It
states whether a release summary becomes importable at the publishing action's
linearization or at a provider event no later than its retirement. Fence
publication and import still follow the carrier and reads-from rules in
`docs/spec-dataflow-memory-consistency.md`; a fence does not become a free
global barrier.

`BoundedCompletion` guarantees the declared number of rising-edge ticks of its
exact `progress_clock` whenever required participants remain clocked, the
operation is admitted, and downstream retirement is ready. The referenced
clock must cover the consistency provider that owns completion. `FairEventual`
guarantees eventual retirement under the domain's already declared fair grant
policies and downstream progress but claims no numeric bound. There is no
`Unknown`, timeout, best-effort, or provider-private progress mode. A provider
unable to implement the selected contract is typed `Unsupported`.

Linearization and completion are fixed schema invariants because the current
Canonical Dataflow memory model admits no weaker alternative. Visibility point
and progress class are persisted because they distinguish observable hardware
capability. Concrete modification order, reads-from, visibility frontiers,
queue occupancy, and grant state remain dynamic execution state.

The hardware-domain catalog is one canonical sorted-unique sequence keyed by
`HardwareDomainRef`. A domain reference is declared exactly once. Each
domain's member sequence is canonical sorted-unique over the complete
`FabricHardwareDomainMemberRef` bytes. The same exact member may belong to at
most one domain of a given kind; membership in one Clock and one Reset domain
is legal. Every required occurrence-qualified slot belongs to exactly one
same-kind domain. Duplicate membership within one declaration, duplicate Clock
or Reset declarations, wrong-kind slot membership, an unbound required slot,
and membership of one exact member in two Clock or two Reset domains are
invalid.

The required direct Clock/Reset membership set is derived exactly from the
root's complete `FabricClockResetDirectOwnerRef` inventory after excluding the
crossing transport resources described above. The required imported-Module
membership set is the complete occurrence-slot projection. These two derived
sets are the only coverage authority for builtins and custom Systems; a builder
cannot supply a smaller required-member list.

Endpoint ownership is related to domain membership through the total exact
endpoint-owner to inventory-owner projection defined by
`docs/spec-fabric-identity.md`. Validation compares complete typed references.
It never extracts a parent `EntityId`, silently skips an owner-relative member,
or treats two different owner-relative references as the same entity. A
SpatialCore endpoint obtains its effective domain from its exact
`spatial_attachment`, Module boundary assignment, and occurrence-slot binding;
it does not obtain a domain from its AccCore or SpatialCore parent. A crossing
carrier has no ordinary single Clock or Reset membership; its ingress and
egress faces derive their source and destination domains from its exact
crossing contract.

Every atomic or fence capability references exactly one
`MemoryConsistencyDomain`. Scope compatibility is derived from the exact
Dataflow `SyncScopeRef`, compiler-target scope semantics, explicit domain
participants, and Fabric topology; Fabric does not copy target scope keys into
a second vocabulary. The domain guarantees the dynamic modification-order,
release/acquire, fence, and global sequentially-consistent relations required
by every contract it admits. Concrete modification order, reads-from,
synchronizes-with, sequentially-consistent order, queue occupancy, and grant
state are execution state and never persistent Fabric fields.

The domain is a capability contract, not a second dynamic consistency engine.
An exact execution provider interprets that contract. A provider may execute
the domain locally only when its complete participant and service closure is
inside that provider's simulation boundary. A domain that crosses the system
boundary delegates its external dynamic state to the selected whole-system
provider through the typed Spatial Service boundary. Fabric does not prescribe
an execution trace or duplicate provider-owned modification order,
reads-from, cache, or coherence state.

A reset with synchronous assertion or deassertion names exactly one clock
domain in `synchronous_to`; a fully asynchronous reset omits it. Release
latency is measured in that clock domain and is zero only when release has no
synchronous delay. Reset-domain membership and every clock crossing are
complete typed facts used by RTL and constraint derivation.

Every stateful imported Module owner obtains exactly one effective Clock and
the Reset coverage required by its exact resource contract through the slot
relation. `loom.fabric 4.1` admits no implicit resetless stateful owner. A
backend cannot supply a default Reset contract or infer one from Clock
membership.

Where a software contract permits several legal behaviors, such as weak
compare-exchange spurious failure, the Fabric domain declares only the
supported behavior envelope. The exact Evaluation model identifies the
deterministic provider policy used for one execution. A model that cannot
complete a reachable choice is unsupported; Fabric and Mapping must not fill
the gap with an implicit policy.

One domain may cover several services, caches, and AccCores. A composite fence
provider may use several internal barriers and an all-of join only when that
behavior is an explicit domain-owned use pattern. Mapping cannot synthesize a
hidden multi-domain fence. The op does not use an open dictionary. Runtime
`ProtectionDomain` remains an invocation fact. Membership alone never implies
a transport crossing; every crossing remains an explicit endpoint, resource,
or pattern.

`fabric.system.connection` is one closed union. Its canonical variant ordinals
are `Transport = 0` and `MemoryService = 1`. A Transport connection is directed
and one-to-one from one transport output to one transport input. A
MemoryService connection is directed and one-to-one from one Memory-plane
manager endpoint to one Memory-plane subordinate endpoint. A manager endpoint
has at most one outgoing MemoryService connection, and a subordinate endpoint
has at most one incoming MemoryService connection. The two variants never
coerce, compare, or connect across planes.

Fanout, fan-in, multicast, arbitration, buffering, conversion, and protocol
crossing must be represented by explicit resources, transfer patterns, or
service transforms with the exact applicable contract. A transport
`spatial_attachment` is one-to-one between its Module and occurrence faces. A
memory attachment additionally binds that occurrence face to exactly one
System service endpoint. Neither variant has hidden behavior, and Mapping
cannot replace or select the bound endpoint.

The explicit memory-service closure rooted at a bound System service endpoint
uses only two Fabric-owned transitions. From a manager endpoint it may follow
the exact MemoryService connection to its subordinate destination without
changing the request. From the exact ordered subordinate input sequence of a
service transform it may apply that transform and continue from the transform's
ordered manager output sequence. A subordinate endpoint owned by a memory
service exposes that service's regions and terminates the branch. A transform
with several outputs creates several ordered branches; the closure does not
merge them or choose one implicitly. Only finite simple paths are selectable;
a repeated connection or transform is not a legal service-target path.

Every System operation-service reference in a connection, transform, or domain
resolves through a `SystemServiceEndpointRef` at ordinal zero.
Host cores, AccCores, memory services, transforms, and external boundaries are
not accepted as endpoint substitutes. A service transform owns only its typed
transformation relation; any physical input or output port it exposes is an
explicit service-endpoint entity owned by that transform.

Clock/reset validation walks the complete root-owned point-connection and
spatial-attachment ranges and every imported Module's complete local
connection and slot-assignment ranges. Every endpoint owner must resolve
through its exact inventory-owner or occurrence-qualified projection. The two
faces of a spatial attachment resolve to the same occurrence slot because an
attachment has no crossing behavior. Each ordinary point or Module-local
connection must likewise join equal effective face domains. A crossing
carrier's ingress and egress faces obtain their effective domains from its
contract, so a legal cross-domain path is represented by explicit same-domain
legs into and out of that carrier. Omitting a relation from a consumer is
impossible because consumers cannot supply the relation.

### Interconnect Implementation Sibling

`fabric.interconnect_implementation` is a separate Fabric-family root with one
exact `fabric.system` Artifact reference, one exact versioned protocol-schema
identity, one canonical protocol-specific implementation body, and one
declarative refinement region. The protocol schema, not an open dictionary,
owns every concrete endpoint, bundle, channel, field, packet, flit, queue, and
state type in the implementation body. Adding a new protocol requires a typed
schema version; a protocol name string cannot admit arbitrary payload.

This typed operation is the only legal
`InterconnectImplementation` root payload. A `fabric.module`, `fabric.system`,
generic region, or opaque byte payload labeled with root-kind ordinal 2 is
invalid. A build in which this operation and its canonical owner provider are
not registered must fail closed as
`Unsupported(FabricRootProviderUnavailable)`; it cannot report a successful
module-payload finalization. This provider-availability failure does not alter
the stable root-kind ordinal or permit the reserved-unavailable
`ImplementationInput` dependency role.

In `loom.fabric 4.x`, the protocol-schema identity is a closed root-local schema
tag interpreted by the typed interconnect implementation body. It is not an
external Artifact reference and does not authorize a generic implementation
dependency. The root has exactly one direct `RefinedSystem` dependency and no
`ImplementationInput` dependency. All protocol endpoint, resource, transfer,
and configuration references are local to that canonical implementation body.
An external protocol or IP artifact may be admitted only after a later Fabric
schema version defines its exact owner, root kind, local-reference codec, and
dependency-use contract as specified by `docs/spec-fabric-artifact.md`.

The refinement region contains only `fabric.interconnect.refinement` records.
Each record is one closed typed variant:

```text
EndpointRefinement(
  FabricTransportEndpointRef,
  ProtocolEndpointRef)

ResourceStateRefinement(
  FabricResourceStateRef,
  canonical non-empty set<ProtocolResourceRef>)

TransferPatternRefinement(
  FabricTransferPatternRef,
  canonical non-empty ordered sequence<ProtocolTransferRef>)

ConfigurationRefinement(
  FabricSemanticConfigFieldRef,
  ProtocolConfigurationFieldRef)
```

The complete relation must cover every architecture endpoint, state, selected
pattern behavior, and semantic configuration field required by the
implementation. Refinement proves the architecture contract; it neither edits
`fabric.system` nor adds Mapping choices. Protocol-specific implementation
schemas may be developed independently, but they cannot change these four
relation variants or the architecture root without a Fabric schema revision.

SystemMapping-selected routes are checked for combinational handshake closure
against the root-complete architecture relation before configuration or
deployment. The interconnect implementation must realize that already-valid
selection and may independently re-run the same derived check after
refinement; it cannot remove an active arc, enable an unselected route, or
insert an undeclared stateful break to repair a cycle.

### Canonicalization And Ownership

Authoring names, operation order, builder insertion order, source locations,
comments, and visualization metadata are non-semantic. The complete
failure-atomic root finalization, direct dependency framing, canonical bytes,
and identity contract is owned by `docs/spec-fabric-artifact.md`. The
Fabric entity and structural-reference variants used by that process are
closed by `docs/spec-fabric-identity.md`; this document cannot add an
unregistered reference kind through a generic child or property record.

The C++ ADG Builder creates this typed model and invokes the same finalizer; it
does not own a parallel schema. Protocol implementations, concrete routers,
arbiters, cache controllers, and simulator models consume the declared roots
and refinements. Free-form kinds, protocol-as-capability, parameter bags,
placeholder records, and compatibility wrappers are invalid.

Before a `FabricSystemRootView` is returned or a root is published, the
canonical importer/finalizer rejects duplicate point connections, duplicate
hardware-domain references, duplicate domain members, conflicting same-kind
membership, duplicate crossing fields for one carrier, wrong-kind typed domain
refinements, and any connection or attachment hidden from the complete
relation. It rejects a memory spatial attachment with a missing, foreign,
same-role, or wrong-plane System service endpoint and a transport spatial
attachment carrying any System service endpoint. It also rejects a
service-leg attachment whose memory endpoint is neither a System service
endpoint nor the occurrence member of one exact memory spatial attachment, or
whose kind is absent from that endpoint case's exact System capability
authority. Foreign endpoints, out-of-range leg ordinals, direction or
payload-domain incompatibility, incomplete System-endpoint or
occurrence-endpoint leg coverage, and every `MessageTransfer` attachment are
invalid. These checks are root validation, not optional consumer policy.
Compatibility between a selected workload member and the bound System
endpoint's capability domain is SystemMapping base verification and System PnR
domain construction; it is not a workload-independent Fabric root fact.

## Validation Anchors

Anchor-level validation should cover:

* exact module-to-AccCore attachment coverage and typed continuity, including
  one required memory-plane, complementary-role System service endpoint on
  every memory attachment, no such endpoint on a transport attachment, and
  rejection of missing, foreign, same-role, or wrong-plane bindings;
* one derived InstructionCore context whose atomic Fabric-owned execution use
  pattern, initial state, capacity, requester order, and exact grant contract
  reject a Mapping-defined scheduler or split claim;
* exact InstructionCore architecture fingerprinting, including compatibility
  across microarchitectural changes and incompatibility after an ISA change;
* arbitrary directed Transport Architecture routing with a shared bottleneck;
* one-ingress multicast and competing single-ingress arbitration patterns;
* all six Canonical Service kinds, exact leg order, actor-contract ownership,
  exact dynamic-mask leg derivation, and rejection of implicit plain, atomic,
  volatile, or coherence semantics;
* canonical service-leg carrier attachment order and duplicate normalization,
  complete rows on both members of a memory spatial-attachment pair, reuse of
  one System-endpoint row across several occurrence endpoints, multiple
  carrier alternatives, one carrier reused by several kinds or legs, and
  rejection of unbound occurrence endpoints, missing or foreign endpoints,
  unsupported kinds, out-of-range legs, role-direction mismatch, incompatible
  payload domains, incomplete pair-member coverage, and every
  `MessageTransfer` row;
* structural confirmation that service-leg carrier attachments contain no
  actor, payload, width, protocol, capability-domain, capability-ordinal, or
  independent identity field;
* one multi-participant MemoryConsistency domain, exact atomic and fence
  domain closure, and rejection when no one compatible domain covers a fence;
* one `CoherentMemory` transform whose explicit region correspondence permits
  coherent overlap, exact region-relative address translation, output closure,
  and input-domain alternatives, plus rejection of duplicate region members,
  unequal extents, or overlap inferred from domain membership;
* one non-trapping at-most-once MMIO region plus rejection of a trapping or
  provider-replayed SpatialCore binding;
* closed `fabric.system` child operations, Artifact-global `EntityId`
  assignment, owner-local port/leg/pattern ordinals, and rejection of generic
  node or property records;
* refinement rejection when an implementation hides sharing or weakens a
  guarantee;
* complete typed Endpoint, ResourceState, TransferPattern, and Configuration
  refinement plus deterministic lowering from architecture and Mapping to
  implementation;
* one explicit asynchronous clock crossing and rejection of a hidden direct
  crossing or invalid reset-to-clock relation;
* one imported Module whose complete Clock/Reset slots bind exactly once, plus
  rejection of a missing, duplicate, wrong-kind, or foreign slot binding;
* two AccCores importing the same Module with independently qualified internal
  state and equal or different concrete domains, with no aliasing of either
  occurrence;
* rejection of `AccCoreOccurrenceRef` or bare `SpatialCoreOccurrenceRef` as a
  Clock/Reset inheritance shortcut;
* the omitted-bypass counterexample: a cross-domain point connection present
  in the root is rejected even though no consumer supplies a connection list;
* duplicate crossing contracts for one carrier, duplicate Clock or Reset
  declarations, duplicate reset membership, and conflicting same-kind domain
  membership are rejected;
* owner-relative direct members and occurrence-qualified SpatialCore slots are
  validated by their full typed reference and are never skipped through an
  entity-ID projection;
* an InstructionCore-only SystemMapping with no imported SpatialMapping; and
* Deployment closure including imported Mapping dependencies and a programmable
  transport unit, plus workload-independent Gem5 Simulation Binding reuse and
  rejection of an InstructionCore model incompatible with the exact
  Architectural Contract, exact Microarchitectural Realization, or Compiler
  Target Binding.

Tests should not freeze protocol packet layout, gem5 internal queue state,
builder container shape, authoring order, or visualization metadata.

A graph-visible fault protocol is explicitly deferred. It may be introduced
only after Canonical Dataflow owns a typed fault value, propagation, recovery,
and completion contract. Fabric does not retain a dormant fault variant in the
meantime.
