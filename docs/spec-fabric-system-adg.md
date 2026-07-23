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

`docs/spec-fabric-identity.md` is the sole owner of the Mapping-visible
Fabric entity and structural-reference catalog. This document owns the
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
* typed service endpoints and service transforms;
* one Transport Architecture owned by the `fabric.system` root;
* zero or more independent Interconnect Implementation objects refining that
  architecture;
* external boundaries;
* address, clock, reset, power, and memory-consistency domains; and
* optional non-semantic visualization metadata.

These are typed Fabric concepts. A generic node kind, open dictionary, protocol
string, or placeholder record is not an alternative system schema.

## Canonical Service Schema

The Canonical Service Schema is the sole owner of logical operation semantics
shared by software obligations, Fabric capabilities, Mapping, simulation, and
implementation refinement. Version 2.0 has exactly six parameterized kinds:

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
DSL. Version 1.0's three-kind plain-only memory model is superseded because it
cannot express these actor contracts without duplicated operation kinds.

## AccCore And SpatialCore Attachment

The architecture uses:

```text
AccCore = InstructionCore + SpatialCore
```

An AccCore is one physical occurrence and contains exactly one InstructionCore
plus exactly one SpatialCore attachment. The InstructionCore description has an
Architectural Contract for binary compatibility and a Microarchitectural
Realization for execution structure, timing, and capacity. Simulator model
names and compiler target spellings are bindings over that description, not
Fabric hardware facts.

For a selected AccCore, its exact InstructionCore Architectural Contract
mechanically selects and validates the Compiler Target Binding owned by
`docs/spec-runtime-abi.md` section `Compiler Target And Binary Compatibility`.
Neither that binding nor its target-specific binary enters `fabric.system`
identity or SystemMapping identity.

Its SpatialCore references one exact `fabric.module` template. Multiple
AccCores may reference the same template while remaining distinct physical
resources.

Because the InstructionCore cardinality is one, its Mapping reference is
derived rather than allocated. `docs/spec-fabric-identity.md` owns the
persistent reference framing; this specification owns the one-per-AccCore
cardinality:

```text
InstructionCoreContextRef = (AccCoreOccurrenceRef, 0)
```

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

Every fully elaborated occurrence has a typed one-to-one attachment between
each module boundary endpoint and the corresponding AccCore-local SpatialCore
endpoint. An attachment stores only the two structural references. Direction,
type, service capability, and role are derived from the endpoints.

An attachment is not a route and cannot hide conversion, buffering,
arbitration, clock-domain crossing, or any other stateful behavior. Such
behavior requires an explicit Fabric resource or transfer pattern. Every
module endpoint attaches exactly once.

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
      NonTrapping | ExplicitFaultProtocol
      AtMostOnceLogicalOperation provider observation
    }
```

The region kind is a physical service fact, not a Dataflow operation name or
ordering mode. A SpatialCore mapping may use an `Mmio` region only when its
declared range is non-trapping for the selected accesses; an explicit fault
protocol requires a matching graph fault contract, which the current
Canonical Dataflow Program does not provide. A capability that admits a
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
Mapping. The first shared grant-policy atoms are:

```text
fixed_priority(exact requester order)
round_robin(exact requester order, reset cursor, advance on successful grant)
```

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

A workload-independent Gem5 Simulation Binding maps exact Fabric and
Interconnect Implementation objects to gem5 models, SimObjects, parameters,
and the Bridge ABI. It is a simulator binding, not hardware truth. Every modeled
InstructionCore must validate all three authorities: the exact InstructionCore
Architectural Contract; the exact InstructionCore Microarchitectural
Realization, including execution structure, timing, capacity, and
mapping-visible resources; and the compatible Compiler Target Binding used by
its target-specific binary. The system-simulator descriptor owns role-labeled
subject slots `deployment` and `system_model`; an ordinary `EvaluationRequest`
binds their exact subjects. Exact workload and runtime data use
`SimulationWorkload` and `SimulationRuntimeInput` references, and remaining
simulator parameters use `ResolvedModelBinding`. There is no separate
system-simulation request family.

These are compatibility relations, not a persistent binding schema. The exact
`Gem5SimulationBinding` root remains downstream open work as specified by
`docs/spec-runtime-abi.md` section `Gem5 Simulation Binding`.

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
complete service-member set.

This document's Canonical Service Schema remains the sole owner of each
member's local request, data, response, completion, direction, payload, and
ordinal semantics. Mapping combines the obligation key, exact
`ServiceMemberRef`, and schema-local leg ordinal as one
`CanonicalServiceLegKey`, then realizes each required leg explicitly over the
Transport Architecture. Protocol channels are derived only by Interconnect
refinement.

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
  exact FabricModuleTemplateRef spatial_core

fabric.system.memory_service
  EntityId
  canonical ServiceRegion records
  CanonicalServiceCapability records
  Fabric-owned ResourceState, UsePattern, capacity, timing, and grant contracts

fabric.system.service_endpoint
  EntityId
  exact owner reference
  CanonicalServiceCapability records

fabric.system.service_transform
  EntityId
  exact input and output endpoint references
  closed ServiceTransformContract

fabric.system.transport_resource
  EntityId
  typed ports
  ResourceState, capacity, timing, progress, and grant contracts

fabric.system.transfer_pattern
  SystemTransportResourceRef owner
  owner-local pattern ordinal
  one ingress port
  canonical non-empty egress-port sequence
  atomic typed resource-use vector
  timing, ordering, progress, eligibility, and semantic controls

fabric.system.connection
  exact source endpoint or output-port reference
  exact destination endpoint or input-port reference

fabric.system.spatial_attachment
  exact FabricModuleBoundaryEndpointRef
  exact AccCore-local SpatialCore endpoint reference

fabric.system.hardware_domain
  EntityId
  closed domain kind
  canonical member references
  kind-owned typed contract

fabric.system.external_boundary
  EntityId
  canonical owned endpoint references
  external service and transfer contract
```

`CanonicalServiceCapability` binds one exact Canonical Service kind, one
operation-relative `Initiate | Serve` role, and a closed accepted access or
actor-contract domain, plus address/range, alignment, issue/accept rate,
outstanding capacity, consistency-domain reference, visibility, and progress.
For an addressed memory kind, the domain is tested against the exact derived
`CanonicalMemoryAccessView`; it does not copy actor-owned fields. The
consistency-domain reference is present exactly when required by the selected
service kind or accepted actor contract. Fields not owned by that service kind
are absent rather than populated with defaults.

`ServiceTransformContract` is a closed sum. Version 2.0 admits
`AddressOffset`, `AddressMaskXor`, `StaticInterleave`, and `CoherentMemory`;
each
variant owns its exact typed parameters and total input-to-output relation:

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
  canonical non-empty region correspondence
}
  every input-region occurrence named by the correspondence is a physical
  copy or proxy of its output service region under the exact domain contract
```

All non-address arguments and results pass unchanged, and the output endpoint
capabilities must accept the transformed range and exact service signature.
An identity transform is represented by a direct connection and has no
operation. `CoherentMemory` is the only authority that permits overlapping
physical service regions to represent one coherent service identity. Domain
membership alone never implies storage identity, replication, or coherence.
Cache behavior beyond this architecture-level coherence relation, arbitrary
hashing, programmable callbacks, and opaque custom transforms require a future
closed variant rather than an open extension bag.

`hardware_domain` version 2.0 uses the closed kinds `Clock`, `Reset`, `Power`,
`Address`, and `MemoryConsistency`. Their contracts are exactly:

```text
Clock { period_fs: positive uint64, phase_fs: uint64 where phase_fs < period_fs }
Reset { polarity: ActiveHigh | ActiveLow,
        assertion: Synchronous | Asynchronous,
        deassertion: Synchronous | Asynchronous,
        initial_state: Asserted | Deasserted }
Power { nominal_voltage_uv: positive uint64 }
Address { address_width: positive uint32,
          canonical disjoint half-open unsigned ranges }
MemoryConsistency {
  canonical non-empty participant service or provider references
  exact visibility, linearization, completion, and progress guarantees
  closed ResourceState and atomic UsePattern domains
}
```

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

Connections are directed and one-to-one from one output to one input. Fanout,
fan-in, multicast, arbitration, buffering, conversion, and protocol crossing
must be represented by explicit resources and transfer patterns. A
`spatial_attachment` is likewise one-to-one and has no hidden behavior.

### Interconnect Implementation Sibling

`fabric.interconnect_implementation` is a separate Fabric-family root with one
exact `fabric.system` Artifact reference, one exact versioned protocol-schema
identity, one canonical protocol-specific implementation body, and one
declarative refinement region. The protocol schema, not an open dictionary,
owns every concrete endpoint, bundle, channel, field, packet, flit, queue, and
state type in the implementation body. Adding a new protocol requires a typed
schema version; a protocol name string cannot admit arbitrary payload.

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

### Canonicalization And Ownership

Authoring names, operation order, builder insertion order, source locations,
comments, and visualization metadata are non-semantic. Finalization resolves
all references, validates complete attachment and endpoint ownership,
canonically labels independently meaningful entities, assigns consecutive
`EntityId` values, sorts structural records by complete semantic key, emits
canonical Fabric bytes, and applies the Common SHA-256 v1 contract. Equal
hardware descriptions therefore have equal canonical bytes and identity.
The Mapping-visible entity and structural-reference variants used by this
process are closed by `docs/spec-fabric-identity.md`; this document cannot add
an unregistered reference kind through a generic child or property record.

The C++ ADG Builder creates this typed model and invokes the same finalizer; it
does not own a parallel schema. Protocol implementations, concrete routers,
arbiters, cache controllers, and simulator models consume the declared roots
and refinements. Free-form kinds, protocol-as-capability, parameter bags,
placeholder records, and compatibility wrappers are invalid.

## Validation Anchors

Anchor-level validation should cover:

* exact module-to-AccCore attachment coverage and typed continuity;
* one derived InstructionCore context whose atomic Fabric-owned execution use
  pattern, initial state, capacity, requester order, and exact grant contract
  reject a Mapping-defined scheduler or split claim;
* arbitrary directed Transport Architecture routing with a shared bottleneck;
* one-ingress multicast and competing single-ingress arbitration patterns;
* all six Canonical Service kinds, exact leg order, actor-contract ownership,
  exact dynamic-mask leg derivation, and rejection of implicit plain, atomic,
  volatile, or coherence semantics;
* one multi-participant MemoryConsistency domain, exact atomic and fence
  domain closure, and rejection when no one compatible domain covers a fence;
* one `CoherentMemory` transform whose explicit region correspondence permits
  coherent overlap, plus rejection of overlap inferred from domain membership;
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
* an InstructionCore-only SystemMapping with no imported SpatialMapping; and
* Deployment closure including imported Mapping dependencies and a programmable
  transport unit, plus workload-independent Gem5 Simulation Binding reuse and
  rejection of an InstructionCore model incompatible with the exact
  Architectural Contract, exact Microarchitectural Realization, or Compiler
  Target Binding.

Tests should not freeze protocol packet layout, gem5 internal queue state,
builder container shape, authoring order, or visualization metadata.
