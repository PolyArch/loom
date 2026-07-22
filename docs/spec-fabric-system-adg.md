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
* address, coherence, consistency, clock, reset, power, and protection
  domains; and
* optional non-semantic visualization metadata.

These are typed Fabric concepts. A generic node kind, open dictionary, protocol
string, or placeholder record is not an alternative system schema.

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

Its SpatialCore references one exact `fabric.module` template. Multiple
AccCores may reference the same template while remaining distinct physical
resources.

Because the InstructionCore cardinality is one, its Mapping reference is
derived rather than allocated:

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
Schema defines typed operations such as message transfer and memory read,
write, or atomic access. Each operation owns typed arguments and results,
logical effects, ordering and completion semantics, abstract transfer legs,
and visibility or coherence requirements.

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
and the Bridge ABI. The system-simulator descriptor owns role-labeled subject
slots `deployment` and `gem5_binding`; an ordinary `EvaluationRequest` binds
their exact artifacts. Exact workload and runtime data use
`SimulationWorkload` and `SimulationRuntimeInput` references, and remaining
simulator parameters use `ResolvedModelBinding`. There is no separate
system-simulation request family.

The generated gem5 projection is derived configuration. Handwritten simulator
configuration cannot become another topology, memory, route, timing, or
protocol authority.

## SystemMapping Boundary

SystemMapping binds an exact Canonical Dataflow Program and fully elaborated
Fabric system. Its only persistent coverage root is a canonical non-empty set
of root thread launches. The exact imported SpatialMapping set is the finite,
deduplicated value range of normalized graph execution bindings over all
reachable static graph launches and their legal may-domains. There is no
separate editable selected-set field and no fixed cardinality.

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

System service obligations use a closed typed key:

```text
SystemServiceObligationKey =
    TransferObligationFamilyKey
  | OperationServiceObligationFamilyKey
```

Transfer obligations cover an exact producer-terminal family and its complete
sink set, including multicast. Operation Service obligations cover a logical
service root or view and the operations required by the Canonical Service
Schema. Each Schema-derived request, data, response, or completion leg is
realized explicitly over the Transport Architecture. Protocol channels are
derived only by Interconnect refinement.

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

The verified cross-layer closure defines `ThreadDispatchImage`,
`SpatialLaunchImage`, and `AdmissionImage`. Deployment combines the applicable
images with the exact software, Fabric, complete SystemMapping, every direct
and transitive imported Mapping dependency, binaries, selected Hardware and
Interconnect Implementations and refinements, exact `ConfigurationABI`
artifacts, memory images, and platform bindings as a dependency graph. Its
`HardwareConfigurationImage` set covers every selected ABI Programming Unit,
including SpatialCore context banks and programmable transport or other
configuration units. Imported SpatialMappings are one dependency source, not
the definition of deployment closure. It does not write runtime tables or
implementation facts back into SystemMapping.

An InstructionCore-only Deployment omits `SpatialLaunchImage`, SpatialCore
configuration images, and other absent Spatial payload while retaining
complete SystemMapping, `ThreadDispatchImage`, and `AdmissionImage`. It still
includes any transport or other Programming Unit actually selected by that
SystemMapping.

## Domains And External Boundaries

Clock, reset, power, address, coherence, consistency, and protection domains
are explicit typed system facts. Domain crossings require explicit resources
and verified semantics. Hierarchy or visualization grouping does not imply
membership.

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

## Representation And Implementation Ownership

The ownership, connectivity, Mapping, runtime, and deployment semantics above
are closed. Exact Fabric operation names, assembly spelling, and typed
attribute layouts exist only where a dedicated Fabric operation specification
defines them. Protocol implementations, concrete routers, arbiters, cache
controllers, and simulator model coverage are implementation choices subject
to the declared refinement contract.

Unspecified spelling must not be filled with free-form kinds, protocol-as-capability,
open parameter dictionaries, placeholder records, or compatibility wrappers.

## Validation Anchors

Anchor-level validation should cover:

* exact module-to-AccCore attachment coverage and typed continuity;
* one derived InstructionCore context whose atomic Fabric-owned execution use
  pattern, initial state, capacity, requester order, and exact grant contract
  reject a Mapping-defined scheduler or split claim;
* arbitrary directed Transport Architecture routing with a shared bottleneck;
* one-ingress multicast and competing single-ingress arbitration patterns;
* Canonical Service Schema memory request and response legs;
* refinement rejection when an implementation hides sharing or weakens a
  guarantee;
* deterministic lowering from architecture and Mapping to implementation;
* an InstructionCore-only SystemMapping with no imported SpatialMapping; and
* Deployment closure including imported Mapping dependencies and a programmable
  transport unit, plus workload-independent Gem5 Simulation Binding reuse.

Tests should not freeze protocol packet layout, gem5 internal queue state,
builder container shape, or unconfirmed Fabric assembly spelling.
