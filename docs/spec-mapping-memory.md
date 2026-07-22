# Mapping Memory Boundary

## Purpose

This document assigns memory truth across the Canonical Dataflow Program,
Fabric, TechMapping, SpatialMapping, SystemMapping, Deployment, and runtime.
The central invariant is that a software memory space, a physical memory
service, and an interface used to reach that service are separate objects.

## Canonical Objects And Relations

The Canonical Dataflow Program owns `LogicalMemoryRoot` and view identity,
logical intervals, layout, aliasing, lifetime, and the load/store ordering
network. Fabric owns `PhysicalMemoryService` and region identity plus typed
manager/requester and subordinate/provider endpoints.

For each canonical load or store, consumers mechanically derive the
nonpersistent `CanonicalMemoryAccessView` defined by
`docs/spec-dataflow-vectorization.md`. Exact access form, memory-element type,
access lane shape, and mask semantics remain owned by Dataflow. Fabric owns the
parameterized access domains and operation use patterns of its physical memory
ports. Mapping selects and records only the non-derived correspondence between
those authorities.

The required semantic relations are:

```text
MemoryBinding
  LogicalMemoryView -> PhysicalMemoryServiceRegion

MemoryEngineBinding + AccessEntry + MemoryBinding
  software memory actor
    -> operation placement + LocalMemoryService or manager dispatch target
    -> physical memory service region

ExposureEntry + MemoryBinding
  software memory output -> subordinate/provider terminal + dispatch target
                         -> physical memory service region
```

These relations do not require three parallel top-level record families.
Their persistent owners are the existing Mapping records described below.

The relation is sparse and many-to-many. Several logical roots may bind to
one physical service. One endpoint may carry several bindings. One binding
may be reachable through several manager or subordinate endpoints. These
cases do not create duplicate storage identities.

A logical root spans several independent physical services only through
explicit disjoint partitioning or through a Fabric-declared replication,
coherence, sharding, striping, or service-transform capability. Overlapping
rows must not imply replication or coherence by convention.

## Canonical Memory Order

The Canonical Dataflow Program is the only owner of logical memory ordering.
Required happens-before relations are typed event edges in its memory-order
network. Mapping must not duplicate them as `MemoryOrder` records, infer them
from textual order, or replace them with a physical schedule.

Physical contention may add stalls but may not weaken RAW, WAR, WAW, or other
explicit causal relations. Final verification proves that the selected ports,
services, routes, exact Fabric or Mapping-selected hardware-refinement grant
behavior, and `ResourceUse` preserve every logical obligation. Mapping,
runtime, and simulation do not supply missing cycle-visible arbitration.

Plain, atomic, volatile, RMW, compare-exchange, and fence software semantics
are owned by `docs/spec-dataflow-memory-consistency.md`. This Mapping
specification currently defines only the plain and vector-memory realization
surface. Atomic, volatile, fence, consistency-domain, and coherence
realization remains inadmissible until the corresponding Fabric and Mapping
contracts are added; Mapping must not reinterpret these actors as plain
load/store.

## TechMapping Memory Realization

TechMapping owns each selected Memory Realization:

* exact load/store actor coverage and logical-root association;
* selected Fabric memory semantic capability;
* actor-to-operation correspondence;
* exact graph and implementation-boundary correspondence;
* exact parameterized access, alignment, mask, narrow-access, and declared
  use-pattern compatibility; and
* exact selected internal-edge witnesses.

Compatibility is not total payload-width equality. It proves the exact
`element | contiguous | indexed` access form, memory-element width, access
lane-shape projection, flattened lane count, complete address/data/mask
capacities, dynamic-mask support, alignment, and subword-write contract.
Equal-width accesses with
different element or lane geometry remain distinct. A declared Fabric use
pattern may realize one actor firing with several internal service
transactions, but TechMapping cannot invent that decomposition.

An internal-edge witness identifies one canonical software source and sink and
one Fabric-allowed internal connection. Actors sharing a Memory Realization do
not make all edges between them internal. Only witnessed edges are absorbed;
all remaining edges become ordinary external routing obligations.

A selected load still has one retirement event that publishes `data + done`
atomically across its internal and external obligations. Mapping may select
destinations but cannot split that packet or weaken its backpressure. Store
retirement remains one `done` event.

## SpatialMapping Records

SpatialMapping has exactly five persistent record families:

```text
ComputeBinding
MemoryEngineBinding
MemoryBinding
RouteTree
ResourceUse
```

Memory does not add a generic binding bag, a string-key extension record, or a
parallel configured-table authority.

### MemoryBinding

One `MemoryBinding` is the atomic relation:

```text
one LogicalMemoryInterval -> one PhysicalMemoryServiceRegion
```

It stores a typed logical memory or view reference, logical interval, typed
physical service reference, physical region, and only a selected Fabric-owned
address transform that cannot be derived from the endpoints. It receives a
Mapping-local identity because several rows may bind the same logical root and
Access or Exposure children must reference an exact row.

Whole-root placement is the degenerate case. Disjoint rows express
partitioning. Multiple roots may independently bind to one service. Multiple
manager or subordinate endpoints that reach the same service do not require
multiple `MemoryBinding` rows unless the logical interval or physical region
actually differs.

Each `MemoryBinding` owns its `ExposureEntry` children. An Exposure Entry binds
one software memory-output obligation to one selected subordinate/provider
terminal, its existing Memory Binding, and one closed typed
`LocalMemoryServiceRef | ManagerEndpointRef` dispatch target. The service path
belongs to route or service realization; provider-decode rows are derived
configuration.

### MemoryEngineBinding And AccessEntry

Each `MemoryEngineBinding` is keyed by one TechMapping Memory Realization and
selects one concrete `fabric.mem` Operation Engine. It owns exactly one
`AccessEntry` for every covered canonical load or store actor.

Operation placement uses a closed typed reference:

```text
MemoryOperationPlacementRef =
    Spatial  { PhysicalMemoryOperationPortRef }
  | Temporal { PhysicalMemoryOperationPortRef, OperationContextOrdinal }
```

An Access Entry stores the actor reference, operation placement, one
`MemoryBinding` reference, one closed typed
`LocalMemoryServiceRef | ManagerEndpointRef` dispatch target, and only
selected non-derived typed `sink <- source` choices. This is where a selected
`load.data -> store.data`, `done -> ctrl`, or other Fabric-allowed
memory-internal dependency is recorded.

The Access Entry does not copy access form, memory-element type, lane shape,
lane count, mask mode, endpoint widths, or transaction decomposition. Those
values are derived from the exact Dataflow actor, selected Fabric operation
port, and selected Fabric use pattern. A dynamic mask is an ordinary actor
operand whose external endpoint correspondence or selected internal source is
verified like any other required operand.

The AccessEntry and ExposureEntry target fields collectively are the only
persistent owner of selected `C_dispatch`; there is no parallel dispatch
relation record. Fabric alone owns eligible `H_dispatch`, and the Mapping
verifier proves `C_dispatch` is its subset. The selected target may be the
occurrence's Local Memory Service or a declared manager endpoint/path. Those
are alternatives in one typed service model, not different request protocols.
Runtime ABI owns the single
`SpatialServiceRequest`/`SpatialServiceResponse` boundary used by either
target.

An Access Entry does not own a Physical Tag. The actual tag value is a typed
sharing assignment on the real tagged writer or ingress `ResourceUse`, and is
present only where may-overlap incompatible interpretations in a local Fabric
match domain require distinction. It is not global firing, iteration,
invocation, logical-token, or memory identity. Memory continuity and selected
context derive the operation-row match value.

### RouteTree And ResourceUse

Token edges not absorbed by the Memory Realization use the ordinary flat
`RouteTree` model. A memory-local traversal is not free unless it is an exact
selected internal-edge witness. The same route-tree rules apply to address,
data, mask, control, and completion transport. Each residual vector address,
data, or mask value is one complete logical token and must fit every selected
endpoint and traversal. Mapping cannot split it across unrelated endpoints or
use Physical Tags as lane identifiers.

`ResourceUse` owns event-relative activation, typed use-pattern parameters,
capacity occupancy, and sharing assignments over the already selected engine,
service, route, port, queue, bank, or context. It does not copy Fabric capacity,
duration, latency, or use vectors.

For a memory actor, the selected Fabric use pattern owns any internal lane or
beat transactions and their resource claims. Mapping supplies only its typed
parameters and reservations. The operation endpoint payload width and backing
service beat width therefore remain separate Fabric facts rather than route or
AccessEntry fields.

Configured `memory_operation_table` rows, dispatch selectors, provider decode,
and response tracking are deterministic semantic projections of the five
record families plus the exact Dataflow, TechMapping, and Fabric inputs.
`ConfigurationABI` alone maps those fields to physical bits, addresses, and
programming operations; `HardwareConfigurationImage` artifacts carry the
encoded result.

## SystemMapping Memory And Service

SystemMapping has one non-empty root-thread-launch coverage set. Its exact
imported SpatialMapping set is the finite, deduplicated range of normalized
`B_graph` over all reachable static graph launches and their legal may-domains.
There is no separately editable selected-set field and no fixed cardinality.
An InstructionCore-only closure has an empty imported SpatialMapping set
without a placeholder artifact.

The only persistent SystemMapping families are:

```text
ExecutionBinding
ServiceRealization
ResourceUse
```

### ExecutionBinding

`ExecutionBinding` contains two typed variants:

```text
ThreadExecutionBinding
  RootThreadLaunchRef -> BindingRelation<AccCoreOccurrenceRef>

GraphExecutionBinding
  (ThreadExecutionBindingKey, StaticGraphLaunchRef)
    -> BindingRelation<SpatialMappingImportRef>
```

Relations are total on the Canonical Dataflow Program's may-domain and use
the closed `PresburgerPartition` or `StableKeyLookup` algebra. Runtime only
evaluates these immutable relations for concrete coordinates and parameters.
It never chooses another AccCore or SpatialMapping.

### ServiceRealization

SystemMapping has one `ServiceRealization` family keyed by one of two typed
software obligations:

```text
SystemServiceObligationKey =
    TransferObligationFamilyKey
  | OperationServiceObligationFamilyKey
```

A Transfer obligation represents one exact producer-terminal family and its
canonical non-empty sink set. Multicast remains one obligation; Mapping may
not split it into unrelated per-sink routes. An Operation Service obligation
represents one logical service root or view and the typed operation set
required by the Canonical Service Schema. Memory access and exposure share
this owner so they cannot select contradictory services.

Each `ServiceRealization` contains canonical owner-local `ServicePlan` values
and a total plan-selection relation. A plan contains:

* atomic logical-service-interval to Fabric-service-region bindings;
* one `TransferLegRealization` per leg mechanically required by the Canonical
  Service Schema; and
* mapping-visible physical refinement assignments.

The Canonical Service Schema, not Mapping or a protocol name, defines memory
request, write data, read data, response, completion, and any other operation
legs. A leg is realized by the ordinary flat system Route Tree over typed
Fabric endpoints and traversals. AXI, TileLink, CXL, packet, flit, header,
virtual-channel encoding, and protocol state belong to Interconnect
Implementation.

### Cross-Layer Service Chain

The only memory-service ownership chain is:

```text
Canonical logical service
  -> SpatialMapping local service or explicit boundary proxy
  -> Fabric module-to-AccCore attachment
  -> SystemMapping ServiceRealization
  -> system provider service or explicit external provider
```

A Spatial-local service that completes the operation produces no system
obligation. A boundary proxy creates an operation-service obligation, and
SystemMapping uniquely selects the system route, provider region, and address
transform. Partial closure may end only at an explicit external provider; it
must not pretend that a proxy is final storage.

Whether service is local or crosses a manager endpoint does not change the
runtime request schema. Both use the Runtime ABI's typed
`SpatialServiceRequest` and `SpatialServiceResponse`; adapters translate that
one boundary to a local model, standalone external-service model, RTL harness,
or manager Bridge without reinterpreting the Mapping binding.

System `ResourceUse` for a selected plan element uses
`ServicePlanElementRef = (ServiceRealizationKey, canonical plan ordinal,
typed element key)`. Applicability is derived from `plan_selection`; the use
does not copy its predicate, target, or selected plan. System `ResourceUse`
owns occupancy, event-relative activation, typed capacity demand, Physical
Tags, and sharing assignments for selected system services and transport.
Spatial `ResourceUse` remains inside the imported immutable
SpatialMapping and is occurrence-qualified by the derived cross-layer closure;
SystemMapping does not copy it.

## Runtime And Deployment Boundary

`docs/spec-configuration-deployment.md` is the sole owner of runtime-image
child membership, identity status, schemas, canonical child keys and ordering,
and presence conditions. Runtime consumes those owner-finalized children to
evaluate immutable memory and service bindings, supplies invocation-specific
allocations and authorization, and atomically admits preselected activation
sets. It may wait or apply backpressure, but it may not remap execution,
service, route, tag, context, or configuration.

This memory contract contributes the exact Mapping dependencies, memory and
service bindings, and memory-image obligations that Deployment must close. It
does not own the package list or executable closure. The imported
SpatialMapping set remains only one source of those obligations, and package
indices cannot become a second memory, service, or Mapping authority.

## Validation Anchors

Anchor-level tests should cover:

* sparse many-to-many binding with several logical roots sharing one service,
  plus rejection of unsupported overlap;
* one Access Entry per covered actor, an exact internal edge, and atomic load
  `data + done` retirement;
* element, contiguous, indexed, masked, and unmasked access compatibility,
  including rejection of an equal-width but semantically incompatible port;
* routing of complete vector address, data, and mask tokens without lane Tags
  or implicit endpoint splitting;
* one declared multi-transaction Fabric use pattern with one logical actor
  retirement and rejection of Mapping-invented decomposition;
* local Physical Tag ownership by the real writer/ingress `ResourceUse`;
* local-service and manager-endpoint AccessEntry targets using the same typed
  Spatial Service request/response boundary;
* AccessEntry and ExposureEntry dispatch ownership plus rejection of
  `C_dispatch` outside Fabric-owned `H_dispatch`;
* one system-memory request/response plan and one unsplit multicast; and
* Deployment-owner closure of every selected memory and service dependency,
  with package-index authority and runtime remapping rejected.

Tests should not freeze configured table layout, protocol encoding, printer
format, or runtime cache structure.
