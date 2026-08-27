# Mapping Memory Boundary

## Purpose

This document assigns memory truth across the Canonical Dataflow Program,
Fabric, TechMapping, SpatialMapping, SystemMapping, Deployment, and runtime.
The central invariant is that a software memory space, a physical memory
service, and an interface used to reach that service are separate objects.
`docs/spec-fabric-identity.md` owns the persistent Fabric-local references to
those services, interfaces, operation ports, and contexts; this document owns
only their selected Mapping relations.

## Canonical Objects And Relations

The Canonical Dataflow Program owns `LogicalMemoryRoot` and view identity,
logical intervals, layout, aliasing, lifetime, canonical memory actors, and
their ordering network. Fabric owns `PhysicalMemoryService` and region
identity, `MemoryConsistencyDomain`, and typed manager/requester and
subordinate/provider endpoints.

The persistent `LogicalMemoryRootRef` is imported from the exact
Dataflow-owned entity catalog. It identifies one static root role, not one
runtime allocation. A `CanonicalMemoryActorRef` is the exact Dataflow
`ActorRef` after memory-kind validation. A logical memory view is a
root-preserving typed structural reference
`(LogicalMemoryRootRef, canonical root-local view ordinal)` resolved by the
Dataflow importer. `LogicalMemoryRootOrViewRef`, `ContextualActorRef`, and
`MemoryExposureRef` are the closed forms owned by
`docs/spec-compiler-part-3-dfg.md`; Mapping does not assign another root,
actor, view, contextual-actor, or exposure ID.

Runtime may bind two imported root roles to one object through explicit alias
topology, and each fresh allocation occurrence combines its static root
reference with an invocation occurrence. These execution facts do not mutate
the static Mapping references or create Mapping-local memory identities.

Graph launch memory bindings are exact memref capability bindings. Mapping
consumes each Dataflow-owned `LogicalMemoryRootOrViewRef` through the same
`MemoryBinding` relation and never derives a pointer adaptation. A
`PointerAddressed` actor retains its pointer value edge while its independent
service capability participates in the ordinary memory binding; storage
placement and addressed memory actors remain the complete physical accounting
owners.

For each canonical addressed memory actor, consumers mechanically derive the
nonpersistent `CanonicalMemoryAccessView` defined by
`docs/spec-dataflow-vectorization.md`. Exact access form, memory-element type,
access lane shape, mask semantics, and actor contract remain owned by Dataflow.
Fence has no addressed-access view. Fabric owns the parameterized actor and
access domains, MemoryConsistency domains, and operation use patterns of its
physical memory ports. Mapping selects and records only the non-derived
correspondence between those authorities.

The required semantic relations are:

```text
MemoryBinding
  LogicalMemoryInterval
    -> LocalRegion(FabricMemoryServiceRegionRef, physical byte offset)
     | BoundaryProxy

MemoryEngineBinding + MemoryOperationEntry + optional MemoryBinding
  software memory actor or fence
    -> operation placement + LocalMemoryService or manager dispatch target
    -> local physical service region or Spatial service obligation

ExposureEntry + MemoryBinding
  MemoryExposureRef -> subordinate/provider terminal + dispatch target
                    -> local physical service region or Spatial service obligation
```

These relations do not require three parallel top-level record families.
Their persistent owners are the existing Mapping records described below.

The relation is sparse and many-to-many. Several logical roots may bind to
one physical service. One endpoint may carry several bindings. A boundary
proxy may be dispatched through a manager or subordinate endpoint without
turning that endpoint into storage. These cases do not create duplicate
storage identities.

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
are owned by `docs/spec-dataflow-memory-consistency.md`. Canonical Service
Canonical Service Schema 2.0 owns their shared operation-service shapes.
Mapping must not
reinterpret one actor contract as another or infer atomic, volatile, MMIO, or
coherence behavior from an unrelated physical capability.

Mapping owns no dynamic consistency state. Its selected operation target,
`MemoryBinding`, use pattern, and exact `MemoryConsistencyDomain` mechanically
determine whether execution remains in a Fabric-local provider or crosses the
typed Spatial Service boundary to an external provider. Modification order,
reads-from, synchronizes-with, sequentially-consistent order, visibility
frontiers, queue state, and provider timing are execution facts and must not be
cached or serialized as Mapping truth.

Compatibility also checks the domain's exact release visibility point, fixed
linearization and completion guarantees, progress variant, ResourceStates,
and atomic UsePatterns against the selected actor contract and complete route
and service closure. Mapping neither copies those fields nor substitutes a
local default when a domain omits or cannot provide a required guarantee.

## TechMapping Memory Realization

TechMapping owns each selected Memory Realization:

* exact read, write, RMW, compare-exchange, or fence actor coverage;
* one selected `FabricMemoryEngineTemplateRef`, plus exact template-relative
  operation-port and capability-alternative references;
* actor-to-operation correspondence;
* exact graph and implementation-boundary correspondence;
* exact parameterized actor-contract, access, alignment, mask, narrow-access,
  synchronization-scope, and declared use-pattern compatibility; and
* exact template-relative internal-edge witnesses.

Addressed compatibility is not total payload-width equality. It proves that
the selected alternative belongs to the selected template operation port,
then proves
the exact operation kind and actor contract, `element | contiguous | indexed`
access form, memory-element width, access-lane-shape projection, flattened
lane count, complete address/data/mask capacities, dynamic-mask support,
alignment, subword-write contract, and MemoryConsistency-domain requirement.
Equal-width accesses with different element or lane geometry remain distinct.
A derived `ElementAccessOnly | VectorAccessOnly | ElementAndVectorAccess`
label is never selected or persisted. A shared hybrid physical port and
separate element and vector ports have different inventories, ResourceState,
and capacity even when both derive aggregate element-plus-vector support.
A declared Fabric operation-port use pattern may derive one Direct child or
row-major active-lane children under the actor firing's single parent service
request, but TechMapping cannot invent or change that projection. The selected
service contract independently owns physical beat realization. Fence
compatibility proves its exact contract, mandatory Direct projection, and a compatible
single-domain operation capability without fabricating an address or memory
root.

An internal-edge witness identifies one canonical software source and sink and
one Fabric-allowed template-relative internal connection. Actors sharing a
Memory Realization do not make all edges between them internal. Only witnessed
edges are absorbed; all remaining edges become ordinary external routing
obligations.

A selected memory actor still has one retirement publication across all
internal and external obligations. Mapping may select destinations but cannot
split result and completion publication or weaken its backpressure. Fence
retirement remains one `done` event after its selected domain operation
completes.

TechMapping does not own an addressed actor's logical-memory association. An
`ActorRef` identifies one reusable graph definition, while the exact memory
root or view is obtained only by composing that actor's memory capability with
one `RootedGraphLaunchRef`. SpatialMapping owns that contextual association.

## SpatialMapping Records

`docs/spec-mapping-artifact.md` solely owns SpatialMapping's persistent record
families and their wire shapes. This specification defines the memory
validity relations over those records.

Memory does not add a generic binding bag, a string-key extension record, or a
parallel configured-table authority.

### MemoryBinding

One `MemoryBinding` is the atomic relation owned by
`docs/spec-mapping-artifact.md`: one owner-defined `LogicalMemoryInterval`
maps to one owner-defined `MemoryBindingTarget`. It stores a typed logical
memory or view reference, logical interval, and one
closed target. It receives a Mapping-local identity because several rows may
bind the same logical root and Access or Exposure children must reference an
exact row. The same identity names a BoundaryProxy; Mapping defines no
separate proxy entity.

A `LocalRegion` target stores one exact Fabric Local Memory Service region and
one physical byte offset. Its translated logical interval must be finite and
fully contained by that region. `Whole` derives its extent from the exact
Dataflow root or view and is legal locally only when that extent is statically
finite. `ByteRange` uses unsigned byte units, has positive size, and must be
contained by the exact logical root or view.

A `BoundaryProxy` target states that the interval is not closed by a local
service in the Spatial Mapping. It stores no Fabric service, service region,
endpoint, transform, provider, or system address. A dynamically unbounded
`Whole` interval must use this target unless exact pre-Mapping specialization
has made the bound finite. SystemMapping later derives the existing
operation-service obligation from the logical owner and interval and selects
the system provider region and transform.

Whole-root placement is the degenerate case. Disjoint rows express
partitioning. Multiple roots may independently bind to one service. Multiple
local endpoints that reach the same local service do not require multiple
`MemoryBinding` rows unless the logical interval or physical region actually
differs. Selecting a different manager path does not itself create another
BoundaryProxy binding because dispatch remains owned by the child entry.

Each `MemoryBinding` owns its `ExposureEntry` children. An Exposure Entry binds
one exact `MemoryExposureRef` to one selected subordinate/provider terminal,
its existing Memory Binding, and one closed typed
`LocalMemoryServiceRef | ManagerEndpointRef` dispatch target. The service path
belongs to route or service realization; provider-decode rows are derived
configuration. An exposure provides a capability boundary. It is not a
`ServiceMemberRef`, creates no Canonical Service leg, and receives no
independent ID. A local dispatch target requires a LocalRegion owned by that
same local service. A manager dispatch target requires a BoundaryProxy.

### MemoryEngineBinding And MemoryOperationEntry

Each `MemoryEngineBinding` is keyed by one TechMapping Memory Realization and
selects one concrete `fabric.mem` Operation Engine. The selected occurrence
must have an Operation Engine and
`memoryEngineTemplate(occurrence)` must equal the exact template selected by
the Memory Realization. It owns exactly one `MemoryOperationEntry` for every
covered canonical memory actor. The entry owns the actor's physical placement
once and a complete nested use inventory for its rooted launch contexts.

Operation placement uses the closed typed `MemoryOperationPlacementRef` owned
by `docs/spec-mapping-artifact.md`; this specification only validates its
Spatial port or Temporal context against the selected engine.

`MemoryOperationEntry` is the closed union defined by
`docs/spec-mapping-artifact.md`:

* `AddressedOperation` covers read, write, RMW, and compare-exchange and stores
  the actor, placement, and a non-empty canonical array of addressed uses;
* each addressed use stores one `RootedGraphLaunchRef`, one `MemoryBinding`,
  and one typed `LocalMemoryServiceRef | ManagerEndpointRef` dispatch target;
* `FenceOperation` stores the actor, placement, and a non-empty canonical array
  of fence uses;
* each fence use stores one `RootedGraphLaunchRef` and one typed
  `MemoryConsistencyDomainRef | ManagerEndpointRef` consistency-service
  target, but no `MemoryBinding`.

The parent actor plus a child rooted launch mechanically forms the existing
Dataflow-owned `ContextualActorRef`; the wire does not persist a competing
contextual identity. Uses are sorted by canonical `RootedGraphLaunchRef` bytes
and exactly cover every rooted launch whose callee graph owns the parent
actor. Missing, duplicate, foreign, stale, or wrong-graph uses are invalid.

For each addressed use, the Dataflow owner resolves the actor's memory
capability through the exact rooted launch. The referenced `MemoryBinding`
must name the resulting `LogicalMemoryRootOrViewRef`. This permits two static
launches of one reusable graph to share one operation placement while binding
different logical memories. Fence uses have no logical-memory association.

The TechMapping Memory Realization is the sole owner of a selected
`load.data -> store.data`, `done -> ctrl`, or other Fabric-allowed
memory-internal dependency. SpatialMapping derives the corresponding concrete
connection through the exact occurrence-to-template relation. A
`MemoryOperationEntry` cannot select, omit, or replace that connection.

The operation entry does not copy access form, memory-element type, lane shape,
lane count, mask mode, actor-contract fields, endpoint widths, consistency
domain, selected capability alternative, derived access-geometry class,
configured row, or transaction decomposition. Those values are derived from
the exact Dataflow actor, TechMapping realization, selected Fabric target,
operation port, and use pattern. A dynamic mask is an ordinary actor operand
whose external endpoint correspondence or selected internal source is verified
like any other required operand. The selected internal source is read from the
TechMapping witness, not duplicated in the operation entry.

The MemoryOperationUse and ExposureEntry target fields collectively are the
only persistent owner of selected `C_dispatch`; there is no parallel dispatch
relation record. Fabric alone owns eligible `H_dispatch`, and the Mapping
verifier proves `C_dispatch` is its subset. An addressed target may be the
occurrence's Local Memory Service or a declared manager endpoint/path. A fence
target is one local MemoryConsistency domain or a declared manager endpoint to
one provider. These are alternatives in one typed service model, not different
request protocols. Runtime ABI owns the single
`SpatialServiceRequest`/`SpatialServiceResponse` boundary used by either
target.

An addressed use's target must agree with its referenced MemoryBinding:
`LocalMemoryServiceRef` selects `LocalRegion` in that exact service, while
`ManagerEndpointRef` selects `BoundaryProxy`. The endpoint is a request path,
not a service-region alias. Fence has no MemoryBinding and continues to select
only its consistency target.

A MemoryOperationEntry does not own a Physical Tag. Each externally supplied
operand role derives its own input `(physical ingress endpoint, tag)` from the
real tagged writer or ingress `ResourceUse`; each externally exposed result
role derives its own output `(physical egress endpoint, tag)` from the real
tagged writer assignment and route. A selected memory-internal source removes
the external match for that role. Tag values must be unique only for
may-overlap incompatible interpretations at the same local physical ingress
and may be reused across disjoint match domains. They are not forced equal
within one operation row and are never global firing, iteration, invocation,
logical-token, atomic-object, memory, or vector-lane identity.
Several input roles in one Temporal row may select the same tagged ingress
only because the Fabric-owned row architecture provides one independent
matcher and ordered queue per role. Their may-overlap interpretations require
distinct tags. Output role endpoints within one row remain injective under
`loom.fabric 7.0` and Mapping 6.0; Mapping cannot synthesize an unmodeled
result serializer.

Every addressed atomic operation and fence resolves through its selected
target and use pattern to exactly one compatible MemoryConsistency domain. A
fence domain must cover every memory effect constrained by its incoming and
outgoing causal edges in that execution context. A Fabric-declared composite
domain may implement internal barriers and an all-of join; Mapping may not
construct a hidden multi-domain fence.

An addressed operation whose logical service requirement is MMIO must bind to
a physical `Mmio` service region with a compatible accepted-access domain,
non-trapping behavior for that range, and the required at-most-once logical
provider observation. A volatile actor additionally requires a capability that
admits its exact volatile contract. Mapping cannot infer MMIO from an address,
upgrade ordinary storage to MMIO, or use protocol retries to excuse duplicate
provider-visible effects.

### RouteTree And ResourceUse

Token edges not absorbed by the Memory Realization use the ordinary flat
`RouteTree` model. A memory-local traversal is not free unless it is an exact
TechMapping template-relative internal-edge witness projected onto the selected
occurrence. The same route-tree rules apply to address,
data, mask, control, and completion transport. Each residual vector address,
data, or mask value is one complete logical token and must fit every selected
endpoint and traversal. Mapping cannot split it across unrelated endpoints or
use Physical Tags as lane identifiers.

`ResourceUse` owns event-relative activation, typed use-pattern parameters,
capacity occupancy, and sharing assignments over the already selected engine,
service, route, port, queue, bank, or context. It does not copy Fabric capacity,
duration, latency, or use vectors.

For a memory actor, the selected operation-port use pattern owns port-local
child-transaction projection (`Direct` or `ActiveLanesRowMajor`), holding,
assembly, and claims under one parent Canonical Service request. Each selected
service use pattern separately owns physical beats and service-local claims.
Mapping consumes the Fabric owner-defined
`deriveMemoryPortTransactionPlan` result, supplies only typed parameters and
ordinary `ResourceUse` records for those exact owner-local patterns, and
cannot reconstruct lane addresses, masks, or assembly. It cannot combine the
port and service into a cross-owner pattern, change transaction projection,
create another request identity, or derive a projection from endpoint or beat
widths. The operation endpoint payload width and backing service beat width
therefore remain separate Fabric facts rather than route or
MemoryOperationEntry fields.

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
  RootedGraphLaunchRef -> BindingRelation<SpatialMappingImportRef>
```

Relations are total on the Canonical Dataflow Program's may-domain and use
the closed `PresburgerPartition` or `StableKeyLookup` algebra. Runtime only
evaluates these immutable relations for concrete coordinates and parameters.
It never chooses another AccCore or SpatialMapping.

### ServiceRealization

SystemMapping has one `ServiceRealization` family keyed by one of two typed
software obligations. `docs/spec-mapping-identity.md` is the sole owner of the
exact `SystemServiceObligationKey` variants and wire.

For memory, the `OperationServiceObligationFamilyKey` contains only one
`LogicalMemoryRootOrViewRef` or `FenceActorFamilyRef`. The exact Dataflow
program derives the complete contextual addressed-operation member set and
the separate complete `MemoryExposureRef` set for a logical-memory owner, or
the exact contract and contextual-member set for a fence owner. None of these
sets or contracts is copied into the key. Memory access and exposure therefore
share one owner and cannot select contradictory services.

Each `ServiceRealization` contains canonical owner-local `ServicePlan` values
and the complete contextual plan-selection rows owned by
`docs/spec-mapping-artifact.md`. A plan contains:

* one closed target-binding variant: logical-service interval to Fabric
  service region for addressed memory, or fence family to one
  MemoryConsistency domain; each memory-region target owns its complete
  provider-terminal bindings keyed by `MemoryExposureRef`;
* one `TransferLegRealization` per member-relative
  `CanonicalServiceLegKey` mechanically required by the Canonical Service
  Schema; and
* mapping-visible physical refinement assignments.

A fence plan has exactly one consistency target. The selected domain may be a
Fabric-declared composite domain, but Mapping cannot replace it with several
unrelated targets and a hidden join. An exposure binding has no
`TransferLegRealization`; the addressed actors that use the exposed capability
own any request and response legs.

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
  -> Fabric memory spatial_attachment
       (Module/occurrence endpoint pair + exact System service endpoint)
  -> SystemMapping ServiceRealization
  -> system provider service or explicit external provider
```

A Spatial-local service that completes the operation produces no system
obligation. A boundary proxy creates an operation-service obligation, and
SystemMapping uniquely selects the system route, provider region, and address
transform within the service/transform closure of the attachment-bound System
service endpoint. It does not select or replace that endpoint. Partial closure
may end only at an explicit external provider; it must not pretend that a proxy
is final storage.

The attachment-bound endpoint is mechanically derived from the applicable
execution-binding result, selected Module-local manager path, and Fabric. An
immutable SpatialMapping supplies that path in the current hierarchical search.
The selected AccCore occurrence qualifies the corresponding Module boundary and
therefore fixes the one Fabric attachment row. Reusing one finalized
SpatialMapping on another AccCore occurrence may reach another System endpoint
with a different capability closure.

System PnR domain construction is factorized by the exact bound endpoint; it
neither unions capabilities across endpoints nor requires every occurrence to
satisfy an intersection. Finalized plan selection uses the owner-relative
anchor and `ExecutionContextKey` defined by Mapping identity after `B_graph`
targets immutable SpatialMappings. Incompatibility makes only a candidate
using the affected endpoint infeasible.

Whether service is local or crosses a manager endpoint does not change the
runtime request schema. Both use the Runtime ABI's typed
`SpatialServiceRequest` and `SpatialServiceResponse`; adapters translate that
one boundary to a local model, standalone external-service model, RTL harness,
or manager Bridge without reinterpreting the Mapping binding.

System `ResourceUse` for a selected plan element uses
`ServicePlanElementRef = (ServiceRealizationKey, canonical plan ordinal,
typed element key)`. Applicability is derived from the owning contextual
plan-selection rows; the use does not copy its predicate, target, or selected
plan. System `ResourceUse`
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
* one MemoryOperationEntry per covered actor, including addressed and fence
  variants, an exact internal edge, and joint load `data + done` retirement;
* one reusable graph memory actor launched against two logical roots, with one
  actor placement, two rooted uses, exact binding resolution, and rejection of
  missing, duplicate, foreign, or wrong-graph use rows;
* element, contiguous, indexed, masked, and unmasked access compatibility,
  including rejection of an equal-width but semantically incompatible port;
* routing of complete vector address, data, and mask tokens without lane Tags
  or implicit endpoint splitting;
* one declared Fabric child-transaction plan with one parent request and one
  logical actor retirement, plus rejection of Mapping-invented decomposition;
* local Physical Tag ownership by the real writer/ingress `ResourceUse`;
* distinct tags for may-overlap input roles sharing one Temporal ingress,
  legal tag reuse across disjoint ingress match domains, and rejection of
  shared output endpoints within one operation row;
* shared hybrid operation-port capacity versus separate element and vector
  ports, with rejection of a persisted derived geometry class;
* local-service and manager-endpoint MemoryOperationUse targets using the
  same typed Spatial Service request/response boundary, with exact
  LocalRegion-versus-BoundaryProxy agreement;
* local finite-range containment and dynamically unbounded Whole selection of
  BoundaryProxy, plus rejection of endpoint-as-service aliases;
* one atomic operation and one fence resolving to exactly one compatible
  MemoryConsistency domain, plus rejection of a hidden multi-domain fence;
* one volatile MMIO binding with non-trapping at-most-once provider behavior,
  plus rejection of ordinary storage or provider-visible replay;
* MemoryOperationUse and ExposureEntry dispatch ownership plus rejection of
  `C_dispatch` outside Fabric-owned `H_dispatch`;
* one system-memory request/response plan and one unsplit multicast; and
* Deployment-owner closure of every selected memory and service dependency,
  with package-index authority and runtime remapping rejected.

Tests should not freeze configured table layout, protocol encoding, printer
format, or runtime cache structure.
