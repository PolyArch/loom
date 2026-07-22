# Place And Route

This document is the normative owner of Loom Spatial and System PnR
algorithms, native state, deterministic search protocols, final closure, and
the common Spatial and System MappingConstraintSet semantic and wire algebra.
The Mapping profile documents own persistent Mapping Artifact spelling and
canonical serialization; this document separately owns the constraint-family
roots and clause wire schema.
Evaluation and central DSE documents own objective, gate, Evidence, and
resolved-model schemas. This document consumes those interfaces without
restating them.

There is one Dataflow-to-Fabric Mapping artifact family. TechMapping selects
semantic realizations, SpatialMapping adds physical realization inside a
SpatialCore, and SystemMapping binds complete execution and service behavior
across an architecture-only Fabric system. There is no `PhysicalMapping`
profile and no fourth profile for flat System search.

## MappingConstraintSet Artifact Family

This document is the sole semantic and wire owner of:

```text
loom.mapping_constraints 1.0
```

The family has exactly two complete root operations:

```text
mapping.constraints.spatial
mapping.constraints.system
```

The root operation is the profile discriminator; there is no parallel profile
attribute or generic root with inactive optional fields. Both roots use the
same three clause atoms, four carrier encodings, canonicalization, and outcome
algebra defined here. They differ only in exact upstream bindings and their
closed typed projection catalogs.

An empty canonical clause sequence is a real exact Artifact. It means no
additional restriction beyond base legality; it is not a missing input, null
constraint, config default, or wildcard object. The Artifact uses the Common
SHA-256 v1 finalization contract and the family-owned canonical textual writer.
Native indexes are removable projections and never persistent alternatives.

Each root owns one single-block declarative clause region with no block
arguments, SSA values, CFG successors, symbols, or runtime terminator. Its
children are exactly:

```text
mapping.constraint.domain_restriction
mapping.constraint.equal
mapping.constraint.disjoint
```

These operations carry the typed `projection`, subject references, and unique
carrier encoding described below. Unknown children, fields, projection values,
or carrier variants are rejected. Schema identity and version are supplied by
the Artifact family framing and are not duplicated as editable root fields.

The family owns one canonical textual MLIR writer: UTF-8, LF line endings, no
trailing spaces, and exactly one final newline. It emits root bindings in the
schema order shown below, the System SpatialMapping reference table in complete
ArtifactIdentity order, and clauses in canonical record order. Locations,
comments, aliases, authoring symbols, and generic printer flags are excluded.
An importer may accept legal noncanonical whitespace and authoring order, but
must parse to the typed model and re-emit these canonical bytes before Common
finalization. There is no parallel JSON, binary, or host-struct wire authority.

## Invocation Contracts

### Spatial PnR

A Spatial PnR invocation consumes exactly these five authorities:

```text
D = Canonical Dataflow Program
T = TechMapping
F = fully elaborated Fabric Hardware Description
C = ResolvedPnrConfigView
K = MappingConstraintSet
```

`D` is one Canonical Dataflow Program. `T` is a verifier-clean, profile-complete
TechMapping bound to exact `D` and `F`. `F` is fully elaborated. `C` is the
immutable `ResolvedPnrConfigView` mechanically derived from one complete
ResolvedConfig. `K` is the independent immutable Artifact defined by the
Spatial MappingConstraintSet Contract below.

`freezeSpatialPnrProblem(D, T, F, C, K)` is the only aggregate freeze entry.
It rejects every identity, profile, reference, schema, and config mismatch
before native allocation. `C` is a typed component view, not an artifact or a
second config authority. It is mechanically recoverable from the exact
ResolvedConfig and cannot be authored or patched independently. The borrowed
input grouping is not a request artifact.

In particular, freeze requires `T.D == D.id`, `T.F == F.id`, and every
`K.D/T/F == D.id/T.id/F.id`. These are exact bindings, not compatibility
checks or rebinding permission.

The persistent authorities are `D`, `T`, `F`, the complete ResolvedConfig, and
`K`. A SpatialMapping root binds exact `T`, `D`, and `F`; `C` and `K` are bound
by the `InvocationManifest` and exact admission, not by SpatialMapping semantic
identity.

### System PnR

A System PnR invocation consumes exactly:

```text
SystemPnrProblemInputs {
  canonical_dataflow_program D
  architecture_only_fabric_system F
  root_thread_launch_closure R
  system_pnr_search_domain H
  resolved_system_pnr_config_view C
  system_mapping_constraint_set K
}
```

`D`, `F`, and `R` use exact artifact identities and stable references. `F`
owns the exact Transport Architecture and excludes every protocol-specific
Interconnect Implementation. `R` is the closure rooted at the requested
non-empty set of root thread launches.
The separate immutable System MappingConstraintSet `K` is governed solely by
the System contract below. System PnR does not invent a system-wide TechMapping
input. `C` has the same resolved-view contract as Spatial PnR and
includes the exact Evaluation binding table used by its
`ObjectiveProjection`.

`H` is the immutable, canonical-framed finite search-domain view mechanically
elaborated from `D`, `F`, `R`, resolved Compilation and DSE policy, and `K`.
It owns finite binding atoms and the legal AccCore, SpatialCore, service, and
endpoint domains for each atom. It is not an artifact, Mapping result, or
second config authority. Its canonical identity participates in the native
cache key and its exact descriptor and digest are recorded in the
`InvocationManifest`, so a changed candidate domain is a changed invocation.

`R`, `H`, `C`, and `K` affect search, closure, and admission but do not enter
the semantic identity of a selected SystemMapping. The `InvocationManifest`
binds their exact references or component-view descriptors and digests. The
persistent root owns `D`, `F`, the non-empty root thread launch set, the exact
derived SpatialMapping imports, and its selected records.

An InstructionCore-only closure may have no SpatialMapping catalog or reopened
Spatial subproblem. That case still uses the ordinary SystemMapping and System
MappingConstraintSet profiles; it never creates a dummy graph or
SpatialMapping.

## Spatial MappingConstraintSet Contract

This section owns the Spatial root and projection catalog. Mapping artifact and
verifier documents may define where `K` is referenced or applied, but do not
redefine this algebra or the common family above.

`K` is an independent immutable Artifact. It is not a `ResolvedConfig` field
or component view, Fabric content, Mapping result state, mutable solver state,
or a cache. Its closed Spatial root is:

```text
mapping.constraints.spatial {
  exact Canonical Dataflow Program binding D
  exact TechMapping binding T
  exact Fabric Hardware Description binding F
  canonical clause sequence
}
```

The sequence is conjunctive.
An empty sequence is a real exact Artifact bound to that same `D/T/F`; it
means unrestricted beyond base legality and is not absence, a null input, or
a hidden default.

Each clause is exactly one of three closed record variants:

```text
DomainRestriction {
  projection
  subject
  admissible_domain
}

Equal {
  projection
  subjects
}

Disjoint {
  projection
  subjects
}
```

`DomainRestriction` has exactly one subject and requires the subject's
projected set to be a subset of `admissible_domain`. `Equal` and `Disjoint`
each have a canonical variadic subject sequence with at least two members, all
typed by the same projection. `Equal` requires their projected sets to be
exactly equal; `Disjoint` requires every pair of projected sets to have empty
intersection. An n-ary `Disjoint` remains one variadic record and is not
expanded into binary pairs.

`ProjectionKind` is the closed, statically typed wire discriminator for the
subject decoder, result carrier, result cardinality, admissible-domain
encoding, and final-SpatialMapping projector and verifier. A subject may
reference only a typed entity in the exact `D/T/F` inputs or a stable
pre-result structural key mechanically derived from those inputs. Mapping
records, candidate handles, solver variables, freeze-local indices, and all
other result-time entities are illegal subjects. Every closed projection is a
total function for a legal subject and base-valid SpatialMapping; its result
is exactly singleton, non-empty set, or zero-or-more set as declared below.

### Closed Spatial Projection Catalog

The graph-local net and transfer-terminal subjects are the closed pre-result
keys:

```text
SpatialLogicalNetKey = ExposedCanonicalProducerEndpointRef

SpatialTransferTerminalKey =
    Source(SpatialLogicalNetKey)
  | Sink(SpatialLogicalNetKey,
         ExposedCanonicalConsumerEndpointRef)

PhysicalAddressPoint = (PhysicalMemoryServiceRef, Address)
```

`SpatialLogicalNetKey` identifies the residual transfer obligation derived
from exact `D/T` after realization-internal sinks are removed. It never refers
to a result-time RouteTree, Tag assignment, or native net index. The first
Spatial `ProjectionKind` catalog is exactly:

```text
compute_placement:
  ComputeRealizationRef -> Set<FabricFuOccurrenceRef> [singleton]

compute_parent_pe:
  ComputeRealizationRef -> Set<FabricPeOccurrenceRef> [singleton]

compute_instruction_context:
  ComputeRealizationRef -> Set<InstructionContextRef> [singleton]

compute_fu_context:
  ComputeRealizationRef
    -> Set<(FabricFuOccurrenceRef, InstructionContextRef)> [singleton]

memory_placement:
  MemoryRealizationRef -> Set<FabricMemOccurrenceRef> [singleton]

net_assigned_tag_values:
  SpatialLogicalNetKey -> Set<PhysicalTagValue> [zero-or-more]

net_selected_physical_traversals:
  SpatialLogicalNetKey -> Set<FabricPhysicalTraversalRef> [zero-or-more]

net_traversal_resource_states:
  SpatialLogicalNetKey -> Set<FabricResourceStateRef> [zero-or-more]

spatial_transfer_attachment:
  SpatialTransferTerminalKey
    -> Set<FabricTransportEndpointRef> [singleton]

memory_operation_port:
  CanonicalMemoryActorRef
    -> Set<FabricMemoryOperationPortRef> [singleton]

memory_bound_services:
  LogicalMemoryRootRef -> Set<PhysicalMemoryServiceRef> [non-empty]

memory_address_region:
  LogicalMemoryRootRef -> Set<PhysicalAddressPoint> [zero-or-more]
```

`compute_placement` reads the selected FU occurrence.
`compute_parent_pe` derives that occurrence's owning PE, while
`compute_instruction_context` reads the independently selected context and
`compute_fu_context` preserves their required correlation as a full tuple.
`CanonicalMemoryActorRef` is restricted to a canonical load or store covered
by a Memory Realization in exact `T`. A zero-hop net may produce empty route
and route-resource sets; an untagged net may produce no Physical Tag values;
and a zero-sized memory object may have an empty address region. These are
declared cardinalities, not absent or unknown projection results.

There is no generic configuration projection. In particular, no string field
path, raw control value, arbitrary property bag, or
`fabric_control_value(resource, field)` escape exists. The initial generic
configuration projection inventory is closed and empty.

### Common Persistent Carrier Encodings

Each `ProjectionKind` selects exactly one of four persistent admissible-domain
encodings:

1. Typed nominal or structural references use a canonical sorted unique
   exact-reference set.
2. Unsigned scalar values such as Physical Tags use sorted, merged,
   non-overlapping half-open intervals.
3. Physical addresses are grouped in canonical
   `PhysicalMemoryServiceRef` order; each service contains sorted, merged,
   non-empty half-open address intervals.
4. Closed tuple carriers use a lexicographically sorted unique full-tuple set.
   They are never split into component domains or a Cartesian product.

Persistent bit masks, bitsets, alternate encodings, and carrier property bags
are forbidden. `SparseIds`, `DenseWords`, and other bitset forms are derived
`FrozenConstraintIndex` choices only and cannot be written back into `K`.
There is no predicate DSL, `Exists` or `NonEmpty` atom, runtime extension
registry, or last-wins interpretation.

### Common Canonicalization And Hot Compilation

Both roots perform one deterministic finite normalization over their own closed
projection catalog:

1. Resolve exact `D/T/F`, every typed subject, `ProjectionKind`, operand, and
   carrier value, then normalize each admissible domain in its unique
   persistent encoding before freeze.
2. For each `ProjectionKind` independently, compute equality closure, select
   the minimum canonical subject key as each class representative, and emit
   exactly one sorted `Equal` record for every non-trivial class.
3. Intersect all `DomainRestriction` domains in an equality class and retain
   at most one merged restriction on its representative.
4. Canonically deduplicate identical authored subjects, then rewrite every
   distinct `Disjoint` subject to its equality-class representative. Before
   deduplicating representatives, apply the forced-empty rule below when
   equality closure collapses distinct subjects into one class. Remove
   representatives already constrained empty, discard groups with fewer than
   two remaining members, remove exact duplicate records, and retain every
   other group as one canonically sorted variadic record.
5. Sort all records by stable schema wire discriminant and canonical payload.

Authoring order, duplicate domain elements or records, and different `Equal`
chains that form the same closure therefore produce the same canonical `K`.
There is no last-wins precedence. An n-ary `Disjoint` is not canonicalized to
a binary clique, and a binary clique is not reconstructed into an n-ary
record.

Absence of a `DomainRestriction` means unrestricted beyond base legality;
an explicit empty admissible domain requires the projected set to be empty.
That requirement is valid for a zero-or-more projection and is
`ProvenInfeasible` for a singleton or non-empty projection. If equality
closure maps multiple members of one `Disjoint` record to the same class, that
class is likewise forced empty: canonicalization records an empty restriction
for a zero-or-more class and reports `ProvenInfeasible` for a singleton or
non-empty class. An empty merged domain intersection follows the same rule.

The constraint pipeline keeps four non-success categories distinct:

* `Invalid` for malformed encoding or an unresolved, ambiguous, foreign, or
  ill-typed reference, subject, projection, or carrier;
* `ProvenInfeasible` only with an exact contradiction witness after canonical
  `K` exists and before search begins;
* `Inconclusive` when bounded analysis or search establishes no complete
  result; and
* `InternalError` for an implementation invariant failure or disagreement
  between freeze, incremental checking, and final admission.

These outcomes are not fields or status values in `K` or a Mapping Artifact.
Lack of a proof never becomes `ProvenInfeasible`.

Freeze mechanically compiles the exact root inputs and `K` into an immutable,
projection-sharded `FrozenConstraintIndex`. It has no ArtifactIdentity or
semantic authority. Equality classes and variadic disjoint groups remain
distinct; `Disjoint` is not expanded into binary pairs or a dense matrix.
Every hot table, finite-domain representation, and reverse index is a derived,
rebuildable cache. Final admission runs only on a base-valid Mapping and
independently recomputes the profile's canonical projections and clauses
without trusting that cache. The projection reads the final Mapping, but its
subject remains the pre-result anchor stored in `K`. Neither `K` nor the
admission result enters Mapping semantic content or identity.

## System MappingConstraintSet Contract

The System root is:

```text
mapping.constraints.system {
  exact Canonical Dataflow Program binding D
  exact architecture-only Fabric system binding F
  canonical non-empty root_thread_launches
  derived canonical spatial_mapping_reference_table
  canonical clause sequence
}
```

The root launch set is identical to the SystemMapping coverage root. The
reachable closure `R` is derived from exact `D` and that set and is never stored
as another authority. The spatial mapping reference table contains exactly the
sorted unique complete `ArtifactReference<SpatialMapping>` values mentioned by
`graph_selected_spatial_mapping` admissible domains. Clause payloads use table
ordinals, but the table is only a canonical reference encoding: it cannot add
an acceptable mapping, select a mapping, or replace exact references.

System subjects are limited to stable pre-result keys derived from exact
`D/F/root_thread_launches` and the Canonical Service Schema:

```text
SystemTransferLegKey =
  (SystemServiceObligationKey, canonical_service_leg_ordinal)

SystemTransferTerminalKey =
    Source(SystemTransferLegKey)
  | Sink(SystemTransferLegKey, canonical_sink_ordinal)
```

The closed System `ProjectionKind` catalog is exactly:

```text
thread_target_acc_core:
  ThreadExecutionBindingKey
    -> Set<AccCoreOccurrenceRef> [zero-or-more]

graph_selected_spatial_mapping:
  GraphExecutionBindingKey
    -> Set<ArtifactReference<SpatialMapping>> [zero-or-more]

graph_target_spatial_core:
  GraphExecutionBindingKey
    -> Set<SpatialCoreOccurrenceRef> [zero-or-more]

service_target_region:
  OperationServiceObligationFamilyKey
    -> Set<FabricServiceRegionRef> [zero-or-more]

transfer_terminal_attachment:
  SystemTransferTerminalKey
    -> Set<FabricTransportEndpointRef> [zero-or-more]

transfer_selected_traversals:
  SystemTransferLegKey
    -> Set<FabricPhysicalTraversalRef> [zero-or-more]

transfer_resource_states:
  SystemTransferLegKey
    -> Set<FabricResourceStateRef> [zero-or-more]

transfer_assigned_tag_values:
  SystemTransferLegKey
    -> Set<PhysicalTagValue> [zero-or-more]
```

Each projection returns the canonical union over the complete normalized
binding relation or all selectable service plans for its subject. An empty
range is a real empty set, for example when a statically empty logical domain
has no selected execution point or a zero-hop transfer uses no traversal.
`graph_target_spatial_core` is derived from each exact selected
SpatialMapping's target occurrence; it does not copy that fact into `K`.

The System root uses the same `DomainRestriction`, `Equal`, and `Disjoint`
records, carrier encodings, canonicalization, and four outcome categories as
the Spatial root. A clause subject cannot be a candidate ID, native index,
route-tree node, Mapping record, selected plan ordinal, or any other result-time
entity. There is no generic field path, predicate DSL, or System-only atom.

Freeze validates the exact `D/F/root_thread_launches` bindings, resolves every
subject and admissible-domain reference, and builds the System projection
shards in the ordinary `FrozenConstraintIndex`. Final
`SystemMappingConstraintAdmission` independently recomputes all projections
from the base-verified SystemMapping and its exact imported SpatialMappings.
The System `K` and admission result remain invocation facts and never enter
SystemMapping identity.

For example, a constraint set may restrict one thread binding to
`{acc_core_0, acc_core_1}`, require two graph bindings to select the same set of
SpatialMapping Artifacts with `Equal(graph_selected_spatial_mapping, ...)`, and
require two service legs to share no Fabric ResourceState with
`Disjoint(transfer_resource_states, ...)`. These are ordinary compositions of
the three atoms. Co-location, separation, and route disjointness therefore need
no dedicated atom or predicate language.

## Semantic Ownership

TechMapping alone owns selected Compute and Memory Realizations, semantic
encodings, configured-function match relations, configured-memory internal
connectivity witnesses, and software boundary correspondence. Spatial PnR
must not regroup actors, reconstruct a deleted `dataflow.subgraph`, rematch raw
Dataflow and Fabric, or select another semantic realization.

Fabric owns immutable topology, occurrences, endpoints, traversals, resource
and capacity schemas, use patterns, service contracts, and physical refinement
domains. Mapping selects only declared alternatives and owns physical legality
and the domain-independent PnR measures `V` and `G`. Evaluation owns all
accelerator- and workload-aware observations `Q`. The central resolved
`ObjectiveProjection` is the only composition of `V`, `G`, and `Q`.

MappingConstraintSet adds hard restrictions to the profile's base legality.
`C` exposes resolved search policy, derived deterministic
work accounting, temporary-violation admission, and candidate comparison
through mechanical component views. Numeric work limits remain owned by the
typed policies that define the corresponding work units. Neither objective
weights nor temporary penalties can legalize a base or `K` violation.

## Persistent Projection Boundary

The native candidate projects only into the exact SpatialMapping and
SystemMapping roots owned by `docs/spec-mapping-artifact.md`. That document is
the sole authority for profile versions, record families, field shapes,
structural keys, defaults, and canonical assembly. Identity and reference
semantics come only from `docs/spec-mapping-identity.md`.

PnR maintains selected decisions in cache-oriented native form and invokes the
Mapping owner to build and finalize the persistent root. Persistent-facing C++
records, importers, writers, and hot views are mechanically generated or
projected from the Mapping schema. They may add rebuildable dense indices,
CSR/SoA tables, and algorithm scratch, but they cannot define another record,
field, version, identity, or serialization authority. Search state, scores,
candidate collections, histories, journals, and negotiation prices never
become Mapping records.

## Native State

The hot path has exactly four ownership classes:

1. immutable aggregate `FrozenModel`;
2. per-restart mutable `CandidateState`;
3. worker-local mutable `SearchScratch`; and
4. `MoveTransaction` for atomic candidate changes.

Submodules may expose typed builders or read-only subviews. They may not
publish independently constructed realization, routing, constraint, or
configuration freezes that can be mixed across inputs.

### FrozenModel

Spatial freeze has one publication sequence:

```text
validate exact D/T/F/C/K coupling and profile completeness
  -> derive normalized semantic and physical rows
  -> preflight every count, offset, product, and PnrIndex requirement
  -> build canonically ordered contiguous tables
  -> build CSR and SoA hot indices
  -> run the aggregate verifier
  -> atomically publish the immutable FrozenModel
```

Failure exposes no partial model. Freeze validates, resolves, indexes, and
precomputes; it never selects placement, context, attachment, route, tag,
buffer, `ResourceUse`, memory binding, or physical refinement.

If a count, offset, product, or maximum index cannot be represented by the
build-selected `PnrIndex`, freeze fails before allocation, cache publication,
or search. This is a Loom build-capacity error, not Mapping infeasibility. The
diagnostic identifies the affected artifact/table/domain, required maximum,
current `LOOM_PNR_INDEX_BITS` value, and the exact remedy of reconfiguring and
rebuilding with `LOOM_PNR_INDEX_BITS=64`. Loom must not truncate, wrap,
publish a partial model, or switch native width at runtime. Persistent
`EntityId` exhaustion remains a separate artifact-finalization error.

The aggregate Spatial model contains at least these complete groups:

* exact TechRealization, actor, edge, and port disposition, exposed terminals,
  residual logical nets, and service-leg projections;
* Fabric occurrences, contexts, endpoints, traversals, `ResourceState`s, and
  tag, buffer, memory-service, and refinement capabilities;
* factorized occurrence, context, attachment, and refinement domains;
* derived compiled `K` indexes, fully resolved `C`, and reachability,
  lower-bound, dependency, and reverse-incidence indices.

It never contains selected decisions, occupancy, claims, costs, Evaluation
results, statistics, history, or a transaction journal.

The Spatial PnR cache key hashes exact `D.id`, `T.id`, `F.id`, and `K.id`; the resolved component-view
descriptor and mechanically derived `component_view_digest(C)`; freeze and
importer semantics; the native-layout ABI; and the actual `PnrIndex` width. The
digest uses the framing owned by `docs/spec-config-ssot.md`. The complete
ResolvedConfig identity remains in the `InvocationManifest`. Two complete
configs may reuse a freeze only when they produce identical `C`. A cache hit
revalidates the descriptor, canonical view bytes, digest framing, and exact
artifact inputs before reuse.

System freeze applies the same atomic publication rule to all six inputs. Its model contains
canonical dense indices, compatible target domains, arbitrary directed
Transport Architecture CSR, endpoint-domain lower bounds, resource/capacity
schemas, binding-channel-service reverse dependencies, and either immutable
hierarchical imports or exact flat reopen domains. It does not contain
selected bindings, routes, prices, observations, or history.

All independent dense universes use typed `DenseIndex<Tag, PnrIndex>` values.
Persistent `EntityId` values appear only at import and projection boundaries.
SoA is used for independently accessed hot fields, CSR for one-to-many and
reverse incidence, and compact AoS only for small records always read
together. Layout tuning may not alter typed universes, canonical ordering,
semantics, or the persistent schema.

### Factorized Domains

Freeze never materializes a Cartesian product of occurrence, context,
attachment, and configuration choices. It owns these relations:

```text
Unit -> compatible occurrence domain
(Unit, Occurrence) -> compatible InstructionContext domain
(Unit, Occurrence, PortDemand) -> local attachment endpoint domain
ConfigurationOwner -> semantic-preserving physical refinement domain
selected facts -> ProgrammedConfigurationKey
```

Every selected capability-template port is classified exactly once as
`Internal`, `ExternalDemand`, or `InactiveQuiescent`. The last case is legal
only when the operation schema and capability relation prove no consume, no
produce, and no backpressure. A missing `PortDemand` is not an inactive marker.

For a Spatial unit `u`:

```text
CandidateDomain(u) = ImplDomain(u)
                   intersection UnaryEligible(u)
                   intersection ConstraintDomain(u)
```

`ImplDomain` is owned by the exact implementation membership in `F`.
`UnaryEligible` checks only facts provable from one unit and one occurrence:
encodable exact configuration, context/configuration/runtime-state capacity,
non-empty attachment domains, legal inactive ports, and required local
matching. It does not prove cross-unit sharing, global routing, tags,
resource-time closure, or deadlock freedom. `ConstraintDomain` is the derived
unary filter from applicable `DomainRestriction` records. `Equal` and
`Disjoint` compile to separate relation-propagation and conflict indexes; they
are not pre-enumerated into unary candidate products. An empty well-formed
intersection is `ProvenInfeasible`, not `Invalid`.

Compute occurrence and context domains remain correlated. Memory domains use
concrete `fabric.mem` occurrences and operation placement capabilities.
Context co-residency compares the complete derived
`ProgrammedConfigurationKey`, not a template or encoding identifier. That key
is rebuildable native state, not persistent identity.

System `H` factorizes parameterized binding relations into finite atoms for
search. The candidate cannot synthesize new Presburger predicates or alter a
logical domain. Finalization reconstructs the closed persistent
`PresburgerPartition` or `StableKeyLookup` relations from the selected atoms.

### CandidateState

`CandidateState` is one complete set of selected Mapping decisions. It is not
a collection of independently authoritative placement, routing, tag, memory,
and resource subresults.

Spatial selected decisions include only non-derived choices:

* Compute and Memory Realization occurrence bindings and correlated contexts;
* selected attachment endpoints for every external `PortDemand`;
* route-tree root/sink bindings and parent traversal relations;
* non-derived event-relative reservations, buffer choices, and sharing values;
* Physical Tag values at continuity origins;
* memory occurrence, operation-port/context, service-region, interval, and
  address-transform selections;
* mapping-visible physical refinements such as a selected FIFO bypass mode.

Candidate caches may contain only exact functions of `FrozenModel` plus those
decisions: occurrence/context/port/service/buffer occupancy, route-derived
claims and switch configuration, tag continuity/interference domains, reverse
incidence, and typed `V/G` components. Search work is not a cost component.
Timing, slack, criticality, runtime, power, and other `Q` remain in exact
Evaluation adapter state.

System candidates add `B_thread` and `B_graph` selections, selected immutable
SpatialMappings or flat Spatial decisions, ExecutionBinding context choices,
service plans and route trees, and system `ResourceUse` occupancy and sharing
assignments. `ServicePlan` owns System service, physical-buffer, and
physical-refinement selections; `ResourceUse` owns only the event-relative
occupancy and sharing assignments of already selected elements. These
decisions are kept in the same candidate as reopened Spatial decisions in flat
mode.

Version 1.0 derives each InstructionCore context from the selected AccCore as
`InstructionCoreContextRef = (AccCoreOccurrenceRef, 0)`. A selected service
plan element is addressed by
`ServicePlanElementRef = (ServiceRealizationKey, canonical plan ordinal,
typed element key)`. Neither reference creates a second target-selection
decision.

Any discrepancy between selected decisions and a rebuildable cache is an
internal invariant failure. Full owners report the drift and terminate the
attempt; they never overwrite the cache and continue.

### SearchScratch

`SearchScratch` owns reusable A* distance, queue, predecessor, epoch, and
touched arrays; route overlays and arenas; matching and repair worklists; and
the active Action's PathFinder or DualSubgradient state. It has no semantic
identity and is discarded after its owner operation. Negotiation history is
never carried across Actions.

### Actions And MoveTransaction

All search policies use one closed Action algebra:

```text
SpatialMappingAction =
    RealizationBindingAction
  | TransportRoutingAction
  | ResourceAllocationAction
```

For System PnR the same variants use System anchors: execution binding,
channel or service routing, and resource allocation. There is no System-only
transition model.

An Action is immutable intent shaped as `(kind, typed_anchor, typed_choice)`.
For immutable `M`, resolved `C`, candidate `S`, and Action `a`, the dynamic
domain `A(M,C,S)`, dependency closure `Dep(M,S,a)`, and transition
`Apply(M,C,S,a)` are deterministic. Randomness belongs only to the selector.

`MoveTransaction` is the sole mutation mechanism. It computes the complete
dependency closure, reserves any storage that can fail, journals selected
decisions and derived state, applies the change in a shadow candidate, and
commits or rolls back Mapping and Evaluation state together. A binding change
must invalidate old attachments and route claims, rebuild every incident
route dependency, and update resource-time, buffer, tag, memory, `V/G`, and
affected Evaluation subjects before commit. A resource change that invalidates
placement or routing follows the same closure.

Exact references, domain membership, type and width compatibility, directed
connectivity, and a route being either explicitly unrouted or a valid rooted
arborescence are never relaxable. Implicit broadcast or merge, same-net
reconvergence, invalid tags, and unresolved references are not candidate
states. Only closed kinds admitted by `TemporaryViolationPolicy` may remain in
a committed search candidate; all must be zero before finalization.

## Edge Disposition And Routing

### Internal Realizations And Residual Nets

Every canonical software edge is accounted for exactly once. Closure first
recognizes every explicit realization-internal owner confirmed by `D/T/F` and
the selected physical facts:

* the configured FU relation of a Compute Realization;
* the configured `fabric.mem` internal-connectivity witness of a Memory
  Realization;
* an explicitly supported temporal PE register-file absorption; or
* another explicit Fabric internal realization with the same typed proof.

An internally realized edge has no residual logical net and no `RouteTree`.
Temporal PE co-location alone is not absorption: if the register file is
exposed as ordinary transport traversal rather than an internal realization,
the edge remains residual and the route must consume that explicit traversal
and its resources. The same rule applies to local selectors, switches, FIFOs,
boundaries, and module connections. No connection is inferred from ownership,
coordinates, names, or co-location.

For every residual producer endpoint, freeze groups all residual sink
obligations into one deterministic multi-sink logical net:

```text
SpatialLogicalNetKey = ExposedCanonicalProducerEndpointRef
```

Already internal sinks are omitted. If none remain, no logical net exists.
Dense net indices are rebuildable native indices, never persistent identity.

Memory and other operation-relative services do not have a hard-coded request
route plus response route. The Canonical Service Schema mechanically derives
the exact abstract request, data, response, completion, or other transfer legs
required by each typed operation. Only residual legs become transfer
obligations. Spatial and System routing realize those legs without adding,
deleting, combining, or reinterpreting them.

### Route Trees

Each residual logical net is realized by one rooted arborescence with shared
trunks and explicit branches. Its persistent field shape, owner key, canonical
node ordering, sink attachments, and System transfer-leg form are owned only
by `docs/spec-mapping-artifact.md`.

The routing algorithm must maintain the same semantic invariants before
projection: the root has no incoming traversal; every non-root node has one
parent and one incoming physical traversal whose source is the parent
endpoint; a sink attaches to a node with the exact required endpoint; and a
zero-length connection attaches at the root. One endpoint cannot appear as two
nodes, so reconvergence is structurally impossible. Shared trunks are claimed
once. Fanout is legal only at a Fabric traversal or endpoint that explicitly
supports broadcast. Search pointer layout, insertion order, selected-edge
bitsets, and derived claims are disposable native state and never persistent
schema.

### Endpoint-Only A*

The only A* state identity is:

```text
AStarSearchState = TransportEndpointIndex
```

`RouteQuery` fixes the logical net or service leg, legal source frontier,
target endpoint domain, payload kind and width, applicable `K` restrictions,
and the frozen candidate cost and occupancy views. Filtering enforces endpoint
direction, type, width, boundary conversion, and selected configuration.
Predecessors store only physical traversals.

Width legality is route-wide, not an endpoint-only approximation. The
software payload width must fit the data field of every selected transport
endpoint and traversal. Tag fields never contribute payload capacity. In a
tagged domain, the assigned tag must independently be representable without
loss by every tag field that still distinguishes the flow. Same-kind physical
connections may widen and later narrow according to Fabric's low-bit-aligned
rule, but no selected segment may narrow below the software payload width.
Thus an `i16` transfer may use `bits<32> -> bits<64> -> bits<32>`, but it may
not use `bits<8>` or borrow the tag field of `bits_tag<8,8>`. These checks are
structural legality and cannot be relaxed into congestion cost or repaired by
an implicit adapter.

For canonical target domain `T`, the production heuristic is exactly the
static minimum Mapping lower-bound route cost from endpoint `v` to any target:

```text
h(v, T) = min static_lower_bound_cost(v -> t), for t in T
```

It is computed by reverse multi-source shortest paths on the fully elaborated
directed topology. `FrozenModel` owns the topology, target domains, and
nonnegative lower-bound arc costs; `SearchScratch` caches the exact distance
table by target-domain index. Coordinates, Manhattan distance, landmarks,
all-pairs matrices, and silent `h = 0` fallback are not authorities.

Each A* invocation freezes this checked integer proposal cost:

```text
arc_cost(a) = mapping_lower_bound_cost(a)
            + mapping_dynamic_penalty(a)
            + optional_evaluation_route_guidance(a)
```

All terms are nonnegative. Evaluation guidance is zero when unavailable and
may order proposals only when one exact resolved model safely supplies an
arc-local value. It cannot filter legal arcs, prove legality, alter the
Mapping-owned admissible heuristic, or replace full `Q` evaluation.

`RouteCost` is `uint64_t`; `UINT64_MAX` is infinity. All arithmetic is checked,
with typed overflow distinct from unreachable topology and work-budget
exhaustion. The open queue order is `(f, h, endpoint_index)` ascending.
Multi-source endpoints and outgoing traversals use canonical index order. Only
a strictly smaller `g` replaces a predecessor; equal `g` does not. Stale heap
entries are discarded, and a target is accepted only when popped.

For a multi-sink net, the router repeatedly performs one multi-source,
multi-target search from legal branch points in the existing tree to all
unresolved sink domains. It collects every equal-best target until the minimum
open `f` exceeds the best target cost, then uses:

```text
(total_branch_cost ascending,
 optional_evaluation_sink_priority descending,
 canonical_sink_obligation_index ascending,
 selected_target_endpoint_index ascending)
```

The selected branch is normalized at its last intersection with the tree and
discharges exactly one sink obligation. Overlapping target domains do not
implicitly discharge multiple sinks. Failure of any sink rejects the whole
tree proposal; partial trees are never committed.

### Negotiated Routing

`RoutingNegotiationPolicy` is a closed union:

```text
RoutingNegotiationPolicy =
  PathFinder {
    price_kernel: Multiplicative | Additive
    present_pressure_initial
    present_pressure_growth_numerator
    present_pressure_growth_denominator
    history_pressure_increment
  }
  | DualSubgradient {
    direction_kernel:
      ProjectedSigned
      | PositiveViolationOnly
      | MomentumDeflected {
          beta_numerator
          beta_denominator
        }
    step_schedule: DualStepSchedule
  }
```

Inactive fields are invalid. `PathFinder + Multiplicative` is the canonical
global default. If DualSubgradient is selected, its canonical direction
default is `ProjectedSigned` and its schedule-family default is
`GeometricDecay`. Exact numeric defaults belong only to the versioned config
schema and resolver; PnR kernels have no hidden fallback values.

For traversal `a`, normalized claim `q(a,r)`, usage `u(r)`, and capacity
`cap(r)`, define:

```text
base_cost(a,r) = q(a,r)
lower_bound_cost(a) = sum_r q(a,r)
X(a,r) = max(0, u(r) + q(a,r) - cap(r))

MultiplicativeCost(a,r) = q(a,r) * (1 + P * X(a,r)) * (1 + H(r))
AdditiveCost(a,r)       = q(a,r) + P * X(a,r) + q(a,r) * H(r)
arc_cost(a)             = sum_r ResourceCost(a,r)
```

A pure structural traversal with no claim may have zero cost. Both kernels
share occupancy, iteration order, update rules, A*, route trees, and
transaction protocol. PathFinder uses deterministic Gauss-Seidel occupancy:
each net removes only its selected old claims, reroutes against the current
working overlay, and installs its new claims. `P` and `H` remain frozen within
the iteration. The next iteration uses:

```text
O_k(r) = max(0, U_k(r) - capacity(r))
H_(k+1)(r) = H_k(r) + history_pressure_increment * O_k(r)
P_(k+1) = ceil_mul_div(P_k,
                       present_pressure_growth_numerator,
                       present_pressure_growth_denominator)
```

`H_0(r) = 0`, `P_0 >= 1`, history increment is at least one, and the reduced
growth ratio is at least one. Updates occur atomically only after a complete,
non-closed iteration. At the start of iteration `k`, PathFinder derives this
complete key from the previous complete route overlay and then freezes it for
the whole iteration:

```text
NetOrderKey_k(n) = (
  route_state_rank(n),
  descending generic_conflict_pressure_k(n),
  descending optional_evaluation_priority_k(n),
  canonical_net_index(n)
)

route_state_rank(n):
  0 = currently unrouted
  1 = contributes to negotiated routing violations
  2 = other participating net

generic_conflict_pressure_k(n) =
  sum_r claim_k(n,r) * max(0, occupancy_k(r) - capacity(r))
```

A shared Route Tree prefix contributes once under normalized claim semantics.
The optional Evaluation priority comes from the invocation-frozen exact
Evaluation binding and is zero when unavailable; Mapping cannot substitute a
private criticality. Checked nonnegative integer arithmetic and canonical net
index make the key total. It is recomputed only from the next complete overlay,
never from per-net Gauss-Seidel updates. There is no seeded shuffle, permanent
container order, ordering plugin, or hidden weight.

DualSubgradient routes every net independently against a fixed price snapshot,
aggregates normalized claims in canonical net order, and updates prices only
after the complete synchronous iteration. Region-external fixed occupancy is
first subtracted from physical capacity to obtain effective `C(r)`. At price
snapshot `lambda_k`, each selected traversal uses:

```text
dual_arc_cost(a) = sum_r q(a,r) * (1 + lambda_k(r))
```

After routing the complete region:

```text
g_k(r) = U_k(r) - C(r)

ProjectedSigned:       d_k(r) = g_k(r)
PositiveViolationOnly: d_k(r) = max(0, g_k(r))
MomentumDeflected:     d_k(r) = g_k(r) + beta * d_(k-1)(r)

lambda_(k+1)(r) = max(0, lambda_k(r) + alpha_k * d_k(r))
```

`beta = beta_numerator / beta_denominator` with
`0 <= beta_numerator < beta_denominator`. The single numeric protocol is:

```text
DualPrice     = uint64_t
DualDirection = int64_t
DualStep      = uint64_t
```

Every operation uses checked widened integer arithmetic. Overflow rejects and
rolls back the Action; wrap, saturation, and representation switching are
forbidden.

The closed `DualStepSchedule` variants are:

```text
Constant { step }
GeometricDecay {
  initial_step
  minimum_step
  decay_numerator
  decay_denominator
}
HarmonicDecay {
  numerator
  offset
  minimum_step
}
```

They produce, respectively, `step`, a ratio-scaled value bounded below by
`minimum_step`, and a harmonic value:

```text
Constant:
  alpha_k = step

GeometricDecay:
  alpha_0 = initial_step
  alpha_(k+1) = max(minimum_step,
                    scale_toward_zero(alpha_k,
                                      decay_numerator,
                                      decay_denominator))

HarmonicDecay:
  alpha_k = max(minimum_step,
                scale_toward_zero(numerator, 1, offset + k))
```

Every step is at least one. Geometric decay requires
`initial_step >= minimum_step >= 1` and a reduced ratio strictly between zero
and one. Harmonic decay requires positive numerator, offset, and minimum step.
Inactive fields are invalid, and degenerate constant schedules canonicalize to
`Constant`. `scale_toward_zero` uses checked widened multiplication and signed
division rounded toward zero; it is also the only rounding authority for
momentum.

Both negotiation algorithms share one outcome protocol. An iteration is
eligible only after every participating net and claim aggregation completes.
A zero negotiated violation vector returns immediately. A non-closed iterate
may be retained only when all remaining violations are admitted by
`TemporaryViolationPolicy`; it is ranked through the existing
`ObjectiveProjection` using route-related Mapping `V/G`, not A* cost or private
prices. Equal rank retains the earlier canonical iterate. Work exhaustion is
not infeasibility. Zero violation is the only normal early-convergence test;
there is no epsilon, stagnation window, route-signature cycle detector, or
hidden no-progress threshold. Exhausting the routing policy's owner-local work
limit returns the best admissible temporary iterate only for a non-final
Action; otherwise it returns typed non-closure and rolls back. Final global
closure never returns a temporary iterate.

Only the selected overlay is applied once through `MoveTransaction`.
PathFinder pressure/history, Dual prices/directions, best-iterate metadata,
and iteration traces are discarded on commit or rollback.

## Resource Use, Tags, Buffers, And Memory

Mapping has no absolute cycle-slot Schedule IR. The Structured Program
Candidate owns software schedule decisions. Physical resource-time behavior is
derived from a Fabric-owned use pattern plus Mapping-owned event-relative use.

The persistent `ResourceUse` record shape, Spatial and System owner unions,
typed use-site references, activation algebra, parameters, and sharing
assignments are owned only by `docs/spec-mapping-artifact.md`. PnR resolves each
selected use site to one Fabric-owned use pattern and maintains the resulting
event-relative claims in native state. The Fabric pattern remains the sole
owner of parameter order and domains, raw capacity, duration, latency,
initiation interval, periodicity, and service guarantees.

Each stateful Fabric resource schema also owns its closed typed
`ResourceState` set, canonical initial state, capacity dimensions, atomic
UsePatterns, stable typed requester order, and exact GrantPolicy or exact
refinement domain. One UsePattern may atomically claim multiple states. PnR
may select an exposed refinement and bind workload values, but it cannot split
an atomic pattern or construct a parallel generic resource/arbiter graph.

PnR emits a persistent use only for a non-derived activation, reservation,
release, or sharing assignment required by that schema. Static claims implied
by a selected traversal are not duplicated, and multiplicity derives from
software obligations and pattern parameters rather than duplicate records. A
causal release holds occupancy until its schema-selected event occurs; runtime
cannot infer an earlier release from observation, fairness, or record order.

For a concrete event occurrence, all immediate `ResourceUse` records with the
same owner, trigger, and concrete logical parameters form one derived atomic
activation set. The set has no persistent identity or record. The event fires
only when every member can acquire its required mapping-visible resources;
otherwise it remains waiting without partial acquisition. Future-event uses
are not reserved unless an explicit earlier-triggered `ResourceUse` does so.

The unified progress invariant is: if the Canonical Dataflow Program can
continue under fair, resource-unbounded abstract execution, a final Spatial or
System Mapping must not introduce permanent stalling through finite buffers,
routes, tags, contexts, service capacity, or arbitration. The final verifier
mechanically derives wait-for dependencies, reachable closed wait sets, and
SCC analysis from canonical program semantics, Fabric guarantees, selected
Mapping records, and the atomic activation sets. These analyses are
identity-free, rebuildable views; a dependency cycle alone is not a failure.

A statically proven closed wait set with no existing progress mechanism is the
Mapping violation `HardProgressViolation`. A deadlock observed by a model or
simulator is an Evaluation finding. Failure to establish either progress or a
counterexample is `Incomplete(proof_not_established)`. These outcomes cannot
be collapsed into one penalty, persistent `deadlock_free` flag, or generic
diagnostic authority.

Physical Tag is local to Fabric-owned interpretation domains. A selected value
is stored exactly once in the sharing assignment of a real temporal writer or
tagged ingress. Route trees and Fabric writer, rewriter, and remover points
mechanically divide the route into continuity segments. Closure intersects
every segment's allowed match domains and builds local interference from
co-residency and incompatible interpretation. Switch rows, operand matches,
memory rows, and encoded tag fields are derived from the one origin value.

An empty allowed-set intersection, an unrepresentable fixed value, or an
uncolorable local interference graph is a typed tag closure violation. Search
may retag, reroute, change endpoints or placement, or change resource-time
co-residency through ordinary Actions. There is no `TemporalTagAssignment`,
tag namespace, or independent tag-claim family.

Every Spatial physical buffer or storage choice that is not derived from a
selected traversal or service must be represented by the owning binding or
tree node as a declared traversal or physical refinement. Every System
physical-buffer or physical-refinement choice belongs to its `ServicePlan`.
`ResourceUse` expresses occupancy of those selected resources, never physical
selection. Mapping cannot insert an abstract register, FIFO, flop, or delay
not declared by Fabric.

Memory operation placement, access entries, and memory bindings use only the
closed persistent forms owned by `docs/spec-mapping-artifact.md` and
`docs/spec-mapping-memory.md`. PnR selects those typed choices in native state;
each AccessEntry and ExposureEntry selects exactly one
`LocalMemoryServiceRef | ManagerEndpointRef` dispatch target. Those fields are
the persistent `C_dispatch`; PnR checks them against Fabric-owned
`H_dispatch` and does not define another dispatch relation. The Canonical Service Schema
owns operation legs and their ordering, while route trees and service plans
realize residual legs. Provider decode, dispatch rows, response tracking, and
semantic `sw_configs` are derived from `D/T/F`, selected bindings, routes,
resource uses, and physical refinements. Physical image fields are later
encoded through the exact ConfigurationABI.

## Search Policy And Determinism

### Resolved View

The resolved view contains these orthogonal component projections:

```text
ResolvedPnrConfigView {
  SearchPolicy
  DeterminismPolicy
  DeterministicWorkBudgetView
  TemporaryViolationPolicy
  ObjectiveProjection
  ResolvedEvaluationBinding[]
}
```

The canonical binding table is referenced by `ObjectiveProjection` and
optional route guidance through `ResolvedEvaluationBindingRef`. Its entries are
exact resolved inputs, not another policy or model-selection mechanism.

`SearchPolicy` owns exploration, including initialization, Action proposal,
annealing, negotiated routing, focused closure, and exact repair. Its
`ActionProposalPolicy` contains the nonnegative integer weights
`realization_binding_weight`, `transport_routing_weight`, and
`resource_allocation_weight`. They cannot all be zero and are reduced by GCD;
empty Action kinds are removed before the remaining weights are normalized.

`TemporaryViolationPolicy` selects from the closed Mapping `V` descriptor
registry defined below. Only selected descriptors may remain nonzero in a
committed search candidate, and every selected kind must be visible to the
resolved objective. These are search permissions, not final legality or
budgets. Structural errors are never temporary.

Every typed policy that defines a semantic work unit owns its numeric limit.
Initialization owns seed-attempt work; annealing owns calibration and Action
proposal work; routing owns local A* and negotiated-iteration work; focused
closure and exact repair own their work; and final global closure owns its
Action work. The resolved controller derives only this read-only audit view:

```text
DeterministicWorkBudgetView =
  canonical set<(owner-local WorkUnitDescriptorRef, uint64 limit)>
```

`WorkUnitDescriptorRef` is the stable owner policy schema/version plus an
owner-local ordinal. The view has no independently authored numeric fields and
cannot override or reinterpret a work unit. Worker count, wall time, memory
reservation, licenses, process retries, and external cancellation are
execution controls and cannot change the formal candidate sequence.

### Objective Projection

Mapping owns the closed `V` descriptors:

```text
UnroutedObligation
CapacityOveruse
ResourceTimeOverbooking
BufferOveruse
TagUnassigned
TagConflict
HardProgressViolation
HardServiceContractShortfall
```

Mapping also owns every domain-independent `G` measure. Evaluation owns
`MetricKind`, `MetricObservation`, and typed findings for `Q`. Structural
invalidity, a `K` failure, or a base-verifier failure is never converted into
`V` or an objective penalty.

The source algebra, exact resolved Evaluation bindings, metric and finding
queries, quality-gate CNF, objective dimensions, normalization, weighted
levels, and checked `ObjectiveCode` encoding are owned only by
`docs/spec-dse-feedback.md` and `docs/spec-evaluation-metrics.md`.
`ResolvedPnrConfigView` carries their mechanically derived references for this
invocation; PnR does not restate or extend those schemas.

Before freeze, PnR preflights the exact projection and every selected hot
Evaluation binding against its candidate subject projection, requested
observations, full and incremental interfaces, candidate completeness, and
all temporary `V` kinds admitted by SearchPolicy. Authorized alternatives are
already resolved; an unavailable required source produces typed
`ObjectiveUnavailable` and cannot select another provider, become zero,
infinity, NaN, or invoke a private fallback. A temporary violation kind that
SearchPolicy admits into committed candidates must have a positive objective
term so search cannot erase its closure obligation.

SearchPolicy may explicitly include central metric or gate-deviation sources
as ephemeral `Q` guidance. Those sources create no formal Request, Evidence,
or pre-publication gate obligation. Formal quality-gate truth and candidate
selection remain post-publication `Promote` behavior. PnR sees only the exact
resolved projection and its deterministic `ObjectiveCode` evaluator.

```text
energy(candidate) = ObjectiveCode(candidate)

rank(candidate) =
  (ObjectiveCode ascending,
   canonical candidate semantic key ascending)

reward(old, new) =
  signed_difference(ObjectiveCode(old), ObjectiveCode(new))
```

The semantic candidate key breaks equal-code rank ties but does not enter the
annealing delta. Seed initialization, negotiated best iterate, annealing,
focused closure, repair, final rank, and RL all call this same evaluator. PnR
does not own another score, gate, direction, normalization, or ordering.

### Annealing And Replay

The annealing policy is the single fixed-point protocol:

```text
SearchPolicy.annealing {
  positive_delta_quantile
  target_initial_acceptance
  fallback_temperature
  minimum_temperature
  cooling_numerator
  cooling_denominator
}
```

Fractions use canonical integer or fixed-point values. Quantile is in `[0,1]`,
target acceptance is in `(0,1)`, temperatures are positive, and the reduced
cooling ratio is strictly between zero and one. Per-level proposal count is:

```text
proposals_per_level_base
  + proposals_per_movable_decision
    * movableDecisionCount(FrozenModel, candidate_at_level_start)
```

Calibration rolls back its fixed proposal count, sorts positive deltas in
stable order, and selects index `floor(q * (n - 1))`. It chooses the minimum
positive integer temperature that reaches the target under the exact
acceptance kernel. The absence of positive deltas uses `minimum_temperature`;
an invalid estimate uses `fallback_temperature`. Cooling is:

```text
T_next = max(minimum_temperature,
             floor(T * cooling_numerator / cooling_denominator))
```

There is no reheating, online acceptance-ratio adaptation, wall-time or
stagnation termination, or accepted-Action budget.

`DeterminismPolicy` contains exactly:

```text
DeterminismPolicy {
  master_seed: u64
  prng_protocol: Sha256SeededXoshiro256StarStar_1_0
  acceptance_protocol: ExpNegativeQ64Table_1_0
}
```

Each seed index derives independent `InitializerDiversification`,
`Calibration`, `ActionProposal`, `Acceptance`, and `ExactRepair` streams from
canonical SHA-256 framing. Loom's rejection-sampling `nextBounded(n)` selects
from canonically sorted domains. Host entropy, thread identity, scheduling,
container iteration, and implementation-defined random distributions are
forbidden.

For positive `deltaE`, the acceptance protocol computes
`ceil(deltaE * 256 / temperature)`, looks up the canonical Q64 exponential
threshold, and accepts only when the next `u64` is strictly below it.
Non-positive deltas accept directly. The checked-in table is the protocol;
runtime `exp` is not.

Owner policies allocate logical work slots in canonical order before parallel
scheduling. A cache hit and a cache miss consume the same slot; a seed or
generator attempt that reproduces an existing Artifact consumes its attempt
slot before Artifact-identity deduplication. Retrying the same exact
`EvaluationRequest` is execution work and creates no new semantic slot, while
a new `replicate_index` is a new request and work unit. Resume reuses finalized
outputs under their original stable ordinals and neither renumbers nor
reconsumes completed slots. Worker completion order, cache population, retry
count, and interruption therefore cannot change formal results. The
`InvocationManifest` records owner-local planned and consumed summaries without
copying their limits; owner invocation records retain attempt and checkpoint
details when recovery requires them.

### Canonical Search Sequence

Spatial PnR and System PnR use the same sequence and state:

```text
freeze and sound fail-fast checks
  -> fixed isolated seed attempts
  -> each viable seed runs coupled transactional simulated annealing
  -> focused timing and buffer closure with the same Actions
  -> optional bounded exact repair
  -> final global negotiated closure and full owner recomputation
  -> independent final verification and finalization
```

The initializer policy owns exactly `N = seed_attempt_count` isolated attempt
slots. Each attempt has its owner-local limits, builds a structurally valid
candidate, and executes an explicit global `TransportRoutingAction`. A seed
may continue only after route preparation has left every net as either a
complete valid Route Tree or an explicit policy-admitted unrouted violation;
partial trees are forbidden. It then establishes authoritative `fullPnrCost`
for `V/G`, every exact resolved hot-model input for `Q`, and the central
objective code required by the annealer.

The PnR generator runs every viable fixed attempt as its own restart with the
original stable attempt ordinal; failed attempts are not refilled. Each restart
that reaches independent final verification may emit a formal Mapping Artifact.
Selection begins only after that boundary: the central DSE `Promote` node
applies `AllPassing`, `TopK`, or `Pareto` to the canonical set of finalized
Artifact references. Mutable `CandidateState` objects never enter that central
candidate set.

The annealer interleaves binding, routing, and resource Actions. There is no
global placement freeze before routing. A binding move performs bounded local
route closure for incident nets in the same transaction. A
`TransportRoutingAction` explicitly requests whole-net, subtree, sink-subset,
region, or global negotiated routing. Full-design negotiation occurs only in
the initializer, an explicitly configured routing Action, or final global
closure; it is never an implicit temperature-boundary mutation. The final
global negotiated closure is the last budgeted `TransportRoutingAction` and
is applied through the same `MoveTransaction`, objective, and rollback
protocol as every other candidate-changing Action.

Hierarchical System search selects from a fixed, complete immutable
SpatialMapping catalog and cannot reopen its internal decisions. Flat System
search places exact Spatial reopen domains and optional immutable seeds in the
same candidate as System decisions. Both use the same Actions, transactions,
router, `V/G/Q`, objective, and verifier. Flat finalization first rebuilds and
fully verifies every new or changed SpatialMapping, assigns its stable
identity, rewrites `B_graph` and service references, and only then finalizes
the ordinary SystemMapping.

### Focused Closure

Every viable restart candidate receives the same deterministic focused timing
and buffer closure. The already resolved ephemeral Evaluation binding first
runs the full oracle for its selected model; this creates no formal Request,
Evidence, or Artifact. Closure is triggered only by a nonzero central quality-
gate deviation or typed metric explicitly selected for QoR polish by
SearchPolicy. Mapping owns no private frequency, timing, or buffer target.

An ephemeral `ClosureRegion` is derived from Evaluation critical paths,
recurrences, bottlenecks, and findings, then expanded through the ordinary
Action dependency closure. If Evaluation cannot localize the cause, the region
is the complete candidate. Proposals are ordered by unresolved required
witness first, optional Evaluation priority descending, and canonical Action
key. Each probe uses `MoveTransaction`; only the strictly best rank-improving
Action commits. Equal rank does not commit, and there is no random acceptance.

Closure stops when all selected QoR deviations reach zero, no strict
improvement exists, the deterministic proposal budget is exhausted, or a
required hot Evaluation binding fails. It then runs full Mapping and
Evaluation checkpoints. Remaining Mapping `V` must enter bounded exact repair
or prevent finalization. Remaining `Q` may enter repair when SearchPolicy asks
for it, but cannot prevent publication of a base-valid Mapping that passes the
exact `K` admission; only post-publication `Promote` applies formal
quality gates. Finalization cannot silently repair either class.

### Bounded Exact Repair

Exact repair is either disabled or the explicitly selected in-process C++
OR-Tools `CpSat_1_0` protocol. It solves only a complete bounded dependency
region derived from one canonical unresolved Mapping or Evaluation witness.
The ephemeral inputs are `FrozenModel`, resolved `C`, current `CandidateState`,
the closed conflict region, and the exact Evaluation model and constraint
identities. There is no repair artifact, alternate candidate authority,
solver plugin schema, Python path, or external solver binary.

Region closure includes affected realizations, nets and route branches,
attachments, contexts, tags, buffers, memory and service bindings,
`ResourceUse`, constraint groups, and conflicting occupancy. Outside decisions
are fixed and their claims are subtracted from available capacity. If the
complete closure exceeds `max_region_decisions`, the result is
`RegionTooLarge`; truncation is forbidden.

The solver assignment is diffed against the candidate and rebuilt as one
canonical ephemeral `ActionBatch` containing only the three existing Action
variants. One `MoveTransaction` applies the batch atomically. Mapping hard
constraints and exactly representable objective terms may enter the solver.
Approximate Evaluation information may order exploration but cannot prove
feasibility. When required `Q` is not exactly encodable, Mapping-feasible
assignments are reconstructed in canonical order and checked by the exact
full Evaluation model under its deterministic evaluation budget.

The result vocabulary is:

```text
Repaired
RegionInfeasibleUnderFixedBoundary
UnknownBudgetExhausted
RegionTooLarge
UnsupportedEncoding
InternalError
```

Only an exhaustive whole-candidate domain with every required constraint
exactly represented, or exhaustive finite enumeration with full Evaluation,
can be reported as global `ProvenInfeasible`. The adapter uses one CP-SAT
worker, a seed derived from the restart's `ExactRepair` stream, and a pinned
deterministic work limit. Budget exhaustion leaves the original candidate
unchanged.

## Evaluation Transaction

Every Action uses one online protocol:

```text
S' = ApplyAndClose(S, Action) in shadow state
VG' = exact Mapping incremental evaluation of S'
Q'  = exact resolved EvaluationModel evaluation of S'
code' = ObjectiveProjection(VG', Q')
accept or reject
commit or roll back Mapping and Evaluation state atomically
```

The Mapping full oracle is
`fullPnrCost(FrozenModel, CandidateDecisions)`. Each Evaluation model owns its
full execution semantics. An incremental adapter is only an exact execution
optimization for that same model identity. A lower-fidelity predictor is a
different model identity, not an approximate adapter for a higher-fidelity
model.

Each exact `ResolvedEvaluationBindingRef` may create one ephemeral adapter with
`rebuild`, `probe`, `commit`, `discard`, and optional frozen route guidance.
`PnrCandidateDelta` and a borrowed read-only shadow candidate view are its only
change source. The adapter may not own, copy, replace, or independently mutate
`CandidateState`. A probe returns the exact provisional metrics and findings
requested by the objective. Mapping and every adapter must succeed before the
transaction can commit or discard under one decision. Runtime unsupported,
execution failure, or cancellation from a preflighted required hot binding
makes the attempt `Incomplete`; it cannot switch provider or assign a worst
candidate score.

Full checkpoints rebuild authoritative results from the same selected
decisions. They run at authoritative candidate initialization, every protocol
boundary that requires full selection, and finalization. Optional
consistency checkpoints do not consume semantic work or alter candidates.
Incremental/full disagreement is an internal oracle-drift failure, never a
candidate penalty or an invitation to repair caches in place.

## Final Closure And Verification

Final verification is not search. The immediately preceding final global
negotiated closure remains the last budgeted `TransportRoutingAction`: it may
change the candidate only through the ordinary `MoveTransaction`, exact owner
updates, objective decision, and atomic commit or rollback. Full owner
recomputation then checks the committed decisions. The independent verifier
only proves closure and admission; it never repairs or changes a candidate.

A selected Spatial candidate must complete that global routing Action, Mapping full
recomputation, zero all final `V`, and pass independent base verification and
exact `K` admission. A search policy may require a full `Q` oracle checkpoint
before selecting the candidate, but this is an ephemeral search protocol
rather than persistent Evidence or an Artifact validity condition.

`SpatialMappingBaseVerifier(D,T,F,S)` reconstructs intrinsic closure without
`FrozenModel`, `CandidateState`, `C`, `K`, history, an `InvocationManifest`, or
owner invocation records. In dependency order it checks exact predecessor
coupling; realization coverage and record totality; occurrence, context, port,
and refinement compatibility; residual edge coverage and route-tree
arborescences, including route-wide data-field capacity and independent
tag-field representability; memory binding, access, exposure, and Canonical
Service Schema legs; derived configuration; `ResourceUse`, capacity, buffers,
and tags;
ordered dataflow; and progress/deadlock closure.
Its artifact outcome is only `Valid` or `Invalid(typed diagnostics)`; search
infeasibility and budget outcomes are not artifact states.
If a supported verifier cannot establish the required progress proof, the
invocation ends as `Incomplete(proof_not_established)` before an artifact
outcome or publication. That is neither `Valid` nor an `Invalid` counterexample.

`SpatialMappingConstraintAdmission(D,T,F,K,S)` separately checks the exact
run's `K`. Rejection by `K` does not make an intrinsically valid artifact
base-invalid. Only after both checks pass does finalization assign canonical
ordinals and local IDs, write canonical Mapping bytes, derive Common identity,
publish atomically, and derive semantic `sw_configs`. Physical image encoding
is a later mechanical derivation owned by
`docs/spec-configuration-deployment.md`; PnR does not emit bitstream content or
own physical field layout.

System verification derives the one non-persistent
`SystemMappingClosureProjection` from exact `D`, `F`, complete SystemMapping,
and its exact SpatialMapping set.
`SystemMappingBaseVerifier(D,F,M,ExactSpatialMappingSet(M))` uses that shared
projection to verify coverage and typed references, end-to-end service and
path continuity, capacity and acquire/release closure,
tag/context/configuration continuity, and progress/deadlock closure. It does
not read `C`, `K`, an `InvocationManifest`, owner invocation records, or runtime
traces.
`SystemMappingConstraintAdmission(D,F,root_thread_launches,K,M)` applies the
required exact System `K` only after base verification. It also requires the
root launch set in `K` to equal the Mapping coverage root exactly.

System base verification returns only:

```text
Verified
Rejected(typed closure findings)
Incomplete(unsupported | proof_not_established)
InternalError
```

A proven closed wait set without a Fabric progress mechanism is the
`HardProgressViolation` closure finding and is `Rejected`; an observed
deadlock remains an Evaluation finding; failure to establish a proof is
`Incomplete(proof_not_established)`. Finite simulation without an observed
deadlock is not proof. Only `Verified` plus exact `K` admission can publish.

Formal Evaluation starts after publication because an `EvaluationRequest`
binds an exact finalized Mapping Artifact. The central `Promote` node acquires
Evidence, applies quality gates, and selects among published candidates.
Neither missing Evidence nor a failed quality gate retroactively changes
Mapping validity or Artifact identity.

Unsupported input, invalid input, proven infeasibility, no prepared seed,
budget exhaustion, interruption, and failure of a required pre-publication hot
Evaluation binding are typed Mapping invocation outcomes. They never publish
partial, rejected, degraded, or best-so-far Mapping artifacts. Failure of a
formal post-publication Evaluation is instead owned by that Evaluation
invocation and never invalidates an already finalized Mapping.

## Validation Anchors

Tests protect semantic anchors rather than implementation shape:

* exact Spatial five-input and System six-input coupling,
  including foreign and wrong-kind reference rejection, mechanical `C`
  derivation, and exact `K` profile/root matching;
* exact Spatial and System `K` root bindings, the three shared closed record
  variants, variadic relation arity, both complete projection catalogs with
  exact subjects, carriers, and cardinalities, the four persistent carrier
  encodings, and no persistent bitset form;
* empty-unrestricted behavior, projection-local equality closure, merged
  domain intersection, variadic Disjoint rewrite, cardinality-sensitive empty
  domains, outcome separation, pre-result subjects, derived hot indexes, and
  rejection of result-time subjects and extension escapes;
* deterministic aggregate freeze, MLIR-to-native projection, factorized
  domains, cache framing, native index capacity, and derived work-budget view;
* complete internal-edge accounting for configured FU, configured
  `fabric.mem`, temporal register-file absorption, and residual logical nets;
* endpoint-only A*, multi-sink route trees, explicit broadcast, checked route
  cost, PathFinder net order and termination, and all negotiation kernels;
* route-wide widening acceptance plus rejection of a narrowing bottleneck or
  attempted payload borrowing from tag bits;
* atomic Action commit and rollback across placement, routes, resources, and
  exact preflighted Evaluation adapters without candidate copying;
* stable logical slots across cache, retry, replicate, and resume; fixed seed
  attempts; central `Promote` separation; replay-stable annealing; focused
  closure; and exact-repair taxonomy;
* shared objective dimensions, CNF truth/deviation agreement, independent full
  `V/G` and `Q`, objective code, base verification, and exact admission;
* `ServicePlan` versus `ResourceUse` ownership, trigger/release and atomic
  activation derivation, progress outcome classification, and hierarchical/flat
  persistent-result equivalence.

Tests must not preserve container layout, printer whitespace, path insertion
order, a greedy or place-then-route baseline, objective weight matrices,
protocol implementation details, cache strategy, or solver internal shape.
