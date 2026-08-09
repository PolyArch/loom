# Mapping Artifact

This document is the schema authority for the persistent Dataflow-to-Fabric
Mapping artifact family. A Mapping artifact records one immutable selected
realization over exact finalized upstream artifacts. It never mutates or
reinterprets those artifacts.

Persistent identity and reference semantics are owned by
`docs/spec-mapping-identity.md`; imported Fabric-local target variants are
owned by `docs/spec-fabric-identity.md`. Observable verifier contracts are
owned by `docs/spec-mapping-verification.md`. The complete Spatial and System
MappingConstraintSet family, roots, projection catalogs, and admission algebra
are owned by `docs/spec-pnr.md`; they are not part of the `loom.mapping`
schema.

## Schema Family

All persistent Mapping objects belong to the single schema family
`loom.mapping`. The family has three profile-complete semantic roots:

| Semantic root | Schema version | Required root bindings | Top-level record families |
|---------------|----------------|------------------------|---------------------------|
| `mapping.tech` | `5.0` | Canonical Dataflow Program `D`, Fabric Hardware Description `F` | `ComputeRealization`, `MemoryRealization` |
| `mapping.spatial` | `5.0` | TechMapping `T`, Canonical Dataflow Program `D`, Fabric Hardware Description `F` | `ComputeBinding`, `MemoryEngineBinding`, `MemoryBinding`, `RouteTree`, `ResourceUse` |
| `mapping.system` | `5.0` | Canonical Dataflow Program `D`, architecture-only Fabric Hardware Description `F`, canonical SpatialMapping import table | `ExecutionBinding`, `ServiceRealization`, `ResourceUse` |

The root operation is the profile discriminator. A Mapping object does not
also carry a profile enum, a generic artifact root, or inactive optional
fields for other profiles. Each persistent MLIR object contains exactly one
semantic profile root. A fixed `builtin.module` may be used as a parser and
transport container, but it is not semantic content, an identity owner, or a
second profile discriminator.

The Mapping schema declaration is the single owner of the schema identity,
current version, fields, defaults, and canonical assembly rules. The root
version framing and Common `ArtifactSchemaDescriptor` are mechanically
derived from that declaration. Callers must not construct schema strings or
maintain parallel version facts.

The current schema is the complete `loom.mapping 5.0` contract with all three
roots. Version 5.0 requires an ordered-cardinality adapter realization to use
the exact Fabric-owned contract and its intrinsic release. A 4.0 resource use
could instead pair that actor with a one-cycle contract and collapse repeated
or zero productions into `AllOf(activeResults)`; this is not a complete
realization. The accepted cross-artifact relation therefore changes even
though the ResourceUse wire variants do not, so a 5.0 parser rejects every 4.0
root rather than reinterpreting it.

Version 4.0 replaced the optional single causal release point in 3.0
with one explicit closed release condition: intrinsic release or a canonical
nonempty conjunction of existing event points. This changes accepted canonical
content, so a 4.0 parser rejects every 3.0 root rather than treating one absent
or present point as an implicit condition. Version 3.0 replaced the
incompatible System message-route rule in 2.0:
one plan covers the exact applicable `(sink terminal, execution owner)` set,
rather than requiring one attachment for every static sink terminal, and a
message plan may be empty only when that applicable set is proven empty. These
changes altered the accepted canonical content, so a 3.0 parser rejected every
2.0 root rather than reinterpreting it. TechMapping retains its 3.0 payload
shape but uses 5.0 because `loom.mapping` has one family version, not
independent profile versions. `loom.mapping 1.0`'s
load/store-only `AccessEntry` and logical-memory-only operation-service owner
are superseded by the closed `MemoryOperationEntry` and fence-aware
ServiceRealization contract. An earlier draft assigned `1.0`, `1.1`, and `1.2`
according to the order in which profiles were discussed; that unpublished
history is retired.
Future minor-version upgrades must first parse and verify the source version
under its own schema, then use an explicit typed adapter to construct and
finalize a new artifact with a new identity. Text patching, implicit upgrades,
and retaining the old identity are invalid.

Mapping MLIR is finalized-only. Mutable builders and native search state are
not persistent Mapping objects. There are no `partial`, `rejected`,
`degraded`, draft-root, placeholder-record, or consumer-defined completeness
forms.

## Root Bindings

Every root declares each required exact upstream artifact identity exactly
once as a typed `UpstreamArtifactBinding`. Internal upstream references use
that root scope. They do not repeat complete digests and cannot use symbols,
paths, free-form kinds, or compatibility labels. The exact wire and reference
rules are specified by `docs/spec-mapping-identity.md`.

The binding order is semantic and fixed:

* `mapping.tech`: `D`, then `F`;
* `mapping.spatial`: `T`, then `D`, then `F`; and
* `mapping.system`: `D`, then `F`, followed by its canonical SpatialMapping
  import table.

For `mapping.spatial`, `D` and `F` are scoped-reference aliases, not competing
lineage. `T` remains the owner of its exact Dataflow and Fabric coupling, and
the verifier requires `T.D == D` and `T.F == F`.

For `mapping.system`, every imported SpatialMapping must bind the same exact
`D` and `F`. Its exact TechMapping predecessor remains transitive lineage;
SystemMapping does not repeat a TechMapping set.

## TechMapping Root

The `mapping.tech` root has the fixed semantic field order
`version`, `dataflow`, `fabric`, and `covers`, followed by one single-block
declarative record region. `covers` is a canonical non-empty set of
Dataflow-owned `GraphRef` values from `D`. The region contains only
`mapping.compute_realization` and `mapping.memory_realization` records.

Each realization owns one single-block declarative child-record region.
Neither root nor child regions have block arguments, SSA results, CFG
successors, symbol tables, or runtime terminators.

Coverage is closed over every graph in `covers`:

* every actor belongs to exactly one Compute Realization or Memory
  Realization;
* canonical addressed memory actors and fences belong only to Memory
  Realizations, and all other actors belong only to Compute Realizations;
* no realization crosses a graph-definition boundary;
* every canonical edge is classified exactly once as realization-internal or
  realization-external;
* every exposed software endpoint has complete typed physical
  correspondence; and
* graph boundaries and value, stream, control, state, and memory obligations
  are complete, with no implicit default realization.

Coverage is part of the TechMapping root. It is not a separate Mapping Scope
entity.

### Compute Realization

A Compute Realization owns one selected actor grouping and its selected
Fabric FU structural/capability template. Its persistent basis is:

* one artifact-global Mapping `EntityId`;
* one exact selected Fabric-owned `FabricFuCapabilityTemplateRef`; and
* complete `mapping.compute_actor` and `mapping.compute_boundary` child
  relations.

`mapping.compute_actor` records the exact Dataflow-owned `ActorRef`, selected
Fabric op, and complete ordered `operand_ports` and `result_ports` maps. Software port
ordinal is the array index and the selected physical operation-port ordinal
is the value. This representation covers ordinary operations, multiple
results, and legal narrower variadic mappings without a generic
correspondence record.

`mapping.compute_boundary` records each required exact actor-port to
owner-FU-boundary-port correspondence. Owner-relative ports are expanded from
the selected template and do not repeat the FU implementation identity in
each child.

A mapped actor result with no canonical consumers is not an exposed software
endpoint and therefore has no `mapping.compute_boundary` child. When the
selected capability does not suppress that physical production, the actor's
ordered result-port map and the selected template's exact edge to an FU output
boundary mechanically derive a required PE output `Discard`. That disposition
is finalized by SpatialMapping after it selects a concrete FU occurrence. It
is not a logical edge, residual net, route, entity, or additional persistent
TechMapping record. An unmatched selected-template edge is valid only for this
exact dead-result disposition; every other extra edge remains invalid.

The exact FU implementation is derived from the selected template owner. The
actor set is derived from the actor-relation domain. Exact software types,
constants, predicates, arity, and other semantic parameters remain owned by
`D`. The configured function, active ports, masks, and `sw_configs` are
derived from `D`, `F`, the selected template, and the ordered TechMapping
relations. They are not persistent fields.

The selected reference is encoded in the exact Fabric scope already bound by
the TechMapping root. It resolves to one canonical capability-template record
owned by one `FabricFuTemplateRef`. A later SpatialMapping may choose only an
FU occurrence whose exact Fabric definition relation names that owner.

The former Mapping-owned compute `EncodingId`, `EncodingRef`,
`EncodingDescriptor`, and copied configured-operation graph are retired with
no persistent replacement or compatibility shadow. Matching and
materialization may use removable hot caches keyed by the exact `D`, exact
`F`, selected capability-template reference, and correspondence, but those
caches are not artifact fields. Memory Realizations use the corresponding
Fabric-owned Memory Operation Engine template relation defined below; they do
not restore a generic or Mapping-owned encoding.

All Fabric operations use the same parameterized capability and match model.
For `dataflow.sync`, ordered all-of lane correspondence is expressed by the
ordinary actor operand/result maps and the mask is derived. Mux and demux
choice lanes use ordered maps while their runtime selector remains a Dataflow
operand. Constants and other parameterized operations use their registered
operation schemas and exact Dataflow semantics.

An earlier exact-mode enumeration and sync-specific Mapping-record model is
retired. Sync now uses the ordinary parameterized capability relation and
exact ordered TechMapping relations; masks and configured fields are derived.

### Memory Realization

A Memory Realization owns a selected implementation of a canonical software
memory subgraph. Its persistent basis is:

* one artifact-global Mapping `EntityId`;
* one selected `FabricMemoryEngineTemplateRef`; and
* complete `mapping.memory_actor`, `mapping.memory_graph_boundary`, and
  `mapping.memory_internal_edge` child relations.

`mapping.memory_actor` records the exact Dataflow-owned `ActorRef` for a read,
write, RMW, compare-exchange, or fence actor, the selected
template-relative memory-operation port and capability alternative, and
complete ordered operand/result port maps.
`mapping.memory_graph_boundary` records each required Dataflow token, value,
or control graph terminal to template-relative Operation Engine endpoint
correspondence. It never names a `MemoryExposureRef`, memref capability, Local
Memory Service, manager endpoint, or subordinate endpoint; those belong to
Spatial or System memory binding.
`mapping.memory_internal_edge` records the exact canonical software edge and
the selected template-relative Fabric internal connection that implements it.

An addressed actor reference mechanically supplies the nonpersistent
`CanonicalMemoryAccessView`; fence has no addressed-access view. The selected
port, capability alternative, and ordered port maps must satisfy the
Fabric-owned parameterized actor-contract and operation relation, including access form,
memory-element width, access-lane-shape projection, lane count, address and
data capacities, dynamic-mask capability, alignment, synchronization scope,
and the declared operation use-pattern domain. Total payload width alone is
not a semantic match.

The Operation Engine definition is the selected template owner. No concrete
memory occurrence, service, or dispatch target exists in TechMapping. Actor
sets are derived from the relation domain. TechMapping does not associate an
addressed graph actor with a logical memory root: that relation is contextual
to one rooted graph launch and belongs to SpatialMapping. The root does not
repeat implementation or actor lists. Actors sharing a Memory Realization do
not make their edges internal; only an exact selected internal-edge witness
does so.

Selected capability alternatives and internal connections must satisfy the
Fabric-owned parameterized actor-contract, access, alignment, narrow-access,
mask, fanout, port-role, use-pattern, and required MemoryConsistencyDomain
contracts. TechMapping stores the non-derived correspondence, not duplicate
operation semantics, access views, concrete contract values, transaction
decomposition, service selection, or a memory-service model. It does not
enumerate active port or connection subsets as semantic encodings.

### TechMapping Record Rules

The five role-specific child kinds are the complete child catalog:

```text
mapping.compute_actor
mapping.compute_boundary
mapping.memory_actor
mapping.memory_graph_boundary
mapping.memory_internal_edge
```

Child records have meaning only inside their owning realization and do not
receive `EntityId` values. Duplicate actor, actor-port, graph-port, or edge
keys are invalid; there is no last-wins rule. The schema has no generic
correspondence union, arbitrary property bag, or placeholder representation
adapter.

An edge internal to a Compute Realization is implemented by its configured FU
topology. A Memory Realization absorbs only edges named by exact internal-edge
witnesses. Every remaining canonical edge is an external logical obligation
derived from `D` and the exact TechMapping correspondences; TechMapping does
not persist a duplicate netlist.

## SpatialMapping Root

The `mapping.spatial` root begins with the fixed `T`, `D`, and `F`
`UpstreamArtifactBinding` fields and then one single-block declarative record
region. The region permits exactly five top-level record families, in
schema-owned family order:

```text
ComputeBinding
MemoryEngineBinding
MemoryBinding
RouteTree
ResourceUse
```

The profile preserves all TechMapping realization decisions. It cannot
regroup actors, rematch a capability, select another semantic realization,
replace a memory engine template or internal-edge witness, or modify
software-to-hardware correspondence.

The resolved PnR config view `C` affects search. `K` is an independent
canonical MappingConstraintSet Artifact used for invocation admission. Neither
changes the semantic content or identity of a selected SpatialMapping. The
`InvocationManifest` binds the exact
`K` ArtifactIdentity and admission result. `EvaluationEvidence` references
only its exact `EvaluationRequest` and does not become a second
derivation-provenance owner.

### ComputeBinding

Each TechMapping Compute Realization has exactly one `ComputeBinding`, keyed
by its exact `ComputeRealizationRef`. The record stores the selected
`FabricFuOccurrenceRef`, selected `InstructionContextRef`, and any
non-derived owner-local physical refinement assignments. The realization key
already identifies the record, so `ComputeBinding` has no Mapping-local ID.

An `InstructionContextRef` is exactly a Fabric PE occurrence plus a context
ordinal. It is not a schedule slot, optional sentinel, or independent entity.

### MemoryEngineBinding

Each TechMapping Memory Realization has exactly one `MemoryEngineBinding`,
keyed by its exact `MemoryRealizationRef`. It stores the selected
`FabricMemoryOccurrenceRef` and one `MemoryOperationEntry` child for every
canonical memory actor covered by the realization. It has no Mapping-local ID.
The selected occurrence must have an Operation Engine and its exact
`memoryEngineTemplate(occurrence)` relation must equal the template selected by
the Memory Realization. Template-relative ports, capability alternatives,
endpoints, and internal connections then project mechanically to concrete
occurrence-relative references.

`MemoryOperationEntry` is one closed child union:

```text
MemoryOperationEntry =
    AddressedOperation {
      actor_ref
      operation_placement
      uses : nonempty canonical array<AddressedOperationUse>
    }
  | FenceOperation {
      actor_ref
      operation_placement
      uses : nonempty canonical array<FenceOperationUse>
    }

AddressedOperationUse = {
  rooted_graph_launch_ref
  MemoryBindingRef
  dispatch_target : LocalMemoryServiceRef | ManagerEndpointRef
}

FenceOperationUse = {
  rooted_graph_launch_ref
  consistency_target : MemoryConsistencyDomainRef | ManagerEndpointRef
}

MemoryOperationUse =
    AddressedOperationUse
  | FenceOperationUse
```

The addressed variant covers read, write, RMW, and compare-exchange. Fence has
no fabricated logical-memory binding. Both variants use the same closed
placement reference:

```text
MemoryOperationPlacementRef =
    Spatial  { FabricMemoryOperationPortRef }
  | Temporal { FabricMemoryOperationContextRef }
```

The Spatial variant derives its static context from the physical port. The
Temporal context reference contains that exact port plus a valid
Fabric-declared resident-context ordinal. Placement is selected exactly once
for the definition-level `ActorRef`. Each use stores only one
`RootedGraphLaunchRef`; the parent actor and child launch mechanically form the
existing Dataflow-owned `ContextualActorRef`. The use inventory must contain
exactly one row for every rooted launch whose callee graph owns the actor, in
canonical rooted-launch order, with no missing, duplicate, foreign, or
wrong-graph row.

The collection of operation-use targets and corresponding ExposureEntry
targets is the persistent owner of Mapping's selected `C_dispatch`. The
verifier checks each selection against Fabric-owned `H_dispatch`.

For an addressed use, the selected dispatch target and referenced
`MemoryBinding` must agree. A `LocalMemoryServiceRef` requires a
`LocalRegion` target owned by that exact local service. A
`ManagerEndpointRef` requires a `BoundaryProxy` target. The manager endpoint
remains the dispatch path and is never reclassified as a memory service or
service region.

For an addressed use, Dataflow composition resolves the actor's exact memory
capability in that rooted launch to one `LogicalMemoryRootOrViewRef`. The
referenced `MemoryBinding` must name that exact logical memory and contain the
selected logical interval. Two uses of one actor may therefore share a
placement while selecting different bindings and dispatch targets. Fence uses
have no `MemoryBinding`.

An Operation Entry does not select an internal connection again. The exact
concrete connection is derived from the TechMapping internal-edge witness and
the selected occurrence-to-template relation. Any additional row-local source
selection would duplicate the TechMapping witness and is invalid.

The exact MemoryConsistencyDomain is derived from the selected target and
Fabric use pattern. No operation entry stores a duplicate domain reference.
Every addressed atomic operation and every fence must resolve to exactly one
compatible domain. A fence domain must cover every memory effect constrained
by the actor's incoming and outgoing causal edges; Mapping cannot synthesize a
hidden multi-domain fence. Entries do not store Physical Tags, access form,
element or vector geometry, mask mode, concrete actor-contract fields, derived
`ElementAccessOnly | VectorAccessOnly | ElementAndVectorAccess` class,
operation-table rows, dynamic consistency state, service-transaction
decomposition, dispatch selectors, provider decode, response tracking
configuration, or raw `sw_configs`.

For a Temporal placement, configured row input matches and output writes are
derived per actor role from the selected TechMapping port correspondence,
selected internal sources, and Physical Tag assignments on the real tagged
writers or ingresses. One operation entry does not own a common tag. The
operation kind and capability alternative are configured row state rather
than runtime content-match fields.

### MemoryBinding

A `MemoryBinding` is one atomic relation:

```text
LogicalMemoryInterval =
    Whole
  | ByteRange { offset_bytes : u64, size_bytes : positive u64 }

MemoryBindingTarget =
    LocalRegion {
      service_region_ref : FabricMemoryServiceRegionRef
      physical_offset_bytes : u64
    }
  | BoundaryProxy

one LogicalMemoryInterval -> one MemoryBindingTarget
```

It stores one artifact-global Mapping `EntityId`, the typed logical
memory/view reference, logical interval, and exactly one target variant. The
`EntityId` is also the persistent identity of a `BoundaryProxy`; no separate
proxy entity or reference kind exists.

`LocalRegion` names one exact region of a Local Memory Service in `F` and one
unsigned physical byte offset within that region. The logical interval must
have a finite byte extent, and its entire translated range must fit in the
selected service region. `Whole` is legal for local placement only when the
exact Dataflow root or view has a finite statically derivable byte extent.

`BoundaryProxy` states only that the logical interval crosses the SpatialCore
memory-service boundary. It stores no Fabric service, region, endpoint,
address transform, provider, or system route. A dynamically unbounded
`Whole` interval therefore requires `BoundaryProxy` unless an exact
pre-Mapping specialization has already produced a finite bound. SystemMapping
derives the existing operation-service obligation from the logical owner and
interval, then selects its real provider region and address transform.

Multiple disjoint records represent partitioning. Replication, mirroring,
coherence, or overlapping placement requires an explicit Fabric composite
service or transform; overlapping rows cannot imply those semantics. A
different manager dispatch path alone does not create another MemoryBinding
because dispatch remains owned by the addressed or exposure child.

Each MemoryBinding owns `ExposureEntry` children ordered by software boundary
key. An exposure stores the software memory-output obligation, selected
subordinate or provider terminal, and one
`LocalMemoryServiceRef | ManagerEndpointRef` dispatch target. It has no
independent ID and does not form a top-level ExposureBinding family. Its
target must agree with its owning MemoryBinding by the same LocalRegion versus
BoundaryProxy rule as an addressed MemoryOperationEntry.

### RouteTree

Each `RouteTree` is keyed by the `SpatialLogicalNetKey` mechanically derived
from a canonical software net and the selected TechMapping boundary
correspondences. It has no Mapping-local ID. Its only wire form is a flat tree:

```text
RouteTree {
  SpatialLogicalNetKey
  root_endpoint
  nodes [
    node 0 = root
    node N = {
      parent_node_ordinal
      incoming_physical_traversal
      physical_refinement_assignments
    }
  ]
  sink_attachments [
    sink_obligation_key -> node_ordinal
  ]
}
```

The root is ordinal zero and has no parent or incoming traversal. Every
non-root endpoint is derived from its incoming traversal destination, and the
traversal source must equal the parent endpoint. Sink endpoints are derived
from their attached node; a zero-length route attaches directly to the root.
One physical endpoint cannot appear as two nodes, structurally excluding
reconvergence. Shared trunks appear once, and fanout is legal only where
Fabric explicitly supports it.

Canonical node ordinals use depth-first preorder from the root, with children
ordered by the full physical-traversal semantic key. Sink attachments are
ordered by sink-obligation key. Search insertion order, reached-endpoint
copies, selected-edge bitsets, switch rows, resource claims, and allocator
state are not persistent.

### ResourceUse

SpatialMapping has one root-level flat `ResourceUse` family:

```text
ResourceUse {
  owner_ref
  use_site_ref
  relative_activation
  typed_pattern_parameters
  typed_sharing_assignments
}
```

Spatial `relative_activation` is a closed graph-local event relation:

```text
SpatialActivityEventRef =
    ActorTransition {
      ActorRef
      transition_case_ordinal
    }
  | Produced { CanonicalGraphProducerEndpointRef }
  | Consumed { CanonicalGraphConsumerEndpointRef }

relative_activation:
  trigger = SpatialActivityEventRef + optional guaranteed offset
  release = Intrinsic
          | AllOf(nonempty canonical set of
                  SpatialActivityEventRef + optional guaranteed offset)
```

`transition_case_ordinal` resolves in the exact actor's canonical
`ActorHandshakeCase` projection from OperationSchema. It is not a new event
entity, operation registry, firing mode, or simulator-private transition.
Produced and Consumed name the exact existing graph terminal event; the two
directions remain distinct even when they name opposite ends of one canonical
edge. Every referenced actor or endpoint must belong to inherited TechMapping
coverage and must be causally applicable to the selected owner's use; a
MemoryBinding shared by several covered graphs does not acquire a synthetic
single-graph owner.

A guaranteed offset is an owner-typed value decoded and re-encoded by the
exact selected Fabric use-pattern timing provider. It is legal only when that
provider proves the offset for every admitted execution. A provider without
such a codec admits only the absent form; Mapping cannot interpret an integer
as cycles, infer a clock domain, or persist an absolute time.

`AllOf` members are sorted by their complete canonical event-point bytes and
must be unique. They refer to occurrences causally applicable to the same
dynamic use selected by `trigger`; an empty conjunction, duplicate member,
foreign graph event, or unrelated occurrence rejects. The effective release is
the later of the Fabric use pattern's intrinsic release eligibility and
completion of every member. A one-member conjunction is the ordinary
single-event case and receives no shorthand encoding or aggregate event ID.

For every exact built-in operation contract that Fabric classifies as
requiring active-result handoff, a compute ResourceUse triggers on the selected
actor transition and derives one `Produced` member for every logical result in
that exact `ActorHandshakeCase::activeResults`. The closed set currently
contains the registered one-cycle elastic pattern and the exact registered
`LoopStream` pattern. The set describes the complete held tuple. Mapping
consumes this Fabric-owned classification; it cannot select one result,
maintain a family table, invent an all-results event, or release the use from
authoring order.

A fixed-vector parallelize or serialize capability carrying the exact
portable ordered-cardinality ResourceContract instead uses `Intrinsic`
release. Its one ResourceUse still triggers on the selected
`ActorTransition(case ordinal)`, but the Fabric-owned internal production
sequence decides retirement. Mapping must not derive an `AllOf` condition from
the legacy active-result union: an all-zero serialize mask produces no event,
and a partial close repeats the phase result without creating a second static
result ordinal. The previously used exact one-cycle elastic record remains
valid when imported as a generic Fabric record. Pairing it with either
ordered-cardinality actor rejects the prospective TechMapping capability seed
as `CapabilityInadmissible`, so no ResourceUse is created. If architecture RTL
lowering is invoked directly on that family/contract combination, the portable
provider returns typed `Unsupported`. No Mapping release projection is defined
for the mismatched pair or another unregistered contract shape.

Pattern parameters and sharing assignments are canonical positional arrays in
the exact use site's closed owner schemas. Each value is encoded by that
position's owner codec, adopted as an immutable typed value, re-encoded, and
required to match byte-for-byte. Unknown, missing, extra, malformed, or
noncanonical values reject. These arrays are not keyed property bags, generic
attributes, or a second declaration of the Fabric parameter domains. An empty
owner schema has exactly one valid value: the empty array.

Its closed owner union is a `ComputeBindingKey`, `MemoryEngineBindingKey`,
`MemoryBindingRef`, or `(RouteTreeKey, RouteNodeOrdinal)`. The use site must
resolve a Fabric-owned use pattern within the occurrence, service, or
traversal already selected by that owner. Fabric owns the resource vector,
capacity, duration, latency, initiation interval, optional owner-defined commit
transition, and parameter schema; Mapping supplies only workload-specific typed
values and sharing assignments.

Physical Tags are stored only as typed sharing assignments at real temporal
writers or ingress points. Instruction contexts remain owned by bindings.
Static claims mechanically implied by a selected traversal are not duplicated
as ResourceUse records.

Each maximal tagged continuity segment has exactly one assignment ResourceUse
at its origin:

* an actor-result route source uses its selected ComputeRealization or
  MemoryRealization owner;
* a graph-ingress route source uses the RouteTree root node owner; and
* a boundary writer or rewriter uses the destination RouteTree node owner.

All three trigger on `Produced { SpatialLogicalNetKey }` and select the exact
Fabric assignment pattern projected for the origin endpoint. They carry no
parameters and exactly one Physical Tag sharing value. Transport-only tagged
nodes, switch rows, lookup tables, and downstream matcher state derive that
value and do not persist another assignment. A boundary remover terminates the
incoming segment and creates no output assignment.

Strict import independently reconstructs continuity segments from the exact
RouteTree and Fabric boundary contracts. It requires every origin once,
rejects an extra or incorrectly owned assignment, decodes the sharing value
through the selected Fabric pattern, and rejects equal values for distinct
segments that intersect one Fabric-owned match domain. Values may be reused
across disjoint match domains.

For a Temporal PE compute binding, Mapping derives operand-queue uses without
another scheduler or event catalog. An external actor input enqueues at that
exact `Consumed { ActorTokenOperandRef }` event. A transition that consumes an
external actor input dequeues its corresponding logical operand queue at the
same `ActorTransition` event as the selected operation use. Inputs internal to
one Compute Realization do not cross the PE operand-queue boundary and derive
no queue use. Spatial PE bindings derive only their operation uses. These
records are mandatory owner projections: omission, duplication, a different
trigger, or a use pattern outside the selected instruction context rejects.

### Physical Refinement

Any semantic-preserving physical or QoR choice not derived from the selected
occurrence, traversal, or service is an owner child:

```text
PhysicalRefinementAssignment {
  FabricPhysicalRefinementDomainRef
  normalized_typed_value
}
```

This child is not a sixth record family or a generic property bag. Fabric owns
the domain, value type, allowed values, encoding relation, and proof that the
choice preserves software semantics. Active singleton domains are derived and
omitted. Active non-singleton domains require one explicit assignment,
including when the selected value equals a Fabric default.

### Spatial Derived State

Active masks, configured-function copies, programmed-configuration keys,
resource claims, continuity segments, Tag interference graphs, switch rows,
memory operation tables including derived access projections and mask-source
selectors, the selected combinational handshake graph and its cycle-check
scratch, raw `sw_configs`, bitstreams, cost vectors, search history, statistics,
and transaction journals are deterministic projections or external records.
They are not SpatialMapping semantic content.

## SystemMapping Root

The `mapping.system` root has this fixed semantic order:

```text
mapping.system {
  exact Canonical Dataflow Program UpstreamArtifactBinding D
  exact Fabric Hardware Description UpstreamArtifactBinding F
  canonical SpatialMappingImportTable imports
  canonical non-empty root_thread_launches
  single-block declarative record region {
    ExecutionBinding
    ServiceRealization
    ResourceUse
  }
}
```

The record region has no block arguments, SSA results, CFG successors, symbol
table, or runtime terminator. The root launch set is the only persistent
coverage root and is a canonical non-empty set of Dataflow-owned
`RootThreadLaunchRef` values from `D`. Thread definitions, reachable static graph
launches, graph definitions, channels, memory obligations, and
external-boundary obligations are derived from its closure in `D`; competing
scope lists are invalid.

The derived root-thread-launch closure `R`, System search domain `H`, and
resolved config `C` affect construction and search. A separate immutable
System MappingConstraintSet `K` is required for admission under the contract
owned by `docs/spec-pnr.md`; this document defines no competing constraint
contract. `K` does not enter
SystemMapping semantic content or identity, and the `InvocationManifest`
binds its exact ArtifactIdentity and admission result. `EvaluationEvidence`
references only its exact `EvaluationRequest` and does not become a second
derivation-provenance owner.

The System root binds Fabric architecture and the exact Transport
Architecture, not an AXI, TileLink, CXL, or other Interconnect Implementation.
Implementation selection belongs to HardwareImplementation, simulation
binding, and Deployment. Replacing an implementation while preserving the
same architecture contract and ConfigurationABI neither changes this Mapping
nor the derived configuration-image identity.

### Exact SpatialMapping Imports

The exact selected SpatialMapping set is the finite unique range of normalized
`B_graph` over all reachable static graph launches and their legal logical
may-domain points:

```text
ExactSpatialMappingSet(M) =
  unique { B_graph(launch, point)
           | launch is reachable from root_thread_launches
           | point belongs to launch's legal logical domain }
```

The finite `B_graph` clause catalog may target only a finite set of immutable
SpatialMapping identities, so the range remains statically computable even
when dynamic invocation count is unbounded. There is no fixed minimum,
maximum, or equality-to-AccCore cardinality rule; the import count is exactly
the size of this finite unique range.

The canonical writer derives the SpatialMapping `UpstreamArtifactBinding`
import table from exactly that set, removes duplicates, and orders entries by
complete `ArtifactIdentity`. `B_graph` uses compact
`SpatialMappingImportRef` ordinals assigned only after this ordering. The
table is a reference-resolution and serialization structure, not a separately
editable selected-set authority.

The parser and verifier reject a missing, extra, duplicate, unreachable, or
foreign-`D/F` import. An InstructionCore-only mapping still has a non-empty
root launch set, but may have no reachable graph launch; in that case
`B_graph`, the exact set, and the import table are all empty. No dummy
SpatialMapping is required.

### ExecutionBinding

`ExecutionBinding` has exactly two typed variants with natural structural
keys:

```text
ThreadExecutionBindingKey = RootThreadLaunchRef
ThreadExecutionBinding    = key + BindingRelation<AccCoreOccurrenceRef>

GraphExecutionBindingKey  = RootedGraphLaunchRef
GraphExecutionBinding     =
  key + BindingRelation<SpatialMappingImportRef>
```

Every root thread launch has exactly one ThreadExecutionBinding. Every
reachable static graph launch in each root context has exactly one
GraphExecutionBinding. `RootedGraphLaunchRef` is the Dataflow-owned structural
pair of that root launch and static graph-launch site; Mapping does not define
another tuple. These records do not receive Mapping-local IDs.

Each relation is exactly one closed variant:

```text
BindingRelation<T> =
    PresburgerPartition<T>
  | StableKeyLookup<T>
```

The relation input signature and legal may-domain are owned by `D`.
Presburger cells use canonical integer sets over Dataflow-owned coordinates,
launch parameters, and stable logical-item components. Stable-key lookup uses
a Dataflow-owned stable-key tuple projection and finite exact key rows. A
relation is total and single-valued over its legal may-domain.

Presburger cells must be disjoint and lookup keys unique. A default denotes
the legal-domain complement: it is required exactly when that complement is
non-empty and forbidden when the complement is empty. Canonicalization
intersects entries with the may-domain, removes empty and default-equivalent
entries, rejects overlap or conflicting keys, merges canonical sets with the
same target, and sorts by canonical set or key bytes plus the complete target
semantic key.

`B_thread` and `B_graph` remain separate typed functions. For every graph
event point, the selected SpatialMapping must cover the callee graph and its
target SpatialCore parent must belong to the AccCore selected by `B_thread`.
ExecutionBinding owns only where computation executes; it owns no service
route, capacity, or relative-time facts.

Mapping 5.0 consumes the Fabric-owned rule that each AccCore has exactly one
InstructionCore context. Its `InstructionCoreContextRef` is mechanically
derived through the framing owned by `docs/spec-fabric-identity.md`.

`B_thread` selects only the AccCore. InstructionCore-resident ResourceUse
records reference the derived context and own event-relative occupancy; they
cannot select another execution target. `InstructionCoreContextRef` is a
different typed domain from the temporal-PE `InstructionContextRef`.
Their `use_site_ref` resolves exactly one Fabric-owned InstructionCore
`UsePattern` under that derived context; it is not a second target, scheduler,
or generic resource record.

### ServiceRealization

There is one ServiceRealization for every system service obligation in the
derived closure. Its key is the exact `SystemServiceObligationKey` owned by
`docs/spec-mapping-identity.md`.

A transfer obligation key contains only one exact
`CanonicalProducerTerminalRef`. The exact Dataflow program, the root's
canonical non-empty root-thread-launch set, and that producer mechanically
derive one canonical sorted unique non-empty sink-terminal set. The sink set
is not copied into the key. This static universe does not require each sink to
have a `source_map` preimage at every producer point. Channels, graph-launch
transfers, external messages, and multicast use this rule; multicast sinks
with one producer remain one family. Merge, zip, reorder, and reduction
require an explicit Dataflow actor.

For a channel obligation, the Canonical Dataflow Program remains the sole
owner of `source_map` and flat dynamic message correspondence. Mapping does not
store an activation pairing, message ordinal, epoch, or segment record. It must
prove that every selected route and sharing choice preserves the branch-local
producer and consumer sequences under overlapping endpoint instances, using
serialization, independent contexts or queues, or a Fabric-supported
deterministic reorder mechanism as required. Physical Tags are local
resource-sharing assignments and cannot serve as launch or message identity.

The operation-service variant is the closed minimal owner anchor defined by
`docs/spec-mapping-identity.md`. The exact Dataflow program derives the
logical-memory variant's complete typed addressed-operation member set and
separate complete `MemoryExposureRef` set. An exposure is a capability
boundary, not a service member, and therefore has no request or response leg.
Memory access and exposure, cache or proxy service, and external providers use
the same logical owner rather than parallel service-owner families. The fence
variant anchors one static fence actor family; the actor and Canonical Service
Schema derive its exact `FenceContract` and the Dataflow program derives its
reachable contextual members. Member sets, exposure sets, and contracts are
not copied into the key.

Each ServiceRealization has one or more owner-local plans and a complete plan
selection relation:

```text
ServiceRealization {
  SystemServiceObligationKey
  plans: [ServicePlan]
  plan_selections: [ServicePlanSelection]
}

ServicePlanSelection {
  ServicePlanSelectionKey
  relation: BindingRelation<ServicePlanOrdinal>
}

ServicePlan {
  service_target_bindings
  transfer_leg_realizations
  physical_refinement_assignments
}
```

A plan may contain no child only for `MessageTransfer` over an exact selection
range whose applicable sink-owner set is empty. Such a plan represents no
physical transfer and contains no sinkless `TransferLegRealization`. Every
other plan has the complete non-empty child set derived from its obligation,
Canonical Service Schema, and selected targets.

`ServicePlanSelectionKey` and its closed member-or-exposure anchor are owned by
`docs/spec-mapping-identity.md`. The exact Dataflow program derives the
complete anchor set. There is exactly one non-empty selection row for every
reachable `(anchor, ExecutionContextKey)` pair and no other row. For one
anchor, the row domains are disjoint and their union is exactly the anchor's
legal may-domain. The relation in each row is total and single-valued over that
row's context-restricted domain. Across all rows, the finite unique union of
relation ranges is exactly the ServiceRealization's canonical plan-ordinal set;
a missing selection, unreachable row, or unselected plan is invalid.

For the singleton `MessageTransfer` anchor, relation inputs are the producer
event's exact Dataflow-owned `EventLogicalProjection` and, for DynamicWork,
its separately owned stable-item projection. The execution context is derived
without a new transfer-context key:

* a root-thread boundary transfer uses the root's Instruction context;
* a graph-launch boundary transfer uses the rooted graph's Spatial context;
* a thread-channel producer uses its root's Instruction context; and
* a graph-stream producer uses its rooted graph's Spatial context.

Root start and value-input sources are fixed HostCore/runtime endpoints and
their sinks belong to the derived Instruction context; root completion uses
the reverse ownership. Both terminals of a graph-launch boundary transfer are
owned by the selected AccCore, with the InstructionCore/SpatialCore direction
derived from the Dataflow transfer kind. Each channel consumer's exact logical
point and execution binding are derived from the Dataflow-owned `source_map`;
multicast does not copy a sink-context tuple into Mapping. For producer point
`p`, Mapping derives the canonical applicable set

```text
ApplicableMessageSinks(p) = unique sorted {
  (sink_terminal, execution_owner(q))
  | q is in sink_terminal's consumer domain
  | source_map_sink_terminal(q) = p
}
```

`execution_owner(q)` is the fixed HostCore/runtime owner or the AccCore
mechanically selected by the applicable `B_thread` and `B_graph` values. A
plan-selection range may target one plan only when this complete set is
constant throughout the range. The plan contains one route sink attachment
for every pair. The same static terminal may therefore occur more than once
when its consumer points execute on distinct owners; several points selecting
the same terminal and owner collapse to one physical attachment. A terminal
with no preimage at `p` is absent. If the complete set is empty, the selected
message plan is the empty plan defined above. A different applicable set
requires another plan and relation range, not a union of endpoint domains
inside one route.

An addressed-memory or fence member anchor and a memory-exposure anchor use
their exact Dataflow-derived contextual logical input signature and legal
may-domain within the rooted graph launch. Their context is the Spatial
context derived from the applicable `B_thread` and `B_graph` results. This
reuses the execution-binding functions; the selection relation chooses only an
owner-local plan ordinal.

A plan element referenced by ResourceUse has the structural reference:

```text
ServicePlanElementRef =
  (ServiceRealizationKey, canonical plan ordinal, typed element key)

ServicePlanElementKey =
    TransferLegElement { CanonicalServiceLegKey }
  | MemoryRegionElement {
      LogicalMemoryRootOrViewRef
      LogicalMemoryInterval
      FabricMemoryServiceRegionRef
      ordered array<SystemServiceTransformRef>
    }
  | ConsistencyElement {
      FenceActorFamilyRef
      MemoryConsistencyDomainRef
    }
```

The typed element key is exactly the natural structural key of the referenced
closed `ServicePlan` child. A `MemoryRegionElement` deliberately excludes its
derived exposure children, and no element key copies a selected use pattern,
capability ordinal, execution context, or plan-selection predicate. A service
leg always uses the member-relative structural key owned by
`docs/spec-mapping-identity.md`. Its member reference selects the exact
Dataflow-derived operation or transfer member. The Canonical Service Schema
owns the local leg ordinal's direction, payload, completion, and ordering
meaning. No flattened artifact-global leg ordinal or copied operation list is
another authority. These structural objects receive no EntityId.

`ServiceTargetBinding` is one closed child union:

```text
ServiceTargetBinding =
    MemoryRegionTarget {
      logical service interval
      selected Fabric service region
      selected_transform_path : ordered array<SystemServiceTransformRef>
      exposures[] {
        MemoryExposureRef
        selected subordinate/provider terminal
      }
    }
  | ConsistencyTarget {
      FenceActorFamilyRef
      selected MemoryConsistencyDomainRef
    }
```

The `MemoryRegionTarget` children for one addressed subject form one complete
target plan, not independent region choices. Each child carries the complete
source logical interval and identifies one terminal branch. For every source
address, the composed Fabric transform contracts select exactly one child and
place its transformed address inside that child's selected region. A
`StaticInterleave` therefore produces one child for each distinct
`(transform path, terminal region)` branch group reached by a non-empty output
ordinal. Several output ordinals may collapse to one child only when Fabric
derives that same group for all of them. `CoherentMemory` instead derives each
branch domain from the input side of its selected region correspondence and
the region-relative address map to the output side; unused coherent provider
alternatives are absent. Repeated source intervals do not mean that every
address reaches every child. Output ordinals, correspondence members, strided
subsets, and transformed intervals are derived from Fabric and are not copied
into Mapping.

The transform path stores only non-derived selection. It is empty when the
bound endpoint, explicit Fabric MemoryService connections, and complete target
branch set uniquely imply the path, including the direct identity case.
Otherwise each child stores the exact ordered path of Fabric-owned
`SystemServiceTransformRef` values needed to disambiguate its service chain.
Fabric remains the sole owner of every transform's closed kind, parameters,
ordered endpoint relation, and region correspondence; Mapping never copies an
offset, mask, interleave rule, or coherence relation. Between consecutive path
elements, and before the first or after the last element where applicable, the
verifier follows only explicit Fabric MemoryService connections. The complete
target plan must honor each selected transform's exact ordered subordinate
inputs and manager outputs through the next connection or a terminal branch
group. The transform contract decides which outputs are collective and which
are alternatives: every non-empty `StaticInterleave` output domain is covered,
while `CoherentMemory` selects the canonical subset of correspondence branches
whose input-region domains cover the source interval exactly once. A transform
sequence uniquely implied by the bound endpoint and complete branch set is
derived and must be omitted rather than redundantly persisted.

A memory-region target owns every exposure child that selects that region.
The child is keyed by its `MemoryExposureRef` and provider terminal and has no
`ServiceMemberRef`, service leg, or independent ID. Missing, extra, duplicate,
or wrong-owner exposure children are completeness failures.

The selected service region or consistency domain must belong to the explicit
service/transform closure rooted at the System service endpoint bound by the
corresponding memory `spatial_attachment`. A target binding selects within
that closed domain; it never selects or replaces the endpoint itself.

A fence plan contains exactly one `ConsistencyTarget`; its selected domain
must cover all constrained effects in that execution context. A
`TransferLegRealization` binds one transfer leg derived from the Canonical
Service Schema to a flat `RouteTree` over system physical traversals. The
route tree selects transport terminals only within the role-selected carrier
sets of the exact Fabric-bound endpoint pair inside the owning attachment row;
it chooses neither that pair nor the row's System service endpoint.
Protocol packets, flits, headers, concrete virtual-channel encoding, and
implementation-specific bus encoding remain owned by the selected interconnect
implementation.

Every materialized `TransferLegRealization` has at least one sink attachment.
For `MessageTransfer`, the attachment's semantic key is
`(SystemTransferTerminalKey, execution owner)`. The owner is derived from the
attached route-node endpoint and exact Fabric; it is not a persistent field.
Repeating one terminal is legal only for distinct derived owners, while a
duplicate terminal-owner pair is invalid. Non-message service legs retain the
terminal key as their sink-attachment key. Canonical System sink order uses
this complete derived key, so authoring order is never semantic.

For `MessageTransfer`, the terminal domain is derived directly from matching
transport-plane service endpoints. For a memory or fence leg, the terminal
domain is derived from the exact three-reference memory-plane
`spatial_attachment`: its Module/occurrence endpoint pair and exact System
service endpoint. Canonical leg
direction and endpoint roles select one endpoint for each terminal, then that
endpoint's Fabric-owned `ServiceLegCarrierAttachment` row supplies the carrier
set for the kind and schema-local leg ordinal. The pair's System service
endpoint remains the sole capability authority even when the selected carrier
row belongs to the occurrence endpoint. The selected `RouteTree` stores only
its existing transport
terminal references and traversals. It does not copy the attachment row,
memory endpoint, capability domain, payload, width, or protocol, and no new
ServiceRealization child kind is introduced.

One service-leg `RouteTree` is shared by the leg's ordered independent
`ServiceValue` tokens. Its terminals and every traversal must satisfy the
nonpersistent maximum-width envelope defined by the
[Fabric System service-leg carrier contract](spec-fabric-system-adg.md#service-leg-carrier-attachment).
Mapping derives and verifies that envelope from the exact upstream owners; it
does not persist a width, packed tuple, role-specific route, or field layout.

In a finalized SystemMapping, each selection row uses the closed
`ExecutionContextKey` owned by `docs/spec-mapping-identity.md`. A Spatial
context uses the paired `B_thread` and immutable `B_graph` targets; an
Instruction context uses only `B_thread`. The same SpatialMapping semantic
target paired with two AccCore occurrences forms two keys because their exact
Fabric attachments and service endpoints may differ. None of those derived
Fabric facts is copied into the key or plan. This persistent key does not
constrain how System PnR represents a mutable flat candidate before its
SpatialMapping identities exist. Relations may reference only the anchor's
derived typed input slots and cannot persist another projection or reinterpret
input order. Plans have no `EntityId`; the finalizer sorts and deduplicates
complete plan semantic keys, assigns owner-local ordinals, rewrites every
selection relation to those ordinals, and sorts selection rows by their
complete semantic keys.

ServiceRealization is the only SystemMapping family for selected system
routes, physical buffers, target service regions, address transforms, and
mapping-visible service refinements. There are no parallel ChannelRoute,
MulticastRoute, System MemoryBinding, ExternalBinding, TerminalBinding, or
generic service-graph families. It does not persist modification order,
reads-from, synchronizes-with, sequentially-consistent order, queue contents,
or any other dynamic MemoryConsistencyDomain state.

### System ResourceUse

SystemMapping uses the same closed ResourceUse shape as SpatialMapping.
Its closed owner union is:

```text
SystemResourceOwnerRef =
    InstructionExecutionResourceOwnerRef {
      RootThreadLaunchRef
      InstructionCoreContextRef
    }
  | ServicePlanElementRef
```

The root-thread reference is the exact `ThreadExecutionBinding` key, and the
context must be the one mechanically derived from an AccCore selected by that
binding. Service-plan uses reference the exact child key defined above.
Applicability is mechanically derived from the owning execution-binding or
contextual plan-selection rows. A ResourceUse cannot copy that predicate or
plan choice, nor select a different target, context, configuration, route, or
service. Its use site must resolve a Fabric-owned use pattern exposed by the
selected owner.

Its relative activation has one trigger and one typed release policy:

```text
relative_activation:
  trigger = EventFamilyKey + optional guaranteed offset
  release = Intrinsic
          | AllOf(nonempty canonical set of
                  EventFamilyKey + optional guaranteed offset)
```

`EventFamilyKey` is exactly the Dataflow-owned closed union of transfer events
and rooted contextual actor transitions defined by the Closed Structural
Reference Catalog in `docs/spec-compiler-part-3-dfg.md`. The latter is the
rooted form of the same OperationSchema-owned actor-transition fact used by
Spatial ResourceUse, not a service issue ID or a second event catalog. The key
contains no concrete coordinates, launch-parameter values, copied logical
projection, or Mapping-local event ID. Every trigger and causal release
imports the exact Dataflow-derived `EventLogicalProjection`; runtime binds
concrete values for that schema when the corresponding occurrence is observed.
Two records referring to the same static event therefore use one key, while
context and parameter relations own any legal variation across its logical
domain.

System `AllOf` uses the same canonical nonempty, sorted, unique conjunction and
effective-release rule as the Spatial form. Its members are System event
points and receive no Mapping-local aggregate identity.

InstructionCore occupancy for one root/context pair triggers on the consumed
root-start transfer event and uses the produced root-completion transfer event
as its causal release. An addressed-memory or fence service use triggers on
the unique issue transition of its exact contextual actor. The Canonical
Dataflow memory contract defines that transition commit as issue of one
logical operation; the selected Fabric UsePattern supplies intrinsic
completion. A missing or non-unique issue transition rejects finalization.

For every addressed or fence member selected by a plan, each independent
Fabric ResourceContract required by the selected service capability and
consistency target contributes exactly one ResourceUse. An addressed service
use is owned by its exact `MemoryRegionElement`; a fence use is owned by its
exact `ConsistencyElement`. The selected `use_site_ref` must be one admissible
pattern of a matching capability on the derived bound provider. When several
patterns remain legal, PnR selects one and persists that choice only as the
ordinary `use_site_ref`; it never persists the capability ordinal. An
interleaved or coherent target branch is applicable only on the source-address
subset mechanically derived from the selected Fabric transform contract.

A `MemoryExposureRef` identifies a capability crossing, not a dynamic service
request, and Dataflow deliberately defines no event occurrence for it.
Selecting an exposure target therefore creates no ResourceUse by itself.
Actual external accesses require an independently specified invocation event
domain before they can create occupancy; Mapping must not approximate them by
holding a provider for the whole graph launch. Static claims implied by a
selected transport traversal likewise remain derived and are not duplicated.

Offsets are legal only when guaranteed by the Fabric service contract.
`Intrinsic` uses the Fabric pattern's finite or periodic completion contract.
A causal release holds the resource until every referenced existing Dataflow
event occurs and the intrinsic release point is eligible. Dynamic event
occurrences, absolute start times, queue state, and runtime arbitration state
are not persisted.

System ResourceUse owns event-relative activation, reservation and release,
typed workload parameters, demand selections, and Physical Tag or other typed
sharing assignments. Fabric owns the physical resources, capacity dimensions,
use vectors, latency, initiation interval, service guarantees, and parameter
domains. Imported SpatialMapping uses are occurrence-qualified in a derived
closure projection; SystemMapping does not copy them into System ResourceUse.

SystemMapping has no independent ScheduleBinding, BufferBinding,
ResourceSharing, TemporalTagAssignment, or UseTemplate family. Physical
buffer selection belongs to ServiceRealization and occupancy belongs to
ResourceUse. Flattened calendars and guaranteed resource-time envelopes are
derived views, not persistent records.

### System Record Identity

ExecutionBinding and ServiceRealization are uniquely located by their closed
structural keys. ResourceUse is uniquely located by its complete typed
structural key. The initial SystemMapping schema needs no independently
referenceable Mapping-local entity, so its artifact-global `EntityId`
namespace may be empty. SpatialMapping import ordinals and ServicePlan
ordinals are scoped serialization aids, not EntityId values.

## Canonical Assembly

Mapping owns one versioned canonical textual assembly writer. The writer
produces UTF-8 with LF line endings, no trailing spaces, and exactly one final
newline. It excludes the transport `builtin.module`, locations, comments,
aliases, debug names, provenance, and other non-semantic metadata. Generic
MLIR printer flags and raw MLIR bytecode are not identity authorities, and
there is no parallel persistent binary schema.

Textual authoring order is not semantic. The finalizer emits record families
in schema-owned order and keyed records in canonical semantic-key order after
canonical labeling and ID assignment. Explicitly ordered port maps, route
ordinals, service legs, and other semantically ordered arrays preserve their
defined order. Schema defaults are completed and printed or omitted only by
the schema declaration.

A standalone importer may accept legal noncanonical whitespace and record
order, then parse to the typed model and emit canonical bytes. Unknown
operations, fields, enum values, versions, duplicate semantic keys,
noncanonical persistent ID assignment, and incomplete profiles are rejected.
An Artifact Store reader additionally requires the stored bytes, descriptor,
digest, canonical assignment, and profile verifier result to agree; it does
not repair an object while reading it.

## Finalization

A TechMapping writer resolves exact upstream bindings and scoped references
through each upstream family's independently verified read-only projection,
validates closed coverage and complete selected relations, performs exact
semantic-graph canonical labeling and artifact-global ID assignment, emits
the root, runs the finalized profile verifier, and invokes the canonical
writer. The Common SHA-256 v1 finalizer then computes identity and performs
collision-checked atomic publication.

For a Dataflow binding, resolution uses the exact
`CanonicalDataflowProgramView`. Mapping cannot assign, repair, or persist a
shadow graph, actor, launch, terminal, service-member, event, memory-view, or
memory-exposure catalog.

This includes memory views derived by the Dataflow-owned graph-launch binding.
They resolve through the existing logical-memory and `MemoryBinding` records;
there is no Mapping record for a pointer conversion or boundary cast.

A SpatialMapping writer first rebuilds and verifies intrinsic base closure
from `D`, `T`, `F`, and the selected records, then runs separate admission
against the exact invocation `K`. Only after base verification and admission
succeed may it canonicalize record order, assign the MemoryBinding IDs and RouteTree
owner-local ordinals, run structural verification on the finalized root, emit
canonical bytes, and invoke the Common finalizer.

The SystemMapping writer resolves `D`, `F`, and the imported SpatialMappings;
normalizes execution-binding and plan-selection relations, plans, route trees,
and resource uses; derives and checks the exact import table; verifies
complete cross-layer base closure; and runs separate admission against the
exact invocation `K`. Only a verified and admitted draft may receive import
and owner-local ordinals, pass finalized structural root verification,
produce canonical bytes and Common identity, and be published atomically.

An artifact's own identity never enters its semantic preimage. Failure,
partial construction, unsupported proof, or budget exhaustion cannot publish
a Mapping artifact. The corresponding reader parses the exact supported
version, resolves every exact upstream reference, runs the same independent
base verifier, and only then derives immutable C++ views or native hot
projections.

## Validation Anchors

Stable Mapping anchors require one ResourceUse per selected adapter transition
case and `Intrinsic` release for the exact portable ordered-cardinality
contract. Sparse and all-zero masks are executions of the same static
serialize active-group ResourceUse, not different Mapping records. The anchors
reject any exact-adapter projection that derives `AllOf` from the active-result
union or that accepts a missing Dataflow activity-definedness proof. Reordered
authoring produces the same canonical ResourceUse bytes. Compatibility anchors
also import the old one-cycle Fabric record, reject its ordered-adapter
TechMapping seed as `CapabilityInadmissible`, and require the portable RTL
provider to return typed `Unsupported`.

## Derived And External State

The following are not Mapping artifacts or additional schema authorities:

* mutable candidates, search queues, action histories, solver caches, and
  transaction state;
* `FrozenModel`, dense indices, CSR or SoA projections, configured-function
  views, and closure projections;
* resolved config views, ObjectiveProjection, search domains, exact
  MappingConstraintSet ArtifactIdentity bindings, and admission results;
* configured Fabric, raw `sw_configs`, bitstreams, runtime images, Deployment
  payloads, and simulator-specific bindings;
* legality booleans, proof witnesses, costs, metrics, rankings, and
  acceptance decisions; and
* failures, diagnostics, reports, manifests, provenance, and Evaluation
  Evidence.

MappingConstraintSet is never a Mapping Artifact or Mapping result state. Only
its exact reference and the result of applying it to a base-valid Mapping
belong to invocation metadata or Evaluation Evidence.

Derived views are immutable and deterministically rebuildable from their
exact semantic inputs. Cache keys bind the complete dependency closure and
producer semantics. A cache cannot transfer Mapping coverage, local
references, legality conclusions, or physical decisions into another
artifact context.

Evaluation owns observations, metrics, findings, and model identity. The
central DSE controller owns objective policy, quality gates, ranking,
promotion, fallback resolution, and required Evidence. Mapping search may use
resolved typed projections of those authorities, but none become Mapping
semantic content.

## Non-Goals

The Mapping schema does not define a separate absolute Schedule IR, a generic
property or correspondence bag, runtime occurrence identity, a fourth
physical Mapping profile, Evaluation Request or Evidence schemas, Deployment
payload schemas, protocol encodings, or bitstream formats.
