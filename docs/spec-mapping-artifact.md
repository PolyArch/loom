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
| `mapping.tech` | `2.0` | Canonical Dataflow Program `D`, Fabric Hardware Description `F` | `ComputeRealization`, `MemoryRealization` |
| `mapping.spatial` | `2.0` | TechMapping `T`, Canonical Dataflow Program `D`, Fabric Hardware Description `F` | `ComputeBinding`, `MemoryEngineBinding`, `MemoryBinding`, `RouteTree`, `ResourceUse` |
| `mapping.system` | `2.0` | Canonical Dataflow Program `D`, architecture-only Fabric Hardware Description `F`, canonical SpatialMapping import table | `ExecutionBinding`, `ServiceRealization`, `ResourceUse` |

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

The current schema is the complete `loom.mapping 2.0` contract with all three
roots. Version 1.0's load/store-only `AccessEntry` and logical-memory-only
operation-service owner are superseded by the closed `MemoryOperationEntry`
and fence-aware ServiceRealization contract. An earlier draft assigned `1.0`,
`1.1`, and `1.2` according to the order in which profiles were discussed; that
unpublished history is retired.
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
and logical-root sets are derived from the relation domain and the referenced
Dataflow actors; a fence-only realization has no logical root. The root does
not repeat implementation, actor, or root lists. Actors sharing a Memory
Realization do not make their edges internal; only an exact selected
internal-edge witness does so.

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
      MemoryBindingRef
      dispatch_target : LocalMemoryServiceRef | ManagerEndpointRef
    }
  | FenceOperation {
      actor_ref
      operation_placement
      consistency_target : MemoryConsistencyDomainRef | ManagerEndpointRef
    }
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
Fabric-declared resident-context ordinal. The
collection of operation-entry targets and corresponding ExposureEntry targets
is the persistent owner of Mapping's selected `C_dispatch`. The verifier
checks each selection against Fabric-owned `H_dispatch`.

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
writers or ingresses. One operation row does not own a common tag. The
operation kind and capability alternative are configured row state rather
than runtime content-match fields.

### MemoryBinding

A `MemoryBinding` is one atomic relation:

```text
one LogicalMemoryInterval -> one PhysicalMemoryServiceRegion
```

It stores one artifact-global Mapping `EntityId`, the typed logical
memory/view reference, logical interval, typed physical-service reference,
physical region, and any selected Fabric-owned address transform that cannot
be derived from the endpoints.

Multiple disjoint records represent partitioning. Replication, mirroring,
coherence, or overlapping placement requires an explicit Fabric composite
service or transform; overlapping rows cannot imply those semantics.

Each MemoryBinding owns `ExposureEntry` children ordered by software boundary
key. An exposure stores the software memory-output obligation, selected
subordinate or provider terminal, and one
`LocalMemoryServiceRef | ManagerEndpointRef` dispatch target. It has no
independent ID and does not form a top-level ExposureBinding family.

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
  release = intrinsic
          | causal_event(SpatialActivityEventRef + optional guaranteed offset)
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

Version 2.0 has exactly one InstructionCore per AccCore. Its context reference
is mechanically derived as:

```text
InstructionCoreContextRef = (AccCoreOccurrenceRef, 0)
```

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
is not copied into the key. Channels, graph-launch transfers, external
messages, and multicast use this rule; multicast sinks with one producer
remain one family. Merge, zip, reorder, and reduction require an explicit
Dataflow actor.

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
  plan_selection
}

ServicePlan {
  service_target_bindings
  transfer_leg_realizations
  physical_refinement_assignments
}
```

A plan element referenced by ResourceUse has the structural reference:

```text
ServicePlanElementRef =
  (ServiceRealizationKey, canonical plan ordinal, typed element key)
```

The typed element key is the natural key of the closed child kind, such as a
`CanonicalServiceLegKey` or a Fabric physical-refinement domain. A service
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
      optional non-derived address transform
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

A memory-region target owns every exposure child that selects that region.
The child is keyed by its `MemoryExposureRef` and provider terminal and has no
`ServiceMemberRef`, service leg, or independent ID. Missing, extra, duplicate,
or wrong-owner exposure children are completeness failures.

A fence plan contains exactly one `ConsistencyTarget`; its selected domain
must cover all constrained effects in that execution context. A
`TransferLegRealization` binds one transfer leg derived from the Canonical
Service Schema to a flat `RouteTree` over system physical traversals. The
route tree itself owns terminal endpoint choices. Protocol packets, flits,
headers, concrete virtual-channel encoding, and implementation-specific bus
encoding remain owned by the selected interconnect implementation.

Plan selection first derives an `ExecutionContextKey` from the evaluated
`B_thread` and `B_graph` targets. Only reachable contexts are stored. Within a
context, the same closed binding-relation algebra may select a plan from
Dataflow-owned logical inputs. An event-rooted relation resolves its complete
input universe from the exact Dataflow-owned `EventLogicalProjection` and, for
a DynamicWork domain, its separately owned stable-item projection. It may
reference any typed subset of those inputs but cannot persist another
projection or reinterpret input order. Plans have no `EntityId`; the
finalizer sorts and deduplicates complete plan semantic keys before assigning
owner-local ordinals.

ServiceRealization is the only SystemMapping family for selected system
routes, physical buffers, target service regions, address transforms, and
mapping-visible service refinements. There are no parallel ChannelRoute,
MulticastRoute, System MemoryBinding, ExternalBinding, TerminalBinding, or
generic service-graph families. It does not persist modification order,
reads-from, synchronizes-with, sequentially-consistent order, queue contents,
or any other dynamic MemoryConsistencyDomain state.

### System ResourceUse

SystemMapping uses the same closed ResourceUse shape as SpatialMapping.
InstructionCore-resident uses reference their ExecutionBinding and derived
`InstructionCoreContextRef`; service-plan uses reference the exact
`ServicePlanElementRef`. Applicability is mechanically derived from the
existing `plan_selection`. A ResourceUse cannot copy that predicate or plan
choice, nor select a different target, context, configuration, route, or
service. Its use site must resolve a Fabric-owned use pattern exposed by the
selected owner.

Its relative activation has one trigger and one typed release policy:

```text
relative_activation:
  trigger = EventFamilyKey + optional guaranteed offset
  release = intrinsic
          | causal_event(EventFamilyKey + optional guaranteed offset)
```

`EventFamilyKey` is exactly the Dataflow-owned `StaticTransferEventRef` alias.
It does not contain concrete coordinates, launch-parameter values, a copied
logical projection, or a Mapping-local event ID. Every trigger and causal
release imports the exact Dataflow-derived `EventLogicalProjection`; runtime
binds concrete values for that schema when the corresponding occurrence is
observed. Two records referring to the same static event therefore use one
key, while context and parameter relations own any legal variation across its
logical domain.

Offsets are legal only when guaranteed by the Fabric service contract.
`intrinsic` uses the Fabric pattern's finite or periodic completion contract.
A causal release holds the resource until the referenced existing Dataflow
event occurs. Dynamic event occurrences, absolute start times, queue state,
and runtime arbitration state are not persisted.

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
normalizes binding relations, plans, route
trees, and resource uses; derives and checks the exact import table; verifies
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
