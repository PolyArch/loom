# Place And Route

This document is the normative owner of Loom Spatial and System PnR
algorithms, native state, deterministic search protocols, final closure, and
the common Spatial and System MappingConstraintSet semantic and wire algebra.
It also owns the one invocation-only diagnostic channel shared by production
TechMapping, Spatial PnR, and System PnR.
`docs/spec-tech-mapping.md` separately owns production TechMapping generation.
The Mapping profile documents own persistent Mapping Artifact spelling and
canonical serialization; this document separately owns the constraint-family
roots and clause wire schema.
Evaluation and central DSE documents own objective, gate, Evidence, and
resolved-model schemas. This document consumes those interfaces without
restating them.
`docs/spec-fabric-identity.md` owns every Fabric-local entity, endpoint,
traversal, state, and service reference consumed by the projection catalogs.

There is one Dataflow-to-Fabric Mapping artifact family. TechMapping selects
semantic realizations, SpatialMapping adds physical realization inside a
SpatialCore, and SystemMapping binds complete execution and service behavior
across an architecture-only Fabric system. There is no `PhysicalMapping`
profile and no fourth profile for flat System search.

## Mapping Invocation Diagnostics

Mapping diagnostics use one process-wide logger and one environment binding:

```text
LOOM_DEBUG_VERBOSE = nonnegative decimal integer
```

The binding is invocation-local debug input. It is not ResolvedConfig,
Mapping state, an Artifact field, an Evaluation observation, semantic work, or
a persistent trace. The logger parses it once per process. An unset, empty,
non-decimal, or zero value selects level zero; values above three select level
three. Level `N` includes every lower nonzero level:

* level one emits invocation lifecycle, termination or failure, and cumulative
  statistics;
* level two additionally emits candidate, seed, negotiation-iteration,
  decoded-capacity-conflict, Action proposal/outcome, and System context-choice
  events; and
* level three additionally emits per-net route detail, exact owner/state/claim
  and endpoint/traversal detail, exact cut or reachability evidence, and Action
  deltas.

Every event is one line-atomic JSON object on stderr. It contains
`schema = "loom.mapping.debug.1"`, numeric `level`, closed ASCII `event` and
`stage` spellings, and an invocation-local `sequence`. Events add the stable
candidate, worker, iteration, Action, owner, state, logical-net, endpoint,
traversal, or claim ordinals applicable to that event. Concurrent emission
order is diagnostic timing, not replay identity; stable event keys permit
post-processing without treating line order as a semantic authority.

Level-one statistics include the applicable candidate rows and publications,
Actions proposed, accepted, rejected, and rolled back, A* expansions,
negotiated iterations, capacity conflicts, arithmetic failures, and final
closure status. Higher levels refine those counts with the exact events that
produced them. A decoded capacity conflict reports the physical owner and
state, exact `usage/capacity`, contributing logical nets and claims, and their
endpoints and traversals. When level three requests cut or reachability detail,
the logger reports the result of the same frozen-topology analysis owned by
Negotiated Routing. It does not recompute a diagnostic-only approximation. The
event distinguishes an exact fixed-terminal capacity-cut certificate from
reachability evidence that does not establish one. Neither result can prove
infeasibility by timeout.

The level-zero hot path performs no JSON construction, stderr locking,
diagnostic-only analysis, or diagnostic allocation. An exact capacity-cut
analysis required by Negotiated Routing is algorithm work at every diagnostic
level; enabling the logger only observes its already computed result. At every
level, diagnostic collection, formatting, lock acquisition, output failure,
and extra presentation detail cannot affect candidate order, random draws,
search decisions, deterministic work accounting, termination, Mapping bytes,
or Artifact identity. Events never
include raw Dataflow values, source or host paths, environment contents,
credentials, external-tool restricted data, raw pointers, or implementation
container order. No subsystem may parse `LOOM_DEBUG_VERBOSE` independently or
create a second Mapping statistics/logging channel.

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

The built-in root-complete Spatial PnR candidate generator is a typed
composition over this exact invocation, not another freeze entry. It consumes
a finite canonical set of TechMapping Artifacts and exactly one `F`. For each
`T`, it strictly imports `T`, requires `T.F == F.id`, reconstructs the unique
`D` reference from `T.D`, and asks the MappingConstraintSet owner to publish
the exact empty `K(D,T,F)`. It then invokes
`generateSpatialMappings(D,T,F,C,K)` unchanged. The empty clause sequence is
therefore still a real, durable Artifact; a missing `K`, null input, config
default, or wildcard is never accepted as equivalent.

This descriptor is only the root-complete unconstrained convenience path.
Any caller with one or more Spatial constraint clauses uses the ordinary exact
five-input invocation and supplies its independently finalized `K`. The
adapter has no redundant `D` input, no `K` config field, and no private
constraint or PnR semantics. `ProvenInfeasible` contributes no candidate;
`Generated`, including a semantic-limit prefix, contributes its complete
published set; and `Incomplete` or `Unsupported` stops canonical `T` traversal
while retaining only candidates completed for earlier `T` inputs.
`Invalid` means the finite input binding or one of its exact owner tuples is
not a legal invocation, while `InternalError` means the PnR owner failed its
own invariant. Either aborts the complete adapter invocation and produces no
formal output or lineage prefix. Immutable objects published before that
failure may remain visible in the ArtifactStore, but they are not outputs of a
completed or incomplete invocation and cannot be promoted through that failed
plan node.

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
`SelectedObjectiveClosure`.

`H` is the immutable, canonical-framed finite search-domain view mechanically
elaborated from `D`, `F`, `R`, resolved Compilation and DSE policy, and `K`.
It owns finite execution-binding atoms and each atom's legal AccCore or graph
target domain, plus endpoint-factorized service compatibility relations. Those
relations are indexed only by exact existing Dataflow and Fabric references.
They contain neither a selected endpoint nor the finalized
`ExecutionContextKey` owned by `docs/spec-mapping-identity.md`. `H` is not an
artifact, Mapping result, or second config authority. Its canonical digest
participates in the native cache key and its exact descriptor and digest are
recorded in the `InvocationManifest`, so a changed candidate domain is a
changed invocation.

`R`, `H`, `C`, and `K` affect search, closure, and admission but do not enter
the semantic identity of a selected SystemMapping. The `InvocationManifest`
binds their exact references, the component-view descriptor and digest for
`C`, and the owner-specific view descriptor and digest for `H`. The
persistent root owns `D`, `F`, the non-empty root thread launch set, the exact
derived SpatialMapping imports, and its selected records.

An InstructionCore-only closure may have no SpatialMapping catalog or reopened
Spatial subproblem. That case still uses the ordinary SystemMapping and System
MappingConstraintSet profiles; it never creates a dummy graph or
SpatialMapping.

The built-in root-complete System PnR candidate generator is a typed
composition over the same exact invocation. It consumes exactly one Canonical
Dataflow Artifact `D`, a finite canonical set of immutable SpatialMapping
Artifacts, and exactly one System Fabric Artifact `F`. It derives `R` as the
complete canonical root-thread-launch inventory of `D`. When that inventory is
empty, the descriptor completes with an empty SystemMapping output set and
does not invent a System PnR invocation. Otherwise it asks the System
MappingConstraintSet owner to publish the exact empty `K(D,F,R)`, projects the
whole-domain Presburger partition and hierarchical `H` through their existing
owners, and invokes `generateSystemMappings(D,F,R,H,C,K)` unchanged. Every
SpatialMapping input must bind `D` and an attached Module lineage admitted by
`F`; an InstructionCore-only invocation may supply an empty SpatialMapping
set.

This descriptor is only the root-complete unconstrained hierarchical path. A
caller with System constraint clauses, a strict subset of root launches, a
precomputed exact partition, or a flat Spatial reopen domain uses the ordinary
six-authority System PnR invocation. The adapter has no optional constraint,
root, partition, or search-domain slot and no private binding, routing,
progress, or finalization semantics. `Generated` contributes only independently
finalized SystemMapping references, `ProvenInfeasible` contributes a completed
empty set, and `Incomplete` remains typed without publishing a partial
Mapping. An invalid owner tuple or internal failure aborts the Generate
invocation and produces no formal output or lineage edge. Its descriptor
references the ordinary PnR owner's complete work-unit catalog unchanged;
adapter-local aggregation, omitted search work, and a second System-only work
taxonomy are forbidden.

### System Search-Domain View

`H` atomizes each Dataflow-owned execution relation without selecting a
target, then derives service compatibility from those legal execution choices.
Its only binding-partition shape is:

```text
BindingPartitionShape =
    PresburgerCells {
      canonical non-empty Presburger cell sequence
    }
  | StableKeyGroups {
      canonical non-empty stable-key group sequence
    }
```

For every `RootThreadLaunchRef` and reachable `RootedGraphLaunchRef`, one shape
is a complete, disjoint partition of the exact legal may-domain owned by `D`.
Presburger cells are canonical integer sets over Dataflow-owned coordinates
and launch parameters. Stable-key groups are canonical non-empty finite sets
of values from the Dataflow-owned stable-key projection. Empty, overlapping,
gapped, foreign-domain, duplicate, or mixed-representation partitions are
invalid.

The shape contains no AccCore, SpatialMapping, SpatialCore, service, endpoint,
route, or other selected target. For each resulting binding atom, `H`
mechanically derives exactly one applicable closed target-domain variant from
exact `D/F/R/C/K`:

```text
SystemSearchAtomDomain =
    ThreadBinding {
      compatible_acc_cores
    }
  | HierarchicalGraphBinding {
      compatible_spatial_mappings
    }
  | FlatGraphBinding {
      exact_spatial_reopen_problems :
        canonical non-empty sequence<FlatSpatialReopenProblem>
      compatible_immutable_seeds :
        canonical sorted unique set<ArtifactRootReference>
    }

FlatSpatialReopenProblem {
  tech_mapping_ref : exact TechMapping ArtifactRootReference
  spatial_config_view : ResolvedPnrConfigView
  spatial_constraint_set_ref :
    exact Spatial MappingConstraintSet ArtifactRootReference
}
```

The graph variants are mutually exclusive for one invocation. A hierarchical
domain contains only complete immutable SpatialMapping
`ArtifactRootReference` values. A `FlatSpatialReopenProblem` plus enclosing
`D/F` is exactly one Spatial five-input problem derived by the resolved policy;
its imported native freeze supplies the legal decision domains. Its complete
Spatial `C` includes the component-view descriptor, canonical bytes, and
digest. Every `T` and Spatial `K` must bind the enclosing `D/F` exactly.

The seed set is always present and may be empty. Every seed must bind the same
exact `D/F`, match one listed problem's `T`, and have passed independent base
verification. Problems are ordered by complete `T` reference, canonical `C`
bytes, and complete `K` reference; seeds are ordered by complete
ArtifactIdentity. Neither collection contains a provisional Mapping reference,
identity, or `B_graph` target. A candidate may hold a native handle to its
mutable flat state, but the handle is scoped to that candidate and cannot
enter `H`, a cache key, canonical comparison, or persistent Mapping. Flat
finalization first verifies and identifies each selected SpatialMapping and
only then rewrites `B_graph` to its ordinary immutable target.

Service restrictions do not create another binding-atom variant or a third
partition shape. The System MappingConstraintSet catalog owns its restrictions
by the exact obligation or transfer-terminal subject defined below; it does not
own a partition-qualified service subject. While deriving each compatibility
row, `H` mechanically intersects the applicable endpoint-independent
restriction from exact `C/K` with compatibility from the row's exact Fabric
endpoint. If there is no narrower restriction, the restriction input is the
complete finite typed target universe from `F`; absence never means
unrestricted. This intersection is one derived row domain, not two candidate
authorities that CandidateState must reconcile.

Service compatibility has this endpoint-factorized shape:

```text
SystemSearchServiceDomain {
  SystemServiceObligationKey
  target_compatibility[] {
    ServiceTargetSubject
    bound_system_service_endpoint : SystemServiceEndpointRef
    compatible_service_regions | compatible_consistency_domains
  }
  transfer_terminal_compatibility[] {
    SystemTransferTerminalKey
    BoundTerminalEndpoint
    compatible_transport_endpoints
  }
}

ServiceTargetSubject =
    ServiceMember {
      ServiceMemberRef::AddressedMemoryActor
        | ServiceMemberRef::FenceActor
    }
  | MemoryExposure { MemoryExposureRef }

BoundTerminalEndpoint =
    Message { FabricTransportEndpointRef }
  | MemoryOrFence { FabricMemoryEndpointRef }
```

Only the target-domain variant applicable to the exact obligation kind is
present. There is exactly one `SystemSearchServiceDomain` for every obligation
derived from `D/R` and no other one. `ServiceTargetSubject` is a closed typed
refinement of the existing Dataflow references. The member carrier retains its
complete Dataflow-owned variant and wire; `MessageTransfer` is excluded because
it has no operation target. The refinement creates no generic service-member
identity. Its nested row key sets are derived as follows:

* `MessageTransfer` has no target-domain row. For an addressed-memory
  obligation, target rows cover every contextual addressed-operation member
  and every separate `MemoryExposureRef`. For a fence obligation, they cover
  every contextual fence member. A memory exposure remains its Dataflow-owned
  type and does not become a `ServiceMemberRef` or acquire a service leg.
* For each target subject, rows cover exactly the sorted unique
  `SystemServiceEndpointRef` values that a legal execution binding and
  hierarchical SpatialMapping or flat reopen decision can mechanically bind.
  The row derives compatibility from that subject and endpoint, then applies
  the exact endpoint-independent `C/K` restriction for the obligation.
* For each `SystemTransferTerminalKey`, terminal rows cover exactly the sorted
  unique typed Fabric endpoints that those same legal choices can
  mechanically bind. `MessageTransfer` uses the transport variant; memory and
  fence use the memory variant selected by canonical leg direction and
  endpoint role. The row applies the exact endpoint-independent `C/K`
  restriction for that terminal.

For a `MessageTransfer` terminal, mechanically bindable endpoints are further
restricted by its Dataflow-defined execution owner. A root-thread runtime side
uses the unique HostCore occurrence; its hardware side uses every AccCore in
the legal `B_thread` range. Graph-boundary terminals use the legal parent
AccCore range of their rooted graph, while a channel producer or consumer uses
the legal AccCore range of its own rooted terminal. Rows cover only matching
transport-plane service endpoints owned by those occurrences. Endpoints owned
by a memory service, transform, transport resource, hardware domain, external
boundary, or unrelated core are not mechanically bindable message terminals.
Multiple role-compatible endpoints on one legal owner remain separate exact
rows and may later be route alternatives.

An empty domain for a reachable subject-and-endpoint or
terminal-and-endpoint pair remains present. A missing, extra, duplicate,
wrong-owner, wrong-plane, or unreachable pair is invalid. Target rows are not
multiplied by SpatialMapping identity or AccCore occurrence when the exact
bound System endpoint is equal: its capability is one Fabric fact and is
reused as such. Terminal rows likewise collapse only when their exact selected
typed endpoint is equal. Different occurrence endpoints necessarily use
different terminal rows even when they share the same target-compatibility
row.

For a `SystemTransferTerminalKey`, `compatible_transport_endpoints` is derived
without a target-owned pairing table. A `MessageTransfer` terminal uses the
matching transport-plane service endpoint directly. A memory or fence
terminal uses the row's exact `FabricMemoryEndpointRef`. Candidate resolution
of that row first follows the selected Module-local manager or subordinate
path to its Module boundary, qualifies that boundary through the selected
AccCore occurrence, and follows the unique Fabric memory
`spatial_attachment`. This fixes the three-reference row and, within it, the
Module/occurrence endpoint pair and exact System service endpoint.
Canonical leg direction, the pair's complementary
roles, and whether the queried terminal is source or destination select exactly
one member of that pair. The pair's exact System service endpoint is the sole
capability authority for both members. The selected pair member's
`ServiceLegCarrierAttachment(endpoint, kind, leg_ordinal)` carrier set alone
enters that terminal domain. The exact selected Dataflow service member is
tested against the pair's System endpoint capability before either terminal
is admitted; incompatibility produces required empty rows for both sides and
never causes a search for another endpoint. Direction and payload
compatibility are recomputed from the selected pair-member role, the Canonical
Service leg, and that one capability domain. The singular payload width in the
resulting System `RouteQuery` is the nonpersistent maximum-width service-leg
envelope owned by the
[Service-Leg Carrier Attachment](spec-fabric-system-adg.md#service-leg-carrier-attachment)
contract. `H` neither copies those facts nor chooses either memory endpoint by
capability search. It does not infer a carrier from entity ownership,
endpoint ordinals, equal width, protocol names, or physical targets. A missing
binding, incompatible bound endpoint, or empty derived carrier set is proven
infeasibility.

For each target-compatibility row, `H` carries only the target-domain field
selected by its exact obligation-key variant. A logical-memory operation
obligation has `compatible_service_regions`; a fence operation obligation has
`compatible_consistency_domains`, derived from the matching
`FenceCapabilityDomain` records and the exact Dataflow-owned fence effects; a
`MessageTransfer` obligation has neither field. The two operation target
domains are mutually exclusive. Each row is the exact intersection of its
applicable `C/K` restriction and compatibility with the bound System service
endpoint and its explicit service/transform closure. Capability compatibility
can filter regions or consistency domains within that closure; it cannot
introduce another endpoint. An empty applicable domain is proven infeasibility,
not permission to substitute a memory region, manager endpoint, or
candidate-private target.

The owner-specific view descriptor identity is
`loom.system_pnr_search_domain`, version 3.0. Its exact descriptor bytes are
the ASCII bytes `loom.system_pnr_search_domain.3.0`, without a trailing zero
byte. Version 2.0's unqualified service-domain rows are incompatible and must
be rejected; a consumer cannot recover the missing bound endpoint by
inspecting endpoint order, taking a union or intersection, or consulting a
candidate-private cache. Canonical view bytes order thread bindings, rooted
graph bindings, and service obligations by their complete typed semantic keys.
Within a service obligation, target rows are ordered by complete
`ServiceTargetSubject` bytes and then complete `SystemServiceEndpointRef`
bytes. Terminal rows are ordered by complete `SystemTransferTerminalKey`
bytes, the `BoundTerminalEndpoint` variant ordinal (`Message = 0`,
`MemoryOrFence = 1`), and complete endpoint bytes. Presburger cells use their
canonical integer-set bytes.
Stable keys are sorted within each group, groups are sorted by complete key
sequence, and every legal target domain is sorted by complete target semantic
key. Counts and ordinals use fixed-width big-endian integer framing.

The `SystemSearchAtomDomain` canonical variant ordinals are
`ThreadBinding = 0`, `HierarchicalGraphBinding = 1`,
and `FlatGraphBinding = 2`.
`ServiceTargetSubject` uses `ServiceMember = 0` and `MemoryExposure = 1`; the
nested `ServiceMemberRef` retains its Dataflow-owned variant framing. Every
closed variant is framed as its `u32be` ordinal, followed by
`u64be(length(payload_bytes))` and the exact existing canonical payload bytes.
The same framing applies to `BoundTerminalEndpoint` with the ordinals above.
No native index, seed position, authoring order, or candidate handle enters
canonical bytes.

The digest is:

```text
SHA-256(
  bytes("loom.system.pnr.search.domain.digest.v1\0")
  || u32be(length(schema_descriptor_bytes))
  || schema_descriptor_bytes
  || u64be(length(canonical_view_bytes))
  || canonical_view_bytes
)
```

This digest is an invocation/cache integrity value, not Common
`ComponentViewDigest`, ArtifactIdentity, or persistent SystemMapping content.

Only fields applicable to each typed binding atom or service-compatibility row
are present; an empty required domain is proven infeasibility for the affected
candidate rather than permission to invent a target. CandidateState selects
one complete legal target plan for each atom through the ordinary System
Actions. For an addressed target, a plan is derived from the exact Fabric
memory-service closure plus the applicable H row; it is not another field of
`H`.
Finalization merges atoms with the same selected target and mechanically
reconstructs the existing
`BindingRelation<AccCoreOccurrenceRef>` or
`BindingRelation<SpatialMappingImportRef>`. It does not persist the atomization
or a second relation.

`H` cannot change logical coordinates, extents, launch parameters, stable
keys, schedule, channel `source_map`, or may-domain predicates. It cannot use
physical coordinates or synthesize a predicate. Different partition shapes
are separate resolved generator invocations with distinct owner-specific view
digests; they are not mutable alternatives inside one `H`. A changed digest
changes invocation and cache lineage but may still produce the same canonical
SystemMapping identity.

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
SpatialLogicalNetKey = CanonicalGraphProducerEndpointRef

SpatialTransferTerminalKey =
    Source(SpatialLogicalNetKey)
  | Sink(SpatialLogicalNetKey,
         CanonicalGraphConsumerEndpointRef)

PhysicalAddressPoint = (FabricMemoryServiceRef, Address)
```

`SpatialLogicalNetKey` identifies one graph-local producer from exact `D`.
Exact `D/T` derives its residual transfer obligation after
realization-internal sinks are removed. "Exposed" is therefore a derived
property of the sink relation rather than endpoint identity. The key never
refers to a result-time RouteTree, Tag assignment, or native net index. The
first Spatial `ProjectionKind` catalog is exactly:

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
  MemoryRealizationRef -> Set<FabricMemoryOccurrenceRef> [singleton]

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
  LogicalMemoryRootRef -> Set<FabricMemoryServiceRef> [zero-or-more]

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
and a zero-sized memory object may have an empty address region. A root whose
selected MemoryBindings are all BoundaryProxy has an empty
`memory_bound_services` and `memory_address_region` projection. Base
verification still requires complete MemoryBinding coverage; an empty local
service projection is therefore not an absent or unknown Mapping fact. These
are declared cardinalities, not absent or unknown projection results.

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
   `FabricMemoryServiceRef` order; each service contains sorted, merged,
   non-empty half-open address intervals.
4. Closed tuple carriers use a lexicographically sorted unique full-tuple set.
   They are never split into component domains or a Cartesian product.

Persistent bit masks, bitsets, alternate encodings, and carrier property bags
are forbidden. `SparseIds`, `DenseWords`, and other bitset forms are derived
`FrozenConstraintIndex` choices only and cannot be written back into `K`.
There is no predicate DSL, `Exists` or `NonEmpty` atom, runtime extension
registry, or last-wins interpretation.

Canonical comparison never uses printed attribute text. A Mapping-local
realization reference compares by its unsigned `u64be` EntityId. Dataflow and
Fabric references reuse the canonical owner-local comparison bytes defined by
their reference codecs. Closed variants prepend their `u32be` discriminator,
and tuples compare their length-framed component keys in field order. Unsigned
interval endpoints are emitted using the smallest nonzero unsigned MLIR
integer width that represents the value; an authored integer width carries no
independent semantics.

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
`D/F/root_thread_launches` and the Canonical Service Schema.
`docs/spec-mapping-identity.md` owns `CanonicalServiceLegKey`; this document
adds only the terminal role needed by the projection catalog:

```text
SystemTransferTerminalKey =
    Source(CanonicalServiceLegKey)
  | Sink(CanonicalServiceLegKey, canonical sink ordinal)
```

`ServiceMemberRef` is derived from the exact obligation anchor and bound
Dataflow program. The Canonical Service Schema owns the local leg ordinal.
For a multi-sink leg, the sink ordinal indexes the canonical sorted sink set
derived from the exact workload scope; neither the sink set nor a flattened
global leg number is stored in the obligation key.

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
    -> Set<FabricMemoryServiceRegionRef> [zero-or-more]

transfer_terminal_attachment:
  SystemTransferTerminalKey
    -> Set<FabricTransportEndpointRef> [zero-or-more]

transfer_selected_traversals:
  CanonicalServiceLegKey
    -> Set<FabricPhysicalTraversalRef> [zero-or-more]

transfer_resource_states:
  CanonicalServiceLegKey
    -> Set<FabricResourceStateRef> [zero-or-more]

transfer_assigned_tag_values:
  CanonicalServiceLegKey
    -> Set<PhysicalTagValue> [zero-or-more]
```

`transfer_terminal_attachment` contains the exact transport endpoints selected
by the realized RouteTrees. For memory and fence service legs, every selected
endpoint must belong to the corresponding Fabric-owned
`ServiceLegCarrierAttachment` domain. For `MessageTransfer`, no such row
exists and the selected endpoint remains the transport-plane service endpoint.
The projection never returns a memory endpoint, attachment key, capability
ordinal, or protocol-specific channel.

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

TechMapping alone owns selected Compute and Memory Realizations, selected
Fabric-owned FU capability and Memory Operation Engine templates,
configured-function match relations, template-relative memory internal-
connectivity witnesses, and software boundary correspondence. Spatial PnR
must not regroup actors, reconstruct a deleted `dataflow.subgraph`, rematch raw
Dataflow and Fabric, or select another semantic realization.

Fabric owns immutable topology, occurrences, endpoints, traversals, resource
and capacity schemas, use patterns, service contracts, and physical refinement
domains. Mapping selects only declared alternatives and owns physical legality
and the domain-independent PnR measures `V` and `G`. Evaluation owns all
accelerator- and workload-aware observations `Q`. The central resolved
`SelectedObjectiveClosure` is the only PnR composition of `V`, `G`, and `Q`.

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

Freeze resolves every exact persistent entity and structural reference once.
The published hot model retains only typed dense indices, owner-local offsets,
and reverse-incidence tables. ArtifactIdentity digests, recursive persistent
references, symbols, paths, and authoring handles never enter inner search
state.

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
* Fabric occurrences, contexts, endpoints, traversals, `ResourceState`s,
  `UsePattern`s, their owner-defined commit transitions, and tag, buffer,
  memory-service, and refinement capabilities;
* derived canonical memory-access views, parameterized operation-port
  compatibility, and declared memory use-pattern domains;
* factorized occurrence, context, attachment, and refinement domains;
* derived compiled `K` indexes, fully resolved `C`, and reachability,
  lower-bound, dependency, and reverse-incidence indices.

The handshake part of the Spatial model is one compact flattening of the
Fabric-owned sealed owner models:

```text
FrozenHandshakeNode = EndpointValid
                    | EndpointReady
                    | OwnerLocalJunction

FrozenHandshakeArc {
  source
  destination
}

FrozenHandshakeFragment {
  contribution_offset
  contribution_count
}
```

Endpoint nodes retain a reverse table to exact `HandshakeSignalRef` values.
Junction nodes are view-local and have no persistent reference or routing
meaning. Freeze stores unique potential arcs in source-major CSR, the exact
reverse CSR, fragment-to-arc incidence, Mapping-decision-to-fragment reverse
incidence, and canonical owner/refinement/occurrence ranges. FU physical
internal structure is expanded once per occurrence; capability rows share arc
and incidence storage. Switch broadcast projection is linear in physical
connectivity through owner-local conjunction junctions rather than a
materialized boundary transitive closure. Memory projection remains factored
by operation port, semantic role, capability alternative, and refinement. No
actor by occurrence by route by tag product is legal.

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
GraphBoundaryTerminal -> compatible Module-boundary attachment domain
ConfigurationOwner -> semantic-preserving physical refinement domain
selected facts -> ProgrammedConfigurationKey
```

Every selected capability-template port is classified exactly once as
`Internal`, `ExternalDemand`, or `InactiveQuiescent`. The last case is legal
only when the operation schema and capability relation prove no consume, no
produce, and no backpressure. A missing `PortDemand` is not an inactive marker.

A `PortDemand` belongs only to a Compute or Memory Realization whose concrete
occurrence is selected by SpatialMapping. It names the exact external
template-relative terminal and therefore acquires an occurrence-relative
attachment domain only after choosing that occurrence. A graph ingress or
egress has no occurrence and must not be represented by a synthetic boundary
unit or a placement-independent `PortDemand`. Instead, freeze derives its
domain from the exact Module root's canonical token-plane boundary attachment
relation, direction, semantic payload width, and tag capacity. The
`FabricModuleBoundaryEndpointRef` itself never enters the routing graph; its
occurrence-local endpoint is the RouteTree terminal. An empty well-formed graph
boundary domain is `ProvenInfeasible`.

For a Compute occurrence, the fixed FU occurrence port and each compatible PE
or explicitly declared local terminal are connected only by exact
Fabric-owned selector or local traversals. For a Memory occurrence, the exact
Memory Operation Engine endpoint projects to that occurrence's transport
endpoint. Freeze may cache these factorized domains and local matching
feasibility, but neither chooses an endpoint nor invents a connection.

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
only concrete `fabric.mem` occurrences whose exact
`memoryEngineTemplate(occurrence)` relation equals the Memory Realization's
selected engine template, followed by exact operation placement capabilities.
For each memory actor, `UnaryEligible` derives its
`CanonicalMemoryAccessView` and proves one selected physical operation port can
carry the complete address, data, and optional mask tokens and supports the
exact access form, memory-element width, lane geometry, alignment, mask
mode, and at least one declared use pattern. Equal total width is insufficient.
Any Memory Realization with no compatible occurrence, port/context, dispatch
target, or use pattern is `ProvenInfeasible` before search; PnR does not keep it
as a repairable routing or congestion state.
An unresolved dynamic mask cannot shrink a domain or static resource-claim
envelope. PnR verifies the complete Fabric-declared mask domain; actual masks
may reduce only execution-time transactions and Evaluation observations.
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
incidence, the selected handshake-arc adjacency and incremental topological or
SCC indexes, and typed `V/G` components. These handshake indexes are exact
rebuildable functions of `F` and current selections; they are not another
legality authority. Search work is not a cost component.
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

For each addressed target subject and its currently bound System service
endpoint, the candidate mechanically derives a finite
`ServiceTargetPlanDomain` from the frozen Fabric closure and that exact `H`
compatibility row. One domain member contains the canonical non-empty set of
terminal service-region branches needed to cover the subject's complete
logical interval; every branch contains its exact ordered transform path and
must end in a region admitted by the H row. MemoryService connections are
derived Fabric edges and do not enter the key. The candidate selects one
complete domain member; it cannot independently select a region and later ask
materialization to find a path.

A one-output address transform normally yields one branch group. A
`StaticInterleave` yields the exact terminal branches reached by every
non-empty output-ordinal domain, grouped only when their exact transform path
and terminal region are equal. A `CoherentMemory` correspondence maps one
input-region domain to its unique output region by region-relative offset; its
reachable output branches are provider alternatives, and the target plan
selects only a canonical exact cover of the source interval. For every address
in the complete source interval, the composed Fabric contracts must select
exactly one branch and place the transformed address inside that branch's
region. Thus an interleaved or coherent interval may repeat the same source
logical interval across several persistent `MemoryRegionTarget` children
without claiming that every address reaches every child. The branch relation,
output ordinal, correspondence member, and transformed address are derived
from Fabric; Mapping stores no strided-range language or copied transform
parameters.

The domain enumerates only finite simple paths, rejects repeated transforms or
connections, and keeps distinct exact paths when endpoint plus terminal region
does not imply one. Its canonical semantic key includes the exact path and
canonical branch regions, so two physically different ambiguous paths never
collapse. Finalization omits a path from Mapping only when the selected bound
endpoint and complete branch set uniquely derive it. `H` remains the SSOT for
endpoint-indexed target compatibility, Fabric remains the SSOT for topology
and transform behavior, and CandidateState owns only the selected composed
plan.

For each service anchor, candidate-native relation atoms pair the applicable
thread and graph decision atoms with one complete plan semantic key. They are
mutable search state, not provisional `ExecutionContextKey` values. For a
message anchor and producer point `p`, the producer event, every consumer
domain, and the Dataflow-owned `source_map` mechanically derive the canonical
unique set of `(sink terminal, selected execution owner)` pairs whose consumer
points map to `p`. Candidate relation atoms partition the producer domain so
that this set and the complete plan semantic key are constant in each atom.
One route sink is required for every pair. Repeated points for the same pair
collapse, while the same terminal on distinct owners remains distinct. A
terminal with no preimage in an atom is absent. An empty complete set selects
the canonical childless message plan and creates no route tree. Equal complete
plan semantic keys may be deduplicated. A candidate cannot merge endpoints
owned by different execution choices into one route domain or add a terminal
that is inactive in the current atom.

For each affected service subject or terminal, a service move mechanically
resolves one exact bound endpoint from the selected `B_thread`, the selected
hierarchical SpatialMapping or current flat Spatial decisions, and `F`. It
then selects the one matching H 3.0 endpoint-compatibility row, whose domain
already incorporates the exact endpoint-independent `C/K` restriction. Several
subjects in one obligation use the intersection of only their matching row
domains. A missing row is an invalid frozen view; an empty row or intersection
is infeasible. Neither case may fall back to another occurrence, endpoint,
global scan, union, or candidate-private compatibility cache.

For `MessageTransfer`, H lookup occurs only for pairs in the current applicable
set. The exact H rows considered for one pair are the factorized rows whose
transport service endpoint belongs to that pair's owner: the fixed
HostCore/runtime side or the AccCore selected by the applicable thread
decision. Multiple compatible endpoints on that one owner remain route
alternatives; endpoints on another HostCore or AccCore do not enter the
domain. A thread, graph, or route move rebuilds the applicable pair set, plan
semantic key, and route feasibility from the current decisions. Reusing a
route after a pair appears, disappears, or changes owner is an invariant
failure even when its endpoint remains globally reachable.

In flat mode this lookup occurs before a changed SpatialMapping has an
ArtifactIdentity. After independent Spatial verification and identity
assignment, finalization derives the persistent `ExecutionContextKey`,
rewrites `B_graph`, canonicalizes complete ServicePlan semantic keys, and emits
the exact owner-relative anchor/context selection rows. It then discards all
native flat handles, decision-atom ordinals, and H lookup state. Hierarchical
and flat search therefore share one endpoint-compatibility rule without
sharing mutable identity.

Mapping 4.0 derives each Fabric-owned `InstructionCoreContextRef` from the
selected AccCore and its one-per-AccCore cardinality. A selected service
plan element is addressed by
`ServicePlanElementRef = (ServiceRealizationKey, canonical plan ordinal,
typed element key)`, where the element key is exactly the natural key of the
referenced TransferLeg, MemoryRegion, or Consistency child owned by
`docs/spec-mapping-artifact.md`. Neither reference creates a second
target-selection decision.

System resource closure derives one InstructionCore use for every reachable
root/context pair in `B_thread`. It triggers on consumed root start and releases
causally on produced root completion. For an addressed-memory or fence member,
closure derives the unique rooted actor issue transition from Dataflow's
OperationSchema-owned `ActorHandshakeCase`, then selects exactly one admissible
Fabric UsePattern for every independently required service or consistency
ResourceContract. That pattern selection is a Candidate decision when more
than one legal pattern remains; the persistent `use_site_ref` is its only
authority. Capability ordinals, matching predicates, and selected-plan copies
never enter Mapping.

Provider-branch applicability is rebuilt from the exact selected Fabric
transform relation and the source address domain. A branch-local claim cannot
be charged to every address, omitted for an address it serves, or moved to a
different provider with an equivalent pattern. A memory exposure alone derives
no claim because it is not a Dataflow event. Static route claims remain
traversal-derived. Missing, duplicate, foreign, non-admissible, or
wrong-activation System ResourceUse records reject strict import and final
verification.

Any discrepancy between selected decisions and a rebuildable cache is an
internal invariant failure. Full owners report the drift and terminate the
attempt; they never overwrite the cache and continue.

For selected handshake legality, each restart owns dense arrays indexed by the
shared Frozen model:

```text
arc_refcount[potential arc]
active_arc_bitset[potential arc]
topological_order[node]
topological_rank[node]
```

An arc is active exactly when its reference count is nonzero. Reference counts
are required because several selected fragments may contribute the same owner
arc. Candidate adjacency is the immutable Frozen CSR filtered by the active
bitset; it is not rebuilt as a map or heap graph.

### SearchScratch

`SearchScratch` owns reusable A* distance, queue, predecessor, epoch, and
touched arrays; route overlays and arenas; matching and repair worklists; and
the active Action's PathFinder or DualSubgradient state. It has no semantic
identity and is discarded after its owner operation. Negotiation history is
never carried across Actions.

Handshake search scratch additionally owns epoch-marked forward and backward
visit arrays, parent arcs, bounded worklists, and a touched-rank journal. These
arrays are allocated before the search loop and reused. A local move performs
no heap allocation after warm-up.

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
affected Evaluation subjects before commit. The transaction also updates every
affected Fabric-owned handshake fragment and runs the selected combinational
handshake gate before commit. A newly closed directed cycle rolls back the
transaction as intrinsic invalidity. A resource change that invalidates
placement or routing follows the same closure.

Handshake mutation uses the decision-to-fragment and fragment-to-arc incidence
tables. It removes old contributions first; arc deletion cannot create a
cycle. It then inserts newly active arcs in canonical arc order. If the current
rank of an inserted arc's source is lower than its destination rank, the check
is constant time. Otherwise an array-based Pearce-Kelly bounded dynamic
topological update searches only the affected forward and backward regions. A
meeting produces a deterministic cycle witness and rolls back the complete
transaction; otherwise only the affected rank interval is reordered and
journaled. The implementation must not copy the complete `CandidateState` for
a move.

Exact references, domain membership, type and width compatibility, directed
connectivity, and a route being either explicitly unrouted or a valid rooted
arborescence are never relaxable. Implicit broadcast or merge, same-net
reconvergence, invalid tags, and unresolved references are not candidate
states. Neither is a selected combinational handshake cycle; it cannot become
a penalty, Evaluation metric, or policy-admitted temporary violation. An
explicitly unrouted partial candidate proves no final closure merely because
the missing arcs are absent. Only closed kinds admitted by
`TemporaryViolationPolicy` may remain in a committed search candidate; all
must be zero before finalization. Independent final verification discards all
incremental cycle caches and recomputes the complete selected graph from the
published facts.

### Native Performance Contract

Performance does not change Mapping semantics, but it is an implementation
admission requirement because the selected search kernels run inside every
Tech, Spatial, and System exploration. Canonical reference resolution,
sorting, and owner compilation occur once during freeze. Candidate mutation
uses dense `PnrIndex` values, contiguous SoA/CSR storage, bitsets, reverse
incidence, and preallocated scratch. Hot loops contain no persistent-reference
comparison, string lookup, virtual dispatch, global lock, or per-move heap
allocation.

The exact shared and worker-local byte counts are computed with checked
arithmetic before workers launch. The implementation selects the worker count
as:

```text
min(nproc - 4, 120, memory-derived worker limit)
```

The immutable `FrozenModel` is shared across workers. `CandidateState` and
`SearchScratch` are worker-local and aligned so workers do not mutate shared
cache lines. A 32-bit build should normally require approximately
`12 * potential_arc_count + 4 * fragment_incidence_count + O(node_count)` bytes
for the principal shared handshake arrays and
`4.125 * potential_arc_count + 8 * node_count + touched scratch` bytes per
worker. These are planning estimates, not wire-format constants; the checked
actual byte count governs admission and 64-bit builds use their actual layout.

Freeze is linear in the complete Fabric projection plus potential nodes, arcs,
and incidence after canonical owner-local ordering. Activating or removing a
selection is linear in changed incidence. A rank-respecting arc insertion is
constant time; a violating insertion is linear in the affected vertices and
active arcs, with unavoidable worst case linear in the selected graph. Initial
construction, global actions, and independent final verification use a full
linear Kahn or SCC pass rather than the incremental cache.

Focused performance anchors must prove:

* 64-way and 256-way atomic broadcast owner models grow linearly in fanout;
* duplicating a physical occurrence grows owner-model storage linearly and
  does not copy capability-row graphs;
* a local selection move performs no full graph rebuild or post-warm-up heap
  allocation;
* on the pinned 10,000-node, 50,000-potential-arc benchmark, median incremental
  update time is at least five times faster than full recomputation; and
* on representative regular and irregular 1,000-actor PnR workloads, selected
  handshake maintenance consumes at most ten percent of PnR CPU time. More
  than twenty percent blocks integration and requires profiling.

The complete Spatial PnR target remains a verifier-clean result within twenty
minutes for a supported approximately 1,000-actor graph, with peak RSS below
8 GiB. A performance failure cannot be repaired by reducing semantic work,
skipping the independent verifier, changing candidate order, or publishing a
best-so-far invalid Mapping.

## Edge Disposition And Routing

### Internal Realizations And Residual Nets

Every canonical software edge is accounted for exactly once. Closure first
recognizes every explicit realization-internal owner confirmed by `D/T/F` and
the selected physical facts:

* the configured FU relation of a Compute Realization;
* the Memory Realization's template-relative internal-connectivity witness,
  projected through the exact selected memory occurrence;
* an explicitly supported temporal PE register-file absorption; or
* another explicit Fabric internal realization with the same typed proof.

An internally realized edge has no residual logical net and no `RouteTree`.
Temporal PE co-location alone is not absorption: if the register file is
exposed as ordinary transport traversal rather than an internal realization,
the edge remains residual and the route must consume that explicit traversal
and its resources. The same rule applies to local selectors, switches, FIFOs,
boundaries, and module connections. No connection is inferred from ownership,
coordinates, names, or co-location.

A canonical actor result with no consumers is not a software edge and never
creates a residual logical net. If its selected FU template exposes the mapped
physical result at an FU output boundary, freeze derives one mandatory
occurrence-local PE output `Discard` from exact `D/T/F`. Candidate construction
must exclude an occurrence that cannot realize that discard. `Disconnected`
does not consume a produced token, and routing the dead result to an ordinary
transport endpoint would invent an obligation absent from `D`.

For every residual producer endpoint, freeze groups all residual sink
obligations into one deterministic multi-sink logical net keyed by the
`SpatialLogicalNetKey` defined above.

The producer and every sink use the Dataflow-owned graph-local token endpoint
catalog. Already internal sinks are omitted. If none remain, no logical net
exists. Exposure is this derived relation, not a competing endpoint kind.
Dense net indices are rebuildable native indices, never persistent identity.

Memory and other operation-relative services do not have a hard-coded request
route plus response route. The Canonical Service Schema mechanically derives
the exact abstract request, data, response, completion, or other transfer legs
required by each typed operation. Only residual legs become transfer
obligations. Spatial and System routing realize those legs without adding,
deleting, combining, or reinterpreting them.

For vector memory actors, each residual address, data, or mask operand remains
one complete logical token. Indexed addresses are one vector token, not a set
of lane routes. A Fabric-declared memory use pattern may turn one accepted
actor firing into several internal service transactions, but those
transactions remain internal execution events. Their potential resource claims
come from the selected Fabric use pattern and `ResourceUse`; they are not new
residual logical nets.

### Route Trees

Each residual logical net is realized by one rooted arborescence with shared
trunks and explicit branches. Its persistent field shape, owner key, canonical
node ordering, sink attachments, and System transfer-leg form are owned only
by `docs/spec-mapping-artifact.md`.

A System memory or fence service-leg RouteTree uses only transport endpoints
from the exact Fabric service-leg carrier attachment domains at its source and
sinks. The attachment relation supplies terminal candidates, not traversals:
all selected edges remain ordinary `FabricPhysicalTraversalRef` values in the
root-complete Transport Architecture. `MessageTransfer` continues to route
between its transport-plane service endpoints without this projection.

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
query payload width must fit the data field of every selected transport
endpoint and traversal. For a canonical multi-value service leg, that one
query width is the Fabric-owned derived envelope, while the ordered
`ServiceValue` tokens remain independent values under one transaction and one
shared route. Tag fields never contribute payload capacity. In a tagged
domain, the assigned tag must independently be representable without loss by
every tag field that still distinguishes the flow. Same-kind physical
connections may widen and later narrow according to Fabric's low-bit-aligned
rule, but no selected segment may narrow below the query payload width. Thus
an `i16` transfer may use `bits<32> -> bits<64> -> bits<32>`, but it may not
use `bits<8>` or borrow the tag field of `bits_tag<8,8>`. These checks are
structural legality and cannot be relaxed into congestion cost or repaired by
an implicit adapter.

The no-split and no-serialize rule applies independently to every service
value, including vector-memory address, data, and mask tokens. A route cannot
split one value over several endpoints, serialize it by convention, or assign
one Physical Tag per lane. The maximum-width envelope is not a packed tuple
and does not authorize a narrower path for any value. Any lane or memory-beat
decomposition occurs only inside the selected Fabric memory use pattern.

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

All terms are nonnegative. Evaluation guidance is absent unless routing names
one exact `ResolvedPnrEvaluationBindingRef` whose model safely supplies an
arc-local value encoded as the same Q-scaled `RouteCost` used by Mapping.
Once selected, unavailable guidance, a noncanonical guidance value, or a
failed guidance query fails the owning Action; it never becomes zero or
silently selects another model. Guidance can order proposals only. It cannot
filter legal arcs, prove legality, alter the Mapping-owned admissible
heuristic, or replace full `Q` evaluation.

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
 canonical_sink_attachment_index ascending,
 selected_target_endpoint_index ascending)
```

The sink-attachment index is derived from the persistent key owned by
`docs/spec-mapping-artifact.md`: the sink obligation for a Spatial route or
non-message service leg, and the `(terminal, execution owner)` pair for a
System message leg. It is a dense search index, not persistent identity.
The selected branch is normalized at its last intersection with the tree and
discharges exactly one sink attachment. Overlapping target domains do not
implicitly discharge multiple attachments. Failure of any sink rejects the
whole tree proposal; partial trees are never committed.

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

For traversal `a` and each claimed resource-state capacity dimension `r`, use
the cost-only projections defined by `Resource Use, Tags, Buffers, And Memory`:

```text
Q = 2^32

Q-scaled values:
  q_cost, x_cost, overuse_cost, H, lambda, RouteCost

dimensionless nonnegative integer scalars:
  P, history_pressure_increment, alpha

base_cost(a,r) = q_cost(a,r)
lower_bound_cost(a) = sum_r q_cost(a,r)

MultiplicativeCost(a,r) =
  ceil(q_cost(a,r)
       * (Q + P * x_cost(a,r))
       * (Q + H(r))
       / Q^2)

AdditiveCost(a,r) =
  q_cost(a,r)
  + P * x_cost(a,r)
  + ceil(q_cost(a,r) * H(r) / Q)

arc_cost(a)             = sum_r ResourceCost(a,r)
```

A Multiplicative resource cost has exactly one final ceiling after the complete
three-factor product. Staged Q-scaled multiplication is noncanonical because
its intermediate ceiling can change deterministic route order. Additive terms
use the one ceiling shown above. Callers cannot reinterpret a Q-scaled value as
a raw amount or duplicate these formulas.

A pure structural traversal with no claim may have zero cost. Both kernels
share occupancy, iteration order, update rules, A*, route trees, and
transaction protocol. PathFinder uses deterministic Gauss-Seidel occupancy:
each net removes only its selected old claims, reroutes against the current
working overlay, and installs its new claims. `P` and `H` remain frozen within
the iteration. The next iteration uses:

```text
O_k(r) = overuse_cost_k(r)
H_(k+1)(r) = H_k(r) + history_pressure_increment * O_k(r)
P_(k+1) = ceil_mul_div(P_k,
                       present_pressure_growth_numerator,
                       present_pressure_growth_denominator)
```

`H_0(r) = 0`, `P_0 >= 1`, history increment is at least one, and the reduced
growth ratio is at least one. Updates occur atomically only after a complete,
non-closed iteration. At the start of each iteration, the complete current
cost projection is derived from the complete working route overlay before any
net is selected. Ripping up or installing one net may then update only costs
whose raw occupancy changed, but every untouched cost retains that complete
baseline rather than reverting to its lower bound. At the start of iteration
`k`, PathFinder also derives this complete key from the previous complete route
overlay and then freezes it for the whole iteration:

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
  sum_r ceil(q_cost_k(n,r) * overuse_cost_k(r) / Q)
```

A shared Route Tree prefix contributes once under normalized claim semantics.
The optional Evaluation priority comes from the invocation-frozen exact
Evaluation binding and is zero when unavailable; Mapping cannot substitute a
private criticality. Checked nonnegative integer arithmetic and canonical net
index make the key total. It is recomputed only from the next complete overlay,
never from per-net Gauss-Seidel updates. There is no seeded shuffle, permanent
container order, ordering plugin, or hidden weight.

DualSubgradient routes every net independently against a fixed price snapshot
and updates prices only after the complete synchronous iteration. Region-
external fixed occupancy is subtracted from raw physical capacity before
normalization. At price snapshot `lambda_k`, each selected traversal uses:

```text
dual_arc_cost(a) =
  sum_r ceil(q_cost(a,r) * (Q + lambda_k(r)) / Q)
```

After routing the complete region, `U_k(r)` is the total raw selected amount
and `C(r)` is the effective raw capacity. Their signed difference is normalized
once with the same `Q` scale. Per-claim rounded costs are not summed to
determine the pressure sign:

```text
g_k(r) = sign(U_k(r) - C(r))
         * ceil(abs(U_k(r) - C(r)) * Q / C(r))

ProjectedSigned:       d_k(r) = g_k(r)
PositiveViolationOnly: d_k(r) = max(0, g_k(r))
MomentumDeflected:     d_k(r) = g_k(r) + beta * d_(k-1)(r)

lambda_(k+1)(r) = max(0, lambda_k(r) + alpha_k * d_k(r))
```

`g`, `d`, and `lambda` use the same Q scale. `alpha` is the dimensionless
integer returned by the selected step schedule. `beta` is the exact ratio
`beta_numerator / beta_denominator` with
`0 <= beta_numerator < beta_denominator`. The single numeric protocol is:

```text
DualPrice     = uint64_t
DualDirection = int64_t
DualStep      = uint64_t
```

Every operation uses checked widened integer arithmetic and one owner-provided
Q-scaled multiplication primitive. Every finite published cost is strictly
less than `UINT64_MAX`, which remains the A* infinity sentinel. Overflow
rejects and rolls back the Action; wrap, saturation, dynamic rescaling,
floating-point substitution, and representation switching are forbidden.

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
`SelectedObjectiveClosure` using route-related Mapping `V/G`, not A* cost or
private prices. Equal rank retains the earlier canonical iterate. Work
exhaustion is not infeasibility. Zero violation is the only normal early-
convergence test; there is no epsilon, stagnation window, route-signature cycle
detector, or hidden no-progress threshold. Exhausting the routing policy's
owner-local work limit returns the best admissible temporary iterate only for
a non-final Action; otherwise it returns typed non-closure and rolls back.
Final global closure never returns a temporary iterate.

After a complete non-closed iteration, Negotiated Routing may also derive an
exact fixed-terminal capacity-cut certificate. For one overused physical
capacity dimension `r`, choose any canonical subset `N` of participating nets.
For each `n` in `N`, remove from `n`'s frozen payload-compatible routing graph
every traversal carrying a positive claim on `r`. If at least one selected sink
of `n` is then unreachable from its selected source, every legal route for that
fixed terminal assignment must consume `r`. Define:

```text
minimum_claim(n, r) =
  min positive raw amount among payload-compatible traversals of n claiming r

mandatory_usage(r) =
  initial_occupancy(r)
  + sum_n minimum_claim(n, r)
```

The sum ranges only over nets for which the removal test establishes the
separating cut. All arithmetic is checked. If `mandatory_usage(r) >
capacity(r)`, the current frozen source, sinks, placement, and attachment
assignment cannot reach capacity closure. This is an exact conditional proof,
not a stagnation heuristic and not proof that another terminal assignment,
placement, SpatialCore, or Fabric is infeasible. Implementations may use the
nets contributing to the current overuse as the conservative canonical subset;
failure to establish the inequality simply continues normal negotiation.

The cut test is part of the completed negotiated-iteration work unit and runs
before pressure, history, price, or direction advances. A non-final invocation
whose policy admits the remaining violation immediately restores and returns
its best admissible temporary iterate so that outer realization, placement, or
resource Actions can change the frozen assignment. Otherwise the invocation
returns typed non-closure and rolls back. Checked routing-cost overflow remains
an Action failure when no earlier exact certificate applies; saturation, wrap,
dynamic rescaling, and treating overflow as infeasibility remain forbidden.

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

Capacity legality and search cost use separate numeric projections. For each
Fabric-owned resource-state capacity dimension `r`:

```text
amount(a, r) = exact integer sum of the Fabric uint32 claims induced by a
capacity(r)  = exact Fabric uint32 capacity

usage_raw(r)   = exact integer sum of all selected and fixed amounts
overuse_raw(r) = max(0, usage_raw(r) - capacity(r))
```

Each equation is evaluated for one exact owner-derived concurrent occupancy
query. "All selected and fixed" means every claim in that event-relative
overlap envelope, not every `ResourceUse` record in the artifact regardless of
time. Fabric use patterns and Mapping activation algebra remain the only owner
of those envelopes.

Base legality, temporary `CapacityOveruse`, and final zero-overuse closure use
only these raw integers. An individual atomic claim envelope whose amount
exceeds capacity is inadmissible before search. A zero-capacity dimension
therefore admits only zero amount. A normalized or rounded value must never
decide capacity legality.

Search cost alone uses the fixed scale `Q = 2^32`:

```text
q_cost(a, r) = ceil(amount(a, r) * Q / capacity(r))

x_cost(a, r) =
  ceil(max(0,
           usage_raw_before(a, r) + amount(a, r) - capacity(r))
       * Q / capacity(r))

overuse_cost(r) = ceil(overuse_raw(r) * Q / capacity(r))
```

These divisions are evaluated only for positive capacity. A zero amount on a
zero-capacity dimension has zero cost; every positive amount was already
rejected as inadmissible.

`usage_raw_before` is the exact working occupancy after removing only the old
claims replaced by the current proposal. All products, sums, and ceilings use
checked widened integer arithmetic and produce checked `uint64` values. A
positive amount therefore has positive `q_cost`; zero amount has zero cost.
Rounding may change proposal order but cannot create or erase a raw violation.
The normalized values encode the nonnegative real value `encoded / Q`; they
are not raw capacity units. Their composition is owned by `Negotiated Routing`
and cannot use unscaled integer multiplication.

Each stateful Fabric resource schema also owns its closed typed
`ResourceState` set, canonical initial state, capacity dimensions, atomic
UsePatterns, stable typed requester order, and exact GrantPolicy or exact
refinement domain. One UsePattern may atomically claim multiple states. PnR
may select an exposed refinement and bind workload values, but it cannot split
an atomic pattern or construct a parallel generic resource/arbiter graph.

The base `fabric.boundary` contributes one stateless atomic use over all active
input and output legs. A Temporal PE contributes operand-entry, queue-order,
enqueue-service, and dequeue-service claims derived from its exact mode and
required `operand_buffer_size`. Freeze and search must not substitute an
implicit queue depth, split a boundary join/fork, or replace the declared
round-robin grant relation with candidate order.

PnR emits a persistent use only for a non-derived activation, reservation,
release, or sharing assignment required by that schema. Static claims implied
by a selected traversal are not duplicated, and multiplicity derives from
software obligations and pattern parameters rather than duplicate records. A
causal `AllOf` release holds occupancy until the Fabric-local release point and
every schema-selected event occurrence are complete; runtime cannot infer an
earlier release from observation of a subset, fairness, or record order. PnR
derives one wait dependency per conjunct and never creates an aggregate event
node or splits the claim envelope.

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

The initial exact Spatial provider proves absence of a closed wait set when
the Canonical Dataflow actor-dependency relation is acyclic, selected
combinational handshake closure is acyclic, every required `ResourceUse` is
present, and every selected Fabric contract supplies its validated atomic
progress guarantee. Topological induction then establishes that some enabled
actor can retire under fair execution. A feedback cycle is not itself a
deadlock witness; until a supported typed token, finite-buffer, and initial
occupancy analysis proves or refutes that cyclic case, Mapping returns
`Incomplete(proof_not_established)`. The actor projection and topological
proof are linear in the canonical actor and dependency counts and are rebuilt
independently by final verification.

Physical Tag is local to Fabric-owned interpretation domains. A selected value
is stored exactly once in the sharing assignment of a real temporal writer or
tagged ingress. Route trees and Fabric writer, rewriter, and remover points
mechanically divide the route into continuity segments. Closure intersects
every segment's allowed match domains and builds local interference from
co-residency and incompatible interpretation. Switch rows, operand matches,
memory rows, and encoded tag fields are derived from the one origin value.

Each residual external Temporal memory operand or result role is an independent
continuity endpoint. PnR derives that role's input match or output write from
the corresponding real writer or ingress assignment. It does not introduce a
row-wide tag variable, require all roles in one memory operation to share a
value, or use the operation kind as a match key. TechMapping-selected memory-
internal sources, projected through the exact occurrence relation, remove the
corresponding external continuity obligation; SpatialMapping does not select
them again.

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

Memory operation placement, MemoryOperationEntries, and memory bindings use only the
closed persistent forms owned by `docs/spec-mapping-artifact.md` and
`docs/spec-mapping-memory.md`. PnR selects those typed choices in native state;
each addressed MemoryOperationEntry and ExposureEntry selects exactly one
`LocalMemoryServiceRef | ManagerEndpointRef` target, while each FenceOperation
selects one `MemoryConsistencyDomainRef | ManagerEndpointRef` target. Those
fields are the persistent `C_dispatch`; PnR checks them against Fabric-owned
`H_dispatch` and does not define another dispatch relation. The Canonical Service Schema
owns operation legs and their ordering, while route trees and service plans
realize residual legs. Provider decode, dispatch rows, response tracking, and
semantic `sw_configs` are derived from `D/T/F`, selected bindings, routes,
resource uses, and physical refinements. Physical image fields are later
encoded through the exact ConfigurationABI.

For addressed and exposure entries, PnR jointly selects the dispatch target
and the referenced MemoryBinding target. Local dispatch requires LocalRegion
in that exact service and participates in local region-capacity and address
checks. Manager dispatch requires BoundaryProxy and creates no candidate
system provider variable during Spatial search. System provider and address
selection remain SystemMapping decisions.

## Search Policy And Determinism

### Resolved View

Spatial and System PnR have distinct component-view descriptors:

```text
loom.spatial_pnr.config.2.0
loom.system_pnr.config.2.0
```

They use the same field types and codecs but project the independently selected
Spatial or System policy domain. Their exact descriptor bytes are the ASCII
bytes shown above without a trailing zero. A digest from one view kind cannot
be adopted as the other.

Each resolved view is self-contained:

```text
ResolvedPnrConfigView {
  search_policy: SearchPolicy
  determinism_policy: DeterminismPolicy
  temporary_violation_policy: TemporaryViolationPolicy
  selected_objective_closure: SelectedObjectiveClosure
}

SelectedObjectiveClosure {
  evidence_obligation_templates:
    canonical sequence<EvidenceObligationTemplate>
  objective_dimensions:
    canonical sequence<ObjectiveDimension>
  weighted_levels:
    canonical sequence<WeightedLevel>
  total_orderings:
    canonical sequence<TotalOrdering>
  selected_total_ordering: TotalOrderingRef
  selected_search_energy: SearchEnergyRef
  focused_closure_dimensions:
    canonical set<ObjectiveDimensionRef>
  evaluation_bindings:
    canonical sequence<ResolvedPnrEvaluationBinding>
}

ResolvedPnrEvaluationBinding {
  obligation_template: EvidenceObligationTemplateRef
  interaction_domain: EvaluationInteractionDomainRef
}
```

`EvidenceObligationTemplateRef`, `ObjectiveDimensionRef`, `WeightedLevelRef`,
`TotalOrderingRef`, and `ResolvedPnrEvaluationBindingRef` are zero-based
`uint32` ordinals into the corresponding table in this exact view.
`SearchEnergyRef` is the DSE-owned role-specific alias of one local
`WeightedLevelRef`. The canonical bytes use the Common component-view digest
contract; neither descriptor nor digest is embedded again inside the view.

The record schemas and semantic keys for obligation templates, dimensions,
levels, orderings, and interaction domains remain owned by
[DSE Feedback](spec-dse-feedback.md#objectives-and-quality-gates). The
references above are local to the exact PnR view. The central projector starts
from the selected total ordering, search energy, focused-closure dimensions,
and routing guidance binding, computes their complete transitive closure,
sorts every owner-typed catalog by its complete semantic key, assigns dense
view-local ordinals, and rewrites every internal reference. It does not retain
an ordinal into `ResolvedDseConfigView`, copy an unselected record, or include
the complete DSE-view digest.

Every Evaluation metric dimension in the selected rank, energy, or focused
closure has exactly one binding to its obligation template. Objective use
requires that the exact model descriptor admit the binding's interaction
domain in `Incremental` mode. A routing `route_guidance_binding`, when present,
requires `Guidance` mode. A binding may satisfy both uses only when the exact
descriptor advertises both modes for that same domain. Required modes are
derived from uses and are not serialized in the binding. Missing, duplicate,
foreign, stale, or mode-incompatible bindings are invalid at projection or
adoption.

The PnR acquisition input catalogs are fixed by the component-view descriptor:

```text
SpatialPnrEvidenceInputSlot =
    0 CanonicalDataflowProgram(D)
  | 1 TechMapping(T)
  | 2 Fabric(F)
  | 3 MappingConstraintSet(K)

SystemPnrEvidenceInputSlot =
    0 CanonicalDataflowProgram(D)
  | 1 Fabric(F)
  | 2 MappingConstraintSet(K)
```

An `EvidenceAcquisitionInputSlotRef` in a selected template is a `u32be`
ordinal in the corresponding catalog. The exact invocation supplies those
Artifacts; schema and case-role validation must succeed before candidate state
is allocated. System `R` and `H` are typed invocation views rather than
Artifacts and cannot be smuggled through an Evaluation subject slot. A model
that needs another noncandidate Artifact must fix it exactly in the template.
There is no dynamic subject map or slot-name lookup.

The template remains the ordinary reusable Evaluation owner record. During
search, the exact invocation binds its noncandidate subject roles and the
ephemeral `PnrCandidateView` supplies the distinguished candidate role. After
a Mapping is published, the same template instantiates an ordinary
`EvaluationRequest` for the full oracle. PnR does not own a second request,
model-binding, metric, or finding schema.

Every focused-closure dimension must use an `EvaluationMetricSource`. An empty
focused-closure set disables metric-triggered focused closure without changing
annealing or final rank. An empty Evaluation binding table is valid only when
the selected closure contains no Evaluation metric and routing selects no
guidance.

`SearchPolicy` has exactly this closed shape:

```text
SearchPolicy {
  initializer {
    seed_attempt_count: positive uint32
    assignment_attempt_limit_per_seed: positive uint64
  }
  action_proposal {
    realization_binding_weight: uint64
    transport_routing_weight: uint64
    resource_allocation_weight: uint64
  }
  routing {
    endpoint_expansion_limit: positive uint64
    negotiation_iteration_limit: positive uint64
    negotiation_policy: RoutingNegotiationPolicy
    route_guidance_binding: optional<ResolvedPnrEvaluationBindingRef>
  }
  annealing {
    calibration_proposal_count: positive uint64
    positive_delta_quantile: ExactRatio
    target_initial_acceptance: ExactRatio
    fallback_temperature: positive uint64
    minimum_temperature: positive uint64
    cooling_ratio: ExactRatio
    proposals_per_level_base: uint64
    proposals_per_movable_decision: uint64
  }
  focused_closure {
    proposal_limit: positive uint64
  }
  exact_repair:
      Disabled
    | CpSat {
        max_region_decisions: positive uint64
        max_solver_calls: positive uint64
      }
}
```

`ExactRatio` is the reduced `uint64 numerator/denominator` pair with a positive
denominator. The quantile admits zero and one, target acceptance is strictly
between zero and one, and cooling is strictly between zero and one. The Action
weights are nonnegative, cannot all be zero, and are reduced by GCD. Empty
Action kinds are removed before the remaining weights are normalized. The two
proposal-count terms cannot both be zero. Inactive union fields are invalid.

The canonical view encoder writes fields in the schema order above. Closed
union and enum discriminants and local references use `u32be`; counts and
numeric limits use `u64be`; ratios encode numerator then denominator; optionals
use a `u32be` absent/present discriminant followed by the payload when present.
Canonical sequences use `u64be(count)` followed by records in semantic-key
order, except TotalOrdering level order, which remains semantic. Adoption
validates the component-view digest, decodes the complete value, validates and
canonicalizes all owner records, re-encodes, and requires exact byte equality.
There is no JSON, property-map, raw-byte callback, or compatibility codec for
this view.

`TemporaryViolationPolicy` selects from the closed Mapping `V` descriptor
registry defined below. Only selected descriptors may remain nonzero in a
committed search candidate, and every selected kind must be visible to the
resolved objective. These are search permissions, not final legality or
budgets. Structural errors are never temporary.

Every typed policy that defines a semantic work unit owns its numeric limit.
Initialization owns seed-attempt work; annealing owns calibration and Action
proposal work; routing owns local A* and negotiated-iteration work; focused
closure and exact repair own their work; and final global closure owns its
Action work. The resolved controller derives only the read-only
`DeterministicWorkBudgetView` owned by `docs/spec-dse-feedback.md`. Its
owner-defined work-unit references and limits cannot be authored or
reinterpreted by PnR. Worker count, wall time, memory
reservation, licenses, process retries, and external cancellation are
execution controls and cannot change the formal candidate sequence.

### Initial Builtin Policies

ResolvedConfig 3.0 emits every field; the values below are schema data, not
PnR-kernel defaults. All initial builtin profiles use Action weights
`1:3:2` for realization, routing, and resource Actions; PathFinder
`Multiplicative` with initial pressure `1`, reduced growth ratio `3/2`, and
history increment `1`; quantile `3/4`; target acceptance `4/5`; fallback
temperature `1024`; minimum temperature `1`; cooling ratio `19/20`; no route
guidance; and deterministic master seed `0` with the protocols named above.

```text
profile                seeds  assignments  endpoint_expansions  negotiations
report_only                1         4096                16384             8
quick_explore              2        16384                65536            16
balanced_explore           4        65536               262144            64
performance_explore        8       262144              1048576           128
implementation            16       524288              2097152           256
strict_implementation     32      1048576              4194304           512

profile                calibration  level_base  per_movable  focused
report_only                     16          16            1       64
quick_explore                   64          64            2      512
balanced_explore               256         128            8     4096
performance_explore            512         256           16     8192
implementation                1024         512           24    16384
strict_implementation         2048        1024           32    32768

profile                exact_repair
report_only            Disabled
quick_explore          CpSat(64, 128)
balanced_explore       CpSat(256, 1024)
performance_explore    CpSat(512, 4096)
implementation         CpSat(1024, 8192)
strict_implementation  CpSat(2048, 16384)
```

The two `CpSat` values are `max_region_decisions` and `max_solver_calls`.
Every initial profile admits all five Mapping violation descriptors as
temporary and requires all five to be zero at finalization. Its selected
closure contains one Minimize dimension per violation and one for
`TotalSelectedTraversalClaim`, each with origin `0`, quantum `1`, and bounds
`[0, UINT64_MAX]`. Final total ordering first compares one equal-weight level
over all violations, then the traversal-claim level. Search energy is a third
level containing every violation with weight `4294967296` and traversal claim
with weight `1`; the checked `uint128` accumulator covers the complete declared
domain. Focused-closure dimensions and Evaluation bindings are initially empty.
The resolver emits all three levels explicitly and canonicalizes their weights;
PnR does not synthesize them.

### Deterministic Initialization And Action Proposal

Each fixed initializer attempt starts from the same immutable FrozenModel and
an empty selected-decision assignment. It uses this exact bounded protocol:

1. propagate singleton domains and hard relation consequences to a fixed
   point;
2. if every decision is assigned, run one explicit global
   `TransportRoutingAction` and validate the resulting seed;
3. otherwise choose the unbound decision with the smallest current legal
   domain, breaking ties by its canonical typed decision key;
4. derive a without-replacement choice order from that canonical domain; and
5. attempt each choice, propagate, and either recurse or roll back the complete
   assignment and all derived state on contradiction.

The first relation solve is the baseline root assignment. Seed attempt zero
uses canonical choice order. Every other seed uses only its
`InitializerDiversification` PRNG stream: repeated `nextBounded(remaining)`
selection over the canonical remaining domain defines its deterministic
without-replacement permutation.

Before dependent decisions become active, one exact preference refinement may
replace the baseline choices of independent compute roots. The refinement uses
this closed protocol:

1. a compute root participating in any constraint-owned relation retains its
   baseline choice;
2. memory roots retain their baseline choices;
3. process every remaining compute root in canonical owner order and select a
   legal choice whose exact `InstructionContextRef` currently has the least
   selected-root count;
4. among equal-count choices, minimize the sum of directed frozen-topology hop
   distances to already processed compute roots joined to this root by a
   logical net. For each such producer-consumer incidence, the distance is the
   minimum number of payload-compatible `FrozenSpatialRoutingGraph` arcs over
   the attachment endpoints admitted by the two candidate placements. An
   unreachable incidence ranks after every finite distance but does not remove
   the choice. If there is no already processed compute neighbor, or scores
   remain equal, use circular canonical choice order beginning at that root's
   baseline choice;
5. any attachment root participating in a constraint-owned relation retains
   its baseline choice. Process every other occurrence-relative
   `PortAttachment` root in canonical demand order and every other
   graph-boundary attachment root in canonical boundary order. Restrict a
   `PortAttachment` root's preference candidates to attachment options owned
   by the already selected realization placement; a graph-boundary root uses
   its existing legal attachment domain. Choose an exact physical endpoint
   with the least current selected-attachment count. Ties use circular
   canonical choice order beginning at that root's baseline choice;
6. invoke the same root relation solver once with the preferred compute and
   attachment roots fixed, so placement compatibility and every hard
   attachment relation are re-established by their existing owner; and
7. if the preferred fixed roots have no complete assignment within the
   remaining initializer work, retain the complete baseline assignment.

The selected-root count, frozen-topology distance, and selected-attachment
endpoint count are only deterministic search preferences. None is capacity, none
removes a legal choice, and none can prove infeasibility. The distance and
endpoint identity use the same immutable endpoint, arc, payload-width, and
attachment domains already owned by the FrozenModel; they ignore mutable route
costs and resource occupancy and therefore do not form a second router or
crosspoint model. Fabric `ResourceContract`, raw candidate capacity projection,
and final verification remain the only capacity authorities. The refinement
consumes no PRNG words beyond the baseline solve. Every relation assignment
attempted by either solve consumes one initializer work unit under the single
shared limit. A work-limit stop without an already complete baseline is
incomplete initialization, not infeasibility. The configured seed-attempt
slots are fixed before execution; a failed slot is never replaced by an extra
attempt.

The current legal domain is defined only for an active decision. Realization
binding decisions and graph-boundary attachment decisions are roots. They
share one MRV relation model, so independent graph boundaries participate in
the same singleton propagation and choice ordering as hard realization
relations. This root relation model reaches a complete assignment before any
dependent decision becomes active. An occurrence-relative `PortDemand` or
memory operation plan then resolves against its selected owner. An addressed
memory dispatch or exposure becomes active only after its exact `MemoryBinding`
target is selected. Inactive decisions are not empty domains and do not consume
work or PRNG words.

The canonical typed decision-key order is:

```text
RealizationBinding(Compute before Memory, owner ordinal)
GraphBoundaryAttachment(boundary ordinal)
PortAttachment(demand ordinal)
MemoryOperationPlan(actor ordinal)
LogicalMemoryBinding(binding ordinal)
MemoryUseDispatch(rooted-use ordinal)
MemoryExposure(exposure ordinal)
```

Owner arrays define each canonical choice order and the physical-context
preference only chooses where that order begins. The one
`assignment_attempt_limit_per_seed` applies to the complete attempt across the
root relation model and every subsequently activated decision; entering a new
owner stage does not reset it. A globally policy-admitted capacity or
resource-time violation does not remove an otherwise structurally legal
initializer choice. It remains a selected Candidate violation for the search
objective and later closure.

Initializer-local physical offsets do not enumerate every byte in a service
region. Logical memory bindings become active in canonical binding order. For
each selected local target, the initializer places the binding immediately
after the already selected canonical-prefix bindings on that target; a
BoundaryProxy uses offset zero. This left-compacted projection is complete for
initializer feasibility because the current local-memory contract observes
only containment and non-overlap, not gaps. It retains `O(bindings * targets)`
factorized target work rather than a byte-address Cartesian domain. A later
typed Action may select any other owner-legal offset when an exact selected
contract makes that distinction observable.

Transport-routing intent has exactly these scopes:

```text
TransportRoutingScope =
    WholeNet(exact logical net or service leg)
  | SingleSink(exact sink obligation)
  | RootedSubtree(exact net, current route-tree root node)
  | WitnessRegion(exact typed unresolved witness)
  | Global
```

`RootedSubtree` denotes the complete current subtree rooted at the selected
node. `WitnessRegion` is the deterministic Action dependency closure of one
typed Mapping or Evaluation witness. Neither form carries an arbitrary sink
set. There is no generic sink-subset powerset, hidden route list, or callback
scope. These scopes compose through successive Actions and therefore retain
whole-net, branch, local-region, and full-design repair capability.

For every Action proposal, the dynamic domain contains only kinds with at
least one legal anchor and choice. The selector reduces the configured
nonnegative kind weights by GCD, calls `nextBounded(sum_of_live_weights)` once
to choose the kind, then calls `nextBounded` over the canonical anchor domain
and canonical typed choice domain. It does not retry through empty anchors,
use container order, or consume host entropy. Each proposal slot has the same
formal selection calls whether the later transaction commits or rolls back.

### Objective Projection

Mapping owns the closed `V` descriptors:

```text
UnroutedObligation
CapacityOveruse
TagUnassigned
TagConflict
HardProgressViolation
```

The typed static registry descriptor identity is
`loom.mapping.pnr.objective`, version 2.0. A
`MappingViolationDescriptorRef` is that descriptor plus the zero-based ordinal
in the closed catalog order shown above; it is not a string key. Each
descriptor owns one exact nonnegative integer magnitude. Mapping also owns the
initial closed domain-independent `G` catalog, which contains exactly:

```text
TotalSelectedTraversalClaim
```

`MappingMeasureDescriptorRef` uses the same registry descriptor and a
zero-based owner-local ordinal in this separate typed catalog. Version 2.0 has
the single ordinal zero.

The five violation magnitudes have these exact owners:

* `UnroutedObligation` is the checked count of residual logical sink
  obligations that are not covered by their selected complete Route Tree.
* `CapacityOveruse` is the checked sum of exact raw overuse over the canonical
  unique concurrent occupancy queries derived from selected Mapping facts and
  Fabric-owned resource contracts. One query combines all claims and durable
  state occupancy that may coexist for one exact owner state and capacity
  dimension. Pipeline or result holding, event-relative claim overlap, FIFO or
  operand-buffer occupancy, enqueue or dequeue service slots, memory-service
  outstanding capacity, context capacity, and route-resource capacity are not
  separate Mapping violation kinds. Their exact Fabric owner and typed witness
  remain available for diagnostics and focused repair.
* `TagUnassigned` is the checked count of required Physical Tag assignment
  origins without a selected canonical value.
* `TagConflict` is the checked count of exact local interference relations
  whose endpoints select the same Physical Tag value.
* `HardProgressViolation` is zero when the supported exact progress analysis
  proves no reachable closed wait set and one when it proves at least one.
  The verifier retains the canonical first typed witness for focused repair and
  diagnostics rather than turning witness count into another objective. When
  the supported analysis can establish neither progress nor a counterexample,
  the source is unavailable and the invocation is
  `Incomplete(proof_not_established)`; it is never zero by default.

Raw capacity legality is evaluated before any normalized search projection.
For each canonical occupancy query and capacity dimension, the contribution is
`max(0, usage_raw - capacity)`. The query partition must be disjoint for the
same owner state and event-relative occupancy cell, so a use cannot contribute
again under a route, time, buffer, or service label. Q-scaled pressure remains
search-only guidance and cannot create, erase, or replace this exact raw
violation.

An incompatible operation or service capability, malformed `ResourceUse`,
missing provider, or structurally impossible choice is rejected by typed
domain construction or base verification rather than converted into a
capacity magnitude. A service latency, throughput, power, or quality shortfall
belongs to exact `K` admission or Evaluation. A proven permanent wait belongs
only to `HardProgressViolation`. There is no `ResourceTimeOverbooking`,
`BufferOveruse`, or `HardServiceContractShortfall` alias or compatibility
projection in registry 2.0.

`TotalSelectedTraversalClaim` is the checked sum of `q_cost` over every unique
selected traversal claim envelope in the candidate. One shared Route Tree
prefix contributes once, regardless of sink count. A selected traversal with
no claim contributes
zero. PathFinder pressure, history, dual price, A* queue state, proposal count,
and search order are scratch and never enter `G`.

Evaluation owns `MetricKind`, `MetricObservation`, and typed findings for `Q`.
Structural invalidity, a `K` failure, or a base-verifier failure is never
converted into `V` or an objective penalty.

The source algebra, exact resolved Evaluation bindings, objective dimensions,
exact affine quantization, ObjectiveVector, WeightedLevel, TotalOrdering,
SearchEnergyRef, and quality-gate CNF are owned only by
`docs/spec-dse-feedback.md` and `docs/spec-evaluation-metrics.md`.
`ResolvedPnrConfigView` carries mechanically derived references to those
records for this invocation; PnR does not restate or extend their schemas.

Before freeze, PnR preflights the exact projection and every selected hot
Evaluation binding against its candidate subject projection, requested
observations, full and incremental interfaces, candidate completeness, and
all temporary `V` kinds admitted by SearchPolicy. Authorized alternatives are
already resolved; an unavailable required source produces typed
`ObjectiveUnavailable` and cannot select another provider, become zero,
infinity, NaN, or invoke a private fallback. A temporary violation kind that
SearchPolicy admits into committed candidates must have a positive objective
term in the WeightedLevel selected by
`SelectedObjectiveClosure.selected_search_energy`
so search cannot erase its closure obligation. Specifically, that `V`
descriptor must be referenced by a Minimize dimension with `origin = 0`,
`quantum = 1`, and a positive reduced weight.

Evaluation metric dimensions reached from `SelectedObjectiveClosure` provide
ephemeral `Q` guidance. Those dimensions create no formal Request, Evidence,
or pre-publication gate obligation. Formal quality-gate truth and candidate
selection remain post-publication `Promote` behavior. A quality gate is never
converted to a numeric deviation.

```text
energy(candidate) =
  value(SelectedObjectiveClosure.selected_search_energy,
        ObjectiveVector(candidate))

rank(candidate) =
  (TotalOrdering.weighted_level_values lexicographically ascending,
   canonical candidate semantic key ascending)

reward(old, new) =
  signed_difference(energy(old), energy(new))
```

The semantic candidate key breaks equal-rank ties but does not enter the
annealing delta. Seed initialization, negotiated best iterate, focused
closure, repair, and final local rank consume the same resolved
TotalOrdering. Annealing and RL consume the explicitly selected
SearchEnergyRef.
Pareto remains a post-publication central Promote operation over the shared
ObjectiveVector dimensions. PnR does not own another score, gate, direction,
normalization, or ordering.

### Annealing And Replay

The annealing policy is the single fixed-point protocol:

```text
SearchPolicy.annealing {
  calibration_proposal_count
  positive_delta_quantile
  target_initial_acceptance
  fallback_temperature
  minimum_temperature
  cooling_ratio
  proposals_per_level_base
  proposals_per_movable_decision
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

`movableDecisionCount` counts each canonical typed selected-decision anchor
whose current dynamic Action domain contains at least one legal alternative,
plus one routing decision for each residual logical net or service leg.
`SingleSink`, `RootedSubtree`, `WitnessRegion`, and `Global` are neighborhoods
over those routing decisions and do not add decisions. Choice cardinality does
not multiply the count. The domain is rebuilt once at level start for this
count; later proposal-local rebuilds cannot change the already fixed level
length.

Calibration rolls back its fixed proposal count, sorts positive deltas in
stable order, and selects index `floor(q * (n - 1))`. It chooses the minimum
positive integer temperature that reaches the target under the exact
acceptance kernel. The absence of positive deltas uses `minimum_temperature`;
an invalid estimate uses `fallback_temperature`. Cooling is:

```text
T_next = max(minimum_temperature,
             floor(T * cooling_ratio.numerator
                     / cooling_ratio.denominator))
```

The annealer executes exactly one complete proposal level at
`minimum_temperature`, then terminates. If calibration chooses a temperature
at or below the minimum, that first level is the required minimum-temperature
level. Otherwise, after each complete level above the minimum, it computes
`T_next`; the first level whose temperature equals the minimum is executed in
full and is the last level. There is no separate temperature-level limit,
reheating, online acceptance-ratio adaptation, wall-time or stagnation
termination, or accepted-Action budget.

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

The `Sha256SeededXoshiro256StarStar_1_0` seed preimage is exactly:

```text
ASCII("loom.pnr.prng.sha256_seeded_xoshiro256starstar.1.0")
  || u64be(master_seed)
  || u64be(seed_index)
  || u32be(stream_purpose_ordinal)
```

The fixed ASCII domain separator is 50 bytes and has no terminator. Stream
purpose ordinals are `InitializerDiversification = 0`, `Calibration = 1`,
`ActionProposal = 2`, `Acceptance = 3`, and `ExactRepair = 4`. The four
consecutive 64-bit big-endian words of the SHA-256 digest initialize the
xoshiro256** state. If all four words are zero, word zero is replaced by
`0x9e3779b97f4a7c15` and the other words remain zero.

One `nextU64()` result and transition use the published xoshiro256** 1.0
integer recurrence: result is `rotl(state[1] * 5, 7) * 9`; the state update
uses `t = state[1] << 17`, the xor sequence `(2 ^= 0, 3 ^= 1, 1 ^= 2,
0 ^= 3, 2 ^= t)`, and `state[3] = rotl(state[3], 45)`, all modulo `2^64`.
`nextBounded(n)` requires positive `n`, repeatedly draws `x`, rejects while
`x < ((0 - n) mod 2^64) mod n`, and otherwise returns `x mod n`. This is the
only bounded-selection projection; library distributions are not equivalent.

For positive `deltaE`, the acceptance protocol computes
`ceil(deltaE * 256 / temperature)`, looks up the canonical Q64 exponential
threshold, and accepts only when the next `u64` is strictly below it.
Non-positive deltas accept directly. The checked-in table is the protocol;
runtime `exp` is not.

`ExpNegativeQ64Table_1_0` contains exactly 11,356 nonzero `uint64` entries.
Entry `i - 1` is the threshold for positive ratio index `i` in
`[1, 11356]`; every larger index has threshold zero. The intended real-number
projection is `floor(2^64 * exp(-i / 256))`, but the checked-in integers, not a
runtime floating-point evaluation, are authoritative. Concatenating entries
in index order as `u64be` produces 90,848 bytes whose SHA-256 is
`88a35fea368b5df890aa790239ca681154f69541c3e7dab05cf60dbc3890bfbf`.
The first threshold is `0xff007fd55ffdde38`, index 256 is
`0x5e2d58d8b3bcdf1a`, index 11356 is `1`, and index 11357 is zero.

Computing `ceil(deltaE * 256 / temperature)` uses an exact widened product and
ceiling division with no narrowing overflow. Zero temperature is invalid. A
ratio above 11356 maps directly to threshold zero; it cannot clamp to a
nonzero threshold. A non-positive delta consumes no Acceptance-stream word. A
positive delta consumes exactly one word even when its threshold is zero.

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
ObjectiveVector, TotalOrdering rank, and SearchEnergyRef required by the
search.

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
`TransportRoutingAction` explicitly selects `WholeNet`, `SingleSink`,
`RootedSubtree`, `WitnessRegion`, or `Global` negotiated routing. Full-design
negotiation occurs only in
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
Evidence, or Artifact. Closure is triggered only when a metric dimension in
`SelectedObjectiveClosure.focused_closure_dimensions` has nonzero directed
code.
The dimension uses the central source, bound, and quantization contract;
Mapping owns no private frequency, timing, buffer target, or quality-gate
deviation.

An ephemeral `ClosureRegion` is derived from Evaluation critical paths,
recurrences, bottlenecks, and findings, then expanded through the ordinary
Action dependency closure. If Evaluation cannot localize the cause, the region
is the complete candidate. Proposals are ordered by unresolved required
witness first, optional Evaluation priority descending, and canonical Action
key. Each probe uses `MoveTransaction`; only the strictly best rank-improving
Action commits. Equal rank does not commit, and there is no random acceptance.

Closure stops when every selected metric dimension reaches directed code zero,
no strict improvement exists, the deterministic proposal budget is
exhausted, or a required hot Evaluation binding fails. It then runs full
Mapping and Evaluation checkpoints. Remaining Mapping `V` must enter bounded
exact repair or prevent finalization. Remaining `Q` may enter repair when
SearchPolicy asks for it, but cannot prevent publication of a base-valid
Mapping that passes the exact `K` admission; only post-publication `Promote`
applies formal quality gates. Finalization cannot silently repair either
class.

### Bounded Exact Repair

Loom builds one required in-process C++ OR-Tools `CpSat_1_0` adapter pinned to
OR-Tools v9.15 commit
`551ad10d94835c99e5e1e684500d3db398c0e345`. SearchPolicy decides whether a
candidate has bounded exact-repair work; adapter availability is not a runtime
provider alternative. There is no repair artifact, alternate candidate
authority, solver plugin schema, Python path, or external solver binary.

The adapter solves only a complete bounded dependency region derived from one
canonical unresolved Mapping or Evaluation witness. The ephemeral inputs are
`FrozenModel`, resolved `C`, current `CandidateState`, the closed conflict
region, and the exact Evaluation model and constraint identities.

Region closure includes affected realizations, nets and route branches,
attachments, contexts, tags, buffers, memory and service bindings,
`ResourceUse`, constraint groups, and conflicting occupancy. Outside decisions
are fixed and their claims are subtracted from available capacity. If the
complete closure exceeds `max_region_decisions`, the result is
`RegionTooLarge`; truncation is forbidden.

The repair policy owns exactly two semantic work limits:

```text
max_region_decisions
max_solver_calls
```

One model decision admitted to the complete closed region consumes one region
decision. Every call into CP-SAT consumes one solver call before cache or
result reuse. OR-Tools branch count, conflict count, wall time, and
`deterministic_time` are not Loom semantic work units. Wall-time, memory, and
cancellation controls may interrupt execution, but an interrupted repair
cannot change the original candidate.

The solver assignment is diffed against the candidate and rebuilt as one
canonical ephemeral `ActionBatch` containing only the three existing Action
variants. One `MoveTransaction` applies the batch atomically. Mapping hard
constraints and the exactly representable WeightedLevel selected by
`SelectedObjectiveClosure.selected_search_energy` may enter the solver.
TotalOrdering and
Pareto are not flattened into a solver-private scalar.
Approximate Evaluation information may order exploration but cannot prove
feasibility. When required `Q` is not exactly encodable, Mapping-feasible
assignments are reconstructed in canonical order and checked by the exact
full Evaluation model under its deterministic evaluation budget.

Every solver call uses one worker, the fixed restart-derived seed, integer
decision and objective encodings, and the complete explicit `CpSat_1_0`
parameter record. The adapter does not request an unordered solution pool.
Only `OPTIMAL` and `INFEASIBLE` are proof-bearing statuses. `FEASIBLE` is an
unproven incumbent, `UNKNOWN` is unproven termination, and `MODEL_INVALID` is
an adapter `InternalError`.

One repair invocation consumes exactly one `nextU64()` word from its
restart-local `ExactRepair` stream before its first solver call. Its OR-Tools
`random_seed` is the nonnegative signed integer formed by the low 31 bits of
that word. Every optimization and feasibility call in the same invocation
reuses that seed and consumes no further PRNG words. The complete
`CpSat_1_0` parameter construction starts from the pinned v9.15 protobuf
defaults and applies exactly these overrides:

```text
num_workers = 1
random_seed = low31(exact_repair_stream_word)
search_branching = FIXED_SEARCH
randomize_search = false
cp_model_presolve = true
enumerate_all_solutions = false
use_lns = false
use_lns_only = false
log_search_progress = false
log_to_stdout = false
```

All other fields retain the defaults of the pinned OR-Tools commit. In
particular, the adapter sets no solver wall-time, deterministic-time, branch,
conflict, or incumbent limit. Loom's owner-local solver-call budget and the
outer execution controls remain the only corresponding limits.

Canonical extraction is one fixed protocol:

1. solve the exact selected repair objective, when present, and require
   `OPTIMAL`; a pure feasibility model must likewise return `OPTIMAL`;
2. fix the proven objective value in the model;
3. visit decision variables in canonical typed decision-key order;
4. test each variable's legal values in canonical order using pure feasibility
   calls, fixing the first value that returns `OPTIMAL` and skipping values
   that return `INFEASIBLE`; and
5. require one complete assignment, then rebuild and verify it through the
   ordinary Mapping and exact Evaluation owners.

This yields the lexicographically smallest assignment among exact optimum, or
among exact feasible assignments when no objective is present. A `FEASIBLE`,
`UNKNOWN`, execution interruption, or exhaustion of `max_solver_calls` at any
point returns `UnknownBudgetExhausted` before mutation. The order in which
OR-Tools discovers incumbents is never observable.

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
can be reported as global `ProvenInfeasible`. A local `INFEASIBLE` result is
only `RegionInfeasibleUnderFixedBoundary`. It becomes global
`ProvenInfeasible` only when that region is the complete candidate domain and
all required Mapping and Evaluation constraints are represented exactly.
Approximate `Q` can never support that proof. Every non-repaired outcome leaves
the original candidate unchanged.

## Evaluation Transaction

Every Action uses one online protocol:

```text
S' = ApplyAndClose(S, Action) in shadow state
VG' = exact Mapping incremental evaluation of S'
Q'  = exact resolved EvaluationModel evaluation of S'
vector' = ObjectiveVector(VG', Q')
rank' = TotalOrdering(vector')
energy' = value(SelectedObjectiveClosure.selected_search_energy, vector')
accept or reject
commit or roll back Mapping and Evaluation state atomically
```

The Mapping full oracle is
`fullPnrCost(FrozenModel, CandidateDecisions)`. Each Evaluation model owns its
full execution semantics. An incremental adapter is only an exact execution
optimization for that same model identity. A lower-fidelity predictor is a
different model identity, not an approximate adapter for a higher-fidelity
model.

Each exact `ResolvedPnrEvaluationBindingRef` may create one ephemeral adapter
with
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
It reconstructs the same five Mapping violation facts from exact owner inputs,
requires all five to be zero, and compares no cached Candidate aggregate. A
capacity diagnostic identifies the exact Fabric resource-state owner, capacity
dimension, raw usage, raw capacity, and canonical occupancy witness; buffer,
service, route, and pipeline wording is diagnostic context only. A progress
diagnostic identifies the canonical first proven closed wait set. The verifier
does not recreate the retired registry-1.0 categories.
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

System base verification returns only the closed result algebra owned by
`docs/spec-mapping-verification.md#systemmapping-base-verifier`; PnR does not
redefine its variants or failure classification.

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
* System `H` Presburger and stable-key partitions, overlap/gap and target-in-
  shape rejection, mutually exclusive hierarchical versus flat graph domains,
  exact flat five-input tuple and seed validation, same-target canonical
  merge, and changed-view digest with unchanged possible Mapping identity;
* H 3.0 endpoint-factorized service compatibility, including one immutable
  SpatialMapping reused by two AccCore occurrences whose bound endpoints have
  different capability domains, correct sharing when two contexts bind the
  same endpoint, member-local intersection, exact request-source,
  request-sink, response-source, and response-sink pair-member carrier rows,
  required empty rows, H 2.0 rejection, and no provisional SpatialMapping
  identity or fallback endpoint scan;
* exact channel `source_map` image partitioning, including non-surjective
  inactive terminals, a canonical empty message plan, one static terminal on
  distinct execution owners, same-owner pair collapse, and rebuild after a
  pair appears, disappears, or changes owner;
* complete internal-edge accounting for configured FU, configured
  `fabric.mem`, temporal register-file absorption, and residual logical nets;
* endpoint-only A*, multi-sink route trees, explicit broadcast, checked route
  cost, PathFinder net order and termination, all negotiation kernels, and the
  five closed TransportRouting scopes without arbitrary sink subsets;
* raw capacity legality versus Q-scaled cost-only normalization, including a
  legal set of small claims whose individually rounded costs exceed one
  capacity unit, normalized history update, and shared-prefix `G` accounting;
* route-wide widening acceptance plus rejection of a narrowing bottleneck or
  attempted payload borrowing from tag bits;
* exact memory access-form, element, lane, mask, and use-pattern domain freeze,
  including fail-fast empty domains and rejection of equal-width semantic
  mismatches;
* complete vector address, data, and mask routing plus one declared internal
  multi-transaction memory pattern with no Mapping-invented lane routes;
* atomic Action commit and rollback across placement, routes, resources, and
  exact preflighted Evaluation adapters without candidate copying;
* stable logical slots across cache, retry, replicate, and resume; fixed seed
  attempts; deterministic initializer backtracking and Action selection;
  central `Promote` separation; replay-stable annealing; focused closure; and
  exact-repair taxonomy, proof-bearing statuses, solver-call budget, and
  lexicographically canonical extraction;
* shared objective dimensions, three-valued CNF truth, independent full `V/G`
  and `Q`, TotalOrdering versus SearchEnergy separation, base verification,
  and exact admission;
* `ServicePlan` versus `ResourceUse` ownership, trigger/release and atomic
  activation derivation, progress outcome classification, and hierarchical/flat
  persistent-result equivalence.

Tests must not preserve container layout, printer whitespace, path insertion
order, a greedy or place-then-route baseline, objective weight matrices,
protocol implementation details, cache strategy, or solver internal shape.
