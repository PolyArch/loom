# Place And Route

This specification owns the provider-independent semantic contract for Loom
Spatial and System placement and routing. It defines exact invocation inputs,
Mapping constraints, physical-demand reconstruction, legality, objective
sources, typed outcomes, determinism, and publication. It does not prescribe
one search algorithm, mutable state layout, cache, worker strategy, router, or
solver.

The observable finite-search behavior of the in-tree provider is owned by
[Builtin Mapping PnR Replay Profile](spec-pnr-provider-builtin.md). That profile
may select algorithms and limits, but it cannot redefine Mapping legality or
the independent verifiers defined here. C++ data layout and machine scheduling
are implementation details and are not specification state.

The words **must**, **must not**, **required**, **shall**, **shall not**,
**should**, **should not**, **recommended**, **may**, and **optional** are
normative.

## Mapping Invocation Diagnostics

Every Mapping generator must expose a runtime-configurable diagnostic stream
with at least invocation, freeze, candidate, routing, repair, verification, and
publication events. Normal execution is quiet. Diagnostics must identify the
exact invocation owner, typed termination reason, consumed semantic work, and
the first stable failing subject when one exists.

Diagnostics are observation only. Their level, destination, formatting,
timestamps, host data, wall time, and message text do not enter Artifact
identity, candidate ordering, work accounting, or replay. A diagnostic may
summarize a proof witness, but it cannot replace the typed outcome or the
independent verifier. Diagnostic text is never serialized as the durable
infeasibility witness.

## Semantic Owners

PnR combines facts owned elsewhere without copying their authority:

| Fact | Semantic owner |
|---|---|
| Program behavior, events, and logical nets | Canonical Dataflow |
| Realization selection and internal edges | TechMapping |
| Hardware capability, topology, resources, timing contracts, and progress boundaries | Fabric |
| Selected placement, routes, tags, resources, and System bindings | SpatialMapping or SystemMapping |
| Caller restrictions on a Mapping invocation | MappingConstraintSet |
| Search policy and selected objective algebra | ResolvedConfig component views |
| Candidate-specific physical demand | Rebuildable Mapping projection defined here |
| Search order and finite-prefix replay | Selected provider profile |
| Final structural legality | Independent Mapping verifier |
| Runtime or physical observations | EvaluationEvidence |

PnR must not make endpoint names, coordinates, ownership, co-location, cached
indexes, or a search heuristic into a competing source of physical meaning.
When two consumers need the same derived physical fact, they must call the same
typed projection or validate a cache against it.

## Derived Contexts And Bounded Reuse

PnR separates immutable target-derived state from workload-derived demand and
mutable search state. These layers are rebuildable implementation views, not
Artifacts and not additional semantic owners:

```text
Fabric identity
  -> FabricStaticContext
       canonical Fabric view and shared dependency views
       resource and capacity supply inventory
       endpoint and traversal topology
       tag-continuity index
       compiled handshake owner models

FabricStaticContext + PhysicalTimingProfile digest
  -> FabricTimingContext
       validated traversal timing lookup
       timing-annotated routing index

Dataflow + TechMapping + constraints + selected config
  -> SpatialActiveProblem
       realizations and compatible placements
       memory, transfer, port, route, and handshake demand
       progress and recurrence projections

System identity
  -> SystemStaticContext
       execution-core and target-class inventory
       endpoint and traversal topology
       static carrier and use-pattern domains

SystemStaticContext + SpatialMapping set + Dataflow + constraints + timing
  -> SystemActiveProblem
       execution and graph decisions
       workload service terminals and legs
       memory and SpatialMapping bindings

active problem + exact selections
  -> mutable candidate and transaction state
```

A context key contains every exact Artifact identity or component-view digest
on which its value depends, plus the owner-defined schema and algorithm
versions for the derived representation. A path, application label, object
address, timestamp, allocation order, or invocation ordinal is never a key.
Changing any dependency produces a miss; a hit performs the owner's cheap
exact-key and structural revalidation before use.

Context reuse is bounded by an explicit invocation or execution-session owner.
Values are shared immutable handles and are destroyed with that owner. A
process-global unbounded map is forbidden. Mutable candidates, transactions,
scratch arenas, PRNG state, work budgets, solver state, and interruption state
are never cached across candidate attempts. An exact whole-problem cache may
provide a same-tuple fast path, but it cannot replace the coarser static and
timing contexts needed by sibling TechMapping or SpatialMapping candidates.

The provider records, for each context kind, hit and miss counts, construction
wall time, retained bytes, and deterministic construction work. It also records
the full and active endpoint, traversal, handshake, placement, and relation
counts needed to expose accidental full-target work. These observations are
diagnostic infrastructure only: they do not affect identity, ordering, work
budgets, or replay.

Diagnostic analysis and emission are separate operations. A report consumed
only by diagnostics, including Fabric topology quality, is not computed while
its required diagnostic level is disabled. If enabled at several Mapping
stages, one Fabric-identity analysis may be emitted several times without being
recomputed. A DSE objective that consumes the same projection obtains it from
its typed hardware-evaluation owner instead of relying on Mapping diagnostics.

Reuse never weakens closure. Cache hits do not bypass input admission, exact
repair, final global closure, Artifact finalization, or the cold independent
Mapping verifier. Cache misses and hits must publish identical canonical
results and deterministic work accounting.

A root-complete Spatial invocation may import several TechMappings that share
one Canonical Dataflow. It imports that Dataflow at most once per complete root
reference, including schema identity, schema version, and artifact identity.
Every reuse revalidates the cached Artifact and program-view identities against
the exact TechMapping lineage. The cache is destroyed with the provider
invocation and reports requests, hits, misses, construction time, retained
bytes, and deterministic import work.

`SystemActiveProblem` imports the canonical SpatialMapping set exactly once and
imports each lineage TechMapping at most once per complete root reference.
Each SpatialMapping still performs exact Dataflow, Module, and TechMapping
lineage validation on reuse. The TechMapping import cache exists only while the
active context is constructed; its requests, hits, and misses are included in
the active-context statistics and deterministic construction work.

Within one `SpatialActiveProblem` construction, compute attachment projection
is factored by the exact physical tuple `(FU occurrence, parent PE, schedule,
FU template port, logical terminal direction, payload width)`. The Fabric and
routing contexts are fixed inputs of that table. A miss derives the canonical
endpoint, local traversal, durable progress boundary, and shared Temporal
operand-enqueue unit once; a hit reuses that immutable list while each logical
PortDemand still receives its own placement domain and option ownership. The
table is destroyed after construction and never contains a selection. Active
context diagnostics report its lookup, hit, and miss counts in addition to the
fully materialized logical option count.

The handshake part of `SpatialActiveProblem` is an exact demand relation over
the shared Fabric handshake context. It retains a compact owner-model and local
fragment reference for every placement, memory plan, traversal witness, and
switch activation admitted by the active realization and routing domains. It
must not flatten the union of those potential fragments into candidate nodes,
arcs, adjacency tables, or topological state.

The mutable candidate materializes a graph only from its currently selected
fragments. Boundary signals with equal typed Fabric identity are unified;
owner-local junctions remain private. A transaction derives physical arc
insertions and removals only from fragment refcounts that cross zero. Deletions
cannot introduce a cycle. Insertions are checked against the committed graph
plus the exact transaction overlay by affected reachability, without rebuilding
the candidate graph, canonical order, or a full-Fabric topology. Rollback drops
the overlay. The closure records whether applying the exact incremental rank
repair is estimated to exceed the deterministic work of rebuilding the
selected graph;
that choice is acted on only after the proposal is accepted. A rejected probe
therefore never pays for an optional whole-graph witness or rebuild. Commit
uses the cheaper exact rank-repair or selected-graph reconstruction path; a
refcount-only commit whose active fragment set did not change reuses the
existing graph. Final independent verification remains identical for both
paths. This graph is mutable candidate state and is never placed in an
invocation, session, static, or global cache.
Candidate selections, refcounts, scratch storage, PRNG state, and budgets also
remain candidate-owned.

An endpoint router may retain exact reverse lower-bound distance tables inside
one prepared search scratch. Its key covers the router algorithm version,
lower-bound cost revision, endpoint targets, traversal eligibility, payload and
tag requirements, and every timing input consumed by the reverse search. The
stored distances must losslessly reconstruct the complete `RouteCost`. A
compact common-scale table may use explicit wide-value exceptions, but
narrowing or saturation is forbidden because a cache hit must preserve the
cold path, cost, tie break, and deterministic forward expansion work. Storage
is allocated only on a miss and is bounded by both bytes and entry count. The
search diagnostics report hits, builds, evictions, populated entries, and
retained bytes. This cache never survives the search scratch and is not part of
a static or active context.

This construction rule does not prune the legal candidate domain. Every
placement and plan admitted by the exact realization domain remains
selectable, and first use must materialize the same fragments that eager
expansion would have produced. Exact repair may request any admitted entry.
Final Mapping verification reconstructs the selected handshake relation from
the complete Fabric models and exact Mapping decisions without trusting the
candidate's materialized graph. Diagnostics report potential fragment and
contribution counts separately from materialized fragment, node, arc, and
contribution counts. They also report uncached materialization time, retained
bytes, deterministic work, transaction closure count, inserted and removed
arcs, affected-node visits, and affected rank span.

## Invocation Contracts

### TechMapping Hardware Demand Feedback

Tech cover search applies the same exact all-different compute-context relation
to partial and complete covers. When that relation has no complete matching,
the canonical alternating-tree Hall subset is the Mapping-owned demand
witness. Search retains at most one deterministic maximally actionable
observation: larger positive gap, then larger Hall demand set, then canonical
payload order.

The transient feedback groups Hall demands by typed FU capability and demand
multiplicity. It does not serialize compatible context lists. Adoption against
the exact Fabric re-runs `deriveSpatialComputeContextPlacementDomain`, forms
the context union, and requires a positive Hall gap. Therefore generator,
feedback consumer, diagnostics, and any hardware-candidate response cannot
maintain independent capability or resident-context interpretations.

An observed partial-cover Hall deficit is exact for that relation but is not a
global infeasibility proof. If bounds prevent exhausting alternate covers, the
TechMapping outcome remains `ProofNotEstablished`. Summary diagnostics report
both `proof_scope=observed_cover_relation` and the typed counts. The payload is
invocation-local and cannot replace final Mapping verification or be restored
by rerunning search during terminal replay.

Memory-row construction charges every prospective row that reaches a physical
compatibility decision, including rows rejected by occurrence-global resident
capacity, operation-port exclusivity, boundary correspondence, or Temporal
external-ingress uniqueness. Spatial operation-port conflicts and Temporal
resident-count overflow are pruned while forming the selection domain. A
memory family retains a deterministic row frontier bounded by the resolved
row, evaluation, and publication limits. Reaching that frontier makes the
domain non-exhaustive; it never becomes an infeasibility proof.

When the rectangular actor-by-row search surface exceeds the stable constructive
search switch threshold of 4096, Tech cover search uses a bounded constructive
exact-cover frontier. This algorithm-selection threshold is independent of the
resolved expansion budget so that changing a budget does not silently change
the search family or canonical candidate order; the resolved budget still
limits the frontier's expansions. The frontier selects the uncovered actor
with the smallest current row domain, prefers rows that cover more actors,
prefers Temporal memory supply and wider physical occurrence domains, and
validates every completed cover with the same exact compute-context and
memory-occurrence relations. An empty or budget-limited frontier is
`ProofNotEstablished`; exhaustive best-first search is not entered after the
constructive bound has taken ownership of the search. Final TechMapping and
downstream Mapping verifiers remain independent and authoritative. Summary
diagnostics expose memory-row frontier limits, constructive-search invocations,
completed-cover checks, publications, and the existing deterministic row and
cover work counters.

### Spatial PnR

One Spatial invocation has the exact tuple:

```text
SpatialPnr(D, T, F, C, K)
```

`D` is one Canonical Dataflow Program, `T` one independently verified
TechMapping over `D`, `F` one finalized Fabric Artifact, `C` one adopted
Spatial PnR config component view, and `K` one finalized Spatial
MappingConstraintSet over the exact `D/T/F` tuple. All upstream identities must
match exactly. The empty constraint set is a real Artifact; absence is invalid.

The current in-tree Spatial config descriptor is
`loom.spatial_pnr.config.15.1`. A config digest from another domain or version
cannot be adopted. The config is invocation input and does not enter the
semantic identity of a published SpatialMapping.

Each published result must be a finalized SpatialMapping that independently
passes base verification and exact `K` admission. Mutable candidates, solver
assignments, route-price history, no-goods, and proof caches are never outputs.
Every terminal outcome, including success, reports deterministic invocation
work for seed preparation, placement assignment, endpoint expansion,
negotiation, annealing, exact repair, final closure, independent verification,
Artifact finalization, and publication. A zero exact-repair count means the
candidate closed without invoking repair; it never means repair or final
verification was disabled.

### System PnR

One System invocation has the exact tuple:

```text
SystemPnr(D, F, R, H, C, K)
```

`R` is the canonical nonempty root-thread-launch set. `H` is the immutable
System search-domain component view derived for exact `D/F/R/K`. `C` is one
adopted System PnR config view. `K` is one finalized System
MappingConstraintSet. Every selected SpatialMapping in `H` must independently
verify against the same `D` and an exact SpatialCore occurrence in `F`.

The current descriptors are:

```text
loom.system_pnr_search_domain.4.0
loom.system_pnr.config.8.0
```

The System config may carry a canonical root-keyed binding-partition intent.
Each entry names one exact `RootThreadLaunchRef` and a positive partition
count. The intent changes only the Presburger search granularity used to build
`H`; it neither restricts the legal logical domain nor selects an AccCore.
Absent roots retain one logical partition; this conservative fallback does not
speculate concurrent execution. A foreign,
repeated, zero-count, or non-canonical entry is invalid. Because the intent is
part of the adopted config bytes and digest, journal replay and hardware
reopen cannot silently change it. Only a finalized SystemMapping proves which
partitions were assigned to which physical resources.

System search is hierarchical. A rooted graph selects one immutable compatible
SpatialMapping and its exact SpatialCore occurrence. There is no Flat graph
domain, no implicit reopening of Spatial decisions, and no compatibility codec
for the removed shell. Cross-layer exploration must request another upstream
candidate or apply invocation-local feedback; it must not mutate a published
SpatialMapping inside System PnR.

For `FirstVerifiedCandidate`, System PnR performs an exact imported-capacity
preflight before System service routing. It enumerates the existing
thread/graph initializer relation under the configured assignment bound and
projects only the immutable ResourceUses and routes imported by each selected
SpatialMapping into occurrence-qualified AccCore capacity namespaces. A
capacity-closed assignment permits normal initialization; the initializer then
uses the same projection to reject over-capacity assignments before routing and
continues until one complete routed candidate is found or its ordinary bound
is exhausted. The first capacity-closed assignment is not privileged and is
not assumed routable.

If every assignment in the complete bounded relation has imported capacity
pressure, the outcome is `ProofNotEstablished` with one exact occurrence
witness. Reaching the preflight's per-seed assignment bound without exhausting
the relation produces no hardware demand and cannot suppress the remaining
configured initializer seeds; ordinary capacity-closed initialization
continues, and only its final bounded outcome may be `SemanticLimitReached`.
Neither outcome proves global Mapping infeasibility. If the exact initializer
relation has no assignment independently of imported capacity, its existing
structural proof remains `ProvenInfeasible`. System service routes, targets,
ResourceUses, progress, recurrence, and final legality remain outside the
preflight and are constructed and independently verified by the ordinary
System candidate path.

Complete exhaustion of the imported-capacity relation may persist one
`loom.mapping.system_execution_binding_checkpoint` 2.0. That checkpoint binds
the exact Dataflow, parent System, MappingConstraintSet, resolved PnR
configuration digest, and search-domain digest. It stores canonical thread
and graph Presburger cells with their selected AccCore and SpatialMapping
targets, plus a typed imported-capacity witness and its dependency roots. It
is not a SystemMapping and owns no service target, route, ResourceUse, progress,
recurrence, or legality claim.

One exact parent-to-child AccCore correspondence may combine that checkpoint
with a child System into a
`loom.pnr.system_mapping_checkpoint_migration_seed` 5.0. The seed also names
the exact parent AccCore occurrence whose capacity witness caused the child.
The correspondence must come from typed hardware lineage and must cover every
preserved or reopened parent AccCore; Module equality and occurrence ordinal
are not substitutes. System PnR releases precisely the checkpoint thread cells
bound to the witness occurrence, validates every remaining thread and graph
choice against the frozen child domain, and visits that impact-cone initializer
before the ordinary fresh seed family.

A finalized parent SystemMapping may instead be projected into one
`loom.pnr.system_mapping_finalized_migration_seed` 5.0. That seed binds the
exact parent Mapping, child constraints, child SpatialMapping frontier,
resolved PnR configuration, and canonical parent-to-child entity, transfer,
Module, and AccCore correspondences. Unlike the incomplete checkpoint, it may
preserve service targets and routes whose complete referenced hardware lineage
survives in the child. The child importer validates the complete correspondence
and rebases every retained Mapping reference before this invocation-local
preference reaches PnR.

The same seed owns schedule-preserving repair on an unchanged System. In that
case the exact equal System identity mechanically supplies the identity
correspondence, and the seed additionally carries a canonical set of Dataflow
root invalidation roots. System PnR releases only thread and graph decisions
owned by those roots, preserves the remaining execution decisions, and reopens
the dependent System service/route cone. The initial route projection is
conservative and reopens every System service leg when a schedule root changes;
it must not claim finer locality until the service dependency projection proves
it. This is still incremental repair rather than legality reuse: the resulting
candidate passes the ordinary capacity, routing, progress, finalization, and
independent import gates.

Migration is preference, never constraint or proof. Missing, ambiguous,
unmatched, or empty impact cones produce one typed fallback and fresh search
continues. Released choices are solved through the same hard relation and
complete-candidate closure callbacks as cold initialization; they do not bypass
capacity, routing, progress, or legality checks. The current checkpoint
deliberately carries no service selection. Its child therefore reopens service
legs, while a finalized parent seed or a typed Module rebase may preserve
service identity when the hardware correspondence proves that the target and
transport are unchanged. Work accounting separates preserved, rebased,
invalidated, repaired, and reopened thread bindings, graph bindings, service
legs, and route resources. Every migrated candidate still passes global closure,
the cold Mapping verifier, and MappingConstraintSet admission before
publication. A missing or ambiguous correspondence produces one typed
cold-fallback record; it cannot be treated as a preserved seed.

The resource-time transition owner is separate from the migration seed. A
seed may prefer and repair a child SystemMapping, but only
`finalizeResourceTimeTransition` can publish a compiler-preverified edge. It
strictly imports both SystemMapping and Deployment endpoints, derives resource,
complete Deployment configuration, and route delta digests from their canonical
owners, and then replays the independent closure verifier. Resource and route
digests bind the complete SystemMapping plus every imported SpatialMapping;
the configuration digest additionally binds exact executable,
hardware-binding, configuration-image, static-memory, dispatch, launch, and
admission semantics. Reprogramming cost separately compares the physical
hardware-binding and configuration-payload state. An authored digest, authored
cost, or successful child PnR is not transition proof.

`ResourceTimeTransitionGraph` is the finite compiler-owned catalog. It names
one exact Mapping/Deployment entry, unique endpoints, and preverified edges;
`verifyResourceTimeTransitionGraph` independently imports every endpoint,
requires one canonical root-launch scope, replays every edge closure, and
rejects foreign states or an edge whose `completed_before` frontier cannot be
reached monotonically from the entry. A completion without an edge may advance
the frontier while staying at the current endpoint; it cannot erase a prior
completion. Runtime may select only a graph member and cannot synthesize
Mapping or invoke PnR.

The admitted completion profile requires `completed_before` plus the one
active completing root to be a unique subset of the Canonical Dataflow root
inventory. The completing root is absent after the edge; roots not yet started
may begin under the child Mapping. No logical memory, channel-typed state, or
DynamicWork may persist across the edge. Hardware-programming state is
unchanged and the owner derives exact zero reprogramming and migration time.
Ordinary completed thread and graph computation is allowed. Explicit safe
points, changed hardware programming, surviving in-flight work, and composite
or token boundaries fail closed until typed proof owners are available. These
restrictions prevent the graph from becoming an online PnR or arbitrary
in-flight preemption mechanism.

Hardware-impact reuse reports one closed disposition: `preserved`,
`local_repair`, or `cold_fallback`. `Unchanged` and `Rebase` Tech/Spatial
layers may retain exact Mapping frontiers after child reimport. A `Reopen`
layer invalidates Mapping roots owned by the affected Module while unaffected
Module frontiers remain available to the ordinary generator. A global impact
or an empty typed root set is `cold_fallback`; it is never silently treated as
local repair. System-only transport, service, or attachment changes preserve
independently verified Tech/Spatial frontiers and reopen System PnR. These
dispositions describe work reuse only and cannot admit the child Mapping.

Each published result must be a finalized SystemMapping that closes thread and
graph binding, SpatialMapping selection, service realization, System transport,
resource capacity, and progress for the exact root-launch closure. System RTL
or whole-System physical implementation is not implied by SystemMapping.

### Input Admission

Before mutable state is allocated, an invocation must reject:

* malformed or noncanonical component-view bytes;
* digest or schema mismatch;
* foreign Dataflow, TechMapping, Fabric, constraint, or SpatialMapping roots;
* an empty required root domain;
* a selected objective source without a complete Mapping owner;
* a routing, repair, or search capability the selected provider does not
  implement; and
* arithmetic domains that cannot be represented without saturation.

An invalid owner tuple is `Invalid`. A well-formed capability selection absent
from the provider is `Unsupported` where that outcome exists, or a fail-closed
config-adoption error. Neither is evidence of Mapping infeasibility.

## Generation Outcomes And Candidate Completeness

The semantic outcome classes are:

```text
Generated {
  independently verified candidates
  provider termination: FixedWorkCompleted | SemanticLimitReached
  exact work accounting
}

ProvenInfeasible {
  exhaustive proof over the complete admitted domain
  typed internal contradiction kind
  exact work accounting
}

Incomplete {
  independently verified retained candidates, if any
  ProofNotEstablished | NoPreparedSeed | SemanticLimitReached
  exact work accounting
}

Interrupted {
  independently verified retained candidates, if any
  owner stage, canonical frontier, best selected rank when available
  closure residual and process resource observations
  exact work accounting
}

Unsupported
Invalid
InternalError
```

Provider APIs may use domain-specific typed variants, but they must preserve
these distinctions. `Generated` says every returned member is valid; it does
not claim the mathematical candidate domain was exhausted. The termination
field says whether the provider completed its configured finite work or hit a
semantic work limit.

`Interrupted` is an execution outcome, not a semantic search result. Tech,
Spatial, and System owners observe cancellation only at their atomic work
boundaries and return a typed snapshot without changing formal work order.
The DSE adapter maps this outcome to `CancelledOrTimeout` while retaining every
already finalized candidate. A wall-clock limit can never become
`SemanticLimitReached`, `ProvenInfeasible`, or a zero-valued closure residual.

`ProvenInfeasible` is permitted only when a sound, exhaustive proof covers the
complete admitted invocation domain and every required constraint. A failed
seed, unreachable route under fixed terminals, local solver `INFEASIBLE`,
unsupported repair encoding, timeout, budget exhaustion, cyclic progress
basis, or finite prefix is never enough.

Spatial PnR owns these closed infeasibility-proof kinds:

```text
SpatialPnrInfeasibilityProofKind =
    FrozenDerivedContext       // tag 0
  | FrozenActiveProblem        // tag 1
  | InitializerRelation        // tag 2
  | GraphBoundaryEndpointHall  // tag 3
```

System PnR owns these closed infeasibility-proof kinds:

```text
SystemPnrInfeasibilityProofKind =
    FrozenStaticContext        // tag 0
  | FrozenActiveProblem        // tag 1
  | ImportedCapacityRelation   // tag 2
  | InitializerRelation        // tag 3
```

The kind preserves which internal relation reported the contradiction, but a
kind and diagnostic do not constitute a durable proof. A DSE adapter may admit
an internal kind as `ProvenInfeasible` only when its reason-specific witness is
independently checked against the complete exact input closure. Otherwise it
maps the result to `Incomplete(ProofNotEstablished)`. A provider may add a new
durable kind only with an owner validator that proves its complete admitted
scope rather than merely naming the search site that failed.

These PnR kinds classify the internal relation that reported a contradiction;
they are not persistent Candidate Generator proofs. No current Spatial or
System PnR descriptor registers an owner infeasibility-proof contract because
none carries a reason-specific witness that can be independently reconstructed
and checked from the exact invocation inputs. Every internal Spatial or System
`ProvenInfeasible` result therefore maps to
`Incomplete(ProofNotEstablished)` at the DSE boundary.

The graph-boundary Hall deficit is transient hardware-reopen feedback, not a
durable infeasibility proof. The current Spatial relation model does not create
a global graph-boundary all-different relation because legal temporal or
causally separated endpoint reuse must remain admissible. Counts or diagnostic
text cannot substitute for that missing constraint and witness. An empty
root-complete input frontier remains ordinary completed empty output; timeout,
semantic limits, and unverified contradictions remain typed incomplete
outcomes.

`Incomplete` is not contagious across independent candidates. A generator
adapter must retain every already finalized candidate, continue other
independent parent candidates, and expose aggregate domain incompleteness
separately from the retained set. Downstream Generate nodes may consume those
retained candidates. `Invalid` or `InternalError` invalidates the complete
generator invocation and contributes no selected output binding.

## MappingConstraintSet Artifact Family

`loom.mapping_constraints 1.0` is the single schema family for Spatial and
System invocation constraints. Its MLIR roots and canonical codecs own one
finite canonical clause sequence and exact upstream bindings. Constraint
content does not enter Mapping semantic identity; the invocation manifest
binds the exact constraint Artifact and admission result.

The closed clause kinds are:

```text
DomainRestriction(projection, subject, admissible_domain)
Equal(projection, subjects)
Disjoint(projection, subjects)
```

Subjects and values are typed closed unions. Strings, opaque property bags,
untyped integer ordinals, and provider callbacks are forbidden. A domain is a
canonical set. Clause order is canonical. Duplicate, contradictory, empty
where prohibited, foreign, or ill-typed carriers are rejected during
finalization.

Constraint admission is separate from base Mapping verification:

```text
AdmitSpatial(D, T, F, K, S)
AdmitSystem(D, F, K, M)
```

Admission rebuilds every projection from the sealed Mapping and exact owners.
It must not consume CandidateState, solver variables, search caches, or a
generator-provided truth value. Rejection by `K` is invocation-specific and
does not make the immutable Mapping intrinsically invalid.

### Spatial Projection Catalog

The closed Spatial projection kinds are:

```text
compute_placement
compute_parent_pe
compute_instruction_context
compute_fu_context
memory_placement
net_assigned_tag_values
net_selected_physical_traversals
net_traversal_resource_states
spatial_transfer_attachment
memory_operation_port
memory_bound_services
memory_address_region
```

Subjects are one of a Tech compute realization, Tech memory realization,
canonical producer endpoint, exact transfer terminal, actor, or logical-memory
root. Values are the corresponding typed Fabric occurrences, contexts,
traversals, resource states, transport endpoints, memory operation ports,
services, unsigned intervals, or service-relative address regions.

### System Projection Catalog

The closed System projection kinds are:

```text
thread_target_acc_core
graph_selected_spatial_mapping
graph_target_spatial_core
service_target_region
transfer_terminal_attachment
transfer_selected_traversals
transfer_resource_states
transfer_assigned_tag_values
```

Subjects are one of a root thread launch, rooted graph launch, service
obligation, System transfer terminal, or canonical service leg. Values are the
corresponding typed AccCore occurrence, SpatialMapping reference, SpatialCore
occurrence, memory service region, transport endpoint, traversal, resource
state, or unsigned interval.

Adding a projection or carrier is a breaking schema change. A provider-private
constraint kind is forbidden.

## Spatial Physical Demand Projections

Spatial physical-demand projections are a closed family of identity-free,
rebuildable functions over canonical Dataflow, the selected TechMapping,
Fabric, and exact selected Spatial choices. They are not one aggregate object,
are not persistent Artifacts, and own no new choice. Each function composes
facts at one semantic joint while leaving the primitive mechanism with Fabric,
the realization correspondence with TechMapping, and the selected occurrence
choice with SpatialMapping.

The family derives:

```text
logical residual nets
compute context demand
Temporal Memory occurrence-global rows
Temporal Memory ingress keys and role demand
PE operand-queue atomic match groups
eligible and selected RegFIFO local transfers
residual external route trees
typed durable progress boundaries
Temporal switch route signatures and packed rows
exact resource, tag, timing, and handshake demand
```

PnR search and strict Mapping verification call these functions directly or
validate a cache against their exact output. Configured-hardware projection
uses the same functions to derive Fabric-owned semantic field values. A
simulator imports an exact execution plan from the verified Mapping and that
configured projection. RTL lowering consumes the configured Fabric mechanism
and configuration fields; it does not depend on Mapping or call a Spatial
projection. No layer may regroup by a broad owner, endpoint, input, tag, or
route-tree shape and invent another interpretation.

Distinct graph-boundary tokens cannot select the same untagged Spatial-switch
endpoint. The frozen attachment relation derives endpoint exclusivity only
when the incident switch traversal retains a Mapping-resident Fabric claim;
Temporal-switch traversals expose runtime-service occupancy and remain
shareable through their tag and packed-row projection. This is a physical
admission relation, not a generic port-dispersion heuristic. Its Hall witness
reports the exact boundary-token and endpoint cardinalities before routing,
while final RouteTree and capacity verification remain independent.

### Edge Disposition

Every canonical software edge has exactly one disposition:

1. a Tech realization-internal edge is consumed by `T`;
2. an eligible selected same-Temporal-PE RegFIFO transfer is local; or
3. the edge belongs to one residual external logical net and one RouteTree.

The initial RegFIFO domain admits only one producer, one consumer, one exact
Temporal PE, compatible data width, tag and ordering, and an exact write
traversal, FIFO, and read traversal. The local disposition owns those
traversals and their resource uses. Multicast and mixed local/external fanout
remain one external net. Locality is a soft preference; lack of a local FIFO
must leave the external route alternative available.

Co-location, a register-file name, or common ownership is not a local transfer.
No edge may be absent from both the local and external projections or appear in
both.

### Temporal PE Ingress

Fabric exposes each logical operand queue as a typed durable progress boundary.
One queue may serve several FU-local SSA consumers through the FU's explicit
broadcast semantics; those consumers share one queue entry and one common
dequeue. For one exact physical input and Physical Tag, all distinct matching
operand queues form one atomic ingress group:

```text
ready = any_match && all(!match[i] || queue_ready[i])
fire  = valid && ready
enqueue[i] = fire && match[i]
```

No matching queue may consume independently. Mapping progress and configured-
hardware projection rebuild the exact group; simulator plan import preserves
it; and the Fabric RTL implements the corresponding queue admission contract.
Each queue-ready term observes only capacity free at the start of the PE clock
cycle, so neither valid nor ready propagates combinationally through the
ingress traversal. A PE selector or endpoint alone is not durable storage.

Before placement search, the physical-demand projection groups external actor
operands in each Tech Compute Realization by their concrete FU boundary input.
Producer-identity verification proves that consumers in one group belong to
the same logical net. Their attachment decisions form a structural equality
relation over the exact selector disposition, while the RouteTree retains
every logical sink obligation. Strict SpatialMapping import rebuilds one
physical logical-queue match with the complete canonical consumer set.

The Dataflow-owned actor handshake cases also derive each canonical set of
external input roles that one firing may consume together. Tech boundary
correspondence removes realization-internal roles, and producer identity
collapses repeated roles of one logical net into one atomic-fanout member.
Distinct logical producers remain independent ordered members. Frozen Spatial
search records both group-to-demand and demand-to-group incidence, so one
attachment move updates only its affected shared-ingress pressure. This
pressure is a central objective measure; it is neither a hard relation nor a
deadlock proof.

The frozen attachment domain records the Fabric-derived operand allocation
unit only when every resident-context choice for that concrete FU input names
the same unit. Each FU-boundary broadcast class contributes one representative
to the structural disjoint relation for distinct queues that select the same
ingress endpoint and shared unit. Other consumers in the class are tied to the
representative by selector equality and do not consume another enqueue service
or entry. A context-dedicated unit needs no extra relation because distinct
context choices already select distinct units. The strict physical-demand
projection reconstructs the exact `(ingress, Physical Tag)` match group and
validates its unique queue set with the Fabric operand-buffer contract.

### Temporal Memory

Temporal Memory row capacity `K` belongs to the complete memory occurrence,
not to each operation port. Row ordinals are assigned in canonical realization
and actor order with one occurrence-global cursor. Operation port is row state,
not a second row owner.

An external ingress key is the exact Fabric ingress endpoint plus canonical
Dataflow producer. A token must match exactly one row and role. Unlike PE
operand ingress, Temporal Memory input matching never fans one token into
several rows. The generator, materializer, strict verifier, and configured-
hardware projection call the same ingress and role-demand derivation. The
simulator imports its exact result, while Fabric RTL implements the configured
row mechanism without a Mapping dependency.

### Temporal Switch Row Packing

Each selected segment through one Temporal switch has a route signature:

```text
(input, exact output set)
```

Physical Tag is part of row identity. All selected demands with the same
`(switch occurrence, Physical Tag)` must occupy one resident row; demands with
distinct tags occupy distinct rows even when their signatures are compatible.
Same-tag signatures are legal exactly when:

* signatures with the same input have identical output sets; and
* signatures with different inputs have pairwise disjoint output sets.

The resident row is the union of only those proven-compatible signatures. Any
same-tag collision that creates an unselected crosspoint or broadcast is
illegal. The Mapping row projection applies the Fabric-owned compatibility
predicate, groups by occurrence and numeric tag, and orders rows by unsigned
tag. Capacity and marginal route cost count newly required resident rows, not
continuity segments.

While a candidate still contains `TagUnassigned`, the Fabric owner preserves
all assigned rows and places each unassigned demand into the first compatible
assigned or provisional row. This provisional projection is a search-only
lower bound with no configuration identity. Final PnR verification,
configuration, handshake, and execution require the exact tagged projection.
No consumer may infer or cache an independent row membership.

An incompatible same-tag row remains one physical row and one explicit
`TagConflict` during repair. PnR derives its actual combined handshake demand
instead of erasing it. Strict Mapping verification rejects that row before
configuration or execution.

A search candidate may temporarily require more packed rows than the Fabric
owns. The complete row projection remains the owner of
`TagResidentCapacityOveruse` and route cost. Handshake projection instantiates
only the occurrence-local resident-row prefix because an overflow row has no
physical activation. This does not make the candidate publishable: after row
pressure returns within capacity, every surviving row is reconstructed and
must pass the full row-aware handshake check. Strict Mapping verification and
configuration reject any remaining overflow independently.

The atomic execution key is `(resident row, input)`. A Fabric requester group
may arbitrate a physical owner, but it cannot merge different configured rows
or logical activations.

### Durable Progress Boundaries

A durable progress boundary is a typed Fabric fact selected at an exact sink.
Current kinds are a buffered FIFO traversal and a Temporal PE operand queue.
A bypass FIFO, combinational switch, endpoint, selector, Temporal Memory row,
ownership relation, or route prefix is not durable. A new boundary kind is
valid only after Fabric exposes physical storage and its enqueue/dequeue
progress contract and Mapping reconstructs it from a selected sink.

For an atomic multicast whose prerequisite sink can causally reach a dependent
sink, the dependent branch must encounter a durable boundary after divergence
from the prerequisite branch. A boundary on the shared prefix is insufficient.
A dependent Temporal PE queue at a shared external endpoint is sufficient
because atomic ingress commits all matching queues together and each queue
holds its token. The router must not fabricate a detour back to an endpoint
already in the RouteTree.

Canonical Dataflow may also contain initialized feedback. Dataflow derives
the only recognized feedback-input inventory from registered actor transition
semantics. Mapping removes all such edges and accepts the logical basis only
when the remaining actor dependency graph is acyclic. It then retains only
feedback edges whose consumer reaches their producer in that DAG. Any other
cycle remains `ProofNotEstablished`.

Every retained initialized feedback edge is a physical progress obligation.
An external disposition must reach a buffered FIFO traversal or the exact
Temporal PE operand queue at its sink. A selected same-PE RegFIFO disposition
satisfies the obligation through its explicit write, finite FIFO, and read
contract. A compute-internal edge without a declared durable implementation,
a bypass path, co-location, or an unselected local alternative does not.
Search and strict verification derive this inventory from the same Dataflow,
TechMapping, Fabric, and selected edge dispositions.

Reconvergent capacity is proved per physical FIFO occurrence, never per tag.
One producer binding owns at most one active transfer: its next firing cannot
complete until every sink of the current transfer has reached a durable
acceptance point. Consequently each distinct selected logical net contributes
at most one resident token to a FIFO shared pool, including a distance-one
initialized-feedback token. The canonical owner-local capacity obligation is
the number of distinct selected logical nets, while StrictFifo or tag-local VC
classes independently determine dequeue order. A selected pool below that
bound is `ProvenClosedWaitSet(reconvergent_capacity_shortfall)` because the
proof cannot remove downstream-capacity waits. A sufficient pool removes those
capacity edges from the closed-wait graph; it does not excuse a remaining
global-HOL or same-tag order cycle.

The proof remains `ProofNotEstablished` when a VC tag value is unavailable,
one producer-to-consumer channel re-enters the same physical FIFO occurrence,
or initialized-feedback removal does not leave an actor DAG. Multiple known
actual tag values and initialized feedback are not by themselves proof debt.
The incremental Spatial state caches only selected-net incidence, owner route
anchors, queue-class values, and the resulting debt or shortfall; final
materialization reconstructs the obligation from Mapping and Fabric.

## Spatial Legality

A Spatial candidate is base-legal only when all of the following hold:

* every realization and attachment belongs to the exact `D/T/F` closure;
* every compute and memory decision selects a compatible exact Fabric domain;
* every residual external net has one connected acyclic RouteTree with its
  exact source and complete sink set;
* every local net selects one exact valid RegFIFO option;
* every selected traversal, resource state, capacity claim, event interval,
  and Physical Tag is owned and compatible;
* Temporal switch signatures pack without extra crosspoints or broadcasts;
* Temporal Memory rows and ingress keys satisfy occurrence-global capacity and
  uniqueness;
* every atomic, route, resident-row, and event-relative capacity is respected;
* every selected tag is in range and has no incompatible collision;
* the durable-boundary progress proof succeeds;
* all configuration values are derivable without a private default; and
* the exact MappingConstraintSet admits the result.

The five canonical final violation magnitudes are:

```text
UnroutedObligation
CapacityOveruse
TagUnassigned
TagConflict
HardProgressViolation
```

All must be zero before publication. Temporary-violation policy affects only
search states and cannot relax final legality.

## System Legality And Progress

A System candidate is base-legal only when it closes the complete `D/F/R`
execution domain and all selected hierarchical SpatialMapping references. It
must satisfy exact thread-to-AccCore and graph-to-SpatialMapping binding,
service target selection, terminal attachment, System RouteTrees, resource
states, tags, capacity, configuration, and `K` admission.

System service-target decisions are keyed by the exact
`ServicePlanSelectionAnchor` inside an execution context. Each addressed
memory member, fence member, and memory exposure selects from the exact target
domain derived from its own Spatial attachment. PnR must not intersect those
domains merely because the anchors share an execution context. A candidate
may share one plan only when the complete selected target and route semantics
are identical; materialization and the independent verifier reconstruct that
equivalence rather than assuming it.

For a finite logical-memory interval, this target domain is the exact
address-relation closure. For a dynamically unbounded `Whole` interval, it is
the complete structural service-envelope domain. PnR still selects one fixed
plan; invocation-specific extent and address admission remain Runtime ABI
state and cannot alter that selection.

System progress is candidate-dependent. The proof must rebuild the following
from selected Spatial mappings, System routes, services, UsePatterns, and
Fabric ResourceContracts:

```text
Dataflow event-causality index
selected Spatial and System route dependencies
post-divergence typed durable-boundary obligations
occurrence-qualified capacity cells and static route occupancy
atomic activation groups
trigger alternatives
capacity claims
causal release alternatives
occurrence-qualified physical owners, requesters, and grant policies
```

Activations with the same exact execution context, relation domain, and trigger
set acquire as one atomic group. Active holders and pending acquisitions are
different wait-for nodes. An active holder waits for pending groups whose
trigger is a causal predecessor of its release. A pending group waits for
active holders whose claims block its capacity. Ordinary contention between
two pending groups is not hold-and-wait.

The holder-to-pending edge uses the exact open activation interval. The two
activation relation domains must have a nonempty Presburger intersection, and
one compatible trigger alternative must be proven strictly after holder
acquisition and strictly before causal release. Fabric applies release before
acquisition at one event coordinate, so same-coordinate replacement and an
event preceding holder acquisition cannot manufacture a physical wait cycle.
Unordered alternatives do not establish a mandatory holder-to-pending wait
and therefore do not create an edge.

The proof returns `ProvenNoClosedWaitSet` only when the reconstructed wait-for
graph is acyclic and every arbitration case has a sufficient Fabric progress
guarantee. An actor cycle remaining after initialized feedback removal, an
initialized feedback edge without a durable selected disposition, a possible
physical wait cycle, a fixed-priority owner with multiple possible requesters,
or an unrepresentable relation returns `ProofNotEstablished`. It must not be
reported as a proven deadlock. An atomic activation whose demand plus baseline
occupancy exceeds a capacity is `ProvenClosedWaitSet`.

Search-time System progress and strict SystemMapping verification must rebuild
the same `MappingProgressProjection` and call the same closure algorithm.
Imported Spatial route obligations and selected System service-leg branches
are joined with capacity and ResourceContract activations before any
`ProvenNoClosedWaitSet` result is constructed. A reduced route-only,
arbitration-only, or Dataflow-only approximation cannot establish publication
legality.

## Objective Projection

The Mapping objective registry is `loom.mapping.pnr.objective 3.2`. It owns six
violation sources: the five structural/hard sources above plus
`ProgressProofDebt`, which is nonzero exactly for a selected
`ProofNotEstablished` activity witness. It also owns these ten nonnegative
measures in stable
ordinal order:

| Ordinal | Measure |
|---:|---|
| 0 | `TotalSelectedTraversalClaim` |
| 1 | `StaticSchedulePressure` |
| 2 | `RecurrenceMinimumInitiationIntervalCycles` |
| 3 | `ResourceMinimumInitiationIntervalCycles` |
| 4 | `TransportBitCycleDemand` |
| 5 | `WorstRouteArrivalDelayQuanta` |
| 6 | `TotalRouteNegativeSlackQuanta` |
| 7 | `SharedOperandIngressPressure` |
| 8 | `ProgressCapacityShortfall` |
| 9 | `ProgressRouteAnchorCount` |

`HardProgressViolation` is derived only from `ProvenClosedWaitSet`.
`ProgressProofDebt` is derived only from `ProofNotEstablished`; it remains a
temporary search violation and must be zero at publication. The capacity
shortfall is nonzero only when an exact static capacity proof exceeds the
selected shared pool. Route-anchor count is the number of distinct selected
physical traversals in the chosen hard or unestablished witness. These values
are one projection of `MappingProgressClosure`; a PnR consumer cannot infer
them independently from raw shared-FIFO incidence.

`SharedOperandIngressPressure` is the sum, over Dataflow-owned co-firing input
groups, of independently produced Temporal operand members beyond the number
of distinct selected physical ingresses. Repeated consumers of one logical
producer are collapsed before this count because their selected disposition is
atomic fanout. The measure is maintained incrementally from the frozen
demand-to-group incidence and is ranking-only; zero pressure is not a liveness
proof, while nonzero pressure is not an infeasibility proof.
System freeze partitions the measure by covered graph and associates each
value with the exact graph-choice ordinal. It sums selected graph executions;
it does not charge an inactive graph merely because the same SpatialMapping
artifact covers it.

`RecurrenceMinimumInitiationIntervalCycles` and
`ResourceMinimumInitiationIntervalCycles` derive from selected Fabric
UsePatterns, recurrence constraints, service plans, resource timing, and route
timing.
`TransportBitCycleDemand` is the canonical Dataflow transport payload width
multiplied by the exact cycle occupancy of transport demand. Dataflow owns this
width, including layout padding; Fabric endpoint and traversal widths are
capacity bounds and never redefine it. These are cycle and throughput costs,
not physical combinational delay.

Physical timing comes from one exact Fabric-owned
`FabricPhysicalTimingProfileView`. It binds the Fabric identity, required
combinational-delay budget, every exact traversal delay in integer provider
quanta, registered-destination boundaries, profile kind, provider identity,
technology identity, characterization identity, canonical bytes, and digest.
Mapping accumulates delay along each selected combinational route segment;
arrival resets after a registered destination. It derives worst arrival and
the sum of negative slack from the candidate's exact RouteTrees. System PnR
aggregates these measures from the unique selected SpatialMappings.

The normalized topology profile
`loom.fabric.physical_timing.normalized_topology.1.0` is deterministic,
target-neutral routing guidance. One normalized clock budget is eight quanta.
It is not target frequency, post-route slack, or EDA Evidence. A target timing
provider replaces it only through the same
`loom.fabric.physical_timing_profile.1.0` component-view contract. Its exact
profile is an explicit Spatial freeze input; silently reconstructing the
normalized profile in a target-characterized invocation is invalid. The
profile kind remains available to every diagnostic and Evidence consumer, so
a normalized result cannot be relabeled as target characterization.

Every selected objective dimension must have a complete owner in the candidate
domain. PnR config rejects Evaluation metric dimensions because the current
provider has no online Evaluation owner. Missing measures cannot be replaced
with zero. Objective arithmetic is checked and cannot saturate.

The builtin Spatial policy excludes
`RecurrenceMinimumInitiationIntervalCycles` from its selected ordering and
search energy because a Spatial boundary proxy deliberately leaves the
provider service plan to SystemMapping. The builtin System policy selects the
otherwise identical ordering and energy with recurrence included. A custom
Spatial policy may select recurrence only when every admitted candidate has a
complete local timing owner; an external manager dispatch then terminates as
`ProofNotEstablished` rather than receiving a provisional latency.

Both builtin total orderings place hard closure first, followed by proof-debt
witness count, capacity shortfall, and route-anchor count, before timing,
static schedule, traversal, and `SharedOperandIngressPressure`. Search energy
contains the same visible dimensions; PathFinder and annealing may not add an
unprojected activity penalty. `SharedOperandIngressPressure` remains the last
tie-break level and cannot override route, capacity, timing, or activity
quality.

The selected total ordering ranks candidates. The selected search energy may
guide stochastic acceptance but cannot legalize a violation. Final independent
verification remains authoritative even when a candidate has the best rank.

Annealing owns an explicit positive temperature-level limit in addition to its
per-level proposal formula. Calibration and objective magnitude affect the
temperature values and acceptance probabilities, never the maximum number of
levels. A bounded schedule executes at most that many levels and exactly one
minimum-temperature level. Each transition selects the colder of the ordinary
cooling result and the integer geometric envelope required to reach the
minimum within the remaining levels; a linear integer envelope covers the
sub-two ratio case. The deterministic work-budget projection publishes this
limit independently of the proposal counts.

## Search Policy And Determinism

Determinism is a global semantic requirement; algorithm choice is not. For one
exact invocation tuple, config bytes, provider/replay profile, and semantic
work limits, the canonical candidate set, termination class, work accounting,
and publication order must be independent of worker count, task completion
order, cache hits, allocator behavior, host topology, and wall-clock timing.

The provider profile owns every behavior that can change a bounded candidate
prefix, including candidate order, PRNG protocol, action order, router and
solver parameters, tie-breaking, repair extraction, and termination. The
global spec owns only these invariants:

* all finite domains have canonical typed keys;
* semantic work is counted before execution by its owner-defined unit;
* execution limits may interrupt work but cannot alter formal work order;
* isolated restart slots retain their original ordinals;
* parallel workers reduce results in canonical slot order;
* every accepted mutation is atomic and rollback restores all derived state;
* search always retains the best feasible incumbent found in a restart;
* `ExhaustConfiguredWork` does not stop quality optimization merely because a
  legal candidate was found;
* `FirstVerifiedCandidate` may stop annealing when a feasible incumbent is
  available, but may stop later restart slots only after that incumbent passes
  final global closure, independent verification, finalization, and
  publication;
* a first-candidate result reports `SemanticLimitReached`, never exhaustive
  completion or infeasibility;
* the selected incumbent always receives final global closure and independent
  verification;
* one incomplete parent or restart does not suppress independent work;
* a heuristic preference cannot remove a legal value from the domain; and
* a temporary cut or repair witness cannot become a persistent Mapping fact.

Provider-specific work limits, route negotiation, annealing, and exact repair
are specified only in
[Builtin Mapping PnR Replay Profile](spec-pnr-provider-builtin.md). The global
contract intentionally defines no FrozenModel layout, CSR/SoA representation,
thread-count formula, PathFinder recurrence, annealing schedule, CP-SAT
parameter block, or cache structure.

## Cross-Layer Candidate Feedback

TechMapping, SpatialMapping, and SystemMapping remain distinct immutable result
views. Their publication boundaries must not become irreversible search
boundaries.

The preferred design is a lazy Tech row and cover domain consumed by a bounded
joint search. A provider may instead return invocation-local typed conflicts or
no-goods from Spatial or System search to the upstream generator. Such feedback
must identify exact canonical upstream decisions, be reconstructable from the
same owner inputs, remain scoped to one invocation, and never become a
persistent Artifact or a second legality owner.

Feedback may prune only the exact conflicting combination proven by its
witness. A failed route under fixed attachments cannot globally exclude a Tech
realization, and a temporary cut cannot survive rollback unless its proof is
valid for every assignment it excludes. Retained verified candidates and
search completeness are independent outputs throughout the pipeline.

A System fixed-terminal capacity witness identifies the exact capacity cell
and contributing service legs. Its graph-binding feedback is derived through
the selected service contexts; it is not an independently authored conflict.
When the proof depends on several graph decisions, those decisions and their
hard relation closure must be reopened jointly. Such feedback consumes the
provider's ordinary bounded work and is discarded when its candidate basis is
invalidated. Exhausting that work before a reopened binding closes is
`ProofNotEstablished`, not an internal error and not proof that every upstream
SpatialMapping combination is infeasible.

An imported-capacity witness may request hardware reconsideration only after
the complete bounded execution-binding relation has been exhausted. The
witness names the exact System, complete SpatialMapping input frontier, target
Module, compatible AccCore occurrence count, assignment work, usage, and
capacity. It requests one monotonic candidate extension; it does not claim one
additional AccCore is sufficient, cannot exclude a software alternative, and
cannot bypass ordinary System routing or verification on the child.

Mutable search may cache and transactionally update a removable physical
projection, but every transaction is defined by immutable before/after
selections and an explicit affected dependency cone. Rejection must restore
the exact prior candidate without inverse reconstruction. A candidate that
becomes the working or best feasible state must agree with a full projection
rebuilt from its selections; final publication repeats that independent path.
Delta caches, objective increments and rollback records are provider state and
never enter Mapping identity.

## Final Closure And Independent Verification

Before publication, the provider must apply one final global closure to the
selected best feasible incumbent. That closure uses the ordinary candidate
transaction and exact selected routing semantics. It must not silently keep a
partial RouteTree, provisional tag, over-capacity resource, unclosed service,
or unresolved progress proof.

Finalization then performs three independent checks:

1. the provider's candidate-state invariant checker;
2. the cold Mapping base verifier, which rebuilds all physical demand,
   capacity, tags, configuration, timing inputs, and progress without search
   caches; and
3. exact MappingConstraintSet admission.

The base verifier is the publication authority. A solver result, zero-valued
incremental counter, cached projection, or generator self-report cannot replace
it. Any disagreement is `InternalError`, not a candidate rejection that search
may conceal.

Only after all checks succeed may canonical Mapping MLIR be finalized and its
ArtifactRootReference enter the returned candidate set. Configuration images,
Deployment, simulation, and hardware backends consume only finalized Mapping
references and mechanically rebuilt projections.

## Validation Anchors

Reusable tests are warranted at semantic joints where an implementation could
otherwise return a plausible but false result. The minimum retained evidence
must cover:

* Tech realization admission and independent-verifier agreement;
* a sound exact-cover lower bound and cross-family candidate fairness;
* isolation of incomplete candidates from independent traversal;
* preservation of the best feasible Spatial and System incumbents;
* unconditional final global closure and independent verification;
* Temporal Memory occurrence-global rows and unique ingress;
* atomic Temporal PE operand fanout and durable progress boundaries;
* RegFIFO local disposition with legal external fallback;
* Temporal switch signature packing and row-aware execution;
* candidate-dependent System wait-for reconstruction;
* physical timing affecting route and candidate ranking;
* rejection of schema or capability claims absent from the provider;
* preservation of generic descriptor-validated infeasibility proofs through
  the Candidate Generator boundary and replay;
* rejection of a forged or corrupted owner witness against the exact input
  closure;
* fail-closed adaptation of current Spatial and System internal
  contradictions to `ProofNotEstablished`; and
* preservation of ordinary completed empty output, timeout, and
  `ProofNotEstablished` without reclassification as infeasible.

Schema-only fixtures and mock success paths do not replace a real application
tuple. Integration acceptance requires a fresh, independently verified
Dataflow-to-TechMapping-to-SpatialMapping-to-SystemMapping chain, derived
configuration and Deployment, execution evidence, and exact SpatialCore
physical Evidence within the HardwareImplementation scope.

## Temporal Operand Progress

The PnR progress projection consumes the Fabric-owned ordered queue projection
after ingress, tag, context, FU, and input-role selections are known. It checks
allocation-unit capacity and service claims, then derives a transient
qualified-pairing risk from the same QueueKeys used by the simulator and strict
import. A likely shared-ingress risk only changes route objective ordering;
only an exact closed wait with a complete causal cone may reject a candidate or
request a bounded local repair. Unknown rates, dynamic aliases, and incomplete
queue witnesses remain `ProofNotEstablished` or `Unsupported`.

Selected transport storage is a separate feedback boundary. If a closed actor
wait crosses multiple route-storage owners and no one FIFO or operand queue is
the complete witness, runtime must name the canonical producer and selected
physical traversals. Spatial feedback independently joins those references to
the finalized RouteTree. A bounded retry may exclude one exact contested
traversal through `NetSelectedPhysicalTraversals`; it may not infer a storage
resize or bypass queue/FIFO ownership. Without a verified mutable checkpoint,
the retry is a constrained cold Mapping invocation rather than an incremental
repair claim.

When route cost, capacity, and functional objective are equal, the Spatial
router prefers distinct compatible ingresses for input roles in one potential
wait component, followed by pairing-ready selector arrangements. This is a
central objective measure, not a hidden router legality rule. TechMapping may
provide ordered-role, boundary, internal-edge, rate, and fanout facts, but it
never chooses a physical ingress, tag, context, or allocation unit.
