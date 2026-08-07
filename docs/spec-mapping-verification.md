# Mapping Verification

This document is the authority for observable verification behavior of the
three immutable Mapping profiles. `docs/spec-mapping-artifact.md` owns their
persistent records and schema. `docs/spec-mapping-identity.md` owns identity
and reference semantics. `docs/spec-tech-mapping.md` owns production
TechMapping generation, while `docs/spec-pnr.md` owns the complete Spatial and
System MappingConstraintSet family and admission algebra.

A profile verifier proves intrinsic legality and closure for one complete
artifact. It does not search, repair records, select a fallback, complete a
consumer-specific view, mutate upstream artifacts, or persist a proof.

## Common Reader Contract

A persistent Mapping read follows one ordered contract:

```text
verify Common envelope, schema descriptor, and digest
  -> dispatch the exact supported loom.mapping version parser
  -> parse exactly one typed profile root
  -> resolve exact UpstreamArtifactBindings through owner-family importers
  -> resolve scoped references in those independently verified projections
  -> verify canonical IDs, ordinals, and structural keys
  -> run the independent profile base verifier
  -> derive immutable consumer views
```

Unknown operations, fields, enum values, record families, or unsupported
versions are rejected. A reader cannot preserve them in an extension bag,
ignore them, infer a default not owned by the schema, or fill a missing field
from a C++ view, native cache, runtime image, or manifest.

The verifier may collect multiple findings while reference resolution remains
safe. Findings have stable typed subjects and categories, but diagnostic text
and ordering within one subject are not persistent schema. A consumer cannot
append or alter records to make an artifact valid; a producer must construct
and finalize another complete artifact.

## Failure Boundary

An invalid or incomplete candidate is not a Mapping artifact. Unsupported
input, proven infeasibility, inconclusive search, budget exhaustion, external
failure, and internal error are invocation outcomes. They do not become
Mapping root status values.

Human-readable diagnostics and temporary witnesses belong to ordinary
results or reports. When an evaluation was requested, the corresponding
observations and findings belong to Evaluation Evidence. No failure, report,
finding, proof witness, or Evidence record enters Mapping semantic bytes.

A malformed, foreign, wrong-kind, wrong-owner, or out-of-range persistent
reference is invalid input and cannot enter search. A purported Mapping root
that omits any service obligation, member, or leg mechanically derived from
its exact workload scope fails profile completeness. In contrast, a
well-formed obligation whose compatible Fabric target domain is empty is a
`ProvenInfeasible` Mapping invocation outcome, not a reference or identity
error.

## TechMapping Verifier

The TechMapping verifier consumes one `mapping.tech` `3.0` root and its exact
Canonical Dataflow Program `D` and Fabric Hardware Description `F`. It checks
at least:

* the exact `D` and `F` UpstreamArtifactBindings, the Dataflow-owned
  `CanonicalDataflowProgramView`, and all scoped reference kinds and owners;
* the canonical non-empty covered-graph set;
* one artifact-global `EntityId` namespace shared by Compute and Memory
  Realizations;
* unique realization and child semantic keys;
* disjoint and complete actor coverage for every covered graph;
* correct addressed-memory and fence ownership by Memory Realizations and all
  other actor ownership by Compute Realizations;
* exact resolution of each selected `FabricFuCapabilityTemplateRef`, including
  owner, ordinal, active nodes, active edges, and exact parameterized
  capability matching against Dataflow actor semantics;
* complete ordered actor operand/result and FU-boundary correspondences;
* derived FU implementation, actor set, configured-function topology,
  active-port behavior, and semantic configuration;
* exact classification of a selected-template edge from a mapped dead actor
  result to an FU output boundary as a derived discard requirement, with no
  `mapping.compute_boundary` or residual logical net, and rejection of every
  other unmatched selected-template edge;
* exact resolution of each selected `FabricMemoryEngineTemplateRef`, its
  complete engine contract, token endpoints, operation ports, capability
  alternatives, ResourceContracts, and internal-connection relation;
* exact template-relative operation-port and capability-alternative
  correspondence, logical-root coherence, and token/value/control graph-
  boundary correspondence;
* exact equality between selected template-relative Fabric internal
  connections and canonical software-edge witnesses;
* exact derived `CanonicalMemoryAccessView` compatibility with the selected
  Fabric operation port, capability alternative, and declared use-pattern
  domain, including actor contract, access form, memory-element width,
  access-lane-shape projection, lane count, address, data, and mask capacity,
  alignment, and narrow-access semantics;
* exact distinction between a shared hybrid operation port and separate
  element and vector ports, with no persisted derived geometry class;
* Fabric-owned fanout, service-domain, port-kind, representation, and capacity
  rules, with equal total width insufficient for a memory match; and
* exact classification of every canonical edge as realization-internal or an
  externally derived obligation.

The verifier derives state, timing, and use-pattern requirements from the
selected template's active concrete Fabric nodes. It rejects a
Mapping-owned encoding descriptor, copied configured graph, copied resource
contract, or backend-local support record as a competing authority. Missing
backend support is checked separately and reported as typed `Unsupported`; it
does not make a semantically valid custom Fabric or TechMapping malformed.

Software-edge identity is the exact `D` binding plus typed producer and
consumer endpoint keys. There is no edge entity, number, symbol, path,
printer-order, or insertion-order fallback.

For sync and other parameterized operations, verification uses the registered
operation schema, selected Fabric capability template, and ordinary ordered
actor port relations. It derives masks and configured fields. It must not
require an operation-specific persistent record or verifier path.

A valid TechMapping is profile-complete. Candidate enumeration status, search
scores, and proof that no alternative cover exists are not TechMapping
verifier concerns.

## SpatialMapping Base Verifier

The intrinsic verifier is:

```text
SpatialMappingBaseVerifier(D, T, F, S)
```

It reads only the exact upstream artifacts and the complete SpatialMapping
`S`. It does not read resolved config `C`, MappingConstraintSet `K`, candidate
or FrozenModel caches, search history, Evaluation Evidence, or runtime state.

The verifier checks in dependency order:

* the `mapping.spatial` `3.0` root shape and exact `T`, `D`, and `F` bindings;
* `T.D == D`, `T.F == F`, and complete inherited TechMapping coverage;
* exactly one ComputeBinding per Compute Realization and one
  MemoryEngineBinding per Memory Realization;
* for every TechMapping-derived dead-result discard requirement, an exact
  selected FU occurrence whose enclosing PE output selector is `Discard`, not
  `Disconnected`, with no RouteTree or transport attachment for that result;
* unique canonical MemoryBinding semantic keys and all record, child,
  logical-net, and ResourceUse structural keys;
* FU and memory occurrence membership, instruction and operation context
  range, physical port compatibility, and active physical-refinement domains;
* equality between each placed FU occurrence's Fabric-owned definition and
  the owner of the Compute Realization's selected capability-template
  reference, with template nodes and ports mapped mechanically to that
  occurrence;
* equality between each selected memory occurrence's
  `memoryEngineTemplate(occurrence)` relation and the Memory Realization's
  selected engine template, with template ports, capability alternatives,
  endpoints, and internal connections projected mechanically to that
  occurrence;
* one MemoryOperationEntry per covered memory actor, its exact addressed or
  fence variant, exact definition-level memory placement, and derived internal
  source;
* one canonical rooted-use row per reachable `ContextualActorRef`, including
  exact addressed-memory resolution, required MemoryBinding, typed dispatch or
  consistency target, and rejection of missing, duplicate, foreign, stale, or
  wrong-graph rows;
* complete vector address, data, and optional mask endpoint correspondence,
  selected use-pattern compatibility, and absence of any Mapping-invented
  lane/beat decomposition;
* for every Temporal memory row, one derived input match per externally
  supplied role and one output write per externally exposed result role,
  uniqueness of incompatible interpretations within each local physical
  ingress match domain, and legal tag reuse across disjoint domains;
* one typed target on every MemoryOperationUse and ExposureEntry, with the
  reconstructed `C_dispatch` contained in Fabric-owned `H_dispatch`;
* exact agreement between each addressed or exposure target and its
  MemoryBinding target: local dispatch selects a LocalRegion owned by the
  same local service, while manager dispatch selects a BoundaryProxy;
* exact actor-contract compatibility and derivation of one compatible
  MemoryConsistencyDomain for every addressed atomic actor and fence, with
  complete fence-effect coverage and no Mapping-created multi-domain join;
* compatible volatile and MMIO service-region behavior, including exact
  accepted access, non-trapping SpatialCore execution, and at-most-once
  provider-observable operation semantics;
* each MemoryBinding's logical interval and closed target; finite logical and
  translated physical containment for LocalRegion; absence of local service,
  region, endpoint, transform, and provider state for BoundaryProxy; and
  partition, replication, and coherence legality;
* every RouteTree's root, parent traversal continuity, arborescence, explicit
  fanout, no reconvergence, and complete sink coverage;
* every selected route segment's data field can carry the complete software
  payload, tag bits are not counted as payload capacity, and every assigned
  tag is independently losslessly representable wherever it remains live;
* each vector-memory address, data, and mask token remains complete on its
  route, and Physical Tags identify sharing interpretations rather than lanes;
* complete ResourceUse ownership, Fabric use-pattern resolution, atomic
  multi-ResourceState claim envelopes, owner-defined commit-transition
  resolution, relative activation, typed parameters, capacity, sharing
  assignment, exact grant-policy refinement, and Physical Tag legality;
* configured semantic realization and mapping-visible physical refinement
  without a duplicate configuration authority;
* ordered-dataflow, resource-time, memory-service, Tag, and configuration
  continuity; and
* progress and deadlock closure from existing causal events, selected routes,
  finite resources, queues, atomic admission, releases, and Fabric
  guarantees.

### Selected Combinational Handshake Closure

Combinational handshake closure is an intrinsic Mapping gate over the exact
upstream Fabric and the exact Mapping under verification:

```text
verifySelectedHandshakeClosure(FabricArtifactView F,
                               SpatialMapping | SystemMapping M)
  -> success
   | Spatial Invalid(SelectedCombinationalHandshakeCycle)
   | System Rejected(SelectedCombinationalHandshakeCycle)
```

`F` must be the sealed root-complete view produced after whole-root
elaboration. A residual `fabric.instantiate`, partial connection list, or
caller-supplied dependency graph is invalid input rather than an incomplete
cycle proof.

The verifier first derives the complete configured selection from existing
Mapping facts: selected FU capability templates and correspondences, memory
operation/use patterns, RouteTree traversals, service-plan transfer patterns,
resident switch rows and tags, ResourceUse selections, and physical refinement
assignments. It then asks the owning Fabric resource contracts for the exact
ready/valid dependency arcs of those selected fragments and composes them with
the root-complete point connections. The signal and arc types are owned by
`docs/spec-fabric-module.md`. This configured graph is derived state and is
never persisted in Mapping, Fabric, ConfigurationABI, or Deployment.

Only selected active alternatives contribute arcs. Disabled switch outputs,
unused temporal rows, inactive FU nodes, and unselected FIFO modes contribute
nothing. Conversely, every resident alternative that may be active under the
selected configuration contributes its Fabric-owned arcs; runtime tags,
traffic values, token coordinates, or observed traces cannot be used to erase
a structural dependency. ConfigurationABI bits are derived later and cannot
change this graph.

For `fabric.fifo`, bypass contributes its transparent forward-valid and
backward-ready arcs. Buffered mode contributes exactly the arcs derived by the
FIFO contract, including any remaining same-cycle ready dependency; it is not
assumed to break every cycle merely because it owns storage. A stronger
registered break is available only through an exact Fabric capability or
Mapping-selected refinement.

The final graph must be a directed acyclic graph. A cycle is an intrinsic
base-verifier failure, not congestion, a temporary capacity violation, a QoR
metric, or an Evaluation finding. The diagnostic may carry a canonical sorted
cycle witness for explanation, but that witness is not persistent semantic
content.
SpatialMapping checks the complete SpatialCore-local graph. SystemMapping
requires every exact imported SpatialMapping to have passed its own gate, then
composes the active arcs mechanically derived from those immutable mappings
with the System transport and spatial-attachment arcs and checks the complete
combined graph. `hierarchical` and `flat` are search decompositions, not
verifier modes. A flat search that changed a reopened Spatial decision must
first finalize the replacement SpatialMapping before constructing the one
ordinary SystemMapping.

For the System form, `M`'s exact import table and upstream bindings
mechanically resolve those SpatialMappings and their Module Fabric views. They
are not an additional caller parameter or a supplied graph summary.

Validation fails fast by dependency layer: schema and upstream references;
coverage and uniqueness; local placement, memory, and route structure;
derived configuration, resources, Tags, and selected handshake closure; then
progress closure. A later layer cannot guess a repair when an earlier fact is
invalid.

The derived-configuration layer invokes the unique cold operation specified by
[Fabric Reconfigurable Operations](spec-fabric-reconfigurable-op.md). For each
Mapping-selected physical configuration slot, it resolves the exact
TechMapping actor and capability relation, the SpatialMapping occurrence,
context, and the Fabric-owned typed field projector.
Repeated derivations of one slot must produce byte-identical canonical typed
values. A missing value, an unencodable value, or two different values for one
slot is `Invalid` Mapping input. Distinct independently configurable contexts
remain distinct slots.

The resulting `ConfiguredHardwareProjection` is a sealed, removable in-memory
view indexed only by existing Dataflow, Fabric, and Mapping references. It is
not serialized in Mapping, assigned a new identity, or reconstructed by a
simulator or backend. Mapping import may retain it as an invocation-local
cache. CGRA admission consumes that validated view, while configuration-image
finalization passes the same semantic values to the exact
`ConfigurationABI` encoder.

Physical refinements are not part of this projection. The current exact
Mapping contract has no generic
physical-refinement value codec, so strict Mapping import rejects every
nonempty refinement assignment before configured-hardware projection. A
concrete Fabric owner must first publish the domain's closed typed value codec
and admissibility relation. Raw bytes, an ordinal fallback, or a
simulator-private interpretation cannot stand in for that owner.

The intrinsic semantic result is either `Valid` or `Invalid` with typed
diagnostics. A passing verifier does not persist `valid`, closure projections,
claims, calendars, or proof witnesses in `S`.

## Spatial Constraint Admission

Spatial constraint admission is the separate invocation gate:

```text
SpatialMappingConstraintAdmission(D, T, F, K, S)
```

It runs only after base verification and answers whether the base-valid `S`
satisfies the exact MappingConstraintSet `K` used for that run. Admission
independently evaluates the owner-defined projections from `S` for the stable
pre-result subjects in `K` and recomputes every canonical clause. It does not
trust search-state claims, FrozenModel indexes, or solver caches. The closed
atoms, subject and carrier typing, canonicalization, and outcome distinctions
are owned only by `docs/spec-pnr.md`.

Failure is `RejectedByConstraintSet`, not intrinsic invalidity. `K` remains an
independent canonical MappingConstraintSet Artifact. Its exact ArtifactIdentity and the admission
result belong to invocation metadata or Evaluation Evidence and do not enter
SpatialMapping identity. The same immutable `S` may therefore be base-valid,
admitted by one exact `K`, and rejected by another.

Only a base-valid and admitted draft may proceed through canonical ID and
ordinal assignment. Structural root verification then checks the canonical
MemoryBinding IDs, RouteTree node ordinals, attachment order, and all finalized
references before canonical serialization and Common finalization.

## SystemMapping Base Verifier

The intrinsic verifier is:

```text
SystemMappingBaseVerifier(D, F, M, ExactSpatialMappingSet(M))
```

`M` is a `mapping.system` `3.0` root. `F` supplies the architecture-only
Fabric system and exact Transport Architecture; protocol-specific
Interconnect Implementation is not a Mapping input.

It first derives `ExactSpatialMappingSet(M)` from the finite unique range of
normalized `B_graph` over every reachable static graph launch and legal
may-domain point. It requires the canonical SpatialMapping import table to
equal that set exactly. Missing, extra, duplicate, unreachable, or foreign
imports are rejected. An empty set is valid only for a closure with no
reachable static graph launch; `root_thread_launches` remains non-empty.

Every imported SpatialMapping must resolve, pass its own base verifier, bind
the same exact `D` and `F`, and preserve its exact TechMapping lineage. The
System verifier does not rematch Dataflow to Fabric or reopen imported
SpatialMapping decisions.

The shared Selected Combinational Handshake Closure gate above is one layer of
this base verifier. It resolves imported active arcs through
`ExactSpatialMappingSet(M)` and introduces no System-specific graph authority.

The verifier derives one nonpersistent shared projection:

```text
SystemMappingClosureProjection =
  Derive(D, F, M, ExactSpatialMappingSet(M))
```

The projection occurrence-qualifies imported Spatial resource uses, rebases
their event families into each graph-launch context, composes complete
cross-Spatial/System service paths, and derives capacity, acquire, release,
and wait-for closure. It has no ArtifactIdentity and is not a fourth Mapping
profile, record family, proof object, or runtime image.

Using that projection, the base verifier checks:

* exact root-launch coverage and derivation of all reachable thread, graph,
  channel, memory, and external obligations;
* exact resolution of every `EventFamilyKey` as one Dataflow-owned
  `StaticTransferEventRef`, mechanical derivation of its canonical
  `EventLogicalProjection`, and rejection of copied, foreign, duplicate,
  wrong-kind, out-of-range, or noncanonical event input slots;
* exactly one ThreadExecutionBinding per root launch and one
  GraphExecutionBinding per reachable static graph launch in each root
  context, keyed by the Dataflow-owned `RootedGraphLaunchRef`;
* total, single-valued, well-typed `B_thread` and `B_graph` relations over
  Dataflow-owned may-domains, including exact default-complement rules;
* agreement between every selected SpatialMapping target and the AccCore
  selected by its parent thread binding;
* exact derivation of each reachable `ExecutionContextKey` from the applicable
  execution-binding results, including `B_thread`-only Instruction contexts
  and distinct Spatial contexts when one SpatialMapping is reused by different
  AccCore occurrences;
* derivation of exactly one Fabric-owned `InstructionCoreContextRef` for each
  selected AccCore, with no competing target in InstructionCore ResourceUse;
* exact resolution of every InstructionCore use site to a Fabric-owned atomic
  `UsePattern`, including its initial state, capacity, requester order, grant
  contract, typed demand, activation, optional commit transition, and release;
* exactly one ServiceRealization per derived transfer or operation-service
  obligation;
* exact derivation of each obligation's complete canonical member, sink, and
  `CanonicalServiceLegKey` universes from the workload scope and Canonical
  Service Schema;
* exact derivation and binding of every `MemoryExposureRef` through its
  logical-memory owner, with no exposure admitted as a `ServiceMemberRef` or
  assigned a service leg;
* exact derivation of each memory or fence leg's source and sink terminal
  domains from the evaluated execution bindings, immutable SpatialMapping-
  selected Module-local manager path, occurrence-qualified memory endpoint,
  unique Fabric memory `spatial_attachment` row, Module/occurrence endpoint
  pair and exact System service endpoint within that row, and each
  role-selected `ServiceLegCarrierAttachment`, including canonical leg
  direction and exact compatibility between every selected Dataflow service
  member and the pair's one System endpoint capability domain, including when
  the selected terminal carrier row belongs to the occurrence endpoint, with
  no cross-endpoint union or intersection, Fabric-root workload compatibility
  result, Module-boundary capability projection, alternative-endpoint search,
  or attachment row used for `MessageTransfer`;
* exact derivation of every ServiceRealization's complete member-or-exposure
  selection-anchor set, including the singleton message anchor relative to its
  producer obligation, and rejection of a missing, extra, foreign, duplicate,
  or wrong-kind anchor;
* exactly one non-empty plan-selection row for every reachable
  `(anchor, ExecutionContextKey)` pair, no unreachable row, disjoint row
  domains whose union is the anchor's legal may-domain, total single-valued
  relations, exact plan-ordinal ranges with no unused plan, and canonical
  selection-row ordering;
* mechanical derivation of each message anchor's Instruction or Spatial
  execution context from its root-boundary, graph-boundary, thread-channel, or
  graph-stream producer kind, plus the exact canonical applicable
  `(sink terminal, execution owner)` set at every producer point from every
  consumer domain and Dataflow-owned `source_map`, without a copied
  terminal-context tuple;
* exact agreement between every selected message route sink and that
  applicable pair set throughout its plan-selection range: no missing or
  inactive terminal, no extra owner, no cross-owner endpoint union, no stale
  route reuse, and no endpoint fallback; one terminal may repeat only for
  distinct owners derived from its attached Fabric endpoints, while a
  duplicate terminal-owner pair is invalid;
* acceptance of a childless `MessageTransfer` plan exactly when its complete
  applicable sink-owner set is empty throughout the selected relation range,
  and rejection of a childless non-message plan or a sinkless
  `TransferLegRealization`;
* valid service targets, canonical service legs, flat route-tree continuity,
  multicast ownership, and physical refinements;
* total selection of one legal source and the exact required sink-attachment
  set by every materialized service-leg RouteTree, including every static sink
  for memory and fence legs and every applicable terminal-owner pair for a
  message leg, rejection of a terminal outside its exact attachment domain,
  rejection of a service target outside the bound endpoint's explicit
  service/transform closure, and no copied memory endpoint, execution owner,
  capability ordinal, payload, width, or protocol field in Mapping;
* exact continuity and minimality of each memory target plan across explicit
  Fabric MemoryService connections and selected service-transform paths,
  composition of Fabric-owned address width, offset, mask, interleave, and
  coherence contracts without copied parameters, arithmetic without overflow,
  and exact collective branch coverage: every source address reaches exactly
  one selected target child and its transformed address is contained in that
  child's selected service region, with no missing or extra transform output;
* one exact MemoryConsistencyDomain target for each fence plan, compatible
  with its synchronization scope and all constrained memory effects;
* complete System ResourceUse ownership, exact `ServicePlanElementRef`
  resolution where applicable, use-pattern resolution, atomic
  multi-ResourceState claim envelopes, owner-defined commit-transition
  resolution, relative activation and release, typed demand, capacity, exact
  grant-policy refinement, and sharing assignments;
* occurrence qualification of every imported Spatial use without copying it
  into System records;
* end-to-end attachment, service, route, address, Tag, context, and
  configuration continuity;
* atomic activation and release closure over all selected and imported uses;
  and
* progress and deadlock closure using the confirmed Fabric guarantees and
  existing Dataflow causal events.

All verifier modules consume the same closure projection. They cannot build
independent interpretations that disagree about imported use qualification,
service continuity, capacity, or progress.

The System base result uses this closed algebra:

```text
Verified
Rejected(typed closure findings)
Incomplete(unsupported | proof_not_established)
InternalError
```

A malformed reference, broken path, hard-capacity counterexample, or replayable
closed wait set is `Rejected`. Failure to establish a required progress proof
is `Incomplete(proof_not_established)` and is distinct from a proven
deadlock. A finite simulation that observes no deadlock cannot produce
`Verified`. Only `Verified` is eligible to proceed to final admission.

## System Constraint Admission

System constraint admission uses the exact System root and algebra owned by
`docs/spec-pnr.md`; this document defines no competing contract:

```text
SystemMappingConstraintAdmission(
  D, F, root_thread_launches, K, M)
```

It requires `K` to bind the same exact `D`, `F`, and canonical non-empty root
launch set as `M`, then independently recomputes every System projection and
clause from the base-verified Mapping and its exact SpatialMapping imports. The
search domain, resolved config, constraint-set ArtifactIdentity, and admission result
remain invocation metadata or Evaluation Evidence and do not enter
SystemMapping identity.

Constraint rejection does not relabel a base-verified SystemMapping as
intrinsically invalid. Quality gates, objective availability, candidate
ranking, and promotion remain central DSE and Evaluation responsibilities,
not Mapping constraint or base-verifier outcomes.

## Determinism

Verification depends only on canonical semantic inputs and the applicable
profile contract. Ordering derives from exact artifact identities, typed
artifact-local identities, structural keys, and explicit semantic ordinals.
Symbols, source-vector order, textual record order, native dense indices, and
builder insertion order are not authorities.

The verifier recomputes derived facts rather than trusting persistent
`legal`, `verified`, active-mask, claim, closure, or configured-view fields.
Those fields do not exist in the Mapping schema.

## Anchor Tests

Tests should protect stable semantic anchors:

* exact 3.0 schema profile, rejection of every 2.0 profile root,
  UpstreamArtifactBinding, predecessor, and import coupling;
* authoring-order invariance of canonical bytes and ArtifactIdentity;
* foreign, wrong-kind, wrong-owner, unknown-field, duplicate-key, and
  noncanonical-ID rejection;
* TechMapping closed coverage, ordered parameterized capability relations,
  multiple results, variadic ports, exact vector-memory access compatibility,
  and exact memory internal-edge witnesses;
* Spatial record totality, mixed identity rules, RouteTree arborescence,
  route-wide widening acceptance, narrowing and tag-borrow rejection,
  complete vector-memory token routing, declared multi-transaction memory use
  patterns, MemoryBinding and exposure closure, ResourceUse, Tags, and
  progress;
* one potential hardware cycle with two mutually exclusive switch traversals,
  where the Fabric remains valid, one selected Mapping is acyclic, and another
  selected Mapping is rejected; one equivalent bypass/refinement case; and one
  selected cycle crossing an elaborated former instance boundary;
* separation of Spatial base validity from exact `K` admission;
* exact equality of the System import table and normalized `B_graph` range,
  including a legal InstructionCore-only empty table;
* System relation totality, parent-AccCore agreement, service continuity,
  service-leg carrier attachment continuity, imported-use occurrence
  qualification, capacity, and positive and negative progress anchors;
* message-plan projection for a non-surjective `source_map`, including an
  inactive terminal and a childless plan; non-injective projection of one
  terminal to distinct owners, including one branch per owner; collapse of
  several consumer points to one terminal-owner pair; and rejection of a
  missing, extra, or duplicate pair; and
* separation of System base verification, exact `K` admission, and Evaluation
  quality gates.

Tests must not preserve printer whitespace, diagnostic prose, C++ container
layout, native cache shape, optional-field Cartesian products, retired
record models, placeholder profiles, generic bags, runtime/deployment payload
round trips, or implementation-specific verifier decomposition.
