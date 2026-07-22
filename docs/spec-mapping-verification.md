# Mapping Verification

This document is the authority for observable verification behavior of the
three immutable Mapping profiles. `docs/spec-mapping-artifact.md` owns their
persistent records and schema. `docs/spec-mapping-identity.md` owns identity
and reference semantics. `docs/spec-pnr.md` owns the Spatial constraint
algebra and the single `MappingConstraintSet Persistent Family Frontier`
statement.

A profile verifier proves intrinsic legality and closure for one complete
artifact. It does not search, repair records, select a fallback, complete a
consumer-specific view, mutate upstream artifacts, or persist a proof.

## Common Reader Contract

A persistent Mapping read follows one ordered contract:

```text
verify Common envelope, schema descriptor, and digest
  -> dispatch the exact supported loom.mapping version parser
  -> parse exactly one typed profile root
  -> resolve exact UpstreamArtifactBindings and scoped references
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

## TechMapping Verifier

The TechMapping verifier consumes one `mapping.tech` `1.0` root and its exact
Canonical Dataflow Program `D` and Fabric Hardware Description `F`. It checks
at least:

* the exact `D` and `F` UpstreamArtifactBindings and all scoped reference
  kinds and owners;
* the canonical non-empty covered-graph set;
* one artifact-global `EntityId` namespace shared by Compute and Memory
  Realizations;
* unique realization and child semantic keys;
* disjoint and complete actor coverage for every covered graph;
* correct load/store ownership by Memory Realizations and all other actor
  ownership by Compute Realizations;
* selected FU capability-template ownership and exact parameterized
  capability matching against Dataflow actor semantics;
* complete ordered actor operand/result and FU-boundary correspondences;
* derived FU implementation, actor set, configured-function topology,
  active-port behavior, and semantic configuration;
* selected memory semantic-encoding ownership, operation-template and port
  correspondence, logical-root coherence, and graph-boundary correspondence;
* exact equality between selected Fabric internal connections and canonical
  software-edge witnesses;
* Fabric-owned access-size, alignment, narrow-access, fanout, service-domain,
  port-kind, representation, and capacity rules; and
* exact classification of every canonical edge as realization-internal or an
  externally derived obligation.

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

* the `mapping.spatial` `1.0` root shape and exact `T`, `D`, and `F` bindings;
* `T.D == D`, `T.F == F`, and complete inherited TechMapping coverage;
* exactly one ComputeBinding per Compute Realization and one
  MemoryEngineBinding per Memory Realization;
* unique canonical MemoryBinding semantic keys and all record, child,
  logical-net, and ResourceUse structural keys;
* FU and memory occurrence membership, instruction and operation context
  range, physical port compatibility, and active physical-refinement domains;
* one AccessEntry per covered memory actor and exact memory placement,
  internal source selection, MemoryBinding, typed dispatch target, and
  exposure closure;
* one typed dispatch target on every AccessEntry and ExposureEntry, with the
  reconstructed `C_dispatch` contained in Fabric-owned `H_dispatch`;
* each MemoryBinding's logical interval, physical service region, transform,
  partition, replication, and coherence legality;
* every RouteTree's root, parent traversal continuity, arborescence, explicit
  fanout, no reconvergence, and complete sink coverage;
* every selected route segment's data field can carry the complete software
  payload, tag bits are not counted as payload capacity, and every assigned
  tag is independently losslessly representable wherever it remains live;
* complete ResourceUse ownership, Fabric use-pattern resolution, atomic
  multi-ResourceState claims, relative activation, typed parameters,
  capacity, sharing assignment, exact grant-policy refinement, and Physical
  Tag legality;
* configured semantic realization and mapping-visible physical refinement
  without a duplicate configuration authority;
* ordered-dataflow, resource-time, memory-service, Tag, and configuration
  continuity; and
* progress and deadlock closure from existing causal events, selected routes,
  finite resources, queues, atomic admission, releases, and Fabric
  guarantees.

Validation fails fast by dependency layer: schema and upstream references;
coverage and uniqueness; local placement, memory, and route structure;
derived configuration, resources, and Tags; then progress closure. A later
layer cannot guess a repair when an earlier fact is invalid.

The intrinsic semantic result is either `Valid` or `Invalid` with typed
diagnostics. A passing verifier does not persist `valid`, closure projections,
claims, calendars, or proof witnesses in `S`.

## Spatial Constraint Admission

Spatial constraint admission is conditional on the
`MappingConstraintSet Persistent Family Frontier` owned by
`docs/spec-pnr.md`. No concrete invocation or Spatial PnR publication exists
before that frontier closes. Once available, the separate invocation gate is:

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

Failure is `RejectedByConstraintSet`, not intrinsic invalidity. Once the
persistent family exists, `K` remains an independent canonical
MappingConstraintSet Artifact. Its exact ArtifactIdentity and the admission
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

`M` is a `mapping.system` `1.0` root. `F` supplies the architecture-only
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
* exactly one ThreadExecutionBinding per root launch and one
  GraphExecutionBinding per reachable static graph launch in each root
  context;
* total, single-valued, well-typed `B_thread` and `B_graph` relations over
  Dataflow-owned may-domains, including exact default-complement rules;
* agreement between every selected SpatialMapping target and the AccCore
  selected by its parent thread binding;
* derivation of exactly one
  `InstructionCoreContextRef = (AccCoreOccurrenceRef, 0)` for each selected
  AccCore, with no competing target in InstructionCore ResourceUse;
* exact resolution of every InstructionCore use site to a Fabric-owned atomic
  `UsePattern`, including its initial state, capacity, requester order, grant
  contract, typed demand, activation, and release;
* exactly one ServiceRealization per derived transfer or operation-service
  obligation;
* reachable execution contexts, complete plan selection, valid service
  targets, canonical service legs, flat route-tree continuity, multicast
  ownership, and physical refinements;
* complete System ResourceUse ownership, exact `ServicePlanElementRef`
  resolution where applicable, use-pattern resolution, atomic
  multi-ResourceState claims, relative activation and release, typed demand,
  capacity, exact grant-policy refinement, and sharing assignments;
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
`Verified`. Once the persistent-family frontier permits System PnR
finalization, only `Verified` is eligible to proceed to it.

## System Constraint Admission

System constraint admission is conditional on the
`MappingConstraintSet Persistent Family Frontier` owned by
`docs/spec-pnr.md`; this document defines no competing frontier contract. No
concrete `SystemMappingConstraintAdmission` invocation or System PnR
finalization exists before that frontier closes. Once available, the search
domain, resolved config, constraint-set ArtifactIdentity, and admission result
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

* exact schema profile, UpstreamArtifactBinding, predecessor, and import
  coupling;
* authoring-order invariance of canonical bytes and ArtifactIdentity;
* foreign, wrong-kind, wrong-owner, unknown-field, duplicate-key, and
  noncanonical-ID rejection;
* TechMapping closed coverage, ordered parameterized capability relations,
  multiple results, variadic ports, and exact memory internal-edge witnesses;
* Spatial record totality, mixed identity rules, RouteTree arborescence,
  route-wide widening acceptance, narrowing and tag-borrow rejection,
  MemoryBinding and exposure closure, ResourceUse, Tags, and progress;
* separation of Spatial base validity from conditional exact `K` admission,
  including rejection before persistent-family closure;
* exact equality of the System import table and normalized `B_graph` range,
  including a legal InstructionCore-only empty table;
* System relation totality, parent-AccCore agreement, service continuity,
  imported-use occurrence qualification, capacity, and positive and negative
  progress anchors; and
* separation of System base verification, exact `K` admission, and Evaluation
  quality gates.

Tests must not preserve printer whitespace, diagnostic prose, C++ container
layout, native cache shape, optional-field Cartesian products, retired
record models, placeholder profiles, generic bags, runtime/deployment payload
round trips, or implementation-specific verifier decomposition.
