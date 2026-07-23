# Structured Compiler Frontend

This document owns Loom's mechanical LLVM-to-SCF boundary and the Structured
Program Candidate optimization contract. The SCF-stage name describes the
primary structured optimization surface; candidates may contain `func`,
`arith`, `math`, `ub`, `llvm`, `cf`, `memref`, `scf`, vector, and other
standard dialects when their semantics are preserved.

The complete frontend path is:

```text
LLVM IR
  -> LLVM-dialect and raised standard MLIR
  -> initial Structured Program Candidate S0
  -> structured optimization and DSE
  -> selected Structured Program Candidate Sn
  -> mechanical SCF-to-Dataflow lowering
  -> initial Canonical Dataflow Program D0
  -> Dataflow-only optimization and DSE
  -> selected Canonical Dataflow Program D*
```

## Mechanical Raising

Raising to S0 recovers and normalizes existing semantics. It may structurize a
CFG, recover counted loops, canonicalize types, and derive exact target facts
needed to interpret LLVM IR. It does not select a performance-distinct
schedule, parallel decomposition, vector shape, memory movement, reduction,
or InstructionCore/SpatialCore ownership boundary.

S0 has no implicit ownership meaning:

* `func.func` is a callable and ABI unit, not a core assignment;
* a recovered serial loop remains serial until a typed decision transforms it;
* source hints and provenance are analysis inputs, not committed choices; and
* function, loop, local-memory, or source-region shape never commits
  acceleration by itself.

Mechanical raising may require analysis and may reject an input that cannot be
represented exactly. Mechanical means that no performance choice is hidden in
the derivation, not that the implementation is trivial.

### Canonical Compute Spelling

Mechanical raising uses the standard `arith` or `math` operation schema when
it exactly represents an LLVM computation. Ordinary LLVM add, subtract,
multiply, integer or floating comparison, select, casts, and other semantic
aliases do not survive merely because they originated in LLVM IR. This gives
Canonical Dataflow one operation-schema identity for one basic computation and
prevents Fabric capability registries from listing dialect aliases.

An LLVM-dialect compute intrinsic may remain only when no exact standard MLIR
operation represents it and it later satisfies the canonical actor contract.
Target-specific intrinsics should be normalized to target-neutral scalar or
vector operations when such a representation exists.

FMA normalization is semantic rather than name based. An exact fused LLVM FMA
may become `math.fma`. An operation whose contract permits or requires a
non-fused multiply followed by add becomes the explicit `arith.mulf` then
`arith.addf` graph. In particular, an `fmuladd` spelling alone never proves
fused semantics.

## StructuredProgramCandidate

S0 and every transformed S1 through Sn belong to one immutable
`StructuredProgramCandidate` Artifact family. There are no separate initial,
optimized, selected, or complete families and no persistent state bits with
those names. Selection is an exact Artifact reference in DSE and invocation
lineage.

Hot search may use an ephemeral region-local draft or delta. A value entering
lineage, cache, an InvocationManifest, or the Artifact Store is always a full
immutable program snapshot.

### Parent-Local References

A typed structured decision refers to an entity only within its exact parent:

```text
StructuredEntityRef {
  parent_candidate_id
  entity_kind
  canonical_parent_local_ordinal
}
```

The child is reanalyzed and receives new local references. Reusing a parent
reference across candidates fails. Source locations, textual names, printer
positions, permanent UUIDs, and `origin` attributes are not entity identity.
Split, clone, fusion, and unroll relations are represented by the typed
parent-to-child decision and the two candidate identities, not by a parallel
entity-lineage database.

### Finalizability

Finalizability is a derived gate:

```text
finalizable(candidate, lowering_semantics, relevant_lowering_config)
```

It proves only that the candidate can be mechanically lowered to a valid
Canonical Dataflow Program. It is not a QoR claim and does not require any
optional optimization. The gate checks explicit and structurally legal
AccCore/SpatialCore ownership, supported `loom.spatial_region` contents,
complete value/stream/memory boundaries, absence of unresolved plans or
target-specific software annotations, and constructible memory/event
networks.

Unknown aliasing can induce conservative order. An operation that cannot be
expressed conservatively remains on the InstructionCore or makes that
candidate non-finalizable. The result is never stored as a boolean because a
lowering-semantic or relevant-config change can change it.

## Derived Analysis Views

`AnalysisView` is a descriptive term for typed in-process derived data, not an
Artifact, base class, registry, or second IR. The minimum dependency chain is:

```text
StructuredProgramCandidate
  -> IterationDomainView
  -> MemoryAccessView
  -> DependenceView
  -> ReductionView
  -> FootprintReuseView
```

The views own these facts:

* `IterationDomainView`: iteration points, bounds, and structured control
  domains;
* `MemoryAccessView`: address/access relations, effects, and may-alias facts;
* `DependenceView`: data, control, and memory dependences derived from program
  order, domains, and accesses;
* `ReductionView`: recurrence, combiner, identity, and explicit algebraic
  policy required for legal reordering; and
* `FootprintReuseView`: exact symbolic sets, bounds, and reuse relations.

Predicted cache misses, traffic, bandwidth, latency, resource pressure, and
QoR are Evaluation observations, not analysis truth. A capability projection
from exact Fabric can filter a domain but does not alter software legality or
enter candidate IR.

An analysis cache key contains the candidate identity, analyzer semantic
identity and version, analysis config, and every auxiliary subject that
participates in derivation. The cache is removable and must not affect results.
A typed transform decision records normalized parameters and exact
parent/child lineage, not an analysis snapshot or copied proof.

## Transformation Algebra

Loom reuses existing LLVM/MLIR operations, interfaces, analyses, and utilities
before adding implementation. Preferred foundations include SCF, Affine,
Linalg, Vector, Presburger, Transform, Bufferization, and SparseTensor.
Transform IR, ISL, and other upstream forms may be ephemeral materialization
tools; they do not become a persistent Schedule IR or second candidate
authority. Loom adds only semantics that upstream cannot express.

For each statement domain `D_s`, a schedule is conceptually a lexicographic
map `theta_s : D_s -> Q`. Typed decisions materialize tiling, strip-mining,
interchange, skewing, fusion, fission, distribution, and affine/polyhedral
schedules into child IR. A polyhedral decision is legal only for an exactly
representable Presburger region; irregular or sparse control is not forced
into an affine model.

For a scheduled dimension, execution decomposition is:

```text
q_d = thread_base_d(logical_coordinate)
    + temporal_d * (spatial_width_d * vector_width_d)
    + spatial_lane_d * vector_width_d
    + vector_lane_d
```

Factor one and an empty projection are ordinary endpoints. The decomposition
does not select a physical AccCore. Jam is the materialized property that
spatial replicas share inner temporal control, invariants, and memory
schedule; it is not a boolean IR flag. Arbitrary-rank vector semantics, masks,
lane order, memory forms, tails, and reduction constraints are owned by
`docs/spec-dataflow-vectorization.md` and use standard ranked vector types.

Each transform proves its own preconditions from derived views. Unknown facts
are not legality proofs. Interchange preserves dependence direction;
parallel/vector replication requires independence or an explicit reduction
strategy; jam requires fusible domain, control, effects, and memory order.
Ordered recurrences preserve lexical order unless an explicit algebraic policy
allows reassociation.

### Memory Movement And Overlap

Logical buffers use standard MLIR allocation, subview, layout, copy,
bufferization, promotion, packing, hoisting, and pipelining mechanisms. They
own shape, layout, lifetime, and copy causality, not a physical SRAM, bank, or
`fabric.mem` choice.

Multibuffering uses one count: one is single-buffered, two is double-buffered,
and larger values form a ring. Pipeline stages express logical iteration
offsets, never cycle slots or predicted initiation interval. SCF materializes
prologue, kernel, epilogue, and buffer reuse; mechanical lowering derives
Dataflow concurrency from data and memory dependencies.

SpatialCore-owned `memref.copy` is expanded into structured load/store loops
before canonical lowering. The current canonical Dataflow ISA has no dedicated
`dataflow.transfer`; adding one requires independent software semantics and a
closed Fabric capability rather than a lowering shortcut. Random access or
cross-time reuse remains logical memory. A producer/consumer relation becomes
a channel only when an ordered SPSC schedule is proved.

## Transactional Decisions

One lineage edge is one typed semantic transform. Compound operations that
upstream implements atomically, such as unroll-and-jam, may be one decision;
other compositions form an immutable candidate chain. Loom does not define a
generic decision bundle, OptimizationPlan object, or transform DSL.

Applying a decision:

1. resolves parent-local references on the exact parent;
2. recomputes required analysis views;
3. clones a working IR;
4. invokes the upstream transform or smallest Loom extension;
5. validates the candidate;
6. runs fixed deterministic canonicalization; and
7. publishes the child and lineage atomically.

Failure exposes no partial child. Equal canonical semantic bytes deduplicate by
ArtifactIdentity. A no-op such as factor one produces no new candidate.

## Exact Fabric Target

Before frontend work begins, the invocation resolves one exact immutable,
fully elaborated Fabric artifact. It can come from user Fabric MLIR, a builtin
target, or the designated builtin default. All authoring paths converge on one
exact Fabric identity. The same code revision and ResolvedConfig reproduce the
same builtin bytes; changing a default requires an explicit template, version,
or config change.

Frontend use of the target has three disjoint forms:

1. exact ABI, DataLayout, address-width, and target-semantic facts used by
   mechanical translation;
2. a removable typed `FabricCapabilityView` for exact, cheap capability and
   aggregate-bound filtering; and
3. central Evaluation requests for resource pressure, traffic, latency,
   bandwidth, topology bounds, and other predicted quality or feasibility.

A pure-software Evaluator need not include Fabric in its subject. A
hardware-aware Evaluator references the exact Fabric identity. Neither a view
nor Evidence enters software IR or copies Fabric capability authority.

Frontend filtering may prove absence of a capability, aggregate insufficiency,
arbitrary-topology disconnection, or a cut/bandwidth lower bound. Only proved
impossibility is a hard prune. A positive or unknown result is not proof of
mappability. The frontend never creates a realization, selects an occurrence,
route, tag, buffer, or configuration, calls a Mapping solver, or emits a
Mapping Artifact. Central DSE may explicitly promote a small survivor set to
Mapping and feed resulting Evidence into a later compilation iteration.

## Compilation DSE

Compilation DSE uses the central finite SSA-like Generate/Promote plan. It does
not create a frontend optimizer controller or enumerate one global Cartesian
product. Four typed generator families own candidate production:

```text
Schedule
ExecutionShape
MemoryCommunication
Ownership
```

They cover schedule/polyhedral transformations; spatial/vector/unroll/jam and
reduction shape; layout/staging/multibuffer/pipeline/channel choices; and
InstructionCore/SpatialCore plus thread-domain ownership. The groups are
generator capabilities, not persistent IR families. A resolved DSE policy may
interleave them and revisit a family.

Each Generate node consumes complete immutable candidates, exact Fabric, and
descriptor-declared analysis or Evidence projections. The transform scope is
the smallest structured ancestor closed over every data, control, memory,
channel, and ownership dependence that the decision may change. Independent
regions are explored separately only when analysis proves no cross-region
coupling.

Pruning has exactly three authorities:

1. semantic legality from dependence, alias, effect, shape, and reduction
   proof;
2. exact target admissibility from Fabric and a derived capability view; and
3. Evaluation/QoR selection through Evidence and central Promote gates,
   `AllPassing`, `TopK`, or `Pareto`.

A heuristic score is not legality, and unknown is not unmappable. Generators
consume only the typed Evidence projections declared by the central plan; they
do not scan an Artifact Store, choose the latest result, or accept free-form
backend advice.

## Ownership Materialization And Handoff

The Structured Program Candidate lineage is the sole owner of schedule, loop,
vector, parallel, reduction, fixed logical-domain width `P[]`, aggregation
materialization, and execution ownership choices. Any choice that can alter
candidate semantics or performance must be materialized in Sn before handoff.
A residual unsupported or unmaterialized aggregation or reduction form makes
Sn non-finalizable; Part 3 neither drops it nor selects a fallback.

Sn uses two ownership carriers:

* `dataflow.thread` is the selected AccCore carrier. Its body remains the
  InstructionCore stored-program and structured-control surface.
* compiler-internal `loom.spatial_region`, terminated by
  `loom.spatial_yield`, marks one selected SpatialCore boundary inside a
  thread. It is semantically transparent to inlining.

`loom.spatial_region` has no launch, firing, Mapping, runtime-state, or
hardware-configuration semantics. Mechanical Part 3 lowering consumes Sn
without changing any structured choice, atomically publishes D0, and removes
every temporary spatial region and residual imperative control. Unsupported
conversion fails without publishing partial canonical IR.

D0 may then undergo typed Dataflow-only rewrites to produce D*. `D0 == D*` is
legal. D* remains target-independent software and contains no Mapping facts.

## Verification Anchors

Tests protect only stable contracts:

* a cross-candidate entity reference is rejected;
* candidate, analyzer, config, or auxiliary-subject changes invalidate a
  derived analysis cache;
* unknown aliasing prevents an unsafe reorder but permits conservative order;
* reduction reassociation requires an explicit recognized algebraic policy;
* a transform is atomic and deterministic, and a no-op emits no child;
* finalizability rejects unsupported SpatialCore surface without publishing
  partial D0; and
* frontend capability filtering never invokes Mapping or writes hardware facts
  into software IR.

Tests do not build matrices over loop shapes, pass order, cache backends,
transform combinations, or Fabric targets.

## References

* `docs/spec-compiler-part-1-source.md`
* `docs/spec-compiler-part-3-dfg.md`
* `docs/spec-dataflow-vectorization.md`
* `docs/spec-dse-feedback.md`
* `docs/spec-fabric-system-adg.md`
