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

* an imported `llvm.func` remains the sole callable and ABI envelope for its
  LLVM function and is not a core assignment;
* `func.func` is reserved for genuinely standard-MLIR-native callables and
  helpers, and never mirrors an imported LLVM ABI;
* a recovered serial loop remains serial until a typed decision transforms it;
* source hints and provenance are analysis inputs, not committed choices; and
* function, loop, local-memory, or source-region shape never commits
  acceleration by itself.

Mechanical raising may require analysis and may reject an input that cannot be
represented exactly. Mechanical means that no performance choice is hidden in
the derivation, not that the implementation is trivial.

### Callable Envelope And CFG Structuring

The final linked LLVM module enters S0 without replacing its LLVM callable
envelopes. The LLVM dialect operation remains the sole owner of linkage,
calling convention, COMDAT, personality, argument and result attributes,
memory effects, target features, floating-point environment, and every other
LLVM ABI fact.

Mechanical CFG recovery operates on callable regions rather than requiring
conversion to `func.func`. For an imported LLVM function, Loom converts LLVM
branch structure to exact `cf` structure where required, invokes the upstream
region-level CFG-to-SCF transformation, and uses an LLVM-compatible adapter
for return and unreachable behavior. Pass wrappers that participate in this
pipeline operate on `FunctionOpInterface`, a callable region, or the upstream
region utility. They must not copy a function into another dialect to obtain a
particular pass wrapper.

CFG structuring may preserve multiple PHI lanes for one recurrence. Mechanical
raising removes a duplicate `scf.while` lane only when it has the same SSA
initial value and the same SSA `scf.condition` value as an earlier lane, and
both lanes feed those values back through their respective identity
`scf.yield` operands. The surviving lane replaces both loop arguments and both
results. No value numbering, pointer-alias assumption, or performance choice
participates in this exact quotient.

The lift-owned exit scaffold may publish a `scf.while` result through a value
defined outside and dominating the loop. Mechanical raising projects that
result directly to the exact published SSA value while retaining the loop and
its control effects. This rule applies equally to defined, poison, and undef
values; it never substitutes an ordinary constant for an exceptional value.

### Mechanical Disposition

Every input construct has exactly one mechanical disposition:

* **Preserve.** Retain the original operation and its complete semantics.
* **Normalize exactly.** Replace it only when types, attributes, enclosing
  semantic context, and target facts prove exact equivalence.
* **Project provenance, then erase.** Preserve standard locations and imported
  debug provenance before erasing a carrier proven to have no program
  semantics.
* **Fail closed.** Reject the selected candidate at the first boundary that
  requires a narrower surface and cannot represent the construct exactly.

LLVM profile and branch-weight metadata is preserved while its owning control
operation remains in S0. An unselected or InstructionCore-owned weighted
branch cannot make ordinary module compilation fail. A whole Structured
Program Candidate is published atomically; the candidate whose selected
Spatial region requires structured lowering and cannot preserve or consume the
weight is non-finalizable, while independently derived ownership candidates
remain eligible. Loom does not copy weights into a second metadata schema.

LLVM loop metadata has a loop owner only when its carrier terminator closes a
backedge to one exact dominating loop header. Mechanical structuring moves that
hint to the recovered loop. Metadata on a terminator with no such backedge is
an orphan under the LLVM loop contract: successful structuring removes the
orphan carrier, preserves the corresponding analysis fact as unknown, and does
not guess a loop owner. A carrier that can close backedges to multiple headers
remains unstructured because its owner is ambiguous.

InstructionCore ownership is not a fifth disposition and is never an implicit
raising fallback. Ownership belongs to a Structured Program Candidate. If a
selected `loom.spatial_region` contains an unsupported construct, that whole
candidate is non-finalizable and no partial D0 is published. DSE may construct
a different candidate with a different ownership boundary, but the mechanical
lowerer does not silently move code.

### Canonical Compute Spelling

Mechanical raising uses the standard `arith` or `math` operation schema when
it exactly represents an LLVM computation under the complete operation type,
operation attributes, overflow and exact flags, fast-math policy, rounding
mode, enclosing function floating-point environment, and DataLayout. An
apparent LLVM alias does not normalize merely because it has a familiar
opcode. Exact normalization gives Canonical Dataflow one operation-schema
identity for one basic computation and prevents Fabric capability registries
from listing dialect aliases.

An LLVM-dialect compute intrinsic may remain only when no exact standard MLIR
operation represents it and it later satisfies the canonical actor contract.
Target-specific intrinsics should be normalized to target-neutral scalar or
vector operations when such a representation exists. Otherwise, preserving
the registered LLVM operation is preferable to weakening its semantics.
The signed and unsigned saturating add and subtract intrinsics are such
registered operations: no standard `arith` operation states their exact
saturation semantics, so mechanical raising preserves
`llvm.intr.{s,u}{add,sub}.sat` rather than expanding a private clamp graph.

The LLVM dialect `passthrough` function attribute is an importer-owned lossless
container, not a floating-point-environment authority. Mechanical raising uses
one closed classifier owned by the exact-spelling projection. Typed LLVM
floating environment attributes, `strictfp`, incompatible exception policy,
and unknown string attributes block standard spelling. LLVM enum function
attributes and explicitly classified code-generation-only strings do not.
Clang's default `no-trapping-math=true` is compatible with the ordinary
non-constrained floating operation spelling; any other value fails closed.
Ownership materialization and Dataflow lowering never reinterpret this
classification.

FMA normalization is semantic rather than name based. An exact fused LLVM FMA
becomes `math.fma`. `llvm.intr.fmuladd` remains unchanged in S0 until one typed
`ExecutionShape` decision materializes either `Fused` or
`Split(arith.mulf, arith.addf)` under the exact floating environment and
fast-math contract. That decision is candidate lineage and may be evaluated as
a performance choice; target code generation cannot choose it implicitly.
After materialization, no `fmuladd` operation may remain in a finalizable Sn or
be registered as a Canonical Dataflow actor.

Ordinary calls are never expanded from a symbol-name or arity match. A call
remains a call, becomes visible through LLVM linking or LTO, or is handled by a
future explicit typed and versioned library model. Recognizing a source
library spelling is not a semantic proof. The first version deliberately has
no such opaque-library model; it is reopened only when a required library body
cannot be exposed through the ordinary final LLVM link.

Mechanical raising may devirtualize an indirect call only when pinned LLVM
interprocedural constant propagation proves one exact direct `Function`
target in the complete linked module and the call and target function types
are identical. The proof runs on a disposable clone. Raising projects only
the proven called operand back to the input module; constant folding, dead-code
removal, branch simplification, global-memory rewriting, function cloning, and
all other optimizer effects from the proof clone are discarded. Unknown,
escaped, type-adjusted, or path-dependent callees remain indirect calls. This
is call-kind canonicalization, not library modeling, inlining, ownership, or a
performance decision.

When one defined dispatcher is called at different sites with different exact
function constants, no module-global target exists. A separate deterministic
call-site canonicalization may clone that dispatcher once per distinct exact
callback binding and redirect only the matching direct call sites. A binding
is admitted only when the actual value is an exact defined `Function`, the
corresponding formal is used directly as an indirect callee, and the target
function type and calling convention equal every such call. The clone replaces
only the bound formal uses; the original dispatcher remains the authority for
unknown or escaped callback values. This canonicalization neither picks one
target globally nor inlines the callback body, chooses Spatial ownership, or
creates a library model. Target compilation and a source-backed native oracle
must consume the same production canonicalizer so their finite direct-call
lineage cannot diverge.

LLVM leading- and trailing-zero count intrinsics with
`is_zero_poison = false` normalize mechanically to `math.ctlz` and
`math.cttz`. The poison-flagged forms retain their LLVM spelling and project
that flag through the registered typed semantic case; the standard Math ops
cannot carry the poison-on-zero contract.

### Exceptional Values And Floating Semantics

Mechanical raising preserves defined values, poison, undef, and `freeze`
without replacing them with ordinary constants or diagnostics. Exact integers
and supported floating values use arbitrary-precision semantic
representations; host `int64_t` and `double` are not semantic authorities.
Fixed vectors may carry exceptional state per lane. A normalization is legal
only when it preserves the operation-specific rules for propagation,
non-observation, and undefined behavior.

Scalable vectors are legal S0 values. Before a region containing one can
finalize as a SpatialRegion, a typed structured transform must materialize its
semantics as fixed-width chunks, loops, and masks or tails. If it cannot, the
selected candidate is non-finalizable. A scalable vector is never presented
to the fixed-ranked Canonical Dataflow contract as though its runtime
`vscale` were a constant.

## StructuredProgramCandidate

S0 and every transformed S1 through Sn belong to one immutable
`StructuredProgramCandidate` Artifact family. There are no separate initial,
optimized, selected, or complete families and no persistent state bits with
those names. Selection is an exact Artifact reference in DSE and invocation
lineage.

Hot search may use an ephemeral region-local draft or delta. A value entering
lineage, cache, an InvocationManifest, or the Artifact Store is always a full
immutable program snapshot.

### Artifact Schema And Canonical Bytes

The Artifact family is fixed as:

```text
loom.structured_program 1.0
```

Its semantic root is exactly one `builtin.module` containing the complete
mixed-dialect program snapshot. There is no wrapper operation, stage-state
record, optimization-state flag, or direct Artifact dependency table in schema
1.0. The linked LLVM and MLIR operations inside that module own the program,
ABI, target triple, DataLayout, control, memory, and selected structured
decisions. Candidate lineage and the exact Fabric used to evaluate a candidate
belong to their existing owners and do not become Structured Program fields.

The Structured Program family owns one canonical writer. It operates on a
private clone and:

1. verifies the complete mixed-dialect module and every registered operation;
2. rejects unresolved source-provider metadata, transform plans, placeholders,
   and target-specific software annotations;
3. removes locations, debug provenance, consumed source hints, visual metadata,
   derived analysis state, and any author-supplied identity carrier;
4. resolves symbol uses, normalizes private symbols, SSA names, block labels,
   attribute ordering, and nonsemantic symbol-table member order;
5. preserves externally visible symbol spelling, ABI facts, ordered
   stored-program operations, region semantics, and every registered semantic
   field; and
6. emits deterministic MLIR bytecode using the schema-owned writer and the
   pinned MLIR dialect/version set.

The exact canonical semantic bytes passed to the Common Artifact finalizer are:

```text
bytes("loom.structured_program.semantic.v1\0")
|| u64be(canonical_mlir_bytecode_length)
|| canonical_mlir_bytecode
```

The family writer, not a generic MLIR printer flag or arbitrary bytecode
emission path, owns these bytes. A writer change that changes canonical bytes
requires a compatible schema-minor append or an incompatible schema-major
change according to the ordinary Artifact rules. The specification fixes the
canonical result, not the symbol-normalization or graph-labeling algorithm.

Finalization is failure-atomic. After canonical bytes are formed, the family
parses them in a fresh context with the exact registered dialect set,
reconstructs a sealed read-only `StructuredProgramCandidateView`, runs the
complete verifier again, re-encodes through the same writer, and requires exact
byte equality before the Common single-object `put`. Import performs the same
parse, verify, and re-encode checks and never repairs stale or noncanonical
content. Malformed content is `Invalid`; an exact, valid candidate whose
selected SpatialRegion cannot be mechanically lowered is simply
non-finalizable and is not silently reassigned to the InstructionCore.

Source-provider candidate hints are transient analysis inputs. The frontend
must consume or explicitly discard them before Artifact finalization. The raw
hint never enters canonical bytes or identity. A hint can affect identity only
indirectly when a typed decision materializes different semantic IR in the
resulting candidate. Two candidates with identical canonical semantic content
deduplicate even if one derivation observed a hint and the other did not.

### Parent-Local References

A typed structured decision refers to an entity only within its exact parent:

```text
StructuredEntityRef {
  parent_candidate_id
  entity_kind
  canonical_parent_local_ordinal
}
```

Schema 1.0 keeps this one reference shape. `entity_kind` is a closed
Structured Program family ordinal with exactly these structural categories:

```text
StructuredEntityKind = Operation | Region | Block | Value
```

The stable schema-1.0 ordinals are `Operation = 0`, `Region = 1`, `Block = 2`,
and `Value = 3`. The standalone comparison and persistence wire is:

```text
parent_candidate_id[32]
|| u32be(entity_kind)
|| u64be(canonical_parent_local_ordinal)
```

The static reference type fixes the exact `loom.structured_program` schema, so
the field does not repeat a free schema string. A containing record that also
owns the parent candidate must require exact parent equality; it cannot omit or
reinterpret the reference's parent field.

The finalizer assigns a dense canonical ordinal independently within each
category. A Value is an existing block argument or operation result in the
canonical module; the resolver derives which variant and owner it has rather
than storing another public value-reference union. The referenced operation's
actual MLIR operation name, traits, interfaces, and semantic fields remain
owned by that operation schema. `entity_kind` does not duplicate loop kinds,
memory kinds, transformation kinds, or operation names.

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

Unknown aliasing can induce conservative order. Once a candidate has selected
a SpatialRegion, any contained operation that cannot be expressed
conservatively makes that candidate non-finalizable. A separate candidate may
choose InstructionCore ownership, but finalization never changes ownership.
The result is never stored as a boolean because a lowering-semantic or
relevant-config change can change it.

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
An active DSE invocation may retain the exact parent reference, scope, and
typed decision for a selected child so downstream validation can replay that
derivation. This invocation-local lineage is neither candidate identity nor a
persistent Structured Program field.

### Artifact Anchors

Only stable family boundaries require dedicated tests:

* private symbol, SSA, block-label, source-location, debug-provenance, and
  consumed-hint changes preserve canonical bytes and identity;
* an operation, semantic field, ABI fact, ordered effect, ownership boundary,
  or selected structured decision change changes canonical bytes and identity;
* a wrong-parent, wrong-kind, stale, or out-of-range `StructuredEntityRef` is
  rejected;
* finalizer output imports and re-encodes byte-for-byte, while malformed or
  noncanonical bytes publish nothing; and
* an unsupported selected SpatialRegion is non-finalizable without changing
  the original candidate or silently moving its operations.

Tests must not build operation, dialect, loop-shape, transform-order, metadata,
or printer-option matrices. They call the production writer, importer, and
reference resolver rather than copying canonicalization formulas.

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

The compiler invocation nevertheless always has one exact Fabric target. A
frontend pass cannot discover another target through ambient process state,
global registries, or an implicit default. A hardware-sensitive legality,
filtering, or quality decision consumes either the typed capability view
derived from that exact target or an EvaluationRequest whose case signature
includes that exact Fabric Artifact.

Frontend filtering may prove absence of a capability, aggregate insufficiency,
arbitrary-topology disconnection, or a cut/bandwidth lower bound. Only proved
impossibility is a hard prune. A positive or unknown result is not proof of
mappability. The frontend never creates a realization, selects an occurrence,
route, tag, buffer, or configuration, calls a Mapping solver, or emits a
Mapping Artifact. Central DSE may explicitly promote a small survivor set to
Mapping and feed resulting Evidence into a later compilation iteration.

Promotion first mechanically derives the exact canonical Dataflow Program
`D_i` from the selected Structured Program Candidate `S_i`. The resulting
Mapping `M_i` is permanently coupled to `D_i` and the exact Fabric `F`, and the
high-fidelity Evaluation case names that exact tuple. A structured candidate
generator may consume a declared typed projection of the resulting Evidence
to produce a new immutable `S_j`. It cannot inspect Mapping-private records,
mutate `S_i`, or reinterpret `M_i` as a Mapping of `S_j`.

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

### Complete Candidate Dispositions

An Ownership Generate invocation enumerates one finite scope-local domain in
canonical Structured operation order. Every defined callable considered as a
whole-callable scope appears exactly once. A callable that cannot yet be
materialized records one scope coordinate with no decision and a typed
`NonFinalizable` disposition. External declarations and operations that are
not ownership scopes are not candidate attempts. Every accepted scope then
enumerates its typed decision domain in owner-defined canonical order.

Each concrete decision records exactly one of:

```text
exact child Structured Program ArtifactRootReference
NonFinalizable(reason)
ExactFabricInadmissible(reason)
```

Unexpected invocation, ArtifactStore, provider, or implementation failures
abort the invocation and cannot be relabeled as candidate rejection. Parallel
execution may evaluate coordinates in any physical order, but the completed
disposition sequence retains canonical domain order. Multiple coordinates may
produce the same child identity; central candidate Evaluation deduplicates by
ArtifactIdentity while invocation provenance retains each valid derivation.

This sequence is invocation-local work accounting and the mechanical source
for the `InvocationManifest` candidate records defined by the central DSE
specification. It is not an Artifact, a graph-free status, or a second lineage
authority. A completed graph-free result requires this total sequence plus an
explicit stored-program `CandidateDecision` backed by workload-aware Evidence;
silently skipped scopes or decisions make that conclusion incomplete.

## Ownership Materialization And Handoff

The Structured Program Candidate lineage is the sole owner of schedule, loop,
vector, parallel, reduction, fixed logical-domain width `P[]`, aggregation
materialization, and execution ownership choices. Any choice that can alter
candidate semantics or performance must be materialized in Sn before handoff.
A residual unsupported or unmaterialized aggregation or reduction form makes
Sn non-finalizable; Part 3 neither drops it nor selects a fallback.

Every selected Spatial region containing a dynamic LLVM GEP also materializes
one explicit canonical address-index width in its Structured candidate. This
rule is identical for whole-callable and operation-owned regions. Candidate
generation may enumerate widths admitted by the exact Fabric, but neither the
lowerer nor ambient process configuration selects one. A source integer wider
than the selected width is narrowed only when its complete signed value domain
is proven to fit; otherwise that candidate is non-finalizable. The resulting
fixed index DataLayout entry and explicit casts are ordinary candidate
semantics and are the sole input consumed by mechanical Dataflow lowering.

A loop-carried raw pointer is not a second memory-capability recurrence. Before
the selected region finalizes, a proven constant-stride pointer induction is
materialized as one loop-invariant base capability plus a fixed-width integer
element-offset recurrence. Constant-stride means invariant for that loop; the
stride may be a runtime value defined outside the loop. The proof must identify
one finite counted-loop domain, one exact access element type per pointer lane,
an integral element stride with a finite value range, and an accumulated offset
that fits the selected signed index width.
The transformed GEP is derived mechanically from the base and offset. If any
of those facts is unknown, the candidate is non-finalizable; Part 3 never
places the raw pointer in `dataflow.carry` or invents a dynamic capability.

Sn uses two ownership carriers:

* `dataflow.thread` is the selected AccCore carrier. Its body remains the
  InstructionCore stored-program and structured-control surface.
* compiler-internal `loom.spatial_region`, terminated by
  `loom.spatial_yield`, marks one selected SpatialCore boundary inside a
  thread. It is semantically transparent to inlining.

Every ownership cut is the smallest structured scope closed over the data,
control, memory, channel, and ownership dependences changed by the decision.
Its dynamic live-ins are explicit `loom.spatial_region` inputs and its value
live-outs are explicit region results. When materialization introduces a new
rank-zero thread launch inside an existing LLVM callable, an ordinary value
live-out crosses that thread boundary through caller-owned result storage: the
caller allocates the slot, passes it as a thread-only input, waits for thread
completion, and then loads the value. The selected graph returns the value to
InstructionCore code inside the thread, which stores it into that slot. The
slot is not a Spatial-region memory input, and `dataflow.thread` gains no data
result. Memory capabilities and channels continue to use their existing typed
boundary relations and are never encoded as ordinary result-slot values.

`loom.spatial_region` has no launch, firing, Mapping, runtime-state, or
hardware-configuration semantics. Mechanical Part 3 lowering consumes Sn
without changing any structured choice, atomically publishes D0, and removes
every temporary spatial region and residual imperative control. Unsupported
conversion fails without publishing partial canonical IR.

D0 may then undergo typed Dataflow-only rewrites to produce D*. `D0 == D*` is
legal. D* remains target-independent software and contains no Mapping facts.

## Verification Anchors

Tests protect only stable contracts:

* imported LLVM function ABI facts remain owned by the original `llvm.func`;
* exact aliases normalize while non-equivalent LLVM operations remain
  preserved;
* ordinary calls are not expanded from symbol names or arity;
* poison, undef, and floating-point policy survive mechanical raising;
* scalable vectors either materialize to fixed structured semantics before a
  selected SpatialRegion finalizes or make that candidate non-finalizable;
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
