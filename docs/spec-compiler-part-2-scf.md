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
The overloaded `llvm.fptosi.sat` and `llvm.fptoui.sat` intrinsics follow the
same rule. They remain exact `llvm.call_intrinsic` operations because no
standard operation states their NaN-to-zero and range-clamping semantics.
Their full overloaded spelling and function type must remain unchanged until
the Canonical Dataflow actor projection validates them against LLVM's
intrinsic registry.

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
a performance choice; target code generation cannot choose it implicitly. The
Ownership generator selects the Spatial region but does not own this decision.
The ExecutionShape generator resolves it before Schedule or Dataflow lowering
may consume the candidate. After materialization, no `fmuladd` operation may
remain in a finalizable Sn or be registered as a Canonical Dataflow actor.

The scalar `math` operations assigned to the registered `ScalarMath*`
implementation families use one closed accuracy domain:

```text
SpecialMathAccuracyTier =
    CorrectlyRounded
  | Max1Ulp
  | Max2Ulp
  | Max4Ulp
```

`CorrectlyRounded` means the exact mathematical result rounded once to the
destination floating format under the actor's rounding contract; it does not
mean that an irrational real number is represented exactly. For a defined
finite numerical result, `MaxNUlp` permits at most `N` adjacent representable
destination-format values between the produced result and that
correctly-rounded reference. Exceptional values, subnormal handling, signed
zero, and other fast-math permissions remain owned by the native operation
schema and its floating behavior contract. An accuracy tier neither grants nor
copies those permissions.

The strength order is:

```text
CorrectlyRounded < Max1Ulp < Max2Ulp < Max4Ulp
```

where a lower value is a stronger guarantee and satisfies any higher accepted
maximum. The selected tier is the exact `loom.special_math_accuracy`
dialect attribute on the Structured operation and later belongs to its closed
canonical actor projection. The attribute's codec and validation are owned by
this domain. It is not inferred from an operation name, a backend recipe, or a
target library.

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

The current `StructuredEntityRef` contract keeps this one reference shape.
`entity_kind` is a closed
Structured Program family ordinal with exactly these structural categories:

```text
StructuredEntityKind = Operation | Region | Block | Value
```

The stable current ordinals are `Operation = 0`, `Region = 1`, `Block = 2`,
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
a channel only when one total ordered producer sequence and one or more total
ordered consumer sequences are proved.

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
product. Five typed generator families own candidate production:

```text
Schedule
ExecutionShape
SpecialMathAccuracy
MemoryCommunication
Ownership
```

They cover schedule/polyhedral transformations; spatial/vector/unroll/jam and
reduction shape; accepted special-math accuracy; layout/staging/multibuffer/
pipeline/channel choices; and InstructionCore/SpatialCore plus thread-domain
ownership. The groups are generator capabilities, not persistent IR families.
A resolved DSE policy may interleave them and revisit a family, but every path
presented to Structured Evaluation or Dataflow lowering must cross
SpecialMathAccuracy after its last Schedule or MemoryCommunication transform.
The current canonical plan is:

```text
Ownership
-> ExecutionShape
-> Schedule
-> MemoryCommunication
-> SpecialMathAccuracy plus D0/exact-Fabric closure
```

This ordering is semantic, not an optimization heuristic. Schedule and memory
communication may remove or replace operations that occur in an intermediate
Structured child, so complete actor admission before those transforms would
mistake a transient representation for the final SpatialCore surface.

Each Generate node consumes complete immutable candidates plus only the exact
Fabric, analysis, or Evidence projections declared by its descriptor. The
transform scope is the smallest structured ancestor closed over every data,
control, memory, channel, and ownership dependence that the decision may
change. Independent regions are explored separately only when analysis proves
no cross-region coupling.

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

### Structured Schedule Generator

The current Structured Schedule generator consumes a finite set of exact
Structured Program references and one exact finalized Fabric. It emits the
input set plus every distinct child obtained by one legal atomic schedule
decision. An empty input set produces an empty output set, so an ordered
central plan remains total when an earlier generator has no candidates. Its
resolved component view contains only the positive `scope_expansion_limit`
owned by the Resolved Configuration View.

Loop scopes are `scf.for` operations in the parent's canonical Structured
operation order. The first `scope_expansion_limit` loop scopes form the finite
Generate domain; later loops are outside that invocation domain. Static tile,
unroll, and unroll-and-jam factors are the sorted proper divisors of the exact
static trip count. Factor one is a no-op and the full trip count is not emitted
by this generator. Dynamic, non-host-representable, prime, and unit trip counts
have no such factor decision in the current contract; other generator
families or later invocations may still transform their enclosing structure.

Unroll is hard-pruned only when exact aggregate Fabric capacity proves the
replicated body impossible. Actor instances are grouped by the canonical
typed OperationSchema projection. For each group, the generator divides the
number of admitted concrete Fabric occurrences by the group's body
multiplicity; the minimum quotient bounds the unroll factor. This projection
does not prove placement, routing, contention freedom, or performance.

Interchange is one adjacent swap of a perfect two-loop nest. Both loops must
have no loop-carried results, inner bounds and step must be invariant to the
outer loop, and the common dependence/effect analysis must prove independent
iterations for both dimensions. Unknown dependence rejects the decision.
Parallelize is one factorless conversion of an `scf.for` without loop-carried
results into `scf.forall`. It uses that same common dependence/effect owner and
requires the loop bounds and step to be invariant at the resulting parallel
scope. Unknown aliasing, calls with unresolved effects, volatile or atomic
effects without an exact supported relation, or an unproved cross-iteration
dependence reject the decision. The child stores the logical parallel domain
in ordinary SCF; it carries no physical coordinate, AccCore binding, placement,
or routing fact. A later ownership decision may retain that domain inside one
Spatial graph or materialize it as a logical `dataflow.thread` domain.
Unroll-and-jam uses the same perfect-nest and independence proof, additionally
requires every nested loop bound and step to be invariant to the selected
outer loop, and obeys the same exact aggregate Fabric-capacity bound as
ordinary unroll. It is one atomic decision implemented by the pinned upstream
SCF utility; the shared inner control and replicated body are materialized in
the child IR and no jam flag is persisted. Arbitrary permutations and compound
non-atomic schedules are composed through immutable lineage rather than
enumerated factorially. Tile, unroll, and unroll-and-jam use the pinned upstream
SCF utilities; interchange preserves the exact loop bounds, comparison
convention, attributes, body, and induction-variable uses while exchanging
dimensions.

Each generated decision resolves its parent-local `StructuredEntityRef`,
clones the complete parent, applies one transform, verifies the result, and
passes it through the sole Structured Program finalizer. Equal children
deduplicate by Artifact identity. No schedule tree, factor table, hidden pass
state, or persisted analysis view exists.

The generator may use exact aggregate Fabric capacity to reject a schedule
decision that is already impossible, but it does not lower the complete child
to D0 or admit every intermediate actor. MemoryCommunication may still remove
or replace part of that actor surface. The terminal SpecialMathAccuracy gate
performs complete D0 actor admission after the last such transform. Thus a
five-lane `dataflow.sync` created by tiling is rejected there when it survives
to the final child and the selected Fabric admits at most four lanes, while an
intermediate pointer operation later consumed by channel promotion is not a
hardware requirement. This is hard-negative capability pruning, not a
placement, routing, contention, or QoR conclusion.

The provider for this behavior has implementation semantic identity
`loom.compiler.structured_schedule.generator.v2`. The historical v1 provider
performed complete D0 admission before downstream Structured transforms and
cannot be reinterpreted as v2.

### Structured ExecutionShape Generator

The current Structured ExecutionShape generator consumes a finite set of exact
Structured Program references. An empty input set produces an empty output
set. A parent with no unresolved selected-Spatial execution-shape choice passes
through unchanged. A parent containing an unresolved, exactly representable
`llvm.intr.fmuladd` emits the canonical pair of complete Structured children:

```text
Fused -> math.fma
Split -> arith.mulf followed by arith.addf
```

One decision applies uniformly to every unresolved `fmuladd` in the selected
Spatial ownership of that complete parent. It never rewrites residual
InstructionCore operations or operations owned by nested callables. This is a
two-element semantic policy domain, not one independent Boolean dimension per
operation. Distinct per-operation combinations are not part of the current
contract.

Each child preserves the exact floating type, fast-math contract, source
location, Ownership lineage, and source-provenance projection. It is verified
and finalized through the sole Structured Program finalizer before publication
to the output set. Schedule and MemoryCommunication may then form further
complete Structured children. The terminal SpecialMathAccuracy generator is
the selected-Spatial semantic-closure gate that first lowers the final complete
candidate to D0 and checks exact concrete Fabric admission. No unresolved
parent, mixed Fused/Split child, hidden backend default, or
target-code-generation choice may cross the ExecutionShape boundary.

The resolved current component view is empty. Worker count and cached typed
candidates are invocation-local execution policy and do not change the finite
semantic domain. A cache entry is keyed by the exact Structured Artifact
reference and must match its already validated canonical bytes; it is removable
and cannot serve as a second candidate authority.

### Structured SpecialMathAccuracy Generator

The SpecialMathAccuracy generator consumes a finite set of exact Structured
Program references and one exact finalized Fabric. It closes every
selected-Spatial scalar `ScalarMath*` operation before mechanical D0
publication. It does not inspect or rewrite residual HostCore or
InstructionCore computation. A special-math operation without the native
approximate-functions permission `afn` has one legal selected tier,
`CorrectlyRounded`; the generator materializes that tier mechanically and
creates no accuracy choice. `afn` authorizes candidate generation over the
finite domain above but does not itself select a tier or imply `Max4Ulp`.

A candidate with no selected-Spatial special-math operation passes through this
generator without a semantic decision, but still crosses the same mechanical
D0 and exact-Fabric admission gate. SpecialMathAccuracy is therefore the one
terminal D0 owner rather than duplicating partial lowering or inventing an
unresolved actor projection in earlier Structured generators. A central plan
may revisit other generator families, but it must invoke this gate again after
the last D0-affecting Structured transform before promotion.

For an `afn` operation, the finite domain is `CorrectlyRounded`, `Max1Ulp`,
`Max2Ulp`, and `Max4Ulp`. A resolved Loom policy or a source-derived typed hint
accepted by that policy may narrow the domain to a stronger maximum, but it
cannot authorize a tier that the source operation forbids. For example, a
`Max2Ulp` bound retains `CorrectlyRounded`, `Max1Ulp`, and `Max2Ulp`. A complete
Generate invocation emits every member of the resulting domain; early budget
termination remains incomplete rather than silently shrinking it. The selected
tier, not the hint, is the persistent semantic fact. Candidate production uses
immutable lineage and never serializes a Cartesian table of per-operation
choices.

Every child preserves the exact operation type, remaining fast-math flags,
floating environment, location, Ownership lineage, and source-provenance
projection. Before publication it passes the sole Structured finalizer,
mechanical D0 lowering, and exact Fabric admission. The concrete Fabric
capability must guarantee a tier no weaker than the actor's selected accepted
maximum. A missing selection, malformed tier, non-`CorrectlyRounded` tier
without `afn`, or absence of an admitted concrete capability makes that
selected-Spatial candidate non-finalizable.

SpecialMathAccuracy is not an ExecutionShape choice. It does not replace an
actor with a different operation graph, and it cannot be delayed to a
Dataflow-to-Dataflow rewrite or backend recipe. The D0-to-D* lineage preserves
the exact selected tier byte for byte.

### Structured MemoryCommunication Generator

The current Structured MemoryCommunication generator consumes a finite set of
exact Structured Program references. It emits every input parent plus each
distinct child obtained by one legal atomic logical memory decision. It
additionally follows
`PromoteOrderedBufferToChannel` children through the same decision owner until
no further promotion is legal or the resolved scope budget is exhausted.
Every lineage edge remains one atomic promotion. Other decision kinds are not
recursively expanded by this generator. An empty input set produces an empty
output set. Its resolved component view contains only the positive
`scope_expansion_limit` owned by the Resolved Configuration View.

The current closed decision catalog and stable ordinals are:

```text
StructuredMemoryCommunicationDecisionKind =
    StageConstantGlobal          = 0
  | PermuteLocalBufferLayout     = 1
  | PipelineStagedLoop           = 2
  | PromoteOrderedBufferToChannel = 3
```

The canonical lineage-payload schema is
`loom.structured_memory_communication.decision.3.0`. Its bytes are:

```text
u32be(kind)
|| StructuredEntityRef(anchor)
|| kind_payload

kind_payload(StageConstantGlobal)        = empty
kind_payload(PermuteLocalBufferLayout)   = u64be(adjacent_storage_position)
kind_payload(PipelineStagedLoop)         = empty
kind_payload(PromoteOrderedBufferToChannel) = empty
```

The anchor is respectively the selected Spatial memory block argument, local
`memref.alloc` result, `scf.for` operation, or fresh temporary allocation
result. The kind determines the anchor category and payload shape; there is no
generic option dictionary, dormant parameter slot, or memory-plan record.
Malformed anchors, unknown kinds, unexpected payload bytes, and every 1.x
decision payload are rejected rather than reinterpreted.

The provider for this catalog, decision schema, and component-view schema has
implementation semantic identity
`loom.compiler.structured_memory_communication.generator.v4`. The existing
`loom.compiler.structured_memory_communication.generator.v1`,
`loom.compiler.structured_memory_communication.generator.v2`, and
`loom.compiler.structured_memory_communication.generator.v3` identities remain
bound to their incompatible historical behavior and cannot be reinterpreted.
Registry, manifest, cache, replay, and lineage validation must therefore
distinguish the providers without a compatibility switch.

Memory-relevant structural scopes are visited in canonical Structured entity
order. Initial parents are visited in canonical input order. Admitted channel
children are appended to a FIFO frontier in canonical decision order and are
deduplicated by Structured Program Artifact identity before expansion. The
first `scope_expansion_limit` scopes across that complete ordered traversal are
inspected. Each inspected scope retains its complete applicable kind and
parameter domain; the limit cannot cut the adjacent-layout choices of one
admitted allocation. Only channel decisions are attempted on a derived
frontier child. Because each such decision eliminates one exact temporary
allocation, this closure is finite without a depth parameter. The two
descriptor-owned work units remain inspected memory scopes and attempted
decisions. Worker count, completion order, exact Fabric capacity, and wall time
cannot change this finite semantic domain.

#### `StageConstantGlobal`

Its scope is an exact memory block argument of one selected
`loom.spatial_region`, identified by the parent-local `StructuredEntityRef`.
The decision exists only when all of the following are proved from the
complete parent:

1. the argument has a statically shaped identity-layout ranked memref type;
2. every use of that exact region argument is a direct `memref.load` in the
   selected region;
3. the corresponding region operand is an exact formal argument of its
   enclosing `dataflow.thread`; and
4. every root `dataflow.thread.launch` of that thread binds the formal to a
   direct `memref.get_global` whose exact `memref.global` owns a constant
   `ElementsAttr` initializer; and
5. for every explicit load alignment greater than one, the identity layout,
   exact constant indices, element byte size, and proposed allocation-base
   alignment prove the effective byte address satisfies that alignment.

No launch, an uninitialized or mutable global, a store, a derived use of the
selected region argument, an unknown memory use, or an indirect provenance
chain makes this decision absent rather than guessing read-only behavior. An
additional launch operand that aliases the same constant global does not by
itself invalidate staging: mutation of that constant object through any alias
is already undefined source behavior. Different root launches may bind
different constant globals because the logical staging copy is executed
independently for each thread activation.

Materialization clones the complete parent, allocates one invocation-local
logical memref at the selected region entry, copies the region memory input
into it, and redirects only the proved direct loads to that buffer. The parent
thread launch and global access remain outside the Spatial region. The child
allocation carries the strongest proved base alignment required by any
redirected load; each load retains its own explicit alignment unchanged. A
dynamic or otherwise unproved effective-address alignment makes the decision
absent rather than weakening the source contract. The child is verified and
finalized through the sole Structured Program finalizer. At the terminal
SpecialMathAccuracy gate, mechanical D0 lowering expands `memref.copy` into
ordinary Dataflow load/store actors; finalized D0 contains neither
`memref.copy` nor `memref.get_global` inside a graph, and complete exact-Fabric
actor admission runs once over that final surface.

#### `PermuteLocalBufferLayout`

This decision anchors one `memref.alloc` result wholly owned by a selected
Spatial region. The allocation must have a static rank of at least two, a
positive static shape, and a dense permutation layout. It must not escape the
region or cross a callable, region, yield, or ownership boundary. Its complete
use closure contains only direct `memref.load`, `memref.store`, compatible
`memref.copy`, and `memref.dealloc` uses whose index and element semantics are
preserved by changing the allocation type. Unknown aliases, casts, rank
reduction, opaque calls, atomics, volatile accesses, or an unrepresentable
address make the decision absent.

`adjacent_storage_position` is in `[0, rank - 1)` and exchanges two adjacent
positions in the allocation's current dense storage order. The logical shape
and every logical access index remain unchanged. Repeated immutable decisions
compose arbitrary storage-order permutations without enumerating a factorial
domain. Materialization writes the selected dense strided layout into the
ordinary memref type and updates the exact closed use set; it adds no layout
record or target annotation.

Mechanical D0 lowering computes each load, store, and copy endpoint from that
endpoint's exact memref offset and strides. A copy iterates the common logical
index domain and computes source and destination addresses independently, so
different admitted dense layouts remain semantic copies. Rejecting a layout
that cannot be lowered is preferable to publishing an incomplete candidate or
silently reverting to identity layout.

#### `PipelineStagedLoop`

This decision anchors one normalized `scf.for` with an exact trip count of at
least two. The current profile admits one closed two-stage shape: each
iteration has one complete staging copy into a fresh private local buffer,
followed by a compute suffix whose accesses to that buffer are read-only. The
copy source is an exact static memref or non-rank-reducing static view. The
buffer does not escape, and exact alias, effect, and dependence analysis proves
that all cross-iteration dependences other than reuse of that private buffer
are absent. In particular, source accesses and externally visible output
effects are iteration-disjoint, and an unknown call, atomic, volatile access,
or unproved memory relation makes the decision absent.

Materialization uses ordinary SCF and memref operations to form a prologue,
overlapped kernel, epilogue, and ring-buffer selection. The copy for logical
iteration `i + 1` may overlap the compute suffix for `i`; program values and
effects retain their original logical iteration order. The buffer count is not
an independent choice:

```text
buffer_count = maximum_live_iteration_distance + 1
```

The current two-stage schedule has distance one and therefore exactly two
buffers. A future catalog that admits a longer logical stage schedule derives
its ring count by the same equation. Pipeline offsets are logical iteration
offsets, not cycle slots, a predicted initiation interval, physical memory
latency, or a Mapping reservation.

#### `PromoteOrderedBufferToChannel`

This decision anchors one fresh temporary allocation in an InstructionCore
caller. Its complete use and symbol closure must identify exactly one producer
thread launch and one or more distinct consumer thread launches. After any
required private thread-definition specialization, the producer performs one
total ordered write sequence and every consumer performs one total ordered
read sequence over the allocation's exact logical domain. Each consumer's
affine event correspondence with the producer must be a bijection with
identical payload type and order. The allocation has no other read, write,
capture, alias, escape, or observation.

The fresh allocation may be an ordinary static `memref.alloc`, or a
source-origin `llvm.alloca` that the same proof can eliminate completely. The
LLVM form is eligible only when it allocates exactly one non-empty,
statically-sized aggregate with one primitive scalar leaf type, is not
`inalloca`, and has one exact lifetime interval when lifetime markers are
present. Outside those markers, its result may occur only as the matching body
operand of the producer and admitted consumer launches, either directly or
through an exact transparent owned-thread wrapper call. Such a wrapper is one
defined, non-variadic, void LLVM callable with one block containing exactly one
dependency-free `dataflow.thread.launch`, the matching single-token
`dataflow.thread.wait`, and a void return. Every launch body operand is the
same-position wrapper formal, and the selected call is a direct void call with
exactly matching operands. Within each selected thread, every use of that
formal must resolve through in-bounds constant-offset GEPs, or the zero-offset
formal itself, to one scalar load or store. The producer and every consumer
lexical event sequence must each cover the same dense offsets exactly once in
strictly increasing order. Unknown size, padding between scalar leaves,
dynamic or repeated offsets, casts, nested pointer escape, a non-transparent
callable, or any residual use makes the decision absent.

The launch and effect proof must also establish that moving every consumer
launch before the producer-completion wait is behavior-preserving: each
consumer performs no externally visible effect before its first blocking
receive, and all remaining producer/consumer and consumer/consumer effects are
disjoint or keep the original causal order. A partial domain, reordered
access, unknown effect, visible temporary state, or any unproved launch motion
makes the decision absent.

Materialization creates one fresh `dataflow.channel.create`, specializes only
the selected producer and consumer definitions when they have other users,
replaces the proved writes and reads with one ordered send endpoint and one
ordered receive endpoint per consumer, updates all selected launches, and
removes the dead allocation. Passing the same channel handle to every consumer
is the sole software multicast representation. The canonical Dataflow channel
owner derives endpoint identity, event correspondence, and each consumer's
source map from the resulting program. The decision does not select logical
capacity, a physical FIFO, a NoC, a memory-backed queue, replication
placement, or a route.

Before rewriting an LLVM source allocation, each admitted transparent wrapper
call is mechanically replaced at that exact call site by the wrapper's thread
launch and matching wait with the call operands. This exposes the same
InstructionCore launch sequence without inlining Spatial work, merging graphs,
or changing the wrapper definition for other call sites. The ordinary channel
rewrite then owns all endpoint and launch changes.

For a source-origin LLVM allocation, materialization also removes its proved
GEPs and lifetime markers. No LLVM pointer, allocation, memory-service root, or
pointer-to-memref bridge survives from that temporary into D0. Both allocation
representations therefore materialize the same channel operations and the same
decision kind rather than defining representation-specific transforms.

All four decisions choose software-visible logical storage, order, or carrier
structure. They never choose a physical SRAM, bank, address, coalescing or
burst service, service endpoint, route, overlap latency, or Mapping record.
Bulk intrinsics have already been expanded before ownership. Each decision
composes as an ordinary immutable child and cannot introduce a parallel memory
authority.

### Workload-Aware Ownership Selection

Ownership promotion is a whole-workload decision. Before a promotion gate
ranks ownership candidates, an Evaluation model executes the exact source
program with the exact `StructuredProgram` SimulationWorkload and
SimulationRuntimeInput defined by the Simulation Artifact owner. The same
source workload/input pair remains fixed while S0 produces alternative Sn
candidates and while those candidates are ranked against the exact Fabric.
Candidate-specific Spatial workload/input pairs exist only after D0 and serve
the graph-replay semantic gate; they are not an alternative source workload.
`EvaluationEvidence` is the sole persistent owner of normalized observations.
Dynamic callable invocation counts, structured-scope activation counts, loop
trip counts, path coverage, and memory traffic may be exposed to generators as
descriptor-owned typed projections or retained as removable in-memory analysis
views. They do not form a `ProfileArtifact`, candidate-owned counter table, or
second workload identity.

Schedule descendants inherit the exact source workload and the ownership
correspondence of their immutable parent lineage. Evaluation executes the
descendant Structured candidate against that fixed source workload and replays
its own mechanically derived D0; it does not synthesize a candidate-specific
source workload or recover lineage from symbol names or operation positions.

The cost of one ownership candidate is evaluated over the complete workload:
remaining HostCore and InstructionCore work, logical-thread launch and
synchronization, boundary transfer and memory work, and dynamically activated
Spatial work under the exact Fabric capability projection. A statically small
helper therefore receives credit only for the dynamic work it actually covers.
There is no function-name classification, benchmark-specific preference, or
fixed hot-region percentage. A candidate with zero dynamic activations is
inapplicable to that workload and cannot satisfy an accelerator promotion gate.

The resolved ownership gate first applies the analytical whole-workload
`TopK` objective against the unmodified host baseline, then applies the
`functional_mismatch` absence `AllPassing` gate only to that ranked prefix. If
a completed adverse finding removes a ranked candidate, the controller
deterministically extends the analytical prefix by the number of missing
survivors and evaluates only the newly exposed candidates. Expansion stops
when `k` candidates pass or every analytically profitable candidate has been
exhausted. Missing, Unsupported, ExecutionFailed, or CancelledOrTimeout
functional Evidence remains an incomplete promotion and never triggers a
refill. This staged acquisition produces the same final cost order as eager
functional evaluation of every profitable candidate without making cache
state, worker completion order, or wall time part of selection.

The descriptor-owned source-activity projection resolves this condition before
ownership candidate cloning or publication. Such a scope is outside that exact
workload-specific Generate domain: it receives no candidate disposition and no
candidate cost or functional Evidence. This is an applicability projection of
the existing source workload, not a semantic rejection, hidden wall-time limit,
or second profile authority.

The remaining active scopes form one ownership hierarchy derived from the
Structured Program's exact operation ownership. A callable is a root; a nested
loop, selection, other structured region, or exact inlineable direct leaf call
names its nearest enclosing ownership scope. A direct leaf call is a scope only
for the existing explicit inline decision; arbitrary regionless operations are
not scopes. This parent relation is an ephemeral generator view, not a new
persistent program reference. The resolved Structured ownership policy expands
this hierarchy through the workload-prioritized finite frontier defined by the
central DSE specification. Every expanded scope retains its complete
scope-local decision domain. Descendants beyond the resolved semantic expansion
limit are outside the finite Generate domain rather than rejected candidates.

An ownership decision is one dependency-closed selected-entity decision, not
separate whole-callable, structured-region, and direct-call semantics.
Callable, loop, selection, nested-region, and exact direct-call roots differ
only in how their common data, control, memory, channel, and ownership closure
is derived. A direct call may be specialized or inlined before closure, or
remain in InstructionCore while independent regions on either side are
considered. A canonical graph never contains a general call.

The current `SpatialOwnershipDecisionPoint` contract represents candidate-local direct
call inlining with an optional exact parent-local operation reference. The
reference identifies the call site; a callee symbol is only an input to typed
symbol resolution and never becomes candidate identity. The absent coordinate
is retained. If that coordinate leaves a general call in the selected scope,
materialization records a typed `NonFinalizable` disposition rather than
rejecting the callable before its decision domain is visible.

An inline coordinate is admitted when the selected scope contains exactly one
general call, or is itself that exact direct leaf call, and the call's exact
module definition is non-variadic, single-block, and contains no general call.
The pinned MLIR inliner applies the transformation only to the private
candidate clone. The parent Structured Program and the callee definition remain
unchanged. For a direct-call root, the private materializer derives one
exact-once dependency closure around the inlined body and removes that
ephemeral boundary before publishing the child; it is not a new Structured
operation, Artifact, or graph leaf. Structural preflight may defer only that
exact call leaf while it checks every other operation in a selected enclosing
scope; the call does not thereby become graph-lowerable. After inlining and any
selected specialization, the lowering-owned structural preflight checks the
resulting selected scope in its materialized context. A fresh allocation
directly in the resulting callable block is at the prospective graph frontier
and is therefore legal; an unsupported callee leaf rejects only the inline
coordinate. A residual `llvm.call` or `llvm.invoke` after the selected
transform rejects that coordinate. The absent inline coordinate of a
direct-call root remains part of the finite domain and records
`NonFinalizable`; it does not silently keep the call in a Spatial graph.

A refusal returned by the pinned inliner is a candidate-local
`NonFinalizable` result. Malformed input, post-inline IR verification failure,
or incomplete block-activity lineage is an invocation or implementation
failure and aborts generation. The inliner's `IRMapping` must account for every
cloned nested block while the private candidate is materialized. Direct-call
inlining nevertheless publishes no source block-activity projection: it
repartitions callee activity by call site, which cannot be recovered from
aggregate source-callee block counts. Analytical Evaluation of that child must
therefore use exact candidate native observations. Structure-preserving
transforms continue to publish their exact removable block lineage.

The address normalizer owns one typed negative result for an otherwise valid
address projection. Only that typed result becomes `NonFinalizable`.
Unsupported graph structure is classified by lowering-owned structural
preflight before materialization. Once the mechanical Structured-to-Dataflow
transaction starts, clone correspondence, block replacement, specialization,
pass verification, canonical artifact/view, and tracked-launch failures are
invocation failures; callers may not flatten an owner's entire error channel
into candidate pruning.

Decision-domain projection includes requirements exposed by the exact callee.
In particular, address-index choices are derived from the union of the
selected scope and the admitted callee body before materialization. A caller
that contains no GEP therefore still receives the exact root-relative and
pointer-addressed choices required by a callee that does. No independent
callee profile, address table, or post-inline default is permitted.

One ownership invocation does not enumerate subsets of multiple call sites.
Call-graph reshaping that needs more than one atomic call transformation must
first publish ordinary Structured Program children; ownership is then
recomputed from each exact child. This keeps each lineage edge atomic and
prevents a call-site powerset from becoming a second program authority.

`UniformExactConstants` is the scope-local direct-call specialization choice.
It applies to the selected scope's nearest owning `llvm.func`, or to the
selected function itself. The function must be a defined, non-variadic local
symbol whose complete symbol-use set consists of exact direct calls. A
fixed-point proof over that closed direct-call graph may bind a formal only
when every call supplies the same typed canonical constant, either directly or
through another proven formal. Constant, zero, and symbol-address values are
admissible; poison, undef, unknown, conflicting, indirect, address-escaped, and
foreign uses are not. All proven and used formals constitute one decision. The
generator does not enumerate subsets of arguments.

Materialization changes only the candidate clone: it preserves the callable
ABI, substitutes the proven constants, simplifies unreachable control, and
then re-resolves the selected nested scope. Removal of that scope rejects only
that decision. Unused formals are omitted from the derived Spatial boundary
without changing the callable ABI. The resulting Structured Program is the
only semantic owner; neither the proof table nor the direct-call relation is a
persistent program representation. Workload path coverage cannot replace this
whole-program proof.

Dense thread-domain choices are part of the same Structured candidate. They
materialize exact rank, extents, source-IV reconstruction, and inner loop shape
before handoff. For example, a selected `[0, 1024)` source iteration domain may
materialize an extent-eight logical thread domain whose coordinate `t`
reconstructs the tile `[t * 128, (t + 1) * 128)`, with the tile-local loop in a
`loom.spatial_region`. These are logical coordinates, not physical X/Y
coordinates or AccCore selections. SystemMapping alone owns their physical
binding.

#### Schedule Coordinate Projection

Every selected nested schedule is materialized into the Structured Program
Candidate through three conceptual coordinate groups:

```text
L = AccCore-launch logical coordinates
S = graph-static actor and vector-lane coordinates
T = graph-temporal stream, recurrence, and state coordinates
```

`L` becomes the arbitrary-rank dense coordinate suffix of a
`dataflow.thread`, or the existing stable item projection of a DynamicWork
domain. `S` becomes explicit graph actor replication or typed vector lanes.
`T` becomes explicit graph streams, carry, invariant, gate, memory-frontier,
and state-transition structure. These groups are a materialization rule, not a
persistent schedule tree or another program representation. Tiling may add
coordinates, while fusion, linearization, vectorization, and interchange may
remove or reparameterize them; no equality with source loop depth is required.

Nested `sequence`, `select`, `repeat`, and `parallel` constructs compose
recursively at both the thread and graph ownership boundaries. A thread may
launch zero or more graphs sequentially or conditionally, but a graph never
launches another thread. The selected candidate must make every coordinate,
guard, source-IV reconstruction, and cross-boundary value explicit before
mechanical lowering. Neither the Dataflow finalizer nor Mapping may recover a
hidden schedule from operation position, a source loop tree, or physical
topology.

### Complete Candidate Dispositions

An Ownership Generate invocation enumerates one finite scope-local domain in
canonical Structured operation order. Every maximal dependency-closed
structured scope, and every exact inlineable direct leaf call admitted above,
with positive activation under the exact source workload and considered for
ownership appears exactly once. A scope that cannot yet be materialized records
one coordinate with no decision and a typed `NonFinalizable` disposition.
External declarations and other operations that are not ownership scopes are
not candidate attempts. Every accepted scope then enumerates its typed decision
domain in owner-defined canonical order.

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

Every selected Spatial region containing a dynamic LLVM GEP materializes one
of the address projections owned by the Canonical Dataflow specification. A
`RootRelative` candidate selects one canonical address-index width. Candidate
generation may enumerate widths admitted by the exact Fabric, but neither the
lowerer nor ambient process configuration selects one. A source integer wider
than the selected width is narrowed only when its complete signed value domain
is proven to fit. An explicit module `index` width fixes the only admissible
`RootRelative` width; it does not select that address form or suppress the
independent `PointerAddressed` candidate. A `PointerAddressed` candidate instead
retains the exact LLVM pointer type, GEP index path, no-wrap semantics, and
module DataLayout-derived pointer format. It does not acquire a synthetic
canonical index width.
When a parallel candidate retains this `PointerAddressed` form, overlap
validation compares exact DataLayout byte-address expressions at the pointer
address-space index width. It does not truncate those expressions to a
`RootRelative` canonical index width. Unresolved roots, mixed pointer-index
widths for one root, non-affine lane expressions, or byte ranges whose
disjointness is unknown reject the parallel candidate. Only a `RootRelative`
candidate must prove that its element-index expression is representable in its
selected canonical index width.

The Structured candidate materializes a selected `RootRelative` LLVM memory
access with the identity-critical unit marker `loom.root_relative_address` on
that exact `llvm.load` or `llvm.store`. No other carrier or payload is valid.
The graph-memory owner consumes the marker, derives the exact capability root
and element-index expression, emits an index-addressed Dataflow memory actor,
and removes the original pointer-address computation. A `PointerAddressed`
selection has no such marker and retains its typed pointer expression. The
marker participates in Structured Program identity and is forbidden in a
finalized Canonical Dataflow Program; it is not Mapping metadata or a second
address-form authority.

A proven constant-stride pointer induction may be materialized as one
loop-invariant base capability plus a fixed-width integer element-offset
recurrence. Constant-stride means invariant for that loop; the stride may be a
runtime value defined outside the loop. The proof identifies one finite
counted-loop domain, one exact access element type per pointer lane, an integral
element stride with a finite value range, and an accumulated offset that fits
the selected signed index width. This is an optimization candidate, not the
only legal pointer representation. When that proof is unavailable, the exact
pointer recurrence may remain a first-class Spatial value if OperationSchema,
Fabric capability, provider, and simulator contracts admit it. A candidate is
non-finalizable only when neither exact representation is supported; no pass
invents a dynamic memory capability or silently casts a pointer to one.

Every selected LLVM load or store must resolve one memory-service root at the
exact Spatial boundary. Candidate preflight and graph-memory lowering consume
the same pointer-lineage resolver. A pointer loaded from a descriptor may be an
ordinary Spatial value, but using that loaded pointer for a second memory
access also requires an independently bound service capability. If the service
is unavailable at the larger cut, that coordinate is `NonFinalizable`; a
smaller dependency-closed scope may take the loaded pointer as a live-in and
bind its service explicitly. This is a candidate boundary choice, not a rule
that pointer representation always belongs to InstructionCore. Once the
mechanical Structured-to-Dataflow transaction begins, a residual memory
operation remains an invocation failure and is never reclassified by its
diagnostic text.

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
