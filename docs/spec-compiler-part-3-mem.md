# Loom Compiler Part 3 Memory Dependence Model

This document specifies the memory-dependence model used by Part 3
when lowering structured `scf.*` accelerator regions to Loom's
dataflow IR. It is the single source of truth for the compositional
chain model, the alias-oracle interface, the dependence builder,
loop-carried memory state, and the token-wiring rules that turn the
dependence snapshot into explicit `ctrl` and `done` SSA edges
inside each `dataflow.graph` definition's body.

The first-principles IR carriers and SCF flattening templates live
in `docs/spec-compiler-part-3-dfg.md`. The pass pipeline, lit-test
layout, and acceptance checklist live in
`docs/spec-compiler-part-3-impl.md`. Placement decisions follow
`docs/spec-compiler-part-3-placement-framework.md`. This document
intentionally takes no stance on placement, dialect-level op
definitions, or fabric-side lowering.

## 1. Scope and Position

Part 3 must preserve the memory behavior of an accelerator region
when its `scf.*` shape becomes a dataflow circuit. Two questions
must be answered separately:

* **Alias.** Can two memory accesses touch overlapping storage?
* **Dependence.** Given alias plus program order plus
  structured-control-flow nesting, must one access retire before
  another can issue?

Alias is symmetric and carries no direction. Dependence is
directed, derived from program order and structured nesting, and
turns into explicit `ctrl` and `done` SSA edges inside each
`dataflow.graph` definition's body.

The chain that carries those edges is not flat. SCF nesting is
intrinsic to the source program, and the dataflow output must
preserve the same ordering hierarchy. This document factors the
chain into a compositional model: each scope of structured control
flow builds its own chain, and nested SCF ops participate in the
parent chain through an explicit boundary contract.

Vector lane masks do not create a separate memory-order network. The
scalar/packed stream boundary is specified in
`docs/spec-dataflow-vectorization.md`; this document remains the source
of truth for alias-aware ordering and for the `ctrl` / `done` edges of
masked or unmasked memory operations.

The model has two complementary token roles. The structural role
encodes dynamic execution path and region phase. The memory-order
role encodes alias-aware ordering, partitioned per alias bucket so
unrelated storage is not serialized. Both roles use the same
`none` token type at the SSA level; the distinction is what each
edge encodes at a given lowering point.

The remainder of this document is organized as follows. Section 2
specifies the compositional model. Section 3 specifies the
`MemAliasOracle` contract and the supported oracle policies.
Section 4 specifies the dependence builder, including the snapshot it
plants on the IR. Section 5 specifies loop-carried memory state. Section 6
specifies the SSA-level wiring that turns the model and the
snapshot into actual `ctrl` and `done` edges. Per-`scf.*` boundary
translation is owned by `docs/spec-compiler-part-3-dfg.md` Section 6.

## 2. Compositional Chain Model

### 2.1 Two Token Roles

A Part 3 `none` token at any SSA edge encodes one of two roles:

* **Structural execution role.** Represents the SCF dynamic
  execution status: region entry, branch selection, loop phase,
  the `scf.while` final-false close, the `scf.for` phase close,
  and similar phase information. Built from `dataflow.carry` on
  phase or condition selectors plus the boundary primitives
  `dataflow.gate`, `dataflow.demux`, and `dataflow.mux`. Does not
  carry any memory-state information.
* **Memory-order role.** Represents alias-aware ordering between
  memory accesses. Each alias partition has its own independent
  `(write_frontier, read_frontier)` pair; the role's tokens flow only
  inside one partition.
  Ordering inside a partition is real and must be preserved;
  ordering across partitions is not a correctness requirement and
  must not be introduced by the lowering.

A leaf memory op's `ctrl` operand is the rendezvous of one
structural permission token with one or more memory-order
predecessor tokens (see Section 2.5 and Section 6). The two roles meet only at
that rendezvous; the rest of the lowering keeps them separate.

This is a conceptual split, not a new IR type or a separate token
network. Both roles use `dataflow` primitive ops on `none` values.
The split exists so the rest of this section, and per-`scf.*`
boundary translation in `docs/spec-compiler-part-3-dfg.md` Section 6, can
reason about them independently.

### 2.2 Chain Scope, Atom, Effect Summary

The model uses the following terms:

* **Chain scope.** A region that hosts atoms and exposes
  structural and memory-order endpoints. The body of a
  `dataflow.graph` definition is the root chain scope. Each inner
  region of an `scf.*` op is also a chain scope. Inner regions of
  nested `scf.*` ops are chain scopes recursively.
* **Atom.** A direct child of a chain scope that participates in
  chains. Three kinds:
  - **Leaf memory event.** A `dataflow.load`, `dataflow.store`, or
    other op that the dependence builder treats as a single
    memory access. `dataflow.thread.fence` is not a leaf event in
    this model: it must appear directly in a `dataflow.thread`
    definition's body per the front-end verifier in
    `docs/spec-compiler-part-3-dfg.md` Section 9, and a thread
    definition's body is not a chain scope as defined here.
  - **Compound `scf.*` atom.** Any `scf.*` op whose transitive
    body content has memory effects. The compound is opaque to
    its enclosing scope's chain construction, except through its
    boundary endpoints.
  - **Pure op.** Any other op. Pure ops do not participate in
    chains; they are connected purely by ordinary SSA def-use.
* **Effect summary.** The set of memory effects associated with
  an atom.
  - For a leaf, it is the single (memref, R/W kind) record.
  - For a compound, it is the recursive union of all leaf effect
    summaries in the compound's transitive body.
  - For a pure op, it is empty.

A chain scope contains zero or more atoms in source order. The
order is taken from the parent block's IR layout; pure ops between
atoms do not affect chain order.

### 2.3 Partition

A partition is the alias bucket key under which the memory-order
role builds its per-partition state pair.

* A **known root** is the SSA value at the end of the
  `MemAliasOracle` walk through view-like memref ops. Distinct
  known roots default to disjoint, following the
  `BasicSsaOracle` assumption stated in Section 3.
* An **unknown-root effect** is one whose root walk encounters an
  op the active oracle does not recognize. Such effects enter a
  conservative bucket. The baseline policy uses a single bucket
  `U` regardless of memref element type or rank. Compatible-type
  filtering is not part of the baseline policy; any refinement must
  be expressed as an explicit oracle policy with its own soundness
  contract.
* `U` may-aliases every known partition in scope. A scope that
  contains any unknown-root effect therefore collapses
  same-bucket behavior with all known partitions in scope, in
  the conservative direction.
* `U` lifts upward across compound boundaries. If a compound
  atom's effect summary contains `U`, the compound participates
  in every known partition's chain at every enclosing scope that
  it is part of, until either the enclosing scope itself has no
  known partitions visible or the `dataflow.graph` definition's
  body is reached. This is the precise statement of "may-aliases
  every known partition" applied to nested scopes: a `U` effect
  at any depth must serialize against every known partition in
  every ancestor scope's chain.

Partition identity is graph-local. Numeric partition ids in the
dependence snapshot (see Section 4) are chosen per `dataflow.graph`
definition and need not match across graphs (or across launches
of the same graph definition; per-launch frontier wiring is the
caller-side concern, see Section 6.5).

### 2.4 Canonical Per-Partition State

For each alias partition `P`, the canonical memory-order state at every
recursive scope boundary is the pair:

```
state_P = (write_frontier_P, read_frontier_P)
```

`write_frontier_P` (`W_P`) covers completion of the most recent write
visible at that boundary. `read_frontier_P` (`R_P`) covers `W_P` and
all reads issued since that write. The invariant is `W_P <= R_P`: waiting
for `R_P` also waits for `W_P`.

Execution permission is a separate value `E`. It says that the current
structured path may execute; it is not memory state and it does not become a
memory-completion frontier merely because it is available. A partition not
touched by a scope passes its pair through unchanged. The unknown partition
`U` participates in every partition it may alias, as defined in Section
2.3.

### 2.5 Leaf Transfer Rules

A leaf access consumes `E` and one partition state, then produces an
updated state. `join` means direct forwarding for one input,
`dataflow.sync` for required same-path inputs, and a selector-matched
`dataflow.mux` for mutually exclusive alternatives.

For a read with completion token `d`:

```
read.ctrl = join(E, W_P)
W_P' = W_P
R_P' = join(R_P, d)
```

For a write with completion token `d`:

```
write.ctrl = join(E, R_P)
W_P' = d
R_P' = d
```

These equations are the hazard authority:

* RAW: a read waits for `W_P`.
* WAR: a write waits for `R_P`.
* WAW: a write waits for `R_P`, which includes the prior write.
* RAR: a read does not wait for prior reads, so no read-read dependence edge
  is created.

### 2.6 Recursive Scope Transfer

Every compound region is summarized as a transfer function from an incoming
`(W_P, R_P)` pair to an outgoing pair for each touched partition. Sequential
atoms compose these transfer functions in program order. Mutually exclusive
regions receive the same incoming pair and join each component with the
region selector. Same-path required tails use `dataflow.sync`; exclusive
tails use `dataflow.mux`.

Source-ordered loops recursively carry the pair needed by the next dynamic
iteration. Zero-trip execution forwards the incoming pair. Parallel regions
share the incoming state only where source semantics and alias proof permit
unordered execution, then rendezvous their resulting completion frontiers.
The exact SSA topology for recursive loops and nested regions is an
implementation task, not a separate semantic model.

### 2.7 Execution Permission

Boundary translation for `scf.if`, `scf.index_switch`, loops, and
parallel regions derives execution permission `E` from structural control.
That translation remains independent of the per-partition state pair. A leaf
memory op combines them only at its `ctrl` operand according to Section
2.5. Raw graph `ctrl_in`, a stream phase, or a region start token alone is
never proof that earlier memory effects have retired.

### 2.8 Boundary Contract and Implementation Status

Every compound structured-control boundary accepts one execution-permission
input and, for each touched partition, one incoming `(W_P, R_P)` pair. It
produces structural completion and one outgoing pair per touched partition.
Untouched partitions pass through in the enclosing scope. Per-op structural
selection remains specified in `docs/spec-compiler-part-3-dfg.md` Section 6.

The canonical model does not select one concrete recursive memory lowering.
In particular, no single token such as `mem_iter_P` is the canonical state
of a partition. A conforming lowering must preserve both `W_P` and `R_P`
through nested structured control and keep execution permission separate.

The current frontend does not yet implement this complete recursive
per-partition `(W_P, R_P)` lowering. Existing direct wiring from a graph
control value to a leaf memory op is provisional implementation behavior and
must not be read as normative semantics.

### 2.9 Non-Goals

This compositional model is scoped to memory dependence inside
`dataflow.graph` definition bodies. It does not, in this
model:

* Define `!dataflow.thread_token` semantics or
  `LoomAsyncOpInterface` participation. Those are launch-side
  protocols specified by `docs/spec-compiler-part-3-dfg.md` Section 3
  and Section 5.4.1.
* Define `dataflow.map_info` direction enforcement or HostCore
  visibility. That is the boundary memory-effect summary
  specified by `docs/spec-compiler-part-3-dfg.md` Section 3 rule 4 and
  rule 7.
* Define `DeviceMappingAttrInterface` or thread-grid mapping
  semantics. Those are placement-side concerns.
* Replace the dataflow primitive op definitions for
  `dataflow.{stream, carry, invariant, gate, mux, demux, sync,
  constant}`. Those definitions are owned by
  `docs/spec-dataflow-part-1-streaming.md` and
  `docs/spec-dataflow-part-2-control.md`.
* Define cross-graph partition identity. `dataflow.graph` is a leaf
  in this memory model and partition ids are graph-local (scoped to
  the def's body, with caller-side per-launch frontier wiring
  materialized at each `dataflow.graph.launch` site, per Section 6.5).
  Any design that composes memory chains across graph definitions
  must add an explicit child-block-arg to parent-operand alias-root
  mapping at the graph boundary; numeric ids alone are not enough.
  That cross-graph composition contract is out of scope here.

The model assumes those other contracts are already enforced by
their respective sections.

## 3. Alias Oracle

The alias oracle answers a single question: do two memory effects
conflict? Conflict is symmetric and direction-free. The dependence
builder in Section 4 turns those answers into directed edges using program
order and structured-control-flow nesting. Effect-summary lift
across compound `scf.*` atoms is also driven from the oracle: per
Section 2.2 a compound's summary is the recursive union of its leaves'
summaries, and per Section 2.3 the unknown bucket `U` is what the oracle
reports for any leaf whose root walk leaves the recognized set. The
two interchangeable implementations below share the `MemAliasOracle`
interface; the C++ class signature and the pass that materializes
oracles per `dataflow.graph` definition's body live in
`docs/spec-compiler-part-3-impl.md`.

### 3.1 BasicSsaOracle

`BasicSsaOracle` is default-on for fast iteration during development.
It performs a walk from each memory access's memref operand back
through view-like ops to either a known terminal root or an
unrecognized producer. The walk has three classes of stop / continue
conditions:

* **Recognized view-like ops (peel and continue).** The walk
  recursively follows the source operand of each of the following
  ops: `memref.cast`, `memref.subview`, `memref.view`,
  `memref.expand_shape`, `memref.collapse_shape`,
  `memref.reinterpret_cast`, `memref.transpose`,
  `dataflow.partition_layout` (the partitioned-data annotation defined
  in `docs/spec-compiler-part-4-partitioned-data.md`, which is a
  same-type view of its source memref), and `dataflow.map_info`
  (the boundary metadata op defined in
  `docs/spec-compiler-part-3-dfg.md` Section 5.4.6, which is also a pure
  view-like alias of its source). Each of these produces a memref
  that shares storage with its source by construction, so the
  walk peels them off without breaking aliasing. Peeling
  `dataflow.map_info` is what lets the cross-boundary continuation
  rule below preserve storage identity end-to-end across a thread
  boundary: an in-thread leaf walks to its block argument, then
  to the matching boundary operand (a `dataflow.map_info` result),
  then peels through `map_info` to its source, and continues from
  there.
* **Recognized terminal roots (stop, treat as known root).** The
  walk stops and records a known root at any of the following:
  `memref.alloca`, `memref.alloc`, `memref.get_global`, or a
  function-block argument with no defining op. These define a fresh
  storage identity for the purposes of the oracle (with the
  symbol-keyed adjustment for `memref.get_global` described below).
* **`IsolatedFromAbove` block arguments (cross-boundary continue).**
  When the walk reaches an entry block argument of an
  `IsolatedFromAbove` def op (notably a `dataflow.graph`
  definition and a `dataflow.thread` definition), it does not stop.
  The block argument is bound positionally to the corresponding
  launch op operand in the enclosing scope, resolved via the def's
  `function_type`: for a `dataflow.graph` definition the index
  excludes the leading `ctrl_in` slot in `function_type.inputs`
  (so user data block-arg `i` corresponds to graph launch operand
  `i + 1`); for a `dataflow.thread` definition the user data
  block-args are the leading `N` block args (per
  `docs/spec-compiler-part-3-dfg.md` Section 5.4.1's
  `(args_*, thread_ctrl, iv_*)` order), so the walk excludes
  `thread_ctrl` and the grid induction-variable args automatically.

  **Multi-launch handling.** The single-launch baseline policy emits
  exactly one launch site per defined graph symbol; the def + launch
  is a 1:1 pairing and the chain analysis in this document is built
  against that pairing. Under this contract, "the matching launch
  operand" is unambiguous and the walk continues on it.

  If a graph definition is reused at multiple launch sites (a graph
  kernel called from multiple program points), the chain analysis
  must run per-launch and the resulting state pairs must be joined
  caller-side, **or** the verifier must additionally enforce an "all
  launch sites supply the same alias-root for each block arg"
  invariant so a single per-def chain remains sound. The
  direction/body-effect compatibility check in Section 3.7 of
  `docs/spec-compiler-part-3-dfg.md` already provides part of the
  per-launch sanity envelope; the multi-launch alias-identity
  invariant is the additional rule required for any per-def
  multi-launch analysis policy.

  The walk therefore continues on the matching launch op operand
  in the enclosing scope. This is what makes storage identity
  stable across the `IsolatedFromAbove` boundary in the baseline:
  two graph
  block arguments bound to the same parent memref through one
  `dataflow.graph.launch` (two subviews of the same alloc, or the
  same value passed twice) walk
  to the same root; otherwise the oracle would treat them as
  disjoint and miss required read/write or write/write dep edges.
  Cross-boundary continuation is allowed only across SSA boundary
  operands; it does not enable looking through unrelated parent
  scopes.
* **Unknown producer (stop, enter `U`).** Any other op that produces
  a memref-typed result terminates the walk without yielding a known
  root. The walk does not invent a new root from such an op; instead
  the access enters the conservative unknown bucket `U` defined in
  Section 2.3. This includes ops whose freshness or aliasing relationship
  is not statically guaranteed by the oracle, for example
  `bufferization.to_memref` (whose resulting memref may share
  storage with an existing buffer depending on the active
  bufferization strategy), an SSA value returned from a
  `func.call`, an `unrealized_conversion_cast` to a memref type,
  custom buffer reshape ops, and any memref-producing op the oracle
  has not been taught about. This is the soundness rule
  that prevents such ops from being silently treated as fresh
  disjoint roots.

Conflict is decided as follows. For two accesses `a` and `b`:

* If both `a` and `b` end with known roots, they conflict iff their
  roots have the same storage identity AND the pair is not
  load-load. The storage identity is the SSA value for
  `memref.alloca`, `memref.alloc`, and function block-args; for
  `memref.get_global` it is the referenced global symbol, not the
  result SSA value, because two distinct `memref.get_global @g` ops
  produce different SSA values for the same storage. Distinct
  storage identities default to disjoint, following the standard
  basic-AA assumption that block-args, allocs, and globals have
  pairwise-disjoint storage by default.
* If at least one of `a` or `b` is in `U`, they may-alias every
  other access of any compatible memref kind in scope, regardless
  of root, with the single exception that two loads still do not
  conflict. This realizes the rule from Section 2.3 that a `U` effect
  may-aliases every known partition.

The baseline policy uses any-memref same bucket as the
"compatible memref kind" predicate. Element type and shape rank are
intentionally NOT used as disjoint witnesses, because view-like ops
and bufferization paths can change element type or rank without
changing underlying storage. This is conservative and matches the
baseline soundness direction. More precise compatibility policies
must refine both the leaf walk and the unknown-bucket policy.

Bounds and offsets are not consulted at any point; the oracle is
intentionally storage-identity only.

### 3.2 MlirAaOracle

`MlirAaOracle` ships in the same library as `BasicSsaOracle` and is
exercised on the representative differential subset described at the
end of this section. It wraps `mlir::AliasAnalysis`, configured with
whatever external alias-analysis interfaces are registered, as a
refinement of `BasicSsaOracle`. It starts from the basic conflict
set and removes pairs that upstream MLIR AA proves `MustNotAlias`.
`MayAlias` and `MustAlias` keep the basic answer. Loads vs. loads
still do not conflict.

The refinement applies to leaf-pair queries only: when both
accesses are leaves visible to the dependence builder at the same
scope, MlirAaOracle may demote a basic-conflict pair to
non-conflicting if upstream AA proves them `MustNotAlias`. This
holds uniformly, including pairs where one or both sides come from
`U`: a specific unknown-producer op proven disjoint from a specific
known root drops out of the leaf-pair conflict set.

Effect-summary lift across compound `scf.*` atoms (Section 3.3) does not
benefit from this refinement. The summary records partition
identity by `BasicSsaOracle`'s classification: a compound's summary
contains `U` whenever any inner leaf is in `U`, regardless of
whether MlirAaOracle would have demoted some inner leaf-pair
conflicts. The conservative compound summary is intentional in
the baseline policy; any tighter policy requires summary-level AA
support and must define how refined summaries compose across
compound atoms.

### 3.3 Effect Summary Lift Rule

Per-leaf alias answers compose into compound atoms through the
effect-summary lift defined in Section 2.2 and Section 2.3:

* A compound atom's effect summary is the recursive union of its
  inner leaves' effect summaries. Membership in the summary is by
  partition identity: each known root the inner leaves touch
  contributes one entry, and any inner leaf that is in `U`
  contributes `U`. Read-only leaves still contribute their
  partition to the summary; the read-read suppression of Section 4 only
  prevents dependence edges, not summary membership.
* If any inner leaf is in `U`, the compound's summary contains `U`.
  The compound participates in `U`'s own per-partition chain at
  every scope that exposes `U`, plus, by the lift rule of Section 2.3,
  every known partition's per-`P` chain at every enclosing scope,
  until either the enclosing scope itself has no known partitions
  visible or the `dataflow.graph` definition's body is reached.
* Frontier membership at the compound boundary uses
  `BasicSsaOracle`'s classification, not pair-level MlirAaOracle
  refinement (Section 3.2). If any leaf inside the compound is in `U`,
  the compound is wired into every enclosing known partition's
  chain, regardless of whether MlirAaOracle would have demoted a
  specific inner-vs-outer leaf pair to `MustNotAlias`. Pair-level
  refinement still applies inside the same scope where both
  leaves are visible, and inside the compound's own scope, but it
  does not change which partitions appear at the compound's
  boundary.
* Within a single scope, `U` participates in its own per-partition
  chain like any other partition: two writes in `U` (or a read and
  a write in `U`) form a same-partition dependence pair under Section 4
  and update `U`'s state pair. Read-read pairs in `U` still
  do not create dependence edges, consistent with Section 3.1's
  load-load rule.

The per-partition state pair of Section 2.4 then carries the compound's
effects through the recursive transfer of Section 2.6.

`BasicSsaOracle` is the baseline default. `MlirAaOracle` is the
refining policy: it must preserve structural IR shape while allowing
the `loom.mem_dep_preds` snapshot to differ when upstream MLIR AA
proves additional `MustNotAlias` pairs. Test coverage for oracle
policies belongs to `docs/spec-compiler-part-3-impl.md`; this
document defines only the required semantic relationship between
the policies.

## 4. Dependence Builder

`MemoryDependenceBuilder` produces a directed dep-edge snapshot derived from
the transfer rules in Section 2.5. Edges live inside one partition and one
chain scope. They support analysis and the current implementation, but they
are not a second authority beside the canonical `(W_P, R_P)` state.

### 4.1 Inputs and Outputs

The builder operates on one `dataflow.graph` definition's body at
a time. Its inputs are the def body's IR after parallel-SCF
normalization (so the leaf set is the same set Section 6 will see), a
configured `MemAliasOracle` per Section 3, and the partition assignment
derived from the Section 3.1 walk on each leaf's memref operand and
lifted to compound atoms by Section 3.3. Its outputs are the per-graph
snapshot consumed by
Section 5 and Section 6: `loom.mem_dep_id` and `loom.mem_dep_preds` on each leaf
memory op, `loom.mem_loop_id` and `loom.mem_loop_states` on each
loop op carrying memory state (consumed only by Section 5), and the
parallel-provenance side data on cloned leaves and generated loops
(`loom.parallel_group`, `loom.parallel_chunk`,
`loom.parallel_chunks`, or an equivalent analysis side table).

### 4.2 Partition Assignment

Partition identity is graph-local and follows the Section 3 alias-oracle
contract. The Section 3.1 walk on each leaf's memref operand assigns a
**primary partition**: either a known root storage identity, or
the conservative bucket `U` when the walk leaves the recognized
set. A leaf in `U` also **participates** in every known partition
visible at the leaf's chain scope, by the Section 2.3 / Section 3.3 lift rule;
the participation is what realizes "U may-aliases every known
partition" at the dep-edge-candidate level. A leaf with a known
primary partition `P` only participates in `P`. Each compound
`scf.*` atom inherits the union of its inner leaves' partition
participations by the Section 3.3 effect-summary lift, including the
implicit lift of any inner `U` leaf into every known partition
visible at the compound's scope and at every enclosing scope.
Numeric partition ids are graph-local.

Two atoms in the same chain scope that participate in the same
partition are the only direct candidates for a same-partition dep
edge in that scope. A `U` leaf and a known-`P` leaf in the same
scope therefore share `P` (through the `U` lift) and are valid
edge candidates in `P`'s chain; the conflict gate of Section 4.3 then
decides whether the pair conflicts. Cross-partition pairs and
cross-scope pairs are never direct edge candidates:
cross-partition ordering uses independent state pairs, and cross-scope
ordering uses the recursive transfer of Section 2.6 and the boundary
contract of Section 2.8.

### 4.3 Per-Partition Edge Construction

For each chain scope `S` and each partition `P` in `S`'s transitive
partition set, the builder constructs the dep edge set over the
atoms in `S` that participate in `P`. Direction comes only from
program order and structured-control-flow nesting at `S`; alias is
symmetric and never defines a direction by itself.

* **Conflict gate.** An ordered pair `(p, o)` with `p` before `o`
  in `A_P(S)` is a dep candidate iff `MemAliasOracle` reports a
  non-`MustNotAlias` answer for the pair restricted to `P` AND the
  pair is not load-load. The query takes one of two forms:
  - **Leaf-vs-leaf, same chain scope `S`.** This is the direct
    leaf-pair query. `MlirAaOracle`'s refinement (Section 3.2) applies
    here; a basic-conflict pair may be demoted to non-conflicting
    if upstream AA proves `MustNotAlias`.
  - **Compound-involving (leaf-vs-compound or compound-vs-compound).**
    The pair conflicts in `P` iff at least one inner-leaf pair
    drawn from the contributing inner leaves on each side
    conflicts. Compound-boundary lift uses `BasicSsaOracle`'s
    classification per Section 3.3 only; `MlirAaOracle`'s leaf-pair
    refinement does not propagate into compound boundaries in
    the baseline policy, regardless of whether some inner-vs-outer
    leaf pair would have been demoted as a direct query. A tighter
    policy must specify the summary-level AA rule before changing
    this behavior.
* **Path-sensitive pruning.** Atoms in mutually exclusive branches
  do not need an edge between each other solely because they
  conflict; each branch's state participates in the componentwise
  selector join from Section 2.6. A
  conservative implementation may serialize more when it cannot
  prove path exclusivity, but it must not omit a dep edge that
  preserves an observable read/write or write/write order.
* **Parallel-provenance exception.** Accesses in different logical
  iterations or different chunks of the same original
  `scf.parallel` are unordered by the source program. The builder
  must not add cross-iteration or cross-chunk dep edges inside a
  parallel-provenance compound solely because they may alias. It
  still records intra-iteration dep edges and dep edges between
  the parallel-provenance compound and surrounding atoms in its
  enclosing scope. The compound's pair transfer must rendezvous the
  chunk completion frontiers as described in Section 2.6.
* **Loop-carried dep edges are real.** If an access in a later
  iteration of a source-ordered loop can conflict with an access
  in an earlier iteration, the loop's per-partition state pair must
  carry that ordering. Section 5 defines the abstract recurrence;
  concrete recursive lowering remains pending.
* **Transitive reduction.** The builder may remove transitively
  implied edges intra-partition and intra-scope, provided the
  remaining immediate predecessor set induces the same partial
  order. It must not collapse edges across partition or scope
  boundaries.

### 4.4 Snapshot Format

The builder plants a per-graph snapshot using only deterministic
graph-local integer ids. Every numeric id introduced by this
snapshot is graph-local: it is chosen per `dataflow.graph`
definition and need not match across graphs (or across launches of
the same definition). The mem-dep, mem-loop, partition, and
parallel group / chunk id namespaces are independent (each may
start at zero) and are all graph-local.

* `loom.mem_dep_id = N` on each leaf memory op (`dataflow.load`
  and `dataflow.store`), in deterministic traversal order.
  `dataflow.thread.fence` is not part of the snapshot because it
  cannot appear in a graph body; see Section 2.2 and the verifier rule
  in `docs/spec-compiler-part-3-dfg.md` Section 9.
* `loom.mem_dep_preds = [P0, P1, ...]` on each leaf, listing the
  immediate dep predecessors in the leaf's partition after the
  transitive reduction of Section 4.3. Each entry references another
  leaf's `loom.mem_dep_id`; entries may name leaves at the same
  chain scope or leaves nested inside a sibling compound atom
  (see "Cross-scope predecessor resolution" below).
* `loom.mem_loop_id = L` on each loop op carrying memory state
  (consumed only by Section 5). `L` is a graph-local loop id, drawn from
  a per-graph numbering policy independent of the `mem_dep_id`
  namespace.
* `loom.mem_loop_states = [...]` on each such loop, referencing
  accesses only by `loom.mem_dep_id`. Internal partition ids and
  all non-reference integer fields inside this attribute are
  graph-local on the same per-graph numbering policy.
* Parallel-provenance side data: `loom.parallel_group`,
  `loom.parallel_chunk`, `loom.parallel_chunks`, on cloned leaves
  and generated loops. Group and chunk ids are graph-local and
  drawn from per-graph namespaces independent of the other id
  attributes; all three are stripped by `loom-finalize-dfg`.

Partition identity is not stored per leaf, but the predecessor
list is multi-partition aware. For a leaf `L` whose primary
partition is a known root `P_L`, `loom.mem_dep_preds` lists every
edge predecessor in `P_L` from the same chain scope (or, via the
cross-scope rule below, from a deeper scope inside a sibling
compound atom). For a leaf `L` whose primary partition is the
unknown bucket `U`, `L` participates in every known partition
visible at its chain scope per Section 2.3 lift; its
`loom.mem_dep_preds` entries cover every such partition's edges
in one combined list, so a known-`P` predecessor of a `U` leaf
appears in the same list as any `U` predecessor or any known-`Q`
predecessor (different partitions are not split into separate
lists). Current consumers use each predecessor leaf's primary
partition (recovered by re-running the Section 3.1 walk on the
predecessor's memref operand) to group the edge. Compound `scf.*`
atoms still present in the graph
do not get their own `loom.mem_dep_id`; only leaves do. Integer ids
keep the snapshot stable across printing,
parsing, and in-place memory-op rewrites.

**Cross-scope predecessor status.** A leaf-only predecessor id does not
encode the outgoing `(W_P, R_P)` pair of a sibling compound. The current
implementation may resolve such ids through one provisional compound tail,
but that is not the canonical recursive transfer. Complete lowering must
derive both components at the compound boundary instead of treating one
tail token as the partition state.

## 5. Loop-Carried Memory State

A source-ordered loop carries the canonical state pair for every alias
partition whose hazards cross an iteration boundary. Abstractly, one dynamic
iteration applies the recursive body transfer:

```
(W_P_next, R_P_next) = body_transfer(E_body, W_P, R_P)
```

The next true iteration receives that pair. A zero-trip loop forwards the
incoming pair unchanged. A final condition check in `scf.while` still
contributes any memory effects executed by the before-region before the
false exit.

Read-only iterations update only `R_P`; they do not create RAR issue
ordering. A later write must nevertheless wait for the accumulated
`R_P`, which preserves WAR. Any write resets both components to that
write's completion token, preserving RAW and WAW for subsequent accesses.

The exact representation of the pair across `dataflow.stream`, `carry`,
`gate`, `demux`, nested regions, and path joins remains pending. The
current implementation has not completed recursive per-partition
`(W_P, R_P)` lowering. This specification therefore does not designate a
single carried token, a completion-only carry, or a particular false-lane
projection as the canonical loop memory state.

## 6. Token Wiring

This section states the target SSA relationship implied by Section 2.
Structural execution permission and the per-partition state pair remain
distinct except at each leaf `ctrl` rendezvous. Section 6.3 identifies the
current implementation gap; Section 6.5 records the unresolved graph
retirement boundary.

### 6.1 Graph Boundary Ports

Each `dataflow.graph` definition carries explicit leading `none`
slots in its `function_type` for `ctrl_in` and `done_out`. The
def's body has a matching leading entry-block argument of type
`none` (the per-launch start signal), and the body's
`dataflow.yield` terminator has a matching leading operand of
type `none` (the per-launch completion signal). These are real
SSA values even when the custom assembly compresses their
spelling. At each `dataflow.graph.launch` use site, the
per-launch ctrl_in is supplied as the launch's leading operand
and the per-launch done_out is the launch's leading result. The
op definitions are owned by `docs/spec-compiler-part-3-dfg.md`
Section 5.5; this document uses the def's leading block argument as the
root chain scope's structural-permission source. It is not implicitly the
root `(W_P, R_P)` state. The def's leading yield operand is the declared
completion port whose full semantics remain subject to Section 6.5.
Per-launch caller-side wiring (which SSA values
feed `ctrl_in` and which uses observe `done_out`) is materialized
at each launch site.

### 6.2 Structural Plane Wiring

The structural plane carries the dynamic execution permission
token from the graph boundary down to every leaf and compound
atom inside the graph. Per Section 2.5 plane orthogonality it never
aggregates memory completion; it expresses only "execution has
reached this position on this dynamic path".

* The graph's `ctrl_in` block argument is the root structural-
  permission token, equivalently `S.struct_at_A` in Section 2.5 when `S`
  is the root scope and `A` is any atom directly nested in the
  graph body.
* For every compound `scf.*` atom traversed on the way to a leaf,
  the compound's per-op boundary translation in
  `docs/spec-compiler-part-3-dfg.md` Section 6 specifies how the token
  splits, mux-joins, or carries through the compound's inner regions.
* At each leaf op `L` whose chain scope is `S`, `S.struct_at_L`
  is the value produced by the innermost boundary translation
  step on the dynamic path from the graph entry to `L`. Its
  identity depends on `L`'s IR ancestry (which `scf.if` branch,
  which loop body, which iteration phase); the per-`scf.*`
  template names and threads it, and the wiring step here just
  consumes it.

### 6.3 Memory Plane Wiring

For each leaf access in partition `P`, lowering must materialize the
Section 2.5 transfer from the current `(W_P, R_P)` pair. The leaf's
execution permission is produced by structured-control lowering and remains
SSA-distinct from both memory frontiers until the leaf `ctrl` rendezvous.

A read joins execution permission with `W_P`, then joins its `done` into
`R_P` while preserving `W_P`. A write joins execution permission with
`R_P`, then assigns its `done` to both components. Recursive scopes and
loops apply the same transfer composition described in Sections 2.6 and 5.

The current `LowerGraphMemoryPass::tryRewriteOne` path primarily tokenizes
memory operations and may wire `ctx.ctrl` directly to their
`ctrl` operands. That is an implementation gap. It is not the normative
per-partition memory plane and does not prove that nested hazards are
represented.

### 6.4 Leaf Op Ctrl Rendezvous

The leaf rendezvous follows the access kind:

```
read.ctrl  = join(execution_permission, W_P)
write.ctrl = join(execution_permission, R_P)
```

The join may collapse when both operands are the same proven SSA dependency.
Such a collapse is an optimization of the equations above, not permission to
treat a raw start token as memory completion.

### 6.5 Graph done_out

The intended graph completion boundary must rendezvous:

* structural execution completion;
* the final `R_P` for every live alias partition, which also covers
  `W_P` by invariant;
* any other operation completion required by the graph contract.

This is a target contract, not a completed lowering. The protocol still lacks
an agreed runtime/IR anchor proving that zero-output close transitions and
all actor resets have retired. Consequently, current graph `done_out`
wiring cannot claim graph retirement closure. In particular, forwarding raw
`ctrl_in`, a launch start token, or any other execution-permission value is
never by itself a completion proof.

### 6.6 Multi-fanout and Read-Read Pairs

Two cross-cutting rules close the wiring specification.

* Multi-fanout of a single `done` token is handled by SSA value reuse,
  not an extra op. A read completion may contribute to `R_P` while
  remaining available to other proven consumers.
* Read-read pairs have no dependence edge, even when they alias,
  per Sections 2.5 and 4. Independent reads may issue without waiting
  for each other; each still waits for `W_P`, and each completion joins
  into `R_P` so a later write waits for all of them.

## 7. References

* `docs/spec-compiler-part-3-dfg.md` -- IR boundary contracts, SCF
  flattening templates, and verifier invariants. The per-`scf.*`
  boundary translation rules that instantiate the contract in Section 2.8
  live in Section 6 of that document.
* `docs/spec-compiler-part-3-impl.md` -- pass pipeline, lit-test
  layout, acceptance checklist, and maintenance plan.
  The `MemAliasOracle` C++ interface signature and the pass that
  materializes oracle instances per `dataflow.graph` definition's
  body are specified there.
* `docs/spec-compiler-part-3-placement-framework.md` -- common
  placement-partition framework. L2 graph placement decides which
  ScalarCore code becomes a `dataflow.graph` definition + a
  `dataflow.graph.launch` at the cut site; this document
  specifies the chain model that runs inside each such graph
  definition's body.
* `docs/spec-dataflow-part-1-streaming.md` -- precise timing
  semantics for `dataflow.stream`, `dataflow.carry`,
  `dataflow.invariant`, and `dataflow.gate`.
* `docs/spec-dataflow-part-2-control.md` -- precise firing semantics
  for `dataflow.constant`, `dataflow.sync`, `dataflow.mux`, and
  `dataflow.demux`.
