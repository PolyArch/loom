# Loom Compiler Part 3 Memory Dependence Model

This document specifies the memory-dependence model used by Part 3
when lowering structured `scf.*` accelerator regions to Loom's
dataflow IR. It is the single source of truth for the compositional
chain model, the alias-oracle interface, the dependence builder,
loop-carried memory state, and the token-wiring rules that turn the
dependence snapshot into explicit `ctrl` and `done` SSA edges
inside each `dataflow.graph`.

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
`dataflow.graph`.

The chain that carries those edges is not flat. SCF nesting is
intrinsic to the source program, and the dataflow output must
preserve the same ordering hierarchy. This document factors the
chain into a compositional model: each scope of structured control
flow builds its own chain, and nested SCF ops participate in the
parent chain through an explicit boundary contract.

The model has two complementary token roles. The structural role
encodes dynamic execution path and region phase. The memory-order
role encodes alias-aware ordering, partitioned per alias bucket so
unrelated storage is not serialized. Both roles use the same
`none` token type at the SSA level; the distinction is what each
edge encodes at a given lowering point.

The remainder of this document is organized as follows. §2
specifies the compositional model. §3 specifies the
`MemAliasOracle` contract and the two milestone implementations.
§4 specifies the dependence builder, including the snapshot it
plants on the IR. §5 specifies loop-carried memory state. §6
specifies the SSA-level wiring that turns the model and the
snapshot into actual `ctrl` and `done` edges. Per-`scf.*` boundary
translation is owned by `docs/spec-compiler-part-3-dfg.md` §6.

## 2. Compositional Chain Model

### 2.1 Two Token Roles

A Part 3 `none` token at any SSA edge encodes one of two roles:

* **Structural execution role.** Represents the SCF dynamic
  execution status: region entry, branch selection, loop phase,
  the `scf.while` final-false reset, the `scf.for` sentinel reset,
  and similar phase information. Built from `dataflow.carry` on
  rwc-style or condition selectors plus the boundary primitives
  `dataflow.gate`, `dataflow.demux`, and `dataflow.mux`. Does not
  carry any memory-state information.
* **Memory-order role.** Represents alias-aware ordering between
  memory accesses. Each alias partition has its own independent
  frontier; the role's tokens flow only inside one partition.
  Ordering inside a partition is real and must be preserved;
  ordering across partitions is not a correctness requirement and
  must not be introduced by the lowering.

A leaf memory op's `ctrl` operand is the rendezvous of one
structural permission token with one or more memory-order
predecessor tokens (see §2.5 and §6). The two roles meet only at
that rendezvous; the rest of the lowering keeps them separate.

This is a conceptual split, not a new IR type or a separate token
network. Both roles use `dataflow` primitive ops on `none` values.
The split exists so the rest of this section, and per-`scf.*`
boundary translation in `docs/spec-compiler-part-3-dfg.md` §6, can
reason about them independently.

### 2.2 Chain Scope, Atom, Effect Summary

The model uses the following terms:

* **Chain scope.** A region that hosts atoms and exposes
  structural and memory-order endpoints. The body of a
  `dataflow.graph` is the root chain scope. Each inner region of
  an `scf.*` op is also a chain scope. Inner regions of nested
  `scf.*` ops are chain scopes recursively.
* **Atom.** A direct child of a chain scope that participates in
  chains. Three kinds:
  - **Leaf memory event.** A `dataflow.load`, `dataflow.store`, or
    other op that the dependence builder treats as a single
    memory access. `dataflow.thread.fence` is not a leaf event in
    this model: it must appear directly in a `dataflow.thread`
    body per the front-end verifier in
    `docs/spec-compiler-part-3-dfg.md` §9, and a thread body is
    not a chain scope as defined here.
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
role builds its per-partition frontier.

* A **known root** is the SSA value at the end of the
  `MemAliasOracle` walk through view-like memref ops. Distinct
  known roots default to disjoint, following the
  `BasicSsaOracle` assumption stated in §3.
* An **unknown-root effect** is one whose root walk encounters an
  op the active oracle does not recognize. Such effects enter a
  conservative bucket. The first milestone uses a single bucket
  `U` regardless of memref element type or rank. Compatible-type
  filtering is intentionally off in the first milestone; B.1
  follow-up work refines the leaf walk and the bucket policy.
* `U` may-aliases every known partition in scope. A scope that
  contains any unknown-root effect therefore collapses
  same-bucket behavior with all known partitions in scope, in
  the conservative direction.
* `U` lifts upward across compound boundaries. If a compound
  atom's effect summary contains `U`, the compound participates
  in every known partition's chain at every enclosing scope that
  it is part of, until either the enclosing scope itself has no
  known partitions visible or the `dataflow.graph` boundary is
  reached. This is the precise statement of "may-aliases every
  known partition" applied to nested scopes: a `U` effect at any
  depth must serialize against every known partition in every
  ancestor scope's chain.

Partition identity is graph-local. Numeric partition ids in the
dependence snapshot (see §4) are chosen per `dataflow.graph` and
need not match across graphs.

### 2.4 Per-Partition Memory Frontier

Each chain scope, for every partition `P` transitively touched by
any atom in the scope, exposes two memory-order endpoints:

* `incoming_P`: the memory-order frontier flowing into the scope
  for partition `P`.
* `outgoing_P`: the memory-order frontier flowing out of the
  scope for partition `P`.

A partition that is not touched anywhere in the scope is absent
from the scope's interface. The scope contributes neither an
incoming nor an outgoing frontier for it, and the parent scope's
frontier flows through unchanged. The single exception is the
unknown bucket `U`: a scope whose effect summary contains `U`
also participates in every known partition `P` in any enclosing
scope, per the lift rule in §2.3. Operationally, the parent
scope wires the compound into both `U` and every known
partition's per-`P` chain, so a `U` effect inside the compound
serializes against every conflicting known-partition effect
outside.

For the root scope (a `dataflow.graph` body), `incoming_P` for
every touched partition is derived from the graph's `ctrl_in`
control port and any pre-graph dependence tail. The graph's
`done_out` is the rendezvous of all final per-partition outgoing
frontiers (see §6.3).

### 2.5 Single-Level Chain Rule

For a fixed chain scope `S`, chain construction proceeds in two
planes.

The structural plane links atoms in source order. Each atom has
one structural-permission input and one structural-completion
output. Pure-control splits and joins (`dataflow.demux`,
`dataflow.mux`, `dataflow.carry`, `dataflow.gate`) are introduced
only by per-`scf.*` boundary translation; the single-level rule
itself does not insert them.

**Plane orthogonality invariant.** A structural-completion token
signals only that dynamic execution has advanced past its source
op. It does not signal that memory effects inside the op have
retired. Memory retirement is exclusively the memory plane's
responsibility; a leaf op's memory `done` flows independently
into its partition's frontier. The two planes meet only at the
leaf rendezvous `ctrl = sync(structural_permission,
memory_predecessor)` defined later in this subsection. The
structural chain therefore must not aggregate memory completion;
otherwise unrelated partitions would be serialized through the
structural plane.

The memory plane treats `S` as if its atom set were sliced by
partition. For each partition `P` in `S`'s transitive partition
set:

```
A_P = atoms in S that touch P (leaf-touch-P or compound-touch-P)

build dep edges over A_P:
  same-partition default may-alias holds for known/known and
  known/unknown pairs;
  structured-CFG path-sensitive pruning: atoms that occur on
  mutually exclusive dynamic paths do not chain through each
  other directly;
  no read-read edge.

for each atom A in A_P:
  let preds_P(A) = immediate dep predecessors of A in P at S
  if preds_P(A) is empty:
    incoming_A_P = S.incoming_P
  if preds_P(A) is one same-path predecessor B:
    incoming_A_P = B.outgoing_P
  if preds_P(A) is multiple same-path predecessors:
    incoming_A_P = dataflow.sync(...).out0
  if preds_P(A) is multiple mutually exclusive predecessors:
    incoming_A_P = selector-matched dataflow.mux of those tails
  if preds_P(A) is mixed (some same-path required, some
  mutually exclusive alternatives):
    1. partition preds_P(A) into per-path groups by which dynamic
       path produces each predecessor.
    2. for each path, build a per-path tail by dataflow.sync over
       the same-path required tails on that path.
    3. selector-matched dataflow.mux the per-path tails.
    Equivalently, when one predecessor is common to all paths and
    the others are alternatives:
    incoming_A_P = dataflow.sync(common_tail,
                                  dataflow.mux(alt_tails)).
  loop-carried-from-S, if any:
    additionally include the loop-carried P-state token
    (see §5)

  outgoing_A_P = the P-tail produced by A at S.
```

When `A` is a leaf memory event, its physical `ctrl` operand is
`dataflow.sync(S.struct_at_A, incoming_A_P)`, where
`S.struct_at_A` is the structural permission token at `A`'s
position. Its `done` result becomes the source of `outgoing_A_P`
(no further wrapping).

When `A` is a compound `scf.*` atom, the boundary translation rule
in `docs/spec-compiler-part-3-dfg.md` §6 specifies how
`(A.struct_in, {incoming_A_P}_{P touched by A})` flow into `A`'s
inner regions and how `(A.struct_done, {outgoing_A_P})` are
collected from those regions.

`S.outgoing_P` is computed from the per-`P` chain tails of atoms
that have no `P`-successor at `S`:

* If exactly one such tail exists, it is `S.outgoing_P` directly.
* If multiple same-path tails exist, `S.outgoing_P` is the
  `dataflow.sync` rendezvous of those tails.
* If multiple tails exist on mutually exclusive paths,
  `S.outgoing_P` is the selector-matched `dataflow.mux` of those
  tails.
* If a dynamic path contributes no atoms touching `P`, that path
  forwards `S.incoming_P` unchanged through the selector-matched
  join above.

### 2.6 Recursive Descent

Chain construction visits scopes in post-order: every compound
atom's inner scopes are constructed before the compound is
treated as an atom in the enclosing scope.

```
build_chain(scope S):
  for each direct child C of S in source order:
    if C is a pure op:
      skip
    if C is a leaf memory event:
      record C as a leaf atom of S
    if C is a compound scf.* op:
      for each inner region R of C:
        build_chain(R)
      compute C's effect summary by lifting all inner leaves'
      summaries
      record C as a compound atom of S

  apply the single-level chain rule of §2.5 at S, using the
  recorded atoms and their effect summaries

  for each compound atom C of S:
    apply C's per-op boundary translation to convert its
    abstract endpoints (struct_in, {incoming_C_P}) and
    (struct_done, {outgoing_C_P}) into concrete dataflow
    primitives that drive C's inner regions and gather their
    tails. The translation is specified per scf op in
    docs/spec-compiler-part-3-dfg.md §6.
```

Post-order is required because the enclosing scope's chain
construction queries the compound's effect summary (which
partitions it touches) before placing it in the per-`P` chains.

**Parallel-provenance compound atoms.** A compound atom carrying
parallel-provenance metadata (the temporary attributes planted
by the dependence builder; see §4) follows fork-join boundary
semantics. Each chunk's body is its own chain scope; chunks
share the compound's `incoming_P` for every touched partition
`P`; the compound's `outgoing_P` is the `dataflow.sync`
rendezvous of all chunk tails for `P`. No cross-iteration or
cross-chunk dependence edges are introduced inside the compound,
and no loop-carried memory state of §5 is created for partitions
inside it. Source-ordered loops without parallel provenance use
the loop-state ring of §5 instead.

### 2.7 Join Rules

The model fixes a hard rule for how multiple memory-order tails
are joined.

* Mutually exclusive dynamic alternatives must join with a
  selector-matched `dataflow.mux`. The selector is the same
  control bit that chose the path. Never `dataflow.sync`.
* Multiple required predecessors on the same dynamic path must
  join with `dataflow.sync`. Never `dataflow.mux`.
* A single predecessor is direct SSA forwarding. No primitive op
  is needed.
* No predecessor on the chain inside a scope means the scope's
  `incoming_P` is used directly; for the structural plane, the
  scope's structural entry token is used.
* If a dynamic path inside a scope contributes no atoms touching
  `P`, the path forwards the scope's `incoming_P` through the
  selector-matched join at the scope's tail.

A mixed predecessor set, with both same-path required tails and
mutually exclusive alternatives, is joined hierarchically: build
per-path tails by `dataflow.sync` over each path's required
predecessors, then `dataflow.mux` over the per-path tails. There
is no third primitive needed; the composition of `sync` and
selector-matched `mux` is the canonical form.

`dataflow.gate`, `dataflow.carry`, and `dataflow.demux` are
boundary translation and phase conversion primitives. They are
not a third tail-join primitive; the model never uses them at the
single-level join points specified above.

### 2.8 Boundary Translation Contract

A compound `scf.*` atom must satisfy a two-plane contract at its
boundary with the enclosing scope.

* **Structural plane.** The compound takes one
  structural-permission input and produces one
  structural-completion output. Inner regions are split,
  reentered, or muxed using `dataflow.demux` / `dataflow.mux` /
  `dataflow.carry` / `dataflow.gate` according to the SCF op's
  control shape.
* **Memory plane.** For every partition `P` transitively touched
  by the compound, the compound takes one `incoming_C_P` and
  produces one `outgoing_C_P`. Internal path-sensitive forwarding
  follows §2.5 and §2.7 rules. A partition not touched anywhere
  in the compound is not part of the compound's interface; the
  enclosing scope simply does not connect to it.

The shape that each `scf.*` op uses to satisfy this contract is
specified per op in `docs/spec-compiler-part-3-dfg.md` §6. The
following table summarizes the shape only; the per-op section
is the source of truth for SSA-level wiring.

| op | structural shape | per-partition shape |
|----|------------------|---------------------|
| `scf.execute_region` | pass-through | pass-through |
| `scf.if` | `demux %cond` at entry, `mux %cond` at done | `demux/forward` at entry, `mux %cond` at tail |
| `scf.index_switch` | N+1 way `demux` / `mux` | N+1 way `mux` at tail |
| `scf.forall` / `scf.parallel` | fork (shared struct_in); `sync` rendezvous of struct_dones | fork incoming_P; `sync` over chunk_tail_P |
| `scf.for` | `stream` + `carry` on loop-rwc + sentinel reset | `carry` on loop-rwc + per-P next_P feedback; mem_after_P from false-lane projection |
| `scf.while` | `carry` on `%cond` + `gate`; before K+1 / after K | `carry` on `%cond` + per-P feedback ring; `gate %cond` into after-region; false-lane projection to mem_after_P |

The per-op SSA wiring exists in
`docs/spec-compiler-part-3-dfg.md` §6 today only for the
structural plane in some templates. Memory-plane per-partition
wiring is added to those templates in subsequent commits of this
milestone; the contract here is the target each template must
satisfy.

### 2.9 Non-Goals

This compositional model is scoped to memory dependence inside
`dataflow.graph` regions. It does not, in this milestone:

* Define `!dataflow.thread_token` semantics or
  `LoomAsyncOpInterface` participation. Those are launch-side
  protocols specified by `docs/spec-compiler-part-3-dfg.md` §3
  and §5.4.1.
* Define `dataflow.map_info` direction enforcement or HostCore
  visibility. That is the boundary memory-effect summary
  specified by `docs/spec-compiler-part-3-dfg.md` §3 rule 4 and
  rule 7.
* Define `DeviceMappingAttrInterface` or thread-grid mapping
  semantics. Those are placement-side concerns.
* Replace the existing dataflow primitive op definitions for
  `dataflow.{stream, carry, invariant, gate, mux, demux, sync,
  constant}`. Those definitions are owned by
  `docs/spec-dataflow-part-1-streaming.md` and
  `docs/spec-dataflow-part-2-control.md`.
* Cross-graph partition identity. `dataflow.graph` is a leaf in
  the first milestone and partition ids are graph-local. Future
  graph-in-graph or split-graph designs would need an explicit
  child-block-arg to parent-operand alias-root mapping at the
  graph boundary; numeric ids alone are not enough. This is
  out of scope here.

The model assumes those other contracts are already enforced by
their respective sections.

## 3. Alias Oracle

The alias oracle answers a single question: do two memory effects
conflict? Conflict is symmetric and direction-free. The dependence
builder in §4 turns those answers into directed edges using program
order and structured-control-flow nesting. Effect-summary lift
across compound `scf.*` atoms is also driven from the oracle: per
§2.2 a compound's summary is the recursive union of its leaves'
summaries, and per §2.3 the unknown bucket `U` is what the oracle
reports for any leaf whose root walk leaves the recognized set. The
two interchangeable implementations below share the `MemAliasOracle`
interface; the C++ class signature and the pass that materializes
oracles per `dataflow.graph` live in
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
  `memref.reinterpret_cast`, `memref.transpose`. Each of these
  produces a memref that shares storage with its source by
  construction, so the walk peels them off without breaking aliasing.
* **Recognized terminal roots (stop, treat as known root).** The
  walk stops and records a known root at any of the following:
  `memref.alloca`, `memref.alloc`, `memref.get_global`, or a
  function-block argument with no defining op. These define a fresh
  storage identity for the purposes of the oracle (with the
  symbol-keyed adjustment for `memref.get_global` described below).
* **Unknown producer (stop, enter `U`).** Any other op that produces
  a memref-typed result terminates the walk without yielding a known
  root. The walk does not invent a new root from such an op; instead
  the access enters the conservative unknown bucket `U` defined in
  §2.3. This includes ops whose freshness or aliasing relationship
  is not statically guaranteed by the oracle, for example
  `bufferization.to_memref` (whose resulting memref may share
  storage with an existing buffer depending on the active
  bufferization strategy), an SSA value returned from a
  `func.call`, an `unrealized_conversion_cast` to a memref type,
  custom buffer reshape ops, and any future memref-producing op the
  oracle has not been taught about. This is the soundness rule
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
  conflict. This realizes the rule from §2.3 that a `U` effect
  may-aliases every known partition.

The first milestone uses any-memref same bucket as the
"compatible memref kind" predicate. Element type and shape rank are
intentionally NOT used as disjoint witnesses, because view-like ops
and bufferization paths can change element type or rank without
changing underlying storage. This is conservative and matches the
B.1 milestone direction; later milestones may refine compatibility
once the leaf walk and the bucket policy are tightened.

Bounds and offsets are not consulted at any point; the oracle is
intentionally storage-identity only.

### 3.2 MlirAaOracle

`MlirAaOracle` is default-on for the full lit suite. It wraps
`mlir::AliasAnalysis`, configured with whatever external
alias-analysis interfaces are registered, as a refinement of
`BasicSsaOracle`. It starts from the basic conflict set and removes
pairs that upstream MLIR AA proves `MustNotAlias`. `MayAlias` and
`MustAlias` keep the basic answer. Loads vs. loads still do not
conflict.

The refinement applies to leaf-pair queries only: when both
accesses are leaves visible to the dependence builder at the same
scope, MlirAaOracle may demote a basic-conflict pair to
non-conflicting if upstream AA proves them `MustNotAlias`. This
holds uniformly, including pairs where one or both sides come from
`U`: a specific unknown-producer op proven disjoint from a specific
known root drops out of the leaf-pair conflict set.

Effect-summary lift across compound `scf.*` atoms (§3.3) does not
benefit from this refinement. The summary records partition
identity by `BasicSsaOracle`'s classification: a compound's summary
contains `U` whenever any inner leaf is in `U`, regardless of
whether MlirAaOracle would have demoted some inner leaf-pair
conflicts. The conservative compound summary is intentional in
the first milestone; tightening it requires summary-level AA
support that is out of scope here.

### 3.3 Effect Summary Lift Rule

Per-leaf alias answers compose into compound atoms through the
effect-summary lift defined in §2.2 and §2.3:

* A compound atom's effect summary is the recursive union of its
  inner leaves' effect summaries. Membership in the summary is by
  partition identity: each known root the inner leaves touch
  contributes one entry, and any inner leaf that is in `U`
  contributes `U`. Read-only leaves still contribute their
  partition to the summary; the read-read suppression of §4 only
  prevents dependence edges, not summary membership.
* If any inner leaf is in `U`, the compound's summary contains `U`.
  The compound participates in `U`'s own per-partition chain at
  every scope that exposes `U`, plus, by the lift rule of §2.3,
  every known partition's per-`P` chain at every enclosing scope,
  until either the enclosing scope itself has no known partitions
  visible or the `dataflow.graph` boundary is reached.
* Frontier membership at the compound boundary uses
  `BasicSsaOracle`'s classification, not pair-level MlirAaOracle
  refinement (§3.2). If any leaf inside the compound is in `U`,
  the compound is wired into every enclosing known partition's
  chain, regardless of whether MlirAaOracle would have demoted a
  specific inner-vs-outer leaf pair to `MustNotAlias`. Pair-level
  refinement still applies inside the same scope where both
  leaves are visible, and inside the compound's own scope, but it
  does not change which partitions appear at the compound's
  boundary.
* Within a single scope, `U` participates in its own per-partition
  chain like any other partition: two writes in `U` (or a read and
  a write in `U`) form a same-partition dependence pair under §4
  and chain through `U`'s frontier. Read-read pairs in `U` still
  do not create dependence edges, consistent with §3.1's
  load-load rule.

The per-partition memory frontier of §2.4 then wires the compound
into the appropriate per-`P` chains using this summary; the
single-level chain rule of §2.5 and the join rules of §2.7 take
over once each scope's atom set is known.

Both oracles pass the same lit suite. They may produce different
`loom.mem_dep_preds` snapshots, because a stronger oracle can prove
that fewer ordered pairs conflict. The test suite is parameterized so
each relevant case is run twice, once per oracle.

## 4. Dependence Builder

`MemoryDependenceBuilder` produces the directed dep edge set that
the single-level chain rule of §2.5 consumes, structured around the
per-partition frontier of §2.4. Edges live inside one partition at
one chain scope.

### 4.1 Inputs and Outputs

The builder operates on one `dataflow.graph` body at a time. Its
inputs are the graph body's IR after parallel-SCF normalization (so
the leaf set is the same set §6 will see), a configured
`MemAliasOracle` per §3, and the partition assignment derived from
the §3.1 walk on each leaf's memref operand and lifted to compound
atoms by §3.3. Its outputs are the per-graph snapshot consumed by
§5 and §6: `loom.mem_dep_id` and `loom.mem_dep_preds` on each leaf
memory op, `loom.mem_loop_id` and `loom.mem_loop_states` on each
loop op carrying memory state (consumed only by §5), and the
parallel-provenance side data on cloned leaves and generated loops
(`loom.parallel_group`, `loom.parallel_chunk`,
`loom.parallel_chunks`, or an equivalent analysis side table).

### 4.2 Partition Assignment

Partition identity is graph-local and follows the §3 alias-oracle
contract. Each leaf is assigned exactly one partition by the §3.1
walk on its memref operand: a known root storage identity, or the
conservative bucket `U` when the walk leaves the recognized set.
Each compound `scf.*` atom inherits a set of touched partitions by
the §3.3 effect-summary lift: every known root any inner leaf
touches, plus `U` if any inner leaf is in `U`. A compound that
contains a `U` leaf additionally lifts into every known partition
visible at every enclosing scope, per the §2.3 lift rule. Numeric
partition ids are graph-local.

Two atoms in the same chain scope that share a partition are the
only direct candidates for a same-partition dep edge in that
scope. Cross-partition pairs and cross-scope pairs are never
direct edge candidates: cross-partition ordering is carried by
independent frontiers, and cross-scope ordering is carried by the
boundary translation of §2.8.

### 4.3 Per-Partition Edge Construction

For each chain scope `S` and each partition `P` in `S`'s transitive
partition set, the builder constructs the dep edge set over the
atom set `A_P(S)` defined in §2.5. Direction comes only from
program order and structured-control-flow nesting at `S`; alias is
symmetric and never defines a direction by itself.

* **Conflict gate.** An ordered pair `(p, o)` with `p` before `o`
  in `A_P(S)` is a dep candidate iff `MemAliasOracle` reports a
  non-`MustNotAlias` answer for the pair restricted to `P` AND the
  pair is not load-load. The query takes one of two forms:
  - **Leaf-vs-leaf, same chain scope `S`.** This is the direct
    leaf-pair query. `MlirAaOracle`'s refinement (§3.2) applies
    here; a basic-conflict pair may be demoted to non-conflicting
    if upstream AA proves `MustNotAlias`.
  - **Compound-involving (leaf-vs-compound or compound-vs-compound).**
    The pair conflicts in `P` iff at least one inner-leaf pair
    drawn from the contributing inner leaves on each side
    conflicts. Compound-boundary lift uses `BasicSsaOracle`'s
    classification per §3.3 only; `MlirAaOracle`'s leaf-pair
    refinement does not propagate into compound boundaries in
    this milestone, regardless of whether some inner-vs-outer
    leaf pair would have been demoted as a direct query.
* **Path-sensitive pruning.** Atoms in mutually exclusive branches
  do not need an edge between each other solely because they
  conflict; each branch's tail participates in the parent merge
  through a selector-matched `dataflow.mux` per §2.7. A
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
  enclosing scope. The compound's `outgoing_P` frontier remains
  the chunk-tail rendezvous of §2.6.
* **Loop-carried dep edges are real.** If an access in a later
  iteration of a source-ordered loop can conflict with an access
  in an earlier iteration, the loop's per-partition frontier must
  carry that ordering; §5 materializes the state ring from the
  partition membership and dep edges recorded here.
* **Transitive reduction.** The builder may remove transitively
  implied edges intra-partition and intra-scope, provided the
  remaining immediate predecessor set induces the same partial
  order. It must not collapse edges across partition or scope
  boundaries.

### 4.4 Snapshot Format

The builder plants a per-graph snapshot using only deterministic
graph-local integer ids:

* `loom.mem_dep_id = N` on each leaf memory op (`dataflow.load`
  and `dataflow.store`), in deterministic traversal order.
  `dataflow.thread.fence` is not part of the snapshot because it
  cannot appear in a graph body; see §2.2 and the verifier rule
  in `docs/spec-compiler-part-3-dfg.md` §9.
* `loom.mem_dep_preds = [P0, P1, ...]` on each leaf, listing the
  immediate dep predecessors in the leaf's partition after the
  transitive reduction of §4.3. Each entry references another
  leaf's `loom.mem_dep_id`; entries may name leaves at the same
  chain scope or leaves nested inside a sibling compound atom
  (see "Cross-scope predecessor resolution" below).
* `loom.mem_loop_id = L` on each loop op carrying memory state
  (consumed only by §5).
* `loom.mem_loop_states = [...]` on each such loop, referencing
  accesses only by `loom.mem_dep_id`.
* Parallel-provenance side data: `loom.parallel_group`,
  `loom.parallel_chunk`, `loom.parallel_chunks`, on cloned leaves
  and generated loops, stripped by `loom-finalize-dfg`.

Partition identity is not stored per leaf; each predecessor list
is implicitly scoped to the leaf's own partition and chain scope.
Only leaves carry `loom.mem_dep_id` in this milestone; compound
`scf.*` atoms still present in the graph do not get their own id.
Integer ids keep the snapshot stable across printing, parsing,
and in-place memory-op rewrites.

**Cross-scope predecessor resolution.** When a `loom.mem_dep_preds`
entry on leaf `L` at chain scope `S` names a leaf `L'` that lives
in a deeper chain scope `S'` (typically inside a sibling compound
atom `C ∈ A_P(S)`), §6 wiring resolves the predecessor through
`C`'s `outgoing_P` frontier per §2.5 and §2.6, not by using `L'`'s
`done` directly. The wiring walks each predecessor id back to its
defining leaf in the IR; if that leaf is at `L`'s own scope `S`,
the wiring uses the leaf's `done`; otherwise it walks up the IR
ancestor chain to the deepest ancestor that is a sibling of `L`
in `S` (such an ancestor is necessarily a compound atom `C` that
touches `P`), and uses that compound's `outgoing_P` frontier.
Multiple predecessor entries that resolve to the same compound's
`outgoing_P` deduplicate. This keeps the snapshot leaf-only while
preserving the §2.5 chain rule that a leaf's incoming frontier in
`P` includes the per-`P` tail of every sibling compound it depends
on, materialized through the boundary translation rules in
`docs/spec-compiler-part-3-dfg.md` §6.

## 5. Loop-Carried Memory State

### 5.1 Position in the Model

A loop-carried memory dependence is represented as hidden loop state,
not as an implicit property of the loop op. This section is the
loop-boundary instance of the per-partition memory frontier of §2.4:
a source-ordered loop compound atom carries its per-partition
incoming frontier across iterations through a hidden
`dataflow.carry`-driven state ring, exposes path-sensitive tails
through §2.7, and projects the loop-exit memory state out of the
loop through the loop's structural reset. The state must be visible
in dataflow primitives so graph scheduling and verification see the
same ordering as later hardware lowering.

This section applies to source-ordered loops only. Per §2.6 and §4.3,
parallel-provenance compound atoms (`scf.parallel`, or `scf.forall`
normalized to parallel-provenance `scf.for`) follow fork-join
semantics with no cross-iteration ordering and therefore have no
loop-carried memory state; their per-partition `outgoing_P` is the
chunk-tail rendezvous of §2.6.

### 5.2 Abstract Pattern

For a source-ordered loop compound `L`, the lowering instantiates a
per-partition state ring parameterized by:

* a **structural selector token** for `L` -- a control bit produced
  by `L`'s structural lowering that distinguishes continuing
  iterations from the exit cycle. Each loop op supplies its own
  selector; §5.3 and §5.4 give the concrete choices.
* the partition set `Π_L` -- the partitions for which `L` carries
  loop-state. Per §4.3, `P ∈ Π_L` iff some access in one dynamic
  iteration of `L` may conflict with some access in a later
  iteration of `L` in `P`. Read-read pairs alone never force a
  partition into `Π_L`. Partitions outside `Π_L` flow through `L`
  as ordinary per-partition frontiers (§2.4) without a state ring.

For each `P ∈ Π_L`, the lowering introduces a hidden `none`-typed
carry with four canonical tokens (names are descriptive;
implementations may choose different SSA names):

```
%mem_iter_P = carry %selector, %mem_init_P, %mem_next_P : none
```

* `%mem_init_P` is `L`'s `incoming_P` per §2.4: the memory-order
  frontier flowing into `L` for `P`, derived from the graph's
  `ctrl_in`, a pre-loop dependence tail, or an enclosing loop's
  per-`P` state.
* `%mem_iter_P` is the start-of-current-iteration memory state for
  `P`. Body-region accesses in `P` use `%mem_iter_P` as their
  scope's `incoming_P` and chain through it per §2.5.
* `%mem_next_P` is the end-of-current-iteration memory state for `P`,
  the input that feeds `%mem_iter_P` on the next iteration. It is
  the §2.5 / §2.7 join of the per-path body tails for `P`: a path
  that performs no access in `P` forwards `%mem_iter_P` unchanged;
  mutually exclusive paths are joined with a selector-matched
  `dataflow.mux` (never `sync`); same-path required tails are joined
  with `dataflow.sync` (never `mux`).
* `%mem_after_P` is the memory-order frontier flowing out of `L` for
  `P` -- equivalently, `L`'s `outgoing_P` per §2.4. It is the
  loop-exit projection of the final carried state, taken from the
  cycle that produces `L`'s structural reset; the zero-trip path
  forwards `%mem_init_P`. Post-loop accesses in `P` that may conflict
  with loop-body accesses use `%mem_after_P` as their predecessor.

Independent partitions get independent rings sharing only the
structural selector, so unrelated memrefs are not serialized.

### 5.3 scf.for Instantiation

`scf.for` parameterizes §5.2 with:

* selector = the loop-level rwc bit produced by the loop's
  `dataflow.stream`. The same rwc drives the structural carry, the
  iter-arg carries, and every per-`P` memory carry in `Π_L`.
* body region count = 1. The body is a single chain scope; §2.5
  applies inside it for each `P ∈ Π_L` with `%mem_iter_P` as
  `incoming_P`.
* phase rule: rwc=true on body iterations, rwc=false on the sentinel
  reset cycle that marks loop exit. The body's per-`P` tail feeds
  `%mem_next_P` on the true lane; the false lane projects the final
  carried state out as `%mem_after_P` (the same projection forwards
  `%mem_init_P` for the zero-trip case).

The structural rwc and the per-`P` memory carry are independent
state rings over the same selector; the structural plane never
aggregates the memory tails (§2.5 plane orthogonality).

### 5.4 scf.while Instantiation

`scf.while` parameterizes §5.2 with:

* selector = `%cond`, the `i1` value produced by `scf.condition` at
  the before-region terminator. The same `%cond` drives the
  structural carry, the structural `gate` into the after-region, and
  every per-`P` memory carry in `Π_L`.
* body region count = 2. The before-region and after-region are each
  their own chain scope. For K iterations the before-region executes
  K+1 times (the final false check still runs it) and the
  after-region executes K times.
* per-`P` flow inside one iteration:
  - `%mem_iter_P` enters the before-region as its `incoming_P`;
    `%before_tail_P` is the before-region's per-`P` tail, forwarding
    `%mem_iter_P` if the before-region performs no `P` access.
  - on the true lane, the after-region's `incoming_P` is
    `%after_in_P = gate %cond, %before_tail_P`; `%after_tail_P` is
    the after-region's per-`P` tail, forwarding `%after_in_P` if the
    after-region performs no `P` access.
  - `%mem_feedback_P = mux %cond, %before_tail_P, %after_tail_P`
    feeds `%mem_iter_P` on the next iteration. The operand order
    follows the §6 selector convention (lane 0 = false-lane =
    `%before_tail_P`, lane 1 = true-lane = `%after_tail_P`): on a
    true iteration the after-region tail is carried, and on the
    final false iteration the before-region tail is carried
    (because the final false iteration's before-region still ran).
* loop-exit projection: `%mem_after_P` is the false-lane projection
  of `%before_tail_P` from the final before-region execution. The
  zero-trip case (`%cond` false on the first check) reduces to the
  same projection over the single before-region run.

The after-region's structural rwc-style token (`%after_rwc`, if
exposed) is not on the memory critical path. Per §2.5 plane
orthogonality, after-region memory ops use `sync(struct_after,
%after_in_P)` for `ctrl`; the structural token provides only phase
permission, while `%after_in_P` carries the alias-aware ordering.

### 5.5 Nested Loops

Nested loops compose. From the enclosing loop's point of view, an
inner loop is an ordinary compound atom in the §2.5 chain: its
`%mem_after_P` is one event in the outer loop's per-`P` chain, and
its `%mem_init_P` is the outer loop's per-`P` frontier at the inner
loop's position. Each loop applies §5.2 to its own `Π`. For a
partition `P`:

* `P` is not touched anywhere inside the inner loop: the inner
  loop is not part of `P`'s chain, and the outer scope's frontier
  for `P` flows past the inner loop unchanged per §2.4.
* `P` is touched inside the inner loop but is not in `Π_inner`
  (no cross-iteration ordering is required, e.g. read-only body
  accesses): the inner loop participates in `P`'s chain through
  ordinary `incoming_P` / `outgoing_P` per §2.4 with no state
  ring; the per-iteration body just feeds tails through ordinary
  §2.5 chain construction.
* `P ∈ Π_inner`: the inner loop applies §5.2 to `P` with its own
  state ring; `%mem_init_P` is the outer-loop frontier at the
  inner loop's position, and `%mem_after_P` is one event in the
  outer-loop chain.

### 5.6 Parallel-Provenance Groups Inside Loops

A parallel-provenance compound atom nested inside a source-ordered
loop participates in the outer loop's per-`P` chain like any other
compound, per §2.6. The outer loop's per-`P` state at the group's
position is the group's `incoming_P`; the group's `outgoing_P` is
the chunk-tail rendezvous of §2.6 and feeds the outer loop's per-`P`
body tail. Chunks remain unordered, and no loop-carried state is
created for partitions internal to the parallel compound.

### 5.7 Snapshot for Loop-Carried State

The dependence builder records a per-loop plan in the
`loom.mem_loop_states` attribute on the source loop op. Each plan
parameterizes §5.2 for one loop and uses only deterministic
graph-local integer ids:

* graph-local loop id (also written as `loom.mem_loop_id`),
* per-partition records for every `P ∈ Π_L`, each containing:
  - partition id (graph-local),
  - member access ids (the leaves in `P` carried by the ring),
  - body-tail contributor access ids (the leaves whose `done`
    feeds `%mem_next_P` on some dynamic path through the loop
    body),
  - `%mem_after_P` consumers (the access ids that read
    `%mem_after_P` as a predecessor after the loop).

Path identity is not stored as separate snapshot fields. The
wiring in §6 reconstructs each dynamic path through the loop body
from the IR's structured-control-flow ancestry of every member
access: the path of a contributor leaf is the sequence of
`scf.if` / `scf.index_switch` branches and `scf.for` / `scf.while`
nestings between the loop op and the leaf. Paths whose IR
ancestry contains no member access in `P` carry no contributor
and forward `%mem_iter_P` per §5.2; paths with one or more
contributors join their tails by §2.7 (`dataflow.sync` for
same-path required tails, selector-matched `dataflow.mux` for
mutually exclusive tails).

Access ids reference `loom.mem_dep_id` values from §4.4. The loop-id
namespace is per-graph and separate from the `mem_dep_id` namespace;
both are graph-local and chosen deterministically. The plan does not
duplicate the type contract of `carry`, `mux`, `demux`, or `sync`;
the primitive op specs are authoritative for those.

### 5.8 Soundness Notes

Omitting a required loop-carried memory state is illegal: it lets
the lowering reorder accesses across iterations in a way the source
program does not permit, and graph verification will not catch the
omission because the resulting circuit is still well-typed.

Adding an extra conservative state ring is legal; it only
over-serializes. Tests should catch unnecessary serialization when
the active alias oracle proves partitions independent, so that
`Π_L` matches §4.3 rather than a coarser upper bound.

## 6. Token Wiring

The control-token wiring rule turns the compositional model of §2
and the dependence snapshot of §4 into explicit `ctrl` and `done`
SSA edges inside each `dataflow.graph`. It is the SSA-level
instantiation of the abstract join rules in §2.7.

* Each `dataflow.graph` has one explicit `ctrl_in` operand of type
  `none`, a matching leading block argument, one explicit leading
  `done_out` result of type `none`, and a matching leading yield
  operand. These are real SSA values even if the custom assembly
  format chooses to compress their spelling.
* For each load / store op `o` in the graph, its `ctrl` operand is
  `none`. The lowering first builds a ctrl source set:
  - immediate dependence predecessors contribute their `done` outputs;
  - loop-carried memory dependences contribute the relevant hidden
    `%mem_iter_P` or `%mem_after_P` state token described above;
  - a following access that depends on a completed parallel-provenance
    group contributes the group's tail token, which is the
    `dataflow.sync` rendezvous of all chunk tails.
* If `o`'s ctrl source set is empty, `o` uses `ctrl_in`. If the set
  has one value, `o` uses that value directly. If the set has multiple
  values, `o` uses output zero of a `dataflow.sync` rendezvous over
  all values in the set.
* The graph `done_out` value is output zero of a `dataflow.sync` over
  all `done` tokens of memory accesses with no immediate dependence
  successor.
* Multi-fanout of a single done is handled by SSA value reuse, not by
  an extra op.
* Read-read pairs have no dependence edge, even when they alias, so
  independent reads can be reordered freely.

## 7. References

* `docs/spec-compiler-part-3-dfg.md` -- IR boundary contracts, SCF
  flattening templates, and verifier invariants. The per-`scf.*`
  boundary translation rules that instantiate the contract in §2.8
  live in §6 of that document.
* `docs/spec-compiler-part-3-impl.md` -- pass pipeline, lit-test
  layout, milestone acceptance checklist, and maintenance plan.
  The `MemAliasOracle` C++ interface signature and the pass that
  materializes oracle instances per `dataflow.graph` are specified
  there.
* `docs/spec-compiler-part-3-placement-framework.md` -- common
  placement-partition framework. L2 graph placement decides which
  ScalarCore code becomes a `dataflow.graph`; this document
  specifies the chain model that runs inside each such graph.
* `docs/spec-dataflow-part-1-streaming.md` -- precise timing
  semantics for `dataflow.stream`, `dataflow.carry`,
  `dataflow.invariant`, and `dataflow.gate`.
* `docs/spec-dataflow-part-2-control.md` -- precise firing semantics
  for `dataflow.constant`, `dataflow.sync`, `dataflow.mux`, and
  `dataflow.demux`.
