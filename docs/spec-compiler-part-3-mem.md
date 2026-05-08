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
    memory access. `dataflow.thread.fence`, when it appears in a
    chain scope reachable by Part 3 lowering, is also a leaf
    event for chain purposes.
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

Two interchangeable oracle implementations share the `MemAliasOracle`
interface; the C++ class signature and the pass that materializes
oracles per `dataflow.graph` live in
`docs/spec-compiler-part-3-impl.md`. The interface answers only
the alias question framed in §1; the dependence builder in §4
turns its symmetric answers into directed edges. Effect-summary
lift across compound `scf.*` atoms (§2.2 and §2.8) reads from the
same interface; a compound's summary is the union of its leaves'
queries.

* `BasicSsaOracle` (default-on for fast iteration during development):
  - Walks each operand back through the chain
    `memref.cast | memref.subview | memref.view | memref.expand_shape |
     memref.collapse_shape` to a root SSA value.
  - Two accesses conflict iff their roots are equal and they are not
    both loads. Bounds and offsets are not consulted; the oracle is
    intentionally conservative.
* `MlirAaOracle` (default-on for the full lit suite):
  - Wraps `mlir::AliasAnalysis`, configured with whatever external
    alias-analysis interfaces are registered, as a refinement of
    `BasicSsaOracle`. It starts from the basic conflict set and removes
    pairs that upstream MLIR AA proves `MustNotAlias`. `MayAlias` and
    `MustAlias` keep the basic answer. Loads vs. loads still do not
    conflict.

Both oracles pass the same lit suite. They may produce different
`loom.mem_dep_preds` snapshots, because a stronger oracle can prove
that fewer ordered pairs conflict. The test suite is parameterized so
each relevant case is run twice, once per oracle.

## 4. Dependence Builder

`MemoryDependenceBuilder` consumes a `MemAliasOracle` and produces a
directed graph over memory accesses inside one `dataflow.graph`. The
graph it produces is the per-partition dep edge set used by §2.5;
this section specifies the per-graph snapshot the builder leaves on
the IR for downstream passes.

* The builder assigns deterministic integer ids in traversal order:
  `loom.mem_dep_id = N`.
* For each ordered pair `p` before `o`, the pair conflicts iff
  `query(p, o) != MustNotAlias` and the pair is not load-load.
  Conflicting ordered pairs become dependence candidates `p -> o`,
  subject to the structured-control and parallel-provenance rules
  below.
* Direction comes only from program order and structured-control-flow
  nesting. Alias answers are symmetric; they never define a direction.
* Dependences are path-sensitive to the extent exposed by structured
  control flow. Accesses in mutually exclusive branches do not need an
  edge between each other solely because they conflict; each branch's
  tail participates in the parent merge through a selector-matched
  `dataflow.mux`. A conservative implementation may serialize more
  when it cannot prove path exclusivity, but it must not omit a
  dependence that preserves an observable read/write or write/write
  order.
* Parallel-provenance groups are the exception to source-order loop
  dependence. Accesses in different logical iterations or different
  chunks of the same original `scf.parallel` are unordered by the
  source program. The builder must not add cross-iteration or
  cross-chunk dependence edges for such pairs solely because they may
  alias. It still records intra-iteration dependences and dependences
  between the parallel group and surrounding code.
* Loop-carried dependences are real dependences. If an access in a
  later iteration can conflict with an access in an earlier iteration,
  the lowered loop token structure must carry that ordering, rather
  than treating each iteration as independent.
* The builder may remove transitively implied edges. The snapshot
  records only immediate predecessors:
  `loom.mem_dep_preds = [P0, P1, ...]`.

The snapshot uses integer ids rather than operation references so it is
stable under printing, parsing, and later in-place memory-op rewrites.

## 5. Loop-Carried Memory State

A loop-carried memory dependence is represented as hidden loop state,
not as an implicit property of the loop op. The lowering must make the
state visible in dataflow primitives so graph scheduling, graph
verification, and later hardware lowering all see the same ordering.
This section is the loop-boundary instance of the per-partition
memory frontier defined in §2.4: an `scf.for` or `scf.while`
compound atom carries its per-partition incoming frontier across
iterations through a hidden `dataflow.carry`-driven state ring,
exposes path-sensitive tails through the join rules of §2.7, and
projects the loop-exit memory state through the false-lane of the
loop's structural reset.

This subsection applies to source-ordered loops such as user
`scf.for` and `scf.while`. It does not create loop-carried memory state
for loops generated from `scf.parallel` with parallel provenance.
Cross-iteration memory races in the original `scf.parallel` are source
undefined behavior or unspecified behavior, not an ordering obligation
for Loom. The generated loops still keep their ordinary intra-iteration
memory dependences and their incoming / outgoing group-tail
dependences.

For each structured loop `L`, the dependence builder computes memory
partitions inside `L`:

* The initial partitioning key is the alias root used by the active
  oracle. A more precise implementation may split or merge by the
  conflict graph's strongly connected components, but it must be
  conservative: two accesses that may need cross-iteration ordering
  must appear in at least one common partition.
* A partition needs loop-carried state when an access in one dynamic
  iteration can conflict with an access in a later dynamic iteration.
  Read-read pairs never force such a partition by themselves.
  Parallel-provenance loops are excluded from this rule; they use the
  group-tail rule above instead of hidden loop-carried state.
* Each partition that needs loop-carried state gets one deterministic
  partition id unique within the graph and one hidden `none`-typed
  carry in the lowered loop. Independent partitions get independent
  carries so unrelated memrefs are not serialized.

The canonical state names below are descriptive; implementations may
choose different SSA names.

```
%mem_iter_P = carry %rwc, %mem_init_P, %mem_next_P : none
```

* `%mem_init_P` is the dominating memory-order token before the first
  dynamic iteration of `L` for partition `P`. It is derived from the
  graph `ctrl_in` token, a pre-loop dependence tail, or the enclosing
  loop's memory-state token.
* `%mem_iter_P` is the start-of-current-iteration memory state. Any
  access in partition `P` that has a loop-carried predecessor syncs
  with this token in addition to its ordinary intra-iteration
  dependence predecessors.
* `%mem_next_P` is the end-of-current-iteration memory state. It is
  built from the done tokens of accesses in partition `P` whose
  completion must precede the next dynamic iteration. If a dynamic
  path through the loop body performs no access in `P`, that path
  forwards `%mem_iter_P`. Mutually exclusive tails are joined with the
  same selector that chose the path; they are never joined with
  `sync`.
* `%mem_after_P` is the memory state after the loop. The zero-trip
  path forwards `%mem_init_P`. The nonzero path is the final carried
  state. Post-loop accesses in partition `P` use `%mem_after_P` as
  their predecessor when they may conflict with loop-body accesses.

`scf.for` uses the `stream`-produced loop-level rwc bit for the
hidden memory-state carry, following the same loop-phase rule as
iter_args. The true lane is the per-iteration body memory state; the
false lane is the loop-exit memory state. The body tail feeds
`%mem_next_P`, and the loop-exit state handles both zero-trip and
nonzero execution.

`scf.while` has two regions and therefore two relevant memory tails.
The before-region executes on both true and false condition checks.
The false path exits the loop with the before-region tail. The true
path continues through the after-region, and the after-region tail
feeds `%mem_next_P` for the next iteration.

Nested loops are treated compositionally. An inner loop's
`%mem_after_P` is an ordinary memory-order event in the enclosing
loop's partition. If the same alias root participates in both loops,
the outer loop state gates the inner loop entry and the inner loop
exit feeds the outer loop's body tail.

Parallel-provenance groups nested inside a source-ordered loop follow
the same compositional rule at the group boundary. The outer loop's
memory state gates the parallel group entry when the group may touch
the same partition, and the group's tail token feeds the outer loop's
body tail. The chunks inside that group remain unordered with respect
to each other.

The loop-state plan stored by the memory-dependence builder is the
`loom.mem_loop_states` attribute on the source loop. Each record uses
only deterministic integer ids:

* loop id,
* partition id,
* member memory-access ids,
* access ids that define the per-path `%mem_next_P` tails,
* access ids that consume `%mem_after_P` after the loop.

This avoids operation-reference attributes and keeps the snapshot
stable across printing and parsing. The plan intentionally does not
duplicate the type contract of `carry`, `mux`, `demux`, or `sync`; the
primitive op definitions and the dataflow op semantics specs are the
single source of truth for which types those ops accept and when they
fire.

Omitting a required loop-carried memory state is illegal. Adding an
extra conservative state is legal for correctness, but tests should
catch it when it serializes partitions that the active oracle proves
independent.

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
