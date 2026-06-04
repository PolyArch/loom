# Loom Compiler Part 3 Implementation Notes

This document collects the engineering details that support the Part 3
(SCF-to-DFG) front-end design: the pass pipeline, the lit-test layout,
the milestone acceptance checklist, and the extension points.

Part 3 itself (`docs/spec-compiler-part-3-dfg.md`) holds the
first-principles IR content -- boundary contracts, SCF flattening
templates, and verifier invariants. The memory-dependence model
that this part lowers to is owned by
`docs/spec-compiler-part-3-mem.md`. Material in this file exists so
that one canonical implementation is pinned; readers who only need
the design contract can skip this file.
References below use Part 3 section names rather than numeric indices,
so that Part 3 can renumber without forcing edits here.

## 1. Lowering Pipeline

The scf-to-dfg lowering is implemented as an ordered sequence of MLIR
passes registered together under `loom-lower-scf-to-dfg`. Each pass is
small, has its own lit tests, and may be run individually for
debugging.

```
[1] loom-normalize-acc-regions
[2] loom-materialize-forall-aggregation
[3] loom-classify-thread-regions
[4] loom-promote-map-info
[5] loom-build-thread-skeleton
[6] loom-prepare-scalarcore-calls
[7] loom-extract-graph-regions
[8] loom-build-memory-dependencies
[9] loom-lower-scf-to-dfg-bodies
[10] loom-finalize-dfg
```

The pass numbering is a sequencing convenience only; downstream
documentation never refers to the numeric position.

### 1.1 `loom-normalize-acc-regions`

* Consumes the `loom.acc_region` ops produced by Part 2. The pass
  never treats a whole `func.func` body as an accelerator region by
  default.
* Host code outside `loom.acc_region` is preserved and is not scanned
  for graph extraction or thread promotion.
* The optional `wrap-standalone-kernel` mode is a test and bring-up
  convenience for hand-written Part 3 inputs. When enabled, a selected
  `func.func` body with no `loom.acc_region` is wrapped in one
  synthetic `loom.acc_region` before the rest of this pipeline runs.
  The option is disabled by default and must not be used by the
  LLVM-to-SCF pipeline for ordinary host programs.
* Verifies the Part 2 boundary contract before lowering: the
  accelerator region has no direct data results, all values crossing
  the boundary are explicit operands or memory effects, and the region
  body is structured enough for the SCF lowering rules in Part 3.
* Records the region-local default mapping used if a scalar-only
  accelerator region must be normalized into a 1x1 mapped forall. If
  Part 2 did not provide a mapping policy, this part uses a single
  spatial grid point as the conservative default.

### 1.2 `loom-materialize-forall-aggregation`

* Runs inside accelerator regions before any thread promotion or graph
  extraction. Its job is to convert aggregation-form `scf.forall` into
  effect-form `scf.forall` using explicit destination buffers.
* Handles `scf.forall` with `shared_outs`, op results, or non-empty
  `scf.forall.in_parallel`. The canonical first-milestone combining
  op is `tensor.parallel_insert_slice`; support for additional
  `ParallelCombiningOpInterface` implementations may be added later
  by extending this pass.
* The pass may use upstream one-shot bufferization infrastructure, but
  its output contract is Loom-specific: after it runs, every
  `scf.forall` inside an accelerator region has no `shared_outs`, no
  op results, and an empty `scf.forall.in_parallel` terminator. Tensor
  results that are still needed by surrounding code are represented
  through the materialized destination buffer and the necessary
  `bufferization.to_tensor` or equivalent bridge ops.
* Nested aggregation-form forall is materialized recursively. If an
  inner forall's `shared_out` is a slice, extract, or other view of an
  outer materialized destination buffer, the inner destination is the
  corresponding `memref.subview` or equivalent buffer view. Inner
  combining actions then become explicit writes to that view. If the
  nested destination relationship cannot be represented with ordinary
  buffer/view operations, the pass emits a diagnostic instead of
  inventing a new aggregation protocol.
* The pass preserves forall bounds, steps, induction variables, and
  `mapping` attributes. The newly materialized destination buffers are
  ordinary SSA values; if such a buffer crosses a mapped forall
  boundary, `loom-promote-map-info` handles it exactly like any other
  memref value.
* This pass must never silently drop combining actions. If an
  aggregation-form forall cannot be materialized, it emits a clear
  diagnostic and the pipeline stops.

### 1.3 `loom-classify-thread-regions`

* Walks every `loom.acc_region` body in the module. It does not walk
  ordinary host code outside accelerator regions.
* Identifies every `scf.forall` whose `mapping` attribute is
  non-empty, contains at least one `DeviceMappingAttrInterface`
  element recognizable as a `#loom.thread_axis<...>` instance, AND
  contains no foreign `DeviceMappingAttrInterface` element. A
  `mapping` array that mixes
  Loom-recognized entries with at least one foreign entry is
  rejected with a diagnostic at this pass (per
  `docs/spec-compiler-part-3-dfg.md` §4 Mapping attribute rules);
  Part 2 or an earlier Part 3 pass must remove or translate the
  foreign entries before promotion can run.
* Marks each Loom-only mapped forall with a temporary attribute
  `loom.thread_promotion = unit`. Nested mapped foralls are marked
  individually; the relative nesting order is preserved by IR
  traversal order.
* If an accelerator region contains zero mapped foralls, the pass adds
  a synthetic outermost `scf.forall (%i) in (1) { ... }` with the
  region-local default mapping from `loom-normalize-acc-regions`,
  wrapping the accelerator region body. This guarantees every explicit
  accelerator region lowers to at least one `dataflow.thread`
  definition + launch pair without turning unselected host code
  into AccCore code.
* The synthetic 1x1 case is allowed only for an explicit
  `loom.acc_region`. It is the implementation path for a selected
  scalar-only accelerator region, not a recovery path for failed graph
  placement in ordinary host code.

### 1.4 `loom-promote-map-info`

* For each marked forall, computes the set of values defined outside
  the forall body and used inside it.
* For outermost foralls created from a scalar-only `loom.acc_region`,
  the surrounding values are the explicit accelerator-region boundary
  operands, not arbitrary values captured from the enclosing
  `func.func`.
* For every `memref<...>` value or partitioned-data handle that crosses
  the thread boundary, inserts a `dataflow.map_info ...
  direction=tofrom` immediately outside the forall. Scalar values do
  not need `map_info`; they become by-value launch operands.
* The pass records a deterministic boundary-operand list on the marked
  forall: mapped data handles first in SSA discovery order, then
  scalar launch operands in SSA discovery order. The next pass uses
  this list to build an isolated `dataflow.thread` definition and
  rewrite body uses to the corresponding entry block arguments.
* Future optimizer passes can refine `tofrom` to `to` or `from` based
  on read/write effect summaries; this pass is intentionally
  conservative.

### 1.5 `loom-build-thread-skeleton`

* Replaces every marked `scf.forall` with a **`dataflow.thread`
  definition** at module scope plus a **`dataflow.thread.launch`** at
  the original forall site. The def carries the grid bounds, mapping,
  body region, and terminator coming from the forall; the launch
  carries the per-instance async deps, dynamic-grid values, and body
  operands.
* The pass is deterministic: it picks the def's symbol name as
  `t_<funcSym>_<seq>`, where `<funcSym>` is the enclosing
  `func.func` symbol name and `<seq>` is the zero-based index of
  the marked forall inside that function in source-traversal order.
  Symbol-grammar-illegal characters in `<funcSym>` are sanitized
  (replaced with `_`), and a deterministic disambiguating suffix is
  appended if the resulting name collides with another module-scope
  symbol. The launch's `callee` field references this symbol.
* Requires effect-form forall input. If a marked forall still has
  `shared_outs`, op results, or a non-empty `scf.forall.in_parallel`,
  the pass emits a diagnostic pointing to
  `loom-materialize-forall-aggregation`.
* The thread definition's body entry block uses the layout
  `(args_*, thread_ctrl, iv_*)` per
  `docs/spec-compiler-part-3-dfg.md` §5.4.1: the leading `N` block
  args mirror `function_type.inputs` exactly, then one
  `thread_ctrl : none`, then one `index`-typed iv per grid dim
  (in source-dim order). This ordering preserves the upstream
  `FunctionOpInterface` invariant.
* Body operands at the launch site keep the deterministic ordering
  computed by `loom-promote-map-info`. Their types form the def's
  `function_type.inputs`, position-by-position. Memref-like launch
  operands are the direct SSA results of `dataflow.map_info` ops
  (verifier-enforced); the def's matching block arg has the same
  memref type. Scalar launch operands keep their original type.
  The pass rewrites every in-body use of a surrounding SSA value
  to the corresponding entry-block arg before the verifier sees
  the isolated def.
* The empty `scf.forall.in_parallel` terminator is replaced with an
  empty `dataflow.thread.yield` inside the def body. No tensor
  aggregation action is dropped by this pass; all such actions must
  already have been materialized as explicit memory effects.
* `scf.forall` is an implicit synchronization point. The launch
  produces an `Optional<!dataflow.thread_token>` result `%t`; the
  replacement must make continuation ordering explicit on `%t`. The
  pass applies the following mechanical rule, in order:
  1. Compute `%t = produced thread launch token`.
  2. Identify `next_op` as the op that immediately follows the
     original `scf.forall` in source order at its continuation
     point. "Immediately follows" means structurally adjacent in the
     parent block, with no intervening op (no `arith.*`, no
     `func.call`, no other op of any kind sits between the original
     `scf.forall` and `next_op`).
  3. The pass omits the explicit sync op iff both of the following
     hold:
     - (A) `next_op` implements `LoomAsyncOpInterface`.
     - (B) On exit from this lowering, `next_op`'s
       `asyncDependencies` operand list contains `%t`.

     The two conditions are independent. Structural adjacency
     without `%t` use is not enough; a use of `%t` somewhere later
     in the program but not on the immediately following op is not
     enough either.
  4. If either condition fails, or cannot be verified at lowering
     time, the pass inserts `dataflow.thread.wait %t` at the
     original continuation point, before any continuation op.
     Subsequent ops keep their source-order position.

  The fallback to `dataflow.thread.wait` is the spec contract, not
  an optimization opportunity. The wait carries the default-resource
  memory barrier introduced by `docs/spec-compiler-part-3-dfg.md`
  §3 Constitutional Rule 8, so subsequent host or parent-context
  ops cannot be reordered to before the synchronization.
* Once every marked forall inside a `loom.acc_region` has been
  replaced by a def + launch pair (def at module scope, launch in
  place), the temporary accelerator-region wrapper is erased and
  its body is spliced back at the original host program point.
  Because `loom.acc_region` is `IsolatedFromAbove`, its body uses
  entry block arguments rather than the surrounding host SSA values,
  so the erasure step must rewrite every use of body block argument
  `i` to the corresponding `loom.acc_region.boundaryOperands[i]` in
  the enclosing host scope before the wrapper is removed. The
  substitution is mechanical (positional, type-equal by
  `IsolatedFromAbove` invariant) and runs over the entire body. No
  `loom.acc_region` remains after this pass.

### 1.6 `loom-prepare-scalarcore-calls`

* Runs after thread skeleton construction and before graph extraction.
* Inspects every `func.call` reachable inside a `dataflow.thread`
  definition's body. The call is a ScalarCore operation, not a
  SpatialCore operation.
* If the callee contains operations that must be graph-extracted in
  the caller's thread definition context, the pass inlines or
  specializes the callee into that thread definition before graph
  extraction. This keeps the prospective `dataflow.graph` definition
  reachable through a `dataflow.graph.launch` lexically inside the
  active thread definition's body without requiring the thread
  definition to become a symbol table or carry an implicit function
  context.
* A non-inlined `func.call` may remain only if the callee is
  ScalarCore-legal and graph-free after this preparation. Such calls
  are treated as ScalarCore side-effecting operations by later passes.
* Unsupported calls reachable from a thread definition's body produce
  a diagnostic. The first implementation may require all non-trivial
  ScalarCore calls to be inlined.

### 1.7 `loom-extract-graph-regions`

* Instantiates the L2 graph-placement problem from
  `docs/spec-compiler-part-3-placement-framework.md`. The input is
  each innermost `dataflow.thread` definition's body after
  ScalarCore call preparation. The placement unit is the
  `dataflow.graph` definition (paired with a
  `dataflow.graph.launch` at the cut site); residual code stays in
  the surrounding ScalarCore thread definition's body.
* For each chosen run, the pass emits two ops: a
  **`dataflow.graph` definition** at module scope, and a
  **`dataflow.graph.launch`** at the original cut site. The def's
  symbol name is `g_<threadSym>_<seq>`, where `<threadSym>` is the
  enclosing thread definition's symbol name and `<seq>` is the
  zero-based source-order index of the cut inside that thread
  definition. Symbol-grammar-illegal characters are sanitized as in
  §1.5 above. The launch's `callee` field references this symbol.
* The pass is structured as admission constraints, a cost model,
  and an exploration policy. Admission constraints are correctness
  rules: admitted graph definition bodies must satisfy Part 3's
  graph verifier contract, `IsolatedFromAbove`, explicit
  ctrl_in/done_out wiring (which now lives inside `function_type`),
  and effect visibility. The cost model ranks only legal
  partitions. The exploration policy decides which legal partition
  to materialize.
* The baseline cost model is deterministic and trivial: it prefers
  the partition produced by the baseline source-order greedy policy,
  with stable tie-breaking by lexical operation order. This cost
  model is intentionally not a performance model. Future policies
  may use graph launch count, reconfiguration estimates, graph-
  result traffic, fabric pressure, or profile data without changing
  the IR contract.
* The baseline policy walks the direct operations of the thread
  definition's body in source order. A graph run opens at the first
  graph-admissible op, accumulates following graph-admissible ops,
  closes at a required cut, and may open again at the next
  graph-admissible op. A run is materialized as a (def + launch)
  pair only when it contains a baseline graph anchor. Baseline
  anchors are memory accesses that will become
  `dataflow.load` / `dataflow.store`, or structured-control ops
  whose nested regions contain such accesses. Pure-only admissible
  runs may remain ScalarCore code in the baseline policy; a future
  policy may choose to place them in graphs if its cost model
  prefers that.
* Bridge to the framework vocabulary: a graph run is the L2
  candidate-partition unit produced by the baseline policy, the
  (def + launch) pair it materializes is the placement unit, and
  the graph-admissible / required-cut / graph-anchor predicates
  together encode this layer's admission constraints. Future
  cost-model and exploration-policy work should keep these terms
  aligned with
  `docs/spec-compiler-part-3-placement-framework.md` §3-§5.
* "Graph-admissible" is not inferred from the `Pure` trait alone.
  `dataflow.map_info` and the partitioned-data ops in
  `docs/spec-compiler-part-4-partitioned-data.md` are also `Pure`, but they
  are boundary-only or thread-body-only and are intentionally
  excluded from graph definition bodies. The baseline admitted set
  is: `arith.*`, `math.*`, allowed LLVM computation ops,
  `dataflow.{stream, carry, invariant, gate, mux, demux, sync,
  constant}`, `memref.{load, store}`, and supported structured
  `scf.*` ops whose nested regions recursively satisfy the same
  graph admission rules.
* Required cuts close the current graph run and remain in the
  ScalarCore thread definition's body. The required cuts are
  `dataflow.thread.fence`, non-inlined `func.call`,
  `dataflow.map_info`, partitioned-data query or layout ops,
  graph-illegal ops, and the parent terminator. The policy also
  cuts before any structured-control op whose nested regions
  contain a required cut. Unsupported required SpatialCore
  placement is a diagnostic; optional unadmitted code stays
  ScalarCore.
* `dataflow.thread.launch` is not merely a cut: per
  `docs/spec-compiler-part-3-dfg.md` §3 Constitutional Rule 2,
  thread hierarchy is strictly layered. Whenever the thread
  definition's body the pass is processing has any direct
  `dataflow.thread.launch` op, the pass emits no
  `dataflow.graph.launch` at that level at all. Graph extraction
  runs only on innermost executable thread bodies: bodies with no
  direct child `dataflow.thread.launch`, and with graph-admissible
  code or scalar-only residual code. This matches the placement
  framework's L2 graph-placement rule in
  `docs/spec-compiler-part-3-placement-framework.md` §7.
* Connectedness is not part of the baseline admission rule. If two
  memory-access clusters are adjacent in source order and separated
  only by graph-admissible compute, they are placed in the same
  graph run. Future policies may split such a run for resource or
  reconfiguration reasons, but the baseline output is mechanical
  and deterministic.
* Within a single graph definition, the `scf.*` control-flow ops
  appear as unflattened children that the next-but-one pass will
  lower into `dataflow` token primitives. The extraction pass does
  not modify control-flow shape; it only moves ops into the
  module-scope def's body and supplies the def's
  `function_type = (none, T0..TN) -> (none, R0..RM)`, the matching
  entry block layout `(%ctrl_in : none, %arg_0..%arg_N)`, and the
  matching `dataflow.yield (%done_out : none, %r_0..%r_M)`. The
  per-launch ctrl/done plumbing lives on the launch op:
  `(%done, %r) = dataflow.graph.launch @sym(%ctrl, %args) : ...`.
  Graph-to-graph ordering is represented by ordinary SSA use of
  one launch's `done_out` result as another launch's `ctrl_in`
  operand.
* A graph launch with no graph-launch predecessor and no explicit
  ScalarCore fence predecessor uses the enclosing thread
  definition's `thread_ctrl` block argument as its `ctrl_in`
  operand. If ScalarCore work must complete before a graph launch
  starts, the lowering inserts or preserves a
  `dataflow.thread.fence` at that program point and uses the fence
  result as the launch's `ctrl_in` operand.
* `dataflow.graph.launch` ops are ScalarCore launch points for
  SpatialCore work. The `ctrl_in` operand is an additional graph-
  level start dependency; it is not the only sequencing rule. The
  graph launch also occurs at the launch op's position in the
  enclosing ScalarCore program. SpatialCore completion becomes
  visible to later ScalarCore code only when the launch's
  `done_out` is consumed by `dataflow.thread.fence`.
* Because `dataflow.graph` (def) is `IsolatedFromAbove`, the
  extraction pass also computes every surrounding value used by
  the run's body and materializes it as an explicit launch operand
  paired with a matching def entry block argument. Values produced
  inside the run and used outside it are materialized as explicit
  launch results paired with `dataflow.yield` operands.

### 1.8 `loom-build-memory-dependencies`

* Builds the per-graph memory-dependence snapshot consumed by body
  lowering. A memory access means `memref.load` / `memref.store`
  before rewrite and `dataflow.load` / `dataflow.store` after rewrite.
  The compositional chain model the snapshot feeds, the alias-oracle
  contract, the dependence builder rules, the loop-carried memory
  state pattern, and the SSA-level token wiring are specified in
  `docs/spec-compiler-part-3-mem.md`. This section pins the
  implementing pass; the model itself is owned by that document.
  Although the pass name focuses on dependence construction, this pass
  also performs the final parallel-SCF normalization described below.
  Keeping both tasks together ensures memory accesses cloned by
  normalization immediately enter the same deterministic id assignment.
* Before assigning memory-access ids, performs the remaining
  parallel-SCF normalization inside each `dataflow.graph` definition's
  body: effect-form `scf.forall` with an empty mapping is normalized
  to `scf.parallel`, and `scf.parallel` is normalized to one or more
  `scf.for` loop nests plus any required reduction-merge `scf.if`
  ops. This is the point where parallel provenance is planted on the
  generated loop nests. Every cloned memory access receives its own
  deterministic `loom.mem_dep_id`.
* Chunk-bound arithmetic introduced by parallel-SCF normalization,
  such as trip-count computation and per-chunk lower / upper bounds,
  is materialized inside the same `dataflow.graph` definition's body
  that contained the original `scf.parallel`. These new ops use only
  graph-local SSA values and must satisfy the existing graph-body
  whitelist for pure computation ops.
* A user-written `scf.parallel` with a non-empty `mapping` attribute is
  rejected here. Mapping is honored only on `scf.forall` in this
  milestone.
* Materializes a `MemAliasOracle` instance for each `dataflow.graph`
  definition's body. The oracle is a per-graph conflict oracle; the
  rest of the lowering reads only through it, never directly off
  MLIR's analysis manager.
* The interface is:
  ```
  enum class AliasAnswer { MustNotAlias, MayAlias, MustAlias };
  class MemAliasOracle {
  public:
    virtual ~MemAliasOracle();
    virtual AliasAnswer query(::mlir::Operation *a,
                              ::mlir::Operation *b) = 0;
  };
  ```
* Two implementations ship in the same library and are selectable by
  the pass option `--mem-alias=basic|mlir-aa`. The default in
  milestone 1 is `--mem-alias=basic`; it drives the full lit suite.
  `--mem-alias=mlir-aa` is exercised on a representative differential
  subset (see `Testing Strategy` below). Future front-end passes that
  consume alias information must not implicitly select `mlir-aa`;
  they opt in through an explicit pass option, and they obtain alias
  answers only through the `MemAliasOracle` interface per
  `docs/spec-compiler-part-3-dfg.md` §3 Rule 5.
  - `BasicSsaOracle`: an SSA-source walk that recognizes a fixed set
    of view-like and terminal memref ops; any other memref-producing
    op enters the conservative unknown bucket `U` per
    `docs/spec-compiler-part-3-mem.md` §3. The recognized view-like
    ops are `memref.cast`, `memref.subview`, `memref.view`,
    `memref.expand_shape`, `memref.collapse_shape`,
    `memref.reinterpret_cast`, `memref.transpose`,
    `dataflow.partition_layout`, and `dataflow.map_info` (both
    same-type view-like producers per
    `docs/spec-compiler-part-4-partitioned-data.md` and
    `docs/spec-compiler-part-3-dfg.md` §5.4.6); the walk peels each
    into its source operand. The recognized terminal roots are
    `memref.alloca`, `memref.alloc`, `memref.get_global`, and
    function-block arguments. Entry-block arguments of
    `IsolatedFromAbove` ops (`dataflow.graph` def, `dataflow.thread`
    def) are not terminal: the walk continues on the matching
    launch-side operand in the enclosing scope (resolved via the
    callee symbol when crossing a `dataflow.thread.launch` or
    `dataflow.graph.launch`) per
    `docs/spec-compiler-part-3-mem.md` §3.1, so that storage
    identity is preserved across the boundary. Other memref
    producers, including
    `bufferization.to_memref`, `unrealized_conversion_cast`, and
    memref-returning `func.call`, enter `U`; their freshness or
    aliasing relationship is not statically guaranteed. Two accesses
    with known roots conflict iff their roots have the same storage
    identity and the pair is not load-load. The storage identity
    is the SSA value for `memref.alloca`, `memref.alloc`, and
    block-args, and the referenced global symbol for
    `memref.get_global` (so two distinct `memref.get_global @g` ops
    correctly share storage identity). Distinct storage identities
    default to disjoint. Any access in `U` may-aliases every other
    access of a compatible memref kind in scope, regardless of root,
    with the same load-load exception.
  - `MlirAaOracle`: forwards to `mlir::AliasAnalysis` from
    `mlir/Analysis/AliasAnalysis.h` as a refinement of
    `BasicSsaOracle`. It starts from the basic conflict set and removes
    pairs that upstream MLIR AA proves `MustNotAlias`. The refinement
    applies to leaf-pair queries only, uniformly across pairs where
    one or both sides come from `U`: a specific unknown-producer op
    proven disjoint from a specific known root or another unknown
    producer drops out of the leaf-pair conflict set. Effect-summary
    lift across compound `scf.*` atoms uses `BasicSsaOracle`'s
    classification only and does not benefit from this refinement
    (see `docs/spec-compiler-part-3-mem.md` §3.3). When upstream AA
    cannot prove anything stronger, the oracle behaves exactly like
    the basic oracle.
* Runs a `MemoryDependenceBuilder` after alias queries are available.
  The builder visits memory accesses in deterministic program order.
  Alias answers are symmetric and never define direction by
  themselves; direction always comes from program order plus the
  enclosing structured-control-flow path. The builder constructs dep
  edges per partition, where the partition is the alias bucket key
  defined by `docs/spec-compiler-part-3-mem.md` §3 (a known root
  storage identity from the §3.1 walk, or the conservative bucket
  `U`). Two atoms in the same chain scope and same partition are the
  only direct candidates for a dep edge; cross-partition and
  cross-scope ordering is carried by per-partition frontiers and by
  per-`scf.*` boundary translation, never by an edge.
* Compound `scf.*` ops still inside a `dataflow.graph` definition's
  body at this point in the pipeline participate as compound atoms
  via the §3.3 effect-summary lift. The builder queries the alias oracle on inner leaves
  as the unit of conflict: a compound conflicts with a leaf in
  partition `P` iff at least one inner leaf the compound contributes
  to `P` conflicts with the outer leaf, and two compounds conflict
  in `P` iff at least one inner-vs-inner pair on each side
  conflicts. Compound-boundary lift uses `BasicSsaOracle`'s
  classification; the `MlirAaOracle` leaf-pair refinement does not
  propagate into the lift. Path-sensitive pruning, the parallel-
  provenance exception, the loop-carried real-edge rule, and the
  optional transitive reduction follow
  `docs/spec-compiler-part-3-mem.md` §4.3.
* The builder consumes parallel provenance from generated loops. For
  accesses in different logical iterations or different chunks of the
  same original `scf.parallel`, it must not create a dependence edge
  solely from source order or alias conflict. Intra-iteration program
  order is still preserved. Dependences from code before the parallel
  group to each chunk, and from the whole group to following code, are
  still modeled when aliasing requires them.
* Each parallel-provenance group has one group tail token plan. The
  group tail is the rendezvous of all chunk tail tokens. A later memory
  access that depends on the completed memory effects of the original
  `scf.parallel` uses this group tail as its predecessor.
* For structured loops, the builder also records per-loop memory
  plans. Each plan is keyed by a deterministic loop id and a memory
  partition id, and references memory accesses only by integer ids.
  Each partition record has a kind: `carried` for partitions
  requiring cross-iteration ordering (lowered to one hidden `none`
  state-ring carry in `loom-lower-scf-to-dfg-bodies`, per
  `docs/spec-compiler-part-3-mem.md` §5.2 abstract pattern), or
  `completion` for partitions touched in the body but not requiring
  cross-iteration ordering (lowered to one hidden completion-only
  carry that aggregates per-iteration body-tail tokens into the
  loop's `outgoing_P`, per `docs/spec-compiler-part-3-mem.md` §5.2
  touched-but-not-carried case). Both record kinds are pinned in
  the snapshot so `loom-lower-scf-to-dfg-bodies` does not need to
  re-analyze.
* The pass leaves a stable IR snapshot so subsequent passes need no
  re-analysis: each leaf memory access gets `loom.mem_dep_id = N` and
  `loom.mem_dep_preds = [P0, P1, ...]`, where `N` and every `P*` are
  deterministic integer ids inside the graph definition's body.
  Only leaf memory accesses (`memref.load` / `memref.store` before
  rewrite, `dataflow.load` / `dataflow.store` after rewrite) carry
  `loom.mem_dep_id` in this milestone; compound `scf.*` atoms still
  in the graph definition's body do not get their own id, and their
  parent-chain behavior is reconstructed by §2.5 / §2.6 of
  `docs/spec-compiler-part-3-mem.md` applied to the boundary
  translation rules in `docs/spec-compiler-part-3-dfg.md` §6. Each
  loop with hidden memory state gets `loom.mem_loop_id = L` and
  `loom.mem_loop_states = [...]`, a loop-local memory-state plan
  whose fields are deterministic integer ids, never operation
  references. Parallel-provenance groups are recorded with
  deterministic group and chunk ids, as temporary attributes such as
  `loom.parallel_group`, `loom.parallel_chunk`, and
  `loom.parallel_chunks`, or an equivalent analysis side table. They
  are implementation details consumed before final verification. The
  lowering transfers per-access attributes from source `memref` ops
  to replacement `dataflow` ops. `loom-finalize-dfg` drops all
  temporary memory-dependence and parallel-provenance attributes.

### 1.9 `loom-lower-scf-to-dfg-bodies`

* Inside every `dataflow.graph` definition's body, replaces each
  `scf.*` control-flow op with the canonical dataflow token rewrite
  (see Part 3's Per-scf Lowering Templates).
* Inside every `dataflow.thread` definition's body (outside any
  `dataflow.graph.launch`), `scf.*` ops are kept as-is; ScalarCore
  code remains structured.
* Non-inlined ScalarCore-legal `func.call` operations remain outside
  graph definition bodies and are preserved as ScalarCore calls.
* Memory ops (`memref.load`, `memref.store`) are rewritten in place
  as `dataflow.load` / `dataflow.store`. The pass implements the
  per-plane wiring rules in `docs/spec-compiler-part-3-mem.md` §6:
  every leaf op's `ctrl` operand is materialized as
  `dataflow.sync(S.struct_at_L, incoming_L_P)`, where the structural
  permission token comes from the per-`scf.*` boundary translation in
  `docs/spec-compiler-part-3-dfg.md` §6 and the memory-plane
  predecessor token comes from the partition-`P` chain (immediate dep
  predecessors at the same scope contribute their `done`; a sibling
  compound atom contributes its `outgoing_P` per the cross-scope
  resolution in mem.md §4.4; loop-carried predecessors contribute
  `%mem_iter_P` or `%mem_after_P` per mem.md §5). Multiple
  same-path predecessors join through `dataflow.sync`; multiple
  mutually exclusive predecessors join through selector-matched
  `dataflow.mux`; mixed sets compose hierarchically. The graph's
  `done_out` yield operand is the boundary `dataflow.sync` over the
  per-partition root `outgoing_P` tails computed at the root scope
  (mem.md §6.5); when the graph touches no partition, `done_out`
  forwards `ctrl_in` directly.

### 1.10 `loom-finalize-dfg`

* Runs the existing dataflow-graph verifier in strict mode.
* Strips the `loom.mem_dep_id`, `loom.mem_dep_preds`,
  `loom.mem_loop_id`, and `loom.mem_loop_states` attributes.
* Strips temporary parallel-provenance attributes such as
  `loom.parallel_group`, `loom.parallel_chunk`, and
  `loom.parallel_chunks`.
* Provides a `--keep-mem-dep` debug option that suppresses the
  attribute strip so the snapshot remains observable in the final
  IR. The option exists only for testing the
  `lower_scf/diff/` differential subset and for hand debugging;
  production pipelines must run finalize without the option so the
  exit IR matches the documented front-end output. The option does
  not change any other finalize behavior.
* Asserts that no temporary `loom.acc_region` op remains.
* Asserts the front-end exit invariant: no `scf.*` op remains inside
  any `dataflow.graph` definition's body; every
  `dataflow.thread.launch` produces exactly one
  `!dataflow.thread_token` and the launch site has no data results;
  every `dataflow.graph` definition's `function_type` includes
  the leading `none` ctrl_in / done_out slots; every
  `dataflow.graph.launch` has a well-formed explicit `ctrl_in`
  operand and `done_out` result; every graph launch's `ctrl_in` is
  sourced from the enclosing thread definition's `thread_ctrl` block
  arg, a preceding graph launch's `done_out`, or a
  `dataflow.thread.fence`.

## 2. Testing Strategy

The lit-test layout grows three new directories:

* `test/frontend/unit/` -- one subdirectory per new dialect element.
  Each subdirectory has `valid.mlir`, `invalid.mlir`, and a
  `roundtrip.mlir` confirming the printer / parser stability.
  Coverage targets:
  - `thread/`, `thread_launch/`, `thread_yield/`, `thread_fence/`,
    `thread_wait/`, `map_info/`, `graph/`, `graph_launch/`.
    `thread/` and `graph/` cover the def-side contracts (Symbol,
    FunctionOpInterface, function_type shape, body block-arg
    layout, IsolatedFromAbove); `thread_launch/` and `graph_launch/`
    cover launch-side contracts (callee resolution, type checking
    against the def's function_type, map_info-provenance,
    direction/body-effect compatibility, ctrl/done plumbing).
    Unit-test coverage for partitioned-data ops is owned by
    `docs/spec-compiler-part-4-partitioned-data.md`.
  - `thread/`, `thread_launch/`, `graph/`, and `graph_launch/`
    include invalid cases that directly reference surrounding SSA
    values from isolated regions.
  - `thread/` includes cases for ScalarCore-legal `func.call` in a
    thread definition's body and rejection of `func.func`
    definitions inside a thread definition.
  - `thread_launch/` includes cases that check the boundary
    memory-effect summary for `to`, `from`, and `tofrom` mapped
    operands, rejection of mapped operands not produced by
    `dataflow.map_info`, and the direction/body-effect
    compatibility check (a launch declaring `direction = to` whose
    callee writes through that arg is rejected with a diagnostic
    naming both ops).
  - `thread/` includes mapping-attribute fixtures for the
    `#loom.thread_axis<kind, axis, domain?>` form
    (per `docs/spec-compiler-part-3-dfg.md` §5.2 and §9):
    - valid: `#loom.thread_axis<parallel, 0>` and
      `#loom.thread_axis<multiplexed, 1>` entries with no domain
      qualifier.
    - valid: explicit-domain entries such as
      `#loom.thread_axis<parallel, 0, @D>` and
      `#loom.thread_axis<multiplexed, 1, @D>`, where `@D` is a
      visible `dataflow.partition_domain`.
    - invalid: a duplicate `(kind, domain, axis)` triple.
    - invalid: a domain-qualified entry whose symbol does not resolve
      to `dataflow.partition_domain`.
    - invalid: an axis value outside `[0, domain_rank)` for a
      domain-qualified entry.
    - invalid: a foreign (non-Loom) `DeviceMappingAttrInterface`
      attribute mixed with Loom-recognized entries (per Part 3 §3
      Mapping attribute rules, retained from earlier milestone).
  - `graph/` includes invalid cases for `func.call` and
    `func.func` inside a graph definition's body, and invalid
    cases for `dataflow.graph.launch` / `dataflow.thread.launch`
    appearing inside a graph definition's body.
  - `graph_launch/` includes invalid cases for graph launches at
    host scope (graph launch must be inside a thread definition's
    body), for callee mismatches (callee is not a `dataflow.graph`
    definition, function_type does not match operand/result types),
    and for the parent-side strict-layering rule (a graph launch in a
    non-innermost thread body that directly launches child threads).
* `test/frontend/lower_scf/` -- one subdirectory per scf op. Each
  directory holds:
  - `before.mlir` (the scf input).
  - `after.mlir` (the dataflow output, with explicit ctrl/done
    plumbing visible).
  - One default `RUN` line using `--mem-alias=basic` and one expected
    fixture file under that oracle. A small representative subset
    under `test/frontend/lower_scf/diff/` adds a second `RUN` line for
    `--mem-alias=mlir-aa` and an additional fixture that pins the
    snapshot-and-derived-wiring difference. Differential cases use
    `--keep-mem-dep` so the `loom.mem_dep_id` /
    `loom.mem_dep_preds` / `loom.mem_loop_*` attributes are
    observable on the final IR they FileCheck against; without the
    flag the snapshot would have been stripped by `loom-finalize-dfg`
    (per §1.10). The differential subset is intentionally small in
    milestone 1 with two minimum-coverage floors: one case per loop
    family (`for/`, `while/`, `forall/`-effect-form normalized to
    `parallel/`, plus straight-line `if/` and `index_switch/`) that
    exercises mlir-aa refinement on at least one access pair, and at
    least one case where the refinement changes the derived ctrl/done
    wiring shape (not only the `loom.mem_dep_preds` snapshot).
    Outside the differential subset, the basic-oracle fixture is
    the only ground truth.
  - The `forall/` directory has separate cases for effect-form forall
    and aggregation-form forall. Aggregation tests check that
    `scf.forall.in_parallel` combining actions are materialized into
    explicit destination-buffer effects before any
    `dataflow.thread` definition is built, and that residual
    non-empty `in_parallel` terminators produce diagnostics. Mapped
    forall tests also check that the original implicit
    synchronization point is represented by a token dependency on
    the produced launch's `!dataflow.thread_token` or by an
    explicit `dataflow.thread.wait`.
  - The `for/` and `while/` directories include loop-carried memory
    cases: an in-place stencil, zero-trip and one-trip execution,
    conditional memory effects whose tails must be joined by
    selector-matched `mux`, nested loops, and two independent memrefs
    that must produce independent memory-state partitions under the
    active oracle.
  - The `for/` directory includes a stream phasing case:
    `dataflow.stream` emits the trailing sentinel, `dataflow.gate`
    normalizes the index into body phase, memory side effects consume
    only body-phase values, no-iter-arg loops produce no data result,
    and iter-arg loops use the `carry` / `demux` / `mux` state-ring
    template to produce the zero-trip initial value or final body
    yield as the loop result.
  - The `while/` directory includes a before-to-after phasing case:
    the before-region executes one more time than the after-region,
    `dataflow.gate` produces the after-region value stream plus the
    after-local rwc, and a separate false-path projection preserves
    the `scf.condition` trailing operands as loop results.
  - `scalarcore_calls/` covers inlining or specialization of callees
    that contain graph-extractable code, preservation of graph-free
    ScalarCore calls, and diagnostics for unsupported callees.
  - `graph_placement/` covers the baseline L2 placement policy:
    adjacent memory-access clusters separated only by graph-
    admissible compute become one (graph def + launch) pair,
    required cuts split graph runs into multiple def + launch
    pairs, pure-only admissible runs may remain ScalarCore code,
    and graph-illegal pure ops such as `dataflow.map_info` or
    partitioned-data query ops stay outside graph definition bodies.
    Tests pin the deterministic symbol-naming convention
    (`g_<threadSym>_<seq>`) so cuts produce the same names across
    runs.
* `test/frontend/integration/` -- end-to-end small kernels covering
  the SPGPU / Chapel-style spatial idioms (matmul, stencil, LU,
  page-rank-style irregular loop) at the IR level only. No
  hardware execution; the assertion is structural well-formedness
  and round-trip stability.

In addition, the existing `test/dataflow/unit/graph/` and
`test/dataflow/unit/subgraph/` lit tests are migrated to the new
def + launch shape in the same change as the IR change: each test
(a) lifts the regional graph body to a module-scope
`dataflow.graph` definition with a deterministic symbol name,
(b) replaces the original regional op with a `dataflow.graph.launch`
referencing that symbol, and (c) preserves the explicit ctrl/done
plumbing on both the def's `function_type` (leading `none` slots)
and the launch (per-instance `ctrl_in` / `done_out`). Any test
that relies on the old regional graph form is updated to use the
new shape, and the migration is explicit in the diff.

Baseline L2 placement tests pin only the baseline policy output. A
future cost-aware graph-placement policy must introduce its own fixtures
or option-specific checks rather than rewriting the baseline
expectations.

## 3. Acceptance Criteria

The first milestone is considered complete when all of the
following hold simultaneously:

* Every `scf.*` operation enumerated under Part 3's
  Scope and Contract has a working
  lowering template, with at least one positive lit test under
  `test/frontend/lower_scf/<op>/` and at least one negative test
  exercising a verifier diagnostic.
* Aggregation-form `scf.forall` with
  `tensor.parallel_insert_slice` is accepted and materialized into
  explicit destination-buffer effects before thread promotion. No
  non-empty `scf.forall.in_parallel` reaches
  `loom-build-thread-skeleton`, and no combining action is silently
  discarded.
* Every promoted mapped `scf.forall` preserves the source
  operation's implicit synchronization point by following the
  mechanical rule in §1.5: either the immediately structurally
  following op is a `LoomAsyncOpInterface` op whose
  `asyncDependencies` operand list includes the produced token (the
  SSA use itself is the synchronization, no extra op needed), or a
  `dataflow.thread.wait` consuming the token is present before any
  continuation op (the conservative fallback). Acceptance verifies
  both conditions independently; the wait's effect-visibility
  barrier from `docs/spec-compiler-part-3-dfg.md` §3 Constitutional
  Rule 8 prevents subsequent ops from being reordered to before
  the synchronization.
* Root graph launch `ctrl_in` wiring is mechanical: graph launches
  with no preceding graph launch and no preceding ScalarCore fence
  consume the enclosing thread definition's `thread_ctrl` block
  argument, ScalarCore-to-graph-launch ordering uses
  `dataflow.thread.fence`, and child-thread launch completion can
  feed graph-launch control through that same fence op.
* `func.call` inside a `dataflow.thread` definition's body is
  handled as ScalarCore control: graph-containing callees are
  inlined or specialized before graph extraction, graph-free
  ScalarCore calls may remain, and no `func.call` or `func.func`
  appears inside a `dataflow.graph` definition's body.
* `BasicSsaOracle` drives `loom-build-memory-dependencies` and passes
  the full lit suite under `test/frontend/`. `MlirAaOracle` drives the
  differential subset under `test/frontend/lower_scf/diff/` and
  produces structurally identical IR modulo the `loom.mem_dep_preds`
  snapshot and derived ctrl/done wiring.
* Loop-carried memory dependences lower to explicit hidden `none`
  memory-state carries. Post-loop conflicting accesses depend on the
  loop-exit memory state, zero-trip loops forward the pre-loop state,
  branch-local loop tails use selector-matched joins, and independent
  memory partitions are not serialized when the active oracle proves
  them independent.
* `scf.while` lowering preserves the one-extra-before execution
  semantics: the before-to-after boundary is normalized by
  `dataflow.gate`, after-local state is driven by the gated rwc, and
  loop results still come from the false-cycle `scf.condition`
  operands rather than from the gate.
* `scf.for` lowering preserves stream phasing: the sentinel index is
  not consumed by body memory effects, body-only state uses the gated
  rwc, iter_arg state uses the loop-level rwc plus
  `carry` / `demux` / `mux`, zero-trip loops forward initial iter_arg
  values, and nonzero loop results come from the final body yield.
* All previously existing tests in `test/dataflow/unit/graph/` and
  `test/dataflow/unit/subgraph/` continue to pass after the
  def + launch migration.
* `loom-finalize-dfg` rejects, with a clear diagnostic, every
  input produced by the lowering pipeline that contains a residual
  `scf.*` op inside a `dataflow.graph` definition's body.
* The verifiers for the `dataflow.thread` and `dataflow.graph`
  definitions reject any direct use of a surrounding SSA value from
  inside their isolated bodies; all such values must flow through
  explicit launch operands and matching entry block arguments. The
  verifiers for `dataflow.thread.launch` and
  `dataflow.graph.launch` reject unresolved callee symbols, callee-
  kind mismatches, type mismatches against the resolved def's
  `function_type`, and (for thread launch) map_info-provenance and
  direction-vs-body-effect-compatibility violations.
* `dataflow.thread.launch` reports external memory effects through
  its `MemoryEffectsOpInterface` implementation, projecting mapped
  boundary operands to their `dataflow.map_info` sources according
  to `direction`. The launch additionally declares a conservative
  effect on a custom `LoomAsyncResource` resource so generic
  CSE / DCE never removes a launch even when its callee body has
  no host-visible memory effects (per
  `docs/spec-compiler-part-3-dfg.md` §3 Constitutional Rule 8).
  `dataflow.graph.launch` reports external memory effects by
  resolving its `callee` and walking the callee body. No acceptance
  test depends on `RecursiveMemoryEffects` to discover host-
  visible thread or graph reads or writes through the launch
  boundary; the def-side `RecursiveMemoryEffects` trait is for
  module-scope walkers only.
* Integration tests in `test/frontend/integration/` run under
  `--mem-alias=basic` only in milestone 1; differential coverage in
  this directory is reserved for follow-up work if oracle-pair
  behavior becomes a regression risk for end-to-end kernels.
* `make test` runtime stays within the existing budget; the test
  suite is parallel-safe (the existing `lit_top_slowest.py` machinery
  is kept).

## 4. Maintenance and Extension Points

* Adding a new `scf` op: extend Part 3's Per-scf Lowering Templates
  with a template, add a
  rewrite implementation under `lib/Frontend/Lowering/`, add a
  `test/frontend/lower_scf/<op>/` directory with `before.mlir` /
  `after.mlir` / verifier coverage.
* Adding a new `DeviceMappingAttrInterface` instance (e.g.
  `#loom.warp<...>`): add the attribute class in
  `include/Frontend/IR/LoomMappingAttrs.td`, register it with the
  dialect, and write a `test/frontend/unit/thread/` case that uses
  it.
* Tightening `dataflow.map_info` direction: write an analysis pass
  under `lib/Frontend/Analysis/`. The pass must run after
  `loom-promote-map-info` and before `loom-build-thread-skeleton`.
  The default direction stays `tofrom` until the analysis fires.
* Supporting another `scf.forall.in_parallel` combining op: extend
  `loom-materialize-forall-aggregation` with a materialization rule
  for that `ParallelCombiningOpInterface` implementation. The
  `dataflow.thread` definition and `dataflow.thread.launch` shapes
  do not change.
* Adding a stronger alias oracle: implement
  `MemAliasOracle` in a new translation unit under
  `lib/Frontend/Analysis/`, and add a `--mem-alias=<name>` value to
  `loom-build-memory-dependencies`. The lit suite is expected to pass
  unchanged except for deliberate fixture differences in
  `loom.mem_dep_preds` and derived ctrl/done wiring.
* Adding a stronger L2 graph-placement policy: implement it under
  `lib/Frontend/Placement/` or an equivalent placement module, expose it
  through an explicit pass option, and add option-specific tests. The
  baseline source-order greedy policy remains the default reference
  policy.
* `Dataflow_GraphOp::build(...)` C++ surface change. The op is now
  a function-like definition (per
  `docs/spec-compiler-part-3-dfg.md` §5.5.1). Every existing
  regional-form `build(...)` overload is replaced by a function-
  like builder accepting `(StringRef sym_name, FunctionType
  functionType, ArrayRef<NamedAttribute> attrs)` and optional
  `arg_attrs` / `res_attrs` arrays. Callers no longer construct an
  inline body region; the body is added through the standard
  `FunctionOpInterface` body-construction path, with the entry
  block carrying the leading `none` ctrl_in block argument and the
  user-data block arguments matching `function_type.inputs`. This
  is intentionally source-incompatible: the op is now a callable,
  not a region executor. Construction of per-launch sites uses the
  separate `Dataflow_GraphLaunchOp` builders.
* `Dataflow_ThreadOp::build(...)` C++ surface change. The op is
  now a function-like definition (per
  `docs/spec-compiler-part-3-dfg.md` §5.4.1). The previous regional
  builder is replaced by a function-like builder analogous to the
  graph builder above, with additional grid attributes
  (`staticGrid*`, `mapping`) and entry-block layout
  `(args_*, thread_ctrl, iv_*)`. Per-launch sites use the separate
  `Dataflow_ThreadLaunchOp` builders. This is intentionally source-
  incompatible.

## 5. References

* `docs/spec-compiler-part-3-dfg.md` -- Part 3 main spec (boundary
  contracts, SCF flattening templates, verifier invariants).
* `docs/spec-compiler-part-3-mem.md` -- compositional chain model,
  alias oracle, dependence builder, loop-carried memory state, and
  token wiring. The contract that `loom-build-memory-dependencies`
  implements lives in that document.
* `docs/spec-compiler-part-3-placement-framework.md` -- common
  placement-partition framework and the L2 graph-placement model used
  by `loom-extract-graph-regions`.
* `docs/spec-compiler-part-4-partitioned-data.md` -- partitioned-data spec; the
  test plan above defers to Part 4 for partitioned-data unit-test
  coverage.
* Upstream MLIR references used by the passes above are listed in
  Part 3's References section.
