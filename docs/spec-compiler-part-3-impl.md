# Loom Compiler Part 3 Implementation Notes

This document collects the engineering details that support the Part 3
(SCF-to-DFG) front-end design: the pass pipeline, the lit-test layout,
the milestone acceptance checklist, and the extension points.

Part 3 itself (`docs/spec-compiler-part-3-dfg.md`) holds the
first-principles content -- IR boundary contracts, SCF flattening
templates, the memory-dependence model, and verifier invariants.
Material in this file exists so that one canonical implementation is
pinned; readers who only need the design contract can skip this file.
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
* Identifies every `scf.forall` whose `mapping` attribute is non-empty
  and contains at least one `DeviceMappingAttrInterface` element
  recognizable as a `#loom.spatial<...>` or `#loom.temporal<...>`
  instance.
* Marks each such forall with a temporary attribute
  `loom.thread_promotion = unit`. Nested mapped foralls are marked
  individually; the relative nesting order is preserved by IR
  traversal order.
* If an accelerator region contains zero mapped foralls, the pass adds
  a synthetic outermost `scf.forall (%i) in (1) { ... }` with the
  region-local default mapping from `loom-normalize-acc-regions`,
  wrapping the accelerator region body. This guarantees every explicit
  accelerator region lowers to at least one `dataflow.thread` without
  turning unselected host code into AccCore code.

### 1.4 `loom-promote-map-info`

* For each marked forall, computes the set of values defined outside
  the forall body and used inside it.
* For outermost foralls created from a scalar-only `loom.acc_region`,
  the surrounding values are the explicit accelerator-region boundary
  operands, not arbitrary values captured from the enclosing
  `func.func`.
* For every `memref<...>` value or spatial-array handle that crosses
  the thread boundary, inserts a `dataflow.map_info ...
  direction=tofrom` immediately outside the forall. Scalar values do
  not need `map_info`; they become by-value launch operands.
* The pass records a deterministic boundary-operand list on the marked
  forall: mapped data handles first in SSA discovery order, then
  scalar launch operands in SSA discovery order. The next pass uses
  this list to build an isolated `dataflow.thread` and rewrite body
  uses to the corresponding entry block arguments.
* Future optimizer passes can refine `tofrom` to `to` or `from` based
  on read/write effect summaries; this pass is intentionally
  conservative.

### 1.5 `loom-build-thread-skeleton`

* Replaces every marked `scf.forall` with a `dataflow.thread` whose
  grid bounds, mapping, body operands, body region, and
  terminator come from the forall.
* Requires effect-form forall input. If a marked forall still has
  `shared_outs`, op results, or a non-empty `scf.forall.in_parallel`,
  the pass emits a diagnostic pointing to
  `loom-materialize-forall-aggregation`.
* The thread body gets a leading `thread_ctrl : none` block argument,
  followed by the forall induction variables (one per grid dim, all
  `index`).
* Body operands keep the deterministic ordering computed by
  `loom-promote-map-info` and become entry block arguments after the
  leading control and induction-variable arguments. Each block
  argument has the same type as the matching body operand: memref-like
  operands produced by `dataflow.map_info` are passed through as the
  same memref type, and scalar operands keep their original type.
  The pass rewrites every in-body use of a surrounding SSA value to
  the corresponding block argument before the verifier sees the
  isolated thread.
* The empty `scf.forall.in_parallel` terminator is replaced with an
  empty `dataflow.thread.yield`. No tensor aggregation action is
  dropped by this pass; all such actions must already have been
  materialized as explicit memory effects.
* `scf.forall` is an implicit synchronization point. The replacement
  therefore must make continuation ordering explicit: either the
  produced `!dataflow.thread_token` becomes an async dependency of a
  following thread-like op, or a `dataflow.thread.wait` is inserted at
  the original continuation point. The first milestone uses the
  conservative form and inserts the wait unless an immediately
  following thread dependency is materialized in the same rewrite.
* Once every marked forall inside a `loom.acc_region` has been replaced
  by `dataflow.thread`, the temporary accelerator-region wrapper is
  erased and its body is spliced back at the original host program
  point. No `loom.acc_region` remains after this pass.

### 1.6 `loom-prepare-scalarcore-calls`

* Runs after thread skeleton construction and before graph extraction.
* Inspects every `func.call` reachable inside a `dataflow.thread` body.
  The call is a ScalarCore operation, not a SpatialCore operation.
* If the callee contains operations that must be graph-extracted in the
  caller's thread context, the pass inlines or specializes the callee
  into that `dataflow.thread` before graph extraction. This keeps
  `dataflow.graph` lexically inside the active thread without requiring
  `dataflow.thread` to become a symbol table or carry an implicit
  function context.
* A non-inlined `func.call` may remain only if the callee is
  ScalarCore-legal and graph-free after this preparation. Such calls
  are treated as ScalarCore side-effecting operations by later passes.
* Unsupported calls reachable from a thread body produce a diagnostic.
  The first implementation may require all non-trivial ScalarCore calls
  to be inlined.

### 1.7 `loom-extract-graph-regions`

* Within each innermost `dataflow.thread` body (no further mapped
  forall inside), groups eligible operations into one or more
  `dataflow.graph` regions.
* Eligibility for a graph region: a maximal connected sub-DAG of ops
  rooted at one or more `memref.{load, store}` operations, extended
  upward only through ops admitted by the graph body whitelist in
  Part 3's verifier rules -- `arith.*`, `math.*`, and
  `dataflow.{stream, carry, invariant, gate, mux, demux, sync,
  constant}` -- and stopping at the first
  `scf.{if,while,for,parallel,index_switch,execute_region}` boundary
  that contains side-effecting code, or at any `func.call`,
  `dataflow.thread` boundary, or at any `func.return` /
  `dataflow.thread.yield` terminator. The `Pure` trait alone is not
  sufficient for admission: `dataflow.map_info` and the spatial-array
  ops in `docs/spec-compiler-part-4-spatial.md` are also `Pure` but
  are boundary-only or thread-body-only, and the whitelist
  intentionally excludes them.
* Within a single graph, the `scf.*` control-flow ops appear as
  unflattened children that the next-but-one pass will lower into
  `dataflow` token primitives. The extraction pass does not modify
  control-flow shape; it only moves ops into a region container and
  supplies the graph's explicit `ctrl_in` operand/block-arg plus its
  explicit `done_out` result/yield slot. Graph-to-graph ordering is
  represented by ordinary SSA use of one graph's `done_out` result as
  another graph's `ctrl_in` operand.
* A graph with no graph predecessor and no explicit ScalarCore fence
  predecessor uses the enclosing thread body's `thread_ctrl` block
  argument as its `ctrl_in`. If ScalarCore work must complete before a
  graph starts, the lowering inserts or preserves a
  `dataflow.thread.fence` at that program point and uses the fence
  result as the graph's `ctrl_in`.
* `dataflow.graph` ops are ScalarCore launch points for SpatialCore
  work. The `ctrl_in` operand is an additional graph-level start
  dependency; it is not the only sequencing rule. The graph launch also
  occurs at the graph op's position in the enclosing ScalarCore
  program. SpatialCore completion becomes visible to later ScalarCore
  code only when its `done_out` is consumed by
  `dataflow.thread.fence`.
* Because `dataflow.graph` is `IsolatedFromAbove`, the extraction pass
  also computes every surrounding value used by the graph body and
  materializes it as an explicit graph operand and entry block
  argument. Values produced inside the graph and used outside it are
  materialized as explicit graph results and `dataflow.yield` operands.

### 1.8 `loom-build-memory-dependencies`

* Builds the per-graph memory-dependence snapshot consumed by body
  lowering. A memory access means `memref.load` / `memref.store`
  before rewrite and `dataflow.load` / `dataflow.store` after rewrite.
  Although the pass name focuses on dependence construction, this pass
  also performs the final parallel-SCF normalization described below.
  Keeping both tasks together ensures memory accesses cloned by
  normalization immediately enter the same deterministic id assignment.
* Before assigning memory-access ids, performs the remaining
  parallel-SCF normalization inside each `dataflow.graph` body:
  effect-form `scf.forall` with an empty mapping is normalized to
  `scf.parallel`, and `scf.parallel` is normalized to one or more
  `scf.for` loop nests plus any required reduction-merge `scf.if`
  ops. This is the point where parallel provenance is planted on the
  generated loop nests. Every cloned memory access receives its own
  deterministic `loom.mem_dep_id`.
* Chunk-bound arithmetic introduced by parallel-SCF normalization,
  such as trip-count computation and per-chunk lower / upper bounds, is
  materialized inside the same `dataflow.graph` that contained the
  original `scf.parallel`. These new ops use only graph-local SSA
  values and must satisfy the existing graph-body whitelist for pure
  computation ops.
* A user-written `scf.parallel` with a non-empty `mapping` attribute is
  rejected here. Mapping is honored only on `scf.forall` in this
  milestone.
* Materializes a `MemAliasOracle` instance for each `dataflow.graph`
  region. The oracle is a per-graph conflict oracle; the rest of the
  lowering reads only through it, never directly off MLIR's analysis
  manager.
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
  the pass option `--mem-alias=basic|mlir-aa`:
  - `BasicSsaOracle`: an SSA-source roll-up over `memref.alloca`,
    function block-args, `memref.global`, `memref.cast`,
    `memref.subview`, `memref.view`, `memref.expand_shape`,
    `memref.collapse_shape`. Two accesses conflict iff their root
    memrefs are the same SSA root, regardless of offset/shape, and
    they are not both loads.
  - `MlirAaOracle`: forwards to `mlir::AliasAnalysis` from
    `mlir/Analysis/AliasAnalysis.h` as a refinement of
    `BasicSsaOracle`. It starts from the basic conflict set and removes
    pairs that upstream MLIR AA proves `MustNotAlias`. When upstream AA
    cannot prove anything stronger, it behaves exactly like the basic
    oracle.
* Runs a `MemoryDependenceBuilder` after alias queries are available.
  The builder visits memory accesses in deterministic program order.
  Alias answers are symmetric and never define direction by
  themselves; direction always comes from program order plus the
  enclosing structured-control-flow path.
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
* For each ordered pair `p` before `o`, the builder records a
  directed dependence `p -> o` iff the pair conflicts. The builder may
  drop transitively implied edges, provided the remaining immediate
  predecessor and successor sets induce the same memory partial order.
* For structured loops, the builder also records loop-carried memory
  state plans. Each plan is keyed by a deterministic loop id and a
  memory partition id, and references memory accesses only by integer
  ids. A partition requiring cross-iteration ordering lowers to one
  hidden `none` carry in `loom-lower-scf-to-dfg-bodies`.
* The pass leaves a stable IR snapshot so subsequent passes need no
  re-analysis: each memory access gets `loom.mem_dep_id = N` and
  `loom.mem_dep_preds = [P0, P1, ...]`, where `N` and every `P*` are
  deterministic integer ids inside the graph. Each loop with hidden
  memory state gets `loom.mem_loop_id = L` and
  `loom.mem_loop_states = [...]`, a loop-local memory-state plan whose
  fields are deterministic integer ids, never operation references.
  Parallel-provenance groups are recorded with deterministic group and
  chunk ids. These may be temporary attributes such as
  `loom.parallel_group`, `loom.parallel_chunk`, and
  `loom.parallel_chunks`, or an equivalent analysis side table. They
  are implementation details consumed before final verification. The
  lowering transfers per-access attributes from source `memref` ops to
  replacement `dataflow` ops. `loom-finalize-dfg` drops all temporary
  memory-dependence and parallel-provenance attributes.

### 1.9 `loom-lower-scf-to-dfg-bodies`

* Inside every `dataflow.graph`, replaces each `scf.*` control-flow
  op with the canonical dataflow token rewrite (see Part 3's
  Per-scf Lowering Templates).
* Inside every `dataflow.thread` body (outside any graph), `scf.*`
  ops are kept as-is; ScalarCore code remains structured.
* Non-inlined ScalarCore-legal `func.call` operations remain outside
  graphs and are preserved as ScalarCore calls.
* Memory ops (`memref.load`, `memref.store`) are rewritten in place
  as `dataflow.load` / `dataflow.store`. The lowering builds a ctrl
  source set for each memory op from immediate predecessor `done`
  tokens and any required hidden loop-carried memory-state token. If
  the set is empty, it uses the graph entry `ctrl_in` block argument;
  if the set has one value, it uses that value directly; otherwise it
  uses output zero of a `dataflow.sync` over the set. This is legal
  only because all values in the set are required predecessors on the
  same dynamic path; mutually exclusive tails are joined by
  selector-matched `dataflow.mux`, not by `dataflow.sync`. The graph
  `done_out` yield operand is output zero of a `dataflow.sync` over
  all memory accesses with no dependence successor (see Part 3's
  Memory Dependence Model).

### 1.10 `loom-finalize-dfg`

* Runs the existing dataflow-graph verifier in strict mode.
* Strips the `loom.mem_dep_id` and `loom.mem_dep_preds` attributes.
* Strips temporary parallel-provenance attributes such as
  `loom.parallel_group`, `loom.parallel_chunk`, and
  `loom.parallel_chunks`.
* Asserts that no temporary `loom.acc_region` op remains.
* Asserts the front-end exit invariant: no `scf.*` op remains inside
  any `dataflow.graph` body; every `dataflow.thread` produces exactly
  one `!dataflow.thread_token` and no data results; every
  `dataflow.graph` has a well-formed explicit `ctrl_in` / `done_out`
  control-port pair; every graph `ctrl_in` is sourced from the
  enclosing thread `thread_ctrl`, a preceding graph `done_out`, or a
  `dataflow.thread.fence`.

## 2. Testing Strategy

The lit-test layout grows three new directories:

* `test/frontend/unit/` -- one subdirectory per new dialect element.
  Each subdirectory has `valid.mlir`, `invalid.mlir`, and a
  `roundtrip.mlir` confirming the printer / parser stability.
  Coverage targets:
  - `thread/`, `thread_yield/`, `thread_fence/`, `thread_wait/`,
    `map_info/`,
    `graph_control_ports/` (modifications to existing graph op).
    Unit-test coverage for spatial-array ops is owned by
    `docs/spec-compiler-part-4-spatial.md`.
  - `thread/` and `graph_control_ports/` include invalid cases that
    directly reference surrounding SSA values from isolated regions.
  - `thread/` includes cases for ScalarCore-legal `func.call` in a
    thread body and rejection of `func.func` definitions inside a
    thread.
  - `thread/` includes cases that check the boundary memory-effect
    summary for `to`, `from`, and `tofrom` mapped operands, and
    rejection of mapped operands not produced by `dataflow.map_info`.
  - `graph_control_ports/` includes invalid cases for `func.call` and
    `func.func` inside a graph body.
* `test/frontend/lower_scf/` -- one subdirectory per scf op. Each
  directory holds:
  - `before.mlir` (the scf input).
  - `after.mlir` (the dataflow output, with explicit ctrl/done
    plumbing visible).
  - `RUN` lines for both `--mem-alias=basic` and `--mem-alias=mlir-aa`,
    each FileChecking against a distinct expected fixture file. The
    two fixture files differ only by memory-dependence snapshots and
    derived ctrl/done wiring; the structural rewrite is identical.
  - The `forall/` directory has separate cases for effect-form forall
    and aggregation-form forall. Aggregation tests check that
    `scf.forall.in_parallel` combining actions are materialized into
    explicit destination-buffer effects before any `dataflow.thread`
    is built, and that residual non-empty `in_parallel` terminators
    produce diagnostics. Mapped forall tests also check that the
    original implicit synchronization point is represented by a token
    dependency or `dataflow.thread.wait`.
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
* `test/frontend/integration/` -- end-to-end small kernels covering
  the SPGPU / Chapel-style spatial idioms (matmul, stencil, LU,
  page-rank-style irregular loop) at the IR level only. No
  hardware execution; the assertion is structural well-formedness
  and round-trip stability.

In addition, the existing `test/dataflow/unit/graph/` and
`test/dataflow/unit/subgraph/` lit tests are migrated to the explicit
graph-control-port shape in the same change as the IR change. Any
test that relies on the old graph form is updated to use the new
form, and the migration is explicit in the diff.

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
  operation's implicit synchronization point by explicit
  `!dataflow.thread_token` use: either a following thread-like op
  consumes the token as a dependency, or a `dataflow.thread.wait`
  appears before continuation code that can observe the effects.
* Root graph `ctrl_in` wiring is mechanical: graphs with no graph or
  ScalarCore fence predecessor consume the enclosing `thread_ctrl`,
  ScalarCore-to-graph ordering uses `dataflow.thread.fence`, and
  child-thread completion can feed graph control through that same
  fence op.
* `func.call` inside a `dataflow.thread` is handled as ScalarCore
  control: graph-containing callees are inlined or specialized before
  graph extraction, graph-free ScalarCore calls may remain, and no
  `func.call` or `func.func` appears inside a `dataflow.graph`.
* The two `MemAliasOracle` implementations both drive
  `loom-build-memory-dependencies` and pass the entire lit suite under
  `test/frontend/`.
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
  graph-control-port migration.
* `loom-finalize-dfg` rejects, with a clear diagnostic, every input
  produced by the lowering pipeline that contains a residual
  `scf.*` op inside a `dataflow.graph` body.
* The verifiers for `dataflow.thread` and `dataflow.graph` reject any
  direct use of a surrounding SSA value from inside their isolated
  regions; all such values must flow through explicit operands and
  entry block arguments.
* `dataflow.thread` reports external memory effects through its
  `MemoryEffectOpInterface` implementation, projecting mapped boundary
  operands to their `dataflow.map_info` sources according to
  `direction`. No acceptance test depends on recursive region effects
  to discover host-visible thread reads or writes.
* The integration tests in `test/frontend/integration/` produce
  structurally identical IR under `--mem-alias=basic` and
  `--mem-alias=mlir-aa`, modulo the `loom.mem_dep_preds` snapshot and
  the ctrl/done wiring derived from it.
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
  `dataflow.thread` op shape does not change.
* Adding a stronger alias oracle: implement
  `MemAliasOracle` in a new translation unit under
  `lib/Frontend/Analysis/`, and add a `--mem-alias=<name>` value to
  `loom-build-memory-dependencies`. The lit suite is expected to pass
  unchanged except for deliberate fixture differences in
  `loom.mem_dep_preds` and derived ctrl/done wiring.

## 5. References

* `docs/spec-compiler-part-3-dfg.md` -- Part 3 main spec (boundary
  contracts, SCF flattening templates, memory-dependence model,
  verifier invariants).
* `docs/spec-compiler-part-4-spatial.md` -- spatial-array spec; the
  test plan above defers to Part 4 for spatial-op unit-test coverage.
* Upstream MLIR references used by the passes above are listed in
  Part 3's References section.
