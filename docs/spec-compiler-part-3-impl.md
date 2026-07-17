# Loom Compiler Part 3 Pipeline Contract

This document collects the engineering details that support the Part 3
(SCF-to-DFG) front-end design: the pass pipeline, the lit-test layout,
the acceptance checklist, and the extension points.

Part 3 itself (`docs/spec-compiler-part-3-dfg.md`) holds the
first-principles IR content -- boundary contracts, SCF flattening
templates, and verifier invariants. The memory-dependence model
that this part lowers to is owned by
`docs/spec-compiler-part-3-mem.md`. Material in this file exists so
that one canonical implementation is pinned; readers who only need
the design contract can skip this file.
References below use Part 3 section names rather than numeric indices,
so that Part 3 can renumber without forcing edits here.

## Implementation Guidance Boundary

This file is implementation guidance, not the owning target contract.
The target IR semantics are owned by `docs/spec-compiler-part-3-dfg.md`,
the memory-dependence model is owned by
`docs/spec-compiler-part-3-mem.md`, and global evidence policy is owned
by `docs/spec-loom-stack.md`. If this file conflicts with those owning
specs, the owning specs win and this file must be updated.

Future execution plans may replace the pass ordering, test layout, or
implementation decomposition when they preserve the target IR and
verification contracts.

## 1. Lowering Pipeline

The executable `loom-lower-scf-to-dfg` pipeline is defined once in
`lib/Frontend/Lowering/Pipeline.cpp`:

```text
loom-lower-forall-to-thread
loom-lower-for-to-graph
canonicalize
loom-lower-known-library-calls
loom-lower-graph-memory
loom-lower-graph-constants
canonicalize
```

Individual passes remain runnable for focused diagnostics, but this ordering is
the production contract.

### 1.1 `loom-lower-forall-to-thread`

* Runs before graph extraction so mapped parallel work is represented inside
  module-scope `dataflow.thread` definitions.
* Materializes explicit thread launch dependencies and preserves ScalarCore
  code outside promoted regions.
* Leaves graph-local structured control for the recursive graph owner.

### 1.2 `loom-lower-for-to-graph`

* Selects graph boundary classification before moving or cloning body
  operations. The graph `FunctionType` contains only application payloads;
  normalized `input_segments` and `result_segments` record value, stream,
  and memory kinds.
* Creates the distinguished leading `start : none` entry argument and a
  structural `graph.return` seed separately from the payload function type.
* Reorders application payloads into normalized segment order and rewires the
  launch with the same order.
* Rejects loop-carried memory capabilities that cannot be projected into an
  explicit index domain. It does not route pointer or memref values through
  carry, mux, demux, gate, or invariant as ordinary data.

### 1.3 Canonicalization

* Removes dead bridge values and normalizes local SSA shape before recursive
  lowering.
* Does not infer graph port kinds or completion.

### 1.4 `loom-lower-known-library-calls`

* Expands the bounded library-call surface that has an explicit graph
  implementation.
* Leaves unsupported or effect-only calls for fail-closed graph-memory
  diagnostics.

### 1.5 `loom-lower-graph-memory`

* This pass is the single recursive owner of graph structured-control,
  memory-order, value-publication, and retirement lowering.
* It validates all graph-region preconditions before live mutation and checks
  normalized residual memory effects on a scratch module. Speculative
  index-domain materialization has local rollback and leaves the original
  recurrence intact when conversion fails.
* Boundary memory capabilities are selected by normalized segment metadata.
  Canonical roots are derived through views, conversion bridges, globals, and
  static pointer bases. Boundary capabilities without explicit no-alias
  evidence share one conservative alias root; unknown accesses cover every
  live partition.
* Each live alias partition has exactly one
  `(write_frontier, read_frontier)` pair. Straight-line accesses and recursive
  selection, repeat, and parallel transfer follow
  `docs/spec-compiler-part-3-mem.md`. Structural execution remains a separate
  state component.
* Raw parallel SCF, unsupported residual containers, observable operations
  without explicit completion events, and memory capabilities transported on
  dataflow control primitives fail closed.
* Supported memory leaves become `dataflow.load` and `dataflow.store`.
  Retirement combines structural execution with every live partition's final
  read frontier after causal transitive reduction. Final values are published
  through that same frontier. A no-work graph may retire from start; a graph
  with real work, including zero-output work, may not.
* No persisted dependence-id, hidden effect scan, or loop-plan record is a
  second correctness authority.

### 1.6 `loom-lower-graph-constants`

* Promotes remaining top-level literals to `dataflow.constant` using the graph
  start protocol.
* Nested literals are already projected by recursive lowering.
* The pass does not add completion witnesses.

### 1.7 Closing Canonicalization

* Removes dead bridge and projection values after graph-memory lowering.
* Preserves the explicit return frontier and normalized segment metadata.

### 1.8 Native Finalization Gate

* Direct DFG simulation and PnR mapping call the same native finalized-graph
  validator before execution or mapping. This gate is mandatory and is not
  replaceable by Python preprocessing.
* The gate rejects residual SCF/CFG/region containers, memory capabilities on
  dataflow transport primitives, nontrivial graphs that use raw start as a
  completion witness, and retirement frontiers that fail to cover payloads,
  memory operations, observable effects, or stateful close/reset.
* `done_out = all_of(graph.return.complete)`. Validation may prove that the
  declared frontier covers required behavior, but it does not synthesize an
  alternate completion event from effect scans or graph quiescence.
* The simulator treats frontier firing as retirement and rejects subsequent
  operation firing. Unsupported boundary behavior, including memory export
  identity simulation, reports `unsupported` without flattening the segment
  into scalar output.

## 2. Testing Strategy

Tests are organized by stable semantic boundary:

* `test/dataflow/unit/graph_func/` verifies the payload-only function type,
  normalized segment sizes, exact per-segment kinds and types, mandatory
  non-empty completion, and launch symbol/type matching.
* `test/raise/` verifies graph extraction, recursive SCF lowering, canonical
  per-partition memory frontiers, zero-trip and descending loops, nested
  selection/repeat structure, index-domain carry narrowing, transactional
  rollback, and fail-closed diagnostics for parallel residue, pointer
  capability transport, and effects without completion events.
* `test/dfg/` verifies the strict native finalized-graph gate independently
  of simulator and PnR frontends.
* `test/simulator/` verifies retirement-time execution, value and stream
  segments, explicit unsupported memory export, phase/reset/re-entry behavior,
  vector pack/serialize, scalar broadcast, pointer and integer primitives,
  dynamic extents, known library operations, and artifact simulation.
* `test/pnr/` verifies the same native gate at mapping entry and preserves
  placement, routing, operation, status, and diagnostic assertions for
  canonical graphs.
* `test/adg/` verifies deterministic builder output, exact canonical fixtures
  where retained, and complete placement/routing of workloads that exercise
  retirement demux and typed publication syncs.

Residual-SCF execution inside a finalized graph is not a supported behavior.
Fixtures whose only contract was leaf reconstruction through residual
containers are represented by compact strict-finalization rejection tests.
Operation semantics unrelated to that rejected behavior remain covered at
their native raise, simulator, PnR, or ADG boundary.

## 3. Acceptance Criteria

The Part 3 slice is coherent only when all of the following hold:

* Graph `FunctionType` contains only application payloads, and normalized
  segment metadata is the single authority for value, stream, and memory
  classification.
* Graph and return verifiers reject count, kind, recursive capability, or exact
  type mismatches.
* Recursive lowering preserves one canonical
  `(write_frontier, read_frontier)` pair per live alias partition and applies
  the documented selection, repeat, and parallel transfer rules.
* Retirement is the non-empty explicit `graph.return.complete` all-of.
  Nontrivial graphs cannot use raw start, and no effect scan or quiescence rule
  creates a second completion mechanism.
* Final values, stream close/boundary commit, memory capability establishment
  and visibility, observable effects, stateful close/reset, and non-detached
  async work are causally covered by the declared frontier.
* Pointer or memref capabilities that cannot be projected into the explicit
  memory plane fail closed rather than entering dataflow carry or selection.
* Direct simulator and PnR entry reject residual structured containers and
  validate the same ABI and retirement contract.
* Unsupported simulator boundary semantics report `unsupported` without
  flattening segment kinds.
* Focused raise, DFG, simulator, PnR, and ADG suites pass, followed by the full
  locked `make test` invocation and `git diff --check`.

## 4. Maintenance and Extension Points

* Adding a structured graph operation requires one recursive transfer rule in
  `GraphRegionLowering.cpp`, a focused raise anchor, and strict native-gate
  coverage if the operation may survive finalization.
* Increasing alias precision must strengthen canonical root or partition
  selection without creating persisted dependence metadata as a second
  authority. Conservative boundary aliasing remains the fallback.
* Supporting another memory capability representation requires updating the
  graph port verifier, explicit frontend classification, canonical-root
  traversal, simulator boundary handling, and PnR validation together.
* Adding a completion-bearing operation requires a causal witness that can
  enter `graph.return.complete`; effect-only acceptance without such a witness
  is not permitted.
* ADG support for new retirement shapes should add only the operation modes and
  routes required by canonical graphs, with deterministic builder output and
  unchanged placement/routing assertions.
* `Dataflow_GraphFuncOp::build(...)` accepts a payload-only
  `FunctionType` plus normalized segment attributes. The body adds the
  separate leading `start : none` argument. Per-launch start and done use the
  separate `Dataflow_GraphLaunchOp` protocol operands/results.

## 5. References

* `docs/spec-compiler-part-3-dfg.md` -- Part 3 main spec (boundary
  contracts, SCF flattening templates, verifier invariants).
* `docs/spec-compiler-part-3-mem.md` -- canonical alias partitions,
  recursive `(write_frontier, read_frontier)` state, and token wiring owned by
  graph-region lowering.
* `docs/spec-compiler-part-3-placement-framework.md` -- common
  placement-partition framework and the L2 graph-placement model used during
  graph extraction.
* `docs/spec-compiler-part-4-partitioned-data.md` -- partitioned-data spec; the
  test plan above defers to Part 4 for partitioned-data unit-test
  coverage.
* Upstream MLIR references used by the passes above are listed in
  Part 3's References section.
