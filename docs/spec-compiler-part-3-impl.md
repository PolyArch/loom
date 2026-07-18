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
* Copies source function argument dictionaries onto captured thread payload
  arguments in capture order. The separately introduced ctrl and IV arguments
  have no payload metadata.
* Leaves graph-local structured control for the recursive graph owner.

### 1.2 `loom-lower-for-to-graph`

* Stages each selected structured candidate in `loom.spatial_region` before
  creating any canonical graph. The candidate boundary records normalized
  value, stream-channel, and memory segments plus one `source_map` per stream
  input.
* Publishes graphs on a cloned scratch module and replaces the live module
  only after every candidate converts and the native finalized-program
  validator succeeds. Failure leaves temporary candidates and cannot expose a
  canonical graph containing SCF or CFG.
* Propagates a nested graph launch completion through enclosing `scf.if`
  results before adding it to the thread completion frontier. Other enclosing
  structured controls fail before publication.
* Stream input/output channel segments become payload-typed graph stream
  ports and remain channel bindings only at `dataflow.graph.launch`. Each
  input binding preserves its `source_map`. The recursive graph owner
  rendezvous each receive/send with its structured execution frontier, removes
  the endpoint, and erases every transient channel argument before
  verification. A lowering-only schedule tree gives every sequential or
  structured mutually exclusive static site one binding-wide ordinal. Each
  activation emits that fixed ordinal sequence and mechanically filters
  inactive choice sites, so unequal or empty branches remain one ordered
  dynamic sequence without making later branch conditions startup
  dependencies. The filtered ordinal demuxes graph stream inputs and muxes
  graph stream outputs. Choice selectors are materialized before structured
  control is erased, enclosing loops repeatedly activate the schedule, and
  schedule close/reset joins the recursive execution frontier. Parallel
  endpoint sites without a deterministic total order remain ambiguous and
  fail atomically without deriving an order from traversal.
* The graph `FunctionType` contains only application payloads; normalized
  `input_segments` and `result_segments` record value, stream, and memory
  kinds.
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
  Canonical roots are explicit graph memory inputs, fresh `memref.alloc`
  results, and verified side-effect-free views. Conversion bridges preserve an
  established root but never create one. Globals and static pointer bases must
  be resolved outside the graph and imported explicitly. Boundary capabilities
  without explicit no-alias evidence share one conservative alias root.
* Each live alias partition has exactly one
  `(write_frontier, read_frontier)` pair. Straight-line accesses and recursive
  selection and repeat transfer follow `docs/spec-compiler-part-3-mem.md`.
  Graph-owned parallel transfer requires a provenance-marked, compile-time
  fixed P[] representation selected by an upstream owner. Each lane lowers
  recursively from the same incoming state, and incomparable execution and
  per-partition frontier exits join with all-of. Structural execution remains
  a separate state component.
* A graph stream input remains unbounded payload. A receive rendezvous is
  aligned by its structured execution input, and a send's completion becomes
  part of that same execution frontier. Channel handles and channel endpoint
  operations never survive in the graph body.
* Unselected, dynamic-width, resource-mapped, resultful, or reduction-bearing
  parallel SCF, unsupported residual containers, observable operations without
  explicit completion events, and memory capabilities transported on dataflow
  control primitives fail closed.
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

* Direct DFG simulation calls the native finalized-graph validator before
  execution. The native Mapping core separately validates an exact
  TechMapping against its Canonical Dataflow Program and Fabric Hardware
  Description before constructing `FrozenRealizationGraph` and
  `FrozenRoutingGraph`. These gates are mandatory and are not replaceable by
  Python preprocessing.
* The gate rejects residual SCF/CFG/region containers, memory capabilities on
  dataflow transport primitives, nontrivial graphs that use raw start as a
  completion witness, and retirement frontiers that fail to cover payloads,
  memory operations, observable effects, or stateful close/reset.
* The gate statically requires exact-one value outputs and completion
  witnesses and a proven close/commit for stream outputs. A zero-or-more
  stream path does not become exact-one merely by passing through
  `dataflow.sync`; the narrow exception is a direct graph stream input
  rendezvous limited by an exact-one activation input.
* The program-level portion is the single owner of channel topology. It
  rejects channel producers, escapes, missing or duplicate producer bindings,
  missing consumers, rank mismatches, and `source_map` relations that cannot
  be proven in bounds over the consumer domain.
* `done_out = all_of(graph.return.complete)`. Validation may prove that the
  declared frontier covers required behavior, but it does not synthesize an
  alternate completion event from effect scans or graph quiescence.
* The simulator treats the declared frontier as retirement authority. It does
  not add a post-retirement quiescence rule or synthesize another completion
  event. Within one invocation, imported-root re-exports and fresh
  `memref.alloc` exports preserve alias-class membership; contents remain
  memory observables rather than scalar payloads. The report records a derived
  invocation-local alias-class label for each imported and exported memory
  port, separate from the memory contents. Numeric labels may be reused by a
  later invocation and are not stable object identities. Stable
  cross-invocation memory-object identity remains unimplemented and blocks any
  artifact or simulator consumer that requires such correlation.
* There is no direct PnR frontend in this implementation. Mapping MLIR
  persistence and parsing, a fully resolved PnR Config, search, and the
  Physical Mapping delta remain unimplemented.

## 2. Testing Strategy

Tests are organized by stable semantic boundary:

* `test/dataflow/unit/graph/` verifies the payload-only function type,
  normalized segment sizes, exact per-segment kinds and types, mandatory
  non-empty completion, and launch symbol/type matching.
* `test/raise/` verifies graph extraction, recursive SCF lowering, canonical
  per-partition memory frontiers, zero-trip and descending loops, nested
  selection/repeat structure, index-domain carry narrowing, transactional
  rollback, fixed-width graph parallel transfer, one-shot and looped stream
  boundary publication, unequal and empty stream choices, stream choices
  driven by earlier receives, fail-closed diagnostics for unselected parallel
  residue, pointer capability transport, and effects without completion
  events.
* `test/dfg/` verifies the strict native finalized-graph gate independently
  of the simulator.
* `test/simulator/` verifies retirement-time execution, value and stream
  segments, imported-root and fresh memory exports, phase/reset/re-entry
  behavior, vector pack/serialize, scalar broadcast, pointer and integer
  primitives, dynamic extents, known library operations, and artifact
  simulation.
* `test/mapping/` verifies exact TechMapping identity/reference closure,
  configured-function correspondence, and deterministic realization/routing
  freezes.
* `test/pnr/` verifies checked native index behavior and that the removed
  rematcher and JSON-input tools cannot return. It is not a placement or
  routing test matrix.
* `test/adg/` verifies deterministic builder output and exact canonical Fabric
  fixtures where retained. It does not claim complete placement or routing.

Residual-SCF execution inside a finalized graph is not a supported behavior.
Fixtures whose only contract was leaf reconstruction through residual
containers are represented by compact strict-finalization rejection tests.
Operation semantics unrelated to that rejected behavior remain covered at
their native raise, simulator, Mapping, or ADG boundary.

## 3. Acceptance Criteria

The Part 3 slice is coherent only when all of the following hold:

* Graph `FunctionType` contains only application payloads, and normalized
  segment metadata is the single authority for value, stream, and memory
  classification.
* Graph and return verifiers reject count, kind, recursive capability, or exact
  type mismatches.
* Recursive lowering preserves one canonical
  `(write_frontier, read_frontier)` pair per live alias partition and applies
  the documented selection and repeat transfer rules. Graph-owned parallel
  transfer enters only after its Structured Program Candidate representation
  has been selected.
* Retirement is the non-empty explicit `graph.return.complete` all-of.
  Nontrivial graphs cannot use raw start, and no effect scan or quiescence rule
  creates a second completion mechanism.
* Final values, stream close/boundary commit, memory capability establishment
  and visibility, observable effects, stateful close/reset, and non-detached
  async work are causally covered by the declared frontier.
* Value outputs and completion witnesses are statically exact-one, and stream
  outputs have a statically proven close/commit. Dynamic simulator success is
  not a substitute for this proof.
* Every finalized channel root has exactly one producer binding, at least one
  consumer binding, and a valid `source_map` relation. Channel use and topology
  are checked once at program scope.
* `loom.spatial_region` never appears in a finalized program. Graph and launch
  publication is atomic across all staged candidates, and unsupported stream
  endpoint conversion fails without exposing partial canonical IR.
* Pointer or memref capabilities that cannot be projected into the explicit
  memory plane fail closed rather than entering dataflow carry or selection.
* Direct simulation rejects residual structured containers through the native
  finalized-program validator. TechMapping validation and realization/routing
  freeze consume only a Canonical Dataflow Program that has passed that gate.
* Unsupported simulator boundary semantics fail through the shared
  finalized-graph gate without flattening segment kinds.
* Validated TechMapping plus `FrozenRealizationGraph` and
  `FrozenRoutingGraph` are available as native C++ library boundaries.
  Mapping MLIR persistence/parser, fully resolved PnR Config, search, and the
  Physical Mapping delta are explicit unimplemented boundaries.
* Focused raise, DFG, simulator, Mapping, PnR-index, Fabric-Tech, and ADG suites
  pass, followed by the full locked `check-fabric` target and
  `git diff --check`.

## 4. Maintenance and Extension Points

* Adding a structured graph operation requires one recursive transfer rule in
  `GraphRegionLowering.cpp`, a focused raise anchor, and strict native-gate
  coverage if the operation may survive finalization.
* Increasing alias precision must strengthen canonical root or partition
  selection without creating persisted dependence metadata as a second
  authority. Conservative boundary aliasing remains the fallback.
* Supporting another memory capability representation requires updating the
  graph port verifier, explicit frontend classification, canonical-root
  traversal, simulator boundary handling, and Mapping validation together.
* Adding a completion-bearing operation requires a causal witness that can
  enter `graph.return.complete`; effect-only acceptance without such a witness
  is not permitted.
* ADG support for new retirement shapes should add only the operation modes and
  routes required by canonical graphs, with deterministic builder output and
  ConfiguredFunction projection anchors where semantic correspondence changes.
* A future PnR frontend must begin from canonical Mapping MLIR and a fully
  resolved PnR Config, call the existing native validation/freeze APIs, and
  emit only a Physical Mapping delta that references its exact TechMapping
  predecessor. It must not restore graph/Fabric rematching or JSON inputs.
* `Dataflow_GraphOp::build(...)` accepts a payload-only
  `FunctionType` plus normalized segment attributes. The body adds the
  separate leading `start : none` argument. Per-launch start and done use the
  separate `Dataflow_GraphLaunchOp` protocol operands/results.

## 5. References

* `docs/spec-compiler-part-3-dfg.md` -- Part 3 main spec (boundary
  contracts, SCF flattening templates, verifier invariants).
* `docs/spec-compiler-part-3-mem.md` -- canonical alias partitions,
  recursive `(write_frontier, read_frontier)` state, and token wiring owned by
  graph-region lowering.
* `docs/spec-compiler-part-3-placement-framework.md` -- software placement
  policy outside the canonical graph ABI and publication mechanics.
* `docs/spec-compiler-part-4-partitioned-data.md` -- partitioned-data spec; the
  test plan above defers to Part 4 for partitioned-data unit-test
  coverage.
* Upstream MLIR references used by the passes above are listed in
  Part 3's References section.
