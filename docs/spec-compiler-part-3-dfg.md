# Loom Compiler Part 3: SCF to DFG

This document specifies the third compiler part of the Loom front-end:
lowering SCF-shaped accelerator regions into Loom's native dataflow
representation, ready for the existing `fabric` lowering tool-chain.
It starts after source integration and LLVM-to-SCF raising have already
selected explicit accelerator regions. It does not decide which source
program regions should run on AccCores.

The canonical IR sources of the existing `dataflow` and `fabric` dialects
are `include/Dataflow/IR/*.td` and `include/Fabric/IR/*.td`; the verifier
implementations live in `lib/Dataflow/IR/*.cpp` and `lib/Fabric/IR/*.cpp`
respectively. This spec modifies the `dataflow` dialect (additive
changes only), consumes the temporary `loom.acc_region` op produced by
Part 2, and introduces a new pass library under `lib/Frontend/`.
The precise timing semantics of `dataflow.stream`, `dataflow.carry`,
`dataflow.invariant`, and `dataflow.gate` are specified separately in
`docs/spec-dataflow-part-1-streaming.md`. The precise firing semantics
of `dataflow.constant`, `dataflow.sync`, `dataflow.mux`, and
`dataflow.demux` are specified separately in
`docs/spec-dataflow-part-2-control.md`.

## 1. Scope and Contract

The compiler front-end is documented in three parts:

* **Part 1, source integration.** LLVM IR plus Loom metadata is the
  source-facing compiler contract. Any high-level language provider may
  participate if it can emit that contract; embedded clang for C / C++
  is the first limited provider.
* **Part 2, LLVM to SCF.** LLVM/CFG-shaped input is raised to
  SCF-shaped MLIR. This part selects accelerator regions, recognizes
  parallel loops, and materializes memory-region metadata. Its output
  uses `loom.acc_region` to mark code selected for AccCore execution.
* **Part 3, SCF to DFG.** This document. It consumes SCF-shaped
  accelerator regions and lowers them to `dataflow.thread` and
  `dataflow.graph`.

Input to this part is an MLIR module with `func.func` host containers.
Host code may remain outside accelerator regions. AccCore code must be
inside explicit `loom.acc_region` ops, except in the
`wrap-standalone-kernel` test mode described under
`loom-normalize-acc-regions`. A `func.func` is therefore an ABI and
ownership container, not an implicit accelerator boundary.

Output is the canonical Loom front-end IR: module-level `func.func`
symbols holding ordinary HostCore or ScalarCore code and zero or more
`dataflow.thread` regions, each holding zero or more `dataflow.graph`
regions with no `scf.*` left inside any graph body. All `scf` ops are
supported inside accelerator regions:
`scf.if`, `scf.while` with `scf.condition`, `scf.for` with
`scf.yield`, `scf.forall` with `scf.forall.in_parallel`,
`scf.parallel` with `scf.reduce` and `scf.reduce.return`,
`scf.index_switch`, `scf.execute_region`. Tensor-result aggregation in
`scf.forall` is supported by materializing `scf.forall.in_parallel`
combining actions into explicit destination-buffer effects before
thread promotion. `dataflow.thread` itself remains a launch region
with a completion token and mapped-memory data transfer, not a
tensor-result returning op. Memory dependence construction runs in
this part; alias analysis is only the conflict oracle used by that
builder (see "Memory Dependence Model").

## 2. Hardware Model

Loom's execution target is a heterogeneous chip composed of a single
HostCore plus a fabric of AccCores. Each AccCore is a fused
`(ScalarCore + SpatialCore)` pair; the SpatialCore is the CGRA tile
described by the body of one `fabric.module`, while the ScalarCore is
described by hardware parameters carried on that same `fabric.module`.

The front-end's IR mirrors this trio:

| Hardware | Front-end IR carrier |
|----------|----------------------|
| HostCore | Host-call-context `func.func` body code outside any `dataflow.thread` |
| Mesh of AccCores | An outermost `dataflow.thread` with `mapping = [#loom.spatial<...>, ...]` (hardware grid coords) and optional non-spatial dims for time-multiplexed work |
| ScalarCore on one AccCore | The body of the innermost `dataflow.thread`, minus its `dataflow.graph` regions, plus ScalarCore-legal `func.call` callees after inlining or specialization |
| SpatialCore on one AccCore | Each `dataflow.graph` nested inside the innermost `dataflow.thread` |

A single `dataflow.thread` instance corresponds to one launch of a
multi-dimensional iteration domain, distributed across a mesh of
AccCores. The thread body is "what one AccCore runs"; the spatial
coords pulled from the mapping attribute identify which AccCore.

### 2.1 IR Carrier Responsibilities

* `func.func` is a callable symbol and ABI unit. It does not by itself
  choose HostCore or AccCore placement. A function may be HostCore-only,
  ScalarCore-callable, or legal in both contexts depending on the
  Part 2 call-context classification.
* `loom.acc_region` is a temporary Part 2 to Part 3 marker for a
  structured region selected for AccCore execution. This part consumes
  it and erases it.
* `dataflow.thread` is the final AccCore execution boundary. It carries
  launch operands, mapping, mapped-memory transfer information, and the
  completion token.
* `dataflow.graph` is a SpatialCore leaf DFG. It represents CGRA work
  and cannot contain `func.func`, `func.call`, `dataflow.thread`, or
  another `dataflow.graph`.

Function definitions remain module-level symbols in this design.
`dataflow.thread` is not a symbol table and does not physically contain
`func.func` definitions. A `func.call` inside a `dataflow.thread` body
is a ScalarCore call. If the callee contains code that must become
`dataflow.graph` or nested `dataflow.thread`, Part 3 must inline or
specialize that callee into the active thread context before graph
extraction. Non-inlined ScalarCore calls may remain only when their
callee body is graph-free after this preparation.

## 3. Constitutional Rules

The seven rules below are invariants that downstream passes and
verifiers must enforce; the rest of this spec is a refinement of how
each rule lands in IR.

1. `dataflow.thread` is the outermost parallel and host/accelerator
   boundary primitive. Multi-level nesting is allowed; depth has no
   hard upper bound but is documented as recommended at most three
   levels (outer spatial grid, inner SIMT-style fan-out, innermost
   ScalarCore). Future hardware that calls for deeper nesting can
   raise that recommendation without an IR change. The thread body has
   a leading `thread_ctrl : none` block argument that fires once the
   thread instance starts executing on an AccCore. The body may contain
   ScalarCore operations and ScalarCore-legal `func.call` operations,
   but not `func.func` definitions.
2. `dataflow.graph` is a leaf-level region. Its body must not contain
   any `func.func`, `func.call`, `dataflow.thread`, or another
   `dataflow.graph`. The graph body is a single graph-kind region; it
   already permits feedback edges (existing semantics).
3. Every `dataflow.graph` has explicit control ports: a leading
   `ctrl_in : none` operand, a matching leading region block
   argument, a leading `done_out : none` result, and a matching
   leading `dataflow.yield` operand. These `none` values are real SSA
   values in the operation state, because they lower to physical
   start/done ports on hardware. Custom assembly may hide or compress
   them for readability, but generic form and verifier logic treat
   them as ordinary operands/results. The contract is "graph clients
   may begin issuing memory ops once `ctrl_in` is hot; `done_out`
   becomes hot when every memory op in the graph has retired."
4. The HostCore-to-AccCore data plane is mediated by
   `dataflow.map_info`. Every value that crosses a thread boundary as
   data (memref, spatial-array handle) must pass through one
   `dataflow.map_info` op before being consumed inside the thread.
5. Memory dependence construction runs in the front-end. Alias
   analysis answers whether two memory accesses can conflict; program
   order and structured-control-flow order give edge direction; the
   dependence builder wires `ctrl` and `done` tokens inside each graph.
   The default alias oracle is a simple SSA-source-of-memref oracle; a
   stronger oracle based on `mlir::AliasAnalysis` ships alongside it
   and the two are interchangeable through one C++ interface, both
   validated by the same lit suite.
6. `loom.acc_region` is the explicit AccCore selection boundary
   consumed by this part. `scf.forall` with a
   `mapping = [#loom.spatial<...> | #loom.temporal<...>, ...]`
   attribute is the trigger for thread promotion only inside such a
   region. A scalar-only accelerator region may be normalized into a
   synthetic 1x1 mapped `scf.forall`, but ordinary host code outside
   `loom.acc_region` is never promoted merely because it appears in a
   `func.func`. Before promotion, every `scf.forall` must be in effect
   form: no `shared_outs`, no op results, and an empty
   `scf.forall.in_parallel` terminator. Tensor-result aggregation is
   lowered to explicit destination-buffer writes before this point.
   `scf.parallel`, `scf.forall` without mapping, and plain `scf.for` /
   `scf.while` / `scf.if` / etc. are flattened inside
   `dataflow.graph` regions.
7. `dataflow.thread` and `dataflow.graph` are both
   `IsolatedFromAbove`. No operation inside either region may directly
   use an SSA value defined in the surrounding scope. Every boundary
   value must appear as an explicit op operand and as a matching entry
   block argument. For `dataflow.thread`, this operand list is the
   HostCore-to-AccCore launch ABI: memrefs and spatial-array handles
   cross through `dataflow.map_info`, while scalar values cross by
   value. For `dataflow.graph`, operands and results are the explicit
   SpatialCore data/control ports. `dataflow.thread` implements its own
   boundary memory-effect summary from mapped operands; it does not rely
   on recursive region effects to expose host-visible memory behavior.

## 4. Glossary

* **HostCore.** The general-purpose CPU that runs host-call-context
  `func.func` body code outside any `dataflow.thread`.
* **AccCore.** One CGRA-attached compute element described by one
  `fabric.module`. Composed of a ScalarCore plus a SpatialCore.
* **ScalarCore-callable function.** A module-level `func.func` that
  Part 2 classified as legal to call from code running inside a
  `dataflow.thread`. Such a function remains a symbol; Part 3 either
  preserves calls to it as ScalarCore calls or inlines / specializes it
  before graph extraction.
* **Spatial dim.** A grid dim of a `dataflow.thread` that is mapped
  to a physical core-grid coordinate of the AccCore mesh.
* **Temporal dim.** A grid dim of a `dataflow.thread` that is
  time-multiplexed on the same AccCores (analogous to the SPGPU paper's
  z-dim).
* **Spatial array.** A `memref<...>` annotated with a tile-and-mesh
  layout descriptor; lets in-thread code query its local tile via
  `dataflow.local_range`.
* **Mapping attribute.** Any attribute that implements
  `mlir::DeviceMappingAttrInterface`; the front-end ships
  `#loom.spatial<...>` and `#loom.temporal<...>` instances and
  consumes any other implementation transparently.
* **Thread token.** A value of type `!dataflow.thread_token`, a
  one-shot completion signal modelled on `!async.token`.
* **Thread control token.** The leading `none`-typed block argument of
  a `dataflow.thread` body. It is the per-instance AccCore start
  signal used to launch root `dataflow.graph` regions or ScalarCore /
  SpatialCore fences.
* **Thread fence.** A ScalarCore barrier op that waits at a precise
  `dataflow.thread` body program point for zero or more SpatialCore
  `none` tokens and/or child `!dataflow.thread_token` values, then
  emits a `none` token usable as a `dataflow.graph` `ctrl_in`.
* **Map info handle.** A value of type `!loom.mapped<T>` produced by
  `dataflow.map_info`; carries direction (`to`/`from`/`tofrom`) and
  optional bound information.
* **Mem alias oracle.** A `MemAliasOracle` C++ interface returning
  `MustNotAlias` / `MayAlias` / `MustAlias` for any pair of memory
  access ops inside one `dataflow.graph`. It answers conflict only;
  it does not define execution order.
* **Memory dependence edge.** A directed edge `p -> o` saying memory
  access `o` must wait for memory access `p` before issuing its
  side effect or externally visible read.
* **Loop-carried memory state.** A hidden `none`-typed control state
  carried by a lowered loop for one alias/dependence partition. It
  represents "all memory effects in this partition from previous
  dynamic iterations have retired."
* **Aggregation-form forall.** An `scf.forall` with `shared_outs`,
  op results, or non-empty `scf.forall.in_parallel` combining actions
  such as `tensor.parallel_insert_slice`.
* **Effect-form forall.** An `scf.forall` with no `shared_outs`, no
  op results, and an empty `scf.forall.in_parallel` terminator. Its
  observable behavior is expressed through explicit memory effects.

## 5. IR Additions

This section enumerates every new dialect element the front-end
introduces. All additions are local to the existing `dataflow` and
new `loom` namespaces; nothing outside this list is added.

### 5.1 New Types

* `!dataflow.thread_token`
  - One-shot completion signal. Equivalent of `!async.token` for the
    Loom front-end.
  - Refcounted by a future runtime ABI; first milestone only manipulates
    the type as an SSA value.

* `!loom.mapped<T>`
  - Wraps any allowed inner type `T` (memref, spatial-array-annotated
    memref, or a future spatial-array type) plus a `direction` enum
    `to | from | tofrom | alloc | release`. Produced by
    `dataflow.map_info`; consumed as a `dataflow.thread` boundary
    operand when a mapped data object enters the AccCore program.
  - `T` is preserved for downstream type matching.

### 5.2 Attribute Interface Instances

Two new attribute classes implement the upstream
`mlir::DeviceMappingAttrInterface`:

* `#loom.spatial<x | y | z | linear_dim_0 | ... | linear_dim_9>`
  - A grid dim of a `dataflow.thread` is bound to a physical core-grid
    coordinate.
  - `getMappingId()` returns a closed enum value; `isLinearMapping()`
    is `true` for `linear_dim_*` and `false` for `x | y | z`;
    `getRelativeIndex()` returns the position within the spatial
    group.

* `#loom.temporal<x | y | z | linear_dim_0 | ... | linear_dim_9>`
  - Same shape as `#loom.spatial<...>` but marks a time-multiplexed
    dim. The lowering pass keeps temporal dims as ordinary block-arg
    indices on the thread body and does not bind them to physical
    coordinates.

The two attribute classes are deliberately symmetric so that a future
optimizer can swap a temporal dim for a spatial one without touching
op shapes.

### 5.3 New Operation Interfaces

* `LoomAsyncOpInterface`
  - Shape mirrors upstream `GPU_AsyncOpInterface`: the op accepts a
    variadic operand prefix of `!dataflow.thread_token` dependencies
    and optionally produces a `!dataflow.thread_token` result.
  - First milestone has only `dataflow.thread` and
    `dataflow.thread.wait` implementing it; future memory
    ops at the host scope (alloc, memcpy) can adopt it later.

### 5.4 New Operations (signatures only)

Each op below is given by its TableGen-level signature: arguments,
results, regions, traits. Implementation bodies are out of scope for
this spec.

#### 5.4.1 `dataflow.thread`

```
arguments:
  Variadic<Dataflow_ThreadToken>:$asyncDependencies,
  Variadic<Index>:$dynamicGridLowerBound,
  Variadic<Index>:$dynamicGridUpperBound,
  Variadic<Index>:$dynamicGridStep,
  DenseI64ArrayAttr:$staticGridLowerBound,
  DenseI64ArrayAttr:$staticGridUpperBound,
  DenseI64ArrayAttr:$staticGridStep,
  DeviceMappingArrayAttr:$mapping,
  Variadic<AnyType>:$bodyOperands;
results:
  Optional<Dataflow_ThreadToken>:$asyncToken;
regions:
  SizedRegion<1>:$body;
traits:
  AttrSizedOperandSegments,
  AutomaticAllocationScope,
  IsolatedFromAbove,
  SingleBlockImplicitTerminator<"ThreadYieldOp">,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
  DeclareOpInterfaceMethods<LoomAsyncOpInterface>.
```

* `mapping` is a `DenseArrayAttr` of `DeviceMappingAttrInterface`,
  one per grid dim. Mixed `#loom.spatial<...>` and
  `#loom.temporal<...>` in the same array is allowed; the relative
  order in the array equals the relative order of the grid dim.
* `bodyOperands` is the complete explicit set of non-grid values that
  cross into the thread body. The thread is `IsolatedFromAbove`, so
  these operands are the only way the body can refer to surrounding
  SSA values.
* The entry block has one leading `thread_ctrl : none` block argument,
  then one block argument per grid dim (the iteration index, an
  `index`), followed by one block argument per body operand. A scalar
  body operand appears as a block argument of the same type. A
  `!loom.mapped<T>` body operand is unwrapped to a block argument of
  type `T`, so in-thread code sees the mapped object directly while
  the boundary still records the mapping protocol.
* `thread_ctrl` is produced by the thread launch once async
  dependencies are satisfied and the AccCore instance begins
  execution. Root `dataflow.graph` ops with no ScalarCore predecessor
  use this value as their `ctrl_in`.
* `staticGrid*` arrays carry the static bounds; entries equal to
  `ShapedType::kDynamic` refer to the corresponding `dynamicGrid*`
  operand. `OpFoldResult` helpers (`getMixedGridLowerBound()` etc.)
  are exposed on the C++ class.
* The op produces an `Optional<!dataflow.thread_token>` result. The
  first milestone always produces it (pure async style); the
  optional shape leaves room for a later non-async lowering.
* The op has no data results. Values produced by AccCore execution
  cross the HostCore-to-AccCore boundary through mapped memory effects;
  the token is the readiness signal for those effects.
* `dataflow.thread` implements `MemoryEffectOpInterface` directly. The
  interface reports host-visible effects by projecting each
  `!loom.mapped<T>` body operand back through its defining
  `dataflow.map_info` op to the map source:
  - `direction = to` reports `Read` on the source.
  - `direction = from` reports `Write` on the source.
  - `direction = tofrom` reports both `Read` and `Write` on the source.
  - `direction = alloc` reports `Allocate` on the source.
  - `direction = release` reports `Free` on the source.
  Scalar body operands do not contribute memory effects.
* Effects are reported on the `dataflow.map_info` source value, not on
  the `!loom.mapped<T>` handle. In nested-thread cases that source may
  itself be a block argument of the enclosing thread body; the parent
  thread's own boundary summary is responsible for projecting its
  effects one level further outward.
* Recursive inspection of the thread body may be used by verifier or
  diagnostics to check that the body respects its declared boundary
  operands, but it is not the external effect contract of
  `dataflow.thread`.

#### 5.4.2 `dataflow.thread.yield`

```
arguments:
  none;
results:
  none;
regions:
  none;
traits:
  Terminator,
  ParentOneOf<["::dataflow::ThreadOp"]>,
  Pure.
```

* The operand list is intentionally empty. Tensor-result aggregation
  from `scf.forall` is materialized into explicit destination-buffer
  writes before thread promotion, so `dataflow.thread` does not need a
  parallel combining region or thread data results. Values defined
  inside an isolated thread body never escape by direct SSA use.

#### 5.4.3 `dataflow.thread.fence`

```
arguments:
  Variadic<AnyTypeOf<[NoneType, Dataflow_ThreadToken]>>:$deps;
results:
  NoneType:$ctrl;
regions:
  none;
traits:
  ParentOneOf<["::dataflow::ThreadOp"]>,
  MemoryEffectOpInterface.
```

* ScalarCore-side fence and token bridge. It must appear directly in a
  `dataflow.thread` body, outside any `dataflow.graph`.
* With no operands, it fires when ScalarCore execution reaches this
  program point and all preceding ScalarCore side effects in the
  thread body are complete; it then emits a `none` token.
* With operands, it also waits for every SpatialCore `none` token and
  child `!dataflow.thread_token` dependency before emitting its result.
  The result can feed a `dataflow.graph` `ctrl_in`, so the op bridges
  child-thread completion into SpatialCore graph control.
* This op is the only sanctioned bridge between thread completion and
  graph-level control. There is no general cast between
  `!dataflow.thread_token` and `none`. Ordering a child thread after a
  graph completes is expressed by placing the child launch after
  `dataflow.thread.fence(%graph_done)` in ScalarCore program order.
* ScalarCore operations after the fence in thread-body program order
  are sequenced after the fence. Consuming a graph `done_out` with this
  op therefore expresses "wait for SpatialCore completion before
  continuing on ScalarCore". Placing a nested `dataflow.thread` after
  such a fence expresses "launch the child thread only after the graph
  is complete".

#### 5.4.4 `dataflow.thread.wait`

```
arguments:
  Variadic<Dataflow_ThreadToken>:$asyncDependencies;
results:
  none;
traits:
  DeclareOpInterfaceMethods<LoomAsyncOpInterface>.
```

* Synchronous wait in the enclosing control context: HostCore for an
  outer thread launch, ScalarCore code for a nested thread launch.
  After this op, all listed thread tokens are guaranteed complete.
  Inside a `dataflow.thread`, prefer `dataflow.thread.fence` when the
  wait result must feed a SpatialCore `ctrl_in`.

#### 5.4.5 `dataflow.map_info`

```
arguments:
  AnyType:$source,
  Loom_MapDirectionAttr:$direction,
  OptionalAttr<DenseI64ArrayAttr>:$staticBounds,
  Variadic<Index>:$dynamicBounds;
results:
  Loom_MappedType:$mapped;
traits:
  Pure.
```

* `source` is a `memref<...>` (or a spatial-array-annotated memref
  in a later milestone).
* `direction` is the closed enum `to | from | tofrom | alloc |
  release`. The first milestone defaults every front-end-injected
  `map_info` to `tofrom`; an optional optimizer can later refine to
  the narrowest direction.
* `staticBounds` / `dynamicBounds` together describe the per-dim
  half-open `[lo, hi)` ranges that the thread will touch. Empty
  bounds mean "the entire memref"; partial information is
  represented with `ShapedType::kDynamic` sentinels.
* The op is `Pure`; alias analysis and bufferization can treat it as
  a typed view.

Spatial-array related ops (`dataflow.spatial_layout`,
`dataflow.local_range`, `dataflow.spatial_coord`,
`dataflow.spatial_linear_id`) are specified in
`docs/spec-compiler-part-4-spatial.md`. `dataflow.spatial_layout`
appears at host scope or inside a `dataflow.thread` body (the
ScalarCore portion); the query ops appear only inside a thread
body. None of them appear inside a `dataflow.graph`, and none of
them participate in the SCF flattening templates in this document.

### 5.5 Modifications to Existing Ops

* `Dataflow_GraphOp` (graph region container).
  - The op remains `IsolatedFromAbove`; all values used inside the
    graph body must enter through explicit graph operands and entry
    block arguments.
  - The operand list grows an explicit leading `none`-typed operand
    `ctrl_in`. Existing user inputs follow.
  - The entry block grows a matching leading `none`-typed block
    argument, also named `ctrl_in` by convention. Existing user block
    arguments follow.
  - `dataflow.yield` operand list grows an explicit leading
    `none`-typed value `done_out`. Existing user yield values follow.
  - The result type list of `dataflow.graph` grows an explicit
    leading `none` result. That result is the graph completion token.
  - The verifier checks that the leading control operand, leading
    block argument, leading yield operand, and leading result all have
    type `none`, and that data operands/results still match their
    corresponding block arguments and yield values after the control
    slot is skipped.
  - The custom parser/printer may offer a compact form for the
    control ports, but the generic form must expose the leading
    operand/result. No analysis may depend on a hidden compiler-global
    graph start or completion state.
  - All existing `dataflow.graph` lit tests are migrated in the same
    change as this spec: each test grows the explicit control operand,
    block argument, result, and `done_out` plumbing.

* `Dataflow_YieldOp`.
  - The verifier's parent-result-count and parent-result-type checks
    are updated to know about the leading explicit control result.

* No other existing op is modified by this milestone.

## 6. Lowering Pipeline

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

### 6.1 `loom-normalize-acc-regions`

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
  body is structured enough for the SCF lowering rules in this spec.
* Records the region-local default mapping used if a scalar-only
  accelerator region must be normalized into a 1x1 mapped forall. If
  Part 2 did not provide a mapping policy, this part uses a single
  spatial grid point as the conservative default.

### 6.2 `loom-materialize-forall-aggregation`

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

### 6.3 `loom-classify-thread-regions`

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

### 6.4 `loom-promote-map-info`

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

### 6.5 `loom-build-thread-skeleton`

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
  leading control and induction-variable arguments.
  `!loom.mapped<T>` operands become block arguments of type `T`;
  scalar operands keep their original type. The pass rewrites every
  in-body use of a surrounding SSA value to the corresponding block
  argument before the verifier sees the isolated thread.
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

### 6.6 `loom-prepare-scalarcore-calls`

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

### 6.7 `loom-extract-graph-regions`

* Within each innermost `dataflow.thread` body (no further mapped
  forall inside), groups eligible operations into one or more
  `dataflow.graph` regions.
* Eligibility for a graph region: a maximal connected sub-DAG of ops
  rooted at one or more `memref.{load, store}` operations, extended
  upward only through ops admitted by the graph body whitelist in
  §10 -- `arith.*`, `math.*`, and `dataflow.{stream, carry,
  invariant, gate, mux, demux, sync, constant}` -- and stopping at
  the first `scf.{if,while,for,parallel,index_switch,execute_region}`
  boundary that contains side-effecting code, or at any `func.call`,
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

### 6.8 `loom-build-memory-dependencies`

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

### 6.9 `loom-lower-scf-to-dfg-bodies`

* Inside every `dataflow.graph`, replaces each `scf.*` control-flow
  op with the canonical dataflow token rewrite (see "Per-scf Lowering
  Templates").
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
  all memory accesses with no dependence successor (see "Memory
  Dependence Model").

### 6.10 `loom-finalize-dfg`

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

## 7. Per-scf Lowering Templates

This section gives the canonical pseudocode template for each
`scf` op in terms of dataflow primitives. The supported op list in
"Scope and Contract" maps these templates: implement and lit-test the
simpler ops first.

The dataflow primitive set is the existing one
(`stream`, `carry`, `invariant`, `gate`, `mux`, `demux`, `sync`,
`constant`, `load`, `store`, `yield`). This section describes how SCF
ops are mechanically rewritten with those primitives. The precise
state machines and token lengths of `stream`, `carry`, `invariant`,
and `gate` are the single source of truth in
`docs/spec-dataflow-part-1-streaming.md`. The precise firing semantics
of `constant`, `sync`, `mux`, and `demux` are the single source of
truth in `docs/spec-dataflow-part-2-control.md`.

The control op set is `mux`, `demux`, `sync`, `constant`. Crucially:
the rwc bit fed into `carry` / `invariant` / `gate` does not have to
come from `stream`; any `i1` SSA stream from arbitrary computation
inside the graph plays the same role. This is what lets
`scf.while` lower without a new op.

Selection lanes follow the control-op contract. For `i1` selectors,
lane 0 is the `false` lane and lane 1 is the `true` lane:

```
%false_value, %true_value = demux %cond, %value : (i1, T) -> (T, T)
%value = mux %cond, %false_value, %true_value : (i1, T, T) -> T
```

For `index` selectors, lane `k` is operand/result position `k`.
This convention is required for the templates below to be mechanical.
`dataflow.mux` is selective: it consumes only the selector and selected
input lane. `dataflow.demux` is selective: it emits only the selected
output lane. `dataflow.sync` is the all-input rendezvous op.

Allowed pure compute ops inside `dataflow.graph`, such as
`arith.*`, `math.*`, and allowed LLVM computation ops, follow strict
all-operand firing: each dynamic firing consumes one token from every
operand and emits one token on every result. In particular,
`arith.select` is an eager three-input compute op in this model, not a
short-circuiting dataflow mux.

SSA multi-use is token broadcast. If one SSA stream value has multiple
uses, each use observes the same ordered token sequence. This is not a
destructive single-consumer read. The `scf.for` template relies on
this property because the loop rwc stream is consumed by `gate`,
`carry`, `demux`, and `mux`, and the loop-exit value may feed both
the loop result and the feedback reset lane.

Frontend `memref<...>` values are not stream values in this sense.
They represent memory-region bindings for `dataflow.load` /
`dataflow.store`. Lowering must not feed memref bindings through
stream-shaping ops; it shapes address, data, operation, and explicit
`none` memory-order streams instead. The generic result-selection
templates below apply to scalar/data streams and `none` ordering
streams. A memref-typed structured-control result inside graph
extraction must be rewritten to explicit memory effects, kept in
ScalarCore code, or rejected before graph lowering.

The templates below show user-visible SSA value lowering. Loop-carried
memory ordering is added by the Memory Dependence Model as hidden
`none`-typed state; it is not an optional optimization.

#### RWC Phasing Rule

An rwc stream is a loop-control stream, not a plain valid bit. For a
counted loop with `N` body executions, `dataflow.stream` emits `N + 1`
`(index, rwc)` pairs: `N` true pairs plus one trailing false sentinel.
The loop-level rwc, or a derived body-local close stream in the
body's phase, must remain visible to stateful stream-shaping ops that
need the sentinel to close or reset state, such as `carry`,
`invariant`, `gate`, and loop-carried memory-state wiring.

Values that enter a loop body must be in the body's phase. The
canonical way to convert a raw `(rwc, value)` pair into body phase is
`dataflow.gate`: its value result has exactly the true-condition
tokens, while its condition result is the body-local close stream. A
true body-local condition means the current body execution is not the
last execution; a false body-local condition means the current body
execution is the last execution. Memory side effects and address
computation that must not observe the sentinel index consume the gated
value stream or are guarded by the corresponding body-local rwc.

Different regions of one source loop may therefore have different rwc
streams. The loop-level rwc decides whether the source loop continues
or exits; a gated body rwc controls state local to the body region
whose value stream has already been normalized.

### 7.1 `scf.if`

Source shape:

```
%r... = scf.if %cond -> (T_r, ...) {
  ... then computation using live-in streams ...
  scf.yield %then_r... : T_r, ...
} else {
  ... else computation using live-in streams ...
  scf.yield %else_r... : T_r, ...
}
```

`scf.if` regions have no block arguments, but graph lowering must not
let branch-local computation directly consume parent-phase data streams.
For every non-memref stream live-in used by either branch, the lowering
projects the stream into branch phase with the same selector that
routes control.
The `%ctrl` stream is supplied by the current lowering context: graph
`ctrl_in` for a top-level if, loop body control for an if inside a
loop body, or a selected parent-branch control stream for a nested if.

```
%cond : i1
%t_else, %t_then = demux %cond, %ctrl : i1 -> (none, none)

# For every non-memref stream live-in %x : T used in either branch:
%x_else, %x_then = demux %cond, %x : (i1, T) -> (T, T)

# then-region runs with %t_then and %x_then...; produces %v_then...
# else-region runs with %t_else and %x_else...; produces %v_else...

%result = mux %cond, %v_else, %v_then : (i1, T, T) -> T
%done_after = mux %cond, %done_else, %done_then : (i1, none, none) -> none
```

* Each side's loads / stores fork from the side's local ctrl token
  and join back through a branch-local tail token.
* Frontend `memref<...>` bindings are not demuxed. The branch-specific
  address, data, operation, and explicit `none` order streams are
  demuxed instead.
* If a live-in is used by only one branch, the projection for the other
  branch is a dead output. Per the control-op contract, it is discarded
  by target lowering and does not require a `dataflow.drop` op or
  runtime queue.
* Mutually exclusive branch tails are joined with `mux`, not `sync`.
  `sync` is only used inside one dynamically executed path, where all
  inputs are expected to fire. The un-selected branch produces no
  done token because `demux` only fires the selected output, while the
  exit `mux` waits only for the selected branch's `done` token.
* If the `scf.if` has no else body, the false-path `done` is the
  false-path local ctrl token. If a branch has no memory side effect
  or other control-only work, that branch's `done` is its local ctrl
  token.
* MLIR requires an else region whenever `scf.if` has results. An
  `scf.if` without an else region therefore has no results; only the
  control token needs to be joined.
* Multi-result `scf.if` lowers one result mux per result position,
  all driven by the same `%cond` stream.

For a three-token parent-phase invocation with `%cond = [T, F, T]`
and a scalar live-in `%x = [10, 20, 30]`:

| Stream | Tokens |
|--------|--------|
| `%x_then` | `[10, 30]` |
| `%x_else` | `[20]` |
| `%v_then` | `[then(10), then(30)]` |
| `%v_else` | `[else(20)]` |
| `%result` | `[then(10), else(20), then(30)]` |
| `%done_after` | `[done_then0, done_else1, done_then2]` |

Branch live-in demuxing is required for phase correctness. Without it,
tokens for an unselected branch can remain buffered inside branch-local
ops and be consumed by a later selected invocation at the wrong dynamic
position.

### 7.2 `scf.while` with `scf.condition`

Source shape:

```
%res... = scf.while (%a0_i = %init_i, ...) : (A_i, ...) -> (B_j, ...) {
^before(%a_i : A_i, ...):
  %cond, %b_j... = ... before computation ...
  scf.condition(%cond) %b_j... : B_j, ...
} do {
^after(%b_after_j : B_j, ...):
  %a_next_i... = ... after computation ...
  scf.yield %a_next_i... : A_i, ...
}
```

The before-argument types `A_i` and the after/result types `B_j` are
independent. If the after region executes `K` times, the before region
executes `K + 1` times. The `scf.condition` operands are therefore in
before phase: true-cycle operands enter the after region; the single
false-cycle operand tuple becomes the while result tuple.

Canonical lowering skeleton:

```
# Structural loop entry and loop-back control. This exists even when
# the source while has no data inits.
%iter_ctrl = carry %cond, %entry_ctrl, %ctrl_feedback : none

# Each before block argument is loop-carried in before phase.
%a_i = carry %cond, %init_i, %a_feedback_i : A_i

# The before region consumes %iter_ctrl and %a_i..., then produces:
#   %cond        : i1
#   %b_j         : B_j, one stream per scf.condition trailing operand
#   %before_done : none, the tail of before-region side effects

# scf.condition true operands enter after; false operands are results.
%b_exit_j, %b_after_j =
  demux %cond, %b_j : (i1, B_j) -> (B_j, B_j)

# The same before tail opens the after execution stream and produces an
# after-local close stream.
%after_rwc, %after_ctrl =
  gate %cond, %before_done : none

# The after region consumes %after_ctrl and %b_after_j..., then
# produces:
#   %a_next_i... : A_i, the scf.yield operands
#   %after_done  : none, the after-region completion token; if the
#                  region has no side effects and no extra control-only
#                  work, this may be %after_ctrl

# False-cycle reset tokens let carry consume the final cond=false.
%a_reset_i, %unused_a_true_i =
  demux %cond, %a_i : (i1, A_i) -> (A_i, A_i)
%a_feedback_i =
  mux %cond, %a_reset_i, %a_next_i : (i1, A_i, A_i) -> A_i

%ctrl_reset, %unused_ctrl_true =
  demux %cond, %before_done : (i1, none) -> (none, none)
%ctrl_feedback =
  mux %cond, %ctrl_reset, %after_done : (i1, none, none) -> none

%res_j = %b_exit_j
```

* `%cond` is the i1 token computed by the before-region's
  `scf.condition`. There is no `stream` op here; an arbitrary `i1`
  stream produced by before-region computation drives the loop.
* The before-region executes once more than the after-region. The
  `gate` on `%before_done` captures that phase relation:
  `%after_ctrl` has exactly one token per after execution, while
  `%after_rwc` is the after-region's local close stream. A true
  `%after_rwc` token means the corresponding after execution is not
  the last one; a false token means it is the last one.
* `%after_rwc` is not an after-entry token. It must not be placed on
  the critical path that completes the same after execution. Otherwise
  the first after execution would wait for an `%after_rwc` token that
  is only produced after the next before execution, creating a cycle:
  after completion -> next before -> `%after_rwc` -> same after
  completion. After-region main computation is started by
  `%after_ctrl` and `%b_after_j`; after-local stateful streaming ops
  may use `%after_rwc` to advance or reset state for subsequent
  after executions.
* `%b_exit_j` becomes the loop result. `demux` is required because
  `gate` intentionally drops the false-cycle value.
* Each `%a_feedback_i` has length `K + 1`: `K` true-cycle values from
  the after region plus one false-cycle reset value projected from
  the current `%a_i`. Without the false-cycle reset token,
  `dataflow.carry` would wait forever on the final `cond=false`.
* Before-region invariants use the before-phase `%cond` stream.
  After-region-only invariants must be projected into after phase; a
  robust lowering first replays the value in before phase, then routes
  it through the true lane of `demux %cond`. This keeps zero-trip
  loops from producing an after-only value.
* For each loop-carried memory partition, the loop also has a hidden
  `none` carry following the same structure as `%iter_ctrl`. The
  before-region starts from that partition's incoming memory state.
  On the true path, the after-region tail feeds the next iteration's
  memory state. On the false path, the before-region tail is the
  loop-exit memory state. This preserves memory effects performed by
  the final condition-checking iteration.

For `K = 2`, the dynamic sequence is:

```
before0: cond0 = true,  b0 -> after0
after0:  yield a1
before1: cond1 = true,  b1 -> after1
after1:  yield a2
before2: cond2 = false, b2 -> while result
```

The corresponding token lengths are:

| Stream | Tokens |
|--------|--------|
| `%cond` | `[T, T, F]` |
| `%a_i` | `[a0, a1, a2]` |
| `%b_j` | `[b0, b1, b2]` |
| `%b_after_j` | `[b0, b1]` |
| `%b_exit_j` | `[b2]` |
| `%after_ctrl` | `[before_done0, before_done1]` |
| `%after_rwc` | `[T, F]` |
| `%a_next_i` | `[a1, a2]` |
| `%a_feedback_i` | `[a1, a2, a2]` |

The final `%a_feedback_i = a2` is a reset/dummy token. It is consumed
with `%cond = false` by `dataflow.carry`, emits no new before value,
and returns the carry to its init state. The same rule applies to the
structural `%ctrl_feedback` and hidden memory-state feedback streams.

### 7.3 `scf.for` with `scf.yield`

There are two distinct cases.

#### No Iter Args

Source:

```
scf.for %i = %c0 to %n step %c1 {
  %x = memref.load %A[%i] : memref<?xi32>
  memref.store %x, %B[%i] : memref<?xi32>
}
```

Lowering:

```
%i_raw, %loop_rwc = stream %lb, %ub, %step {step_op="+=", cont_cond="<"} : iN
%body_rwc, %i = gate %loop_rwc, %i_raw : iN
# body memory and address computation consume %i, never %i_raw

# Optional structured control stream when body side effects need one:
%ctrl_raw = invariant %loop_rwc, %ctrl_in : none
%loop_exit_ctrl, %body_ctrl =
  demux %loop_rwc, %ctrl_raw : (i1, none) -> (none, none)
```

For `N` dynamic body executions:

| Stream | Length | Meaning |
|--------|--------|---------|
| `%loop_rwc` | `N + 1` | `N` true tokens plus one false sentinel |
| `%i_raw` | `N + 1` | `N` body indices plus one sentinel index |
| `%i` | `N` | body-phase induction values |
| `%body_rwc` | `N` | body-local close stream, empty when `N = 0` |
| `%ctrl_raw` | `N + 1` | optional repeated control token |
| `%body_ctrl` | `N` | optional body control tokens |
| `%loop_exit_ctrl` | `1` | optional structured exit token |

The no-result case has no data loop result to compute. The only
required invariant is that body dataflow never observes the sentinel
index. If the body contains memory side effects, their ctrl operands
are wired by the Memory Dependence Model; `%body_ctrl` above is only
the canonical structured-control source when such a token is needed.
Loop-invariant memref operands are not replayed with
`dataflow.invariant`; they remain memory bindings on the lowered
loads and stores.

#### With Iter Args

Source:

```
%sum = scf.for %i = %c0 to %n step %c1
          iter_args(%acc = %init) -> i32 {
  %x = memref.load %A[%i] : memref<?xi32>
  %next = arith.addi %acc, %x : i32
  scf.yield %next : i32
}
```

Lowering:

```
%i_raw, %loop_rwc = stream %lb, %ub, %step {step_op="+=", cont_cond="<"} : iN
%body_rwc, %i = gate %loop_rwc, %i_raw : iN

%acc_raw = carry %loop_rwc, %init, %acc_feedback : i32

%acc_exit, %acc_body =
  demux %loop_rwc, %acc_raw : (i1, i32) -> (i32, i32)

# body executes only in body phase
%x = dataflow.load %A[%i], ... : memref<?xi32>
%next = arith.addi %acc_body, %x : i32

%acc_feedback =
  mux %loop_rwc, %acc_exit, %next : (i1, i32, i32) -> i32

%sum = %acc_exit
```

The iter-arg state stream is deliberately in loop phase, not body
phase. `carry` sees `%loop_rwc`, so it emits an `N + 1` state stream:
the initial value, then one carried value after each true iteration.
The same `%loop_rwc` projects that state stream:

* true lane -> `%acc_body`, exactly `N` values consumed by the body;
* false lane -> `%acc_exit`, exactly one value used as the loop
  result.

The feedback to `carry` must also have length `N + 1`. `mux` builds it
from `%next` on true iterations and `%acc_exit` on the final false
sentinel. The final false-lane feedback token is a reset/dummy token;
it lets `carry` consume `%loop_rwc = false` and return to its init
state. It is not a body execution.

For `N = 0`:

| Stream | Tokens |
|--------|--------|
| `%loop_rwc` | `[F]` |
| `%i_raw` | `[0]` |
| `%i` | `[]` |
| `%body_rwc` | `[]` |
| `%acc_raw` | `[init]` |
| `%acc_body` | `[]` |
| `%next` | `[]` |
| `%acc_exit` | `[init]` |
| `%acc_feedback` | `[init]` |
| `%sum` | `init` |

For `N = 1`:

| Stream | Tokens |
|--------|--------|
| `%loop_rwc` | `[T, F]` |
| `%i_raw` | `[0, 1]` |
| `%i` | `[0]` |
| `%body_rwc` | `[F]` |
| `%acc_raw` | `[init, next0]` |
| `%acc_body` | `[init]` |
| `%next` | `[next0]` |
| `%acc_exit` | `[next0]` |
| `%acc_feedback` | `[next0, next0]` |
| `%sum` | `next0` |

For `N = 2`:

| Stream | Tokens |
|--------|--------|
| `%loop_rwc` | `[T, T, F]` |
| `%i_raw` | `[0, 1, 2]` |
| `%i` | `[0, 1]` |
| `%body_rwc` | `[T, F]` |
| `%acc_raw` | `[init, next0, next1]` |
| `%acc_body` | `[init, next0]` |
| `%next` | `[next0, next1]` |
| `%acc_exit` | `[next1]` |
| `%acc_feedback` | `[next0, next1, next1]` |
| `%sum` | `next1` |

Multiple iter_args lower independently using the same pattern, one
`carry` / `demux` / `mux` state ring per iter_arg. Body operations may
freely combine the body-lane values from multiple iter_args before
feeding the corresponding yielded values back through their muxes.
Memref operands are not iter_arg-like stream state; only explicit
`none` memory-order state is carried for memory dependences.

* For each loop-carried memory partition, the loop has a hidden
  `none` carry initialized by the memory state before the first
  dynamic iteration. It follows the same loop-phase rule as iter_args:
  the carry is driven by `%loop_rwc`, body accesses consume the
  true-lane projected state, and the false lane is the loop-exit
  memory state. The zero-trip case forwards the initial memory state.

### 7.4 `scf.forall` with `scf.forall.in_parallel`

`scf.forall` is not lowered directly to streaming dataflow ops. It is
handled as a parallel-region normalization problem before ordinary SCF
body lowering:

1. Aggregation-form forall is materialized into effect-form forall.
2. Mapped effect-form forall becomes `dataflow.thread`.
3. Unmapped effect-form forall becomes `scf.parallel`, then follows the
   `scf.parallel` template below.

This keeps tensor aggregation, hardware thread mapping, and SpatialCore
DFG construction as separate concerns.

For this spec, an effect-form forall has no `shared_outs`, no op
results, and an empty `scf.forall.in_parallel` terminator:

```mlir
scf.forall (%i) in (%N) {
  %x = memref.load %A[%i] : memref<?xf32>
  %y = arith.mulf %x, %x : f32
  memref.store %y, %B[%i] : memref<?xf32>
  scf.forall.in_parallel {}
}
```

Its result is represented only by explicit side effects in the body.

An aggregation-form forall has `shared_outs`, op results, or a
non-empty `scf.forall.in_parallel` region. The canonical first required
combining op is `tensor.parallel_insert_slice`:

```mlir
%out = scf.forall (%i) in (%N)
    shared_outs(%o = %init) -> tensor<?xf32> {
  %v = compute(%i) : f32
  %slice = tensor.from_elements %v : tensor<1xf32>

  scf.forall.in_parallel {
    tensor.parallel_insert_slice %slice into %o[%i] [1] [1]
      : tensor<1xf32> into tensor<?xf32>
  }
}
```

`scf.forall.in_parallel` exists only to describe tensor-result
aggregation for `scf.forall`. It must not reach final dataflow IR.
After `loom-materialize-forall-aggregation`, every `scf.forall` that
Part 3 continues to lower must be in effect form.

Aggregation materialization rewrites each shared tensor result into an
explicit destination buffer:

```mlir
%buf = buffer_for_tensor_value(%init)

scf.forall (%i) in (%N) {
  %v = compute(%i) : f32
  memref.store %v, %buf[%i] : memref<?xf32>
  scf.forall.in_parallel {}
}

%out = tensor_value_from_buffer(%buf)
```

The materialization contract is:

* The destination buffer's initial contents are equivalent to the
  corresponding `shared_out` tensor.
* A `tensor.parallel_insert_slice` becomes explicit writes to the
  destination subview selected by its offsets, sizes, and strides.
* Uses of the `shared_out` block argument inside the forall body become
  reads from the same destination buffer. For example,
  `tensor.extract_slice` from the block argument is rewritten to the
  corresponding `memref.subview` or load sequence on the materialized
  buffer.
* Tensor elements not updated by any invocation keep their initial
  value.
* If multiple invocations update the same element, the source
  `tensor.parallel_insert_slice` semantics are already undefined or
  unspecified; Loom does not introduce a deterministic order.
  Read/write conflicts through the shared destination are treated the
  same way: the source forall did not provide an inter-invocation
  order, so materialization must not invent one.
* The pass preserves forall bounds, steps, induction variables, and
  `mapping` attributes.
* The produced buffers are ordinary values for boundary analysis. If a
  destination buffer crosses a mapped forall boundary,
  `loom-promote-map-info` treats it like any other memref-like value.
* If any non-empty `scf.forall.in_parallel` combining action cannot be
  materialized, lowering emits a diagnostic. Dropping the combining
  action is never legal.
* Nested aggregation-form forall follows the same materialization
  contract recursively. An inner shared destination that denotes a view
  of an outer shared destination is rewritten to the corresponding
  buffer view, and the inner combining actions become writes through
  that view.

Mapped effect-form forall is a thread boundary. A mapped forall is one
whose non-empty `mapping` attribute contains Loom-recognized
`#loom.spatial<...>` or `#loom.temporal<...>` entries:

```mlir
scf.forall (%tx) in (%N) {
  memref.store %v, %B[%tx] : memref<?xf32>
  scf.forall.in_parallel {}
} {mapping = [#loom.spatial<x>]}
```

It is promoted to `dataflow.thread` by the thread-skeleton pipeline:

```mlir
%tok = dataflow.thread ... mapping = [#loom.spatial<x>] {
^bb0(%thread_ctrl : none, %tx : index, ...):
  memref.store %v, %B[%tx] : memref<?xf32>
  dataflow.thread.yield
}
dataflow.thread.wait %tok
```

The forall grid bounds and mapping become the thread grid and mapping.
The mapping array length must equal the forall rank; this is already an
upstream `scf.forall` verifier invariant and is repeated here as an
input requirement for thread promotion.
The forall induction variables become thread entry block arguments
after the leading `thread_ctrl : none`. Values captured from outside
the forall become explicit thread operands and entry block arguments.
The empty `scf.forall.in_parallel` terminator becomes
`dataflow.thread.yield`.

This promotion creates the AccCore boundary only. Code inside the
thread body is still ScalarCore code until graph extraction moves an
eligible region into `dataflow.graph`. Only the graph body is later
lowered to SpatialCore dataflow operations. Memory operations that
remain outside any graph stay in the ScalarCore part of the thread.

The implicit synchronization point of `scf.forall` becomes explicit
thread-token ordering. The produced `!dataflow.thread_token` is either
consumed by a following thread-like op as a dependency or waited on with
`dataflow.thread.wait` at the original continuation point.

Unmapped effect-form forall is generic parallel work, not a hardware
thread boundary:

```mlir
scf.forall (%i) in (%N) {
  memref.store %v, %B[%i] : memref<?xf32>
  scf.forall.in_parallel {}
}
```

It normalizes to `scf.parallel`:

```mlir
scf.parallel (%i) = (%c0) to (%N) step (%c1) {
  memref.store %v, %B[%i] : memref<?xf32>
  scf.reduce
}
```

The upstream `scf-forall-to-parallel` conversion may be reused for this
effect-form case, because the forall has no outputs and its empty
`scf.forall.in_parallel` becomes an empty `scf.reduce`. The generated
`scf.parallel` must carry parallel provenance so that later
normalization to `scf.for` loop nests does not invent
cross-invocation memory order.

By the time this path runs, `loom-build-thread-skeleton` has already
promoted every Loom-recognized mapped forall to `dataflow.thread`.
Therefore, a forall that reaches this path must have an empty mapping
attribute. A non-empty non-Loom mapping is rejected before
normalization.

If a forall has a non-empty `mapping` attribute that Part 3 does not
recognize as a Loom mapping, the pipeline must not silently ignore it
inside an accelerator region. Part 2 or an earlier Part 3 pass must
either remove or translate that mapping with an explicit downgrade
decision, or emit a diagnostic before this template runs.

### 7.5 `scf.parallel` with `scf.reduce`

`scf.parallel` is not a second dataflow loop primitive. Part 3 first
normalizes it to one or more ordinary `scf.for` loop nests, then reuses
the already specified `scf.for` template. No new `dataflow.parallel`,
`dataflow.reduce`, or reduction enum is introduced.

A user-written `scf.parallel` with a non-empty `mapping` attribute is
rejected in the first milestone. Mapping has Loom semantics only on
`scf.forall`, because mapped forall is the construct that establishes a
`dataflow.thread` boundary.

The important semantic difference from `scf.for` is the absence of a
cross-iteration program order. The source `scf.parallel` iteration
space may execute in any order and may execute concurrently. If two
iterations race through memory, the source behavior is undefined.
Therefore, the normalization must preserve a `parallel provenance`
marker on the generated loop nests. The memory-dependence builder uses
that marker to avoid inventing loop-carried memory order between
different logical iterations or chunks of the same original
`scf.parallel`.

The normalization has a tunable split factor `K`. The pipeline option
`--parallel-split-factor=<K>` controls this value; the default is
`K = 1`, and `K` must be positive. The first milestone applies one
global split factor to every `scf.parallel` that reaches this
normalization. There is no per-loop override in the required
implementation.

* `K = 1` is the required baseline. The whole iteration domain becomes
  one lexicographic `scf.for` loop nest.
* `K > 1` is an exploration point. The iteration domain is partitioned
  into `K` ordered, disjoint chunks whose union is the original domain.
  Each chunk becomes an independent `scf.for` loop nest with the same
  body. Lowering those loop nests later naturally duplicates the
  stream/carry/gate DFG structure.
* The implementation may initially split one selected dimension into
  contiguous subranges whose boundaries are aligned to that dimension's
  step. The default selected dimension is the outermost dimension; a
  later cost model may choose another dimension, but the choice must be
  deterministic for a fixed input IR and pass option set. More advanced
  linearized or tiled partitions are legal only if they cover the
  original iteration space exactly once and assign every chunk a
  deterministic ordinal.
* For a one-dimensional contiguous split with positive `%step`, use the
  following reference arithmetic:
  ```
  %trip_count = ceildiv(max(%ub - %lb, 0), %step)
  %first_k    = floor(k * %trip_count / K)
  %limit_k    = floor((k + 1) * %trip_count / K)
  %chunk_lb_k = %lb + %first_k * %step
  %chunk_ub_k = (k + 1 == K) ? %ub : %lb + %limit_k * %step
  ```
  The last chunk uses the original upper bound so the generated loop
  preserves the source half-open range even when `%ub - %lb` is not a
  multiple of `%step`.
* `K` may exceed the dynamic iteration count. Empty chunks are legal;
  they produce `valid = false` for reductions and a normal chunk tail
  for control synchronization. The chunk tail of a zero-iteration
  chunk is the structured `scf.for` loop-exit control produced by the
  `scf.for` template: the chunk's incoming control is forwarded through
  the loop-exit path.
* Chunk loop nests start from the same parent control point. Their done
  tokens are joined only where the surrounding program needs the
  `scf.parallel` to have completed. They are not sequenced by source IR
  order unless a true external dependence requires that ordering.
  All chunk tails in one parallel-provenance group rendezvous into a
  group tail token. Any later memory access that must observe the
  parallel's memory effects depends on this group tail.
* The provenance marker must be mechanically available to Part 3
  lowering. It may be a temporary `DictionaryAttr` on the generated
  loops or an analysis side table, but it is consumed before final
  `dataflow.graph` verification. It is not a final dataflow IR feature.
* Different `K` values may produce different reduction results when the
  user's reduction region is not both associative and commutative. All
  such results are allowed by `scf.parallel`'s unspecified reduction
  order. Users who require a specific reduction order must not express
  that computation with `scf.parallel`.

For an effect-only one-dimensional parallel loop:

```mlir
scf.parallel (%i) = (%c0) to (%N) step (%c1) {
  %x = memref.load %A[%i] : memref<?xf32>
  %y = arith.mulf %x, %x : f32
  memref.store %y, %B[%i] : memref<?xf32>
  scf.reduce
}
```

the `K = 1` baseline is equivalent to:

```mlir
scf.for %i = %c0 to %N step %c1 {
  %x = memref.load %A[%i] : memref<?xf32>
  %y = arith.mulf %x, %x : f32
  memref.store %y, %B[%i] : memref<?xf32>
}
```

With `K = 2`, a valid contiguous split is conceptually:

```mlir
%mid = split_point(%c0, %N, %c1)

scf.for %i = %c0 to %mid step %c1 {
  ...
}

scf.for %i = %mid to %N step %c1 {
  ...
}
```

Both generated loop nests carry the same `parallel provenance` id and
different chunk ordinals. They represent independent chunks of one
parallel iteration space, not two source-ordered loops. If the
normalizer materializes them as adjacent SCF operations, Part 3 must
use the shared provenance id to lower them as one chunk group: all
chunks receive the group's incoming control, and their done tokens are
joined before the continuation.

If the `scf.parallel` has no results, the upstream
`scf-parallel-for-to-nested-fors` conversion may be reused for the
`K = 1` case. That upstream conversion is not sufficient for resultful
`scf.parallel`, because it rejects `scf.parallel` ops with results.
Loom must lower resultful `scf.parallel` itself.

For resultful `scf.parallel`, each result position is associated with
one initial value, one `scf.reduce` operand, and one `scf.reduce`
region. The reduction region is the reduction operator; Loom does not
encode the reduction kind as an attribute. The region is inlined at
normalization time by substituting:

* the first reduction block argument with the current accumulator;
* the second reduction block argument with the current iteration's
  reduction operand;
* `scf.reduce.return` with the yielded next accumulator value.

For `K = 1`, this becomes the same structure as `scf.for` with
`iter_args`:

```mlir
%sum = scf.parallel (%i) = (%c0) to (%N) step (%c1)
    init (%zero) -> f32 {
  %x = memref.load %A[%i] : memref<?xf32>
  scf.reduce(%x : f32) {
  ^bb0(%lhs : f32, %rhs : f32):
    %r = arith.addf %lhs, %rhs : f32
    scf.reduce.return %r : f32
  }
}
```

normalizes to:

```mlir
%sum = scf.for %i = %c0 to %N step %c1
    iter_args(%acc = %zero) -> f32 {
  %x = memref.load %A[%i] : memref<?xf32>
  %next = arith.addf %acc, %x : f32
  scf.yield %next : f32
}
```

For `K > 1`, every chunk computes a partial reduction without assuming
that the reduction has an identity element. This is required because
blindly initializing every chunk with the original `init` value would
apply the init value once per chunk. Instead, each chunk returns:

* a `valid` flag that is true iff the chunk executed at least one
  iteration;
* one `partial` value per reduction result, initialized from the first
  executed iteration in that chunk and updated by the reduction region
  for later iterations in the same chunk.

The final merge starts from the original `init` tuple and folds only
valid chunk partials in deterministic chunk-ordinal order by inlining
the same reduction regions:

```mlir
%valid0, %partial0 = reduce_chunk_0(...)
%valid1, %partial1 = reduce_chunk_1(...)

%acc0 = %zero
%acc1 = scf.if %valid0 -> f32 {
  %next0 = inline_reduce(%acc0, %partial0)
  scf.yield %next0 : f32
} else {
  scf.yield %acc0 : f32
}
%acc2 = scf.if %valid1 -> f32 {
  %next1 = inline_reduce(%acc1, %partial1)
  scf.yield %next1 : f32
} else {
  scf.yield %acc1 : f32
}
```

For multi-result `scf.parallel`, the `valid` flag is shared by all
reduction results of the same chunk. Each chunk carries one partial per
result position, and the final merge conditionally folds the accumulator
tuple. Each result position uses its own `scf.reduce` region. The
conditional fold may be represented as one multi-result `scf.if` or as
equivalent per-result control/data wiring, as long as all result
positions observe the same chunk validity.

This partial-and-merge scheme is correct for arbitrary `scf.reduce`
regions. It chooses one legal reduction order allowed by
`scf.parallel`; it does not require associativity, commutativity, or a
known identity element. If a later analysis proves stronger algebraic
properties, a tree merge or target-specific reduction network may be
selected as an optimization, but that is not part of the required
normalization contract.

After normalization, all generated `scf.for` and `scf.if` operations
use the existing templates in this section. Their stream, carry, gate,
demux, mux, and memory-order behavior is inherited from those templates.

### 7.6 `scf.index_switch`

`scf.index_switch` has the same selected-region shape as `scf.if`, but
its source selector is an arbitrary `index` value matched against a
dense array of case constants. `dataflow.mux` and `dataflow.demux`
require dense lane selectors, so lowering first normalizes the source
argument to a dataflow lane id.

Lane convention follows the operation's region order:

```
lane 0     = default region
lane i + 1 = case region i
```

The zero-case form has only the default region and lowers by inlining
that region into the surrounding graph. There is no selector, demux, or
mux. The one-case form has two dynamic lanes and uses an `i1` selector:
`false` selects default, `true` selects the single case. With two or
more cases, the normalized selector has `index` type.

For two or more cases, the normalized selector is computed as ordinary
data, not with `dataflow.mux`. A `dataflow.mux` is selective and would
leave each unselected case-lane constant token in its queue. Across
many switch invocations those leftover tokens would accumulate without
bound under any bounded-buffer runtime and would eventually saturate
hardware discard/disconnect paths. Ordinary `arith.select` follows
all-operand firing, so it consumes every candidate lane value on each
firing and leaves no residue.

```
# Normalize arbitrary case values to dense dataflow lanes.
%lane0 = dataflow.constant %ctrl {const_value = 0 : index} : index
%lane = ... compare %arg to each case value and arith.select lane i+1

%default_ctrl, %case0_ctrl, %case1_ctrl, ... =
  demux %lane, %ctrl : (index, none) -> (none, none, none, ...)

# For every non-memref stream live-in %x : T used by any selected region:
%x_default, %x_case0, %x_case1, ... =
  demux %lane, %x : (index, T) -> (T, T, T, ...)

... each selected region produces one result tuple and one done token ...

%result =
  mux %lane, %r_default, %r_case0, %r_case1, ... : (index, T, T, T, ...) -> T
%done =
  mux %lane, %done_default, %done_case0, %done_case1, ...
    : (index, none, none, none, ...) -> none
```

* This is a generalization of `scf.if`'s template after selector
  normalization. Demux routes control and non-memref live-in streams
  to exactly one selected region; mux collects the selected result and
  done token.
* The default region participates as lane 0. Case region `i`
  participates as lane `i + 1`. This is different from source case
  values; case values are used only while computing `%lane`.
* `%lane` is constructed to be in range `[0, num_cases]`: unmatched
  source values keep lane 0, while matched case `i` selects lane
  `i + 1`. No dynamic selector-out-of-range diagnostic is required at
  this lowering point.
* A selected region with no memory side effect or other control-only
  work has its done token equal to its local ctrl token.
* The one-case form uses `i1` demux/mux with the same lane convention:
  `false` is default and `true` is the single case. The comparison
  result is an ordinary SSA stream; multiple demuxes and muxes reuse it
  by token broadcast.
* Multi-result `scf.index_switch` lowers one result mux per result
  position, all driven by the same normalized selector.
* If a live-in is used by only some selected regions, projections for
  unused lanes are dead outputs and are discarded by target lowering.
* The zero-case form is spliced during
  `loom-lower-scf-to-dfg-bodies`, before the surrounding graph body is
  finalized. Memory-dependence snapshots continue to identify memory
  ops by their existing deterministic ids; the splice does not create a
  selector-dependent memory path.

For cases `[2, 5]` and argument stream `[2, 7, 5]`, the normalized
selector stream is `[1, 0, 2]`:

| Stream | Tokens |
|--------|--------|
| `%lane` | `[1, 0, 2]` |
| `%default_ctrl` | `[ctrl1]` |
| `%case0_ctrl` | `[ctrl0]` |
| `%case1_ctrl` | `[ctrl2]` |
| `%arg_default` | `[7]` |
| `%arg_case0` | `[2]` |
| `%arg_case1` | `[5]` |
| `%r_default` | `[default(arg=7)]` |
| `%r_case0` | `[case2(arg=2)]` |
| `%r_case1` | `[case5(arg=5)]` |
| `%result` | `[case2(arg=2), default(arg=7), case5(arg=5)]` |
| `%done_default` | `[done_default0]` |
| `%done_case0` | `[done_case0_0]` |
| `%done_case1` | `[done_case1_0]` |
| `%done` | `[done_case0_0, done_default0, done_case1_0]` |

### 7.7 `scf.execute_region`

* No control structure to flatten. The pass inlines the region body
  into the surrounding scope and rewires SSA values; ctrl/done
  forwarding follows program order.

### 7.8 `scf.yield`

* Already a thin terminator. The lowering of the parent op produces
  the yield's effect; the standalone yield is removed.

## 8. Memory Dependence Model

The first milestone separates alias, dependence, and token wiring.
The first-principles requirement is to preserve the memory behavior of
the input program when the graph becomes a dataflow circuit. Alias
analysis only answers "can these two accesses touch overlapping
storage?" A memory dependence edge answers the directional scheduling
question: `o` cannot issue until `p` has retired on the same dynamic
execution path.

### 8.1 Alias Oracle

Two interchangeable oracle implementations share the `MemAliasOracle`
interface (see `loom-build-memory-dependencies` under "Lowering
Pipeline").

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

### 8.2 Dependence Builder

`MemoryDependenceBuilder` consumes a `MemAliasOracle` and produces a
directed graph over memory accesses inside one `dataflow.graph`.

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

### 8.3 Loop-Carried Memory State

A loop-carried memory dependence is represented as hidden loop state,
not as an implicit property of the loop op. The lowering must make the
state visible in dataflow primitives so graph scheduling, graph
verification, and later hardware lowering all see the same ordering.

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

The loop-state plan stored by `loom-build-memory-dependencies` is the
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

### 8.4 Token Wiring

The control-token wiring rule is derived from the dependence snapshot:

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

## 9. Spatial Array

Spatial-array layout, in-thread queries, and halo exchange are
specified in `docs/spec-compiler-part-4-spatial.md`. They are not
required for SCF-to-DFG flattening; this document references them
only at the boundary points (see §5.4 and §10).

## 10. Verifier Rules (Front-End Specific)

In addition to the existing dataflow / fabric verifier set:

* `dataflow.thread`
  - `mapping` array length equals grid dim count.
  - Every `mapping` entry implements
    `DeviceMappingAttrInterface`.
  - Static-bounds arrays match `dynamicGrid*` operand counts (mixed
    static / dynamic via sentinel).
  - The op has no data results. In the first milestone it produces
    exactly one `!dataflow.thread_token`.
  - Entry block argument count equals
    `1 + gridDimCount + numBodyOperands`. The leading argument must
    have type `none` and is the thread control token.
  - The body is `IsolatedFromAbove`: every value used in the body and
    defined outside it must have a matching body operand and entry
    block argument.
  - For each scalar body operand, the corresponding entry block
    argument type must match exactly. For each `!loom.mapped<T>` body
    operand, the corresponding entry block argument type must be `T`.
  - Each `!loom.mapped<T>` body operand must be the direct result of a
    `dataflow.map_info` op in the enclosing control context. The
    thread's `MemoryEffectOpInterface` must project effects to that
    `map_info` source according to its `direction` attribute, as
    specified in the `dataflow.thread` op contract.
  - Body must not contain a `dataflow.graph` whose body contains any
    `dataflow.thread` (graph is a leaf).
  - Body may contain `func.call` only when the callee has been proven
    ScalarCore-legal or is scheduled for inlining before graph
    extraction. Body must not contain `func.func` definitions.
  - At least one `dataflow.thread` ancestor is required for every
    `dataflow.graph`. The graph-extraction pass only runs inside
    `dataflow.thread` bodies created from explicit accelerator
    regions, or from `wrap-standalone-kernel` test-mode input.

* `dataflow.thread.yield`
  - No operand allowed. The parent thread has no data results; its
    completion token is produced by the parent op, not yielded as a
    body value.

* `dataflow.thread.fence`
  - Must appear directly in a `dataflow.thread` body, not at host
    scope and not inside `dataflow.graph`.
  - Every operand has type `none` or `!dataflow.thread_token`.
  - The result has type `none`.

* `dataflow.thread.wait`
  - At least one operand. Each is `!dataflow.thread_token`.

* `dataflow.map_info`
  - `direction` is one of the closed enum values.
  - `staticBounds` rank, if present, equals the source memref rank.
  - The op may appear at host scope or inside another
    `dataflow.thread`'s ScalarCore region; it must not appear inside
    `dataflow.graph`.

Verifier rules for `dataflow.spatial_layout`,
`dataflow.local_range`, `dataflow.spatial_coord`, and
`dataflow.spatial_linear_id` are specified in
`docs/spec-compiler-part-4-spatial.md`.

* `dataflow.graph` (modified)
  - First operand has type `none`. First block argument has type
    `none`. First yield value has type `none`. First op result has
    type `none`. All four values are the explicit graph control
    ports.
  - The graph body is `IsolatedFromAbove`: after the leading control
    slot, every user operand must have a matching entry block
    argument of the same type; every externally visible graph value
    must be a graph result produced by `dataflow.yield`.
  - Body may contain `dataflow.{stream, carry, invariant, gate, mux,
    demux, sync, constant, load, store, yield}` plus ordinary
    pure ops permitted in the existing graph body whitelist.
  - Body may not contain `scf.*`, `func.func`, `func.call`,
    `dataflow.thread`, `dataflow.thread.fence`,
    `dataflow.map_info`, any spatial-array op specified in
    `docs/spec-compiler-part-4-spatial.md`, or another
    `dataflow.graph`.

## 11. Testing Strategy

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

## 12. Acceptance Criteria

The first milestone is considered complete when all of the
following hold simultaneously:

* Every `scf.*` operation enumerated under "Scope and Contract"
  has a working
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

## 13. Maintenance and Extension Points

* Adding a new `scf` op: extend "Per-scf Lowering Templates" with a
  template, add a
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

## 14. Non-Goals (First Milestone)

The following are explicitly out of scope for the scf-to-dfg
milestone and have placeholders only:

* Outlining `dataflow.thread` to a `fabric.module` symbol with
  a symbol reference. The thread op stays inline in this milestone,
  but it is already isolated and has an explicit boundary operand
  list.
* Native `dataflow.thread` data results, async value types, thread
  groups, and thread-level aggregation regions. Tensor-result
  aggregation is handled by materializing it into mapped-memory
  effects before thread promotion.
* LLVM IR provider integration, source-language integration, and clang
  embedding. Those concerns belong to Part 1 and Part 2.
* Optimization of `dataflow.map_info` direction. Default `tofrom`.
* Strong-typed `!dataflow.spatial_array`, the symbol-form
  `dataflow.mesh`, and the entire `dataflow.halo_exchange` op
  (signature, verifier, and lowering). All three are listed as
  future work in `docs/spec-compiler-part-4-spatial.md` and are not
  required for this milestone.

## 15. References

* `docs/spec-fabric-module.md`, `docs/spec-fabric-pe.md`,
  `docs/spec-fabric-fu.md` -- the existing fabric-side IR that the
  front-end output eventually targets.
* `docs/spec-compiler-part-1-source.md` -- high-level source
  integration and metadata emission.
* `docs/spec-compiler-part-2-scf.md` -- LLVM-to-SCF raising,
  accelerator-region selection, and `loom.acc_region`.
* `docs/spec-compiler-part-4-spatial.md` -- spatial-array annotation,
  in-thread queries, and halo exchange.
* `docs/spec-dataflow-part-1-streaming.md` -- precise timing
  semantics for `dataflow.stream`, `dataflow.carry`,
  `dataflow.invariant`, and `dataflow.gate`.
* `docs/spec-dataflow-part-2-control.md` -- precise firing semantics
  for `dataflow.constant`, `dataflow.sync`, `dataflow.mux`, and
  `dataflow.demux`.
* Upstream MLIR references (LLVM `externals/llvm/mlir/...`):
  - `Dialect/SCF/IR/SCFOps.td`,
    `Dialect/SCF/IR/DeviceMappingInterface.td`.
  - `Dialect/Async/IR/AsyncOps.td`,
    `Dialect/Async/IR/AsyncTypes.td`.
  - `Dialect/GPU/IR/GPUOps.td`, `Dialect/GPU/IR/GPUBase.td`.
  - `Dialect/OpenMP/IR/OpenMPOps.td`,
    `Dialect/OpenACC/IR/OpenACCOps.td`.
  - `Conversion/SCFToGPU/SCFToGPU.cpp`.
