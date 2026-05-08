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

Implementation engineering -- the pass pipeline that produces this IR
shape, the lit-test layout, the milestone acceptance checklist, and the
maintenance plan -- is documented in
`docs/spec-compiler-part-3-impl.md`. The main body of this document
keeps only the first-principles content: IR boundary contracts, SCF
flattening templates, the memory-dependence model, and verifier
invariants.

Placement decisions across the compiler are described by
`docs/spec-compiler-part-3-placement-framework.md`. Part 3 owns the L2
instance: choosing which code inside a `dataflow.thread` body becomes
`dataflow.graph`. The placement framework does not weaken the verifier
or IR contracts in this document; it only states how legal partitions
are generated, ranked, and made replaceable by later cost-aware
policies.

## 1. Scope and Contract

The compiler front-end is documented in four parts:

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
  `dataflow.graph`. The accompanying
  `docs/spec-compiler-part-3-impl.md` documents the pass pipeline,
  testing, and acceptance for this part. L2 graph placement follows
  the common placement framework in
  `docs/spec-compiler-part-3-placement-framework.md`.
* **Part 4, spatial array.** Annotation and in-thread queries for
  tile-and-mesh memrefs, plus a future-thoughts discussion of
  neighborhood communication / distributed-buffer protocols (see
  `docs/spec-compiler-part-4-spatial.md`).

Input to this part is an MLIR module with `func.func` host containers.
Host code may remain outside accelerator regions. AccCore code must be
inside explicit `loom.acc_region` ops, except in the
`wrap-standalone-kernel` test mode (see
`docs/spec-compiler-part-3-impl.md`). A `func.func` is therefore an
ABI and ownership container, not an implicit accelerator boundary.

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
builder (see `docs/spec-compiler-part-3-mem.md`).
Graph placement inside each thread is governed by the L2 placement
instance specified by the placement framework and by the implementation
contract in `docs/spec-compiler-part-3-impl.md`.

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

The eight rules below are invariants that downstream passes and
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
   already permits feedback edges (existing semantics). Additionally,
   from the parent side: a `dataflow.thread` body must not directly
   contain both a `dataflow.graph` and a nested `dataflow.thread` at
   the same nesting level. This is a separate constraint from the
   leaf rule above (the leaf rule constrains the graph body; this
   parent-side constraint is on what may sit alongside a graph in
   its enclosing thread body). ScalarCore code (`scf.*` structured
   control flow, ScalarCore-legal `func.call`, ScalarCore memory
   ops, `dataflow.thread.fence`, and so on) is always allowed in the
   thread body and may freely sit before, between, or after whichever
   of the two shapes the thread body contains; the parent-side rule
   only forbids the simultaneous presence of a direct `dataflow.graph`
   and a direct nested `dataflow.thread`. Both rules are enforced by
   the verifier (see §9).
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
   Front-end passes that need alias information must obtain it only
   through the `MemAliasOracle` C++ interface; bypassing the interface
   to call upstream MLIR analysis APIs directly is forbidden. Two
   implementations ship in the same library and are interchangeable
   through that interface: a simple SSA-source-of-memref oracle and a
   stronger oracle based on `mlir::AliasAnalysis`. The basic oracle
   is the milestone 1 default and drives the full lit suite; the
   stronger oracle is exercised on a representative differential
   subset that pins oracle-pair equivalence modulo refinement. The
   compositional chain model and the oracle / builder / loop-state /
   wiring details are specified in
   `docs/spec-compiler-part-3-mem.md`.
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
   boundary memory-effect summary from mapped operands; this projection,
   not recursive region effects, is what host code observes for an async
   thread launch. `dataflow.graph`, by contrast, runs synchronously from
   the enclosing ScalarCore program's perspective and uses recursive
   region effects to report what its body touches.
8. Effect visibility contract. Every front-end op whose execution
   affects program order, memory state, or async completion must
   declare its effects through MLIR's `MemoryEffectOpInterface` (or
   an equivalent recursive trait) accurately enough that generic
   optimizers (CSE, LICM, scheduling, code motion) preserve the
   intended observable behavior. The first milestone uses MLIR's
   default-resource barrier pattern -- broad, conservative
   `MemRead + MemWrite` declarations -- where a precise per-resource
   binding would require op-side machinery beyond this milestone's
   scope. Tighter per-resource bindings (for example, load/store
   keyed on the `$mem` operand) are explicit follow-up work.

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
* **Map info result.** A value produced by `dataflow.map_info` that
  carries the same type as its source memref. It is a pure, view-like
  alias of the source; by IR convention it must only be consumed as a
  `dataflow.thread` body operand. Direction and optional bound
  information live as attributes on the producing op, not on the
  result type.
* **`MemAliasOracle`.** The C++ interface (canonical spelling
  matches the C++ class name) returning
  `MustNotAlias` / `MayAlias` / `MustAlias` for any pair of memory
  access ops inside one `dataflow.graph`. It answers conflict only;
  it does not define execution order. Specified in
  `docs/spec-compiler-part-3-mem.md` §3.
* **Memory dependence edge.** A directed edge `p -> o` saying memory
  access `o` must wait for memory access `p` before issuing its
  side effect or externally visible read. Specified in
  `docs/spec-compiler-part-3-mem.md` §4.
* **Loop-carried memory state.** A hidden `none`-typed control state
  carried by a lowered loop for one alias/dependence partition. It
  represents "all memory effects in this partition from previous
  dynamic iterations have retired." Specified in
  `docs/spec-compiler-part-3-mem.md` §5.
* **rwc bit.** A loop-control bit produced by `dataflow.stream` for
  counted loops: it fires `true` once per body iteration and one
  trailing `false` token at the sentinel reset cycle that closes the
  loop. The combined `(true, ..., true, false)` stream phases the
  structural carry and any per-partition memory carry of the loop;
  the false-lane projection is the loop-exit value. The exact
  timing semantics live in `docs/spec-dataflow-part-1-streaming.md`.
* **Streaming token.** Any `none`-typed token consumed or produced
  by the streaming primitives `dataflow.stream`, `dataflow.gate`,
  `dataflow.invariant`, and `dataflow.carry`. Streaming tokens
  carry phase / iteration information rather than memory-state
  information; their precise timing semantics are owned by
  `docs/spec-dataflow-part-1-streaming.md`. The rwc bit above is
  one specific streaming token.
* **Memory-order token.** A `none`-typed token used to encode
  alias-aware ordering between memory accesses inside a
  `dataflow.graph`. Each per-partition frontier (see §2.4 of
  `docs/spec-compiler-part-3-mem.md`) flows through its own
  memory-order tokens; the leaf rendezvous in §6.4 of that
  document combines a structural permission token with a
  memory-order predecessor token at each load / store. Memory-order
  tokens do not encode dynamic execution path (that is the
  structural execution role of §2.1 there).
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

This milestone introduces no other types. The host-to-AccCore data
plane uses `dataflow.map_info` (see §5.4.5), whose result preserves
the source type. The "this value crossed the boundary through
`dataflow.map_info`" provenance is enforced by the verifier on
`dataflow.thread`, not by a wrapper type.

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

* `mapping` is a `DeviceMappingArrayAttr` (an `ArrayAttr` whose
  every entry implements `DeviceMappingAttrInterface`), one per
  grid dim. Mixed `#loom.spatial<...>` and `#loom.temporal<...>`
  in the same array is allowed; the relative order in the array
  equals the relative order of the grid dim.
* `bodyOperands` is the complete explicit set of non-grid values that
  cross into the thread body. The thread is `IsolatedFromAbove`, so
  these operands are the only way the body can refer to surrounding
  SSA values.
* The entry block has one leading `thread_ctrl : none` block argument,
  then one block argument per grid dim (the iteration index, an
  `index`), followed by one block argument per body operand. Each
  body operand and its corresponding entry block argument share the
  same type exactly. A memref-like body operand must be the direct
  SSA result of a `dataflow.map_info` op in the enclosing context;
  the verifier enforces this provenance, and the in-thread block
  argument is the same memref type as the source memref.
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
  interface reports host-visible effects by walking each memref-like
  body operand back through its defining `dataflow.map_info` op and
  reading the `direction` attribute there:
  - `direction = to` reports `Read` on the map source.
  - `direction = from` reports `Write` on the map source.
  - `direction = tofrom` reports both `Read` and `Write` on the map
    source.
  - `direction = alloc` reports `Allocate` on the map source.
  - `direction = release` reports `Free` on the map source.
  Scalar body operands do not contribute memory effects.
* Effects are reported on the `dataflow.map_info` source value, not on
  the `dataflow.map_info` result (which is a view-like alias). The
  source value is then peeled through any recognized view-like ops
  before the effect is projected, using the same view-like list as
  the alias oracle in `docs/spec-compiler-part-3-mem.md` §3.1; in
  particular, `dataflow.spatial_layout` is one such view-like
  producer, so a thread whose `map_info` source is a
  `dataflow.spatial_layout` result reports its effects on the
  underlying `spatial_layout` source memref (per
  `docs/spec-compiler-part-4-spatial.md` §3.1). In nested-thread
  cases that source may itself be a block argument of the enclosing
  thread body; the parent thread's own boundary summary is
  responsible for projecting its effects one level further outward.
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
  MemoryEffects<[MemRead, MemWrite]>.
```

* ScalarCore-side fence and token bridge. It must appear directly in a
  `dataflow.thread` body, outside any `dataflow.graph`.
* **Async dependencies.** Each `none` operand in `$deps` is a
  SpatialCore graph `done_out` token, and each `!dataflow.thread_token`
  operand is a child-thread completion token. SSA def-use already
  enforces "the fence happens after every defining op of `$deps`"; no
  extra effect machinery is needed for that part of the contract.
* **ScalarCore-side memory barrier.** The fence declares
  `MemRead + MemWrite` on MLIR's default memory resource. This is the
  default-resource barrier pattern (broad and conservative, comparable
  to an `MPI_Barrier`-style global barrier). Any ScalarCore op in the
  same thread body that touches memory through specific memrefs (via
  `memref.{load, store}`, `func.call`, etc.) declares effects that the
  default resource subsumes. MLIR therefore must not reorder any such
  op across the fence in either direction. Pure ops with no declared
  effects may still be reordered freely; this is intentional and does
  not change observable behavior.
* **Result token.** The `none` result fires after both the async
  dependencies are satisfied and the ScalarCore-side barrier is
  observed. The result can feed a downstream `dataflow.graph`'s
  `ctrl_in` to express "graph B starts only after graph A and the
  surrounding ScalarCore side effects".
* This op is the only sanctioned bridge between thread completion and
  graph-level control. There is no general cast between
  `!dataflow.thread_token` and `none`. Ordering a child thread after a
  graph completes is expressed by placing the child launch after
  `dataflow.thread.fence(%graph_done)` in ScalarCore program order.

#### 5.4.4 `dataflow.thread.wait`

```
arguments:
  Variadic<Dataflow_ThreadToken>:$asyncDependencies;
results:
  none;
traits:
  DeclareOpInterfaceMethods<LoomAsyncOpInterface>,
  MemoryEffects<[MemRead, MemWrite]>.
```

* Synchronous wait in the enclosing control context: HostCore for an
  outer thread launch, ScalarCore code for a nested thread launch.
  After this op, all listed thread tokens are guaranteed complete.
  Inside a `dataflow.thread`, prefer `dataflow.thread.fence` when the
  wait result must feed a SpatialCore `ctrl_in`.
* The op produces no SSA result, so subsequent host or parent-context
  memory ops cannot be made to depend on it through SSA def-use. To
  preserve "wait for async completion before observing memory" across
  generic MLIR optimizers, the op declares `MemRead + MemWrite` on
  the default memory resource. This is the same default-resource
  barrier pattern used by `dataflow.thread.fence`: it does not mean
  the wait itself touches memory, only that no surrounding memory op
  may be moved across it.

#### 5.4.5 `dataflow.map_info`

```
arguments:
  AnyType:$source,
  Loom_MapDirectionAttr:$direction,
  OptionalAttr<DenseI64ArrayAttr>:$staticBounds,
  Variadic<Index>:$dynamicBounds;
results:
  AnyType:$result;
traits:
  Pure,
  AllTypesMatch<["source", "result"]>.
```

* `source` is a `memref<...>` (or a spatial-array-annotated memref
  in a later milestone).
* `result` has the same type as `source`. The op is a pure,
  view-like alias of its source: alias analysis must treat the
  result as may-alias of the source, and bufferization must treat
  the op as a metadata pass-through. The op exists to attach
  boundary metadata (direction, bounds) and to give the verifier a
  single canonical producer for thread body operands.
* `direction` is the closed enum `to | from | tofrom | alloc |
  release`. The first milestone defaults every front-end-injected
  `map_info` to `tofrom`; an optional optimizer can later refine to
  the narrowest direction.
* `staticBounds` / `dynamicBounds` together describe the per-dim
  half-open `[lo, hi)` ranges that the thread will touch. Empty
  bounds mean "the entire memref"; partial information is
  represented with `ShapedType::kDynamic` sentinels.

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
  - C++ builder API breaking change. Every existing
    `Dataflow_GraphOp::build(...)` overload acquires a leading
    `Value ctrlIn` parameter. Callers in the front-end and any
    downstream user of `OpBuilder` for `dataflow.graph` must pass
    the explicit control-port value at the call site; there is no
    auto-supplied default. The generated `OperationState` builders
    follow the same shape. This is a source-incompatible change to
    the C++ surface and is intentional: the leading `none`-typed
    operand is part of the op contract and must be visible to
    every constructor.
  - The op declares `RecursiveMemoryEffects`. MLIR's default
    implementation walks the graph body and reports the union of all
    inner ops' memory effects through the op boundary. This makes a
    graph that contains side-effecting body ops (notably
    `dataflow.{load, store}`) visible as a memory-touching op to the
    surrounding ScalarCore code, so that standard optimizers do not
    reorder it across `dataflow.thread.fence` or across other
    ScalarCore memory ops. Recursive aggregation is the right model
    for graph because graph runs synchronously from the enclosing
    ScalarCore program's perspective; this is the complement of the
    boundary-projection model that `dataflow.thread` uses for its
    async launch.

* `Dataflow_YieldOp`.
  - The verifier's parent-result-count and parent-result-type checks
    are updated to know about the leading explicit control result.

* `dataflow.load` and `dataflow.store`.
  - The first milestone tightens these existing dataflow primitives
    with explicit memory-effect traits so that
    `dataflow.graph`'s `RecursiveMemoryEffects` correctly aggregates
    body effects:
    - `dataflow.load`  declares `MemoryEffects<[MemRead]>`.
    - `dataflow.store` declares `MemoryEffects<[MemWrite]>`.
  - These use MLIR's default memory resource. They are deliberately
    coarse for the first milestone: any load may-read all memory,
    any store may-write all memory. This is sufficient for graph
    body effects to roll up correctly through `RecursiveMemoryEffects`
    and for surrounding optimizers to keep ScalarCore memory ops
    correctly ordered relative to graphs.
  - Tightening these effects to a per-`$mem`-operand declaration
    (so two loads on disjoint memrefs become reorderable) is
    explicit follow-up work on the dataflow dialect, not part of
    this milestone.

* No other existing op is modified by this milestone.

## 6. Per-scf Lowering Templates

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
memory ordering is added by the compositional chain model in
`docs/spec-compiler-part-3-mem.md` as hidden `none`-typed state; it is
not an optional optimization.

### RWC Phasing Rule

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

### 6.1 `scf.if`

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
# Lane convention: lane 0 = false, lane 1 = true
# demux %cond, %v : (i1, T) -> (T, T) yields (%v_else, %v_then)
# mux %cond, %v_else, %v_then : (i1, T, T) -> T (operand order:
# false-lane first, true-lane second)
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

#### Boundary Translation

This template instantiates the boundary translation contract of
`docs/spec-compiler-part-3-mem.md` §2.8 for `scf.if`.

**Structural plane.** The compound's `struct_in` enters
`demux %cond` at the entry, splitting into a then-lane structural-
permission token and an else-lane structural-permission token. Each
inner region is its own chain scope per
`docs/spec-compiler-part-3-mem.md` §2.2 and uses its lane's token as
its `S.struct_at_*` source per
`docs/spec-compiler-part-3-mem.md` §6.2. The compound's `struct_done`
is `mux %cond` over the two branches' `struct_done` tokens, following
the §6 selector convention (lane 0 = false-lane = else, lane 1 =
true-lane = then). The same `mux` shape is reused for any data
result of the `scf.if`.

**Memory plane (per touched partition `P`).** The compound's
`incoming_C_P` enters a `demux %cond` at the entry, projecting it
into a then-lane `then_in_P` and an else-lane `else_in_P` token.
Only the active lane's projection fires, matching the dual-plane
contract of `docs/spec-compiler-part-3-mem.md` §2.8 (a raw SSA
fork would risk stale memory tokens being buffered in the
unselected branch and consumed on a later selected invocation).
Each branch chain scope's `incoming_P` is its lane's projected
token. Each branch's per-`P` tail is path-forwarding per
`docs/spec-compiler-part-3-mem.md` §2.7: a branch that performs no
access in `P` forwards its lane projection unchanged; a branch
that performs accesses in `P` builds its tail by the single-level
chain rule of `docs/spec-compiler-part-3-mem.md` §2.5 inside that
branch. Call those `then_tail_P` and `else_tail_P`. The compound's
`outgoing_C_P` is the selector-matched `mux %cond` of the two
tails, following the §6 selector convention (lane 0 =
`else_tail_P`, lane 1 = `then_tail_P`). Per leaf rendezvous,
`docs/spec-compiler-part-3-mem.md` §6.4 still applies inside each
branch: a leaf at branch scope `S_branch` uses
`L.ctrl = dataflow.sync(S_branch.struct_at_L, incoming_L_P)`.

No loop-carried state. `scf.if` does not introduce a `dataflow.carry`
either on the structural plane or on any per-`P` plane.

### 6.2 `scf.while` with `scf.condition`

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
# Lane convention: lane 0 = false, lane 1 = true. demux yields
# (false-lane, true-lane); mux operand order is (false-lane,
# true-lane).
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

#### Boundary Translation

This template instantiates the boundary translation contract of
`docs/spec-compiler-part-3-mem.md` §2.8 for `scf.while`.

**Structural plane.** The compound's `struct_in` initializes the
structural carry `%iter_ctrl = carry %cond, %entry_ctrl,
%ctrl_feedback`. The before-region is its own chain scope and uses
`%iter_ctrl` as its `S.struct_at_*` source. The after-region is a
separate chain scope; its structural-permission source is
`%after_ctrl` from `gate %cond, %before_done` per the existing
template. The compound's `struct_done` is the false-cycle exit
projection of the carry, equivalently the false-lane of `demux %cond,
%before_done` reused at the boundary. The before-region executes
`K + 1` times for `K` after-region executions, matching the existing
template.

**Memory plane (per touched partition `P`).** The compound applies
the loop-carried memory state pattern of
`docs/spec-compiler-part-3-mem.md` §5.2 with `selector = %cond` and
the before-region / after-region instantiation of
`docs/spec-compiler-part-3-mem.md` §5.4. For every `P` carried by the
loop (per `docs/spec-compiler-part-3-mem.md` §4.3,
`P ∈ Π_L` iff some access in one iteration may conflict with some
access in a later iteration in `P`), the lowering introduces a hidden
per-iteration ring:

```
%mem_iter_P = carry %cond, %mem_init_P, %mem_feedback_P : none
```

* `%mem_init_P` is the compound's `incoming_C_P` per
  `docs/spec-compiler-part-3-mem.md` §2.4, drawn from the enclosing
  scope's per-`P` frontier at the `scf.while`'s position.
* `%mem_iter_P` enters the before-region as its `incoming_P` for
  `P`. The before-region's per-`P` tail `%before_tail_P` is built by
  the single-level chain rule of
  `docs/spec-compiler-part-3-mem.md` §2.5 inside the before-region;
  it forwards `%mem_iter_P` unchanged when the before-region performs
  no access in `P`.
* The after-region's `incoming_P` is `%after_in_P = gate %cond,
  %before_tail_P`, so only true-cycle iterations expose a
  `incoming_P` to the after-region. The after-region's per-`P` tail
  `%after_tail_P` is path-forwarding for the same reason
  (`%after_tail_P = %after_in_P` when the after-region performs no
  access in `P`).
* The feedback that closes the ring is `%mem_feedback_P = mux %cond,
  %before_tail_P, %after_tail_P` following the §6 selector convention
  (lane 0 = false-lane = `%before_tail_P`, lane 1 = true-lane =
  `%after_tail_P`). On a true iteration the after-region tail is
  carried; on the final false iteration the before-region tail is
  carried, because the final false iteration's before-region still
  ran.
* Loop-exit projection: the compound's `outgoing_C_P = %mem_after_P`,
  taken from the false-lane projection of `%before_tail_P` on the
  final false iteration (equivalently the false-lane of `demux %cond,
  %before_tail_P`). The zero-trip case (`%cond` false on the first
  check) reduces to the same projection over the single before-region
  run. This matches `docs/spec-compiler-part-3-mem.md` §5.4 verbatim
  and preserves any memory effect performed by the final
  condition-checking iteration.

The structural `%after_rwc` from the existing template is on the
structural plane only and is not on the memory critical path. Per
`docs/spec-compiler-part-3-mem.md` §2.5 plane orthogonality and
`docs/spec-compiler-part-3-mem.md` §5.4, after-region memory ops use
`L.ctrl = dataflow.sync(struct_after, %after_in_P)` per
`docs/spec-compiler-part-3-mem.md` §6.4; the structural token grants
phase permission while `%after_in_P` carries the alias-aware
ordering. Independent partitions in `Π_L` get independent rings
sharing only the `%cond` selector, so unrelated memrefs are not
serialized.

For a partition `P` touched somewhere in the before-region or the
after-region but not in `Π_L`, no state ring is created. The
compound's `incoming_C_P` flows into the before-region as its
`incoming_P`; the before-region's per-iteration body-tail in `P`,
plus (on the true path) the after-region's body-tail in `P`, are
gathered through the compound's structural-selector-driven
rendezvous (per `docs/spec-compiler-part-3-mem.md` §5.2) into the
compound's `outgoing_C_P`. No cross-iteration ordering is
introduced; the rendezvous only signals that every executed
body access in `P` has retired. A partition not touched anywhere
in the compound is absent from its interface, per §2.4.

#### K=2 Worked Trace

Consider a small `scf.while` whose before-region and after-region
each contain one store to the same memref:

```mlir
scf.while (%i = %c0) : (index) -> index {
  %cond = arith.cmplt %i, %c2 : index
  memref.store %v, %A[%i] : memref<10xf32>     // before-region, P
  scf.condition(%cond) %i : index
} do {
^bb0(%i: index):
  memref.store %w, %A[%i] : memref<10xf32>     // after-region, P
  %j = arith.addi %i, %c1 : index
  scf.yield %j : index
}
```

Let `P` be the alias bucket of `%A`. For `K = 2` (`%cond = [T, T,
F]`), the before-region executes three times and the after-region
executes twice. The chain through `P` traverses 7 named tokens:

```
mem_init_P
  -> before_tail_P_0
  -> after_tail_P_0
  -> before_tail_P_1
  -> after_tail_P_1
  -> before_tail_P_2
  -> mem_after_P
```

Per-iteration values:

```
iter 0 (cond_0 = true):
  mem_iter_P_0    = carry %cond, mem_init_P, mem_feedback_P
                  -> mem_init_P                     // first activation
  before_tail_P_0 = store A[0]
                    ctrl = sync(struct_before_0, mem_iter_P_0)
  after_in_P_0    = gate %cond_0, before_tail_P_0   // fires on true
  after_tail_P_0  = store A[0]
                    ctrl = sync(struct_after_0, after_in_P_0)
  mem_feedback_P_0 = mux %cond_0,
                       before_tail_P_0,             // false lane
                       after_tail_P_0               // true lane
                   = after_tail_P_0

iter 1 (cond_1 = true):
  mem_iter_P_1    = carry feedback -> mem_feedback_P_0
                  = after_tail_P_0
  before_tail_P_1 = store A[1]
                    ctrl = sync(struct_before_1, mem_iter_P_1)
  after_in_P_1    = gate %cond_1, before_tail_P_1
  after_tail_P_1  = store A[1]
                    ctrl = sync(struct_after_1, after_in_P_1)
  mem_feedback_P_1 = mux %cond_1,
                       before_tail_P_1, after_tail_P_1
                   = after_tail_P_1

iter 2 (cond_2 = false; final false before, no after):
  mem_iter_P_2    = carry feedback -> mem_feedback_P_1
                  = after_tail_P_1
  before_tail_P_2 = store A[2]
                    ctrl = sync(struct_before_2, mem_iter_P_2)
  // gate %cond_2 = false: after_in_P_2 not produced; the after
  // region does not fire this iteration.
  // false-lane projection of before_tail_P_2 leaves the loop:
  mem_after_P     = false-lane(%cond_2, before_tail_P_2)
                  = before_tail_P_2
```

Two observations close the trace. First, the final false before
execution is memory-visible: its `before_tail_P_2` becomes the
loop-exit memory state for `P`, exactly as
`docs/spec-compiler-part-3-mem.md` §5.4 specifies for the
final-false-iteration before-tail projection. Second, `%after_rwc`
is not on the same-execution memory critical path: after-region
memory ops use `sync(struct_after, after_in_P)` for `ctrl` per
`docs/spec-compiler-part-3-mem.md` §2.5 plane orthogonality, while
`%after_rwc` only advances or resets after-region structural state
for subsequent iterations. If `P` were independent of some other
partition `Q`, the entire trace runs in parallel for `Q` with its
own `mem_iter_Q` / `mem_feedback_Q` / `mem_after_Q`; no
cross-partition serialization is introduced through any single
whole-while done token.

### 6.3 `scf.for` with `scf.yield`

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
are wired by the compositional chain model in
`docs/spec-compiler-part-3-mem.md`; `%body_ctrl` above is only the
canonical structured-control source when such a token is needed.
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

#### Boundary Translation

This template instantiates the boundary translation contract of
`docs/spec-compiler-part-3-mem.md` §2.8 for `scf.for`.

**Structural plane.** `dataflow.stream` produces the loop-level
rwc, which doubles as the structural selector. The structural
plane diverges by case to match the existing data-value templates
above:

* No-iter-arg case (per "No Iter Args" template above). The
  compound's `struct_in` is consumed by a `dataflow.invariant`
  to gate the body-phase ctrl, and the `demux %loop_rwc, %ctrl_raw`
  splits out `%loop_exit_ctrl` on the sentinel cycle. There is no
  data state ring; the body-phase structural-permission token is
  the true-lane projection of the rwc-driven ctrl, and the
  compound's `struct_done` is `%loop_exit_ctrl`.
* With-iter-arg case (per "With Iter Args" template above). The
  compound's `struct_in` seeds a `dataflow.carry` driven by
  `%loop_rwc`, and each iter_arg gets its own `carry` ring on the
  same selector. The body region is a single chain scope and
  uses the body-phase structural-permission token derived from
  `%loop_rwc` as its `S.struct_at_*` source per
  `docs/spec-compiler-part-3-mem.md` §6.2. The compound's
  `struct_done` is the false-lane projection of the structural
  carry on the sentinel cycle, equivalently the `%loop_exit_ctrl`
  output of the existing template.

In both cases the body region is a single chain scope under
`docs/spec-compiler-part-3-mem.md` §2.2, and `dataflow.invariant`
or `dataflow.carry` is the only structural-control primitive
introduced at the boundary; no `dataflow.demux` / `dataflow.mux`
sits at the boundary's structural plane outside the rwc-driven
sentinel reset.

**Memory plane (per touched partition `P`).** The compound applies
the loop-carried memory state pattern of
`docs/spec-compiler-part-3-mem.md` §5.2 with `selector = %loop_rwc`,
specialized to `scf.for` per `docs/spec-compiler-part-3-mem.md` §5.3.
For every `P` carried by the loop (per
`docs/spec-compiler-part-3-mem.md` §4.3,
`P ∈ Π_L` iff some access in one iteration may conflict with some
access in a later iteration in `P`), the lowering introduces:

```
%mem_iter_P = carry %loop_rwc, %mem_init_P, %mem_next_P : none
```

* `%mem_init_P` is the compound's `incoming_C_P`, drawn from the
  enclosing scope's per-`P` frontier at the `scf.for`'s position.
* `%mem_iter_P` is gated by `%loop_rwc` exactly like iter_args:
  the true-lane projection enters the body as its `incoming_P`
  for `P`, and the false-lane projection becomes
  `%mem_after_P` for the enclosing scope. Body-region accesses
  chain through the true-lane projection per
  `docs/spec-compiler-part-3-mem.md` §2.5; they never observe
  the sentinel-cycle (rwc=false) value.
* `%mem_next_P` feeds the carry on the rwc=true lane and is built
  from the body's per-`P` tail per
  `docs/spec-compiler-part-3-mem.md` §2.5 / §2.7 (a body path that
  performs no access in `P` forwards `%mem_iter_P` unchanged;
  same-path required tails join via `dataflow.sync`; mutually
  exclusive tails join via selector-matched `dataflow.mux`).
* Loop-exit projection: the compound's `outgoing_C_P = %mem_after_P`,
  taken from the false-lane projection of the carried state on the
  sentinel cycle (same false-lane shape as `%acc_exit` in the
  with-iter-arg case). The zero-trip case (rwc=false on the first
  cycle) gives `%mem_after_P = %mem_init_P` directly, because the
  carry produces its initializer on the first activation and the
  false-lane projects that initializer out unchanged.

The body has no after-region; `scf.for` has a single body chain
scope. Independent partitions in `Π_L` get independent rings sharing
only the `%loop_rwc` selector, so unrelated memrefs are not
serialized. Per `docs/spec-compiler-part-3-mem.md` §2.5 plane
orthogonality, the structural rwc carry and the per-`P` memory carry
are independent state rings over the same selector; the structural
plane never aggregates the memory tails.

For a partition `P` that is touched somewhere in the body but
not in `Π_L` (typically a read-only partition), no state ring is
created. The compound's `incoming_C_P` flows into the body as
its per-iteration `incoming_P`, and the compound's `outgoing_C_P`
is the streamed rendezvous of every executed iteration's body
tail in `P`, per `docs/spec-compiler-part-3-mem.md` §5.2. No
cross-iteration ordering is introduced; the rendezvous only
signals that every body access in `P` has retired before the
loop's `outgoing_C_P` fires. A partition not touched anywhere in
the body does not appear at the compound's interface and the
enclosing scope's frontier flows past unchanged (§2.4).

### 6.4 `scf.forall` with `scf.forall.in_parallel`

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
After aggregation materialization, every `scf.forall` that Part 3
continues to lower must be in effect form.

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
  destination buffer crosses a mapped forall boundary, the
  boundary-promotion step that inserts `dataflow.map_info` treats it
  like any other memref-like value.
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

By the time this normalization runs, every Loom-recognized mapped
forall has already been promoted to `dataflow.thread`. Therefore, a
forall that reaches this path must have an empty mapping attribute.
A non-empty non-Loom mapping is rejected before normalization.

If a forall has a non-empty `mapping` attribute that Part 3 does not
recognize as a Loom mapping, the pipeline must not silently ignore it
inside an accelerator region. Part 2 or an earlier Part 3 pass must
either remove or translate that mapping with an explicit downgrade
decision, or emit a diagnostic before this template runs.

#### Boundary Translation

This template instantiates the boundary translation contract of
`docs/spec-compiler-part-3-mem.md` §2.8 for `scf.forall`.

A mapped `scf.forall` is promoted to a `dataflow.thread` by
`loom-build-thread-skeleton` (per
`docs/spec-compiler-part-3-impl.md` §1.5) before the
`dataflow.graph` chain model ever runs over it. Mapped foralls
therefore never appear as compound atoms inside a chain scope;
their launch and completion are governed by the
`!dataflow.thread_token` async protocol, which is explicitly
out of scope for the chain model per
`docs/spec-compiler-part-3-mem.md` §2.9.

An empty-mapping `scf.forall` does reach a chain scope. The pass
`loom-build-memory-dependencies` (per
`docs/spec-compiler-part-3-impl.md` §1.8) normalizes such a
forall to `scf.parallel` and from there to one or more `scf.for`
loop nests with parallel-provenance metadata. The compound that
stands for the original forall in the chain is therefore the
parallel-provenance compound described in §6.5 below.

### 6.5 `scf.parallel` with `scf.reduce`

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
implementation. The N-Dim Parallel With M Reductions subsection
below assumes the per-dim chunk count `K_d` may differ across dims
in a future implementation while still using the global factor in
milestone 1; the carry-placement and merge contract specified
there is independent of the K choice, so a future cost-model-driven
per-dim K can land without changing the IR contract.

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

#### N-Dim Parallel With M Reductions

The single-dimensional, multi-result discussion above does not by
itself pin the IR shape for the multi-dimensional case. This
subsection extends the partial-and-merge scheme to a `scf.parallel`
over `N` parallel dimensions with `M` reduction results.

**Generated loop-nest layout.** After parallel-SCF normalization (per
`docs/spec-compiler-part-3-impl.md` §1.8), an `N`-dim `scf.parallel`
becomes one or more `scf.for` loop nests that share a single
parallel-provenance group, plus any required reduction-merge
`scf.if` ops. Each parallel dim becomes one `scf.for`. The loop-nest
order is outermost-first-in-source-order: the outermost generated
`scf.for` corresponds to the leftmost parallel dim of the source
`scf.parallel`, and the innermost generated `scf.for` corresponds to
the rightmost parallel dim, which is therefore the most tightly
bound. The implementation may choose, per dim, how many chunks `K`
to split that dim into; that choice is implementation-defined and
does not affect the carry placement contract specified below. When a
dim is split into `K > 1` chunks, the chunked dim is materialized as
two nested `scf.for` ops at that dim's position in the nest -- an
outer K-chunk loop iterating over the chunk ordinal `0 .. K-1`, and
an inner per-chunk loop iterating the dim's subrange. The K-chunk
loop is the carrier of the cross-chunk reduction merge; the inner
per-chunk loop participates in the same intra-chunk body as any
unchunked dim.

**Reduction iter_arg placement.** For `M` reductions over the same
`N` parallel dims, each reduction has one (valid, partial) tuple per
chunk-tuple. Each reduction's `%iter_arg` is hung on the **innermost**
generated per-chunk `scf.for` -- the loop with the smallest stride
and the tightest binding -- so that the partial accumulates across
all iterations of all parallel dims within one chunk-tuple. The
reduction's `%iter_init` is the chunk-empty seed described under
"valid flag wiring" below; it flows in from the loop-nest scope that
encloses the innermost per-chunk loop. The reduction's `%iter_yield`
is the partial after one iteration completes inside the chunk-tuple,
and the reduction's `%iter_final` is the partial after the
innermost per-chunk loop completes -- this is the per-chunk-tuple
partial value that the outer reduction-merge `scf.if` consumes.
Other parallel dims' per-chunk `scf.for` loops do not carry the
reduction iter_arg directly; the partial is hoisted out of the
innermost per-chunk loop as a normal `scf.for` result. When the
intra-chunk iteration space is represented as a nest of multiple
per-chunk `scf.for` loops (one per parallel dim), each reduction's
`(valid, partial)` tuple must be threaded as iter_args through every
enclosing per-chunk loop in the nest, not only the innermost one.
The seed at the outer per-chunk loop's iter_arg is the dummy seed
`(false, init_r)`; intermediate per-chunk loops yield their inner
loop's final tuple; the outermost per-chunk loop yields the
chunk-tuple's `(valid, partial_r)` pair. Without this pass-through
each outer-loop iteration would restart the inner reduction from
the seed and lose the accumulated partial. An equivalent canonical
representation flattens the intra-chunk N-D iteration space into a
single linearized `scf.for`, in which case each reduction has a
single `iter_arg` on that one loop. Implementations may choose
either canonical form.

**Valid flag wiring.** The per-chunk `valid` flag is shared by all
`M` reductions of the same chunk-tuple. It is computed inside the
innermost per-chunk loop the same way as in the one-dim case:
`%valid` starts false at the chunk-tuple entry and is set true on
the first executed iteration. The `M` reduction `%iter_init` values
are dummy seeds; their concrete value is irrelevant because the
outer merge `scf.if` only folds a chunk-tuple's partial when its
`%valid` is true, which guarantees that at least one body iteration
overwrote the dummy seed before the partial was produced.
Implementations may pick any deterministic seed (for example, the
original `init` value, or an undef poison value); the seed choice
does not affect the merged result.

For arbitrary `scf.reduce` bodies that lack a usable identity (for
example, a non-commutative or otherwise no-identity reduction), the
overwrite is not implicit: the innermost body must branch on the
chunk-local `%valid` so that the first executed iteration yields
the iteration value as the partial directly, and subsequent
iterations inline the source `scf.reduce` body with the running
partial and the iteration value. The worked example below uses
sum and max where the natural identity (`0` and `-inf`) makes the
branch unnecessary; the generic non-identity scheme is what
arbitrary reductions lower to.

**K > 1 multi-chunk tuple nesting.** When the implementation splits
one or more parallel dims with `K_d > 1`, the chunk-tuples are
enumerated by a K-chunk `scf.for` nest -- one `scf.for` per chunked
dim -- placed outside the per-chunk loop nest. The chunk-tuples
form a flat sequence indexed by a deterministic chunk-tuple ordinal
that respects source dim order: the outermost K-chunk loop varies
slowest. For each reduction `r` in `0 .. M-1`, the K-chunk nest
carries a running accumulator `%acc_r` as an iter_arg on every
K-chunk `scf.for` in the nest, threaded through with `scf.yield` so
that the value reaching the merge `scf.if` is the accumulator after
the prior chunk-tuple in canonical order. The accumulator is
**seeded** by the source `init_r` on the outermost K-chunk
`scf.for`'s `iter_args`; each inner K-chunk loop's iter_arg is
seeded from the enclosing K-chunk loop's iter_arg block argument,
not from `init_r` directly. Inside the innermost K-chunk loop, a
reduction-merge `scf.if` consumes:

* the per-chunk-tuple partial `%partial_r` produced by reduction
  `r`'s innermost per-chunk `iter_arg`'s `%iter_final`;
* the running accumulator `%acc_r` carried in from the prior
  chunk-tuple via the K-chunk nest's iter_args;
* the per-chunk-tuple `%valid` bit, which determines whether the
  merge fires for this chunk-tuple.

For `M` reductions, `M` independent (valid, partial, accumulator)
triples nest in the same K-chunk loop nest as `M` parallel
iter_args on every K-chunk `scf.for`. The reduction-merge `scf.if`
ops may be expressed as one multi-result `scf.if` that yields the
next `M`-tuple `(%acc_0', .., %acc_{M-1}')`, or as `M` single-result
`scf.if` ops sharing the same `%valid` selector; both shapes are
equivalent under the §6.1 template. Whichever shape is chosen, the
merged tuple is yielded back into the K-chunk nest's iter_args,
and the final values flowing out of the outermost K-chunk
`scf.for` are the `M` `scf.parallel` results.

When every parallel dim has `K_d = 1`, there is no K-chunk loop at
all, and the merge `scf.if` collapses to a single fold equivalent
to the one-dim `K = 1` case. The `M` reduction `(valid, partial)`
tuples still pass through every enclosing per-chunk loop in the
intra-chunk nest as iter_args, with the dummy seeds at the
outermost per-chunk loop and the final tuple yielded out of the
outermost per-chunk loop into the merge `scf.if`; this is the same
pass-through-through-all-dims rule as the multi-chunk case, just
without the outer K-chunk wrapper. The reduction iter_args still
do their actual update at the innermost per-chunk loop's body, but
they are not hung directly on that innermost loop alone. An
implementation that flattens the intra-chunk N-D iteration space
into a single linearized `scf.for` is the equivalent canonical
form and may carry each reduction as a single iter_arg on that one
loop. When a chunked dim has `K_d = 1` it is omitted from the
K-chunk nest entirely.

**Worked example: 2D parallel, 2 reductions.** Consider:

```mlir
%sum, %max = scf.parallel (%i, %j) = (%c0, %c0) to (%I, %J)
    step (%c1, %c1) init (%zero, %neginf) -> (f32, f32) {
  %x = memref.load %A[%i, %j] : memref<?x?xf32>
  scf.reduce(%x, %x : f32, f32) {
  ^bb0(%lhs0 : f32, %rhs0 : f32):
    %s = arith.addf %lhs0, %rhs0 : f32
    scf.reduce.return %s : f32
  }, {
  ^bb0(%lhs1 : f32, %rhs1 : f32):
    %m = arith.maximumf %lhs1, %rhs1 : f32
    scf.reduce.return %m : f32
  }
}
```

For the implementation choice `K_i = K_j = 2` (two chunks per dim,
four chunk-tuples total), the normalized loop nest has the shape:

```mlir
%sum_final, %max_final =
  scf.for %ki = %c0 to %c2 step %c1
      iter_args(%sum_acc = %zero, %max_acc = %neginf) -> (f32, f32) {
    %sum_acc1, %max_acc1 =
      scf.for %kj = %c0 to %c2 step %c1
          iter_args(%sum_acc_j = %sum_acc, %max_acc_j = %max_acc)
          -> (f32, f32) {
        %i_lb, %i_ub = chunk_bounds(%ki, %c0, %I, %c1, %c2)
        %j_lb, %j_ub = chunk_bounds(%kj, %c0, %J, %c1, %c2)

        %valid_chunk, %sum_partial_chunk, %max_partial_chunk =
          scf.for %i = %i_lb to %i_ub step %c1
              iter_args(%v_outer = %false,
                        %s_outer = %zero,
                        %m_outer = %neginf) -> (i1, f32, f32) {
            %v_inner, %s_inner, %m_inner =
              scf.for %j = %j_lb to %j_ub step %c1
                  iter_args(%v = %v_outer,
                            %s = %s_outer,
                            %m = %m_outer) -> (i1, f32, f32) {
                %x = memref.load %A[%i, %j] : memref<?x?xf32>
                %s_next = arith.addf %s, %x : f32
                %m_next = arith.maximumf %m, %x : f32
                scf.yield %true, %s_next, %m_next : i1, f32, f32
              }
            scf.yield %v_inner, %s_inner, %m_inner : i1, f32, f32
          }

        %sum_acc_j_next = scf.if %valid_chunk -> f32 {
          %merged = arith.addf %sum_acc_j, %sum_partial_chunk : f32
          scf.yield %merged : f32
        } else {
          scf.yield %sum_acc_j : f32
        }
        %max_acc_j_next = scf.if %valid_chunk -> f32 {
          %merged = arith.maximumf %max_acc_j, %max_partial_chunk : f32
          scf.yield %merged : f32
        } else {
          scf.yield %max_acc_j : f32
        }

        scf.yield %sum_acc_j_next, %max_acc_j_next : f32, f32
      }
    scf.yield %sum_acc1, %max_acc1 : f32, f32
  }
```

The intra-chunk-tuple body sits inside the innermost per-chunk
`scf.for` over `%j`, which carries the three iter_args `(%v, %s,
%m)`. The first executed iteration of the chunk-tuple flips `%v`
to `true` and overwrites the seed values of `%s` and `%m`, so the
chunk-tuple's partial is well-defined whenever `%valid_chunk` is
true. The two reduction-merge `scf.if` ops sit inside the inner
K-chunk loop over `%kj` and fold each reduction independently
under the shared `%valid_chunk` selector. Both running
accumulators `%sum_acc` and `%max_acc` are threaded as iter_args
through both K-chunk loops -- seeded by `%zero` and `%neginf` on
the outer K-chunk loop over `%ki`, threaded through the inner
K-chunk loop over `%kj` via its own iter_args, and yielded back
through both K-chunk loops -- so that the value entering the merge
`scf.if` on chunk-tuple `(ki, kj)` is the accumulator after the
prior chunk-tuple in canonical order. The outermost K-chunk loop's
results `%sum_final` and `%max_final` are the `scf.parallel`'s two
results.

The K choice (chunks per parallel dim) is implementation-defined;
this section pins the carry placement and merge structure
regardless of K, so a future implementation may pick K based on
cost-model decisions without changing the IR contract. In
particular, switching any dim from `K_d = 1` to `K_d > 1` only
adds one K-chunk `scf.for` for that dim into the K-chunk nest and
extends the running accumulators' iter_arg threading through it;
the per-chunk-tuple body and the per-reduction `%iter_arg`
placement on the innermost per-chunk loop are unchanged.

After normalization, all generated `scf.for` and `scf.if` operations
use the existing templates in this section. Their stream, carry, gate,
demux, mux, and memory-order behavior is inherited from those templates.

#### Boundary Translation

This template instantiates the boundary translation contract of
`docs/spec-compiler-part-3-mem.md` §2.8 for `scf.parallel`.

After parallel-SCF normalization (per
`docs/spec-compiler-part-3-impl.md` §1.8), `scf.parallel` becomes one
or more `scf.for` loop nests with parallel-provenance attributes. The
outer compound that "stands for" the original `scf.parallel` in the
chain model is the parallel-provenance compound: it is the analysis-
visible group of generated chunk loops sharing one
`loom.parallel_group` id, not a new IR op. Each chunk loop body is
its own chain scope per `docs/spec-compiler-part-3-mem.md` §2.2.

**Structural plane.** The compound's `struct_in` forks: every chunk
receives the same SSA value as its structural-permission input
(shared `struct_in` across chunks per the §2.8 table for
`scf.forall` / `scf.parallel`). Each chunk's `struct_done` is the
`scf.for` template's `struct_done` for that chunk. The compound's
`struct_done` is `dataflow.sync` over all chunk `struct_done` tokens,
matching the rendezvous in `docs/spec-compiler-part-3-mem.md` §2.6
for parallel-provenance compound atoms.

**Memory plane (per touched partition `P`).** All chunks share the
compound's `incoming_C_P`: the same SSA value forks into each chunk
loop's per-iteration `incoming_P` (§5.6 of
`docs/spec-compiler-part-3-mem.md` applies recursively if a
parallel group is nested inside a source-ordered loop). Each
chunk's per-`P` tail `%chunk_tail_P` is independent and is built
under the parallel-provenance override: the chunk loop applies
§6.3's structural plane (stream + carry on rwc + sentinel reset)
without building a per-`P` loop-carried state ring, since its
iterations remain logical iterations of the original
`scf.parallel`. The chunk's body memory accesses still chain
through their partition's frontier within a single iteration, and
each chunk's rendezvous of completed per-iteration tails feeds its
`%chunk_tail_P`. The compound's `outgoing_C_P = dataflow.sync`
over all `%chunk_tail_P` tokens, per
`docs/spec-compiler-part-3-mem.md` §2.6 chunk-tail rendezvous and
the parallel-provenance exception of
`docs/spec-compiler-part-3-mem.md` §4.3 and §5.6.

No loop-carried memory state is created at the parallel-provenance
compound boundary, per the parallel-provenance exception of
`docs/spec-compiler-part-3-mem.md` §4.3 and the no-state-ring rule
of `docs/spec-compiler-part-3-mem.md` §5.6: cross-iteration and
cross-chunk dependence edges inside the compound are suppressed by
the dependence builder, so the compound never builds a per-`P` ring.
Each generated chunk loop carries its own parallel-provenance
metadata, since its iterations are still logical iterations of the
original `scf.parallel`; per
`docs/spec-compiler-part-3-mem.md` §4.3 / §5.6 it therefore does
not build a per-`P` loop-carried state ring across its own
iterations. The §6.3 boundary translation supplies only the
chunk loop's structural plane (stream-driven rwc, sentinel reset,
iter_args for non-memory loop state); the chunk loop's memory
plane reduces to "no cross-iteration memory ordering inside this
loop". Memory accesses inside the chunk loop's body still chain
through their partition's frontier within a single iteration and
participate in the compound's `outgoing_C_P` rendezvous via the
chunk-tail token described above. The compound atom is marked
with parallel-provenance metadata
(`loom.parallel_group`, `loom.parallel_chunk`, `loom.parallel_chunks`)
per `docs/spec-compiler-part-3-mem.md` §4.3 so the chain construction
identifies it correctly.

### 6.6 `scf.index_switch`

`scf.index_switch` has the same selected-region shape as `scf.if`, but
its source selector is an arbitrary `index` value matched against a
dense array of case constants. `dataflow.mux` and `dataflow.demux`
require dense lane selectors, so lowering first normalizes the source
argument to a dataflow lane id.

Lane convention is a normalized lowering convention (it is not the
print order of the source op, which lists case regions before the
default region in the MLIR `scf.index_switch` op):

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
# Lane convention: lane 0 = default region, lane i+1 = case region i
# (this is the lowering's normalized lane order; the source op prints
# case regions before the default region in MLIR's scf.index_switch).
# demux yields default-lane first, then case 0, case 1, ...; mux
# operand order matches.
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
* The zero-case form is spliced during scf body lowering, before the
  surrounding graph body is finalized. Memory-dependence snapshots
  continue to identify memory ops by their existing deterministic
  ids; the splice does not create a selector-dependent memory path.

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

#### Boundary Translation

This template instantiates the boundary translation contract of
`docs/spec-compiler-part-3-mem.md` §2.8 for `scf.index_switch`.

**Structural plane.** The compound's `struct_in` enters an `(N + 1)`
way `dataflow.demux` keyed on the normalized lane id `%lane` per the
existing template (lane 0 = default region, lane `i + 1` = case
region `i`). Each selected region is its own chain scope per
`docs/spec-compiler-part-3-mem.md` §2.2 and uses its lane's structural-
permission token as its `S.struct_at_*` source per
`docs/spec-compiler-part-3-mem.md` §6.2. The compound's `struct_done`
is `dataflow.mux` over all `(N + 1)` regions' `struct_done` tokens,
keyed on the same `%lane`. The same `(N + 1)` way `mux` shape
applies to every data result of the `scf.index_switch`.

**Memory plane (per touched partition `P`).** The compound's
`incoming_C_P` enters an `(N + 1)` way `dataflow.demux` keyed on
the same normalized `%lane`, projecting it into per-region tokens
`default_in_P`, `case0_in_P`, ..., `caseN_in_P`. Only the selected
region's projection fires, matching the dual-plane contract of
`docs/spec-compiler-part-3-mem.md` §2.8 (a raw SSA fork would risk
stale memory tokens being buffered in unselected regions and
consumed on a later selected invocation). Each region chain scope's
`incoming_P` is its lane's projected token. Each region's per-`P`
tail is path-forwarding per
`docs/spec-compiler-part-3-mem.md` §2.7: a region that performs no
access in `P` forwards its lane projection unchanged; a region
that performs accesses in `P` builds its tail by the single-level
chain rule of `docs/spec-compiler-part-3-mem.md` §2.5 inside that
region. Call those `default_tail_P`, `case0_tail_P`, ...,
`caseN_tail_P`. The compound's `outgoing_C_P` is the
selector-matched `(N + 1)` way `dataflow.mux %lane` of these tails
(lane 0 = `default_tail_P`, lane `i + 1` = `case_i_tail_P`). Per
leaf rendezvous, `docs/spec-compiler-part-3-mem.md` §6.4 still
applies inside each region.

No loop-carried state. `scf.index_switch` does not introduce a
`dataflow.carry` either on the structural plane or on any per-`P`
plane.

### 6.7 `scf.execute_region`

* No control structure to flatten. The pass inlines the region body
  into the surrounding scope and rewires SSA values; ctrl/done
  forwarding follows program order.

#### Boundary Translation

This template instantiates the boundary translation contract of
`docs/spec-compiler-part-3-mem.md` §2.8 for `scf.execute_region`.

**Structural plane.** Pass-through. `scf.execute_region` has a single
inner region with no control selector. The inner region's chain
scope inherits the compound's `struct_in` directly as its
`S.struct_at_*` source per `docs/spec-compiler-part-3-mem.md` §6.2,
and its `struct_done` directly becomes the compound's `struct_done`.
No `dataflow.demux` / `dataflow.mux` / `dataflow.carry` /
`dataflow.gate` is introduced by the boundary translation.

**Memory plane (per touched partition `P`).** Pass-through.
`incoming_C_P` directly enters the inner region as its `incoming_P`
per `docs/spec-compiler-part-3-mem.md` §2.4; the inner region's
`outgoing_P`, computed by the single-level chain rule of
`docs/spec-compiler-part-3-mem.md` §2.5 inside the region, directly
becomes the compound's `outgoing_C_P`. No loop-carried state.

If the inlining pass described above runs first, the compound boundary
disappears and the inner region's atoms become direct children of the
enclosing scope, with the same effective wiring as the pass-through
description above.

### 6.8 `scf.yield`

* Already a thin terminator. The lowering of the parent op produces
  the yield's effect; the standalone yield is removed.

## 7. Memory Dependence Model

The compositional chain model, alias oracle, dependence builder,
loop-carried memory state, and token wiring rules are specified in
`docs/spec-compiler-part-3-mem.md`. Per-`scf.*` boundary translation
rules in §6 instantiate that model with op-specific structural and
memory-plane wiring.

## 8. Spatial Array

Spatial-array layout and in-thread queries are specified in
`docs/spec-compiler-part-4-spatial.md`, along with future-thoughts
discussion of neighborhood communication / distributed-buffer
protocols. They are not required for SCF-to-DFG flattening; this
document references them only at the boundary points (see §5.4 and
§9).

## 9. Verifier Rules (Front-End Specific)

In addition to the existing dataflow / fabric verifier set:

* `dataflow.thread`
  - `mapping` array length equals grid dim count.
  - Every `mapping` entry implements
    `DeviceMappingAttrInterface`.
  - No two `mapping` entries share the same `(kind, mappingId)`
    pair: the verifier rejects, for example, two grid dims that
    are both labeled `#loom.spatial<x>` or both labeled
    `#loom.temporal<linear_dim_0>`. Uniqueness is checked across
    the whole `mapping` array, where `kind` is the discriminator
    between `#loom.spatial<...>` and `#loom.temporal<...>` (and any
    future sibling attribute that implements
    `DeviceMappingAttrInterface`) and `mappingId` is the per-kind
    axis identifier (`x | y | z | linear_dim_0 | ... | linear_dim_9`
    per §5.2).
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
  - Each body operand and its corresponding entry block argument
    must share the same type exactly.
  - Each memref-like body operand must be the direct SSA result of a
    `dataflow.map_info` op in the enclosing control context. The
    thread's `MemoryEffectOpInterface` must walk to that `map_info`
    op and project effects on the map source according to its
    `direction` attribute, as specified in the `dataflow.thread` op
    contract.
  - Body must not contain a `dataflow.graph` whose body contains any
    `dataflow.thread` (graph is a leaf).
  - The body must not directly contain both a `dataflow.graph` and a
    nested `dataflow.thread` at the same nesting level. ScalarCore
    code (`scf.*` ops, ScalarCore-legal `func.call`, ScalarCore
    memory ops, `dataflow.thread.fence`, etc.) is always allowed in
    the thread body and may freely interleave with whichever of the
    two shapes the body holds; this rule only rejects the
    simultaneous direct presence of a graph and a nested thread.
    The legal shapes are therefore:
    - any number of `dataflow.graph` regions interleaved with
      ScalarCore code, with no nested `dataflow.thread`;
    - any number of nested `dataflow.thread` ops interleaved with
      ScalarCore code, with no direct `dataflow.graph`;
    - ScalarCore code only, with neither.
    Mixing direct graphs with direct nested threads at the same
    level violates §3 Constitutional Rule 2's parent-side
    constraint that a thread body must not directly contain both
    a `dataflow.graph` and a nested `dataflow.thread` (this is a
    separate rule from the graph-body leaf rule, per the same
    Rule 2 wording).
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
  - The op's `MemoryEffectOpInterface` implementation must report
    `MemRead + MemWrite` on MLIR's default memory resource. Lowering
    and verification must not weaken this to a per-resource effect or
    to no effect; doing so breaks the ScalarCore-side barrier
    contract specified in §3 rule 8.

* `dataflow.thread.wait`
  - At least one operand. Each is `!dataflow.thread_token`.
  - The op's `MemoryEffectOpInterface` implementation must report
    `MemRead + MemWrite` on MLIR's default memory resource. The wait
    has no SSA result, so this barrier is the only mechanism that
    keeps surrounding host or parent-context memory ops from being
    moved across it.

* `dataflow.map_info`
  - `direction` is one of the closed enum values.
  - `staticBounds` rank, if present, equals the source memref rank.
  - The op may appear at host scope or inside another
    `dataflow.thread`'s ScalarCore region; it must not appear inside
    `dataflow.graph`.
  - The op's result must be used only as a `dataflow.thread` body
    operand. Any other use -- passing the result to `memref.load`,
    `memref.subview`, `func.call`, another `dataflow.map_info`, or any
    op other than `dataflow.thread` -- is rejected. This complements
    the `dataflow.thread` rule that "each memref-like body operand
    must be the direct SSA result of a `dataflow.map_info` op":
    together the two rules close the loop on map_info provenance and
    keep the same-type passthrough memref from being treated as an
    ordinary memref by the rest of the IR.

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
  - Position contract for the four lists. Let `N` be the number of
    user (non-control) operands and `M` the number of user
    (non-control) results. Then the graph op's operand list has
    length `N + 1` (control at index 0, user operands at indices
    1..N), the entry block argument list has length `N + 1`
    (control at index 0, user block arguments at indices 1..N),
    the `dataflow.yield` operand list has length `M + 1` (control
    at index 0, user yield operands at indices 1..M), and the
    graph op's result list has length `M + 1` (control at index 0,
    user results at indices 1..M). The verifier enforces, for
    every index `i` in `1..N`, that operand `i` and block argument
    `i` have the same type, and for every index `j` in `1..M`,
    that yield operand `j` and graph result `j` have the same
    type. The index-zero slots are the explicit control ports
    already constrained to type `none` by the previous bullet.
    There is no implicit reordering between any of the four lists;
    the i-th user operand is bound to the i-th user block argument,
    and the j-th user yield operand is bound to the j-th user
    result, in declaration order.
  - Body may contain `dataflow.{stream, carry, invariant, gate, mux,
    demux, sync, constant, load, store, yield}` plus ordinary
    pure ops permitted in the existing graph body whitelist.
  - Body may not contain `scf.*`, `func.func`, `func.call`,
    `dataflow.thread`, `dataflow.thread.fence`,
    `dataflow.map_info`, any spatial-array op specified in
    `docs/spec-compiler-part-4-spatial.md`, or another
    `dataflow.graph`.
  - The op declares `RecursiveMemoryEffects`, so generic MLIR
    optimizers see the body's memory effects through the op
    boundary. For this to be useful, body ops with side effects
    (notably `dataflow.load` / `dataflow.store`) must themselves
    declare `MemoryEffects` accurately; see §5.5 for the milestone
    contract on the dataflow primitives.

## 10. Non-Goals (First Milestone)

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
* Spatial-array carrier promotion to a strong-typed
  `!dataflow.spatial_array`, the symbol-form `dataflow.mesh`, and
  any future neighborhood communication / distributed-buffer
  protocol for tile-and-mesh memrefs. These are documented as
  Part 4 future thoughts in `docs/spec-compiler-part-4-spatial.md`
  and are not required for this milestone. In particular, the
  first milestone does not commit to any stencil-specific op
  signature for neighbor exchange.

## 11. References

* `docs/spec-fabric-module.md`, `docs/spec-fabric-pe.md`,
  `docs/spec-fabric-fu.md` -- the existing fabric-side IR that the
  front-end output eventually targets.
* `docs/spec-compiler-part-1-source.md` -- high-level source
  integration and metadata emission.
* `docs/spec-compiler-part-2-scf.md` -- LLVM-to-SCF raising,
  accelerator-region selection, and `loom.acc_region`.
* `docs/spec-compiler-part-3-impl.md` -- pass pipeline, lit-test
  layout, milestone acceptance checklist, and maintenance plan
  for the SCF-to-DFG front-end.
* `docs/spec-compiler-part-3-mem.md` -- compositional chain model,
  alias oracle, dependence builder, loop-carried memory state, and
  token-wiring rules used inside each `dataflow.graph`. Per-`scf.*`
  boundary translation rules in §6 of this document instantiate
  that model.
* `docs/spec-compiler-part-3-placement-framework.md` -- common
  placement-partition framework; Part 3 owns the L2 graph-placement
  instance.
* `docs/spec-compiler-part-4-spatial.md` -- spatial-array
  annotation, in-thread queries, and future-thoughts discussion of
  neighborhood communication / distributed-buffer protocols.
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
