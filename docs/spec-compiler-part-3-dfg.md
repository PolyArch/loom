# Loom Compiler Part 3: SCF to DFG

This document specifies the third compiler part of the Loom front-end:
lowering SCF-shaped accelerator regions into Loom's native dataflow
representation, ready for fabric mapping and lowering.
It starts after source integration and LLVM-to-SCF raising have already
selected explicit accelerator regions. It does not decide which source
program regions should run on AccCores.

The target Part 3 dataflow surface uses module-scope, Symbol-bearing,
function-like definitions for both `dataflow.thread` and
`dataflow.graph`. Execution is materialized only by
`dataflow.thread.launch` and `dataflow.graph.launch`. Graph control
ports are explicit: `ctrl_in` and `done_out` are part of the
`dataflow.graph` definition's `function_type` and of every launch
site. Part 3 consumes the transient `loom.acc_region` op produced by
Part 2.
The precise timing semantics of `dataflow.stream`, `dataflow.carry`,
`dataflow.invariant`, and `dataflow.gate` are specified separately in
`docs/spec-dataflow-part-1-streaming.md`. The precise firing semantics
of `dataflow.constant`, `dataflow.sync`, `dataflow.mux`, and
`dataflow.demux` are specified separately in
`docs/spec-dataflow-part-2-control.md`.

Implementation engineering -- the pass pipeline that produces this IR
shape, the lit-test layout, the acceptance checklist, and the
maintenance plan -- is documented in
`docs/spec-compiler-part-3-impl.md`. The main body of this document
keeps only the first-principles content: IR boundary contracts, SCF
flattening templates, the memory-dependence model, and verifier
invariants.

Placement decisions across the compiler are described by
`docs/spec-compiler-part-3-placement-framework.md`. Part 3 owns the
L2 instance: choosing which code inside a `dataflow.thread`
definition's body becomes a `dataflow.graph` definition + a
`dataflow.graph.launch` at the cut site. The placement framework
does not weaken the verifier or IR contracts in this document; it
only states how legal partitions are generated, ranked, and made
replaceable by later cost-aware policies.

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
  accelerator regions and lowers them to `dataflow.thread`
  definitions + `dataflow.thread.launch` ops, plus
  `dataflow.graph` definitions + `dataflow.graph.launch` ops at
  the cut sites. The accompanying
  `docs/spec-compiler-part-3-impl.md` documents the pass pipeline,
  testing, and acceptance for this part. L2 graph placement follows
  the common placement framework in
  `docs/spec-compiler-part-3-placement-framework.md`.
* **Part 4, partitioned data.** Annotation and in-thread queries for
  tile-and-domain memrefs, plus the extension point for neighborhood
  communication / distributed-buffer protocols (see
  `docs/spec-compiler-part-4-partitioned-data.md`).

Input to this part is an MLIR module with `func.func` host containers.
Host code may remain outside accelerator regions. AccCore code must be
inside explicit `loom.acc_region` ops, except in the
`wrap-standalone-kernel` test mode (see
`docs/spec-compiler-part-3-impl.md`). A `func.func` is therefore an
ABI and ownership container, not an implicit accelerator boundary.

Output is the canonical Loom front-end IR: module-level `func.func`
symbols holding ordinary HostCore or ScalarCore code; module-level
`dataflow.thread` definitions reached by zero or more
`dataflow.thread.launch` ops; and module-level `dataflow.graph`
definitions reached by zero or more `dataflow.graph.launch` ops
inside thread definitions. No `scf.*` op is left inside any
`dataflow.graph` definition's body. All `scf` ops are
supported inside accelerator regions:
`scf.if`, `scf.while` with `scf.condition`, `scf.for` with
`scf.yield`, `scf.forall` with `scf.forall.in_parallel`,
`scf.parallel` with `scf.reduce` and `scf.reduce.return`,
`scf.index_switch`, `scf.execute_region`. Tensor-result aggregation
in `scf.forall` is supported by materializing
`scf.forall.in_parallel` combining actions into explicit
destination-buffer effects before thread promotion. The
`dataflow.thread.launch` op carries the completion token and
mapped-memory data transfer; the def remains a callable kernel
body, not a tensor-result returning op. Memory dependence
construction runs in this part; alias analysis is only the
conflict oracle used by that builder (see
`docs/spec-compiler-part-3-mem.md`).
Graph placement inside each thread is governed by the L2 placement
instance specified by the placement framework and by the implementation
contract in `docs/spec-compiler-part-3-impl.md`.

## 2. Hardware Model

Loom's execution target is a heterogeneous system containing HostCore
execution and one or more AccCore execution resources. A physical
AccCore is a system-level resource described by `fabric.system`: the
`acc_core` node carries ScalarCore parameters and references a
`fabric.module` symbol as its SpatialCore template. `fabric.module`
remains the SpatialCore or CGRA template only; it does not own the
physical AccCore instance or ScalarCore parameters.

The front-end IR in this document remains a software and logical
execution model. Binding logical execution cells to physical AccCore
instances, and selecting the system-level ScalarCore/SpatialCore
resources, belongs to mapping and binding artifacts.

The front-end's IR mirrors this trio:

| Hardware | Front-end IR carrier |
|----------|----------------------|
| HostCore | Host-call-context `func.func` body code outside any `dataflow.thread.launch` |
| Logical execution domain | An outermost `dataflow.thread` definition (Symbol-bearing, module-scope), launched at host scope by `dataflow.thread.launch` with `mapping = [#loom.thread_axis<...>, ...]` logical execution-axis tags. Each dynamic thread instance is a logical execution cell before binding. The cell-to-AccCore binding is a separate concern. |
| ScalarCore | The body of an innermost executable `dataflow.thread` definition, minus its `dataflow.graph.launch` ops, plus ScalarCore-legal `func.call` callees after inlining or specialization. The body is "what one logical execution cell runs once binding maps it to a physical AccCore". |
| SpatialCore | Each `dataflow.graph` definition referenced by a `dataflow.graph.launch` inside an innermost executable `dataflow.thread` definition's body, again per bound logical execution cell. |

A single `dataflow.thread.launch` corresponds to one launch of a
multi-dimensional iteration domain, distributed across a logical
execution domain, of the kernel defined by the referenced
`dataflow.thread` callable. The thread body is "what one
logical execution point runs"; the logical thread coordinates pulled from
the mapping attribute identify which point. The domain is a
software concept -- the programmer's view of how work and data are
partitioned -- and does not commit to a specific fabric topology.
A fabric whose physical PE / memory graph is not a Cartesian mesh
is supported by the same mapping mechanism. The binding from a
logical execution point to a physical AccCore is a separate concern;
see the placement framework and the later binding/PnR specs.

An innermost executable thread is a `dataflow.thread` whose body, at
the thread-body placement level, does not launch another
`dataflow.thread`. Such a body may contain ScalarCore residual code and
`dataflow.graph.launch` ops. Only dynamic instances of innermost
executable threads are eligible to become one physical AccCore
execution slot after binding/PnR. Non-innermost threads remain logical
parallel hierarchy and scheduling structure before binding.

A ScalarCore-only innermost executable thread body is legal, but this is
only an IR legality statement. Such a thread remains AccCore work only
when L1 placement, explicit source intent, or a DSE policy selected the
enclosing region for accelerator execution. L2 graph placement failure
must not synthesize a new accelerator offload from unselected host code.

Thread completion and graph/dataflow control are distinct token domains.
`!dataflow.thread_token` is the inter-thread asynchronous completion
token produced by `dataflow.thread.launch`. `none` values are the
graph-control, graph-completion, streaming-control, and memory-order
tokens used inside dataflow. There is no implicit cast or general
conversion between the two domains. `dataflow.thread.fence` is the
explicit bridge that accepts `none` and/or `!dataflow.thread_token`
dependencies and emits one `none` control value. `dataflow.thread.wait`
consumes `!dataflow.thread_token` values for host or parent-context
synchronization and emits no SSA value.

Thread hierarchy transforms before binding are legal only as explicit
optimization policies. They may reorder independent thread levels,
collapse adjacent independent levels, or tile and split a level when the
transform preserves the logical instance set, each instance's scalar
values, memory-order constraints, async launch and fence ordering, and
the strict layering rule between child thread launches and graph
launches. The deterministic baseline policy performs only annotation
and canonicalization; it must not silently change hierarchy shape as a
verifier or parsing side effect.

### 2.1 IR Carrier Responsibilities

* `func.func` is a callable symbol and ABI unit. It does not by itself
  choose HostCore or AccCore placement. A function may be HostCore-only,
  ScalarCore-callable, or legal in both contexts depending on the
  Part 2 call-context classification.
* `loom.acc_region` is a transient Part 2 to Part 3 marker for a
  structured region selected for AccCore execution. This part consumes
  it and erases it.
* `dataflow.thread` is the logical accelerator execution-domain
  **definition** (Symbol-bearing, module-scope, function-like). It
  owns the kernel body, the static grid shape, and the mapping
  attribute. It does not itself execute; dynamic logical instances are
  materialized by one or more `dataflow.thread.launch` ops at use
  sites, then later binding decides which innermost executable
  instances occupy physical AccCore slots.
* `dataflow.thread.launch` is the logical accelerator execution
  boundary. It
  references a `dataflow.thread` callable by symbol, supplies async
  dependencies, dynamic-grid values, and body operands (memrefs
  through `dataflow.map_info`, scalars by value), and produces the
  completion token.
* `dataflow.graph` is the SpatialCore leaf DFG **definition**
  (Symbol-bearing, module-scope, function-like). Its body cannot
  contain `func.func`, `func.call`, `dataflow.thread.launch`,
  `dataflow.graph.launch`, or another `dataflow.graph` definition.
* `dataflow.graph.launch` is the SpatialCore execution boundary
  inside a `dataflow.thread` definition's body. It references a
  `dataflow.graph` callable by symbol, supplies the leading
  `ctrl_in : none` and user operands, and yields a leading
  `done_out : none` plus user results.
* `dataflow.subgraph` is a migration-only adapter input. Canonical
  `dataflow.graph` definitions do not persist L3 partitions; TechMapping owns
  actor grouping and selected FU realization.

Function definitions remain module-level symbols in this design.
`dataflow.thread` definitions are also module-level symbols (and
not symbol tables themselves) and do not physically contain
`func.func` definitions. A `func.call` inside a `dataflow.thread`
definition's body is a ScalarCore call. If the callee contains
code that must become a `dataflow.graph` definition or a nested
`dataflow.thread` launch, Part 3 must inline or specialize that
callee into the active thread definition before graph extraction.
Non-inlined ScalarCore calls may remain only when their callee
body is graph-free after this preparation.

## 3. Constitutional Rules

The eight rules below are invariants that downstream passes and
verifiers must enforce; the rest of this spec is a refinement of how
each rule lands in IR.

1. `dataflow.thread` is the logical parallel execution-domain
   primitive used for selected accelerator work. It is a
   Symbol-bearing, module-scope, function-like definition (Part 3
   Section 5.4.1); dynamic logical instances are materialized by
   `dataflow.thread.launch` ops. Multi-level launch nesting is
   allowed; depth has no hard upper bound. A dynamic instance becomes
   a physical AccCore execution slot only after binding/PnR, and only
   when it is an innermost executable thread instance. The thread
   definition's body has a `thread_ctrl : none` block argument that
   fires once the logical thread instance starts executing
   (entry-block layout: `(args_*, thread_ctrl, iv_*)`, see Section 5.4.1).
   The body may contain ScalarCore operations and ScalarCore-legal
   `func.call` operations, but not `func.func` definitions.
2. `dataflow.graph` is a leaf-level definition. It is also a Symbol-
   bearing, module-scope, function-like definition (Part 3 Section 5.5);
   execution is materialized by `dataflow.graph.launch` ops inside a
   thread definition's body. Its body must not contain any
   `func.func`, `func.call`, `dataflow.thread.launch`,
   `dataflow.graph.launch`, or another `dataflow.graph` definition.
   The graph body is a single graph-kind region; it already permits
   feedback edges (accepted semantics). Additionally, from the
   parent side: a `dataflow.thread` definition's body must not
   directly contain both a `dataflow.graph.launch` and a
   `dataflow.thread.launch` at the same thread-body placement level.
   This is a
   separate constraint from the leaf rule above (the leaf rule
   constrains the graph body; this parent-side constraint is on
   what may sit alongside a graph launch in its enclosing thread
   body). The accepted thread hierarchy is strictly layered:
   non-innermost thread bodies may contain ScalarCore orchestration
   code and direct `dataflow.thread.launch` ops, but must not directly
   contain `dataflow.graph.launch` ops. Innermost executable thread
   bodies may contain ScalarCore residual code and direct
   `dataflow.graph.launch` ops, but must not directly contain child
   `dataflow.thread.launch` ops. A single thread-body placement level
   must never directly mix thread launches and graph launches. Both
   rules are enforced by the verifier (see Section 9).
3. Every `dataflow.graph` definition has explicit control ports
   inside its `function_type`: the inputs lead with `ctrl_in : none`,
   the results lead with `done_out : none`, the entry block of the
   body has a matching leading `ctrl_in : none` block argument, and
   `dataflow.yield` has a matching leading `done_out : none` operand.
   These `none` values are real SSA values in the operation state and
   the function signature, because they lower to physical start/done
   ports on hardware and because they expose the synchronization
   contract symbolically (so launch-to-launch sequencing inside the
   same innermost thread body can match end-to-end at the symbol-ref
   level). Custom assembly may hide or compress them for readability,
   but generic form and verifier logic treat them as ordinary
   signature elements, operands, and results. At each
   `dataflow.graph.launch` site the
   ctrl/done slots become real per-launch SSA values: the contract
   is "graph clients may begin issuing memory ops once the launch's
   `ctrl_in` operand is hot; the launch's `done_out` result becomes
   hot when every memory op in the launched graph has retired."
4. The HostCore-to-AccCore data plane is mediated by
   `dataflow.map_info`. Every value that crosses a thread boundary
   as data (memref, partitioned-data handle) at a
   `dataflow.thread.launch` op must be the direct SSA result of one
   `dataflow.map_info` op in the launch's enclosing context, before
   being consumed inside the thread definition's body.
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
   is the baseline default. The stronger oracle is a refinement policy
   whose output must remain structurally compatible with the basic
   oracle while allowing fewer dependence predecessors where MLIR AA
   proves `MustNotAlias`. The
   compositional chain model and the oracle / builder / loop-state /
   wiring details are specified in
   `docs/spec-compiler-part-3-mem.md`.
6. `loom.acc_region` is the explicit AccCore selection boundary
   consumed by this part. `scf.forall` with a
   `mapping = [#loom.thread_axis<...>, ...]`
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
   `dataflow.graph` definition bodies.
7. `dataflow.thread` and `dataflow.graph` definitions are both
   `IsolatedFromAbove`. No operation inside either definition's body
   may directly use an SSA value defined in the surrounding scope.
   Every boundary value must appear as an explicit launch op
   operand and as a matching entry block argument of the
   referenced definition. For a `dataflow.thread.launch`, the
   operand list is the HostCore-to-AccCore launch ABI: memrefs and
   partitioned-data handles cross through `dataflow.map_info`, while
   scalar values cross by value. For a `dataflow.graph.launch`,
   operands and results are the explicit SpatialCore data/control
   ports. The `dataflow.thread.launch` op implements
   `MemoryEffectsOpInterface` directly and projects boundary
   effects from its `dataflow.map_info`-rooted operands; this
   projection, not recursive region effects on the def, is what
   host code observes for an async thread launch. The
   `dataflow.graph.launch` op also implements
   `MemoryEffectsOpInterface` directly: it resolves the callee
   symbol and walks the callee body to project effects (a
   sibling-symbol launch has no nested region, so the upstream
   `RecursiveMemoryEffects` trait is the wrong tool here). Each def
   carries `RecursiveMemoryEffects` so module-scope walkers can
   observe per-callable effects without re-implementing the
   boundary projection.
8. Effect visibility contract. Every front-end op whose execution
   affects program order, memory state, or async completion must
   declare its effects through MLIR's `MemoryEffectOpInterface` (or
   an equivalent recursive trait) accurately enough that generic
   optimizers (CSE, LICM, scheduling, code motion) preserve the
   intended observable behavior. The baseline policy uses MLIR's
   default-resource barrier pattern -- broad, conservative
   `MemRead + MemWrite` declarations -- where a precise per-resource
   binding would require op-side machinery outside this contract.
   Tighter per-resource bindings (for example, load/store keyed on
   the `$mem` operand) are explicit extensions. In
   addition, `dataflow.thread.launch` declares a conservative
   side effect on a custom `LoomAsyncResource` resource so that
   generic CSE / DCE never removes a launch even when its callee
   body has no host-visible memory effects (see Section 5.4 for the launch
   signatures).

## 4. Glossary

* **HostCore.** The general-purpose CPU that runs host-call-context
  `func.func` body code outside any `dataflow.thread.launch`.
* **AccCore.** One physical accelerator execution resource represented
  by a `fabric.system` `acc_core` node. The node carries ScalarCore
  metadata and references a `fabric.module` symbol as its SpatialCore
  template. Part 3 does not create physical AccCore instances; it
  creates logical accelerator work that later binding/PnR maps to
  AccCore resources.
* **ScalarCore-callable function.** A module-level `func.func` that
  Part 2 classified as legal to call from code running inside a
  `dataflow.thread` definition's body. Such a function remains a
  symbol; Part 3 either preserves calls to it as ScalarCore calls or
  inlines / specializes it before graph extraction.
* **Parallel thread axis.** A grid dim of a `dataflow.thread`
  definition tagged as `#loom.thread_axis<parallel, axis>`. Distinct
  dynamic values along that logical axis may be bound to distinct
  AccCore execution slots and run concurrently when resources and
  policy allow it. This is an execution-axis intent, not a hardware
  coordinate.
* **Multiplexed thread axis.** A grid dim of a `dataflow.thread`
  definition tagged as `#loom.thread_axis<multiplexed, axis>`.
  Distinct dynamic values along that logical axis may reuse an AccCore
  execution slot through time multiplexing. This is an execution-axis
  intent, not a hardware coordinate.
* **Partitioned data.** A `memref<...>` annotated with a tile-and-domain
  layout descriptor; lets in-thread code query its local tile via
  `dataflow.local_range`. The domain is the same logical
  partition-domain referenced by thread-axis tags.
* **Software subgraph.** A `dataflow.subgraph` inside a
  `dataflow.graph` definition. It groups graph-compute operations for
  L3 fabric matching. It carries no fabric placement, route, schedule,
  temporal tag, or physical-resource binding semantics.
* **Mapping attribute.** Any attribute that implements
  `mlir::DeviceMappingAttrInterface`. The target front-end ships
  `#loom.thread_axis<kind, axis, domain?>` instances and recognizes
  them for thread promotion and verifier checks. A third-party
  attribute that implements the same interface is not recognized for
  thread promotion. Three treatment cases for an `scf.forall`'s
  `mapping` array, in
  agreement with Section 6.4 lowering rules:
  - **Empty `mapping` attribute** (the array is literally empty,
    or the attribute is absent): the forall is unmapped and is
    flattened by Part 3's `scf.parallel` normalization path.
  - **Mapping array with at least one Loom-recognized entry and
    no foreign entry**: the forall is promoted to a
    `dataflow.thread` definition + a `dataflow.thread.launch` at
    the original site.
  - **Mapping array with at least one foreign (non-Loom) entry**
    (whether or not it also contains Loom-recognized entries):
    the front-end rejects it with a diagnostic. Part 2 or an
    earlier Part 3 pass must remove or translate the foreign
    entries before this point; the placement framework cannot
    decide which dim a foreign entry binds.
  Adding new Loom-recognized mapping attributes (for example
  `#loom.warp<...>`) is an extension point in
  `docs/spec-compiler-part-3-impl.md` Section 4.
* **Thread token.** A value of type `!dataflow.thread_token`, a
  one-shot completion signal modelled on `!async.token`. It belongs to
  the inter-thread asynchronous-completion domain, not to the
  `none`-typed graph/control token domain.
* **Thread control token.** A `none`-typed entry-block argument of
  a `dataflow.thread` definition's body (named `thread_ctrl`,
  positioned after the function-signature args per Section 5.4.1). It is
  the per-instance AccCore start signal used to launch root
  `dataflow.graph.launch` ops or ScalarCore / SpatialCore fences.
* **Thread fence.** A ScalarCore barrier op that waits at a precise
  `dataflow.thread` definition's body program point for zero or
  more SpatialCore `none` tokens and/or child
  `!dataflow.thread_token` values, then emits a `none` token usable
  as a `dataflow.graph.launch` `ctrl_in` operand. It is the only
  primitive that bridges thread-completion tokens into graph-control
  tokens.
* **Map info result.** A value produced by `dataflow.map_info` that
  carries the same type as its source memref. It is a pure, view-
  like alias of the source; by IR convention it must only be
  consumed as a `dataflow.thread.launch` body operand. Direction
  and optional bound information live as attributes on the producing
  op, not on the result type.
* **`MemAliasOracle`.** The C++ interface (canonical spelling
  matches the C++ class name) returning
  `MustNotAlias` / `MayAlias` / `MustAlias` for any pair of memory
  access ops inside one `dataflow.graph` definition's body. It
  answers conflict only; it does not define execution order.
  Specified in `docs/spec-compiler-part-3-mem.md` Section 3.
* **Memory dependence edge.** A directed edge `p -> o` saying memory
  access `o` must wait for memory access `p` before issuing its
  side effect or externally visible read. Specified in
  `docs/spec-compiler-part-3-mem.md` Section 4.
* **Loop-carried memory state.** A hidden `none`-typed control state
  carried by a lowered loop for one alias/dependence partition. It
  represents "all memory effects in this partition from previous
  dynamic iterations have retired." Specified in
  `docs/spec-compiler-part-3-mem.md` Section 5.
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
  `dataflow.graph` definition's body. Each per-partition frontier
  (see Section 2.4 of
  `docs/spec-compiler-part-3-mem.md`) flows through its own
  memory-order tokens; the leaf rendezvous in Section 6.4 of that
  document combines a structural permission token with a
  memory-order predecessor token at each load / store. Memory-order
  tokens do not encode dynamic execution path (that is the
  structural execution role of Section 2.1 there).
* **Aggregation-form forall.** An `scf.forall` with `shared_outs`,
  op results, or non-empty `scf.forall.in_parallel` combining actions
  such as `tensor.parallel_insert_slice`.
* **Effect-form forall.** An `scf.forall` with no `shared_outs`, no
  op results, and an empty `scf.forall.in_parallel` terminator. Its
  observable behavior is expressed through explicit memory effects.

## 5. IR Additions

This section enumerates every new dialect element the front-end
introduces. All additions are local to the `dataflow` and `loom`
namespaces; nothing outside this list is added.

### 5.1 New Types

* `!dataflow.thread_token`
  - One-shot completion signal. Equivalent of `!async.token` for the
    Loom front-end.
  - Belongs only to the inter-thread asynchronous-completion domain.
    It is not a `none`-typed graph-control token, and there is no
    implicit cast between the two domains.
  - Runtime ABI ownership and refcounting are specified by the runtime
    ABI; Part 3 manipulates the type as an SSA value.

This spec introduces no other types. The host-to-AccCore data
plane uses `dataflow.map_info` (see Section 5.4.6), whose result preserves
the source type. The "this value crossed the boundary through
`dataflow.map_info`" provenance is enforced by the verifier on
`dataflow.thread.launch`, not by a wrapper type.

### 5.2 Attribute Interface Instances

One new attribute class implements the upstream
`mlir::DeviceMappingAttrInterface`:

* `#loom.thread_axis<kind, axis : i64, domain = SymbolRefAttr?>`
  - `kind` is a closed enum with two values: `parallel` and
    `multiplexed`.
  - `axis` is a non-negative logical execution-axis identifier.
    There is no closed enum and no fixed cap on axis count.
  - `domain` is an optional `SymbolRefAttr` qualifier that names a
    logical partition domain when one is needed to disambiguate
    layout queries. It is not a fabric module, PE coordinate, memory
    tile, router, x/y coordinate, or topology statement.
  - `parallel` means distinct dynamic values along this logical axis
    may be bound to distinct AccCore execution slots and run
    concurrently when resources and policy allow it.
  - `multiplexed` means distinct dynamic values along this logical
    axis may reuse an AccCore execution slot through time
    multiplexing.
  - **Print form.** Without a domain qualifier, the printer emits
    `#loom.thread_axis<parallel, 2>` or
    `#loom.thread_axis<multiplexed, 2>`. With a qualifier, the
    printer emits `#loom.thread_axis<parallel, 2, @D>`.
    The parser accepts only this positional form.
  - `getMappingId()` returns the integer `axis` packed into the
    interface's `int64_t` slot; `isLinearMapping()` is `false`; and
    `getRelativeIndex()` returns the position of this entry within
    the enclosing thread's `mapping` array.

### 5.3 New Operation Interfaces

* `LoomAsyncOpInterface`
  - Shape mirrors upstream `GPU_AsyncOpInterface`: the op accepts a
    variadic operand prefix of `!dataflow.thread_token` dependencies
    and optionally produces a `!dataflow.thread_token` result.
  - The baseline participants are `dataflow.thread.launch` and
    `dataflow.thread.wait`. Host-scope async memory ops such as
    alloc or memcpy may adopt the same interface through an explicit
    runtime / ABI extension.

### 5.4 New Operations (signatures only)

Each op below is given by its TableGen-level signature: arguments,
results, regions, traits. Implementation bodies are out of scope for
this spec.

The thread half of the front-end IR is split into a **definition**
op (`dataflow.thread`, Section 5.4.1) and a **launcher** op
(`dataflow.thread.launch`, Section 5.4.2). The definition op is a Symbol-
bearing, function-like, module-scope callable; the launcher op
references the definition by symbol and materializes one async
launch instance per use site. Every executable thread in the IR is a
def + at least one launch. This split mirrors `gpu.func` /
`gpu.launch_func`.

#### 5.4.1 `dataflow.thread` (definition)

```
arguments:
  TypeAttr:$function_type,
  SymbolNameAttr:$sym_name,
  StrAttr:$sym_visibility,
  DenseI64ArrayAttr:$staticGridLowerBound,
  DenseI64ArrayAttr:$staticGridUpperBound,
  DenseI64ArrayAttr:$staticGridStep,
  DeviceMappingArrayAttr:$mapping,
  OptionalAttr<DictionaryAttr>:$arg_attrs,
  OptionalAttr<DictionaryAttr>:$res_attrs;
results:
  none;
regions:
  SizedRegion<1>:$body;
traits:
  AutomaticAllocationScope,
  IsolatedFromAbove,
  Symbol,
  HasParent<"ModuleOp">,
  SingleBlockImplicitTerminator<"ThreadYieldOp">,
  DeclareOpInterfaceMethods<CallableOpInterface>,
  DeclareOpInterfaceMethods<FunctionOpInterface>,
  RecursiveMemoryEffects.
```

* `dataflow.thread` is a Symbol-bearing, module-scope, function-
  like callable. It does not itself execute; one or more
  `dataflow.thread.launch` ops materialize launches of it.
* `function_type` is a `FunctionType` whose inputs are the kernel's
  user-data operand types `(T0, ..., TN)` and whose results are
  empty. The thread definition has no SSA data results; the
  per-launch completion token is launch-side, not part of the
  callable signature. The asynchronous-execution semantics is
  expressed at the launch op via `LoomAsyncOpInterface`, not via
  the function type.
* `sym_name` is required and module-unique. `sym_visibility` is
  required and must equal `"private"` under the baseline visibility
  policy. The verifier rejects `"public"` and `"nested"` unless
  cross-module linkage is enabled by a separate spec.
* `mapping` is a `DeviceMappingArrayAttr` (an `ArrayAttr` whose
  every entry implements `DeviceMappingAttrInterface`), one per
  grid dim. The target Loom mapping entries are
  `#loom.thread_axis<parallel, ...>` and
  `#loom.thread_axis<multiplexed, ...>`. The relative order in the
  array equals the relative order of the grid dim. Each entry's
  `axis` refers to a logical execution axis (per Section 5.2). If an axis
  participates in a partitioned-data query, it must carry the
  relevant logical partition-domain symbol explicitly. No entry is
  interpreted as a hardware coordinate by Part 3 alone; any binding
  from logical execution cell to physical AccCore is a separate
  concern (see `docs/spec-compiler-part-4-partitioned-data.md`).
* `staticGrid*` arrays describe kernel-shape, not per-call values.
  They live as op attributes on the def. Entries equal to
  `ShapedType::kDynamic` refer to the corresponding `dynamicGrid*`
  operand at every launch site that references this definition.
  Static / dynamic mixing is per-axis and is consistent across all
  launches of the same def.
* The entry block of `body` has the layout
  `(args_*, thread_ctrl, iv_*)`:
  - The first `N` block arguments mirror `function_type.inputs`
    exactly (each user body operand). Putting the signature args
    first preserves the upstream `FunctionOpInterface` invariant
    that the entry block's first `N` arguments correspond to
    `function_type.inputs[0..N]`. This matches the `gpu.func`
    precedent of "function args first, implicit extras after".
  - `thread_ctrl : none` is the per-launch AccCore start signal.
    It is produced by the launch op once async dependencies are
    satisfied and the AccCore instance begins execution. Root
    `dataflow.graph.launch` ops with no ScalarCore predecessor use
    this value as their `ctrl_in` operand.
  - `iv_0, ..., iv_K : index` are the per-instance grid iteration
    indices, one per static-grid rank entry, in source-dim order.
* The body is `IsolatedFromAbove`. No SSA value defined outside
  the def's body may be used inside it; the launch's body operands
  are the only inputs.
* `dataflow.thread` implements `RecursiveMemoryEffects` so module-
  scope walkers can observe per-callable effects without re-
  implementing the boundary projection. This is **not** the
  primary effect surface seen by host code; that is the
  `dataflow.thread.launch` op's own
  `MemoryEffectsOpInterface` projection (Section 5.4.2). A graph
  reached through this body is exposed to the def's recursive
  effect rollup via its inner launch's effects.

#### 5.4.2 `dataflow.thread.launch`

```
arguments:
  Variadic<Dataflow_ThreadToken>:$asyncDependencies,
  Variadic<Index>:$dynamicGridLowerBound,
  Variadic<Index>:$dynamicGridUpperBound,
  Variadic<Index>:$dynamicGridStep,
  Variadic<AnyType>:$bodyOperands,
  FlatSymbolRefAttr:$callee;
results:
  Optional<Dataflow_ThreadToken>:$asyncToken;
traits:
  AttrSizedOperandSegments,
  DeclareOpInterfaceMethods<SymbolUserOpInterface>,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
  DeclareOpInterfaceMethods<LoomAsyncOpInterface>.
```

`dataflow.thread.launch` deliberately does **not** implement
`CallOpInterface`. The op's only result is a `!dataflow.thread_token`,
which is a launch-level async-completion handle, not a callable
return value (the callee's `function_type` results are empty by
Section 5.4.1). Generic call-graph and inliner consumers that read
`CallOpInterface::getResults()` would get a misleading "call returns
a thread token" picture; matching the upstream `gpu.launch_func`
precedent (which also exposes async tokens but does not implement
`CallOpInterface`), thread launch carries only `SymbolUserOpInterface`
plus `LoomAsyncOpInterface`. Symbol-resolution machinery still works
through `SymbolUserOpInterface` and the explicit `callee` attribute;
custom Loom analyses can introspect the callable through
`SymbolTable::lookupNearestSymbolFrom(...)`.

* `callee` is a flat symbol reference that must resolve to a
  `dataflow.thread` definition in the same module. The verifier
  rejects launches whose `callee` cannot be resolved or whose
  resolved op is not a `dataflow.thread`.
* `bodyOperands` types must equal `callee.function_type.inputs`
  position-by-position.
* `dynamicGrid*` operand counts must equal the count of
  `ShapedType::kDynamic` sentinels in the corresponding
  `callee.staticGrid*` array. The static / dynamic mix is
  per-axis and per-array as on the def; mixing strategy across
  the three arrays follows the source `scf.forall` pattern.
* `asyncDependencies` is the variadic prefix of incoming
  `!dataflow.thread_token` dependencies (this op's
  `LoomAsyncOpInterface` slot). The op produces an
  `Optional<!dataflow.thread_token>` `asyncToken` result. The
  baseline policy always produces it (pure async style); any
  non-async lowering policy must be specified separately.
* The op has no data results. Values produced by AccCore execution
  cross the HostCore-to-AccCore boundary through mapped memory
  effects; the token is the readiness signal for those effects.
* Each memref-like operand in `bodyOperands` must be the direct
  SSA result of a `dataflow.map_info` op in the launch's enclosing
  context. The verifier enforces this provenance; the in-thread
  block argument bound to the operand is the same memref type as
  the source memref. With the def + launch split, provenance belongs
  to the launch site, where `dataflow.map_info` is reachable.
* `dataflow.thread.launch` implements `MemoryEffectsOpInterface`
  directly. The interface reports host-visible effects by walking
  each memref-like operand in `bodyOperands` back through its
  defining `dataflow.map_info` op and reading the `direction`
  attribute there:
  - `direction = to` reports `Read` on the map source.
  - `direction = from` reports `Write` on the map source.
  - `direction = tofrom` reports both `Read` and `Write` on the map
    source.
  - `direction = alloc` reports `Allocate` on the map source.
  - `direction = release` reports `Free` on the map source.
  Scalar operands do not contribute memory effects.
* Effects are reported on the `dataflow.map_info` source value, not on
  the `dataflow.map_info` result (which is a view-like alias). The
  source value is then peeled through any recognized view-like ops
  before the effect is projected, using the same view-like list as
  the alias oracle in `docs/spec-compiler-part-3-mem.md` Section 3.1; in
  particular, `dataflow.partition_layout` is one such view-like
  producer, so a launch whose `map_info` source is a
  `dataflow.partition_layout` result reports its effects on the
  underlying `partition_layout` source memref (per
  `docs/spec-compiler-part-4-partitioned-data.md`). In nested-launch
  cases inside a parent thread definition, the parent's `map_info`
  source may itself be an entry-block argument of the parent thread
  definition's body; the parent launch's own boundary summary is
  responsible for projecting its effects one level further outward.
* **Direction / body-effect compatibility check.** With one def
  reused at multiple launch sites, the launch's projection from
  `map_info.direction` must not under-report effects relative to
  the callee body's actual reads / writes on the corresponding
  block argument. For each memref-like body operand `i` of
  `dataflow.thread.launch`, the projected `direction` must cover
  every effect that the callee body declares on its `i`-th
  function-signature block argument:
  - `direction = to` requires the callee body to perform no writes
    through arg `i` (read-only).
  - `direction = from` requires the callee body to perform no
    reads through arg `i` (write-only).
  - `direction = tofrom` accepts any combination of reads and
    writes.
  - `direction = alloc` requires the callee body's first effect on
    arg `i` to be an allocation-style write before any read.
  - `direction = release` requires the callee body to perform no
    further reads / writes after the release point.
  The body's effect on each block arg is computed by walking the
  body for `MemoryEffectsOpInterface` ops keyed on that arg (and on
  aliases reachable through `dataflow.partition_layout` / view-like
  ops, per the alias oracle). Violations are diagnosed at the
  launch op with a message that names both the launch and the
  offending body op.
* **Anti-CSE / anti-DCE protection.** A `dataflow.thread.launch`
  whose callee body has no host-visible memory effects but whose
  execution is still observable (for example, a thread that only
  writes to scratch memory not visible to the host) must not be
  considered freely removable by generic MLIR optimizers. The op
  declares a conservative side effect on a custom
  `LoomAsyncResource` resource (in addition to its
  `map_info`-derived effects). Generic CSE / DCE see this resource
  as a write barrier and refuse to merge or delete launches even
  when no other memory effect is reported.

#### 5.4.3 `dataflow.thread.yield`

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
  writes before thread promotion, so `dataflow.thread` (def) does not
  need a parallel combining region or thread data results. Values
  defined inside an isolated thread definition's body never escape by
  direct SSA use.

#### 5.4.4 `dataflow.thread.fence`

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
  `dataflow.thread` definition's body, outside any
  `dataflow.graph.launch`.
* **Async dependencies.** Each `none` operand in `$deps` is a
  `dataflow.graph.launch` `done_out` result, and each
  `!dataflow.thread_token` operand is a child-thread completion
  token produced by a `dataflow.thread.launch`. SSA def-use already
  enforces "the fence happens after every defining op of `$deps`";
  no extra effect machinery is needed for that part of the
  contract.
* **ScalarCore-side memory barrier.** The fence declares
  `MemRead + MemWrite` on MLIR's default memory resource. This is the
  default-resource barrier pattern (broad and conservative, comparable
  to an `MPI_Barrier`-style global barrier). Any ScalarCore op in the
  same thread definition's body that touches memory through specific
  memrefs (via `memref.{load, store}`, `func.call`, etc.) declares
  effects that the default resource subsumes. MLIR therefore must not
  reorder any such op across the fence in either direction. Pure ops
  with no declared effects may still be reordered freely; this is
  intentional and does not change observable behavior.
* **Result token.** The `none` result fires after both the async
  dependencies are satisfied and the ScalarCore-side barrier is
  observed. The result can feed a downstream `dataflow.graph.launch`'s
  `ctrl_in` operand to express "graph B starts only after graph A and
  the surrounding ScalarCore side effects".
* This op is the only sanctioned bridge between thread completion and
  graph-launch control. There is no general cast between
  `!dataflow.thread_token` and `none`. Ordering a child thread launch
  after a graph launch completes is expressed by placing the child
  `dataflow.thread.launch` after
  `dataflow.thread.fence(%graph_done)` in ScalarCore program order.
  The fence's default-resource memory barrier (per Section 3 Constitutional
  Rule 8) keeps any op with declared memory effects from being
  reordered across the fence, which covers the common case where the
  child thread launch has at least one mapped operand and therefore
  reports boundary memory effects through `MemoryEffectsOpInterface`.
  For the uncommon scalar-only child-thread case (no mapped operands,
  no reported boundary memory effects), the front-end lowering must
  additionally close the SSA path so generic code motion has no
  freedom: it emits a trailing `dataflow.thread.fence(%child_done)`
  that consumes the scalar-only child `dataflow.thread.launch`'s
  `!dataflow.thread_token` result, and threads the prior fence's
  `none` result into the same trailing fence's operand list (the
  fence verifier accepts a mix of `none` and `!dataflow.thread_token`
  operands, see Section 9). The trailing fence's memory barrier then anchors
  the launch sequence on both sides. The lit suite covers this
  scalar-only case; in the common case the leading fence alone is
  sufficient. Note that the launch op's `LoomAsyncResource` effect
  (Section 5.4.2) is an additional belt-and-braces guard against generic
  CSE / DCE in the scalar-only case, but the fence + trailing fence
  pair remains the canonical ordering primitive.

#### 5.4.5 `dataflow.thread.wait`

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
  outer `dataflow.thread.launch`, ScalarCore code for a nested
  `dataflow.thread.launch` inside a parent thread definition's
  body. After this op, all listed thread tokens are guaranteed
  complete. Inside a `dataflow.thread` definition's body, prefer
  `dataflow.thread.fence` when the wait result must feed a
  `dataflow.graph.launch`'s `ctrl_in`.
* The op produces no SSA result, so subsequent host or parent-context
  memory ops cannot be made to depend on it through SSA def-use. To
  preserve "wait for async completion before observing memory" across
  generic MLIR optimizers, the op declares `MemRead + MemWrite` on
  the default memory resource. This is the same default-resource
  barrier pattern used by `dataflow.thread.fence`: it does not mean
  the wait itself touches memory, only that no surrounding memory op
  may be moved across it.

#### 5.4.6 `dataflow.map_info`

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

* `source` is a `memref<...>` or a partitioned-data-annotated memref
  accepted by the Part 4 partitioned-data contract.
* `result` has the same type as `source`. The op is a pure,
  view-like alias of its source: alias analysis must treat the
  result as may-alias of the source, and bufferization must treat
  the op as a metadata pass-through. The op exists to attach
  boundary metadata (direction, bounds) and to give the verifier a
  single canonical producer for `dataflow.thread.launch` body
  operands.
* `direction` is the closed enum `to | from | tofrom | alloc |
  release`. The baseline policy defaults every front-end-injected
  `map_info` to `tofrom`; an optimizer may refine to the narrowest
  direction when it can prove the narrower contract.
* `staticBounds` / `dynamicBounds` together describe the per-dim
  half-open `[lo, hi)` ranges that the thread will touch. The
  encoding pairs static and dynamic entries by dimension: for a
  source memref of rank `R`, `staticBounds`, when present, has
  length `2 * R` storing `(lo_0, hi_0, lo_1, hi_1, ..., lo_{R-1},
  hi_{R-1})` in source-dim order; `dynamicBounds` is the variadic
  list of `index` operands referenced when a `staticBounds` slot
  holds the `ShapedType::kDynamic` sentinel, in left-to-right
  iteration order over `staticBounds`. An entirely-omitted
  `staticBounds` (the attribute is not present at all) means
  "the entire memref" on every dim; in that case `dynamicBounds`
  must be empty. Partial information is encoded by setting only
  the affected slots to `kDynamic` and supplying a corresponding
  `dynamicBounds` operand for each.

Partitioned-data related ops (`dataflow.partition_layout`,
`dataflow.local_range`, `dataflow.thread_coord`,
`dataflow.thread_linear_id`) are specified in
`docs/spec-compiler-part-4-partitioned-data.md`. `dataflow.partition_layout`
appears at host scope or inside a `dataflow.thread` definition's
body (the ScalarCore portion); the query ops appear only inside a
thread definition's body. None of them appear inside a
`dataflow.graph` definition's body, and none of them participate in
the SCF flattening templates in this document.

### 5.5 Modifications to Existing Ops

The graph half of the front-end IR is split into a **definition**
op (`dataflow.graph`, Section 5.5.1) and a **launcher** op
(`dataflow.graph.launch`, Section 5.5.2). The definition op is a
Symbol-bearing, function-like, module-scope callable; the launcher
op references the definition by symbol from inside a
`dataflow.thread` definition's body, supplies a per-launch
`ctrl_in : none` operand and user data operands, and produces a
per-launch `done_out : none` result and user data results. Every
executable graph in the IR is a def + at least one launch.

#### 5.5.1 `dataflow.graph` (definition)

```
arguments:
  TypeAttr:$function_type,
  SymbolNameAttr:$sym_name,
  StrAttr:$sym_visibility,
  OptionalAttr<DictionaryAttr>:$arg_attrs,
  OptionalAttr<DictionaryAttr>:$res_attrs;
results:
  none;
regions:
  SizedRegion<1>:$body;
traits:
  IsolatedFromAbove,
  Symbol,
  HasParent<"ModuleOp">,
  DeclareOpInterfaceMethods<CallableOpInterface>,
  DeclareOpInterfaceMethods<FunctionOpInterface>,
  RecursiveMemoryEffects.
```

* `dataflow.graph` is a Symbol-bearing, module-scope, function-like
  callable. It does not itself execute; one or more
  `dataflow.graph.launch` ops materialize launches of it.
* `function_type` is a `FunctionType` whose inputs and results
  include the explicit control ports as part of the symbolic
  signature. Inputs are `(none, T0, ..., TN)` where the leading
  `none` is the `ctrl_in` start port and the remaining types are
  the kernel's user-data inputs. Results are `(none, R0, ..., RM)`
  where the leading `none` is the `done_out` completion port and
  the remaining types are the kernel's user-data results. Keeping
  ctrl and done in `function_type` exposes the synchronization
  signature symbolically, so launch-to-launch sequencing inside the
  same innermost thread body can be checked at the symbol-ref type
  level rather than by walking graph bodies.
* `sym_name` is required and module-unique. `sym_visibility` is
  required and must equal `"private"` under the baseline visibility
  policy. The verifier rejects `"public"` and `"nested"` unless
  cross-module linkage is enabled by a separate spec.
* The body is `IsolatedFromAbove`. All values used inside the
  graph definition's body must enter through the entry block.
* The entry block has the layout `(%ctrl_in : none, %arg_0 : T0,
  ..., %arg_N : TN)`, matching `function_type.inputs`. The leading
  `ctrl_in` block argument is the per-launch start signal.
* The body's `dataflow.yield` terminator has operand list
  `(%done_out : none, %r_0 : R0, ..., %r_M : RM)`, matching
  `function_type.results`. The leading `done_out` operand is the
  per-launch completion signal. These `none` values are real SSA
  values in the operation state and the function signature, because
  they lower to physical start / done ports on hardware.
* The custom parser/printer may offer a compact form for the
  control ports, but the generic form must expose the leading
  `none` slots. No analysis may depend on a hidden compiler-global
  graph start or completion state.
* `dataflow.graph` lit tests use module-scope graph definitions with
  deterministic symbol names and `dataflow.graph.launch` use sites.
  The tests carry the explicit control operand, block argument,
  result, and `done_out` plumbing through the def + launch shape.
* C++ builders construct `dataflow.graph` as a function-like
  definition from `(StringRef sym_name, FunctionType functionType,
  ArrayRef<NamedAttribute> attrs)` plus optional `arg_attrs` /
  `res_attrs` arrays. The body is added via the standard
  `FunctionOpInterface` body-construction path, with the entry block
  carrying the leading `none` `ctrl_in` block argument and the
  user-data block arguments.
* The op declares `RecursiveMemoryEffects` so module-scope walkers
  can observe per-callable effects without re-implementing the
  per-launch projection. This is **not** the primary effect surface
  seen by enclosing ScalarCore code; that is the
  `dataflow.graph.launch` op's own `MemoryEffectsOpInterface`
  projection (Section 5.5.2, which resolves the callee and walks the
  callee body).

#### 5.5.2 `dataflow.graph.launch`

```
arguments:
  none:$ctrl_in,
  Variadic<AnyType>:$bodyOperands,
  FlatSymbolRefAttr:$callee;
results:
  none:$done_out,
  Variadic<AnyType>:$results;
traits:
  DeclareOpInterfaceMethods<CallOpInterface>,
  DeclareOpInterfaceMethods<SymbolUserOpInterface>,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>.
```

* `callee` is a flat symbol reference that must resolve to a
  `dataflow.graph` definition in the same module. The verifier
  rejects launches whose `callee` cannot be resolved or whose
  resolved op is not a `dataflow.graph`.
* The verifier checks that
  `(none, type(bodyOperands)) == callee.function_type.inputs`
  position-by-position, and that
  `(none, type(results)) == callee.function_type.results`
  position-by-position. The leading `none` slots are the per-launch
  ctrl/done ports; the user data slots match the def's user inputs
  and outputs.
* The op materializes a per-launch firing of the callee at this
  exact program point. The launch is synchronous from the enclosing
  ScalarCore program's perspective: by the time `done_out` becomes
  hot, every memory op in the launched graph has retired.
* The op must appear inside a `dataflow.thread` definition's body,
  not at host scope and not inside another `dataflow.graph`
  definition's body. The verifier enforces this placement.
* The op implements `MemoryEffectsOpInterface` directly. Effects
  are projected by resolving the `callee` symbol, walking the
  callee definition's body, and reporting the union of body op
  effects through the launch boundary. If callee resolution fails
  during partial IR construction (e.g., the def has not been
  emitted yet), the launch falls back to conservative
  `MemRead + MemWrite` on MLIR's default memory resource so the
  surrounding optimizer never sees an effect-free graph launch.
  Note that the upstream `RecursiveMemoryEffects` trait is the
  wrong tool here: it aggregates effects from a region nested
  inside the op, but a graph launch references a sibling symbol
  and has no nested region. The manual implementation above is the
  intended substitute. The launch's recursive effect aggregation
  is what makes a graph that contains side-effecting body ops
  (notably `dataflow.{load, store}`) visible as a memory-touching
  op to the surrounding ScalarCore code, so that standard
  optimizers do not reorder it across `dataflow.thread.fence` or
  across other ScalarCore memory ops. This is the synchronous
  complement of the boundary-projection model
  `dataflow.thread.launch` uses for its async launch.

* `Dataflow_YieldOp`.
  - The verifier's parent-result-count and parent-result-type checks
    are updated to know about the leading explicit control result of
    the parent `dataflow.graph` definition (so the leading `none`
    slot is required in the yield operand list, matching
    `function_type.results`).

* `dataflow.load` and `dataflow.store`.
  - These dataflow primitives carry explicit memory-effect traits so that
    `dataflow.graph.launch`'s manual effect projection correctly
    aggregates body effects:
    - `dataflow.load`  declares `MemoryEffects<[MemRead]>`.
    - `dataflow.store` declares `MemoryEffects<[MemWrite]>`.
  - These use MLIR's default memory resource. They are deliberately
    coarse in the baseline policy: any load may-read all memory,
    any store may-write all memory. This is sufficient for graph
    body effects to roll up correctly through the launch's manual
    projection and for surrounding optimizers to keep ScalarCore
    memory ops correctly ordered relative to graph launches.
  - Tightening these effects to a per-`$mem`-operand declaration
    (so two loads on disjoint memrefs become reorderable) is
    an explicit dataflow dialect extension.

* No other dataflow op is modified by this spec.

## 6. Per-scf Lowering Templates

This section gives the canonical pseudocode template for each
`scf` op in terms of dataflow primitives. The supported op list in
"Scope and Contract" maps these templates: implement and lit-test the
simpler ops first.

The dataflow primitive set is
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

### Def + Launch Output Convention

The pseudocode templates in Section 6.1-Section 6.8 below show the **graph body
contents** for clarity. Every template's actual lowering output is a
`dataflow.graph` definition + a `dataflow.graph.launch` pair, with
the body shown lifted to module scope and the launch carrying the
per-instance ctrl/done plumbing:

```mlir
// At module scope (sibling of func.func):
dataflow.graph @<deterministic_sym>
    (%ctrl_in : none, <user inputs>) -> (none, <user results>) {
  // <body contents per the template>
  dataflow.yield %done_out, <user yield values> : none, <result types>
}

// At the cut site inside the enclosing dataflow.thread definition's
// body:
%done, <user results> = dataflow.graph.launch @<deterministic_sym>
    (%ctrl, <user operands>) : (none, <input types>) -> (none, <result types>)
```

The deterministic symbol naming convention is
`g_<thread_sym>_<seq>`, where `<thread_sym>` is the enclosing
`dataflow.thread` definition's symbol name and `<seq>` is the
zero-based index of the graph cut inside that thread (in source
order). Callers within `dataflow.thread.launch` cycle independently
through their own `t_<func_sym>_<seq>` namespace. The pass that
emits these symbols (see `docs/spec-compiler-part-3-impl.md`) must
be deterministic for a fixed input + option set.

The same convention applies to `dataflow.thread`: every promotion
of an `scf.forall` produces a `dataflow.thread` definition at
module scope plus a `dataflow.thread.launch` at the original
`scf.forall` site. The thread definition's body holds whatever the
templates below place inside the thread.

The templates therefore omit the def + launch wrap to keep the
body's structural diff readable. The wrap is mandatory output, not
an optimization, and is verified by the front-end's standard
verifier rules in Section 9.

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

#### If Boundary Translation

This template instantiates the boundary translation contract of
`docs/spec-compiler-part-3-mem.md` Section 2.8 for `scf.if`.

**Structural plane.** The compound's `struct_in` enters
`demux %cond` at the entry, splitting into a then-lane structural-
permission token and an else-lane structural-permission token. Each
inner region is its own chain scope per
`docs/spec-compiler-part-3-mem.md` Section 2.2 and uses its lane's token as
its `S.struct_at_*` source per
`docs/spec-compiler-part-3-mem.md` Section 6.2. The compound's `struct_done`
is `mux %cond` over the two branches' `struct_done` tokens, following
the Section 6 selector convention (lane 0 = false-lane = else, lane 1 =
true-lane = then). The same `mux` shape is reused for any data
result of the `scf.if`.

**Memory plane (per touched partition `P`).** The compound's
`incoming_C_P` enters a `demux %cond` at the entry, projecting it
into a then-lane `then_in_P` and an else-lane `else_in_P` token.
Only the active lane's projection fires, matching the dual-plane
contract of `docs/spec-compiler-part-3-mem.md` Section 2.8 (a raw SSA
fork would risk stranded memory tokens being buffered in the
unselected branch and consumed on a later selected invocation).
Each branch chain scope's `incoming_P` is its lane's projected
token. Each branch's per-`P` tail is path-forwarding per
`docs/spec-compiler-part-3-mem.md` Section 2.7: a branch that performs no
access in `P` forwards its lane projection unchanged; a branch
that performs accesses in `P` builds its tail by the single-level
chain rule of `docs/spec-compiler-part-3-mem.md` Section 2.5 inside that
branch. Call those `then_tail_P` and `else_tail_P`. The compound's
`outgoing_C_P` is the selector-matched `mux %cond` of the two
tails, following the Section 6 selector convention (lane 0 =
`else_tail_P`, lane 1 = `then_tail_P`). Per leaf rendezvous,
`docs/spec-compiler-part-3-mem.md` Section 6.4 still applies inside each
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

#### While Boundary Translation

This template instantiates the boundary translation contract of
`docs/spec-compiler-part-3-mem.md` Section 2.8 for `scf.while`.

**Structural plane.** The compound's `struct_in` initializes the
structural carry `%iter_ctrl = carry %cond, %entry_ctrl,
%ctrl_feedback`. The before-region is its own chain scope and uses
`%iter_ctrl` as its `S.struct_at_*` source. The after-region is a
separate chain scope; its structural-permission source is
`%after_ctrl` from `gate %cond, %before_done` per the while
structural template. The compound's `struct_done` is the false-cycle exit
projection of the carry, equivalently the false-lane of `demux %cond,
%before_done` reused at the boundary. The before-region executes
`K + 1` times for `K` after-region executions, matching the structural
template.

**Memory plane (per touched partition `P`).** The compound applies
the loop-carried memory state pattern of
`docs/spec-compiler-part-3-mem.md` Section 5.2 with `selector = %cond` and
the before-region / after-region instantiation of
`docs/spec-compiler-part-3-mem.md` Section 5.4. For every `P` carried by the
loop (per `docs/spec-compiler-part-3-mem.md` Section 4.3,
`P in Pi_L` iff some access in one iteration may conflict with some
access in a later iteration in `P`), the lowering introduces a hidden
per-iteration ring:

```
%mem_iter_P = carry %cond, %mem_init_P, %mem_feedback_P : none
```

* `%mem_init_P` is the compound's `incoming_C_P` per
  `docs/spec-compiler-part-3-mem.md` Section 2.4, drawn from the enclosing
  scope's per-`P` frontier at the `scf.while`'s position.
* `%mem_iter_P` enters the before-region as its `incoming_P` for
  `P`. The before-region's per-`P` tail `%before_tail_P` is built by
  the single-level chain rule of
  `docs/spec-compiler-part-3-mem.md` Section 2.5 inside the before-region;
  it forwards `%mem_iter_P` unchanged when the before-region performs
  no access in `P`.
* The after-region's `incoming_P` is `%after_in_P = gate %cond,
  %before_tail_P`, so only true-cycle iterations expose a
  `incoming_P` to the after-region. The after-region's per-`P` tail
  `%after_tail_P` is path-forwarding for the same reason
  (`%after_tail_P = %after_in_P` when the after-region performs no
  access in `P`).
* The feedback that closes the ring is `%mem_feedback_P = mux %cond,
  %before_tail_P, %after_tail_P` following the Section 6 selector convention
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
  run. This matches `docs/spec-compiler-part-3-mem.md` Section 5.4 verbatim
  and preserves any memory effect performed by the final
  condition-checking iteration.

The structural `%after_rwc` from the structural template is on the
structural plane only and is not on the memory critical path. Per
`docs/spec-compiler-part-3-mem.md` Section 2.5 plane orthogonality and
`docs/spec-compiler-part-3-mem.md` Section 5.4, after-region memory ops use
`L.ctrl = dataflow.sync(struct_after, %after_in_P)` per
`docs/spec-compiler-part-3-mem.md` Section 6.4; the structural token grants
phase permission while `%after_in_P` carries the alias-aware
ordering. Independent partitions in `Pi_L` get independent rings
sharing only the `%cond` selector, so unrelated memrefs are not
serialized.

For a partition `P` touched somewhere in the before-region or the
after-region but not in `Pi_L`, no state ring is created. The
compound's `incoming_C_P` flows into the before-region as its
`incoming_P`; the before-region's per-iteration body-tail in `P`,
plus (on the true path) the after-region's body-tail in `P`, are
gathered through the compound's structural-selector-driven
rendezvous (per `docs/spec-compiler-part-3-mem.md` Section 5.2) into the
compound's `outgoing_C_P`. No cross-iteration ordering is
introduced; the rendezvous only signals that every executed
body access in `P` has retired. A partition not touched anywhere
in the compound is absent from its interface, per Section 2.4.

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
`docs/spec-compiler-part-3-mem.md` Section 5.4 specifies for the
final-false-iteration before-tail projection. Second, `%after_rwc`
is not on the same-execution memory critical path: after-region
memory ops use `sync(struct_after, after_in_P)` for `ctrl` per
`docs/spec-compiler-part-3-mem.md` Section 2.5 plane orthogonality, while
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
# Source scf.for IVs are typed `index`. dataflow.stream requires its
# %lb / %ub / %step / iv stream to share a signless integer-like
# type (see docs/spec-dataflow-part-1-streaming.md). The lowering
# therefore inserts arith.index_cast at the boundary: %lb / %ub /
# %step are cast from index to a chosen iN, and the gated body IV
# %i is cast back to index before memref indexing. The chosen iN
# is the smallest signless int wide enough to hold the loop's bound
# range; iN here is shorthand for that choice (typically i32 or
# i64).

%lb_iN, %ub_iN, %step_iN  = arith.index_cast %lb, %ub, %step : index to iN
%i_raw, %loop_rwc = stream %lb_iN, %ub_iN, %step_iN
                      {step_op="+=", cont_cond="<"} : iN
%body_rwc, %i_iN = gate %loop_rwc, %i_raw : iN
%i = arith.index_cast %i_iN : iN to index
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
# Same IV index<->iN cast pattern as the No Iter Args case, see
# the lowering above.
%lb_iN, %ub_iN, %step_iN  = arith.index_cast %lb, %ub, %step : index to iN
%i_raw, %loop_rwc = stream %lb_iN, %ub_iN, %step_iN
                      {step_op="+=", cont_cond="<"} : iN
%body_rwc, %i_iN = gate %loop_rwc, %i_raw : iN
%i = arith.index_cast %i_iN : iN to index

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

#### For Boundary Translation

This template instantiates the boundary translation contract of
`docs/spec-compiler-part-3-mem.md` Section 2.8 for `scf.for`.

**Structural plane.** `dataflow.stream` produces the loop-level
rwc, which doubles as the structural selector. Both data-value
template cases above (No Iter Args and With Iter Args) share the
same structural-plane shape:

* The compound's `struct_in` is replicated by `dataflow.invariant`
  into the rwc-driven ctrl stream so each loop cycle carries a
  body-phase permission token. `demux %loop_rwc, %ctrl_raw`
  projects the true-lane body-phase ctrl (the body region's
  structural permission for each executed iteration) and the
  false-lane `%loop_exit_ctrl` for the sentinel reset cycle.
* The body region is a single chain scope under
  `docs/spec-compiler-part-3-mem.md` Section 2.2 and uses the
  body-phase ctrl as its `S.struct_at_*` source per
  `docs/spec-compiler-part-3-mem.md` Section 6.2.
* The compound's `struct_done` is `%loop_exit_ctrl`, taken
  unchanged from the false-lane projection above. No additional
  `dataflow.carry` is introduced on the structural plane; the
  `dataflow.carry` rings present in the With Iter Args template
  are data-value rings for iter_args, not structural-plane state.

The structural plane is therefore independent of whether the
loop has iter_args; the iter_args contribute additional
data-value `carry` / `demux` / `mux` primitives that the data
plane uses but that do not enter the structural plane wiring.

**Memory plane (per touched partition `P`).** The compound applies
the loop-carried memory state pattern of
`docs/spec-compiler-part-3-mem.md` Section 5.2 with `selector = %loop_rwc`,
specialized to `scf.for` per `docs/spec-compiler-part-3-mem.md` Section 5.3.
For every `P` carried by the loop (per
`docs/spec-compiler-part-3-mem.md` Section 4.3,
`P in Pi_L` iff some access in one iteration may conflict with some
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
  `docs/spec-compiler-part-3-mem.md` Section 2.5; they never observe
  the sentinel-cycle (rwc=false) value.
* `%mem_next_P` feeds the carry on the rwc=true lane and is built
  from the body's per-`P` tail per
  `docs/spec-compiler-part-3-mem.md` Section 2.5 / Section 2.7 (a body path that
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
scope. Independent partitions in `Pi_L` get independent rings sharing
only the `%loop_rwc` selector, so unrelated memrefs are not
serialized. Per `docs/spec-compiler-part-3-mem.md` Section 2.5 plane
orthogonality, the structural rwc carry and the per-`P` memory carry
are independent state rings over the same selector; the structural
plane never aggregates the memory tails.

For a partition `P` that is touched somewhere in the body but
not in `Pi_L` (typically a read-only partition), no state ring is
created. The compound's `incoming_C_P` flows into the body as
its per-iteration `incoming_P`, and the compound's `outgoing_C_P`
is the streamed rendezvous of every executed iteration's body
tail in `P`, per `docs/spec-compiler-part-3-mem.md` Section 5.2. No
cross-iteration ordering is introduced; the rendezvous only
signals that every body access in `P` has retired before the
loop's `outgoing_C_P` fires. A partition not touched anywhere in
the body does not appear at the compound's interface and the
enclosing scope's frontier flows past unchanged (Section 2.4).

### 6.4 `scf.forall` with `scf.forall.in_parallel`

`scf.forall` is not lowered directly to streaming dataflow ops. It is
handled as a parallel-region normalization problem before ordinary SCF
body lowering:

1. Aggregation-form forall is materialized into effect-form forall.
2. Mapped effect-form forall becomes a `dataflow.thread` definition
   at module scope plus a `dataflow.thread.launch` at the original
   forall site.
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
`#loom.thread_axis<...>` entries:

```mlir
scf.forall (%tx) in (%N) {
  memref.store %v, %B[%tx] : memref<?xf32>
  scf.forall.in_parallel {}
} {mapping = [#loom.thread_axis<parallel, 0>]}
```

It is promoted to a `dataflow.thread` definition + a
`dataflow.thread.launch` by the thread-skeleton pipeline:

```mlir
// At module scope (sibling of func.func):
dataflow.thread @t_<funcSym>_<seq>(%B_arg : memref<?xf32>, ...)
    attributes { mapping = [#loom.thread_axis<parallel, 0>],
                 staticGridLowerBound = [0],
                 staticGridUpperBound = [...],
                 staticGridStep = [1],
                 sym_visibility = "private" } {
^bb0(%B_arg : memref<?xf32>, ..., %thread_ctrl : none, %tx : index):
  memref.store %v, %B_arg[%tx] : memref<?xf32>
  dataflow.thread.yield
}

// At the original scf.forall site (after map_info materialization):
%mB = dataflow.map_info %B { direction = #to } : memref<?xf32>
%tok = dataflow.thread.launch @t_<funcSym>_<seq>(%mB, ...) async
       : (memref<?xf32>, ...) -> !dataflow.thread_token
dataflow.thread.wait %tok : !dataflow.thread_token
```

The forall grid bounds and mapping become the def's grid attributes
and `mapping`. The mapping array length must equal the forall rank;
this is already an upstream `scf.forall` verifier invariant and is
repeated here as an input requirement for thread promotion. The
forall induction variables become the trailing `iv_*` block-args of
the def's entry block (after the leading `args_*` and `thread_ctrl`,
per Section 5.4.1's `(args_*, thread_ctrl, iv_*)` layout). Values captured
from outside the forall become explicit launch operands at the use
site and matching def block-args (the leading `args_*` of the entry
block). The empty `scf.forall.in_parallel` terminator becomes
`dataflow.thread.yield` inside the def's body.

This promotion creates the AccCore boundary only. Code inside the
thread definition's body is still ScalarCore code until graph
extraction moves an eligible region into a `dataflow.graph`
definition (referenced by a `dataflow.graph.launch` at the cut
site). Only the graph definition's body is later lowered to
SpatialCore dataflow operations. Memory operations that remain
outside any graph stay in the ScalarCore part of the thread
definition's body.

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

#### Forall Boundary Translation

This template instantiates the boundary translation contract of
`docs/spec-compiler-part-3-mem.md` Section 2.8 for `scf.forall`.

A mapped `scf.forall` is promoted to a `dataflow.thread` by
`loom-build-thread-skeleton` (per
`docs/spec-compiler-part-3-impl.md` Section 1.5) before the
`dataflow.graph` chain model ever runs over it. Mapped foralls
therefore never appear as compound atoms inside a chain scope;
their launch and completion are governed by the
`!dataflow.thread_token` async protocol, which is explicitly
out of scope for the chain model per
`docs/spec-compiler-part-3-mem.md` Section 2.9.

An empty-mapping `scf.forall` does reach a chain scope. The pass
`loom-build-memory-dependencies` (per
`docs/spec-compiler-part-3-impl.md` Section 1.8) normalizes such a
forall to `scf.parallel` and from there to one or more `scf.for`
loop nests with parallel-provenance metadata. The compound that
stands for the original forall in the chain is therefore the
parallel-provenance compound described in Section 6.5 below.

### 6.5 `scf.parallel` with `scf.reduce`

`scf.parallel` is not a second dataflow loop primitive. Part 3 first
normalizes it to one or more ordinary `scf.for` loop nests, then reuses
the already specified `scf.for` template. No new `dataflow.parallel`,
`dataflow.reduce`, or reduction enum is introduced.

A user-written `scf.parallel` with a non-empty `mapping` attribute is
rejected by Part 3. Mapping has Loom semantics only on
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
`K = 1`, and `K` must be positive. The baseline policy applies one
global split factor to every `scf.parallel` that reaches this
normalization. There is no per-loop override in the baseline policy.
The N-Dim Parallel With M Reductions subsection below permits the
per-dim chunk count `K_d` to differ across dims under a cost-model-
driven policy; the carry-placement and merge contract specified there
is independent of the K choice, so per-dim K does not change the IR
contract.

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
  lowering. It may be a transient `DictionaryAttr` on the generated
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
`docs/spec-compiler-part-3-impl.md` Section 1.8), an `N`-dim `scf.parallel`
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
equivalent under the Section 6.1 template. Whichever shape is chosen, the
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
regardless of K, so an implementation policy may pick K based on
cost-model decisions without changing the IR contract. In
particular, switching any dim from `K_d = 1` to `K_d > 1` only
adds one K-chunk `scf.for` for that dim into the K-chunk nest and
extends the running accumulators' iter_arg threading through it;
the per-chunk-tuple body and the per-reduction `%iter_arg`
placement on the innermost per-chunk loop are unchanged.

After normalization, all generated `scf.for` and `scf.if` operations
use the templates in this section. Their stream, carry, gate,
demux, mux, and memory-order behavior is inherited from those templates.

#### Parallel Boundary Translation

This template instantiates the boundary translation contract of
`docs/spec-compiler-part-3-mem.md` Section 2.8 for `scf.parallel`.

After parallel-SCF normalization (per
`docs/spec-compiler-part-3-impl.md` Section 1.8), `scf.parallel` becomes one
or more `scf.for` loop nests with parallel-provenance attributes. The
outer compound that "stands for" the original `scf.parallel` in the
chain model is the parallel-provenance compound: it is the analysis-
visible group of generated chunk loops sharing one
`loom.parallel_group` id, not a new IR op. Each chunk loop body is
its own chain scope per `docs/spec-compiler-part-3-mem.md` Section 2.2.

**Structural plane.** The compound's `struct_in` forks: every chunk
receives the same SSA value as its structural-permission input
(shared `struct_in` across chunks per the Section 2.8 table for
`scf.forall` / `scf.parallel`). Each chunk's `struct_done` is the
`scf.for` template's `struct_done` for that chunk. The compound's
`struct_done` is `dataflow.sync` over all chunk `struct_done` tokens,
matching the rendezvous in `docs/spec-compiler-part-3-mem.md` Section 2.6
for parallel-provenance compound atoms.

**Memory plane (per touched partition `P`).** All chunks share the
compound's `incoming_C_P`: the same SSA value forks into each chunk
loop's per-iteration `incoming_P` (Section 5.6 of
`docs/spec-compiler-part-3-mem.md` applies recursively if a
parallel group is nested inside a source-ordered loop). Each
chunk's per-`P` tail `%chunk_tail_P` is independent and is built
under the parallel-provenance override: the chunk loop applies
Section 6.3's structural plane (stream + carry on rwc + sentinel reset)
without building a per-`P` loop-carried state ring, since its
iterations remain logical iterations of the original
`scf.parallel`. The chunk's body memory accesses still chain
through their partition's frontier within a single iteration, and
each chunk's rendezvous of completed per-iteration tails feeds its
`%chunk_tail_P`. The compound's `outgoing_C_P = dataflow.sync`
over all `%chunk_tail_P` tokens, per
`docs/spec-compiler-part-3-mem.md` Section 2.6 chunk-tail rendezvous and
the parallel-provenance exception of
`docs/spec-compiler-part-3-mem.md` Section 4.3 and Section 5.6.

No loop-carried memory state is created at the parallel-provenance
compound boundary, per the parallel-provenance exception of
`docs/spec-compiler-part-3-mem.md` Section 4.3 and the no-state-ring rule
of `docs/spec-compiler-part-3-mem.md` Section 5.6: cross-iteration and
cross-chunk dependence edges inside the compound are suppressed by
the dependence builder, so the compound never builds a per-`P` ring.
Each generated chunk loop carries its own parallel-provenance
metadata, since its iterations are still logical iterations of the
original `scf.parallel`; per
`docs/spec-compiler-part-3-mem.md` Section 4.3 / Section 5.6 it therefore does
not build a per-`P` loop-carried state ring across its own
iterations. The Section 6.3 boundary translation supplies only the
chunk loop's structural plane (stream-driven rwc, sentinel reset,
iter_args for non-memory loop state); the chunk loop's memory
plane reduces to "no cross-iteration memory ordering inside this
loop". Memory accesses inside the chunk loop's body still chain
through their partition's frontier within a single iteration and
participate in the compound's `outgoing_C_P` rendezvous via the
chunk-tail token described above. The compound atom is marked
with parallel-provenance metadata
(`loom.parallel_group`, `loom.parallel_chunk`, `loom.parallel_chunks`)
per `docs/spec-compiler-part-3-mem.md` Section 4.3 so the chain construction
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
  continue to identify memory ops by their assigned deterministic
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

#### Index Switch Boundary Translation

This template instantiates the boundary translation contract of
`docs/spec-compiler-part-3-mem.md` Section 2.8 for `scf.index_switch`.

**Structural plane.** The compound's `struct_in` enters an `(N + 1)`
way `dataflow.demux` keyed on the normalized lane id `%lane` per the
structural template (lane 0 = default region, lane `i + 1` = case
region `i`). Each selected region is its own chain scope per
`docs/spec-compiler-part-3-mem.md` Section 2.2 and uses its lane's structural-
permission token as its `S.struct_at_*` source per
`docs/spec-compiler-part-3-mem.md` Section 6.2. The compound's `struct_done`
is `dataflow.mux` over all `(N + 1)` regions' `struct_done` tokens,
keyed on the same `%lane`. The same `(N + 1)` way `mux` shape
applies to every data result of the `scf.index_switch`.

**Memory plane (per touched partition `P`).** The compound's
`incoming_C_P` enters an `(N + 1)` way `dataflow.demux` keyed on
the same normalized `%lane`, projecting it into per-region tokens
`default_in_P`, `case0_in_P`, ..., `caseN_in_P`. Only the selected
region's projection fires, matching the dual-plane contract of
`docs/spec-compiler-part-3-mem.md` Section 2.8 (a raw SSA fork would risk
stranded memory tokens being buffered in unselected regions and
consumed on a later selected invocation). Each region chain scope's
`incoming_P` is its lane's projected token. Each region's per-`P`
tail is path-forwarding per
`docs/spec-compiler-part-3-mem.md` Section 2.7: a region that performs no
access in `P` forwards its lane projection unchanged; a region
that performs accesses in `P` builds its tail by the single-level
chain rule of `docs/spec-compiler-part-3-mem.md` Section 2.5 inside that
region. Call those `default_tail_P`, `case0_tail_P`, ...,
`caseN_tail_P`. The compound's `outgoing_C_P` is the
selector-matched `(N + 1)` way `dataflow.mux %lane` of these tails
(lane 0 = `default_tail_P`, lane `i + 1` = `case_i_tail_P`). Per
leaf rendezvous, `docs/spec-compiler-part-3-mem.md` Section 6.4 still
applies inside each region.

No loop-carried state. `scf.index_switch` does not introduce a
`dataflow.carry` either on the structural plane or on any per-`P`
plane.

### 6.7 `scf.execute_region`

* No control structure to flatten. The pass inlines the region body
  into the surrounding scope and rewires SSA values; ctrl/done
  forwarding follows program order.

#### Execute Region Boundary Translation

This template instantiates the boundary translation contract of
`docs/spec-compiler-part-3-mem.md` Section 2.8 for `scf.execute_region`.

**Structural plane.** Pass-through. `scf.execute_region` has a single
inner region with no control selector. The inner region's chain
scope inherits the compound's `struct_in` directly as its
`S.struct_at_*` source per `docs/spec-compiler-part-3-mem.md` Section 6.2,
and its `struct_done` directly becomes the compound's `struct_done`.
No `dataflow.demux` / `dataflow.mux` / `dataflow.carry` /
`dataflow.gate` is introduced by the boundary translation.

**Memory plane (per touched partition `P`).** Pass-through.
`incoming_C_P` directly enters the inner region as its `incoming_P`
per `docs/spec-compiler-part-3-mem.md` Section 2.4; the inner region's
`outgoing_P`, computed by the single-level chain rule of
`docs/spec-compiler-part-3-mem.md` Section 2.5 inside the region, directly
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
rules in Section 6 instantiate that model with op-specific structural and
memory-plane wiring.

## 8. Partitioned Data

Partitioned-data layout and in-thread queries are specified in
`docs/spec-compiler-part-4-partitioned-data.md`, along with the extension
point for neighborhood communication / distributed-buffer protocols. They are
not required for SCF-to-DFG flattening; this
document references them only at the boundary points (see Section 5.4 and
Section 9).

## 9. Verifier Rules (Front-End Specific)

In addition to the dataflow / fabric verifier set:

* `dataflow.thread` (definition, Section 5.4.1)
  - The op is a Symbol-bearing, function-like callable; it must
    be a direct child of a `ModuleOp` (`HasParent<"ModuleOp">`).
  - `sym_name` is required and module-unique among
    `dataflow.thread` definitions and other Symbol-bearing ops in
    the same module.
  - `sym_visibility` is required and must equal `"private"` under the
    baseline visibility policy. `"public"` and `"nested"` are rejected
    unless cross-module linkage is enabled by a separate spec.
  - `function_type` inputs are the user body operand types
    `(T0..TN)`; `function_type` results are empty regardless of the
    callable's grid shape.
  - `mapping` array length equals grid dim count.
  - Every `mapping` entry implements
    `DeviceMappingAttrInterface`.
  - No two `mapping` entries share the same `(kind, domain, axis)`
    triple. The verifier rejects, for example, two grid dims both
    labeled `#loom.thread_axis<parallel, 0, @D>` or both labeled
    `#loom.thread_axis<multiplexed, 2, @D>`. `kind` is `parallel`
    or `multiplexed`; `domain` is the optional explicit logical
    partition-domain symbol; and `axis` is the per-entry `i64`
    logical execution-axis identifier.
  - If a `#loom.thread_axis<...>` entry carries a domain qualifier,
    that symbol must resolve to a visible `dataflow.partition_domain`.
    Its `axis` must be in `[0, domain_rank)`. If the entry has no
    domain qualifier, Part 3 checks only that `axis` is non-negative.
  - Part 3 does not infer a domain qualifier from partitioned-data
    layouts. Partitioned-data query ops that need a domain require
    explicitly qualified `#loom.thread_axis<..., axis, @D>` entries;
    those rules live in
    `docs/spec-compiler-part-4-partitioned-data.md`.
  - Entry block argument count equals
    `numBodyOperands + 1 + gridDimCount`. The block-arg layout is
    `(args_*, thread_ctrl, iv_*)`: the first `N == numBodyOperands`
    block args mirror `function_type.inputs` exactly, then one
    `none`-typed `thread_ctrl` block arg, then one `index`-typed
    block arg per grid dim (in source-dim order). This ordering
    keeps the first `N` block args aligned with
    `function_type.inputs`, satisfying the upstream
    `FunctionOpInterface` invariant.
  - The body is `IsolatedFromAbove`: every SSA value used in the
    body and defined outside it is rejected.
  - Body must not contain a `dataflow.graph` definition (a graph
    definition is a sibling at module scope, not a body element).
    A `dataflow.graph.launch` is the only way to invoke a graph
    callable from inside a thread definition's body.
  - Body must not contain a `dataflow.thread` definition (thread
    definitions are also module-scope siblings). A
    `dataflow.thread.launch` is the only way to invoke a thread
    callable from inside another thread definition's body.
  - The body must follow strict thread layering. ScalarCore code
    (`scf.*` ops, ScalarCore-legal `func.call`, ScalarCore memory
    ops, `dataflow.thread.fence`, etc.) is always allowed in the
    thread definition's body. Direct launch shapes are constrained:
    - an innermost executable thread body may contain any number of
      direct `dataflow.graph.launch` ops interleaved with ScalarCore
      residual code, and no direct `dataflow.thread.launch`;
    - a non-innermost thread body may contain any number of direct
      `dataflow.thread.launch` ops interleaved with ScalarCore
      orchestration code, and no direct `dataflow.graph.launch`;
    - a ScalarCore-only body with neither launch shape is legal; by
      absence of direct child thread launches it is an innermost
      executable scalar-only AccCore binding candidate. This verifier
      rule does not itself select AccCore execution; placement must have
      selected the enclosing accelerator region first.
    Mixing direct graph launches with direct thread launches at the
    same thread-body placement level violates Section 3 Constitutional
    Rule 2's parent-side constraint.
  - Body may contain `func.call` only when the callee has been
    proven ScalarCore-legal or is scheduled for inlining before
    graph extraction. Body must not contain `func.func`
    definitions.
  - Reachability is a pass-pipeline invariant, not a verifier
    rule. The verifier accepts a `dataflow.thread` definition
    even when no `dataflow.thread.launch` references it (an
    unreferenced private symbol is dead code, not invalid IR);
    `loom-dead-symbol-prune` is the cleanup pass that removes
    such symbols before pipeline exit.

* `dataflow.thread.launch` (Section 5.4.2)
  - `callee` resolves to a `dataflow.thread` definition in the
    same module (verifier rejects unresolved or wrong-kind callee).
  - `bodyOperands` types equal `callee.function_type.inputs`
    position-by-position.
  - `dynamicGrid*` operand counts equal the count of
    `ShapedType::kDynamic` sentinels in
    `callee.staticGrid*`. Per-axis static / dynamic mixing
    follows the def's static-bounds pattern.
  - The op produces an `Optional<!dataflow.thread_token>` result.
    Under the baseline policy the result is always present.
  - Each memref-like operand in `bodyOperands` is the direct SSA
    result of a `dataflow.map_info` op in the launch's enclosing
    context. The launch's `MemoryEffectsOpInterface` walks back
    through that `map_info` op and projects effects on the map
    source according to its `direction` attribute (per Section 5.4.2
    contract).
  - **Direction / body-effect compatibility.** For each memref-
    like operand `i`, the projected `direction` must cover every
    effect that the callee body declares on its `i`-th
    function-signature block argument:
    - `direction = to` requires no writes through arg `i`.
    - `direction = from` requires no reads through arg `i`.
    - `direction = tofrom` accepts any combination.
    - `direction = alloc` requires the body's first effect on
      arg `i` to be allocation-style (write before read).
    - `direction = release` requires no further reads / writes
      after the release point.
    Violations are diagnosed at the launch op with a message
    naming both the launch and the offending body op.
  - The op declares a conservative effect on the
    `LoomAsyncResource` resource so generic CSE / DCE never
    removes a launch even when its callee body has no host-
    visible memory effects (per Section 3 Constitutional Rule 8).
  - May appear at host scope (`func.func` body) or inside a
    parent `dataflow.thread` definition's body (nested launch).
    Must not appear inside a `dataflow.graph` definition's body.

* `dataflow.thread.yield`
  - No operand allowed. The parent `dataflow.thread` definition
    has no data results; the per-launch completion token is
    produced by the launch op, not yielded as a body value.
  - Parent op must be a `dataflow.thread` definition (enforced by
    `ParentOneOf<["::dataflow::ThreadOp"]>`).

* `dataflow.thread.fence`
  - Must appear directly in a `dataflow.thread` definition's body,
    not at host scope and not inside a `dataflow.graph.launch` or a
    `dataflow.graph` definition's body.
  - Every operand has type `none` or `!dataflow.thread_token`.
  - The result has type `none`.
  - This op is the only explicit bridge from thread completion to
    graph-control `none` values. No verifier or canonicalizer may
    replace it with an implicit cast.
  - The op's `MemoryEffectOpInterface` implementation must report
    `MemRead + MemWrite` on MLIR's default memory resource. Lowering
    and verification must not weaken this to a per-resource effect or
    to no effect; doing so breaks the ScalarCore-side barrier
    contract specified in Section 3 rule 8.

* `dataflow.thread.wait`
  - At least one operand. Each is `!dataflow.thread_token` produced
    by a `dataflow.thread.launch`.
  - The op has no SSA result and therefore does not produce a
    graph-control `none` value. Use `dataflow.thread.fence` instead
    when child-thread completion must feed a graph launch's `ctrl_in`.
  - The op's `MemoryEffectOpInterface` implementation must report
    `MemRead + MemWrite` on MLIR's default memory resource. The wait
    has no SSA result, so this barrier is the only mechanism that
    keeps surrounding host or parent-context memory ops from being
    moved across it.

* `dataflow.map_info`
  - `direction` is one of the closed enum values.
  - `staticBounds`, if present, has length `2 * R` where `R` is
    the source memref rank, encoding `(lo_0, hi_0, ..., lo_{R-1},
    hi_{R-1})` in source-dim order; `dynamicBounds` length matches
    the count of `ShapedType::kDynamic` sentinels in `staticBounds`,
    in left-to-right iteration order. An omitted `staticBounds`
    means "the entire memref" and requires `dynamicBounds` to be
    empty.
  - The op may appear at host scope or inside another
    `dataflow.thread` definition's ScalarCore region; it must not
    appear inside a `dataflow.graph` definition's body.
  - The op's result must be used only as a `dataflow.thread.launch`
    body operand. Any other use -- passing the result to
    `memref.load`, `memref.subview`, `func.call`, another
    `dataflow.map_info`, or any op other than
    `dataflow.thread.launch` -- is rejected. This complements the
    `dataflow.thread.launch` rule that "each memref-like body
    operand must be the direct SSA result of a `dataflow.map_info`
    op": together the two rules close the loop on map_info
    provenance and keep the same-type passthrough memref from being
    treated as an ordinary memref by the rest of the IR.

Verifier rules for `dataflow.partition_layout`,
`dataflow.local_range`, `dataflow.thread_coord`, and
`dataflow.thread_linear_id` are specified in
`docs/spec-compiler-part-4-partitioned-data.md`.

* `dataflow.subgraph`
  - The op appears only inside a `dataflow.graph` definition body.
  - The op is `IsolatedFromAbove`; all external values enter through
    explicit operands and region block arguments.
  - Operand and result types are graph-compute scalar types or `none`
    control values. Memrefs do not cross a subgraph boundary.
  - The body contains only ops supported by the fabric-op support
    matrix plus `dataflow.yield`. Nested `dataflow.graph`,
    nested `dataflow.subgraph`, `dataflow.load`, and
    `dataflow.store` are rejected.
  - The op carries no hardware topology, PE identity, route,
    schedule slot, spatial / temporal mode, temporal tag, or
    resource-sharing attribute.

* `dataflow.graph` (definition, Section 5.5.1)
  - The op is a Symbol-bearing, function-like callable; it must
    be a direct child of a `ModuleOp` (`HasParent<"ModuleOp">`).
  - `sym_name` is required and module-unique among
    `dataflow.graph` definitions and other Symbol-bearing ops in
    the same module.
  - `sym_visibility` is required and must equal `"private"` in the
    baseline visibility policy. `"public"` and `"nested"` are rejected
    unless cross-module linkage is enabled by a separate spec.
  - `function_type` inputs are `(none, T0..TN)` where the leading
    `none` is the `ctrl_in` start port and the remaining types
    are the kernel's user-data inputs. `function_type` results are
    `(none, R0..RM)` where the leading `none` is the `done_out`
    completion port and the remaining types are the kernel's
    user-data results.
  - The graph definition's body is `IsolatedFromAbove`: every SSA
    value used in the body and defined outside it is rejected.
  - Entry block argument list mirrors `function_type.inputs`
    exactly: `(%ctrl_in : none, %arg_0 : T0, ..., %arg_N : TN)`.
  - The body's `dataflow.yield` terminator operand list mirrors
    `function_type.results` exactly:
    `(%done_out : none, %r_0 : R0, ..., %r_M : RM)`.
  - Body may contain `dataflow.{stream, carry, invariant, gate,
    mux, demux, sync, constant, load, store, yield}` plus ordinary
    pure ops permitted in the graph body whitelist.
  - Body must not contain `scf.*`, `func.func`, `func.call`,
    `dataflow.thread.launch`, `dataflow.graph.launch`,
    `dataflow.thread.fence`, `dataflow.map_info`, any partitioned-data
    op specified in `docs/spec-compiler-part-4-partitioned-data.md`,
    another `dataflow.graph` definition, or a `dataflow.thread`
    definition.
  - The op declares `RecursiveMemoryEffects` so module-scope
    walkers can observe per-callable effects without re-
    implementing the per-launch projection. The
    primary effect surface seen by the enclosing ScalarCore code
    is the `dataflow.graph.launch`'s manual projection (next
    bullet group).

* `dataflow.graph.launch` (Section 5.5.2)
  - `callee` resolves to a `dataflow.graph` definition in the
    same module (verifier rejects unresolved or wrong-kind callee).
  - `(none, type(bodyOperands)) == callee.function_type.inputs`
    position-by-position; the leading `none` slot is the
    per-launch `ctrl_in` operand.
  - `(none, type(results)) == callee.function_type.results`
    position-by-position; the leading `none` slot is the
    per-launch `done_out` result.
  - The op must appear inside a `dataflow.thread` definition's
    body, not at host scope and not inside another
    `dataflow.graph` definition's body.
  - The op implements `MemoryEffectOpInterface` directly: it
    resolves `callee`, walks the callee body, and reports the
    union of body op effects through the launch boundary. If
    callee resolution fails during partial IR construction, the
    launch reports conservative `MemRead + MemWrite` on MLIR's
    default memory resource. The upstream
    `RecursiveMemoryEffects` trait is **not** appropriate here
    because the launch references a sibling symbol, not a nested
    region.

* `Dataflow_YieldOp`
  - When the parent op is a `dataflow.graph` definition, the
    operand list must equal `function_type.results` exactly, with
    the leading `none` slot for `done_out` and user-data slots
    matching the def's user results in declaration order.
  - The verifier enforces that every yield operand has the
    matching `function_type.results[i]` type.

## 10. Non-Goals (First Milestone)

The following are explicitly out of scope for the scf-to-dfg
contract:

* Outlining `dataflow.thread` to a `fabric.module` symbol with
  a symbol reference. The thread op remains front-end software IR;
  fabric binding is a mapping and lowering concern. The thread op is
  already isolated and has an explicit boundary operand list.
* Native `dataflow.thread` data results, async value types, thread
  groups, and thread-level aggregation regions. Tensor-result
  aggregation is handled by materializing it into mapped-memory
  effects before thread promotion.
* LLVM IR provider integration, source-language integration, and clang
  embedding. Those concerns belong to Part 1 and Part 2.
* Optimization of `dataflow.map_info` direction. Default `tofrom`.
* Strong-typed partitioned-data carriers, logical-domain-point to
  fabric-resource binding, and neighborhood communication /
  distributed-buffer protocol for tile-and-domain memrefs. These are
  not part of this contract. In particular, this spec does not commit
  to any stencil-specific op signature for neighbor exchange, nor to a
  default mapping from a
  `dataflow.partition_domain @D` point to any `fabric.pe` /
  `fabric.mem` instance.

## 11. References

* `docs/spec-fabric-module.md`, `docs/spec-fabric-pe.md`,
  `docs/spec-fabric-fu.md` -- the fabric-side IR that the front-end
  output eventually targets.
* `docs/spec-compiler-part-1-source.md` -- high-level source
  integration and metadata emission.
* `docs/spec-compiler-part-2-scf.md` -- LLVM-to-SCF raising,
  accelerator-region selection, and `loom.acc_region`.
* `docs/spec-compiler-part-3-impl.md` -- pass pipeline, lit-test
  layout, acceptance checklist, and maintenance plan
  for the SCF-to-DFG front-end.
* `docs/spec-compiler-part-3-mem.md` -- compositional chain model,
  alias oracle, dependence builder, loop-carried memory state, and
  token-wiring rules used inside each `dataflow.graph`. Per-`scf.*`
  boundary translation rules in Section 6 of this document instantiate
  that model.
* `docs/spec-compiler-part-3-placement-framework.md` -- common
  placement-partition framework; Part 3 owns the L2 graph-placement
  instance.
* `docs/spec-compiler-part-4-partitioned-data.md` -- partitioned-data
  annotation, in-thread queries, and the extension point for neighborhood
  communication / distributed-buffer protocols.
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
