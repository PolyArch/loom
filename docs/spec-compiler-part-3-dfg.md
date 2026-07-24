# Loom Compiler Part 3: SCF to DFG

This document specifies the third compiler part of the Loom front-end:
mechanically lowering selected SCF-stage accelerator regions into the initial
Canonical Dataflow Program, Loom's final target-independent software IR.

Translation and recovery through the initial SCF-stage IR are mechanical.
SCF-to-SCF transformation is the primary compiler optimization and DSE domain;
its selected immutable Structured Program Candidate must already materialize
the schedule, parallel, vector, reduction, memory-overlap, AccCore ownership,
and SpatialCore ownership decisions consumed here. Part 3 does not choose or
repair those decisions.

The target Part 3 dataflow surface uses module-scope, Symbol-bearing,
function-like definitions for both `dataflow.thread` and
`dataflow.graph`. Execution is materialized only by
`dataflow.thread.launch` and `dataflow.graph.launch`. Graph control
ports are explicit in the current graph ABI: `ctrl_in` and launch-facing
`done_out` are invocation protocol endpoints represented at every launch
site, not application payload slots in the `dataflow.graph` function type.
The graph body does not return `done_out`; its structural
`dataflow.graph.return.complete` frontier is the unique authority from which
the launch result is derived. Part 3 consumes each explicit
`loom.spatial_region` inside its owning `dataflow.thread` and publishes the
corresponding graph definition and launch only after complete conversion and
native finalization succeed.
The precise timing semantics of `dataflow.stream`, `dataflow.carry`,
`dataflow.invariant`, and `dataflow.gate` are specified separately in
`docs/spec-dataflow-part-1-streaming.md`. The precise firing semantics
of `dataflow.constant`, `dataflow.sync`, `dataflow.mux`, and
`dataflow.demux` are specified separately in
`docs/spec-dataflow-part-2-control.md`.

The initial Canonical Dataflow Program may subsequently participate in its own
typed, semantics-preserving Dataflow optimization lineage. Such a rewrite
produces another immutable Canonical Dataflow Program and may use pure
Dataflow or optional Fabric-aware Evaluation to rank candidates. It must not
reselect a structured schedule or ownership boundary, and Fabric facts never
become Canonical Dataflow semantics.

This document owns only the target contract: IR boundaries, structured-control
flattening, memory-dependence integration, and verifier invariants. Pass
decomposition, test layout, and maintenance sequencing are implementation
choices and are not a second tracked specification.

Fabric realization and actor grouping are TechMapping concerns. Part 3
performs only structural eligibility and canonical graph publication inside
an already established `dataflow.thread`; it does not assign a target or
retain target-specific grouping in program IR. Absence of a realization on a
particular Fabric is a Mapping result, not a graph-validity failure.

## 1. Scope and Contract

The compiler front-end is documented in four parts:

* **Part 1, source integration.** LLVM IR plus optional typed Loom hints is the
  source-facing compiler contract. Any high-level language provider may
  participate by emitting valid LLVM IR; embedded clang for C / C++ is the
  first limited provider. Missing hints reduce available guidance but do not
  make otherwise valid provider input illegal.
* **Part 2, LLVM to initial SCF.** LLVM/CFG-shaped input is mechanically
  raised and normalized into mixed-dialect initial SCF-stage MLIR. Imported
  LLVM callable envelopes and any operation without an exact standard
  equivalent remain LLVM dialect. The part recovers structured execution and
  preserves analysis inputs without selecting a QoR-distinct schedule, vector
  form, reduction strategy, or ownership boundary.
* **Part 3, SCF to DFG.** This document. It consumes explicit
  `loom.spatial_region` candidates inside the selected Structured Program
  Candidate's `dataflow.thread` definitions and mechanically publishes
  `dataflow.graph` definitions plus `dataflow.graph.launch` ops at those
  candidate sites.
* **Part 4, logical domains and data views.** The canonical dense-coordinate
  and dynamic-work domain ABI, work-item termination, source-IV
  reconstruction, and ordinary value or memory views derived from those
  domains (see
  `docs/spec-compiler-part-4-partitioned-data.md`).

Between Parts 2 and 3, SCF optimization and DSE produce the selected
Structured Program Candidate. That domain owns all performance-distinct
structured choices. Part 3 begins only after those choices and their typed
ownership carriers are explicit.

Input to graph extraction is an MLIR module containing module-scope
`dataflow.thread` definitions. Every selected SpatialCore candidate is already
materialized as a `loom.spatial_region` inside exactly one thread. Other thread
body code remains InstructionCore-resident, including SCF-shaped code outside
an explicit spatial boundary. Imported Host or InstructionCore code remains in
its `llvm.func` envelope; genuinely standard-MLIR-native `func.func` callables
may coexist in the module. Either callable is ownership-neutral and does not
authorize graph creation through its signature, body shape, memory effects, or
return convention.

Output is an initial Canonical Dataflow Program: module-level `llvm.func`
symbols for imported LLVM callables, any genuinely native `func.func` helpers,
module-level
`dataflow.thread` definitions reached by zero or more
`dataflow.thread.launch` ops; and module-level `dataflow.graph`
definitions reached by zero or more `dataflow.graph.launch` ops
inside thread definitions. No `scf.*` op is left inside any
`dataflow.graph` definition's body after successful graph-region lowering.
The recursive lowering contract accepts arbitrary nesting of
`scf.if`, source-sequential `scf.for`, `scf.while`, and fixed-width
graph-owned `scf.parallel` or effect-form `scf.forall`. A graph-owned parallel
op must have a compile-time fixed domain, and all facts needed to establish
ownership, width, and cross-lane legality must be present in the current
Structured Program Candidate's semantic IR and resolved lowering config.
The lowerer re-proves those facts; lineage, cached analyses, and external
provenance cannot make an otherwise invalid candidate legal. Dynamic-width,
resource-mapped, and result or reduction forms fail before any graph is
mutated; the graph owner does not infer ownership, serialization, unrolling,
or reduction order. The
`dataflow.thread.launch` op carries the completion token and
mapped-memory data transfer; the def remains a callable kernel
body, not a tensor-result returning op. Memory dependence
construction runs in the recursive graph owner using basic graph-local alias
roots and per-partition write/read frontiers (see
`docs/spec-compiler-part-3-mem.md`).
The Structured Transfer Algebra defines graph-owned parallel composition only
after the Structured Program Candidate has materialized its P[] ownership and
schedule form in semantic SCF. That fixed-domain SCF is the transient input
representation for mechanical lowering. It is recursively replicated into
static lanes and removed; no parallel control op or schedule record survives
in canonical graph IR.
Graph candidate eligibility and atomic publication are governed by this
document. TechMapping, SpatialMapping, and SystemMapping realization are
outside this IR.

## 2. Execution Ownership Model

Loom's execution target is a heterogeneous system containing HostCore
execution and one or more AccCore execution resources. Fabric owns each typed
AccCore occurrence, its node-local InstructionCore description, its
SpatialCore occurrence, and the typed attachment to an exact `fabric.module`
template. `fabric.module` remains the SpatialCore or CGRA template only; it
does not own the physical AccCore occurrence or InstructionCore description.
The typed system ownership contract is specified by
`docs/spec-fabric-system-adg.md`.

The front-end IR in this document remains a software and logical
execution model. SystemMapping binds logical execution cells to physical
AccCore instances. TechMapping and SpatialMapping realize canonical graph
execution on the selected SpatialCore resources.

The front-end IR separates these execution roles:

| Execution role | Front-end IR carrier |
|----------------|----------------------|
| HostCore | Host-call-context callable body code outside any `dataflow.thread.launch`; imported LLVM callables remain `llvm.func` |
| Logical execution domain | A `dataflow.thread` definition (Symbol-bearing, module-scope) plus each caller-side `dataflow.thread.launch`. A dense instance is identified by its coordinate tuple; a dynamic-work instance is identified by its `WorkItemId`. SystemMapping binds either logical identity to an AccCore. |
| InstructionCore | The body of a `dataflow.thread` definition, minus its `dataflow.graph.launch` ops, plus InstructionCore-legal `llvm.call` or `func.call` callees after inlining or specialization. The body is "what one logical execution cell runs once binding maps it to a physical AccCore". |
| SpatialCore | Each `dataflow.graph` definition referenced by a `dataflow.graph.launch` inside a `dataflow.thread` definition's body, again per bound logical execution cell. |

A single `dataflow.thread.launch` starts exactly one domain instance of the
kind declared by the referenced `dataflow.thread`: either a multidimensional,
zero-based dense domain or a responsibility-tracked dynamic work domain. The
thread body is "what one logical execution point runs"; dense coordinates or
the current work-item identity distinguish executions. The domain is a
software concept and does not commit to a specific fabric topology.
A fabric whose physical PE / memory graph is not a Cartesian mesh
is supported by the same Mapping profiles. The binding from a
logical execution point to a physical AccCore is a SystemMapping concern;
see `docs/spec-mapping-artifact.md` and `docs/spec-pnr.md` for SystemMapping
execution binding and PnR.

Every `dataflow.thread` body may contain InstructionCore code and
`dataflow.graph.launch` ops, but it cannot launch another thread. A dynamic
worker may use `dataflow.work.spawn` to publish a child in its current domain;
that operation does not create or target another thread launch.
Dynamic instances become physical AccCore execution slots only through
SystemMapping.

An InstructionCore-only thread body is legal. Failure to form a canonical graph
must not synthesize a new accelerator boundary or move unselected host code
into a thread.

Thread completion and graph/dataflow control are distinct token domains.
`!dataflow.thread_token` is the inter-thread asynchronous completion
token produced by `dataflow.thread.launch`. `none` values are the
graph-control, graph-completion, streaming-control, and memory-order
tokens used inside dataflow. There is no implicit cast or general
conversion between the two domains. `dataflow.thread.wait` consumes
one or more `!dataflow.thread_token` values for caller-side causal
synchronization and emits no SSA value or graph-control value. It is
not a memory barrier.

Thread hierarchy transforms before SystemMapping are legal only as explicit
optimization policies. They may reorder independent thread levels,
collapse adjacent independent levels, or tile and split a level when the
transform preserves the logical instance set, each instance's scalar
values, memory-order constraints, and thread-completion causal order.
Launch placement remains caller-side only. The deterministic baseline
policy performs only annotation and canonicalization; it must not
silently change hierarchy shape as a verifier or parsing side effect.

### 2.1 IR Carrier Responsibilities

* `llvm.func` remains the sole function and ABI owner for a callable imported
  from the final linked LLVM module. `func.func` is used only for a genuinely
  standard-MLIR-native callable or helper; it cannot mirror the LLVM ABI.
  Neither callable kind chooses HostCore or AccCore ownership. Call-context
  classification decides where calls are legal.
* `loom.spatial_region` is temporary compiler IR inside a
  `dataflow.thread`. It owns one structured graph candidate with normalized
  value, stream-channel, and memory boundary segments. It never appears in a
  finalized Canonical Dataflow Program.
* `dataflow.thread` is the logical accelerator execution-domain
  **definition** (Symbol-bearing, module-scope, function-like). It owns the
  kernel body and one domain kind. A dense definition owns coordinate rank
  through its canonical entry-block shape; a dynamic definition designates
  one ordinary argument as its work-item payload.
  It does not itself execute; dynamic logical instances are
  materialized by one or more `dataflow.thread.launch` ops at use
  sites, then SystemMapping decides which instances occupy physical AccCore
  slots.
* `dataflow.thread.launch` is the logical accelerator execution
  boundary. It references a `dataflow.thread` callable by symbol, supplies
  async dependencies and ordinary body operands, plus one non-negative extent
  per dense coordinate dimension. A dynamic launch instead supplies one root
  work item and no extents. Both produce one collective completion token.
* `dataflow.work.spawn` publishes one child of the currently executing dynamic
  work item after atomically acquiring its termination responsibility. It is
  illegal in a dense thread or any graph and is not nested thread launch.
* `dataflow.graph` is the SpatialCore leaf DFG **definition**
  (Symbol-bearing, module-scope, function-like). Its body cannot
  contain callable definitions, `llvm.call`, `func.call`,
  `dataflow.thread.launch`, `dataflow.graph.launch`, or another
  `dataflow.graph` definition.
  It is final target-independent software IR: its validity does not assert
  that any current Fabric can realize it. TechMapping owns that decision.
* `dataflow.graph.launch` is the SpatialCore execution boundary
  inside a `dataflow.thread` definition's body. It references a
  `dataflow.graph` callable by symbol, supplies dependency events, value
  inputs, stream channel bindings, and memory imports, and yields value
  outputs, memory exports, and a trailing `done : none` result.

Function definitions remain module-level symbols in this design.
`dataflow.thread` definitions are also module-level symbols (and
not symbol tables themselves) and do not physically contain
`llvm.func` or `func.func` definitions. An `llvm.call` or `func.call` inside a
`dataflow.thread` definition's body is an InstructionCore call. If the callee
contains code that must become a `dataflow.graph` definition, Part 3 must
inline or specialize that callee into the active thread definition
before graph extraction. A `dataflow.thread.launch` is invalid
transitively inside every thread or graph definition. Non-inlined
InstructionCore calls may remain only when their callee body is graph-free
after this preparation.

## 3. Constitutional Rules

The eight rules below are invariants that downstream passes and
verifiers must enforce; the rest of this spec is a refinement of how
each rule lands in IR.

1. `dataflow.thread` is the logical execution-domain
   primitive used for selected accelerator work. It is a
   Symbol-bearing, module-scope, function-like definition (Part 3
   Section 5.4.1); logical instances are materialized by
   `dataflow.thread.launch` and, for one dynamic domain only, its controlled
   `dataflow.work.spawn` operations. Launches appear only in host/runtime
   orchestration outside every thread or graph definition. An instance becomes
   a physical AccCore execution slot only through SystemMapping.
   The thread definition's body has a `thread_ctrl : none` block
   argument that fires once the logical thread instance starts executing
   (dense entry-block layout: `(args_*, thread_ctrl, coord_*)`; dynamic work has
   no coordinate suffix, see Section 5.4.1).
   The body may contain InstructionCore operations and InstructionCore-legal
   `llvm.call` or `func.call` operations, but not callable definitions or
   `dataflow.thread.launch` ops.
2. `dataflow.graph` is a leaf-level definition. It is also a Symbol-
   bearing, module-scope, function-like definition (Part 3 Section 5.5);
   execution is materialized by `dataflow.graph.launch` ops inside a
   thread definition's body. Its body must not contain any
   `llvm.func`, `func.func`, `llvm.call`, `func.call`,
   `dataflow.thread.launch`,
   `dataflow.graph.launch`, or another `dataflow.graph` definition.
   The graph body is a single graph-kind region; it already permits
   feedback edges (accepted semantics). A thread body may contain
   InstructionCore code and `dataflow.graph.launch` ops, but it
   never launches another thread. The verifier enforces this launch
   containment transitively (see Section 9).
3. Every `dataflow.graph` definition has an explicit `%start : none`
   entry value and a structural `dataflow.graph.return` with four
   segments: `values`, `streams`, `memories`, and `complete`.
   `complete` is a mandatory non-empty variadic unordered all-of set of
   `none` values. A no-work graph may return `%start` as its sole
   completion witness; real work, including zero-output work, must expose a
   causally derived frontier. The launch-facing result is exactly
   `done_out = all_of(graph.return.complete)`. It never appears among the
   return operands, and no effect scan or graph-quiescence rule can replace
   the explicit frontier.
4. The HostCore-to-AccCore data plane is the explicit ordinary-operand segment
   of `dataflow.thread.launch`. Values, memrefs, dynamic source lower bounds,
   source steps, and other launch parameters cross directly as typed SSA
   operands and matching thread block arguments. Dedicated `map_info`,
   partition-domain, or layout carrier ops are not part of this ABI.
5. Graph-local memory ordering is constructed in the front-end by one
   recursive graph-region owner. It discovers basic root alias partitions,
   threads independent write and read frontiers through sequential and
   structured control, and emits ordinary Dataflow event edges. Unknown
   accesses conservatively cover every known partition. There is no
   persistent alias oracle, dependence snapshot, or later wiring pass. The
   complete transfer rules are specified in
   `docs/spec-compiler-part-3-mem.md`.
6. `loom.spatial_region` is the temporary publication boundary inside an
   existing `dataflow.thread`. Its operands are normalized as value inputs,
   stream input channels, memory inputs, and stream output channels; its
   results are value outputs followed by memory outputs. Each stream input
   has one affine `source_map` from the consumer thread domain to the producer
   thread domain. The lowering collects all explicit candidates before
   attempting publication, performs conversion and native validation on a
   scratch module, and replaces the live module only on success. A public pass
   failure therefore leaves temporary candidates and never exposes a partial
   canonical graph. Current publication supports nested `scf.if` completion
   propagation. Stream channel segments become payload-typed graph stream
   ports plus launch-site channel bindings; receive/send sites rendezvous with
   the recursively lowered execution frontier and are removed. Input
   `source_map` attributes are preserved exactly, and channel handles never
   enter the canonical graph body. One binding denotes one ordered dynamic
   event sequence. A fixed structured scope may contain multiple sequential
   or structured mutually exclusive sites. Lowering emits one fixed ordinal
   schedule, filters inactive branch sites, demuxes each input from the
   filtered ordinal, and muxes outputs back into that same dynamic order.
   Branches may have unequal or empty site sets, and a later branch selector
   may depend on an earlier input event. Enclosing loops repeatedly activate
   the same schedule, so one or several static body sites may each fire
   dynamically. Across repeated thread launches, each endpoint binding and
   logical point concatenates these per-instance sequences in deterministic
   launch issue order. Channel delivery pairs the resulting producer and
   consumer sequences by message ordinal after applying `source_map`; it does
   not pair thread activations or create activation-owned segments. Endpoint
   sites nested under `scf.parallel` or `scf.forall` have no inferred
   traversal order and fail before publication. Unselected or non-fixed
   graph-owned parallel forms also fail closed.
7. `dataflow.thread` and `dataflow.graph` definitions are both
   `IsolatedFromAbove`. No operation inside either definition's body
   may directly use an SSA value defined in the surrounding scope.
   Every boundary value must appear as an explicit launch op
   operand and as a matching entry block argument of the
   referenced definition. For a `dataflow.thread.launch`, the
   operand list is the HostCore-to-AccCore launch ABI: every value crosses as
   an ordinary typed operand. For a `dataflow.graph.launch`,
   operands and results are the explicit SpatialCore data/control
   ports. A `dataflow.thread.launch` completion token expresses
   launch-retirement causality only. The
   `dataflow.graph.launch` op resolves its callee through
   `SymbolUserOpInterface`; it does not implement
   `MemoryEffectsOpInterface` or project sibling-callee effects.
   Each definition carries `RecursiveMemoryEffects` so walkers of
   that callable can observe its body effects. Launch-site ordering
   and retirement are represented by explicit dependencies, the
   segmented memory-capability ABI, the `done` result, and native
   finalized-program validation.
8. Effect visibility contract. Every front-end op whose execution
   affects memory state must declare its effects through MLIR's
   `MemoryEffectOpInterface` (or an equivalent recursive trait)
   accurately enough that generic optimizers preserve the intended
   observable behavior. Causal ordering and completion are defined
   by their individual op contracts, not by memory effects. The
   baseline policy uses MLIR's default-resource barrier pattern --
   broad, conservative `MemRead + MemWrite` declarations -- where a
   precise per-resource binding would require op-side machinery
   outside this contract.
   Tighter per-resource bindings (for example, load/store keyed on
   the `$mem` operand) are explicit extensions.

## 4. Glossary

* **HostCore.** The general-purpose CPU that runs host-call-context callable
  code outside any `dataflow.thread.launch`. Imported LLVM code remains in its
  `llvm.func` envelope.
* **AccCore.** One typed physical accelerator execution resource owned by
  `fabric.system`, containing a node-local InstructionCore description and a
  SpatialCore occurrence attached to an exact `fabric.module` template. Part
  3 does not create physical AccCore occurrences; it creates logical
  accelerator work that SystemMapping later binds to AccCore resources.
* **InstructionCore-callable function.** A module-level `llvm.func` or native
  `func.func` that Part 2 classified as legal to call from code running inside
  a `dataflow.thread` definition's body. Such a function remains a symbol;
  Part 3 either preserves calls to it as InstructionCore calls or inlines or
  specializes it before graph extraction.
* **Logical thread coordinate.** One zero-based `index` value in the trailing
  coordinate segment of a `dataflow.thread` entry block. A rank-`K` launch
  provides a coordinate tuple in the Cartesian domain
  `[0, extent_0) x ... x [0, extent_K)`. Coordinates carry no physical
  topology or execution-order promise.
* **Logical thread domain.** The instance set derived from one launch and its
  callee's closed domain kind. A dense domain uses the extent vector; a
  `DynamicWork` domain starts with one root and grows through registered child
  work. SystemMapping, rather than a software mapping attribute, decides how
  either kind shares or occupies AccCore resources.
* **Derived data view.** An ordinary typed value or memref view computed from
  launch operands and logical coordinates. Source-IV reconstruction, tiling,
  local ranges, and explicit linearization remain ordinary candidate semantics
  rather than a second partition-domain ABI.
* **Thread token.** A value of type `!dataflow.thread_token`, a
  one-shot completion signal modelled on `!async.token`. It belongs to
  the inter-thread asynchronous-completion domain, not to the
  `none`-typed graph/control token domain.
* **Thread control token.** A `none`-typed entry-block argument of
  a `dataflow.thread` definition's body (named `thread_ctrl`,
  positioned after the function-signature args per Section 5.4.1). It is
  the per-instance AccCore start signal used to launch root
  `dataflow.graph.launch` ops.
* **Basic alias partition.** A graph-local compiler analysis bucket keyed by
  a recognized memory root. View-like values are peeled to their root;
  graph boundary arguments are conservatively grouped unless explicit
  no-alias evidence distinguishes them; fresh allocations have distinct
  roots; globals and raw pointer bases must be imported explicitly before a
  graph is finalized. Partition identity is not written into IR.
* **Memory dependence edge.** An ordinary `none` SSA causal edge emitted by
  the recursive graph owner from the current per-partition frontier. No
  persistent dependence snapshot is retained.
* **Loop-carried memory state.** The canonical
  `(write_frontier, read_frontier)` pair carried recursively for one
  alias partition. Touched components are materialized with independent
  `dataflow.carry` and false/true `dataflow.demux` projections. Specified in
  `docs/spec-compiler-part-3-mem.md`.
* **Phase bit.** A loop-control bit produced by `dataflow.stream` for
  counted loops: it fires `true` once per body iteration and one
  trailing `false` token that closes the activation. The combined
  `(true, ..., true, false)` stream phases structural state and may
  select the body and exit projections of loop-carried memory state, but is
  not itself a memory frontier. The exact timing semantics live
  in `docs/spec-dataflow-part-1-streaming.md`.
* **Streaming token.** Any typed token stream consumed or produced by the
  streaming primitives `dataflow.stream`, `dataflow.gate`,
  `dataflow.invariant`, and `dataflow.carry`. Payloads may be ordinary data,
  `i1` phase, or `none` control according to the operation contract. Streaming
  tokens carry phase, iteration, or payload information rather than
  memory-frontier authority; their precise timing semantics are owned by
  `docs/spec-dataflow-part-1-streaming.md`. The phase bit above is one specific
  streaming token.
* **Memory-order token.** A `none`-typed token used to encode one
  component or join of
  alias-aware ordering between memory accesses inside a
  `dataflow.graph` definition's body. Each per-partition state pair
  (see `Canonical Frontier State` in
  `docs/spec-compiler-part-3-mem.md`) flows through its own
  memory-order tokens; the `One Recursive Owner` transfer in that document combines a
  structural permission token with a
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

This spec introduces no other types. Host-to-AccCore values cross the launch
boundary as ordinary typed SSA operands; no wrapper or provenance-only type is
introduced.

### 5.2 Attributes And Interface Instances

No Loom-specific thread mapping attribute is introduced. Coordinate rank and
domain come from the definition body shape and launch extents. Parallel versus
temporal AccCore use is a SystemMapping relation plus event-relative
`ResourceUse`, not a property copied into the program ABI.

#### Canonical Actor Schema Projection

Every operation admitted as a Canonical Dataflow actor resolves to exactly one
registered `OperationSchemaId`. The operation's native definition remains the
owner of its source semantics. A typed
`CanonicalDataflowActorOpInterface`, implemented directly or through an
external model, projects only the Loom contract needed downstream:

```text
CanonicalActorSchemaProjection {
  operation_schema_id
  actor_kind
  closed_semantic_attribute_projection
  instance_verifier
  transition_descriptor_identity
}
```

This notation describes one registered typed projection, not an IR operation,
attribute, Artifact, or second semantic language. The projection must classify
every property and attribute that can affect one firing. Unknown or
unclassified actor state is rejected; consumers never copy an arbitrary
attribute dictionary. A field may be excluded only when its owning spec proves
it nonsemantic, as for source provenance.

Graph admission, canonical relation construction, Configured Function
materialization, simulator dispatch, and Fabric capability matching all consume
the same `OperationSchemaId` and projection. The canonical actor classifier is
a derived query over this registry, not another whitelist. Simulator providers
own executable transition implementations, while Hardware Sharing Groups own
only genuine physical sharing relations. Neither may redefine software
semantics or maintain a competing operation-name table.

Canonical actor values distinguish defined, poison, and undef state. A defined
state carries the exact type-appropriate bits or logical identity; fixed
vectors may carry state independently per lane. The owning operation schema
defines propagation, masking, non-observation, freezing, and undefined
behavior. In particular, selection does not observe its unselected value,
inactive masked-memory lanes do not observe address or data, active stores may
store poison and loads restore it, and graph outputs may carry poison or undef.
There is no global rule that a terminal exceptional value is an execution
error. Ordinary LLVM pointers remain outside the canonical graph surface
unless a future registered actor contract explicitly admits them.

Finalized Canonical Dataflow Programs use one derived identity attribute:

```text
#dataflow.entity_id<42>
```

The payload is an unsigned 64-bit value; `42` above is illustrative. The
Dataflow finalizer is the sole producer of this attribute. It appears as
the namespaced `dataflow.entity_id` attribute on entity-bearing operations and
inside the existing function-like argument dictionary for an imported logical
memory root. It is never an authoring input, Mapping annotation, provenance
handle, execution occurrence, or target binding. The canonical-labeling
contract in `Canonical Artifact Finalization And Entity Identity` defines its
closed carrier set and validation.

### 5.3 Thread Completion

No separate operation interface is introduced for thread completion.
The launch, wait, and yield contracts are specified directly below.

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
  Dataflow_ThreadDomainAttr:$domain,
  OptionalAttr<DictArrayAttr>:$arg_attrs;
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
  callable signature. Asynchronous execution is expressed by launch
  dependencies and the mandatory launch completion token, not by the
  function type.
* `sym_name` is required and module-unique. `sym_visibility` is
  required and must equal `"private"` under the baseline visibility
  policy. The verifier rejects `"public"` and `"nested"` unless
  cross-module linkage is enabled by a separate spec.
* `domain` is the closed `DenseRectangular` or
  `DynamicWork { work_item_arg_ordinal }` value owned by Part 4. A dynamic
  definition has coordinate rank zero and its ordinal must select exactly one
  `function_type` input. A dense definition has no work-item ordinal.
* A dense entry block has the layout `(args_*, thread_ctrl, coord_*)`; a
  dynamic entry block has `(args_*, thread_ctrl)`:
  - The first `N` block arguments mirror `function_type.inputs`
    exactly (each user body operand). Putting the signature args
    first preserves the upstream `FunctionOpInterface` invariant
    that the entry block's first `N` arguments correspond to
    `function_type.inputs[0..N]`. This matches the `gpu.func`
    precedent of "function args first, implicit extras after".
  - `thread_ctrl : none` is the per-launch AccCore start signal.
    It is produced by the launch op once async dependencies are
    satisfied and the AccCore instance begins execution. Root
    `dataflow.graph.launch` ops with no InstructionCore predecessor use
    this value as their `ctrl_in` operand.
  - For a dense definition, `coord_0, ..., coord_{K-1} : index` are the
    per-instance logical
    coordinates, one per launch-domain dimension, in source-dimension order.
    Their count is the definition's coordinate rank. Rank is derived from
    this canonical suffix after the `function_type` inputs and unique
    `thread_ctrl`; there is no duplicate rank, grid, or mapping attribute.
  - For a dynamic definition, the designated ordinary argument carries the
    current work-item payload. The runtime `WorkItemId` is execution identity,
    not an additional SSA argument or payload wrapper.
* `arg_attrs` is indexed only by the `args_*` payload prefix. Forall
  promotion copies each captured source function argument dictionary,
  including arbitrary attributes such as `llvm.noalias`, into capture order.
  Locally defined captures have an empty dictionary. `thread_ctrl` and
  `coord_*` are not payload arguments and never inherit source argument
  metadata.
* The body is `IsolatedFromAbove`. No SSA value defined outside
  the def's body may be used inside it; the launch's body operands
  are the only inputs.
#### 5.4.2 `dataflow.thread.launch`

```
arguments:
  Variadic<Dataflow_ThreadToken>:$asyncDependencies,
  Variadic<Index>:$extents,
  Variadic<AnyType>:$bodyOperands,
  FlatSymbolRefAttr:$callee;
results:
  Dataflow_ThreadToken:$asyncToken;
traits:
  AttrSizedOperandSegments,
  DeclareOpInterfaceMethods<SymbolUserOpInterface>.
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
and resolves its callee through the explicit `callee` attribute.

* `callee` is a flat symbol reference that must resolve to a
  `dataflow.thread` definition in the same module. The verifier
  rejects launches whose `callee` cannot be resolved or whose
  resolved op is not a `dataflow.thread`.
* `bodyOperands` types must equal `callee.function_type.inputs`
  position-by-position. Memrefs, values, source lower bounds, source steps,
  and other launch parameters all use this same ordinary operand segment.
* `extents` contains exactly one `index` value per dense callee coordinate.
  Each extent must be non-negative. A dense rank-zero launch has no extent
  operands and creates one instance. If any dense extent is zero, the launch
  creates no instances and its collective token retires after its dependencies.
  Static verification rejects a provably negative extent; runtime admission
  rejects a negative value before creating any instance. A dynamic-work callee
  also has no extents, but creates one root item from its designated ordinary
  body operand; the callee's domain kind distinguishes it from dense rank zero.
* Each dense instance receives a zero-based coordinate tuple. Source lower
  bounds and steps, when needed, are ordinary operands and the thread body
  reconstructs `source_iv = lower + coordinate * step`. The ABI specifies no
  row-major linearization, issue order, physical grid, or topology.
* `asyncDependencies` is the variadic prefix of incoming
  `!dataflow.thread_token` dependencies. They form an all-of start
  ordering. The op always produces exactly one
  `!dataflow.thread_token` `asyncToken` result for collective
  retirement of all logical instances. For a dynamic-work launch this means
  the root source is closed and the active responsibility set owned by Part 4
  is empty; queue emptiness alone is insufficient.
* The op has no data results. Its mandatory token is the only
  launch-level completion result.
#### 5.4.3 `dataflow.thread.yield`

```
arguments:
  Variadic<NoneType>:$completionFrontier;
results:
  none;
regions:
  none;
traits:
  Terminator,
  ParentOneOf<["::dataflow::ThreadOp"]>.
```

* `completionFrontier` is a variadic unordered all-of frontier of
  `none` values. Structural verification checks only that each operand
  has type `none` and that the op terminates a `dataflow.thread` body.
  Finalized-program validation additionally requires the frontier to be a
  duplicate-free minimal terminal antichain for every
  `dataflow.graph.launch` in the thread body. Each launch's mandatory
  `done : none` must be yielded directly or lie in the path-aware causal
  closure of a yielded terminal event. Every executable SCF predecessor on
  which the launch exists must forward or cover its completion; a fallback is
  valid only on a mutually exclusive path where that branch-local launch does
  not exist. A `dataflow.mux` similarly covers a completion only on every
  selector lane where the completion may exist. Matching `dataflow.demux`
  activation can prove a launch absent from the other lanes.

  Non-region causal edges come only from explicit completion semantics:
  `dataflow.graph.launch` done follows its dependencies, and `dataflow.sync`
  waits for all inputs. Other operations do not acquire completion semantics
  merely by accepting a `none` operand. Two distinct yielded events are
  invalid when either causally covers the other. Each remaining frontier
  member must also be necessary for at least one graph-launch completion that
  no other member covers. Independent launch events must all be covered, while
  a chain yields only its terminal event. A thread with no graph launch has an
  empty frontier. These checks derive from SSA and structured-control
  causality, never from textual operation order, and do not infer additional
  completion obligations from effects, DMA, or other operations.
  Any supported tensor-result aggregation has already been materialized by
  Part 2 as accepted explicit effects; an unmaterialized form is
  non-finalizable. The frontier therefore carries no thread data result.

  In a `DynamicWork` thread, completion of the frontier retires the current
  work item exactly once. It does not directly retire the launch token; the
  token retires only after the domain responsibility set becomes empty.

#### 5.4.4 `dataflow.work.spawn`

```text
arguments:
  AnyType:$childItem;
results:
  none;
regions:
  none;
traits:
  DeclareOpInterfaceMethods<MemoryEffectOpInterface>.
```

* The op is legal transitively inside a `dataflow.thread` with a
  `DynamicWork` domain and outside every `dataflow.graph`. Its operand type must
  equal the definition's designated work-item argument type.
* It atomically acquires one child responsibility before publishing that child
  to the current domain. The child receives the current item as parent and the
  next program-order child ordinal. The op has no target symbol, result handle,
  queue choice, priority, or Mapping field.
* It is effectful and cannot be removed, duplicated, reordered across the
  current item's retirement, or treated as a nested `dataflow.thread.launch`.
  Exact identity and termination are owned by Part 4.

#### 5.4.5 `dataflow.thread.wait`

```
arguments:
  Variadic<Dataflow_ThreadToken>:$asyncDependencies;
results:
  none;
traits:
  AtLeastNOperands<1>.
```

* A caller-side ordered stored-program wait. It consumes at least one
  thread completion token and completes only after every supplied token
  has retired. The operand set is unordered all-of.
* The op produces no SSA result and no graph-control `none` value. It
  is not a memory barrier and does not define memory visibility.
* The op is not `Pure`; it remains a causal wait in the stored program.

#### 5.4.6 Boundary Operands And Derived Views

No dedicated boundary or partition-carrier operation is part of the canonical
thread ABI. A launch operand and its matching ordinary thread block argument
are the same typed software value. Alias, no-alias, bounds, and access-summary
facts used by optimization remain ordinary MLIR analysis facts or semantic
candidate operations; they are not encoded by a provenance-only passthrough
op.

Code inside the thread may derive source induction variables, subviews, local
ranges, or an explicit linear id from ordinary launch operands and trailing
logical coordinates. Those computations are program semantics and therefore
survive whenever their results remain observable. Part 4 defines this boundary
without introducing `map_info`, partition-domain, layout, or coordinate-query
ops.

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
  OptionalAttr<DictArrayAttr>:$arg_attrs,
  OptionalAttr<DictArrayAttr>:$res_attrs;
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
* The current `function_type` ABI is `(T0, ..., TN) -> (R0, ..., RM)` and
  contains only application payloads. `input_segments` and `result_segments`
  classify those payloads as values, streams, and memories. The graph
  `%start` and launch-facing `done_out` are explicit invocation protocol
  endpoints outside the function type. `graph.return` payload segments match
  all result types, and `graph.return.complete` derives `done_out`.
* `sym_name` is required and module-unique. `sym_visibility` is
  required and must equal `"private"` under the baseline visibility
  policy. The verifier rejects `"public"` and `"nested"` unless
  cross-module linkage is enabled by a separate spec.
* The body is `IsolatedFromAbove`. All values used inside the
  graph definition's body must enter through the entry block.
* The entry block has the layout `(%ctrl_in : none, %arg_0 : T0,
  ..., %arg_N : TN)`. The application arguments match
  `function_type.inputs`; the distinguished leading `ctrl_in` block argument
  is the per-launch start signal and is not part of the function type.
  Accordingly, `arg_attrs` is indexed only by application arguments and has no
  entry for `ctrl_in`; `res_attrs` is indexed by application results. The
  custom assembly form preserves both arrays through textual and bytecode
  serialization.
* The body's terminator is structural:

  ```text
  dataflow.graph.return
    values(%final_values...)
    streams(%output_streams...)
    memories(%output_memories...)
    complete(%retirement_frontier...)
  ```

  The payload segments, in that order, match all function results.
  `complete` contains one or more `none` witnesses and is the only
  completion truth. The compact `%complete, %values... : none, types...`
  form is permitted when there is one witness and the stream and memory
  segments are empty.
* `dataflow.graph` lit tests use module-scope graph definitions with
  deterministic symbol names and `dataflow.graph.launch` use sites. Tests
  anchor the explicit start argument, segmented return payloads, non-empty
  completion frontier, and launch-facing done result.
* C++ builders construct `dataflow.graph` as a function-like
  definition from `(StringRef sym_name, FunctionType functionType,
  ArrayRef<NamedAttribute> attrs)`, with optional `arg_attrs` / `res_attrs`
  arrays carried in the function-interface attributes. The body is added via
  the standard `FunctionOpInterface` body-construction path, with the entry
  block carrying the leading `none` `ctrl_in` block argument and the
  user-data block arguments.
* The op declares `RecursiveMemoryEffects` so module-scope walkers
  can observe per-callable effects. This does not provide an alternate
  launch-completion rule; retirement remains owned exclusively by
  `graph.return.complete`.

#### 5.5.2 `dataflow.graph.launch`

```
arguments:
  FlatSymbolRefAttr:$callee,
  Variadic<NoneType>:$dependencies,
  Variadic<AnyType>:$valueInputs,
  Variadic<ChannelType>:$streamInputs,
  Variadic<AnyType>:$memoryInputs,
  Variadic<ChannelType>:$streamOutputs;
results:
  Variadic<AnyType>:$valueResults,
  Variadic<AnyType>:$memoryResults,
  none:$done;
traits:
  DeclareOpInterfaceMethods<SymbolUserOpInterface>.
```

* `callee` is a flat symbol reference that must resolve to a
  `dataflow.graph` definition in the same module. The verifier
  rejects launches whose `callee` cannot be resolved or whose
  resolved op is not a `dataflow.graph`.
* The verifier checks each operand and result segment against the callee's
  normalized `[value, stream, memory]` FunctionType segments. Stream payloads
  bind to consumer or producer `!dataflow.channel<T>` endpoints; they are not
  launch SSA data results. The mandatory trailing `done : none` result is the
  per-launch retirement event.
* Each stream input binding carries one symbol-free affine `source_map`. Its
  dimensions are the consumer thread coordinates and its results select the
  producer thread coordinates. Direction is derived from the launch operand
  segment: stream inputs are consumer bindings and stream outputs are producer
  bindings. There is no independent channel direction or mode attribute.
  Graph-launch verification owns local count and consumer-domain checks. The
  finalized-program validator owns the cross-launch relation: one producer,
  at least one consumer, producer/result-rank agreement, bounds over the full
  consumer domain, and complete permitted channel use topology.
* The op materializes a per-launch firing of the callee at this exact program
  point. `done_out` is the all-of of the callee's
  `graph.return.complete` operands. Their causal closure covers final values,
  stream close and boundary commit, memory capability establishment and
  promised visibility, all observable effects, invocation-local state
  close/reset, and non-detached async work. A graph with real work cannot use
  raw `%start` as a fake completion witness.
* The op must appear inside a `dataflow.thread` definition's body,
  not at host scope and not inside another `dataflow.graph`
  definition's body. The verifier enforces this placement.
* The launch intentionally does not implement `MemoryEffectsOpInterface` and
  does not project effects from its sibling callee. The native
  finalized-program validator proves that the callee's explicit complete
  frontier covers all outputs, state closure, and observable effects; explicit
  dependencies and memory capability ports carry launch-site ordering.

#### 5.5.3 `dataflow.graph.wait`

```text
arguments:
  Variadic<NoneType>:$completionFrontier;
results:
  none;
traits:
  AtLeastNOperands<1>.
```

* This op is the only explicit InstructionCore stored-program wait for graph
  retirement. It blocks until every event in its unordered all-of completion
  frontier has occurred and produces no SSA result.
* The op must be transitively contained by exactly one `dataflow.thread`
  definition. It is invalid at host scope, inside `dataflow.graph`, or inside a
  nested thread definition.
* Each operand is either a `dataflow.graph.launch` `done` result or an event
  whose path-aware causal closure contains at least one such result. The
  finalized-program validator proves that every operand is a valid terminal
  graph-completion frontier; textual order and generic `none` use do not
  establish that fact.
* The wait inherits only the retirement and visibility obligations already
  owned by those graph completion events. It is not a system memory barrier,
  channel or NoC drain, thread-collective wait, or conversion to
  `!dataflow.thread_token`.
* The op is not `Pure`. A lowering inserts it only before the first
  stored-program continuation that actually requires retirement. Deferred SSA
  value readiness, launch `dependencies`, channel transport, and
  `dataflow.thread.yield` remain their existing finer- or coarser-grained
  mechanisms and must not acquire redundant waits.
* A channel message may cover a retirement-related causal dependency only when
  analysis proves one exact dynamic message relation under
  `docs/spec-dataflow-part-1-streaming.md`: the exact channel instance and
  endpoint bindings, the consumer-to-producer `source_map`, applicable path
  predicates, producer publication and consumer observation positions, and
  equality of the symbolic producer and consumer event positions. Coordinate
  equality, static-site equality, or identity `source_map` alone never proves
  that two repeated launch occurrences use the same message. Unknown
  cardinality or ordering fails closed and retains the required retirement
  dependency.
* `loom.spatial_region` is a transparent structured boundary. A blocking
  receive inside that region cannot be justified by a send that follows the
  region in the same stored-program strand merely because the published graph
  launch becomes asynchronous. Such a transformation would turn an
  inline-semantics deadlock into progress. A resulting retirement/send cycle
  therefore identifies a deadlocking or incorrectly cut candidate; lowering
  must not remove a wait by inventing a same-activation channel witness.

* `dataflow.load` and `dataflow.store`.
  - These dataflow primitives carry explicit memory-effect traits:
    - `dataflow.load`  declares `MemoryEffects<[MemRead]>`.
    - `dataflow.store` declares `MemoryEffects<[MemWrite]>`.
  - These use MLIR's default memory resource. They are deliberately
    coarse in the baseline policy: any load may-read all memory,
    any store may-write all memory. This is sufficient for graph
    body effects to roll up through the graph definition's
    `RecursiveMemoryEffects` trait. It does not create launch-site
    effect projection.
  - Tightening these effects to a per-`$mem`-operand declaration
    (so two loads on disjoint memrefs become reorderable) is
    an explicit dataflow dialect extension.

* No other dataflow op is modified by this spec.

## 6. Per-scf Lowering Templates

Graph-region lowering carries execution permission, captured values, and per-partition
`(write_frontier, read_frontier)` state in one recursive traversal. The
compiler-local transfer is:

```text
lower_region(E_in, values_in, {W_in[p], R_in[p]})
  -> (E_out, values_out, {W_out[p], R_out[p]})
```

This transfer is not an IR object. The final graph contains only ordinary SSA
values and the existing Dataflow primitives. Memref bindings remain static;
only values, addresses, data, selectors, and event streams are projected.
Leaf memory completion updates `W/R` but never silently replaces execution
permission.

This section records Dataflow templates for SCF boundaries. Recursive lowering
applies the same transfer to `scf.if`, normalized `scf.index_switch`,
source-sequential `scf.for`, `scf.while`, and fixed-domain
effect-form `scf.parallel` / `scf.forall`. A zero-case `scf.index_switch` is
replaced by its default region during structured normalization. Other
unsupported source forms must be normalized by Part 2 before handoff of the
selected Structured Program Candidate and are rejected if they remain in a
graph.

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
the phase bit fed into `carry` / `invariant` / `gate` does not have to
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
Control-only and mixed boundary-publication syncs are canonical software
actors. A mixed boundary-publication sync has canonical shape
`(none, T) -> (none, T)`. TechMapping may realize a control-only sync with a
wider all-control Fabric capability and must prove compatible arity and
positional semantic widths for every selected realization. Lack of such a
capability on one Fabric does not invalidate the canonical graph.

Registered pure compute actors inside `dataflow.graph`, including the
registered arithmetic, math, and LLVM computation operations, follow strict
all-operand firing: each dynamic firing consumes one token from every
operand and emits one token on every result. In particular,
`arith.select` is an eager three-input compute op in this model, not a
short-circuiting dataflow mux.

SSA multi-use is token broadcast. If one SSA stream value has multiple
uses, each use observes the same ordered token sequence. This is not a
destructive single-consumer read. The `scf.for` template relies on
this property because the loop phase stream independently drives
`carry`, `gate`, and `demux`; those consumers do not need to fire in
lockstep.

Frontend `memref<...>` values are not stream values in this sense.
They represent memory-region bindings for `dataflow.load` /
`dataflow.store`. Lowering must not feed memref bindings through
stream-shaping ops; it shapes address, data, operation, and explicit
`none` memory-order streams instead. The generic result-selection
templates below apply to scalar/data streams and `none` ordering
streams. A memref-typed structured-control result inside graph
extraction must be rewritten to explicit memory effects, kept in
InstructionCore code, or rejected before graph lowering.

Graph memory normalization must reject every residual LLVM load, store,
memcpy, memset, atomic, volatile, or fence operation. They do not implement the
canonical Dataflow memory capability and explicit completion-event contract;
source order, a value result, or an effect scan cannot substitute for it. A
supported LLVM memory operation must first normalize to `dataflow.load`,
`dataflow.store`, or another explicitly specified canonical memory actor.

The templates below show user-visible SSA value lowering. The same recursive
owner threads independent `none`-typed write and read frontiers through each
boundary as specified in `docs/spec-compiler-part-3-mem.md`; this is not an
optional optimization or a later reconstruction pass.

### Def + Launch Output Convention

The pseudocode templates in Section 6.1-Section 6.8 below show the **graph body
contents** for clarity. Every template's actual lowering output is a
`dataflow.graph` definition + a `dataflow.graph.launch` pair, with
the body shown lifted to module scope and the launch carrying the
per-instance ctrl/done plumbing:

```mlir
// At module scope (sibling of callable definitions):
dataflow.graph @<construction_local_sym>
    (%start : none, <user inputs>) -> (<user results>) {
  // <body contents per the template>
  dataflow.graph.return values(<user yield values>) streams() memories()
      complete(<retirement frontier>)
}

// At the explicit spatial-region site inside the enclosing
// dataflow.thread definition's body:
<user value results>, <memory results>, %done =
    dataflow.graph.launch @<construction_local_sym>
      deps(%dependency events) values(<value operands>)
      stream_inputs(<consumer channels>) memories(<memory imports>)
      stream_outputs(<producer channels>)
      : (<operand types>) -> (<value result types>, <memory result types>, none)
```

The publisher creates a deterministic, collision-free construction-local
symbol for each outlined graph. An existing `loom.spatial_region.graph_name`
may be used only as a readability or debug stem. The temporary region does not
own graph identity, and symbol spelling does not encode cut selection, source
order, graph identity, or artifact identity.

The templates therefore omit the def + launch wrap to keep the
body's structural diff readable. The wrap is mandatory output, not
an optimization, and is verified by the front-end's standard
verifier rules in Section 9.

### Phase Phasing Rule

A phase stream is loop control, not a plain valid bit. For a counted loop
with `N` body executions, `dataflow.stream` emits `N` IV tokens and a phase
stream `T^N F`. The final false token closes the activation and resets each
stateful consumer, but it has no paired IV or body execution.

The stream IV already has body cardinality and enters body arithmetic and
memory directly. Parent-domain captured values from `invariant` have
`N + 1` tokens and are projected through `dataflow.gate` before body use.
Recurrence values that also need a false-lane exit use selector-matched
`dataflow.demux`; loop results and memory-frontier exits consume that false
lane. A true body-local condition means the current body execution is not the
last execution; a false body-local condition means it is the last execution.

Different regions of one source loop may therefore have different phase
streams. The loop-level phase decides whether the source loop continues
or exits; a gated body phase controls state local to the body region
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

This translation uses the recursive graph owner. The condition
demuxes execution, captured non-memref values, and both frontier components
for every partition touched by either branch. Each branch is lowered
recursively. The same condition then muxes execution, each result position,
`W_P`, and `R_P` componentwise.

A missing else is an identity false lane. An unexecuted path forwards its
incoming frontier and never performs a safe-address access or emits a fake
completion. Same-path prerequisites use `sync`; mutually exclusive exits use
`mux`. Execution remains distinct from both memory components.

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

Emitted lowering skeleton:

```
# Structural loop entry and loop-back control. This exists even when
# the source while has no data inits.
%iter_ctrl = carry %cond, %entry_ctrl, %after_done : none

# Each before block argument is loop-carried in before phase.
%a_i = carry %cond, %init_i, %a_next_i : A_i

# The before region consumes %iter_ctrl and %a_i..., then produces:
#   %cond        : i1
#   %b_j         : B_j, one stream per scf.condition trailing operand
#   %before_done : none, the tail of before-region side effects

# scf.condition true operands enter after; false operands are results.
# Lane convention: lane 0 = false, lane 1 = true.
%b_exit_j, %b_after_j =
  demux %cond, %b_j : (i1, B_j) -> (B_j, B_j)

# The recursively lowered before exit is projected with the same selector.
%while_done, %unused_true =
  demux %cond, %before_done : (i1, none) -> (none, none)
%after_phase, %after_ctrl =
  gate %cond, %before_done : (i1, none) -> (i1, none)

# The after region consumes %after_ctrl and %b_after_j..., then
# produces:
#   %a_next_i... : A_i, the scf.yield operands
#   %after_done  : none, the after-region completion token; if the
#                  region has no side effects and no extra control-only
#                  work, this may be %after_ctrl

%res_j = %b_exit_j
```

* `%cond` is the i1 token computed by the before-region's
  `scf.condition`. There is no `stream` op here; an arbitrary `i1`
  stream produced by before-region computation drives the loop.
* The before-region executes once more than the after-region. Demuxing the
  before exit gives exactly `K` after permissions and one while exit. The
  final false before execution is therefore part of loop completion.
* `%b_exit_j` becomes the loop result. The same selector projects values,
  execution, and memory-frontier components into matching phases.
* Each `%a_next_i` has length `K`, one value from each after-region
  execution. `dataflow.carry` consumes a next value only with
  `cond=true`; `cond=false` closes and resets the carry without
  consuming feedback.
* Before-region invariants use the before-phase `%cond` stream.
  After-region-only invariants are replayed in before phase and projected
  through a true-lane demux. This keeps zero-trip loops from producing an
  after-only value.
* Each touched partition has independent write-frontier and read-frontier
  carries following the same structure as `%iter_ctrl`. Before starts from
  their outputs. True lanes enter after and feed the next before activation;
  false lanes are the loop exits. This preserves memory effects performed by
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
| `%while_done` | `[before_done2]` |
| `%a_next_i` | `[a1, a2]` |

The final `%cond = false` token is consumed without `%a_next_i` or
`%after_done`. It emits no new before value and returns each carry to
its init state. Independent write-frontier and read-frontier carries follow
the same selector contract in `docs/spec-compiler-part-3-mem.md`.

#### While Boundary Translation

This translation uses condition-driven carry rings for
execution, source inits, and each touched `W_P/R_P` component. Carry outputs
enter before directly. After before is lowered, the false lanes are the while
execution, result, and frontier exits. `dataflow.gate` projects execution and
captured values into after phase; true condition-argument and frontier lanes
enter after through their selector-matched projections. After exits feed the
next before activation.

Before therefore executes `K + 1` times when after executes `K`. The final
false before effects are included in the outgoing pair. A final-false read
updates `R_P` at loop exit and a following write must wait for it. False does
not consume dummy feedback.

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
# %init / %limit / %step / iv stream to share a scalar signless integer
# type (see docs/spec-dataflow-part-1-streaming.md). The lowering
# therefore inserts arith.index_cast at the boundary: %lb / %ub /
# %step are cast from index to a chosen iN, and the body IV %i is
# cast back to index before memref indexing. The chosen iN is Loom's
# configured index-width integer type.

%lb_iN, %ub_iN, %step_iN  = arith.index_cast %lb, %ub, %step : index to iN
%i_iN, %loop_phase = stream %lb_iN, %ub_iN, %step_iN
                      step add while slt : iN
%i = arith.index_cast %i_iN : iN to index
# body memory and address computation consume %i directly

# Source-sequential execution recurrence and zero-trip exit:
%ctrl_raw = carry %loop_phase, %ctrl_in, %body_done : none
%loop_exit_ctrl, %body_ctrl =
  demux %loop_phase, %ctrl_raw : (i1, none) -> (none, none)
```

For `N` dynamic body executions:

| Stream | Length | Meaning |
|--------|--------|---------|
| `%loop_phase` | `N + 1` | `N` true tokens plus one false close |
| `%i_iN` / `%i` | `N` | body induction values |
| `%ctrl_raw` | `N + 1` | initial permission plus body feedback |
| `%body_ctrl` | `N` | source-sequential body permissions |
| `%loop_exit_ctrl` | `1` | structured exit token |

The no-result case has no data loop result to compute. The stream emits
exactly one IV per body execution and no IV for the close transition. The
recursively lowered body returns `%body_done`, which authorizes the next
source iteration. Memory leaves additionally wait on their partition
frontiers as specified in `docs/spec-compiler-part-3-mem.md`.
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
%i_iN, %loop_phase = stream %lb_iN, %ub_iN, %step_iN
                      step add while slt : iN
%i = arith.index_cast %i_iN : iN to index

%acc_raw = carry %loop_phase, %init, %next : i32

%acc_exit, %acc_body =
  demux %loop_phase, %acc_raw : (i1, i32) -> (i32, i32)

# body executes only in body phase
%x = dataflow.load %A[%i], ... : memref<?xi32>
%next = arith.addi %acc_body, %x : i32

%sum = %acc_exit
```

The iter-arg state stream is deliberately in loop phase, not body
phase. `carry` sees `%loop_phase`, so it emits an `N + 1` state stream:
the initial value, then one carried value after each true iteration.
The same `%loop_phase` demuxes that state stream. The true lane produces
exactly `N` `%acc_body` values and the false lane produces exactly one
`%acc_exit` value used as the loop result.

The feedback to `carry` has length `N`: `%next` is produced once per
true iteration. On the final false phase, `carry` consumes no next
value, emits no additional state, and returns to its init state.

For `N = 0`:

| Stream | Tokens |
|--------|--------|
| `%loop_phase` | `[F]` |
| `%i` | `[]` |
| `%acc_raw` | `[init]` |
| `%acc_body` | `[]` |
| `%next` | `[]` |
| `%acc_exit` | `[init]` |
| `%sum` | `init` |

For `N = 1`:

| Stream | Tokens |
|--------|--------|
| `%loop_phase` | `[T, F]` |
| `%i` | `[0]` |
| `%acc_raw` | `[init, next0]` |
| `%acc_body` | `[init]` |
| `%next` | `[next0]` |
| `%acc_exit` | `[next0]` |
| `%sum` | `next0` |

For `N = 2`:

| Stream | Tokens |
|--------|--------|
| `%loop_phase` | `[T, T, F]` |
| `%i` | `[0, 1]` |
| `%acc_raw` | `[init, next0, next1]` |
| `%acc_body` | `[init, next0]` |
| `%next` | `[next0, next1]` |
| `%acc_exit` | `[next1]` |
| `%sum` | `next1` |

Multiple iter_args lower independently using the same pattern, one
`carry` / `demux` state ring per iter_arg. Body operations may
freely combine the body-lane values from multiple iter_args before
feeding the corresponding yielded values directly to their carries.
Memref operands are not iter_arg-like stream state; only explicit
`none` memory-order state is carried for memory dependences.

* For each touched memory partition, the loop has independent hidden
  `none` carries for `W_P` and `R_P`. Both are initialized from the incoming
  frontier pair, driven by `%loop_phase`, sent to the body on the true lane,
  and returned as loop exits on the false lane. The zero-trip case forwards
  both initial components.

#### For Boundary Translation

This translation uses one loop selector from
`dataflow.stream` and independent `carry -> demux` rings for execution,
iter_args, and each touched `W_P/R_P` component. True lanes enter the
recursively lowered body; false lanes are loop exits. Captured non-memref
values use `invariant` followed by true-lane projection.

The body feeds every ring independently. Zero trip produces only the false
selector token, so init execution, values, and frontier components transfer
unchanged. Read-only state does not create RAR order; write feedback preserves
RAW, WAR, and WAW across source-sequential iterations.

### 6.4 `scf.forall`

#### Accepted Input Contract

The `Ownership Materialization And Handoff` section of
`docs/spec-compiler-part-2-scf.md` is the sole normative owner of forall
normalization. Before Part 3 begins, the selected Structured Program Candidate
must have:

* materialized a forall selected as an AccCore thread domain as a canonical
  `dataflow.thread` definition and `dataflow.thread.launch`;
* retained a graph-owned forall only as a mapping-free, effect-form,
  compile-time fixed-domain construct whose `P[]` width, ownership, and
  cross-lane legality are materialized in semantic IR and can be re-proved;
  and
* materialized every supported aggregation or reduction into accepted
  semantics, or failed finalizability truthfully.

Part 3 does not convert aggregation form, decide thread ownership, rewrite
forall to parallel as an optimization policy, infer `P[]`, serialize lanes, or
select a reduction strategy. A dynamic domain, mapping attribute, shared
output, result, combining action, or failed legality re-proof causes atomic
failure before canonical graph publication. Cached provenance never changes
this result.

For this boundary, an accepted effect-form forall has no `shared_outs`, no op
results, and an empty `scf.forall.in_parallel` terminator. In the example,
`%N` must resolve to the selected candidate's compile-time fixed extent:

```mlir
scf.forall (%i) in (%N) {
  %x = memref.load %A[%i] : memref<?xf32>
  %y = arith.mulf %x, %x : f32
  memref.store %y, %B[%i] : memref<?xf32>
  scf.forall.in_parallel {}
}
```

Its result is represented only by explicit side effects in the body.

The following is an aggregation form and is not accepted by Part 3:

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

Part 3 rejects this form before graph mutation. It never drops the combining
region or publishes a `dataflow.graph` that omits the aggregation. Any legal
materialization belongs to the Part 2 owner; this document intentionally does
not define a bufferization or combining algorithm.

If Part 2 selects an effect-form forall as an AccCore thread domain, the input
accepted by Part 3 is already the definition-and-launch carrier shape below.
The rank-one source sketch is retained only to relate the source induction
variable to the canonical logical-coordinate ABI; it is not a Part 3
transformation:

```mlir
scf.forall (%tx) in (%N) {
  memref.store %v, %B[%tx] : memref<?xf32>
  scf.forall.in_parallel {}
}
```

```mlir
// At module scope (sibling of callable definitions):
dataflow.thread @t_<funcSym>_<seq>(%B_arg : memref<?xf32>, ...)
    attributes { sym_visibility = "private" } {
^bb0(%B_arg : memref<?xf32>, ..., %thread_ctrl : none, %coord : index):
  // For normalized zero-based unit-step forall, source IV equals coordinate.
  memref.store %v, %B_arg[%coord] : memref<?xf32>
  dataflow.thread.yield
}

// At the original scf.forall site:
%tok = dataflow.thread.launch @t_<funcSym>_<seq>
       extents(%N) args(%B, ...)
       : (memref<?xf32>, ...) -> !dataflow.thread_token
dataflow.thread.wait %tok : !dataflow.thread_token
```

The launch extents own an arbitrary-rank dense zero-based domain. The thread
entry block has one trailing logical-coordinate argument per extent, in source
dimension order. For each dimension, nonzero or dynamic source lower bounds
and steps cross as ordinary operands and the body reconstructs
`source_iv = lower + coordinate * step`. Values captured from outside the
source forall likewise become ordinary launch operands and matching definition
arguments. Sections 5.4.1 and 5.4.2 and
`docs/spec-compiler-part-4-partitioned-data.md` own the complete ABI.

Code inside the thread definition remains InstructionCore code unless the
selected Structured Program Candidate explicitly wraps it in
`loom.spatial_region`. That compiler-internal region remains the temporary
SpatialCore ownership carrier until Part 3 atomically replaces it with a
finalized `dataflow.graph` definition and launch. Memory operations outside
the region remain in the InstructionCore body. The thread token preserves the
source continuation dependency through another launch dependency or an
explicit `dataflow.thread.wait`.

#### Forall Boundary Translation

Within an explicit `loom.spatial_region`, an accepted graph-owned forall is
recursively replicated into its already selected static lanes. Every lane
starts from the same incoming execution and per-partition `(W, R)` frontier,
and lane exits are reduced with fixed-arity all-of joins. Empty domains are
identity transfers. The forall and its empty
`scf.forall.in_parallel` terminator are removed. No forall boundary,
partition id, dependence summary, or traversal order survives into canonical
graph IR.

### 6.5 `scf.parallel` with `scf.reduce`

#### Accepted Input Contract

Parallel normalization is owned exclusively by the Structured Program
Candidate lineage specified by the "Ownership Materialization And Handoff"
section of `docs/spec-compiler-part-2-scf.md`. Part 3 accepts a graph-owned
`scf.parallel` only when it is:

* effect-form, with no op results, init values, or reduction operands and with
  an empty `scf.reduce` terminator;
* mapping-free and nested under the selected `loom.spatial_region` carrier;
  and
* compile-time fixed over an arbitrary-rank logical domain whose selected
  `P[]` widths and cross-lane legality are explicit in the candidate and can
  be re-proved from current semantics.

Dynamic-width, resultful, reduction-bearing, mapped, or otherwise unproved
forms fail before graph mutation. Part 3 does not choose any chunk count
`K`, including `K = 1`; flatten, serialize, or partition the iteration space
as policy; invent `P[]`; or select and inline a reduction order. Any such
choice must already be materialized as accepted semantic IR by Part 2. This
document intentionally defines no future upstream chunking or reduction
algorithm.

For an accepted one-dimensional effect-form loop, `%N` denotes a
compile-time-resolved extent and the candidate has already selected
`P[] = [%N]`:

```mlir
scf.parallel (%i) = (%c0) to (%N) step (%c1) {
  %x = memref.load %A[%i] : memref<?xf32>
  %y = arith.mulf %x, %x : f32
  memref.store %y, %B[%i] : memref<?xf32>
  scf.reduce
}
```

`scf.parallel` is not a second Dataflow loop primitive. No
`dataflow.parallel`, `dataflow.reduce`, reduction enum, schedule record, or
parallel control op is introduced.

#### Parallel Boundary Translation

For rank `r` and fixed widths `P[]`, the graph owner creates one static lane
for each logical coordinate tuple in the selected Cartesian domain. Each lane
starts from the same incoming execution and per-partition `(W, R)` frontier,
substitutes its already selected source induction values, and recursively
lowers the existing body. Incomparable exits are joined with fixed-arity
all-of; an empty domain is an identity transfer. The Cartesian rank is not
bounded by this lowering contract.

Lane enumeration is an implementation detail and never creates
cross-iteration program order. A failed independence or ownership re-proof
causes atomic failure. No parallel boundary, coordinate tuple, `P[]` record,
dependence summary, or traversal order survives into canonical graph IR.

### 6.6 `scf.index_switch`

Structured normalization replaces a zero-case `scf.index_switch` with its
default region. Every remaining switch lowers through the same recursive
selection transfer used by `scf.if`. The canonical graph contains only
ordinary selector arithmetic and the existing `dataflow.demux` /
`dataflow.mux` actors; it does not retain SCF or introduce a second switch
abstraction.

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

The one-case form has two dynamic lanes and uses an `i1` selector: `false`
selects default and `true` selects the single case. With two or more cases,
the normalized selector has `index` type. The zero-case form has only the
default region and is eliminated before this template is applied.

For two or more cases, the normalized selector is computed as ordinary
data, not with `dataflow.mux`. A `dataflow.mux` is selective and would
leave each unselected case-lane constant token in its queue. Across
many switch invocations those leftover tokens would accumulate without
bound under any bounded-buffer runtime and eventually apply backpressure.
They cannot be discarded because every candidate lane remains semantically
live for a later selector token. Ordinary `arith.select` follows
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
* A zero-case form is rejected atomically before recursive graph lowering.
  No selector, branch transfer, memory-frontier projection, or graph mutation
  is created for it.

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

`GraphRegionLowering` normalizes the source argument once, orders lanes as
default followed by source case order, and invokes the shared recursive
selection transfer. That exact selector drives execution permission,
projected non-memory captures, every result mux, selected execution
completion, and each touched partition's `W_P` and `R_P` demux/mux pair.
Each region is recursively lowered only from its lane-specific inputs, so an
unselected region receives no execution or frontier token and its effects do
not execute. Multiple results produce one mux per result position. A branch
that does not touch a partition returns its lane-specific incoming frontier,
while a selected branch contributes its recursively reduced causal frontier.

### 6.7 `scf.execute_region`

Structured normalization inlines a supported `scf.execute_region` before
graph-region lowering. A residual region means the Structured Program
Candidate is not finalizable and is rejected before graph mutation.

#### Execute Region Boundary Translation

After upstream inlining, the contents participate in ordinary sequential
recursive lowering. No dedicated Dataflow actor or persistent region summary
is required.

### 6.8 `scf.yield`

* Already a thin terminator. The lowering of the parent op produces
  the yield's effect; the standalone yield is removed.

## 7. Memory Frontier Model

`docs/spec-compiler-part-3-mem.md` specifies the single recursive owner,
basic graph-local alias partitions, leaf transfer equations, and independent
write/read recurrence state. Section 6 of this document specifies how the
same selectors project execution, values, and both frontier components at
each supported SCF boundary.

## 8. Logical Domains And Data Views

`docs/spec-compiler-part-4-partitioned-data.md` specifies the two canonical
launch domains, dynamic responsibility transfer and termination, source-IV
reconstruction, and derived-view boundary. These semantics add no partition
carrier or mapping attribute to SCF-to-DFG flattening.

## 8.1 Canonical Artifact Finalization And Entity Identity

### Artifact Owner And Schema

The Canonical Dataflow Program is the single semantic root of the fixed
Artifact family:

```text
loom.canonical_dataflow 1.0
```

The family owns its admitted module surface, canonical semantic relation
graph, canonical writer, artifact-local entity catalog, and importer. Common
owns only the shared Artifact envelope, schema/version framing, SHA-256 v1
identity calculation, and collision-checked publication. Mapping, simulation,
Evaluation, visualization, and native caches consume Dataflow-owned
references; none may assign or reinterpret a Dataflow entity ID.

Finalization is failure-atomic. It operates on a private clone of the complete
program, validates every canonical graph and the whole-program thread, launch,
channel, memory-root, symbol, and completion relations, constructs canonical
bytes, invokes the Common finalizer, and publishes only the complete valid
Artifact. There is no `is_finalized` operation attribute or partially
finalized program state. A valid Common envelope, exact schema descriptor,
canonical bytes, and successful independent family verification together
define a finalized Artifact.

### Closed Entity Catalog

The first schema has exactly five independently referenceable entity kinds:

```text
CanonicalDataflowEntityKind =
    Graph
  | Actor
  | RootThreadLaunch
  | StaticGraphLaunch
  | LogicalMemoryRoot
```

Their carriers are:

* **Graph.** Each reachable finalized `dataflow.graph` definition.
* **Actor.** Each operation in a graph body accepted as a real actor by the
  shared canonical Dataflow actor classifier. Structural terminators and
  boundary block arguments are not actors.
* **RootThreadLaunch.** Each retained static `dataflow.thread.launch` site.
  Thread launches cannot occur inside another thread or graph, so every such
  site is a root launch.
* **StaticGraphLaunch.** Each retained static `dataflow.graph.launch` site in
  a thread definition.
* **LogicalMemoryRoot.** Each static imported-memory formal role and each
  canonical fresh-allocation definition. A view preserves an existing root
  and does not create another root entity.

`dataflow.thread` definitions do not receive IDs in this schema. Every
persistent use begins at a root launch and recovers the definition through its
typed callee relation. Private functions, thread definitions, actor
operands/results, graph boundaries, software edges, memory views, channel
branches, and dynamic invocation, work-item, memory-object, or firing
occurrences are likewise not independent entities. They are recovered through
identified owners plus typed semantic ordinals, canonical relations, or
execution-local identity. A future independently referenceable semantic
object requires an explicit schema catalog change; a consumer cannot mint an
ID for convenience.

All five kinds share one Artifact-global unsigned 64-bit `EntityId` namespace.
Zero is a valid ID and there is no sentinel value. The finalizer assigns the
dense range `[0, entity_count)` in canonical-slot order, but serialized record
position is not identity and consumers must resolve the explicit ID.

The complete typed persistent references are:

```text
GraphRef             = (CanonicalDataflow ArtifactIdentity, Graph EntityId)
ActorRef             = (CanonicalDataflow ArtifactIdentity, Actor EntityId)
RootThreadLaunchRef   = (CanonicalDataflow ArtifactIdentity,
                         RootThreadLaunch EntityId)
StaticGraphLaunchRef  = (CanonicalDataflow ArtifactIdentity,
                         StaticGraphLaunch EntityId)
LogicalMemoryRootRef  = (CanonicalDataflow ArtifactIdentity,
                         LogicalMemoryRoot EntityId)
```

An artifact that already binds the exact Dataflow identity may use a compact
typed local ID on the wire. The full meaning still includes that binding.
Wrong-kind, foreign-artifact, missing, duplicate, out-of-range, or
noncanonical IDs are invalid.

### Closed Structural Reference Catalog

Objects below the five entity kinds use closed owner-relative structural
references. They do not receive another `EntityId`, and consumers must not
replace them with symbol paths, operation positions, generic field paths, or
native dense indices.

A graph launch is interpreted in the context of one root thread launch:

```text
RootedGraphLaunchRef =
  (RootThreadLaunchRef, StaticGraphLaunchRef)
```

The referenced static graph-launch site must belong to the thread definition
resolved from the root launch. This context is required because one thread
definition may be used by several root launches with different channel
bindings, memory roots, execution bindings, or physical targets. It remains a
static structural reference, not a dynamic invocation identity.

Graph-local token endpoints use the following closed forms:

```text
GraphIngressTokenRef =
    Start(GraphRef)
  | ValueInput(GraphRef, value-input ordinal)
  | StreamInput(GraphRef, stream-input ordinal)

GraphEgressTokenRef =
    ValueOutput(GraphRef, value-output ordinal)
  | StreamOutput(GraphRef, stream-output ordinal)
  | CompletionFrontier(GraphRef, completion-frontier ordinal)

ActorTokenResultRef  = (ActorRef, result ordinal)
ActorTokenOperandRef = (ActorRef, operand ordinal)

CanonicalGraphProducerEndpointRef =
    GraphIngressTokenRef
  | ActorTokenResultRef

CanonicalGraphConsumerEndpointRef =
    ActorTokenOperandRef
  | GraphEgressTokenRef
```

The Dataflow actor-port classifier must validate every actor ordinal as a
token-plane port. A memory capability operand or result is never admitted by
these unions. The exact producer endpoint and Dataflow def-use relation derive
one complete canonical sink set. TechMapping may remove sinks proven internal
to a selected realization, but it cannot change endpoint identity or create a
second software-edge catalog.

The thread/graph ABI exposes one-message boundary transfers separately from
thread-level channels:

```text
RootThreadBoundaryTransferRef =
    Start(RootThreadLaunchRef)
  | ValueInput(RootThreadLaunchRef, value body-operand ordinal)
  | Completion(RootThreadLaunchRef)

GraphLaunchBoundaryTransferRef =
    Start(RootedGraphLaunchRef)
  | ValueInput(RootedGraphLaunchRef, value-input ordinal)
  | ValueResult(RootedGraphLaunchRef, value-result ordinal)
  | Done(RootedGraphLaunchRef)
```

Root-thread value inputs exclude channel handles and memory capabilities.
Extents and derived coordinates belong to the Thread Dispatch parameter
contract rather than this message catalog. Each boundary-transfer reference
owns exactly one source terminal and one sink terminal. Root start and value
inputs flow from the runtime boundary to the selected InstructionCore; root
completion flows back to runtime retirement. Graph start and value inputs flow
from the selected InstructionCore to its SpatialCore; graph value results and
done flow in the reverse direction. Explicit thread-token dependencies remain
part of Thread Dispatch and do not create a second completion-message graph.

Channel endpoints are:

```text
ThreadChannelSendSiteRef =
  (RootThreadLaunchRef, canonical send-site ordinal)

ThreadChannelReceiveSiteRef =
  (RootThreadLaunchRef, canonical receive-site ordinal)

ChannelProducerRef =
    GraphStreamOutput(RootedGraphLaunchRef, stream-output ordinal)
  | ThreadSend(ThreadChannelSendSiteRef)

ChannelConsumerRef =
    GraphStreamInput(RootedGraphLaunchRef, stream-input ordinal)
  | ThreadReceive(ThreadChannelReceiveSiteRef)
```

Send and receive ordinals index Dataflow-owned canonical endpoint inventories
for the rooted thread. They are not textual positions. The exact channel
relation and each consumer-owned `source_map` derive the complete canonical
consumer set for one producer. Dynamic message correspondence remains the
ordered event relation specified by
`docs/spec-dataflow-part-1-streaming.md`; no message ordinal, activation
pairing, epoch, or Physical Tag enters these static references.

The complete transfer-terminal unions are:

```text
CanonicalProducerTerminalRef =
    RootThreadBoundarySource(RootThreadBoundaryTransferRef)
  | GraphLaunchBoundarySource(GraphLaunchBoundaryTransferRef)
  | ChannelProducer(ChannelProducerRef)

CanonicalSinkTerminalRef =
    RootThreadBoundarySink(RootThreadBoundaryTransferRef)
  | GraphLaunchBoundarySink(GraphLaunchBoundaryTransferRef)
  | ChannelConsumer(ChannelConsumerRef)
```

A boundary source derives its one paired sink. A channel producer derives its
complete non-empty sorted consumer set. Graph value results and later graph
value inputs remain two distinct InstructionCore-facing ABI transfers. A
compiler that wants direct graph-to-graph streaming must represent it with a
channel/stream or fuse the graphs; Mapping cannot invent that rewrite.

Memory references remain on the capability plane:

```text
LogicalMemoryViewRef =
  (LogicalMemoryRootRef, canonical root-local view ordinal)

LogicalMemoryRootOrViewRef =
    Root(LogicalMemoryRootRef)
  | View(LogicalMemoryViewRef)

ContextualActorRef =
  (RootedGraphLaunchRef, ActorRef)

MemoryExposureRef =
  (RootedGraphLaunchRef, graph memory-result ordinal)

FenceActorFamilyRef =
  ActorRef validated as dataflow.fence
```

The root-local view inventory owns each root-preserving static view relation.
Instantiating one reusable graph view under different logical roots therefore
produces distinct structural references in the corresponding root
inventories, without allocating view entities. A contextual actor must belong
to the graph called by its rooted launch. A memory exposure identifies a
launch-contextual graph memory result and resolves through the Dataflow memory
relation to exactly one logical root or view.

System service members use one closed obligation-relative union:

```text
ServiceMemberRef =
    MessageTransfer
  | AddressedMemoryActor(ContextualActorRef)
  | FenceActor(ContextualActorRef)
```

`MessageTransfer` is the singleton member of a transfer obligation, including
multicast. Addressed-memory and fence members derive their exact Canonical
Service kind and local legs from the actor semantics. A memory exposure is not
a service member and has no request or response leg; it is a capability
boundary selected by a service target binding or Mapping exposure entry.

System resource-time anchors reuse transfer terminals:

```text
StaticTransferEventRef =
    Produced(CanonicalProducerTerminalRef)
  | Consumed(CanonicalSinkTerminalRef)

EventFamilyKey =
  (StaticTransferEventRef,
   canonical projection of Dataflow-owned logical coordinates
   and launch parameters)
```

There is no static-event `EntityId`. The projection is selected from the
Dataflow-owned logical signature and denotes a static event family. Runtime
may append a transient occurrence handle, but that handle never enters
Artifact identity, Mapping, channel ordering, or Physical Tag assignment.
SpatialMapping-local actor activity remains owned by the SpatialMapping and is
rebased to these System-visible boundary events only by the derived
SystemMapping closure.

### Canonical Semantic Relation Graph

Before labeling, the finalizer removes every pre-existing
`dataflow.entity_id` from its private clone. The relation graph contains the
complete semantic program, not only the five entity nodes. It includes:

* each actor's registered `OperationSchemaId`, exact types, and closed
  schema-owned semantic property and attribute projection;
* explicit operand/result ordinals, SSA def-use, block-successor, containment,
  region, boundary-segment, symbol-use, and launch-callee relations;
* logical-memory root and root-preserving view relations; and
* explicit execution order in HostCore and InstructionCore stored-program
  regions.

A `dataflow.graph` body is a graph region, so actor textual order contributes
no relation. Module and symbol-table order are also nonsemantic. Stored-program
block/control/operation order is semantic and remains in the relation graph.
This distinction prevents a generic "ignore operation order" rule from
silently changing InstructionCore behavior.

SSA and block labels, private symbol spelling, source and filesystem
locations, debug/provenance metadata, visual coordinates, printer order, and
builder insertion order are excluded. Private symbols are resolved to typed
relations and receive canonical printed labels. An externally visible linkage
name is ABI semantics rather than a private printer label and is included
together with its linkage and visibility contract. Every other registered
non-actor field is semantic by default. Every actor field must be classified
by its closed operation-schema projection. Excluding any field requires an
explicit owner-spec rule rather than an open ignore list.

The equivalence boundary is exact typed and attributed structural isomorphism.
It does not prove algebraic or whole-program functional equivalence. A
semantics-preserving Dataflow rewrite whose graph is non-isomorphic therefore
produces a different Canonical Dataflow Artifact, as required by the Dataflow
optimization lineage.

Canonical labeling determines semantic slots independently of source handles.
Entities in one automorphism orbit have no recoverable nonsemantic source
identity. The finalizer may return a source-object-to-final-ID provenance map
for the current derivation, but it does not enter canonical bytes and is not a
reference authority.

### Materialization, Import, And Memory Instances

After canonical slots are fixed, the finalizer assigns IDs and materializes
the single `dataflow.entity_id` attribute on each entity carrier. A logical
memory root carried by an ordinary function-like argument uses that argument's
existing attribute dictionary. The canonical writer emits normalized private
symbol, SSA, and block labels, canonical unordered collections, the derived
IDs, and all semantic relations. It omits locations and the explicitly
nonsemantic metadata above. The Common finalizer hashes exactly those
family-owned canonical bytes.

The derived IDs are excluded while recomputing canonical labels, avoiding a
circular identity definition. A finalized importer independently reconstructs
the relation graph and requires every materialized ID to match the canonical
assignment. A mutable authoring program may omit IDs; any supplied values are
discarded on the private finalization clone rather than trusted.

One Dataflow-owned read-only `CanonicalDataflowProgramView` projects the five
typed ID maps, every closed structural-reference inventory above, canonical
actor and endpoint relations, rooted launch contexts, channel producer and
consumer relations, launch-callee closure, logical-memory root/view and
exposure relations, service-member derivation, and static transfer events.
Its native indices and lookup tables are disposable caches. Mapping's
draft/search structures and simulator event tables may cache this view, but
cannot define another persistent graph, actor, launch, terminal, member,
event, or memory catalog.

`LogicalMemoryRootRef` identifies a static software root role, not one runtime
object. An imported runtime object is bound by the exact launch and runtime
memory registry. A fresh graph allocation instance is derived from its static
root reference and graph invocation occurrence. If two imported roles alias at
runtime, the runtime registry relates them to the same object without merging
their static entity IDs. A memory view remains a typed structural reference
whose root relation resolves to exactly one `LogicalMemoryRootRef`.

### Anchor Verification

Anchor-level tests cover:

* invariance under private-symbol and SSA renaming, location changes, module
  definition reordering, and graph actor textual reordering;
* identity changes for actor kind, type, semantic attribute, operand ordinal,
  edge, stored-program order, or externally visible linkage changes;
* one registered operation schema drives graph admission, canonical actor
  projection, simulator lookup, and Fabric matching, while an unclassified
  actor property is rejected;
* equal canonical bytes and valid unique IDs for isomorphic symmetric inputs,
  without asserting a source-handle-to-slot correspondence;
* rejection of stale, missing, duplicate, noncanonical, foreign, or wrong-kind
  references and unresolved symbol or root relations;
* distinct rooted graph-launch references when one thread definition is
  reached from two root launches;
* token-plane endpoint rejection for a memory-capability ordinal and
  out-of-range actor or boundary ordinals;
* complete canonical sink derivation for one multicast channel producer;
* rejection when a memory exposure is interpreted as a service member or
  assigned a service leg;
* static transfer-event round trip without a static event entity ID; and
* DFG-sim actor import from an exact Canonical Dataflow Artifact without any
  Mapping Artifact.

Tests do not pin printer whitespace, a particular graph-labeling algorithm,
native container layout, source-handle provenance, or a broad operation
fixture matrix.

## 9. Verifier Rules (Front-End Specific)

In addition to the Dataflow dialect and finalized-program verifier set:

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
    `(T0..TN)`; `function_type` results are empty.
  - `domain` is one closed Part 4 domain. For `DenseRectangular`, entry block
    argument count equals `numBodyOperands + 1 + coordinateRank`. For
    `DynamicWork`, the rank is zero, the count is `numBodyOperands + 1`, and
    `work_item_arg_ordinal` selects one ordinary input. The dense block-arg
    layout is `(args_*, thread_ctrl, coord_*)`: the first
    `N == numBodyOperands`
    block args mirror `function_type.inputs` exactly, then one
    `none`-typed `thread_ctrl` block arg, then one `index`-typed
    block arg per logical coordinate (in source-dimension order). The
    coordinate suffix length is the sole definition of rank. This ordering
    keeps the first `N` block args aligned with
    `function_type.inputs`, satisfying the upstream
    `FunctionOpInterface` invariant.
  - The body is `IsolatedFromAbove`: every SSA value used in the
    body and defined outside it is rejected.
  - Body must not contain a `dataflow.graph` definition (a graph
    definition is a sibling at module scope, not a body element).
    A `dataflow.graph.launch` is the only way to invoke a graph
    callable from inside a thread definition's body.
  - Body must not contain a `dataflow.thread` definition or a
    `dataflow.thread.launch`; thread definitions are module-scope
    siblings and launches are caller-side only. The launch verifier
    checks this restriction transitively through nested regions.
  - `dataflow.work.spawn` is legal only in a `DynamicWork` definition and
    outside every nested `dataflow.graph`; its operand type equals the
    designated work-item input.
  - InstructionCore code and `dataflow.graph.launch` ops are allowed in a
    thread body. An InstructionCore-only body with no graph launch is also
    legal; this verifier rule does not itself select AccCore execution.
  - Body may contain `llvm.call` or `func.call` only when the callee has
    been proven InstructionCore-legal or is scheduled for inlining before
    graph extraction. Body must not contain `llvm.func` or `func.func`
    definitions.
  - Reachability is a pipeline invariant, not a local verifier rule. The
    verifier may accept an unreferenced private definition as dead IR, but
    finalized program publication removes unreachable private symbols.

* `dataflow.thread.launch` (Section 5.4.2)
  - `callee` resolves to a `dataflow.thread` definition in the
    same module (verifier rejects unresolved or wrong-kind callee).
  - `bodyOperands` types equal `callee.function_type.inputs`
    position-by-position.
  - For a dense callee, `extents` count equals coordinate rank. Every extent
    has `index` type. Statically known negative extents are rejected; runtime
    values are checked before instance creation. Dense rank zero creates one
    instance and any zero extent creates none. A dynamic callee has no extents
    and the designated body operand supplies exactly one root work item.
  - The op always produces exactly one `!dataflow.thread_token`
    result for collective retirement of all logical instances. Dynamic work
    retires only after its active responsibility set is empty.
  - Must appear outside every `dataflow.thread` and `dataflow.graph`
    definition, including through nested regions.

* `dataflow.thread.yield`
  - Accepts zero or more `none` operands as an unordered all-of
    completion frontier. The parent `dataflow.thread` definition has
    no data results; the per-launch completion token is produced by
    the launch op, not yielded as a body value. The verifier checks
    only frontier operand types and terminator placement.
  - Parent op must be a `dataflow.thread` definition (enforced by
    `ParentOneOf<["::dataflow::ThreadOp"]>`).
  - In a dynamic definition, the terminator retires the current item exactly
    once after its completion frontier; it does not close a channel or retire
    the collective token while another responsibility remains active.

* `dataflow.work.spawn`
  - Must appear transitively inside one `DynamicWork` thread and outside every
    graph. Dense threads, host code, and graph bodies reject it.
  - Its one operand type equals the definition's designated work-item input.
  - It is effectful, has no result or target, and acquires responsibility
    before making the child visible.

* `dataflow.thread.wait`
  - At least one operand. Each is `!dataflow.thread_token` produced
    by a `dataflow.thread.launch`.
  - Must appear outside every `dataflow.thread` and `dataflow.graph`
    definition, including through nested regions.
  - The op has no SSA result and therefore produces no graph-control
    `none` value. It is an ordered stored-program causal wait, not a
    memory barrier.

* `dataflow.graph` (definition, Section 5.5.1)
  - The op is a Symbol-bearing, function-like callable; it must
    be a direct child of a `ModuleOp` (`HasParent<"ModuleOp">`).
  - `sym_name` is required and module-unique among
    `dataflow.graph` definitions and other Symbol-bearing ops in
    the same module.
  - `sym_visibility` is required and must equal `"private"` in the
    baseline visibility policy. `"public"` and `"nested"` are rejected
    unless cross-module linkage is enabled by a separate spec.
  - `function_type` inputs are `(T0..TN)` and results are `(R0..RM)`, containing
    only application payloads. Normalized `input_segments` and
    `result_segments` classify value, stream, and memory ports. The graph
    start and launch done endpoints are not function-type slots.
  - The graph definition's body is `IsolatedFromAbove`: every SSA
    value used in the body and defined outside it is rejected.
  - Entry block arguments are `(%ctrl_in : none, %arg_0 : T0, ...,
    %arg_N : TN)`: the trailing arguments mirror `function_type.inputs`, while
    `%ctrl_in` is the explicit start protocol endpoint.
  - The body's `dataflow.graph.return` terminator has `values`, `streams`,
    `memories`, and mandatory non-empty `complete` segments. Concatenated
    payload segments match all `function_type.results`. Done is not a return
    payload or function-type slot.
  - Every finalized actor resolves to exactly one registered
    `OperationSchemaId` and passes its
    `CanonicalDataflowActorOpInterface` instance verifier. The derived
    canonical actor classifier consumes this registry for compute, control,
    and memory actors; it is not a separate whitelist. Lowering does not infer
    actor support from dialect or operation names.
  - A registered LLVM-dialect compute operation is eligible only through that
    same interface and only when it has explicit SSA operands and results, no
    regions or successors, no hidden memory, control, ABI, or runtime state,
    deterministic typed per-firing semantics, no unrecorded DataLayout
    dependency, no ordinary LLVM pointer graph value, and explicit semantic
    parameters. This is the sole LLVM exception; registered arithmetic, math,
    scalar, and vector compute actors remain legal under the same contract.
    An LLVM operation that is an exact semantic alias of an available standard
    `arith` or `math` actor is non-canonical and must have been normalized
    before graph finalization. Exact fused FMA uses `math.fma`; a non-fused
    multiply-add remains two explicit actors.
  - Residual imperative LLVM surface is forbidden. This includes calls and
    unresolved intrinsics, inline assembly, loads, stores, atomics, fences,
    allocation and pointer manipulation, branches, switches, PHI nodes,
    memory-copy or memory-set operations, and ABI, exception, stack, or
    runtime operations. Supported source forms must be normalized into
    canonical actors and explicit event networks before finalization.
  - Body must not contain `scf.*`, `llvm.func`, `func.func`, `llvm.call`,
    `func.call`,
    `dataflow.thread.launch`, `dataflow.graph.launch`,
    `dataflow.thread.wait`, `dataflow.graph.wait`,
    another `dataflow.graph` definition, or a `dataflow.thread`
    definition.
  - The op declares `RecursiveMemoryEffects` so module-scope
    walkers can observe per-callable effects. Launch completion is still
    defined only by the explicit return frontier.

* `dataflow.graph.launch` (Section 5.5.2)
  - `callee` resolves to a `dataflow.graph` definition in the
    same module (verifier rejects unresolved or wrong-kind callee).
  - Operand and result segments bind mechanically to the callee's normalized
    value, stream, and memory segments. Stream ports bind channel endpoints.
  - The mandatory trailing `done : none` result is the retirement protocol
    endpoint and equals `all_of(callee.graph.return.complete)`. No effect scan or
    quiescence rule provides an alternate completion authority.
  - The op must appear inside a `dataflow.thread` definition's
    body, not at host scope and not inside another
    `dataflow.graph` definition's body.
  - The launch does not reconstruct completion from callee effects. Native
    finalization validates that the explicit return frontier covers every
    observable effect before mapping or simulation.

* `dataflow.graph.wait` (Section 5.5.3)
  - Accepts a non-empty unordered all-of frontier of `none` completion events
    and has no result.
  - Must be transitively contained by exactly one `dataflow.thread` definition
    and must not appear at host scope or inside a `dataflow.graph` definition.
  - Finalized-program validation requires every operand to be a direct graph
    `done` result or a path-aware terminal event whose causal closure contains
    one. Generic `none` values and textual order are insufficient.
  - It is a stored-program graph-retirement wait, not a memory barrier, channel
    drain, thread-token conversion, or alternate completion authority.

* `Dataflow_GraphReturnOp`
  - `complete` is non-empty, variadic, unordered all-of, and `none`-typed.
  - `values`, `streams`, and `memories`, in that order, match the parent
    payload result types.
  - A single completion witness with no stream or memory outputs may use the
    compact `%complete, %values...` syntax; all other shapes print named
    segments.

## 10. Non-Goals

The following are explicitly out of scope for the scf-to-dfg
contract:

* Binding `dataflow.thread` directly to a `fabric.module` symbol. The thread
  remains target-independent software IR; SystemMapping binds logical thread
  execution to an AccCore, while TechMapping and SpatialMapping realize each
  selected graph on that AccCore's SpatialCore. The thread is already isolated
  and has an explicit boundary operand list.
* Native `dataflow.thread` data results, async value types, thread
  groups, and thread-level aggregation regions. Part 2 must materialize any
  supported tensor-result aggregation into accepted effect form before thread
  promotion; a residual aggregation form fails finalizability.
* LLVM IR provider integration, source-language integration, and clang
  embedding. Those concerns belong to Part 1 and Part 2.
* Logical-domain-point to fabric-resource binding and neighborhood
  communication or distributed-buffer protocols. These are not part of this
  contract. In particular, this spec does not commit to a stencil-specific
  neighbor-exchange op or a default mapping from a logical coordinate to any
  `fabric.pe` or `fabric.mem` instance.
* Channel routing or thread-endpoint simulation. Graph publication preserves
  typed stream input/output bindings and `source_map` while mechanically
  converting region-local endpoints to the canonical graph stream network.
  One binding owns one ordered dynamic event sequence: sequential and
  structured mutually exclusive static sites share a fixed ordinal schedule,
  with inactive choice sites filtered from that sequence. Repeated launch
  instances concatenate their contributions in deterministic issue order;
  producer and consumer events correspond by flat sequence ordinal, not by a
  one-to-one activation relation. Publication does not invent routing,
  endpoint creation, a parallel channel mode, or a traversal order for
  ambiguous parallel endpoint sites.

## 11. References

* `docs/spec-fabric-module.md`, `docs/spec-fabric-pe.md`,
  `docs/spec-fabric-fu.md` -- Fabric hardware semantics consumed by Mapping,
  not embedded in the Canonical Dataflow Program.
* `docs/spec-compiler-part-1-source.md` -- high-level source
  integration and metadata emission.
* `docs/spec-compiler-part-2-scf.md` -- LLVM-to-SCF raising and structured
  thread-boundary preparation.
* `docs/spec-compiler-part-3-mem.md` -- recursive graph-region memory
  lowering, basic alias partitions, write/read frontier transfers, and
  structured recurrence used inside each `dataflow.graph`.
* `docs/spec-core-dialect-boundary.md` -- compiler, Dataflow, Fabric, Mapping,
  and runtime ownership outside the canonical graph ABI.
* `docs/spec-mapping-artifact.md`, `docs/spec-mapping-memory.md`, and
  `docs/spec-pnr.md` -- TechMapping, canonical-memory realization, and
  Spatial/System physical realization after canonical graph publication.
* `docs/spec-compiler-part-4-partitioned-data.md` -- canonical logical launch
  domains, source-IV reconstruction, ordinary data views, and the physical
  Mapping boundary.
* `docs/spec-dataflow-part-1-streaming.md` -- precise timing
  semantics for `dataflow.stream`, `dataflow.carry`,
  `dataflow.invariant`, and `dataflow.gate`.
* `docs/spec-dataflow-part-2-control.md` -- precise firing semantics
  for `dataflow.constant`, `dataflow.sync`, `dataflow.mux`, and
  `dataflow.demux`.
* Upstream MLIR references (LLVM `externals/llvm/mlir/...`):
  - `Dialect/SCF/IR/SCFOps.td`.
  - `Dialect/Async/IR/AsyncOps.td`,
    `Dialect/Async/IR/AsyncTypes.td`.
  - `Dialect/GPU/IR/GPUOps.td`, `Dialect/GPU/IR/GPUBase.td`.
  - `Dialect/OpenMP/IR/OpenMPOps.td`,
    `Dialect/OpenACC/IR/OpenACCOps.td`.
  - `Conversion/SCFToGPU/SCFToGPU.cpp`.
