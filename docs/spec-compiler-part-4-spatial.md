# Loom Compiler Part 4: Spatial Array

This document specifies the spatial-array side of the Loom compiler
front-end: how a `memref<...>` is annotated with a tile-and-mesh
layout and how in-thread code queries its local tile. These concerns
are first-principles for SPGPU-style spatial GPUs, but they are not
first-principles for SCF to DFG flattening. Part 3 stays focused on
flattening structured control flow into dataflow primitives; this
part owns the spatial-array story. A general neighborhood
communication / distributed-buffer protocol -- which would cover
halo / stencil neighbor exchange among other patterns -- is reserved
for future work (see §6).

The canonical IR source is `include/Dataflow/IR/DataflowOps.td`; the
verifier implementation lives in `lib/Dataflow/IR/DataflowOps.cpp`.
This document is the design-level companion for compiler specs that use
these ops, especially `docs/spec-compiler-part-3-dfg.md`.

## 1. Scope

Part 4 owns:

* The annotation contract that lets a `memref<...>` carry a
  tile-and-mesh layout description.
* In-thread queries for the calling thread instance's local data range,
  spatial coordinates, and flattened spatial id.

Part 4 does not own:

* The HostCore-to-AccCore boundary or the `dataflow.thread` shape.
  Those belong to Part 3.
* The structured-control-to-dataflow flattening rules. Those belong to
  Part 3.
* The fabric-side hardware mapping. Layout-aware fabric lowering is a
  later concern that consumes the annotations defined here.

## 2. Annotation Approach

The first milestone takes the annotation route, preserving
`memref<...>` as the data carrier and attaching layout information
through `dataflow.spatial_layout`.

* The annotation is zero-cost: the result type equals the source
  type, so existing memref-aware passes do not need updates.
* In-thread queries go through `dataflow.local_range`,
  `dataflow.spatial_coord`, and `dataflow.spatial_linear_id`.
* Neighborhood communication between tiles, including halo /
  stencil neighbor exchange, is deferred. No dedicated neighbor-
  exchange op is part of this milestone; the first milestone
  intentionally does not commit to a stencil-specific op signature
  and instead leaves any such protocol to a follow-up that is
  driven by real workloads (see §6).
* The `dataflow.mesh @M { shape = [...] }` symbol-form mesh is
  deferred. All meshes in the first milestone are inline arrays on
  `dataflow.spatial_layout`.
* A future milestone may promote the annotation to a strong-typed
  carrier (`!dataflow.spatial_array<...>`); the in-thread queries'
  signatures are designed to remain stable across that change (only
  the source type widens).

## 3. New Operations (signatures only)

Each op below is given by its TableGen-level signature: arguments,
results, regions, traits. Implementation bodies are out of scope for
this spec.

### 3.1 `dataflow.spatial_layout`

```
arguments:
  AnyMemRef:$source,
  DenseI64ArrayAttr:$tileDims,
  DenseI64ArrayAttr:$meshShape,
  ArrayAttr:$splitAxes;
results:
  AnyMemRef:$annotated;
traits:
  Pure,
  AllTypesMatch<["source", "annotated"]>.
```

* Zero-cost annotation, modelled on `shard.shard`. The result type
  equals the input type.
* `tileDims` describes the per-dim tile size on the AccCore mesh.
  The first milestone restricts every entry to a power-of-two no
  smaller than the cache line, matching the SPGPU paper's prototype
  constraint; the verifier rejects other values.
* `meshShape` is a small inline array; the symbol-form mesh
  (`dataflow.mesh @M { ... }`) is deferred to a later milestone.
* `splitAxes` follows the `shard.sharding` convention: a length-N
  array (where N is the source memref rank) of arrays of mesh-axis
  indices; the empty inner array means "fully replicated on this
  data dim".
* Halo / ghost-cell metadata is intentionally not present in this
  milestone. It is reserved for whichever future neighborhood
  communication / distributed-buffer protocol Part 4 eventually
  adopts (see §6); pinning attribute fields without a consumer
  would freeze a shape that the consumer might want to change.

### 3.2 `dataflow.local_range`

```
arguments:
  AnyMemRef:$source,
  IndexAttr:$dim;
results:
  Index:$lo,
  Index:$hi;
traits:
  Pure.
```

* Returns the half-open `[lo, hi)` range of indices on
  data-dim `dim` owned by the calling thread instance.
* The op must appear inside a `dataflow.thread` body, possibly
  nested inside `scf.*` regions of that thread body. The thread's
  `mapping` must reach the mesh axes of the rooted
  `dataflow.spatial_layout` so that the local range query has a
  well-defined answer; this part of the contract is unchanged.
* The `source` operand is required to root in a
  `dataflow.spatial_layout` result through the recursive provenance
  chain defined below. The `local_range` op need not be lexically
  adjacent to the `dataflow.spatial_layout`: the chain is the
  binding mechanism, not lexical proximity.

#### 3.2.1 Source provenance chain

The chain is evaluated relative to a *provenance context*: the
chain scope where the candidate `%v` is currently being checked.
Initially the context is the lexical scope that holds the
`local_range` op. Crossing a `dataflow.thread` boundary moves the
context outward to the enclosing scope; the recursion may eventually
reach host scope, which is also a valid terminating context.

A value `%v` is an admissible `local_range` source iff at least one
of the following cases holds in the current provenance context:

1. **Direct match.** `%v` is the result of a `dataflow.spatial_layout`
   op in the current provenance context. By §4 of this document, a
   `dataflow.spatial_layout` op may appear at host scope or inside a
   `dataflow.thread` body. Both placements are admissible terminating
   contexts: when the chain has unwound to either scope and `%v` is a
   `dataflow.spatial_layout` result there, the chain terminates.
   Recursive descent through `scf.*` regions of the same scope is
   transparent.

2. **One thread boundary.** `%v` is the entry block argument at
   position `i` of a `dataflow.thread` body, where `i` indexes the
   body-operand segment of the entry block argument list (the segment
   that follows the leading `thread_ctrl` argument and the per-grid-dim
   induction-variable arguments, in declaration order). Let `%w` be
   the body operand at element `i` of that thread op's `bodyOperands`
   segment in the enclosing context (this is a positional index into
   the `bodyOperands` ODS segment of `dataflow.thread`, not an absolute
   op operand index, so it ignores any non-body operand groups such as
   async dependencies or dynamic grid bounds). Per Part 3 §5.4.1 / §9,
   every memref-like body operand must be a `dataflow.map_info` result.
   Let `%s` be the `source` operand of that `dataflow.map_info`. Case 2
   is satisfied iff `%s` is itself an admissible source by the same
   recursive rule, evaluated in the enclosing context.

3. **Nested thread boundaries.** Apply rule 2 recursively. Each
   thread layer crossed must satisfy the `dataflow.map_info`
   passthrough at that layer; the recursion terminates at a
   `dataflow.spatial_layout` result by rule 1, in either host scope
   or a thread body, whichever is reached first.

The verifier walks this chain explicitly. There is no type-level
marker on the boundary memref: by Part 3 §5.4.5, `dataflow.map_info`
preserves the source type unchanged, so the in-thread block argument
at index `i` has exactly the same memref type as the outer
`dataflow.spatial_layout` source. Same-type passthrough is therefore
not an annotation that travels with the value; it is a property the
verifier reconstructs by walking SSA edges.

#### 3.2.2 Why a chain rather than a type marker

The first milestone deliberately keeps `dataflow.spatial_layout` as
a zero-cost annotation: its result type equals its source type (see
§3.1 and §2). Two alternative type-level designs were considered and
rejected for this milestone:

* A boundary / provenance wrapper such as `!loom.mapped<T>` (which
  was originally proposed for the `dataflow.thread` boundary; dropped
  in commit `fef88b9`). Its job would have been to mark "this memref
  came from a `dataflow.map_info`" so a downstream verifier could
  type-check provenance.
* A spatial-layout carrier such as `!dataflow.spatial_array<...>`
  that would replace the source memref type at the
  `dataflow.spatial_layout` result. Its job would have been to carry
  the layout annotation in the type system rather than in an SSA edge
  to the layout op.

Both options would have let `local_range` admit any operand whose
type is the marker. Neither was adopted in this milestone, in order
to keep the annotation transparent to existing memref-aware passes
and to match the `shard.shard` precedent. As a consequence, the
binding from `local_range` back to its rooting
`dataflow.spatial_layout` must be expressed as an SSA chain rather
than as a single type check. A future milestone that adopts the
strong-typed spatial carrier (see §6) may collapse the chain check
to a type check, but the in-thread query signature stays the same.

### 3.3 `dataflow.spatial_coord` and `dataflow.spatial_linear_id`

```
spatial_coord:
  arguments: none;
  results: Variadic<Index>:$coords;

spatial_linear_id:
  arguments: none;
  results: Index:$id;
```

* Both must appear inside a `dataflow.thread` body. They report the
  current thread instance's coordinate vector (`spatial_coord`) or
  flattened identifier (`spatial_linear_id`), based on the
  `#loom.spatial<...>` mapping of the enclosing thread.
* These ops let the inner ScalarCore code reason about its own
  position without depending on entry-block argument ordering, which
  is convenient for templated lowerings.

## 4. Verifier Rules

* `dataflow.spatial_layout`
  - `tileDims` rank equals source memref rank.
  - Every `tileDims` entry is a power of two and not less than the
    cache-line size (the verifier reads the cache-line constant from
    a Loom-wide config; a wrong value yields a clear diagnostic).
  - `splitAxes` outer length equals source memref rank; every inner
    array entry is a valid mesh-axis index.
  - `meshShape` is a non-empty positive integer vector.
  - May appear at host scope or inside a `dataflow.thread` body
    (the ScalarCore portion of a thread). The verifier rejects it
    inside `dataflow.graph`. When the annotated memref crosses a
    `dataflow.thread` boundary, it does so as the source of a
    `dataflow.map_info`; the annotation stays on the producing
    `dataflow.spatial_layout` op rather than on the boundary memref
    type. Because `dataflow.map_info` passes its source type through
    unchanged, the in-thread block argument has the same type as the
    outer source, but no type-level marker carries the annotation.
    Verifiers that need to recover the annotation walk the SSA chain
    back through `dataflow.map_info` to the rooting
    `dataflow.spatial_layout` (see §3.2.1).

* `dataflow.local_range`, `dataflow.spatial_coord`,
  `dataflow.spatial_linear_id`
  - Must be inside a `dataflow.thread` body. The verifier rejects
    them at host scope or inside `dataflow.graph`. Their results are
    per-thread-instance constants; if a `dataflow.graph` body needs
    those values, they must be computed in the surrounding
    ScalarCore code and passed in as ordinary graph operands.
  - For `dataflow.local_range`, the verifier additionally checks the
    source provenance chain defined in §3.2.1. Starting from the
    `source` operand, it unwinds entry-block-argument-to-body-operand
    edges across each enclosing `dataflow.thread` and follows the
    `dataflow.map_info` passthrough at each layer until it reaches a
    `dataflow.spatial_layout` result, or rejects the op if no such
    chain exists. The thread mapping that the query reads must
    reach the mesh axes of that rooted `dataflow.spatial_layout`.
  - A ScalarCore-callable `func.func` is a module-level symbol and
    is lexically outside any `dataflow.thread` body, so it cannot
    contain these query ops directly. A ScalarCore-callable helper
    that needs a spatial query must be inlined or specialized into
    the active thread context before verifier / finalization runs.
    This matches the ScalarCore-call handling Part 3 already
    requires for callees that contain code which must become
    `dataflow.graph` or nested `dataflow.thread` (see Part 3 §2.1).

## 5. Interaction with Part 3

* Spatial-array ops are not part of the SCF-to-DFG flattening
  templates in Part 3 and do not appear inside a `dataflow.graph`
  body. `dataflow.spatial_layout` is a memref annotation emitted at
  host scope or inside the ScalarCore portion of a thread body. The
  query ops `dataflow.local_range`, `dataflow.spatial_coord`, and
  `dataflow.spatial_linear_id` appear only inside thread bodies.
  Their results are per-thread-instance constants; when a graph
  body needs them, they are computed in ScalarCore and passed in
  through ordinary graph operands.
* Spatial-array values still cross the HostCore-to-AccCore boundary
  through the same `dataflow.map_info` protocol that Part 3 defines
  for any other memref-shaped boundary value. The annotation lives
  on the producing `dataflow.spatial_layout` op; because
  `dataflow.map_info` passes its source type through unchanged, the
  in-thread block argument has the same memref type as the outer
  source but carries no type-level marker. Verifiers that need to
  recover the annotation walk the SSA chain (see §3.2.1).
* `dataflow.thread` mapping (`#loom.spatial<...>` /
  `#loom.temporal<...>`) is the first-class binding from grid dim to
  physical core-grid coordinate. The spatial-array ops use that
  binding to compute local ranges and coordinates; they do not
  introduce a parallel hardware-mapping mechanism.

## 6. Future Design Thoughts

* Strong-typed `!dataflow.spatial_array<...>` carrier. The in-thread
  queries above keep their signatures; only the source type widens.
* Symbol-form `dataflow.mesh @M { shape = [...] }` so that several
  layouts can refer to the same mesh declaration.
* Neighborhood communication / distributed-buffer protocol. Halo /
  stencil neighbor exchange is an important spatial pattern, but
  the first milestone deliberately does not introduce a dedicated
  stencil-specific op such as a hypothetical `dataflow.halo_exchange`.
  Pinning a software-visible op without a concrete consumer would
  freeze a shape that real workloads might need to change, which is
  at odds with the Occam's-razor principle this spec follows. In the
  first milestone the required behavior is expressed through explicit
  memory effects on annotated memrefs, runtime-managed transfers, or
  later fabric-level transport primitives. A future design should
  start from a general neighborhood communication / distributed-buffer
  abstraction -- one that covers stencil halos as a special case
  alongside other neighbor-exchange shapes -- rather than from a
  stencil-specific adhoc op. Whatever that future design looks like,
  it would also be the right time to add any halo / ghost-cell
  metadata to `dataflow.spatial_layout` that its consumer needs.
* Optional analysis to refine `dataflow.map_info` direction based on
  read/write effects on spatial-array operands; this is the same
  optimizer extension point Part 3 already documents for ordinary
  memrefs.

## 7. References

* `docs/spec-compiler-part-3-dfg.md` -- SCF-to-DFG lowering.
  Part 3 defines `dataflow.thread`, `dataflow.graph`, and
  `dataflow.map_info`; Part 4 extends the dialect with the
  spatial-array ops listed here.
* `docs/spec-dataflow-part-1-streaming.md`,
  `docs/spec-dataflow-part-2-control.md` -- streaming and control
  primitive semantics. Spatial-array ops do not change those
  semantics; they appear as ordinary in-thread or in-graph ops.
* `include/Dataflow/IR/DataflowOps.td` -- canonical operation
  definitions.
* `lib/Dataflow/IR/DataflowOps.cpp` -- verifier implementation.
* Upstream MLIR references (LLVM `externals/llvm/mlir/...`):
  - `Dialect/Shard/IR/ShardOps.td`,
    `Dialect/Shard/IR/ShardBase.td`.
  - `Dialect/SCF/IR/DeviceMappingInterface.td`.
