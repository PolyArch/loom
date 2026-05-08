# Loom Compiler Part 4: Spatial Array

This document specifies the spatial-array side of the Loom compiler
front-end: how a `memref<...>` is annotated with a tile-and-lattice
layout, and how in-thread code queries its local tile. These concerns
are first-principles for SPGPU-style spatial accelerators, but they
are not first-principles for SCF to DFG flattening. Part 3 stays
focused on flattening structured control flow into dataflow
primitives; this part owns the spatial-array story. A general
neighborhood communication / distributed-buffer protocol -- which
would cover halo / stencil neighbor exchange among other patterns --
is reserved for future work (see §7).

> **Non-assumption.** Part 4's lattice is a logical, software-side
> partition domain. Part 4 does NOT assume any default mapping from
> lattice cells to fabric resources. A fabric backend whose physical
> PE / memory graph is not a Cartesian mesh -- an irregular fabric
> built out of arbitrary `fabric.pe`, `fabric.mem`, `fabric.switch`,
> and `fabric.fifo` instances -- is supported by Part 4 without
> modification. The binding from logical lattice cell to physical
> fabric instance is a separate concern; see §7 for where the
> binding lives.

The canonical IR source is `include/Dataflow/IR/DataflowOps.td`; the
verifier implementation lives in `lib/Dataflow/IR/DataflowOps.cpp`.
This document is the design-level companion for compiler specs that use
these ops, especially `docs/spec-compiler-part-3-dfg.md`.

## 1. Scope

Part 4 owns:

* The declaration of a logical partition lattice via
  `dataflow.mesh @M { shape = [...] }`.
* The annotation contract that lets a `memref<...>` carry a
  tile-and-lattice layout description.
* In-thread queries for the calling thread instance's local data
  range, lattice coordinates, and flattened lattice cell id.

Part 4 does not own:

* The HostCore-to-AccCore boundary or the `dataflow.thread` shape.
  Those belong to Part 3.
* The structured-control-to-dataflow flattening rules. Those belong to
  Part 3.
* The lattice-cell to fabric-resource binding. Layout-aware fabric
  lowering is a later concern that consumes the annotations defined
  here. Part 4 stays topology-agnostic; binding is the only place
  where fabric topology enters the picture.

## 2. Annotation Approach

The first milestone takes the annotation route, preserving
`memref<...>` as the data carrier and attaching layout information
through `dataflow.spatial_layout` rooted at a logical mesh declared
by `dataflow.mesh`.

* The lattice is declared once with `dataflow.mesh @M { shape = [...] }`.
  Multiple layouts and threads in the same module reference `@M`.
  Multiple meshes per module are allowed when a kernel partitions
  more than one independent index space.
* The annotation is zero-cost: the `dataflow.spatial_layout` result
  type equals the source memref type, so existing memref-aware
  passes do not need updates.
* In-thread queries go through `dataflow.local_range`,
  `dataflow.spatial_coord`, and `dataflow.spatial_linear_id`. Each
  query reports per-instance values relative to the logical lattice
  the rooted layout references.
* Neighborhood communication between tiles, including halo /
  stencil neighbor exchange, is deferred. No dedicated neighbor-
  exchange op is part of this milestone; the first milestone
  intentionally does not commit to a stencil-specific op signature
  and instead leaves any such protocol to a follow-up that is
  driven by real workloads (see §7).
* A future milestone may promote the annotation to a strong-typed
  carrier (`!dataflow.spatial_array<...>`); the in-thread queries'
  signatures are designed to remain stable across that change (only
  the source type widens).

## 3. New Operations (signatures only)

Each op below is given by its TableGen-level signature: arguments,
results, regions, traits. Implementation bodies are out of scope for
this spec.

### 3.1 `dataflow.mesh`

```
arguments:
  SymbolNameAttr:$sym_name,
  DenseI64ArrayAttr:$shape;
results:
  none;
regions:
  none;
traits:
  Symbol.
```

* `dataflow.mesh @M { shape = [d_0, d_1, ..., d_{k-1}] }` declares a
  logical partition lattice with rank `k` and per-axis size
  `d_0, ..., d_{k-1}`. All entries are positive integers.
* The lattice is a software index space, not a hardware coordinate
  system. It defines:
  - **Lattice rank.** `k = len(shape)`. There is no closed cap on
    rank; a kernel that needs more than three axes simply declares
    a `shape` with the corresponding length.
  - **Per-axis size.** `d_i` is the number of cells along axis `i`.
  - **Stable cell identity.** Each cell has a stable row-major
    `cell_id` derived from its coordinate vector
    `(c_0, c_1, ..., c_{k-1})` with `c_i ∈ [0, d_i)`. The flatten
    convention is row-major over the declared shape:
    `cell_id = c_0 * (d_1 * d_2 * ... * d_{k-1}) + c_1 * (d_2 * ...) + ... + c_{k-1}`.
    This identity is observable to programs through
    `dataflow.spatial_linear_id` (see §3.4) and is reusable by any
    future binding artefact, so two independent consumers do not
    invent two divergent identities for the same cell.
* `dataflow.mesh` carries no fabric-binding information. There is
  **no default rule** that maps cell `(i, j)` to a `fabric.pe` named
  `(i, j)` or to any other fabric instance. A future binding spec
  (see §7) is responsible for picking an assignment from logical
  cells to fabric resources.
* `dataflow.mesh` is a top-level symbol (declared at module scope
  alongside `func.func` and `fabric.module`). It implements
  `Symbol`; references to it use `SymbolRefAttr`.
* Multiple `dataflow.mesh` declarations are allowed in the same
  module. Each is independent; layouts and threads must reference
  one explicitly when more than one is in scope.

### 3.2 `dataflow.spatial_layout`

```
arguments:
  AnyMemRef:$source,
  DenseI64ArrayAttr:$tileDims,
  SymbolRefAttr:$mesh,
  ArrayAttr:$splitAxes;
results:
  AnyMemRef:$annotated;
traits:
  Pure,
  AllTypesMatch<["source", "annotated"]>,
  DeclareOpInterfaceMethods<SymbolUserOpInterface>.
```

* Zero-cost annotation, modelled on `shard.shard`. The result type
  equals the input type.
* `dataflow.spatial_layout` is a view-like alias of its source
  memref: the result is `Pure` and shares storage identity with
  `source` for alias-analysis purposes. The `MemAliasOracle` walk
  in `docs/spec-compiler-part-3-mem.md` §3.1 peels
  `dataflow.spatial_layout` like the other recognized view-like
  ops, so a leaf access on the `annotated` result roots in the
  same storage identity as a leaf access on `source`. The op
  itself is side-effect-free; any `MemoryEffectOpInterface`
  projection performed on a `dataflow.thread` body operand whose
  matching `dataflow.map_info` source is the `annotated` result
  reads through to the underlying `source` memref's effects, by
  the same passthrough rule that `dataflow.map_info` uses (see
  Part 3 §5.4.5 and §9 thread verifier rules).
* `tileDims` describes the per-data-dim tile size, expressed in
  number of elements along each data dim. Each entry is a positive
  integer. Part 4 imposes no further numeric constraint: tile
  power-of-two alignment, cache-line alignment, and other
  fabric-prototype constraints are not first-principles properties
  of a logical layout and live with the binding spec instead (see
  §7). Per-fabric backends that need such constraints validate
  them at binding time, not against `dataflow.spatial_layout`.
* `mesh` is a `SymbolRefAttr` referring to a `dataflow.mesh @M`
  declaration. The lattice rank `k = len(mesh.shape)` defines the
  number of valid axes for `splitAxes` entries.
* `splitAxes` follows the `shard.sharding` convention: a length-N
  array (where N is the source memref rank) of arrays of lattice-
  axis indices; the empty inner array means "fully replicated on
  this data dim". Each axis index must lie in `[0, k)`. The same
  axis may not be referenced twice across `splitAxes` (an axis
  cannot simultaneously partition two data dims).
* Halo / ghost-cell metadata is intentionally not present in this
  milestone. It is reserved for whichever future neighborhood
  communication / distributed-buffer protocol Part 4 eventually
  adopts (see §7); pinning attribute fields without a consumer
  would freeze a shape that the consumer might want to change.

### 3.3 `dataflow.local_range`

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
  data-dim `dim` owned by the calling thread instance, expressed
  in the source memref's index space along that data dim.
* `dim` is required to be in `[0, rank(source))`, where
  `rank(source)` is the rank of the `source` memref. A `dim` outside
  this half-open range is a verifier diagnostic.
* The op must appear inside a `dataflow.thread` body, possibly
  nested inside `scf.*` regions of that thread body. The thread's
  `mapping` must reach the lattice axes of the rooted
  `dataflow.spatial_layout` so that the local range query has a
  well-defined answer. **Reach** is defined precisely: for every
  lattice-axis index `a` referenced by `splitAxes` of the rooted
  `dataflow.spatial_layout`, there is at least one entry in the
  enclosing thread's `mapping` array of attribute kind
  `#loom.spatial<...>` whose `axis` equals `a` and whose resolved
  `lattice` symbol equals the layout's `mesh` symbol. Lattice
  axes that `splitAxes` does not reference (a fully replicated
  data dim) do not need to be reached; lattice axes referenced by
  `splitAxes` but not covered by any matching `#loom.spatial<...>`
  mapping entry on the thread cause the verifier to reject the
  `local_range` query. Temporal mapping entries
  (`#loom.temporal<...>`) do not contribute to reach. This rule
  is the same as the previous milestone modulo the open mapping
  ID and the symbol-form mesh: `mappingId == m` becomes
  `(axis, resolved-lattice) == (a, layout.mesh)`.
* The `source` operand is required to root in a
  `dataflow.spatial_layout` result through the recursive provenance
  chain defined below. The `local_range` op need not be lexically
  adjacent to the `dataflow.spatial_layout`: the chain is the
  binding mechanism, not lexical proximity.

#### 3.3.1 Source provenance chain

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

#### 3.3.2 Why a chain rather than a type marker

The first milestone deliberately keeps `dataflow.spatial_layout` as
a zero-cost annotation: its result type equals its source type (see
§3.2 and §2). Two alternative type-level designs were considered and
rejected for this milestone:

* A boundary / provenance wrapper such as `!loom.mapped<T>` (an
  earlier proposal for the `dataflow.thread` boundary, not adopted
  in this milestone). Its job would have been to mark "this memref
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
strong-typed spatial carrier (see §7) may collapse the chain check
to a type check, but the in-thread query signature stays the same.

### 3.4 `dataflow.spatial_coord` and `dataflow.spatial_linear_id`

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
  flattened cell id (`spatial_linear_id`) within the logical
  partition lattice, based only on the `#loom.spatial<...>` entries
  of the enclosing thread's `mapping`.
* **Single-lattice query contract.** Both queries are op-local
  rules: when present, they require the enclosing thread's
  `#loom.spatial<...>` entries to resolve to **exactly one**
  lattice (per Part 3 §9 lattice resolution). A `spatial_coord` or
  `spatial_linear_id` op inside a thread whose spatial entries
  resolve to more than one distinct lattice is rejected with an
  ambiguous-mapping diagnostic; such a kernel must (a) split the
  no-operand query into separate threads, one per lattice, or (b)
  use `dataflow.local_range` instead, whose `source` memref operand
  uniquely roots the relevant lattice via the §3.3.1 chain. The
  same-lattice constraint applies to the query op, not to the
  enclosing thread; multi-lattice threads remain legal as long as
  they do not host these no-operand queries (`local_range` works
  unchanged in such threads, see §3.3 reach rule).
* **Resolved lattice.** When the single-lattice query contract
  holds, the queries report against the resolved lattice `@M`. In
  the single-lattice case, `#loom.spatial<...>` entries may omit
  the optional `lattice` qualifier; the resolved lattice is the
  unique mesh reached by the spatial-array layouts in the thread
  body. In a multi-lattice scope (the body reaches more than one
  distinct mesh), every `#loom.spatial<...>` entry must carry the
  explicit qualifier (per Part 3 §9), and the spatial entries'
  qualified symbols must agree on a single mesh for the query op
  to be admissible.
* `spatial_coord` returns one `index` result per `#loom.spatial<...>`
  entry of the enclosing thread's `mapping`, in mapping-array order.
  `#loom.temporal<...>` entries do not contribute to the result
  vector. The result vector length therefore equals the count of
  spatial mapping entries, not the total grid rank. A thread with
  no spatial mapping entries is rejected for hosting a
  `spatial_coord` op (the result would be an empty vector with no
  meaningful query).
* **`spatial_linear_id` axis-coverage contract.** `spatial_linear_id`
  requires the enclosing thread's `#loom.spatial<...>` entries to
  cover every axis of the resolved lattice **exactly once**, in any
  order. The verifier rejects a thread whose spatial entries miss
  an axis or duplicate an axis. The flatten then proceeds in
  axis-index order (NOT mapping-array order): the coordinates are
  reordered into `(c_0, c_1, ..., c_{k-1})` -- where `c_i` is the
  coordinate of the spatial entry whose `axis == i` -- and the
  result is the row-major flatten defined for `dataflow.mesh` cell
  identity in §3.1. The result equals the resolved lattice's
  `cell_id` for the calling instance regardless of the
  programmer's mapping-array order. Temporal mapping entries do
  not enter the linearization. The flatten convention is
  observable to programs and is reusable by future binding
  artefacts (see §7).
* `spatial_coord`, by contrast, does **not** require complete
  axis coverage and does not reorder. A thread that maps a
  proper subset of lattice axes is legal for `spatial_coord` (it
  simply returns a partial vector); `spatial_linear_id` is
  rejected for the same thread because no `cell_id` is well-
  defined without complete coverage.
* These ops let the inner ScalarCore code reason about its own
  position without depending on entry-block argument ordering, which
  is convenient for templated lowerings.

## 4. Verifier Rules

* `dataflow.mesh`
  - `shape` is a non-empty positive-integer vector. Each entry is a
    positive integer; `shape` length is the lattice rank `k` and
    must be at least one.
  - `sym_name` participates in the enclosing `SymbolTable`; the
    standard MLIR symbol-table verifier enforces uniqueness.
  - `dataflow.mesh` is a top-level op (declared at module scope).
    Nested placement is rejected; references to `@M` are made via
    `SymbolRefAttr`.
  - `dataflow.mesh` carries no fabric-binding info; the verifier
    does not consult fabric IR or impose any topology constraint.
    Every `dataflow.mesh` is admissible against every `fabric.module`
    in the same compilation unit at this layer.

* `dataflow.spatial_layout`
  - `tileDims` rank equals source memref rank; every entry is a
    positive integer. The verifier imposes no further numeric
    constraint -- in particular, **no** power-of-two requirement and
    **no** cache-line minimum at this layer. Such constraints, if
    needed by a specific fabric backend (e.g., the SPGPU-style Bank
    Selection Unit), are validated at binding time and live with the
    binding spec, not with Part 4 (see §7).
  - `mesh` resolves to a `dataflow.mesh @M` symbol that is in scope
    of the layout op. Unresolved or wrong-kind symbols are rejected
    via `SymbolUserOpInterface::verifySymbolUses`.
  - `splitAxes` outer length equals source memref rank. Every inner
    array entry is a valid lattice-axis index in `[0, k)`, where
    `k = len(@M.shape)`. The same axis must not appear twice across
    `splitAxes` (an axis cannot simultaneously partition two data
    dims).
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
    `dataflow.spatial_layout` (see §3.3.1).

* `dataflow.local_range`, `dataflow.spatial_coord`,
  `dataflow.spatial_linear_id`
  - Must be inside a `dataflow.thread` body. The verifier rejects
    them at host scope or inside `dataflow.graph`. Their results are
    per-thread-instance constants; if a `dataflow.graph` body needs
    those values, they must be computed in the surrounding
    ScalarCore code and passed in as ordinary graph operands.
  - For `dataflow.spatial_coord`, the result vector length must
    equal the count of `#loom.spatial<...>` entries in the enclosing
    thread's `mapping` array (per §3.4); the verifier rejects a
    `spatial_coord` op inside a thread whose `mapping` has no
    `#loom.spatial<...>` entry, since the result vector would be
    empty.
  - For `dataflow.spatial_linear_id`, the enclosing thread's
    `mapping` must contain at least one `#loom.spatial<...>` entry;
    the linear id is the row-major flattening of the spatial-only
    coordinate vector against the resolved lattice's shape, using
    the cell-identity convention defined in §3.1.
  - For `dataflow.spatial_coord` and `dataflow.spatial_linear_id`,
    all `#loom.spatial<...>` entries on the enclosing thread must
    resolve to a single lattice (per §3.4 single-lattice query
    contract). Resolution is per-entry (Part 3 §9): each entry
    independently produces a `(kind, lattice, axis)` triple, and
    the query op is admissible iff the spatial entries' resolved
    `lattice` symbols agree on a single mesh. The verifier rejects
    a `spatial_coord` / `spatial_linear_id` op whose enclosing
    thread's spatial entries resolve to more than one distinct
    lattice; the diagnostic names the candidate meshes. The
    enclosing thread itself remains legal (other ops in the same
    thread, including `dataflow.local_range`, are unaffected by
    this op-local rule).
  - For `dataflow.spatial_linear_id`, the resolved lattice's
    spatial entries must cover every axis of the lattice exactly
    once. A missing or duplicated axis is rejected with a
    diagnostic that names the gap or duplicate. `spatial_coord`
    does not have this requirement (it returns a partial vector
    when only a subset of axes is mapped).
  - For `dataflow.local_range`, the verifier additionally checks the
    source provenance chain defined in §3.3.1. Starting from the
    `source` operand, it unwinds entry-block-argument-to-body-operand
    edges across each enclosing `dataflow.thread` and follows the
    `dataflow.map_info` passthrough at each layer until it reaches a
    `dataflow.spatial_layout` result, or rejects the op if no such
    chain exists. The thread mapping that the query reads must
    reach the lattice axes of that rooted `dataflow.spatial_layout`
    (per §3.3 reach definition). The `dim` attribute must lie in
    `[0, rank(source))`; the verifier rejects an out-of-range
    `dim`. Multi-lattice bodies are unambiguous for `local_range`
    because the `source` operand uniquely identifies the rooted
    layout (and hence its lattice) via the §3.3.1 chain.
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
  recover the annotation walk the SSA chain (see §3.3.1).
* `dataflow.thread` mapping (`#loom.spatial<...>` /
  `#loom.temporal<...>`) is the first-class binding from grid dim to
  logical lattice axis. The spatial-array ops use that binding to
  compute local ranges and lattice coordinates; they do not
  introduce a parallel hardware-mapping mechanism. The mapping
  attribute does not commit to a fabric topology by itself; any
  mapping from logical lattice cells to physical fabric resources
  is supplied by a separate binding artefact (see §7).

## 6. Testing Strategy

Spatial-array unit tests follow the same lit-test layout pattern that
`docs/spec-compiler-part-3-impl.md` §2 establishes for Part 3 dialect
elements. Tests live under `test/frontend/unit/spatial/`, with one
subdirectory per spatial op: `mesh/`, `spatial_layout/`,
`local_range/`, `spatial_coord/`, and `spatial_linear_id/`. Each
subdirectory holds `valid.mlir`, `invalid.mlir`, and
`roundtrip.mlir` (the last confirming printer / parser stability).

* `mesh/`. valid.mlir covers a 1-D mesh, a 2-D mesh, a higher-rank
  mesh (e.g., rank 4 to exercise the absence of a closed cap), and
  multiple mesh declarations side by side in the same module.
  invalid.mlir pins each §4 `dataflow.mesh` verifier rule: an empty
  `shape`, a `shape` with a non-positive entry, two declarations
  sharing the same `sym_name` (rejected by the symbol-table
  verifier), and a nested-rather-than-top-level placement.
* `spatial_layout/`. valid.mlir covers a layout op at host scope, a
  layout op inside a `dataflow.thread` body, a layout op inside
  `scf.*` regions nested in a thread body, a 1-D-mesh case (single
  lattice axis, single split axis), and a non-power-of-two `tileDims`
  entry whose acceptance demonstrates that Part 4 no longer enforces
  the SPGPU prototype constraint (the constraint is relocated to
  binding-time validation per §7). invalid.mlir pins each §4
  `dataflow.spatial_layout` verifier rule on at least one fixture:
  a layout op inside a `dataflow.graph` body (rejected per §1 and
  §4); a layout op operating on a non-memref source; a layout op
  whose `tileDims` rank does not equal the source memref rank; a
  layout op whose `tileDims` contains a non-positive entry; a
  layout op whose `splitAxes` outer length does not equal the
  source memref rank; a layout op whose `splitAxes` references an
  axis index that is out of `[0, k)` for the resolved
  `dataflow.mesh @M`; a layout op whose `splitAxes` references the
  same axis index twice; a layout op whose `mesh` symbol is
  unresolved; and a layout op whose `mesh` symbol resolves to a
  non-`dataflow.mesh` op kind.
* `local_range/`. valid.mlir exercises each provenance-chain rule
  from §3.3.1 separately: rule 1 with the layout in the same thread
  body as the query (direct match); rule 2 with the layout at host
  scope, the value crossing one `dataflow.thread` boundary, and the
  body operand at the matching position produced by
  `dataflow.map_info`; rule 3 with at least one chain length covering
  two thread layers, so the verifier walk is exercised through more
  than one `dataflow.map_info` passthrough; and a case with `scf.*`
  regions nested between the layout op and the query inside the same
  thread body, confirming that scf nesting is transparent. valid.mlir
  also includes a multi-lattice fixture: the same `dataflow.thread`
  body hosts two `dataflow.spatial_layout` ops that reference two
  distinct `dataflow.mesh @M0` and `@M1`, with corresponding
  `#loom.spatial<axis, @M0>` and `#loom.spatial<axis, @M1>` entries
  in the thread's `mapping`, and two separate `dataflow.local_range`
  queries (one per layout). The fixture asserts that the §3.3.1
  chain disambiguates each query's rooted lattice without rejecting
  the multi-lattice thread. invalid.mlir covers a
  `local_range` whose source does not chain to any
  `dataflow.spatial_layout` (chain walk fails); a `local_range`
  outside any `dataflow.thread` body (rejected per §4); a
  `local_range` whose enclosing thread mapping does not reach the
  lattice axes of the rooted `dataflow.spatial_layout`; a chain that
  crosses a `dataflow.thread` boundary where the body operand at the
  matching position is not a `dataflow.map_info` result, so rule 2
  fails; a `local_range` whose `dim` attribute is out of range for
  the source memref rank (per §4 verifier rule); and a `local_range`
  inside a thread whose mapping reaches more than one lattice
  without explicit `lattice` qualifiers on each spatial entry (the
  rejection happens at the Part 3 §9 thread-mapping verifier, rule
  iii, before the `local_range` op itself fails -- this fixture
  pins the upstream diagnostic).
* `spatial_coord/` and `spatial_linear_id/`. Both ops take no
  arguments per §3.4, so verifier coverage is purely placement and
  result-shape oriented. valid.mlir places the query inside a
  `dataflow.thread` body, in both a non-nested thread body and a
  nested-thread body, and confirms that `spatial_coord` returns a
  result vector whose length equals the count of `#loom.spatial<...>`
  entries in the enclosing thread's `mapping` (temporal entries do
  not contribute, per §3.4). valid.mlir also covers a thread where
  every `#loom.spatial<...>` entry carries an explicit `lattice`
  qualifier referring to the same single lattice (positive case
  for the optional qualifier). invalid.mlir covers a query at host
  scope, a query inside a `dataflow.graph` body (rejected per §1
  and §4), a query inside an `scf.*` region that is itself outside
  any thread body, a query inside a thread whose `mapping` contains
  no `#loom.spatial<...>` entry (the empty-result case is rejected
  per §3.4), a `spatial_coord` op inside a thread with `K` spatial
  mapping entries whose declared variadic result count is not `K`
  (the verifier rejects an out-of-arity result list), and a query
  inside a thread whose spatial entries resolve to more than one
  distinct lattice (rejected per §3.4 single-lattice query
  contract). Because Part 3 §9 rule iii already requires explicit
  qualifiers in any thread reaching multiple meshes, this fixture
  is built with explicit `<axis, @M0>` / `<axis, @M1>` entries
  whose qualifiers themselves disagree -- the §3.4 query-op rule
  is the one that fires.
* `spatial_linear_id/` additionally pins:
  - The row-major flatten convention from §3.1: a fixture that
    asserts the result for a known coordinate vector and lattice
    shape, so the cell-identity contract is observable from tests
    (e.g., a `lit` test that constant-folds the linear id when
    coords are SSA constants, or a runtime smoke test).
  - The axis-coverage contract from §3.4: a positive fixture
    where `#loom.spatial<...>` entries cover every lattice axis
    exactly once but in non-monotonic order (e.g.,
    `[#loom.spatial<1>, #loom.spatial<0>]` against a 2-D lattice),
    asserting that the result still flattens in axis-index order;
    a negative fixture missing one axis; a negative fixture
    duplicating one axis.
* Cross-cutting graph rejection. Each subdirectory's invalid.mlir
  also pins a small block where the spatial op appears inside a
  `dataflow.graph` body and is rejected. No spatial op may appear
  inside `dataflow.graph` (cross-ref §1 Scope and §4 Verifier
  Rules); the per-subdirectory cases keep that single rule
  observable from each op's own test surface.

The `local_range` provenance-chain coverage is the discipline that
makes the verifier walk testable: rule 1, rule 2, and rule 3 are
exercised on separate IR fixtures rather than collapsed into one
maximal example, and at least one rule-3 fixture has chain length
greater than or equal to two thread layers so the walk traverses
more than one `dataflow.map_info`. Negative chain cases are kept
small and distinct so the diagnostic surface stays mechanical.

Integration tests for spatial-array kernels (matmul-like, stencil-
like, and other SPGPU / Chapel-style idioms) live with the Part 3
integration suite under `test/frontend/integration/` and exercise
the chain in a realistic kernel context; the unit tests here only
pin individual op verifier rules.

## 7. Future Design Thoughts

* **Lattice-cell to fabric-resource binding.** Part 4 defines the
  logical lattice (`dataflow.mesh @M`) and a stable cell identity
  (row-major `cell_id`, §3.1), but commits to **no default mapping**
  from a lattice cell to a `fabric.pe`, `fabric.mem`, or any other
  fabric instance. A binding artefact -- a programmer-supplied table
  for irregular fabrics, an algebraic formula for SPGPU-style
  Cartesian-mesh fabrics, or a placement-pass output -- is required
  for end-to-end execution. The home of the binding spec is to be
  decided. Plausible options include extending
  `docs/spec-compiler-part-3-placement-framework.md` with an L4
  binding tier, or a new sibling `docs/spec-compiler-part-5-binding.md`.
  Whatever home is chosen, the binding spec is responsible for:
  - the exchange format that lists, for each lattice cell, the
    fabric instance(s) it co-locates with;
  - the legality contract against the Part 4 lattice (cell ids and
    splitAxes references must round-trip);
  - the validation of fabric-prototype constraints relocated out of
    Part 4 (see next bullet).
* **Relocated SPGPU prototype constraints.** Earlier drafts of Part 4
  enforced that every `tileDims` entry was a power of two and not
  smaller than the cache line. That rule is the SPGPU paper's Bank
  Selection Unit address-arithmetic constraint, not a property of a
  logical layout. It was removed from Part 4 verifier rules so that
  fabrics whose physical PE / memory graph is not a Cartesian mesh
  are not silently rejected. The constraint reappears as a
  binding-time check in any future SPGPU-style fabric-binding spec
  (see the previous bullet); Part 4 itself stays agnostic.
* **Strong-typed `!dataflow.spatial_array<...>` carrier.** The
  in-thread queries above keep their signatures; only the source
  type widens.
* **Neighborhood communication / distributed-buffer protocol.**
  Halo / stencil neighbor exchange is an important spatial pattern,
  but the first milestone deliberately does not introduce a dedicated
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
* **Optional `MemoryEffect` refinement.** Optional analysis to
  refine `dataflow.map_info` direction based on read/write effects
  on spatial-array operands; this is the same optimizer extension
  point Part 3 already documents for ordinary memrefs.

## 8. References

* `docs/spec-compiler-part-3-dfg.md` -- SCF-to-DFG lowering.
  Part 3 defines `dataflow.thread`, `dataflow.graph`, and
  `dataflow.map_info`; Part 4 extends the dialect with the
  spatial-array ops listed here.
* `docs/spec-compiler-part-3-impl.md` §2 Testing Strategy --
  lit-test layout pattern (`test/frontend/unit/`,
  `test/frontend/lower_scf/`, `test/frontend/integration/`) and
  per-op `valid.mlir` / `invalid.mlir` / `roundtrip.mlir`
  conventions that §6 of this document mirrors.
* `docs/spec-dataflow-part-1-streaming.md`,
  `docs/spec-dataflow-part-2-control.md` -- streaming and control
  primitive semantics. Spatial-array ops do not change those
  semantics; they appear as ordinary thread-body ops, never inside
  `dataflow.graph` (per §1 and §4).
* `include/Dataflow/IR/DataflowOps.td` -- canonical operation
  definitions.
* `lib/Dataflow/IR/DataflowOps.cpp` -- verifier implementation.
* Upstream MLIR references (LLVM `externals/llvm/mlir/...`):
  - `Dialect/Shard/IR/ShardOps.td`,
    `Dialect/Shard/IR/ShardBase.td`.
  - `Dialect/SCF/IR/DeviceMappingInterface.td`.
