# Loom Dataflow Vector Semantics

This document specifies Loom's canonical one-dimensional dataflow vector
semantics. It covers the four stream and representation boundary operations
`dataflow.parallelize`, `dataflow.serialize`, `dataflow.pack`, and
`dataflow.unpack`, plus contiguous vector access through `dataflow.load` and
`dataflow.store` and rank-1 gather access through `dataflow.load`.

The semantic data and mask types are standard MLIR `vector<NxT>` and
`vector<Nxi1>`. Those types are the only source of vector length, shape,
element type, and mask shape. The operations have no independent vector-size
attribute.

## Scope

The boundary supports fixed-size rank-1 vectors with `N > 0`. `T` is a
nonzero-width MLIR integer or floating-point type. Scalable vectors,
rank-zero vectors, and vectors with rank greater than one are illegal.

Materialization of arbitrary-rank vectors remains unspecified. No consumer
may infer a flattening order for such vectors.

Arithmetic, comparison, and math over semantic vectors use standard MLIR
`vector`, `arith`, and `math` operations. The dataflow boundary does not
define duplicate vector compute operations.

The following concerns are outside this contract:

* ranked vectors beyond rank one;
* vector-address scatter operations;
* duplicate scatter-address policy;
* software-to-Fabric port adaptation;
* Fabric memory masks;
* PnR routing for vector ports;
* alignment, burst, and coalescing policy;
* atomic, read-modify-write, fence, and volatile semantics;
* source or structured-control vectorization and lowering;
* vectorization search or DSE policy.

## Semantic Ownership

`include/Dataflow/IR/DataflowOps.td` defines the canonical operation
signatures. `DataflowActorSemantics` owns fixed-rank-1 data-vector legality,
mask compatibility, and the scalar-versus-vector memory-access type relation.
Operation verification and DFG simulation consume that shared contract.

Scalar/group firing and reset rules are owned by
`DataflowActorSemantics`. Graph cardinality validation and DFG simulation
consume that shared contract. They must not define independent rules for
which phase consumes data, when a group is published, or when actor state
resets.

`dataflow.pack` and `dataflow.unpack` are stateless exact-one actors.
`dataflow.parallelize` and `dataflow.serialize` are phase adapters with
variable true-token cardinality and exactly one terminal false token per
activation.

`dataflow.pack` and `dataflow.unpack` are the only canonical
vector-to-integer representation boundary. An `llvm.bitcast` between a
vector and an integer is not a canonical Dataflow actor. Other legal LLVM
bitcasts are outside this representation rule.

Standard elementwise `arith` and `math` vector actors use the same scalar
primitive semantics independently for each lane. The simulator accepts only
fixed-size rank-1 vectors whose shapes and scalar element types are compatible
with that primitive. Unsupported vector forms fail before consuming tokens.

## `dataflow.parallelize`

The canonical signature is:

```mlir
%vector, %mask, %group_phase =
  dataflow.parallelize %data, %scalar_phase
    : (T, i1) -> (vector<NxT>, vector<Nxi1>, i1)
```

A true scalar phase consumes one scalar data token and appends it at the next
lane in ascending lane order. A false scalar phase consumes no data token.

A full group is published immediately. Publication emits one data vector,
one active-lane mask, and one true group phase token. For an activation with
`K` true scalar items, the number of true groups is:

```text
G = ceil(K / N)
```

When false arrives with a pending partial group, the actor first publishes a
zero-filled data vector, its active-lane mask, and a true group phase token.
It then publishes one false group phase token. When `K = 0`, it publishes
only the false group phase token. In both cases it clears all pending lanes
and returns to its initial state.

For every published vector, mask lane `i` is true exactly when lane `i`
contains an active scalar item. Inactive data lanes are zero-filled using the
element type's all-zero bit representation.

## `dataflow.serialize`

The canonical signature is:

```mlir
%data, %scalar_phase =
  dataflow.serialize %vector, %mask, %group_phase
    : (vector<NxT>, vector<Nxi1>, i1) -> (T, i1)
```

A true group phase consumes exactly one data vector and one mask vector. It
visits lanes in ascending order and publishes one scalar data token followed
by one true scalar phase token for each active lane. It does not publish a
false token after an individual true group.

An all-zero mask on a true group is legal. The actor consumes both vectors
and publishes no scalar tokens for that group.

A false group phase consumes neither vector. It publishes one false scalar
phase token and resets the activation.

Together, `parallelize` and `serialize` preserve active scalar item order and
activation boundaries.

## `dataflow.pack`

The canonical signature is:

```mlir
%packed = dataflow.pack %vector : vector<NxT> -> iM
```

`M` must equal `N * bitwidth(T)`. The operation consumes one vector token and
publishes one integer token. Lane zero occupies bits
`[bitwidth(T) - 1 : 0]`; lane `i` occupies the next higher bit slice.

Integer elements preserve their exact bits. Floating-point elements use
their exact MLIR floating-point bit representation, including signed zero,
infinities, and NaN payloads. Packing does not perform numeric conversion.

A mask is not an implicit operand. When a packed mask is required, the
`vector<Nxi1>` value is passed through a separate `dataflow.pack`.

## `dataflow.unpack`

The canonical signature is:

```mlir
%vector = dataflow.unpack %packed : iM -> vector<NxT>
```

The width and lane-order rules are identical to `dataflow.pack`.
`dataflow.unpack` consumes one integer token and publishes one vector token.
For every legal type pair:

```text
unpack(pack(vector)) = vector
pack(unpack(bits)) = bits
```

The bit representation uses arbitrary-width integers. Host 64-bit integer
width is not a semantic limit.

## Rank-1 Vector Memory

The existing `dataflow.load` and `dataflow.store` operations support scalar
and fixed-rank-1 contiguous access. `dataflow.load` additionally supports
fixed-rank-1 gather access. They remain the only canonical plain
memory-access mnemonics.

A scalar access uses one `%addr : index` and returns or stores one `T`. A
contiguous vector access uses one `%addr : index` and a `vector<NxT>` value;
lane `i` addresses memory element `%addr + i`. A gather load instead uses
`%addresses : vector<Nxindex>` and returns `vector<NxT>` with the same `N`;
result lane `i` reads the memory element named by address lane `i`.

The vector forms are:

```mlir
%data, %done = dataflow.load %mem[%addr] %ctrl
    : memref<?xT>, vector<NxT>
%data, %done = dataflow.load %mem[%addr] %ctrl mask %mask
    : memref<?xT>, vector<NxT>
%data, %done = dataflow.load %mem[%addresses] %ctrl
    : memref<?xT>, vector<Nxindex>, vector<NxT>
%data, %done = dataflow.load %mem[%addresses] %ctrl mask %mask
    : memref<?xT>, vector<Nxindex>, vector<NxT>

%done = dataflow.store %mem[%addr] %data %ctrl
    : memref<?xT>, vector<NxT>
%done = dataflow.store %mem[%addr] %data %ctrl mask %mask
    : memref<?xT>, vector<NxT>
```

The optional mask type is `vector<Nxi1>` with exactly the data vector's
shape. A gather address must be a fixed-size rank-1 `vector<Nxindex>` with
the same `N`. Scalar accesses reject masks. Omitting the mask makes every
lane active. A data type exactly equal to the memref element type remains a
scalar element access even when that element type is itself a vector; that
scalar access also rejects masks.

Only active lanes evaluate an element address or access memory. An inactive
lane may therefore correspond to an out-of-range element address. A masked
load fills every inactive result lane with the element type's all-zero bit
representation. A masked store leaves every inactive memory element
unchanged.

An all-zero mask still consumes the firing's address, mask, and control
operands, plus store data when present. It performs no memory access. A load
publishes one zero-filled vector and one done token; a store publishes one
done token.

Load data and done become visible together after all active lanes have read
their elements. Store done becomes visible only after all active lanes have
written their elements. Each firing publishes exactly one load data token
and one done token, or exactly one store done token. The existing explicit
`ctrl` to `done` event network remains the sole memory-ordering authority.

DFG simulation visits active lanes in ascending lane-index order against its
element-indexed abstract memory. Contiguous access evaluates `%addr + i`;
gather access evaluates address lane `i`. Repeated gather addresses are
legal and preserve result lane order. Vector index tokens place lane zero in
the least-significant bit slice and derive each lane's width from the
enclosing MLIR DataLayout or the existing configured index-width fallback.
This ordering is deterministic functional semantics, not a burst,
coalescing, port-width, or hardware-lane policy.

These accesses are plain, non-atomic, and non-volatile. Ranked vectors beyond
rank one, alignment policy, and physical memory-mask projection are not part
of this boundary. Vector-address store and scatter semantics remain
unimplemented and are explicitly rejected.

## Graph Cardinality

Graph validation treats `pack` and `unpack` as one-token-in,
one-token-out actors. Their result is statically exact-one when their input is
statically exact-one.

Scalar and vector `load` and `store` use the existing canonical-actor
cardinality rule. Load data and done, or store done, are statically exact-one
when every dynamic operand is statically exact-one. A mask changes lane
activity inside one firing; it does not change actor token cardinality or
create a separate ordering network.

The group phase from `parallelize` closes when its scalar phase closes. The
scalar phase from `serialize` closes when its group phase closes. Retirement
coverage may project a downstream false close through either adapter to the
corresponding upstream close only when graph analysis proves all payloads
required by every true input phase are aligned to that phase. This requires
scalar data for `parallelize`, and both a data vector and mask vector for
`serialize`. Without those proofs, finalized graph validation fails closed.

Data vectors and masks from `parallelize` have one token for each true group
phase and no token for the false group phase. `serialize` requires both
vectors only for a true group phase. These rules are also the simulator's
firing requirements.

## Verification

Verification rejects:

* non-vector semantic data or masks;
* scalable or non-rank-1 vectors;
* non-integer and non-floating-point element types;
* zero-width integer elements;
* scalar and vector element-type mismatches;
* mask shapes that differ from the data vector shape;
* mask element types other than `i1`;
* packed integer widths other than `N * bitwidth(T)`;
* vector memory element types that differ from the memory element type;
* masks on scalar memory accesses;
* gather address vectors that are scalable, not rank one, or do not contain
  `index` elements;
* gather address or mask lane counts that differ from the load result;
* vector addresses on `dataflow.store`.
