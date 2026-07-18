# Loom Dataflow Vector Boundary

This document specifies Loom's canonical one-dimensional dataflow vector
boundary. The four boundary operations are `dataflow.parallelize`,
`dataflow.serialize`, `dataflow.pack`, and `dataflow.unpack`.

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

* vector load and store operations;
* gather and scatter operations;
* software-to-Fabric port adaptation;
* Fabric memory masks;
* PnR routing for vector ports;
* vectorization search or DSE policy.

## Semantic Ownership

`include/Dataflow/IR/DataflowOps.td` defines the canonical operation
signatures. `lib/Dataflow/IR/DataflowOps.cpp` verifies type and shape
invariants.

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

## Graph Cardinality

Graph validation treats `pack` and `unpack` as one-token-in,
one-token-out actors. Their result is statically exact-one when their input is
statically exact-one.

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
* packed integer widths other than `N * bitwidth(T)`.
