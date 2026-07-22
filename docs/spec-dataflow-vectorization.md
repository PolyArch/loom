# Loom Dataflow Vector Semantics

This document owns the semantic vector contract of a Canonical Dataflow
Program. It defines fixed ranked vector values and masks, vector stream and
bit-representation boundaries, vector compute, and vector forms of the
canonical memory actors.

Structured vectorization, schedule selection, and `P[]`/`N[]` exploration are
owned by the compiler front-end specifications. Physical lane realization,
port-width adaptation, and transport encoding are owned by TechMapping,
SpatialMapping, and the interconnect implementation. Those layers consume this
contract; they do not redefine it.

## Semantic Ownership

This specification is the semantic authority. Dataflow ODS declarations,
operation verifiers, `DataflowActorSemantics`, simulators, and Mapping importers
are implementation projections of it. They must share one implementation of
the rules where practical and must not become independent authorities.

Vector shape, element type, and mask shape come only from standard MLIR types.
There is no independent vector-size or lane-count attribute.

The canonical forms are:

```text
vector<N0 x ... x Nk x T>
vector<N0 x ... x Nk x i1>
```

Every dimension is fixed and greater than zero. The first form is a semantic
data value and the second is its same-shape active-lane mask. Scalable and
rank-zero vectors are outside this contract. Data elements are nonzero-width
MLIR integer or floating-point values; gather and scatter addresses use
`index` elements.

Axes follow the selected scheduled-dimension order. Row-major flattening is
canonical: the last axis varies fastest. Flattened lane zero is the first
logical element and occupies the lowest bit slice when a semantic vector is
packed.

## Vector Compute

Arithmetic, comparison, selection, shape, and math over semantic vectors use
standard MLIR `arith`, `vector`, and `math` operations. Dataflow does not add
duplicate vector-compute mnemonics.

An elementwise operation applies the corresponding scalar primitive to every
lane with identical operand and result shapes. A total, side-effect-free,
non-trapping operation may evaluate inactive lanes and forward the mask as an
explicit value. An operation that may trap or has effects requires explicit
predication. A mask is never hidden state attached to a vector value.

The canonical functional result is independent of a physical realization.
TechMapping may realize one vector actor with a vector FU, several scalar
actors, or a hybrid implementation, but it must preserve shape, lane order,
mask behavior, and token publication.

## Stream Cardinality Boundary

`dataflow.parallelize` and `dataflow.serialize` are the only canonical actors
that convert between an ordered scalar stream and fixed-width vector groups.
They use one-dimensional vectors because they group one linear stream. This
does not restrict the rank of vectors produced directly by structured
vectorization.

### `dataflow.parallelize`

```mlir
%vector, %mask, %group_phase =
  dataflow.parallelize %data, %scalar_phase
    : (T, i1) -> (vector<NxT>, vector<Nxi1>, i1)
```

A true scalar phase consumes one scalar data token and appends it at the next
lane in ascending lane order. A false scalar phase consumes no data token.

A full group is published immediately. Publication emits one data vector, one
active-lane mask, and one true group phase token. For an activation with `K`
true scalar items:

```text
G = ceil(K / N)
group_phase = T^G F
```

When false arrives with a pending partial group, the actor first publishes a
zero-filled vector, its active-lane mask, and a true group phase. It then
publishes one false group phase and resets. When `K = 0`, it publishes only the
false group phase. Inactive lanes use the element type's all-zero bit
representation.

### `dataflow.serialize`

```mlir
%data, %scalar_phase =
  dataflow.serialize %vector, %mask, %group_phase
    : (vector<NxT>, vector<Nxi1>, i1) -> (T, i1)
```

A true group phase consumes exactly one data vector and one mask. It publishes
one scalar data token followed by one true scalar phase for each active lane,
in ascending lane order. It does not publish a false phase after an individual
true group. An all-zero mask is legal and publishes no scalar item.

A false group phase consumes neither vector nor mask. It publishes one false
scalar phase and resets. Together, `parallelize` and `serialize` preserve the
order of active scalar items and activation boundaries.

These actors are semantic cardinality adapters. Physical serialization,
packetization, or a narrow Fabric port is not a reason to insert either actor.

## Bit-Representation Boundary

`dataflow.pack` and `dataflow.unpack` are stateless, exact-one semantic
bit-representation conversions. They are orthogonal to stream cardinality and
vector computation.

```mlir
%packed = dataflow.pack %vector
    : vector<N0x...xNkxT> -> iM
%vector = dataflow.unpack %packed
    : iM -> vector<N0x...xNkxT>
```

The width is exact:

```text
M = product(N0, ..., Nk) * bitwidth(T)
```

The vector is flattened in canonical row-major order. Flattened lane zero
occupies the least-significant bit slice and each following lane occupies the
next higher slice. Integer and floating-point elements preserve their exact bit
representations, including floating-point NaN payloads. Packing is not a
numeric conversion.

A mask is not implicit. A source-visible packed mask uses a separate
`dataflow.pack` on its `vector<...xi1>` value. For every legal type pair:

```text
unpack(pack(vector)) = vector
pack(unpack(bits)) = bits
```

The implementation uses arbitrary-width integers; a host integer width is not
a semantic limit. A source or Structured Program Candidate introduces these
actors only when the program observes the bit representation. Mapping-side
port adaptation and transport packing never introduce them into the Canonical
Dataflow Program.

## Vector Memory

`dataflow.load` and `dataflow.store` remain the only canonical plain memory
actors. Their address and data shapes distinguish scalar, contiguous vector,
and gather or scatter access:

```text
index                         + T                       -> scalar access
index                         + vector<S x T>           -> contiguous access
vector<S x index>             + vector<S x T>           -> gather/scatter
```

`S` denotes the complete fixed ranked shape. A scalar linear address names a
memory element. For contiguous access, lane `i` in canonical row-major order
accesses `base + i`. A gather or scatter address vector has the same shape as
the data vector, and each active lane accesses the element named by its address
lane.

Representative forms are:

```mlir
%data, %done = dataflow.load %mem[%base] %ctrl
    : memref<?xT>, vector<N0xN1xT>
%data, %done = dataflow.load %mem[%base] %ctrl mask %mask
    : memref<?xT>, vector<N0xN1xT>
%data, %done = dataflow.load %mem[%addresses] %ctrl mask %mask
    : memref<?xT>, vector<N0xN1xindex>, vector<N0xN1xT>

%done = dataflow.store %mem[%base] %data %ctrl mask %mask
    : memref<?xT>, vector<N0xN1xT>
%done = dataflow.store %mem[%addresses] %data %ctrl mask %mask
    : memref<?xT>, vector<N0xN1xindex>, vector<N0xN1xT>
```

The optional mask has the complete data-vector shape and `i1` elements.
Omitting it makes every lane active. Scalar accesses reject masks. A memory
whose element type is itself a vector still performs a scalar element access
when the actor data type exactly equals that element type.

Only active lanes evaluate addresses or access memory. Therefore an inactive
lane may contain an out-of-range address. A masked load fills inactive result
lanes with the element type's all-zero bit representation. A masked store
leaves inactive elements unchanged.

An all-zero mask still consumes the firing's address, control, and mask, plus
store data when present. It performs no memory access. A load publishes one
zero-filled vector and one done token; a store publishes one done token.

Load data and done become visible together after every active lane has read.
Store done becomes visible after every active lane has written. The explicit
`ctrl` to `done` event network remains the sole memory-ordering authority; a
mask does not create a second ordering mechanism.

Repeated gather addresses are legal and preserve result-lane order. For a
plain non-atomic scatter, duplicate active addresses are not assigned a hidden
lane order. The compiler must prove them distinct, scalarize the access under
an explicit program order, or reject it until an explicit ordered or atomic
semantic is available.

Alignment, burst formation, coalescing, physical port width, byte enables, and
bank selection do not change this software contract. They belong to lowering,
Mapping, and Fabric realization. A software lane mask to physical byte-enable
projection is mechanically derived and cannot become a second semantic owner.

## Cardinality And Ordered Execution

`pack` and `unpack` consume and publish exactly one token. A scalar or vector
`load` or `store` has the existing canonical memory-actor cardinality: its data
and done results are exact-one when all dynamic operands are exact-one. Lane
activity changes effects within one firing, not actor token cardinality.

The group phase from `parallelize` closes when its scalar phase closes. The
scalar phase from `serialize` closes when its group phase closes. Retirement
analysis may project a terminal false phase through an adapter only after
proving that every payload required by each true phase is aligned with it.

Physical vector lanes may execute internally out of order only if the complete
vector token, memory completion, or serialized scalar boundary restores the
canonical ordered result.

## Front-End And DSE Boundary

Structured vectorization is the primary compilation path. Regular loop nests
are vectorized while their iteration domains, dependences, reductions, tails,
and memory relations are explicit. They lower directly to vector-valued
actors; they are not first converted to scalar streams and regrouped with
`dataflow.parallelize`.

For a scheduled dimension `d`:

```text
P_d = graph-static actor replication
N_d = elements carried by one actor activation
logical width = P_d * N_d
d = chunk_base + p * N_d + n
chunk step = P_d * N_d
```

`P[]` and `N[]` are orthogonal. `P_i = 4, N_j = 8` means four actors each
process `vector<8xT>`. `N_i = 2, N_j = 8` means each actor processes
`vector<2x8xT>`. The Structured Program Candidate owns the selected schedule,
unroll and jam structure, vector factors, tail policy, reduction strategy, and
ownership boundaries. Mechanical SCF-to-Dataflow lowering does not select or
revise them.

Compiler Evaluation may use the resolved Fabric Hardware Description through
the central Evaluation interface to screen vector candidates. It may not run
hidden TechMapping or PnR, copy Fabric facts into software IR, or treat an
unsupported hardware realization as a change to vector semantics.

The detailed transform order, legality views, immutable candidate lineage, and
Dataflow-to-Dataflow optimization boundary are specified by the compiler
front-end documents. This specification owns only the resulting Dataflow
vector behavior.

## Mapping Boundary

The Canonical Dataflow Program preserves semantic vector shape and lane order.
TechMapping may flatten, split, or combine physical lanes and may bind them to
one or more physical endpoints. SpatialMapping routes only the residual edges
after a realization has absorbed its internal dependencies. The Mapping
Artifact records every physical representation and endpoint binding needed to
reconstruct the selected realization.

Physical adaptation is never represented by silently changing a semantic
vector type, inserting semantic `pack` or `serialize` actors, or deriving
meaning from an equal total bit width. A `vector<4xf32>` and a `vector<2xf64>`
remain different requirements even though both occupy 128 bits.

## Verification

Verification rejects:

* scalable or rank-zero vectors;
* zero vector dimensions;
* unsupported or zero-width element types;
* mask shapes that differ from their data-vector shapes;
* mask element types other than `i1`;
* packed integer widths that differ from the complete flattened bit width;
* scalar/vector operand or result shape mismatches;
* vector memory element types that differ from the memory element type;
* masks on scalar memory accesses;
* gather or scatter address shapes that differ from the data shape;
* gather or scatter address elements other than `index`;
* plain scatter operations whose active duplicate addresses are neither proven
  absent nor lowered to an explicit ordered form.

Stable anchor tests cover:

* rank-one partial-group `parallelize` and `serialize` behavior;
* exact and partial-tail activation closure;
* rank-one and multi-rank `pack`/`unpack` round trips, including floating-point
  payload bits;
* multi-rank contiguous and gather/scatter addressing in row-major lane order;
* inactive-lane address suppression and all-zero-mask completion;
* repeated gather addresses and rejected unresolved duplicate scatter
  addresses;
* rejection of attempts to encode physical port adaptation as semantic
  `pack`, `serialize`, or a changed vector type.
