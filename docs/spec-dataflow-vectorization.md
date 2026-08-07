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
rank-zero vectors are outside this Canonical Dataflow contract. Scalable
vectors remain legal in the S0 Structured Program Candidate. Before a selected
SpatialRegion finalizes, a typed structured transform must materialize their
exact semantics as fixed-width chunks, loops, and masks or tails. Failure to
do so makes that candidate non-finalizable; it does not make scalable vectors
globally illegal or silently choose InstructionCore ownership. Data elements
are nonzero-width MLIR integer or floating-point values; gather and scatter
addresses use `index` elements.

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
TechMapping may select a vector FU whose declared internal implementation uses
several lanes or beats, but it preserves one actor and every complete actor
port. Several canonical scalar or narrower-vector actors exist only after an
explicit Dataflow-to-Dataflow candidate rewrite. Both forms must preserve
shape, lane order, mask behavior, and token publication.

### Explicit Elementwise Decomposition

This transform is the single
`DataflowRewriteKind::ElementwiseVectorDecompose` rule owned by the
[Canonical Dataflow Rewrite Catalog](spec-compiler-part-3-dfg.md#canonical-dataflow-rewrite-catalog).
Its normalized decision has exactly two modes, `LeadingChunk(C)` and
`Scalarize`. The modes are not independent rewrite kinds or an implicit
Mapping fallback.

The Dataflow rewrite owner may derive narrower candidates from one exact pure,
total, regionless fixed-vector Compute actor with elementwise semantics. Every
operand and the one result have the same nonempty shape; element types may
differ, as for a vector comparison. Memory actors, reductions, stream or
cardinality actors, stateful actors, and operations with effects or potentially
trapping behavior are outside this transform.

The generated OperationSchema registry is the sole owner of whether a schema
has this pointwise decomposition semantic. The decision is attached to the
exact schema, including a selected instance of a generic carrier such as a
registered LLVM intrinsic; it is not inferred from an MLIR operation trait,
operation spelling, implementation-family membership, or provider
availability. After a typed rewrite changes an exact function type, the
OperationSchema owner mechanically regenerates any source-owned overloaded
selector spelling before canonical finalization.

One leading-chunk decision chooses a positive proper divisor `C` of the
leading dimension `N`. It materializes `N / C` actors with the same registered
operation schema and attributes at leading dimension `C`. Multi-operand input
tokens first cross one `dataflow.sync`, so no chunk can consume one
activation's operand while another operand remains pending. Standard
`vector.shuffle` actors select complete leading blocks and concatenate all
chunk results in ascending block order. The final result publishes only after
every chunk of that activation is available.

One scalarization decision materializes one scalar actor for every row-major
position, using static standard `vector.extract` and `vector.insert` actors.
It is legal only when an operand has the exact complete result-vector type, so
the insert chain has a typed destination while replacing every position. A
result whose element type differs from every operand, such as many comparison
results, has no scalarization decision under this rule; leading-chunk
decomposition remains available.

Both decisions preserve the exact scalar operation behavior, lane order,
mixed defined/poison/undef state, one atomic multi-operand activation, and one
complete result publication. They introduce no packed-integer interpretation,
hidden lane route, memory-access decomposition, or Mapping fact. Finalization
assigns all derived actor identities, and the result is a distinct immutable
Canonical Dataflow Artifact. Mapping may select it only through ordinary exact
actor, port, capability, and route admission.

### Fixed-Vector Structural Operations

Canonical Dataflow admits the standard MLIR `vector.extract`,
`vector.insert`, and `vector.shuffle` operations directly. They are typed
compute actors, not bit casts, stream-cardinality adapters, memory operations,
or aliases for target instructions.

`vector.extract` selects one scalar or one complete trailing subvector from a
source vector using a position over its leading dimensions. `vector.insert`
is the inverse update: it preserves every destination lane outside the
selected trailing block and replaces that block with the supplied scalar or
subvector. Static and dynamic position components retain the exact MLIR
semantics. A poison position produces a poison result; an out-of-range dynamic
position has the source operation's undefined result semantics rather than a
simulator- or hardware-defined wraparound.

`vector.shuffle` treats each leading-dimension element as one complete block.
It selects blocks from its two operands in mask order, may duplicate a source
block, and produces a poison block for a mask entry of `-1`. The operands and
result retain their exact standard vector types, including common trailing
shape and element type. A poison block poisons only its result lanes; it does
not relax defined sibling blocks.

The registered OperationSchema projection is the sole owner of static
positions and shuffle masks. Dynamic positions remain ordered actor operands.
No flattened byte offset, lane selector table, hardware mode, or scalarized
replacement is stored beside that projection in Canonical Dataflow.

A structural actor over `vector<index>` retains `index` as its canonical
semantic element type. Physical admission resolves every index lane and every
dynamic position through the exact target index-width projection; it may not
admit the actor as an arbitrary equal-width integer vector.

### Exceptional Lane State

A fixed vector semantic value is a tuple of lane states. Each lane is exactly
one of defined, poison, or undef; defined integer and floating lanes carry
their exact arbitrary-precision semantic value. Elementwise operations apply
their registered scalar operation schema per lane. Vector selection observes
only the selected lane value. An inactive masked-memory lane observes neither
its address nor its data and produces the specified defined zero fill for an
inactive load lane.

Exceptional state is not a physical bit pattern. The registered schemas for
`dataflow.pack`, `dataflow.unpack`, `dataflow.parallelize`, and
`dataflow.serialize` own their exact exceptional-state projection across
shape or cardinality boundaries. Every such relation must be closed and
verified before the actor is admitted. A simulator or semantic provider that
cannot represent the required mixed-lane state reports unsupported rather than
coercing poison or undef to zero. The bit-level round-trip equation below
applies to fully defined values; the explicit `pack` and `unpack` projection
below separately defines exceptional values and its exact identity domain.

A mask consumer observes each lane as an activity decision. Defined zero and
defined one mean inactive and active, respectively. Poison or undef does not
become an activity bit. An execution provider that cannot represent the
resulting non-singleton cardinality or effect relation must reject the firing
atomically as typed `Unsupported`; it may not choose a bit, consume operands,
change memory, or report a blocked wait set. The current exact DFG and CGRA
execution provide only the exact single-path activity model and therefore
reject such an exceptional mask firing.

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
`dataflow.pack` on its `vector<...xi1>` value.

For exceptional state, `pack` produces scalar poison when any vector lane is
poison, otherwise scalar undef when any lane is undef, and otherwise the
defined packed bits. `unpack` maps scalar poison to an all-poison vector,
scalar undef to an all-undef vector, and defined bits to the corresponding
all-defined lane values. It never invents per-lane defined bits for an
exceptional scalar.

Consequently, for every legal type pair, `pack(unpack(bits))` is an identity
over defined, poison, and undef scalar states. `unpack(pack(vector))` is an
identity exactly for all-defined vectors, all-poison vectors, and all-undef
vectors. A mixed vector such as `[defined, poison]` packs to scalar poison and
unpacks to an all-poison vector, so that composition is not an identity on the
general vector value-state domain:

```text
pack(unpack(bits)) = bits

unpack(pack(vector)) = vector
  iff vector is all-defined, all-poison, or all-undef
```

The implementation uses arbitrary-width integers; a host integer width is not
a semantic limit. A source or Structured Program Candidate introduces these
actors only when the program observes the bit representation. Mapping-side
port adaptation and transport packing never introduce them into the Canonical
Dataflow Program.

## Vector Memory

`dataflow.load` and `dataflow.store` are the canonical read and write actors.
This section owns their scalar, contiguous-vector, and gather/scatter geometry.
`docs/spec-dataflow-memory-consistency.md` separately owns their one typed
plain, atomic, and volatile access contract and the additional atomic actors.
The two specifications compose; neither copies the other's fields.

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
lane order. The compiler must prove them distinct or scalarize the access
under an explicit program order. A `PerLane` atomic access permits duplicate
active addresses under the addressed objects' modification orders and still
does not create a lane order.

Distinctness is re-proved from the finalized canonical actor, address
expressions, active-lane relation, and exact structured decision lineage. Loom
does not persist a proof object, `distinct=true` escape hatch, or Mapping-owned
legality bit. A nonvolatile plain scatter may be scalarized before Dataflow
finalization when the structured compiler can materialize one exact program
order. A volatile plain scatter cannot use that rewrite because several
provider-visible scalar operations are not the same observable operation as
one vector firing; it remains InstructionCore code or makes the selected
candidate non-finalizable. Repeated addresses remain legal for `PerLane`
atomics.

Alignment, burst formation, coalescing, physical port width, byte enables, and
bank selection do not change this software contract. They belong to lowering,
Mapping, and Fabric realization. A software lane mask to physical byte-enable
projection is mechanically derived and cannot become a second semantic owner.

Atomic source alignment is the actor-owned contract defined by
`docs/spec-dataflow-memory-consistency.md`. It is not one of the physical
formation choices listed above.

Consumers derive one nonpersistent `CanonicalMemoryAccessView` from the exact
actor and its types. The view has no independent identity or serialized
fields. It projects:

```text
operation       = load | store | atomic_rmw | cmpxchg
access_contract = exact actor-owned typed access contract
access_form     = element | contiguous | indexed
memory_element_type = exact memref element type
element_bits    = bit width of one complete memory element
lane_shape      = exact ranked access shape for contiguous or indexed
lane_count      = access_form == element ? 1 : product(lane_shape)
data_bits       = lane_count * element_bits
address_count   = indexed ? lane_count : 1
index_bits      = bit width of one address element under canonical data layout
address_bits    = address_count * index_bits
mask_form       = absent | dynamic
mask_bits       = mask_form == dynamic ? lane_count : 0
atomic_granularity = absent | whole_payload | per_lane
```

`element` means one complete memref element, even when that element is itself
a vector. Its `lane_shape` is absent and its lane count is one. It is therefore
distinct from a contiguous access with the same data type and total bit width.
The exact type, shape, operation kind, and access contract remain owned by
this Dataflow actor; flattened counts, widths, and normalized enum projections
are derived compatibility facts, not a replacement type system or serialized
record. `dataflow.fence` has no memory-addressed access view.

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
TechMapping may place the row-major flattened bits in one declared complete
physical-port representation and may select a Fabric realization that
internally decomposes lane or beat work. Each external actor port still
corresponds to one endpoint capable of carrying the complete token unless the
Canonical Dataflow Program contains an explicit semantic adapter.
SpatialMapping routes only the residual edges after a realization has absorbed
its internal dependencies. The Mapping Artifact records every physical
representation and endpoint binding needed to reconstruct the selected
realization.

A vector memory actor remains one firing, one address token, at most one mask
token, one data token for load or store, and one retirement event. The address
token is scalar for `element` and `contiguous`, and is the complete flattened
address vector for `indexed`. Mapping may not split one of these semantic
tokens across unrelated physical endpoints or reinterpret vector lanes as
Physical Tags.

The physical operation endpoint width and the backing memory-service beat
width are different facts. Every selected endpoint and transport segment must
carry its complete software token. A Fabric memory engine may nevertheless
implement one actor firing with several internal service transactions when its
declared use pattern preserves inactive-lane suppression, row-major lane
order, result assembly, and the actor's single retirement event. Mapping may
select such a declared realization but may not invent transaction
decomposition.

Physical adaptation is never represented by silently changing a semantic
vector type, inserting semantic `pack` or `serialize` actors, or deriving
meaning from an equal total bit width. A `vector<4xf32>` and a `vector<2xf64>`
remain different requirements even though both occupy 128 bits.

Every routed vector operand or result is one complete semantic token. Mapping
may not stripe its lanes across unrelated endpoints, create one RouteTree per
lane, or recover high bits after a narrower transport segment. If no selected
physical realization and route can carry an actor's complete token, a
Dataflow-to-Dataflow candidate transform may explicitly split or scalarize the
actor only after proving values, exceptional lanes, firing atomicity, memory
effects, ordering, and backpressure equivalent. The transformed actors then
form a different immutable Canonical Dataflow candidate. Mapping itself never
performs that rewrite.

## Verification

Verification rejects:

* scalable or rank-zero vectors at Canonical Dataflow finalization;
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
  absent nor lowered to an explicit ordered form;
* whole-payload atomic accesses that are not one unmasked `element` access;
* atomic granularity on a plain access; and
* vector atomic shapes that violate the contracts in
  `docs/spec-dataflow-memory-consistency.md`.

Stable anchor tests cover:

* exact leading-chunk masks, operand rendezvous, chunk pairing, result order,
  retirement, and graph-return wiring, plus scalarization when the leading
  extent is one;
* exclusion of non-total elementwise-looking operations from the generated
  decomposition domain;
* per-lane poison and undef propagation, lazy vector selection, and inactive
  masked-lane non-observation;
* static and dynamic extract/insert of scalar and trailing-subvector values,
  plus shuffle selection, duplication, and poison blocks;
* rejection of a selected SpatialRegion whose scalable vector has not been
  materialized to fixed structured semantics;
* rank-one partial-group `parallelize` and `serialize` behavior;
* exact and partial-tail activation closure;
* rank-one and multi-rank `pack`/`unpack` round trips, including floating-point
  payload bits;
* homogeneous defined, poison, and undef pack/unpack identities, plus rejection
  of mixed-lane vector round-trip identity;
* multi-rank contiguous and gather/scatter addressing in row-major lane order;
* inactive-lane address suppression and all-zero-mask completion;
* repeated gather addresses and rejected unresolved duplicate scatter
  addresses;
* distinct `element`, `contiguous`, and `indexed` access views for otherwise
  equal payload widths;
* rejection of attempts to encode physical port adaptation as semantic
  `pack`, `serialize`, or a changed vector type.
