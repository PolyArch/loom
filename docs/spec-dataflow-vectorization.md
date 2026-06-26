# Loom Dataflow Vectorization

This document specifies the dataflow-level vector boundary used by Loom
to adapt scalar streams, packed vector-like values, and masked memory
interfaces. It is the single source of truth for
`dataflow.parallelize`, `dataflow.pack`, `dataflow.unpack`, and
`dataflow.serialize`.

The canonical IR source is `include/Dataflow/IR/DataflowOps.td`; the
verifier implementation lives in `lib/Dataflow/IR/DataflowOps.cpp`.
DFG-sim semantics are specified here and implemented in
`lib/Simulator/DFGSimulator.cpp`.

## Scope

The vector boundary has three jobs:

* group scalar token streams into fixed-width lane groups;
* represent partial groups with an explicit mask instead of silent
  padding;
* convert between lane-parallel streams and one packed integer value
  that can later be connected to masked memory and Fabric memory ports.

This document does not own generic loop-vectorization legality. Loom
should reuse LLVM and MLIR infrastructure for canonical loop, affine,
and vector IR analysis where possible. Loom owns the lowering contract
from those canonical forms into dataflow streams, Fabric memory, PnR,
simulation evidence, and DSE feedback.

## Token Cardinality

The vector boundary changes token cardinality. A scalar stream with
many active element tokens may become one packed token plus one mask
token per group. Conversely, one packed token plus one mask token may
become multiple scalar tokens and one trailing false continuation token.

DFG-sim completion checks must not require a packed vector result to
produce one returned token per scalar input token. This exception is
limited to direct results of the vector boundary ops, and it only
applies after every `dataflow.parallelize` group has either filled or
been flushed by a false continuation token. It must not weaken
memory-load completeness, store completion, or ordinary scalar result
checks.

When `dataflow.serialize` sees a zero mask, it emits only the trailing
false continuation token. If a graph returns the serialized data stream
itself, the data result is missing under the current report contract and
the simulation is blocked; callers that need zero-length streams must
return an explicit count or mask alongside the continuation.

## Common Attributes

All four operations carry:

* `vec_size : i64` -- number of logical lanes.

`vec_size` must be a power of two in `[1, 64]`. The mask type is always
`iN`, where `N = vec_size`; bit `i` describes lane `i`.

The current implementation supports signless integer lane values and
packed signless integer values in DFG-sim. Packed lanes are bit-pattern
values: unpacking an `i8` lane with all bits set reports `i8:255`,
rather than applying a signed interpretation. The target contract
permits the compiler to introduce these ops from higher-level LLVM or
MLIR vectorizable patterns, but that lowering is a separate compiler
pass.

## `dataflow.parallelize`

`dataflow.parallelize` groups scalar tokens into lane-parallel streams:

```mlir
%lane0, ..., %laneN_minus_1, %mask =
  dataflow.parallelize %data, %cont {vec_size = N : i64}
    : (T, i1) -> (T, ..., T, iN)
```

An optional stride operand may be present:

```mlir
%lane0, ..., %laneN_minus_1, %mask =
  dataflow.parallelize %data, %cont, %stride {vec_size = N : i64}
    : (T, i1, T) -> (T, ..., T, iN)
```

The operation maintains a persistent lane pointer initialized to zero.
For each true `%cont` token, it consumes one `%data` token, consumes one
`%stride` token if present, writes the data token into the current lane,
sets the corresponding mask bit, and advances the pointer by the stride
or by one if no stride exists.

When the pointer crosses `vec_size`, the operation emits the pending
active lane tokens and one mask token, then wraps the pointer modulo
`vec_size`.

For a false `%cont` token, the operation consumes and discards the
paired `%data` token, emits any pending partial group, clears the mask,
and resets the pointer to zero. Inactive lanes emit no token.

## `dataflow.pack`

`dataflow.pack` converts lane streams into one packed integer:

```mlir
%packed = dataflow.pack %lane0, ..., %laneN_minus_1 mask %mask
  {vec_size = N : i64} : (T, ..., T, iN) -> iM
```

The packed type width `M` must equal `bitwidth(T) * vec_size`.
`dataflow.pack` waits for one `%mask` token, consumes only the lane
tokens whose mask bits are set, and emits one packed integer token.
Lane zero occupies the least significant slot. Inactive packed slots are
zero-filled.

## `dataflow.unpack`

`dataflow.unpack` converts one packed integer back into lane streams:

```mlir
%lane0, ..., %laneN_minus_1 =
  dataflow.unpack %packed, %mask {vec_size = N : i64}
    : (iM, iN) -> (T, ..., T)
```

The packed type width `M` must equal `bitwidth(T) * vec_size`.
`dataflow.unpack` consumes one packed token and one mask token. For each
set mask bit, it emits the corresponding lane value. For each unset mask
bit, it emits no lane token.

## `dataflow.serialize`

`dataflow.serialize` converts lane streams plus a mask back into a scalar
data stream and a continuation stream:

```mlir
%data, %cont = dataflow.serialize %lane0, ..., %laneN_minus_1 mask %mask
  {vec_size = N : i64} : (T, ..., T, iN) -> (T, i1)
```

The operation waits for one `%mask` token, consumes only active lane
tokens in lane order, emits each active lane on `%data`, emits a true
`%cont` token for each active lane, and emits one trailing false
`%cont` token. Inactive lanes are not consumed.

## Memory Mask Relationship

The mask produced by `dataflow.parallelize` may be forwarded to
`dataflow.pack`, and the same mask may later be forwarded to a masked
`dataflow.store` or `dataflow.load` contract. Software memory ordering
remains owned by `docs/spec-compiler-part-3-mem.md`; masks select which
sub-elements of a vector memory operation are active, but they do not
define a separate ordering network.

Hardware byte-enable lowering is owned by `docs/spec-fabric-mem.md`.
The mapping from a software lane mask to a Fabric byte mask must be a
pure projection from active lanes to the bytes that carry those lanes.

## DSE and Compiler Ownership

Vectorization is a DSE decision, not a source pragma or a hard CLI
switch. The compiler may generate scalar, packed-vector, masked-memory,
or tiled-memory candidates from the same source region. DSE profiles and
continuous weights decide which candidate is preferred, based on
software estimates, mapping success, CGRA-sim feedback, and hardware
cost.

Configuration defaults and profile weights are owned by
`docs/spec-config-ssot.md`. Vectorization constants must not be
scattered through compiler passes.

## Current Implementation Status

The current implementation defines the four dataflow ops, verifies the
integer lane shape, and implements DFG-sim behavior for packed widths up
to 64 bits. It does not yet implement:

* automatic source-loop vectorization;
* lowering from MLIR `vector` dialect to Loom vector boundary ops;
* masked `dataflow.load` / `dataflow.store`;
* masked `fabric.mem` ports;
* PnR and CGRA-sim evidence for vector memory resources.
