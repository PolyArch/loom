# Loom Compiler Part 4: Logical Domains And Data Views

This document specifies the software-side logical-domain and derived-data-view
contract used at Loom thread boundaries. The canonical ABI is intentionally
small: a thread definition owns behavior and coordinate rank, while each
launch owns one zero-based dense extent per rank dimension and passes all data
as ordinary typed operands.

The earlier `thread_axis`, `staticGrid*`, `dataflow.map_info`,
`dataflow.partition_domain`, and `dataflow.partition_layout` design was
removed because it duplicated domain, schedule, and boundary facts already
owned by the Structured Program Candidate, launch ABI, and SystemMapping.

## 1. Authority And Scope

Part 4 owns:

* the logical-coordinate interpretation of a thread definition's trailing
  `index` block arguments;
* launch-domain cardinality rules;
* source induction-variable reconstruction; and
* the boundary between ordinary software data views and physical Mapping.

Part 4 does not own:

* thread or SpatialCore outlining decisions;
* parallel, temporal, tiling, interchange, vector, or unroll choices;
* physical AccCore selection, route, Tag, reservation, or topology;
* a second memory-transfer or partitioned-data ABI; or
* sparse/worklist domains and neighborhood protocols, which require their own
  explicit semantics before use.

## 2. Canonical Logical Domain

A `dataflow.thread` entry block has this canonical shape:

```text
(ordinary_args..., thread_ctrl : none, coord_0 : index, ..., coord_{K-1} : index)
```

The suffix length `K` is the coordinate rank and is derived from this block
shape. No rank, axis-kind, grid, layout, or topology attribute duplicates it.

Each `dataflow.thread.launch` supplies exactly `K` `index` extents. The dynamic
instance set is:

```text
[0, extent_0) x ... x [0, extent_{K-1})
```

Every extent is non-negative. Rank zero creates exactly one instance. If any
extent is zero, the domain is empty and the collective completion token retires
after launch dependencies without executing a thread body. Static verification
rejects a provably negative extent; runtime admission rejects a dynamic negative
value before creating any instance.

The coordinate tuple identifies an instance but defines no row-major linear
order, issue order, physical grid, or hardware topology. If program semantics
require a linear id, the Structured Program Candidate computes it explicitly
from coordinates and extents.

## 3. Source Induction Variables

Source lower bounds and steps are ordinary launch operands. The thread body
reconstructs each source induction variable mechanically:

```text
source_iv_d = lower_d + coord_d * step_d
coord_d in [0, extent_d)
```

This equation is program semantics. It supports dynamic lower bounds and
steps without making source-loop bounds part of the thread ABI. The SCF
optimizer must compute an extent that covers exactly the selected source
iteration domain and must preserve overflow and signedness semantics required
by the source program.

## 4. Derived Values And Memory Views

Values and memrefs cross a thread launch as ordinary typed operands and become
matching ordinary definition arguments. Tiling, local ranges, subviews,
address calculations, and explicit linearization use upstream MLIR operations
such as `affine`, `arith`, and `memref` while the program remains in the SCF
stage. Loom does not add a metadata-only passthrough operation.

The ownership optimizer decides whether a derived computation remains on the
InstructionCore or enters a `loom.spatial_region`. A computation selected for
the SpatialCore must mechanically lower to the canonical Dataflow actor
surface; otherwise it stays outside the graph or makes that candidate
non-finalizable. Analysis facts such as alias classes, access ranges, and
memory footprints remain derived analyses unless they change program
semantics.

## 5. Mapping Boundary

Logical coordinates and launch parameters are software facts. SystemMapping's
`B_thread` relation consumes them to select an AccCore for each legal logical
instance. Event-relative `ResourceUse` separately owns occupancy and release.
Neither relation may reinterpret coordinates as Cartesian hardware positions.

Data partitioning visible to the program is expressed by its ordinary index
and view computations. Physical placement of storage, memory services, and
routes is owned by SpatialMapping and SystemMapping. No software view silently
selects a `fabric.pe`, `fabric.mem`, transport endpoint, or protocol.

## 6. Verification And Tests

Anchor-level verification covers:

* exact agreement between launch extent count and callee coordinate rank;
* rank-zero, empty-domain, and negative-extent behavior;
* exact ordinary-operand type agreement;
* source-IV reconstruction for nonzero and dynamic lower or step values; and
* rejection of physical topology or Mapping authority in the software ABI.

Tests should assert these stable boundaries rather than preserve a particular
analysis cache, view-chain implementation, textual op order, or optimization
heuristic.

## 7. Deferred Semantics

Dynamic nonrectangular, sparse, and worklist-generated item domains are not
encoded by the dense extent ABI. Their item identity, termination, channel
interaction, and Mapping domain must be specified together before they become
canonical. Distributed-buffer and neighborhood-exchange behavior likewise
requires explicit dataflow and service semantics rather than hidden layout
metadata.
