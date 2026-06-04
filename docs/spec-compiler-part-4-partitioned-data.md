# Loom Compiler Part 4: Partitioned Data

This document specifies the software-side partitioned-data contract used
by Loom front-end IR. It describes how a memref is associated with a
logical partition domain, and how code inside a `dataflow.thread`
definition queries the portion of that memref owned by the current
logical thread instance.

The partition domain is a software index space. It is not a hardware
mesh, x/y coordinate system, PE coordinate space, routing graph, or
fabric topology. Binding logical thread instances and partition-domain
points to physical AccCore resources is owned by binding/PnR, not by the
ops in this document.

## 1. Scope

Part 4 owns:

* Logical partition-domain declarations.
* View-like annotations that associate memrefs with partition domains.
* Thread-body queries for local data ranges and logical thread
  coordinates.
* Verifier rules that connect partitioned data to
  `#loom.thread_axis<...>` mapping entries.

Part 4 does not own:

* HostCore-to-AccCore boundary selection.
* `dataflow.thread` definition and launch shape.
* Structured-control-to-dataflow lowering.
* Physical AccCore selection, core id assignment, routing, placement, or
  time-slot scheduling.
* Distributed-buffer neighbor exchange protocols.

## 2. Concepts

### Logical Partition Domain

A logical partition domain is a named, ranked software index space. Its
shape records the number of logical positions along each domain axis.

The target op is:

```text
dataflow.partition_domain @D { shape = [d0, d1, ...] }
```

Each `di` is a positive integer attribute. The rank is the number of
shape entries. A domain symbol carries no physical topology. It only
names a logical index space that thread-axis attributes and partition
layouts can reference.

### Thread-Axis Mapping

Part 3 defines the target mapping attribute:

```text
#loom.thread_axis<parallel, axis>
#loom.thread_axis<multiplexed, axis>
#loom.thread_axis<parallel, axis, @D>
#loom.thread_axis<multiplexed, axis, @D>
```

The optional symbol names a logical partition domain. A `parallel` axis
may bind different dynamic values to different AccCore execution slots
when binding/PnR chooses to do so. A `multiplexed` axis may reuse the
same AccCore execution slot over time. Neither kind is a physical
coordinate or topology statement.

### Partition Layout

A partition layout is a view-like memref annotation. It preserves the
source memref type while recording how memref dimensions are partitioned
over a logical domain.

The target op is:

```text
%view = dataflow.partition_layout %source
    { domain = @D,
      split_dims = [[axis0, ...], [axis1, ...], ...] }
    : memref<...>
```

`split_dims` has one entry per source memref dimension. Each entry lists
the domain axes that partition that memref dimension. An empty entry
means the dimension is replicated with respect to the domain.

The result aliases the source. Memory effects through the layout result
are effects on the underlying source storage.

### Local Range

`dataflow.local_range` returns the half-open range of a source memref
dimension owned by the current logical thread instance.

The target op is:

```text
%lo, %hi = dataflow.local_range %view { dim = d } : memref<...>
```

The operand must be the result of `dataflow.partition_layout` or a
same-storage view chain rooted in such a result. The `dim` attribute is a
source memref dimension. The result is expressed in the source memref's
index space.

The enclosing thread must provide matching `parallel` thread-axis entries
for every domain axis that partitions `dim`. `multiplexed` axes do not
establish data ownership and therefore do not satisfy local-range reach.

### Thread Coordinate

`dataflow.thread_coord` returns the current logical coordinate for a
domain in the enclosing `dataflow.thread` definition.

The target op is:

```text
%c0, %c1, ... = dataflow.thread_coord @D : index, index, ...
```

The op returns one coordinate per domain axis in axis order. For every
domain axis, the enclosing thread must have exactly one matching
`#loom.thread_axis<parallel, axis, @D>` entry. The coordinate value is
the corresponding thread-body grid IV. `multiplexed` entries do not
contribute to `thread_coord`.

### Thread Linear Id

`dataflow.thread_linear_id` returns a row-major flattening of
`dataflow.thread_coord @D` against the domain shape.

The target op is:

```text
%id = dataflow.thread_linear_id @D : index
```

The op has the same admissibility requirement as `dataflow.thread_coord
@D`: every domain axis must be covered exactly once by a matching
`parallel` thread-axis entry.

## 3. Verifier Contract

`dataflow.partition_domain`:

* Is a top-level symbol.
* Has non-empty shape.
* Every shape entry is a positive integer.
* Carries no fabric-binding or routing attributes.

`dataflow.partition_layout`:

* Takes a memref-like source.
* Returns the same type as the source.
* References a visible `dataflow.partition_domain`.
* Has one `split_dims` entry per source memref dimension.
* Every listed axis is in the referenced domain's axis range.
* A domain axis may appear in at most one source-dimension entry unless
  an explicit future design introduces replicated ownership.
* Is treated as a view-like alias for effect projection and
  boundary-materialization analysis.

`dataflow.local_range`:

* Appears inside a `dataflow.thread` definition body.
* Has a valid `dim` for the rooted source memref.
* Its operand's same-storage view chain resolves to exactly one
  `dataflow.partition_layout`.
* Every domain axis referenced by the layout entry for `dim` is covered
  by exactly one `#loom.thread_axis<parallel, axis, @D>` entry on the
  enclosing thread mapping.
* `multiplexed` thread-axis entries do not satisfy the reach rule.

`dataflow.thread_coord` and `dataflow.thread_linear_id`:

* Appear inside a `dataflow.thread` definition body.
* Reference a visible `dataflow.partition_domain`.
* Require every domain axis to be covered by exactly one matching
  `#loom.thread_axis<parallel, axis, @D>` entry.
* Reject mappings that leave a domain axis uncovered or cover the same
  domain axis more than once.

## 4. Interaction With Part 3

Part 3 owns thread creation and `#loom.thread_axis<...>`. Part 4 consumes
those mapping entries only as logical execution-axis tags.

Partitioned-data ops are ScalarCore-side helpers. They do not appear
inside a `dataflow.graph` definition. If graph code needs a local range,
coordinate, or linear id, the value is computed in the enclosing
`dataflow.thread` body and passed to `dataflow.graph.launch` as an
ordinary operand.

Partition-layout values cross a `dataflow.thread.launch` boundary through
the same `dataflow.map_info` protocol used for other memref-like values.
Because `dataflow.partition_layout` preserves the source memref type,
the verifier recovers the partition metadata by walking the same-storage
view chain.

## 5. Testing Expectations

Tests for this part should cover:

* Valid and invalid `dataflow.partition_domain` declarations.
* Valid and invalid `dataflow.partition_layout` annotations.
* Same-storage view-chain recovery for `dataflow.local_range`.
* `local_range` acceptance when all needed domain axes are covered by
  matching `parallel` thread-axis entries.
* `local_range` rejection when coverage is missing, duplicated, or only
  provided by `multiplexed` entries.
* `thread_coord` and `thread_linear_id` acceptance for full-domain
  `parallel` coverage.
* Rejection of `thread_coord` and `thread_linear_id` outside a
  `dataflow.thread` body.
* Confirmation that none of these ops carry physical topology,
  adjacency, routing, or fabric-resource binding semantics.

## 6. Future Work

Future work may define distributed-buffer protocols and neighbor
exchange, but those protocols must be explicit. They must not be hidden
inside the partition-domain or partition-layout ops.
