# Loom Compiler Part 4: Logical Domains And Data Views

This document specifies the software-side logical-domain and derived-data-view
contract used at Loom thread boundaries. The canonical ABI is intentionally
small: a thread definition owns behavior and exactly one logical-domain kind,
while each launch supplies that domain's root parameters and passes all data as
ordinary typed operands. The two domain kinds are a zero-based dense Cartesian
domain and a responsibility-tracked dynamic work domain.

The earlier `thread_axis`, `staticGrid*`, `dataflow.map_info`,
`dataflow.partition_domain`, and `dataflow.partition_layout` design was
removed because it duplicated domain, schedule, and boundary facts already
owned by the Structured Program Candidate, launch ABI, and SystemMapping.

## 1. Authority And Scope

Part 4 owns:

* the closed logical-domain kind of a thread definition;
* the logical-coordinate interpretation of a thread definition's trailing
  `index` block arguments;
* launch-domain cardinality rules;
* dynamic work-item identity, publication, retirement, and termination;
* source induction-variable reconstruction; and
* the boundary between ordinary software data views and physical Mapping.

Part 4 does not own:

* thread or SpatialCore outlining decisions;
* parallel, temporal, tiling, interchange, vector, or unroll choices;
* physical AccCore selection, route, Tag, reservation, or topology;
* a second memory-transfer or partitioned-data ABI; or
* a Runtime queue implementation, channel-lifecycle protocol, work-stealing
  policy, or multi-process atomic-memory protocol.

## 2. Logical Domain Kinds

Every `dataflow.thread` definition carries exactly one closed domain value:

```text
ThreadDomain =
    DenseRectangular
  | DynamicWork { work_item_arg_ordinal }
```

The canonical attribute spellings are:

```text
#dataflow.thread_domain<dense>
#dataflow.thread_domain<dynamic_work, work_item_arg = N>
```

This value is program semantics. It is neither Mapping policy nor a scheduler
configuration. `work_item_arg_ordinal` identifies exactly one ordinary
`function_type` input; it is absent for `DenseRectangular`. No string kind,
extension registry, or second work-domain object exists.

### 2.1 Dense Rectangular Domain

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

### 2.2 Dynamic Work Domain

A `DynamicWork` thread has no coordinate suffix. Its designated ordinary
argument is the typed work-item payload; all other ordinary arguments are
launch captures reused by every item in that domain instance. A caller-side
`dataflow.thread.launch` supplies exactly one root payload and no extents.

Each dynamic item has the stable runtime identity:

```text
WorkItemId =
  (domain_instance, root_or_parent_item_id, child_launch_ordinal)
```

The root uses the distinguished parent `Root` and ordinal zero. A child's
ordinal is the zero-based program-order occurrence of `dataflow.work.spawn`
within its parent item execution. Payload equality does not merge items, and
queue position, worker identity, AccCore choice, address, and wall-clock time
never enter identity.

`dataflow.work.spawn %payload` is legal only while executing a
`DynamicWork` thread. Its operand type must equal the designated work-item
argument type. It publishes one child to the current domain; it is not an
arbitrary nested `dataflow.thread.launch`, creates no new domain, returns no
handle, and cannot target another thread definition.

The first DynamicWork profile has no channel endpoints. A DynamicWork thread
cannot create, send, receive, capture, or bind `!dataflow.channel<T>`, and a
graph launch owned by such a thread cannot bind graph stream ports to a
channel. Work-list payload, child publication, active responsibility, memory,
and collective thread completion already form one complete model; adding a
`WorkItemId` message correspondence or non-affine channel relation would be a
second dynamic identity mechanism. A later profile may reopen the boundary
only with a concrete program that cannot use a dense channel domain, explicit
payload termination, or memory-backed work sharing.

The semantic termination authority is the domain's active responsibility set:

* launch admission acquires the root responsibility before the root is visible;
* the root source closes immediately after that one root publication, so later
  items can arise only through registered child spawn;
* spawn atomically acquires a child responsibility before making that child
  visible to any worker;
* a queued, in-flight, or executing item retains exactly one responsibility;
* completion of `dataflow.thread.yield` retires the current item exactly once;
  and
* the launch's collective `!dataflow.thread_token` retires exactly when the
  root source is closed and the active responsibility set is empty.

An active-count implementation is a derived cache of that set, not a second
semantic authority. Publication-before-retirement prevents a transient zero
from terminating a domain while a child is becoming visible. One logical
coordinator remains responsible for this atomic transfer. The Runtime ABI may
place this responsibility kernel behind a bounded, execution-local scheduler.
Such placement implements the same publication and retirement transactions;
it creates no program-visible queue, second completion condition, logical
identity, or Mapping policy. Program-visible, multi-process, or device-side
shared queues still require explicit atomic and memory-order actors plus a
compatible Fabric consistency and coherence realization; DynamicWork does not
infer one.

Dynamic-domain completion does not close a `!dataflow.channel`, emit EOS, or
terminate an unrelated graph or thread. A caller may use the ordinary
`!dataflow.thread_token` to order downstream work after quiescence. Concurrent
consumers that must discover end-of-stream still require an explicit payload
protocol or a future independently specified operation; no channel
open/close/reset state is added.

For a breadth-first traversal of a rooted tree, the root node is the launch
payload. Processing a node emits one `dataflow.work.spawn` per child in the
tree's canonical child order, then yields. A leaf only yields. The collective
token retires after the last descendant yields even if the implementation queue
was temporarily empty between parent execution and child visibility. General
graph BFS duplicate suppression and concurrent visited-set updates additionally
require explicit atomic software operations and a compatible Fabric
consistency and coherence realization; DynamicWork does not hide either
requirement.

### 2.3 Static And Dynamic Thread Identity

The Canonical Dataflow entity catalog identifies a static root launch with a
`RootThreadLaunchRef`; it does not assign an independent ID to the referenced
`dataflow.thread` definition. The exact definition is recovered through the
launch's Dataflow-owned callee relation.

A logical thread point is a derived value in that root-launch context:

```text
LogicalThreadPoint =
    DensePoint(RootThreadLaunchRef, coordinate_tuple)
  | DynamicPoint(RootThreadLaunchRef, WorkItemId)
```

The exact launch-parameter environment supplies extents and admitted ordinary
integer parameters used to interpret the point, but does not create another
identity object. A dense coordinate tuple is body-visible through the trailing
`index` arguments. It is never implicitly flattened; a program that needs a
linear identifier computes one explicitly from coordinates and extents.

One static root launch may execute repeatedly. Runtime owns one transient
`ThreadDispatchOccurrenceId` per concrete dispatch, and a concrete dense
instance is `(ThreadDispatchOccurrenceId, coordinate_tuple)`. The
`domain_instance` component of a `WorkItemId` is the same dispatch occurrence,
not a second counter or persistent identity. Occurrence IDs disappear after
execution and are unavailable to Mapping. If program behavior or Mapping must
distinguish repeated launches, the Structured Program must expose that
distinction as a coordinate, launch parameter, or DynamicWork stable-item
component rather than relying on a hidden epoch.

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

Logical coordinates, `WorkItemId`, designated work payload, and launch
parameters are software facts. SystemMapping's `B_thread` relation consumes the
legal logical domain to select an AccCore for each instance. A dynamic domain
may expose a typed stable-key tuple mechanically derived from its item identity
and payload for the existing `StableKeyLookup` relation; Mapping cannot use
queue order or invent another item identity. Event-relative `ResourceUse`
separately owns occupancy and release. Neither relation may reinterpret logical
identity as Cartesian hardware position.

For a dense domain, `B_thread` is evaluated over the exact
`RootThreadLaunchRef`, coordinate tuple, and Dataflow-owned launch-parameter
projection. For DynamicWork it uses the root launch plus the stable-item
projection. The transient dispatch occurrence is deliberately absent: one
verified relation applies to every execution of the same exposed logical
domain.

Data partitioning visible to the program is expressed by its ordinary index
and view computations. Physical placement of storage, memory services, and
routes is owned by SpatialMapping and SystemMapping. No software view silently
selects a `fabric.pe`, `fabric.mem`, transport endpoint, or protocol.

## 6. Verification And Tests

Anchor-level verification covers:

* exact agreement between launch extent count and callee coordinate rank;
* rank-zero, empty-domain, and negative-extent behavior;
* exact ordinary-operand type agreement;
* source-IV reconstruction for nonzero and dynamic lower or step values;
* rejection of physical topology or Mapping authority in the software ABI;
* root identity, deterministic child ordinals, acquire-before-publish, and
  exactly-once retirement for a dynamic work tree;
* collective completion only when the dynamic responsibility set is empty; and
* worker-independent item identity across Runtime assignment and stealing.

Tests should assert these stable boundaries rather than preserve a particular
analysis cache, view-chain implementation, textual op order, or optimization
heuristic.

## 7. Runtime Scheduling Boundary

The bounded scheduler profile is owned by `docs/spec-runtime-abi.md`. It may
place not-yet-started responsibilities on finite-capacity worker deques, but it
must preserve the same `WorkItemId` across assignments and delegate every
root admission, spawn, and retirement transaction to `DynamicWorkDomain`.
Queue pressure, worker idleness, cancellation requests, and scheduling trace
state cannot complete the domain. The scheduler does not read SystemMapping;
worker or deque selection cannot evaluate or replace `B_thread`. Connecting
dynamic items to selected production execution remains an independent Runtime
and Mapping integration obligation.

## 8. Deferred Semantics

Program-visible or device-side shared work queues, priority queues,
application duplicate suppression, distributed termination, active-item
migration, and remapping remain deferred. They require explicit atomic,
ordering, coherence, state-transfer, or service contracts. Distributed-buffer
and neighborhood-exchange behavior likewise requires explicit dataflow and
service semantics rather than hidden layout metadata.

Static halo and neighborhood exchange are compilation patterns, not new
Dataflow entities. A structured candidate expresses them through existing
logical-memory roots and views, explicit load/store dependencies, dense channel
source maps, and ordinary thread/graph completion. Lowering may coalesce those
operations only when it preserves the same visible memory, stream, and
completion contract. No `DistributedView`, `HaloArtifact`, or implicit address
ownership map is introduced.

The first version has no generic asynchronous bulk-movement operation. A
provider-specific DMA or collective is admitted only after it has an
independently observable software firing, completion, ordering, and failure
contract that cannot be represented by the current operations. Until then,
bulk copies lower to existing memory and channel semantics and use their exact
completion tokens.

General device-side runtime spawn, spawn-then-feed of an independently blocked
consumer, channel sessions or EOS, and arbitrary DynamicWork channel
correspondence are also deferred. They require independent observable
semantics; they are not inferred from `dataflow.work.spawn` or responsibility
termination.
