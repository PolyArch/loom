# Mapping Placement Records

This document specifies mapping records that bind software execution
objects to hardware resources. Routing, scheduling, memory binding, and
visualization are specified by separate mapping specs.

Placement records do not create software work or hardware capacity. They
only record the selected relation between existing software objects and
existing hardware objects.

## Placement Families

The placement part of a mapping artifact contains three record families:

* thread binding;
* graph binding;
* operation binding.

Each family is optional only when the mapped workload has no object of
that family.

## Thread Binding

A thread-binding record maps a logical `dataflow.thread` instance domain
to physical `acc_core` execution resources and optional execution
batches.

Required fields:

* `record_id`.
* `thread_ref`: software reference to a `dataflow.thread` definition,
  launch, or logical thread instance domain.
* `binding_kind`: `explicit` or `parametric`.
* `target_kind`: `acc_core`.
* `target_nodes`: non-empty list of hardware node references whose
  node kind is `acc_core`.

Optional fields:

* `logical_domain`: symbolic description of logical instance
  coordinates, extents, and axis names.
* `batch`: execution batch index or parametric batch expression.
* `lane`: implementation-specific lane within an AccCore when the
  hardware exposes lanes as explicit resources.
* `predicate`: symbolic condition under which this binding applies.
* `metrics`: placement-local metrics such as estimated occupancy or
  batch count.

For `explicit` binding, the record lists logical instance descriptors
and target nodes one-to-one.

For `parametric` binding, the record describes a deterministic function
from logical coordinates to the ordered `target_nodes` list and optional
batch index. The function may use integer arithmetic over logical
coordinates, domain extents, and the target-node list length. It must
not use mesh coordinates, Manhattan distance, visual layout coordinates,
or coordinate adjacency to infer hardware topology. If an architecture
builder wants a coordinate-shaped distribution, it must materialize the
target node list and all hardware connectivity explicitly before PnR
records the binding.

Thread binding preserves the logical instance set. It may batch more
logical instances than physical AccCores, but it must not drop,
duplicate, or merge logical thread instances.

## Graph Binding

A graph-binding record maps a selected `dataflow.graph` execution
context to a SpatialCore execution context inside an AccCore.

Required fields:

* `record_id`.
* `graph_ref`: software reference to a `dataflow.graph` definition,
  launch, or graph instance.
* `thread_binding`: mapping reference to the enclosing thread-binding
  record.
* `acc_core`: hardware reference to the selected AccCore node.
* `spatial_module`: hardware reference to the selected `fabric.module`
  template visible from that AccCore.

Optional fields:

* `instance`: graph instance descriptor when one graph definition has
  many dynamic instances.
* `spatial_context`: symbolic context name for the selected SpatialCore
  configuration.
* `scalar_residual_context`: reference to ScalarCore residual code that
  remains outside the graph.
* `schedule_context`: mapping reference to a schedule-context record.
* `configuration_context`: mapping reference to reconfiguration data
  needed before graph execution.

Graph binding is legal only for graph launches inside innermost
executable thread bodies. A non-innermost orchestration thread may have
thread binding but must not directly own graph binding unless the
software IR contains a legal graph launch at that level.

## Operation Binding

An operation-binding record maps a software operation, software
subgraph, or compute region to a hardware resource inside the graph
binding's SpatialCore context.

Required fields:

* `record_id`.
* `software_ref`: software reference to `operation`, `subgraph`, or a
  named compute region.
* `graph_binding`: mapping reference to the enclosing graph-binding
  record.
* `binding_depth`: `operation`, `subgraph`, or `region`.
* `hardware_ref`: hardware reference to the selected `fabric.fu`,
  `fabric.pe`, `fabric.mem`, `fabric.switch`, `fabric.boundary`,
  `fabric.fifo`, or other explicit module resource.
* `compatibility`: symbolic compatibility class checked by the verifier,
  such as operation-set, FU materialization, memory-port, or adapter.

Optional fields:

* `selected_config`: configuration values needed by the hardware
  resource, such as `op_list` choice or `hw_params` choice.
* `share_group`: mapping reference to a resource-sharing record when
  the hardware resource is shared.
* `schedule_use`: mapping reference to a resource-use or schedule-slot
  record.
* `temporal_tag`: mapping reference to a temporal-tag assignment.
* `fallback`: `scalar_core` when the operation is intentionally left on
  ScalarCore residual execution.
* `metrics`: local cost, estimated latency, or utilization.

An operation binding may target a `fabric.pe` only when the PE-level
selection and internal FU/resource selection are also represented by
configuration, schedule, or resource-sharing records. If the mapping
requires a more precise target to route values, it must bind at that
more precise resource level.

For SpatialCore compute, the primary placement unit is the Compute
Realization owned by the Mapping Artifact. It groups Canonical Dataflow
Program actors and binds them to one compatible `fabric.fu` encoding with
complete actor-to-operation and boundary-port correspondence. This grouping
does not create or mutate a grouping op in the software artifact.

When a `fabric.pe` contains multiple FUs, the mapping must record which
FU is active for each use. Spatial PE use allows only one active FU for
the selected configuration. Temporal PE use may time-multiplex multiple
software uses, but every active FU selection must be separated by legal
schedule, tag, or reconfiguration evidence.

## Exclusivity and Sharing

Placement records do not by themselves legalize sharing. If two software
objects bind to the same exclusive hardware resource, the artifact must
also contain schedule or resource-sharing records that prove the uses do
not conflict.

Resources that are architecturally concurrent may accept multiple
bindings only when Fabric ADG explicitly declares that capacity.

## Placement Validation

The placement verifier checks:

* every thread target is an AccCore node;
* explicit thread binding covers each referenced logical instance once;
* parametric thread binding is deterministic and total over its logical
  domain;
* graph binding refers to the AccCore selected by an enclosing thread
  binding;
* graph binding uses a visible `fabric.module` template;
* operation binding refers to software inside the graph binding's
  software graph context;
* operation binding targets a compatible resource;
* every Compute Realization has exact actor coverage, selected FU and
  encoding ownership, configured-function equality, and boundary
  correspondence;
* PE-level bindings identify the active FU and prove that inactive FUs
  do not consume the same PE slot;
* shared exclusive resources have matching schedule or resource-sharing
  records;
* ScalarCore fallback is explicitly recorded rather than inferred from
  missing bindings.

## Acceptance Criteria

Placement records are complete when:

* nested logical thread domains can map onto a heterogeneous AccCore
  set without assuming a mesh;
* graph launches bind to SpatialCore contexts only through legal
  innermost executable thread contexts;
* operation and subgraph bindings can target FU-level, PE-level,
  memory-level, or ScalarCore residual execution explicitly;
* every exclusive shared resource use is backed by schedule or
  resource-sharing evidence.
