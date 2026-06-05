# Mapping Schedule and Buffer Records

This document specifies workload-specific schedule, temporal sharing,
tag, reconfiguration, and buffer records in a Loom mapping artifact.

Fabric ADG declares hardware capacity. The mapping artifact assigns a
specific workload's use of that capacity.

## Schedule Context

A schedule-context record names a schedule domain for a graph binding,
thread batch, or resource-sharing group.

Required fields:

* `record_id`.
* `owner`: mapping reference to a thread-binding, graph-binding, or
  resource-sharing record.
* `time_model`: `cycle`, `slot`, `batch`, `event`, or `unknown_bounded`.

Optional fields:

* `initiation_interval`.
* `period`.
* `start_cycle`.
* `end_cycle`.
* `clock_domain`.
* `reconfiguration_epoch`.
* `metrics`.

`unknown_bounded` is allowed only when the artifact still records enough
ordering, buffering, and exclusivity facts for consumers to validate
legality. It is not a license to omit required resource-use records.

## Resource Use

A resource-use record assigns one placed software object to one hardware
resource in a schedule context.

Required fields:

* `record_id`.
* `placement`: mapping reference to a placement record.
* `hardware_ref`: hardware resource reference.
* `schedule_context`: mapping reference to a schedule-context record.
* `use_kind`: `compute`, `route`, `memory`, `buffer`, `adapter`, or
  `configuration`.

Optional fields:

* `start`: cycle, slot, batch, or event expression.
* `duration`.
* `initiation_interval`.
* `exclusive`: boolean, default true unless the hardware declares
  multi-tenant capacity.
* `capacity_units`.
* `temporal_tag`: mapping reference to a temporal-tag assignment.
* `resource_sharing`: mapping reference to a resource-sharing record.

Exclusive resource uses must not overlap in the same schedule context
unless distinguished by a legal temporal tag or arbitration model.

## Temporal Tag Assignment

A temporal-tag record assigns workload-specific tag values to uses of
tagged hardware resources.

Required fields:

* `record_id`.
* `tag_domain`: hardware reference to the resource or domain that owns
  the tag namespace.
* `tag_width`: positive integer width declared by hardware.
* `assignments`: non-empty list of software or mapping references to
  integer tag values.

Optional fields:

* `reuse_policy`: `unique`, `time_partitioned`, or `resource_local`.
* `conflict_class`: identifier used by the verifier to check mutual
  exclusion.

Tags are mapping facts. Dataflow IR must not carry temporal tags before
mapping. Fabric ADG may declare tag width and tag-capable resources, but
it does not assign workload-specific tag values.

## Resource Sharing

A resource-sharing record identifies a hardware resource shared by
multiple software objects.

Required fields:

* `record_id`.
* `hardware_ref`: exclusive or capacity-limited hardware resource.
* `members`: non-empty list of placement or route records that share
  the resource.
* `sharing_policy`: `schedule_partitioned`, `tag_partitioned`,
  `batched`, `arbitrated`, or `capacity_partitioned`.

Optional fields:

* `schedule_context`.
* `temporal_tag`.
* `capacity_units`.
* `conflict_class`.
* `metrics`.

The record must name the evidence that makes sharing legal: schedule
partition, tag partition, batch partition, explicit arbiter, or
declared capacity.

## Reconfiguration

A reconfiguration record names workload-specific configuration events
required by reconfigurable hardware.

Required fields:

* `record_id`.
* `target`: hardware resource or configuration domain.
* `configuration`: mapping reference or inline symbolic configuration
  values.
* `schedule_context`.

Optional fields:

* `start`.
* `duration`.
* `epoch`.
* `blocking`: whether computation or routing must stop during the
  reconfiguration event.

Reconfiguration records are separate from placement records. Placement
selects resources; reconfiguration selects their workload-specific
configuration over time.

## Buffer Binding

A buffer-binding record maps a software stream, token, route, or queue
to physical storage.

Required fields:

* `record_id`.
* `subject`: software reference or mapping reference to the edge or
  route being buffered.
* `hardware_ref`: hardware reference to FIFO, memory, queue, register
  file, boundary buffer, or explicit buffer resource.
* `depth`: non-negative integer or symbolic expression.
* `backpressure`: `blocking`, `drop_illegal`, `overwrite_illegal`,
  `credit`, or `custom`.

Optional fields:

* `initial_occupancy`.
* `latency`.
* `schedule_context`.
* `producer_route`.
* `consumer_route`.
* `metrics`.

A depth of zero is legal only for purely combinational paths whose
producer-consumer timing is proven by schedule records.

## Validation

The schedule and buffer verifier checks:

* schedule contexts have valid owners;
* exclusive resource uses do not conflict;
* temporal tags fit declared tag widths and do not collide within a
  conflict class;
* resource-sharing records cite evidence that makes sharing legal;
* reconfiguration events do not overlap illegal resource uses;
* buffer depths are sufficient for the declared route and schedule
  assumptions;
* every route or edge that requires storage has a buffer-binding record;
* dataflow IR carries no workload-specific temporal tags.

## Acceptance Criteria

Schedule and buffer records are complete when:

* temporal sharing can be represented without modifying dataflow or
  Fabric IR;
* CGRA-sim can model resource conflicts, tags, reconfiguration, and
  buffer backpressure from the artifact;
* RTL lowering can identify required queues, FIFOs, and configuration
  events;
* the verifier can reject same-slot double booking of exclusive
  resources.
