# Mapping Routing Records

This document specifies route records that bind software producer-
consumer edges to explicit hardware connectivity.

Routing records are directed. Their semantic direction is producer
output to consumer input, master to slave, or manager to subordinate,
matching the directed channel endpoint model in Fabric ADG.

## Route Record

An edge-route record maps one software edge to one ordered hardware
route.

Required fields:

* `record_id`.
* `edge_ref`: software reference to a value edge, control-token edge,
  done-token edge, or memory-order edge.
* `producer_binding`: mapping reference to the placement record that
  produces the edge.
* `consumer_binding`: mapping reference to the placement record that
  consumes the edge.
* `payload_kind`: `data`, `control`, `done`, `memory_order`, `request`,
  or `response`.
* `segments`: non-empty ordered list of route segments.

Optional fields:

* `latency`: route latency estimate or selected latency model.
* `bandwidth`: selected route bandwidth when the hardware model exposes
  bandwidth.
* `buffer_binding`: mapping reference to buffer records used along the
  route.
* `protocol`: protocol name for compound protocols such as AXI-MM.
* `metrics`: route length, congestion estimate, hop count, or activity
  estimate.

`record_id` is the stable identity for route configuration records.
`edge_ref` must distinguish parallel software edges between the same
producer and consumer, such as separate SSA result-to-operand uses.

## Route Segment

Each route segment records one directed hardware connectivity use.

Required fields:

* `segment_id`: unique within the route.
* `segment_kind`: `system_link`, `module_path`, `resource_edge`,
  `boundary_crossing`, `adapter`, or `buffer`.
* `source_endpoint`: hardware reference to a directed source channel
  endpoint.
* `sink_endpoint`: hardware reference to a directed sink channel
  endpoint.

Optional fields:

* `hardware_ref`: reference to the `fabric.link`, switch path, FIFO,
  boundary, adapter, or other resource used by this segment.
* `channel`: protocol channel name.
* `direction_role`: output-to-input, master-to-slave,
  manager-to-subordinate, request, response, or custom protocol role.
* `conversion`: width, clock, reset, power-domain, or protocol
  conversion performed by the segment.
* `latency`: segment latency.
* `capacity`: segment capacity when the hardware declares it.

Compound protocol bundles are represented by multiple directed channel
segments. A bundle-level route may be present only as visualization or
reporting metadata; legality is checked on the individual directed
channel endpoints.

## Contiguity

The ordered route must be contiguous:

* the first segment source must be reachable from the producer
  placement's output endpoint;
* every adjacent segment pair must connect sink to source through the
  same endpoint or an explicit adapter/boundary/buffer resource;
* the last segment sink must reach the consumer placement's input
  endpoint.

A route may cross from system-level Fabric ADG into a `fabric.module`
template only through explicit boundary resources or module instance
ports. The mapping artifact must not invent implicit crossings.

## Fanout and Broadcast

One software fanout is represented as one route per consumer edge unless
the hardware contains an explicit broadcast or multicast resource.

If hardware broadcast is used, the route records must reference that
broadcast resource and still identify each consumer endpoint. The
artifact must not treat one output as implicitly driving many inputs.

All connections remain explicit and one-to-one at the directed channel
endpoint level.

## Arbitration and Contention

When multiple route records use a contended hardware resource, the
artifact must include either:

* schedule records proving uses happen in non-conflicting slots;
* temporal-tag records proving hardware can distinguish the uses;
* arbitration-resource records proving the hardware resource owns
  conflict resolution; or
* diagnostics reporting the conflict.

Route legality does not depend on visual proximity, mesh coordinates, or
Manhattan distance. Route cost may use explicit route weights, latency,
bandwidth, or capacity metadata, but it must not infer cost from
visualization coordinates.

## Adapter Requirements

A route segment that changes protocol, width, clock domain, reset
domain, power domain, or buffering behavior must reference an explicit
adapter, boundary, FIFO, buffer, or clock-crossing resource.

Clock-domain crossing is illegal when source and sink are in the same
clock domain. Domain metadata may be absent; absent domain metadata
means the default domain and does not by itself require an adapter.

## Routing Validation

The routing verifier checks:

* every route endpoint resolves to a directed channel endpoint;
* segment direction is legal for the endpoint roles;
* segment hardware exists in Fabric ADG or in the referenced
  `fabric.module` template;
* route segments are contiguous;
* producer and consumer endpoints match placement records;
* fanout is represented by explicit routes or explicit broadcast
  resources;
* no route assumes topology from visualization coordinates;
* all required adapters are explicit;
* contended resources have schedule, tag, arbitration, or diagnostics.

## Acceptance Criteria

Routing records are complete when:

* an arbitrary non-mesh topology can be routed through explicit links;
* a mesh-like topology is routed through explicit Fabric links, not
  coordinate adjacency;
* compound protocols are routed channel-by-channel;
* fanout and broadcast are explicit;
* CGRA-sim can compute route activity and contention from the artifact
  without asking PnR for internal state.
