# Mapping Routing Boundary

This document records confirmed routing ownership and implemented transport
constraints. It does not define the still-open persistent SpatialMapping Route
Tree schema.

## Logical Nets

A software edge is identified by its typed producer endpoint plus typed
consumer endpoint. Freeze groups all external edges with the same producer
endpoint into one deterministic multi-sink logical net.

The physical realization of one logical net is a rooted Route Tree with shared
trunks and one or more sink branches. The target model is not one-edge-one-
route. It does not persist one symbolic path per software edge or infer route
identity from edge numbers, symbols, paths, printer order, or insertion order.

Compute internality is derived mechanically from the selected configured-
function topology and its exact actor-to-operation and boundary
correspondences. Only Memory Realization records carry explicit
`DataflowEdgeRef` witnesses for selected memory-internal connectivity. Every
other edge remains an external transport obligation.

## Explicit Connectivity

Routing uses only fully elaborated Fabric endpoints, explicit directed point-
to-point arcs, and explicit resource traversals. Coordinates and visualization
layout are not connectivity.

PE-local, FU-local, switch-local, memory-local, boundary, and FIFO traversal is
not free. A connection must appear in Fabric topology or be absorbed by a
configured-function correspondence for Compute or an explicit Memory
Realization internal-edge witness. Mapping must not reconstruct a second
topology from symbols or owner hierarchy.

The PnR import boundary uses the canonical Fabric elaboration API. Elaboration
is atomic; failure rejects the input without publishing a partial hardware
view or mutating the source artifact.

## Payload Capacity

For a selected physical path, usable software payload capacity is the minimum
data-field width of every traversed transport endpoint:

```text
payload_capacity = min(data_field_width(endpoint))
```

For both `bits<W>` and `bits_tag<W, T>`, the data-field width is `W`. Tag bits
are not software payload capacity. A route or route-cache lookup must include
the required payload width so a path proven for a narrower payload is not
reused for a wider one.

Canonical control and completion events have zero software payload width and
use the same typed transport model. They do not require a separate synthetic
control-route schema.

Same-kind width normalization follows Fabric-owned connection semantics.
Mapping must not persist a competing adapter or conversion description.

## Boundary Resources And Tags

For `fabric.boundary`, only the data projection is a canonical software
payload. Tag inputs, tag outputs, writer configuration, rewriting, and removal
remain Fabric-owned behavior combined with the selected SpatialMapping tag
assignment. A tag field does not create another route or software edge.

## Search State

Route search uses endpoint-only A* over the explicit directed graph. Per-net
mutable hot state is one rooted arborescence with shared prefixes;
`RouteTreeState` implements this rebuildable search representation.

Hot state, PathFinder occupancy and history, A* queues, predecessor arrays,
and dense endpoint indices are not persistent Mapping records. They may be
discarded and reconstructed from the exact inputs and selected candidate.

## Unresolved Schema

Persistent Route Tree parent relations, selected attachments, traversal and
resource references, buffering, tag continuity, resource-time use, and final
closure fields remain open. No compatibility parser or one-edge-one-route
record may be added while those records are unresolved.

## Validation

Current tests cover explicit directed reachability, payload-width filtering,
shared resource identity, deterministic topology freeze, multi-sink logical-
net grouping, configured-function Compute internality, and exact Memory
internal-edge witness absorption. Persistent Route Tree tests wait for the
closed SpatialMapping schema.
