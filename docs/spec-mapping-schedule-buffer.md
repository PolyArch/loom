# Mapping Resource Use, Tags, And Buffers

This document records confirmed ownership boundaries. The precise persistent
SpatialMapping record schema remains open.

## No Independent Schedule Authority

Mapping does not define a persistent `ScheduleContext`, absolute cycle or slot
schedule, schedule tree, or schedule candidate. The immutable Structured
Program Candidate owns selected software schedule and ownership decisions.

Physical resource-time behavior is formed mechanically from the Fabric-owned
use pattern and Mapping-owned event-relative use. `ResourceUse` may refer to
existing causal events, activation, occupancy, reservation, release, demand,
and legal sharing parameters. It must not create a second software schedule or
infer order from serialized record position.

## Binding And Use

SpatialMapping binding owns where a Compute or Memory Realization resides.
`ResourceUse` consumes that selected binding. It cannot select another FU,
instruction context, memory occurrence, route mode, semantic encoding, or
configuration by repeating those fields.

Configuration follows the same ownership rule: TechMapping owns semantic
encoding; SpatialMapping owns selected physical realization modes that are not
mechanically derived; `ResourceUse` owns only event-relative use of those
choices.

## Physical Tags

Physical Tag is not a global Mapping ID, a per-token sequence number, a firing
identity, an invocation epoch, or a software integer carried by Canonical
Dataflow. Tag values are meaningful only inside Fabric-owned match and
interpretation domains.

A selected value is stored once at an existing writer output or tagged ingress
binding. Downstream match rows and configured fields are derived along the
selected route until a real Fabric writer, rewriter, or remover changes that
continuity. There is no independent `TemporalTagAssignment`, `TagClaim`, tag
namespace, or sharing-policy record family.

Freeze and search may derive continuity segments, conflict graphs, dense
indices, and coloring caches. These objects are rebuildable and have no
artifact identity. Canonicalization may rename interchangeable colors, but it
must not change route, resource use, or value-sensitive constraints.

## Buffers

SpatialMapping must preserve every selected physical buffer or storage choice
that cannot be mechanically derived from Fabric and the other selected
records. Buffer legality remains orthogonal to routing, tags, ordered output,
capacity, and deadlock closure. No generic edge-or-route buffer record shape is
defined here.

## Unresolved Schema

The minimal persistent `ResourceUse`, Physical Tag, buffer, reconfiguration,
and sharing fields remain open. This document does not define numeric defaults,
absolute timestamps, independent schedule records, or compatibility parsers.

## Validation

Tests for persistent resource use, tags, and buffers wait for the closed
SpatialMapping schema. Existing tests may cover only Fabric-owned tag width and
transport behavior, rebuildable freeze facts, and the absence of retired
record families.
