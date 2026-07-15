# Mapping Routing Constraints

This document records only the confirmed representation constraints that
apply when Physical Mapping eventually owns concrete communication
realization. It intentionally does not define route records or routing
schema.

The Physical Mapping ownership boundary and exact-predecessor rule are
specified by `docs/spec-mapping-artifact.md`. Fabric owns physical
connection behavior.

## Payload Capacity

For a selected physical path, the usable software payload capacity is the
minimum data-field width of every traversed transport endpoint:

```text
payload_capacity = min(data_field_width(endpoint))
```

For `bits<W>`, the data-field width is `W`. For `bits_tag<W, T>`, the
data-field width is also `W`. The `T` tag bits are never software payload
capacity.

A software representation is transportable only when its required payload
width does not exceed this minimum. For example, a 16-bit payload may use a
path whose data fields are 32, 64, and 32 bits, but not one containing an
8-bit data field. A `bits_tag<8, 8>` endpoint still provides only eight
payload bits.

## Same-Kind Width Normalization

A width difference between connected endpoints of the same Fabric port
kind does not require a Mapping adapter. It uses the connection semantics
owned by the Fabric specification. Mapping must derive that behavior from
the finalized Fabric endpoints and must not persist a competing conversion
description.

This document does not duplicate the Fabric-owned bit-alignment rule.

## Zero-Payload Events

Canonical control and completion events have zero software payload width.
They may use ordinary Fabric transport whose data-field capacity satisfies
that zero-width requirement. The transfer carries event occurrence; it
does not require a separate control-specific route schema or a synthetic
payload representation.

## Deferred Routing Model

The following remain deliberately unspecified here:

* route-tree and shared-trunk representation;
* route, segment, or symbolic record identifiers;
* resource-time claims and contention proof;
* temporal-tag allocation, remapping, and `t2t` behavior;
* schedule, instruction-slot, and reconfiguration records;
* buffer, memory, switch, boundary, and adapter record schemas;
* fanout and multicast record shape; and
* Physical Mapping completeness checks for those facts.

In particular, this document does not adopt the old one-edge-one-route
model. Those topics remain owned by their unresolved architecture work and
must not be inferred from this minimal constraint set.
