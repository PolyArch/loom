# Fabric Switch

This document specifies `fabric.switch`, the leaf-level routing op of the
fabric dialect.

## Identity

* Mnemonic: `switch`.
* Variadic SSA inputs and variadic SSA outputs (anonymous form) or a
  zero-operand / zero-result template (named form) whose port signature
  is captured in a `function_type` attribute.
* Carries a mandatory `Fabric_ScheduleAttr` predicate `[spatial]` or
  `[temporal]` (reused from `fabric.pe`).
* Optional `sym_name` so a switch can be referenced by
  `fabric.instantiate` (mirrors `fabric.pe` and `fabric.fu`).
* The op has no body region; routing is fully described by attributes.
* Allowed only inside a `fabric.module` body.

`fabric.switch` owns architecture capability: physical ports, allowed
input-to-output traversals, bounded configured-entry capacity, and observable
transfer, grant, and progress behavior. SpatialMapping owns selected
traversals, Physical Tags at real writers or ingresses, event-relative use,
and configured route semantics. A Mapping-selected exact hardware refinement
may select among Fabric-declared cycle-observable alternatives. Finalization
assigns semantic rows deterministically. `ConfigurationABI` alone owns their
physical encoding; an implementation owns only the circuitry and microstate
that implement the exact Fabric/Mapping contract.

## Schedule predicate and port types

The schedule predicate selects the port type kind of every input and
output of the op. The two cases are mutually exclusive.

| Schedule    | Port type                  | Uniformity                              |
|-------------|----------------------------|-----------------------------------------|
| `spatial`   | `!fabric.bits<W>`          | All ports must share the same `W` (>= 0). |
| `temporal`  | `!fabric.bits_tag<W, T>`   | All ports must share the same `(W, T)` (`W` >= 0, `T` >= 1). |

Spatial ports may not use `bits_tag`; temporal ports may not use `bits`.

In both forms, `K = numInputs() >= 1` and `L = numOutputs() >= 1`. For
the named form `K`/`L` are taken from the `function_type` signature; for
the anonymous form they are the SSA operand and result counts.

This uniformity rule describes the switch's own declared physical
ports. It does not require a neighboring producer or consumer to use
the same width. A connection from a differently sized endpoint into a
switch input, or from a switch output into a differently sized endpoint,
uses the module-level LSB-aligned width rule in
`spec-fabric-module.md`. That connection does not add an adapter and
does not change the switch's internal crossbar width. Port-kind changes
remain illegal without an explicit `fabric.boundary`.

For an anonymous switch, each incoming type-list entry may spell both
endpoints as `source-type to destination-port-type`. The source type
resolves the SSA operand; the destination type is the switch input-port
type used by the uniformity and schedule checks. The clause accepts only
`bits` to `bits` or `bits_tag` to `bits_tag`; widths normalize with the
module-level LSB-aligned semantics. It rejects port-kind changes and
`memref`. When `to` is absent, the destination type equals the source
type. Named switch templates continue to derive all port types from
their declared `function_type` signature.

Differing anonymous input-port types are retained in the ODS-owned typed
`inner_input_types : ArrayRef<Type>` property. The custom assembly syntax
renders that state through the per-input `to` clauses and leaves the
property empty when every source and destination type is equal.
A non-empty property has one entry per operand and must contain at least
one actual endpoint-type difference.

## Hardware parameters

Hardware parameters live in `hw_params`, an ArrayAttr of length 1
wrapping a DictionaryAttr (the same `[ ... ]` convention used by other
fabric ops). `fabric.switch` requires `hw_params` to be present.

```
[{connectivity_table = ["0110", "1011", "1111"]}]                              // spatial
[{connectivity_table = ["0110", "1011", "1111"], route_table_size = 8 : i32}]  // temporal
```

### connectivity_table

* ArrayAttr of `L` `StringAttr`s (one row per output).
* Each row has length `K` (one character per input port).
* Characters are exactly `'0'` or `'1'`.
* **Bit-string convention: MSB on the left.** The leftmost character of
  a row corresponds to bit index `K - 1` (the highest input port index);
  the rightmost character is bit index `0` (input port 0). This is the
  universal convention for bit-string attributes in the fabric dialect.
* Per-row constraint: each row must contain at least one `'1'` (each
  output has at least one physical input source).
* Per-column constraint: across the `L` rows, each column index (i.e.,
  each input port) must contain at least one `'1'` in some row (each
  input has at least one physical destination).

### route_table_size (temporal only)

* `IntegerAttr` (`i32`), value `>= 1`.
* Number of resident route-table entries the hardware allocates. It is an
  explicit bounded capacity, independent of `2^T`, and does not allocate one
  row for every representable tag value.
* Spatial switches MUST NOT carry `route_table_size`.

## Configured Mapping Projection

Configured parameters are a deterministic projection of SpatialMapping and
are not part of the canonical hardware capability. The semantic projection is
one closed sum:

```text
SwitchConfiguration =
    Disabled
  | Active { route_table, physical_refinements }
```

`Disabled` carries no route table, tag, selector, or refinement. The physical
enable bit and inactive encoding belong only to ConfigurationABI. An active
projection whose route table selects no traversal canonicalizes to
`Disabled`.

The bit strings in `connectivity_table` and `route_table` are canonical
semantic representations of allowed and selected traversals. They are not a
register layout or configuration-memory bit assignment. Only
`ConfigurationABI` maps these semantic fields to physical encoding.

Canonical hardware-only Fabric has no selected configured projection. A
finalized configured view must contain exactly one closed variant; mixing a
raw enable field with an independently optional route table is invalid.

### route_table -- spatial

* ArrayAttr of `L` `StringAttr`s (one row per output).
* Row `j` has length equal to the count of `'1'`s in
  `connectivity_table[j]` (i.e., the number of physically-connected
  inputs for output `j`).
* Bit-string convention: **MSB on the left** (same as
  `connectivity_table`).
* Each row has AT MOST ONE `'1'` bit (zero `'1'`s means the output is
  temporarily routed nowhere).
* The position of the `'1'` selects which physically-connected input is
  routed to that output. Counting bit positions from the right (LSB-first)
  on the `route_table` row, position `p` selects the `p`-th `'1'` (also
  counted from the right, LSB-first) in the corresponding
  `connectivity_table` row.

Spatial switches allow **broadcast** (one input may be selected by
multiple `route_table` rows simultaneously) but FORBID **fan-in** to a
single output (the per-row `'1'` <= 1 rule enforces this).

At the enclosing `fabric.module` SSA level, broadcast does not reuse the
input transport at multiple consumers. The switch consumes that input
once and exposes each selected output port as a distinct SSA result.
Each result remains a point-to-point transport under
`spec-fabric-module.md`.

### route_table -- temporal

* ArrayAttr of exactly `route_table_size` entries.
* Each entry is one closed typed variant:
  * `Unused`, with no fields; or
  * `Active { route_sel, tag }`.
* Each Active entry has two fields:
  * `route_sel` -- ArrayAttr of `L` `StringAttr`s with the same shape as
    a spatial `route_table`.
  * `tag` -- `IntegerAttr` of width `T` (matching the port tag width),
    value in `[0, 2^T)`.
* Per-entry `route_sel` follows the spatial-route-table per-row rules
  (each row at most one `'1'`).
* **Tag uniqueness.** Among Active entries, all `tag`
  values must be distinct. Two Active entries sharing a tag value is a
  configuration error (it would cause runtime ambiguity when a token
  arrives carrying that tag). An all-Unused table canonicalizes to
  `Disabled`.

The table performs exact content match across Active resident entries. The tag
does not directly index a `2^T`-deep array, and route-table row identity is not
a software stream identity. The configured tag is derived from Mapping-owned
writer continuity and `ResourceUse` sharing assignments. It is a local
interpretation key where co-resident incompatible routes require distinction,
not a firing, iteration, invocation, or logical-token identity.

### Handshake Dependency Projection

`connectivity_table` declares capability and contributes no active arc by
itself. Absent an exact holding or registered refinement, each selected
spatial row or resident temporal row derives both the input-valid to selected-
output-valid dependency and the selected-output-ready to selected-input-ready
dependency. A unicast traversal therefore propagates both halves of
ready/valid flow. For atomic broadcast, every selected output-ready signal
participates in the one selected input's readiness because the source cannot
retire until every selected sink accepts. A Fabric-declared temporal grant,
holding, or registered refinement owns its replacement arc set and any
stateful break; it cannot silently retain or remove the zero-state arcs.

Disabled outputs, Unused temporal entries, and connectivity alternatives not
selected by the configured route table contribute no arc. All resident Active
temporal entries contribute their possible arcs even though runtime tags choose
which entry handles a token; a trace-specific tag value cannot erase a
structural dependency. These projections feed the two gates in
`docs/spec-fabric-module.md` and `docs/spec-mapping-verification.md` and are not
stored as a second switch graph.

### Tag-driven trigger semantics

When a token arrives at an input port carrying tag value `t`, the
switch looks up the unique Active `route_table` entry whose `tag == t`
and uses that entry's `route_sel` as the spatial-style routing for the
cycle's tokens carrying tag `t`. Different tags routed in the same cycle
share the physical crossbar; same-cycle conflicts on a single output
port (multi-input-to-same-output across different tags) request a grant under
the switch's exact Fabric-owned `GrantPolicy` or a Mapping-selected exact
hardware refinement declared by Fabric. The implementation executes that
policy; it does not choose one.

### Hardware payload opacity

A switch routes physical valid/ready transfers and treats the payload
bits as opaque. It does not assign or interpret software-level value,
control, or completion semantics.

## Transfer Guarantees And Arbitration

The architecture-level routing rules are:

* Spatial: broadcast (single input -> multiple outputs) is allowed;
  fan-in (multiple inputs -> single output) is FORBIDDEN. The verifier
  enforces fan-in rejection via the per-row `'1'` <= 1 rule on
  `route_table`.
* Temporal: broadcast and time-multiplexed fan-in are allowed;
  multi-input-to-same-output requests across different tags compete for the
  declared output capacity.

Temporal fan-in never means combinational merging. Each Active row still
selects at most one input per output. Competing rows share a physical resource
over time. Fabric owns request eligibility, output capacity, exact grant and
state-update behavior, latency, and backpressure visibility. The first closed
grant-policy domain is:

```text
fixed_priority(exact requester order)
round_robin(exact requester order, reset cursor, advance on successful grant)
```

The switch schema is the unique owner of its typed `ResourceState` values,
canonical initial state, capacity dimensions, atomic transfer UsePatterns,
and stable typed requester order. One broadcast pattern atomically claims the
ingress and every selected egress or crosspoint state; it cannot be split into
independent per-egress grants. Mapping selects only a declared exact policy
refinement and supplies typed route and workload values. Cursor, occupancy,
queue, and reservation state is nonpersistent execution state.

No policy is required when complete Mapping proves that at most one requester
can be eligible at once. If reachable contention exists, an exact policy must
be part of Fabric capability or a Mapping-selected exact refinement and must
participate in exact Fabric/Mapping identity. A deterministic exact simulator
rejects a contended switch whose contract gives only loose guarantees. Runtime,
Mapping, simulation, and RTL lowering may not supply a default policy. The
implementation owns the arbiter circuit and transient cursor, but not the
cycle-visible policy semantics.

### Broadcast backpressure contract

A selected single-input/multi-output transfer is one atomic message
replication. The source retires only when every selected sink accepts, or when
the implementation first commits the complete replication into explicit
holding state. Partial delivery, duplicate delivery, hidden draining, and
reordering are invalid. The exact ready/valid circuit is an implementation
choice and must not become an additional architecture or Mapping authority.

## Assembly format

Anonymous spatial:

```mlir
%o:3 = fabric.switch [spatial] %i0, %i1, %i2, %i3
       [{connectivity_table = ["0110", "1011", "1111"]}]
       : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>)
      -> (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>)
```

Anonymous temporal:

```mlir
%o:3 = fabric.switch [temporal] %i0, %i1, %i2, %i3
       [{connectivity_table = ["0110", "1011", "1111"], route_table_size = 8 : i32}]
       : (!fabric.bits_tag<32, 4>, !fabric.bits_tag<32, 4>, !fabric.bits_tag<32, 4>, !fabric.bits_tag<32, 4>)
      -> (!fabric.bits_tag<32, 4>, !fabric.bits_tag<32, 4>, !fabric.bits_tag<32, 4>)
```

The corresponding configured projections are represented semantically as:

```text
Active {
  route_table = ["01", "100", "0100"]
}

Active {
  route_table = [
    Active { route_sel = ["01", "100", "0100"], tag = 10 },
    Active { route_sel = ["10", "001", "0001"], tag = 11 },
    Unused, ...
  ]
}
```

Named hardware template (spatial):

```mlir
fabric.switch @MySw [spatial]
       (!fabric.bits<32>, !fabric.bits<32>) -> (!fabric.bits<32>, !fabric.bits<32>)
       [{connectivity_table = ["11", "11"]}]
```

Named hardware template (temporal):

```mlir
fabric.switch @MySwT [temporal]
       (!fabric.bits_tag<32, 4>, !fabric.bits_tag<32, 4>)
        -> (!fabric.bits_tag<32, 4>, !fabric.bits_tag<32, 4>)
       [{connectivity_table = ["11", "11"], route_table_size = 1 : i32}]
```

Hardware-only forms have no selected configured projection.
Configured examples illustrate the derived Mapping projection; they do not
make `sw_configs` part of Fabric hardware identity.

Reusable named templates are hardware-only. A configured projection belongs
to a concrete elaborated occurrence's derived configured view and may not be
stored on the template as shared workload state. Its
`HardwareConfigurationImage` encoding is produced only through the exact
`ConfigurationABI`.

## Verifier rules

* `K >= 1`, `L >= 1`.
* Schedule + port type-kind correspondence (spatial -> `bits`,
  temporal -> `bits_tag`); uniform `W` (and `T` for temporal).
* `hw_params` shape: length-1 ArrayAttr wrapping a DictionaryAttr.
* `connectivity_table`: length `L`, each row length `K`, only `'0'`/`'1'`,
  per-row `>= 1` `'1'`, per-column `>= 1` `'1'`.
* Spatial: `route_table_size` MUST NOT be present.
* Temporal: `route_table_size` MUST be present and `>= 1`.
* A configured occurrence is exactly `Disabled` or `Active`; the inactive
  variant carries no route fields.
* Active: `route_table` shape and per-row constraints, with an all-`Unused`
  table canonicalized to `Disabled`.
* Spatial: `route_table` per-row `'1'` count <= 1.
* Temporal: `route_table` length equals `route_table_size`; per-entry
  `route_sel` follows the spatial-row rules; `tag` integer width equals
  `T`; among Active entries `tag` values are distinct.
* Named form has zero SSA operands and zero SSA results; signature lives
  in `function_type`, and reusable templates do not carry `sw_configs`.
  Anonymous form has variadic SSA operands and variadic SSA results and must
  NOT carry `function_type`.

## Cross-references

* `spec-fabric-module.md` -- top-level module body whitelist (which lists
  `fabric.switch` alongside `fabric.pe`, `fabric.fifo`,
  `fabric.boundary`, etc.).
* `spec-fabric-pe.md` -- schedule predicate (`spatial` / `temporal`)
  shared with `fabric.switch`.
* `spec-fabric-system-adg.md` -- Transport Architecture capacity and
  guarantee ownership versus concrete Interconnect Implementation state.
