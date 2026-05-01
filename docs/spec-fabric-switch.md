# Fabric Switch

This document specifies `fabric.switch`, the leaf-level routing op of the
fabric dialect. The canonical IR source is `Fabric_SwitchOp` in
`include/Fabric/IR/FabricOps.td`; verifier and parser/printer live in
`lib/Fabric/IR/FabricSwitchOp.cpp`.

## Identity

* Mnemonic: `switch`. C++ class: `::fabric::SwitchOp`.
* Variadic SSA inputs and variadic SSA outputs (anonymous form) or a
  zero-operand / zero-result template (named form) whose port signature
  is captured in a `function_type` attribute.
* Carries a mandatory `Fabric_ScheduleAttr` predicate `[spatial]` or
  `[temporal]` (reused from `fabric.pe`).
* Optional `sym_name` so a switch can be referenced by
  `fabric.instantiate` (mirrors `fabric.pe` and `fabric.fu`).
* The op has no body region; routing is fully described by attributes.
* Allowed only inside a `fabric.module` body; the module body whitelist
  was extended to admit `fabric.switch`.

## Schedule predicate and port types

The schedule predicate selects the port type kind of every input and
output of the op. The two cases are mutually exclusive.

| Schedule    | Port type                  | Uniformity                              |
|-------------|----------------------------|-----------------------------------------|
| `spatial`   | `!fabric.bits<W>`          | All ports must share the same `W` (>= 0). |
| `temporal`  | `!fabric.bits_tag<W, T>`   | All ports must share the same `(W, T)` (`W` >= 0, `T` >= 1). |

Spatial ports may not use `bits_tag`; temporal ports may not use `bits`.
The verifier emits a "schedule mismatch with port kind" diagnostic on
violation.

In both forms, `K = numInputs() >= 1` and `L = numOutputs() >= 1`. For
the named form `K`/`L` are taken from the `function_type` signature; for
the anonymous form they are the SSA operand and result counts.

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
* Number of route-table entries the hardware allocates (one entry per
  programmable tag value the switch can route in a given configuration).
* Spatial switches MUST NOT carry `route_table_size` (rejected by the
  verifier with an "all-or-nothing" / "spatial fabric.switch must not
  carry temporal-only attribute" diagnostic).

## Software configuration

Software parameters live in `sw_configs`, a DictionaryAttr printed in
`{ ... }`. Two attributes:

1. `route_table` — shape depends on the schedule.
2. `switch_enable` — `BoolAttr`. Power/clock-gate equivalent. `true`
   means the switch is active; `false` gates the entire switch off.

**All-or-nothing rule.** When the switch is programmed, BOTH
`route_table` and `switch_enable` must be present. When the op is a
hardware-only declaration (not yet programmed), BOTH must be absent.
Mixing the two is rejected as an "all-or-nothing violation".

### route_table — spatial

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

### route_table — temporal

* ArrayAttr of exactly `route_table_size` entries.
* Each entry is a DictionaryAttr with three keys:
  * `route_sel` — ArrayAttr of `L` `StringAttr`s with the same shape as
    a spatial `route_table`.
  * `tag` — `IntegerAttr` of width `T` (matching the port tag width),
    value in `[0, 2^T)`.
  * `valid` — `BoolAttr`.
* Per-entry `route_sel` follows the spatial-route-table per-row rules
  (each row at most one `'1'`).
* **Tag uniqueness.** Among entries with `valid == true`, all `tag`
  values must be distinct. Two valid entries sharing a tag value is a
  configuration error (it would cause runtime ambiguity when a token
  arrives carrying that tag). Total valid entries can be 0 (degenerate
  switch: no routing currently active).

### Tag-driven trigger semantics

When a token arrives at an input port carrying tag value `t`, the
switch looks up the unique valid `route_table` entry whose `tag == t`
and uses that entry's `route_sel` as the spatial-style routing for the
cycle's tokens carrying tag `t`. Different tags routed in the same cycle
share the physical crossbar; same-cycle conflicts on a single output
port (multi-input-to-same-output across different tags) are arbitrated
in hardware via round-robin.

## Limits and arbitration

These rules are documented here but NOT enforced by the IR verifier
(they are runtime/hardware properties):

* Spatial: broadcast (single input -> multiple outputs) is allowed;
  fan-in (multiple inputs -> single output) is FORBIDDEN. The verifier
  enforces fan-in rejection via the per-row `'1'` <= 1 rule on
  `route_table`.
* Temporal: broadcast and fan-in are both allowed; multi-input-to-same-
  output across different tags is permitted and resolved at runtime.
  Same-cycle same-output conflicts get round-robin arbitration.

### Broadcast valid/ready protocol

To preserve broadcast semantics without creating combinational loops:

* Upstream `ready = AND(downstream_readys)`.
* Each downstream's `valid = upstream_valid AND AND(other_downstreams_readys)`.

This guarantees that a broadcast token is presented to all downstreams
exactly when every downstream is ready, without any downstream's `valid`
combinationally depending on its own `ready`.

## Assembly format

Anonymous spatial:

```mlir
%o:3 = fabric.switch [spatial] %i0, %i1, %i2, %i3
       [{connectivity_table = ["0110", "1011", "1111"]}]
       {route_table = ["01", "100", "0100"], switch_enable = true}
       : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>)
      -> (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>)
```

Anonymous temporal:

```mlir
%o:3 = fabric.switch [temporal] %i0, %i1, %i2, %i3
       [{connectivity_table = ["0110", "1011", "1111"], route_table_size = 8 : i32}]
       {
         route_table = [
           {route_sel = ["01", "100", "0100"], tag = 10 : i4, valid = true},
           {route_sel = ["10", "001", "0001"], tag = 11 : i4, valid = true}
         ],
         switch_enable = true
       }
       : (!fabric.bits_tag<32, 4>, !fabric.bits_tag<32, 4>, !fabric.bits_tag<32, 4>, !fabric.bits_tag<32, 4>)
      -> (!fabric.bits_tag<32, 4>, !fabric.bits_tag<32, 4>, !fabric.bits_tag<32, 4>)
```

Named template (spatial):

```mlir
fabric.switch @MySw [spatial]
       (!fabric.bits<32>, !fabric.bits<32>) -> (!fabric.bits<32>, !fabric.bits<32>)
       [{connectivity_table = ["11", "11"]}]
       {route_table = ["01", "10"], switch_enable = true}
```

Named template (temporal):

```mlir
fabric.switch @MySwT [temporal]
       (!fabric.bits_tag<32, 4>, !fabric.bits_tag<32, 4>)
        -> (!fabric.bits_tag<32, 4>, !fabric.bits_tag<32, 4>)
       [{connectivity_table = ["11", "11"], route_table_size = 1 : i32}]
       {
         route_table = [{route_sel = ["10", "01"], tag = 0 : i4, valid = true}],
         switch_enable = true
       }
```

Hardware-only forms omit the `{ ... }` `sw_configs` block entirely.

## Verifier rules

* `K >= 1`, `L >= 1`.
* Schedule + port type-kind correspondence (spatial -> `bits`,
  temporal -> `bits_tag`); uniform `W` (and `T` for temporal).
* `hw_params` shape: length-1 ArrayAttr wrapping a DictionaryAttr.
* `connectivity_table`: length `L`, each row length `K`, only `'0'`/`'1'`,
  per-row `>= 1` `'1'`, per-column `>= 1` `'1'`.
* Spatial: `route_table_size` MUST NOT be present.
* Temporal: `route_table_size` MUST be present and `>= 1`.
* All-or-nothing: `route_table` and `switch_enable` are both present
  (programmed) or both absent (hardware-only).
* Programmed: `route_table` shape and per-row constraints.
* Spatial: `route_table` per-row `'1'` count <= 1.
* Temporal: `route_table` length equals `route_table_size`; per-entry
  `route_sel` follows the spatial-row rules; `tag` integer width equals
  `T`; among valid entries `tag` values are distinct.
* Named form has zero SSA operands and zero SSA results; signature lives
  in `function_type`. Anonymous form has variadic SSA operands and
  variadic SSA results and must NOT carry `function_type`.

## Diagnostic substrings

The following substrings appear in `expected-error` directives in
`test/fabric/unit/switch/invalid.mlir` (used by other tooling that
greps for stable diagnostic anchors):

* `schedule mismatch with port kind`
* `connectivity_table' row #0 must have at least one '1'`
* `connectivity_table' column #1 must have at least one '1'`
* `spatial route_table row has '1' count > 1`
* `temporal duplicate valid tag`
* `all-or-nothing violation`

## Cross-references

* `spec-fabric-module.md` — top-level module body whitelist (which lists
  `fabric.switch` alongside `fabric.pe`, `fabric.fifo`,
  `fabric.boundary`, etc.).
* `spec-fabric-pe.md` — schedule predicate (`spatial` / `temporal`)
  shared with `fabric.switch`.

## Maintenance

The canonical sources of truth are:

* `Fabric_SwitchOp` in `include/Fabric/IR/FabricOps.td` for the IR shape;
* `SwitchOp::parse`, `SwitchOp::print`, and `SwitchOp::verify` in
  `lib/Fabric/IR/FabricSwitchOp.cpp` for parser, printer, and verifier
  logic.

When adding new connectivity-table or route-table semantics, update both
this spec, the verifier, and at least one positive and one negative lit
test under `test/fabric/unit/switch/`.
