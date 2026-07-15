# Fabric Boundary Op

This document specifies the unified `fabric.boundary` op, the canonical
conversion op between the spatial domain (`!fabric.bits<BW>`) and the
temporal, tagged domain (`!fabric.bits_tag<BW, TW>`). The op exists in
three direction-predicated variants (`s2t`, `t2t`, `t2s`) that share a
single op definition and a single verifier.

The canonical IR source is `Fabric_BoundaryOp` in
`include/Fabric/IR/FabricOps.td`. Verifier rules live in
`lib/Fabric/IR/FabricBoundaryOps.cpp`. The direction enum is
`Fabric_BoundaryDirectionAttr`.

## Direction predicate

Every `fabric.boundary` op carries a mandatory `direction` attribute
printed/parsed as a bracketed predicate, mirroring `fabric.pe`'s
`[spatial|temporal]` style:

```mlir
fabric.boundary [s2t] ...
fabric.boundary [t2t] ...
fabric.boundary [t2s] ...
```

The keyword between `[` and `]` is one of `s2t`, `t2t`, `t2s`. The
parser converts it via `BoundaryDirection::symbolize`. Any other
keyword is rejected with a parse-time diagnostic.

## Purpose

The fabric dialect distinguishes two stream domains:

* **Spatial domain.** A `!fabric.bits<BW>` channel carries a bit-vector
  data payload accompanied by an implicit valid/ready handshake. There
  is no notion of a "tag" in this domain.
* **Temporal domain.** A `!fabric.bits_tag<BW, TW>` channel additionally
  carries a `TW`-bit tag alongside the data. The tag identifies, for
  example, an iteration index in a temporally-shared compute pipeline.

A loom fabric expresses both domains in the same module. The three
boundary directions describe the transitions between the two domains:

| Direction | From                            | To                              |
|-----------|---------------------------------|---------------------------------|
| `s2t`     | spatial `bits`                  | temporal `bits_tag`             |
| `t2t`     | temporal `bits_tag<BW, TW1>`    | temporal `bits_tag<BW, TW2>`    |
| `t2s`     | temporal `bits_tag`             | spatial `bits` (and tag)        |

`fabric.boundary` carries the `Pure` trait. It does not perform any
handshake mediation (FIFO, mux, demux, etc.); it only transforms the
type of the stream.

`fabric.boundary` is required for a real port-kind or tagged-domain
transition. It is not required for an ordinary `bits` to `bits` or
`bits_tag` to `bits_tag` width mismatch. The latter follows the
module-level physical connection rule in `spec-fabric-module.md` and
does not instantiate an adapter resource.

## Placement

`fabric.boundary` is allowed only in a `fabric.module` body. The
verifier rejects it anywhere else (e.g., inside `fabric.fu`,
`fabric.pe`, the top-level builtin `module`, or any nested op body
not listed in `fabric.module`'s whitelist). See
`spec-fabric-module.md` for the module body whitelist.

`fabric.boundary` is not a container op. It has no body, no symbol,
and no nested verifier rules.

## `fabric.boundary [s2t]` -- spatial to temporal

`s2t` combines a spatial `bits<BW>` data stream with a `TW`-bit tag
into a `bits_tag<BW, TW>` tagged channel. The op exists in two
disjoint syntactic forms differentiated by operand count.

### General form (2 operands -- data, tag)

```mlir
%out = fabric.boundary [s2t] %data, %tag
       : (!fabric.bits<BW>, !fabric.bits<TW>) -> !fabric.bits_tag<BW, TW>
```

* Operand `#0` (`%data`): `!fabric.bits<BW>`.
* Operand `#1` (`%tag`):  `!fabric.bits<TW>`.
* Result: `!fabric.bits_tag<BW, TW>`.

The result's data-width `BW` and tag-width `TW` must equal the two
operand widths, respectively.

### Constant-tag form (1 operand + `sw_configs.tag`)

When the tag is fixed at fabric-program time the second operand is
elided and the constant is supplied through the `sw_configs`
dictionary instead:

```mlir
%out = fabric.boundary [s2t] %data {sw_configs = {tag = 10 : i4}}
       : !fabric.bits<BW> -> !fabric.bits_tag<BW, TW>
```

* Operand `#0` (`%data`): `!fabric.bits<BW>`.
* Result: `!fabric.bits_tag<BW, TW>`.
* `sw_configs.tag` is required: an `IntegerAttr` whose bit-width
  equals the result tag-width `TW`. The integer value is interpreted
  as an unsigned `TW`-bit pattern.

The two-operand form must not carry `sw_configs.tag`; the tag is
already supplied by SSA.

### S2T Verifier Rules

* Operand count is 1 or 2; result count is 1.
* Result type is `!fabric.bits_tag<BW, TW>`.
* Operand `#0` type is `!fabric.bits<BW>` (data-width must match the
  result data-width exactly).
* Two-operand form: operand `#1` type is `!fabric.bits<TW>` (tag-width
  must match the result tag-width exactly). `sw_configs.tag` must be
  absent.
* One-operand form: `sw_configs.tag` must be an `IntegerAttr` whose
  type's bit-width equals the result tag-width.
* `sw_configs.tag` must be a non-negative integer literal.
  Signless `iN` literals are normalized to a bit-pattern at parse
  time and so cannot be syntactically distinguished from their
  twos-complement positive twin; the rule is enforced when the
  literal carries an explicit signed/unsigned type (`siN`/`uiN`).
* `hw_params` must be absent on `[s2t]`.

## `fabric.boundary [t2t]` -- temporal to temporal (tag remap)

`t2t` remaps the tag of a `bits_tag<BW, TW1>` channel into a
`bits_tag<BW, TW2>` channel via a hardware lookup table. The data
field is preserved; only the tag changes (and may also be resized).

```mlir
%out = fabric.boundary [t2t] %in
       {hw_params = [{lut_size = 8 : i32}],
        sw_configs = {lookup_table =
            [{src_tag = 0 : i4, dst_tag = 1 : i8},
             {src_tag = 1 : i4, dst_tag = 7 : i8}]}}
       : !fabric.bits_tag<32, 4> -> !fabric.bits_tag<32, 8>
```

The IR uses a sparse representation: only the valid LUT entries are
materialized in `lookup_table`. The hardware-side LUT capacity is
declared via `hw_params[0].lut_size` and the runtime selection is
described by `sw_configs.lookup_table`.

### T2T Verifier Rules

* Operand count is 1; result count is 1.
* Operand and result are both `!fabric.bits_tag<...>`.
* `BW` (operand data-width) equals the result data-width. The op only
  remaps the tag; it never changes the data field.
* `TW1` (operand tag-width) and `TW2` (result tag-width) may differ.
* `hw_params` is required: a length-1 array wrapping a dictionary
  with key `lut_size` (`IntegerAttr`, value >= 1).
* `sw_configs` is required: a dictionary with key `lookup_table`
  (`ArrayAttr`).
* `lookup_table.size() <= lut_size` (extra entries are rejected with
  "more LUT entries than declared lut_size").
* Each entry is a dictionary with keys `src_tag` (`IntegerAttr`,
  width == `TW1`, non-negative literal) and `dst_tag` (`IntegerAttr`,
  width == `TW2`, non-negative literal). The negative-literal rule
  has the same signless caveat as `[s2t]`.
* All `src_tag` values across the LUT entries are distinct (rejected
  with "duplicate src_tag value <V>").

### Config-mem encoding

At the eventual hardware/config-mem level, every LUT slot is encoded
as a single packed word laid out from MSB to LSB:

```
+-----------+-----------+-------+
| DST_TAG   | SRC_TAG   | valid |
+-----------+-----------+-------+
   TW2 bits   TW1 bits   1 bit
```

Slot width is `TW1 + TW2 + 1` bits. The dense slot array has length
`lut_size`. The IR's sparse `lookup_table` carries only the valid
entries; the codegen materializes the dense array, fills in the
present (`src_tag`, `dst_tag`) pairs with `valid = 1`, and zero-fills
all remaining slots with `valid = 0`. The slot order within the
dense array is implementation-defined (codegen's responsibility);
the IR rule only requires the `src_tag` keys to be unique so that no
two slots can fire on the same incoming tag.

## `fabric.boundary [t2s]` -- temporal to spatial

`t2s` splits a `bits_tag<BW, TW>` tagged channel back into the
spatial domain. Two disjoint syntactic forms are differentiated by
result count.

### Split form (2 results -- data, tag)

```mlir
%data, %tag = fabric.boundary [t2s] %in
              : !fabric.bits_tag<BW, TW>
                -> (!fabric.bits<BW>, !fabric.bits<TW>)
```

Result `#0` is the data field as a `bits<BW>`; result `#1` is the
tag field as a `bits<TW>`. Both follow the operand's widths exactly.

### Drop-tag form (1 result -- data only)

```mlir
%data = fabric.boundary [t2s] %in
        : !fabric.bits_tag<BW, TW> -> !fabric.bits<BW>
```

The tag field is consumed by the op and not surfaced. This is the
canonical "exit the temporal domain and discard the tag" form.

### T2S Verifier Rules

* Operand count is 1; result count is 1 or 2.
* Operand is `!fabric.bits_tag<BW, TW>`.
* Result `#0` is `!fabric.bits<BW>` (must equal the operand's
  data-width).
* Two-result form: result `#1` is `!fabric.bits<TW>` (must equal
  the operand's tag-width).
* `hw_params` must be absent.
* `sw_configs` must be absent.

## Connection Widths

The verifier rules above define the boundary resource's own declared
transformation. For example, `s2t` preserves the declared data field
while introducing a tag field, and `t2t` performs the declared tag
lookup rather than an implicit bit reinterpretation.

Connections into and out of the boundary remain ordinary
`fabric.module` physical connections. A same-kind width mismatch on
either side therefore uses the canonical LSB-aligned truncation or
zero-extension rule from `spec-fabric-module.md`; it does not require
an upstream FIFO, PE, or synthetic adapter. The boundary remains the
explicit resource only for the semantic transition it owns:

* `s2t` introduces the tagged temporal domain;
* `t2t` performs configured tag-value remapping;
* `t2s` exits the tagged temporal domain and optionally exposes the tag.

An incoming boundary operand may spell both endpoints as
`source-type to destination-port-type`. The source type resolves the SSA
operand, while the destination type is the boundary input-port type used
by the direction-specific verifier rules above. Only `bits` to `bits`
and `bits_tag` to `bits_tag` are legal; widths follow the module-level
LSB-aligned semantics. Port-kind changes and `memref` are rejected. When
`to` is absent, the destination type equals the source type.

Differing destination input-port types are retained in the ODS-owned
typed `inner_input_types : ArrayRef<Type>` property. The custom assembly
syntax renders that state through the per-input `to` clauses and leaves
the property empty when every source and destination type is equal.
A non-empty property has one entry per operand and must contain at least
one actual endpoint-type difference.

A boundary consumes its operands once and exposes each result as a
distinct point-to-point module transport. Multiple boundary results do
not authorize reusing any one result at multiple consumers.

A plain same-kind resize must not be represented as a boundary op.

## Cross-references

* `spec-fabric-module.md` -- the canonical container that hosts
  `fabric.boundary` (and the body whitelist that admits it).
* `spec-fabric-fu.md` and `spec-fabric-pe.md` -- describe the
  `fabric.fu` and `fabric.pe` containers that, by their own body
  whitelists, exclude `fabric.boundary`.
* `spec-fabric-instantiate.md` -- template instantiation rules for
  Fabric resources whose external connections remain governed by the
  module-level compatibility rule.
