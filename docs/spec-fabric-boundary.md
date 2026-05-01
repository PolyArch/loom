# Fabric Boundary Ops

This document specifies the three fabric boundary ops `fabric.s2t`,
`fabric.t2t`, and `fabric.t2s`. They are the canonical conversion ops
between the spatial domain (`!fabric.bits<BW>`) and the temporal,
tagged domain (`!fabric.bits_tag<BW, TW>`).

The canonical IR sources are `Fabric_S2tOp`, `Fabric_T2tOp`, and
`Fabric_T2sOp` in `include/Fabric/IR/FabricOps.td`. Verifier rules
live in `lib/Fabric/IR/FabricBoundaryOps.cpp`.

## Purpose

The fabric dialect distinguishes two stream domains:

* **Spatial domain.** A `!fabric.bits<BW>` channel carries a bit-vector
  data payload accompanied by an implicit valid/ready handshake. There
  is no notion of a "tag" in this domain.
* **Temporal domain.** A `!fabric.bits_tag<BW, TW>` channel additionally
  carries a `TW`-bit tag alongside the data. The tag identifies, for
  example, an iteration index in a temporally-shared compute pipeline.

A loom fabric expresses both domains in the same module. The three
boundary ops describe the transitions between the two domains:

| Op             | From                            | To                              |
|----------------|---------------------------------|---------------------------------|
| `fabric.s2t`   | spatial `bits`                  | temporal `bits_tag`             |
| `fabric.t2t`   | temporal `bits_tag<BW, TW1>`    | temporal `bits_tag<BW, TW2>`    |
| `fabric.t2s`   | temporal `bits_tag`             | spatial `bits` (and tag)        |

All three ops carry the `Pure` trait. None of them performs handshake
mediation (FIFO, mux, demux, etc.); they only transform the type of
the stream.

## Placement

All three ops are allowed only in a `fabric.module` body. The
verifier rejects them anywhere else (e.g., inside `fabric.fu`,
`fabric.pe`, the top-level builtin `module`, or any nested op body
not listed in `fabric.module`'s whitelist). See
`spec-fabric-module.md` for the module body whitelist.

The three ops are not container ops. They have no body, no symbol,
and no nested verifier rules.

## `fabric.s2t` -- spatial to temporal

`fabric.s2t` combines a spatial `bits<BW>` data stream with a
`TW`-bit tag into a `bits_tag<BW, TW>` tagged channel. The op exists
in two disjoint syntactic forms differentiated by operand count.

### General form (2 operands -- data, tag)

```mlir
%out = fabric.s2t %data, %tag
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
%out = fabric.s2t %data {sw_configs = {tag = 10 : i4}}
       : !fabric.bits<BW> -> !fabric.bits_tag<BW, TW>
```

* Operand `#0` (`%data`): `!fabric.bits<BW>`.
* Result: `!fabric.bits_tag<BW, TW>`.
* `sw_configs.tag` is required: an `IntegerAttr` whose bit-width
  equals the result tag-width `TW`. The integer value is interpreted
  as an unsigned `TW`-bit pattern. (MLIR's printer renders signless
  integers using a signed numeric literal that shares the same bit
  pattern; `tag = 10 : i4` and `tag = -6 : i4` denote the same tag.)

The two-operand form must not carry `sw_configs.tag`; the tag is
already supplied by SSA. The verifier rejects mixing the two.

### Verifier rules

* Operand count is 1 or 2.
* Result type is `!fabric.bits_tag<BW, TW>`.
* Operand `#0` type is `!fabric.bits<BW>` (data-width must match the
  result data-width exactly).
* Two-operand form: operand `#1` type is `!fabric.bits<TW>` (tag-width
  must match the result tag-width exactly). `sw_configs.tag` must be
  absent.
* One-operand form: `sw_configs.tag` must be an `IntegerAttr` whose
  type's bit-width equals the result tag-width.

There are no `hw_params`. The op describes a pure data adapter; no
hardware-side parameter has been specified yet.

## `fabric.t2t` -- temporal to temporal (tag remap)

`fabric.t2t` remaps the tag of a `bits_tag<BW, TW1>` channel into a
`bits_tag<BW, TW2>` channel via a hardware lookup table. The data
field is preserved; only the tag changes (and may also be resized).

```mlir
%out = fabric.t2t %in
       {hw_params = [{lookup_table =
           [{input_tag = 0 : i4, output_tag = 1 : i8},
            {input_tag = 1 : i4, output_tag = 7 : i8}]}]}
       : !fabric.bits_tag<32, 4> -> !fabric.bits_tag<32, 8>
```

The lookup table lives in `hw_params` because the table itself is a
hardware property of the op (it is fused into the LUT memory at
generation time). `sw_configs` is reserved for future multi-LUT
runtime selection and is not used in this iteration.

### Verifier rules

* Operand and result are both `!fabric.bits_tag<...>`.
* `BW` (operand data-width) equals the result data-width. The op only
  remaps the tag; it never changes the data field.
* `TW1` (operand tag-width) and `TW2` (result tag-width) may differ.
* `hw_params` is required: a length-1 array wrapping a dictionary
  with key `lookup_table`.
* `lookup_table` is a non-empty array of dictionaries. Each entry
  has keys `input_tag` and `output_tag`, both `IntegerAttr`s.
* `input_tag` integer width equals `TW1`. `output_tag` integer width
  equals `TW2`.
* All `input_tag` values across the LUT entries are distinct (no
  duplicate keys).

The integer values in each `IntegerAttr` are interpreted as unsigned
bit patterns of the corresponding tag-width; range checking is
implicit in the integer-attribute type.

### Multi-LUT roadmap

`sw_configs` is reserved for a future multi-LUT runtime-selectable
form. In that form, `hw_params` will hold an array of LUTs and
`sw_configs.lut_sel` will pick one at fabric-program time. The
single-LUT form documented above is the only form supported today;
the verifier rejects any unrecognized `sw_configs` keys for
`fabric.t2t`.

## `fabric.t2s` -- temporal to spatial

`fabric.t2s` splits a `bits_tag<BW, TW>` tagged channel back into the
spatial domain. Two disjoint syntactic forms are differentiated by
result count.

### Split form (2 results -- data, tag)

```mlir
%data, %tag = fabric.t2s %in
              : !fabric.bits_tag<BW, TW>
                -> (!fabric.bits<BW>, !fabric.bits<TW>)
```

Result `#0` is the data field as a `bits<BW>`; result `#1` is the
tag field as a `bits<TW>`. Both follow the operand's widths exactly.

### Drop-tag form (1 result -- data only)

```mlir
%data = fabric.t2s %in : !fabric.bits_tag<BW, TW> -> !fabric.bits<BW>
```

The tag field is consumed by the op and not surfaced. This is the
canonical "exit the temporal domain and discard the tag" form.

### Verifier rules

* Operand is `!fabric.bits_tag<BW, TW>`.
* Result count is 1 or 2.
* Result `#0` is `!fabric.bits<BW>` (must equal the operand's
  data-width).
* Two-result form: result `#1` is `!fabric.bits<TW>` (must equal
  the operand's tag-width).

No `hw_params`, no `sw_configs`.

## Width relaxation

The three boundary ops have strict typing rules: every operand and
every result type must match exactly under the verifier rules above.
There is no `to <inner-type>` clause on any of them.

If a wider SSA source needs to feed a narrower boundary-op operand
(or vice-versa) the width adaptation is achieved via an upstream
`fabric.fifo` or by routing through a `fabric.pe` template that
performs the width adaptation at its boundary. The boundary ops
themselves only describe the spatial/temporal type transition.

## Cross-references

* `spec-fabric-module.md` -- the canonical container that hosts all
  three boundary ops (and the body whitelist that admits them).
* `spec-fabric-fu.md` and `spec-fabric-pe.md` -- describe the
  `fabric.fu` and `fabric.pe` containers that, by their own body
  whitelists, exclude the boundary ops.
* `spec-fabric-instantiate.md` -- alternative routing path for
  width adaptation that is intentionally NOT folded into the
  boundary ops.
