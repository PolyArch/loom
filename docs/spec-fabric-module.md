# Fabric Module

This document specifies `fabric.module`, the top-level container of the
fabric dialect. The canonical IR source is `Fabric_ModuleOp` in
`include/Fabric/IR/FabricOps.td`; verifier rules live in
`lib/Fabric/IR/FabricOps.cpp`.

## Identity

* Mnemonic: `module`. C++ class: `::fabric::ModuleOp`.
* The op is a `Symbol` with a required `sym_name`.
* The body is a single block (a `SizedRegion<1>` with one block).
* The body region is `Graph`-kind, so SSA dominance is not enforced and
  back-references between body ops are permitted.
* The op is `IsolatedFromAbove`: every value used inside the body must
  come from the body's block arguments or be defined inside the body.
  No external SSA value may leak into the module body.
* The body is closed by a `fabric.yield` terminator.

## Inputs (entry-block arguments)

`fabric.module` carries zero SSA operands of its own. Module inputs are
declared as the entry block's arguments, mirroring `func.func`. The
syntax is:

```mlir
fabric.module @top(%a : !fabric.bits<32>,
                   %b : memref<8xi32>,
                   %c : !fabric.bits_tag<8, 2>,
                   %d : !fabric.bits_tag<0, 3>) -> (...) {
  ...
}
```

Allowed input types:

| Type                      | Allowed | Rationale                                     |
|---------------------------|---------|-----------------------------------------------|
| `!fabric.bits<W>`         | yes     | Native handshake-bearing fabric port.         |
| `!fabric.bits_tag<W, T>`  | yes     | Native fabric port; `W = 0` is the tag-only form. |
| `memref<...>`             | yes     | Memory ports use MLIR's native memref.        |
| Any other MLIR type       | no      | Rejected by the verifier.                     |

`i32`, `f32`, `vector`, `tensor`, `index`, etc. are not valid module
input types and are rejected with a clear diagnostic.

Each input port has its own type; widths and shapes are independent
across ports.

## Outputs

`fabric.module` declares Variadic output port types on the op
signature:

```mlir
fabric.module @top(...) -> (!fabric.bits<32>, memref<8xi32>) {
  ...
}
```

The same allowed-type table applies to outputs. Outputs are produced by
the `fabric.yield` terminator inside the body. The yield value count
must equal the module's declared output count, and yield value types
must conform to the width-relaxation rule below.

A module may have zero outputs (`-> ()`) or zero inputs
(`fabric.module @top()`).

## Body whitelist

`fabric.module` body may only contain (today):

* `fabric.pe` (both `[spatial]` and `[temporal]`)
* `fabric.switch` (both `[spatial]` and `[temporal]`; see
  `docs/spec-fabric-switch.md`)
* `fabric.fifo`
* `fabric.module` (nested or sibling top-level brought in by symbol
  reference -- the body whitelist accepts top-level `fabric.module`
  ops directly so cross-module references via `fabric.instantiate`
  resolve correctly)
* `fabric.instantiate` (binds a previously-defined fabric symbol
  into this scope; see `docs/spec-fabric-instantiate.md`)
* `fabric.boundary` (single op covering all three boundary directions
  -- `[s2t]`, `[t2t]`, `[t2s]` -- between the spatial `bits` domain
  and the temporal `bits_tag` domain; see
  `docs/spec-fabric-boundary.md`)
* `fabric.yield` (terminator)

Future container ops -- `fabric.mem` -- are listed in the dialect
roadmap and will be added to the whitelist as they land.

`builtin.unrealized_conversion_cast` is **not** in the whitelist. All
fabric module values must come from a real fabric producer (a sub-
module result) or from the module's entry-block arguments.

## Width-relaxation rule (three connection points)

`fabric.module` body has three connection points where width relaxation
is permitted as long as the type-kind matches:

1. **Module input -> sub-module operand.** A `fabric.module` block
   argument feeding a sub-module (e.g., `fabric.pe [spatial]`,
   `fabric.fifo`).
2. **Sub-module result -> sub-module operand.** A sub-module's result
   inside the same module body consumed by another sub-module.
3. **Sub-module result -> module yield.** The value flowing out of the
   module to one of its declared result ports.

At each of these boundaries the source SSA value's type may differ from
the destination's declared type **in width only**, while the
type-kind must be identical:

* `bits` -> `bits`
* `bits_tag` -> `bits_tag`
* `memref` -> `memref` (no width relaxation; types must match exactly).

### Semantics: low-bit alignment, zero-fill on extension

For `bits<Ws>` -> `bits<Wd>` the two values are aligned at the LSB.

* If `Ws > Wd`: the high `Ws - Wd` bits of the source are dropped
  (truncation).
* If `Ws < Wd`: the high `Wd - Ws` bits of the destination are
  zero-filled.

For `bits_tag<Ws, Ts>` -> `bits_tag<Wd, Td>` the rule applies
independently to the bits field and the tag field. Each field aligns
low-bit-first; extension is zero-fill, truncation drops high bits. The
tag-only form `bits_tag<0, T>` follows the same rule on the tag field.

For `memref<...>` -> `memref<...>` no width relaxation is allowed. The
two memref types must match exactly: same element type, same shape,
same layout, same memory space. Mismatch is a verifier error.

### IR-level expression of the relaxation

The relaxation is made explicit at each connection point through a
`to T_inner` (or `to T_module_result`) clause, mirroring `fabric.fu`'s
existing FU-boundary truncation syntax:

* `fabric.pe [spatial]` operands accept an optional
  `to <inner-type>` clause:
  ```mlir
  fabric.pe [spatial](%pa = %src : !fabric.bits<32>
                                to !fabric.bits<16>) -> ...
  ```
  Outer (operand source) type is `bits<32>`, inner (PE block arg) type
  is `bits<16>`. The PE's K, L, W rules still govern the inner side.
* `fabric.fifo` operand accepts the same clause:
  ```mlir
  %0 = fabric.fifo %src to !fabric.bits<8>
                  [max_depth = 4, bypassable = false] : !fabric.bits<8>
  ```
  The FIFO operates internally at the inner type; the SSA source may
  be a different width within the same kind.
* `fabric.yield` accepts a per-value optional `to T_module_result`
  clause:
  ```mlir
  fabric.yield %v0 : !fabric.bits<32> to !fabric.bits<16>,
               %v1 : !fabric.bits<8>
  ```
  The first value is yielded as a `bits<16>` module result with low-bit
  alignment; the second yields a `bits<8>` value as-is.

Without the `to ...` clause, source and destination types must match.
The `to ...` clause is rejected on `memref` operands.

The relaxation does **not** apply inside a sub-module's body. Inside
`fabric.pe [spatial]` the existing `fabric.fu` `to T_inner` syntax for
the FU boundary still governs FU-input asymmetry; everything else stays
strict per the existing rules.

## Verifier rules

* The body whitelist accepts only `fabric.pe` (both schedules),
  `fabric.switch` (both schedules), `fabric.fifo`, `fabric.module`,
  `fabric.instantiate`, `fabric.boundary` (covering all three
  directions `[s2t]` / `[t2t]` / `[t2s]`), and the `fabric.yield`
  terminator. Any other op is rejected with a diagnostic that lists
  the allowed names.
* Each block-argument type must be one of the allowed module port
  types (`!fabric.bits<W>`, `!fabric.bits_tag<W,T>`, `memref<...>`).
* Each declared result type must be one of the same allowed types.
* The block-argument count and types must match the declared input
  types.
* The region kind is `Graph`.
* The op is `IsolatedFromAbove`: external SSA values cannot leak in;
  entry-block arguments are the only inputs.
* `fabric.yield` inside `fabric.module` must have exactly as many
  operands as the module's declared result count, and each yield value
  must satisfy the width-relaxation rule against the corresponding
  module result type.

## Negative tests

The verifier exercises the following rejections (see
`test/fabric/unit/module/invalid.mlir`):

* Module with a non-allowed input type (e.g. `i32`).
* Module with a non-allowed output type.
* Module body containing `builtin.unrealized_conversion_cast`.
* Module body containing any non-whitelisted op.
* Yield value count not matching the declared result count.
* Yield type-kind mismatch (e.g. yielding a `bits_tag` for a `bits`
  result).
* `memref` width or shape mismatch on yield.
* `to T_inner` clause used on a `memref` operand.

## Cross-references

* `spec-fabric-pe.md` -- inner PE container (spatial and temporal
  schedules), including PE-side width and FU-boundary details.
* `spec-fabric-instantiate.md` -- the `fabric.instantiate` op that
  binds a previously-defined `fabric.{module, pe, fu}` symbol into
  the current scope as a fresh hardware instance.
* `spec-fabric-reconfigurable-op.md` -- per-op runtime axes that
  populate spatial PE configurations.
* `spec-fabric-hw-share-group.md` -- legal hardware-share groups for
  `fabric.op` `op_list` members.

## Maintenance

The canonical sources of truth are:

* `Fabric_ModuleOp` in `include/Fabric/IR/FabricOps.td` for the IR
  shape;
* `ModuleOp::parse`, `ModuleOp::print`, and `ModuleOp::verify` in
  `lib/Fabric/IR/FabricOps.cpp` for parser, printer, and verifier
  logic.

When adding a new whitelisted body op (e.g., `fabric.mem`), update both
the verifier's whitelist and the diagnostic message that lists the
allowed names.
