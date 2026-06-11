# Fabric Instantiate

This document specifies `fabric.instantiate`, the op that binds a
previously-defined fabric template symbol into a legal parent scope as a
fresh hardware instance with its own SSA inputs and outputs.
The canonical IR source is `Fabric_InstantiateOp` in
`include/Fabric/IR/FabricOps.td`; verifier rules live in
`lib/Fabric/IR/FabricOps.cpp`.

## Op shape and assembly syntax

```mlir
%r0, %r1 = fabric.instantiate @callee(%a : !fabric.bits<32>,
                                      %b : !fabric.bits<32> to !fabric.bits<16>,
                                      %m : memref<8xi32>)
                              -> (!fabric.bits<32>, memref<8xi32>)
```

Operands form:

* `@callee` -- a flat symbol reference to a previously-defined
  `fabric.module`, `fabric.pe`, `fabric.switch`, `fabric.mem`, or
  PE-local `fabric.fu`, subject to the parent/target table below.
* `(%v : T_outer [to T_inner], ...)` -- per-operand SSA value plus the
  operand's outer (SSA source) type and an optional `to T_inner` clause
  that names the target's declared input port type at this position.
  When the `to` clause is absent, `T_inner` defaults to `T_outer` (no
  width relaxation).
* `-> (T_out0, T_out1, ...)` -- declared result types as seen by
  consumers. Output direction is strict in this iteration: each result
  type must equal the target's declared output port type.

The IR-level operand types reflect the SSA source side. The internal
inner-input types are stashed in an `inner_input_types : ArrayAttr` only
when at least one operand has a width-relaxing `to` clause; otherwise
the attribute is omitted to keep the no-relaxation case round-tripping
unchanged.

`fabric.instantiate` implements `SymbolUserOpInterface`, so the symbol
table verifier dispatches `verifySymbolUses` automatically.

## Allowed instantiation sites and targets

| Parent of `fabric.instantiate` | Legal target kinds                                |
| ------------------------------ | ------------------------------------------------- |
| `builtin.module` (top-level)   | `fabric.module` only                              |
| `fabric.module` body           | `fabric.module`, `fabric.pe`, `fabric.switch`, or `fabric.mem` |
| `fabric.pe` body               | `fabric.fu` only                                  |
| Anywhere else                  | rejected                                          |

The verifier dispatches on the resolved target's op kind; mismatch
emits a parent-site-specific diagnostic that names the unsupported
target kind and the offending symbol.

## Named definitions

`fabric.pe`, `fabric.switch`, `fabric.mem`, and `fabric.fu` exist in
two disjoint syntactic forms by `sym_name` presence; the parser branches
on whether `@sym` appears right after the op keyword.

* **Anonymous form** (definition + use combined): variadic SSA operands
  bound via `(%pa = %a : T [to T_inner], ...)` plus variadic SSA
  results via `-> T` / `-> (T0, T1, ...)`. Same shape as before. The
  op produces SSA values in the enclosing scope.
* **Named template form** (declaration only): zero SSA operands, zero
  SSA results in the enclosing legal parent scope. The port signature
  is captured in a `function_type : FunctionType` attribute and the
  body's entry block carries the input port types as block-arguments.
  The body terminator is `fabric.yield`, whose value list matches
  `function_type.getResults()` for body-bearing ops. Actual usage of a
  named template goes through `fabric.instantiate @sym(...)`.

```mlir
fabric.module @Core() -> () {
  fabric.pe @ALU [spatial] (!fabric.bits<32>, !fabric.bits<32>)
                           -> (!fabric.bits<32>) {
  ^bb0(%pa: !fabric.bits<32>, %pb: !fabric.bits<32>):
    fabric.fu @F (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32> {
    ^bb0(%fa: !fabric.bits<32>, %fb: !fabric.bits<32>):
      %v = fabric.op [@arith.muli] (%fa, %fb)
           : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
      fabric.yield %v : !fabric.bits<32>
    }

    %y = fabric.instantiate @F(%pa : !fabric.bits<32>,
                               %pb : !fabric.bits<32>)
         -> (!fabric.bits<32>)
    fabric.yield %y : !fabric.bits<32>
  }
  fabric.yield
}
```

Both `fabric.pe` and `fabric.fu` implement the standard
`SymbolOpInterface` with `isOptionalSymbol() == true`: when `sym_name`
is present the op participates in the enclosing `SymbolTable` and the
0-results requirement of the symbol verifier is enforced; when
`sym_name` is absent the op is not a symbol. The anonymous form is
rejected as an `fabric.instantiate` target (the lookup fails because
there is no symbol to bind), and the named form is forced to use the
function-type signature (any anonymous-form operand binding is a
parser error).

`fabric.pe` carries the `SymbolTable` trait so its body can host named
`fabric.fu` definitions. `fabric.module` carries `Symbol` and
`SymbolTable` traits, so its body can host named `fabric.pe`,
`fabric.switch`, and `fabric.mem` templates in addition to its own role
as a fabric symbol. FU definitions remain PE-local resources; they are
not module-level tiles.

## Width-relaxation rules at the instantiate boundaries

The instantiate op has two connection points:

1. **Operand SSA outer type vs. target's declared input port type
   (input direction).** The same connection-point rule used elsewhere
   in the dialect (low-bit alignment, zero-fill on extension) applies:
   * `bits` -> `bits`: widths may differ; aligned at the LSB.
   * `bits_tag` -> `bits_tag`: widths may differ on each of the bits
     and tag fields independently.
   * `memref` -> `memref`: types must match exactly.
   The `to T_inner` clause expresses the relaxation explicitly. Without
   the clause, outer == inner.
2. **Result SSA type vs. target's declared output port type (output
   direction).** Strict equality is the target contract; the result type
   must equal the target's output port type. Attempts to relax the
   output type are diagnosed as
   "result #N type ... must equal callee '@<sym>' output port type
   ... (output direction is strict; no width relaxation)".

`memref` types are always exact-match. The `to <inner-type>` clause is
rejected on `memref` operands.

## Symbol resolution rules

* **Nearest-symbol-table lookup.** The verifier resolves `@callee` via
  `SymbolTable::lookupNearestSymbolFrom`, walking outward through
  enclosing `SymbolTable` ops. Both `fabric.module` and `fabric.pe`
  carry the `SymbolTable` trait; lookup tries each enclosing
  `SymbolTable` in turn so a sibling top-level `fabric.module` is
  reachable from inside another `fabric.module`'s body, even though
  `fabric.module` is `IsolatedFromAbove`.
* **Forward-reference forbidden.** When the target op is in the same
  block as the closest ancestor of the `fabric.instantiate` site, the
  target must textually precede the use. Forward references (the
  named definition appears below its use) are rejected.
* **Self-reference forbidden.** The target op MUST NOT be the closest
  enclosing `fabric.{module, pe, fu}` of the `fabric.instantiate`
  site. Recursive instantiation (a fabric.module's body instantiating
  its own enclosing fabric.module) is therefore illegal.
* **Scope leakage prevented.** A named pe defined inside a
  fabric.module body is reachable only inside that module's body. A
  top-level `fabric.instantiate @inner_pe` cannot reach an
  `inner_pe` that is nested inside another fabric.module's body; the
  lookup fails with "references undefined symbol '@inner_pe'".

## Verifier checklist

`InstantiateOp::verify` (operand-only, fast path) checks:

* Per-operand outer/inner kind agreement.
* memref operands cannot use the `to <inner-type>` clause.

`InstantiateOp::verifySymbolUses` (cross-symbol checks) performs:

1. Symbol resolution as described above; failure emits "references
   undefined symbol '@<sym>'".
2. Target kind matches the parent-of-instantiate rule.
3. Self-reference prohibition.
4. Forward-reference prohibition.
5. Operand count equals the target's input port count.
6. Result count equals the target's output port count.
7. For each input port, the declared inner type equals the target's
   declared input port type.
8. For each output port, the result SSA type equals the target's
   declared output port type (strict).

## Body whitelist updates

* `fabric.module` body now also accepts `fabric.module` (nested) and
  `fabric.instantiate`. The implicit `fabric.yield` terminator is
  unchanged.
* `fabric.pe` body accepts `fabric.fu` and `fabric.instantiate`. The
  PE body must contain at least one compute resource: either a
  `fabric.fu` (anonymous or named template) or a `fabric.instantiate`
  whose resolved callee is a `fabric.fu`. In the named PE template
  form the PE body is additionally terminated by `fabric.yield` whose
  value list matches the PE's `function_type` results.

## Cross-references

* `spec-fabric-module.md` -- SpatialCore/CGRA template container, port
  types, and width-relaxation rule at the three intra-module
  connection points.
* `spec-fabric-pe.md` -- PE container, schedule predicate, body
  whitelist.
