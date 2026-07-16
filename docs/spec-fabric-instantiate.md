# Fabric Instantiate

This document specifies `fabric.instantiate`, the op that binds a
previously-defined fabric template symbol into a legal parent scope as a
fresh hardware instance with its own SSA inputs and outputs.
The canonical IR source is `Fabric_InstantiateOp` in
`include/Fabric/IR/FabricOps.td`; verifier rules live in
`lib/Fabric/IR/FabricInstantiateOp.cpp`.

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
inner-input types are retained in the ODS-owned typed
`inner_input_types : ArrayRef<Type>` property only when at least one
operand has a width-relaxing `to` clause; otherwise the property remains
empty to keep the no-relaxation case round-tripping unchanged. ODS owns
the container and element-type contract; the operation verifier owns
operand-count and endpoint-compatibility checks.

The property is canonical: it is empty if every destination input type
equals its SSA operand type. A non-empty property has one entry per
operand and must contain at least one actual endpoint-type difference.

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

## Canonical root-local elaboration

The public Fabric IR API
`fabric::elaborateInstances(fabric::ModuleOp root)` and the registered
`loom` development pass `--loom-elaborate-fabric-instances` provide the
single canonical instance elaboration path. The API elaborates one selected
top-level `fabric.module` root directly contained by `builtin.module` and
preserves that root operation's identity. A non-top-level root is diagnosed as
unsupported because selecting its counterpart in a cloned hierarchy would
require guessing or inventing a hierarchy path identity. The pass applies the
same implementation to every top-level `fabric.module` directly contained by
the input `builtin.module`.

Elaboration has the following observable contract:

* Every concrete `fabric.instantiate` under an elaborated root is removed.
  Module targets inline their physical body operations at the use site in
  deterministic textual order. Named PE, FU, switch, and memory targets
  create fresh anonymous physical occurrences with independent operation,
  region, and mapping state for every site.
* Named definitions remain declarations. They are not physical occurrences,
  and module inlining does not copy them into the caller. Nested instances are
  resolved in the source template's symbol-table context before their fresh
  occurrences are created in the destination root.
* Symbol identity and visibility are not copied to anonymous occurrences.
  Capability and configuration attributes remain owned by the target
  declaration and are copied as the physical occurrence contract requires.
* A `fabric.instantiate @module` directly under `builtin.module` remains valid
  source syntax but is unsupported by root-local elaboration because there is
  no enclosing `fabric.module` occurrence owner. The pass diagnoses this
  boundary and does not invent a wrapper or top-level ownership model.
* Successful elaboration is deterministic, independent of Graph-region
  operation order, verifier-valid, round-trippable, and leaves no concrete
  `fabric.instantiate` under the selected root. Legal forward SSA and feedback
  through materialized physical occurrences are preserved. An instance-only
  alias cycle with no physical producer is diagnosed because it has no
  representable producer after all module boundaries are removed. Unused named
  declarations may remain.

Validation has two parts inside the transactional scratch IR. For each isolated
Graph-region block, elaboration creates every fresh occurrence and builds one
complete block-local replacement plan without erasing an instance or rewriting
a surviving use. Graph-region cloning establishes all result mappings before
operands, so forward SSA and feedback do not depend on textual order. The
block-local plan then proves the semantic rules that cannot be recovered by
verifying the elaborated leaf operations. Removing a module boundary may
compose two existing low-bit normalizations into one direct endpoint connection
only when the composition is equivalent. For a source width `S`, removed
intermediate module-port width `M`, and adjacent destination width `D`,
equivalence requires `M >= min(S, D)`. If `M` is narrower than both adjacent
widths, the first connection discards bits that a direct `S -> D` connection
would preserve, so elaboration rejects the instance as unsupported. For
`bits_tag`, this rule is checked independently for the payload and tag widths.
Memrefs retain their existing exact-type requirement. Elaboration does not
insert an adapter, FIFO, wrapper, or routing resource for a non-representable
composition.

Inlining and leaf materialization change the enclosing module seen by
`resolveLoomAddrBits` and `resolveLoomMemBusWidth`. The semantic check therefore
requires the definition and destination effective values to match for both
settings for module, PE, FU, switch, and memory targets. A mismatch is
diagnosed instead of rebinding resource semantics or copying module-scoped
configuration into a second authority.

The complete containing `builtin.module` is cloned before block-local planning
begins, preserving sibling symbol context. Each successful block plan rewires
its surviving uses and removes its concrete instances in the scratch IR. The
existing Fabric op verifiers then validate the materialized anonymous PE, FU,
switch, and memory occurrences and the fully elaborated scratch roots. The
outer complete-module scratch transaction provides failure atomicity across
all processed blocks and roots. Leaf legality is not duplicated in a second
hand-maintained preflight.

Both the reusable API and the builtin-module pass are failure-atomic. The pass
publishes the successful scratch builtin-module body only after every selected
root rewrites and verifies. The API finds the selected cloned top-level root by
its symbol, rewrites and verifies it in the complete scratch builtin-module
context, and then moves only its body into the original root. Any semantic,
rewrite, or verifier failure discards the scratch IR and leaves the original
IR untouched.

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
