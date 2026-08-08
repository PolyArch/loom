# Fabric Instantiate

This document specifies `fabric.instantiate`, the op that binds a
previously-defined fabric template symbol into a legal parent scope as a
fresh hardware instance with its own SSA inputs and outputs.

Template instantiation elaborates immutable hardware capability. It does not
select a workload Mapping, create a software memory space, or authorize runtime
remapping.

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
  consumers. Each result type equals the target's declared output port type.

The IR-level operand types reflect the SSA source side. The internal
inner-input types are retained in the ODS-owned typed
`inner_input_types : ArrayRef<Type>` property only when at least one
operand has a width-relaxing `to` clause; otherwise the property remains
empty as the canonical equal-type representation. ODS owns the container and
element-type contract; the operation verifier owns
operand-count and endpoint-compatibility checks.

The property is canonical: it is empty if every destination input type
equals its SSA operand type. A non-empty property has one entry per
operand and must contain at least one actual endpoint-type difference.

An instantiate whose resolved target is a `fabric.module` additionally owns
one required typed, authoring-only domain-slot correspondence property:

```text
::fabric::ModuleInstanceDomainSlotBinding = {
  kind : Clock | Reset
  child_slot_ordinal : ordinal
  parent_slot_ordinal : ordinal
}

domain_slot_bindings :
  canonical array<::fabric::ModuleInstanceDomainSlotBinding>
```

Both ordinal fields cite the one semantic ordinal domain owned by
`FabricModuleDomainSlotRef.ordinal` in `docs/spec-fabric-identity.md`; this
record restates no narrower width. The resolved callee supplies the child
Module context and the enclosing
`fabric.module` supplies the parent Module context. A record therefore denotes
the exact correspondence
`(callee, kind, child_slot_ordinal) ->
 (parent, kind, parent_slot_ordinal)` without copying either Module identity.
The sequence is ordered and unique by `(kind, child_slot_ordinal)` and contains
exactly one row for every child Clock and Reset slot. Every selected parent
ordinal must exist in the same kind. Several child slots may deliberately map
to one parent slot, but one child slot cannot split across several parent
slots.
The ADG Builder's same-named `loom::adg` handle record is only an authoring
input and must resolve mechanically to this `::fabric` owner record before
Fabric finalization.

The property is empty only for a non-Module target. Fabric finalization first
materializes an omitted Module relation, so every Module target has at least one
Clock slot and one Reset slot and requires a complete binding relation. An
omitted property on a Module target fails closed. There is no omitted-property
default, name matching, ordinal matching, connectivity inference, containment
inheritance, or parent-wide Clock/Reset shortcut. The correspondence belongs
only to this instance edge; it is not a persistent Fabric local reference or a
second slot-assignment catalog.

## Allowed instantiation sites and targets

| Parent of `fabric.instantiate` | Legal target kinds                                |
| ------------------------------ | ------------------------------------------------- |
| `builtin.module` (top-level)   | none; rejected                                    |
| `fabric.module` body           | `fabric.module`, `fabric.pe`, `fabric.switch`, or `fabric.mem` |
| `fabric.pe` body               | `fabric.fu` only                                  |
| Anywhere else                  | rejected                                          |

The verifier dispatches on the resolved target's op kind. Every legal
instantiation is therefore owned by a concrete `fabric.module` root and has a
live elaboration path. A top-level instantiate is invalid rather than a
well-formed entry that later becomes unsupported or ignored.

## Named definitions

`fabric.pe`, `fabric.switch`, `fabric.mem`, and `fabric.fu` exist in
two disjoint syntactic forms by `sym_name` presence; the parser branches
on whether `@sym` appears right after the op keyword.

A named `fabric.mem` with an Operation Engine carries the engine's Spatial or
Temporal schedule. A storage-only memory template carries no schedule. The
schedule is not an outer memory or Local Memory Service property.

* **Anonymous form** (definition + use combined): variadic SSA operands
  bound via `(%pa = %a : T [to T_inner], ...)` plus variadic SSA
  results via `-> T` / `-> (T0, T1, ...)`. The op produces SSA values
  in the enclosing scope.
* **Named template form** (declaration only): zero SSA operands, zero
  SSA results in the enclosing legal parent scope. The port signature
  is captured in a `function_type : FunctionType` attribute and the
  body's entry block carries the input port types as block-arguments.
  A named `fabric.module` or `fabric.fu` uses value-bearing `fabric.yield` to
  match `function_type.getResults()`. A named `fabric.pe` uses zero-operand
  `fabric.yield`; its function type owns result port types and configured PE
  output selectors own their internal sources. Actual usage of a
  named template goes through `fabric.instantiate @sym(...)`.

```mlir
fabric.module @Core() -> () {
  fabric.pe @ALU [spatial] (!fabric.bits<32>, !fabric.bits<32>)
                           -> (!fabric.bits<32>) {
  ^bb0(%pa: !fabric.bits<32>, %pb: !fabric.bits<32>):
    fabric.fu @F (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32> {
    ^bb0(%fa: !fabric.bits<32>, %fb: !fabric.bits<32>):
      %v = fabric.op [@arith.muli] (%fa, %fb)
           {implementation_family =
              #fabric.implementation_family<example_multiply>}
           : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
      fabric.yield %v : !fabric.bits<32>
    }

    %y = fabric.instantiate @F(%pa : !fabric.bits<32>,
                               %pb : !fabric.bits<32>)
         -> (!fabric.bits<32>)
    fabric.yield
  }
  fabric.yield
}
```

`example_multiply` is schematic; the exact enum value is supplied by the
normative HSG registry. The example includes it to preserve the invariant that
every concrete `fabric.op` explicitly binds one implementation family.

When `sym_name` is present, `fabric.pe` and `fabric.fu` participate in the
enclosing symbol table and have no SSA results at the declaration site. When
it is absent, the op is an anonymous occurrence rather than an instantiation
target. Named forms use their function-type signature; anonymous operand
bindings are not valid on a named declaration.

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
   must equal the target's output port type. Output width relaxation is
   invalid.

`memref` types are always exact-match. The `to <inner-type>` clause is
rejected on `memref` operands.

## Memory Capability Roles

Instantiation preserves endpoint roles mechanically from the target
signature:

* each target `memref` input is a manager/requester endpoint and becomes an
  instantiate operand;
* each target `memref` result is a subordinate/provider endpoint and becomes
  an instantiate result.

Roles are endpoint-relative and are not attached permanently to the SSA value.
A subordinate result may legally feed a manager operand to compose services.
The instantiated endpoint remains a path to a physical service, not a logical
memory or storage identity. Sparse logical-memory-to-endpoint relations belong
to explicit Mapping records, not symbol instantiation or operand position.

## Symbol resolution rules

* **Nearest-symbol-table lookup.** Resolution walks outward through enclosing
  symbol tables. A sibling top-level `fabric.module` is reachable from inside
  another `fabric.module` body even though the latter is `IsolatedFromAbove`.
* **Authoring order is irrelevant.** A valid symbol may be declared before or
  after its use. Textual order cannot become hardware identity or elaboration
  semantics.
* **Recursive instantiation is forbidden.** The resolved instantiation graph
  reachable from each selected Fabric root must be finite and acyclic. This
  rejects direct self-instantiation and indirect cycles through other module
  templates without using declaration order as an accidental cycle breaker.
* **Scope leakage prevented.** A named PE defined inside a `fabric.module`
  body is reachable only inside that module body. An instantiate inside a
  sibling module cannot reach that nested PE.

## Verifier checklist

Local shape verification checks:

* Per-operand outer/inner kind agreement.
* memref operands cannot use the `to <inner-type>` clause.
* Domain-slot bindings use only the closed Clock and Reset kinds and are
  sorted-unique by child slot.

Cross-symbol verification checks:

1. Symbol resolution as described above.
2. Target kind matches the parent-of-instantiate rule.
3. Operand count equals the target's input port count.
4. Result count equals the target's output port count.
5. For each input port, the declared inner type equals the target's
   declared input port type.
6. For each output port, the result SSA type equals the target's
   declared output port type (strict).
7. A non-Module target has no domain-slot binding.
8. A Module target binds every callee Clock and Reset slot exactly once to an
   existing same-kind slot of the enclosing Module, with no extra row.

Whole-root elaboration additionally rejects every direct or indirect cycle in
the resolved instantiation graph.

## Canonical Elaboration

Fabric finalization selects a top-level `fabric.module` root directly contained
by `builtin.module` and elaborates its complete concrete instance graph. Every
legal `fabric.instantiate` below that root is live: it must resolve, satisfy the
parent/target and signature contracts, and materialize as a fresh occurrence.
If any such use cannot be elaborated, the entire root is rejected. No valid
instantiate may survive finalization. An instantiate directly under
`builtin.module` is rejected by the placement rule rather than deferred to
elaboration.

An authoring environment may make another module template available as an
elaboration input, but the finalized Module does not retain that source as an
`ImportedModule` dependency. Its concrete body is expanded, semantic
canonicalization removes the authoring route, and the final Module root admits
no direct dependency. Nested PE, FU, switch, and memory templates likewise
remain local elaboration inputs. This keeps symbol lookup out of persistent
identity and ensures equivalent inline and instantiated authoring forms
canonicalize to the same hardware artifact.

Module targets inline their physical body at the use site. Named PE, FU,
switch, and memory targets create fresh anonymous physical occurrences with
independent occurrence identity and physical state for every use. Named
templates remain declarations and do not themselves count as physical
resources. Nested references resolve in the source template's symbol context
before fresh occurrences are placed in the destination root.

Before a Module boundary is removed, elaboration materializes an omitted
callee domain relation and then composes the callee's exact slot assignments
with this instance's domain-slot correspondence. Every fresh
child internal owner is assigned directly to the selected parent slot. A child
boundary face obtains its effective parent slot through the same composition;
the adjacent parent-side connection must have that same effective Clock and
Reset assignment. Child boundary assignments disappear with the boundary, and
the binding property disappears with `fabric.instantiate`. Elaboration never
creates a fresh parent slot or retains an expanded instance-domain table.

Nested correspondences compose transitively. Thus two uses of the same child
Module may bind its slots differently, while equivalent inline and instantiated
authoring forms still produce the same flat Module relation. A missing,
duplicate, wrong-kind, foreign, or out-of-range binding rejects the complete
elaboration before any root is published.

Elaboration copies only immutable hardware capability and declared exact
implementation refinement. Workload-selected route, memory operation,
service dispatch, Physical Tag, and other Mapping configuration remain derived
per occurrence after Mapping. Physical configuration encoding remains owned by
`ConfigurationABI`, not by the template or elaborator.

The result is deterministic and independent of Graph-region operation order or
template declaration order. Legal forward SSA and physical feedback are
preserved. An instance-only alias cycle with no physical producer is invalid
because no producer remains after module boundaries disappear.

Removing a module boundary may compose two existing low-bit normalizations
only when the direct connection has identical semantics. For source width `S`,
removed intermediate width `M`, and destination width `D`, this requires:

```text
M >= min(S, D)
```

The rule applies independently to payload and tag fields. Memory capabilities
retain exact `memref` type equality. Elaboration never inserts an adapter,
FIFO, wrapper, or routing resource to repair a non-equivalent composition.
Widths come only from the canonical endpoint types; there are no module-local
address-width or memory-bus-width overrides to reconcile.

Elaboration is failure-atomic across the selected roots: it publishes only a
complete, verifying Fabric Hardware Description. PnR and SystemMapping consume
that immutable fully elaborated artifact. They do not instantiate templates,
add occurrences, or change topology during search. Every independently
bindable memory bank, switch, PE, FU, and memory Operation Engine therefore has
a concrete occurrence identity before Mapping begins.

## Container Whitelists

* A `fabric.module` body accepts nested named `fabric.module` declarations and
  `fabric.instantiate`, together with its `fabric.yield` terminator.
* `fabric.pe` body accepts `fabric.fu` and `fabric.instantiate`. The
  PE body must contain at least one concrete compute resource: either an
  anonymous `fabric.fu` occurrence or a `fabric.instantiate` whose resolved
  callee is a `fabric.fu`. A named FU declaration alone does not satisfy this
  requirement. In the named PE template
  form the PE body is additionally terminated by zero-operand
  `fabric.yield`; configured output selectors, not terminator operands, choose
  the sources for the PE's `function_type` results.

## Validation Anchors

Anchor-level validation covers one nested module instance, one fresh PE/FU
instance pair, one named FU template instantiated once and counted as exactly
one concrete occurrence, rejection of a concrete FU whose ports exceed its PE,
rejection of a top-level instantiate, rejection of recursive or out-of-scope
references, exact total child-to-parent Clock/Reset slot binding, two instances
of one child with different bindings, many-to-one slot binding, transitive
nested binding composition, equivalence between inline and instantiated forms,
rejection of an empty Module binding,
rejection of a missing, duplicate, wrong-kind, foreign, or out-of-range
binding, preservation of a legal physical feedback cycle, rejection of an
alias-only cycle, equivalent width-composition acceptance, and failure
atomicity. Tests do not freeze diagnostic wording, implementation API names,
every symbol-table nesting, or parser formatting.

## Cross-references

* `spec-fabric-module.md` -- SpatialCore/CGRA template container, port
  types, and width-relaxation rule at the three intra-module
  connection points.
* `spec-fabric-pe.md` -- PE container, schedule predicate, body
  whitelist.
* `spec-fabric-mem.md` -- Operation Engine, Local Memory Service, and
  manager/subordinate endpoint capability.
* `spec-mapping-memory.md` -- Mapping-owned memory bindings, access and
  exposure relations, and configured projections.
