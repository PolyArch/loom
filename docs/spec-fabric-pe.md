# Fabric PE

This document specifies `fabric.pe`, a processing element container that
holds one or more `fabric.fu` instances. The op carries a mandatory
`schedule` predicate (`spatial` | `temporal`) that selects how the
contained FUs are time-shared. The canonical IR source is `Fabric_PeOp`
in `include/Fabric/IR/FabricOps.td`; verifier rules live in
`lib/Fabric/IR/FabricOps.cpp`.

## Schedule predicate

`fabric.pe` is a single op specialized by a mandatory `schedule` enum
attribute. The schedule appears in `[...]` immediately after the op
keyword (and after the optional `@sym_name`), mirroring
`fabric.op [@arith.muli]`. The op exists in two disjoint syntactic
forms by `sym_name` presence (anonymous vs named template).

Anonymous form:

```mlir
%out:2 = fabric.pe [spatial] (%a = %x : !fabric.bits<32>)
                   -> (!fabric.bits<32>, !fabric.bits<32>) {
  fabric.fu(...) -> ...
}
```

Named template form:

```mlir
fabric.pe @ALU [spatial] (!fabric.bits<32>, !fabric.bits<32>)
                         -> (!fabric.bits<32>) {
^bb0(%pa: !fabric.bits<32>, %pb: !fabric.bits<32>):
  fabric.fu(...) -> ...
  fabric.yield %pa : !fabric.bits<32>
}
```

* `spatial`: at most one inner `fabric.fu` is architecturally active per
  PE configuration. Routing between PE ports and the active FU's ports
  is described by the PE's runtime configuration record (see "Software
  configuration"). The verifier rules in this document apply to
  this branch.
* `temporal`: time-multiplexes multiple FUs / instructions through the
  PE. The temporal-branch IR shape, hardware parameters, and software
  configuration record are documented in
  `spec-fabric-pe-temporal.md`.

The `schedule` predicate is orthogonal to the container kind.
`fabric.pe`, `fabric.switch`, and `fabric.mem` share the same
`[spatial] | [temporal]` predicate convention in the SpatialCore tile
matrix. Cross-reference `docs/spec-fabric-module.md` for SpatialCore
placement of `fabric.pe`.

In any given configuration of a `fabric.pe [spatial]`, the inner FU
selects exactly zero or one of its FUs as architecturally active.

## Background

`fabric.fu` already models a CGRA-style functional unit: a PE-internal
graph-region container of `fabric.op` / `fabric.mux` / `fabric.demux`
whose inner sw_configs materialize different software graphs.
`fabric.pe [spatial]` wraps a *set* of such FUs so that one PE-level
configuration picks which FU is active and how the PE's external ports
are wired to that FU's ports.

Compared to a bare `fabric.fu`:

* The PE adds a top-level `opcode` selector across multiple FUs.
* The PE adds explicit input-mux and output-demux fields that route the
  PE's external ports to the active FU's ports.
* The PE provides a single, unified runtime configuration record whose
  layout is fixed and self-describing.

The PE input-mux and output-demux terms refer to local configuration
fields inside a SpatialCore template. They are not `fabric.system`
interconnect primitives. System-level routing and arbitration use the
primitive node kinds specified in `docs/spec-fabric-system-adg.md`.

Compared to allowing arbitrary multi-FU placement:

* At most one physical `fabric.fu` inside a `fabric.pe [spatial]` may be
  architecturally active per configuration. This is a hard legality rule,
  not a placement preference.

## Structural model

### Anonymous vs named template form

The op carries an optional `sym_name`. The two forms are syntactically
disjoint:

* **Anonymous form** (no `sym_name`). Definition + use combined: the
  op has variadic SSA operands and variadic SSA results, the body has
  no terminator (per the PE-level routing model below).
* **Named template form** (with `sym_name`). Template-only: zero SSA
  operands, zero SSA results in the enclosing `fabric.module` body;
  signature recorded in a `function_type : FunctionType` attribute.
  Body's entry block arguments match `function_type.getInputs()`.
  Body terminator is `fabric.yield` whose value list matches
  `function_type.getResults()`. Actual usage goes through
  `fabric.instantiate @sym(...)`. See `docs/spec-fabric-instantiate.md`
  for symbol resolution and the per-form `fabric.instantiate` rules.

Both forms share the body whitelist and the `bits<W>` uniform width
rule below.

### Body whitelist

The body of `fabric.pe [spatial]` is a single block whose only legal
contents are `fabric.fu` ops and `fabric.instantiate` ops (the latter
must resolve to a `fabric.fu` symbol; see
`docs/spec-fabric-instantiate.md`). The named template form
additionally terminates the body with `fabric.yield`. No other op
kind is permitted in the body. Specifically:

* No `fabric.op`, `fabric.mux`, `fabric.demux`, or `fabric.fifo` may
  appear directly in the PE body. They are only allowed inside an
  inner `fabric.fu`.
* No `fabric.yield` may appear directly in the anonymous-form PE
  body. The anonymous PE has no terminator (see below). In the named
  template form `fabric.yield` is the body terminator and supplies the
  values that match the PE's declared `function_type` results.
* No `fabric.pe [spatial]`, `fabric.pe [temporal]`, `fabric.module`, or
  any other body-bearing fabric op may be nested directly inside a
  PE body.
* No non-fabric ops (e.g. `arith.*`, `func.*`, `dataflow.*`) may
  appear in the PE body. They live inside `fabric.fu` (wrapped by
  `fabric.op`) or higher-level dataflow regions, not the PE.

The PE body must contain at least one compute resource: either a
`fabric.fu` directly, or a `fabric.instantiate` whose resolved callee
is a `fabric.fu`.

The same body whitelist applies to `fabric.pe [temporal]`: a temporal
PE body is also restricted to `fabric.fu` ops. The two PE kinds
differ only in how their FUs are time-shared at the PE level, not in
what may be placed in their bodies.

### Body shape

An anonymous-form `fabric.pe [spatial]` body holds one or more
`fabric.fu` instances -- the PE's FU set. The anonymous-form body has
no terminator: there is no `fabric.yield` directly inside that PE body.
The PE's external inputs and outputs are not wired by SSA values that
flow through a yield; instead they are implicitly wired to inner FU
ports through the PE-level input-mux and output-demux fields (see
"Software configuration").

Inner `fabric.fu` instances may be homogeneous (every FU has the same
op_list / hw_params shape) or heterogeneous. The PE imposes no rule on
FU similarity beyond what the verifier requires for individual FUs.

Conceptually, the PE level organizes one or more FU resources and, in
temporal mode, describes how those resources may be time-multiplexed.
It is not the software partition boundary: `dataflow.subgraph` maps to
`fabric.fu` candidates, and place and route operates at that
`dataflow.subgraph` -> `fabric.fu` granularity before considering the
enclosing PE resource context.

PE-level routing (input mux, output demux) is expressed entirely
through the PE's `sw_configs`, not through nested IR ops, because PE-
level routing is a fixed structural pattern that does not need first-
class ops to vary. This is why `fabric.mux` / `fabric.demux` belong
inside `fabric.fu` (where they reshape in-FU connectivity) and not in
the PE body.

### FU-boundary truncation

Inside a fabric.pe [spatial] body, a `fabric.fu` may declare a block argument
of type `!fabric.bits<F>` while its operand SSA source is the PE's
`!fabric.bits<W>` value (`W >= F`). The textual form is
`(%fa = %src : !fabric.bits<W> to !fabric.bits<F>)` -- without the
`to ...` clause, the inner type defaults to the outer type. The high
`W - F` bits of the source are dropped; the
inner block argument carries the low `F` bits.

This input-direction relaxation lets ops with narrower inner ports
(notably `fabric.op[@dataflow.constant]` whose ctrl input is
`!fabric.bits<0>`) live inside a PE that runs at a wider uniform
width. Output ports remain strict: the `fabric.yield` value types
must equal the FU's outer result types, and those equal the PE's
`bits<W>`.

## Hardware parameters

All hardware parameters of `fabric.pe [spatial]` are either declared on the
op signature or inferred from the FU set in the body. None of them are
attribute knobs.

### Explicit (from op signature)

`K` -- the number of PE input ports. Equal to the number of operands of
the `fabric.pe [spatial]` op.

`L` -- the number of PE output ports. Equal to the number of results of
the `fabric.pe [spatial]` op.

`W` -- the PE bit width. Every PE input and every PE output must be
`!fabric.bits<W>`, and every port must use the same `W`. Mixing widths
or using non-`bits` fabric types (e.g. `bits_tag`) on the PE boundary
is a verifier error. Mixing widths inside the FU body is permitted; the
PE boundary is the place where one uniform width is enforced.

These three are not attributes; they are read directly from the op's
operand/result lists.

### Explicit (from body)

The PE's FU set is the body's list of `fabric.fu` ops, in body
definition order. The PE has no separate `op_list` -- the FU set is the
op_list at the PE level. FU definition order is the same order used for
opcode numbering.

### Implicit (derived from the FU set)

`max_fu_inputs` -- the maximum, across all FUs in the body, of each FU's
input-port count. If FU `f0` has 2 inputs and FU `f1` has 3 inputs,
`max_fu_inputs = 3`.

`max_fu_outputs` -- the maximum, across all FUs in the body, of each
FU's output-port count. Same shape.

`fu_config_bitwidth(fu_i)` -- the total bit width needed to encode all
runtime sw_config knobs inside FU `i`. This is the sum of:

* every `fabric.mux` mode field inside `fu_i` (each is
  `[sel | discard | disconnect]` with `sel` width `log2Ceil(N_mux)`
  where `N_mux` is that mux's port count);
* every `fabric.demux` mode field inside `fu_i` (same shape);
* every `fabric.op` runtime axis inside `fu_i` (`op_sel` if the op_list
  has more than one member; `bitmask` for variadic `dataflow.sync` /
  `dataflow.mux` / `dataflow.demux`; attribute axes such as `predicate`,
  `step_op`, `cont_cond`, `const_hex_value` when restricted by
  `hw_params`). The exact set of runtime axes is the canonical set
  defined in `spec-fabric-reconfigurable-op.md`.

Inner `fabric.op`s and inner mux/demux without runtime axes contribute
zero bits.

`fu_config_bitwidth_max` -- the maximum of `fu_config_bitwidth(fu_i)`
across all FUs in the body. Because at most one FU is active per PE
configuration, the PE allocates one shared payload sized to the maximum;
the unused tail bits are don't-care for inactive FUs.

`fu_config_bitwidth` and the related width formulas below are unaffected
by FU-boundary truncation: they sum sub-field widths internal to each
FU (mux/demux mode fields and per-op runtime axes), not the FU's
external port widths.

### Verifier constraints on hardware parameters

* `K >= 1` and `L >= 1`. A PE must have at least one input and at least
  one output.
* Every operand and every result of `fabric.pe [spatial]` has type
  `!fabric.bits<W>` with the same `W >= 1`. Violations report
  `'fabric.pe [spatial]' op requires uniform 'bits<W>' on all PE ports`.
* The body must contain at least one `fabric.fu`. An empty PE is
  rejected.
* For every inner FU `f`, `f.numInputs() <= K` and
  `f.numOutputs() <= L`. Equivalently, `max_fu_inputs <= K` and
  `max_fu_outputs <= L`. Violations report which FU exceeded the bound.
* Every inner FU's outer port types (the operand and result types of
  the `fabric.fu` op itself, as visible in the fabric.pe [spatial] body) must
  be `!fabric.bits<W>` matching the PE's `W`. Inner FU input port
  types (the FU body's block argument types) may be any
  `!fabric.bits<F>` with `F <= W`. When `F < W`, the high `W - F`
  bits of the incoming PE data are dropped at the FU boundary
  (high-bit truncation, hardware-implemented). Symmetrically, an
  inner FU body value yielded via
  `fabric.yield %v : !fabric.bits<G> to !fabric.bits<W>` may carry an
  inner width `G <= W`; hardware zero-fills the high `W - G` bits at
  the FU boundary so the value reaching the PE port is strict
  `!fabric.bits<W>`. The PE's uniform-W invariant constrains the
  PE's port-list types and the FU's outer (op-level) input/result
  types only -- inner FU-body block-arg types and inner yield value
  types are not constrained by the PE's `W`.
* The anonymous-form body contains only `fabric.fu` ops. There is no
  terminator: the region uses MLIR's no-terminator form. Placing
  `fabric.op` / mux / demux / fifo / yield directly in the anonymous
  PE body, or nesting another `fabric.pe [spatial]`, is rejected.

## Software configuration (the PE instruction word)

The PE has a single self-describing runtime configuration record that
the configuration generator emits as one bit string. The fields are
listed below from least-significant bit (LSB) to most-significant bit
(MSB).

```
+----------------------------------------------------------------+
|              SPATIAL_PE INSTRUCTION (MSB -> LSB)               |
+-----------------+-----------------+-----------------+----------+
| fu_sw_configs   | output demux x  | input mux x     | opcode   |
|                 | max_fu_outputs  | max_fu_inputs   |          |
+-----------------+-----------------+-----------------+----------+
                                                            | enable
                                                            +-(LSB)
```

Equivalently, low-to-high:

1. `pe_enable` (1 bit, LSB)
2. `opcode` (`O = log2Ceil(num_fu)` bits)
3. `max_fu_inputs` input-mux fields, each `[sel | discard | disconnect]`
4. `max_fu_outputs` output-demux fields, each
   `[sel | discard | disconnect]`
5. `fu_sw_configs` payload of `fu_config_bitwidth_max` bits

### `pe_enable`

Bit `[0]`. When `0`, the PE is architecturally inactive: it produces no
output activity and may be clock- or power-gated. When `1`, the
remaining fields select an FU and its routing.

The default reset value is `0`. An unprogrammed PE serializes to a word
whose `enable` bit is `0`, all mux/demux fields have `disconnect = 1`,
and all FU sw_config bits are `0`.

### `opcode`

`O = log2Ceil(num_fu)` bits, where `num_fu` is the count of inner FUs.
Numbering is `0` for the first `fabric.fu` in body definition order,
`1` for the second, and so on. When `num_fu = 1`, `O = 0` and the
opcode field is omitted.

When `enable = 1`, exactly one FU is active: the FU whose body index
equals `opcode`.

### Input-mux fields

There are `max_fu_inputs` such fields. Field `i` describes how PE input
port indices are routed onto the active FU's input port `i`. Each
field's low-to-high layout is:

```
| sel | discard | disconnect |
```

* `sel`   : `log2Ceil(K)` bits (`0 <= sel < K`). Selects which PE input port (0 .. K-1)
  feeds FU input `i`. Hardware must treat `sel` as 0 when `disconnect =
  1` regardless of the encoded value.
* `discard`    : 1 bit. When set, the FU input is locally drained:
  the FU input's `valid` is forced low and the selected PE input's
  `ready` is forced high so upstream tokens dissipate.
* `disconnect` : 1 bit. When set, the FU input is inert: the selected
  PE input's `ready` is forced low and the FU input's `valid` is forced
  low.

Constraints on a single field:

* It is illegal to set `discard = 1` and `disconnect = 1` simultaneously
  (`'fabric.pe [spatial]' op input mux N has both discard and disconnect`).
* When `enable = 0`, all input-mux fields must serialize as
  `disconnect = 1`, `discard = 0`, `sel = 0`.

If the active FU has fewer than `max_fu_inputs` input ports, the
trailing input-mux fields (those whose index is greater than or equal
to that FU's input count) are ignored at runtime. They must still
follow the discard/disconnect mutual-exclusion rule when programmed.

The PE input mux is a **selector only**, not a fan-in. It must not be
used to merge two distinct software flows onto one FU input. Flow
mixing belongs to a higher-level switch / fabric structure
(`fabric.switch`), not the PE input mux.

### Output-demux fields

There are `max_fu_outputs` such fields. Field `j` describes how the
active FU's output port `j` is routed onto a PE output port. Each
field's low-to-high layout is:

```
| sel | discard | disconnect |
```

* `sel`   : `log2Ceil(L)` bits (`0 <= sel < L`). Selects which PE output port (0 .. L-1)
  the FU output `j` drives.
* `discard`    : 1 bit. When set, the FU output is drained locally
  (FU output's `ready` is forced high; no PE output sees the value).
* `disconnect` : 1 bit. When set, the route is severed; FU output's
  `ready` is forced low and the selected PE output's `valid` is forced
  low.

Constraints:

* `discard = 1` and `disconnect = 1` simultaneously is illegal
  (`'fabric.pe [spatial]' op output demux N has both discard and disconnect`).
* When `enable = 0`, all output-demux fields must serialize as
  `disconnect = 1`, `discard = 0`, `sel = 0`.
* The PE output demux is a **selector only**, not a fan-out. It must
  not be used to broadcast one FU output to multiple PE output ports.
  Broadcast belongs to higher-level switches.

If the active FU has fewer than `max_fu_outputs` output ports, the
trailing output-demux fields are ignored at runtime.

### `fu_sw_configs` payload

A single contiguous payload of `fu_config_bitwidth_max` bits. The
payload is interpreted **per active FU**: when `opcode = i`, the bits
encode `fu_i`'s runtime sw_configs in body definition order. The
reading rule for one FU's slice:

* For each `fabric.op` / `fabric.mux` / `fabric.demux` inside `fu_i`,
  in body definition order, append that op's runtime sw_config sub-field
  to the slice. The sub-field shape is determined by the op kind and by
  any `hw_params` restrictions, exactly as defined in
  `spec-fabric-reconfigurable-op.md`. Ops with no runtime axes
  contribute zero bits.

If `fu_config_bitwidth(fu_i)` is less than `fu_config_bitwidth_max`,
the unused upper bits of the payload are don't-care when `opcode = i`.
The configuration generator must zero them.

## Width formulas

Let:

* `K` = PE input count (`>= 1`)
* `L` = PE output count (`>= 1`)
* `num_fu` = number of inner FUs (`>= 1`)
* `O = log2Ceil(num_fu)` (zero when `num_fu = 1`)
* `mux_field_width = log2Ceil(K) + 2` (the `+2` is `discard | disconnect`)
* `demux_field_width = log2Ceil(L) + 2`
* `fu_config_bitwidth_max` as defined above

```
pe_word_width =
    1                                       // enable
  + O                                       // opcode
  + max_fu_inputs  * mux_field_width        // input-mux fields
  + max_fu_outputs * demux_field_width      // output-demux fields
  + fu_config_bitwidth_max                  // active-FU runtime config
```

When `num_fu = 1`, `O = 0` and the opcode contribution is zero.

## Default reset configuration

The unprogrammed serialization of a `fabric.pe [spatial]` is:

* `pe_enable = 0`
* `opcode = 0`
* every input-mux field: `disconnect = 1`, `discard = 0`, `sel = 0`
* every output-demux field: `disconnect = 1`, `discard = 0`, `sel = 0`
* every bit of `fu_sw_configs` payload: `0`

This serialization is what the configuration generator emits when no
PE-level `sw_configs` attribute is attached, and it is the value the
hardware must come out of reset in.

## Mapping implications

The mapper and tech-mapping passes must respect the following:

* Within one `fabric.pe [spatial]`, at most one physical `fabric.fu` is
  active per mapped configuration. Two distinct physical FUs from the
  same PE cannot be co-active. A mapping that requires two physical
  FUs from the same PE is illegal and must be rejected.
* Multiple software ops may still be tech-mapped onto the same active
  FU through the FU's own internal sw_configs (mux/demux/op_sel/etc.).
* `fabric.fu` instances inside a `fabric.pe [spatial]` are compute
  resources. They may terminate routed edges at FU inputs or originate
  routed edges at FU outputs, but they must not be used as generic
  transit hops to relay unrelated traffic.
* For every routed software edge that touches a PE, the mapping result
  must record: which PE input was used, which FU input it reached,
  which FU output produced the value, and which PE output carried it
  outward. This information feeds both configuration generation and
  any visualization layer.

## Placement rules

The SpatialCore/CGRA template container is `fabric.module`. See
`spec-fabric-module.md` for the full module-side specification:
declared inputs/outputs, body whitelist, the three connection points
that admit width relaxation (module-input -> sub-module-operand,
sub-module-result -> sub-module-operand, sub-module-result ->
module-yield), and the `IsolatedFromAbove` requirement that bars
external SSA leakage.

Four fabric ops carry body regions at the SpatialCore level:
`fabric.module`, `fabric.pe [spatial]`, `fabric.pe [temporal]`, and
`fabric.fu`. Anonymous PEs contain FU resources or FU instantiations
and have no terminator. Named PE templates contain the same resource
forms and use `fabric.yield` to match their declared function type.
Inner FUs are connected to PE external ports through PE-level input-mux
and output-demux fields rather than through PE-body SSA yields in the
anonymous form.

`fabric.pe [spatial]` placement rules:

* `fabric.pe [spatial]` must be inside a `fabric.module` body. It may
  not appear at the top of `builtin.module`, inside a `func.func`, or
  inside any non-fabric container.
* `fabric.pe [spatial]` may not be nested inside another
  `fabric.pe [spatial]`, inside `fabric.pe [temporal]`, or inside any
  `fabric.fu`. The verifier rejects nested PEs.
* Inner `fabric.fu` instances may only appear inside a
  `fabric.pe [spatial]` or a `fabric.pe [temporal]` body. They may not
  appear directly inside `fabric.module` or any other container.

## Errors (verifier)

The verifier emits free-form diagnostics for the following conditions:

* PE port type or width violations (mixed types, mixed widths,
  non-`bits` types, `K < 1` or `L < 1`).
* Empty body (no `fabric.fu`).
* Anonymous-form body contains an op other than `fabric.fu` (in
  particular, no `fabric.yield` may appear directly inside an
  anonymous-form `fabric.pe [spatial]`).
* `max_fu_inputs > K` or `max_fu_outputs > L`.
* An inner FU's outer port width does not match the PE's `W` (input
  or output). Inner block-arg widths narrower than the outer operand
  width are accepted; widening (`outer < inner`) is rejected by the
  FU's own verifier.
* `discard` and `disconnect` simultaneously set on any mux or demux
  field.
* Nesting violations: `fabric.pe [spatial]` placed outside a
  `fabric.module` body, nested inside another `fabric.pe [spatial]` /
  `fabric.pe [temporal]` / `fabric.fu`.

The exact diagnostic strings are defined in `FabricOps.cpp`; this
document fixes the set of conditions that must trigger a diagnostic.

## Maintenance

Implementation locations that must mirror this spec are:

* `Fabric_PeOp` in `include/Fabric/IR/FabricOps.td` for the IR
  shape;
* `PeOp::verify` in `lib/Fabric/IR/FabricOps.cpp` for the
  verifier rules;
* the related runtime-axis catalogue in
  `spec-fabric-reconfigurable-op.md` for the per-op sub-field shapes
  that contribute to `fu_sw_configs`;
* the share-group catalogue in `spec-fabric-hw-share-group.md` for
  legal multi-member `op_list`s inside inner FUs;
* `spec-fabric-instantiate.md` for the rules governing
  `fabric.instantiate` inside a PE body and named `fabric.fu`
  definitions.

When extending `fabric.pe [spatial]` (for instance, adding a new field
to the instruction word), update this document and add a unit test
under `test/fabric/unit/pe/` that pins the new layout.
