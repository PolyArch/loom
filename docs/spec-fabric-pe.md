# Fabric PE

This document specifies `fabric.pe`, a processing element container that
holds one or more `fabric.fu` instances. The op carries a mandatory
`schedule` predicate (`spatial` | `temporal`) that selects how the
contained FUs are time-shared.

## Schedule predicate

`fabric.pe` is a single op specialized by a mandatory `schedule` enum
attribute. The schedule appears in `[...]` immediately after the op
keyword (and after the optional `@sym_name`), mirroring
`fabric.op [@arith.muli]`. The op exists in two disjoint syntactic
forms by `sym_name` presence (anonymous vs named template).

`schedule` is a closed typed enum with exactly `spatial` and `temporal` in
this profile. A string-valued mode discriminator is invalid.

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
  fabric.yield
}
```

* `spatial`: at most one inner `fabric.fu` is architecturally active per
  PE configuration. Routing between PE ports and the active FU's ports
  is described by the PE's configured view (see "Configured Spatial PE
  View"). The verifier rules in this document apply to
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

In any given configuration, a `fabric.pe [spatial]` activates at most one
contained FU. A disabled PE activates none.

## Boundary selector crosspoint contract

Every spatial or temporal PE boundary is implemented by finite selector logic
between the PE ports and its internal FU-facing ports. For both schedule
branches, let `K` be the PE input-port count and `L` the PE output-port count,
derived from the anonymous SSA signature or named `function_type`. Both counts
must be positive, and the overflow-safe product must satisfy `K * L <= 64`.

A PE with `K * L <= 16` is valid without an efficiency diagnostic. A PE with
`16 < K * L <= 64` remains valid but verification emits one non-fatal
implementation-efficiency warning for that occurrence. The warning is
advisory, does not enter Fabric identity, and changes neither Mapping legality
nor selector capacity. A product greater than 64 is invalid. Thus `4 x 4` is
quiet, `4 x 5` warns, `8 x 8` warns but remains valid, and `9 x 8` is invalid.

The product models the PE boundary selector fabric, not FU functionality.
Architectures that require more boundary connectivity compose multiple PEs and
explicit Fabric routing resources instead of hiding a larger crossbar inside
one PE. Switches use the same overflow-safe crosspoint arithmetic but retain
their independently owned warning and hard thresholds from
`spec-fabric-switch.md`.

## Background

`fabric.fu` models a CGRA-style functional unit: a PE-internal graph-region
container of `fabric.op`, `fabric.mux`, and `fabric.demux` resources whose
finite structural/capability templates admit exact TechMapping realizations.
`fabric.pe [spatial]` wraps a set of such FUs so that a SpatialMapping
configuration selects one concrete FU occurrence, its resident context, and
the PE-boundary routes to that FU's ports.

Compared to a bare `fabric.fu`:

* The PE adds a top-level selected-FU field across multiple FUs.
* The PE adds explicit input-mux and output-demux fields that route the
  PE's external ports to the active FU's ports.
* The PE provides one Fabric-owned typed configuration schema. The finalizer
  derives its configured field values from TechMapping and SpatialMapping;
  `ConfigurationABI` alone defines their physical encoding.

The PE input-mux and output-demux terms refer to local configuration
fields inside a SpatialCore template. They are not `fabric.system`
interconnect primitives. System-level routing, replication, and arbitration
belong to the typed Transport Architecture and Interconnect Implementation
owned by `fabric.system`; `docs/spec-fabric-system-adg.md` and the SystemMapping
profile own their exact hardware and Mapping schemas.

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
  Body terminator is a zero-operand `fabric.yield`. The function type alone
  owns PE result port types, while configured output selectors choose their
  internal sources. Actual usage goes through
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
  template form zero-operand `fabric.yield` is only the signature terminator;
  it does not duplicate the PE's configured output selections.
* No `fabric.pe [spatial]`, `fabric.pe [temporal]`, `fabric.module`, or
  any other body-bearing fabric op may be nested directly inside a
  PE body.
* No non-fabric ops (e.g. `arith.*`, `func.*`, `dataflow.*`) may
  appear in the PE body. They live inside `fabric.fu` (wrapped by
  `fabric.op`) or higher-level dataflow regions, not the PE.

The PE body must contain at least one concrete compute resource: either an
anonymous `fabric.fu` occurrence or a `fabric.instantiate` whose resolved
callee is a `fabric.fu`. A named FU declaration is a reusable template, not a
physical resource, and does not satisfy this requirement by itself.

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
"Configured Spatial PE View").

Inner `fabric.fu` instances may be homogeneous (every FU has the same
op_list / hw_params shape) or heterogeneous. The PE imposes no rule on
FU similarity beyond what the verifier requires for individual FUs.

All PE shape, capacity, selector, and configuration rules consume one
`ConcreteFuOccurrenceSet`. Before finalization it consists of anonymous
`fabric.fu` occurrences plus each `fabric.instantiate` that resolves to an FU;
named FU declarations are excluded. Canonical elaboration replaces every such
instantiate with one fresh anonymous FU occurrence and removes all instantiate
ops, so the finalized set is exactly the anonymous FU occurrences in the PE.
No verifier or projection may count both a named declaration and its
instantiation.

Conceptually, the PE level organizes one or more FU resources and, in temporal
mode, describes how those resources may be time-multiplexed. It is not the
software partition boundary. TechMapping records actor groups, a selected FU
structural/capability template, and exact ordered actor/op/port/FU-boundary
correspondence before place and route considers concrete PE resources.
Canonical Dataflow IR does not persist those target-specific groups.

PE-level input and output selection uses typed configuration fields rather than
nested routing ops. SpatialMapping owns the concrete FU occurrence,
`InstructionContextRef`, and PE-boundary routing choices. The finalizer derives
the corresponding PE `sw_configs`. This is separate from FU-local
`fabric.mux` and `fabric.demux`, whose selected topology is part of
TechMapping when it changes the configured software graph.

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

They describe hardware capacity and typed configuration domains, not selected workload
configuration. Inner `fabric.op.hw_params` remains part of each operation's
parameterized capability; `sw_configs` is derived only after TechMapping and
SpatialMapping.

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

The PE's FU set is its `ConcreteFuOccurrenceSet`. The PE has no separate
`op_list`; `op_list` remains a projection of one concrete `fabric.op`
capability. Body authoring order and named-template declaration order do not
change FU identity or configured semantics.

### Implicit (derived from the FU set)

`max_fu_inputs` -- the maximum, across the concrete FU occurrence set, of each
FU's input-port count. If FU `f0` has 2 inputs and FU `f1` has 3 inputs,
`max_fu_inputs = 3`.

`max_fu_outputs` -- the maximum, across the concrete FU occurrence set, of each
FU's output-port count. Same shape.

Each inner FU also exposes its typed configuration-field domain. That domain
is interpreted together with the FU's hardware parameters and exact selected
software configuration; it is not a PE bit-width parameter. Physical field
positions, packing, padding, and total width are derived only by the selected
`ConfigurationABI`.

### Verifier constraints on hardware parameters

* `K >= 1` and `L >= 1`. A PE must have at least one input and at least
  one output.
* Every operand and every result of `fabric.pe [spatial]` has type
  `!fabric.bits<W>` with the same `W >= 1`. Violations report
  `'fabric.pe [spatial]' op requires uniform 'bits<W>' on all PE ports`.
* The `ConcreteFuOccurrenceSet` must be non-empty.
* For every concrete FU occurrence `f`, `f.numInputs() <= K` and
  `f.numOutputs() <= L`. Equivalently, `max_fu_inputs <= K` and
  `max_fu_outputs <= L`. Violations report which FU exceeded the bound.
* Every concrete FU occurrence's outer port types (the operand and result types of
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
* A finalized anonymous-form body contains only anonymous `fabric.fu`
  occurrences. There is no terminator: the region uses MLIR's no-terminator
  form. Pre-finalization authoring may also contain named FU declarations and
  `fabric.instantiate` under the instantiate specification. Placing
  `fabric.op` / mux / demux / fifo / yield directly in the anonymous
  PE body, or nesting another `fabric.pe [spatial]`, is rejected.

## Configured Spatial PE View

The derived configured view is one closed sum:

```text
SpatialPeConfiguration =
    Disabled
  | Active {
      selected_fu
      input_selections
      output_selections
      fu_sw_configs
      physical_refinements
    }
```

`Disabled` carries no selected FU, selector, tag, or FU configuration. The
physical enable bit and inactive encoding belong only to ConfigurationABI.
The `Active` variant contains one selected-FU field, typed input and output
selector records, and the selected FU's typed configuration fields.
TechMapping owns the exact FU realization;
SpatialMapping owns the concrete FU/context/boundary routing and
semantic-preserving refinements. The finalizer derives the view from those
facts. It is not an independent capability or Mapping authority.

Fabric owns each field's semantic meaning and legal value domain.
`ConfigurationABI` alone owns bit positions, field widths, packing, padding,
reset encoding, and the programming representation.

### Configuration field identity and codec

One Spatial PE owns one static, factorized field schema. It contains:

* one activation field for the PE;
* one input-selector field for every outer input of every concrete FU
  occurrence owned by the PE; and
* one output-selector field for every outer output of every concrete FU
  occurrence owned by the PE.

The activation field is ordinal zero. Input-selector fields follow in canonical
`(FU occurrence, outer input ordinal)` order, then output-selector fields in
canonical `(FU occurrence, outer output ordinal)` order. Every field is a
`FabricSemanticConfigFieldRef` owned by the PE occurrence. The sealed PE
configuration-schema view relates each field reference to its exact role and
`FabricFuOccurrencePortRef`; a caller never recovers that role by interpreting
the ordinal. An input-selector field targets an Input port and an
output-selector field targets an Output port.

The activation field has this finite semantic domain:

```text
PeActivation = Disabled | Active(FabricFuOccurrenceRef)
```

Its active references are exactly the concrete FU occurrences owned by the PE.
The canonical value codec uses `u32be(0)` for `Disabled` and `u32be(1)`
followed by the canonical FU occurrence reference bytes for `Active`. A
ConfigurationABI must use canonical `Disabled` as this field's
`inactive_value`; another value could activate an unmapped PE and is invalid.

The exact `InputSelection` and `OutputSelection` domains are defined below.
Every carried endpoint is a `FabricTransportEndpointRef` whose owner is this PE
and whose Fabric-owned endpoint role has the required direction. The input
codec uses `u32be(0)`, `u32be(1)`, and `u32be(2)` for `Disconnected`, `Route`,
and `Discard`, followed by canonical endpoint reference bytes for the two
variants that carry an endpoint. The output codec uses the same three tags;
only `Route` carries canonical endpoint reference bytes. A foreign owner,
wrong endpoint direction, or endpoint outside the PE inventory is not in the
domain. Reference payloads use the exact Fabric local-reference bytes defined
by `spec-fabric-identity.md`, without `ArtifactLocalReference` framing. Decode
rejects an unknown tag, truncation, trailing bytes, or a value that does not
re-encode byte-for-byte identically.

These PE fields do not absorb FU operation fields. A resolved capability owns
each operation field through its exact `FabricFuTemplateNodeRef`. The existing
template-to-occurrence relation mechanically replaces that owner with the
corresponding `FabricFuOccurrenceNodeRef` in a configured physical projection,
while preserving the field ordinal, domain, and codec. It does not create a
second field or another domain owner. `fu_sw_configs` is the configured
composition of those projected fields for the selected FU, not a copy inside a
PE codebook. A `Disabled` configured view emits no Mapping projection row for
any PE or FU field; the ABI substitutes the activation field's mandatory
`Disabled` inactive value and the other validated inactive values. An `Active`
view projects `Active(f)`, every selector field for `f`, and the required FU
fields; fields belonging to other FUs remain omitted and their ABI-declared
inactive values are unobservable.

The schema is finite without enumerating the Cartesian product of FU choice,
port routes, and FU operation behavior. Physical code assignment and packing
remain solely ConfigurationABI facts.

### `selected_fu`

`selected_fu` identifies one member of the PE's concrete FU set. Its legal
domain is exactly the FU occurrences owned by this PE. It is the `Active`
payload of the PE activation field rather than a second field.

In the `Active` variant, exactly one FU is active: the FU selected by
`selected_fu`.

### Input-mux fields

One typed record per active-FU input describes how a PE input is routed onto
that FU input. The static field inventory contains the corresponding record for
every concrete FU input, while the configured view projects only the selected
FU's records. A value is one member of a closed typed sum:

```text
InputSelection = Disconnected
               | Route(FabricTransportEndpointRef)
               | Discard(FabricTransportEndpointRef)
```

* `Route` selects one of this PE's `K` Input-role transport endpoints and
  connects it to the FU input.
* `Discard` drains the selected PE input locally:
  the FU input's `valid` is forced low and the selected PE input's
  `ready` is forced high so upstream tokens dissipate.
* `Disconnected` makes the FU input inert: no PE input is selected and the FU
  input's `valid` is forced low.

The semantic view has no trailing records beyond the active FU's input count.
A fixed-capacity hardware representation and any padding belong to
`ConfigurationABI`.

The PE input mux is a **selector only**, not a fan-in. It must not be
used to merge two distinct software flows onto one FU input. Flow
mixing belongs to a higher-level switch / fabric structure
(`fabric.switch`), not the PE input mux.

`Discard` is an explicit PE-boundary behavior. It cannot make an invalid
FU-internal broadcast or mutually exclusive topology legal, and it cannot
discharge a logical edge that the exact Mapping realization requires.

### Output-demux fields

One typed record per active-FU output describes how that output is routed onto
a PE output. The static field inventory contains the corresponding record for
every concrete FU output, while the configured view projects only the selected
FU's records:

```text
OutputSelection = Disconnected
                | Route(FabricTransportEndpointRef)
                | Discard
```

* `Route` selects one of this PE's `L` Output-role transport endpoints.
* `Discard` drains the FU output locally
  (FU output's `ready` is forced high; no PE output sees the value).
* `Disconnected` severs the route and forces the FU output's `ready` low.

* The PE output demux is a **selector only**, not a fan-out. It must
  not be used to broadcast one FU output to multiple PE output ports.
  Broadcast belongs to higher-level switches.

The semantic view has no trailing records beyond the active FU's output count.

Output `Discard` is likewise an explicit PE-boundary sink. It cannot replace
the FU-local demux/mux topology required to prevent tokens from entering an
inactive internal branch.

### `fu_sw_configs`

`fu_sw_configs` is the selected FU's closed typed configuration record. Its
values come from the selected TechMapping template and exact ordered
correspondence, exact Dataflow-owned semantics accepted by the capability,
mechanically derived values such as a sync active set, and
SpatialMapping-selected semantic-preserving physical refinements.

Software mux/demux runtime selectors remain data operands and never appear in
this record. Fields with singleton domains and values irrelevant under the
selected relation do not create alternate semantic configurations. The
`ConfigurationABI` encodes this typed record for the selected hardware.

## Reset State

At reset, a spatial PE is in the canonical `Disabled` state and has no selected
software realization. Reset does not create a default TechMapping candidate,
`sw_configs` authority, or `InstructionContextRef` identity. The exact reset
bit pattern and programming sequence belong to `ConfigurationABI`.

## Mapping Ownership

TechMapping selects the FU structural/capability template and binds exact
software actors to inner operations, ordered operation ports, and fixed FU
boundary ports. It does not select a concrete PE or FU occurrence and does not
persist raw configuration fields.

SpatialMapping selects the concrete FU occurrence and resident instruction
context. `docs/spec-fabric-identity.md` owns the persistent
`InstructionContextRef` framing; the PE owns the context inventory and range:

```text
InstructionContextRef =
  (FabricPeOccurrenceRef, ContextOrdinal)

ComputeRealizationRef -> selected FabricFuOccurrenceRef
ComputeRealizationRef -> selected InstructionContextRef
```

A Spatial PE mechanically provides only `ContextOrdinal = 0`. The reference
names only the resident configuration/runtime-state namespace in which the
configured graph executes. It does not own or copy that graph, its capability,
or its exact semantic realization. It is not an FU-local or operation-local
context, an encoding identifier, or a duplicate configuration record. The
final verifier must prove that the selected FU occurrence and
`InstructionContextRef` have the same parent PE.

SpatialMapping also owns the PE input/output selector choices and the concrete
routes beyond the FU boundary. The combined Mapping facts must retain which PE
input reaches each selected FU input and which selected FU output reaches each
PE output. The finalizer derives `selected_fu`, boundary selectors, and
`fu_sw_configs` from these facts and the Fabric field domains. The selected
`ConfigurationABI` alone encodes those fields.

Within one `fabric.pe [spatial]`, at most one physical `fabric.fu` is active in
a programmed configuration. Multiple exact software actors may belong to one
TechMapping realization when the selected FU template and parameterized
capability support the complete actor group. They are not made legal by raw
`sw_configs` or by an inactive branch drain.

The configured FU is a physical graph/capability boundary, not a macro firing
or actor-atomicity boundary. Each active operation follows its Canonical
Dataflow actor transition and ordered publication semantics; context residency
does not introduce a group-level readiness or commit event.

A Canonical Dataflow edge disappears from external routing only when an
explicit configured-FU relation, configured-memory relation, or temporal-PE
register-file realization proves it internal. Placing endpoints in the same
PE, FU, or `InstructionContextRef` never absorbs an edge by itself.

`fabric.fu` instances are compute resources. They may terminate routed edges
at FU inputs or originate routed edges at FU outputs, but they must not be used
as generic transit hops for unrelated traffic.

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
forms and use a zero-operand `fabric.yield` as their signature terminator.
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
* An overflow or `K * L > 64` in the PE boundary selector crosspoint count.
  A product in `(16, 64]` is valid and emits the advisory warning exactly once
  for that PE occurrence.
* Empty concrete FU occurrence set.
* A finalized anonymous-form body contains an op other than an anonymous
  `fabric.fu` occurrence (in particular, no `fabric.yield` may appear directly
  inside an anonymous-form `fabric.pe [spatial]`).
* A named-form body is missing its zero-operand `fabric.yield` terminator or
  the terminator carries an operand.
* `max_fu_inputs > K` or `max_fu_outputs > L`.
* An inner FU's outer port width does not match the PE's `W` (input
  or output). Inner block-arg widths narrower than the outer operand
  width are accepted; widening (`outer < inner`) is rejected by the
  FU's own verifier.
* A selector variant with a foreign or out-of-range PE port reference.
* A PE configuration field with a foreign owner, mismatched role or FU port,
  noncanonical value, or value outside its exact finite domain.
* Nesting violations: `fabric.pe [spatial]` placed outside a
  `fabric.module` body, nested inside another `fabric.pe [spatial]` /
  `fabric.pe [temporal]` / `fabric.fu`.

Diagnostics must identify the violated semantic condition without making
diagnostic wording part of the contract.

## Cross-References

* `spec-fabric-reconfigurable-op.md` for the parameterized operation
  capability and finalization rules that derive `fu_sw_configs`;
* `spec-configuration-deployment.md` for the physical codebook, inactive
  encoding, and payload-placement rules for these semantic fields;
* the share-group catalogue in `spec-fabric-hw-share-group.md` for
  legal multi-member `op_list`s inside inner FUs;
* `spec-fabric-instantiate.md` for the rules governing
  `fabric.instantiate` inside a PE body and named `fabric.fu`
  definitions.
