# Fabric FU

This document specifies `fabric.fu`, the CGRA-style functional-unit
container that wraps `fabric.op`, `fabric.mux`, and `fabric.demux` ops
into one PE-internal resource and presents a fixed external port
interface to the enclosing `fabric.pe`.

`fabric.mux` and `fabric.demux` are FU-local configurable connectivity
ops. They are part of a SpatialCore template inside `fabric.module`;
they are not `fabric.system` interconnect primitives and must not be
used to model system-level fan-in, fanout, arbitration, or routing.

## Capability And Realization Ownership

An FU owns physical resources, fixed boundary ports, SSA wiring, and a finite
normalized domain of structural/capability templates. Each inner `fabric.op`
owns one concrete parameterized capability through its explicit
`ImplementationFamilyId`, `op_list` projection, `hw_params`, physical ports,
and typed constraints.
Registered operation schemas own exact software semantics, while typed
Hardware Sharing Groups own only genuine physical implementation-family
sharing legality.

[Software Function To FU Synthesis](spec-generalize-subgraphs-to-fu.md) owns
the reverse construction contract from canonical software-function sets to
this same FU capability model. It does not create a second FU schema.

Hardware parameters and selected software configuration jointly determine the
supported configured software function. HSG membership and `op_list` syntax
alone never grant a concrete operation arbitrary capability; the complete
typed capability relation and exact software binding must accept it.

TechMapping selects an FU structural/capability template and binds exact
Canonical Dataflow actors to inner operations with ordered operand, result,
and FU-boundary correspondence. A FU-local mux, demux, or operation choice that
changes the configured software graph, selected physical operation, topology,
or boundary relation is part of that TechMapping realization. SpatialMapping
may choose only Fabric-declared physical or QoR refinements that preserve the
selected software function.

Canonical Fabric does not persist a workload's `sw_configs`. The finalizer
derives them from Fabric capability, the exact TechMapping realization, and
SpatialMapping's semantic-preserving refinements. Fabric owns configuration
field semantics and legal domains; only `ConfigurationABI` owns their physical
bit encoding and programming representation.

The configured projection is a closed sum:

```text
FuConfiguration =
    Disabled
  | Active { configured_graph, physical_refinements }
```

`Disabled` carries no mux, demux, operation, or refinement selections. The
physical inactive encoding belongs only to ConfigurationABI. `Active` denotes
one uniquely meaningful configured software graph; two configuration values
that materialize the same typed graph and physical behavior are invalid
duplicates, not alternative functions.

An FU is the physical configured-graph and capability boundary. It is not a
macro actor, an all-input rendezvous, or a dynamic atomicity boundary. Active
inner `fabric.op` resources execute the Canonical Dataflow actors' ordinary
transitions independently. An FU or `fabric.op` must not create an
operation-local context identifier. `InstructionContextRef` identifies only
the PE-owned resident configuration/runtime-state namespace selected by
SpatialMapping; it does not own the configured graph or its dynamic execution.

## Op shape

`fabric.fu` carries an optional `sym_name`. The op exists in two
disjoint syntactic forms by `sym_name` presence; the parser branches
on whether `@sym` appears right after the op keyword.
The schematic `example_multiply` family spelling below illustrates the
required typed attribute; exact builtin family IDs belong to the normative
HSG registry.

### Anonymous form (definition + use combined)

```mlir
%r = fabric.fu (%fa = %a : !fabric.bits<W> [to !fabric.bits<F>],
                %fb = %b : !fabric.bits<W>)
              -> !fabric.bits<W> {
  %v = fabric.op [@arith.muli] (%fa, %fb)
       {implementation_family =
          #fabric.implementation_family<example_multiply>}
       : (!fabric.bits<F>, !fabric.bits<W>) -> !fabric.bits<W>
  fabric.yield %v : !fabric.bits<W>
}
```

* Variadic SSA operands matched 1:1 with body block arguments via the
  inline `(%blockArg = %ssaSrc : <T_outer> [to <T_inner>], ...)`
  syntax.
* Variadic SSA results, with `fabric.yield` supplying the values that
  flow out of the FU.
* The optional `to <inner-type>` clause on each operand declares an
  inner block-argument width narrower than the outer SSA operand
  width; the high `(W - F)` bits are dropped at the FU boundary.
* Must live inside a `fabric.pe` body.

### Named template form (declaration only)

This snippet shows the PE-body fragment for a named FU template; a
complete Fabric module must nest it inside a `fabric.pe` body.

```mlir
fabric.fu @F (!fabric.bits<W>, !fabric.bits<W>) -> !fabric.bits<W> {
^bb0(%fa: !fabric.bits<W>, %fb: !fabric.bits<W>):
  %v = fabric.op [@arith.muli] (%fa, %fb)
       {implementation_family =
          #fabric.implementation_family<example_multiply>}
       : (!fabric.bits<W>, !fabric.bits<W>) -> !fabric.bits<W>
  fabric.yield %v : !fabric.bits<W>
}
```

* Zero SSA operands, zero SSA results in the enclosing `fabric.pe`
  body.
* Port signature captured in a `function_type : FunctionType`
  attribute.
* Body's entry block carries the input port types as block arguments;
  `fabric.yield` supplies the values matching `function_type` results.
* Implements `SymbolOpInterface` with `isOptionalSymbol() == true`,
  so the op participates in the enclosing `SymbolTable` and standard
  symbol-table lookup applies.
* Actual use goes through `fabric.instantiate @F(...)` (see
  `docs/spec-fabric-instantiate.md`).
* Must live inside a `fabric.pe` body.

## Body whitelist

* `fabric.op`, `fabric.mux`, `fabric.demux` are the only compute /
  routing ops permitted directly in the body.
* No nested `fabric.fu`. No `fabric.fifo`. No `fabric.pe` /
  `fabric.module`.
* The body must contain at least one `fabric.op`.
* The body terminator is `fabric.yield` (always, in both forms).

The mux / demux ops inside an FU let different structural/capability
templates realize different internal compute graphs over the same hardware:
they reshape in-FU operation connectivity rather than attaching to the FU's
external inputs or outputs. Allowing back-edges in the body lets configurable
compute resources such as a `LoopCarry` `fabric.op` and cyclic exact
software-function projections be matched to an FU through TechMapping.
The FU body region is a graph region
(`RegionKindInterface::Graph`).

## Explicit routing semantics

Multiple uses of one FU-local SSA value are a real token broadcast. Every
consumer participates in delivery and backpressure; configuration does not
implicitly drain inputs of an inactive `fabric.op`, masked-off variadic
`fabric.op` inputs, or non-selected inputs of a `fabric.mux`.

Mutually exclusive datapaths must route shared inputs through explicit
`fabric.demux` ops and collect their results through a matching `fabric.mux`.
For example, an FU that selects between separate add and multiply datapaths
needs one demux per shared input and one result mux. Directly connecting each
input to both operations describes broadcast to both datapaths, not selection.
Any configuration that would require an implicit drain is invalid.

`fabric.mux` and `fabric.demux` selections that determine the active internal
graph are correlated in the selected capability template and exact
TechMapping correspondence. They are not independent allowed-value sets.
Inactive ports may be omitted from token and routing obligations only when the
registered operation schema and concrete capability relation explicitly
guarantee no consumption, no production, and no backpressure.

## Edge Realization Boundary

A Canonical Dataflow edge is internal to an FU only when the exact configured
FU relation selected by TechMapping explicitly realizes that actor-to-actor
connection through the configured FU topology. Placing both actors in the same
FU, PE, or instruction context is not such a witness.

The only other relations that may make a software edge internal are an
explicit configured-memory relation and an explicit temporal-PE register-file
realization. Without one of these three typed relations, the edge remains an
external transfer obligation. Mere physical co-location, a selector, an
available local buffer, or an inactive branch never absorbs it.

## FU-boundary truncation (input side)

Anonymous form only: each operand may declare an inner block-argument
width narrower than its outer SSA operand width via the
`to <inner-type>` clause. Hardware drops the high `(W - F)` bits at
the FU boundary on each token; the inner block argument carries the
low `F` bits. Without the `to` clause, inner == outer.

## FU-boundary widening (output side)

Anonymous form only: each `fabric.yield` value may declare an outer
result width wider than the inner SSA value's width via the
`to <outer-type>` clause:

```mlir
fabric.yield %v : !fabric.bits<inner> to !fabric.bits<outer>
```

The clause is only valid when both types are `!fabric.bits<N>` and
`inner <= outer`. Hardware zero-fills the high `(outer - inner)` bits
at the FU boundary on each token; the low `inner` bits carry the
inner value. The declared outer type must equal the FU's declared
outer result type. Without the `to` clause, inner == outer (strict).

This output-side widening is the dual of the input-side
`to <inner-type>` truncation: input drops high bits at the boundary,
output zero-fills high bits at the boundary. Both keep the FU's outer
port types strict `!fabric.bits<W>`, so the enclosing PE's uniform-W
invariant is preserved. The named template form does not support
either relaxation: `fabric.yield` types must equal
`function_type.getResults()` exactly.

## Verifier checklist

Both forms:

* The body's last op is `fabric.yield`; its value count and per-value
  type match the FU's declared result list.
* The body contains at least one `fabric.op`. Other ops in the body
  must be `fabric.mux` or `fabric.demux`.
* Every external port type is `!fabric.bits<N>` for some `N`.

Anonymous form:

* The op must not carry a `function_type` attribute.
* Block-argument count equals operand count, and per-position the
  outer operand width is greater than or equal to the inner
  block-arg width.
* For each `fabric.yield` value `i`, when the per-value `to <type>`
  clause is present it must equal the FU's declared outer result
  type `i`, both inner and outer types must be `!fabric.bits<N>`,
  and inner-bits-width must be less than or equal to
  outer-bits-width (low-bit-aligned widening).
* Parent must be a `fabric.pe`.

Named template form:

* The op carries a `function_type : FunctionType` attribute and zero
  SSA operands / zero SSA results.
* Block-argument types equal `function_type.getInputs()`.
* Yield value types equal `function_type.getResults()`.
* Parent must be a `fabric.pe` body.

## Cross-references

* `spec-fabric-pe.md` -- PE container, schedule predicate, body
  whitelist, `K`/`L`/`W` rules.
* `spec-fabric-instantiate.md` -- symbol resolution, allowed
  parent/target table, width-relaxation rules at the
  `fabric.instantiate` site.
* `spec-fabric-reconfigurable-op.md` -- parameterized capability,
  TechMapping realization, and derived configuration for inner resources.
