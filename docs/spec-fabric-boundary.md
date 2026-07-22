# Fabric Boundary Op

This document specifies the unified `fabric.boundary` op, the canonical
conversion op between the spatial domain (`!fabric.bits<BW>`) and the
temporal, tagged domain (`!fabric.bits_tag<BW, TW>`). The op exists in
three direction-predicated variants (`s2t`, `t2t`, `t2s`) that share a
single op definition and a single verifier.

This op is a token-plane resource. It is distinct from a Dataflow graph
boundary, a `fabric.module` endpoint, a module-to-AccCore attachment, and a
system external boundary. In particular, it never converts or transports a
memory capability.

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
parser rejects any other keyword.

## Purpose

The fabric dialect distinguishes two stream domains:

* **Spatial domain.** A `!fabric.bits<BW>` channel carries a bit-vector
  data payload accompanied by an implicit valid/ready handshake. There
  is no notion of a "tag" in this domain.
* **Temporal domain.** A `!fabric.bits_tag<BW, TW>` channel additionally
  carries a `TW`-bit Physical Tag alongside the data. The tag is a
  Mapping-assigned local interpretation key for a Fabric match domain. It is
  not a source iteration, dynamic firing, invocation, logical-token identity,
  or globally unique number.

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

## Configured Projection

Every occurrence uses one closed configured projection:

```text
BoundaryConfiguration =
    Disabled
  | Active { direction_specific_fields, physical_refinements }
```

`Disabled` carries no tag, lookup entry, selector, or refinement. It performs
no transfer. `ConfigurationABI` alone owns its physical inactive encoding.
The Active payload is determined by the immutable op shape:

* two-operand `s2t` and either `t2s` form carry no semantic field;
* one-operand `s2t` carries exactly one configured `tag`; and
* `t2t` carries a nonempty sparse `lookup_table` bounded by `lut_size`.

An Active `t2t` projection with no lookup entry canonicalizes to `Disabled`.
Canonical hardware-only Fabric contains no selected projection. The
`sw_configs` spellings below are only the Active projection's semantic fields;
they are not optional values independent of the closed variant.

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

The tag operand is a Fabric token-plane source governed by the selected
Mapping continuity relation. This shape does not authorize Canonical Dataflow
to compute an arbitrary runtime integer and reinterpret it as a Physical Tag.
Any future runtime-computed-tag feature requires a new explicit semantic and
verification contract.

### Configurable-tag form (1 operand)

When hardware provides a configurable tag writer, the second operand is
elided. Canonical Fabric leaves the selected value absent. A configured
projection supplies it through `sw_configs.tag`:

```mlir
%out = fabric.boundary [s2t] %data {sw_configs = {tag = 10 : i4}}
       : !fabric.bits<BW> -> !fabric.bits_tag<BW, TW>
```

* Operand `#0` (`%data`): `!fabric.bits<BW>`.
* Result: `!fabric.bits_tag<BW, TW>`.
* In a configured projection, `sw_configs.tag` is required and is an
  `IntegerAttr` whose bit-width equals the result tag-width `TW`. The integer
  value is interpreted as an unsigned `TW`-bit pattern.
* In canonical hardware-only Fabric, `sw_configs` is absent; the op describes
  only a configurable tag-writer capability and is not executable until an
  exact Mapping is finalized.

The two-operand form must not carry `sw_configs.tag`; the tag is
already supplied by SSA.

`sw_configs.tag` is a configured projection selected by SpatialMapping. It is
not a Fabric hardware fact or a second Physical Tag owner. The actual tagged
writer assignment remains a typed `ResourceUse` sharing assignment at the
real writer or ingress. Repeated firings on the same continuity segment reuse
that assignment.

### S2T Verifier Rules

* Operand count is 1 or 2; result count is 1.
* Result type is `!fabric.bits_tag<BW, TW>`.
* Operand `#0` type is `!fabric.bits<BW>` (data-width must match the
  result data-width exactly).
* Two-operand form: operand `#1` type is `!fabric.bits<TW>` (tag-width
  must match the result tag-width exactly). `sw_configs.tag` must be
  absent.
* One-operand form: canonical hardware-only Fabric has no configured
  projection. An Active projection contains exactly one `tag` field whose
  `IntegerAttr` type width equals the result tag-width.
* A present `sw_configs.tag` must be a non-negative integer literal.
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

The Active projection uses a sparse representation: only selected LUT entries
are materialized in `lookup_table`. The hardware-side LUT capacity is
declared via `hw_params[0].lut_size`. Canonical hardware-only Fabric omits
`sw_configs`; the configured selection is later projected as
`sw_configs.lookup_table`.

`lut_size` is bounded hardware capacity and is independent of `2^TW1`.
The lookup is exact content match over the Active entries, not direct indexing
by tag. The configured entries are a deterministic projection of selected
Mapping tag continuity and retagging; they do not own logical stream identity.

### T2T Verifier Rules

* Operand count is 1; result count is 1.
* Operand and result are both `!fabric.bits_tag<...>`.
* `BW` (operand data-width) equals the result data-width. The op only
  remaps the tag; it never changes the data field.
* `TW1` (operand tag-width) and `TW2` (result tag-width) may differ.
* `hw_params` is required: a length-1 array wrapping a dictionary
  with key `lut_size` (`IntegerAttr`, value >= 1).
* Canonical hardware-only Fabric has no configured projection. An Active
  projection contains exactly one nonempty `lookup_table` key (`ArrayAttr`).
* `lookup_table.size() <= lut_size`; an empty selection is canonical
  `Disabled`, not an Active empty table.
* Each entry is a dictionary with keys `src_tag` (`IntegerAttr`,
  width == `TW1`, non-negative literal) and `dst_tag` (`IntegerAttr`,
  width == `TW2`, non-negative literal). The negative-literal rule
  has the same signless caveat as `[s2t]`.
* All `src_tag` values across the LUT entries are distinct.

### Physical Encoding Ownership

The former packed LUT-word layout in this specification is retired.
`ConfigurationABI` is the sole owner of physical bit and address layout,
destination slices, inactive values, and the programming and activation
contract. Fabric owns the bounded LUT capability and semantic configuration
field domains; Mapping owns the selected retag relation. The configured
`lookup_table` is their deterministic semantic projection, which an exact
`ConfigurationABI` encodes for its `ProgrammingUnit`. This specification does
not define slot order, packed width, reserved bits, or another physical
encoding authority.

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
* The Active projection has no direction-specific semantic field.

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

## Mapping And Implementation Ownership

Fabric owns the boundary direction, port and table capacities, legal
transform, and semantic configuration fields and domains. SpatialMapping owns
whether the resource is selected, the real writer or ingress Physical Tags,
the configured constant or lookup relation, and event-relative use. The
`sw_configs` forms in this document are configured projections of those
choices. `ConfigurationABI` alone maps those semantic fields to physical bits,
addresses, and programming operations.

A concrete SpatialCore implementation owns comparator and selector circuitry
that implements the exact Fabric contract. This boundary has no hidden
buffering, arbitration, or latency. Any refinement that changes
cycle-observable state or timing must be declared as a Fabric capability or a
Mapping-selected exact hardware refinement and therefore participate in the
exact Fabric/Mapping identity. No runtime, simulator, or backend may infer a
different tag relation, add capacity, or invent cycle-visible behavior.

## Graph And Module Boundary Symmetry

Canonical graphs classify both inputs and outputs as `value`, `stream`, or
`memory`. Value and stream use the token plane and may be realized through
`bits`, `bits_tag`, switches, FIFOs, and this boundary resource. Memory remains
a separate capability plane and never passes through `fabric.boundary`.

Graph memory import means that software uses an external memory capability;
graph memory export means that software provides one. The hardware analogue is
a `fabric.module` manager/requester input and subordinate/provider output.
These directions describe capability crossing, not load/store direction or
ownership transfer. Mapping may bind one graph memory to several endpoints or
one endpoint to several graph memories through explicit records; no positional
or type-conversion rule in this op establishes that relation.

## Validation Anchors

Anchor-level validation covers a Mapping-assigned constant `s2t` tag reused by
repeated firings, bounded `t2t` content matching without direct tag indexing,
`t2s` tag removal, and rejection of any physical LUT layout outside the exact
`ConfigurationABI`. Tests do not freeze diagnostic wording, comparator
topology, row placement, or raw configuration bits.

## Cross-references

* `spec-fabric-module.md` -- the canonical container that hosts
  `fabric.boundary` (and the body whitelist that admits it).
* `spec-fabric-fu.md` and `spec-fabric-pe.md` -- describe the
  `fabric.fu` and `fabric.pe` containers that, by their own body
  whitelists, exclude `fabric.boundary`.
* `spec-fabric-instantiate.md` -- template instantiation rules for
  Fabric resources whose external connections remain governed by the
  module-level compatibility rule.
* `spec-mapping-memory.md` -- software-memory, physical-service, and
  manager/subordinate endpoint binding ownership.
