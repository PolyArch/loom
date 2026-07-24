# ADG Builder

This document defines the public C++ authoring model, finalization boundary,
and builtin-target construction contract of Loom's ADG Builder library.

## Role

The ADG Builder is a normal C++ library for hardware architects. A user links
it into an ordinary C++ program to construct a multi-core heterogeneous
spatial-accelerator design.

Its canonical semantic product is a finalized Fabric Hardware Description:

```text
C++ ADG Builder program
  -> Fabric construction
  -> Fabric verification and finalization
  -> exact content-addressed Fabric Hardware Description objects
```

Builder objects, helper names, insertion order, generated labels, and C++
ownership state are authoring details. Downstream compilation, Mapping,
simulation, visualization, RTL generation, and Evaluation consume the exact
Fabric objects and never inspect Builder state. An architecture-only
`fabric.system` object is independently valid. Any Interconnect Implementation
is a separate Fabric-family object that references the exact Transport
Architecture it refines, even when both objects are serialized in one MLIR
file.

The earlier `loom` and `loom-devel` terminology was only a distribution
analogy. This contract requires public headers and a linkable library; it does
not require an RPM split or any other packaging mechanism.

## Fabric Is The Hardware SSOT

The Builder never defines hardware semantics outside the Fabric dialect. Every
helper must elaborate into verifier-visible Fabric facts before finalization.
If a downstream component needs a topology, capability, timing, capacity,
configuration, memory, coherence, or implementation fact, that fact must be
owned by Fabric or by another explicit artifact family with that semantic
responsibility.

Builder-only metadata may improve diagnostics or authoring, but cannot be
required by a downstream consumer. Optional visualization metadata is removed
from the Fabric semantic identity preimage and may be retained only in a
projection that references the exact finalized Fabric identity.

## Construction Coverage

The complete public construction surface must cover every verifier-legal
target Fabric description without requiring users to write raw MLIR. This
includes both hardware levels:

* a SpatialCore or CGRA represented by `fabric.module`; and
* a heterogeneous AccCore SoC represented by `fabric.system`.

SpatialCore construction covers PE, FU, switch, memory, FIFO, boundary,
instantiation, ports, capabilities, configuration domains, and explicit
Graph-region connectivity as owned by the corresponding Fabric specifications.
An FU remains PE-local; the Builder cannot promote it into a parallel
module-level resource kind.

System-family construction covers an architecture-only `fabric.system` object
with heterogeneous AccCore occurrences, InstructionCore descriptions,
SpatialCore attachments, memory and service capabilities, Transport
Architecture, external boundaries, and hardware domains. It may separately
construct Interconnect Implementation refinement objects as owned by
`spec-fabric-system-adg.md`.

The Builder must not replace these typed concepts with generic node, port,
channel, link, property-bag, or string-kind records. A typed C++ convenience
structure is valid only when its elaboration has one unambiguous Fabric
meaning.

## Topology Independence

Regular and irregular designs are equally fundamental.

Regular helpers may construct chains, arrays, systolic structures, mesh-like
graphs, torus-like graphs, or repeated heterogeneous clusters. Irregular
construction must support arbitrary directed topology, sparse long links,
trees, heterogeneous islands, mixed spatial and temporal resources, and
nonuniform memory or transport structures.

Coordinates and layout hints are optional authoring or visualization data.
They never define connectivity, placement distance, routing legality, or cost.
Inside `fabric.module`, SSA connections own topology. Inside `fabric.system`,
typed transport resources, endpoints, patterns, and directed connections own
topology.

## Exact And Convenient Authoring

One public library may expose both exact typed construction and convenience
templates, but they are not separate semantic layers:

```text
convenience template or helper
  -> exact typed Builder operations
  -> the same Fabric finalization path
```

A user may combine helper-created and exact objects in one design. The emitted
Fabric contains no marker that distinguishes how an object was authored.

The exact public API uses typed enums, references, and compact owner-specific
specification values rather than strings or arbitrary parameter maps. Strings
remain suitable for nonsemantic labels.

## Typed Construction Surface

The public construction surface is a thin typed facade over actual Fabric IR.
It does not build a parallel C++ hardware graph and does not render Fabric as
text for reparsing. A successful exact construction call immediately creates
the corresponding typed Fabric operation, type, attribute, value, or
reference in the Builder-owned draft IR.

The public surface has one design owner and three scoped construction views:

```text
DesignBuilder
  SpatialCoreBuilder
  SystemBuilder
  InterconnectImplementationBuilder
```

`DesignBuilder` owns the draft IR, construction diagnostics, and root closure.
The scoped views create `fabric.module`, `fabric.system`, and optional
Interconnect Implementation roots. They are authoring views over that one
draft, not independent schemas or persistent objects.

Handles reflect essential Fabric distinctions. Spatial graph values and
ports, AccCores, memory services, transport resources, endpoints, hardware
domains, and implementation refinements use typed references that cannot be
silently exchanged across owners or roles. There is no generic `NodeRef`,
string `kind`, property bag, textual type, textual operation name, or
user-managed SSA name as a semantic input.

Construction calls return `llvm::Expected<Handle>` or `llvm::Error`.
Fluent calls that accumulate a hidden invalid state are not part of the public
contract. Builder-local checks cover stale or foreign handles, incomplete
helper expansion, duplicate nonsemantic labels where diagnostics would become
ambiguous, and other authoring failures. Fabric verification remains the only
semantic hardware authority.

Convenience topology and resource helpers use only this exact public surface.
They may return typed groups of created handles, but cannot create hidden
hardware facts or use a private emitter.

## Failure-Atomic Finalization

Finalization consumes the draft design. Its conceptual public boundary is:

```text
llvm::Expected<FinalizedFabricDesign> DesignBuilder::finalize() &&
```

`FinalizedFabricDesign` is a transient immutable C++ closure over finalized
Fabric roots and exact dependency references. It is not a new Artifact family,
hardware schema, or source of semantic identity.

The root variants, direct dependency framing, canonicalization, publication,
and failure classification are owned by `docs/spec-fabric-artifact.md`.
Builder finalization calls that owner; it does not serialize or hash a parallel
hardware model.

Finalization performs one all-or-none derivation:

1. close construction scopes and expand all helpers;
2. resolve every typed reference;
3. run authoring-boundary checks;
4. invoke the canonical Fabric verifier and finalizer for every root;
5. derive canonical bytes, identities, and the complete dependency closure;
6. expose the finalized closure only after every member succeeds.

Artifact-store publication may write immutable content blobs before the root,
but it publishes the root reference only after the complete closure is
available. A failed attempt may leave only unreachable cache content, never a
partially valid hardware target. Stream formatting and filesystem paths are
output bindings and cannot weaken this finalization contract.

## Builtin Targets

Loom builtin hardware targets are versioned ADG templates implemented through
the same Builder/Fabric path. Selecting a builtin preset expands its exact
template identity, version, and parameters, then finalizes an ordinary Fabric
Artifact. A preset name is authoring provenance, not a substitute for the
expanded hardware facts.

An external Fabric file and a builtin template are distinct source forms, but
all downstream stages see the same finalized Fabric contract. If hardware
selection is omitted, configuration resolution selects the designated builtin
default before compilation begins; downstream stages never observe a missing
hardware target.

Every builtin descriptor contains one stable template identity, one `X.Y`
schema version, one closed typed parameter schema, and one expansion function.
The expansion function uses only the public typed ADG Builder API. Compiler
selection has no private emitter, prebuilt opaque Fabric body, textual shortcut,
or privileged validation path.

A published builtin also requires complete provider closure for every
ImplementationFamilyId, memory form, transport resource, and external binding
it advertises. Missing closure fails builtin publication. A valid user-authored
custom Fabric may still finalize without a selected backend provider; a later
backend request reports typed `Unsupported` without changing that Fabric.

The initial general-purpose family has three closed authoring presets:

```text
BuiltinTargetPreset = Small | Default | Large
```

The enum is accepted only at authoring boundaries. Resolution replaces it
with the exact template identity, version, and fully expanded typed parameters.
The preset spelling is provenance; it is not a hardware fact or a substitute
for the finalized Fabric identity.

All three presets share one general-purpose capability set. They differ in
resource multiplicity, resident capacity, buffering, and topology scale, not
in workload-specific operations. In particular, the catalog does not define
an attention target, sparse target, DSP target, or other application-specific
hardware profile.

Each fully resolved preset owns a complete construction recipe for:

* the AccCore inventory and explicit system Transport Architecture;
* every Spatial and Temporal PE occurrence, PE port shape, resident capacity,
  operand-buffer organization, register FIFO, and inner FU occurrence;
* every FU's exact Fabric operation capabilities, hardware-sharing groups,
  hardware parameters, and software-configuration domains;
* every Spatial and Temporal switch occurrence, physical connectivity table,
  temporal route-table capacity, and explicit directed connection;
* every Spatial and Temporal memory occurrence, operation-port inventory,
  element and vector alternatives, service endpoints, local service, resident
  capacity, and dispatch capability; and
* every required Spatial-to-Temporal or Temporal-to-Spatial boundary.

The common construction pattern contains both an untagged Spatial network and
a tagged Temporal network connected only through explicit Fabric boundaries.
This is an authoring recipe, not a second topology schema: expansion produces
ordinary explicit Fabric resources and connections.

The initial scale anchors are:

| property                         | `Small` | `Default` | `Large` |
| -------------------------------- | ------: | --------: | ------: |
| AccCore occurrences              |       4 |         8 |      16 |
| PE occurrences per SpatialCore   |      16 |        36 |      64 |
| Spatial : Temporal PE ratio      |    12:4 |      27:9 |   48:16 |
| memory occurrences per core      |       2 |         4 |       8 |
| Spatial : Temporal memory ratio  |     1:1 |       2:2 |     4:4 |
| Temporal resident-context anchor |       2 |         4 |       8 |
| cross-schedule gateway anchor    |       2 |         4 |       8 |

These values are resolved inputs to one template, not fields persisted in
Fabric in addition to the resources they generate. Exact per-helper resource
inventory, hardware parameters, switch construction, memory-port capacity, and
buffer capacities are part of the same versioned family contract and must be
fixed before that catalog version is implementation-complete. The
`CoreAluFu` and `MacFu` scalar portions and the complete initial
`LoopControlFu` contract are fixed below; the other helpers, switch
construction, and remaining capacities are not implied by those definitions.
Helper resource tables reference normative
`ImplementationFamilyId` values; operation-family membership remains owned by
the HSG registry. They do not duplicate member lists, spell operation names as
dispatch keys, or define backend modes.

### General-Purpose FU Library

The initial family uses a small typed FU-construction library. The names below
identify public Builder helpers and reviewable recipe fragments; they do not
become Fabric resource kinds, persisted classifications, or a second
capability registry. Expansion produces only ordinary `fabric.fu`,
`fabric.op`, `fabric.mux`, and `fabric.demux` resources.

| Builder helper       | Constructed capability |
| -------------------- | ---------------------- |
| `CoreAluFu`          | Scalar integer and floating-point arithmetic, logic, shifts, comparisons, min/max, selection, and casts. |
| `MacFu`              | Integer and floating-point multiply, floating-point fused multiply-add, and explicit multiply-add or accumulate configured graphs. |
| `VectorComputeFu`    | Fixed-ranked elementwise arithmetic, comparison, selection, and multiply-add capabilities. |
| `VectorAdapterFu`    | `dataflow.pack`, `dataflow.unpack`, `dataflow.parallelize`, and `dataflow.serialize` capabilities. |
| `LoopControlFu`      | `dataflow.stream`, `dataflow.carry`, `dataflow.invariant`, and `dataflow.gate` capabilities. |
| `TokenControlFu`     | `dataflow.constant`, `dataflow.sync`, `dataflow.mux`, and `dataflow.demux` capabilities. |
| `SpecialMathFu`      | Low-density divide/remainder, square-root, exponential, logarithmic, trigonometric, and rounding capabilities. |

This table is a construction decomposition, not an HSG table. Every concrete
`fabric.op` still binds exactly one typed Hardware Sharing Group implementation
family through its explicit `ImplementationFamilyId` attribute. Distinct
integer, floating-point, multiply, and special-function datapaths remain
distinct physical operations unless the normative HSG registry and
Fabric-to-RTL provider keyed by that same family ID prove genuine circuit
sharing. A configured FU that selects among separate datapaths uses explicit
coherent input `fabric.demux` and output `fabric.mux` topology.

`MacFu` is an FU graph, not a synthetic MAC HSG. `VectorAdapterFu` similarly
does not imply that its four software operation families share one circuit.
`dataflow.pack` and `dataflow.unpack` may share an implementation family only
when the typed HSG registry and backend realize one genuine reinterpretation
datapath. The stateful `dataflow.parallelize` and `dataflow.serialize`
capabilities remain distinct unless one backend-supported stateful
implementation family proves otherwise.

Memory actors, including load, store, atomic, compare-exchange, and fence, are
implemented by `fabric.mem` and never enter this FU library.

#### CoreAluFu Resource Inventory

`CoreAluFu` constructs one concrete `fabric.op` resource for each of the
following implementation families:

```text
ScalarIntegerAddSub
ScalarIntegerLogic
ScalarIntegerShift
ScalarIntegerCompareMinMax
ScalarValueSelect
ScalarIntegerCast
ScalarBitReinterpret
ScalarFloatSign
ScalarFloatAddSub
ScalarFloatCompareMinMax
ScalarFloatWidthCast
ScalarIntegerToFloat
ScalarFloatToInteger
```

The exact operation members are owned by
`docs/spec-fabric-hw-share-group.md`. The FU uses explicit coherent
`fabric.demux` and `fabric.mux` topology to select among physically separate
resources. It does not create a `CoreAlu` implementation family.

Multiply and FMA resources belong to `MacFu`. Integer or floating divide and
remainder, roots, rounding, and transcendental functions belong to
`SpecialMathFu`. The initial catalog lowers integer absolute value to ordinary
compare, select, and subtract actors rather than advertising an unproven
family. Target-specific packed intrinsics must first become target-neutral
scalar or vector actors.

#### MacFu Resource Inventory

`MacFu` constructs one concrete resource for each of:

```text
ScalarIntegerMultiply
ScalarFloatMultiply
ScalarFloatFma
ScalarIntegerAddSub
ScalarFloatAddSub
```

It also contains one canonical `dataflow.carry` operation resource so an
explicit accumulator graph can remain FU-local. That resource binds the
normative `LoopCarry` implementation family and references its exact Fabric
resource contract rather than duplicating either authority.

The finite normalized FU template domain includes:

```text
integer multiply
floating-point multiply
true fused floating-point FMA
integer multiply -> add/sub
non-fused floating-point multiply -> add/sub
multiply or FMA -> add/carry recurrence
```

These templates select explicit resources, internal edges, and coherent FU
boundary correspondence. They do not define a synthetic MAC operation or HSG.
An LLVM `fmuladd` spelling is never sufficient to select the fused template;
canonical actor semantics determine whether the input is `math.fma` or an
explicit multiply-add graph.

#### LoopControlFu Resource Inventory

`LoopControlFu` constructs concrete resources from these implementation
families:

```text
LoopStream
LoopCarry
LoopInvariant
LoopGate
```

The exact family members and physical resource contracts are owned by
`docs/spec-fabric-hw-share-group.md` and
`docs/spec-fabric-reconfigurable-op.md`. The helper does not create a
`LoopControl` implementation family. Stream resources with different fixed
step kinds are distinct `LoopStream` occurrences rather than configured modes
of one occurrence.

#### LoopControlFu Concrete Contract

Each initial `LoopControlFu` contains exactly two `LoopStream` resources and
one resource from each of `LoopCarry`, `LoopInvariant`, and `LoopGate`. The two
stream resources have distinct fixed step kinds selected by the builtin
distribution below. Every stream resource supports integer widths
`{8, 16, 32, 64}` and every registered schema-valid integer continuation
predicate.

The fixed helper boundary is:

```text
outer:
  bits<128>, bits<128>, bits<128>, bits<128>
    -> bits<128>, bits<128>, bits<128>

inner roles:
  d0: bits<128>
  d1: bits<128>
  d2: bits<128>
  c0: bits<1>
    -> r0: bits<128>
       r1: bits<128>
       p0: bits<1>
```

Builtin expansion uses the ordinary anonymous-FU boundary rule to truncate
the low bit of the fourth outer input into `c0` and zero-extend `p0` into the
third outer result. The data roles retain the full 128-bit payload floor.
`LoopStream` actors use only their selected exact low 8, 16, 32, or 64 bits;
the transparent payload resources may carry an exact supported scalar or
fixed-ranked vector up to 128 bits. The role names above are stable catalog
ordinals, not software operand names or additional Fabric attributes.

The active structural template is exactly one member of this closed set:

| Template | FU inputs | FU outputs | Internal relation |
| -------- | --------- | ---------- | ----------------- |
| `stream(S)` | `d0`, `d1`, `d2` | `r0`, `p0` | One concrete `LoopStream` with fixed step `S` |
| `carry` | `c0`, `d0`, `d1` | `r0` | One `LoopCarry` |
| `invariant` | `c0`, `d0` | `r0` | One `LoopInvariant` |
| `gate` | `c0`, `d0` | `r0`, `p0` | One `LoopGate` |
| `carry -> gate` | `c0`, `d0`, `d1` | `r0`, `r1`, `p0` | Raw carry output on `r0`; the same value is gated onto `r1` |
| `invariant -> gate` | `c0`, `d0` | `r0`, `r1`, `p0` | Raw invariant output on `r0`; the same value is gated onto `r1` |

For the two fused templates, the selected `c0` token is a real software-graph
broadcast to both stateful actors. The carry or invariant output is likewise
broadcast to the raw FU result and the gate value input. Preserving the raw
parent-domain value is necessary for an external exit or frontier projection;
the projected value and phase belong to the child domain.

All mutually exclusive input routes, operation inputs, result roles, and
output routes use explicit coherent `fabric.demux` and `fabric.mux` topology.
Direct SSA multi-use occurs only for the two broadcasts above. An operation
result that changes between a raw output and an internal gate input uses an
explicit demux before those mutually exclusive routes.

`FuConfiguration` remains `Disabled` or one active template. The valid
configuration domain is the normalized sum of the table rows and the exact
semantic parameter point admitted by the selected operation resources. It is
not the Cartesian product of local mux, demux, operation, width, and predicate
fields. Selector values that do not form one coherent row are invalid, and
irrelevant fields are absent or canonicalized. No initial template activates
`stream` together with another loop-control actor or activates disconnected
actors merely because the resources share one FU.

#### CoreAluFu And MacFu Concrete Scalar Contract

The initial scalar helpers use exact family-specific typed capability records.
They do not use arbitrary-width integers, a generic semantic parameter bag, or
one independently editable backend mode table.

| Implementation family | Concrete builtin parameter domain |
| --------------------- | --------------------------------- |
| `ScalarIntegerAddSub` | integer widths `{8, 16, 32, 64}` |
| `ScalarIntegerLogic` | integer widths `{1, 8, 16, 32, 64}` |
| `ScalarIntegerShift` | integer widths `{8, 16, 32, 64}` |
| `ScalarIntegerCompareMinMax` | operand widths `{8, 16, 32, 64}` and all registered schema-valid integer comparison predicates |
| `ScalarValueSelect` | `i1` condition and value types `i1/i8/i16/i32/i64/f16/bf16/f32/f64` |
| `ScalarIntegerCast` | source and destination widths `{1, 8, 16, 32, 64}` under the registered schema-valid pair relation |
| `ScalarBitReinterpret` | equal-total-width pairs among the enabled scalar types |
| `ScalarFloatSign` | formats `{f16, bf16, f32, f64}` and the strict floating behavior profile |
| `ScalarFloatAddSub` | formats `{f16, bf16, f32, f64}` and the strict floating behavior profile |
| `ScalarFloatCompareMinMax` | formats `{f16, bf16, f32, f64}`, all registered schema-valid floating comparison predicates, and the strict floating behavior profile |
| `ScalarFloatWidthCast` | source and destination formats `{f16, bf16, f32, f64}` under the registered schema-valid widening or narrowing relation |
| `ScalarIntegerToFloat` | integer widths `{8, 16, 32, 64}`, formats `{f16, bf16, f32, f64}`, and the strict floating behavior profile |
| `ScalarFloatToInteger` | formats `{f16, bf16, f32, f64}`, integer widths `{8, 16, 32, 64}`, and the strict floating behavior profile |
| `ScalarIntegerMultiply` | integer widths `{8, 16, 32, 64}` |
| `ScalarFloatMultiply` | formats `{f16, bf16, f32, f64}` and the strict floating behavior profile |
| `ScalarFloatFma` | formats `{f16, bf16, f32, f64}` and exact single-rounding strict fused semantics |

Resolved index casts use the compiler target's exact 32-bit or 64-bit index
width before capability matching. Boolean `i1` is admitted only by families
whose ordinary semantics require boolean data or conversion; it does not
broaden arithmetic, shift, multiply, or integer/floating conversion into an
arbitrary `iN` capability.

The strict floating behavior profile follows the registered operation
semantics. Ordinary arithmetic uses round-to-nearest-ties-to-even, preserves
subnormals and signed zero, and performs no implicit flush-to-zero or
fast-math transformation. Conversions follow their exact registered rounding
and exceptional-value semantics rather than inheriting an arithmetic rounding
rule. A strict implementation may realize a relaxed actor only through a proof
supplied by the registered operation schema. Operation identity continues to
distinguish the different floating min/max NaN contracts.

Each concrete operation uses 64-bit untagged scalar data ports. Conditions are
one-bit ports. A comparison result occupies the low bit of a 64-bit physical
result and zero-fills the remaining bits. The helper boundary shapes are:

```text
CoreAluFu outer:  bits<128>, bits<128>, bits<128> -> bits<128>
CoreAluFu inner:  bits<64>,  bits<64>,  bits<1>   -> bits<64>

MacFu outer:      bits<128>, bits<128>, bits<128>, bits<128> -> bits<128>
MacFu inner:      bits<64>,  bits<64>,  bits<64>,  bits<1>   -> bits<64>
```

The third Core input is the condition source for value selection. The fourth
Mac input is the phase source for recurrence templates. Exact TechMapping
correspondence may map software operand ordinals to these physical roles; the
listed order is the stable helper role order, not a change to software
operation syntax. Unused helper ports have no token, production, or
backpressure obligation in a configured template.

Builtin expansion constructs anonymous concrete FUs so the existing FU input
truncation and output widening rules explicitly connect the 128-bit PE
transport boundary to the 64-bit data and one-bit condition network. Internal
data `fabric.mux` and `fabric.demux` resources use 64-bit ports. No implicit
adapter, hidden drain, or 128-bit arithmetic datapath is inferred.

Every stateless scalar compute resource in these two helpers has one
registered elastic result stage; that stage is its sole result holding slot.
It has latency one and initiation interval one under downstream progress,
retains a stalled result, and supports same-cycle result consumption and
replacement acceptance. The multi-operation template timing is derived from
its selected resources and explicit topology.

The imported recurrence resource uses the exact `LoopCarry` family contract.
It has initial/running mode state but stores no carried payload, and its
elastic-transparent forwarding adds no registered cycle. Consequently, a
registered add followed by canonical carry retains a one-cycle recurrence
path and initiation interval one under downstream progress. The carry
operation schema remains the sole owner of its logical transition semantics.

The Mac resource is physically local to `MacFu`; it is not a reference to a
`LoopControlFu` occurrence. In a recurrence template, the carry output is a
real broadcast to the FU result and the selected arithmetic recurrence input,
the registered arithmetic result supplies `carry.next`, the stable Mac phase
role supplies `carry.cond`, and the exact template supplies the initial value.
The template reuses the normative family, resource, transition, and timing
contracts without copying their configuration or state machine.

### Payload And Type Floor

Every preset in the initial family uses a 128-bit ordinary PE and
intra-SpatialCore data-transport payload capacity. Narrower scalar values
occupy the low payload bits under the Fabric width rules. Physical Tags remain
a separate field and never contribute payload capacity.

The common scalar type floor covers the exact integer and resolved-index
domains above through 64 bits, plus floating-point element types `f16`,
`bf16`, `f32`, and `f64`, subject to each registered operation schema. The
fixed-vector floor covers
row-major-flattened payloads no wider than 128 bits, including the maximal
lane modes:

```text
16xi8
8xi16
4xi32
2xi64
8xf16
8xbf16
4xf32
2xf64
```

Smaller lane counts and any fixed rank fitting the same typed lane capacities
are represented by their exact standard MLIR vector types. The actor type
mechanically determines the active lane count; no independent vector-size
attribute or operation-name suffix exists. Shape-sensitive operations such as
reductions or shuffles require an explicit typed capability and cannot be
inferred from flattened width. Scalable vectors are outside this initial
family.

Equal physical width never proves semantic compatibility. For example,
`vector<4xf32>` and `vector<2xf64>` are separate capability points even though
both occupy 128 payload bits.

### Deterministic FU Distribution

FU occurrence density is applied independently to the Spatial and Temporal PE
sets of every SpatialCore. For one schedule kind with `n` PEs, the recipe
constructs:

| FU helper          | occurrence count            | ordinal offset |
| ------------------ | ---------------------------: | -------------: |
| `CoreAluFu`        |                         `n`   | all sites      |
| `MacFu`            |               `ceil(n / 2)`  |              0 |
| `VectorComputeFu`  |               `ceil(n / 4)`  |              1 |
| `LoopControlFu`    |               `ceil(n / 4)`  |              2 |
| `TokenControlFu`   |               `ceil(n / 4)`  |              3 |
| `VectorAdapterFu`  |       `max(1, ceil(n / 8))`  |              4 |
| `SpecialMathFu`    |      `max(1, ceil(n / 16))`  |              7 |

For a non-core family with count `k`, occurrence `j` selects the schedule-local
canonical site ordinal:

```text
(floor(j * n / k) + family_offset) mod n
```

for `0 <= j < k`. Selected ordinals are then sorted before construction. The
rule spreads each family without using XY coordinates, topology distance,
randomness, or authoring insertion order. Several FU families may occur in
one PE. A Spatial PE still activates at most one FU configuration, while a
Temporal PE uses its Fabric-declared resident instruction contexts.

After placement in both schedule kinds, all `LoopControlFu` occurrences in
one SpatialCore are ordered first by the closed schedule-kind order
`Spatial, Temporal`, then by canonical site ordinal, then by same-site helper
ordinal. Occurrence `q` receives the step-kind pair at
`q mod 4` from:

```text
0: add, sub
1: mul, sdiv
2: udiv, shl
3: ashr, lshr
```

This assignment adds no profile enum or second capability table; it supplies
the two fixed `LoopStream` parameters of each explicit helper expansion. Every
preset therefore exposes all eight stream step kinds in each SpatialCore.
The Small preset has exactly four total `LoopControlFu` occurrences and covers
the list once. Default and Large repeat the same four-pair catalog and increase
multiplicity without changing the software capability set.

Applying the occurrence-density rule separately guarantees that every helper
kind is represented in both schedule kinds. Exact operation-resource
multiplicity may intentionally differ between those kinds; the common
software capability floor belongs to the complete SpatialCore rather than
being duplicated in each schedule kind. For example, the Small preset has
`12/6/3/3/3/2/1` occurrences across its 12 Spatial PEs and
`4/2/1/1/1/1/1` occurrences across its four Temporal PEs in the table's FU
order.

A builtin descriptor may claim this catalog version only when every listed
software operation has a registered typed operation schema, a legal concrete
Fabric capability, and provider closure for every referenced implementation
family. Operation names, backend-local classifications, or equal port widths
cannot substitute for any of these requirements.

## Builtins As Public Examples

The canonical preset expansion functions are reference-quality C++ examples
of the public ADG Builder API. They are compiled as production builtin
generators and remain readable as examples for hardware architects.

A focused example executable may select a preset, populate a `DesignBuilder`,
finalize it, and print or publish the resulting Fabric. It must call the same
descriptor and expansion function as `loom-cc` and `loom-c++`; it must not
copy the recipe. Users may instead invoke the parameterized expansion on an
unfinalized Builder and extend the result through the same typed API, in which
case the modified output is a custom Fabric target rather than the named
preset.

Therefore one implementation serves three uses without duplication:

```text
compiler builtin target
public ADG Builder reference example
starting point for a user-authored custom target
```

## Determinism

For one Builder semantic version and equal semantic inputs, elaboration must
produce the same finalized Fabric identity.

Fabric finalization owns canonical entity identity, ordering, serialization,
and semantic hashing. Builder construction order and generated textual names
cannot become tie breakers. Name collisions or unresolved references are
diagnosed before finalization.

## Validation

Validation has two complementary boundaries:

* Builder validation reports invalid references, incomplete helper expansion,
  and authoring errors using user-facing construction context.
* Fabric validation is the final semantic authority and accepts or rejects the
  emitted `fabric.module` and `fabric.system` roots.

Builder validation must not reproduce Fabric verifier rules as a second
authority. It may invoke or adapt those rules to produce better diagnostics.
No Builder path may bypass Fabric verification.

## Downstream Orchestration

The library may expose convenient entry points that invoke downstream Loom
components after Fabric finalization, for example RTL generation or an
Evaluation model. Those entry points are orchestration only:

* Fabric-to-RTL owns hardware lowering and `HardwareImplementation`;
* Evaluation owns frequency, power, area, timing, and other evidence;
* Mapping owns software-to-hardware realization; and
* visualization owns removable projections.

The Builder does not gain ownership of those outputs merely because a C++
helper invokes their producer.

Hardware DSE invokes the same Builder through typed `FabricTemplateConfig` or
applies typed `FabricRewriteConfig` to an exact Fabric artifact. The central
DSE plan owns orchestration and lineage; the Builder owns deterministic
elaboration only. There is no generic hardware action language, mutable
candidate graph, or DSE-only construction path.

## Conformance Anchors

The stable Builder anchors are deliberately small:

1. A regular SpatialCore program emits and verifies one exact
   `fabric.module` with explicit connectivity.
2. An irregular SpatialCore program emits and verifies a topologically
   non-grid `fabric.module` without semantic coordinates.
3. A heterogeneous system program emits and verifies one `fabric.system` with
   multiple distinct AccCores and explicit transport and memory services.
4. A builtin target and its direct Builder expansion finalize to the same
   Fabric identity.
5. Invalid authoring cannot produce a finalized Fabric Hardware Description
   object.
6. The compiler builtin path and public example path for each preset invoke
   the same expansion function and finalize to the same Fabric identity.
7. Every preset expands the FU occurrence counts and schedule-local ordinal
   distribution defined by the catalog.
8. A typed FU capability accepts one supported exact scalar or vector point
   and rejects an equal-width semantic type outside that capability.
9. FU materialization and reverse synthesis satisfy the configured-function
   round-trip contract for both equality and strict-superset outcomes.
10. The scalar helper boundary preserves the 128-bit PE interface while its
    internal 64-bit data and one-bit condition roles obey low-bit
    normalization.
11. A stateless scalar resource exhibits the declared one-cycle elastic
    timing, while an imported `LoopCarry` preserves that one-cycle recurrence
    path through elastic-transparent forwarding.
12. A fused carry-or-invariant plus gate template exposes the raw
    parent-domain value, projected child-domain value, and child phase without
    admitting incoherent selector products.
13. Every preset SpatialCore derives the same eight-step stream capability
    from the canonical `LoopControlFu` occurrence order.

These anchors test the public boundary and determinism. They do not require a
fixture for every helper, topology, operation ordering, generated name, or
Fabric operation.

## Non-Goals

The Builder is not a software mapper, simulator, Evaluation result owner, RTL
semantic owner, second hardware IR, or packaging specification. It does not
embed one workload's placement, route, bitstream, simulation result, or DSE
decision into Fabric.

Fabric-to-RTL-to-GDSII implementation closure remains owned by
`docs/spec-hardware-implementation.md`, `docs/spec-implementation-platform.md`,
`docs/spec-rtl-lowering.md`, and `docs/spec-eda-tooling.md`. Preset source
layout, example executable names, installation layout, and packaging are
nonsemantic implementation choices.
