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
Fabric in addition to the resources they generate. Exact per-helper
operation/HSG/hardware-parameter tables, switch construction, memory-port
capacity, and buffer capacities are part of the same versioned family contract
and must be fixed before that catalog version is implementation-complete.

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
family. Distinct integer, floating-point, multiply, and special-function
datapaths remain distinct physical operations unless the normative HSG
registry and Fabric-to-RTL backend prove genuine circuit sharing. A configured
FU that selects among separate datapaths uses explicit coherent input
`fabric.demux` and output `fabric.mux` topology.

`MacFu` is an FU graph, not a synthetic MAC HSG. `VectorAdapterFu` similarly
does not imply that its four software operation families share one circuit.
`dataflow.pack` and `dataflow.unpack` may share an implementation family only
when the typed HSG registry and backend realize one genuine reinterpretation
datapath. The stateful `dataflow.parallelize` and `dataflow.serialize`
capabilities remain distinct unless one backend-supported stateful
implementation family proves otherwise.

Memory actors, including load, store, atomic, compare-exchange, and fence, are
implemented by `fabric.mem` and never enter this FU library.

### Payload And Type Floor

Every preset in the initial family uses a 128-bit ordinary PE and
intra-SpatialCore data-transport payload capacity. Narrower scalar values
occupy the low payload bits under the Fabric width rules. Physical Tags remain
a separate field and never contribute payload capacity.

The common scalar type floor covers integer and resolved index widths through
64 bits and floating-point element types `f16`, `bf16`, `f32`, and `f64`,
subject to each registered operation schema. The fixed-vector floor covers
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

Applying the rule separately guarantees that every preset exposes the common
capability floor in both schedule kinds. For example, the Small preset has
`12/6/3/3/3/2/1` occurrences across its 12 Spatial PEs and
`4/2/1/1/1/1/1` occurrences across its four Temporal PEs in the table's FU
order.

A builtin descriptor may claim this catalog version only when every listed
software operation has a registered typed operation schema, a legal concrete
Fabric capability, and a compatible Fabric-to-RTL implementation. Operation
names or equal port widths cannot substitute for any of these requirements.

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

These anchors test the public boundary and determinism. They do not require a
fixture for every helper, topology, operation ordering, generated name, or
Fabric operation.

## Non-Goals

The Builder is not a software mapper, simulator, Evaluation result owner, RTL
semantic owner, second hardware IR, or packaging specification. It does not
embed one workload's placement, route, bitstream, simulation result, or DSE
decision into Fabric.

Fabric-to-RTL-to-GDSII implementation closure remains a separate design
subject. Preset source layout, example executable names, installation layout,
and packaging are nonsemantic implementation choices.
