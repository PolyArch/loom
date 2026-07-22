# ADG Builder Boundary

This document defines the architectural boundary of Loom's public C++ ADG
Builder library. It deliberately does not freeze concrete class names or
convenience-helper signatures; those belong to the dedicated public API design.

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
remain suitable for nonsemantic labels. A dedicated public API specification
uniquely owns concrete signatures and handle types; this boundary owns only
their required elaboration into closed Fabric schemas and defines no
placeholder handle hierarchy.

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

These anchors test the public boundary and determinism. They do not require a
fixture for every helper, topology, operation ordering, generated name, or
Fabric operation.

## Non-Goals

The Builder is not a software mapper, simulator, Evaluation result owner, RTL
semantic owner, second hardware IR, or packaging specification. It does not
embed one workload's placement, route, bitstream, simulation result, or DSE
decision into Fabric.

The concrete public API, builtin preset catalog, and Fabric-to-RTL-to-GDSII
implementation closure remain separate design subjects. This boundary only
constrains them to preserve Fabric as the single hardware semantic source.
