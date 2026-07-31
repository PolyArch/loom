# Fabric Capability Rationale

Normative contracts are owned by
[Hardware Sharing Groups](../spec-fabric-hw-share-group.md),
[Reconfigurable Operations](../spec-fabric-reconfigurable-op.md),
[Fabric FU](../spec-fabric-fu.md),
[FU Synthesis](../spec-generalize-subgraphs-to-fu.md),
[Fabric PE](../spec-fabric-pe.md), and
[Temporal PE](../spec-fabric-pe-temporal.md).

## Why Capability Has Several Owners

Four facts must remain distinct:

* OperationSchema owns exact software actor semantics and parameter codecs;
* an HSG implementation family owns which typed actor families can share one
  real datapath implementation;
* a concrete `fabric.op` owns the subset, hardware parameters, physical ports,
  timing, state, and constraints actually implemented by that resource; and
* a provider registry owns whether a backend implementation is available.

Collapsing these facts creates false implications. HSG membership does not
give every concrete instance every family member, and a supported semantic
operation does not prove an RTL provider exists. Conversely, a backend name
table cannot define what the hardware means.

A software operation may be admitted by several physical families, but one
concrete capability template selects one family. This preserves genuine
hardware alternatives without making family selection ambiguous.

GEP illustrates why family membership is physical sharing rather than semantic
aliasing. A stable integral pointer's final address formation may reuse the
same custom SpatialCore adder used for integer arithmetic, or a Fabric may
provide a dedicated address-generation unit. Both families can admit the one
LLVM-owned GEP schema. A concrete integer adder supports GEP only when its
`op_list` and pointer-format relation say so; neither the RISC-V InstructionCore
nor a coincidentally wide port grants that capability implicitly.

Bit-preserving token resources need no parallel pointer-format catalog. They
do not interpret a pointer, so the exact module DataLayout and concrete port
capacity completely determine whether all representation bits can be carried.
GEP, pointer conversion, comparison, and dereference remain separately admitted
operations. Requiring every mux, sync, carry, or channel to enumerate the same
pointer formats would duplicate the DataLayout without adding a hardware fact.

Ordinary and saturating floating-to-integer conversion share most of one real
converter datapath. Treating saturation as a separate implementation family
would hide that physical sharing; adding a `supports_saturation` flag would
repeat the enabled-member fact already owned by `op_list`. The existing
converter family therefore admits both semantic forms, while each concrete
resource lists only the forms its circuitry implements. This keeps physical
sharing in the HSG, exact semantics in OperationSchema, and the installed
feature subset in one concrete capability.

## Why Capability Is Parameterized

The earlier model enumerated every exact semantic mode and took Cartesian
products of local selectors. It cannot finitely represent arbitrary constants,
predicates, arities, or correlated parameter domains. It also turns don't-care
bits and equivalent encodings into fake functions and explodes the
TechMapping search space.

Loom instead uses a compact typed relation. Operation schemas interpret exact
actors; the selected HSG constrains a real implementation family; `op_list`,
`hw_params`, ports, and typed constraints describe which actor semantics and
ordered port correspondences a concrete resource accepts. Exact constants,
types, predicates, masks, and attributes remain owned by the Dataflow actor and
are bound by TechMapping.

Physical payload width alone is insufficient. A 128-bit port does not prove
support for either four `f32` lanes or two `f64` lanes; element representation,
lane geometry, operation semantics, and policy must be admitted by the typed
capability relation.

## Why FU Topology Is Explicit

A `fabric.fu` contains real operations and configurable mux/demux topology. An
FU can materialize a small software graph, not merely one instruction. The
reason target-specific grouping is owned by Mapping rather than Dataflow is
centralized in [Mapping And PnR](mapping-and-pnr.md); this section concerns only
why the physical FU topology must be explicit.

Inside an FU, multiple SSA uses are real broadcast obligations. A demux is a
selective one-of-N router, not broadcast. If an adder and multiplier are
mutually exclusive, each shared input needs an explicit demux and the results
need a matching mux. Directly feeding both datapaths means both must accept the
token; an inactive operation cannot be assumed to drain it.

Hidden drains for unselected operations or mux inputs were rejected because
they repair an incomplete topology with invisible behavior. They also mask
deadlock and backpressure errors. Every mutually exclusive path and every real
fanout must be represented in Fabric.

## Why Effective Functions Are Unique

An FU exposes normalized structural/capability templates, not raw bit
assignments. Conditional fields disappear when inactive. Two assignments in
the same realization template and actor/port relation that materialize the
same complete typed software graph are a Fabric design error, not an
enumerator dedup opportunity.

This rule removes meaningless configuration space without erasing real
alternatives. Distinct physical templates or actor-to-op relations may produce
isomorphic software functions and remain distinct TechMapping candidates.
Pipeline depth, bypass, and other semantic-preserving physical choices remain
SpatialMapping refinements. Equivalent raw encodings are canonicalized by the
ConfigurationABI writer.

## Why Tech And Physical Configuration Are Separate

Some fields select which software graph is realized; others preserve that
graph while changing latency, buffering, frequency pressure, or power. The
first class belongs to TechMapping. The second class belongs to SpatialMapping
only when Fabric explicitly declares the refinement. Configuration image bits
are a final encoding of those selected facts, not another source of truth.

Runtime selectors of software `dataflow.mux` and `dataflow.demux` remain
tokens. Static selectors of FU-local `fabric.mux` and `fabric.demux` are
configuration. Sharing one implementation framework does not make their
selector semantics interchangeable.

## Why Synthesis And Materialization Share One Relation

FU synthesis from a set of software graphs and materialization from the
resulting FU are inverse views of one capability relation. Requiring the source
set to be contained in the materialized set permits useful extra capability
without demanding exact equality. Both equality and strict-containment cases
matter; extra functions must arise from legal HSG capability, not don't-care
or Cartesian-product accidents.

The materialized domain may be symbolic or large. Counting every encoding is
not a universal legality test. Exact counts are metrics only for finite domains
where they can be computed without forcing enumeration.

## Why PE Endpoint Binding Is Factorized

Pre-enumerating every FU-port to PE-port assignment multiplies endpoint choices
before routing has any information. The mapper instead places a realization at
a compatible FU occurrence and retains a domain of compatible PE boundary
endpoints for each FU port. Route search selects endpoint and path together.

Spatial endpoints cannot serve two simultaneous logical edges. Temporal reuse
is possible only through explicit tagged match domains, context bindings,
capacity, and non-conflicting Physical Tags. Untagged endpoints never gain
multi-edge capacity by convention.

## Why Instruction Context Owns Runtime State

Static instruction memory holds configured operation and routing choices;
runtime state holds stream recurrence, carry, invariant, gate, and operand
queue state. The latter is isolated per Instruction Context and is not
bitstream configuration.

An FU or Compute Realization is not one dynamic firing unit. Active actors fire
according to their own Dataflow transitions while sharing physical resources
under the PE contract. This avoids macro-actor semantics and lets a temporal PE
schedule multi-actor configured graphs without redefining the software.
