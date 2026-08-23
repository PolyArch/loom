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

The integer/floating conversion parameter record consequently owns only its
supported endpoint relation. Giving it a `FloatBehaviorProfile` would duplicate
rounding and exceptional-result semantics already fixed by the enabled
OperationSchemas, and would admit profile values that no actor can select.
Removing that profile preserves one owner for each fact and prevents an orphan
configuration dimension.

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

## Why Special-Math Accuracy Is A Capability

A special-function circuit's worst-case numerical error is an observable
hardware guarantee. It is neither an operation-name property nor a gate-level
implementation detail. A correctly-rounded implementation can refine an actor
that accepts two ULP, but the reverse is false, so exact admission needs an
ordered relation between the actor-owned allowance and the Fabric-owned
guarantee.

Fast-math cannot own this relation. `afn` grants permission to approximate and
other flags grant distinct exceptional-value freedoms; none quantify error.
Overloading the permission mask with an implied ULP level would prevent two
Fabrics with different numerical quality from admitting the same authorized
actor honestly. Likewise, choosing accuracy through DesignWare, ChipWare, or
an FPGA recipe would let tool availability change Fabric semantics.

The special-math families therefore use one typed parameter record that
composes the existing floating formats and behavior profile with one guarantee
from the compiler-owned accuracy domain. Ordinary floating arithmetic, divide,
and remainder keep their existing records. This avoids an optional accuracy
field on every floating family and gives the twenty-two `ScalarMath*` families
the one additional fact their physical implementations actually need.

## Why RootRelative Index Width Belongs To Each Access Row

DataLayout owns the exact width of a software `index`; a Fabric endpoint owns
only physical payload capacity. Inferring accepted index widths from that
capacity would make a 128-bit endpoint silently choose 32-bit or 64-bit index
semantics. A root-wide index width would duplicate DataLayout and would prevent
one Fabric from admitting several exact software layouts.

The RootRelative memory row therefore records the widths that its address
generation circuitry accepts. Keeping this domain in the existing reduced
product preserves the important correlation with lane count: a 128-bit
indexed endpoint can admit four 32-bit indices and two 64-bit indices without
also claiming four 64-bit indices. A separate pair table would duplicate that
relation, while independent width and lane sets would create the false
cross-product.

PointerAddressed access is different: its semantic admission depends on the
exact DataLayout-owned pointer format, not on an `index` width. Reusing the
same address-form-selected relation slot keeps these alternatives disjoint and
avoids a meaningless index-width field on pointer rows.

## Why Vector Hardware Has Several Physical Owners

One universal vector engine would combine elementwise arithmetic, slice
alignment, arbitrary shuffle, memory transactions, and stream adaptation into
one mode product. Those functions have different ports, state, timing,
resource use, and implementation cost. Combining them would make a convenient
FU name rather than a truthful hardware-sharing relation.

Elementwise arithmetic therefore remains in its existing typed vector
families. Extract and insert share one slice-align-merge family because one
real unit can reuse position decode, alignment, and merge logic. Shuffle uses
a separate two-input block-selection family because requiring every slice unit
to contain a full permutation network would overstate its circuit. Vector
load/store remains a `fabric.mem` responsibility because it owns requests,
mask suppression, child transactions, consistency, and retirement.

This split does not create several vector semantics. Standard Dataflow/MLIR
actors remain the sole semantic owner, and each physical family consumes the
same exact actor projection. A custom architecture can add another family only
when a provider implements that real circuit; equal width or FU co-location is
not evidence of sharing.

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

PE configuration fields follow the same factorization. One activation field
chooses the active FU, and one selector field names each concrete FU boundary
port. Existing FU operation fields remain owned by their capability relation.
A single PE-wide codebook would enumerate the Cartesian product of activation,
routing, and operation behavior and would duplicate the FU codecs. A field set
that changed with the selected Mapping would instead make ConfigurationABI a
workload-specific schema. The factorized static inventory retains every real
choice once while letting Mapping project only the fields observable in the
selected configuration.

## Why Instruction Context Owns Runtime State

Static instruction memory holds configured operation and routing choices;
runtime state holds stream recurrence, carry, invariant, gate, and operand
queue state. The latter is isolated per Instruction Context and is not
bitstream configuration.

An FU or Compute Realization is not one dynamic firing unit. Active actors fire
according to their own Dataflow transitions while sharing physical resources
under the PE contract. This avoids macro-actor semantics and lets a temporal PE
schedule multi-actor configured graphs without redefining the software.

Instruction-context occupancy is nevertheless not a shareable ResourceUse.
One context is one resident dispatch slot for one Compute Realization. A
Spatial PE has one such slot and therefore one active FU selection. A Temporal
PE has `num_instruction` slots, so several resident graphs may activate and use
several FUs while exact resource contracts arbitrate their simultaneous
firings. Letting two realizations alias one context would require an unmodeled
instruction selector and would conflate their runtime actor state; equal
configuration bytes or non-overlapping events cannot supply that missing
hardware.

### Operand Queue Sharing

The three operand-buffer modes trade replicated storage for service coupling.
Per-instruction queues isolate contexts, per-input-port banks share a concrete
FU input, and all-FU-share maximizes storage sharing while exposing the
strongest admission coupling. Independent logical heads and an explicit
reservation capability are therefore more expressive than a global arrival
FIFO. Pair-aware admission prevents a fair arbiter from repeatedly filling an
unmatched queue, while retaining the cycle-start full-queue rule and atomic
fanout semantics.
