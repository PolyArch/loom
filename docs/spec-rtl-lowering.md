# Fabric To RTL

This document defines the hardware-generation boundary from one exact Fabric
Hardware Description to an immutable RTL `HardwareImplementation`.

## Input And Output

Fabric-to-RTL consumes:

* one fully elaborated exact `FabricRootKind::System` Hardware Description and
  its exact imported Module dependency closure;
* one exact `SpatialCoreOccurrenceRef` subject in that System;
* one exact `ConfigurationABI` for that Fabric;
* one resolved hardware candidate-generator binding;
* one exact `ImplementationPlatform` when the emitted implementation is bound
  to an ASIC technology release or FPGA ordering code; and
* exact provider-owned external input bindings required by selected Fabric
  resources or implementation recipes.

It produces one `loom.hardware_implementation 4.1` whose closed `Rtl`
representation root owns the exact top Module locator and content-addressed
SystemVerilog source closure for that SpatialCore occurrence. The
implementation also owns that occurrence's interfaces, constraints, black-box
contracts, activity-point catalog, and implementation manifest.

The System supplies the concrete occurrence qualification, imported Module
target, Clock/Reset membership, attachment boundary, and occurrence-local
configuration projection. The selected Module hierarchy supplies the internal
RTL structure. A bare Module cannot publish an occurrence-qualified
`HardwareImplementation` because it does not select those System-owned facts.
Conversely, this product does not claim complete System RTL: HostCore,
InstructionCore, System transport, and interconnect implementation remain
outside the SpatialCore implementation boundary.

`docs/spec-hardware-implementation.md` owns the output root, payload roles,
interface and activity catalogs, semantic closure, and finalization.
`docs/spec-implementation-platform.md` owns only the shared ASIC or FPGA target
manifest and technology-corner catalog. Provider descriptors own the exact
external files or tool-bundled resources they consume.

The lowering does not consume Dataflow or Mapping and does not create a
workload-specific RTL design. Workload execution combines the reusable
`HardwareImplementation` with an exact `Deployment`, configuration images, and
runtime inputs.

The `portable_spatial_core_rtl` generator accepts the System and
ConfigurationABI plus zero or one exact ImplementationPlatform. Omitting the
platform publishes architecture-only RTL. Supplying it publishes a distinct
platform-bound HImpl for every SpatialCore occurrence; the generator does not
infer a target from an external tool or rebind an existing HImpl downstream.
Both forms retain the same Fabric-derived source and interface semantics.

This explicit lowering is distinct from the payload-free `FabricModel`
HardwareImplementation used by semantic DFG/CGRA execution. Constructing a
Deployment or requesting core semantic execution must not invoke this lowering,
emit SystemVerilog, or compile an RTL model as a side effect.

## Semantic Ownership

Fabric is the hardware semantic SSOT. RTL lowering implements, but never
extends, these Fabric facts:

* occurrences, ports, directed connectivity, and module/system boundaries;
* compute, switch, memory, FIFO, boundary, and transport capabilities;
* spatial/temporal organization, ResourceState, UsePattern, and GrantPolicy;
* atomic claim acquisition/release and owner-defined resource commit
  transitions;
* latency, initiation interval, capacity, buffering, ordering, backpressure,
  reset, and progress behavior;
* clock, reset, power, address, memory, coherence, and protection domains;
* the selected occurrence's exact System attachment boundary; and
* Mapping-visible semantic and physical configuration domains.

Selector logic, comparator structure, gate decomposition, and naming may vary
when they preserve this contract. A pipeline stage, arbitration order, buffer
visibility, or other cycle-observable difference requires a different Fabric
capability or Mapping-selected Fabric refinement. The backend cannot hide such
a choice in RTL.

Visualization metadata is stripped before Fabric identity and has no effect on
hardware generation.

For a memory or fence service leg, the Fabric-owned
`ServiceLegCarrierAttachment` relation identifies architecture-level candidate
transport endpoints and the SystemMapping RouteTree selects among them. The
relation is not itself a protocol channel. This lowering exposes only the
selected SpatialCore occurrence's declared attachment interfaces. A separate
System interconnect product may refine the selected path into AXI, TileLink,
CXL, custom request or response subchannels, packet or flit fields, adapters,
RTL/IP, and physical encodings. Neither product may infer or persist a second
pairing between memory and transport endpoints.

### Non-Defined Value Refinement

RTL payloads are total fixed-width bit values, while each Canonical Dataflow
`SemanticLane` or scalar result component may independently be Defined, Poison,
or Undef. Fabric-to-RTL applies one lane-local refinement rule through the exact
actor-result-to-physical-payload correspondence:

* when one canonical result lane is `Defined(v)`, the corresponding emitted RTL
  payload bits must equal `v` exactly;
* when one canonical result lane is Poison or Undef, only that lane's
  corresponding RTL payload bits may take any type-correct value; and
* protocol, state, commit, and side-effect behavior must remain within the
  exact actor and Fabric contracts.

A provider cannot turn a poison-generating precondition into a trap, stall,
valid suppression, checker state, or additional sideband. Such observable
behavior is legal only when the exact Fabric capability declares it. A normal
total circuit value is a valid refinement of a non-defined payload; an
unsynthesizable `X`, placeholder, or stub is not a provider implementation.
One non-defined vector lane never relaxes a defined sibling lane.

This refinement observes the canonical output; it does not rewrite Canonical
Dataflow propagation, masks, observation, or `freeze` semantics. If an owning
operation, including a later observation or freeze, produces a canonical
`Defined(v)` lane, its provider must reproduce `v` exactly. A family/recipe that
cannot preserve this relation remains typed `Unsupported`. The same rule covers
disjoint LLVM OR, integer overflow promises, exact shifts, zero-poison count
operations, and every other registered semantic promise.

### Special-Math Accuracy Refinement

A `ScalarMath*` provider consumes both the actor's exact selected
`SpecialMathAccuracyTier` and the concrete Fabric resource's accuracy
guarantee. The common admission relation has already proved that the hardware
guarantee is no weaker than the actor allowance. Lowering must implement that
Fabric guarantee or return typed `Unsupported`; it cannot weaken the guarantee,
reinterpret `afn` as an accuracy level, or select a different tier.

The version 8.1 builtin target has one typed
`BuiltinSpecialMathCapabilityProfile` authoring parameter. `FullCatalog`
retains every version 8.0 elementary-math format under strict IEEE behavior and
a correctly-rounded guarantee. `PortableProviderClosed` selects the exact
per-family format, behavior, and accuracy table owned by
[ADG Builder](spec-adg-builder.md#specialmathfu-resource-inventory). All current
builtin presets select `PortableProviderClosed`. Scalar integer and floating
divide/remainder are invariant across the two profiles.

The profile is resolved before Fabric finalization and is not a backend recipe
or a second selector persisted in HardwareImplementation. A provider sees only
the resulting Fabric-owned capability parameters. It cannot inspect the
original profile spelling, broaden a narrowed domain, or replace a
`FullCatalog` capability with the provider-closed relation. Conversely,
`PortableProviderClosed` does not authorize a runtime query of installed
providers while constructing Fabric; its table is frozen by the target schema.

Backend recipes may realize the guarantee with portable synthesizable RTL,
DesignWare, ChipWare, or FPGA primitives and configured IP. Those choices may
change structure and PPA but not the selected actor tier, Fabric guarantee,
exception behavior, latency, initiation interval, progress, or configuration
contract. A vendor block whose documented numerical behavior is too weak or
not provable under the exact binding is unavailable for that occurrence.

Provider conformance uses an independent higher-precision numerical oracle over
the exact supported formats, relevant boundary classes, and the claimed ULP
bound. A provider-generated golden function is not independent evidence. Full
application correctness remains owned by the application oracle; a leaf-level
ULP test cannot replace it.

The production portable registry must completely materialize the capability
domain emitted by `PortableProviderClosed`; this is a conformance obligation,
not ownership of the target table. Missing tool availability remains an
execution condition, and unsupported topology, interface, or another Fabric
capability still has its ordinary typed outcome. `FullCatalog` is a valid
Fabric target even when the portable recipe cannot implement it. Changing the
provider-closed table requires a new builtin target schema and regeneration of
the Fabric, Mapping, ConfigurationABI, Deployment, HardwareImplementation, and
EDA provenance closure; changing a provider binary alone cannot reinterpret an
existing Fabric.

## Common CIRCT Skeleton

Fabric-to-RTL first constructs one target-independent CIRCT skeleton. The
skeleton is transient compiler IR inside one candidate-generator invocation;
it is not an Artifact, a cache authority, or another hardware semantic model.
An optional diagnostic dump is a removable report projection.

The skeleton uses:

* `hw.module`, `hw.instance`, and HW aggregate types for hierarchy, ports, and
  structural composition;
* `comb` for target-independent combinational logic;
* `seq` for Fabric-declared state, clocks, enables, and reset behavior; and
* `hw.module.generated` for a leaf whose exact Fabric contract is known but
  whose implementation recipe has not yet been materialized.

The skeleton has one subject-rooted top hierarchy. The exact
`SpatialCoreOccurrenceRef` resolves through the System to one imported Module
root, and every internal object is qualified by that occurrence. Clock/Reset
ports, attachment ports, configuration units, operation recipes, external
implementation bindings, memory bindings, and activity relations are all
projected for that subject only.

Module-local hierarchy construction may use ordinary private derived indexes,
but no persistent specialization key or definition-rebased occurrence model is
part of the product. Two SpatialCore occurrences that import the same Module
still produce two independently finalized HardwareImplementations with
different subjects and occurrence-qualified interface closures. Content-
addressed RTL payloads may deduplicate byte-identical source blobs without
merging the Artifact identities or physical ownership of those occurrences.
Workload-selected semantic values are not specialization inputs.

Each Loom-generated abstract leaf is mechanically associated with one exact
Fabric occurrence and `ResolvedFabricOpCapabilityView`. The association is an
internal lookup key, not a second capability record. The skeleton never stores
operation-name classifications, provider ecosystem names, PDK paths, target
part data, or independently chosen latency and resource facts.

The skeleton pass owns all target-independent structure: module boundaries,
connections, handshake composition, FIFOs, configuration transport, clocks,
resets, resource control, and external interfaces. It does not emit a portable,
DesignWare, ChipWare, AMD/Xilinx, or Intel/Altera implementation for an
abstract operation leaf.

The resolved candidate-generator binding then drives target specialization.
For every abstract leaf it selects one typed occurrence recipe, obtains the
same Fabric-owned capability view, and invokes the registered provider. A
provider may replace the leaf with ordinary `hw`/`comb`/`seq` logic, a
target-specific wrapper and external module, or a configured primitive/IP
contract. It may not rebuild the surrounding Fabric structure.

After specialization, no Loom abstract generated leaf may remain. A
target-specific external module may remain only when the resulting
HardwareImplementation contains its exact `BlackBoxContract` and external
implementation binding. The complete module is verified before and after
target specialization, lowered through the registered CIRCT legalization and
Seq-to-SV pipeline, verified again, and only then exported as SystemVerilog.
Failure at any boundary publishes no partial HardwareImplementation.

Fabric is already the exact latency-insensitive, state, scheduling, and HSG
authority. Fabric-to-RTL therefore does not route the complete design through
a second Handshake, DC, scheduling, or operator-library semantic layer.
Individual CIRCT transformations may be used internally only when they
provably preserve the exact Fabric contract and leave no competing persistent
authority.

## Operation Provider Registry

Operation lowering is dispatched by the same closed
`ImplementationFamilyId` used by the normative Hardware Sharing Group
registry:

```text
ImplementationFamilyId -> RTL provider callback
```

Before emission, the backend mechanically constructs one
`ResolvedFabricOpCapabilityView` for each concrete `fabric.op`. A provider
consumes that exact view, including its one sealed
`FabricOpSemanticFieldRelation`,
and the exact `ConfigurationABI`. It may own emitter code, behavioral or
external-IP implementation availability, and typed external dependencies. It
does not own family membership, operation types, HSG legality, behavior-key
equivalence, timing semantics, or configuration encoding.

A provider materializes one decoder from the exact relation and ABI pair. It
cannot rebuild a mode table from operation names, `op_list` order, capability
iteration order, or inactive physical codes. `None` emits no selector,
`Finite` decodes exactly the Fabric-owned finite behavior keys, and `Direct`
uses the exact fixed-width carrier and Fabric-owned validity domain without
finite enumeration. Mapping owns the authoritative actor and refinement
selections. The Fabric projector derives their one joint semantic value,
Mapping's transient `ConfiguredHardwareProjection` carries that value, and the
decoder observes only its ConfigurationABI encoding. The provider cannot split
the joint value into independently selectable modes or infer it from workload
IR.

When a mapped configured function is projected, the selected
`FabricFuCapabilityTemplateRef` supplies only the active FU node and edge set.
Provider dispatch and availability are derived from the active operation
nodes' `ImplementationFamilyId` values. Providers do not own capability
template identity, FU topology, admitted software members, or a substitute
state/timing descriptor.

Operation-name string classification, backend-local exact-mode enumeration,
and global `(operation name, variant)` selection are forbidden as semantic or
dispatch authorities. A behavioral or golden provider uses the same family ID
and exact Fabric contract rather than a second operation-name support table.
Missing provider support is typed `Unsupported`; a partially lowered or
semantically substituted implementation is never produced.

The provider must distinguish the exact resource contract from the selected
operation semantics. The initial scalar `CoreAluFu` and arithmetic `MacFu`
resources lower as the Fabric-declared one-stage registered elastic resources.
That implementation rule is not a default wrapper for every `fabric.op`.
Stateful operations such as `dataflow.stream`, `dataflow.carry`,
`dataflow.invariant`, and `dataflow.gate` require providers that implement
their registered actor transitions together with their exact Fabric-owned
state capacity, atomic use patterns, transition timing, result holding, and
backpressure contract. A generic shell may not consume an inactive operand,
publish an inactive result, advance logical state while blocked, or convert
an operation-specific state machine into a stateless pipeline.

`FixedVectorParallelize` and `FixedVectorSerialize` consume the registered
ordered production groups rather than the one-tuple shell. Their provider
computes only the schema-owned data, mask, phase, and adapter-state relation.
Fabric owns the logical-use claim; the common skeleton materializes that claim
and its capacity-one production slot. It samples the provider's
contract-derived final-production signal only when it captures the matching
group and retains continuation state across every non-final handoff. A held
group or continuation makes the use busy. A zero-output case retires at commit.
The final group handoff releases before acquisition at the same coordinate, so
the following firing may replace it without a bubble. Reset clears both the
slot and continuation together. A provider-local busy convention, hidden
output queue, or independently decoded lane counter is not a valid substitute.

The `LoopStream`, `LoopCarry`, `LoopInvariant`, and `LoopGate` providers are
dispatched by those exact family IDs. They consume the operation schema's
closed typed transition-case descriptors; they must not reconstruct a second
condition decoder or operation-name state table. The enclosing PE supplies
the `InstructionContextRef` selection and state-bank namespace. A provider
implements the state shape for that context but never creates another context
identity.

The initial `LoopCarry`, `LoopInvariant`, and `LoopGate` providers implement
the Fabric-declared elastic-transparent contract with no hidden result queue
or registered forwarding stage. `LoopStream` implements the exact
result-publication, next-state, in-flight-capacity, and initiation-interval
contract of its concrete resource, including multi-cycle recurrence updates
when declared. Provider-local mode tables and a universal stateful-machine
wrapper are not semantic authorities.

`FixedVectorSliceAlignMerge` lowers to the exact position decode, alignment,
slice, and merge network admitted by its resolved capability. Dynamic indices
are runtime ports; static offsets, strides, and any programmable shape or mode
are decoded from the exact ConfigurationABI fields. `FixedVectorShuffle`
lowers to the admitted two-input block-selection network and its ordered
selectors. A poison selector may choose any type-correct output bits for only
that block under the non-defined refinement rule. Neither provider may
scalarize the mapped actor, add lane handshakes, or use a backend-local mask or
width table.

All operation providers derive their signal widths from resolved physical
ports and their supported semantic payload from the family capability. There
is no portable-, vendor-, or target-specific 128-bit default. A provider may
return typed `Unsupported` for a width outside its exact implementation
binding, but cannot substitute a narrower implementation or silently split
the token.

## Implementation Recipes

Implementation choices are classified by their first observable difference:

* A choice that changes exact operation semantics, numeric accuracy, or the
  accepted actor domain is a different Fabric capability or `hw_params`
  contract.
* A choice that changes latency, initiation interval, state, buffering,
  capacity, or progress is a different Fabric contract. It is a Mapping
  physical refinement only when Fabric declares that exact runtime-selectable,
  semantic-preserving domain.
* A choice that preserves all Fabric-observable semantics, timing, capacity,
  progress, and `ConfigurationABI` may be a backend implementation recipe,
  such as two gate decompositions with different PPA.

The exact hardware `ResolvedCandidateGeneratorBinding` selects backend recipes
per occurrence:

```text
FabricPhysicalOccurrenceOwnerRef -> typed BackendRecipeKey
```

Recipe selection is not global by operation name. It is recorded in
the resolved generator configuration, and `InvocationManifest` records the
selection on the derivation edge. The selected provider must materialize every
resulting implementation fact in the HardwareImplementation payloads and
bindings. Fabric identity remains unchanged. Accuracy, timing, or other
Fabric-visible differences cannot be hidden behind a recipe key. A provider
may report an unavailable recipe or external dependency, but it may not
silently choose another contract.

The initial recipe catalog supports five provider families through the same
occurrence-scoped mechanism:

* portable synthesizable SystemVerilog;
* Synopsys DesignWare;
* Cadence ChipWare;
* AMD/Xilinx primitives and configured IP; and
* Intel/Altera primitives and configured IP.

These recipes may materialize different RTL, wrapper modules, primitive or IP
instances, parameters, black-box contracts, and external implementation
bindings. DesignWare, ChipWare, AMD/Xilinx, and Intel/Altera recipes are not
aliases for portable RTL and are not aliases for one another. A downstream EDA
tool consumes the already selected implementation; choosing that tool cannot
silently rewrite the occurrence recipe.

This list is an implementation-provider catalog, not a second operation
catalog. One provider keyed by an `ImplementationFamilyId` may implement every
compatible operation admitted by that HSG family. It consumes each concrete
operation's resolved capability view instead of classifying operation-name
strings.

Vendor inference is not an implementation identity. A vendor-specific recipe
emits an explicit wrapper, primitive or IP instantiation, and exact
`BlackBoxContract` where required. The resolved generator binding owns the
selected recipe keys and required provider-library slots for that invocation.
An explicit external file is selected by its provider-owned typed input slot
and exact content fingerprint. A resource distributed with DesignWare,
ChipWare, Vivado, or Quartus is selected by the exact provider tool/build
identity and resource key. No recipe imports a PDK, IP tree, device database,
or tool installation into ImplementationPlatform.

`external_implementation_bindings` record every provider dependency that
remains necessary to reconstruct or consume the resulting
`HardwareImplementation`. A binding materializes the selected provider,
resource or file identity, occurrence relation, and representation locator;
it never contains a host path or a free-form property map.

Portable RTL remains a separately selectable recipe and is mandatory for the
open-source flow. It may also be selected explicitly for a commercial flow.
Failure to resolve a selected vendor recipe is `Unsupported`; the backend may
not silently replace it with portable RTL, downstream inference, or a different
vendor primitive. Conversely, merely running a vendor EDA tool does not change
a portable recipe into a vendor-specific one.

One exact Fabric and ConfigurationABI may feed several resolved implementation
flows with different occurrence recipe maps. Portable, DesignWare, ChipWare,
AMD/Xilinx, and Intel/Altera selections normally produce distinct immutable RTL
HardwareImplementations because their materialized sources, black-box
contracts, or external bindings differ. If two selections produce identical
canonical implementation state, they converge on one HardwareImplementation
identity while `InvocationManifest` retains both derivation paths. Compatible
downstream evaluators may observe any resulting implementation, but their tool
choice cannot rewrite its materialized recipe. This explicit fan-out is the
only cross-ecosystem comparison boundary; there is no implicit provider
conversion.

An HLS product such as Stratus or Vitis HLS is not another RTL recipe. It may
be registered as a candidate generator only when Loom has an exact typed
high-level body, interface, protocol, numeric, and timing contract that the
generator consumes directly. The baseline Fabric-to-RTL path already emits RTL
and never reconstructs such a body from generated SystemVerilog. Software
deployment, platform packaging, and device execution through Vitis are likewise
separate typed generators or evaluators when their required owners exist.

## Structural Lowering

Every Fabric connection lowers to explicit RTL connectivity. Replication,
fan-in, arbitration, temporal sharing, tag-domain transformation, and protocol
conversion appear only when represented by the corresponding Fabric primitive
or selected refinement. Same-kind endpoint width normalization is not a
resource or refinement; RTL derives it directly from the two endpoint types.

Fabric's port rule remains low-bit aligned: a wider source is truncated at the
high end and a narrower source is zero-extended at the high end. The rule
applies independently to payload and tag fields of same-kind `bits_tag`
connections. `bits` and `bits_tag` never convert into one another implicitly.
RTL emits the required slice or zero-fill wiring without an adapter node,
configuration field, or route hop. Hardware modules must preserve each
endpoint's declared tag width and temporal tag behavior.

Unrealizable or unsupported resources fail lowering. They are not replaced by
a similar primitive or silently emitted as behaviorally different logic.
`fabric.fifo` lowering preserves the capability and selected-mode contract in
`docs/spec-fabric-fifo.md`; implementation structure cannot change buffered
visibility, bypass backpressure, or inactive-state semantics. A
`per_tag_virtual_channel` occurrence lowers to one compacted slot bank with an
offer cursor over Physical Tag values: each cycle presents the arrival-oldest
entry of the resident channel nearest the cursor in wrapped ascending tag
order, and a grant or a refused offer moves the cursor past that channel,
exactly the arbitration transition the Fabric contract owns.

An unbound or inactive operation input is not an implicit sink. A provider
must not assert readiness merely to drain and discard tokens unless the exact
Fabric capability explicitly defines that consumption and backpressure
behavior. FU-local selection remains the explicit `fabric.mux` and
`fabric.demux` topology owned by Fabric.

Boundary lowering implements the normative atomic ready/valid equations in
`docs/spec-fabric-boundary.md`. Two-input `s2t` cannot consume either input
alone, and split `t2s` cannot publish either output alone. The base boundary
has no register or holding state; adding one is a behavior-changing Fabric
refinement, not an RTL convenience. Reusable RTL lowering emits the complete
Fabric-owned configuration domain and does not consume or specialize to a
Mapping. Before mapped execution, the exact Mapping must pass the selected
combinational handshake closure in `docs/spec-mapping-verification.md`; that
Deployment or execution gate cannot be bypassed by the RTL harness. The
lowerer cannot omit Fabric alternatives, add a backend-local loop-breaking
rule, or treat one mapped selection as the reusable hardware topology.

Every consumer readiness in the lowered transport network is observable
before the token's valid arrives, because the atomic fanout equations of
`docs/spec-fabric-switch.md` assert valid on one selected output only after
every peer output is ready. A Temporal switch therefore presents a candidate
input's tag on each output it routes to whenever no valid requester holds that
output: valid requesters are presented by the exact GrantPolicy; among idle
candidates whose selected outputs overlap, a free-running rotation presents
one at a time, and idle candidates whose selected outputs no other candidate
claims are presented together. Only another input's grant excludes an idle
candidate. An input is ready only while it is presented on every output its
resident row selects. A row that contends with no other resident row is always
presented, so its readiness reflects only its outputs' readiness and never the
port's own valid. Physically admitted crosspoints that no resident row selects
create no configured combinational dependency or grant exclusion. Rows of
different inputs that select a common output are presented one at a time and
granted by the exact `GrantPolicy`; within the resulting configured component,
every input readiness observes every component
input validity. Round-robin output validity observes the complete component;
fixed-priority output validity follows only the exact directed requester-prefix
relation. These are the configured grant and readiness-presentation
dependencies owned by `docs/spec-fabric-switch.md` and checked by the selected
closure in `docs/spec-mapping-verification.md`. RTL implements idle
presentation with one switch-local cursor over the full typed policy order:
FixedPriority starts at position zero, RoundRobin starts at its typed reset
requester, and every non-reset edge advances once modulo the order. From that
position, it greedily presents configured candidates with disjoint selected
outputs. The RTL implementation identity covers this mechanism, including its
order, reset, advance, and greedy selection. Clock and Reset ports follow the
nonempty ResourceState rule in `docs/spec-fabric-module.md`, independently of
whether a cursor exists. The presentation cursor never reorders grants among
valid requesters.

A Temporal PE presents the context its context-evaluation service grants to
one FU as that FU's single dispatch context for the clock cycle. The FU and
each operation shell inside it evaluate the configuration slot, state bank, and
operand heads of that context alone; no operation shell infers its context from
the tokens on its inputs. A boundary token presented to the FU belongs to the
dispatch context by construction, so an operation transition fires exactly when
the heads consumed by its schema-owned case are valid, whatever its other inputs
hold: a `dataflow.carry` in its initial state consumes the Init head alone, and
a running `dataflow.stream` continues with no input at all. An FU-internal
result names the context that produced it and is deliverable to another
operation of the same FU only while that context is the dispatch context; its
producer stays busy until then. Every FU output reports the producing context so
the PE applies that context's result selectors; an output holding no result
reports the dispatch context. Result egress follows the same
readiness-before-valid rule as the switches; the Temporal PE specification
owns the distinct offer and service-grant cursor events. A PE output port
offers valid FU outputs by the canonical round-robin policy, advancing its
cursor past every offered requester whether the downstream accepted or
refused the offer (the offer rotation of the per-tag virtual channel
discipline). A result whose downstream is not ready for its tag never holds
the port against the other valid results; the grant is the offer that is
accepted. A register FIFO also presents one selected FU output, but its
write cursor follows the Fabric register-FIFO service contract and advances
only when that write commits.
While no valid requester holds a port, the port presents readiness to the
routed output of one idle FU at a time, the same FU on every port of the PE,
so an operation that publishes several results atomically observes their
capacity before it asserts any valid. The idle FU follows the
context-evaluation service, not a free-running rotation. Under a shared
service, it is the FU granted this cycle; under per-FU services, a pointer
holds each FU with eligible rows for one complete pass of its dispatch
rotation, so every resident context of every FU is presented on its ports
while it is the dispatch context (a rotation whose period shares a factor with
the FU count would otherwise never align with it).
Inside a Temporal PE's FU, several
operations may hold results for one FU boundary output at the same time because
their resident contexts differ; that output grants exactly one producing route
per cycle by the canonical round-robin policy over its routes, presents the
granted result and its context, and hands the boundary's readiness to the
granted route alone, so a handoff retires exactly one result. Admissibility of
a route never observes that route's own source valid, which keeps readiness
observable before a transparent operation publishes. The PE tells each FU
output whether it is offering that output to a port this cycle, and the FU's
cursor advances past the granted route on every offered cycle, accepted or
refused, so the same offer rotation applies among the held results of one FU;
an output the PE is not presenting keeps its grant. An operation input is driven by at most one active route per
cycle, because one capability template is active per dispatch context, and
therefore carries no cursor.

Temporal-PE operand storage is emitted from the exact required
`operand_buffer_size` and mode-derived allocation units. The base contract has
one enqueue and one dequeue service per allocation unit per local cycle, with
the declared canonical round-robin policy where contention is possible. RTL
must not substitute a default depth, extra port, global arrival-order head, or
implementation-private priority.

RTL derives the QueueKey and allocation-unit inventories directly from the
Fabric operand-buffer contract. A boundary input is ready exactly when its
token matches at least one active selector and every matched logical queue has
cycle-start capacity in its allocation unit and holds that unit's single
enqueue service; the whole MatchKey fanout commits together or not at all.
Readiness observes configuration, the presented tag, cycle-start queue and
occupancy state, and the competing requesters of a shared unit. It never
observes the port's own valid, so an atomic upstream fanout that reaches
several ports of one PE cannot close a combinational loop through this
boundary. Where several queues of one shared unit request its enqueue service
in one cycle, the grant follows the Fabric transaction order: a transaction
completing a context/FU tuple, then a near-full complementary transaction,
then ordinary transactions under the unit's canonical round-robin requester
order, whose cursor advances only on a committed enqueue. A dedicated unit has
at most one requester and carries no policy. Queue heads and near-full state
are observed at cycle start, so priority does not create same-cycle
replacement capacity.

Synthesis preserves this semantic hierarchy. Each canonical `fabric.op`
recipe, switch form, memory form, FIFO, operand queue, and other repeated leaf
is compiled once per exact implementation dependency closure. A SpatialCore
implementation composes those compiled blocks with its occurrence-independent
local interconnect and top interfaces; occurrence qualification then binds the
exact System subject, ConfigurationABI, platform, and external interfaces.
Neither synthesis nor a provider adapter may flatten the complete SpatialCore
as its primary implementation unit or recompile an identical leaf or Module
representation for every occurrence. A required cross-boundary optimization
must be an explicit bounded refinement whose result remains verifiable against
the same Fabric contracts.

## Clocks, Reset, And Quiescence

RTL exposes the exact Fabric clock/reset domains and only their declared
crossings. Stateful resources start in their canonical initial state and, for a
legal completed invocation, satisfy Fabric's self-reset/quiescence contract
before the same physical slot is reused.

The backend obtains these facts by refining the exact imported Fabric artifact
to `FabricSystemRootView`, selecting the exact SpatialCore subject, and
consuming the subject's occurrence-qualified effective-domain, connection,
attachment, transport-resource, and crossing closure. Enclosing System facts
are used to resolve and validate that closure, not to claim a complete System
RTL product. The backend does not accept a backend-owned clock/reset manifest,
caller-supplied connection list, inherited AccCore domain, or copied crossing
catalog. Derived RTL or SDC indexes are disposable caches validated against
the selected subject in that one root view.

Power, clock-gating, reset synchronization, and backend constraints are emitted
only from explicit Fabric implementation facts. An asynchronous clock crossing
lowers only from the exact `ClockCrossingContract` on its owning transport
resource. Reset synchronizers and release latency lower only from the exact
Reset domain contract. Missing implementation support is a typed failure, not
permission to omit required behavior or invent a crossing.

## Memory And Occurrence Interfaces

`fabric.mem` lowers its operation engine, internal dependency forwarding,
configurable service dispatch, optional local storage, and manager/subordinate
interfaces without adding storage semantics absent from Fabric. The
SpatialCore product terminates at its exact attachment interfaces; System
interconnect implementation is a separate product that must preserve the
architecture service, multicast, ordering, capacity, and progress contract.

The backend implements the Fabric-owned memory operation-port inventory,
capability alternatives, parameterized access domains, mask endpoints, and
declared use patterns. A complete element, contiguous, or indexed
address/data/mask token enters one operation endpoint. A selected use pattern
may decompose that firing across several service transactions or beats and
must implement inactive-lane suppression, masked-load zero fill, row-major
result assembly, and one logical retirement event. Endpoint payload width and
service beat width are independent facts; the backend cannot infer
decomposition from their ratio or reinterpret Physical Tags as vector lanes.

Data, scalar-address, indexed-address, and mask endpoints may all have
different widths. The generated interface follows each selected endpoint type
independently. LSB truncation or zero-fill occurs only on an explicit
same-kind Fabric connection; the selected Mapping has already proved that the
complete semantic token fits every traversed segment. The memory provider may
derive narrower service transactions only from the selected owner-defined
transaction projection and service contract.

An implementation of a `MemoryConsistencyDomain` must preserve its exact
release-visibility point, fixed linearization and retirement rules,
`BoundedCompletion` or `FairEventual` progress contract, ResourceStates, and
atomic UsePatterns. Backend queues, caches, or protocol adapters may realize
those facts but cannot silently strengthen one emitted configuration and
weaken another behind the same Fabric identity.

The leaf-channel shape is mechanical:

```text
!fabric.bits<W>       -> data[W] when W > 0, valid, ready
!fabric.bits_tag<W,T> -> data[W] when W > 0, tag[T], valid, ready
```

`!fabric.bits<0>` therefore emits only valid/ready, while
`!fabric.bits_tag<0,T>` emits tag plus valid/ready. RTL must not create a
zero-width data vector. Spatial memory uses the untagged form. Temporal memory
implements the configured per-role input `(endpoint, tag)` matches and output
`(endpoint, tag)` writes; it must not replace them with one common row tag or
use the operation kind as a runtime match key.

Manager and subordinate `memref` capabilities remain typed internal service
interfaces. Their AXI, TileLink, CXL, or custom physical pinout is selected by
the exact HardwareImplementation and is not inferred from the `fabric.mem`
operation-channel schedule.

### Common Portable Spatial Service Profile

The common target-independent skeleton provides one explicitly selected
portable profile for the current plain load/store service domain. It is not a
Fabric default and it is not inferred from a `memref`. A candidate that selects
this profile materializes that choice in the subject's implementation payload,
interface closure, and external implementation bindings.

The profile carries one complete Runtime ABI-owned logical request rather than
inventing a beat protocol. Its request channel is ready/valid and contains the
plain read/write selector, the complete address payload, write-data payload,
dynamic lane mask, `All | Bits` active-lanes selector, access-form selector,
address-form selector, element width, flattened lane count, address-lane width,
base address, and transient context. Its response channel is ready/valid and
contains the complete read-data payload. Completion is the response transfer
itself. A store ignores response data. Fields inactive for the selected
request are zero. Zero-width payload fields are omitted rather than emitted as
zero-width HDL vectors.

The address, data, and mask carrier widths are the maxima mechanically derived
from the exact operation endpoints and local-service capabilities in the
selected SpatialCore occurrence. The access-form selector uses the closed encoding
`Element(0) | Contiguous(1) | Indexed(2)`; the request-kind selector uses
`Read(0) | Write(1)`; the address-form selector uses
`RootRelative(0) | Pointer(1)`; the active-lanes selector uses
`All(0) | Bits(1)`. Element width, lane count, and base address are unsigned
64-bit values, address-lane width is unsigned 32-bit, and transient context is
unsigned 64-bit. This profile admits one outstanding logical request per
physical endpoint, so the Runtime ABI transaction handle has the singleton
physical encoding and no HDL signal. These encodings belong only to this
implementation profile and do not become Fabric or Runtime semantic owners.

A manager endpoint emits the request and accepts the response. A subordinate
endpoint accepts the request and emits the response. Module-boundary memory
attachments and fixed memory-service connections wire those two directions
mechanically. Provider decode, constant base translation, response return, and
backpressure remain inside the portable memory controller.

The initial mapped provider decoder admits `Range` and `Prefix` match fields.
`AddressSpace` and `Context` require exact values from owners that the current
Mapping artifact does not project into a provider row, so this profile returns
typed `Unsupported` instead of guessing either value or reinterpreting a
Physical Tag as context. A later profile may consume those owner projections
without changing Fabric provider-decode semantics.

One physical operation row contains one base address and one service target.
When a reusable actor has several rooted uses, this profile can project that
row only if every use selects the same target and root-relative base
translation. A Mapping with different per-use choices remains valid Mapping,
but selecting this implementation profile for it is typed Unsupported; the
backend cannot discard a use, choose one use as authoritative, or add an
unowned runtime selector.

The portable profile is available only when every reachable operation is a
plain load or store and every selected access fits the derived carriers. A
fence, atomic load/store, RMW, compare-exchange, wider service, or stronger
outstanding-request contract requires another exact implementation profile;
the common provider returns typed `Unsupported` instead of silently weakening
it. AXI, TileLink, CXL, and custom profiles may translate the same Runtime ABI
boundary, but they do not change its Mapping, retirement, or memory-consistency
semantics.

The profile implements only Local Memory Service regions whose exact behavior
is `Storage`. An `Mmio` region requires an explicit implementation binding and
is typed `Unsupported`; it is never realized by the portable storage array.
Before a local request is accepted, its access form, address form, element
width, flattened lane count, active-lanes form, address-lane width, and every
active byte address must match one selected service capability and one of that
capability's `Storage` regions. A nonmatching or out-of-range request remains
backpressured and cannot alias through truncated SRAM address bits. Inactive
lanes do not require an in-range address because they do not perform a memory
access.

Behavioral memory models and black boxes are legal only when Fabric or its
implementation binding explicitly declares that realization. The
`HardwareImplementation` records their contracts and unresolved external
dependencies.

## Configuration ABI

Fabric-to-RTL implements an exact `ConfigurationABI` for every exposed
Programming Unit. The ABI, not RTL source order or backend-local structs, owns
bit positions, codebooks, padding, programming visibility, and image loading.
The ABI is rooted in the same exact System and uses occurrence-qualified
physical owners and configuration fields for imported Module internals. A
bare Module-local field cannot configure every reuse of one Module as though
they were one physical unit.

Backend-local configuration signal names are implementation details. Every
configuration input and decoder relation must be mechanically derived from the
exact `ConfigurationABI`; an independently designed `cfg_*` interface is not a
public or semantic authority.

Mapped RTL execution must program the implementation through decoded
`HardwareConfigurationImage` artifacts from the exact `Deployment`. Reading a
Mapping directly in a testbench and bypassing the physical programming path is
invalid.

### Common Portable AXI4-Lite Configuration Transport

The common target-independent SpatialCore implementation uses one shared
AXI4-Lite subordinate configuration port. This is an implementation profile,
not a second ConfigurationABI. It has 32-bit `AWADDR` and `ARADDR`, 32-bit
`WDATA` and `RDATA`, four byte strobes, standard `AW/W/B` and `AR/R` channels,
and no burst, ID, or outstanding-reordering mechanism. It shares the selected
SpatialCore Clock and Reset domain; a clock crossing requires another exact
implementation profile.

The top-level signal prefix is `cfg_`. The profile supports one outstanding
write response and one outstanding read response. Write address and data
channels remain independent as AXI4-Lite requires: the implementation captures
at most one address and one data beat in either arrival order, applies the
write only after both are present, and does not accept another pair until the
response is consumed. Accepting the second beat is the write-completion state
transition: the write is applied and `BVALID` becomes asserted from that same
Clock edge, with no separate execute cycle. `BVALID` and `BRESP` then remain
stable until the response is consumed.

The shared `ConfigurationTransportLayout` begins at byte address zero. It
selects the occurrence-local Programming Units and sorts them by the canonical
bytes of their definition-rebased Fabric resource closure and field schema.
For each unit it allocates, without gaps:

```text
payload_word_count = ceil(payload_bit_count / 32)
payload addresses  = base + 4 * [0, payload_word_count)
commit address     = base + 4 * payload_word_count
status address     = commit address + 4
next base          = status address + 4
```

All addresses are four-byte aligned. Layout derivation rejects a total span
that cannot be represented by the 32-bit address bus. Each entry retains the
exact `ProgrammingUnitRef`; the local address order does not replace global
occurrence-qualified identity.

The controller stores two structured arrays of 32-bit words and one active-bank
selector. Payload writes use the dynamic word index and `WSTRB` to update only
the inactive bank; they do not expand the image into byte registers or a
linear address mux. One four-lane generation tag per word and one exact covered
byte count record which required payload bytes have been received since reset
or the last commit. The one-bit generation is sufficient because a successful
commit requires every required byte to carry the current generation before it
toggles the active bank; after the toggle, every retained tag therefore denotes
the previous generation. A write that sets any ABI-unused high bit in the final
word is rejected without changing bank or coverage state. Payload reads use a
dynamic word index into the active bank, so readback after commit verifies the
configuration that the SpatialCore is actually using rather than merely
echoing staging state.

Writing bit zero as one at a unit's commit address requests activation. The
write succeeds only when every required payload byte is covered and all other
strobed command bits are zero. On the accepting Clock edge, the active-bank
selector toggles atomically and the covered count returns to zero; the
previously active bank becomes the next shadow bank. An incomplete or malformed
commit returns `SLVERR` and leaves the active image unchanged. The status word
is read-only; bit zero reports whether the shadow coverage is complete and all
other bits are zero.

Reset restores the exact ABI inactive image at every observable read and field
projection and clears all coverage. Physical bank words may remain unreset
because a reset `initialized` bit selects the ABI inactive word until the first
complete atomic commit; uninitialized storage can never reach a consumer.
Valid payload and status accesses return `OKAY`; a write to status, a malformed
payload or commit write, or another request forbidden by the profile returns
`SLVERR`; an unallocated or misaligned address returns `DECERR`. Responses are
deterministic and never partially update active configuration.

The controller reads each active physical word ordinal once. A
`ConfigurationBundlePlan` is transient derived metadata containing only unique
`(transport_unit_ordinal, 32_bit_word_ordinal, used_bit_mask)` entries required
by one hierarchy subtree, ordered by the first two tuple members. The
controller masks unused co-resident bits and publishes one packed
`hw.array<word_count x i32>` bundle for each top-level component. Every
internal module accepts at most one word-array bundle containing only words it
consumes directly or forwards to a child; parent-to-child projection selects
the exact word subset and reapplies the child masks. No child receives the full
Programming Unit payload unless its exact field closure consumes all of it.

Field decoding reads only the bundle words named by that field's canonical ABI
destination slices, applies wide extracts at word boundaries, and concatenates
the fragments in source-bit order. It never expands a contiguous slice into
per-bit extracts. Interned Switch, FIFO, and Memory cores consume that decoded
module-local value through one canonical input; the root performs the
occurrence-specific word-to-field projection. Physical word placement can
therefore neither split a semantic core definition nor be confused across
occurrences. Bundle membership, ordering, and masks are caches derived solely
from ConfigurationABI and never become a schema or independent configuration
owner. Only the top-level AXI4-Lite port is a programming interface, and no
test or provider may bypass its staging, readback, and atomic-commit behavior.

## Activity And Observability

The implementation owns a deterministic activity-point catalog. Each
`HardwareImplementationActivityPointRef` identifies one observable scalar
signal bit in that exact implementation and mechanically relates it to the
emitted hierarchy/signal locator required by backend tools. Where applicable,
the catalog also records its exact Fabric correlation; Mapping can then derive
actor correlation without making emitted names semantic identities.

The activity-point reference is semantic and implementation-owned. HDL paths,
escaped identifiers, waveform handles, and vendor-native names are locators,
not identities. The complete catalog schema, canonical order, and persistent
closure are owned by the HardwareImplementation Artifact contract. An RTL or
mapped-RTL producer must use that catalog for canonical
`ImplementationSignals` summaries rather than emit private paths or ordinals.

Waveforms, toggle files, testbench logs, and vendor-native products remain
owner-attempt or scratch material and have no current Artifact schema. An
architecture-evaluation descriptor references a
case signature with the `implementation` role. A mapped-RTL simulator
descriptor references a case signature with ordered `implementation` and
`deployment` roles. Requests bind exact
`HardwareImplementation` and, where applicable, exact `Deployment` artifacts;
the models produce Evidence, while mapped execution also produces
`SimulationExecution`. The RTL
simulator alone owns HDL event time; Loom numbers cycles only from explicit
clock-domain edges. Architecture-only lint, elaboration, reset, ABI, or formal
checks produce Evidence without an empty execution artifact.

## Constraint And Verification-Harness Derivation

Generated SDC or equivalent constraints are `GenerationConstraint` payloads of
the exact HardwareImplementation. They are mechanically derived from Fabric
clock, reset, timing, and crossing facts; HardwareImplementation locators; the
resolved generator binding; and the exact target manifest when one is present.
A generated clock, asynchronous relation, false path, multicycle path, IO
delay, or load is legal only when one of those typed owners supplies the fact.
An external PDK or library file is an implementation input, not an independent
timing-constraint authority. Report parsing or backend heuristics cannot
silently create a semantic constraint.

An architecture verification harness is derived from
`HardwareImplementation + ConfigurationABI`. A mapped workload harness is
derived from `HardwareImplementation + Deployment + SimulationWorkload +
SimulationRuntimeInput`. Harness source, simulator scripts, and waveforms are
materialized in an ExternalToolInvocationBundle as owner-attempt execution
material; Loom does not add a `TestbenchArtifact` or another stimulus schema.
The bundle manifest references the exact semantic owners rather than copying
their contracts. The harness must program mapped RTL through the exact
ConfigurationABI path and use the implementation interface catalog rather than
private hierarchy guesses. Its configuration driver is clocked and free of
delay controls: each task samples the handshake at the rising edge and drives
the request channels from it with nonblocking assignments, so the design
observes every request from the following edge; the write task presents the
address and data channels together and retires them per channel, and every
write still completes through its own B response before the next begins. A
simulator evaluates the configuration fan-out at every edge the driver acts
on, so the driver acts only at the sampling edge. The response wait of the
commit write and of the status read alone is taken at the falling edge: the
readback and the kernel launch follow those responses through an idle channel,
so they become visible one edge earlier, at the edges a falling-edge driver
used, which keeps the launch phase of the design's free-running arbitration
rotations and therefore the retirement coordinates of the result identical to
that driver; every other transaction is followed by a busy controller edge and
gains nothing from it. The programmed words, their order, the readback of
every active word, and the atomic commit are unchanged by that shape. The
boundary between the configuration, readback, and kernel stages is announced
once at the ordinary verbosity level with its simulation time, because those
three stages have very different cost and the stamp gives each its exact cycle
count.

`loom-system-run --mapped-rtl` receives the selected simulator's machine-local
binding only through its explicit mapped-RTL local configuration input. The
separately declared exact version-probe line becomes the
`MappedRtlSimulatorBinding` in the Request; the driver cannot derive that
semantic identity from the ambient process. Provider preparation probes the
configured binding, requires the resulting line to equal the Request binding,
and admits it only through the backend catalog's validated-release relation.
The external-tool invocation specification owns the explicit-binding
requirements of this driver boundary.

The portable mapped-RTL provider derives a tool-local hierarchical compilation
plan without making generated SystemVerilog a second hierarchy authority. The
portable publisher freezes the exact post-`LowerSeqToSV` and `HWMemSimImpl`
CIRCT/HW definition catalog and direct instance graph rooted at the exact
HardwareImplementation top. It assigns one CIRCT `OutputFileAttr` to every
concrete definition and emits the single semantic `RtlSource` through the
ordinary streaming exporter. CIRCT's own output-file framing gives every
module an exact byte range and digest inside that source. The publisher
requires the framed ranges, preamble, and framing bytes to cover the complete
source digest and byte count, then independently reprojects the post-export HW
graph and requires the module, port, parameter, dependency-multiplicity, and
reachability facts to agree.

Bundle preparation cold-rebuilds that same portable implementation and accepts
the transient graph only when the complete HardwareImplementation and
`RtlSource` identity are unchanged. It then validates every recorded range
against the exact stored source before materializing deterministic
`<module>.sv` library members. Text inspection may locate the header terminator
inside one already CIRCT-delimited and name-checked definition solely to add a
Verilator block metacomment; it cannot discover modules, dependencies,
multiplicity, reachability, or source closure. Missing, overlapping, foreign,
or digest-mismatched ranges fail closed.

Block selection uses unique transitive module-DAG weight together with exact
root-instance multiplicity. The HardwareImplementation root, large memory and
Temporal-PE definitions, and sufficiently reused FIFO or switch definitions
form bounded coarse blocks; small low-reuse definitions remain in their exact
parent closure. The HardwareImplementation root is distinct from the generated
Verilator testbench top and may be a hierarchical block; the testbench top is
never annotated. The testbench is the only explicit SystemVerilog input, and
the Hardware root is resolved lazily through the derived `-y` library. The plan
records the complete CIRCT dependency DAG, per-module source and derived bytes
and digests, selected blocks, exact unmarked block closures, paths, and policy
parameters. The plan is published as `loom.mapped_rtl_hierarchy_plan.2`.
These generated files and the plan are ordinary manifest-digested bundle
inputs, not another HardwareImplementation Artifact.

For Verilator 5.050, the first frozen command owns hierarchy planning and all
child/root Verilation; it does not use `--build` and no later command repeats a
`hier_verilation` target. Its `-j` value is the Verilation job count and the
job count of the make that Verilator runs for hierarchical Verilation, and the
same value is the make `-j` of the generated C++ build. The simulation model
thread count is a separate option emitted once as both `--threads` and
`--hierarchical-threads`, so the generated main, the root model, and the
hierarchical schedule agree. Both counts use the closed domain 1, 2, 4, or 8.
Verilator propagates every explicit SystemVerilog input of the planning
command into each child argument file, and the testbench is the only explicit
input, so each child would otherwise elaborate the complete design through
the harness. The generated hierarchy makefile therefore launches Verilator
through the hierarchy launcher, a Loom-built auxiliary tool frozen by path and
digest in the manifest's typed auxiliary-tool domain and configured through
make command-line variables that name the frozen Verilator executable and the
harness path. For a child argument file the launcher publishes an immutable
filtered sibling beside the Verilator-generated file, never edits that file,
requires the harness token exactly once and otherwise exits 42, records the
input and output digests on the Verilation command's captured error stream,
and executes Verilator on the sibling; root argument files pass through
unchanged. It never reads SystemVerilog. After that barrier, one typed
auxiliary build-tool command runs `hier_build` from the generated Mdir using
the makefile basename. A final tool-produced executable command owns
simulation. The exact make, C++ compiler, linker, archiver, and hierarchy
launcher are frozen in the manifest's typed auxiliary-tool domain and passed
explicitly to the generated build. This compilation plan is operational
evidence, not a claim that a particular large design meets its wall-time or
memory budget.

Synopsys VCS is the second member of the mapped-RTL simulator set. Its bundle
materializes the same semantic inputs and the same generated harness, one VCS
argument file, and two frozen commands: the compile, whose executable is the
catalog-frozen VCS launcher followed by the mandatory `-full64` token and the
argument file, and the simulation, which runs the tool-produced simulator from
its work directory. The argument file selects SystemVerilog, applies the
harness's own femtosecond timescale to every module so the clock periods are
exact, names the harness top, the parallel compilation count from the same
closed job domain, the compile work directory, and the simulator output, and
lists the exact semantic RTL source before the harness. VCS elaborates the
semantic source directly: no hierarchy plan, derived library, block
metacomment, or auxiliary build tool exists for this member, because VCS
compiles and links the simulator itself. The generated harness is legal for
every member: a variable that the initialization block and a clocked process
both write is driven by a general clocked process, not an `always_ff`, which
SystemVerilog forbids to share drivers and which VCS rejects.

Cadence Xcelium is the third member of the mapped-RTL simulator set. Its
bundle materializes the same semantic inputs and the same generated harness,
one xrun argument file, and two frozen commands: the elaboration, whose
executable is the catalog-frozen xrun launcher followed by the mandatory
`-64bit` token, the elaborate-only mode, and the argument file, and the
simulation, which runs the last elaborated snapshot of the bundle's library
directory through the same launcher with the same `-64bit` token. The
argument file selects SystemVerilog, applies the harness's femtosecond
timescale to every module, names the harness top and the snapshot library
directory one level below the bundle's `work/` root, turns off the launcher's
log, key, ordinary-history, and environment-history files, and lists the
exact semantic RTL source before the harness; the simulation command repeats
the library directory and suppressions because the snapshot run reads no
argument file. Xcelium elaborates the semantic source directly into a snapshot
library, not a program, so this member lists no tool-produced executable, admits the cycle
limit as its only provider option, and elaborates and simulates
single-threaded. The xmvlog, xmelab, and xmsim executables beside the frozen
xrun launcher are fingerprinted as typed auxiliary tools; changing a snapshot's
compiler, elaborator, or simulator invalidates ordinary invocation reuse. The
two event-driven members share one bundle projection:
the harness, the argument file, the compile command, the tool-produced
executables of that command, and the simulation command.

A plan that selects no block is a distinct Verilation style rather than a
degenerate hierarchical plan. Flat Verilation annotates no module, emits
neither `--hierarchical` nor `--hierarchical-threads`, carries no hierarchy
launcher configuration because no child argument file exists, and builds the
ordinary `V<top>.mk` whose target is the simulator executable; hierarchical
Verilation keeps `V<top>_hier.mk` and `hier_build`. The plan records its style
next to the makefile it names, and the same three commands - Verilation, build,
simulation - describe both styles.

A hierarchical block is opaque to its parent's scheduler: Verilator treats
every block output as combinationally dependent on every block input, so
ready/valid coupling across a block boundary reports `UNOPTFLAT` circular
logic at the parent even where the flat design has no cycle. Such diagnostics
are classified against the flat module before any performance conclusion is
drawn from them, and neither class is masked.

The mapped-RTL operating point is frozen from measurement, and the numbers
below are operational evidence for one design on one host, never a guarantee.
On the product Matmul bundle - 993 library modules, 49.56 MB of SystemVerilog,
one 10,784-word configuration image - an A/B over the compiler level, the
model thread count, the Verilation style, the root-closure byte budget, and
the shape of the configuration driver selected: hierarchical Verilation with a
four-megabyte root-closure byte budget, which selects 128 blocks and leaves a
69-module root closure; eight model threads and eight build jobs; Verilator's
own `-Os` fast objects and no object cache; and the clocked configuration
driver described with the harness. Through the canonical driver that
configuration Verilates in 80.0 s, builds and links in 139.4 s, and simulates
55,000 testbench cycles in 127.0 s - a 346.3 s cell at 8.2 GiB with no swap,
against 984.5 s for the same bundle before the A/B. The observations that
generalize beyond this design are structural: more and smaller hierarchical
blocks reduce the single-threaded root Verilation tail (25.5 s at 128 blocks
against 113 s at 64) and also reduce simulation cost, because a smaller opaque
block re-evaluates less logic per input change; flat Verilation of a design
this size is not viable at all, because one generated translation unit reaches
hundreds of megabytes and exceeds the C++ compiler; eight model threads
simulate three times faster than one; the delay-free clocked configuration
driver removes one evaluation slot per handshake and one full clock cycle per
configuration write; and the C++ optimization level buys under ten percent of
simulation rate for two and a half times the build. Host memory placement
dominates all of these: confining the model threads to a larger last-level
cache complex measured 1.9 times the rate of the smaller complex on the same
executable, which is a property of the host, not of the bundle, and is
therefore never expressed in a frozen command.

A protocol checker generated from the same provider may verify interface,
reset, handshake, and ABI invariants. It is not an independent functional
oracle for the logic generated by that provider. Workload correctness is
compared against an independently produced SimulationExecution, such as DFG or
CGRA execution, through the ordinary comparison Evaluator.

## Determinism

Identical exact Fabric, SpatialCore occurrence subject, ConfigurationABI,
ImplementationPlatform, resolved candidate-generator binding, and producer
identity must yield byte-identical
canonical HardwareImplementation content. Workload-selected semantic
configuration is not an input to HardwareImplementation generation. Emitted
labels derive deterministically from canonical structural references but remain
presentation details.

## Anchor Verification

Stable anchors cover:

* one Fabric lowering to a target-independent verified HW/Comb/Seq skeleton,
  with abstract leaves tied only to exact Fabric occurrences;
* one System importing the same Module twice with independently bound
  Clock/Reset slots, producing two occurrence-scoped implementations with
  distinct subjects and no cross-occurrence interface ownership;
* the same skeleton specialized to portable and vendor-backed RTL without
  rebuilding structural topology;
* rejection of a skeleton that reaches SystemVerilog export with an unresolved
  Loom abstract leaf;
* acceptance of a target-specific external module only with an exact black-box
  contract and external implementation binding;
* one regular and one arbitrary-topology Fabric lowering;
* exact replication/arbitration and width/tag behavior;
* fixed-vector slice/merge and shuffle providers at one non-power-of-two
  physical payload width, including lane-local poison refinement;
* one narrow and one wide custom Fabric proving that operation and memory
  providers contain no 128-bit datapath default;
* atomic two-input `s2t` and two-output `t2s` handshakes with no partial
  transfer or hidden holding;
* parallelize partial-close and sparse serialize providers using the exact
  common outer-claim materialization, including per-group backpressure,
  zero-output commit, reset during drain, final-production sampling with group
  capture, and release-before-acquire replacement;
* dispatch of one operation schema through two implementation families and
  typed rejection of a missing provider;
* correctly-rounded, one-ULP, two-ULP, and four-ULP special-math admission,
  independent numerical-oracle checks, and typed rejection when a recipe cannot
  prove its selected guarantee;
* exact `None`, finite, and direct provider decoding from the shared Fabric
  relation and ConfigurationABI, with no provider-local key ordering;
* temporal context and memory-operation behavior;
* Temporal PE `per_instruction` depths 1 and 2 with the exact derived queue,
  service, and round-robin resource contract;
* vector element, contiguous, indexed, and masked memory operation lowering,
  including one declared narrower-beat realization and one logical retirement;
* Spatial and Temporal element-only, vector-only, and shared-hybrid memory
  ports, including distinct per-role Temporal tags and zero-payload control
  channels;
* clock/reset domain and self-reset closure;
* SDC derivation from one explicit clock, one declared crossing, and one reset
  release contract, with a backend-invented exception rejected;
* ConfigurationABI programming through one mapped workload;
* an architecture harness and a mapped harness using exact implementation
  interfaces, with same-provider golden logic rejected as an independent
  functional oracle;
* deterministic scalar activity-point identity and rejection of provisional
  hierarchy-name identity; and
* rejection of an unsupported or behavior-changing hidden refinement,
  including implicit input draining.

Tests do not preserve whole RTL text, vendor command lines, hierarchy names, or
per-family exhaustive matrices. Syntax, elaboration, formal, simulation, and
physical observations are ordinary Evaluations of the finalized implementation.
