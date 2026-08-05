# Fabric To RTL

This document defines the hardware-generation boundary from one exact Fabric
Hardware Description to an immutable RTL `HardwareImplementation`.

## Input And Output

Fabric-to-RTL consumes:

* one fully elaborated exact `FabricRootKind::System` Hardware Description and
  its exact imported Module dependency closure;
* one exact `ConfigurationABI` for that Fabric;
* exact Interconnect Implementation roots required by that Fabric system;
* one resolved hardware candidate-generator binding;
* one exact `ImplementationPlatform` when the emitted implementation is bound
  to an ASIC technology release or FPGA ordering code; and
* exact provider-owned external input bindings required by selected Fabric
  resources or implementation recipes.

It produces one `loom.hardware_implementation 2.0` whose closed `Rtl`
representation root owns the exact top Module locator and content-addressed
SystemVerilog source closure. The implementation also owns the interfaces,
constraints, black-box contracts, activity-point catalog, and implementation
manifest needed by downstream tools.

Only a System owns the concrete occurrence, Clock/Reset, external-interface,
and Transport Architecture closure required by an export-complete RTL design.
A Module may lower only as an internal slot-parameterized hierarchy fragment
within that exact System closure. A Module alone cannot publish an
export-complete SystemVerilog design, `ConfigurationABI`, or RTL
`HardwareImplementation`.

`docs/spec-hardware-implementation.md` owns the output root, payload roles,
interface and activity catalogs, semantic closure, and finalization.
`docs/spec-implementation-platform.md` owns only the shared ASIC or FPGA target
manifest and technology-corner catalog. Provider descriptors own the exact
external files or tool-bundled resources they consume.

The lowering does not consume Dataflow or Mapping and does not create a
workload-specific RTL design. Workload execution combines the reusable
`HardwareImplementation` with an exact `Deployment`, configuration images, and
runtime inputs.

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
* Transport Architecture and exact Interconnect Implementation refinement; and
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
relation is not itself a protocol channel. Only the exact Interconnect
Implementation may refine the selected path into AXI, TileLink, CXL, custom
request or response subchannels, packet or flit fields, adapters, RTL/IP, and
physical encodings. Lowering cannot infer or persist a second pairing between
memory and transport endpoints.

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

The skeleton has one System-rooted top hierarchy. Every SpatialCore occurrence
contributes one Module specialization request. Requests share one reusable
definition exactly when this derived key is equal:

```text
ModuleSpecializationKey =
  (exact ImportedModule root reference,
   exact definition-rebased occurrence projection of ConfigurationABI,
   exact Module-slot-keyed Clock/Reset contract projection,
   exact definition-rebased candidate-generator projection for every internal
     recipe, external implementation, and memory implementation choice)
```

Definition rebasing removes only physical occurrence identity. It maps every
`SpatialCoreInternalOccurrenceRef` back to its exact Module-local target, maps
each occurrence-slot binding to its Module slot and concrete contract, and
projects each occurrence-scoped recipe, external implementation, or memory
implementation selection to the semantic choice applied to that local target.
The projections are ordered by their definition-local typed references. They
contain no `SpatialCoreOccurrenceRef`, `AccCoreOccurrenceRef`, System
`HardwareDomainRef`, global binding ordinal, or caller-authored name. A global
external or memory binding is therefore represented by its exact selected
contract and dependency closure after rebasing, not by the ordinal assigned in
one complete HardwareImplementation.

Rebasing does not discard a definition-affecting value. Different physical
occurrences with byte-identical rebased projections share one definition;
different ABI field schemas, codebooks, slices, inactive encodings,
Clock/Reset contracts, recipes, external dependency closures, or memory
choices produce different keys even when the imported Module root is equal.
Workload-selected semantic values are not specialization inputs. The System
instance retains the original occurrence-qualified relations and concrete
domain references used to bind that shared definition.

The key is a removable derived compiler value, not a persistent record or a
caller-authored cache key. It contains no local executable path or attempt
state. Each distinct key lowers to one definition with one explicit Clock or
Reset port per symbolic Module slot; every member occurrence instantiates that
definition and binds its ports from the exact System occurrence-slot
memberships. A different occurrence identity alone does not force a duplicate
definition, but any differing decoder structure, reset behavior, provider
recipe, external implementation, or memory implementation does. Internal
state, configuration, recipe, memory, and activity relations remain
occurrence-qualified; definition sharing never merges the physical identity of
two instances.

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
visibility, bypass backpressure, or inactive-state semantics.

An unbound or inactive operation input is not an implicit sink. A provider
must not assert readiness merely to drain and discard tokens unless the exact
Fabric capability explicitly defines that consumption and backpressure
behavior. FU-local selection remains the explicit `fabric.mux` and
`fabric.demux` topology owned by Fabric.

Boundary lowering implements the normative atomic ready/valid equations in
`docs/spec-fabric-boundary.md`. Two-input `s2t` cannot consume either input
alone, and split `t2s` cannot publish either output alone. The base boundary
has no register or holding state; adding one is a behavior-changing Fabric
refinement, not an RTL convenience. RTL lowering consumes only a Mapping that
has passed the selected combinational handshake closure in
`docs/spec-mapping-verification.md` and invokes that same derived gate when it
revalidates its inputs. It cannot union hardware alternatives, omit selected
arcs, or apply a backend-local loop-breaking rule.

Temporal-PE operand storage is emitted from the exact required
`operand_buffer_size` and mode-derived allocation units. The base contract has
one enqueue and one dequeue service per allocation unit per local cycle, with
the declared canonical round-robin policy where contention is possible. RTL
must not substitute a default depth, extra port, global arrival-order head, or
implementation-private priority.

## Clocks, Reset, And Quiescence

RTL exposes the exact Fabric clock/reset domains and only their declared
crossings. Stateful resources start in their canonical initial state and, for a
legal completed invocation, satisfy Fabric's self-reset/quiescence contract
before the same physical slot is reused.

The backend obtains these facts by refining the exact imported Fabric artifact
to `FabricSystemRootView` and consuming its complete System domain membership,
Module slot assignment, occurrence-qualified effective-domain, connection,
attachment, transport-resource, and crossing ranges. It does not accept a
backend-owned clock/reset manifest, caller-supplied connection list, inherited
AccCore domain, or copied crossing catalog. Derived RTL or SDC indexes are
disposable caches validated against that one root view.

Power, clock-gating, reset synchronization, and backend constraints are emitted
only from explicit Fabric implementation facts. An asynchronous clock crossing
lowers only from the exact `ClockCrossingContract` on its owning transport
resource. Reset synchronizers and release latency lower only from the exact
Reset domain contract. Missing implementation support is a typed failure, not
permission to omit required behavior or invent a crossing.

## Memory And System Interfaces

`fabric.mem` lowers its operation engine, internal dependency forwarding,
configurable service dispatch, optional local storage, and manager/subordinate
interfaces without adding storage semantics absent from Fabric. System
interconnect lowers the selected exact implementation protocol while preserving
the architecture service, multicast, ordering, capacity, and progress contract.

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
owner-attempt or scratch material until the raw detailed-bundle Artifact owner
is defined. An architecture-evaluation descriptor references a
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
private hierarchy guesses.

A protocol checker generated from the same provider may verify interface,
reset, handshake, and ABI invariants. It is not an independent functional
oracle for the logic generated by that provider. Workload correctness is
compared against an independently produced SimulationExecution, such as DFG or
CGRA execution, through the ordinary comparison Evaluator.

## Determinism

Identical exact Fabric, ConfigurationABI, ImplementationPlatform, resolved
candidate-generator binding, and producer identity must yield byte-identical
canonical HardwareImplementation content. Workload-selected semantic
configuration is not an input to HardwareImplementation generation. Emitted
labels derive deterministically from canonical structural references but remain
presentation details.

## Anchor Verification

Stable anchors cover:

* one Fabric lowering to a target-independent verified HW/Comb/Seq skeleton,
  with abstract leaves tied only to exact Fabric occurrences;
* one System importing the same Module twice with independently bound
  Clock/Reset slots, distinct internal occurrence identity, and one reusable
  slot-parameterized Module definition;
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
* dispatch of one operation schema through two implementation families and
  typed rejection of a missing provider;
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
