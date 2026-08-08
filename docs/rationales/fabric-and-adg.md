# Fabric And ADG Construction Rationale

Normative contracts are owned by
[ADG Builder](../spec-adg-builder.md),
[Fabric Artifact](../spec-fabric-artifact.md),
[Fabric Module](../spec-fabric-module.md),
[Fabric System ADG](../spec-fabric-system-adg.md), and
[Fabric Identity](../spec-fabric-identity.md).

## Why Fabric Is The Hardware Truth

The C++ ADG Builder exists to make hardware construction usable by architecture
designers. It is not a second hardware model. Custom builders and builtin
templates both construct real Fabric IR and invoke the same finalizer; every
downstream component consumes the resulting immutable Fabric artifact.

A string node/property graph followed by printing and reparsing was rejected.
It would require a parallel type system, capability registry, connection
validator, and identity model. A thin typed C++ facade gives users convenience
while preserving Fabric as the only semantic owner.

FU feedback is exposed as one generic typed backedge because cyclic dataflow is
an essential graph distinction used by recurrence hardware. A MAC-specific
feedback API would duplicate that topology rule in every helper, while an
implicit cycle would hide a real Fabric edge. Requiring explicit resolution
before FU closure keeps the public surface small and prevents authoring
placeholders from leaking into persistent hardware.

The same move-only placeholder is scoped at the SpatialCore boundary because
arbitrary hardware topology includes rings and torus-like cycles whose source
may be authored after its sink. Making cyclic module construction a special
topology helper would add a shadow connection model; resolving one typed
backedge into the ordinary Fabric SSA edge keeps cyclic and acyclic hardware
under the same semantic owner.

Builtin Small, Default, and Large targets are complete hardware examples, not
size knobs or capability summaries. Their FU distribution, memory, transport,
InstructionCore, clocks, resets, and semantic capabilities must be
deterministic so the same public API can reproduce and teach the exact
hardware. Expanding their Reset contract into ordinary Fabric facts prevents
the backend from becoming a second, target-dependent reset-policy owner.

The general builtin memory is broader than the initial Hybrid32 convenience
recipe because a preset-wide scalar type floor is meaningful only when both
compute and memory resources admit it. Reusing Hybrid32 made a 64-bit ALU
result computable but impossible to store. A second persistent memory kind was
rejected: both recipes construct the same canonical Fabric memory contracts,
and differ only in the exact typed access domain authored into those records.

Memory recipe inputs distinguish accepted RootRelative index widths from the
physical indexed-address endpoint width. Deriving the former by dividing the
latter by 32 made one catalog convention an undeclared semantic default and
could not represent correlated 32-bit and 64-bit lane limits. The helper now
authors those correlations through the existing reduced-product rows, after
which the generated Fabric relation and endpoint types are the only authority.
PointerAddressed rows continue to use exact pointer formats, so no helper-wide
index setting leaks into pointer admission.

Builtin extension is split at the same publication boundary as custom
hardware. An open SpatialCore recipe is extended and finalized first; only its
durably published Module can then be imported by an open System recipe. A
monolithic mutable target object would either hide this exact dependency or
become a second hardware model beside Fabric.

A switch projects one resource pattern per physical traversal, not one pattern
per configured broadcast subset. The latter is an exponential restatement of
the route table. A spatial switch uses one configuration requester because
Mapping has already selected and capacity-closed the whole static route image;
inventing per-input runtime arbitration would add hardware behavior that does
not exist. Temporal patterns instead use input-owned requesters, and the
existing event-derived atomic activation set joins the selected egresses. This
keeps physical capability linear in connectivity while retaining all-or-nothing
broadcast and exact temporal fan-in arbitration.

The initial fixed-vector families are separate from scalar families even when
they use the same software operation schema. Shape is a physical organization
fact, while the canonical actor type remains the semantic owner. Adapter and
token-control actors use singleton families because co-location does not prove
shared circuitry. Special math is also singleton except where quotient and
remainder are genuine projections of one signedness-specific divider. This
larger generated family vocabulary is preferable to a smaller but false
hardware-sharing claim.

Fabric publication proves semantic hardware closure, not tool availability.
Requiring every builtin to have an RTL or EDA provider would make an external
backend installation alter whether the same hardware description exists.
Provider closure is therefore checked by the requested realization stage and
reported as typed `Unsupported` without changing Fabric identity.

## Why Width Is Not A Fabric-Root Property

A heterogeneous SpatialCore may combine a narrow scalar network, wider vector
units, a still wider memory endpoint, and a narrower service beat. One root
`datapath_width` would either reject that architecture or duplicate the exact
width already owned by every physical endpoint. It would also encourage
Mapping and backends to infer compatibility from a global number rather than
the selected path and capability relation.

ADG catalog helpers therefore receive transient typed width parameters and
emit ordinary Fabric endpoint types. The parameters disappear after
construction. Small, Default, and Large may intentionally pass the same
128-bit ordinary-payload policy, but that is one builtin recipe choice rather
than an ADG Builder default or Loom limit. Custom helpers can compose different
widths without introducing another persistent hardware model.

Keeping the policy transient also preserves the purpose of the builtins: they
are reproducible public examples, not a hidden global architecture profile.
The finalized Fabric remains sufficient for Mapping, simulation, RTL, and EDA;
none of those consumers needs the helper inputs that produced it.

## Why Module And System Are Separate Fabric Roots

`fabric.module` is a reusable SpatialCore template. `fabric.system` is the
architecture-level multi-AccCore system that attaches module occurrences to
InstructionCores, memory/services, domains, external boundaries, and Transport
Architecture. Keeping both in Fabric preserves one hardware dialect while
separating reusable local structure from physical system occurrences.

Interconnect Implementation is a sibling refinement rather than part of
SystemMapping or the architecture root. Bandwidth, latency, ordering,
coherence, resources, and externally visible grant guarantees are architecture
facts. AXI, TileLink, CXL, packet formats, arbitration microstate, and protocol
components are implementation facts. Mapping selects use of the architecture;
it does not choose a hidden protocol interpretation.

## Why Module Domains Are Symbolic And System Domains Are Concrete

A reusable Module can appear several times in one System and can eventually
contain more than one clocked island. Giving it concrete clock periods or Reset
release policy would make those System facts part of a reusable definition.
Conversely, assigning one domain to an entire SpatialCore or AccCore would
create implicit inheritance, erase independently typed InstructionCore
semantics, and require exceptions as soon as a Module has two domains.

Symbolic Module slots express exactly the missing reusable fact: which
boundaries and physical owners occupy the same clock or Reset topology.
Association is total even for combinational owners so an ordinary connection
never needs to infer a domain through neighboring state. Whether an owner
actually consumes Clock and Reset signals is the separate, mechanically
derived ResourceState fact. System domain membership then supplies the concrete
contract for each physical occurrence. One slot relation replaces both a
repeated per-resource System table and an implicit connectivity inference while
remaining explicit enough to validate multi-clock reuse and to derive RTL ports
and constraints mechanically.

The omitted authoring form is a canonical shorthand for the common one-Clock,
one-Reset Module, not a second source of domain semantics. Finalization expands
it once into the same explicit total relation that a caller could author. Any
explicit domain row disables the shorthand, so multi-domain Modules and partial
relations cannot acquire inferred assignments. Requiring every simple authoring
fixture to restate the total single-domain relation would add repetition without
expressing another hardware distinction; making the choice configurable would
create a competing policy owner.

The shorthand also removes an otherwise unsupported slotless Module category:
every finalized Module has at least one Clock and one Reset slot. It does not
erase the separate fact owned by a Module instance edge. That edge still binds
every effective child slot explicitly, and a closed Builder exposes those slot
handles without copying their meaning. Keeping the edge explicit prevents
containment from becoming domain inheritance and allows one child definition to
be reused under different parent correspondences.

An SSA `SpatialValue` belongs only to the connectivity plane; making it also
identify a physical owner would conflate two facts and fail for boundary faces
or resources with several owners, such as a memory occurrence with operation
ports and a local service. The Module relation already has one closed
`Boundary | Internal` member union, so one owner-checked
`ModuleDomainMemberHandle` is its smallest faithful authoring projection.
Separate boundary and internal assignment APIs would duplicate the same
relation, while adding Clock and Reset parameters to every resource constructor
would couple orthogonal topology choices and proliferate PE, FU-node, memory,
and instruction-context exceptions. The resource-construction call is the
single public authoring boundary that can expose the owners it creates, so its
role-specific accessors derive the single unified handle mechanically from
those Fabric draft entities rather than owning or storing another inventory.
Finalization discards the handles and persists only the existing
`ModuleDomainAssignment` relation.

When a Module instantiates another Module, the instance edge is the only place
that knows both symbolic slot contexts. Giving that edge one explicit total
child-to-parent correspondence records the essential relationship without
turning containment, names, ordinals, or connectivity into hidden domain
semantics. Several child slots may intentionally converge on one parent slot;
the reverse relation is not required. Keeping the correspondence authoring-only
lets elaboration rewrite child assignments directly into the enclosing flat
Module and then delete the instance. Persisting a nested instance or expanded
domain table would duplicate the final assignments and break equivalence with
inline authoring. The binding cites the identity-owned slot ordinal domain
rather than restating a narrower width, so no hidden narrowing can appear
between the authoring record and the persistent relation, and an inlined and
an instantiated Module keep one assignment semantics.

Imported Module identifiers remain definition-local because cloning or
renumbering them would make physical identity depend on how many times a
template is used. Qualifying an exact Module-local target by its SpatialCore
occurrence adds only the essential physical distinction. It separates state,
configuration, recipes, memory bindings, and activity without copying the
Module or creating a generic hierarchical path.

Crossings stay explicit because hierarchy cannot prove synchronization or
buffering behavior. An asynchronous carrier also needs exact source and
destination Reset authorities; otherwise "released on both sides" has no
meaning. Until a Module-local crossing resource owns that behavior, rejecting a
cross-slot connection is smaller and safer than inserting a hidden
synchronizer. The same reasoning excludes AccCore-wide Clock/Reset defaults.

## Why Topology Is Explicit And Fully Elaborated

SpatialCores and systems may have arbitrary directed topology. Module-level
transport is therefore explicit endpoint-to-endpoint connectivity. Fanout and
fan-in require resources with the corresponding hardware behavior; SSA reuse
cannot create a free module-level broadcast. Width conversion within one port
kind follows the declared low-bit convention, while a port-kind change requires
an explicit boundary resource.

PnR must see a fixed hardware problem. Templates and instantiations are fully
expanded before finalization; search cannot add resources, instantiate modules,
or mutate topology. Hardware DSE produces another immutable Fabric candidate
outside PnR.

Coordinates and regular grids remain optional authoring and visualization
metadata. They cannot affect Fabric identity or route legality. This prevents
Manhattan-distance assumptions from entering an arbitrary-topology mapper.

## Why Persistent References Are Owner-Relative

Independent physical occurrences such as PEs, FUs, memories, switches,
boundaries, AccCores, and transport resources need artifact-local identity.
Ports, FU-internal nodes, traversals, resource states, use patterns, and memory
regions are recoverable from those owners and use closed structural references.

A universal `{kind, path, ordinal}` was rejected because it requires a generic
path interpreter and string kind registry. Giving every leaf an EntityId was
also rejected because unrelated local changes would churn identities across
Mapping and Deployment. Typed reference domains preserve the distinction
between transport endpoints, memory endpoints, resource state, and directed
traversal.

Builder handles, symbols, source paths, and printer order are authoring aids.
They never cross the finalization boundary as identity.

## Why System Service Ports Are Explicit Endpoint Entities

A System operation-service port has independent physical meaning: it has one
plane, one direction or memory role, one capability set, and for message
transfer one exact physical carrier. Making every host core, AccCore, memory
service, transform, and external boundary own a separate endpoint inventory
would duplicate the same schema and let the logical owner become a second
physical-interface authority. A generic shared inventory record would merely
move that duplication into an untyped bag.

Fabric therefore uses one `fabric.system.service_endpoint` entity for one
physical operation-service port. The surrounding System entity owns that
endpoint by an exact typed reference, while the endpoint alone owns its
physical interface facts. Multiple ports are multiple entities, so the
endpoint's selected transport or memory inventory contains only ordinal zero.
This removes an unnecessary ordinal namespace and makes connection, domain,
Mapping, and backend references converge on one persistent object.

`SystemMemoryService` still owns storage regions and service behavior;
`SystemServiceTransform` still owns its transformation relation; an external
boundary still groups the hardware interface. Those distinctions are
essential, but none requires another endpoint schema. Their outward ports are
mechanically the service-endpoint entities that name them as owner.

SpatialCore module-boundary ports and System transport-resource ports remain
direct inventories because their ordered port topology is intrinsic to those
resources. They are not operation-service endpoints and collapsing them into
service-endpoint entities would erase a real structural distinction.

## Why A Memory Spatial Attachment Names Its System Service Endpoint

A Module memory boundary and its occurrence-qualified SpatialCore memory
endpoint describe only the two faces of one imported Module. They do not say
which System service endpoint continues that memory path. Capability matching
cannot recover the missing topology: two endpoints may intentionally expose
the same capability while connecting to different services, transforms, or
physical networks.

The memory variant of the existing spatial attachment therefore names the
exact System service endpoint. This adds the one missing fact to the relation
that already owns the Module-to-occurrence correspondence. A second attachment
table would split one physical relation across two owners and require another
coverage proof. Letting Mapping choose among matching endpoints would instead
make a software-placement artifact define hardware wiring. Transport
attachments need no third endpoint and retain their two-face form.

The Module boundary owns its exact memref type and endpoint-relative role, but
it does not own a workload capability-domain copy. The complete read, write,
atomic, compare-exchange, or fence requirement and exact attachment row appear
only after an exact Dataflow member, selected Spatial semantics, and AccCore
occurrence are known. The Spatial semantics may come from an immutable
SpatialMapping or a mutable flat candidate; neither changes the Fabric-owned
attachment. Fabric root finalization therefore validates the three-reference
topology, memory plane, and complementary roles. SystemMapping domain
construction and verification test the selected member against the capability
set of the mechanically bound endpoint; failure makes only that binding
infeasible and never causes a search for another endpoint.

Deriving or persisting a Module-boundary capability catalog would require a
new rule for combining every configurable internal use and would compete with
the endpoint and selected Dataflow member as capability authorities. Deferring
the workload-dependent comparison keeps both facts at their existing owners
without weakening the exact hardware binding.

The old 2.x payload cannot be upgraded by selecting the only endpoint visible
on one host: uniqueness is ambient state, not persisted hardware semantics.
The closed relation consequently enters through a Fabric major-version
boundary and fails closed rather than preserving an inference path.

## Why Memory-Service Legs Need An Explicit Carrier Relation

A memory endpoint and a transport endpoint are deliberately different planes.
Making a memory endpoint double as a token carrier would merge operation
admission, address and consistency semantics with routable direction and
payload transport. Inferring a carrier from a common owner, matching ordinal,
or equal width would instead create an unwritten topology rule that fails as
soon as one service has several ports or one carrier is shared.

Fabric therefore owns one structural relation from the existing memory
endpoint, service kind, and schema-local leg to a non-empty set of existing
transport endpoints. This is the minimum additional fact needed to expose a
service leg to system routing. Both members of a memory spatial-attachment
pair use this same relation rather than separate occurrence-side and
service-side relation kinds. The Canonical Service Schema still owns what the
leg means, the attachment's exact System endpoint capability still owns what
operations the pair admits, each pair member's row owns only its carrier set,
and the Transport Architecture still owns how messages traverse and contend.

The occurrence endpoint does not acquire a capability copy. Its unique memory
spatial attachment supplies the one System capability authority, and its
Module-derived manager or subordinate role supplies the complementary terminal
direction. This lets a request source and response sink use the exact
occurrence endpoint while the request sink and response source use the exact
service endpoint. Adding a second carrier-relation schema or promoting the
occurrence endpoint into another service entity would duplicate an existing
key or capability owner without introducing a new hardware distinction.

No capability ordinal is needed because one System endpoint has at most one
capability for a given kind and role, and an occurrence endpoint has exactly
one memory spatial attachment. No payload, width, protocol, or workload
identity is stored because each is already derivable from an existing owner.
Allowing one carrier in several attachment rows preserves real shared
hardware; its capacity and arbitration remain visible through transfer
patterns and resource use rather than being duplicated in the relation.

The required carrier width is the maximum width of the independently
transported values in the canonical leg, not the sum of their widths. Summing
would silently invent a packed tuple, field offsets, and an interface layout;
checking only one selected value would fail to cover the capability domain.
The maximum is the smallest single routing bound that proves every value fits
the shared route while leaving transactions, beats, packets, and physical
serialization with their existing owners. Thus an address64, data128, mask16,
control0 write request requires a 128-bit carrier envelope rather than a
208-bit fabricated tuple.

## Why Fabric Finalization Is Root-Complete

Canonical identity must reflect the complete elaborated hardware, including
connections, domains, crossings, capabilities, and dependencies. A public API
that freezes an arbitrary partial root would let callers bypass completeness
or manufacture a view that later validators trust.

The finalizer therefore validates private authoring state before identity,
canonicalizes the complete root, publishes through Common, and exposes only a
sealed imported view that can be independently reverified. Clock/reset
validation derives all membership and connections from that same root; a
shadow connection vector was rejected after it allowed omitted crossings to
pass and duplicated edges to fail.

Fabric dependencies are independently published artifacts. Root publication
does not create a transaction over them. Every dependency is resolved and
strictly imported before the root's one-object commit.

## Why System Connections Cover Both Planes

A service transform receives requests as a subordinate and emits transformed
requests as a manager. Composing two transforms therefore requires an explicit
identity edge from the first manager output to the second subordinate input;
reaching an ordinary memory service requires the same edge. Capability
equivalence cannot supply it because that would let Mapping or a backend invent
hardware topology.

Adding a separate memory-connection operation was rejected because transport
and memory connections share the same essential directed one-to-one relation,
canonical ownership, and no-hidden-behavior rule. An identity transform was
also rejected: it would give a wire an unnecessary entity and behavior
contract. Fabric 3.0 instead closes the existing connection operation with a
`MemoryService` variant alongside `Transport`. The endpoints' plane and roles
choose the variant's legal relation, so there is no generic edge or
caller-authored kind flag.

This closes an internal 3.0 omission rather than extending its semantic
language. The existing schema already required operation-service references in
connections and represented identity transforms as direct connections, while
the detailed variant and direction were missing. Transport keeps its existing
variant-zero meaning; no previously valid record is reinterpreted.

## Why Coherence Correspondence Is Region-Relative

`CoherentMemory` must say which physical regions are copies or proxies of one
another, not merely that their services participate in one consistency domain.
The existing pair of Fabric service-region references already owns the two
absolute ranges. Requiring equal nonzero extents makes their relative-offset
mapping exact without adding an offset field or a second address-transform
language.

Adding input and output endpoint ordinals to every correspondence was rejected
because the transform's ordered endpoint relation and explicit MemoryService
connections already own topology. Restricting the whole transform to one input
and one output was also rejected because a real coherent fabric can expose
several ingress ports and provider alternatives. A canonical partial bijection
over region references is the smaller complete rule: input and output regions
remain unique, all input endpoints share that relation, and an output is usable
only when its explicit closure reaches the paired output region. Mapping then
selects an exact cover of the source-address domain rather than copying either
endpoint ordinals or coherence payload.

For example, a proxy region `[0x1000, 0x2000)` paired with a backing region
`[0x8000, 0x9000)` maps source interval `[0x1400, 0x1500)` to
`[0x8400, 0x8500)`. The difference is derived from the two Fabric-owned bases;
there is no stored `0x7000` delta. If an explicit transform output cannot reach
the backing region, that pair is not a target candidate. If another overlapping
input region leads to a different coherent output, the two output-region
branches are alternatives and one selected plan must still cover every source
address exactly once.

## Why Handshake Owners Use Private Junctions

Ready/valid behavior belongs to the concrete Fabric resource. A consumer-owned
arc table would duplicate that behavior, while forcing every internal Boolean
term to become a transport endpoint would give implementation detail persistent
identity and make it accidentally routable.

The sealed owner model keeps one semantic compiler beside the normative
resource equations. Exact Mapping selections activate typed fragments of that
model; unconditional Fabric validation universally quantifies the same local
fragments. Both gates therefore consume one owner rather than independently
reconstructing behavior.

Atomic broadcast exposes why a compact internal graph is necessary. With `N`
selected sinks, expanding every peer-ready dependency at the boundary is
quadratic. Canonical prefix and suffix conjunction nodes represent the same
dependency reachability with linear storage and constant-size change for one
selected sink. Those nodes are removable derived structure, so they receive no
Fabric entity, persistent reference, route capacity, or configuration field.
The semantic contract remains atomic replication; the compact graph is only
its efficient projection.

## Why One Dependency Role Is Reserved But Unavailable

An ordinal alone does not define the owner, schema, root kind, local target,
or use of an implementation input. Reusing HardwareImplementation would create
a dependency cycle, while broadening ImplementationPlatform would make it a
generic IP container. Fabric 3.0 therefore recognizes the stable wire ordinal
but rejects authoring and import before any store lookup.

This fail-closed reservation preserves format diagnosis without pretending
that an owner contract exists. A future owner must be introduced explicitly;
it cannot be inferred by a consumer.

## Why Hardware State Must Close Cleanly

Graph invocations can overlap and time-share resources only when Mapping proves
non-conflicting resource-time use. Stateful hardware must nevertheless reach
its canonical closed state before the owning invocation retires. Passing a
reset token on every launch would expose implementation detail in software;
ignoring close state would make reentrancy and pipelining unsound.

The closure rule is derived from each resource's explicit contract and exact
clock/reset domains rather than a global hidden reset policy.

## Why Instruction Architecture And Microarchitecture Are Separate

The InstructionCore is the stored-program fallback inside an AccCore, but it
is not necessarily scalar or simple. A small in-order RISC-V core, an
out-of-order core, and an RVV core can all occupy that role. Calling it a
ScalarCore would therefore encode an accidental implementation choice into the
system model.

Binary compatibility and execution behavior are different truths. The ISA,
ABI, privilege, memory-ordering, and relocation envelope decides whether a
program can execute. Pipeline widths, queues, execution-unit timing, and
thread capacity decide how it executes. Combining them would force every
microarchitecture change to create an unrelated compiler target; omitting the
microarchitecture would force Mapping, Evaluation, and gem5 bindings to invent
their own hardware description.

The closed `RiscV` architecture variant and the closed `InOrder | OutOfOrder`
realization keep those truths explicit without introducing a generic CPU
property bag. Representative profiles are ordinary values of the same schema,
not new variants. Provider names remain bindings because they identify a tool
implementation, not the hardware contract itself.
