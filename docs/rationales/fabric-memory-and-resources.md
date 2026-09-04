# Fabric Memory And Resource Rationale

Normative contracts are owned by
[Fabric Memory](../spec-fabric-mem.md),
[Fabric Resource Contract](../spec-fabric-resource-contract.md),
[Fabric Boundary](../spec-fabric-boundary.md),
[Fabric FIFO](../spec-fabric-fifo.md),
[Fabric Switch](../spec-fabric-switch.md), and
[Fabric Instantiate](../spec-fabric-instantiate.md).

## Why `fabric.mem` Is One Composed Resource

Spatial versus temporal execution and scalar versus vector access are
orthogonal capability axes, not six operation names. A single memory resource
can contain an optional operation engine, an optional local storage service,
manager/subordinate endpoints, operation ports, and explicit internal
connectivity. Splitting these into many ops would duplicate storage identity,
service capacity, consistency, and configuration.

The operation engine's spatial/temporal mode describes how accesses are
scheduled. It does not classify the outer memory object or local service.
Scalar, vector, gather, scatter, atomic, and other access forms are admitted by
typed operation rows and geometry rather than exact-shape duplication.

One physical memory can support multiple logical memories, and one logical
memory can be realized through several services, only when each Mapping proves
width, access, throughput, consistency, lifetime, and routing constraints.
Equal payload width is not enough: element width, lane geometry, address form,
mask, atomic granularity, and inactive-lane semantics remain typed facts.
Conversely, a wide physical data endpoint need not force exact-width scalar
software accesses. A memory advertises each admitted scalar element width and
the physical zero-extension or byte-enable guarantee explicitly. This keeps
ordinary low-bit-aligned transport adaptation and memory-specific subword
behavior separate without introducing another width policy.

Local placement and manager-backed placement are targets of the same typed
operation request, not distinct memory actor kinds. A memory occurrence may
therefore expose both a local service and a manager endpoint while Mapping
selects exactly one target for each operation entry. A shared module memref can
feed several occurrences because it denotes access to one service capability,
not duplicated token data. Keeping the builtin address channel wider than its
local capacity also avoids conflating address representation with storage
range: Mapping still proves that a local interval fits, while the same engine
can issue a wider manager address without a second request protocol.

A manager endpoint is a path out of the SpatialCore, not storage. Requiring
every Spatial MemoryBinding to name a Fabric service region would therefore
either reject an operation-engine-only memory or falsely promote its manager
endpoint into a service. The closed LocalRegion-or-BoundaryProxy target keeps
that distinction explicit. The proxy reuses the MemoryBinding identity because
the logical interval is the only persistent Spatial fact that SystemMapping
must extend; assigning another proxy identity would describe no additional
object. SystemMapping then owns the first real provider-region and address
selection outside the SpatialCore.

Keeping boundary candidates factorized also matters for PnR. Spatial search
chooses among local regions or one proxy value per logical interval; it does
not materialize a product with every possible system provider. System search
later expands only the selected proxy obligations against compatible
providers. Dense local ordinals and reverse incidence tables can therefore
update memory moves without decoding persistent references or scanning the
system service catalog.

## Why Memory Owns Several Independent Widths

Data, scalar address, indexed address, lane mask, and service beat represent
different information. A contiguous vector carries one scalar base address,
while an indexed vector of the same data width may carry many resolved-index
values. A memory service may also accept a logical operation endpoint wider
than one physical beat. Collapsing these facts into one bus width would admit
impossible gathers and would make beat splitting an undocumented convention.

Each endpoint and service therefore owns its exact width. The actor-derived
memory access view supplies complete data, address, and mask requirements, and
the Fabric capability relation decides whether each role fits. Internal lane
or beat work exists only through the declared transaction projection and
service contract. This permits a useful wide endpoint over a narrower service
without letting Mapping invent a transaction protocol.

The same separation explains why equal data payloads remain different memory
capabilities. An element load of a vector-valued memref and a contiguous load
of scalar elements need different address and child-transaction behavior even
when both return the same vector type.

## Why Service And Operation Ports Are Separate

An addressed memory actor is a software service request with data, address,
control, result, and completion legs. A memory capability exposure is only a
boundary through which a service can be used; it is not a zero-leg operation.
Combining the two would force every exported memory to invent request and
response behavior.

Operation rows select real internal source-to-sink relations and service
targets. Temporal rows may share tagged ingress only through explicit role
matchers, queues, and capacity. Same-row outputs remain injective so one
transaction cannot silently duplicate distinct response roles.

Resident row count, service reachability, and selected Mapping configuration
are deliberately separate. The engine owns the finite row domain, the
occurrence owns fixed dispatch and internal-connectivity eligibility, and
Mapping chooses active rows, targets, tags, and edges. Putting those facts in
every port would duplicate one hardware cross-connect; persisting configured
rows in Fabric would duplicate Mapping. Queue and holding capacities remain in
the operation port's ResourceContract rather than being repeated beside the
resident count.

Subordinate provider decode is part of the same connectivity owner because it
uses the same physical service targets and response tracking. A bounded set of
match fields and an optional constant base offset cover the direct hardware
case. Hashing, translation, cache, or coherence remain explicit service
transforms instead of growing a predicate language inside `fabric.mem`.
The closed match atoms use one schema-wide 64-bit address/context domain and a
32-bit address-space domain. Keeping those widths in the schema avoids both an
occurrence-level width knob and a backend default while retaining an explicit
typed-transform path for services outside that bounded direct case.

Local and System memory services share one exact service-capability record
because actor admission, regions, physical beats, resource use, and progress
mean the same thing at either scope. They differ only where root locality is a
real hardware distinction: a local service provides its own consistency
guarantee relative to the occurrence clock, while a System service may bind to
an explicit `MemoryConsistencyDomainRef`.

A behavior-only service attribute was rejected because `Storage` does not
state which actors, widths, regions, beats, timing, or contention the hardware
supports. A generic service path was rejected because each consumer would
then invent those facts independently. Reusing the operation-port actor and
access domains keeps software compatibility single-owned, while the service's
one ResourceContract keeps physical capacity and arbitration single-owned.

## Why Memory Engines Have Canonical Templates

TechMapping must prove a memory subgraph against a hardware definition before
SpatialMapping chooses a concrete physical occurrence. Occurrence-relative
ports alone cannot express that proof without prematurely performing placement.
The earlier Mapping-owned `MemorySemanticEncoding` instead copied Fabric
operation rows and internal connections into an enumerated configured subset.
That made the same capability relation independently editable in two owners and
grew with every legal active subset.

Fabric finalization therefore derives one canonical Memory Operation Engine
template from each occurrence's exact engine contract, token endpoint types,
complete operation-port records, and internal connection relation. Equal
definitions deduplicate. TechMapping selects template-relative structures;
SpatialMapping selects an occurrence with the same template and obtains the
concrete structures by mechanical projection.

Local Memory Service, manager and subordinate endpoints, dispatch domains, and
topology are deliberately absent from the template. They are occurrence-level
placement and service facts. Conversely, operation-port capability,
ResourceContract, schedule, resident capacity, and engine-internal connectivity
must be present because they determine whether the software realization is
semantically implementable before placement. This is the same definition-
versus-occurrence distinction already used by FU capability templates, applied
to the independently meaningful memory engine boundary.

## Why One Memory Consistency Kernel Spans Fidelity

Dataflow owns ordering, scope, issue, linearization, retirement, and progress
semantics. Fabric owns the hardware guarantees and resource capability that
can implement them. Mapping binds the two. DFG, CGRA, and system providers
execute the same logical consistency contract at different hardware fidelity.

Separate memory models per simulator would disagree on modification order,
reads-from, atomicity, visibility, or completion. Conversely, implementing a
complete coherent system inside Loom would duplicate gem5. The shared kernel
handles software semantics; exact local Fabric or external system providers
own physical timing and contention.

## Why Resource Claims And State Transitions Differ

An atomic UsePattern acquisition reserves temporary service capacity. Queue
contents, head/tail, occupancy, and other committed resource state persist
after that use. The earlier model tried to represent both as claims, which
would require one use to release another use's claim, transfer claim ownership,
or add per-claim lifetimes.

The distilled model keeps one all-or-nothing claim envelope and an optional
owner-defined atomic commit transition. A temporal PE enqueue temporarily
claims its enqueue service and commits an append; dequeue claims its service
and commits a removal. Durable queue state remains in the Fabric resource.

The same distinction handles an elastic result slot without a second retire
use. Acceptance owns one capacity claim for the complete active result tuple;
a publish transition materializes claim-local holding state, and the same use
retains it until all active result handoffs occur. Releasing the claim destroys
that local tuple. Distinct result handoffs clear claim-local obligation bits,
not independent resource claims or payload slots; only the final handoff
releases capacity. This lets ordinary ready/valid consumers make independent
progress without turning a multi-result operation into an atomic broadcast.
No later use drains inherited state, and no per-result claim can release the
tuple early.

Mapping represents that release as a conjunction of existing Dataflow event
points. A new aggregate event identity would duplicate the canonical terminal
events, while a single selected result would be wrong for multi-result actors.
The nonempty sorted `AllOf` set is therefore the smallest relation that keeps
Dataflow as event SSOT and preserves one atomic claim envelope.

This also gives a precise same-cycle rule. Dequeue observes cycle-start state;
capacity is checked after that removal; enqueue cannot bypass a new token
directly to dequeue. Combined actor inputs must acquire all required services
or none, so Mapping cannot partially consume an actor.

## Why Operand Buffer Capacity Is Fabric Truth

All temporal operand-buffer organizations share FIFO semantics but project
logical queues to physical allocation units differently. Omitting depth for
one mode or inventing a backend default makes capacity depend on the consumer.
Every mode therefore has explicit positive capacity per derived allocation
unit.

Tags dispatch at PE ingress into logical operand queues. Tagged ports can carry
multiple logical nets only when tag width and non-conflicting assigned values
prove separation. Untagged ports cannot. Deterministic round-robin resolves
contention where several queues share one service; a private backend scheduler
would be a second execution authority.

The entry pool, per-cycle enqueue and dequeue services, and any operation or
memory-service holding slots remain distinct owner-typed states inside that one
contract. Mapping reports an exact capacity excess through its single
`CapacityOveruse` fact and carries the owner reference as diagnostic context;
it does not reinterpret these states through global buffer, time, or service
violation categories.

## Why Boundary Is An Atomic Rendezvous

Spatial-to-temporal and temporal-to-spatial boundaries expose separate data
and tag legs. Partial consumption or partial publication would lose their
correspondence. Adding hidden holding state would turn a type-domain boundary
into an implicit FIFO.

The boundary is therefore stateless and atomic across all active legs. It
transfers only with joint validity and readiness. Tag lookup must cover every
reachable value uniquely. Designs needing storage or a pipeline use explicit
resources whose state and timing enter Fabric identity.

Treating the tag output of a split boundary as another software route would
mix physical correspondence metadata with the software payload and can violate
that payload's width contract. Selecting one boundary leg therefore activates
the whole owner; Fabric derives the companion handshake instead of Mapping
inventing a second logical net.

## Why FIFO Bypass Remains A Mapping Refinement

A declared FIFO can operate buffered or transparently only when Fabric exposes
that choice. Bypass improves cycle count but lengthens the combinational path;
buffering adds latency while cutting both forward-valid and backward-ready
dependencies. Its input capacity and output occupancy are cycle-start
registered facts. A full queue therefore cannot borrow capacity from a
current-cycle dequeue; preserving that full-queue throughput would require an
explicit skid or credit refinement with separately declared physical capacity.
The choice is semantic-preserving but performance-relevant, so SpatialMapping
selects it under Evaluation guidance.

Handshake-cycle legality is split by owner. Fabric rejects unconditional
combinational cycles in the fully expanded hardware. Mapping derives the exact
selected active graph and rejects cycles introduced by selected bypasses or
switch alternatives. A topology that can form a cycle under some configuration
is valid Fabric; finalization does not union mutually exclusive rows or modes.
Every selected buffered FIFO is a complete ready/valid combinational isolation
point, while every selected bypass is transparent in both directions.

This changes both cycle timing and which selected configurations are legal, so
it cannot reinterpret a 3.x Fabric Artifact in place. Version 4.0 introduced
the complete isolation contract and rejects 3.x roots through the existing
exact-version dependency and import boundary.

## Why Switch Behavior Must Be Explicit

A switch can express broadcast and, when temporal, multiplex several inputs to
one output under its declared policy. Spatial fan-in alternatives are resolved
by one statically capacity-closed configuration, so giving them a runtime grant
policy would fabricate an arbiter. Temporal fan-in is real runtime contention
and keeps its input-owned requesters and exact policy. These atomic transfer,
backpressure, arbitration, and resource-use facts cannot be inferred from
multiple module-level SSA uses or from a route tree alone.

The mapper selects traversals and configuration from the switch's exact
connectivity and resource contracts. The architecture-level switch owner
defines observable grant transitions and the configured readiness-presentation
invariant. Protocol packetization, the exact idle-presentation mechanism, and
physical register realization remain implementation details.

Potential fan-in is not active contention. A switch may admit crosspoints that
no resident row selects, and globally forbidding every cycle those alternatives
could form would reject legal configured hardware. Fabric therefore owns one
compact conditional handshake shape; Mapping supplies the exact resident-row
selection that activates it.

Readiness follows configured contention components because idle presentation
observes only selected output conflicts. Unused physical crosspoints therefore
cannot affect presentation. Output validity cannot use the same undirected
approximation. Round-robin grant state may place any requester first, but fixed
priority has a stable direction. A priority-prefix projection preserves that
distinction in linear space, so a lower-priority requester does not gain a
false dependency on an unrelated earlier output. Presentation-only tag or data
selection is not repackaged as a Valid dependency.

One `fabric.switch` is one physical crossbar, so its implementation cost is not
independent of its shape. Area, wiring, selector depth, timing closure, and
configuration grow with both port dimensions. Treating a schedule-wide
crossbar as an abstract free routing node would make physically implausible
hardware valid and would hide the real distributed routing problem from PnR.

The crosspoint product, rather than either dimension alone, bounds the
primitive's representable physical scope. The advisory threshold above 64
crosspoints preserves an escape hatch for unusual but still bounded hardware,
including asymmetric selectors or fanout, while making an expensive choice
visible. More than 256 crosspoints is invalid. A warning cannot affect Fabric
identity or Mapping semantics; otherwise diagnostic policy would become
another architecture owner. Larger networks compose ordinary switch
occurrences and connections, which exposes their real links, capacities, and
contention to every consumer.
