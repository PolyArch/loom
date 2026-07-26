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

## Why Boundary Is An Atomic Rendezvous

Spatial-to-temporal and temporal-to-spatial boundaries expose separate data
and tag legs. Partial consumption or partial publication would lose their
correspondence. Adding hidden holding state would turn a type-domain boundary
into an implicit FIFO.

The boundary is therefore stateless and atomic across all active legs. It
transfers only with joint validity and readiness. Tag lookup must cover every
reachable value uniquely. Designs needing storage or a pipeline use explicit
resources whose state and timing enter Fabric identity.

## Why FIFO Bypass Remains A Mapping Refinement

A declared FIFO can operate buffered or transparently only when Fabric exposes
that choice. Bypass improves cycle count but lengthens the combinational path;
buffering adds latency while cutting forward-valid dependency. The choice is
semantic-preserving but performance-relevant, so SpatialMapping selects it
under Evaluation guidance.

Handshake-cycle legality is split by owner. Fabric rejects unconditional
combinational cycles in the fully expanded hardware. Mapping derives the exact
selected active graph and rejects cycles introduced by selected bypasses or
switch alternatives. Neither owner uses a blanket rule that every FIFO cuts
both valid and ready.

## Why Switch Behavior Must Be Explicit

A switch can express broadcast and, when temporal, multiplex several inputs to
one output under its declared policy. Those are real atomic transfer patterns,
backpressure, arbitration, and resource use. They cannot be inferred from
multiple module-level SSA uses or from a route tree alone.

The mapper selects traversals and configuration from the switch's exact
connectivity and resource contracts. Protocol packetization and implementation
microstate remain outside the architecture-level switch owner.
