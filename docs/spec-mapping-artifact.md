# Mapping Artifact

This document specifies the canonical persistent Dataflow-to-Fabric
Mapping Artifact. A Mapping Artifact records one selected realization of
one finalized Canonical Dataflow Program on one finalized Fabric Hardware
Description. It does not mutate either input.

The persistent model has one Mapping schema. TechMapping and Physical
Mapping are immutable artifacts checked under different closed verifier
profiles of that schema. They are not competing formats or mutable states
of one artifact.

Persistent identity and reference rules are specified by
`docs/spec-mapping-identity.md`.

## Artifact Header

Every Mapping Artifact has one canonical header containing at least:

* a Mapping schema identity and version in `X.Y` form;
* one closed completeness profile, initially `tech_mapping` or
  `physical_mapping`;
* its content-derived artifact identity; and
* the exact upstream artifact references required by that profile.

`X` denotes a breaking or incompatible schema change. `Y` denotes a
non-breaking schema improvement. A profile is a verifier contract, not a
lifecycle state. Exact Mapping dialect assembly syntax is outside this
document.

The current Mapping schema version is `2.0`. Version `2.0` requires the
complete semantic encoding, configured-function projection, and exact
correspondence witness described by this specification. Version `1.x` does not
carry those required semantics and is not accepted through a compatibility
adapter.

## Semantic Owners

The Canonical Dataflow Program owns software execution semantics,
including graph definitions, actors, typed edges, graph boundaries, and
explicit control, state, stream, and memory semantics.

The Fabric Hardware Description owns hardware facts, including elaborated
topology, FU topology, capabilities, configuration domains, ports, and
physical implementation structure. Its finalized facts include typed compute
occurrences, exact FU membership, schedule kind, physical endpoints, and
explicit directed local arcs. The same transport-endpoint identity namespace
also covers endpoints owned by typed compute-only switch, FIFO, and boundary
resources. Fabric owns each endpoint's exact owner, direction, port kind, and
explicit native transport kind (`bits` or `bits_tag`), together with
independent payload and tag capacities. Fabric also owns every explicit
directed point-to-point arc and resource traversal. Boundary resources retain
the canonical Fabric boundary direction (`s2t`, `t2t`, or `t2s`) rather than a
Mapping-local spelling. Fabric's definition-level facts also include memory
service domains, implementation families, load/store operation-port
templates, explicit typed implementation boundary ports, normalized semantic
encodings, typed internal connectivity, source-endpoint fanout capacities,
and one-beat access contracts.

The TechMapping-profile Mapping Artifact owns the selected logical
realization relation between those exact artifacts. It owns
target-specific compute actor grouping, selected FU realization,
software-to-FU correspondence, the selected Fabric-defined valid semantic
encoding, and Memory Realizations against exact Fabric memory semantics.

The Physical-Mapping-profile artifact owns the concrete physical
realization and legality facts added after TechMapping. At this level only
the ownership categories are fixed: resource binding, communication
realization, temporal and resource use, and storage realization. Their
persistent record schemas remain deferred.

Evaluation Evidence owns measured or estimated results, cost, metrics,
diagnostics, and fidelity. It references Mapping Artifact identities and
must not copy mapping decisions into a second authority.

## Immutability

A persistent Mapping Artifact is finalized, immutable, serializable, and
profile-verifiable. Producers may use mutable builders, solver state, or
failed candidates internally, but those objects are not Mapping Artifacts.

There is no persistent `partial` to `complete` lifecycle. A TechMapping
artifact either satisfies its profile for its declared coverage or is
invalid. A Physical Mapping either composes with its exact TechMapping
predecessor to satisfy its profile or is invalid.

Rejected candidates, incomplete construction state, search queues, and
failed-attempt diagnostics belong to producer state or Evaluation
Evidence.

## Exact Input Coupling

A TechMapping artifact is coupled to exactly:

* one Canonical Dataflow Program identity; and
* one complete Fabric Hardware Description identity.

Both references are required semantic content. Every software and
hardware reference resolves through those exact immutable artifacts.

A TechMapping artifact cannot be rebound to another Fabric Hardware
Description, including one described as compatible by another tool. An FU
implementation content digest may support deduplication or a pure cache,
but it cannot replace the exact Fabric artifact identity in a persistent
reference. A different Fabric identity requires a different TechMapping
artifact.

A Physical Mapping references exactly one immutable TechMapping
predecessor. It inherits the predecessor's Canonical Dataflow Program,
Fabric Hardware Description, coverage, Compute Realizations, and Memory
Realizations. It must not restate or copy those facts as independent
authority.

Conceptually:

```text
TechMapping = realization witness for exact D + F
PhysicalMapping = exact TechMapping predecessor + physical delta
```

This relation does not define final Mapping assembly syntax.

## TechMapping Profile

The TechMapping profile declares a canonical, non-empty set of covered
`dataflow.graph` definitions from the referenced Canonical Dataflow
Program. Coverage is part of the Mapping artifact; it does not create a
separate Mapping Scope entity.

Coverage is closed for every declared graph:

* every actor belongs to exactly one Compute Realization or Memory
  Realization;
* `dataflow.load` and `dataflow.store` actors belong only to Memory
  Realizations, while all other actors belong only to Compute Realizations;
* no realization crosses a graph-definition boundary;
* every canonical edge is classified exactly once as realization-internal
  or realization-external;
* every external software endpoint has one exact typed FU or memory
  operation-template port correspondence;
* graph boundary and typed value, stream, control, state, and memory
  obligations are accounted for; and
* there are no unmapped actors, duplicate coverage, dangling records, or
  implicit default realizations.

Only an immutable artifact satisfying this closed coverage gate may enter
PnR.

The profile contains Compute Realizations, Memory Realizations, and any typed
representation obligations required by those realizations. Adapter record
syntax remains deferred.

## Compute Realization

A Compute Realization is the single record that owns a target-specific
actor group and its selected FU realization. There is no separate Software
Actor Group record because the group has no independent meaning inside a
selected mapping candidate.

Its confirmed conceptual content is:

* a persistent record identity;
* one or more actor references within one graph definition;
* one exact FU implementation reference in the referenced Fabric artifact;
* complete actor-to-`fabric.op` correspondence;
* complete typed software-boundary-to-FU-template-port correspondence;
* a reference to the selected Fabric-defined valid semantic encoding; and
* the typed legality and representation obligations for the match.

This list defines semantic ownership, not Mapping dialect field syntax.
The encoding representation and configured-function projection are specified
by `docs/spec-fabric-reconfigurable-op.md`.

The configured function is a derived typed and attributed projection of
the selected FU topology under the selected encoding.
It is used to verify the actor group, but it is not persisted as a second
software graph, does not receive independent program identity, and does
not replace either input artifact.

Mapping does not copy selected `sw_configs` into canonical Fabric. A backend
may derive transient `sw_configs = {mode = N}` values for `fabric.op`
resources and route selections from the selected encoding, but those values
are caches or lowering products, not another capability or type authority.

## Memory Realization

A Memory Realization covers a non-empty set of canonical `dataflow.load` and
`dataflow.store` actors within one graph definition. Singleton records model
independent accesses. Multi-actor records model a software memory subgraph
implemented by one selected normalized `fabric.mem` semantic encoding. Memory
Realization is orthogonal to Compute Realization and does not introduce a
persistent common actor-realization wrapper.

The neutral Dataflow importer produces one validated derived view for each
memory actor. That view jointly owns the canonical operation kind, logical
memory root, semantic port roles, access width, access size, and alignment.
These facts are not Mapping-selected metadata or a second operation authority.
Each logical root owns one graph definition and zero or more memory imports and
exports in that graph. This covers imported-only, fresh, exported-only,
re-exported, aliased/viewed, and non-exported scratch roots without changing
the root model. Every graph memory port belongs to exactly one root. Import and
export types may differ when a re-export exposes another view or layout.
Memory capability identity is never represented as a Dataflow edge.

The implemented neutral record contains:

* stable actor references and bijective actor-to-operation-template
  correspondence;
* the logical memory root associated with each actor and the record's exact
  root set;
* one selected Fabric-defined normalized memory semantic encoding;
* exact typed address/data/mask/control/result/done boundary-port
  correspondences;
* the minimal graph-port-to-memory-implementation-boundary correspondence
  required by selected internal graph-boundary connections; and
* internal-edge witnesses pairing absorbed canonical edge references with
  selected Fabric-defined typed internal connections.

Fabric internal connections use one endpoint variant: an explicit memory
implementation boundary port or a memory operation-template port. Typed and
directed connections may run from a boundary input to an operation input,
from an operation output to an operation input, or from an operation output
to a boundary output. Actor endpoints are derived from actor-to-operation
correspondence; graph endpoints require the explicit Mapping correspondence.
Every connection selected by the normalized encoding has exactly one canonical
edge witness, and every witness names a selected connection. Actors in the
same Memory Realization do not make their edges internal automatically.
Unselected edges remain external. Fanout is legal only within the capacity
declared once by the Fabric source endpoint.

Each selected operation template supplies normalized per-access-size tuples.
Each tuple owns one access size and its required alignment. Listing a store
tuple means the hardware implements canonical store semantics for that exact
shape, including preservation of bytes outside a narrow write. Hardware that
would clobber those bytes does not list the tuple. Software width must not
exceed physical data width, and the software alignment guarantee must be a
multiple of the tuple's required alignment. A narrow load may use low-bit
extraction from the physical result. The neutral core does not split accesses
into multiple beats or hide read-modify-write behavior.

Every logical memory root resolves to one coherent service-domain obligation
across all Memory Realizations. Different roots may map many-to-one to the same
service. A root cannot silently move between unrelated service domains.

The record contains no concrete memory occurrence, physical operation port or
context, bank, tag, base or range, route, buffer, schedule, arbitration, or
resource-time choice. The current implementation is limited to the neutral C++
Artifact and Verifier model plus the ephemeral PnR structural projection
described below. It does not add Mapping MLIR persistence or physical memory
binding records.

## Edge Ownership

An edge whose producer and consumer actors belong to the same Compute
Realization is implemented by the configured FU topology. Its legality is
part of the Compute Realization witness and it creates no external physical
communication obligation.

An edge crossing an actor-realization boundary remains owned by the Canonical
Dataflow Program. Its source and sink realizations provide exact typed FU or
memory operation-template port correspondences for the exposed endpoints. A
PnR importer mechanically derives the external communication obligation from
those facts. A Memory Realization internal-edge witness is the only mechanism
that removes one of its canonical edges from this external set.

Graph memory imports and exports belong to the capability plane. They are
accounted for by logical roots and service obligations, never by ordinary
Dataflow edges or token sink accounting.

Importers must not infer correspondence from textual order, symbol
spelling, paths, or port names. Canonical fanout groups edges with the same
exact canonical source endpoint into one multi-sink logical obligation; a
duplicate persistent TechMapping netlist is not another authority.

## Physical Mapping Profile

A Physical Mapping is exactly one immutable TechMapping predecessor plus
a physical delta. The delta owns only concrete physical realization facts
and must preserve every predecessor Compute Realization and Memory
Realization.

Physical Mapping must not regroup actors, select another FU
implementation, change the selected semantic encoding, reinterpret FU
configuration, replace a selected memory implementation or operation
template, semantic encoding, logical-root association, or internal-edge
witness, or guess software-to-hardware correspondence. Any such change requires
a new TechMapping artifact.

The exact physical record families and their completeness rules are not
defined here. Route trees, resource-time claims, instruction slots,
temporal tags, schedules, buffers, memory bindings, boundary realization,
and related schemas remain subject to their owning discussions and specs.

## Derived Projections And Caches

Importers, viewers, simulators, and PnR kernels may build immutable derived
projections such as configured-FU graphs, dense indices, logical adjacency,
or pure match results.

Every projection is non-authoritative, deletable, and deterministically
rebuildable from exact finalized inputs. Cache keys bind all semantic
inputs and the producing algorithm semantics.

The implemented `freezeRealizationGraph` projection is the bounded
Dataflow/TechMapping/Fabric structural input to later PnR construction. It
rechecks the two exact input identities, consumes the immutable canonical
compute-occurrence projection produced by Fabric validation, assigns dense
native indices by persistent entity identity, records actor ownership, derives
external multi-sink logical nets from canonical edges and exact boundary
correspondence, and derives logical-memory-root service obligations from
selected Memory Realizations. Its dense terminal table contains only selected
FU or memory operation-template terminals needed by those logical nets. Graph
boundary endpoints remain embedded typed terminal variants in logical-net
sources and sinks. Graph memory import and export capability ports are not
token terminals.

For each Compute Realization, the projection derives `ImplDomain` solely from
exact selected-FU membership in the finalized occurrence table. Each
implementation occurrence retains factorized `PortDemand` records and flat
compatible-endpoint ranges derived from the explicit local arc, direction,
port kind, payload and tag capacity, and intrinsic role and compatible type
facts owned by Fabric. Spatial unary feasibility uses bipartite matching to
prove that all exposed FU ports can bind distinct endpoints. It does not
enumerate endpoint permutations or persist Cartesian local configurations.
Temporal endpoint ranges are only structural capability; tag sharing and
resource-time legality remain deferred.

Fabric validation is the single semantic owner of occurrence identity,
cross-kind uniqueness, exact FU membership, schedule kind, endpoint/type facts,
and local arcs. It copies those facts into deterministic vectors, builds one
sorted FU-to-occurrence range table and one sorted
occurrence/FU/direction/port-to-arc range table, and retains that immutable
projection with the validated Mapping value. Freeze does not inspect or
revalidate the source occurrence vectors. Mutating those source vectors after
validation cannot change frozen output, while an input identity mismatch still
invalidates the freeze request.

Fabric validation also owns one canonical routing projection. Persistent typed
endpoint IDs are globally unique across compute and resource-owned endpoints,
and persistent typed transport-resource IDs are globally unique across the
Fabric artifact namespace. Resource endpoint ownership is structural: every
resource endpoint is nested under exactly one switch, FIFO, or boundary
resource, while every compute endpoint is nested under exactly one compute
occurrence. Validation resolves all references against the exact Fabric
identity before retaining the projection.

A routing endpoint declares both its software port kind and its native Fabric
transport kind. `bits` has zero tag capacity. `bits_tag` has positive tag
capacity. The native kind is never inferred from tag capacity.
`PortKind::Memory` compute endpoints are omitted from this compute-only
projection, while memory resource endpoints or routing references to excluded
memory endpoints are malformed. Memory capability and service connectivity
remain a distinct future projection and do not enter token-routing CSR.

A point-to-point arc is legal only from one output endpoint to one input
endpoint with the same software port kind and native Fabric transport kind.
Each output has at most one direct point-arc consumer and each input has at
most one direct point-arc producer; fanout and fan-in therefore require
explicit Fabric resources. A switch or FIFO traversal is legal only from an
input endpoint to an output endpoint owned by the referenced resource, with
the same software port kind and native Fabric transport kind. Switch
connectivity consists solely of its explicitly listed traversals. A FIFO has
exactly one input, one output, and one explicit forward traversal.

A boundary has exactly one input, one output, one explicit forward traversal,
and one canonical `fabric::BoundaryDirection`. Its payload path must match the
Fabric boundary verifier exactly: `s2t` is `bits<W> -> bits_tag<W,T>`, `t2t`
is `bits_tag<W,T1> -> bits_tag<W,T2>`, and `t2s` is
`bits_tag<W,T> -> bits<W>`. Payload widths must be equal, and every tagged
endpoint must have positive tag capacity. No other native-kind conversion is
legal. The lightweight Fabric-owned `checkBoundaryDataPath` helper is the
single authority for this data-path relation and is consumed by both Fabric IR
verification and Mapping routing validation. Duplicate arcs, duplicate
traversals, foreign or wrong-kind references, wrong-direction references, and
cross-resource traversals are malformed input. Missing arcs or traversals that
are not structurally required may leave otherwise valid endpoints or topology
components disconnected; disconnected and unreachable topology is not a
validation failure.

The implemented `freezeRoutingGraph` projection consumes only that retained
validated routing projection after rechecking the exact Fabric identity. It
orders resources and endpoints by persistent typed identity, converts all
indices and range boundaries through checked `PnrIndex` operations, and emits
a directed CSR adjacency table. Each frozen arc records whether it is a bare
point arc or a resource traversal, the traversal's resource index when
present, and independent effective payload and tag capacities. Each capacity
is the minimum of the source and target endpoint capacities in that field; tag
capacity is never added to payload capacity.

The frozen routing resource table records the canonical boundary direction
when the resource is a boundary. The frozen routing endpoint table records
only typed identity, owner kind and dense owner index, direction, software port
kind, native Fabric transport kind, and the two capacities. Resource endpoint
ranges are flat index vectors rather than nested graph objects. A separate
`computeEndpointVertices` vector follows the non-memory subsequence of the
exact compute-endpoint ordering already used by
`FrozenRealizationGraph::physicalEndpoints`, so later multi-source and
multi-target token routing can consume factorized endpoint domains without
selecting an occurrence, endpoint, path, configuration, capacity claim, or
route tree. The routing graph contains no strings, MLIR containers,
coordinates, names, implicit reverse arcs, geometric adjacency, symbol-order
adjacency, topology-specific shortcuts, reachability matrix, memory service
connectivity, or routing-search state.

Descriptor-vector permutation cannot change structural equality, canonical
table ordering, or CSR adjacency ordering. Arc order is derived from source
endpoint identity, target endpoint identity, arc kind, and resource identity.
The CSR graph preserves only explicit structural capability, so an irregular
topology may make two individually valid compute occurrences unreachable from
one another without turning either unary candidate or the Fabric artifact into
malformed input.

Let `O`, `M`, `E`, and `A` be the occurrence, membership, endpoint, and local
arc counts. Validation canonicalization costs sorting time over those vectors,
bounded by `O((O + M + E + A) log(O + M + E + A))`. For a realization with
`P` exposed FU ports and `K` occurrences in its exact FU range, `ImplDomain`
lookup costs `O(log F + K)` for `F` indexed FUs. A port domain lookup costs
`O(log Q_o + sum(log(1 + T_e)))` for `Q_o` keyed port ranges in that occurrence
and the compatible-type counts `T_e` of the relevant arcs' endpoints. If `R_o`
is the number of relevant arcs examined, this is bounded by
`O(log Q_o + R_o log(1 + T_max))`. Spatial matching uses occurrence-local
endpoint indices and a reused augmenting-path workspace; for `D_o` compatible
domain edges it costs `O(P * D_o)` time. The workspace retains buffers sized
to the largest observed `P`, `E_o`, and `D_o`, so scratch is
`O(P_max + E_max + D_max)` and allocation occurs only through amortized vector
growth. Matching and visit marks use 64-bit generations: a candidate does not
clear matching state across the endpoint range, and each augmenting-path probe
touches only endpoints it visits. A full mark reset occurs only on generation
rollover after `2^64 - 1` generations. Thus constant-port construction is
bounded by
`O(log F + K log Q_max + sum(R_o log(1 + T_max) + D_o))` after
canonicalization, including matching, rather than cubic or quartic in
artifact-wide occurrence or endpoint counts.

Malformed Fabric structure or references are rejected during validation and
cannot be represented as an empty domain. `InvalidComputeOccurrence` is the
single category for an invalid schedule, empty or repeated FU membership,
invalid endpoint signatures, repeated compatible types, and invalid local-arc
structure. Foreign, unresolved, or wrong-kind references retain their precise
reference categories, and malformed graph or memory port connections retain
`InvalidPortConnection`. A well-formed realization with no exact
implementation occurrence or no unary-eligible occurrence instead produces
structured mapping infeasibility. Occurrence and endpoint ordering is derived
from persistent typed identities, so harmless source-vector permutations do
not alter the frozen result.

Both frozen projections are ephemeral and have no independent artifact
identity, serialization, canonical byte encoding, or persistence form. Their
equality is structural only. Neither contains a selected occurrence, selected
endpoint, complete candidate domain, configuration, placement, route, tag,
buffer, resource-time, or physical-memory decision.
`freezeRealizationGraph` is the three-input bounded projection;
`freezeRoutingGraph` consumes the exact Fabric identity retained by the
validated Mapping value. They are not the complete four-input `FrozenModel`;
`ConfigDomain` and the full `CandidateDomain` remain outside this boundary.

The structural subview intentionally retains canonical edge identities, dense
terminal references, and deletable occurrence and endpoint-domain caches. It
must not reinterpret Fabric facts or create a second occurrence-membership,
local-connectivity, or legacy routing authority.

A cache must not transfer mapping coverage, artifact-local references,
current-Fabric legality conclusions, or physical decisions into another
artifact context. After a cache hit, the producer still resolves current
references, constructs new artifact-local results where required, and runs
the applicable profile verifier. Cache behavior must not change semantic
output or deterministic ordering.

## Evaluation Evidence Boundary

Evaluation Evidence is a separate immutable artifact. It may reference the
Canonical Dataflow Program, Fabric Hardware Description, Mapping Artifact,
evaluator model, runtime inputs, and other evaluated subjects.

Evidence must not duplicate Compute Realizations, coverage, selected FU
configuration, or physical realization facts. Referencing the exact
Mapping Artifact identity identifies those facts. Evidence cannot make an
incomplete or invalid mapping acceptable.

Synthesis and DSE evidence may record explicit encoding count, distinct
input-covered encoding count, and extra capability count. These are candidate
metrics derived from Fabric encodings and coverage witnesses; they do not add
another Mapping or capability authority.

## Profile Validation

The shared Mapping verifier supports closed completeness profiles.

The TechMapping profile verifies at least:

* exact Canonical Dataflow Program and Fabric Hardware Description
  identities;
* resolution and type correctness of every persistent reference;
* unique finalized compute occurrence and endpoint identities, exact local FU
  membership, valid endpoint ownership, and explicit local FU-port arcs;
* closed coverage for the declared graph-definition set;
* disjoint and complete actor coverage across Compute Realizations and Memory
  Realizations;
* complete actor-to-`fabric.op` and boundary-port correspondence;
* exact graph-owned memory root and import/export port partition, selected
  normalized encoding, bijective operation-template ownership, graph and actor
  boundary correspondence, and selected-connection/internal-edge witness
  equality;
* coherent service-domain consistency for each logical memory root;
* correlated one-beat access width, size, required alignment, and narrow-store
  legality;
* selected FU and encoding ownership; and
* configured-function equality for the actor group, including exact semantic
  types, attributes, ordered edges, fanout, and boundary correspondence;
* all required typed realization and representation obligations.

Port legality uses exact port kind and intrinsic role plus compatible payload
capacity. In particular, `bits` and `bits_tag` do not correspond implicitly,
while an untagged physical payload may be wider than the software requirement
under the low-bit-aligned widening and narrowing rules owned by Fabric.

The Physical Mapping profile verifies the exact immutable predecessor,
rejects copied or conflicting TechMapping authority, and checks that the
physical delta preserves the predecessor. Detailed physical completeness
checks remain deferred with their record schemas.

## Non-Goals

This document does not define:

* Mapping dialect assembly syntax;
* physical delta record schemas;
* route-tree, resource-time, schedule, tag, buffer, memory, or boundary
  schemas;
* the complete four-input `FrozenModel` and later physical PnR data layout;
* `ConfigDomain`, the complete `CandidateDomain`, placement choice, endpoint
  selection, or route search;
* Hardware Sharing Group registry syntax;
* SystemMapping composition; or
* bitstream format.
