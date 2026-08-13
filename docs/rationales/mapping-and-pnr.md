# Mapping And PnR Rationale

Normative contracts are owned by
[Mapping Artifact](../spec-mapping-artifact.md),
[Mapping Identity](../spec-mapping-identity.md),
[Mapping Memory](../spec-mapping-memory.md),
[Mapping Verification](../spec-mapping-verification.md),
[TechMapping Generation](../spec-tech-mapping.md), and
[Place And Route](../spec-pnr.md).

## Why Mapping Has Three Profiles

Software-to-hardware realization contains three cumulative questions:

* TechMapping selects a Fabric-owned FU capability template or Memory Operation
  Engine template and proves that a canonical software subgraph is supported;
* SpatialMapping selects concrete resources, contexts, endpoints, routes,
  buffers, tags, and physical refinements inside a SpatialCore; and
* SystemMapping binds thread/graph execution, channels, services, and transport
  across heterogeneous AccCores.

TechMapping has no route because it selects a logical materialization against a
template, not a physical occurrence. Spatial and System Mapping both route,
but in different resource universes. A fourth PhysicalMapping profile was
rejected; concrete physical choices already belong to the latter two profiles.

All three are immutable roots in one Mapping artifact family. A mutable
partial/complete object or a separate witness schema would force consumers to
interpret lifecycle state. Each profile is complete under its own verifier and
references its exact predecessor.

## Why Ordered Cardinality Requires Mapping 5.0

The 4.0 release algebra can encode both `Intrinsic` and
`AllOf(activeResults)`, but an ordered-cardinality actor changes which form
is a complete realization. A one-cycle adapter contract cannot represent a
zero-production firing or repeated production of one result ordinal. Pairing
it with parallelize or serialize was therefore an accepted-language error,
not missing backend availability.

Three repairs were considered. Keeping the pair valid and rejecting it only
during RTL lowering would let a complete Mapping describe an unrealizable
claim lifetime. Adding an implicit 4.0 compatibility interpretation would
create a second release authority and still lack the dynamic production
sequence needed to adapt the record. The selected repair makes the exact
Fabric ordered-cardinality contract a Mapping 5.0 admissibility condition.
Version 5.0 rejects every 4.0 root, while the old one-cycle contract remains a
valid generic Fabric record. A source-version migration must rebuild and
finalize Mapping from the exact Dataflow and Fabric closure; it cannot preserve
the old Mapping identity or patch its text.

## Why Mapping Uses One Graded Diagnostic Channel

TechMapping and PnR failures often become visible far from their cause. A final
overflow alone cannot show which iteration accumulated the cost, which exact
capacity state was oversubscribed, which nets contributed claims, or whether a
reachable alternate path existed. Temporary print statements answer one case
but require rebuilding and tend to leave several incompatible debug paths.

One invocation-only JSONL logger therefore spans all three Mapping profiles.
It consumes the same Common-owned `LOOM_VERBOSE_LEVEL` binding as other Loom
diagnostics. A single numeric level is sufficient: summaries answer routine
performance questions, decision events reconstruct search progress, and
detailed exact owner/net/cut events diagnose topology or algorithm failures.
Independent category flags, subsystem-specific environment bindings, a
persistent statistics artifact, and profile-specific loggers were rejected
because they duplicate configuration and event ownership without adding
semantic capability.

The channel is deliberately outside ResolvedConfig and Mapping identity.
Turning it on may spend diagnostic presentation time, but cannot change search
work, ordering, random draws, or outcomes. Stable typed ordinals are more
useful and safer than raw SSA values, pointers, source paths, or container
dumps. Negotiated Routing owns the one exact fixed-terminal cut analysis used
for conditional non-closure; level-three logging exposes that result and its
reachable sets rather than creating a second diagnostic oracle.

## Why Compute Realization Replaces `dataflow.subgraph`

An FU can implement a small configured software graph. The selected actor
group, capability template, actor-to-op correspondence, boundary port
correspondence, and absorbed internal edges are target-specific facts. They
belong in a Compute Realization, not Canonical Dataflow.

The group has no independent semantics without its selected capability, so a
separate SoftwareActorGroup record was rejected. Memory actors use a parallel
Memory Realization because service, consistency, ports, and internal memory
relations are not FU compute topology.

TechMapping persists only non-derived selected relations. Exact actor
semantics remain in Dataflow, capability remains in Fabric, and configured
software projections are reconstructed from both. SpatialMapping then chooses
an occurrence of the same Fabric FU definition; it cannot re-group actors or
copy a configured graph.

The same definition-to-occurrence split applies to memory. TechMapping selects
one canonical Memory Operation Engine template and template-relative ports,
capability alternatives, boundaries, and internal connections. SpatialMapping
then chooses one concrete `fabric.mem` occurrence whose Fabric-owned template
relation matches exactly. This preserves the distinction between semantic
realization and physical placement without inventing a Mapping-owned memory
encoding or asking TechMapping to name an occurrence prematurely.

## Why Memory Placement And Rooted Uses Are Factorized

An `ActorRef` belongs to one reusable graph definition. The same graph may be
launched at several static sites, and each site may bind its memory formal to a
different logical root. TechMapping can still select one operation-engine
realization for the actor because its operation semantics and port contract do
not change. SpatialMapping must likewise select its physical operation
placement once, but memory binding and dispatch are properties of each rooted
use.

Cloning the graph per launch was rejected because it makes canonical program
identity and graph size depend on call-site specialization. Repeating a full
operation placement per contextual actor was also rejected: it duplicates one
physical decision and requires reconciliation when the copies disagree.
Instead, one actor-level MemoryOperationEntry owns placement and nested rooted
uses own binding and dispatch. The parent actor and child rooted launch derive
the existing Dataflow `ContextualActorRef`; Mapping adds no identity.

The same factorization is retained in native search state. Actor placement is
one dense array, while rooted uses occupy contiguous CSR ranges containing
launch, binding, and dispatch ordinals. Import and cold verification are
linear in actors plus rooted uses. A placement move visits only that actor's
use range, and a binding or dispatch move touches one use and its reverse
incidence. This avoids both graph cloning and a placement-by-launch Cartesian
domain while preserving every semantically necessary contextual decision.

## Why System Memory Targets Reference Fabric Transform Paths

A System memory target may reach its selected service region through an
address, interleave, or coherence transform chain. Fabric already gives every
transform a stable `SystemServiceTransformRef` and uniquely owns its closed
kind, parameters, input/output endpoints, and region correspondence. Copying a
signed offset or another transform payload into Mapping would create two
authorities for the same hardware behavior.

Persisting only one transform reference was rejected because legal Fabric
topology may compose several transforms. A Mapping-owned transform expression
or opaque typed value was also rejected because either would duplicate or hide
the Fabric contract. A separate Mapping transform entity would add identity
and lookup machinery without owning independent semantics.

`MemoryRegionTarget` therefore stores only an ordered path of existing Fabric
transform references when endpoint and target region do not already imply a
unique path. The empty path is the direct identity case. The verifier composes
the referenced Fabric contracts across Fabric-owned MemoryService connections,
checks endpoint continuity and address/range legality, and proves that the
final output reaches the selected region. A uniquely derivable path is omitted
so the same fact is not stored twice.

A single target region was rejected as the unit of an addressed decision
because `StaticInterleave` can send one logical interval to several terminal
regions. Introducing a strided-range language in Mapping would duplicate the
Fabric transform equation. Instead, one candidate target plan selects the
complete canonical set of terminal branches. Its `MemoryRegionTarget` children
may repeat the source interval; the Fabric transform and terminal region
derive which addresses belong to each branch. Collective verification proves
that every source address reaches exactly one selected branch. This keeps the
only interleave semantics in Fabric while still making the physical plan an
explicit Mapping choice.

Not every multi-output transform is collective. `CoherentMemory` owns a
partial bijection between equal-sized input and output regions, so an incoming
address maps by relative offset and may have several coherent provider
alternatives. Treating every reachable output as a required branch would turn
replicas into a broadcast obligation. CandidateState instead selects the
canonical subset of correspondence branches whose input domains cover the
source interval exactly once. The selected terminal output regions and any
necessary transform path identify that choice; correspondence members and
endpoint ordinals remain derived Fabric facts.

## Why Dead Results Derive A Physical Discard

Dataflow owns whether an actor result has consumers. A dead result therefore
cannot acquire a synthetic software edge merely because the selected physical
operation produces a token. Conversely, a disconnected ready/valid output can
backpressure the operation forever and cannot be treated as if the token had
vanished.

For example, a two-lane `dataflow.demux` may route a token to a result that the
selected program does not consume. TechMapping still maps both semantic result
ordinals to the exact operation ports. If the selected FU template exposes the
dead result at an FU boundary, SpatialMapping configures that concrete PE
output as `Discard`, which keeps ready asserted and consumes any selected
token. The live result alone receives an ordinary boundary correspondence and
route.

Persisting a `dataflow.drop`, a fake `mapping.compute_boundary`, or another
discard record would duplicate facts already owned by exact Dataflow, the FU
template, and the PE selector contract. Deriving the disposition from those
three owners preserves one authority for deadness, one authority for physical
topology, and one authority for occurrence configuration. It also keeps the
TechMapping independent of a concrete occurrence while requiring Spatial
closure to prove the real sink.

## Why TechMapping Uses Lazy Exact Cover

TechMapping rows are finite and exact, but the set of complete covers is not
generally small. Even a graph whose actors each have two independent legal
realizations has exponentially many covers. Requiring the generator to
enumerate all of them would turn an implementation accident into a semantic
obligation and would spend the expensive downstream Evaluation budget on
nearly equivalent materializations.

The generator therefore constructs complete typed rows, propagates forced
choices, factors independent incidence components, and returns a deterministic
finite prefix. This is still exact: every published candidate has complete
coverage and passes the independent verifier. The finite prefix limits the
invocation domain without relabeling unexplored alternatives as infeasible.

The seed domain starts at semantic membership rather than the Cartesian
product of every actor, every Fabric operation, and every port permutation.
OperationSchema and HSG already own whether an operation member and an ordered
port correspondence exist. Fabric memory operation ports likewise own the
canonical actor-contract, access, role, width, and operation-pattern relation.
Counting combinations those owners declare impossible would make a fixed work
budget depend on unrelated operations or memory ports added elsewhere in a
general-purpose Fabric. Reusing the owner projections keeps the candidate set
unchanged while reserving match-row attempts for complete prospective
persistent payloads whose remaining joint topology and capacity checks can
meaningfully pass or fail.

Global CP-SAT was rejected as the primary TechMapping search. The row problem
already has a small direct exact-cover model, while placing it in the same
solver as physical routing or later repair would duplicate Mapping state and
couple a semantic realization decision to a backend search mechanism. CP-SAT
is reserved for a closed local repair region after ordinary Spatial or System
search has identified a concrete conflict.

## Why Root-Complete Generation Uses Typed Adapters

The production TechMapping owner accepts an exact graph subset because two
independent regions of one Dataflow program may need separate mapping and
evaluation. Central DSE, however, commonly receives a finite set of complete
Dataflow candidates and needs one ordinary candidate set for the next plan
node. Making either use case the hidden default would erase the other.

The root-complete adapter resolves the difference by composition. It derives
the complete graph catalog from each exact Dataflow Artifact and invokes the
unchanged subset-capable owner. For example, a program containing canonical
graphs `g0` and `g1` produces the ephemeral scope `{g0, g1}` for one owner
invocation. A caller evaluating only `g1` still invokes the owner with `{g1}`;
it does not pass through the root-complete adapter.

Persisting `{g0, g1}` as another Artifact would duplicate a relation already
owned by the Dataflow root. Encoding it in generator config would mix
Artifact-local references with implementation policy. Letting the central
controller infer it for every mapping generator would make a domain rule into
global workflow semantics. A typed adapter has the smallest conceptual
surface: exact Artifact inputs remain the plan authority, Dataflow remains the
graph-catalog authority, and TechMapping remains the realization authority.

Application exploration cannot use the complete catalog implicitly. One
Canonical Dataflow Artifact may retain several externally callable definitions
while one selected application executes only a subset. Treating every retained
definition as simultaneously active produces false capacity conflicts; putting
entry names or graph lists in DSE config creates a competing scope authority.
The existing System MappingConstraintSet already owns the exact non-empty root
launch set. Application-scoped adapters therefore consume that root, derive
its reachable graph set through Dataflow, and otherwise delegate to the same
TechMapping and System PnR owners. A planner may derive the initial root set
from exact Spatial workloads, but workload values never enter Mapping identity.
This preserves one scope owner while keeping the complete-catalog adapter
available for whole-library analysis.

The following Spatial boundary has the inverse mismatch. The exact PnR owner
must accept an independently authored MappingConstraintSet, while the common
root-complete exploration path has no additional clauses. Omitting `K` would
make absence a hidden default and weaken the five-input identity contract.
Passing both `D` and `T` would duplicate an identity already sealed by `T` and
create a disagreement state. The Spatial adapter therefore consumes finite
`T` plus exact `F`, recovers the unique `D` from `T`, and publishes the real
empty `K(D,T,F)` through the existing constraint owner before invoking PnR.

For example, if `T0` binds `D0/F0` and `T1` binds `D1/F0`, one plan node over
`{T0,T1}` performs two ordinary invocations with `K(D0,T0,F0)` and
`K(D1,T1,F0)`. If `T1` instead binds `F1`, the node rejects it rather than
rebinding it to `F0`. If a user requires `compute_placement(actor) in {fu3}`,
that nonempty `K` bypasses the convenience adapter and enters the exact owner
directly. This preserves one constraint language, one PnR implementation, and
one source for every identity while still composing finite candidate sets.

Only an incomplete search over valid tuples retains candidates from earlier
canonical `T` inputs. A malformed Mapping profile, foreign tuple, or PnR
invariant failure makes the finite binding itself unusable, so the invocation
does not claim a partial output set. Objects already published by an earlier
tuple remain valid store objects, but treating them as outputs of the failed
invocation would collapse input validity into search incompleteness.

## Why Spatial Resource Time Uses Graph-Local Events

Spatial resource occupancy must distinguish actor firing from token production
and consumption. A mux transition, for example, consumes its selector and only
the selected data input; naming one operand alone would lose the atomic firing
relation. Conversely, a route reservation may begin when a producer publishes
a token and end only when an exact consumer observes it. Treating all three as
one generic event would move actor semantics into Mapping.

SpatialMapping therefore references the existing Dataflow actor and graph
terminal catalogs. Actor transition ordinals come from the exact
OperationSchema handshake projection, while produced and consumed terminal
events retain their direction. No event receives another EntityId, and no
symbol, operation position, simulator occurrence, or absolute cycle becomes a
persistent key. SystemMapping later rebases applicable Spatial activity onto
its Dataflow-owned boundary event families rather than copying this local
catalog.

The selected Fabric UsePattern remains the sole owner of timing, claims,
parameter positions, sharing positions, and any guaranteed-offset codec.
Mapping stores only the event-relative selection and owner-coded values. This
keeps custom resource types extensible without a generic property bag or a
second resource registry in Mapping.

Temporal operand queues need no new Mapping event. The incoming token already
has one exact consumer-terminal event, so that event is the enqueue boundary.
The actor handshake transition already identifies the atomic consumption of
its required inputs, so it is also the dequeue and operation-use boundary.
Using either event avoids an instruction-local scheduler, while attaching an
enqueue to the later transition would hide queue occupancy and attaching a
dequeue to an operand alone would lose atomic multi-input firing semantics.

Search builds dense event, use, and overlap-envelope tables once from these
records. Reverse CSR from a decision to affected envelopes makes a move cost
proportional to its incidence rather than to the whole mapping. Raw occupancy
and overuse stay in contiguous candidate arrays and are journaled
transactionally; canonical references, event relations, and owner codecs stay
in immutable frozen storage. These tables are disposable projections, so a
cold verifier can rebuild them from the exact artifacts and catch divergence
without making a performance cache authoritative.

## Why PnR Has Exact Immutable Inputs

Spatial PnR binds an exact Canonical Dataflow program, TechMapping, fully
elaborated Fabric, resolved search-policy view, and MappingConstraintSet. The
five-input coupling prevents a witness or constraint from being rebound to
similar hardware or a later program. System PnR uses the corresponding exact
system inputs.

Fabric must be elaborated before search. PnR cannot instantiate resources or
change capability. Mapping constraints are a separate Artifact because they
restrict the legal mapping space; ResolvedConfig controls how that space is
searched; the Mapping result records the selected solution. A generic `config`
object for all three would duplicate ownership and cache dependencies.

Constraint canonicalization reuses each reference owner's local comparison
wire rather than printed MLIR text. Printer spelling is presentation, can order
numeric values incorrectly, and would create another identity authority.
Likewise, an authored integer type width on an interval endpoint does not
describe Fabric width or capacity; preserving it would make equal value sets
produce different constraint identities without expressing a real distinction.

## Why PnR Receives A Selected Objective Closure

PnR needs objective records during hot mutable search, but the complete DSE
catalog contains models, obligations, and policies unrelated to one Mapping
invocation. Keeping global catalog ordinals in the PnR view would make those
unrelated records hidden semantic dependencies. Keeping the complete DSE digest
would avoid stale references but would destroy cache reuse whenever an
unselected record changed.

The resolved PnR view therefore contains the selected transitive closure with
view-local references. DSE still owns every obligation template, dimension,
quantization, weighted level, and ordering. PnR owns only which closed records
this search consumes and the exact interaction-domain relation needed to use
them. Spatial and System search use the same record algebra but distinct view
descriptors, so they may choose different policies without sharing or copying
mutable search state.

Evaluation interaction modes are derived from use. A metric that participates
in mutable candidate ranking needs the exact incremental protocol; route
guidance needs the guidance protocol. Storing a second mode flag beside the
binding could disagree with the use and was rejected. Likewise, selected
guidance cannot silently become zero when its provider fails: absence is an
explicit policy choice, while failure of a selected authority is a failed
Action or invocation.

The PnR freeze cache consequently depends on exact program, realization,
hardware, and constraint identities plus the exact PnR view descriptor and
digest. It does not depend on the complete ResolvedConfig identity. The
InvocationManifest still records that full identity, preserving provenance
without poisoning component-local cache reuse.

## Why Persistent MLIR And Native Search Data Coexist

Mapping MLIR is the typed persistent wire and independent verification
surface. MLIR operations, attributes, strings, JSON objects, and recursive
references are inappropriate for millions of hot search accesses. Import
therefore validates exact references once and freezes a removable native model
with dense indices, SoA/CSR storage, precomputed endpoint domains, adjacency,
and distance/connectivity caches.

This is not a second authority. The frozen model is rebuilt only from exact
artifacts and config views, has no persistent identity, and cannot write search
decisions back into the inputs. Index width is a centralized compile-time
choice; overflow fails with a diagnostic requesting a 64-bit rebuild rather
than silently truncating.

## Why Endpoints And Traversals Are The Routing Atoms

Resource-level adjacency loses port direction, kind, width, switch traversal,
and boundary behavior. RTL bit graphs are too low level. PnR routes over a
directed graph of typed Fabric endpoints and explicit traversals.

Placement selects a compatible FU occurrence but does not preselect a complete
PE-port assignment. Each software endpoint retains a factorized domain of
compatible physical endpoints; route search jointly selects endpoint and path.
This avoids a Cartesian explosion and lets congestion inform attachment.

Graph boundaries need a related but distinct domain. They have no Compute or
Memory occurrence, so treating them as `PortDemand` values would require a
fake placement unit and would make boundary identity a second topology owner.
The finalized Module already owns the exact correspondence from each
resource-attached signature boundary to its occurrence-local endpoint. An
unused boundary or direct boundary passthrough supplies no placeable routing
terminal and therefore no row. Freeze filters the relation by the graph
terminal's direction and payload requirements, while RouteTree uses only the
resulting physical endpoint. SystemMapping later composes the same Module
boundary correspondence with the selected AccCore SpatialCore occurrence
binding and its per-boundary endpoint attachment.
This keeps one boundary identity owner and one physical routing graph without
making Module boundary references routable resources.

A residual logical net represents exactly one canonical producer and all
unabsorbed sinks. Multi-sink routing builds one route tree with explicit
replication resources. A broadcast branch that reconverges without logical
work is redundant program structure to optimize, not a reason to relax tag or
endpoint constraints.

One scalar or vector actor port is likewise one routing demand. Routing lanes
independently would add lane identities absent from Dataflow, multiply route
trees, and let Mapping change firing atomicity. Every selected segment must
therefore carry the complete semantic token. For a connection between
different physical widths, the smaller endpoint width is the usable capacity;
a later wider segment cannot recover discarded high bits. This derives route
legality from the existing endpoint types without adding an adapter or a
second width field to Mapping.

## Why System Mapping Selects But Does Not Own Service Carriers

System PnR must route memory requests, responses, and fences over transport
endpoints, but the legal correspondence from a memory-service endpoint to
those endpoints is a hardware-topology fact. Copying it into Mapping would let
two mappings disagree about the same Fabric and would force Mapping identity to
repeat capability and payload facts.

CandidateState therefore follows its selected Module-local manager path
through the AccCore occurrence's exact Fabric memory `spatial_attachment`.
That mechanical projection yields the fixed Module/occurrence endpoint pair
and System service endpoint; it is not another candidate choice. The System
endpoint is the pair's one capability authority. Canonical leg direction and
source or sink position choose one pair member, whose existing carrier row
supplies only that terminal's transport candidates. The frozen search domain
indexes target compatibility by the exact System endpoint and terminal
compatibility by the exact selected pair member. CandidateState chooses only
ordinary transport terminals and traversals. The persistent RouteTree records
only the selected route, and final verification repeats the same derivation
from the exact Fabric. An incompatible member makes all affected rows for only
that candidate's bound pair empty; it does not authorize a capability search
for another endpoint. Fabric root finalization cannot own this workload
comparison because it does not know the selected Dataflow member or Mapping
decisions.

`MessageTransfer` needs no projection because its service endpoint already is
a transport endpoint. Protocol subchannels and encodings remain a later
Interconnect Implementation refinement, so neither PnR nor the Mapping
artifact becomes a hardware-topology or protocol owner.

The same boundary applies to service targets. Mapping may choose a compatible
region or consistency domain exposed through the bound endpoint's explicit
service/transform closure, but it cannot use capability equivalence to jump to
another endpoint. This keeps optimization choices in Mapping while the
physical association remains one Fabric fact.

The relation cites rather than copies the Dataflow-owned structural ordinal
domain and the existing `CanonicalServiceLegKey` framing. A Fabric-local leg
integer or codec would create a second authority for the same schema-local
position without adding a hardware distinction.

System routing needs one scalar width bound even though a service leg contains
several ordered values. Reusing the Fabric-owned maximum-width envelope gives
the route search exactly the strongest per-value capacity requirement without
making Mapping own a tuple layout or one route per role. Each value still obeys
the no-split rule independently, so taking the maximum cannot legalize a path
narrower than any value in the leg.

Deferring carrier-width legality entirely to an EDA or interconnect backend
would let Mapping accept an unrealizable route and discover the failure only
after search. Route selection must therefore prove structural representability
while protocol realization remains backend-owned.

## Why Placement And Routing Are Coupled

Greedy placement followed by one routing pass cannot escape congestion,
capacity, or temporal-context interactions. Early experiments showed that
simulated annealing with only single-unit moves and static-distance cost simply
reproduced greedy solutions even with many moves and restarts. Transactional
swaps, endpoint-aware routes, capacity proxies, negotiated congestion, and
closure actions are necessary neighborhoods and feedback.

The production model couples placement, endpoint selection, routing, tags,
buffers, memory binding, resource use, and refinements through typed Actions
and atomic MoveTransactions. A move either re-closes all affected obligations
or rolls back. Strict place-then-route remains expressible as an action policy;
it is not a second solver or Mapping schema.

## Why Arbitrary Topology Uses A* And Negotiation

Mesh coordinates and Manhattan distance cannot define route legality for an
arbitrary directed Fabric. Endpoint-only A* uses exact connectivity and an
admissible topology-derived heuristic. Multi-net contention then requires a
negotiated router such as the specified PathFinder or dual-subgradient kernels.

Negotiation state, prices, occupancy, and best iterates are native scratch,
not Mapping fields. Stable iteration ordering, checked numeric protocols,
canonical aggregation, and deterministic work units make randomized or
parallel search replayable. Budget exhaustion is inconclusive, never proof of
infeasibility.

Bounded exact repair is a final focused solver for a small unresolved closure,
not a global replacement for annealing and routing. It consumes an explicit
bounded problem and cannot publish a best-so-far invalid Mapping.

Generic sink-subset proposals were rejected. Their powerset is large, their
container representation is unstable, and they add no essential routing
operation. Whole-net, one-sink, existing-subtree, witness-closure, and global
scopes compose to express the useful neighborhoods while retaining a finite
typed action algebra.

OR-Tools remains a required pinned implementation of bounded exact repair, but
its internal time estimates, incumbent order, and solution-pool order are not
Loom semantics. Only proof-bearing optimum and infeasible results may affect a
candidate. Canonical variable fixing after a proven optimum prevents an
upstream solver heuristic or version-internal ordering from selecting a
different persistent Mapping. A local infeasible repair region proves only
infeasibility under its fixed boundary; it is not global proof unless the
region is the complete exact problem.

The adapter uses one worker and one restart-local seed for the entire repair
invocation. Consuming one ExactRepair word rather than one word per solver call
keeps the canonical fixing protocol from making later variable choices depend
on how many earlier values were infeasible. The low 31-bit projection matches
OR-Tools' signed seed field without implementation-defined narrowing. Solver
time and search counters remain execution behavior rather than semantic work;
only Loom's region-decision and solver-call budgets can alter the formal repair
result.

## Why Selected Handshake Legality Is Incremental

Every placement, route, refinement, or sharing move may change a small subset
of the selected combinational handshake graph. Rebuilding and sorting the whole
graph after each move multiplies a linear validation cost by the annealer's
large move count. Materializing boundary transitive closure is worse: atomic
broadcast can make one physical owner quadratic before search begins.

The Frozen model therefore stores one compact owner-local potential graph and
typed reverse incidence. A candidate stores only arc reference counts, an
active bitset, and topological order. Reference counts preserve the fact that
several selected decisions may activate one physical dependency without
duplicating arc identity. Dense arrays and preallocated epoch scratch keep the
hot path local and allocation-free.

Array-based Pearce-Kelly updates were chosen because most moves touch a small
rank interval, rank-respecting insertions are constant time, and the algorithm
uses the same compact adjacency already required by PnR. More elaborate
worst-case dynamic algorithms add substantial state and update machinery
without changing Mapping semantics. Full Kahn or SCC recomputation remains the
simple independent authority at initialization, global repair boundaries, and
final verification.

This is an SSOT split rather than a second graph: Fabric owns the equations and
owner-model fragments; Frozen state mechanically flattens them; Candidate
state selects fragments; the final verifier discards every incremental cache
and derives the graph again. Performance gates reject an inefficient
projection, but never change candidate legality or search semantics.

## Why Legality And QoR Are Separate

Mapping's local solver cost can measure domain-independent facts such as
unrouted nets, capacity violation, distance, congestion, and constraint
violation. Accelerator runtime, frequency, power, memory behavior, system
performance, and any future registered energy metric belong to shared
Evaluation models. Central policy combines these
observations; PnR must not grow a private accelerator cost model.

Some Fabric configurations preserve function but change performance, such as
FIFO bypass. Mapping selects them, but Evaluation supplies the relevant cycle,
timing, power, or bandwidth feedback. This preserves one metric and evidence
system across compiler, mapper, and hardware DSE.

The same separation applies inside resource accounting. Fabric integer claims
and capacities decide legality exactly. Normalized fixed-point values are
useful for comparing pressure across unlike resources, but rounding each claim
up can make several individually small claims appear to exceed capacity. Those
values therefore order search only; they cannot create a capacity violation.

## Why Mapping Has One Capacity Violation

A pipeline slot, FIFO entry, operand queue, enqueue port, memory-service
outstanding slot, and routed transport reservation have different physical
meanings, but each concrete Fabric owner already expresses its exact integer
capacity, state transition, atomic use, and event-relative lifetime through one
`ResourceContract`. Mapping selects those uses and derives their canonical
concurrent occupancy. A second Mapping taxonomy of timed, buffered, and service
resources would therefore classify a fact that Fabric already owns.

The classification also has no total physical meaning. A registered result
slot is both pipeline state and holding storage. A temporal operand-buffer
enqueue claims a per-cycle service slot while atomically changing durable queue
occupancy. A memory `UsePattern` may claim request tracking, a port, and
response holding in one indivisible envelope. Assigning one global label is
arbitrary; assigning several labels counts the same atomic use several times.

A Temporal switch makes the lifetime distinction concrete. Its input/output
UsePatterns specify per-cycle service and arbitration, while its route table
specifies which tag-keyed routes may reside simultaneously. Summing every
resident route's service claims would falsely prohibit legal time sharing;
ignoring route-table rows would accept hardware that cannot hold the Mapping.
Fabric therefore projects runtime service and Mapping residency separately,
and both feed the same final `CapacityOveruse` only at their actual lifetime.

Mapping consequently owns one `CapacityOveruse` magnitude over exact raw
capacity queries. Owner-typed witnesses retain the detail required to report or
repair an operand queue, service slot, route, or operation stage without making
those diagnostic names independent objective sources. Search guidance may use
Q-scaled pressure or Evaluation metrics, but only raw Fabric capacity decides
legality. Permanent closed waits remain a separate progress proof because a
design can respect every capacity and still deadlock.

For progress, an acyclic canonical actor dependency graph gives a small exact
basis, but it is not sufficient by itself. An atomic multicast can feed both a
prerequisite actor and a causally downstream actor. If the downstream branch
cannot accept until the prerequisite retires, requiring both branches to
accept the multicast at once forms a closed wait even though the actor graph
and selected combinational handshake graph are acyclic. A Buffered FIFO after
the route divergence lets the atomic fork retire into storage; buffering the
shared prefix does not. Mapping therefore owns one derived sink-dependency
projection and applies it both during PnR and final verification. After that
route closure and Fabric atomic progress are verified, topological induction
always exposes a next actor under fair execution. Extending the proof to
feedback requires typed initial tokens, finite-buffer occupancy, and wait-for
relations. Treating every cycle as a deadlock would reject ordinary streaming
loops, while treating it as safe would hide real closed waits. Failing closed
for unsupported cyclic proofs therefore preserves one progress authority
without inventing either answer.

The earlier eight-entry objective registry was removed rather than retained as
aliases. Aliases would allow central DSE to weight one physical counterexample
several times and would force Candidate state and final verification to agree
through reconciliation code. A major schema transition is smaller and more
honest than preserving unimplemented categories as permanent compatibility
surface.

## Why Routing Cost Uses One Q-Scaled Algebra

Raw claim sizes cannot rank unlike Fabric resources. A claim of 32 units from
a capacity of 64 and a claim of 4 units from a capacity of 8 both consume one
half of their resource. Pricing the raw integers would prefer the second claim
for an accidental unit-size reason. The fixed `Q = 2^32` projection gives both
claims the same search cost while leaving exact raw integers in sole control of
legality.

Once a value is Q-scaled, every multiplicative factor must preserve that unit.
Treating `Q` as the integer one causes an ordinary full-capacity claim and one
full-capacity conflict to overflow immediately. PathFinder, conflict ordering,
DualSubgradient, and optional Evaluation route guidance therefore share one
checked algebra. Multiplicative PathFinder applies one final ceiling to the
complete product; staged ceilings are rejected because even a one-unit change
can alter deterministic A* ties.

Dual residuals normalize the signed aggregate difference once, rather than
summing rounded per-claim projections. Rounding the nonzero magnitude upward
preserves its sign, while normalization makes the same proportional violation
produce the same pressure on unlike capacities.

The hot representation remains `uint64_t`. Floating point would make replay
depend on host arithmetic. Dynamic scaling would add an iteration-owned
exponent and another ordering rule. A 128-bit cost would double the principal
A* cost arrays while only postponing, rather than eliminating, unbounded price
growth. Checked overflow is consequently an explicit failed Action, not
saturation, infeasibility, or permission to change representation.

An exact capacity cut is the necessary boundary between inner routing and
outer Mapping search. Continuing PathFinder price growth after two fixed nets
are each proven to require the same unit-capacity switch input cannot discover
a route; it only delays the resource or attachment Action that can change one
of their terminals. Treating that condition as global Fabric infeasibility is
equally wrong because another endpoint assignment may remove the cut.

The certificate therefore removes every traversal claiming one capacity from
the frozen payload-compatible graph and proves a selected sink unreachable.
Summing the minimum positive raw claim for each proven net gives a conservative
lower bound: ignored constraints can only add restrictions, and alternate
claiming traversals cannot consume less than that minimum. Exceeding physical
capacity proves only the current fixed-terminal routing subproblem cannot
close. Returning the best policy-admitted temporary iterate lets the existing
outer Action algebra repair the assignment without a new restart mechanism,
new violation kind, or numeric exception.

The exact cut and the no-progress work bound answer different questions. The
cut proves that one fixed-terminal routing subproblem cannot close. The
no-progress bound proves nothing about future iterations; it only prevents an
explicitly bounded search from repeatedly spending the dominant all-net route
work without improving the Mapping-owned selected rank.

Comparing only against the historical best gives useful patience, but a short
regression can be necessary before negotiated routing improves again. A second
exact comparison against the immediately previous rank captures that local
direction without inventing another score. The trailing sign window keeps a
net-improving recovery alive and stops flat, worsening, or balanced oscillating
search only after the best-rank patience is also exhausted. Reusing
`SelectedObjectiveClosure` avoids a second convergence score, while excluding
candidate-key tie breaks prevents equal-quality route churn from resetting
either signal. Returning `NoProgress` as work exhaustion preserves the
distinction between search incompleteness and infeasibility, and still lets a
non-final Action consume the retained temporary iterate through the existing
repair algebra.

The complete dynamic-cost baseline is derived once from the complete route
overlay at an iteration boundary. Candidate-local changes reuse Fabric's raw
occupancy owner and the frozen capacity-to-claim-to-traversal-to-arc incidence
to update only affected costs. The cost cache does not copy per-net claim
ownership and can be deleted and rebuilt without changing the candidate. This
keeps one authority while allowing contiguous, allocation-free A* hot paths.

Objective facts also do not imply one universal scalar. Final ordering, Pareto
dominance, and annealing energy ask different questions. They share the same
central dimensions and exact normalized codes, but derive a lexicographic
ordering, a componentwise vector relation, or one selected weighted energy.
Changing an unrelated dimension bound must not rescale the acceptance
probability of a local search move.

The minimum-temperature level is explicit for the same replay reason. Stopping
as soon as cooling reaches the minimum can either skip that level or execute it
twice depending on loop shape. Executing exactly one complete level at the
minimum gives one finite rule without a second temperature-level budget or a
host-time termination heuristic.

An exact all-zero final violation vector is different from heuristic no
progress. It is already the condition the independent final verifier requires,
so calibration or further equal-energy rerouting cannot improve legality and
can only spend work or leave the solution. Checking that owner-derived fact on
entry and after an accepted Action removes the random walk without a visited
set, candidate-history state, timeout, or second convergence rule.

The same candidate can also propose the same deterministic reroute many times.
Re-running A* cannot change its result while the selected decisions, route
trees, and derived costs are unchanged. Remembering only exact self-loops and
typed failures for that candidate removes repeated graph work without banning
an equal-energy but physically different route. Clearing the dense key cache
on the first accepted semantic change avoids a history-dependent tabu rule and
keeps the cache a removable execution optimization.

The annealing level length counts movable decision owners rather than Action
choices or routing neighborhoods. A decision with 100 alternatives should
receive more varied proposals through its choice domain, not make the whole
search 100 times longer. Whole-net and local routing scopes likewise describe
ways to modify the same residual-net decision; counting every scope would make
the schedule depend on implementation-owned neighborhood richness. Anchoring
the count to canonical selected decisions keeps work proportional to problem
size and stable as search tactics improve.

Initializer dependencies are explicit because treating an unavailable
occurrence-relative domain as an empty domain would falsely prove
infeasibility. Realization and graph-boundary roots can be compared directly by
MRV. Completing that hard-relation root model before activating dependent
choices prevents a provisional occurrence from exposing an attachment domain
that relation propagation later removes. Port, memory-plan, dispatch, and
exposure choices enter the dependent MRV only when their owner references can
be resolved. This keeps one generic MRV and rollback protocol without inventing
placeholder choices or a retry loop per decision kind.

An instruction context is a resident dispatch slot, not a runtime resource
claim. A Spatial PE has one context and one active FU selection. A Temporal PE
has one context per instruction-memory row and may keep several FU selections
active. In both schedules, two independent Compute Realizations cannot occupy
the same context: doing so would require an unmodeled instruction selector and
would alias their configured graph and dynamic actor state. ResourceUse owns
pipeline, queue, endpoint, and service occupancy after a realization is
resident; it cannot create another resident slot. The initializer therefore
projects every compute choice through its exact `InstructionContextRef` and
places all Compute Realizations in one hard `Disjoint` relation. A relation
whose member count exceeds its exact value union is a zero-search pigeonhole
proof.

The relation is intentionally over Compute Realizations rather than raw
actors. TechMapping may prove that several actors and their internal edges form
one FU realization, in which case that group consumes one context. Conversely,
configuration equality or event separation cannot merge independently selected
realizations. This preserves actor grouping without inventing temporal sharing
for Spatial PEs, while a Temporal PE exposes exactly the additional residency
that its `num_instruction` hardware declares.

Distinct resident contexts alone are not sufficient for a useful seed. A chain
placed once on every PE of a bounded one-dimensional mesh can still follow a
topology-independent
occurrence permutation. About half of its logical nets then cross the same
middle cut even though a neighboring placement exists. The locality preference
among still-unoccupied contexts therefore uses minimum directed hop distance
to active topology anchors. Already processed compute neighbors provide the
recurring anchor.
Directly incident graph-boundary domains also anchor a graph's first compute
root; without that anchor, a chain can begin in the middle of a bounded path,
walk to one end, and require one schedule-wide jump to reach the unused half.
The score is derived from the existing frozen attachment domains and
payload-compatible routing arcs, with no coordinate metadata, premature
boundary selection, or mutable congestion term. It is a cheap placement
preference, not a shadow capacity model, a second routing search, or an
infeasibility certificate. An independent placer would duplicate relation and
routing ownership. The locality refinement instead composes the two existing
facts needed for a useful seed: exact resident-context identity and exact
frozen connectivity.

Schedule is also a physical seed property rather than an interchangeable
placement label. A Temporal placement terminates the configured instruction
and operand paths in resident state, while a Spatial placement may leave the
selected local endpoint, FU, and switch path in one combinational handshake
component. A relation-satisfying assignment that begins with Spatial
occurrences merely because they have lower canonical ordinals can therefore
create a handshake cycle before the annealer receives its first Candidate.
Preferring Temporal occurrences for independent roots uses the schedule already
owned by Fabric to seek a stateful initial witness. It does not make Temporal a
hard requirement: constrained roots retain their witness, every Spatial choice
remains in the Candidate action domain, and final cost and legality remain
schedule-neutral except for the resource contracts those placements actually
expose.

Memory roots need the same treatment. Canonical memory enumeration groups
occurrences by construction order, so retaining the first ordinal can bind
unrelated roots to one Spatial bank even when stateful Temporal banks are
available. The initializer first prefers Temporal memory and then distributes
equal-schedule choices by exact occurrence selection count. That count is a
seed heuristic, not a capacity claim: sharing remains legal, and the resource
model and router still own the consequences of sharing. This is the minimum
additional ranking needed to prevent enumeration order from acting as an
undocumented architecture policy.

The hard relation solver's baseline assignment is a feasibility witness, not a
physical ranking. In the canonical restart, using each baseline ordinal as the
locality tie origin would let propagation order permute otherwise equal
contexts and change routing work. Canonical preference therefore begins at
owner-local ordinal zero. Later restarts begin at their diversified baseline,
so variation remains deterministic and isolated to the declared initializer
stream.

Attachment roots expose the same issue one level below placement. If two
inputs or outputs of `dataflow.sync`, `dataflow.mux`, `dataflow.demux`, or
another variadic operation both take the first occurrence-relative attachment,
the initial route duplicates a physical endpoint even when the selected FU and
PE expose enough distinct ports. Enumerating port permutations is
functionally redundant, but collapsing simultaneous logical terminals onto one
endpoint is not. Opcode-specific permutation tables would duplicate operation
semantics, and hard endpoint disjointness would incorrectly reject legal
temporal sharing. The initializer instead prefers the least-selected exact
endpoint in every unconstrained occurrence-relative root's selected-placement
domain.

Graph boundaries have an additional essential distinction: they are the
remaining endpoint of an already selected logical-net terminal relation. A
pure load-balancing rule can select an unused but disconnected boundary face,
or a distant face that creates avoidable cut pressure. Their preference
therefore uses the frozen directed topology first: minimize unreachable
opposite terminals, then total hop distance, and only then endpoint selection
count. This is not a second attachment mechanism; both root kinds retain their
existing legal domains, constrained baselines, and the same relation-solver
validation. It is the minimum topology-aware ranking needed to avoid creating
an invalid seed while keeping endpoint counts out of capacity authority.

Local-memory byte offsets need a finite search representation. Enumerating
every fitting byte would make a 4 GiB region contribute billions of choices
even though, under the current containment and non-overlap contract, empty
gaps carry no meaning. Canonically left-compacting bindings on each selected
target preserves every feasible target assignment: any non-overlapping set of
finite intervals can be shifted left in canonical binding order without
changing target, dispatch, exposure, or capacity facts. The Candidate still
stores the exact resulting offset, while the initializer avoids turning
address-space size into search complexity.

The search PRNG and acceptance test use fixed integer protocols because a
named engine without exact seed framing still permits implementations to
diverge. Domain-separated SHA-256 gives every seed and purpose an independent,
schedule-independent starting state. Fixed-endian state loading and Loom-owned
rejection sampling remove host-library distribution behavior from replay.

Acceptance likewise uses one checked-in Q64 table rather than runtime `exp`.
Different libm implementations, floating-point contraction, and rounding modes
can change a threshold by one and therefore change every later candidate. The
table is finite because the Q64 threshold eventually becomes zero; its byte
digest and a few boundary anchors detect accidental regeneration or editing.
The mathematical exponential explains the intended probability, while the
integer table remains the one executable truth.

## Why Tags Are Local Allocation

A Physical Tag distinguishes logical uses that may overlap on the same tagged
match domain. It is not a software event ID, channel epoch, or globally unique
label. Nets with no possible physical competition can reuse a value; competing
uses require distinct values within the declared width.

Tag coloring is therefore derived from selected routes, contexts, resource
sharing, and overlap. Tagged endpoints may carry several logical nets only
under that proof. Untagged endpoints remain single-use for overlapping
transfers.

Persisting one value at each maximal continuity-segment origin is sufficient.
A PE or memory result creates the tag under its realization owner, a graph
ingress creates it at the route root, and a boundary writer or rewriter creates
it at that route node. Every downstream switch match, lookup row, and tagged
transport endpoint is a mechanical projection of that value until a remover or
rewriter ends the segment. Persisting those projections would create several
editable copies of one decision.

The assignment is a separate stateless Fabric UsePattern because adding it to
an operation, queue, or transfer pattern would duplicate that pattern's claims
and commit transition in Mapping. The endpoint-owned assignment pattern carries
only one typed sharing slot, so the existing resource behavior remains intact
while the value still has an exact owner codec. This also lets strict import
rebuild local interference directly from routes and reject collisions without
trusting PnR's removable coloring cache.

Tag width and resident table depth are different hardware axes. A wide tag can
name many values without allocating `2^T` rows, while a deep table can retain
only its declared number of continuity segments. Keeping row capacity on the
Fabric match domain lets rerouting change residency without creating another
tag namespace or duplicating switch configuration. The configured-hardware
codec then groups traversals by the one origin value and independently checks
the derived table during strict Mapping import.

## Why Mapping Owns The Configured Hardware Projection

TechMapping selects exact software-to-capability semantics, SpatialMapping
selects physical occurrences and contexts, and Fabric owns each typed
configuration-field domain. Only their complete relation can determine which
semantic value one physical configuration slot must hold. Giving that
derivation to CGRA-sim, RTL lowering, or configuration-image code would make
each consumer a competing Mapping verifier.

The complete Mapping verifier therefore derives the projection once as a
sealed, removable view. Equal demands on one slot collapse; conflicting demands
invalidate the Mapping before execution or encoding. CGRA-sim uses the view as
a cold admission witness, while configuration-image finalization sends the same
semantic values through the ABI-owned physical encoder. Keeping the projection
ephemeral avoids copying configuration into Mapping without weakening the
single validation boundary.

Instruction contexts qualify resident compute configuration because one
temporal occurrence may retain different settings in different contexts.

SpatialMapping cannot own a System-qualified slot because its exact hardware
binding is a reusable Module. Keeping its projection Module-local preserves
that reuse. SystemMapping already owns the execution binding that selects each
physical SpatialCore occurrence, so qualifying the complete local slot there
is mechanical and introduces no new decision. Qualifying only the field would
lose the residency invariant; copying values into SystemMapping would create a
second configuration authority.

Residency is part of the Fabric-qualified slot because a raw
`InstructionContextRef` is neither universal nor optional by convention.
Static route tables and component modes have no PE context, while a Temporal
PE may physically bank its FU configuration per resident instruction. The
closed `Static | InstructionContext` residency selected from immutable Fabric
shape represents exactly that distinction. A sentinel context would alias a
real context ordinal, and making every field context-indexed would duplicate
shared switch, memory, and per-FU state.

Physical refinement does not imply one universal value language. Pipeline
insertion, a transport-specific handshake break, and a memory implementation
choice have different legal domains and semantic-preservation proofs. Encoding
all of them as opaque bytes would make Mapping or a backend the accidental
owner of their meaning. Until a concrete Fabric resource publishes its closed
typed codec, rejecting a nonempty refinement assignment before configured
projection is therefore stricter and smaller than introducing a generic
refinement schema that later owners would have to escape or supersede.

## Why SystemMapping Is Required

Binding threads to AccCores and channels to the system transport is not runtime
policy left after SpatialMapping. It determines which heterogeneous cores run
which logical points, which SpatialMapping each rooted graph launch uses, and
how ordered cross-kernel streams and services traverse the NoC.

The same kernel's thread points may be distributed among AccCores, while
different kernels may occupy distinct subsets and pipeline through channels.
Execution binding and channel routing must be solved together because a
software schedule may or may not permit FIFO streaming. Runtime only admits
and authorizes the immutable selected resources; it cannot remap them.

Hierarchical System PnR reuses immutable SpatialMappings. A flattened global
mode remains a search option, not a different artifact authority. Both produce
the same SystemMapping schema and face the same final verifier.

## Why System Service Uses Reuse Dataflow Actor Transitions

System service occupancy needs a rooted event for each addressed-memory or
fence actor issue. The existing system transfer-event domain covers thread,
graph, and channel boundaries, but an internal graph actor is not one of those
terminals. Choosing an arbitrary operand consumption would make operand order
an accidental lifecycle authority, while reserving the provider for the whole
graph launch would turn request capacity into coarse launch residency.

Dataflow already owns the essential fact: the OperationSchema
`ActorHandshakeCase` whose commit is one memory issue. `EventFamilyKey`
therefore composes the existing static transfer reference with a rooted
contextual actor-transition reference. Mapping stores only that reference;
the Canonical Dataflow memory contract keeps issue semantics, and Fabric keeps
UsePattern timing and completion. This adds no Mapping-local request ID,
service state machine, or copied event projection.

The same ownership split explains why a memory exposure creates no use on its
own. It identifies that a memory capability crosses a graph boundary, not that
an access occurred. Treating exposure as a graph-long reservation would reject
otherwise legal overlap and charge resources for programs that never issue an
external access. A future external-access workflow must first define its own
invocation event domain; SystemMapping cannot manufacture one.

Imported Spatial uses retain graph-local producer and consumer endpoint event
families. An endpoint can participate in several mutually exclusive
OperationSchema handshake cases, so rebasing it by selecting one transition or
conjoining all transitions changes behavior. The Canonical Dataflow Program
instead derives the complete alternative set of existing rooted transition or
boundary events. System verification and admission preserve each original
release conjunction as an `AllOf` of per-endpoint `AnyOf` sets, adding no
Mapping-local event catalog.

Service ResourceUse points at the natural key of an existing ServicePlan
child. The selected Fabric use-pattern reference records the only additional
choice. Persisting a capability ordinal would duplicate a match that can be
recomputed from the bound endpoint, actor contract, target region, and chosen
pattern. Persisting a branch predicate would similarly duplicate the selected
Fabric transform. Deriving both keeps one semantic owner and still allows PnR
to compare genuinely different admissible patterns under capacity pressure.

## Why The System Search Domain Does Not Select Targets

System binding relations can cover large Presburger domains or finite dynamic
stable-key domains. Expanding every point is impractical, while storing a
chosen target in the search-domain view would duplicate the candidate and
eventually the SystemMapping.

The immutable `H` view therefore owns only a complete partition into typed
atoms, legal hierarchical targets or flat reopen domains for each atom, and
factorized compatibility relations over existing software subjects and exact
Fabric endpoints. CandidateState still selects `B_thread`, an immutable
`B_graph` target or flat Spatial decisions, and service targets. Those choices
mechanically derive the endpoint used to query the relation. Finalization
merges equal-target atoms into the existing persistent binding relation. This
permits block, cyclic, affinity, and stable-key grouping without a new
schedule, predicate language, shadow mapping, or provisional identity.

Putting transform paths into `H` was rejected because paths are compositions
of the already frozen Fabric topology, while H owns only endpoint-indexed
compatibility. It would duplicate topology and require another search-domain
schema whenever Fabric gains a compatible connection form. Choosing a region
first and asking the materializer to find a path was also rejected because the
materializer would become a hidden solver and ambiguous physical plans would
collapse. CandidateState instead derives the finite complete target-plan domain
from `F + H`, then selects one exact plan. This is the same ownership split as
route selection: immutable hardware and compatibility define the legal domain;
the candidate owns the choice; finalization only persists non-derived parts.

Endpoint factorization is necessary when one Module and one SpatialMapping are
reused by several AccCore occurrences. One occurrence boundary may attach to
an endpoint that accepts a read while another attaches to an endpoint that
does not. A global union would let the latter borrow the former's capability;
an intersection would reject the former; selecting either endpoint while
building `H` would preempt `B_thread`. Indexing the atomic compatibility fact
by the already existing bound endpoint distinguishes those cases and lets two
contexts correctly share a row when they really bind the same endpoint.

An additional partition-scoped service-constraint atom would have no semantic
owner. The System MappingConstraintSet restricts service choices through its
existing obligation and transfer-terminal subjects, while Fabric owns endpoint
capability and attachment closure. `H` therefore applies the exact resolved
constraint while deriving each subject-and-endpoint compatibility row. The row
is a disposable projection of those two authorities, and its digest changes
when either input changes. CandidateState performs one exact lookup instead of
reconciling a shadow constraint domain with a compatibility domain. If a
future contract needs service restrictions that vary independently by
execution partition, the MappingConstraintSet projection catalog must first
own that distinction explicitly.

Using the persistent `ExecutionContextKey` instead would require the complete
immutable SpatialMapping identity. That identity exists for hierarchical
imports but not while flat System search is still changing Spatial decisions.
Creating a temporary identity would add rekeying and invalidation authority;
freezing SpatialMapping first would remove joint flat search. `H` 3.0 needs
neither. Flat candidates derive their bound endpoints directly, and only
successful finalization creates the ordinary `ExecutionContextKey` used by
SystemMapping and Deployment.

Keeping partition shape target-free also preserves software ownership. A
search policy may choose how coarsely to group a logical may-domain, but it
cannot alter coordinates, extents, launch parameters, channel correspondence,
or introduce physical coordinates. Different shapes are different resolved
invocations, not mutable hidden state within one run.

## Why Service Plan Selection Uses Anchor Plus Context

A ServicePlan can contain an obligation-wide route or service-target choice,
while the legal physical endpoints may vary across the logical domain. A root
thread can execute on one AccCore for one relation cell and another AccCore for
another cell; a channel producer and its consumers may also execute in
different contexts derived through `source_map`. One obligation-global route
would either borrow endpoints across owners or reject a legal binding.

Mapping therefore keys plan selection by an existing Dataflow-owned service
anchor plus the existing finalized `ExecutionContextKey`. The anchor supplies
the logical relation domain; the context is rederived from `B_thread` and,
where applicable, `B_graph`; the relation selects only a canonical owner-local
plan ordinal. Consumer contexts remain derived through Dataflow correspondence.
This avoids a new producer-plus-sinks context tuple, which would copy the sink
set and `source_map`, and avoids storing endpoints beside the RouteTree, which
would duplicate Fabric and route facts.

Mutable flat PnR uses its existing decision atoms until immutable
SpatialMapping identities exist. Finalization replaces those native indices
with the ordinary anchor/context selection rows. The same rule therefore
supports hierarchical and flat search without provisional identity, rekeying,
or a second plan-selection authority.

The route itself must cover the applicable terminal-owner set, not the static
terminal catalog. Consider a producer domain `{0, 1}` and a consumer point
`q = 0` with `source_map(q) = 2q`. That terminal applies at producer point
`0`, but it has no preimage at point `1`; forcing it into both plans invents a
delivery at point `1`. Conversely, if consumer points `q = 0` and `q = 1` both
map to producer point `0` and their execution bindings select different
AccCores, one static terminal needs two physical branches. Requiring one
attachment per terminal loses one owner. Both failures disappear when the
plan is the exact unique set of `(terminal, owner)` pairs for its relation
range.

This pair is derived rather than stored. Dataflow remains the sole owner of
consumer domains and `source_map`; execution binding remains the sole owner of
the selected AccCore; Fabric remains the sole owner of endpoint membership;
and the RouteTree stores the terminal and selected route node it already
needs. Several consumer points on the same owner collapse to one physical
attachment, while different owners remain separate branches. If no consumer
point maps to a producer range, the canonical empty message plan records that
no physical transfer exists instead of manufacturing a sinkless route.

Adding a persistent delivery-branch identity would copy both Dataflow
correspondence and execution partitioning into Mapping. Restricting channels
to surjective, owner-functional `source_map` relations would instead reject
valid rate conversion and multicast already admitted by Dataflow. Deriving the
pair set changes only the necessary route-coverage invariant and therefore has
the smallest conceptual surface while preserving one owner for every fact.

The change uses Mapping 3.0 because a 2.0 reader rejects both a repeated
terminal and a childless ServicePlan. Reusing 2.0 would give the same schema
identity two incompatible accepted languages. The TechMapping and
SpatialMapping payloads do not otherwise change, but all three profiles move
together because one `loom.mapping` family declaration owns their shared
version. This is a version boundary, not a compatibility mode or a second
parser path.

## Why System Spatial-Temporal Is A Spectrum

System-level spatial-temporal allocation and resource-level temporal hardware
are independent questions. At system level, the thread binding relation maps
logical domain points to AccCore occurrences. Thread groups, resident groups,
batches, and waves are derived partitions or scheduling views of that relation,
not additional persistent records or mutually exclusive execution modes.

`MaxSpatial` and `MaxTemporal` are useful profile names for opposite search
preferences, not the only legal answers. One candidate may dedicate different
AccCore subsets to different kernels and pipeline their channels, while each
subset processes points of its own kernel in batches. Another may distribute
one kernel across most cores. Intermediate mixtures are expected and remain
expressible by the same binding and event-relative resource-use algebra.

This spectrum must not be confused with `fabric.pe`, `fabric.switch`, or
`fabric.mem` spatial/temporal capability. The latter states how one physical
resource accepts concurrent or time-multiplexed uses. SystemMapping states
which logical work uses which physical occurrences. Keeping the axes
orthogonal prevents a profile label from becoming a hidden hardware mode or a
second schedule authority.

## Why Examples Cover Regular And Irregular Hardware

The minimal five-input example isolates coupling and endpoint routing. Memory,
multicast, irregular, and torus examples extend the same contracts rather than
introducing demo-only schemas. Narrow software values may traverse wider
same-kind links under the low-bit rule, and control tokens may use explicitly
adapted shared transport. These examples demonstrate separation between
software semantics and physical representation.

Regular and irregular hardware must expose the same compatible semantic
candidate set when their capabilities match, even though their complete
resource inventories and route domains differ. This is the useful invariant;
forcing identical resource multisets would erase topology diversity.
