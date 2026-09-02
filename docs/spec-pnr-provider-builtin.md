# Builtin Mapping PnR Replay Profile

This document owns the observable bounded-search behavior of Loom's in-tree
Spatial and System PnR provider. It is a versioned replay profile under the
provider-independent contract in [Place And Route](spec-pnr.md). It does not
redefine Mapping legality, Fabric semantics, constraint admission, or final
verification.

The replay identity is the exact tuple of existing component descriptors and
digests; this document introduces no new persistent schema:

```text
Spatial:
  loom.spatial_pnr.config.15.6
  loom.spatial_pnr.freeze.2.25
  loom.mapping.pnr.objective 3.4
  selected FabricPhysicalTimingProfile descriptor and digest

System:
  loom.system_pnr.config.8.5
  loom.system_pnr_search_domain.4.0
  loom.mapping.pnr.objective 3.4
  exact selected SpatialMapping references
```

Any incompatible change to finite domain construction, formal candidate order,
work consumption, PRNG use, routing cost, repair extraction, or publication
requires changing the descriptor that owns that behavior.

## Capability Matrix

| Capability | Spatial provider | System provider |
|---|---|---|
| Graph domain | Exact TechMapping input | Hierarchical immutable SpatialMappings |
| Negotiated routing | PathFinder | PathFinder or DualSubgradient |
| Coupled binding and routing Actions | Yes | Yes |
| Best feasible incumbent | Yes | Yes |
| Final global routing closure | Required | Required |
| CP-SAT exact repair | Bounded supported profiles below | Disabled |
| Focused Evaluation closure | Absent | Absent |
| Online Evaluation objective | Absent | Absent |
| Flat Spatial reopening | Absent | Absent |

Both providers consume the config-owned `completion_goal`. The
`exhaust_configured_work` goal executes every configured restart and annealing
slot. The `first_verified_candidate` goal follows the same canonical work
order, may stop annealing when a feasible incumbent is available, and stops
later restarts only after final closure, independent verification,
finalization, and publication. The latter returns retained candidates with
`SemanticLimitReached`; it is a bounded usable prefix, not an exhaustive
search claim.

Objective registry 3.3 separates proven closed waits, proof debt, and exact
runtime counterexamples.
`HardProgressViolation` is nonzero only for `ProvenClosedWaitSet`;
`ProgressProofDebt` is nonzero only for `ProofNotEstablished`.
`RuntimeCounterexampleViolation` is nonzero only when every literal of a
persisted runtime no-good still holds. It is a hard violation and is never in
the temporary-violation policy. `ProgressCapacityShortfall` and
`ProgressRouteAnchorCount` refine the progress ordering without becoming
legality owners. Spatial and System config descriptors 15.5 and 8.4 select
these dimensions before ordinary QoR. Final Spatial
publication independently rebuilds the Mapping closure and admits only
`ProvenNoClosedWaitSet`; an ordinary Mapping carrying proof debt remains
importable but unpublished without identity-bound retirement evidence.

`loom.mapping_constraints 1.3` is the provider's only persistent owner for a
promoted runtime-counterexample legality rule. The feedback object itself is
invocation-local. Promotion requires replay-verified Evidence, complete
certificate-derived anchors, and an independently verified invariant under
the exact Evaluation Request. A promoted clause records durable parent
SpatialMapping, Request, Evidence, execution, and certificate lineage and
contains an exact-parent Mapping identity literal. That literal is a
conservative complete-assignment blocker, not a minimal closed-SCC core. An
explicitly authored clause has no runtime lineage. Frozen indexes and
CandidateState cache literal truth but cannot become an alternative legality
or provenance owner.

Config projection and adoption fail closed for every unsupported combination.
There is no compatibility fallback, ignored field, focused-closure field,
route-guidance field, or Evaluation binding in the current PnR config bytes.

## Frozen Domains

Spatial freeze validates exact `D/T/F/C/K`, the selected physical timing
profile, and all finite owner domains before candidate allocation. It derives
canonical compute and memory choices, attachment alternatives, RegFIFO
alternatives, residual nets, routing topology, resource contracts, packed-row
fragments, tags, progress dependencies, objective inputs, and reverse indexes.
Freeze descriptor `loom.spatial_pnr.freeze.2.25` covers those observable
domains. Internal storage layout is not replay state.
The frozen-model cache key is a SHA-256 over the domain
`loom.spatial_pnr.frozen_model.key.v2.25`, whose minor follows the freeze
descriptor, plus the exact `D/T/F/C/K` identities, the ResolvedConfig view
descriptor and digest, the freeze descriptor, the importer identity
`loom.spatial_pnr.importers.2.1`, the native layout tag
`loom.spatial_pnr.native_layout.2.12`, the PnR index width, and the physical
timing profile digest. The native layout tag versions only the in-memory
storage layout: changing it invalidates cached frozen models without changing
any replay identity.

System freeze validates exact `D/F/R/H/C/K`, imports and independently verifies
every SpatialMapping in `H`, and derives thread, graph, service, attachment,
route, resource, timing, and candidate-progress domains. Search-domain 4.0 has
only `SystemHierarchicalGraphBindingDomain`; a graph choice is an ordinal into
its exact compatible SpatialMapping set.

Every domain is canonicalized by its complete typed key before a local dense
ordinal is assigned. Dense ordinals are removable hot indexes and cannot cross
an invocation boundary.

## Deterministic Streams

The selected protocol is
`Sha256SeededXoshiro256StarStar_1_0`. A stream is derived from the master seed,
restart ordinal, and one typed purpose:

```text
InitializerDiversification
Calibration
ActionProposal
Acceptance
ExactRepair
```

Purposes never share state. `nextBounded` uses rejection sampling, so modulo
bias and host integer width cannot change choice order. Stochastic acceptance
uses the checked-in `ExpNegativeQ64Table_1_0` protocol. Floating-point math,
standard-library distributions, and thread-local implicit seeds are forbidden.

## Semantic Work Units

The config owns these counted units:

```text
SeedAttempt
AssignmentAttemptPerSeed
EndpointExpansion
NegotiationIteration
ConsecutiveNoProgressIteration
NoProgressTrendTransition
CalibrationProposal
ProposalPerLevelBase
ProposalPerMovableDecision
ExactRepairRegionDecision
ExactRepairSolverCall
```

Work is charged in canonical discovery order before execution. Cache hits,
deduplication, failed publication, and parallel completion do not refund it.
Wall-clock timeout, memory reservation, external cancellation, and worker count
are execution controls rather than semantic work units. An interruption returns
an incomplete result and cannot substitute a larger semantic budget.

For Spatial PnR, `planned` counts only logical slots that the semantic owner has
actually admitted for immediate execution. It is never copied from a policy
limit or reconstructed from an after-the-fact consumed count. The owner
increments `consumed` after that atomic slot has executed. Initializer choice
attempts, A* endpoint expansions, negotiation iterations, proposal-domain
slots, and CP-SAT calls are charged at their respective execution sites. An
exact-repair region decision is charged once when it first enters the canonical
closed repair model; certificate-driven region growth charges only newly added
decisions. Final closure attempts retain the same internal planned/consumed
boundary but do not create a second public DSE work-unit catalog.

An error or cancellation before an admitted slot's owner boundary completes
preserves the live `planned > consumed` suffix in the typed outcome. A typed
negative outcome produced after that boundary completes consumes the slot. A
normal `ExhaustConfiguredWork` return consumes every configured restart and
every admitted nested slot, so every public work-summary row has
`planned == consumed`. `FirstVerifiedCandidate` remains an explicitly
incomplete bounded prefix; restart slots beyond that executed prefix remain
configuration capacity rather than being fabricated as planned work.

The invocation-local `ExecutionControlView` combines the DSE journal's
graceful-stop state with its absolute dispatch deadline. Tech, Spatial, and
System providers query it only between atomic owner work units. Their typed
interruption snapshot contains the owner stage, the exact consumed frontier,
the best selected rank when one is defined, a per-violation closure residual
when it can be reconstructed, retained finalized references, active wall time,
process CPU-time delta, allocator observation, and process peak resident memory
when the host provides them. Process values can include concurrent work and
are not invocation-local attribution. Unavailable rank or residual dimensions
remain absent; they are never replaced with zero. Mapping debug output
serializes this snapshot as one nested payload, and is disabled without
constructing that payload.

## Spatial Restart Sequence

The Spatial provider allocates exactly `seed_attempt_count` isolated restart
slots. Slots may execute in parallel, but their results are reduced by original
restart ordinal.

`ExhaustConfiguredWork` bounds parallel restart workers by the configured
candidate-worker request, restart count, active RouteGraph unit count, the
admitted CPU claim, and the admitted memory reservation. A missing or zero
resource dimension is unconstrained. The active RouteGraph unit count is the
saturated sum of active endpoints, traversals, and routing arcs, with one as
the minimum serial execution capacity. When a memory reservation is present,
canonical restart zero supplies a deterministic retained-scratch estimate for
the active problem; that already-required result is retained and is not
generated again. The shared frozen-problem projection is charged once.
Root-complete exhaustive ranking may transfer its already constructed
ordinal-zero seed to this same owner; the handoff is consumed once and its
initializer and routing work is charged once. The formal owner still performs
the normal candidate verification. A failed transferred seed retains its typed
failure and is classified by the formal restart owner without a cold retry.
Root active-problem diagnostics report both prepared and consumed handoff
counts so this transfer remains independently auditable.
`FirstVerifiedCandidate` remains a serial bounded-prefix execution. A plan
publication bound is applied only after the exhaustive restart sequence and
cannot reduce, serialize, or reclassify its configured work. Worker allocation
and host-process observations are diagnostic only and cannot change restart
streams, ordinal reduction, candidate identity, or formal work accounting.

Each slot executes:

```text
canonical relation initialization
  -> coupled initial route closure
  -> transactional annealing
  -> restore best feasible incumbent when one exists
  -> bounded repair/global-closure loop
  -> candidate invariant verification
```

A failed slot is not refilled. A proven-infeasible freeze or relation proof may
terminate the invocation only when its proof covers the complete admitted
domain. Other slot failures are retained as typed incomplete results while
independent slots continue.

### Relation Initialization

Initialization uses minimum-current-domain first and canonical typed-key
tie-breaking. Attempt zero uses canonical value order. Later attempts use a
without-replacement permutation from the restart's initializer stream.

The hard root model contains:

* compute resident-context disjointness;
* Spatial Memory operation-port disjointness;
* Temporal Memory occurrence-global weighted row capacity;
* Temporal Memory external-ingress uniqueness;
* attachment-progress compatibility;
* RegFIFO and external disposition compatibility; and
* caller-authored domain, equality, and disjoint clauses.

Topology, locality, Temporal scheduling, port spread, and RegFIFO reuse are
soft value-order preferences only. They do not delete a hard-domain value. The
root solver first closes compute and memory choices, then attachment choices,
while preserving a legal external route fallback for every optional local
transfer.

### Coupled Actions And Incumbents

Annealing interleaves realization binding, transport routing, and resource
allocation Actions with weights from config. An Action and every incident
route, capacity, tag, packed-row, timing, and progress update commit in one
transaction. Rejection restores the exact prior state.

The search records the best feasible candidate under the selected total
ordering independently of the current annealing state. It continues the
configured quality work after first reaching zero violations. At restart end,
the best feasible incumbent is restored; if no feasible candidate was seen,
the best current state may enter repair but cannot be published.

### PathFinder Routing

Spatial routing uses negotiated PathFinder with complete multicast RouteTrees.
Endpoint A* expands the frozen payload-compatible topology and accumulates:

* selected traversal claim;
* present and historical congestion;
* exact incremental resource overuse;
* newly required packed Temporal switch rows;
* tag conflict and resident-row pressure;
* durable-boundary progress compatibility;
* normalized or target physical traversal delay; and
* logical-net structural criticality and negative slack.

The selected physical timing profile, not coordinates alone, owns traversal
delay. Registered destinations terminate the current combinational arrival.
Equal-cost ties use canonical endpoint and traversal keys. A route failure
rolls back every route-derived cut and cache whose proof is not valid outside
that transaction.

The route cost owner validates the arc cost arrays it publishes at every
write, finite lower-bound and current costs with no current cost below its
lower bound, and certifies that validation through the input revisions the
endpoint router receives. The router scans a request's cost arrays itself only
when a revision is absent, stale, or uncertified, so the validation happens
once per write instead of once per query without weakening it.

The negotiated router observes the invocation-local `ExecutionControlView`
between its atomic work units only: before each negotiation iteration and
between the net routes of one iteration, never inside an endpoint A* search.
An observed stop is the typed `Interrupted` closure failure: the active move
rolls back its route overlay, a negotiation iteration that was planned but not
completed stays unconsumed, and the enclosing seed construction, annealing,
exact repair, or final closure owner reports the restart as interrupted at
that stage rather than as a work-limit or rejected transition. The stop
latency is therefore bounded by one net route, and an uninterrupted run
executes the identical proposal, route, and accounting sequence.

### Exact Repair Profiles

The Spatial provider uses the required in-process `CpSat_3_0` adapter from the
pinned OR-Tools v9.15 source commit
`551ad10d94835c99e5e1e684500d3db398c0e345`.

Every canonical solve runs with presolve probing disabled and under a fixed
deterministic-time budget of 2.0. Probing computes failed-literal information
the canonical `FIXED_SEARCH` strategy never consumes, and the budget is an
instruction-count clock, so the same model and seed exhaust it identically on
every host. A budget-exhausted solve is the existing typed `Unknown` outcome:
the repair remains incomplete and can never prove infeasibility or consume the
invocation deadline. These solver constants are part of the versioned config
descriptors above; changing them is a formal search-order change.

There are two actual repair profiles:

1. **Transport closure** encodes compute and memory placement, exact terminal
   attachments, graph-boundary attachments, RegFIFO versus external
   disposition, affected routes and tags, relation closure, fixed outside
   claims, and retained fixed-terminal cut certificates.
2. **Atomic capacity** encodes compute binding and its complete hard relation
   closure. It does not encode memory binding decisions.

Before Candidate state is allocated, the provider rejects statically
recognizable frozen-domain capability mismatches. Every Spatial constraint
projection is classified exhaustively by its binding, route, tag, or memory
owner.
`CpSat_3_0` is rejected as unsupported when an atomic compute relation closure
can reach a non-compute decision, or when a selectable memory operation plan,
memory dispatch, or exposure provider can contribute atomic capacity overuse.
No search begins for such a domain, and no alternate repair algorithm is
selected implicitly.

A candidate-local witness that has no complete typed encoding returns
`UnsupportedEncoding`, including a route-progress dependency violation without
a finite-buffer owner witness. This runtime result remains incomplete and
cannot prove infeasibility. Region overflow returns `RegionTooLarge`;
solver-call exhaustion or any non-proof-bearing status returns
`UnknownBudgetExhausted`.

The result vocabulary is:

```text
Repaired
RegionInfeasibleUnderFixedBoundary
UnknownBudgetExhausted
RegionTooLarge
UnsupportedEncoding
InternalError
```

Only `Repaired` mutates the candidate. A local infeasibility remains
`RegionInfeasibleUnderFixedBoundary`. It cannot become invocation-level
`ProvenInfeasible` unless the exact region is independently proven to be the
complete invocation domain.

Each repair invocation consumes one word from its exact-repair stream. The low
31 bits seed a one-worker CP-SAT run. Search is fixed, randomized search and
LNS are disabled, presolve remains enabled, and no solver wall-time or
deterministic-time limit replaces Loom's solver-call budget. Canonical
mixed-radix extraction fixes optimum values in typed decision-key order and
splits blocks before any signed-integer safety limit.

Transport repair first tests the current exact assignment. Failed route probes
add an invocation-local search no-good over the complete observed placement,
attachment, and local versus external disposition tuple. This temporary
exclusion is not a promoted runtime-counterexample clause. A fixed-terminal
certificate excludes only assignments for which its separating capacity proof
remains valid. Certificate growth is monotonic inside one invocation and
cannot become persistent Mapping state. A successful probe that realizes its
assignment, removes the primary witness, and preserves atomic capacity is legal
even when its selected objective rank does not improve. Objective preference
cannot turn that legal assignment into a hard no-good; the cold closure and
verifier remain the legality gates.

### Final Spatial Closure

Repair and global closure alternate until global closure succeeds, a typed
incomplete result prevents continuation, or the restart-wide solver-call limit
is consumed. Successful repair always requires another global closure.

The final candidate must have all five Mapping violations at zero. Candidate
invariants are checked before materialization. The cold SpatialMapping verifier
and exact `K` admission then run during finalization. Only the resulting
Artifact reference enters the canonical candidate set.

## System Restart Sequence

The System provider allocates one isolated restart slot per configured fresh
seed attempt, plus one migration slot when a migration projection is present.
As in the Spatial sequence, slots may execute in parallel under
`ExhaustConfiguredWork`, but their results are reduced by original restart
ordinal: accounting accumulation, incomplete classification, candidate
publication, and interruption reporting all follow canonical attempt order, so
scheduling cannot change candidate identity, formal work accounting, or the
first-incomplete diagnostic. Draft materialization, SystemMapping
finalization, and publication run only in the ordinal reduction. The migration
slot executes before the fresh slots because its direct-publication trial and
its annealed candidate share one state. `FirstVerifiedCandidate` remains a
serial bounded-prefix execution. Worker allocation is bounded by the
configured candidate-worker request and the fresh restart count and is
diagnostic only.

Each restart slot executes:

```text
hierarchical binding and service initialization
  -> coupled route preparation
  -> transactional annealing
  -> restore best feasible incumbent when one exists
  -> one strict global System routing Action
  -> candidate invariant verification
  -> cold SystemMapping base verification
  -> exact K admission and finalization
```

Initialization and every Action project the selected graph SpatialMappings,
service contexts, terminal attachments, System routes, static route occupancy,
occurrence-qualified owners, grant requesters, activation triggers, causal
releases, and capacity claims into the shared System progress model. Candidate
progress and the cold verifier therefore evaluate the same wait-for graph.

A fixed-terminal System capacity certificate remains attached to its route
probe even when temporary-violation policy admits the probe's best route set.
The certificate names its capacity cell and canonical service legs. The Action
executor mechanically projects those legs through their service contexts to
the affected graph decisions. If at least one affected decision has an
alternative, one coupled execution-binding reopen Action releases the complete
relation closure of all affected graph decisions and invokes the existing root
solver jointly. This avoids requiring an individually illegal intermediate
binding to become a candidate.

The reopen Action consumes the next ordinary annealing proposal slot and all
of its assignment and routing work is charged normally. It receives no hidden
repair slot. Pending reopen Actions are invocation-local and are invalidated by
an accepted candidate change; a certificate produced by that accepted probe is
then projected against the new candidate. No certificate, Action, or exclusion
enters a Mapping artifact or a persistent no-good set. If no ordinary slot
remains, strict final closure reports proof not established rather than an
internal failure.

System Actions use immutable delta transactions. A probe records the changed
thread and graph decisions, service legs, service targets and ResourceUses,
together with capacity, progress and intrinsic timing values before and after
the mutation. The current candidate is never modified: discarding a
calibration probe, rejecting an annealing proposal or receiving a typed Action
failure is rollback by dropping the proposed owner.

The removable candidate cache separates selected non-route demand from System
service-route demand. A transport transaction rebuilds only its selected
service routes and their route-progress obligations while retaining the exact
non-route projection. A resource transaction rebuilds its mutable
InstructionCore and service ResourceUse prefix while retaining imported
Spatial routes, graph progress and System service routes. Execution-binding
changes rebuild their complete dependency closure because they can alter every
component. Every accepted transaction is checked against a cold projection
from immutable selections before it replaces the working state. Final closure
and independent Mapping verification repeat the cold path; a cache or delta
record is never a legality owner.

System exact repair is disabled. A final global Action that reaches a semantic
work limit yields an incomplete restart. Capacity overuse, unresolved progress,
or verifier rejection prevents publication. Other restarts continue, and any
verified candidate remains available to the aggregate result.

## Builtin Policies

All presets use realization:routing:resource Action weights `1:3:2`,
PathFinder multiplicative pricing with initial present pressure `1`, growth
`3/2`, and history increment `1`. They use no-progress iteration limit `8`,
trend window `4`, positive-delta quantile `3/4`, target initial acceptance
`4/5`, fallback temperature `1024`, minimum temperature `1`, cooling `19/20`,
master seed `0`, and the deterministic protocols above.

| Preset | Seeds | Assignments | Endpoint expansions | Negotiations | Calibration | Temperature levels | Level base | Per movable |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `report_only` | 1 | 4096 | 16384 | 8 | 16 | 4 | 16 | 1 |
| `quick_explore` | 2 | 16384 | 65536 | 16 | 64 | 8 | 64 | 2 |
| `balanced_explore` | 4 | 65536 | 262144 | 64 | 256 | 16 | 128 | 8 |
| `performance_explore` | 8 | 262144 | 1048576 | 128 | 512 | 32 | 256 | 16 |
| `implementation` | 16 | 524288 | 2097152 | 256 | 1024 | 64 | 512 | 24 |
| `strict_implementation` | 32 | 1048576 | 4194304 | 512 | 2048 | 128 | 1024 | 32 |

`temperature_level_limit` is a semantic work bound. Cooling proceeds normally
when that schedule reaches the minimum within the bound. Otherwise each level
uses the colder of ordinary cooling and a deterministic integer envelope over
the remaining range, and the last permitted level is the unique
minimum-temperature level. Consequently, an energy weight or calibrated delta
can change acceptance probabilities but cannot silently increase the number
of proposal slots or collapse every non-final level into the same hot regime.

Spatial `report_only` disables repair. Other Spatial presets select CP-SAT
with `(max_region_decisions, max_solver_calls)` of `(64,128)`, `(256,1024)`,
`(512,4096)`, `(1024,8192)`, and `(2048,16384)` respectively. Every System
preset disables repair while retaining its other limits.

## Builtin Objective Closure

The builtin catalog contains all five violations and all seven Mapping
measures from objective registry 3.0. Its weighted levels in canonical catalog
order are traversal, schedule, closure, timing, and search energy. The selected
total ordering is:

```text
closure -> timing -> schedule -> traversal
```

Closure gives every violation equal weight. Timing gives every cycle,
throughput, transport, arrival, and slack measure equal rank weight. Search
energy gives violations weight `2^48`, schedule and non-transport timing
measures weight `2^32`, and traversal plus transport bit-cycle demand weight
`1`. Checked wide arithmetic rejects overflow. These weights guide search only;
final legality still requires zero violations.

## Publication And Replay

Spatial restart results are reduced by attempt ordinal and finalized references
are sorted by canonical ArtifactRootReference order with exact deduplication.
System attempts are traversed canonically and their finalized references receive
the same canonical set reduction. Publication does not expose restart order as
Mapping content.

Replay requires exact semantic inputs, config bytes and digest, provider-owned
descriptors, objective registry, physical timing profile and digest, pinned
solver build, and owner-local work limits. Worker count may change only physical
scheduling. A replay mismatch is an internal provider error; it cannot be
explained by cache state or host concurrency.
