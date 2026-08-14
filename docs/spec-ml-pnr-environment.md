# ML PnR Environment

This document defines the interaction boundary through which a learned search
policy places and routes one Mapping problem. An episode holds one exact
Spatial PnR invocation — a Canonical Dataflow Program, its TechMapping, a fully
elaborated Fabric, a resolved PnR configuration, and a MappingConstraintSet —
and presents its search as a sequence of typed Place and Route Actions. The
policy binds each realization to a hardware occurrence until every realization
is placed, and then either hands the result to a distance-bounded annealing
cleanup or keeps repairing it itself.

The environment is a selector, not a placer. Every Action it applies is a
member of the closed `SpatialMappingAction` algebra
[Place And Route](spec-pnr.md#actions-and-movetransaction) already owns, every
transition goes through that owner's `MoveTransaction`, and every score is a
projection of that invocation's own objective closure. What the environment
replaces is the deterministic proposal selector, and nothing else.

The relationship to [ML DSE Environment](spec-ml-dse-environment.md) is
symmetric and the difference is worth stating at the top, because it inverts
one rule. There, a closed Mapping is a gate: a design that cannot run the
workload is outside the space, and a step that reaches one is rejected. Here, a
closed Mapping is the goal: residual Mapping violations are ordinary candidate
state that the objective prices, and an episode spends most of its life in
states that are not yet closed. The two documents share their observation
container, action surface, and Python boundary through
[ML Environment Core](spec-ml-core-environment.md); they share no episode rule.

## Ownership

Every fact this document depends on resolves to one exact owner:

- [Spatial PnR](spec-pnr.md#spatial-pnr) owns the exact five-input invocation
  `(D, T, F, C, K)`, its aggregate freeze entry, and its exact binding
  requirements;
- [Native State](spec-pnr.md#native-state) owns `FrozenModel`,
  `CandidateState`, `SearchScratch`, the factorized candidate domains, and the
  freeze cache key;
- [Actions And MoveTransaction](spec-pnr.md#actions-and-movetransaction) owns
  the closed `SpatialMappingAction` algebra, the dynamic domain `A(M,C,S)`, the
  transition `Apply(M,C,S,a)`, the dependency closure a binding change carries,
  and the sole mutation mechanism;
- [Deterministic Initialization And Action Proposal](spec-pnr.md#deterministic-initialization-and-action-proposal)
  owns the canonical typed decision-key order, the canonical choice orders, the
  transport-routing scopes, and the frozen-topology hop distance;
- [Annealing And Replay](spec-pnr.md#annealing-and-replay) owns the annealing
  policy, `realization_move_radius`, the acceptance protocol, and the seeded
  PRNG protocol;
- [Evaluation Transaction](spec-pnr.md#evaluation-transaction) owns the online
  probe protocol and the ephemeral Evaluation adapter;
- [Objective Projection](spec-pnr.md#objective-projection) owns the Mapping
  violation catalog `V` and measure catalog `G`, and
  [Final Closure And Verification](spec-pnr.md#final-closure-and-verification)
  owns what makes a candidate closed;
- [Resolved View](spec-pnr.md#resolved-view) owns `ResolvedPnrConfigView` and
  `SelectedObjectiveClosure`;
- [Spatial MappingConstraintSet Contract](spec-pnr.md#spatial-mappingconstraintset-contract)
  owns `K` and its projection catalog;
- [Evaluation and DSE](spec-dse-feedback.md#objectives-and-quality-gates) owns
  the objective algebra, quantization, and the statement that a reward is the
  signed difference of a selected search energy;
- [TechMapping Generation](spec-tech-mapping.md) owns `T` and its exact `D` and
  `F` binding;
- [Fabric Identity](spec-fabric-identity.md#owner-local-reference-kind-catalog)
  owns the local-reference kind catalog the observation's `EntityKind` column
  carries;
- [ML Environment Core](spec-ml-core-environment.md) owns the observation
  container and its combined node-and-link space, the action surface and
  masking algebra, the step accounting identity, the
  termination-versus-truncation rule, the reward boundary, the PRNG preimage
  shape, the `loomml` package layering and its interaction contract, and the
  benchmarking obligations every harness satisfies; and
- [Resolved Configuration](spec-config-ssot.md#component-views) owns
  component-view framing, canonical view bytes, and `component_view_digest`,
  and at [Cache Dependencies](spec-config-ssot.md#cache-dependencies) owns the
  cache-family contract.

This document owns only the episode arms, its resolved configuration view and
its curriculum-neutral partition, the frozen-model and seed-state retention
policy, the construction, repair, and cleanup protocols, the PnR action index
contract, the PnR observation catalogs, the step and failure outcome algebra,
the trajectory retention record, and its own benchmarking stages.

## Nonsemantic Boundary

The environment is a nonsemantic search harness, on the terms
[Nonsemantic Boundary](spec-ml-core-environment.md#nonsemantic-boundary) states
for every ML environment. It is not a placer, a router, a resource allocator,
an objective, or a Mapping authority. It owns no occupancy model, no route
cost, no distance heuristic of its own, and no second candidate representation.

It publishes no `SpatialMapping`. An episode's final candidate is mutable
`CandidateState`, which
[Canonical Search Sequence](spec-pnr.md#canonical-search-sequence) already says
never enters a central candidate set. Nothing in an episode runs independent
final verification, assigns a Mapping identity, or emits a lineage edge.

Three consequences follow.

Legality is never the environment's answer. A proposed Action either produces a
committed transition or is rolled back by `MoveTransaction`, and the
environment reports which. It does not pre-screen a choice the owner would
reject, does not retry a rolled-back Action under a different anchor, and does
not repair a candidate the owner refused to change.

Cost is never the environment's answer either. The objective the reward
differences is the one `pnr_config_view.selected_objective_closure` already
declares, evaluated by the ordinary online protocol. A policy that wants a
different trade-off binds a different closure; it does not get a reward term
this document invents.

Infeasibility is never the environment's answer. `CandidateDomain(u)` is
derived at freeze from `F` and `K`, so a choice the environment never
enumerates is a choice the owner never offered. The environment adds mask bits,
and a mask bit is a phase rule, not a proof.

## Episode Arms

An episode is one of two arms, each carrying exactly the fields its own
protocol needs:

```text
PnrEpisodeArm =
    ConstructThenAnneal {                        // 0
      cleanup_annealing_policy: SearchPolicy.annealing
      cleanup_displacement_reward_code: uint64
    }
  | ConstructThenRepair {                        // 1
      repair_step_bound: positive uint64
      repair_step_reward_code: uint64
    }
```

Ordinals are stable. A new arm appends; reordering, deleting, or repurposing
one is an incompatible change to this document's schema version.

Both arms run the same construction phase and differ only in what happens once
every realization has been placed. `ConstructThenAnneal` hands the candidate to
one bounded annealing run and ends; the agent's whole contribution is the
construction, and it is scored on the quality of what construction produced
*minus how much cleanup that construction turned out to need*.
`ConstructThenRepair` keeps the agent in control, letting it rebind whatever it
placed badly, and prices each repair it takes.

The arms are two answers to one question — who fixes an imperfect placement —
and they are kept as arms rather than as a flag pair because the fields are not
shared. A radius and an annealing policy mean nothing to an agent-driven
repair, and a repair bound and a per-repair price mean nothing to a single
annealing transition. A record carrying all five would have a
required-or-forbidden rule per field and no way to state it except prose.

`SearchPolicy.annealing` is the exact annealing record
[Annealing And Replay](spec-pnr.md#annealing-and-replay) owns, including its
`realization_move_radius` field, carried here as a complete value rather than
as a reference into `pnr_config_view.search_policy`. The two govern different
runs and have no reason to match: the view's policy is what an ordinary Spatial
PnR invocation of this problem would use, while the cleanup is a deliberately
short, deliberately local run. Neither is an environment-authored policy, and
adoption validates this one exactly as its owner requires.

The distance bound the arm needs is that record's own
`realization_move_radius`, and the arm carries no second radius field beside
it. A field naming the same bound twice would be two things a future edit could
make disagree, and the one the annealing run actually reads would win silently.
What this arm requires instead is the `Bounded` arm of that field. An
unbounded cleanup makes `cleanup_displacement` a measure of the annealer rather
than of the construction, and it would leave the return range's displacement
term with no bound to read; a `ConstructThenAnneal` arm carrying `Unbounded` is
rejected at adoption.

## Resolved PnR Environment Configuration

The environment consumes one immutable component view with schema descriptor
bytes `loom.ml_pnr_environment.config.1.0`, following the framing, canonical
byte representation, and digest contract owned by
[Resolved Configuration](spec-config-ssot.md#component-views):

```text
ResolvedMlPnrEnvironmentConfigView {
  episode_arm: PnrEpisodeArm
  problem_pool: canonical nonempty set<SpatialPnrProblemBinding>
  pnr_config_view: ResolvedPnrConfigView
  episode_policy: PnrEpisodePolicy
  observation_policy: PnrObservationPolicy
  determinism_policy: EnvironmentDeterminismPolicy
}

SpatialPnrProblemBinding {
  dataflow: ArtifactRootReference          // D
  tech_mapping: ArtifactRootReference      // T
  fabric: ArtifactRootReference            // F
  constraints: ArtifactRootReference       // K
}

PnrEpisodePolicy {
  anchor_selection: AgentSelected | CanonicalNext
  step_bound: positive uint64
  consecutive_failure_bound: positive uint64
  failed_transition_reward_code: uint64
  incomplete_closure_reward_code: uint64
}

PnrObservationPolicy {
  enumeration_bound: positive uint32
  include_route_edges: bool
  include_resource_states: bool
}

```

`SpatialPnrProblemBinding` is exactly the persistent part of the five-input
invocation. `C` is absent from it and carried once by the view, because every
episode of one configuration searches under the same resolved policy and a
per-problem `C` would let two members of one pool disagree about what the
reward means. Adoption requires each member to satisfy the exact bindings
[Spatial PnR](spec-pnr.md#spatial-pnr) states — `T.D == D.id`, `T.F == F.id`,
and `K.D/T/F == D.id/T.id/F.id` — because a binding that does not is not an
invocation at all and failing at the first `reset` that draws it would report a
configuration error as a runtime one.

There is exactly one objective closure here, and that is deliberate. The DSE
environment carries two because its probe is an inner search whose own
objective must not be the reward; this environment *is* the search, so the
closure that steers it and the closure that scores it are necessarily the same
one. A second would have to disagree with the first about what a good Mapping
is, and the policy would learn the disagreement rather than the problem. The
reward authority is therefore `pnr_config_view.selected_objective_closure` and
the view carries no closure of its own.

`PnrEpisodePolicy` holds what applies to the whole episode rather than to one
phase. `step_bound` counts every step an episode takes, construction and repair
alike, and `consecutive_failure_bound` and the two reward codes apply in both
phases; only `anchor_selection` is construction-specific. Splitting a
per-construction record from a per-repair one was rejected because
`ConstructThenAnneal` has no repair phase and would leave the second record
entirely inert.

`anchor_selection` chooses what the policy decides each construction step.
`AgentSelected` enumerates every unplaced realization crossed with its legal
occurrence domain, so the policy chooses both which realization to place and
where. `CanonicalNext` fixes the anchor as the first unplaced realization in
the canonical typed decision-key order and enumerates only its choices. The
trade-off is stated rather than defaulted away: `AgentSelected` makes placement
order part of what is learned, which is the more interesting half of the
problem, and it makes the enumeration the sum of every unplaced realization's
domain rather than one realization's, so `enumeration_bound` has to be sized
for it. `CanonicalNext` is the cheaper regime and the honest baseline against
which learned ordering is measured.

The canonical view encoder writes fields in the schema order above, under the
encoding and adoption rules
[Resolved Configuration](spec-config-ssot.md#component-views) owns. This view
adds only that an `ArtifactRootReference` and an embedded owner record use
their owner's canonical encoding.

The PRNG preimage shape, the meaning of `effective_seed` and the copy
coordinates, and the `reset` seed override are owned by
[Determinism And Copy Coordinates](spec-ml-core-environment.md#determinism-and-copy-coordinates).
This document adds only its own domain separator, which is

```text
ASCII("loom.ml_pnr_environment.prng.sha256_seeded_xoshiro256starstar.1.0")
```

and its stream purpose ordinals, which are `ProblemSelection = 0` and
`CleanupSeeding = 1`.

`CleanupSeeding` exists because the cleanup run is an ordinary annealing run
and therefore consumes PnR's own `Calibration`, `ActionProposal`, and
`Acceptance` streams, which that owner derives from a master seed and a seed
index. The environment draws that seed index once per episode from
`CleanupSeeding` rather than reusing a counter, so the cleanup is a function of
the episode's effective seed and copy coordinates exactly as every other
episode-local draw is. Reusing `local_episode_index` directly was rejected: a
`reset` seed override would then change which problem an episode ran but not
which cleanup it got.

### Curriculum-Neutral Fields

[Curriculum](spec-ml-core-training.md#curriculum) requires each environment
document to partition its own view by what a field determines. Here the
shape-fixing side is `episode_arm`, `pnr_config_view`, `observation_policy`,
`determinism_policy`, and, inside `PnrEpisodePolicy`, `anchor_selection` and
the two reward codes. The arm decides which phases exist and therefore which
Action kinds are ever live; the Place and Route view carries the objective
closure the reward is a difference of; `anchor_selection` decides whether the
policy chooses anchors at all, which changes the enumeration's extent rather
than its size.

`problem_pool` is neutral, and that it is neutral is the point of stating the
rule by what a field determines rather than by naming a record. The pool sits
outside `PnrEpisodePolicy` here while the design-space environment's
corresponding field sits inside its episode policy, so a rule phrased as one
record name would forbid growing the problem pool — the one curriculum this
environment most obviously wants. `step_bound` and `consecutive_failure_bound`
are neutral for the same reason they are there: they bound an episode's length
and price nothing.

Growing the pool costs nothing already warm. A stage that adds problems keeps
every retained frozen model and canonical seed state for the problems it
already had, because those are keyed on one `SpatialPnrProblemBinding` rather
than on `problem_pool`: the key names a member, not the set, so extending the
set invalidates nothing in it.

## Problem Instance And Frozen Model Reuse

Freeze is the dominant cost in this environment and it is the reason `reset` is
worth a section of its own. `freezeSpatialPnrProblem(D, T, F, C, K)` validates,
resolves, indexes, and precomputes the complete Fabric projection; a step, by
contrast, touches only changed incidence. An implementation that froze per
episode would spend most of a training run in `reset`.

It does not have to. The Spatial PnR cache key
[Native State](spec-pnr.md#native-state) defines already hashes exact `D.id`,
`T.id`, `F.id`, `K.id`, the fields of `C` that freeze reads, freeze and
importer semantics, the native-layout ABI, and the actual `PnrIndex` width.
Every one of those is constant across the episodes of one configuration that
draw the same `SpatialPnrProblemBinding`, so the environment retains published
`FrozenModel` values under that key and reuses them across episodes and across
the vector copies living in one process.

The key names the freeze-relevant projection of `C` rather than
`component_view_digest(C)`, and the difference is the one that decides whether
a study is affordable. Freeze precomputes a Fabric projection; a reward closure
and a search policy are not inputs to it. Keying on the whole digest would make
a sweep over `selected_objective_closure` — the obvious study over this
environment — re-freeze every problem in the pool for a result byte-identical
to the one it discarded, which is the same trap
[Configuration Split](spec-ml-core-training.md#configuration-split) draws the
training and environment views apart to avoid. The full digest is still
compared on a hit, as the revalidation below requires; it is a check, not the
key. A `FrozenModel` is immutable and explicitly
shared across workers, so sharing it across environment copies introduces no
mutability the owner does not already permit.

A cache hit revalidates the descriptor, canonical view bytes, digest framing,
and exact artifact inputs before reuse, exactly as the owner requires.
Retention across episodes is an ordinary cache under
[Cache Dependencies](spec-config-ssot.md#cache-dependencies): a hit and a miss
produce the same formal result, and evicting the whole cache changes cost and
nothing else.

This is also why the cleanup radius is a search-policy field rather than a
per-episode `K`: `K` is an input to freeze, so a per-episode one would give
every episode a distinct cache key and destroy exactly the reuse this section
depends on. The field's home is
[Annealing And Replay](spec-pnr.md#annealing-and-replay).

## Construction Phase

Every episode begins in the construction phase and leaves it when every Compute
and Memory Realization has been placed exactly once.

`reset` builds the candidate through `createCanonicalSpatialCandidate`
-equivalent construction: the canonical assignment the owner's attempt-zero
initializer produces, with every RouteTree left visibly unrouted. The
environment then marks every realization *unplaced* in an episode-local bitset.
That bitset is the only construction state the environment owns.

Each construction step applies exactly one `RealizationBindingAction` whose
anchor is an unplaced realization, through the ordinary probe-and-commit path.
On commit the anchor becomes placed. Routing of that realization's dependencies
is not something this document adds: a binding change already invalidates old
attachments and route claims, rebuilds every incident route dependency, and
updates resource-time, buffer, tag, memory, and handshake state inside the same
transaction, which is precisely "route the dependencies of the node just
placed". The environment issues no routing Action of its own during
construction.

A realization becomes placed whether or not its incident nets closed. A net
whose other endpoint is still unplaced routes against that endpoint's canonical
seed binding and is re-closed when that endpoint is later bound, and a net that
cannot close at all leaves an `UnroutedObligation` violation in the candidate.
Both are ordinary candidate state that the objective prices and the reward
reports, which is what makes construction a dense-reward problem rather than
one that pays only at the end.

An unplaced realization whose dynamic Action domain offers no alternative to
its current binding becomes placed without a step. The owner's domain contains
only anchors with at least one legal alternative, so a realization whose
`CandidateDomain` is the singleton its canonical assignment already selected
contributes no entry and the agent has nothing to choose; treating it as
unplaced would let construction stall with realizations outstanding and no live
entry to advance on, which the action surface has no defined behavior for.
Marking it placed is exact rather than a concession: the choice the agent would
have made is the only choice there is, and the candidate already holds it.

Settling is therefore re-evaluated in both directions after `reset` and after
every commit, because a commit can shrink another anchor's domain to a
singleton and can equally restore an alternative to one already settled. A
settled realization whose domain regains an alternative returns to unplaced and
re-enters the enumeration. One-way settling would leave the exactness claim
false: a realization settled while its only choice was forced would stay
settled after the choice came back, and the agent would never be offered a
decision that had become real.

The rebind-sweep formulation has a real cost and it is stated rather than
hidden. Because `CandidateState` is always a complete assignment, an early
construction step routes nets whose far ends are still arbitrary, and a later
step redoes that work. A genuinely partial candidate with an unbound sentinel
would do strictly less work and was rejected as a semantic change to
[Native State](spec-pnr.md#native-state); why that trade went the way it did is
[Why Learned Place And Route Selects Existing Actions](rationales/ml.md#why-learned-place-and-route-selects-existing-actions).

Construction is not an initializer replacement in the owner's sense. The
owner's initializer propagates singleton domains and hard relation consequences
to a fixed point and backtracks on contradiction; the environment does neither,
because every state it occupies is already a complete legal assignment and
there is nothing to backtrack from. What the policy replaces is the choice of
which anchor to move next and which choice to take, which is the only part of
that protocol that is a heuristic rather than a proof.

## Cleanup Phase

When the last realization is placed, the arm decides what happens next. The
step that places it reports its ordinary transition either way; what differs is
whether that step also ends the episode.

### Bounded-Radius Annealing

Under `ConstructThenAnneal`, the step that completes construction also runs one
annealing run over the same candidate and then ends the episode with
`CleanupComplete`. The agent takes no action during the cleanup; it is a single
transition from the agent's point of view, and the observation that step
returns is the post-cleanup state.

The run is an ordinary one. It uses `cleanup_annealing_policy` unchanged, PnR's
own acceptance kernel and cooling schedule, and PnR's own streams seeded by the
index drawn from `CleanupSeeding`. The run-start occupancy its radius anchors
on is exactly the placement construction produced.

That anchoring is the whole point of the bound this arm requires: a bounded
cleanup can only tidy locally, so what it recovers is a measure of what
construction left on the table nearby.

`cleanup_displacement` is that measure, and it is a mechanical projection of
two candidates rather than a statistic the run reports:

```text
cleanup_displacement =
    sum over realizations of
      directed frozen-topology hop distance from its
      construction-final occurrence to its cleanup-final occurrence
  + count of realizations whose occurrence changed
```

The distance is the same one
[Deterministic Initialization And Action Proposal](spec-pnr.md#deterministic-initialization-and-action-proposal)
defines and the radius itself measures over, so no second distance model
appears. The two terms answer different questions and both are needed: the hop
sum says how far the cleanup had to reach, and the moved count says how broadly
it had to intervene. A single far move and a hundred adjacent ones are not the
same failure, and a sum alone cannot tell them apart.

`cleanup_displacement` is zero exactly when the cleanup accepted no move that
changed an occurrence, which is the case the arm exists to reward.

### Agent Repair

Under `ConstructThenRepair`, completing construction advances the episode into
the repair phase and the episode continues. The live enumeration opens to the
complete dynamic Action domain: the agent may rebind an already-placed
realization, issue a `WholeNet`, `SingleSink`, `RootedSubtree`, or
`WitnessRegion` routing Action, or take a resource-allocation Action. Nothing
in the domain is masked by phase any more.

Each committed step in this phase is a repair step and is charged
`repair_step_reward_code`. The episode ends when the agent elects to stop or
when it has taken `repair_step_bound` repair steps.

Two properties make this a different problem from the annealing arm rather than
a slower version of it. The agent decides *when* it is done, so it is scored on
recognizing a good enough Mapping and not only on producing one; and it may
take routing and resource Actions, which construction never offers, so it can
repair a congestion problem without moving anything. Neither is available to a
bounded annealing run whose whole neighborhood is realization rebinding.

The repair phase deliberately has no radius. The agent's every move is scored,
so an unhelpful long move is already paid for by its energy delta and its step
charge, and bounding it as well would price the same mistake twice while
removing the one capability — reaching across the fabric when that is genuinely
right — that distinguishes a learned repair from a local search.

That holds because a radius bounds an annealing run's proposal domain and not
the candidate's Action domain, per
[Annealing And Replay](spec-pnr.md#annealing-and-replay). The only radius that
governs anything in an episode is the one the arm's `cleanup_annealing_policy`
carries, and it governs the cleanup run alone.

## Action Space

The action surface, the `enumeration_bound` capacity, the masking algebra, and
the elective stop are owned by
[Action Surface And Masking](spec-ml-core-environment.md#action-surface-and-masking).

The enumeration is the candidate's own dynamic Action domain, in the owner's
canonical order. For the current candidate, the environment enumerates the
realization-binding choices, then the transport-routing choices, then the
resource-allocation choices, each anchor in canonical typed decision-key order
and each anchor's choices in that anchor's canonical choice order. An action is
one ordinal in that concatenation:

```text
EnumeratedPnrAction {
  kind: RealizationBinding | TransportRouting | ResourceAllocation
  anchor: ordinal in that kind's canonical anchor domain
  choice: ordinal within that anchor's contiguous choice range
}

ActionIndex = uint32
```

Nothing here is a new enumeration. `A(M,C,S)` is already deterministic, already
partitioned by kind, and already grouped into contiguous per-anchor choice
ranges; this document only fixes that kinds concatenate in the order shown and
that an `ActionIndex` is the resulting ordinal. Two runs with equal candidate,
frozen model, and resolved configuration therefore produce byte-identical
enumerations in the same order, and that follows from the owner's determinism
rather than from a rule here.

Phase decides which of those entries are live, and this is the only place the
environment clears a mask bit:

- in the construction phase, every entry whose kind is not `RealizationBinding`
  is cleared, and every `RealizationBinding` entry whose anchor is already
  placed is cleared;
- under `anchor_selection` of `CanonicalNext`, every `RealizationBinding` entry
  whose anchor is not the first unplaced realization in canonical order is also
  cleared;
- in the repair phase, nothing is cleared; and
- the stop outcome is cleared for the whole construction phase, and under
  `ConstructThenAnneal` for the whole episode.

Stop is cleared during construction because an episode that stops before every
realization is placed has produced nothing a Mapping could be made from, and an
always-available exit from a phase whose early rewards are negative is an exit
a policy learns to take. Under `ConstructThenAnneal` the agent never reaches a
phase where stopping is meaningful at all, so the outcome is cleared for the
whole episode; the episode ends when construction does.

A cleared mask bit is a phase rule and never a legality claim. Every entry the
mask clears is a legal Action of the current candidate, and the same entry
becomes live again when the phase changes. This is the opposite of the DSE
environment's per-state rejection mask, which records that a decision was tried
and failed; nothing here is masked because it failed.

The policy scores an action from the two nodes it names — the anchor it acts on
and the choice it selects — both of which are ordinary nodes of the observation
graph, named by the entry's own columns. That is what keeps an action a single
ordinal even though it is a pair.

## Observation

### The PnR Node And Link Space

The combined node-and-link space, its `GraphNodeRole` catalog, the
connection-as-node rule, the two enumeration encodings, and the target-closure
rule are owned by
[Combined Node And Link Space](spec-ml-core-environment.md#combined-node-and-link-space)
and [Enumeration Encoding](spec-ml-core-environment.md#enumeration-encoding).
This section states only what that space contains for a PnR episode.

Four role blocks are present in every episode, and this is the environment that
needs all four at once:

- `FabricOccurrence` spans every occurrence in the frozen model's Fabric
  projection, in canonical owner order — the hardware graph;
- `FabricConnection` spans every physical traversal between them, each
  replacing the direct arc it stands for — the routing resources, which must be
  nodes here rather than arcs because a routing Action names them and a route's
  occupancy is a per-traversal fact;
- `DataflowOperation` spans the Realizations of `T` — the software graph's
  nodes, which are what construction places; and
- `DataflowValue` spans the residual logical nets — the dependencies that
  routing closes.

The `Decision` block is absent, because this environment uses the
`DecisionColumns` encoding.

Placement arcs are unconditional here, unlike in the DSE environment where they
are a configuration option. The current mapping is not extra context in this
environment; it is the state. Each placed Realization node carries one
`Placement` arc to the occurrence bound to it, so "where is this node placed"
is one hop for the encoder rather than a numeric column it has to learn to
dereference.

When `include_route_edges` is set, each logical-net node additionally carries
one `Route` arc to every `FabricConnection` node its current RouteTree
traverses. This is the expensive part of the observation and it is optional for
that reason: a fully routed candidate has far more route incidences than
placements, and a policy that only places may not need them. A policy that
takes routing Actions in the repair phase does.

Every live action names exactly two nodes — its anchor and its choice — and it
names them by column rather than by arc. Both resolve under the target-closure
rule, which for this environment means every anchor and every choice of every
live entry is addressable: a realization binding names a Realization node and
an occurrence node, a routing Action names a logical-net node and, for a scoped
variant, the endpoint or traversal node its scope anchors on, and a
resource-allocation Action names its demand and its selected endpoint.

A PnR action names two nodes and never more, so its arity is fixed and the core
selects `DecisionColumns` for it. The overhead the core's rule warns about is
not marginal here: under `AgentSelected` the enumeration is the sum over
unplaced realizations of each one's legal occurrence domain, so an ordinary
problem enumerates far more actions than the design has entities.

What the two encodings share is the part that matters to a policy, and it is
the part this environment depends on: an action is scored from the embeddings
of the nodes it names. A DSE decision's value is a prototype ordinal or a
bounded delta, so a value term can read it from a column directly. A PnR
action's value is a graph node — an occurrence, an endpoint, a traversal — so
the column holds that node's ordinal and the value term reads its embedding. A
policy head that scored a PnR action from its anchor alone would make every
occurrence on one realization indistinguishable, which is the entire decision.

### Column Catalogs

The `Observation` and `GraphInstance` container shapes, the no-padding rule,
the negative-one absent sentinel, the buffer lifetime, and the obligations
every column catalog satisfies are owned by
[The Graph Instance](spec-ml-core-environment.md#the-graph-instance). The
closed column catalogs this environment owns are:

```text
PnrNodeFeatureColumn =
    Role                      // 0
  | EntityKind                // 1
  | CapabilityCount           // 2
  | CapacityMagnitude         // 3
  | CapacityUsage             // 4
  | CapacityOveruse           // 5
  | BufferDepth               // 6
  | InDegree                  // 7
  | OutDegree                 // 8
  | SelfCycle                 // 9
  | Placed                    // 10
  | UnroutedObligationCount   // 11
  | RouteClaimCount           // 12
  | TagUnassignedCount        // 13
  | TagConflictCount          // 14

PnrArcRole =
    Structural                // 0
  | Placement                 // 1
  | Route                     // 2

PnrDecisionColumn =
    ActionKind                // 0
  | AnchorNode                // 1
  | ChoiceNode                // 2
  | ChoiceDistance            // 3

PnrScalarFeatureColumn =
    StepOrdinal               // 0
  | StepBound                 // 1
  | Phase                     // 2
  | PlacedCount               // 3
  | RealizationCount          // 4
  | UnmaskedActionCount       // 5
  | RepairStepCount           // 6
  | RepairStepBound           // 7
  | ConsecutiveFailures       // 8

EpisodePhase =
    Construction              // 0
  | Repair                    // 1
```

`EntityKind` is the owner-local reference kind ordinal from
[Fabric Identity](spec-fabric-identity.md) for a Fabric node and the Dataflow
or TechMapping owner's kind ordinal for a Realization or net node, so a new
Fabric occurrence kind extends the observation without a new column and without
a new role.

`CapacityMagnitude`, `CapacityUsage`, and `CapacityOveruse` are the raw
declared capacity, the raw current usage, and the raw current overuse of the
referenced entity, projected from the candidate's own occupancy caches. Overuse
is carried separately rather than left to be derived from the other two because
it is the quantity the objective prices and a policy should read it directly
rather than reconstruct it. The typed atoms of the `FabricResourceStateRef`
catalog the core appends under `include_resource_states` are owned by
[Fabric Resource Contract](spec-fabric-resource-contract.md).

`Placed` carries the construction bitset on a Realization node. Which
occurrence a Realization is bound to, and which Realization occupies an
occurrence, are not columns: the `Placement` arc carries that relation in both
directions, so a policy reads it in one message-passing hop rather than by
dereferencing an ordinal, which is a thing an embedding space cannot do.

`UnroutedObligationCount`, `RouteClaimCount`, `TagUnassignedCount`, and
`TagConflictCount` are per-node projections of four of the five Mapping
violation magnitudes and of the traversal claim, attributed to the entity that
carries them. The fifth violation, hard progress, is a whole-candidate fact
with no per-node attribution and appears only through `objective_codes`.

`PnrDecisionColumn` is the row catalog of `decisions`, one row per live entry
in the enumeration order Action Space defines, so a row's ordinal is its
`ActionIndex`.

`AnchorNode` and `ChoiceNode` hold node ordinals into the same observation's
`graph`, under the core's rule for a target-naming column. They are the entry's
complete reference set, and the target-closure rule binds them exactly as it
binds a target arc.

`ActionKind` carries the entry's `SpatialMappingAction` kind.

`ChoiceDistance` is the directed frozen-topology hop distance from the anchor's
current occurrence to the choice's occurrence for a realization binding, and
the absent sentinel otherwise. It is the one derived column in any of this
environment's catalogs, and it is derived because distance is the single most
load-bearing quantity in placement while being the one a graph encoder is worst
at recovering: reading it off the embeddings would mean propagating information
across as many message-passing layers as the fabric has hops, so a policy on a
large fabric could not represent it at all at any practical depth. Its two
endpoints are Fabric occurrences and the metric is the frozen topology's, so
the value for one occurrence pair is constant for the whole problem and an
entry's distance changes only when its anchor rebinds.

The selected closure whose codes `objective_codes` carries is
`pnr_config_view.selected_objective_closure`.

A failed step neither advances nor resets, so under the core's buffer-lifetime
rule it does not invalidate the buffers; the candidate was rolled back to the
value it already had, and only the step and failure scalars change in place.

## Episode Start

An episode is created from one problem binding. There is no workload set and no
seed design: the problem is fixed for the whole episode, and what varies is the
candidate.

`reset` performs this ordered protocol:

1. derive the episode's PRNG streams from the effective seed and the copy
   coordinates;
2. select one `SpatialPnrProblemBinding` from `problem_pool` through
   `ProblemSelection`, or adopt the one the start override supplies;
3. under `ConstructThenAnneal`, adopt the effective seed as the cleanup seed
   index when a start override is present, and otherwise draw it from
   `CleanupSeeding`, in either case on every episode of that arm whether or not
   construction ever completes;
4. acquire the `FrozenModel` and the canonical seed state for that binding and
   `pnr_config_view` from the retained cache, or publish them by freezing and
   building;
5. adopt the canonical candidate, its objective vector, its search energy, and
   its settled bitset, and mark every realization unplaced;
6. adopt the retained canonical enumeration and its refusal verdict; and
7. build the first observation.

Step 3 precedes step 4 so that the stream position after `reset` does not
depend on whether a cache hit occurred, which is what keeps a run with a warm
cache and a run with a cold one formally identical.

The canonical seed state is retained beside the `FrozenModel` and under the
same key, because it is constant under the same key. The canonical candidate is
the owner's attempt-zero initializer output, and its objective vector, search
energy, settled bitset, first enumeration, and that enumeration's empty-or-
over-capacity verdict are pure functions of it — none reads the episode's seed,
its streams, or anything an episode has yet done. The enumeration is retained
with the rest, which matters most of all: under `AgentSelected` it spans every
unplaced realization's occurrence domain and is the largest of the episode.
Every episode drawing one problem would otherwise rebuild and re-evaluate an
identical value, once per episode for the whole run. Steps 5 and 6 are
therefore copies, and the energy is not recomputed at all but carried. This is
retention on the same terms as the frozen model: a hit and a miss produce the
same formal result.

Step 6 precedes step 7 because refusing is cheaper than projecting a graph the
episode will not use, and a configuration that could never offer a legal action
should be rejected without paying for an observation.

An override adopts the effective seed rather than drawing, because
`CleanupSeeding` is derived from a preimage that includes the copy coordinates
and an overridden episode must not depend on them. Drawing there would make a
`ConstructThenAnneal` case anneal differently on each runner, which is
precisely the reproducibility the override exists to provide and which the test
protocol depends on. The effective seed is the caller's own input, so a case is
determined by its problem and its seed alone.

The episode start override this environment defines is one
`SpatialPnrProblemBinding`, carried by the Gymnasium `options` argument. It is
the payload itself rather than a record wrapping it, since a record with one
field is a name for that field.

The core's override rules bind it: it must appear in `problem_pool`, and when
present it replaces step 2 entirely, so `ProblemSelection` is not consulted.

```text
PnrEpisodeStartOutcome =
    Started
  | ProblemProvenInfeasible { problem, diagnostic }
  | FreezeCapacityExceeded { problem, required_index_width }
  | EnumerationBoundExceeded { problem, bound, required }
  | Invalid { violated precondition }
```

`ProblemProvenInfeasible` reports that freeze proved an empty well-formed
domain, which is a sound proof about the problem and not about this episode; it
is reported rather than retried, because every retry would draw from the same
pool and a pool member that is infeasible is infeasible on every draw.
`FreezeCapacityExceeded` is separate because it is a Loom build-capacity error
rather than Mapping infeasibility, exactly as
[Native State](spec-pnr.md#native-state) requires, and it names the required
`LOOM_PNR_INDEX_BITS` width so the remedy is stated rather than guessed.

`EnumerationBoundExceeded` reports a canonical candidate with more admissible
Actions than the action space can index, naming both the capacity and the
required length so the configuration can be corrected rather than guessed at.
It is separate from `Invalid` for that reason: the core's refusal rule requires
both numbers, and folding it into a violated-precondition report would drop
them.

`Invalid` covers a canonical candidate with no live entry, which cannot happen
for a well-formed problem with at least one movable realization and is
therefore a projection defect rather than a state to retry from. It also covers
a binding whose exact `D`/`T`/`F`/`K` coupling does not hold and a problem
outside `problem_pool` named by an override.

There is no retry budget, and its absence is the point. A DSE episode redraws
because a seed may be unusable; a PnR problem that adoption accepted is usable
by construction, so a failure here is a configuration or owner defect and
retrying would hide it.

## Step

One step applies exactly one enumerated Action. The protocol is ordered, and
stages 1 through 5 may fail:

1. validate the action against the live mask;
2. decode the index to a typed `SpatialMappingAction` against the current
   enumeration;
3. probe the transition, which computes its complete dependency closure,
   applies it in a shadow candidate, closes every affected route, and evaluates
   the resulting `V/G` and the selected search energy;
4. rebuild the dynamic Action domain against the probed shadow candidate, and
   fail when its length exceeds `enumeration_bound`;
5. commit the probe when stages 3 and 4 both succeeded, and roll it back
   otherwise;
6. mark the anchor placed when stage 5 committed a construction binding, and
   settle every realization the commit left with no alternative;
7. run the cleanup and end the episode when stage 6 placed the last realization
   under `ConstructThenAnneal`, or advance the phase to `Repair` under
   `ConstructThenRepair`;
8. build the observation over the candidate the step left behind, applying the
   phase mask to that candidate's domain.

Stage 8 names the resulting candidate rather than stage 4's, because stage 4
builds against a shadow that a failed step discards; masking that domain after
a rollback would describe a candidate the episode does not hold. A failed step
leaves the candidate, its domain and its mask exactly as they were, so it
recomputes none of them — the buffers are still the ones the core's
buffer-lifetime rule keeps valid.

A step that ends the episode returns an observation of the state it ended in,
carrying an empty `decisions` and an all-clear mask, which
[Action Surface And Masking](spec-ml-core-environment.md#action-surface-and-masking)
exempts from the admissible-outcome rule precisely because no action is ever
sampled from a terminal observation. That is what keeps the cleanup outside the
capacity contract: stage 7 rebinds realizations and can leave a candidate whose
domain exceeds `enumeration_bound`, and there is nothing to roll back to by
then, but the observation that reports it enumerates nothing and so has nothing
to exceed.

Stage 3 is the whole of the owner's online protocol and this document adds
nothing to it. In particular there is no acceptance test: the annealing policy
resolves a probe against a temperature and may reject an
improving-in-expectation move, whereas the policy here has already chosen, so
every probe that the owner does not fail and stage 4 admits is committed.
Reintroducing an acceptance kernel on top of a learned proposal would put two
selectors in series and make the reward describe neither.

Stage 4 precedes the commit because that is the only place the capacity test
can protect the candidate. The enumeration is a function of the candidate, so
it does not exist until the transition has been applied; testing it after the
commit would leave the episode holding a state the observation cannot
represent, with nothing to roll back to. Rebuilding against the probe's shadow
candidate tests the state the step would produce while the step can still be
undone, which is what lets `EnumerationBoundExceeded` be an ordinary transition
failure that leaves the candidate unchanged.

The enumeration's length does not depend on the phase, which is why the phase
transition at stage 7 needs no capacity test of its own. The domain is the
candidate's complete dynamic Action domain under every phase; what the phase
changes is the mask, and clearing a mask bit removes no entry. Opening the
repair phase therefore raises `UnmaskedActionCount` and leaves the enumeration
exactly the length stage 4 already admitted. The two are different quantities
and are named differently for that reason: the enumeration's length is how many
entries `decisions` carries, which the mask never changes, while
`UnmaskedActionCount` is how many of them the policy may currently select.

The outcome algebra is closed:

```text
PnrStepResult {
  transition:  optional<PnrStepTransition>
  episode_end: optional<PnrTerminalReason>
}

PnrStepTransition =
    Advanced { energy_delta_code, energy_delta_sign }
  | Failed { reason: PnrTransitionFailureReason }

PnrTransitionFailureReason =
    IntrinsicInvalid { diagnostic }
  | WorkLimit { diagnostic }
  | ObjectiveUnavailable { dimension }
  | EnumerationBoundExceeded { bound, required }

PnrTerminalReason =
    ElectiveStop
  | CleanupComplete
  | RepairBoundReached
  | StepBoundReached
  | ConsecutiveFailureBoundReached
  | EnumerationEmpty
```

`IntrinsicInvalid` and `WorkLimit` are exactly the two members of the owner's
transition-failure taxonomy, and keeping them apart is the same
proof-versus-budget discipline the rest of the stack uses. `IntrinsicInvalid`
means the Action cannot produce a legal candidate — a newly closed directed
handshake cycle is the canonical case — and is a fact about the Action.
`WorkLimit` means the router exhausted its configured budget for this move and
establishes nothing at all about whether the Action was good. A blended reason
would let a rising router-budget problem read as a policy that proposes illegal
moves.

`ObjectiveUnavailable` is this environment's name for the core's
unavailable-objective outcome, and its transaction is rolled back with the
rest.

A transition and an episode ending are separate facts because they co-occur, on
the terms
[Step Accounting And Episode Endings](spec-ml-core-environment.md#step-accounting-and-episode-endings)
states. The step that places the last realization under `ConstructThenAnneal`
both advances and ends the episode; the failure that reaches
`consecutive_failure_bound` both fails and ends it; an elective stop ends it
with no transition at all. The core's `non_advancing` class is `Failed` here.

`ElectiveStop`, `CleanupComplete`, and `EnumerationEmpty` are terminations;
`RepairBoundReached`, `StepBoundReached`, and `ConsecutiveFailureBoundReached`
are truncations, each cutting the episode off by a configured limit while the
candidate remained ordinary.

## Reward

The per-step reward is the signed difference of the selected search energy
across the transition, on the terms
[Reward Contract](spec-ml-core-environment.md#reward-contract) states. The
parent energy is retained from when the parent was entered and is not
recomputed, so a step evaluates one energy.

This is the same reward shape the DSE environment uses and it is worth naming
what it amounts to here: with energy as a potential, a per-step signed energy
difference is potential-based shaping, so the return of an episode telescopes
to the total improvement from the canonical candidate to the final one, and the
dense per-step signal changes what is learnable without changing what is
optimal. That the weighting inside it — violations against traversal claim —
comes from the resolved closure rather than from constants in this document is
what lets one environment serve a search that cares about congestion and one
that cares about latency.

Three terminal terms are charged on top, and each answers one requirement of
its arm.

`ConstructThenAnneal` charges the product of `cleanup_displacement_reward_code`
and `cleanup_displacement` with a negative sign on the step that ends the
episode, alongside that step's ordinary energy delta, which is the energy the
cleanup recovered. The episode's return is therefore the final Mapping's
quality minus the amount of cleanup that quality required. Both halves are
necessary: without the energy term the policy is not scored on the design at
all, and without the displacement term it learns that a mediocre construction
is free because the annealer will fix it.

`ConstructThenRepair` charges `repair_step_reward_code` with a negative sign on
each committed repair step. The episode's return is the final Mapping's quality
minus how long repair took, which is the same trade the other arm makes with
the annealer's work in place of the agent's.

Both arms charge `incomplete_closure_reward_code` with a negative sign on the
final step of any episode whose final candidate is not closed, meaning its five
Mapping violation magnitudes are not all zero. Those magnitudes are the ones
the probe already maintains incrementally and evaluated for this very
candidate, so the charge reads a value the step produced rather than triggering
a recomputation — which is also what keeps the rule consistent with this
environment never running independent final verification.

That last charge is deliberately not an ending price, and the distinction
matters because the core prices a truncation at zero. What the core refuses to
charge for is the *limit*, which the policy did not choose. Closure is a
property of the candidate, which the policy did choose, and a truncated episode
that leaves an unclosed candidate has failed at the task regardless of why it
stopped. Pricing the limit and pricing the state are different charges and only
the first is forbidden.

A `Failed` transition yields `failed_transition_reward_code` with a negative
sign, independent of the reason. It does not distinguish `IntrinsicInvalid`
from `WorkLimit`, because charging more for one would make the reward depend on
a router budget.

`cleanup_displacement_reward_code` multiplied by a displacement over a large
design is the one product here with real magnitude, and it is checked like
every other.

## Trajectory Retention And Replay

An episode retains one transient record:

```text
PnrTrajectory {
  config_view_digest: ComponentViewDigest
  problem: SpatialPnrProblemBinding
  effective_seed: u64
  env_runner_index: uint32
  vector_index: uint32
  local_episode_index: uint64
  cleanup_seed_index: optional<u64>
  advanced_actions: ordered sequence<EnumeratedPnrAction>
}
```

`cleanup_seed_index` is present exactly under `ConstructThenAnneal`.

Replay is deterministic re-execution. Given the same configuration digest, the
same problem binding, and the recorded action sequence, replaying against a
`FrozenModel` for that binding reproduces an identical final `CandidateState`,
an identical objective vector, and an identical energy. Nothing about the copy
coordinates or the seed is needed for that, because the recorded actions
already name every choice the streams would have made; the seed fields are
retained so that the episode itself can be regenerated rather than only its
outcome.

Turning a replayed candidate into a published `SpatialMapping` is outside this
document. A candidate becomes a Mapping only through the canonical search
sequence's own path — final global negotiated closure, full owner
recomputation, independent verification, and finalization — and reaching that
path from a learned policy means invoking the policy as a search-policy
selector inside an ordinary Spatial PnR invocation, which is a change
[Search Policy And Determinism](spec-pnr.md#search-policy-and-determinism) owns
and this document does not make. Until that exists, this environment produces
trained policies and measurements, not Mappings. Stating the boundary is
preferable to a replay rule that publishes from a harness, which would put a
training artifact on the product path.

## Python Boundary

The `loomml` package layering, the RLlib conformance target and its
obligations, the zero-copy array contract, the outcome-versus-exception rule,
and the threading and `fork` rules are owned by
[Python Boundary](spec-ml-core-environment.md#python-boundary). This
environment occupies `loomml.env.pnr` and `loomml.rllib.pnr` and adds no rule
of its own.

One consequence is worth naming because it is this environment's dominant
implementation constraint. The native layer holds a `FrozenModel`, a
`CandidateState`, and the owner's reusable scratch, none of which crosses into
Python; what crosses is the observation buffers and one action index. A step is
therefore a native transaction plus one marshalling boundary, and the shared
`FrozenModel` is what makes several vector copies in one process affordable.
Under the core's rule that an instance is owned by one thread and shares no
mutable state, the copies share only that immutable model.

## Benchmarking

The harness obligations, the required breakdowns, the instrumentation rule, and
the ratio-based regression budget are owned by
[Benchmarking Harness Contract](spec-ml-core-environment.md#benchmarking-harness-contract).
This environment ships `loom-pnr-env-bench`, whose failure-reason breakdown is
keyed by `PnrTransitionFailureReason` and whose action partition is the
`SpatialMappingAction` kind, because a routing Action and a binding Action do
not cost remotely the same.

The harness decomposes one step into these stages and reports each separately:

```text
enumeration     rebuild the dynamic Action domain and the phase mask
decode          map the action index to a typed SpatialMappingAction
probe_closure   the shadow transition and its incident route closure
probe_objective incremental V/G recomputation and search-energy evaluation
resolve         commit or roll back the transaction
cleanup         the bounded annealing run and its displacement projection
observation     build the graph and the decision columns
marshal         expose buffers across the Python boundary
```

`probe_closure` and `probe_objective` are the two halves of the single probe
the step protocol performs, split here because they are the two costs worth
telling apart and reported as one probe nowhere. Stages partition the step;
none contains another, so the stage sum is the step.

A reset decomposes into `problem_lookup`, `freeze`, `seed_state_build`,
`enumeration`, and `first_observation`. The seed state's build and evaluation
are one stage because retention makes them one event: on a hit neither runs.
`seed_state_build` covers the canonical enumeration as well, since retention
makes the whole seed state one event: on a hit none of it runs. It is reported
separately from `freeze` because the two differ by orders of magnitude and a
blended `reset` percentile over a mixed hit-and-miss population describes
neither.

Two further breakdowns are required beyond the core's two.

Every measure is reported per phase the arm actually runs. A construction step
and a repair step run the same code over candidates with completely different
route densities, and a blended percentile moves whenever an episode's phase mix
moves, which it does throughout training as the policy learns to finish
construction faster. Under an arm with one phase the breakdown is that phase,
reported without a second empty column, and the core's action partition
collapses the same way: it is over the kinds the bound arm can actually make
live, which under `ConstructThenAnneal` is `RealizationBinding` alone.

And `reset` is reported with the frozen-model cache hit rate alongside it,
separated into hit and miss populations rather than blended, because a single
`reset` percentile over a mixed population describes neither and moves with the
pool size rather than with any code change. The hit rate is the number that
decides whether the environment is affordable at all, and it is reported as a
first-class measure and not as a footnote to a latency.

The harness reports the deterministic PnR work-unit counts — assignment
attempts, endpoint expansions, negotiation iterations — alongside its timings
and states which is which, because those are the cross-machine cost measure and
the timings are not.

## Anchor Verification

Stable tests cover adoption rejecting a `problem_pool` member whose `T.D`,
`T.F`, or `K.D/T/F` binding is not exact; adoption rejecting a
`ConstructThenAnneal` arm whose `cleanup_annealing_policy` carries an
`Unbounded` `realization_move_radius`, and a `ConstructThenRepair` arm whose
`repair_step_bound` is zero; the enumeration equalling the owner's dynamic
Action domain in canonical kind, anchor, and choice order, and being
byte-identical across two runs with equal candidate and configuration; the
construction mask clearing every non-`RealizationBinding` entry and every entry
whose anchor is already placed, and `CanonicalNext` additionally clearing every
anchor but the canonically first unplaced one; the repair phase clearing
nothing; the stop outcome being clear for the whole construction phase and for
the whole of a `ConstructThenAnneal` episode; a cleared entry becoming live
again when the phase changes, and no cleared entry ever being reported as
illegal; the `Decision` block being absent and `decisions` carrying one row per
enumerated entry and no arcs, in the enumeration order, with a row's
ordinal equalling its `ActionIndex`; every live entry's `AnchorNode` and
`ChoiceNode` resolving to exactly one node ordinal in the same observation's
`graph`, including an occurrence-valued, a traversal-valued, an
endpoint-valued, and a net-valued reference; `ChoiceDistance` equalling the
owner's directed frozen-topology hop distance for a realization binding and the
absent sentinel for every other kind; a `Placement` arc joining a placed
Realization node to its occurrence node in both directions and none joining an
unplaced one; route arcs being absent when `include_route_edges` is clear and
spanning exactly the current RouteTree traversals when it is set;
resource-state columns being absent when `include_resource_states` is clear; a
committed binding closing every incident net through the owner's own dependency
closure with no environment-issued routing Action; a net to an unplaced
realization remaining an ordinary candidate violation rather than a step
failure; construction advancing to the repair phase or to the cleanup exactly
when the last realization is placed and never before; a `ConstructThenAnneal`
episode running exactly one cleanup, reporting `CleanupComplete` on the same
step that placed the last realization, and returning the post-cleanup
observation; `cleanup_displacement` being zero exactly when no occurrence
changed, and counting both the hop sum and the moved count; a
`ConstructThenRepair` episode charging exactly one `repair_step_reward_code`
per committed repair step and none during construction; an unplaced realization
with no alternative to its current binding being settled without a step at
`reset` and again after a commit that shrinks its domain to a singleton, and
construction never reaching a state with realizations outstanding and no live
entry; an `IntrinsicInvalid` rollback leaving the candidate equal to its
pre-probe value in every selected decision and every rebuildable cache;
`IntrinsicInvalid` and `WorkLimit` never being collapsed into one reason;
`ObjectiveUnavailable` producing no numeric reward and rolling back its
transaction; a transition whose resulting enumeration exceeds
`enumeration_bound` failing with `EnumerationBoundExceeded` before the commit
and leaving the candidate unchanged; the enumeration length being identical
before and after the construction-to-repair phase advance while
`UnmaskedActionCount` rises while the enumeration's length does not; a masked,
stale, or out-of-range index being refused
with the candidate unchanged; the step accounting identity holding over a
complete episode with `Failed` as its non-advancing class; termination and
truncation being reported distinctly for their respective terminal reasons; a
truncation yielding zero for the ending itself while still charging
`incomplete_closure_reward_code` when its final candidate is unclosed, and
charging nothing when it is closed; the frozen model being reused across two
episodes drawing the same problem binding and not being reused under a changed
the freeze-relevant projection of `C` or a changed `K` identity, and being
reused across a changed `selected_objective_closure`; a warm-cache run and a
cold-cache run producing identical stream positions and identical trajectories;
`ProblemProvenInfeasible` and `FreezeCapacityExceeded` being reported
distinctly and neither being retried; a canonical candidate whose enumeration
exceeds `enumeration_bound` reporting `EnumerationBoundExceeded` with both the
capacity and the required length rather than `Invalid`; a `ConstructThenAnneal`
case under a start override annealing identically at two runner counts and two
copy coordinates; a settled realization whose domain regains an alternative
returning to unplaced and re-entering the enumeration; the observation returned
on a step that ends the episode carrying an empty `decisions` and an all-clear
mask, and no non-terminal state ever presenting one; a failed step leaving the
enumeration and the mask it had before the step rather than those of the
discarded shadow candidate; a `pnr_config_view` whose
`realization_move_radius` being adopted at either arm and changing no
enumeration this environment produces; an episode start override consulting no
`ProblemSelection` draw and
producing the same problem on every copy, and one naming a problem outside
`problem_pool` being rejected; two environment copies with distinct coordinates
drawing disjoint problem sequences from one seed; a `reset` seed argument
becoming the effective seed and being recorded as such; and equal
configuration, coordinates, seed, and action sequence reproducing an identical
trajectory, an identical final `CandidateState`, and an identical search
energy.

Tests do not pin the live node, arc, or action counts of any particular
candidate, the bound or radius values a configuration selects, which occurrence
a policy chooses, achieved energies or closure rates, cleanup displacement
values, wall-time numbers, per-kind cost ratios, cache hit rates, corpus
contents, diagnostic text, or Python formatting.
