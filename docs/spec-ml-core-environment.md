# ML Environment Core

This document defines the contract every Loom reinforcement-learning search
environment shares: how a search state is projected into one observation, how
an action index addresses one member of a live enumeration, how an episode's
transitions and endings are accounted for, how reward crosses the integer
boundary, how a parallel sampler stays reproducible, and how the whole thing is
presented to Python and measured.

It defines no design space of its own. What is being searched, which decisions
exist, what makes one legal, and what an episode is trying to achieve are owned
by the environment documents that build on this one:

- [ML DSE Environment](spec-ml-dse-environment.md) searches Loom's design space
  by selecting candidate-generator decisions over a Fabric subject; and
- [ML PnR Environment](spec-ml-pnr-environment.md) searches one fixed Mapping
  problem by selecting typed Place and Route Actions.

Two environments already share more machinery than either owns alone, which is
why the machinery is here rather than in whichever document happened to need it
first. A third environment appends a bullet above and states the small set of
obligations each section below names; it does not reopen this contract.

## Ownership

Every fact this document depends on resolves to one exact owner:

- [Evaluation and DSE](spec-dse-feedback.md#objectives-and-quality-gates) owns
  `ObjectiveDimension`, `ExactAffineQuantization`, `ObjectiveVector`,
  `WeightedLevel`, `TotalOrdering`, `SearchEnergyRef`, and the statement that a
  reward is the signed difference of a selected search energy;
- [Search Policy And Determinism](spec-pnr.md#search-policy-and-determinism)
  owns the seeded PRNG protocol, its `nextBounded` projection over canonically
  sorted domains, and its prohibition on host entropy and library
  distributions;
- [Evaluation Metrics](spec-evaluation-metrics.md) owns `ExactRatio`;
- [Resolved Configuration](spec-config-ssot.md#component-views) owns
  component-view framing, canonical view bytes, and `component_view_digest`;
- [Operational Observations](spec-dse-feedback.md#operational-observations)
  owns the nonsemantic status of wall time, the prohibition on summing
  concurrent times, and deterministic work summaries as the cross-machine cost
  measure; and
- [Full-Stack Architecture](spec-loom-stack.md#external-dependency-pinning)
  owns the exact revisions of every search-harness dependency, including the
  Ray fork the RLlib section targets.

The RLlib environment definition, its env context, its registration mechanism,
and its sampling topology are owned by that external dependency. This document
states which of its obligations bind an environment and does not restate,
extend, or version them; the training document that configures an environment
configures it.

This document owns only the observation container, its combined node-and-link
space, the two enumeration encodings, and the storage form of every observation
array including the packed mask layout; the action surface and masking algebra;
the step accounting identity and the termination-versus-truncation rule; the
reward boundary; the determinism record, the PRNG preimage shape, and the
episode-start override contract; the trajectory record's status; the `loomml`
package layering and its interaction contract; and the benchmarking obligations
every harness satisfies.

## Nonsemantic Boundary

An ML environment is a nonsemantic search harness. It is not a Candidate
Generator, Mapping, Evaluation, objective, configuration, or Artifact
authority. It defines no action language of its own and no second design space.
Every decision it applies is an existing owner-typed decision, every legality
answer comes from that decision's ordinary owner, and every reward is a
projection of the objective algebra owned by
[Objectives and Quality Gates](spec-dse-feedback.md#objectives-and-quality-gates).

No environment operation publishes an `EvaluationRequest`,
`EvaluationEvidence`, an `InvocationManifest` record, or a lineage edge, and no
environment value enters a Fabric, Mapping, or Evaluation Artifact. An
observation column, a reward, a mask bit, a step record, and every output of
every harness in the ML stack are removable projections; regenerating them may
change presentation but must preserve every referenced semantic fact, and no
consumer may treat one as a semantic result. Nothing on that list is an
Artifact, Evidence, or a report schema, and no selection, promotion, or
conformance decision consumes one.

An environment introduces no persistent schema family beyond its own resolved
configuration, which is an ordinary component view. Its episode state is
process-local and its trajectory record is a transient value: a trajectory is
not an Artifact, not Evidence, and not a plan, and it carries no outcome, no
reward, and no selection claim. Only the steps that advanced are retained, so a
non-advancing attempt stays an episode-local observation and never becomes
fake history.

An environment run is therefore never an authority for selection. A result an
environment discovers becomes real only when its environment document states an
ordinary owner-owned path that reconstructs it.

## Combined Node And Link Space

Decisions target entities and relations alike. A decision may name an
occurrence, a connection, a transport link, an actor, a graph, or a route, so a
projection that represented only entities as graph nodes would leave every
relation-valued decision pointing at nothing. The observation is therefore a
combined node-and-link space: an entity that another representation would draw
as an arc is instead a node, and the arc it replaces becomes two arcs through
it.

Nodes are laid out in role blocks in this order, each block dense and in
canonical owner order:

```text
GraphNodeRole =
    FabricOccurrence    // 0
  | FabricConnection    // 1
  | DataflowOperation   // 2
  | DataflowValue       // 3
  | Decision            // 4
```

Ordinals are stable. A new role appends; reordering, deleting, or repurposing
one is an incompatible change to this document's schema version. An environment
whose state contains no member of a block contributes no rows for it and needs
no flag to say so, because its own contract already decides which blocks exist.

A `FabricConnection` node replaces the direct arc it stands for: an original
connection from occurrence `u` to occurrence `v` contributes node `l` and the
two arcs `u -> l` and `l -> v`. `DataflowValue` nodes are promoted from values
under the same rule. A parallel connection is one node carrying its current
count as a feature, not one node per parallel lane, so a decision that changes
a count has exactly one target and never changes the node inventory.

"Decision" is this document's generic name for one member of the live
enumeration; an environment document names what its members actually are. The
`Decision` block is present exactly when the environment uses the
`DecisionNodes` encoding defined below.

The space is closed under decision targets. For every member of the live
enumeration, every owner-local reference its normalized payload carries
resolves to exactly one node ordinal in this graph. A reference that resolves
to no node, to more than one node, or to a node outside the current state is a
projection defect rather than an unaddressable action. This closure is what
lets the enumeration and the graph agree: a policy scores a decision by
attending over the nodes it names, so an action is per-node and per-link even
though the action index itself is an ordinal in a canonical order.

## Enumeration Encoding

The closure rule says every live entry names graph nodes. How it names them is
one of two forms:

```text
EnumerationEncoding =
    DecisionNodes    // 0
  | DecisionColumns  // 1
```

Under `DecisionNodes`, each live entry is a node in the `Decision` block and
each reference it carries is an arc from that node to the node it names. An
entry's target count is its out-degree, its target roles are arc features, and
its own categorical and numeric properties are columns of the node matrix. A
decision's ordinal within the `Decision` block is its `ActionIndex`.

Under `DecisionColumns`, the `Decision` block is absent and the enumeration is
carried alongside the graph as its own dense matrix, one row per live entry.
The references an entry carries are columns holding node ordinals into the same
instance's graph, and its own properties are further columns of the same row.
An entry's `ActionIndex` is its row ordinal.

The choice between them is not free. **An enumeration whose entries have
variable target arity uses `DecisionNodes`; one whose entries have fixed arity
uses `DecisionColumns`.** A decision that names one entity plus a reference set
of unbounded size needs out-degree to express its targets, and columns would
force either a maximum arity or an offset table beside the matrix. An action
that names exactly an anchor and a choice needs two columns, and a node with
two arcs says the same thing while adding a row and two arcs to the graph for
every member of an enumeration that is routinely far larger than the state it
ranges over. Neither form is a preference: an environment whose arity is fixed
and that chose nodes anyway would spend most of its encoder on its own action
set.

What is common to both is the part a policy actually consumes. An entry names
graph nodes, those nodes carry the encoder's representation of what the entry
acts on, and the entry's index is an ordinal in the canonical order. A model
scores an entry from the embeddings of the nodes it names under either
encoding; what differs is where it reads the naming from.

## The Graph Instance

The observation is one variable-size graph, carried by the Gymnasium `Graph`
space and produced as one `GraphInstance` per state. It is not padded, not
bounded, and not reshaped to a fixed extent: the node count, arc count, and
their feature matrices are exactly the size the state requires, and a state
that needs more room simply produces a larger instance.

```text
Observation {
  graph:            GraphInstance
  decisions:        GraphInstance
  action_mask:      packed bitmask over enumeration_bound + 1 outcomes
  objective_codes:  dense array<ObjectiveDimensionRef, uint32>
  scalar_features:  dense array<ScalarFeatureColumn, int64>
}

GraphInstance {
  nodes:      per-column array<node ordinal, NodeFeatureColumn>
  edges:      per-column array<arc ordinal, ArcFeatureColumn>
  edge_links: dense array<(arc ordinal, 2), uint32>
}
```

`decisions` carries the enumeration under `DecisionColumns` and is empty under
`DecisionNodes`, where the enumeration is already in `graph`. It is a
`GraphInstance` with rows and no arcs rather than a bare matrix, because the
observation already needs one ragged per-instance carrier and this is it: the
same space, the same batching, and no second capability asked of a consumer
that can already carry one. Its `nodes` matrix is indexed by live entry ordinal
over the environment's own `DecisionColumn` catalog, and its `edges` and
`edge_links` are empty.

Under `DecisionNodes`, an entry's targets are arcs from its decision node, so
its target count is its out-degree and nothing needs an offset array, a slot
table, or a per-decision count column. This is the same move the combined space
makes for connections: an entity with a variable number of relations becomes a
node, and the relations become arcs.

`objective_codes` holds the current directed code of each dimension in the
environment's selected closure, in ascending `ObjectiveDimensionRef` order,
exactly as produced by the quantization owned by Evaluation and DSE. It is the
state's objective position, not a reward.

`NodeFeatureColumn`, `ArcFeatureColumn`, `ScalarFeatureColumn`, and
`DecisionColumn` are closed catalogs each environment document owns, because
the facts worth exposing about a Fabric being edited and about a Mapping being
built are not the same facts. A shared catalog would have to be the union of
both and would carry an inapplicable majority in either environment. Six
obligations bind every such catalog:

1. `NodeFeatureColumn` ordinal zero is `Role`, carrying the node's
   `GraphNodeRole`, so a consumer can interpret every other column without
   knowing which environment produced the instance;
2. a column that does not apply to a node's role is encoded as negative one, so
   one node matrix with role-conditioned columns keeps every node addressable
   by one ordinal, which is what lets a policy attend over entity and decision
   nodes together;
3. `ArcFeatureColumn` ordinal zero is `ArcRole`, whose ordinal zero is
   `Structural`, whose ordinal one is `Placement` where the environment
   projects a placement relation, and whose ordinal two is `Route` where it
   projects a routing relation; an environment appends its own roles after
   those and omits the ones it does not project. A target arc's role is carried
   by `ArcRole` itself rather than by a second column that would be
   inapplicable on every structural arc. An environment with no second arc
   column declares its `ArcRole` catalog and no `ArcFeatureColumn` wrapper,
   since a one-member catalog is a name for its member;
4. a categorical column holds an owner catalog's ordinal rather than an
   environment-local enum wherever such a catalog exists, so a new owner kind
   extends the observation without a new column and without a new role; and
5. a `DecisionColumn` that names a target holds a node ordinal into the same
   instance's `graph`, and the target-closure rule binds it exactly as it binds
   a target arc. A `DecisionColumn` catalog is empty under `DecisionNodes`; and
6. a node catalog is extended by one capacity column and one usage column per
   member of the state's fixed canonical `FabricResourceStateRef` catalog, in
   that catalog's order, exactly when the environment sets
   `include_resource_states`. The appended group is fixed for the episode, so
   the catalog keeps a fixed column count even though its row count varies.
   This group is stated
   here rather than per environment because every environment appends the same
   one from the same owner catalog; an environment needing a second, different
   appended group states that one itself.

Every array is C-contiguous with the declared element type; its lifetime and
the prohibition on writing it are owned by
[Interaction Contract](#interaction-contract). Column order within each catalog
is part of this contract; row counts are not, and no consumer may infer a state
fact from an extent beyond the count that extent literally is.

### Storage Widths

Every observation array is stored in the narrowest form that represents its
declared value domain exactly. Nothing is stored wider to make two arrays the
same type.

The domain fixes the form, so no catalog declares a width. A boolean domain is
one bit per entry, packed. A bounded-ordinal or bounded-count domain is the
narrowest integer holding it: a categorical column's from its owner catalog's
cardinality, a counting column's from the bound its owner already places on
what it counts, a magnitude's from the range its quantization declares. A
column that uses the negative-one absence encoding is signed and its width
accounts for that value; a column that is never absent is unsigned. Requiring a
declaration as well would put a second source beside a fact the owner already
fixes, and the two would eventually disagree about a column neither had
changed.

`scalar_features` is the one named exemption and stays `int64`: it is one fixed
short array per observation, so narrowing it would buy nothing measurable
against a per-column rule where a single type reads more clearly.

Because widths differ per column, `nodes` and `edges` are column-major: one
contiguous array per column, in catalog order. They are not one
two-dimensional matrix. This costs nothing a consumer wanted — the model
embeds and projects each column separately, so it indexes columns individually
in either layout — and it is what lets a one-byte column occupy one byte per
row instead of being widened to its matrix's element type.

A column spans only the role blocks it applies to, and is stored over exactly
those blocks' contiguous ordinal ranges. Blocks are already dense and in a
fixed order, so a column that is inapplicable to a role has no entries for it
rather than a run of absence values. Addressability is unaffected, since a
block is a contiguous range of node ordinals. The absence encoding remains for
a column that is inapplicable to some member of a block it does span.

A consumer that needs a wider or unpacked form produces it after transfer, on
the device that consumes the value, and never in the environment or in a layer
between them. Widening at the boundary would restore exactly the bytes this
rule exists to avoid, in the hottest place and for the whole journey through
the connectors and the sample buffer; widening on device is a cast against
arithmetic that was going to touch the value anyway. This is the only rule
about the two forms: nothing else in this document distinguishes a value by
which one it is in.

Why the observation is worth this much attention, and why the mask is the
sharpest case, is
[Why The Mask Is Packed](rationales/ml.md#why-the-mask-is-packed).

#### The Packed Mask Layout

`action_mask` is the one boolean-domain array, so it is the one array the rule
above packs. Bit `i` is bit `i & 7` of byte `i >> 3`, least significant first,
making the array `ceil((enumeration_bound + 1) / 8)` bytes of `uint8`, and
every bit above `enumeration_bound` in the final byte is zero.

Zeroing the pad bits is what keeps two states with the same admissible set
byte-identical. Leaving them undefined would make the observation's bytes
depend on whatever the producer's buffer last held, and would do it only for
states whose live count is not a multiple of eight — breaking the
byte-identical-enumeration property
[Action Surface And Masking](#action-surface-and-masking) states,
intermittently.

The bit order is fixed here rather than left to a producer because both sides
index it. A mask is the one observation member where a layout disagreement is
silent: a wrong feature column trains a worse policy, while a wrong bit order
masks a different action than the one the environment refused, and the first
symptom is an importance ratio that is wrong for reasons no loss curve
explains.

`graph` and `decisions` are the only parts of the observation whose extent
varies, and neither is padded for the same reason: a bound large enough for the
largest reachable state wastes most of every batch on inert rows, and this is
most acute for `decisions`, whose row count is the live enumeration length and
whose capacity would have to be `enumeration_bound`. `objective_codes` and
`scalar_features` are fixed-length over closed catalogs, and the action mask is
fixed-length because a Gymnasium action space is fixed at construction and
`Discrete(enumeration_bound)` cannot vary per state. What distinguishes the
mask is that its extent is a declared capacity plus the stop entry rather than
a real count: decision entries at or beyond the live enumeration length are
clear, and the observation carries no entry for them under either encoding.

## Action Surface And Masking

The interaction surface is:

```text
action = {
  decision: Discrete(enumeration_bound)
  stop:     Discrete(2)
}
```

`enumeration_bound` is a static capacity so the space has a fixed declared
shape, while the live enumeration length varies with the state. A policy
therefore scores real decisions rather than a fixed factorization of decision
kind against node index. This replaces a flat product space over node slots and
decision types: such a space must declare a static per-type stride and decode
with the dynamic one, which either wastes the majority of the declared space or
lets the two strides disagree.

An environment's enumeration is a pure function of its current state and its
resolved configuration, so two runs with equal state and configuration produce
byte-identical enumerations in the same order.

Masking is the primary invalid-action mechanism. An index at or beyond the live
count, an index whose mask bit an environment cleared, and an out-of-range
index are each errors: the environment refuses the call, leaves the state
unchanged, and reports the violated precondition. A masked index is not
silently coerced to a no-op, to the nearest legal index, or to a terminal step,
because each of those makes the training signal indistinguishable from a real
decision.

`stop` set to one ends the episode electively at the current state and is
mutually exclusive with advancing; the `decision` field is ignored in that
call. An elective stop is the only outcome the policy itself controls.

`action_mask` covers one more outcome than `enumeration_bound` because it
covers the stop outcome as well as the decision slots, and it is indexed
exactly as the action distribution's outcomes are: bit `i` below
`enumeration_bound` masks the decision at that index, and bit
`enumeration_bound` masks stopping. One mask indexed one way is what lets a
model apply it additively to its logits without a second layout to keep in
step.

An environment may therefore clear stop for a state in which electing to stop
has no meaning. What it may not produce is a *non-terminal* state with no
admissible outcome at all: such a state has no defined action distribution, and
reaching one while the episode continues is a contract violation rather than an
ending. Packed, that test is whether every byte of `action_mask` is zero, which
the pad-bit rule makes exact.

The mask of an observation returned alongside an ending is never a distribution
input. No action is sampled from it, the value head does not read it, and no
consumer composes it or counts it as a fully-masked fallback. It may therefore
be all-clear, and an environment with no admissible outcome at a terminal state
returns one rather than manufacturing an outcome that does not exist.

A state whose enumeration exceeds `enumeration_bound` is refused rather than
truncated, because truncation permanently hides whichever part of the
enumeration sorts last. The refusal names both the capacity and the required
length, so the configuration can be corrected rather than guessed at.
`enumeration_bound` is the only capacity in this contract, because the
observation graph has none.

## Step Accounting And Episode Endings

Every step carries at most one transition, and carries none exactly when it is
an elective stop, so an episode record satisfies

```text
steps == advanced + non_advancing + elective_stops
```

by construction with no unaccounted remainder. `non_advancing` is the class of
steps that carried a transition which did not take effect. Each environment
names that class in its own vocabulary — one search calls it a rejection and
another a failure — and reports its count and its reasons under that name. What
is fixed here is the accounting, not the word.

A transition and an episode ending are separate facts because they co-occur: a
non-advancing step may both fail and end the episode, an advance may both
advance and
land in a state with no admissible decision, and an elective stop ends the
episode with no transition at all. A single three-way union could express none
of those without discarding half of what happened, so a step result carries
both, and its ending member is present exactly when the episode is over. An
episode records exactly one ending, on its final step, whichever transition
that step did or did not carry.

An ending an environment reaches by advancing into a state with no admissible
decision is reported on the step that produced that state, not on a later call,
so a consumer is never handed a state it has no legal index to act on.

An ending is either a termination or a truncation, and the two are reported
distinctly. A termination means the episode reached a state its contract
defines as final, and a value estimate must not bootstrap past it. A truncation
means the episode was cut off by a harness limit while the state remained
ordinary, and a value estimate must bootstrap. Reporting the two as one flag is
invalid.

An ending an environment defines as a truncation is itself priced at zero.
Charging a penalty for a limit the policy did not choose teaches it to avoid
states that merely take longer to leave, so among endings only the one the
policy elects carries a price.

That rule prices the *ending*, not the state the episode ended in. An
environment may charge for a property of its final candidate on whichever step
ends the episode, truncating or not, because such a charge is about something
the policy produced rather than about a limit it did not choose. The two are
told apart by what they read: an ending price reads the terminal reason and
nothing else, and a terminal state charge reads the state and never the reason.
An environment that charged more for the same final state under one ending than
under another would be pricing the limit through a state charge, which this
rule forbids in either spelling.

## Reward Contract

Reward is the signed difference of a selected search energy across a
transition. Energy is the value of the selected `WeightedLevel`, its checked
`uint128` arithmetic, and the sign-plus-magnitude form of its difference are
all owned by
[Objectives and Quality Gates](spec-dse-feedback.md#objectives-and-quality-gates);
an environment document states only which two states the difference is taken
between and what its non-transition outcomes cost.

With energy as a potential, a per-step signed energy difference is
potential-based shaping, so an episode's undiscounted return telescopes to the
total improvement between its first and last state while every step still
carries signal. An environment states only what breaks that property for it.

The energy is a directed code, so a decrease is an improvement. The reported
sign is positive when the child energy is below the parent energy and negative
when it is above. Every conversion, subtraction, product, and sum is checked;
an overflow is a resolved-policy failure, never a clamp, a saturation, or a
candidate penalty. A parent's energy is retained rather than recomputed, so a
step evaluates one energy, not two.

The native environment emits the exact `(sign, magnitude)` pair and nothing
else. It performs no scaling, normalization, clipping, discounting, or shaping.
The `float64` the RLlib environment definition requires is produced in
`loomml.env`, which applies one `ExactRatio` scale, and every other reward
transform belongs to the training document that configures that environment.
Keeping the conversion there is what lets the core stay integer-only and keeps
a mantissa width out of a canonical, digest-covered configuration; the adapter
is also the only layer that can state the exactness condition its own float
type imposes.

That scale is `reward_adapter.scale`, a training-view field. `loomml.env`
binds only an environment view, so the layer that reads a training
configuration passes the resolved ratio down at construction and `loomml.env`
applies the ratio it was given. Putting the scale in the environment view
instead would make every scale change a new environment digest and discard that
environment's caches for a number the environment does not use.

Reward codes an environment charges for non-transition outcomes are stated as
`uint64` magnitudes in the energy domain and applied with the sign the outcome
fixes, so no configuration view carries a signed or floating-point number. Such
a code is a declared constant and is never derived from the state, so a step
that carried no evaluated candidate can never be scored as though it had one.

An unavailable objective source has no numeric value at all. A step whose
selected closure cannot be evaluated fails rather than being scored from a
substitute, a default, or a neighbouring dimension; each environment names the
outcome it reports, and none of them produces a number.

## Determinism And Copy Coordinates

The seeded PRNG protocol, its `nextBounded` projection over canonically sorted
domains, and its prohibition on host entropy and library distributions are
owned by
[Search Policy And Determinism](spec-pnr.md#search-policy-and-determinism).
Every environment view carries the same two-field record, because both fields
are inputs to that protocol rather than facts about a search:

```text
EnvironmentDeterminismPolicy {
  master_seed: u64
  prng_protocol: Sha256SeededXoshiro256StarStar_1_0
}
```

Each environment adds only its own domain separator and its own stream
purposes, in this preimage shape:

```text
ASCII(environment domain separator)
  || u64be(effective_seed)
  || u32be(env_runner_index)
  || u32be(vector_index)
  || u64be(local_episode_index)
  || u32be(stream_purpose_ordinal)
```

`effective_seed` is the episode's seed input, which is the configuration's
declared master seed unless the consumer supplied an override at `reset`.
`env_runner_index` and `vector_index` identify one environment copy within a
parallel sampling topology and are supplied by the consumer; a single
non-parallel consumer supplies zero for both. `local_episode_index` counts
episodes on that copy from zero.

Including the copy coordinates in the preimage is what guarantees two copies
never draw the same sequence, and it is why a copy needs no coordination with
any other copy to stay independent. A copy coordinate is not host entropy: it
is a declared input the consumer supplies and replay reproduces.

A supplied `reset` seed becomes that episode's effective seed, and the local
episode counter restarts. It is an input the caller supplies rather than an
override of a digest-covered configuration field, so the configured master seed
remains the default the view declares and the trajectory records which seed
actually ran. Silently ignoring the argument is invalid: it makes an
environment that appears seeded but is not, which is a failure mode that
survives every test that only checks a run against itself.

Every environment accepts an exact episode start through the Gymnasium
`options` argument instead of drawing one, and names the payload that carries
it. This is an obligation rather than a permission because a peer contract
consumes it: [Test Protocol](spec-ml-core-training.md#test-protocol) runs every
test case by passing its instance as that override, so an environment without
one cannot be evaluated at all.

Four rules bind every override, whatever payload it carries. Its members must
already appear in the inventory the resolved configuration declares, so an
override selects from that inventory and cannot introduce anything the
configuration does not. When present it replaces the drawing steps entirely, so
no selection stream is consulted and the copy coordinates do not reach the
choice. The same override on any copy therefore produces the same episode,
which is what lets a fixed evaluation set run identically across runners and is
the only way an episode start is not a draw.

And a start failure under an override is returned as it stands. Retrying
requires a draw to retry and an override supplies none, so no redraw is
attempted and no retry budget an environment may otherwise declare is
consulted. Retrying would be wrong even where it were possible: a case that
silently ran a different instance than the one it names is not that case.

Policy sampling is not an environment stream: the learned policy owns its own
randomness and supplies an action index, so an environment replay is
reproducible given the same configuration, coordinates, seed, and action
sequence.

## Python Boundary

### Layering

A learned policy interacts through a Python package `loomml` with three layers,
each depending only on the one below:

```text
loomml.rllib   RLlib environment definitions, registration, and env context
loomml.env     the Gymnasium Env surfaces
loomml._core   native extension over the episodes, enumerations, and arrays
```

`loomml._core` exposes each episode lifecycle, its enumeration, its observation
arrays, and its closed outcome unions. `loomml.env` adds the Gymnasium `Env`
surface and holds no state the core does not own. `loomml.rllib` adds the
RLlib-specific obligations below. One environment occupies one module in each
layer, named for itself, so `loomml.env` and `loomml.rllib` are packages rather
than modules and adding an environment adds modules rather than editing shared
ones.

The layering is required, not stylistic. `loomml._core` must not import Ray,
and `loomml.env` must not import Ray, so the native boundary and the episode
semantics stay independent of a training framework. A second trainer, an
offline replay tool, or a plain scripted search therefore uses an environment
without acquiring RLlib, and RLlib version movement cannot reach the C++ side.

### RLlib Environment Definition

An environment is consumed as an RLlib environment, and the single-agent
environment definition of the Loom RLlib fork is the normative conformance
target. That definition is `gymnasium.Env` with graph observation support, so
conformance is stated against the fork and satisfied through Gymnasium rather
than through a parallel adapter. `loomml.rllib` supplies, per environment:

- a creator registered through `register_env` that accepts the env-context
  configuration and returns one `gymnasium.Env` instance;
- static `observation_space` and `action_space` attributes fixed for the
  instance's lifetime; and
- `reset(seed, options)` and `step(action)` returning the five-tuple with
  `terminated` and `truncated` reported distinctly.

Three obligations follow from that target.

The observation uses the Gymnasium `Graph` space, so a batch of observations is
a batch of `GraphInstance` values with differing node and arc counts rather
than a stackable rectangular array. Upstream RLlib's connectors flatten
fixed-shape spaces and cannot carry that batch; the fork's graph-space support
is what makes this space consumable, and it is the specific capability the fork
exists to add. This document therefore depends on a forked trainer rather than
on an unmodified upstream release, which
[External Dependency Pinning](spec-loom-stack.md#external-dependency-pinning)
records with the pinning and patch-stack terms that choice implies.

Padding to a fixed extent was rejected for the reason
[The Graph Instance](#the-graph-instance) states, and the fork's graph-space
support is what makes the unpadded space consumable. A ragged batch is the
honest shape of the data.

Masked actions use the parametric-action convention. The mask and the decision
nodes are members of the observation, not a side channel, an environment
method, or a wrapper attribute, so the connector pipeline carries them to the
model unchanged. The decision nodes and their target arcs are the
available-action embeddings that convention expects.

The mask enters that space as a fixed-shape `Box` of `uint8` and not as
`MultiBinary`, because `MultiBinary` is one entry per outcome and packing it is
the point. Its shape is a capacity and therefore constant, so it stacks
rectangularly and needs nothing the fork's graph support adds; the mask is the
one observation member the unpadded argument does not apply to, since its
extent never varied with the state to begin with.

The copy coordinates in the determinism preimage are RLlib's `worker_index` and
`vector_index`, which it passes in the env context across `num_env_runners`
actors and `num_envs_per_env_runner` vector slots; the adapter forwards both. A
run reproduces exactly when the seed and both counts are unchanged. Changing
either count changes which seeds are drawn, which is a property of the sampling
topology rather than a defect, and it is reported rather than hidden.

### Interaction Contract

The array contract is zero-copy. Each observation array is exposed as a
buffer-protocol view over environment-owned memory with the element type and
layout declared above, valid until the next call that advances or resets that
environment, and never written by the consumer. A consumer that needs an array
past that point copies it. The environment does not defensively copy on every
step, because observation marshalling is otherwise the dominant per-step Python
cost. RLlib's own connectors copy what they retain, so the buffer lifetime is
compatible with sampling without a defensive copy per step.

A non-advancing step neither advances nor resets, so it does not invalidate the
buffers: the state being observed is the one already observed, and only the
step and failure scalars change in place. Rebuilding the observation there is
permitted but pointless, and an environment names only which of its scalars
move.

No layer between the environment and the model unpacks the mask or widens a
narrow column, on the terms [Storage Widths](#storage-widths) sets. Both stay
in the environment's own form until the model consumes them, which
[Masking](spec-ml-core-model.md#masking) and
[Observation Batching](spec-ml-core-model.md#observation-batching) place on the
device.

Closed native outcome unions map to Python by kind rather than by message. An
ordinary non-advancing or terminal outcome is data returned from `step`, not an
exception, because both are expected transitions. A violated precondition, a
masked or out-of-range action index, a malformed configuration view, and an
owner internal failure raise distinct typed exceptions carrying the exact
reason discriminant. Diagnostic text is presentation and is never parsed.

One environment instance is owned by one thread. Several vector copies may live
in one process; they share no state, no arrays, and no PRNG stream. An instance
is not inherited across `fork`; an env-runner process constructs its own.

Every dependency named here is a search-harness dependency, and its exact
revision — including the Ray fork whose environment definition this section
targets, and the Python runtimes this document states conformance against — is
pinned by
[External Dependency Pinning](spec-loom-stack.md#external-dependency-pinning),
which also owns the containment argument that makes a forked trainer tolerable.
This document does not pin them separately.

## Benchmarking Harness Contract

Step and reset latency are first-class engineering concerns because their cost
compounds across a search, so every environment ships a harness that measures
them directly, on the removable-projection terms
[Nonsemantic Boundary](#nonsemantic-boundary) sets for every harness here.

Each environment document names its own harness binary and its own closed list
of step and reset stages, because the stages of a search over drafts and the
stages of a search over one frozen problem have nothing in common. What every
harness owes is the same.

Reported measures are per-stage wall time at p50, p90, and p99; steps per
second per environment copy; throughput against both `num_env_runners` and
`num_envs_per_env_runner`, since those scale differently and an aggregate
worker count hides which one is saturating; cache hit rate for every cache the
environment declares; the non-advancing rate by the environment's own reason
catalog;
and allocation counts per stage.

Two breakdowns are required of every harness because an aggregate over either
is misleading. Every measure is reported separately for advancing and for
non-advancing steps: a step that fails early contributes no sample to the
stages after its failure point, so a blended percentile moves whenever the
policy's failure mix moves, and a real regression can hide behind a rising
early-failure rate. And every measure is reported per the environment's own
action or decision partition, because the members of one enumeration do not
cost the same. An environment document may require further breakdowns; it may
not drop these.

Stage instrumentation is inactive outside the harness. A stage boundary implies
enough clock reads and allocator interception to be a measurable fraction of
the shortest stages, so an environment stepped by a trainer carries none of it,
and harness-reported per-stage times are not comparable to uninstrumented steps
per second.

The nonsemantic status of wall time, the prohibition on summing concurrent
times, and deterministic work summaries as the cross-machine cost measure are
owned by
[Operational Observations](spec-dse-feedback.md#operational-observations). A
harness reports the deterministic work-unit counts alongside its timings and
states which is which. A regression budget is a ratio against a recorded
baseline tuple of exact identities and configuration digests, never an absolute
duration.

## Anchor Verification

Stable tests cover every reference of every live enumeration member resolving
to exactly one node ordinal in the combined space, including a relation-valued
target, under either enumeration encoding; a multi-reference decision exposing
every member of its set or tuple in canonical payload order rather than only
its first; a parallel connection appearing as one node whose count is a feature
and a count-adjusting decision leaving the node inventory unchanged; under
`DecisionNodes`, a decision's target arcs carrying its complete reference set
with the correct `ArcRole` per arc, its out-degree equalling that set's size,
its ordinal within the `Decision` block equalling its `ActionIndex`, and
`decisions` being empty; under `DecisionColumns`, the `Decision` block being
absent, `decisions` carrying exactly one row per live entry and no arcs, a
row's ordinal equalling its `ActionIndex`, and every target-naming column
holding a node ordinal that resolves in the same instance's `graph`; an
environment whose entries have variable target arity being refused the
`DecisionColumns` encoding; two states with different node counts producing
`GraphInstance` values of different extents, with no inert node, no inert arc,
and no extent that is not a real count; every environment's `NodeFeatureColumn`
catalog beginning with `Role` and its `ArcRole` catalog beginning with
`Structural`; a column stored over only the role blocks it applies to, and the
negative-one absence value appearing only within a block the column does span;
every observation array being stored in the narrowest exact form of its value
domain with no catalog declaring a width, and `scalar_features` being the one
exemption; a column-major `nodes` and `edges` rather than a two-dimensional
matrix; the action mask being the only
fixed-length array whose extent is a capacity, covering `enumeration_bound + 1`
outcomes with stopping at the last, packed one bit per outcome at bit `i & 7`
of byte `i >> 3` with every bit above `enumeration_bound` in the final byte
zero, its decision bits at or beyond the live count being clear, and its bit
indices agreeing with the action distribution's outcome indices; two states
with the same admissible set producing byte-identical masks at every live count
including one that is not a multiple of eight; the mask crossing the Python
boundary packed and being unpacked by no layer below the model; an environment
clearing the stop bit for a state in which stopping has no meaning, and a state
with every byte zero being a contract violation rather than an ending, and the
mask of an observation returned alongside an ending being neither composed,
counted as a fallback, nor read by the value head; a state
whose enumeration exceeds
`enumeration_bound` being refused with both the capacity and the required
length reported rather than being shortened; a masked, stale, or out-of-range
index being refused with the state unchanged and no coercion to a legal index
or to a terminal step; every environment defining an episode start override, an
override selecting only from declared inventory, consulting no selection
stream, producing the same episode on any copy, and a start failure under one
being returned as it stands with no redraw and no retry budget consulted; a
non-advancing step invalidating no buffer; an elective stop ignoring the
decision field; the
canonical enumeration order being identical across runs for equal state and
configuration; the step accounting identity holding over a complete episode;
termination and truncation being reported distinctly, a truncation's ending
price being zero, and a terminal state charge applying identically under a
truncation and a termination that leave the same state; a buffer remaining
valid across a call that neither advances nor resets and being invalidated by
one that does; an ordinary non-advancing or terminal outcome being returned as
data
while a precondition violation, a masked index, a malformed view, and an
internal failure each raise a distinct typed exception carrying its reason
discriminant; `loomml._core` and `loomml.env` importing without Ray present;
two environment copies with distinct coordinates drawing disjoint sequences
from one seed; a `reset` seed argument becoming the effective seed and being
recorded as such rather than ignored; and an instrumented harness run producing
the same formal results as an uninstrumented one.

Tests do not pin the live node, arc, or decision counts of any particular
state, the bound values a profile selects, wall-time numbers, per-decision cost
ratios, allocation counts, diagnostic text, or Python formatting.
