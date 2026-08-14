# ML PnR Model Architecture

This document defines the learned policy that consumes the ML PnR Environment's
observations and emits its actions. It owns the embedding of that environment's
column catalogs, the two-factor policy head that scores an action from the
nodes it names, the model's own configuration view, and the harness that
measures its forward pass.

The model is a search policy on the terms
[ML Model Core](spec-ml-core-model.md) states. It proposes Actions; the
environment applies them through Place and Route's own `MoveTransaction`, and
every legality, closure, and objective fact remains owned where
[ML PnR Environment](spec-ml-pnr-environment.md) places it.

Everything a Loom policy does that is not specific to what it is searching —
the batching, the embedding discipline, the graph-transformer trunk, the
context pooling, the value head, masking, the action distribution, and the
checkpoint boundary — is owned by [ML Model Core](spec-ml-core-model.md). This
document is the difference between that and a placer.

## Ownership

Every fact this document depends on resolves to one exact owner:

- [ML Model Core](spec-ml-core-model.md) owns the module boundary, observation
  batching, the feature-embedding discipline, the encoder, the graph context,
  the value head, masking, the action distribution, the checkpoint boundary,
  and the obligations every forward-pass harness satisfies;
- [Column Catalogs](spec-ml-pnr-environment.md#column-catalogs) owns
  `PnrNodeFeatureColumn`, `PnrArcRole`, `PnrScalarFeatureColumn`,
  `PnrDecisionColumn`, and `EpisodePhase`;
- [Action Space](spec-ml-pnr-environment.md#action-space) owns the enumeration,
  its canonical order, `ActionIndex`, and which entries the phase mask clears;
- [The PnR Node And Link Space](spec-ml-pnr-environment.md#the-pnr-node-and-link-space)
  owns the role blocks, the placement and route arcs, and the rule that an
  action names its anchor and choice by column;
- [Reward](spec-ml-pnr-environment.md#reward) owns the reward the value head
  regresses toward;
- [Enumeration Encoding](spec-ml-core-environment.md#enumeration-encoding) owns
  the `DecisionColumns` form this model consumes;
- [Actions And MoveTransaction](spec-pnr.md#actions-and-movetransaction) owns
  the Action kinds an entry's `ActionKind` column names; and
- [Resolved Configuration](spec-config-ssot.md#component-views) owns
  component-view framing.

This document owns only the PnR feature mapping, the policy head, the model
configuration view, and the forward benchmarking stages.

## Module Boundary

The model is `LoomPnrRLModule`, one `TorchRLModule` implementing
`ValueFunctionAPI` on the terms
[Module Boundary](spec-ml-core-model.md#module-boundary) states. It adds no
entry point and changes no signature.

## Feature Embedding

The core fixes how a column class is consumed; this section fixes which class
each PnR column belongs to and why, for the cases where the answer is not
mechanical.

Embedded categorical columns:

```text
column          table size                            source catalog
Role            GraphNodeRole cardinality             environment core
EntityKind      Fabric local-reference kinds plus
                Dataflow and TechMapping kinds        Fabric Identity, Dataflow
ActionKind      SpatialMappingAction kind cardinality Place And Route
Phase           EpisodePhase cardinality              PnR environment
```

`Phase` is a graph-level categorical rather than a node column, and it is
embedded rather than projected for the ordinary reason: `Construction` and
`Repair` are two regimes, not two points on a scale, and the model's behaviour
in one should not be constrained to lie on a line through the other.

Every `PnrNodeFeatureColumn` not named below is a projected numeric column with
the absence indicator the core requires. `CapacityMagnitude`, `CapacityUsage`,
and `CapacityOveruse` are projected with the occupancy ratio the core requires
alongside them, and the same treatment extends to each capacity and usage pair
the environment appends when `include_resource_states` is set.

Two columns need a stated decision rather than a classification.

`AnchorNode` and `ChoiceNode` are not features at all. They are indices, and
the head consumes them by gathering `h` at those ordinals; projecting an
ordinal would assert a distance between two unrelated nodes that happen to sort
near each other. Whether an entry's anchor is already placed is likewise not a
column: `Placed` on the anchor node is that bit, and the head already has the
anchor's embedding in hand.

`ChoiceDistance` is the load-bearing feature of this model and gets its own
treatment. Hop counts near zero decide most placements and hop counts far from
zero are nearly interchangeable, so a single linear projection would spend its
resolution in the wrong place. Small distances are embedded from a table
indexed by exact hop count up to a configured bucket count, and distances
beyond it are projected through the core's signed logarithmic transform into a
shared remainder. The near range therefore keeps exact resolution while the far
range still generalizes across magnitudes, and the absent sentinel — every
action kind but a realization binding — takes the core's zero-plus-indicator
form rather than a distance of zero, which would otherwise read as "adjacent".

Scalar features are graph-level, with the ratios the core requires supplied
alongside the magnitudes: `StepOrdinal` against `StepBound`, `PlacedCount`
against `RealizationCount`, and `RepairStepCount` against `RepairStepBound`.
`UnmaskedActionCount` and `ConsecutiveFailures` are supplied as magnitudes.
`PlacedCount` over `RealizationCount` is how far construction has progressed,
which is the quantity that most changes what a good action looks like, so
requiring the model to divide two magnitudes to recover it would be a
gratuitous obstacle.

One absence is a decision rather than an oversight. The model receives
`CapabilityCount` for an occurrence but not the identity of the operations that
occurrence supports. The prior architecture embedded a per-node instruction
bitmask because its candidate mask was computed separately and the network had
to learn which unit could host which operation. Here the enumeration is derived
from `CandidateDomain`, so an occurrence that cannot host a realization is
never offered as a choice and compatibility is never a question the policy has
to answer. What remains useful is scarcity — whether an occurrence is a
specialist worth saving for something else — and a count carries that. A
capability-set feature would be paying for a distinction the mask already made.

## Encoder

The encoder is the core's `GraphTransformerStack`, unchanged, with
`PnrArcRole` supplying edge features. Its three roles are the three
relations a placement policy has to tell apart: `Structural` is the fabric's
own connectivity and the software graph's dependencies, `Placement` is the
current mapping, and `Route` is where the nets currently run.

`Route` arcs are what make the trunk expensive: the optional projection
[Observation](spec-ml-pnr-environment.md#observation) governs can dominate arc
count, and therefore attention cost, when it is enabled.

## Graph Context And Value Head

Both are the core's, unchanged. The value head reads the pooled context and
emits one bounded scalar per instance, and its bound is checked against the
range [Online Training](spec-ml-pnr-training.md#online-training) declares
rather than chosen.

## Policy Head

An action names two nodes and carries its own kind and distance, so the head
scores it from those directly. Nothing is reconstructed from an index and no
action has an embedding of its own.

```text
anchor(a) = ( AnchorNode of a, ActionKind of a )

s_anchor(a) = MlpAnchor([ h[AnchorNode(a)], e_kind(a), context ])
s_choice(a) = MlpChoice([ h[AnchorNode(a)], h[ChoiceNode(a)],
                          e_distance(a), e_kind(a), context ])
logit(a)    = s_anchor(a) + s_choice(a)
```

`h` is the shared trunk's node embedding, indexed by the ordinal the entry's
column holds. This is the whole of "act on the predicted node": the score of
placing a realization on an occurrence is a function of that realization's
embedding and that occurrence's embedding, both produced by one pass over the
state.

`s_choice` is this model's selection term, so actions sharing an anchor form
one group and compose on the terms
[Policy Head Factorization](spec-ml-core-model.md#policy-head-factorization)
states.

The hierarchy is what this environment is shaped for. During construction every
entry of one unplaced realization is one group, so the anchor factor is "which
realization to place next" and the choice factor is "where" — exactly the
decomposition the search actually makes. Under `CanonicalNext` anchor selection
the mask leaves one group live and the anchor factor degenerates to a constant,
which is the correct behaviour rather than a special case: the environment
already made that choice, so the policy has nothing to contribute to it.

### Conditioning The Choice On The Anchor

`s_choice` reads the anchor's embedding as well as the choice's, and that is a
deliberate departure from the architecture this model translates. The prior
`SwapSchedulerModel` scored hardware slots from the slot embedding and the
graph context alone, conditioning on the selected software node only through
the mask. It could therefore learn which slots are good in general — central,
uncongested, well connected — but not which slot suits *this* node, which is
most of the placement problem: a producer wants to be near its consumers, and
which occurrence satisfies that is a property of the pair.

The cost is that `s_choice` is evaluated once per live action rather than once
per node, and this document is precise about which part of that cost is
recoverable.

The prior architecture's `_SplitScorer` factoring is admitted and extended. A
scorer of the form `out(activation(W_1 x_1 + ... + W_k x_k))` is exactly the
scorer `out(activation(W · concat(x_1, ..., x_k)))` with `W` partitioned, so
`MlpAnchor` and `MlpChoice` may be evaluated in the factored form. What that
buys is memory: the concatenation is never materialized and never saved for the
backward pass, which matters at live-action counts in the hundreds of
thousands. What it does not buy is arithmetic, because the activation and the
output projection are still evaluated per action. A specification that claimed
otherwise would be promising a saving the algebra does not contain.

A genuine per-action reduction requires a scorer whose pair term is bilinear —
an inner product between one projection of the anchor and one of the choice, so
each side is projected once per node and the pair costs a dot product. That
computes a different function, with strictly less expressive pair interaction
than an activation over the sum. It is permitted as a configured variant and is
validated against the reference form on the same inputs and parameters like any
other optimization; it is not presented as an equivalent implementation of the
same head.

### Stop

The environment masks the core's stop logit for the whole construction phase
and for the whole of a `ConstructThenAnneal` episode, and the head applies that
mask like any other: it never derives its own stop admissibility from the phase
feature, because the mask is the authority and two sources for one fact
eventually disagree.

The joint distribution over `enumeration_bound + 1` outcomes and its stop
encoding are the core's.

## Masking

Masking is the core's contract. Two properties of this environment are worth
stating because they make it easier here than in the design-space environment.

The environment's mask is a phase rule rather than a rejection record. Every
entry it clears is a legal Action of the current candidate, cleared because the
phase does not admit it, and the same entry becomes live again when the phase
changes. Nothing in a PnR episode is masked because it was tried and failed, so
no mask bit depends on episode history.

Anchor-group consistency is automatic. Construction clears every entry of an
already-placed realization, which is precisely a whole anchor group, and clears
whole kinds rather than individual choices. The core's rule that a group is
masked exactly when all its members are masked therefore holds without the head
arranging for it, and a partially masked group only arises from an advisory
mask.

## Model Configuration

The model's shape is one immutable component view with schema descriptor bytes
`loom.ml_pnr_model.config.1.0`, following the framing owned by
[Component Views](spec-config-ssot.md#component-views) and the three properties
[Model Configuration Framing](spec-ml-core-model.md#model-configuration-framing)
fixes:

```text
ResolvedMlPnrModelConfigView {
  embedding_widths: total table<embedded column, positive uint32>
  distance_bucket_count: positive uint32
  encoder: GraphTransformerStack
  head_widths: PnrHeadWidths
  choice_scorer: Factored | Bilinear
  value_bound: positive uint64
  advisory_mask_set: canonical set<AdvisoryMaskRef>
  initialization: InitializationConstants
}

PnrHeadWidths {
  anchor_hidden: positive uint32
  choice_hidden: positive uint32
  value_hidden: positive uint32
}
```

`distance_bucket_count` is the exact-hop table height Feature Embedding
describes; it sizes a parameter tensor, which is why it belongs in this view
and not in a training one. `choice_scorer` selects between the two forms
Conditioning The Choice On The Anchor defines, and it is a view field rather
than an implementation switch because `Bilinear` changes the function and
therefore the parameter shapes, so a checkpoint is correctly incompatible
across it.

## Forward Benchmarking

This model ships `loom-pnr-model-bench`, on the obligations
[Forward Benchmarking Obligations](spec-ml-core-model.md#forward-benchmarking-obligations)
states.

The harness decomposes one forward pass into these stages:

```text
gather      assemble the batched graph and decision columns, move to device
embed       embedding lookups, numeric projection, input projection
encode      the graph-transformer stack
pool        per-instance multi-scale context
anchor      anchor scoring, once per anchor group
choice      choice scoring, once per live action
mask        mask composition and additive application
distribute  distribution construction and sampling
value       the value head, when the entry point computes it
```

`anchor` and `choice` are separate stages, where a model whose value is a
catalog ordinal can report one policy stage. They scale with different
quantities — anchor-group count and live-action count — and the whole argument
of Conditioning The Choice On The Anchor is that the second is the price of the
first being worth having. A blended policy stage would make it impossible to
say whether that price is what it was expected to be, or whether the factored
form is doing anything.

`gather` is reported with the decision columns separated from the graph, since
their row count is the live-action count and the graph's is the state size, and
those move independently: a repair-phase state has a stable graph and a much
larger enumeration than a late-construction one.

Beyond the core's required measures, three breakdowns are required here.

Every measure is reported per phase, on the terms
[Benchmarking](spec-ml-pnr-environment.md#benchmarking) states for the
environment harness. What differs here is that a construction and a repair
forward pass differ in live-action count as well as in route density, so the
scaling below separates on both.

Scaling is reported against node count, arc count, live-action count,
anchor-group count, and the mean anchor-group size. The last is what makes the
anchor-group saving
[Policy Head Factorization](spec-ml-core-model.md#policy-head-factorization)
describes legible.

The `choice_scorer` variants are measured against each other and against the
reference form. `Bilinear` is admitted for speed and computes a different
function, so its measurement reports both its stage cost and its divergence
from `Factored` on the same inputs, rather than reporting a latency alone and
leaving the quality question to a separate exercise.

The rollout split against the environment step is reported first, as the core
requires, and it matters more here than for a design-space policy: a PnR
environment step contains a Mapping probe with incident route closure, which is
plausibly an order of magnitude more expensive than this forward pass. A model
optimization that halves an already-small share buys nothing, and the split is
what says so before the work is done rather than after.

## Conformance Anchors

Stable tests cover a `Placement` arc reaching the encoder as an edge feature
and no node column restating that relation; `ChoiceDistance` at a hop count
inside the bucket range reaching an exact table row and one beyond it reaching
the projected remainder, and the absent sentinel producing the core's
zero-plus-indicator rather than the encoding of distance zero; two actions
sharing an anchor node and kind receiving an identical `s_anchor`; two choices
on one anchor receiving different scores; an action whose anchor and choice
embeddings are swapped receiving a different score, so the pair term is not
symmetric; the factored scorer reproducing the concatenated reference exactly;
the `Bilinear` variant being reported with its divergence from `Factored`
rather than as an equivalent; scoring restricted to live actions producing
logits identical to scoring every slot; a `CanonicalNext` configuration leaving
exactly one live anchor group and its anchor factor contributing no gradient to
the choice among realizations; a masked anchor group receiving no probability
and contributing no entropy; construction clearing whole anchor groups so that
no partially masked group arises without an advisory mask; the stop logit being
masked throughout construction and throughout a `ConstructThenAnneal` episode,
and the head deriving stop admissibility from the mask rather than from the
phase feature; the two-level and flat formulations producing identical
distributions to numerical tolerance; `_forward_train` and
`_forward_exploration` producing identical logits for equal parameters,
observations, and masks; `_forward_inference` computing no value;
`compute_values` producing identical predictions with and without precomputed
embeddings; a decision-column node ordinal being offset into the batched node
union before use, so an action never scores another instance's nodes; and a
checkpoint load failing across a changed `distance_bucket_count` or a changed
`choice_scorer`.

Tests do not pin layer counts, widths, head counts, the bucket count, the
scorer variant, initialization constants, the advisory mask set, learned
parameter values, wall-time numbers, per-phase cost ratios, device placement,
precision configurations, or tolerance values.
