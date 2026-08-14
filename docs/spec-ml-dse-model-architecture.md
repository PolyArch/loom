# ML DSE Model Architecture

This document defines the learned policy that consumes the ML DSE Environment's
observations and emits its actions. It owns the embedding of the environment's
graph observation, the graph-transformer encoder over that embedding, the
hierarchical policy head that scores enumerated decisions, the value head, the
masking contract, and the parameter and checkpoint boundary.

The model is a search policy on the terms
[ML Model Core](spec-ml-core-model.md) states, and every legality, mappability,
and objective fact remains owned where
[ML DSE Environment](spec-ml-dse-environment.md) places it.

## Ownership

Every fact this document depends on resolves to one exact owner:

- [ML Model Core](spec-ml-core-model.md) owns the module boundary, observation
  batching, the feature-embedding discipline, the encoder, the graph context,
  the value head, masking, the action distribution, the checkpoint boundary,
  and the obligations every forward-pass harness satisfies;
- [The Graph Instance](spec-ml-core-environment.md#the-graph-instance) owns the
  observation container, and
  [The Graph Instance](spec-ml-dse-environment.md#the-graph-instance) owns this
  environment's node and arc feature catalogs, action mask, objective codes,
  and scalar features;
- [Action Space](spec-ml-dse-environment.md#action-space) owns the enumerated
  decisions, their canonical order, their targets, and `ActionIndex`;
- [Combined Node And Link Space](spec-ml-core-environment.md#combined-node-and-link-space)
  owns the node roles, the arc-role obligation, and the target-closure rule
  this document's policy head relies on;
- [Reward](spec-ml-dse-environment.md#reward) owns the reward the value head
  regresses toward;
- [RLlib Environment Definition](spec-ml-core-environment.md#rllib-environment-definition)
  owns the trainer-facing environment contract and the forked trainer this
  model plugs into;
- [Evaluation and DSE](spec-dse-feedback.md#model-parameters-and-training) owns
  `ModelParameterContractRef` and the registered prediction contracts, which
  this model is not; and
- [ML Training Core](spec-ml-core-training.md) owns the algorithm binding and
  the loss it brings, the rollout and learner topology, the stage sequence, and
  every hyperparameter schedule, and
  [ML DSE Training](spec-ml-dse-training.md) owns this environment's reachable
  return range.

This document owns only the DSE feature mapping, the hierarchical policy head,
the model configuration view, and its own benchmarking stages. It restates no
environment fact and defines no new observation, action, or reward.

## Module Boundary And Batching

The module boundary, the three forward entry points and their shared encoder
pass, `compute_values` accepting precomputed embeddings, and the disjoint-union
batching of ragged graph observations are owned by
[Module Boundary](spec-ml-core-model.md#module-boundary) and
[Observation Batching](spec-ml-core-model.md#observation-batching). This model
is `LoomDseRLModule` and adds no entry point and changes no signature.

This environment uses the `DecisionNodes` enumeration encoding, so its
enumerated decisions are nodes of the observation graph and it supplies no
separate decision instance. `ActionIndex` is a per-instance decision-block
ordinal and is mapped through `ptr` and that block's offset.

## Feature Embedding

The embedded categorical columns are:

```text
column                    table size                       source catalog
Role                      GraphNodeRole cardinality        environment
EntityKind                Fabric local-reference kinds
                          plus Dataflow node/value kinds   Fabric Identity, Dataflow
DecisionDomain            ExplorationDomainKind cardinality environment
DecisionKind              sum over domains of that
                          domain's decision-union size     candidate generators
DecisionValueForm         value-form cardinality           environment
DecisionValueOrdinal      per-domain prototype cardinality candidate generators
DecisionDeltaSign         sign cardinality                 environment
ArcRole                   DseArcRole cardinality           environment
```

`DecisionKind` is embedded on the `(DecisionDomain, DecisionKind)` pair rather
than on the kind ordinal alone, because kind ordinals restart per domain and a
shared table would tie unrelated decisions together. `DecisionValueOrdinal` is
the direct analogue of the old architecture's per-instruction embedding table:
it is what lets the policy reason about a specific replacement prototype rather
than about an opaque index, and it is the axis along which the value factor of
the hierarchical head operates.

Numeric projection, the absent-column rule, the signed logarithmic transform,
the capacity-usage-occupancy triple, the mapping of `objective_codes` into
`[0, 1]`, and the concatenate-then-project input path are owned by
[Feature Embedding](spec-ml-core-model.md#feature-embedding). Two of this
environment's columns need the treatment named rather than derived.

`ResidualViolationCount` and `PlacedDegree` are projections of the probe result
rather than of the design, so they change when the same design is probed
differently; they are projected as ordinary magnitudes and carry no special
status, but a configuration comparing runs across a changed probe view is
comparing two different features under one name.

`StepOrdinal` and `StepBound` are supplied as their ratio as well as their
magnitudes, so remaining budget is directly available; `ConsecutiveRejections`
is supplied as a magnitude, as is the `Decision` block's size.

## Encoder And Context

The `GraphTransformerStack` record, pre-normalization, the feed-forward
sublayer, edge-aware attention, jumping knowledge, the one-shared-trunk rule,
and the multi-scale per-instance graph context are owned by
[Encoder](spec-ml-core-model.md#encoder) and
[Graph Context](spec-ml-core-model.md#graph-context).

This environment's `DseArcRole` catalog supplies the edge features. `Placement`
and `Route` are the probe's realization and traversal relations, which differ
in arity and in which decisions move them, and `DecisionTargetPrimary` against
`DecisionTargetMember` is what lets a decision node's attention distinguish the
entity it rewrites from the set it rewrites it against.

`DecisionDeltaSign` is embedded rather than projected because a sign is
categorical, which is what the value term's delta form requires: a magnitude
carries the size of a change and the sign carries its direction, and projecting
the sign as a number would make a negative delta a small positive one.

## Hierarchical Policy Head

The environment's action is one `ActionIndex` plus a stop flag, and its
decision nodes carry both their target arcs and their value columns. The head
therefore scores decisions where they live, on the graph, and factors the score
into a part that depends on where the decision acts and a part that depends on
what value it selects.

Every decision node `d` has an anchor and a value:

```text
anchor(d) = ( primary target node of d,
              pooled GroupMember target set of d,
              DecisionDomain of d,
              DecisionKind of d )

value(d)  = ValuePrototype(DecisionValueOrdinal)  if form is ReplacementPrototype
          | ValueDelta(DecisionDeltaMagnitude, DecisionDeltaSign)
                                                  if form is BoundedIntegerDelta
          | Absent                                if form is NoValue
```

The anchor includes the whole target set, not only the primary, so a decision
naming an entity plus a reference set is scored on the set it actually
rewrites. Decisions sharing an anchor form one anchor group; a group is a
singleton exactly when its decisions carry no value.

`value(d)` reads whichever columns its form populates. A `BoundedIntegerDelta`
decision has no prototype ordinal, and its ordinal column is the absent
sentinel, so a value term reading only `DecisionValueOrdinal` would give every
delta on one target an identical score and make the magnitudes
indistinguishable. A delta is embedded from its magnitude and sign instead: the
sign is categorical, and the magnitude is projected, so adjacent magnitudes
score similarly and the head generalizes across a range rather than memorizing
each step of it.

The head produces two scores per decision:

```text
s_anchor(d) = MlpAnchor([ h_primary(d), h_members(d), e_kind(d), context ])
s_value(d)  = 0                                        if value(d) is Absent
            = MlpValue([ h_primary(d), e_value(d), e_kind(d), context ])
                                                       otherwise
logit(d)    = s_anchor(d) + s_value(d)
```

`s_anchor` is constant within an anchor group by construction, because the
anchor includes every input `MlpAnchor` reads: the primary target, the pooled
member set, and the kind, evaluated against one shared context. This is what
makes the factorization a genuine hierarchy rather than a reparameterization:

A boolean decision, one carrying no value, is a singleton group whose
conditional factor is one, so it is scored directly on its node or link exactly
as asked. A value-carrying decision is scored on its node or link by the anchor
factor and among its alternatives by the value factor, which is the operation
selection of the prior architecture generalized to every value-carrying
decision kind rather than to instructions alone.

The factorization, the group-constancy rule, the flat-versus-two-level
equivalence, the broadcast saving, and the stop logit are owned by
[Policy Head Factorization](spec-ml-core-model.md#policy-head-factorization);
the joint distribution these logits feed and its stop encoding by
[Action Distribution](spec-ml-core-model.md#action-distribution).

Slots beyond the instance's `Decision` block size have no decision node and
receive a masked logit; the stop outcome is masked only when the environment's
mask clears it.

## Value Head, Masking, And Distribution

The value head and its bound, the masking discipline and its five rules, and
the joint action distribution over `enumeration_bound + 1` outcomes are owned
by [Value Head](spec-ml-core-model.md#value-head),
[Masking](spec-ml-core-model.md#masking), and
[Action Distribution](spec-ml-core-model.md#action-distribution). The
checkpoint boundary is owned by
[Parameters And Checkpoints](spec-ml-core-model.md#parameters-and-checkpoints).

One property of this environment is worth naming against the masking contract.
The environment's own mask already carries a per-state rejection record, so a
bit it clears may encode that a decision was tried at this state and failed.
That is a fact about the episode rather than about the decision, and it is the
reason the recorded mask must travel in the batch: recomputing it at update
time against a different episode position would mask a decision that was
available when it was sampled.

## Model Configuration

The model's shape is one immutable component view with schema descriptor bytes
`loom.ml_dse_model.config.1.0`, following the framing owned by
[Component Views](spec-config-ssot.md#component-views) and the three properties
[Model Configuration Framing](spec-ml-core-model.md#model-configuration-framing)
fixes:

```text
ResolvedMlDseModelConfigView {
  embedding_widths: total table<embedded column, positive uint32>
  encoder: GraphTransformerStack
  head_widths: DseHeadWidths
  value_bound: positive uint64
  advisory_mask_set: canonical set<AdvisoryMaskRef>
  initialization: InitializationConstants
}
```

```text
DseHeadWidths {
  anchor_hidden: positive uint32
  value_head_hidden: positive uint32
  value_hidden: positive uint32
}
```

`value_bound` is the symmetric range the value head clamps to, in the scaled
reward's units. `advisory_mask_set` names the optional masks the masking
contract admits.

## Inference Benchmarking

The harness obligations, the two regimes, the entry-point split, the scaling
rule, the rollout split, invariance against an unoptimized reference, and
ratio-based budgets are owned by
[Forward Benchmarking Obligations](spec-ml-core-model.md#forward-benchmarking-obligations).
This model ships `loom-dse-model-bench`.

The harness decomposes one forward pass into these stages:

```text
gather        assemble the batched graph and move it to the device
embed         embedding lookups, numeric projection, input projection
encode        the graph-transformer stack
pool          per-instance multi-scale context
policy        anchor and value scoring over decision nodes
mask          mask composition and additive application
distribute    distribution construction and sampling
value         the value head, when the regime computes it
```

`encode` is the expected dominant term, since attention cost scales with arc
count, and `gather` is the expected dominant term for a small graph on an
accelerator, where the transfer costs more than the arithmetic. Both
expectations are worth measuring rather than assuming.

Each stage is reported against node count, arc count, and live decision count,
with the fitted growth of `encode` against arc count and of `policy` against
live decision count.

Three optimizations specific to this head are measured against an unoptimized
baseline, and each carries a failure mode a naive measurement will miss.

Anchor-group evaluation, on the terms
[Policy Head Factorization](spec-ml-core-model.md#policy-head-factorization)
states. The mean group size is reported alongside the `policy` stage.

Live decisions only. The head emits logits over `enumeration_bound` slots, but
only the live ones have decision nodes. Scoring runs over live decisions and
scatters into the emitted logit vector; running the scoring
multilayer perceptrons over dead slots wastes work proportional to the gap
between capacity and live count, which is largest exactly when the capacity is
set conservatively.

Batched inference across copies. Vectorized environment copies on one runner
present several instances at once, and batching them into one union graph
amortizes the per-call overhead that dominates latency-bound inference. The
harness reports latency against the copy count to show where that amortization
saturates.

Device placement is measured across the size range and the crossover is
reported rather than assumed: for small graphs, host inference can beat
accelerator inference because the transfer in `gather` exceeds the arithmetic
it feeds.

## Conformance Anchors

Stable tests cover an `ActionIndex` resolving to its own instance's decision
block; a decision carrying a reference set contributing every member to its
anchor summary rather than only its primary; a value-free decision receiving a
zero value term and forming a singleton anchor group; two decisions sharing an
anchor receiving an identical anchor score; two `BoundedIntegerDelta` decisions
on one target with different magnitudes receiving different scores; the
two-level and flat formulations producing identical distributions to numerical
tolerance; a stop logit competing in the same normalization as the decision
logits; a per-resource capacity and usage pair reducing to one resource
embedding rather than widening the node vector with the resource catalog;
scoring restricted to live decisions producing logits identical to scoring
every slot; the mean anchor-group size being reported alongside the `policy`
stage; and every optimized configuration reproducing the unoptimized
reference's logits and values within the declared tolerance.

Tests do not pin layer counts, widths, head counts, initialization constants,
the advisory mask set, learned parameter values, wall-time numbers, device
crossover points, or throughput.
