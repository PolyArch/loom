# ML Model Core

This document defines the contract every Loom learned search policy shares: how
a batch of observations becomes one disjoint-union graph, how raw observation
columns become encoder inputs, the graph-transformer trunk over them, the graph
context both heads read, the value head, the masking discipline, the action
distribution, the checkpoint boundary, and what every forward-pass benchmark
owes.

It defines no policy head. Which actions exist, how they are anchored, and what
scores them are owned by the model documents that build on this one:

- [ML DSE Model Architecture](spec-ml-dse-model-architecture.md) scores
  candidate-generator decisions for
  [ML DSE Environment](spec-ml-dse-environment.md); and
- [ML PnR Model Architecture](spec-ml-pnr-model-architecture.md) scores typed
  Place and Route Actions for [ML PnR Environment](spec-ml-pnr-environment.md).

A model is a search policy, not a Loom semantic authority. It proposes actions;
the environment decides what those actions mean and whether they are
admissible, and every legality, mappability, and objective fact remains owned
where [ML Environment Core](spec-ml-core-environment.md) and the environment
documents place it. A model checkpoint is not an Artifact, is not
`EvaluationEvidence`, and never becomes a registered prediction contract.

## Ownership

Every fact this document depends on resolves to one exact owner:

- [ML Environment Core](spec-ml-core-environment.md) owns the observation
  container, the combined node-and-link space, the enumeration encodings, the
  action surface, the action mask, and the reward this model's value head
  regresses toward;
- each environment document owns its own node, arc, scalar, and decision column
  catalogs, and the enumeration whose entries a policy head scores;
- [Evaluation and DSE](spec-dse-feedback.md#objectives-and-quality-gates) owns
  `ExactAffineQuantization` and the objective algebra whose codes enter as
  features;
- [Evaluation and DSE](spec-dse-feedback.md#model-parameters-and-training) owns
  `ModelParameterContractRef` and the registered prediction contracts, which a
  policy is not;
- [Resolved Configuration](spec-config-ssot.md#component-views) owns
  component-view framing, canonical view bytes, and `component_view_digest`;
- [Operational Observations](spec-dse-feedback.md#operational-observations)
  owns the nonsemantic status of wall time and deterministic work summaries as
  the cross-machine cost measure; and
- [Full-Stack Architecture](spec-loom-stack.md#external-dependency-pinning)
  owns the exact revisions of every search-harness dependency, including the
  Ray fork whose module and distribution contracts this document targets.

The `RLModule` interface, its column names, and its distribution protocol are
owned by that external dependency. This document states which of its
obligations bind a model and does not restate, extend, or version them; the
training documents configure it.

This document owns only the parts that are the same in every model: what
happens between an observation and the values `Columns.ACTION_DIST_INPUTS` and
`Columns.VF_PREDS` carry, the framing every model configuration view follows,
and the obligations every forward-pass harness satisfies.

## Module Boundary

A model is one `TorchRLModule` implementing `ValueFunctionAPI`, in the new
RLlib API stack:

```text
LoomRLModule(TorchRLModule, ValueFunctionAPI) {
  setup()
  _forward_inference(batch)   -> {ACTION_DIST_INPUTS}
  _forward_exploration(batch) -> {ACTION_DIST_INPUTS}
  _forward_train(batch)       -> {ACTION_DIST_INPUTS, VF_PREDS, ...}
  compute_values(batch, embeddings) -> VF_PREDS
  get_train_action_dist_cls() / get_exploration_action_dist_cls()
    -> the model's action distribution
}
```

The three forward entry points share one encoder pass and differ only in what
they return and in whether exploration-time logit shaping is applied. A
`_forward_train` call must produce logits identical to the
`_forward_exploration` call that collected the same batch, given the same
parameters and the same recorded masks; any divergence makes the policy ratio
wrong for every on-policy update built on it.

`compute_values` accepts precomputed embeddings so that a training step
evaluates the encoder once. Recomputing the encoder for the value head is a
correctness-neutral waste of the dominant cost in the forward pass, and it is
the largest single saving available in the training path.

## Observation Batching

The environment emits a variable-size `GraphInstance` per state, so a batch is
ragged. The model consumes it as one disjoint-union graph in the standard
message-passing form:

```text
BatchedGraph {
  node_columns: per-column array<total_nodes>, each at its declared width
  arc_columns:  per-column array<total_arcs>, each at its declared width
  edge_index:   int64 [2, total_arcs]
  batch:        int64 [total_nodes]
  ptr:          int64 [batch_size + 1]
}
```

Per-instance node ordinals are offset into the union, and `batch` records each
node's originating instance so pooling and per-instance normalization stay
correct. `ActionIndex` is a per-instance ordinal, so the model maps it through
`ptr` and the instance's own offset rather than treating it as a union-wide
index; conflating the two silently scores another episode's actions.

An environment using the `DecisionColumns` encoding supplies a second ragged
instance, and it batches the same way into its own union with its own `batch`
and `ptr` vectors. A column of that union that names a node holds a
per-instance node ordinal and is offset into the node union before it is used
as an index, for the same reason `ActionIndex` is.

Nothing in the batch is padded. The observation's fixed-length arrays batch as
ordinary rectangular tensors.

The batch carries columns, not feature matrices. Arrays arrive in the forms
[Storage Widths](spec-ml-core-environment.md#storage-widths) declares and stay
in them: each column is concatenated across instances at its own declared
width, and no dense `[rows, feature_dim]` tensor of widened columns is ever
built. Building one would restore exactly the bytes that section exists to
remove, on the two most numerous extents, and would do it on the device where
the encoder then has to read around it. There is no node-feature matrix in this
record because the first thing the model does to every column is embed or
project it separately, and the vector those produce is the encoder's input.

`edge_index` is the one widened tensor. The scatter operations require `int64`,
so it is produced on device from the observation's narrower `edge_links` — a
tensor the model builds, not bytes it received.

A large union may be processed in chunks when device memory requires it. A
chunked pass must produce logits and values identical to an unchunked one; a
chunking scheme that changes results is a defect, not a memory strategy, and
per-instance pooling makes the identity achievable because no quantity is
reduced across instances.

## Feature Embedding

Raw observation columns are not fed to the encoder as numbers. Every
categorical column is an ordinal in a closed owner catalog, and an ordinal
consumed as a scalar asserts an order and a distance the catalog does not have:
it would let the encoder interpolate between two unrelated entity kinds. Each
is therefore embedded through its own table, sized from its owner catalog's
cardinality.

Numeric columns are projected rather than embedded, but never consumed raw. A
column the environment marks absent with negative one contributes a zero
magnitude and a set absence indicator, because feeding negative one as a
magnitude makes absence a large negative quantity the encoder cannot
distinguish from a real extreme value. Magnitudes that span orders of magnitude
are projected through a signed logarithmic transform before their linear layer.

A capacity and a usage of the same resource are supplied with their occupancy
ratio alongside the two magnitudes, since headroom is the quantity a placement
or a hardware decision actually trades and the encoder should not have to learn
division.

A capacity and a usage column appended per `FabricResourceStateRef` member are
projected per resource state and reduced to one resource embedding per node, so
the node vector width does not grow with the resource catalog. The environment
core appends that group identically for every environment, so the reduction is
stated once here rather than in each model document.

`objective_codes` are exact bounded integers whose bounds are known: each
dimension's `ExactAffineQuantization` declares its index range, so a code maps
to its exact position in `[0, 1]` with no estimated scale and no running
normalizer. Objective position enters as a graph-level feature so the encoder
can condition a local decision on the current global position.

Scalar features enter as graph-level features. A pair of a counter and its
bound is supplied as its ratio as well as its magnitudes, so remaining budget
is directly available rather than inferred.

All embeddings and projections are concatenated per node and passed through one
input projection to the encoder width, so the encoder sees one uniform node
vector regardless of role. A graph-level feature may instead be projected once
per instance and added to that result, which is exact because one linear layer
over a concatenation is the sum of its per-block projections; materializing a
duplicated row per node and multiplying through it costs with node count where
the equivalent form costs with batch size. Arc embeddings are concatenated to
the arc feature width and supplied to the encoder as edge features.

## Encoder

The encoder is a stack of graph-transformer layers with edge-aware attention.
Attention is over graph neighborhoods rather than over all node pairs, which is
what makes it tractable on states whose node count varies by an order of
magnitude across a curriculum, and what makes the combined node-and-link space
meaningful: a connection node attends to exactly the occurrences it joins.

```text
GraphTransformerStack {
  layers:      positive integer
  width:       positive integer, divisible by head count
  heads:       positive integer
  feedforward: positive integer
  dropout:     ratio in [0, 1)
  edge_dim:    arc embedding width
  jumping_knowledge: None | Concat | Max
}
```

Each layer is pre-normalized: the layer normalizes its input, applies
multi-head edge-conditioned attention, applies dropout, and adds the unmodified
residual, then does the same around a position-wise feed-forward sublayer.
Pre-normalization is required rather than optional, because post-normalization
destabilizes gradient flow at the depths and graph sizes this search reaches. A
final normalization follows the stack so the heads receive bounded features.

The feed-forward sublayer is a departure from the prior architecture and is
stated as one. That stack was attention and residual only, so its single
nonlinearity was the attention softmax and depth bought message-passing hops
rather than capacity. Separating the two is what lets a configuration reach
further across a graph without also being forced to add parameters, and add
parameters without also being forced to reach further.

Arc features are supplied to every layer. Environments distinguish structural
relations from placement, routing, and target relations, and an encoder that
ignored arc features would treat a placement relation, a hardware connection,
and a decision's reference to its target as the same edge. This is also a
departure from the prior architecture, and a sharper one than it appears: that
system had an edge projection and an edge-dimension-aware convolution, but no
arc features ever reached the convolution, so the projection was untrained
weight in the optimizer. Arc features here are load-bearing, not optional.

Jumping-knowledge aggregation over per-layer outputs is supported so the model
can select an effective neighborhood radius per node instead of committing to
the stack depth. `Max` aggregation preserves the encoder width; `Concat`
multiplies it by the layer count, and every head width sized from the encoder
width must be sized from the aggregated width instead. A new aggregation
appends and must state its output width. A configuration whose head widths and
aggregated encoder width disagree is invalid at adoption rather than at the
first forward pass.

One shared encoder produces the graph embedding, and every head reads it. Any
per-head refinement is a small stack applied after the shared trunk, never a
second trunk from raw features: separate trunks discard the representation
sharing that makes the value head's signal useful to the policy, and double the
dominant cost of the forward pass.

## Graph Context

Every head needs a whole-graph summary, and a single mean over nodes is a poor
one because graph size varies across the curriculum. The context is a
multi-scale pooling of the node embeddings, per batch instance:

```text
context = project(concat(mean_pool(h), sum_pool(h), max_pool(h)))
```

`mean` is size-invariant, `sum` carries extent, and `max` carries the single
most salient node; together they let a head distinguish a small state from a
uniformly-scaled large one. The sum branch is scaled down by the square root of
the instance's node count and the projection is followed by normalization,
because an unscaled sum grows with node count and would otherwise make context
magnitude a proxy for graph size and drive logit scale with it.

Pooling is per instance using the `batch` vector. Pooling across the union is a
defect that leaks one episode's state into another's decisions.

## Value Head

The value head reads the shared graph embedding through the same context
pooling and emits one scalar per instance. It is an ordinary multilayer
perceptron over the context, whose dropout comes from the encoder record and
whose initialization gain is one of the constants a model view declares.

The value head predicts the return of the environment's reward, which is a
signed exact-integer energy difference scaled once at the adapter. Its output
is bounded to a configured symmetric range. That bound is a stability
mechanism, not a semantic claim: a value estimate that escapes the reachable
return range destabilizes the advantage estimate long before it becomes visible
as a policy failure. Its cost is stated rather than hidden — a saturated
estimate has no gradient — which is why the bound is checked against the
reachable return range a training document declares, and not chosen. What that
range is, and how a `value_bound` below it is refused, are owned by
[The Reachable Return Range](spec-ml-core-training.md#the-reachable-return-range).

The value head never reads the action mask, any per-action feature, or anything
else that distinguishes one available action from another. It estimates the
state's worth, and giving it action-set information invites it to model the
enumeration rather than the state.

## Masking

Masking is applied additively to logits, using a large finite negative
constant. Negative infinity is not used: it produces undefined entropy and NaN
gradients when a whole group is masked, and every masked-everything case must
degrade to a defined distribution rather than to a NaN.

Two mask sources compose, in this order:

1. the environment's `action_mask`, which is authoritative and already covers
   slots beyond the live count and any entry the environment itself cleared;
   and
2. advisory masks, which remove actions the configuration declares obviously
   unproductive.

The first arrives packed, in the layout
[Action Surface And Masking](spec-ml-core-environment.md#action-surface-and-masking)
fixes. It is unpacked on the device that will consume it, after the batch has
been transferred and as part of the forward pass, so the packed form is what
crosses the sampler, the sample buffer, and the transfer, and only the expanded
form the logits actually need is ever materialized. Unpacking is a shift and a
test over a tensor whose width is the action capacity — trivial against the
encoder it sits behind, and cheaper than the memory traffic it removes.

Composition is bitwise. The environment's mask and each advisory mask are
combined by conjunction, which is not an implementation note but the exact
statement of the first rule below: an AND can only clear bits, so a
model-side mask that tried to add an action cannot express itself in the
composition at all.

The first is always applied and needs no model-side recomputation: a slot the
environment cleared is a slot with no live entry, and rederiving that bit per
forward pass would scatter ordinals into a capacity-wide tensor in the hot path
to reproduce a value that arrived correct. The second is where liberal masking
is available and also where it is bounded, because an advisory mask that
removes a legal action is a policy commitment rather than a correctness
statement, and must never be presented as proof that an action is bad.

Five rules keep masking from corrupting training.

Masks never add actions. A mask may only clear a bit the environment set; a
model-side mask that sets a bit the environment cleared proposes an action the
environment will refuse as masked.

A fully masked state falls back. If composing the masks leaves no admissible
outcome, the composition is discarded and the environment's mask alone is used,
and the fallback is counted. Silently emitting a uniform distribution over
impossible actions produces a step the environment refuses for reasons no
diagnostic explains.

The mask used at collection is recorded and reused at update. Advisory masks
may depend on configuration that changes across a run, so recomputing them at
training time can mask an action that was available when it was sampled. The
resulting log-probability is not the behavior policy's, and the importance
ratio is silently wrong. The mask travels in the batch, packed, in the
environment's own layout: a batch that carried the expanded form would spend
eight times the memory on the one column it retains for the whole update, and
the recorded bits are the same bits either way.

Mask composition and distribution construction run in full precision, outside
any reduced-precision region, on an explicitly cast copy of the logits. The
additive constant is far outside the range of common half-precision formats, so
a masked logit computed in half precision saturates to infinity and
reintroduces exactly the undefined entropy and NaN gradients that choosing a
finite constant avoided. The related hazard is worth naming: under mixed
precision a trunk may remain in one precision while a head's linear layers emit
another, so a head that scatters into a buffer typed from the trunk fails on a
dtype it did not expect.

A logit vector materialized at the action capacity is filled at its unreal
slots with the masking constant, never with zero. A trainer rebuilds the
distribution from `ACTION_DIST_INPUTS` to compute the update log-probability,
so a zero fill gives those slots real probability mass and normalizes the
rebuilt distribution differently from the one that was actually sampled. The
two log-probabilities then disagree for reasons no loss curve reveals.

That is a rule about a vector inside a forward pass, not about what the batch
retains. `ACTION_DIST_INPUTS` travel at the live count, ragged, exactly as
`decisions` does. Retaining them at the capacity would put four bytes per
outcome per sample beside a mask that
[Storage Widths](spec-ml-core-environment.md#storage-widths) just reduced to
one bit, held for as many epochs as the algorithm takes, and the slots so
retained are at the masking constant and contribute nothing to the ratio or to
the KL term. The rebuilt distribution is identical either way, because a slot
at the masking constant and an absent slot carry the same probability.

Logits are not clipped. Clipping bounds them by zeroing the gradient on exactly
the actions the policy is most confident about, and the masked distribution is
already bounded without it. A model that needs clipping to stay finite has a
scale problem in its heads that clipping would hide.

Hierarchical masking is consistent by construction: an anchor group is masked
exactly when all its members are masked, so a group never retains probability
its members cannot receive, and a member never receives probability from a
masked group.

## Action Distribution

The distribution is a custom RLlib Torch distribution over the environment's
two-component action space, exposing sampling, log-probability, entropy, and
KL. Its inputs are the masked joint logits, and it samples one joint outcome
and emits the `{decision, stop}` pair that outcome encodes.

The head emits one joint distribution over the `enumeration_bound + 1` outcomes
those two components can jointly take:

```text
outcome i < enumeration_bound  ->  { decision: i, stop: 0 }
outcome enumeration_bound      ->  { decision: 0, stop: 1 }
```

`decision` is ignored by the environment when `stop` is one, so the emitted
value is fixed at zero to keep the encoding one-to-one; a sampler that emitted
a live index alongside a set stop flag would produce two encodings of one
action and split its probability across them. Log-probability and entropy are
computed over the joint outcome, never per component, because the components
are not independent: exactly one of the two carries the action.

The single joint softmax is what makes acting and stopping mutually exclusive.
Two independent per-component distributions would let a policy simultaneously
raise an action's probability and the stop probability, which the environment
resolves by discarding the action, so the gradient on that action would be
attributed to something that never occurred.

Entropy is computed over admissible outcomes only. Including masked slots at
the mask constant contributes a vanishing but nonzero term that scales with
`enumeration_bound` rather than with the live count, which makes the entropy
bonus depend on the action-space capacity instead of on the choice actually
available.

Sampling is deterministic given a seed and the logits. The model owns its own
sampling randomness; an environment's determinism policy covers episode
construction only.

## Policy Head Factorization

A policy head scores each live entry from the embeddings of the nodes that
entry names. Each model owns what its own terms read; what every model shares
is the shape those terms compose in.

A head factors an entry's logit into an anchor term over what the entry acts on
and a selection term over what it selects, and entries sharing an anchor form
one anchor group. `s_anchor` is constant within a group by construction,
because the anchor includes every input the anchor term reads, which is what
makes the factorization a hierarchy rather than a reparameterization:

```text
P(e)          = P(anchor(e)) * P(e | anchor(e))

P(anchor)     = softmax over anchor groups of
                  s_anchor(g) + logsumexp over members of s_select
P(e | anchor) = softmax over that group's members of s_select(e)
```

The two-level form and the flat softmax over `logit(e)` induce the same
distribution, so an implementation may compute either; the hierarchy is
normative for how the scores compose, not for the order of arithmetic. Because
`s_anchor` is constant within a group it may be evaluated once per group and
broadcast, and a harness reports the mean anchor-group size alongside its head
stage, since the saving is exactly proportional to that mean and is largest in
the configurations where the hierarchy is doing the most work.

What is not permitted is a head that scores an entry from its anchor alone with
no selection term, which collapses every alternative on one anchor into one
indistinguishable action.

Stop is one additional logit computed from the graph context alone, since
electing to stop is a property of the state rather than of any entry.

## Parameters And Checkpoints

A checkpoint is the parameter tensors plus the exact model configuration that
shapes them. Loading requires an exact match of every dimension the
configuration fixes, including the embedding table sizes, which are derived
from owner catalog cardinalities. A catalog that gains a member invalidates the
tables sized from it, and the load fails rather than silently truncating or
zero-extending a table whose ordinals have shifted.

A model checkpoint is not a `ModelParameterBundle` and its reference is not a
`ModelParameterContractRef`. Those name registered prediction contracts whose
outputs become metric predictions inside `EvaluationEvidence`, subject to
support-region and leakage rules. A policy network predicts which action to try
next; it produces no metric, enters no Evidence, and is never consulted for a
value a quality gate or an objective dimension reads. Confusing the two would
put an unvalidated search heuristic inside the evaluation path.

A checkpoint carries parameters, and a consumer that loads one for a purpose
other than resuming its own run takes the parameters and nothing else.
Optimizer moments, a learning-rate position, and any algorithm-owned running
statistic belong to the run that produced them and are meaningless under a
different objective; a training document that sequences one algorithm after
another states that boundary, and this document fixes only that a checkpoint is
able to be read that way — its parameter tensors are addressable without its
training state.

## Model Configuration Framing

Each model's shape is one immutable component view of its own, following the
framing owned by [Component Views](spec-config-ssot.md#component-views). This
document fixes only three properties of any such view.

Every embedding table's size is derived from its owner catalog's cardinality
rather than declared, so a view carries widths and the catalogs fix heights.

A view sizes the parameter tensors, which is why no other view may carry one of
its fields. What a checkpoint pins is the tensor-shaping projection of the
view: the embedding widths, the encoder record, the head widths, and any field
that selects between scorers of different parameter shapes. The rest of the
view — the value bound and the advisory mask set — is recorded on the
checkpoint and checked for reporting, not required to match.

The split is the same one
[Configuration Split](spec-ml-core-training.md#configuration-split) draws for
the training view, and for the same reason: a value bound is a clamp scalar and
an advisory set is a masking policy, and neither changes the shape of a single
tensor. Pinning them would make a sweep over either discard every checkpoint
the run had produced and retrain from scratch, which confuses a parameter-shape
constraint with a policy preference. A run that raises its `step_bound` mid-run
may have to raise its value bound with it, and that must not invalidate the
parameters it has already learned.

A view declares the encoder record, the head widths, the value bound, the
advisory mask set, and the initialization constants. `HeadWidths` is one
positive hidden width per head the model declares, named for that head, so a
model with two policy heads and a value head declares three; the record is not
fixed here because the heads are not.

`AdvisoryMaskRef` names one member of the closed catalog of advisory masks a
model document declares, and `InitializationConstants` is that document's
per-tensor initialization gains. Both are per-model for the same reason the
heads are. Two models never share one
view, because their catalogs differ and a shared view would size a table for a
catalog the other model does not have.

## Forward Benchmarking Obligations

A model runs once per environment step during rollout collection, so its
forward latency multiplies by every step of every episode of every worker.
Every model therefore ships a harness that measures the forward pass directly,
on the removable-projection terms
[Nonsemantic Boundary](spec-ml-core-environment.md#nonsemantic-boundary) sets
for every harness in this stack.

Each model document names its own harness binary and its own closed list of
forward stages, because the heads differ and a shared stage list would either
be too coarse to locate a regression or carry stages one model does not have.
Stages partition the pass: none contains another, so the stage sum is the pass.
What every harness owes is the same.

Unpacking the mask and widening a narrow column are not stages. A cast is
required to happen immediately before the operation that consumes it and only
for the columns that operation reads, so it is inside `embed` for a feature
column and inside the masking stage for the mask; timing it separately would
either double-count those stages or force an unfused materialized copy, and the
copy is the capacity-width tensor
[Storage Widths](spec-ml-core-environment.md#storage-widths) exists to prevent.
An instrumented run that materialized it would also stop measuring the
uninstrumented one.

What every harness reports instead is the quantity that makes the layout
falsifiable: bytes unpacked and bytes cast, attributed to the stage that did
it, together with a reference run that widens at the boundary rather than on
device. The difference between the two is the layout's actual value, and it is
a number rather than an argument. A layout claimed to save bandwidth and
measured by a stage that only exists when the saving is discarded would answer
the wrong question.

Two regimes are measured separately and never averaged. Rollout inference is
latency-bound: batch size is one instance per environment copy, graphs are
small, and fixed per-call overhead dominates. Training is throughput-bound:
batches are large, graphs are batched into one union, and arithmetic dominates.
An optimization that helps one routinely harms the other, so one number for
both hides which is binding.

The three forward entry points are measured separately, and `compute_values` is
measured both with and without precomputed embeddings, because the encoder-once
rule is the largest single saving in the training path and a harness that never
measures it cannot show that it holds.

Cost is reported as a function of state size, never as a single number. A
curriculum moves states across an order of magnitude, so a latency measured on
a small one predicts nothing about a large one. Each stage is reported against
the size quantities that drive it, and a stage whose measured growth exceeds
its expected order is a defect worth finding.

The split against the environment step is reported before anything else.
Sampling throughput is set by the sum of the environment step and this model's
forward pass, and a forward pass that is a small fraction of a step can be made
twice as fast for no throughput gain at all. Reporting the split first is what
prevents optimizing the smaller half.

Every optimization is validated against an unoptimized reference on the same
inputs and parameters, and a configuration whose logits or values differ beyond
a declared tolerance is reported as a failed configuration rather than as a
faster one. Inference-time and training-time numerics must agree within that
same tolerance: a model whose rollout path is optimized differently from its
learner path produces log-probabilities the learner did not produce, and the
resulting importance ratio is wrong for the same reason a recomputed mask makes
it wrong. Speed obtained by letting the two paths diverge is not speed; it is a
silent change to the objective being optimized.

Two optimizations carry traps general enough to state here. A compiled or
graph-captured execution path that treats node and arc counts as static will
recompile continuously and run slower than the eager path, and the
recompilation is invisible in a steady-state average, so recompilation counts
are reported alongside latency and any bucketing that reduces them is reported
with the bucket boundaries it introduces. And reduced precision excludes mask
application, for the reason Masking states, so a harness verifies that masked
logits remain finite in every precision configuration it measures.

A regression budget is a ratio against a recorded baseline of exact parameter
and configuration identities at a fixed state size, never an absolute duration.
The primary reported figure is the model's share of rollout wall time, because
that is the quantity an improvement has to move to matter.

## Conformance Anchors

Stable tests cover a categorical column reaching the encoder through an
embedding table rather than as a scalar, and an absent numeric column being
distinguishable from a real extreme value; a capacity and usage pair reaching
the encoder with their occupancy ratio; an objective code mapping to its exact
position in `[0, 1]` from its declared quantization bounds; a graph-level
feature added once per instance producing the same result as a per-node
broadcast; per-instance pooling never mixing nodes across a batched union; a
node-naming column of a batched decision union being offset into the node union
before use; a chunked pass reproducing an unchunked pass exactly; an
`ActionIndex` resolving to its own instance's entries; a configuration whose
head widths disagree with a `Concat` jumping-knowledge width being refused at
adoption; the value head producing identical predictions whether or not it is
given precomputed embeddings, and never varying with the action mask; a masked
slot receiving no probability and contributing no entropy; a fully masked
composition falling back to the environment mask and counting the fallback; a
model-side mask that sets a bit the environment cleared being refused, and mask
composition being a conjunction that cannot express such a mask at all; the
environment's mask being unpacked on the device that consumes it and reaching
the batch and the sample buffer packed; an unpacked mask agreeing bit for bit
with the environment's packed one, including across the pad bits of a final
byte; a mask recorded at collection being the mask applied at update, and a
batch whose recomputed mask differs being rejected rather than used; unreal
slots of a
materialized logit vector carrying the masking constant at its unreal slots,
`ACTION_DIST_INPUTS` being retained at the live count rather than at the
capacity, and a distribution rebuilt from those inputs reproducing the sampled
distribution's log-probability; masked logits remaining finite in every
supported precision
configuration and mask composition running outside any reduced-precision
region; logits reaching the distribution unclipped; a sampled stop outcome
decoding to a zero `decision` component, and the joint distribution never
assigning probability to a set stop flag beside a live index; a stop logit
competing in the same normalization as the action logits; `_forward_train` and
`_forward_exploration` producing identical logits for equal parameters,
observations, and masks; `_forward_inference` computing no value; a checkpoint
load failing when a catalog cardinality no longer matches the embedding table
it sized; and every optimized configuration reproducing the unoptimized
reference's logits and values within the declared tolerance.

Tests do not pin layer counts, widths, head counts, initialization constants,
advisory mask sets, learned parameter values, wall-time numbers, device
placement, precision configurations, or tolerance values.
