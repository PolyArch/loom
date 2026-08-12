# ML Training Core

This document defines the contract every Loom reinforcement-learning training
run shares: how a run's settings split across immutable component views, how an
algorithm is bound and how a run made of several algorithm stages hands off
between them, how hyperparameters and schedules are expressed exactly, what an
episode statistic may be computed from, how a fixed test set is executed and
aggregated, and what a checkpoint and a run identity are.

It fits no particular policy to no particular environment. Which search is
being trained, what a corpus of training instances is, and which statistics are
worth reporting about that search are owned by the training documents that
build on this one:

- [ML DSE Training](spec-ml-dse-training.md) fits the design-space policy; and
- [ML PnR Training](spec-ml-pnr-training.md) fits the place-and-route policy.

A training run is a search-harness activity. It publishes no Artifact, acquires
no Evidence, and produces no candidate. Its outputs are a policy checkpoint,
which
[Parameters And Checkpoints](spec-ml-core-model.md#parameters-and-checkpoints)
already establishes is not a registered prediction contract, and a stream of
removable statistics.

## Ownership

Every fact this document depends on resolves to one exact owner:

- [ML Environment Core](spec-ml-core-environment.md) owns the episode
  protocol's shape, the action surface, the reward boundary, the step
  accounting identity, the termination-versus-truncation rule, and the
  determinism and copy-coordinate contract every seed stream derives from;
- [ML Model Core](spec-ml-core-model.md) owns the module boundary, the masking
  discipline, the action distribution, and the checkpoint's parameter and
  configuration match;
- each environment document owns its own resolved configuration view, its
  priced outcomes, its terminal reasons, and its trajectory record;
- each model document owns its own configuration view and the tensors it sizes;
- [Objectives and Quality Gates](spec-dse-feedback.md#objectives-and-quality-gates)
  owns `ObjectiveDimension`, `ExactAffineQuantization`, directed codes,
  `ObjectiveVector`, `WeightedLevel`, and `SearchEnergyRef`;
- [Model Parameters and Training](spec-dse-feedback.md#model-parameters-and-training)
  owns the training of registered prediction contracts, which is a different
  activity from this one and shares no record with it;
- [Evaluation Metrics](spec-evaluation-metrics.md#metric-registry) owns
  `ExactRatio`;
- [Component Views](spec-config-ssot.md#component-views) owns view framing,
  canonical bytes, and `component_view_digest`;
- [Search Policy And Determinism](spec-pnr.md#search-policy-and-determinism)
  owns the seeded PRNG protocol;
- [Operational Observations](spec-dse-feedback.md#operational-observations)
  owns the nonsemantic status of wall time; and
- [External Dependency Pinning](spec-loom-stack.md#external-dependency-pinning)
  owns the Ray fork revision and its patch stack.

This document owns only the configuration split and its composition rule, the
nonsemantic boundary, the algorithm binding surface and the stage sequence, the
hyperparameter and schedule expression, the reachable-return-range obligation,
the reward adapter, the rollout and learner topology, the stage invariance
rule, the statistics production rule and the environment-independent
statistics, the test protocol, checkpoint and run identity, the reproduction
claims, and the harness surface.

## Nonsemantic Boundary

A training run is not an evaluation, a selection, a promotion, or a candidate
generation. It publishes no Artifact, no `EvaluationRequest`, no
`EvaluationEvidence`, no `InvocationManifest` record, and no lineage edge. A
statistic, a test score, a checkpoint, and a log stream are removable
projections: regenerating them may change presentation but must preserve every
referenced semantic fact, and no consumer may treat one as a semantic result.

A result a training run discovers becomes real only through the replay path its
environment document defines. A test score never gates promotion, never enters
a quality gate, and never selects a candidate; it selects a checkpoint, which
is a search heuristic.

## Configuration Split

A run's settings are several separate immutable component views plus one that
binds them, each following the framing, canonical byte representation, and
digest contract owned by
[Component Views](spec-config-ssot.md#component-views). Every training document
declares its own descriptor bytes, qualified by the search it fits, because two
runs over different searches share no view and an unqualified descriptor would
let one be adopted as the other.

The run view binds one digest of each participating view and nothing else.
Every training document declares its own, and each carries at least a training
view, a model view, and a test-set view; what else it binds depends on where
its training instances come from.

Composition is by digest rather than by inlining, and three rules follow from
it.

No field appears in two views. A quantity is owned by exactly one view, and a
view that repeats another's field creates two sources for one fact that a
future edit will silently disagree about. In particular a training view carries
no episode bound and no observation policy; those are the environment's, and a
run that wants them different binds a different environment view.

Changing a hyperparameter does not disturb the environment digest. Every
environment declares some cached asset whose key names nothing a training view
carries, so a hyperparameter sweep leaves that cache valid. Had the two been
one view, every sweep would have discarded a cache that takes real work to
warm, and the sweep's first iterations would have measured a cold start rather
than learning.

A checkpoint pins the model digest alone. Loading requires an exact match of
the model view, because that is what sizes the parameter tensors; it requires
nothing of the training view, because fine-tuning a checkpoint under different
hyperparameters is an ordinary operation and refusing it would confuse a
parameter-shape constraint with a policy preference. The environment digest is
recorded rather than required, since a checkpoint remains loadable against a
different instance pool but not against a different action-space shape.

Adoption validates each view independently, then validates the run view's
cross-view conditions. A run whose views are individually valid and jointly
inconsistent fails at adoption, not at the first step that notices.

### Inventory Identity

Every inventory a view names — instances, corpora, demonstrations, test cases —
is enumerated by exact identity. A run never names a directory, and no
inventory is ever resolved by globbing a path. A globbed set changes when a
file lands in it, so two runs that recorded the same configuration would have
trained on different data with nothing in either record to distinguish them,
and the digest that is supposed to identify a configuration would identify only
its spelling.

Where an inventory is generated rather than named, generating it again from the
recorded seed, protocol, and view digests reproduces it exactly. No host
entropy, no wall clock, and no container iteration order participates. A
generator that cannot be re-run to the same set makes its own view's digest
meaningless, since the digest would then name a recipe with more than one
result.

Each training document names the inventory types its run carries and the
identity by which each is compared; the rule that they are enumerated and
regenerable is this one.

## Training Configuration

```text
ResolvedTrainingConfigView {
  stages: ordered nonempty sequence<TrainingStage>
  topology: RolloutTopology
  reward_adapter: RewardAdapter
  logging: LoggingPolicy
  evaluation: EvaluationSchedule
  checkpointing: CheckpointPolicy
}

RolloutTopology {
  num_env_runners: uint32
  num_envs_per_env_runner: positive uint32
  num_learners: uint32
  rollout_fragment_length: positive uint64 | AutoFromBatch
  total_env_steps: positive uint64
}

RewardAdapter {
  scale: ExactRatio
  advantage_standardization: PerMinibatch | None
}
```

A training document instantiates this record under its own descriptor and adds
no field to it.

The view carries no seed. Every stream a run draws from is seeded elsewhere and
by something narrower: an episode's randomness by the environment view's
[Determinism And Copy Coordinates](spec-ml-core-environment.md#determinism-and-copy-coordinates),
and a test case's
residual randomness by its own `case_seed`. A seed here would seed nothing that
is not already seeded, and would be ambiguous with the environment view's
wherever a reader met the two together.

### Algorithm Binding

Training uses the algorithm implementations of the pinned Ray fork. Loom
authors no policy-gradient loss, no advantage estimator, no clipping rule, no
KL control, and no imitation weighting. The value of grounding a run in an
existing implementation comes entirely from not modifying it: a divergence
between a Loom-authored update and the published algorithm is invisible in a
loss curve and is the first thing a failed run would otherwise have to rule
out.

```text
AlgorithmBinding =
    Ppo(PpoBinding)        // 0
  | Marwil(MarwilBinding)  // 1

PpoBinding {
  gamma: Schedule
  lambda: Schedule
  clip_param: Schedule
  vf_clip_param: ExactRatio
  vf_loss_coeff: Schedule
  entropy_coeff: Schedule
  kl_coeff: Schedule
  kl_target: ExactRatio
  learning_rate: Schedule
  grad_clip: ExactRatio
  grad_clip_by: GlobalNorm | Value
  train_batch_size: positive uint64
  minibatch_size: positive uint64
  num_epochs: positive uint64
}

MarwilBinding {
  beta: ExactRatio
  gamma: Schedule
  vf_loss_coeff: Schedule
  learning_rate: Schedule
  grad_clip: ExactRatio
  grad_clip_by: GlobalNorm | Value
  train_batch_size: positive uint64
  minibatch_size: positive uint64
  num_epochs: positive uint64
  advantage_norm_update_rate: ExactRatio
  advantage_exponent_clamp: ExactRatio
}
```

Ordinals are stable; a new algorithm appends. The union exists because an
offline stage and an online stage are the same run over the same policy, and a
document whose algorithm field admitted only one of them could not express that
at all.

An offline algorithm reads demonstrations rather than sampling the environment.
Which view supplies them is not a field of the binding: the run view already
binds one digest of each participating view, so a binding that named the corpus
again would be a second source for a fact already fixed. A training document
whose run binds an offline stage names the demonstration view there, and states
what a valid one contains.

`MarwilBinding.beta` at zero is behaviour cloning and at one is full advantage
weighting. That distinction is a property of the demonstrations rather than a
preference: on demonstrations whose returns barely vary, the exponential
weighting collapses toward one and the run is cloning whatever `beta` says, so
a training document states which regime its demonstration source supports
rather than leaving `beta` to be tuned into meaninglessness.

`advantage_exponent_clamp` is normative rather than a tuning knob. The
advantage weight is an exponential, and its argument grows with the advantage
estimate an untrained value head produces; past a modest bound the exponential
leaves the range of a single-precision float and every weight becomes infinite
in one step. Clamping the exponent bounds the weight without changing its
ordering.

`vf_loss_coeff` is stated against the policy loss's magnitude rather than
defaulted. A masked per-node policy loss and a scalar value loss differ by
orders of magnitude, and a coefficient chosen without regard to that difference
gives the value head effectively no gradient, whereupon it regresses to the
mean and every advantage the imitation weight reads is noise.

Loom supplies exactly four things to whichever algorithm a stage binds, each
owned elsewhere and none of them the update itself: the environment through
`loomml.rllib`, as
[RLlib Environment Definition](spec-ml-core-environment.md#rllib-environment-definition)
specifies; the `RLModule`, as
[Module Boundary](spec-ml-core-model.md#module-boundary) specifies, together
with the custom action distribution; callbacks that observe episodes and emit
the statistics below; and the custom evaluation function that runs the fixed
test set.

A `Learner` subclass is permitted for instrumentation only. It may read
gradients, activations, and parameter norms and emit statistics; it may not
change a loss term, an optimizer step, a gradient, or an update order. An
instrumented run and an uninstrumented run must produce identical parameters
from identical inputs, which is the same invariance
[Forward Benchmarking Obligations](spec-ml-core-model.md#forward-benchmarking-obligations)
requires of the model's optimizations.

Three obligations the environment and model documents impose land on the
trainer's configuration rather than on its code. Termination and truncation
stay distinct, and the value target must bootstrap past a truncation and must
not bootstrap past a termination; a connector or wrapper that collapses the two
flags is invalid. The mask travels in the batch unchanged from collection to
update, as [Masking](spec-ml-core-model.md#masking) requires, because a
pipeline that lets the learner recompute it produces an importance ratio that
is wrong in a way no diagnostic reveals. And the observation stays a ragged
graph batch, so a configuration that inserts a flattening connector ahead of
the module defeats the only reason the fork exists.

### Stages And Handoff

A run is an ordered sequence of stages, each binding one algorithm and one
environment view:

```text
TrainingStage {
  algorithm:                 AlgorithmBinding
  environment_config_digest: ComponentViewDigest
  advance:                   optional<StageAdvance>
}

StageAdvance =
    AtEnvSteps(uint64)
  | AtPlateau {
      metric: LearnerStatisticRef
      window: positive uint32
      tolerance: ExactRatio
      grace_env_steps: uint64
    }
```

A single-algorithm, single-environment run is the one-stage case and needs no
special form. The final stage carries no advance and every earlier stage
carries one; the run ends when the final stage does. There is no `AtRunEnd`
member, because being last is a fact the ordered sequence already fixes and a
member spelling it would be a second source that adoption then has to check
against the position.

`AtEnvSteps` counts sampled steps from the run's start rather than from the
stage's, so the values across a run strictly increase and the reading is
unambiguous at any point.

A stage binds a whole environment view rather than a delta against its
predecessor, so the configuration that produced any point of a run is
recoverable exactly, and a stage boundary is one atomic change rather than a
set of independently applied overrides.

A stage boundary always transfers **model parameters**. It transfers
**algorithm state** — optimizer moments, the advantage normalizer, the
iteration counter — exactly when the two stages bind the same arm of
`AlgorithmBinding`, and never when the arm changes. Carrying moments
accumulated under one objective into gradients of another produces first
updates that are neither stage's; discarding them across a boundary that only
moved the environment would make every advance a cold restart of the
optimizer, which is the cost that would make a many-stage run unaffordable.

Two failure modes are contract obligations rather than implementation notes,
because both present as a successful run.

A parameter load that does not reach the module is a cold start wearing a warm
start's name. A checkpoint's component tree is addressed by name, and a restore
aimed at a component path that does not exist succeeds silently and loads
nothing. The boundary is therefore observable: the run reports the loaded
parameter count and a digest of the loaded tensors, and a stage that reports
neither has not demonstrated that it loaded anything.

An evaluation at a stage boundary must observe the loaded parameters. Sampling
workers hold their own copies, and an evaluation that draws weights from a
worker rather than from the learner reports the parameters that worker last
held, which at a boundary is the previous stage's or the initial ones. The
boundary evaluation is the reading everything downstream compares against, so
reporting the wrong parameters there mislabels every later comparison.

`AtPlateau` names a learner statistic and stops the stage when its moving
window varies by less than `tolerance`, after `grace_env_steps`. It exists
because a stage's useful length is not always known in advance: a stage that
ends when its objective stops improving is measuring a property of the run
rather than asserting a step count. The metric must be one the stage's own
algorithm emits.

### Exact Rational Hyperparameters

No binary floating-point number appears in a training view. Every real-valued
hyperparameter is an `ExactRatio`, and the `float64` a trainer requires is
produced once where the configuration crosses into Python, exactly as
[Reward Contract](spec-ml-core-environment.md#reward-contract) requires of the
reward scale.

The reason is identity. A view is digest-covered, and a digest over a decimal
literal rounded to a mantissa is a digest over a rounding: two runs that
recorded the same configuration could differ in the last bit of a learning
rate, and no record would distinguish them. Keeping the exact ratio in the view
keeps the configuration a value and moves the conversion to the one layer that
can state the exactness condition its own float type imposes.

The model's `value_bound` is not here. It sizes a clamp inside the model and
belongs to the model view, which is what a checkpoint pins.

### Schedules

A hyperparameter that varies over a run is a schedule:

```text
Schedule =
    Constant(ExactRatio)
  | PiecewiseLinear(ordered nonempty sequence<{
      at_env_steps: uint64
      value: ExactRatio
    }>)
```

The first breakpoint is at zero, breakpoints strictly increase, and the value
past the last breakpoint is that breakpoint's value.

Breakpoints are keyed on environment steps sampled, never on training
iterations. An iteration count depends on batch size, on how many steps a
sampler happened to return, and on how a stage was resumed; a sampled step
count does not. A schedule is therefore a pure function of the step count, so a
resumed run continues the anneal from where it was rather than restarting it,
and two runs that differ only in topology still follow the same schedule
against the same abscissa.

### The Reachable Return Range

The model view's `value_bound` is checked rather than chosen, and what it is
checked against is an obligation on each training document rather than a
formula here.

A training document declares the per-episode return bound of the environment
its run binds, in that environment's own reward vocabulary. It is the document
that can: the arithmetic reads the environment's codes and step bound on one
side and this document's `reward_adapter.scale` and stage sequence on the
other, so neither the environment nor this core holds both halves. The bound is
derived from two things the environment already owns: the quantization width of
its selected search energy, since each
dimension's `ExactAffineQuantization` fixes the interval and a single
transition's signed difference cannot exceed its width; and the codes it
charges for outcomes that are not transitions, together with the bound on how
many of each an episode may carry. Environments charge structurally different
sets of codes, so a formula fixed here would be one search's arithmetic imposed
on another's.

Two rules bind every such declaration. The bound accounts for every step, not
one of each kind: an episode that spends its whole step budget on the most
expensive outcome is not a corner case but the first thing an untrained policy
produces. And where a term's magnitude depends on the instance rather than the
configuration, the declared bound is the maximum over the instance pool, taken
at adoption.

A declared bound is in the environment's own integer reward units, and
`value_bound` is in the scaled units the value head emits, so the comparison
multiplies the declared bound by `reward_adapter.scale` before making it. A
scale above one would otherwise let through a bound that clips every return it
was checked against.

A run's requirement is that scaled maximum across every stage it binds, because
a later stage may raise a bound, a code, or the scale. Adoption requires
`value_bound` to be at least that maximum. A bound below it clips a return the
environment can actually produce, which biases the advantage estimate exactly
where the search is doing best; a bound far above it is permitted and merely
weakens the stability mechanism.

That the range is computable at all is a consequence of the environment
refusing to normalize. Reward is exact, integral, and drawn from a declared
bounded quantization, so nothing here needs an estimated scale or a running
normalizer.

### Reward Transforms

The adapter applies exactly one transform: the declared `ExactRatio` scale that
turns an integer reward into the float the trainer requires.

Running normalization is rejected for three reasons. It makes the reward a
function of the batch rather than of the transition, so the same transition is
worth different amounts at different points of a run. It destroys the exactness
the environment paid for. And it hides exactly the failure it appears to fix: a
reward whose scale is wrong is a configuration error, and normalizing it away
means never seeing it.

Clipping, discounting inside the adapter, potential-based shaping terms, and
per-dimension reweighting are likewise excluded. A run that wants a different
weighting binds a different objective closure, which is a semantic change with
a digest, rather than a transform with none.

`advantage_standardization` is not a reward transform. It standardizes the
advantage estimate inside the loss, which is the algorithm owner's own
mechanism, and it is exposed here only because it is a configuration field a
run sets.

## Rollout And Learner Topology

`num_env_runners` and `num_envs_per_env_runner` are part of the reproduction
tuple, not free scaling knobs. Every environment derives each copy's PRNG
streams from its own coordinates, so changing either count changes which seeds
are drawn and which instances an episode sees. That is a property of the
sampling topology rather than a defect, and the run records both counts so a
run that does not reproduce can be told apart from one that was never
configured the same way.

Sizing follows the measured split rather than a default. Each environment
document reports what dominates its step and each model document reports what
fraction of rollout time inference takes, so a topology is chosen by moving
runner count against that measured split rather than by matching learner count
to available accelerators.

The environment's own interaction contract governs instance lifetime, and this
document adds no second lifecycle.

## Curriculum

A curriculum is not a separate record. It is what the stage sequence already
expresses: consecutive stages whose environment views differ. No nested second
sequence is needed, and admitting one would give a run two advance mechanisms
for the same thing.

Consecutive stages may differ in their environment views only in fields that
fix neither the action-space shape, the observation shape, nor the reward
semantics. A policy cannot be trained across a change in any of the three: the
action space would change extent mid-run, the embedding tables would be sized
for a catalog the observation no longer produces, and the value head would be
regressing toward a different quantity than before.

The rule is stated by what a field determines rather than by naming a record,
because environments partition their views differently and the same field name
sits inside an episode policy in one and beside it in another. Each environment
document names which of its own fields fall on each side, and adoption checks
the partition that document declares. A rule written as one record name would
forbid, in some other environment, exactly the curriculum that environment most
obviously wants.

An environment's cached assets survive a stage advance whenever the advance
changes no field their key names. A key that names an individual instance
rather than the set the view declares therefore survives any advance that only
grows that set, which is what makes a many-stage run affordable rather than a
repeated cold start. Each environment document says which of its caches have
that property.

## Episode Statistics

### The Production Rule

Every episode statistic is a function of values the environment already emits:
observation columns, objective codes, scalar features, the step result, the
action taken, and the retained trajectory. A callback never re-derives a fact
from an Artifact, never calls a semantic owner, and never computes a quantity
an owner would compute differently. A statistic that cannot be produced this
way is not a statistic this document defines; it is a request for the
environment to emit something it does not.

Every statistic is keyed by a member of a closed catalog. No key is built by
formatting a file name, a path, or an identity digest into a metric name.
String-keyed series multiply with the instance pool and cannot be enumerated in
advance, so a consumer cannot tell a missing series from a series that was
never produced.

### Objective Dimension Statistics

One record per dimension of the environment's selected closure, per episode:

```text
DimensionEpisodeStatistic {
  entry_code:       uint64   // at reset
  final_code:       uint64   // at the episode's last state
  best_code:        uint64   // minimum over the episode's states
  step_of_best:     uint64
}
```

A consumer reads improvement as `entry_code - final_code` and best improvement
as `entry_code - best_code`; neither is a stored field, because both operands
are already here and a stored difference is a second source for one fact.

Improvement is uniformly oriented without a per-dimension rule, because a
directed code is already direction-normalized by its `ExactAffineQuantization`:
a lower code is better for every dimension regardless of which direction the
dimension optimizes. `entry_code` is retained because a final value alone is
not a result — episodes start from different instances, and a final code says
nothing about whether the policy improved anything.

`retain_step_series` extends these to a per-step series and is off by default,
because it is one array per dimension per episode and is the largest logging
term by a wide margin.

### Outcome And Action Statistics

Every episode reports the accounting the environment core's step identity
already guarantees:

```text
episode_return                    scaled, matching the trainer's own return
episode_length                    steps, spanning sampler chunks
advanced_steps
non_advancing_steps
elective_stops                    zero or one
unaccounted_steps                 episode_length - the three above
terminal_reason                   one member of the environment's catalog
mean_live_decision_count
max_live_decision_count
mask_fallback_count
```

The names here are the neutral ones
[Step Accounting And Episode Endings](spec-ml-core-environment.md#step-accounting-and-episode-endings)
fixes. Each environment reports the same quantities under its own word for its
non-advancing class and its own word for one member of the live enumeration;
what is fixed is the accounting, not the vocabulary.

There is no headroom statistic. The distance from `max_live_decision_count` to
the environment's `enumeration_bound` is a difference of two values already
recorded, and a stored difference is a second source for one fact on the same
terms the objective statistics above refuse one.

`unaccounted_steps` exists to be zero. The environment guarantees the identity
by construction, so a nonzero mean is a defect in the callback rather than in
the environment, and the usual cause is an episode length read off a single
sampler chunk.

`terminal_reason` is reported as a rate per member of the environment's
terminal catalog rather than as a single mode, because the members mean
different things: one is the ending the policy elected and the rest are limits
the harness imposed, and a policy that never elects its own ending is a
different failure from one that elects it immediately.

A run reports the non-advancing rate per member of the environment's own
failure catalog, never as one aggregate. A single rate conflates causes that
call for opposite responses, and the environment already separated them.

`max_live_decision_count` against the environment's `enumeration_bound` is what
predicts a capacity refusal before it happens.
`mask_fallback_count` counts the model's fully-masked fallback, reported here
because this is the layer that observes it rather than because this layer owns
the advisory set.

Action frequencies are keyed by the environment's own action partition and each
is paired with that partition member's success rate — advances over attempts.
The frequency says what the policy tries and the pair says whether trying
works, and a policy collapsed onto one member of the partition is visible here
and nowhere else.

### Cost Statistics

```text
mean_step_duration
mean_reset_duration
episodes_per_hour_per_copy
model_inference_share_of_rollout
```

Wall time is nonsemantic in the sense
[Operational Observations](spec-dse-feedback.md#operational-observations)
defines, and these four are labelled as such wherever they are reported. None
is summed across concurrent copies. Deterministic work summaries remain the
cross-machine cost measure, and `model_inference_share_of_rollout` is the
figure a model optimization has to move.

Per-stage step decomposition is not reported here. It belongs to the benchmark
harnesses, whose instrumentation is inactive elsewhere; a trainer carrying
stage boundaries per step would pay a measurable fraction to produce numbers
not comparable to its own uninstrumented throughput.

### Breakdown Axes

```text
LoggingPolicy {
  enabled_axes: canonical set<BreakdownAxis>
  retain_step_series: bool
  per_episode_sample_rate: ExactRatio
}
```

`BreakdownAxis` is a closed catalog each training document owns, because the
axes worth splitting a statistic along are the ones its own search has. Two
members are mandatory and take ordinals zero and one in every such catalog:
`Stage`, over the run's stage sequence, and `TerminalReason`, over the
environment's terminal catalog. Fixing their ordinals is what lets a consumer
reading two searches' statistics key on the two axes both are guaranteed to
have; everything after ordinal one is per-document.

Axes are opt-in because their cost is multiplicative: an axis over an instance
pool turns every statistic into one series per pool member, and two such axes
multiply. The default is no axis, and a diagnosis enables the one axis it
needs.

An axis's members are exact references, keyed by identity rather than by a file
name, so a series survives a reorganization of where anything is stored.

The sink is presentation. No sink is normative, and a run that writes to no
sink computes the same policy.

### Chunk Invariance

An episode statistic must not depend on where the sampler cuts an episode. A
sampler returns fragments, and an episode may span several; a statistic
computed per fragment and averaged is a different quantity from the same
statistic computed per episode.

Two forms are permitted: a statistic accumulated across fragments until the
episode ends and emitted once at its end, and a statistic that is a pure
function of the episode's final state. Two are forbidden: a statistic emitted
per fragment as though the fragment were an episode, and a statistic whose
value depends on the fragment length the topology happened to produce.

## Learner Statistics

Loss terms, entropy, KL, explained variance, gradient norms, and timing are the
algorithm owner's statistics, reported unchanged. This document neither
redefines nor renames them, and three of them are worth reading against
contracts stated elsewhere.

Entropy is over admissible outcomes only, per
[Action Distribution](spec-ml-core-model.md#action-distribution), so it is
comparable across states with different live counts. An entropy that tracks
`enumeration_bound` rather than the live count indicates entropy computed over
masked slots.

Explained variance is against a return whose range is declared, so a
persistently negative value is a value-head failure rather than a scaling
problem.

A KL spike at a curriculum boundary is the expected consequence of the
environment view changing under a fixed action space, and a KL spike at a stage
boundary is the expected consequence of the objective changing under fixed
parameters. Neither is a defect; both are worth distinguishing from a spike
with no boundary near it.

## Test Protocol

### The Test Set

```text
ResolvedTestSetConfigView {
  environment_config_digest: ComponentViewDigest
  cases: ordered nonempty sequence<TestCase>
  case_timeout: optional<nonsemantic duration>
}

TestCase {
  instance: <the environment's episode start override payload>
  case_seed: u64
}
```

A case names its instance exactly. Nothing is drawn. A case runs by passing its
instance as the episode start override its environment defines, so no selection
stream is consulted and the copy coordinates cannot reach the episode. That is
what makes a case's result independent of which evaluation runner executed it,
and why the protocol needs no coordination between runners.

A test set's inventory is disjoint from that of every stage the run binds,
compared by exact identity. A case the policy trained on measures how well the
run memorized it, which is the one thing a test score is not for. Each training
document names only the granularity at which identity is compared, since what
counts as one instance differs by search.

A test set binds its own environment view, which is what makes disjointness
expressible at all: an override may only name inventory the view it runs
against declares, so a test set sharing a training view would have to name
inventory that view declares and could never be adopted. Adoption checks that
the case inventory is exactly what the test set's own view declares.

The two view kinds must agree on everything a checkpoint depends on. A test-set
environment view differs from the training views only in fields that fix
neither the action-space shape, the observation shape, nor the reward semantics
— the same partition consecutive stages obey — because a score produced
against a different action space is not a score of the same policy.

The set is ordered and complete. Every case runs on every evaluation, in case
order, exactly once. A sampled test set is rejected: a score that moves because
a different subset was drawn is indistinguishable from a score that moves
because the policy changed, which defeats the only purpose the set has.

### Determinism

A test episode is deterministic in both halves. The policy acts greedily —
exploration off, the distribution's mode rather than a sample — so the model
contributes no randomness, and `case_seed` seeds whatever residual randomness
an implementation retains. The environment is reproducible given its
configuration, the copy coordinates, the seed, and the action sequence, and a
case consults no selection stream at all.

Both halves being deterministic is what discharges the exact-reproduction claim
[Reproduction](#reproduction) makes for a test run, and that claim in turn is
the only reason two checkpoints are comparable at all.

### Execution And Aggregation

Cases are distributed across evaluation runners by case ordinal. Distribution
is a throughput decision with no semantic content.

```text
TestCaseOutcome =
    Completed { the episode statistics catalog above }
  | StartFailed { the environment's episode start outcome }
  | TimedOut { steps completed }

EvaluationSchedule {
  every_env_steps: uint64
  at_run_end: bool
}
```

A failed or timed-out case is reported, counted, and excluded from the
completed aggregate — never dropped. Dropping biases toward the cases that ran,
so a degrading policy would show an improving mean as its hardest cases quietly
left the set. A result therefore carries the completed count alongside every
aggregate, and an aggregate over a changed number of cases is not comparable to
its predecessor.

`case_timeout` is nonsemantic wall time. A timed-out case retains its partial
statistics, tagged as partial, and they are never averaged into a completed
aggregate.

Evaluation runs on a schedule keyed on sampled environment steps, for the same
reason schedules are.

### What A Test Score Is Not

A test episode falls on the search-heuristic side of
[Nonsemantic Boundary](#nonsemantic-boundary), on the same terms as a training
episode. The boundary is worth restating here because a test episode is where
it is easiest to lose: these episodes run the same machinery a real search
would and read the same objective dimensions, so the numbers look like results,
and treating them as results would put an unvalidated search heuristic where a
verified one belongs.

## Checkpoints And Run Identity

```text
CheckpointPolicy {
  every_env_steps: uint64
  retain_last: positive uint32
  retain_best_by_test_score: bool
}
```

A run is identified by its run view's digest and by nothing beside it. There is
no run-key record, because a record with one field is a name for that field and
a second place to change it. Everything a key would want is already inside the
digested view — the topology counts, the stage sequence, every participating
view — so a run at a different runner count already has a different identity,
and carrying any of it again would be two sources for one fact.

A checkpoint records that digest, the sampled step count, the stage ordinal and
stage it was taken in, and the parameters. Loading requires an exact
model-view match per
[Parameters And Checkpoints](spec-ml-core-model.md#parameters-and-checkpoints),
and a load whose catalog cardinality no longer matches an embedding table fails
rather than truncating it.

A checkpoint selected by test score records which test-set digest selected it,
because a score is only meaningful against the set that produced it.

## Reproduction

This document makes exactly two reproducibility claims and declines a third.

A test run reproduces exactly: one checkpoint and one test-set digest give
identical statistics on any machine at any runner count.

A trajectory reproduces exactly: the recorded configuration, coordinates, seed,
and action sequence reconstruct the episode its environment's replay path
defines.

A learning curve does not reproduce bit-for-bit, and claiming otherwise would
be false. Concurrent learner reductions, accelerator kernel nondeterminism, and
asynchronous sample collection whose completion order depends on wall time all
perturb the update sequence. What a run records is enough to reconstruct its
data, its configuration, and every episode it ran — which is what reproduction
has to mean here.

## Harnesses

Every training document names its own harnesses, qualified by the search they
fit, on the removable-projection terms
[Nonsemantic Boundary](spec-ml-core-environment.md#nonsemantic-boundary) sets
for every harness in this stack.

Two obligations bind them wherever they appear. A harness that runs a test set
against a checkpoint runs the same protocol the training-time evaluation runs,
so a score produced during a run and a score produced afterward are the same
quantity. And a harness that prepares training data verifies an existing
preparation without regenerating it, so a corpus can be checked without being
rebuilt.

## Conformance Anchors

Stable tests cover a hyperparameter edit changing the training digest and
leaving the environment digest and every cached asset valid; a checkpoint
loading under a changed training view and failing under a changed model view; a
schedule evaluated at a given sampled step count producing the same value on a
resumed run as on an uninterrupted one; a stage boundary transferring
parameters always, and transferring optimizer moments, the advantage
normalizer, and the iteration counter across a same-arm boundary while
transferring none of them across an arm change; a stage boundary reporting the
loaded parameter count and
tensor digest, and a restore aimed at a path that does not exist failing rather
than loading nothing silently; a boundary evaluation observing the loaded
parameters rather than a sampler's stale copy; an `AtPlateau` advance naming a
statistic its own stage's algorithm emits; a single-stage run behaving
identically to the same configuration expressed without stages; an instrumented
learner producing parameters identical to an uninstrumented one from identical
inputs; a truncation bootstrapping and a termination not; a batch whose mask
was recomputed rather than carried being rejected; every episode statistic
being a function of environment-emitted values alone; `unaccounted_steps` being
zero over a complete episode; an episode statistic computed across a sampler
chunk boundary equalling the same statistic computed from an unchunked episode;
consecutive stages differing only in fields the environment declares neutral
being accepted and ones differing in a shape-fixing field being rejected;
algorithm state carrying across a same-arm stage boundary and resetting across
an arm change; a
`value_bound` below the environment's declared return range being rejected,
including where a term's magnitude is instance-dependent and the maximum over
the pool is what binds, and the comparison being made against the declared
bound scaled by `reward_adapter.scale`; a test set running every case exactly
once in case order; a test case consulting no environment selection stream and
producing an identical trajectory at two runner counts and two copy
coordinates; one checkpoint and one test-set digest reproducing identical
statistics across machines; a failed or timed-out case being counted and
excluded from the completed aggregate rather than dropped; and a run's identity
being the run view digest alone.

Tests do not pin hyperparameter values, schedule breakpoints, stage counts,
plateau tolerances, topology counts, the contents of any particular test set,
wall-time numbers, throughput, learning curves, achieved test scores, sink
formats, or diagnostic text.
