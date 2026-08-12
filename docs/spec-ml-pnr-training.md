# ML PnR Training

This document defines how the place-and-route policy of
[ML PnR Model Architecture](spec-ml-pnr-model-architecture.md) is fitted
against the environment of [ML PnR Environment](spec-ml-pnr-environment.md). It
owns the demonstration corpus that pretraining imitates, the two-stage run that
turns those demonstrations into an online policy, this environment's
return-range arithmetic and statistics, and the test protocol that says whether
the run improved anything.

A run has two stages and they answer different questions. The first learns
where a good placement puts things, from demonstrations a simulated annealer
produced. The second learns to do better than that, from its own experience.
The whole point of their being one run rather than two is that the second
begins from the first's parameters.

## Ownership

Every fact this document depends on resolves to one exact owner:

- [ML Training Core](spec-ml-core-training.md) owns everything a Loom training
  run does that is not specific to what is being searched: the nonsemantic
  boundary, the configuration split and inventory identity, the algorithm
  binding surface, the stage sequence and its handoff, exact-rational
  hyperparameters, schedules, the reward adapter, topology, the stage
  invariance rule, the statistics production rule, the test protocol,
  checkpoints and run identity, reproduction, and the harness surface;
- [ML PnR Environment](spec-ml-pnr-environment.md) owns the episode arms, the
  action space and its phase mask, the observation, the reward and every code
  it charges, `PnrTerminalReason`, `PnrTransitionFailureReason`,
  its episode start override payload, `PnrTrajectory`, and
  `ResolvedMlPnrEnvironmentConfigView`;
- [Model Configuration](spec-ml-pnr-model-architecture.md#model-configuration)
  owns `ResolvedMlPnrModelConfigView` and the tensors it sizes;
- [Annealing And Replay](spec-pnr.md#annealing-and-replay) owns the annealing
  policy, its acceptance kernel, and the seeded PRNG protocol the demonstration
  generator draws from;
- [Deterministic Initialization And Action Proposal](spec-pnr.md#deterministic-initialization-and-action-proposal)
  owns the Action proposal selector and the frozen-topology hop distance
  `PlacementLocality` orders by;
- [Objectives and Quality Gates](spec-dse-feedback.md#objectives-and-quality-gates)
  owns `ExactAffineQuantization` and the search energy the return range is
  derived from; and
- [Component Views](spec-config-ssot.md#component-views) owns view framing,
  canonical bytes, and `component_view_digest`.

This document owns only the demonstration corpus and its generator, the PnR
stage configuration, this environment's return-range arithmetic, its statistics
catalog and breakdown axes, its test set and comparison rules, and its
harnesses.

## Configuration Split

This run's views, under the framing
[Configuration Split](spec-ml-core-training.md#configuration-split) owns:

```text
loom.ml_pnr_environment.config.1.0     ResolvedMlPnrEnvironmentConfigView
loom.ml_pnr_model.config.1.0           ResolvedMlPnrModelConfigView
loom.ml_pnr_training.config.1.0        ResolvedMlPnrTrainingConfigView
loom.ml_pnr_demonstrations.config.1.0  ResolvedMlPnrDemonstrationConfigView
loom.ml_pnr_test_set.config.1.0        ResolvedMlPnrTestSetConfigView
loom.ml_pnr_run.config.1.0             ResolvedMlPnrRunConfigView
```

```text
ResolvedMlPnrRunConfigView {
  training_config_digest:      ComponentViewDigest
  model_config_digest:         ComponentViewDigest
  demonstration_config_digest: ComponentViewDigest
  test_set_config_digest:      ComponentViewDigest
}
```

`ResolvedMlPnrTrainingConfigView` is the core's `ResolvedTrainingConfigView`
under this descriptor, with no field added.

There is no corpus view, and its absence is the substantive difference from the
design-space run. That run generates its training instances, so it needs an
architecture corpus, a synthetic workload corpus, structural axes, and a
calibration record to argue that what it generated resembles what it will meet.
A PnR run does not generate anything: its instances are exact
`SpatialPnrProblemBinding` values naming Artifacts that already exist, and a
problem is a problem whether or not anyone calibrated it. What the
demonstration view carries is not a corpus of instances but a corpus of
*placements* for instances the configuration already names.

Adoption validates each view, then three cross-view conditions: every problem
the demonstration view names appears in the training stages' bound environment
views; the test set's problems appear in its own environment view and in no
training stage's; and the model view's `value_bound` covers the return range
below, maximized over every bound stage.

## Demonstration Corpus

Pretraining imitates simulated annealing. What it imitates is where the
annealer *arrived*, not how it got there, and that distinction is the whole
design of this section.

### What Is Imitated

An annealing run is a Metropolis walk. Its intermediate states are bad by
construction — that is what the acceptance kernel is for, and a run that never
occupied a worse state was not annealing. A demonstration recording that walk
teaches a policy to reproduce it: to place badly, then move, then move again.
The prior system recorded exactly that, a greedy fill followed by every
accepted swap, and it was imitating a search rather than a solution.

So the generator keeps the annealer's final placement and discards its path.
The demonstration is a construction sequence that reaches that placement
directly: each realization bound once, to the occurrence the annealer left it
on, in one sweep with no move ever undone.

### The Generator

For each problem in the view, under streams derived from `generator_seed` by
the protocol
[Search Policy And Determinism](spec-pnr.md#search-policy-and-determinism)
owns:

```text
1. reset the environment on that problem
2. step it with a scripted annealer, which selects each Action through the
   proposal selector and acceptance kernel Place and Route already owns,
   under `annealing_policy`, until that policy's schedule completes
3. read the final placement: each Compute and Memory Realization, and the
   occurrence it is bound to
4. discard the action sequence entirely
5. reset again, and emit one `RealizationBindingAction` per realization, in
   `demonstration_order`, binding each directly to its final occurrence
6. record the placement of step 3, the sweep's length, and the energy and
   closure the sweep ended at
```

The scripted annealer is not a second search authority. It selects Actions with
the owner's own proposal and acceptance protocols, and every Action it takes is
an ordinary environment step, so every recorded index is by construction a
legal `ActionIndex` under the phase mask that produced it. Nothing here needs a
recording hook inside Place and Route, and nothing depends on a diagnostic
stream.

`demonstration_order` is a real parameter rather than a detail. Under
`AgentSelected` anchor selection the policy learns which realization to place
next as well as where, so the order a demonstration uses is part of what it
teaches. `CanonicalDecisionKey` uses the owner's canonical typed decision-key
order and is the honest default. `PlacementLocality` orders each realization
after the already-placed neighbours it shares a net with, nearest first by the
frozen-topology hop distance, which is the order a person would place by hand
and the one worth measuring the default against.

### What The Record Carries

```text
ResolvedMlPnrDemonstrationConfigView {
  problems:  canonical nonempty set<SpatialPnrProblemBinding>
  generator: DemonstrationGenerationRecord
  admitted:  canonical nonempty set<DemonstrationRecord>
}

DemonstrationGenerationRecord {
  generator_seed: u64
  prng_protocol: Sha256SeededXoshiro256StarStar_1_0
  annealing_policy: SearchPolicy.annealing
  demonstration_order: CanonicalDecisionKey | PlacementLocality
  pnr_config_view_digest: ComponentViewDigest
}

DemonstrationRecord {
  problem: SpatialPnrProblemBinding
  final_placement: ordered sequence<PlacedOccurrence>
  action_count: uint64
  replayed_final_energy_code: uint64
  replayed_closed: bool
}
```

`final_placement` is the annealer's result and the only thing about the
annealing run this document keeps: the occurrence each Compute and Memory
Realization ended bound to, in the owner's canonical decision-key order, so its
length is the realization count and its meaning needs no second field to say
which entry is which. A `PlacedOccurrence` is whichever occurrence reference
that realization's binding relation targets — a `FabricFuOccurrenceRef` for a
Compute Realization and a `FabricMemoryOccurrenceRef` for a Memory one — so an
entry is exactly what the `RealizationBindingAction` the sweep emits carries,
and no new reference kind appears here.

It is a field rather than something re-derived because it is the one input to
a demonstration that nothing else determines. Given it, the recorded sweep is a
pure function of the problem, the order, and the environment view, so replay
runs no annealer at all. Without it, the only path back to the same placement
is re-running the annealing pass, which this document elsewhere prices as the
dominant cost in this environment — and adoption, every read of the offline
stage, and every verification would each pay it again for a result already
computed once.

`replayed_final_energy_code` is the energy the second pass produced, and it is
the authority. It is not the energy the annealing run reported, and the two
differ: placing realizations in a different order routes their nets
differently, so the same final placement carries a different routing and a
different energy. Recording the annealer's number would describe a state the
demonstration never occupies.

A record carries no observation and no action sequence. Both are regenerated by
sweeping `final_placement` in `demonstration_order` against the environment
view, which is deterministic and involves no search — so the view stays a
configuration rather than becoming a dataset, and a demonstration set is a
value that can be compared by digest. A placement vector is bounded by the
realization count and does not make it one.

### Admission

A demonstration is admitted only if it is something the online stage could have
produced. Three conditions, each a check on the record rather than a run of
it:

Its `action_count` is within the `step_bound` every training stage's
environment view declares. The prior system did not check this, and its offline
episodes ran to lengths the online policy was never permitted to reach, so
pretraining taught a behaviour the environment then forbade.

Its `final_placement` binds every realization exactly once: its length is the
problem's realization count and no anchor repeats. That is what the
"every action is unmasked" condition reduces to here — the construction mask
admits realization bindings on unplaced anchors, so a sweep that never places
one twice is unmasked at every step by construction, and a repeated anchor is
the only way a configured `demonstration_order` could break it. The same length
check establishes that no realization is left unplaced.

Adoption therefore performs no sweep. Both structural conditions are decidable
from the placement vector directly, and running a construction pass per
demonstration to rediscover them would make every adoption — and every restart,
and every sweep iteration that re-adopts — pay one probe with route closure per
realization for facts already in the record. Checking that the recorded energy
and closure are the ones a sweep actually produces is
[Harnesses](#harnesses) work, done once against a corpus rather than at every
run that binds it.

A demonstration covers construction and nothing else. Under
`ConstructThenAnneal` that is the whole episode, so pretraining sees complete
episodes. Under `ConstructThenRepair` the repair phase has no demonstration at
all, and learning it is the online stage's job — which is the division of
labour the two stages exist for, not a gap in the corpus.

### Determinism

The demonstration set is the inventory this run carries, enumerated and
regenerable on the terms
[Inventory Identity](spec-ml-core-training.md#inventory-identity) sets. What
discharges the regeneration half here is that the same `generator_seed`,
`annealing_policy`, `demonstration_order`, and `pnr_config_view_digest`
reproduce the same admitted set exactly: every draw the generator makes comes
from a stream the protocol derives. The prior generator failed all of this at
once — it
shuffled with an unseeded generator, it passed its seed to a flag the scheduler
did not read for scheduling so every retry re-ran an identical stream, and its
output layout varied with task completion order — and the resulting corpus
could not be rebuilt or compared.

## Stages

The default run is two stages:

```text
stage 0   Marwil   advancing AtPlateau
stage 1   Ppo      final, carrying no advance
```

Both are the core's `TrainingStage`, and the handoff is the core's: stage 1
begins from stage 0's parameters and, the arm having changed, from no algorithm
state at all. The demonstrations stage 0 reads are the ones
`ResolvedMlPnrRunConfigView.demonstration_config_digest` binds; the binding
names no corpus of its own, so there is exactly one place a run says which
placements it pretrains on.

`gamma` must agree across the two stages, and adoption checks it. The reward is
an exact energy difference, so a discount is not a tuning knob but a statement
about which future improvements count; a pretraining stage that discounted
differently would fit a value head to a different return than the one the
online stage optimizes, and the transferred head would be biased in a direction
no diagnostic attributes to the handoff.

A single-stage run is legal and is how an ablation isolates either half: stage
0 alone measures what the demonstrations contain, and stage 1 alone measures
what online training reaches without them. The pair is the claim; the singles
are the evidence for it.

### Pretraining

The `Marwil` stage reads its demonstrations by sweeping each admitted record's
`final_placement` through the environment, which yields observation, action,
reward, and termination per step at the cost of one construction pass and no
search. A demonstration sample carries **no behaviour-policy
log-probability**, and none is required: the algorithm applies no importance
correction, and a synthesized placement sweep has no behaviour policy that
could report one. A pipeline that fabricates a log-probability to satisfy a
schema is reporting a number about a policy that never acted.

`beta` is stated against what the demonstration set contains. When every
demonstration is a completed annealing placement, returns vary little across
the set, the advantage weighting flattens toward one, and the stage is
behaviour cloning whatever `beta` nominally says. A configuration that wants
weighting to do work needs demonstrations of visibly different quality — a set
generated under several annealing policies, or one that admits partial-quality
placements — and this document requires the choice to be deliberate rather than
inherited.

### Online Training

The `Ppo` stage is the core's binding, over the environment views the stages
declare. Its return range is this environment's:

```text
step_bound * max(energy_width + repair_step_reward_code,
                 failed_transition_reward_code)
  + cleanup_displacement_reward_code * max_displacement
  + incomplete_closure_reward_code
```

`energy_width` is the quantization width of the selected search energy.
`repair_step_reward_code` participates only under `ConstructThenRepair` and
`cleanup_displacement_reward_code` only under `ConstructThenAnneal`, and the
maximum is taken over whichever arm each bound stage declares.

The repair charge is added to the energy width rather than maximized against
it, and that is the difference from the design-space arithmetic worth naming. A
rejected step there carries a code and no energy delta, because nothing was
committed, so the two are alternatives and a maximum is exact. A committed
repair step here carries both: the transition's own energy delta *and* the
per-repair charge. Maximizing them would understate every repair step by
whichever term is smaller, and a `value_bound` checked against that
understatement clips the returns of exactly the arm it was computed for.

`max_displacement` is the one term that depends on the problem rather than the
configuration. A cleanup moves each realization at most `r` hops, where
`Bounded(r)` is the arm's `cleanup_annealing_policy.realization_move_radius`,
and [Annealing And Replay](spec-pnr.md#annealing-and-replay) anchors
displacement at the run-start occurrence, so displacement is bounded by the
realization count times `r` plus the moved count, itself bounded by the
realization count. Only a `ConstructThenAnneal` arm has a cleanup, and that arm
is required to carry `Bounded`, so the term always has a radius to read and
contributes nothing under the other arm — the same condition
`cleanup_displacement_reward_code` already carries.

Both quantities are available without a frozen model: the realization count is
how many Compute and Memory Realizations the TechMapping the problem binds
declares, and `r` is a configuration field. That matters because freezing every
pool member to obtain an integer already present in `T` would make adoption pay
this environment's dominant cost for nothing. The bound is computed per problem
and the run's requirement is the
maximum over the pool. That a term is instance-dependent is exactly why
[The Reachable Return Range](spec-ml-core-training.md#the-reachable-return-range)
leaves the arithmetic to a training document rather than fixing a formula.

## Statistics

The production rule, the objective-dimension record, the outcome accounting,
the cost statistics, and chunk invariance are the core's. This environment
names its own vocabulary and adds what only it has.

Its non-advancing step class is `failed`, so the core's accounting identity
reads `steps == advanced + failed + elective_stops`, and the failure rate is
reported per member of `PnrTransitionFailureReason`. Keeping `IntrinsicInvalid`
apart from `WorkLimit` matters here more than a blended rate would suggest: one
says the policy proposed something that cannot produce a legal candidate and
the other says the router ran out of budget, and a rising second with a flat
first is a configuration problem that a blended rate would present as a policy
problem.

Its action partition is the `SpatialMappingAction` kind, so action frequencies
and their success rates are keyed by kind. Under construction only one kind is
live, so the partition is informative exactly in the repair phase — which is
where the question "is the policy doing anything but rebinding" is worth
asking.

Four statistics are this environment's own:

```text
placed_fraction_at_end            placed realizations over realization count
closure_rate                      episodes whose final candidate closed
cleanup_displacement              ConstructThenAnneal only
repair_steps_taken                ConstructThenRepair only
```

`cleanup_displacement` is the arm-A quantity the reward already prices,
reported directly because it is the measure of what construction left undone. A
run whose energy improves while its displacement also rises is not improving
construction; it is leaning harder on the annealer, and only the pair
distinguishes the two.

```text
PnrBreakdownAxis =
    Stage              // 0, mandatory
  | TerminalReason     // 1, mandatory
  | Problem            // 2
  | EpisodeArm         // 3
  | ActionKind         // 4
```

`Problem` is the axis that answers whether a run improved everywhere or only on
the instances it saw most, and it is the expensive one: it multiplies every
statistic by the pool size, so it is opt-in like every other axis.

## Test Protocol

`ResolvedMlPnrTestSetConfigView` is the core's `ResolvedTestSetConfigView`
under this descriptor, and this document supplies only its case payload:

```text
PnrTestCase {
  instance: SpatialPnrProblemBinding
  case_seed: u64
}
```

A `SpatialPnrProblemBinding` names a Canonical Dataflow Program, a TechMapping,
a Fabric, and a MappingConstraintSet, so a test set is a grid over exactly the
two things a run wants to hold fixed and vary: the workloads, which vary `D`,
and the hardware versions, which vary `F`. Both are named rather than drawn, so
the set is the same set at every evaluation.

### Results And Comparison

```text
PnrTestCaseResult {
  outcome: TestCaseOutcome
  final_energy_code: uint64
}
```

`final_energy_code` is the only quantity a case adds. Everything else worth
reading about the episode — its length, its terminal reason, its closure, its
displacement — is already inside the `Completed` arm of `outcome`, which
carries this environment's episode statistics catalog. Repeating any of it here
would be two sources for one fact, multiplied by every case and every
evaluation the run retains.

The run retains one series per case, appending a result at each evaluation.
What makes a series a measurement rather than a log is what it is compared
against, and here that is the run's own history: an evaluation reports each
case's result together with its difference from the same case at the previous
evaluation and at the most recent stage boundary.

The stage-boundary reading is the one that carries the argument. It is what the
pretrained policy achieved before online training touched it, so the difference
against it is exactly the question the two-stage design asks — whether PPO
improved on what the demonstrations delivered. The previous-evaluation
difference is the ordinary progress signal.

Improvement over iterations is therefore a trend in a per-case series against a
fixed reference, not a movement in an aggregate. An aggregate over cases hides
the case that regressed, and a policy that improves its mean while losing its
hardest problems is the failure this reporting exists to catch.

Two comparability rules, because a series only means something if its terms
measure the same thing. Results compare only across evaluations sharing both a
test-set digest and an environment view digest. And a stage advance that
changes either starts a new series rather than extending the old one, with the
new series' first reading becoming its own reference.

There is deliberately no annealer baseline in these numbers. Comparing a policy
to the annealer that produced its demonstrations is a real question, but it is
a question about a finished checkpoint rather than about a run in progress, and
answering it during training would require a full annealing invocation per case
per evaluation — which is the dominant cost in this environment and would make
evaluation cost more than training. A finished comparison runs offline against
the results `loom-pnr-train` recorded.

## Harnesses

```text
loom-pnr-demos   generate, extend, and verify a demonstration view
loom-pnr-train   run training against a run view
loom-pnr-test    run a test set against a checkpoint
```

Their outputs are removable projections, on the terms
[Harnesses](spec-ml-core-training.md#harnesses) states.

`loom-pnr-demos` verifies an existing set without regenerating it, which
`final_placement` is what makes possible: every named problem resolves, every
admitted record's sweep reproduces its recorded energy and closure, and every
record satisfies the three admission conditions against the environment views
the run binds. No annealing runs. That last check is what catches a
demonstration set that was valid for the run that produced it and is not valid
for the run about to use it.

## Conformance Anchors

Stable tests cover a demonstration whose recorded energy is the replay's rather
than the annealing run's, and the two differing when `demonstration_order`
changes; a demonstration containing exactly one binding per realization, in
`demonstration_order`, with no realization bound twice and none left unplaced;
every demonstration action being unmasked at the step it is taken; a
demonstration whose `action_count` exceeds a bound stage's `step_bound` being
refused at adoption; a demonstration set regenerated from the same seed,
policy, order, and Place and Route view digest reproducing the same admitted
set; a record's sweep and its admission checks running without any annealing
invocation; the scripted
annealer selecting every Action through the owner's proposal and acceptance
protocols and consuming no host entropy; a `Marwil` stage requiring no
behaviour-policy log-probability and rejecting a batch that fabricates one; the
two stages being rejected at adoption when their `gamma` differs; a stage-1
evaluation at the boundary reporting the parameters stage 0 produced; a
single-stage run of either algorithm being legal; the return range accounting
for `step_bound` occurrences of the most expensive per-step charge rather than
one, including its arm-conditional terms, a committed repair step counting its
energy delta and its repair charge together rather than the larger of the two,
and its `max_displacement` term being the maximum over the problem pool rather
than a configured constant and being computed without freezing any problem; a
`value_bound` below that maximum being refused;
the failure rate being reported per `PnrTransitionFailureReason` with
`IntrinsicInvalid` and `WorkLimit` never blended; action frequencies keyed by
`SpatialMappingAction` kind; `cleanup_displacement` reported only under
`ConstructThenAnneal` and `repair_steps_taken` only under
`ConstructThenRepair`; a test case naming a problem that appears in no training
stage's environment view; a result series comparing only across evaluations
sharing a test-set digest and an environment view digest, and a stage advance
that changes either starting a new series; each evaluation reporting a per-case
difference against both the previous evaluation and the most recent stage
boundary; a case result carrying no quantity its `outcome` already carries; and
`loom-pnr-demos` refusing a demonstration set that is valid for its generating
run and invalid for the run binding it.

Tests do not pin `beta`, annealing policy values, demonstration counts, the
chosen `demonstration_order`, plateau tolerances, stage lengths, the contents
of any particular demonstration set or test set, achieved energies, closure
rates, displacement values, wall-time numbers, or diagnostic text.
