# ML DSE Training

This document defines how the search policy of
[ML DSE Model Architecture](spec-ml-dse-model-architecture.md) is fitted
against the environment of [ML DSE Environment](spec-ml-dse-environment.md). It
owns the split of a run's settings into separate resolved configurations, the
binding to an existing PPO implementation, the per-episode statistics a run
reports, the fixed test set and the deterministic protocol that evaluates it,
and the offline corpus of training architectures and synthetic training
workloads.

## Ownership

Every fact this document depends on resolves to one exact owner:

- [ML DSE Environment](spec-ml-dse-environment.md) owns the episode protocol,
  the action space, the observation, the reward's exact `(sign, magnitude)`
  pair, the step and rejection outcome algebra, the terminal reasons, the
  `Trajectory`, the Python layering, and `ResolvedMlDseEnvironmentConfigView`;
- [ML DSE Model Architecture](spec-ml-dse-model-architecture.md) owns the
  module boundary, the encoder, the heads, the action distribution, the mask
  composition and its fallback, and the checkpoint's parameter and
  configuration match;
- [ML Training Core](spec-ml-core-training.md) owns the nonsemantic boundary,
  the configuration split and inventory identity, the algorithm binding
  surface, the stage sequence and its handoff, exact-rational hyperparameters,
  schedules, the reward adapter, topology, the stage invariance rule, the
  statistics production rule, the test protocol, checkpoints and run identity,
  and reproduction;
- [Objectives and Quality Gates](spec-dse-feedback.md#objectives-and-quality-gates)
  owns `ObjectiveDimension`, `ExactAffineQuantization`, directed codes,
  `ObjectiveVector`, `WeightedLevel`, and `SearchEnergyRef`;
- [Model Parameters and Training](spec-dse-feedback.md#model-parameters-and-training)
  owns the training of registered prediction contracts, which is a different
  activity from this one and shares no record with it;
- [Candidate Generators](spec-dse-feedback.md#candidate-generators) owns the
  `fabric_template` generator this document's architecture corpus is produced
  by;
- [Evaluation Metrics](spec-evaluation-metrics.md#metric-registry) owns every
  `MetricKind` and `ExactRatio`, and
  [FPA Evaluation](spec-fpa-estimation.md#metrics) owns which physical metrics
  are registered and what may produce them;
- [Component Views](spec-config-ssot.md#component-views) owns view framing,
  canonical bytes, and `component_view_digest`, and
  [Cache Dependencies](spec-config-ssot.md#cache-dependencies) owns the cache
  family the corpus prepopulates;
- [Search Policy And Determinism](spec-pnr.md#search-policy-and-determinism)
  owns the seeded PRNG protocol;
- [Source Integration](spec-compiler-part-1-source.md) and the compiler
  pipeline own how any program, synthetic or real, becomes a Canonical Dataflow
  Program;
- [Corpus](spec-loom-stack.md#corpus), [LoomBench](spec-loombench.md), and
  [Real Application Portfolio](spec-application-portfolio.md) own the real
  program inventories, which this document's synthetic corpus is never part of;
  and
- [External Dependency Pinning](spec-loom-stack.md#external-dependency-pinning)
  owns the Ray fork revision and its patch stack.

This document owns only this environment's return-range arithmetic and its
discounting consequence, the DSE statistics catalog and breakdown axes, the
test case payload and the granularity its isolation is compared at, the corpus
view and its offline generator, and its harnesses.

## Nonsemantic Boundary

Under
[Nonsemantic Boundary](spec-ml-core-training.md#nonsemantic-boundary), the
owner-owned path by which a design this run discovers becomes a real candidate
is the one
[Trajectory Retention And Replay](spec-ml-dse-environment.md#trajectory-retention-and-replay)
defines.

The corpus this document generates is the one place training touches durable
state, and it touches it only through existing owners. A training architecture
is an ordinary finalized Fabric Artifact and a training workload is an ordinary
Canonical Dataflow Program Artifact, each produced by the owner that already
produces them. The corpus view names those roots; it does not define a second
Artifact kind, a second program authority, or a store of its own.

## Configuration Split

This run's views, under the framing
[Configuration Split](spec-ml-core-training.md#configuration-split) owns:

```text
loom.ml_dse_environment.config.1.0  ResolvedMlDseEnvironmentConfigView
loom.ml_dse_model.config.1.0        ResolvedMlDseModelConfigView
loom.ml_dse_training.config.1.0     ResolvedMlDseTrainingConfigView
loom.ml_dse_corpus.config.1.0       ResolvedMlDseCorpusConfigView
loom.ml_dse_test_set.config.1.0     ResolvedMlDseTestSetConfigView
loom.ml_dse_run.config.1.0          ResolvedMlDseRunConfigView
```

```text
ResolvedMlDseRunConfigView {
  training_config_digest: ComponentViewDigest
  model_config_digest: ComponentViewDigest
  corpus_config_digest: ComponentViewDigest
  test_set_config_digest: ComponentViewDigest
}
```

`ResolvedMlDseTrainingConfigView` is the core's `ResolvedTrainingConfigView`
under this descriptor, with no field added. Every stage binds `Ppo`; a run that
wants a curriculum spends more than one stage on it, and one that does not is
the single-stage case.

The cached asset the core's digest-independence rule names is, here, the
Mapping cache family; it takes a few dozen episodes to become useful.

Adoption validates each view independently, then validates the run view's
cross-view conditions: every `seed_roots` and `workload_pool` member of every
bound training environment view appears in the corpus view; the test set's own
environment view and its cases are admitted under
[Test Protocol](#test-protocol) and are disjoint from the corpus under
[Isolation](#isolation); the
model view's embedding table sizes match the owner catalog cardinalities the
environment view's bound domains reach; and the model view's value bound covers
the reachable return range below, maximized over every bound environment view.

## Training Configuration

The training view, the algorithm bindings, the stage sequence and its handoff,
exact-rational hyperparameters, schedules, and the reward adapter's single
scale are owned by
[Training Configuration](spec-ml-core-training.md#training-configuration). This
environment supplies two things of its own.

### The Reachable Return Range

This environment's declaration under the core's obligation. A transition's
reward magnitude is bounded by the quantization width of the search energy's
dimensions, and every step contributes one term: an advancing step at most that
width, a rejected step `rejection_reward_code`, and the one step that may be an
elective stop `stop_reward_code`. An episode carries at most `step_bound`
steps, so the undiscounted return is bounded by

```text
step_bound * max(energy_width, rejection_reward_code) + stop_reward_code
```

all times the adapter's scale. The core's every-step rule is what makes the
rejection code multiply `step_bound` here rather than appear once, and it binds
because a policy early in training rejects far more often than it advances.

### Path Dependence And Discounting

Discounting at one has an exact meaning here worth stating. When
`warm_start_from_parent` is clear, the environment's energy is
path-independent, so an undiscounted return telescopes to the total improvement
from the seed to the final design and every intermediate reward cancels. A run
that wants that property sets both, and a run that discounts is deliberately
preferring earlier improvement to later. With warm starting set the telescoping
does not hold, because the energy a state reports depends on how it was
reached, and this document states that rather than letting a configuration
assume otherwise.

## Algorithm Binding And Topology

The algorithm binding surface, what Loom supplies to the trainer, the
instrumentation-only rule for a `Learner` subclass, rollout and learner
topology, and the stage sequence with its invariance rule are owned by
[ML Training Core](spec-ml-core-training.md); the fields a stage may vary are
declared by
[Curriculum-Neutral Fields](spec-ml-dse-environment.md#curriculum-neutral-fields).

One of those lands on this environment's own values. Its step is dominated by
per-workload Mapping probes, as
[Benchmarking](spec-ml-dse-environment.md#benchmarking) reports, so sampling is
normally the binding cost and the topology is chosen by moving runner count
against that split.

## Episode Statistics

The production rule, `DimensionEpisodeStatistic`, the outcome accounting, the
cost statistics, `LoggingPolicy`, and chunk invariance are owned by
[Episode Statistics](spec-ml-core-training.md#episode-statistics). This
environment names its own vocabulary and adds what only it has.

Its non-advancing step class is `rejected`, so the core's accounting identity
reads `steps == advanced + rejected + elective_stops`, and the rejection rate
is reported per member of `RejectionReason` and per `ExplorationDomainKind`. A
single aggregate rate conflates causes that call for opposite responses:
`AlreadyVisited` is search behavior, `WorkloadMappingNotClosed` is a
probe-budget problem, `WorkloadProvenInfeasible` is a real proof about the
design, and `EnumerationBoundExceeded` is a configuration error. Its action
partition is the `(ExplorationDomainKind, decision kind)` pair, because a
decision kind ordinal restarts per domain and a shared key would tie unrelated
decisions together.

Two statistics are this environment's own. `visited_revisit_rate` is the share
of rejections that were `AlreadyVisited`, which is what says whether the policy
is circling. And a dimension of the selected closure that sits outside the
search-energy level appears in these statistics and contributes nothing to the
reward, which is how a run watches a metric it does not optimize; a dimension
absent from the closure produces an absent statistic rather than a zero.

```text
DseBreakdownAxis =
    Stage             // 0, mandatory
  | TerminalReason    // 1, mandatory
  | SeedRoot          // 2, per architecture
  | Workload          // 3, per Canonical Dataflow Program
  | Domain            // 4, per ExplorationDomainKind
  | RejectionReason   // 5
```

`SeedRoot` and `Workload` are the expensive members: a corpus with a few
hundred architectures and a few thousand workloads turns every statistic into
hundreds of thousands of series, which is why axes are opt-in.

### Occupancy Statistics

Utilization is reported when the environment's `include_resource_states` is
set, and is absent otherwise. Per episode, at entry and at the final state:

```text
resource_occupancy[FabricResourceStateRef]    usage / capacity, as ExactRatio
placed_occurrence_fraction                    nodes with PlacedDegree > 0
mean_residual_violation_count
```

Occupancy is the ratio of the usage and capacity columns the observation
already carries per resource state, aggregated over the episode's Fabric nodes.
It is the quantity a hardware decision actually trades — headroom — and it is
the direct check on whether the search is removing resources the workload was
not using or resources it was. A run whose area improves while occupancy stays
flat is removing slack; one whose area improves while occupancy climbs toward
one is approaching the feasibility gate, and its rejection rate will confirm
it.

`placed_occurrence_fraction` is the Mapping's own view of the same question,
from the probe result's `PlacedDegree`, and it is reported alongside because
the two disagree in an informative way: a fabric can be resource-saturated
while most occurrences go unplaced, and that is a routing problem rather than a
capacity one.

## Test Protocol

The test-set container, its determinism, its execution and aggregation, and
what a test score is not are owned by
[Test Protocol](spec-ml-core-training.md#test-protocol). This environment
supplies its case payload and its isolation rule.

```text
DseTestCase {
  instance: EpisodeStartOverride
  case_seed: u64
}
```

The instance is the environment's own override payload rather than a re-spelled
copy of its fields, so a field added there reaches a test case without an edit
here.

### Isolation

The disjointness the core requires is compared here by exact Artifact root
identity: adoption rejects a run whose test-set seed roots intersect the
corpus's architectures, or whose test-set Canonical Dataflow roots intersect
the corpus's Canonical Dataflow roots.

The workload comparison is on the Dataflow root alone, not on the
`WorkloadBinding` pair. A binding is a Dataflow root together with a
TechMapping, and the same program bound to a different TechMapping is a
different pair but the same program; comparing pairs would let a training
workload reappear in the test set under a regenerated TechMapping and pass as
disjoint, which is precisely the leakage this rule exists to prevent. The
TechMapping is a derived artifact of the architecture it was built against, so
it carries no independent identity worth isolating.

Root identity is the right granularity for the rest because it is what the
environment consumes. A test workload generated from the same synthetic
parameters as a training workload but with a different seed is a different
program and a different root, and is admissible; the same program under a
different file name is the same root, and is not. Checking names instead of
roots would admit exactly the duplicate the rule exists to exclude.

A test set may also draw its cases from the real corpora — LoomBench cases or
portfolio operators — and this is the more informative choice, because the
synthetic corpus is calibrated to resemble them and a test on real programs
measures whether that calibration transferred. Those corpora are owned by
[LoomBench](spec-loombench.md) and
[Real Application Portfolio](spec-application-portfolio.md); the test set
references their Artifacts and defines no membership rule of its own.

## Training Corpus

### Two Halves, One Pairing

```text
ResolvedMlDseCorpusConfigView {
  architectures: canonical nonempty set<ArtifactRootReference>
  workloads: canonical nonempty set<WorkloadBinding>
  pairings: canonical nonempty set<FeasibilityPairing>
  generation: CorpusGenerationRecord
  calibration: CalibrationRecord
}

FeasibilityPairing {
  architecture: ArtifactRootReference
  workload: WorkloadBinding
}
```

The corpus is admitted as a bipartite pairing rather than as two independent
lists. A pairing asserts that the offline generator obtained a closed Mapping
for that workload on that architecture at the episode subject's closure level,
under the probe view the run will use. Adoption requires every architecture to
appear in at least one pairing and every workload to appear in at least one.

Without that requirement a corpus can be individually valid and jointly
useless. The environment's `reset` is a precondition, not a best effort: a seed
that cannot map every selected workload is retried and, past
`start_retry_bound`, reports `RetryBudgetExhausted`. A corpus whose
architectures and workloads were generated independently produces exactly that
failure at a rate nobody predicted, and it produces it after paying cold PnR
for each attempt. Establishing the pairing offline, once, converts a per-reset
gamble into an adoption check.

A pairing is not a promise that every episode start succeeds. The environment
selects several workloads per episode and a pairing is per workload, so a draw
can still fail on a combination. The pairing bounds that failure to
combinations rather than admitting workloads no architecture can run at all.

### Architecture Corpus

An architecture is an ordinary finalized `fabric.module` or `fabric.system`
Artifact, matching the episode subject. It is produced by the `fabric_template`
generator of kind 12 through the ordinary ADG Builder and finalization path,
and it is published. The environment's episode start already admits a
`fabric_template` output as a seed, so this introduces nothing.

Kind 12 is where the architecture corpus lives precisely because the
environment excludes it as an exploration domain: a template expansion selects
an episode seed rather than advancing an episode. The same generator therefore
serves both roles without either one reaching into the other.

The corpus declares its coverage over the axes the template generator
parameterizes — array extent, processing-element composition and operation mix,
memory capacity and count, boundary inventory width, switch topology and
connectivity, FIFO depth, and for a `System` subject the core count and
transport topology. Coverage is reported per axis bucket rather than as a
total, because a corpus of a thousand architectures that all differ in FIFO
depth teaches a policy one axis.

The extent range is not an aesthetic choice. The encoder is a graph transformer
whose cost scales with arc count and whose inference latency
[Forward Benchmarking Obligations](spec-ml-core-model.md#forward-benchmarking-obligations)
reports against node count, and the curriculum is expected to move designs
across an order of magnitude in node count. A corpus spanning one decade of
node count is what makes that curriculum expressible; one spanning a factor of
two makes a stage sequence meaningless.

### Synthetic Workload Corpus

A training workload is a synthetic program. Its operations compute nothing
meaningful: it has no reference output, no oracle, and no correctness claim,
and it exists to have the structure of a program rather than the behavior of
one.

A synthetic program is nonetheless an ordinary program. The generator emits
ordinary compiler input, and the ordinary frontend compiles it into a Canonical
Dataflow Program Artifact with its TechMapping, through exactly the path
[Source Integration](spec-compiler-part-1-source.md) and the compiler pipeline
define. There is no synthetic-only path into the environment, no hand-authored
Dataflow, and no generator-owned Artifact kind. A synthetic program the
compiler rejects is a generator defect rather than a special case to
accommodate — and the compiler rejecting it is the cheapest possible discovery
that the generator is producing something no real program could be.

The corresponding exclusion is exact. A synthetic workload never enters
`SourceTranslationUnitInventory` or `OperatorWorkloadInventory`, never becomes
a LoomBench case, and never joins the application portfolio. Those inventories
require a reference oracle and a semantic claim, which a program of meaningless
operations cannot supply. A synthetic workload is admissible for training and
for nothing else; it may not produce Evidence, satisfy a conformance gate, or
appear in a correctness result.

### Structural Axes And Calibration

The corpus is meant to resemble programs found in the wild, and what it
resembles is their structure, not their computation. The generator is
parameterized over a closed set of structural axes:

```text
StructuralAxis =
    OperationMix              // distribution over the target's operation catalog
  | GraphDepth                // longest dependence chain
  | GraphWidth                // available parallelism at a level
  | FanoutDistribution        // out-degree of a produced value
  | ReconvergenceRate         // values consumed by more than one path
  | RecurrenceDistance        // loop-carried dependence distance
  | LoopNestDepth
  | TripCountMagnitude
  | StreamCount               // concurrent memory streams
  | StrideRegularity          // affine versus irregular access
  | ReuseDistance
  | PredicationRate
  | DataDependentBranchRate
  | VectorWidthDistribution
  | LiveValuePressure         // concurrent live values at a program point
```

Each axis has a declared target distribution, and a generated program is a draw
from their joint parameterization. The axes are the ones that change what a
Mapping and a fabric decision must contend with: depth against width sets what
parallelism a fabric can exploit, recurrence distance sets what a temporal
element must hold, stream count and stride regularity set what the memory
frontier must supply, and predication rate sets how much control the fabric
must absorb. An axis that does not change what the search decides does not
belong here, however faithfully it describes a real program.

Calibration is a comparison, not a claim:

```text
CalibrationRecord {
  reference_corpus: canonical nonempty set<ArtifactRootReference>
  per_axis_tolerance: total table<StructuralAxis, ExactRatio>
}
```

The reference distribution is measured from the real corpora — the LoomBench
cases and portfolio operators the reference set names — by the same axis
extractor that measures a generated program, and the corpus is admitted when
every axis matches within its declared tolerance. The reference is derived from
the Artifacts `reference_corpus` names rather than written down, so it moves
when the real corpus moves and cannot become a stale table describing programs
nobody runs anymore. Deriving it is not the same as re-measuring it: a root is
immutable and content-addressed, so one root's axis vector may be retained
against that root indefinitely, and what each adoption recomputes is the
aggregate over whichever roots the set now names.

A corpus that has never been compared is not calibrated, and this is worth
stating because the failure is invisible. A generator with plausible parameters
produces programs that look reasonable to a reader and can still be
systematically shallower, narrower, or more regular than anything real, and a
policy trained on them learns a design space that does not exist. The
comparison is what converts "we chose sensible parameters" into a checkable
statement.

Matching every axis marginally is not matching the joint distribution, and this
document does not claim otherwise. Marginal agreement per axis is the admission
test because it is checkable and because it catches the large errors; a
generated corpus remains a model of real programs, and the test set drawn from
real programs is what measures whether the model transferred.

### Offline Generation

Corpus generation is a batch job that runs before training and produces three
things: the published architecture and workload Artifacts, the pairings, and a
prepopulated Mapping cache.

```text
CorpusGenerationRecord {
  generator_seed: u64
  prng_protocol: Sha256SeededXoshiro256StarStar_1_0
  architecture_axis_targets
  workload_axis_targets
  target_counts_per_bucket
  probe_binding: CorpusProbeBinding
}

CorpusProbeBinding {
  episode_subject:
      SpatialModule
    | System { system_pnr_config_view_digest: ComponentViewDigest }
  spatial_pnr_config_view_digest: ComponentViewDigest
  tech_mapping_config_view_digest: ComponentViewDigest
}
```

The unit of work is one `(architecture, workload)` pair, which is also the unit
of the Mapping cache. The job establishes a pairing by obtaining a closed
Mapping, and that Mapping is written into the ordinary cache family
[Cache Dependencies](spec-config-ssot.md#cache-dependencies) owns, under the
key the environment's episode start already uses: the seed root, that one
workload with its TechMapping, and the probe view. No second store, no second
key, and no corpus-local mapping format.

This is where the cost of training moves offline. The environment's own
analysis is that step 5 of `reset` dominates start cost and that per-workload
keying is what lets the cache reach a useful hit rate; running that step for
every pairing ahead of time means a training run's resets hit a warm cache from
the first episode rather than after a few dozen. It also means the pairing and
the cache entry are produced by the same work, so a corpus cannot claim a
pairing it never computed.

The job is parallel over pairs, resumable, and content-addressed. A pair whose
Mapping is already cached and whose pairing is already recorded is skipped, so
an interrupted run continues rather than restarting, and two jobs over the same
corpus converge on the same result. Generation order is by bucket so that an
interrupted job leaves coverage spread across the axis buckets rather than
complete on the first bucket and empty on the rest.

A pair whose Mapping does not close is recorded as a non-pairing and is not
retried indefinitely; a workload that closes on no architecture in the corpus
is reported and excluded, which is the signal that the workload generator has
drifted outside what the architecture generator produces.

`CorpusProbeBinding` binds the cache entries to the complete set of views that
produced them, because that is what the cache key is. The environment's key
covers the seed root, the workload with its TechMapping, and *every* probe view
the episode ran under: the Spatial PnR view and the TechMapping view always,
and the System PnR view its subject arm carries for a `System` episode. A
record naming only the Spatial view could not distinguish two corpora generated
under different System contracts, so a `System` run would take a hit on an
entry a miss would not have produced — which is exactly the case per-view
keying exists to prevent.

Adoption compares the complete binding against every bound environment view's
`probe_policy` and subject arm, and reports a mismatch rather than letting a
run silently pay cold PnR for every reset while a full cache sits unused.

### Corpus Identity

The inventory this run carries is the corpus, and its identity is the exact
`ArtifactRootReference`, on the terms
[Inventory Identity](spec-ml-core-training.md#inventory-identity) sets. An
enumerated corpus is therefore a value: extending it produces a new digest, and
every run that used the old one remains exactly reproducible.

What discharges the regeneration half here is that the same `generator_seed`,
axis targets, and compiler configuration produce byte-identical Artifacts and
therefore identical roots, so a corpus can be reconstructed rather than
archived. The seeded PRNG protocol and its prohibition on host entropy are the
PnR owner's; this generator adds only its own domain separator and stream
purposes.

## Checkpoints, Reproduction, And Harnesses

Checkpoints and run identity, the two reproduction claims and the one this
document declines, and the obligations every harness satisfies are owned by
[ML Training Core](spec-ml-core-training.md). This run ships:

```text
loom-dse-corpus   generate, extend, verify, and calibrate a corpus view
loom-dse-train    run training against a run view
loom-dse-test     run a test set against a checkpoint
```

`loom-dse-corpus` verifies an existing corpus without regenerating it: every
named root resolves, every pairing's Mapping is cached under the recorded probe
binding, coverage per bucket meets the declared targets, and every calibration
axis is within tolerance.

## Conformance Anchors

Stable tests cover the run view rejecting a corpus that omits a bound training
environment view's seed root or pool workload, a test-set environment view
whose inventory intersects the corpus by root identity, a test case naming
inventory outside its own environment view, a test-set environment view
differing from the training views in any field
[Curriculum-Neutral Fields](spec-ml-dse-environment.md#curriculum-neutral-fields)
declares shape-fixing, including a reward code inside `EpisodePolicy`, a model
view
whose embedding table sizes do not match the reachable catalog cardinalities,
and a `value_bound` below the return range computed from the declared
quantization bounds, `step_bound`, `rejection_reward_code`, and
`stop_reward_code`; a run whose test set is admissible being adopted at all, so
that the corpus, isolation, and override rules are jointly satisfiable rather
than only individually stated; a `value_bound` sized for one rejection being
rejected against an episode of `step_bound` rejections; a hyperparameter edit
changing the training digest and leaving every cached Mapping valid; a stage
advance differing only in fields
[Curriculum-Neutral Fields](spec-ml-dse-environment.md#curriculum-neutral-fields)
declares neutral being accepted and one differing in `enabled_domains`,
`selected_objective_closure`, or `enumeration_bound` being rejected; a stage
advance retaining the Mapping cache entries earlier stages produced;
`improvement` being positive for a `Minimize` and for a `Maximize` dimension
that each improved; a closure dimension outside the
search-energy level appearing in the statistics and contributing nothing to the
reward; a dimension absent from the closure producing an absent statistic
rather than zero; a step rejected with `ObjectiveUnavailable` producing no
numeric estimate; rejection rates being reported per `RejectionReason` and per
`ExplorationDomainKind`; action frequencies being keyed by
`(domain, decision kind)` pairs and no statistic key being built from a file
name; occupancy statistics being absent when `include_resource_states` is
clear; corpus adoption rejecting an architecture or workload with no pairing; a
corpus pairing being backed by a cached Mapping under the recorded probe
binding; a training run whose Spatial PnR, TechMapping, or System PnR view
differs from the corpus's reporting the mismatch at adoption, and a `System`
-subject run being refused a corpus whose binding carries no System PnR digest;
a synthetic workload reaching the environment only as a compiled Canonical
Dataflow Program Artifact and being refused admission to every real-program
inventory; a corpus regenerated from the same seed and axis targets producing
identical Artifact roots; a corpus resolved by enumeration rather than by
directory contents, so that adding a file changes no existing run's data; a
calibration comparison failing when a generated axis distribution exceeds its
declared tolerance against the measured reference; and an interrupted corpus
job resuming without recomputing a cached pairing.

Tests do not pin hyperparameter values, schedule breakpoints, layer or width
choices, topology counts, corpus size, axis target distributions, tolerance
values, the contents of any particular corpus or test set, wall-time numbers,
throughput, learning curves, achieved test scores, sink formats, or diagnostic
text.
