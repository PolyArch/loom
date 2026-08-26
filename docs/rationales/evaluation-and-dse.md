# Evaluation And DSE Rationale

Normative contracts are owned by
[Evaluation And DSE](../spec-dse-feedback.md),
[Evaluation Metrics](../spec-evaluation-metrics.md),
[External Tool Invocation](../spec-external-tool-invocation.md), and
[FPA Evaluation](../spec-fpa-estimation.md).

## Why Evaluation Is Shared Across The Stack

Compiler transformations, Mapping choices, and hardware candidates all need
feedback. Separate evaluator systems would duplicate metric definitions,
fidelity labels, cache rules, tool adapters, evidence, and objective weights.
Loom therefore uses one central Evaluation and DSE framework with domain-
specific candidate generators and models.

Evaluators report observations and findings. The central resolved policy alone
combines them into objectives, quality gates, Pareto sets, promotion, or final
selection. A mapper, simulator, or backend cannot embed a private score and
silently override policy.

Functional mismatch therefore uses one shared FindingKind rather than a
frontend-, simulator-, or backend-local status. Its occurrence is deliberately
zero-field: the exact Request already identifies both the semantic relation and
all compared inputs, while detailed first-difference paths are removable
diagnostics. Copying those paths into persistent Evidence would make every
comparison provider own a competing output-reference language without changing
the only stable gate fact, mismatch presence.

The same boundary is required before Mapping. Structured and Dataflow
optimization need both software-only estimates and exact hardware-aware
estimates, while hardware exploration also needs hardware-only models. Exact
case signatures therefore cover software-only, software-plus-Fabric,
hardware-only, and mapped subject closures without creating separate evaluator
frameworks for each compiler stage.

This forms a feedback graph rather than a one-way fidelity ladder. A mapped
result may provide Evidence that causes the central DSE plan to generate a new
Structured or Dataflow candidate. It never mutates the evaluated candidate or
rebinds the existing Mapping; immutable lineage keeps each observation tied to
the exact program and hardware that produced it.

## Why Request And Evidence Are Separate Artifacts

An Evaluation case asks a ground-truth question about exact immutable subjects,
conditions, queries, and model binding. Evidence records the outcome of one
attempt against that request. Keeping both artifacts makes caching, retries,
replicates, training, comparison, and replay explicit.

Evidence does not copy Mapping, program, hardware, or case facts. It references
the exact Request and stores only its normalized typed outputs and outcome.
Completed adverse results remain valid Evidence; unsupported capability,
execution failure, and cancellation remain distinct honest outcomes.

Raw reports, logs, waveforms, and tool databases are not normalized Evidence.
The earlier generic detailed-bundle field was removed because no exact owner,
schema, importer, or lineage existed. Such payloads remain attempt or scratch
material until a real owner is specified through a schema revision.

## Why Unavailable Typed Owners Differ From Missing Schemas

Forward compatibility depends on whether a semantic owner already exists, not
merely on whether its implementation is present. Execution activity already
has an exact Simulation-owned summary, ordinal, lineage, and validation
contract. Evaluation can therefore retain its closed typed binding variant
while authoring and import fail closed until that owner and provider are
available.

Raw detailed bundles had no family, root, content inventory, importer, or
same-Request lineage. Keeping an always-invalid generic reference would have
created an empty second authority, so the field was removed entirely. This is
why reserved-unavailable typed variants and absent fields are not one generic
provider-availability policy.

## Why Case Signature Is Model-Independent

Several model descriptors may answer the same question. If each descriptor
owned subject slots, equivalent cases would receive different identities and
could not compare or share cache entries. A shared case-signature registry owns
ordered subject roles once; each descriptor references one exact signature and
declares its supported queries and interaction domains.

Scope is query-local and typed. An anchor, dependency closure, and local target
must resolve against exact subjects. Generic paths, whole-subject wildcards,
and consumer-defined entity tuples were rejected because they erase owner type
and permit stale or foreign references.

A workload-running model may exist solely to publish one required typed output
Artifact, such as a `SimulationExecution`. Requiring a metric or finding in
that case would force a dummy normalized result that the model does not own.
An empty query set is therefore admitted only when the descriptor already
requires a nonempty Completed output binding. Models without such an output
still require a metric or finding, so the exception adds no generic no-op
evaluation path.

## Why Metrics, Findings, And Conditions Are Registries

A central typed registry gives each metric and finding one unit, value form,
scope admissibility, and interpretation. Tool-specific report fields remain in
the adapter or raw material; they do not expand the global registry without a
stable cross-model meaning.

Conditions such as process, voltage, temperature, clock requirement, workload,
and other exact case inputs use closed typed forms and canonical ordering.
Duplicate or conflicting assignments are invalid. A generic predicate DSL or
property map would move model semantics into runtime parsing.

Exact ratios use one normalized checked representation because clock and unit
relations must compare and hash identically across models. Silent overflow or
multiple equivalent fractions would undermine Request identity.

## Why Reference-Cycle Admission Is Shared

The metric scope form uniquely knows whether an observation requires an exact
case reference cycle, while the case-signature registry uniquely owns the
resolver and resulting basis. If descriptor admission and Request verification
each switched on metric names, they could accept different capability sets or
silently construct different bases.

Both paths therefore use the metric registry's one admission operation. Static
descriptor admission proves that the referenced case signature can supply the
required basis; exact Request admission invokes the same rule for its case.
Neither the Request nor an evaluator copies the requirement, resolver, or
basis into a second registry.

The exact case and cycle projections receive both Artifact and Blob stores.
This keeps validation with the Artifact-family owner when a root's semantic
closure includes logical payloads; treating the root bytes alone as sufficient
would make Evaluation a competing partial importer.

Terminal findings need two different codecs because they represent two
different facts. The Finding owner encodes the typed witness stored inside the
execution, while Simulation Artifacts encodes the output-relative reference
stored in Evidence. Sharing raw bytes between those roles would either copy the
witness into Evidence or make Evaluation interpret simulator-owned payloads.

## Why Fidelity Is Not One Ladder

Fidelity depends on the subject, workload, modeled phenomena, method, and
conditions. SCF analysis can be cheaper but more structurally informed than a
Dataflow-only model; DFG simulation is exact for software firing semantics but
hardware-unconstrained; CGRA simulation adds mapped microarchitecture;
system, RTL, EDA, FPGA, or measured providers answer different subsets at
different cost.

The descriptor therefore states capabilities and execution method. Confidence
is expressed per observation, not as a global `L1/L2/L3` label. Promotion
policy requests the evidence it needs rather than assuming every candidate
must traverse one fixed ladder.

The mapped RTL descriptor uses its own case signature because neither the
CGRA subject closure nor the System-simulation closure asks the same exact
question. HardwareImplementation owns the RTL state, Deployment owns the
occurrence-qualified Mapping and configuration closure, and a Spatial workload
owns the standalone launch. Binding those owners once is smaller than copying
configuration facts into Evaluation, smuggling HardwareImplementation through a
model-input slot, or adding an RTL discriminator to SimulationExecution.

## Why DSE Uses Typed Use-Def

The central plan is an SSA-like DAG of typed producer outputs and consumer
inputs. Candidate sets, Evidence, model parameters, and execution products use
the same use-def mechanism. A special CandidateSet reference or artifact-store
query for `latest` would create hidden ordering and mutable plan semantics.

Domain candidate generators own their candidate type, legal transformations,
deduplication, and local work. The central controller owns lineage, scheduling,
Evidence acquisition, promotion, Pareto selection, and deterministic replay.
It does not define one generic mutable candidate or action language.

Parameter training also needs exact generator identity. A single generic GBDT
descriptor with a caller-selected contract would make one output slot change
type at runtime and duplicate the parameter registry. The initial FPA and
System Runtime trainers therefore have separate stable generator kinds and
fixed output contracts while sharing an ordinary implementation library. A
new algorithm or payload contract registers another descriptor instead of
adding an algorithm switch to central DSE.

The Dataflow rewrite default is larger than the unrelated Structured scope
defaults because the normalized catalog includes reversible local rules and
charges attempts before identity deduplication. Even a small graph can
therefore traverse dozens of immutable identities and their inverse edges.
Using the same numeric default for all generators would be accidental coupling,
while suppressing inverse charges would make replay work depend on cache state.
The rewrite default remains only an operating point: the typed semantic-limit
outcome, rather than an implicit partial candidate set, handles larger domains.

When two domain owners expose different granularities, an explicitly typed
generator adapter composes them without broadening central semantics. The
root-complete TechMapping adapter is the representative case: the plan binds
whole Artifact candidates, Dataflow derives the complete local graph catalog,
and TechMapping consumes that ephemeral scope. A generic graph-cover value in
the plan would instead duplicate Dataflow-local identity and turn one mapping
policy into a second cross-domain authority.

Whole-program and System composition need two different derived covers. The
root-complete adapter intentionally asks whether one target Module can realize
the complete program graph catalog. Hierarchical SystemMapping instead assigns
each rooted graph launch independently and therefore needs ordinary
singleton-graph TechMappings from which it can select SpatialMappings for
different AccCore occurrences. The canonical-graph adapter derives those
singleton covers by visiting the same Dataflow-owned catalog. Keeping the two
adapters explicit preserves both questions without a persistent graph-scope
object, a generic scope language, or a System planner that inspects Dataflow
internals.

The root-complete Spatial adapter applies the same rule to a different
granularity mismatch. A TechMapping already seals its Dataflow and Fabric
owners, while the exact Spatial PnR API must also receive a real constraint
Artifact. The adapter recovers `D` from `T`, verifies the separately supplied
exact `F`, and asks the constraint owner to publish the empty `K`; it does not
add duplicate `D` or optional `K` plan slots. Constrained searches continue to
use the exact owner directly. This keeps the central plan generic while making
the common unconstrained `D -> T -> SpatialMapping` chain explicit and fully
replayable.

The root-complete System adapter completes that chain with the same ownership
rule. It keeps an explicit `D` input because the SpatialMapping set is legally
empty for InstructionCore-only execution and therefore cannot be used as a
surrogate Dataflow owner. It keeps one exact System `F` because imported Module
lineage and System transport belong to that root. The remaining inputs are
mechanical: Dataflow owns the root inventory, the constraint owner publishes
empty `K`, and the PnR owners derive the whole-domain partition and
hierarchical `H`. Exposing any of those as optional plan slots would create a
second way to say "root-complete"; a polymorphic Mapping-stage adapter would
instead erase the materially different Tech, Spatial, and System input
contracts. A dedicated thin descriptor is the smaller abstraction.

The thin adapter also reuses the complete PnR work-unit catalog. System search
does not become cheaper or semantically different merely because the adapter
derives its root closure and empty constraints. Reporting only initializer and
router work would hide fixed seed, Action, closure, and repair budgets, while
adding a System-only summary would create a competing accounting authority.
The ordinary PnR owner therefore supplies the same typed work ordinals to both
Mapping adapters; root-free execution is the all-zero instance of that
catalog.

The same ownership rule applies to hardware migration accounting. A child
System can preserve lower Mapping artifacts while reopening a distinct
execution, resource-use, or service cone; those facts must remain separate
from artifact counts. Keeping parent/preserved/reopened counts for thread and
graph bindings, resource uses, service realizations, and service legs in the
existing rebase ledger prevents a System-only summary from becoming a second
legality authority or from overstating schedule preservation.

The adapter retains a canonical prefix only when valid search work terminates
incompletely. A malformed later `T`, foreign owner tuple, or PnR invariant
failure invalidates the whole invocation rather than converting already
published store objects into selected outputs. This is the same distinction as
ArtifactStore publication: immutable objects can survive a failed attempt,
while only a successful or explicitly incomplete controller invocation owns a
formal output set.

### Why Candidate-Local Input Selection Is Narrowing

A plan value and an Evaluation case do not always have the same cardinality.
For example, one Mapping promotion may receive two Canonical Dataflow roots
`D0` and `D1` plus SpatialMappings `M0` and `M1`, where strict Mapping lineage
proves `M0 -> D0` and `M1 -> D1`. The CGRA case signature still requires one
exact Dataflow root for each Mapping candidate. Requiring the complete
invocation set in every case would create invalid two-program cases; adding a
second candidate-to-program table would duplicate Mapping lineage.

Loom instead permits the acquisition provider to narrow an already bound
typed input for one task. The controller proves that the selected value is a
canonical subset of the original slot, then the ordinary Request verifier
proves case cardinality and lineage. The provider can recover `D0` from `M0`,
but it cannot supply `D2`, move a Fabric into the Dataflow slot, or change the
template's workload. The selection is invocation-local and removable.

This keeps each fact with its existing owner: the plan owns available input
sets, Mapping owns `M -> D/F`, the acquisition policy owns mechanical task
resolution, and Evaluation owns the case and Evidence. It also avoids one
Promote node per runtime-generated candidate without introducing a dynamic
workflow or persistent association map.

## Why Invocation Output And Lineage Are Separate

A generator invocation may retain an input unchanged, return several Artifacts
through one slot, return Artifacts through several typed slots, or preserve
several valid paths to one deduplicated Artifact. Treating its complete output
as one lineage hyperedge would erase those distinctions. Conversely, repeating
the complete input closure and resolved configuration on every single-output
edge would make large candidate sets duplicate invocation facts.

The manifest therefore records the invocation's typed input and output
bindings once and nests ordinary single-child lineage edges beneath that
occurrence. Output bindings answer what the invocation returned; lineage edges
answer how those Artifacts were produced. Recursive generation may need several
atomic decisions before a candidate becomes returnable. Keeping the complete
rooted derivation DAG preserves those decisions without pretending that every
intermediate is a plan output, and avoids an owner-specific composite path
payload. Internal targets must be durably published, rooted, acyclic, and lead
to a returned sink, so lineage cannot become scratch state or an unrelated
history log. This is not duplicate authority because neither binding can be
derived from the other when an input is retained, an internal path is needed,
or several paths converge. Artifact dependency closure continues to own
semantic data dependence, while owner codecs preserve typed decisions without
creating a universal action language.

## Why Model Parameters Are Ordinary Domain Candidates

A training request Artifact would repeat facts already fixed by the typed
Generate inputs, generator binding, resolved plan, run key, and invocation
manifest: exact input Evidence, trainer identity, configuration, seed, and
occurrence provenance.
Including those facts in parameter identity would also make two training
occurrences with identical parameter semantics produce different Artifacts.

Training therefore uses the same `Generate` mechanism as every other domain
search. Its semantic output is an immutable parameter-bundle candidate. The
bundle identifies only one registry-owned parameter contract and canonical
parameter payload; the invocation manifest retains how that payload was
obtained. Equal payloads under one contract deduplicate, while a contract or
payload change creates a different identity. A promoted default may later be
checked into the codebase, but online state never silently changes an existing
model identity.

Checking in a promoted bundle publishes only the same contract-owned canonical
parameter object used by inference. It does not publish, redact, summarize, or
copy its source Evidence. Direct EDA Evidence and training provenance stay in
local stores, while the bundle schema structurally excludes those occurrence
facts. This preserves one model-weight authority and avoids a public/private
pair that would require synchronization.

One shared bundle family does not make the payload opaque. The exact model
parameter-contract registry owns the typed adopter, encoder, feature
projection, and pure inference kernel exactly once. Predictor and validator
descriptors reference that contract rather than reinterpreting its bytes.
Creating an Artifact family for every trained model would repeat the same
framing, storage, and publication machinery; accepting an untyped blob would
instead erase the semantic owner. The shared typed envelope plus one
registry-owned contract is the smaller complete abstraction between those
extremes.

## Why Calibration Uses Bundle And Evidence Subjects

Promotion already associates a candidate through one exact case-subject role.
Extending it to fill a descriptor-local model input would create a second
partial shape of `ResolvedModelBinding`, while the resulting bundle would be
unreachable from Evaluation scope because scope anchors must be case subjects.
It would also tie the payload contract to one consumer slot even though a
predictor and an independent validator must share it.

Calibration therefore evaluates one exact bundle together with one exact
nonempty collection of ground-truth Evidence as ordinary case subjects. Each
Evidence root retains its Request and complete original subject-role pairing,
so a case over `(D1,F1)` and another over `(D2,F2)` cannot be flattened into an
accidental `(D1,F2)` combination. Whole-case calibration metrics then mean
exactly "this bundle over this Evidence collection", and ordinary Promotion
recovers the bundle through the unchanged subject role. The bundle is the one
distinguished candidate role; the Evidence collection arrives through an
ordinary typed Promote input. Allowing that explicit noncandidate subject
binding is smaller than either freezing future Evidence identities into the
resolved configuration or inventing a calibration-specific plan node.

The validator calls the bundle contract's pure feature and inference functions
directly. The same contract names its prediction cases and exact ground-truth
models, owns their target-key relation, and consumes every result-affecting
operating condition as a typed feature. It does not call another evaluator,
hide a downstream Request, drop a condition, or duplicate predictor formulas.
A selected bundle later enters a
predictor's ordinary model input slot only after exact contract compatibility
succeeds. This keeps model consumption distinct from candidate evaluation
without a parameter-specific Promotion path or candidate-to-Evidence map.

Evidence identity alone is not a sufficient dataset boundary. Two tool runs
can produce different Evidence roots for the same circuit under different
seeds or attempts, allowing structural leakage while passing an identity-set
check. The parameter contract therefore owns a removable sample-group
projection appropriate to its feature semantics. Training, validation, and
held-out sets are pairwise disjoint by that key. All three are explicit trainer
inputs, so ordinary SSA use-def gives admission the complete sets before fitting
without an implicit scheduler edge. Only the Training slot is visible to the
fitting callback. Validation can rank candidates, while HeldOut is structurally
restricted to a terminal release gate. This adds neither a persistent dataset
Artifact nor a scheduler-dependent prefix.

Operating condition is intentionally not a partition escape hatch. Corner,
voltage, temperature, and activity change feature values and therefore the
prediction question, but observations of one underlying circuit remain in one
sample group. Otherwise the same implementation could leak from training into
held-out merely by changing a tool seed or corner.

Ordinary relative error divides by the observed value and therefore needs an
epsilon, infinity, or special rejection when a valid power observation is
zero. Symmetric relative error instead has one exact zero/zero result and a
finite closed range without a hidden scale. Median and P90 are not separate
semantic quantities from other quantiles, so they reuse the existing typed
`Quantile` condition. This adds four physical-prediction error MetricKinds, not
eight percentile-specific kinds or a generic metric-of-metric DSL.

## Why Learned Targets And Support Are Contract Owned

An Evaluation case identifies subjects and conditions, but it does not by
itself identify which physical provider, flow, library cohort, normalization,
or simulation fidelity supplied an observation. Pooling samples solely because
they share a case and MetricKind would silently train one function against
several different targets. Copying provider identity into every sample row
would instead create a second dataset schema and let trainers disagree about
which fields matter.

The parameter contract therefore names exact ground-truth model descriptors
and owns one derived target key embedded in its canonical payload. Provider
semantics, normalization, and fidelity define that observation function;
hardware, flow, library, and operating-condition variation remain typed
features of the function. Trainers, validators, and predictors share the same
projector and equality rule. A model that intentionally learns across sources
can register another contract whose feature view explicitly contains source
identity; the generic infrastructure does not guess that equivalence.

The same owner decides whether a valid feature view lies inside the learned
support region. Returning typed Unsupported is more honest than extrapolating
a number with no evidence and smaller than storing confidence labels or
diagnostic text in every bundle. Invalid input remains an error, while an OOD
case remains a stable semantic capability boundary. This also explains why a
predictor can rank and promote candidates but cannot prove infeasibility.

## Why The Initial Learned Provider Is Tabular And Extensible

Fabric structure, operating conditions, Mapping summaries, and system runtime
inputs are naturally mixed numeric and categorical tabular features. A
gradient-boosted tree ensemble provides deterministic CPU inference, explicit
serialization, useful nonlinear interactions, and bounded training cost
without introducing a tensor runtime into every Loom deployment. It is an
initial owner choice, not a universal model taxonomy.

The first physical target is one exact routed OpenROAD model. An open-source,
scriptable provider makes the large collection reproducible and schedulable
without weakening the requirement for real routed tool evidence. Commercial
flows remain valid independent ground truth, but assigning them separate model
descriptors prevents different algorithms, libraries, and report semantics
from becoming accidental labels for one function.

The versioned parameter-contract interface is the extension point. Another
algorithm registers its own payload codec, feature view, support rule, target
relation, and in-process inference kernel. Training may use an external
program, but inference remains inside the provider boundary. A central
algorithm enum or generic tensor payload would make DSE, rather than the model
owner, interpret parameters and would turn every new algorithm into a schema
change.

## Why Hardware DSE Uses Typed Domain Generators

Fabric already has precise typed owners for modules, occurrences,
connections, capabilities, schedules, services, and Systems. A generic
`HardwareAction` over node names and property bags would erase those
distinctions and require a second verifier. Mutating one shared graph would
also make parent identity, recovery, and lineage depend on execution order.

Hardware search therefore starts from an exact finalized seed and uses typed
template, topology, microarchitecture, and System-composition generators.
These create fresh drafts through the public Builder and publish only after
ordinary Fabric finalization. One Hardware-owned generator then performs the
pure portable-System-RTL derivation from the exact System and its independently
finalized ConfigurationABI. Provider-specific generators already own
all later gate, physical, FPGA, and native implementation transitions; an
additional generic implementation-flow wrapper would create competing
provider and work authorities. Module children are useful intermediate
designs, but the optimization subject is a complete
`fabric.system`: multi-core heterogeneity, transport, memory, services, and
InstructionCore realization affect software partitioning and system quality
even when detailed RTL remains scoped to SpatialCore modules.

## Why Joint Search Uses Finite Alternating Batches

Flattening a software frontier and a hardware frontier into a Cartesian
product spends work on combinations that neither parent-local search selected.
A mutable joint candidate or runtime loop would then make termination,
deterministic work, recovery, and cache identity scheduler-dependent.

Finite Generate and Promote nodes already express bounded exploration. The
plan alternates software and hardware batches, evaluates parent-local children,
and permits cross-pair reconsideration only through an explicit bounded
frontier join. Every cost and use-def edge is visible before execution; there
is no new workflow language or Journal-owned current best. Exact admission and
sound bounds may reject impossible designs, while analytical and learned
estimates only rank, promote, or choose which candidates receive expensive
evidence.

The frontier join is expressed by two narrow Candidate Generator adapters. One
reuses the Dataflow rewrite owner over a bounded Dataflow/System pair prefix;
the other reuses System composition for a bounded System/Module prefix and the
three root-complete Mapping owners for the resulting bounded Dataflow/System
prefix. This split preserves the real software-generation, hardware-action,
and Mapping boundaries while allowing a later Promote node to narrow either
frontier. A single generic joint-action generator would absorb compiler,
Mapping, hardware, and Evaluation semantics. A persistent pair Artifact would
add identity without an independently observable fact. Dynamically adding
plan nodes after execution would instead create the rejected runtime workflow
and make recovery scheduler-dependent.

Diagonal canonical pair order gives both input frontiers early representation
without allocating the full product. The positive pair bound is resolved
semantic policy, so truncation means exactly that the declared finite prefix
was searched. It does not imply that unvisited pairs are infeasible or that a
selected result is globally optimal.

## Why Ground-Truth Collection Reuses Plans And Evidence

A campaign is operationally large but semantically ordinary: a finite plan
requests exact Evidence and partitions it for one parameter contract. A
`DatasetArtifact` would duplicate Requests, conditions, providers, results, and
sample lineage already owned by Evidence. A `CampaignArtifact` would duplicate
the resolved plan and Journal. Reusing those owners keeps resume exact and
lets every accepted sample remain independently auditable.

The ten-minute sample and twenty-three-hour campaign bounds exist because
ground-truth throughput is a project-critical resource, not because wall time
changes model semantics. Shared uncached Mapping and implementation work must
be visible and charged, otherwise a nominally fast campaign merely hides its
cost in precomputation. A deterministic pilot from the same plan makes ETA
actionable without discarding evidence. Live counts, percentiles, bottleneck,
and ETA remain removable Journal projections, while graceful stop imports
atomic completions and resumes missing WorkUnitKeys. Timeouts stay incomplete;
they do not become negative training labels or infeasibility proofs.

Reconstructing a finalized owner report by invoking the provider again would
charge real work during resume and could change lineage or work accounting even
when every output root was already terminal. The owner therefore may publish a
typed immutable recovery record at the same atomic boundary as terminal work.
The Journal stores only its digest and verifies that it names the same run key,
WorkUnitKey, invocation closure, terminal outcome, and roots. Unlike an
in-flight checkpoint, this record deliberately spans later occurrences of the
same run key: it describes accepted finalized work rather than mutable attempt
state. Keeping the codec with the work owner preserves exact resume without
making the Journal a second report or selection authority.

## Why Derived Quantities Use Ordinary Metrics And Models

A generic formula registry would introduce a second semantic language beside
the MetricKind and Evaluation-model registries. It would need its own kind
identity, input typing, scope and case compatibility, unit algebra, bounds,
uncertainty propagation, versioning, and failure rules. Those are already the
responsibilities that distinguish one registered metric and model from
another.

Loom therefore promotes a reusable derived quantity only by registering an
ordinary MetricKind and an exact producing model. For example, `Runtime` is a
registered metric whose model must own the exact timing basis. Energy is not
computed merely because power and runtime happen to be available: its
integration window, activity basis, scope, units, and uncertainty semantics
must first be owned by a registered metric and model. The same rule keeps
throughput, speedup, and performance-per-area from becoming report-local
formulas or silently comparable observations.

## Why Objective Facts Have Several Projections

One semantic fact must have one source and normalization, but not every
consumer asks the same question. Pareto selection compares dimensions
componentwise. Final TopK selection needs a total lexicographic order and a
stable candidate-key tie break. Annealing needs one local energy scale whose
difference controls acceptance.

Packing all dimensions into one mixed-radix integer would preserve one chosen
lexicographic order, but it would also make an unrelated dimension bound
rescale local annealing deltas. It would not represent Pareto incomparability.
Loom therefore owns each source, direction, and exact affine quantization once,
then derives ObjectiveVector, WeightedLevel, TotalOrdering, and SearchEnergy
for their distinct uses.

A domain search may materialize only the selected transitive closure of these
records in its own component view. Copying the owner-typed record and remapping
references is preferable to either a dangling ordinal into the complete DSE
view or a dependency on its entire digest. The projection remains removable
and mechanically validated; Evaluation and DSE remain the sole semantic
owners.

The same distinction explains the PnR interaction binding. An obligation
template identifies the exact full oracle and reusable request shape, while an
interaction-domain reference identifies the typed ephemeral candidate
protocol. Their relation is sufficient. Incremental or guidance requirements
follow from how the binding is selected, so persisting another mode selector
would duplicate a fact that validation can derive.

Input slots belong to the exact evidence-acquisition policy rather than to
Promote as a universal mechanism. Central Promote and PnR therefore use the
same template primitive with different closed typed slot catalogs. This avoids
both a Promote-only template fork and an untyped subject map inside PnR.

Only Point metrics enter an objective because they denote one scalar. Choosing
an interval midpoint or a censored bound would create an estimator not owned by
the model descriptor. A policy that needs such an estimate must request a
different model whose completed output is a Point rather than hiding the
choice in DSE.

## Why Quality Gates Are Three-Valued

An interval may prove a threshold, disprove it, or straddle it. A censored
observation and `NotApplicable` have the same need for an explicit unknown
truth state. Treating these as false would discard candidates for lack of
proof; treating them as true would silently waive the gate.

Metric and Finding gates therefore produce definitely true, definitely false,
or indeterminate truth. Every referenced atom is a required obligation, so one
indeterminate atom makes promotion incomplete rather than being hidden by
another branch of a disjunction. Fully determined atoms then use ordinary CNF.
Gates remain acceptance predicates, not numeric objectives. Search guidance
names the underlying metric explicitly instead of inventing a gate-deviation
score with no metric owner or unit.

## Why Semantic Work And Execution Limits Differ

The amount of formal exploration can change the result. Candidate counts,
annealing moves, solver iterations, model calls, and repair attempts therefore
belong to their typed owner policies and resolved semantic configuration.

Wall time, process parallelism, host resources, scheduler queues, license
availability, and storage quotas only determine whether the plan finishes.
They may produce an incomplete attempt but cannot change which candidate is
formally selected. Promoting the best candidate observed before a timeout would
make results machine- and load-dependent.

Retries of the same request do not consume new semantic work merely because a
license or process failed. Conversely, an internal effort or threading setting
that changes a tool's result must be promoted into the model binding rather
than hidden as execution infrastructure.

## Why Operational Cost Stays In InvocationManifest

Compiler wall time, CPU consumption, peak resident memory, and effective
parallelism are essential feedback for improving Loom, but they do not describe
the evaluated program or hardware. Putting them in EvaluationEvidence would
make a loaded host, scheduler decision, or worker count look like a semantic
quality metric. A separate Telemetry Artifact would instead duplicate the exact
invocation, plan-node, and work-summary identities already present in
InvocationManifest.

The manifest is therefore the smallest correct owner for optional operational
observations. The same PlanNodeRef joins node time to deterministic work without
copying counts. Whole-invocation peak RSS is honest; per-node RSS is not,
because shared allocators, caches, and concurrent nodes make additive
attribution fictitious. Concurrent node wall times likewise remain overlapping
observations rather than a sum that pretends to reconstruct elapsed time.

Deterministic work stays primary for cross-machine comparison. Wall time, CPU,
and RSS are valuable within a compatible execution context, but recording CPU
count and worker request does not prove two machines equivalent. Keeping the
observations nonsemantic permits profiling and budget gates without letting a
host measurement change candidate identity, ordering, replay, or formal
completion.

External EDA execution has a different authority boundary. Loom prepares a
script and later imports an atomic result, while the caller or scheduler owns
the process in between. That owner may measure an exact attempt, but making Loom
monitor it would recreate the process-supervision authority explicitly rejected
by the external-tool contract. Missing EDA resource measurements therefore do
not invalidate real EDA Evidence.

## Why Evaluators Cannot Call One Another

Hidden recursive model calls conceal cost, authorization, lineage, failure,
and cache dependencies. Every cross-model dependency is an explicit plan edge
with exact upstream Evidence. Shared pure kernels and parameter bundles remain
libraries, not fake evaluator nodes.

The common in-process model interface provides exact artifacts and immutable
configuration views. Scratch placement, cancellation, resource scheduling,
and operational logging remain caller-owned execution concerns; they are not
semantic model inputs or ambient provider authority. An external model instead
prepares an exact invocation bundle and imports its declared result. If the
valid exact Request lies outside a stable provider capability, preparation may
instead return the existing typed Unsupported outcome. Evaluation, not the
provider, binds that outcome to the Request and finalizes Evidence. Neither
interface gives a model mutable DSE state, objective weights, or permission to
promote or replace candidates.

External Candidate Generators use the same ownership split without becoming
Evaluators. Their descriptor binds exact typed inputs and resolved generator
configuration, preparation emits only an attempt bundle, and import publishes
only descriptor-owned candidate outputs and returns their dense output bindings
plus typed lineage contributions to the central manifest owner. Reusing one generic
callback for both in-process computation and long-lived external execution
would hide the prepared-but-not-run state and encourage implicit process
launch; a descriptor-owned provider form keeps that distinction explicit.

The provider result also carries the invocation's work summary as one dense
transient field outside the outcome variant. The descriptor owns the stable
work-unit ordinals; the provider alone observes the planned and consumed
counts at runtime. Planning means admission of one immediate logical slot, not
copying a configured policy limit, and consumption follows the slot's atomic
owner boundary. This ordering preserves a truthful live gap when execution
fails or is cancelled before that boundary completes; a completed typed
negative outcome still consumes its slot. A mutable accounting side channel
would let callback state drift from the validated report and would invite
controller inference from output cardinality, which collapses whenever one
attempt yields several candidates or none. Returning the counts inside the one
validated transient report keeps them reviewable against the descriptor's
dense coverage rule, while the persistent owner of record remains the central
invocation manifest.

## Why External Tools Are Script Driven

EDA and external simulation tools already own Tcl, Python, shell, module,
license, and container conventions. Reimplementing those conventions in a C++
process supervisor would make Loom an environment manager and create another
place where site policy, tool setup, resource behavior, and debugging output
could diverge.

Loom instead resolves machine bindings once and emits an independently
executable bundle. Its manifest preserves structured commands and exact
provenance; generated Bash is only their executable projection. This keeps
tool-specific drivers reviewable and lets the same material run under a local
shell, Make, Ninja, Slurm, or site orchestration without changing compiler
semantics.

Some external simulators are compilers: the frozen tool first produces a
work-directory executable and only that executable can run the modeled
hierarchy. Treating the produced program as another discovered tool would add
a false binding authority, while allowing arbitrary later commands would lose
the bundle's executable closure. The manifest therefore names only fresh
tool-produced executables under `work/`; the launcher removes stale instances
and verifies each newly produced path before execution. A generated controller
may receive other listed generated executables as exact arguments when one
modeled simulator requires cooperating processes. Keeping those child paths in
the same manifest and checking them against the controller's semantic input
preserves one executable closure without making ExternalTool a process
supervisor or a general shell workflow language.

Separating prepare from import is the smallest lifecycle that supports that
execution boundary. The exact descriptor and semantic closure derive the
bundle and importer identity; callers cannot name them independently. An
interrupted bundle without completion is simply incomplete. Loom cannot infer
whether an external process still lives, so retry and concurrency remain with
the caller or site scheduler; another authorized attempt may retain the same
logical WorkUnitKey. Adding execution claims or mutable Job states would repeat
facts already owned by the bundle completion record, ExecutionJournal, and
external scheduler.

The same boundary is the narrowest sound place to reuse expensive successful
tool results. Provider adapters already converge on one structured invocation
manifest and one strict importer, so adapter-local caches would duplicate key
codecs and let two providers disagree about whether an input changed. A cache
in Evaluation or DSE would instead confuse raw attempt bytes with semantic
Evidence or candidate identity. ExternalTool therefore derives one
domain-separated input/configuration/tool-version key from the exact prepared
manifest, restores only verified declared bytes, and lets the existing owner
importer decide their meaning.

Caching the old completion record was rejected because it binds another
manifest and would create a competing success authority. A hit republishes a
fresh current-manifest completion only after current input and version
validation. Caching failures was also rejected: a license outage, timeout,
host interruption, or tool crash is an observation about one attempt, not a
stable capability fact. Typed Unsupported remains the negative-cache form for
owner-proven capability exclusions.

Three visible digests are retained instead of hashing the complete manifest.
The complete manifest contains local executable, module, external-file, and
bundle paths whose changes do not alter the tool question. The input digest
owns exact consumed material, the configuration digest owns the exact
operation and semantic closure, and the tool digest owns the exact executing
version and launcher bytes. This separation preserves cache reuse across
equivalent local bindings while preventing a seed, generated file, library
byte, runtime, provider version, or silently replaced launcher from being
ignored.

A stable capability rejection precedes that lifecycle. Returning typed
Unsupported before bundle construction preserves the exact Request as a
negative cache without inventing an executable no-op attempt. Restricting this
branch to `RuntimeCapabilityUnavailable` also prevents preparation from
claiming tool failure, cancellation, or completed observations that only a real
attempt and strict importer can establish.

Passing three independent strings or byte arrays to bundle finalization would
still let each adapter implement a private descriptor codec and accidentally
pair one closure with another importer. A single
`ExternalToolSemanticContract`, derived by the CandidateGenerator or
Evaluation owner, is the smaller boundary. ExternalTool owns the common hash
framing while each semantic owner supplies its descriptor-reference and
closure codecs. Adapters only transport the resulting value, so adding another
EDA ecosystem does not add another semantic authority.

Discovery belongs before bundle finalization. Explicit configuration has
priority over the current environment, and module discovery is only a final
fallback. Freezing the exact executable, loaded-module closure, version, and
runtime prevents repeated searches from selecting different tools in parallel
or on replay.

Tool and PolyArch/container choices are independent but require composition
preflight. A site module may already wrap a vendor container, an absolute tool
path may not be mounted, or a license environment may be unavailable inside an
outer runtime. Treating every pair as valid would make orthogonality a false
claim; rejecting an invalid pair before execution preserves both dimensions
without adding a hidden fallback.

Memory, CPU, namespace, process-tree, wall-time, image, and scheduler policy
belong to the shell, container, scheduler, or site that executes the script.
An externally stopped run may leave an incomplete attempt. It cannot select a
different model, fabricate partial Evidence, or justify a Loom-owned cgroup
abstraction.
