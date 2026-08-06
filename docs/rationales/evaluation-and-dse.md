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

## Why DSE Uses Typed Use-Def

The central plan is an SSA-like DAG of typed producer outputs and consumer
inputs. Candidate sets, Evidence, model parameters, and execution products use
the same use-def mechanism. A special CandidateSet reference or artifact-store
query for `latest` would create hidden ordering and mutable plan semantics.

Domain candidate generators own their candidate type, legal transformations,
deduplication, and local work. The central controller owns lineage, scheduling,
Evidence acquisition, promotion, Pareto selection, and deterministic replay.
It does not define one generic mutable candidate or action language.

When two domain owners expose different granularities, an explicitly typed
generator adapter composes them without broadening central semantics. The
root-complete TechMapping adapter is the representative case: the plan binds
whole Artifact candidates, Dataflow derives the complete local graph catalog,
and TechMapping consumes that ephemeral scope. A generic graph-cover value in
the plan would instead duplicate Dataflow-local identity and turn one mapping
policy into a second cross-domain authority.

The root-complete Spatial adapter applies the same rule to a different
granularity mismatch. A TechMapping already seals its Dataflow and Fabric
owners, while the exact Spatial PnR API must also receive a real constraint
Artifact. The adapter recovers `D` from `T`, verifies the separately supplied
exact `F`, and asks the constraint owner to publish the empty `K`; it does not
add duplicate `D` or optional `K` plan slots. Constrained searches continue to
use the exact owner directly. This keeps the central plan generic while making
the common unconstrained `D -> T -> SpatialMapping` chain explicit and fully
replayable.

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
directly. The same contract names its prediction and ground-truth case
signatures and consumes every result-affecting operating condition as a typed
feature. It does not call another evaluator, hide a downstream Request, drop a
condition, or duplicate predictor formulas. A selected bundle later enters a
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
prepares an exact invocation bundle and imports its declared result. Neither
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
counts at runtime. A mutable accounting side channel would let callback state
drift from the validated report and would invite controller inference from
output cardinality, which collapses whenever one attempt yields several
candidates or none. Returning the counts inside the one validated transient
report keeps them reviewable against the descriptor's dense coverage rule,
while the persistent owner of record remains the central invocation manifest.

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

Separating prepare from import is the smallest lifecycle that supports that
execution boundary. The exact descriptor and semantic closure derive the
bundle and importer identity; callers cannot name them independently. An
interrupted bundle without completion is simply incomplete. Loom cannot infer
whether an external process still lives, so retry and concurrency remain with
the caller or site scheduler; another authorized attempt may retain the same
logical WorkUnitKey. Adding execution claims or mutable Job states would repeat
facts already owned by the bundle completion record, ExecutionJournal, and
external scheduler.

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
