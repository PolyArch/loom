# Evaluation And DSE Rationale

Normative contracts are owned by
[Evaluation And DSE](../spec-dse-feedback.md),
[Evaluation Metrics](../spec-evaluation-metrics.md),
[Evaluation ToolRunner](../spec-evaluation-tool-runner.md), and
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

## Why Evaluators Cannot Call One Another

Hidden recursive model calls conceal cost, authorization, lineage, failure,
and cache dependencies. Every cross-model dependency is an explicit plan edge
with exact upstream Evidence. Shared pure kernels and parameter bundles remain
libraries, not fake evaluator nodes.

The common in-process model interface provides artifacts, scratch, a controlled
ToolRunner, cancellation, resource leases, and structured logging. It does not
give a model mutable DSE state, objective weights, or permission to promote or
replace candidates.

## Why ToolRunner Is Narrow

External EDA and simulation tools need consistent process control, environment
binding, output inventory, timeout handling, and secret isolation. ToolRunner
owns that local execution primitive, not tool semantics or workflow policy.
Adapters own script generation, report parsing, and typed Evidence.

A structured argv and explicit environment avoid a second shell language.
Tool versions and result-affecting options enter model identity; executable
paths, module locations, license endpoints, and temporary directories remain
nonsemantic bindings.

External tools may create descendant process trees, so a per-process address-
space limit or periodic process sampling cannot enforce an aggregate memory
budget. When a hard process-tree limit is requested, ToolRunner requires one
kernel-enforced containment owner that accounts all descendants, reports a
local OOM event, records aggregate peak memory, and terminates the complete
tree. The specification names the behavior rather than one operating-system
service API.

Touching a hard threshold is not equivalent to failure because reclaim can
allow execution to continue. Conversely, an ordinary allocation failure does
not prove that the containment owner killed the process tree. The terminal
status is therefore committed only from the owner-local hard-limit event. Peak
memory remains a raw execution fact; an Evaluation model must explicitly
interpret it before it can become a metric or Evidence.

## Why Training Produces Immutable Candidates

Calibration and learned models can improve over time without making a running
Evaluation mutable. Training consumes exact datasets and parameters and
produces an immutable model-parameter candidate. The same Evaluation/DSE
machinery validates and promotes it. A promoted default may later be checked
into the codebase, but online state never silently changes an existing model
identity.
