# Evaluation and DSE

This document specifies Loom's shared Evaluation infrastructure and central
design-space exploration boundary. Compilation, Mapping, hardware, and model
parameter search use the same Evaluation semantics. They do not own separate
metric systems, finding systems, fidelity ladders, candidate wrappers, or
evidence formats.

## Scope and Ownership

The central DSE controller owns:

- canonical candidate sets and typed candidate lineage;
- resolved objective, quality-gate, selection, and Evidence-acquisition policy;
- deterministic orchestration of the Evaluation dependency graph;
- cross-owner semantic-work admission and accounting;
- candidate comparison, promotion, Pareto selection, and formal controller
  outcomes; and
- the invocation summary and crash-recovery coordination records.

Domain generators and local solvers own their typed candidate spaces,
transformations, search mechanics, and owner-local work limits. Verifiers own
static legality and artifact closure. Evaluation models own only the metric
observations and findings they can honestly produce. Execution owners retain
their own attempts, checkpoints, and raw materials.

An evaluator never emits a policy score, accepts or rejects a candidate,
chooses a fallback model, invokes another evaluator, or mutates controller
state. The controller does not copy domain transformations into a universal
action algebra.

## Persistent Evaluation Boundary

The Evaluation black box has exactly two persistent semantic artifact
families:

```text
EvaluationRequest Artifact
  -> EvaluationModel black box
  -> EvaluationEvidence Artifact
```

Request and Evidence are the sole semantic Evaluation input and output. There
is no persistent `EvaluationResultArtifact`, partial-result artifact, model
artifact, or evaluator-specific request/result family. Queue entries, running
jobs, retries, attempts, and checkpoints are execution records rather than
semantic Evaluation artifacts.

An EvaluationRequest fixes one immutable problem and one exact resolved model
binding. EvaluationEvidence references that exact Request and is the only
persistent owner of normalized outcome, metric results, and finding results.
Both artifacts are immutable and have independent version spaces. Their shared
typed data model is the schema authority; canonical serialization is a
cold-path representation of that model, not another schema authority.

Raw execution material is deliberately outside normalized Evidence:

- a workload-running simulator owns its exact `SimulationExecution` artifact;
- a future raw detailed-bundle Artifact family will own scripts, logs, raw
  reports, and process material after its exact schema and importer are
  defined; and
- owner-specific attempt records own runtime provenance and retry history.

Evidence binds evaluator-produced semantic Artifacts through descriptor-owned
typed output slots. A workload-running simulator uses one
`SimulationExecution` output slot. `evaluation.evidence.1.0` has no raw-bundle
field or generic Artifact-reference escape hatch. Until the detailed-bundle
owner lands, providers may retain raw material only through owner-specific
attempt or scratch storage; it cannot enter Request or Evidence identity.

## Models, Metrics, and Findings

Fidelity is a typed description of model capability, not a global ordinal
ladder. A model can be detailed in workload behavior and abstract in hardware,
or detailed in physical implementation and unaware of application runtime.
The model descriptor states those capabilities without relabeling a fast
estimate as simulation, EDA, or physical measurement.

The metric registry defined by `docs/spec-evaluation-metrics.md` is the sole
owner of each `MetricKind`, including its definition, value domain, physical
dimension, canonical unit, permitted scope, and observation forms. The finding
registry is the sole owner of each `FindingKind`, including its semantic
definition, permitted scope, and typed occurrence payload. Metric and finding
queries use the same closed `EvaluationScope` algebra; there is no separate
finding scope or retained `MetricScope` authority.

Descriptors declare supported metric and finding query/result subsets. They do
not copy registry definitions or own severity, thresholds, weights, scores,
optimization direction, or candidate acceptance. Those are central resolved
DSE policy.

A stable derived metric is produced by an explicit typed derived-metric model
and ordinary EvaluationEvidence. Its inputs are exact upstream Evidence
references in descriptor-owned model slots. Benchmark weighting,
normalization, ranking, and reward remain DSE policy rather than MetricKinds or
evaluator outputs.

## Evaluation Case Foundation

The model-independent case uses one static typed
`EvaluationCaseSignature` registry. It is not an Artifact and it does not
contain model implementation facts:

```text
EvaluationCaseSignatureDescriptor {
  case_kind: EvaluationCaseKind
  subject_roles: ordered CaseSubjectRoleDescriptor tuple
  workload: forbidden | optional | required, with accepted Artifact schemas
  runtime_input: forbidden | optional | required, with accepted Artifact schemas
  workload/runtime-input compatibility
  whole_case_cycle_basis:
      Absent
    | UniqueReferenceCycle {
        source:
            AbstractCaseCycle
          | ExactSubjectCycle {
              accepted_reference_type: SubjectReferenceType
            }
        resolve(exact EvaluationCase, CaseArtifactResolution, ArtifactStore)
          -> ReferenceCycleBasis
      }
  permitted_base_conditions:
    canonical set<ConditionApplicabilityPattern>
}

ReferenceCycleBasis =
    AbstractCaseCycle
  | ExactSubjectCycle(SubjectTargetRef)

CaseSubjectRoleDescriptor {
  role_ref: CaseSubjectRoleRef
  semantic_role
  accepted Artifact schemas
  cardinality
  cross-role compatibility
}
```

The persistent registry reference is exactly
`(Evaluation schema version, EvaluationCaseKind)`. It is not an Artifact
reference, digest, string name, or model-descriptor-local ordinal.
`CaseSubjectRoleRef` is a stable ordinal local to one exact case-signature
version. The signature, not an Evaluation model, is the sole owner of subject
role, schema, cardinality, and compatibility. An
`EvaluationModelDescriptor` references exactly one case signature and declares
only its capability to evaluate that case. If one implementation supports
incompatible case signatures, the registry exposes separate model
descriptors.

`UniqueReferenceCycle` is an executable case-signature-owned projection, not a
boolean capability flag. `AbstractCaseCycle` is legal only when the signature's
semantics define one intrinsic tick, such as an abstract DFG cycle.
`ExactSubjectCycle` must be a canonical `SubjectTargetRef` anchored in the exact
case, resolve through its family-owned local-reference codec, and satisfy the
signature's declared reference-cycle type. The resolved variant must match the
descriptor's declared `source`. Resolution failure, a foreign anchor, a
noncanonical local reference, or more than one possible result is invalid.
`Absent` means that this case signature provides no reference-cycle basis.
Only the Metric registry decides which scope forms require such a basis.
The resolved basis is derived from the exact case and is not serialized as a
Request field; the exact signature ref and its registered typed resolver are
the authority. A model cannot derive, invent, or replace the basis in config.
The Metric registry's single `validateMetricScopeAdmissibility` operation owns
whether a form requires this resolver and is called by both model-descriptor
admission and `RequestVerifier`; neither caller duplicates metric-specific
cycle rules.

This replaces descriptor-local subject-slot definitions. Such slots made the
supposedly model-independent case key depend on model-specific ordinals, so two
different models could fail to align an otherwise identical case. There is no
global `SubjectKind`, unordered subject bag, or one-subject-per-schema rule.

Evaluation uses Common's exact typed references through case-signature roles:

```text
EvaluationSubjectBindings =
  total table<CaseSubjectRoleRef,
              canonical ArtifactRootReference collection>
```

Collections contain no duplicates and are ordered by complete root-reference
canonical key, including exact schema and identity; authoring order has no
meaning. Request verification enforces totality relative to the exact case
signature. Distinct roles may accept the same schema, so a comparison
signature can bind `reference_execution` and `candidate_execution` without
losing role.

The model-independent case has two orthogonal exact references:

```text
workload_ref: optional<ArtifactRootReference>
runtime_input_ref: optional<ArtifactRootReference>
```

The workload owns the work definition, shape, or problem size. The runtime
input owns concrete values, memory images, streams, and launch arguments.
Request never inlines tensors, file lists, or an arbitrary parameter map.

Base and request-specific conditions use one closed tagged union,
`EvaluationCondition`. Base conditions change the ground-truth problem shared
by every requested metric and finding. Request-specific conditions change only
the associated query. Conditions are not string-key maps, override layers,
arbitrary predicate languages, or consumer-private payloads. Model effort is
part of the model binding; tool paths, timeout, host parallelism, and licenses
are nonsemantic execution bindings.

The exact condition kinds, payloads, ordered target projection, allowed
locations, assignment keys, canonical order, and duplicate/conflict rules are
owned by `docs/spec-evaluation-metrics.md`. The case-signature descriptor owns
complete ordered role/reference-type patterns for Base conditions. Metric and
Finding descriptors own the same kind of complete patterns for
request-specific conditions. A model descriptor declares which otherwise
legal exact patterns it consumes, requires, or proves invariant; it cannot
redefine their meaning, widen them through independent role/type sets, or
silently ignore an unsupported condition.

Case keys are removable derived indexes:

```text
base_case_key = DomainSeparatedDigest(
  exact EvaluationCaseSignatureRef,
  canonical subject bindings,
  workload_ref,
  runtime_input_ref,
  canonical base_conditions)

metric_case_key = DomainSeparatedDigest(
  base_case_key,
  MetricQuery,
  canonical metric-request conditions)
```

Neither key is serialized into Request or Evidence. Full Request identity also
depends on the exact model binding, both canonical request sets, and
`replicate_index`. Evidence references Request rather than copying case facts
or derived keys.

Two model descriptors produce the same base or metric case key only when they
reference the same exact case signature and bind identical role-labeled
subjects, workload, runtime input, and conditions. Their distinct model
bindings still produce distinct Request identities. For example, a timing
predictor and a physical timing tool can share one Hardware Implementation
case while retaining different parameter, technology-tool, and execution
bindings.

### Cross-Stack Subject Profiles

The shared case-signature registry expresses the exact subjects consumed by a
model at any point in the stack. The initial production profiles are:

| Evaluation purpose | Required subject closure |
| --- | --- |
| structured software analysis | Structured Program Candidate |
| hardware-aware structured analysis | Structured Program Candidate, Fabric |
| Dataflow software analysis | Canonical Dataflow Program |
| hardware-aware Dataflow analysis | Canonical Dataflow Program, Fabric |
| hardware-only analysis | Fabric |
| mapped or CGRA analysis | Canonical Dataflow Program, Fabric, Mapping |

This table does not introduce a second case-kind enum or a generic optional
subject bag. Every real model registers one exact
`EvaluationCaseSignatureDescriptor` with ordered roles, accepted Artifact
schemas, workload and runtime-input requirements, conditions, and cycle-basis
resolver. An analytic `(D,F)` model and a simulation `(D,F)` model with a
required workload are therefore distinct exact signatures even when their
subject roles match.

The compiler invocation always resolves one exact Fabric target. A
software-only signature means only that the selected model does not consume
Fabric; it cannot authorize ambient target lookup. For a mapped case, the
Mapping must resolve to the same exact Dataflow and Fabric Artifacts supplied
in the other roles. A Mapping reference alone is not a substitute for the
complete subject closure, and Request verification rejects any mismatch.

Evidence participates in compilation only through typed use-def edges in the
central DSE plan. Mapping-aware Evidence may be projected into a Structured or
Dataflow generator, but that generator creates a new immutable candidate. It
cannot copy Mapping-private records into software IR, mutate an existing
candidate, or rebind the Mapping. Promotion to Mapping is an explicit,
potentially expensive branch for a selected survivor set rather than an
implicit action performed by frontend passes.

## EvaluationRequest and Model Descriptor

`evaluation.request.1.0` has one strict typed root:

```text
EvaluationRequest {
  subject_bindings: EvaluationSubjectBindings
  workload_ref: optional<ArtifactRootReference>
  runtime_input_ref: optional<ArtifactRootReference>
  base_conditions: canonical set<EvaluationCondition>
  metric_requests: canonical set<MetricRequest>
  finding_requests: canonical set<FindingRequest>
  model_binding: ResolvedModelBinding
  replicate_index: uint64
}

MetricRequest {
  query: MetricQuery
  conditions: canonical set<EvaluationCondition>
}

FindingRequest {
  query: FindingQuery {
    kind: FindingKind
    scope: EvaluationScope
  }
  conditions: canonical set<EvaluationCondition>
}
```

`MetricQuery` also uses `EvaluationScope`. The two request sets are independent,
but their total cardinality must be nonzero. A finding-only Request is legal.
The same query may appear with different request-specific conditions; only an
exact duplicate request is invalid.

Request does not serialize its case signature separately. Its exact model
descriptor resolves one `EvaluationCaseSignatureRef`, and subject bindings are
verified against that signature. Case keys include the resolved signature
reference so the signature participates once without becoming a second
Request-owned authority.

An API overload that accepts an already constructed `EvaluationCase` must
compare that case's exact signature ref with the exact signature owned by the
resolved model descriptor before projecting any case fields. A mismatch is an
error; the API must not discard the supplied signature and reconstruct or
rebind the fields under the descriptor's signature. An overload that accepts
only component fields has no caller-supplied signature and constructs the case
directly under the descriptor-owned exact signature.

Finalization sorts each set by complete canonical content. The resulting
positions are the only request-local `MetricRequestOrdinal` and
`FindingRequestOrdinal` identities. Evidence refers to a request item by exact
Request identity and ordinal. It never copies the query or its conditions.

Request stores neither its own ArtifactIdentity nor any case, model, or cache
digest. Common artifact finalization derives identity from the one canonical
root.

### Descriptor Capability Authority

`EvaluationModelDescriptor` is an immutable, versioned capability entry in the
Evaluation library's static typed registry, not an Artifact. It owns:

- model kind, descriptor schema and version, and implementation semantic
  identity;
- one exact `EvaluationCaseSignatureRef`;
- permitted exact `ConditionApplicabilityPattern` consume, require, and
  invariant capabilities;
- supported metric and finding queries and result forms;
- descriptor-owned model input slots, role-labeled typed output slots, and
  resolved model-config schema;
- modeled phenomena, execution method, and determinism contract; and
- exact optional domain-specific incremental and guidance interaction
  capabilities.

Schema 1.0 owns these closed capability enums and stable zero-based `uint32`
wire tags:

```text
ModeledPhenomenon =
    StructuredProgram
  | CanonicalDataflow
  | SpatialResources
  | RoutedTransport
  | FiniteBuffering
  | MemoryContention
  | ClockTiming
  | SystemMemoryHierarchy
  | Coherence
  | RTLBehavior
  | PhysicalImplementation

EvaluationExecutionMethod =
    Analytic
  | Simulation
  | Emulation
  | ToolMeasurement
  | PhysicalMeasurement
```

The Evaluation registry owns the enum semantics. `StructuredProgram` covers
pre-Dataflow program structure and analyses; `CanonicalDataflow` covers actor
and logical-net semantics; `SpatialResources` covers compute, memory, and
transport capabilities; `RoutedTransport` covers selected transport paths;
`FiniteBuffering` covers bounded queues and backpressure; `MemoryContention`
covers competing accesses to memory services; `ClockTiming` covers cycle and
critical-path timing; `SystemMemoryHierarchy` covers caches, NoC, and external
memory; `Coherence` covers coherence-visible behavior; `RTLBehavior` covers
RTL execution semantics; and `PhysicalImplementation` covers synthesized or
placed physical properties. A model lists every phenomenon whose semantics it
directly models, not facts merely imported through typed Evidence inputs.

`Analytic` performs formula-based or learned inference without executing the
subject's dynamic implementation. `Simulation` advances an explicit state
model of that implementation. `Emulation` executes it on a nonfinal hardware
or accelerated proxy. `ToolMeasurement` invokes a tool flow that measures or
derives an implementation property without being classified as workload
simulation. `PhysicalMeasurement` observes an actual physical realization.
Process location does not select the method: an external RTL simulator remains
`Simulation`, while external STA remains `ToolMeasurement`. Exact model
identity, parameter-bundle inputs, and uncertainty distinguish implementations
within one method.

The modeled-phenomenon set is sorted by ordinal and contains no duplicate. It
may be empty for a purely derived model whose exact upstream Evidence inputs
own all modeled facts. Execution method is exactly one value and is recovered
from the model descriptor rather than copied into observations.

The static descriptor's canonical capability projection encodes the phenomenon
set as `u64be(count)` followed by its sorted `u32be(tag)` values and encodes the
execution method as one `u32be(tag)`. C++ registration uses the closed enums,
not strings or raw integers. Human-readable spellings are registry-derived and
never an identity input.

Full Evaluation has no separate domain field. Every registered model must
implement the public full `evaluate(EvaluationRequest)` contract, and its full
domain is operationally the exact set of Requests accepted by
`RequestVerifier`. That verifier intersects the descriptor's case signature,
condition capabilities, metric and finding capabilities, input and output
slots, resolved config view, mandatory terminal findings, determinism, and
replicate rules. Copying a second full-domain description would be another
authority.

Incremental and guidance protocols are optional cross-owner interactions:

```text
EvaluationInteractionDomainRef {
  owner_registry_identity: canonical nonempty ASCII
  owner_registry_version: X.Y
  owner_local_domain_kind: uint32
}

EvaluationInteractionMode = Incremental | Guidance

EvaluationInteractionCapability {
  domain_ref: EvaluationInteractionDomainRef
  modes: canonical nonempty set<EvaluationInteractionMode>
}
```

The domain owner, not Evaluation, registers the exact referenced descriptor.
That descriptor owns the immutable candidate-view type and, for each admitted
mode, the incremental delta or guidance query/value types, completeness rules,
typed C++ protocol, validators, and equivalence rule against the model's full
Evaluation oracle. Incremental protocols own typed rebuild, probe, commit, and
discard operations; guidance protocols own typed query and value operations.
The owner creates separate domain kinds when candidate completeness or admitted
temporary states differ. Evaluation does not add a generic candidate property
bag or copy those domain facts.

Candidate views, deltas, guidance queries, and guidance values are ephemeral
owner-typed in-process values. They do not enter EvaluationRequest,
EvaluationEvidence, a generic replay record, or this domain-ref wire. If an
owner needs recovery persistence, its own checkpoint Artifact owns the codec
and lineage. Attempting generic interaction-payload serialization is an
unavailable capability, not permission to store opaque bytes.

For example, Mapping owns a PnR interaction-domain kind whose typed values are
`PnrCandidateView`, `PnrCandidateDelta`, `FrozenRouteQuery`, and a nonnegative
route-guidance value. A model that lists the exact ref may support incremental,
guidance, or both; a model that lists no interaction capabilities remains a
complete full-only evaluator.

The complete domain-ref canonical key is `u64be(owner_identity_byte_length)`,
the exact owner-identity bytes,
`u32be(major)`, `u32be(minor)`, and `u32be(owner_local_domain_kind)`.
Capabilities sort by that key. Each capability then encodes
`u64be(mode_count)` followed by its sorted `u32be(mode_tag)` values. Duplicates,
unknown owners or kinds, and a mode not implemented by the owner descriptor are
invalid. These capabilities belong to the static model descriptor and are not
serialized again in EvaluationRequest.

`EvaluationModelDescriptorRef` is the only persistent descriptor carrier. Its
exact `(schema version, model kind)` tuple immutably selects one descriptor,
including its implementation semantic identity and config-view contract.
Ambient codebase identity may verify that the registry implementation is
compatible, but it is not another selector. There is no descriptor Artifact or
opaque descriptor payload in Request.
Registry admission requires canonical enum sets, resolvable interaction refs,
valid metric/finding form subsets, complete slot tables, one config-view
contract, and every mode-specific typed callback. The descriptor ref recovers
those static facts during replay; changing an existing fact without changing
its owning schema or model kind is an incompatible registry error rather than a
runtime fallback.

An external tool's exact version, technology inputs, semantic switches, and
result-affecting effort or threading enter the model binding. Executable paths,
modulefiles, license servers, hosts, and scratch paths remain execution
bindings.

Each Request fixes one exact binding:

```text
ResolvedModelBinding {
  descriptor_ref
  input_bindings:
    canonical table<ModelInputSlotRef,
                    canonical ArtifactRootReference collection>
  resolved_model_config: ResolvedModelConfigViewWire
}

ResolvedModelConfigViewWire {
  canonical_view_bytes: canonical byte string
  component_view_digest: ComponentViewDigest
}
```

The exact model descriptor owns one `ResolvedModelConfigViewContract`:

```text
ResolvedModelConfigViewContract {
  schema_descriptor_bytes
  project(exact ResolvedConfig) -> owner-typed immutable view
  encode(owner-typed view) -> canonical_view_bytes
  adopt(canonical_view_bytes, component_view_digest)
    -> owner-typed immutable view
}
```

The central config library invokes the registered projector. The adopter
recomputes Common's fixed `component_view_digest`, decodes and validates the
owner-typed value, re-encodes it, and requires exact byte equality. The
evaluator receives that adopted typed view and cannot inspect raw bytes or the
full ResolvedConfig. A deliberately empty view uses empty canonical bytes and
the digest of the descriptor plus those empty bytes.

The persistent binding does not repeat the schema descriptor because the exact
`descriptor_ref` recovers it. Canonical JSON represents view bytes as lowercase
hexadecimal and the digest as exactly 64 lowercase hexadecimal characters.
Omitting the digest, accepting a foreign descriptor, exposing only a raw-byte
validator, or disagreeing on decode/re-encode is invalid. The descriptor ref
itself encodes as `u32be(schema_major)`, `u32be(schema_minor)`, and
`u32be(model_kind)` in canonical binary keys; Request's canonical JSON uses the
equivalent integer fields.

`ModelInputSlotRef` is a stable ordinal local to one descriptor version. Its
slot descriptor alone owns accepted Artifact schemas, cardinality, and
compatibility. The same mechanism binds parameter bundles, calibration
bundles, and role-labeled upstream EvaluationEvidence without an unordered
Evidence bag.

`ModelOutputSlotRef` is likewise a stable descriptor-local typed ordinal. Its
descriptor owns role, Artifact schema, and closed cardinality per Evidence
outcome. Evidence binds exact produced Artifact references but does not copy
the slot definition. A workload-running simulator declares one
`SimulationExecution` output slot: a `Retired` or `Halted` execution produces
exactly one such output with Completed Evidence. Any output allowed for an
incomplete outcome is governed by the same closed outcome-cardinality table,
not evaluator discretion.

A workload-running simulator descriptor also owns the complete set of
mandatory terminal `FindingQuery` values for every `Halted` kind it can emit.
A legal Request explicitly requests all of them. The Request verifier rejects
omission before execution; the evaluator cannot append an unsolicited
terminal finding afterward.

The canonical binding bytes derive a removable `resolved_model_key`; the key
is absent from Request. Only the typed ResolvedConfig component consumed by
the model enters the binding. Its canonical view bytes and digest are both
included in the binding, and the digest remains a checked mechanical
projection, not a second config authority. Unrelated visualization or
output-path settings cannot change Request or cache identity, while every
consumed semantic model setting and input must.

### Replicates, Attempts, and Admission

`replicate_index` is a nonnegative `uint64` in Request. It changes Request
identity without changing base-case, metric-case, or resolved-model keys.
Deterministic models accept only zero. Independent stochastic or physical
samples use distinct replicate Requests.

Retrying an exact Request creates no new case, binding, replicate, or semantic
work. Evaluation's owner-specific attempt record stores request-local attempt
ordinals, runtime provenance, Execution Limits outcomes, and retained Evidence
or bundle references. Attempt metadata never enters Request or Evidence
canonical bytes, and an earlier attempt is never overwritten.

Admission has three nonoverlapping owners:

1. `RequestVerifier` checks canonical form, descriptor and case-signature
   resolution, case-role totality and compatibility, workload/runtime
   requirements, condition location and applicability, scope anchors and
   targets, metric and finding capabilities, shared Metric scope
   admissibility and reference-cycle resolution, model slots and config, and
   replicate validity.
2. `EvaluationPlanAdmission` checks model authorization, Evidence obligations,
   dependency readiness, and deterministic semantic work.
3. `ExecutionAdmission` checks tool availability, licenses, storage, host
   resources, and Execution Limits.

None of these layers rewrites the immutable Request or performs another
layer's work.

## EvaluationEvidence

`evaluation.evidence.1.0` has one tagged root:

```text
EvaluationEvidence {
  request_ref: exact EvaluationRequest reference
  output_bindings:
    table<ModelOutputSlotRef, canonical ArtifactRootReference collection>
  outcome:
    Completed {
      metric_results: canonical array<MetricResult>
      finding_results: canonical array<FindingResult>
    }
    | Unsupported { reason: OutcomeReason }
    | ExecutionFailed { reason: OutcomeReason }
    | CancelledOrTimeout { reason: OutcomeReason }
}
```

The two result arrays have exactly the cardinality of the corresponding
canonical Request arrays. Array position is the request-local unsigned 64-bit
ordinal; a result does not serialize that ordinal again. Missing, extra, or
reordered results are invalid.

Output bindings must satisfy the exact descriptor's cardinality for the
selected outcome. Slot role, schema, and cardinality are recovered through the
Request; Evidence stores only exact output references. An analytical model may
declare an empty output signature.

`Completed` means the model fulfilled the Request, not that the candidate has
good quality. Negative slack, a deadlock, a functional mismatch, or another
adverse observation remains a Completed evaluation. Each result array is
exactly total over its corresponding request ordinals: no omissions,
duplicates, reordering, or unsolicited results are permitted. For a
finding-only Request, the metric array is empty and total, while every finding
request has one result.

Non-completed outcomes structurally have no metric or finding result arrays.
Partial tool output from before a failure or cancellation stays in retained raw
material. Controller-level unsatisfied obligations are represented by the
controller's `Incomplete` outcome, not a fifth Evidence outcome.

`OutcomeReason` is a closed typed union and is the only normalized failure
classification in Evidence. Evidence has no generic diagnostic string, list,
or key-value bag. Human-readable messages, vendor warnings, stdout, stderr,
and partial reports remain owner-attempt or scratch material until the raw
detailed-bundle owner is defined. Timestamps, host details, retry history, and
execution-limit details remain owner-attempt material.

Metric result position is the requested ordinal; the result contains only its
observation form, value or bounds, uncertainty, and any referenced calibration
input-slot ordinals.
Observation forms are `Point`, `Interval`, `Censored`, and `NotApplicable`;
execution failure, timeout, and unsupported capability are not observation
forms. Metric kind, scope, conditions, unit, dimension, permitted forms, and
evidence method are recovered from Request and the registries rather than
copied into Evidence.

Finding result position is the requested ordinal; the result contains one of:

- `Absent`, with no occurrences;
- `Present`, with a nonempty canonical typed occurrence set; or
- `NotApplicable`, with a typed reason and no occurrences.

Absence is therefore an explicit result for a requested finding, never an
inference from a missing record. Occurrence payloads come from the
FindingKind registry. Finding results never contain severity, a score, or a
candidate decision.

For a simulator, `Retired` returns `Absent` for every mandatory terminal
finding. `Halted { kind, witness }` returns `Present` for the corresponding
query and `Absent` for every other mandatory terminal query. That `Present`
occurrence is a `TerminalWitnessRef` containing the descriptor-local
`SimulationExecution` output slot and the execution ordinal within that
slot's canonical output binding. The referenced execution owns the typed
witness instance; Evidence does not copy it. Both terminals produce Completed
Evidence with total result arrays.

Schema 1.0 deliberately contains no `detailed_bundle_refs`. The earlier generic
`canonical set<ArtifactRootReference>` shape was removed because no exact raw
detailed-bundle schema, root kind, canonical payload, or importer owns those
references. Keeping an always-invalid generic field would be a second authority
and permanent wire slop. A later schema minor may add one exact typed reference
only after that Artifact owner defines immutable content inventory, exact
Request lineage, canonical framing, and the prohibition on normalized outcome,
MetricResult, or FindingResult copies.

The schema-1.0 dependency direction is therefore:

```text
SimulationExecution -> EvaluationRequest
EvaluationEvidence -> EvaluationRequest + typed output Artifacts
```

A simulator that executes a workload retains the exact `SimulationExecution`
as a typed Artifact. It owns terminal
execution observations, output values and streams, visible logical-memory final
state or diffs, completion and retirement observations, typed activity
summaries. Schema 1.0 contains no trace-manifest field. Trace-chunk and waveform
persistence remains unavailable until the raw detailed-bundle owner and a
Simulation Artifacts schema minor land; a simulator cannot replace them with
paths, opaque bytes, or provider-private references. `SimulationExecution` contains no
normalized metrics, findings, Evaluation outcome, DSE decisions, or second
simulator result schema.

## Candidate Lineage and Evaluation DAG

A central DSE candidate is an existing exact `ArtifactRootReference`. There is no
`DseCandidateArtifact`, generic CandidateKind, or wrapper identity. A mutable
domain-local search state enters a central candidate set only after it is
finalized as its domain Artifact.

Invocation lineage has exactly two edge kinds:

```text
MechanicalDerivation:
  exact input ArtifactRootReferences + producer/config
  -> output ArtifactRootReference

CandidateDecision:
  parent candidate ArtifactRootReferences + typed decision
  -> child ArtifactRootReference
```

Mechanical lowering cannot be represented as an optimization decision, and a
decision edge cannot replace an Artifact's own dependency closure. If several
paths produce the same ArtifactIdentity, the central set contains one candidate
node while the invocation manifest may retain every valid lineage edge.
Ranking, selection, and Evaluation deduplicate by Artifact identity rather than
generation path.

The Evaluation DAG is derived from exact references; there is no
`EvaluationDagArtifact`. Requests reference finalized subject Artifacts through
descriptor-owned role slots and any
upstream Evidence in descriptor-owned model slots. Evidence references its
exact Request and typed output bindings. These references mechanically recover the
persistent data dependencies. A model dependency cycle discovered while
resolving the plan is rejected before any Request is created.

Promotion and gate order are resolved policy, not data-dependency edges.
Pending, ready, running, retry, and blocked states belong to the mutable
ExecutionJournal. An evaluator cannot create a hidden downstream Request or
select a runtime fallback.

The controller resolves each obligation before execution:

```text
ResolvedDseConfigView
+ exact candidate ArtifactRootReference
+ objective or promotion gate
-> EvidenceObligationTemplate instantiation
-> ResolvedEvidenceObligation
-> exact EvaluationRequest
```

Model authorizations permit descriptor and binding domains but do not select a
provider. Capability alternatives are deterministically resolved to an exact
binding before Request construction. A runtime unsupported or failed outcome
cannot switch provider. Resolved obligations are rebuildable controller state;
Request and Evidence are the persistent facts.

Formal controller outcomes are:

```text
CompletedSelection {
  selected ArtifactRootReferences
  satisfied Evidence obligations
}

CompletedNoFeasibleCandidate {
  completed deterministic plan
  empty selection
}

Incomplete {
  unsatisfied obligations
  retained finalized Artifacts and Evidence
  typed interruption reason
}
```

Invalid inputs or resolved configuration fail verification before a run.
External failure, cancellation, or exhausted Execution Limits can produce only
`Incomplete`; best-so-far state is not a formal selection.

## Deterministic Work, Candidate Sets, and Cache

Semantic work limits remain with the policy that defines the algorithm. PnR
search policy owns move, restart, and expansion limits. Candidate-generator
policy owns expansion counts. Acquisition policy owns Request and replicate
counts. A `TopK` selection owns `k`. ResolvedConfig stores each number once and
the controller derives a read-only audit view:

```text
DeterministicWorkBudgetView =
  canonical set<(owner-local WorkUnitDescriptorRef, uint64 limit)>
```

`WorkUnitDescriptorRef` combines the owner policy schema/version with an
owner-local stable ordinal. It is not a string-key registry. The derived view
supports cross-owner admission, accounting, and replay without redefining any
work unit.

Each logical work unit receives a stable ordinal in owner-defined canonical
order before parallel scheduling. A cache hit consumes the same logical slot
as a cache miss. A generator attempt that reproduces an existing Artifact
consumes its attempt slot before candidate deduplication. A retry of the same
Request consumes no new semantic work; a new replicate is a new Request and
work unit. Worker count, wall time, license concurrency, process retry limits,
and host resources are Execution Limits and cannot change the formal plan.

The controller owns finite candidate sets as canonical sets of complete typed
ArtifactRootReferences. They are controller-local values, not Artifacts. Every
promotion has one shape:

```text
input candidate set
  -> instantiate required Evidence obligations
  -> require comparable Completed results
  -> apply resolved quality gates
  -> apply one selection policy
  -> output candidate set
```

The selection policy is exactly one of:

```text
AllPassing
TopK { total_ordering, k }
Pareto { objective_dimensions }
```

`TopK` and `Pareto` consume the same central objective facts. Pareto retains
all nondominated candidates in the deterministic finite input set; there is no
implicit cap based on container size, arrival order, or Execution Limits.

Every candidate entering one comparison receives same-shaped Evidence
obligations. A missing result is never interpreted as zero, infinity, or worst
score. A candidate removed by an earlier resolved gate need not acquire later
expensive Evidence, but runtime timing and cache state cannot alter gate order.

The Artifact store may maintain a removable index:

```text
exact EvaluationRequest identity
  -> canonical set<ArtifactRootReference>
```

Every value in this index has the exact `evaluation.evidence.1.0` schema.

Completed Evidence satisfies an obligation only when its total results and
forms match that obligation. Unsupported is a stable negative cache for the
same exact Request but never satisfies Required Evidence. ExecutionFailed and
CancelledOrTimeout are execution history and may be retried. Conflicting
terminal semantic outcomes for one exact Request, or differing Completed
normalized results from a deterministic model, are determinism violations;
neither latest timestamp nor attempt order resolves them.

The semantic closure of a run is its root Artifact identities, exact
ResolvedConfig, and explicitly selected preexisting Evidence identities. The
work-budget view and resolved plan are derived from that configuration. Resume
reuses finalized outputs for their original stable ordinals and does not
renumber or recount completed work. The resumed formal result must equal an
uninterrupted run with the same closure. Changing a root input, semantic
configuration, owner-local budget, model epoch, obligation, or acquisition
replicate creates a different semantic run; changing an Execution Limit does
not.

## Resolved DSE Policy

The central plan is one ordered SSA-like block. Every produced value uses one
reference form:

```text
PlanOutputRef = (producer_node_ordinal, output_slot_ordinal)

PlanValueRole =
    CandidateSet
  | EvidenceSet
  | SimulationExecutionSet
  | ModelParameterBundleSet

PlanValueCardinality = ExactlyOne | ZeroOrOne | NonEmptySet | FiniteSet

PlanOutputSlotDescriptor {
  role: PlanValueRole
  accepted_artifact_schema: ArtifactSchemaDescriptor
  cardinality: PlanValueCardinality
}

PlanInputBinding =
    ExactArtifacts(canonical set<ArtifactRootReference>)
  | ProducedValue(PlanOutputRef)
```

An output slot's immutable runtime value is always a canonical set of exact
`ArtifactRootReference` values satisfying its schema and cardinality. A candidate
set is therefore the `CandidateSet` role, not a distinct reference mechanism.
The descriptor and resolved node own slot meaning; neither a container type nor
the Artifact Store may reinterpret it.

The plan has two node kinds:

```text
ResolvedDsePlanNode =
    Generate {
      typed_input_bindings
      generator_binding
    }
  | Promote {
      typed_input_bindings
      acquisition_policy
      quality_gate_policy_ref
      selection_policy
    }
```

An input binds either an explicit canonical set of exact static Artifacts or an
earlier `PlanOutputRef`. Resolution checks role, exact Artifact schema,
cardinality, producer-before-consumer ordering, and acyclicity. Independent
nodes may execute in parallel, but canonical inputs and node policy alone
determine each output. Artifact Store scans, `latest` selection, implicit model
recursion, and mutation of a resolved plan are invalid dependency mechanisms.

Slot descriptors are not copied into resolved nodes. A Generate node derives
them from its exact `CandidateGeneratorDescriptor`. The built-in Promote
contract owns the selected-set slot, while its exact acquisition policy derives
the Evidence and descriptor-output slots. `PlanOutputRef` addresses those
owner-defined ordinals. Import, resolution, and replay reject any descriptor or
policy whose derived slots differ; there is no node-local override.

`Generate` expands or transforms a design space through a typed domain
generator. `Promote` acquires Evidence, applies gates, and narrows a set. Its
typed outputs include the selected candidate set and, when declared, the exact
Evidence or descriptor-produced Artifacts acquired by that node. For example,
a simulator promotion may expose both `EvidenceSet` and
`SimulationExecutionSet`; a later training generator consumes the Evidence and
produces a `ModelParameterBundleSet`. Evidence-to-candidate association remains
derived from `EvaluationEvidence -> EvaluationRequest -> subject`; the plan
does not persist a parallel association map.

Acquisition policy owns obligation templates, replicate generation, and its
owner-local work limits. Selection policy remains `AllPassing`, `TopK`, or
`Pareto`. There is no `PromotionPolicy`, runtime loop, generic workflow DSL, or
mutable workflow authority. Repeated finite Generate and Promote nodes express
cross-domain iteration.

Candidate-generator capabilities live in a static typed descriptor registry.
A resolved binding fixes typed inputs through the common plan bindings, typed
generator configuration, and objective/Evaluation projections.
The generator produces domain Artifacts and `CandidateDecision` lineage. Its
domain owns transformations and local search semantics; the central plan sees
only canonical input and output Artifact sets.

### Objectives and Quality Gates

One central dimension type owns the fact being optimized and its direction:

```text
ObjectiveDimension {
  source: ObjectiveScalarSourceRef
  direction: Minimize | Maximize
}
```

`TopK` references a `TotalOrdering` composed from ordered weighted levels;
`Pareto` references a canonical nonempty set of the same dimensions. Domain
objective projections derive from the same dimensions and ordering. Source,
direction, normalization, rank, energy, and reward therefore have one semantic
owner and are not Evaluation outputs.

A quality gate is finite canonical conjunctive normal form:

```text
QualityGatePolicy =
  canonical AND<canonical nonempty OR<QualityGateAtom>>
```

The only atoms are typed `MetricGate` and `FindingGate`. An empty policy means
no quality constraint; an empty clause is invalid. Finalization canonicalizes
clauses and removes duplicates. It does not add a predicate language,
callbacks, SAT representation, or Boolean-equivalence engine. Every referenced
atom creates an Evidence obligation, so Boolean short-circuiting does not hide
required evaluation. Clause deviation may guide search, but final acceptance
uses CNF truth rather than a weighted penalty.

### Resolved Configuration View

The fully elaborated component view is:

```text
ResolvedDseConfigView {
  model_authorizations
  evidence_obligation_templates
  objective_dimensions_and_orderings
  quality_gate_policies
  resolved_plan_nodes
}
```

Authoring-level allowed models resolve only to `model_authorizations`.
Authoring-level required Evidence resolves only to
`EvidenceObligationTemplate`. A template fixes the exact model binding, case,
conditions, and metric/finding requests while retaining one descriptor-typed
candidate `CaseSubjectRoleRef` binding. Promote fills that role to construct
exact Requests;
gate and objective references use template-local request ordinals rather than
copying metric or finding definitions.

The complete ResolvedConfig is policy SSOT. `ResolvedDseConfigView` is its
versioned canonical component view. Candidate sets, Evaluation DAG, stable work
ordinals, work-budget view, objective projections, cache indexes, scheduler
state, and mutable domain search state are derived or rebuildable and are not
Artifacts. Profiles and inheritance must elaborate before execution; the
controller cannot add defaults or hidden promotion while running.

## Invocation and Recovery Records

Formal semantic results remain the selected domain Artifacts and the exact
EvaluationEvidence satisfying their obligations. Central execution recording
has three layers:

```text
InvocationManifest
  immutable execution-occurrence summary and provenance

ExecutionJournal
  mutable crash-recovery state

Owner attempt/checkpoint records
  evaluator, tool, or domain-specific raw execution state
```

`InvocationManifest` is a versioned persistent record, not an Artifact. It has
no ArtifactIdentity and cannot be consumed as compilation, Mapping,
Evaluation, or deployment input. `ExecutionJournal` and owner records are also
nonsemantic. They cannot affect candidate identity, ranking, or selection.

The cache family owns one versioned derived run key:

```text
DseRunKey = DomainSeparatedDigest(
  producer semantic/build identity,
  canonical ArtifactRootReferences,
  exact ResolvedConfig ArtifactIdentity,
  canonical explicitly selected preexisting Evidence references)

InvocationOccurrenceRef = (DseRunKey, uint64 occurrence_ordinal)
```

The key's family owns domain framing, dependency order, and version. Derived
views are reconstructed from ResolvedConfig and are not serialized again into
the key preimage. Different occurrence ordinals distinguish retry, resume, or
another execution of the same semantic closure without changing the formal
result.

The manifest records:

- occurrence and semantic-closure references;
- descriptors and verification digests for component views actually consumed;
- optional resume provenance and one controller outcome;
- canonical `MechanicalDerivation` and `CandidateDecision` records;
- selected or retained Artifact and Evidence references;
- owner-local planned/consumed work summaries;
- retained owner attempt/checkpoint references; and
- nonsemantic execution provenance.

Component digests are verification copies, not configuration owners. Work
summaries do not copy budget limits. `CompletedSelection` records selected
Artifact references and satisfied Evidence. `CompletedNoFeasibleCandidate`
records an empty selection and completed plan. `Incomplete` records unsatisfied
obligations and retained finalized material but no formal selected output.

Each recoverable logical unit uses a stable derived key:

```text
WorkUnitKey =
  PlanNodeRef
  + owner-local WorkUnitDescriptorRef
  + stable ordinal
```

The mutable ExecutionJournal may record starts, finalized output references,
attempt references, and checkpoint references by WorkUnitKey. Physical journal
event order has no semantic meaning. The Journal cannot own a current-best
answer, override an Artifact or Evidence, replace the resolved plan, or publish
a checkpoint as formal selection.

Resume recomputes the run key and resolved plan, verifies closure and each
owner schema, revalidates Artifact preimages and Request/Evidence references,
and reuses only fully finalized outputs bound to the expected WorkUnitKey.
In-flight work is safely retried with its original ordinal. Resume cannot
renumber work, consume the same logical slot twice, substitute another
candidate, or complete from best-so-far state.

Attempts and checkpoints remain owner-specific. Evaluation uses its
request-local attempt record; ToolRunner retains scripts, stdout, stderr, raw
reports, and process outcome in attempt or scratch material; PnR, training, and other
domains define typed checkpoints only when real recovery requires them. A
checkpoint binds the exact run key, occurrence, plan node, WorkUnitKey, owner
schema, and version. There is no generic Attempt Artifact or all-domain
checkpoint payload.

## Candidate Generators

Candidate generation preserves domain semantics while using one central plan.
Generator capability is registered through a static typed descriptor rather
than a persistent Artifact:

```text
CandidateGeneratorDescriptor {
  generator_kind
  descriptor_schema_and_version
  implementation_semantic_identity
  typed_input_slot_descriptors
  typed_output_slot_descriptors
  resolved_generator_config_schema
  determinism_contract
  owner_local_work_unit_descriptors
  objective_and_evaluation_projection_slots
}

ResolvedCandidateGeneratorBinding {
  descriptor_ref
  exact_static_input_bindings
  resolved_generator_config
  objective_and_evaluation_projection_refs
}
```

Central `Generate` plan nodes connect exact static values or earlier
`PlanOutputRef` values to descriptor-owned input slots. The resolved node's
output slots are mechanically obtained from the descriptor's role, schema, and
cardinality contract.
A generator publishes normal domain Artifacts and candidate lineage, then
deduplicates by ArtifactIdentity. Compiler transformations, Mapping Actions,
hardware transformations, and model training remain owned by their respective
domains; the controller does not define a universal Action or mutable candidate
IR.

An external flow that preserves a new `HardwareImplementation` is a hardware
Candidate Generator even when the same process also emits reports. The new
implementation is finalized before an Evaluation observes it. An
`EvaluationModelDescriptor` never mutates or replaces its subject.

Hardware generation uses three descriptor-owned typed configuration roots:

```text
FabricTemplateConfig {
  template_identity
  template_version
  typed_parameters
}

FabricRewriteConfig {
  base_fabric_ref
  typed_structural_decisions
}

ImplementationFlowConfig {
  input_fabric_or_implementation_ref
  implementation_platform_ref?
  provider_bindings[]
  occurrence_recipe_bindings[]
  typed_flow_decisions
}
```

`FabricTemplateConfig` invokes the public ADG Builder expansion path and
produces a Fabric Artifact. `FabricRewriteConfig` produces another exact Fabric
Artifact because it changes architecture semantics or structure.
`ImplementationFlowConfig` preserves Fabric semantics while producing an
immutable HardwareImplementation child or initial RTL implementation. Each
descriptor owns a closed schema for its typed parameter and decision records;
there is no generic property bag, hardware action language, mutable candidate
IR, or evaluator-owned rewrite.

The central plan may compose and rank these generators, but it does not copy
their semantics. Builtin search ranges and heuristics are resolved generator
configuration, not new persistent schema families. Candidate outputs are
deduplicated by their normal Fabric or HardwareImplementation identity.

## Integration Boundaries

### Mapping

Mapping and Evaluation meet through `CostVector = (V, G, Q)`:

- Mapping owns `V`, the closed typed set of temporary closure violations
  recomputed from Fabric contracts and Mapping selections.
- Mapping owns `G`, domain-independent PnR costs derived from topology,
  connectivity, routes, occupancy, distance, and generic congestion.
- Evaluation owns `Q`, accelerator-aware metrics and findings such as latency,
  throughput, timing, memory performance, area, power, and energy.

Structural invalidity is rejected directly. Mapping does not copy `Q`, and
Evaluation does not copy Mapping legality. Central resolved policy projects
`V`, `G`, and `Q` into ranking, search energy, reward, and quality gates. A
finalizable Mapping has no remaining `V`; failure of a quality gate over `Q`
does not become Mapping illegality.

PnR may use an ephemeral domain-specific incremental adapter for hot probes.
Its full model remains the oracle, its cache is removable, and probes create no
Request or Evidence. Any finalized candidate that starts an authorized external
evaluation uses the ordinary Request/Evidence boundary and retains raw material
only in owner-attempt or scratch state until its exact Artifact owner exists.
Evaluation-derived route guidance may order proposals, but cannot
change legal arcs, prove legality, replace complete `Q`, or enter a Mapping
Artifact.

### Model Parameters and Training

Training is separate candidate generation:

```text
ModelTrainingRequest Artifact
  -> ModelTrainer
  -> ModelParameterBundle Artifact
```

The `ModelTrainingRequest` is the sole training input root. It binds exact
immutable Evidence plus deterministic trainer identity, configuration, and
seed. Evidence has no detailed-material closure. A trainer that requires an
execution or, after its Artifact owner exists, a persistent raw payload must
receive it through a
descriptor-owned typed Request slot included in canonical Request bytes; it
must not accept a hidden side input. Training produces a new immutable
parameter bundle with provenance; parameters never mutate in place. Candidate
bundles are evaluated
through ordinary Request/Evidence on fixed validation cases and selected by the
same central DSE policy, budget, cache, and lineage rules. A new online epoch is
a new bundle and binding. Updating a released baseline is a separate explicit
action, not a side effect of Evaluation or training.

## E-EXIT Evaluation and DSE Closure

The closed core has one owner for every semantic fact:

- Request and Evidence own Evaluation input and normalized output;
- the case-signature registry owns model-independent subject, workload,
  runtime-input, and base-condition shape;
- descriptors and bindings own capability and one exact model selection;
- metric and finding registries own query semantics and scope forms, while
  each Artifact family owns its imported local target references;
- the condition registry owns condition payload, location, assignment-key, and
  canonicalization semantics;
- `ObjectiveDimension`, ordering, and CNF gates own optimization policy;
- `PlanOutputRef` owns all typed plan use-def, while Generate and Promote own
  central candidate expansion, Evidence acquisition, and narrowing;
- owner policies own semantic work limits;
- domain Artifacts and Evidence own formal semantic results;
- InvocationManifest owns one immutable occurrence summary;
- ExecutionJournal owns only mutable recovery state; and
- owner-specific records own attempts, checkpoints, and raw material.

The design therefore has no `EvaluationResultArtifact`, partial Evidence,
`DseCandidateArtifact`, `EvaluationDagArtifact`, `DseResultArtifact`,
`PromotionPolicy`, workflow DSL, generic diagnostic/result bag, generic Attempt
Artifact, or Journal-owned current-best truth. New simulator, tool, finding,
metric, trainer, cache backend, or journal implementation work extends its real
owner without expanding the central semantic model.

## Conformance Anchors

Only these stable semantic anchors belong at this boundary:

- Different model descriptors referencing one exact case signature derive the
  same case key for identical role bindings, workload, runtime input, and
  conditions, while retaining distinct Request identities.
- An EvaluationRequest constructor given an explicit EvaluationCase rejects an
  exact-signature mismatch with the resolved model descriptor before
  projecting fields; it never silently rebinds those fields.
- A finding-only Request is valid when its descriptor declares the capability,
  and Completed Evidence returns one explicit result for every finding ordinal.
- Completed Evidence is exactly total over both request sets, while
  Unsupported, ExecutionFailed, and CancelledOrTimeout carry only a typed
  OutcomeReason and no result arrays.
- EvaluationEvidence schema 1.0 rejects a `detailed_bundle_refs` field or any
  other generic raw-material reference until the exact bundle owner and a
  later schema minor define it.
- Multiple lineage paths to one Artifact deduplicate candidate Evaluation, and
  replay or resume with the same run closure and stable work ordinals produces
  the same formal selection as uninterrupted execution.
- Template, Fabric rewrite, and implementation-flow generators preserve their
  distinct typed owners while the central plan composes and deduplicates their
  ordinary Artifact outputs.
