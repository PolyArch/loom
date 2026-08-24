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

Evaluation itself owns exactly two persistent semantic Artifact families:

```text
EvaluationRequest Artifact
  -> EvaluationModel black box
  -> EvaluationEvidence Artifact
```

There is no persistent `EvaluationResultArtifact`, partial-result artifact,
evaluator-produced model-result artifact, or evaluator-specific request/result
family. Descriptor-owned typed output Artifacts may cross the evaluator
interface, but their domain families, not Evaluation, own their schemas and
identity. A `ModelParameterBundle` is one such separate domain Artifact
produced by ordinary candidate generation, never an evaluator output family.
It may later be an Evaluation case subject or a typed model input. Queue
entries, running jobs, retries, attempts, and checkpoints are execution
records rather than semantic Evaluation artifacts.

An EvaluationRequest fixes one immutable problem and one exact resolved model
binding. EvaluationEvidence references that exact Request and is the only
persistent owner of normalized outcome, metric results, and finding results.
Both artifacts are immutable and have independent version spaces. Their shared
typed data model is the schema authority; canonical serialization is a
cold-path representation of that model, not another schema authority.

Raw execution material is deliberately outside normalized Evidence:

- a workload-running simulator owns its exact `SimulationExecution` artifact;
- owner-attempt or scratch storage retains scripts, logs, raw reports, and
  process material; and
- owner-specific attempt records own runtime provenance and retry history.

Evidence binds evaluator-produced semantic Artifacts through descriptor-owned
typed output slots. A workload-running simulator uses one
`SimulationExecution` output slot. `evaluation.evidence.1.0` has no raw-bundle
field or generic Artifact-reference escape hatch. Providers retain raw
material only through owner-specific attempt or scratch storage; it cannot
enter Request or Evidence identity.

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
        resolve(exact EvaluationCase, CaseArtifactResolution,
                ArtifactStore, BlobStore)
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
  verify_cross_role_compatibility(
      exact subject,
      complete subject bindings,
      CaseArtifactResolution,
      ArtifactStore,
      BlobStore)
}
```

Workload/runtime-input compatibility receives the same exact
`EvaluationCase`, `CaseArtifactResolution`, `ArtifactStore`, and `BlobStore`.
It may strict-import the orthogonal workload/runtime roots and their logical
payloads, but it cannot duplicate either Artifact family's codec.

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

The compatibility callback may strict-import an accepted subject through its
Artifact-family owner using the supplied stores. The imported view is
ephemeral and never enters the case schema or `CaseArtifactResolution`. The
`BlobStore` is present because a root such as `ModelParameterBundle` cannot be
strict-imported without validating its referenced logical payload. This
permits a role whose root schema is shared by several typed contracts to
validate the exact contract carried by each root without adding a generic
property path or copying owner fields into Evaluation.

`UniqueReferenceCycle` is an executable case-signature-owned projection, not a
boolean capability flag. `AbstractCaseCycle` is legal only when the signature's
semantics define one intrinsic tick, such as an abstract DFG cycle.
`ExactSubjectCycle` must be a canonical `SubjectTargetRef` anchored in the exact
case, resolve through its family-owned local-reference codec, and satisfy the
signature's declared reference-cycle type. Its resolver receives both stores
because the unique cycle may depend on a strict-imported logical payload such
as Deployment. The resolved variant must match the descriptor's declared
`source`. Resolution failure, a foreign anchor, a noncanonical local reference,
or more than one possible result is invalid.
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
| mapped RTL simulation | HardwareImplementation, Deployment |
| physical implementation analysis | HardwareImplementation |
| system simulation | Deployment, Gem5 Simulation Binding |

Evaluation registry schema 3.0 owns every case, model, MetricKind,
FindingKind, condition-kind, capability-enum, scope-form, and related registry
ordinal. It registers these exact case signatures used by the pre-Mapping,
DFG-simulation, FPA, and system-simulation flows described here:

Registry 3.0 is the exact current catalog. Its case and model descriptors bind
the current Fabric, ConfigurationABI, occurrence-scoped HardwareImplementation,
Deployment, and runtime contracts. Exact 2.0 and 2.1 references are unsupported;
they are never cloned onto the current catalog or reinterpreted as 3.0. The
`evaluation.request.1.0` and
`evaluation.evidence.1.0` root record shapes remain unchanged because they
already carry exact versioned descriptor and Artifact references; newly
constructed roots use registry-3.0 refs.

| Case kind | Stable spelling | Ordered roles | Workload/runtime input |
| --- | --- | --- | --- |
| 0 | `structured_program_with_fabric` | `0: Structured Program Candidate`, `1: Fabric` | both required; the workload owns the exact source Structured Program and the runtime input reaches that workload |
| 1 | `canonical_dataflow_with_fabric` | `0: Canonical Dataflow Program`, `1: Fabric` | both forbidden |
| 2 | `structured_program_functional_comparison` | `0: selected Structured Program Candidate` | both required; the workload owns the exact source Structured Program and the runtime input reaches that workload |
| 3 | `canonical_dataflow_simulation` | `0: Canonical Dataflow Program` | both required; the workload is Spatial, owns the exact Canonical Dataflow Program, and the runtime input reaches that workload |
| 4 | `fpa_model_parameter_calibration` | `0: exactly one Model Parameter Bundle with an FPA prediction view`, `1: one or more completed ground-truth Evaluation Evidence roots` | both forbidden |
| 5 | `hardware_implementation_physical` | `0: exact loom.hardware_implementation 4.1` | both forbidden |
| 6 | `system_simulation` | `0: Deployment`, `1: Gem5 Simulation Binding` | both required; the workload and runtime input are System roots coupled to the exact Deployment |
| 7 | `cgra_simulation` | `0: Canonical Dataflow Program`, `1: Fabric`, `2: SpatialMapping` | both required; the workload is Spatial, owns the exact Canonical Dataflow Program, and the runtime input reaches that workload |
| 8 | `simulation_execution_comparison` | `0: reference SimulationExecution`, `1: candidate SimulationExecution` | both forbidden; each execution's exact Request closure must resolve the same workload and runtime input |
| 9 | `canonical_dataflow_source_functional_comparison` | `0: Canonical Dataflow Program`, `1: selected Structured Program parent` | both required; the workload owns the exact source Structured Program and the runtime input reaches that workload |
| 10 | `fabric_hardware_analysis` | `0: Fabric` | both forbidden |
| 11 | `system_runtime_model_parameter_calibration` | `0: exactly one Model Parameter Bundle with a System Runtime prediction view`, `1: one or more completed ground-truth Evaluation Evidence roots` | both forbidden |
| 12 | `mapped_rtl_simulation` | `0: exact loom.hardware_implementation 4.1`, `1: Deployment` | both required; the workload and runtime input are Spatial roots, and the Deployment resolves their exact Dataflow launch to the exact SpatialCore occurrence implemented by role 0 |

The matching initial model descriptors are:

| Model kind | Stable spelling | Case kind | Capability |
| --- | --- | --- | --- |
| 2 | `structured_fabric_low_confidence` | 0 | deterministic Analytic point estimates for whole-case Runtime, limiting clock frequency, total area, dynamic power, and leakage power |
| 3 | `canonical_dataflow_fabric_low_confidence` | 1 | the same deterministic Analytic metric set for a Canonical Dataflow/Fabric pair |
| 4 | `structured_program_functional` | 2 | deterministic Simulation result for the whole-case `functional_mismatch` finding only |
| 5 | `dfg_simulator` | 3 | deterministic Simulation of the exact Spatial workload, with one `SimulationExecution` output and exact whole-case CycleCount |
| 6 | `fpa_model_parameter_calibration` | 4 | deterministic Analytic calibration-error quantiles over one exact FPA parameter bundle and one exact ground-truth Evidence set |
| 7 | `structured_fabric_calibrated_fpa` | 0 | deterministic Analytic point estimates for limiting clock frequency, total area, dynamic power, and leakage power using one exact FPA parameter bundle |
| 8 | `canonical_dataflow_fabric_calibrated_fpa` | 1 | the same calibrated FPA predictions for one Canonical Dataflow/Fabric pair |
| 9 | `cgra_simulator` | 7 | deterministic Simulation of the exact mapped Spatial workload, with one `SimulationExecution` output and exact whole-case CycleCount |
| 10 | `simulation_execution_comparison` | 8 | deterministic comparison of compatible execution observations for the whole-case `functional_mismatch` finding |
| 11 | `canonical_dataflow_source_functional` | 9 | deterministic Simulation comparison of one Canonical Dataflow candidate and its selected Structured parent against the workload-owned source, producing only the whole-case `functional_mismatch` finding |
| 12 | `cadence_voltus_static_rail` | 5 | deterministic ToolMeasurement point observation of whole-case `MaximumVoltageDrop` using the shared static rail-analysis contract |
| 13 | `fabric_low_confidence` | 10 | deterministic Analytic point estimates for limiting clock frequency, total area, dynamic power, and leakage power over exact hardware structure and admitted activity conditions |
| 14 | `fabric_calibrated_fpa` | 10 | deterministic Analytic predictions for the same four FPA metrics using one exact FPA parameter bundle |
| 15 | `system_runtime_model_parameter_calibration` | 11 | deterministic Analytic `RuntimePredictionError` quantiles over one exact System Runtime parameter bundle and one exact gem5-CGRA ground-truth Evidence set |
| 16 | `gem5_cgra_system_runtime_predictor` | 6 | deterministic Analytic point prediction of whole-case Runtime using one exact System Runtime parameter bundle |
| 17 | `gem5_system_dfg` | 6 | deterministic Simulation of the exact System workload with DFG SpatialCore participants and one `SimulationExecution` output |
| 18 | `gem5_system_cgra` | 6 | deterministic Simulation of the exact System workload with mapped CGRA SpatialCore participants and one `SimulationExecution` output plus whole-case Runtime |
| 19 | `gem5_system_rtl` | 6 | deterministic Simulation of the exact System workload with mapped RTL SpatialCore participants and one `SimulationExecution` output plus whole-case Runtime |
| 20 | `openroad_routed_static_fpa` | 5 | OpenROAD ToolMeasurement point observations of limiting clock frequency, total area, dynamic power, and leakage power for one exact routed implementation target |
| 21 | `mapped_rtl_simulator` | 12 | deterministic external Simulation of the exact Spatial workload on one mapped RTL implementation, with one `SimulationExecution` output and exact whole-case CycleCount |

Model kinds 2, 3, and 13 consume the exact shared low-confidence config-view
contract. Model kinds 4, 5, and 6 each consume a distinct zero-field config
view because their semantics are fixed by their descriptor, case, and, for
kind 6, the bundle's exact parameter contract. Model kinds 7 and 8 each have
one `ExactlyOne` model-input slot for the exact initial FPA parameter contract
and use a zero-field config view. Model kind 14 consumes that same contract for
the hardware-only case. Model kinds 15 and 16 consume the exact System Runtime
parameter contract. Model kinds 17, 18, and 19 are separate descriptors rather
than values of a bridge-private engine enum; none may fall back to another
fidelity. Kinds 17 and 18 use distinct zero-field config views. Kind 19 uses
the same exact HDL-simulator config-view contract and
`MappedRtlSimulatorBinding` owner as kind 21. This reuse selects only the HDL
compiler build; it neither adopts kind 21 Evidence nor changes kind 19's
System case, Request, importer, or gem5 time authority. Model kind 20 is the
initial exact FPA ground-truth descriptor; its
resolved config view names one OpenROAD provider build rather than treating all
physical Evidence as interchangeable. The exact descriptor owns the routed
static-analysis and report-normalization contract.
Model kind 21 is an `ExternalPrepareImport` descriptor whose resolved config
view names one exact HDL simulator build. It admits only a target-independent
or platform-specialized `Rtl` representation root whose complete
HardwareImplementation is selected by the exact Deployment. A GateNetlist or
another HDL fidelity requires another descriptor rather than being silently
reinterpreted under kind 21.
Implementation flow, library cohort, Fabric structure, and operating conditions
remain typed features. Another physical provider registers another exact model
descriptor. Physical execution limits remain
owner-attempt bindings and do not enter these views. Model kind 4 compares the
candidate's selected whole-program native execution with the source execution
owned by the exact workload/runtime pair. Exact Artifact identity is sufficient
to establish equality for the unchanged source baseline; all other candidates
must execute. Unsupported projection or execution capability produces typed
`Unsupported`, provider failure produces typed `ExecutionFailed`, and only
completed unequal observations produce `functional_mismatch = Present`.

Model kind 5 uses the DFG timing model's `AbstractCycle` as its exact case
cycle basis. A retired run publishes one `SimulationExecution`; whole-case
CycleCount is mechanically derived from that execution's launch-accepted and
graph-retirement progress anchors. Unsupported capability, execution failure,
or an attempt limit cannot publish a fabricated retired execution. The first
provider does not emit `Halted`, because no complete terminal-witness owner is
registered for it; a non-retired run without such a proof is an execution
failure rather than a guessed deadlock.

Model kind 9 uses case kind 7's exact SpatialMapping-rooted reference-cycle
projection. The cycle resolver returns the exact SpatialMapping subject root;
the case signature then derives the unique reference domain mechanically from
that Mapping, its exact Dataflow and Fabric owners, and the mapped launch. The
Mapping does not persist another clock selector. A provider may emit integer
CycleCount only when graph retirement lies on an integral reference-cycle
coordinate; otherwise that metric implementation is unavailable rather than
rounded. Model kind 10 has no cycle basis and compares no timing metric. It
requires exact workload/runtime-input identity through each execution's
Request closure and emits only the ordinary `functional_mismatch` result.

Model kind 11 uses a distinct zero-field config view. Its exact Canonical
Dataflow and selected Structured parent are both case subjects; neither is
recovered from provider-local cache state. It has the same source-backed
functional outcome boundary as model kind 4 and does not claim timing or
physical observations.

Model kind 21 uses case kind 12's Deployment-owned Spatial launch and
configuration closure. The Deployment must reference role 0, and its exact
Spatial Launch relation must resolve the workload's rooted launch and dense
coordinates to one `SpatialExecutionContextKey`, one SpatialMapping, and the
complete required HardwareConfigurationImage set. The implementation's Fabric
and ConfigurationABI must equal that Deployment closure. The provider executes
only the selected SpatialCore boundary; HostCore, InstructionCore, NoC, cache,
and external-memory execution remain inactive, so this remains a Spatial-only
environment. The case derives its unique reference cycle from the selected
SpatialMapping context. A nonintegral retirement coordinate makes CycleCount
unavailable rather than rounded. The external importer finalizes the ordinary
Spatial `SimulationExecution`; it cannot add an RTL execution subtype, infer a
configuration, choose another occurrence, or fall back to CGRA simulation.

Model kinds 13 and 14 have no workload and cannot publish `Runtime`. Dynamic
power requires an exact admitted Fabric-rooted activity condition; absence is
typed `Unsupported`, never a default toggle rate. Model kind 16 predicts the
runtime semantics of exact model kind 18 only. Model kinds 17 through 19 share
the workload-independent `Gem5SimulationBinding` and the same bridge ABI, but
the exact descriptor fixes the SpatialCore fidelity. The gem5 event queue is
their sole whole-system time authority. DFG execution may publish functional
observations without claiming CGRA or RTL timing; kind 18 is the initial
learned-runtime ground-truth target, and kind 19 remains a distinct higher-cost
fidelity. Kind 19 compiles and runs the same Deployment-selected mapped-RTL
closure as kind 21 under its exact `MappedRtlSimulatorBinding`. The
Gem5SimulationBinding independently owns the exact gem5 build. An invocation
must verify both bindings and may not hide either executable behind an
unidentified command or reinterpret one build identity as the other.

Model kind 20 consumes and requires the exact physical Base-condition patterns
for `ProcessCorner`, `SupplyVoltage`, `Temperature`, and
`RequiredClockPeriod`. Its DynamicPower request additionally requires exactly
one `ActivityBinding`, either a complete compatible ExecutionActivity source
or the registered two-target explicit assumption. The initial descriptor
admits one global activity clock and no `RelativeClockSchedule`; multi-clock
analysis requires another exact model descriptor. Its one HardwareImplementation
must be a completed placed-and-routed closure whose exact Fabric and
ImplementationPlatform remain reachable. A provider semantic/build identity,
normalization contract, or fidelity change changes the descriptor-owned
ground-truth target key described below.
An analytical estimate, synthesis-only result, unmatched provider result, or
different fidelity cannot enter this model's training population merely
because it reports the same MetricKinds.

Its resolved config-view schema 1.0 contains exactly one canonical nonempty
`stable_provider_build_identity`. Machine-local executable, module, container,
license, scratch, and concurrency choices remain execution bindings. The
descriptor and build identity form the target-key provider portion; the exact
HardwareImplementation, ImplementationPlatform, routed-flow and library facts,
and Base conditions enter feature projection instead of the config view.

Case kind 5 initially permits the following HardwareImplementation-rooted
Base-condition patterns. Both positions of the explicit-assumption form name
the same sole subject role; their payload roles remain distinct and ordered:

```text
ProcessCorner                         -> [HardwareImplementation root]
SupplyVoltage                         -> [HardwareImplementation root]
Temperature                           -> [HardwareImplementation root]
RequiredClockPeriod                   -> [HardwareImplementation root]
ActivityBinding.ExecutionActivity     -> [HardwareImplementation root]
ActivityBinding.ExplicitAssumption    -> [HardwareImplementation root,
                                           HardwareImplementation root]
```

Model kind 12 consumes and requires exactly `ProcessCorner`, `SupplyVoltage`,
`Temperature`, `RequiredClockPeriod`, and the two-target explicit
`ActivityBinding` pattern. Temperature and the required period target the same
global HardwareImplementation root as the supply and explicit assumption's
clock domain. Transition density is measured per that clock, so neither an
opaque SDC payload, a PGV library, nor a tool default may provide a second
absolute-frequency authority. A technology corner does not encode temperature,
so neither it nor a provider default may become a second thermal authority. Its
descriptor owns one fixed provider-neutral analysis contract:

```text
RailAnalysisModelConfig {
  method: Static
  activity_basis: ExplicitAssumption
  network_coverage: CompleteAnalyzedNetwork
  uncertainty: ExactWithinModel
}
```

This provider-neutral config does not duplicate a provider's named solver
mode. The registered implementation semantic identity owns any fixed
provider-specific algorithm selection that is not a portable model dimension.
For `loom.eda.cadence.voltus.rail@1`, that selection is Voltus
high-definition rail accuracy. The adapter emits it explicitly; an ambient
tool default cannot select it. Changing that selection requires a new
implementation semantic identity rather than a mutable invocation option or
another field in `RailAnalysisModelConfig`.

Its resolved config-view schema 3.0 consumes exactly one typed provider binding
from ResolvedConfig 6.0:

```text
CadenceVoltusStaticRailProviderBinding {
  stable_provider_build_identity
  power_grid_library_members:
    canonical nonempty array<relative_path, sha256>
  power_grid_library_entrypoints:
    ordered nonempty array<relative_path>
}
```

The provider build, PGV member table, and exact technology-first entrypoint
order affect the result and therefore enter the exact `ResolvedModelBinding`
and Request identity. Every entrypoint must reference one member; fingerprints
remain owned only by the member table. Machine-local PGV paths and keys remain
outside the binding. An absent provider binding makes model projection
unavailable; a projector cannot choose a configured tree or infer PGV roots
from member names.

The corresponding provider configuration is derived only from that typed
provider binding, descriptor contract, and validated Request conditions. The
initial model admits one global applied supply, one global temperature, and one
global activity clock
with one exact period for an always-on implementation. A multi-supply,
multi-temperature, multi-clock, partial-network, or execution-activity case is
typed `Unsupported`; a provider cannot select one domain, invent a nominal
voltage, temperature, or clock period, or reinterpret a partial network as the
whole-case metric. Supporting another method or activity basis requires
another exact model descriptor and config-view contract, not a mutable
invocation flag.

Case kinds 0, 1, and 10 admit the exact Fabric-anchored target patterns owned by
`ProcessCorner`, `SupplyVoltage`, `Temperature`, `RequiredClockPeriod`,
`RelativeClockSchedule`, and `ActivityBinding`; their model-input closure must
contain any referenced ImplementationPlatform. Case kind 5 admits the
corresponding HardwareImplementation-anchored patterns and has no whole-case
cycle basis. Each model declares the exact permitted subset it consumes,
requires, or proves invariant. Case kinds 4 and 11 admit only their respective
MetricRequest-local `Quantile` condition. Case kind 6 is the single
shared system-simulation signature referenced by the Runtime ABI and Fabric
System contracts. Its exact Deployment/Gem5-binding compatibility and System
workload/runtime-input coupling are those owners' typed relations, not copied
fields in Evaluation.
Case kind 12 is the corresponding Spatial-only external-HDL signature. Its
HardwareImplementation/Deployment compatibility, unique mapped launch context,
and Spatial workload/runtime-input coupling are likewise verified through
those owners rather than copied into Evaluation.

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

Workload-aware compilation uses the same mechanism. An exact source-program
Request binds the `StructuredProgram` roots of `SimulationWorkload` and
`SimulationRuntimeInput` defined by the Simulation Artifact owner; its
`StructuredEntityRef` entry already binds the exact source Structured Program.
Evidence owns normalized dynamic observations. A compiler generator may
consume only the model descriptor's typed projection of those observations.
Fine-grained counters or traces needed while evaluating one invocation remain
removable execution state and do not create a profile Artifact, candidate
identity field, or alternate workload authority. The workload reference and
runtime-input reference in the Evaluation case are the sole input identity
shared by the source baseline and every Structured candidate. Candidate-
specific Spatial workload/input pairs are derived only for graph replay and
cannot become the promotion key.

Promotion of a Spatial ownership candidate has two independent gates. The
semantic gate requires source-program equivalence, selected Structured-program
equivalence, and exact graph replay equivalence for every observed dynamic
activation. The benefit gate compares whole-workload metrics against the
stored-program baseline. Passing one gate cannot imply the other. A host-only
selection is a valid DSE result only when complete candidate dispositions and
workload-aware Evidence justify it; it is not an accelerator-success result.

Pre-Mapping ownership selection exposes two spellings at its API boundary.
`SemanticConformance` is the implemented bounded planner and is
feasibility-only; it makes no QoR-optimality claim. `BenefitQualified` remains
an accepted legacy request spelling, but the current planner resolves it to
the same bounded semantic frontier and records both requested and resolved
mode in invocation provenance. It must not be described as having applied a
benefit gate or Dataflow rewrite search. QoR comparison belongs to the later
bounded-quality owner, where every compared point is a verified
SystemMapping. Multiple explicit protocol roots remain parent-local
alternatives and are never greedily composed.

For `SemanticConformance`, one shared source observation projects an exact
protocol-root activity count and a conservative direct dependency relation.
A dependency exists only when two protocol roots are direct calls in the same
source block, the earlier root may write a derived SSA memory object, and the
later root may read that same object. Each relation records the number of such
objects, the sum of statically known object extents in bytes, and the number of
objects whose extent is unknown. A dynamic extent remains explicitly unknown;
it is never guessed. Indirect calls, cross-block order, unknown aliasing,
channel traffic, and downstream Mapping legality are outside this source
projection and cannot be inferred from an absent edge.

The controller first creates an invocation-local spectrum of typed
`PreMappingCoordinate` ownership points: full-root and partial-root coverage,
root-local schedule intents, high-activity and ordinary singleton regions,
proven dependency edges, producer/consumer groups, connected components, and
deterministic canonical prefixes. These coordinates are intentionally
unclassified. A full root set is compatible with either a spatial schedule or
a temporal schedule, and a singleton is not temporal merely because its
logical domain has one point. The materialized logical-domain fact records
epochs, launch/synchronization lower bounds, occupancy, and live state, but it
does not classify an endpoint. `MaxSpatial`, `MaxTemporal`, and `intermediate`
are result classes that may be written only after a real SystemMapping has
been imported and its explicit event-relative active sets and per-region
resource allocations have been verified. An endpoint request without such a
schedule is typed `Unsupported`; it cannot be satisfied by selecting another
seed or by increasing TopK.
The planner uses adjacency lists and incremental candidate projections; it
never enumerates an arbitrary root powerset. A resolved
`PreMappingFrontierPolicy` limits coordinate generation, Structured-program
materialization, analytic evaluation, functional replay, Dataflow promotion,
and Mapping-pair admission separately. Each limit is checked before the
corresponding expensive call. Exhausting a limit records a typed incomplete
disposition rather than an infeasibility result.

The downstream `JointDesignPolicy` independently bounds the software and
System frontiers, admitted software/System pairs, TechMapping candidates per
target Module, and retained SpatialMappings per pair. The TechMapping bound is
represented by the ordinary `BoundedPlanOutputJoin` between the Tech and
Spatial Generate nodes. Its producer bound reaches the Tech generator before
candidate publication, and its retained bound reaches the root-complete
Spatial generator before any Spatial PnR invocation. Consequently a small
Spatial output limit cannot hide an unbounded sequence of discarded
TechMapping inputs. Any truncated producer remains typed plan incompleteness;
it is never reported as Mapping infeasibility.

`SemanticConformance` expands each coordinate through a bounded hierarchical
beam. At every expansion depth the central Objective/Pareto/TopK owners rank a
small parent frontier and retain objective-best, minimum-coverage, and
maximum-coverage representatives before the next transform layer. Ownership,
execution shape, schedule, memory/channel, and special-math derivations remain
ordinary candidate-generator outputs; the frontier controller only owns
coordinate identity, fanout, and budgets. Thus a locally inferior parent may
survive when it is a distinct beam representative, while unrestricted Cartesian
composition is still forbidden.

The ownership provider keeps three bounds distinct. Its resolved Generate
binding carries a positive decision-materialization attempt grant reserved by
the enclosing frontier; its plan-derived output demand limits distinct child
Artifacts published to analytic evaluation; the later Promote node owns the
smaller terminal functional-replay width. Exact rejection of an early decision
therefore consumes an attempt but not a publication slot. A publication or
attempt truncation remains `SemanticLimitReached`; neither one proves the
unseen domain infeasible. Special-math choice/mechanical closure uses the same
separation between its materialization grant and published child bound.

Only a terminal beam survivor is functionally replayed. A survivor retained
solely as the immutable parent of another transformation layer is an analytic
continuation: it may contribute typed transformation lineage and a model
estimate, but it has no functional result and cannot be published to Mapping.
When a later terminal Promote selects that parent or one of its descendants,
the functional owner executes or exactly reuses one replay within the already
reserved terminal slot. Persisted finding-only Evidence is not a substitute
for the transient replay cases needed by application materialization; the
invocation retains those cases explicitly after the functional owner supplies
them.

Every candidate record contains its exact owned roots, seed/lineage, host and
accelerator region counts, known and unknown internal/cut bytes, activity,
fanout, channel opportunities, analytic estimate, and support/confidence. A
candidate that reaches bounded Dataflow materialization additionally carries
its exact Dataflow root, logical-domain/core lower-bound projection, actor and
edge inventory, and transport inventory. Unmaterialized candidates leave those
fields absent rather than guessing them.
Dependency knowledge is typed as proven-present, proven-absent, or unknown;
unknown alias/order is never treated as zero cut cost. If a provider cannot
complete, the record retains the original typed plan reason (`Unsupported`,
`ProofNotEstablished`, `ExecutionFailed`, `SemanticLimitReached`, or
`CancelledOrTimeout`) instead of rewriting it as a generic budget disposition.
The shared source
projection is computed once and joined by exact root identity. Heuristic
estimates can rank or promote a candidate, but only a sound necessary-condition
failure can set an exact rejection disposition; unsupported, timeout, and
`ProofNotEstablished` remain distinct.

Search completion is a product of independent facts, not a synonym for “the
current beam returned a candidate.” Invocation provenance serializes
`domainComplete`, `budgetComplete`, `providerComplete`, `evidenceComplete`,
and `selectionComplete`. A timeout, interrupt, provider incompleteness,
unknown dependency, heuristic truncation, or unsupported estimate keeps the
corresponding fact false. An incomplete result always publishes an invocation
checkpoint binding source/Fabric/workload/runtime identities, the frontier
policy digest, eligible-coordinate count, truncation state, the
source-observation ledger plus the six candidate-frontier work ledgers,
completeness facts, and retained candidate roots. Unconsumed ledger
units are classified as rejected for semantic/provider limits and cancelled
only for an observed cancellation or deadline; the checkpoint never turns a
budget stop into a timeout.

This is a dependency-aware bounded planning heuristic, not a complete software
partitioner or a proof of QoR optimality. `FirstVerified` reports time to first
feasible mapping. `BoundedQuality` evaluates a fixed, reproducible frontier of
current-hardware, hardware-reopen, and spectrum alternatives, then applies the
central QoR objective to the verified application outcomes. The original source
workload and runtime input remain the sole functional oracle. The application
owner carries the planning inventory and joins each downstream attempt by its
exact planning-record ordinal. That separate outcome row names the Dataflow,
System, verified SystemMappings, typed disposition, and any incomplete plan
node/reason. It does not copy those results back into the planning record, so
prefilter recall, work accounting, and time-to-best can be audited without
creating a second candidate identity.
Each materialized planning record also carries a candidate identity digest
derived only from its semantic coordinate and transformation lineage:
Structured/Dataflow roots, owned protocol roots, and the immutable candidate
projection. Source, Fabric, workload, runtime, resolved policy, estimates,
schedule hints, temporal witnesses, rank, disposition, and cache state are
invocation evaluation provenance and cannot alter candidate identity. Mapping
and application diagnostics must repeat and validate the semantic digest while
carrying that provenance separately; the ordinal is only an invocation-local
index and is never the cross-layer identity.
The BoundedQuality policy also carries one non-empty, unique label for every
objective dimension. The labels are invocation provenance for interpreting the
objective codes; they do not define ordering or introduce a second objective
catalog. A complete observation must provide one objective vector with the
declared arity, while an unsupported, failed, or cancelled observation retains
the typed reason and no score.
After one selected Structured candidate lowers mechanically to D0, the
controller queries the exact Fabric capability index. If every D0 actor is
admitted, the controller retains that exact input directly as D*, instantiates
the ordinary functional Evidence obligation, and does not create a Dataflow
rewrite Generate invocation. If D0 has a first inadmissible actor, the
controller invokes the Dataflow rewrite generator to seek an admitted D*.

This feasibility shortcut does not alter the rewrite generator's domain. When
that generator is invoked, every fixed rule remains eligible for every popped
frontier Artifact, including an admitted Artifact, and vector decomposition is
added only for the first missing actor as specified below. A
`SemanticLimitReached` result retains every fully finalized candidate and may
feed those candidates to later plan nodes, while the generator domain remains
explicitly incomplete. A functional
mismatch on an admitted D0 produces no feasible semantic-conformance result;
semantics-preserving rewrites cannot be used as a fallback to repair it.

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

`MetricQuery` also uses `EvaluationScope`. Standalone query serialization is
owned by `evaluation.metric_query 1.0` and `evaluation.finding_query 1.0`;
those wire roots carry registry-3.0 references and do not own their ordinals.
The two request sets are independent. Their total cardinality must be nonzero
unless the selected model descriptor declares at least one output slot whose
`Completed` cardinality is `ExactlyOne` or `OneOrMore`. In that one case the
required typed output Artifact is itself the requested semantic result, so an
empty query set is canonical. A finding-only Request is legal. A descriptor
with no required completed output cannot use an empty query set.
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
- descriptor-owned model input slots, including exact accepted parameter
  contract refs where applicable, role-labeled typed output slots, and resolved
  model-config schema;
- one exact `ProviderForm`;
- one exact decimal-result finalization contract whenever any declared result
  can contain `DecimalValue`;
- modeled phenomena, execution method, and determinism contract; and
- exact optional domain-specific incremental and guidance interaction
  capabilities.

DSE provider boundaries share one closed form tag:

```text
ProviderForm =
    InProcess              // tag 0
  | ExternalPrepareImport  // tag 1
```

Evaluation registry schema 3.0 owns the following capability enums and retains
their stable zero-based `uint32` wire tags:

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

The closed decimal-result finalization contract is a positive significant-digit
count in `[1,18]` and `RoundToNearestTiesToEven`. It applies only after exact
model arithmetic and aggregation have selected the mathematical result. A
descriptor that can emit `DecimalValue` without this contract, or that carries
the contract while no result can contain a decimal, is invalid. This field is
part of descriptor semantic identity; evaluator implementations cannot choose
host precision or another rounding mode.

The static descriptor's canonical capability projection encodes the provider
form first as one `u32be(tag)`, the phenomenon set as `u64be(count)` followed
by its sorted `u32be(tag)` values, and the execution method as one
`u32be(tag)`. C++ registration uses the closed enums, not strings or raw
integers. Human-readable spellings are registry-derived and never an identity
input.

The same projection encodes decimal finalization as `u32be(0)` when absent or
`u32be(1) || u32be(significant_decimal_digits) || u32be(0)` when present; the
final zero is the registry-schema-2.0 tag for
`RoundToNearestTiesToEven`. Unknown tags or
a presence state inconsistent with the descriptor's result capabilities are
registry errors.

Full Evaluation has no separate domain field. Its full domain is operationally
the exact set of Requests accepted by `RequestVerifier`. That verifier
intersects the descriptor's case signature, condition capabilities, metric and
finding capabilities, input and output slots, resolved config view, mandatory
terminal findings, determinism, and replicate rules. Copying a second full-
domain description into either provider form would be another authority.

For each exact descriptor reference, the provider implementation registered in
the process must match the descriptor's `ProviderForm` exactly:

```text
EvaluationProviderImplementation =
    InProcess {
      evaluate(EvaluationRequest,
               CaseArtifactResolution,
               ArtifactStore,
               BlobStore) -> Expected<EvaluationModelResult>
    }
  | ExternalPrepareImport {
      prepare(EvaluationRequest,
              CaseArtifactResolution,
              ArtifactStore,
              BlobStore,
              ExternalToolPreparationContext)
        -> Expected<EvaluationPreparationResult>
      import(EvaluationRequest,
             CaseArtifactResolution,
             PreparedExternalToolInvocation,
             ArtifactStore,
             BlobStore) -> Expected<EvaluationModelResult>
    }

EvaluationModelResult {
  output_bindings:
    dense array<ModelOutputSlotRef, canonical ArtifactRootReference collection>
  outcome: EvaluationEvidenceOutcome
}

EvaluationPreparationResult =
    PreparedExternalToolInvocation
  | Unsupported { reason: RuntimeCapabilityUnavailable }
```

`EvaluationModelResult` is a transient provider return, not another persistent
result schema. Its output slots and result arrays obey the same descriptor and
Request cardinality rules as `EvaluationEvidence`; the Evaluation owner alone
validates it, binds the exact Request, and finalizes the persistent Evidence
root. The provider cannot add another outcome kind or omit the dense output
slot array.

The exact finalized `EvaluationRequest` is the external form's complete
semantic closure. `RequestVerifier`, case-signature callbacks, and Artifact-
family importers remain the admission owners; `prepare` invokes those owners as
needed and adds only descriptor-specific consumption plus local invocation
preflight. When the exact valid Request is outside the provider's stable
capability, `prepare` returns only the existing typed
`Unsupported(RuntimeCapabilityUnavailable)` result. The Evaluation owner
mechanically supplies the descriptor's dense empty output bindings, validates
the result, and finalizes Evidence; the provider does not create Evidence. No
script, bundle, attempt, or completion record exists on that path.

Otherwise `prepare` materializes one finalized bundle but does not execute a
process, publish an output Artifact, or publish Evidence. `import` accepts the
same Request and exact prepared bundle only after strict completion validation.
It first finalizes every descriptor-owned output Artifact, then returns one
`EvaluationModelResult`. The Evaluation owner validates that result and
finalizes `EvaluationEvidence`; the provider never writes Evidence directly.
Already finalized but unreferenced output roots may remain after a later
Evidence-publication failure, but no partial output binding or Evidence is
published. `ExecutionFailed`, `CancelledOrTimeout`, and completed results cannot
be produced by preparation because each requires a real attempted execution.

The caller alone decides whether, where, and when to execute `run.sh`. The
ordinary `evaluateRequest` facade applies only to `InProcess`; a synchronous
external CLI is an explicit composition of prepare, caller execution, and
import rather than an implicit callback behavior. Changing an existing
descriptor's provider form requires a new descriptor version or reference; a
registry cannot reinterpret an exact ref.

Every owner-local static typed registry reference in this document uses one
canonical key framing:

```text
u64be(owner_registry_identity byte length)
|| exact owner_registry_identity bytes
|| u32be(owner_registry_version.major)
|| u32be(owner_registry_version.minor)
|| u32be(owner-local kind)
```

The owner identity is canonical nonempty ASCII. Each concrete reference type
names the semantic meaning of its final owner-local kind field; this shared
framing does not create a generic interchangeable reference type.

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

The complete domain-ref canonical key uses the shared owner-local registry
reference framing with `owner_local_domain_kind` as its final field.
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
compatibility. The same mechanism binds parameter bundles and role-labeled
upstream EvaluationEvidence without an unordered Evidence bag.

```text
ModelInputSlotDescriptor {
  slot_ref: ModelInputSlotRef
  accepted_artifact_schemas
  cardinality
  model_parameter_contract_ref: optional<ModelParameterContractRef>
  verify_compatibility(exact bound roots, exact EvaluationCase,
                       CaseArtifactResolution, ArtifactStore, BlobStore)
}
```

Compatibility receives the exact bound roots, the complete case resolution,
`ArtifactStore`, and `BlobStore`. A slot owner may strict-import a root and its
logical blob payload, but cannot copy either owner's codec. Evidence that
defines the ground-truth question belongs in an Evaluation case-subject role;
Evidence that supplies implementation material to a derived model belongs in
a descriptor-owned model-input slot. One Evidence root cannot change roles by
consumer convention.

A slot that admits `loom.model_parameter_bundle 1.0` references exactly one
registered `ModelParameterContractRef`. The slot does not own the parameter
codec or inference semantics. Request verification strict-imports every bound
bundle through both stores and requires its exact contract reference to equal
the slot's accepted contract. The descriptor registry rejects a bundle slot
without this reference
and rejects a parameter-contract reference on a slot that does not admit the
bundle schema. A raw-byte validator, untyped property map, consumer-owned
parameter codec, or schema-only compatibility check is not equivalent.

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
  outcome: EvaluationEvidenceOutcome
}

EvaluationEvidenceOutcome =
    Completed {
      metric_results: canonical array<MetricResult>
      finding_results: canonical array<FindingResult>
    }
    | Unsupported { reason: OutcomeReason }
    | ExecutionFailed { reason: OutcomeReason }
    | CancelledOrTimeout { reason: OutcomeReason }
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
and partial reports remain owner-attempt or scratch material. Timestamps, host
details, retry history, and
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

`evaluation.evidence.1.0` deliberately contains no `detailed_bundle_refs`.
The earlier generic
`canonical set<ArtifactRootReference>` shape was removed because no exact raw
detailed-bundle schema, root kind, canonical payload, or importer owns those
references. Keeping an always-invalid generic field would be a second authority
and permanent wire slop. Raw material therefore remains owner-attempt or
scratch state; this contract does not predefine a future bundle reference.

The `evaluation.request.1.0`, `evaluation.evidence.1.0`, and
`loom.simulation_execution 1.0` dependency direction is therefore:

```text
SimulationExecution -> EvaluationRequest
EvaluationEvidence -> EvaluationRequest + typed output Artifacts
```

A simulator that executes a workload retains the exact `SimulationExecution`
as a typed Artifact. It owns terminal
execution observations, output values and streams, visible logical-memory final
state or diffs, completion and retirement observations, typed activity
summaries. `loom.simulation_execution 1.0` contains no trace field; diagnostic
traces and waveforms remain attempt or scratch state. A simulator cannot
replace them with paths, opaque bytes, or provider-private references in the
Artifact. `SimulationExecution` contains no
normalized metrics, findings, Evaluation outcome, DSE decisions, or second
simulator result schema.

## Candidate Lineage and Evaluation DAG

A central DSE candidate is an existing exact `ArtifactRootReference`. There is no
`DseCandidateArtifact`, generic CandidateKind, or wrapper identity. A mutable
domain-local search state enters a central candidate set only after it is
finalized as its domain Artifact.

Invocation lineage has exactly two edge kinds. Facts fixed for one Generate
invocation remain in its enclosing record rather than being repeated on every
edge:

```text
CandidateGeneratorLineageContribution =
    MechanicalDerivation {
      output_slot
      output: ArtifactRootReference
    }
  | CandidateDecision {
      output_slot
      output: ArtifactRootReference
      parents: canonical set<ArtifactRootReference>
      owner_payload: canonical descriptor-owned bytes
    }
```

`MechanicalDerivation` derives its one output from the enclosing exact inputs
and resolved producer binding. `CandidateDecision` derives its one child from
its canonical parent set and the exact descriptor-owned typed decision bytes.

Each edge has exactly one output or child. A Generate invocation that returns
several Artifacts therefore does not become a multi-output lineage hyperedge.
Instead, one invocation occurrence record contains the exact typed input
bindings, one resolved candidate-generator binding, the exact output bindings,
and zero or more single-child lineage edges:

```text
GenerateInvocationRecord {
  plan_node_ref
  typed_input_bindings
  resolved_generator_binding
  output_bindings
  lineage_edges: canonical array<CandidateGeneratorLineageContribution>
}

OutputBinding {
  descriptor_output_slot
  canonical set<ArtifactRootReference>
}
```

`PlanNodeRef` is the unsigned 64-bit ordinal of a node in the exact resolved
plan. The enclosing `InvocationManifest` alone owns the
`InvocationOccurrenceRef`; a nested Generate record cannot repeat or override
it. One manifest contains at most one Generate record for each executed
Generate plan node.

Output bindings and lineage edges own different facts. An output binding owns
which Artifacts the invocation returned through one descriptor-owned slot. A
lineage edge owns one production step in a rooted derivation DAG whose returned
sinks occur in output bindings. A recursive generator may therefore retain a
durably published intermediate Artifact only as an internal lineage target and
parent; that Artifact does not become a returned plan value. An exact input
Artifact may be retained in an output binding without a fabricated self-edge,
while several distinct edges may target one deduplicated output Artifact.
Outputs in different slots have no positional pairing; any semantic dependency
between them belongs to the output Artifacts' own dependency closures.

Every completed or incomplete Generate record contains exactly one input
binding for every descriptor input slot and exactly one output binding for
every descriptor output slot. Both binding arrays follow dense descriptor slot
ordinal. An incomplete record retains an empty or partial canonical Artifact
set in a slot rather than omitting the slot. The maximum cardinality applies to
every record; a completed record also satisfies every minimum.

Every lineage edge names its exact descriptor output slot. Its target must
match that slot's exact schema, be independently and durably published, and
either occur in that slot's output binding or be a parent of another lineage
edge in the same invocation. Every CandidateDecision parent must be an exact
invocation input or another edge target. The resulting dependency graph is
acyclic; every edge is rooted in the exact invocation inputs or a
MechanicalDerivation, and every internal target reaches a returned output.
Every finalized output member that is not an exact retained input requires at
least one rooted lineage edge, including a member retained by an incomplete
invocation. Completely identical edges are deduplicated.
CandidateDecision parent references form a canonical set using Common's
root-reference order. Edges with different parents or canonical owner decision
bytes remain distinct; the producer binding is inherited from the enclosing
record and is not an edge field.

Generate records are ordered by `PlanNodeRef` rather than completion time and
duplicate refs are invalid. Artifact sets use Common's canonical root-reference
order. Lineage edges use edge-kind, output-slot, output-reference, canonical
parent-reference set, and canonical owner-payload bytes in that order; the
parent and payload fields are empty for MechanicalDerivation. Descriptor role,
schema, and cardinality fields are recovered from the exact descriptor and are
not copied into the manifest.

Each nested Generate record owns its completion boolean independently of its
retained output bindings. `ProofNotEstablished` and `SemanticLimitReached`
preserve fully finalized outputs and allow later plan nodes to consume them;
later independent or dependent Generate records therefore may follow an
incomplete record. Other Generate incomplete reasons stop plan traversal after
retaining that node's valid outputs. An interruption at a non-Generate node
does not create a Generate record for that node.

A completed manifest has only completed Generate records. An `Incomplete`
controller outcome names the blocking node when execution stopped. If all
incompleteness was retained candidate-domain incompleteness and plan traversal
finished, it names the first such Generate node in `PlanNodeRef` order. The
per-record booleans remain the complete authority for which Generate domains
were exhausted; the outer outcome is not used to infer a completed prefix.

A higher-level compiler selection may consume the completed downstream Promote
output of such an execution when it contains a verified incumbent. That
selection remains usable while its nested Generate records report the
non-exhaustive search. An empty downstream selection cannot become completed
infeasibility because the unvisited generator domain may still contain a
feasible candidate.

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
  incomplete PlanNodeRef
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

For an in-process Generate occurrence, the executor passes one non-owning
`ExecutionControlView` that mechanically combines the mutable journal stop bit
and the policy's absolute dispatch deadline. The view is not serialized and
does not enter run, invocation, provider, candidate, or Artifact identity.
Providers may observe it only at their own atomic boundaries. A provider that
observes it returns its domain-owned typed interruption with any finalized
outputs retained; the Generate adapter maps that outcome to
`CancelledOrTimeout`, and the execution journal records the work unit as
`TimedOut`. Providers that do not perform long-running bounded work may ignore
the view, but cannot reinterpret it as a semantic budget.

The Structured ownership generator uses one positive
`scope_expansion_limit` owned by its ResolvedConfig policy. Its finite scope
domain includes the mechanically derived nearest enclosing ownership-scope
relation. For one exact workload, scopes with zero dynamic activation are
removed by the descriptor-owned applicability projection. The remaining roots
form a deterministic priority frontier. Priority is descending dynamic
executable-leaf count, then descending activation count, then ascending
canonical scope ordinal. Popping a scope makes its direct children eligible;
therefore no child can consume a work slot before its parent. One popped scope
consumes one expansion slot and contributes its complete typed decision domain
or its one typed scope rejection.

Scopes not reached before the resolved limit are outside that invocation's
finite Generate domain. They receive no candidate disposition and no Evidence;
they are not mislabeled as infeasible or workload-inapplicable. Cache state and
parallel worker completion order cannot change the frontier or its stable work
ordinals.

The Dataflow rewrite generator uses the positive
`dse.dataflow_rewrite.scope_expansion_limit`. For each exact frontier Artifact
it enumerates the normalized decisions from the
[Canonical Dataflow Rewrite Catalog](spec-compiler-part-3-dfg.md#canonical-dataflow-rewrite-catalog)
in that catalog's exact canonical decision order. Each decision applies
exactly one match and produces at most one immutable child. Kinds 0 through 6
charge one expansion per attempted decision. An attempted-decision key is the
tuple of exact parent ArtifactIdentity, decision-schema identity and version,
and canonical decision payload bytes. Including the parent is mandatory
because every payload reference is parent-local. Attempted keys and visited
ArtifactIdentities terminate inverse cycles without suppressing a distinct
intermediate candidate.

All exact input Artifacts are ordered by complete ArtifactRootReference and
inserted once into one FIFO frontier. A previously unseen finalized child is
appended when its decision's logical slot is reached, even if workers build
several children concurrently. The generator pops one Artifact, classifies it,
enumerates and charges its decisions in catalog order, and only then pops the
next Artifact. A no-op consumes its attempted-decision charge but appends
nothing. This discovery order assigns all semantic work ordinals and cannot be
changed by worker completion, cache state, or artifact-store insertion order.

A Fabric-admissible candidate enters the retained output set. Fixed-rule
decisions remain eligible when every frontier Artifact is popped, including a
retained one. For a non-admissible candidate, vector-decomposition decisions
are additionally enumerated only for the first missing actor in canonical
Dataflow actor order, so Fabric affects candidate generation but never the
rewrite's software legality.

The provider for catalog and decision schema 2.0 has implementation semantic
identity `loom.compiler.dataflow_rewrite.generator.v3`. The existing
`loom.compiler.dataflow_rewrite.generator.v2` identity remains bound to the
incompatible decision-1.0 behavior and cannot be reinterpreted. Registry,
binding, manifest, cache, and lineage validation therefore distinguish the two
providers without a compatibility flag or cache invalidation exception.

The actor's typed decomposition domain is the one owned by
[Explicit Elementwise Decomposition](spec-dataflow-vectorization.md#explicit-elementwise-decomposition):
proper leading-dimension divisors in descending order, followed by the one
scalarization decision when legal. A chunk decision charges the number of
narrow compute actors it materializes. A scalarization decision charges the
fixed vector element count. Publication and exact-identity deduplication do not
refund work.

If the next canonical decision does not fit the remaining resolved budget, the
generator returns `Incomplete` with reason `SemanticLimitReached`. Already
admitted candidates are retained as fully finalized outputs and may be consumed
by later plan nodes. The generator domain is not exhausted, and the outer
controller remains incomplete even if those candidates later satisfy a
selection. Worker count, cache hits, artifact-store state, or completion order
cannot alter the frontier, decision order, charge, or outcome.

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

After every candidate has comparable completed objective Evidence, `TopK`
acquires deferred gate Evidence one candidate at a time in the resolved total
order until it has selected `k` independently verified candidates or exhausted
that order. A candidate-local Unsupported result does not prove that candidate
feasible or infeasible. The controller retains the typed incompleteness, skips
that candidate for incumbent selection, and continues with the next candidate.
If this produces a verified incumbent, the Promote output remains available to
downstream plan nodes while the plan outcome reports that search did not
complete. Without an incumbent, execution remains stopped and no empty output
can imply proven infeasibility. Provider-wide acquisition failure, incomplete
objective Evidence, ExecutionFailed, and CancelledOrTimeout still stop the
promotion; they cannot be attributed to and skipped as one candidate. This
bounded acquisition order does not weaken exact Evidence validation or any
independent verifier applied to a selected candidate.

Every candidate entering one comparison receives same-shaped Evidence
obligations. A missing result is never interpreted as zero, infinity, or worst
score. A candidate removed by an earlier resolved gate need not acquire later
expensive Evidence, but runtime timing and cache state cannot alter gate order.

An `AllPassing` node with a resolved `functional_mismatch` absence gate retains
exactly candidates whose corresponding completed Finding result is `Absent`.
`Present` is a completed adverse observation and removes that candidate;
`NotApplicable` does not satisfy an absence gate. Missing, Unsupported,
ExecutionFailed, or CancelledOrTimeout Evidence makes the promotion
`Incomplete` rather than silently removing or accepting the candidate. The
Evidence remains retained whether its candidate passes or fails.

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

CalibrationPartitionRole = Training | Validation | HeldOut

PlanValueRole =
    CandidateSet
  | EvidenceSet
  | SimulationExecutionSet

PlanValueCardinality = ExactlyOne | ZeroOrOne | NonEmptySet | FiniteSet

PlanInputSlotDescriptor {
  role: PlanValueRole
  accepted_artifact_schema: ArtifactSchemaDescriptor
  cardinality: PlanValueCardinality
  model_parameter_contract_ref: optional<ModelParameterContractRef>
  calibration_partition_role: optional<CalibrationPartitionRole>
}

PlanOutputSlotDescriptor {
  role: PlanValueRole
  accepted_artifact_schema: ArtifactSchemaDescriptor
  cardinality: PlanValueCardinality
  model_parameter_contract_ref: optional<ModelParameterContractRef>
  calibration_partition_role: optional<CalibrationPartitionRole>
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

`model_parameter_contract_ref` is required exactly when the accepted schema is
`loom.model_parameter_bundle 1.0` and is forbidden for every other schema. A
trainer output slot therefore declares its one produced contract without a
parameter-specific plan value role. Plan resolution compares that static
producer contract with every consuming bundle-slot contract before execution;
a mismatch is an ill-typed use-def edge, not a runtime training outcome.

`calibration_partition_role` is legal exactly for an `EvidenceSet` slot whose
accepted schema is `evaluation.evidence.1.0`; it is forbidden otherwise. A
produced Evidence partition and its consuming input must carry the same tag.
An exact static Evidence set acquires its role from the consuming input slot.
This slot fact supplies the ordinary use-def needed for partition readiness;
it is not copied into Evidence identity.

`CalibrationPartitionRole` has stable `u32be` tags in declaration order.
Optional slot refinements encode `u32be(0)` when absent or `u32be(1)` followed
by the referenced contract key or partition tag when present. Input and output
slot descriptors otherwise retain their existing role, schema, and cardinality
order. Unknown discriminants, tags, or a refinement forbidden by the slot
schema are invalid.

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
      purpose: CandidateSelection | ModelRelease
    }
```

Promote purpose uses stable tags `CandidateSelection = 0` and
`ModelRelease = 1`; it is part of the resolved plan's canonical node bytes.

An input binds either an explicit canonical set of exact static Artifacts or an
earlier `PlanOutputRef`. Resolution checks role, exact Artifact schema,
cardinality, parameter-contract compatibility when applicable,
calibration-partition compatibility when applicable, producer-before-consumer
ordering, and acyclicity. Independent nodes may execute in parallel, but
canonical inputs and node policy alone
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
produces a `CandidateSet` whose accepted schema is
`loom.model_parameter_bundle 1.0`. Evidence-to-candidate association remains
derived from `EvaluationEvidence -> EvaluationRequest -> candidate case
subject role`; the plan does not persist a parallel association map.

Acquisition policy owns obligation templates, replicate generation, and its
owner-local work limits. Selection policy remains `AllPassing`, `TopK`, or
`Pareto`. Ordinary nodes use `CandidateSelection`. `ModelRelease` is legal
only for `CandidateSet<loom.model_parameter_bundle 1.0>`, requires
`AllPassing`, may reference only held-out calibration obligations, and must be
a terminal consumer of its selected-candidate output. It is a closed purpose
on the existing Promote node, not a separate release action, mutable model
registry update, or generic workflow mechanism. There is no `PromotionPolicy`,
runtime loop, generic workflow DSL, or mutable workflow authority. Repeated
finite Generate and Promote nodes express cross-domain iteration.

Candidate-generator capabilities live in a static typed descriptor registry.
A resolved Generate node fixes typed inputs through the common plan bindings
and fixes owner-typed generator configuration through its generator binding.
The generator produces domain Artifacts and its provider result supplies
descriptor-owned output bindings. `InvocationManifest` records that outcome
and each owner-supplied `MechanicalDerivation` or `CandidateDecision` edge.
Its domain owns transformations, typed decision payloads, and local search
semantics; the central plan sees only canonical input and output Artifact sets
plus the closed lineage contributions admitted by the exact generator
descriptor.

The built-in root-complete TechMapping generator is one such descriptor-owned
composition. It consumes a finite set of exact Canonical Dataflow Artifacts
and one exact Fabric Artifact, derives each Dataflow root's complete canonical
graph cover through the Dataflow owner, and delegates candidate construction
to the TechMapping owner. This is not a central-controller default: an
independent graph scope remains an explicit TechMapping invocation. The plan
therefore carries only Artifact references, while owner-local `GraphRef`
values remain ephemeral and no graph-cover Artifact or resolved-config field
is introduced.

The built-in application-graph TechMapping generator is the corresponding
System-composition adapter. Its descriptor has kind 21, spelling
`mapping.application_graph_tech_mapping`, and implementation semantic identity
`loom.mapping.application_graph_tech_mapping.generator.v5`. Its exact input
slots are `dataflow: ExactlyOne`, `system_constraints: ExactlyOne`, and
`fabric: ExactlyOne`; the constraint root must bind the same Dataflow and a
System whose attached Module catalog contains the exact Fabric input. The
descriptor uses the same output slot, resolved TechMapping config view,
determinism, work catalog, and outcome algebra as the root-complete adapter.
It derives the unique graph set reachable from the constraint root's canonical
non-empty root-thread-launch set, then invokes the ordinary TechMapping owner
once per graph with a singleton cover. It returns the canonical union of the
resulting ordinary TechMapping Artifacts and mechanical lineage edges. A graph
proven infeasible contributes no candidate. A finite-prefix or incomplete
graph invocation retains its finalized candidates, marks aggregate search
completeness, and does not suppress later independent graphs. Invalid or
internal owner failure aborts the adapter invocation.

This adapter exists because hierarchical SystemMapping selects one
SpatialMapping for each `RootedGraphLaunchRef`; it must be able to compose
different graph definitions onto different AccCore occurrences. It does not
split a Canonical Dataflow Artifact, invent a graph-scope Artifact, reinterpret
root-complete results, treat an unselected callable definition as active, or
permit one TechMapping realization to cross a graph definition. The existing
System MappingConstraintSet remains the only persistent application Mapping
scope. A caller that needs another selected multi-graph cover still invokes
the ordinary TechMapping owner explicitly.

The built-in root-complete Spatial PnR generator composes the next boundary in
the same typed plan. Its implementation semantic identity is
`loom.mapping.root_complete_spatial_pnr.generator.v21`; the direct constrained
Spatial provider uses `loom.mapping.spatial_pnr.generator.v14`. It consumes the
finite TechMapping output and the same exact Fabric Artifact. Each `T` already
binds one unique Canonical Dataflow identity, so the descriptor strictly
recovers `D` from `T` instead of accepting a second `D` slot. It mechanically
publishes the exact empty Spatial
MappingConstraintSet for `D/T/F` through the constraint owner, then delegates
to the ordinary Spatial PnR owner with the descriptor's resolved Spatial PnR
config view. Its finite output contains only ordinary SpatialMapping
Artifacts. A constrained invocation remains a direct five-authority Spatial
PnR call; the central plan does not interpret absent constraints as empty and
does not acquire a constraint language, Mapping state, or search algorithm.

The root-complete adapter preserves the resolved completion goal. Under
`ExhaustConfiguredWork`, it may construct an initial routed candidate for each
prepared `T`, rank those candidates through the selected Spatial objective,
and traverse the resulting complete quality order. Under
`FirstVerifiedCandidate`, candidate quality outside the verified prefix is not
part of the requested result. The adapter therefore performs no speculative
routing for ordering. It repeatedly selects the prepared `T` covering the
greatest number of not-yet-covered graph references, breaks ties by canonical
Artifact reference, and invokes ordinary Spatial PnR. A graph becomes covered
only after that invocation publishes a finalized candidate; failure continues
to the next alternative. Traversal stops when every graph represented by the
input set has one verified SpatialMapping or when a typed limit, cancellation,
or proven-infeasible frontier prevents closure. Work accounting includes only
search actually executed for ranking or verification under the selected goal.

The built-in root-complete System PnR generator composes the final Mapping
boundary without widening the central plan. Its descriptor has kind 9,
spelling `mapping.root_complete_system_pnr`, schema
`loom.mapping.root_complete_system_pnr.generator.v11`, and exact input slots
`dataflow: ExactlyOne`, `spatial_mapping: FiniteSet`, and
`fabric: ExactlyOne`, `physical_timing_profile: FiniteSet`, and
`migration_seed: ZeroOrOne`. Its sole output slot is
`system_mapping: CandidateSet<loom.mapping 6.0>, FiniteSet`; its resolved view
is the exact System PnR component view. The explicit Dataflow input remains
necessary because an InstructionCore-only closure has no SpatialMapping from
which to recover `D`, and because `D` uniquely owns the complete root-launch
inventory.

For a nonempty root inventory the adapter publishes the exact empty System
MappingConstraintSet, projects the root-keyed cyclic partition requested by
the adopted System PnR config and the hierarchical System search-domain view,
and delegates to the ordinary System PnR owner. A
root-free Dataflow input completes with an empty output set. The descriptor
references the same complete PnR work-unit catalog used by the Spatial PnR
generator: seed attempt, assignment attempt per seed, endpoint expansion,
negotiation iteration, calibration proposal, proposal-per-level base,
proposal-per-movable-decision, exact-repair region decision, and exact-repair
solver call at ordinals 0 through 8. The ordinary
System PnR owner supplies those counts; the adapter neither aggregates nor
reclassifies them. A root-free invocation reports the same catalog with zero
planned and consumed work. Outputs carry MechanicalDerivation lineage; the
adapter owns no candidate decision payload. `ProvenInfeasible` completes with
an empty set, proof or semantic limits remain the corresponding typed
incomplete result, unsupported `H` projection remains `Unsupported`, and an
invalid or internal owner result aborts the Generate invocation. Descriptor
v1, which exposed only assignment and endpoint work, is not compatible and is
not registered.

The built-in application-scoped System PnR generator is the strict-scope
counterpart. Its descriptor has kind 22, spelling
`mapping.application_system_pnr`, implementation semantic identity
`loom.mapping.application_system_pnr.generator.v10`, and exact input slots
`dataflow: ExactlyOne`, `spatial_mapping: FiniteSet`, `fabric: ExactlyOne`, and
`physical_timing_profile: FiniteSet`, `system_constraints: ExactlyOne`, and
`migration_seed: ZeroOrOne`. The constraint root must bind exactly that
Dataflow and Fabric System. Its non-empty `root_thread_launches` is the sole
coverage root. The adapter projects the config-owned root-keyed partition and
hierarchical search domain from that exact set, supplies the finite
SpatialMapping candidates as the ordinary graph-search input, and invokes the
unchanged System PnR owner. Output, work accounting, lineage, incomplete
outcomes, and failures are identical to the root-complete adapter. It neither
copies the root set into config nor derives another scope from the Dataflow
catalog. The optional migration input in both descriptors is one exact
PnR-owned checkpoint seed; its admission and fallback semantics remain owned
by `docs/spec-pnr.md`.

The built-in SpatialMapping CGRA acquisition consumes a finite SpatialMapping
candidate set, a nonempty Canonical Dataflow owner set, one exact Fabric, one
exact Spatial workload, and one exact runtime input. For each candidate it
strictly imports the SpatialMapping, recovers its unique Dataflow and Fabric
owners through the Mapping owner, and task-locally selects that one Dataflow
from the already bound set. The recovered Fabric must equal the exact Fabric
input. The CGRA Simulation model remains the sole owner of the case signature,
lineage resolution, cycle metric, execution, and Evidence. A foreign owner,
workload, or runtime relation is invalid; acquisition cannot scan the
ArtifactStore, infer a similar owner, or copy a private Mapping relation.

### Objectives and Quality Gates

One central dimension type owns the fact being optimized, its direction, and
its exact normalization:

```text
ObjectiveDimension {
  source: ObjectiveScalarSourceRef
  direction: Minimize | Maximize
  normalization: ExactAffineQuantization
}
```

The source algebra is closed:

```text
ObjectiveScalarSourceRef =
    MappingViolationSource(MappingViolationDescriptorRef)
  | MappingMeasureSource(MappingMeasureDescriptorRef)
  | EvaluationMetricSource(EvidenceObligationTemplateRef,
                           MetricRequestOrdinal)
```

`EvidenceObligationTemplateRef` is the canonical local ordinal in the exact
ResolvedDseConfigView obligation-template table. It is meaningful only with
that component view and never becomes a persistent cross-config reference.
An owning component view may mechanically materialize a selected transitive
closure of DSE records and assign new references local to that exact view, as
specified for PnR by
[Search Policy And Determinism](spec-pnr.md#search-policy-and-determinism).
Such a projection copies the typed owner record and rewrites its references; it
does not carry this ordinal across views or redefine the record schema.

An objective source cannot reference an obligation tagged `HeldOut`. A held-out
obligation is reserved for a terminal `ModelRelease` gate and cannot affect
candidate generation, TopK/Pareto ranking, search energy, or an ordinary
`CandidateSelection` gate.

Mapping owns the referenced `V` and `G` descriptor semantics. The resolved DSE
configuration owns only the typed references. An Evaluation source resolves
one exact metric request in one obligation template. Its Metric descriptor
must have scalar `IntegerValue` or `DecimalValue` form, and the source accepts
only a Completed `Point` observation. A non-scalar Point, `Interval`,
`Censored`, `NotApplicable`, a missing result, or any non-Completed Evidence
outcome makes that objective source `ObjectiveUnavailable`; no midpoint,
bound, zero, infinity, NaN, or provider fallback is permitted.

The quantization record is:

```text
ExactAffineQuantization {
  origin: exact value in the source value domain
  quantum: positive exact value in the same domain and unit
  lower_index: uint64
  upper_index: uint64
}
```

`lower_index <= upper_index`. Integer and canonical `DecimalValue` inputs are
converted to exact rational arithmetic without binary floating point. For
source value `x`:

```text
index = floor((x - origin) / quantum)
require lower_index <= index <= upper_index

directed_code(Minimize) = index - lower_index
directed_code(Maximize) = upper_index - index
```

The result is a bounded `uint64`. Mapping integer sources use `origin = 0` and
`quantum = 1`; their declared bounds still remain explicit. Every conversion,
subtraction, division, product, sum, and bound calculation is checked. An
out-of-domain value or arithmetic overflow is a resolved-policy or model
contract failure, never a clamp or candidate penalty.

The resolved configuration assigns canonical local references to dimensions
and derives exactly these three consumers:

```text
ObjectiveVector = canonical sequence<(ObjectiveDimensionRef, uint64 code)>

WeightedLevel {
  canonical non-empty sequence<{
    dimension: ObjectiveDimensionRef
    weight: positive uint64
  }>
}

TotalOrdering = ordered non-empty sequence<WeightedLevelRef>

SearchEnergyRef = WeightedLevelRef
```

Dimensions are unique and sorted by complete source key, direction, origin,
quantum, and bounds before `ObjectiveDimensionRef` ordinals are assigned.
ObjectiveVector follows ascending dimension ordinal. Canonical integer and
DecimalValue codecs from the metric registry encode origin and quantum; there
is no text-number or binary-floating representation. WeightedLevels are
normalized before duplicate elimination and `WeightedLevelRef` assignment.
TotalOrdering level order remains semantic and is not sorted; duplicate level
references are invalid. Pareto dimension references form a sorted unique set.
`SearchEnergyRef` is a role-specific use of an existing `WeightedLevelRef`; it
does not create a second registry, ordinal, or record.

A `WeightedLevel` sorts terms by dimension reference, rejects duplicate
dimensions, and divides all weights by their greatest common divisor. Its
value is the checked `uint128` sum of `weight * directed_code`. Its signed
difference is represented as a sign plus a checked `uint128` magnitude, so the
difference of two valid level values cannot overflow a signed host integer.

`TotalOrdering` compares WeightedLevel values lexicographically, then compares
the canonical candidate semantic key. `TopK` references one TotalOrdering.
`Pareto` references a canonical non-empty dimension set and uses componentwise
comparison over the corresponding ObjectiveVector codes. A domain search
policy that needs annealing energy or reward references one SearchEnergyRef;
energy is that one WeightedLevel value and reward is its signed difference.

Objective facts and normalized dimension codes therefore have one owner, but
total rank, Pareto dominance, and local search energy remain distinct derived
projections. There is no universal mixed-radix objective code, hidden domain
score, or implicit conversion between these consumers.

A quality gate is finite canonical conjunctive normal form:

```text
QualityGatePolicy =
  canonical AND<canonical nonempty OR<QualityGateAtom>>

QualityGateAtom = MetricGate | FindingGate

MetricGate {
  metric: (EvidenceObligationTemplateRef, MetricRequestOrdinal)
  comparator: LT | LE | EQ | NE | GE | GT
  threshold: exact value in the metric's canonical value domain and unit
}

FindingGate {
  finding: (EvidenceObligationTemplateRef, FindingRequestOrdinal)
  required_state: Present | Absent
}
```

An empty policy means no quality constraint; an empty clause is invalid.
Finalization validates each request-local reference and threshold, sorts atoms
and clauses by their complete canonical keys, and removes exact duplicates. It
does not add a predicate language, callbacks, SAT representation, or Boolean-
equivalence engine. Every referenced atom creates an Evidence obligation, so
Boolean short-circuiting cannot suppress acquisition of required Evidence.

Gate comparison uses exactly three proof values:

```text
GateTruth = DefinitelyTrue | DefinitelyFalse | Indeterminate
```

The metric registry defines the set of exact values represented by a Point,
Interval, or Censored observation. A MetricGate is `DefinitelyTrue` only when
every represented value satisfies the comparator and threshold, and
`DefinitelyFalse` only when no represented value satisfies it. A straddling
set is `Indeterminate`. If the registered censored form does not establish a
closed represented set sufficient for either proof, the result is
`Indeterminate`. `NotApplicable` is also `Indeterminate`.

A FindingGate compares the completed result state directly. The requested
state is `DefinitelyTrue`, the opposite `Present` or `Absent` state is
`DefinitelyFalse`, and `NotApplicable` is `Indeterminate`. Missing,
Unsupported, ExecutionFailed, and CancelledOrTimeout results remain incomplete
Evidence obligations rather than gate truth values.

Every referenced atom is required. If any atom is `Indeterminate`, promotion
returns `Incomplete` before Boolean CNF selection even when another atom in
the same disjunction is definitely true. Once every atom is determinate, OR
and AND use their ordinary Boolean definitions. Incomplete Evidence has the
same promotion outcome. Neither case can remove or accept the candidate
through a fallback value.

Quality gates own acceptance only. There is no numeric gate-deviation source.
Search guidance must reference the underlying Metric as an explicit
ObjectiveDimension; a Finding does not acquire an implicit severity score.

### Resolved Configuration View

The fully elaborated component view is:

```text
ResolvedDseConfigView {
  model_authorizations
  evidence_obligation_templates
  objective_dimensions_with_exact_affine_quantizations
  weighted_levels
  total_orderings
  quality_gate_policies
  resolved_plan_nodes
}
```

Authoring-level allowed models resolve only to `model_authorizations`.
Authoring-level required Evidence resolves only to
`EvidenceObligationTemplate`:

```text
EvidenceObligationTemplate {
  exact model binding, case, conditions, and metric/finding requests
  candidate_subject_role: exactly-one CaseSubjectRoleRef
  input_subject_bindings:
    canonical table<CaseSubjectRoleRef, EvidenceAcquisitionInputSlotRef>
  calibration_partition_role: optional<CalibrationPartitionRole>
}
```

`EvidenceAcquisitionInputSlotRef` is a stable typed input ordinal owned by the
exact acquisition policy. A central Promote node and a PnR interaction binding
are two acquisition policies with distinct closed slot catalogs; the template
never carries an interchangeable global slot. A template may leave the
distinguished candidate role and the explicitly listed input-bound roles
unresolved; every other case role is fixed exactly in the template.
Instantiation fills the candidate role from the candidate set or ephemeral
candidate view, fills each listed role from the consuming policy's ordinary
typed input binding, then runs the ordinary Request verifier. The input slot
schema must be accepted by the role, and the selected task value must satisfy
the role cardinality. A role cannot be both fixed and input-bound, and the
candidate role cannot appear in the table. Promotion
recovers candidate association only through the distinguished role. It cannot
bind a model input, construct a partial `ResolvedModelBinding`, merge a
candidate into a collection, or persist a parallel candidate-to-Evidence map.

An acquisition input slot describes the complete invocation-level Artifact
set, while one Evaluation case may require a candidate-local subset of that
set. The acquisition provider may therefore return one optional task-local
input selection. When present, it must contain every input slot referenced by
the exact template exactly once, retain the same slot ordinals, use canonical
unique Artifact order, and select only Artifacts already present in the
corresponding bound input. Absence selects the complete bound input. The
central controller validates this subset relation before instantiation, and
the ordinary Request verifier still validates exact role cardinality, schema,
lineage, and compatibility. Plan resolution requires the input slot to
guarantee the role's nonempty minimum; it does not require the invocation set's
upper bound to equal one case role's upper bound. A provider cannot invent an
Artifact, move one between slots, change fixed workload or runtime facts, or
persist a candidate-to-input association.

`input_subject_bindings` sorts by `CaseSubjectRoleRef`, rejects duplicate roles,
and encodes each pair as two `u32be` ordinals after a
`u64be(count)`. The optional partition role uses the same optional framing and
role tag as plan slots. These fields are part of the resolved component-view
bytes; runtime completion order cannot alter them.

`calibration_partition_role` is absent outside an explicitly partitioned model
training flow. The acquisition policy's derived EvidenceSet output copies a
present template tag; one output cannot mix tagged and untagged obligations or
different partition tags. A training generator descriptor has exactly three
Evidence input slots tagged `Training`, `Validation`, and `HeldOut`; all three are
ordinary plan inputs even though only the Training slot supplies fitting
samples. A case-kind-4 calibration template is tagged `Validation` or
`HeldOut`, and its ground-truth Evidence role binds the identically tagged
Promote input. Plan resolution rejects a missing or repeated partition, a tag
mismatch, or a calibration template tagged `Training`. Gate and objective
references use template-local request ordinals rather than copying metric or
finding definitions.

The complete ResolvedConfig is policy SSOT. `ResolvedDseConfigView` is its
versioned canonical component view. Candidate sets, Evaluation DAG, stable work
ordinals, work-budget view, objective projections, cache indexes, scheduler
state, and mutable domain search state are derived or rebuildable and are not
Artifacts. Profiles and inheritance must elaborate before execution; the
controller cannot add defaults or hidden promotion while running.

A selected PnR closure is a second component projection from the same complete
ResolvedConfig, not a reference into this complete DSE view. Its projector
starts from the records selected by the Spatial or System PnR policy, computes
the full template/dimension/level/ordering dependency closure, canonicalizes
owner records by the semantic keys defined here, and rewrites all references to
that PnR view's local tables. Unselected DSE records and this view's digest are
not dependencies of the PnR view. Changing a selected owner record changes the
PnR bytes; adding, reordering, or changing only an unselected record does not.

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
- canonical Generate invocation records binding exact typed inputs, one
  resolved producer binding, exact descriptor-slot output sets, and
  single-child `MechanicalDerivation` or `CandidateDecision` lineage edges;
- selected or retained Artifact and Evidence references;
- owner-local planned/consumed work summaries;
- retained owner attempt/checkpoint references; and
- nonsemantic execution provenance.

Component digests are verification copies, not configuration owners. Work
summaries do not copy budget limits. `CompletedSelection` records selected
Artifact references and satisfied Evidence. `CompletedNoFeasibleCandidate`
records an empty selection and completed plan. `Incomplete` records unsatisfied
obligations and retained finalized material but no formal selected output.

### Operational Observations

`loom.dse.invocation_manifest 1.3` is the current compatible extension of that
persistent record family. Version 1.1 added one optional nonsemantic
`InvocationOperationalObservations` block to 1.0. Version 1.2 admits incomplete
Generate records before later executed plan nodes while preserving the same
per-record completion field and canonical encoding. Version 1.3 adds the
`CancelledOrTimeout` terminal to the promotion-acquisition incomplete-reason
domain. It is execution history, never evidence of infeasibility, and retains
stable tag 4 after `Unsupported` tag 3. Versions 1.0, 1.1, and 1.2 remain
importable.

```text
PromotionAcquisitionIncompleteReason =
    ProviderUnavailable // tag 0
  | SemanticWorkLimit   // tag 1
  | ObjectiveUnavailable // tag 2
  | Unsupported         // tag 3
  | CancelledOrTimeout  // tag 4
```

The operational block remains:

```text
InvocationOperationalObservations {
  total_active_wall_time_ns    : uint64
  total_process_cpu_time_ns    : uint64
  peak_resident_bytes          : uint64
  requested_worker_count       : positive uint64
  available_logical_cpu_count  : positive uint64
  plan_nodes                   : canonical array<PlanNodeOperationalObservation>
}

PlanNodeOperationalObservation {
  plan_node_ref                : PlanNodeRef
  active_wall_time_ns          : uint64
  process_cpu_time_ns          : uint64
}
```

The block is absent when the invocation owner did not collect all required
whole-invocation observations. Absence has no effect on the controller outcome,
Artifact selection, Evidence validity, replay, or resume. Within a present
block, `plan_nodes` contains at most one row for each admitted plan node, sorts
by canonical `PlanNodeRef`, and omits unobserved nodes rather than representing
them as zero. Every reference must resolve in the exact plan.

Active wall time uses monotonic intervals during which that invocation is live;
resume gaps and time after a stopped occurrence are excluded. The total is the
union of whole-invocation intervals, and each node value is the union of that
node's active intervals. Concurrent node values overlap and must never be
summed to reconstruct the total. CPU time is process execution charged during
the corresponding scope. `peak_resident_bytes` is one whole-invocation process
high-water observation; there is no per-node resident-memory field or inferred
allocation. Unsigned overflow is invalid rather than saturated.

The existing producer semantic/build identity, requested worker count, and
available logical CPU count provide necessary execution context but do not by
themselves prove two hosts comparable. Deterministic work summaries remain the
primary cross-machine cost measure. Wall time, CPU time, and resident memory may
be compared only when the consuming report or gate proves a compatible
execution context. These observations are neither MetricKinds nor semantic
budget owners, and they never copy planned or consumed work counts.

Caller-owned execution of an `ExternalToolInvocationBundle` is outside this
block. A site runner or scheduler may retain resource observations for one
exact prepared-manifest handle and attempt, but Loom does not monitor that
process or import those observations as Evidence. Their absence cannot
invalidate otherwise complete EDA Evidence.

Each recoverable logical unit uses a stable derived key:

```text
WorkUnitKey =
  PlanNodeRef
  + owner-local WorkUnitDescriptorRef
  + stable ordinal
```

The mutable ExecutionJournal may record starts, finalized output references,
attempt references, checkpoint references, and finalized-work recovery-record
references by WorkUnitKey. Physical journal event order has no semantic
meaning. The Journal cannot own a current-best answer, override an Artifact or
Evidence, replace the resolved plan, interpret an owner payload, or publish any
recovery record as formal selection.

Resume recomputes the run key and resolved plan, verifies closure and each
owner schema, revalidates Artifact preimages and Request/Evidence references,
and reuses only fully finalized outputs bound to the expected WorkUnitKey.
In-flight work is safely retried with its original ordinal. Resume cannot
renumber work, consume the same logical slot twice, substitute another
candidate, or complete from best-so-far state.

A terminal in-process Generate record is recovered directly from its immutable
owner finalized-work record. Recovery strict-imports that record, revalidates
its exact invocation closure and every referenced Artifact preimage, and
reconstructs the canonical provider result. It must not invoke the generator,
freeze a Mapping problem, allocate candidate or solver state, or enter search.
Plan reconstruction may use a bounded execution-session artifact-resolution
cache keyed by complete root reference and import algorithm version; the cache
is removable, preserves every integrity check, and cannot turn a path or object
address into identity. Terminal recovery reports import/cache work separately
from newly executed provider work.

For an external-tool attempt, a valid atomic completion permits import of that
exact bundle. A prepared bundle without valid completion remains incomplete;
the controller cannot infer process liveness, acquire an execution claim, or
automatically retry it. If the external execution owner explicitly authorizes
another attempt, it retains the same `WorkUnitKey` and materializes an
independent bundle. This is owner-attempt recovery, not a new semantic work item
or generic Job state machine.

Attempts and recovery records remain owner-specific. Evaluation uses its
request-local attempt record; an ExternalToolInvocationBundle retains generated
scripts, frozen local bindings, stdout, stderr, raw reports, and its atomic
completion record in attempt or scratch material; PnR, training, and other
domains define typed checkpoints only when real recovery requires them. An
in-flight checkpoint binds the exact run key, occurrence, plan node,
WorkUnitKey, owner schema, and version because its mutable state belongs to one
attempt. A finalized-work recovery record instead binds the exact run key,
plan node, WorkUnitKey, owner schema and version, exact owner invocation
closure, terminal outcome, and finalized output roots. It is immutable and may
be reused by later occurrences of the same run key solely to reconstruct the
owner report already accepted at that WorkUnitKey. Recovery strict-imports the
record through its owner codec, revalidates all referenced Artifact preimages,
and requires its terminal outcome and root set to equal the Journal record.
It cannot provide best-so-far state, authorize a new output, or act as formal
selection. There is no generic Attempt Artifact, generic recovery payload, or
all-domain checkpoint codec.

## Ground-Truth Collection Campaign

A model-data collection campaign is an ordinary finite resolved DSE plan whose
selected exact Requests are partitioned as Training, Validation, and HeldOut.
The plan and ResolvedConfig own what will run; completed
`EvaluationEvidence` roots own the usable observations; InvocationManifest and
ExecutionJournal own provenance and recovery. There is no `CampaignArtifact`,
`DatasetArtifact`, sample-row schema, mutable experiment database, or second
work scheduler semantics.

One model-data sample is usable only when one admitted ground-truth model
produces completed Evidence that satisfies its parameter contract's exact
model descriptor, target key, conditions, required Point observations, and
sample-group partition. A timeout, cancellation, unsupported target, failed
tool, invalid import, or incomplete prerequisite produces no sample and cannot
be relabeled infeasible. Final signoff runs may use other exact model
descriptors or longer execution limits, but their Evidence cannot enter a
parameter bundle whose collection contract excludes them.

The initial collection policy imposes two hard active-wall-time bounds:

- for every Evidence root that would enter model data, active elapsed wall time
  from dispatch of its earliest newly required uncached ancestor work unit to
  terminal Evidence completion is at most 600 seconds; a shared ancestor's
  interval applies to every dependent sample but is executed and charged only
  once; and
- one complete Training, Validation, and HeldOut collection invocation has a
  campaign limit of at most 23 hours of active wall time.

Shared uncached Mapping, RTL derivation, HardwareImplementation, and external
tool prerequisites are explicit work units in the same plan and are charged
once to that campaign before their outputs are reused. A precomputed Artifact
or Evidence root is free only when it is an explicit selected preexisting
input. A runner cannot hide prerequisite time in an unrecorded warmup or
separate launch and still claim the campaign bound.

Before admitting a large collection, the runner executes a deterministic
prefix of the same resolved plan as a pilot. Its finalized outputs and Evidence
remain ordinary resumable work; it is not a disposable benchmark with another
configuration. From completed and censored pilot work, the operational policy
derives conservative throughput and p90 remaining-time estimates. It refuses
new dispatch when the current resource allocation cannot finish within the
campaign limit, unless the user supplies a smaller explicit plan. The policy
does not silently reduce samples, move cases across partitions, change a
provider, or weaken a model target.

The campaign runner coordinates through a caller- or site-owned scheduler that
is resource-aware over declared CPU, memory, scratch, external-tool, and
license capacities. These are operational claims, not semantic candidate
properties, and do not give Loom ownership of an external process tree.
Independent work units may run concurrently; one exclusive resource claim
serializes only its actual users. Scheduling must preserve stable WorkUnitKeys
and deterministic semantic work accounting regardless of completion order.

While running, a removable projection reports completed, running, queued,
failed, timed-out, and unsupported counts; recent throughput; p50 and p90
per-kind durations; estimated completion time; and the currently limiting
declared resource. Every value derives from the resolved plan, Journal, and
owner attempt records. A JSONL stream or dashboard may persist this projection
for operators but is not an Artifact, Evidence, or scheduling authority.

A graceful stop admits no new work, allows explicitly selected in-flight work
to reach an atomic owner boundary, imports every valid completion, flushes the
Journal, and records the outer controller as `Incomplete`. Resume recomputes
the same run key and continues only missing WorkUnitKeys. It neither discards
the pilot nor creates new sample identities. Held-out quality gates, not a
fixed universal sample count, decide whether a parameter bundle is releasable.

## Candidate Generators

Candidate generation preserves domain semantics while using one central plan.
Generator capability is registered through a static typed descriptor rather
than a persistent Artifact:

Candidate-generator descriptor registry schema 3.0 is the exact current
registry namespace. Its descriptor reference uses the shared owner-local
registry framing with `loom.candidate_generator_descriptor`, version 3.0, and
the generator kind. Registry 3.0 admits exact occurrence-scoped
HardwareImplementation 4.0 slots and assigns kind 16 to the per-SpatialCore
portable RTL generator. Exact 1.0 and 2.0 references are unsupported and are
never reinterpreted as current descriptors.

```text
CandidateGeneratorDescriptor {
  generator_kind
  descriptor_schema_and_version
  implementation_semantic_identity
  provider_form: ProviderForm
  typed_input_slot_descriptors
  typed_output_slot_descriptors
  resolved_generator_config_view_contract
  optional owner_lineage_payload_contract
  optional owner_feedback_payload_contract
  determinism_contract
  owner_local_work_unit_descriptors
}

ResolvedCandidateGeneratorBinding {
  descriptor_ref
  resolved_generator_config: ResolvedGeneratorConfigViewWire
}

ResolvedGeneratorConfigViewWire {
  canonical_view_bytes: canonical byte string
  component_view_digest: ComponentViewDigest
}

ResolvedGeneratorConfigViewContract {
  schema_descriptor_bytes
  project(exact ResolvedConfig) -> owner-typed immutable view
  encode(owner-typed view) -> canonical_view_bytes
  adopt(canonical_view_bytes, component_view_digest)
    -> owner-typed immutable view
}

OwnerLineagePayloadContract {
  schema_descriptor_bytes
  encode(owner-typed decision) -> canonical_payload_bytes
  adopt(canonical_payload_bytes, canonical exact parent references)
    -> owner-typed decision
}

OwnerFeedbackPayloadContract {
  schema_descriptor_bytes
  encode(owner-typed invocation-local feedback) -> canonical_payload_bytes
  adopt(canonical_payload_bytes, exact typed input bindings)
    -> owner-typed invocation-local feedback
}
```

`ProviderForm` uses the shared tags above and is encoded as one `u32be` field
in both descriptor projections. The exact
descriptor ref therefore recovers the form before a provider implementation is
looked up. A callback pointer, local provider name, or runtime availability bit
does not enter descriptor identity.

Central `Generate` plan nodes connect exact static values or earlier
`PlanOutputRef` values to descriptor-owned input slots. The resolved node's
output slots are mechanically obtained from the descriptor's role, schema, and
cardinality contract. Those `typed_input_bindings` are the sole owner of the
generator's Artifact inputs; the generator binding does not copy them.

The config-view contract follows the component-view framing and validation
owned by `docs/spec-config-ssot.md`. The exact descriptor recovers the schema
descriptor. The adopter recomputes the digest, decodes and validates the
owner-typed value, re-encodes it, and requires exact byte equality. A
`ResolvedCandidateGeneratorBinding` is a versioned canonical value in the
resolved plan and invocation record, not an Artifact, candidate identity, or
independently authorable configuration.

It has no caller-authored identity field. Whenever a bundle or attempt needs a
compact binding identity, that value is mechanically derived from the exact
descriptor reference and the adopted canonical resolved-config bytes:

```text
CandidateGeneratorBindingIdentity = SHA-256(
  bytes("loom.candidate_generator_binding.v1\0")
  || u64be(length(canonical descriptor-reference bytes))
  || canonical descriptor-reference bytes
  || u64be(length(canonical resolved-config-view bytes))
  || canonical resolved-config-view bytes)
```

The registry owns the descriptor-reference codec and this framing. A display
name, provider label, local path, or arbitrary string cannot substitute for
that projection.

Central ranking and promotion consume their own typed plan inputs and policies;
they are not copied into a generic generator binding. If a generator itself
consumes an objective, Evidence, or another Artifact, that dependency is an
explicit typed input slot. Any other owner-local decision that changes its
generation behavior belongs in the descriptor-owned config view.

A generator publishes normal domain Artifacts, then deduplicates by
ArtifactIdentity. `InvocationManifest` may retain every valid derivation path
to the converged Artifact. Compiler transformations, Mapping Actions, hardware
transformations, and model training remain owned by their respective domains;
the controller does not define a universal Action or mutable candidate IR.

The generic provider boundary must not discard owner-produced lineage. The
exact candidate-generator descriptor contains the optional owner lineage
payload contract and therefore uniquely selects the codec, schema, and version
for any typed CandidateDecision payload. Descriptor registration rejects a
conflicting contract for the same descriptor reference, and provider-output
validation rejects a CandidateDecision edge when the descriptor has no such
contract. The adopter decodes, validates all owner-local references against the
edge's canonical exact parent set, re-encodes, and requires exact byte equality.
A descriptor without the contract can publish only MechanicalDerivation edges.
This is an
owner-typed byte contract, not a universal decision algebra.

Rejected owner-local attempts remain work-summary or attempt records rather
than fake lineage edges. Every completed or incomplete invocation records a
dense binding for every descriptor output slot and only fully finalized
retained outputs. A completed invocation satisfies each descriptor-owned
minimum and maximum cardinality; an incomplete invocation may remain below a
minimum but cannot exceed a maximum.

An invalid typed input, invalid owner tuple, or provider invariant failure is
not an incomplete search. It aborts the complete Generate invocation and
therefore creates no invocation record, retained output binding, or lineage
edge. A domain Artifact published before the failure remains an independently
valid immutable object in the ArtifactStore, but the failed Generate node does
not select it. Retained prefixes exist only for descriptor-defined incomplete
termination after all traversed inputs and emitted outputs were valid.

An external flow that preserves a new `HardwareImplementation` is a hardware
Candidate Generator even when the same process also emits reports. The new
implementation is finalized before an Evaluation observes it. An
`EvaluationModelDescriptor` never mutates or replaces its subject.

For each exact descriptor reference, the registered implementation must match
the descriptor's `ProviderForm` exactly:

```text
CandidateGeneratorProviderImplementation =
    InProcess {
      invoke(typed input bindings,
             ResolvedCandidateGeneratorBinding,
             ArtifactStore,
             BlobStore) -> Expected<CandidateGeneratorProviderResult>
    }
  | ExternalPrepareImport {
      prepare(typed input bindings,
              ResolvedCandidateGeneratorBinding,
              ArtifactStore,
              BlobStore,
              ExternalToolPreparationContext)
        -> Expected<PreparedExternalToolInvocation>
      import(typed input bindings,
             ResolvedCandidateGeneratorBinding,
             PreparedExternalToolInvocation,
             ArtifactStore,
             BlobStore) -> Expected<CandidateGeneratorProviderResult>
    }

CandidateGeneratorProviderResult {
  outcome:
    Completed {
      output_bindings:
        dense array<CandidateGeneratorOutputSlotRef,
                    canonical ArtifactRootReference collection>
      lineage_contributions:
        canonical array<CandidateGeneratorLineageContribution>
    }
  | Incomplete {
      reason: CandidateGeneratorIncompleteReason
      retained_output_bindings:
        dense array<CandidateGeneratorOutputSlotRef,
                    canonical ArtifactRootReference collection>
      lineage_contributions:
        canonical array<CandidateGeneratorLineageContribution>
    }
  work_summary:
    dense array<CandidateGeneratorWorkUnitRef, planned, consumed>
  owner_feedback:
    absent | descriptor-owned canonical payload bytes
}

CandidateGeneratorIncompleteReason =
    ProofNotEstablished  // tag 0
  | SemanticLimitReached // tag 1
  | ProviderUnavailable  // tag 2
  | Unsupported          // tag 3
  | ExecutionFailed      // tag 4
  | CancelledOrTimeout   // tag 5
```

`CandidateGeneratorProviderResult` is a transient report to the controller,
not a persistent invocation outcome. A completed result satisfies every
descriptor-owned minimum and maximum cardinality. An incomplete result obeys
every maximum, may remain below a minimum, and carries only fully finalized
retained outputs. The controller validates either variant, derives the one
outer manifest outcome, and records the exact nested Generate completion
boolean. Invalid typed inputs, a violated provider contract, or malformed
returned data are errors rather than another incomplete reason.

`owner_feedback` is optional, transient, and descriptor-owned. The central
controller validates its canonical bytes against the exact typed invocation
inputs but does not interpret, journal, cache, or persist it. A domain-specific
alternating planner may consume it only during the same bounded in-process
reopen. Terminal replay does not re-enter generation merely to recreate
feedback. A workflow that must resume the feedback across invocations first
defines an immutable Artifact under the Mapping or HardwareImplementation
owner; generic feedback bytes never become a hidden persistent channel.

`work_summary` is one dense row per descriptor-owned work unit, outside the
outcome variant and never duplicated inside it. The descriptor is the sole
owner of the stable work-unit ordinals and their meanings. The provider is the
sole runtime observation source of the planned and consumed counts; controller
inference from output cardinality, an accounting sink, or a side channel never
replaces them. The controller validates dense descriptor-order coverage and
`consumed <= planned`, splits the outcome data and the work summary into the
existing Generate record and work-summary owners, and `InvocationManifest`
remains the sole persistent owner. A controller-produced `ProviderUnavailable`
report carries the mechanically derived all-zero dense summary.
Each lineage contribution is the closed single-child edge shape defined by
`Candidate Lineage and Evaluation DAG`. Its target must occur in the named
output binding; the enclosing typed inputs and resolved generator binding are
not repeated. `MechanicalDerivation` has no parent or owner-payload field, and
`CandidateDecision` is legal only when the exact descriptor supplies the
corresponding owner lineage payload contract.

Both forms share the same descriptor, exact typed input bindings, exact
`ResolvedCandidateGeneratorBinding`, output-slot validation, work-unit owner,
and `InvocationManifest` lineage contract. Plan admission uniquely validates
that every typed input slot is ready and total. Artifact-family importers and
descriptor callbacks uniquely validate schema, cardinality, parameter
contracts, and provider-specific compatibility. The external layer adds only
local tool/runtime preflight. No layer restates the union of those checks as a
second "total admission" authority.

For an existing HardwareImplementation input, `prepare` must consume or reject
its exact representation root, top, constraints, external bindings, and memory
bindings before materializing a downstream bundle. A generator that creates
the first HardwareImplementation cannot require those output facts before they
exist; its `import` validates them against the descriptor's declared output
contract before returning the finalized root.

`prepare` materializes one deterministic finalized bundle. It neither executes
`run.sh` nor publishes an Artifact, output binding, lineage edge, or Evidence.
`import` accepts the same exact typed closure and bundle, requires a valid
atomic completion, finalizes complete output Artifacts, and returns output
bindings plus typed lineage contributions. The central
`InvocationManifest` validator alone records those contributions; no importer
publishes lineage. A later manifest failure may leave an unreferenced complete
Artifact but cannot publish a partial output binding or edge.

A Candidate Generator cannot publish `EvaluationEvidence`. In the baseline
contract, reports from a generation bundle remain attempt material and are not
reused by an evaluator. A subsequent exact EvaluationRequest over the finalized
output Artifact prepares and imports its own bundle. This avoids selecting a
nonsemantic generation attempt when several derivations converge to one
Artifact; a future reuse optimization requires a separately versioned typed
cross-attempt contract.

Local executable, module, container, license, queue, and resource availability
remain execution admission owned by the invocation environment. They do not
enter the semantic generator binding and cannot trigger semantic fallback to a
different provider.

Hardware DSE begins from at least one exact seed: a finalized builtin Fabric,
a user-supplied finalized Fabric, or one output of the template generator. It
never begins from an empty mutable graph. Candidate-generator registry 3.0
assigns these initial hardware and parameter-training kinds without changing
kinds 0 through 11:

| Generator kind | Stable spelling | Exact semantic output |
| --- | --- | --- |
| 12 | `fabric_template` | one or more finalized `fabric.module` or `fabric.system` roots |
| 13 | `spatial_topology_rewrite` | finalized `fabric.module` children |
| 14 | `spatial_microarchitecture_rewrite` | finalized `fabric.module` children |
| 15 | `system_composition_rewrite` | finalized `fabric.system` children |
| 16 | `portable_spatial_core_rtl` | one finalized portable RTL `loom.hardware_implementation 4.1` child per AccCore occurrence in the input System, architecture-only when the optional ImplementationPlatform input is absent and exact-platform-bound when present |
| 17 | `fpa_gbdt_training` | exactly one finalized `loom.model_parameter_bundle 1.0` child for `ModelParameterContractRef("loom.fpa", 3.0, 0)` |
| 18 | `system_runtime_gbdt_training` | exactly one finalized `loom.model_parameter_bundle 1.0` child for `ModelParameterContractRef("loom.system_runtime", 1.0, 0)` |
| 19 | `joint_dataflow_frontier` | finalized Canonical Dataflow children produced for an explicit bounded Dataflow/System frontier |
| 20 | `joint_mapping_frontier` | finalized System children, TechMapping, SpatialMapping, and SystemMapping roots plus the exact successfully mapped Dataflow and System roots |

The hardware and frontier kinds use the following descriptor-owned typed
configuration roots:

```text
FabricTemplateConfig {
  template_descriptor_ref
  owner_typed_parameters including builtin mesh dimension
}

SpatialTopologyRewriteConfig {
  decision_domains:
    canonical nonempty set<SpatialTopologyDecisionDomain>
  max_children_per_parent: positive uint64
}

SpatialMicroarchitectureRewriteConfig {
  decision_domains:
    canonical nonempty set<SpatialMicroarchitectureDecisionDomain>
  max_children_per_parent: positive uint64
}

SystemCompositionRewriteConfig {
  decision_domains:
    canonical nonempty set<SystemCompositionDecisionDomain>
  max_children_per_parent: positive uint64
}

BoundedFrontierPolicy {
  maximum_pairs: positive uint64
}

JointDataflowFrontierConfig {
  frontier: BoundedFrontierPolicy
  dataflow_rewrite: exact resolved DataflowRewriteGeneratorConfigView
}

JointMappingFrontierConfig {
  composition_frontier: BoundedFrontierPolicy
  mapping_frontier: BoundedFrontierPolicy
  tech_mapping: exact resolved TechMappingConfigView
  spatial_pnr: exact resolved SpatialPnrConfigView
  system_pnr: exact resolved SystemPnrConfigView
}

```

The current spatial-microarchitecture configuration descriptor is
`loom.spatial_microarchitecture_rewrite.config.2.1`; the candidate-decision
descriptor is `loom.spatial_microarchitecture_candidate_decision.3.0`, and the
provider identity is `loom.spatial_microarchitecture_rewrite.generator.v3`.
Config 2.1 appends the two Temporal operand-buffer decisions below without
renumbering 2.0 decision tags. Decision 3.0 additionally carries the complete
finalizer-produced parent-to-child Module occurrence correspondence, so a
consumer never guesses a child occurrence from a parent dense ordinal.

Kind 16 has one empty canonical resolved-config view. Its exact descriptor
fixes the portable operation-provider catalog. It consumes exactly one
finalized `fabric.system`, exactly one finalized ConfigurationABI describing
that System, and a finite set of exact interconnect implementations, then
produces one architecture-only portable RTL HardwareImplementation when the
complete System is supported. No ABI encoding policy, provider selector, or
downstream flow decision is repeated in this view.

Kinds 17 and 18 are distinct `InProcess` descriptors because one trainer
output slot must name exactly one parameter contract. They may share the same
implementation library and descriptor-owned deterministic GBDT configuration
schema, but neither descriptor accepts the other contract or carries a
caller-authored contract selector. Each has three required Evidence input
slots for Training, Validation, and HeldOut, one optional prior-bundle slot for
its own exact contract, and one exactly-one bundle output slot. Its resolved
configuration owns the seed, positive tree count, positive maximum depth,
positive minimum Training rows per leaf, and canonical learning rate in
`(0, 1]`. The implementation semantic identity owns the exact split search,
equal-gain tie breaking, arithmetic, and multi-head fitting algorithm. Changing
those semantics requires another exact descriptor reference; central DSE does
not gain a trainer-algorithm enum.

The template descriptor registry owns each parameter schema and expansion
function. `FabricTemplateConfig` invokes that exact public ADG Builder path and
produces a Fabric Artifact. A user Fabric remains an ordinary static plan input
and does not acquire a synthetic template identity.

Each decision-domain type below is a descriptor-owned closed union parallel to
its decision union. One domain member fixes the exact target selector plus a
finite canonical set of replacement prototypes/values or an inclusive bounded
integer delta range, as applicable to that decision kind. Empty value sets,
unbounded ranges, a target selector that resolves outside the exact parent, and
two members with the same canonical domain key are invalid. There is no hidden
default neighborhood. `max_children_per_parent` truncates the descriptor's
canonical decision order before construction and is semantic work policy, not
an execution limit.

The topology generator consumes a finite canonical set of `fabric.module`
parents. Each generated decision names and changes exactly one parent. Its
closed decision union is `AddOccurrence`, `RemoveOccurrence`,
`ReplacePointConnection`, `AdjustParallelConnectionCount`, and
`ChangeBoundaryInventory`. Each decision uses an exact typed PE, switch,
memory, FIFO, or boundary prototype and exact Fabric local references; there is
no generic node record. It changes connectivity, occurrence inventory, or
module interface, but not an occurrence's internal implementation policy.

The spatial-microarchitecture generator has the same finite-parent input and
one-exact-parent decision rule. Its closed decision union is `ChangePeKind`,
`ResizeInstructionStore`, `ResizeInstructionStores`, `ChangeFuInventory`,
`ChangeFuCapability`,
`ChangeSwitchModeOrScheduleCapacity`, `ResizeMemory`,
`ChangeMemoryOperationTable`, `ResizeFifo`, and
`ChangeFifoBypassCapability`, `ChangeTemporalOperandBufferMode`, and
`ResizeTemporalOperandBuffer`. The referenced Fabric owners define every typed
parameter domain. The generator cannot create an operation capability, memory
contract, scheduling rule, or bypass meaning outside those domains.
`ChangeTemporalOperandBufferMode` selects one of the three exact
`OperandBufferMode` values on one Temporal PE. It changes allocation-unit,
capacity, service, queue, and progress projection and therefore reopens the
affected Spatial cone. `ResizeTemporalOperandBuffer` changes the positive
entries-per-allocation-unit value while retaining the selected mode; its
Mapping choices may be rebased and must be independently reverified. Neither
decision is `ResizeFifo`: an explicit `fabric.fifo` and a Temporal PE operand
pool have different owners, state, services, and invalidation roots.

The current `OperandAdmissionPolicy` is mechanically derived from mode and has
no independent Fabric field or alternative physical behavior. It is therefore
not a Hardware-DSE decision. A policy refinement remains typed Unsupported
until Fabric defines its physical state, configuration field, RTL/simulator
semantics, and canonical identity; adding an inert decision tag is forbidden.
`ResizeInstructionStores` is the only multi-owner member. It carries a
nonempty canonical set of unique Temporal PE targets and positive capacities
from one exact parent Module, applies the complete set to one fresh Builder
draft, and finalizes one child. This atomicity is required because Fabric
occurrence references are scoped to the exact parent artifact and canonical
labeling may rename otherwise unchanged occurrences after the first resize.
It cannot contain any other microarchitecture decision or be decomposed by
rebinding stale parent references against an intermediate child.
`ChangeFuInventory` may select FU prototypes from any PE in the exact parent
Module; the ADG Builder admits only prototypes whose parent-PE input inventory
can be mapped to the target PE exactly by ordinal and type. The decision thus
redistributes existing typed capabilities without a private capability codec
or an implicit builtin-template rewrite.

The system-composition generator consumes a finite canonical set of exact
`fabric.system` parents and an explicit canonical set of admissible finalized
Module candidates. Each generated decision changes exactly one System parent.
Its closed decision union is `AddAccCore`, `RemoveAccCore`,
`ReplaceSpatialAttachment`, `SelectInstructionCoreRealization`,
`ChangeTransportResource`, `ChangeTransportConnection`, and
`ChangeServiceOrMemoryAttachment`. It preserves the root-complete ISA/ABI
cohort, domain, attachment, service, and transport invariants. Several AccCore
occurrences may reference one exact Module while retaining distinct
occurrence-qualified resources and cost multiplicity.

The portable-System-RTL generator preserves Fabric semantics while producing
an immutable first HardwareImplementation. It consumes an exact finalized
ConfigurationABI for the selected System and uses the Hardware-owned portable
operation-provider catalog fixed by its descriptor. The ABI remains the sole
owner of physical encodings and inactive values. Unsupported operation or
structure coverage is a typed incomplete outcome and never selects a native
provider implicitly.

Gate-netlist, placed, routed, extracted, FPGA, and native-provider transitions
remain owned by their existing provider-specific Candidate Generator
descriptors. Their exact prior HardwareImplementation, optional
ImplementationPlatform, provider binding, recipes, and flow decisions stay in
those descriptors and resolved configuration views. A DSE plan composes these
ordinary generators through explicit use-def edges. A generic implementation-
flow wrapper would duplicate provider form, configuration, preparation,
import, and work ownership, so none exists.

Kinds 13 through 15 apply one owner-typed decision to a fresh Builder draft
derived from one exact parent. The kind-14 `ResizeInstructionStores` decision
is one atomic typed decision even though its closed payload names several PE
targets. The ordinary Builder and Fabric finalizer
must accept the complete child before it is published or returned. A rejected
draft produces no child and no lineage edge; it cannot partially mutate the
parent or leave a DSE-only Fabric form. A completed child carries one
`CandidateDecision` lineage contribution whose payload is owned by that exact
generator descriptor. Identity deduplication occurs only after finalization.

The kind-15 System-composition lineage payload
`loom.system_composition_candidate_decision.2.0` also carries the exact
parent-to-child `AccCoreOccurrenceRef` correspondence for every preserved
parent AccCore. The System finalizer derives this one-to-one relation while it
canonically relabels the child; the controller may not reconstruct it from
Module equality or occurrence ordinal. `AddAccCore` and `RemoveAccCore`
validate the expected cardinality delta, while every other System rewrite
preserves the cardinality. Every unchanged descendant retains its exact Module
target. The sole exception is the target of a typed
`ReplaceSpatialAttachment`, which must select that decision's exact Module.
This lineage records structural descent only and does not claim that parent and
child resources are interchangeable.

The TechMapping providers may return
`loom.mapping.tech_compute_context_hall_feedback.1.0`. It represents one exact
Hall-deficient compute-cover relation observed by Mapping, not a proof that the
entire bounded TechMapping domain is infeasible. Its canonical payload carries
the cover demand and matching counts plus a canonical capability-to-demand
multiplicity set. Compatible resident contexts and the resulting Hall gap are
rebuilt from the exact Fabric through the same physical-demand projection used
by Tech cover search. A hardware reopen may respond with existing typed FU or
instruction-store decisions; the feedback neither mutates Fabric nor changes
an overall `ProofNotEstablished` outcome into `ProvenInfeasible`.
The root-complete Spatial provider may return
`loom.mapping.spatial_graph_boundary_endpoint_hall_feedback.1.0`. It names the
exact Module and TechMapping whose graph-boundary attachment relation is Hall
deficient. Its canonical payload retains the independent input/output demand
and Hall-neighbor endpoint counts; aggregate cardinalities are derived rather
than serialized again. The directional split is necessary because one
additional builtin gateway contributes one input and one output endpoint; the
required gateway increment is the larger directional deficit. This feedback is
not a routing failure, does not imply that another TechMapping has the same
boundary demand, and cannot weaken endpoint exclusivity.

The root-complete System provider may return
`loom.mapping.system_acc_core_capacity_pressure.3.0` only after its exact
thread/graph relation is completely exhausted under the configured assignment
bound and every assignment retains imported Spatial capacity pressure. The
payload names the exact System, complete SpatialMapping input frontier, target
Module, compatible AccCore count, assignment attempts, the exact witness
AccCore occurrence, and its usage/capacity witness. It also names one exact
`loom.mapping.system_execution_binding_checkpoint` 2.0 that retains the
thread and graph choices of that exhausted assignment. The checkpoint also
binds the exact constraint, resolved PnR configuration, search-domain digest,
and a typed imported-capacity witness with its dependency root threads. It
requests one additional compatible AccCore candidate. It neither claims that
the child is sufficient nor includes System service routing in its proof. A
work-limit outcome has no such payload or checkpoint.

The central joint DSE controller consumes all three feedback forms through the
resolved Fabric-template recipe that produced the current System. Compute
feedback first projects an exact kind-14 atomic `ResizeInstructionStores`
decision on the parent Module. Its child Module is then attached through
typed kind-15 `ReplaceSpatialAttachment` decisions for every AccCore that
actually uses that Module; the per-decision System correspondences are
composed into one parent-to-child lineage. Unchanged target Modules receive
identity correspondence entries. This is the schedule-preserving path: the
downstream rebase may use only those exact Module identities and re-verifies
the child legality. These per-PE capacities are not converted into the builtin
template's uniform `temporal_resident_contexts` parameter. That parameter may
describe the recipe that originally built the System, but it is neither the
owner nor an upper bound for a later typed subset resize.

Spatial boundary feedback currently has no typed Module-growth decision because
the Fabric boundary owner intentionally rejects invented domain rows. It uses
the ordinary template provider and therefore publishes a typed cold-fallback
reason for Mapping rather than pretending that selections survived. Mixed
Tech/Spatial/System mutations likewise take the cold path until each mutation
has an exact owner-level transformation. System capacity feedback raises the
AccCore count by exactly one only when every current occurrence targets the
named Module; a heterogeneous recipe that cannot express the requested
compatible occurrence is rejected rather than silently homogenized. There is no
mapper-side Fabric mutation, local occurrence exception, or ordinal-based
attachment rewrite.

When a child changes only the AccCore count, the target Module roots are
identity-equal to the parent targets. The controller binds the already
verified immutable SpatialMapping frontier directly to the new System
provider, after rechecking each Mapping's exact Dataflow and Module owners.
For `AddAccCore`, it combines the capacity witness's checkpoint with the exact
kind-15 AccCore correspondence and publishes one
`loom.pnr.system_mapping_checkpoint_migration_seed` 5.0. The seed identifies
the parent witness AccCore as its unique execution-binding reopen root. System
PnR releases exactly the checkpoint thread cells selected on that parent,
preserves all other admissible thread cells and every admissible graph cell,
and solves the released relation before its unchanged fresh seed family. All
service targets, service routes, ResourceUses, progress closure, and final
legality are rebuilt. A typed migration fallback records why projection could
not be used and does not suppress fresh search. TechMapping and Spatial PnR
are not repeated for a System-only composition change. A resident-context
change follows the typed kind-14/kind-15 Module lineage above, rebasing Tech
and Spatial selections against the child and reopening only after the child
owner rejects an unchanged selection. A gateway change creates a different
Module without such lineage and is explicitly cold. Apart from the exact FIFO
feedback path below, no equivalent preserve claim exists for arbitrary
topology, memory, switch, or transport changes; those paths must publish their
typed fallback reason until a corresponding hardware decision owner is added.

### Runtime-Derived Spatial FIFO Feedback

CGRA runtime may request a bounded SpatialCore FIFO hardware child only from
an exact quiescent wait witness. The witness must join the parent SystemMapping
and SpatialMapping to one canonical `FabricFifoOccurrenceRef`, full occupancy
and capacity, a closed transfer- or actor-level wait cycle containing that
FIFO, and one granted physical action held past intrinsic release solely by the
same actor's unsatisfied causal release. A dense storage ordinal, an isolated
full queue, an ambiguous FIFO set, or a provider-incomplete wait graph is not a
hardware request. It remains `ProofNotEstablished` or `Unsupported` and
publishes no child.

The candidate set is finite and minimal. When a full depth-one FIFO is the
exact liveness witness, `ResizeFifo` first proposes depth two. For a deeper
full FIFO, the first proposal is one additional slot; this is a repair probe,
not proof that one slot is sufficient. `ChangeFifoBypassCapability` may add one
alternative only when the existing Fabric admits the bypass traversal. The
ordinary `SpatialMicroarchitectureCandidateGenerator` and
`HardwareImpactProjection` own these changes; runtime owns neither a second
Fabric writer nor a Mapping exception.

FIFO semantics remain owner-defined. A depth-one registered FIFO may enqueue
and dequeue in one cycle only when it was non-full at cycle start. A full
feedback loop cannot borrow the capacity released later in that same cycle.
Bypass removes registered latency but adds combinational timing and Fmax
pressure and can expose a selected handshake cycle. No universal fifty-percent
throughput rule is inferred from depth one. The central objective orders
liveness before II and throughput, latency, timing and Fmax, area, power and
energy, reconfiguration cost, and repair work. A missing metric is typed
`Unsupported`, not zero.

Every admitted child follows the existing module-to-System replacement,
typed impact, Mapping invalidation-cone, migration-seed, cold-fallback, and
independent Fabric/Tech/Spatial/System/CGRA verification path. Parent and child
identities, preserved/reopened/repaired work, fallback reason, and the final
runtime disposition are evidence. A child that merely maps, or a resize that
removes the observed queue without reproducing application execution, is not a
successful feedback result.

When the parent has a finalized SystemMapping, the controller may instead
publish `loom.pnr.system_mapping_finalized_migration_seed` 5.0. It binds that
Mapping, the exact child constraints and Spatial frontier, the resolved PnR
configuration, and the complete typed entity, transfer, Module, and AccCore
lineage. The child rebase preserves only references proved by that lineage,
including service targets and routes when their entire hardware closure is
unchanged; every accepted result still passes child finalization and the
independent verifier. Rebase failure produces a typed cold fallback and leaves
the canonical fresh seed available.

For adjacent resource-time states on the same immutable System, the same seed
uses a mechanically derived identity correspondence plus canonical Dataflow
root invalidation roots. It preserves cone-external execution choices and
releases the changed roots into the ordinary System initializer. The first
implementation conservatively reopens the System service/route cone; a smaller
cone requires a typed service-dependency proof. This seed is only a Mapping
preference. A resource-time transition remains `Unsupported` or
`ProofNotEstablished` until the separate safe-point, Deployment configuration
and route deltas, live-state correspondence, and migration-time contract are
all derived by their existing owners.

The rebase ledger must account for the System layer separately from lower
Mapping layers. For every parent SystemMapping it records parent, preserved,
and reopened thread bindings, graph bindings, resource uses, service
realizations, and service legs. An execution-root impact reopens the binding
cone; a transport, route, service, or memory impact reopens the corresponding
resource/service cone. Unaffected System relations remain preserved. Artifact
counts alone are insufficient evidence for any of these classifications.

Hardware reopen is a bounded feedback chain rather than a Cartesian hardware
search. One failed execution produces at most one monotonically grown recipe
child; a failed child may produce the next typed witness. Hardware-parent
promotion remains bounded by the System frontier. Within one promoted parent,
deterministic local repair has a separate positive
`maximumHardwareRepairProbes` bound and an additive
planned/reserved/consumed/rejected/cancelled ledger. A Tech-context probe runs
only the ordinary Tech generator; Spatial and System PnR remain undispatched
until the Tech gate covers every required graph. Intermediate repair children
are not additional QoR alternatives. The chain stops immediately when its
probe bound is exhausted, no supported typed feedback remains, execution is
cancelled, or a verified SystemMapping is published. Exhaustion is incomplete,
not infeasible. Thus a second hardware dimension becomes eligible only after
ordinary Mapping proves it necessary.
The owner also stops a non-progressing typed feedback sequence. If successive
context-growth probes increase the observed Hall demand and context-value set
by the same amount while preserving the deficit, the aggregate feedback is not
a valid continuation proof for another child. The owner emits
`ProofNotEstablished(hall_repair_stagnation)` and records the untried
alternative repair domain; it does not spend the remaining probe budget or
call Spatial/System PnR again. This is a search-boundary diagnosis, not a
claim that the application or Fabric is infeasible.
The controller has no deficit-size threshold or Application-specific reopen
heuristic. Switch, memory, FIFO, topology, and other System resources remain
unchanged unless a later typed Mapping witness and its hardware owner define a
corresponding recipe change.

Hardware reopen is ordered after the bounded current-hardware software/System
frontier. The controller first visits those exact pairs in their declared
preference order. `FirstVerified` returns the first verified Mapping. Under
`BoundedQuality`, the complete bounded parent frontier is visited first; exact
failed-candidate feedback and already verified parents then share one explicit
hardware-parent budget. Actionable failed candidates are ordered by exact
accelerated root, graph, and actor coverage, rather than simulation-workload
row count or log order. Only the declared prefix is promoted. Any remaining
budget may expand verified parents in analytic order. A repairable failure of
an earlier software candidate cannot therefore grow hardware ahead of a later
candidate that is already legal on the parent System, while a high-coverage
exact failure is not hidden by an unrelated small verified candidate.
Cancellation or timeout still stops execution and is never converted into
permission to reopen.

Kinds 19 and 20 are the only built-in cross-frontier adapters. A two-frontier
join indexes both canonical input sets and visits pairs by increasing
`left_ordinal + right_ordinal`, then by increasing left ordinal. The visited
domain is the prefix of that order with at most `maximum_pairs` members. The
adapter computes each next pair directly and never materializes either the
complete Cartesian product or a persistent pair object. An empty required
input set therefore completes with empty outputs. Reaching `maximum_pairs`
completes the declared finite domain; it is not an incomplete search or a
claim of global optimality.

Kind 19 invokes the exact registered Dataflow rewrite generator separately for
each visited pair and returns its ordinary Canonical Dataflow children. The
underlying generator remains the sole owner of rewrite admission, decisions,
lineage payloads, and local work. Convergent children deduplicate by normal
Dataflow identity. The adapter adds only explicit pair-attempt accounting and
cannot define another rewrite rule or Fabric-capability predicate.

Kind 20 first forms a bounded System/Module composition frontier. For each
visited pair it visits the System's canonical AccCore occurrences and, when an
occurrence has a SpatialCore attachment different from the candidate Module,
constructs the exact kind-15 `ReplaceSpatialAttachment` decision. The ordinary
System-composition generator alone validates and materializes that child. The
mapping frontier then contains the input Systems and every completed child.
The adapter invokes the exact root-complete TechMapping, Spatial PnR, and
System PnR generators in that order for each visited Dataflow/System mapping
pair. It returns every finalized stage Artifact, and returns a Dataflow or
System root in its pass-through output exactly when at least one complete
SystemMapping for that pair was produced. A proven-infeasible stage contributes
no SystemMapping for that pair and does not invalidate other pairs. A typed
incomplete stage stops at that pair and retains only complete stage Artifacts
already returned by the nested owners. Invalid or internal owner failure
aborts the Generate invocation. Nested work catalogs are projected with
stage-qualified names; their meanings and counts remain derived from the
registered owner descriptors.

Neither adapter creates a `JointCandidate`, changes Mapping legality, ranks a
candidate, acquires Evidence, or owns a mutable frontier. Their input bindings,
resolved pair bound, nested exact config views, output slots, and work summaries
make the complete bounded join visible in the ordinary Generate record.

Module candidates are intermediate hardware design inputs. A joint
software/hardware search promotes a complete `fabric.system` before system
Mapping or release; a collection of unrelated Modules is not a System
candidate. Software generators still own thread, graph, channel, memory, and
compilation decisions, while SystemMapping alone owns physical AccCore targets,
SpatialMapping imports, routes, services, multicast, and ResourceUse.

The initial joint search is an explicit finite sequence of alternating
`Generate` and `Promote` nodes. A hardware batch evaluates parent-local
children against the selected software set; a software batch evaluates its
children against a bounded selected System frontier. Only an explicit bounded
frontier join may request cross-pair reevaluation. The plan never implicitly
flattens two candidate sets into a Cartesian product, and there is no
`JointCandidate`, mutable current design, or runtime loop. Exact workload
Artifact sets are ordinary plan inputs: one application, a domain subset, and
a cross-domain set naturally yield application-specific, domain-specific, and
general designs without a scope-mode enum.

Hardware generator and evaluator descriptors introduced by this contract
accept or produce exact `loom.hardware_implementation 4.1` slots. A registry
must allocate a new descriptor version or exact reference when changing an
existing slot from an earlier root shape to 4.0 or when changing an existing
provider from `InProcess` to `ExternalPrepareImport`; it cannot reinterpret a
published descriptor reference. EvaluationRequest and EvaluationEvidence root
shapes do not change merely because their exact case signature admits the new
HardwareImplementation schema.

The central plan may compose and rank these generators, but it does not copy
their semantics. Builtin search ranges and heuristics are resolved generator
configuration, not new persistent schema families. Candidate outputs are
deduplicated by their normal Fabric or HardwareImplementation identity.

## Integration Boundaries

### Mapping

Mapping and Evaluation meet through `CostVector = (V, G, Q)`:

- Mapping owns `V`, the closed typed set of temporary closure violations
  recomputed from Fabric contracts and Mapping selections.
- Mapping owns `G`, the closed domain-independent PnR measure catalog. Its
  initial member is the normalized total selected traversal claim defined by
  the PnR owner; dynamic congestion prices and search state are excluded.
- Evaluation owns `Q`, registered accelerator-aware metrics and findings such
  as runtime, cycle count, timing, area, power, and functional mismatch.

Structural invalidity is rejected directly. Mapping does not copy `Q`, and
Evaluation does not copy Mapping legality. Central resolved policy projects
`V`, `G`, and Point-valued `Q` into shared objective dimensions and derives
ranking, Pareto dominance, search energy, and reward. Quality gates consume
their exact Evaluation metric or finding requests directly. A finalizable
Mapping has no remaining `V`; failure of a quality gate over `Q` does not
become Mapping illegality.

PnR may use an ephemeral domain-specific incremental adapter for hot probes.
Its full model remains the oracle, its cache is removable, and probes create no
Request or Evidence. Any finalized candidate that starts an authorized external
evaluation uses the ordinary Request/Evidence boundary and retains raw material
only in owner-attempt or scratch state until its exact Artifact owner exists.
Evaluation-derived route guidance may order proposals, but cannot
change legal arcs, prove legality, replace complete `Q`, or enter a Mapping
Artifact.

### Resource-Time Scheduling

`MaxSpatial`, `MaxTemporal`, and `intermediate` are schedule classifications,
not ownership seeds. A classification requires one verified SystemMapping and
an explicit event-relative schedule containing, for every relevant point,
the active region set and each region's resource allocation. The accelerated
root coverage, the active set at an event, and the per-region allocation are
independent facts; none can be reconstructed from root count, logical-point
count, or available AccCore count alone.

The immutable SystemMapping already admits causal release and resource
capacity. It must not be confused with runtime reconfiguration. A runtime
resize, remap, or reprogramming transition is legal only at a compiler-known
safe point (or a completion event) and switches to another pre-verified
SystemMapping. Loom does not assume arbitrary mid-kernel preemption.

The single resource-time transition owner is the following finite contract;
it references existing SystemMapping and Deployment artifacts rather than
defining a second legality model:

```text
ResourceTimeTransition {
  trigger_event: canonical completion or safe-point event
  safe_point: compiler-owned safe-point artifact
  before_mapping: verified SystemMapping reference
  after_mapping: verified SystemMapping reference
  before_active_regions: Canonical Dataflow RootThreadLaunchRef values
  after_active_regions: Canonical Dataflow RootThreadLaunchRef values
  before_resource_allocation: canonical per-region allocation
  after_resource_allocation: canonical per-region allocation
  before_live_work: canonical live-work identities
  after_live_work: canonical live-work identities
  token_live_state_correspondence: typed correspondence or empty
  resource_delta: typed Fabric occurrence/capacity delta
  configuration_delta: Deployment configuration delta
  route_delta: Deployment route delta
  migration_time: exact or typed unsupported
  status: Verified | Unsupported | ProofNotEstablished | CancelledOrTimeout
}
```

The initial implementation is a compiler-precomputed finite transition
sequence. It does not run online DSE or PnR and does not add a second
Mapping-legality owner. Schedule-preserve uses the same contract for a
hardware parent-to-child change and for adjacent resource-time states: the
previous Mapping is the preference, only the typed transitive invalidation
cone is reopened, and the complete child closure plus independent verifier
remain mandatory. Unaffected Tech/Spatial/System bindings, routes, services,
and configurations are preserved by correspondence. A Dataflow-changing
logical-domain transformation requires an explicit typed state correspondence;
it cannot bypass the existing Dataflow identity check.

Joint schedule exploration is event-driven. Completion, readiness, and token
events extend a bounded beam/Pareto state. Candidate-specific lower bounds for
resource speedup, FIFO rate/II/depth, host transfer, occupancy, and
configuration/live-state migration cost screen proposals before incremental
PnR. Every expensive operation reserves work first, and a timeout publishes a
typed incomplete checkpoint rather than infeasibility.

Performance is part of the resource-time exploration contract. The owner uses
one cheap-to-expensive funnel rather than materializing the Cartesian product
of active sets, allocations, transformations, and transitions. One immutable
source/dependency/event/token projection is shared by the invocation. Frozen
per-region transformation features, resource-speedup curves, cut/rate/FIFO and
occupancy facts, and configuration/live-state costs feed incremental typed
state/action deltas. Sound necessary conditions may reject a state. Analytic or
calibrated estimates may only rank a bounded beam/Pareto frontier, apply
dominance and successive halving, preserve diversity, and select promotion
candidates. Functional replay, Dataflow materialization, and evidence
promotion are single-flight operations over the smaller survivor set. Only a
separately bounded finalist set may invoke real Tech/Spatial/System PnR. A
schedule hint therefore never supplies Mapping legality or an endpoint class;
every published endpoint and representative intermediate point still passes
the ordinary complete SystemMapping closure and independent verifier.

The current owner has one sound pre-PnR consequence of this rule: a focused
MaxTemporal candidate is retained for the ordinary bounded funnel even when a
region has more than one logical epoch. The epoch count is carried as
correspondence evidence, but it is not an endpoint proof and it is not an
Unsupported gate. The independent verifier requires multiple covered regions,
an explicit event-relative active-set change, exact concurrency bounds, and
maximum useful allocation before it can publish `MaxTemporal`. A singleton or
partition count without that schedule remains an intermediate point.

The event frontier computes its full graph lower bound once for the initial
state. Each admitted admission/event child then updates the bound from frozen
successor tails, active completion times, newly-ready regions, and remaining
minimum resource work; it must not rescan the complete dependency graph. The
frontier ledger reports full lower-bound evaluations separately from
incremental updates, and both are part of the deterministic state-work
reconciliation.

Future-state memoization stores the semantic ready/active/live state once and
retains a real Pareto set of path observations for that future. Two path
observations with different peak concurrency are incomparable because they may
support opposite spectrum endpoints. At equal peak concurrency, one path may
discard another only when its lower bound, allocated resource-time, and model
support are all no worse. Independent extrema from different paths must not be
combined into a synthetic envelope and used as a dominance proof.

The retained-byte limit applies to the state memo, the current and next beam,
and the bounded terminal-hint inventory that are live at the same time. It is
not cumulative generated-state work. Terminal hints are retained online as
the objective-best inventory plus the temporal and spatial extrema needed by
the final selector; the complete terminal set is never materialized merely to
truncate it afterward. Generated, retained, and pruned terminal-hint counts
are part of the frontier ledger.

The application integration enforces this ordering as two distinct passes. It
first consumes the already available Canonical Dataflow views only to build the
resource-time invocation projection and analytic frontier. It publishes
workloads and constructs joint Mapping plans only for retained finalist
identities. Invocation evidence reports generated, estimated, replayed,
materialized, and plan candidates separately, so a larger pre-Mapping software
inventory cannot be mistaken for expensive Mapping work.

The real Mapping-finalist unit is one `(software candidate, schedule hint)`
pair. A stable digest of the hint's actions, event-relative states,
allocations, and estimated timing is evaluation provenance; it does not enter
the software candidate identity and does not prove legality. The bounded
promotion inventory may therefore retain temporal, spatial, and objective-best
schedules for one software candidate without re-materializing its Structured
Program, Canonical Dataflow, workload, or functional replay. Each retained
schedule derives its own exact System partition intent and at most one joint
Tech/Spatial/System plan. Spectrum verification consumes only the schedule
that admitted that plan plus the full frontier's exact concurrency bounds; it
must not test every retained hint against a Mapping built for only the
objective-best hint.

An endpoint-focused invocation (`MaxTemporal`, `MaxSpatial`, or
`intermediate`) changes bounded schedule-hint ordering and may prioritize an
exact feedback parent in the bounded hardware-repair queue. The ordinary
pre-Mapping software frontier remains `Automatic` because it cannot consume a
future SystemMapping proof. The product flow accepts the focused invocation
only after the selected application Mapping outcome contains the requested
class from the independent Spectrum verifier; otherwise it returns typed
unsupported and publishes no Deployment. Neither hint order nor feedback
priority changes legality or supplies an endpoint label. This prevents both
pre-Mapping heuristics and a CLI flag from becoming endpoint-label authorities.

One static schedule finalist may reuse a verified SystemMapping across changing
active sets, but each region's allocation must remain constant while that
Mapping is in force. If a hint changes one region's allocation, Application
preflight marks that schedule `Unsupported` before workload materialization or
PnR unless the hint carries the complete verified safe-point Mapping transition
contract. Taking the maximum allocation and silently treating it as one static
partition intent is forbidden.

The resource-time funnel also publishes one additive frontier ledger for the
candidate frontiers that were actually evaluated. It contains planned,
reserved, consumed, rejected, cancelled, and elapsed work for source
projection, event actions, states, analytic estimates, and retained finalists,
plus state-memo hits/misses, beam pruning, and peak retained bytes. A candidate
that was stopped before frontier evaluation contributes no synthetic reserved
work. Application-side promotion counters remain separate because workload
publication, functional replay, plan construction, and provider dispatch have
different semantic owners; each owner reports its own reserve-before-dispatch
ledger.

Within one enclosing resource-time funnel invocation, the owner may retain a
bounded exact memo keyed by the complete invocation, feature, and policy
inputs. A memo hit reuses only the completed analytic frontier or a sound
fixed-input rejection; it contributes no second reserved work entry and never
skips finalist materialization, SystemMapping verification, or independent
Spectrum verification. Incomplete and cancelled outcomes are not memoized.
The memo has deterministic entry and byte limits, and its
hit/miss/single-flight-wait/capacity-bypass/retained-byte observations are
reported separately from the event-state memo. Concurrent equal misses share
one active computation; an incomplete active result may satisfy its current
waiters but is removed immediately afterward. The session is invocation-local
rather than a second global cache or a Mapping authority. Memo-capacity and
Mapping-finalist limits do not enter the analytic-result key because they do
not change the resource-time search result.

Before detailed event-frontier search, every candidate receives one cheap
lower-bound/support projection. The owner retains objective-best, coverage
extrema, resource-concentration, and canonical representatives, then advances
them in that deterministic order until the bounded Mapping-finalist inventory
has explicit schedule hints. Later candidates remain analytic deferrals, not
infeasible results. A candidate with a complete allocation domain and no
capacity-fitting point may still take the sound gate without consuming a
survivor slot. A no-fit result over an incomplete allocation domain is
`ProofNotEstablished` and cannot become a capacity proof.

Nested independent Spectrum verification reuses the enclosing immutable
SystemMapping import session when it belongs to the same ArtifactStore. The
application Mapping execution owns one such bounded session across all finalist
joins, while a standalone Spectrum call creates its own fallback session. This
removes duplicate strict reconstruction but does not skip the Spectrum
verifier or its allocation, active-set, transition, and endpoint checks. The
reported import totals are sampled after verification so cache hits, misses,
retained bytes, and deterministic work cover both materialization and the
independent verifier.

For each retained Mapping finalist, the application owner derives one
canonical `SystemBindingPartitionIntent` from the maximum allocation of each
root across that hint's event-relative states. This typed intent enters the
System PnR config digest and is consumed only by the existing Presburger
partition owner. It contains no physical AccCore identity and cannot make a
Mapping legal. Hardware reopen preserves the same intent. If the independently
verified Mapping allocation does not match the hint or exceeds a
workload-specific bound, spectrum verification returns typed
`ProofNotEstablished`; it does not reject the Mapping itself or reinterpret the
analytic estimate as legality.

A partitioned logical domain is a distinct fact from temporal reuse. A
`partitionCount` greater than one creates separate Presburger cells for the
root's logical points; it does not prove that one temporal epoch is using more
resources. Spectrum verification therefore carries the logical-epoch count as
correspondence evidence and refuses a `MaxTemporal` label unless the attached
SystemMapping schedule itself shows a changing event-relative active set with
the exact minimum concurrency and maximum useful allocation. No singleton,
full-root set, or counter-derived epoch value can supply that proof.

Mapping-spectrum verification and application execution are separate joins.
Before it classifies any schedule point, the Spectrum verifier consumes the
Mapping-owned progress basis retained by the independently imported
SystemMapping. An acyclic basis is qualified by the already verified
route/resource closure. An initialized-feedback basis remains
`ProofNotEstablished(finite_buffer_recurrence_not_established)` until one owner
proves its initial-token, finite occupancy, rate, and dequeue progress; durable
storage alone is not that proof. This does not reject the ordinary Mapping, but
it publishes no `MaxTemporal`, `MaxSpatial`, or `intermediate` spectrum label.

A spectrum-qualified Mapping may still fail its later DFG/CGRA replay. That
failure publishes no execution artifact and cannot increment the
application-level endpoint counters; the application may continue through the
remaining bounded candidates. A successful application-level endpoint requires
both the qualified SystemMapping scenario and completed functional/CGRA replay
for the same candidate identity. A finite successful replay cannot replace the
static recurrence proof.

The same owner distinguishes three kinds of reuse. Exact memoization is a
removable derived cache for a completely identical semantic input.
Schedule-preserving incremental reuse consumes a typed delta and retains only
transformation and Mapping decisions outside its transitive invalidation cone;
the repaired child still passes complete verification. Analytic or predictive
evaluation supplies a lower bound or QoR estimate with support, confidence,
and out-of-domain status; except for a separately identified sound necessary
condition, it cannot prove infeasibility. These mechanisms cannot share a
vague similarity key or become a second Mapping authority.

An exact resource-time state key binds source/Dataflow lineage, ready, active,
and live work, tokens and live state, per-region allocation, Fabric, workload,
runtime input, resolved configuration, and the frozen model snapshot. A
transition or migration key additionally binds the parent Mapping and
Deployment, safe point, typed schedule/resource/hardware delta, constraints,
algorithm identity, and child target. Any semantic input change mechanically
invalidates the derived entry. Concurrent equal misses are single-flight, and
a cache hit or incremental seed never skips the owner verifier. One invocation
uses one versioned immutable model snapshot so provider completion order cannot
change ranking.

Every frontier and cache has deterministic work, memory, and capacity limits.
The invocation ledger separately records generated, gated, estimated,
replayed, materialized, and PnR-admitted candidates; actual Mapping dispatches;
exact-cache hit, miss, and wait; incremental preserved, reopened, and repaired
work; retained bytes and peak resident memory; wall time, time to first
feasible, and time to best. Planned, reserved, consumed, rejected, cancelled,
hit, miss, and wait totals must reconcile before publication. Worker count,
execution order, and cache warmth cannot change semantic candidate identity,
formal work accounting, or the final selection for an identical evidence set.
Timeout remains typed incomplete even when no finalist reached Mapping.

For the detailed bounded sample already evaluated by one invocation, the same
owner records an analytic-shadow comparison without another frontier or PnR
call. It counts compared candidates, exact-complete feasible candidates,
screening-admissible candidates, their feasible intersection, whether the
analytic lower-bound order selected the exact-best candidate, out-of-domain
rows, and the maximum exact-makespan minus screening-lower-bound gap. A lower
bound that exceeds its exact completed makespan is a provider-contract error,
not a calibration miss. These counters are evidence about the sampled detailed
set; they do not generalize recall to deferred candidates and never establish
Mapping legality.

Application operation diagnostics may report `peak_resident_bytes` as the
whole-process RSS high-water observed at that boundary. It is not a per-stage
or per-candidate allocation attribution; such attribution is unsupported
unless a provider owns and measures it directly.

The schedule evidence used by this contract is an explicit witness, not a
counter-derived endpoint label. Each region interval records its readiness,
start, completion, and allocation; each event snapshot records the active set
and the Mapping reference in force at that event. A completion-gated consumer
may start only after each prerequisite edge marked `Completion` completes. A
prerequisite edge marked `FifoToken` may release its consumer at an explicitly
recorded token-ready event before that producer region completes. Readiness is
therefore owned per dependency edge, not once per region, and the scenario
does not duplicate it in a `has_fifo` flag. Region identity is the exact
Dataflow-owned `RootThreadLaunchRef`; an arbitrary Artifact root plus a second
correspondence table is forbidden. A witness is accepted only
when its active sets, intervals, transition chain, Mapping references,
concurrency bound, and makespan agree mechanically. The witness itself carries
no `MaxSpatial`, `MaxTemporal`, or `intermediate` class. Those classes remain
available only after the same schedule is attached to a real independently
verified SystemMapping and the endpoint comparison policy proves the class.
A complete unpruned event-frontier search derives the exact minimum and
maximum reachable peak-active-region counts while retaining both extrema under
future-state memoization. `MaxTemporal` requires the exact minimum peak plus
the maximum useful allocation for every active region. `MaxSpatial` requires
the exact maximum peak plus the minimum feasible allocation for every active
region. The maximum is not the number of covered regions: dependency and
capacity constraints may make a smaller value exact, as in a five-region
schedule whose maximum legal active set has three members. A truncated beam,
budget stop, or missing allocation bound can still produce a verified
intermediate scenario but cannot prove either endpoint.
A token-only timestamp, or a timestamp completing multiple roots, is not
encoded as one arbitrary root-completion event: until the Dataflow event owner
publishes canonical token-publication and composite-event families, such a
hint remains typed `Unsupported` and cannot enter Spectrum verification.
A one-region schedule is a verified intermediate point because the all-active
and one-active concurrency bounds coincide and cannot distinguish the two
endpoints.

### Model Parameters and Training

Training is ordinary typed candidate generation. It does not define a
`ModelTrainingRequest` Artifact:

```text
exact Training, Validation, and HeldOut Evidence partitions
+ optional prior parameter bundles
+ exact trainer descriptor, resolved configuration, and seed
  -> Generate
  -> CandidateSet<loom.model_parameter_bundle 1.0>
```

The generator descriptor owns the three partition-tagged Evidence input slots,
optional prior-bundle slots, trainer semantic identity, resolved configuration,
determinism, seed interpretation, and work units. Only the Training slot enters
feature fitting; Validation and HeldOut are admission inputs used only to prove
sample-group isolation before fitting starts. The resolved plan and
`DseRunKey` fix all those inputs. The
`InvocationManifest` alone records training provenance, attempts, and every
valid lineage path. None of those occurrence facts enter parameter identity.

Parameter semantics are selected through one versioned static typed registry:

```text
ModelParameterContractRef {
  owner_registry_identity: canonical nonempty ASCII
  owner_registry_version: SchemaVersion
  owner_local_contract_kind: uint32
}

ModelParameterContractDescriptor {
  reference: ModelParameterContractRef
  semantic_definition
  prediction_case_signatures:
    canonical nonempty set<EvaluationCaseSignatureRef>
  ground_truth_model_descriptors:
    canonical nonempty set<EvaluationModelDescriptorRef>
  consumed_base_condition_patterns:
    total table<EvaluationCaseSignatureRef,
                canonical set<ConditionApplicabilityPattern>>
  prediction_schema_descriptor_bytes
  prediction_decimal_finalization_contract
  adopt(canonical payload bytes) -> owner-typed immutable parameters
  encode(owner-typed immutable parameters) -> canonical payload bytes
  parameter_ground_truth_target_key(owner-typed immutable parameters)
    -> canonical nonempty byte string
  project_features(exact source case, CaseArtifactResolution,
                   ArtifactStore, BlobStore)
    -> owner-typed immutable feature view
  infer(owner-typed parameters, owner-typed feature view)
    -> ModelParameterInferenceOutcome
  ground_truth_target_key(exact ground-truth EvaluationRequest,
                          CaseArtifactResolution,
                          ArtifactStore,
                          BlobStore)
    -> canonical nonempty byte string
  calibration_sample_group_key(
      exact ground-truth EvaluationEvidence,
      its exact EvaluationRequest,
      CaseArtifactResolution,
      ArtifactStore,
      BlobStore)
    -> canonical byte string
}

ModelParameterInferenceOutcome =
    Prediction(owner-typed immutable prediction view)
  | Unsupported

ModelParameterBundle {
  parameter_contract_ref: ModelParameterContractRef
  payload_blob_digest: BlobDigest
}
```

The registry identity, version, and local kind use the same framing discipline
as every other owner-local typed registry. A registered descriptor owns the
parameter payload interpretation, accepted prediction cases, exact
ground-truth models, condition domain, owner-typed feature and prediction
schemas, feature projection, pure inference kernel, prediction finalization,
ground-truth target relation, and sample-group relation exactly once. Its
ground-truth case set is derived from the referenced model descriptors; it is
not another registered field.
Predictor models and calibration validators invoke that same descriptor;
neither calls another evaluator or copies its formulas. Owner-typed parameters,
feature views and predictions are ephemeral in-process values, not generic
persistent vectors, property bags, or raw-byte APIs.

The complete contract-reference canonical key uses the shared owner-local
registry reference framing with `owner_local_contract_kind` as its final
field. Registry admission requires a known owner and local kind, nonempty
prediction-case and ground-truth-model sets, a total condition-pattern table
over the prediction cases and the ground-truth models' derived case set,
nonempty prediction-schema descriptor bytes, and all typed operations above.
Duplicate references or an
owner whose registered descriptor changes under one exact version are
incompatible registry errors. Sample-group keys compare by the returned
contract-owned canonical bytes and are never persisted as another dataset
identity.

The owner-typed parameter payload embeds the contract-owned nonempty
ground-truth target key. `parameter_ground_truth_target_key` recovers it
without interpreting a trainer-private layout.
`ground_truth_target_key` derives the corresponding key from an exact admitted
ground-truth Request. It includes the exact model descriptor, provider
semantic/build identity, normalization contract, and fidelity that define the
observation function. It excludes subjects, implementation-flow choices,
library and platform cohorts, operating-condition values, replicate indexes,
attempts, and host execution controls; those result-affecting semantic facts
must instead be consumed as typed features where applicable. Every trainer and
calibration validator requires exact key equality for every sample. Pooling
observations from another provider or fidelity therefore requires a distinct
parameter contract whose feature view explicitly carries source-model identity;
matching metric names or case shape cannot silently merge targets.
An optional prior bundle must expose the same target key as every Training
sample before it can initialize fitting.

`infer` returns `Unsupported` when a structurally valid prediction case lies
outside the admitted training-support region encoded by the exact parameters.
The bundle stores no diagnostic string, confidence label, or mutable support
state; the contract derives support from its canonical payload and feature
view. An invalid case, malformed payload, or unavailable required owner remains
an error rather than `Unsupported`. A predictor maps this outcome to typed
`Unsupported(RuntimeCapabilityUnavailable)` Evidence and cannot extrapolate a
numeric result. An Unsupported Validation or HeldOut case cannot satisfy
promotion or release.

The one shared bundle schema is `loom.model_parameter_bundle 1.0`; Common
identity framing supplies that schema descriptor rather than copying it into
semantic bytes. The exact contract ref and payload digest are the complete
root. Canonical semantic bytes are the complete contract-reference key followed
by the fixed 32-byte `BlobDigest`. Canonical JSON uses exactly
`owner_registry_identity`, integer contract schema-major/schema-minor/local-kind
fields, and the 64-character lowercase payload digest; unknown or alternate
fields are invalid. Authoring receives trainer-produced owner-typed parameters,
encodes them through the resolved contract, adopts those canonical bytes, then
re-encodes and requires exact byte equality. Only then may it publish the
validated payload through `BlobStore` and publish the bundle root.
Import resolves the contract before reading the blob, relies on the Blob Store
read contract to rehash the logical bytes, performs the same typed adopt and
re-encode check, and rejects an unknown contract owner or kind, noncanonical
payload, missing blob, or corrupt blob without repair. A failed root
publication may leave a complete canonical unreferenced blob; known-invalid
bytes are never published, and no transaction or cleanup manifest is added.

Identical payload bytes under the same exact contract converge on one bundle
identity even when distinct training occurrences produced them. Changing the
contract or payload changes identity. One logical payload reused by different
contracts may share a `BlobDigest`, but each contract has a distinct bundle
root. A bundle contains no dataset references, trainer identity, seed, metrics,
confidence label, mutable epoch, or provenance copy.

One model-parameter trainer descriptor produces bundles for exactly one
`ModelParameterContractRef`; its bundle output slot carries that exact ref and
every output candidate must carry it. Plan resolution compares a producer
slot's contract with every consuming bundle-input slot before execution.
`RequestVerifier` alone strict-imports each bound bundle and checks its actual
contract against the consumer slot when constructing an exact Request.
`EvaluationPlanAdmission` does not repeat either check; it owns only
authorization, readiness, semantic work, and the calibration partition check
below. The generic plan does not acquire a parameter-specific type-refinement
DSL.

The initial FPA contract is
`ModelParameterContractRef("loom.fpa", 3.0, 0)`. Its prediction case set is
exactly case kinds 0, 1, and 10, and its ground-truth model set contains exactly
model kind 20. For all four derived signatures, its feature projector consumes every
result-affecting Base condition: process corner, supply voltage, temperature,
activity binding, and any present required-clock or relative-clock condition.
The architecture cases target the exact Fabric-owned domains; the physical
case targets the exact HardwareImplementation-owned domains and recovers the
same pre-attempt Fabric structure through the implementation's exact
`fabric_ref`. The
projector validates the typed target relation, includes condition payloads in
its feature view, and rejects any Base condition outside its declared table.
Its prediction-schema descriptor bytes are exactly the FPA owner's canonical
`FpaMetricPredictionView 1.0` descriptor bytes, compared byte-for-byte. Its
prediction finalization uses 18 significant decimal digits and
`RoundToNearestTiesToEven`.

Model kinds 7, 8, and 14 consume this contract in model-input slot 0 and one exact
ImplementationPlatform in model-input slot 1. The platform supplies the
TechnologyCorner owner needed by architecture-case conditions; the exact
HardwareImplementation case recovers the same platform through its exact
`implementation_platform_ref`. Changing a condition, platform, bundle, source
case, or HardwareImplementation identity changes the Request or makes
projection invalid. The contract never treats two operating
conditions as one prediction question merely because their structural subject
matches.

FPA contract major 3 adds the Fabric-only prediction case, exact
ground-truth-model target relation, support-region outcome, and target-key
payload requirement. Its prediction payload schema remains
`FpaMetricPredictionView 1.0`; unchanged output fields do not permit an older
contract ref to acquire those new admission and inference semantics.

The first kind-0 payload is a deterministic gradient-boosted decision-tree
ensemble over the contract-owned typed tabular feature view, support-region
summary, ground-truth target key, and four metric heads. Training may run in a
separate process or library, but authoring must encode the result through this
contract and every predictor invokes the registered in-process inference
kernel. A later linear, neural, or other algorithm uses another exact
`ModelParameterContractRef` with its own payload and inference owner; central
DSE does not acquire an algorithm enum or a generic tensor format.

For both initial tree contracts, the support-region summary is the exact
Training envelope: each numeric feature records the inclusive minimum and
maximum observed after canonical feature finalization, and each categorical or
presence feature records the canonical nonempty set observed in Training.
Inference is supported only when every field lies in that envelope. Validation
and HeldOut never expand it. This is a typed OOD guard, not a confidence bound
or feasibility proof; correlations inside the marginal envelope remain an
ordinary model limitation measured by calibration.

The initial System Runtime contract is
`ModelParameterContractRef("loom.system_runtime", 1.0, 0)`. Its sole prediction
case is kind 6 and its sole ground-truth model is kind 18. Its typed feature
view projects the exact Deployment, Gem5 Simulation Binding, System workload,
runtime input, mapped software partitioning, complete Fabric System,
SystemMapping, and admitted runtime conditions. Its prediction view contains
one whole-case `Runtime` point. Its target key fixes the gem5-CGRA descriptor,
gem5 provider build, Bridge ABI, timing contract, and fidelity; the modeled
platform remains a typed feature. DFG and RTL observations belong to model
kinds 17 and 19 and cannot enter this bundle. The first payload uses the same
deterministic gradient-boosted tabular family and in-process inference boundary
as the FPA contract, with a distinct owner codec, feature view, support region,
and parameter identity.

Its prediction-schema descriptor bytes are exactly
`u64be(35) || bytes("loom.system_runtime.prediction_view") || u32be(1) ||
u32be(0) || u64be(1) || u32be(MetricKind::Runtime)`. The Runtime value is a
canonical DecimalValue in seconds finalized to 18 significant digits with
`RoundToNearestTiesToEven`. The view is ephemeral and has no confidence,
condition copy, generic metric map, or independent identity.

The System Runtime sample-group key is derived from the exact source-backed
System workload and runtime input before Deployment, Fabric, Mapping, gem5
replicate, or attempt. All hardware and software candidate observations for
one source/input pair therefore remain in one partition, preventing the same
application input from appearing in Training and HeldOut under different
hardware. Target-key equality separately keeps provider and fidelity fixed.

Model kind 16 consumes the System Runtime bundle. Model kind 15 validates it
through case kind 11 and `RuntimePredictionError`. Case kind 11 uses the same
strict import, exact target-key equality, sample-group partitioning, and
Validation/HeldOut rules defined below for FPA calibration, substituting the
System Runtime contract, model kind 18, and one required completed Runtime
Point observation.

Validation and held-out evaluation use case kind 4. Its complete subject shape
is:

```text
fpa_model_parameter_calibration {
  role 0 parameter_bundle:
    ExactlyOne loom.model_parameter_bundle 1.0
    whose contract prediction schema is FpaMetricPredictionView 1.0
  role 1 ground_truth_evidence:
    OneOrMore evaluation.evidence.1.0
  workload: Forbidden
  runtime_input: Forbidden
  whole_case_cycle_basis: Absent
}
```

The case owner strict-imports the bundle, every Evidence root, and each
Evidence's exact Request through `ArtifactStore` and `BlobStore`. The supplied
`CaseArtifactResolution` must be total not only over the bundle and Evidence
roots, but over every imported source Request's subjects, workload, runtime
input, model inputs, condition dependencies, and their owner-declared
dependency closures. The same expanded resolution is passed to feature
projection and sample-group derivation.

Admission requires byte equality between the bundle contract's prediction
schema descriptor and the FPA-owned descriptor; requires every source Request
to use one of that contract's exact ground-truth model descriptors; requires
its contract-derived ground-truth target key to equal the bundle payload's
target key; and requires every source Base condition to match and be consumed
by the contract's table.
Each source Evidence must be `Completed` and contain exactly one
`WholeExactCase` `Point` observation for each of
`LimitingClockFrequency`, `TotalArea`, `DynamicPower`, and `LeakagePower`.
Additional typed results are allowed but do not enter this calibration
contract. An interval, censored or not-applicable observation, a missing or
duplicate required metric, or a non-Point form makes that Evidence inadmissible
for case kind 4 without making it invalid Evidence elsewhere. Keeping
ground-truth Evidence as one role preserves each original Request's
multi-subject pairing; the case never flattens software, Fabric, Mapping,
implementation, or condition collections into an accidental Cartesian
product.

The calibration model derives features from each imported source case, calls
the contract-owned `infer`, rejects an `Unsupported` sample for calibration,
compares each Prediction with the corresponding ground-truth observations, and
returns the registered whole-case calibration error metrics. Its descriptor
fixes 18 significant decimal digits and
`RoundToNearestTiesToEven` for the selected error result. Its Evidence therefore
describes one exact bundle over one exact Evidence collection. Ordinary
Promotion binds each candidate bundle to case role 0, binds role 1 from the
template's typed Validation or HeldOut input, and recovers association through
`Evidence -> Request -> role 0`.
The selected bundle may subsequently enter an ordinary predictor model input
slot that accepts the same contract ref. No candidate ever binds directly to a
model input during Promotion.

Training, validation, and held-out collections are pairwise disjoint by the
contract-owned `calibration_sample_group_key`, not merely by Evidence identity.
The key groups semantically shared samples such as one circuit observed under
different seeds or attempts according to the contract's declared leakage
boundary. Operating conditions remain projector-consumed features but do not
split a circuit across partitions. Because the trainer binds all three exact
partitions as ordinary typed inputs, producer-before-consumer scheduling makes
them available without an implicit wait edge. `EvaluationPlanAdmission`
strict-imports their source Evidence and Requests, derives canonical group-key
bytes through the one contract, and rejects any pairwise overlap before the
trainer callback executes. The trainer callback receives fitting access only
to the Training slot.

Validation obligations may feed ordinary candidate selection. HeldOut
obligations cannot feed an objective, ranking policy, search energy, training
features, or `CandidateSelection`; they may appear only in the terminal
`ModelRelease` gate. This is a readiness, leakage, and use-def check on the
existing plan, not a new plan node or persistent dataset Artifact.

Trainer failure, cancellation, or an exhausted Execution Limit produces no
partial bundle and leaves the controller `Incomplete`. Unsupported validation
is ordinary `Unsupported` Evidence and cannot satisfy promotion. Candidate
bundles are selected by the same central gates, objectives, Pareto policy,
budget, cache, and lineage rules as every other domain Artifact. A new online
epoch is a new bundle and binding. Updating a released baseline is a separate
explicit action, not a side effect of Evaluation or training.

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
- `ObjectiveDimension`, exact affine quantization, ObjectiveVector,
  WeightedLevel, TotalOrdering, SearchEnergy, and three-valued CNF gates own
  optimization policy;
- `PlanOutputRef` owns all typed plan use-def, while Generate and Promote own
  central candidate expansion, Evidence acquisition, and narrowing;
- the model-parameter contract registry owns payload, feature, target,
  support-region, inference, and calibration-group semantics; model slots and
  trainers reference exact contracts, while `ModelParameterBundle` owns
  immutable parameter identity;
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
- Equal canonical payloads under one exact parameter contract converge despite
  different training lineage; a different contract or payload changes bundle
  identity.
- Parameter authoring and import require contract-owned adopt/encode equality;
  an unknown or mismatched contract, noncanonical payload, missing blob, or
  corrupt blob fails closed without repair, and invalid bytes are rejected
  before Blob Store publication.
- A bundle model-input slot without an exact parameter contract and a contract
  on a non-bundle slot are registry errors; plan resolution rejects a producer
  contract mismatch, and Request verification rejects an exact bound bundle
  whose contract differs from its slot.
- Calibration Evidence scopes one bundle against one exact nonempty
  ground-truth Evidence collection, preserves each source Request's subject
  pairing, binds that collection through a typed Promote input, and remains
  associated through the bundle case role.
- FPA calibration rejects a foreign source case, an unconsumed Base condition,
  or a source Evidence lacking any of the four required completed Point
  observations; two activity or PVT payloads produce distinct feature views.
- FPA and System Runtime calibration reject an exact ground-truth model or
  target key different from the bundle, and no shared case or MetricKind can
  merge provider or fidelity targets implicitly.
- A valid feature view outside a bundle's support region produces typed
  Unsupported Evidence without a numeric prediction; invalid input remains an
  error, and Unsupported Validation or HeldOut cannot satisfy release.
- The System Runtime contract accepts gem5-CGRA ground truth only, predicts one
  whole-case Runtime, and cannot consume DFG or RTL execution as an alias.
- Any pairwise overlap among training, validation, and held-out sample-group
  keys is rejected before training; distinct Evidence identities for one
  leakage group do not evade the check, and a held-out obligation cannot feed
  an objective, ranking policy, fitting callback, or nonrelease gate.
- An EvaluationRequest constructor given an explicit EvaluationCase rejects an
  exact-signature mismatch with the resolved model descriptor before
  projecting fields; it never silently rebinds those fields.
- A finding-only Request is valid when its descriptor declares the capability,
  and Completed Evidence returns one explicit result for every finding ordinal.
- Completed Evidence is exactly total over both request sets, while
  Unsupported, ExecutionFailed, and CancelledOrTimeout carry only a typed
  OutcomeReason and no result arrays.
- `evaluation.evidence.1.0` rejects a `detailed_bundle_refs` field or any
  other generic raw-material reference; raw material remains attempt or
  scratch state.
- Multiple lineage paths to one Artifact deduplicate candidate Evaluation, and
  replay or resume with the same run closure and stable work ordinals produces
  the same formal selection as uninterrupted execution.
- Exact affine quantization covers Minimize and Maximize direction, decimal and
  integer inputs, explicit bounds, overflow rejection, and no floating-point
  or clamping path.
- TotalOrdering, Pareto, and SearchEnergy consume the same dimension codes but
  preserve lexicographic, componentwise, and local-energy semantics
  respectively; changing an unrelated dimension bound cannot rescale the
  selected SearchEnergy.
- Metric Point, interval, censored, and NotApplicable observations and Finding
  states exercise definitely true, definitely false, and indeterminate CNF
  outcomes without a numeric gate-deviation projection.
- Template, topology, microarchitecture, System-composition, and
  implementation-flow generators preserve their distinct typed owners while
  the central plan composes and deduplicates ordinary finalized Artifact
  outputs; a rejected draft publishes nothing and cannot mutate its parent.
- A joint plan alternates finite parent-local software and hardware batches,
  and only an explicit bounded frontier join may form cross-parent pairs; no
  implicit Cartesian product or Journal-owned current best appears.
- A collection pilot and resumed campaign retain identical WorkUnitKeys and
  accepted Evidence; hidden prerequisite work, a sample dependency slice above
  600 seconds of active elapsed wall time, a campaign limit above 23 hours, or
  treating timeout as a sample is rejected by the initial collection policy.
- Candidate Generator admission rejects a missing slot, wrong cardinality,
  incompatible contract, or unreadable Artifact/Blob closure before external
  `prepare` is entered.
- Equal admitted closure and local binding produce byte-identical bundles;
  `prepare`, caller execution, and `import` remain independently callable and
  expose no Job, scheduler, or process handle.
- InvocationManifest 1.0, 1.1, and 1.2 remain importable; 1.3 round-trips
  absent and present operational observations plus retained incomplete
  Generate frontiers and typed promotion timeout canonically, rejects unknown
  plan-node references, duplicate or unsorted rows, zero context counts, and
  arithmetic overflow, and never changes formal selection or Evidence.
- Candidate-generator binding identity is derived from the exact descriptor
  and canonical config view, and a caller-authored replacement is rejected.
- An external generator publishes only complete descriptor output Artifacts;
no completion, partial output, malformed representation root, or direct
Evidence output can produce a binding or lineage edge.

## Temporal Operand-Buffer Feedback

The queue/match/pairing projection is one removable derived view shared by
Dataflow screening, Fabric contract checks, TechMapping structural handoff,
SpatialMapping progress, simulator planning, and Hardware DSE. Its qualified
pairing key is `(InstructionContextRef, concrete FU occurrence, Physical Tag)`;
it is not an Artifact identity. TechMapping supplies semantic rate, fanout,
ordered-role, and internal-edge facts only. SpatialMapping owns physical
ingress/selector preference and exact queue-level blockers.

Analytic queue pressure is a ranking feature and may prune only a sound
necessary condition. An exact closed-wait witness with a complete invalidation
cone may request a finite, shared-budget set of operand-buffer mode/depth
children (the current provider admits at most the next separated mode and one
depth-growth alternative). Each child must pass ordinary Fabric, Tech,
Spatial, System, and independent CGRA verification. The alternatives are
sequentially dispatched under the invocation deadline; a deadline or provider
stop publishes an incomplete outcome rather than infeasibility. Unknown or
incomplete queue evidence is typed and cannot launch full PnR or become
infeasibility. Exact invocation-local projections use the existing
session/cache and work-ledger rules; repeated observations are deduplicated
without recursive unbounded feedback. The child provider publishes its
candidate limit and planned/reserved/consumed/rejected/cancelled counts; those
counts join the enclosing hardware-repair ledger before Application selection.

Runtime feedback carries one canonical ordered-head projection digest and the
exact SystemMapping plus Dataflow/Fabric/Tech/Spatial owner references. The
independent feedback verifier reimports those owners and reconstructs the
Mapping projection; it does not trust runtime labels as legality. Missing head
provenance, incomplete required roles, sequence mismatch, projection-digest
drift, or an unmatched wait edge remains `ProofNotEstablished` or
`Unsupported`. Only a capacity-blocked queue edge in the actual closed
transfer/actor wait cycle is `ExactClosedWait`; aggregate occupancy, a finite
halt, or analytic `LikelyRisk` cannot request a hardware child.

## Resource-Time Evaluation Funnel

Resource-time exploration is a correctness boundary as well as a performance
boundary. A policy must not form a Cartesian product of active sets,
per-region allocations, transformations, and transitions and then invoke
complete Tech/Spatial/System Mapping for every row. The sole resource-time
policy owner instead applies a cheap-to-expensive funnel:

1. Freeze the immutable source/Dataflow dependency, event, token, and per-region
   feature projection once for an invocation.
2. Apply sound necessary bounds before materialization. Cache exact per-region
   speedup, rate/FIFO, occupancy, communication, and configuration/live-state
   features under the existing invocation-local sessions.
3. Advance a bounded event frontier by typed state/action deltas. Ready sets,
   active sets, resource allocation, cuts, traffic, and lower bounds are updated
   incrementally; a successor is not allowed to rescan the complete graph.
4. Use analytic or calibrated estimates only for beam/Pareto/dominance and
   promotion ordering. They carry support, confidence, and out-of-domain
   status and cannot prove general infeasibility or Mapping legality.
5. Run functional replay, Dataflow materialization, and real
   Tech/Spatial/System Mapping only for bounded survivors. Exact identical
   inputs may share a derived result; a typed schedule delta may preserve the
   unaffected Mapping cone; neither reuse path skips the owner verifier.

These are three distinct reuse relations. Exact memoization requires byte-exact
semantic inputs. Schedule-preserving incremental reuse requires a typed delta,
safe point, parent Mapping/Deployment, and a mechanically derived transitive
invalidation cone. Analytic prediction is never a legality result. Cache keys
bind source/Dataflow lineage, event-relative ready/active/live state,
token/live-state, per-region allocation, Fabric, workload/runtime, resolved
configuration, model snapshot, parent Mapping/Deployment where applicable, safe
point, delta, constraints, and algorithm identity. Frontier policy, rank,
estimated time, witness, disposition, and cache state are provenance and do not
change candidate identity.

Every expensive owner reserves work before dispatch and closes planned,
reserved, consumed, rejected, cancelled, cache-hit, cache-miss, and wait
counts. Cache capacity, retained bytes, frontier width, worker count, and
deadline are deterministic invocation inputs. Concurrent equal misses are
single-flight; completion order, worker count, and warm versus cold state may
not change identity or the final choice over the same evidence. A timeout,
cancelled provider, or incomplete verifier is `Incomplete`, never
`ProvenInfeasible`.

The final `MaxSpatial`, `MaxTemporal`, or intermediate label is emitted only by
an independently verified SystemMapping. A schedule hint can rank or gate a
candidate but cannot supply that label. Evidence must report generated,
screened, estimated, replayed, materialized, promoted, and actually dispatched
candidates, along with exact/incremental/analytic reuse, cold-versus-incremental
work, retained bytes, time-to-first-feasible, time-to-best, and bounded
analytic-vs-exact recall/error. Missing evidence keeps the invocation typed
incomplete rather than being replaced by a larger timeout or extra search.

An incomplete or no-Mapping parent may still retain an exact owner feedback
payload. It remains `Incomplete` and cannot satisfy a sibling-selection gate,
but an explicit endpoint-focused invocation may promote that payload into one
bounded hardware-repair parent. Tech Hall pressure is ranking provenance for
that bounded promotion; it does not change legality. When the minimal
compatible-PE Hall closure is not a robust PnR seed, the hardware owner may
admit one uniform Temporal-PE capacity alternative with a deterministic
context/byte/work budget. The Module occurrence capacities and the System
recipe parameter must agree, and the child still requires complete
Fabric/Tech/Spatial/System/CGRA verification. A multi-graph Tech frontier is
coverage-capped before child planning; if the declared bound cannot retain at
least one mapping per required graph, the child is typed unsupported rather
than exceeding the bound.
