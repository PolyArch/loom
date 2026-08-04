# Frequency, Power, And Area Evaluation

Frequency, power, and area are ordinary typed Evaluation metrics. Loom does not
define an `FPAReport` artifact or a separate FPA evidence pipeline.

## Subjects

Physical metrics are evaluated against one exact `HardwareImplementation`.
Fabric alone may be the subject of an explicitly analytical architecture model,
but such a result must not be labeled as synthesis, layout, or signoff evidence.

Workload-dependent power or energy also requires compatible activity. Activity
is owned by an exact `SimulationExecution`; an evaluator may mechanically
translate an actor or Fabric basis through exact Mapping and implementation
lineage, or consume the exact implementation-signal basis directly. There is
no independent `ActivityProfile` artifact.

The sole pre-Mapping exception is an explicitly analytical `(S,F)` or `(D,F)`
model whose immutable descriptor semantics define a structure-derived
reference-activity rule over its exact software subject. That rule produces
only low-confidence architecture estimates and cannot be reused as activity
for a `HardwareImplementation`, RTL, synthesis, or layout case. Those cases
still require the typed activity binding below. A model cannot silently add a
default toggle rate when neither its exact subject contract nor an
`ActivityBinding` owns the activity basis.

The physical-evaluation descriptor references case kind 5,
`hardware_implementation_physical`, whose required role is:

```text
implementation: HardwareImplementation
```

Activity changes the physical question and is therefore one explicit base
condition:

```text
ActivityBinding {
  target: SubjectTargetRef
  source:
      ExecutionActivity {
        simulation_execution_ref
        activity_summary_ordinal: uint64
      }
    | ExplicitAssumption {
        clock_domain: SubjectTargetRef
        static_probability: ExactRatio
        transitions_per_clock: ExactRatio
      }
}
```

Leakage-only queries may omit activity. A workload-dependent query with no
required binding is `Unsupported`; there is no hidden default toggle rate.
The exact target, probability, density, assignment-key, and duplicate rules
are owned by `docs/spec-evaluation-metrics.md`; this document does not define a
second activity-condition schema.

An evaluator declares which `ActivitySummary` payload kinds, windows, and
coverage forms it accepts. A partial summary never makes unlisted targets
zero. A model may return an interval, a censored observation, or a typed
finding when its declared method can soundly use partial coverage; otherwise
the request is `Unsupported`. Statistical or sampled raw activity cannot be
silently promoted to an exact summary.

Direct SAIF or toggle-table projection uses an exact
`ImplementationSignals` summary. Projection from `FabricResources` is legal
only when the exact HardwareImplementation owns a complete activity-point
catalog and a mechanical Fabric-to-activity-point relation for the selected
resources. Projection from `ActorTransitions` additionally requires exact
Mapping lineage. These projections are disposable evaluator inputs and do not
become another persistent activity authority.

## Implementation Derivation Versus Evaluation

An external flow that creates new persistent hardware state performs a
`HardwareImplementation` derivation. Examples include synthesis that produces a
gate netlist and physical implementation that produces placed, routed, or
extracted design state. Invocation lineage records the input implementation,
tool/library bindings, and output implementation.

Analysis of an immutable implementation is Evaluation. Timing, area, power,
DRC findings, and similar normalized observations are `EvaluationEvidence`.
Raw reports remain owner-attempt or scratch material until their exact Artifact
owner is defined. Evaluation never mutates the implementation.

A backend invocation may do both in sequence, but the two semantic outputs stay
separate:

```text
derive HardwareImplementation -> evaluate that exact implementation
```

## Metrics

Metric kinds, dimensions, canonical units, scopes, and observation forms are
owned by the central registry. The initial shared physical metrics are
`LimitingClockFrequency`, `TotalArea`, `DynamicPower`, and `LeakagePower`.
Other representative future metrics include:

* critical-path delay and timing slack;
* cell, macro, or routing-area breakdowns; and
* energy derived from compatible power and runtime observations.

Every observation records its exact request ordinal and provenance. Point,
interval, censored, and not-applicable results retain their ordinary Evaluation
meaning. Missing activity, unsupported corners, failed tools, and timeouts do
not become numeric zero.

Runtime and energy are derived by named `DerivedMetricModel` evaluations. They
must reference exact upstream Evidence and prove compatible implementation,
workload, conditions, and units. A user-facing FPA table is only a projection of
those Evidence records.

## Model Families

Loom may provide analytical, calibrated, RTL/library-based, synthesis,
post-layout, FPGA implementation, or measured models. These are descriptor
capabilities, not rungs in a global fidelity ladder. Each model states exactly
which phenomena and metrics it supports.

Fast model parameters use exact `loom.model_parameter_bundle 1.0` references
in descriptor-owned model input slots when they are stable semantic inputs.
Training is ordinary typed DSE candidate generation; fixed validation and
held-out cases use ordinary EvaluationRequest/EvaluationEvidence promotion.
The bundle owns only its exact registry-owned parameter contract and canonical
payload digest. `InvocationManifest` owns training provenance, and
`ResolvedModelBinding` owns predictor consumption of the selected bundle.
Expensive raw tool products remain owner-attempt or scratch material until an
exact raw detailed-bundle Artifact owner exists; they are not committed as
routine test fixtures.

A released parameter bundle may be committed as that same canonical
`ModelParameterBundle`; Loom does not define a public-weight projection or a
second serialization. Repository eligibility requires the contract-owned
payload to contain only derived predictive or analytical parameters. It must
not encode training samples, Evidence or Invocation references, sample-group
keys, report excerpts, tool paths, local platform paths, proprietary payloads,
or other attempt data. The canonical bundle root and its canonical payload are
the complete publishable model-weight unit. Source Evidence, partition
membership, calibration Evidence, training invocation manifests, and all
direct EDA products remain local under the disclosure boundary in
[EDA Tooling](spec-eda-tooling.md).

The initial FPA parameter contract is exactly
`ModelParameterContractRef("loom.fpa", 1.0, 0)`. Its prediction case signatures
are `structured_program_with_fabric` and
`canonical_dataflow_with_fabric`; its ground-truth signature is
`hardware_implementation_physical`. It projects typed structural and condition
features from those exact cases and returns one owner-typed
`FpaMetricPredictionView 1.0` containing point predictions for
`LimitingClockFrequency`, `TotalArea`, `DynamicPower`, and `LeakagePower`. Its
payload codec, accepted case/condition domain, feature and prediction schemas,
feature projector, inference kernel, prediction finalization, and sample-group
projection are one registry entry shared by every trainer, predictor, and
calibration validator. A model input slot references this exact contract and
never owns a copied coefficient layout or inference formula.

For every accepted source case, the projector consumes process corner, supply
voltage, temperature, activity binding, and any present required-clock or
relative-clock conditions. It validates exact target and implementation
lineage and treats every consumed payload as a feature. Operating conditions
therefore change a prediction question even though the sample-group projection
deliberately keeps observations of one circuit in one leakage partition. The
projector rejects an unconsumed Base condition or an unavailable typed target;
it never drops a result-affecting condition to obtain a feature vector.

The prediction view is an ephemeral owner-typed value with exactly those four
`MetricKind` entries in registry order. Every value is a canonical
`DecimalValue` in the Metric registry's unit and domain. It has no generic
metric map, confidence field, condition copy, serialized wire, or independent
identity; the exact source case supplies conditions and the parameter contract
supplies its type and inference semantics.

The view's canonical descriptor bytes are
`u64be(31) || bytes("loom.fpa.metric_prediction_view") || u32be(1) ||
u32be(0) || u64be(4)`, followed by the central Metric registry's four exact
`u32be(MetricKind)` tags in the order listed above. The literal length is the
byte length of the exact ASCII owner string. Metric value type, unit, and
domain remain derived from those MetricKind descriptors rather than being
copied into this view descriptor.

The registered calibrated predictor descriptors are model kind 7 for the
Structured Program/Fabric case and model kind 8 for the Canonical
Dataflow/Fabric case. Each consumes exactly one matching bundle and one exact
ImplementationPlatform. The contract and platform are explicit model inputs;
ambient codebase, tool installation, or an inferred technology target cannot
select them.

FPA calibration uses the `fpa_model_parameter_calibration` case with one
bundle and a nonempty exact collection of completed EDA Evidence. Every member
must retain a `hardware_implementation_physical` Request under conditions
consumed by the contract and must contain one completed Point observation for
each of the four FPA metrics. Other valid Evidence forms remain usable
elsewhere but are not samples for this four-output contract. Its
sample-group key is derived from the pre-attempt architecture subject and
implementation family before tool seed, attempt, replicate, or operating
condition. Consequently, observations of one circuit at different seeds or
corners stay in one partition. Training, validation, and held-out partitions
are pairwise disjoint by this key. The trainer binds all three as typed inputs
before fitting and receives feature-fitting access only to Training. Validation
may rank bundles; held-out Evidence is excluded from objectives and candidate
ranking and appears only in the terminal model-release gate.

Calibration requests use the four prediction-error MetricKinds owned by the
central metric registry with `Quantile(1/2)` and `Quantile(9/10)`. They do not
create FPA-private error fields, dataset summaries, or confidence labels in the
bundle.

### Complete Low-Confidence Architecture Model

The initial pre-Mapping analytical models over exact `(StructuredProgram,
Fabric)` and `(CanonicalDataflowProgram, Fabric)` cases request and return this
complete shared metric set:

```text
Runtime
LimitingClockFrequency
TotalArea
DynamicPower
LeakagePower
```

Every result is a point observation with `UncertaintyKind::Unquantified`:
the model supplies a point estimate but no quantified error bound. Values use
the central registry's canonical physical units even though the model's
absolute coefficients are not calibrated. The exact model descriptor identity
owns those coefficients and the structure-derived activity rule; neither
Fabric nor DSE owns a fallback cost table.

Static frequency, area, and leakage projections consume the complete finalized
Fabric inventory, concrete operation capability families and physical widths,
resource contracts, and actual System attachment multiplicity. Runtime and
dynamic power additionally consume the exact software candidate's instruction
ownership, canonical actor demand, scheduling pressure, type widths, and
structure-derived activity. Relative ordering must remain physically sensible,
including greater cost for floating-point than integer arithmetic at equal
width and increasing cost with vector or physical payload width.

An EDA-backed model returns the same metric kinds and numeric domains with a
different descriptor, method, conditions, and uncertainty. Calibration may
therefore compare or replace estimates without changing EvaluationRequest,
EvaluationEvidence, or DSE result schemas. An analytical value is never labeled
as synthesis, layout, signoff, or measured evidence.

## Tool And Library Binding

EDA preparation emits an ExternalToolInvocationBundle from an exact model
binding containing the result-affecting tool version, technology/library data,
parser, and semantic effort. PVT, required clock, and the activity binding are
typed base conditions. Local executable paths, activation scripts, licenses,
scratch roots, and container selection are invocation bindings. Host
concurrency and resource policy remain owned by the caller or scheduler.
Optional execution invokes the generated bundle script; Loom does not supervise
the EDA process tree or own its resource environment.

An authorized expensive model is genuinely executed when the resolved DSE plan
selects it. Loom must not silently replace it with an estimate. Cancellation or
timeout maps to `CancelledOrTimeout`; tool or adapter failure maps to
`ExecutionFailed`. Resource unavailability leaves attempt and controller state
incomplete but cannot create partial Evidence or change formal candidate
selection.

## DSE Boundary

Central DSE consumes normalized Evidence and owns objectives, gates, promotion,
Pareto selection, and model-training orchestration. Mapping may query an exact
resolved model through its shared Evaluation adapter, but Mapping does not own
area, frequency, power, energy, or fallback formulas.

## Anchor Verification

Stable tests cover exact implementation coupling, separation of implementation
derivation from Evaluation, activity basis/window/coverage compatibility,
missing-is-unknown behavior, exact projection lineage, unit-safe derived
metrics, explicit missing/failed outcomes, one parameter contract shared by a
predictor and independent validator, condition-sensitive feature projection,
preservation of each ground-truth Evidence case pairing, rejection of
non-Point calibration samples, and rejection of cross-partition sample-group
leakage.
Tests do not pin vendor log text or a tool-by-tool report matrix.
