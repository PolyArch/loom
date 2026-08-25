# Frequency, Power, And Area Evaluation

Frequency, power, and area are ordinary typed Evaluation metrics. Loom does not
define an `FPAReport` artifact or a separate FPA evidence pipeline.

## Subjects

Physical metrics are evaluated against one exact `HardwareImplementation`.
Fabric alone may be the subject of an explicitly analytical architecture model,
but such a result must not be labeled as synthesis, layout, or signoff evidence.

Workload-dependent `DynamicPower` requires compatible activity. Activity is
owned by an exact `SimulationExecution`; an evaluator may mechanically
translate an actor or Fabric basis through exact Mapping and implementation
lineage, or consume the exact implementation-signal basis directly. There is
no independent `ActivityProfile` artifact. Energy is not a current registered
MetricKind.

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

Activity changes the physical question and is therefore represented by the
exact `ActivityBinding` base condition owned by
`docs/spec-evaluation-metrics.md`. Its source is either that owner's typed
execution-summary reference or its explicit clock-relative assumption; this
specification does not repeat the condition schema.

Leakage-only queries may omit activity. A workload-dependent query with no
required binding is `Unsupported`; there is no hidden default toggle rate.
The exact target, probability, density, assignment-key, and duplicate rules
remain with that owner.

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
owned by Evaluation registry 3.0. The initial FPA prediction view covers
`LimitingClockFrequency`, `TotalArea`, `DynamicPower`, and `LeakagePower`.
`MaximumVoltageDrop` is an ordinary shared physical MetricKind but is outside
that initial parameter-bundle and calibration contract. Critical-path delay,
timing slack, other physical breakdowns, and energy are unsupported until
their exact MetricKind and producing model owners are registered.

Every observation records its exact request ordinal and provenance. Point,
interval, censored, and not-applicable results retain their ordinary Evaluation
meaning. Missing activity, unsupported corners, failed tools, and timeouts do
not become numeric zero.

A current `Runtime` result must be produced by an exact registered model that
proves its timing basis and compatible implementation, workload, conditions,
and units. A user-facing FPA table is only a projection of registered
MetricKinds in Evidence; it cannot synthesize an unregistered energy or other
derived quantity.

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
Expensive raw tool products remain owner-attempt or scratch material with no
current Artifact schema and are never committed as test fixtures.

`EdaPredictionModelWeight` is the immutable in-process FPA view of one such
bundle. Import admits only the exact `loom.fpa` parameter contract and its
owner-typed `FpaGbdtParameters`; inference consumes that view without another
payload, codec, artifact identity, registry, or mutable latest-weight state.

A `HardwareImplementation` represents one exact SpatialCore occurrence closure,
not a complete System shell. Offline leaf assessment binds an exact
occurrence-local PE, FU, memory, switch, or transport boundary through that
owner. It returns `IndependentlyRoutedLeafUnavailable` because the current
implementation domain has no independently routed leaf product, or
`RoutedAsicImplementationUnavailable` when the enclosing implementation is not
a routed ASIC product. A malformed, foreign, or non-leaf local reference is an
error. These dispositions do not invalidate routed SpatialCore FPA Evidence
and are not physical Evidence or signoff claims.

An EDA-derived parameter bundle remains the same canonical
`ModelParameterBundle` in its machine-local Artifact Store;
`EdaPredictionModelWeight` is only its validated typed view and not a public
projection or second serialization. The canonical
bundle root, its payload, source Evidence, partition membership, calibration
Evidence, training invocation manifests, EDA-produced selection bindings and
provenance, and all direct EDA products remain local under the disclosure
boundary in [EDA Tooling](spec-eda-tooling.md).
Analytical parameters authored without direct EDA lineage follow their own
source and licensing disclosure rules, but the FPA schema does not grant them
repository eligibility.

The current FPA parameter contract is exactly
`ModelParameterContractRef("loom.fpa", 4.0, 0)`. Its prediction case signatures
are `structured_program_with_fabric`, `canonical_dataflow_with_fabric`, and
`fabric_hardware_analysis`; its sole ground-truth model descriptor is
`openroad_routed_static_fpa` over `hardware_implementation_physical`. It projects
typed structural and condition
features from those exact cases and returns one owner-typed
`FpaMetricPredictionView 1.0` containing point predictions for
`LimitingClockFrequency`, `TotalArea`, `DynamicPower`, and `LeakagePower`. Its
payload codec, accepted case/condition domain, feature and prediction schemas,
feature projector, inference kernel, prediction finalization, support region,
ground-truth target key, and sample-group projection are one registry entry
shared by every trainer, predictor, and
calibration validator. A model input slot references this exact contract and
never owns a copied coefficient layout or inference formula.

Contract major 3 adds the Fabric-only prediction case, exact
ground-truth-model target relation, support-region outcome, and target key.
`FpaMetricPredictionView 1.0` remains the output payload schema because its
metric tuple and codec do not change; the versioned contract ref, not the
payload shape, owns subject admission.

Contract major 4.0 caps the canonical parameter payload at decimal 10 GB.
Rejecting a larger payload is incompatible with 3.0, so no 3.x bundle is
reinterpreted under the new contract. Bundle import rejects a larger stored
object before mapping or copying it; the kind-0 GBDT payload codec and
prediction view remain unchanged.

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
Structured Program/Fabric case, model kind 8 for the Canonical
Dataflow/Fabric case, and model kind 14 for the Fabric-only case. Each consumes
exactly one matching bundle and one exact
ImplementationPlatform. The contract and platform are explicit model inputs;
ambient codebase, tool installation, or an inferred technology target cannot
select them.

The initial kind-0 payload is a deterministic gradient-boosted decision-tree
ensemble over a contract-owned typed tabular feature view, four metric heads,
a support-region summary, and one nonempty ground-truth target key. Training
may be external or in-process, but the canonical payload codec and inference
kernel are registered here and inference is always in-process. Other
algorithms register a different exact parameter contract; the shared DSE and
Evaluation APIs do not expose a model-algorithm enum, tensor bag, or caller-
defined feature vector.

The initial in-process producer is candidate-generator kind 17,
`fpa_gbdt_training`. Its output slot is fixed to this exact contract; the
trainer cannot select another parameter owner from configuration. The
corresponding System Runtime contract uses distinct kind 18 even though both
descriptors reuse the same deterministic tree implementation library.

The payload's support summary is the exact Training envelope. Every finalized
numeric feature must lie in the inclusive Training minimum/maximum, and every
categorical or presence feature must belong to the canonical set observed in
Training. Validation and HeldOut do not expand this envelope. Passing it is
not a confidence bound or feasibility proof; it only prevents explicit
marginal extrapolation, while calibration measures remaining error.

The target key distinguishes the selected physical model, provider build,
report normalization, and fidelity while excluding the individual circuit,
implementation flow, library/platform cohort, operating-condition values,
replicate, attempt, and host controls. Those excluded semantic inputs remain
typed features. Every training and calibration Evidence member must derive the
same key as the bundle. Mixing another provider or fidelity requires another
contract that explicitly models source identity; equal MetricKinds are not
enough. A valid feature view outside the payload's support region returns typed
`Unsupported(RuntimeCapabilityUnavailable)` rather than a numeric
extrapolation, and an Unsupported Validation or HeldOut case cannot satisfy a
model-release gate.

FPA calibration uses the `fpa_model_parameter_calibration` case with one
bundle and a nonempty exact collection of completed EDA Evidence. Every member
must retain an `openroad_routed_static_fpa` Request under conditions consumed by
the contract, derive the bundle's exact target key, and contain one completed
Point observation for
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

The hardware-only model kind 13 over exact Fabric returns the same four
physical metrics and no Runtime. Model kind 14 is its parameter-backed form.
Dynamic power in either form requires an exact admitted activity assumption;
the absence of software does not authorize a hidden toggle-rate default.

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

Application bounded-quality selection imports a frozen
`EdaPredictionModelWeight` and evaluates its parameter-backed descriptor
in-process. The pair decision derives its quality dimension labels, quantized
objective codes, and Evidence roots from the shared `ObjectiveProgram` and
`JointDesignExecutionSummary`; it does not reinterpret a code as a physical
unit. Pre-Mapping hardware-promotion observations retain every assessed System
and identify the bounded finalist set that entered ordinary Mapping/PnR work.
That promoted disposition is work provenance, not a feasibility or signoff
claim. Runtime-qualified tail invocations retain their own InvocationManifest
run key and local plan-ordinal base; their observations are never folded into
the final invocation's `JointDesignExecutionSummary`.

Runtime-feedback repair plans use that same bounded-quality policy before a
repair Mapping may re-enter Application validation. Each child invocation
retains its own quality and hardware-promotion observations. The shared
`ObjectiveProgram`, Pareto dimensions, and final total ordering select the one
repair Mapping eligible for the Application join. Selection reconstructs the
validated ObjectiveVector from each child invocation's recorded codes and
never repeats quality acquisition. Any typed incomplete child blocks that
selection instead of becoming an inferior numeric objective.

## Anchor Verification

Stable tests cover exact implementation coupling, separation of implementation
derivation from Evaluation, activity basis/window/coverage compatibility,
missing-is-unknown behavior, exact projection lineage, unit-safe derived
metrics, explicit missing/failed outcomes, one parameter contract shared by a
predictor and independent validator, condition-sensitive feature projection,
preservation of each ground-truth Evidence case pairing, exact target-key
separation across providers and fidelities, typed out-of-distribution
rejection, rejection of non-Point calibration samples, and rejection of cross-
partition sample-group leakage.
Tests do not pin vendor log text or a tool-by-tool report matrix.
