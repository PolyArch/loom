# Evaluation Metric And Finding Primitives

This specification owns the reusable metric, finding, scope, condition,
numeric-value, and observation algebra used by every Evaluation model.
Evaluation Request and Evidence roots are specified by
`spec-dse-feedback.md`; this document does not define a parallel result
artifact or report schema.

## Ownership

The Evaluation library has one typed registry for `MetricKind` and one typed
registry for `FindingKind`. These registries are the semantic source of truth
for compilation analysis, Mapping guidance, simulation, RTL and EDA models,
training, calibration, and user-facing projections.

A model descriptor declares which registered queries it supports. It does not
copy their definitions. A DSE policy selects objectives, thresholds, weights,
and acceptance rules. It does not change metric or finding meaning.

Tool-specific report fields, local-search penalties, arbitrary strings, and
temporary scores do not become registered kinds. A quantity is promoted into
the registry only when it has stable semantics that must be compared across
models, requested by users, consumed by DSE, or used by training and
calibration.

## Shared Artifact Atoms

The Common artifact contract owns `SchemaVersion`, `ArtifactIdentity`, and
typed `ArtifactReference`. Evaluation reuses those types directly. It does not
define aliases with different semantics, nullable identity sentinels, or a
second identity recipe.

Canonical text uses `X.Y` schema versions and lowercase hexadecimal artifact
identities. Individual artifact roots own their schema identity, supported
versions, and canonical semantic bytes.

## Evaluation Scope

Metrics and findings share one closed `EvaluationScope` algebra. There is no
separate `MetricScope` or finding-scope schema.

The earlier `WholeSubject | Entity | Relation` union is replaced because
`WholeSubject` was ambiguous for multi-subject cases and a generic Evaluation
entity catalog would duplicate the reference catalogs owned by each Artifact
family.

Every `MetricKind` and `FindingKind` descriptor owns one or more
query-kind-local scope-form descriptors:

```text
ScopeFormDescriptor {
  form_ref: ScopeFormRef
  semantic_definition
  roles: ordered non-repeating ScopeRoleDescriptor tuple
}

ScopeRoleDescriptor {
  role_ref: ordinal in this form
  semantic_role
  accepted artifact-root or artifact-local reference kinds
}
```

`ScopeFormRef` is a stable ordinal local to one query-kind descriptor in the
exact Evaluation schema version. The containing `MetricKind` or `FindingKind`
always resolves it, so it is not a global relation registry or a free string.
A breaking change to an existing form requires a new form or an incompatible
schema version.

The persistent value is:

```text
EvaluationScope {
  form_ref: ScopeFormRef
  targets: exact ordered tuple<SubjectTargetRef>
}

SubjectTargetRef {
  case_subject_role: CaseSubjectRoleRef
  anchor_subject_artifact: exact bound Artifact reference
  target:
      ArtifactRoot(exact ArtifactIdentity)
    | ArtifactLocal(exact typed ArtifactReference<T>)
}
```

The target tuple has exactly the descriptor arity and uses descriptor role
order. It is never sorted as an unordered set. A zero-role form denotes the
entire exact Evaluation case. A one-role form can denote one subject Artifact
root or one artifact-local object. A multi-role form denotes an ordered
relation such as `source, target` or `reference, candidate`.

An anchor must be an exact member of the selected case subject-role binding.
The target Artifact must be that anchor or occur in its exact semantic
dependency closure. The target's Artifact family, not Evaluation, owns the
closed local entity and structural-reference kinds, their payload encoding,
and target validation. A query descriptor imports those types unchanged and
owns any additional relation-specific verifier. Evaluation does not define a
global `EvaluationEntityKind`, generic string path, source location, printer
position, or consumer-local index.

The canonical scope key is the form ordinal, exact arity framing, and each
fully framed target in descriptor role order. It includes the case role,
anchor identity, target Artifact identity, target-kind discriminator, and
family-owned canonical local payload. The Request verifier rejects a foreign
role, unbound anchor, unreachable target Artifact, wrong target kind,
malformed local payload, or failed relation-specific verification. Large
traces, waveforms, histograms, and timelines remain detailed Artifacts rather
than scopes.

## Metric Registry

Every `MetricKind` descriptor owns exactly these facts:

```text
stable enum value and canonical spelling
semantic definition
value type and domain
physical dimension and canonical unit
owned EvaluationScope form descriptors
permitted observation forms and typed reasons
```

The enum and its single registry definition own the exact set of kinds.
Parsing, printing, enumeration, and descriptor lookup derive mechanically from
that definition. Components must not repeat spellings or local unit tables.

Cycle count and physical time are distinct kinds. Total energy and energy per
work are distinct kinds. Quantities with different ground-truth definitions or
scope semantics remain distinct even when a tool gives them similar labels.

The descriptor never owns optimization direction, target, tolerance,
normalization, threshold, weight, score, or candidate acceptance. Those facts
belong to resolved DSE policy.

## Finding Registry

Every `FindingKind` descriptor owns:

```text
stable enum value and canonical spelling
semantic definition
owned EvaluationScope form descriptors
typed occurrence payload schema
```

For an execution-terminal finding, the descriptor additionally owns the
terminal-witness payload schema, while its Evidence occurrence carrier is the
`TerminalWitnessRef` defined by Simulation Artifacts. The concrete witness
instance is owned by the referenced `SimulationExecution`; Evidence never
copies it. Nonterminal findings continue to store their registry-defined
typed occurrence payloads inline.

The registry does not own severity, candidate acceptance, or a numeric score.
For example, deadlock, functional mismatch, negative slack, or a physical-rule
violation can be present in a successfully completed Evaluation. The resolved
DSE policy decides what that observation means for a candidate.

## Canonical Numeric Values

Persistent Evaluation values never use a bare floating-point number paired
with a unit string. The core value algebra is:

```text
IntegerValue:
  signed canonical integer

DecimalValue:
  signed canonical integer coefficient
  signed base-10 exponent

ExactRatio:
  unsigned canonical numerator
  positive unsigned denominator
```

Discrete counts use `IntegerValue`. Physical time, area, power, energy,
bandwidth, and other continuous quantities use normalized `DecimalValue` in
the descriptor's canonical unit. A nonzero decimal removes trailing decimal
zeros from its coefficient and adds them to its exponent; zero has coefficient
zero and exponent zero. Overflow during normalization is invalid.

Evaluators may calculate with tool-native or floating-point values internally.
Finalization converts them deterministically according to the exact model
descriptor's precision and rounding contract. Original text and units remain
in the raw detailed bundle.

`ExactRatio` is not a third `MetricValue` form. It is used only by typed fields
whose semantics are an exact dimensionless ratio or probability, or an exact
coordinate or phase in reference cycles. Numerator and denominator are `uint64`; the
denominator is nonzero; the pair is reduced by greatest common divisor; and
zero has the sole encoding `0/1`. Arithmetic used for validation or
normalization is checked. Absolute physical quantities remain `DecimalValue`,
so Decimal and Ratio never compete to encode the same fact.

## Evaluation Conditions

Base and request-specific conditions use one closed tagged union:

```text
EvaluationCondition {
  kind: EvaluationConditionKind
  payload: kind-owned typed payload
}
```

The central Condition registry is the sole owner of each kind's spelling,
semantic definition, typed payload schema, allowed containing locations,
assignment-key projection, canonical encoder, and validator. The containing
field determines `Base`, `MetricRequest`, or `FindingRequest`; location is not
copied into the condition payload.

Semantic applicability has three nonoverlapping owners:

* the exact `EvaluationCaseSignature` permits base-condition kinds and targets;
* a Metric or Finding descriptor permits request-specific condition kinds; and
* an `EvaluationModelDescriptor` declares which permitted conditions it
  consumes, requires, or proves irrelevant to all of its requested outputs.

The model cannot redefine a condition payload or silently ignore an
unrecognized condition. Model effort belongs to the model binding. Executable
paths, timeout, host parallelism, licenses, and scratch locations remain
nonsemantic execution bindings.

Schema 1.0 has these condition kinds:

```text
ProcessCorner {
  target: SubjectTargetRef
  corner: exact TechnologyCornerRef
}

SupplyVoltage {
  power_domain: SubjectTargetRef
  volts: positive DecimalValue
}

Temperature {
  thermal_domain_or_root: SubjectTargetRef
  kelvin: positive DecimalValue
}

RequiredClockPeriod {
  clock_domain: SubjectTargetRef
  seconds: positive DecimalValue
}

RelativeClockSchedule {
  reference_clock: SubjectTargetRef
  dependent_clock: SubjectTargetRef
  dependent_period_per_reference_period: positive ExactRatio
  dependent_phase_in_reference_cycles: ExactRatio
}

ActivityBinding {
  target: SubjectTargetRef
  source:
      ExecutionActivity {
        simulation_execution_ref: exact SimulationExecution reference
        activity_summary_ordinal: uint64
      }
    | ExplicitAssumption {
        clock_domain: SubjectTargetRef
        static_probability: ExactRatio
        transitions_per_clock: ExactRatio
      }
}

Quantile {
  probability: ExactRatio
}
```

`TechnologyCornerRef` is an exact family-owned typed reference into immutable
technology data. Its provider Artifact owns the corner catalog; a bare
`"slow"` or tool-private corner string is invalid. The Request verifier checks
subject and model-input compatibility with that exact technology data.

`RelativeClockSchedule` requires distinct clock domains. Its period ratio is
positive. Phase is normalized modulo the dependent period into
`[0, dependent_period_per_reference_period)`. It denotes dependent clock edges
at `phase + k * period_ratio` in reference cycles. Absolute clock targets use
`RequiredClockPeriod` and `DecimalValue` seconds.

`ExecutionActivity` refers to one exact summary owned by
`SimulationExecution`; that family owns canonical summary order, ordinal range,
source-basis coverage, and Request-lineage validation.
`ActivityBinding.target` is the destination Evaluation target to which the
evaluator projects that summary. It does not identify or override the
summary's actor, Fabric, or HardwareImplementation source basis. The evaluator
must prove that its model accepts the selected payload kind, window, coverage,
and exact source-to-target lineage. Missing targets in a partial summary are
unknown and cannot be interpreted as zero or filled by a hidden default.

`ExplicitAssumption` is a small uniform vectorless assumption, not an Activity
Artifact or arbitrary per-signal map. Static probability is in `[0,1]`;
transition density is nonnegative and is measured per selected clock. An
assumption requiring richer activity must use an exact execution summary or be
introduced later as a new typed condition, never as an opaque property bag.

`Quantile` is request-specific and Metric-only in schema 1.0. Its probability
is in `[0,1]`. Sample aggregation uses nearest-rank semantics: after canonical
sorting of a nonempty sample set, `q = 0` selects the first sample and otherwise
selects zero-based index `ceil(q * N) - 1`. A different quantile definition is
a different future typed condition or formula contract.

The other six kinds are Base-only in schema 1.0. Their assignment keys are,
respectively, target, power domain, thermal domain or root, clock domain,
ordered clock pair, and activity target. `Quantile` has one empty assignment
key within its containing request.

Condition collections sort by:

```text
(EvaluationConditionKind, canonical assignment key, complete payload key)
```

An exact duplicate is invalid. Two values with the same kind and assignment
key but different payloads are a conflict and are also invalid. Values of the
same kind with different assignment keys are legal, such as voltage
conditions for two distinct power domains. There is no last-wins behavior,
override layer, generic predicate DSL, or string-key escape hatch.

## Queries

`MetricQuery` pairs one registered `MetricKind` with one valid
`EvaluationScope`. `FindingQuery` does the same for `FindingKind`.

Conditions are not embedded in either query atom. `EvaluationRequest` pairs a
query with a canonical set of typed `EvaluationCondition` values. This permits
the same query to be requested under distinct percentiles or other
query-specific conditions without changing the query definition.

Canonical query collections sort by registry kind and the complete canonical
scope key. Exact duplicates are invalid. Empty/nonempty requirements belong to
the containing Request, not to a generic collection helper.

## Metric Observations

A metric observation has three orthogonal dimensions:

```text
ObservationForm = Point | Interval | Censored | NotApplicable
UncertaintyKind = ExactWithinModel | Bounded | Statistical | Unknown
EvidenceMethod  = derived from EvaluationModelDescriptor
```

Point and interval values must match the descriptor's value type and domain.
Intervals have ordered compatible bounds. Censored observations carry the
known bound or bounds and a descriptor-permitted typed reason.
`NotApplicable` carries a typed reason and no numeric value. Timeout,
cancellation, unsupported execution, and tool failure are Evidence outcomes,
not observation forms.

The method is recovered from the exact model descriptor and is not copied into
the observation. Exactness means exact within that model's declared semantics;
it does not claim physical-world accuracy.

Within persistent `EvaluationEvidence`, a `MetricResult` references the exact
Request-local `MetricRequestOrdinal` and stores only its observation form,
value or bounds, uncertainty, and permitted calibration-input references. The
Request and registry recover the kind, scope, conditions, dimension, unit, and
model method. A result must not duplicate them.

## Finding Results

A persistent finding result references one exact Request-local
`FindingRequestOrdinal` and has one of these states:

```text
Absent
Present(nonempty canonical typed occurrence set)
NotApplicable(typed reason)
```

Absence is explicit; a missing result cannot prove that a finding is absent.
Completed Evidence contains exactly one result for every requested metric and
finding and no unsolicited results.

For an execution-terminal `FindingKind`, each `Present` set contains the
registry-permitted reference carrier:

```text
TerminalWitnessRef {
  execution_output_slot_ref: ModelOutputSlotRef
  execution_output_ordinal: uint64
}
```

The containing Evidence and its exact Request resolve this pair to one
`SimulationExecution` in `output_bindings`. The referenced execution must
belong to the same Request and contain a `Halted` terminal of the requested
kind. The witness payload remains in that terminal. No direct execution
Artifact reference, copied witness payload, or witness ordinal is stored in
the finding result.

## Derived Metrics

A reusable derived quantity is produced by an ordinary typed
`DerivedMetricModel`. Its Request binds exact upstream Evidence through
descriptor-owned input slots and selects a versioned `FormulaKind`. The model
checks input kinds, canonical units, scopes, case compatibility, and formula
preconditions, then propagates bounds and uncertainty by the formula's typed
rules.

Representative formulas include runtime from cycle count and clock period,
energy from power and runtime, throughput from work and runtime, and
performance per area. Unsupported or not-applicable inputs never become zero,
infinity, or NaN.

Benchmark weighting, normalization, Pareto preference, annealing cost, and
other candidate-ranking aggregates remain DSE policy. They do not create a
MetricKind, FormulaKind, or Evidence artifact.

## Persistence Boundary

Metric, finding, scope, condition, Decimal, query, and result encodings are
reusable value schemas inside `evaluation.request.1.0` and
`evaluation.evidence.1.0`. `ExactRatio` is the same canonical scalar wire
wherever an exact typed reference-cycle coordinate or phase is required,
including `SimulationExecution`; consumers must not redefine it. None of these
schemas creates an independent Metric, Finding, condition, query-set, or
report artifact family. Canonical encoders use fixed field ordering and enum
spellings, integer JSON tokens for integer values, Decimal components, and
ExactRatio components, and strict rejection of unknown fields or noncanonical
bytes.

Raw tool reports, distributions, samples, logs, and trace chunks belong to
immutable detailed bundles. A workload execution's typed trace manifest and
ordering and its exact typed activity summaries belong to
`SimulationExecution`. Normalized observations and findings belong only to
exact Evaluation Evidence.

## Anchor Tests

Stable tests cover:

* one registry authority for enum conversion and descriptor lookup;
* shared case-signature roles producing model-independent scope keys;
* zero-, one-, and multi-role scope validation, including foreign anchors,
  unreachable targets, wrong target kinds, and role-order sensitivity;
* Decimal normalization, ExactRatio normalization, invalid denominators, and
  checked-overflow rejection;
* condition location, applicability, exact-duplicate, conflicting-assignment,
  and distinct-target behavior;
* value-domain, interval, censored, and not-applicable validation;
* deterministic query ordering and duplicate rejection;
* activity-summary ordinal resolution, destination-target compatibility,
  missing-is-unknown behavior, and rejection of incompatible payload,
  coverage, or lineage;
* completed-result totality, explicit finding absence, and terminal-witness
  reference resolution; and
* derived-formula type, unit, scope, and bound propagation.

Tests must not enumerate every registry entry, PVT permutation, clock ratio,
tool report field, JSON formatting variant, or DSE policy combination.
