# Evaluation Metric And Finding Primitives

This specification owns the reusable metric, finding, scope, value, and
observation algebra used by every Evaluation model. Evaluation Request and
Evidence roots are specified by `spec-dse-feedback.md`; this document does not
define a parallel result artifact or report schema.

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

The base forms are:

```text
WholeSubject
Entity(typed ArtifactReference)
Relation(canonical typed ArtifactReference tuple)
```

Each registry descriptor states which forms and entity kinds are legal. A
relation has a descriptor-defined role order; it is not an unordered bag or a
string path. Large traces, waveforms, histograms, and timelines are detailed
artifacts, not scopes.

## Metric Registry

Every `MetricKind` descriptor owns exactly these facts:

```text
stable enum value and canonical spelling
semantic definition
value type and domain
physical dimension and canonical unit
permitted EvaluationScope forms
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
permitted EvaluationScope forms
typed occurrence payload schema
```

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

## Queries

`MetricQuery` pairs one registered `MetricKind` with one valid
`EvaluationScope`. `FindingQuery` does the same for `FindingKind`.

Conditions are not embedded in either query atom. `EvaluationRequest` pairs a
query with a canonical set of typed `EvaluationCondition` values. This permits
the same query to be requested under distinct percentiles or other
query-specific conditions without changing the query definition.

Canonical query collections sort by registry kind and canonical scope key.
Exact duplicates are invalid. Empty/nonempty requirements belong to the
containing Request, not to a generic collection helper.

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

Metric, finding, scope, value, query, and result encodings are reusable value
schemas inside `evaluation.request.1.0` and `evaluation.evidence.1.0`. They do
not create independent Metric, Finding, query-set, or report artifact families.
Canonical encoders use fixed field ordering and enum spellings, integer JSON
tokens for integer values and decimal components, and strict rejection of
unknown fields or noncanonical bytes.

Raw tool reports, distributions, samples, logs, and trace chunks belong to
immutable detailed bundles. A workload execution's typed trace manifest and
ordering belong to `SimulationExecution`. Normalized observations and findings
belong only to exact Evaluation Evidence.

## Anchor Tests

Stable tests cover:

* one registry authority for enum conversion and descriptor lookup;
* `EvaluationScope` validation shared by metric and finding queries;
* decimal normalization and invalid overflow;
* value-domain, interval, censored, and not-applicable validation;
* deterministic query ordering and duplicate rejection;
* completed-result totality and explicit finding absence; and
* derived-formula type, unit, scope, and bound propagation.

Tests must not enumerate every registry entry, tool report field, condition
permutation, JSON formatting variant, or DSE policy combination.
