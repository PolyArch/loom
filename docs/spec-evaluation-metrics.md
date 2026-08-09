# Evaluation Metric And Finding Primitives

This specification owns the reusable metric, finding, scope, condition,
numeric-value, and observation algebra used by every Evaluation model.
Evaluation Request and Evidence roots are specified by
`spec-dse-feedback.md`; this document does not define a parallel result
artifact or report schema.

## Ownership

The exact Evaluation registry schema 2.0 owns every `MetricKind`,
`FindingKind`, `EvaluationConditionKind`, case, model, scope-form, and related
registry ordinal. Its typed Metric and Finding registries are the semantic
source of truth for compilation analysis, Mapping guidance, simulation, RTL
and EDA models, training, calibration, and user-facing projections. Query and
Artifact-root wire schemas remain separate owners and do not renumber these
registry domains.

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
typed `ArtifactReference`, plus the `ArtifactRootReference` and
`EncodedArtifactLocalReference` heterogeneous carriers. Evaluation reuses
those types directly. It does not define aliases with different semantics,
nullable identity sentinels, a consumer-local entity erasure, or a second
identity recipe.

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
  applicability:
      WholeExactCase
    | ExactTargetPatterns(canonical nonempty set<OrderedTargetPattern>)
  reference_cycle_requirement:
      NotRequired
    | ExactCaseUniqueReferenceCycle
}

ScopeRoleDescriptor {
  role_ref: ordinal in this form
  semantic_role
}

SubjectReferenceType =
    ArtifactRootType {
      schema: exact ArtifactSchemaDescriptor
    }
  | ArtifactLocalType {
      type: exact ArtifactLocalReferenceTypeDescriptor
    }

OrderedTargetPattern {
  case_signature: exact EvaluationCaseSignatureRef
  targets: exact ordered tuple<SubjectTargetPattern>
}

SubjectTargetPattern {
  case_subject_role: CaseSubjectRoleRef in the exact case signature
  reference_type: SubjectReferenceType
}
```

`ReferenceCycleRequirement` is encoded as an unsigned 32-bit big-endian
discriminant in declaration order. It is owned by the Metric scope-form
descriptor, is immutable after registry construction, and is not repeated in
an Evaluation Request.

`ScopeFormRef` is a stable ordinal local to one query-kind descriptor in the
exact Evaluation schema version. The containing `MetricKind` or `FindingKind`
always resolves it, so it is not a global relation registry or a free string.
`WholeExactCase` is legal exactly for a zero-role form and requires an empty
target tuple. `ExactTargetPatterns` is legal exactly for a form with one or
more roles and every admitted pattern has that exact arity. `WholeExactCase`
denotes the one exact case recovered through the Request's model descriptor and
is not a wildcard over Artifact roots, subject roles, or local reference kinds.
Model capability still decides whether that exact model can produce the query
for its exact case signature. Each exact target pattern is one complete
positional alternative, and accepted target types are not independent per-role
sets whose accidental Cartesian product admits invalid relations. A breaking
change to an existing form or pattern requires a new form or an incompatible
schema version.

Exact target-pattern collections sort by exact case-signature reference,
arity, and then each positional `(case role, root/local discriminant, owner
schema, owner-local kind when present)` key. Duplicate patterns are invalid.
This is descriptor canonicalization, not a second target-reference encoding.

The Metric registry owns one shared admissibility operation:

```text
MetricScopeAdmissionContext =
    DescriptorAdmission(EvaluationCaseSignatureDescriptor)
  | RequestAdmission(
      exact EvaluationCase,
      CaseArtifactResolution,
      ArtifactStore)

MetricScopeAdmissionResult(context) =
    Unit
      if context is DescriptorAdmission
  | optional<ReferenceCycleBasis>
      if context is RequestAdmission

validateMetricScopeAdmissibility(
  MetricKind,
  ScopeFormRef,
  context: MetricScopeAdmissionContext)
    -> MetricScopeAdmissionResult(context)
```

Both model-descriptor admission and `RequestVerifier` call this operation.
Descriptor admission returns only success or a typed failure. Request admission
returns no basis for `NotRequired`. `ExactCaseUniqueReferenceCycle` requires the
case-signature-owned `UniqueReferenceCycle` descriptor; descriptor admission
checks its static source, type, and resolver contract, while Request admission
executes that resolver for the exact case and returns the validated basis.
Metric implementations do not hardcode metric names or call a private cycle
resolver. The case-signature registry remains the sole owner of the actual
basis; the Metric registry owns only which scope form requires it. Evaluation
registry 2.0 Finding scope forms must use `NotRequired`.

The persistent value is:

```text
EvaluationScope {
  form_ref: ScopeFormRef
  targets: exact ordered tuple<SubjectTargetRef>
}

SubjectTargetRef {
  case_subject_role: CaseSubjectRoleRef
  anchor_subject_artifact: exact bound ArtifactRootReference
  target:
      ArtifactRoot(exact ArtifactRootReference)
    | ArtifactLocal(exact EncodedArtifactLocalReference)
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
closed local entity and structural-reference kinds, each stable owner-local
kind ordinal, canonical payload bytes, typed decoder, and target validation.
Evaluation stores the complete existential framing from
`docs/spec-full-stack-traceability.md`, invokes the owner codec, and then owns
only anchor/closure/pattern and relation-specific verification. It does not
define a global `EvaluationEntityKind`, `ofEntities` tuple, generic string
path, source location, printer position, consumer-local index, or bare integer
reference.

The canonical scope key is the form ordinal, exact arity framing, and each
fully framed target in descriptor role order. It includes the case role,
anchor root reference, target root reference, owner-local type descriptor, and
family-owned canonical local payload. After owner validation, the Request
verifier derives the exact `OrderedTargetPattern` from the case signature,
roles, and resolved reference types and requires one descriptor-owned exact
match for an `ExactTargetPatterns` form. A `WholeExactCase` form instead
requires zero roles and zero targets and accepts the exact case already fixed
by the Request. Validation rejects a foreign role, unbound anchor, unreachable
target Artifact, wrong schema or local kind, malformed or noncanonical local
payload, pattern role-order mismatch, or failed relation-specific verification.
Large traces, waveforms, histograms, and timelines remain attempt or scratch
material rather than scopes. They become persistent only through a future exact
owner-specific Artifact schema; Evaluation does not provide a generic raw
material owner.

## Metric Registry

Every `MetricKind` descriptor owns exactly these facts:

```text
stable enum value and canonical spelling
semantic definition
value type and domain
physical dimension and canonical unit
owned EvaluationScope form descriptors
permitted observation forms and typed reasons
permitted and required request-specific ConditionApplicabilityPatterns
```

The enum and its single registry definition own the exact set of kinds.
Parsing, printing, enumeration, and descriptor lookup derive mechanically from
that definition. Components must not repeat spellings or local unit tables.

Required request-condition patterns are checked independently for every
`MetricRequest` of that kind after ordinary condition location and
applicability validation. The containing request must include one match for
each required pattern. Existing assignment-key conflict rules still apply, so
a required pattern with one assignment key cannot be supplied twice. A model
descriptor may consume, require, or prove invariant only patterns already
permitted by the Metric descriptor; it cannot weaken a Metric-owned per-request
requirement.

The required collection is a canonical duplicate-free subset of the permitted
collection. Descriptor projection encodes the permitted collection by its
existing canonical pattern keys, then `u64be(required_count)` and the sorted
canonical keys of the required subset. An unknown pattern or a required pattern
not present in the permitted collection is a registry error.

The closed metric-value domain algebra is:

```text
MetricValueDomain =
    NonNegative
  | Positive
  | ClosedDecimalInterval {
      lower: canonical DecimalValue
      upper: canonical DecimalValue
    }
```

`ClosedDecimalInterval` is legal only for a `DecimalValue` metric; its bounds
use the descriptor's canonical unit and satisfy `lower <= upper`. Registry
admission and Evidence validation use this one domain value; a model cannot add
a private bound check or widen it. The four prediction-error kinds use
`ClosedDecimalInterval[0,2]`.

Every registered MetricKind has a nonempty scope-form table. Evaluation
registry 2.0 gives each initial metric one owner-defined form at
`ScopeFormRef(0)`:

```text
CycleCount            form 0: WholeExactCase
ClockPeriod           form 0: WholeExactCase
Runtime               form 0: WholeExactCase
LimitingClockFrequency form 0: WholeExactCase
TotalArea             form 0: WholeExactCase
DynamicPower          form 0: WholeExactCase
LeakagePower          form 0: WholeExactCase
MaximumVoltageDrop    form 0: WholeExactCase
LimitingClockFrequencyPredictionError form 0: WholeExactCase
TotalAreaPredictionError              form 0: WholeExactCase
DynamicPowerPredictionError           form 0: WholeExactCase
LeakagePowerPredictionError           form 0: WholeExactCase
```

`Runtime` and the five whole-case physical metrics cover the exact evaluated
case directly. `CycleCount` and `ClockPeriod` use the exact case signature's executable
`UniqueReferenceCycle` resolver; a model must not advertise either form when
the signature declares `Absent` and cannot choose or rederive a different basis
locally. A later clock-domain-specific form uses an exact target pattern rather
than changing form 0. An empty scope-form table, an unknown form ordinal, or a
model capability naming a form not owned by the MetricKind is invalid.

The Evaluation registry 2.0 requirements are exact: `CycleCount` form 0 and
`ClockPeriod` form 0 use `ExactCaseUniqueReferenceCycle`; every other form 0
uses `NotRequired`. These descriptor fields, not duplicated switches in model
registration or Request validation, control admissibility.

The initial physical metric semantics and canonical units are:

```text
LimitingClockFrequency : positive DecimalValue, hertz
TotalArea              : nonnegative DecimalValue, square_meter
DynamicPower           : nonnegative DecimalValue, watt
LeakagePower           : nonnegative DecimalValue, watt
MaximumVoltageDrop     : nonnegative DecimalValue, volt
LimitingClockFrequencyPredictionError : DecimalValue in [0,2], one
TotalAreaPredictionError              : DecimalValue in [0,2], one
DynamicPowerPredictionError           : DecimalValue in [0,2], one
LeakagePowerPredictionError           : DecimalValue in [0,2], one
```

`ClockPeriod` remains the duration of one exact reference cycle.
`LimitingClockFrequency` is instead the whole-case maximum common frequency
permitted by its limiting synchronous domain, so a multi-clock case does not
invent one local cycle as the whole-case reference cycle. `TotalArea` includes
cells, macros, and allocated routing footprint. Dynamic power is workload and
activity dependent; leakage power is not. A model lacking the required
activity returns typed `Unsupported`, never a hidden toggle-rate default.

`MaximumVoltageDrop` is the greatest nonnegative difference between the
applied supply voltage and delivered voltage over every supply node included
in the exact model's complete analyzed power network. Its whole-case scope is
the maximum across all analyzed power domains; a provider must not publish a
selected node, average, percentile, or vendor severity as this metric. Static
and dynamic rail analyses may produce the same MetricKind because their
method, activity basis, coverage, and uncertainty belong to their exact model
descriptors. A physical model declares the exact HardwareImplementation-
anchored `SupplyVoltage` base-condition patterns it consumes or requires. An
unavailable or incomplete rail network cannot be replaced by a nominal-
voltage default. A later per-domain question requires another owner-defined
scope form rather than a vendor-specific MetricKind.

Each prediction-error descriptor owns one permitted-and-required
request-condition pattern: `Quantile` with the exact
`fpa_model_parameter_calibration` case signature and an empty target tuple.
Thus each request is admitted only for that case and contains exactly one
`Quantile`: the required pattern provides at least one and Quantile's empty
assignment key rejects a duplicate. It reports the nearest-rank quantile over
the case's nonempty ground-truth Evidence role. Case admission has already
required one completed Point observation for each of the four source physical
metrics, so every sample has one exact observed value. For one predicted value
`p` and observed value `o`, the per-sample symmetric relative error is:

```text
0                                      when p = 0 and o = 0
2 * abs(p - o) / (abs(p) + abs(o))     otherwise
```

The model represents Decimal coefficients, powers of ten, ratio numerators,
denominators, and comparison cross-products with arbitrary-precision signed
integers. It normalizes signs and common factors, applies the shared
nearest-rank rule to exact rational ordering, and converts only the selected
value to canonical `DecimalValue` using the calibration descriptor's 18-digit
`RoundToNearestTiesToEven` contract. No finite host-integer width may turn an
otherwise valid comparison into overflow. There is no epsilon, infinity, host
floating-point ordering, or per-sample early rounding. Median and P90 are
therefore ordinary requests with `Quantile(1/2)` and `Quantile(9/10)` rather
than separate MetricKinds.

Cycle count and physical time are distinct kinds. Any future total-energy and
energy-per-work metrics would likewise require distinct registered kinds.
Quantities with different ground-truth definitions or scope semantics remain
distinct even when a tool gives them similar labels.

The descriptor never owns optimization direction, target, tolerance,
normalization, threshold, weight, score, or candidate acceptance. Those facts
belong to resolved DSE policy.

## Finding Registry

Every `FindingKind` descriptor owns:

```text
stable enum value and canonical spelling
semantic definition
owned EvaluationScope form descriptors
exact FindingOccurrenceCodec selection
optional typed terminal-witness instance schema
```

The selected occurrence schema is a complete owner codec, not a byte validator:

```text
FindingOccurrenceCodec {
  occurrence_schema_descriptor
  encode(owner_typed_occurrence) -> canonical bytes
  decode(canonical bytes) -> owner_typed_occurrence
  validate(owner_typed_occurrence, FindingOccurrenceContext)
}

FindingOccurrenceContext {
  exact EvaluationRequest
  FindingRequestOrdinal
  exact descriptor-owned output bindings of the containing Evidence
  Artifact resolution and store access required by the occurrence owner
}
```

Registration requires all four operations. Evaluation owns only outer array
framing, lowercase-hex text encoding, canonical byte ordering, duplicate
rejection, and dispatch. Import resolves array position to
`FindingRequestOrdinal`, resolves that ordinal to `FindingKind`, selects the
descriptor's exact codec, decodes through the codec owner, validates in the
exact Request and result context, re-encodes, and requires byte equality.
Evaluation may retain a type-erased handle after successful owner adoption, but
raw bytes alone are not a typed occurrence and are never exposed as a second
semantic API.

For an execution-terminal finding, the FindingKind descriptor owns the typed
`Halted` witness-instance schema and selects the `TerminalWitnessRef` occurrence
codec owned by Simulation Artifacts. The concrete witness instance is owned by
the referenced `SimulationExecution`; Evidence never copies it. Nonterminal
findings select their Finding-registry-owned inline occurrence codecs.

Evaluation registry 2.0 reserves finding ordinal `0` for
`functional_mismatch`. Its only
scope form is `WholeExactCase`, and its occurrence schema is
`evaluation.functional_mismatch.1.0`. The occurrence is a zero-field typed
singleton. `Present` contains exactly that one occurrence when two exact
functional observations differ under the comparison model's proven
deterministic relation; `Absent` means they agree. The Request already owns the
exact subjects, workload, runtime input, conditions, and scope, so the
occurrence does not copy an output path, value bytes, candidate identity, or a
diagnostic string. A comparison that cannot establish the required relation
uses the existing `NotApplicable` or non-Completed outcome rather than
reporting a mismatch.

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

Discrete counts use `IntegerValue`. Physical time, area, power,
dimensionless observed metrics, and other registered continuous quantities use
normalized `DecimalValue` in the descriptor's canonical unit. A nonzero decimal
removes trailing decimal zeros from its coefficient and adds them to its
exponent; zero has coefficient zero and exponent zero. Overflow during
normalization is invalid.

Evaluators may calculate with tool-native or floating-point values internally.
Finalization converts them deterministically according to the exact model
descriptor's precision and rounding contract. Original text and units remain
owner-attempt or scratch material and have no current Artifact schema.

`ExactRatio` is not a third `MetricValue` form. It is used only by exact typed
configuration fields whose semantics are a ratio or probability, or an exact
coordinate or phase in reference cycles. Numerator and denominator are
`uint64`; the denominator is nonzero; the pair is reduced by greatest common
divisor; and zero has the sole encoding `0/1`. Arithmetic used for validation
or normalization is checked. Observed dimensionless metrics remain
`DecimalValue` under their model descriptor's fixed precision, so Decimal and
Ratio never compete to encode the same persistent fact.

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

Each condition descriptor also owns one typed projection from a validated
payload to its exact ordered tuple of `SubjectTargetRef` values. The projection
order is semantic payload-field order:

```text
ProcessCorner             -> [target]
SupplyVoltage             -> [power_domain]
Temperature               -> [thermal_domain_or_root]
RequiredClockPeriod       -> [clock_domain]
RelativeClockSchedule     -> [reference_clock, dependent_clock]
ActivityBinding.ExecutionActivity    -> [target]
ActivityBinding.ExplicitAssumption   -> [target, clock_domain]
Quantile                  -> []
```

This projection is derived and never serialized as another list. It lets all
applicability owners use the same exact value:

```text
ConditionApplicabilityPattern {
  kind: EvaluationConditionKind
  targets: OrderedTargetPattern
}
```

Semantic applicability has three nonoverlapping owners:

* the exact `EvaluationCaseSignature` owns a canonical set of complete
  `ConditionApplicabilityPattern` values for Base conditions;
* a Metric or Finding descriptor owns the corresponding complete permitted
  patterns and its per-request required subset for request-specific
  conditions; and
* an `EvaluationModelDescriptor` declares which already-permitted exact
  patterns it consumes, requires, or proves irrelevant to all of its requested
  outputs.

The model cannot redefine a condition payload or silently ignore an
unrecognized condition. Model effort belongs to the model binding. Executable
paths, timeout, host parallelism, licenses, and scratch locations remain
nonsemantic execution bindings.

Validation first invokes every target Artifact owner's codec and validator,
then projects the ordered targets, derives their exact case roles and root or
local reference types, and requires one exact pattern match for the containing
location. An accepted-role set and an accepted-reference-type set are not
matched independently. There is no wildcard target, unordered target bag,
implicit role coercion, or model-private applicability predicate.
For a Base pattern stored by a case signature, the pattern's exact
`case_signature` must be that containing signature. Scope and request-specific
patterns retain the explicit signature because their registry owners can serve
more than one case.

Evaluation registry 2.0 has these condition kinds:

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

`TechnologyCornerRef` is exactly the `TechnologyCorner` local-reference kind
owned by `loom.implementation_platform 1.0`. Its typed API is
`ArtifactReference<TechnologyCornerId>`; its heterogeneous persistent form
uses that family's local-kind ordinal and eight-byte canonical payload. The
ImplementationPlatform owner, not Evaluation, decodes and validates it. A bare
`"slow"`, arbitrary `uint64`, Evaluation entity tuple, or tool-private corner
string is invalid. The Request verifier additionally checks that the exact
platform is admitted by the selected subject and model-input closure and that
the case-signature-owned relation permits that corner for the condition target.

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

Evaluation registry 2.0 contains the typed `ExecutionActivity` source form.
Consuming it requires an activity-summary adopter for
`loom.simulation_execution 1.0`, an ordinal resolver, same-Request validation,
and exact source-to-target lineage validation. The SimulationExecution root,
publisher, and importer do not by themselves provide that Evaluation adapter.
When the adapter is unavailable, authoring, Request finalization, and import
fail with a typed owner-unavailable error. Parsing or printing the closed
reference-plus-ordinal shape does not activate the source form and cannot
permit opaque execution bytes, skipped lineage, or a reinterpreted ordinal.

`ExplicitAssumption` is a small uniform vectorless assumption, not an Activity
Artifact or arbitrary per-signal map. Static probability is in `[0,1]`;
transition density is nonnegative and is measured per selected clock. An
assumption requiring richer activity must use an exact execution summary or be
introduced later as a new typed condition, never as an opaque property bag.

`Quantile` is request-specific and Metric-only in Evaluation registry 2.0. Its
probability is in `[0,1]`. Sample aggregation uses nearest-rank semantics:
after canonical sorting of a nonempty sample set, `q = 0` selects the first
sample and otherwise selects zero-based index `ceil(q * N) - 1`. A different
quantile definition requires a distinct registered condition contract.

MetricRequest and FindingRequest construction query the central Condition
registry's location set. Therefore `Quantile` is accepted only in a
MetricRequest condition set and is rejected in Base and FindingRequest
condition sets before request canonicalization. Finding descriptors and models
cannot widen that location set through their capability tables.

The other six kinds are Base-only in Evaluation registry 2.0. Their assignment
keys are, respectively, target, power domain, thermal domain or root, clock
domain, ordered clock pair, and activity target. `Quantile` has one empty
assignment key within its containing request.

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
UncertaintyKind = ExactWithinModel | Bounded | Statistical | Unquantified
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

Within persistent `EvaluationEvidence`, a `MetricResult` occupies the exact
Request-local `MetricRequestOrdinal` array position and stores only its
observation form, value or bounds, uncertainty, and the canonical
`calibration_input_slots` set of descriptor-local `ModelInputSlotRef` values
whose bound Artifacts calibrated that observation. The set may be empty. A
case subject, including a calibrated bundle, is represented by the Request and
must not be copied into this field. The Request and registry recover the kind,
scope, conditions, dimension, unit, and model method. A result must not
duplicate them or serialize its ordinal.

## Finding Results

A persistent finding result occupies one exact Request-local
`FindingRequestOrdinal` array position and has one of these states:

```text
Absent
Present(nonempty canonical typed occurrence set)
NotApplicable(typed reason)
```

Absence is explicit; a missing result cannot prove that a finding is absent.
Completed Evidence contains exactly one result for every requested metric and
finding and no unsolicited results.

For an execution-terminal `FindingKind`, `Present` contains exactly one
registry-permitted `TerminalWitnessRef` owned by
`docs/spec-simulation-artifacts.md`. The containing Evidence and its exact
Request resolve that owner-defined value to one
`SimulationExecution` in `output_bindings`. The referenced execution must
belong to the same Request and contain a `Halted` terminal of the requested
kind. The witness payload remains in that terminal. No direct execution
Artifact reference, copied witness payload, or witness ordinal is stored in
the finding result.

`FindingRequestOrdinal` is an unsigned 64-bit index into the Request's
canonical finding-request array. A Completed Evidence finding-result array has
length `N`; its position is the ordinal and therefore covers exactly `[0, N)`
without serializing an ordinal field. The ordinal resolves the `FindingKind`,
so a `FindingOccurrence` does not repeat a kind tag.

For `Present`, the wire is a nonempty array of owner-produced canonical payload
bytes, represented as lowercase hexadecimal in canonical JSON. Payloads sort
lexicographically by those exact bytes and duplicates are invalid. The
Simulation Artifacts owner supplies the complete type and codec for
`TerminalWitnessRef`; a terminal Finding descriptor references that codec
rather than defining another encoding.

## Derived Quantities

A reusable derived quantity is an ordinary registered `MetricKind` produced by
an exact registered Evaluation model. The model descriptor owns its input
slots and admissible case, scope, condition, unit, and compatibility rules.
Evaluation has no generic formula registry or formula DSL beside those model
descriptors.

`Runtime` is a current registered MetricKind and may be produced only by a
registered model whose exact semantics establish its required timing basis.
Energy, throughput, speedup, and performance-per-area are unsupported until
their exact MetricKind semantics and producing model owners are registered.
They are never inferred by a report, stored as independent observations under
another kind, or computed from unmatched workloads, inputs, scopes, timing
bases, or units. Unsupported or not-applicable inputs never become zero,
infinity, or NaN.

Active wall time, process CPU time, peak resident memory, worker count, and
logical CPU count recorded by `InvocationManifest` are nonsemantic operational
observations. They are not MetricKinds and do not become
`EvaluationEvidence`. A report may project them beside semantic work summaries
only while preserving their distinct owners and execution-context limits.

Benchmark weighting, normalization, Pareto preference, annealing cost, and
other candidate-ranking aggregates remain DSE policy. They do not create a
MetricKind, Evaluation model, or Evidence artifact.

## Persistence Boundary

Metric, finding, scope, condition, Decimal, and result encodings are reusable
value schemas inside `evaluation.request.1.0` and
`evaluation.evidence.1.0`. Standalone query wires have their own exact roots:
`evaluation.metric_query 1.0` and `evaluation.finding_query 1.0`. Those wire
owners carry Evaluation registry 2.0 kind and scope-form references without
owning or renumbering the referenced registries. `ExactRatio` is the same
canonical scalar wire wherever an exact typed reference-cycle coordinate or
phase is required, including `SimulationExecution`; consumers must not
redefine it. None of these schemas creates an independent Metric, Finding,
condition, query-set, or report artifact family. Canonical encoders use fixed
field ordering and enum spellings, integer JSON tokens for integer values,
Decimal components, and
ExactRatio components, and strict rejection of unknown fields or noncanonical
bytes. `SubjectTargetRef` uses the complete Common root/local-reference framing;
the owner-local payload is emitted as lowercase hexadecimal and is never
decoded by the Evaluation serializer itself.

Implementation dependency is strict: Common root/local-reference framing and
the referenced Artifact family's codec/validator must exist before Evaluation
can persist that local target. In particular, the ImplementationPlatform owner
precedes `ProcessCorner` persistence. Evaluation must report an unavailable
owner codec as an implementation/capability error; it cannot publish a fallback
integer, tuple, path, or opaque property payload.

Raw tool reports, distributions, samples, logs, and diagnostic traces remain
owner-attempt or scratch material. A workload execution's exact typed activity
summaries belong to `SimulationExecution`; its invocation-local
`SpatialDiagnosticTrace` has no persistent Evaluation form.
Normalized observations and findings belong only to exact Evaluation Evidence.

## Anchor Tests

Stable tests cover:

* one registry authority for enum conversion and descriptor lookup;
* shared case-signature roles producing model-independent scope keys;
* zero-, one-, and multi-role scope validation, including foreign anchors,
  unreachable targets, wrong owner schemas or local kinds, noncanonical
  owner-local payloads, role-order sensitivity, and one intrinsic whole-case
  form that carries no wildcard target authority;
* one shared reference-cycle admissibility path used by descriptor and Request
  validation, including missing-resolver and exact-case resolution failure;
* Decimal normalization, ExactRatio normalization, invalid denominators, and
  checked-overflow rejection;
* condition location, ordered role/reference-type applicability, exact-
  duplicate, conflicting-assignment, and distinct-target behavior, including
  Quantile rejection in Base and FindingRequest;
* one exact `TechnologyCornerRef` import plus wrong-platform, wrong-kind,
  malformed-payload, and unresolved-corner rejection;
* value-domain, interval, censored, and not-applicable validation;
* deterministic query ordering and duplicate rejection;
* symmetric prediction-error zero handling, exact pre-rounding order, fixed
  nearest-rank median and P90, and rejection of a missing or duplicate Quantile
  condition;
* activity-summary ordinal resolution, destination-target compatibility,
  missing-is-unknown behavior, and rejection of incompatible payload,
  coverage, or lineage, plus deterministic owner-unavailable rejection before
  the SimulationExecution importer is registered;
* completed-result ordinal totality, explicit finding absence, occurrence
  owner encode/decode/re-encode equality, and terminal-witness reference
  resolution; and
* derived-formula type, unit, scope, and bound propagation.

Tests must not enumerate every registry entry, PVT permutation, clock ratio,
tool report field, JSON formatting variant, or DSE policy combination.
