# Evaluation Metric Primitives

This specification records the implemented boundary of the first Evaluation
slice. The typed C++ API in `include/Evaluation/Metric.h` is authoritative.
Canonical JSON is its persistent interchange form, not a second schema owner.

## Shared Artifact Atoms

`include/Common/Artifact.h` owns these repository-wide atoms:

- `SchemaVersion`
- `ArtifactIdentity`
- `ArtifactReference<EntityId>`

Mapping preserves source names through aliases to the Common types. Loom does
not preserve the old internal C++ mangled ABI and defines no compatibility
wrapper, second representation, or conversion path.

`include/Common/ArtifactText.h` owns canonical `X.Y` schema-version text and
lowercase hexadecimal `ArtifactIdentity` text. Individual artifact schemas
still own their supported schema identities, versions, and canonical semantic
bytes. A required contextual reference structurally contains a valid,
fixed-width `ArtifactIdentity`; absence is represented outside the reference.
These text codecs parse and format the Common value without defining a second
identity recipe.

## Metric Registry

The closed registry contains only metrics used by this slice:

| MetricKind | Value | Unit | Forms | Censored reason |
| --- | --- | --- | --- | --- |
| `cycle_count` | non-negative integer | cycle | all | `subject_did_not_complete` |
| `clock_period` | positive decimal | second | point, interval, not applicable | none |
| `runtime` | non-negative decimal | second | all | `subject_did_not_complete` |

The descriptor registry is the only metric list. Enumeration, lookup, parsing,
and printing derive from it. Each descriptor owns its spelling, semantic
definition, value type, dimension, unit, domain, scope permission, observation
forms, and censored-reason policy. It contains no optimization direction,
threshold, weight, score, objective, or acceptance policy.

## Values And Scope

`IntegerValue` is a signed 64-bit integer. Metric validation applies the
registry domain after construction.

`DecimalValue` is a signed 64-bit coefficient and signed base-10 exponent.
Construction removes trailing decimal zeros from a nonzero coefficient and
adds them to the exponent. Zero is always represented as coefficient zero and
exponent zero. Normalization that would overflow the exponent is rejected.

`MetricScope` is either the whole subject or one
`ArtifactReference<MetricEntityId>`. An entity scope always contains an exact
finalized artifact identity. Multi-entity relations and string paths are
outside this slice.

## Metric Queries

`MetricQuery` is an in-memory value pairing one registered `MetricKind` with
one typed `MetricScope`. Query and observation validation use the same scope
validation implementation.

`canonicalizeMetricQueries` accepts an empty list and returns a validated copy
ordered by metric registry spelling, scope kind, and, for entity scopes,
artifact identity bytes followed by `MetricEntityId`. Exact duplicate queries
are rejected. The same metric at distinct typed scopes remains distinct.

`MetricQuery` is also a canonical persisted Evaluation primitive. The public
persistence API is:

- `llvm::Expected<std::string> serializeMetricQuery(const MetricQuery &)`
- `llvm::Expected<MetricQuery> parseMetricQuery(llvm::StringRef json)`

The document uses schema identity `evaluation.metric_query` and version `1.0`.
Versions are strings in `X.Y` form: `X` changes are breaking/incompatible and
`Y` changes are non-breaking. The root contains exactly one metric and scope.

`canonicalizeMetricQueries` remains the sole authority for in-memory query
sets. This slice defines no independent persisted query-set root or schema. A
query has no metric-specific condition map. A future request type, rather than
this value atom, may require a nonempty query list and own its persisted list
shape.

## Observations

`MetricObservation` separates:

- the registered metric and typed scope;
- `UncertaintyKind`;
- one value form: point, interval, censored, or not applicable.

Intervals require ordered, type-compatible bounds. Numeric values must match
the registry type and domain. `subject_did_not_complete` is valid only for
`cycle_count` and `runtime`; it requires a lower bound and forbids an upper
bound. `clock_period` forbids censored observations. A not-applicable
observation has a typed reason, no numeric value, and uses unknown uncertainty.
Evidence method and execution status remain outside this slice.

## Canonical JSON

`evaluation.metric.1.0`, represented by root fields
`schema: "evaluation.metric"` and `schema_version: "1.0"`, serializes one
`MetricObservation`. `evaluation.metric_query.1.0`, represented by
`schema: "evaluation.metric_query"` and `schema_version: "1.0"`, serializes
one `MetricQuery`. These are cold-path representations of the typed C++ model,
not independent schema authorities.

Canonical JSON has fixed field ordering, compact encoding, lowercase
hexadecimal artifact identities, canonical enum spellings, and integer JSON
tokens for integer values, decimal coefficients, and base-10 exponents.
Decimal values are never encoded as JSON floating-point numbers. Parsers reject
unknown fields at every object level, unsupported schema identities or
versions, malformed artifact and entity references, invalid typed values or
scopes, trailing JSON, reordered or otherwise noncanonical bytes, and any input
whose reserialization differs.

## Explicit Exclusions

This slice does not define Evaluation Request or Evidence objects, case keys,
model descriptors or registries, metric-specific conditions, query hashing,
persisted query-set artifacts, derived metrics, tool execution, artifact
storage, training, incremental evaluation, simulator or PnR report migration,
or any score, objective, or acceptance field.
