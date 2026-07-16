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
`ArtifactReference<MetricEntityId>`. An entity scope requires a nonempty exact
artifact identity. Multi-entity relations and string paths are outside this
slice.

## Metric Queries

`MetricQuery` is an in-memory value pairing one registered `MetricKind` with
one typed `MetricScope`. Query and observation validation use the same scope
validation implementation.

`canonicalizeMetricQueries` accepts an empty list and returns a validated copy
ordered by metric registry spelling, scope kind, and, for entity scopes,
artifact identity bytes followed by `MetricEntityId`. Exact duplicate queries
are rejected. The same metric at distinct typed scopes remains distinct.

A query has no metric-specific condition map. It is not a persistent artifact
and defines no query JSON schema or serialization contract. A future request
type, rather than this value atom, may require a nonempty query list.

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

The only schema defined here is `evaluation.metric.1.0`, represented by the
root fields `schema: "evaluation.metric"` and `schema_version: "1.0"`. It
serializes one `MetricObservation`; it does not define a new persistent Metric
artifact family.

Canonical JSON has fixed field ordering, compact encoding, lowercase
hexadecimal artifact identities, canonical enum spellings, and integer JSON
tokens for integer values, decimal coefficients, and base-10 exponents.
Decimal values are never encoded as JSON floating-point numbers. The parser
rejects unknown fields at every object level, unsupported schema identities or
versions, noncanonical decimals, invalid observations, reordered or otherwise
noncanonical bytes, and any input whose reserialization differs.

## Explicit Exclusions

This slice does not define Evaluation Request or Evidence objects, case keys,
model descriptors or registries, metric-specific conditions, query hashing or
JSON artifacts, derived metrics, tool execution, artifact storage, training,
incremental evaluation, simulator or PnR report migration, or any score,
objective, or acceptance field.
