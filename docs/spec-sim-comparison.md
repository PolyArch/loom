# Simulation Comparison

Simulation comparison is an ordinary Evaluation model. It is not a report
pipeline, a special artifact family, or a simulator mode.

## Inputs And Output

The model descriptor owns two role-labeled subject slots,
`reference_execution` and `candidate_execution`, both accepting exact
`SimulationExecution` artifacts. The ordinary `EvaluationRequest` binds those
slots and canonical sets of typed `MetricRequest` and `FindingRequest`
records. The model is not restricted to DFG versus CGRA and may compare
repeated runs or RTL/system executions when their observables are compatible.

The output is one ordinary `EvaluationEvidence` plus optional raw detailed
material. There is no `ComparisonReport` artifact. Human-readable tables and
dashboards are removable projections of the request, executions, and Evidence.

## Comparability Gate

Before comparing values, the model proves that the requested observations have
compatible meaning. Required checks include:

* compatible workload and runtime-input identities;
* compatible visible value, stream, and logical-memory output contracts;
* compatible terminal status and modeled semantic scope;
* matching metric identity, unit, statistic, aggregation, and scope; and
* any model-specific role requirements.

Hardware Mapping identities may differ by design. A CGRA execution must still
identify the exact Mapping it consumed, while a DFG execution has no Mapping
subject. Subject differences are accepted only when the comparison model
explicitly defines how their common observations align.

A statically unsupported subject relation, query, condition, or result form is
rejected by the Request verifier and produces no Evidence. After a Request has
passed static verification, a relation whose support depends on execution-time
facts may return `Unsupported`. When the model supports the Request but one
requested observation has no compatible relation for the concrete executions,
that request item receives the existing typed `NotApplicable` result. It is
not converted, zero-filled, silently omitted, or represented by a second
comparison-status algebra.

## Functional Comparison

For a finalized Canonical Dataflow Program, equivalent runtime input, and a
legal complete Mapping, DFG-sim and CGRA-sim must agree on:

* returned values;
* ordered stream payloads and termination;
* externally visible logical-memory state or diffs; and
* graph completion outcome.

A mismatch between complete compatible observations is a correctness finding.
Different terminal forms under the same requested semantic execution contract
are therefore a functional or completion mismatch, not an escape from
comparison. Missing or invalid producer references and statically unsupported
capabilities cannot form a valid comparison Request. Runtime-dependent lack of
support is `Unsupported` rather than a comparison finding. A concrete
requested observation that is outside an otherwise supported relation is
`NotApplicable`. Hardware timing differences never excuse a functional
mismatch.

## Performance Comparison

Only metrics with compatible central definitions can be related. DFG logical
cycles, CGRA cycles, RTL cycles, wall time, frequency, latency, throughput,
power, and dimensionless work scores are distinct metrics unless a named
`DerivedMetricModel` defines the conversion or composition.

For example, DFG operation count divided by CGRA cycle count is not a speedup.
Likewise, cycle count multiplied by a frequency from another implementation is
invalid unless exact subject compatibility is established by a derived model.

Expected hardware-aware causes of a CGRA/DFG cycle difference include finite
compute, route, memory, buffer, and tag resources; physical latency;
backpressure; arbitration; temporal sharing; and selected Fabric configuration.
System cache, coherence, NoC, and InstructionCore effects require sys-sim
Evidence rather than being attributed to CGRA-sim.

## Determinism And Results

Given the same request and exact input artifacts, comparison produces the same
Evidence. Canonical role order and metric identities provide tie breaks; file
order and presentation order do not.

Comparison uses only the ordinary Evaluation result algebra. A requested
functional-mismatch finding is `Absent` for a match, `Present` with typed
occurrences for a mismatch, or `NotApplicable` when the concrete observation
cannot participate in the otherwise supported relation. Requested metric
relations similarly return ordinary `MetricResult` values or
`NotApplicable`. There is no generic comparable/not-comparable result and no
unsolicited match or difference record.

An invalid Request is rejected by the Request verifier and produces no
Evidence. A missing finalized producer leaves the controller obligation
`Incomplete`. `Unsupported`, `ExecutionFailed`, and `CancelledOrTimeout`
outcomes have no finding or metric results.

## Anchor Verification

Stable tests cover exact-role validation, visible-memory alignment, functional
mismatch detection, rejection of incompatible metrics, and deterministic
Evidence. Tests do not pin table formatting or duplicate simulator semantics.
