# Mapping Verification and Diagnostics

This document specifies verifier behavior and diagnostic records for
Loom mapping artifacts.

The verifier is a consumer-facing legality checker. It does not run PnR,
repair mappings, choose fallback placements, infer routes, or mutate
software or hardware IR.

## Verifier Inputs

Required inputs:

* mapping artifact;
* referenced software dataflow IR;
* referenced Fabric ADG;
* referenced `fabric.module` templates.

Optional inputs:

* workload shape or profile referenced by the artifact;
* DFG-sim, CGRA-sim, FPA, or DSE report referenced by metrics;
* policy profile controlling which optional record families are required
  for a specific consumer.

## Verification Order

The verifier runs in deterministic order:

1. Schema and identity checks from `docs/spec-mapping-identity.md`.
2. Software and hardware reference resolution.
3. Placement checks from `docs/spec-mapping-placement.md`.
4. Routing checks from `docs/spec-mapping-routing.md`.
5. Schedule, sharing, tag, reconfiguration, and buffer checks from
   `docs/spec-mapping-schedule-buffer.md`.
6. Memory checks from `docs/spec-mapping-memory.md`.
7. Visualization checks from `docs/spec-mapping-visualization.md` when
   visualization metadata is present.
8. Consumer-profile checks, such as CGRA-sim-required schedule records
   or RTL-required buffer records.

The verifier may collect multiple diagnostics in one run. It must not
stop at the first error unless a missing input prevents safe reference
resolution.

## Diagnostic Record

A diagnostic record is part of an artifact when PnR emits a partial,
rejected, or degraded mapping. A verifier may also emit diagnostics as a
separate report without modifying the artifact.

Required fields:

* `record_id`.
* `severity`: `error`, `warning`, `note`, or `metric`.
* `code`: stable diagnostic code.
* `message`: human-readable message.
* `subjects`: non-empty list of software, hardware, or mapping
  references.
* `rule`: specification rule or verifier category.

Optional fields:

* `producer`: tool that emitted the diagnostic.
* `policy`: policy profile under which the diagnostic applies.
* `repair_hint`: human-readable hint. It must not be interpreted as an
  automatic rewrite rule.
* `related`: related diagnostic record IDs.

Diagnostic messages are for humans. Tests and tools key on `code`,
`severity`, `subjects`, and `rule`.

## Required Diagnostic Codes

The base diagnostic-code set includes:

* `schema_missing_required_field`
* `schema_unknown_required_family`
* `reference_unresolved`
* `artifact_identity_mismatch`
* `placement_incompatible_resource`
* `placement_missing_scalar_fallback`
* `route_non_contiguous`
* `route_endpoint_direction_mismatch`
* `route_missing_adapter`
* `route_implicit_fanout`
* `resource_double_booked`
* `temporal_tag_conflict`
* `buffer_missing`
* `buffer_depth_invalid`
* `memory_address_out_of_range`
* `memory_missing_coherence_domain`
* `memory_consistency_unsupported`
* `visualization_bad_reference`
* `consumer_profile_missing_record`

Implementations may add codes, but they must not change the meaning of
these base codes.

## Consumer Profiles

Different consumers require different record completeness.

Base profile:

* identity;
* placement for every mapped software object;
* routes for every mapped inter-placement edge;
* memory records for mapped memory operations;
* diagnostics for unmapped or degraded objects.

CGRA-sim profile:

* base profile;
* schedule or event ordering records sufficient to simulate conflicts;
* buffer bindings for every route or stream that can stall;
* temporal tags and resource-sharing records for shared resources;
* memory consistency and coherence records for shared memory.

RTL lowering profile:

* base profile;
* exact resource bindings;
* route, adapter, and boundary records;
* buffer depths;
* reconfiguration records;
* clock/reset/power crossing records when domains differ.

Visualization profile:

* base profile;
* visualization metadata is optional, but if present must verify.

## Determinism Requirements

For the same inputs and consumer profile, verifier diagnostics must be
emitted in deterministic order:

* severity order: error, warning, note, metric;
* then diagnostic code;
* then first subject reference;
* then diagnostic record ID.

Deterministic diagnostics are required for tests, DSE triage, and GUI
diffing.

## Validation

The verifier validates itself through conformance tests:

* one negative test per required diagnostic code;
* one positive test per detailed mapping spec;
* stale artifact identity rejection;
* arbitrary-topology route validation;
* compound-protocol channel endpoint validation;
* CGRA-sim consumer-profile completeness;
* RTL consumer-profile completeness;
* visualization metadata optionality.

## Acceptance Criteria

Mapping verification is complete when:

* invalid artifacts are rejected before CGRA-sim, runtime, RTL, or FPA
  consumers rely on them;
* diagnostics are structured and stable enough for tests and GUI tools;
* consumer profiles can require additional records without changing
  base artifact semantics;
* the verifier never repairs or completes missing mapping decisions.
