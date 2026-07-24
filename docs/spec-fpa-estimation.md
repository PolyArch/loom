# Frequency, Power, And Area Evaluation

Frequency, power, and area are ordinary typed Evaluation metrics. Loom does not
define an `FPAReport` artifact or a separate FPA evidence pipeline.

## Subjects

Physical metrics are evaluated against one exact `HardwareImplementation`.
Fabric alone may be the subject of an explicitly analytical architecture model,
but such a result must not be labeled as synthesis, layout, or signoff evidence.

Workload-dependent power or energy also requires compatible activity. Activity
is owned by an exact `SimulationExecution`; an evaluator may mechanically
translate its logical activity through the exact implementation activity-name
map. There is no independent `ActivityProfile` artifact.

The physical-evaluation descriptor references the shared case signature whose
required role is:

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
        activity_summary_ordinal
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

## Implementation Derivation Versus Evaluation

An external flow that creates new persistent hardware state performs a
`HardwareImplementation` derivation. Examples include synthesis that produces a
gate netlist and physical implementation that produces placed, routed, or
extracted design state. Invocation lineage records the input implementation,
tool/library bindings, and output implementation.

Analysis of an immutable implementation is Evaluation. Timing, area, power,
DRC findings, and similar observations are `EvaluationEvidence` plus raw
detailed material. Evaluation never mutates the implementation.

A backend invocation may do both in sequence, but the two semantic outputs stay
separate:

```text
derive HardwareImplementation -> evaluate that exact implementation
```

## Metrics

Metric kinds, dimensions, canonical units, scopes, and observation forms are
owned by the central registry. Representative metrics include:

* achievable or constrained clock frequency;
* critical-path delay and timing slack;
* cell, macro, routing, or total area;
* dynamic and leakage power; and
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

Fast model parameter bundles may be tracked when they are stable project inputs.
Training and calibration are explicit Evaluation/DSE workflows that create new
immutable parameter bundles and model bindings. Expensive raw tool products are
stored in the configured artifact/bundle store, not committed as routine test
fixtures.

## Tool And Library Binding

EDA execution uses the common ToolRunner and an exact model binding containing
the result-affecting tool version, technology/library data, parser, and
semantic effort. PVT, required clock, and the activity binding are typed base
conditions. Local executable paths, activation scripts, licenses, scratch
roots, and host concurrency are invocation bindings.

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
derivation from Evaluation, activity compatibility, unit-safe derived metrics,
and explicit missing/failed outcomes. Tests do not pin vendor log text or a
tool-by-tool report matrix.
