# Fidelity Ladder

This document specifies Loom's cross-tool fidelity ladder. The ladder
defines which evidence level produced each metric and how metrics from
DFG-sim, CGRA-sim, RTL, EDA tooling, FPA estimation, and calibrated
models may be combined.

## Purpose

The fidelity ladder answers this question:

```text
Which performance, frequency, area, power, energy, and confidence
numbers are available for this workload and hardware candidate, and
what evidence supports each number?
```

Fidelity is evidence quality. It is not project maturity and not a
claim that a lower level is inaccurate. Lower-fidelity levels are still
useful when their limits are explicit.

## Fidelity Levels

### `analytic`

Analytic model evidence from Fabric ADG, mapping summaries, calibrated
tables, or custom cost models. Analytic evidence may cite backend
reports as calibration inputs, but it remains analytic unless the metric
is directly produced by a backend evidence class.

Inputs: Fabric ADG, optional mapping artifact, cost model identity,
calibration identity, optional activity assumption.

Outputs: estimated frequency, area, power, energy, confidence, and
diagnostics.

Unavailable metrics: direct backend timing closure, direct backend
area, direct backend power, and measured workload activity unless an
activity source is separately provided.

### `dfg_software`

Software-only functional and optimistic performance evidence from
DFG-sim.

Inputs: dataflow IR, runtime input data, initial memory state, DFG-sim
configuration.

Outputs: functional outputs, memory diffs, token counts, event counts,
critical-path or optimistic step estimates, diagnostics.

Unavailable metrics: hardware frequency, physical area, hardware
power, route activity, resource contention, timing closure.

### `cgra_mapped`

Hardware-aware mapped execution evidence from CGRA-sim.

Inputs: dataflow IR, Fabric ADG, mapping artifact, runtime input data,
initial memory state, CGRA-sim configuration.

Outputs: functional outputs, memory diffs, cycles, resource activity,
route activity, queue occupancy, stalls, memory and coherence activity,
temporal reuse, diagnostics.

Unavailable metrics: final RTL timing, backend area, backend power
unless separately combined through FPA.

### `mapped_activity`

Analytic or calibrated FPA evidence that consumes mapping records and
CGRA-sim activity. It is distinct from direct backend evidence.

Inputs: Fabric ADG, mapping artifact, CGRA-sim report, activity
summary, FPA model identity, calibration identity.

Outputs: activity-aware estimated power, energy, resource activity
breakdowns, confidence, and diagnostics.

Unavailable metrics: direct RTL timing, direct backend area, direct
backend power, and physical routing evidence.

### `rtl_functional`

Functional evidence from generated RTL and a harness or testbench.

Inputs: RTL manifest, SystemVerilog source set, testbench or harness,
optional mapped workload package.

Outputs: pass/fail status, observable output comparison, waveform or
trace references, simulation diagnostics, optional activity artifacts.

Unavailable metrics: signoff frequency, final physical area, final
power unless a backend profile also produces those reports.

### `rtl_structural`

Structural area and timing evidence from RTL elaboration, synthesis, or
structural analysis.

Inputs: RTL manifest, SystemVerilog source set, constraints, selected
tool profile, selected library profile.

Outputs: structural area, resource counts, estimated or achieved
frequency, timing status, backend diagnostics.

Unavailable metrics: workload-specific dynamic power unless activity is
provided.

### `rtl_activity`

Power and energy evidence from RTL or backend switching activity.

Inputs: RTL manifest, SystemVerilog source set, activity data,
synthesis or power profile, library profile.

Outputs: dynamic power, leakage power when available, energy when
combined with runtime, activity coverage, diagnostics.

Unavailable metrics: physical routing effects unless the backend
profile provides them.

### `physical_estimate`

Physical or floorplan-aware backend evidence.

Inputs: RTL manifest, constraints, floorplan or physical profile,
selected library profile, optional activity data.

Outputs: physical timing, physical area, interconnect-aware estimates,
power estimates when supported, diagnostics.

Unavailable metrics: signoff claims unless the selected profile
explicitly declares signoff-level evidence.

### `fpga_estimate`

FPGA-oriented synthesis, implementation, or prototyping evidence.

Inputs: RTL manifest, FPGA profile, part or platform profile,
constraints, optional activity data.

Outputs: resource utilization, timing estimate, frequency estimate,
power estimate when available, diagnostics.

Unavailable metrics: ASIC standard-cell area and ASIC power unless a
separate ASIC profile produces them.

### `custom_calibrated`

User-provided calibrated model evidence.

Inputs: model identity, model version, calibration identity, required
input artifacts, configuration.

Outputs: metrics declared by the model, confidence, diagnostics.

The report must identify the model and must not present custom metrics
as if they were generated by a built-in simulator or EDA backend.

## Metric Availability

Every metric record in a Loom report must identify its fidelity level
and evidence source.

| Metric class | Required source rule |
|--------------|----------------------|
| Functional result | DFG-sim, CGRA-sim, RTL functional, native app run, or CMSIS run. |
| Steps or cycles | DFG-sim optimistic steps or CGRA-sim / RTL cycle evidence. |
| Activity | CGRA-sim activity, RTL waveform/activity, backend activity, mapped activity, or custom model. |
| Frequency | Analytic FPA model, RTL structural timing, physical estimate, FPGA estimate, or custom model. |
| Area | Analytic FPA model, RTL structural report, physical estimate, FPGA estimate, or custom model. |
| Dynamic power | Analytic FPA with explicit activity source, mapped activity, backend power, FPGA estimate, or custom model. |
| Leakage power | Analytic FPA, mapped activity, or backend evidence that supports leakage reporting. |
| Energy | Derived from runtime and power with both source records visible. |
| Throughput | Derived from workload size and runtime or cycles plus frequency. |
| Latency | Derived from functional execution, cycles, or runtime evidence. |
| Confidence | Required for analytic, mapped activity, and custom models, optional for direct backend evidence. |

Unsupported metrics must be absent or marked unsupported with a
diagnostic. A report must not fill unsupported hardware metrics with
zero unless zero is the measured or modeled value and the evidence says
so.

## Composition Rules

A combined report may compose metrics from different fidelity levels.
For example, CGRA-sim cycles may be combined with an RTL structural
frequency estimate to derive runtime. The report must preserve the
source identity of both numbers.

Derived metrics must record:

* formula name;
* input metric identities;
* fidelity levels of the input metrics;
* derived metric value;
* units;
* diagnostics or confidence notes.

If two metric sources disagree, the combined report records both values
and classifies the disagreement. It must not silently replace one value
with the other.

## Ordering And Comparison

DFG-sim performance is an optimistic software baseline. CGRA-sim should
not report a more optimistic constrained execution for the same
workload and input unless the report records a valid explanation such
as different modeled work, bypassed functionality, or an unsupported
comparison.

Higher-fidelity hardware evidence may refine or override lower-fidelity
estimates only in derived summaries that explicitly cite the newer
evidence. The original lower-fidelity report remains unchanged.

## Fingerprint Mismatch Handling

Reports must reject mismatched fingerprints when fingerprints are
present. If a required fingerprint is absent, the report may proceed
only when the selected reproducibility mode permits missing
fingerprints and records that reduced confidence.

Fingerprint checks apply to:

* dataflow IR;
* Fabric ADG;
* mapping artifacts;
* runtime input data;
* RTL manifests;
* activity files;
* backend reports;
* tool and library profiles;
* calibration models.

## Relationship To Other Specs

DFG-sim is specified in `docs/spec-sim-dfg.md`.

CGRA-sim is specified in `docs/spec-sim-cgra.md`.

Simulation comparison is specified in `docs/spec-sim-comparison.md`.

RTL lowering is specified in `docs/spec-rtl-lowering.md`.

EDA tooling profiles are specified in `docs/spec-eda-tooling.md`.

FPA estimation is specified in `docs/spec-fpa-estimation.md`.

Full-stack report packaging is specified in
`docs/spec-full-stack-reporting.md`.

## Acceptance Criteria

The fidelity ladder target is complete when:

* every reportable metric class has a source rule;
* every metric record names its fidelity level and evidence source;
* unsupported metrics are diagnosed instead of defaulted silently;
* combined reports preserve input metric identities;
* stale or mismatched inputs are rejected when fingerprints are
  available;
* DSE can compare candidate reports without confusing optimistic
  software evidence with hardware-constrained evidence.
