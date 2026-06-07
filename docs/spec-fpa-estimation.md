# FPA Estimation

This document specifies Loom frequency, power, and area estimation.
FPA estimation combines hardware structure, backend evidence, and
activity information to produce feedback for DSE and user reports.

## Purpose

FPA estimation answers this question:

```text
For this hardware candidate and workload evidence, what frequency,
power, and area estimates are available, at what fidelity, and with
what confidence?
```

FPA estimation consumes:

* Fabric ADG;
* optional mapping artifact;
* optional CGRA-sim report;
* optional simulation comparison report;
* optional RTL manifest from `docs/spec-rtl-lowering.md`;
* optional RTL activity data;
* selected EDA tool and library profiles from
  `docs/spec-eda-tooling.md`;
* estimation configuration.

FPA estimation produces:

* normalized FPA report;
* optional backend-specific report references;
* diagnostics.

## Fidelity Levels

FPA estimation supports multiple fidelity levels. Fidelity describes
evidence quality, not project maturity.

Baseline fidelity levels are:

* `analytic`: estimate from Fabric ADG, parameters, and calibrated cost
  tables;
* `mapped_activity`: analytic or calibrated estimate using CGRA-sim
  activity and mapping information;
* `rtl_structural`: estimate from generated RTL synthesis or structural
  analysis;
* `rtl_activity`: estimate from generated RTL plus RTL switching
  activity;
* `physical_estimate`: estimate from physical or floorplan-aware
  backend evidence;
* `fpga_estimate`: estimate from FPGA-oriented backend evidence;
* `custom`: user-defined estimation adapter with an explicit model
  name.

Reports must state the fidelity level for every frequency, power, and
area number.

## Frequency

Frequency estimation may use:

* analytical critical-path models;
* RTL synthesis timing;
* static timing analysis;
* physical estimate timing;
* FPGA timing reports;
* custom calibrated models.

Frequency results must record:

* target clock period or target frequency when provided;
* achieved or estimated clock period;
* limiting domain;
* critical-path summary when available;
* timing status;
* tool and library profile identities;
* fidelity level.

## Area

Area estimation may use:

* Fabric resource cost models;
* RTL synthesis area;
* macro or memory model area;
* physical estimate area;
* FPGA resource counts;
* custom calibrated models.

Area results must record total area and should record hierarchy or
resource breakdowns when available. A report must distinguish standard
cell area, macro area, memory area, interconnect estimate, and FPGA
resource classes when the selected backend provides those categories.

## Power

Power estimation may use:

* static analytical power models;
* CGRA-sim activity;
* RTL switching activity;
* synthesis or power-analysis reports;
* physical estimate activity;
* custom calibrated models.

Power results must distinguish dynamic power and leakage power when the
evidence supports that split. Activity source must be explicit:

* `none`;
* `default_toggle`;
* `cgra_sim`;
* `rtl_waveform`;
* `rtl_activity_file`;
* `backend_internal`;
* `custom`.

If no activity is available, the report must mark power as static-only,
default-toggle, or unsupported rather than presenting it as measured
workload power.

## Cycle-Frequency-Power-Area Feedback

CGRA-sim reports hardware-aware cycle counts and activity. FPA reports
frequency, power, and area estimates. Loom combines them in user-facing
and DSE reports to derive:

* runtime estimate from cycles and frequency;
* energy estimate from power and runtime;
* throughput or latency-per-area metrics;
* performance-per-watt metrics;
* Pareto comparisons across software mappings and hardware candidates.

The combined report must keep the source of each number visible. A
cycle count from CGRA-sim and a frequency estimate from RTL synthesis
are compatible evidence, but they are not produced by the same tool.

## Report Contract

A normalized FPA report must identify:

* hardware candidate identity;
* optional mapping artifact identity;
* optional CGRA-sim report identity;
* optional RTL manifest identity;
* selected tool profile id;
* selected library profile id;
* estimation configuration;
* fidelity level;
* frequency results;
* area results;
* power results;
* combined cycle-frequency-power-area metrics when enough inputs exist;
* backend report references;
* diagnostics.

Reports should prefer portable artifact ids and profile ids over local
paths. Private local run logs may contain local paths, but portable
summary reports should not require them for interpretation.

## Relationship To DSE

FPA reports are evidence for DSE. They may be referenced by mapping-set
manifests, hardware candidate manifests, and later PnR runs. They do
not modify Fabric ADG, dataflow IR, or mapping artifacts.

If FPA feedback motivates a new hardware candidate or mapping, a later
tool must produce a new explicit artifact.

## Error Handling

FPA estimation must distinguish:

* missing input artifact;
* missing tool profile;
* missing library profile;
* unsupported fidelity request;
* backend execution failure;
* backend parser failure;
* timing violation;
* incomplete power evidence;
* incompatible activity source;
* stale fingerprints.

Diagnostics must be structured and must not silently downgrade fidelity
without recording that downgrade.

## Non-Goals

FPA estimation is not RTL lowering. It consumes RTL artifacts when they
exist.

FPA estimation is not CGRA-sim. It consumes simulator reports when they
exist.

FPA estimation is not PnR. It does not choose software-to-hardware
mapping.

FPA estimation is not final signoff. Its reports are fast feedback for
compiler and architecture DSE unless a selected backend profile
explicitly declares signoff-level evidence.

## Acceptance Criteria

FPA estimation is complete at the target-spec level when:

* it can emit a normalized FPA report for an analytic Fabric ADG model;
* it can consume CGRA-sim activity when available;
* it can consume RTL manifest and backend reports when available;
* every frequency, power, and area number records its fidelity and
  evidence source;
* missing tools, missing libraries, stale inputs, and unsupported
  fidelity requests produce structured diagnostics;
* combined cycle-frequency-power-area reports can feed DSE without
  mutating source IR or mapping artifacts.
