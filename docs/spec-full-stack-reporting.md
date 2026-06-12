# Full-Stack Reporting

This document specifies Loom's target report bundle for source,
compiler, mapping, simulation, RTL, EDA, FPA, and DSE evidence. The
report bundle is the user-facing and DSE-facing summary of one workload
and hardware candidate or of a set of candidates.

## Purpose

Full-stack reporting answers this question:

```text
For this source workload, hardware candidate, mapping, runtime input,
and backend profile set, what happened, which artifacts prove it, and
which metrics are available at which fidelity?
```

The bundle does not replace the underlying artifacts. It references
them by identity and summarizes their metrics, diagnostics, and
relationships.

## Bundle Kinds

### Workload Report Bundle

One workload report bundle summarizes one source workload for one
selected hardware candidate and one selected mapping candidate.

Required fields:

* report schema version;
* bundle id;
* workload identity;
* source artifact identity;
* compiler command identity;
* runtime input identity;
* selected hardware candidate identity;
* selected mapping artifact identity when hardware mapping is used;
* input artifact identities and fingerprints;
* runtime evidence summary when runtime packaging is used;
* structured diagnostic records;
* report status;
* diagnostics summary.

Optional fields:

* LLVM IR artifact identity;
* raised MLIR artifact identity;
* dataflow IR artifact identity;
* DFG-sim report identity;
* CGRA-sim report identity;
* simulation comparison report identity;
* runtime package identity;
* RTL manifest identity;
* EDA report identities;
* normalized FPA report identity;
* derived metric records;
* visualization artifact identities;
* DSE feedback record identity.

The normalized FPA report identity references the JSON artifact whose
kind is `fpa_report`. Table exports such as `rtl-fpa-summary.csv` are
metric projections and may appear as metric evidence sources, but they
do not replace the normalized FPA report identity.

### Hardware Candidate Report Bundle

One hardware candidate report bundle summarizes a Fabric ADG candidate
independent of a specific workload or across a set of workloads.

Required fields:

* report schema version;
* hardware candidate identity;
* Fabric ADG identity;
* ADG Builder recipe identity when available;
* RTL manifest identity when available;
* EDA and FPA report identities when available;
* supported workload classes;
* input artifact identities and fingerprints;
* structured diagnostic records;
* diagnostics summary.

Rules:

* When the report references an ADG hardware summary, its ADG Builder
  recipe identity must match the selected ADG hardware summary row for
  the hardware candidate. Empty identities are legal only when the
  selected row has no known ADG Builder recipe.

### DSE Report Bundle

One DSE report bundle summarizes a set of software placement,
hardware, mapping, simulator, and FPA candidates.

Required fields:

* report schema version;
* DSE run id;
* objective records;
* candidate list;
* selected candidate or Pareto set;
* rejected candidate summaries;
* referenced workload report bundle identities;
* referenced hardware candidate report bundle identities;
* referenced DSE candidate artifact identities;
* input artifact identities and fingerprints;
* structured diagnostic records;
* diagnostics summary.

The DSE report bundle references candidate artifacts. It must not
duplicate placement, routing, schedule, or hardware structure records.

## Metric Record Schema

Every metric record has these required fields:

* metric id;
* metric class;
* value;
* unit;
* fidelity level from `docs/spec-fidelity-ladder.md`;
* evidence source artifact id;
* producer component;
* derivation kind;
* diagnostics.

Optional fields are:

* input metric ids for derived metrics;
* confidence;
* lower and upper bounds;
* workload-normalized value;
* hardware-normalized value;
* objective weight or ranking contribution.

Metric classes include:

* functional status;
* optimistic steps;
* hardware cycles;
* estimated runtime;
* throughput;
* latency;
* resource activity;
* frequency;
* area;
* dynamic power;
* leakage power;
* energy;
* energy per operation;
* performance per watt;
* performance per area;
* diagnostics count.

## Summary Table Exports

Full-stack reporting must support compact table exports for regression
tracking and DSE dashboards. A table export is a projection of report
bundle metrics. It does not replace the report bundle.

The report bundle JSON is the stable program-to-program interface.
Every table export derived from a report bundle must identify the source
bundle id, bundle schema version, export profile, and source metric ids
when metrics are projected into columns.

Portable intermediate artifact gate schemas are specified in
`docs/spec-intermediate-artifacts.md`. Full-stack report bundles may
consume those artifacts, but they do not replace the per-artifact
schemas or content-audit requirements.

The required simulator cycle summary export has one row per kernel or
app case and these required columns:

* `kernel`;
* `dfg_sim_cycles`;
* `cgra_sim_cycles`.

The three required columns appear first and in that order. Additional
columns may follow when the selected profile requests more context,
such as mapping id, hardware candidate id, FPA report id, frequency,
area, power, energy, or diagnostics.

Missing simulator evidence must be represented by explicit unsupported
or diagnostic values according to the export profile. A missing DFG-sim
or CGRA-sim cycle value must not be represented as numeric zero unless
the simulator report itself produced zero.

The global evidence policy in `docs/spec-loom-stack.md` applies to
report exports. Table rows that do not trace back to real source bundle
metrics are projections or fixtures, not report evidence.

## Derived Cycle/Frequency/Power/Area Metrics

Derived cycle/frequency/power/area metrics must preserve their source
records.

Runtime is derived from cycles and frequency. The report must identify
the cycle source and the frequency source.

Energy is derived from runtime and power. The report must identify the
runtime source and the dynamic or leakage power sources.

Performance-per-watt and performance-per-area are derived metrics. They
must identify the workload-size metric, runtime or throughput metric,
power metric, and area metric used.

If any input metric is unsupported, the derived metric is unsupported
and must record the missing input rather than defaulting it.

## Diagnostics

Full-stack reports must distinguish:

* ordinary compiler failure;
* optional artifact unsupported scope;
* dataflow lowering failure;
* DFG-sim failure;
* PnR failure;
* mapping verifier failure;
* CGRA-sim failure;
* runtime package failure;
* RTL lowering failure;
* EDA tool failure;
* FPA estimation failure;
* report schema failure;
* stale or mismatched artifact identity;
* metric derivation failure;
* DSE objective mismatch.

Diagnostics are structured records. Logs may contain backend-specific
details, but the report bundle must expose stable diagnostic classes
for tests and DSE.

## Privacy And Portability

Portable report bundles prefer artifact ids, fingerprints, profile ids,
and logical paths. They must not require private workstation paths,
license details, credentials, user names, or host names for
interpretation.

Private run logs may contain local execution details outside the public
report contract.

## Relationship To Runtime

Runtime execution may produce or update dynamic run evidence such as
launch status, simulator status, output buffers, and fallback records.
Runtime must not mutate source IR, Fabric IR, or mapping artifacts.
The report bundle records runtime evidence by reference.

## Relationship To DSE

DSE consumes report bundles as immutable evidence. If a report suggests
that a new software placement, hardware candidate, or mapping should be
tried, a later tool creates new artifacts and a new report bundle. The
old report remains immutable.

The feedback contract is specified in `docs/spec-dse-feedback.md`.

## Acceptance Criteria

The full-stack reporting target is complete when:

* one workload report bundle can summarize source, compiler, dataflow,
  DFG-sim, PnR, CGRA-sim, runtime, RTL, EDA, and FPA evidence when
  those artifacts exist;
* workload bundles that cite FPA evidence reference the normalized FPA
  JSON report, while FPA CSV summaries remain projections of that
  evidence;
* every metric records fidelity and evidence source;
* simulator cycle summary exports preserve the required
  `kernel`, `dfg_sim_cycles`, and `cgra_sim_cycles` columns;
* every table export identifies the source bundle and export profile;
* derived cycle/frequency/power/area metrics preserve input metric
  identities;
* missing optional stages are represented as unsupported-scope or
  skipped-with-reason records, not as successful evidence;
* report bundles avoid private machine details;
* DSE can consume bundles without reading backend logs or mutating
  source artifacts.
