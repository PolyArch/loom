# DSE Feedback

This document specifies Loom's target design-space exploration feedback
contract. DSE uses immutable artifacts and reports to choose new
software placement, hardware, mapping, simulator, RTL, or estimation
candidates.

## Purpose

DSE feedback answers this question:

```text
Given a set of compiler, hardware, mapping, simulation, RTL, and FPA
evidence, which new candidate should Loom generate or select next, and
which objective explains that choice?
```

DSE feedback is not a hidden side channel. A feedback decision must be
represented by explicit records that reference the artifacts and
metrics that motivated it.

## Feedback Fidelity Taxonomy

DSE feedback records use these fidelity classes:

* `analytic_prefilter`: software static features and a default hardware
  resource model, used for quick cost or performance screening;
* `techmap_estimate`: Compute Realization grouping, selected FU encodings,
  techmap cost, and calibrated structural estimates;
* `dfg_sim_feedback`: input-driven DFG-sim results for software
  semantics and dynamic execution baseline evidence;
* `pnr_feedback`: mapping artifact evidence from placement, routing,
  resource use, schedule, buffer, and memory-binding records;
* `cgra_sim_feedback`: input-driven CGRA-sim evidence after mapping;
* `eda_fpa_feedback`: RTL, EDA, backend report, or normalized FPA JSON
  evidence.

Low-fidelity feedback may be used for prefiltering. Formal ranking and
selection must declare the fidelity used for every metric input, cite
the evidence source, and record diagnostics when a required fidelity is
missing or incompatible. An estimate must not be relabeled as simulator
or backend evidence. Candidate summaries must expose a
`hardware_evidence_kind` so analytic model-only candidates remain
distinguishable from backend-backed hardware evidence.

## Feedback Boundary

DSE may consume:

* source workload identities;
* compiler placement candidates;
* dataflow IR artifacts;
* DFG-sim reports;
* ADG Builder recipes;
* Fabric ADG artifacts;
* mapping artifacts and mapping-set manifests;
* CGRA-sim reports;
* simulation comparison reports;
* runtime reports;
* RTL manifests;
* EDA reports;
* FPA reports;
* full-stack report bundles;
* user constraints and objectives.

DSE may produce:

* new compiler placement requests;
* new TechMapping search requests;
* new ADG Builder recipe requests;
* new PnR search requests;
* new simulator or FPA evaluation requests;
* selected candidate records;
* Pareto set records;
* rejected candidate records;
* DSE report bundles.

DSE must not mutate source IR, dataflow IR, Fabric ADG, mapping
artifacts, simulator reports, RTL manifests, EDA reports, or FPA
reports. It creates new explicit artifacts when it explores new
candidates.

DSE candidates are immutable data points. A change to a candidate
creates a new candidate with a new identity; it does not update the old
candidate in place.

The global evidence policy in `docs/spec-loom-stack.md` applies to DSE
inputs. DSE-specific selection criteria must not treat unsupported,
blocked, scaffold, fixture, or missing-metric records as passing
candidate evidence.

The configuration SSOT in `docs/spec-config-ssot.md` owns DSE weights,
objective profiles, policy selection, seeds, and fidelity requirements.
DSE may consume typed views of the resolved configuration, but it must
not own an independent objective-default table or silently fall back from
an unknown objective to another objective.

## Objective Records

An objective record has these required fields:

* objective id;
* objective kind;
* metric inputs;
* feedback fidelity for each metric input;
* priority or weight;
* constraint or optimization mode;
* comparison direction;
* units;
* validity conditions.

Baseline objective kinds include:

* minimize runtime;
* maximize throughput;
* minimize area;
* minimize dynamic power;
* minimize leakage power;
* minimize energy;
* maximize performance per watt;
* maximize performance per area;
* satisfy timing target;
* satisfy memory capacity;
* satisfy resource utilization bound;
* minimize unsupported-scope diagnostics;
* custom objective with explicit model identity.

When objectives conflict, the selected policy must state how conflicts
are resolved: weighted score, lexicographic ordering, Pareto ranking,
constraint filtering, or custom policy.

Continuous weights and named preset profiles are both first-class DSE
configuration. Presets resolve to ordinary objective records and weights
before candidate generation or selection. Compiler placement choices,
including whether a loop remains a logical `dataflow.thread` frontier or
is placed inside a SpatialCore graph, are candidates selected by these
configured objectives and feedback records rather than direct force
switches.

## Candidate Records

A candidate record identifies:

* candidate id;
* candidate kind;
* parent candidate ids when derived from earlier candidates;
* referenced input artifacts;
* generated output artifacts;
* objective records used;
* metric records used;
* feedback fidelity records used;
* status;
* diagnostics.

Candidate kinds include:

* compiler L1 accelerator placement candidate;
* compiler L2 graph placement candidate;
* TechMapping candidate containing Compute Realizations;
* hardware ADG candidate;
* Physical Mapping candidate;
* simulator configuration candidate;
* RTL/FPA profile candidate;
* combined full-stack candidate.

## Feedback Targets

### Compiler Placement Feedback

DSE may request new L1 or L2 software placement candidates. The request must
cite reports or metrics that motivate the change. The compiler must produce
new dataflow artifacts rather than modifying an old artifact in place.

### TechMapping Feedback

DSE may request a new L3 Compute Realization search over exact immutable
Canonical Dataflow and Fabric artifacts. The result is a new TechMapping
artifact containing selected actor groups, FU encodings, and complete
correspondence witnesses. It must not persist a competing software partition
or mutate either input artifact.

### Hardware Candidate Feedback

DSE may request a new ADG Builder recipe or new Fabric ADG candidate.
The request must cite hardware metrics, mapping failures, simulator
stalls, route congestion, memory pressure, FPA reports, or user
constraints.

### PnR Feedback

DSE may request new PnR runs with updated objectives, constraints, or
cost-model weights over an exact TechMapping predecessor. The output is a new
Physical Mapping artifact or mapping-set manifest.

### Simulator And FPA Feedback

DSE may request additional simulator or FPA runs to resolve missing or
low-confidence metrics. The output is a new report, not a mutation of
an old report.

## Reproducibility

DSE feedback must record:

* selected policy id;
* policy configuration;
* resolved configuration identity and fingerprint;
* random seed when stochastic search is used;
* input artifact identities and fingerprints when available;
* objective records;
* candidate ordering rule;
* selected candidate or Pareto set;
* rejected candidate summaries.

Given the same inputs, policy, and seed, deterministic and seeded
policies must reproduce the same selected candidate records.

## Diagnostics

DSE diagnostics must distinguish:

* missing objective;
* unknown objective;
* unknown policy;
* conflicting configuration sources;
* mismatched configuration fingerprint;
* missing required metric;
* unsupported feedback target;
* conflicting hard constraints;
* no candidate satisfies constraints;
* stale artifact fingerprint;
* incompatible report fidelity;
* non-reproducible stochastic run without seed;
* custom model unavailable;
* candidate generation failed.

Diagnostics must identify the artifact or metric that caused the
failure when applicable.

## Relationship To PnR

PnR physically realizes one exact TechMapping candidate and emits a Physical
Mapping artifact or mapping-set manifest. DSE may run PnR repeatedly, compare
the resulting artifacts, and select one candidate. PnR does not own cross-run
candidate policy unless it is acting as the selected DSE policy for physical
mapping search and records that policy in a manifest.

## Relationship To Reporting

Full-stack report bundles are immutable DSE inputs. DSE report bundles
summarize candidate sets and selections. They must cite the workload
and hardware report bundles they compare.

## Acceptance Criteria

The DSE feedback target is complete when:

* objectives are explicit records rather than hidden command-line
  assumptions;
* objective defaults, weights, presets, policy ids, and seeds are read
  from the resolved configuration SSOT rather than from component-local
  constants;
* candidate records identify input and output artifacts;
* DSE can request new compiler placement, hardware, mapping, simulator,
  RTL, or FPA candidates without mutating old artifacts;
* candidate selection is reproducible for deterministic or seeded
  policies;
* selected candidates and Pareto sets cite the metrics that justify
  them;
* selected candidates and Pareto sets declare the feedback fidelity and
  provenance used for ranking;
* DSE selection rejects records that violate the global evidence policy;
* DSE selection rejects records whose required configuration
  fingerprints are incompatible;
* unsupported feedback targets and missing metrics produce structured
  diagnostics.
