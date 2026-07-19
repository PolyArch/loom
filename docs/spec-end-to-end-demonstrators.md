# End-to-End Demonstrators

This document specifies the target demonstrator matrix for proving that
Loom's public specs and implementation form a complete source-to-report
stack. Demonstrators are not separate product features. They are
curated verification flows that exercise the artifact contracts across
the stack.

## Purpose

End-to-end demonstrators answer this question:

```text
Which concrete workloads and hardware candidates prove that Loom can
move from source programs and architecture descriptions to mapped
simulation, RTL, FPA, and combined reports?
```

The demonstrator matrix is required because isolated unit tests cannot
prove that artifact identities, reports, mappings, simulators, and EDA
adapters agree.

## Demonstrator Rules

Every demonstrator must define:

* demonstrator id;
* purpose;
* input artifacts;
* generated intermediate artifacts;
* required reports;
* required validation commands or command families;
* expected pass diagnostics;
* expected negative diagnostics;
* allowed unsupported-scope records;
* owned public specs;
* minimum regression tier.

Compatibility mode and acceleration/reporting mode are separate. An
ordinary source build must not fail because optional Loom artifact
generation is unsupported. A reporting demonstrator may require later
artifacts, but it must state that requirement explicitly.

The global evidence, unsupported-scope, and timeout policies in
`docs/spec-loom-stack.md` apply to every demonstrator. Demonstrator
specs add only the matrix-specific input, output, and diagnostic
requirements.

## Required Demonstrators

### App Full-Stack Demonstrator

Purpose: prove that one repository-owned C/C++ app can traverse the
source-to-report path.

Input artifacts:

* one manifest-owned `test/app` case;
* ordinary compiler flags;
* runtime input data;
* selected Fabric ADG;
* selected TechMapping, PnR, and simulator configurations;
* selected RTL and FPA profiles when available.

Required generated artifacts:

* ordinary native or Loom compatibility executable;
* LLVM IR;
* raised MLIR;
* dataflow IR;
* DFG-sim report;
* validated TechMapping and one complete SpatialMapping;
* CGRA-sim report;
* simulation comparison report;
* simulator metric comparison table when comparable metrics exist;
* deployable runtime package with separate Thread Dispatch and Spatial Launch
  bindings when those consumer schemas are available;
* RTL manifest and SystemVerilog source set;
* EDA reports when selected;
* FPA report;
* workload report bundle.
* intermediate artifact gate rows covering this demonstrator.

Minimum positive behavior: compatibility-mode output matches the app
oracle, and reporting mode emits every required artifact for the
selected fidelity profile.

### CMSIS Drop-In Demonstrator

Purpose: prove that a CMSIS-DSP or CMSIS-NN case can use Loom as a
drop-in compiler and enter the Loom artifact pipeline.

Input artifacts:

* representative CMSIS-DSP or CMSIS-NN source set;
* ordinary CMSIS include paths and target flags;
* optional acceleration and report options.

Required generated artifacts:

* ordinary requested compiler output;
* LLVM IR;
* raised MLIR or structured unsupported-scope diagnostic;
* dataflow IR or structured unsupported-scope diagnostic;
* DFG-sim report for at least one supported CMSIS case;
* mapped simulation and FPA reports for at least one supported mapped
  CMSIS case when a compatible ADG is selected.

Minimum positive behavior: replacing the ordinary compiler driver with
`loom-cc` or `loom-c++` preserves build behavior in compatibility mode
for the selected case.

### Heterogeneous Non-Mesh Hardware Demonstrator

Purpose: prove that ADG Builder and Fabric ADG support arbitrary
topology and heterogeneous AccCores without relying on x/y coordinates.

Input artifacts:

* ADG Builder C++ description with one HostCore and at least two
  heterogeneous AccCores;
* explicit non-mesh Transport Architecture connectivity;
* memory hierarchy and external memory;
* coherence and consistency declarations.

Required generated artifacts:

* Fabric ADG MLIR;
* Fabric verifier report;
* optional visualization metadata;
* RTL manifest and SystemVerilog source set;
* EDA/FPA report for at least one selected profile;
* hardware candidate report bundle.

Minimum positive behavior: every hardware connection is represented by typed
resources, endpoints, and directed connectivity, and the emitted hardware
passes Fabric verification. Exact system record syntax remains open until the
typed Fabric schema is closed.

### Regular-Topology Hardware Demonstrator

Purpose: prove that regular topologies are supported as explicit
graphs, with coordinates only as optional visualization metadata.

Input artifacts:

* ADG Builder C++ description for a mesh-like, array-like, or
  systolic-like accelerator;
* explicit typed connectivity for every adjacency;
* optional `grid2d` or `grid3d` visualization metadata.

Required generated artifacts:

* Fabric ADG MLIR;
* visualization metadata;
* Fabric verifier report;
* optional RTL/FPA report.

Minimum positive behavior: deleting visualization metadata does not
change Fabric legality, PnR legality, simulation behavior, RTL
lowering, or FPA estimation.

### Mapped Workload Demonstrator

Purpose: prove TechMapping production, the boundary between Spatial PnR and
CGRA-sim, and the combined cycle/frequency/power/area report.

Input artifacts:

* supported dataflow workload;
* selected Fabric ADG;
* TechMapping producer or search configuration;
* PnR configuration;
* runtime input data;
* selected simulator and FPA profiles.

Required generated artifacts:

* validated TechMapping;
* one complete SpatialMapping emitted by Spatial PnR;
* TechMapping and SpatialMapping verifier reports;
* CGRA-sim report;
* DFG-sim report for comparison;
* simulation comparison report;
* simulator metric comparison table when comparable metrics exist;
* FPA report;
* workload report bundle with derived cycle/frequency/power/area
  metrics.
* intermediate artifact gate rows covering mapping, simulation, FPA,
  and reporting evidence.

Minimum positive behavior: TechMapping search or another validated producer
emits the exact TechMapping predecessor, Spatial PnR consumes that predecessor
and emits one complete SpatialMapping, CGRA-sim consumes the SpatialMapping
without choosing a new mapping, functional results match DFG-sim for a legal
mapping, and the full-stack report cites the cycle and FPA sources separately.

## Negative Demonstrators

The matrix must include negative cases that prove failures are not
masked:

* app manifest references a missing source file;
* compatibility-mode app run produces a wrong oracle result;
* dataflow lowering requests unsupported scope;
* PnR has no legal route over explicit topology;
* CGRA-sim receives a stale SpatialMapping;
* RTL lowering sees an unsupported Fabric primitive;
* EDA profile is missing or fails activation;
* FPA receives incomplete activity evidence;
* full-stack report attempts to derive energy without power or runtime.

## Artifact Storage

Generated demonstrator artifacts must live in deterministic output
directories. Source cases, builder examples, manifests, and small
golden files may be tracked when justified by the test contract.
Generated build outputs, simulator reports, RTL build directories, and
backend logs are not tracked unless a test specifically needs a small
stable golden artifact.

## Acceptance Criteria

The demonstrator matrix target is complete when:

* all required demonstrators have manifest records or test descriptors;
* every demonstrator has positive and negative validation commands;
* the app demonstrator reaches a workload report bundle;
* at least one demonstrator emits a simulator metric comparison table with
  producer-owned metric definitions and units;
* the CMSIS demonstrator proves drop-in compatibility and at least one
  artifact pipeline path;
* the heterogeneous non-mesh demonstrator proves arbitrary topology;
* the regular-topology demonstrator proves coordinates are metadata;
* the mapped workload demonstrator proves PnR, mapping verification,
  CGRA-sim, FPA, and combined reporting boundaries;
* required intermediate artifact gate rows exist and pass content audit
  for the demonstrators they summarize;
* timeout, blocked, skipped, and unsupported outcomes are recorded with
  structured diagnostics;
* unsupported-scope records are explicit and never counted as ordinary
  pass evidence.
