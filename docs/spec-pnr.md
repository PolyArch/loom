# PnR

This document specifies Loom place-and-route. PnR establishes the
relation between software dataflow abstractions and a concrete hardware
Fabric ADG. Its persistent output is the independent mapping artifact
specified in `docs/spec-mapping-artifact.md`.

Detailed mapper contracts are split across:

* `docs/spec-mapping-identity.md`
* `docs/spec-mapping-placement.md`
* `docs/spec-mapping-routing.md`
* `docs/spec-mapping-schedule-buffer.md`
* `docs/spec-mapping-memory.md`
* `docs/spec-mapping-verification.md`
* `docs/spec-mapping-visualization.md`
* `docs/spec-mapping-search.md`

## Purpose

PnR maps software onto hardware. It consumes:

* software dataflow IR;
* a selected `fabric.system`;
* referenced `fabric.module` SpatialCore templates;
* user constraints and objectives;
* optional workload shape, profile data, or previous evaluation
  metrics.

PnR produces:

* one or more mapping artifacts;
* optional mapping-set manifests for DSE;
* diagnostics for rejected, partial, or degraded mappings;
* mapping-quality metrics used for search and reporting.

PnR is not a simulator. It may use analytical estimates, profile data,
DFG-sim results, previous CGRA-sim results, or FPA estimates as cost
inputs, but it does not execute workloads and it must not present its
estimates as hardware-aware simulation results.

The first hard PnR target is verifier-grade mapping artifacts. Such
artifacts may be emitted by PnR, hand-authored for tests, imported from
another mapper, or requested by DSE. PnR automation is not a prerequisite
for consumers to use a verifier-clean artifact.

The subsystem ownership boundary that separates dataflow facts, Fabric
facts, mapping facts, runtime facts, and simulation facts is specified
in `docs/spec-core-dialect-boundary.md`. PnR follows that boundary and
serializes its choices in mapping artifacts.

## Boundary With CGRA-sim

PnR and CGRA-sim have different responsibilities:

| Tool | Responsibility | Persistent output |
|------|----------------|-------------------|
| PnR | Choose and record how software maps to hardware. | Mapping artifact and optional mapping-set manifest. |
| CGRA-sim | Simulate mapped software under hardware constraints for concrete inputs. | Hardware-aware simulation report. |

CGRA-sim is hardware-aware simulation, not only simulation of the
`fabric.module` or SpatialCore portion. Despite the name, CGRA-sim
simulates the mapped workload against the selected hardware graph,
including AccCore execution, SpatialCore resources, ScalarCore residual
execution where modeled, interconnect, memory hierarchy, buffers,
routing, resource sharing, temporal schedules, and activity metrics.

PnR must not depend on CGRA-sim internal state. CGRA-sim must not choose
placements, routes, schedules, or bindings. If simulation reveals that a
mapping candidate is poor or invalid under a stronger model, the DSE
loop may invoke PnR again with updated objectives, constraints, or
feedback. The new PnR run produces a new mapping artifact.

## Relation To Compiler Placement

The compiler placement framework in
`docs/spec-compiler-part-3-placement-framework.md` decides software-side
partition boundaries:

* L1 selects accelerator regions;
* L2 selects SpatialCore graph regions inside selected accelerator
  code;
* L3 partitions graph bodies into software `dataflow.subgraph` units.

PnR happens after those software-side choices are available. PnR binds
selected logical software entities to physical hardware resources. It
does not rewrite L1, L2, or L3 IR boundaries unless a larger compiler
DSE loop explicitly asks the compiler to produce a new software
candidate.

## Core Model

PnR is a search problem with four separate concerns:

* legality rules;
* candidate construction;
* cost model;
* search policy.

Legality decides whether a candidate mapping is valid. The cost model
chooses among legal candidates. A cost model must never make an illegal
candidate legal. A legality rule must not encode a preference unless
violating that rule would break software semantics, hardware semantics,
or the mapping artifact contract.

The baseline PnR policy must be deterministic. Given the same software
IR, hardware IR, mapping options, and workload shape, it must produce
the same mapping artifact and diagnostics.

PnR treats hardware as an arbitrary directed graph. Coordinates, grid
metadata, and visualization layouts are display metadata. Placement and
route legality are derived only from explicit Fabric nodes, resources,
directed channel endpoints, links, boundaries, adapters, buffers,
memories, and protocol channels. Cost models may use explicit hardware
weights such as latency, bandwidth, capacity, or user-declared edge
weights; they must not derive hardware cost from visualization
coordinates.

## Candidate Structure

A complete PnR candidate contains these decisions when relevant:

* thread binding;
* graph binding;
* operation or subgraph binding;
* route assignment for software value, control, and memory-order edges;
* schedule binding;
* temporal tag assignment;
* buffer assignment;
* memory binding;
* resource-sharing assignment;
* optional visualization overlay metadata;
* metrics and diagnostics.

The candidate is serialized as a mapping artifact. PnR-internal state is
not a valid substitute for artifact records.

## Legality Rules

PnR legality includes at least the following rule families.

### Reference Legality

Every software and hardware reference used by a candidate must resolve.
Fingerprints must match when provided. PnR must diagnose stale or
ambiguous references.

### Thread Legality

Logical `dataflow.thread` instance domains may bind to `acc_core` nodes
and execution batches. Binding must preserve the logical instance set,
logical thread coordinates, memory-order constraints, async
launch/fence ordering, and thread hierarchy rules defined by the
dataflow specs.

Thread binding may be explicit or parametric. Parametric binding must
describe a deterministic relation from logical coordinates to hardware
resources and batches. It must not assume mesh topology unless that
topology is explicitly represented by Fabric links; visualization
coordinates alone are not hardware topology.

### Graph Legality

Each mapped `dataflow.graph` execution context must bind to an
`acc_core` whose SpatialCore template is compatible with the graph
resources required by the candidate. Non-innermost orchestration thread
bodies do not directly bind graph executions unless the software IR
contains legal graph launches at that level.

### Operation Legality

Each software operation, subgraph, or compute region must bind to a
compatible hardware resource. Supported operation sets, data widths,
types, port counts, and side effects must match the target `fabric.fu`,
`fabric.pe`, `fabric.mem`, or other referenced resource.

Exclusive resources may be shared only through explicit schedule or
temporal-tag records that make same-time conflicts impossible.

### Route Legality

Routes must use explicit hardware connectivity. System-level routes use
`fabric.link` channel connectivity. SpatialCore routes use explicit
resources inside the referenced `fabric.module`, such as switches,
boundaries, PEs, FUs, memories, or FIFOs.

Every route must be contiguous from the placed producer to the placed
consumer. Fanout, arbitration, broadcast, protocol conversion, width
conversion, and clock conversion must use explicit hardware resources.
Routes must not assume x/y coordinates, Manhattan distance, or mesh
adjacency. Mesh-like hardware is legal only because its links are
explicit Fabric connectivity.

### Schedule Legality

Schedules must respect resource capacity, temporal-tag capacity, buffer
availability, operation dependencies, memory-order edges, control
tokens, and reconfiguration limits. A scheduled resource use must not
conflict with another use of the same exclusive resource in the same
slot unless the hardware resource explicitly supports that concurrency.

### Buffer Legality

Buffer assignment must provide enough physical queue or storage
resources for all software value streams, control tokens, done tokens,
memory-order tokens, and routed edges that require buffering. Assigned
depth and backpressure policy must be visible in the mapping artifact
when CGRA-sim or RTL lowering needs those facts.

### Memory Legality

Memory binding must target memory-capable ports and legal address
spaces. It must respect terminal memory target ranges, cache coherence
domains, consistency model, partitioned-data regions, and any explicit
memory-order constraints.

### Visualization Legality

Visualization metadata is optional. When present, it must reference
existing software objects, hardware objects, mapping records, and Fabric
visualization layouts. Visualization metadata never makes an otherwise
illegal mapping legal and never makes a legal mapping illegal unless the
metadata itself has invalid references or invalid layout coordinates.

## Cost Model

The PnR cost model ranks legal candidates. It may include:

* estimated cycles;
* route length and congestion;
* resource utilization;
* buffer pressure;
* memory bandwidth pressure;
* cache and coherence pressure;
* temporal reuse quality;
* reconfiguration count;
* energy, area, or frequency estimates from
  `docs/spec-fpa-estimation.md`;
* diagnostic severity;
* DFG-sim, CGRA-sim, or FPA feedback from previous candidates.

The baseline cost model may be simple, but it must define a
deterministic total order. Multi-objective search may use weighted
scores, lexicographic scores, constraints plus objectives, or Pareto
ranking. The selected policy must record enough configuration in the
mapping artifact or mapping-set manifest to reproduce the decision.

## Search Policy

PnR search owns candidate generation and final selection. The target
design supports multiple policies:

* deterministic greedy baseline;
* beam search;
* simulated annealing;
* integer or mixed-integer programming;
* profile-guided search;
* feedback-driven DSE using prior simulation or estimation metrics.

Every policy must preserve the same legality contract and artifact
schema. Policies may prune candidates, but they must diagnose the case
where no legal candidate exists for required mapping.

## Baseline Policy

The required baseline policy is deterministic and debug-friendly. It
does not need to be performance-optimal.

Baseline ordering requirements:

* process thread instances in stable logical order;
* process graph launches in stable software order;
* process operations and subgraphs in stable topological order, with a
  stable tie breaker;
* choose hardware resources using stable symbol order after filtering
  by compatibility;
* route over the explicit hardware graph using a deterministic shortest
  legal path metric;
* assign schedules and buffers using stable earliest-legal placement;
* emit records in deterministic artifact order.

The baseline route metric must operate over the explicit graph. It may
use latency, bandwidth, or user weights when present, but it must not
derive adjacency from coordinates alone.

## Outputs And Diagnostics

For each candidate, PnR emits one mapping artifact. For a DSE run, PnR
may also emit a mapping-set manifest.

Diagnostics must identify:

* the software object involved;
* the hardware object involved when known;
* the legality rule or resource constraint that failed;
* whether the failure is fatal, partial, or a quality degradation;
* which search policy produced the diagnostic.

Diagnostics must not be hidden inside logs only. They must be available
to users, tests, DSE reports, and visualization tools.

## Interface With CGRA-sim

CGRA-sim is specified in `docs/spec-sim-cgra.md`. It consumes:

* software dataflow IR;
* Fabric ADG;
* a mapping artifact;
* concrete runtime input data;
* simulator configuration.

PnR supplies the mapping artifact. CGRA-sim may verify artifact
consistency before simulation, but it does not repair or select the
mapping. If CGRA-sim needs a schedule, buffer depth, temporal tag,
route, or memory binding, that information must be in the mapping
artifact.

CGRA-sim reports observed cycles, resource activity, queue occupancy,
memory activity, route activity, temporal reuse, stalls, and other
hardware-aware metrics. Its reports can be compared with DFG-sim reports
through `docs/spec-sim-comparison.md`. These reports may feed a
later PnR or DSE run as cost feedback, but they are not part of the
original PnR decision unless explicitly referenced in a new mapping-set
manifest.

## Acceptance Criteria

PnR is complete at the target-spec level when:

* it emits independent mapping artifacts rather than mutating dataflow
  or Fabric IR;
* it can validate and consume verifier-clean mapping artifacts regardless
  of whether they were generated by PnR, imported, or hand-authored for
  tests;
* the deterministic baseline policy maps a toy graph onto a non-mesh
  arbitrary topology;
* the deterministic baseline policy maps a regular mesh-like topology
  using explicit Fabric links rather than coordinate assumptions;
* route records are contiguous and reference existing hardware
  resources;
* resource-sharing, schedule, buffer, and memory records are emitted
  when required by the mapping;
* illegal mappings produce structured diagnostics;
* multiple search policies can share the same legality and artifact
  contracts;
* CGRA-sim can consume the emitted artifact without reading PnR
  internal state;
* DSE can compare multiple candidate artifacts through a mapping-set
  manifest.
