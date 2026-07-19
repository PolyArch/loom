# ADG Builder

This document specifies the target C++ ADG Builder API for constructing
Fabric Architecture Description Graphs at two hardware levels:

* SpatialCore or CGRA templates emitted as `fabric.module`;
* system-level heterogeneous SoCs emitted as `fabric.system`.

The ADG Builder is a human-facing construction frontend for hardware and
system architects. Its only persistent output is Fabric dialect IR that
satisfies the relevant Fabric specs.

## Purpose

The ADG Builder must make it ergonomic to describe both individual
SpatialCore templates and heterogeneous multi-core spatial-accelerator
SoCs:

```text
SpatialCore = fabric.module over fabric.{pe,switch,mem} [spatial|temporal]
HostCore + AccCore x M + cache hierarchy + interconnect + external memory
```

The API must support three equally important use modes:

* SpatialCore construction, where users describe CGRA modules with PE,
  switch, memory, FIFO, boundary, instantiate, and FU-template
  structure;
* high-level architectural construction, where users describe common
  structures such as heterogeneous accelerator clusters, cache
  hierarchies, crossbar-like fabrics, mesh-like router graphs, NoCs,
  memory maps, and external system boundaries;
* exact typed construction, where users describe every system resource,
  endpoint, attachment, service, directed connection, refinement, and domain
  supported by the closed Fabric system schema.

The exact mode exists for cases where the user needs complete control of the
emitted typed Fabric description. Its C++ signatures remain open with the
corresponding Fabric schemas.

## Core Rule

The Builder is a pure construction frontend. It must not define
hardware semantics outside Fabric ADG. Every high-level helper must
deterministically lower to explicit `fabric.module` or `fabric.system`
IR as appropriate.

SpatialCore helpers must lower to:

* a `fabric.module` body represented by Graph-region SSA values;
* `fabric.pe`, `fabric.switch`, and `fabric.mem` tiles with explicit
  `[spatial|temporal]` schedules;
* `fabric.fifo`, `fabric.boundary`, and `fabric.instantiate` support
  constructs when required;
* `fabric.fu` templates or instances only as PE-contained functional
  units.

System helpers must lower to:

* typed HostCore and AccCore occurrence facts;
* typed InstructionCore descriptions and SpatialCore occurrence facts;
* typed SpatialCore endpoint attachments;
* typed memory and service capabilities;
* explicit Transport Architecture endpoints, resources, and directed
  connections;
* Interconnect Implementation refinements and external boundaries; and
* explicit clock, reset, power, address-space, memory-model, and
  coherence-domain declarations.

The current builder may emit generic `fabric.node`, port, channel, and link
records while that remains the only runnable implementation. Those records are
not the target API or schema and must not be expanded as a substitute for the
typed replacement.

Builder-only concepts may exist while the C++ program is running, but
they must not survive as required semantics in the emitted MLIR. If a
downstream verifier, PnR tool, simulator, RTL generator, or FPA tool
needs a fact, that fact must be present in Fabric ADG or in the explicit
artifact that owns that fact, such as a mapping artifact or manifest,
normalized FPA JSON report, report bundle, artifact manifest, runtime
descriptor, or another named contract.

## Audience

The API is for hardware and system architecture designers. It should
read like an architecture description, not like raw MLIR construction
unless the user intentionally enters exact mode. The API must make the
common path compact while preserving an escape hatch for exact control.

The Builder must not require users to think in mesh coordinates, PE indices,
Manhattan distance, or fixed topology templates. Coordinates are optional
metadata. Inside a `fabric.module`, SSA value flow is the SpatialCore
connectivity source of truth. Inside a `fabric.system`, typed Transport
Architecture resources, endpoints, and directed connections own system
topology. Their exact operation and attribute syntax remains open.

## API Layers

The ADG Builder has three API layers. All layers write into explicit
Fabric IR objects. A builder instance may construct a `fabric.module`, a
`fabric.system`, or an MLIR module containing both.

### SpatialCore Layer

The SpatialCore layer owns a `fabric.module` being constructed. It
provides entry points for module identity, module ports, spatial and
temporal tiles, boundary conversions, FIFO resources, named templates,
and MLIR emission.

Required interface shape:

```cpp
class ModuleBuilder;
class TileRef;
class PeRef;
class SwitchRef;
class MemRef;
class FuRef;
class BoundaryRef;
class FifoRef;
class ModuleValueRef;
```

The core tile matrix is `fabric.{pe,switch,mem}` crossed with
`[spatial|temporal]`. The Builder must expose that matrix directly. It
must not model `fabric.fu` as a module-level tile parallel to PE,
switch, or memory; FUs belong inside PEs. It must not introduce
module-level `fabric.link` semantics; module connectivity is expressed
by the SSA values in the emitted graph region.

### System Layer

The system layer owns one `fabric.system` under construction. Its semantic
surface covers typed HostCore and AccCore occurrences, node-local
InstructionCore descriptions, SpatialCore occurrences and attachments,
memory and service capabilities, Transport Architecture, Interconnect
Implementation, external boundaries, domains, validation, and emission.

The exact `SystemBuilder` API, handle types, ownership rules, and function
signatures await the closed typed Fabric system schema. This document does not
define generic node, port, channel, or link references as a target API.

### Exact Fabric Layer

The target exact layer mirrors the typed `fabric.system` construction surface
one-for-one. Every call corresponds to one confirmed Fabric concept and emits
facts owned by that concept. A free-form `addNode(kind, params)` entry point is
not the target exact layer.

The complete typed C++ signatures remain open with the corresponding Fabric
schemas. The builder must not invent generic wrappers or placeholder specs in
advance. Once closed, the exact layer must let a user construct any
verifier-legal typed system and diagnose invalid structure before or during
emission.

### Convenience Layer

The convenience layer provides compact architecture-oriented helpers.
It must lower into exact-layer calls before emission.

Required helper families:

* SpatialCore PE, switch, and memory-tile construction;
* spatial/temporal boundary and FIFO construction;
* PE-local FU-template construction;
* host and accelerator construction;
* typed memory and service construction;
* address-space and memory-map construction;
* cache-coherence domain construction;
* Transport Architecture and implementation-refinement construction;
* mesh-like graph construction as optional convenience;
* heterogeneous accelerator cluster construction;
* external-boundary construction;
* domain assignment helpers for clock, reset, and power.

Convenience helpers may attach optional metadata such as coordinates or
labels. They must not make that metadata semantically required unless
the same fact is represented by explicit Fabric ADG fields.

Convenience helpers for regular structures should attach optional
visualization metadata when it improves human inspection. For example, a
mesh-like helper may emit a `grid2d` visualization layout and per-resource
coordinates, and a stacked-grid helper may emit a `grid3d` visualization
layout. These hints are for GUI tools only. Typed system connectivity remains
the topology source of truth.

## Topology Breadth

The Builder must support both regular and irregular CGRA construction as
first-class targets.

Regular helpers cover traditional mesh-like, array-like, chain,
systolic-row, and clustered-array structures. They may produce layout
metadata that lets visualization tools draw the result as the same
regular topology the architect requested.

Irregular construction covers arbitrary explicit topology: heterogeneous
islands, trees, sparse long links, cross-coupled switches, mixed
temporal/spatial regions, and exact hand-built connectivity. These
topologies must be expressible without mesh coordinates or template
adjacency assumptions.

In both cases, the emitted ADG remains an explicit Fabric graph. For
`fabric.module`, Graph-region SSA values define connectivity. For
`fabric.system`, typed Transport Architecture resources, endpoints, and
directed connections define connectivity. Coordinates, layout names, ranks,
labels, rows, columns, and other display hints are visualization metadata only,
as specified by
`docs/spec-mapping-visualization.md`. They must not affect Fabric
verification, PnR placement legality, routing legality, routing costs,
simulator behavior, RTL lowering, FPA, or DSE candidate scoring.

## Ergonomics Requirements

The common construction path must be concise:

* Create a system and choose the required memory model.
* Add a host core.
* Add heterogeneous AccCores with typed InstructionCore descriptions and
  SpatialCore `fabric.module` references.
* Add physical memory targets and caches.
* Connect components through high-level helper functions.
* Emit verifier-legal Fabric MLIR.

The exact construction path must be explicit:

* Users can declare every typed resource, endpoint, attachment, service, and
  directed connection supported by the closed schemas.
* Users can manually build arbitrary topology without topology-specific
  helpers.
* Users can select an Interconnect Implementation refinement without using a
  protocol name as architecture capability.
* Users can express every legal typed clock-domain crossing and its selected
  implementation refinement once those schemas are closed.
* Users can express every legal coherence-domain and memory-model
  declaration.

The API should prefer typed enums and small spec structs over stringly typed
builders for baseline semantics. Strings are appropriate for non-authoritative
user labels.

## Deterministic Lowering

Every convenience helper must produce the same canonical semantic graph for
the same semantic inputs and builder version. Fabric finalization owns
canonical labeling, artifact-local identities, and serialization order.

An emitter may choose a deterministic dependency-respecting textual order and
derived human-readable labels. Symbol spelling, builder construction order,
connection enumeration order, and diagnostic order are not hardware identity,
connectivity, or Mapping tie breakers. Generated-name collisions must be
diagnosed or resolved before finalization without changing semantic endpoint
identity.

## Inline-Fabric Control

The Builder must support local exact construction inside otherwise
high-level architecture construction. A user must be able to do all of
the following in one system:

* create most of the SoC with convenience helpers;
* open an exact construction region for one subsystem;
* declare exact typed resources, endpoints, attachments, services,
  connections, refinements, and domains in that subsystem;
* connect helper-created objects to exact typed objects;
* emit one uniform Fabric ADG with no marker that downstream tools must
  treat specially.

Inline-fabric regions are API structure only. They must not emit a
distinct high-level operation or require downstream tools to understand
how the region was written.

## Validation

The Builder must provide validation before MLIR emission and after MLIR
emission:

* pre-emission validation checks builder object consistency, unresolved typed
  references, label collisions, and helper expansion completeness;
* emission produces Fabric ADG;
* post-emission validation invokes the relevant Fabric verifier
  contracts, including `docs/spec-fabric-module.md` for `fabric.module`
  roots and `docs/spec-fabric-system-adg.md` for `fabric.system` roots.

Builder diagnostics must identify the user-level helper call or exact
object that introduced the problem. Diagnostics should also identify the
target Fabric object when one exists.

## Examples Required By The Spec

The implementation must provide example programs for these cases:

* minimal SpatialCore with spatial PE, switch, and memory tile;
* temporal SpatialCore with temporal PE, temporal switch, temporal
  memory tile, and required spatial/temporal boundaries;
* SpatialCore using named PE templates, PE-local FU templates, and
  `fabric.instantiate`;
* mixed spatial and temporal resources connected through SSA values and
  boundary ops inside one `fabric.module`;
* regular SpatialCore CGRA templates such as chain, mesh, array, or
  systolic-like structures, where any coordinates are visualization
  metadata and every connection is explicit in the emitted
  `fabric.module` graph;
* irregular SpatialCore CGRA templates such as heterogeneous islands,
  reduction trees, sparse long-link graphs, or mixed spatial/temporal
  regions, without relying on a fixed grid topology;
* minimal host plus one accelerator plus one memory target;
* heterogeneous host plus two different accelerator cores;
* cache-coherent host/cache/accelerator/memory system;
* arbitrary non-mesh topology using exact typed system connectivity;
* mesh-like router graph built through convenience helpers;
* crossbar-like helper expansion into explicit Transport Architecture
  resources and an Interconnect Implementation refinement;
* NoC helper expansion into explicit transport resources, endpoints,
  connections, and implementation refinements;
* mixed high-level and inline-fabric construction in one system.

Each example must emit MLIR and run the Fabric ADG verifier. The
examples are part of the API contract, not incidental demos.

## Target Universe

The ADG Builder target universe covers both construction levels.

SpatialCore Builder coverage includes:

* every verifier-legal `fabric.module` port shape;
* `fabric.pe [spatial]` and `fabric.pe [temporal]`;
* `fabric.switch [spatial]` and `fabric.switch [temporal]`;
* `fabric.mem [spatial]` and `fabric.mem [temporal]`;
* PE-contained `fabric.fu` templates and instances;
* `fabric.fifo`, `fabric.boundary`, and `fabric.instantiate`;
* named and anonymous forms where the Fabric specs allow both;
* deterministic emission, validation, and diagnostics.

System Builder coverage follows the typed ownership in
`docs/spec-fabric-system-adg.md`: HostCore and AccCore occurrences,
InstructionCore descriptions, SpatialCore occurrences and attachments,
memory and service capabilities, Transport Architecture, Interconnect
Implementation, external boundaries, address spaces, coherence, consistency,
and hardware domains. Exact API coverage is defined only after those Fabric
schemas close.

## Required Evidence

Builder evidence consists of emitted MLIR, verifier results, and
example-run reports. Each example must identify:

* the builder entry point used;
* the emitted Fabric root symbol;
* whether the root is a `fabric.module`, a `fabric.system`, or both;
* verifier status and diagnostics;
* stable output ArtifactIdentity when used by later artifacts.

## Objective Verification

The Builder target is objectively verifiable when:

* each required example emits deterministic MLIR;
* each emitted MLIR artifact verifies;
* every closed target Fabric construct has at least one builder-positive test
  or example;
* invalid builder inputs produce structured diagnostics before or during
  emission;
* downstream tools can consume emitted Fabric IR without reading
  builder-only state.

## Unsupported Scope Policy

A Builder helper may be absent only when the corresponding closed typed Fabric
construction path can still emit the target IR and the missing helper is
recorded as unsupported convenience scope. Before the typed system schema is
closed, absent exact SystemBuilder API is an explicit implementation gap rather
than permission to standardize generic references.

## Relationships To Other Contracts

SpatialCore Builder output follows `docs/spec-fabric-module.md` and the
SpatialCore fabric specs. System Builder output follows
`docs/spec-fabric-system-adg.md`. PnR, simulators, RTL lowering, FPA,
reports, and DSE consume the emitted Fabric IR through those specs, not
through Builder internals.

## Current Implementation Notes

This section is non-normative. It records current repository facts for
orientation only and is not part of target acceptance.

The current implementation contains a `ModuleBuilder` seed, a
`SystemBuilder` seed, a `shared_reduction_adg` SpatialCore example that
emits a `fabric.module`, and a heterogeneous SoC example that emits a
baseline `fabric.system`. The system builder seed covers host,
SpatialCore-backed accelerator, fixed-function accelerator, memory
nodes, string-described port channels, explicit links, and hardware
summary evidence. This note records the current repository surface only;
the Builder target contract is defined by the sections above and by the
owning Fabric specs.

## Non-Goals

The Builder is not a placement tool. It does not map software dataflow
graphs onto hardware. Placement and routing belong to the PnR tool and
its mapping artifact.

The Builder is not a simulator. It may attach timing, bandwidth,
latency, or capacity metadata, but it does not execute workloads.

The Builder is not an RTL generator. It emits Fabric ADG, which can be
consumed by later RTL and estimation tools specified in
`docs/spec-rtl-lowering.md` and `docs/spec-fpa-estimation.md`.

The Builder is not a second hardware IR. It must not preserve helper
semantics that downstream tools must understand separately from Fabric
ADG.

## Acceptance Criteria

The ADG Builder target is complete when:

* the SpatialCore layer can emit every verifier-legal target
  `fabric.module` construct;
* the exact layer can emit every verifier-legal typed `fabric.system`
  construct whose schema is closed;
* high-level helpers lower only to explicit Fabric ADG constructs;
* all required examples emit deterministic MLIR;
* emitted MLIR verifies under the Fabric ADG verifier;
* canonical semantic output is deterministic without making generated names
  or operation order identity;
* regular-topology helpers can emit optional visualization metadata;
* regular SpatialCore CGRA examples render as regular layouts when
  visualization metadata is present, but removing that metadata does
  not change Fabric legality, PnR legality, simulation behavior, RTL
  lowering, or FPA estimation;
* irregular SpatialCore CGRA examples prove the Builder and Fabric
  dialect are not limited to mesh, array, or coordinate-adjacent
  topology;
* no convenience helper requires mesh, x/y coordinates, Manhattan
  routing, or homogeneous accelerator cores;
* inline-fabric construction can be mixed with convenience construction
  without producing special downstream semantics.
