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
  memory maps, and system boundary ports;
* exact low-level construction, where users write Fabric ADG nodes,
  ports, channels, domains, coherence domains, external ports, and
  links one-for-one through C++ calls.

The second mode is the equivalent of inline assembly for architecture
construction. It exists for cases where the user needs exact control of
the emitted Fabric MLIR.

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

* physical `fabric.node` operations;
* complete `#fabric.port` attributes;
* complete `#fabric.channel` attributes;
* complete `fabric.external_port` operations;
* one-to-one `fabric.link` operations;
* explicit clock, reset, power, address-space, memory-model, and
  coherence-domain declarations.

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

The Builder must not require users to think in mesh coordinates, PE
indices, Manhattan distance, or fixed topology templates. Coordinates
are optional metadata. Inside a `fabric.module`, SSA value flow is the
SpatialCore connectivity source of truth. Inside a `fabric.system`,
explicit `fabric.link` operations are the system topology source of
truth.

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

The system layer owns a `fabric.system` being constructed. It provides
entry points for system identity, memory model, clock/reset/power
domains, address spaces, nodes, external ports, links, coherence
domains, validation, and MLIR emission.

Required interface shape:

```cpp
class SystemBuilder;
class NodeRef;
class PortRef;
class ChannelRef;
class ExternalPortRef;
class LinkRef;
class DomainRef;
class CoherenceDomainRef;
```

These are stable handles to graph objects, not owning C++ hardware
semantics. Handles remain valid until the owning `SystemBuilder` is
destroyed. Handles must expose enough identity to connect endpoints and
attach metadata, but they must not bypass verifier rules.

### Exact Fabric Layer

The exact layer mirrors the system-level `fabric.system` construction
surface one-for-one. Every call in this layer corresponds to one target
system Fabric concept. This is the required escape hatch for precise
system-level control.

Required interface shape:

```cpp
NodeRef addNode(Symbol name, NodeKind kind, NodeSpec spec);
ExternalPortRef addExternalPort(Symbol name, PortSpec port);
LinkRef addLink(Endpoint src, Endpoint dst, LinkSpec spec = {});
DomainRef addClockDomain(Symbol name, ClockDomainSpec spec = {});
DomainRef addResetDomain(Symbol name, ResetDomainSpec spec = {});
DomainRef addPowerDomain(Symbol name, PowerDomainSpec spec = {});
CoherenceDomainRef addCoherenceDomain(Symbol name,
                                       CoherenceDomainSpec spec);
```

`NodeSpec`, `PortSpec`, `ChannelSpec`, `LinkSpec`, and domain specs must
map directly to the Fabric ADG fields in `docs/spec-fabric-system-adg.md`.
The exact layer must let a user produce any verifier-legal
`fabric.system` construct. It must also reject or diagnose invalid
system ADG before or during emission.

### Convenience Layer

The convenience layer provides compact architecture-oriented helpers.
It must lower into exact-layer calls before emission.

Required helper families:

* SpatialCore PE, switch, and memory-tile construction;
* spatial/temporal boundary and FIFO construction;
* PE-local FU-template construction;
* host and accelerator construction;
* fixed-function accelerator construction;
* cache and memory construction;
* address-space and memory-map construction;
* cache-coherence domain construction;
* crossbar-like interconnect expansion;
* router and NoC construction;
* mesh-like graph construction as optional convenience;
* heterogeneous accelerator cluster construction;
* system-boundary external port construction;
* domain assignment helpers for clock, reset, and power.

Convenience helpers may attach optional metadata such as coordinates or
labels. They must not make that metadata semantically required unless
the same fact is represented by explicit Fabric ADG fields.

Convenience helpers for regular structures should attach optional
visualization metadata when it improves human inspection. For example,
a mesh-like helper may emit a `grid2d` visualization layout and per-node
coordinates, and a stacked-grid helper may emit a `grid3d` visualization
layout. These hints are for GUI tools only. Explicit links remain the
topology source of truth.

## Ergonomics Requirements

The common construction path must be concise:

* Create a system and choose the required memory model.
* Add a host core.
* Add heterogeneous accelerator cores that reference SpatialCore
  `fabric.module` templates and scalar-core profiles.
* Add physical memory targets and caches.
* Connect components through high-level helper functions.
* Emit verifier-legal Fabric MLIR.

The exact construction path must be explicit:

* Users can declare every node kind, port, channel, and link.
* Users can manually build arbitrary topology without topology-specific
  helpers.
* Users can use built-in protocol schemas or fully explicit `custom`
  protocol schemas.
* Users can express every legal clock-domain crossing form, including
  link-level crossing metadata and explicit `clock_converter` nodes.
* Users can express every legal coherence-domain and memory-model
  declaration.

The API should prefer typed enums and small spec structs over stringly
typed builders for baseline semantics. Strings are appropriate for
symbols, user labels, and stable custom names.

## Deterministic Lowering

Every convenience helper must lower deterministically. Given the same
inputs and builder version, emission must produce stable symbol names,
stable operation ordering, stable link ordering, and stable diagnostics.

Required deterministic ordering:

* domain declarations before nodes that reference them;
* `fabric.module` references before `acc_core` nodes that reference
  them, when the builder emits both in one MLIR module;
* node operations in construction order unless a helper documents a
  stable generated order;
* external ports in construction order;
* links in construction order after helper expansion;
* coherence domains after their member ports have been declared.

Generated names must be deterministic and user-overridable. A helper
must diagnose generated-name collisions instead of silently renaming
objects in a way that changes endpoint identity.

## Inline-Fabric Control

The Builder must support local exact construction inside otherwise
high-level architecture construction. A user must be able to do all of
the following in one system:

* create most of the SoC with convenience helpers;
* open an exact construction region for one subsystem;
* declare exact nodes, ports, channels, and links in that subsystem;
* reconnect helper-created handles to exact handles;
* emit one uniform Fabric ADG with no marker that downstream tools must
  treat specially.

Inline-fabric regions are API structure only. They must not emit a
distinct high-level operation or require downstream tools to understand
how the region was written.

## Validation

The Builder must provide validation before MLIR emission and after MLIR
emission:

* pre-emission validation checks builder handle consistency, unresolved
  symbols, duplicate names, and helper expansion completeness;
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
* heterogeneous system with one SpatialCore-backed accelerator and one
  fixed-function accelerator;
* cache-coherent host/cache/accelerator/memory system;
* arbitrary non-mesh topology using exact links;
* mesh-like router graph built through convenience helpers;
* crossbar-like helper expansion into route decoders, arbiters, and
  explicit one-to-one links;
* NoC helper expansion into routers, network endpoints, adapters, and
  explicit links;
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

System Builder coverage includes every verifier-legal `fabric.system`
construct from `docs/spec-fabric-system-adg.md`, including nodes, ports,
channels, external ports, explicit links, domains, address spaces,
memory regions, memory model, and coherence domains.

## Required Evidence

Builder evidence consists of emitted MLIR, verifier results, and
example-run reports. Each example must identify:

* the builder entry point used;
* the emitted Fabric root symbol;
* whether the root is a `fabric.module`, a `fabric.system`, or both;
* verifier status and diagnostics;
* stable output identity or fingerprint when used by later artifacts.

## Objective Verification

The Builder target is objectively verifiable when:

* each required example emits deterministic MLIR;
* each emitted MLIR artifact verifies;
* every target Fabric construct has at least one builder-positive test
  or example;
* invalid builder inputs produce structured diagnostics before or during
  emission;
* downstream tools can consume emitted Fabric IR without reading
  builder-only state.

## Unsupported Scope Policy

A Builder helper may be absent only when the corresponding exact Fabric
construction path can still emit the target IR and the missing helper is
recorded as unsupported convenience scope. Missing exact-layer support
for a verifier-legal target construct is a Builder target gap.

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
* the exact layer can emit every verifier-legal `fabric.system` construct;
* high-level helpers lower only to explicit Fabric ADG constructs;
* all required examples emit deterministic MLIR;
* emitted MLIR verifies under the Fabric ADG verifier;
* generated names and operation ordering are stable;
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
