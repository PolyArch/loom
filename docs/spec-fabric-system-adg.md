# Fabric System ADG

## Purpose

`fabric.system` is the finalized system-level hardware description for a
heterogeneous Loom target. It owns typed system resources, explicit directed
connectivity, architecture-level service capability, implementation
refinement, and the attachments that connect SpatialCore templates to their
system occurrences.

It does not own software execution semantics, selected workload Mapping,
runtime remapping, simulator observations, or DSE decisions.

## Target Ownership

The typed target distinguishes these concepts:

* HostCore instances;
* AccCore instances;
* node-local immutable `#fabric.instruction_core<...>` descriptions;
* SpatialCore occurrences referencing reusable `fabric.module` templates;
* typed module-to-occurrence endpoint attachments;
* memory and service capabilities;
* Transport Architecture resources, endpoints, connections, and service
  contracts;
* Interconnect Implementation and refinement;
* external boundaries;
* address spaces, coherence, consistency, clock, reset, power, and protection
  domains; and
* optional visualization metadata.

These concepts require typed Fabric schemas. A generic node kind plus an open
dictionary is not their target representation. Exact operation names,
attribute fields, cardinalities, and assembly syntax that have not been
confirmed remain open.

## AccCore

The architectural composition is:

```text
AccCore = InstructionCore + SpatialCore
```

An AccCore is one physical system occurrence. Its InstructionCore is a typed,
immutable hardware description of the stored-program execution resource. Its
SpatialCore occurrence references one exact `fabric.module` template.

The InstructionCore description has two orthogonal typed parts: an
Architectural Contract for binary compatibility and a Microarchitectural
Realization for execution structure, timing, and capacity. It is attached to
the physical HostCore or AccCore occurrence rather than named through a string
profile or a separate symbol-bearing template.

InstructionCore replaces the retired scalar-only name. It may describe a
simple in-order core, vector-capable core, SIMD/VLIW machine, superscalar core,
or other instruction-stream architecture. Compiler Target Binding and system
simulation binding are separate derived contracts over the same typed Fabric
description.

Multiple AccCore occurrences may reference the same SpatialCore template while
remaining distinct physical resources. A template reference does not identify
the occurrence or its system attachments.

Exact child fields, AccCore operation syntax, and binding schemas remain open
unless specified by their dedicated contracts.

## SpatialCore Attachment

`fabric.module` describes a reusable SpatialCore hardware template. A system
occurrence requires typed one-to-one correspondence between module boundary
endpoints and the AccCore-local instantiated SpatialCore endpoints exposed to
system services.

An attachment stores structural references to its two exact endpoints. It is
not a route, does not copy endpoint type or capability, and does not replace
explicit Fabric connectivity. The complete attachment schema and verifier are
still open.

## Transport Architecture

Transport Architecture is a technology-neutral, explicit directed routable
graph. It contains the architecture-level resources needed to reason about
routing, multicast, contention, capacity, deadlock, buffering, latency,
bandwidth, coherence visibility, and resource-time use.

Its core concepts are typed endpoints, transport resources, directed
connections, transfer relations, and service contracts. One-to-one
forwarding, replication, arbitration, and temporal sharing are capabilities of
that graph rather than unrelated generic node kinds.

An endpoint-pair bandwidth or latency matrix is insufficient because it hides
shared resources and contention. Coordinates and visualization layout are
also insufficient and never define connectivity.

Ordinary point-to-point connections use typed source and destination endpoint
keys. Any link, lane, FIFO, or service with independent state, capacity,
configuration, or parallel identity must be an explicit typed Fabric resource.

## Interconnect Implementation

Interconnect Implementation owns concrete protocol and microarchitecture
choices such as AXI, TileLink, CXL, custom protocols, port bundles,
subchannels, packets or flits, virtual channels, router pipelines, adapters,
RTL/IP blocks, and configuration encoding.

Protocol names are implementation identity, not architecture-level service
capability. A refinement must prove that the selected implementation satisfies
the Transport Architecture contract. It must not hide an implementation
bottleneck or semantic restriction that the architecture layer does not
express.

The exact refinement and proof schemas remain open.

## Memory And Services

Fabric owns typed physical memory resources, address spaces, access
capabilities, service domains, coherence behavior, consistency guarantees,
latency, bandwidth, and capacity. These facts are independent of any selected
workload.

SpatialMapping selects concrete SpatialCore memory occurrences and local
service use. SystemMapping selects end-to-end `ServiceRealization` and
event-relative `ResourceUse` over system resources. Runtime supplies
invocation-specific memory capabilities and allocations. None of those layers
may replace the Fabric capability contract with protocol strings or open
parameters.

Exact system memory/service operation fields and ownership cardinalities that
remain under discussion are not defined here.

## Explicit Connectivity

All verifier-visible topology is explicit and directed. Helpers may construct
meshes, trees, rings, crossbars, or irregular networks, but the finalized
Fabric Hardware Description contains the actual endpoints, resources, and
connections.

Direction, payload, flow control, ordering, service role, and domain crossing
are typed properties owned by the relevant endpoint or resource schema.
String paths, symbol conventions, source order, and coordinates are not
connectivity or legality authorities.

External ports expose complete typed system boundaries. Missing adapters,
attachments, routes, services, or domain crossings are errors or unsupported
scope; they are not supplied by an implicit default.

## Domains And Protection

Clock, reset, power, address, coherence, consistency, and protection domains
are explicit typed system facts. Domain membership and crossings must be
verifier-visible. Visualization grouping and hierarchy do not imply domain
membership.

Runtime `ProtectionDomain` and memory authorization are invocation-level
contracts over Fabric-owned capabilities. Fabric must not encode a tenant or
runtime invocation as permanent hardware topology.

## Visualization

Visualization metadata may describe coordinates, hierarchy, grouping, or
preferred layout. It is optional and must be ignorable without changing
hardware identity, connectivity, Mapping legality, simulation, RTL lowering,
or Evaluation.

## Mapping Boundary

Fabric records hardware facts only. TechMapping selects semantic realizations;
SpatialMapping selects concrete SpatialCore realization; SystemMapping owns
`ExecutionBinding`, `ServiceRealization`, and `ResourceUse` across system
resources.

SystemMapping continuity composes the source SpatialMapping, source typed
attachment, system service realization, destination typed attachment, and
destination SpatialMapping. No layer copies another layer's endpoint,
connectivity, or realization authority.

Exact SystemMapping cardinality and persistent record fields remain open.

## ADG Builder Output

The C++ ADG Builder is an ergonomic producer of finalized Fabric Hardware
Descriptions. Builder-only state is not downstream authority. Built-in
templates and external descriptions must elaborate to the same typed Fabric
model before Mapping, simulation, or backend consumption.

Convenience APIs may generate regular or heterogeneous structures, but they
must emit explicit typed resources and connectivity. They must not preserve a
generic node dictionary as a parallel target schema.

## Current Generic Implementation

This section is non-normative and records the runnable-path blocker.

The current repository still uses `Fabric_NodeOp` and related string-kind and
dictionary-parameter logic for part of `fabric.system`. A complete typed
replacement for AccCore, memory/service, transport, implementation, and
attachment concepts does not yet exist on this branch.

The generic implementation therefore remains to avoid breaking the only
runnable path. It must not be expanded or treated as target design. Its code
may be deleted only in the same self-contained change that provides the
complete typed replacement and migrates all producers, consumers, verifiers,
and anchor tests. No compatibility wrapper is permitted.

## Open Boundaries

The following remain open:

* exact typed `fabric.system` operation and attribute schemas;
* complete InstructionCore architectural and microarchitectural fields;
* typed AccCore and SpatialCore attachment records;
* Transport Architecture and Interconnect Implementation record syntax;
* memory/service capability fields and refinement proofs;
* complete domain and external-boundary schemas; and
* backend and simulator binding records.

These gaps must not be filled with free-form kinds, protocol-as-capability,
open parameter dictionaries, or placeholder typed records.

## Validation

Target anchor tests will cover typed resource ownership, exact module and
occurrence references, explicit endpoint connectivity, attachment continuity,
memory/service capability, domain legality, and refinement closure after the
corresponding schemas exist.

Current generic-node textual or fixture matrices are not target acceptance
tests. Until the typed replacement is complete, existing runnable-path tests
may protect current behavior but must not establish string kinds or open
parameters as permanent architecture authority.
