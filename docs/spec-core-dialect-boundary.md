# Core Dialect Boundary

## Purpose

This document assigns each full-stack semantic fact to one primary owner.
Dataflow owns canonical software semantics, Fabric owns hardware structure and
capability, Mapping owns selected realization, Runtime executes an already
selected deployment, simulation produces observations, Evaluation owns typed
evidence, and DSE owns candidate acceptance.

No consumer may reconstruct a missing upstream fact from names, coordinates,
record order, or implementation conventions.

## Ownership Table

| Owner | Owns | Must not own |
|-------|------|--------------|
| Structured Program Candidate | Selected software scheduling, parallelization, vectorization, reduction, and AccCore/SpatialCore ownership decisions before canonical Dataflow lowering. | Hardware placement, physical routes, Mapping records, runtime remapping. |
| `dataflow.thread` | Non-recursive logical thread domains, InstructionCore program structure, async thread completion, and graph-launch containment. | Physical AccCore identity, physical scheduling, routes, tags, or hardware capacity. |
| `dataflow.graph` | Symbol-bearing canonical SpatialCore software DFG definitions, typed endpoints, causal edges, and graph completion. | FU selection, physical resources, routes, buffers, or system transport. |
| `fabric.module` | SpatialCore hardware templates, typed resources, endpoints, connectivity, capability, and use patterns. | Software actor grouping, selected workload placement, or Evaluation results. |
| `fabric.system` | Typed AccCore, InstructionCore, SpatialCore attachment, memory/service, Transport Architecture, implementation refinement, domains, and explicit system connectivity. | Software execution semantics, selected Mapping, runtime policy, or observations. |
| TechMapping | Compute and Memory Realizations, selected semantic encodings, actor-to-operation and boundary correspondence that, with configured-function topology, determines Compute internality, and exact Memory internal-edge witnesses. | Concrete physical placement, routes, software rewrites, QoR, or candidate ranking. |
| SpatialMapping | Exact TechMapping predecessor plus selected SpatialCore bindings and the minimal non-derived physical realization facts required for closure. | Regrouping actors, rematching capability, absolute software schedules, diagnostics, metrics, or DSE selection. |
| SystemMapping | `ExecutionBinding`, `ServiceRealization`, and `ResourceUse` across exact thread-launch coverage and system resources. | Runtime remapping, copied SpatialMapping facts, or a parallel transport authority. |
| Runtime ABI | Thread Dispatch, Spatial Launch, invocation memory descriptors, admission of already selected uses, dynamic handles, and runtime reports. | Choosing Mapping, repairing closure, or inventing implicit fallback. |
| DFG-sim | Hardware-unconstrained execution of Canonical Dataflow semantics. | Fabric topology, Mapping, or hardware contention. |
| CGRA-sim | Hardware-aware execution of one mapped SpatialCore using exact Fabric and SpatialMapping inputs. | InstructionCore or whole-system execution, Mapping search, or record repair. |
| Evaluation | Typed observations, metrics, provenance, fidelity, and evidence. | Mapping legality, selected Mapping facts, or candidate promotion. |
| Central DSE | Objectives, thresholds, candidate requests, ranking, acceptance, and promotion. | Mutating finalized Dataflow, Fabric, Mapping, or Evidence artifacts. |

## Dataflow Boundary

`dataflow.thread` is architecture-neutral and non-recursive. Host or runtime
orchestration launches a logical thread domain; a thread body may launch
`dataflow.graph` definitions; a graph body never launches a thread.

`!dataflow.thread_token` is the thread-completion domain. Graph control and
completion use canonical Dataflow event edges. Runtime handles are dynamic ABI
objects and are not either Dataflow token type.

`dataflow.graph` is the only canonical SpatialCore software DFG surface. Actor
grouping and selected FU realization belong to TechMapping, not to a
persistent subgraph operation. Logical axes and source layout do not imply
hardware coordinates or topology.

## Fabric Boundary

Fabric owns fully elaborated hardware structure. `fabric.module` describes a
reusable SpatialCore template. `fabric.system` describes typed system-level
resources and explicit directed connectivity. Coordinates and visualization
metadata never define reachability or legality.

The target system model uses typed AccCore, InstructionCore, SpatialCore,
memory/service, transport, implementation, and attachment concepts. Protocol
names identify implementation refinements; they do not replace architecture-
level capability and service contracts.

Ordinary directed connections use typed source and destination endpoints.
Resources with independent state, configuration, capacity, or parallel
identity are explicit Fabric entities rather than numbered generic edges.

## Mapping Boundary

The Mapping artifact family has three immutable completeness profiles:
TechMapping, SpatialMapping, and SystemMapping. There is no fourth physical
profile and no partial, rejected, or degraded Mapping lifecycle.

TechMapping is the only owner of software actor grouping and selected Fabric
semantic realization. Spatial PnR consumes the exact immutable predecessor and
adds concrete SpatialCore realization. SystemMapping composes thread execution,
services, transport, and resource use over exact system coverage.

A software edge is identified by its typed producer and consumer endpoints.
Spatial routing realizes one logical net as a rooted Route Tree with shared
trunks and sink branches; it does not persist one symbolic route per edge.

Mapping has no independent absolute Schedule IR. Resource-time behavior is
derived from Fabric-owned use patterns and selected event-relative
`ResourceUse`. Physical Tags are local to Fabric interpretation domains and
are not global token or firing identities.

Mapping verification owns legality and closure only. Failures and unsupported
inputs are ordinary results or reports. Diagnostics, QoR metrics, predictions,
fidelity, candidate ranking, and acceptance belong to Runtime, Evaluation, or
DSE according to their semantics, never to Mapping records.

Exact persistent SpatialMapping and SystemMapping record schemas remain open.
Consumers must not fill the gap with generic binding, route, schedule,
diagnostic, or metric records.

## Runtime Boundary

System execution has two distinct control boundaries:

```text
HostCore/runtime -> Thread Dispatch ABI -> AccCore.InstructionCore
AccCore.InstructionCore -> Spatial Launch ABI -> AccCore.SpatialCore
```

Thread Dispatch materializes a `dataflow.thread.launch` domain and its thread
completion. Spatial Launch materializes a `dataflow.graph.launch`, selects the
already bound compatible SpatialMapping, binds graph ports and memory
capabilities, and returns graph results and completion.

The two ABIs may reuse typed descriptor atoms, but they do not share one
generic launch record or completion domain. Runtime may wait or apply
backpressure while admitting fixed Mapping uses; it cannot choose another
AccCore, SpatialMapping, route, tag, context, or configuration.

## Simulation Boundary

DFG-sim consumes Canonical Dataflow and concrete runtime inputs without
Fabric or Mapping. CGRA-sim additionally consumes the exact SpatialCore Fabric
description and complete SpatialMapping. It validates but never repairs those
inputs.

InstructionCore, caches, coherence, NoC, and system time belong to the
external system simulator. A Loom Bridge invokes the shared SpatialCore
simulation library only at the Spatial Launch boundary.

Simulation outputs are Evaluation Evidence. They do not become Mapping facts
or legality exceptions.

## Evaluation And DSE Boundary

Evaluation owns typed observations and metric provenance. The central DSE
controller consumes immutable artifacts and Evidence to request or select new
Structured Program, Fabric, TechMapping, SpatialMapping, SystemMapping,
simulation, or backend candidates.

DSE may request a new artifact but must not mutate an existing artifact or
store selected-candidate state inside Mapping.

## Validation

The boundary is satisfied when every semantic fact has one owner, every
cross-artifact reference uses exact identity, and no consumer relies on hidden
state or textual order. Tests should cover typed ownership, exact identity
coupling, closed Mapping coverage, endpoint structural keys, deterministic
derived state, and verifier closure rather than retired record shapes.
