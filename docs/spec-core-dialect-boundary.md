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
| `fabric.module` | SpatialCore hardware templates, typed resources, endpoints, connectivity, parameterized capability, and use patterns. | Software actor grouping, selected workload placement, or Evaluation results. |
| `fabric.system` | Typed AccCore, InstructionCore, SpatialCore occurrence binding, per-boundary endpoint attachments, memory/service, Transport Architecture, domains, and explicit architecture-level system connectivity. | Interconnect Implementation identity, software execution semantics, selected Mapping, runtime policy, or observations. |
| TechMapping | Compute and Memory Realizations, selected FU capability templates or Memory Operation Engine templates, and exact actor, operation, port, boundary, and internal-edge relations. | Concrete physical placement, routes, copied semantic parameters, raw configuration bits, QoR, or candidate ranking. |
| SpatialMapping | Exact TechMapping predecessor plus selected SpatialCore bindings, memory service placement, Route Trees, event-relative ResourceUse, and semantic-preserving physical refinements. | Regrouping actors, rematching capability, absolute software schedules, diagnostics, metrics, or DSE selection. |
| SystemMapping | Root thread-launch coverage, `B_thread` and `B_graph`, system service realization, system transport, and event-relative system ResourceUse. | Runtime remapping, copied SpatialMapping facts, a separately editable selected SpatialMapping set, or parallel transport authority. |
| ConfigurationABI | Programming-unit definitions, physical field encoding, and load, visibility, and activation contract for one exact Fabric. | Hardware capability, selected Mapping facts, backend-local alternate encodings, or runtime remapping. |
| HardwareConfigurationImage | Immutable encoded state for one exact ConfigurationABI programming unit and source Mapping. | Copied semantic configuration, implicit occurrence rebinding, or delta-patch state. |
| Deployment | Exact executable dependency closure across software, Mapping, implementations, images, runtime payloads, memory, and platform bindings. | Runtime handles, fallback Mapping, package-path semantics, or copied upstream facts. |
| Runtime ABI | Thread Dispatch, Spatial Launch, invocation memory descriptors, admission of already selected uses, dynamic handles, and transient runtime state. | Choosing Mapping, repairing closure, owning persistent reports, or inventing implicit fallback. |
| DFG-sim | Hardware-unconstrained execution of Canonical Dataflow semantics. | Fabric topology, Mapping, or hardware contention. |
| CGRA-sim | Hardware-aware execution of one mapped SpatialCore using exact Fabric and SpatialMapping inputs. | InstructionCore or whole-system execution, Mapping search, or record repair. |
| Evaluation | EvaluationRequest/Evidence, normalized observations, metric and finding registries, model capability descriptors, and observation-method references. | Invocation or attempt provenance, Mapping legality, selected Mapping facts, or candidate promotion. |
| Central DSE | Objectives, thresholds, candidate requests, ranking, acceptance, and promotion. | Mutating finalized Dataflow, Fabric, Mapping, or Evidence artifacts. |

## Dataflow Boundary

`dataflow.thread` is architecture-neutral and non-recursive. Host or runtime
orchestration launches a logical thread domain; a thread body may launch
`dataflow.graph` definitions; a graph body never launches a thread.

`!dataflow.thread_token` is the thread-completion domain. Graph control and
completion use canonical Dataflow event edges. Runtime handles are dynamic ABI
objects and are not either Dataflow token type.

`dataflow.graph.launch` is asynchronous. `dataflow.graph.wait` is the only
explicit InstructionCore stored-program wait for one or more graph-retirement
events; it does not convert them into thread tokens or introduce a generic
event-wait domain. Deferred value readiness, launch dependencies, channels,
and the thread completion frontier retain their distinct causal roles.

`dataflow.graph` is the only canonical SpatialCore software DFG surface. Actor
grouping and selected FU realization belong to TechMapping, not to a
persistent subgraph operation. Logical axes and source layout do not imply
hardware coordinates or topology.

## Fabric Boundary

Fabric owns fully elaborated hardware structure. `fabric.module` describes a
reusable SpatialCore template. `fabric.system` describes typed system-level
resources and explicit directed connectivity. Coordinates and visualization
metadata never define reachability or legality.

Fabric owns canonical FU definitions, parameterized FU capability-template
inventories and references, operation and memory
implementation families, ports, configuration and physical-refinement
domains, use-pattern schemas, capacity, and service guarantees. It owns each
configuration field's semantic meaning and typed value domain, but not its
physical bit or address encoding. Exact semantic realization is established by
Dataflow semantics and TechMapping relations. `ConfigurationABI` alone owns
the physical encoding specified by
`docs/spec-configuration-deployment.md`.

The Fabric Hardware Description family uses typed AccCore, InstructionCore,
SpatialCore, memory and service, transport, implementation, and attachment
concepts. The architecture-level `fabric.system` root ends at the Transport
Architecture. Interconnect Implementations are independent content-addressed
Fabric-family objects that reference and refine that exact architecture.
Protocol names identify those implementation refinements; they do not replace
architecture-level capability and service contracts or enter SystemMapping.

Ordinary directed connections use typed source and destination endpoints.
Resources with independent state, configuration, capacity, or parallel
identity are explicit Fabric entities rather than numbered generic edges.

## Mapping Boundary

The single `loom.mapping` family has three closed immutable roots in its
current complete schema version, `3.0`: `mapping.tech`, `mapping.spatial`, and
`mapping.system`.
`docs/spec-mapping-artifact.md` is the sole record and assembly authority;
`docs/spec-mapping-identity.md` owns Mapping-local identity, scoped imports,
and service-obligation keys; `docs/spec-fabric-identity.md` owns imported
Fabric-local target references; and
`docs/spec-mapping-verification.md` owns verifier behavior.

TechMapping binds one exact Canonical Dataflow Program and one exact Fabric
Hardware Description. It selects parameterized capability templates and
stores the exact ordered relations needed to instantiate semantic
realizations. Operation-specific masks, configured fields, raw `sw_configs`,
and configured-function views are derived rather than copied into Mapping.
The selected FU capability is an exact Fabric-owned
`FabricFuCapabilityTemplateRef`; no Mapping-owned compute encoding or copied
configured graph is a parallel authority.

SpatialMapping binds one exact TechMapping predecessor plus exact Dataflow and
Fabric aliases for scoped references. It preserves TechMapping semantics and
owns the non-derived SpatialCore physical selections required for complete
closure. Resolved config and the independent MappingConstraintSet Artifact
affect search and admission, but do not enter SpatialMapping identity. The
invocation binds the exact constraint-set ArtifactIdentity.

SystemMapping has one canonical non-empty root-thread-launch coverage set.
Its imported SpatialMapping table must equal the finite unique range of
normalized `B_graph` over all reachable static graph launches and legal
may-domain points. The table is a canonical reference structure, not a second
selected-set authority. A graph-free InstructionCore-only closure has an
empty table without a dummy SpatialMapping.

A software edge is identified by its typed producer and consumer endpoints.
Spatial and system routing realize one logical transfer family as a rooted
flat Route Tree with shared trunks and sink attachments; they do not persist
one symbolic route per edge or sink.

Mapping has no independent absolute Schedule IR. Resource-time behavior is
derived from existing Dataflow causal events, Fabric-owned use patterns, and
selected event-relative ResourceUse. Physical Tags are typed sharing
assignments at real writers or ingress points, not global token or firing
identities.

Base verification owns intrinsic legality and closure. MappingConstraintSet
admission is a separate invocation gate and does not change artifact identity
or intrinsic validity. Quality gates, metrics, findings, model fallback,
ranking, and promotion remain Evaluation and central DSE responsibilities.

Failures, unsupported results, reports, diagnostics, proof witnesses, and
Evidence are outside Mapping semantic bytes. Mapping has no partial, rejected,
degraded, diagnostic, or fourth physical profile.

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

Workload-running simulation produces one `SimulationExecution` and normalized
`EvaluationEvidence` under their distinct ownership contracts. Raw material is
attempt or scratch state until its exact Artifact owner is defined.
Architecture-only checks produce no empty execution.
None of these outputs becomes a Mapping fact or legality exception.

## Evaluation And DSE Boundary

Evaluation owns typed observations and metric provenance. The central DSE
controller consumes immutable artifacts and Evidence to request or select new
Structured Program, Fabric, TechMapping, SpatialMapping, SystemMapping,
simulation, or backend candidates.

DSE may request a new artifact but must not mutate an existing artifact or
store selected-candidate state inside Mapping.

## Validation

The boundary is satisfied when every semantic fact has one owner, every
cross-artifact reference uses exact identity, every Mapping profile is
complete, and no consumer relies on hidden state or textual order. Tests cover
typed ownership, exact binding and import coupling, parameterized capability
relations, closed coverage, structural keys, deterministic canonicalization,
base verification, separate constraint admission, and cross-layer closure
rather than retired record shapes or implementation details.
