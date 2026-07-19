# Runtime ABI

## Purpose

The Runtime ABI executes an already compiled, mapped, and packaged workload.
It carries dynamic invocation values, memory capabilities, dependencies,
admission requests, and completion handles. It does not define Dataflow
semantics, Fabric topology, Mapping legality, or candidate selection.

Complete system execution has two control boundaries that must remain
distinct:

```text
HostCore/runtime -> Thread Dispatch ABI -> AccCore.InstructionCore
AccCore.InstructionCore -> Spatial Launch ABI -> AccCore.SpatialCore
```

The two ABIs may reuse typed descriptor atoms such as artifact identity,
address, size, permission, and domain coordinates. They do not share one
generic launch descriptor, completion domain, or fallback policy.

## Thread Dispatch ABI

Thread Dispatch materializes one logical `dataflow.thread.launch` operation
and its parameterized instance domain. It carries the information needed to
admit work to the selected AccCore and InstructionCore execution context,
including:

* exact deployable package and SystemMapping identity;
* the static root thread-launch operation;
* logical domain coordinates and parameters;
* explicit dependencies;
* authorized memory capabilities and invocation data; and
* long-lived reservations selected by SystemMapping where required.

Completion produces a host-visible runtime handle or event associated with
the thread dispatch. That dynamic object is not a persistent Mapping identity
and is not the `!dataflow.thread_token` value in canonical IR.

Thread Dispatch does not select graph placement or directly invoke a DFG or
CGRA simulator. The dispatched InstructionCore program issues Spatial Launch
requests for its local graph operations.

## Spatial Launch ABI

Spatial Launch materializes one static `dataflow.graph.launch` operation on
the selected local SpatialCore. It carries the information needed to configure
and admit one graph invocation, including:

* exact Canonical Dataflow graph and compatible SpatialMapping identity;
* the static graph-launch operation and logical invocation parameters;
* typed value, stream, control, and memory port bindings;
* invocation-local state context requirements;
* selected mapping-visible configuration; and
* dependencies and result/completion destinations.

The exact persistent or wire representation of these fields remains open.
Spatial Launch is a binding over existing Fabric endpoints and Mapping
services, not a new endpoint type or Mapping record family.

Completion returns graph results and the graph-local done event to the issuing
InstructionCore context. It does not complete the enclosing thread unless the
canonical program makes that causal relation explicit.

## Memory And Data Movement

Runtime memory descriptors identify invocation-specific buffers or memory
capabilities, access permissions, extents, element layout, and platform
handles required by the deployment. Exact descriptor fields remain subject to
the deployment and platform contract.

Fabric owns physical memory structures, address spaces, coherence, service
capability, and explicit transport. SpatialMapping and SystemMapping own the
selected physical and service realization. Runtime may bind concrete host or
device allocations to an already selected obligation; it must not choose a
new bank, route, service, coherence policy, or address authority.

Data movement and stream traffic use Fabric ports and selected
`ServiceRealization` paths. They do not introduce a third generic launch ABI.

## Admission

Runtime admission evaluates the dependencies and capacities required to
activate one already selected event-relative use set. Admission is atomic for
the mapping-visible resources required before that event fires. If the set is
not currently admissible, the request waits or applies backpressure.

Admission cannot change the selected AccCore, SpatialMapping, route, Physical
Tag, instruction context, service, or configuration. Offline verification
remains responsible for closure and deadlock legality; runtime opportunity
cannot repair an invalid mapping.

## Execution Disposition

InstructionCore-only execution is a normal compiler and SystemMapping
ownership choice, not a runtime fallback from failed SpatialMapping. A graph
selected for SpatialCore execution has no implicit InstructionCore substitute.

Any deployment that offers an alternative host or InstructionCore path must
name that disposition explicitly and prove its artifact, semantic, and ABI
contract. Those exact Deployment Artifact fields remain open. Runtime must not
invent a fallback flag or silently redirect work when a mapped launch fails.

## Runtime Package

A deployable package references the exact immutable software, Fabric,
Mapping, configuration, binary, and backend artifacts needed by its selected
execution dispositions. It may contain InstructionCore binaries,
SpatialCore configuration, memory images, and platform bindings.

The package is a dependency graph, not one linear launch record. Thread
Dispatch consumes its system-level execution binding; Spatial Launch consumes
the selected graph and SpatialMapping binding. Runtime-local handles and
addresses do not become persistent artifact identity.

Exact packaging and public driver output remain open under the Deployment
Artifact contract.

## Simulator Integration

HostCore and InstructionCore execution belong to the external system
simulator. The Loom Bridge is invoked only for Spatial Launch and calls the
shared SpatialCore DFG/CGRA/RTL model selected by the simulation binding.

The gem5 event queue is the whole-system time authority. A SpatialCore session
returns boundary events to the Bridge rather than maintaining a competing
system clock.

Standalone DFG-sim and CGRA-sim tools use the same SpatialCore simulation
library. Runtime must not route a HostCore dispatch directly to those tools as
an alternative launch mode.

## Diagnostics And Evidence

Runtime failures, admission waits, platform errors, and execution diagnostics
belong to runtime reports or Evaluation Evidence. They must cite exact package,
Mapping, Fabric, configuration, and input identities where applicable.

Runtime diagnostics do not become Mapping-owned records. Simulator and
performance observations are Evaluation facts, not runtime legality authority.

## Open Boundaries

This document does not define:

* exact Thread Dispatch or Spatial Launch wire structures;
* deployment disposition fields for explicit alternative executions;
* platform-specific memory descriptor encoding;
* gem5 binding structures; or
* public driver packaging syntax.

No generic launch wrapper, implicit fallback flag, or compatibility record may
stand in for those unclosed schemas.

## Validation

Anchor tests should cover separation of the two completion domains, exact
artifact coupling, typed memory authorization, fixed Mapping admission,
Spatial Launch bridge invocation, and deterministic runtime reports. Tests
must not preserve a single generic launch descriptor or implicit
InstructionCore fallback behavior.
