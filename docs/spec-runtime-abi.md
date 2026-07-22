# Runtime ABI

## Purpose

The Runtime ABI executes an already compiled, mapped, verified, and packaged
workload. It carries concrete invocation values, memory capabilities,
dependencies, admission requests, and completion handles. It does not define
Dataflow semantics, Fabric topology, Mapping legality, or candidate selection.

Complete system execution has two distinct control boundaries:

```text
HostCore/runtime -> Thread Dispatch ABI -> AccCore.InstructionCore
AccCore.InstructionCore -> Spatial Launch ABI -> AccCore.SpatialCore
```

They may share typed atoms such as artifact identity, address, extent,
permission, and logical coordinates. They do not share one generic launch
record or completion domain.

## Immutable Mapping Contract

Runtime consumes one complete, independently verified SystemMapping. The
shared non-persistent `SystemMappingClosureProjection` derives the exact facts
needed by the verifier, Deployment builder, runtime, and simulator Bridge from:

```text
exact Canonical Dataflow Program
exact fully elaborated architecture-only Fabric system
complete SystemMapping
exact imported SpatialMapping set
```

The projection is rebuildable and has no artifact identity. It does not read
search configuration, constraint admission, Evaluation Evidence, or runtime
trace. Its caches may be deleted without losing truth.

Runtime may evaluate compiled relations for concrete coordinates and
parameters and choose an admission base time. It may not change the selected
AccCore, SpatialMapping, route, Physical Tag, context, service, or
configuration. Using different physical choices requires a different immutable
SystemMapping. A Physical Tag is a Mapping-assigned local interpretation key
for a may-overlap Fabric conflict domain, not a runtime firing, iteration,
invocation, or logical-token identity.

## Compiler Target And Binary Compatibility

For each concrete thread occurrence, its Thread Execution Binding selects one
AccCore. That selected AccCore resolves its exact InstructionCore Architectural
Contract and the derived `InstructionCoreContextRef`. A Compiler Target Binding
is mechanically selected and validated from that Architectural Contract;
SystemMapping does not carry or choose a compiler-target identity.

Compiler Target Binding is the sole owner of target triple, ABI, data layout,
code model, backend CPU and feature spelling, runtime and library requirements,
and the binary compatibility proof. These compiler-facing facts reference the
exact InstructionCore Architectural Contract but do not become Fabric hardware
truth. A binding cannot add an architectural capability or weaken an
architectural requirement.

A target-specific binary is compiled under a compatible Compiler Target
Binding. Deployment must preserve and validate that relation against every
SystemMapping-selected AccCore on which the binary may execute. Runtime
consumes the validated relation; it does not renegotiate a target, ABI, feature
set, runtime, or library.

This section closes only semantic ownership and compatibility. The exact
persistent carrier or root schema for Compiler Target Binding, its
multiplicity and tie-breaking, and the exact binary artifact binding remain
downstream open work. Nothing here defines canonical fields, bytes, or an
implementable wire contract.

## Deployment Runtime Images

`docs/spec-configuration-deployment.md` is the sole owner of Deployment
runtime-image child membership, identity status, schemas, canonical child keys
and ordering, and presence conditions. Runtime consumes only the children in a
finalized Deployment and rejects a closure mismatch; it does not derive,
reconcile, or rewrite that child set.

### ThreadDispatchImage

Runtime consumes `ThreadDispatchImage` to evaluate the immutable Thread
Execution Binding, resolve each concrete occurrence to its selected AccCore and
derived `InstructionCoreContextRef`, validate coordinates, parameters, memory
authorization, dependencies, and admission, and deliver thread completion.

For one concrete thread occurrence, runtime supplies coordinates, parameters,
authorized memory handles, and transient completion state. Completion is a
host-visible runtime event. It is neither a persistent Mapping identity nor a
replacement for the canonical `!dataflow.thread_token` relation.

Thread Dispatch never directly invokes a SpatialCore simulator. The selected
InstructionCore binary issues Spatial Launch requests for its local graph
launches.

### SpatialLaunchImage

Runtime consumes `SpatialLaunchImage` to evaluate the immutable Graph Execution
Binding, select the already bound SpatialMapping and configuration, bind typed
value, stream, control, and memory boundaries, request graph-start admission,
and deliver graph results and completion.

Spatial Launch is a binding over existing Fabric endpoints and Mapping
services, not a new endpoint type or record family. Completion returns graph
results and graph-local done to the issuing InstructionCore context. It does
not complete the enclosing thread unless the Canonical Dataflow Program makes
that relation explicit.

### AdmissionImage

Runtime consumes `AdmissionImage` to evaluate immutable atomic activation,
release, and capacity relations for one concrete event occurrence. Absolute
start times, dynamic occurrence identities, runtime queues, and arbitration
microstate remain transient runtime state.

Runtime may apply one owner-defined admission relation to both Thread Dispatch
and Spatial Launch. Their launch and completion contracts remain distinct.

## Admission

For one concrete event occurrence, runtime first checks explicit dependencies,
authorization, context availability, and every Mapping-visible resource that
must be acquired before activation. The derived activation set commits
atomically. If any required use is unavailable, the event does not fire and
waits or applies backpressure.

Admission acquires only resources required at that event. It does not reserve
all future uses of a thread or graph unless SystemMapping contains explicit
long-lived `ResourceUse` records triggered at the earlier event. Release follows
the compiled intrinsic or causal-event rule.

Offline verification remains responsible for capacity, ordering, progress,
and deadlock closure for all Fabric-permitted executions. Runtime opportunity,
finite simulation, or a favorable arbitration trace cannot repair an invalid
or incomplete Mapping proof.

## Runtime State

Runtime owns transient state only:

* concrete logical coordinates and parameters;
* event-occurrence and completion handles;
* pending launch, channel, and memory requests;
* committed activation sets and active causal releases;
* dynamic capacity and admission-calendar state;
* buffer occupancy and credits exposed at the runtime boundary;
* invocation-specific allocation and address handles; and
* platform errors and completion events.

These values disappear after execution and never enter artifact identity.
They may affect when an immutable choice executes, but not which mapping choice
is selected.

## Memory And Data Movement

Runtime memory descriptors bind invocation-specific allocations and
authorization to an already selected logical-memory obligation, physical
service envelope, address range, and permission contract. Runtime may choose a
concrete address or handle within that envelope. It may not choose a new bank,
service, route, coherence policy, address transform, or storage identity.

Value, stream, control, completion, and cross-AccCore token traffic use typed
transfer contracts selected by `ServiceRealization`. Memory uses the typed
operation-service contract and Canonical Service Schema legs. It is not recast
as an untyped data plane. Neither form creates a third launch ABI.

Responses return through the selected service path to the transaction's
recorded origin and transient context. Runtime does not configure an
independent response route.

Runtime ABI owns the one typed Spatial memory-service boundary used by local
memory service, manager endpoints, standalone models, RTL harnesses, and the
gem5 Bridge:

```text
SpatialServiceRequest {
  exact_memory_or_service_binding
  transient_transaction_handle
  typed_operation
  logical_object_association
  address
  size
  mask
  optional_write_data
  ordering_and_visibility_context
}

SpatialServiceResponse {
  transient_transaction_handle
  optional_read_data
  completion_and_visibility_event
}
```

The transient handle exists only for one execution. Logical-object association
is simulation/runtime metadata and need not be carried on physical wires.
The selected `MemoryEngineBinding + AccessEntry + MemoryBinding` records may
target a Local Memory Service or a manager endpoint without defining another
request type. A local model,
external-service model, RTL adapter, or Bridge translates this boundary but
does not reinterpret the Memory Binding, address space, ordering, or visibility
contract. One request has exactly one timing authority.

When a `fabric.mem` load response retires, read data and completion become one
atomic `data + done` publication across all selected internal and external
obligations. A store response retires as one `done` event. Runtime and adapters
must preserve those retirement events rather than splitting or reordering them.

## Execution Disposition

InstructionCore-only execution is a normal SystemMapping result, not a runtime
fallback from failed Spatial launch or admission. Runtime consumes the exact
runtime-image and configuration-image set finalized by the Deployment owner;
it neither synthesizes a Spatial launch path nor interprets an absent one as a
failed attempt. Runtime also cannot omit a selected programmable transport or
other configuration unit from the owner-validated Deployment closure.

A graph mapped to SpatialCore execution has no implicit InstructionCore
substitute. Any explicit alternative execution must be a separately compiled,
mapped, and packaged disposition selected before runtime.

## Deployment Artifact

`docs/spec-configuration-deployment.md` is the sole owner of the Deployment
root, exact dependency closure, ConfigurationABI and
HardwareConfigurationImage relations, package projection, and finalization
rules. Runtime consumes that exact Deployment and does not restate or repair
its closure. The finalized Deployment must preserve the confirmed compatibility
relation among each selected AccCore, compiler target, and target-specific
binary, while the exact binary artifact binding remains open as stated above.

Runtime-local addresses, handles, mutable leases, and admission state do not
enter Deployment identity. An immutable platform binding referenced by the
Deployment is a packaged dependency, not a runtime-selected address. The
normalized Graph Execution Binding range helps derive imported
SpatialMappings, but it is not a second Deployment selection authority.

## Gem5 Simulation Binding

Gem5 Simulation Binding is workload-independent. It relates the exact Fabric
Hardware Description and selected Interconnect Implementation refinement to
gem5 model and SimObject correspondences, model parameters, gem5 build
identity, and Bridge ABI identity.

Gem5 Simulation Binding is a simulator binding, not Fabric,
HardwareImplementation, or SystemMapping truth. For each modeled
InstructionCore it must validate all three authorities:

* the exact InstructionCore Architectural Contract;
* the exact InstructionCore Microarchitectural Realization, including
  execution structure, timing, capacity, and mapping-visible resources; and
* the compatible Compiler Target Binding used by the target-specific binary.

It does not reference or copy SystemMapping. The system-simulator descriptor
owns role-labeled subject slots `deployment` and `system_model`; an ordinary
`EvaluationRequest` binds their exact `Deployment` and Gem5SimulationBinding
subjects. Its workload and concrete runtime data are exact
`SimulationWorkload` and `SimulationRuntimeInput` references; only the
remaining simulator model parameters belong to `ResolvedModelBinding`.
The same Gem5SimulationBinding may therefore execute several workloads or
compatible Deployments on the same hardware implementation. No separate
system-simulation request family exists.

The exact persistent `Gem5SimulationBinding` root schema remains downstream
open work. The relations and Evaluation roles above do not define its fields,
cardinalities, serialization, canonical bytes, or implementable wire contract.

Unsupported or incompatible ISA, ABI, data layout, code model, backend feature,
component, protocol, or correspondence is a typed diagnostic. The binding must
not silently substitute an approximate topology, CPU, interconnect, memory
system, or binary contract.

## Bridge And Time Authority

HostCore and InstructionCore execution, system memory hierarchy, coherence,
and Interconnect Implementation microstate belong to gem5 during system
simulation. The Loom Bridge is invoked only at the Spatial Launch boundary and
calls the shared SpatialCore DFG, CGRA, or RTL simulation library selected by
the request.

The gem5 event queue is the only whole-system time authority. A SpatialCore
execution advances to its next system-boundary observable, such as a memory
request, completion, interrupt, mapped boundary transfer, or deterministic
wakeup time. The Bridge translates that event and resumes execution when the
corresponding gem5 event or response occurs.

Gem5 executes concrete arbiter, queue, credit, protocol, cache, and memory
microstate from the selected implementation. Every cycle-visible grant follows
the exact Fabric contract or Mapping-selected exact hardware refinement. Gem5,
the Bridge, and runtime do not choose a fallback arbitration policy or
reinterpret SystemMapping routes and reservations.

Standalone DFG-sim and CGRA-sim tools use the same SpatialCore simulation
library. Runtime must not route a HostCore Thread Dispatch directly to those
tools as an alternate launch mode.

## Diagnostics And Evidence

Runtime waits, actual arbitration, completion events, terminal observables,
typed activity summaries, and the trace manifest belong to
`SimulationExecution`. Opaque trace chunks and tool payloads belong to an
immutable raw detailed bundle. Attempt timestamps, host/tool bindings, retries,
and execution-limit outcomes belong to the runtime owner's attempt record.
Normalized outcome, metrics, and findings belong only to
`EvaluationEvidence`; human-readable runtime reports are projections of those
records. Their exact Request recovers Deployment, Mapping, Fabric,
implementation, binding, configuration, and input identities.

Diagnostics and observations do not become Mapping records. They cannot add a
missing binding, route, `ResourceUse`, service leg, or progress proof.

## Runtime Encoding Boundary

`docs/spec-configuration-deployment.md` owns runtime-image child schemas,
identity, canonicalization, Deployment placement, and package projection.
Platform-specific dynamic handle representation and gem5 model adapters are
runtime implementation details unless they change the typed Runtime ABI.
Runtime encoding is not physical hardware-configuration encoding;
`ConfigurationABI` remains the sole owner of the latter.

No runtime encoding may introduce a generic launch wrapper, implicit fallback,
runtime remapping flag, protocol-specific SystemMapping field, or second
Deployment selection authority.

## Validation Anchors

Anchor-level tests should cover:

* separate Thread Dispatch and Spatial Launch completion domains;
* atomic admission success and backpressure without remapping;
* the same typed Spatial Service request/response at a local service and a
  manager endpoint, with dual timing ownership rejected;
* atomic load `data + done` and single-event store retirement;
* consumption of a Deployment-owner-validated runtime-image child set, with a
  closure mismatch rejected and no missing Spatial launch path synthesized;
* mechanical Compiler Target Binding selection from the chosen InstructionCore
  Architectural Contract, with incompatible target-specific binaries rejected;
* gem5 model rejection on mismatch with the exact InstructionCore
  Architectural Contract, exact Microarchitectural Realization, or compatible
  Compiler Target Binding; and
* gem5 as the only whole-system time authority while following the exact
  Fabric/refinement grant policy through Evaluation roles `deployment` and
  `system_model`.

Tests should not preserve runtime queue layout, platform handle encoding,
gem5 internal statistics, or package printer format.
