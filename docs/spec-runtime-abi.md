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

The persistent runtime-owned families are:

```text
loom.runtime_platform_binding 3.1
loom.gem5_simulation_binding  2.0
```

RuntimePlatformBinding 3.1 extends its exact dependency admission to
`loom.hardware_implementation 4.1`, including the payload-free `FabricModel`;
Gem5SimulationBinding 2.0 admits exact `loom.fabric 5.0` roots. Their record
shapes remain as specified below; no prior-version reference is reinterpreted
with a different accepted dependency schema.

Concrete device handles, leases, addresses, queues, and process state remain
transient. There is no generic runtime manifest or public manual-launch schema.

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

Compiler Target Binding is the sole owner of the final InstructionCore codegen
target triple, ABI, data layout, code model, backend CPU and feature spelling,
runtime and library requirements, and the binary compatibility proof. A
relocatable accelerator payload retains the LLVM module's own source triple and
data layout as validated projections under
`docs/spec-compiler-part-1-source.md`; those projections do not select the
InstructionCore target. The binding validates their compatibility while
referencing the exact InstructionCore Architectural Contract. It cannot add an
architectural capability or weaken an architectural requirement.

A target-specific binary is compiled under a compatible Compiler Target
Binding. Deployment must preserve and validate that relation against every
SystemMapping-selected AccCore on which the binary may execute. Runtime
consumes the validated relation; it does not renegotiate a target, ABI, feature
set, runtime, or library.

The exact persistent carrier, unique resolution rule, and binary relation are
owned by `docs/spec-executable-closure.md`. Runtime imports those artifacts and
cannot author another target tuple or compatibility cache.

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

Runtime assigns one execution-local `ThreadDispatchOccurrenceId` when a
dispatch commits. A dense instance is identified transiently by that ID plus
its coordinate tuple. A DynamicWork domain uses the same ID as the
`domain_instance` component of every `WorkItemId` in that dispatch. The ID is
unique only within the owning execution session, is never supplied as an
implicit thread-body argument, and cannot select Mapping, binary, route, or
configuration state. Repeated execution of the same root launch therefore
reuses its persistent bindings while retaining distinct runtime state.

Thread Dispatch never directly invokes a SpatialCore simulator. The selected
InstructionCore binary issues Spatial Launch requests for its local graph
launches.

For every committed thread occurrence, generated host glue forms one closed
`SpatialInvocationDemand` from the exact Canonical Dataflow boundary and the
concrete source-language call. The demand contains the Canonical Dataflow
identity, rooted thread and graph launch references, dense coordinates, typed
value tokens, authorized memory capabilities, and result destinations. Its
serialized Runtime ABI is invocation-local and disappears after execution; it
is not another workload, Mapping, or Deployment artifact.

The generated host glue, Thread Dispatch device, InstructionCore entry, Spatial
Bridge, and Spatial engine must transport this same byte sequence. None of
them may independently infer argument order, payload width, result ownership,
or memory authorization. Decoding reprojects the boundary shapes from the
exact Canonical Dataflow owner and rejects a foreign identity, reference,
coordinate rank, non-dense value ordinal, width mismatch, noncanonical padding,
or result destination outside the admitted capability surface.

A graph whose complete runtime input is already derived from Deployment and
System channel state has no source-call value or result payload. Its Thread
Dispatch invocation descriptor is the canonical zero-address, zero-size pair,
and the Spatial engine consumes the exact stored runtime input named by the
static launch projection. A nonzero address with zero size, or zero address
with nonzero size, is malformed. The empty descriptor cannot replace a
required dynamic demand or introduce another runtime-input owner.

The compiled contract is owned by the selected `ThreadEntryBinding` in
`loom.instruction_core_binary`. Its optional `spatial_invocation` names one
exact rooted graph and `loom.spatial_invocation_abi.v1`. Runtime projection,
Thread Dispatch, and the Spatial engine use that field to distinguish the
non-empty wire form from the static form. They must not infer the distinction
from graph operand/result counts, result uses, workload observability, or an
engine-specific heuristic.

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

The launch request carries two distinct planes. The immutable plane is the
exact Deployment `SpatialLaunchImage` payload. The dynamic plane is the
`SpatialInvocationDemand` received through Thread Dispatch. The Bridge keeps
both planes separate, validates their total size, and sends one framed request
to the selected engine. The engine must byte-compare the immutable plane with
the Deployment projection and independently decode the dynamic plane against
the same rooted graph launch. Treating a static launch image as if it supplied
runtime values is invalid.

For value results, the dynamic demand owns exact destination capabilities. The
engine derives result bytes only from the selected Spatial workload's typed
functional observations and writes them through the Bridge before publishing
completion. The InstructionCore then completes Thread Dispatch, and only then
may host glue read the result. This ordering is the mechanical connection
between the original software call and the Spatial execution; a detached
Spatial launch performed beside an unmodified HostCore call is not System
execution evidence.

### AdmissionImage

Runtime consumes `AdmissionImage` to evaluate immutable atomic activation,
release, and capacity relations for one concrete event occurrence. Absolute
start times, dynamic occurrence identities, runtime queues, and arbitration
microstate remain transient runtime state.

The image's canonical physical-capacity catalog is the compiled projection of
the exact Fabric contracts and selected static route image. Runtime initializes
each counter from `baseline_occupancy`, resolves activation-member claims by
their image-local cell ordinals, and validates the referenced physical owner
and UsePattern against the packaged Fabric closure. It does not renumber cells,
recompute route reservations, or treat Fabric initial occupancy as the whole
runtime baseline.

Runtime may apply one owner-defined admission relation to both Thread Dispatch
and Spatial Launch. Their launch and completion contracts remain distinct.

For an imported Spatial endpoint event, runtime matches any member of the
Dataflow-derived rooted endpoint alternative set. One matched trigger performs
one atomic acquisition. A causal release completes only after every original
release point has observed any one member of its own alternative set. Runtime
must not wait for every mutually exclusive actor transition and must not treat
one alternative as a second acquisition.

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

* thread-dispatch occurrence IDs and their concrete logical points;
* concrete logical coordinates and parameters;
* event-occurrence and completion handles;
* pending launch, channel, and memory requests;
* per-channel-branch endpoint order, message counters, and ordered-commit or
  reorder state;
* committed activation sets and active causal releases;
* dynamic capacity and admission-calendar state;
* buffer occupancy and credits exposed at the runtime boundary;
* invocation-specific allocation and address handles; and
* platform errors and completion events.

These values disappear after execution and never enter artifact identity.
They may affect when an immutable choice executes, but not which mapping choice
is selected.

An invocation result envelope retains the exact dynamic-demand bytes beside
the normalized Spatial boundary result in external-tool attempt material. The
System importer reconstructs the transient Spatial runtime input from those
bytes, revalidates the workload and result shapes, and checks every result write
against the decoded destination table. The envelope has no ArtifactIdentity
and cannot substitute for final `SimulationExecution` validation.

### Channel Event Execution

Logical channel message correspondence is owned exclusively by
`docs/spec-dataflow-part-1-streaming.md`. Runtime does not pair producer and
consumer thread activations. For each dynamic channel instance and logical
branch, it executes the specified flat producer and consumer event sequences
and delivers event `n` to receive event `n`.

Repeated endpoint instances are ordered by their deterministic launch issue
order, then by their normalized binding-local event order. Runtime may execute
later instances speculatively, but it cannot commit their channel events ahead
of earlier sequence contributions. Serialization, independent contexts and
queues, or deterministic reorder state may implement this rule. A runtime
arrival race cannot select message ownership.

Endpoint occurrence ordinals, event counters, and reorder entries are
transient execution state. They are not Mapping records, Artifact identities,
message fields, channel sessions, or Physical Tags. Runtime cannot use them to
choose a different route, context, service, or configuration.

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
  canonical_service_requirement
  logical_object_association
  exact request-leg payload projected from the Canonical Service Schema
  derived_addressed_view = absent | {
    access_geometry =
      Element {
        address
        element_bits
      }
    | Contiguous {
        base_address
        element_bits
        lane_count
      }
    | Indexed {
        addresses[lane_count]
        element_bits
      }
    active_lanes = All | Bits[lane_count]
  }
  ordering_and_visibility_context
}

SpatialServiceResponse {
  transient_transaction_handle
  exact response-leg payload projected from the Canonical Service Schema
  completion_and_visibility_event
}
```

`ordering_and_visibility_context` is a non-owning transient projection of the
issued actor's source execution context and incoming causal obligations. It
references the exact Dataflow contract and, when physical service has been
selected, its exact `MemoryConsistencyDomain`; it does not copy ordering,
scope, modification order, reads-from, coherence state, or a provider policy.

`completion_and_visibility_event` is the provider acknowledgement required to
advance that exact issued operation toward retirement. It reports fulfillment
of the selected contract; it does not become another visibility authority or a
persistent event record. Local providers and the gem5 Bridge use the same
boundary while retaining their distinct timing and dynamic-state ownership.

The canonical service requirement fixes read, write, RMW, compare-exchange, or
fence shape and its exact actor contract. Addressed operations carry the
runtime projection of their `CanonicalMemoryAccessView`; fence has no access
geometry, active lanes, address, or logical memory binding. The Canonical
Service Schema mechanically determines request values such as write data,
update, expected, desired, or mask and response values such as read data, old
value, or compare-exchange success. Runtime ABI does not duplicate that closed
operation union.

`derived_addressed_view` is a non-owning runtime projection of the exact actor
and request-leg values; it does not own software type, ranked vector shape, or
access semantics. `Element`
is one complete memory element even when that element's type is a vector.
`Contiguous` and `Indexed` use canonical row-major lane order. `Element`
requires `All`; an omitted software vector mask also projects to `All`, while a
dynamic mask projects to `Bits` with exactly `lane_count` entries. Inactive
lanes do not evaluate addresses or reach a memory service.

The geometry derives the complete addressed payload bit count. Exact request
and response fields retain that width, including canonical zero-filled inactive
load lanes. Neither geometry nor runtime payloads become another operation,
contract, or type authority.

The transient handle exists only for one execution. Logical-object association
is simulation/runtime metadata and need not be carried on physical wires.
The selected `MemoryEngineBinding + MemoryOperationEntry + optional
MemoryBinding` records may target a Local Memory Service, local consistency
domain, or manager endpoint without defining another request type. A local
model, external-service model, RTL adapter, or Bridge translates this boundary but
does not reinterpret the Memory Binding, address space, ordering, or visibility
contract. One request has exactly one timing authority.

The Operation Engine submits at most one logical `SpatialServiceRequest` for
one accepted actor firing. The selected Fabric use pattern and service adapter
may lower it to several implementation transactions or beats, but those child
transactions and per-lane atomic actions do not become new Runtime ABI
requests, transaction handles, actor firings, or independent retirement
events. They also cannot create additional provider-visible volatile
operations. Their ordering, resource claims, and result assembly are derived
from Fabric. An all-zero mask reaches no service and completes locally with
the canonical masked-load or masked-store result.

When a `fabric.mem` load response retires, read data and completion become one
indivisible `data + done` publication across all selected internal and external
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
binary.

Runtime-local addresses, handles, mutable leases, and admission state do not
enter Deployment identity. An immutable RuntimePlatformBinding referenced by
the Deployment is a packaged dependency, not a runtime-selected address. The
normalized Graph Execution Binding range helps derive imported
SpatialMappings, but it is not a second Deployment selection authority.

## Runtime Platform Binding And Loader

```text
RuntimePlatformBinding {
  version
  hardware_implementation_ref
  provider_binding {
    descriptor_identity
    descriptor_version
    implementation_semantic_identity
    runtime_abi_identity
  }
  identity_verification :
      HardwareReported { implementation_identity_endpoint_ref }
    | TrustedImmutable { attestation_blob }
  programming_bindings[] {
    programming_unit_ref
    implementation_interface_ref
    provider_endpoint_ref
  }
  memory_interface_bindings[] {
    implementation_interface_ref
    provider_endpoint_ref
  }
  completion_interface_bindings[] {
    implementation_interface_ref
    provider_endpoint_ref
  }
}
```

The HardwareImplementation reference recovers the exact Fabric System,
SpatialCore occurrence, ConfigurationABI, and ImplementationPlatform facts.
It does not recover or imply a System interconnect implementation. The binding
does not copy these facts. The provider binding identifies one
static typed provider contract and implementation; endpoint references are
closed typed values owned by that descriptor. Paths, device nodes, process IDs,
bus addresses, and handles are transient invocation state.

Every required subject-local programming, memory, completion, interrupt, and
external interface has exactly one binding. A configuration interface is
required only when its Programming Unit scope is exclusively the implementation
subject. Extra, missing, foreign, ambiguous, cross-occurrence, or
wrong-direction bindings are invalid. `HardwareReported` reads and validates
the exact HardwareImplementation identity through a declared interface.
`TrustedImmutable` binds an immutable signed or otherwise trusted attestation
blob when the hardware cannot report identity. The first version defines
identity verification, not a general package-signing or multi-tenant security
model.

The generated loader protocol is mechanical:

```text
validate package and Deployment closure
  -> enumerate provider devices
  -> verify exact implementation identity
  -> acquire authorization and exclusive lease
  -> quiesce and establish declared reset state
  -> install and verify all configuration images
  -> install static logical-memory images
  -> register host and InstructionCore entries
  -> activate
  -> execute and retire
```

For the common portable AXI4-Lite configuration profile, installation derives
the exact `ConfigurationTransportLayout` from the bound implementation,
SpatialCore occurrence, and ConfigurationABI. It writes the complete image,
commits the target unit, reads every active payload word back through the same
port, masks only ABI-unused high bits, and compares the result with the exact
`HardwareConfigurationImage`. A successful write response without matching
active readback is a programming failure. No runtime-authored address table or
wide parallel configuration shortcut is permitted.

The loader may coalesce several required programming operations into one
provider multicast only when all of the following are true:

* every exact image and programming binding remains independently present in
  the verified Deployment;
* the provider declares an atomic multicast operation for the selected
  endpoints;
* the definition-rebased configuration transport layouts are byte-identical;
* the complete image payload bytes and payload bit counts are equal; and
* all target SpatialCore occurrences are quiesced and share the same required
  activation event.

The provider receives a set of exact programming endpoint bindings. It may
lower that set to Core IDs, a hardware target bitmask, or repeated unicast
writes. Those encodings are provider-local transport facts and never enter
ConfigurationABI, HardwareConfigurationImage, Deployment, or the local
SpatialCore RTL interface. Failure on any target fails the multicast operation
and subjects every affected target to the existing reset-and-reverify rule.

Before activation, failure releases acquired resources after restoring the
declared clean state. After any partial programming or runtime fault, the
provider must reset and reverify the implementation before reuse; if it cannot
prove that state, the device is quarantined for that process and the execution
fails. Runtime never repairs a package, substitutes a compatible artifact, or
remaps work. A stable hand-written user launch API, dynamic shared-object
loading, firmware update protocol, remote deployment service, and partial
reconfiguration are deferred until they have concrete independent semantics.

RuntimePlatformBinding canonical JSON contains exact direct references and
canonically ordered interface bindings. Finalization verifies provider schema,
complete interface coverage, identity-verification support, and an independent
round-trip import before publication. Runtime enumeration results never enter
its identity.

## Gem5 Simulation Binding

Gem5 Simulation Binding is workload-independent. Its closed root is:

```text
Gem5SimulationBinding {
  version
  fabric_ref
  interconnect_implementation_ref
  gem5_build_identity {
    repository_identity
    full_commit_identity
    build_configuration_digest
    binary_fingerprint
  }
  bridge_abi_identity
  correspondences[] : Gem5Correspondence
}

Gem5Correspondence =
    Processor {
      processor_ref : HostCoreOccurrenceRef | InstructionCoreContextRef
      sim_object_ref
    }
  | SpatialBridge {
      spatial_core_occurrence_ref
      fabric_spatial_launch_boundary_ref
      bridge_endpoint_ref
    }
  | MemoryOrService {
      fabric_memory_or_service_ref
      sim_object_ref
      sim_port_ref
    }
  | Transport {
      fabric_transport_resource_or_endpoint_ref
      sim_object_ref
      sim_port_ref
    }
  | ExternalEndpoint {
      fabric_external_endpoint_ref
      sim_object_ref
      sim_port_ref
    }
```

`binary_fingerprint` is the exact SHA-256 of the executable produced by that
source and build-configuration identity. It is semantic build identity, not a
machine-local path. A readiness record proves that one local executable has
that fingerprint; it cannot supply or replace the binding-owned value.

The correspondence table is total over every modeled Fabric occurrence and
boundary and canonically ordered by typed Fabric reference. A SimObject or port
may serve several entries only when its exact gem5 model contract explicitly
permits that sharing. Free-form object paths and partial best-effort topology
matching are invalid.

Canonical JSON stores the exact direct Fabric and Interconnect Implementation
references, gem5 and Bridge identities, and this sorted table. Finalization
reimports the root, resolves every typed correspondence, proves totality and
declared sharing, and publishes atomically. Generated gem5 Python or SimObject
configuration is a projection, not a second binding authority.

Gem5 Simulation Binding is a simulator binding, not Fabric,
HardwareImplementation, or SystemMapping truth. Its finalizer validates the
first two authorities for each modeled InstructionCore, and Deployment
admission joins the third:

* the exact InstructionCore Architectural Contract;
* the exact InstructionCore Microarchitectural Realization, including
  execution structure, timing, capacity, and mapping-visible resources; and
* the compatible Compiler Target Binding used by the target-specific binary.

Because `loom.fabric 5.0` admits only the `RiscV` Architectural Contract, the
selected gem5 build and every `Processor` correspondence must provide a
compatible RISC-V ISA model. A build without that ISA or a correspondence to a
different ISA is typed `Unsupported`; the binding cannot retarget the binary
or substitute a host-native model.

It does not reference or copy SystemMapping or CompilerTargetBindings. The
system-simulator descriptor
references the shared system-simulation case signature with ordered
`deployment` and `system_model` roles; an ordinary `EvaluationRequest` binds
their exact `Deployment` and Gem5SimulationBinding
subjects. Its workload and concrete runtime data are exact
`SimulationWorkload` and `SimulationRuntimeInput` references; only the
remaining simulator model parameters belong to `ResolvedModelBinding`.
The same Gem5SimulationBinding may therefore execute several workloads or
compatible Deployments on the same hardware implementation. No separate
system-simulation request family exists.

Deployment admission joins every selected binary's exact
CompilerTargetBinding with the Fabric-owned Architectural Contract and the
corresponding `Processor` entry. A mismatch rejects the pair; it does not add a
CompilerTargetBinding copy to this root.

Unsupported or incompatible ISA, ABI, data layout, code model, backend feature,
component, protocol, or correspondence is a typed diagnostic. The binding must
not silently substitute an approximate topology, CPU, interconnect, memory
system, or binary contract.

## Bridge And Time Authority

HostCore and InstructionCore execution, system memory hierarchy, coherence,
and Interconnect Implementation microstate belong to gem5 during system
simulation. The Loom Bridge is invoked only at the Spatial Launch boundary and
calls the shared SpatialCore DFG, CGRA, or RTL execution library selected by
the exact Evaluation model descriptor. The Bridge does not own another engine
enum, fallback order, or simulator policy. Standalone simulation and the
Bridge use the same engine implementation; only the environment adapter and
time authority differ.

For mapped RTL, "the same engine implementation" means the same exact
Deployment-selected RTL, configuration image, harness semantics, result
normalization, and `MappedRtlSimulatorBinding` used by the standalone mapped
RTL provider. The System descriptor prepares and imports its own invocation;
it does not adopt standalone Evidence. The frozen HDL compiler produces the
invocation-local engine executable. That executable may launch only the exact
gem5 binary admitted by the `Gem5SimulationBinding`; the bundle must validate
that binary as a content-fingerprinted external input before execution. This
keeps the existing one-frozen-tool bundle invariant without inventing a
multi-tool selection authority.

The gem5 event queue is the only whole-system time authority. A SpatialCore
execution advances to its next system-boundary observable, such as a memory
request, completion, interrupt, mapped boundary transfer, or deterministic
wakeup time. The Bridge translates that event and resumes execution when the
corresponding gem5 event or response occurs.

The gem5 Thread Dispatch MMIO surface carries both target selection and the
address and size of the dynamic invocation wire. Dispatch snapshots all fields
atomically before activating an InstructionCore. That core receives the exact
static launch descriptor and dynamic invocation descriptor in separate ABI
registers. The zero-address, zero-size pair selects the static runtime-input
form defined above; all other dispatches require both fields. The Spatial
Bridge performs separate DMA reads and frames them only after the required
reads complete. Mutable MMIO registers, DMA scratch buffers, CPU state, socket
state, and event budgets are never cached as candidate-invariant state.

Gem5 executes concrete arbiter, queue, credit, protocol, cache, and memory
microstate from the selected implementation. Every cycle-visible grant follows
the exact Fabric contract or Mapping-selected exact hardware refinement. Gem5,
the Bridge, and runtime do not choose a fallback arbitration policy or
reinterpret SystemMapping routes and reservations. For a consistency domain
that crosses the system boundary, gem5 alone owns modification order,
reads-from, cache/coherence state, and system ordering. The Bridge carries the
typed request, response, and completion/visibility acknowledgement; it must not
maintain a shadow global consistency state.

Standalone DFG-sim and CGRA-sim tools use the same SpatialCore simulation
library. Runtime must not route a HostCore Thread Dispatch directly to those
tools as an alternate launch mode.

The first executable system provider is RISC-V machine-mode bare metal. Its
exact Deployment, Compiler Target Bindings, binaries, RuntimePlatformBinding,
and Gem5SimulationBinding remain ordinary typed owners. This provider choice
does not add an OS-image field to a Simulation Artifact and does not prevent a
later Linux/full-system provider from registering a distinct exact binding.

## Diagnostics And Evidence

Runtime waits, actual arbitration, completion events, terminal observables,
and typed activity summaries belong to `SimulationExecution` 1.0. Diagnostic
traces and tool payloads remain attempt or scratch material and have no
persistent runtime schema. Attempt timestamps, host/tool bindings, retries, and
execution-limit outcomes belong to the runtime owner's attempt record.
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
* element, contiguous, and indexed request geometry, dynamic masks, inactive-
  lane suppression, and all-zero-mask local completion;
* one logical Spatial Service request lowered by a declared Fabric use pattern
  to several implementation transactions without additional actor retirement;
* atomic load `data + done` and single-event store retirement;
* one local consistency-domain request and one gem5-owned external request with
  no duplicate modification-order or reads-from authority;
* consumption of a Deployment-owner-validated runtime-image child set, with a
  closure mismatch rejected and no missing Spatial launch path synthesized;
* mechanical Compiler Target Binding selection from the chosen InstructionCore
  Architectural Contract, with incompatible target-specific binaries rejected;
* complete RuntimePlatformBinding interface coverage, exact identity
  verification, failure-atomic programming, and quarantine when clean-state
  recovery cannot be proved;
* total Gem5Correspondence coverage with duplicate, foreign, partial, and
  undeclared-sharing cases rejected;
* gem5 model rejection on mismatch with the exact InstructionCore
  Architectural Contract, exact Microarchitectural Realization, or compatible
  Compiler Target Binding; and
* gem5 as the only whole-system time authority while following the exact
  Fabric/refinement grant policy through Evaluation roles `deployment` and
  `system_model`.

Tests should not preserve runtime queue layout, platform handle encoding,
gem5 internal statistics, or package printer format.
