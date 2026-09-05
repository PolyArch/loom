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
loom.runtime_platform_binding 4.0
loom.gem5_simulation_binding  2.0
```

RuntimePlatformBinding 4.0 retains the optional provider-owned
`resource_time_cost_model` derived from the exact registered descriptor. Its
four strictly positive components price one copy setup, each copied byte, each
changed configuration word, and each configuration commit. This is an
incompatible addition to 3.1. RuntimePlatformBinding still admits exact
`loom.hardware_implementation 4.1` dependencies, including the payload-free
`FabricModel`; Gem5SimulationBinding 2.0 admits exact `loom.fabric 7.1` roots.
No prior-version reference is reinterpreted with a different record shape or
accepted dependency schema.

Concrete device handles, leases, addresses, queues, and process state remain
transient. There is no generic runtime-owned manifest or public manual-launch
schema. The Application layer does publish the incompatible
`loom.application.runtime_manifest 6.0` activation closure. It references one
strictly imported StructuredProgram source workload/runtime pair, the exact
source-backed Spatial replay cases, one completed pair decision, the selected
SystemMapping and Deployment, completed runtime/oracle Evidence, the exact
Deployment-owned System workload/runtime pair, and an optional finite verified
resource-time transition graph. Version 6.0 admits the `copied` logical-memory
migration disposition and retains its provider-derived nonzero cost; this is
an incompatible extension of 5.0. The canonical set of durable
`loom.dse.hardware_mutation_repair_record` roots produced while the pair was
evaluated and a nullable exact selected-record root remain required. The
exact set and selection are owned by `loom.application.activation_decision`
2.0; construction and strict import require agreement with that owner.
An empty set and null selection mean that no
hardware-mutation repair was executed or selected. These
references authorize no new Mapping, route, entry, or input construction at
runtime. Strict import proves that every record has the selected Mapping's
Dataflow owner and belongs to a mutation lineage reachable from the original
pair Fabric. When a `hardware_dse_alternative` was produced by an executed
mutation repair, it names the unique reachable record whose child System and
preserve-first child SystemMapping equal the selected owners; a general
hardware-frontier selection has no parent-Mapping repair record and leaves the
field null. A unique repair record that selects the activation SystemMapping
must be named; omission is invalid. Records for evaluated but nonselected
children remain separate provenance. Every non-hardware pair disposition
requires a null selected record.

An Application package contains the exact object/blob closure of that
manifest, every endpoint Deployment, and every manifest-named hardware
mutation repair record. Repair-record closure includes its parent and child
Systems and the complete parent, cold, incremental, and quality-observation
SystemMapping closures. It also strictly imports every failed Tech or Spatial
rebase parent and includes that lower Mapping's Dataflow, Module, and upstream
Tech dependency. Execution first validates the source package, copies its
immutable stores into a new workspace, and strictly imports the workspace copy
in isolated import sessions. The System runner consumes the manifest's
Deployment-owned activation roots. A command-line program-entry or freshly
published workload/runtime pair cannot override the package.
The additive summary projection
`loom.application_runtime_manifest_binding` 1.1 names the same repair-record
set and nullable exact selected record; 1.0 carried neither field.

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

## Ordered Channel Session ABI

`OrderedChannelABI` is the direct transient owner of one ordered channel
session. It owns the producer send sequence, one receive sequence and at most
one reservation per consumer branch, acknowledgement, multicast retention,
and the selected bounded message capacity. The caller commits producer events
in the canonical order defined by Dataflow; concurrent arrival never selects
sequence order. One accepted send receives the next sequence exactly once, a
consumer reserves only its oldest unacknowledged message, and storage is
reclaimed only after every branch acknowledges that message. A full capacity
returns `WouldBlock` without consuming a sequence.

The optional reusable profile binds finite flat producer and per-consumer
event counts to one transient generation. The exact Dataflow launch/channel
lineage and already admitted service envelope remain the session authority;
the ABI's generation ordinal only distinguishes consecutive uses of that same
session. It is not persistent identity and cannot select endpoints, Mapping,
route, capacity, or hardware state. Opening a generation changes none of those
facts.

The producer may finish only after accepting its declared count. Each consumer
may finish only after producer finish, acknowledgement of its declared count,
and release of any live reservation. At that boundary, receive returns a
generation-bound `EndOfGeneration` lifecycle ticket rather than a payload.
Collective join requires the producer and every consumer to finish, no live
reservation, and no retained message. Cancellation invalidates reservations
and discards retained transient messages; it neither completes a
`DynamicWorkDomain` nor manufactures program-visible data. Reset is legal only
after collective join or cancellation and increments the generation before
reopening pristine counters.

Deficit, excess static rate, pending consumer, outstanding reservation,
cancelled generation, stale ticket, lifecycle misuse, and identity exhaustion
remain distinct typed outcomes. Rejected operations do not consume sequence,
acknowledgement, reservation, or generation state. A one-shot caller that does
not bind finite rates retains the original ordered send/receive ABI, but it
cannot claim reusable-session or lifecycle conformance.

This lifecycle spans one complete logical channel invocation. It does not
segment producer or consumer activations, reinterpret rate conversion, or use
queue emptiness as EOS. A production path claims this profile only when its
finite counts and endpoint membership are derived from the same canonical
launch correspondence that created the channel and it uses the existing
pre-admitted service/route envelope.

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

#### Bounded Dynamic Work Scheduling

For one admitted DynamicWork dispatch, the bounded scheduler kernel owns a
fixed transient worker set, one finite-capacity deque per worker, queue
placement, live assignment validation, cooperative cancellation delivery, and
an ordered transition trace. It retains exactly one `DynamicWorkDomain`-issued
responsibility for every queued or assigned item. `DynamicWorkDomain` remains
the sole owner of root and child responsibility acquisition, retirement, and
the collective completion transition. Empty deques, idle workers, cancellation
requests, and trace position cannot complete the dispatch.

A worker has at most one active assignment. It acquires the back of its local
deque or, when idle, visits other workers in cyclic ordinal order and steals
the front of the first nonempty deque. Only queued, not-yet-started items may
be stolen. The move-only assignment is bound to its scheduler, live item, and
worker; worker ordinal and queue position remain transient and cannot alter
`WorkItemId` or select Mapping.

Child publication checks local capacity before asking `DynamicWorkDomain` to
spawn. A full deque returns typed `WouldBlock` without acquiring responsibility
or consuming the parent's next child ordinal. The parent may retry only after
relevant queue state changes. The scheduler mutex serializes publication,
acquisition, cancellation, and retirement. Its unlock/lock order makes the
queued payload and responsibility visible to the acquiring host worker, but it
does not replace program atomics or a selected Fabric memory-consistency
realization.

The holder of the execution-local scheduler is the cancellation authority; a
`WorkItemId` passed to that scheduler is only a selector. Queued cancellation
immediately retires that responsibility and returns the exact
`RetirementEffect`. Active cancellation is one idempotent request that blocks
normal completion and later child publication until the assigned worker
cancels; cancellation without such a request is rejected. It does not
recursively cancel already published descendants. The ordered trace contains
successful state-changing transitions and their item, source worker, and
target worker; its vector position is the sole transition order. Empty
acquisition, `WouldBlock`, repeated cancellation, and rejected calls add no
transition.

The worker set and queued capacity are bounded; retained replay history grows
with committed transitions. Bounded trace retention is not part of this
standalone kernel.

The scheduler kernel itself does not read Mapping. The bounded execution
adapter composes it with Canonical Dataflow and verified SystemMapping. The
adapter admits one byte-addressable root payload, no launch captures, and at
most one direct graph launch. It allocates a nonzero dispatch occurrence,
admits the root through the bounded scheduler, and removes execution-local
dispatch and item lineage to recover the Dataflow-owned domain execution
class. Every item in this first profile shares that one stable class because
it executes the same thread definition; `WorkItemId` remains its sole logical
identity. Runtime evaluates the exact stable-key thread, graph, and
service-plan bindings from the class and never from worker placement.

The generic synchronous executor boundary supplies those selections to an
external execution owner once per item. One returned finite child group is
published atomically after queue-capacity and payload-width validation, then
the parent retires. Workers are visited in deterministic cyclic order, so an
idle worker performs the scheduler's ordinary front steal while local work
uses the back. Executor failure or an invalid result cooperatively cancels the
active item and every queued responsibility before the adapter returns.
Explicit item cancellation does not recursively cancel descendants; remaining
items continue until the responsibility-domain join. The external owner's
completion report is not independent source-body execution evidence. Repeated
calls reuse persistent Mapping while receiving distinct occurrence IDs.

The concrete CGRA entry is the first closed execution profile. It requires one
signless scalar integer work item forwarded unchanged to the sole value input
of one direct graph. The thread body may contain only that graph launch and its
yield, and the graph boundary has no stream or memory ports. Runtime derives
the graph input token from the assigned little-endian payload bytes, prepares
the CGRA engine from the SpatialMapping selected by SystemMapping, and reports
success only after CGRA retirement and collective DynamicWork completion.
Event-frame exhaustion or a halted engine returns the original typed CGRA
outcome and counters as incomplete evidence; it is never reclassified as
Mapping infeasibility.

The adapter does not define a source-level child-publication operation or
lineage, carry launch captures, or connect StableKeyLookup through the
version-1.0 Thread Dispatch and Spatial Launch images to a hardware provider.
Callback-returned children therefore establish scheduler execution and join,
not compiler-generated `dataflow.work.spawn` execution. Representable
out-of-profile cases retain typed projection, execution, or runtime-image
reasons; they are not collapsed into a generic unsupported domain. Provider
image transport remains unavailable.
Active safe-point migration and remapping are separate contracts and are not
inferred from queued-item stealing.

#### Finite Resource-Time Selection

`ResourceTimeTransitionSelectionSession` consumes one immutable
`ResourceTimeTransitionGraph` only after the PnR-owned independent verifier
has imported every Mapping and Deployment endpoint and replayed every edge.
The caller also supplies the finalized entry Deployment; its exact reference
and SystemMapping must equal the graph entry. The graph remains caller-owned
invocation input. Runtime does not persist it in a Deployment image, recover it
from vector order, or make it a second Mapping authority.

The session begins at the exact entry endpoint and records collective root
completions in caller commit order. Its root inventory comes from the imported
entry SystemMapping. A completion derives its trigger from Canonical Dataflow,
and an optional child is an explicit selection request. Selection succeeds
only when one verified edge matches the current parent, full child endpoint,
completion trigger, completion safe point, and exact canonical
`completedBefore` subset. An absent child is an explicit stay decision. Edge
and endpoint ordinals are never priority or replay identity.

The graph verifier proves exact endpoint closure, one shared canonical root
scope, individual edge closure, and monotonically realizable completion
frontiers from the entry. The selector still rechecks the exact completed-root
set at each call and leaves an edge unselected unless the caller has committed
the required frontier. `completeRoot` records that pure decision without
changing a provider. `completeRootAndActivate` instead commits a selected edge
only through the loaded Deployment that exactly matches the current endpoint.

`createPrepared` independently verifies the graph against the loaded entry,
derives one prepared operation for every selectable verified edge, imports its
child Deployment and executable closure during invocation setup, and requires
the same registered provider descriptor, implementation semantics, Runtime
ABI, and provider-derived cost model. An entry-only graph needs no provider
capability. The admitted profile also requires empty parent and child
static-memory images. For a zero-work edge the provider copies the child
activation into a provider-owned transient handle. For a nonzero edge it
additionally copies the exact child configuration-word delta and logical-memory
source/destination plan into that handle. Logical-memory contents are not read
during preparation. Preparation cost is setup cost; it is not silently
attributed to the PnR-owned reprogramming or migration fields.

The returned session and loaded Deployment share one transient, unforgeable
association for that exact graph preparation. A pure selection or replay
session has no such association, and a session prepared for another loaded
Deployment cannot activate this one. A failed provider preparation retains no
state from that call. Runtime discards every earlier prepared handle. Complete
cleanup restores the unprepared state and permits a later preparation attempt;
if a discard itself fails, the loaded Deployment is locked against another
attempt and its ordinary quiesce/reset lifecycle remains the final bounded
cleanup owner.

At a completion frontier, the combined commit path is the only owner allowed
to select a prepared handle. Every selectable edge has one reusable handle;
this includes an edge returning to the entry because a monotonically growing
completion frontier may revisit an earlier Mapping endpoint. The switch
performs no Artifact or Blob import and sends no executable or runtime-image
bytes. Under the existing lease, the provider reads each complete live-memory
source, applies every exact changed configuration word and commit, writes each
equal-extent destination, and changes the active endpoint atomically.
Provider rejection leaves the handle reusable and the parent Deployment active,
while the selector restores its endpoint, completion frontier, and replay log.
The host cost of this bounded control operation remains ordinary runtime
timing. Edge costs come only from the RuntimePlatformBinding's derived provider
model: changed words and commits determine reprogramming time, while copy setup
and complete byte extent determine migration time. A zero edge cost does not
mean zero CPU instruction latency.
Loaded Deployment teardown explicitly discards every retained handle before
the ordinary quiesce/reset and lease release. Reset invalidates all prepared
handles even when an earlier discard failed.

`ApplicationResourceTimeExecutionSession` is the synchronous Application
adapter over this selector. It accepts the exact root lifecycle observation
at the provider commit boundary, including the occurrence and event
coordinate. A root completion selects a child only when the current frontier
has exactly one legal edge. No legal edge is an accepted, typed
`NoLegalTransition` stay decision. More than one legal edge requires an
explicit runtime policy and leaves the selector and loaded Deployment
unchanged. A selected edge is committed only through
`completeRootAndActivate`; the adapter never derives a Mapping, runs PnR, or
uses graph order as policy.

The joined event sequence may be published as
`loom.application.resource_time_execution_trace` version 2.0. Version 2.0
requires an exact `loom.application.runtime_manifest 6.0` root and is
incompatible with 1.0 rather than reinterpreting its accepted manifest
dependency. The trace names
its exact Application runtime manifest, retains the root event occurrence and
coordinate, parent and resulting Mapping/Deployment endpoints, actual active
and completed root sets, typed outcome, and the manifest-owned QoR evidence
references. For a selected child, the event, parent, and resulting endpoint
identify one exact edge in the manifest graph. Strict import replays the
selector and mechanically restores that edge's allocation and live-state
sets, resource/configuration/route deltas, and migration/reprogramming costs.
The trace does not copy those fields into a competing transition owner.

Publishing such a trace requires a joined synchronous session. A lifecycle
file imported after execution is not a synchronous session and cannot be
published as activation evidence. Providers that cannot acknowledge the
completion callback before later root dispatch remain incapable of this
contract.

`loadApplicationDeployment` is the invocation-local Application-owned join
from a retained `ApplicationDeploymentArtifacts` result into this runtime
lifecycle. It first loads the exact `FinalizedDeployment` through the ordinary
loader. If and only if the compiler result carries a
`ResourceTimeTransitionGraph`, it then creates the prepared selector against
that same loaded object. The Deployment and verified graph remain the
operational sources of truth; build-time hardware, binary, Spectrum, and
transition-evidence projections do not independently authorize runtime state.
An absent graph produces an ordinary loaded Deployment with no selector, never
a runtime-derived entry graph. This API does not define deployment-package
serialization; a package consumer must recover the exact graph and every
endpoint Deployment closure before invoking it.

Rejected calls do not change the endpoint, completed-root subset, lifecycle,
or replay. Replay uses typed roots and full endpoints and re-executes every
accepted completion, stay, join, and cancellation decision. Cancellation is
idempotent for the selector but does not cancel provider, channel, or
DynamicWork responsibilities. `joinMappedRoots` proves only that every root in
the entry Mapping has collectively completed; host residual work and process
termination remain separate runtime owners. Replay is deliberately a pure
decision replay. A runtime wishing to reproduce execution applies those
validated records one at a time through a prepared session, so a provider
failure cannot hide or discard the already accepted selector prefix.

The admitted selector profile contains verified completion edges whose
logical-memory records are either `retained_in_place` at exact zero cost or
`copied` between one complete, statically bounded pair of distinct equal-extent
targets at provider-derived nonzero cost. Changed child configuration images
are reduced to exact word ordinals and have a nonzero provider-derived word and
commit cost. The selector does not install static memory, snapshot a scheduler,
or move tokens or channel payloads. Prepared activation replacement does not
itself report the completion event, start remaining child roots, or resume
DynamicWork; those actions require their existing execution owners to call the
combined commit path and continue from the child Mapping. A graph with an
unknown or split memory extent, ordered-channel state, DynamicWork state, or an
explicit safe point other than canonical root completion remains a typed
refusal.

The Thread Dispatch provider maintains one bounded transient record per exact
Deployment target. Target selection addresses that record for submission and
status queries; it is not the completion identity. A successful submission
atomically snapshots the invocation descriptor, assigns a nonzero occurrence
ID, and may coexist with submissions to independent InstructionCore contexts.
Targets that resolve to one physical InstructionCore remain mutually exclusive
and wait for that context rather than manufacturing another hardware thread.
Worker completion is qualified by its target record, and Host glue verifies the
record's occurrence ID before accepting completion. The record bound is the
existing runtime invocation bound; no unbounded software queue is implied.

Generated Host glue preserves the Canonical Dataflow asynchronous boundary. It
submits all points of a root at `dataflow.thread.launch`, returns one transient
collective handle, and joins the recorded point occurrences only at the
corresponding `dataflow.thread.wait`. An all-of wait may join handles in any
deterministic order after every preceding launch has been submitted. Replacing
each launch with submit-and-immediate-wait is invalid because it can prevent a
later channel producer from ever becoming active.

For a statically bounded dense root, Canonical Dataflow alone enumerates the
finite coordinate tuples in row-major order under the runtime invocation
bound. Generated host glue emits one completed Thread Dispatch per tuple and
evaluates the verified SystemMapping relation to select the corresponding
Deployment target. Dispatches may select the same InstructionCore more than
once: those occurrences are ordered, mutually exclusive uses of one compiled
context, not additional resident contexts. Each occurrence rebuilds its
invocation wire and memory snapshot after the preceding occurrence completes;
no mutable wire, queue, CPU, bridge, or engine state is reused as a derived
fact. A dynamic-bound, nested, over-bound, or non-dense domain remains typed
Unsupported rather than being truncated or assigned an inferred coordinate.
The bounded DynamicWork adapter follows the separate stable-class contract
above and never invents a coordinate.

Reachable selected roots need not share one source callable. Generated host
glue groups the flat rooted-launch set by its exact callable owner while
preserving global launch and dispatch-target ordinals. It materializes concrete
call sites in decreasing direct-call-path depth. When a selected callable
directly invokes another selected callable on the same path, the outer clone's
exact call site is rebound to the already materialized inner clone; calls with
another caller, callee, ordinal, or path prefix are unchanged. This composes
nested operator protocols without treating a symbol name as a global call-site
selector or executing an unmodified inner callable from an accelerated outer
clone.

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

For a CGRA actor action with causal release, owner work completion and physical
claim retirement are distinct events. Grant or owner commit completes the
intrinsic operation and permits the actor's selected transport instances to
advance. The physical claim retires only after those instances satisfy the
Mapping-derived causal release. Treating claim retirement as actor work
completion creates a self-wait; treating owner completion as claim retirement
releases the physical resource too early.

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

A Deployment dispatch target, a physical Spatial Bridge, and a dynamic Spatial
invocation have independent cardinalities. A dispatch target selects one exact
InstructionCore entry, Spatial workload, SpatialMapping, configuration, and
execution context. All targets assigned to the same AccCore share one physical
Bridge session and one PIO range. Any target in that session may be invoked
more than once by the target program. The Bridge therefore publishes one dense
dynamic result sequence for the session, not one result per dispatch target.

The current incompatible Spatial Bridge message ABI is
`loom.gem5_spatial_bridge_abi.v5`. Its Spatial launch envelope has magic
`LGL2` and carries the physical bridge-session ordinal, the immutable static
launch bytes, and the optional dynamic invocation bytes. The ordinal selects
one entry partition in the exact System projection; it cannot select another
Mapping, route, service, or configuration. A result continues to name its
entry by the session-local ordinal described below, rather than by the
provider's process-wide entry index.

The current strict gem5 System projection schema is
`loom.gem5_system_projection.12`. For every Bridge session it records aligned
arrays of dispatch-target ordinals, execution-context keys, and Spatial
workload identities. These arrays are derived together from the immutable
Deployment and System execution projection. Provider command arguments only
start the shared or per-Bridge engine; they are not a semantic source from
which an importer may infer target ownership. Sharing one engine across
several Bridges therefore does not move workloads into the command-owning
Bridge.

Projection 12 also requires `dispatch.root_event_trace_path`, a logical target
count, and parallel endpoint offset/enable arrays. The arrays define a finite
runtime endpoint table over the immutable dispatch records; they cannot create
new targets. An endpoint with dispatch disabled may only be selected as a
terminal safe-point decision, after which any later dispatch is rejected by the
device. The optional `dispatch.root_event_control_path` enables an acknowledged
controller for a prepared finite transition graph. The path is empty for an
ordinary invocation and otherwise names a Unix stream socket relative to the
bundle root. Connect, send, and receive operations have one bounded
`gem5RootEventControlTimeoutMilliseconds` deadline; a missing or stalled
controller is an invocation failure, never an unbounded simulation wait.

The endpoint table of a controlled invocation is derived from the
independently verified transition graph by `deriveGem5RootEventEndpointTable`
(`Gem5RootEventEndpointTable`): ordinal zero is the entry Deployment and every
later ordinal is one other graph endpoint in graph order, addressed by its
exact Deployment reference. Because a nonzero endpoint has no dispatch
targets, the table admits only terminal edges (every mapped root completed or
completing, no region active under the child); a non-terminal edge is the
typed refusal `non_terminal_edge` and a later dispatch after a terminal
activation is a device protocol failure, never a silent continuation. The
production driver is the System DFG cell of `loom-system-run`: when the
package manifest carries a transition graph, the driver loads the Application
Deployment with its prepared selector before the cell starts, prepares the
completion-controlled invocation, serves the control socket through
`Gem5RootEventController`, answers every request from
`LoadedApplicationDeployment::driveGem5RootEvent` (start continues, a typed
stay keeps the endpoint, a selected child activates its endpoint ordinal)
before the device continues, executes the bundle fresh, and requires the
device-published lifecycle to equal the acknowledged sequence. The trace is
published only from that synchronous session; the CGRA cell runs uncontrolled
and its lifecycle is cross-checked against the DFG cell. A graph the drive
refuses runs the entry Deployment on both cells and records the typed refusal
in the workspace manifest instead of activation evidence.

Projection 12 requires the Thread Dispatch device to write one transient
big-endian root-lifecycle stream:

```text
RootLifecycleStream {
  magic: u32 = LRE2
  records: array<RootLifecycleRecord>
}

RootLifecycleRecord {
  root_thread_launch_entity: u64
  occurrence: u64
  action: u32 = Start(0) | Completion(1)
  gem5_tick: u64
  delta: u64
  acknowledgement_generation: u64 (zero when control is disabled)
  decision: u32 = Continue(0) | Stay(1) | ActivateEndpoint(2) | Reject(3)
  endpoint: u64
}
```

When control is enabled, each record is preceded by a fixed-size request and
followed by an acknowledgement on the socket. The request carries the
monotonic generation, root entity, occurrence, action, gem5 tick, and delta.
The acknowledgement must echo the generation and select a decision and finite
endpoint from the projection. `Start` accepts only `Continue`; `Completion`
accepts `Stay` or `ActivateEndpoint`. A rejected decision, a foreign endpoint,
or a transition while another dispatch record is active is a protocol failure.
The device changes its active endpoint only after the acknowledgement passes
these checks. The socket protocol is an execution control boundary, not a new
Mapping or route authority; the controller may select only compiler-prepared
endpoint edges.

The device assigns a globally increasing nonzero `occurrence` when it accepts
a `Start` command and returns that value through the root-occurrence MMIO
registers. A `Completion` command must supply that same occurrence. Record
coordinates are globally ordered by gem5 tick and a device-assigned delta
within one tick. A partial record, unknown action, zero occurrence, or failed
declared output is an invocation failure.

Generated Host glue emits `Start` only after every point-specific Thread
Dispatch submission for the root invocation has been accepted. It emits
`Completion` only after the collective wait has observed the matching dynamic
occurrence complete for every point and reset those target records. Repeated
invocation of one static root receives a new occurrence. The ordinary System
result importer consumes this stream, derives the canonical Dataflow
`RootThreadStart` or `RootThreadCompletion` `EventFamilyKey`, and supplies the
typed sequence to `SimulationExecution` finalization. The raw stream has no
Artifact identity and cannot bypass the exact Request, Mapping, coordinate,
lifecycle, or `Retired` closure checks.

The current incompatible invocation-result envelope has magic `LGX3`. In
addition to the exact invocation bytes, effective runtime-input snapshot, and
Spatial boundary result, it carries the session-local entry ordinal selected
by the engine. The ordinal is validated against the immutable ordered target
table in the gem5 projection; it is not an Artifact identity, Mapping choice,
Physical Tag, or mutable cache key. Importers use the selected table entry to
recover the exact workload and execution context, then perform the ordinary
runtime-input reconstruction and result verification. Every declared session
entry must occur in accepted execution evidence, and changing result count or
entry ownership cannot bypass independent validation.

The dynamic Spatial invocation wire has one current incompatible identity,
`loom.spatial_invocation_abi.v2`. Its canonical payload is:

```text
SpatialInvocationWire {
  thread_launch_entity
  graph_launch_entity
  dense_coordinates
  values[ordinal] {
    bit_count
    pointer_target = absent | { object_ordinal, byte_offset }
    little_endian_bits
  }
  memory_objects[ordinal] { guest_address, initial_bytes }
  memory_root_bindings { logical_memory_root_entity,
                         object_ordinal, byte_offset }
  result_destinations[ordinal] { bit_count, guest_address }
}
```

Object ordinals preserve the exact invocation-local alias classes captured
from the source execution. Pointer provenance, memory-root bindings, and
memory-service requests all refer to that one object table; guest addresses
are transient transport coordinates rather than persistent storage identity.
Each object address is the canonical backing-allocation base for that dynamic
call, not the current graph view pointer. The host dispatch projection carries
that exact base as an ephemeral helper argument, snapshots the complete object
from it, and derives every memory-root and pointer-target byte offset from the
actual boundary pointer minus that base for each invocation. A repeated loop
call may therefore select a different subview without creating an overlapping
object or retaining a stale static offset. The base argument and patched wire
offsets are transient ABI state and never become Mapping or Artifact fields.
Every writable logical root is observed as `DiffFromRuntimeInput`, and the
engine returns the resulting nonconflicting byte writes through the exact guest
addresses. Runtime admission rejects missing, overlapping, out-of-range, or
type-inconsistent records. Version 1 is not retained as a compatibility path.

The result-destination table is finite and ordered but not restricted to one
entry. One selected callable may publish several scalar or fixed-width value
results through distinct caller-owned destinations; the graph boundary,
capture plan, wire table, engine result, and independent verifier must agree on
the complete ordered count and widths.

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

The production gem5 Spatial provider materializes this contract through one
invocation-local ordered sequence owner per selected channel service. It keeps
monotonic `SendSeq`, one cursor and at most one reservation per consumer
branch, atomically commits receives, and reclaims a message only after every
multicast branch acknowledges it. Its message-slot capacity is the minimum
`maxOutstanding` guarantee of the exact MessageTransfer service endpoints on
the selected route. One Dataflow message consumes one slot regardless of its
wire encoding size. An exhausted queue returns a typed backpressure/timeout
outcome and never overwrites an unacknowledged message. A graph stream's
`ClosedAfterLast` observation is only the horizon of that graph activation and
does not implicitly close the thread-level channel.

`OrderedChannelABI` is the direct in-process Runtime call boundary for that
same sequence owner. `send` exposes the candidate `SendSeq` without advancing
it on a blocked, rate-exceeding, cancelled, or invalid-lifecycle call.
`receive` returns the next message, `WouldBlock`, or the optional generation
terminal together with the consumer branch, `RecvSeq`, and generation;
acknowledgement or cancellation must match that currently live reservation.
Native compiler-generated endpoint calls and the gem5 multi-Bridge provider
use this owner rather than maintaining parallel cursor, reservation, or
acknowledgement state. The optional lifecycle described above adds no
persistent channel identity and is conforming only for adapters that bind the
finite lineage-derived rate table. The ABI does not schedule endpoint
occurrences: its caller must submit sends in Dataflow's canonical event commit
order rather than arrival-race order, and each execution adapter owns blocking
and retry around `WouldBlock`.

The native selected-program adapter is not a thread scheduler. Its dense-thread
ownership projection erases launch and wait carriers, then invokes channel
callbacks synchronously in lexical launch order on one JIT call stack. A
receive that would block cannot suspend that stack for a later producer, so a
consumer-first projection that depends on such suspension is outside this
adapter. Supporting it requires an execution owner that schedules and resumes
thread occurrences while preserving source waits, rather than a retry loop in
the channel callback.

Before JIT lowering, this adapter proves nonblocking execution for its closed
serial channel profile. Every channel launch is rank zero, names a direct
channel instance created earlier in the same block, and contains only top-level
sends or only top-level receives. For each receiver branch, all of its receive
events must already be covered by earlier complete producer launches. A mixed,
nested, dynamic-grid, or insufficiently supplied launch is typed Unsupported
before any callback can expose an unwritten receive slot.

That proof also derives the finite flat producer and per-branch consumer event
counts for each complete dynamic invocation of an exact channel-create
lineage. The generated adapter assigns that lineage one execution-local dense
slot, creates its `OrderedChannelABI` instance with the proven producer count
as bounded message capacity, and opens the first generation with those counts
before the first endpoint call. No generation can retain more messages than
its complete producer count, so that bound is exact: the host path allocates
no unbounded storage and never observes `WouldBlock`. Runtime does not infer
these counts from queue occupancy or observed execution.

Reaching the producer count finishes the producer. A branch's final
acknowledgement exposes its generation terminal and finishes that consumer;
the final branch collectively joins the generation. A later dynamic execution
of the same exact channel-create lineage reuses that ABI instance only by
resetting the joined generation and reopening the same finite rate table. The
execution-local dense slot is a lowering lookup for the existing SSA lineage,
not channel, route, Mapping, or session identity. Re-entry before join is a
typed lifecycle failure rather than an overlapping generation.

The adapter transports every rejected ABI outcome as its original typed
`OrderedChannelABIError`; it does not flatten backpressure, sequence
exhaustion, rate excess, cancellation, or lifecycle misuse into a generic
execution string. Any failure after entry cancels every generation that has
not joined, including failures while finishing endpoints or deinitializing the
native image. A fully terminal path alone joins. Each native execution owns
fresh transient lineage slots; their ABI instances never cross executions.

The Spatial Bridge binding's `maximumMessageBytes` is a separate provider wire
and staging limit. It may reject an unrepresentable invocation or message with
a typed provider outcome, but it cannot change logical capacity, split one
message across several SendSeq ordinals, or admit more outstanding messages.
Likewise, a consumer launch that arrives before its next message remains a
bounded pending launch; absence at that instant is not infeasibility. The
session retries it only after relevant channel state advances and reports a
closed wait or execution budget exhaustion distinctly.

A producer activation may emit more messages than the selected outstanding
capacity. The provider executes that activation once, encodes its messages
once, and advances a retained per-channel publication cursor as credits become
available. Each message becomes visible atomically at its own SendSeq commit;
the producer completion remains pending until every message from that
activation has been published. Re-executing the producer to recover output
credit, requiring the complete activation output to fit simultaneously, or
partially publishing one message is invalid.

After an activation retires successfully, its reserved input messages commit
before its output publication cursors advance. This ordering releases the
credits consumed by that activation and permits a capacity-one feedback
channel to make progress. A non-retired activation instead cancels every input
reservation and publishes no output.

That sequence owner is scoped to the exact System invocation and spans every
physical Bridge session used by the selected producer and consumer branches.
It is a one-shot session over that invocation: repeated launches of the same
static channel, such as consecutive application epochs of one promoted
producer/consumer pair, append to the same `SendSeq` stream behind the
messages earlier consumers acknowledged. The provider does not claim the
reusable generation profile because its per-launch message counts are runtime
stream lengths rather than lineage-derived static counts; a non-retired launch
cancels its reservations and publishes nothing.
The physical Bridges retain distinct PIO ranges, clocks, launch cursors, and
result collections, but a provider may multiplex their connections through
one execution session so that one selected channel service has one mutable
sequence state. Partitioning work by AccCore, socket, or provider process may
not partition that state or turn connection arrival order into message
ownership.

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
For a dynamically unbounded `Whole` logical interval, admission reprojects the
descriptor's concrete positive byte range through the selected Fabric
transform paths and proves exact coverage by the selected terminal regions.
Failure to fit or cover that range rejects the invocation; it does not trigger
remapping or truncation.

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

The CGRA provider must preserve an incapable-memory refusal as a typed payload
carrying the exact volatile, atomic access, atomic RMW, compare-exchange, or
fence kind and its canonical actor reference. A generic unsupported diagnostic
is not capability evidence. This payload alone does not authorize a host
fallback: Application must additionally establish real host execution, the
stopping-policy selection rule, an exact host-only Deployment path, and the
associated Evidence before recording a fallback disposition.

A graph mapped to SpatialCore execution has no implicit InstructionCore
substitute. Any explicit alternative execution must be a separately compiled,
mapped, and packaged disposition selected before runtime.

## Deployment Artifact

`docs/spec-configuration-deployment.md` is the sole owner of the Deployment
root, exact dependency closure, ConfigurationABI and
HardwareConfigurationImage relations, package projection, and finalization
rules. Runtime consumes that exact Deployment and does not restate or repair
its closure. Runtime accepts only `loom.deployment 6.0`, whose hardware
bindings require exact `loom.runtime_platform_binding 4.0` roots. Deployment
6.0 is incompatible with 5.1 because the accepted child descriptor changed;
neither version is reinterpreted as the other. The finalized Deployment must
preserve the confirmed compatibility relation among each selected AccCore,
compiler target, and target-specific binary.

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
    resource_time_cost_model: null | {
      memory_copy_setup_picoseconds
      memory_copy_byte_picoseconds
      configuration_word_picoseconds
      configuration_commit_picoseconds
    }
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

A Deployment may also contain a valid direct System configuration image. The
current RuntimePlatformBinding schema has no System-level implementation
subject and cannot bind that image through a SpatialCore interface. Such a
Deployment remains a valid semantic package, but this loader returns typed
`ProviderMismatch` before device enumeration. Executing it requires a future
independent System implementation and provider contract; assigning the image
to an arbitrary SpatialCore is invalid.

The payload-free FabricModel operational descriptor is
`loom.runtime.fabric_model` version 2.0. It implements the common portable
AXI4-Lite runtime ABI and prepared activation replacement for the exact
FabricModel HardwareImplementation identities supplied to its transient device
instance. Its shadow configuration, commit/readback, lease, and active
Deployment state model this runtime boundary only. The selected Simulation
provider remains the sole owner of computation and functional observations;
the FabricModel operational provider neither executes a graph nor manufactures
simulation evidence.

The test-oriented in-process operational descriptor is
`loom.runtime.in_process` version 3.0. Its descriptor owns the deterministic
resource-time cost model used by closure and its prepared transition operation
atomically applies exact changed configuration words, copies complete logical
memory objects between canonical Mapping targets, and changes the active
Deployment. Its direct live-target operations are fixtures for the simulated
device and are absent from the generic `RuntimeProviderInstance` and
Deployment ABI.

The payload-free mapped-RTL operational descriptor is `loom.runtime.mapped_rtl`
version 2.0 with implementation semantic identity
`loom.hardware.mapped_rtl.simulation_transport.v1`. It binds an RTL
HardwareImplementation to the same portable AXI4-Lite configuration runtime
ABI through identity, programming, memory, and completion endpoints, reports
hardware identity, and supports neither trusted immutable identity, atomic
programming multicast, nor prepared activation replacement. The mapped-RTL
Evaluation model separately binds either Verilator or VCS as the typed HDL
simulator that owns compilation, computation, and functional observations;
the operational descriptor names only their shared simulation-transport
boundary driven by `loom-system-run --mapped-rtl`.

The generated loader protocol is mechanical:

```text
validate package and Deployment closure
  -> enumerate provider devices
  -> acquire authorization and exclusive lease
  -> verify exact implementation identity under that lease
  -> quiesce and establish declared reset state
  -> reverify exact implementation identity under that lease
  -> install and verify all configuration images
  -> install static logical-memory images
  -> register host and InstructionCore entries
  -> activate
  -> execute and retire
```

Lease acquisition atomically binds the provider-owned lease to the exact
enumerated device and excludes device replacement or rebinding until release.
Identity and trusted-attestation reads accept only that live lease, never a
bare enumeration handle. The first leased verification prevents reset of a
foreign implementation; the second proves that reset preserved the selected
identity before any package state is installed. Failure at either boundary
uses the ordinary typed release, recovery, and quarantine rules.

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
fails. Quarantine and lease release are one atomic provider disposition over
the live lease and return only `Released` or `Quarantined`. An ordinary release
failure must install a process-persistent provider quarantine that owns any
unresolved lease before returning. That quarantine survives instance teardown
and excludes later acquisition through every instance of the exact descriptor;
the result diagnostic records the underlying release failure without weakening
the typed terminal state. Runtime never repairs a package, substitutes a
compatible artifact, or remaps work. A stable hand-written user launch API,
dynamic shared-object loading, firmware update protocol, remote deployment
service, and partial reconfiguration are deferred until they have concrete
independent semantics.

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
that fingerprint; it cannot supply or replace the binding-owned value. The
provider establishes that proof by hashing the executable's bytes once per
process-local gem5 facts session and reuses it only while the file's observed
identity (device, inode, mode, link count, size, modification and change
times) is unchanged; the invocation launcher still revalidates the bytes
before every attempt.

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

Because `loom.fabric 7.1` admits only the `RiscV` Architectural Contract, the
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

External Spatial-engine socket readiness enters through gem5's poll queue and
is migrated to the owning Bridge event queue before state changes. Sending a
launch returns control to gem5; the Bridge consumes a response only when its
socket becomes readable. A consumer waiting for a channel message therefore
keeps its own Bridge pending without blocking another InstructionCore, Bridge,
DMA completion, or producer launch. Blocking on a host socket inside a gem5
device event is invalid because it creates a host-induced closed wait absent
from both Dataflow and SystemMapping.

The gem5 Thread Dispatch MMIO surface carries target selection, the address and
size of the dynamic invocation wire, per-target status/error, and the assigned
occurrence ID. Dispatch snapshots all submission fields atomically before
activating an InstructionCore. The worker receives a target-qualified
completion slot rather than mutating one global dispatch state. That core
receives the exact static launch descriptor and dynamic invocation descriptor
in separate ABI registers. The zero-address, zero-size pair selects the static
runtime-input form defined above; all other dispatches require both fields. The
Spatial Bridge performs separate DMA reads and frames them only after the
required reads complete. Mutable MMIO registers, target records, DMA scratch
buffers, CPU state, socket state, and event budgets are never cached as
candidate-invariant state.

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
the narrow System root-lifecycle sequence, and typed activity summaries belong
to `SimulationExecution` 2.0. General diagnostic traces and tool payloads
remain attempt or scratch material and have no persistent runtime schema. The
raw `LRE1` stream is only the provider-to-importer carrier for the mandatory
typed root-lifecycle observations; it is not a persistent diagnostic trace.
Attempt timestamps, host/tool bindings, retries, and execution-limit outcomes
belong to the runtime owner's attempt record.
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
