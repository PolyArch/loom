# Runtime And Deployment Rationale

Normative contracts are owned by
[Executable Closure](../spec-executable-closure.md),
[Configuration And Deployment](../spec-configuration-deployment.md),
[Runtime ABI](../spec-runtime-abi.md), and
[Implementation Platform](../spec-implementation-platform.md).

## Why ConfigurationABI Is Separate

Fabric owns hardware capability and Mapping owns selected semantic and
physical choices. Neither should own where a provider places those choices in
a bitstream. ConfigurationABI defines programming units, field encodings,
visibility, load, and activation for one exact Fabric implementation contract.

The HardwareConfigurationImage is a framed raw payload bound to exact Mapping,
Fabric closure, ABI, and programming unit. It is a final encoding, never a
second Mapping record. Equivalent raw encodings are canonicalized by the ABI;
unknown fields or hidden backend defaults are invalid.

Instruction-context residency banks storage rather than semantics. Requiring
one encoding for every residency of a physical field lets a Temporal PE share
the field's single Fabric relation and hardware decoder across contexts while
retaining independently placed configuration bits. Context-specific codecs
would add no behavior and would force duplicate operation implementations or
another runtime decode authority.

## Why Deployment Is The Executable Closure

Mapping alone does not contain host code, InstructionCore binaries, static
memory images, hardware implementations, runtime platform bindings, or
configuration images. Deployment binds the exact selected leaves needed to
execute the program. A separate Executable artifact would duplicate that
closure and create another definition of readiness.

Deployment stores direct references and derives transitive architecture,
Mapping, implementation, and platform facts from their owners. It does not copy
topology or configuration fields. A content-addressed directory package is a
projection of Deployment, not an installation or distribution system.

Compatible schema evolution does not make an exact Artifact reference
version-polymorphic. Deployment consumes the current exact
`loom.hardware_implementation 4.1` descriptor; accepting an older major would
require an explicit additional schema alternative rather than reinterpretation
or a compatibility alias.

Ordinary compiler `-o` semantics remain those of the selected Clang/GCC driver
mode. Complete Deployment requires an explicit output request at final link.
This preserves drop-in behavior and avoids turning compile-only objects into
partially linked accelerator executables.

## Why Compiler Target Binding Is Mechanical

The Fabric InstructionCore Architectural Contract owns ISA and ABI facts. A
CompilerTargetBinding mechanically selects exact compiler triple, CPU,
features, ABI, and DataLayout compatible with that contract. Target-specific
binaries are built under the binding and cannot be substituted by runtime.

One binding may serve several same-kind InstructionCore occurrences because
microarchitecture is not code-generation identity. That reuse is justified by
equality of the complete canonical Architectural Contract, not by a friendly
CPU name or a digest alone. HostCore and InstructionCore bindings remain
separate because they occupy different executable ownership domains even when
their RISC-V contracts match.

The target triple, feature list, and DataLayout are derived by the pinned LLVM
provider rather than copied from a relocatable payload. The payload's module
DataLayout remains authoritative for that module; final link later proves
structural compatibility without rewriting either owner. This keeps Fabric,
compiler policy, LLVM code generation, and source-module identity as four
non-overlapping facts.

Microarchitectural realization is a different Fabric projection. It may affect
gem5 timing and capacity but does not change a compatible binary's ISA/ABI.
Fabric stores neither LLVM target spelling nor a gem5 model name; the two
bindings validate against one hardware owner independently.

## Why Static Memory Starts From LLVM Bytes

A source-level constant table does not imply a dedicated ROM. LLVM owns the
initializer and DataLayout; Dataflow owns the logical memory root; Mapping owns
whether that root uses local SRAM or a manager-backed external service. A
single mechanical projection from the final linked LLVM module therefore feeds
pre-Mapping simulation and the later Deployment leaf. Re-parsing constants in
each simulator or backend would create competing byte-layout authorities.

The logical root alone is not a sufficient Deployment key. A reusable thread
definition can be launched twice while each launch binds the same memory formal
to a different linked global. `RootedGraphLaunchRef` is already the Dataflow-
owned invocation context needed to distinguish those two source relations, so
the static-memory leaf pairs it with the existing logical root. Restricting the
program to one global would discard valid source semantics, while a separate
binding artifact would duplicate Dataflow and SystemMapping ownership.

Only complete relocation-free initializers may become local preload images.
Everything else remains runtime-provided and can still use external memory.
This fail-closed split keeps pointer relocation and system-memory behavior with
their existing owners while allowing ordinary read-only tables to exploit
SpatialCore memory without introducing a separate `fabric.rom` primitive.

## Why Runtime Cannot Remap

SystemMapping has already selected AccCores, SpatialMappings, routes, tags,
contexts, resources, and configuration. Runtime establishes authorization,
leases, isolation state, loader order, and failure recovery for those exact
resources. Letting admission choose a different core or route would hide a new
mapper behind execution.

The runtime verifies exact identities before programming. It cannot repair a
package, swap a binary, select similar hardware, or synthesize a missing
mapping. Failed or ambiguous programming enters failure recovery and quarantine
rather than pretending the old or partial configuration is usable.

Admission reuses the complete Dataflow-owned `EventFamilyKey`. Boundary and
channel activity uses the existing produced or consumed transfer variants;
memory and fence service occupancy uses the rooted actor-transition variant
whose commit is the OperationSchema-owned issue event. Narrowing this child to
transfer terminals would either lose a legal Mapping use or manufacture a fake
terminal and a second event authority.

SpatialMapping also names graph-local endpoint events. One endpoint may be
produced or consumed by several mutually exclusive actor transition cases, so
choosing one case loses behavior while requiring all cases can never complete.
Dataflow therefore projects the endpoint to a canonical alternative set of its
existing event keys. Admission preserves an original causal conjunction as an
`AllOf` of those per-point `AnyOf` sets. This derives the needed runtime index
without adding an endpoint-event identity to Deployment.

Admission capacity uses one derived catalog rather than either rebuilding an
index in every consumer or repeating complete physical keys in every case.
The former would let Deployment, Runtime, and simulation disagree on dense
ordinals; the latter would duplicate the same physical cell throughout the
payload. The shared SystemMapping closure projection instead sorts exact
occurrence-qualified Fabric cell keys once, folds selected static route claims
into baseline occupancy, and lets activation members reference that local
catalog. Fabric remains the sole capacity and UsePattern authority because
strict import rederives every copied value and claim.

## Why There Are Two Launch Boundaries

Host/runtime dispatches a `dataflow.thread` to an AccCore InstructionCore. That
InstructionCore launches a graph on its local SpatialCore. Combining these
boundaries would either expose every Spatial launch to the host or make thread
placement implicit.

The two-level ABI mirrors the machine model and keeps thread channels/system
services distinct from graph-local value, stream, and memory ports. Generated
runtime images are projections of exact Mapping and Deployment facts, not
editable scheduling programs.

## Why Binary Entries Are Root-Launch Bound

A thread definition owns reusable behavior, but a root launch owns the exact
static context in which that behavior executes: parameters, channel and memory
bindings, Mapping relations, and Deployment selection. Two root launches may
call the same definition and share one machine-code entry, or compilation may
specialize one launch into a different entry. A definition-only key cannot
express both cases without recreating launch context beside it.

InstructionCoreBinary therefore declares a many-to-one relation from existing
`RootThreadLaunchRef` keys to binary-local entry ordinals. This is compiled
capability and provenance, not target selection. Deployment chooses one
declared binary entry for each mapped root/target case, while runtime creates
transient occurrences and passes coordinates and parameters to that entry.
No source symbol, operation position, thread-definition ID, or dynamic
instance table becomes a competing program authority.

## Why System Transport Has Three Layers

Software channels describe producer-consumer behavior. Fabric Transport
Architecture describes logical hardware capacity, latency, ordering,
coherence, eligibility, and grant guarantees. Interconnect Implementation
describes protocol and microarchitecture. Keeping these layers distinct lets
the same SystemMapping target different protocol implementations that refine
the same architecture without rewriting software semantics.

Gem5 executes dynamic arbitration and interconnect state from the exact
implementation binding. Architecture guarantees and Mapping-visible controls
remain Fabric truth. Runtime counters and reorder state are transient and do
not become Mapping fields.

A provider process or socket boundary is not a transport-semantic boundary.
One mapped channel can connect graph launches on several AccCores, so its
ordered message state must be shared across the corresponding physical Spatial
Bridges while each Bridge retains its own PIO and completion session. This
keeps the Dataflow-owned correspondence and SystemMapping-owned service intact
instead of making process placement an accidental message-routing rule.

Service outstanding capacity counts messages, whereas bridge framing limits
bytes. Conflating them makes a large payload consume several logical credits
or lets a small payload create an unbounded queue. Runtime therefore derives
message slots from the selected endpoint rate contracts and treats framing
bytes only as a provider admission bound.

Output credit can stall retirement after computation has already finished.
Retaining the encoded messages and a publication cursor makes that stall an
ordinary bounded transport wait. Replaying the computation would duplicate
memory effects, while requiring all activation messages to fit at once would
deadlock whenever a legal stream is longer than the channel capacity.

A reusable channel service needs a terminal boundary without turning control
state into data. Runtime therefore binds a generation to one complete logical
channel invocation, derives its flat endpoint counts from the existing
Dataflow correspondence, and exposes EOS only as an ABI lifecycle result.
Per-activation epochs would break legal rate conversion, while a sentinel
message would contaminate the payload type and multicast sequence. Resetting
only after join or cancellation lets a physical service be reused without
making its generation ordinal a new channel, route, or Mapping identity.

The serial host oracle can prove a complete finite count for each dynamic use
of one exact channel-create lineage, so it reuses one bounded ABI instance by
joining and resetting between those uses. The gem5 Spatial provider instead
learns stream lengths while executing the invocation; it keeps one one-shot
sequence and does not claim the reusable profile. These are distinct execution
profiles over the same Dataflow correspondence, not alternative channel
identities.

A resource-time decision similarly cannot infer a Mapping from runtime queue
state. Keeping a finite graph with PnR lets its independent verifier remain the
only legality authority; Runtime supplies only an explicit exact child choice
at a canonical completion frontier. Requiring the full child endpoint avoids
turning graph container order into an undocumented policy when several child
states share one trigger. Recording an explicit stay preserves the equally
important decision not to transition.

Decision derivation remains separate from activation commit. A caller may
validate and replay completion-frontier choices without touching a provider,
while the combined commit path requires an independently verified edge and an
exact loaded parent Deployment. Before execution, Runtime imports the finite
edge catalog and the provider copies each unique selectable child executable
and activation image into a reusable transient prepared handle. The entry is
prepared only when a later completion frontier has a verified return edge to
its Mapping endpoint. The safe-point operation then performs
one failure-atomic handle switch under the existing lease, without Artifact
I/O or image transfer. Setup and runtime control costs retain their ordinary
execution timing owners; they are not relabeled as PnR reprogramming or
live-state migration. Runtime does not derive another Mapping, reprogram
hardware, or reinterpret a graph vector as policy. This keeps the graph and
Mapping verifier as the legality owner while making the admitted no-live-state
activation primitive executable at its completion edge.

That replacement is deliberately not a general continuation mechanism. It
does not snapshot scheduler state, carry tokens or DynamicWork, generate a
completion callback, or launch remaining roots on its own. The bounded session
records and replays the canonical completion choices and joins the entry
Mapping's root inventory. Provider execution, DynamicWork cancellation, host
residual work, and process termination keep their existing authorities.

An invocation which retains a completed Application build result may hand this
pair to Runtime through one composition point: the exact finalized Deployment
is loaded first, and its optional compiler-built transition graph is prepared
against that same lease. This composition owns no new schema and does not
reconcile the build result's reporting projections. Without a graph, the
Application retains an ordinary loaded Deployment and no resource-time
selector is invented.

The Application activation manifest and package persist the exact join needed
to reproduce that composition. Source StructuredProgram workload/runtime roots
remain distinct from the Deployment-owned System workload/runtime roots. The
source-backed replay set joins the former to the selected canonical Dataflow;
the activation pair names its Deployment owner directly. Package import
replays both domains, the runtime/oracle Evidence, every transition endpoint,
and the copied workspace closure. This avoids reconstructing an entry or input
from command-line values and avoids treating package presence as execution
evidence.

Memory capability failure is narrower than generic execution failure. The CGRA
provider reports the exact unsupported contract kind and canonical actor, so
Application can distinguish a fixed incapable Fabric from an adapter failure.
That typed refusal is necessary evidence for a possible source-backed host
execution, but it is not sufficient to select or deploy one. Application must
still prove the host execution and preserve it as a distinct runtime path; it
must never relabel an atomic operation as Plain, invent accelerator cycles, or
treat a diagnostic string as capability evidence.

The same separation applies to control progress. A channel consumer may be the
first mapped thread submitted, so Host glue must preserve launch handles and
defer its join to the source wait. Each target has independent transient
dispatch state; one global busy bit would serialize otherwise independent
InstructionCores. Once a Spatial launch is sent, gem5 socket readiness is an
external event, not a reason to block the simulator thread. Using gem5's poll
queue lets a pending consumer coexist with the producer whose publication will
make it ready, while gem5 remains the sole simulated-time authority.
Targets sharing one InstructionCore are reconsidered when its running worker
reports completion, rather than polling the same busy state every simulated
cycle.

## Why Technology Corners Stay Platform-Local

An ImplementationPlatform already owns the immutable technology or device
universe used to build hardware. A technology corner is one local selection of
that owner's timing-model inputs, so creating a sibling Technology Artifact
would split platform identity and payload closure. Letting Evaluation assign a
bare integer or tuple would instead make a consumer the local-reference owner.

The platform therefore owns the typed corner catalog and its owner-local
reference codec. Evaluation conditions and EDA adapters carry that exact
reference and validate it through the platform; voltage, temperature, clock
requirements, activity, extraction choices, and tool effort remain in their
existing Evaluation or implementation-flow owners. Runtime device discovery
is separate again: compatibility with a design-time platform does not prove a
particular installed device exists.

## Why First-Version Runtime Is Single-Tenant

Protection, virtualization, migration, preemption, and resource remapping add
policy and state beyond the first complete execution path. Empty tenant IDs or
placeholder protection domains would not provide isolation and would constrain
later design. The first version therefore has one implicit tenant and no
persistent multi-tenant schema.
### Runtime Witness Provenance

Queue-level blocked state is execution evidence, not a second Mapping legality
owner. A runtime witness names the exact QueueKey, allocation unit, ingress,
tag, reservation, physical action, and causal release that it observed. A
child hardware replay must independently re-import and verify the resulting
Fabric and Mapping; an absent cycle remains `ProofNotEstablished` rather than
being promoted from timeout or finite successful replay.
