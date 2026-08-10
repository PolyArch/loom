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
`loom.hardware_implementation 3.0` descriptor; accepting an older major would
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
