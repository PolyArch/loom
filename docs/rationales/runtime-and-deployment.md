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

Ordinary compiler `-o` semantics remain those of the selected Clang/GCC driver
mode. Complete Deployment requires an explicit output request at final link.
This preserves drop-in behavior and avoids turning compile-only objects into
partially linked accelerator executables.

## Why Compiler Target Binding Is Mechanical

The Fabric InstructionCore Architectural Contract owns ISA and ABI facts. A
CompilerTargetBinding mechanically selects exact compiler triple, CPU,
features, ABI, and DataLayout compatible with that contract. Target-specific
binaries are built under the binding and cannot be substituted by runtime.

Microarchitectural realization is a different Fabric projection. It may affect
gem5 timing and capacity but does not change a compatible binary's ISA/ABI.
Fabric stores neither LLVM target spelling nor a gem5 model name; the two
bindings validate against one hardware owner independently.

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

## Why There Are Two Launch Boundaries

Host/runtime dispatches a `dataflow.thread` to an AccCore InstructionCore. That
InstructionCore launches a graph on its local SpatialCore. Combining these
boundaries would either expose every Spatial launch to the host or make thread
placement implicit.

The two-level ABI mirrors the machine model and keeps thread channels/system
services distinct from graph-local value, stream, and memory ports. Generated
runtime images are projections of exact Mapping and Deployment facts, not
editable scheduling programs.

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
