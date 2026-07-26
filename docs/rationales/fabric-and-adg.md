# Fabric And ADG Construction Rationale

Normative contracts are owned by
[ADG Builder](../spec-adg-builder.md),
[Fabric Artifact](../spec-fabric-artifact.md),
[Fabric Module](../spec-fabric-module.md),
[Fabric System ADG](../spec-fabric-system-adg.md), and
[Fabric Identity](../spec-fabric-identity.md).

## Why Fabric Is The Hardware Truth

The C++ ADG Builder exists to make hardware construction usable by architecture
designers. It is not a second hardware model. Custom builders and builtin
templates both construct real Fabric IR and invoke the same finalizer; every
downstream component consumes the resulting immutable Fabric artifact.

A string node/property graph followed by printing and reparsing was rejected.
It would require a parallel type system, capability registry, connection
validator, and identity model. A thin typed C++ facade gives users convenience
while preserving Fabric as the only semantic owner.

Builtin Small, Default, and Large targets are complete hardware examples, not
size knobs or capability summaries. Their FU distribution, memory, transport,
InstructionCore, clocks, resets, and provider closure must be deterministic so
the same public API can reproduce and teach the exact hardware.

## Why Module And System Are Separate Fabric Roots

`fabric.module` is a reusable SpatialCore template. `fabric.system` is the
architecture-level multi-AccCore system that attaches module occurrences to
InstructionCores, memory/services, domains, external boundaries, and Transport
Architecture. Keeping both in Fabric preserves one hardware dialect while
separating reusable local structure from physical system occurrences.

Interconnect Implementation is a sibling refinement rather than part of
SystemMapping or the architecture root. Bandwidth, latency, ordering,
coherence, resources, and externally visible grant guarantees are architecture
facts. AXI, TileLink, CXL, packet formats, arbitration microstate, and protocol
components are implementation facts. Mapping selects use of the architecture;
it does not choose a hidden protocol interpretation.

## Why Topology Is Explicit And Fully Elaborated

SpatialCores and systems may have arbitrary directed topology. Module-level
transport is therefore explicit endpoint-to-endpoint connectivity. Fanout and
fan-in require resources with the corresponding hardware behavior; SSA reuse
cannot create a free module-level broadcast. Width conversion within one port
kind follows the declared low-bit convention, while a port-kind change requires
an explicit boundary resource.

PnR must see a fixed hardware problem. Templates and instantiations are fully
expanded before finalization; search cannot add resources, instantiate modules,
or mutate topology. Hardware DSE produces another immutable Fabric candidate
outside PnR.

Coordinates and regular grids remain optional authoring and visualization
metadata. They cannot affect Fabric identity or route legality. This prevents
Manhattan-distance assumptions from entering an arbitrary-topology mapper.

## Why Persistent References Are Owner-Relative

Independent physical occurrences such as PEs, FUs, memories, switches,
boundaries, AccCores, and transport resources need artifact-local identity.
Ports, FU-internal nodes, traversals, resource states, use patterns, and memory
regions are recoverable from those owners and use closed structural references.

A universal `{kind, path, ordinal}` was rejected because it requires a generic
path interpreter and string kind registry. Giving every leaf an EntityId was
also rejected because unrelated local changes would churn identities across
Mapping and Deployment. Typed reference domains preserve the distinction
between transport endpoints, memory endpoints, resource state, and directed
traversal.

Builder handles, symbols, source paths, and printer order are authoring aids.
They never cross the finalization boundary as identity.

## Why System Service Ports Are Explicit Endpoint Entities

A System operation-service port has independent physical meaning: it has one
plane, one direction or memory role, one capability set, and for message
transfer one exact physical carrier. Making every host core, AccCore, memory
service, transform, and external boundary own a separate endpoint inventory
would duplicate the same schema and let the logical owner become a second
physical-interface authority. A generic shared inventory record would merely
move that duplication into an untyped bag.

Fabric therefore uses one `fabric.system.service_endpoint` entity for one
physical operation-service port. The surrounding System entity owns that
endpoint by an exact typed reference, while the endpoint alone owns its
physical interface facts. Multiple ports are multiple entities, so the
endpoint's selected transport or memory inventory contains only ordinal zero.
This removes an unnecessary ordinal namespace and makes connection, domain,
Mapping, and backend references converge on one persistent object.

`SystemMemoryService` still owns storage regions and service behavior;
`SystemServiceTransform` still owns its transformation relation; an external
boundary still groups the hardware interface. Those distinctions are
essential, but none requires another endpoint schema. Their outward ports are
mechanically the service-endpoint entities that name them as owner.

SpatialCore module-boundary ports and System transport-resource ports remain
direct inventories because their ordered port topology is intrinsic to those
resources. They are not operation-service endpoints and collapsing them into
service-endpoint entities would erase a real structural distinction.

## Why Fabric Finalization Is Root-Complete

Canonical identity must reflect the complete elaborated hardware, including
connections, domains, crossings, capabilities, and dependencies. A public API
that freezes an arbitrary partial root would let callers bypass completeness
or manufacture a view that later validators trust.

The finalizer therefore validates private authoring state before identity,
canonicalizes the complete root, publishes through Common, and exposes only a
sealed imported view that can be independently reverified. Clock/reset
validation derives all membership and connections from that same root; a
shadow connection vector was rejected after it allowed omitted crossings to
pass and duplicated edges to fail.

Fabric dependencies are independently published artifacts. Root publication
does not create a transaction over them. Every dependency is resolved and
strictly imported before the root's one-object commit.

## Why One Dependency Role Is Reserved But Unavailable

An ordinal alone does not define the owner, schema, root kind, local target,
or use of an implementation input. Reusing HardwareImplementation would create
a dependency cycle, while broadening ImplementationPlatform would make it a
generic IP container. Fabric 1.0 therefore recognizes the stable wire ordinal
but rejects authoring and import before any store lookup.

This fail-closed reservation preserves format diagnosis without pretending
that an owner contract exists. A future owner must be introduced explicitly;
it cannot be inferred by a consumer.

## Why Hardware State Must Close Cleanly

Graph invocations can overlap and time-share resources only when Mapping proves
non-conflicting resource-time use. Stateful hardware must nevertheless reach
its canonical closed state before the owning invocation retires. Passing a
reset token on every launch would expose implementation detail in software;
ignoring close state would make reentrancy and pipelining unsound.

The closure rule is derived from each resource's explicit contract and exact
clock/reset domains rather than a global hidden reset policy.

## Why Instruction Architecture And Microarchitecture Are Separate

The InstructionCore is the stored-program fallback inside an AccCore, but it
is not necessarily scalar or simple. A small in-order RISC-V core, an
out-of-order core, and an RVV core can all occupy that role. Calling it a
ScalarCore would therefore encode an accidental implementation choice into the
system model.

Binary compatibility and execution behavior are different truths. The ISA,
ABI, privilege, memory-ordering, and relocation envelope decides whether a
program can execute. Pipeline widths, queues, execution-unit timing, and
thread capacity decide how it executes. Combining them would force every
microarchitecture change to create an unrelated compiler target; omitting the
microarchitecture would force Mapping, Evaluation, and gem5 bindings to invent
their own hardware description.

The closed `RiscV` architecture variant and the closed `InOrder | OutOfOrder`
realization keep those truths explicit without introducing a generic CPU
property bag. Representative profiles are ordinary values of the same schema,
not new variants. Provider names remain bindings because they identify a tool
implementation, not the hardware contract itself.
