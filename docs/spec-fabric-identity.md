# Fabric Persistent Identity And References

This document is the single source of truth for the Mapping-visible
persistent identity and reference vocabulary owned by the Fabric Hardware
Description family. It defines which targets are independent entities, which
targets are owner-relative structures, the closed role-specific reference
unions, canonical reference ordering, and validation.

It does not redefine the hardware semantics of a PE, FU, memory, switch, FIFO,
boundary, system service, or transport pattern. The specification that owns a
resource owns its endpoint inventory, traversal relation, resource states,
use patterns, configuration fields, and refinement domains. This document
only gives those owned objects one unambiguous persistent reference.

The catalog below is the complete Mapping-visible catalog. Fabric producer
closure, all authoring-only template kinds, root finalization, and dependency
publication are owned by `docs/spec-fabric-artifact.md`. A Fabric producer cannot make a new
target visible to Mapping without extending this catalog.

## Exact Artifact Scope

Every complete cross-artifact reference uses the Common exact framing:

```text
ArtifactReference<T> =
  (exact Fabric ArtifactIdentity, typed Fabric-local target T)
```

A Mapping root that declares one exact Fabric upstream binding may encode only
`T`. The omitted digest is recovered from that binding. This compact form
does not permit rebinding, compatibility matching, or target lookup in another
Fabric artifact.

An Artifact root is always a Common `ArtifactRootReference`. It never uses this
local-target framing. `docs/spec-fabric-artifact.md` owns the exact Fabric root
dependency table and the compact dependency-ordinal projection used when a
Fabric payload references a local target inside one of those roots.

Symbols, source paths, printer positions, builder handles, native pointers,
and freeze-local PnR indices are never persistent target identity.

## Owner-Local Reference Kind Catalog

`loom.fabric 1.0` owns this complete existential local-reference catalog. The
ordinal is the Common `owner_local_kind`; the payload is the exact canonical
bytes of the named Fabric reference defined in this specification.

| Ordinal | Typed target |
| ---: | --- |
| 0 | `FabricModuleTemplateRef` |
| 1 | `FabricPeOccurrenceRef` |
| 2 | `FabricFuTemplateRef` |
| 3 | `FabricFuOccurrenceRef` |
| 4 | `FabricMemoryOccurrenceRef` |
| 5 | `FabricSwitchOccurrenceRef` |
| 6 | `FabricFifoOccurrenceRef` |
| 7 | `FabricBoundaryOccurrenceRef` |
| 8 | `HostCoreOccurrenceRef` |
| 9 | `AccCoreOccurrenceRef` |
| 10 | `SystemMemoryServiceRef` |
| 11 | `SystemServiceEndpointRef` |
| 12 | `SystemServiceTransformRef` |
| 13 | `SystemTransportResourceRef` |
| 14 | `HardwareDomainRef` |
| 15 | `ExternalBoundaryRef` |
| 16 | `SpatialCoreOccurrenceRef` |
| 17 | `InstructionCoreContextRef` |
| 18 | `InstructionContextRef` |
| 19 | `FabricFuTemplateNodeRef` |
| 20 | `FabricFuOccurrenceNodeRef` |
| 21 | `FabricFuTemplatePortRef` |
| 22 | `FabricFuNodePortRef` |
| 23 | `FabricFuOccurrencePortRef` |
| 24 | `FabricFuCapabilityTemplateRef` |
| 25 | `FabricModuleBoundaryEndpointRef` |
| 26 | `FabricTransportEndpointRef` |
| 27 | `FabricMemoryEndpointRef` |
| 28 | `FabricMemoryOperationPortRef` |
| 29 | `FabricMemoryCapabilityAlternativeRef` |
| 30 | `FabricMemoryOperationContextRef` |
| 31 | `FabricMemoryServiceRef` |
| 32 | `FabricMemoryServiceRegionRef` |
| 33 | `FabricTransferPatternRef` |
| 34 | `FabricResourceStateRef` |
| 35 | `FabricUsePatternRef` |
| 36 | `FabricSemanticConfigFieldRef` |
| 37 | `FabricPhysicalRefinementDomainRef` |
| 38 | `FabricPhysicalTraversalRef` |
| 39 | `LocalMemoryServiceRef` |
| 40 | `ManagerEndpointRef` |
| 41 | `SubordinateEndpointRef` |
| 42 | `MemoryConsistencyDomainRef` |
| 43 | `ClockDomainRef` |
| 44 | `ResetDomainRef` |

One generated Fabric declaration owns this table, the C++ enum, each typed
codec registration, strict decoder, and validator dispatch. A kind ordinal is
stable for all schema 1.x versions. New kinds append under a compatible minor
revision; reordering, deleting, repurposing, or changing a target's payload
meaning requires a major revision. Root references use their separate Common
variant and consume no local-kind ordinal.

Role-refined kinds intentionally reuse the canonical bytes of their underlying
reference while selecting a stricter validator. This is one target encoding
with several static contracts, not several identities. Nested owner unions and
endpoint components are encoded by their containing target codec and do not
receive standalone local-kind ordinals unless this catalog explicitly lists
them.

## Mapping-Visible Entity Catalog

All entity kinds share the one artifact-global unsigned 64-bit `EntityId`
namespace of the finalized Fabric Hardware Description:

```text
FabricModuleTemplate
FabricPeOccurrence
FabricFuTemplate
FabricFuOccurrence
FabricMemoryOccurrence
FabricSwitchOccurrence
FabricFifoOccurrence
FabricBoundaryOccurrence

HostCoreOccurrence
AccCoreOccurrence
SystemMemoryService
SystemServiceEndpoint
SystemServiceTransform
SystemTransportResource
HardwareDomain
ExternalBoundary
```

Their typed references are respectively:

```text
FabricModuleTemplateRef
FabricPeOccurrenceRef
FabricFuTemplateRef
FabricFuOccurrenceRef
FabricMemoryOccurrenceRef
FabricSwitchOccurrenceRef
FabricFifoOccurrenceRef
FabricBoundaryOccurrenceRef

HostCoreOccurrenceRef
AccCoreOccurrenceRef
SystemMemoryServiceRef
SystemServiceEndpointRef
SystemServiceTransformRef
SystemTransportResourceRef
HardwareDomainRef
ExternalBoundaryRef
```

Named templates are not physical occurrences. A template reference is legal
only in a field whose schema asks for that template kind. TechMapping may
reference an FU template; SpatialMapping placement always references a
concrete FU occurrence.

Canonical elaboration creates a distinct entity for every concrete physical
occurrence. Two AccCores that instantiate the same module template therefore
own distinct PE, FU, memory, switch, FIFO, and boundary occurrence entities.
Template reuse never merges their capacity, state, configuration, or physical
identity.

One AccCore has exactly one SpatialCore attachment. The structural occurrence
reference is:

```text
SpatialCoreOccurrenceRef = (AccCoreOccurrenceRef, fixed ordinal zero)
```

The existing InstructionCore reference remains a different typed domain:

```text
InstructionCoreContextRef = (AccCoreOccurrenceRef, fixed ordinal zero)
```

Equal numeric ordinals do not make these references interchangeable.

## Module Template Boundary References

A reusable module boundary endpoint is owner-relative and is not a physical
occurrence endpoint:

```text
FabricModuleBoundaryEndpointRef =
  (FabricModuleTemplateRef, Input | Output, endpoint ordinal)
```

Its type, direction, token or memory plane, width, and role are recovered from
the exact Module root. A System root that imports that Module encodes the
module dependency-table ordinal followed by this local reference. The scoped
pair is equivalent to the complete cross-artifact reference; neither the
dependency ordinal nor the local payload is meaningful alone.

Module boundary references define attachment correspondence. They are not
Spatial RouteTree endpoints or independently consumable capacity resources.

## FU-Internal Structural References

An FU is the placement and configured-graph boundary. Its inner
`fabric.op`, `fabric.mux`, and `fabric.demux` nodes cannot be placed
independently and therefore do not receive `EntityId` values.

```text
FabricFuTemplateNodeRef =
    Op(FabricFuTemplateRef, canonical node ordinal)
  | Mux(FabricFuTemplateRef, canonical node ordinal)
  | Demux(FabricFuTemplateRef, canonical node ordinal)

FabricFuOccurrenceNodeRef =
    Op(FabricFuOccurrenceRef, canonical node ordinal)
  | Mux(FabricFuOccurrenceRef, canonical node ordinal)
  | Demux(FabricFuOccurrenceRef, canonical node ordinal)
```

The exact Fabric template-to-occurrence relation derives the occurrence node
from the selected template node. Mapping cannot pair unrelated node ordinals
or infer correspondence from textual order.

Template and occurrence ports use distinct structural references:

```text
FabricFuTemplatePortRef =
  (FabricFuTemplateRef, Input | Output, port ordinal)

FabricFuNodePortRef =
  (FabricFuTemplateNodeRef, Input | Output, port ordinal)

FabricFuOccurrencePortRef =
  (FabricFuOccurrenceRef, Input | Output, port ordinal)
```

TechMapping uses template and node ports. Spatial routing uses occurrence
ports. No generic `FabricFuRef` or untyped port reference erases this boundary.

Every finalized FU occurrence has exactly one Fabric-owned definition:

```text
fuTemplate(FabricFuOccurrenceRef) -> FabricFuTemplateRef
```

Named and anonymous FU authoring forms are both projected into this canonical
definition inventory. Names and authoring-site identity are nonsemantic. Two
definition graphs that are identical after Fabric canonicalization share one
`FabricFuTemplateRef`; their physical occurrences remain distinct.

## FU Capability Template References

Each canonical FU definition owns one finite normalized inventory of
condition-relevant physical graph templates:

```text
FabricFuCapabilityTemplateRef =
  (FabricFuTemplateRef, capability-template ordinal)

FabricFuCapabilityTemplateEndpointRef =
    BoundaryPort(FabricFuTemplatePortRef)
  | NodePort(FabricFuNodePortRef)

FabricFuCapabilityTemplateRecord {
  active_nodes[] : sorted unique FabricFuTemplateNodeRef
  active_edges[] : sorted unique
      (FabricFuCapabilityTemplateEndpointRef,
       FabricFuCapabilityTemplateEndpointRef)
}
```

Every active edge must be a directed physical connection in the owning FU
definition under one legal coherent selection of its configurable muxes,
demuxes, and operation resources. The record does not copy operation schemas,
HSG membership, `hw_params`, ports, state, timing, configuration values, or a
materialized software graph. Those facts are recovered from the referenced
Fabric nodes and their owning contracts.

The finalizer rejects an empty active-node set, an out-of-definition node or
endpoint, a nonphysical edge, an incoherent selection, and duplicate records.
Distinct records may materialize the same software function only when they
select genuinely different physical nodes or topology; they remain distinct
physical TechMapping candidates.

For each FU definition, records are ordered lexicographically by their
canonical bytes and receive dense zero-based ordinals. A local reference
encodes the owning `FabricFuTemplateRef` followed by the unsigned 64-bit
big-endian ordinal. A standalone reference uses the Common exact
`ArtifactReference<FabricFuCapabilityTemplateRef>` framing. Symbols, textual
variant names, configured-function hashes, and Mapping-local encoding IDs are
not template identity.

After SpatialMapping selects a concrete FU occurrence, the exact
template-to-occurrence relation mechanically derives the occurrence node and
port corresponding to every template node and port. An occurrence is eligible
only when `fuTemplate(occurrence)` equals the owner of the selected capability
template.

## Transport And Memory Endpoints

Token transport and memory-service capability are separate planes.

```text
FabricTransportEndpointRef =
  (closed FabricTransportEndpointOwnerRef, endpoint ordinal)

FabricMemoryEndpointRef =
  (closed FabricMemoryEndpointOwnerRef, endpoint ordinal)
```

`FabricTransportEndpointOwnerRef` is the closed union of Mapping-visible
owners that expose `bits` or `bits_tag` terminals:

```text
SpatialCoreOccurrenceRef
FabricPeOccurrenceRef
FabricFuOccurrenceRef
FabricMemoryOccurrenceRef
FabricSwitchOccurrenceRef
FabricFifoOccurrenceRef
FabricBoundaryOccurrenceRef
SystemServiceEndpointRef
SystemTransportResourceRef
```

`FabricMemoryEndpointOwnerRef` is the closed union of owners that expose a
manager/requester or subordinate/provider memory-service endpoint:

```text
SpatialCoreOccurrenceRef
FabricMemoryOccurrenceRef
SystemServiceEndpointRef
```

Each owner specification supplies one canonical ordered inventory for each
plane. Direction, `bits` versus `bits_tag`, payload and tag widths, service
role, accepted operation domain, and all other endpoint facts are derived
from that inventory. A reference never copies them.

At System scope, `SystemServiceEndpointRef` is the only operation-service
endpoint owner. Its selected plane contains exactly one endpoint at ordinal
zero. `HostCoreOccurrenceRef`, `AccCoreOccurrenceRef`,
`SystemMemoryServiceRef`, `SystemServiceTransformRef`, and
`ExternalBoundaryRef` may own `fabric.system.service_endpoint` entities, but
they do not expose parallel endpoint inventories. A System message endpoint
projects to the transport plane; a System addressed-memory or fence endpoint
projects to the memory plane. `SystemTransportResourceRef` separately owns its
explicit token-port inventory and `SpatialCoreOccurrenceRef` retains the
module-boundary inventories derived from its exact imported module.

An endpoint ordinal is valid only in the inventory selected by the typed
owner and reference plane. A token endpoint cannot be reinterpreted as a
memory endpoint even when the integer ordinals happen to match.

## Memory Structural References

`fabric.mem` owns these Mapping-visible structural targets:

```text
FabricMemoryOperationPortRef =
  (FabricMemoryOccurrenceRef, operation-port ordinal)

FabricMemoryCapabilityAlternativeRef =
  (FabricMemoryOperationPortRef, capability-alternative ordinal)

FabricMemoryOperationContextRef =
  (FabricMemoryOperationPortRef, operation-context ordinal)

FabricMemoryServiceRef =
    Local(FabricMemoryOccurrenceRef)
  | System(SystemMemoryServiceRef)

FabricMemoryServiceRegionRef =
  (FabricMemoryServiceRef, service-region ordinal)
```

The `Local` variant is valid only when the memory occurrence declares its
optional Local Memory Service. Independently bindable banks are separate
`FabricMemoryOccurrence` entities rather than service-region ordinals.

Operation kind, access form, actor contract, active endpoints, and use pattern
are derived from the selected capability alternative. They are not copied
into an operation-port or context reference.

Each `FabricMemoryOperationPortRef` owns one complete embedded
`ResourceContractRecord`. Its state and use-pattern array positions are the
resource-state and use-pattern ordinals for that owner. A count, capability
alternative ordinal, memory-occurrence ordinal, or consumer-local dense index
cannot stand in for either reference. Capability alternatives store typed
`UsePatternKey` selections into that same contract and strict import projects
them as complete `FabricUsePatternRef` values. The memory-specific semantic
record at the same pattern ordinal is part of that operation port and has no
independent reference or identity.

Existing role-specific names are typed refinements, not alternate encodings:

```text
LocalMemoryServiceRef =
  FabricMemoryServiceRef::Local

ManagerEndpointRef =
  FabricMemoryEndpointRef whose owner inventory declares Manager

SubordinateEndpointRef =
  FabricMemoryEndpointRef whose owner inventory declares Subordinate

MemoryConsistencyDomainRef =
  HardwareDomainRef whose domain kind is MemoryConsistency

ClockDomainRef =
  HardwareDomainRef whose domain kind is Clock

ResetDomainRef =
  HardwareDomainRef whose domain kind is Reset
```

The refined name is selected by the consuming field's static type. Its
canonical bytes remain those of the underlying reference, and validation
checks the owner-declared role or domain kind. No copied role field, wrapper
record, or second identity is permitted.

## Instruction And Resource Structures

The PE-owned resident context remains:

```text
InstructionContextRef =
  (FabricPeOccurrenceRef, context ordinal)
```

A spatial PE has only ordinal zero. A temporal PE admits exactly the range
owned by its `num_instruction` contract.

The following reference families use a closed owner union followed by an
owner-local ordinal:

```text
FabricResourceStateRef =
  (FabricResourceStateOwnerRef, resource-state ordinal)

FabricUsePatternRef =
  (FabricUsePatternOwnerRef, use-pattern ordinal)

FabricSemanticConfigFieldRef =
  (FabricConfigurationOwnerRef, configuration-field ordinal)

FabricPhysicalRefinementDomainRef =
  (FabricRefinementOwnerRef, refinement-domain ordinal)
```

An owner-local `ResourceTransitionKey` is embedded inside its exact use-pattern
record and is recovered through `FabricUsePatternRef`. It has no standalone
persistent-reference kind or ordinal in this catalog because it cannot be
selected, routed, or used independently of that pattern.

The four role-specific owner types are distinct typed projections of this one
closed constructor catalog:

```text
FabricInventoryOwnerRef =
    ModuleTemplate(FabricModuleTemplateRef)
  | SpatialCoreOccurrence(SpatialCoreOccurrenceRef)
  | PeOccurrence(FabricPeOccurrenceRef)
  | FuTemplate(FabricFuTemplateRef)
  | FuOccurrence(FabricFuOccurrenceRef)
  | FuTemplateNode(FabricFuTemplateNodeRef)
  | FuOccurrenceNode(FabricFuOccurrenceNodeRef)
  | MemoryOccurrence(FabricMemoryOccurrenceRef)
  | MemoryOperationPort(FabricMemoryOperationPortRef)
  | MemoryService(FabricMemoryServiceRef)
  | SwitchOccurrence(FabricSwitchOccurrenceRef)
  | FifoOccurrence(FabricFifoOccurrenceRef)
  | BoundaryOccurrence(FabricBoundaryOccurrenceRef)
  | InstructionContext(InstructionContextRef)
  | InstructionCoreContext(InstructionCoreContextRef)
  | HostCoreOccurrence(HostCoreOccurrenceRef)
  | AccCoreOccurrence(AccCoreOccurrenceRef)
  | SystemServiceEndpoint(SystemServiceEndpointRef)
  | SystemServiceTransform(SystemServiceTransformRef)
  | SystemTransportResource(SystemTransportResourceRef)
  | TransferPattern(FabricTransferPatternRef)
  | HardwareDomain(HardwareDomainRef)
  | ExternalBoundary(ExternalBoundaryRef)
```

`FabricResourceStateOwnerRef`, `FabricUsePatternOwnerRef`,
`FabricConfigurationOwnerRef`, and `FabricRefinementOwnerRef` retain distinct
static types while using exactly this constructor catalog. The selected
instance must expose the corresponding canonical inventory for a child
reference to be valid. The shared constructor catalog avoids four
independently drifting copies; it is not a generic path or property reference.

Membership in an owner union does not imply that every instance has a
nonempty inventory. A reference is valid only when the selected instance
declares the indexed object.

Every `FabricTransportEndpointOwnerRef` and
`FabricMemoryEndpointOwnerRef` has one total, exact projection into
`FabricInventoryOwnerRef`. The projection preserves the complete typed owner
payload. For example, a `SpatialCoreOccurrenceRef` remains
`SpatialCoreOccurrence`, and a `SystemServiceEndpointRef` remains
`SystemServiceEndpoint`. It never collapses an owner-relative reference to its
parent `EntityId`, substitutes the logical owner of a System endpoint, or
returns absent. Hardware-domain membership and other owner-based relations use
this projection rather than an entity-ID helper.

## Directed Physical Traversals

`FabricPhysicalTraversalRef` is a closed sum:

```text
FabricPhysicalTraversalRef =
    PointConnection(
      source : FabricTransportEndpointRef,
      destination : FabricTransportEndpointRef)
  | PeSelectorTraversal(
      owner : FabricPeOccurrenceRef,
      source : FabricTransportEndpointRef,
      destination : FabricTransportEndpointRef)
  | PeRegisterFifoTraversal(
      owner : FabricPeOccurrenceRef,
      register_fifo_ordinal,
      closed path role)
  | SwitchTraversal(
      owner : FabricSwitchOccurrenceRef,
      input ordinal,
      output ordinal)
  | FifoTraversal(
      owner : FabricFifoOccurrenceRef,
      Buffered | Bypass)
  | BoundaryTraversal(
      owner : FabricBoundaryOccurrenceRef,
      output ordinal)
  | SystemTransferPatternLeg(
      owner : FabricTransferPatternRef,
      egress ordinal)
```

`FabricTransferPatternRef` is structural:

```text
FabricTransferPatternRef =
  (SystemTransportResourceRef, transfer-pattern ordinal)
```

A point connection is valid only when the fully elaborated Fabric contains
one unique directed fixed connection between the exact endpoints. Parallel
links with independent capacity or behavior must be explicit resource
entities and cannot share the same point-connection key.

A switch traversal is the already confirmed switch occurrence plus input and
output ordinals. It is not a switch row, route-table entry, configuration
value, or capacity resource. A FIFO traversal names the selected semantic
mode; only a bypass-capable occurrence admits `Bypass`. A system transfer
pattern with multiple egresses contributes one leg reference per selected
egress while the pattern's Fabric-owned atomic use vector remains shared.

FU-internal configured connectivity and `fabric.mem` internal dependency
connectivity are TechMapping realization witnesses. They are not physical
routing traversals. A module-to-AccCore spatial attachment is an exact
one-to-one endpoint correspondence, not a traversal. Neither may be inserted
into a RouteTree.

A temporal-PE register FIFO has two mutually exclusive uses for one software
edge. An explicit register-file internal realization absorbs the edge, so no
residual logical net or RouteTree exists. Otherwise, when the PE contract
exposes the register FIFO as ordinary transport, the edge remains residual
and its route selects `PeRegisterFifoTraversal` plus the associated selectors
and resource states. One edge cannot claim both forms.

Fabric is the sole owner of:

```text
FabricPhysicalTraversalRef
  -> canonical set<FabricResourceStateRef>
```

Mapping stores selected traversals. It derives resource claims from this
relation and never serializes a second traversal-to-state table.

## Canonical Wire And Ordering

Every entity reference encodes its closed entity-kind tag followed by its
unsigned 64-bit `EntityId`. Every structural reference encodes its closed
variant tag followed by its fields in the declaration order above. Entity IDs
and structural ordinals use unsigned 64-bit semantic values; native PnR index
width never changes the persistent range.

Each displayed closed declaration assigns zero-based discriminants in
declaration order. Those assigned values are immutable for every Fabric root
schema version that imports this catalog. Resource-owned nested enums, such
as a PE register-FIFO path role, use the same rule in their owning resource
schema. Generated parser, printer, byte encoder, and importer tables must all
consume that one schema declaration rather than copy the numbers.

Canonical bytes use unsigned 32-bit big-endian variant tags and unsigned
64-bit big-endian IDs and ordinals. Nested references are encoded recursively
without optional fields, padding, native layout, or duplicated owner facts.
Canonical ordering is lexicographic over those semantic fields, not over
symbol spelling, textual order, or native indices.

The MLIR textual form is a direct typed projection of the same records:
one registered reference attribute per named reference family, one closed
enum keyword for each variant, and unsigned decimal numeric fields in
declaration order. Unknown variants, extra fields, omitted fields, negative
values, and noncanonical aliases are rejected. Textual spelling is not a
second identity encoding; parse followed by canonical printing and canonical
byte emission must recover the same typed record.

Adding a new closed entity, owner, endpoint, or traversal variant requires a
Fabric schema revision. Existing variant tags and field meanings never
change. Reinterpreting an existing variant is an incompatible major schema
change.

## Validation And Failure Classification

Import resolves every persistent reference before PnR freeze. The importer
rejects:

* a foreign or wrong-kind Fabric artifact;
* an unknown entity or an entity of the wrong typed kind;
* an owner that cannot expose the selected reference family;
* an out-of-range endpoint, port, node, context, state, pattern, region, or
  traversal ordinal;
* a point connection absent from the fully elaborated Fabric;
* a traversal disallowed by the owning resource contract;
* a token-plane reference used as a memory capability or the reverse; and
* a deprecated alias or generic path/property escape.

Import also rejects any root-complete relation whose element cannot be
represented by its declared closed owner union. It never skips an
owner-relative member because that member lacks a standalone `EntityId`.

These are invalid inputs. A well-formed reference whose target cannot support
the requested software operation remains a Mapping feasibility failure, not
an identity error.

After validation, freeze may assign deterministic dense native indices and
build CSR adjacency, reverse maps, distance tables, and resource-state caches.
Those caches are removable derived data and never enter persistent identity.

## Verification Anchors

Anchor-level tests cover:

* two occurrences elaborated from one template receive distinct occurrence
  references while retaining exact template correspondence;
* named and anonymous authoring forms of one FU definition resolve to the same
  definition and capability-template references;
* capability-template record reorder is identity-neutral while one active
  node or edge change changes the selected reference;
* a switch traversal, point connection, and induced resource state remain
  three distinct typed objects;
* token and memory endpoint references cannot be interchanged;
* every token and memory endpoint owner projects exactly to its complete
  inventory-owner reference, including `SpatialCoreOccurrenceRef`;
* wrong-owner, out-of-range, foreign-artifact, and deprecated-alias inputs are
  rejected;
* parse, canonical print, byte emission, and import resolve the same exact
  reference;
* every registered owner-local kind round-trips through its owner codec, while
  unknown, reordered, or wrong-refinement kinds are rejected; and
* a module boundary endpoint resolves only against its exact Module dependency
  and cannot be treated as a concrete occurrence endpoint.

Tests do not enumerate every owner/ordinal pair, freeze data structure, printer
whitespace, or native PnR index width.

## Related Specifications

* `docs/spec-fabric-artifact.md` owns root variants, direct dependencies,
  canonicalization, finalization, and publication.
* `docs/spec-fabric-module.md` owns module connection and boundary semantics.
* `docs/spec-fabric-pe.md` and `docs/spec-fabric-pe-temporal.md` own PE
  endpoint, selector, context, and register-FIFO inventories.
* `docs/spec-fabric-fu.md` owns FU graph and port semantics.
* `docs/spec-fabric-mem.md` owns memory operation, service, endpoint, state,
  and use-pattern inventories.
* `docs/spec-fabric-switch.md`, `docs/spec-fabric-fifo.md`, and
  `docs/spec-fabric-boundary.md` own their traversal capabilities.
* `docs/spec-fabric-system-adg.md` owns system entities, services, transport
  resources, transfer patterns, and attachments.
* `docs/spec-mapping-identity.md` imports these references but does not
  redefine them.
