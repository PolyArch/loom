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

Symbols, source paths, printer positions, builder handles, native pointers,
and freeze-local PnR indices are never persistent target identity.

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
AccCoreOccurrenceRef
SystemServiceEndpointRef
SystemTransportResourceRef
ExternalBoundaryRef
```

`FabricMemoryEndpointOwnerRef` is the closed union of owners that expose a
manager/requester or subordinate/provider memory-service endpoint:

```text
SpatialCoreOccurrenceRef
FabricMemoryOccurrenceRef
AccCoreOccurrenceRef
SystemMemoryServiceRef
SystemServiceEndpointRef
SystemServiceTransformRef
ExternalBoundaryRef
```

Each owner specification supplies one canonical ordered inventory for each
plane. Direction, `bits` versus `bits_tag`, payload and tag widths, service
role, accepted operation domain, and all other endpoint facts are derived
from that inventory. A reference never copies them.

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

The four role-specific owner types are distinct typed projections of this one
closed constructor catalog:

```text
FabricInventoryOwnerRef =
    ModuleTemplate(FabricModuleTemplateRef)
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
* a switch traversal, point connection, and induced resource state remain
  three distinct typed objects;
* token and memory endpoint references cannot be interchanged;
* wrong-owner, out-of-range, foreign-artifact, and deprecated-alias inputs are
  rejected; and
* parse, canonical print, byte emission, and import resolve the same exact
  reference.

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
