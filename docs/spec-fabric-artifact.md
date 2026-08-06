# Fabric Artifact

This document defines the persistent artifact boundary for finalized Fabric
hardware descriptions. Fabric dialect specifications own hardware semantics;
this document owns only root variants, dependency framing, canonical bytes,
identity, finalization, and publication.

## Artifact Family And Root Variants

The current persistent family is:

```text
loom.fabric 3.0

ArtifactSchemaDescriptor {
  identity = "loom.fabric"
  version = 3.0
}

FabricRoot =
    Module
  | System
  | InterconnectImplementation
```

Version 3.0 is one atomic breaking boundary. It closes every memory-plane
System spatial attachment over its exact System service endpoint. A Fabric
owner accepts and emits only the exact `loom.fabric 3.0` descriptor; there is
no 2.x compatibility owner, fallback importer, in-place upgrade, or alternate
identity path. A 2.x System does not contain enough information to recover the
binding when several compatible System service endpoints exist. The
RootRelative memory index-width relation introduced in 2.0 remains part of
this boundary.

The three variants share one artifact family because they use the same Fabric
semantic model, reference framing, canonicalization rules, and finalization
boundary. They retain distinct typed root payloads and verifiers. The variant
tag cannot be inferred from a symbol name, path, or MLIR operation order.

`Module` is one SpatialCore hardware template. `System` is one architecture-
only multi-core Fabric system. `InterconnectImplementation` is the exact
protocol and implementation sibling for one `System`; it refines, but never
redefines, that system's Transport Architecture.

Each root variant has one typed MLIR root operation and no substitute:

```text
Module                     -> fabric.module
System                     -> fabric.system
InterconnectImplementation -> fabric.interconnect_implementation
```

The authoring `builtin.module` is only a symbol and dependency-resolution
container. It is never the persistent root payload. A root-kind tag paired
with a different operation, a generic module payload, or a caller-declared
root kind is structurally invalid. If a running Loom build has not registered
the exact typed operation, canonical codec, importer, and semantic verifier for
a known root variant, that variant fails closed as
`Unsupported(FabricRootProviderUnavailable)`. It cannot fall back to another
root operation. This is distinct from
`fabric_artifact_owner_contract_unavailable`, which means the schema itself has
no enabled owner contract, as for `ImplementationInput` in schema 3.x.

There is no persistent finalized-design wrapper, separate family per variant,
or generic hardware manifest.

## Direct Dependencies And Derived Closure

Every direct dependency is an exact Common `ArtifactRootReference`. Artifact
roots are never encoded as `ArtifactReference<T>`, a reserved owner-local kind,
or a sentinel target. When a root payload addresses a target inside a direct
dependency, it stores the dependency-table ordinal plus that owner's canonical
local target bytes. This compact form mechanically recovers the complete
`ArtifactReference<T>` and does not create another reference authority.

The dependency-role catalog remains unchanged in `loom.fabric 3.0`:

```text
ImportedModule       = 0
RefinedSystem        = 1
ImplementationInput  = 2  // reserved-unavailable in schema 3.x
```

A `Module` root admits no direct dependency: every authoring template use is
fully elaborated into the canonical Module and no `fabric.instantiate`
survives. A `System` root admits only `ImportedModule`. An
`InterconnectImplementation` root admits exactly one `RefinedSystem` and no
other direct dependency in schema 3.x. `ImplementationInput = 2` retains its
wire ordinal so schema 3.x never renumbers a published discriminant, but it has
no accepted artifact family, schema version, root kind, owner-local target
kind, or dependency-use contract in schema 3.x. It is therefore not an enabled
dependency role and cannot appear in a canonical Fabric root.

The enabled schema-3.0 dependency contracts are exact:

```text
ImportedModule:
  owner schema = loom.fabric 3.0
  required root = Module

RefinedSystem:
  owner schema = loom.fabric 3.0
  required root = System
```

A `loom.fabric 2.x` Module has no 3.0 dependency contract and is rejected
rather than republished under a new identity without exact finalization.
Likewise, a `RefinedSystem` dependency cannot cross a Fabric schema version or
name a Module or InterconnectImplementation root. A later compatible Fabric
minor version must explicitly publish its own dependency-contract table; role
ordinals alone never imply cross-version admission.

An authoring draft, encoder input, or imported envelope containing an
`ImplementationInput` row fails structurally with
`fabric_artifact_owner_contract_unavailable` before the referenced object is
looked up or imported. The ordinal is not permission to accept an arbitrary
`ArtifactRootReference`, protocol name, blob, path, or property bag. A later
Fabric schema version may enable the role only by defining a finite table of
exact accepted contracts. Each table entry must fix the dependency owner's
`ArtifactSchemaDescriptor`, required root kind, any admitted owner-local target
kind and canonical codec, and one closed dependency-use validator. The role
ordinal alone never owns those facts.

Dependency use is determined by the static field that contains the compact
reference. There is no generic dependency-use tag, path, or property bag. The
closed schema 3.x field catalog is:

```text
System AccCore spatial_core
  role: ImportedModule
  target: FabricModuleTemplateRef

System spatial_attachment module endpoint
  role: ImportedModule
  target: FabricModuleBoundaryEndpointRef

System hardware_domain SpatialCoreSlot member
  role: ImportedModule
  target: FabricModuleDomainSlotRef selected through the member's AccCore

InterconnectImplementation refined_system
  role: RefinedSystem
  target: root only

InterconnectImplementation EndpointRefinement architecture target
  role: RefinedSystem
  target: FabricTransportEndpointRef

InterconnectImplementation ResourceStateRefinement architecture target
  role: RefinedSystem
  target: FabricResourceStateRef

InterconnectImplementation TransferPatternRefinement architecture target
  role: RefinedSystem
  target: FabricTransferPatternRef

InterconnectImplementation ConfigurationRefinement architecture target
  role: RefinedSystem
  target: FabricSemanticConfigFieldRef
```

Each targetful field encodes exactly `u64be(dependency_table_ordinal)` followed
by the dependency owner's canonical local-target bytes. The root-only field
encodes only the ordinal. The decoder obtains the required role and target type
from the field schema, checks the ordinal and role, invokes the dependency
owner's strict local-reference decoder, validates the target against the exact
imported root view, and requires decode/re-encode equality. A row is used when
at least one valid field, including the root-only `refined_system` field,
references its ordinal. After walking the complete typed root, every direct
dependency row must have been used and every external use must resolve to one
row. Duplicate uses are legal; duplicate or unused rows are not.

A repeated use of one enabled dependency repeats its table ordinal in the
payload; it does not duplicate the dependency row. Rows are sorted by `(role,
ArtifactRootReference canonical bytes)` and exact duplicate rows are invalid.

An occurrence-qualified Module slot or internal target in a System does not
add another direct dependency use. Its `SpatialCoreOccurrenceRef` selects the
AccCore's already-declared `ImportedModule` row, and validation resolves the
slot or local target through that exact dependency. The occurrence-qualified
reference never repeats a dependency ordinal, Module identity, or dependency
row.

The transitive dependency closure is derived mechanically. It is never stored
as another list. Every enabled direct dependency must already be durably
published as its own Artifact before Fabric root publication begins. After
rejecting unavailable roles, the finalizer resolves each exact root reference
through Common `ArtifactStore::get`, invokes that dependency family's strict
owner importer, and recursively validates the reachable closure. Common
validates object framing, schema, and identity; the dependency owner validates
its canonical semantic bytes and root kind; Fabric validates the dependency
role, referenced local targets, use, uniqueness, and acyclic closure. None of
these layers duplicates another layer's checks.

Missing, foreign, wrong-kind, duplicate, cyclic, or unused direct dependencies
make finalization fail before the root `put`. A dependency publication that is
still in flight is either absent or complete to the finalizer; an absent read
fails and may be retried rather than waiting on temporary store state.

Builder handles, helper names, preset names, source locations, visualization
metadata, file paths, and printer positions are provenance or projections and
do not enter dependency identity.

## Canonical Semantic Relation And Bytes

Artifact identity is computed from one canonical semantic relation, not from
authoring order or raw source text. Canonicalization:

1. validates and composes every Module-instance domain-slot binding while
   expanding every `fabric.instantiate` needed by the root;
2. resolves typed direct references;
3. strips nonsemantic names, locations, and visualization metadata;
4. constructs one private, identifier-free, structurally root-complete
   candidate;
5. verifies all dialect, resource, capability, and domain contracts on that
   complete candidate;
6. selects the lexicographically least canonical serialization among semantic
   graph isomorphisms;
7. derives and deduplicates canonical FU definitions and Memory Operation
   Engine definitions, then establishes each concrete occurrence relation;
8. assigns root-local entity identifiers and structural ordinals from that
   canonical form; and
9. writes one deterministic MLIR bytecode payload in canonical entity order.

For a Module root, that relation includes the canonical symbolic Clock/Reset
slot inventory and every Module boundary or internal-owner assignment. For a
System root, it includes the canonical occurrence-slot domain memberships.
The effective domain of an occurrence-qualified internal target is derived
from those two relations and is not serialized as a second expanded member
list.

A nested Module's authoring-only slot binding is consumed during expansion.
Its child boundary and slot inventory disappear, and its fresh internal owners
are assigned directly to the selected slots of the enclosing Module. The
binding itself is not a dependency, local reference, canonical root field, or
separate identity input. Consequently, equivalent inline and instantiated
forms converge on the same complete flat relation.

The exact Fabric canonical semantic bytes passed to the Common Artifact
SHA-256 v1 finalizer are:

```text
bytes("loom.fabric.semantic.v3\0")
|| u32be(root_variant)
|| u64be(direct_dependency_count)
|| repeated direct_dependency_count times {
     u32be(dependency_role)
     || u32be(length(dependency.schema.identity))
     || bytes(dependency.schema.identity)
     || u32be(dependency.schema.version.major)
     || u32be(dependency.schema.version.minor)
     || dependency.ArtifactIdentity[32]
   }
|| u64be(canonical_mlir_bytecode_length)
|| canonical_mlir_bytecode
```

Root variant ordinals are `Module = 0`, `System = 1`, and
`InterconnectImplementation = 2`. The dependency-role ordinals above and the
root ordinals are immutable throughout schema 3.x. Counts and lengths are
unsigned big-endian values, there is no padding or native layout, and the
decoder rejects truncation, trailing bytes, noncanonical dependency order,
duplicates, unused rows, and payload references outside the dependency table.
Decoding the known ordinal `ImplementationInput = 2` does not make it legal;
schema validation rejects it as reserved-unavailable before dependency lookup.

The MLIR payload encodes each external root use as a `u64be` dependency-table
ordinal followed by the referenced owner's canonical local target bytes when
a target is required. It does not repeat an ArtifactIdentity. The Fabric
semantic envelope does not repeat its own schema descriptor because the Common
identity preimage already owns that framing.

The specification fixes the canonical result, not a canonical-labeling
algorithm. Individualization-refinement, orbit pruning, or another exact
algorithm is an implementation choice. A resource bound may produce typed
`Incomplete`; it may not publish a noncanonical identity.

For one supported Loom codebase and resolved configuration, semantically
equal hardware has identical canonical bytes and identity. Different canonical
bytes under one digest are an internal error. A semantic hardware difference
must change canonical bytes and identity.

## Finalization And Publication

Finalization is failure-atomic:

```text
authoring draft
  -> close scopes and helpers
  -> require exactly one typed root operation for the selected root variant
  -> resolve direct typed references
  -> validate root/role cardinality and reject unavailable dependency roles
  -> get and strict-import every already-published direct dependency
  -> recursively validate the exact dependency closure
  -> decode every typed external use and reject missing or unused rows
  -> validate and compose every Module-instance domain-slot binding
  -> expand instantiations
  -> reject every residual fabric.instantiate
  -> build a private identifier-free root-complete candidate
  -> verify semantic contracts on that candidate
  -> derive canonical FU and Memory Operation Engine definitions
  -> canonicalize and assign local identities and occurrence relations
  -> write canonical bytes
  -> compute the unpublished candidate ArtifactIdentity
  -> import canonical bytes and independently reverify
  -> Common ArtifactStore::put the Fabric root object only
  -> return the published ArtifactRootReference
```

Envelope encode/decode, dependency-role preflight, and Common identity
calculation are necessary prefixes of this pipeline, not a reduced
finalization mode. An implementation that has not decoded all typed dependency
uses, rejected unused rows, expanded the complete instance graph, built the
root-complete semantic view, performed semantic canonicalization, and strictly
reimported the result cannot return `FinalizedFabricRoot` or claim artifact
success. It must return the typed unavailable, invalid, incomplete, or store
failure owned by the first unsatisfied stage.

Fabric failure atomicity means one root object is complete or absent; it does
not mean that the root and its dependency graph become visible in one
transaction. Dependencies are independently valid, immutable, shareable
Artifacts and may be visible before this root or remain unreachable after a
failed root attempt. Fabric defines no multi-object transaction, publication
manifest, rollback, or dependency cleanup protocol.

All semantic checks and dependency imports complete before the root's atomic
namespace insertion. A store reader may observe the complete root after that
insertion but before the publishing call receives its durability
acknowledgement; this is safe because the exact dependency closure was already
published and validated. The finalizer returns no successful root reference
until `put` reports success. If a crash or `artifact_store_io` occurs after
insertion, recovery retries the same deterministic root publication; the store
contains either no root or the complete expected root, never a partial root.

Failure classes retain their existing owners:

* an absent exact dependency is a missing-artifact failure;
* a reserved-unavailable dependency role is
  `fabric_artifact_owner_contract_unavailable` and is rejected before lookup;
* a known root variant whose typed root provider is not registered is
  `Unsupported(FabricRootProviderUnavailable)`;
* a wrong root operation, role, root kind, local target, dependency use,
  duplicate or unused row, residual instantiation, dependency cycle, or
  owner-semantic mismatch is structurally `Invalid` Fabric input; a cycle in
  the root-complete unconditional handshake graph is specifically
  `Invalid(UnconditionalCombinationalHandshakeCycle)`;
* malformed dependency storage, key/preimage mismatch, or identity collision
  is Common store corruption or collision;
* canonicalization resource exhaustion is `Incomplete`;
* absent backend support remains `Unsupported`; and
* root publication or durability failure is `artifact_store_io` and returns no
  successful root reference for that attempt.

Import uses the same boundary. Common validates the root object, the Fabric
importer first validates root/role cardinality and rejects unavailable roles,
then recursively resolves and owner-imports its exact enabled dependencies. A
sealed `FabricArtifactView` is produced only after the complete closure passes.
A stored root whose dependency later becomes unavailable remains a complete
stored object but cannot be imported as a complete Fabric root; import reports
the missing or corrupt dependency and never repairs, rewrites, or deletes the
root.

Semantic verification uses the structurally complete root relation, not a
caller-supplied connection shadow. It validates all resource-local handshake
alternatives and rejects a cycle only when every arc in that cycle is
unconditional in every legal configured view. It must not union mutually
exclusive switch traversals, FIFO modes, tags, or physical refinements into a
fabricated active graph. The complete graph for one selected configuration is
owned by SpatialMapping or SystemMapping verification under
`docs/spec-mapping-verification.md`.

Cross-instance structure is never validated one template at a time. The
private candidate is built only after the complete reachable instantiation
graph has been expanded, and any residual `fabric.instantiate` is invalid.
Consequently point connections and resource-local dependencies that close a
cycle across former instance boundaries are visible to both the unconditional
Fabric structural gate and the later selected Mapping gate.

## Immutable Root-Complete Views

The canonical Fabric root is the only authority for its structural relations.
The C++ import API exposes those facts through one sealed, immutable
`FabricArtifactView`. That view exists only after canonical IDs and bytes have
been assigned, either inside the owner finalizer for independent reimport or
through strict import of an existing complete root and dependency closure. It
has no public constructor, cannot be subclassed, and cannot be assembled from
caller-provided relation fragments.

Pre-canonical semantic verification uses an owner-internal immutable view of
the complete identifier-free candidate. It is not `FabricArtifactView`, has no
persistent-reference API, and cannot escape finalization. The finalizer itself
derives it from one complete authoring root after closing helpers, resolving
references, and expanding instantiations; callers cannot manufacture it or
assert completeness.

Tests that need invalid input submit an invalid whole authoring root to the
real finalizer or corrupt a complete serialized root for import rejection.
They do not call a public freeze hook, subclass `FabricArtifactView`, or supply
partial relation answers.

For every root kind, the view exposes canonical complete ranges for all
relations owned by that root, including its entities, owner inventories, token
and memory endpoints, directed point connections, and dependency-derived
occurrence facts. It also exposes the FU definition inventory, the exact FU
occurrence-to-definition relation, each FU definition's canonical
capability-template inventory, the Memory Operation Engine definition
inventory, and the exact memory-occurrence-to-engine-definition relation.
Convenience queries such as `fuTemplate(occurrence)`,
`fuCapabilityTemplates(template)`, and `memoryEngineTemplate(occurrence)` are
indexes over those complete ranges, not additional authorities. For a Module
root, the view exposes the canonical symbolic Clock/Reset slot inventory and
the complete boundary/internal-owner assignment relation. It also exposes the
complete canonical token-plane
resource-attachment relation. Each `FabricModuleBoundaryEndpointRef` directly
connected to a resource maps to the one occurrence-local
`FabricTransportEndpointRef` reached by that signature input or producing that
signature result in the finalized Module body. Unused boundaries and direct
boundary-to-boundary passthroughs have no row; the view never invents a
resource endpoint for them. This relation is derived from the canonical Module
SSA graph and is not another serialized catalog. A Module boundary reference
remains an attachment correspondence rather than a transport endpoint,
traversal, or capacity owner.

The same Module view separately exposes the complete canonical token-plane
boundary-passthrough relation. Each row contains one exact input
`FabricModuleBoundaryEndpointRef` and one exact output
`FabricModuleBoundaryEndpointRef` connected directly by the canonical Module
SSA graph. Rows follow output signature ordinal; original signature ordinals
are retained even when memory-plane endpoints create holes in the token-plane
inventory. Inputs and outputs are each unique in the relation. A boundary
present in an attachment row cannot also appear in a passthrough row. The
relation contains no resource endpoint, traversal, capacity, EntityId, or
serialized payload and does not change Fabric Artifact identity.

Memory-plane Module boundaries remain in the typed memory endpoint model and
never appear in this token-plane relation. A `System` root additionally
exposes complete canonical ranges for spatial attachments, hardware-domain
declarations and direct or occurrence-slot membership, system transport
resources, transfer patterns, and each transport resource's optional crossing
contract. The same view derives the complete effective-domain relation for
each occurrence-qualified Module boundary and internal target. That derived
range is an index over the exact Module assignments and System slot
memberships, not another serialized catalog.

All range elements are exact typed Fabric references or immutable views of
root-owned records. Ranges use canonical order and contain no duplicates.
Convenience queries such as `entityKind`, `hardwareDomainKind`,
`hasPointConnection`, membership lookup, or crossing lookup are derived
indexes over these same ranges. An implementation may cache the indexes, but a
query and a scan of the authoritative range must always agree. A query is
never an independent callback authority.

The critical C++ boundary is conceptually:

```text
finalizeFabricRoot(complete authoring root) -> FinalizedFabricRoot
importEntireFabricRoot(ArtifactRootReference,
                       canonical bytes,
                       exact dependencies) -> FabricArtifactView
requireSystemRoot(FabricArtifactView) -> FabricSystemRootView
requireModuleRoot(FabricArtifactView) -> FabricModuleRootView

FabricModuleRootView::domainSlots()
  -> canonical range<FabricModuleDomainSlotRef>
FabricModuleRootView::domainAssignments()
  -> canonical range<ModuleDomainAssignmentView>

FabricArtifactView::pointConnections()
  -> canonical range<FabricPointConnectionPayload>
FabricArtifactView::handshakeOwners()
  -> canonical range<FabricHandshakeOwner>
compileHandshakeOwnerModel(FabricArtifactView,
                           FabricHandshakeOwner)
  -> sealed HandshakeOwnerModel
resolveSelectedHandshake(
    HandshakeOwnerModel,
    exact typed owner selection)
  -> canonical range<HandshakeActivationFragmentOrdinal>
deriveUnconditionalHandshakeDependencyArcs(FabricArtifactView)
  -> canonical range<HandshakeDependencyArc>
FabricArtifactView::memoryOperationPorts(FabricMemoryOccurrenceRef)
  -> canonical range<FabricMemoryOperationPortRef>
FabricArtifactView::memoryOperationPort(FabricMemoryOperationPortRef)
  -> exact MemoryOperationPortView
FabricArtifactView::memoryCapabilityAlternative(
    FabricMemoryCapabilityAlternativeRef)
  -> exact MemoryCapabilityAlternativeView

FabricSystemRootView::spatialAttachments()
  -> canonical range<SpatialAttachmentRecordView>
FabricSystemRootView::hardwareDomains()
  -> canonical range<HardwareDomainRef>
FabricSystemRootView::hardwareDomainContract(HardwareDomainRef)
  -> exact closed HardwareDomainContractView
FabricSystemRootView::hardwareDomainMembers(HardwareDomainRef)
  -> canonical range<FabricHardwareDomainMemberRef>
FabricSystemRootView::effectiveHardwareDomain(
    Direct(FabricClockResetDirectOwnerRef)
      | SpatialCore(SpatialCorePhysicalDomainTargetRef),
    Clock | Reset)
  -> exact HardwareDomainRef
FabricSystemRootView::transportResources()
  -> canonical range<SystemTransportResourceRef>
FabricSystemRootView::transferPatterns(SystemTransportResourceRef)
  -> canonical range<FabricTransferPatternRef>
FabricSystemRootView::clockCrossing(SystemTransportResourceRef)
  -> optional<ClockCrossingContractView>
```

`FabricHandshakeOwner` is a sealed view-only union of existing
occurrence-level Fabric owners and fixed point connections.
`HandshakeActivationFragmentOrdinal` is an owner-model-local index. Neither
receives a persistent reference kind or identity.
`HandshakeOwnerModel` exposes ordered boundary signal bindings, owner-local
dependency junctions, unique potential arcs, and typed activation fragments.
Its internal junctions are not transport endpoints and cannot be routed,
serialized, or referenced by Mapping records.

The selection resolver accepts only the complete typed choice owned by the
resource: an occurrence-local traversal group, an FU occurrence plus one exact
capability row, a memory occurrence plus one exact operation plan, or a system
transfer pattern, including every selected physical refinement declared by
that owner. Missing, foreign, stale, definition-only, or contradictory choices
are rejected. A registered traversal break may validly resolve to no
combinational fragment; it is not an endpoint-projection error.

This owner model is the only API by which Mapping, simulation, or RTL obtain
resource-local ready/valid dependencies. Consumers cannot reconstruct them
from operation names, latency, a generic "stateful" flag, independent
`UsePattern` interpretation, or caller-provided arc lists. The compact
owner-local graph may introduce private dependency junctions, but for every
selection it must preserve the exact boundary dependency reachability implied
by the normative resource equations.

Before persistent IDs exist, the finalizer invokes the same owner compiler over
its private root-complete semantic view and obtains identifier-free owner
models. It derives the unconditional boundary relation from those models and
the declared local configuration domains. The post-ID API above is the
canonical reference projection of the same equations, not a second algorithm.
Private keys and junctions cannot escape finalization or be compared with
persistent references.

`FabricSystemRootView` is a zero-copy typed refinement of the same immutable
storage. It has no independent constructor or relation lists. Refinement of a
non-`System` root is a typed wrong-root-kind error, not an empty view.

`MemoryOperationPortView` exposes the exact endpoint inventory, validated
embedded `ResourceContract`, canonical ResourceState and UsePattern references,
memory-specific operation-pattern semantics, and capability-alternative
references. `MemoryCapabilityAlternativeView`
exposes the OperationSchema-owned actor-contract domain, canonical service-role
bindings, optional parameterized access domain, and typed admissible
use-pattern references. These views are the only C++ projection of the
operation-port persistent records in `docs/spec-fabric-mem.md`;
counts, raw MLIR attributes, and consumer-owned geometry tables are not
alternative APIs.

`FinalizedFabricRoot` is an owner result that contains the exact
`ArtifactRootReference`, canonical bytes, direct dependency references, and
sealed `FabricArtifactView`. It is not a new Artifact family or a second root
model. There is no public `freezeEntireFabricRoot` API.

`FabricImportBinding` proves only that a compact reference is interpreted
against the expected exact artifact and root kind. It neither proves relation
completeness nor authorizes a caller to supplement, omit, or replace root
facts. Whole-root validators consume the appropriate root view directly and
must not accept shadow topology, domain, membership, or crossing catalogs.

Malformed hardware is `Invalid`. Any semantically complete Fabric, whether
custom or expanded from a builtin template, remains a valid Fabric artifact
when an RTL or EDA provider is absent. The consumer requesting that backend
reports typed `Unsupported`. Official backend-ready qualification of a builtin
requires complete provider closure, but qualification is not Fabric
publication and cannot become a second Fabric identity.

## Ownership Boundaries

This artifact family does not own:

* software operation semantics, owned by the canonical operation schema;
* physical sharing legality, owned by the HSG registry;
* concrete `fabric.op` capability, owned by Fabric operation contracts;
* backend availability or recipes, owned by Fabric-to-RTL providers;
* software-selected actor and refinement facts, owned by Mapping;
* physical configuration encoding, owned by ConfigurationABI;
* implementation realization and tool outputs, owned by
  HardwareImplementation; or
* timing, power, area, and other measured or predicted observations, owned by
  EvaluationEvidence.

Fabric itself remains the authority for resource state, capacity, use-pattern,
transition-timing, and progress capability contracts. HardwareImplementation
and Evaluation may realize or observe those contracts, but neither may
redefine them.

These owners may reference a Fabric artifact. They may not copy its topology,
capability, identity catalog, or canonicalization rules.

## Anchor Verification

Anchor tests cover:

* equivalent regular and irregular Fabrics with different source names and
  construction order producing identical bytes and identity;
* one semantic topology, capability, state, timing, or domain change producing
  a different identity;
* wrong-kind, foreign, duplicate, cyclic, and missing direct references;
* fixed byte vectors for every root variant, zero and multiple dependencies,
  dependency-table target uses, and malformed count or length framing;
* the `loom.fabric.semantic.v3` envelope, Module slot/assignment relation,
  System occurrence-slot membership, and memory-plane spatial service binding
  changing identity exactly when their semantic content changes;
* strict field-owned dependency-use decoding, target re-encoding, and
  rejection of missing, wrong-role, wrong-target, and unused dependency rows;
* rejection of any envelope-only or dependency-preflight path that attempts to
  construct or return `FinalizedFabricRoot` before the complete finalization
  pipeline and strict reimport have succeeded;
* preservation of the `ImplementationInput = 2` wire ordinal together with
  authoring, encoding, finalization, and import rejection before object lookup;
* owner-local reference kind round trips and rejection of unknown or
  repurposed kind ordinals;
* rejection before root publication when one exact dependency is missing or
  owner-invalid;
* single-object complete-or-absent root publication, independently visible
  dependencies, and deterministic retry after ambiguous publication failure;
* independent import and re-verification of canonical bytes;
* exact agreement between every complete relation range and its convenience
  queries;
* exact agreement between Module assignments plus System slot membership and
  the derived occurrence-qualified effective-domain range, with no serialized
  expanded member catalog;
* rejection of a hidden clock-domain crossing even when a caller would have
  omitted that point connection from a former shadow list;
* rejection of attempts to construct, subclass, or publicly freeze a partial
  root view;
* full expansion of a nested instance whose cross-instance connections expose
  an invalid unconditional cycle, plus rejection of every residual
  `fabric.instantiate`;
* rejection of a root-kind-2 generic module payload and typed
  `FabricRootProviderUnavailable` when the exact
  `fabric.interconnect_implementation` owner provider is absent;
* a valid custom Fabric with a missing backend provider reporting
  `Unsupported`; and
* a builtin target publishing with complete semantic capability while a later
  backend request reports typed `Unsupported` for missing provider closure.

Tests do not freeze one canonical-labeling implementation, MLIR printer
whitespace, Builder handle order, filesystem layout, or a large topology
fixture matrix.
