# Fabric Artifact

This document defines the persistent artifact boundary for finalized Fabric
hardware descriptions. Fabric dialect specifications own hardware semantics;
this document owns only root variants, dependency framing, canonical bytes,
identity, finalization, and publication.

## Artifact Family And Root Variants

The first persistent family is:

```text
loom.fabric 1.0

ArtifactSchemaDescriptor {
  identity = "loom.fabric"
  version = 1.0
}

FabricRoot =
    Module
  | System
  | InterconnectImplementation
```

The three variants share one artifact family because they use the same Fabric
semantic model, reference framing, canonicalization rules, and finalization
boundary. They retain distinct typed root payloads and verifiers. The variant
tag cannot be inferred from a symbol name, path, or MLIR operation order.

`Module` is one SpatialCore hardware template. `System` is one architecture-
only multi-core Fabric system. `InterconnectImplementation` is the exact
protocol and implementation sibling for one `System`; it refines, but never
redefines, that system's Transport Architecture.

There is no persistent finalized-design wrapper, separate family per variant,
or generic hardware manifest.

## Direct Dependencies And Derived Closure

Every direct dependency is an exact Common `ArtifactRootReference`. Artifact
roots are never encoded as `ArtifactReference<T>`, a reserved owner-local kind,
or a sentinel target. When a root payload addresses a target inside a direct
dependency, it stores the dependency-table ordinal plus that owner's canonical
local target bytes. This compact form mechanically recovers the complete
`ArtifactReference<T>` and does not create another reference authority.

The dependency-role catalog for `loom.fabric 1.0` is:

```text
ImportedModule       = 0
RefinedSystem        = 1
ImplementationInput  = 2
```

A `Module` or `System` root admits only `ImportedModule`. An
`InterconnectImplementation` root requires exactly one `RefinedSystem` and
admits zero or more `ImplementationInput` dependencies. A repeated use of one
dependency repeats its table ordinal in the payload; it does not duplicate the
dependency row. Rows are sorted by `(role, ArtifactRootReference canonical
bytes)` and exact duplicate rows are invalid.

The transitive dependency closure is derived mechanically. It is never stored
as another list. Every direct dependency must already be durably published as
its own Artifact before Fabric root publication begins. The finalizer resolves
each exact root reference through Common `ArtifactStore::get`, invokes that
dependency family's strict owner importer, and recursively validates the
reachable closure. Common validates object framing, schema, and identity; the
dependency owner validates its canonical semantic bytes and root kind; Fabric
validates the dependency role, referenced local targets, use, uniqueness, and
acyclic closure. None of these layers duplicates another layer's checks.

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

1. expands every `fabric.instantiate` needed by the root;
2. resolves typed direct references;
3. strips nonsemantic names, locations, and visualization metadata;
4. constructs one private, identifier-free, structurally root-complete
   candidate;
5. verifies all dialect, resource, capability, and domain contracts on that
   complete candidate;
6. selects the lexicographically least canonical serialization among semantic
   graph isomorphisms;
7. assigns root-local entity identifiers and structural ordinals from that
   canonical form; and
8. writes one deterministic MLIR bytecode payload in canonical entity order.

The exact Fabric canonical semantic bytes passed to the Common Artifact
SHA-256 v1 finalizer are:

```text
bytes("loom.fabric.semantic.v1\0")
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
root ordinals are immutable throughout schema 1.x. Counts and lengths are
unsigned big-endian values, there is no padding or native layout, and the
decoder rejects truncation, trailing bytes, noncanonical dependency order,
duplicates, unused rows, and payload references outside the dependency table.

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
  -> resolve direct typed references
  -> get and strict-import every already-published direct dependency
  -> recursively validate the exact dependency closure
  -> expand instantiations
  -> build a private identifier-free root-complete candidate
  -> verify semantic contracts on that candidate
  -> canonicalize and assign local identities
  -> write canonical bytes
  -> compute the unpublished candidate ArtifactIdentity
  -> import canonical bytes and independently reverify
  -> Common ArtifactStore::put the Fabric root object only
  -> return the published ArtifactRootReference
```

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
* a role, root-kind, local-target, duplicate, cycle, or owner-semantic mismatch
  is structurally `Invalid` Fabric input;
* malformed dependency storage, key/preimage mismatch, or identity collision
  is Common store corruption or collision;
* canonicalization resource exhaustion is `Incomplete`;
* absent backend support remains `Unsupported`; and
* root publication or durability failure is `artifact_store_io` and returns no
  successful root reference for that attempt.

Import uses the same boundary. Common validates the root object, the Fabric
importer recursively resolves and owner-imports its exact dependencies, and a
sealed `FabricArtifactView` is produced only after the complete closure passes.
A stored root whose dependency later becomes unavailable remains a complete
stored object but cannot be imported as a complete Fabric root; import reports
the missing or corrupt dependency and never repairs, rewrites, or deletes the
root.

Semantic verification uses the structurally complete root relation, not a
caller-supplied connection shadow. For every contained module it derives the
complete combinational ready/valid dependency graph from canonical point
connections and zero-state resource contracts, then applies the cycle rule in
`docs/spec-fabric-module.md` before canonical bytes or identity are published.

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
occurrence-to-definition relation, and each FU definition's canonical
capability-template inventory. Convenience queries such as
`fuTemplate(occurrence)` and `fuCapabilityTemplates(template)` are indexes over
those complete ranges, not additional authorities. A `System` root
additionally exposes complete canonical
ranges for spatial attachments, hardware-domain declarations and membership,
system transport resources, transfer patterns, and each transport resource's
optional crossing contract.

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

FabricArtifactView::pointConnections()
  -> canonical range<FabricPointConnectionPayload>

FabricSystemRootView::spatialAttachments()
  -> canonical range<SpatialAttachmentRecordView>
FabricSystemRootView::hardwareDomains()
  -> canonical range<HardwareDomainRef>
FabricSystemRootView::hardwareDomainContract(HardwareDomainRef)
  -> exact closed HardwareDomainContractView
FabricSystemRootView::hardwareDomainMembers(HardwareDomainRef)
  -> canonical range<FabricInventoryOwnerRef>
FabricSystemRootView::transportResources()
  -> canonical range<SystemTransportResourceRef>
FabricSystemRootView::transferPatterns(SystemTransportResourceRef)
  -> canonical range<FabricTransferPatternRef>
FabricSystemRootView::clockCrossing(SystemTransportResourceRef)
  -> optional<ClockCrossingContractView>
```

`FabricSystemRootView` is a zero-copy typed refinement of the same immutable
storage. It has no independent constructor or relation lists. Refinement of a
non-`System` root is a typed wrong-root-kind error, not an empty view.

`FinalizedFabricRoot` is an owner result that contains the exact
`ArtifactRootReference`, canonical bytes, direct dependency references, and
sealed `FabricArtifactView`. It is not a new Artifact family or a second root
model. There is no public `freezeEntireFabricRoot` API.

`FabricImportBinding` proves only that a compact reference is interpreted
against the expected exact artifact and root kind. It neither proves relation
completeness nor authorizes a caller to supplement, omit, or replace root
facts. Whole-root validators consume the appropriate root view directly and
must not accept shadow topology, domain, membership, or crossing catalogs.

Malformed hardware is `Invalid`. A well-formed custom Fabric whose selected
backend lacks a provider remains a valid Fabric artifact; the backend reports
typed `Unsupported`. A published builtin target must have provider closure for
every capability it advertises and therefore fails publication when such a
provider is absent.

## Ownership Boundaries

This artifact family does not own:

* software operation semantics, owned by the canonical operation schema;
* physical sharing legality, owned by the HSG registry;
* concrete `fabric.op` capability, owned by Fabric operation contracts;
* backend availability or recipes, owned by Fabric-to-RTL providers;
* software-selected configuration, owned by Mapping and ConfigurationABI;
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
* owner-local reference kind round trips and rejection of unknown or
  repurposed kind ordinals;
* rejection before root publication when one exact dependency is missing or
  owner-invalid;
* single-object complete-or-absent root publication, independently visible
  dependencies, and deterministic retry after ambiguous publication failure;
* independent import and re-verification of canonical bytes;
* exact agreement between every complete relation range and its convenience
  queries;
* rejection of a hidden clock-domain crossing even when a caller would have
  omitted that point connection from a former shadow list;
* rejection of attempts to construct, subclass, or publicly freeze a partial
  root view;
* a valid custom Fabric with a missing backend provider reporting
  `Unsupported`; and
* a builtin target refusing publication when provider closure is incomplete.

Tests do not freeze one canonical-labeling implementation, MLIR printer
whitespace, Builder handle order, filesystem layout, or a large topology
fixture matrix.
