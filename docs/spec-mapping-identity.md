# Mapping Identity And References

This document is the identity and reference authority for persistent Mapping
artifacts. `docs/spec-mapping-artifact.md` owns profile records and root
assembly. `docs/spec-full-stack-traceability.md` owns the repository-wide
Common artifact identity contract.

Persistent references use exact finalized artifact identity plus typed local
identity or a closed structural key. Symbols, paths, coordinates, printer
order, builder insertion order, filesystem location, and source location are
never reference authority.

## Common Artifact Identity

Every finalized Mapping artifact uses the fixed Common ArtifactIdentity
SHA-256 v1 contract. Common owns the domain tag, unambiguous binary framing,
SHA-256 algorithm, 32-byte output width, and 64-character lowercase
hexadecimal external spelling. None is configurable or profile-dependent.

The Mapping schema family owns only its schema descriptor and canonical
semantic bytes. Those inputs include every semantic upstream binding and are
passed unchanged to the Common finalizer. Common does not parse Mapping MLIR,
complete defaults, reorder records, or reinterpret profile content.

`ArtifactIdentity` always contains one valid finalized SHA-256 v1 identity.
It has no empty, unbound, invalid, or pending sentinel value. Optionality
belongs to the surrounding typed schema or builder state. An all-zero digest,
if produced by SHA-256, remains an ordinary valid identity and cannot be
reserved for absence.

The artifact's own identity is not part of its semantic content. The
`InvocationManifest` owns derivation lineage and invocation-level provenance;
owner-specific attempt records own timestamp, host, tool location, retry, and
execution-limit outcomes. Reports are removable projections, and
`EvaluationEvidence` owns only normalized observations and exact references.
None of these facts enters Mapping identity.

Publication and validated reads follow the Common store contract. Identical
full preimages deduplicate. A different preimage under the same digest is a
hard identity collision, and malformed content or a key mismatch is store
corruption. Neither case permits overwrite, reconciliation, or heuristic
selection.

## UpstreamArtifactBinding

Each Mapping profile root declares every required exact upstream artifact
identity once as a typed `UpstreamArtifactBinding`. A binding consists of a
closed binding kind and one exact `ArtifactIdentity`. The root schemas in
`docs/spec-mapping-artifact.md` own which bindings exist and their fixed
order.

An internal reference to an upstream entity has the compact scoped wire form:

```text
(UpstreamArtifactBinding kind,
 closed entity kind,
 typed artifact-local EntityId<T>)
```

Its complete meaning is always:

```text
ArtifactReference<T> =
  (root-declared exact upstream ArtifactIdentity,
   typed artifact-local EntityId<T>)
```

The binding kind selects exactly one root declaration. It is not a weak
reference, compatibility class, lookup hint, or rebinding permission.
Changing a root binding changes the full meaning of every scoped reference
through that binding and therefore changes Mapping semantic content and
identity.

References to upstream structural objects carry the same binding scope plus
the closed typed owner reference and semantic ordinals that form the
structural key. They do not promote the subordinate object to an entity or
repeat the upstream digest at each use.

Binding kinds, entity kinds, directions, roles, and structural-key variants
are closed schema enums. Free-form strings, symbols, paths, generic subject
bags, and string-key escape hatches are invalid. `UpstreamArtifactBinding`
must not be abbreviated as `slot`; a Fabric instruction slot is an unrelated
hardware term.

## Dataflow-Owned Upstream References

The Canonical Dataflow family, not Mapping, owns the closed upstream entity
catalog and its IDs. `docs/spec-compiler-part-3-dfg.md` defines the exact
`GraphRef`, `ActorRef`, `RootThreadLaunchRef`, `StaticGraphLaunchRef`, and
`LogicalMemoryRootRef` forms. Mapping imports one independently verified
`CanonicalDataflowProgramView` and uses those references unchanged.

A Mapping implementation must not reconstruct Dataflow IDs from symbols,
operation positions, C++ container order, or Mapping record order. It also
must not maintain Mapping-local `GraphId`, `ActorId`, or
`LogicalMemoryRootId` authorities. A compact scoped reference in Mapping is
only the wire projection of the exact Dataflow `UpstreamArtifactBinding` plus
the typed Dataflow entity ID.

Actor results and operands, graph boundaries, software edges, memory views,
and channel branches remain Dataflow structural references. A thread
definition is recovered from a `RootThreadLaunchRef` through the canonical
launch-callee relation and has no Dataflow EntityId in schema 1.0. A
`CanonicalMemoryActorRef` is an `ActorRef` whose imported actor kind is a
canonical memory actor; it is not another ID type.

## Mapping-Local References

An independently referenceable record owned by a Mapping artifact receives a
typed local reference in that artifact's single global `EntityId` namespace.
A persistent reference from another artifact has the exact Mapping
`ArtifactIdentity` plus that typed local `EntityId`.

A reference within the same Mapping root may encode only the typed local
`EntityId` because the containing artifact identity and target kind are
already fixed by context. It cannot copy the entity into another profile's
namespace or use a local ID without the owning artifact context.

SystemMapping Spatial imports are artifact references, not entity references.
Each import table entry is a typed SpatialMapping `UpstreamArtifactBinding`
with one exact immutable SpatialMapping identity. After canonical sorting by
complete identity, a `SpatialMappingImportRef` uses the resulting table
ordinal as a compact root-local alias. That ordinal is not an `EntityId`, has
no meaning outside the root, and cannot participate in selection semantics or
canonical relation ordering. Its complete semantic target is the imported
ArtifactIdentity.

ServicePlan ordinals and owner-child ordinals follow the same rule: they are
assigned after canonical ordering within a known owner and do not create
independent identity.

## EntityId Namespace

Each finalized Canonical Dataflow Program, Fabric Hardware Description, and
Mapping artifact has its own artifact-global local entity namespace. All
entity kinds within one artifact share that namespace. Graphs, modules,
record families, and entity kinds do not create nested numeric namespaces.

The Dataflow namespace and its assignment are owned entirely by the Dataflow
finalizer. This document defines only how Mapping scopes and validates imported
Dataflow references and how Mapping assigns its own local records.

Persistent `EntityId` has one unsigned 64-bit semantic range. Namespace
exhaustion is a finalization failure before identity generation. It cannot be
handled by truncation, wrapping, partial publication, or dependence on the
native PnR index width.

The closed Mapping profile classifications are:

| Profile | Mapping records with `EntityId` | Records located structurally |
|---------|---------------------------------|------------------------------|
| TechMapping | `ComputeRealization`, `MemoryRealization` | All role-specific child relations |
| SpatialMapping | `MemoryBinding` | `ComputeBinding`, `MemoryEngineBinding`, `RouteTree`, `ResourceUse`, and owner children |
| SystemMapping | None | `ExecutionBinding`, `ServiceRealization`, `ResourceUse`, and owner children |

The SystemMapping namespace may therefore be empty. Diagnostics, viewers,
printers, and native arrays must use the confirmed typed structural keys or
local dense indices; convenience is not grounds for allocating another
persistent entity.

## Typed Structural Keys

A subordinate object that is uniquely and mechanically recoverable from an
identified owner uses a typed structural key instead of a redundant
`EntityId`. Confirmed forms include:

```text
actor result       = actor EntityId + result ordinal
actor operand      = actor EntityId + operand ordinal
graph boundary     = graph EntityId + boundary kind + port ordinal
FU port            = FU EntityId + direction + port ordinal
software edge      = typed producer endpoint + typed consumer endpoint
point connection   = source hardware endpoint + destination hardware endpoint
switch traversal   = switch occurrence EntityId + input ordinal + output ordinal
InstructionContextRef = Fabric PE occurrence ref + context ordinal
InstructionCoreContextRef = AccCore occurrence ref + fixed ordinal zero
ServicePlanElementRef = ServiceRealizationKey
                      + canonical plan ordinal
                      + typed element key
```

A point-connection key is valid only when the fully elaborated Fabric has one
unique directed connection between those endpoints. A switch-traversal key
denotes the traversal mechanically derived from the switch connectivity table;
it is not a point connection, route configuration, capacity resource, or
dynamic token occurrence.

If parallel objects have independent capacity, configuration, state, or
physical role and owner plus typed ordinals cannot distinguish them, the
upstream schema must model them as independent entities. A suffix, edge
number, printer position, or freeze enumeration index cannot disambiguate
them.

The Mapping profiles also use these confirmed structural owners:

* a ComputeBinding is keyed by its exact ComputeRealization reference;
* a MemoryEngineBinding is keyed by its exact MemoryRealization reference;
* a RouteTree is keyed by its `SpatialLogicalNetKey`;
* a ResourceUse is keyed by its complete typed owner, use site, activation,
  parameter, and sharing-assignment tuple;
* a ThreadExecutionBinding is keyed by `RootThreadLaunchRef`;
* a GraphExecutionBinding is keyed by its parent thread-binding key and
  `StaticGraphLaunchRef`; and
* a ServiceRealization is keyed by `SystemServiceObligationKey`;
* an InstructionCore context is derived from the selected AccCore and fixed
  ordinal zero; and
* a ServicePlan element is identified by the exact
  `ServicePlanElementRef` above and does not receive an EntityId.

An `EventFamilyKey` is also a typed structural key: it combines an existing
static semantic event reference with a projection of Dataflow-owned logical
coordinates and launch parameters. It is not an entity or dynamic occurrence
identity. Runtime may add a transient occurrence handle, but that handle
never enters Mapping identity or persistent references.

## Canonical Labeling

Finalization constructs an exact semantic graph before assigning Mapping
`EntityId` values. Canonical labeling uses only semantic facts, including
entity and operation kinds, typed ports and ordinals, semantic attributes,
directed typed relations, containment, selected capability, state, and
artifact boundaries.

It excludes symbol spelling, source and filesystem locations, debug and
provenance metadata, visual coordinates, printer order, and builder insertion
order. If an order or label changes behavior, that distinction must first be
represented by an explicit semantic relation.

The equivalence boundary is exact typed and attributed graph isomorphism. It
does not prove algebraic equivalence, optimized-circuit equivalence, or
functional equivalence between distinct microarchitectures. The particular
canonical-labeling algorithm is not part of the persistent schema.

Entities in one graph-automorphism orbit have no recoverable non-semantic
builder identity. A finalizer may return a construction-object-to-`EntityId`
provenance map for diagnostics and lineage, but that map is not semantic
content or reference authority.

Canonical labeling, local-ID assignment, canonical serialization, and
ArtifactIdentity share this one semantic source. Persistent IDs must never be
assigned from mutable addresses, traversal order, textual order, or names.

## Ordering And Local Indices

Canonical serialization order derives from canonical semantic slots,
complete typed semantic keys, and explicitly meaningful array ordinals.
Lexical symbols and record labels are not tie breakers. Consumers must not
infer legality, execution order, or identity from serialized record position.

Persistent `EntityId`, an owner-local canonical ordinal, a consumer-local
dense index, and provenance are distinct:

* `EntityId` supports persistent references to independently meaningful
  objects;
* an owner-local ordinal compactly references a canonically ordered child;
* a dense index supports one derived native model or cache; and
* provenance records source relationships without becoming execution or
  reference authority.

Native PnR may use a build-selected `PnrIndex` and maintain checked
`EntityId`-to-index maps. Native indices are disposable and never persistent.
When both native widths can represent the same input, they must produce the
same Mapping semantic content and identity.

Caches bind exact artifact identities and all relevant producer semantics.
They invalidate as a unit when those inputs change and cannot export local
references, coverage, legality conclusions, or physical decisions into a
different artifact context.

## Classification Boundary

The table in this document closes identity classification for every Mapping
record family in the complete `loom.mapping 2.0` schema. This document does
not make additional classifications for upstream Dataflow channels,
unqualified Fabric resources, deployment objects, or other objects whose
independent reference requirements are owned elsewhere.

A future Mapping object receives an `EntityId` only if it is independently
meaningful, cannot be uniquely recovered from an identified owner and closed
typed structural key, and must be referenced by another semantic record. Such
an addition requires an explicit Mapping schema upgrade. It cannot be
introduced as a generic ID-bearing placeholder.

## Validation

Identity and reference validation requires:

* a valid fixed-width Common ArtifactIdentity for every required binding;
* the exact binding catalog and order required by the profile root;
* one collision-free artifact-global unsigned 64-bit Mapping `EntityId`
  namespace, which may be empty;
* correct target kind and owner for every typed local or scoped reference;
* exact resolution of every upstream artifact and entity;
* exact resolution of every owner plus structural-key reference;
* canonical import and owner-child ordinal assignment;
* exact predecessor coupling required by the profile; and
* no symbol, path, location, textual order, provenance handle, or native
  dense-index reference authority.

Identity validation does not establish Mapping profile legality. The profile
verifiers in `docs/spec-mapping-verification.md` consume these valid reference
semantics and establish closure.
