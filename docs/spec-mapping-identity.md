# Mapping Identity and References

This document specifies finalized content identity and persistent
references for Mapping Artifacts and their Canonical Dataflow Program and
Fabric Hardware Description inputs.

Persistent entity references use finalized artifact identity and typed
artifact-local entity identity. Objects mechanically identified by typed
structural keys use the same artifact identity without receiving redundant
entities. Symbol spelling, paths, printer order, builder insertion order,
filesystem location, and source location are not reference authority.

## Identity Model

Each Canonical Dataflow Program, Fabric Hardware Description, and Mapping
Artifact is finalized independently. A finalized artifact has:

* one schema identity and version in `X.Y` form;
* one required content-derived artifact identity;
* one artifact-global namespace of local `EntityId` values; and
* immutable canonical semantic content.

`X` denotes a breaking or incompatible schema change. `Y` denotes a
non-breaking schema improvement. Schema identity and version participate
in artifact identity.

The conceptual persistent reference form is:

```text
PersistentEntityRef<T> = finalized artifact identity + typed local EntityId<T>
```

The reference field's schema constrains `T`. A generic subject reference
also requires a closed entity-kind tag. Arbitrary kind strings and owner
paths are not ordinary persistent references.

## Finalization

Mutable compiler construction, Dataflow canonicalization, Fabric
elaboration, and Mapping construction complete before persistent identity
is assigned.

Finalization follows one conceptual sequence:

```text
typed semantic relations without local IDs
  -> exact semantic-graph canonical labeling
  -> canonical slots
  -> artifact-global EntityId assignment
  -> canonical semantic serialization
  -> collision-checked artifact digest
  -> immutable artifact
```

This gives local IDs, serialization, and artifact identity one semantic
source. Persistent IDs must not derive from mutable addresses, symbol
order, printer order, traversal order, or insertion order.

If the artifact-local namespace cannot be represented without loss,
finalization fails. It must not truncate IDs or emit a partially finalized
artifact.

## Artifact Identity

Mapping uses the repository-wide fixed `ArtifactIdentity` contract defined
by the Artifact Identity section of
`docs/spec-full-stack-traceability.md`. Mapping artifact families provide
their typed schema descriptors and canonical semantic bytes to that Common
finalizer; this specification does not define another preimage, digest, or
external spelling.

Canonical semantic serialization includes every typed upstream artifact
reference that is part of the artifact's semantics. A TechMapping artifact
therefore includes its exact Canonical Dataflow Program and Fabric Hardware
Description references. A SpatialMapping includes its exact immutable
TechMapping predecessor reference. SystemMapping includes its exact immutable
predecessor references once its cardinality schema is closed.

Producer names, timestamps, host paths, search seeds, invocation order,
debug names, source locations, viewer layout, and provenance do not change
content identity unless their semantic projection is explicitly part of
the artifact content. Derivation lineage and execution settings belong to
manifests or Evaluation Evidence.

Optional symbols or compatibility labels cannot substitute for required
artifact identity. A mismatch is an identity failure, not permission to
reinterpret or heuristically rebind a reference.

## Artifact-Global EntityId Namespace

Every independently referenceable semantic object owned by a finalized
artifact receives one artifact-local `EntityId`. All entity kinds in that
artifact share one global local-ID namespace. Graphs, modules, record
families, and entity kinds do not create nested numeric namespaces.

Persistent `EntityId` has one unsigned 64-bit semantic range. Native
consumers may derive narrower or differently arranged dense indices, but
those indices are not persistent identities.

The confirmed structural core applies this rule to the graph, actor, FU,
`fabric.op`, Fabric encoding, and Compute Realization entities required by
TechMapping structural validation. This list does not classify every
possible Dataflow or Fabric object.

Human-readable labels may exist as metadata, but record identity is an
`EntityId`, not a string such as `family/symbol/path`.

## Typed Structural Keys

An object uniquely and mechanically derived from one identified owner uses
a typed structural key instead of receiving a redundant `EntityId`.

Confirmed forms include:

```text
actor result   = actor EntityId + result index
actor operand  = actor EntityId + operand index
graph boundary = graph EntityId + boundary kind + port index
FU port        = FU EntityId + direction + port index
software edge  = typed producer endpoint + typed consumer endpoint
```

If an owner plus typed structural key cannot distinguish semantic parallel
objects, those objects require independent entities. Printer position,
user spelling, and consumer-local array index are not valid
disambiguators.

## Cross-Artifact References

A Mapping software-entity reference consists of the exact Canonical Dataflow
Program identity and a typed `EntityId` from that artifact. An artifact-
qualified software-edge reference instead contains that exact artifact
identity and the typed producer/consumer endpoint pair. A Mapping hardware
reference consists of the exact Fabric Hardware Description identity and a
typed `EntityId` from that artifact.

References to records owned by one TechMapping artifact use that artifact's
local namespace. A SpatialMapping references its TechMapping predecessor
by exact artifact identity and addresses predecessor entities through that
identity. It does not copy predecessor IDs into its own namespace as newly
owned facts.

An FU implementation reference resolves inside the exact Fabric Hardware
Description named by the TechMapping artifact. An implementation content
digest may be used for deduplication or a pure cache key, but it is not a
persistent cross-Fabric reference and does not permit rebinding.

## Canonical Labeling Boundary

Canonical labeling considers only semantic facts, including entity and
operation kinds, typed ports and ordinals, semantic attributes, directed
typed edges, containment and instance relations, state, capability, and
artifact boundary interfaces.

It excludes symbol spelling, source and filesystem locations, debug and
provenance metadata, visual coordinates, printer order, and builder
insertion order. If an order or label changes software or hardware
behavior, that fact must first be represented as an explicit semantic
relation.

The equivalence boundary is exact typed and attributed graph isomorphism.
Canonical labeling does not prove algebraic equivalence, optimized-circuit
equivalence, or functional equivalence between distinct
microarchitectures. The particular canonical-labeling algorithm is not
part of the persistent schema.

Entities in one graph-automorphism orbit have no recoverable non-semantic
identity such as an original builder handle or an implicit numeric name. A
producer may retain a construction-object-to-`EntityId` provenance map for
diagnostics, but that map does not participate in content identity and is
not reference authority.

## Deterministic Ordering

Canonical serialization order derives from canonical semantic slots and
explicit semantic order. Lexical symbols, record labels, and printer order
are not tie breakers.

Arrays preserve semantic order where the model defines one. Unordered sets
and maps use canonical serialization order. Consumers must not infer
legality or execution order from serialized record position.

## Dense Indices, Provenance, And Caches

Persistent `EntityId`, consumer-local dense index, and provenance are
separate concepts:

* `EntityId` supports persistent cross-artifact references;
* a dense index supports one derived native model or cache and is
  disposable; and
* provenance supports source traceability and DSE attribution but is not
  execution or reference authority.

Caches bind exact artifact identities and all relevant producer semantics.
They invalidate as a unit when those inputs change. They must not export
artifact-local references, coverage, current-artifact legality conclusions,
or physical decisions into another artifact context.

## Deferred Classification

This document does not yet decide whether channels or unqualified Fabric
resources receive independent `EntityId` values. Fabric connections,
capacity objects, memory objects, tag objects, external linkage objects,
and other route or deployment entities must be classified from concrete
reference requirements by the general entity-versus-structural-key rule.

No implementation may treat this deferral as permission to use symbols,
paths, coordinates, or traversal order as persistent identity.

## Validation

Identity validation requires:

* a valid content-derived artifact identity;
* one collision-free artifact-global unsigned 64-bit `EntityId` namespace;
* a valid target kind for every typed reference;
* exact resolution of every referenced artifact and local entity;
* exact TechMapping coupling to one Canonical Dataflow Program and one
  Fabric Hardware Description;
* exact SpatialMapping coupling to one immutable TechMapping
  predecessor;
* no symbol, path, printer-order, source-location, or filesystem-path
  reference authority; and
* no persistent use of consumer-local dense indices or provenance handles.

## Non-Goals

This document does not define Mapping dialect syntax, the
canonical-labeling algorithm, native PnR index layout, deferred entity
classifications, representation-adapter records, SystemMapping references,
or bitstream identifiers.
