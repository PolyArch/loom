# Mapping Identity And References

This document is the identity and reference authority for persistent Mapping
artifacts and System service obligation keys. `docs/spec-mapping-artifact.md`
owns profile records and root assembly. `docs/spec-fabric-identity.md` owns
the closed Fabric-local entity and structural-reference catalog.
`docs/spec-compiler-part-3-dfg.md` owns the closed Dataflow-local entity and
structural-reference catalog.
`docs/spec-fabric-system-adg.md` owns Canonical Service Schema member and leg
semantics.
`docs/spec-full-stack-traceability.md` owns the repository-wide Common
artifact identity contract.

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
launch-callee relation and has no Dataflow EntityId under Canonical Dataflow
3.0. A
`CanonicalMemoryActorRef` is an `ActorRef` whose imported actor kind is a
canonical memory actor; it is not another ID type.

## Fabric-Owned Upstream References

`docs/spec-fabric-identity.md` is the sole catalog and framing authority for
Fabric-local persistent targets. It defines the Mapping-visible entity kinds,
template-versus-occurrence references, FU capability-template references, FU
structural nodes and ports, token
and memory endpoints, memory-operation structures, instruction contexts,
resource states, use patterns, semantic configuration fields, physical
refinement domains, and directed physical traversal variants.

Mapping imports those types unchanged. It cannot add an unqualified
`FabricResourceRef`, generic owner path, string kind, symbol, printer
position, builder handle, or native PnR index as an escape hatch. It also
cannot promote a Fabric structural object to a Mapping-owned entity.

Every complete reference still has the Common exact form:

```text
ArtifactReference<T> =
  (exact Fabric ArtifactIdentity,
   typed Fabric-local target T)
```

A Mapping root that already declares the exact Fabric
`UpstreamArtifactBinding` may encode only `T`. Wrong root kind, foreign
artifact, wrong entity kind, wrong owner, or an out-of-range structural
ordinal is invalid. A well-formed target that lacks a compatible capability is
a Mapping feasibility failure rather than an identity error.

TechMapping's selected FU capability uses the exact local shape:

```text
FabricFuCapabilityTemplateRef =
  (FabricFuTemplateRef, capability-template ordinal)
```

The containing TechMapping root supplies the exact Fabric identity. Mapping
does not wrap this target in an `EncodingId`, copy the template record, or
derive identity from a configured software graph. SpatialMapping resolves the
selected template through the Fabric-owned occurrence-to-definition relation.

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

A Spatial `MemoryBinding` whose target is `BoundaryProxy` uses its existing
Mapping-local `EntityId` as the proxy identity. There is no `BoundaryProxyId`,
Fabric proxy entity, endpoint-as-service alias, or System-owned replacement
identity.

## Typed Structural Keys

A subordinate object that is uniquely and mechanically recoverable from an
identified owner uses a typed structural key instead of a redundant
`EntityId`. Dataflow imports use the exact `RootedGraphLaunchRef`,
`ActorTokenResultRef`, `ActorTokenOperandRef`, `GraphIngressTokenRef`,
`GraphEgressTokenRef`, producer/sink terminal, memory view/exposure, service
member, and static transfer-event forms owned by
`docs/spec-compiler-part-3-dfg.md`. Mapping does not restate their variants.
Fabric and Mapping-owned examples include:

```text
FU template port   = FabricFuTemplateRef + direction + port ordinal
FU occurrence port = FabricFuOccurrenceRef + direction + port ordinal
memory engine template port = FabricMemoryEngineTemplateRef
                            + operation-port ordinal
memory engine template alternative = memory engine template port
                                   + capability-alternative ordinal
memory engine template endpoint = FabricMemoryEngineTemplateRef
                                + token-endpoint ordinal
memory engine template internal connection = FabricMemoryEngineTemplateRef
                                           + source endpoint
                                           + sink endpoint
point connection   = source hardware endpoint + destination hardware endpoint
switch traversal   = switch occurrence EntityId + input ordinal + output ordinal
InstructionContextRef = Fabric PE occurrence ref + context ordinal
InstructionCore context = exact Fabric-owned `InstructionCoreContextRef`
ServicePlanElementRef = ServiceRealizationKey
                      + canonical plan ordinal
                      + typed element key
```

A point-connection key is valid only when the fully elaborated Fabric has one
unique directed connection between those endpoints. A switch-traversal key
denotes the traversal mechanically derived from the switch connectivity table;
it is not a point connection, route configuration, capacity resource, or
dynamic token occurrence.

A Memory Operation Engine template internal connection is valid only when its
exact directed endpoint pair occurs in the owning template relation. It is not
a concrete point connection and has no independent connection ordinal. After
SpatialMapping selects a memory occurrence whose Fabric-owned template relation
matches exactly, the template-relative port, alternative, endpoint, and
internal-connection references project mechanically to occurrence-relative
Fabric references.

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
* a GraphExecutionBinding is keyed by the Dataflow-owned
  `RootedGraphLaunchRef`;
* a ServiceRealization is keyed by `SystemServiceObligationKey`;
* a ServicePlan selection row is keyed by its exact owner-relative
  `ServicePlanSelectionKey`;
* an InstructionCore context is derived from the selected AccCore and fixed
  ordinal zero; and
* a ServicePlan element is identified by the exact
  `ServicePlanElementRef` above and does not receive an EntityId.

`EventFamilyKey` is the Dataflow-owned typed alias of one exact
`StaticTransferEventRef`; it has no Mapping-owned fields and no static-event
`EntityId`. The exact program mechanically derives its
`EventLogicalProjection` as the canonical ordered coordinate and launch-
parameter input schema. That projection is not serialized inside the key and
cannot be selected or rewritten by Mapping. Mapping imports the Dataflow-owned
key, projection, canonical comparison order, and wire contract unchanged.
Runtime may add concrete projection values and a transient occurrence handle,
but neither enters Mapping identity, persistent references, channel message
order, or Physical Tag assignment.

### Execution Context Key

`ExecutionContextKey` is the closed structural key for the hardware execution
context selected by the existing execution bindings:

```text
ExecutionContextKey =
    InstructionExecutionContextKey {
      AccCoreOccurrenceRef
    }
  | SpatialExecutionContextKey {
      AccCoreOccurrenceRef
      exact SpatialMapping semantic target
    }
```

The Spatial variant is derived by evaluating `B_graph` at the Dataflow-owned
graph logical point and `B_thread` at that point's Dataflow-owned parent thread
point. Its SpatialMapping must cover the applicable graph and its target
SpatialCore parent must belong to the selected AccCore. The Instruction
variant is used only where no graph execution target exists. A graph-backed
service member always uses the Spatial variant.

The exact SpatialMapping semantic target is its complete immutable
ArtifactIdentity. Within a finalized SystemMapping, the existing canonical
import table encodes that target through `SpatialMappingImportRef`; ordering
and comparison first resolve the alias back to the complete identity. A
mutable flat System PnR candidate has no Spatial variant until its reopened
Spatial decisions have passed independent verification and received that
identity. It cannot use a provisional digest, native handle, reopen ordinal,
or candidate generation as a substitute semantic target.

The canonical semantic variant ordinals are `Instruction = 0` and
`Spatial = 1`. Canonical comparison is lexicographic over the variant ordinal,
the exact canonical Fabric-local bytes of `AccCoreOccurrenceRef`, and, for the
Spatial variant, the complete canonical ArtifactIdentity bytes of the
SpatialMapping target. Canonical semantic bytes use this framing:

```text
u32be(variant_ordinal)
|| u64be(length(acc_core_ref_bytes))
|| acc_core_ref_bytes
|| if Spatial {
     u64be(length(spatial_mapping_identity_bytes))
     || spatial_mapping_identity_bytes
   }
```

The containing root supplies the exact Fabric identity for the local AccCore
reference and admits only the canonical Fabric reference encoding. A
SystemMapping may serialize the Spatial target through its canonical
`SpatialMappingImportRef`, but its finalizer must resolve that alias before
semantic comparison and must reproduce the semantic bytes above. Deployment
reuses the same finalized key and import-table encoding. System PnR `H` does
not serialize or index `ExecutionContextKey`; its invocation-local service
compatibility is factorized by existing Dataflow and Fabric endpoint
references. Native dense indices, import ordinals, authoring order, and PnR
candidate handles never enter semantic comparison.

The key stores no thread or graph predicate, service endpoint, attachment,
capability, route, or resource selection. Those facts remain derived from the
Dataflow member, the applicable execution binding or bindings, Fabric, and,
for the Spatial variant, the exact imported SpatialMapping. Reusing one
SpatialMapping on two AccCore occurrences therefore yields two distinct
execution contexts even when both occurrences import the same Module template.

## System Service Obligation Keys

A SystemMapping root and a System MappingConstraintSet root each bind one
exact Canonical Dataflow Program and one canonical non-empty root-thread-launch
set. Together they define the workload scope from which all reachable
software service obligations are derived:

```text
SystemWorkloadScope =
  (exact Canonical Dataflow ArtifactIdentity,
   canonical root-thread-launch set)
```

The scope is owned once by the root and is not repeated in each obligation
key. The only System service obligation variants are:

```text
SystemServiceObligationKey =
    TransferObligationFamilyKey
  | OperationServiceObligationFamilyKey

TransferObligationFamilyKey =
  CanonicalProducerTerminalRef

OperationServiceObligationFamilyKey =
    LogicalMemoryRootOrViewRef
  | FenceActorFamilyRef
```

Each key stores only its minimal derivation anchor. For a transfer key, the
exact Dataflow program, workload scope, and producer derive one canonical
sorted unique non-empty sink set. The sink set is not serialized in the key.
This is the static obligation universe, not a claim that every sink has a
`source_map` preimage at every producer point. Mapping 3.0 derives per-point
message route applicability without extending the obligation key.
For a logical-memory owner, the exact Dataflow program derives the complete
canonical addressed-memory member set, including applicable load, store,
atomic RMW, and compare-exchange actors, plus the separate complete memory
exposure set. An exposure is a capability boundary rather than a service
member and has no service leg. For a fence owner, the exact actor derives its
fence contract and all reachable rooted launch contexts. Neither member sets,
exposure sets, nor contracts are copied into the key.

`ServiceMemberRef` is imported from the Dataflow-owned closed union.
`MessageTransfer` is the singleton member of a transfer obligation. An
addressed-memory or fence member carries the exact contextual actor rooted at
one graph launch. The obligation kind statically determines which member
variant is legal.

Every service leg uses the member-relative structural key:

```text
CanonicalServiceLegKey =
  (SystemServiceObligationKey,
   ServiceMemberRef,
   schema-local leg ordinal)
```

The exact Dataflow family must own the closed producer-terminal,
sink-terminal, memory-root-or-view, fence-actor-family, and service-member
reference variants. The Canonical Service Schema owns each member's local leg
count, direction, payload, completion rule, and ordinal meaning. Mapping owns
neither catalog and cannot flatten all legs into an independently interpreted
global ordinal. Memory exposures instead use the Dataflow-owned
`MemoryExposureRef` and are consumed only by service target bindings or
Mapping exposure entries.

Finalization derives the complete obligation and leg universes from the root
scope. Missing any derived `ServiceRealization`, addressed-operation member,
member leg, or required memory-exposure binding is a Mapping completeness
failure. A malformed, foreign, wrong-kind, wrong-owner, or out-of-range
reference is invalid. A well-formed obligation for which no compatible Fabric
target exists is proven infeasible for that Mapping invocation rather than an
identity error.

### Service Plan Selection Key

A ServiceRealization selects its owner-local plans through rows with this
closed structural key:

```text
ServicePlanSelectionAnchor =
    Member(ServiceMemberRef)
  | Exposure(MemoryExposureRef)

ServicePlanSelectionKey =
  (ServicePlanSelectionAnchor, ExecutionContextKey)
```

The key is relative to its owning `SystemServiceObligationKey`. A transfer
obligation admits exactly the singleton `MessageTransfer` member anchor; its
owner key supplies the exact producer terminal. An operation-service
obligation admits exactly its Dataflow-derived addressed-memory or fence
member anchors and, where applicable, its separate Dataflow-derived exposure
anchors. An exposure remains outside `ServiceMemberRef` and does not acquire a
service leg. A foreign, missing, extra, duplicate, or wrong-kind anchor is
invalid.

The anchor variant ordinals are `Member = 0` and `Exposure = 1`. Canonical
anchor comparison is lexicographic over the variant ordinal and the complete
canonical Dataflow reference bytes of its payload. Canonical selection-key
comparison then appends the complete semantic `ExecutionContextKey` bytes.
The semantic framing is:

```text
u32be(anchor_variant_ordinal)
|| u64be(length(anchor_payload_bytes))
|| anchor_payload_bytes
|| u64be(length(execution_context_key_bytes))
|| execution_context_key_bytes
```

Within a SystemMapping root, a Spatial context may be serialized through its
canonical `SpatialMappingImportRef`, but semantic comparison first resolves
that alias to the complete imported ArtifactIdentity. A ServicePlan ordinal is
assigned only after complete plan semantic keys are canonicalized, sorted, and
deduplicated within the owning ServiceRealization. Neither the selection key
nor the ordinal receives an `EntityId`, stores a Fabric endpoint, or creates a
second execution-target decision.

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

Import and freeze resolve every persistent entity and structural reference
once before publishing the native model. Hot search tables use typed dense
indices and owner-local offsets; they do not carry ArtifactIdentity digests,
recursive persistent references, symbols, or paths.

Caches bind exact artifact identities and all relevant producer semantics.
They invalidate as a unit when those inputs change and cannot export local
references, coverage, legality conclusions, or physical decisions into a
different artifact context.

## Classification Boundary

The table in this document closes identity classification for every Mapping
record family in the complete `loom.mapping 3.0` schema. This document does
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
* exact derivation of every System service obligation, member, sink, and leg
  from the bound workload scope;
* canonical import and owner-child ordinal assignment;
* exact predecessor coupling required by the profile; and
* no symbol, path, location, textual order, provenance handle, or native
  dense-index reference authority.

Identity validation does not establish Mapping profile legality. The profile
verifiers in `docs/spec-mapping-verification.md` consume these valid reference
semantics and establish closure.
