# Full-Stack Traceability

Traceability records how immutable Loom artifacts were derived, evaluated,
selected, and deployed. It does not duplicate their semantic content.

## Owners

Each persistent fact has one owner:

* an artifact family owns its canonical semantic schema and bytes;
* Common artifact finalization owns identity framing, hashing, and exact
  cross-artifact reference framing;
* the Structured Program family owns each complete mixed-dialect candidate,
  its canonical bytes, parent-local structural references, and read-only
  importer projection;
* the Canonical Dataflow family owns its graph, actor, root-thread-launch,
  static-graph-launch, and logical-memory-root entity catalog, canonical
  labeling, and read-only importer projection;
* `InvocationManifest` owns derivation lineage and exact semantic inputs;
* `ExecutionJournal` owns mutable execution progress and infrastructure
  outcomes;
* `EvaluationRequest` owns one immutable evaluation input;
* `EvaluationEvidence` owns normalized evaluation output;
* the model-parameter contract registry owns each payload, accepted
  case/condition domain, feature, inference, prediction, and calibration-group
  contract, while `ModelParameterBundle`
  owns one immutable typed payload reference under that exact contract as
  specified by [Evaluation and DSE](spec-dse-feedback.md#model-parameters-and-training);
* the architecture-only `fabric.system` owns each InstructionCore
  Architectural Contract and Microarchitectural Realization, plus the exact
  Transport Architecture;
* Compiler Target Binding owns compiler-facing target choices and binary
  compatibility proof as specified by `docs/spec-executable-closure.md`;
* Gem5 Simulation Binding owns simulator-model correspondences and their
  validation against exact architecture and Interconnect Implementation;
  Deployment admission separately validates selected compiler bindings;
* HardwareImplementation owns immutable RTL, netlist, ASIC, and FPGA
  implementation state plus its exact provider-owned external dependencies,
  while ImplementationPlatform owns the selected ASIC technology release or
  FPGA ordering code and typed technology-corner keys;
* Deployment owns the complete selected executable closure, and
  RuntimePlatformBinding owns provider-facing installation compatibility;
* `SimulationExecution` 2.0 owns workload terminal observables, activity, and
  the narrow System root-lifecycle progress sequence;
* owner-attempt or scratch storage retains large external payloads and raw
  tool products that have no semantic Artifact owner; and
* report/visualization exports are removable projections.

File names, private symbol names, local paths, timestamps, and printer order
are never substitutes for typed identities. Externally visible linkage names
remain software ABI semantics and are included by the Canonical Dataflow
family rather than used as entity references.

## Required Artifact Graph

The complete stack can contain these derivations:

```text
source + driver configuration
  -> LLVM IR
  -> relocatable accelerator payload in each selected object/archive member
selected relocatable payloads
  -> LLVM Linker/LTO merged LLVM IR
  -> initial Structured Program Candidate S0
  -> selected Structured Program Candidate Sn
  -> initial Canonical Dataflow Program D0
  -> selected Canonical Dataflow Program D*

ADG Builder recipe or built-in template
  -> architecture-only fabric.system
  -> exact Transport Architecture
architecture-only fabric.system + protocol-specific implementation definition
  -> sibling Interconnect Implementation and exact refinement
architecture-only fabric.system
  -> ConfigurationABI
architecture-only fabric.system + exact Interconnect Implementation/refinement
  + ConfigurationABI + resolved generator binding
  + optional ImplementationPlatform
  -> HardwareImplementation(RTL)
  -> HardwareImplementation(synthesized/physical/FPGA)

Canonical Dataflow Program + exact Fabric architecture
  -> TechMapping
  -> SpatialMapping
logical thread definition + architecture-only fabric.system
  + exact Transport Architecture + exact SpatialMapping imports
  -> SystemMapping
SystemMapping-selected AccCore
  -> exact InstructionCore Architectural Contract
SystemMapping-selected AccCore
  -> exact InstructionCore Microarchitectural Realization
SystemMapping-selected AccCore
  -> derived InstructionCoreContextRef
exact InstructionCore Architectural Contract
  -> mechanically selected and validated Compiler Target Binding
InstructionCore program + Compiler Target Binding
  -> target-specific binary
complete Mapping + ConfigurationABI
  -> HardwareConfigurationImage
HardwareImplementation + provider binding
  -> RuntimePlatformBinding
linked host program + complete SystemMapping
  + exact Compiler Target Bindings and InstructionCore binaries
  + exact HardwareImplementations and RuntimePlatformBindings
  + configuration images + static logical-memory images
  -> Deployment

exact Fabric InstructionCore Architectural Contract
  + exact InstructionCore Microarchitectural Realization
  + exact Interconnect Implementation/refinement
  + gem5 model, build, and Bridge identities
  -> Gem5SimulationBinding
Deployment + Gem5SimulationBinding
  -> EvaluationRequest subjects {deployment, system_model}

immutable subjects + exact model binding
  -> EvaluationRequest
  -> descriptor-owned typed output Artifacts
EvaluationRequest + typed output bindings
  -> EvaluationEvidence

exact Training, Validation, and HeldOut Evidence partitions
  + optional prior parameter bundles
  + exact trainer binding, resolved configuration, and seed
  -> ModelParameterBundle candidate
ModelParameterBundle + exact typed Validation or HeldOut Evidence input
  + exact calibration validator binding
  -> EvaluationRequest
  -> EvaluationEvidence

candidate sets + Evidence + resolved DSE policy
  -> CandidateDecision lineage and selected/Pareto artifact set
```

These edges are required traceability relations, not duplicated field schemas.
Their exact roots are owned by the linked artifact specifications; this view
cannot reopen or weaken them.

In the hardware edges, the resolved generator binding labels the invocation
derivation recorded by `InvocationManifest`; it is not a field of the output
HardwareImplementation. Each HardwareImplementation independently owns the
complete exact dependencies and payload closure needed to consume its
represented state. A later implementation may name an earlier implementation
as a typed derivation input without making that input an implicit semantic
parent.

Not every invocation traverses every edge. Compatibility compilation may stop
at ordinary compiler output. Mapping, simulation, RTL, EDA, deployment, and DSE
are explicit requested derivations or evaluations.

## Required Coupling

Every cross-component consumer validates the exact identities required by its
semantic contract. Key boundaries are:

* DFG-sim consumes `{Canonical Dataflow Program}`;
* the final-link frontend consumes exact compatible relocatable payloads and
  derives one merged LLVM module without selecting Fabric or Mapping;
* TechMapping consumes exact Dataflow and Fabric;
* Spatial PnR consumes exact `D/T/F/C/K` under the Spatial
  `loom.mapping_constraints 1.0` root;
* System PnR consumes exact `D/F/R/H/C/K` under the System root; the finalized SystemMapping
  persists only its exact `D/F`, root launches, derived spatial imports, and
  selected Mapping records;
* SystemMapping binds the architecture-only `fabric.system` and its exact
  Transport Architecture, never an Interconnect Implementation, Compiler Target
  Binding, binary, HardwareImplementation, gem5 model, or Deployment identity;
* each concrete occurrence of a logical thread definition resolves through its
  SystemMapping Thread Execution Binding to a selected AccCore and derived
  `InstructionCoreContextRef`;
* Compiler Target Binding is mechanically selected and validated from that
  AccCore's exact InstructionCore Architectural Contract, and each
  target-specific binary is validated under that binding;
* CGRA-sim consumes `{D,F,SpatialMapping}`;
* HardwareImplementation binds the exact Interconnect Implementation required
  by its role; Gem5 Simulation Binding references the exact same implementation
  independently, and Deployment recovers it through its selected
  HardwareImplementation closure;
* sys-sim consumes `{Deployment,Gem5SimulationBinding}` through exact
  Evaluation subject roles `deployment` and `system_model`, and the gem5 model
  must be compatible with all three authorities: the exact Fabric
  InstructionCore Architectural Contract; the exact InstructionCore
  Microarchitectural Realization, including execution structure, timing,
  capacity, and mapping-visible resources; and the compatible Compiler Target
  Binding used by the binary. Gem5 Simulation Binding remains simulator
  binding, never hardware truth;
* RTL/EDA Evaluation consumes exact `HardwareImplementation` subjects;
* mapped RTL execution consumes exact `{HardwareImplementation,Deployment}`;
* `HardwareConfigurationImage` is finalized only from the exact Mapping and
  ConfigurationABI closure specified by
  `docs/spec-configuration-deployment.md`; and
* Deployment references, rather than copies, its exact software, hardware,
  mapping, compiler target, target-specific binary, configuration, memory,
  runtime-platform, and runtime ABI dependencies.

An identity mismatch is an invalid input, not an Evaluation finding and not a
reason to repair or reinterpret the consumer artifact.

## Invocation Lineage

`docs/spec-dse-feedback.md` section `Invocation and Recovery Records` is
the sole owner of `InvocationManifest` and `ExecutionJournal` fields. This
traceability view references those records and does not repeat their schema,
copy artifact fields, or copy normalized Evidence results. Repeated lineage
paths to identical semantic content converge on the same ArtifactIdentity.
An ExternalToolInvocationBundle is an owner-specific nonsemantic attempt
record referenced by that manifest; its local paths, module closure, generated
scripts, raw outputs, and completion record are never copied into semantic
lineage.

Training occurrence, dataset, trainer, configuration, seed, and attempt facts
therefore remain manifest lineage. They are not copied into a parameter bundle;
two occurrences producing the same canonical payload under the same exact
registry-owned parameter contract converge on one bundle identity.

One code revision plus one resolved semantic configuration and identical input
artifacts must produce identical semantic outputs. Wall time, host parallelism,
licenses, paths, and storage locations can affect whether an attempt completes,
but not which formal result is selected.

## Artifact Identity Contract

Every finalized artifact crossing a semantic boundary has one
`ArtifactIdentity`. Common computes SHA-256 over exactly:

```text
bytes("loom.artifact.identity.v1\0")
|| u32be(length(schema_identity))
|| bytes(schema_identity)
|| u32be(schema_version.major)
|| u32be(schema_version.minor)
|| u64be(length(canonical_semantic_bytes))
|| canonical_semantic_bytes
```

The algorithm, domain tag, framing, and 32-byte digest width are fixed. External
text is exactly 64 lowercase hexadecimal characters. Each artifact family owns
its schema descriptor and canonical semantic serialization; Common owns only
the framing, digest, and validated store behavior.

Canonical semantic bytes exclude timestamps, producer metadata, invocation
bindings, host paths, diagnostics, visualization layout, and lineage unless the
artifact schema explicitly makes a typed upstream reference semantic.

Import never performs an implicit schema upgrade. A compatible minor-version
adapter first validates the source under its own schema, then constructs and
finalizes a new artifact with a new identity. A major-version migration is an
explicit owner-provided conversion, not a Common fallback. Without such a
converter, an old artifact is accepted only by a consumer supporting that exact
schema or is rejected as unsupported. In-place identity preservation, textual
patching, and latest-version guessing are forbidden.

## Blob Digest Contract

Large non-Artifact payloads use one distinct `BlobDigest` value type:

```text
BlobDigest := SHA-256(logical_blob_bytes)
```

The algorithm and 32-byte width are fixed. External text is exactly 64
lowercase hexadecimal characters. `BlobDigest` and `ArtifactIdentity` are
different static types even though both contain a SHA-256 result. A blob has no
Artifact schema descriptor, semantic framing, lineage, or artifact-local
entity catalog.

The digest covers the exact logical bytes presented to consumers. Transparent
storage compression, filesystem paths, chunk placement, indexes, and transport
encoding do not change those bytes or the digest. A compressed store object
must be decoded before its logical bytes are rehashed. A zero-length logical
blob has its ordinary SHA-256 digest. Values are always exactly 32 bytes and
absence uses outer optionality. An all-zero 32-byte value, if produced, remains
an ordinary digest and cannot be reserved as an absence sentinel.

The blob store publishes complete bytes atomically, verifies the full byte
sequence on deduplication, and rehashes on read. Equal bytes deduplicate. A
digest occupied by different bytes is a hard collision, while malformed
storage or a digest/bytes mismatch is corruption. Neither case may be repaired
by selecting one payload. Blob ownership, media type, and relation to a typed
Artifact remain in the referencing owner's manifest; the digest itself owns
only content identity.

## Artifact Reference Framing

Every semantic cross-artifact reference has one complete meaning:

```text
ArtifactReference<T> =
  (exact finalized ArtifactIdentity,
   typed artifact-local target T)
```

Common owns only this pair framing and the requirement that the identity
resolve and validate exactly. The referenced artifact family owns the closed
entity and structural-target variants admitted by `T`, their canonical
encoding, and target validation.

Heterogeneous containers such as Evaluation scope cannot name one C++ `T` at
compile time. They use one existential wire projection without transferring
type ownership to the container:

```text
ArtifactRootReference {
  schema: exact ArtifactSchemaDescriptor
  artifact: exact finalized ArtifactIdentity
}

ArtifactLocalReferenceTypeDescriptor {
  owner_schema: exact ArtifactSchemaDescriptor
  owner_local_kind: uint32
}

EncodedArtifactLocalReference {
  artifact: ArtifactRootReference
  owner_local_kind: uint32
  payload: owner-produced canonical bytes
}
```

The root schema is present because an `ArtifactIdentity` digest alone does not
select an importer or validator. `owner_local_kind` is a stable closed ordinal
owned by that exact Artifact family and schema version. It is not a global
entity kind, consumer enum, textual type name, or native variant index.
The complete local-reference type descriptor is derived exactly once as
`(artifact.schema, owner_local_kind)`; it is not duplicated in the encoded
value.

Each family that permits existential import statically registers, for every
local kind, its typed C++ target, canonical encoder, strict decoder, and
validator against one exact imported Artifact. Common owns only the outer
framing:

```text
u32be(length(schema identity)) || bytes(schema identity)
|| u32be(schema major) || u32be(schema minor)
|| 32-byte ArtifactIdentity
|| u32be(owner-local kind)
|| u64be(payload length) || payload bytes
```

Canonical JSON represents the payload bytes as lowercase hexadecimal. Import
resolves the exact schema descriptor, validates and imports the exact Artifact,
looks up that owner's local-kind codec, decodes the payload, requires exact
decode/re-encode equality, and invokes the owner validator. Unknown schemas or
kinds, malformed or noncanonical payloads, wrong-owner targets, and unresolved
targets are invalid. A consumer must not reinterpret the payload, erase it to a
consumer-local integer, or substitute an `ofEntities` tuple, path, symbol, or
property bag.

The typed in-memory API remains `ArtifactReference<T>`. An
`EncodedArtifactLocalReference` is only the persistent or heterogeneous carrier
used to recover that typed value through the owner codec. An Artifact root is a
separate `ArtifactRootReference` variant; it is never represented by a reserved
local kind or sentinel payload.

An artifact root that already binds the exact referenced identity may encode
only `T` at each internal use. This compact wire is a mechanical projection of
the complete pair, not a weak reference, compatibility class, lookup hint, or
rebinding permission. A heterogeneous field may omit the complete existential
type framing only when its containing schema statically fixes the exact owner
schema and local kind. A symbol, path, printer position, construction handle,
or consumer-local dense index cannot replace either component.

## Artifact Store

The Common store owns exactly one immutable object per `put`. It does not own
artifact dependency graphs, multi-object transactions, publication manifests,
or family-specific import. An artifact family must resolve and validate its
dependencies before asking the store to publish that family's root object.

The store keys objects by full `ArtifactIdentity` and retains enough framing to
validate the exact schema and preimage. One `put` writes and validates the
complete identity preimage in a temporary object on the same filesystem,
durably flushes its bytes, atomically inserts the final identity-derived name
without replacement, and flushes the containing directory before reporting
success. A reader therefore observes either no final object or one complete
validated object, never a partial object.

A successful return proves that the one object is durably published. If the
publisher crashes or receives an I/O error before that return, it must not infer
that the object is absent: a crash or directory-flush failure may occur after
the complete final name became visible. Recovery deterministically repeats the
same `put` or performs a validated `get`. The resulting store state is either
absent or the complete expected object; there is no pending, partial, or
rollback state and no cleanup transaction.

Publishing identical content deduplicates, including concurrent publication.
Different content at an existing valid key is an identity collision; malformed
framing or key/preimage mismatch is corruption. Publication never overwrites
an existing object. The current ArtifactStore contract exposes no object
mutation or deletion API;
out-of-band removal is reported as missing and out-of-band modification as
corruption.

Validated reads derive the object path only from identity, reject symbolic
links and non-regular files, verify the expected schema descriptor, recompute
the digest, and return exactly the canonical semantic bytes. The caller
provides an already established non-symlink store root; the store does not
create parent directories.

Failure classification is exact:

* an absent identity is `artifact_store_missing`;
* a present object with another schema is `artifact_schema_mismatch`;
* malformed framing, a key/preimage mismatch, or an invalid stored object is
  `artifact_store_corruption`;
* distinct complete preimages occupying one identity are
  `artifact_identity_collision`; and
* filesystem, flush, or durability failures are `artifact_store_io`.

An `artifact_store_io` result from `put` does not promise absence. The caller
returns no successful artifact reference for that attempt and uses idempotent
retry to determine the final state.

## Diagnostics

Every boundary distinguishes:

* missing artifact;
* schema/version mismatch;
* malformed or unresolved typed reference;
* identity mismatch or store corruption;
* unsupported transformation/evaluation capability;
* structurally invalid input;
* incomplete derivation or external execution; and
* completed Evaluation with adverse findings.

Diagnostics reference the owning artifacts and invocation. They do not mutate
them or become alternative pass/fail authority.

## Projection Boundary

Reports, tables, dashboards, and `--loom-viz-export` bundles may join this graph
by exact identity. They can be deleted and regenerated without changing any
semantic artifact, Evaluation result, DSE choice, or deployment.

## Anchor Verification

Stable tests cover the fixed identity preimage, canonicalization invariance,
store corruption detection, exact consumer coupling, lineage without semantic
duplication, parameter-bundle convergence across distinct training provenance,
contract divergence, exact calibration Evidence closure, deterministic replay,
and gem5 rejection when any of its three InstructionCore compatibility
authorities disagree. Tests do not pin path layouts beyond the store contract
or duplicate every producer/consumer pair in a fixture matrix.

Store anchors cover single-object complete-or-absent visibility, successful
durability acknowledgement, identical concurrent publication, retry after a
post-insertion failure, and the five failure classes above. They do not add a
mock transaction coordinator, dependency manifest, deletion protocol, or
failure-point cross product.

Reference-framing anchors cover one owner-local fixed byte vector, typed
round-trip through the owner codec, and rejection of a wrong schema, wrong
local kind, noncanonical payload, and unresolved target. They do not duplicate
the same tests for every Artifact family or permit a generic reference-property
fixture framework.

Blob-digest anchors cover one fixed logical-byte vector, the zero-length blob,
strict binary and lowercase-text widths, transparent compression round-trip,
atomic publication, equal-byte deduplication, and collision or corruption
rejection. They do not establish a digest-algorithm registry, duplicate blob
tests for every owner, or pin storage layout and compression choices.
