# Full-Stack Traceability

Traceability records how immutable Loom artifacts were derived, evaluated,
selected, and deployed. It does not duplicate their semantic content.

## Owners

Each persistent fact has one owner:

* an artifact family owns its canonical semantic schema and bytes;
* Common artifact finalization owns identity framing, hashing, and exact
  cross-artifact reference framing;
* the Canonical Dataflow family owns its graph, actor, root-thread-launch,
  static-graph-launch, and logical-memory-root entity catalog, canonical
  labeling, and read-only importer projection;
* `InvocationManifest` owns derivation lineage and exact semantic inputs;
* `ExecutionJournal` owns mutable execution progress and infrastructure
  outcomes;
* `EvaluationRequest` owns one immutable evaluation input;
* `EvaluationEvidence` owns normalized evaluation output;
* the architecture-only `fabric.system` owns each InstructionCore
  Architectural Contract and Microarchitectural Realization, plus the exact
  Transport Architecture;
* Compiler Target Binding owns compiler-facing target choices and binary
  compatibility proof as specified by `docs/spec-executable-closure.md`;
* Gem5 Simulation Binding owns simulator-model correspondences and their
  validation against exact architecture and Interconnect Implementation;
  Deployment admission separately validates selected compiler bindings;
* HardwareImplementation owns immutable RTL, netlist, ASIC, and FPGA
  implementation state, while ImplementationPlatform owns immutable technology
  inputs;
* Deployment owns the complete selected executable closure, and
  RuntimePlatformBinding owns provider-facing installation compatibility;
* `SimulationExecution` owns workload terminal observables, activity, and trace
  manifest;
* raw detailed bundles own large external payloads and tool products; and
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
  -> descriptor-owned typed output Artifacts and optional raw detailed bundle
EvaluationRequest + typed output bindings + retained raw material
  -> EvaluationEvidence

candidate sets + Evidence + resolved DSE policy
  -> CandidateDecision lineage and selected/Pareto artifact set
```

These edges are required traceability relations, not duplicated field schemas.
Their exact roots are owned by the linked artifact specifications; this view
cannot reopen or weaken them.

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

An artifact root that already binds the exact referenced identity may encode
only `T` at each internal use. This compact wire is a mechanical projection of
the complete pair, not a weak reference, compatibility class, lookup hint, or
rebinding permission. A symbol, path, printer position, construction handle,
or consumer-local dense index cannot replace either component.

## Artifact Store

The store keys objects by full `ArtifactIdentity` and retains enough framing to
validate the exact schema and preimage. Publishing identical content
deduplicates. Different content at an existing valid key is an identity
collision; malformed framing or key/preimage mismatch is corruption. Publication
never overwrites an existing object.

Validated reads derive the object path only from identity, reject symbolic links
and non-regular files, verify the expected schema descriptor, recompute the
digest, and return exactly the canonical semantic bytes. The caller provides an
already established non-symlink store root; the store does not create parent
directories.

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
duplication, deterministic replay, and gem5 rejection when any of its three
InstructionCore compatibility authorities disagree. Tests do not pin path
layouts beyond the store contract or duplicate every producer/consumer pair in
a fixture matrix.

Blob-digest anchors cover one fixed logical-byte vector, the zero-length blob,
strict binary and lowercase-text widths, transparent compression round-trip,
atomic publication, equal-byte deduplication, and collision or corruption
rejection. They do not establish a digest-algorithm registry, duplicate blob
tests for every owner, or pin storage layout and compression choices.
