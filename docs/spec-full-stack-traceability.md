# Full-Stack Traceability

Traceability records how immutable Loom artifacts were derived, evaluated,
selected, and deployed. It does not duplicate their semantic content.

## Owners

Each persistent fact has one owner:

* an artifact family owns its canonical semantic schema and bytes;
* Common artifact finalization owns identity framing and hashing;
* `InvocationManifest` owns derivation lineage and exact semantic inputs;
* `ExecutionJournal` owns mutable execution progress and infrastructure
  outcomes;
* `EvaluationRequest` owns one immutable evaluation input;
* `EvaluationEvidence` owns normalized evaluation output;
* the architecture-only `fabric.system` owns each InstructionCore
  Architectural Contract and Microarchitectural Realization, plus the exact
  Transport Architecture;
* Compiler Target Binding owns compiler-facing target choices and binary
  compatibility proof as specified by `docs/spec-runtime-abi.md` section
  `Compiler Target And Binary Compatibility`;
* Gem5 Simulation Binding owns simulator-model correspondences and their
  validation against exact architecture, compiler, and implementation inputs;
* `SimulationExecution` owns workload terminal observables, activity, and trace
  manifest;
* raw detailed bundles own large external payloads and tool products; and
* report/visualization exports are removable projections.

File names, symbol names, local paths, timestamps, and printer order are never
substitutes for typed identities.

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
  + ConfigurationABI
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
selected software + complete Mapping + Compiler Target Binding
  + target-specific binary + exact implementation refinements + images
  + memory and platform bindings
  -> Deployment

exact Fabric InstructionCore Architectural Contract
  + exact InstructionCore Microarchitectural Realization
  + Compiler Target Binding
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

These edges are required traceability relations, not exact field schemas or an
exhaustive executable-closure catalog. Compiler Target Binding and
Gem5SimulationBinding denote confirmed ownership and compatibility edges only.
Their persistent carrier or root schemas, Compiler Target Binding multiplicity
and tie-breaking, exact binary artifact binding, and exact
`Gem5SimulationBinding` schema remain downstream open work rather than wire
contracts defined here.

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
* HardwareImplementation, Gem5 Simulation Binding, and Deployment each bind
  the exact Interconnect Implementation and refinement required by their role;
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
  platform, and runtime ABI dependencies.

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
layouts beyond the store contract, duplicate every producer/consumer pair in a
fixture matrix, or freeze an open Compiler Target Binding or
`Gem5SimulationBinding` carrier schema.
