# Loom Compiler Part 1: Source Integration

This document specifies the source-facing part of the Loom compiler frontend.
Loom's only required language-neutral handoff is semantically valid LLVM IR.
Optional typed Loom hints and provenance can improve analysis, but no other
LLVM-producing frontend must implement a Loom-private metadata protocol.

Part 1 does not lower code to `dataflow`. It emits LLVM IR and any available
typed hints that the mechanical LLVM-to-SCF raising pipeline can preserve in
initial SCF-stage MLIR. Later analysis, structured optimization, and DSE may
use those hints when constructing a selected Structured Program Candidate.
Missing hints are proved by analysis where possible or remain unknown; the
mechanical pipeline does not select accelerator regions or decomposition.

## 1. Scope

Part 1 owns:

* Accepting LLVM IR as the stable cross-language hand-off format.
* Embedding, linking, or importing source frontends that can produce LLVM IR.
* Providing embedded clang as the first limited provider for C / C++.
* Preserving optional source intent and provenance on functions, loops,
  lexical regions, memory objects, and calls when available.
* Providing a stable hand-off format to Part 2, even when the source
  language does not outline accelerator code into separate functions.
* Defining the carrier-independent relocatable accelerator payload embedded in
  ordinary compile-only objects and its deterministic final-link merge.

Part 1 does not own:

* Deciding the final HostCore-to-AccCore boundary.
* Selecting loop schedules, parallel decomposition, vectorization,
  reduction strategy, or ownership boundaries.
* Recovering structured control flow.
* Building `dataflow.thread` or `dataflow.graph`.
* Selecting Fabric, Mapping, configuration images, or Deployment content.

## 2. Provider Model

The Loom source integration model is language-neutral at the compiler
boundary:

* The required compiler input is LLVM IR.
* Source language support is provided by front-end providers. A
  provider may be embedded as a library, linked into the Loom binary,
  or replaced by an external producer that emits the same LLVM IR contract.
* C / C++ through embedded clang is the first engineering target, not
  the semantic limit of Loom.
* A provider may mark whole functions, sub-function source ranges,
  loop nests, or compiler-created regions as accelerator candidates.
  None of these source constructs is a committed AccCore boundary.

The provider contract is deliberately weaker than the later IR
contract. Part 1 preserves intent and provenance. The initial raising
pipeline mechanically recovers structured form, while later structured
optimization and DSE decide which legal candidates are selected.

## 3. Optional Typed Hints

Part 1 may emit the following typed hints. Part 2 may ignore a hint, but it
must not turn an absent hint into a proven fact. A property can become known
only through preserved source information or an owning analysis.

* **Accelerator candidates.** Mark a function, loop, lexical compound
  statement, outlined helper, or compiler-generated code range as a
  candidate for AccCore execution. A candidate is not yet a committed
  ownership boundary.
* **Parallel intent.** Mark loops or loop nests as parallel, temporal,
  spatial, vector-like, or reduction-like. These hints are inputs to
  later structured optimization and DSE. They do not cause the
  mechanical raising pipeline to convert `scf.for` to `scf.forall` or
  construct mapping attributes.
* **Memory intent.** Preserve source-level noalias, restrict,
  alignment, address-space, lifetime, and transfer-direction hints.
  These hints guide Part 2 memory-region analysis; Part 3 still builds
  conservative dependence edges.
* **Static local storage intent.** Mark objects that are candidates for
  AccCore-local static storage, scratch allocation, or SpatialCore
  layout. Part 2 decides whether the candidate can be represented in
  SCF-shaped MLIR.
* **Diagnostic anchors.** Preserve source locations and stable names so
  Part 2 and Part 3 can report why a candidate region cannot be
  accelerated.

## 4. Boundary Principle

A source function is not an accelerator boundary by default. Source
front-ends may emit candidate metadata for whole functions, but they may
also mark sub-function lexical regions, loop nests, or compiler-created
regions. Initial SCF-stage MLIR does not commit one of these candidates.
Committed ownership appears only in a selected Structured Program
Candidate produced after structured optimization and DSE.

This rule keeps the source model flexible:

* A single source function may contain zero, one, or many accelerator
  regions.
* Accelerator code may be written inline in an ordinary host function.
* Scalar-only accelerator regions remain representable; they do not
  need a source-level parallel loop just to become valid input.

## 5. Initial Clang Provider

The first source provider is an embedded clang pipeline. `loom-cc` and
`loom-c++` must call the clang and Loom libraries in process so they can
preserve diagnostics and control the LLVM pipeline. Shelling out to a stage
binary is not the product architecture. Developer tools may expose the same
library boundaries without becoming public compiler drivers.

For the CMSIS drop-in compiler target, this provider must also satisfy
the source-facing compatibility contract in
`docs/spec-cmsis-dropin-compiler.md`.

The clang provider is expected to preserve:

* Loop metadata relevant to parallelism and memory dependence.
* Function, variable, and source-range annotations.
* Address spaces and memory attributes.
* Debug locations for diagnostics.

Language-specific pragmas, attributes, builtins, and library calls lower
to the metadata classes above. The exact spelling is outside this spec;
the IR contract is the metadata observable by Part 2.

Other LLVM-producing language frontends can participate without mimicking C or
C++ syntax or emitting Loom hints. A language integration may add equivalent
typed hints, but valid LLVM IR remains the only mandatory handoff.

## 6. Hand-Off to Part 2

Part 1 emits LLVM IR plus any optional typed hints. The handoff contract is:

* LLVM IR remains semantically equivalent to the source program.
* Candidate accelerator regions are hints, not committed boundaries.
* Preserved hints must remain associated with their loops, memory objects,
  calls, and source ranges after mechanical raising and canonical LLVM
  simplification.
* Mechanical raising may recover and normalize structured control, but
  it does not select a performance-distinct loop, parallel, vector, or
  ownership form.
* If a hint is absent, lost, or contradicted by later LLVM transforms, Part 2
  preserves uncertainty unless analysis proves the fact, or emits a diagnostic
  when the selected profile requires that fact.

## 7. Relocatable Accelerator Payload

Compile-only acceleration uses one complete Artifact family:

```text
loom.relocatable_accelerator_payload 1.0
```

Its canonical root is:

```text
RelocatableAcceleratorPayload {
  llvm_provider {
    repository_identity
    full_commit_identity
  }
  target_triple
  data_layout
  abi_compatibility_key
  frontend_config_view {
    component_view_schema_descriptor
    canonical_view_bytes
    component_view_digest
  }
  normalized_llvm_bitcode {
    sha256_digest
    bytes
  }
}
```

`repository_identity` is the closed LLVM provider identity selected by the
build; the initial value denotes `llvm-project`. `full_commit_identity` is the
complete pinned commit, not a release nickname or abbreviated hash. The target
triple and data layout are canonical LLVM strings validated by LLVM parsers.
They are mechanically projected from the normalized module and checked against
it; the module remains their semantic owner. They are not independent target
choices. The later Compiler Target Binding owns final InstructionCore codegen
target selection and must prove compatibility with these module facts.

`ResolvedFrontendConfigView` is the Part 1 component view of ResolvedConfig. It
contains exactly frontend semantic options that can affect emitted LLVM IR or
cross-translation-unit compatibility. Part 1 owns that typed view schema;
`docs/spec-config-ssot.md` owns common component-view framing and digest rules.
The payload stores the descriptor, canonical bytes, and digest together so it
is self-contained; readers recompute the digest and reject disagreement.

The ABI compatibility key is mechanically derived by Part 1 from the exact
LLVM provider and commit, target triple, data layout, and complete frontend
config-view descriptor and digest under the fixed domain separator
`loom.frontend-abi-compatibility-v1`. It is a validated compact comparison key,
not an independently authorable compatibility claim.

Part 1 owns one deterministic LLVM-module normalization and bitcode writer for
this schema version. The stored SHA-256 digest is recomputed over exactly the
normalized bitcode bytes; the bytes, not the digest, remain the module content
authority. Symbol definitions, declarations, linkage, visibility, COMDAT,
module flags, and ODR semantics remain solely in that LLVM module. The payload
contains no copied symbol table or Loom-specific substitute.

The payload must not contain a Fabric, Mapping, MappingConstraintSet,
ConfigurationABI, HardwareConfigurationImage, HardwareImplementation, or
Deployment reference. Those choices occur after final link. Optional source
locations or hints are present only when encoded in the normalized LLVM module
under this version's normalization contract.

### Object Carrier And Final Link

An ordinary object remains self-contained: its carrier adapter embeds the
complete canonical payload bytes. Object section name, alignment, compression,
container metadata, and archive layout are non-semantic projections and do not
enter ArtifactIdentity. A platform adapter may change those details without
changing the payload.

At final link, the ordinary linker is the sole authority for which object and
archive members participate. Loom collects payloads from exactly those selected
members. Every collected payload must have identical LLVM provider/commit,
target triple, data layout, ABI compatibility key, and complete frontend config
view. Version 1.0 has no implicit config merge, precedence rule, or
compatibility lattice; disagreement is a typed link error.

Loom parses and verifies every normalized module, then uses the pinned LLVM
Linker and LTO libraries to perform symbol resolution, COMDAT/ODR handling,
module-flag validation, internalization, and whole-program optimization. The
resulting linked LLVM module is the ordinary Part 1 hand-off to Part 2. Loom
does not reimplement these LLVM semantics or infer them from a copied manifest.

Objects without a payload remain valid external or InstructionCore-only link
inputs. If no selected member contains a payload, no accelerator compilation is
implied. If acceleration is explicitly required but a selected payload is
malformed or incompatible, the driver diagnoses the exact member and mismatch
instead of silently discarding accelerator semantics.

Anchor-level tests cover deterministic normalization and identity, self-
contained carrier round trip, exact compatibility rejection, linker-selected
archive membership, LLVM-owned COMDAT/ODR resolution, and absence of downstream
hardware artifacts. Tests do not freeze section names, compression, archive
layout, or LLVM internal pass structure.

Canonical semantic bytes use the field order above. Fixed digests use raw
32-byte values; strings and variable byte sequences use unsigned 64-bit
big-endian length framing followed by exact bytes. Integers use unsigned
big-endian encoding. There is no parallel JSON, MLIR, or host-struct identity
encoding.

## 8. References

* `docs/spec-compiler-part-2-scf.md` -- mechanical LLVM-to-SCF raising
  and the boundary to later structured selection.
* `docs/spec-compiler-part-3-dfg.md` -- SCF-to-DFG lowering.
