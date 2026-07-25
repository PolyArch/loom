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

The initial C and C++ provider exposes one optional public candidate spelling:

```text
#pragma loom candidate
```

It has no arguments and applies only to the immediately following function
definition or `for`, `while`, or `do` loop. It is a nonbinding candidate hint:
it does not select AccCore ownership, require acceleration, change ordinary
compiler behavior, or enable an acceleration profile. Malformed, dangling, or
misapplied pragmas are source diagnostics. Begin/end regions, required
variants, a companion macro header, and parallel attribute spellings are not
part of the first source contract.

The clang provider lowers the hint through one provider-owned metadata
encoding. Function candidates use a typed function annotation; loop candidates
use a loop metadata operand. Import into S0 projects both to one internal unit
attribute on the owning callable or loop. Part 2 must consume or explicitly
discard that hint before candidate finalization; it cannot remain as unresolved
target-specific metadata in Sn.

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

Other language-specific attributes and builtins may lower to the metadata
classes above, but they are provider extensions rather than additional public
Loom spellings. Library calls do not acquire Loom semantics from a symbol name
or arity.

Other LLVM-producing language frontends can participate without mimicking C or
C++ syntax or emitting Loom hints. A language integration may add equivalent
typed hints, but valid LLVM IR remains the only mandatory handoff.

Direct TOSA, ONNX, Linalg, or framework-graph ingestion is not a first-version
public contract. Such dialects may appear as internal transient MLIR after an
external frontend has produced LLVM IR, but they do not create another
language-neutral product boundary.

## 6. Hand-Off to Part 2

Part 1 emits LLVM IR plus any optional typed hints. The handoff contract is:

* LLVM IR remains semantically equivalent to the source program.
* The final linked LLVM module is the sole owner of imported function and ABI
  facts. Its linkage, calling convention, COMDAT, personality,
  argument/result attributes, memory effects, target features, floating-point
  environment, and other LLVM semantics are not copied into a Loom-private
  function contract.
* Import into MLIR preserves the corresponding `llvm.func` envelope and its
  typed attributes. A later `func.func` may represent a genuinely
  standard-MLIR-native callable or helper, but it must not mirror an imported
  LLVM function merely to satisfy a pass wrapper.
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

Ordinary Clang/LLVM profile-guided optimization continues to use LLVM-owned
profile formats, `!prof`, function-entry counts, and profile diagnostics. Raw
profile inputs may be recorded by BlobDigest in invocation provenance, but the
final linked LLVM module remains the semantic owner of imported weights. Loom
hardware and simulation feedback enters compilation only through exact
`EvaluationEvidence`; there is no `ProfileArtifact`, latest-profile scan, or
second branch-weight schema.

MLIR locations and imported LLVM debug information remain the source
provenance authority through raising and lowering. They do not change the
semantic identity of a finalized Canonical Dataflow Program. Its finalizer may
derive a source-object-to-entity relation for diagnostics and
`--loom-viz-export`; Loom does not introduce a second provenance IR.

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
triple is the pinned LLVM canonical spelling. The data layout is the exact
nonempty module-owned spelling accepted by the pinned LLVM `DataLayout` parser;
LLVM does not define a canonical printer for every valid layout. Both fields
are mechanically projected from the normalized module and checked against it;
the module remains their semantic owner. They are not independent target
choices. The later Compiler Target Binding owns final InstructionCore codegen
target selection and must prove compatibility with these module facts. Its
persistent schema and the exact InstructionCore binary relation are owned by
`docs/spec-executable-closure.md`.

`ResolvedFrontendConfigView` version 1.0 has the schema identity
`loom.config.view.frontend` and an explicitly empty field set. Its
exact schema descriptor bytes are the ASCII bytes
`loom.config.view.frontend.1.0`, without a trailing zero byte. Its
`canonical_view_bytes` is the empty byte sequence. Its
`component_view_digest` is the Common component-view digest over the exact
schema descriptor and those empty bytes. The empty view proves that Part 1
reads no semantic field from ResolvedConfig under this schema version; it does
not permit hidden access to the complete config.

Language, ABI, optimization, target, and command-line decisions that affect
the emitted module are already owned by that LLVM module. Fabric, DSE,
Mapping, and backend choices occur after final link and are forbidden from the
frontend view. Paths, output locations, host concurrency, and execution limits
belong to invocation bindings. A hint that changes the module is already
present in the bitcode. Adding the first real frontend config field therefore
changes the view schema version and its sole deterministic projector. Part 1
owns that view schema; `docs/spec-config-ssot.md` owns the common
component-view digest framing.

The payload stores the frontend-view descriptor, canonical bytes, and digest
together so it is self-contained. A reader recomputes the digest and rejects
disagreement. The ABI compatibility key has this sole preimage:

```text
SHA-256(
  bytes("loom.frontend.abi.compatibility.v1\0")
  || bytes64(repository_identity)
  || bytes64(full_commit_identity)
  || bytes64(canonical_target_triple)
  || bytes64(frontend_view_schema_descriptor)
  || bytes64(frontend_view_canonical_bytes)
)

bytes64(x) = u64be(length(x)) || x
```

The domain separator includes its trailing zero byte. The key does not include
`component_view_digest`: that digest is already derived from the authoritative
descriptor and view bytes and is validated independently. It does not include
the exact data-layout spelling because LLVM defines structural layout equality
but no canonical byte projection for every valid layout. It also does not
include the bitcode digest or module contents, because distinct translation
units must be able to occupy the same compatibility cohort. Changing any
preimage field changes the key. The initial exact `repository_identity`
denotes the pinned `llvm-project` repository; `full_commit_identity` is its
complete normalized commit identity.

The ABI key is a necessary cohort and preflight check, not the complete LLVM
ABI authority. Readers validate the raw provider, canonical triple, exact
DataLayout projection, and view fields as well as the key. The pinned LLVM
Linker and LTO libraries remain the authority for module flags, symbol and
COMDAT resolution, ODR, and all other module-level compatibility rules.

### LLVM Module Normalization

Part 1 owns one deterministic LLVM parser and bitcode writer contract for
payload version 1.0:

1. Parse with the pinned LLVM provider, fully materialize the module, and run
   the LLVM verifier.
2. Require nonempty valid target-triple and DataLayout fields. Canonicalize the
   target triple with pinned LLVM `Triple::normalize`. Parse the DataLayout for
   validation, but retain its exact module spelling. Do not reorder entries,
   remove redundant entries, or replace it with a target-derived default.
3. Preserve module identifier, source filename, debug provenance, module
   flags, attributes, symbols, linkage, visibility, COMDAT, inline assembly,
   named metadata, and module order. Do not sort, rename, strip debug
   information, optimize, or run LTO.
4. Write the complete module through pinned `WriteBitcodeToFile` with
   `ShouldPreserveUseListOrder=false`, `Index=nullptr`, and
   `GenerateHash=false`. Do not add a wrapper, summary index, or
   LLVM-generated module hash.

LLVM defines use-list order directives as non-semantic, so dropping use-list
order is the one normalization approved by this contract. The resulting
determinism guarantee is deliberately narrow:

```text
same complete in-memory LLVM module
+ same pinned LLVM commit
+ same writer contract
=> same normalized bitcode bytes
```

This is serialization determinism, not semantic canonicalization across
arbitrary equivalent LLVM modules. Debug and source provenance remain in the
module, so provenance or source-path differences may change compile-only
payload identity. Two modules whose DataLayout spellings parse to the same LLVM
layout also retain distinct compile-only payload identities when their exact
strings differ. This byte-exact identity is distinct from the later Canonical
Dataflow identity, which excludes provenance under its own finalization
contract. Loom does not create a sidecar provenance IR.

`Triple::computeDataLayout(ABIName)` is a target-and-ABI default generator, not
a validator or canonicalizer. A frontend may use it before payload creation to
populate a module that has no layout only when the exact ABI is already known.
Once a module contains a valid DataLayout, normalization never invokes that
generator to replace or judge the module-owned value.

For example, a verified `riscv64-unknown-elf` module carrying the `lp64e` ABI
may own
`e-m:e-p:64:64-i64:64-i128:128-n32:64-S64`. Payload normalization accepts and
preserves that exact layout. An ABI-unspecified target default ending in
`S128` cannot reject or replace it.

The stored SHA-256 digest is computed over exactly the normalized bitcode
bytes. The bytes, not that digest, remain the module content authority. Symbol
definitions, declarations, linkage, visibility, COMDAT, module flags, and ODR
semantics remain solely in that LLVM module. The payload contains no copied
symbol table or Loom-specific substitute.

Normalization and publication are failure-atomic. Part 1 rejects the payload
before publication when parsing, full materialization, or verification fails;
when the target triple or DataLayout is absent or invalid; when the stored
canonical triple or exact DataLayout projection disagrees with the module; when
the frontend-view digest, ABI key, or bitcode digest is stale or malformed;
when bitcode or schema is newer than the reader supports; or when the input
encoding or writer settings violate this normalization contract.
Canonical-byte validation rewrites the fully materialized module through the
same production writer and requires exact byte equality with the stored
bitcode. The reader must not repair, guess, or silently fall back.

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
archive members participate. Loom collects payloads from exactly those
selected members. Every collected payload must have identical LLVM
provider/commit, canonical target triple, ABI compatibility key, and complete
frontend config view. Each exact DataLayout spelling is reparsed by that pinned
provider, and all selected modules must be structurally equal under LLVM
`DataLayout::operator==`; spelling equality is neither required nor sufficient.
Version 1.0 has no implicit config merge, precedence rule, or compatibility
lattice; structural layout disagreement is a typed link error. This preflight
is not sufficient proof that LLVM modules are link-compatible.

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

The final LLVM link is the only cross-translation-unit program merge before
Dataflow. Loom does not define a Dataflow linker, public cross-object graph
symbols, or a second library expansion mechanism. An opaque precompiled
library whose semantics cannot be exposed through LLVM Linker/LTO is deferred
until a real use requires a typed and versioned library contract.

Anchor-level tests cover:

* changing an unrelated ResolvedConfig field leaves the zero-field frontend
  view unchanged;
* one fixed known vector checks the frontend-view digest and ABI-key framing,
  and changing each ABI-key source field changes that key;
* view-digest and ABI-key tests call the production encoders rather than
  maintaining copied formulas;
* modules that differ only in use-list order produce identical normalized
  bytes, and repeated writes of one complete module produce identical bytes;
* one valid ABI-specific layout that differs from the target's ABI-unspecified
  default is accepted and preserved exactly;
* structurally equivalent DataLayout spellings retain distinct payload
  identities but pass final-link layout compatibility, while structurally
  different layouts fail that preflight;
* module flags, ABI attributes, debug information, and provenance survive
  normalization;
* semantic or module-content changes alter the bitcode digest and payload
  identity;
* stale projections, malformed payloads, unsupported versions, invalid target
  facts, and input violating the fixed writer contract fail closed;
* self-contained carrier round trip, linker-selected archive membership,
  LLVM-owned COMDAT/ODR resolution, preservation of the complete imported LLVM
  function and ABI envelope through the MLIR handoff, and absence of
  downstream hardware artifacts.

Tests do not create a matrix per compiler flag, metadata kind, or LLVM
operation. They do not freeze section names, compression, archive layout, or
LLVM internal pass structure.

Canonical semantic bytes use the field order above. Fixed digests use raw
32-byte values; strings and variable byte sequences use unsigned 64-bit
big-endian length framing followed by exact bytes. Integers use unsigned
big-endian encoding. There is no parallel JSON, MLIR, or host-struct identity
encoding.

## 8. References

* `docs/spec-compiler-part-2-scf.md` -- mechanical LLVM-to-SCF raising
  and the boundary to later structured selection.
* `docs/spec-compiler-part-3-dfg.md` -- SCF-to-DFG lowering.
* `docs/spec-executable-closure.md` -- compiler target and InstructionCore
  binary closure after final link.
