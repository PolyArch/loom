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

Part 1 does not own:

* Deciding the final HostCore-to-AccCore boundary.
* Selecting loop schedules, parallel decomposition, vectorization,
  reduction strategy, or ownership boundaries.
* Recovering structured control flow.
* Building `dataflow.thread` or `dataflow.graph`.

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

## 7. References

* `docs/spec-compiler-part-2-scf.md` -- mechanical LLVM-to-SCF raising
  and the boundary to later structured selection.
* `docs/spec-compiler-part-3-dfg.md` -- SCF-to-DFG lowering.
