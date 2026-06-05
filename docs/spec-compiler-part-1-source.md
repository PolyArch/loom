# Loom Compiler Part 1: Source Integration

This document sketches the source-facing part of the Loom compiler
front-end. Loom's common input contract is LLVM IR plus Loom metadata.
Any high-level language can participate if its front-end can emit
semantically equivalent LLVM IR and enough metadata for the later Loom
passes to recover acceleration intent.

Part 1 does not lower code to `dataflow`. It emits LLVM IR plus
metadata that Part 2 can use when raising LLVM/CFG-shaped code to
SCF-shaped MLIR and selecting accelerator regions.

## 1. Scope

Part 1 owns:

* Accepting LLVM IR as the stable cross-language hand-off format.
* Embedding, linking, or importing source front-ends that can produce
  LLVM IR plus Loom metadata.
* Providing embedded clang as the first limited provider for C / C++.
* Preserving source intent as metadata on functions, loops, lexical
  regions, memory objects, and calls.
* Providing a stable hand-off format to Part 2, even when the source
  language does not outline accelerator code into separate functions.

Part 1 does not own:

* Deciding the final HostCore-to-AccCore boundary.
* Recovering structured control flow.
* Building `dataflow.thread` or `dataflow.graph`.

## 2. Provider Model

The Loom source integration model is language-neutral at the compiler
boundary:

* The required compiler input is LLVM IR plus Loom metadata.
* Source language support is provided by front-end providers. A
  provider may be embedded as a library, linked into the Loom binary,
  or replaced by an external producer that emits the same LLVM IR
  contract.
* C / C++ through embedded clang is the first engineering target, not
  the semantic limit of Loom.
* A provider may mark whole functions, sub-function source ranges,
  loop nests, or compiler-created regions as accelerator candidates.
  None of these source constructs is a committed AccCore boundary.

The provider contract is deliberately weaker than the later IR
contract. Part 1 preserves intent and provenance; Part 2 decides which
candidate regions are legal and profitable enough to become
`loom.acc_region`.

## 3. Metadata Classes

Part 1 may emit the following metadata classes. Part 2 is allowed to
ignore a hint, but it must not invent source intent that was not present
or proven by analysis.

* **Accelerator candidates.** Mark a function, loop, lexical compound
  statement, outlined helper, or compiler-generated code range as a
  candidate for AccCore execution. A candidate is not yet a committed
  `loom.acc_region`.
* **Parallel intent.** Mark loops or loop nests as parallel, temporal,
  spatial, vector-like, or reduction-like. These hints guide
  `scf.forall` recognition and mapping attribute construction in
  Part 2.
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
regions. The only committed accelerator boundary handed to Part 3 is
the `loom.acc_region` op produced by Part 2.

This rule keeps the source model flexible:

* A single source function may contain zero, one, or many accelerator
  regions.
* Accelerator code may be written inline in an ordinary host function.
* Scalar-only accelerator regions remain representable; they do not
  need a source-level parallel loop just to become valid input.

## 5. Initial Clang Provider

The first source provider is an embedded clang pipeline. Loom should
prefer library integration over invoking an external compiler binary so
that it can insert metadata, preserve diagnostics, and control the LLVM
pipeline.

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

Other LLVM-producing language front-ends can be added later by
implementing the provider contract. They do not need to mimic C / C++
syntax; they only need to emit equivalent metadata classes.

## 6. Hand-Off to Part 2

Part 1 emits LLVM IR plus Loom metadata. The hand-off contract is:

* LLVM IR remains semantically equivalent to the source program.
* Candidate accelerator regions are hints, not committed boundaries.
* Metadata must be stable enough for Part 2 to associate it with loops,
  memory objects, calls, and source ranges after canonical LLVM
  simplification.
* If metadata is lost or contradicted by later LLVM transforms, Part 2
  must conservatively keep the affected code on HostCore or emit a
  diagnostic when acceleration was explicitly required.

## 7. References

* `docs/spec-compiler-part-2-scf.md` -- LLVM-to-SCF raising and
  accelerator-region selection.
* `docs/spec-compiler-part-3-dfg.md` -- SCF-to-DFG lowering.
