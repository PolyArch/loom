# Loom Compiler Part 2: LLVM to Initial SCF

This document defines the boundary from LLVM IR and raised standard
MLIR to Loom's initial SCF-stage MLIR. The standard `loom-raise`
pipeline is mechanical: it recovers and normalizes existing semantics
without selecting a performance-distinct program or committing
execution ownership.

The complete structured front-end path is:

```text
LLVM IR
  -> LLVM-dialect MLIR
  -> raised mixed standard-dialect MLIR
  -> initial SCF-stage MLIR S0
  -> SCF optimization and DSE
  -> selected SCF-stage MLIR Sn
  -> mechanical SCF-to-Dataflow lowering
  -> initial Canonical Dataflow Program D0
```

The SCF-stage names do not imply that the artifacts contain only the
`scf` dialect.

## 1. Mechanical Raising Scope

The standard raising pipeline owns:

* Translating LLVM IR and Loom metadata into MLIR using `func`, `llvm`,
  `cf`, `scf`, `arith`, `math`, `memref`, `ub`, and related standard
  dialects.
* Recovering structured control flow where the input semantics support
  an exact reconstruction.
* Recovering counted loops as `scf.for` where possible.
* Applying deterministic normalization needed to produce S0.
* Preserving the mixed-dialect program and provenance needed by later
  structured analysis.

The standard raising pipeline does not own:

* Promoting `scf.for` to `scf.forall` based on inferred parallel safety.
* Selecting schedules, parallel decomposition, vectorization, reduction
  strategy, memory overlap, or ownership boundaries.
* Constructing selected AccCore or SpatialCore ownership.
* Lowering structured control into Dataflow operations.

Mechanical describes decision ownership, not implementation difficulty.
LLVM raising, dialect recovery, CFG structurization, and SCF construction
may require analysis and may diagnose shapes that cannot be recovered
exactly.

## 2. Initial SCF Contract

S0 is mixed standard/SCF MLIR with no committed ownership boundary.
Operations from `func`, `arith`, `math`, `ub`, `llvm`, `cf`, `memref`,
`scf`, and other standard dialects may coexist when they preserve the
raised program semantics.

At the S0 boundary:

* `func.func` is a callable and ABI unit, not an implicit HostCore,
  AccCore, or SpatialCore placement decision.
* Candidate source metadata and provenance remain inputs to later
  structured analysis; they are not committed ownership.
* A recovered serial counted loop remains `scf.for`. The standard
  pipeline does not introduce `scf.forall` or mapping attributes
  automatically.
* No loop shape, function boundary, or local memory pattern commits an
  optimization or ownership choice by itself.

## 3. Selected Structured Program Contract

SCF optimization and DSE occur after S0 and outside `loom-raise`. Their
selected output, Sn, is an immutable Structured Program Candidate. Sn
must already materialize the selected schedule, parallel, vector,
reduction, memory-overlap, and ownership decisions before mechanical
SCF-to-Dataflow lowering begins.

Sn uses exactly these ownership carriers:

* Formal `dataflow.thread` represents a selected AccCore boundary. Its
  body is the InstructionCore stored-program and structured-control
  surface, so structured operations may remain inside it at Sn.
* Compiler-internal `loom.spatial_region`, terminated by
  `loom.spatial_yield`, represents a selected SpatialCore boundary
  within a `dataflow.thread`. It is a transparent structured ownership
  boundary: its program semantics are equivalent to inlining its body
  at the same location.

`loom.spatial_region` does not define graph firing, launch, runtime
state, mapping, or hardware configuration semantics. It exists only to
carry the already-selected SpatialCore ownership boundary through the
Structured Program Candidate.

## 4. Mechanical Dataflow Hand-Off

Mechanical SCF-to-Dataflow lowering consumes Sn without selecting a new
schedule, parallel decomposition, vector width, reduction strategy, or
ownership boundary.

The formal `dataflow.thread` ownership remains the AccCore carrier.
Each `loom.spatial_region` is mechanically outlined into the canonical
`dataflow.graph` and corresponding `dataflow.graph.launch` structure,
then erased completely. The resulting D0 is an initial Canonical
Dataflow Program and cannot contain residual `loom.spatial_region`
operations.

This lowering may reject a structure that cannot be converted exactly;
it must not repair the failure by selecting different structured or
ownership decisions.

## 5. References

* `docs/spec-compiler-part-1-source.md` -- source integration and
  metadata emission.
* `docs/spec-compiler-part-3-dfg.md` -- mechanical SCF-to-Dataflow
  lowering.
