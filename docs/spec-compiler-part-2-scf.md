# Loom Compiler Part 2: LLVM to SCF

This document sketches the middle part of the Loom compiler front-end.
Its job is to raise LLVM/CFG-shaped input into SCF-shaped MLIR, choose
which structured regions run on AccCores, and hand explicit accelerator
regions to Part 3.

Part 2 is where HostCore-vs-AccCore selection lives. Part 3 must not
infer this boundary from `func.func`.

## 1. Scope

Part 2 owns:

* Translating LLVM IR and Loom metadata into MLIR using `func`, `llvm`,
  `scf`, `arith`, `math`, `memref`, `ub`, and related dialects.
* Selecting accelerator regions from source metadata, loop analysis,
  legality checks, and optional cost-model decisions.
* Recovering structured control flow where possible.
* Recognizing parallel loops and lowering them to `scf.forall` with
  mapping attributes.
* Recognizing memory regions, including host-visible buffers,
  AccCore-local scratch candidates, and SpatialCore layout candidates.
* Emitting `loom.acc_region` as the only committed accelerator
  selection boundary consumed by Part 3.

Part 2 does not own:

* Lowering structured control flow into dataflow primitives.
* Building `dataflow.thread` or `dataflow.graph`.
* Treating a whole `func.func` as an accelerator region unless source
  metadata, user options, or analysis explicitly select it.

## 2. Placement Model

Part 2 defines the relationship between ordinary functions, temporary
accelerator regions, and final dataflow placement:

* `func.func` is a callable symbol and ABI unit. It does not imply
  HostCore or AccCore placement by itself. A `func.func` may be used as
  a HostCore function, a ScalarCore-callable function, or both if it
  satisfies both legality contracts.
* `loom.acc_region` is the temporary Part 2 to Part 3 marker saying
  that a structured region has been selected for AccCore execution. It
  is not a final runtime or hardware primitive.
* `dataflow.thread` is the final AccCore execution boundary produced
  by Part 3. It carries launch operands, mapping, memory-transfer
  information, and completion tokens.
* `dataflow.graph` is the SpatialCore leaf DFG produced by Part 3. It
  cannot contain function definitions or calls.

Function calls are ScalarCore-level operations when they appear inside
an accelerator region. Part 2 must classify any `func.call` reachable
from a `loom.acc_region`: the callee must be ScalarCore-legal,
inlined before Part 3, or rejected with a diagnostic. A ScalarCore-legal
callee may itself contain `loom.acc_region` markers if Part 2 selected
nested work for AccCore execution; Part 3 later lowers those markers to
nested `dataflow.thread` launches.

`func.func` symbols remain in the nearest module-level symbol table in
the first design. `dataflow.thread` is not made a symbol table merely to
host local function definitions.

## 3. `loom.acc_region`

`loom.acc_region` is a temporary Loom front-end op. It exists only
between Part 2 and Part 3.

```
arguments:
  Variadic<AnyType>:$boundaryOperands;
attributes:
  DictionaryAttr:$boundary;
  OptionalAttr<DeviceMappingArrayAttr>:$defaultMapping;
results:
  none;
regions:
  SizedRegion<1>:$body;
traits:
  IsolatedFromAbove,
  SingleBlockImplicitTerminator<"AccRegionYieldOp">,
  MemoryEffectOpInterface.
```

The operation means: execute this structured region on AccCore. It is
not tied to a function boundary. A single `func.func` may contain
multiple `loom.acc_region` ops, and ordinary host code may appear before,
between, and after them.

The body may contain `func.call` operations. Those calls are interpreted
as ScalarCore calls after Part 3 lowers the enclosing accelerator
region. They are not candidates for `dataflow.graph` extraction unless
the callee has been inlined or specialized first.

The op has no direct data results. Values produced by AccCore execution
that must be observed by HostCore are represented through explicit
memory effects. Scalar results that must escape are materialized into
temporary memory before the region is committed, or the region is
rejected with a diagnostic. The `loom.acc_region.yield` terminator
therefore has no operands.

`boundaryOperands` and the entry block arguments are positionally
matched. The body is isolated from above, so all scalar values, memrefs,
and layout handles crossing into the region are explicit. The `boundary`
attribute records direction, alias, layout, and diagnostic metadata in a
form Part 3 can lower to `dataflow.map_info` and thread operands.

`defaultMapping` is used only when Part 3 must normalize a scalar-only
accelerator region into a 1x1 mapped `scf.forall`. Parallel structure
inside the region should still be represented directly by mapped
`scf.forall` whenever it exists.

## 4. Region Selection

Part 2 commits a region to AccCore only after legality checks succeed.
The selected region must be single-entry/single-exit at the MLIR level
and structured enough that Part 3 can reason about its control flow.

Region selection may use:

* Explicit source annotations from Part 1.
* Loop and dependence analysis.
* Recognition of parallel loop nests.
* Memory locality and scratch-memory opportunities.
* User options that require or forbid acceleration for a candidate.
* Cost-model decisions.

If a source annotation requests acceleration but the selected code cannot
be represented as `loom.acc_region`, Part 2 emits a diagnostic. It must
not silently widen the boundary to the entire enclosing `func.func`
unless that exact boundary was explicitly selected.

## 5. Structured Control Raising

Part 2 converts supported LLVM/CFG shapes into structured MLIR:

* Reducible branches become `scf.if`, `scf.index_switch`, or structured
  loops.
* Counted loops become `scf.for` when induction, bounds, and step can be
  represented.
* Parallel loops become `scf.forall` with mapping attributes.
* Irreducible or unsupported control remains outside `loom.acc_region`
  unless an explicit user request requires a diagnostic.

The output may still contain dialects such as `arith`, `math`, `memref`,
`ub`, `llvm`, and `func`. Part 3 only requires that code inside
`loom.acc_region` use structured control forms for the operations it
lowers to DFG.

## 6. Function Call Legality

Part 2 assigns each reachable `func.func` one or more call-context
classes:

* **HostCore-callable.** May be called from ordinary host code.
* **ScalarCore-callable.** May be called from code that Part 3 will
  place inside a `dataflow.thread`.
* **Inline-only.** Must be inlined before it can appear in a selected
  accelerator region.
* **Host-only.** Must not be reachable from a `loom.acc_region`.

The first milestone may be conservative and require calls inside
`loom.acc_region` to be inlined or explicitly marked
ScalarCore-callable. Unsupported indirect calls, recursion, exceptions,
unstructured stack behavior, or host-only runtime calls cause a
diagnostic if they are reachable from an accelerator region.

## 7. Memory Region Analysis

Part 2 classifies memory crossing an accelerator boundary:

* Host-visible input, output, and inout buffers.
* Read-only scalar and aggregate launch operands.
* AccCore-local scratch candidates.
* Static local storage candidates.
* Spatial layout candidates that may become `dataflow.spatial_layout`
  annotations specified in Part 4 (see
  `docs/spec-compiler-part-4-spatial.md`).

This analysis is a boundary contract, not a replacement for the Part 3
memory-dependence builder. Part 3 still constructs conservative
dependence edges inside each `dataflow.graph`.

## 8. Hand-Off to Part 3

The Part 2 output contract is:

* AccCore-selected code is inside `loom.acc_region`.
* Host code outside `loom.acc_region` remains host code.
* No `func.func` is an implicit accelerator boundary.
* `func.call` operations reachable from accelerator regions are either
  ScalarCore-legal or scheduled for inlining before Part 3 graph
  extraction.
* Accelerator regions have explicit operands, no direct data results,
  and enough memory-effect metadata for Part 3 to insert
  `dataflow.map_info`.
* Mapped parallelism is represented with `scf.forall` mapping
  attributes.
* Scalar-only accelerator regions are legal and may rely on
  `defaultMapping` for Part 3 normalization.

## 9. References

* `docs/spec-compiler-part-1-source.md` -- source integration and
  metadata emission.
* `docs/spec-compiler-part-3-dfg.md` -- SCF-to-DFG lowering.
