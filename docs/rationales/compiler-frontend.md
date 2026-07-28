# Compiler Frontend Rationale

Normative contracts are owned by
[Source Integration](../spec-compiler-part-1-source.md),
[Structured Compiler Frontend](../spec-compiler-part-2-scf.md),
[SCF To DFG](../spec-compiler-part-3-dfg.md),
[Memory Frontier Lowering](../spec-compiler-part-3-mem.md),
[Logical Domains](../spec-compiler-part-4-partitioned-data.md), and
[Vector Semantics](../spec-dataflow-vectorization.md).

## Why LLVM IR Is The Language Boundary

C and C++ are the initial source languages, but the reusable compiler boundary
is LLVM IR. This admits other frontends only when they preserve the required
ABI, memory, exceptional-value, and runtime semantics. It avoids a framework
IR product boundary before a real language provider exists.

Mechanical raising preserves the source program rather than choosing an
accelerator implementation. `llvm.func` remains the LLVM ABI envelope because
it already owns linkage, calling convention, COMDAT, personality, attributes,
target features, and floating environment. Reconstructing those facts in
`func.func` or a Loom contract would create a second LLVM ABI schema. Region-
level CFG structurization can still introduce SCF inside that envelope.

Region structuring can retain two PHIs that are the same recurrence under
different source SSA names. Keeping both creates fictitious loop state and can
hide that a raw pointer is only an invariant capability plus an offset. Loom
therefore quotients only the exact same-initial-value, same-next-value,
identity-feedback shape. This narrow rule removes accidental CFG structure
without introducing a general equivalence engine or turning canonicalization
into an optimization decision.

LLVM operations are normalized to registered canonical compute schemas only
when exact semantics are preserved. A familiar symbol name is not sufficient.
In particular, `fmuladd` cannot be assumed to be fused; poison, undef, freeze,
overflow, exactness, fast-math, rounding, and predicates remain explicit. A
single generated OperationSchema projection prevents frontend, Dataflow,
simulator, Fabric, and backend from maintaining separate name tables.

The LLVM importer uses one `passthrough` array for every function attribute it
does not expose as a typed LLVMFuncOp field. Treating that storage container as
the floating environment blocks ordinary Clang programs for unrelated facts
such as recursion, synchronization, stack protection, or target selection.
The exact-spelling owner therefore classifies the contained LLVM attributes,
keeps unknown string semantics conservative, and lets only facts that can
change floating execution block normalization. This preserves the ABI envelope
without making ownership materialization a second compute-spelling authority.

Scalable vectors remain structured candidates until a fixed chunk, loop, and
mask materialization is selected. A fixed-width SpatialCore cannot infer a
runtime `vscale` contract.

## Why Source Hints Are Nonbinding

An optional source pragma can identify a region worth considering, but it
cannot promise legality, ownership, profitability, or hardware support. Those
facts depend on whole-program LLVM semantics, structured analyses, exact
Fabric input, and Evaluation. Treating the pragma as a command would either
override those owners or require silent fallback when the promise cannot be
kept.

Loom therefore preserves the hint through ordinary Clang diagnostics and
LLVM/MLIR provenance, then lets candidate generation consume it as one input.
Optimization remarks and locations explain the eventual decision. A global
hint database or sidecar provenance artifact would duplicate mature compiler
infrastructure without changing program semantics.

## Why Relocatable Payloads Exist

Drop-in compilation must survive ordinary separate compilation and archive
selection. A relocatable accelerator payload carries normalized LLVM semantics
through object and archive files; final link collects only linker-selected
members and delegates COMDAT, ODR, module flags, and symbol resolution to LLVM
Linker/LTO.

The payload does not contain Fabric, Mapping, configuration, or Deployment
choices because those occur after whole-program link. Its initial frontend
configuration view is explicitly empty: the final LLVM module already owns
the relevant language and ABI semantics, while copying compiler flags would
invent cross-translation-unit merge rules. Digests and ABI keys are checked
projections, not replacements for the bitcode and LLVM semantics.

## Why SCF Is The Main Optimization Surface

Loops, dependence, aliasing, reduction algebra, sparse access, and ownership
are most visible before conversion to an explicit dataflow graph. Parallelism,
tiling, fusion, fission, interchange, outer and inner unroll, unroll-and-jam,
vectorization, reduction strategy, memory movement, overlap, and ownership
cuts are therefore Structured Program Candidate decisions.

These choices are not independent pass switches. A vector factor changes
memory geometry and tail policy; unroll-and-jam changes reuse and resource
pressure; parallel dimensions affect thread ownership and system transport.
The compiler generates immutable candidates, uses typed analyses for legality,
and asks the central Evaluation/DSE framework to compare legal alternatives,
optionally against an exact Fabric target.

## Why Structured Candidates Are Artifacts

Structured optimization and hardware-aware Evaluation compare complete program
states, not mutable pass-manager snapshots. An immutable Structured Program
Artifact gives lineage, cache, replay, and Evaluation one exact subject while
allowing hot search to keep removable drafts and deltas. Separate initial,
optimized, and selected families would encode workflow state as program
semantics, so every published candidate uses the same family and schema.

The candidate stores the complete mixed-dialect `builtin.module` because LLVM
ABI, structured control, standard operations, and selected ownership decisions
must remain one coherent program. A separate Schedule IR, analysis Artifact, or
optimization plan would duplicate facts already materialized in that module.
Derived analyses therefore remain recomputable views.

The canonical payload is family-owned deterministic MLIR bytecode rather than
ordinary printer output. This keeps the expressive MLIR type, region, symbol,
and operation model without introducing a parallel record schema. The family
writer fixes semantic inclusion and normalization; Common only frames and
hashes the resulting bytes. A generic bytecode command is not an identity
authority.

Parent-local structural references deliberately remain one closed reference
shape. They identify a canonical operation, region, block, or value in one
exact parent without adding permanent IDs to mutable MLIR. Actual loop,
operation, memory, and transform semantics stay with their existing owners.

Source hints do not enter candidate identity. They may influence which typed
decision is explored, but equal resulting program semantics must deduplicate.
Making the raw hint semantic would let a nonbinding compiler suggestion create
distinct programs even when it changed no program decision.

## Why Fabric Precedes Structured Optimization

Structured optimization cannot choose useful unroll, vector, ownership,
memory, communication, or topology-sensitive transformations from software
syntax alone. Loom therefore resolves the exact Fabric before producing the
first Structured candidate. Cheap capability projections reject proved
impossibilities, central Evaluation compares legal alternatives, and selected
survivors may be promoted through Dataflow and Mapping for higher-fidelity
feedback.

Fabric identity is not embedded into every Structured or Dataflow Artifact.
Those Artifacts own software semantics and should deduplicate across hardware
when their semantics are equal. Target facts that actually change software
semantics are materialized in the candidate; resource and topology lineage is
carried by the exact EvaluationRequest, DSE use-def edge, and Mapping. This
keeps the hardware dependency explicit without creating a second target
authority inside software IR.

Address-index width illustrates this boundary. Fabric capability may suggest
which widths are worth exploring, but the selected width changes the typed
software representation and therefore belongs in the Structured candidate.
Requiring one explicit decision for every Spatial ownership shape avoids an
ambient default and lets proof-backed narrowing reject only the infeasible
candidate rather than silently truncating source addresses.

The same ownership explains pointer induction. A pointer updated on every
iteration looks like loop state in imperative SSA, but its storage authority
does not change: only an offset from one static capability changes. This remains
true when the fixed per-iteration stride is a runtime value invariant within the
loop. Keeping the base invariant and carrying the typed element offset preserves
that fact, gives memory lowering an exact address relation, and avoids adding
dynamic raw pointer semantics to canonical Dataflow. Shapes whose finite offset
domain or element units cannot be proved remain ordinary InstructionCore
candidates.

A persistent Schedule IR, Placement IR, or generic action DSL was rejected.
Loop structure and transformations already live in the candidate IR;
dependence and logical-domain models are derived analyses; physical binding
belongs to Mapping. Persisting all three again would create competing program
states.

## Why Ownership Uses One Temporary Boundary

AccCore ownership is represented by the real non-recursive `dataflow.thread`
surface. SpatialCore ownership needs a structured temporary boundary because
the body still contains SCF and imperative memory semantics before graph
lowering. The compiler-internal `loom.spatial_region` is therefore transparent
and isolated, with explicit value, channel, and memory boundaries.

It is not an early graph, launch, or Mapping record. It has no runtime state or
hardware identity and cannot hide effects. Reusing canonical `dataflow.graph`
would pollute Dataflow with residual SCF; using an attribute on a generic
region would not enforce explicit capture and boundary semantics.

Finalization is whole-program and failure-atomic. Unsupported selected regions
do not silently fall back to the InstructionCore during lowering. The DSE must
select a different Structured Program Candidate if it wants a different
ownership cut.

## Why SCF-To-Dataflow Is Mechanical

Once the candidate fixes schedule, shape, reduction, and ownership, lowering
has one job: make causal, state, value, channel, and memory relations explicit.
It recursively converts structured control, removes the temporary boundary,
publishes canonical graphs, and rejects residual imperative control. It does
not choose transformation factors or repair an infeasible hardware mapping.

The recursive memory-frontier transfer is compiler-local, not another IR. It
lets nested `for`, `while`, selection, and parallel constructs compose without
an A-by-B rule matrix, then emits ordinary Dataflow control and memory-event
edges. This keeps source order semantics while allowing the canonical graph to
be text-order independent.

## Why Dataflow Has Its Own Optimization Lineage

The earlier assumption that every optimization could happen in SCF was too
strong. Two canonical Dataflow programs can be functionally equivalent but
have different actor topology, synchronization structure, fanout, or
backpressure behavior. Typed Dataflow rewrites therefore form a second
immutable candidate lineage after mechanical lowering.

These rewrites may simplify actor networks, reshape synchronization, or remove
provably redundant structure. They cannot revisit loop scheduling, reduction
algebra, vector factors, or ownership. Functional equivalence preserves values,
streams, memory effects, externally visible ordering, and abstract liveness;
it need not preserve internal actor traces or hardware cycle count.

## Why Structured Vectorization Is Primary

Vector computation is selected while loop and memory structure are available.
Lowering a regular vector loop to scalar streams and rebuilding it with a
grouping operation would lose alignment, masks, reduction order, and memory
access intent. Structured vectorization therefore produces standard vector
types and vector memory operations before graph lowering.

`parallelize` and `serialize` remain local stream-cardinality adapters.
`pack` and `unpack` represent source-visible bit interpretation. Physical bus
width adaptation belongs to Mapping and Fabric. These distinctions prevent a
packed integer from becoming Loom's vector type system.

## Why Representative Kernels Are Structural Anchors

The frontend anchors were selected to force interaction among decisions:
dense vector loops, nested reductions, sparse indirection, irregular control,
DSP reuse, stencils, convolution, and multi-stage attention streaming. They
protect real source-to-Dataflow behavior without creating one fixture for
every pass combination or redefining corpus membership.
