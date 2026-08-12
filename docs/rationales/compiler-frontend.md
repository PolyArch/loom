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

LLVM assigns loop metadata through latch terminators, not arbitrary branches
inside a loop. Optimizers can leave an orphan attachment after reshaping a CFG;
such an attachment no longer proves any loop fact. Letting it block the whole
callable would hide independent structured candidates, while attaching it to a
nearby loop would invent a fact. Loom therefore discards only the orphan on a
successfully structured clone and keeps the analysis fact unknown. Exact and
ambiguous backedge owners remain subject to the stricter preserve-or-migrate
rules.

The same structuring utility can thread exit constants and exceptional-value
placeholders through several adjacent loops as publication latches. When the
failed condition publishes an exact value that already dominates the loop,
retaining the loop result creates a false cross-loop dependence. Projecting
that result back to the same SSA value removes the accidental state without
moving the loop, changing termination, or weakening poison and undef semantics.

## Why Thread Parallelism Uses Coordinate Tuples

A scalar thread ID would impose a linear order and flattening convention on
every logical domain. That would duplicate the extents already carried by a
dense launch, obscure multidimensional schedules, and tempt consumers to
equate logical order with physical topology. Loom therefore gives the thread
body the exact coordinate tuple selected by the Structured candidate. Linear
IDs, tiles, and source induction variables are ordinary explicit arithmetic
when the program needs them.

Nested scheduling remains one recursive structured transformation rather than
a hierarchy of special thread kinds. The candidate projects outer AccCore
parallelism, graph-static work, and graph-temporal work into explicit `L`,
`S`, and `T` coordinates and structure. These projections are consumed by the
existing thread and graph ABIs and disappear as independent analyses. This
keeps the selected Structured Program as the only schedule authority.

LLVM operations are normalized to registered canonical compute schemas only
when exact semantics are preserved. A familiar symbol name is not sufficient.
In particular, `fmuladd` cannot be assumed to be fused; poison, undef, freeze,
overflow, exactness, fast-math, rounding, and predicates remain explicit. A
single generated OperationSchema projection prevents frontend, Dataflow,
simulator, Fabric, and backend from maintaining separate name tables.

Some native operations, notably `llvm.call_intrinsic`, are generic carriers
rather than one semantic operation. Rewriting every irreducible intrinsic as
a new Dataflow operation would duplicate the source owner's semantics, while
registering the carrier wholesale would admit unrelated intrinsics. Finite,
disjoint typed instance selectors preserve the exact LLVM operation and use
LLVM's own intrinsic registry as the semantic authority. Loom then owns only
the stable typed projection needed by downstream components.

Source-backed execution and region extraction need finite call lineage, while
ordinary C harnesses often pass a compile-time constant callback through one
internal dispatcher. Treating that call as permanently opaque loses a fact
LLVM can prove; running a whole interprocedural optimizer on the production
module would instead move unrelated optimization choices ahead of SCF. Loom
therefore runs pinned LLVM constant propagation on a disposable proof clone
and imports only exact, signature-preserving callee resolution. This reuses
LLVM as the proof authority without copying its analysis or admitting its
unrelated rewrites into the mechanical program boundary.

A dispatcher used by several call sites may have no single module-global
callback even though each site passes an exact function constant. Choosing one
target would be wrong, while extending source validation with a second
indirect-call identity model would duplicate call authority. Loom instead
creates one internal dispatcher specialization per distinct exact binding.
Each original call keeps the dispatcher control semantics but reaches a clone
whose callback edge is direct; unknown calls continue to use the original
dispatcher. Applying the same deterministic transform to target and native
oracle modules preserves one direct-call lineage model without a harness-name
special case.

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

Unroll-and-jam is atomic only because the upstream transform changes outer
replication and inner-loop fusion as one semantics-preserving operation. Loom
does not represent it as `unroll` plus a persistent `jam` switch: the latter
would be a second schedule authority and could describe combinations that were
never materialized. The child Structured Program itself shows whether inner
control is shared, while exact dependence and Fabric-capacity projections bound
which children may be generated.

LLVM bulk-memory intrinsics are expanded to exact structured loop semantics
before ownership selection. This exposes direction, bounds, volatile behavior,
and tail handling where vector width, chunking, staging, and unroll decisions
belong. Retaining a second bulk operation in canonical Dataflow would hide
those choices and invent semantics that no Fabric capability necessarily
provides; an explicit DMA-like actor is justified only when a future typed
hardware capability owns it.

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

Schedule factors are derived from the exact loop rather than copied from a
global tuning table. For the Schedule generator over
`loom.structured_program 1.0`, proper divisors give a compact, deterministic
domain with no cleanup tail; aggregate typed Fabric occurrence counts remove
only replication factors already proved impossible. Adjacent interchange is
the primitive relation because repeated lineage edges can express any legal
permutation without a factorial one-invocation domain. These choices keep
schedule semantics in the transformed Structured Program and leave quality,
placement, routing, and contention to their existing Evaluation and Mapping
owners.

Direct-call constant specialization follows the same rule. Treating each
formal as an independent switch would create an argument-subset powerset and a
second call-profile authority. The SCF specification therefore defines one
all-bindings decision derived from the exact linked call graph. Candidate-local
substitution and simplification expose a smaller dependency-closed region while
preserving the original callable ABI; unknown or conflicting callers simply
leave that choice absent. Dynamic workload coverage remains useful for ranking,
but cannot justify changing code that is reachable under another legal input.

Direct-call inlining also needs an exact coordinate rather than a symbol-name
policy. Rejecting every callable that contains a call hides a legal candidate
before central DSE can account for it, while silently inlining every reachable
callee makes the transformed program depend on an unrecorded compiler choice.
The smallest complete coordinate is therefore the parent-local call-site
reference plus the ordinary child Structured Program produced by the pinned
inliner. The no-inline coordinate remains observable and fails normally when
it cannot form a closed Spatial region.

The call site may also be the selected ownership scope when it is an exact
inlineable direct leaf. Consecutive producer and consumer stage calls otherwise
have no independent structured-region roots: requiring source authors to wrap
each call in a one-trip loop or constant branch would make accidental control
syntax an ownership prerequisite. Admitting only the already-owned exact call
coordinate removes that accident without admitting arbitrary regionless
operations. The private materializer may use an exact-once boundary to derive
the post-inline SSA closure, but that boundary disappears before publication;
the existing decision, call-site reference, child Structured Program, and
lineage remain the only authorities.

Nested structured scopes need the same coordinate. Treating `llvm.call` as a
generally supported graph leaf would weaken the finalizer, while rejecting the
scope before decision enumeration would hide the only transformation that can
remove it. Deferring exactly the selected call during preflight preserves both
properties. The inliner can also clone nested region blocks, so its ordinary
`IRMapping` is the only sound source for validating structural block
correspondence; reconstructing that relation from block position would create
a second identity authority. It is not, however, an exact dynamic-activity
projection. A callee used by two callers has aggregate source block counts,
whereas inlining one call splits those counts between the retained callee and
the clone. Executing the exact candidate is the smallest correct activity
owner, so direct-call children require its removable native observations and
publish no misleading block lineage. An ordinary inliner legality refusal
prunes one coordinate, but verifier or lineage failures indicate broken
implementation invariants and therefore cannot be reported as normal search
outcomes.

Candidate failure classification must be owned where the distinction is
known. A caller cannot tell whether an arbitrary lowering error means "this
legal coordinate is not expressible" or "the implementation lost a tracked
block." The address normalizer therefore exposes a narrow typed proof
rejection, while graph structural preflight rejects unsupported leaves before
the final transaction. All remaining errors stay untouched. This avoids both
string-based dispatch and the tempting but incorrect rule that every failed
transform is a normal search result.

The initial atomic form is admitted only when one direct leaf call closes the
selected scope. Enumerating arbitrary call subsets would create an exponential
domain and duplicate program structure in the decision record. Larger call
graphs instead progress through ordinary Structured children, after which the
same ownership analysis is recomputed. Requirements newly visible through the
callee, such as address-index width, are projected from that exact body into
the same finite decision domain. They are not inferred after selection and are
not stored in a separate call summary.

Preflighting a callee in its original function context is too early. A fresh
allocation at the top of a helper is not at a graph frontier there, but it is
at the prospective frontier after the exact call is inlined into a selected
callable. Conversely, an unregistered intrinsic remains unsupported after
inlining. Applying the same lowering-owned proof to the transformed private
candidate preserves both cases without a callee summary or a second table of
context-sensitive exceptions.

Candidate rejection must remain as observable invocation provenance. Dropping
a whole callable because it contains an unresolved call, or dropping one
address-width choice because narrowing cannot be proved, makes a graph-free
result indistinguishable from an unexplored domain. One canonical disposition
sequence preserves that evidence without creating another persistent program
or diagnostic owner: successful coordinates reference ordinary child
Artifacts, expected failures retain their typed compiler reason, and central
Evaluation still deduplicates equal children by Artifact identity.

Pointer induction often admits a cheaper rooted form: a pointer updated on
every iteration may be represented by one invariant capability and a carried
integer offset. This remains true when the fixed per-iteration stride is a
runtime value invariant within the loop. Keeping that form preserves alias and
bounds facts and usually maps to less hardware.

It is not a semantic restriction. Descriptor-loaded pointers, pointer cursors,
and pointers stored as ordinary data cannot always be reduced to one static
root without changing the program. Canonical Dataflow therefore also admits
first-class LLVM pointer values when exact DataLayout, OperationSchema, Fabric,
and provider contracts exist. DSE chooses between the rooted form, pointer
execution in a SpatialCore, and InstructionCore ownership; no frontend rule
forces all address representation into one execution owner.

The service boundary remains distinct from the pointer value. For example, a
descriptor load can produce pointer `p`, while a later `load p` needs both `p`
and a service that can resolve its target object. When `p` is produced inside a
larger candidate, that service is not automatically invented from the pointer
bits. Selecting the inner loop makes `p` an explicit live-in and gives the
existing graph-memory owner a precise service source. Reusing one pointer-root
resolver for this candidate proof and for final lowering prevents the host
prelude heuristic from becoming a second memory model; it also leaves DSE free
to select a future Fabric that admits the larger pointer-processing cut.

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

`dataflow.thread` deliberately has no data-result ABI. A rank-zero extracted
region may nevertheless produce ordinary values: its graph returns them to
the thread's InstructionCore continuation, which writes caller-owned result
storage before completing. Extending thread launch with scalar results would
look simpler for one launch, but would introduce a second aggregation contract
for multi-instance thread domains. Reusing the ordinary stored-program memory
boundary keeps one thread completion model and leaves aggregation explicit.

Finalization is whole-program and failure-atomic. Unsupported selected regions
do not silently fall back to the InstructionCore during lowering. The DSE must
select a different Structured Program Candidate if it wants a different
ownership cut.

## Why Ownership Ranking Is Workload-Aware

Static graph size cannot distinguish a hot compute loop from a cold helper.
Ranking only the selected region can therefore prefer a tiny shape query or
descriptor copy even though accelerating it has no whole-program benefit. A
benchmark-name rule or minimum coverage percentage would hide that modeling
error behind another policy.

The source workload is evaluated once through the central Evaluation system.
Its typed dynamic projection weights each dependency-closed candidate while
the ordinary whole-workload cost accounts for remaining host work, launch and
transfer overhead, and Fabric-constrained Spatial work. This makes a helper
lose for the correct reason: its saved dynamic work does not repay its
boundary cost. The profile is not a second program or workload Artifact, and
the selected schedule and thread domain remain materialized only in the child
Structured Program.

Functional replay executes the complete selected program and is materially
more expensive than the analytical pre-Mapping model. Replaying every candidate
that merely beats the host baseline would undermine the early pruning that the
model exists to provide. Loom therefore validates the best-ranked prefix
and deterministically admits the next-ranked candidates only when a completed
functional mismatch removes a survivor. This retains the result of an eager
filter-then-rank implementation while avoiding replay work for candidates that
cannot enter the resolved `TopK` result.

The same projection excludes a scope with zero dynamic activations before the
compiler clones or publishes a candidate. Materializing it first would consume
artifact I/O and lowering work only to prove an already known applicability
fact. Treating it as a candidate rejection would also be misleading: the scope
may be legal and profitable for a different workload. Keeping it outside the
exact workload-specific Generate domain preserves total accounting without
inventing a cold-scope status or a second profile record.

Eagerly materializing every active callable and every nested structured region
also duplicates large complete program snapshots before any useful comparison
can occur. A wall-time cutoff or worker-count-dependent batch would make the
formal candidate set machine-dependent. The generator instead derives one
scope hierarchy, admits children only after their parent reaches the frontier,
and spends a resolved semantic number of complete scope expansions on the most
dynamically relevant work first. This preserves exact scope-local alternatives,
keeps cache hits and misses equivalent, and bounds artifact publication without
inventing benchmark rules or treating unexplored descendants as failures.

The common workload is rooted in the source Structured Program rather than in
any candidate graph. Otherwise each ownership candidate would change both the
program and the workload key, making source equivalence and whole-workload
ranking circular. One source workload/runtime-input pair can instead feed the
unmodified baseline and every Sn candidate; only exact graph-replay inputs are
derived after lowering. This is also why corpus harnesses are workload
providers, not a separate testing semantics.

QoR selection and feasibility composition deliberately treat multiple
protocol roots differently. Greedily composing `BenefitQualified` winners
would erase the alternatives needed for a global objective. A
`SemanticConformance` request instead needs one program containing every
explicit operator boundary. Chaining one `TopK(1)` child per root produces that
closure without a powerset, and the root count already provides a finite bound.
Adding a second generation-limit knob would describe no new semantic fact and
could silently disagree with the requested root set.

## Why Schedule Admission Follows Materialization

Aggregate body capacity is a useful early bound for unroll, but it cannot
predict every actor shape created by a legal SCF transform. Tiling can add an
inner control frontier and widen a `dataflow.sync` even when each source body
actor has enough aggregate occurrences. Letting that child enter Evaluation
would turn a known exact-Fabric hard negative into `Unsupported` Evidence and
make the entire promotion incomplete.

Parallelization reuses one conservative legality and materialization owner.
Having the pass pipeline and the Schedule generator maintain separate alias or
effect rules would allow one path to create a child that the other path rejects.
The generated child therefore contains an ordinary `scf.forall`, which is the
only schedule fact needed at this point. Physical coordinates and bindings are
deliberately absent: preserving graph-local parallelism or distributing the
logical domain across thread launches is an ownership and later Mapping choice,
not another hidden parallelization hint.

The Schedule generator therefore applies two complementary checks owned by the
same Fabric capability projection: cheap aggregate pruning before cloning, and
complete actor admission after the transformed child is mechanically lowered.
The second check excludes only children with no concrete capability. It does
not approximate Mapping feasibility. Retaining the resulting D0 in an
invocation-local reference-keyed cache avoids a second lowering for analytical
Evaluation and functional replay without creating another program authority.

An empty candidate set is an ordinary finite result rather than a provider
failure. This lets independently composable Generate nodes remain total while
preserving the distinction between no candidates, malformed input, and missing
Evidence.

## Why Execution Shape Is Separate From Ownership

Ownership answers which dependency-closed region belongs to a SpatialCore;
FMA shape answers how one already selected arithmetic semantic choice is
materialized. Combining both domains makes every address, call, thread-domain,
and ownership alternative multiply by the Fused/Split choice before either
owner can prune its own decisions. It also lets a region-selection API acquire
an accidental arithmetic default.

Keeping ExecutionShape as the next independent Generate owner removes that
Cartesian coupling. Ownership publishes one complete immutable Structured
candidate, ExecutionShape produces only the two uniform semantic policies when
needed, and downstream Schedule sees only shape-closed candidates. Applying
one policy to the selected ownership rather than one Boolean per operation
keeps the domain finite without a hidden heuristic: Fused and Split remain
ordinary candidates whose whole-workload Evidence can distinguish their
resource use and numerical behavior.

The invocation retains the just-materialized typed candidate because throwing
it away and immediately importing the same Artifact repeats canonical parsing
and loses removable activity lineage. This retained object is not another
program record: its key is the exact Artifact reference, its bytes must equal
the published object, and deleting it only causes deterministic reconstruction.
The architecture therefore keeps one persistent authority while avoiding work
at the publication boundary.

## Why Special-Math Accuracy Is Selected Before Dataflow

The source `afn` flag says approximation is permitted; it does not say whether
the application accepts one, two, four, or an unbounded number of ULP. Treating
`afn` as a default accuracy would turn one permission bit into an undocumented
policy. Letting an RTL recipe choose the bound would be worse: identical
Dataflow and Fabric identities could then mean different numerical programs.

The selected accepted maximum is therefore a Structured candidate decision.
This is the last optimization surface where the compiler can compare complete
program alternatives before publishing D0, and it lets central Evaluation
trade numerical quality against resource and performance evidence without
changing the source or hiding a backend choice. An operation without `afn`
closes mechanically to correctly rounded, so the new owner creates no useless
branch for strict programs.

Accuracy is separate from ExecutionShape because the two decisions answer
different questions. FMA Fused versus Split changes the operation graph;
special-math accuracy constrains the legal result set of one unchanged actor.
Combining them would couple unrelated candidate domains and make every future
shape decision carry numerical-policy fields. Delaying accuracy to D0-to-D*
was also rejected because that lineage is semantics-preserving: weakening a
correctly-rounded actor after publication would silently change its contract.
Making SpecialMathAccuracy the first selected-Spatial D0 and exact-Fabric
admission gate keeps those decisions separate while giving unresolved semantic
state one closure owner. ExecutionShape publishes only its complete Structured
children; it does not create a partial Dataflow projection or duplicate target
admission before the accuracy contract exists.

One four-element ordered domain is sufficient. It permits exact conformance and
the practical one-, two-, and four-ULP research points without a floating
tolerance property bag. The selected actor allowance and the Fabric circuit
guarantee use the same owner-coded order, so admission is one refinement check
rather than a backend-specific reconciliation table.

## Why Constant Staging Starts At The Spatial Memory Boundary

A constant table is globally named in the enclosing program, but a canonical
Spatial graph consumes an explicit memory capability. Moving
`memref.get_global` into the graph would make a symbol lookup into executable
Spatial semantics and create a memory root that neither Dataflow nor Fabric
owns. Keeping the lookup outside while staging the exact region memory input
preserves the existing boundary: thread launch selects the logical object, and
the graph only performs explicit memory work on the supplied capability.

The first MemoryCommunication decision deliberately accepts only direct loads
from a constant global with an explicit elements initializer at every root
launch. This narrow proof is more general than a symbol-name rule and safer
than a broad read-only guess. If the region stores through the same argument or
derives another use from it, redirecting only its loads could change
read-after-write behavior. Rejecting every such unknown use lets later alias
and effect analysis enlarge the legal domain without a compatibility path or a
second notion of constness. A sibling view of the same constant global does
not need a synthetic `noalias` proof because writing that source object is
already undefined; staging only snapshots its defined immutable contents.

Preserving an explicit load alignment requires more than copying its number to
the new allocation. The allocation base and the load's exact byte offset must
together establish the effective-address alignment. Constant identity-layout
indices permit that proof; a dynamic or misaligned offset does not. The new
logical allocation therefore carries the strongest proved base alignment and
each redirected load retains its own contract unchanged. These facts are
derived from ordinary Structured IR, not copied into generator configuration
or a separate memory schema.

The staging buffer is logical Structured IR rather than a physical memory
choice. Its copy becomes ordinary Dataflow load/store work, so Fabric
capability admission, Mapping, and Evaluation can decide whether any concrete
memory realization is feasible and worthwhile. This keeps software
transformation, memory hardware, and placement as three separate owners while
allowing central DSE to compare the complete immutable alternatives.

## Why Memory Communication Uses Four Closed Transforms

Staging alone leaves three software distinctions that materially change the
program presented to Mapping: logical storage order, overlap between a staged
copy and its consumer, and whether a proved point-to-point temporary is memory
or a stream. These are not physical memory choices. Deferring them to Fabric or
Mapping would make hardware placement silently rewrite the software program;
encoding them in a generic memory-plan record would create a second authority
beside ordinary Structured and Dataflow IR.

The closed four-kind catalog is the smallest current surface that owns those
distinctions. `StageConstantGlobal` introduces an explicit logical buffer,
`PermuteLocalBufferLayout` changes only its dense storage order,
`PipelineStagedLoop` changes only a proved two-stage logical schedule, and
`PromoteOrderedBufferToChannel` replaces one proved ordered temporary with the
canonical channel operations. Each decision materializes ordinary IR and then
disappears; no persistent plan object is needed.

Source LLVM commonly represents a fixed local array as `llvm.alloca`, while
the same logical temporary may already be a `memref.alloc` after an upstream
Structured transform. Requiring an eager pointer-to-memref bridge would add a
second memory-root interpretation and would be unnecessary when the complete
temporary is about to disappear. The channel proof therefore admits only the
closed LLVM allocation form whose size, scalar leaves, lifetime, launch uses,
and dense ordered offsets are all exact. Promotion deletes that entire pointer
closure and emits the same channel operations as the memref form. Any case that
would leave a pointer or require inferred alias authority stays unpromotable.

Adjacent storage-position exchange is sufficient because repeated exchanges
generate every dense permutation. Enumerating complete permutations in one
decision would make the same final layout reachable through a factorial
parameter domain and duplicate work without adding semantics. Logical indices
remain unchanged, while each endpoint's ordinary memref strides own its
address. A copy between different layouts therefore computes its two addresses
independently instead of assuming identical linearization.

Likewise, multibuffering is derived from the proved pipeline rather than a
separate knob. A value live across `d` logical iteration boundaries requires
exactly `d + 1` ring slots; the admitted copy/compute schedule has `d = 1` and
therefore double buffering. Making buffer count independently configurable
would admit redundant or insufficient states and would confuse a logical
iteration relation with physical latency.

Channel promotion requires one exact producer sequence, an exact event
bijection for every consumer, and effect-safe launch motion. The same proof
admits unicast and multicast because canonical Dataflow channel semantics
already owns one producer with one or more independently mapped consumers.
Adding a second multicast decision or synthesizing a fanout carrier would
duplicate that owner without proving anything new. The narrow ordered proof
still avoids inventing a general concurrent alias model. Channel capacity,
physical FIFO choice, banking, coalescing, service endpoints, replication
placement, and routes remain downstream Fabric and Mapping facts. Keeping
those facts out of all four decisions preserves one owner for each
software or hardware distinction.

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

LLVM pointers remain valid in the callable ABI and may enter the graph as
ordinary pointer values. When analysis proves a static logical root, SCF
optimization may instead materialize an exact memref capability plus integer
offset before the graph boundary. Graph launch itself performs no pointer-to-
memref adaptation. Encoding that adaptation as
`builtin.unrealized_conversion_cast` would turn an incomplete dialect-
conversion marker into executable, mappable semantics. Introducing a duplicate
Dataflow pointer type would likewise restate LLVM's DataLayout and provenance
authority. The two closed graph forms are therefore memref capability plus
index, or the original typed LLVM pointer plus an exact memory-service
capability.

Those two forms also require different arithmetic proofs. A root-relative
element index is meaningful only in its selected canonical index width. An
LLVM pointer GEP is already a byte-address computation whose arithmetic width
is owned by the module DataLayout for that address space. Rechecking the latter
at an unrelated canonical `index` width can reject a valid 64-bit pointer lane
merely because the candidate selected 32-bit memref indices. Parallel overlap
checking therefore compares exact pointer byte ranges at the DataLayout width,
while root-relative checking alone proves canonical element-index
representability. This keeps both representations strict without creating a
third address authority. For the same reason, a module-owned fixed `index`
width narrows only the root-relative candidate domain. Treating it as an
implicit address-form selection would erase a valid pointer-addressed
candidate and turn DataLayout into a second DSE policy owner.

The temporary root-relative marker exists because the selected projection must
survive immutable Structured publication until graph memory lowering, while an
LLVM pointer type alone denotes the opposite `PointerAddressed` form. Keeping
the choice only in invocation state would make equal Structured identities
lower differently. Persisting it in Canonical Dataflow would duplicate the
address type's closed `MemoryAddressForm`. An identity-bearing marker on the
exact source access, consumed while replacing that access, is therefore the
smallest complete bridge between the two owners.

## Why Dataflow Has Its Own Optimization Lineage

The earlier assumption that every optimization could happen in SCF was too
strong. Two canonical Dataflow programs can be functionally equivalent but
have different actor topology, synchronization structure, fanout, or
backpressure behavior. Typed Dataflow rewrites therefore form a second
immutable candidate lineage after mechanical lowering.

These rewrites may simplify actor networks, reshape synchronization, or remove
provably redundant structure. They cannot revisit loop scheduling, reduction
algebra, the Structured vectorization factor, or ownership. Functional
equivalence preserves values, streams, memory effects, externally visible
ordering, and abstract liveness; it need not preserve internal actor traces or
hardware cycle count.

The cardinality-commutation and fixed-vector-decomposition rules express
different semantic transformations. Decomposition splits one already selected
semantic vector actor, while commutation moves an actor across existing
cardinality adapters without choosing another vector factor. Neither replaces
the other. Letting source implementation membership define the catalog would
allow HOW to silently delete WHAT; treating chunking and scalarization as
unrelated kinds would duplicate one eligibility and equivalence owner. One
closed catalog with one parameterized decomposition kind preserves all three
boundaries.

The catalog is listed only in the compiler specification. Vectorization owns
the decomposition details, DSE owns enumeration and work accounting, and this
rationale owns only the reason for that split. A plugin registry or generic
rewrite-law DSL would add an authority without making any current rule more
expressive.

## Why Semantic Conformance May Retain D0 Directly

Semantic conformance asks whether the exact source workload has one
functionally equivalent program admitted by the selected Fabric. It does not
ask which equivalent topology has the best predicted quality. Once exact
capability analysis admits every D0 actor, ordinary functional Evidence is the
remaining proof and the already finalized D0 is a legal D*. Running the
recursive rewrite catalog at that point would turn a feasibility gate into an
unrequested QoR search and can spend its finite semantic budget enumerating
independent topology combinations.

The existing selection intent is sufficient to express this distinction; a
runner flag, catalog subset, or special anchor policy would create another
authority. Benefit-qualified compilation still invokes the full rewrite
generator. Semantic conformance also invokes it when D0 is not admitted,
because only a materialized rewrite can establish an alternative feasible D*.
The exact Fabric capability index owns that branch. Runtime duration, worker
count, graph size, and test identity do not participate.

Skipping optional rewrites cannot hide a functional mismatch. Every admitted
D0 still passes the same source-backed functional obligation, and a mismatch
ends the conformance attempt. Since every catalog rule is required to preserve
software semantics, trying another topology would not be a valid repair.

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

Equal bit width also does not make a pack/unpack composition an unconditional
vector identity. Packing collapses mixed lane-level poison or undef into one
scalar exceptional state, so unpacking cannot recover the original lane
mixture. Round-trip elimination therefore consumes the vector owner's exact
exceptional-state projection and requires an identity proof for the source
value domain; representation width alone is insufficient.

A narrow target does not authorize Mapping to split a wide vector implicitly.
Splitting can change floating reduction order, poison observation, memory
firing atomicity, masks, and backpressure. It is therefore an explicit
Dataflow-to-Dataflow candidate transform with its own immutable result and
functional proof. This preserves a simple Mapping contract: either every
complete actor token fits one admitted realization and route, or that Mapping
candidate is invalid.

The minimal exact transform for an elementwise actor uses complete
leading-dimension blocks. Standard `vector.shuffle` already owns block
selection and concatenation, while one `dataflow.sync` preserves the original
joint operand firing. This avoids a new lane IR, packed-integer convention, or
backend-private slicing table. Scalarization is a separate typed decision
because it changes both actor count and the required physical families; it is
not an implicit fallback when chunk admission fails.

Constraining the transform to pure elementwise actors is intentional. A vector
memory operation, reduction, or stateful actor has an atomic effect or ordering
contract that cannot be reconstructed by merely cloning its scalar operation.
Those owners require their own proved transforms rather than exceptions in the
elementwise rule.

The classification belongs in the generated OperationSchema source because an
operation carrier trait is not the software semantic identity. In particular,
a generic registered-intrinsic carrier may have no generic elementwise trait
while its selected saturating arithmetic schema is exactly pointwise. Reading
the generated schema fact admits that actor without a handwritten intrinsic
exception, and keeps HSG membership from becoming a second software-semantics
authority.

## Why Representative Kernels Are Structural Anchors

The frontend anchors were selected to force interaction among decisions:
dense vector loops, nested reductions, sparse indirection, irregular control,
DSP reuse, stencils, convolution, and multi-stage attention streaming. They
protect real source-to-Dataflow behavior without creating one fixture for
every pass combination or redefining corpus membership.
