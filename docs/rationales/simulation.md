# Simulation Rationale

Normative contracts are owned by
[Simulation Artifacts](../spec-simulation-artifacts.md),
[DFG-sim](../spec-sim-dfg.md),
[CGRA-sim](../spec-sim-cgra.md), and
[Simulation Comparison](../spec-sim-comparison.md).

## Why Loom Has Three Simulation Boundaries

DFG-sim executes one canonical SpatialCore graph without Fabric limits. It is
hardware-unconstrained but still models actor latency, initiation interval,
ordered tokens, state, memory semantics, and backpressure. It does not execute
the InstructionCore or whole program.

CGRA-sim adds exact Fabric and SpatialMapping. It models PE/FU, memory,
switches, routes, buffers, tags, temporal contexts, resource contention, and
physical deadlock at cycle fidelity.

System simulation combines an exact Deployment and gem5 binding. Gem5 owns
HostCore and InstructionCore execution, caches, coherent memory, system NoC,
OS/runtime behavior, and whole-system time. Loom's SpatialCore model is a
device-level participant. Building another CPU or SoC simulator inside Loom
would duplicate mature infrastructure and distract from the accelerator
research focus.

This separation also keeps source-backed host validation from becoming an
accidental system simulator. DFG-sim may temporarily retarget an exact selected
region after proving effective layout compatibility, but sys-sim executes the
Deployment-selected target binary on the bound gem5 ISA model. RISC-V support
therefore comes from the exact Fabric architectural contract, compiler target
binding, and gem5 processor correspondence rather than from host JIT behavior.

Source-backed pointer discovery remains ephemeral for the same reason. One
runtime object registry resolves every concrete pointer to an allocation and
byte offset, while canonical graph boundaries expose memory capabilities rather
than raw host pointers. Treating descriptor fields, call operands, and globals
as separate pointer authorities would duplicate aliasing facts and make two
equivalent access paths simulate differently.

The graph launch, rather than a graph-body cast, establishes the typed view of
that object. DFG-sim can therefore seed one byte object and interpret each
formal through the exact Dataflow-owned memref relation. Giving a conversion
marker its own simulator behavior would create semantics that neither the
source program nor Fabric owns.

Source, selected Structured execution, and DFG replay are intentionally three
separate observations. Source-versus-selected comparison validates the
transformation; selected-versus-DFG comparison validates lowering and actor
semantics. Comparing only source and DFG would not identify which boundary was
wrong, while comparing only selected and DFG could bless a miscompiled
candidate. All executions start from the same immutable runtime input so the
comparison cannot be contaminated by earlier mutable state.

SST remains a possible future adapter for large-scale exploration, not a first-
version second system authority.

## Why DFG And CGRA Share Functional Semantics

An actor's firing and state transition must mean the same thing at every
fidelity. The shared semantic kernel defines consumption, production,
linearization, commit, and state update. Execution policies decide when an
enabled transition can run and what resource delays it.

Separate per-simulator semantics would let a graph pass DFG-sim and fail CGRA-
sim for reasons unrelated to hardware. Shared memory consistency is especially
important: local CGRA and external gem5 providers add timing and contention but
cannot redefine reads-from, atomicity, or visibility.

## Why Gem5 Owns System Time

An in-process bridge integrates the reusable SpatialCore simulator as an
out-of-tree gem5 component. The gem5 event queue is the sole system time
authority. The SpatialCore session advances to the next observable boundary,
such as a memory request, interrupt, result, completion, or wakeup time, then
returns control.

Running an independent Loom system event loop would require clock
synchronization, duplicate ordering, and conflict resolution. The bridge is a
typed adapter, not a second scheduler. Fabric remains simulator-neutral; a
Gem5SimulationBinding proves correspondence between exact hardware facts and
the selected gem5 models.

## Why Workload, Runtime Input, And Execution Are Separate

A workload identifies the exact rooted graph launch and fixed problem facts.
Runtime input supplies values, ordered stream sequences and close state, and
byte-addressed memory objects. Execution records terminal and requested
observations. Keeping these objects separate permits replay with new inputs and
prevents expected outputs from contaminating the workload identity.

The rooted launch already owns graph, thread context, ABI, and static launch
identity, so the workload does not repeat them. Dense coordinates are admitted
in schema 1.0; unresolved DynamicWork correspondence fails typed Unsupported
rather than using temporary IDs.

Value, stream, and memory inputs follow the graph ABI instead of carrying a
second port-kind union. Aliasing uses shared canonical memory-object ordinals,
not a separate alias graph. Defined, poison, and undef values use one canonical
semantic-value algebra shared with compiler semantics.

## Why Execution Has One Root And Closed Terminals

The exact EvaluationRequest already selects spatial or system model semantics.
Adding a second execution-kind tag would allow disagreement. One execution
root references that Request and ends as Retired, Halted, or StoppedByLimit.

Deadlock witnesses belong only to Halted and resolve through typed model output
slots. Unsupported capability or provider failure does not publish a
placeholder execution. Output cardinality is validated against the Request so
an adapter cannot omit or add observations.

## Why Observations Are Typed And Positional

The workload contract already owns requested value, stream, and memory targets
and their canonical order. Execution stores observations positionally rather
than repeating keys. Values publish atomically, streams carry an ordered prefix
and close state, and memories report exact visible state; a generic
Complete/Partial/Unavailable wrapper would create nonsensical combinations.

Progress stores only the minimal launch, retirement, and terminal coordinates.
Elapsed cycles are Evaluation metrics, and a full event list is trace data.
Activity uses one envelope with distinct actor, Fabric-resource, and
implementation-signal payloads because their keys and validation differ.

## Why Trace Uses A Small Typed Algebra

Generic `kind + property map` events would let every simulator invent payload
semantics. A per-operation hierarchy would duplicate the same commit,
publication, memory, and physical lifecycle across many types. The shared trace
uses a small closed set of actor, token, memory, and physical events with exact
Dataflow, Fabric, and Mapping references.

Dynamic occurrences are execution-local ordinals, not persistent EntityIds.
Semantic token publication records exact values when that capture level is
requested. Physical request, grant, and retirement are the irreducible facts;
stall is derived from their interval, not stored as another event.

Trace capture levels are inclusive so Semantic contains Firing and
Microarchitecture contains both. DFG-sim supports only the levels it can
truthfully produce; requesting physical microarchitecture from DFG-sim is
Unsupported.

## Why Raw Trace Persistence Is Deferred

Inlining large traces into SimulationExecution would prevent streaming,
deduplication, and selective loading. Opaque simulator-private chunks would
prevent shared validation and visualization. A future raw detailed-bundle
owner may provide typed manifest and chunk inventory, but no such field exists
in schema 1.0.

Until that owner closes, traces are diagnostic attempt or scratch material.
Visualization may consume the shared typed in-memory wire but cannot turn a
private file into semantic evidence.

## Why Simulation Comparison Is Gated

DFG, CGRA, system, and RTL results are comparable only for observations whose
subjects, workload, external services, reference cycles, and implementation
correlations align. Functional equality may be meaningful when cycle equality
is not. Comparison therefore validates exact coupling before deriving metrics
or findings instead of coercing unlike executions into one score.

## Why Source-Backed Validation Uses Ownership Lineage

Pre-Mapping DSE needs one concrete workload before any Dataflow graph exists.
Reusing only the Spatial workload would force a candidate graph to become the
input identity, so two Structured candidates could no longer be compared on
the same source execution. A test-only workload record would create the same
split between production compilation and corpus validation.

The Simulation families therefore append one Structured Program root. Its
entry reference already identifies the exact source program, while its
argument plan and observable contract describe only workload choices. LLVM ABI
facts remain in the Structured Program; concrete runtime values and backing
objects remain in RuntimeInput. Candidate-specific Spatial inputs are derived
for graph replay, not promoted into a second whole-program workload authority.

Direct pointer arguments are represented as bindings into the existing
byte-addressed runtime-object algebra. This preserves finite extent and
aliasing without serializing native addresses. Mutable global input was not
added: the exact program initializer remains authoritative, and a workload
that needs different setup can express it in the entry program. This keeps the
first source root complete for ordinary linked kernels without introducing a
general process image or host-environment model.

An operation-owned Spatial graph is only one region of a complete stored
program. Executing the graph alone cannot recover the values and aliased memory
state produced by residual host code, while moving that residual code into the
graph would falsify ownership. The independent oracle therefore replays the
already selected Structured ownership decision and observes its exact boundary
inside the complete program execution.

For an operation-owned region, matching the target graph back to a separate
native module by debug location, symbol spelling, or operation position would
create a second and unstable identity system. Invocation-local DSE lineage
already names the exact parent, scope, and typed decision, so it is sufficient
and introduces no persistent schema. Requiring effective execution-layout
equivalence before host JIT execution prevents the oracle from becoming a
silent cross-target emulator.

A selected inner region may repeatedly receive different views into one
caller-owned object, such as one row pointer per outer-loop iteration. The
backing object and alias class are static facts, while the view offset is a
dynamic invocation fact. Keeping the former in the capture plan and deriving
the latter at the observed boundary avoids both a shadow memory identity and
the incorrect assumption that every graph activation sees one fixed slice.

A direct leaf call alone is insufficient when its arguments are forwarded
through intermediate callables or when the same leaf call site is reached from
multiple outer call sites. Source-backed validation therefore derives an
ephemeral, finite root-to-leaf direct-call path. The leaf remains the only
value sampling boundary; outer path edges only prove finite object lineage and
gate the selected dynamic context. This preserves one program identity model
while preventing unrelated invocations from being conflated and allowing a
workload to leave one statically reachable branch untaken.
Requiring every static path to execute would confuse control-flow reachability
with one concrete workload; requiring the aggregate to execute keeps an
unobserved candidate from passing.

Target-triple spelling alone is not execution compatibility. Two triples may
name different targets while the exact selected region uses only types,
address spaces, and layouts whose effective projections are identical. A
name-equality gate rejects such regions without adding safety. Conversely,
similar target names do not prove compatible layouts. The oracle therefore
proves compatibility from the selected region's actual layout roots and used
types, then retargets only an ephemeral execution clone. This keeps the
original target artifacts authoritative while allowing an independent host
functional oracle where no target-specific behavior is involved.
