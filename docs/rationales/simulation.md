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
byte offset. Canonical graph boundaries may expose either a memory capability
or a first-class LLVM pointer value, but both resolve through that registry.
Treating descriptor fields, call operands, globals, and graph pointer tokens as
separate pointer authorities would duplicate aliasing facts and make two
equivalent access paths simulate differently.

Initial memory bytes alone cannot preserve that relation: identical pointer
bit patterns in different native processes do not identify the same simulated
object, and canonical object bases need not equal host addresses. Runtime input
therefore stores one typed provenance overlay for defined pointer payloads.
The bytes still own the pointer representation; the overlay owns only the
object-relative target needed to reconstruct it. This avoids both an alias
graph and a second copy of memory content.

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

## Why Engines And Environments Compose

DFG, CGRA, and RTL are three fidelities of one SpatialCore execution boundary.
Spatial-only and gem5-backed System execution are two environments around that
boundary. Treating their Cartesian product as six unrelated simulators would
duplicate actor semantics, terminal observations, Bridge behavior, and error
classification. Treating the environment as a field inside
`SimulationExecution` would duplicate the exact Request and workload root.

The implementation therefore reuses one Spatial engine behind standalone and
Bridge adapters. The Evaluation model descriptor owns the selected engine;
the exact case signature and workload root own the environment. This permits
System + DFG to bring up target binaries, dispatch, NoC, caches, and external
memory before detailed Spatial resources exist, without pretending that DFG
timing is CGRA or RTL timing.

Spatial activity residency uses an exact mapped reference-cycle basis, while
System progress uses gem5 ticks. Those units are not interchangeable. The
first System execution form therefore keeps its activity-summary array empty;
a later System activity payload must own an explicit tick or hardware-clock
time basis instead of reinterpreting the Spatial wire.

Mapped RTL needs more exact input than mapped CGRA simulation. The RTL bytes,
configuration ABI, and external bindings belong to HardwareImplementation,
while the selected occurrence, SpatialMapping context, and complete
configuration image belong to Deployment. A dedicated mapped-RTL case binds
those two owners and a Spatial workload. Reusing the CGRA case would hide the
implementation in a model input; treating the Deployment as a System workload
would execute a different environment. The dedicated case therefore adds no
engine field or execution subtype: Deployment supplies only the exact launch
closure, and the Spatial workload still owns the execution boundary.

The gem5-backed RTL descriptor reuses the standalone descriptor's typed HDL
simulator binding and mapped-RTL closure, but not its Request or Evidence. A
generic multi-tool bundle was rejected because it would add a new tool-cohort
authority to every external provider. A precompiled RTL-engine Artifact was
also rejected because it would duplicate the exact HardwareImplementation,
Deployment, and simulator-build closure. Instead, the frozen HDL compiler
creates the invocation-local engine, while the independently owned gem5 binary
is an external file whose content is checked against the
Gem5SimulationBinding-owned fingerprint before that engine launches it. The
fingerprint belongs to persistent build identity because strict import has no
machine-local configuration and cannot otherwise reconstruct the expected
external input. The readiness file proves local availability only; letting it
author the fingerprint would make attempt state a competing semantic owner.
This is the smallest composition that preserves both build identities and one
System time authority.

The HostCore can be understood as the cluster's additional stored-program
engine, but not as an AccCore: it has no SpatialCore and no `dataflow.thread`
identity. Requiring one compatible RISC-V ISA/ABI cohort for the HostCore and
AccCore InstructionCores keeps the first system executable closure small and
matches gem5's processor composition model. Their microarchitectures and
runtime services remain independently Fabric-owned, so this does not flatten
the distinct physical cores or their persistent references.

This cohort choice does not reinterpret ARM-specific corpus profiles. MVE,
DSP, and NEON source paths remain exact target-profile identities and are
typed incompatible with the RISC-V execution cohort before a workload provider
is built. Counting them as scalar execution would create a second program
authority; treating their incompatibility as an unknown provider would hide a
known architectural reason. Keeping `pass`, profile-incompatible
`Unsupported`, and failure disjoint preserves both facts.

## Why System Overhead Has A Paired Budget

A fixed generous System timeout hides integration overhead on small kernels,
while comparing raw gem5 ticks with DFG or CGRA cycles compares different time
domains. The conformance gate therefore pairs equal workloads and engine
fidelities and measures warmed active wall time. Setup that can be shared,
such as compilation, gem5 construction, and RTL elaboration, is reported but
does not consume the execution ratio.

The initial factor of three is a performance policy rather than an Artifact
field or model parameter. It is strict enough to expose repeated startup,
serialization, polling, or event-loop handoff, while leaving room for actual
HostCore, NoC, cache, and memory work. Tiny measurements use a floor and
repetition so timer noise cannot dominate. A tenfold slowdown is rejected
regardless of the current policy factor because it indicates a different
execution strategy or an avoidable integration bottleneck rather than the
intended weakly coupled composition.

DFG and CGRA require separate Spatial-only absolute budgets because the latter
models finite routes, resources, arbitration, and contention. Guessing the
CGRA budget permanently or allowing each workload to choose one would hide
either simulator defects or difficult mappings. A short bootstrap ceiling is
therefore replaced, after representative measurements, by one tracked
suite-wide value. System execution derives its paired ceiling from that value
instead of creating another timeout authority.

## Why DFG And CGRA Share Functional Semantics

An actor's firing and state transition must mean the same thing at every
fidelity. The shared semantic kernel defines consumption, production,
linearization, commit, and state update. Execution policies decide when an
enabled transition can run and what resource delays it.

Separate per-simulator semantics would let a graph pass DFG-sim and fail CGRA-
sim for reasons unrelated to hardware. Shared memory consistency is especially
important: local CGRA and external gem5 providers add timing and contention but
cannot redefine reads-from, atomicity, or visibility.

The same rule applies to vector structure and exceptional lanes. A shuffle can
produce Defined and Poison blocks in one token, so collapsing the whole token
to one state would either lose defined siblings or invent values for poison.
The shared semantic kernel therefore represents lane state exactly. A packed
fully-defined token or an execution-local arena handle is an implementation
choice, not another value model. CGRA execution adds only the selected physical
resource and timing behavior around that same functional result.

A mask lane is different from an ordinary value lane because it chooses
whether an operand is consumed or a memory effect occurs. Poison or undef
therefore denotes a non-singleton firing relation rather than an arbitrary
zero-or-one bit. A single-path execution engine cannot select one branch
without inventing semantics, and treating the actor as blocked would falsely
report a wait-set problem. Atomic typed `Unsupported` preserves the exact
boundary: no input is consumed and no effect occurs until an execution model
with an explicit exceptional-control relation is available.

CGRA-sim deliberately consumes the Mapping-owned semantic configuration
projection rather than `ConfigurationABI` or a configuration image. Two ABIs
may place or encode the same Fabric semantic value differently while describing
the same selected Dataflow, Fabric, and SpatialMapping behavior. Making either
ABI a CGRA input would split one architectural execution into encoding-specific
case identities. Conversely, interpreting only the Dataflow actor without
replaying the selected Fabric projector could simulate behavior that the mapped
physical slot cannot configure. The transient Mapping projection supplies the
needed proof without importing physical programming into the CGRA fidelity.
Retaining another slot/value vector in the simulator would add no behavior:
the exact actor semantic kernel already owns the functional result and the
selected Fabric contracts own resource and timing behavior. The projection is
therefore checked during cold admission and remains owned by the retained
Mapping view rather than being copied into the hot execution plan.

CGRA cycle metrics use the exact SpatialMapping root as the case-level cycle
anchor because the reference domain is a relation of Mapping, Fabric, Dataflow,
and launch facts rather than a new persistent clock choice. Copying one clock
reference into Request or Mapping would create a competing authority, while a
DFG-style abstract tick would discard the hardware relation. The root anchor
keeps the projection reproducible and permits the case-signature owner to
reject a non-unique or non-integral result without rounding.

## Why Gem5 Owns System Time

An in-process bridge integrates the reusable SpatialCore simulator as an
out-of-tree gem5 component. The gem5 event queue is the sole system time
authority. The SpatialCore session advances to the next observable boundary,
such as a memory request, interrupt, result, completion, or wakeup time, then
returns control.

Keeping the pinned gem5 source unmodified makes the upstream commit the exact
simulator implementation identity and keeps Loom's protocol correspondence in
one reviewable owner. A local patch stack would mix bridge semantics with an
external dependency and make upgrades or replay depend on hidden source state.
The supported out-of-tree extension boundary provides the required integration
without creating that second authority.

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
in `loom.simulation_workload 1.0`; unresolved DynamicWork correspondence fails
with typed `Unsupported` rather than using temporary IDs.

Value, stream, and memory inputs follow the graph ABI instead of carrying a
second port-kind union. Aliasing uses shared canonical memory-object ordinals,
not a separate alias graph. Defined, poison, and undef values use one canonical
semantic-value algebra shared with compiler semantics.

Source-backed validation captures stream traffic at the unique typed logical
channel endpoint that produced the corresponding graph port. This retains the
source execution as the independent oracle while avoiding a second channel
identity scheme or a positional match against native runtime queues. The
capture is activation-local and ephemeral; its ordered values are encoded
directly through the existing runtime-stream and stream-observation planes.

## Why Execution Has One Root And Closed Terminals

The exact EvaluationRequest already selects spatial or system model semantics.
Adding a second execution-kind tag would allow disagreement. One execution
root references that Request and ends as Retired, Halted, or StoppedByLimit.

Deadlock witnesses belong only to Halted and resolve through typed model output
slots. Unsupported capability or provider failure does not publish a
placeholder execution. Output cardinality is validated against the Request so
an adapter cannot omit or add observations.

The Halted wire length-frames bytes produced by the exact Finding owner. This
lets one closed terminal algebra carry different typed witness families without
making Simulation Artifacts a union of every future diagnostic. Strict
decode/validate/re-encode equality preserves canonical identity while the
Evidence occurrence remains only an output-relative reference.

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

Dynamic occurrences are execution-local ordinals, not EntityIds.
Semantic token publication records exact values when that capture level is
requested. Physical request, grant, and retirement are the irreducible facts;
stall is derived from their interval, not stored as another event.

A transfer target uses one canonical set of selected Fabric use patterns.
Keeping only one optional pattern cannot represent a switch broadcast whose
atomic activation joins one pattern per traversal. Emitting one physical
action per pattern would instead erase that atomicity. The empty, singleton,
and multi-pattern set is the smallest uniform representation of direct,
ordinary resource-bearing, and broadcast transfers, and it remains a checked
projection of Fabric and Mapping rather than another grouping decision.

Trace capture levels are inclusive so Semantic contains Firing and
Microarchitecture contains both. DFG-sim supports only the levels it can
truthfully produce; requesting physical microarchitecture from DFG-sim is
Unsupported.

## Why Raw Trace Persistence Is Deferred

Inlining diagnostic traces into SimulationExecution would make optional debug
history part of semantic identity and would force large retained data through
every importer. Opaque simulator-private files cannot provide shared
validation or become evidence merely because a viewer can read them.

The current boundary is therefore exact: `SpatialDiagnosticTrace` is a shared
typed invocation-local value retained only by an attempt context or scratch
storage, and `loom.simulation_execution 1.0` has no trace field. A persistent
design, if ever required, must be approved with its owner, identity, lineage,
framing, validation, and loading semantics at that time; the current contract
does not pre-encode that future schema.

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
