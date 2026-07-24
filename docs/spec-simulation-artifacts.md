# Simulation Artifacts

This document is the sole persistent-schema owner for workload-running
simulation inputs and typed execution observations. DFG-sim, CGRA-sim, sys-sim,
and mapped RTL simulation share these Artifact families while retaining their
own execution semantics and time authorities.

## Artifact Families

The fixed schema descriptors are:

```text
loom.simulation_workload      1.0
loom.simulation_runtime_input 1.0
loom.simulation_execution     1.0
```

Each family has one typed C++ model and one canonical serializer/parser.
Schema versions use `X.Y`: `X` denotes an incompatible change and `Y` denotes
a compatible extension. Simulator-specific request, result, trace, activity,
or report Artifact families are forbidden.

## SimulationWorkload

`SimulationWorkload` has one closed root union:

```text
SimulationWorkload =
    Spatial(SpatialSimulationWorkload)
  | System(SystemSimulationWorkload)
```

The root discriminants are zero and one in declaration order. The spatial
root is:

```text
SpatialSimulationWorkload {
  launch_ref: RootedGraphLaunchRef
  dense_coordinates: array<uint64>
  value_input_plan:
    total table<graph value-input ordinal, SpatialValueInputSource>
  observable_contract: SpatialObservableContract
}

SpatialValueInputSource =
    Fixed(CanonicalValueSequence)
  | Runtime
```

`launch_ref` is the one Dataflow-owned structural reference that recovers the
exact Canonical Dataflow Artifact, root thread launch, static graph launch,
called graph, graph ABI, memory roots, and channel context. The workload must
not copy a `canonical_dataflow_ref`, `GraphRef`, graph symbol, or logical
invocation record.

The dense coordinate count must equal the root thread domain rank. Every
coordinate must be inside any statically known bound. A dynamic-work point is
not admitted by schema 1.0; its persistent identity depends on the separately
owned DynamicWork correspondence contract.

The value-input table is exactly total over graph value-input ordinals. A
`Fixed` entry contains exactly one semantic token and makes that launch
parameter part of workload identity. A `Runtime` entry delegates only that
token to the exact `SimulationRuntimeInput`. Stream inputs are always concrete
runtime sequences, and imported memory roots are always runtime bindings.
Consequently there is no generic four-way `InputSource` union that repeats the
value, stream, and memory planes already owned by the graph ABI.

The spatial observable contract is:

```text
SpatialObservableContract {
  value_results: canonical set<graph value-result ordinal>
  stream_outputs: canonical set<graph stream-output ordinal>
  memories:
    canonical table<SpatialMemoryObservableTarget, MemoryObservationForm>
}

SpatialMemoryObservableTarget =
    LogicalMemory(LogicalMemoryRootOrViewRef)
  | Exposure(graph memory-result ordinal)

MemoryObservationForm =
    FullState
  | DiffFromRuntimeInput
```

Value and stream ordinals are owner-relative to `launch_ref`; their complete
meanings are the corresponding `GraphLaunchBoundaryTransferRef` and
`ChannelProducerRef` forms. An exposure ordinal has the complete meaning of
the corresponding `MemoryExposureRef`. Types and shapes are recovered from
the exact Dataflow owner and are never copied into the contract. Each
collection is sorted by the canonical typed target key and contains no
duplicates.

Graph completion is mandatory for every workload and derives from
`GraphLaunchBoundaryTransferRef::Done(launch_ref)`. It is not an optional
observable selector. Intermediate completion and retirement activity belongs
to execution trace or activity records rather than this contract.

The system root has the structural shape:

```text
SystemSimulationWorkload {
  program_entry_ref: Deployment-owned typed program-entry reference
  external_interface_refs:
    canonical set<Deployment-owned typed external-interface reference>
  observable_contract: SystemObservableContract
}
```

The program-entry reference already carries the exact Deployment identity, so
there is no second `deployment_ref`. The Deployment family must own both
target catalogs, ordinals, and validation. Until the executable-closure and
Deployment frontiers close those references, a schema-1.0 producer must reject
the `System` root rather than serialize a string, symbol, generic path,
provisional ordinal, or private substitute. Closing those referenced catalogs
does not permit this schema to copy Deployment entry or interface fields.

Workloads own fixed problem shape, fixed launch values, and requested
observables. They do not own concrete runtime values, stream contents, memory
images, simulator policy, expected results, trace capture, or execution
limits.

## Canonical Semantic Values

One value representation is shared by fixed workload inputs, runtime value
inputs, and runtime stream tokens:

```text
CanonicalValueSequence {
  token_count: uint64
  lanes: array<SemanticLane>
}

SemanticLane =
    Defined(fixed-width semantic bits)
  | Poison
  | Undef
```

The target type is recovered from the workload and is not serialized again.
For a scalar, each token has one lane. For a fixed-ranked vector, each token
has the product of its dimensions in canonical row-major lane order. The lane
array length must therefore equal `token_count * lanes_per_token`.

A defined integer, index, or floating lane stores the exact fixed-width
software-semantic bit representation. Variant tags and bits use canonical
big-endian wire order. This is serialization framing, not target memory
endianness, Fabric port layout, vector packing, or a hardware representation.
Host integer and floating types are never semantic authorities. Unknown lane
states, wrong widths, extra lanes, and unresolved target widths are invalid.

## SimulationRuntimeInput

`SimulationRuntimeInput` uses the same root discriminant as its exact workload.
The spatial root is:

```text
SpatialSimulationRuntimeInput {
  workload_ref: exact Spatial SimulationWorkload reference
  runtime_values:
    total table<runtime graph value-input ordinal, CanonicalValueSequence>
  runtime_streams:
    total table<graph stream-input ordinal, CanonicalStreamSequence>
  memory_objects: canonical array<RuntimeMemoryObject>
  memory_root_bindings:
    total table<LogicalMemoryRootRef, RuntimeMemoryRootBinding>
}

CanonicalStreamSequence {
  values: CanonicalValueSequence
  termination: ClosedAfterLast | OpenAfterLast
}

RuntimeMemoryObject {
  byte_count: uint64
  initial_bytes: array<SemanticMemoryByte>
}

SemanticMemoryByte =
    Defined(uint8)
  | Poison
  | Undef

RuntimeMemoryRootBinding {
  object_ordinal: uint64
  byte_offset: uint64
}
```

`runtime_values` is exactly total over value-input ordinals whose workload
source is `Runtime`; every sequence has exactly one token. `runtime_streams`
is exactly total over graph stream-input ordinals. `ClosedAfterLast` publishes
the stream close after its final token. `OpenAfterLast` means that no close is
observed within the sequence's owning observation horizon. For runtime input,
that horizon is the complete supplied input: no later token or close exists,
and future timed arrivals require an independently justified typed
environment schedule rather than hidden simulator input. For execution
output, the horizon ends at the execution terminal, so the same form records
an open produced prefix without asserting a counterfactual future.

Every imported logical-memory root reachable from the selected launch has
exactly one root binding, and no unrelated root may appear. A runtime memory
object is neutral byte-addressed software storage. Its initial-byte count must
equal `byte_count`; the exact Canonical Dataflow type, DataLayout, and
root/view relations alone interpret typed accesses. This avoids choosing one
aliased memref role as a privileged storage type or importing physical memory
layout into the software input.

Objects have no author-selected persistent IDs. Before serialization, roots
that share one runtime object form an equivalence class. The canonical key for
that object is the sorted non-empty list of `(LogicalMemoryRootRef,
byte_offset)` bindings. Objects are sorted by this key and receive their
zero-based `object_ordinal` from that order. Binding two or more roots to one
ordinal expresses aliasing; no separate alias graph is serialized. Missing,
empty, duplicate, out-of-range, or unreferenced objects are invalid. Overlap
between two root ranges bound to the same object is legal aliasing and must
not be rejected merely because their ranges intersect.

A fresh graph allocation is not an imported runtime object. Its execution
identity derives from its static `LogicalMemoryRootRef` and graph-invocation
occurrence, and its storage is created under the owning software semantics.
It therefore does not receive a workload-local object ID or root binding.

The system runtime root must bind concrete inputs through the exact
Deployment-owned program-entry and external-interface references in its
workload. Its exact table variants remain gated by the same executable-closure
and Deployment frontier. A producer must not substitute argv arrays, device
names, string-key maps, or simulator-private event records for those typed
interfaces.

Runtime input does not contain model timing, trace-capture policy, execution
limits, Mapping repairs, physical addresses used only by a simulator, or
presentation options.

## Canonical Workload And Input Wire

Every displayed union assigns zero-based unsigned 32-bit discriminants in
declaration order. Unsigned counts and ordinals use 64-bit big-endian values.
Exact Artifact identities use their fixed 32-byte representation. Arrays and
tables use a 64-bit element count followed by elements in canonical order.
Nested Dataflow and Deployment references use their owning canonical
encodings; this schema does not renumber them.

Records encode fields in declaration order without names, optional property
maps, padding, or native C++ layout. Tables are sorted by their typed semantic
keys. Parsers reject duplicate or unsorted keys, unknown variants, missing or
extra fields, noncanonical bytes, unresolved references, and any cardinality
or owner mismatch. JSON, MLIR text, CLI syntax, and visualization files are
derived projections and cannot be accepted as identity bytes.

## Spatial Vecadd Example

For a rooted graph launch whose value-input ordinal zero is `N`, whose imported
logical-memory roots are `A`, `B`, and `C`, and whose computation is
`C[i] = A[i] + B[i]`, the workload contains:

```text
launch_ref = exact rooted vecadd graph launch
dense_coordinates = the selected logical thread point
value_input_plan = {
  0 -> Fixed(one defined index token with value 1024)
}
observable_contract = {
  value_results = {}
  stream_outputs = {}
  memories = {
    LogicalMemory(Root(C)) -> DiffFromRuntimeInput
  }
}
```

The runtime input contains no runtime value or stream entries. It provides
three 4096-byte objects for `A`, `B`, and `C`, binds each root to its object at
byte offset zero, and initializes the bytes using the software DataLayout for
1024 `f32` elements. The serialized object ordinals derive from the sorted
root-binding keys rather than the source names `A`, `B`, and `C`. Binding `A`
and `B` to one object ordinal would instead state that those two imported
roles alias; no other record changes.

## SimulationExecution Root, Coupling, And Terminal

Every model that actually runs a software workload produces one
`SimulationExecution` with one untagged root:

```text
SimulationExecution {
  request_ref: exact EvaluationRequest reference
  terminal: ExecutionTerminal
  functional_observations
  progress_observations
  activity_summaries
  trace_manifest?
}
```

The root does not carry a `spatial` or `system` discriminator. The exact
Request already binds the model descriptor and workload, and the workload
root owns that distinction. Repeating it in the execution would create a
second case-kind authority.

The execution also has no direct observable-contract reference. It recovers
the applicable contract through:

```text
request_ref
  -> EvaluationRequest.workload_ref
  -> SimulationWorkload observable contract
```

The root field order and the terminal record are closed below. The exact
functional-observation, progress-observation, activity-summary, and
trace-manifest records remain the next persistent-wire frontier. Until those
records close, a producer must not serialize these conceptual collection
names as generic maps or claim that the complete
`loom.simulation_execution 1.0` wire is implemented.

The closed terminal algebra is:

```text
ExecutionTerminal =
    Retired
  | Halted {
      finding_kind: FindingKind
      witness: registry-defined terminal-witness payload
    }
  | StoppedByLimit
```

`Retired` means the workload's completion frontier, observable obligations,
and invocation-local quiescence contract have completed. `Halted` records a
sound execution-level witness for deadlock, trap, combinational
nonconvergence, or proven nontermination. `StoppedByLimit` records that an
execution limit ended observation without such a proof.

The `FindingKind` registry is the sole owner of the witness payload schema.
The `SimulationExecution` terminal is the sole owner of the concrete witness
instance. There is no separate `terminal_observations` collection and
Evaluation Evidence must not copy the witness.

Evaluation maps these forms exactly:

```text
Retired        -> Completed with every mandatory terminal finding Absent
Halted         -> Completed with the corresponding mandatory terminal finding
                  Present and every other mandatory terminal finding Absent
StoppedByLimit -> CancelledOrTimeout
```

Every workload-running simulator descriptor declares one typed
`SimulationExecution` output slot and the complete mandatory terminal
`FindingQuery` set. A legal EvaluationRequest must request all of those
findings. The descriptor's closed outcome-cardinality contract requires
exactly one execution output for `Retired` and `Halted`. For
`StoppedByLimit`, it alone declares whether `CancelledOrTimeout` may retain
zero or one stopped execution. `Unsupported` and `ExecutionFailed` bind no
`SimulationExecution`.

For the finding corresponding to a `Halted` terminal,
`EvaluationEvidence.Present` carries one occurrence of:

```text
TerminalWitnessRef {
  execution_output_slot_ref: ModelOutputSlotRef
  execution_output_ordinal: uint64
}
```

The reference resolves through the containing Evidence's `output_bindings`.
Validation requires that the slot schema is `SimulationExecution`, the
ordinal exists, the referenced execution names the same exact Request, and
its terminal is `Halted` with the same `FindingKind`. The singleton terminal
witness needs no additional witness ordinal. Every other mandatory terminal
finding is `Absent`.

Capability rejection before execution is `Unsupported`; simulator, tool, or
adapter failure is `ExecutionFailed`. `SimulationExecution` never owns
normalized metrics, normalized findings, DSE decisions, or human diagnostics.
Those belong only to `EvaluationEvidence` and its registries.

Capability whose absence is discoverable only from runtime values, such as an
unordered conflicting plain-memory access or a required external consistency
behavior not implemented by the exact model, also ends with Evaluation outcome
`Unsupported`. It is not `Halted` and must not fabricate a
`SimulationExecution` result. A genuine dynamic closed wait-set is `Halted`.
A simulator, Bridge, or provider invariant violation is `ExecutionFailed`.
Execution limits and external cancellation remain `StoppedByLimit` or
`CancelledOrTimeout` according to the owning attempt and Evaluation contract.
Static invalid IR, Fabric, or Mapping is rejected before execution and never
becomes dynamic deadlock.

## Spatial Functional Observations

The spatial workload's observable contract is the sole owner of selected
targets, target order, types, and memory observation forms. A spatial
execution therefore stores three positional arrays:

```text
SpatialFunctionalObservations {
  value_results: array<ValueResultObservation>
  stream_outputs: array<CanonicalStreamSequence>
  memories: array<MemoryObservationPayload>
}

ValueResultObservation =
    Published(CanonicalValueSequence)
  | NotPublished
```

The arrays align exactly with the canonical `value_results`,
`stream_outputs`, and `memories` collections recovered through the execution's
Request and workload. They do not repeat graph ordinals, memory targets,
types, or `MemoryObservationForm` tags. A published graph value contains
exactly one token. A stream may contain any token count, including an empty
open prefix.

The workload-selected memory form determines the corresponding payload
without a second persistent discriminator:

```text
FullState payload {
  byte_count: uint64
  bytes: array<SemanticMemoryByte>
}

DiffFromRuntimeInput payload {
  byte_count: uint64
  runs: array<MemoryDiffRun>
}

MemoryDiffRun {
  byte_offset: uint64
  changed_bytes: nonempty array<SemanticMemoryByte>
}
```

A full-state byte array has exactly `byte_count` elements. Diff runs are
sorted by offset, in range, nonoverlapping, nonadjacent, and maximal. Every
encoded byte differs semantically from the exact runtime baseline; adjacent
changed bytes belong to one run. An empty run array uniquely means no change.
Equality distinguishes `Defined`, `Poison`, and `Undef` states as well as
defined byte values.

`DiffFromRuntimeInput` is legal only when the exact Dataflow target and
RuntimeInput mechanically establish one unique baseline range before
simulation. Request verification rejects a target without that correspondence
or with a mismatched extent. A fresh allocation, dynamically unresolved
exposure, MMIO effect, or other target without a runtime byte baseline must
use `FullState` when the exact model supports snapshot semantics.

Each memory entry is the requested target-relative projection of one execution
state. No execution-local memory-object ID or alias graph is added. If selected
targets overlap through relations mechanically recoverable from Dataflow and
RuntimeInput, finalization verifies that their overlapping semantic bytes
agree. This check does not make either projection the storage authority.

All three arrays are exactly total over the selected targets for every
retained execution:

* `Retired` requires every value to be `Published`, every selected stream to
  be `ClosedAfterLast`, and every memory payload to describe final visible
  logical state.
* `Halted` permits a value to be `NotPublished` and a stream to contain an
  open prefix. Memory payloads describe exact logical state visible at the
  halt coordinate.
* a retained `StoppedByLimit` execution uses the same partial-observation
  forms at its stop coordinate.

There is no generic `Complete | Partial | Unavailable` wrapper. Value
publication, stream closure, and memory visibility already provide the three
planes' exact completion facts. The Request verifier rejects a model whose
descriptor statically cannot observe a selected target. Runtime-dependent
absence of required capability produces `Unsupported`; provider failure
produces `ExecutionFailed`. Neither outcome creates an execution containing
an unavailable placeholder.

The System workload remains fail-closed until Deployment owns its exact
program-entry and external-interface catalogs. Its future functional
observations must be selected and ordered by that workload contract and must
not add a second execution-root discriminator or provisional string-key map.

## Trace And Activity

An optional typed trace manifest owns the ordered list of content-addressed
trace-chunk references, time coverage, capture level, and completeness. The
referenced opaque chunk payloads and their inventory belong to an immutable
raw detailed bundle, not to `SimulationExecution` and not to a separate trace
Artifact family. They may be compressed or indexed without changing manifest
order or semantic observations.

When the capture request includes memory-consistency detail, opaque chunks may
record issue, linearization, reads-from, visibility or synchronization, and
retirement observations. These observations are a projection of the exact
execution; they are not a `ConsistencyExecution`, witness, relation, or
simulator-specific Artifact family and are never required for semantic
correctness.

Actor-bearing trace and activity records use the Dataflow-owned `ActorRef`.
Execution-local invocation occurrences and per-actor firing ordinals qualify a
dynamic event but never create persistent Dataflow entities. The remaining
progress, activity, trace-manifest, and canonical chunk records remain owned
by the `SimulationExecution persistent wire` design frontier; this rule fixes
only their upstream identity owner.

Trace capture is a nonsemantic invocation binding. Enabling it may change
execution cost and retained raw material, but must not change scheduling,
outputs, terminal form, cycle count, metrics, or findings. The exact capture
request belongs only to `InvocationManifest`. The execution owner's attempt
record references that request and owns attempt provenance and retained-
material references. The `SimulationExecution` trace manifest alone owns the
actual order, coverage, and completeness of captured trace data.

Activity summaries identify their observation window, coverage, scope,
attachment, and timebase. Waveforms and simulator-native activity files are
raw blobs referenced by the execution. They do not form an `ActivityProfile`
Artifact or duplicate final values and memory state.

## Ownership And Coupling

`EvaluationRequest.workload_ref` and `runtime_input_ref` contain exact Artifact
references. A `SimulationExecution` refers only to that exact Request and
recovers both inputs and the workload observable contract through it rather
than copying their references. `EvaluationEvidence` refers to the Request and
binds the execution through the descriptor-owned output slot without copying
its observations or terminal witness. A terminal finding occurrence uses the
output-binding-relative `TerminalWitnessRef`. The execution never refers to
Evidence, so the content-identity graph is acyclic.

Architecture-only RTL or EDA checks do not execute a workload and therefore
must not create an empty `SimulationExecution`.

## Anchor Verification

Stable workload/input anchors cover `RootedGraphLaunchRef` ownership, dense
coordinate rank, total value-source classification, exact runtime-table
complements, value and stream cardinality, semantic lane states, mandatory
completion derivation, observable target validity, canonical object ordering,
two imported roots aliasing one object, and rejection of physical layout or
arbitrary object identity. One vecadd case covers fixed `N`, runtime `A/B/C`,
and a visible diff of `C`.

Execution anchors separately cover the single untagged root, Request-only
coupling, typed output slot and ordinal resolution, mandatory terminal-finding
totality, the closed terminal algebra and outcome cardinality, witness
ownership and reference validation, runtime-dependent `Unsupported` versus
`Halted` and `ExecutionFailed`, spatial functional-array alignment, terminal-
specific value and stream completeness, unique memory-diff runs, baseline
eligibility, alias-overlap agreement, and rejection of normalized results
inside `SimulationExecution`. Progress observations, activity, and trace add
their own anchors only as their persistent records close. Tests do not pin
report layouts, simulator class hierarchies, broad workload matrices, every
finding kind, every witness payload, or every partial-output combination.
